#!/usr/bin/env python
"""
Modal-order continuation diagnostic for sparse-IMU shape and wrench estimation.

The batch mode compares independent zero initialization against lower-to-higher
modal continuation. Existing virtual-work, sparse-batch fitting, and wrench
helpers are reused; this script only adds diagnostic orchestration/reporting.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from compare_oracle_vs_ekf_wrench_estimation import (
    estimate_modal_oracle,
    fwd_transform_general,
    load_ground_truth_dataset,
    load_imu_measurements,
    wrench_from_modal,
    wrench_metrics,
)
from evaluate_kirchhoff_shape_estimation import GAMMA_DEFAULT, MEAS_STD_DEG_DEF, so3_log
from virtual_work import body_jacobian_at_s

try:
    from scipy.optimize import least_squares as _least_squares
except Exception:  # pragma: no cover - fallback path only without scipy.optimize
    _least_squares = None


L_DEFAULT = 0.1
NUMERICAL_RANK_REL_TOL = 1e-8
EFFECTIVE_RANK_REL_TOL = 1e-3

METRIC_KEYS = [
    "batch_cost",
    "batch_residual_norm",
    "imu_residual_norm",
    "regularization_residual_norm",
    "total_residual_norm",
    "new_coeff_norm",
    "deviation_from_embedded_norm",
    "modal_err_norm",
    "modal_rmse",
    "tip_pos_err_m",
    "tip_rot_err_deg",
    "force_err_N",
    "moment_err_Nm",
    "nrmse_force",
    "nrmse_moment",
    "force_dir_err_deg",
    "moment_dir_err_deg",
    "cond_nonzero",
    "smallest_nonzero_sigma",
    "largest_sigma",
    "numerical_rank",
    "effective_rank",
]


OrderTuple = Tuple[int, int, int]


def _order_label(order: OrderTuple) -> str:
    return f"({order[0]},{order[1]},{order[2]})"


def _n_params(order: OrderTuple) -> int:
    return (order[0] + 1) + (order[1] + 1) + (order[2] + 1)


def _axis_slices(order: OrderTuple) -> Tuple[slice, slice, slice]:
    nx, ny, nz = order
    sx = slice(0, nx + 1)
    sy = slice(nx + 1, nx + 1 + ny + 1)
    sz = slice(nx + 1 + ny + 1, nx + 1 + ny + 1 + nz + 1)
    return sx, sy, sz


def embed_modal_state(m_low: np.ndarray, order_low: OrderTuple, order_high: OrderTuple) -> np.ndarray:
    """
    Embed modal coefficients from lower order into higher order by copying
    matching x/y/z polynomial coefficients and zero-filling newly introduced
    modes.

    State layout:
        [mx coefficients, my coefficients, mz coefficients]
    """
    m_low = np.asarray(m_low, dtype=float)
    if len(m_low) != _n_params(order_low):
        raise ValueError(f"m_low length {len(m_low)} does not match order {order_low}")
    m_high = np.zeros(_n_params(order_high), dtype=float)
    low_slices = _axis_slices(order_low)
    high_slices = _axis_slices(order_high)
    for src, dst in zip(low_slices, high_slices):
        src_vals = m_low[src]
        n_copy = min(len(src_vals), dst.stop - dst.start)
        m_high[dst.start : dst.start + n_copy] = src_vals[:n_copy]
    return m_high


def _self_test_embed() -> None:
    m = np.arange(_n_params((1, 1, 0)), dtype=float)
    up = embed_modal_state(m, (1, 1, 0), (2, 2, 1))
    assert up.shape == (_n_params((2, 2, 1)),)
    assert np.allclose(up[_axis_slices((2, 2, 1))[0]][:2], m[_axis_slices((1, 1, 0))[0]])
    assert np.allclose(up[_axis_slices((2, 2, 1))[1]][:2], m[_axis_slices((1, 1, 0))[1]])
    assert np.allclose(up[_axis_slices((2, 2, 1))[2]][:1], m[_axis_slices((1, 1, 0))[2]])
    down = embed_modal_state(up, (2, 2, 1), (1, 1, 0))
    assert down.shape == m.shape


def _parse_orders(text: str) -> List[OrderTuple]:
    orders: List[OrderTuple] = []
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid order {raw!r}") from exc
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or not all(isinstance(v, int) for v in value)
            or any(v < 0 for v in value)
        ):
            raise argparse.ArgumentTypeError(f"invalid order {raw!r}")
        orders.append(value)
    if not orders:
        raise argparse.ArgumentTypeError("at least one order is required")
    return orders


def _parse_float_list(text: str) -> List[float]:
    vals = [float(raw.strip()) for raw in text.split(",") if raw.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("list must not be empty")
    return vals


def _resolve_path(script_dir: Path, raw_path: str, must_exist: bool = True) -> Path:
    expanded = Path(os.path.expandvars(raw_path))
    out = expanded if expanded.is_absolute() else (script_dir / expanded).resolve()
    if must_exist and not out.exists():
        raise FileNotFoundError(out)
    return out


def _safe_nanmean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _safe_nanstd(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanstd(arr))


def _safe_nanmedian(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmedian(arr))


def _shape_metrics(
    m_est: np.ndarray,
    m_oracle: np.ndarray | None,
    positions_case: np.ndarray,
    orientations_case: np.ndarray,
    order: OrderTuple,
    gamma: int,
    L_phys: float,
) -> Dict[str, float]:
    if m_oracle is None:
        modal_err_norm = float("nan")
        modal_rmse = float("nan")
    else:
        err = m_est - m_oracle
        modal_err_norm = float(np.linalg.norm(err))
        modal_rmse = float(np.sqrt(np.mean(err**2)))

    T_tip = fwd_transform_general(m_est, 1.0, gamma, L_phys, *order)
    p_gt = positions_case[-1]
    R_gt = orientations_case[-1].reshape(3, 3)
    return {
        "modal_err_norm": modal_err_norm,
        "modal_rmse": modal_rmse,
        "tip_pos_err_m": float(np.linalg.norm(T_tip[:3, 3] - p_gt)),
        "tip_rot_err_deg": float(np.rad2deg(np.linalg.norm(so3_log(T_tip[:3, :3] @ R_gt.T)))),
    }


def _observability_metrics(m_est: np.ndarray, order: OrderTuple, gamma: int, L_phys: float) -> Dict[str, float]:
    J_vbm, _ = body_jacobian_at_s(m_est, 1.0, gamma, L_phys, *order)
    A = J_vbm.T
    sigma = np.linalg.svd(A, compute_uv=False)
    if sigma.size == 0 or sigma[0] <= 0:
        return {
            "cond_nonzero": float("nan"),
            "smallest_nonzero_sigma": float("nan"),
            "largest_sigma": float("nan"),
            "numerical_rank": 0,
            "effective_rank": 0,
        }
    ratios = sigma / sigma[0]
    numerical_rank = int(np.sum(ratios >= NUMERICAL_RANK_REL_TOL))
    effective_rank = int(np.sum(ratios >= EFFECTIVE_RANK_REL_TOL))
    nonzero = sigma[ratios >= NUMERICAL_RANK_REL_TOL]
    cond = float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")
    return {
        "cond_nonzero": cond,
        "smallest_nonzero_sigma": float(nonzero[-1]) if nonzero.size else float("nan"),
        "largest_sigma": float(sigma[0]),
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
    }


def _imu_residual(
    m: np.ndarray,
    R_frame: List[np.ndarray],
    imu_pos: np.ndarray,
    order: OrderTuple,
    L_phys: float,
    gamma: int,
) -> np.ndarray:
    rows = []
    for R_meas, s_i in zip(R_frame, imu_pos):
        R_pred = fwd_transform_general(m, float(s_i), gamma, L_phys, *order)[:3, :3]
        rows.append(so3_log(R_meas @ R_pred.T))
    return np.hstack(rows)


def _new_coeff_mask(order_prev: OrderTuple | None, order: OrderTuple, regularize_first_order: bool) -> np.ndarray:
    mask = np.zeros(_n_params(order), dtype=bool)
    high_slices = _axis_slices(order)
    if order_prev is None:
        if regularize_first_order:
            mask[:] = True
        return mask

    prev_degrees = order_prev
    for axis, dst in enumerate(high_slices):
        prev_n = prev_degrees[axis] + 1
        axis_len = dst.stop - dst.start
        if prev_n < axis_len:
            mask[dst.start + prev_n : dst.stop] = True
    return mask


def _regularization_settings(
    mode: str,
    prior_weights: List[float],
    new_coeff_weights: List[float],
) -> List[Tuple[str, float, float]]:
    if mode == "none":
        return [("none", 0.0, 0.0)]
    if mode == "embedded_prior":
        return [("embedded_prior", w, 0.0) for w in prior_weights]
    if mode == "new_coeff_penalty":
        return [("new_coeff_penalty", 0.0, w) for w in new_coeff_weights]
    if mode == "both":
        return [("both", wp, wn) for wp in prior_weights for wn in new_coeff_weights]
    raise ValueError(f"Unknown regularization mode: {mode}")


def _fit_batch_regularized(
    R_frame: List[np.ndarray],
    imu_pos: np.ndarray,
    order: OrderTuple,
    L_phys: float,
    gamma: int,
    batch_max_iter: int,
    batch_multistart: int,
    seed: int,
    m0: np.ndarray | None,
    regularization_mode: str,
    prior_weight: float,
    new_coeff_weight: float,
    embedded_target: np.ndarray | None,
    new_coeff_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)
    n = _n_params(order)
    target = np.zeros(n) if embedded_target is None else np.asarray(embedded_target, dtype=float)
    if target.shape != (n,):
        raise ValueError("embedded target dimension does not match current order")
    new_coeff_mask = np.asarray(new_coeff_mask, dtype=bool)

    use_prior = regularization_mode in ("embedded_prior", "both") and prior_weight > 0.0
    use_new = regularization_mode in ("new_coeff_penalty", "both") and new_coeff_weight > 0.0

    def residual(m: np.ndarray) -> np.ndarray:
        chunks = [_imu_residual(m, R_frame, imu_pos, order, L_phys, gamma)]
        if use_prior:
            chunks.append(np.sqrt(prior_weight) * (m - target))
        if use_new and np.any(new_coeff_mask):
            chunks.append(np.sqrt(new_coeff_weight) * m[new_coeff_mask])
        return np.hstack(chunks)

    starts: List[np.ndarray] = []
    starts.append(np.asarray(m0, dtype=float) if m0 is not None else np.zeros(n))
    if batch_multistart > 1:
        starts.append(np.zeros(n))
    while len(starts) < batch_multistart:
        starts.append(1e-2 * rng.standard_normal(n))

    best = None
    solver_name = "scipy.optimize.least_squares" if _least_squares is not None else "fallback_lm"
    for start in starts[: max(1, batch_multistart)]:
        if _least_squares is not None:
            res = _least_squares(
                residual,
                start,
                method="trf",
                max_nfev=batch_max_iter,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            candidate = {
                "x": res.x,
                "cost": float(res.cost),
                "success": bool(res.success),
                "nfev": int(res.nfev),
                "status": int(res.status),
                "message": " ".join(str(res.message).split())[:120],
            }
        else:
            x = np.asarray(start, dtype=float).copy()
            lam = 1e-3
            nfev = 0
            success = False
            for _ in range(batch_max_iter):
                r = residual(x)
                nfev += 1
                J = np.zeros((len(r), n))
                eps = 1e-6
                for j in range(n):
                    dx = np.zeros(n)
                    dx[j] = eps
                    J[:, j] = (residual(x + dx) - residual(x - dx)) / (2.0 * eps)
                    nfev += 2
                lhs = J.T @ J + lam * np.eye(n)
                rhs = -J.T @ r
                step = np.linalg.pinv(lhs) @ rhs
                r_new = residual(x + step)
                nfev += 1
                if np.linalg.norm(r_new) <= np.linalg.norm(r):
                    x = x + step
                    lam = max(lam * 0.5, 1e-9)
                    if np.linalg.norm(step) < 1e-8:
                        success = True
                        break
                else:
                    lam = min(lam * 5.0, 1e6)
            candidate = {
                "x": x,
                "cost": 0.5 * float(residual(x) @ residual(x)),
                "success": success,
                "nfev": nfev,
                "status": 1 if success else 0,
                "message": "fallback converged" if success else "fallback reached iteration cap",
            }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    assert best is not None
    m_hat = best["x"]
    imu_r = _imu_residual(m_hat, R_frame, imu_pos, order, L_phys, gamma)
    reg_chunks = []
    if use_prior:
        reg_chunks.append(np.sqrt(prior_weight) * (m_hat - target))
    if use_new and np.any(new_coeff_mask):
        reg_chunks.append(np.sqrt(new_coeff_weight) * m_hat[new_coeff_mask])
    reg_r = np.hstack(reg_chunks) if reg_chunks else np.zeros(0)
    total_r = np.hstack([imu_r, reg_r])
    hit_limit = int(best["status"]) == 0 or ((not bool(best["success"])) and int(best["nfev"]) >= batch_max_iter)
    info = {
        "batch_cost": 0.5 * float(total_r @ total_r),
        "batch_success": bool(best["success"]),
        "batch_nfev": int(best["nfev"]),
        "batch_residual_norm": float(np.linalg.norm(total_r)),
        "batch_solver": solver_name,
        "batch_effective_multistart": batch_multistart,
        "batch_status": int(best["status"]),
        "batch_message_short": str(best["message"]),
        "batch_hit_limit": bool(hit_limit),
        "regularization_mode": regularization_mode,
        "prior_weight": float(prior_weight),
        "new_coeff_weight": float(new_coeff_weight),
        "imu_residual_norm": float(np.linalg.norm(imu_r)),
        "regularization_residual_norm": float(np.linalg.norm(reg_r)),
        "total_residual_norm": float(np.linalg.norm(total_r)),
        "new_coeff_norm": float(np.linalg.norm(m_hat[new_coeff_mask])) if np.any(new_coeff_mask) else 0.0,
        "deviation_from_embedded_norm": float(np.linalg.norm(m_hat - target))
        if embedded_target is not None
        else float("nan"),
    }
    return m_hat, info


def _row_for_fit(
    m_hat: np.ndarray,
    fit_info: Dict,
    order: OrderTuple,
    layout: str,
    case_idx: int,
    noise_idx: int,
    case_id: int,
    init_strategy: str,
    solver_mode: str,
    m_oracle: np.ndarray | None,
    positions_case: np.ndarray,
    orientations_case: np.ndarray,
    tau: np.ndarray,
    f_gt: np.ndarray,
    l_gt: np.ndarray,
    gamma: int,
    L_phys: float,
) -> Dict:
    f_est, l_est = wrench_from_modal(m_hat, tau, order[0], order[1], order[2], gamma)
    row = {
        "modal_order": _order_label(order),
        "order_x": order[0],
        "order_y": order[1],
        "order_z": order[2],
        "n_m": _n_params(order),
        "layout_name": layout,
        "case_idx": case_idx,
        "case_id": case_id,
        "noise_idx": noise_idx,
        "init_strategy": init_strategy,
        "solver_mode": solver_mode,
    }
    row.update(fit_info)
    row.update(_shape_metrics(m_hat, m_oracle, positions_case, orientations_case, order, gamma, L_phys))
    row.update(wrench_metrics(f_est, l_est, f_gt, l_gt))
    row.update(_observability_metrics(m_hat, order, gamma, L_phys))
    return row


def _save_csv(rows: List[Dict], path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  -> {path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _save_json(config: Dict, rows: List[Dict], summary: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": rows, "summary": summary}, f, indent=2, default=_json_default)
    print(f"  JSON -> {path}")


def _aggregate(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["modal_order"],
            row["init_strategy"],
            row["solver_mode"],
            row["layout_name"],
            row["regularization_mode"],
            row["prior_weight"],
            row["new_coeff_weight"],
        )
        groups[key].append(row)

    summary: List[Dict] = []
    for (
        modal_order,
        init_strategy,
        solver_mode,
        layout_name,
        regularization_mode,
        prior_weight,
        new_coeff_weight,
    ), items in sorted(groups.items()):
        entry: Dict = {
            "modal_order": modal_order,
            "init_strategy": init_strategy,
            "solver_mode": solver_mode,
            "layout_name": layout_name,
            "regularization_mode": regularization_mode,
            "prior_weight": prior_weight,
            "new_coeff_weight": new_coeff_weight,
            "n_runs": len(items),
            "order_x": items[0]["order_x"],
            "order_y": items[0]["order_y"],
            "order_z": items[0]["order_z"],
            "n_m": items[0]["n_m"],
        }
        success_vals = [float(row["batch_success"]) for row in items if isinstance(row.get("batch_success"), bool)]
        entry["batch_success_rate"] = float("nan") if not success_vals else float(np.mean(success_vals))
        for key in METRIC_KEYS:
            vals = [row.get(key, float("nan")) for row in items]
            entry[f"{key}_mean"] = _safe_nanmean(vals)
            entry[f"{key}_std"] = _safe_nanstd(vals)
            entry[f"{key}_med"] = _safe_nanmedian(vals)
        summary.append(entry)
    return summary


def _write_markdown(summary: List[Dict], config: Dict, path: Path) -> None:
    lines = [
        "# Modal Order Continuation Diagnostic",
        "",
        "Interpretation:",
        "- If continuation improves batch residual, modal, or wrench metrics, initialization matters for high-order fitting.",
        "- If continuation does not improve sparse-batch fitting, sparse sensor information or model conditioning may dominate.",
        "- If high-order direct 6D remains poor despite good batch fit, wrench inverse conditioning dominates.",
        "- These are diagnostics only; final manuscript figures should be selected later.",
        "",
        "Configuration:",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "| order | init | reg | prior | new | n | success | imu_resid | total_resid | new_norm | modal_rmse | nrmse_F | nrmse_M | cond |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary[:60]:
        lines.append(
            "| {modal_order} | {init_strategy} | {regularization_mode} | {prior_weight:g} | {new_coeff_weight:g} | "
            "{n_runs} | {success:.3g} | {imu:.4g} | {total:.4g} | {new_norm:.4g} | "
            "{modal:.4g} | {nf:.4g} | {nm:.4g} | {cond:.4g} |".format(
                success=row.get("batch_success_rate", float("nan")),
                imu=row.get("imu_residual_norm_mean", float("nan")),
                total=row.get("total_residual_norm_mean", float("nan")),
                new_norm=row.get("new_coeff_norm_mean", float("nan")),
                modal=row.get("modal_rmse_mean", float("nan")),
                nf=row.get("nrmse_force_mean", float("nan")),
                nm=row.get("nrmse_moment_mean", float("nan")),
                cond=row.get("cond_nonzero_mean", float("nan")),
                **row,
            )
        )
    lines.extend(["", "## Best Settings", ""])
    metrics = [
        ("best_by_imu_residual", "imu_residual_norm_mean"),
        ("best_by_modal_rmse", "modal_rmse_mean"),
        ("best_by_nrmse_force", "nrmse_force_mean"),
        ("best_by_nrmse_moment", "nrmse_moment_mean"),
    ]
    by_order: Dict[str, List[Dict]] = defaultdict(list)
    for row in summary:
        by_order[row["modal_order"]].append(row)
    for modal_order, rows in sorted(by_order.items()):
        lines.append(f"### {modal_order}")
        lines.append("")
        lines.append("| criterion | init | reg | prior | new | value | imu_resid | nrmse_F | nrmse_M |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for label, metric in metrics:
            candidates = [row for row in rows if np.isfinite(float(row.get(metric, float("nan"))))]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: float(row.get(metric, float("inf"))))
            lines.append(
                "| {label} | {init_strategy} | {regularization_mode} | {prior_weight:g} | {new_coeff_weight:g} | "
                "{value:.4g} | {imu:.4g} | {nf:.4g} | {nm:.4g} |".format(
                    label=label,
                    value=float(best.get(metric, float("nan"))),
                    imu=float(best.get("imu_residual_norm_mean", float("nan"))),
                    nf=float(best.get("nrmse_force_mean", float("nan"))),
                    nm=float(best.get("nrmse_moment_mean", float("nan"))),
                    **best,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  MD   -> {path}")


def run_batch_mode(
    orders: List[OrderTuple],
    layout: str,
    R_meas: np.ndarray,
    imu_pos: np.ndarray,
    positions_gt: np.ndarray,
    orientations_gt: np.ndarray,
    case_ids: np.ndarray,
    tau_gt: np.ndarray,
    f_gt: np.ndarray,
    l_gt: np.ndarray,
    oracle_cache: Dict[Tuple[OrderTuple, int], np.ndarray | None],
    gamma: int,
    L_phys: float,
    batch_max_iter: int,
    batch_multistart: int,
    regularization_mode: str,
    prior_weights: List[float],
    new_coeff_weights: List[float],
    regularize_first_order: bool,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    n_cases = R_meas.shape[0]
    n_noise = R_meas.shape[1]
    all_settings = _regularization_settings(regularization_mode, prior_weights, new_coeff_weights)
    no_reg_setting = ("none", 0.0, 0.0)
    for ci in range(n_cases):
        for ni in range(n_noise):
            prev_order: OrderTuple | None = None
            prev_solutions: Dict[Tuple[str, float, float], np.ndarray] = {}
            prev_default_solution: np.ndarray | None = None
            R_frame = [R_meas[ci, ni, si] for si in range(len(imu_pos))]
            for oi, order in enumerate(orders):
                m_oracle = oracle_cache[(order, ci)]
                base_seed = seed + 10000 * ci + 100 * ni + oi

                zero_target = np.zeros(_n_params(order))
                zero_mask = np.zeros(_n_params(order), dtype=bool)
                m_zero, info_zero = _fit_batch_regularized(
                    R_frame,
                    imu_pos,
                    order,
                    L_phys,
                    gamma,
                    batch_max_iter,
                    batch_multistart,
                    base_seed,
                    m0=None,
                    regularization_mode="none",
                    prior_weight=0.0,
                    new_coeff_weight=0.0,
                    embedded_target=zero_target,
                    new_coeff_mask=zero_mask,
                )
                rows.append(
                    _row_for_fit(
                        m_zero,
                        info_zero,
                        order,
                        layout,
                        ci,
                        ni,
                        int(case_ids[ci]),
                        "independent_zero",
                        "batch",
                        m_oracle,
                        positions_gt[ci],
                        orientations_gt[ci],
                        tau_gt[ci],
                        f_gt[ci],
                        l_gt[ci],
                        gamma,
                        L_phys,
                    )
                )

                settings = all_settings
                if prev_order is None and not regularize_first_order:
                    settings = [no_reg_setting]

                next_solutions: Dict[Tuple[str, float, float], np.ndarray] = {}
                next_default_solution: np.ndarray | None = None
                for si, (reg_mode, prior_weight, new_weight) in enumerate(settings):
                    if prev_order is None:
                        embedded_target = np.zeros(_n_params(order))
                        m0 = embedded_target.copy()
                        init_note = "zero_first_order"
                    else:
                        source = prev_solutions.get((reg_mode, prior_weight, new_weight), prev_default_solution)
                        if source is None:
                            source = prev_default_solution
                        embedded_target = embed_modal_state(source, prev_order, order)
                        m0 = embedded_target.copy()
                        init_note = f"embedded_from_{_order_label(prev_order)}"

                    new_mask = _new_coeff_mask(prev_order, order, regularize_first_order)
                    effective_reg_mode = reg_mode
                    effective_prior = prior_weight
                    effective_new = new_weight
                    if prev_order is None and not regularize_first_order:
                        effective_reg_mode = "none"
                        effective_prior = 0.0
                        effective_new = 0.0

                    m_cont, info_cont = _fit_batch_regularized(
                        R_frame,
                        imu_pos,
                        order,
                        L_phys,
                        gamma,
                        batch_max_iter,
                        batch_multistart,
                        base_seed + 5000 + si,
                        m0=m0,
                        regularization_mode=effective_reg_mode,
                        prior_weight=effective_prior,
                        new_coeff_weight=effective_new,
                        embedded_target=embedded_target,
                        new_coeff_mask=new_mask,
                    )
                    info_cont["continuation_init_note"] = init_note
                    rows.append(
                        _row_for_fit(
                            m_cont,
                            info_cont,
                            order,
                            layout,
                            ci,
                            ni,
                            int(case_ids[ci]),
                            "continuation",
                            "batch",
                            m_oracle,
                            positions_gt[ci],
                            orientations_gt[ci],
                            tau_gt[ci],
                            f_gt[ci],
                            l_gt[ci],
                            gamma,
                            L_phys,
                        )
                    )
                    setting_key = (reg_mode, prior_weight, new_weight)
                    next_solutions[setting_key] = m_cont
                    if setting_key == no_reg_setting or next_default_solution is None:
                        next_default_solution = m_cont
                prev_order = order
                prev_solutions = next_solutions
                prev_default_solution = next_default_solution
    return rows


def run_diagnostic(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    _self_test_embed()
    script_dir = Path(__file__).resolve().parent
    gt_path = _resolve_path(script_dir, args.gt)
    imu_path = _resolve_path(script_dir, args.imu3 if args.layout == "3-IMU" else args.imu2, must_exist=True)
    save_dir = _resolve_path(script_dir, args.save_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets ...")
    positions_gt, orientations_gt, case_ids_gt, gt_meta = load_ground_truth_dataset(gt_path)
    gt_data = np.load(gt_path, allow_pickle=True)
    tau_gt = gt_data["tau"]
    f_gt = gt_data["f_ext"]
    l_gt = gt_data["l_ext"]
    L_phys = float(gt_meta.get("length_m", L_DEFAULT))

    n_cases = min(args.max_cases, len(positions_gt))
    print(f"Applying --max-cases {args.max_cases}: using first {n_cases} cases.")
    positions_gt = positions_gt[:n_cases]
    orientations_gt = orientations_gt[:n_cases]
    case_ids_gt = case_ids_gt[:n_cases]
    tau_gt = tau_gt[:n_cases]
    f_gt = f_gt[:n_cases]
    l_gt = l_gt[:n_cases]

    _, R_meas, _, imu_idx, imu_meta = load_imu_measurements(imu_path)
    R_meas = R_meas[:n_cases]
    n_noise = min(args.n_noise, R_meas.shape[1])
    print(f"Applying --n-noise {args.n_noise}: using first {n_noise} noise realizations.")
    R_meas = R_meas[:, :n_noise]
    imu_pos = np.array(
        imu_meta.get("imu_actual_s", (imu_idx / (positions_gt.shape[1] - 1)).tolist()),
        dtype=float,
    )
    print(f"  {args.layout}: R_meas {R_meas.shape}, imu_pos={imu_pos}")

    orders = _parse_orders(args.orders)
    print("Orders:", ", ".join(_order_label(order) for order in orders))

    oracle_cache: Dict[Tuple[OrderTuple, int], np.ndarray | None] = {}
    for order in orders:
        for ci in range(n_cases):
            try:
                oracle_cache[(order, ci)] = estimate_modal_oracle(
                    positions_gt[ci],
                    orientations_gt[ci],
                    order[0],
                    order[1],
                    order[2],
                    L_phys,
                )
            except Exception:
                oracle_cache[(order, ci)] = None

    rows: List[Dict] = []
    if args.mode in ("batch", "both"):
        print("Running batch continuation diagnostic ...")
        rows.extend(
            run_batch_mode(
                orders,
                args.layout,
                R_meas,
                imu_pos,
                positions_gt,
                orientations_gt,
                case_ids_gt,
                tau_gt,
                f_gt,
                l_gt,
                oracle_cache,
                args.gamma,
                L_phys,
                args.batch_max_iter,
                args.batch_multistart,
                args.regularization_mode,
                args.prior_weight_list,
                args.new_coeff_weight_list,
                args.regularize_first_order,
                args.seed,
            )
        )

    if args.mode in ("ekf", "both"):
        print("EKF continuation mode is not implemented in this diagnostic; batch mode is available.")

    summary = _aggregate(rows)
    config = {
        "gt": str(gt_path),
        "imu": str(imu_path),
        "orders": [_order_label(order) for order in orders],
        "max_cases": args.max_cases,
        "n_noise": args.n_noise,
        "layout": args.layout,
        "mode": args.mode,
        "batch_max_iter": args.batch_max_iter,
        "batch_multistart": args.batch_multistart,
        "regularization_mode": args.regularization_mode,
        "prior_weight_list": args.prior_weight_list,
        "new_coeff_weight_list": args.new_coeff_weight_list,
        "regularize_first_order": args.regularize_first_order,
        "gamma": args.gamma,
        "L_phys_m": L_phys,
        "seed": args.seed,
        "ekf_mode_implemented": False,
    }

    _save_csv(rows, save_dir / "modal_order_continuation_results.csv")
    _save_csv(summary, save_dir / "modal_order_continuation_summary.csv")
    _save_json(config, rows, summary, save_dir / "modal_order_continuation_results.json")
    _write_markdown(summary, config, save_dir / "modal_order_continuation_summary.md")

    print("\nSummary preview:")
    for row in summary[:12]:
        print(
            f"  {row['modal_order']} {row['init_strategy']} "
            f"reg={row['regularization_mode']} prior={row['prior_weight']:g} new={row['new_coeff_weight']:g} "
            f"imu={row['imu_residual_norm_mean']:.4g} total={row['total_residual_norm_mean']:.4g} "
            f"modal_rmse={row['modal_rmse_mean']:.4g} "
            f"nrmse_F={row['nrmse_force_mean']:.4g} "
            f"nrmse_M={row['nrmse_moment_mean']:.4g} "
            f"cond={row['cond_nonzero_mean']:.4g}"
        )
    return rows, summary, config


def build_parser() -> argparse.ArgumentParser:
    default_save = str(Path(tempfile.gettempdir()) / "shape-force-est-IMU-smoke" / "continuation")
    parser = argparse.ArgumentParser(
        description="Diagnose lower-to-higher modal-order continuation for sparse IMU fitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--imu2", default="gt_data/kirchhoff_imu_2imu.npz")
    parser.add_argument("--imu3", default="gt_data/kirchhoff_imu_3imu.npz")
    parser.add_argument("--orders", default="(1,1,0);(1,2,0);(2,2,0);(2,2,1);(3,3,2)")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--n-noise", type=int, default=1)
    parser.add_argument("--layout", choices=["2-IMU", "3-IMU"], default="3-IMU")
    parser.add_argument("--mode", choices=["batch", "ekf", "both"], default="batch")
    parser.add_argument("--batch-max-iter", type=int, default=50)
    parser.add_argument("--batch-multistart", type=int, default=1)
    parser.add_argument(
        "--regularization-mode",
        choices=["none", "embedded_prior", "new_coeff_penalty", "both"],
        default="none",
    )
    parser.add_argument("--prior-weight-list", type=_parse_float_list, default=_parse_float_list("0.0,1e-3,1e-2,1e-1,1.0"))
    parser.add_argument("--new-coeff-weight-list", type=_parse_float_list, default=_parse_float_list("0.0,1e-3,1e-2,1e-1,1.0"))
    parser.add_argument("--regularize-first-order", action="store_true")
    parser.add_argument("--save-dir", default=default_save)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=GAMMA_DEFAULT)
    parser.add_argument("--meas-std-deg", type=float, default=MEAS_STD_DEG_DEF)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be > 0")
    if args.n_noise <= 0:
        parser.error("--n-noise must be > 0")
    if args.batch_max_iter <= 0:
        parser.error("--batch-max-iter must be > 0")
    if args.batch_multistart <= 0:
        parser.error("--batch-multistart must be > 0")
    if any(w < 0.0 for w in args.prior_weight_list):
        parser.error("--prior-weight-list values must be >= 0")
    if any(w < 0.0 for w in args.new_coeff_weight_list):
        parser.error("--new-coeff-weight-list values must be >= 0")
    try:
        _parse_orders(args.orders)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    run_diagnostic(args)


if __name__ == "__main__":
    main()
