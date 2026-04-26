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
    fit_sparse_imu_batch,
    fwd_transform_general,
    load_ground_truth_dataset,
    load_imu_measurements,
    wrench_from_modal,
    wrench_metrics,
)
from evaluate_kirchhoff_shape_estimation import GAMMA_DEFAULT, MEAS_STD_DEG_DEF, so3_log
from virtual_work import body_jacobian_at_s


L_DEFAULT = 0.1
NUMERICAL_RANK_REL_TOL = 1e-8
EFFECTIVE_RANK_REL_TOL = 1e-3

METRIC_KEYS = [
    "batch_cost",
    "batch_residual_norm",
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


def _fit_batch(
    R_frame: List[np.ndarray],
    imu_pos: np.ndarray,
    order: OrderTuple,
    L_phys: float,
    gamma: int,
    batch_max_iter: int,
    batch_multistart: int,
    seed: int,
    m0: np.ndarray | None,
    force_m0_start: bool,
) -> Tuple[np.ndarray, Dict]:
    effective_multistart = batch_multistart
    if force_m0_start and m0 is not None and effective_multistart < 2:
        effective_multistart = 2
    fit = fit_sparse_imu_batch(
        R_meas=np.asarray(R_frame),
        imu_positions=imu_pos,
        order_x=order[0],
        order_y=order[1],
        order_z=order[2],
        L=L_phys,
        gamma=gamma,
        m0=m0,
        max_iter=batch_max_iter,
        multistart=effective_multistart,
        seed=seed,
    )
    m_hat = fit[0]
    info = {
        "batch_cost": float(fit[1]),
        "batch_success": bool(fit[2]),
        "batch_nfev": int(fit[3]),
        "batch_residual_norm": float(fit[4]),
        "batch_solver": str(fit[5]),
        "batch_effective_multistart": effective_multistart,
    }
    if len(fit) >= 9:
        info["batch_status"] = int(fit[6])
        info["batch_message_short"] = str(fit[7])
        info["batch_hit_limit"] = bool(fit[8])
    else:
        info["batch_status"] = float("nan")
        info["batch_message_short"] = ""
        info["batch_hit_limit"] = bool((not bool(fit[2])) and int(fit[3]) >= batch_max_iter)
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
        key = (row["modal_order"], row["init_strategy"], row["solver_mode"], row["layout_name"])
        groups[key].append(row)

    summary: List[Dict] = []
    for (modal_order, init_strategy, solver_mode, layout_name), items in sorted(groups.items()):
        entry: Dict = {
            "modal_order": modal_order,
            "init_strategy": init_strategy,
            "solver_mode": solver_mode,
            "layout_name": layout_name,
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
            "| order | init | mode | layout | n | success | residual | modal_rmse | nrmse_F | nrmse_M | cond |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary[:60]:
        lines.append(
            "| {modal_order} | {init_strategy} | {solver_mode} | {layout_name} | {n_runs} | "
            "{success:.3g} | {resid:.4g} | {modal:.4g} | {nf:.4g} | {nm:.4g} | {cond:.4g} |".format(
                success=row.get("batch_success_rate", float("nan")),
                resid=row.get("batch_residual_norm_mean", float("nan")),
                modal=row.get("modal_rmse_mean", float("nan")),
                nf=row.get("nrmse_force_mean", float("nan")),
                nm=row.get("nrmse_moment_mean", float("nan")),
                cond=row.get("cond_nonzero_mean", float("nan")),
                **row,
            )
        )
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
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    n_cases = R_meas.shape[0]
    n_noise = R_meas.shape[1]
    for ci in range(n_cases):
        for ni in range(n_noise):
            prev_order: OrderTuple | None = None
            prev_continuation_m: np.ndarray | None = None
            R_frame = [R_meas[ci, ni, si] for si in range(len(imu_pos))]
            for oi, order in enumerate(orders):
                m_oracle = oracle_cache[(order, ci)]
                base_seed = seed + 10000 * ci + 100 * ni + oi

                m_zero, info_zero = _fit_batch(
                    R_frame,
                    imu_pos,
                    order,
                    L_phys,
                    gamma,
                    batch_max_iter,
                    batch_multistart,
                    base_seed,
                    m0=None,
                    force_m0_start=False,
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

                if prev_order is None or prev_continuation_m is None:
                    m0 = None
                    init_note = "zero_first_order"
                else:
                    m0 = embed_modal_state(prev_continuation_m, prev_order, order)
                    init_note = f"embedded_from_{_order_label(prev_order)}"

                m_cont, info_cont = _fit_batch(
                    R_frame,
                    imu_pos,
                    order,
                    L_phys,
                    gamma,
                    batch_max_iter,
                    batch_multistart,
                    base_seed + 5000,
                    m0=m0,
                    force_m0_start=m0 is not None,
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
                prev_order = order
                prev_continuation_m = m_cont
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
            f"resid={row['batch_residual_norm_mean']:.4g} "
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
    try:
        _parse_orders(args.orders)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    run_diagnostic(args)


if __name__ == "__main__":
    main()
