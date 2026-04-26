#!/usr/bin/env python3
"""
Diagnose tip-wrench observability versus modal order on the saved Kirchhoff dataset.

This is a new diagnostic-only script. It does not modify the existing Step-4
wrench estimator or overwrite any current result files.

Study goal
----------
For each requested modal-order configuration, project every saved Kirchhoff-rod
 case onto that modal basis, evaluate the tip virtual-work wrench map

    J_Vb_m.T : R^6 -> R^{n_m}

and summarize:
  - modal dimension n_m
  - singular values
  - numerical rank
  - effective rank
  - condition number
  - weakest wrench-space singular direction

The "effective rank" uses a looser relative singular-value threshold to expose
practically weak directions even when the matrix is technically full rank.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_kirchhoff_shape_estimation import load_ground_truth_dataset
from wrench_study_utils import (
    COMPONENT_LABELS,
    DEFAULT_ORDER_CONFIGS,
    L_DEFAULT,
    build_transform_stack,
    estimate_modal_from_gt_frames,
    order_label,
    order_slug,
    parse_order_configs,
    svd_diagnostics,
    total_params,
    build_virtual_work_terms,
)


def resolve_cli_path(path_str: str, script_dir: Path, must_exist: bool) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        path = raw
    elif raw.exists():
        path = raw.resolve()
    else:
        path = (script_dir / raw).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  -> {path}")


def save_json(blob: dict, path: Path) -> None:
    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [_convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    path.write_text(json.dumps(_convert(blob), indent=2), encoding="utf-8")
    print(f"  JSON -> {path}")


def save_npz(arrays: dict, path: Path) -> None:
    np.savez_compressed(path, **arrays)
    print(f"  NPZ  -> {path}")


def build_observability_figure(
    per_case_rows: List[Dict],
    summary_rows: List[Dict],
    order_cfgs: List[Tuple[int, int, int]],
    save_stem: Path,
) -> None:
    order_labels = [order_label(cfg) for cfg in order_cfgs]

    smin_nonzero = [
        np.array(
            [r["smallest_nonzero_sigma"] for r in per_case_rows if r["order_label"] == label],
            dtype=float,
        )
        for label in order_labels
    ]
    log10_cond = [
        np.log10(
            np.array(
                [r["cond_nonzero"] for r in per_case_rows if r["order_label"] == label],
                dtype=float,
            )
        )
        for label in order_labels
    ]

    rank_values = sorted({int(r["numerical_rank"]) for r in per_case_rows})
    eff_rank_values = sorted({int(r["effective_rank"]) for r in per_case_rows})

    weak_heat = np.array(
        [
            [row[f"weak_abs_{comp}"] for comp in COMPONENT_LABELS]
            for row in summary_rows
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # (a) smallest nonzero singular value
    parts = ax1.boxplot(smin_nonzero, patch_artist=True, widths=0.65)
    for patch in parts["boxes"]:
        patch.set_facecolor("#8ecae6")
        patch.set_alpha(0.75)
    ax1.set_xticks(np.arange(1, len(order_labels) + 1))
    ax1.set_xticklabels(order_labels, rotation=20, ha="right")
    ax1.set_ylabel("Smallest nonzero singular value")
    ax1.set_title("(a) Smallest singular value by modal order")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # (b) log10 condition number
    parts = ax2.boxplot(log10_cond, patch_artist=True, widths=0.65)
    for patch in parts["boxes"]:
        patch.set_facecolor("#ffb703")
        patch.set_alpha(0.75)
    ax2.set_xticks(np.arange(1, len(order_labels) + 1))
    ax2.set_xticklabels(order_labels, rotation=20, ha="right")
    ax2.set_ylabel(r"$\log_{10} \kappa(J_{Vb_m}^T)$")
    ax2.set_title("(b) Condition number by modal order")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    # (c) rank fractions (numerical solid, effective hatched)
    x = np.arange(len(order_labels))
    width = 0.36
    bottom_num = np.zeros(len(order_labels))
    bottom_eff = np.zeros(len(order_labels))
    cmap = plt.cm.Set2(np.linspace(0.0, 0.95, max(len(rank_values), len(eff_rank_values), 3)))
    for idx, rank in enumerate(rank_values):
        frac = np.array(
            [row.get(f"numerical_rank_frac_{rank}", 0.0) for row in summary_rows],
            dtype=float,
        )
        ax3.bar(
            x - width / 2,
            frac,
            width=width,
            bottom=bottom_num,
            color=cmap[idx],
            alpha=0.85,
            label=f"num rank {rank}",
        )
        bottom_num += frac
    for idx, rank in enumerate(eff_rank_values):
        frac = np.array(
            [row.get(f"effective_rank_frac_{rank}", 0.0) for row in summary_rows],
            dtype=float,
        )
        ax3.bar(
            x + width / 2,
            frac,
            width=width,
            bottom=bottom_eff,
            color=cmap[idx],
            alpha=0.35,
            hatch="//",
            edgecolor="k",
            linewidth=0.4,
            label=f"eff rank {rank}" if idx == 0 else None,
        )
        bottom_eff += frac
    ax3.set_xticks(x)
    ax3.set_xticklabels(order_labels, rotation=20, ha="right")
    ax3.set_ylim(0.0, 1.02)
    ax3.set_ylabel("Fraction of cases")
    ax3.set_title("(c) Numerical rank vs effective rank")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.3)

    # (d) weak singular direction composition
    im = ax4.imshow(weak_heat, aspect="auto", cmap="magma")
    ax4.set_xticks(np.arange(len(COMPONENT_LABELS)))
    ax4.set_xticklabels(COMPONENT_LABELS)
    ax4.set_yticks(np.arange(len(order_labels)))
    ax4.set_yticklabels(order_labels)
    ax4.set_title("(d) Avg. |weakest right singular vector|")
    for i in range(weak_heat.shape[0]):
        for j in range(weak_heat.shape[1]):
            ax4.text(j, i, f"{weak_heat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle("Tip-wrench observability vs modal order", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ("png", "pdf"):
        out_path = save_stem.parent / f"{save_stem.name}.{ext}"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  FIG  -> {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose tip-wrench observability versus modal order on the Kirchhoff dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt", default="gt_data/kirchhoff_gt_dataset.npz")
    parser.add_argument("--save-dir", default="gt_data/results_wrench_diagnostics")
    parser.add_argument(
        "--orders",
        default=";".join(",".join(str(v) for v in cfg) for cfg in DEFAULT_ORDER_CONFIGS),
        help="semicolon-separated modal-order tuples: ox,oy,oz;ox,oy,oz;...",
    )
    parser.add_argument("--gamma", type=int, default=10)
    parser.add_argument("--numerical-rank-rel-tol", type=float, default=1e-6)
    parser.add_argument("--effective-rank-rel-tol", type=float, default=1e-2)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="process only the first N cases after loading")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be > 0")

    script_dir = Path(__file__).resolve().parent
    gt_path = resolve_cli_path(args.gt, script_dir, must_exist=True)
    save_dir = resolve_cli_path(args.save_dir, script_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    positions_gt, orientations_flat, case_ids_gt, gt_meta = load_ground_truth_dataset(gt_path)
    if args.max_cases is not None:
        n_use = min(args.max_cases, len(case_ids_gt))
        print(f"Applying --max-cases {args.max_cases}: using first {n_use} cases.")
        positions_gt = positions_gt[:n_use]
        orientations_flat = orientations_flat[:n_use]
        case_ids_gt = case_ids_gt[:n_use]
    orientations_gt = orientations_flat.reshape(positions_gt.shape[0], positions_gt.shape[1], 3, 3)
    s_grid = np.linspace(0.0, 1.0, positions_gt.shape[1])
    length_m = float(gt_meta.get("length_m", L_DEFAULT))
    order_cfgs = parse_order_configs(args.orders)

    print("Observability study")
    print(f"  GT dataset      : {gt_path}")
    print(f"  Cases           : {len(case_ids_gt)}")
    print(f"  Tip Jacobian s  : 1.0")
    print(f"  Gamma           : {args.gamma}")
    print(f"  Numerical tol   : {args.numerical_rank_rel_tol}")
    print(f"  Effective tol   : {args.effective_rank_rel_tol}")

    per_case_rows: List[Dict] = []
    npz_arrays: Dict[str, np.ndarray] = {
        "case_ids": case_ids_gt.astype(int),
        "order_labels": np.array([order_label(cfg) for cfg in order_cfgs], dtype=object),
    }

    for order_cfg in order_cfgs:
        ox, oy, oz = order_cfg
        label = order_label(order_cfg)
        slug = order_slug(order_cfg)
        nm = total_params(ox, oy, oz)
        print(f"\n  Evaluating order {label} with n_m={nm}")

        modal_bank = np.zeros((len(case_ids_gt), nm), dtype=float)
        singular_bank = np.zeros((len(case_ids_gt), 6), dtype=float)
        weak_bank = np.zeros((len(case_ids_gt), 6), dtype=float)
        numerical_ranks = np.zeros(len(case_ids_gt), dtype=int)
        effective_ranks = np.zeros(len(case_ids_gt), dtype=int)
        conds = np.zeros(len(case_ids_gt), dtype=float)
        smin_nonzero = np.zeros(len(case_ids_gt), dtype=float)

        for idx, cid in enumerate(case_ids_gt.astype(int)):
            T_obs = build_transform_stack(positions_gt[idx], orientations_gt[idx])
            m_fit = estimate_modal_from_gt_frames(T_obs, s_grid, ox, oy, oz)
            terms = build_virtual_work_terms(
                m=m_fit,
                tau=np.zeros(4),
                order_cfg=order_cfg,
                gamma=args.gamma,
                length_m=length_m,
            )
            diag = svd_diagnostics(
                terms.A,
                numerical_rank_rel_tol=args.numerical_rank_rel_tol,
                effective_rank_rel_tol=args.effective_rank_rel_tol,
            )

            modal_bank[idx] = m_fit
            singular_bank[idx] = diag.singular_values
            weak_bank[idx] = diag.weakest_right_vector
            numerical_ranks[idx] = diag.numerical_rank
            effective_ranks[idx] = diag.effective_rank
            conds[idx] = diag.cond_nonzero
            smin_nonzero[idx] = diag.smallest_nonzero_sigma

            row = {
                "case_id": cid,
                "order_label": label,
                "order_x": ox,
                "order_y": oy,
                "order_z": oz,
                "n_m": nm,
                "numerical_rank": diag.numerical_rank,
                "effective_rank": diag.effective_rank,
                "cond_nonzero": diag.cond_nonzero,
                "smallest_nonzero_sigma": diag.smallest_nonzero_sigma,
                "smallest_effective_sigma": diag.smallest_retained_sigma_effective,
            }
            for s_idx in range(6):
                row[f"sigma_{s_idx + 1}"] = diag.singular_values[s_idx]
            for comp_idx, comp in enumerate(COMPONENT_LABELS):
                row[f"weak_abs_{comp}"] = abs(diag.weakest_right_vector[comp_idx])
            per_case_rows.append(row)

        npz_arrays[f"modal_bank_{slug}"] = modal_bank
        npz_arrays[f"singular_values_{slug}"] = singular_bank
        npz_arrays[f"weakest_right_vec_{slug}"] = weak_bank
        npz_arrays[f"numerical_rank_{slug}"] = numerical_ranks
        npz_arrays[f"effective_rank_{slug}"] = effective_ranks
        npz_arrays[f"cond_nonzero_{slug}"] = conds
        npz_arrays[f"smallest_nonzero_sigma_{slug}"] = smin_nonzero

    summary_rows: List[Dict] = []
    for order_cfg in order_cfgs:
        label = order_label(order_cfg)
        rows = [row for row in per_case_rows if row["order_label"] == label]
        num_counter = Counter(int(row["numerical_rank"]) for row in rows)
        eff_counter = Counter(int(row["effective_rank"]) for row in rows)
        weak_avg = np.mean(
            [[row[f"weak_abs_{comp}"] for comp in COMPONENT_LABELS] for row in rows],
            axis=0,
        )
        summary = {
            "order_label": label,
            "order_x": order_cfg[0],
            "order_y": order_cfg[1],
            "order_z": order_cfg[2],
            "n_m": total_params(*order_cfg),
            "n_cases": len(rows),
            "smallest_nonzero_sigma_mean": float(np.mean([row["smallest_nonzero_sigma"] for row in rows])),
            "smallest_nonzero_sigma_med": float(np.median([row["smallest_nonzero_sigma"] for row in rows])),
            "cond_nonzero_mean": float(np.mean([row["cond_nonzero"] for row in rows])),
            "cond_nonzero_med": float(np.median([row["cond_nonzero"] for row in rows])),
        }
        for rank_value in sorted(set(num_counter) | set(eff_counter)):
            summary[f"numerical_rank_frac_{rank_value}"] = num_counter[rank_value] / len(rows)
            summary[f"effective_rank_frac_{rank_value}"] = eff_counter[rank_value] / len(rows)
        for comp_idx, comp in enumerate(COMPONENT_LABELS):
            summary[f"weak_abs_{comp}"] = float(weak_avg[comp_idx])
        summary_rows.append(summary)

    print("\nSummary by order")
    for row in summary_rows:
        num_fracs = ", ".join(
            f"r{key.split('_')[-1]}={value:.2f}"
            for key, value in row.items()
            if key.startswith("numerical_rank_frac_") and value > 0.0
        )
        eff_fracs = ", ".join(
            f"r{key.split('_')[-1]}={value:.2f}"
            for key, value in row.items()
            if key.startswith("effective_rank_frac_") and value > 0.0
        )
        print(
            f"  {row['order_label']:<8s} n_m={row['n_m']:>2d}  "
            f"smin_med={row['smallest_nonzero_sigma_med']:.3e}  "
            f"cond_med={row['cond_nonzero_med']:.2e}  "
            f"num[{num_fracs}]  eff[{eff_fracs}]"
        )

    save_csv(per_case_rows, save_dir / "kirchhoff_wrench_observability_per_case.csv")
    save_csv(summary_rows, save_dir / "kirchhoff_wrench_observability_summary.csv")
    save_json(
        {
            "config": {
                "workflow": "tip-wrench observability vs modal order",
                "gt": str(gt_path),
                "orders": [list(cfg) for cfg in order_cfgs],
                "gamma": args.gamma,
                "length_m": length_m,
                "numerical_rank_rel_tol": args.numerical_rank_rel_tol,
                "effective_rank_rel_tol": args.effective_rank_rel_tol,
                "projection_method": "curvature least-squares fit from saved GT frames",
            },
            "summary": summary_rows,
            "per_case": per_case_rows,
        },
        save_dir / "kirchhoff_wrench_observability_results.json",
    )
    save_npz(npz_arrays, save_dir / "kirchhoff_wrench_observability_svd.npz")
    build_observability_figure(
        per_case_rows=per_case_rows,
        summary_rows=summary_rows,
        order_cfgs=order_cfgs,
        save_stem=save_dir / "kirchhoff_wrench_observability_vs_order",
    )


if __name__ == "__main__":
    main()
