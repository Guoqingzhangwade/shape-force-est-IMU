"""Render two local-sensitivity figures from saved JSON; run no simulations.

Example:
    python -B render_figures.py --results run/results.json

Outputs default to the results directory. Existing figures are never overwritten.
Only saved numerical results are used; no production modules are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7"}
NAMES = {
    "A": "A: dense oracle",
    "B": "B: noiseless triplet fit",
    "C": "C: noisy triplet fit",
    "D": "D: production EKF",
}


def style_axes(ax):
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def spectra_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", "-.", ":"]
    for key, marker, linestyle in zip("ABCD", markers, styles):
        case = data["cases"][key]
        values = case["directional_gains"]
        singular = np.asarray(values["singular_values"], dtype=float)
        gains = np.asarray(values["per_noise_gain"], dtype=float)
        if singular.shape != gains.shape or singular.ndim != 1:
            raise ValueError(f"Inconsistent saved directional arrays for {key}")
        x = np.arange(1, len(singular) + 1)
        kwargs = dict(color=COLORS[key], marker=marker, linestyle=linestyle,
                      linewidth=1.6, markersize=5.8, markerfacecolor="white")
        axes[0].semilogy(x, singular, label=NAMES[key], **kwargs)
        axes[1].semilogy(x, gains, label=NAMES[key], **kwargs)
    axes[0].set_title("(a) Measurement singular values", loc="left")
    axes[0].set_ylabel(r"Singular value of $H_s = H S / \sigma_R$")
    axes[0].set_ylim(12, 190)
    axes[0].set_yticks([20, 50, 100, 150], ["20", "50", "100", "150"])
    axes[0].legend(frameon=False, loc="lower left", fontsize=9.6)
    conditions = [data["cases"][key]["H_scaled_condition"] for key in "ABCD"]
    axes[0].text(0.97, 0.94,
                 f"All four scaled conditions:\n{min(conditions):.4f} to {max(conditions):.4f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9.5,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    axes[1].set_title("(b) Directional wrench response", loc="left")
    axes[1].set_ylabel(r"Scaled wrench gain / whitened residual: $\|J_x v_i\|/s_i$")
    axes[1].set_ylim(0.007, 3.0)
    for ax in axes:
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Local H direction index (descending singular value)")
        style_axes(ax)
    fig.suptitle("Case 0: local measurement and wrench sensitivity at A / B / C / D",
                 fontsize=13.5, y=0.97)
    fig.text(0.5, 0.075,
             "Indices order each state's own singular vectors; equal indices need not identify the same direction.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.035,
             r"$S^T Q S=I$ (curvature $L^2$ metric); $\sigma_R=0.5^\circ$; wrench scales $M_0=EI/L$, $F_0=EI/L^2$."
             "  Local unregularized response; no EKF covariance interpretation.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.86, wspace=0.25)
    return fig


def perturbations_figure(data):
    rows = data["controlled_perturbations"]
    grouped = {}
    for row in rows:
        key = (int(row["direction_index"]), float(row["target_stacked_whitened_norm"]))
        grouped.setdefault(key, []).append(row)
    groups = sorted(grouped, key=lambda item: (-item[1], item[0]))
    if len(groups) > 5:
        raise ValueError("This compact figure supports at most five saved perturbation groups")
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5))
    labels = []
    for index, ((direction, budget), color) in enumerate(zip(groups, palette)):
        group = sorted(grouped[(direction, budget)], key=lambda row: row["sign"])
        orientation = float(np.mean([r["stacked_measurement_change_deg"] for r in group]))
        labels.append(f"Dir. {direction + 1}\n{orientation:.3g} deg")
        x = np.asarray([r["shape_change_rms_mm"] for r in group])
        y = 1000.0 * np.asarray([r["force_change_N"] for r in group])
        error = 100.0 * np.asarray([r["relative_linear_prediction_error"] for r in group])
        for j, row in enumerate(group):
            marker = "o" if row["sign"] > 0 else "v"
            axes[0].loglog(x[j], y[j], marker=marker, linestyle="none", color=color,
                          markersize=7.5, markerfacecolor="none")
            offset = 0.05 if row["sign"] > 0 else -0.05
            axes[1].semilogy(index + offset, error[j], marker=marker, linestyle="none",
                            color=color, markersize=7.5, markerfacecolor="none")
        axes[0].plot(x, y, color=color, linewidth=1.3)
        axes[1].plot([index - 0.05, index + 0.05], [error[0], error[-1]],
                     color=color, linewidth=1.1)
        # Avoid overlapping the two neighboring 0.05-degree responses.
        offset = (10, 3) if index != 2 else (-12, -15)
        ha = "left" if index != 2 else "right"
        axes[0].annotate(f"Dir. {direction + 1}; {orientation:.3g} deg", (x.mean(), y.mean()),
                         xytext=offset, textcoords="offset points", color=color,
                         fontsize=10, ha=ha, va="center")
        axes[1].text(index, error.max() * 1.3, f"{error.mean():.3g}%",
                     ha="center", va="bottom", color=color, fontsize=10)
    axes[0].set_title("(a) Nonlinear force and shape changes", loc="left")
    axes[0].set_xlabel("Centerline change: true RMS (mm)")
    axes[0].set_ylabel("Force-vector change (mN)")
    axes[0].margins(x=0.42, y=0.22)
    axes[1].set_title("(b) Error of the linear wrench prediction", loc="left")
    axes[1].set_ylabel("Relative error of the scaled 6D wrench vector (%)")
    axes[1].set_xticks(range(len(groups)), labels)
    axes[1].set_xlabel("A local direction / stacked residual-vector change")
    axes[1].set_xlim(-0.5, len(groups) - 0.5)
    all_errors = [100.0 * r["relative_linear_prediction_error"] for r in rows]
    axes[1].set_ylim(min(all_errors) * 0.5, max(all_errors) * 2.4)
    for ax in axes:
        style_axes(ax)
    fig.suptitle("Case 0 at A: bounded positive and negative perturbations", fontsize=13.5, y=0.97)
    fig.text(0.5, 0.105,
             "Open circles: positive step; open triangles: negative step. Direction indices refer only to H at A.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.030,
             "Labels are norms of changes in Log(Rmeas Rpred.T) over three IMUs; 0.05 / 0.005 deg correspond to whitened budgets 0.1 / 0.01.\n"
             "Vector error uses moment / M0 and force / F0 at a common world/base origin; these perturbations are not new noise trials.",
             ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.27, top=0.86, wspace=0.26)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Saved Step 2 results.json; no simulation or production imports are performed.")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure directory (default: results.json parent); existing figure files cause an error.")
    args = parser.parse_args()
    result_path = args.results.resolve(strict=True)
    output_dir = (args.output_dir or result_path.parent).resolve()
    targets = [output_dir / "local_sensitivity_spectra.png",
               output_dir / "controlled_perturbations.png"]
    for target in targets:
        if target.exists():
            parser.error(f"Refusing to overwrite existing figure: {target}")
    raw = result_path.read_bytes()
    data = json.loads(raw)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "savefig.facecolor": "white"})
    figures = [spectra_figure(data), perturbations_figure(data)]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Case-local Step 2 sensitivity diagnostic",
                "Description": "Saved results only; no additional simulations.",
                "ResultsSHA256": hashlib.sha256(raw).hexdigest()}
    for figure, target in zip(figures, targets):
        with target.open("xb") as output:
            figure.savefig(output, format="png", dpi=180, metadata=metadata)
        plt.close(figure)
        print(target)


if __name__ == "__main__":
    main()
