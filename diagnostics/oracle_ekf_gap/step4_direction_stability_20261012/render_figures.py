"""Render the Step 4 figure from the committed compact extraction; run no simulations.

Example:
    python -B render_figures.py --output-dir "$env:TEMP/step4-figures-new"

Existing figures are never overwritten. No production modules are imported.
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

STEP = Path(__file__).resolve().parent
BLUE, INK = "#0072B2", "#444444"


def style_axes(ax):
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir.resolve() / "amplified_direction_summary.png"
    if target.exists():
        parser.error(f"Refusing to overwrite existing figure: {target}")
    raw = (STEP / "results/runs_compact.json").read_bytes()
    rows = [r for r in json.loads(raw) if r["valid"]]
    fit = json.loads((STEP / "results/summary.json").read_text(encoding="utf-8"))["post_hoc"]["log_gain_vs_log_mean_bending"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
                         "axes.labelsize": 10.5, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    marker = dict(marker="o", linestyle="none", color=BLUE, markersize=5.5, markerfacecolor="none", markeredgewidth=1.1)

    base = [r for r in rows if r["noise_realization"] == 0]
    bending = np.array([np.linalg.norm(np.array(r["state_A_q"])[[0, 3]]) for r in base])
    gain = np.array([r["amplified_direction_gain"] for r in base])
    grid = np.geomspace(bending.min() * 0.9, bending.max() * 1.1, 50)
    axes[0].loglog(grid, fit["gain_at_unit_bending"] * grid ** fit["slope"], color=INK, linewidth=1.2)
    axes[0].loglog(bending, gain, **marker)
    axes[0].annotate(f"fit: gain = {fit['gain_at_unit_bending']:.2f} x bending^({fit['slope']:.2f})",
                     (0.97, 0.94), xycoords="axes fraction", ha="right", va="top", color=INK, fontsize=10)
    axes[0].set_title("(a) Straighter rods amplify more (50 cases)", loc="left")
    axes[0].set_xlabel("Mean normalized bending curvature of the oracle state (constant mode)")
    axes[0].set_ylabel("Largest scaled-wrench gain per whitened residual at A")

    d = [r["saved_changes"]["D"] for r in rows]
    predicted = np.array([r["amplified_direction_gain"] for r in rows]) * np.abs([x["amplified_component_whitened_residual"] for x in d])
    actual = np.array([r["metrics"]["D"]["force_error_over_F0"] for r in rows])
    limits = [min(predicted.min(), actual.min()) * 0.7, max(predicted.max(), actual.max()) * 1.4]
    axes[1].loglog(limits, limits, color=INK, linewidth=1.2)
    axes[1].loglog(predicted, actual, **marker)
    axes[1].annotate("equal", (limits[0] * 3, limits[0] * 3), xytext=(8, -10), textcoords="offset points",
                     ha="left", color=INK, fontsize=10)
    axes[1].set_xlim(limits); axes[1].set_ylim(limits)
    axes[1].set_title("(b) EKF force error follows the amplified component (250 runs)", loc="left")
    axes[1].set_xlabel("Gain x |state-error component along the amplified direction| (whitened)")
    axes[1].set_ylabel("Production EKF force error / F0")
    for ax in axes:
        style_axes(ax)
    fig.text(0.5, 0.035,
             "Local analysis at the dense-oracle state A. Five saved noise patterns are shared by all 50 cases; "
             "points are not independent noise trials.", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.92, wspace=0.22)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        fig.savefig(output, format="png", dpi=180, metadata={"Title": "Step 4 amplified-direction summary",
                    "Description": "Saved results only; no additional simulations.",
                    "ResultsSHA256": hashlib.sha256(raw).hexdigest()})
    print(target)


if __name__ == "__main__":
    main()
