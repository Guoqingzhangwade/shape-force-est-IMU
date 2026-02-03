#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-ready figures for EKF shape estimation comparison.

This script produces high-quality figures suitable for manuscript submission,
including proper sizing, fonts, and both vector (PDF) and raster (PNG) formats.

Usage:
    python generate_publication_figures.py --output-dir ./figures
    python generate_publication_figures.py --trials 20 --high-quality
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import sys

# Import from the comparison script
from ekf_measurement_compare_v2 import (
    build_meas_seq, ekf_so3, ekf_quat, reconstruct_shape_3d,
    summarize
)

# ==================== PUBLICATION SETTINGS ====================

# Configure matplotlib for publication quality
rc('font', family='serif', size=10)
rc('text', usetex=False)  # Set to True if you have LaTeX installed
rc('figure', dpi=300)
rc('savefig', dpi=300, bbox='tight', pad_inches=0.05)
rc('axes', linewidth=0.8, labelsize=10)
rc('xtick', labelsize=9)
rc('ytick', labelsize=9)
rc('legend', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
rc('lines', linewidth=1.5, markersize=4)

# Standard figure sizes (inches) for two-column format
SINGLE_COL_WIDTH = 3.5  # Single column width
DOUBLE_COL_WIDTH = 7.0  # Double column width
GOLDEN_RATIO = 1.618

# Color scheme (colorblind-friendly)
COLORS = {
    'true': '#000000',           # Black
    'so3_analytic': '#0072B2',   # Blue (main method)
    'so3_numeric': '#009E73',    # Green
    'quat_numeric': '#D55E00',   # Orange-red
    'quat_analytic': '#CC79A7',  # Purple
}

LABELS = {
    'so3_analytic': 'SO(3) analytic',
    'so3_numeric': 'SO(3) numeric',
    'quat_numeric': 'Quat. numeric',
    'quat_analytic': 'Quat. analytic',
}


# ==================== FIGURE GENERATION FUNCTIONS ====================

def generate_comparison_plot(stats_all, rmse_hist_all, output_dir, args):
    """
    Figure 1: Comparison of all 4 measurement model formulations.

    Double column, shows convergence for all methods.
    Uses log scale only if there's a large performance gap (>10x).
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH*0.5))

    t = np.arange(args.steps)

    # Plot order (SO(3) methods first for better visibility)
    plot_order = ['so3_analytic', 'so3_numeric', 'quat_numeric', 'quat_analytic']

    # Check if we should use log scale (if max RMSE is >10x min RMSE)
    final_rmses = [rmse_hist_all[m][-1] for m in plot_order if m in rmse_hist_all]
    max_rmse = max(final_rmses)
    min_rmse = min([r for r in final_rmses if r > 0])
    use_log = (max_rmse / min_rmse) > 10

    for method in plot_order:
        if method in rmse_hist_all:
            ax.plot(t, rmse_hist_all[method], color=COLORS[method],
                   linewidth=2.5, label=LABELS[method], alpha=0.9)

    ax.set_xlabel('Time step')
    ax.set_ylabel('RMSE')
    ax.set_title('Measurement Model Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', ncol=2)

    if use_log:
        ax.set_yscale('log')
        ax.set_ylim(bottom=min_rmse * 0.5)
    else:
        ax.set_ylim(bottom=0, top=max_rmse * 1.1)

    # Save figure
    save_figure(fig, output_dir, 'fig1_comparison', args)
    plt.close(fig)

    return fig


def generate_all_shapes_comparison(m_true, final_est_all, e3, gamma, output_dir, args):
    """
    Figure 2: Shape comparison for all methods.

    Double column, 2D plot showing all methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH*0.8))

    # Reconstruct true shape
    pos_true, _ = reconstruct_shape_3d(m_true, e3, gamma, num_points=100)

    # 3D view
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
            color=COLORS['true'], linewidth=3, label='Ground truth', alpha=0.7)

    for method in ['so3_analytic', 'so3_numeric', 'quat_numeric']:
        if method in final_est_all:
            pos_est, _ = reconstruct_shape_3d(final_est_all[method], e3, gamma, num_points=100)
            ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
                   color=COLORS[method], linewidth=1.5, label=LABELS[method],
                   linestyle='--', alpha=0.8)

    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title('3D View', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # XY projection
    ax2 = axes[0, 1]
    ax2.plot(pos_true[:, 0], pos_true[:, 1], color=COLORS['true'],
             linewidth=3, label='Ground truth', alpha=0.7)
    for method in ['so3_analytic', 'so3_numeric', 'quat_numeric']:
        if method in final_est_all:
            pos_est, _ = reconstruct_shape_3d(final_est_all[method], e3, gamma, num_points=100)
            ax2.plot(pos_est[:, 0], pos_est[:, 1], color=COLORS[method],
                    linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # XZ projection
    ax3 = axes[1, 0]
    ax3.plot(pos_true[:, 0], pos_true[:, 2], color=COLORS['true'],
             linewidth=3, alpha=0.7)
    for method in ['so3_analytic', 'so3_numeric', 'quat_numeric']:
        if method in final_est_all:
            pos_est, _ = reconstruct_shape_3d(final_est_all[method], e3, gamma, num_points=100)
            ax3.plot(pos_est[:, 0], pos_est[:, 2], color=COLORS[method],
                    linewidth=1.5, linestyle='--', alpha=0.8)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # YZ projection
    ax4 = axes[1, 1]
    ax4.plot(pos_true[:, 1], pos_true[:, 2], color=COLORS['true'],
             linewidth=3, alpha=0.7)
    for method in ['so3_analytic', 'so3_numeric', 'quat_numeric']:
        if method in final_est_all:
            pos_est, _ = reconstruct_shape_3d(final_est_all[method], e3, gamma, num_points=100)
            ax4.plot(pos_est[:, 1], pos_est[:, 2], color=COLORS[method],
                    linewidth=1.5, linestyle='--', alpha=0.8)
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Shape Reconstruction: Method Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    save_figure(fig, output_dir, 'fig2_shape_comparison', args)
    plt.close(fig)

    return fig


def save_figure(fig, output_dir, filename, args):
    """Save figure in multiple formats suitable for publication."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as PDF (vector graphics, preferred for LaTeX)
    pdf_path = output_path / f"{filename}.pdf"
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {pdf_path}")

    # Save as PNG (high resolution raster)
    png_path = output_path / f"{filename}.png"
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {png_path}")

    # Save as SVG if requested (editable vector graphics)
    if args.save_svg:
        svg_path = output_path / f"{filename}.svg"
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"  Saved: {svg_path}")


def save_comparison_table_latex(stats_all, output_dir, filename):
    """Generate LaTeX table for method comparison."""
    output_path = Path(output_dir) / f"{filename}.tex"

    methods = ['quat_numeric', 'quat_analytic', 'so3_analytic', 'so3_numeric']
    method_labels = {
        'quat_numeric': 'Quat. num.',
        'quat_analytic': 'Quat. ana.',
        'so3_analytic': 'SO(3) ana.',
        'so3_numeric': 'SO(3) num.',
    }

    with open(output_path, 'w') as f:
        f.write("% Method comparison table (generated automatically)\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of Measurement Model Formulations}\n")
        f.write("\\label{tab:method_comparison}\n")
        f.write("\\begin{tabular}{l" + "c" * len(methods) + "}\n")
        f.write("\\hline\n")

        # Header
        header = "Metric"
        for m in methods:
            header += f" & {method_labels[m]}"
        f.write(header + " \\\\\n")
        f.write("\\hline\n")

        # Metrics
        metrics = [
            ('Final RMSE', 'rmse_final'),
            ('Mean RMSE', 'rmse_mean'),
            ('Time/step (s)', 'time_per_step'),
        ]

        for label, key in metrics:
            row = label
            for m in methods:
                if m in stats_all:
                    mean, std = stats_all[m][key]
                    row += f" & ${mean:.4f} \\pm {std:.4f}$"
                else:
                    row += " & N/A"
            f.write(row + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"  Saved LaTeX table: {output_path}")


# ==================== MAIN EXECUTION ====================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for EKF comparison"
    )
    parser.add_argument("--output-dir", type=str, default="./figures",
                       help="Output directory for figures")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of Monte Carlo trials")
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of time steps")
    parser.add_argument("--gamma", type=int, default=20,
                       help="Number of integration segments")
    parser.add_argument("--meas-std-deg", type=float, default=0.5,
                       help="Measurement noise (degrees)")
    parser.add_argument("--seed-offset", type=int, default=100,
                       help="Random seed offset")
    parser.add_argument("--save-svg", action="store_true",
                       help="Also save SVG format")
    parser.add_argument("--high-quality", action="store_true",
                       help="Use higher quality settings (more trials, smoother)")

    args = parser.parse_args()

    # Adjust for high quality
    if args.high_quality:
        args.trials = max(args.trials, 20)
        args.gamma = max(args.gamma, 30)
        print("High-quality mode: using trials={}, gamma={}".format(args.trials, args.gamma))

    # Setup
    print("\n" + "="*80)
    print("PUBLICATION FIGURE GENERATOR - MAIN TEXT ONLY")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Configuration: {args.trials} trials x {args.steps} steps, gamma={args.gamma}")
    print("="*80 + "\n")

    # Physical parameters
    imu_pos = np.array([0.25, 0.50, 0.75], dtype=float)
    L_phys = 100.0
    e3 = np.array([0.0, 0.0, L_phys], dtype=float)

    # Generate seeds
    seeds = [args.seed_offset + i for i in range(args.trials)]

    # ========== Run EKF for all methods ==========
    print("Running EKF for all 4 methods...")
    results_all = {'quat_numeric': [], 'quat_analytic': [],
                   'so3_analytic': [], 'so3_numeric': []}
    rmse_hist_all = {k: [] for k in results_all.keys()}
    final_est_all = {k: None for k in results_all.keys()}
    m_true_final = None

    for trial in range(args.trials):
        rng = np.random.RandomState(seeds[trial])
        m_true = rng.uniform(-2, 2, 5)

        meas_q, meas_R = build_meas_seq(m_true, imu_pos, e3, args.gamma,
                                        args.steps, args.meas_std_deg, rng)

        # Run all methods
        out_qn = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma,
                         "quat_numeric", args.meas_std_deg, R_scale=1.0)
        out_qa = ekf_quat(meas_q, m_true, imu_pos, e3, args.gamma,
                         "quat_analytic", args.meas_std_deg, R_scale=1.0)
        out_sa = ekf_so3(meas_R, m_true, imu_pos, e3, args.gamma,
                        "so3_analytic", args.meas_std_deg, R_scale=1.0)
        out_sn = ekf_so3(meas_R, m_true, imu_pos, e3, args.gamma,
                        "so3_numeric", args.meas_std_deg, R_scale=1.0)

        results_all['quat_numeric'].append(out_qn)
        results_all['quat_analytic'].append(out_qa)
        results_all['so3_analytic'].append(out_sa)
        results_all['so3_numeric'].append(out_sn)

        rmse_hist_all['quat_numeric'].append(np.sqrt(np.mean((out_qn["hist"] - m_true) ** 2, axis=1)))
        rmse_hist_all['quat_analytic'].append(np.sqrt(np.mean((out_qa["hist"] - m_true) ** 2, axis=1)))
        rmse_hist_all['so3_analytic'].append(np.sqrt(np.mean((out_sa["hist"] - m_true) ** 2, axis=1)))
        rmse_hist_all['so3_numeric'].append(np.sqrt(np.mean((out_sn["hist"] - m_true) ** 2, axis=1)))

        if trial == args.trials - 1:
            m_true_final = m_true
            final_est_all['quat_numeric'] = out_qn["m_est"]
            final_est_all['quat_analytic'] = out_qa["m_est"]
            final_est_all['so3_analytic'] = out_sa["m_est"]
            final_est_all['so3_numeric'] = out_sn["m_est"]

        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{args.trials} trials")

    stats_all = {k: summarize(v) for k, v in results_all.items()}
    rmse_all = {k: np.mean(np.vstack(v), axis=0) for k, v in rmse_hist_all.items()}

    # ========== Generate figures ==========
    print("\nGenerating figures...")
    generate_comparison_plot(stats_all, rmse_all, args.output_dir, args)
    generate_all_shapes_comparison(m_true_final, final_est_all, e3,
                                   args.gamma, args.output_dir, args)

    # ========== Generate LaTeX table ==========
    print("\nGenerating LaTeX table...")
    save_comparison_table_latex(stats_all, args.output_dir, 'table1_comparison')

    print("\n" + "="*80)
    print("DONE! All figures and tables saved to:", args.output_dir)
    print("="*80)
    print("\nFiles generated:")
    print("  - fig1_comparison.pdf/png")
    print("  - fig2_shape_comparison.pdf/png")
    print("  - table1_comparison.tex")
    print()


if __name__ == "__main__":
    main()
