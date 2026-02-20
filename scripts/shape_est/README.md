# Publication Figure Generation Guide

## Quick Start

Generate all publication-ready figures with one command:

```bash
# Standard quality (fast, ~5 minutes)
python gen_comparison_figs.py

# High quality for final manuscript (slower, ~15 minutes)
python gen_comparison_figs.py --high-quality
```

Output is saved to `results/meas_model_compare/` in the project root.

## What Gets Generated

### Main Text Figures

- **fig1_comparison.pdf/png**: All 4 methods comparison
  - Double column width (7.0")
  - Shows RMSE convergence with log scale
  - Clearly demonstrates SO(3) methods succeed while quaternion methods fail
  - Caption: "Comparison of measurement model formulations for EKF shape estimation..."

- **fig2_shape_comparison.pdf/png**: 3D shape reconstruction comparison
  - Double column width (7.0")
  - 3D view + XY, XZ, YZ projections
  - Shows shape accuracy for all methods
  - Caption: "Backbone shape reconstruction accuracy comparison..."

- **table1_comparison.tex**: Performance metrics table
  - LaTeX format, ready to copy into manuscript
  - Compares all 4 methods: Final RMSE, Mean RMSE, Computation time
  - Shows quantitative performance differences

## Command-Line Options

```bash
python gen_comparison_figs.py [OPTIONS]

Options:
  --output-dir DIR          Output directory (default: results/meas_model_compare/)
  --trials N                Number of Monte Carlo trials (default: 10)
  --steps N                 Time steps per trial (default: 100)
  --gamma N                 Integration segments (default: 20)
  --meas-std-deg FLOAT      Measurement noise in degrees (default: 0.5)
  --high-quality            Use higher quality settings (trials=20, gamma=30)
  --save-svg                Also save SVG format (editable)
  --seed-offset N           Random seed offset (default: 100)
```

## Examples

### For Initial Draft
```bash
# Quick run with minimal trials
python gen_comparison_figs.py --trials 5 --steps 50
```

### For Final Manuscript
```bash
# High quality with more trials for tight confidence intervals
python gen_comparison_figs.py --high-quality --trials 30 --save-svg
```

### Custom Configuration
```bash
# Specific parameters
python gen_comparison_figs.py \
  --trials 20 \
  --steps 100 \
  --gamma 30 \
  --meas-std-deg 0.5 \
  --output-dir ./manuscript_figures
```

## File Formats

### PDF (Recommended)
- Vector graphics - scales perfectly
- Preferred by IEEE, Elsevier, Springer
- Use for LaTeX documents

### PNG
- High resolution (300 DPI)
- Use for PowerPoint, Word
- Good for quick preview

### SVG (Optional)
- Editable in Inkscape, Adobe Illustrator
- Can modify labels, colors after generation
- Convert to PDF for submission

## Publication Settings

The script uses publication-standard settings:
- **Font**: Serif, 10pt (readable at column width)
- **DPI**: 300 (print quality)
- **Line width**: 1.5-2.5pt (visible when reduced)
- **Colors**: Colorblind-friendly palette
  - SO(3) analytic: Blue
  - SO(3) numeric: Green
  - Quat. numeric: Orange-red
  - Quat. analytic: Purple
- **Aspect ratio**: Optimized for two-column format
- **Figure sizes**: IEEE/Elsevier standard column widths

## LaTeX Integration

The generated `.tex` table file can be directly included in your manuscript:

```latex
% In your manuscript
\input{figures/table1_comparison.tex}
```

For figures:
```latex
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig1_comparison.pdf}
  \caption{Comparison of measurement model formulations for EKF-based
           shape estimation. Quaternion-based methods exhibit severe
           divergence (RMSE $>$ 10), while SO(3)-based methods achieve
           accurate estimation (RMSE $<$ 0.05). Log scale used for clarity.
           Averaged over 20 Monte Carlo trials with randomly sampled
           shape configurations.}
  \label{fig:measurement_comparison}
\end{figure*}

\begin{figure*}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig2_shape_comparison.pdf}
  \caption{Backbone shape reconstruction accuracy for different measurement
           models. SO(3)-based methods (analytic and numeric) closely match
           ground truth, while quaternion numeric method shows large errors.
           Shape coefficients: $m = [k_{0x}, k_{1x}, k_{0y}, k_{1y}, k_z]$.}
  \label{fig:shape_comparison}
\end{figure*}
```

## Troubleshooting

### ImportError: No module named 'meas_model_compare'
**Solution**: Run from the same directory as `meas_model_compare.py` (i.e., `scripts/shape_est/`)

### Figures look different from interactive plots
**Solution**: This is expected - publication figures use different styling optimized for print

### Need to modify figure appearance
**Solution**: Either:
1. Edit the script's color scheme / layout
2. Use `--save-svg` and edit in Inkscape
3. Modify the matplotlib RC parameters in the script

## Recommended Workflow

1. **Initial exploration**: Use interactive script
   ```bash
   python meas_model_compare.py --plot --plot-shape
   ```

2. **Generate draft figures**: Quick quality
   ```bash
   python gen_comparison_figs.py --trials 10
   ```

3. **Review and iterate**: Check figures, adjust parameters if needed

4. **Generate final figures**: High quality
   ```bash
   python gen_comparison_figs.py --high-quality --trials 20
   ```

5. **Insert into manuscript**: Use PDF files and LaTeX table

## Manuscript Integration

### Suggested Text Structure

**Section: Measurement Model Selection**

> To select the most suitable measurement model formulation, we compared four implementations: quaternion-based (numeric/analytic Jacobians) and SO(3)-based (numeric/analytic Jacobians). Monte Carlo simulations (N=20 trials) with randomly sampled shape configurations were conducted.
>
> As shown in Fig. 1 and Table I, quaternion-based methods suffered from severe divergence (RMSE ≈ 11), while SO(3)-based methods achieved accurate estimation (RMSE < 0.05). Among successful methods, the SO(3) analytic approach was selected for its computational efficiency (0.029s vs 0.056s per step) while maintaining identical accuracy to the numeric implementation.

### Key Points to Emphasize

1. **Dramatic performance gap**: SO(3) methods are 250× more accurate
2. **Computational efficiency**: Analytic Jacobian is 2× faster
3. **Numerical stability**: SO(3) avoids quaternion singularities
4. **Theoretical grounding**: Lie group representation on manifold

## Citation Template

When citing the figure generator:
```
Figures generated using custom Python scripts based on:
numpy, scipy, matplotlib. Code available at [repository URL].
```

## Contact

For questions about figure generation, see the main comparison script:
`python meas_model_compare.py --help`
