# CRT — Cosserat Rod Tool Scripts

This folder contains the physics model and data-generation pipeline for a
**single-segment tendon-driven continuum robot (TDCR)** modelled as a static
Cosserat rod.

---

## Folder layout

```
crt/
├── cosserat_rod_model.py        # Core physics model  (import this)
├── utils.py                     # Math helpers        (import this)
├── virtual_work.py              # Modal kinematics & virtual-work tools
│
├── crt_gt_gen.py                # Quick visual demo   (no file saved)
├── crt_gt_gen_plot.py           # Same + tip frames   (no file saved)
├── crt_gt_gen_output_file.py    # Generate + SAVE ground-truth dataset  ← main
├── crt_meas_gen_output_file.py  # Add sensor noise → SAVE measurements  ← main
│
└── archive/                     # Old exploratory scripts (not used)
```

---

## The two-step data pipeline

```
Step 1                              Step 2
─────────────────────────────────   ──────────────────────────────────────
crt_gt_gen_output_file.py      →   crt_meas_gen_output_file.py
  samples N random configurations     reads the GT file
  solves Cosserat rod statics         adds cable-tension noise
  saves ground-truth poses            adds IMU orientation noise
                                      saves noisy measurements
  Output: tdcr_gt_samples.npz   →   Output: tdcr_meas_samples.npz
```

Both steps must be run from inside the `crt/` directory so Python can find
the local imports (`cosserat_rod_model`, `utils`).

```bash
cd src/shape_force_est_imu/crt
```

---

## Step 1 — Generate ground-truth dataset

**Script:** `crt_gt_gen_output_file.py`

Draws random cable tensions and tip wrenches, solves the rod statics via
a boundary-value shooting method, and saves every valid configuration.

### Minimal usage

```bash
python crt_gt_gen_output_file.py --n 200 --outfile tdcr_gt_samples.npz
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 10 | Number of valid configurations to generate |
| `--outfile` | `tdcr_gt_samples_10.npz` | Output filename |
| `--seed` | 22 | RNG seed for reproducibility |
| `--length` | 0.10 | Backbone length (m) |
| `--backbone-radius` | 5e-4 | Backbone cross-section radius (m) |
| `--youngs-modulus` | 60e9 | Young's modulus (Pa) |
| `--tendon-offset` | 0.008 | Radial distance of tendons from backbone center (m) |
| `--num-disks` | 40 | Discretisation points along the backbone |
| `--tau-max` | 4.0 | Maximum cable tension for sampling (N) |
| `--inextensible` | True | Use Kirchhoff (inextensible) rod model |
| `--wrench-mode` | `full` | `full` / `force-only` / `moment-only` / `zero` |
| `--force-range` | 0.4 | Uniform half-range for tip force sampling (N) |
| `--moment-range` | 0.04 | Uniform half-range for tip moment sampling (N·m) |
| `--active-cables` | None | Fix which cable indices are active, e.g. `0,1` |
| `--max-tries` | 1000 | Max solver attempts before giving up |

### Output file contents (`tdcr_gt_samples.npz`)

| Array | Shape | Description |
|---|---|---|
| `tau` | `(N, 4)` | Cable tensions for each sample (N) |
| `f_ext` | `(N, 3)` | Tip forces (N) |
| `l_ext` | `(N, 3)` | Tip moments (N·m) |
| `T` | `(N, n_disks, 4, 4)` | Homogeneous transforms along the backbone |
| `meta` | JSON string | Rod parameters, date, git hash, seed |

Reading the file:
```python
import numpy as np, json
data = np.load("tdcr_gt_samples.npz", allow_pickle=True)
tau  = data["tau"]          # (N, 4)
T    = data["T"]            # (N, n_disks, 4, 4)
meta = json.loads(data["meta"].item())
```

### Common recipes

```bash
# Large dataset, no external wrench
python crt_gt_gen_output_file.py --n 500 --wrench-mode zero --outfile gt_no_wrench.npz

# In-plane bending only (cable 0 and 1), force in xz-plane
python crt_gt_gen_output_file.py --n 200 --inplane --active-cables 0,1 \
    --wrench-plane xz --outfile gt_inplane.npz

# Fix the tip force and vary only cable tensions
python crt_gt_gen_output_file.py --n 100 --force "0.1,0.0,0.05" \
    --moment "0,0,0" --outfile gt_fixed_force.npz

# Physically scaled wrench amplitudes (based on rod stiffness EI/L)
python crt_gt_gen_output_file.py --n 300 --use-scaled-wrench \
    --c-f 0.5 --c-m 0.5 --outfile gt_scaled.npz
```

---

## Step 2 — Generate noisy sensor measurements

**Script:** `crt_meas_gen_output_file.py`

Reads the ground-truth `.npz` file and adds Gaussian noise to simulate
real sensor readings: cable force sensors and IMU orientation sensors.

### Minimal usage

```bash
python crt_meas_gen_output_file.py \
    --gt tdcr_gt_samples.npz \
    --out tdcr_meas_samples.npz
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--gt` | `tdcr_gt_samples_10.npz` | Input ground-truth file |
| `--out` | `tdcr_meas_samples_10.npz` | Output measurement file |
| `--tau-noise` | 0.02 | Cable tension noise std dev (N) |
| `--ori-noise` | 0.01745 rad (~1°) | IMU orientation noise std dev (rad) |
| `--seed` | None | RNG seed |

**Sensor locations** are hard-coded to normalised arc-lengths
`s = {12/39, 25/39, 1.00}` (approximately 0.31, 0.64, 1.00) matching
3 IMUs at roughly equal spacing. Edit `s_locations` in the script to change.

### Noise model

- **Cable tension:** `tau_meas = tau_GT + N(0, sigma_tau^2)`
- **Orientation:** `q_meas = q_noise ⊗ q_GT`, where `q_noise` is a small
  random rotation with axis drawn uniformly on S² and angle ~ `N(0, sigma_theta^2)`

### Output file contents (`tdcr_meas_samples.npz`)

| Array | Shape | Description |
|---|---|---|
| `tau_meas` | `(N, 4)` | Noisy cable tensions (N) |
| `q_meas` | `(N, 3, 4)` | Noisy quaternions at each IMU location (w, x, y, z) |
| `meta` | JSON string | Noise parameters, sensor locations, date |

Reading the file:
```python
import numpy as np, json
data   = np.load("tdcr_meas_samples.npz", allow_pickle=True)
tau_m  = data["tau_meas"]   # (N, 4)
q_meas = data["q_meas"]     # (N, 3, 4) — scalar-first quaternion
meta   = json.loads(data["meta"].item())
```

### Full two-step example

```bash
# Step 1: 500 samples, reproducible, no external wrench
python crt_gt_gen_output_file.py \
    --n 500 --seed 42 --wrench-mode zero \
    --outfile gt_500.npz

# Step 2: add sensor noise
python crt_meas_gen_output_file.py \
    --gt gt_500.npz --out meas_500.npz \
    --tau-noise 0.02 --ori-noise 0.0175 --seed 99
```

---

## Visualisation / quick checks

These scripts generate a few random shapes and display a 3-D plot.
They do **not** save any files. Useful for sanity-checking the rod model.

```bash
# Basic: plot backbone curves only
python crt_gt_gen.py

# With tip coordinate frames drawn as RGB axes
python crt_gt_gen_plot.py
```

Both scripts use hard-coded parameters at the top of `main()`.
Edit `n_target`, `τ_max`, `length`, etc. directly in the file if needed.

---

## Core library files (not run directly)

### `cosserat_rod_model.py` — `CosseratRodModel`

The rod physics. Constructs and solves the Cosserat rod BVP via shooting.

```python
from cosserat_rod_model import CosseratRodModel

rod = CosseratRodModel(
    length=0.10,          # m
    backbone_radius=5e-4, # m
    youngs_modulus=60e9,  # Pa
    num_disks=40,
    inextensible=True,    # Kirchhoff (no shear/stretch)
)

# Returns 4x4 tip transform; optionally the full trajectory
T_tip, traj = rod.forward_kinematics(
    tau=np.array([2.0, 0.0, 0.0, 0.0]),  # cable tensions (N)
    f_ext=np.zeros(3),
    l_ext=np.zeros(3),
    return_states=True,
)
# traj.y shape: (18, n_pts)
# rows 0:3  = position p(s)
# rows 3:12 = rotation matrix R(s) flattened
```

### `utils.py` — math helpers

```python
from utils import hat, vec, unit
hat([1,2,3])   # 3x3 skew-symmetric matrix
vec(S)         # inverse of hat
unit(v)        # normalise vector
```

### `virtual_work.py` — modal kinematics

Implements polynomial modal parameterisation of curvature
(κ(s) = Φ(s) m) and the virtual-work cable Jacobian used in
optimisation-based state estimation. Not needed for data generation;
used by the EKF / shape estimation pipeline.

---

## Archive

`archive/` contains early exploratory scripts comparing computation
times of different rod formulations. They are not part of the current
workflow and are kept only for reference.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cosserat_rod_model'`**
Run the scripts from inside the `crt/` directory. These scripts use
local imports.

```bash
cd src/shape_force_est_imu/crt
python crt_gt_gen_output_file.py ...
```

**`RuntimeError: Shooting failed`**
The BVP solver did not converge for a particular sample. The script
automatically discards it and draws a new sample. If failure rate is
high, try reducing `--tau-max` or `--force-range`, or increase
`--max-tries`.

**Slow generation**
Each sample requires solving a nonlinear BVP. Expect roughly 0.1–0.5 s
per sample depending on hardware. Use `--num-disks 20` for faster (less
smooth) results during development.
