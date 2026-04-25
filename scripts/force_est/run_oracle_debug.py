"""
Thin runner wrapper for force_est_curve_fit.py.
Patches plt.show() to save figures to debug_output/ before displaying.
Does NOT modify the original script.
"""
import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_save_dir = os.path.join(os.path.dirname(__file__), "debug_output")
os.makedirs(_save_dir, exist_ok=True)
_run_label = os.environ.get("ORACLE_RUN_LABEL", "oracle")
_fig_counter = [0]

_orig_show = plt.show
def _patched_show(*args, **kwargs):
    fname = os.path.join(_save_dir, f"{_run_label}_fig_{_fig_counter[0]:02d}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[runner] saved figure -> {fname}")
    _fig_counter[0] += 1
    plt.close("all")

plt.show = _patched_show

# Now run the real main
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "force_est_curve_fit",
    os.path.join(os.path.dirname(__file__), "force_est_curve_fit.py")
)
_target = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_target)
_target.main()
