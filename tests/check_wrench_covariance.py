from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CRT_DIR = ROOT / "src" / "shape_force_est_imu" / "crt"
sys.path.insert(0, str(CRT_DIR))

import wrench_study_utils as wsu  # noqa: E402


def check_generalized_load_covariance():
    order_cfg = wsu.OrderConfig(1, 1, 0)
    n_params = 5
    m = np.array([0.4, -0.2, 0.3, 0.1, 0.05])
    tau = np.array([5.0, 4.5, 4.0, 4.2])
    P_modal = np.diag([1e-4, 1e-5, 1e-4, 1e-5, 1e-6])
    R_tau = (0.02**2) * np.eye(len(tau))

    Sigma_b = wsu.propagate_generalized_load_covariance(
        m, tau, P_modal, R_tau, order_cfg
    )

    assert Sigma_b.shape == (n_params, n_params)
    np.testing.assert_allclose(Sigma_b, Sigma_b.T, rtol=0.0, atol=1e-12)
    min_eig = float(np.min(np.linalg.eigvalsh(Sigma_b)))
    assert min_eig > -1e-10

    Sigma_model = 1e-8 * np.eye(n_params)
    Sigma_b2 = wsu.propagate_generalized_load_covariance(
        m,
        tau,
        P_modal,
        R_tau,
        order_cfg,
        Sigma_b_model=Sigma_model,
    )
    assert float(np.trace(Sigma_b2)) > float(np.trace(Sigma_b))

    print(f"[Sigma_b] shape={Sigma_b.shape}")
    print(f"[Sigma_b] min_eig={min_eig:.3e}")
    print(f"[Sigma_b] trace={np.trace(Sigma_b):.3e}")


if __name__ == "__main__":
    check_generalized_load_covariance()
    print("Generalized-load covariance checks passed.")
