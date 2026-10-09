"""Finite-difference regression for run_ekf_for_order's measurement Jacobian.

Uses fixed synthetic orientations, additive modal perturbations, and no datasets.
This checks the general-order implementation for 5D and 8D states; it does not
exercise the separate five-dimensional run_ekf_on_frame implementation.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
CRT_DIR = ROOT / "src" / "shape_force_est_imu" / "crt"
sys.path.insert(0, str(CRT_DIR))
import compare_oracle_vs_ekf_wrench_estimation as pipeline  # noqa: E402


POSITIONS = np.array([25.0 / 99.0, 50.0 / 99.0, 1.0])
GAMMA = 10
FD_STEP = 1e-6
MAX_ABS_TOL = 2e-7
RELATIVE_FROBENIUS_TOL = 1e-6


def capture_measurement_function(observations, orders):
    """Access the production closure without changing its implementation."""
    captured = {}
    target_code = pipeline.run_ekf_for_order.__code__

    def trace(frame, event, arg):
        if frame.f_code is target_code and event == "return":
            captured["measurement"] = frame.f_locals["H_and_r_general"]
        return trace

    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        pipeline.run_ekf_for_order(
            observations, POSITIONS, *orders, gamma=GAMMA,
            meas_std_deg=0.5, alpha=1.0, steps=0,
        )
    finally:
        sys.settrace(previous_trace)
    if "measurement" not in captured:
        raise AssertionError("Could not capture the production measurement closure")
    return captured["measurement"]


def fixed_observations(orders, reference_state):
    """Observations are fixed before evaluating any coefficient perturbations."""
    fixed_left_errors = np.array([
        [0.015, -0.010, 0.005],
        [-0.008, 0.012, 0.006],
        [0.010, 0.004, -0.013],
    ])
    result = []
    for s, error in zip(POSITIONS, fixed_left_errors):
        reference_rotation = pipeline.fwd_transform_general(
            reference_state, float(s), GAMMA, 0.1, *orders,
        )[:3, :3]
        result.append(Rotation.from_rotvec(error).as_matrix() @ reference_rotation)
    return result


def independent_residual(state, observations, orders):
    """Compute the same SO(3) convention without calling the analytic H code."""
    blocks = []
    for s, observed in zip(POSITIONS, observations):
        predicted = pipeline.fwd_transform_general(
            state, float(s), GAMMA, 0.1, *orders,
        )[:3, :3]
        blocks.append(Rotation.from_matrix(observed @ predicted.T).as_rotvec())
    return np.concatenate(blocks)


def run_checks():
    cases = [
        ((1, 1, 0), np.array([0.3, 0.1, -0.25, 0.15, 0.12]), [
            np.array([0.4, -0.2, 0.3, 0.1, 0.05]),
            np.array([-0.2, 0.3, -0.1, 0.25, -0.08]),
        ]),
        ((2, 2, 1), np.array([0.3, 0.1, -0.02, -0.25, 0.15, 0.03, 0.12, -0.04]), [
            np.array([0.4, -0.2, 0.06, 0.3, 0.1, -0.04, 0.05, 0.02]),
            np.array([-0.3, 0.1, 0.07, 0.2, -0.15, 0.03, -0.08, 0.04]),
        ]),
    ]
    results = []
    for orders, reference_state, states in cases:
        observations = fixed_observations(orders, reference_state)
        measurement = capture_measurement_function(observations, orders)
        for state_index, state in enumerate(states):
            analytic, production_residual = measurement(state)
            expected_residual = independent_residual(state, observations, orders)
            np.testing.assert_allclose(production_residual, expected_residual,
                                       rtol=1e-11, atol=1e-12)
            numeric = np.column_stack([
                (independent_residual(state + FD_STEP * e, observations, orders)
                 - independent_residual(state - FD_STEP * e, observations, orders))
                / (2.0 * FD_STEP) for e in np.eye(state.size)
            ])
            assert analytic.shape == numeric.shape == (9, state.size)
            assert np.all(np.isfinite(analytic)) and np.all(np.isfinite(numeric))
            maximum = float(np.max(np.abs(analytic - numeric)))
            relative = float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))
            result = {
                "orders": list(orders), "state_index": state_index,
                "state": state.tolist(), "reference_state": reference_state.tolist(),
                "max_abs_difference": maximum,
                "relative_frobenius_difference": relative,
                "passed": maximum < MAX_ABS_TOL and relative < RELATIVE_FROBENIUS_TOL,
            }
            results.append(result)
            print(f"[general EKF H FD] order={orders}, state={state_index}: "
                  f"max_abs={maximum:.3e}, relative_fro={relative:.3e}, "
                  f"passed={result['passed']}")
    return results


def assert_checks(results):
    failures = [row for row in results if not row["passed"]]
    if failures:
        raise AssertionError(
            f"{len(failures)} of {len(results)} measurement Jacobians failed: "
            f"require max_abs < {MAX_ABS_TOL} and relative_fro < {RELATIVE_FROBENIUS_TOL}"
        )


def main():
    assert_checks(run_checks())
    print("All general-order EKF measurement-Jacobian checks passed.")


if __name__ == "__main__":
    main()
