#!/usr/bin/env python
"""
Print reproducible PowerShell commands for medium and full wrench studies.

This helper prints commands only. It does not create directories, execute
studies, or write result files.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CRT = "src/shape_force_est_imu/crt"
FORCE_EST = "scripts/force_est"


def _git_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path.cwd()


def _ps_quote(text: str) -> str:
    if "$env:" in text:
        return '"' + text.replace("`", "``").replace('"', '`"') + '"'
    return "'" + text.replace("'", "''") + "'"


def _slash(path: str) -> str:
    return path.replace("\\", "/")


def _join(root: str, child: str) -> str:
    return _slash(str(PurePathText(root) / child))


class PurePathText:
    """Small path join helper that preserves PowerShell env-var prefixes."""

    def __init__(self, text: str):
        self.text = text.rstrip("\\/")

    def __truediv__(self, child: str) -> "PurePathText":
        return PurePathText(self.text + "/" + child.strip("\\/"))

    def __str__(self) -> str:
        return self.text


def _cmd(python_cmd: str, script: str, args: list[str]) -> str:
    parts = [python_cmd, _ps_quote(_slash(script))]
    parts.extend(args)
    return " ".join(parts)


def _arg(name: str, value: str | int | float) -> list[str]:
    return [name, _ps_quote(str(value)) if isinstance(value, str) else str(value)]


def _print_block(title: str, description: str, command: str) -> None:
    print(f"# {title}")
    print(f"# {description}")
    print(command)
    print()


def build_commands(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    repo_root = Path(args.repo_root).resolve()
    out_root = _slash(str((repo_root / args.out_root).resolve()) if not Path(args.out_root).is_absolute() else args.out_root)
    temp_root = _slash(args.temp_root)
    py = args.python

    constrained = f"{CRT}/evaluate_kirchhoff_wrench_constrained.py"
    observability = f"{CRT}/diagnose_wrench_observability_vs_order.py"
    compare = f"{CRT}/compare_oracle_vs_ekf_wrench_estimation.py"
    ekf_sweep = f"{CRT}/sweep_ekf_wrench_convergence.py"
    continuation = f"{CRT}/diagnose_modal_order_continuation.py"
    artifacts = f"{FORCE_EST}/generate_final_wrench_artifacts.py"

    obs_orders = "1,1,0;1,2,0;2,2,0;2,2,1;3,3,2"
    modal_orders = "(1,1,0);(1,2,0);(2,2,0);(2,2,1);(3,3,2)"
    compare_orders = ["(1,1,0)", "(1,2,0)", "(2,2,0)", "(2,2,1)"]
    compare_order_args = ["--orders", *[_ps_quote(o) for o in compare_orders]]

    return [
        (
            "1. Medium constrained force-only/contact-direction study",
            "Small constrained force-only run for checking force-only and oracle-direction diagnostics.",
            _cmd(
                py,
                constrained,
                [
                    "--cases",
                    "force_only",
                    "--max-cases",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--steps",
                    str(args.medium_steps),
                    "--save-dir",
                    _ps_quote(_join(temp_root, "constrained_force_only_medium")),
                ],
            ),
        ),
        (
            "2. Full constrained study",
            "Full constrained wrench study with updated tendon-pull convention.",
            _cmd(
                py,
                constrained,
                [
                    "--cases",
                    "force_only",
                    "transverse_force",
                    "moment_only",
                    "--save-dir",
                    _ps_quote(_join(out_root, "constrained_wrench_updated")),
                ],
            ),
        ),
        (
            "3. Medium observability component/subspace study",
            "Medium observability run including component and load-subspace diagnostics.",
            _cmd(
                py,
                observability,
                [
                    "--orders",
                    _ps_quote(obs_orders),
                    "--max-cases",
                    str(args.medium_cases),
                    "--save-dir",
                    _ps_quote(_join(temp_root, "observability_medium")),
                ],
            ),
        ),
        (
            "4. Full observability component/subspace study",
            "Full observability diagnostics for selected modal orders.",
            _cmd(
                py,
                observability,
                [
                    "--orders",
                    _ps_quote(obs_orders),
                    "--save-dir",
                    _ps_quote(_join(out_root, "wrench_observability_updated")),
                ],
            ),
        ),
        (
            "5. Medium oracle-vs-EKF with sparse-batch oracle",
            "Medium oracle/EKF comparison. This CLI takes separate order arguments.",
            _cmd(
                py,
                compare,
                [
                    *compare_order_args,
                    "--max-cases",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--steps",
                    str(args.medium_steps),
                    "--enable-sparse-batch-oracle",
                    "--batch-max-iter",
                    "50",
                    "--batch-multistart",
                    "1",
                    "--save-dir",
                    _ps_quote(_join(temp_root, "oracle_vs_ekf_sparse_batch_medium")),
                ],
            ),
        ),
        (
            "6. Full oracle-vs-EKF study",
            "Full oracle/EKF comparison with sparse-batch oracle enabled.",
            _cmd(
                py,
                compare,
                [
                    *compare_order_args,
                    "--steps",
                    str(args.medium_steps),
                    "--enable-sparse-batch-oracle",
                    "--batch-max-iter",
                    "50",
                    "--batch-multistart",
                    "1",
                    "--save-dir",
                    _ps_quote(_join(out_root, "oracle_vs_ekf_sparse_batch_updated")),
                ],
            ),
        ),
        (
            "7. Medium EKF convergence sweep",
            "Medium sweep over repeated EKF updates, alpha, P0 scale, and Q scale.",
            _cmd(
                py,
                ekf_sweep,
                [
                    "--orders",
                    _ps_quote("(1,1,0)"),
                    "--max-cases",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--layouts",
                    "3-IMU",
                    "--steps-list",
                    _ps_quote("1,5,20,50"),
                    "--alpha-list",
                    _ps_quote("0.5,1.0,2.0"),
                    "--p0-scale-list",
                    _ps_quote("0.1,1.0,10.0"),
                    "--q-scale-list",
                    _ps_quote("0.1,1.0,10.0"),
                    "--init-mode",
                    "zero",
                    "--save-dir",
                    _ps_quote(_join(temp_root, "ekf_convergence_medium")),
                ],
            ),
        ),
        (
            "8. Medium sparse-batch-init EKF convergence sweep",
            "Medium EKF convergence check initialized from sparse-batch fits.",
            _cmd(
                py,
                ekf_sweep,
                [
                    "--orders",
                    _ps_quote("(1,1,0)"),
                    "--max-cases",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--layouts",
                    "3-IMU",
                    "--steps-list",
                    _ps_quote("1,5,20"),
                    "--alpha-list",
                    _ps_quote("1.0"),
                    "--p0-scale-list",
                    _ps_quote("1.0"),
                    "--q-scale-list",
                    _ps_quote("1.0"),
                    "--init-mode",
                    "sparse_batch",
                    "--batch-max-iter",
                    "50",
                    "--batch-multistart",
                    "1",
                    "--save-dir",
                    _ps_quote(_join(temp_root, "ekf_convergence_sparse_init_medium")),
                ],
            ),
        ),
        (
            "9. Medium regularized modal continuation",
            "Medium lower-to-higher modal continuation with regularization grid.",
            _cmd(
                py,
                continuation,
                [
                    "--orders",
                    _ps_quote(modal_orders),
                    "--max-cases",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--layout",
                    "3-IMU",
                    "--mode",
                    "batch",
                    "--batch-max-iter",
                    "50",
                    "--batch-multistart",
                    "1",
                    "--regularization-mode",
                    "both",
                    "--prior-weight-list",
                    _ps_quote("0.0,1e-3,1e-2,1e-1,1.0"),
                    "--new-coeff-weight-list",
                    _ps_quote("0.0,1e-3,1e-2,1e-1,1.0"),
                    "--save-dir",
                    _ps_quote(_join(temp_root, "continuation_regularized_medium")),
                ],
            ),
        ),
        (
            "10. Final artifact dry-run",
            "Dry-run only: prints artifact inputs/outputs without writing files.",
            _cmd(
                py,
                artifacts,
                [
                    "--dry-run",
                    "--out-dir",
                    _ps_quote(_join(out_root, "final_wrench_updated")),
                    "--n-cases-ekf",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--max-cases-oracle",
                    str(args.medium_cases),
                    "--max-cases-constrained",
                    str(args.medium_cases),
                ],
            ),
        ),
        (
            "11. Medium final artifact generation",
            "Medium artifact generation into temp-root, capped by cases/noise.",
            _cmd(
                py,
                artifacts,
                [
                    "--out-dir",
                    _ps_quote(_join(temp_root, "final_wrench_medium")),
                    "--n-cases-ekf",
                    str(args.medium_cases),
                    "--n-noise",
                    str(args.medium_noise),
                    "--max-cases-oracle",
                    str(args.medium_cases),
                    "--max-cases-constrained",
                    str(args.medium_cases),
                ],
            ),
        ),
        (
            "12. Full final artifact generation",
            "Full final artifact generation using current updated convention outputs.",
            _cmd(
                py,
                artifacts,
                [
                    "--out-dir",
                    _ps_quote(_join(out_root, "final_wrench_updated")),
                ],
            ),
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    root = _git_root()
    parser = argparse.ArgumentParser(
        description="Print PowerShell commands for medium and full wrench study runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=str(root))
    parser.add_argument("--out-root", default="src/shape_force_est_imu/crt/gt_data/results")
    parser.add_argument("--temp-root", default="$env:TEMP\\shape-force-est-IMU-medium")
    parser.add_argument("--medium-cases", type=int, default=10)
    parser.add_argument("--medium-noise", type=int, default=3)
    parser.add_argument("--medium-steps", type=int, default=20)
    parser.add_argument("--python", default="python -B")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.medium_cases <= 0:
        parser.error("--medium-cases must be > 0")
    if args.medium_noise <= 0:
        parser.error("--medium-noise must be > 0")
    if args.medium_steps <= 0:
        parser.error("--medium-steps must be > 0")

    print("# Wrench study command plan")
    print("# These commands are printed only; this helper does not execute them or create directories.")
    print("# Run from the repository root in PowerShell.")
    print("# WARNING: old final CSVs are stale after the tendon-pull convention and tendon-tangent scaling patch.")
    print()
    print(f"Set-Location {_ps_quote(_slash(str(Path(args.repo_root).resolve())))}")
    print()

    for title, description, command in build_commands(args):
        _print_block(title, description, command)


if __name__ == "__main__":
    main()
