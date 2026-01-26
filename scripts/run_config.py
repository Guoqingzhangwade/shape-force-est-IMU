#!/usr/bin/env python3
"""
Run GT generation and estimation from JSON configs.

Usage:
  python scripts/run_config.py run-gt [--gt path]
  python scripts/run_config.py run-est [--est path] [--gt-outfile path]
  python scripts/run_config.py run-both [--gt path] [--est path]
  python scripts/run_config.py sweep-est --scales 0.5,0.8,1.0,1.2,1.5
"""
from __future__ import annotations
import json
import os
import subprocess
import sys


DEFAULT_GT = os.path.join(os.path.dirname(__file__), "config", "gt_config.json")
DEFAULT_EST = os.path.join(os.path.dirname(__file__), "config", "est_config.json")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def arg_key_to_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_command(cfg: dict, substitutions: dict) -> str:
    script = cfg.get("script")
    args = cfg.get("args", {})
    if not script:
        raise ValueError("Config missing 'script'")
    parts = ["python", script]
    for key, value in args.items():
        flag = arg_key_to_flag(key)
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        value = str(value).format(**substitutions)
        parts.extend([flag, value])
    return " ".join(parts)


def run_command(command: str) -> int:
    print(f"Running: {command}")
    return subprocess.call(command, shell=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_config.py run-gt|run-est|run-both")
        return 2

    cmd = sys.argv[1].lower()
    gt_path = DEFAULT_GT
    est_path = DEFAULT_EST
    gt_outfile = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--gt" and i + 1 < len(sys.argv):
            gt_path = sys.argv[i + 1]
            i += 2
            continue
        if sys.argv[i] == "--est" and i + 1 < len(sys.argv):
            est_path = sys.argv[i + 1]
            i += 2
            continue
        if sys.argv[i] == "--gt-outfile" and i + 1 < len(sys.argv):
            gt_outfile = sys.argv[i + 1]
            i += 2
            continue
        i += 1

    gt_cfg = load_json(gt_path)
    est_cfg = load_json(est_path)
    gt_out = gt_cfg.get("args", {}).get("outfile")
    if gt_outfile:
        gt_out = gt_outfile

    substitutions = {"gt_outfile": gt_out} if gt_out else {}

    if cmd == "run-gt":
        return run_command(build_command(gt_cfg, substitutions))
    if cmd == "run-est":
        return run_command(build_command(est_cfg, substitutions))
    if cmd == "run-both":
        code = run_command(build_command(gt_cfg, substitutions))
        if code != 0:
            return code
        return run_command(build_command(est_cfg, substitutions))
    if cmd == "sweep-est":
        scales = []
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--scales" and i + 1 < len(sys.argv):
                scales = [float(x) for x in sys.argv[i + 1].split(",") if x]
                i += 2
                continue
            i += 1
        if not scales:
            print("Provide --scales, e.g., 0.5,0.8,1.0,1.2,1.5")
            return 2
        base_E = est_cfg.get("args", {}).get("E")
        if base_E is None:
            print("est_config.json missing args.E for sweep")
            return 2
        for s in scales:
            est_cfg_s = json.loads(json.dumps(est_cfg))
            est_cfg_s["args"]["E"] = float(base_E) * s
            substitutions["scale"] = s
            print(f"\n--- sweep scale {s} (E={est_cfg_s['args']['E']}) ---")
            code = run_command(build_command(est_cfg_s, substitutions))
            if code != 0:
                return code
        return 0

    print("Unknown command. Use: run-gt | run-est | run-both | sweep-est")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
