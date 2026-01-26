#!/usr/bin/env python3
"""
Run named tasks from run_tasks.json.

Usage:
  python scripts/run_tasks.py list
  python scripts/run_tasks.py run <task_name>
"""
from __future__ import annotations
import json
import os
import subprocess
import sys


def load_tasks(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError("Invalid config: expected top-level 'tasks' dict")
    return data["tasks"]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_tasks.py list|run <task_name>")
        return 2

    cmd = sys.argv[1].lower()
    config_path = os.path.join(os.path.dirname(__file__), "run_tasks.json")
    tasks = load_tasks(config_path)

    if cmd == "list":
        for name in sorted(tasks.keys()):
            entry = tasks[name]
            desc = entry.get("desc", "")
            print(f"{name}: {desc}")
        return 0

    if cmd == "run":
        if len(sys.argv) < 3:
            print("Usage: python scripts/run_tasks.py run <task_name>")
            return 2
        name = sys.argv[2]
        if name not in tasks:
            print(f"Unknown task: {name}")
            return 1
        entry = tasks[name]
        command = entry.get("command")
        if not command:
            print(f"Task '{name}' has no command")
            return 1
        print(f"Running: {command}")
        return subprocess.call(command, shell=True)

    print("Unknown command. Use: list | run <task_name>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
