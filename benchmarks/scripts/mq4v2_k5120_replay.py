#!/usr/bin/env python3
"""Run the official replay harness with graceful termination cleanup."""
from pathlib import Path
import runpy
import signal


def stop(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/redline_daemon_harness.py"), run_name="__main__")
