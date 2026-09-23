#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# Throwaway: drive scripts/serve_harness.main with check.py instead of battery.

"""Run baseline constrained grammar smoke via real serve_harness lifecycle.

Imports scripts/serve_harness.py as a module, replaces ``run(cfg, args)`` with a
subprocess to sibling ``check.py``, returns ``[]`` so harness ``main`` still owns
CLI parse, config resolve, serve spawn, path proofs, and cleanup.

Example (parent holds gpu-lock; this process does not acquire it):

  cd /home/kaden/ClaudeCode/warpfront/hipfire-grammar-experiment
  HIP_VISIBLE_DEVICES=3 HIPFIRE_QWEN35_GRAMMAR=1 HIPFIRE_DFLASH_DRAFT= \\
  HIPFIRE_CLI_BIN=…/hipfire HIPFIRE_DAEMON_BIN=…/daemon \\
  python3 .codeinsight+research/grammar-serving/run.py \\
    --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt \\
    --tag qwen3.8:27b-mq4-xt \\
    --speculation off --dflash off --mtp off --ngram off \\
    --kv q8 --kv-backend contiguous \\
    --max-seq 4096 --max-tokens 256 \\
    --thinking off --thinking-effort none --sampling greedy \\
    --port 11520 --devices 3 \\
    --home /tmp/grammar-serving-home \\
    --serve-log /tmp/grammar-serving.serve.log
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # …/hipfire-grammar-experiment
HARNESS_PATH = REPO / "scripts" / "serve_harness.py"
CHECK_PY = HERE / "check.py"
RESPONSE_JSON = HERE / "response.json"


def _load_serve_harness():
    if not HARNESS_PATH.is_file():
        sys.exit(f"run.py: missing serve_harness at {HARNESS_PATH}")
    spec = importlib.util.spec_from_file_location("serve_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        sys.exit(f"run.py: cannot load module spec for {HARNESS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses/self-refs see the name if needed.
    sys.modules["serve_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_run_override(sh):
    def run(cfg, args):
        """Replace battery/session driver with grammar check.py smoke."""
        port = cfg.get("port", 11520)
        model = cfg["model"]
        base_url = f"http://127.0.0.1:{port}"
        cmd = [
            sys.executable,
            str(CHECK_PY),
            "--base-url",
            base_url,
            "--model",
            str(model),
            "--out",
            str(RESPONSE_JSON),
        ]
        label = f"{Path(str(model)).name}|grammar-check"
        print(f"### RUN {label} via check.py ###", flush=True)
        print(f"  cmd={' '.join(cmd)}", flush=True)
        # check=True → nonzero check.py exit becomes CalledProcessError → nonzero here.
        subprocess.run(cmd, check=True)
        print(f"[{label} DONE] check.py passed; retained {RESPONSE_JSON}", flush=True)
        # Empty rows: harness main still runs dflash request proofs (no-op when
        # dflash!=on) and retrieval_missing scan, then kills serve.
        return []

    return run


def main(argv: list[str] | None = None) -> None:
    if not CHECK_PY.is_file():
        sys.exit(f"run.py: missing check.py at {CHECK_PY}")
    # Forward CLI to harness.main (argparse uses sys.argv).
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    sh = _load_serve_harness()
    sh.run = _make_run_override(sh)
    sh.main()


if __name__ == "__main__":
    main()
