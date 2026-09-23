#!/usr/bin/env python3
"""Run candidate WT2 c24 with the pinned legacy-KV evaluator contract."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch-g12gate" / "wt2"
OUT.mkdir(parents=True, exist_ok=True)
HOME = Path("/home/kaden/.hipfire-homes/g12gate-wt2")
HOME.mkdir(parents=True, exist_ok=True)
(HOME / "daemon.pid").unlink(missing_ok=True)
DEVICE = "GPU-6109a4cb5f833235"
EVALUATOR = ROOT / "target/release/examples/eval_hipfire"
MODEL = "/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq"
REF = "/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin"
SEQ = OUT / "candidate.kldseq"
ENV = {key: value for key, value in os.environ.items() if not key.startswith("HIPFIRE_")}
ENV.update(HOME=str(HOME), HIPFIRE_KERNEL_CACHE=str(HOME / ".hipfire_kernels"),
           HIPFIRE_MODELS_DIR="/home/kaden/.hipfire/models", HIPFIRE_GRAPH="0",
           HIPFIRE_NORMALIZE_PROMPT="0", ROCR_VISIBLE_DEVICES=DEVICE,
           HIP_VISIBLE_DEVICES=DEVICE)
COMMAND = [str(EVALUATOR), "--model", MODEL, "--ref", REF, "--kv-mode", "fp8",
           "--kv-v", "q8", "--scoring-mode", "prefill", "--max-chunks", "24",
           "--output", str(SEQ)]
print("WT2 card-D starting", flush=True)
proc = subprocess.run(COMMAND, cwd=ROOT, env=ENV, text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(OUT / "candidate.log").write_text(proc.stdout)
if proc.returncode or "GPU dev 0: gfx1201" not in proc.stdout or "KV cache: fp8-e4m3 backend=legacy" not in proc.stdout:
    raise RuntimeError(f"WT2 exit={proc.returncode}, gfx1201/legacy KV not confirmed; see candidate.log")
match = re.search(r"slice-mean KLD = ([0-9.]+)", proc.stdout)
if not match:
    raise RuntimeError("WT2 did not print slice-mean KLD")
digest = hashlib.md5(SEQ.read_bytes()).hexdigest()
result = {"kld": float(match.group(1)), "md5": digest, "expected_md5": "9d0e860f41db992820ebdc9483c0a041",
          "evaluator_md5": hashlib.md5(EVALUATOR.read_bytes()).hexdigest(),
          "device": DEVICE, "command": COMMAND}
(OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
if digest != result["expected_md5"]:
    raise SystemExit("WT2 sequence mismatch")
