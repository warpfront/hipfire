#!/usr/bin/env python3
"""Fixture-bound gfx1201 KV-default performance guard (card-B only).

One discarded fresh-process warmup precedes three independent matrix processes.
Each measured process runs three resident samples; neither is a substitute for
another process. Artifacts are local evidence, never a portable benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "GPU-e475645fe0200397"
MODEL = Path("/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq")
MODEL_SIZE = 14_987_185_152
MODEL_MD5 = "2cfe88923b3671ca16a8de6ec1122fde"
FLOOR = 3620.0
REFERENCE_DECODE = 36.5


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-backend", choices=("legacy",), help="explicit legacy comparator")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scratch-vmmdefault" / "guard")
    parser.add_argument("--home", type=Path, default=Path("/home/kaden/.hipfire-homes/vmmdefault"))
    args = parser.parse_args()
    cli = ROOT / "target/release/hipfire"
    daemon = ROOT / "target/release/daemon"
    if MODEL.stat().st_size != MODEL_SIZE:
        parser.error(f"model size differs from pinned fixture ({MODEL_SIZE})")
    for binary in (cli, daemon):
        if not binary.is_file():
            parser.error(f"missing release binary: {binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.home.mkdir(parents=True, exist_ok=True)
    (args.home / "daemon.pid").unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(HOME=str(args.home), ROCR_VISIBLE_DEVICES=DEVICE,
               HIP_VISIBLE_DEVICES=DEVICE, HIPFIRE_KERNEL_CACHE=str(args.home / ".hipfire_kernels"),
               HIPFIRE_MODELS_DIR="/home/kaden/.hipfire/models", HIPFIRE_GRAPH="1",
               HIPFIRE_DAEMON_BIN=str(daemon))
    # Matrix bench_prefill constructs exactly these u32 ids (little-endian
    # serialization is ours, not a textual CLI prompt or the unrelated prompt_md5).
    stream_md5 = hashlib.md5(b"".join(struct.pack("<I", 10 + i % 1000) for i in range(8192))).hexdigest()
    fixture = dict(device=DEVICE, model=str(MODEL), model_size=MODEL_SIZE, model_md5=MODEL_MD5,
                   cli_md5=md5(cli), daemon_md5=md5(daemon),
                   token_stream="8192 little-endian u32 tokens: 10 + (i % 1000), i=0..8191",
                   token_stream_md5=stream_md5, kv_backend=args.kv_backend or "automatic",
                   environment={key: env[key] for key in ("HOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                       "HIPFIRE_KERNEL_CACHE", "HIPFIRE_MODELS_DIR", "HIPFIRE_GRAPH", "HIPFIRE_DAEMON_BIN")})
    (args.output_dir / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    medians = []
    decodes = []
    for run in range(4):
        warmup = run == 0
        command = [str(cli), "bench", str(MODEL), "--matrix", "--pp", "512,8192",
                   "--ctx", "128", "--tg", "128", "--spec", "off", "--runs",
                   "1" if warmup else "3", "--warmups", "1", "--kv-mode", "fp8", "--json"]
        if args.kv_backend:
            command += ["--kv-backend", args.kv_backend]
        suffix = "warmup" if warmup else f"process-{run}"
        (args.output_dir / f"{suffix}.command.json").write_text(json.dumps(command) + "\n")
        with (args.output_dir / f"{suffix}.daemon.log").open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=log, text=True, check=False)
        (args.output_dir / f"{suffix}.json").write_text(proc.stdout)
        text = (args.output_dir / f"{suffix}.daemon.log").read_text(errors="replace")
        if proc.returncode:
            raise RuntimeError(f"{suffix}: bench exited {proc.returncode}; see {suffix}.daemon.log")
        if args.kv_backend:
            if "WARNING HIPFIRE_KV_BACKEND=legacy" not in text or "KV cache: fp8-e4m3 backend=legacy (" not in text or "KV cache: Fp8 vmm (" in text:
                raise RuntimeError(f"{suffix}: explicit legacy was not actually allocated")
        elif "KV cache: Fp8 vmm (" not in text:
            raise RuntimeError(f"{suffix}: automatic fp8 VMM allocation missing")
        if warmup:
            print(f"{suffix}: untimed fresh-process GPU/JIT/DPM warmup", flush=True)
            continue
        report = json.loads(proc.stdout)
        rows = [row for row in report["prefill"] if row["tokens"] == 8192]
        if len(rows) != 1 or len(rows[0]["samples"]) != 3:
            raise RuntimeError(f"{suffix}: missing three pp8192 samples")
        samples = rows[0]["samples"]
        median = statistics.median(samples)
        decode = statistics.median(report["decode"][0]["samples"])
        medians.append(median)
        decodes.append(decode)
        print(f"{suffix}: pp8192 samples={samples} median={median:.2f}, decode={decode:.2f} tok/s", flush=True)
        if not args.kv_backend and median < FLOOR:
            raise RuntimeError(f"{suffix}: pp8192 median {median:.2f} < {FLOOR:.0f} tok/s")
    summary = dict(fixture=fixture, pp8192_process_medians=medians, decode_process_medians=decodes,
                   pp8192_min=min(medians), pp8192_max=max(medians),
                   decode_reference=REFERENCE_DECODE, decode_within_one_percent=all(
                       abs(rate / REFERENCE_DECODE - 1) <= 0.01 for rate in decodes))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.kv_backend and not summary["decode_within_one_percent"]:
        raise RuntimeError("decode outside 1% of 36.5 tok/s reference")
    print("PASS: actual fp8 backend and baseline guard", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
