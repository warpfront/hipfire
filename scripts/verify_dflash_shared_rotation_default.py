#!/usr/bin/env python3
"""Run the official serving battery with omitted and explicit-on rotation flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPFIRE_")
           and k not in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    env.update(ROCR_VISIBLE_DEVICES="1", HIPFIRE_CLI_BIN=str(root / "target/release/hipfire"),
               HIPFIRE_DAEMON_BIN=str(root / "target/release/daemon"))
    reports = []
    for arm in ("default", "explicit-on"):
        config = dict(env)
        if arm == "explicit-on":
            for family in ("QKV", "FFN", "CTX_KV"):
                config["HIPFIRE_DRAFT_SHARED_" + family + "_ROTATION"] = "1"
        command = [sys.executable, str(root / "scripts/serve_harness.py"),
                   "--model", "/media/990Pro/hipfire/models/qwen3.8-27b.mq4-xts",
                   "--draft", str(Path.home() / ".hipfire/models/qwen38-27b-dflash-mq4.hfq"),
                   "--speculation", "dflash", "--kv", "q8", "--max-seq", "16384",
                   "--max-tokens", "1024", "--thinking", "off", "--thinking-effort", "none", "--sampling", "greedy",
                   "--mode", "battery", "--port", "11529", "--out", str(out / (arm + ".json"))]
        (out / (arm + "-command.json")).write_text(json.dumps(command, indent=2))
        with (out / (arm + ".log")).open("w") as log:
            subprocess.run(command, cwd=root, env=config, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        rows = json.loads((out / (arm + ".json")).read_text())
        if len(rows) != 5 or any(r.get("finish") != "stop" or r.get("empty") for r in rows):
            raise ValueError("Incomplete serving battery: " + arm)
        reports.append(rows)
        time.sleep(10)
    fields = ("genre", "assistant_content", "finish", "gen", "tau")
    a, b = [[{k: row.get(k) for k in fields} for row in rows] for rows in reports]
    if a != b:
        raise ValueError("Default versus explicit-on serving output/metrics mismatch")
    result = dict(default_explicit_on_parity=True, cases=5,
                  daemon_sha256=hashlib.sha256((root / "target/release/daemon").read_bytes()).hexdigest(),
                  cli_sha256=hashlib.sha256((root / "target/release/hipfire").read_bytes()).hexdigest())
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
