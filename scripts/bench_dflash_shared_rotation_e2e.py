#!/usr/bin/env python3
"""Uninstrumented full-output ABBA for the opt-in shared-rotation bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--daemon", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cli, daemon = args.cli.resolve(), args.daemon.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPFIRE_")
           and k not in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    env.update(ROCR_VISIBLE_DEVICES="1", HIPFIRE_HOME=str(out / "config"),
               HIPFIRE_DAEMON_BIN=str(daemon))
    subprocess.run([str(cli), "config", "set", "developer.dflash_draft",
                    str(Path.home() / ".hipfire/models/qwen38-27b-dflash-mq4.hfq")],
                   env=env, check=True)
    (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    (out / "commit.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root))
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    (out / "binaries.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in (cli, daemon)}, indent=2))
    cases = json.loads(args.cases.read_text())[:5]
    summaries = []
    for case in cases:
        directory = out / case["id"]
        directory.mkdir()
        prompt = directory / "prompt.txt"
        prompt.write_text(case["prompt"], encoding="utf-8")
        rows = []
        for block in range(2):
            for index, enabled in enumerate((False, True, True, False)):
                label = f"{block}-{index}-{'B' if enabled else 'A'}"
                run = directory / label
                run.mkdir()
                config = dict(env)
                for flag in ("QKV", "FFN", "CTX_KV"):
                    config["HIPFIRE_DRAFT_SHARED_" + flag + "_ROTATION"] = str(int(enabled))
                command = [str(cli), "bench", "/media/990Pro/hipfire/models/qwen3.8-27b.mq4-xts",
                           "--spec", "dflash", "--kv-mode", "q8", "--max-seq", "16384",
                           "--prompt-file", str(prompt), "--warmups", "2", "--runs", "1",
                           "--max-tokens", "8192", "--json"]
                (run / "command.json").write_text(json.dumps(command))
                with (run / "report.json").open("w") as stdout, (run / "run.log").open("w") as stderr:
                    subprocess.run(command, env=config, cwd=root, stdout=stdout, stderr=stderr,
                                   check=True, timeout=1800)
                report = json.loads((run / "report.json").read_text())
                if report.get("spec_route") != ["dflash"]:
                    raise ValueError("DFlash route did not run")
                event = report["run_events"][0]
                if not all(isinstance(event.get(key), (int, float)) for key in ("tau", "cycles", "decode_tok_s")):
                    raise ValueError("Missing speculative/timing metrics: " + label)
                if event.get("finish_reason") != "stop" or event.get("tokens", 0) <= 0:
                    raise ValueError("Incomplete output: " + label)
                text = event["bench_output"]
                (run / "output.md").write_text(text, encoding="utf-8")
                row = dict(scenario=case["id"], label=label, enabled=enabled,
                           tokens=event["tokens"], decode_tok_s=event["decode_tok_s"],
                           ttft_ms=event.get("ttft_ms"), tau=event.get("tau"), cycles=event.get("cycles"),
                           output_sha256=hashlib.sha256(text.encode()).hexdigest())
                rows.append(row)
                with (directory / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
                time.sleep(10)
        a = [r["decode_tok_s"] for r in rows if not r["enabled"]]
        b = [r["decode_tok_s"] for r in rows if r["enabled"]]
        summary = dict(scenario=case["id"], A_decode_median=statistics.median(a),
                       B_decode_median=statistics.median(b),
                       speedup=statistics.median(b) / statistics.median(a),
                       A_range=[min(a), max(a)], B_range=[min(b), max(b)],
                       text_parity=len({r["output_sha256"] for r in rows}) == 1,
                       tau_parity=len({r["tau"] for r in rows}) == 1,
                       cycles_parity=len({r["cycles"] for r in rows}) == 1)
        temporary = directory / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2))
        temporary.replace(directory / "summary.json")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
