#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Rank target candidates by DFlash agreement with a paired draft.

This is a calibration selector, not a quality proof. It runs
`dflash_spec_demo` in fresh processes, parses tau/acceptance/decode metrics,
checks for simple attractor failures, and emits a JSON artifact that can be
joined with KLD/PPL evidence before promoting a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "hipfire.dflash_calibrator.v0"


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p / 100.0))
    return ordered[index]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None


def parse_labeled_path(spec: str) -> dict[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"candidate must be LABEL=PATH or PATH, got {spec!r}")
        return {"label": label, "path": str(Path(path).expanduser())}
    path = str(Path(spec).expanduser())
    label = Path(path).name
    return {"label": label, "path": path}


def first_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def parse_token_stats(text: str) -> dict[str, Any]:
    match = re.search(r"DFlash tokens:\s*\[([^\]]*)\]", text, re.DOTALL)
    if not match:
        return {
            "token_count": 0,
            "unique_token_count": 0,
            "unique_ratio": None,
            "max_token_freq": None,
            "max_token_id": None,
        }
    ids: list[int] = []
    for item in match.group(1).replace("\n", " ").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(item))
        except ValueError:
            pass
    if not ids:
        return {
            "token_count": 0,
            "unique_token_count": 0,
            "unique_ratio": None,
            "max_token_freq": None,
            "max_token_id": None,
        }
    counts: dict[int, int] = {}
    for token_id in ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    max_token_id, max_count = max(counts.items(), key=lambda item: item[1])
    return {
        "token_count": len(ids),
        "unique_token_count": len(counts),
        "unique_ratio": len(counts) / len(ids),
        "max_token_freq": max_count / len(ids),
        "max_token_id": max_token_id,
    }


def extract_decoded_region(text: str) -> str:
    parts = re.split(r"decoding \(max [^)]+\)\.\.\.\n", text, maxsplit=1)
    if len(parts) != 2:
        return ""
    decoded = parts[1].split("--- OUTPUT ---", 1)[0]
    # Graph diagnostics are stderr lines that can interleave before stdout.
    lines = [line for line in decoded.splitlines() if not line.startswith("[verify-graph]")]
    return "\n".join(lines).strip() + "\n" if lines else ""


def parse_run(text: str, returncode: int, elapsed_wall_s: float) -> dict[str, Any]:
    emitted_match = re.search(
        r"emitted:\s*([0-9]+)\s+tokens\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)",
        text,
    )
    metrics: dict[str, Any] = {
        "returncode": returncode,
        "elapsed_wall_s": elapsed_wall_s,
        "parse_ok": False,
    }
    if emitted_match:
        metrics.update(
            {
                "emitted_tokens": int(emitted_match.group(1)),
                "decode_secs": float(emitted_match.group(2)),
                "decode_tok_s": float(emitted_match.group(3)),
            }
        )
    for key, pattern in [
        ("prefill_tok_s", r"^prefill_tok_s:\s*([0-9.]+)"),
        ("ttft_ms", r"^ttft_ms:\s*([0-9.]+)"),
        ("decode_tau", r"^decode_tau:\s*([0-9.]+)"),
        ("decode_accept_rate", r"^decode_accept_rate:\s*([0-9.]+)"),
    ]:
        value = first_float(text, pattern)
        if value is not None:
            metrics[key] = value
    for key, pattern in [
        ("cycles", r"^cycles:\s*([0-9]+)"),
        ("accepted", r"^cycles:\s*[0-9]+\s+committed:\s*[0-9]+\s+accepted:\s*([0-9]+)"),
    ]:
        value = first_int(text, pattern)
        if value is not None:
            metrics[key] = value
    adaptive = re.search(r"^adaptive-b:\s*(.*)$", text, re.MULTILINE)
    if adaptive:
        metrics["adaptive_b"] = adaptive.group(1)
    metrics["hit_eos"] = bool(re.search(r"(^|\n)eos(\n|$)", text))
    decoded = extract_decoded_region(text)
    metrics["decoded_md5"] = md5_bytes(decoded.encode("utf-8")) if decoded else None
    metrics["decoded_preview"] = decoded[:320]
    token_stats = parse_token_stats(text)
    metrics["token_stats"] = token_stats
    metrics["hard_fail_reasons"] = []
    if returncode != 0:
        metrics["hard_fail_reasons"].append(f"returncode={returncode}")
    if "decode_tok_s" not in metrics or "decode_tau" not in metrics:
        metrics["hard_fail_reasons"].append("missing_metrics")
    if metrics.get("emitted_tokens", 0) <= 0:
        metrics["hard_fail_reasons"].append("zero_tokens")
    max_freq = token_stats.get("max_token_freq")
    unique_ratio = token_stats.get("unique_ratio")
    if max_freq is not None and max_freq > 0.40:
        metrics["hard_fail_reasons"].append("token_attractor")
    if unique_ratio is not None and unique_ratio < 0.30:
        metrics["hard_fail_reasons"].append("low_unique_ratio")
    metrics["parse_ok"] = "missing_metrics" not in metrics["hard_fail_reasons"]
    return metrics


def build_command(args: argparse.Namespace, candidate_path: str, prompt_file: str) -> list[str]:
    cmd = [
        args.demo,
        "--target",
        candidate_path,
        "--draft",
        args.draft,
        "--prompt-file",
        prompt_file,
        "--max",
        str(args.max_tokens),
        "--temp",
        str(args.temp),
        "--kv-mode",
        args.kv_mode,
        "--ctx",
        str(args.ctx),
    ]
    if args.no_chatml:
        cmd.append("--no-chatml")
    if args.no_adaptive_b:
        cmd.append("--no-adaptive-b")
    if args.block_size is not None:
        cmd.extend(["--block-size", str(args.block_size)])
    if args.ar_baseline:
        cmd.append("--ar-baseline")
    return cmd


def run_one(
    args: argparse.Namespace,
    candidate: dict[str, str],
    prompt_file: str,
    run_index: int,
    log_dir: Path,
) -> dict[str, Any]:
    cmd = build_command(args, candidate["path"], prompt_file)
    env = os.environ.copy()
    if not args.verify_graph:
        env["HIPFIRE_VERIFY_GRAPH"] = "0"
    env.update(item.split("=", 1) for item in args.env)
    log_name = f"{candidate['label']}.{Path(prompt_file).stem}.run{run_index:02d}.log"
    log_path = log_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            env=env,
        )
        output = proc.stdout + proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = 124
    elapsed = time.time() - started
    log_path.write_text(output, encoding="utf-8", errors="replace")
    parsed = parse_run(output, returncode, elapsed)
    parsed.update(
        {
            "candidate": candidate["label"],
            "candidate_path": candidate["path"],
            "prompt_file": prompt_file,
            "run_index": run_index,
            "log_path": str(log_path),
            "command": cmd,
        }
    )
    return parsed


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(run[key]) for run in runs if isinstance(run.get(key), (int, float))]

    decode = values("decode_tok_s")
    tau = values("decode_tau")
    accept = values("decode_accept_rate")
    prefill = values("prefill_tok_s")
    emitted = values("emitted_tokens")
    output_md5s = sorted({run.get("decoded_md5") for run in runs if run.get("decoded_md5")})
    failures = [run for run in runs if run.get("hard_fail_reasons")]
    return {
        "run_count": len(runs),
        "valid_count": len(runs) - len(failures),
        "hard_fail_count": len(failures),
        "hard_fail_reasons": [run.get("hard_fail_reasons") for run in failures],
        "decode_tok_s": {
            "mean": mean(decode),
            "median": median(decode),
            "stdev": stdev(decode),
            "p10": pct(decode, 10),
            "p90": pct(decode, 90),
            "min": min(decode) if decode else None,
            "max": max(decode) if decode else None,
        },
        "decode_tau": {
            "mean": mean(tau),
            "median": median(tau),
            "stdev": stdev(tau),
            "p10": pct(tau, 10),
            "p90": pct(tau, 90),
            "min": min(tau) if tau else None,
            "max": max(tau) if tau else None,
        },
        "decode_accept_rate": {
            "mean": mean(accept),
            "median": median(accept),
            "stdev": stdev(accept),
        },
        "prefill_tok_s": {
            "median": median(prefill),
            "mean": mean(prefill),
        },
        "emitted_tokens": {
            "median": median(emitted),
            "mean": mean(emitted),
        },
        "output_md5s": output_md5s,
        "unique_output_count": len(output_md5s),
        "hit_eos_count": sum(1 for run in runs if run.get("hit_eos")),
    }


def rank_key(summary: dict[str, Any], rank_by: str) -> float:
    if summary.get("hard_fail_count", 0):
        return float("-inf")
    if rank_by == "decode_tok_s":
        return float(summary["decode_tok_s"]["median"] or float("-inf"))
    if rank_by == "tau":
        return float(summary["decode_tau"]["median"] or float("-inf"))
    if rank_by == "accept_rate":
        return float(summary["decode_accept_rate"]["median"] or float("-inf"))
    raise ValueError(rank_by)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="LABEL=PATH or PATH")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt-file", action="append", required=True)
    parser.add_argument("--demo", default="./target/release/examples/dflash_spec_demo")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max", dest="max_tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--no-chatml", action="store_true")
    parser.add_argument("--no-adaptive-b", action="store_true")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--verify-graph", action="store_true", help="leave HIPFIRE_VERIFY_GRAPH enabled")
    parser.add_argument("--ar-baseline", action="store_true", help="rank AR output instead of DFlash")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    parser.add_argument("--rank-by", choices=("decode_tok_s", "tau", "accept_rate"), default="tau")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    for item in args.env:
        if "=" not in item:
            parser.error(f"--env must be KEY=VALUE, got {item!r}")
    if args.runs <= 0:
        parser.error("--runs must be positive")

    candidates = [parse_labeled_path(spec) for spec in args.candidate]
    prompt_files = [str(Path(path).expanduser()) for path in args.prompt_file]
    missing = [path for path in [args.demo, args.draft, *prompt_files, *(c["path"] for c in candidates)] if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_dir = Path(args.log_dir or f"/tmp/dflash-calibrator-{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt_meta = [
        {"path": path, "bytes": Path(path).stat().st_size, "md5": md5_file(Path(path))}
        for path in prompt_files
    ]
    candidate_meta = [
        {
            "label": cand["label"],
            "path": cand["path"],
            "bytes": Path(cand["path"]).stat().st_size,
            "md5": md5_file(Path(cand["path"])),
        }
        for cand in candidates
    ]

    all_runs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cand in candidates:
        for prompt_file in prompt_files:
            key = (cand["label"], prompt_file)
            grouped[key] = []
            for run_index in range(1, args.runs + 1):
                result = run_one(args, cand, prompt_file, run_index, log_dir)
                all_runs.append(result)
                grouped[key].append(result)
                status = "FAIL" if result.get("hard_fail_reasons") else "ok"
                tok_s = result.get("decode_tok_s")
                tau = result.get("decode_tau")
                accept = result.get("decode_accept_rate")
                print(
                    f"{cand['label']} {Path(prompt_file).name} run={run_index} {status} "
                    f"tok_s={tok_s} tau={tau} accept={accept} log={result['log_path']}",
                    flush=True,
                )

    prompt_summaries = []
    by_candidate: dict[str, list[dict[str, Any]]] = {cand["label"]: [] for cand in candidates}
    for (label, prompt_file), runs in grouped.items():
        summary = summarize_runs(runs)
        item = {"candidate": label, "prompt_file": prompt_file, **summary}
        prompt_summaries.append(item)
        by_candidate[label].extend(runs)

    candidate_summaries = []
    for cand in candidates:
        summary = summarize_runs(by_candidate[cand["label"]])
        candidate_summaries.append({"candidate": cand["label"], **summary})
    ranked = sorted(
        candidate_summaries,
        key=lambda item: rank_key(item, args.rank_by),
        reverse=True,
    )

    artifact = {
        "schema": SCHEMA,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank_by": args.rank_by,
        "demo": args.demo,
        "demo_md5": md5_file(Path(args.demo)),
        "draft": args.draft,
        "draft_md5": md5_file(Path(args.draft)),
        "settings": {
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "ctx": args.ctx,
            "kv_mode": args.kv_mode,
            "temp": args.temp,
            "no_chatml": args.no_chatml,
            "no_adaptive_b": args.no_adaptive_b,
            "block_size": args.block_size,
            "verify_graph": args.verify_graph,
            "ar_baseline": args.ar_baseline,
            "env": args.env,
        },
        "log_dir": str(log_dir),
        "prompts": prompt_meta,
        "candidates": candidate_meta,
        "prompt_summaries": prompt_summaries,
        "candidate_summaries": candidate_summaries,
        "ranked": ranked,
        "runs": all_runs,
    }

    print("\n=== DFlash Calibrator Summary ===")
    print(f"rank_by={args.rank_by} log_dir={log_dir}")
    for index, item in enumerate(ranked, 1):
        decode = item["decode_tok_s"]["median"]
        tau = item["decode_tau"]["median"]
        accept = item["decode_accept_rate"]["median"]
        failures = item["hard_fail_count"]
        unique = item["unique_output_count"]
        print(
            f"{index}. {item['candidate']}: "
            f"decode_med={decode:.2f} tau_med={tau:.4f} "
            f"accept_med={accept:.4f} failures={failures} unique_outputs={unique}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(artifact, indent=2 if args.pretty else None, sort_keys=args.pretty) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out_path}")
    else:
        print(json.dumps(artifact, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
