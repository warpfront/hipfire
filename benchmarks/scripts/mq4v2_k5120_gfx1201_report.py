#!/usr/bin/env python3
"""Validate the gfx1201 experiment and summarize all four serving arms."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    if not __debug__:
        raise SystemExit("Validation requires assertions: unset PYTHONOPTIMIZE and do not use -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    a = (root / "logits-0.f32").read_bytes()
    b = (root / "logits-1.f32").read_bytes()
    assert a and a == b, "full logits must be byte-identical"
    for flag in (0, 1):
        report = json.loads((root / f"replay-{flag}.json").read_text())
        assert report["pass"], f"replay gate failed: {flag}"
        expected = "fused_gate_up_mq4g256v2" + ("_k5120_gfx1201" if flag else "")
        other = "fused_gate_up_mq4g256v2" if flag else "fused_gate_up_mq4g256v2_k5120_gfx1201"
        for capture in report["decode"]["captures"]:
            sequence = capture["redline_capture"]["sequence"]
            names = [launch["kernel"] for launch in sequence]
            assert names.count(expected) == 64, (flag, expected, names.count(expected))
            assert other not in names, f"unexpected gate/up route: {other}"
    if args.correctness_only:
        print("PASS: byte-exact logits, replay parity, 64 routed gate/up launches per decode")
        return
    rows = []
    for arm in ("A1", "B1", "B2", "A2"):
        record, = json.loads((root / "abba" / f"{arm}.json").read_text())
        warm = json.loads((root / "abba" / f"{arm}.json.warmup.json").read_text())
        assert warm["gen"] == 128 and warm["saw_done"] and not warm["stream_error"]
        assert record["gen"] == 4096 and record["cached"] == 0
        assert record["finish"] == "length" and record["saw_done"]
        assert not record["stream_error"] and not record["attractor"]
        assert record["terminal_count"] == 1 and not record["post_terminal_bytes"]
        assert not record["decode_estimated"]
        assert not record["reasoning_content"] and record["content"]
        row = {k: record[k] for k in ("gen", "decode_tok_s", "ttft_s", "wall_s", "request_md5")}
        row.update(arm=arm, output_sha256=hashlib.sha256(record["content"].encode()).hexdigest())
        rows.append(row)
    assert len({r["request_md5"] for r in rows}) == 1
    assert len({r["output_sha256"] for r in rows}) == 1
    groups = {}
    for name in ("A", "B"):
        values = [r["decode_tok_s"] for r in rows if r["arm"].startswith(name)]
        groups[name] = dict(min=min(values), median=statistics.median(values), max=max(values))
    summary = dict(rows=rows, generic=groups["A"], k5120=groups["B"],
                   speedup=groups["B"]["median"] / groups["A"]["median"],
                   logits_bytes=len(a), logits_sha256=hashlib.sha256(a).hexdigest(),
                   scope="Fresh-process AR TG4096 ABBA; two samples per arm. TTFT is diagnostic, not a warmed prefill claim.")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
