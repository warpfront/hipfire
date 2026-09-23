#!/usr/bin/env python3
"""Per-symbol calls/ms delta between two run_trace.py outputs: trace_delta.py A B."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / 'trace'


def load(label):
    rows = {}
    for line in (ROOT / label / 'symbols.tsv').read_text().splitlines():
        name, calls, ms = line.split('\t')
        rows[name] = (int(calls), float(ms))
    return rows, json.loads((ROOT / label / 'summary.json').read_text())


a, b = sys.argv[1:3]
ra, sa = load(a)
rb, sb = load(b)
print(f'{"symbol":62s} {a:>18s} {b:>18s} {"delta ms":>9s}')
for name in sorted(set(ra) | set(rb), key=lambda n: -abs(rb.get(n, (0, 0))[1] - ra.get(n, (0, 0))[1])):
    ca, ma = ra.get(name, (0, 0.0))
    cb, mb = rb.get(name, (0, 0.0))
    if abs(mb - ma) < 0.5 and ca == cb:
        continue
    print(f'{name[:62]:62s} {ca:5d}/{ma:11.3f} {cb:5d}/{mb:11.3f} {mb - ma:+9.3f}')
print(f'{"TOTAL GPU":62s} {sa["dispatches"]:5d}/{sa["gpu_ms"]:11.3f} {sb["dispatches"]:5d}/{sb["gpu_ms"]:11.3f} '
      f'{sb["gpu_ms"] - sa["gpu_ms"]:+9.3f}')
ta, tb = sa['timings'], sb['timings']
print(f'server prefill ms {ta["prefill_ms"]} -> {tb["prefill_ms"]}  tok/s {ta["prefill_tok_s"]} -> {tb["prefill_tok_s"]}'
      f'  wall {sa["wall_ms"]:.1f} -> {sb["wall_ms"]:.1f}  text {sa["text"]!r} / {sb["text"]!r}')
