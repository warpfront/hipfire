#!/usr/bin/env python3
"""Print top-N rocprofv3 kernel stats as a percentage of total GPU time.

Usage: rocprof_top.py <trace_kernel_stats.csv> [N]
"""
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 25

rows = list(csv.DictReader(csv_path.open()))
rows.sort(key=lambda r: float(r["TotalDurationNs"]), reverse=True)
total = sum(float(r["TotalDurationNs"]) for r in rows)

print(f"Total: {total/1e6:.1f} ms across {len(rows)} unique kernels")
print()
print(f"{'pct':>6}  {'calls':>7}  {'us/call':>9}  {'total ms':>9}  name")
print("-" * 100)
for r in rows[:n]:
    pct = float(r["TotalDurationNs"]) / total * 100
    calls = int(r["Calls"])
    avg_us = float(r["AverageNs"]) / 1000
    total_ms = float(r["TotalDurationNs"]) / 1e6
    name = r["Name"]
    # Strip C++ mangling noise; truncate to fit
    short = name.replace("void ", "").split("(")[0]
    if len(short) > 70:
        short = short[:67] + "..."
    print(f"{pct:>5.1f}%  {calls:>7}  {avg_us:>9.1f}  {total_ms:>8.1f}  {short}")
