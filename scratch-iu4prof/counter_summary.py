#!/usr/bin/env python3
"""Summarize gfx1201 counters returned for the filtered production iu4 kernels."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys

with Path(sys.argv[1]).open(newline="") as stream:
    rows = list(csv.DictReader(stream))
values = defaultdict(list)
for row in rows:
    values[row["Counter_Name"]].append(float(row["Counter_Value"]))
print(f"rows={len(rows)} dispatches={len({row['Dispatch_Id'] for row in rows})}")
for name in sorted(values):
    data = values[name]
    print(f"{name}: n={len(data)} min={min(data):.9g} "
          f"median={statistics.median(data):.9g} max={max(data):.9g}")
