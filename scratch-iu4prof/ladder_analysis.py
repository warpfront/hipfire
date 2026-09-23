#!/usr/bin/env python3
"""Convert aggregate ladder throughput into whole-model-equivalent us/token."""
from pathlib import Path
import re
import sys

logical_gop_per_token = 48.701112320
rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.fullmatch(r"AGG,step=(\d+),median_us=([0-9.]+),TOPS=([0-9.]+)", line)
    if match:
        step, launch_us, tops = int(match[1]), float(match[2]), float(match[3])
        rows.append((step, launch_us, tops, logical_gop_per_token * 1000.0 / tops))
assert [row[0] for row in rows] == list(range(6))
previous = None
for step, launch_us, tops, equivalent_us in rows:
    delta = 0.0 if previous is None else equivalent_us - previous
    print(f"step={step} launch_us={launch_us:.3f} TOPS={tops:.3f} "
          f"equivalent_us/token={equivalent_us:.3f} marginal_us={delta:+.3f}")
    previous = equivalent_us
print(f"raw_ladder_overhead={rows[-1][3] - rows[0][3]:.3f} us/token")
print(f"measured_gap_from_peak={197.0 - logical_gop_per_token * 1000.0 / 695.085:.3f} us/token")
