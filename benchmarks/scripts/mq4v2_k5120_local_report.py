#!/usr/bin/env python3
"""Summarize event-timed ABBA/BAAB blocks, not independent kernel samples."""
import csv
import io
import json
import statistics as stats
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
header = "ring,round,position,arm,us_per_call,nominal_weight_GBs\n"
rows = list(csv.DictReader(io.StringIO(header + text.split(header, 1)[1])))
result = {}
for ring in (1, 16):
    selected = [r for r in rows if int(r["ring"]) == ring]
    if len(selected) != 40:
        raise ValueError(f"incomplete ring {ring}")
    arms = {}
    for arm in (0, 1):
        times = sorted(float(r["us_per_call"]) for r in selected if int(r["arm"]) == arm)
        if len(times) != 20:
            raise ValueError("incomplete arm")
        arms[arm] = {"median_us": stats.median(times), "trim2_mean_us": stats.mean(times[2:-2]), "min_us": min(times), "max_us": max(times)}
    ratios = []
    for rnd in range(10):
        block = [r for r in selected if int(r["round"]) == rnd]
        expected = [0, 1, 1, 0] if rnd % 2 == 0 else [1, 0, 0, 1]
        if [int(r["arm"]) for r in block] != expected or [int(r["position"]) for r in block] != list(range(4)):
            raise ValueError("invalid balanced block")
        means = [stats.mean(float(r["us_per_call"]) for r in block if int(r["arm"]) == a) for a in (0, 1)]
        ratios.append(means[0] / means[1])
    delta_us = arms[0]["median_us"] - arms[1]["median_us"]
    result[ring] = {"arms": arms, "median_speedup": arms[0]["median_us"] / arms[1]["median_us"], "paired_block_speedup_median": stats.median(ratios), "positive_blocks": sum(r > 1 for r in ratios), "blocks": len(ratios), "paired_ratios": ratios, "projected_64_layer_saving_ms_not_e2e_measurement": delta_us * 64 / 1000}
print(json.dumps(result, indent=2))
