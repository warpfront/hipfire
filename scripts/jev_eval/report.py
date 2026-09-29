# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""Accuracy + top-probability ECE (10 bins) for hipfire vs Jev's committed answers."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="same --out used by run_jevbench/run_calibration")
a = ap.parse_args()
out = Path(a.out)
bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))

# Human decision: these sources' licences restrict redistribution of derived
# statistics, or are currently unclear (bench/jev/DATA-LICENSES.md). Excluded
# from the generated report so a regeneration can't silently re-add them.
WITHHELD = {"ag-news", "duplicates", "yelp-stars", "offensive", "sentiment-it", "hellaswag"}
WITHHELD_NOTE = "Rows for datasets whose licences restrict redistributing derived results are withheld; see ../DATA-LICENSES.md."


def ece(pairs, bins=10):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        ins = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if ins:
            e += len(ins) / len(pairs) * abs(sum(p for p, _ in ins) / len(ins) - sum(c for _, c in ins) / len(ins))
    return e


def bench_stats(path):
    rows = [json.loads(l) for l in open(path)]
    return len(rows), sum(r["correct"] for r in rows) / len(rows), ece([(r["p_top"], r["correct"]) for r in rows])


def cal_stats(path):
    g = defaultdict(list)
    for r in map(json.loads, open(path)):
        if "probs" in r:
            g[r["type"]].append((max(r["probs"]), int(r["pred"] == r["gold"])))
    return {t: (len(v), sum(c for _, c in v) / len(v), ece(v)) for t, v in g.items()}


L = ["# hipfire decide vs Jev", "", "## jev-bench (500 fixed-seed rows per task)", "",
     "| task | n | Jev acc | hipfire acc | Jev ECE | hipfire ECE |", "|---|---:|---:|---:|---:|---:|"]
bench_withheld = False
for p in sorted(glob.glob(str(out / "jevbench/predictions/*.jsonl"))):
    name = Path(p).stem
    if name.startswith("x-"):
        continue
    if name in WITHHELD:
        bench_withheld = True
        continue
    n, acc, e = bench_stats(p)
    jn, jacc, je = bench_stats(bench / "predictions" / f"{name}.jsonl")
    L.append(f"| {name} | {n} | {jacc:.3f} | {acc:.3f} | {je:.3f} | {e:.3f} |")
if bench_withheld:
    L += ["", WITHHELD_NOTE]
L += ["", "## jev-ood-calibration", "", "| set | type | n | Jev acc | hipfire acc | Jev ECE | hipfire ECE |",
      "|---|---|---:|---:|---:|---:|---:|"]
cal_withheld = False
for p in sorted(glob.glob(str(out / "calibration/hipfire_*.jsonl"))):
    s = Path(p).stem.removeprefix("hipfire_")
    if s in WITHHELD:
        cal_withheld = True
        continue
    mine, jev = cal_stats(p), cal_stats(cal / "results" / f"jev_{s}.jsonl")
    for t, (n, acc, e) in mine.items():
        jn, jacc, je = jev.get(t, (0, float("nan"), float("nan")))
        L.append(f"| {s} | {t} | {n} | {jacc:.3f} | {acc:.3f} | {je:.3f} | {e:.3f} |")
if cal_withheld:
    L += ["", WITHHELD_NOTE]
(out / "report.md").write_text("\n".join(L) + "\n")
print((out / "report.md").read_text())
