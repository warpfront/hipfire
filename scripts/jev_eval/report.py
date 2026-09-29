# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""hipfire decide vs Jev: accuracy, top-probability ECE and log loss, raw vs
calibrated (spec §13.6). Reads the frozen reported rows (<out>/probs/) and the
fitted temperatures (<out>/calibration.json; without it every T is 1). No GPU,
no dataset text.

Some jev-bench tasks and calibration sets are not committed to this repo:
their source datasets' licence terms do not clearly permit redistributing
derived per-row data (label + served probabilities, no text). See
bench/jev/DATA-LICENSES.md. Their rows are simply absent from a fresh
checkout's <out>/probs/, and a restricted source (licences.py, the single
choke point) is skipped even when `calibrate.py freeze` has also produced it
locally (gitignored) from the read-only eval worktree — a withheld dataset
must not appear in any report, committed or not (human decision)."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import calibrate as cb
import licences

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="bench/jev/<model>")
a = ap.parse_args()
out = Path(a.out)
bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
cfile = out / "calibration.json"
fit = json.loads(cfile.read_text()) if cfile.exists() else None
temps = fit["decide_calibration"] if fit else {}


def jev_bench(name):
    rows = cb.read_jsonl(bench / "predictions" / f"{name}.jsonl")
    return sum(r["correct"] for r in rows) / len(rows), cb.ece([(r["p_top"], r["correct"]) for r in rows])


def jev_cal(name):
    g = defaultdict(list)
    for r in cb.read_jsonl(cal / "results" / f"jev_{name}.jsonl"):
        if "probs" in r:
            g[r["type"]].append((max(r["probs"]), int(r["pred"] == r["gold"])))
    return {t: (sum(c for _, c in v) / len(v), cb.ece(v)) for t, v in g.items()}


def f3(x):
    return f"{x:.3f}"


def cells(s):
    return f"{f3(s['acc'])} | {{je}} | {f3(s['ece_raw'])} | {f3(s['ece_cal'])} | {f3(s['nll_raw'])} | {f3(s['nll_cal'])} |"


applied = ", ".join(f"{t} T={temps.get(t, 1.0):g}" for t in cb.TYPES)
L = ["# hipfire decide vs Jev", "",
     f"Calibration applied: {applied} "
     f"({'from calibration.json' if fit else 'no calibration.json, so calibrated = raw'}). "
     "Calibrated = softmax(log p / T) per question type (spec §13.3); accuracy cannot change.", "",
     "Not every jev-bench task or calibration set is committed to this repository: a source dataset's own "
     "licence terms have to clearly permit redistributing derived per-row data (label + served "
     "probabilities, no text) before its frozen rows are staged into git — see "
     "`bench/jev/DATA-LICENSES.md` for the per-dataset decision and sources. A withheld source (ag-news, "
     "duplicates, yelp-stars, offensive, sentiment-it, hellaswag) is never in this table, even when this "
     "checkout has also frozen it locally. A row missing below either was not evaluated for this model, or "
     "is withheld for that reason.", ""]
if fit:
    L += [f"## Fit (held-out rows only, build {fit['build']}, spec §13.4)", "",
          "| type | n | T | 95% interval | held-out NLL raw | held-out NLL cal | shipped |",
          "|---|---:|---:|---|---:|---:|---|"]
    for t, v in fit["types"].items():
        if "error" in v:
            L.append(f"| {t} | {v['n']} | - | {v['error']} | - | - | no |")
            continue
        lo, hi = v["t_ci95"]
        L.append(f"| {t} | {v['n']} | {v['t']:.3f} | {lo:.3f}-{hi:.3f} | {f3(v['nll_raw'])} | "
                 f"{f3(v['nll_cal'])} | {'yes' if v['ship'] else 'no'} |")
    L.append("")
L += ["## jev-bench (500 fixed-seed rows per task)", "",
     "| task | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
score = []
for name, p in sorted(cb.find_rows(out / "probs", "jevbench_").items()):
    if licences.is_restricted(name):
        continue  # withheld dataset: never in a report, committed or local (human decision)
    s = cb.summarize(cb.read_jsonl(p), temps)
    jacc, je = jev_bench(name)
    L.append(f"| {name} | {s['n']} | {f3(jacc)} | " + cells(s).format(je=f3(je)))
    if "mae_raw" in s:
        score.append(f"| {name} | {s['n']} | {f3(s['mae_raw'])} | {f3(s['mae_cal'])} |")
L += ["", "## jev-ood-calibration", "",
      "| set | type | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name, p in sorted(cb.find_rows(out / "probs", "cal_").items()):
    if licences.is_restricted(name):
        continue  # withheld dataset: never in a report, committed or local (human decision)
    by = defaultdict(list)
    for r in cb.read_jsonl(p):
        by[r["type"]].append(r)
    jev = jev_cal(name)
    for t in cb.TYPES:
        if t not in by:
            continue
        s = cb.summarize(by[t], temps)
        jacc, je = jev.get(t, (float("nan"), float("nan")))
        L.append(f"| {name} | {t} | {s['n']} | {f3(jacc)} | " + cells(s).format(je=f3(je)))
        if "mae_raw" in s:
            score.append(f"| {name} ({t}) | {s['n']} | {f3(s['mae_raw'])} | {f3(s['mae_cal'])} |")
if score:
    L += ["", "## score answers: mean |E[score] - gold|", "",
          "| rows | n | raw | calibrated |", "|---|---:|---:|---:|"] + score
(out / "report.md").write_text("\n".join(L) + "\n")
print((out / "report.md").read_text())
