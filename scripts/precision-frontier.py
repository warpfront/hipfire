#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Time-optimal per-site precision policy under a KLD budget (Pareto frontier).

Usage:
  precision-frontier.py TABLE.tsv [--budget 0.002] [--time dt:<label>]
        [--baseline f16] [--clamp-negative] [--quantum 1e-6]
        [--env-prefix HIPFIRE_RAON_PREFILL_PRECISION=] [--json]
        [--time-tolerance 0]
  precision-frontier.py --self-test

TABLE.tsv (header first; '#' comments and blank lines skipped) columns:
  site, tier, dkld:<corpus>... (marginal KLD vs all-baseline, may be < 0),
  dt:<label>... (marginal time vs baseline, < 0 = faster), optional note.
Every site implicitly has the baseline tier (all deltas 0). Policy = one tier
per site; additive model; KLD sum <= budget must hold for EVERY corpus.
Exact multiple-choice-knapsack Pareto merge over (corpus KLD sums, time).
"""
import argparse
import itertools
import json
import random
import sys

EPS = 1e-12


def parse_table(lines, baseline):
    rows = [l.rstrip("\n") for l in lines if l.strip() and not l.lstrip().startswith("#")]
    if not rows:
        raise SystemExit("empty table")
    hdr = rows[0].split("\t")
    if "site" not in hdr or "tier" not in hdr:
        raise SystemExit("header needs site and tier columns")
    ci = {h: i for i, h in enumerate(hdr)}
    corpora = [h for h in hdr if h.startswith("dkld:")]
    times = [h for h in hdr if h.startswith("dt:")]
    if not corpora or not times:
        raise SystemExit("need at least one dkld:<corpus> and one dt:<label> column")
    sites = {}
    for n, r in enumerate(rows[1:], 2):
        f = r.split("\t")
        site, tier = f[ci["site"]].strip(), f[ci["tier"]].strip()
        if tier == baseline:
            continue
        opts = sites.setdefault(site, {})
        if tier in opts:
            raise SystemExit(f"duplicate {site}/{tier} (row {n})")
        try:
            opts[tier] = ([float(f[ci[c]]) for c in corpora], [float(f[ci[t]]) for t in times])
        except (ValueError, IndexError):
            raise SystemExit(f"bad row {n}: {r!r}")
    return corpora, times, sites


def prune(points):
    """points: list of (klds tuple, time, node). Return non-dominated subset."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    kept = []
    for p in pts:
        dom = False
        for q in kept:
            if q[1] <= p[1] + EPS and all(a <= b + EPS for a, b in zip(q[0], p[0])):
                dom = True
                break
        if not dom:
            kept.append(p)
    return kept


def merge(sites, nc, tidx, budget, clamp, quantum):
    """sites: {site: {tier: (klds, dts)}}. Returns [(klds, time, {site: tier})]."""
    names = sorted(sites)
    opts = []
    for s in names:
        o = [("", (0.0,) * nc, 0.0)]
        for t, (k, d) in sorted(sites[s].items()):
            k = [max(x, 0.0) for x in k] if clamp else k
            o.append((t, tuple(k), d[tidx]))
        opts.append(o)
    # suffix lower bound per corpus
    suf = [[0.0] * nc for _ in range(len(names) + 1)]
    for i in range(len(names) - 1, -1, -1):
        for c in range(nc):
            suf[i][c] = suf[i + 1][c] + min(k[c] for _, k, _ in opts[i])
    pts = [((0.0,) * nc, 0.0, None)]
    for i, o in enumerate(opts):
        new = {}
        for k0, t0, node in pts:
            for tier, k, dt in o:
                kk = tuple(a + b for a, b in zip(k0, k))
                if any(kk[c] + suf[i + 1][c] > budget + EPS for c in range(nc)):
                    continue
                tt = t0 + dt
                key = tuple(round(x / quantum) for x in kk)
                nd = node if not tier else (node, names[i], tier)
                cur = new.get(key)
                if cur is None or tt < cur[1]:
                    new[key] = (kk, tt, nd)
        pts = prune(list(new.values()))
    out = []
    for k, t, node in pts:
        ch = {}
        while node:
            node, s, tier = node
            ch[s] = tier
        out.append((k, t, ch))
    return out


def solve(sites, nc, tidx, budget, clamp, quantum, tol=0.0):
    pts = [p for p in merge(sites, nc, tidx, budget, clamp, quantum)
           if all(x <= budget + EPS for x in p[0])]
    pts.sort(key=lambda p: (max(p[0]), p[1]))
    tmin = min(p[1] for p in pts)
    chosen = min((p for p in pts if p[1] <= tmin + tol + EPS),
                 key=lambda p: (max(p[0]), p[1]))
    return pts, chosen


def brute(sites, nc, tidx, budget, clamp):
    names = sorted(sites)
    allp = []
    for combo in itertools.product(*[[None] + sorted(sites[s]) for s in names]):
        k = [0.0] * nc
        t = 0.0
        for s, tier in zip(names, combo):
            if tier:
                kk, d = sites[s][tier]
                for c in range(nc):
                    k[c] += max(kk[c], 0.0) if clamp else kk[c]
                t += d[tidx]
        if all(x <= budget + EPS for x in k):
            allp.append((tuple(k), t, None))
    return prune(allp)


def fmt_policy(ch):
    return " ".join(f"{s}={t}" for s, t in sorted(ch.items())) or "(baseline)"


def report(args, corpora, times, pts, chosen):
    cn = [c[5:] for c in corpora]
    fastest = min(pts, key=lambda p: (p[1], max(p[0])))
    envp = ",".join([args.baseline] + [f"{s}={t}" for s, t in sorted(chosen[2].items())])
    if args.json:
        def d(p):
            return {"max_dkld": max(p[0]), "dkld": dict(zip(cn, p[0])), "time": p[1],
                    "policy": dict(sorted(p[2].items()))}
        print(json.dumps({"budget": args.budget, "time_column": args.time,
                          "baseline": args.baseline, "frontier": [d(p) for p in pts],
                          "time_tolerance": args.time_tolerance,
                          "fastest": d(fastest),
                          "chosen": d(chosen), "env": args.env_prefix + envp}, indent=2))
        return
    print("| max dKLD | " + " | ".join(f"dKLD {c}" for c in cn) + f" | {args.time} | policy |")
    print("|---:|" + "---:|" * len(cn) + "---:|---|")
    for p in pts:
        mark = ""
        if p is fastest:
            mark += " <- fastest"
        if p is chosen:
            mark += " <- chosen"
        print(f"| {max(p[0]):.5f} | " + " | ".join(f"{x:.5f}" for x in p[0])
              + f" | {p[1]:.1f} | {fmt_policy(p[2])}{mark} |")
    print()
    print(f"time tolerance: {args.time_tolerance:g}")
    print(args.env_prefix + envp)
    print("predicted: " + ", ".join(f"dKLD {c}={x:.5f}" for c, x in zip(cn, chosen[0]))
          + f", {args.time}={chosen[1]:.1f}")


def self_test():
    # time tolerance: fastest is -100 @0.0015; -97 @0.0002 is within tol 5
    s = {"x": {"a": ([0.0015], [-100.0]), "b": ([0.0002], [-97.0])}}
    pts, ch = solve(s, 1, 0, 0.002, False, 1e-6)
    assert ch[2] == {"x": "a"}, ch
    pts, ch = solve(s, 1, 0, 0.002, False, 1e-6, 5.0)
    assert ch[2] == {"x": "b"}, ch
    pts, ch = solve(s, 1, 0, 0.002, False, 1e-6, 2.0)
    assert ch[2] == {"x": "a"}, ch
    rnd = random.Random(1234)
    for clamp in (False, True):
        sites = {}
        for s in range(6):
            sites[f"s{s}"] = {t: ([round(rnd.uniform(-0.0005, 0.002), 5) for _ in range(2)],
                                  [round(rnd.uniform(-200, 20), 1)])
                              for t in ("a", "b", "c")}
        for budget in (0.002, 0.005, 0.01):
            got = merge(sites, 2, 0, budget, clamp, 1e-9)
            got = [p for p in got if all(x <= budget + EPS for x in p[0])]
            want = brute(sites, 2, 0, budget, clamp)
            g = sorted((tuple(round(x, 7) for x in p[0]), round(p[1], 6)) for p in got)
            w = sorted((tuple(round(x, 7) for x in p[0]), round(p[1], 6)) for p in want)
            assert g == w, (clamp, budget, g, w)
    # infeasible except baseline
    s = {"x": {"a": ([0.5], [-10.0])}}
    pts, ch = solve(s, 1, 0, 0.002, False, 1e-6)
    assert ch[2] == {} and ch[1] == 0 and len(pts) == 1
    # negative deltas used as measured; clamp ignores them
    s = {"x": {"a": ([-0.001], [-10.0])}, "y": {"a": ([0.0029], [-50.0])}}
    pts, ch = solve(s, 1, 0, 0.002, False, 1e-6)
    assert ch[2] == {"x": "a", "y": "a"}, ch
    pts, ch = solve(s, 1, 0, 0.002, True, 1e-6)
    assert ch[2] == {"x": "a"}, ch
    # second corpus binds
    s = {"x": {"a": ([0.0001, 0.005], [-100.0]), "b": ([0.0015, 0.0005], [-60.0])}}
    pts, ch = solve(s, 2, 0, 0.002, False, 1e-6)
    assert ch[2] == {"x": "b"}, ch
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description="Per-site precision Pareto frontier")
    ap.add_argument("table", nargs="?")
    ap.add_argument("--budget", type=float, default=0.002)
    ap.add_argument("--time", default=None)
    ap.add_argument("--baseline", default="f16")
    ap.add_argument("--clamp-negative", action="store_true")
    ap.add_argument("--quantum", type=float, default=1e-6)
    ap.add_argument("--env-prefix", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--time-tolerance", type=float, default=0.0,
                    help="treat times within this (dt units) of the fastest as equal; "
                         "pick lowest max dKLD among them")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.table:
        ap.error("table required")
    with (sys.stdin if args.table == "-" else open(args.table)) as f:
        corpora, times, sites = parse_table(f.readlines(), args.baseline)
    args.time = args.time or times[0]
    if args.time not in times:
        raise SystemExit(f"unknown time column {args.time}; have {times}")
    pts, chosen = solve(sites, len(corpora), times.index(args.time), args.budget,
                        args.clamp_negative, args.quantum, args.time_tolerance)
    report(args, corpora, times, pts, chosen)


if __name__ == "__main__":
    main()
