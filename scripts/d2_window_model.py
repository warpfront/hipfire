#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""d2_window_model.py: CPU-only model of the net of a wider D2 verify window (HIPFIRE_QWEN4_D2_WINDOW_ROWS).

What it answers (no device, numpy only): going from a 4-row verify window (K=3) to W rows (K=W-1, up to 8 / K=7),
 (1) how many MORE distinct cold (host-resident) experts does a layer touch per window, measured on the real route
     dumps (MTP verify windows, step phase 1) against a ranked hot set of B experts;
 (2) what does that cost on the CPU arm / the link, calibrated on a measured (AR, K=3 window) pair of the same card;
 (3) what does it gain, from accepted drafts per window at K=3 and K=7 (tau receipts), and the net tokens/s.

Method.
 * The dumps hold 8-row windows (K=7). The experts a W-row verify touches are those of its first W rows (the draft chain
   is prefix-consistent: row i is draft i of any deeper window). Per layer call, distinct experts of rows [0, W) minus the
   hot set = distinct COLD experts c_l(W); the whole window reads every distinct cold expert once (grouped read, both the
   CPU arm and the PCIe path), so cost is linear in c_l(W) = the "unit" of the measure; the unit is one expert =
   UNIT_MB (2.6112 MB at mq4).
 * Hot set = the top-B experts of a ranking file (global id = layer*512 + expert; B experts of the shipped decode
   sidecar, or the pooled D1 ranking). Experts of the MTP head layer are always cold (host-mapped) and are priced apart.
 * Window time T(W) = T0 + t_e * C(W), C(W) = sum over trunk layers of c_l(W). T0 (everything that does not scale with the
   cold experts) and t_e (ms per cold expert, CPU-arm effective incl. its per-layer sync share) are solved from two
   measured points of the SAME card/load: AR step (W=1) and the K=3 window minus its head drafts. The model then
   PREDICTS T(W) for W=2..8. A head draft step costs HEAD_MS (rows=1, head D2 armed: 1.35 ms per draft on the XTX;
   resident head would be ~0.3 ms; both are parameters).
 * Gain: accepted drafts per window A(K). Two receipts anchor it: A(3) and A(7) (R9700 code, tau 2.85 / 4.405 =>
   1.85 / 3.405 accepted; XTX 1.97 at K=3). Between/below the anchors the per-position conditional acceptance is linearly
   interpolated (a_1..a_3 fitted to A(3), a_4..a_7 fitted to the increment A(7)-A(3), geometric within each block).
   Tokens per window = 1 + A(K); tok/s = (1 + A(K)) / (T(K+1) + K*HEAD_MS).

Caveats (printed in the output as MODEL ASSUMPTIONS): hot-set mask is static (no LRU), window rows beyond 4 add no GPU
time besides the cold experts (optimistic: GDN/attention/shared per-row work is ignored, the few-row GPU tail is cheap),
acceptance anchors come from other prompts than the route dumps, and nothing here measures numerics. It sizes the lever
and tells which K the card sweep should look at; it never replaces that sweep.

usage:
  d2_window_model.py [--dump NAME ...] [--ranking PATH] [--budgets 3500,5000,6500] [--json OUT.json]
        [--ar-ms 29.8] [--k3-window-ms 71.6] [--head-ms 1.35] [--head-resident-ms 0.30]
        [--acc3 1.97] [--acc7 3.405]
defaults: XTX receipts (AR 33.6 tok/s => 29.8 ms; K=3 MTP 41.7 tok/s at 1.97 accepted => 71.1 ms/window).
"""
import argparse
import importlib.util
import json
import math
import sys

import numpy as np

MS5 = "/home/kaden/qcal/release-0.4.3/mgmtp/ms5"
HIT = MS5 + "/scripts/ms5_hit.py"
DUMPS = {
    "code-mtp": MS5 + "/runs/ora-dump2/d-code-mtp/route.bin",
    "div-mtp": MS5 + "/runs/ora-dump2/d-div-mtp/route.bin",
    "mixedlong-mtp": MS5 + "/runs/ora-dump2/d-mixedlong-mtp/route.bin",
    "mixed-mtp": MS5 + "/runs/ora-dump1/d-mixed-mtp/route.bin",
    "story-mtp": MS5 + "/runs/ora-dump1/d-story-mtp/route.bin",
}
DEFAULT_RANKING = "/home/kaden/qcal/release-0.4.3/hotset-confirm-044/model/qwen3.8-flash-next-base.mq4-xts.expert-hotset"
UNIT_MB = 2.6112
E = 512
TRUNK = 48


def load_hit():
    spec = importlib.util.spec_from_file_location("ms5_hit", HIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hot_mask(path, budget):
    r = json.load(open(path))
    rank = np.array(r["rank"], dtype=np.int64)
    hot = np.zeros(TRUNK * E, dtype=bool)
    hot[rank[:budget]] = True
    counts = np.array(r["counts"], dtype=np.float64)
    return hot, float(counts[rank[:budget]].sum() / counts.sum())


def window_stats(H, dump_path, hot, widths):
    """-> ({w: (distinct, cold)}, calls): per width the mean over MTP verify-window layer calls (step phase 1, 8 rows;
    one call = one window of one layer, the dumps carry several lane windows per step) of the distinct experts and the
    distinct cold experts, summed over the 48 trunk layers (= per window)."""
    d = H.load_dump(dump_path, True, False)  # trunk calls only
    topk = int(d["topk"])
    ns = d["call_ns"]
    off = np.concatenate([[0], np.cumsum(ns)])
    ids = d["ids"]
    phase = d["step_phase"]
    dist = {w: np.zeros(TRUNK) for w in widths}
    cold = {w: np.zeros(TRUNK) for w in widths}
    count = np.zeros(TRUNK)
    for c in range(len(ns)):
        if phase[int(d["call_step"][c])] != 1 or ns[c] // topk != 8:
            continue
        layer = int(d["call_layer"][c])
        if not 0 <= layer < TRUNK:
            continue
        rows = ids[off[c]:off[c + 1]].reshape(8, topk)
        count[layer] += 1
        for w in widths:
            u = np.unique(rows[:w])
            u = u[(u >= 0) & (u < E)]
            dist[w][layer] += len(u)
            cold[w][layer] += int((~hot[layer * E + u]).sum())
    seen = count > 0
    if not seen.any():
        return None
    calls = int(count.sum())
    return {w: (float((dist[w][seen] / count[seen]).sum()), float((cold[w][seen] / count[seen]).sum()))
            for w in widths}, calls


def accept_profile(acc3, acc7):
    """per-position conditional acceptance p_1..p_7 with sum_{i<=3} prod = acc3 and sum_{i<=7} = acc7."""
    def geo(a, n):
        return sum(a ** i for i in range(1, n + 1))

    lo, hi = 1e-6, 1 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if geo(mid, 3) < acc3:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    tail_target = max(acc7 - geo(a, 3), 0.0)
    base = a ** 3
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        if base * sum(mid ** i for i in range(1, 5)) < tail_target:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    return [a] * 3 + [b] * 4


def accepted(profile, k):
    run, total = 1.0, 0.0
    for p in profile[:k]:
        run *= p
        total += run
    return total


def predict(c, dist, profile, args, row_ms):
    """Calibrate (T0, t_e) on (AR step, K=3 window minus its head drafts) for a per-extra-row non-expert cost `row_ms`,
    then predict every width. -> dict or None."""
    window4 = args.k3_window_ms - 3 * args.head_ms
    if c[4] <= c[1]:
        return None
    t_e = (window4 - args.ar_ms - 3 * row_ms) / (c[4] - c[1])
    if t_e <= 0:
        return None
    t0 = args.ar_ms - c[1] * t_e
    res = {"t_e_ms": t_e, "t0_ms": t0, "row_ms": row_ms, "k": {}}
    for w in range(2, 9):
        k = w - 1
        t_w = t0 + (w - 1) * row_ms + t_e * c[w]
        a_k = accepted(profile, k)
        res["k"][k] = {
            "rows": w, "window_ms": t_w, "accepted": a_k, "tok_win": 1 + a_k,
            "tok_s": (1 + a_k) / (t_w + k * args.head_ms) * 1000,
            "tok_s_head_resident": (1 + a_k) / (t_w + k * args.head_resident_ms) * 1000,
        }
    k3, k7 = res["k"][3], res["k"][7]
    # Accepted drafts K=7 needs for tok/s parity with K=3 on this cost curve (same head cost).
    res["breakeven_acc7"] = (1 + k3["accepted"]) * (k7["window_ms"] + 7 * args.head_ms) / (
        k3["window_ms"] + 3 * args.head_ms) - 1
    res["best_k"] = max(res["k"], key=lambda kk: res["k"][kk]["tok_s"])
    res["k7_vs_k3"] = k7["tok_s"] / k3["tok_s"] - 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", choices=sorted(DUMPS), help="default: all MTP dumps")
    ap.add_argument("--ranking", default=DEFAULT_RANKING)
    ap.add_argument("--budgets", default="3500,5000,6500")
    ap.add_argument("--json")
    ap.add_argument("--ar-ms", type=float, default=29.8)
    ap.add_argument("--k3-window-ms", type=float, default=71.6)
    ap.add_argument("--head-ms", type=float, default=1.35)
    ap.add_argument("--head-resident-ms", type=float, default=0.30)
    ap.add_argument("--acc3", type=float, default=1.97)
    ap.add_argument("--acc7", type=float, default=3.405)
    ap.add_argument("--row-ms", default="0,3,6",
                    help="assumed non-expert cost of each verified row beyond the first, ms (sensitivity; 0 = the whole "
                         "K=3 window premium is cold-expert work)")
    ap.add_argument("--head-experts", type=int, default=512,
                    help="hot-set experts the VRAM-resident MTP head costs (512 x 2.6112 MB = 1275 MiB = 512 trunk "
                         "experts of the same size; 0 disables the comparison line)")
    ap.add_argument("--verbose", action="store_true", help="print the full per-width table for every cell")
    args = ap.parse_args()
    H = load_hit()
    widths = list(range(1, 9))
    budgets = [int(x) for x in args.budgets.split(",")]
    row_mss = [float(x) for x in args.row_ms.split(",")]
    profile = accept_profile(args.acc3, args.acc7)
    out = {"assumptions": __doc__.split("Caveats")[1].split("usage:")[0].strip(), "profile": profile, "cells": []}
    print(f"acceptance profile p_1..p_7 = {[round(p, 3) for p in profile]}  "
          f"A(3)={accepted(profile, 3):.2f} A(7)={accepted(profile, 7):.2f}   "
          f"AR {args.ar_ms} ms, K=3 window {args.k3_window_ms} ms, head {args.head_ms} ms/draft "
          f"(resident {args.head_resident_ms})")
    for budget in budgets:
        hot, mass = hot_mask(args.ranking, budget)
        pooled = None
        print(f"\n=== hot set B={budget} experts ({mass * 100:.0f}% of routing mass), ranking {args.ranking.split('/')[-1]} ===")
        for name in (args.dump or sorted(DUMPS)):
            got = window_stats(H, DUMPS[name], hot, widths)
            if got is None:
                print(f"  {name}: no 8-row MTP window calls")
                continue
            stats, calls = got
            c = {w: stats[w][1] for w in widths}
            dist = {w: stats[w][0] for w in widths}
            pooled = pooled or {"w": 0, "c": {w: 0.0 for w in widths}, "d": {w: 0.0 for w in widths}}
            pooled["w"] += calls
            for w in widths:
                pooled["c"][w] += c[w] * calls
                pooled["d"][w] += dist[w] * calls
            r = predict(c, dist, profile, args, 0.0)
            line = ("n/a" if r is None else
                    f"K7 vs K3 {r['k7_vs_k3'] * 100:+.1f}% at A(7)={accepted(profile, 7):.2f} (row-ms 0), best K={r['best_k']}; "
                    f"K7 breaks even only at A(7)>={r['breakeven_acc7']:.2f} accepted drafts per window "
                    f"(code-like prompts ~3.4, prose ~0.1-0.2)")
            print(f"  {name}: {calls // TRUNK} windows; cold experts/window 4 rows {c[4]:.0f} -> 8 rows {c[8]:.0f} "
                  f"(+{(c[8] / c[4] - 1) * 100:.0f}%); {line}")
            out["cells"].append({"budget": budget, "dump": name, "calls": calls, "cold": c, "distinct": dist})
        if pooled is None:
            continue
        c = {w: pooled["c"][w] / pooled["w"] for w in widths}
        dist = {w: pooled["d"][w] / pooled["w"] for w in widths}
        print(f"  pooled over dumps ({pooled['w'] // TRUNK} windows): distinct experts/window "
              + " ".join(f"{w}r:{dist[w]:.0f}" for w in range(1, 9)))
        print("                                      cold experts/window  "
              + " ".join(f"{w}r:{c[w]:.0f}" for w in range(1, 9)) + f"   per layer 4r {c[4] / TRUNK:.1f} 8r {c[8] / TRUNK:.1f}")
        for row_ms in row_mss:
            r = predict(c, dist, profile, args, row_ms)
            if r is None:
                print(f"  row-ms {row_ms}: calibration infeasible (the K=3 premium is smaller than the per-row assumption)")
                continue
            print(f"  row-ms {row_ms:g}: t_e={r['t_e_ms'] * 1000:.0f} us/cold expert ({UNIT_MB / r['t_e_ms']:.0f} GB/s effective), "
                  f"T0={r['t0_ms']:.1f} ms; K7 vs K3 {r['k7_vs_k3'] * 100:+.1f}%, best K={r['best_k']}, "
                  f"K7 needs A(7)>={r['breakeven_acc7']:.2f} accepted (have {accepted(profile, 7):.2f}); "
                  f"T(8 rows)={r['k'][7]['window_ms']:.0f} ms vs T(4 rows)={r['k'][3]['window_ms']:.0f} ms")
            if args.verbose or row_ms == row_mss[0]:
                print("      rows   K   T(ms)  acc  tok/win  tok/s(head D2)  tok/s(head resident)")
                for k in range(1, 8):
                    p = r["k"][k]
                    print(f"      {p['rows']:>4}  {k:>2}  {p['window_ms']:>6.1f}  {p['accepted']:>3.2f}  {p['tok_win']:>7.2f}  "
                          f"{p['tok_s']:>14.1f}  {p['tok_s_head_resident']:>20.1f}")
            out["cells"].append({"budget": budget, "pooled": True, "row_ms": row_ms, **r})
        if args.head_experts and budget > args.head_experts:
            # HIPFIRE_QWEN4_MTP_HEAD_RESIDENT: the head's experts take `head_experts` of the hot-set budget; its draft steps
            # then cost head_resident_ms instead of head_ms, the trunk window reads the cold experts of B - head_experts.
            hot2, _ = hot_mask(args.ranking, budget - args.head_experts)
            acc2 = {"w": 0, "c": {w: 0.0 for w in widths}}
            for name in (args.dump or sorted(DUMPS)):
                got = window_stats(H, DUMPS[name], hot2, widths)
                if got is None:
                    continue
                acc2["w"] += got[1]
                for w in widths:
                    acc2["c"][w] += got[0][w][1] * got[1]
            c2 = {w: acc2["c"][w] / acc2["w"] for w in widths}
            r0 = predict(c, dist, profile, args, row_mss[0])
            if r0 is not None:
                for k in (3, 7):
                    w = k + 1
                    t_res = r0["t0_ms"] + (w - 1) * row_mss[0] + r0["t_e_ms"] * c2[w]
                    tps = (1 + r0["k"][k]["accepted"]) / (t_res + k * args.head_resident_ms) * 1000
                    base = r0["k"][k]["tok_s"]
                    print(f"  head resident (hot set {budget} -> {budget - args.head_experts}): K={k}: cold experts/window "
                          f"{c[w]:.0f} -> {c2[w]:.0f} (+{c2[w] - c[w]:.0f}), tok/s {base:.1f} (head on D2) -> {tps:.1f} "
                          f"({(tps / base - 1) * 100:+.1f}%)")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=float)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
