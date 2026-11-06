#!/usr/bin/env python3
"""sc_hit_sim.py: single-card per-expert hot-set hit rate and cold-expert traffic on real route dumps (CPU only, numpy).

Models ONE card holding the top-N ranked experts (global id = layer*experts_per_layer + expert, rank order of a
`hipfire.qwen4.expert_hotset` v1 sidecar) and reading every other expert of a layer call from host memory (PCIe zero-copy
and/or the D2 CPU arm). For each step it counts, per layer call, the DISTINCT experts of the step that are not resident:
a cold expert is read once per call however many rows route to it (the GPU route reads a distinct expert once; the CPU arm
reads its bytes once and computes all rows). Trunk layers only; the MTP head layer is not modelled (see HITSIM.md).

Dumps: the K0/MS5 route dumps (HFK0RT01) read with ms5_hit.py's parser (--ms5-hit). They come from 8-lane oracle runs.
  * AR dump: one call per layer per step with 8 rows = 8 lanes (ns = 80 = 8 x top-k 10), lane i = row i. So a C1 decode
    step is exactly one row (lane) of one step. A rows-W window is modelled as W consecutive steps of ONE lane (the
    perfect-acceptance verify window: row j routes like the AR token at j) -> "ar-proxy".
  * MTP dump (--k 3): 4 calls per layer per step of <= 8 rows; the concatenated rows are LANE-major (lane 0's 4 window rows,
    then lane 1's ...), segmented with the window records in <dump>/run/ids.jsonl (see mtp3_tensor). Full 4-row windows
    only. This is the real K=3 (rows 4) window incl. rejected-draft rows -> "real".
No dump with K=7 exists; rows 8 is the ar-proxy only.

usage: sc_hit_sim.py --out OUT.md [--json OUT.json] [--budgets 3500,...] [--ranking NAME=PATH ...]
Defaults: the three rankings and the dump sets named in DEFAULT_*; everything is read-only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

MS5_HIT = "/home/kaden/qcal/release-0.4.3/mgmtp/ms5/scripts/ms5_hit.py"
K0 = "/home/kaden/qcal/release-0.4.3/mgmtp/k0/runs"
MS5 = "/home/kaden/qcal/release-0.4.3/mgmtp/ms5/runs"
RK = "/home/kaden/qcal/release-0.4.3/mgmtp/d1/rankings"
SHIPPED = "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts.expert-hotset"
DEFAULT_RANKINGS = [
    ("shipped-decode", SHIPPED),
    ("pooled-k0-miheat", f"{RK}/pooled-k0-miheat.expert-hotset"),
    ("k0-cal", f"{RK}/k0-cal.expert-hotset"),
]
# (set label, status, [(domain, path, kind)]) kind: ar | mtp3
DEFAULT_SETS = [
    ("K0 held-out (4 domains x 8 lanes x 384 tok)", "held-out for k0-cal / pooled; shipped: disjoint corpus per its provenance",
     [("code", f"{K0}/r6/ar-held-code", "ar"), ("prose", f"{K0}/r4/ar-held-prose", "ar"),
      ("tool", f"{K0}/r4/ar-held-tool", "ar"), ("multi", f"{K0}/r4/ar-held-multi", "ar")]),
    ("K0 held-out MTP K=3 real windows", "held-out for k0-cal / pooled; shipped: disjoint corpus per its provenance",
     [("code", f"{K0}/r4/mtpb-held-code", "mtp3"), ("prose", f"{K0}/r7/mtpb-held-prose", "mtp3"),
      ("tool", f"{K0}/r7/mtpb-held-tool", "mtp3"), ("multi", f"{K0}/r8/mtpb-held-multi", "mtp3")]),
    ("MS5 W1/mixed prompts (AR, 8 lanes x 256 tok)", "not in K0 cal; shipped/miheat overlap not verified",
     [("story", f"{MS5}/ora-dump1/d-story-ar", "ar"), ("mixed", f"{MS5}/ora-dump1/d-mixed-ar", "ar"),
      ("code", f"{MS5}/ora-dump2/d-code-ar", "ar")]),
    ("K0 cal (IN-SAMPLE for k0-cal and pooled)", "in-sample for k0-cal, pooled-k0-miheat",
     [("code", f"{K0}/r1/ar-cal-code", "ar"), ("prose", f"{K0}/r1/ar-cal-prose", "ar"),
      ("multi", f"{K0}/r2/ar-cal-multi", "ar"), ("tool", f"{K0}/r2/ar-cal-tool", "ar")]),
]
E = 512
LAYERS = 48
TOPK = 10
MB_MIB = 2.6112       # one expert at mq4 (gate_up 1,740,800 + down 870,400 B), decimal MB
MB_F1 = 2.5344        # F1 planar layout
CPU_GBS = 113.0       # measured CPU read bandwidth on hipx (task context)
LINK_GBS = 4.2        # measured zero-copy link bandwidth on hipx (task context)
WINDOWS = {"C1 (rows 1)": 1, "W4 (K=3, rows 4)": 4, "W8 (K=7, rows 8)": 8}


def load_ms5(path):
    spec = importlib.util.spec_from_file_location("ms5_hit", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ranking(path):
    r = json.load(open(path))
    assert r.get("kind") == "hipfire.qwen4.expert_hotset" and r.get("version") == 1, path
    assert r["layers"] == LAYERS and r["experts_per_layer"] == E, path
    rank = np.array(r["rank"], dtype=np.int64)
    assert rank.size == LAYERS * E and np.unique(rank).size == rank.size, path
    return rank


def resident_mask(rank, n):
    m = np.zeros(LAYERS * E, bool)
    m[rank[:n]] = True
    return m.reshape(LAYERS, E)


def layer_mask(layers):
    m = np.zeros((LAYERS, E), bool)
    m[list(layers)] = True
    return m


def ar_tensor(H, path):
    """-> ids[step, layer, lane, k] from an AR dump (one 8-row call per layer per step, trunk only)."""
    d = H.load_dump(os.path.join(path, "route.bin"), use_sidecar=False, use_head=False)
    ns = d["call_ns"]
    assert d["layers"] == LAYERS and d["topk"] == TOPK, path
    assert (d["step_phase"] == 0).all() and (ns == ns[0]).all(), f"{path}: not a uniform AR dump"
    rows = int(ns[0]) // TOPK
    steps = d["step_phase"].size
    assert d["ids"].size == steps * LAYERS * rows * TOPK, f"{path}: call count != steps*layers"
    assert (d["call_layer"].reshape(steps, LAYERS) == np.arange(LAYERS)).all()
    return d["ids"].reshape(steps, LAYERS, rows, TOPK)


def mtp3_tensor(H, path):
    """-> ids[window, layer, pos(4), k]: every full (offered=3, 4-row) lane window of the MTP dump.
    A dump step has 4 calls per layer of <= 8 rows; their rows concatenated are lane-major (lane 0's window rows, then lane
    1's, ...), NOT one lane per row position. Window sizes come from <dump>/run/ids.jsonl window records (rows = offered+1,
    lanes ascending) aligned by order with the dump's phase-1 steps (ms5_hit.full_lane_steps rule); steps whose per-layer row
    total disagrees with the records are skipped (verified: within-window adjacent rows share 3.8 experts, across windows 0.4)."""
    d = H.load_dump(os.path.join(path, "route.bin"), use_sidecar=False, use_head=False)
    assert d["layers"] == LAYERS and d["topk"] == TOPK, path
    win = {}
    with open(os.path.join(path, "run", "ids.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "window" and "step" in r.get("window", {}):
                win.setdefault(r["window"]["step"], []).append(r["window"])
    keys = sorted(win)
    p1 = np.flatnonzero(d["step_phase"] == 1)
    assert len(keys) == len(p1), f"{path}: window records do not align with phase-1 steps"
    cs, ns = d["call_step"], d["call_ns"]
    off = np.concatenate([[0], np.cumsum(ns)])
    out = []
    for j, s in enumerate(p1):
        ws = sorted(win[keys[j]], key=lambda w: w["lane"])
        rows = [w["offered"] + 1 for w in ws]
        c = np.flatnonzero(cs == s)
        if c.size != LAYERS * 4:
            continue
        tot = ns[c].reshape(LAYERS, 4).sum(axis=1) // TOPK
        if not (tot == sum(rows)).all():
            continue
        flat = np.stack([np.concatenate([d["ids"][off[i]:off[i + 1]].reshape(-1, TOPK)
                                         for i in c[l * 4:(l + 1) * 4]]) for l in range(LAYERS)])  # layer, row, k
        b = np.cumsum([0] + rows)
        for w_i, r in enumerate(rows):
            if r == 4:
                out.append(flat[:, b[w_i]:b[w_i] + 4, :])
    return np.stack(out) if out else np.zeros((0, LAYERS, 4, TOPK), np.int64)


def window_cold(ids, resident, r):
    """ids[step, layer, lane, k]; windows of r consecutive steps of one lane (stride r).
    -> (cold distinct experts per window summed over layers [n], distinct per window [n])."""
    steps, _, lanes, _ = ids.shape
    nw = steps // r
    cold, dist = [], []
    lay = np.arange(LAYERS)[None, :, None]
    for lane in range(lanes):
        x = ids[: nw * r, :, lane, :].reshape(nw, r, LAYERS, TOPK).transpose(0, 2, 1, 3).reshape(nw, LAYERS, r * TOPK)
        pres = np.zeros((nw, LAYERS, E), bool)
        w = np.arange(nw)[:, None, None]
        pres[w, lay, x] = True
        dist.append(pres.sum(axis=(1, 2)))
        cold.append((pres & ~resident[None]).sum(axis=(1, 2)))
    return np.concatenate(cold), np.concatenate(dist)


def mtp_window_cold(ids, resident):
    """ids[window, layer, pos, k] -> per-window cold/distinct experts summed over layers."""
    n = ids.shape[0]
    pres = np.zeros((n, LAYERS, E), bool)
    pres[np.arange(n)[:, None, None], np.arange(LAYERS)[None, :, None], ids.reshape(n, LAYERS, -1)] = True
    return (pres & ~resident[None]).sum(axis=(1, 2)), pres.sum(axis=(1, 2))


def agg(cold, dist):
    return dict(cold=float(cold.mean()), dist=float(dist.mean()), hit=float(1 - cold.sum() / dist.sum()), n=int(cold.size))


def evaluate(tensors, resident):
    """tensors: list of (domain, kind, ids). -> {window label: agg over all domains pooled by window count,
    per-domain: {domain: {label: agg}}}"""
    per = {}
    for dom, kind, ids in tensors:
        row = {}
        for lab, r in WINDOWS.items():
            if kind == "ar":
                c, d = window_cold(ids, resident, r)
                row[lab + (" ar-proxy" if r > 1 else "")] = agg(c, d)
        if kind == "mtp3":
            c, d = mtp_window_cold(ids, resident)
            row["W4 (K=3, rows 4) real"] = agg(c, d)
        per[dom] = row
    labels = []
    for row in per.values():
        for k in row:
            if k not in labels:
                labels.append(k)
    pooled = {}
    for k in labels:
        parts = [row[k] for row in per.values() if k in row]
        n = sum(p["n"] for p in parts)
        cold = sum(p["cold"] * p["n"] for p in parts) / n
        dist = sum(p["dist"] * p["n"] for p in parts) / n
        pooled[k] = dict(cold=cold, dist=dist, hit=1 - cold / dist, n=n)
    return pooled, per


def ms(mb, gbs):
    return mb / (gbs * 1000.0) * 1000.0  # MB / (GB/s) -> ms


def fmt_cell(a):
    return f"{100 * a['hit']:.1f}% / {a['cold']:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--budgets", default="3500,4000,5000,6000,6500,8000,8800")
    ap.add_argument("--ranking", action="append", default=[], help="NAME=PATH (replaces the default rankings)")
    ap.add_argument("--ms5-hit", default=MS5_HIT)
    a = ap.parse_args()
    H = load_ms5(a.ms5_hit)
    budgets = [int(x) for x in a.budgets.split(",")]
    rankings = [(n, p) for n, p in (s.split("=", 1) for s in a.ranking)] or DEFAULT_RANKINGS
    ranks = [(n, load_ranking(p), p) for n, p in rankings]
    sets = []
    for label, status, items in DEFAULT_SETS:
        ts = []
        for dom, path, kind in items:
            ts.append((dom, kind, ar_tensor(H, path) if kind == "ar" else mtp3_tensor(H, path)))
        sets.append((label, status, ts))
    result = {"budgets": budgets, "sets": {}}
    summ = {}
    lines = []
    w = lines.append
    w("# HITSIM: single-card per-expert hot set, hit rate and cold traffic from real route dumps\n")
    w("Generated by `scripts/sc_hit_sim.py` (CPU only). Experts: 24,576 (48 layers x 512), one expert = "
      f"{MB_MIB} MB at mq4 ({MB_F1} MB with F1 planar). Cold = distinct experts of a layer call that are not in the card's "
      "resident set; each is read once per call (PCIe zero-copy) or by the CPU arm (bytes read once, all rows computed).\n")
    w("Cells are `distinct hit% / cold experts per step`; the cold count is summed over the 48 trunk layers "
      "(one step/window = one layer call per layer). **C1** = one AR row; **W4** = K=3 window, 4 rows; **W8** = K=7 window, "
      "8 rows. *ar-proxy* = W consecutive AR tokens of one lane (perfect-acceptance verify window); *real* = actual K=3 MTP "
      "window rows from the MTP dumps (incl. rejected drafts). No K=7 dump exists, so W8 is the ar-proxy only. Head (MTP "
      "head layer) experts are not in the ranking and not counted here.\n")
    w("Whole-layer baseline column `WL R=n`: the top ranked layers are NOT chosen; the first `n = floor(N/512)` layers are wholly "
      "resident and every other layer is cold (today's layer-granular split), shown as hit% / cold experts for C1.\n")
    w(f"Projection constants: CPU read {CPU_GBS:g} GB/s, zero-copy link {LINK_GBS:g} GB/s (hipx, task context; the live "
      "values are in the d2_default selector log). `all-CPU` = cold bytes / CPU bw; `all-link` = cold bytes / link bw; "
      "`ideal split` = cold bytes / (CPU bw + link bw) (perfect balance, both arms streaming concurrently; an upper bound "
      "on the split gain, ignoring CPU compute, row work, launch and sync cost).\n")
    for label, status, ts in sets:
        w(f"\n## Eval set: {label}\n")
        w(f"Sample status: {status}. Domains: {', '.join(d for d, _, _ in ts)}.\n")
        result["sets"][label] = {}
        for rname, rank, rpath in ranks:
            w(f"\n### Ranking `{rname}` on {label}\n")
            # one pass over budgets
            tab = {}
            for n in budgets:
                res = resident_mask(rank, n)
                pooled, per = evaluate(ts, res)
                wlm = layer_mask(range(n // E))
                wpooled, _ = evaluate(ts, wlm)
                tab[n] = (pooled, per, wpooled)
            wins = list(tab[budgets[0]][0].keys())
            hdr = ["N experts", "resident MiB", *wins, f"WL baseline {wins[0]}"]
            w("| " + " | ".join(hdr) + " |")
            w("|" + "---|" * len(hdr))
            for n in budgets:
                pooled, per, wp = tab[n]
                resmib = n * MB_MIB * 1e6 / 2**20
                wlk = wins[0]
                row = [str(n), f"{resmib:,.0f}", *[fmt_cell(pooled[k]) for k in wins],
                       f"{n // E} layers: {fmt_cell(wp[wlk])}"]
                w("| " + " | ".join(row) + " |")
            w("")
            w("Cold MB per step and projected arm time (mq4 2.6112 MB/expert):\n")
            hdr = ["N", "window", "cold MB", "cold MB (F1)", "all-CPU ms", "all-link ms", "ideal split ms"]
            w("| " + " | ".join(hdr) + " |")
            w("|" + "---|" * len(hdr))
            for n in budgets:
                pooled = tab[n][0]
                for k in wins:
                    c = pooled[k]["cold"]
                    mb = c * MB_MIB
                    w(f"| {n} | {k} | {mb:.0f} | {c * MB_F1:.0f} | {ms(mb, CPU_GBS):.1f} | {ms(mb, LINK_GBS):.1f} | "
                      f"{ms(mb, CPU_GBS + LINK_GBS):.1f} |")
            w("")
            result["sets"][label][rname] = {str(n): {"pooled": tab[n][0], "per_domain": tab[n][1]} for n in budgets}
            summ[(label, rname)] = {n: tab[n][0] for n in budgets}
            # per-domain C1 spread at N=3500/5000/8800
            doms = [d for d, _, _ in ts]
            w(f"Per-domain `{wins[0]}` hit% / cold experts per step at selected N:\n")
            w("| N | " + " | ".join(doms) + " |")
            w("|" + "---|" * (len(doms) + 1))
            for n in [b for b in budgets if b in (3500, 5000, 6500, 8800)]:
                per = tab[n][1]
                cells = []
                for d in doms:
                    k = next(iter(per[d]))
                    cells.append(fmt_cell(per[d][k]))
                w(f"| {n} | " + " | ".join(cells) + " |")
            w("")
    # whole-layer 7 resident layers table (first-7 layers) on the first set, all window kinds
    label, status, ts = sets[0]
    w("\n## Whole-layer baseline (today): 7 resident layers = 3,584 experts\n")
    w(f"Eval set: {label}. Layers 0-6 resident, 41 layers cold. Compare with N=3500/4000 above.\n")
    w("| resident layers | window | hit% | cold experts/step | cold MB | all-CPU ms | all-link ms | ideal split ms |")
    w("|---|---|---|---|---|---|---|---|")
    for lay_sel_name, lays in (("first 7 (layers 0-6)", range(7)),):
        pooled, _ = evaluate(ts, layer_mask(lays))
        for k, v in pooled.items():
            mb = v["cold"] * MB_MIB
            w(f"| {lay_sel_name} | {k} | {100 * v['hit']:.1f}% | {v['cold']:.0f} | {mb:.0f} | {ms(mb, CPU_GBS):.1f} | "
              f"{ms(mb, LINK_GBS):.1f} | {ms(mb, CPU_GBS + LINK_GBS):.1f} |")
    # best-7 layers by shipped ranking mass (optimistic whole-layer)
    rk = ranks[0][1]
    mass = np.zeros(LAYERS)
    pos = np.empty(rk.size, np.int64)
    pos[rk] = np.arange(rk.size)
    for L in range(LAYERS):
        mass[L] = -pos[L * E:(L + 1) * E].mean()  # layers whose experts rank earliest
    best = np.argsort(-mass)[:7]
    pooled, _ = evaluate(ts, layer_mask(best))
    for k, v in pooled.items():
        mb = v["cold"] * MB_MIB
        w(f"| best 7 by `{ranks[0][0]}` mean rank ({','.join(map(str, sorted(best.tolist())))}) | {k} | "
          f"{100 * v['hit']:.1f}% | {v['cold']:.0f} | {mb:.0f} | {ms(mb, CPU_GBS):.1f} | {ms(mb, LINK_GBS):.1f} | "
          f"{ms(mb, CPU_GBS + LINK_GBS):.1f} |")
    w("")
    head = summary(sets, ranks, budgets, summ)
    lines[2:2] = head
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)
    print(f"wrote {a.out}", file=sys.stderr)


def summary(sets, ranks, budgets, summ):
    """Headline: held-out C1 / W4 hit% and all-CPU ms per ranking and N; link share of the ideal CPU+link split."""
    out = ["## Summary (held-out K0 dumps)\n"]
    held_ar = sets[0][0]
    held_mtp = sets[1][0]
    out.append("`hit% / all-CPU ms` (cold MB at 2.6112 MB/expert over 113 GB/s). C1 from the K0 held-out AR dumps; W4 = real "
               "K=3 MTP windows (K0 held-out MTP dumps); W8 = ar-proxy (K0 held-out AR dumps). `link gain` = how much the "
               f"{LINK_GBS:g} GB/s zero-copy link adds to the {CPU_GBS:g} GB/s CPU arm at perfect balance "
               f"(= {100 * LINK_GBS / (CPU_GBS + LINK_GBS):.1f}% off the all-CPU time, independent of N).\n")
    out.append("| ranking | N | C1 | W4 real | W8 ar-proxy |")
    out.append("|---|---|---|---|---|")
    for rname, _, _ in ranks:
        for n in budgets:
            c1 = summ[(held_ar, rname)][n]
            w4 = summ[(held_mtp, rname)][n]
            cells = []
            for pool, key in ((c1, "C1"), (w4, "W4"), (c1, "W8")):
                k = next(kk for kk in pool if kk.startswith(key))
                cells.append(f"{100 * pool[k]['hit']:.1f}% / {ms(pool[k]['cold'] * MB_MIB, CPU_GBS):.1f} ms")
            out.append(f"| {rname} | {n} | {cells[0]} | {cells[1]} | {cells[2]} |")
    out.append("")
    # self-reported (in-sample) routing-mass coverage of each ranking vs its held-out C1 distinct hit
    out.append("Self-coverage check (ranking file's own counts, in-sample) vs held-out C1 hit:\n")
    out.append("| ranking | N | own-counts mass coverage | K0 held-out C1 hit | gap (pp) |")
    out.append("|---|---|---|---|---|")
    for rname, rank, path in ranks:
        counts = np.array(json.load(open(path))["counts"], dtype=np.float64)
        cs = np.cumsum(counts[rank]) / counts.sum()
        for n in budgets:
            if n not in (3500, 5000, 6500, 8800):
                continue
            c1 = summ[(held_ar, rname)][n]
            k = next(kk for kk in c1 if kk.startswith("C1"))
            out.append(f"| {rname} | {n} | {100 * cs[n - 1]:.1f}% | {100 * c1[k]['hit']:.1f}% | "
                       f"{100 * (cs[n - 1] - c1[k]['hit']):+.1f} |")
    out.append("")
    return out


if __name__ == "__main__":
    main()
