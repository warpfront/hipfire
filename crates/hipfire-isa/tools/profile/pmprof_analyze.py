#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC: turn `peacemaker profile` records into phase statistics.

    pmprof_analyze.py <profiled.map.json> <run_dir> <f2|attn|iu4> [out_dir]

Reads run_dir/{trace.bin (or trace.bin.zst),meta.json} written by pmprof. Writes out_dir
(default run_dir) summary.json, summary.md, waves.csv (one row per wave:
placement, realtime span, cycles per phase) and timeline_wg<N>.csv (every
record of every wave of three workgroups, on the WGP's shared cycle clock).

Cycle values are SHADER_CYCLES_LO deltas within one wave (32-bit wrap
handled); realtime is the 100 MHz constant clock from entry/exit records.
"""
import collections, csv, json, os, subprocess, sys
import numpy as np

HW = lambda h: dict(wave=h & 31, simd=(h >> 8) & 3, wgp=(h >> 10) & 15, sa=(h >> 16) & 1, se=(h >> 18) & 7)

# Kernel-specific phase names for (from, to) record pairs. Barrier rules in
# f2 are qualified #1/#2 by their order inside a K128 block.
PHASES = {
    "f2": {
        ("kblock", "w0.pre"): "block: address + 12 VMEM/4 DS issue",
        ("w0.pre", "w0.post"): "block: wait for first weights/fragments",
        ("w0.post", "bar.pre_signal#1"): "slab 0: 32 WMMA + A1 publish",
        ("bar.pre_signal#1", "bar.post_signal#1"): "barrier signal",
        ("bar.post_signal#1", "bar.post_wait#1"): "barrier wait",
        ("bar.post_wait#1", "bar.pre_signal#2"): "slab 1: 32 WMMA + A0 publish + fold",
        ("bar.pre_signal#2", "bar.post_signal#2"): "barrier signal",
        ("bar.post_signal#2", "bar.post_wait#2"): "barrier wait",
        ("bar.post_wait#2", "kblock"): "block tail (s85++, wait_all)",
        ("bar.post_wait#2", "epilogue"): "block tail (s85++, wait_all)",
        ("epilogue", "exit"): "epilogue: SiLU + Y stores",
    },
    "iu4": {
        ("slab", "lds.pre"): "prefetch VMEM + fragment LDS issue",
        ("lds.pre", "lds.post"): "exposed first-fragment LDS wait",
        ("lds.post", "vmem.pre"): "WMMA and staging before first VMEM drain",
        ("vmem.pre", "vmem.post"): "exposed first VMEM wait",
        ("vmem.post", "vmem0.pre"): "staging between VMEM drains",
        ("vmem0.pre", "vmem0.post"): "exposed final VMEM wait",
        ("vmem0.post", "bar.pre_signal#1"): "WMMA / LDS publish before B1",
        ("bar.pre_signal#1", "bar.post_signal#1"): "barrier signal",
        ("bar.post_signal#1", "bar.pre_wait#1"): "signal-to-wait gap",
        ("bar.pre_wait#1", "bar.post_wait#1"): "barrier wait B1",
        ("bar.post_wait#1", "bar.pre_signal#2"): "WMMA / LDS publish before B2",
        ("bar.pre_signal#2", "bar.post_signal#2"): "barrier signal",
        ("bar.post_signal#2", "bar.pre_wait#2"): "K128 fold after B2 signal",
        ("bar.pre_wait#2", "bar.post_wait#2"): "barrier wait B2",
        ("bar.post_wait#2", "slab"): "slab tail",
        ("bar.post_wait#2", "epilogue"): "slab tail to epilogue",
        ("bar.post_wait#1", "epilogue"): "final K128 slab to epilogue",
        ("epilogue", "exit"): "epilogue and stores",
    },
    "attn": {
        ("tile", "fill.done"): "tile fill: LDS commit t+1, issue t+2 K/V loads",
        ("fill.done", "sub"): "sub loop setup",
        ("fill.done", "bar.pre_drain"): "3 subs: QK + softmax + PV (light point set)",
        ("sub", "softmax"): "QK: K fragments (LDS) + 16 WMMA",
        ("softmax", "pv"): "softmax (incl. QK WMMA result latency)",
        ("pv", "sub.end"): "PV: V fragments (LDS) + 16 WMMA",
        ("sub", "sub.end"): "causally skipped sub",
        ("sub.end", "sub"): "sub loop latch",
        ("sub.end", "bar.pre_drain"): "tile latch",
        ("bar.pre_drain", "bar.pre_signal"): "K/V load drain (s_wait_loadcnt_dscnt 0)",
        ("bar.pre_signal", "bar.post_signal"): "barrier signal",
        ("bar.post_signal", "bar.post_wait"): "barrier wait",
        ("bar.post_signal", "bar.pre_wait"): "barrier signal -> wait gap",
        ("bar.pre_wait", "bar.post_wait"): "barrier wait",
        ("bar.post_wait", "tile"): "barrier -> next tile",
        ("bar.post_wait", "epilogue"): "loop exit",
        ("epilogue", "exit"): "epilogue: O normalize + stores",
    },
}
PERIOD = {"f2": "kblock", "attn": "tile", "iu4": "slab"}
# Load issue -> first use: F2 weights/fragments are issued after block start;
# iu4 VMEM is issued at slab start and its first drain follows 16 WMMAs.
ISSUE_USE = {"f2": ("kblock", "w0.post"), "attn": ("fill.done", "bar.pre_signal"),
             "iu4": ("slab", "vmem.post")}
EXPOSED = {"f2": ("w0.pre", "w0.post"), "attn": ("bar.pre_drain", "bar.pre_signal"),
           "iu4": ("vmem.pre", "vmem.post")}
QUALIFY = {"f2": {"bar.pre_signal", "bar.post_signal", "bar.post_wait"}, "attn": set(),
           "iu4": {"bar.pre_signal", "bar.post_signal", "bar.pre_wait", "bar.post_wait"}}


def parse(map_path, run_dir):
    m = json.load(open(map_path))
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    path = os.path.join(run_dir, "trace.bin")
    if os.path.exists(path):
        raw = np.fromfile(path, dtype=np.uint32)
    else:  # archived runs keep trace.bin.zst
        raw = np.frombuffer(subprocess.run(["zstd", "-dc", path + ".zst"], check=True, capture_output=True).stdout, dtype=np.uint32)
    raw = raw.reshape(meta["waves"], -1)
    if not (raw[:, 0] == m["magic"]).all():
        raise SystemExit(f"{(raw[:, 0] != m['magic']).sum()} waves lack the profile header (slot overflow or unlaunched waves)")
    names = {s["id"]: s["rule"] for s in m["sites"]}
    out = []
    flag = m["exit_flag"]
    for w in range(len(raw)):
        h = raw[w, :8]
        recs = raw[w, 8:].reshape(-1, 2)
        ids = recs[:, 1]
        exits = np.nonzero((ids & flag).astype(bool) & (ids != 0xFFFFFFFF))[0]
        if len(exits) == 0:
            raise SystemExit(f"wave {w}: no exit record (slot too small or kernel did not finish)")
        e = exits[0]
        rt_exit = int(recs[e + 1, 1]) << 32 | int(recs[e + 1, 0])
        if e + 2 < len(recs) and ids[e + 2] != 0xFFFFFFFF:
            raise SystemExit(f"wave {w}: records after exit")
        site = (ids[: e + 1] & ~np.uint32(flag)).astype(np.int64)
        cyc = recs[: e + 1, 0].astype(np.int64)
        rel = (cyc - cyc[0]) & 0xFFFFFFFF
        out.append(dict(wave=w, wg=int(h[3]), wiw=int(h[4]), hw=HW(int(h[1])), hw2=int(h[2]),
                        rt0=int(h[6]) << 32 | int(h[5]), rt1=rt_exit, cyc0=int(cyc[0]), cyc_hi=int(h[7]),
                        rules=[names[int(s)] for s in site], rel=rel))
    return m, meta, out


def qualify(kernel, rules):
    period, q = PERIOD[kernel], QUALIFY[kernel]
    seen = collections.Counter()
    out, started = [], False
    for r in rules:
        if r == period:
            seen.clear(); started = True
        if r in q:
            seen[r] += 1
            r = f"{r}#{seen[r]}"
        out.append(r if started or r in ("entry", "exit") else f"pro:{r}")
    return out


def analyze(kernel, m, meta, waves, out_dir):
    phases = PHASES[kernel]
    seg = collections.defaultdict(list)          # phase -> per-instance cycles
    per_wave_rows = []
    total_cycles = 0
    for wv in waves:
        rules = qualify(kernel, wv["rules"])
        rel = wv["rel"]
        d = np.diff(rel)
        wave_phase = collections.Counter()
        first_period = next((i for i, r in enumerate(rules) if r == PERIOD[kernel]), None)
        for i in range(len(d)):
            a, b = rules[i], rules[i + 1]
            if first_period is not None and i < first_period:
                name = "prologue (to first " + PERIOD[kernel] + ")"
            else:
                name = phases.get((a, b), f"other: {a} -> {b}")
            seg[name].append(int(d[i]))
            wave_phase[name] += int(d[i])
        total_cycles += int(rel[-1])
        mhz = rel[-1] / (wv["rt1"] - wv["rt0"]) * 100.0 if wv["rt1"] > wv["rt0"] else float("nan")
        per_wave_rows.append((wv, int(rel[-1]), mhz, wave_phase))
    # A point costs about what the bare signal segment costs: that segment is
    # one point block plus s_barrier_signal. Used only for corrected shares.
    point_cost = float(np.median(seg["barrier signal"])) if "barrier signal" in seg else 0.0
    nseg = sum(len(v) for v in seg.values())
    issue_use, exposed = [], []
    for wv in waves:
        r = wv["rules"]
        for pair, sink in ((ISSUE_USE[kernel], issue_use), (EXPOSED[kernel], exposed)):
            a_idx = [i for i, x in enumerate(r) if x == pair[0]]
            for i in a_idx:
                j = next((j for j in range(i + 1, len(r)) if r[j] == pair[1]), None)
                if j is not None and all(r[k] != pair[0] for k in range(i + 1, j)):
                    sink.append(int(wv["rel"][j] - wv["rel"][i]))
    names = sorted(seg, key=lambda k: -sum(seg[k]))
    table = []
    for k in names:
        v = np.array(seg[k])
        table.append(dict(phase=k, instances=int(len(v)), median=float(np.median(v)),
                          p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                          share=float(v.sum() / total_cycles),
                          share_corrected=float(max(0.0, v.sum() - len(v) * point_cost) / (total_cycles - nseg * point_cost))))

    # SHADER_CYCLES is a per-SIMD counter: each SIMD of a WGP carries a fixed
    # offset. A barrier releases all waves of a workgroup together, so the
    # median post-wait difference to wave 0 calibrates each SIMD. After
    # calibration the release skew is the residual; arrival spread is real.
    by_wg = collections.defaultdict(list)
    for wv in waves:
        by_wg[wv["wg"]].append(wv)
    wrap = lambda a: (a + 2**31) % 2**32 - 2**31
    skews, raw_skews, arrive, offsets_seen = [], [], [], []
    lag_by_rank = collections.defaultdict(list)
    for wg, ws in by_wg.items():
        post, sig = [], []
        for wv in ws:
            ab = wv["cyc0"] + wv["rel"]
            post.append(ab[[i for i, r in enumerate(wv["rules"]) if r == "bar.post_wait"]])
            sig.append(ab[[i for i, r in enumerate(wv["rules"]) if r == "bar.pre_signal"]])
        n = min(len(p) for p in post)
        if n == 0:
            for wv in ws: wv["offset"] = 0
            continue
        P = wrap(np.array([p[:n] for p in post], dtype=np.int64) - np.array(post[0][:n], dtype=np.int64))
        per_wave = np.median(P, axis=1)
        simd_off = {s: float(np.median([per_wave[k] for k, w in enumerate(ws) if w["hw"]["simd"] == s]))
                    for s in {w["hw"]["simd"] for w in ws}}
        for w in ws: w["offset"] = int(round(simd_off[w["hw"]["simd"]]))
        offsets_seen.extend(abs(v) for v in simd_off.values())
        off = np.array([w["offset"] for w in ws], dtype=np.int64)[:, None]
        C = P - off
        raw_skews.extend((P.max(0) - P.min(0)).tolist())
        skews.extend((C.max(0) - C.min(0)).tolist())
        m_ = min(len(s) for s in sig)
        if m_:
            S = wrap(np.array([s[:m_] for s in sig], dtype=np.int64) - np.array(post[0][:1], dtype=np.int64)) - off
            arrive.extend((S.max(0) - S.min(0)).tolist())
            # Arrival lag by the wave's age rank on its SIMD (rank 0 = lowest
            # HW wave slot of this workgroup on that SIMD).
            lag = np.median(S - S.min(0), axis=1)
            for s in {w["hw"]["simd"] for w in ws}:
                on = sorted((k for k, w in enumerate(ws) if w["hw"]["simd"] == s), key=lambda k: ws[k]["hw"]["wave"])
                for rank, k in enumerate(on): lag_by_rank[rank].append(float(lag[k]))

    # Per-WGP spread (realtime, 10 ns ticks).
    wgp = collections.defaultdict(lambda: dict(wgs=set(), busy=0, last=0, first=None, durs=[]))
    t0 = min(w["rt0"] for w in waves)
    t1 = max(w["rt1"] for w in waves)
    for wg, ws in by_wg.items():
        key = (ws[0]["hw"]["se"], ws[0]["hw"]["sa"], ws[0]["hw"]["wgp"])
        s, e = min(w["rt0"] for w in ws), max(w["rt1"] for w in ws)
        g = wgp[key]
        g["wgs"].add(wg); g["busy"] += e - s; g["last"] = max(g["last"], e)
        g["first"] = s if g["first"] is None else min(g["first"], s); g["durs"].append(e - s)
    wgp_rows = sorted(((k, len(g["wgs"]), g["busy"], g["first"] - t0, g["last"] - t0, float(np.median(g["durs"])))
                       for k, g in wgp.items()), key=lambda r: r[0])
    busy = np.array([r[2] for r in wgp_rows], dtype=float)
    lasts = np.array([r[4] for r in wgp_rows], dtype=float)
    wave_mhz = np.array([r[2] for r in per_wave_rows])
    summary = dict(
        diagnostic="peacemaker profile analysis (instrumented kernel)", kernel=kernel, shape=meta["shape"],
        waves=len(waves), workgroups=len(by_wg), records=int(sum(len(w["rules"]) for w in waves)),
        outputs_identical=meta["outputs_identical"], base_ms=meta["base_ms_median"], profiled_ms=meta["profiled_ms_median"],
        overhead=meta["overhead"], kernel_span_us=(t1 - t0) / 100.0, trace_launch_event_us=meta.get("trace_launch_ms", float("nan")) * 1e3,
        wave_cycles_median=float(np.median([r[1] for r in per_wave_rows])),
        shader_clock_mhz_median=float(np.nanmedian(wave_mhz)), shader_clock_mhz_p10_p90=[float(np.nanpercentile(wave_mhz, 10)), float(np.nanpercentile(wave_mhz, 90))],
        simd_clock_offset_cycles=dict(median=float(np.median(offsets_seen)) if offsets_seen else None, max=float(max(offsets_seen)) if offsets_seen else None),
        barrier_release_skew_raw_cycles=dict(median=float(np.median(raw_skews)) if raw_skews else None, p99=float(np.percentile(raw_skews, 99)) if raw_skews else None),
        barrier_release_skew_cycles=dict(median=float(np.median(skews)) if skews else None, p99=float(np.percentile(skews, 99)) if skews else None, n=len(skews)),
        barrier_arrival_lag_by_simd_age_rank=[dict(rank=r, waves=len(lag_by_rank[r]), median=float(np.median(lag_by_rank[r]))) for r in sorted(lag_by_rank)],
        barrier_arrival_spread_cycles=dict(median=float(np.median(arrive)) if arrive else None, p90=float(np.percentile(arrive, 90)) if arrive else None),
        point_cost_cycles=point_cost,
        load_issue_to_use_cycles=dict(pair=list(ISSUE_USE[kernel]), median=float(np.median(issue_use)), p90=float(np.percentile(issue_use, 90)), n=len(issue_use)),
        load_exposed_wait_cycles=dict(pair=list(EXPOSED[kernel]), median=float(np.median(exposed)), p90=float(np.percentile(exposed, 90)), n=len(exposed)),
        wgps=len(wgp_rows), wgp_busy_us=dict(min=busy.min() / 100, median=float(np.median(busy)) / 100, max=busy.max() / 100),
        wgp_last_exit_us=dict(min=lasts.min() / 100, median=float(np.median(lasts)) / 100, max=lasts.max() / 100),
        wgp_workgroups=dict(collections.Counter(r[1] for r in wgp_rows)),
        phases=table,
    )
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    cols = [t["phase"] for t in table]
    with open(os.path.join(out_dir, "waves.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["wave", "wg", "wave_in_wg", "se", "sa", "wgp", "simd", "slot", "rt_entry", "rt_exit", "cycles", "mhz"] + cols)
        for wv, cyc, mhz, ph in per_wave_rows:
            h = wv["hw"]
            wr.writerow([wv["wave"], wv["wg"], wv["wiw"], h["se"], h["sa"], h["wgp"], h["simd"], h["wave"],
                         wv["rt0"] - t0, wv["rt1"] - t0, cyc, f"{mhz:.1f}"] + [ph.get(c, 0) for c in cols])
    for wg in sorted({0, len(by_wg) // 2, len(by_wg) - 1}):
        ws = sorted(by_wg[wg], key=lambda w: w["wiw"])
        base = min(w["cyc0"] - w.get("offset", 0) for w in ws)
        with open(os.path.join(out_dir, f"timeline_wg{wg}.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["wave_in_wg", "simd", "slot", "record", "rule", "cycles_on_wgp_clock (SIMD offsets removed)"])
            for wv in ws:
                for i, (r, t) in enumerate(zip(qualify(kernel, wv["rules"]), wv["rel"])):
                    wr.writerow([wv["wiw"], wv["hw"]["simd"], wv["hw"]["wave"], i, r, ((wv["cyc0"] - wv.get("offset", 0) + int(t)) - base) & 0xFFFFFFFF])
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# {kernel} {meta['shape']} (DIAGNOSTIC profile)\n\n")
        f.write(f"outputs identical: {meta['outputs_identical']}; base {meta['base_ms_median']*1e3:.1f} us, profiled {meta['profiled_ms_median']*1e3:.1f} us, overhead {meta['overhead']*100:+.2f}%\n\n")
        f.write("| Phase | Instances | Median cycles | p10 | p90 | Share of wave time | Share, point cost removed |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for t in table:
            f.write(f"| {t['phase']} | {t['instances']} | {t['median']:.0f} | {t['p10']:.0f} | {t['p90']:.0f} | {t['share']*100:.1f}% | {t['share_corrected']*100:.1f}% |\n")
    return summary


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    m, meta, waves = parse(sys.argv[1], sys.argv[2])
    s = analyze(sys.argv[3], m, meta, waves, sys.argv[4] if len(sys.argv) > 4 else sys.argv[2])
    print(json.dumps({k: v for k, v in s.items() if k != "phases"}, indent=1))
    for t in s["phases"][:20]:
        print(f"{t['share']*100:6.2f}% ({t['share_corrected']*100:6.2f}%)  med {t['median']:9.0f}  n {t['instances']:8d}  {t['phase']}")


if __name__ == "__main__":
    main()
