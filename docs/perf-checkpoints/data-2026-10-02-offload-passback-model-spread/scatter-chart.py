#!/usr/bin/env python3
"""Scatter figure for the 2026-10-02 pass-back spreads, for the PR.

Panels:
  (a) paired 1:1  — passback vs cpu decode, one dot per run (all models/quants)
  (b) gain vs spilled BYTES (fraction x model size)   — mechanism test
  (c) gain vs spilled FRACTION                        — the axis that doesn't separate
  (d) gain vs model DEPTH at 25% spilled              — no monotone trend

Reads the committed jsonl beside this script (and the quant dir for mq3).
Model sizes are the on-disk file bytes (a proxy for spilled weight bytes: the
file also carries embed/head/norms, so bytes are approximate).
"""
import json, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.join(_HERE, "..", "data-2026-10-02-offload-passback-quant-spread")

# label -> (files, layers, file_bytes)
SETS = {
    "2B mq4":   (["sp_2b.jsonl"], 24, 1285204608),
    "9B mq4":   (["sp_9b.jsonl"], 32, 5313750016),
    "9B mq3":   ([os.path.join(QUANT, "9b_mq3.jsonl")], 32, 4569785344),
    "27B mq4":  (["sp_27b.jsonl", "sp_27b_extra.jsonl", "sp_27b_60.jsonl"], 64, 15662615552),
}
COLOR = {"2B mq4": "#e67e22", "9B mq4": "#1f77b4", "9B mq3": "#16a085", "27B mq4": "#2c3e50"}
MARKER = {"2B mq4": "o", "9B mq4": "s", "9B mq3": "D", "27B mq4": "^"}


def load(files, layers):
    """-> {budget: {arm: [decode per run]}}"""
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(os.path.join(_HERE, f)) if '"decode"' in l]
    by = {}
    for r in rows:
        if r["decode"] is None:
            continue
        by.setdefault(r["budget"], {}).setdefault(r["arm"], []).append(r["decode"])
    return by


def main():
    data = {lab: load(fs, n) for lab, (fs, n, _) in SETS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axa, axb, axc, axd = axes.ravel()

    # ── (a) paired 1:1 ────────────────────────────────────────────────────
    for lab, by in data.items():
        xs, ys = [], []
        for b, arms in by.items():
            if "cpu" in arms and "passback" in arms:
                xs += sorted(arms["cpu"])
                ys += sorted(arms["passback"])
        axa.scatter(xs, ys, s=34, color=COLOR[lab], marker=MARKER[lab], alpha=0.8,
                    edgecolor="white", linewidth=0.5, label=lab)
    hi = max(v for by in data.values() for arms in by.values() for v in arms.get("passback", []) + arms.get("cpu", []))
    axa.plot([0, hi], [0, hi], color="#888", lw=1, ls="--", label="1:1")
    for g in (10, 20):
        axa.plot([0, hi], [0, hi * (1 + g / 100)], color="#bbb", lw=0.9, ls=":")
        axa.annotate(f"+{g}%", (hi * 0.92, hi * 0.92 * (1 + g / 100)), fontsize=8.5, color="#888")
    axa.set_xlabel("`cpu` decode tok/s (per run)")
    axa.set_ylabel("`passback` decode tok/s (per run)")
    axa.set_title("(a) every run above the 1:1 line — pass-back wins", fontsize=11, loc="left")
    axa.grid(alpha=0.3)
    axa.legend(fontsize=8.5, loc="upper left")

    # ── (b) gain vs spilled bytes  /  (c) gain vs spilled fraction ────────
    for ax, axis_mode in ((axb, "bytes"), (axc, "fraction")):
        for lab, by in data.items():
            layers, size = SETS[lab][1], SETS[lab][2]
            pts = []
            for b, arms in by.items():
                if "cpu" in arms and "passback" in arms and arms["cpu"]:
                    frac = (layers - b) / layers
                    g = 100 * (statistics.median(arms["passback"]) / statistics.median(arms["cpu"]) - 1)
                    x = frac if axis_mode == "fraction" else frac * size / 1e9
                    pts.append((x, g))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKER[lab],
                    color=COLOR[lab], lw=1.6, ms=7, label=lab, markeredgecolor="white")
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_ylabel("pass-back gain vs `cpu` (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
        if axis_mode == "bytes":
            ax.set_xlabel("spilled weight bytes (GB, fraction x model file size)")
            ax.set_title("(b) vs spilled bytes — the curves do not collapse", fontsize=11, loc="left")
        else:
            ax.set_xlabel("layers spilled (% of model)")
            ax.set_title("(c) vs spill fraction — 9B-arch is the peak", fontsize=11, loc="left")

    # ── (d) gain vs model depth at 25% spilled ────────────────────────────
    for lab, by in data.items():
        layers = SETS[lab][1]
        target = round(layers * 0.25)
        b = layers - target                      # budget = layers - spilled
        arms = by.get(b)
        if not arms or "cpu" not in arms or "passback" not in arms:
            continue
        g = 100 * (statistics.median(arms["passback"]) / statistics.median(arms["cpu"]) - 1)
        axd.scatter([layers], [g], s=90, color=COLOR[lab], marker=MARKER[lab],
                    edgecolor="white", zorder=3)
        axd.annotate(f"{lab}\n{g:+.1f}%", (layers, g), textcoords="offset points",
                     xytext=(6, 6), fontsize=9, color=COLOR[lab])
    axd.set_xlabel("model layers")
    axd.set_ylabel("pass-back gain vs `cpu` (%)")
    axd.set_title("(d) at 25 % spilled — no monotone size trend", fontsize=11, loc="left")
    axd.set_xlim(18, 72)
    axd.set_ylim(0, 26)
    axd.grid(alpha=0.3)

    fig.suptitle("Offload pass-back — cross-model / cross-quant scatters — gfx1201, 2026-10-02, "
                 "pinned gpu_offload_probe prompt, greedy, 128 tokens",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{_HERE}/passback-scatter.{ext}", dpi=140, bbox_inches="tight")
    print("rendered")

    ndots = sum(len(arms["cpu"]) for by in data.values() for arms in by.values()
                if "cpu" in arms and "passback" in arms)
    print(f"\n(a) paired: {ndots} per-run dots")
    for lab, by in data.items():
        for b, arms in sorted(by.items(), reverse=True):
            if "cpu" in arms and "passback" in arms and arms["cpu"]:
                frac = (SETS[lab][1] - b) / SETS[lab][1]
                print(f"  {lab:8} b{b:<3} {frac*100:4.1f}% spilled  "
                      f"cpu={statistics.median(arms['cpu']):5.1f} "
                      f"pb={statistics.median(arms['passback']):5.1f}  "
                      f"gain={100*(statistics.median(arms['passback'])/statistics.median(arms['cpu'])-1):+5.1f}%")


main()
