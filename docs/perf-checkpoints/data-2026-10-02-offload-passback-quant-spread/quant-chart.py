#!/usr/bin/env python3
"""Figure for the 2026-10-02 pass-back quant spread (9B, mq3 vs mq4, 8 and 16 of
32 layers spilled). Reads the committed jsonl beside this script plus the mq4
rows from the sibling model-spread data dir.

    python3 quant-chart.py            # writes quant-vs-engine.{png,svg} here
"""
import json, statistics, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
MQ4 = os.path.join(_HERE, "..", "data-2026-10-02-offload-passback-model-spread", "sp_9b.jsonl")
MQ3 = os.path.join(_HERE, "9b_mq3.jsonl")
LAYERS = 32
ARMS = ["cpu", "passback", "pcie"]
COLOR = {"cpu": "#c0392b", "passback": "#1f77b4", "pcie": "#7f8c8d"}


def series(path, tag=None, budgets=(24, 16)):
    rows = [json.loads(l) for l in open(path) if '"decode"' in l]
    by = {}
    for r in rows:
        if r["decode"] is None or r["budget"] not in budgets:
            continue
        by.setdefault((r["budget"], r["arm"]), []).append(r["decode"])
    return {k: (statistics.median(v), len(v)) for k, v in by.items()}


def main():
    src = {"mq3": series(MQ3), "mq4": series(MQ4)}
    spills = (24, 16)                      # 8/32 and 16/32 spilled
    groups = [(q, b) for q in ("mq3", "mq4") for b in spills]

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    w = 0.26
    xs = range(len(groups))
    for i, arm in enumerate(ARMS):
        vals = [src[q].get((b, arm), (0, 0))[0] for q, b in groups]
        ax.bar([x + (i - 1) * w for x in xs], vals, w, label=arm, color=COLOR[arm])
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}", (x + (i - 1) * w, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{q}  {LAYERS-b} of {LAYERS} spilled\n({(LAYERS-b)/LAYERS:.0%})"
                        for q, b in groups], fontsize=9.5)
    ax.set_ylabel("decode tok/s")
    ax.set_title("(a) 9B, same size, two quants — decode by engine",
                 fontsize=11.5, loc="left")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9.5)

    g_cpu = [100 * (src[q][(b, "passback")][0] / src[q][(b, "cpu")][0] - 1) for q, b in groups]
    g_pcie = [100 * (src[q][(b, "passback")][0] / src[q][(b, "pcie")][0] - 1) for q, b in groups]
    axg.bar([x - w for x in xs], g_cpu, 2 * w, label="vs `cpu`", color="#1f77b4")
    axg.bar([x + w for x in xs], g_pcie, 2 * w, label="vs `pcie`", color="#2ca02c")
    for x, v in zip(xs, g_cpu):
        axg.annotate(f"{v:+.0f}%", (x - w, v), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=9, weight="bold", color="#1f77b4")
    for x, v in zip(xs, g_pcie):
        axg.annotate(f"{v:+.0f}%", (x + w, v), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=9, weight="bold", color="#2ca02c")
    axg.axhline(0, color="#888", lw=1, ls="--")
    lo, hi = min(g_cpu + g_pcie), max(g_cpu + g_pcie)
    axg.set_ylim(lo - abs(lo) * 0.35 - 5, hi * 1.18)
    axg.set_xticks(list(xs))
    axg.set_xticklabels([f"{q}\n{LAYERS-b} of {LAYERS} spilled" for q, b in groups], fontsize=9.5)
    axg.set_ylabel("pass-back decode gain")
    axg.set_title("(b) pass-back's gain, by baseline", fontsize=11.5, loc="left")
    axg.grid(alpha=0.3, axis="y")
    axg.legend(fontsize=9.5, loc="upper left")

    fig.suptitle("Offload pass-back vs quant at one size — gfx1201, 2026-10-02, "
                 "pinned gpu_offload_probe prompt, greedy, 128 tokens",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{_HERE}/quant-vs-engine.{ext}", dpi=140, bbox_inches="tight")
    print("rendered")
    for q, b in groups:
        cells = "  ".join(f"{a}={src[q].get((b,a),(float('nan'),0))[0]:.1f}"
                          f"(n={src[q].get((b,a),(0,0))[1]})" for a in ARMS)
        print(f"  {q} {LAYERS-b}/{LAYERS} spilled: {cells}")


main()
