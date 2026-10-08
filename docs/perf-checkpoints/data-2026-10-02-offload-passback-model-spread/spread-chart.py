#!/usr/bin/env python3
"""Figure for the 2026-10-02 passback model spread. Compares the pass-back gain
over the two single-engine routes across 2B / 9B / 27B on a *spill-fraction* axis,
so models with different layer counts share one x axis.

Data: the spread jsonl files (one row per run, header/footer rows ignored).
Reads from the paths given on the command line, or the defaults below.
"""
import json, statistics, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tag -> (layers, [jsonl files]). Files resolve beside this script so the
# committed figure regenerates from the committed data; argv overrides a tag.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "2b":  (24, ["sp_2b.jsonl"]),
    "9b":  (32, ["sp_9b.jsonl"]),
    "27b": (64, ["sp_27b.jsonl", "sp_27b_extra.jsonl", "sp_27b_60.jsonl"]),
}
MODELS = ["2b", "9b", "27b"]
LABEL = {"2b": "Qwen3.5-2B mq4 (24 layers)",
         "9b": "Qwen3.5-9B mq4 (32 layers)",
         "27b": "Qwen3.8-27B mq4 (64 layers)"}
COLOR = {"2b": "#e67e22", "9b": "#1f77b4", "27b": "#2c3e50"}
MARKER = {"2b": "o", "9b": "s", "27b": "^"}


def series(layers, paths):
    """(spill_fraction, arm) -> (median_decode, values, n). Reads every path."""
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if '"decode"' in l]
    by = {}
    for r in rows:
        if r["decode"] is None:
            continue
        by.setdefault((r["budget"], r["arm"]), []).append(r["decode"])
    out = {}
    for (budget, arm), v in by.items():
        out[(round((layers - budget) / layers, 4), arm)] = (
            statistics.median(v), sorted(v), len(v))
    return out


def main():
    src = {}
    for tag, (layers, files) in DEFAULTS.items():
        i = MODELS.index(tag)
        if len(sys.argv) > 1 + i and sys.argv[1 + i]:
            paths = [sys.argv[1 + i]]
        else:
            paths = [os.path.join(_HERE, f) for f in files]
        src[tag] = series(layers, paths)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    def line(ax, tag, ref, title, ylab):
        s = src[tag]
        xs = sorted({x for (x, m) in s if m == "passback" and (x, ref) in s})
        ys = [100 * (s[(x, "passback")][0] / s[(x, ref)][0] - 1) for x in xs]
        ax.plot([x * 100 for x in xs], ys, marker=MARKER[tag], color=COLOR[tag],
                lw=2, ms=7, label=LABEL[tag], markeredgecolor="white", markeredgewidth=1.1)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.0f}%", (x * 100, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, weight="bold",
                        color=COLOR[tag])

    for tag in MODELS:
        line(axa, tag, "cpu", None, None)
    axa.axhline(0, color="#888", lw=1, ls="--")
    axa.set_title("(a) pass-back gain over `cpu`", fontsize=11.5, loc="left")
    axa.set_xlabel("layers spilled to host RAM (% of model)")
    axa.set_ylabel("decode tok/s gain vs `cpu`")
    axa.grid(alpha=0.3)
    axa.legend(fontsize=9.5, loc="lower right")

    for tag in MODELS:
        line(axb, tag, "pcie", None, None)
    axb.axhline(0, color="#888", lw=1, ls="--")
    axb.set_title("(b) pass-back gain over `pcie`", fontsize=11.5, loc="left")
    axb.set_xlabel("layers spilled to host RAM (% of model)")
    axb.set_ylabel("decode tok/s gain vs `pcie`")
    axb.grid(alpha=0.3)
    axb.legend(fontsize=9.5, loc="lower left")
    for ax in (axa, axb):
        ys = [y for ln in ax.get_lines() for y in ln.get_ydata() if isinstance(y, (int, float))]
        lo, hi = min(ys), max(ys)
        pad = max(4.0, (hi - lo) * 0.22)
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("Offload pass-back vs model size — gfx1201, 2026-10-02, "
                 "pinned gpu_offload_probe prompt, greedy, 128 tokens",
                 fontsize=12, y=1.02)
    out_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_dir}/passback-gain-vs-spill.{ext}", dpi=140, bbox_inches="tight")
    print("rendered")

    # also print the table for the doc
    for tag in MODELS:
        s = src[tag]
        print(f"\n{LABEL[tag]}")
        for x in sorted({x for (x, _) in s}):
            cells = []
            for arm in ("cpu", "passback", "pcie"):
                if (x, arm) in s:
                    med, vals, n = s[(x, arm)]
                    cells.append(f"{arm}={med:.1f}(n={n})")
            g = (s[(x, 'passback')][0] / s[(x, 'cpu')][0] - 1) * 100
            print(f"  {x*100:.1f}% spilled: " + "  ".join(cells) + f"   vs cpu {g:+.1f}%")


main()
