#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the README GPU support matrix from docs/assets/readme-data.json.

    python3 scripts/readme_charts.py          # write the SVGs
    python3 scripts/readme_charts.py --check  # exit 1 if a committed SVG is stale

Stdlib only; output is byte-deterministic. The chart is written in a light and
a dark variant (transparent background) for the README's <picture> element:
docs/assets/readme/gpu-matrix-{light,dark}.svg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/assets/readme-data.json"
OUT = ROOT / "docs/assets/readme"
WIDTH = 880
FONT = "font-family=\"-apple-system,Segoe UI,Helvetica,Arial,sans-serif\""

THEMES = {
    "light": {"fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "tuned": "#d1242f", "generic": "#bf8700", "none": "#d1d9e0"},
    "dark": {"fg": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d",
             "tuned": "#ff7b72", "generic": "#d29922", "none": "#3d444d"},
}


def text(x, y, s, fill, size=13, anchor="start", weight="normal", font=FONT) -> str:
    return (f'<text x="{x}" y="{y}" {font} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')


def svg(height: int, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">\n'
            f"<title>{escape(title)}</title>\n" + "\n".join(body) + "\n</svg>\n")


def matrix(data: dict, t: dict) -> str:
    m = data["matrix"]
    col_x, col_w, row_h = 250, 105, 40
    body, y = [], 28
    body.append(text(0, y, "GPU support matrix", t["fg"], 18, weight="bold"))
    y += 34
    for i, f in enumerate(m["features"]):
        body.append(text(col_x + i * col_w + col_w / 2, y, f, t["fg"], 12, anchor="middle", weight="bold"))
    y += 12
    for a in m["archs"]:
        body.append(f'<line x1="0" y1="{y}" x2="{WIDTH}" y2="{y}" stroke="{t["grid"]}"/>')
        body.append(text(0, y + 18, a["label"], t["fg"], 13, weight="bold"))
        body.append(text(0, y + 33, a["detail"], t["muted"], 11))
        for i, c in enumerate(a["cells"]):
            cx = col_x + i * col_w + col_w / 2
            fill = t[c]
            if c == "none":
                body.append(text(cx, y + 26, "\u2014", t["muted"], 14, anchor="middle"))
            else:
                body.append(f'<rect x="{cx - 34}" y="{y + 11}" width="68" height="20" rx="10" fill="{fill}"/>')
                body.append(text(cx, y + 25, c, "#ffffff", 11, anchor="middle", weight="bold"))
        y += row_h
    body.append(f'<line x1="0" y1="{y}" x2="{WIDTH}" y2="{y}" stroke="{t["grid"]}"/>')
    y += 22
    body.append(text(0, y, "tuned: arch-specific kernels or default-on route \u00b7 generic: portable typed-dispatch path",
                     t["muted"], 11))
    y += 16
    body.append(text(0, y, "Capability, not admission or certification. Sources: docs/assets/readme-data.json.",
                     t["muted"], 11))
    return svg(y + 10, body, "hipfire GPU support matrix")


def main() -> int:
    check = "--check" in sys.argv[1:]
    data = json.loads(DATA.read_text())
    stale = []
    for theme, t in THEMES.items():
        path = OUT / f"gpu-matrix-{theme}.svg"
        content = matrix(data, t)
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(content)
    if stale:
        print("readme_charts: stale, rerun scripts/readme_charts.py: " + ", ".join(stale))
        return 1
    print("readme_charts: " + ("charts match data" if check else "wrote 2 SVGs"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
