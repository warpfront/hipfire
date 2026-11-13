#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Differential test: `serve/ocr.rs` against surya's own compact-block parser.

`crates/hipfire-cli/src/serve/ocr.rs` mirrors `surya.inference.compact.decode`,
`surya.inference.merged_cells.convert_merged_cells` and
`surya.inference.parsers.parse_full_page_html` (surya-ocr 0.23.0, HEAD
`9016f421`). This script is the reproduction of the differential claim: it
builds a deterministic corpus, runs the Python reference end to end, and
compares it with the committed fixture the Rust unit test checks.

The committed fixture is `crates/hipfire-cli/tests/fixtures/ocr_corpus.json`:

    {
      "cases":       [{"input", "html", "blocks"}],   # Rust must equal both
      "divergences": [{"input", "python_html", "rust_html", "reason"}]
    }

`cases` are inputs on which the port is required to be byte-identical.
`divergences` pin the deliberate differences the module header documents
(BeautifulSoup error recovery; Unicode digits), so they cannot drift silently.

Usage (needs the reference modules; a surya checkout at the pinned revision is
enough — `pip install surya-ocr==0.23.0` also works):

    python3 scripts/ocr-differential.py --surya-path /path/to/surya \
        --rust-output /tmp/rust-ocr.json --write-fixture

`--rust-output` is a JSON the Rust side writes (see the `reference_corpus`
unit test's `OCR_DIFF_DUMP` escape hatch): a list of `{"html", "blocks"}`
in corpus order. Without it the script prints the Python results.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates/hipfire-cli/tests/fixtures/ocr_corpus.json"
)

LABELS = [
    "Text",
    "Table",
    "Page-Header",
    "Section-Header",
    "Figure",
    "Caption",
    "Code-Block",
    "Complex-Block",
]


def load_reference(path: Path | None):
    """Import the three pinned reference modules from a surya checkout."""
    if path is not None:
        sys.path.insert(0, str(path))
    from surya.inference import compact, merged_cells, parsers  # noqa: PLC0415

    return compact, merged_cells, parsers


def reference(compact, merged_cells, parsers, text: str) -> tuple[str, str]:
    """The reference pipeline `parse_full_page_html` runs, step by step."""
    cleaned = parsers._strip_fences(text).strip()
    if "<|" in cleaned:
        cleaned = compact.decode(cleaned)
    if "data-kind" in cleaned:
        cleaned = merged_cells.convert_merged_cells(cleaned)
    blocks = parsers.parse_full_page_html(text)
    if not blocks:
        # `decode_html` / `blocks_json` return a no-header answer unchanged.
        return text, text
    out = [
        {
            "label": b.label,
            "bbox": [int(v) for v in b.bbox],
            "html": b.html,
            "truncated": bool(b.truncated),
        }
        for b in blocks
    ]
    return cleaned, json.dumps(out, ensure_ascii=False)


def corpus() -> list[str]:
    """Deterministic inputs: model-shaped pages plus the decoder's grammar.

    Every input is balanced and ASCII-digit where the port is required to
    match; the deliberate divergences are added separately.
    """
    rng = random.Random(2026_11_13)
    cases: list[str] = []

    # Plain answers: no compact header anywhere, so both decoders return the
    # input untouched.
    cases += [
        "Hello.",
        "no markup at all\nwith two lines",
        "a < b and c > d",
        "text with ```inline``` backticks",
        "```\nconsole.log(1)\n```",
        "```python\nprint('hi')\n```",
    ]

    def header(label: str, bbox=(0, 0, 10, 10)) -> str:
        return "<|%s|>" % label + "".join("<|%d|>" % v for v in bbox)

    bodies = [
        "<p>hello</p>",
        "<p>hello</p>\n<p>world</p>",
        "<p>a < b &amp; c > d</p>",
        "<h1>Title</h1><p>body text</p>",
        "<pre>  keep   spaces\n  and lines</pre>",
        "<ul><li>one<li>two</ul>",
        '<p class="x" style="color:red">styled</p>',
        "<p>भारत</p>",
        "<p>literal <भारत after lt</p>",
        "<img src=\"x.png\">",
        '<a href="https://example.com" alt="t">link</a>',
        "<p>self<br/>break</p>",
        "<div>nested <div>inner</div> tail</div>",
        "<p>text</p></div>after the close",
    ]
    for label in LABELS:
        for body in bodies:
            cases.append(header(label) + body)

    # Several blocks, every separator shape.
    cases.append(
        header("Page-Header", (10, 10, 200, 30))
        + "<p>top</p>"
        + header("Text", (10, 40, 900, 900))
        + "<p>body</p>"
        + header("Page-Footer", (10, 950, 200, 990))
        + "<p>foot</p>"
    )
    cases.append(header("Text") + "<p>one</p></div>tail" + header("Text") + "<p>two</p>")

    # Tables: omitted closers, merged cells, truncation, nesting.
    tables = [
        "<table><tr><td>a<td>b<tr><td>c<td>d</table>",
        '<table><tr><td data-kind="parent" data-cs="2">wide'
        '<td data-kind="ghost">x<td data-kind="ghost">y'
        "<tr><td>c<td>d</table>",
        '<table><tr><td data-kind="parent" data-rs="2">tall'
        '<tr><td data-kind="ghost">x</table>',
        "<table><tr><th>h</th><th>h2</th><tr><td>1<td>2</table>",
        "<table><tr><td><table><tr><td>inner</table></table>",
        "<table><tr><td>unterminated",
        "<table><tr><td>a<td>b",
        '<table><tr><td data-kind="parent" data-cs="2" data-rs="2">big'
        '<td data-kind="ghost">x<td data-kind="ghost">y'
        '<tr><td data-kind="ghost">z<td>empty</table>',
    ]
    for table in tables:
        cases.append(header("Table", (7, 7, 700, 700)) + table)

    # A row/image box (`<|box|>`) before a non-block tag.
    cases.append(header("Table") + "<table><tr><|box|><|1|><|2|><|3|><|4|><td>x</table>")
    cases.append(header("Figure") + "<|box|><|5|><|6|><|7|><|8|><img src=\"p.png\">")

    # Fences: a wrapper is stripped only when the answer opened one.
    wrapped = "```html\n" + header("Text") + "<p>wrapped</p>\n```"
    cases.append(wrapped)
    cases.append("```\n" + header("Text") + "<p>a</p>" + header("Text") + "<p>b</p>" + "\n```")
    cases.append(header("Code-Block") + "<pre>code</pre>\n```")
    cases.append("``` \n" + header("Text") + "<p>space after fence</p>")
    cases.append("```" + header("Text") + "<p>no newline</p>```")

    # Truncation: a dangling header / coordinate run, open structural tags.
    cases.append(header("Text") + "<p>cut mid tag")
    cases.append(header("Text") + "<p>cut mid para")
    cases.append(header("Text", (1, 2, 3, 4))[: -len("<|4|>")])
    cases.append(header("Text") + "<p>a</p><|Text|><|1|><|2|>")
    cases.append(header("Text") + "<p>a</p><|Te")

    # Random multi-block documents from the same alphabet.
    for _ in range(60):
        blocks = []
        for _ in range(rng.randint(1, 5)):
            label = rng.choice(LABELS)
            x0 = rng.randint(0, 900)
            y0 = rng.randint(0, 900)
            bbox = (x0, y0, min(1000, x0 + rng.randint(1, 100)), min(1000, y0 + rng.randint(1, 100)))
            blocks.append(header(label, bbox) + rng.choice(bodies))
        cases.append("".join(blocks))

    return cases


def divergences() -> list[str]:
    """Inputs where the port deliberately differs from surya."""
    return [
        # BeautifulSoup re-balances malformed markup; the linear decoder
        # passes it through (documented in the module header).
        "<|Text|><|0|><|0|><|1|><|1|><p><b>unclosed",
        "<|Text|><|0|><|0|><|1|><|1|><td>stray cell",
        "<|Text|><|0|><|0|><|1|><|1|><div><p>crossed</div></p>",
        # Unicode digits: Python's `\d` accepts them, the port's coordinates
        # are the model vocabulary's ASCII 0-1000.
        "<|Text|><|١|><|٢|><|٣|><|٤|><p>arabic-indic</p>",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--surya-path", type=Path, help="surya checkout (or omit for installed surya)")
    ap.add_argument("--rust-output", type=Path, help="JSON the Rust decoder wrote")
    ap.add_argument("--dump-inputs", type=Path, help="write the corpus inputs + divergences as JSON")
    ap.add_argument("--write-fixture", action="store_true", help=f"rewrite {FIXTURE}")
    args = ap.parse_args()

    compact, merged_cells, parsers = load_reference(args.surya_path)
    inputs = corpus()
    if args.dump_inputs is not None:
        args.dump_inputs.write_text(json.dumps(inputs + divergences(), ensure_ascii=False))
        print(f"wrote {len(inputs) + len(divergences())} inputs to {args.dump_inputs}")
        return 0
    python = [reference(compact, merged_cells, parsers, text) for text in inputs]

    rust = None
    if args.rust_output is not None:
        rust = json.loads(args.rust_output.read_text())
        expected = len(inputs) + len(divergences())
        if len(rust) != expected:
            print(f"rust output has {len(rust)} rows, corpus + divergences has {expected}")
            return 1

    if rust is None:
        print(json.dumps([{"input": i, "html": h, "blocks": b} for i, (h, b) in zip(inputs, python)], indent=1))
        return 0

    # The deliberate-divergence inputs are decoded too, and pinned separately.
    extra = divergences()
    python += [reference(compact, merged_cells, parsers, text) for text in extra]
    inputs += extra

    cases, diverged = [], []
    for text, (html, blocks), got in zip(inputs, python, rust):
        try:
            blocks_match = json.loads(got["blocks"]) == json.loads(blocks)
        except json.JSONDecodeError:
            # No block headers: both sides return the answer unchanged.
            blocks_match = got["blocks"] == blocks
        if got["html"] == html and blocks_match:
            # Pin the Rust bytes exactly; the values were just proven equal
            # to the reference.
            cases.append({"input": text, "html": got["html"], "blocks": got["blocks"]})
        else:
            reason = (
                "html"
                if got["html"] != html
                else "beautifulsoup-inner-html"
            )
            diverged.append(
                {
                    "input": text,
                    "python_html": html,
                    "python_blocks": blocks,
                    "rust_html": got["html"],
                    "rust_blocks": got["blocks"],
                    "reason": reason,
                }
            )

    print(f"{len(cases)}/{len(inputs)} identical; {len(diverged)} differ")
    for div in diverged[:10]:
        print(
            f"  [{div['reason']}] {div['input']!r}\n"
            f"    python {div['python_html']!r}\n    rust   {div['rust_html']!r}"
        )

    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(
            json.dumps({"cases": cases, "divergences": diverged}, indent=1, ensure_ascii=False)
            + "\n"
        )
        print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
