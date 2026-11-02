#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Quantization correctness matrix for the arch-17 embedding model.

For every artifact given (a safetensors dir or an `.hfq` in any format), loads
it in the daemon, encodes fixed byte-identical prompts, and compares the
result against the torch-free NumPy oracle (`scripts/oracle_pplx_embed.py`).
Each artifact must land inside its format's error band; a broken dequant arm
does not produce "a slightly worse embedding", it produces O(1) error, so the
bands are set an order of magnitude below that and well above the format's
real noise.

The band is a function of *quantization* error only. Every artifact is compared
against the same f32 oracle, so the numbers are also a direct bit-width ladder:
q8 ~5e-3, 6-bit ~1.5e-2, 4-bit ~6e-2 on this fixture.

Usage:
  scripts/check_embed_quant.py --oracle-model <dir> [--daemon <bin>] \\
      <artifact> [<artifact> ...] [--json]

  # e.g.
  scripts/check_embed_quant.py --oracle-model ~/.hipfire/models/pplx-embed-v2-late-0.6b \\
      ~/.hipfire/models/pplx-embed-v2-late-0.6b /tmp/qmatrix/*.hfq
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def artifact_format(path):
    """The artifact's real quant format, read from the file — not its name.

    `.hfq`: the dominant (most common) non-raw tensor quant_type from the
    HFQ tensor index, via astrea's parser. A safetensors dir is the f16
    source path. Anything else (or an unreadable index) is None.
    """
    if os.path.isdir(path):
        return "dir"
    try:
        import astrea
        summary, _tensor_map = astrea.read_hfq_index(path, max_tensors=0)
        counts = summary.get("quant_type_counts", {})
        # Drop raw formats (F16/F32) when compressed tensors dominate — the
        # embed trunk is what matters.
        compressed = {k: v for k, v in counts.items() if k not in ("F16", "F32")}
        dom = compressed or counts
        return max(dom, key=dom.get) if dom else None
    except Exception:
        return None


# Quant-type → (label, rel-L2 band). Bands follow the format's measured
# error against the f32 oracle (see the module docstring's ladder) and sit an
# order below the O(1) error a broken dequant produces.
FORMAT_BANDS = {
    "MQ4G256": ("mq4 (4-bit MQ4G256V2)", 2.5e-1),
    "MQ4G256V2": ("mq4 (4-bit MQ4G256V2)", 2.5e-1),
    "MQ3G256": ("mq3 (3-bit MQ3G256)", 2.5e-1),
    "MQ3G256_LLOYD": ("mq3 (3-bit Lloyd)", 2.5e-1),
    "MQ8G256": ("q8/mq8 (8-bit)", 3.0e-2),
    "MQ6G256": ("mq6 (6-bit MQ6G256)", 8.0e-2),
    "MQ6G256V2": ("mq6 (6-bit MQ6G256V2)", 8.0e-2),
    "HFQ4G256": ("hf4 (4-bit HFQ4G256)", 2.5e-1),
    "HFQ4G128": ("hf4 (4-bit HFQ4G128)", 2.5e-1),
    "HFQ6G256": ("hf6 (6-bit HFQ6G256)", 8.0e-2),
    "Q8F16": ("q8 (8-bit Q8F16)", 3.0e-2),
    "HFP4G32": ("hfp4 (4-bit FP)", 2.5e-1),
    "MFP4G32": ("mfp4 (4-bit FP)", 2.5e-1),
    "dir": ("raw / safetensors (f16 path, f32 source)", 2.0e-3),
}


def band_for(path):
    """(name, max rel-L2) expected for this artifact — from the artifact's
    own format, never its filename (a renamed .hfq or an f16 dir whose name
    happens to contain 'mq4' must not pick the wrong band)."""
    fmt = artifact_format(path)
    if fmt is None:
        return None, None
    return FORMAT_BANDS.get(fmt, (f"{fmt} (unbanded)", None))

def run_daemon(daemon, model, prompts):
    """Load `model` once and return {name: (embeddings, n_tokens, dim)}."""
    req = [json.dumps({"type": "load", "model": model, "params": {}})]
    for name, kind, text in prompts:
        req.append(json.dumps({"type": "embed", "id": name, "kind": kind, "input": {"text": text}}))
    req.append(json.dumps({"type": "unload"}))
    proc = subprocess.run(
        [daemon], input="\n".join(req) + "\n", capture_output=True, text=True, timeout=1800
    )
    out = {}
    err = None
    for line in proc.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "embedded":
            out[d["id"]] = (d["embeddings"], d["n_tokens"], d["dim"])
        elif d.get("type") == "error":
            err = d.get("message")
    if err and not out:
        raise SystemExit(f"daemon error: {err}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-model", required=True, help="the f32/f16 reference dir")
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("artifacts", nargs="+")
    args = ap.parse_args()

    import numpy as np

    # The oracle resolves its checkpoint from $ORACLE_MODEL at import time.
    os.environ["ORACLE_MODEL"] = args.oracle_model
    from oracle_pplx_embed import encode  # noqa: E402
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.oracle_model, "tokenizer.json"))
    # Fixtures are `(name, kind, text)`. The marker is NOT in `text`: the
    # daemon prepends it from `kind`, so the oracle must build the same id
    # sequence by hand (`[marker] + encode(text)`) or the two would differ by a
    # token and the comparison would be meaningless.
    # `dl` crosses the 256-token boundary so the eager attention path
    # (attention_gemm: QK/PV WMMA GEMMs + masked row softmax + padding) is
    # exercised, not only the small-n Q-tiled kernel — a numerically wrong
    # fused op on that path is invisible on short fixtures.
    prompts = [
        ("d", "document", "Retrieval-augmented generation grounds a model's answer."),
        ("q", "query", "How does late interaction score a passage?"),
        ("dl", "document", " ".join(
            f"sentence {i} about dense retrieval and late interaction"
            for i in range(40)
        )),
    ]
    markers = {"query": tok.token_to_id("[Q] "), "document": tok.token_to_id("[D] ")}
    for k, v in markers.items():
        if v is None:
            raise SystemExit(f"tokenizer has no {k} marker token")
    # The MultiVectorMask + Normalize modules run on the host AFTER the Dense
    # head, and the daemon applies them: the document side drops the token
    # vectors whose piece is a skiplist word (punctuation), every side
    # L2-normalizes each kept vector. The oracle stops at the Dense head, so it
    # must apply the same mask here or the two would differ by rows.
    # Mirrors `post::skiplist_token_ids` + `post::skiplist_mask`; the marker is
    # protected.
    skiplist = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    def kept_ids(kind, ids):
        if kind != "document":
            return ids
        out = []
        for i, t in enumerate(ids):
            piece = (tok.id_to_token(t) or "").lstrip("\u0120\u010a")
            if i == 0 or t == markers[kind] or piece not in skiplist:
                out.append(t)
        return out

    ref, ref_kept = {}, {}
    for name, kind, text in prompts:
        ids = [markers[kind]] + tok.encode(text).ids
        e = encode(np.array(ids))
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        keep = kept_ids(kind, ids)
        if len(keep) != len(ids):
            # Re-select the rows the daemon will keep.
            mask = [t in keep for t in ids]
            e = e[mask]
        ref[name] = e
        ref_kept[name] = len(keep)

    rows, failed = [], 0
    for art in args.artifacts:
        label, band = band_for(art)
        if band is None:
            rows.append({"artifact": art, "format": label or "unreadable",
                         "rel_l2": None,
                         "error": "no band for this format — unknown artifact or "
                                  "unbanded quant_type; refusing to guess"})
            failed += 1
            continue
        try:
            got = run_daemon(args.daemon, os.path.abspath(art), prompts)
        except SystemExit as e:
            rows.append({"artifact": art, "format": label, "rel_l2": None, "error": str(e)})
            failed += 1
            continue
        worst = 0.0
        for name, _kind, _text in prompts:
            if name not in got:
                rows.append({"artifact": art, "format": label, "rel_l2": None,
                             "error": "missing embedded event"})
                failed += 1
                break
            emb, n, dim = got[name]
            arr = np.array(emb).reshape(n, dim)
            if n != ref_kept[name]:
                rows.append({"artifact": art, "format": label, "rel_l2": None,
                             "error": f"{name}: {n} tokens vs oracle {ref_kept[name]} (mask)"})
                failed += 1
                break
            r = float(np.linalg.norm(arr - ref[name]) / np.linalg.norm(ref[name]))
            worst = max(worst, r)
        else:
            ok = worst <= band
            failed += 0 if ok else 1
            rows.append({"artifact": art, "format": label, "rel_l2": worst,
                         "band": band, "ok": ok})

    if args.json:
        print(json.dumps({"rows": rows, "failed": failed}, indent=2))
    else:
        print(f"{'artifact':34s} {'format':34s} {'rel_l2':>10s} {'band':>9s}  result")
        for r in rows:
            art = os.path.basename(r["artifact"])[:34]
            if r.get("rel_l2") is None:
                print(f"{art:34s} {r['format']:34s} {'—':>10s} {'—':>9s}  FAIL ({r.get('error')})")
            else:
                print(f"{art:34s} {r['format']:34s} {r['rel_l2']:10.2e} {r['band']:9.1e}"
                      f"  {'PASS' if r['ok'] else 'FAIL'}")
        print(f"\n{len(rows) - failed}/{len(rows)} artifacts within band")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
