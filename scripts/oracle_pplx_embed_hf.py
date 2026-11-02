#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""HF-`transformers` reference for the arch-17 embedding model — the third leg.

`oracle_pplx_embed.py` is a NumPy reimplementation written from the HF source.
It agreeing with hipfire proves hipfire implements *that reading*; it cannot
prove the reading is right, because a misread convention would be wrong in both
and still agree. This script closes that hole: it runs the checkpoint through
real `transformers` `Qwen3_5Model` (`is_causal=False` per its config) and
reports a 3-way comparison — HF vs the NumPy oracle vs hipfire — first on the
trunk's `last_hidden_state`, then on the final 128-d embeddings.

Requires `torch` (CPU is enough; the model is 0.6B). transformers >= 5.4 has
the architecture natively. Suggested setup, which leaves the system python
alone:

    python3 -m venv --system-site-packages /tmp/hfref-venv
    /tmp/hfref-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/hfref-venv/bin/python scripts/oracle_pplx_embed_hf.py --model <dir> \\
        [--daemon target/release/daemon]

Reading the result: HF is f32, the NumPy oracle is f64, hipfire is f16 weights.
So expect ~1e-5 for HF-vs-oracle (precision alone) and ~1e-4..1e-3 for
HF-vs-hipfire. What matters is the *shape* of the agreement — an O(1) gap on the
trunk means a misread convention, not precision.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mask_kept_ids(tok, ids, kind, marker):
    """The token ids the daemon keeps after MultiVectorMask (document side)."""
    if kind != "document":
        return ids
    skip = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    keep = []
    for i, t in enumerate(ids):
        piece = (tok.id_to_token(t) or "").lstrip("\u0120\u010a")
        if i == 0 or t == marker or piece not in skip:
            keep.append(t)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--skip-hipfire", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel
    from tokenizers import Tokenizer

    os.environ["ORACLE_MODEL"] = args.model
    from oracle_pplx_embed import encode as oracle_encode  # noqa: E402

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    print(f"loading HF Qwen3_5Model from {args.model} (f32, CPU) ...")
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    print(f"  class: {type(model).__name__}")

    # The Dense head (sentence-transformers module 1) is separate from the trunk.
    from safetensors.numpy import load_file

    dense = load_file(os.path.join(args.model, "1_Dense", "model.safetensors"))["linear.weight"]

    fixtures = [
        ("d", "document", "Retrieval-augmented generation grounds a model's answer."),
        ("q", "query", "How does late interaction score a passage?"),
    ]
    markers = {"query": tok.token_to_id("[Q] "), "document": tok.token_to_id("[D] ")}

    rows = []
    for name, kind, text in fixtures:
        ids = [markers[kind]] + tok.encode(text).ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], dtype=torch.long))
        hf_trunk = out.last_hidden_state[0].to(torch.float32).numpy()
        hf_final = hf_trunk @ dense.T
        hf_final /= np.linalg.norm(hf_final, axis=1, keepdims=True)

        or_trunk = oracle_encode(np.array(ids), hidden_only=True)
        or_final = oracle_encode(np.array(ids))
        or_final /= np.linalg.norm(or_final, axis=1, keepdims=True)

        def rel(a, b):
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))

        r_trunk = rel(or_trunk, hf_trunk)
        r_final = rel(or_final, hf_final)
        r_hip = None
        if not args.skip_hipfire:
            req = [
                json.dumps({"type": "load", "model": os.path.abspath(args.model), "params": {}}),
                json.dumps({"type": "embed", "id": name, "kind": kind, "input": {"text": text}}),
                json.dumps({"type": "unload"}),
            ]
            p = subprocess.run(
                [args.daemon], input="\n".join(req) + "\n",
                capture_output=True, text=True, timeout=1800,
            )
            got = None
            for line in p.stdout.splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "embedded":
                    got = d
                elif d.get("type") == "error":
                    raise SystemExit(f"daemon error: {d.get('message')}")
            if got is None:
                raise SystemExit(f"{name}: no embedded event from the daemon")
            keep = mask_kept_ids(tok, ids, kind, markers[kind])
            keep_mask = [t in keep for t in ids]
            arr = np.array(got["embeddings"]).reshape(got["n_tokens"], got["dim"])
            r_hip = rel(arr, hf_final[keep_mask])
        rows.append((name, r_trunk, r_final, r_hip))

    print()
    print(f"{'fixture':9s} {'HF vs oracle trunk':>19s} {'HF vs oracle final':>19s} {'HF vs hipfire final':>20s}")
    worst_trunk = worst_final = worst_hip = 0.0
    for name, rt, rf, rh in rows:
        hip = "—" if rh is None else f"{rh:.3e}"
        print(f"{name:9s} {rt:19.3e} {rf:19.3e} {hip:>20s}")
        worst_trunk = max(worst_trunk, rt)
        worst_final = max(worst_final, rf)
        if rh is not None:
            worst_hip = max(worst_hip, rh)
    print()
    # Gates: the trunk comparison proves the oracle reproduces HF's trunk;
    # the final comparison adds the Dense head (a misread head would pass
    # every downstream gate, since they all compare through it); the hipfire
    # comparison proves the daemon's post-processing matches HF end-to-end.
    # Same tolerance class as the trunk gate — these are precision checks,
    # not quality budgets.
    ok_trunk = worst_trunk < 1e-3
    ok_final = worst_final < 1e-3
    ok_hip = worst_hip < 1e-2 if not args.skip_hipfire else True
    ok = ok_trunk and ok_final and ok_hip
    print(f"trunk agreement (worst {worst_trunk:.3e}) vs 1e-3: {'PASS' if ok_trunk else 'FAIL — oracle misreads the architecture'}")
    print(f"final-head agreement (worst {worst_final:.3e}) vs 1e-3: {'PASS' if ok_final else 'FAIL — oracle misreads the Dense head'}")
    if not args.skip_hipfire:
        print(f"hipfire end-to-end (worst {worst_hip:.3e}) vs 1e-2: {'PASS' if ok_hip else 'FAIL — daemon diverges from HF'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
