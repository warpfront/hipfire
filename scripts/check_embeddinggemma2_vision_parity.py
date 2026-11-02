#!/usr/bin/env python3
"""Differential instrument for the EmbeddingGemma-2 *vision* leg.

Splits one media leg's error into its three possible sources, without the
trunk, the daemon or the `.vl` quantization in the way:

1. **processor** — hipfire `image::process_image` vs HF `AutoProcessor`
   (`pixel_values` / `image_position_ids` / patch grid / soft-token count);
2. **tower** — hipfire `VisionTower::encode_image` rows vs HF
   `Gemma4VisionModel` + the `embed_vision` projection, both with the real
   checkpoint weights;
3. **operand precision** — the same HF rows with every `nn.Linear`'s operands
   rounded through f16 (hipfire's WMMA contract: f16 operands, f32 accumulate,
   f32 attention), which is the floor this architecture can reach.

Drives `examples/vision_parity.rs` (lab-gated) for (1) and (2), then reports
the per-case max-abs and rel-L2 of the patches, the exactness of the patch
positions, and the rel-L2 of the projected rows against (2) and (3).

    .venv/bin/python scripts/check_embeddinggemma2_vision_parity.py --model "$SNAP" \\
        [--cases img_square_256 ...] [--quantizer target/release/hipfire-quantize]

The case list and the PNG bytes are imported from
`check_embeddinggemma2_parity.py`, so the instrument always measures exactly
the legs the oracle checks.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_embeddinggemma2_parity import D, IMAGE_CASES, Q, _png_bytes, ensure_vision_sidecar  # noqa: E402


def rel_l2(got, ref):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", default="target/release/hipfire-quantize")
    ap.add_argument("--example", default=None, help="vision_parity binary (default: cargo run)")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--no-f16", action="store_true", help="skip the f16-operand baseline")
    args = ap.parse_args()

    cases = [c for c in IMAGE_CASES if args.cases is None or c[0] in args.cases]
    if not cases:
        raise SystemExit(f"no case matched {args.cases}")

    model_dir = ensure_vision_sidecar(args.model, args.quantizer)
    work = tempfile.mkdtemp(prefix="eg2-vision-parity-")
    pngs = []
    for key, _kind, _text, (h, w, pat) in cases:
        import base64

        path = os.path.join(work, f"{key}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(_png_bytes(h, w, pat)))
        pngs.append(path)

    out_dir = os.path.join(work, "dump")
    cmd = (
        [args.example, model_dir, out_dir, *pngs]
        if args.example
        else [
            "cargo", "run", "--release", "-p", "hipfire-arch-embeddinggemma2",
            "--features", "lab", "--example", "vision_parity", "--",
            model_dir, out_dir, *pngs,
        ]
    )
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"vision_parity failed ({r.returncode}):\n{r.stdout[-2000:]}")
    print(r.stdout, end="")

    # ── HF side: the float32 reference implementation ──────────────────
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoProcessor
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    torch.set_num_threads(os.cpu_count())
    torch.set_grad_enabled(False)

    model = Gemma4VisionModel(Gemma4VisionConfig.from_pretrained(model_dir)).float().eval()
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        sd = {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.startswith("vision_tower.") or k.startswith("embed_vision.")
        }
    model.load_state_dict(
        {k[len("vision_tower.") :]: v for k, v in sd.items() if k.startswith("vision_tower.")},
        strict=True,
    )
    w_proj = sd["embed_vision.embedding_projection.weight"].float()  # [512, 768]
    processor = AutoProcessor.from_pretrained(model_dir)

    _orig_linear = torch.nn.functional.linear
    sim = [False]

    def sim_linear(inp, weight, bias=None):
        if sim[0]:
            return _orig_linear(inp.half().float(), weight.half().float(), bias)
        return _orig_linear(inp, weight, bias)

    torch.nn.functional.linear = sim_linear

    def tail(rows):
        """embed_vision: scale-less RMSNorm (eps 1e-6) then the [512,768] projection."""
        x = rows.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x @ w_proj.t()).numpy()

    def hf_rows(image, f16):
        """The padded pixel_values/positions, plus the tower's pooled rows.
        `f16` rounds every nn.Linear operand through f16 (hipfire's WMMA
        contract) while attention stays f32, like hipfire."""
        sim[0] = f16
        inputs = processor(text=[Q + "x <|image|>"], images=[image], return_tensors="pt")
        rows = model(
            pixel_values=inputs["pixel_values"], pixel_position_ids=inputs["image_position_ids"]
        ).last_hidden_state
        sim[0] = False
        pid = inputs["image_position_ids"][0].numpy()
        real = (pid != -1).all(-1)
        return rows.numpy(), inputs["pixel_values"][0].numpy()[real], pid[real]

    for key, _kind, _text, (_h, _w, _pat) in cases:
        base = os.path.join(out_dir, key)
        meta = __import__("json").load(open(base + ".json"))
        ny, nx = meta["grid_h"], meta["grid_w"]
        patches = np.fromfile(base + ".patches.f32", dtype=np.float32).reshape(
            meta["n_patches"], meta["patch_pixels"]
        )
        positions = np.fromfile(base + ".positions.i32", dtype=np.int32).reshape(-1, 2)
        got_rows = np.fromfile(base + ".rows.f32", dtype=np.float32).reshape(
            meta["rows"], meta["text_hidden"]
        )

        img = Image.open(os.path.join(work, f"{key}.png"))
        hf_raw, hf_patches, hf_pos = hf_rows(img, False)
        hf_ny = int(hf_pos[:, 1].max()) + 1
        hf_nx = int(hf_pos[:, 0].max()) + 1

        print(f"\n{key}")
        geom = "ok" if (ny, nx) == (hf_ny, hf_nx) and meta["n_patches"] == len(hf_patches) else "MISMATCH"
        print(
            f"  geometry  {geom}: grid {ny}x{nx} (hf {hf_ny}x{hf_nx})  "
            f"patches {meta['n_patches']} (hf {len(hf_patches)})  soft {meta['n_tokens']} "
            f"(hf {len(hf_patches) // 9})"
        )
        pos_ok = positions.shape == hf_pos.shape and bool((positions == hf_pos).all())
        print(f"  positions {'exact' if pos_ok else 'DIFFER'}")
        if patches.shape == hf_patches.shape:
            d = np.abs(patches - hf_patches)
            print(
                f"  patches   maxabs {d.max():.3e}  rel-L2 {rel_l2(patches, hf_patches):.3e}  "
                f"frac-equal {float((d == 0).mean()):.4f}"
            )
        else:
            print(f"  patches   shape {patches.shape} vs hf {hf_patches.shape}")

        hf32 = tail(torch.from_numpy(hf_raw))
        if not args.no_f16:
            hf16 = tail(torch.from_numpy(hf_rows(img, True)[0]))
        else:
            hf16 = hf32
        if got_rows.shape == hf32.shape:
            print(
                f"  rows      rel-L2 vs hf-f32 {rel_l2(got_rows, hf32):.3e}"
                + (
                    f"  | f16-operand baseline {rel_l2(hf16, hf32):.3e}"
                    f"  (ratio {rel_l2(got_rows, hf32) / max(rel_l2(hf16, hf32), 1e-12):.1f}x)"
                    if not args.no_f16
                    else ""
                )
            )
        else:
            print(f"  rows      shape {got_rows.shape} vs hf {hf32.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
