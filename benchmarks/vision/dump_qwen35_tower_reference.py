#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Independent CPU oracle for the Qwen3.5/3.8-VL vision tower (no text model).

Loads ONLY the 333 `model.visual.*` tensors of an HF checkpoint (via the
safetensors index, opening just the shard(s) that hold them), builds the
transformers `Qwen3_5VisionModel` with eager attention on CPU, runs one image
through `Qwen2VLImageProcessor` + the tower and writes raw F32 patches and
stage activations in the layout produced by `vl_tower_dump` /
`HIPFIRE_VL_DUMP_DIR` so they can be diffed:

  pixel_values.bin (F32 LE [n, 1536]), grid_thw.npy, patch_embed.npy,
  post_pos_embed.npy, block_NN.npy, pre_merger.npy, post_merger.npy/.bin,
  receipt.json

Compare a hipfire run against a reference without re-running torch:
  dump_qwen35_tower_reference.py --compare-only REFDIR CANDDIR
or right after a reference run: ... --compare CANDDIR

Requires torch, transformers>=5 (tested 5.8.0), safetensors, numpy, Pillow.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

EXPECTED_TENSORS = 333
VISUAL_PREFIX = "model.visual."


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metrics(candidate, reference):
    import numpy as np

    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    ref_l1 = max(float(np.abs(ref64).sum()), 1e-30)
    ref_l2 = max(float(np.linalg.norm(ref64)), 1e-30)
    cand_l2 = max(float(np.linalg.norm(cand64)), 1e-30)
    return {
        "status": "measured",
        "max_abs": float(np.abs(delta).max()),
        "relative_l1": float(np.abs(delta).sum() / ref_l1),
        "relative_l2": float(np.linalg.norm(delta) / ref_l2),
        "cosine": float(np.vdot(cand64, ref64) / (cand_l2 * ref_l2)),
    }


def diff_blob(reference, candidate_path):
    """Diff a reference ndarray against a raw F32 LE candidate file."""
    import numpy as np

    if not candidate_path.exists():
        return {"status": "missing"}
    candidate = np.fromfile(candidate_path, dtype="<f4")
    if candidate.size != reference.size:
        return {"status": "shape_mismatch", "candidate_elems": int(candidate.size),
                "reference_elems": int(reference.size)}
    if not np.isfinite(candidate).all():
        return {"status": "non_finite"}
    return metrics(candidate.reshape(reference.shape), reference)


def compare_dirs(ref_dir, cand_dir, stages=None):
    """Compare every stage present in `cand_dir` against `ref_dir`.

    Reference stages come from `<name>.npy` (post_merger/pixel_values also from
    the .bin files). Returns {name: metrics}.
    """
    import numpy as np

    differences = {}
    if stages is None:
        stages = sorted(p.stem for p in ref_dir.glob("*.npy") if p.stem != "grid_thw")
        stages.append("pixel_values")
    for name in stages:
        npy = ref_dir / f"{name}.npy"
        if name == "pixel_values":
            reference = np.fromfile(ref_dir / "pixel_values.bin", dtype="<f4")
            n = reference.size // 1536
            reference = reference.reshape(n, 1536)
        elif npy.exists():
            reference = np.load(npy)
        else:
            continue
        if name == "pixel_values":
            # The hipfire stage dump and the --image patch dump both qualify.
            for cand_name, key in (("pixel_values.bin", "pixel_values"), ("patches.bin", "patches")):
                if (cand_dir / cand_name).exists():
                    differences[key] = diff_blob(reference, cand_dir / cand_name)
            continue
        differences[name] = diff_blob(reference, cand_dir / f"{name}.bin")
    if "post_merger" not in differences:
        differences["post_merger"] = {"status": "missing"}
    return differences


def summarize(differences):
    m = differences.get("post_merger", {})
    if m.get("status") != "measured":
        print(f"post_merger {m.get('status', 'missing')}", flush=True)
    else:
        print(f"post_merger rel_l2={m['relative_l2']:.4e} cos={m['cosine']:.8f} "
              f"max_abs={m['max_abs']:.4e}", flush=True)
    if "patches" in differences and differences["patches"].get("status") == "measured":
        p = differences["patches"]
        print(f"patches rel_l2={p['relative_l2']:.4e} cos={p['cosine']:.8f} "
              f"max_abs={p['max_abs']:.4e}", flush=True)
    for name in sorted(differences):
        m = differences[name]
        if name in ("post_merger", "patches") or m.get("status") != "measured":
            continue
        print(f"  {name}: rel_l2={m['relative_l2']:.4e} cos={m['cosine']:.8f} "
              f"max_abs={m['max_abs']:.4e}", flush=True)


def load_visual_state(parent, torch):
    from safetensors import safe_open

    index = json.loads((parent / "model.safetensors.index.json").read_text())["weight_map"]
    keys = sorted(k for k in index if k.startswith(VISUAL_PREFIX))
    if len(keys) != EXPECTED_TENSORS:
        raise ValueError(f"Expected exactly {EXPECTED_TENSORS} {VISUAL_PREFIX}* tensors in the index, "
                         f"found {len(keys)}; refusing")
    by_shard = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    state = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(str(parent / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                state[key.removeprefix(VISUAL_PREFIX)] = handle.get_tensor(key)
    if len(state) != EXPECTED_TENSORS:
        raise ValueError(f"Loaded {len(state)} tensors, expected {EXPECTED_TENSORS}")
    return state, sorted(by_shard)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, default=Path("/home/kaden/qcal/parents/qwen3.8-27b"),
                        help="HF checkpoint dir (config.json, preprocessor_config.json, "
                             "model.safetensors.index.json + shards)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-pixels", type=int, default=2000000,
                        help="size.longest_edge fed to the HF image processor (hipfire product cap)")
    parser.add_argument("--compare", type=Path, help="vl_tower_dump output dir to diff after the reference run")
    parser.add_argument("--compare-only", nargs=2, type=Path, metavar=("REFDIR", "CANDDIR"),
                        help="Skip the model; diff CANDDIR against a previous reference REFDIR")
    args = parser.parse_args()

    if args.compare_only:
        ref_dir, cand_dir = args.compare_only
        differences = compare_dirs(ref_dir, cand_dir)
        (cand_dir / "compare.json").write_text(json.dumps(
            {"reference": str(ref_dir), "candidate": str(cand_dir), "stage_differences": differences},
            indent=2) + "\n")
        summarize(differences)
        return
    if args.image is None or args.out is None:
        parser.error("--image and --out are required (unless --compare-only)")

    import numpy as np
    import torch
    import transformers
    from PIL import Image
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    except ImportError:
        # transformers>=5 puts the torchvision-backed processor in the default module;
        # the PIL backend (BICUBIC, same smart-resize/normalize) needs no torchvision.
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil as Qwen2VLImageProcessor,
        )
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.parent / "config.json"
    preprocessor_path = args.parent / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    processor_config = json.loads(preprocessor_path.read_text())
    processor_config["size"]["longest_edge"] = args.max_pixels
    processor = Qwen2VLImageProcessor(**processor_config)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    patches, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    np.save(args.out / "pixel_values.npy", patches.float().numpy())
    patches.float().numpy().astype("<f4").tofile(args.out / "pixel_values.bin")
    np.save(args.out / "grid_thw.npy", grid.numpy())
    print(f"patch grid {grid.tolist()}, loading vision weights only", flush=True)

    cfg = Qwen3_5VisionConfig(**config["vision_config"])
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        visual = Qwen3_5VisionModel(cfg)
    source_state, shards = load_visual_state(args.parent, torch)
    visual.load_state_dict(source_state, strict=True, assign=True)
    del source_state
    # RoPE's non-persistent inv_freq buffer is absent from the state dict and
    # stays on the meta device; recreate that small module on CPU.
    head_dim = cfg.hidden_size // cfg.num_heads
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(head_dim // 2)
    if any(value.is_meta for value in visual.buffers()) or any(value.is_meta for value in visual.parameters()):
        raise ValueError("Unmaterialized reference tensor after loading the vision tower")
    visual.to(device="cpu", dtype=getattr(torch, args.dtype)).eval()
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().numpy().copy()
        return hook

    def before_first_block(_module, inputs):
        captured["post_pos_embed"] = inputs[0].detach().float().numpy().copy()

    hooks = [visual.patch_embed.register_forward_hook(capture("patch_embed")),
             visual.blocks[0].register_forward_pre_hook(before_first_block),
             visual.merger.register_forward_hook(capture("post_merger"))]
    for number, block in enumerate(visual.blocks):
        hooks.append(block.register_forward_hook(capture(f"block_{number:02d}")))
    start = time.perf_counter()
    with torch.inference_mode():
        output = visual(patches.to(getattr(torch, args.dtype)), grid_thw=grid)
    seconds = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    features = output.pooler_output.detach().float().numpy()
    expected = (int(grid.prod()) // 4, cfg.out_hidden_size)
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f"Invalid projected image features: {features.shape}, expected {expected}")
    captured["pre_merger"] = captured[f"block_{cfg.depth - 1:02d}"]
    for name, value in captured.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite reference stage {name}")
        np.save(args.out / f"{name}.npy", value)
    features.astype("<f4").tofile(args.out / "post_merger.bin")
    receipt = {
        "schema": "hipfire.qwen35vl.tower_reference.v1", "status": "reference_complete",
        "parent": str(args.parent), "shards": shards, "visual_tensors": EXPECTED_TENSORS,
        "config_sha256": sha256(config_path), "preprocessor_sha256": sha256(preprocessor_path),
        "image_sha256": sha256(args.image),
        "patches_sha256": sha256(args.out / "pixel_values.bin"),
        "projected_sha256": sha256(args.out / "post_merger.bin"),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "transformers_modeling_sha256": sha256(inspect.getfile(Qwen3_5VisionModel)),
        "transformers_configuration_sha256": sha256(inspect.getfile(Qwen3_5VisionConfig)),
        "transformers_processor_sha256": sha256(inspect.getfile(Qwen2VLImageProcessor)),
        "device": "cpu", "dtype": args.dtype, "threads": args.threads,
        "max_pixels": args.max_pixels, "grid_thw": grid.tolist(),
        "projected_shape": list(features.shape), "forward_seconds": seconds,
        "quality_acceptance": False,
    }
    if args.compare:
        differences = compare_dirs(args.out, args.compare)
        receipt["compare_dir"] = str(args.compare)
        receipt["stage_differences"] = differences
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if args.compare:
        summarize(receipt["stage_differences"])


if __name__ == "__main__":
    main()
