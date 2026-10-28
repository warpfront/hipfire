#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# Copyright (c) 2026 HUSRCF
"""Independent CPU oracle for Flash Next's vision tower, never its text model.

Requires torch, transformers (tested with 5.20.2), safetensors, numpy and Pillow.
Only the 333 model.visual tensors are loaded. Outputs raw F32 patches and stage
activations matching qwen4_vision_dump / diff_dumps.py. Receipt hashes the actual
installed upstream source, source weights, configuration and input image.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Vision-only upstream BF16 safetensors")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-pixels", type=int, default=262144,
                        help="Explicit reference image budget recorded in the receipt")
    parser.add_argument("--compare", type=Path, help="Optional hipfire stage dump directory")
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from safetensors.torch import load_file
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    processor_config = json.loads(args.preprocessor.read_text())
    processor_config["size"]["longest_edge"] = args.max_pixels
    processor = Qwen2VLImageProcessor(**processor_config)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    patches, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    np.save(args.out / "pixel_values.npy", patches.float().numpy())
    patches.float().numpy().astype("<f4").tofile(args.out / "pixel_values.bin")
    np.save(args.out / "grid_thw.npy", grid.numpy())
    print(f"patch grid {grid.tolist()}, loading vision weights only", flush=True)

    cfg = Qwen4ExpVisionConfig(**config["vision_config"])
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        visual = Qwen4ExpVisionModel(cfg)
    source_state = load_file(str(args.source), device="cpu")
    if len(source_state) != 333 or any(not key.startswith("model.visual.") for key in source_state):
        raise ValueError("Expected exactly 333 model.visual tensors; refusing a whole text checkpoint")
    if any(value.dtype != torch.bfloat16 for value in source_state.values()):
        raise ValueError("The independent source oracle requires upstream BF16 weights")
    visual.load_state_dict({key.removeprefix("model.visual."): value for key, value in source_state.items()},
                           strict=True, assign=True)
    del source_state
    # RoPE's non-persistent buffers are absent from state_dict. Recreate that
    # small official module on CPU after assigning the checkpoint parameters.
    visual.rotary_pos_emb = type(visual.rotary_pos_emb)(cfg)
    if any(value.is_meta for value in visual.buffers()):
        raise ValueError("Unmaterialized reference buffer after loading the vision tower")
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
    if features.shape != (int(grid.prod()) // 4, 2560) or not np.isfinite(features).all():
        raise ValueError(f"Invalid projected image features: {features.shape}")
    captured["pre_merger"] = captured["block_26"]
    for name, value in captured.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite reference stage {name}")
        np.save(args.out / f"{name}.npy", value)
    features.astype("<f4").tofile(args.out / "post_merger.bin")
    modeling_path = Path(inspect.getfile(Qwen4ExpVisionModel))
    configuration_path = Path(inspect.getfile(Qwen4ExpVisionConfig))
    processor_path = Path(inspect.getfile(Qwen2VLImageProcessor))
    receipt = {
        "schema": "hipfire.qwen4.vision_reference.v1", "status": "reference_complete",
        "source_sha256": sha256(args.source), "config_sha256": sha256(args.config),
        "preprocessor_sha256": sha256(args.preprocessor), "image_sha256": sha256(args.image),
        "patches_sha256": sha256(args.out / "pixel_values.bin"),
        "projected_sha256": sha256(args.out / "post_merger.bin"),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "transformers_modeling_sha256": sha256(modeling_path),
        "transformers_configuration_sha256": sha256(configuration_path),
        "transformers_processor_sha256": sha256(processor_path),
        "device": "cpu", "dtype": args.dtype, "threads": args.threads,
        "max_pixels": args.max_pixels, "grid_thw": grid.tolist(),
        "projected_shape": list(features.shape), "forward_seconds": seconds,
        "quality_acceptance": False,
    }
    if args.compare:
        differences = {}
        for name, reference in captured.items():
            candidate_path = args.compare / f"{name}.bin"
            if not candidate_path.exists():
                differences[name] = {"status": "missing"}
                continue
            candidate = np.fromfile(candidate_path, dtype="<f4")
            if candidate.size != reference.size or not np.isfinite(candidate).all():
                differences[name] = {"status": "invalid"}
                continue
            candidate = candidate.reshape(reference.shape)
            delta = candidate - reference
            differences[name] = {"status": "measured", "max_abs": float(np.abs(delta).max()),
                "relative_l1": float(np.abs(delta).sum() / max(np.abs(reference).sum(), 1e-30)),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
                "cosine": float(np.vdot(candidate, reference) / max(np.linalg.norm(candidate) * np.linalg.norm(reference), 1e-30))}
        receipt["stage_differences"] = differences
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
