#!/usr/bin/env python3
"""Merge-aware rank-16 LoRA QAT for the Qwen3.8 MQ4V2-XT artifact.

Full mode trains the complete text model against the C2 vocabulary-tiled
teacher-logit recipe.  Dry mode uses the three immutable block-boundary pairs
from the L2.5 capture to prove the codec, recurrent-state, optimizer, memory,
and throughput paths on the rented MI300X without pretending that block MSE is KL.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrea  # noqa: E402

from capture import (  # noqa: E402
    RawTensor,
    SafeTensorSource,
    functional_validation,
    instantiate_layer,
    layer_forward,
    load_config,
    rotary_and_masks,
)
from quant_oracle import FUSED4, STANDALONE8, a4_ste, a8_ste, fwht256  # noqa: E402

SOURCE_DEFAULT = Path("/home/kaden/qcal/parents/qwen3.8-27b")
ARTIFACT_DEFAULT = Path("/home/kaden/qcal/qat/awq-a4/qwen3.8-27b.mq4v2.xt.awq-a035.hfq")
DRY_CAPTURE_DEFAULT = Path("/home/kaden/qcal/qat/l25/small-capture")
FULL_CAPTURE_DEFAULT = Path("/home/kaden/qcal/qat/l3")
QUANTIZER_DEFAULT = Path(__file__).resolve().parents[2] / "target/release/hipfire-quantize"
CARD_A_UUID = "GPU-9eb7aeda51c88ffd"
MQ4V2_QTYPE = 44
GROUP = 256
Q8F16_QTYPE = 3
GROUP_BYTES = 136
RANK = 16
LORA_ALPHA = 32.0
SCHEMA = "hipfire.qat.lora.v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def module_at(root: nn.Module, dotted: str) -> nn.Module:
    value: nn.Module = root
    for piece in dotted.split("."):
        value = getattr(value, piece)
    return value


def replace_module(root: nn.Module, dotted: str, replacement: nn.Module) -> None:
    pieces = dotted.split(".")
    parent = module_at(root, ".".join(pieces[:-1])) if len(pieces) > 1 else root
    setattr(parent, pieces[-1], replacement)


def runtime_basis_rows(values: torch.Tensor) -> torch.Tensor:
    return fwht256(values.T.contiguous()).T.contiguous()


def tensor_digest(path: Path, tensor_map: dict[str, dict[str, Any]], names: list[str]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for name in sorted(names):
            item = tensor_map[name]
            digest.update(name.encode())
            digest.update(mapped[item["data_offset"] : item["data_offset"] + item["data_size"]])
    return digest.hexdigest()


def parse_steps(text: str) -> set[int]:
    result = {int(value) for value in text.split(",") if value.strip()}
    if any(value < 0 for value in result):
        raise argparse.ArgumentTypeError("export steps must be nonnegative")
    return result


def parse_blocks(text: str) -> list[int]:
    result = [int(value) for value in text.split(",") if value.strip()]
    if not result or len(result) != len(set(result)) or any(not 0 <= value < 64 for value in result):
        raise argparse.ArgumentTypeError("blocks must be unique integers in [0,63]")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_DEFAULT)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quantizer", type=Path, default=QUANTIZER_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--blocks", type=parse_blocks)
    parser.add_argument("--export-steps", type=parse_steps, default={0, 50, 100})
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--heldout-sequences", type=int)
    parser.add_argument("--logit-tile-vocab", type=int, default=8192)
    parser.add_argument("--dry-tokens", type=int, default=256)
    parser.add_argument("--determinism-tokens", type=int, default=128)
    parser.add_argument("--split-relative-rms", type=float, default=5e-3)
    parser.add_argument("--record-row-chunk", type=int, default=512)
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 20 if args.dry_run else 500
    if args.schedule_steps is None:
        args.schedule_steps = args.steps if args.dry_run else 500
    if args.capture is None:
        args.capture = DRY_CAPTURE_DEFAULT if args.dry_run else FULL_CAPTURE_DEFAULT
    if args.blocks is None:
        args.blocks = [0, 31, 60] if args.dry_run else list(range(64))
    if args.eval_every is None:
        args.eval_every = 5 if args.dry_run else 50
    if args.heldout_sequences is None:
        args.heldout_sequences = 2 if args.dry_run else 32
    if args.steps <= 0 or args.schedule_steps < args.steps or args.accumulate <= 0 or args.rank <= 0:
        parser.error("steps/accumulate/rank must be positive and schedule-steps >= steps")
    if args.warmup < 0 or args.eval_every <= 0 or args.heldout_sequences <= 0:
        parser.error("warmup must be nonnegative; eval cadence/count must be positive")
    if args.dry_run and args.blocks != [0, 31, 60]:
        parser.error("dry-run uses exactly blocks 0,31,60 from the small capture")
    return args


def require_device(args: argparse.Namespace) -> torch.device:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the QAT trainer requires a visible ROCm GPU")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 180 * (1 << 30):
        raise RuntimeError(
            f"QAT requires the single >=180-GiB MI300X; visible device has {props.total_memory / (1 << 30):.1f} GiB"
        )
    return torch.device(args.device)


@dataclass(frozen=True)
class PackedSpec:
    name: str
    m: int
    k: int
    scale: torch.Tensor
    grid: torch.Tensor
    packed: torch.Tensor


class ArtifactReader:
    def __init__(self, path: Path):
        self.path = path
        self.summary, self.tensor_map = astrea.read_hfq_index(path, max_tensors=0)
        self.handle = path.open("rb")
        self.mapped = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self.mapped.close()
        self.handle.close()

    def bytes(self, name: str) -> bytes:
        item = self.tensor_map[name]
        start = int(item["data_offset"])
        return self.mapped[start : start + int(item["data_size"])]

    def spec(self, name: str) -> PackedSpec:
        item = self.tensor_map[name]
        if int(item["quant_type"]) != MQ4V2_QTYPE:
            raise ValueError(f"{name}: expected qt44, got {item['quant_type']}")
        m, k = map(int, item["shape"])
        if k % GROUP:
            raise ValueError(f"{name}: K={k} is not divisible by {GROUP}")
        raw = self.bytes(name)
        groups = m * k // GROUP
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(groups, GROUP_BYTES)
        grid_bits = np.frombuffer(blocks[:, :8].copy().tobytes(), dtype="<u2").reshape(m, k // GROUP, 2, 2)
        grid = torch.from_numpy(grid_bits.copy()).view(torch.float16)
        packed = torch.from_numpy(blocks[:, 8:].copy()).reshape(m, k // 2)
        sidecar = name[: -len(".weight")] + ".awq_scale.weight"
        if sidecar not in self.tensor_map:
            raise KeyError(f"{name}: missing AWQ sidecar {sidecar}")
        scale_item = self.tensor_map[sidecar]
        if list(map(int, scale_item["shape"])) != [k] or int(scale_item["quant_type"]) != 1:
            raise ValueError(f"{sidecar}: expected F16 [{k}]")
        scale_bits = np.frombuffer(self.bytes(sidecar), dtype="<u2").copy()
        scale = torch.from_numpy(scale_bits).view(torch.float16)
        if not bool(torch.isfinite(scale.float()).all()) or not bool((scale > 0).all()):
            raise ValueError(f"{sidecar}: scales must be finite and positive")
        return PackedSpec(name, m, k, scale, grid, packed)

    def q8_tensor(
        self,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        chunk_elements: int = 8 << 20,
    ) -> torch.Tensor:
        item = self.tensor_map[name]
        if int(item["quant_type"]) != Q8F16_QTYPE:
            raise ValueError(f"{name}: expected Q8F16, got qt={item['quant_type']}")
        shape = tuple(map(int, item["shape"]))
        elements = math.prod(shape)
        if elements % 32:
            raise ValueError(f"{name}: Q8F16 element count {elements} is not divisible by 32")
        blocks = np.frombuffer(self.bytes(name), dtype=np.uint8).reshape(elements // 32, 34)
        output = torch.empty(elements, dtype=dtype, device=device)
        block_chunk = max(1, chunk_elements // 32)
        for begin in range(0, blocks.shape[0], block_chunk):
            end = min(begin + block_chunk, blocks.shape[0])
            scale_bits = np.frombuffer(blocks[begin:end, :2].copy().tobytes(), dtype="<u2")
            scales = torch.from_numpy(scale_bits.copy()).view(torch.float16).to(device).float()
            quantized = torch.from_numpy(blocks[begin:end, 2:].copy().view(np.int8)).to(device).float()
            output[begin * 32 : end * 32] = (quantized * scales[:, None]).reshape(-1).to(dtype)
        return output.reshape(shape)


class _FixedGridSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, merged: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        valid = scale != 0
        safe = torch.where(valid, scale, torch.ones_like(scale))
        normalized = (merged - zero) / safe
        codes = torch.floor(normalized + 0.5).clamp_(0, 15)
        decoded = torch.where(valid, codes * scale + zero, zero)
        ctx.save_for_backward(valid & (normalized >= 0) & (normalized <= 15))
        return decoded

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (inside,) = ctx.saved_tensors
        return gradient * inside, None, None


class PackedQATLinear(nn.Module):
    """Packed incumbent grid plus a merge-before-quantize rotated-basis adapter."""

    def __init__(
        self,
        spec: PackedSpec,
        route_site: str,
        adapter: bool,
        rank: int,
        lora_alpha: float,
        device: torch.device,
        compute_dtype: torch.dtype,
    ):
        super().__init__()
        self.artifact_name = spec.name
        self.in_features = spec.k
        self.out_features = spec.m
        self.route_site = route_site
        self.compute_dtype = compute_dtype
        self.lora_scale = float(lora_alpha / rank)
        self.route = "p4"
        self.register_buffer("awq_scale", spec.scale.to(device), persistent=False)
        self.register_buffer("grid", spec.grid.to(device), persistent=False)
        self.register_buffer("packed_codes", spec.packed.to(device), persistent=False)
        if adapter:
            self.lora_a = nn.Parameter(torch.empty(rank, spec.k, dtype=torch.float32, device=device))
            self.lora_b = nn.Parameter(torch.zeros(spec.m, rank, dtype=torch.float32, device=device))
            # Unit expected column norm keeps a zero-B start while letting the
            # 1e-4 pilot cross fixed W4 cell boundaries within 20 steps.
            nn.init.normal_(self.lora_a, mean=0.0, std=rank**-0.5)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @property
    def adapted(self) -> bool:
        return self.lora_a is not None

    def set_route(self, route: str) -> None:
        if route not in {"p4", "p8"}:
            raise ValueError(route)
        self.route = route

    def _codes(self, start: int = 0, end: int | None = None) -> torch.Tensor:
        end = self.out_features if end is None else end
        packed = self.packed_codes[start:end]
        return torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(end - start, self.in_features)

    def _grid_rows(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.grid[start:end].float()
        scale = rows[..., 0].reshape(end - start, self.in_features // 128).repeat_interleave(128, dim=1)
        zero = rows[..., 1].reshape(end - start, self.in_features // 128).repeat_interleave(128, dim=1)
        return scale, zero

    def _base_rows(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, zero = self._grid_rows(start, end)
        codes = self._codes(start, end).float()
        base = torch.where(scale != 0, codes * scale + zero, zero)
        return base, scale, zero

    def fake_weight_rows(self, start: int = 0, end: int | None = None) -> torch.Tensor:
        end = self.out_features if end is None else end
        base, scale, zero = self._base_rows(start, end)
        if self.adapted:
            merged = base + self.lora_scale * (self.lora_b[start:end] @ self.lora_a)
        else:
            merged = base
        return _FixedGridSTE.apply(merged, scale, zero)

    @torch.no_grad()
    def merged_source_rows(self, start: int, end: int) -> torch.Tensor:
        base, _, _ = self._base_rows(start, end)
        merged = base
        if self.adapted:
            merged = merged + self.lora_scale * (self.lora_b[start:end] @ self.lora_a)
        source = fwht256(merged.T.contiguous(), inverse=True).T.contiguous()
        return source / self.awq_scale.float()

    def prepare_input(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], torch.dtype]:
        original_dtype = value.dtype
        shape = tuple(value.shape[:-1])
        flat = value.reshape(-1, self.in_features).float()
        rotated = runtime_basis_rows(flat / self.awq_scale.float())
        column = rotated.T.contiguous()
        if self.route == "p4":
            quantized = a4_ste(column, self.route_site).T
        else:
            quantized = a8_ste(column, "f32").T
        return quantized, shape, original_dtype

    def forward_prepared_rows(self, prepared: torch.Tensor, start: int, end: int) -> torch.Tensor:
        weight = self.fake_weight_rows(start, end)
        return prepared.to(self.compute_dtype) @ weight.T.to(self.compute_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        prepared, shape, original_dtype = self.prepare_input(value)
        output = self.forward_prepared_rows(prepared, 0, self.out_features)
        return output.reshape(*shape, self.out_features).to(original_dtype)

    @torch.no_grad()
    def exported_codes(self, row_chunk: int, assert_initial: bool = False) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, self.out_features, row_chunk):
            end = min(start + row_chunk, self.out_features)
            base, scale, zero = self._base_rows(start, end)
            merged = base
            if self.adapted:
                merged = merged + self.lora_scale * (self.lora_b[start:end] @ self.lora_a)
            safe = torch.where(scale != 0, scale, torch.ones_like(scale))
            selected = torch.where(
                scale != 0,
                torch.floor((merged - zero) / safe + 0.5).clamp_(0, 15),
                torch.zeros_like(merged),
            ).to(torch.uint8)
            if assert_initial and not torch.equal(selected, self._codes(start, end)):
                mismatches = int((selected != self._codes(start, end)).sum().item())
                raise AssertionError(f"{self.artifact_name}: B=0 changed {mismatches} incumbent codes")
            chunks.append(selected.cpu())
        return torch.cat(chunks, dim=0)

    def record_tensors(self, row_chunk: int, assert_initial: bool) -> dict[str, torch.Tensor]:
        return {
            "S_f16": self.awq_scale.detach().cpu(),
            "d_z_f16": self.grid.detach().cpu(),
            "codes_u8": self.exported_codes(row_chunk, assert_initial),
        }


def is_adapter_target(name: str) -> bool:
    return bool(
        re.search(
            r"\.(?:linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b)|self_attn\.(?:q_proj|k_proj|v_proj)|mlp\.(?:gate_proj|up_proj))\.weight$",
            name,
        )
    )


def route_site(name: str) -> str:
    if any(part in name for part in (".out_proj.weight", ".o_proj.weight", ".down_proj.weight")) or name == "lm_head.weight":
        return STANDALONE8
    return FUSED4


def assert_tied_scales(reader: ArtifactReader, names: list[str]) -> None:
    groups: dict[str, list[str]] = {}
    for name in names:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if not match:
            continue
        block = match.group(1)
        if ".mlp." in name and any(value in name for value in ("gate_proj", "up_proj")):
            groups.setdefault(f"{block}:gate_up", []).append(name)
        elif any(value in name for value in ("in_proj_", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
            groups.setdefault(f"{block}:mixer_inputs", []).append(name)
    for group, members in groups.items():
        if len(members) < 2:
            continue
        payloads = []
        for name in members:
            sidecar = name[: -len(".weight")] + ".awq_scale.weight"
            payloads.append(reader.bytes(sidecar))
        if any(payload != payloads[0] for payload in payloads[1:]):
            raise AssertionError(f"shared-input AWQ scales differ for {group}: {members}")


def install_q8_weight(
    root: nn.Module,
    local: str,
    artifact_name: str,
    reader: ArtifactReader,
    device: torch.device,
) -> None:
    module = module_at(root, local)
    decoded = reader.q8_tensor(artifact_name, device, torch.bfloat16)
    if tuple(module.weight.shape) != tuple(decoded.shape):
        raise ValueError(
            f"{artifact_name}: source shape {tuple(module.weight.shape)} != Q8 shape {tuple(decoded.shape)}"
        )
    with torch.no_grad():
        module.weight.copy_(decoded)
    del decoded


def install_dry_layers(
    args: argparse.Namespace,
    reader: ArtifactReader,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[dict[int, nn.Module], list[PackedQATLinear], list[PackedQATLinear]]:
    source = SafeTensorSource(args.source)
    config = load_config(args.source)
    layers: dict[int, nn.Module] = {}
    wrappers: list[PackedQATLinear] = []
    adapted: list[PackedQATLinear] = []
    names = [
        name
        for name, item in reader.tensor_map.items()
        if int(item["quant_type"]) == MQ4V2_QTYPE
        and any(f".layers.{block}." in name for block in args.blocks)
    ]
    assert_tied_scales(reader, names)
    for block in args.blocks:
        layer = instantiate_layer(source, config, block, device)
        prefix = f"model.language_model.layers.{block}."
        for name, item in reader.tensor_map.items():
            if (
                int(item["quant_type"]) == Q8F16_QTYPE
                and name.startswith(prefix)
                and name.endswith(".weight")
            ):
                install_q8_weight(
                    layer,
                    name[len(prefix) : -len(".weight")],
                    name,
                    reader,
                    device,
                )
        for name in sorted(value for value in names if value.startswith(prefix)):
            local = name[len(prefix) : -len(".weight")]
            original = module_at(layer, local)
            spec = reader.spec(name)
            if tuple(original.weight.shape) != (spec.m, spec.k):
                raise ValueError(f"{name}: source shape {tuple(original.weight.shape)} != {(spec.m, spec.k)}")
            wrapper = PackedQATLinear(
                spec,
                route_site(name),
                is_adapter_target(name),
                args.rank,
                args.lora_alpha,
                device,
                compute_dtype,
            )
            replace_module(layer, local, wrapper)
            wrappers.append(wrapper)
            if wrapper.adapted:
                adapted.append(wrapper)
        for parameter in layer.parameters():
            if parameter not in {wrapper.lora_a for wrapper in adapted if wrapper.lora_a is not None} and parameter not in {
                wrapper.lora_b for wrapper in adapted if wrapper.lora_b is not None
            }:
                parameter.requires_grad_(False)
        layer.train()
        layers[block] = layer
    return layers, wrappers, adapted


def _source_to_text_name(source_name: str) -> str | None:
    prefix = "model.language_model."
    if source_name.startswith(prefix):
        return "model." + source_name[len(prefix) :]
    if source_name == "lm_head.weight":
        return source_name
    return None


def load_text_model(
    source_path: Path, device: torch.device, pure_torch_deltanet: bool = False
) -> nn.Module:
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    full_config = AutoConfig.from_pretrained(source_path, trust_remote_code=False)
    config = full_config.text_config
    config._attn_implementation = "eager"
    config.use_cache = False
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    expected = set(model.state_dict())
    index = json.loads((source_path / "model.safetensors.index.json").read_text())
    mapped = {
        target
        for source_name in index["weight_map"]
        if (target := _source_to_text_name(source_name)) is not None
    }
    if mapped != expected:
        raise RuntimeError(
            f"text model source coverage mismatch: missing={sorted(expected - mapped)[:8]} unexpected={sorted(mapped - expected)[:8]}"
        )
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for source_name, shard in index["weight_map"].items():
        target = _source_to_text_name(source_name)
        if target is not None:
            by_shard.setdefault(shard, []).append((source_name, target))
    for shard in sorted(by_shard):
        with safe_open(source_path / shard, framework="pt", device=str(device)) as handle:
            for source_name, target_name in by_shard[shard]:
                set_module_tensor_to_device(
                    model,
                    target_name,
                    device,
                    value=handle.get_tensor(source_name),
                    dtype=torch.bfloat16,
                )
    if pure_torch_deltanet:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.chunk_gated_delta_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule
                layer.linear_attn.recurrent_gated_delta_rule = modeling_qwen3_5.torch_recurrent_gated_delta_rule
    return model


def install_full_model(
    args: argparse.Namespace,
    reader: ArtifactReader,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[nn.Module, list[PackedQATLinear], list[PackedQATLinear]]:
    model = load_text_model(args.source, device)
    for name, item in reader.tensor_map.items():
        if int(item["quant_type"]) != Q8F16_QTYPE:
            continue
        if name.startswith("model.language_model."):
            local = "model." + name[len("model.language_model.") : -len(".weight")]
        elif name == "lm_head.weight":
            local = "lm_head"
        else:
            continue
        install_q8_weight(model, local, name, reader, device)
    wrappers: list[PackedQATLinear] = []
    adapted: list[PackedQATLinear] = []
    names = [name for name, item in reader.tensor_map.items() if int(item["quant_type"]) == MQ4V2_QTYPE]
    assert_tied_scales(reader, names)
    for name in sorted(names):
        if name.startswith("model.language_model."):
            local = "model." + name[len("model.language_model.") : -len(".weight")]
        elif name == "lm_head.weight":
            local = "lm_head"
        else:
            raise RuntimeError(f"qt44 tensor is outside the text model: {name}")
        original = module_at(model, local)
        spec = reader.spec(name)
        if tuple(original.weight.shape) != (spec.m, spec.k):
            raise ValueError(f"{name}: source shape {tuple(original.weight.shape)} != {(spec.m, spec.k)}")
        wrapper = PackedQATLinear(
            spec,
            route_site(name),
            is_adapter_target(name),
            args.rank,
            args.lora_alpha,
            device,
            compute_dtype,
        )
        replace_module(model, local, wrapper)
        wrappers.append(wrapper)
        if wrapper.adapted:
            adapted.append(wrapper)
        del original
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for wrapper in adapted:
        wrapper.lora_a.requires_grad_(True)
        wrapper.lora_b.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, wrappers, adapted


def set_route(wrappers: list[PackedQATLinear], route: str) -> None:
    for wrapper in wrappers:
        wrapper.set_route(route)


def adapter_parameters(adapted: list[PackedQATLinear]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    for wrapper in adapted:
        assert wrapper.lora_a is not None and wrapper.lora_b is not None
        result.extend((wrapper.lora_a, wrapper.lora_b))
    return result


def adapter_state(adapted: list[PackedQATLinear]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for wrapper in adapted:
        assert wrapper.lora_a is not None and wrapper.lora_b is not None
        result[wrapper.artifact_name + ".lora_A"] = wrapper.lora_a.detach().cpu().clone()
        result[wrapper.artifact_name + ".lora_B"] = wrapper.lora_b.detach().cpu().clone()
    return result


def load_adapter_state(adapted: list[PackedQATLinear], state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for wrapper in adapted:
            assert wrapper.lora_a is not None and wrapper.lora_b is not None
            wrapper.lora_a.copy_(state[wrapper.artifact_name + ".lora_A"])
            wrapper.lora_b.copy_(state[wrapper.artifact_name + ".lora_B"])


def learning_rate(step: int, steps: int, base: float, warmup: int) -> float:
    if warmup and step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def read_bf16_sequence(root: Path, meta: dict[str, Any], sequence: int, tokens: int, device: torch.device) -> torch.Tensor:
    shape = tuple(map(int, meta["shape"]))
    raw = np.memmap(root / meta["file"], dtype="<u2", mode="r", shape=shape)
    value = torch.from_numpy(np.asarray(raw[sequence, :tokens]).copy()).view(torch.bfloat16)
    return value.unsqueeze(0).to(device)


def block_forward(
    layer: nn.Module,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    causal: torch.Tensor,
    block_type: str,
    checkpointed: bool,
) -> torch.Tensor:
    def invoke(value: torch.Tensor) -> torch.Tensor:
        return layer_forward(layer, value, position_embeddings, positions, causal, block_type)

    return checkpoint(invoke, hidden, use_reentrant=False) if checkpointed else invoke(hidden)


def dry_eval(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    layers: dict[int, nn.Module],
    wrappers: list[PackedQATLinear],
    device: torch.device,
) -> dict[str, float]:
    train_count = int(manifest["corpus"]["train_sequences"])
    heldout_total = int(manifest["corpus"]["heldout_sequences"])
    count = min(args.heldout_sequences, heldout_total)
    config = load_config(args.source)
    position_embeddings, positions, causal = rotary_and_masks(config, args.dry_tokens, device)
    sums = {"p4": 0.0, "p8": 0.0}
    with torch.no_grad():
        for block in args.blocks:
            block_meta = manifest["capture"]["blocks"][str(block)]
            input_meta = manifest["capture"]["boundaries"][block_meta["input_boundary"]]
            output_meta = manifest["capture"]["boundaries"][block_meta["output_boundary"]]
            for sequence in range(train_count, train_count + count):
                hidden = read_bf16_sequence(args.capture, input_meta, sequence, args.dry_tokens, device)
                teacher = read_bf16_sequence(args.capture, output_meta, sequence, args.dry_tokens, device).float()
                denominator = teacher.square().sum().clamp_min(1e-30)
                for route in ("p4", "p8"):
                    set_route(wrappers, route)
                    output = block_forward(
                        layers[block],
                        hidden,
                        position_embeddings,
                        positions,
                        causal,
                        str(config.layer_types[block]),
                        False,
                    ).float()
                    sums[route] += float(((output - teacher).square().sum() / denominator).item())
    divisor = len(args.blocks) * count
    result = {route: value / divisor for route, value in sums.items()}
    result["objective"] = 0.8 * result["p4"] + 0.25 * result["p8"]
    return result


def dry_determinism(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    layers: dict[int, nn.Module],
    wrappers: list[PackedQATLinear],
    device: torch.device,
) -> dict[str, Any]:
    block = next(value for value in args.blocks if value % 4 != 3)
    meta = manifest["capture"]["boundaries"][str(block)]
    hidden = read_bf16_sequence(args.capture, meta, 0, args.determinism_tokens, device)
    set_route(wrappers, "p4")
    result = functional_validation(
        layers[block],
        load_config(args.source),
        hidden,
        str(load_config(args.source).layer_types[block]),
        device,
        args.determinism_tokens,
    )
    if not result["checkpoint_recompute_bitwise_equal"]:
        raise AssertionError("whole-vs-checkpoint recompute is not bitwise deterministic")
    split = result["whole_vs_sequence_split"]["relative_rms"]
    if split > args.split_relative_rms:
        raise AssertionError(f"split-vs-whole relative RMS {split:.6g} exceeds {args.split_relative_rms:.6g}")
    return result


def save_records(
    records_dir: Path,
    adapted: list[PackedQATLinear],
    artifact_sha: str,
    row_chunk: int,
    assert_initial: bool,
) -> list[str]:
    if records_dir.exists():
        shutil.rmtree(records_dir)
    records_dir.mkdir(parents=True)
    names: list[str] = []
    for index, wrapper in enumerate(sorted(adapted, key=lambda value: value.artifact_name)):
        tensors = wrapper.record_tensors(row_chunk, assert_initial)
        metadata = {
            "name": wrapper.artifact_name,
            "M": str(wrapper.out_features),
            "K": str(wrapper.in_features),
            "qt": str(MQ4V2_QTYPE),
            "source_sha": artifact_sha,
        }
        save_file(tensors, records_dir / f"{index:04d}.safetensors", metadata=metadata)
        names.append(wrapper.artifact_name)
    return names


@torch.no_grad()
def merged_source_tensor(wrapper: PackedQATLinear, row_chunk: int) -> torch.Tensor:
    output = torch.empty(
        (wrapper.out_features, wrapper.in_features), dtype=torch.bfloat16, device="cpu"
    )
    for start in range(0, wrapper.out_features, row_chunk):
        end = min(start + row_chunk, wrapper.out_features)
        output[start:end].copy_(
            wrapper.merged_source_rows(start, end).to(torch.bfloat16).cpu()
        )
    return output


def export_merged_bf16(
    args: argparse.Namespace,
    wrappers: list[PackedQATLinear],
    label: str,
) -> dict[str, Any]:
    from safetensors import safe_open

    output_root = args.out / "exports" / label / "merged-bf16"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    wrapper_map = {wrapper.artifact_name: wrapper for wrapper in wrappers}
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    source = SafeTensorSource(args.source)
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    started = time.monotonic()
    for shard in sorted(by_shard):
        tensors: dict[str, torch.Tensor] = {}
        for name in sorted(by_shard[shard]):
            wrapper = wrapper_map.get(name)
            tensors[name] = (
                merged_source_tensor(wrapper, args.record_row_chunk)
                if wrapper is not None
                else source.get(name).contiguous()
            )
        with safe_open(args.source / shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        temporary = output_root / f".{shard}.tmp"
        save_file(tensors, temporary, metadata=metadata)
        destination = output_root / shard
        os.replace(temporary, destination)
        digest = sha256_file(destination)
        size = destination.stat().st_size
        files.append({"file": shard, "sha256": digest, "bytes": size})
        aggregate.update(shard.encode())
        aggregate.update(bytes.fromhex(digest))
        del tensors
    for source_path in args.source.iterdir():
        if source_path.name.startswith("model-") and source_path.suffix == ".safetensors":
            continue
        destination = output_root / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination)
    receipt = {
        "directory": str(output_root),
        "basis": "BF16 source; W=FWHT_inverse(U0+2BA)/S",
        "shards": files,
        "bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": aggregate.hexdigest(),
        "seconds": time.monotonic() - started,
    }
    atomic_json(output_root / "receipt.json", receipt)
    return receipt


def export_artifact(
    args: argparse.Namespace,
    adapted: list[PackedQATLinear],
    artifact_sha: str,
    tensor_map: dict[str, dict[str, Any]],
    label: str,
    assert_initial: bool = False,
    all_wrappers: list[PackedQATLinear] | None = None,
) -> dict[str, Any]:
    export_root = args.out / "exports" / label
    records_dir = export_root / "records"
    export_root.mkdir(parents=True, exist_ok=True)
    names = save_records(records_dir, adapted, artifact_sha, args.record_row_chunk, assert_initial)
    state_path = export_root / "adapters.safetensors"
    save_file(adapter_state(adapted), state_path, metadata={"schema": SCHEMA, "rank": str(args.rank)})
    output = export_root / "candidate.hfq"
    started = time.monotonic()
    completed = subprocess.run(
        [
            str(args.quantizer),
            "--input",
            str(args.artifact),
            "--output",
            str(output),
            "--mq4v2-final-codes",
            str(records_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output_sha = sha256_file(output)
    _, output_map = astrea.read_hfq_index(output, max_tensors=0)
    touched_input = tensor_digest(args.artifact, tensor_map, names)
    touched_output = tensor_digest(output, output_map, names)
    if touched_input == touched_output and not assert_initial:
        changed = False
    else:
        changed = touched_input != touched_output
    if assert_initial:
        if output_sha != artifact_sha:
            raise AssertionError(f"step-0 artifact SHA differs: {output_sha} != {artifact_sha}")
        if touched_output != touched_input:
            raise AssertionError("step-0 touched tensor bytes differ from the L1 artifact")
    merged_bf16 = None
    if args.full:
        if all_wrappers is None:
            raise ValueError("full export requires all qt44 wrappers")
        # Every qt44 tensor, adapted or frozen, is emitted from its decoded U0
        # so the directory is a complete source-basis model for re-quantization.
        merged_bf16 = export_merged_bf16(args, all_wrappers, label)
    receipt = {
        "label": label,
        "records": len(names),
        "record_names": names,
        "records_dir": str(records_dir),
        "adapter_state": str(state_path),
        "artifact": str(output),
        "artifact_sha256": output_sha,
        "source_artifact_sha256": artifact_sha,
        "touched_source_sha256": touched_input,
        "touched_output_sha256": touched_output,
        "touched_changed": changed,
        "seconds": time.monotonic() - started,
        "quantizer_stderr": completed.stderr.strip(),
        "merged_bf16": merged_bf16,
    }
    atomic_json(export_root / "receipt.json", receipt)
    return receipt


def run_dry(
    args: argparse.Namespace,
    reader: ArtifactReader,
    device: torch.device,
    artifact_sha: str,
) -> dict[str, Any]:
    manifest = json.loads((args.capture / "manifest.json").read_text())
    if manifest.get("capture", {}).get("status") != "complete":
        raise RuntimeError("dry capture is not complete")
    if sorted(map(int, manifest["capture"]["blocks"])) != args.blocks:
        raise RuntimeError("dry capture does not contain exactly blocks 0,31,60")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    layers, wrappers, adapted = install_dry_layers(args, reader, device, torch.bfloat16)
    if len(adapted) != 17:
        raise AssertionError(f"expected 17 adapted tensors over blocks 0,31,60, got {len(adapted)}")
    if any(bool(torch.count_nonzero(wrapper.lora_b).item()) for wrapper in adapted):
        raise AssertionError("LoRA B must be exactly zero at step 0")

    step0_export = export_artifact(
        args, adapted, artifact_sha, reader.tensor_map, "step-000", assert_initial=True
    )
    determinism = dry_determinism(args, manifest, layers, wrappers, device)
    initial_eval = dry_eval(args, manifest, layers, wrappers, device)
    best_eval = dict(initial_eval)
    best_step = 0
    best_state = adapter_state(adapted)

    parameters = adapter_parameters(adapted)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    config = load_config(args.source)
    position_embeddings, positions, causal = rotary_and_masks(config, args.dry_tokens, device)
    train_count = int(manifest["corpus"]["train_sequences"])
    history: list[dict[str, Any]] = [{"step": 0, "heldout": initial_eval}]
    started = time.monotonic()
    processed_tokens = 0
    for step in range(1, args.steps + 1):
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(args.accumulate):
            ordinal = (step - 1) * args.accumulate + micro
            block = args.blocks[ordinal % len(args.blocks)]
            sequence = ordinal % train_count
            block_meta = manifest["capture"]["blocks"][str(block)]
            input_meta = manifest["capture"]["boundaries"][block_meta["input_boundary"]]
            output_meta = manifest["capture"]["boundaries"][block_meta["output_boundary"]]
            hidden = read_bf16_sequence(args.capture, input_meta, sequence, args.dry_tokens, device)
            teacher = read_bf16_sequence(args.capture, output_meta, sequence, args.dry_tokens, device).float()
            denominator = teacher.square().sum().clamp_min(1e-30)
            for route, coefficient in (("p4", 0.8), ("p8", 0.25)):
                set_route(wrappers, route)
                output = block_forward(
                    layers[block],
                    hidden,
                    position_embeddings,
                    positions,
                    causal,
                    str(config.layer_types[block]),
                    True,
                ).float()
                route_loss = (output - teacher).square().sum() / denominator
                weighted = coefficient * route_loss / args.accumulate
                weighted.backward()
                loss_sum += float((coefficient * route_loss).detach().item()) / args.accumulate
            processed_tokens += args.dry_tokens
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_norm).item())
        lr = learning_rate(step, args.schedule_steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        elapsed = time.monotonic() - step_started
        row: dict[str, Any] = {
            "step": step,
            "train_loss": loss_sum,
            "lr": lr,
            "grad_norm_before_clip": grad_norm,
            "tokens_per_second": args.accumulate * args.dry_tokens / elapsed,
        }
        if step % args.eval_every == 0 or step == args.steps:
            heldout = dry_eval(args, manifest, layers, wrappers, device)
            row["heldout"] = heldout
            if heldout["objective"] < best_eval["objective"]:
                best_eval = dict(heldout)
                best_step = step
                best_state = adapter_state(adapted)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if step in args.export_steps:
            export_artifact(args, adapted, artifact_sha, reader.tensor_map, f"step-{step:03d}")

    final_eval = dry_eval(args, manifest, layers, wrappers, device)
    if final_eval["objective"] >= initial_eval["objective"]:
        raise AssertionError(
            f"dry-run held-out loss did not decrease: {initial_eval['objective']:.8g} -> {final_eval['objective']:.8g}"
        )
    load_adapter_state(adapted, best_state)
    best_export = export_artifact(args, adapted, artifact_sha, reader.tensor_map, f"best-step-{best_step:03d}")
    total_seconds = time.monotonic() - started
    return {
        "mode": "dry-run",
        "surrogate": "whole-block relative MSE: 0.8*p4 + 0.25*p8; full mode alone uses vocabulary KL/CE",
        "blocks": args.blocks,
        "tokens": args.dry_tokens,
        "steps": args.steps,
        "accumulate": args.accumulate,
        "step0_export": step0_export,
        "determinism": determinism,
        "initial_heldout": initial_eval,
        "final_heldout": final_eval,
        "best_heldout": best_eval,
        "best_step": best_step,
        "best_export": best_export,
        "loss_curve": history,
        "fit_seconds": total_seconds,
        "fit_tokens": processed_tokens,
        "tokens_per_second": processed_tokens / total_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def logsumexp_tiles(
    hidden: torch.Tensor,
    vocab: int,
    tile: int,
    producer: Any,
) -> torch.Tensor:
    result: torch.Tensor | None = None
    with torch.no_grad():
        for start in range(0, vocab, tile):
            logits = producer(hidden, start, min(start + tile, vocab)).float()
            partial = torch.logsumexp(logits, dim=-1)
            result = partial if result is None else torch.logaddexp(result, partial)
    assert result is not None
    return result


def full_route_backward(
    args: argparse.Namespace,
    model: nn.Module,
    wrappers: list[PackedQATLinear],
    teacher_head: torch.Tensor,
    tokens: torch.Tensor,
    teacher_final: torch.Tensor,
    route: str,
    divisor: float,
) -> dict[str, float]:
    set_route(wrappers, route)
    outputs = model.model(input_ids=tokens, use_cache=False, return_dict=True)
    student_hidden = outputs.last_hidden_state[:, :-1]
    teacher_hidden = teacher_final[:, :-1]
    labels = tokens[:, 1:]
    head = model.lm_head
    if not isinstance(head, PackedQATLinear):
        raise TypeError("lm_head must be MQ4V2 for the tiled objective")
    prepared_original, _, _ = head.prepare_input(student_hidden)
    prepared = prepared_original.detach().requires_grad_(True)
    vocab = head.out_features
    tile = args.logit_tile_vocab
    student_shape = student_hidden.shape[:2]
    student_lse = logsumexp_tiles(
        prepared.detach(),
        vocab,
        tile,
        lambda value, begin, end: head.forward_prepared_rows(value, begin, end).reshape(
            *student_shape, end - begin
        ),
    )
    teacher_lse = logsumexp_tiles(
        teacher_hidden,
        vocab,
        tile,
        lambda value, begin, end: value.to(torch.bfloat16) @ teacher_head[begin:end].T,
    )
    token_count = labels.numel()
    kl_weight = 0.8 if route == "p4" else 0.25
    ce_weight = 0.2 if route == "p4" else 0.0
    kl_value = 0.0
    ce_value = 0.0
    for start in range(0, vocab, tile):
        end = min(start + tile, vocab)
        student_logits = head.forward_prepared_rows(prepared, start, end).reshape(
            *student_shape, end - start
        ).float()
        with torch.no_grad():
            teacher_logits = teacher_hidden.to(torch.bfloat16) @ teacher_head[start:end].T
            teacher_logits = teacher_logits.float()
            teacher_probability = torch.exp(teacher_logits - teacher_lse.unsqueeze(-1))
        kl = (
            teacher_probability
            * (
                teacher_logits
                - teacher_lse.unsqueeze(-1)
                - student_logits
                + student_lse.unsqueeze(-1)
            )
        ).sum() / token_count
        in_tile = (labels >= start) & (labels < end)
        if bool(in_tile.any()):
            positions = in_tile.nonzero(as_tuple=False)
            local = labels[in_tile] - start
            selected = student_logits[positions[:, 0], positions[:, 1], local]
            ce_logits = -selected.sum() / token_count
        else:
            ce_logits = student_logits.sum() * 0.0
        softmax_mass = torch.exp(student_logits - student_lse.unsqueeze(-1)).sum() / token_count
        correction = (kl_weight + ce_weight) * (softmax_mass - softmax_mass.detach())
        loss = (kl_weight * kl + ce_weight * ce_logits + correction) / divisor
        loss.backward()
        kl_value += float(kl.detach().item())
        ce_value += float(ce_logits.detach().item())
    ce_value += float(student_lse.mean().item())
    if prepared.grad is None:
        raise AssertionError("tiled vocabulary objective produced no activation gradient")
    prepared_original.backward(prepared.grad)
    return {
        "kl": kl_value,
        "ce": ce_value if route == "p4" else 0.0,
        "weighted": kl_weight * kl_value + ce_weight * ce_value,
    }


def full_eval_sequence(
    args: argparse.Namespace,
    model: nn.Module,
    wrappers: list[PackedQATLinear],
    teacher_head: torch.Tensor,
    tokens: torch.Tensor,
    teacher_final: torch.Tensor,
    route: str,
) -> dict[str, float]:
    set_route(wrappers, route)
    with torch.no_grad():
        student_hidden = model.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        teacher_hidden = teacher_final[:, :-1]
        labels = tokens[:, 1:]
        head = model.lm_head
        prepared, _, _ = head.prepare_input(student_hidden)
        student_shape = student_hidden.shape[:2]
        student_lse = logsumexp_tiles(
            prepared,
            head.out_features,
            args.logit_tile_vocab,
            lambda value, begin, end: head.forward_prepared_rows(value, begin, end).reshape(
                *student_shape, end - begin
            ),
        )
        teacher_lse = logsumexp_tiles(
            teacher_hidden, head.out_features, args.logit_tile_vocab,
            lambda value, begin, end: value.to(torch.bfloat16) @ teacher_head[begin:end].T,
        )
        kl = 0.0
        target_logits = torch.empty_like(student_lse)
        for start in range(0, head.out_features, args.logit_tile_vocab):
            end = min(start + args.logit_tile_vocab, head.out_features)
            student_logits = head.forward_prepared_rows(prepared, start, end).reshape(
                *student_shape, end - start
            ).float()
            teacher_logits = (teacher_hidden.to(torch.bfloat16) @ teacher_head[start:end].T).float()
            probability = torch.exp(teacher_logits - teacher_lse.unsqueeze(-1))
            kl += float(
                (
                    probability
                    * (teacher_logits - teacher_lse.unsqueeze(-1) - student_logits + student_lse.unsqueeze(-1))
                ).sum(dim=-1).mean().item()
            )
            in_tile = (labels >= start) & (labels < end)
            if bool(in_tile.any()):
                positions = in_tile.nonzero(as_tuple=False)
                local = labels[in_tile] - start
                target_logits[in_tile] = student_logits[positions[:, 0], positions[:, 1], local]
        ce = float((-target_logits + student_lse).mean().item())
    return {"kl": kl, "ce": ce}


def full_eval(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    model: nn.Module,
    teacher_model: nn.Module,
    wrappers: list[PackedQATLinear],
    teacher_head: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    train_count = int(manifest["corpus"]["train_sequences"])
    total = int(manifest["corpus"]["heldout_sequences"])
    count = min(args.heldout_sequences, total)
    seq_len = int(manifest["corpus"]["sequence_length"])
    token_meta = manifest["corpus"]["token_stream"]
    token_array = np.memmap(
        args.capture / token_meta["file"], dtype="<u4", mode="r", shape=tuple(token_meta["shape"])
    )
    sums = {"p4_kl": 0.0, "p4_ce": 0.0, "p8_kl": 0.0}
    was_training = model.training
    model.eval()
    for sequence in range(train_count, train_count + count):
        tokens = torch.from_numpy(np.asarray(token_array[sequence]).astype(np.int64)).view(1, seq_len).to(device)
        with torch.no_grad():
            teacher = teacher_model.model(
                input_ids=tokens, use_cache=False, return_dict=True
            ).last_hidden_state
        p4 = full_eval_sequence(args, model, wrappers, teacher_head, tokens, teacher, "p4")
        p8 = full_eval_sequence(args, model, wrappers, teacher_head, tokens, teacher, "p8")
        sums["p4_kl"] += p4["kl"]
        sums["p4_ce"] += p4["ce"]
        sums["p8_kl"] += p8["kl"]
    if was_training:
        model.train()
    result = {key: value / count for key, value in sums.items()}
    result["objective"] = 0.8 * result["p4_kl"] + 0.2 * result["p4_ce"] + 0.25 * result["p8_kl"]
    return result


def run_full(
    args: argparse.Namespace,
    reader: ArtifactReader,
    device: torch.device,
    artifact_sha: str,
) -> dict[str, Any]:
    manifest = json.loads((args.capture / "manifest.json").read_text())
    if manifest.get("schema_version") != "hipfire.qat.capture.v1" or manifest.get("capture", {}).get("status") != "complete":
        raise RuntimeError("full mode requires a complete C2 hipfire.qat.capture.v1 manifest")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model, wrappers, adapted = install_full_model(args, reader, device, torch.bfloat16)
    if len(adapted) != 368:
        raise AssertionError(f"expected 368 adapted tensors, got {len(adapted)}")
    teacher_model = load_text_model(args.source, device, pure_torch_deltanet=True)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    teacher_head = teacher_model.lm_head.weight
    delta_path = args.capture / "fla_delta.json"
    if not delta_path.is_file():
        raise FileNotFoundError("full mode requires capture/fla_delta.json")
    fla_delta = json.loads(delta_path.read_text())
    if float(fla_delta["relative_rms"]) > args.split_relative_rms:
        raise RuntimeError(f"FLA student surrogate exceeds limit: {fla_delta}")
    if any(bool(torch.count_nonzero(wrapper.lora_b).item()) for wrapper in adapted):
        raise AssertionError("LoRA B must be exactly zero at step 0")
    step0_export = export_artifact(
        args, adapted, artifact_sha, reader.tensor_map, "step-000", True, wrappers
    )
    initial_eval = full_eval(
        args, manifest, model, teacher_model, wrappers, teacher_head, device
    )
    best_eval = dict(initial_eval)
    best_step = 0
    best_state = adapter_state(adapted)

    parameters = adapter_parameters(adapted)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    seq_len = int(manifest["corpus"]["sequence_length"])
    train_count = int(manifest["corpus"]["train_sequences"])
    token_meta = manifest["corpus"]["token_stream"]
    token_array = np.memmap(
        args.capture / token_meta["file"], dtype="<u4", mode="r", shape=tuple(token_meta["shape"])
    )
    history: list[dict[str, Any]] = [{"step": 0, "heldout": initial_eval}]
    processed_tokens = 0
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals = {"p4_kl": 0.0, "p4_ce": 0.0, "p8_kl": 0.0, "weighted": 0.0}
        for micro in range(args.accumulate):
            sequence = ((step - 1) * args.accumulate + micro) % train_count
            tokens = torch.from_numpy(np.asarray(token_array[sequence]).astype(np.int64)).view(1, seq_len).to(device)
            with torch.no_grad():
                teacher = teacher_model.model(
                    input_ids=tokens, use_cache=False, return_dict=True
                ).last_hidden_state
            p4 = full_route_backward(args, model, wrappers, teacher_head, tokens, teacher, "p4", args.accumulate)
            p8 = full_route_backward(args, model, wrappers, teacher_head, tokens, teacher, "p8", args.accumulate)
            totals["p4_kl"] += p4["kl"] / args.accumulate
            totals["p4_ce"] += p4["ce"] / args.accumulate
            totals["p8_kl"] += p8["kl"] / args.accumulate
            totals["weighted"] += (p4["weighted"] + p8["weighted"]) / args.accumulate
            processed_tokens += seq_len
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_norm).item())
        lr = learning_rate(step, args.schedule_steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        elapsed = time.monotonic() - step_started
        row: dict[str, Any] = {
            "step": step,
            "train": totals,
            "lr": lr,
            "grad_norm_before_clip": grad_norm,
            "tokens_per_second": args.accumulate * seq_len / elapsed,
        }
        if step % args.eval_every == 0 or step == args.steps:
            heldout = full_eval(
                args, manifest, model, teacher_model, wrappers, teacher_head, device
            )
            row["heldout"] = heldout
            if heldout["objective"] < best_eval["objective"]:
                best_eval = dict(heldout)
                best_step = step
                best_state = adapter_state(adapted)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if step in args.export_steps:
            export_artifact(
                args, adapted, artifact_sha, reader.tensor_map, f"step-{step:03d}", False, wrappers
            )
    load_adapter_state(adapted, best_state)
    best_export = export_artifact(
        args, adapted, artifact_sha, reader.tensor_map, f"best-step-{best_step:03d}", False, wrappers
    )
    total_seconds = time.monotonic() - started
    return {
        "mode": "full",
        "objective": "0.8*KL(teacher||p4) + 0.2*CE(p4,tokens) + 0.25*KL(teacher||p8)",
        "teacher": "resident BF16 source model; pure torch DeltaNet; online no-grad logits",
        "student_deltanet": {"backend": "FLA chunked", "validation": fla_delta},
        "steps": args.steps,
        "sequence_length": seq_len,
        "accumulate": args.accumulate,
        "effective_tokens": seq_len * args.accumulate,
        "step0_export": step0_export,
        "initial_heldout": initial_eval,
        "best_heldout": best_eval,
        "best_step": best_step,
        "best_export": best_export,
        "loss_curve": history,
        "fit_seconds": total_seconds,
        "fit_tokens": processed_tokens,
        "tokens_per_second": processed_tokens / total_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def command_lines(args: argparse.Namespace) -> dict[str, str]:
    script = Path(__file__).resolve()
    common = (
        f"ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 "
        f"PYTORCH_HIP_ALLOC_CONF=expandable_segments:True python3 {script} --full "
        f"--device cuda --dtype bfloat16 --source {args.source} --artifact {args.artifact} "
        f"--capture {args.capture} --quantizer {args.quantizer}"
    )
    return {
        "mi300x_kill_experiment": common
        + f" --steps 100 --schedule-steps 500 --export-steps 0,50,100 --out {args.out}",
        "mi300x_full": common
        + f" --steps 500 --schedule-steps 500 --export-steps 0,50,100 --out {args.out}",
    }


def main() -> int:
    args = parse_args()
    device = require_device(args)
    if not args.source.is_dir() or not args.artifact.is_file() or not (args.capture / "manifest.json").is_file():
        raise FileNotFoundError("source, artifact, or capture manifest is missing")
    if not args.quantizer.is_file():
        raise FileNotFoundError(args.quantizer)
    args.out.mkdir(parents=True, exist_ok=True)
    artifact_sha = sha256_file(args.artifact)
    reader = ArtifactReader(args.artifact)
    started = time.monotonic()
    try:
        result = run_dry(args, reader, device, artifact_sha) if args.dry_run else run_full(
            args, reader, device, artifact_sha
        )
    finally:
        reader.close()
    props = torch.cuda.get_device_properties(0)
    receipt = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "source": str(args.source.resolve()),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": artifact_sha,
        "capture": str(args.capture.resolve()),
        "device": {
            "name": props.name,
            "total_memory": props.total_memory,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "dtype": args.dtype,
        },
        "hyperparameters": {
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_scale": args.lora_alpha / args.rank,
            "dropout": 0.0,
            "lr": args.lr,
            "warmup": args.warmup,
            "schedule_steps": args.schedule_steps,
            "schedule": "linear warmup then cosine decay",
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "grad_norm": args.grad_norm,
        },
        "commands": command_lines(args),
        "wall_clock_seconds": time.monotonic() - started,
        "result": result,
    }
    atomic_json(args.out / "receipt.json", receipt)
    print(json.dumps({"receipt": str(args.out / "receipt.json"), "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
