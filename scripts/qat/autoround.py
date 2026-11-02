#!/usr/bin/env python3
"""Learn MQ4V2 rounding decisions and export them without a second RTN pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrea  # noqa: E402

MQ4V2_QTYPE = 44
GROUP_WEIGHTS = 256
GROUP_BYTES = 136


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--imatrix", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--heldout-samples", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--layers", help="comma-separated layer indices; omit for all")
    parser.add_argument("--max-layer", type=int, help="inclusive layer ceiling for staged export")
    parser.add_argument("--max-tensors", type=int)
    parser.add_argument("--pilot-only", action="store_true")
    return parser.parse_args()


def layer_number(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def generate_signs(seed: int, n: int, device: torch.device) -> torch.Tensor:
    state = seed
    values = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        values.append(1.0 if ((state >> 16) & 1) else -1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def fwht256(values: torch.Tensor, signs1: torch.Tensor, signs2: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % GROUP_WEIGHTS:
        raise ValueError(f"FWHT input K={values.shape[-1]} is not divisible by 256")
    shape = values.shape
    work = values.reshape(-1, GROUP_WEIGHTS) * signs1
    stride = 1
    while stride < GROUP_WEIGHTS:
        pairs = work.reshape(-1, GROUP_WEIGHTS // (2 * stride), 2 * stride)
        low = pairs[..., :stride].clone()
        high = pairs[..., stride:].clone()
        pairs[..., :stride] = low + high
        pairs[..., stride:] = low - high
        stride *= 2
    work.mul_(1.0 / 16.0).mul_(signs2)
    return work.reshape(shape)


def fake_a4(values: torch.Tensor) -> torch.Tensor:
    blocks = values.reshape(-1, 128)
    amax = blocks.abs().amax(dim=1)
    base = (amax / np.float32(7.0)) * np.float32(0.5)
    multipliers = torch.tensor(
        [
            1.0,
            struct.unpack("<f", struct.pack("<I", 0x3FA49249))[0],
            struct.unpack("<f", struct.pack("<I", 0x3FDB6DB7))[0],
            2.0,
        ],
        dtype=torch.float32,
        device=values.device,
    )
    scales = base[:, None] * multipliers[None, :]
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = torch.round(blocks[:, None, :] / safe_scales[:, :, None]).clamp_(-8, 7)
    dequantized = q * scales[:, :, None]

    # Mirror block_i4_128's four-values-per-lane FMA accumulation and XOR
    # 16,8,4,2,1 wave reduction. argmin keeps the first scale on ties.
    errors = torch.addcmul(blocks[:, None, :], -q, scales[:, :, None])
    lane_errors = errors.reshape(-1, 4, 32, 4)
    lane_mse = torch.zeros_like(lane_errors[..., 0])
    for element in range(4):
        lane_mse = torch.addcmul(
            lane_mse, lane_errors[..., element], lane_errors[..., element]
        )
    lanes = torch.arange(32, device=values.device)
    for offset in (16, 8, 4, 2, 1):
        prior = lane_mse.clone()
        lane_mse = lane_mse + prior[:, :, lanes ^ offset]
    selected = lane_mse[:, :, 0].argmin(dim=1)
    result = dequantized[torch.arange(blocks.shape[0], device=values.device), selected]
    result[amax == 0] = 0
    return result.reshape_as(values)


def fake_a8(values: torch.Tensor) -> torch.Tensor:
    # Runtime E4M3 uses a power-of-two row scale to the finite 224 rail.
    amax = values.abs().amax(dim=1, keepdim=True)
    ratio = torch.where(amax > 0, amax / 224.0, torch.ones_like(amax))
    scale = torch.pow(2.0, torch.ceil(torch.log2(ratio)))
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    return (values / scale).to(torch.float8_e4m3fnuz).to(torch.float32) * scale


def awq_scales(in_sum2: np.ndarray, alpha: float) -> np.ndarray:
    values = np.clip(in_sum2.astype(np.float64), 1e-12, 1e30)
    logs = (alpha * 0.5) * np.log(values)
    scales = np.exp(logs - logs.mean()).astype(np.float32)
    return np.clip(scales, np.float32(1e-2), np.float32(1e2))


def calibration_samples(in_sum2: np.ndarray, count: int, seed: int) -> np.ndarray:
    rms = np.sqrt(np.clip(in_sum2.astype(np.float64), 1e-12, 1e30)).astype(np.float32)
    norm = np.sqrt(np.mean(rms.astype(np.float64) ** 2))
    if math.isfinite(norm) and norm > 0:
        rms /= np.float32(norm)
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.integers(0, 2, size=(count, rms.size), dtype=np.int8)
    signs = signs.astype(np.float32) * 2.0 - 1.0
    return signs * rms[None, :]


class SourceWeights:
    def __init__(self, root: Path):
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        self.root = root
        self.weight_map: dict[str, str] = index["weight_map"]
        self.handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self.handles.get(shard)
        if handle is None:
            handle = safe_open(self.root / shard, framework="pt", device="cpu")
            self.handles[shard] = handle
        return handle.get_tensor(name)


def unpack_tensor(raw: bytes, m: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = m * k // GROUP_WEIGHTS
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(groups, GROUP_BYTES)
    headers = np.frombuffer(blocks[:, :8].copy().tobytes(), dtype="<f2").reshape(groups, 4)
    packed = blocks[:, 8:].copy()
    codes = np.empty((groups, GROUP_WEIGHTS), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return headers, codes.reshape(m, k), blocks.copy()


def store_codes(blocks: np.ndarray, codes: np.ndarray) -> bytes:
    flat = codes.reshape(-1, GROUP_WEIGHTS)
    blocks[:, 8:] = flat[:, 0::2] | (flat[:, 1::2] << 4)
    return blocks.tobytes()


def route_loss(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    a4: torch.Tensor,
    a8: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = q * scale + zero
    error4 = decoded @ a4.T - target
    error8 = decoded @ a8.T - target
    return error4.square().sum(), error8.square().sum(), error4, error8


def optimize_chunk(
    original: torch.Tensor,
    initial_codes: np.ndarray,
    headers: np.ndarray,
    z: torch.Tensor,
    a4: torch.Tensor,
    a8: torch.Tensor,
    z_hold: torch.Tensor,
    a4_hold: torch.Tensor,
    a8_hold: torch.Tensor,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, dict]:
    device = original.device
    rows, k = original.shape
    group_headers = torch.from_numpy(headers.astype(np.float32, copy=False)).to(device)
    scale = group_headers[:, [0, 2]].reshape(rows, k // 128).repeat_interleave(128, dim=1)
    zero = group_headers[:, [1, 3]].reshape(rows, k // 128).repeat_interleave(128, dim=1)
    q0 = torch.from_numpy(initial_codes.astype(np.float32, copy=False)).to(device)
    target = original @ z.T
    target_hold = original @ z_hold.T
    signal4 = target.square().sum().clamp_min(1e-30)
    signal8 = signal4
    signal4_hold = target_hold.square().sum().clamp_min(1e-30)
    signal8_hold = signal4_hold

    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = (original - zero) / safe_scale
    v = torch.zeros_like(original)
    q_at_zero = torch.floor(normalized + 0.5).clamp_(0, 15)
    valid = scale != 0
    parity_mismatches = int(((q_at_zero != q0) & valid).sum().item())
    # The artifact is the exact step-0 contract. Rare host/device boundary
    # differences retain its code and center V in the interval that emits it.
    if parity_mismatches:
        v = torch.where(valid, (q0 - normalized).clamp(-0.499, 0.499), v)

    checkpoints = {0, steps}
    checkpoints.update(step for step in (25, 50, 100, 200) if step <= steps)
    checkpoint_metrics: dict[int, dict] = {}
    best_q = q0.to(torch.uint8).clone()
    best_objective = math.inf

    def evaluate(step: int, q: torch.Tensor) -> None:
        nonlocal best_q, best_objective
        l4, l8, _, _ = route_loss(q, scale, zero, a4, a8, target)
        h4, h8, _, _ = route_loss(q, scale, zero, a4_hold, a8_hold, target_hold)
        rel4 = float((l4 / signal4).item())
        rel8 = float((l8 / signal8).item())
        held4 = float((h4 / signal4_hold).item())
        held8 = float((h8 / signal8_hold).item())
        objective = held4 + 0.25 * held8
        checkpoint_metrics[step] = {
            "train_l4": rel4,
            "train_l8": rel8,
            "heldout_l4": held4,
            "heldout_l8": held8,
            "heldout_objective": objective,
        }
        baseline_l8 = checkpoint_metrics[0]["heldout_l8"] if 0 in checkpoint_metrics else held8
        if held8 <= baseline_l8 * 1.01 and objective < best_objective:
            best_objective = objective
            best_q = q.to(torch.uint8).clone()

    evaluate(0, q0)
    q = q0
    for step in range(1, steps + 1):
        q = torch.floor(normalized + 0.5 + v).clamp_(0, 15)
        _, _, error4, error8 = route_loss(q, scale, zero, a4, a8, target)
        gradient = (error4 @ a4 + 0.25 * (error8 @ a8)) * scale
        raw = normalized + 0.5 + v
        ste_mask = valid & (raw >= 0) & (raw <= 15)
        step_lr = lr * (1.0 - (step - 1) / max(steps, 1))
        v.add_(-step_lr * gradient.sign() * ste_mask).clamp_(-0.5, 0.5)
        if step in checkpoints:
            selected = torch.floor(normalized + 0.5 + v).clamp_(0, 15)
            evaluate(step, selected)

    selected_np = best_q.cpu().numpy()
    changed = int(np.count_nonzero(selected_np != initial_codes))
    return selected_np, {
        "parity_mismatches": parity_mismatches,
        "codes_changed": changed,
        "checkpoint_metrics": checkpoint_metrics,
        "selected_heldout_objective": best_objective,
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    initial4 = sum(row["initial_heldout_l4_sum"] for row in rows)
    final4 = sum(row["final_heldout_l4_sum"] for row in rows)
    initial8 = sum(row["initial_heldout_l8_sum"] for row in rows)
    final8 = sum(row["final_heldout_l8_sum"] for row in rows)
    signal = sum(row["heldout_signal_sum"] for row in rows)
    return {
        "heldout_l4_initial": initial4 / max(signal, 1e-30),
        "heldout_l4_final": final4 / max(signal, 1e-30),
        "heldout_l4_improvement": 1.0 - final4 / max(initial4, 1e-30),
        "heldout_l8_initial": initial8 / max(signal, 1e-30),
        "heldout_l8_final": final8 / max(signal, 1e-30),
        "heldout_l8_ratio": final8 / max(initial8, 1e-30),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ROCm torch cannot see the allocated GPU")
    if args.pilot_only and args.output is not None:
        raise SystemExit("--pilot-only cannot be combined with --output")
    if not args.pilot_only and args.output is None:
        raise SystemExit("--output is required unless --pilot-only is used")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    selected_layers = None if args.layers is None else {int(x) for x in args.layers.split(",")}
    _, tensor_map = astrea.read_hfq_index(args.base, max_tensors=0)
    joined = astrea.match_imatrix_to_hfq(args.base, args.imatrix, max_tensors=0)
    imatrix_names = {item["hfq_name"]: item["imatrix_name"] for item in joined["matches"]}
    targets = []
    for name, item in tensor_map.items():
        layer = layer_number(name)
        if item["quant_type"] != MQ4V2_QTYPE or len(item["shape"]) != 2 or layer is None:
            continue
        if selected_layers is not None and layer not in selected_layers:
            continue
        if args.max_layer is not None and layer > args.max_layer:
            continue
        if name not in imatrix_names:
            continue
        targets.append((name, item))
    targets.sort(key=lambda pair: (layer_number(pair[0]), pair[0]))
    if args.max_tensors is not None:
        targets = targets[: args.max_tensors]
    if not targets:
        raise SystemExit("no eligible MQ4V2 tensors selected")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base, args.output)

    source = SourceWeights(args.source)
    signs1 = generate_signs(42, GROUP_WEIGHTS, device)
    signs2 = generate_signs(1042, GROUP_WEIGHTS, device)
    base_file = args.base.open("rb")
    base_map = mmap.mmap(base_file.fileno(), 0, access=mmap.ACCESS_READ)
    output_file = args.output.open("r+b") if args.output is not None else None
    output_map = mmap.mmap(output_file.fileno(), 0) if output_file is not None else None

    started = time.monotonic()
    tensor_reports = []
    aggregate_rows = []
    try:
        for tensor_index, (name, item) in enumerate(targets, 1):
            tensor_started = time.monotonic()
            m, k = map(int, item["shape"])
            raw = base_map[item["data_offset"] : item["data_offset"] + item["data_size"]]
            headers, codes, blocks = unpack_tensor(raw, m, k)
            logical = imatrix_names[name]
            in_sum2 = np.asarray(
                astrea.read_gguf_f32_tensor(args.imatrix, f"{logical}.in_sum2"), dtype=np.float32
            )
            if in_sum2.size != k:
                raise ValueError(f"{name}: imatrix K={in_sum2.size}, tensor K={k}")
            sidecar = name[: -len(".weight")] + ".awq_scale.weight"
            scales_np = awq_scales(in_sum2, args.alpha) if sidecar in tensor_map else np.ones(k, np.float32)
            train_x = calibration_samples(in_sum2, args.train_samples, 42)
            heldout_x = calibration_samples(in_sum2, args.heldout_samples, 1042)
            scales = torch.from_numpy(scales_np).to(device)
            train = torch.from_numpy(train_x).to(device)
            heldout = torch.from_numpy(heldout_x).to(device)
            z = fwht256(train / scales, signs1, signs2)
            z_hold = fwht256(heldout / scales, signs1, signs2)
            a4 = fake_a4(z)
            a4_hold = fake_a4(z_hold)
            a8 = fake_a8(z)
            a8_hold = fake_a8(z_hold)
            weight = source.get(name)
            if tuple(weight.shape) != (m, k):
                raise ValueError(f"{name}: source shape {tuple(weight.shape)} != {(m, k)}")

            chunk_reports = []
            for row_start in range(0, m, args.chunk_rows):
                row_end = min(row_start + args.chunk_rows, m)
                original = weight[row_start:row_end].to(device=device, dtype=torch.float32)
                original.mul_(scales)
                original = fwht256(original, signs1, signs2)
                group_start = row_start * (k // GROUP_WEIGHTS)
                group_end = row_end * (k // GROUP_WEIGHTS)
                selected, chunk_report = optimize_chunk(
                    original,
                    codes[row_start:row_end],
                    headers[group_start:group_end],
                    z,
                    a4,
                    a8,
                    z_hold,
                    a4_hold,
                    a8_hold,
                    args.steps,
                    args.lr,
                )
                initial_q = torch.from_numpy(codes[row_start:row_end].astype(np.float32)).to(device)
                final_q = torch.from_numpy(selected.astype(np.float32)).to(device)
                chunk_headers = torch.from_numpy(headers[group_start:group_end].astype(np.float32)).to(device)
                scale = chunk_headers[:, [0, 2]].reshape(row_end - row_start, k // 128).repeat_interleave(128, 1)
                zero = chunk_headers[:, [1, 3]].reshape(row_end - row_start, k // 128).repeat_interleave(128, 1)
                target_hold = original @ z_hold.T
                signal = float(target_hold.square().sum().item())
                initial4, initial8, _, _ = route_loss(initial_q, scale, zero, a4_hold, a8_hold, target_hold)
                final4, final8, _, _ = route_loss(final_q, scale, zero, a4_hold, a8_hold, target_hold)
                aggregate_row = {
                    "initial_heldout_l4_sum": float(initial4.item()),
                    "final_heldout_l4_sum": float(final4.item()),
                    "initial_heldout_l8_sum": float(initial8.item()),
                    "final_heldout_l8_sum": float(final8.item()),
                    "heldout_signal_sum": signal,
                }
                aggregate_rows.append(aggregate_row)
                chunk_report.update(aggregate_row)
                chunk_reports.append(chunk_report)
                codes[row_start:row_end] = selected
                del original, initial_q, final_q, scale, zero, target_hold

            if output_map is not None:
                encoded = store_codes(blocks, codes)
                output_map[item["data_offset"] : item["data_offset"] + item["data_size"]] = encoded
            tensor_report = {
                "name": name,
                "layer": layer_number(name),
                "shape": [m, k],
                "seconds": time.monotonic() - tensor_started,
                "codes_changed": sum(chunk["codes_changed"] for chunk in chunk_reports),
                "parity_mismatches": sum(chunk["parity_mismatches"] for chunk in chunk_reports),
                "metrics": aggregate_metrics(chunk_reports),
            }
            tensor_reports.append(tensor_report)
            print(
                f"[{tensor_index}/{len(targets)}] {name} {tensor_report['seconds']:.1f}s "
                f"changed={tensor_report['codes_changed']} parity={tensor_report['parity_mismatches']} "
                f"heldout_l4_improvement={tensor_report['metrics']['heldout_l4_improvement']:.4%}",
                flush=True,
            )
    finally:
        if output_map is not None:
            output_map.flush()
            output_map.close()
        if output_file is not None:
            output_file.close()
        base_map.close()
        base_file.close()

    total_seconds = time.monotonic() - started
    output_sha256 = None
    output_bytes = None
    if args.output is not None:
        digest = hashlib.sha256()
        with args.output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        output_sha256 = digest.hexdigest()
        output_bytes = args.output.stat().st_size
    report = {
        "schema": "hipfire.qat.autoround.v1",
        "base": str(args.base),
        "source": str(args.source),
        "imatrix": str(args.imatrix),
        "output": str(args.output) if args.output is not None else None,
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "alpha": args.alpha,
        "optimizer": "SignSGD",
        "steps": args.steps,
        "lr": args.lr,
        "lr_schedule": "linear_to_zero",
        "clip_optimization": False,
        "train_samples": args.train_samples,
        "heldout_samples": args.heldout_samples,
        "calibration": "imatrix RMS with deterministic PCG64 Rademacher signs (train seed 42, heldout seed 1042)",
        "loss": "linear output NMSE: exact-A4 surrogate + 0.25 * E4M3 A8 surrogate",
        "layers": sorted(selected_layers) if selected_layers is not None else None,
        "max_layer": args.max_layer,
        "total_seconds": total_seconds,
        "aggregate": aggregate_metrics(aggregate_rows),
        "tensors": tensor_reports,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"total_seconds": total_seconds, "aggregate": report["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
