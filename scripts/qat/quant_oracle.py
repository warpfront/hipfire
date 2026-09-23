#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Exact training oracle for HipFire's fixed FWHT, A4, and E4M3 routes.

The public tensor convention is column-token ``[K,T]``.  ``a4_forward``
returns ``(dq_f32, {d, sum, qs, blocks})``.  A4 block metadata is ordered
``[K/128,T]`` and ``qs`` adds a final 64-byte dimension.  CUDA execution
launches the shipping HIP fragment; the CPU path deliberately reproduces its
wave32 arithmetic and exists for fixtures, not high-throughput fitting.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any

import numpy as np
import torch

FUSED4 = "gfx1201_fused4"
STANDALONE8 = "standalone8"
_SITES = {FUSED4: 0, STANDALONE8: 1}
_ROOT = Path(__file__).resolve().parents[2]
_HIP_SOURCE = Path(__file__).with_suffix(".hip")
_IMPORTED_SOURCES = (
    _ROOT / "kernels/src/block_i4_128_quant.hip",
    _ROOT / "kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip",
)
_DEFAULT_CACHE = Path("/home/kaden/qcal/qat/l3/oracle")
_LIB: ctypes.CDLL | None = None

_LIBM = ctypes.CDLL("libm.so.6")
_LIBM.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_LIBM.fmaf.restype = ctypes.c_float
_LIBM.nearbyintf.argtypes = (ctypes.c_float,)
_LIBM.nearbyintf.restype = ctypes.c_float


def _f32(value: float | np.float32) -> np.float32:
    return np.float32(value)


def _add(a: float, b: float) -> np.float32:
    return np.float32(np.float32(a) + np.float32(b))


def _mul(a: float, b: float) -> np.float32:
    return np.float32(np.float32(a) * np.float32(b))


def _div(a: float, b: float) -> np.float32:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.float32(np.divide(np.float32(a), np.float32(b)))


def _fma(a: float, b: float, c: float) -> np.float32:
    return np.float32(_LIBM.fmaf(float(a), float(b), float(c)))


def _rne(value: float) -> np.float32:
    return np.float32(_LIBM.nearbyintf(float(value)))


def _site_id(site: str) -> int:
    try:
        return _SITES[site]
    except KeyError as exc:
        raise ValueError(f"site must be one of {tuple(_SITES)}, got {site!r}") from exc


def _validate_z(z: torch.Tensor, multiple: int) -> tuple[int, int]:
    if not isinstance(z, torch.Tensor):
        raise TypeError("oracle input must be a torch.Tensor")
    if z.dtype != torch.float32:
        raise TypeError(f"oracle input must be float32, got {z.dtype}")
    if z.ndim != 2:
        raise ValueError(f"oracle input must have shape [K,T], got {tuple(z.shape)}")
    K, T = map(int, z.shape)
    if K <= 0 or T <= 0 or K % multiple:
        raise ValueError(f"K must be positive and divisible by {multiple}, got K={K}, T={T}")
    return K, T


def _build_key() -> str:
    digest = hashlib.sha256()
    for path in (_HIP_SOURCE, *_IMPORTED_SOURCES):
        digest.update(path.read_bytes())
    digest.update(b"gfx1201-O3-v1")
    return digest.hexdigest()[:16]


def build_oracle(cache_dir: str | os.PathLike[str] = _DEFAULT_CACHE, force: bool = False) -> Path:
    """Build the training-only HIP launch library for gfx1201."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"quant_oracle_gfx1201_{_build_key()}.so"
    if output.exists() and not force:
        return output
    hipcc = shutil.which("hipcc") or "/opt/rocm/core/bin/hipcc"
    command = [
        hipcc,
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        str(_HIP_SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, cwd=_ROOT, check=True)
    return output


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(build_oracle()))
    pointer = ctypes.c_void_p
    lib.quant_oracle_a4.argtypes = (
        pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer)
    lib.quant_oracle_a4.restype = ctypes.c_int
    lib.quant_oracle_shipping8.argtypes = (
        pointer, pointer, ctypes.c_int, ctypes.c_int, pointer)
    lib.quant_oracle_shipping8.restype = ctypes.c_int
    for name in ("quant_oracle_a8_f32", "quant_oracle_a8_f16"):
        fn = getattr(lib, name)
        fn.argtypes = (
            pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, pointer)
        fn.restype = ctypes.c_int
    _LIB = lib
    return lib


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_ptr(device: torch.device) -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


def _check_hip(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with hipError_t={status}")


def _metadata_from_blocks(blocks: torch.Tensor, K: int, T: int) -> dict[str, torch.Tensor]:
    flat = blocks.reshape(-1, 72)
    d = flat[:, 0:4].contiguous().view(torch.float32).reshape(K // 128, T)
    sums = flat[:, 4:8].contiguous().view(torch.int32).reshape(K // 128, T)
    qs = flat[:, 8:72].reshape(K // 128, T, 64)
    return {"d": d, "sum": sums, "qs": qs, "blocks": blocks}


def _quantize_block_cpu(values: np.ndarray, site: str) -> tuple[bytes, np.ndarray]:
    if values.shape != (128,) or values.dtype != np.float32:
        raise ValueError("CPU A4 block input must be float32[128]")
    lane_amax = np.empty(32, dtype=np.float32)
    for lane in range(32):
        base = lane * 4
        lane_amax[lane] = np.maximum.reduce(np.abs(values[base : base + 4]))
    for offset in (16, 8, 4, 2, 1):
        prior = lane_amax.copy()
        for lane in range(32):
            lane_amax[lane] = np.maximum(lane_amax[lane], prior[lane ^ offset])
    amax = lane_amax[0]

    best_d = np.float32(1.0)
    if amax != np.float32(0.0):
        candidate_base = _mul(_div(amax, 7.0), 0.5)
        if site == FUSED4:
            multipliers = tuple(
                np.array([bits], dtype=np.uint32).view(np.float32)[0]
                for bits in (0x3F800000, 0x3FA49249, 0x3FDB6DB7, 0x40000000)
            )
        else:
            multipliers = tuple(_add(1.0, _div(j, 7.0)) for j in range(8))
        best_mse = np.float32(1.0e30)
        for multiplier in multipliers:
            d = _mul(candidate_base, multiplier)
            lane_mse = np.zeros(32, dtype=np.float32)
            for lane in range(32):
                mse = np.float32(0.0)
                for e in range(4):
                    value = values[lane * 4 + e]
                    q = np.clip(_rne(_div(value, d)), -8.0, 7.0).astype(np.float32)
                    error = _fma(-q, d, value)
                    mse = _fma(error, error, mse)
                lane_mse[lane] = mse
            for offset in (16, 8, 4, 2, 1):
                prior = lane_mse.copy()
                for lane in range(32):
                    lane_mse[lane] = _add(lane_mse[lane], prior[lane ^ offset])
            if lane_mse[0] < best_mse:
                best_mse = lane_mse[0]
                best_d = d

    q4 = np.zeros(128, dtype=np.int32)
    if amax != np.float32(0.0):
        for i, value in enumerate(values):
            q4[i] = int(np.clip(_rne(_div(value, best_d)), -8.0, 7.0))
    total = int(q4.sum(dtype=np.int32))
    qs = np.empty(64, dtype=np.uint8)
    for i in range(64):
        qs[i] = (int(q4[2 * i]) & 15) | ((int(q4[2 * i + 1]) & 15) << 4)
    block = struct.pack("<fi", float(best_d), total) + qs.tobytes()
    dq = np.asarray([_mul(best_d, q) for q in q4], dtype=np.float32)
    return block, dq


def _a4_forward_cpu(z: torch.Tensor, site: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K, T = _validate_z(z, 128)
    source = z.detach().contiguous().numpy()
    raw = bytearray((K // 128) * T * 72)
    dq = np.empty((K, T), dtype=np.float32)
    for group in range(K // 128):
        for token in range(T):
            block, block_dq = _quantize_block_cpu(
                source[group * 128 : (group + 1) * 128, token].copy(), site)
            offset = (group * T + token) * 72
            raw[offset : offset + 72] = block
            dq[group * 128 : (group + 1) * 128, token] = block_dq
    blocks = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(K // 128, T, 72)
    return torch.from_numpy(dq), _metadata_from_blocks(blocks, K, T)


def a4_forward(z: torch.Tensor, site: str = FUSED4) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run exact A4 forward for one producer site on a ``float32[K,T]`` tensor."""
    K, T = _validate_z(z, 128)
    site_number = _site_id(site)
    if not z.is_cuda:
        return _a4_forward_cpu(z, site)
    zc = z.contiguous()
    dq = torch.empty_like(zc)
    blocks = torch.empty((K // 128, T, 72), dtype=torch.uint8, device=zc.device)
    status = _library().quant_oracle_a4(
        _ptr(zc), _ptr(dq), _ptr(blocks), K, T, site_number, _stream_ptr(zc.device))
    _check_hip(status, "quant_oracle_a4")
    return dq, _metadata_from_blocks(blocks, K, T)


def shipping_standalone8_dump(z: torch.Tensor) -> torch.Tensor:
    """Dump raw 72-byte blocks from the unmodified shipping standalone kernel."""
    K, T = _validate_z(z, 128)
    if not z.is_cuda:
        raise ValueError("shipping kernel dump requires a CUDA/ROCm tensor")
    token_major = z.transpose(0, 1).contiguous()
    blocks = torch.empty((K // 128, T, 72), dtype=torch.uint8, device=z.device)
    status = _library().quant_oracle_shipping8(
        _ptr(token_major), _ptr(blocks), K, T, _stream_ptr(z.device))
    _check_hip(status, "quantize_int4_mmq_ds128")
    return blocks


def _decode_e4m3(qs: torch.Tensor) -> torch.Tensor:
    bits = qs.to(torch.int32)
    sign = torch.where((bits & 0x80) != 0, -1.0, 1.0)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 7
    subnormal = mantissa.to(torch.float32) * (2.0 ** -9)
    normal = torch.ldexp(1.0 + mantissa.to(torch.float32) * 0.125, exponent - 7)
    decoded = torch.where(exponent == 0, subnormal, normal)
    decoded = torch.where((exponent == 15) & (mantissa == 7), torch.nan, decoded)
    return decoded * sign


def _row_scale_cpu(amax: np.float32) -> np.float32:
    if not amax > np.float32(0.0):
        return np.float32(1.0)
    reciprocal_224 = np.float32(1.0 / 224.0)
    t = _mul(amax, reciprocal_224)
    bits = struct.unpack("<I", struct.pack("<f", float(t)))[0]
    exponent = ((bits >> 23) & 0xFF) - 127 + (1 if (bits & 0x7FFFFF) else 0)
    exponent = max(-31, min(9, exponent))
    return np.float32(math.ldexp(1.0, exponent))


def _wave_sum_128(values: np.ndarray) -> np.float32:
    lanes = np.empty(32, dtype=np.float32)
    for lane in range(32):
        base = lane * 4
        lanes[lane] = _add(_add(_add(values[base], values[base + 1]), values[base + 2]), values[base + 3])
    for offset in (16, 8, 4, 2, 1):
        prior = lanes.copy()
        for lane in range(32):
            lanes[lane] = _add(lanes[lane], prior[lane ^ offset])
    return lanes[0]


def _a8_forward_cpu(z: torch.Tensor, source: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K, T = _validate_z(z, 256)
    token_major = z.transpose(0, 1).contiguous()
    if source == "f16":
        token_major = token_major.to(torch.float16).to(torch.float32)
    elif source != "f32":
        raise ValueError("A8 source must be 'f32' or 'f16'")
    qs = torch.empty((T, K), dtype=torch.uint8)
    scales = torch.empty(T, dtype=torch.float32)
    half_sums = torch.empty((T, K // 256, 2), dtype=torch.float32)
    decoded = torch.empty((T, K), dtype=torch.float32)
    for token in range(T):
        row = token_major[token]
        scale = _row_scale_cpu(np.float32(row.abs().max().item()))
        scales[token] = float(scale)
        row_qs = (row / float(scale)).to(torch.float8_e4m3fn).view(torch.uint8)
        qs[token].copy_(row_qs)
        row_decoded = _decode_e4m3(row_qs)
        decoded[token].copy_(row_decoded * float(scale))
        raw_decoded = row_decoded.numpy()
        for group in range(K // 256):
            for half in range(2):
                begin = group * 256 + half * 128
                half_sums[token, group, half] = float(_wave_sum_128(raw_decoded[begin : begin + 128]))
    return decoded.transpose(0, 1).contiguous(), {
        "scale": scales,
        "half_sums": half_sums,
        "qs": qs.transpose(0, 1).contiguous(),
        "qs_token_major": qs,
    }


def a8_forward(z: torch.Tensor, source: str = "f32") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the shipping E4M3 per-token power-of-two-to-224 pre-pass."""
    K, T = _validate_z(z, 256)
    if source not in ("f32", "f16"):
        raise ValueError("A8 source must be 'f32' or 'f16'")
    if not z.is_cuda:
        return _a8_forward_cpu(z, source)
    token_major = z.transpose(0, 1).contiguous()
    if source == "f16":
        token_major = token_major.to(torch.float16)
    qs = torch.empty((T, K), dtype=torch.uint8, device=z.device)
    scales = torch.empty(T, dtype=torch.float32, device=z.device)
    half_sums = torch.empty((T, K // 256, 2), dtype=torch.float32, device=z.device)
    function = _library().quant_oracle_a8_f16 if source == "f16" else _library().quant_oracle_a8_f32
    status = function(
        _ptr(token_major), _ptr(qs), _ptr(half_sums), _ptr(scales), K, T, _stream_ptr(z.device))
    _check_hip(status, f"pack_{source}_to_fp8_mq4v2_gfx12")
    decoded = _decode_e4m3(qs) * scales[:, None]
    return decoded.transpose(0, 1).contiguous(), {
        "scale": scales,
        "half_sums": half_sums,
        "qs": qs.transpose(0, 1).contiguous(),
        "qs_token_major": qs,
    }


def fwht_signs(seed: int, n: int = 256, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Generate the runtime's signed LCG table (default seeds are 42/1042)."""
    state = seed & 0xFFFFFFFF
    signs: list[float] = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        signs.append(1.0 if ((state >> 16) & 1) else -1.0)
    return torch.tensor(signs, dtype=torch.float32, device=device)


def fwht256(z: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply fixed ``D2 H256 D1 / 16`` (or its transpose) over K groups."""
    K, T = _validate_z(z, 256)
    groups = K // 256
    signs1 = fwht_signs(42, device=z.device).reshape(1, 256, 1)
    signs2 = fwht_signs(1042, device=z.device).reshape(1, 256, 1)
    y = z.reshape(groups, 256, T)
    y = y * (signs2 if inverse else signs1)
    stride = 1
    while stride < 256:
        shaped = y.reshape(groups, 256 // (2 * stride), 2, stride, T)
        a = shaped[:, :, 0]
        b = shaped[:, :, 1]
        y = torch.stack((a + b, a - b), dim=2).reshape(groups, 256, T)
        stride *= 2
    y = y * np.float32(0.0625) * (signs1 if inverse else signs2)
    return y.reshape(K, T)


class _A4STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, site: str) -> torch.Tensor:
        dq, metadata = a4_forward(z, site)
        d = metadata["d"].repeat_interleave(128, dim=0)
        ctx.save_for_backward((z >= -8.0 * d) & (z <= 7.0 * d))
        return dq

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (in_range,) = ctx.saved_tensors
        return grad_output * in_range, None


def a4_ste(z: torch.Tensor, site: str = FUSED4) -> torch.Tensor:
    """A4 exact forward with identity-in-grid/zero-beyond-clip STE backward."""
    _site_id(site)
    return _A4STE.apply(z, site)


class _A8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, source: str) -> torch.Tensor:
        dq, metadata = a8_forward(z, source)
        scales = metadata["scale"].reshape(1, -1)
        ctx.save_for_backward((z >= -448.0 * scales) & (z <= 448.0 * scales))
        return dq

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (in_range,) = ctx.saved_tensors
        return grad_output * in_range, None


def a8_ste(z: torch.Tensor, source: str = "f32") -> torch.Tensor:
    """E4M3 exact forward with the route's clipped identity STE backward."""
    return _A8STE.apply(z, source)


def _rust_dump_fixture() -> torch.Tensor:
    state = 0x12345678
    values: list[float] = []
    for _ in range(128):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        centered = ((state >> 8) & 0xFFFF) - 32768
        values.append(float(np.float32(centered) * np.float32(1.0 / 4096.0)))
    return torch.tensor(values, dtype=torch.float32).reshape(128, 1)


def _fixtures() -> dict[str, torch.Tensor]:
    zero = torch.zeros((256, 2), dtype=torch.float32)

    ties = torch.zeros((128, 1), dtype=torch.float32)
    tie_pattern = torch.tensor([-7.5, -6.5, -1.5, -0.5, 0.5, 1.5, 6.5, 7.5], dtype=torch.float32)
    ties[:, 0] = tie_pattern.repeat(16)

    rails = torch.linspace(-96.0, 96.0, 128, dtype=torch.float32).reshape(128, 1)
    rails[0, 0], rails[1, 0], rails[2, 0], rails[3, 0] = -1.0e6, 1.0e6, -0.0, 0.0

    cancellation = torch.empty((128, 1), dtype=torch.float32)
    for i in range(64):
        value = np.float32((i + 1) * (1.0 / 17.0))
        cancellation[2 * i, 0] = float(value)
        cancellation[2 * i + 1, 0] = float(-value)

    boundary = torch.zeros((256, 2), dtype=torch.float32)
    boundary[126:130, 0] = torch.tensor([3.25, -7.75, 11.5, -13.0])
    boundary[254:256, 0] = torch.tensor([0.03125, -0.03125])
    boundary[:, 1] = torch.arange(256, dtype=torch.float32).remainder(19) - 9.0

    x = torch.sin(torch.arange(512, dtype=torch.float32).reshape(256, 2) * np.float32(0.03125))
    scales = torch.exp(torch.linspace(-1.0, 1.0, 256, dtype=torch.float32))
    scales = scales / torch.exp(torch.log(scales).mean())
    shared_s = fwht256(x / scales[:, None])

    tiny = torch.nextafter(torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))
    underflow = torch.where(
        (torch.arange(128) & 1).bool(), -tiny, tiny).to(torch.float32).reshape(128, 1)

    return {
        "rust_dump": _rust_dump_fixture(),
        "zero": zero,
        "ties": ties,
        "rails": rails,
        "cancellation": cancellation,
        "half_group_boundary": boundary,
        "shared_s": shared_s,
        "underflow": underflow,
    }


_RUST_EXPECTED = bytes([
    0xE5, 0x54, 0x7A, 0x3F, 0x05, 0x00, 0x00, 0x00, 0xBC, 0xE6, 0x36, 0x59,
    0x5B, 0xD0, 0x8F, 0x68, 0x01, 0x3E, 0x48, 0x9D, 0x72, 0x1E, 0x70, 0xFC,
    0xC5, 0x1C, 0xFA, 0x7B, 0x2F, 0x0A, 0xDC, 0x5A, 0x18, 0x1A, 0xB4, 0x7D,
    0x15, 0xC2, 0xAE, 0x60, 0x3F, 0x17, 0x22, 0x43, 0x4A, 0xE5, 0x25, 0x70,
    0xA1, 0x06, 0x27, 0x91, 0xC8, 0x19, 0xD5, 0x2E, 0x5B, 0x6C, 0xB6, 0xBE,
    0x1E, 0x45, 0x63, 0x25, 0x4C, 0xA6, 0x4A, 0x07, 0xEC, 0xA7, 0xEC, 0x62,
])


def _bytes_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _a8_receipt_bytes(metadata: dict[str, torch.Tensor]) -> bytes:
    return b"".join(
        tensor.detach().cpu().contiguous().numpy().tobytes()
        for tensor in (metadata["scale"], metadata["half_sums"], metadata["qs_token_major"])
    )


def _ste_gradient_check(device: torch.device, site: str) -> bool:
    z = torch.linspace(-17.0, 19.0, 256, dtype=torch.float32, device=device).reshape(128, 2)
    z.requires_grad_(True)
    weights = torch.linspace(0.25, 1.25, z.numel(), dtype=torch.float32, device=device).reshape_as(z)
    a4_ste(z, site).mul(weights).sum().backward()
    _, metadata = a4_forward(z.detach(), site)
    d = metadata["d"].repeat_interleave(128, dim=0)
    expected = weights * ((z.detach() >= -8.0 * d) & (z.detach() <= 7.0 * d))
    return bool(torch.equal(z.grad, expected))


def self_test(device: str = "cuda") -> dict[str, Any]:
    """Run fixture parity, shipping-kernel parity, fixed-transform, and STE checks."""
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm device unavailable")
        build_oracle()
    rows: list[dict[str, Any]] = []
    all_pass = True
    fixtures = _fixtures()
    for name, cpu_z in fixtures.items():
        for site in (FUSED4, STANDALONE8):
            _, cpu_meta = a4_forward(cpu_z, site)
            if target.type == "cuda":
                _, device_meta = a4_forward(cpu_z.to(target), site)
                torch.cuda.synchronize(target)
                cpu_device = torch.equal(cpu_meta["blocks"], device_meta["blocks"].cpu())
                shipping = None
                if site == STANDALONE8:
                    shipping_blocks = shipping_standalone8_dump(cpu_z.to(target))
                    torch.cuda.synchronize(target)
                    shipping = torch.equal(device_meta["blocks"], shipping_blocks)
            else:
                device_meta = cpu_meta
                cpu_device = True
                shipping = None
            rust_match = None
            if name == "rust_dump" and site == FUSED4:
                rust_match = cpu_meta["blocks"].reshape(-1).numpy().tobytes() == _RUST_EXPECTED
            passed = cpu_device and shipping is not False and rust_match is not False
            all_pass &= passed
            rows.append({
                "fixture": name,
                "route": site,
                "sha256": _bytes_hash(device_meta["blocks"]),
                "cpu_device": cpu_device,
                "shipping_kernel": shipping,
                "rust_replica": rust_match,
                "pass": passed,
            })

    a8_input = fixtures["half_group_boundary"]
    for source in ("f32", "f16"):
        _, cpu_meta = a8_forward(a8_input, source)
        if target.type == "cuda":
            _, device_meta = a8_forward(a8_input.to(target), source)
            torch.cuda.synchronize(target)
            parity = all(
                torch.equal(cpu_meta[key], device_meta[key].cpu())
                for key in ("scale", "half_sums", "qs_token_major")
            )
        else:
            device_meta = cpu_meta
            parity = True
        payload = _a8_receipt_bytes(device_meta)
        all_pass &= parity
        rows.append({
            "fixture": "half_group_boundary",
            "route": f"e4m3_{source}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "cpu_device": parity,
            "shipping_kernel": True if target.type == "cuda" else None,
            "rust_replica": None,
            "pass": parity,
        })

    transform_input = torch.arange(512, dtype=torch.float32).reshape(256, 2) * np.float32(1.0 / 127.0)
    transformed = fwht256(transform_input.to(target))
    restored = fwht256(transformed, inverse=True)
    fwht_error = float((restored - transform_input.to(target)).abs().max().item())
    fwht_hash = _bytes_hash(transformed)
    fwht_pass = fwht_error <= 2.0e-5
    all_pass &= fwht_pass

    ste = {site: _ste_gradient_check(target, site) for site in (FUSED4, STANDALONE8)}
    all_pass &= all(ste.values())
    return {
        "all_pass": all_pass,
        "device": str(target),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "arch": torch.cuda.get_device_properties(target).gcnArchName if target.type == "cuda" else None,
        "library": str(build_oracle()) if target.type == "cuda" else None,
        "fixtures": rows,
        "fwht256": {"sha256": fwht_hash, "inverse_max_abs": fwht_error, "pass": fwht_pass},
        "ste_gradient": ste,
    }


def _print_receipt(receipt: dict[str, Any]) -> None:
    print("fixture                 route              sha256            cpu/dev shipping rust pass")
    for row in receipt["fixtures"]:
        def mark(value: Any) -> str:
            return "-" if value is None else ("Y" if value else "N")
        print(
            f"{row['fixture']:<23} {row['route']:<18} {row['sha256'][:16]} "
            f"{mark(row['cpu_device']):>7} {mark(row['shipping_kernel']):>8} "
            f"{mark(row['rust_replica']):>4} {mark(row['pass']):>4}")
    print(f"fwht256 {receipt['fwht256']}")
    print(f"ste_gradient {receipt['ste_gradient']}")
    print(f"all_pass={receipt['all_pass']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()
    if args.force_build and torch.cuda.is_available():
        build_oracle(force=True)
    receipt = self_test(args.device)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        _print_receipt(receipt)
    return 0 if receipt["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
