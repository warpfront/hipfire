#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free implementations of the pinned Qwen4Exp reference equations.

These are intentionally compact layerwise equations, not a model loader.  The
source of truth is recorded by the generator manifest:
Transformers 93c8b7b485963a10800c91f55304db6be211c2bd and vLLM
 e126687a9a828d513c01a07cd69f025f27d63280.  The functions below mirror the
operations used by those references while keeping fixture dimensions small
(except for the real 16-head PLE row width, which is still only a few rows).
"""

from __future__ import annotations

import math
import random
import struct
from typing import Iterable, Sequence

try:
    from .schema import FixtureError, Tensor, tensor
except ImportError:  # direct execution via generate_fixtures.py
    from schema import FixtureError, Tensor, tensor  # type: ignore


MASK64 = (1 << 64) - 1
SIGNED64 = 1 << 63
EOS_TOKEN_ID = 248044
NG_SCALE = (23703573157769, 20109073645365, 8052911324071)
NG_HEAD_VOCAB_SIZES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
NG_HEAD_OFFSETS = tuple(
    [0]
    + [sum(NG_HEAD_VOCAB_SIZES[:i]) for i in range(1, len(NG_HEAD_VOCAB_SIZES))]
)
NG_VALID_ROWS = sum(NG_HEAD_VOCAB_SIZES)
NG_PADDED_ROWS = 320001536


def f32(value: float) -> float:
    """Round one scalar exactly as a stored IEEE-754 binary32 value."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def bf16_word(value: float) -> int:
    """Round float32 to a BF16 storage word (round-to-nearest-even)."""

    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16) & 0xFFFF


def bf16_value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (int(word) & 0xFFFF) << 16))[0]


def bf16_tensor(values: Iterable[float], shape: Sequence[int]) -> Tensor:
    return tensor(shape, (bf16_word(v) for v in values), "uint16")


def bf16_to_f32(values: Tensor) -> Tensor:
    if values.dtype != "uint16":
        raise FixtureError(f"BF16 storage must be uint16 words, got {values.dtype}")
    return tensor(values.shape, (bf16_value(v) for v in values.data), "float32")


def rng_uniform(rng: random.Random, shape: Sequence[int], scale: float = 1.0) -> Tensor:
    count = math.prod(shape)
    return tensor(shape, (f32((rng.random() * 2.0 - 1.0) * scale) for _ in range(count)), "float32")


def rng_normal(rng: random.Random, shape: Sequence[int], scale: float = 1.0) -> Tensor:
    # Box-Muller avoids version-dependent Random.gauss() cache details.
    count = math.prod(shape)
    values: list[float] = []
    while len(values) < count:
        u1 = max(rng.random(), 2.0 ** -53)
        u2 = rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        values.extend((f32(radius * math.cos(2.0 * math.pi * u2) * scale), f32(radius * math.sin(2.0 * math.pi * u2) * scale)))
    return tensor(shape, values[:count], "float32")


def _offset(shape: Sequence[int], indices: Sequence[int]) -> int:
    if len(shape) != len(indices):
        raise FixtureError(f"index rank {len(indices)} does not match shape {tuple(shape)}")
    result = 0
    for dim, index in zip(shape, indices):
        if index < 0 or index >= dim:
            raise FixtureError(f"index {indices} outside shape {tuple(shape)}")
        result = result * dim + index
    return result


def at(value: Tensor, *indices: int) -> object:
    return value.data[_offset(value.shape, indices)]


def as_f32(value: Tensor) -> Tensor:
    return tensor(value.shape, (f32(float(v)) for v in value.data), "float32")


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise FixtureError(f"matmul shapes {a.shape} and {b.shape} are incompatible")
    rows, inner, cols = a.shape[0], a.shape[1], b.shape[1]
    out: list[float] = []
    for i in range(rows):
        arow = a.data[i * inner : (i + 1) * inner]
        for j in range(cols):
            total = 0.0
            for k, left in enumerate(arow):
                total += float(left) * float(b.data[k * cols + j])
            out.append(f32(total))
    return tensor((rows, cols), out, "float32")


def transpose2(value: Tensor) -> Tensor:
    if value.ndim != 2:
        raise FixtureError("transpose2 expects a matrix")
    rows, cols = value.shape
    return tensor((cols, rows), (value.data[r * cols + c] for c in range(cols) for r in range(rows)), value.dtype)


def add(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise FixtureError(f"add shapes {a.shape} and {b.shape} differ")
    dtype = "float32" if a.dtype == "float32" or b.dtype == "float32" else a.dtype
    return tensor(a.shape, (f32(float(x) + float(y)) if dtype == "float32" else int(x) + int(y) for x, y in zip(a.data, b.data)), dtype)


def sub(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise FixtureError(f"sub shapes {a.shape} and {b.shape} differ")
    return tensor(a.shape, (f32(float(x) - float(y)) for x, y in zip(a.data, b.data)), "float32")


def mul(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise FixtureError(f"mul shapes {a.shape} and {b.shape} differ")
    return tensor(a.shape, (f32(float(x) * float(y)) for x, y in zip(a.data, b.data)), "float32")


def scale(value: Tensor, factor: float) -> Tensor:
    return tensor(value.shape, (f32(float(x) * factor) for x in value.data), "float32")


def map_values(value: Tensor, function) -> Tensor:
    return tensor(value.shape, (f32(function(float(x))) for x in value.data), "float32")


def sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def silu_scalar(x: float) -> float:
    return x * sigmoid_scalar(x)


def softplus_scalar(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def rms_norm_group(value: Tensor, groups: int, group_size: int, eps: float = 1e-6, weight: Tensor | None = None, zero_centered: bool = False) -> Tensor:
    if value.ndim != 2 or value.shape[1] != groups * group_size:
        raise FixtureError(f"group RMS input {value.shape} does not equal ({groups}, {group_size})")
    if weight is not None and weight.shape not in ((group_size,), (groups, group_size), (groups * group_size,)):
        raise FixtureError(f"group RMS weight shape {weight.shape} is not compatible")
    output: list[float] = []
    for row in range(value.shape[0]):
        for group in range(groups):
            start = row * groups * group_size + group * group_size
            source = value.data[start : start + group_size]
            inv = 1.0 / math.sqrt(sum(float(x) * float(x) for x in source) / group_size + eps)
            for j, x in enumerate(source):
                if weight is None:
                    multiplier = 1.0
                elif weight.shape == (group_size,):
                    multiplier = float(weight.data[j])
                elif weight.shape == (groups, group_size):
                    multiplier = float(weight.data[group * group_size + j])
                else:
                    multiplier = float(weight.data[group * group_size + j])
                if zero_centered:
                    multiplier += 1.0
                output.append(f32(float(x) * inv * multiplier))
    return tensor(value.shape, output, "float32")


def flatten_streams(value: Tensor, streams: int, width: int) -> Tensor:
    if value.shape != (value.shape[0], streams, width):
        raise FixtureError("flatten_streams expects a [tokens, streams, width] tensor")
    return tensor((value.shape[0], streams * width), value.data, value.dtype)


def hc_prepare(
    hyper_input: Tensor,
    down: Tensor,
    up: Tensor,
    norm_weight: Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Qwen4ExpTextGatedResidual read/mix path."""

    if hyper_input.shape[1] != hc_count * hidden_size:
        raise FixtureError(f"HC input {hyper_input.shape} has wrong width")
    normed = rms_norm_group(hyper_input, hc_count, hidden_size, eps, norm_weight, zero_centered=True)
    low = scale(matmul(normed, transpose2(down)), 1.0 / hc_count)
    low = map_values(low, silu_scalar)
    mix_logits = matmul(low, transpose2(up))
    mix = map_values(mix_logits, sigmoid_scalar).reshape(hyper_input.shape[0], hc_count, hidden_size)
    normed_streams = normed.reshape(hyper_input.shape[0], hc_count, hidden_size)
    mixed: list[float] = []
    for row in range(hyper_input.shape[0]):
        for j in range(hidden_size):
            total = 0.0
            for branch in range(hc_count):
                idx = (row * hc_count + branch) * hidden_size + j
                total += float(mix.data[idx]) * float(normed_streams.data[idx])
            mixed.append(f32(total / hc_count))
    return {
        "normed": normed,
        "lowrank_silu": low,
        "mix_gate": mix,
        "mixed": tensor((hyper_input.shape[0], hidden_size), mixed, "float32"),
    }


def hc_inject(
    prepared: dict[str, Tensor],
    hyper_input: Tensor,
    block_inject: Tensor,
    hc_count: int,
) -> dict[str, Tensor]:
    normed = prepared["normed"]
    weights = scale(matmul(normed, transpose2(block_inject)), 1.0 / hc_count)
    weights = scale(map_values(weights, sigmoid_scalar), 2.0)
    hidden = prepared["mixed"]
    output: list[float] = []
    for row in range(hyper_input.shape[0]):
        for branch in range(hc_count):
            gate = float(weights.data[row * hc_count + branch])
            for j in range(hidden.shape[1]):
                output.append(f32(float(hyper_input.data[row * hc_count * hidden.shape[1] + branch * hidden.shape[1] + j]) + gate * float(hidden.data[row * hidden.shape[1] + j])))
    return {
        "injection_weight": weights,
        "injected": tensor(hyper_input.shape, output, "float32"),
    }


def hc_final_mix(
    hyper_input: Tensor,
    norm_weight: Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float = 1e-6,
    down: Tensor | None = None,
    up: Tensor | None = None,
) -> dict[str, Tensor]:
    """Final GatedResidual read mixer: projections remain, write gate is absent."""
    if down is None or up is None:
        # Identity read gates are useful for a minimal caller, but the
        # generated fixtures always provide the learned down/up projections.
        normed = rms_norm_group(hyper_input, hc_count, hidden_size, eps, norm_weight, zero_centered=True)
        streams = normed.reshape(hyper_input.shape[0], hc_count, hidden_size)
        mixed = [
            f32(sum(float(streams.data[(row * hc_count + branch) * hidden_size + j]) for branch in range(hc_count)) / hc_count)
            for row in range(hyper_input.shape[0])
            for j in range(hidden_size)
        ]
        return {"normed": normed, "mixed": tensor((hyper_input.shape[0], hidden_size), mixed, "float32")}
    prepared = hc_prepare(hyper_input, down, up, norm_weight, hc_count, hidden_size, eps)
    return {
        "normed": prepared["normed"],
        "lowrank_silu": prepared["lowrank_silu"],
        "mix_gate": prepared["mix_gate"],
        "mixed": prepared["mixed"],
    }
 
 


def splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def signed64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & SIGNED64 else value


def shift_right_ignore_eos(token_ids: Sequence[int], shift: int, eos_token_id: int = EOS_TOKEN_ID) -> list[int]:
    """The exact Transformers `_shift_right_ignore_eos` operation."""

    if shift == 0:
        return [int(x) for x in token_ids]
    positions = list(range(len(token_ids)))
    eos_positions = [i if int(token_ids[i]) == eos_token_id else -1 for i in positions]
    previous_eos_inclusive: list[int] = []
    running = -1
    for value in eos_positions:
        running = max(running, value)
        previous_eos_inclusive.append(running)
    previous_eos = [-1] + previous_eos_inclusive[:-1]
    segment_start = [x + 1 for x in previous_eos]
    output: list[int] = []
    for pos in positions:
        source = max(pos - shift, 0)
        valid = pos - segment_start[pos] >= shift and pos - shift >= 0
        output.append(int(token_ids[source]) if valid else eos_token_id)
    return output


def ple_hash_history(tokens: Sequence[int], previous_context: Sequence[int], eos_token_id: int = EOS_TOKEN_ID) -> dict[str, object]:
    if len(previous_context) != 2:
        raise FixtureError("Qwen4 trigram history requires exactly two context tokens")
    history = [int(x) for x in previous_context] + [int(x) for x in tokens]
    shifted = [shift_right_ignore_eos(history, shift, eos_token_id) for shift in range(3)]
    rows: list[int] = []
    for token_index in range(len(tokens)):
        current: list[int] = []
        # Eight bigram heads, then eight trigram heads, matching the pinned
        # embedding's contiguous head layout.
        for ngram, start in ((2, 0), (3, 8)):
            mixed: int | None = None
            for position in range(ngram):
                product = signed64(signed64(shifted[position][token_index + 2]) * NG_SCALE[position])
                mixed = product if mixed is None else signed64(mixed ^ product)
            assert mixed is not None
            for head in range(start, start + 8):
                current.append(int((mixed % NG_HEAD_VOCAB_SIZES[head]) + NG_HEAD_OFFSETS[head]))
        rows.extend(current)
    ids = tensor((len(tokens), 16), rows, "int64")
    return {
        "token_history": tensor((len(history),), history, "int64"),
        "shifted_tokens": tensor((3, len(history)), (v for row in shifted for v in row), "int64"),
        "ple_row_ids": ids,
        "head_vocab_sizes": tensor((16,), NG_HEAD_VOCAB_SIZES, "int64"),
        "head_offsets": tensor((16,), NG_HEAD_OFFSETS, "int64"),
        "multipliers": tensor((3,), NG_SCALE, "int64"),
    }


def row_words_from_ids(ids: Tensor, width: int = 160) -> Tensor:
    """Generate deterministic synthetic BF16 rows without allocating a table."""

    if ids.shape[-1] != 16:
        raise FixtureError("PLE row ids must have sixteen n-gram heads")
    values: list[int] = []
    for token_index in range(ids.shape[0]):
        for head in range(16):
            row_id = int(ids.data[token_index * 16 + head])
            for element in range(width):
                # A deterministic finite [-1,1] value, then exact BF16 storage.
                state = (row_id * 0x9E3779B1 + (head + 1) * 0x85EBCA6B + element * 0xC2B2AE35) & 0xFFFFFFFF
                unit = ((state >> 8) & 0xFFFFFF) / float(1 << 24)
                values.append(bf16_word(unit * 2.0 - 1.0))
    return tensor((ids.shape[0], 16, width), values, "uint16")


def rope_half_split(value: Tensor, positions: Sequence[int], rotary_dim: int | None = None) -> Tensor:
    """Prefix-HalfSplit RoPE (`rotate_half`) for [tokens, heads, dim]."""

    if value.ndim != 3 or len(positions) != value.shape[0]:
        raise FixtureError(f"RoPE value {value.shape} and positions {len(positions)} disagree")
    dim = value.shape[2] if rotary_dim is None else int(rotary_dim)
    if dim <= 0 or dim % 2 or dim > value.shape[2]:
        raise FixtureError(f"invalid rotary dimension {dim} for {value.shape}")
    half = dim // 2
    out = list(float(x) for x in value.data)
    for t, position in enumerate(positions):
        for head in range(value.shape[1]):
            base = (t * value.shape[1] + head) * value.shape[2]
            for j in range(half):
                angle = float(position) / (10000000.0 ** (2.0 * j / dim))
                cosine, sine = math.cos(angle), math.sin(angle)
                first = float(value.data[base + j])
                second = float(value.data[base + j + half])
                out[base + j] = f32(first * cosine - second * sine)
                out[base + j + half] = f32(first * sine + second * cosine)
    return tensor(value.shape, out, "float32")


def l2_normalize_heads(value: Tensor, eps: float = 1e-6) -> Tensor:
    if value.ndim != 3:
        raise FixtureError("head normalization expects [tokens, heads, dim]")
    out: list[float] = []
    dim = value.shape[2]
    for start in range(0, len(value.data), dim):
        row = value.data[start : start + dim]
        inv = 1.0 / math.sqrt(sum(float(x) * float(x) for x in row) + eps)
        out.extend(f32(float(x) * inv) for x in row)
    return tensor(value.shape, out, "float32")

def l2_normalize_heads_bf16(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Mirror the pinned source l2norm while preserving its BF16 boundary."""

    if value.ndim != 3:
        raise FixtureError("BF16 head normalization expects [tokens, heads, dim]")
    out: list[float] = []
    dim = value.shape[2]
    eps_bf16 = bf16_value(bf16_word(eps))
    for start in range(0, len(value.data), dim):
        row = [bf16_value(bf16_word(float(x))) for x in value.data[start : start + dim]]
        sum_f32 = f32(sum(bf16_value(bf16_word(x * x)) for x in row))
        sum_bf16 = bf16_value(bf16_word(sum_f32))
        inv_bf16 = bf16_value(bf16_word(1.0 / math.sqrt(sum_bf16 + eps_bf16)))
        out.extend(bf16_value(bf16_word(x * inv_bf16)) for x in row)
    return tensor(value.shape, out, "float32")


def _qsa_select_one(
    raw_keys: Tensor,
    query: Tensor,
    query_index: int,
    visible_length: int,
    positions: Sequence[int],
    compress_ratio: int,
    token_budget: int,
    rope_dim: int,
) -> tuple[list[int], list[float]]:
    # raw_keys [tokens, index_kv_heads=1, dim] and query [tokens, index_heads, dim].
    if visible_length <= 0:
        return [], []
    visible = list(range(visible_length))
    complete = (len(visible) // compress_ratio) * compress_ratio
    block_indices = [visible[i : i + compress_ratio] for i in range(0, complete, compress_ratio)]
    scores: list[float] = []
    q_heads = query.shape[1]
    dim = query.shape[2]
    for block in block_indices:
        pooled = []
        for j in range(dim):
            pooled.append(sum(float(raw_keys.data[(token * raw_keys.shape[1]) * dim + j]) for token in block) / len(block))
        pooled_tensor = tensor((1, 1, dim), pooled, "float32")
        pooled_tensor = rms_norm_group(tensor((1, dim), pooled, "float32"), 1, dim, 1e-6).reshape(1, 1, dim)
        pooled_rotated = rope_half_split(pooled_tensor, [positions[block[0]]], rope_dim)
        score = 0.0
        qbase = (query_index * q_heads) * dim
        for qh in range(q_heads):
            qvec = query.data[qbase + qh * dim : qbase + (qh + 1) * dim]
            dot = sum(float(qvec[j]) * float(pooled_rotated.data[j]) for j in range(dim)) / math.sqrt(dim)
            score += max(dot, 0.0)
        scores.append(f32(score))
    block_topk = min(token_budget // compress_ratio, len(block_indices))
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:block_topk]
    selected: list[int] = []
    for block_index in order:
        selected.extend(block_indices[block_index])
    selected.extend(visible[complete:])
    return selected, scores


def qsa_indexer(
    index_queries: Tensor,
    raw_keys: Tensor,
    positions: Sequence[int],
    compress_ratio: int = 4,
    token_budget: int = 8,
    rope_dim: int = 4,
) -> dict[str, Tensor]:
    if index_queries.ndim != 3 or raw_keys.ndim != 3:
        raise FixtureError("QSA indexer expects query/raw-key rank three")
    if index_queries.shape[0] != raw_keys.shape[0] or index_queries.shape[2] != raw_keys.shape[2]:
        raise FixtureError("QSA query/key shape mismatch")
    if len(positions) != raw_keys.shape[0]:
        raise FixtureError("QSA positions do not cover keys")
    tokens, q_heads, dim = index_queries.shape
    capacity = token_budget + compress_ratio - 1
    selected = [-1] * (tokens * capacity)
    score_values: list[float] = []
    max_blocks = (tokens + compress_ratio - 1) // compress_ratio
    block_score_rows: list[float] = []
    for query_index in range(tokens):
        row_query = tensor((tokens, q_heads, dim), index_queries.data, "float32")
        indices, scores = _qsa_select_one(raw_keys, row_query, query_index, query_index + 1, positions, compress_ratio, token_budget, rope_dim)
        if len(set(scores)) != len(scores):
            raise FixtureError("QSA fixture scores are tied; tie acceptance is forbidden")
        score_values.extend(scores)
        block_score_rows.extend(scores)
        block_score_rows.extend([0.0] * (max_blocks - len(scores)))
        selected[query_index * capacity : query_index * capacity + len(indices)] = indices
    slot_mask = [False] * (tokens * capacity * tokens)
    token_mask = [False] * (tokens * tokens)
    for query_index in range(tokens):
        for slot in range(capacity):
            token = selected[query_index * capacity + slot]
            if token >= 0:
                slot_mask[(query_index * capacity + slot) * tokens + token] = True
                token_mask[query_index * tokens + token] = True
    return {
        "selected_indices": tensor((tokens, capacity), selected, "int32"),
        "selected_mask": tensor((tokens, capacity, tokens), slot_mask, "bool"),
        "selected_token_mask": tensor((tokens, tokens), token_mask, "bool"),
        "block_scores": tensor((tokens, max_blocks), block_score_rows, "float32"),
    }


def qsa_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    selected_indices: Tensor,
    gate: Tensor,
    output_projection: Tensor,
    positions: Sequence[int],
    rope_dim: int,
) -> dict[str, Tensor]:
    """Selected QSA attention, including the two-way KV repeat and sigmoid gate."""

    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise FixtureError("QSA attention expects [tokens, heads, dim] tensors")
    tokens, q_heads, dim = queries.shape
    kv_heads = keys.shape[1]
    if q_heads % kv_heads or values.shape != keys.shape:
        raise FixtureError("QSA attention does not have a valid GQA layout")
    q_rot = rope_half_split(queries, positions, rope_dim)
    k_rot = rope_half_split(keys, positions, rope_dim)
    capacity = selected_indices.shape[1]
    head_outputs: list[float] = []
    attn_weights: list[float] = []
    for t in range(tokens):
        chosen = [int(x) for x in selected_indices.data[t * capacity : (t + 1) * capacity] if int(x) >= 0]
        for qh in range(q_heads):
            kvh = qh // (q_heads // kv_heads)
            qbase = (t * q_heads + qh) * dim
            logits: list[float] = []
            for token in chosen:
                kbase = (token * kv_heads + kvh) * dim
                logits.append(sum(float(q_rot.data[qbase + j]) * float(k_rot.data[kbase + j]) for j in range(dim)) / math.sqrt(dim))
            if not logits:
                probs = []
            else:
                maximum = max(logits)
                exp_values = [math.exp(x - maximum) for x in logits]
                total = sum(exp_values)
                probs = [x / total for x in exp_values]
            attn_weights.extend(f32(x) for x in probs)
            for j in range(dim):
                result = sum(probs[p] * float(values.data[(token * kv_heads + kvh) * dim + j]) for p, token in enumerate(chosen))
                head_outputs.append(f32(result * sigmoid_scalar(float(gate.data[t * q_heads * dim + qh * dim + j]))))
    heads = tensor((tokens, q_heads * dim), head_outputs, "float32")
    projected = matmul(heads, transpose2(output_projection))
    return {"query_rope": q_rot, "key_rope": k_rot, "attention_weights": tensor((len(attn_weights),), attn_weights, "float32"), "head_output": heads, "output": projected}


def gdn_depthwise_conv(mixed_qkv: Tensor, kernel: Tensor, history: Tensor | None = None, activation: bool = True) -> dict[str, Tensor]:
    if mixed_qkv.ndim != 2 or kernel.ndim != 2 or kernel.shape[0] != mixed_qkv.shape[1]:
        raise FixtureError("GDN depthwise convolution shape mismatch")
    tokens, channels = mixed_qkv.shape
    kernel_size = kernel.shape[1]
    state = kernel_size - 1
    previous = [0.0] * (state * channels) if history is None else [float(x) for x in history.data]
    if len(previous) != state * channels:
        raise FixtureError("GDN history length mismatch")
    combined = previous + [float(x) for x in mixed_qkv.data]
    output: list[float] = []
    for t in range(tokens):
        for channel in range(channels):
            total = 0.0
            for j in range(kernel_size):
                total += combined[(t + j) * channels + channel] * float(kernel.data[channel * kernel_size + j])
            output.append(f32(silu_scalar(total) if activation else total))
    new_history = combined[-state * channels :] if state else []
    return {"output": tensor((tokens, channels), output, "float32"), "history": tensor((state, channels), new_history, "float32")}

def gdn_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    qk_l2norm_eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Pinned recurrent gated-delta rule with float32 state."""

    if query.shape != key.shape or query.ndim != 3 or value.ndim != 3:
        raise FixtureError("GDN q/k/v shapes are invalid")
    tokens, heads, kdim = query.shape
    if value.shape[0] != tokens or value.shape[1] != heads or g.shape != (tokens, heads) or beta.shape != (tokens, heads):
        raise FixtureError("GDN gate/value shapes are invalid")
    vdim = value.shape[2]
    q = l2_normalize_heads_bf16(query, qk_l2norm_eps)
    k = l2_normalize_heads_bf16(key, qk_l2norm_eps)
    query_scale = 1.0 / math.sqrt(kdim)
    q_scaled = tensor(q.shape, (f32(float(x) * query_scale) for x in q.data), "float32")
    state_data = [0.0] * (heads * kdim * vdim) if initial_state is None else [float(x) for x in initial_state.data]
    if len(state_data) != heads * kdim * vdim:
        raise FixtureError("GDN recurrent state shape is invalid")
    output = [0.0] * (tokens * heads * vdim)
    for t in range(tokens):
        for head in range(heads):
            g_factor = math.exp(float(g.data[t * heads + head]))
            base = head * kdim * vdim
            for i in range(kdim * vdim):
                state_data[base + i] = f32(state_data[base + i] * g_factor)
            qbase = (t * heads + head) * kdim
            vbase = (t * heads + head) * vdim
            kv_mem = []
            for j in range(vdim):
                kv_mem.append(f32(sum(state_data[base + i * vdim + j] * float(k.data[qbase + i]) for i in range(kdim))))
            # `beta` is already sigmoid(b), as in the pinned caller.
            delta = [f32((float(value.data[vbase + j]) - kv_mem[j]) * float(beta.data[t * heads + head])) for j in range(vdim)]
            for i in range(kdim):
                kval = float(k.data[qbase + i])
                for j in range(vdim):
                    state_data[base + i * vdim + j] = f32(state_data[base + i * vdim + j] + kval * delta[j])
            obase = (t * heads + head) * vdim
            for j in range(vdim):
                output[obase + j] = f32(
                    sum(state_data[base + i * vdim + j] * float(q_scaled.data[qbase + i]) for i in range(kdim))
                )
    return {
        "query_l2": q,
        "key_l2": k,
        "output": tensor((tokens, heads, vdim), output, "float32"),
        "final_state": tensor((heads, kdim, vdim), (f32(x) for x in state_data), "float32"),
    }


def gdn_expand_qk(query: Tensor, key: Tensor, value_heads: int) -> tuple[Tensor, Tensor]:
    if query.ndim != 3 or key.shape != query.shape or value_heads % query.shape[1] != 0:
        raise FixtureError("GDN expansion expects equal Q/K heads and a divisible V head count")
    repeat = value_heads // query.shape[1]
    values_q: list[float] = []
    values_k: list[float] = []
    for token in range(query.shape[0]):
        for head in range(query.shape[1]):
            qrow = query.data[(token * query.shape[1] + head) * query.shape[2] : (token * query.shape[1] + head + 1) * query.shape[2]]
            krow = key.data[(token * key.shape[1] + head) * key.shape[2] : (token * key.shape[1] + head + 1) * key.shape[2]]
            for _ in range(repeat):
                values_q.extend(qrow)
                values_k.extend(krow)
    return tensor((query.shape[0], value_heads, query.shape[2]), values_q, "float32"), tensor((query.shape[0], value_heads, query.shape[2]), values_k, "float32")


def moe_top10(
    hidden: Tensor,
    router_weight: Tensor,
    gate_up: Tensor,
    down: Tensor,
    shared_gate: Tensor,
    shared_gate_up: Tensor,
    shared_down: Tensor,
    top_k: int = 10,
) -> dict[str, Tensor]:
    """Qwen4 top-k normalized routed experts plus one sigmoid shared expert."""

    if hidden.ndim != 2 or router_weight.ndim != 2 or router_weight.shape[1] != hidden.shape[1]:
        raise FixtureError("MoE hidden/router shapes are invalid")
    tokens, hidden_size = hidden.shape
    experts = router_weight.shape[0]
    if top_k != 10 or experts < top_k:
        raise FixtureError("this fixture contract requires top_k=10 and at least ten experts")
    if gate_up.shape[0] != experts or gate_up.shape[2] != hidden_size or gate_up.shape[1] % 2:
        raise FixtureError("MoE gate/up shape is invalid")
    intermediate = gate_up.shape[1] // 2
    if down.shape != (experts, hidden_size, intermediate):
        raise FixtureError("MoE down shape is invalid")
    router_logits = matmul(hidden, transpose2(router_weight))
    selected: list[int] = []
    scores: list[float] = []
    routed_output: list[float] = []
    for token_index in range(tokens):
        logits = [float(x) for x in router_logits.data[token_index * experts : (token_index + 1) * experts]]
        probabilities = row_softmax(logits)
        order = sorted(range(experts), key=lambda index: (-probabilities[index], index))[:top_k]
        chosen = [probabilities[index] for index in order]
        total = sum(chosen)
        chosen = [x / total for x in chosen]
        selected.extend(order)
        scores.extend(f32(x) for x in chosen)
        result = [0.0] * hidden_size
        for slot, expert in enumerate(order):
            # gate/up are represented row-major by intermediate; the compact
            # fixture uses one scalar activation per intermediate below.
            activation_vec: list[float] = []
            for m in range(intermediate):
                gate_m = sum(float(hidden.data[token_index * hidden_size + j]) * float(gate_up.data[(expert * 2 * intermediate + m) * hidden_size + j]) for j in range(hidden_size))
                up_m = sum(float(hidden.data[token_index * hidden_size + j]) * float(gate_up.data[(expert * 2 * intermediate + intermediate + m) * hidden_size + j]) for j in range(hidden_size))
                activation_vec.append(silu_scalar(gate_m) * up_m)
            for j in range(hidden_size):
                result[j] += chosen[slot] * sum(activation_vec[m] * float(down.data[(expert * hidden_size + j) * intermediate + m]) for m in range(intermediate))
        routed_output.extend(f32(x) for x in result)
    shared_intermediate = shared_gate_up.shape[0] // 2
    if shared_gate_up.shape[0] != 2 * shared_intermediate or shared_down.shape != (hidden_size, shared_intermediate):
        raise FixtureError("shared expert gate/up/down shapes are invalid")
    shared_routed: list[float] = []
    shared_gate_values: list[float] = []
    for token_index in range(tokens):
        activations: list[float] = []
        for m in range(shared_intermediate):
            gate_m = sum(float(hidden.data[token_index * hidden_size + j]) * float(shared_gate_up.data[m * hidden_size + j]) for j in range(hidden_size))
            up_m = sum(float(hidden.data[token_index * hidden_size + j]) * float(shared_gate_up.data[(shared_intermediate + m) * hidden_size + j]) for j in range(hidden_size))
            activations.append(silu_scalar(gate_m) * up_m)
        shared_row = [sum(activations[m] * float(shared_down.data[j * shared_intermediate + m]) for m in range(shared_intermediate)) for j in range(hidden_size)]
        gate_value = sigmoid_scalar(sum(float(hidden.data[token_index * hidden_size + j]) * float(shared_gate.data[j]) for j in range(hidden_size)))
        shared_gate_values.append(f32(gate_value))
        shared_routed.extend(f32(gate_value * x) for x in shared_row)
    output = [f32(x + y) for x, y in zip(routed_output, shared_routed)]
    return {
        "router_logits": router_logits,
        "selected_experts": tensor((tokens, top_k), selected, "int32"),
        "routing_weights": tensor((tokens, top_k), scores, "float32"),
        "routed_output": tensor((tokens, hidden_size), routed_output, "float32"),
        "shared_gate": tensor((tokens, 1), shared_gate_values, "float32"),
        "shared_output": tensor((tokens, hidden_size), shared_routed, "float32"),
        "output": tensor((tokens, hidden_size), output, "float32"),
    }


def row_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exp_values = [math.exp(float(x) - maximum) for x in values]
    total = sum(exp_values)
    return [x / total for x in exp_values]


def dilated_depthwise_conv(value: Tensor, kernel: Tensor, history: Tensor | None = None, dilation: int = 3) -> dict[str, Tensor]:
    if value.ndim != 2 or kernel.ndim != 2 or kernel.shape[0] != value.shape[1]:
        raise FixtureError("dilated depthwise convolution shape mismatch")
    tokens, channels = value.shape
    kernel_size = kernel.shape[1]
    state_len = (kernel_size - 1) * dilation
    previous = [0.0] * (state_len * channels) if history is None else [float(x) for x in history.data]
    if len(previous) != state_len * channels:
        raise FixtureError("dilated convolution history length mismatch")
    combined = previous + [float(x) for x in value.data]
    output: list[float] = []
    for t in range(tokens):
        for channel in range(channels):
            total = 0.0
            for j in range(kernel_size):
                total += combined[(t + j * dilation) * channels + channel] * float(kernel.data[channel * kernel_size + j])
            output.append(f32(silu_scalar(total)))
    return {"output": tensor((tokens, channels), output, "float32"), "history": tensor((state_len, channels), combined[-state_len * channels :], "float32")}


def mtp_embedding_projection(
    token_ids: Sequence[int],
    embedding_words: Tensor,
    backbone_hidden: Tensor,
    fc_embedding: Tensor,
    fc_hidden: Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """The pinned vLLM residual_linear_shared MTP input path."""

    if embedding_words.ndim != 2 or embedding_words.shape[1] != hidden_size:
        raise FixtureError("MTP embedding shape is invalid")
    if backbone_hidden.shape != (len(token_ids), hc_count * hidden_size):
        raise FixtureError("MTP backbone hidden shape is invalid")
    rows: list[float] = []
    for token in token_ids:
        if token < 0 or token >= embedding_words.shape[0]:
            raise FixtureError("MTP token is outside the compact embedding")
        rows.extend(bf16_value(int(x)) for x in embedding_words.data[token * hidden_size : (token + 1) * hidden_size])
    embedding = tensor((len(token_ids), hidden_size), rows, "float32")
    embedding_norm = rms_norm_group(embedding, 1, hidden_size, eps)
    projected_embedding = matmul(embedding_norm, transpose2(fc_embedding))
    hidden_norm = rms_norm_group(backbone_hidden, 1, hc_count * hidden_size, eps)
    hidden_branches = hidden_norm.reshape(len(token_ids) * hc_count, hidden_size)
    projected_hidden = matmul(hidden_branches, transpose2(fc_hidden)).reshape(len(token_ids), hc_count, hidden_size)
    result: list[float] = []
    for token in range(len(token_ids)):
        for branch in range(hc_count):
            for j in range(hidden_size):
                result.append(f32(float(projected_embedding.data[token * hidden_size + j]) + float(projected_hidden.data[(token * hc_count + branch) * hidden_size + j])))
    return {
        "embedding": embedding,
        "embedding_norm": embedding_norm,
        "projected_embedding": projected_embedding,
        "hidden_norm": hidden_norm,
        "projected_hidden": projected_hidden.reshape(len(token_ids), hc_count * hidden_size),
        "residual_input": tensor((len(token_ids), hc_count * hidden_size), result, "float32"),
    }
