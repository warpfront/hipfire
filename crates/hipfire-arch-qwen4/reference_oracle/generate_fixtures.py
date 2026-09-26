#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic, compact Qwen4Exp layerwise reference fixtures.

Synthetic usage::

    python3 crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py \
        --out /tmp/qwen4-reference-fixtures --seed 3800

Upstream parity usage::

    python3 crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py \
        --upstream-reference --out .codeinsight+research/qwen4/oracle --seed 3800

The default path implements only small synthetic tensors (plus optional
selected BF16 PLE rows).  ``--source-bf16-slices`` accepts an NPZ with
``ple_row_ids`` (int64, ``[tokens,16]``) and ``ple_rows_bf16`` (uint16 BF16
words, ``[tokens,16,160]``); shape, dtype, duplicate IDs, and token/row
mismatches are hard errors.  ``--upstream-reference`` instead executes exact
SHA-verified Transformers/vLLM source snapshots through an explicit adapter,
requires only PyTorch, and range-reads bounded trained checkpoint slices;
neither path loads a full checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .equations import (
        EOS_TOKEN_ID,
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_PADDED_ROWS,
        NG_SCALE,
        NG_VALID_ROWS,
        add,
        bf16_to_f32,
        bf16_tensor,
        dilated_depthwise_conv,
        gdn_depthwise_conv,
        gdn_expand_qk,
        gdn_recurrent,
        hc_final_mix,
        hc_inject,
        hc_prepare,
        matmul,
        moe_top10,
        mtp_embedding_projection,
        ple_hash_history,
        qsa_attention,
        qsa_indexer,
        rms_norm_group,
        rng_normal,
        rng_uniform,
        rope_half_split,
        row_words_from_ids,
        scale,
        sigmoid_scalar,
        tensor,
        transpose2,
    )
    from .schema import FixtureError, Tensor, array_metadata, read_npz, sha256_file, tensor_bytes, write_json, write_npz
except ImportError:  # direct execution from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from equations import (  # type: ignore
        EOS_TOKEN_ID,
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_PADDED_ROWS,
        NG_SCALE,
        NG_VALID_ROWS,
        add,
        bf16_to_f32,
        bf16_tensor,
        dilated_depthwise_conv,
        gdn_depthwise_conv,
        gdn_expand_qk,
        gdn_recurrent,
        hc_final_mix,
        hc_inject,
        hc_prepare,
        matmul,
        moe_top10,
        mtp_embedding_projection,
        ple_hash_history,
        qsa_attention,
        qsa_indexer,
        rms_norm_group,
        rng_normal,
        rng_uniform,
        rope_half_split,
        row_words_from_ids,
        scale,
        sigmoid_scalar,
        tensor,
        transpose2,
    )
    from schema import FixtureError, Tensor, array_metadata, read_npz, sha256_file, tensor_bytes, write_json, write_npz  # type: ignore
try:
    from .upstream import generate_upstream, self_test
except ImportError:  # direct execution from this directory
    from upstream import generate_upstream, self_test  # type: ignore


UPSTREAM_REFERENCE_SCHEMA = "hipfire.qwen4.reference_oracle.v2"


HF_TRANSFORMERS_COMMIT = "93c8b7b485963a10800c91f55304db6be211c2bd"
VLLM_COMMIT = "e126687a9a828d513c01a07cd69f025f27d63280"
HF_CONFIG_COMMIT = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
HF_MODEL = "Qwen/Qwen3.8-Flash-Next"
HIDDEN = 8
HC_COUNT = 4
PLE_HEADS = 16
PLE_HEAD_WIDTH = 160
PLE_DIM = PLE_HEADS * PLE_HEAD_WIDTH
RMS_EPS = 1e-6
QSA_COMPRESS = 4
QSA_BUDGET = 8
QSA_INDEX_DIM = 4
QSA_INDEX_HEADS = 4
QSA_KV_HEADS = 1
QSA_ATTN_HEADS = 4
QSA_ATTN_KV_HEADS = 2
QSA_ATTN_DIM = 4
MOE_EXPERTS = 512
MOE_TOPK = 10
MOE_INTERMEDIATE = 8

# The tolerances are fixed from the storage/accumulation contract before a
# candidate is compared.  BF16 inputs are promoted to F32 for reductions and
# projections; F32 outputs use the tighter bound.  They are not fit to output.
TOLERANCES = {
    "exact_integer_or_bytes": {"atol": 0.0, "rtol": 0.0, "policy": "byte_exact"},
    "bf16_input_f32_accumulation": {"atol": 0.015625, "rtol": 0.03125, "policy": "bf16_unit_roundoff_2^-7"},
    "f32_accumulation": {"atol": 1.0e-5, "rtol": 1.0e-5, "policy": "declared_f32_reduction_bound"},
    "f32_state_recurrence": {"atol": 3.0e-5, "rtol": 3.0e-5, "policy": "declared_f32_state_bound"},
}


def _matrix(rng: random.Random, rows: int, cols: int, scale_value: float = 0.1) -> Tensor:
    return rng_normal(rng, (rows, cols), scale_value)


def _concat_rows(tensors: Sequence[Tensor], shape: Sequence[int]) -> Tensor:
    data: list[object] = []
    for value in tensors:
        data.extend(value.data)
    return tensor(shape, data, tensors[0].dtype if tensors else "float32")


def _slice_columns(value: Tensor, start: int, end: int) -> Tensor:
    if value.ndim != 2 or not 0 <= start <= end <= value.shape[1]:
        raise FixtureError(f"invalid column slice {start}:{end} for {value.shape}")
    rows, cols = value.shape
    return tensor((rows, end - start), (value.data[row * cols + col] for row in range(rows) for col in range(start, end)), value.dtype)


def _slice_rows(value: Tensor, start: int, end: int) -> Tensor:
    if value.ndim < 1 or not 0 <= start <= end <= value.shape[0]:
        raise FixtureError(f"invalid row slice {start}:{end} for {value.shape}")
    stride = 1
    for dim in value.shape[1:]:
        stride *= dim
    shape = (end - start,) + value.shape[1:]
    return tensor(shape, value.data[start * stride : end * stride], value.dtype)


def _join_first_axis(parts: Sequence[Tensor]) -> Tensor:
    if not parts:
        raise FixtureError("cannot join no tensors")
    shape = (sum(part.shape[0] for part in parts),) + parts[0].shape[1:]
    return _concat_rows(parts, shape)


def _assert_equal(a: Tensor, b: Tensor, name: str) -> None:
    if a.shape != b.shape or a.dtype != b.dtype or a.data != b.data:
        raise FixtureError(f"deterministic equivalence failed for {name}: {a.shape}/{b.shape}")


def _float_tensor_from_words(words: Tensor) -> Tensor:
    return bf16_to_f32(words)


def _validate_source(path: str | None, expected_ids: Tensor, expected_tokens: int) -> tuple[Tensor | None, dict[str, object]]:
    if path is None:
        return None, {"provided": False, "mode": "deterministic_synthetic_bf16_rows"}
    arrays = read_npz(path)
    required = {"ple_row_ids", "ple_rows_bf16"}
    missing = required - arrays.keys()
    if missing:
        raise FixtureError(f"source slice NPZ is missing required arrays: {sorted(missing)}")
    ids = arrays["ple_row_ids"]
    rows = arrays["ple_rows_bf16"]
    if ids.dtype not in ("int64", "int32"):
        raise FixtureError(f"ple_row_ids must be int64/int32, got {ids.dtype}")
    if ids.shape != (expected_tokens, PLE_HEADS):
        raise FixtureError(f"ple_row_ids shape {ids.shape} != {(expected_tokens, PLE_HEADS)}")
    if rows.dtype != "uint16":
        raise FixtureError(f"ple_rows_bf16 must be uint16 BF16 words, got {rows.dtype}")
    expected_shape = (expected_tokens, PLE_HEADS, PLE_HEAD_WIDTH)
    if rows.shape != expected_shape:
        raise FixtureError(f"ple_rows_bf16 shape {rows.shape} != {expected_shape}")
    if ids.data != expected_ids.data:
        raise FixtureError("source PLE row IDs do not match the pinned hash/history case")
    if len(set(int(x) for x in ids.data)) != len(ids.data):
        raise FixtureError("source PLE row IDs contain duplicates")
    source_digest = sha256_file(path)
    return rows, {
        "provided": True,
        "path": str(Path(path).resolve()),
        "sha256": source_digest,
        "schema": {"ple_row_ids": {"dtype": ids.dtype, "shape": list(ids.shape)}, "ple_rows_bf16": {"dtype": rows.dtype, "shape": list(rows.shape)}},
    }


def _generate_ple(seed: int, source_path: str | None) -> tuple[dict[str, Tensor], dict[str, object], dict[str, object]]:
    rng = random.Random(seed)
    tokens = [31, 47, 53, 59, 61, EOS_TOKEN_ID, 71, 73, 79, 83, EOS_TOKEN_ID, 89, 97]
    hashed = ple_hash_history(tokens, [EOS_TOKEN_ID, EOS_TOKEN_ID], EOS_TOKEN_ID)
    source_rows, source_meta = _validate_source(source_path, hashed["ple_row_ids"], len(tokens))
    if source_rows is None:
        # The row generator is keyed by exact global row IDs; no table is ever allocated.
        rows = row_words_from_ids(hashed["ple_row_ids"], PLE_HEAD_WIDTH)
    else:
        rows = source_rows
    embedding = _float_tensor_from_words(rows).reshape(len(tokens), PLE_DIM)
    arrays = {
        "tokens": tensor((len(tokens),), tokens, "int64"),
        "previous_context": tensor((2,), [EOS_TOKEN_ID, EOS_TOKEN_ID], "int64"),
        "token_history": hashed["token_history"],
        "shifted_tokens": hashed["shifted_tokens"],
        "multipliers": hashed["multipliers"],
        "head_vocab_sizes": hashed["head_vocab_sizes"],
        "head_offsets": hashed["head_offsets"],
        "ple_row_ids": hashed["ple_row_ids"],
        "ple_rows_bf16": rows,
        "ple_embedding_f32": embedding,
    }
    schema = {
        "case": "ple_hash_history",
        "equations": ["EOS-initialized 2-token history", "segment-local shift_right_ignore_eos", "signed-int64 multiply/XOR", "positive remainder then stored head offset"],
        "exact_arrays": ["tokens", "token_history", "shifted_tokens", "multipliers", "head_vocab_sizes", "head_offsets", "ple_row_ids", "ple_rows_bf16"],
        "valid_rows": NG_VALID_ROWS,
        "padded_rows": NG_PADDED_ROWS,
        "ple_embedding_shape": [len(tokens), PLE_DIM],
    }
    return arrays, schema, source_meta


def _generate_hc(seed: int) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    tokens = 6
    hyper = rng_normal(rng, (tokens, HC_COUNT * HIDDEN), 0.7)
    norm_weight = rng_uniform(rng, (HC_COUNT, HIDDEN), 0.08)
    lowrank = 3
    down = _matrix(rng, lowrank, HC_COUNT * HIDDEN, 0.12)
    up = _matrix(rng, HC_COUNT * HIDDEN, lowrank, 0.12)
    block_inject = _matrix(rng, HC_COUNT, HC_COUNT * HIDDEN, 0.12)
    final_down = _matrix(rng, lowrank, HC_COUNT * HIDDEN, 0.12)
    final_up = _matrix(rng, HC_COUNT * HIDDEN, lowrank, 0.12)
    prepared = hc_prepare(hyper, down, up, norm_weight, HC_COUNT, HIDDEN, RMS_EPS)
    injected = hc_inject(prepared, hyper, block_inject, HC_COUNT)
    final = hc_final_mix(hyper, norm_weight, HC_COUNT, HIDDEN, RMS_EPS, final_down, final_up)
    arrays = {
        "hyper_input": hyper,
        "hc_norm_weight_zero_centered": norm_weight,
        "input_mix_weight_down": down,
        "input_mix_weight_up": up,
        "block_inject_weight": block_inject,
        "final_input_mix_weight_down": final_down,
        "final_input_mix_weight_up": final_up,
        "normed": prepared["normed"],
        "lowrank_silu": prepared["lowrank_silu"],
        "mix_gate": prepared["mix_gate"],
        "mixed": prepared["mixed"],
        "injection_weight": injected["injection_weight"],
        "injected": injected["injected"],
        "final_normed": final["normed"],
        "final_lowrank_silu": final["lowrank_silu"],
        "final_mix_gate": final["mix_gate"],
        "final_mixed": final["mixed"],
    }
    return arrays, {
        "case": "hc_prepare_inject_final_mix",
        "equations": ["grouped RMS x (1+w)", "SiLU(down/n) then sigmoid(up)", "mean over four weighted branches", "2*sigmoid(block_inject(normed)/n)", "final mixer has no write projection"],
        "hc_count": HC_COUNT,
        "hidden_size": HIDDEN,
        "lowrank": lowrank,
    }


def _generate_ple_projection(seed: int, ple_arrays: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    tokens = ple_arrays["ple_embedding_f32"].shape[0]
    hidden = rng_normal(rng, (tokens, HC_COUNT * HIDDEN), 0.7)
    key_weight = _matrix(rng, HC_COUNT * HIDDEN, PLE_DIM, 0.025)
    value_weight = _matrix(rng, HIDDEN, PLE_DIM, 0.025)
    norm_key_weight = rng_uniform(rng, (HC_COUNT, HIDDEN), 0.05)
    norm_query_weight = rng_uniform(rng, (HC_COUNT, HIDDEN), 0.05)
    norm_conv_weight = rng_uniform(rng, (HC_COUNT, HIDDEN), 0.05)
    conv_kernel = _matrix(rng, HC_COUNT * HIDDEN, 4, 0.04)
    embeddings = ple_arrays["ple_embedding_f32"]
    key_raw = matmul(embeddings, transpose2(key_weight))  # [T, HC*H]
    key_normed = rms_norm_group(key_raw, HC_COUNT, HIDDEN, RMS_EPS, norm_key_weight, True).reshape(tokens, HC_COUNT, HIDDEN)
    value = matmul(embeddings, transpose2(value_weight))  # [T, H]
    query_normed = rms_norm_group(hidden, HC_COUNT, HIDDEN, RMS_EPS, norm_query_weight, True).reshape(tokens, HC_COUNT, HIDDEN)
    gate: list[float] = []
    for token in range(tokens):
        for branch in range(HC_COUNT):
            base = (token * HC_COUNT + branch) * HIDDEN
            dot = sum(float(key_normed.data[base + j]) * float(query_normed.data[base + j]) for j in range(HIDDEN)) / (HIDDEN ** 0.5)
            gate.append((1.0 if dot >= 0 else -1.0) * math.sqrt(max(abs(dot), 1e-6)))
    gated: list[float] = []
    for token in range(tokens):
        for branch in range(HC_COUNT):
            g = sigmoid_scalar(gate[token * HC_COUNT + branch])
            for j in range(HIDDEN):
                gated.append(g * float(value.data[token * HIDDEN + j]))
    gated_value = tensor((tokens, HC_COUNT * HIDDEN), gated, "float32")
    gated_normed = rms_norm_group(gated_value, HC_COUNT, HIDDEN, RMS_EPS, norm_conv_weight, True)
    conv = dilated_depthwise_conv(gated_normed, conv_kernel, dilation=3)
    output = add(gated_value, conv["output"])
    arrays = {
        "hidden_states": hidden,
        "ple_embedding_f32": embeddings,
        "key_proj": key_weight,
        "value_proj": value_weight,
        "norm_key_weight": norm_key_weight,
        "norm_query_weight": norm_query_weight,
        "norm_conv_weight": norm_conv_weight,
        "conv1d_weight": conv_kernel,
        "key_normed": key_normed.reshape(tokens, HC_COUNT * HIDDEN),
        "query_normed": query_normed.reshape(tokens, HC_COUNT * HIDDEN),
        "gate": tensor((tokens, HC_COUNT), gate, "float32"),
        "value": value,
        "gated_value": gated_value,
        "gated_value_normed": gated_normed,
        "conv_history": conv["history"],
        "conv_output": conv["output"],
        "output": output,
    }
    return arrays, {
        "case": "ple_projection_dilated_conv",
        "equations": ["key/value projection from 16x160 BF16 rows", "per-branch grouped RMS", "sign(dot)*sqrt(max(abs(dot),1e-6))", "sigmoid gate", "dilation=ngram_size=3 depthwise kernel=4 SiLU"],
        "ple_embed_dim": PLE_DIM,
        "short_conv_state_rows": 9,
        "layer_index": 1,
        "token_boundary": 4,
    }


def _generate_gdn(seed: int) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    tokens, key_heads, value_heads, kdim, vdim = 7, 16, 48, 4, 4
    query = rng_normal(rng, (tokens, key_heads, kdim), 0.6)
    key = rng_normal(rng, (tokens, key_heads, kdim), 0.6)
    value = rng_normal(rng, (tokens, value_heads, vdim), 0.6)
    q_expanded, k_expanded = gdn_expand_qk(query, key, value_heads)
    b_logits = rng_normal(rng, (tokens, value_heads), 0.4)
    beta = tensor(b_logits.shape, (sigmoid_scalar(float(x)) for x in b_logits.data), "float32")
    a_logits = rng_normal(rng, (tokens, value_heads), 0.4)
    dt_bias = rng_uniform(rng, (value_heads,), 0.15)
    a_log = rng_uniform(rng, (value_heads,), 0.4)
    g_values: list[float] = []
    for t in range(tokens):
        for head in range(value_heads):
            g_values.append(-math.exp(float(a_log.data[head])) * (math.log1p(math.exp(float(a_logits.data[t * value_heads + head]) + float(dt_bias.data[head])))))
    g = tensor((tokens, value_heads), g_values, "float32")
    conv_dim = key_heads * kdim * 2 + value_heads * vdim
    mixed_qkv = rng_normal(rng, (tokens, conv_dim), 0.08)
    conv_weight = _matrix(rng, conv_dim, 4, 0.025)
    initial_history = tensor((3, conv_dim), (0.0,) * (3 * conv_dim), "float32")
    conv_full = gdn_depthwise_conv(mixed_qkv, conv_weight, initial_history, activation=True)
    first = gdn_depthwise_conv(_slice_rows(mixed_qkv, 0, 3), conv_weight, initial_history, activation=True)
    second = gdn_depthwise_conv(_slice_rows(mixed_qkv, 3, tokens), conv_weight, first["history"], activation=True)
    conv_chunked = _join_first_axis([first["output"], second["output"]])
    _assert_equal(conv_full["output"], conv_chunked, "GDN convolution chunking")
    qkv = conv_full["output"]
    q_conv = _slice_columns(qkv, 0, key_heads * kdim)
    k_conv = _slice_columns(qkv, key_heads * kdim, 2 * key_heads * kdim)
    v_conv = _slice_columns(qkv, 2 * key_heads * kdim, conv_dim)
    q_conv = q_conv.reshape(tokens, key_heads, kdim)
    k_conv = k_conv.reshape(tokens, key_heads, kdim)
    v_conv = v_conv.reshape(tokens, value_heads, vdim)
    recurrent = gdn_recurrent(q_expanded, k_expanded, value, g, beta)
    first_rec = gdn_recurrent(_slice_rows(q_expanded, 0, 3), _slice_rows(k_expanded, 0, 3), _slice_rows(value, 0, 3), _slice_rows(g, 0, 3), _slice_rows(beta, 0, 3))
    second_rec = gdn_recurrent(_slice_rows(q_expanded, 3, tokens), _slice_rows(k_expanded, 3, tokens), _slice_rows(value, 3, tokens), _slice_rows(g, 3, tokens), _slice_rows(beta, 3, tokens), first_rec["final_state"])
    recurrent_chunked = _join_first_axis([first_rec["output"], second_rec["output"]])
    _assert_equal(recurrent["output"], recurrent_chunked, "GDN recurrent chunking")
    core_flat = recurrent["output"].reshape(tokens, value_heads * vdim)
    z = rng_normal(rng, (tokens, value_heads * vdim), 0.4)
    norm_weight = tensor((vdim,), (1.0,) * vdim, "float32")
    core_norm = rms_norm_group(core_flat, value_heads, vdim, RMS_EPS, norm_weight, False)
    output_gate = tensor(core_norm.shape, (float(x) * sigmoid_scalar(float(z.data[i])) for i, x in enumerate(core_norm.data)), "float32")
    out_proj = _matrix(rng, HIDDEN, value_heads * vdim, 0.04)
    output = matmul(output_gate, transpose2(out_proj))
    arrays = {
        "query_16x4": query,
        "key_16x4": key,
        "value_48x4": value,
        "query_expanded_48x4": q_expanded,
        "key_expanded_48x4": k_expanded,
        "mixed_qkv": mixed_qkv,
        "conv_weight": conv_weight,
        "initial_conv_history": initial_history,
        "conv_output": conv_full["output"],
        "conv_final_history": conv_full["history"],
        "query_after_conv": q_conv,
        "key_after_conv": k_conv,
        "value_after_conv": v_conv,
        "b_logits": b_logits,
        "beta_sigmoid": beta,
        "a_logits": a_logits,
        "dt_bias": dt_bias,
        "a_log": a_log,
        "g_decay": g,
        "query_l2": recurrent["query_l2"],
        "key_l2": recurrent["key_l2"],
        "core_attention_output": recurrent["output"],
        "final_recurrent_state": recurrent["final_state"],
        "z_output_gate": z,
        "core_norm": core_norm,
        "output_gate_sigmoid": output_gate,
        "out_proj": out_proj,
        "output": output,
        "chunked_conv_output": conv_chunked,
        "chunked_recurrent_output": recurrent_chunked,
    }
    return arrays, {
        "case": "gdn_recurrence_conv_head_expansion",
        "equations": ["depthwise causal conv + SiLU", "16 Q/K heads repeated 3x to 48 V heads", "L2-normalized Q/K", "float32 recurrent gated-delta state", "beta=sigmoid(b)", "g=-exp(A_log)*softplus(a+dt_bias)", "sigmoid output gate (not SiLU)"],
        "key_heads": key_heads,
        "value_heads": value_heads,
        "key_head_dim": kdim,
        "value_head_dim": vdim,
        "chunk_boundary": 3,
    }


def _generate_qsa(seed: int) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    tokens = 13
    positions = list(range(tokens))
    raw_q = rng_normal(rng, (tokens, QSA_INDEX_HEADS, QSA_INDEX_DIM), 0.7)
    raw_keys = rng_normal(rng, (tokens, QSA_KV_HEADS, QSA_INDEX_DIM), 0.7)
    index_q = tensor(raw_q.shape, (abs(float(x)) + 0.17 + 0.003 * (i % QSA_INDEX_DIM) for i, x in enumerate(raw_q.data)), "float32")
    raw_k = tensor(raw_keys.shape, (abs(float(x)) + 0.11 + 0.007 * (i % QSA_INDEX_DIM) for i, x in enumerate(raw_keys.data)), "float32")
    selected = qsa_indexer(index_q, raw_k, positions, QSA_COMPRESS, QSA_BUDGET, QSA_INDEX_DIM)
    # Recompute rows across chunk boundaries and one-token incremental calls;
    # this exercises the persistent raw-key cache contract.
    chunk_rows: list[int] = []
    for stop in (4, 9, tokens):
        chunk = qsa_indexer(_slice_rows(index_q, 0, stop), _slice_rows(raw_k, 0, stop), positions[:stop], QSA_COMPRESS, QSA_BUDGET, QSA_INDEX_DIM)
        begin = 0 if stop == 4 else (4 if stop == 9 else 9)
        chunk_rows.extend(chunk["selected_indices"].data[begin * chunk["selected_indices"].shape[1] : stop * chunk["selected_indices"].shape[1]])
    incremental_rows: list[int] = []
    for stop in range(1, tokens + 1):
        one = qsa_indexer(_slice_rows(index_q, 0, stop), _slice_rows(raw_k, 0, stop), positions[:stop], QSA_COMPRESS, QSA_BUDGET, QSA_INDEX_DIM)
        capacity = one["selected_indices"].shape[1]
        incremental_rows.extend(one["selected_indices"].data[(stop - 1) * capacity : stop * capacity])
    chunked_indices = tensor(selected["selected_indices"].shape, chunk_rows, "int32")
    incremental_indices = tensor(selected["selected_indices"].shape, incremental_rows, "int32")
    _assert_equal(selected["selected_indices"], chunked_indices, "QSA chunked selection")
    _assert_equal(selected["selected_indices"], incremental_indices, "QSA incremental selection")
    q = rng_normal(rng, (tokens, QSA_ATTN_HEADS, QSA_ATTN_DIM), 0.6)
    k = rng_normal(rng, (tokens, QSA_ATTN_KV_HEADS, QSA_ATTN_DIM), 0.6)
    v = rng_normal(rng, (tokens, QSA_ATTN_KV_HEADS, QSA_ATTN_DIM), 0.6)
    gate = rng_normal(rng, (tokens, QSA_ATTN_HEADS * QSA_ATTN_DIM), 0.4)
    output_proj = _matrix(rng, HIDDEN, QSA_ATTN_HEADS * QSA_ATTN_DIM, 0.04)
    attention = qsa_attention(q, k, v, selected["selected_indices"], gate, output_proj, positions, QSA_ATTN_DIM)
    rope_input = rng_normal(rng, (5, 2, 8), 0.5)
    rope_output = rope_half_split(rope_input, [0, 1, 4, 7, 11], 8)
    arrays = {
        "positions": tensor((tokens,), positions, "int64"),
        "index_queries": index_q,
        "raw_index_keys": raw_k,
        "selected_indices": selected["selected_indices"],
        "selected_indices_chunked": chunked_indices,
        "selected_indices_incremental": incremental_indices,
        "selected_mask": selected["selected_mask"],
        "selected_token_mask": selected["selected_token_mask"],
        "block_scores": selected["block_scores"],
        "attention_queries": q,
        "attention_keys": k,
        "attention_values": v,
        "attention_gate": gate,
        "attention_output_projection": output_proj,
        "query_rope": attention["query_rope"],
        "key_rope": attention["key_rope"],
        "attention_weights": attention["attention_weights"],
        "attention_head_output": attention["head_output"],
        "attention_output": attention["output"],
        "rope_input": rope_input,
        "rope_output": rope_output,
    }
    return arrays, {
        "case": "qsa_pool_selection_mask_tail_rope",
        "equations": ["causal visible tokens grouped into blocks of four", "float32 block mean then RMS norm and Prefix-HalfSplit RoPE at block start", "sum(ReLU(q dot k / sqrt(128))) in source; compact dim is declared", "top-k blocks then every incomplete tail token", "-1 selection slots scatter to dropped sentinel"],
        "non_tied_scores_required": True,
        "compress_ratio": QSA_COMPRESS,
        "token_budget": QSA_BUDGET,
        "selected_capacity": QSA_BUDGET + QSA_COMPRESS - 1,
        "index_heads": QSA_INDEX_HEADS,
        "index_kv_heads": QSA_KV_HEADS,
        "index_head_dim": QSA_INDEX_DIM,
        "attention_heads": QSA_ATTN_HEADS,
        "attention_kv_heads": QSA_ATTN_KV_HEADS,
        "attention_head_dim": QSA_ATTN_DIM,
        "tail_positions": [4, 8, 9, 12],
        "chunk_boundaries": [4, 9, 13],
        "pool_boundaries": [4, 8, 12],
        "layer_index": 3,
        "token_boundary": 4,
    }


def _generate_moe(seed: int) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    tokens = 5
    hidden = rng_normal(rng, (tokens, HIDDEN), 0.6)
    router = _matrix(rng, MOE_EXPERTS, HIDDEN, 0.07)
    gate_up = _matrix(rng, MOE_EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN, 0.045).reshape(MOE_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN)
    down = _matrix(rng, MOE_EXPERTS * HIDDEN, MOE_INTERMEDIATE, 0.045).reshape(MOE_EXPERTS, HIDDEN, MOE_INTERMEDIATE)
    shared_gate = rng_uniform(rng, (HIDDEN,), 0.08)
    shared_gate_up = _matrix(rng, 2 * MOE_INTERMEDIATE, HIDDEN, 0.045)
    shared_down = _matrix(rng, HIDDEN, MOE_INTERMEDIATE, 0.045)
    result = moe_top10(hidden, router, gate_up, down, shared_gate, shared_gate_up, shared_down, MOE_TOPK)
    arrays = {
        "hidden": hidden,
        "router_weight": router,
        "gate_up_weight": gate_up,
        "down_weight": down,
        "shared_gate_weight": shared_gate,
        "shared_gate_up_weight": shared_gate_up,
        "shared_down_weight": shared_down,
        "router_logits": result["router_logits"],
        "selected_experts": result["selected_experts"],
        "routing_weights": result["routing_weights"],
        "routed_output": result["routed_output"],
        "shared_gate": result["shared_gate"],
        "shared_output": result["shared_output"],
        "output": result["output"],
    }
    return arrays, {
        "case": "moe_top10_normalized_shared",
        "equations": ["float32 softmax over all 512 experts", "top-10 then selected-probability renormalization", "SiLU(gate) * up and one weighted routed sum", "sigmoid(shared_gate) * shared expert added once"],
        "num_experts": MOE_EXPERTS,
        "top_k": MOE_TOPK,
        "intermediate_size": MOE_INTERMEDIATE,
        "norm_topk_prob": True,
    }


def _generate_mtp(seed: int, moe_seed: int) -> tuple[dict[str, Tensor], dict[str, object]]:
    rng = random.Random(seed)
    vocab = 32
    tokens = [1, 5, 7, 9]
    embedding_words = bf16_tensor((rng_uniform(rng, (vocab, HIDDEN), 0.5).data), (vocab, HIDDEN))
    backbone = rng_normal(rng, (len(tokens), HC_COUNT * HIDDEN), 0.6)
    fc_embedding = _matrix(rng, HIDDEN, HIDDEN, 0.08)
    fc_hidden = _matrix(rng, HIDDEN, HIDDEN, 0.08)
    projected = mtp_embedding_projection(tokens, embedding_words, backbone, fc_embedding, fc_hidden, HC_COUNT, HIDDEN, RMS_EPS)
    # One full-attention MTP layer.  The explicit indexer result is selected at
    # step 0 and the same target-aligned rows are marked for later reuse.
    raw_q = rng_normal(rng, (len(tokens), QSA_INDEX_HEADS, QSA_INDEX_DIM), 0.6)
    raw_keys = rng_normal(rng, (len(tokens), QSA_KV_HEADS, QSA_INDEX_DIM), 0.6)
    index_queries = tensor(raw_q.shape, (abs(float(x)) + 0.19 + 0.005 * (i % QSA_INDEX_DIM) for i, x in enumerate(raw_q.data)), "float32")
    raw_keys = tensor(raw_keys.shape, (abs(float(x)) + 0.13 + 0.009 * (i % QSA_INDEX_DIM) for i, x in enumerate(raw_keys.data)), "float32")
    mtp_index = qsa_indexer(index_queries, raw_keys, list(range(len(tokens))), QSA_COMPRESS, QSA_BUDGET, QSA_INDEX_DIM)
    later_reused = tensor(mtp_index["selected_indices"].shape, mtp_index["selected_indices"].data, "int32")
    q = rng_normal(rng, (len(tokens), QSA_ATTN_HEADS, QSA_ATTN_DIM), 0.6)
    k = rng_normal(rng, (len(tokens), QSA_ATTN_KV_HEADS, QSA_ATTN_DIM), 0.6)
    v = rng_normal(rng, (len(tokens), QSA_ATTN_KV_HEADS, QSA_ATTN_DIM), 0.6)
    attn_gate = rng_normal(rng, (len(tokens), QSA_ATTN_HEADS * QSA_ATTN_DIM), 0.3)
    attn_proj = _matrix(rng, HIDDEN, QSA_ATTN_HEADS * QSA_ATTN_DIM, 0.04)
    attention = qsa_attention(q, k, v, later_reused, attn_gate, attn_proj, list(range(len(tokens))), QSA_ATTN_DIM)
    # Attention and MoE both use the HC stream grammar: prepare -> operation -> inject.
    norm_weight = rng_uniform(rng, (HC_COUNT, HIDDEN), 0.05)
    down = _matrix(rng, 3, HC_COUNT * HIDDEN, 0.08)
    up = _matrix(rng, HC_COUNT * HIDDEN, 3, 0.08)
    inject_weight = _matrix(rng, HC_COUNT, HC_COUNT * HIDDEN, 0.08)
    attn_prepared = hc_prepare(projected["residual_input"], down, up, norm_weight, HC_COUNT, HIDDEN, RMS_EPS)
    # The QSA output is the operation result injected into the prepared HC streams.
    attn_for_inject = dict(attn_prepared)
    attn_for_inject["mixed"] = attention["output"]
    attn_injected = hc_inject(attn_for_inject, projected["residual_input"], inject_weight, HC_COUNT)
    # The MoE consumes the read-mixed stream after the QSA injection.
    mlp_prepared = hc_prepare(attn_injected["injected"], down, up, norm_weight, HC_COUNT, HIDDEN, RMS_EPS)
    moe_rng = random.Random(moe_seed)
    moe_hidden = mlp_prepared["mixed"]
    router = _matrix(moe_rng, MOE_EXPERTS, HIDDEN, 0.07)
    gate_up = _matrix(moe_rng, MOE_EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN, 0.045).reshape(MOE_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN)
    down_moe = _matrix(moe_rng, MOE_EXPERTS * HIDDEN, MOE_INTERMEDIATE, 0.045).reshape(MOE_EXPERTS, HIDDEN, MOE_INTERMEDIATE)
    shared_gate = rng_uniform(moe_rng, (HIDDEN,), 0.08)
    shared_gate_up = _matrix(moe_rng, 2 * MOE_INTERMEDIATE, HIDDEN, 0.045)
    shared_down = _matrix(moe_rng, HIDDEN, MOE_INTERMEDIATE, 0.045)
    moe = moe_top10(moe_hidden, router, gate_up, down_moe, shared_gate, shared_gate_up, shared_down, MOE_TOPK)
    mlp_for_inject = dict(mlp_prepared)
    mlp_for_inject["mixed"] = moe["output"]
    mlp_injected = hc_inject(mlp_for_inject, attn_injected["injected"], inject_weight, HC_COUNT)
    final_down = _matrix(rng, 3, HC_COUNT * HIDDEN, 0.08)
    final_up = _matrix(rng, HC_COUNT * HIDDEN, 3, 0.08)
    final = hc_final_mix(mlp_injected["injected"], norm_weight, HC_COUNT, HIDDEN, RMS_EPS, final_down, final_up)
    lm_head = _matrix(rng, vocab, HIDDEN, 0.06)
    logits = matmul(final["mixed"], transpose2(lm_head))
    arrays = {
        "token_ids": tensor((len(tokens),), tokens, "int64"),
        "token_embedding_bf16": embedding_words,
        "backbone_hidden": backbone,
        "fc_embedding": fc_embedding,
        "fc_hidden": fc_hidden,
        "embedding": projected["embedding"],
        "embedding_norm": projected["embedding_norm"],
        "projected_embedding": projected["projected_embedding"],
        "hidden_norm": projected["hidden_norm"],
        "projected_hidden": projected["projected_hidden"],
        "residual_input": projected["residual_input"],
        "index_queries": index_queries,
        "raw_index_keys": raw_keys,
        "step0_selected_indices": mtp_index["selected_indices"],
        "step0_selected_token_mask": mtp_index["selected_token_mask"],
        "later_reused_indices": later_reused,
        "attention_queries": q,
        "attention_keys": k,
        "attention_values": v,
        "attention_gate": attn_gate,
        "attention_projection": attn_proj,
        "attention_output": attention["output"],
        "hc_norm_weight": norm_weight,
        "hc_down": down,
        "hc_up": up,
        "hc_inject_weight": inject_weight,
        "final_hc_down": final_down,
        "final_hc_up": final_up,
        "attn_hc_normed": attn_prepared["normed"],
        "attn_hc_mixed": attn_prepared["mixed"],
        "attn_hc_injected": attn_injected["injected"],
        "mtp_moe_hidden": moe_hidden,
        "mtp_moe_router_logits": moe["router_logits"],
        "mtp_moe_selected_experts": moe["selected_experts"],
        "mtp_moe_routing_weights": moe["routing_weights"],
        "final_hc_lowrank_silu": final["lowrank_silu"],
        "final_hc_mix_gate": final["mix_gate"],
        "mtp_moe_output": moe["output"],
        "mlp_hc_normed": mlp_prepared["normed"],
        "mlp_hc_mixed": mlp_prepared["mixed"],
        "mlp_hc_injected": mlp_injected["injected"],
        "final_hc_normed": final["normed"],
        "sample_hidden": final["mixed"],
        "lm_head": lm_head,
        "logits": logits,
    }
    return arrays, {
        "case": "native_mtp_embedding_qsa_moe_hc",
        "equations": ["pre-FC RMSNorm(H) and RMSNorm(4H)", "fc_embedding + shared fc_hidden per HC branch", "one full-attention QSA/MoE/HC layer", "step0 selects QSA indices; later steps reuse target-aligned rows", "final HC mixer emits H and no extra RMS", "LM head is a distinct shared source; MTP has no PLE"],
        "hc_count": HC_COUNT,
        "hidden_size": HIDDEN,
        "num_mtp_layers": 1,
        "qsa_index_share_for_mtp_iteration": True,
        "embedding_source": "token_embedding.weight",
        "lm_head_source": "lm_head.weight",
        "mtp_use_dedicated_embeddings": False,
    }


def _build_manifest(out: Path, seed: int, fixtures: list[dict[str, object]], source_meta: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "hipfire.qwen4.reference_oracle.v2",
        "model": {
            "name": HF_MODEL,
            "architecture": "Qwen4ExpForConditionalGeneration",
            "config_commit": HF_CONFIG_COMMIT,
            "text_model_type": "qwen4_exp_text",
            "layer_layout": {"layers": 48, "gdn_layers": 36, "qsa_layers": 12},
            "real_dimensions": {"hidden_size": 2560, "gdn_key_heads": 16, "gdn_value_heads": 48, "qsa_heads": 24, "qsa_kv_heads": 2, "qsa_head_dim": 256, "qsa_index_heads": 4, "qsa_index_head_dim": 128, "qsa_budget": 2048, "qsa_compress_ratio": 4, "experts": 512, "top_k": 10, "hc_count": 4, "hc_lowrank": 320, "ple_heads": 16, "ple_row_width": 160},
            "fixture_dimensions_are_compact": True,
        },
        "reference_sources": [
            {"project": "huggingface/transformers", "commit": HF_TRANSFORMERS_COMMIT, "path": "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", "url": f"https://github.com/huggingface/transformers/blob/{HF_TRANSFORMERS_COMMIT}/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"},
            {"project": "vllm-project/vllm", "commit": VLLM_COMMIT, "path": "vllm/models/qwen4_exp/amd/mtp.py", "url": f"https://github.com/vllm-project/vllm/blob/{VLLM_COMMIT}/vllm/models/qwen4_exp/amd/mtp.py"},
        ],
        "generator": {"seed": seed, "python": f"{sys.version_info.major}.{sys.version_info.minor}", "dependency_policy": "stdlib_only", "deterministic_npz": True, "no_full_model_residency": True, "source_slice": source_meta},
        "equations": {
            "bf16_storage": "uint16 little-endian BF16 words; F32 promotion before reductions",
            "f32": "IEEE-754 binary32 output records",
            "ple": {"multipliers": list(NG_SCALE), "vocab_sizes": list(NG_HEAD_VOCAB_SIZES), "offsets": list(NG_HEAD_OFFSETS), "valid_rows": NG_VALID_ROWS, "padded_rows": NG_PADDED_ROWS, "eos_token_id": EOS_TOKEN_ID},
            "qsa_ties": "fixture construction rejects tied block scores; no arbitrary tie acceptance",
            "tolerances": TOLERANCES,
        },
        "fixtures": fixtures,
    }


def generate(out: str | Path, seed: int = 3800, source_path: str | None = None) -> dict[str, object]:
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    # Fixed independent streams make adding a case unable to perturb others.
    ple_arrays, ple_schema, source_meta = _generate_ple(seed + 11, source_path)
    hc_arrays, hc_schema = _generate_hc(seed + 23)
    ple_proj_arrays, ple_proj_schema = _generate_ple_projection(seed + 37, ple_arrays)
    gdn_arrays, gdn_schema = _generate_gdn(seed + 53)
    qsa_arrays, qsa_schema = _generate_qsa(seed + 71)
    moe_arrays, moe_schema = _generate_moe(seed + 89)
    mtp_arrays, mtp_schema = _generate_mtp(seed + 107, seed + 89)
    cases = [
        ("ple_hash_history", ple_arrays, ple_schema, seed + 11),
        ("hc_prepare_inject_final_mix", hc_arrays, hc_schema, seed + 23),
        ("ple_projection_dilated_conv", ple_proj_arrays, ple_proj_schema, seed + 37),
        ("gdn_recurrence_conv_head_expansion", gdn_arrays, gdn_schema, seed + 53),
        ("qsa_pool_selection_mask_tail_rope", qsa_arrays, qsa_schema, seed + 71),
        ("moe_top10_normalized_shared", moe_arrays, moe_schema, seed + 89),
        ("native_mtp_embedding_qsa_moe_hc", mtp_arrays, mtp_schema, seed + 107),
    ]
    fixtures: list[dict[str, object]] = []
    for name, arrays, schema, case_seed in cases:
        path = destination / f"{name}.npz"
        digest = write_npz(path, arrays)
        fixtures.append({"name": name, "path": path.name, "sha256": digest, "seed": case_seed, "schema": schema, "arrays": array_metadata(arrays)})
    manifest = _build_manifest(destination, seed, fixtures, source_meta)
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    # The digest is informational and deliberately not written back into the
    # file, avoiding a self-referential manifest and preserving determinism.
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="output directory for NPZ fixtures and manifest.json")
    parser.add_argument("--seed", type=int, default=3800, help="deterministic root seed (default: 3800)")
    parser.add_argument("--source-bf16-slices", "--source-slices", dest="source_path", help="optional selected-row NPZ; requires int row IDs and uint16 BF16 rows")
    parser.add_argument("--upstream-reference", action="store_true", help="execute exact SHA-verified Transformers/vLLM source snapshots with torch and range-read checkpoint slices")
    parser.add_argument("--cache-dir", help="bounded upstream cache (under ~/.hipfire/datasets or .codeinsight+research)")
    parser.add_argument("--self-test", action="store_true", help="validate synthetic preservation and rejection guards")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test and args.upstream_reference:
        print("qwen4 reference fixture generation failed: --self-test and --upstream-reference are exclusive", file=sys.stderr)
        return 2
    if not args.self_test and not args.out:
        print("qwen4 reference fixture generation failed: --out is required unless --self-test is used", file=sys.stderr)
        return 2
    try:
        if args.self_test:
            self_test(args.cache_dir)
            print("qwen4 reference fixture self-test passed")
            return 0
        if args.upstream_reference:
            manifest = generate_upstream(args.out, args.seed, args.cache_dir)
        else:
            if args.cache_dir:
                raise FixtureError("--cache-dir is only valid with --upstream-reference or --self-test")
            manifest = generate(args.out, args.seed, args.source_path)
    except (FixtureError, OSError, ValueError) as exc:
        print(f"qwen4 reference fixture generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(manifest['fixtures'])} fixtures to {Path(args.out)}")
    print(f"manifest: {Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
