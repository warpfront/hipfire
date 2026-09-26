# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed Qwen4 teacher evaluation and Astrea-compatible comparison.

The source path in this module is deliberately separate from the compact
operator fixtures.  It opens the pinned Hugging Face index/config, fetches
only the embedding/PLE rows needed by the corpus, then processes one decoder
layer at a time.  Router-selected expert slices are fetched on demand and the
LM head is consumed in bounded row chunks; no full checkpoint or full output
head is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
try:
    from . import upstream
    from .equations import (
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_SCALE,
    )
    from .schema import FixtureError
except ImportError:  # direct execution from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import upstream  # type: ignore
    from equations import (  # type: ignore
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_SCALE,
    )
    from schema import FixtureError  # type: ignore


QUALITY_SCHEMA = "hipfire.qwen4.quality.v1"
CANONICAL_TOKEN_COUNT = 17
CANONICAL_TOKEN_SHA256 = "e53de8c7b501eaaea637648feb6f569dd17cd564c2f669b2924ccdf1b7e52e2f"
CANONICAL_TOKEN_METADATA = "benchmarks/prompts/qwen4-teacher-forced.tokens.json"
TOP_K = 32
PLE_SHARD_ROWS = 2_500_012
PLE_VALID_ROWS = 320_001_446
PLE_SHARDS = 128
MAX_LM_HEAD_ROWS = 25  # 25 * 2560 * BF16 = 128 KiB




def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FixtureError(f"nonfinite {label}")
    return value


def _load_corpus(path: str | Path) -> tuple[list[int], dict[str, object]]:
    """Load the single canonical u32le payload through its metadata sidecar."""
    requested = Path(path).expanduser()
    metadata_path = requested if requested.suffix == ".json" else requested.with_suffix(".json")
    if not metadata_path.is_file():
        raise FixtureError(f"teacher corpus metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"invalid teacher corpus metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "hipfire.qwen4.teacher_forced_corpus.v1":
        raise FixtureError("teacher corpus metadata schema is not canonical")
    if metadata.get("format") != "u32le" or metadata.get("count") != CANONICAL_TOKEN_COUNT:
        raise FixtureError("teacher corpus metadata must declare 17 u32le tokens")
    declared = metadata.get("sha256")
    if declared != CANONICAL_TOKEN_SHA256:
        raise FixtureError(f"teacher corpus metadata digest mismatch: {declared!r}")
    relative = metadata.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise FixtureError("teacher corpus metadata path must be a relative u32le filename")
    payload = (metadata_path.parent / relative).resolve()
    if not payload.is_file() or payload.parent != metadata_path.parent.resolve():
        raise FixtureError(f"teacher corpus payload is missing or escapes metadata directory: {payload}")
    raw = payload.read_bytes()
    if len(raw) % 4 or len(raw) // 4 != CANONICAL_TOKEN_COUNT:
        raise FixtureError("teacher corpus payload must contain exactly 17 little-endian u32 values")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != declared:
        raise FixtureError(f"teacher corpus payload digest mismatch: expected {declared}, got {digest}")
    tokens = list(struct.unpack("<" + "I" * CANONICAL_TOKEN_COUNT, raw))
    return tokens, {
        "metadata_path": str(metadata_path),
        "payload_path": str(payload),
        "payload_sha256": digest,
        "count": len(tokens),
        "tokens": tokens,
    }


def _tensor_bytes(value: Any) -> bytes:
    value = value.detach().contiguous().cpu()
    if str(value.dtype) == "torch.bfloat16":
        import torch

        return value.view(torch.uint8).numpy().tobytes()
    return value.numpy().tobytes()


class _StreamWeights:
    """Torch conversion over the bounded `PinnedCheckpoint` row API."""

    def __init__(self, checkpoint: upstream.PinnedCheckpoint, torch: Any):
        self.checkpoint = checkpoint
        self.torch = torch
        self.current_bytes = 0
        self.peak_bytes = 0

    def begin_layer(self) -> None:
        self.current_bytes = 0

    def _track(self, value: Any) -> Any:
        self.current_bytes += int(value.numel() * value.element_size())
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        if not bool(self.torch.isfinite(value.float()).all()):
            raise FixtureError("nonfinite streamed checkpoint tensor")
        return value

    def _record(self, record: Mapping[str, object]) -> Any:
        torch = self.torch
        values = record.get("values")
        if values is None or not hasattr(values, "data"):
            raise FixtureError("invalid streamed tensor record")
        shape = tuple(int(item) for item in record["slice_shape"])  # type: ignore[index]
        data = list(values.data)
        dtype = str(record["dtype"])
        if dtype == "BF16":
            output = torch.tensor(data, dtype=torch.uint16).view(torch.bfloat16).reshape(shape)
        elif dtype == "F32":
            output = torch.tensor(data, dtype=torch.float32).reshape(shape)
        elif dtype == "I64":
            output = torch.tensor(data, dtype=torch.int64).reshape(shape)
        elif dtype == "I32":
            output = torch.tensor(data, dtype=torch.int32).reshape(shape)
        else:
            raise FixtureError(f"unsupported streamed dtype {dtype}")
        return self._track(output)

    def rows(self, name: str, row_start: int, row_count: int) -> Any:
        return self._record(self.checkpoint.tensor_rows(name, row_start, row_count))

    def expert_rows(self, name: str, expert: int, row_start: int, row_count: int) -> Any:
        return self._record(self.checkpoint.tensor_expert_rows(name, expert, row_start, row_count))

    def full(self, name: str) -> Any:
        descriptor = upstream._tensor_descriptor(self.checkpoint.client, self.checkpoint.weight_map, name)
        shape = tuple(int(item) for item in descriptor["shape"])
        rows = shape[0]
        row_items = math.prod(shape[1:]) if len(shape) > 1 else 1
        itemsize, _ = upstream._dtype_info(str(descriptor["dtype"]))
        if not itemsize:
            raise FixtureError(f"unsupported dtype for {name}")
        rows_per_read = max(1, upstream.MAX_TENSOR_SLICE_BYTES // (row_items * itemsize))
        pieces = [
            self.rows(name, start, min(rows_per_read, rows - start))
            for start in range(0, rows, rows_per_read)
        ]
        return self.torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]


def _load_hc_weights(store: _StreamWeights, prefix: str, *, combine: bool) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "hc_norm": store.full(f"{prefix}.hc_norm.weight"),
        "down": store.full(f"{prefix}.input_mix_weight_down.weight"),
        "up": store.full(f"{prefix}.input_mix_weight_up.weight"),
    }
    if combine:
        weights["block"] = store.full(f"{prefix}.block_inject_weight.weight")
    return weights


def _stream_source(tokens: Sequence[int], cache_dir: str | Path | None) -> dict[str, object]:
    torch = _import_torch()
    source_text, source_provenance, root = upstream.load_pinned_sources(cache_dir)
    operators = upstream.load_pinned_operators(source_text, source_provenance, torch)
    checkpoint = upstream.open_pinned_checkpoint(root)
    cfg_raw = checkpoint.config.get("text_config", checkpoint.config)
    if not isinstance(cfg_raw, dict):
        raise FixtureError("pinned config has no text_config")
    cfg = cfg_raw
    expected = {"hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48, "hc_count": 4}
    for key, value in expected.items():
        if int(cfg.get(key, -1)) != value:
            raise FixtureError(f"pinned source config {key}={cfg.get(key)!r}, expected {value}")
    if len(tokens) != CANONICAL_TOKEN_COUNT:
        raise FixtureError("source evaluator requires the canonical 17-token corpus")
    if any(int(token) >= int(cfg["vocab_size"]) for token in tokens):
        raise FixtureError("teacher corpus token exceeds pinned vocabulary")
    expected_ple_metadata = {
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": NG_SCALE,
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": NG_HEAD_OFFSETS,
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": NG_HEAD_VOCAB_SIZES,
    }
    for name, expected_values in expected_ple_metadata.items():
        record = checkpoint.tensor_rows(name, 0, len(expected_values))
        values = record.get("values")
        if record.get("dtype") != "I64" or values is None or list(values.data) != list(expected_values):
            raise FixtureError(f"pinned PLE metadata contents mismatch for {name}")
    store = _StreamWeights(checkpoint, torch)
    embedding_name = "model.language_model.embed_tokens.weight"
    embedding = torch.stack([store.rows(embedding_name, int(token), 1).reshape(-1) for token in tokens])
    wide = embedding.repeat(1, int(cfg["hc_count"]))
    layer_states: list[dict[str, object]] = []

    def read_ple_rows(global_rows: Sequence[int]) -> Any:
        pieces = []
        for global_row in global_rows:
            if global_row < 0 or global_row >= PLE_VALID_ROWS:
                raise FixtureError(f"PLE row {global_row} is outside the valid source range")
            shard = global_row // PLE_SHARD_ROWS
            local = global_row % PLE_SHARD_ROWS
            name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            pieces.append(store.rows(name, local, 1).reshape(-1))
        return torch.stack(pieces) if pieces else torch.empty((0, int(cfg["ple_embed_dim"]) // 16), dtype=wide.dtype)

    ple_metadata = {
        "layer_multipliers": store.rows(
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
            0,
            len(NG_SCALE),
        ).reshape(-1),
        "ngram_heads_offsets": store.rows(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            0,
            len(NG_HEAD_OFFSETS),
        ).reshape(-1),
        "ngram_heads_vocab_sizes": store.rows(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
            0,
            len(NG_HEAD_VOCAB_SIZES),
        ).reshape(-1),
    }
    for layer in range(int(cfg["num_hidden_layers"])):
        store.begin_layer()
        if layer == 1:
            ple = operators.ple(
                wide,
                tokens,
                cfg,
                ple_metadata,
                store.full,
                read_ple_rows,
            )
            wide = wide + ple
        attn_prefix = f"model.language_model.layers.{layer}.attn_hyper_connection"
        attn_weights = _load_hc_weights(store, attn_prefix, combine=True)
        mixed, hyper_input, injection_weights = operators.hyperconnection(wide, attn_weights, cfg, combine=True)
        if str(cfg["layer_types"][layer]) in ("qwen_sparse_attention", "full_attention"):
            attention_weights = {
                "index_qk": store.full(f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight"),
                "index_q_norm": store.full(f"model.language_model.layers.{layer}.self_attn.indexer.q_layernorm.weight"),
                "index_k_norm": store.full(f"model.language_model.layers.{layer}.self_attn.indexer.k_layernorm.weight"),
                "q": store.full(f"model.language_model.layers.{layer}.self_attn.q_proj.weight"),
                "q_norm": store.full(f"model.language_model.layers.{layer}.self_attn.q_norm.weight"),
                "k": store.full(f"model.language_model.layers.{layer}.self_attn.k_proj.weight"),
                "k_norm": store.full(f"model.language_model.layers.{layer}.self_attn.k_norm.weight"),
                "v": store.full(f"model.language_model.layers.{layer}.self_attn.v_proj.weight"),
                "out": store.full(f"model.language_model.layers.{layer}.self_attn.o_proj.weight"),
            }
            block_output, qsa_state = operators.qsa(mixed, attention_weights, cfg, layer)
            gdn_state = None
        else:
            gdn_weights = {
                "qkv": store.full(f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
                "z": store.full(f"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight"),
                "b": store.full(f"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight"),
                "a": store.full(f"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight"),
                "conv": store.full(f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"),
                "a_log": store.full(f"model.language_model.layers.{layer}.linear_attn.A_log.weight")
                if f"model.language_model.layers.{layer}.linear_attn.A_log.weight" in checkpoint.weight_map
                else store.full(f"model.language_model.layers.{layer}.linear_attn.A_log"),
                "dt_bias": store.full(f"model.language_model.layers.{layer}.linear_attn.dt_bias"),
                "norm": store.full(f"model.language_model.layers.{layer}.linear_attn.norm.weight"),
                "out": store.full(f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"),
            }
            block_output, gdn_state = operators.gdn(mixed, gdn_weights, cfg, layer)
            qsa_state = None
        wide = operators.inject(hyper_input, block_output, injection_weights)
        mlp_prefix = f"model.language_model.layers.{layer}.mlp_hyper_connection"
        mlp_weights = _load_hc_weights(store, mlp_prefix, combine=True)
        mlp_input, mlp_hyper_input, mlp_injection_weights = operators.hyperconnection(wide, mlp_weights, cfg, combine=True)
        moe = operators.moe(mlp_input, cfg, store.full, store.expert_rows, layer)
        wide = operators.inject(mlp_hyper_input, moe, mlp_injection_weights)
        state = {
            "layer": layer,
            "wide_sha256": hashlib.sha256(_tensor_bytes(wide)).hexdigest(),
            "layer_cache_bytes": store.current_bytes,
        }
        if layer == 1:
            state["ple_output_sha256"] = hashlib.sha256(_tensor_bytes(ple)).hexdigest()
        if qsa_state is not None:
            state["qsa_attention_sha256"] = hashlib.sha256(_tensor_bytes(qsa_state)).hexdigest()
        if gdn_state is not None:
            recurrent = gdn_state.get("recurrent")
            conv = gdn_state.get("conv")
            if recurrent is None or conv is None:
                raise FixtureError("pinned GDN cache did not capture recurrent and convolution state")
            state["gdn_recurrent_sha256"] = hashlib.sha256(_tensor_bytes(recurrent)).hexdigest()
            state["gdn_conv_sha256"] = hashlib.sha256(_tensor_bytes(conv)).hexdigest()
        layer_states.append(state)
    final_prefix = "model.language_model.hyper_connection_mixer"
    final_weights = _load_hc_weights(store, final_prefix, combine=False)
    hidden = operators.hyperconnection(wide, final_weights, cfg, combine=False)
    del final_weights, wide
    logits_rows: list[dict[str, object]] = []
    max_values: list[Any] = [torch.empty(0, dtype=torch.float32) for _ in tokens]
    max_ids: list[Any] = [torch.empty(0, dtype=torch.int64) for _ in tokens]
    logsum = torch.full((len(tokens),), float("-inf"), dtype=torch.float64)
    target_logits = [None for _ in tokens]
    vocab = int(cfg["vocab_size"])
    for start in range(0, vocab, MAX_LM_HEAD_ROWS):
        count = min(MAX_LM_HEAD_ROWS, vocab - start)
        head_rows = store.rows("lm_head.weight", start, count)
        chunk = torch.matmul(hidden, head_rows.transpose(0, 1)).float()
        if not bool(torch.isfinite(chunk).all()):
            raise FixtureError("nonfinite streamed lm_head logits")
        logsum = torch.logaddexp(logsum, torch.logsumexp(chunk.double(), dim=-1))
        for token_index in range(len(tokens)):
            joined_values = torch.cat([max_values[token_index], chunk[token_index]])
            joined_ids = torch.cat([max_ids[token_index], torch.arange(start, start + count, dtype=torch.int64)])
            order = torch.argsort(joined_values, descending=True, stable=True)[: min(TOP_K, joined_values.numel())]
            max_values[token_index] = joined_values[order]
            max_ids[token_index] = joined_ids[order]
            target = int(tokens[token_index + 1]) if token_index + 1 < len(tokens) else None
            if target is not None and start <= target < start + count:
                target_logits[token_index] = float(chunk[token_index, target - start])
        del head_rows, chunk
    for index in range(len(tokens)):
        top_logits = [_finite(float(item), "top logit") for item in max_values[index].tolist()]
        top_ids = [int(item) for item in max_ids[index].tolist()]
        target = int(tokens[index + 1]) if index + 1 < len(tokens) else None
        target_logit = target_logits[index]
        if target is not None and target_logit is None:
            raise FixtureError("streamed lm_head did not produce target logit")
        logits_rows.append(
            {
                "position": index,
                "input_id": int(tokens[index]),
                "target_id": target,
                "target_logit": target_logit,
                "logsumexp": _finite(float(logsum[index]), "logsumexp"),
                "top_ids": top_ids,
                "top_logits": top_logits,
                "top1": top_ids[0],
            }
        )
    state_summary = {"layers": layer_states, "final_hidden_sha256": hashlib.sha256(_tensor_bytes(hidden)).hexdigest()}
    state_digest = hashlib.sha256(json.dumps(state_summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows = _astrea_rows(logits_rows, tokens, variant="qwen4-source-reference", arch="source", notes="pinned AST source layer stream")
    return {
        "schema": QUALITY_SCHEMA,
        "variant": "qwen4-source-reference",
        "tokens": list(tokens),
        "corpus_sha256": CANONICAL_TOKEN_SHA256,
        "rows": logits_rows,
        "quality_rows": rows,
        "ppl": _ppl(logits_rows),
        "source": {
            "model": upstream.HF_MODEL,
            "revision": upstream.HF_CONFIG_COMMIT,
            "sources": source_provenance,
            "operators": operators.provenance(),
            "checkpoint": checkpoint.provenance(),
            "read_bytes": checkpoint.read_bytes,
            "peak_layer_cache_bytes": store.peak_bytes,
            "layer_count": len(layer_states),
            "no_full_model_residency": True,
            "deterministic_verified": False,
        },
        "state_summary_sha256": state_digest,
        "mtp": {"status": "unavailable", "reason": "native MTP adapter is not admitted"},
    }


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise FixtureError(f"source reference requires PyTorch: {exc}") from exc
    return torch


def _ppl(rows: Sequence[Mapping[str, object]]) -> float | None:
    losses = []
    for row in rows[:-1]:
        target = row.get("target_logit")
        if target is None:
            raise FixtureError("missing scored target logit")
        losses.append(float(row["logsumexp"]) - float(target))
    if not losses or any(not math.isfinite(value) for value in losses):
        raise FixtureError("invalid teacher-forced loss rows")
    return math.exp(sum(losses) / len(losses))


def _astrea_rows(rows: Sequence[Mapping[str, object]], tokens: Sequence[int], *, variant: str, arch: str, notes: str) -> list[dict[str, object]]:
    return [{"variant": variant, "arch": arch, "scoring_mode": "teacher_forced", "n_chunks": 1, "mean_kld": None, "mean_kld_ci_lo": None, "mean_kld_ci_hi": None, "p99_kld": None, "ppl": _ppl(rows), "notes": notes + f"; n_tokens={len(tokens)}; n_scored={len(tokens)-1}"}]


def _row_distribution(row: Mapping[str, object], vocab: int) -> tuple[dict[int, float], float]:
    logsum = float(row["logsumexp"])
    ids = [int(item) for item in row["top_ids"]]
    logits = [float(item) for item in row["top_logits"]]
    if len(ids) != len(logits) or len(set(ids)) != len(ids):
        raise FixtureError("quality row top-k representation is malformed")
    masses = {token: math.exp(logit - logsum) for token, logit in zip(ids, logits)}
    if any(not math.isfinite(value) or value < 0.0 for value in masses.values()):
        raise FixtureError("nonfinite quality probability")
    tail = 1.0 - sum(masses.values())
    if tail < -1e-6:
        raise FixtureError("top-k probability mass exceeds one")
    return masses, max(0.0, tail)


def compare(reference_path: str | Path, candidate_path: str | Path, out_path: str | Path) -> dict[str, object]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    for label, value in (("reference", reference), ("candidate", candidate)):
        if not isinstance(value, dict) or value.get("schema") != QUALITY_SCHEMA:
            raise FixtureError(f"{label} is not a {QUALITY_SCHEMA} artifact")
        if value.get("corpus_sha256") != CANONICAL_TOKEN_SHA256:
            raise FixtureError(f"{label} corpus digest is not canonical")
    ref_tokens = reference.get("tokens")
    cand_tokens = candidate.get("tokens")
    if ref_tokens != cand_tokens or ref_tokens != _load_corpus(CANONICAL_TOKEN_METADATA)[0]:
        raise FixtureError("quality artifacts do not use the same canonical token corpus")
    ref_rows, cand_rows = reference.get("rows"), candidate.get("rows")
    if not isinstance(ref_rows, list) or not isinstance(cand_rows, list) or len(ref_rows) != len(cand_rows):
        raise FixtureError("quality artifacts have incompatible rows")
    vocab = 248320
    klds: list[float] = []
    top1_equal = 0
    for ref, cand in zip(ref_rows, cand_rows):
        ref_mass, ref_tail = _row_distribution(ref, vocab)
        cand_mass, cand_tail = _row_distribution(cand, vocab)
        union = set(ref_mass) | set(cand_mass)
        ref_other = ref_tail / max(1, vocab - len(ref_mass))
        cand_other = cand_tail / max(1, vocab - len(cand_mass))
        kl = 0.0
        for token in union:
            p = ref_mass.get(token, ref_other)
            q = cand_mass.get(token, cand_other)
            if p > 0.0 and q <= 0.0:
                raise FixtureError("candidate top-k representation has zero comparison mass")
            if p > 0.0:
                kl += p * math.log(p / q)
        if ref_tail > 0.0:
            kl += ref_tail * math.log(max(ref_other, 1e-300) / max(cand_other, 1e-300))
        klds.append(_finite(kl, "KL divergence"))
        top1_equal += int(int(ref.get("top1")) == int(cand.get("top1")))
    scored = len(ref_rows) - 1
    candidate_ppl = _ppl(cand_rows)
    row = {"variant": "qwen4-candidate", "arch": "source", "scoring_mode": "teacher_forced", "n_chunks": 1, "mean_kld": sum(klds) / len(klds), "mean_kld_ci_lo": sum(klds) / len(klds), "mean_kld_ci_hi": sum(klds) / len(klds), "p99_kld": sorted(klds)[max(0, math.ceil(0.99 * len(klds)) - 1)], "ppl": candidate_ppl, "notes": f"reference={Path(reference_path)}; rows={len(klds)}; scored={scored}; top1_agreement={top1_equal}/{len(klds)}"}
    report = {"schema": QUALITY_SCHEMA, "identity": {"corpus_sha256": CANONICAL_TOKEN_SHA256, "n_tokens": len(ref_rows), "n_scored": scored, "target_shift": 1}, "rows": [row], "comparison": {"reference": str(reference_path), "candidate": str(candidate_path), "mean_kld": row["mean_kld"], "p99_kld": row["p99_kld"], "top1_agreement": top1_equal / len(klds), "ppl_candidate": candidate_ppl}, "mtp": {"status": "unavailable", "reason": "native MTP adapter is not admitted"}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", action="store_true", help="run the genuine pinned layer-streamed source evaluator")
    parser.add_argument("--compare", action="store_true", help="compare two quality artifacts")
    parser.add_argument("--tokens", default=CANONICAL_TOKEN_METADATA, help="canonical corpus metadata JSON")
    parser.add_argument("--reference", help="reference quality artifact for --compare")
    parser.add_argument("--candidate", help="candidate quality artifact for --compare")
    parser.add_argument("--out", required=True, help="quality artifact output path")
    parser.add_argument("--cache-dir", help="bounded source/checkpoint cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tokens, identity = _load_corpus(args.tokens)
        if args.compare:
            if not args.reference or not args.candidate:
                raise FixtureError("--compare requires --reference and --candidate")
            compare(args.reference, args.candidate, args.out)
        elif args.source_reference:
            result = _stream_source(tokens, args.cache_dir)
            result["corpus"] = identity
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            raise FixtureError("choose exactly one of --source-reference or --compare")
    except (FixtureError, OSError, ValueError, RuntimeError) as exc:
        print(f"qwen4 quality failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
