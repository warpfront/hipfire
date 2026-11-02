# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed Qwen4 teacher evaluation, exact full-vocabulary KL, and Astrea-compatible comparison.

Two paths share one layer-major source engine.  The canonical path
(`--source-reference`/`--compare`) is the pinned 17-token top-32 regression.  The exact
path (`--exact-export`, `--exact-floor`/`--paired-floor`, `--exact-compare`,
`--exact-paired`) consumes a frozen `hipfire.qwen4.kl_corpus.v1` manifest of independent
sequences, executes one decoder layer at a time over every sequence (and over the native
and reverse-experts operator variants in lockstep), exports full-vocabulary F32 logits and
all-position routes, and reduces them to exact float64 KL with document bootstraps.
See QWEN4_PROFILES.md ("Exact KL quality method") for usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
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
    """Torch conversion over the bounded `PinnedCheckpoint` row API.

    Tensors land on `device`.  Within one layer (`begin_layer` clears) `full` tensors
    are cached and expert slices live in a byte-capped LRU, so every sequence/variant
    stepping through the layer shares one physical read.  Raw (`*_raw`) checkpoint
    methods are used when the shared reader provides them; otherwise the bounded
    record API (`tensor_rows`) is the fallback.  `io_seconds` covers fetch, decode and
    host-to-device movement.
    """

    def __init__(self, checkpoint: upstream.PinnedCheckpoint, torch: Any, *, expert_cache_bytes: int = 8 << 30, device: Any = None):
        import collections

        self.checkpoint = checkpoint
        self.torch = torch
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.current_bytes = 0
        self.peak_bytes = 0
        self.io_seconds = 0.0
        self.expert_cache_bytes = int(expert_cache_bytes)
        self._full: dict[str, Any] = {}
        self._experts: Any = collections.OrderedDict()
        self._expert_live = 0
        self.counters = {"full_hits": 0, "full_misses": 0, "expert_hits": 0, "expert_misses": 0, "tensors_loaded": 0, "bytes_loaded": 0}

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    def begin_layer(self) -> None:
        self.current_bytes = 0
        self._full.clear()
        self._experts.clear()
        self._expert_live = 0

    def _track(self, value: Any) -> Any:
        size = int(value.numel() * value.element_size())
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.counters["tensors_loaded"] += 1
        self.counters["bytes_loaded"] += size
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
        return output

    def _from_raw(self, record: Mapping[str, object]) -> Any:
        converter = getattr(upstream, "raw_record_to_torch", None)
        if converter is not None:
            return converter(record, self.torch)
        torch = self.torch
        dtype = str(record["dtype"])
        shape = tuple(int(item) for item in record["slice_shape"])  # type: ignore[index]
        buffer = bytearray(record["raw"])  # type: ignore[arg-type]
        mapping = {"BF16": torch.uint16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32}
        if dtype not in mapping:
            raise FixtureError(f"unsupported streamed dtype {dtype}")
        output = torch.frombuffer(buffer, dtype=mapping[dtype])
        if dtype == "BF16":
            output = output.view(torch.bfloat16)
        return output.reshape(shape)

    def _timed(self, produce: Any) -> Any:
        started = time.perf_counter()
        value = produce()
        if self.device.type != "cpu":
            value = value.to(self.device)
        self.io_seconds += time.perf_counter() - started
        return value

    def _fetch_rows(self, name: str, row_start: int, row_count: int | None) -> Any:
        raw = getattr(self.checkpoint, "tensor_rows_raw", None)
        if raw is not None:
            return self._timed(lambda: self._from_raw(raw(name, row_start, row_count)))
        if row_count is None:
            descriptor = upstream._tensor_descriptor(self.checkpoint.client, self.checkpoint.weight_map, name)
            row_count = int(descriptor["shape"][0]) - row_start
        return self._timed(lambda: self._record(self.checkpoint.tensor_rows(name, row_start, row_count)))

    def rows(self, name: str, row_start: int, row_count: int) -> Any:
        return self._track(self._fetch_rows(name, row_start, row_count))

    def gather(self, name: str, rows: Sequence[int]) -> Any:
        """Arbitrary rows of a table, order kept (shape `[len(rows), ...]`)."""
        rows = [int(item) for item in rows]
        gather = getattr(self.checkpoint, "gather_rows_raw", None)
        if gather is not None:
            return self._track(self._timed(lambda: self._from_raw(gather(name, rows))))
        pieces = [self._fetch_rows(name, row, 1) for row in rows]
        return self._track(self.torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0])

    def expert_rows(self, name: str, expert: int, row_start: int, row_count: int) -> Any:
        key = (name, int(expert), int(row_start), int(row_count))
        cached = self._experts.get(key)
        if cached is not None:
            self._experts.move_to_end(key)
            self.counters["expert_hits"] += 1
            return cached
        self.counters["expert_misses"] += 1
        raw = getattr(self.checkpoint, "expert_rows_raw", None)
        if raw is not None:
            value = self._timed(lambda: self._from_raw(raw(name, expert, row_start, row_count)))
        else:
            value = self._timed(lambda: self._record(self.checkpoint.tensor_expert_rows(name, expert, row_start, row_count)))
        self._track(value)
        size = int(value.numel() * value.element_size())
        self._experts[key] = value
        self._expert_live += size
        while self._expert_live > self.expert_cache_bytes and len(self._experts) > 1:
            _old_key, old = self._experts.popitem(last=False)
            self._expert_live -= int(old.numel() * old.element_size())
        return value

    def full(self, name: str) -> Any:
        cached = self._full.get(name)
        if cached is not None:
            self.counters["full_hits"] += 1
            return cached
        self.counters["full_misses"] += 1
        raw = getattr(self.checkpoint, "tensor_rows_raw", None)
        if raw is not None:
            value = self._fetch_rows(name, 0, None)
        else:
            descriptor = upstream._tensor_descriptor(self.checkpoint.client, self.checkpoint.weight_map, name)
            shape = tuple(int(item) for item in descriptor["shape"])
            rows = shape[0]
            row_items = math.prod(shape[1:]) if len(shape) > 1 else 1
            itemsize, _ = upstream._dtype_info(str(descriptor["dtype"]))
            if not itemsize:
                raise FixtureError(f"unsupported dtype for {name}")
            rows_per_read = max(1, upstream.MAX_TENSOR_SLICE_BYTES // (row_items * itemsize))
            pieces = [self._fetch_rows(name, start, min(rows_per_read, rows - start)) for start in range(0, rows, rows_per_read)]
            value = self.torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        self._track(value)
        self._full[name] = value
        return value


def _load_hc_weights(store: _StreamWeights, prefix: str, *, combine: bool) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "hc_norm": store.full(f"{prefix}.hc_norm.weight"),
        "down": store.full(f"{prefix}.input_mix_weight_down.weight"),
        "up": store.full(f"{prefix}.input_mix_weight_up.weight"),
    }
    if combine:
        weights["block"] = store.full(f"{prefix}.block_inject_weight.weight")
    return weights


def _stream_source(tokens: Sequence[int], cache_dir: str | Path | None, local_dir: str | Path | None = None) -> dict[str, object]:
    """Canonical 17-token top-32 source path (regression); runs on the shared layer-major engine."""
    torch = _import_torch()
    engine = _Engine.open(torch, cache_dir, local_dir)
    try:
        cfg = engine.cfg
        if len(tokens) != CANONICAL_TOKEN_COUNT:
            raise FixtureError("source evaluator requires the canonical 17-token corpus")
        if any(int(token) >= int(cfg["vocab_size"]) for token in tokens):
            raise FixtureError("teacher corpus token exceeds pinned vocabulary")
        result = engine.run_pass([list(tokens)], capture_states=True)
        layer_states = result["states"]["default"][0]
        hidden = result["hidden"]["default"][0]
        logits_rows: list[dict[str, object]] = []
        max_values: list[Any] = [torch.empty(0, dtype=torch.float32) for _ in tokens]
        max_ids: list[Any] = [torch.empty(0, dtype=torch.int64) for _ in tokens]
        logsum = torch.full((len(tokens),), float("-inf"), dtype=torch.float64)
        target_logits = [None for _ in tokens]
        for start, count, chunks in engine.head_chunks({"default": hidden}, MAX_LM_HEAD_ROWS):
            chunk = chunks["default"]
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
            del chunk, chunks
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
                "sources": engine.source_provenance,
                "operators": engine.ops["default"].provenance(),
                "checkpoint": engine.checkpoint.provenance(),
                "read_bytes": engine.checkpoint.read_bytes,
                "peak_layer_cache_bytes": engine.store.peak_bytes,
                "layer_count": len(layer_states),
                "no_full_model_residency": True,
                "deterministic_verified": False,
            },
            "state_summary_sha256": state_digest,
            "mtp": {"status": "unavailable", "reason": "native MTP adapter is not admitted"},
        }
    finally:
        engine.close()


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


# ---------------------------------------------------------------------------
# Generalized KL-method path: manifest corpus, layer-major source forward,
# exact full-vocabulary export, exact F64 KL, document bootstrap, timing.
# ---------------------------------------------------------------------------


KL_CORPUS_SCHEMA = "hipfire.qwen4.kl_corpus.v1"
EXACT_EXPORT_SCHEMA = "hipfire.qwen4.exact_export.v1"
EXACT_COMPARE_SCHEMA = "hipfire.qwen4.exact_compare.v1"
EXACT_VERSION = "kl-method-1"
STRATA = ("text", "assistant")
VOCAB_SIZE = 248_320
MAX_THREADS = 16
MAX_MEMORY_GIB = 64
DEFAULT_NICE = 10
DEFAULT_HEAD_ROWS = 4096
DEFAULT_EXPERT_CACHE_BYTES = 8 << 30
DEFAULT_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 20260101
NEGATIVE_KL_TOLERANCE = 1e-9
STABILITY_ABS_HALF_WIDTH = 0.002
STABILITY_REL_HALF_WIDTH = 0.05
STABILITY_REL_MEAN_FLOOR = 0.02
PROJECTED_CORPUS_TOKENS = 327_680
PROJECTED_CORPUS_SCORED = 65_536
CONTEXT_STRATA_EDGES = (128, 512, 2048, 8192)
REDUCTION_ORDERS = ("native", "reverse-experts")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _u32_sha256(values: Sequence[int]) -> str:
    import numpy as np

    return _sha256_hex(np.asarray(values, dtype="<u4").tobytes())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------------ corpus


class CorpusSequence:
    """One independent sequence: IDs, stratum, scored input positions, role mask."""

    __slots__ = ("index", "id", "stratum", "tokens", "score_positions", "role_mask", "meta")

    def __init__(
        self,
        index: int,
        seq_id: str,
        stratum: str,
        tokens: Sequence[int],
        score_positions: Sequence[int],
        role_mask: Sequence[int] | None,
        meta: Mapping[str, object] | None = None,
    ):
        self.index = index
        self.id = seq_id
        self.stratum = stratum
        self.tokens = tuple(int(item) for item in tokens)
        self.score_positions = tuple(int(item) for item in score_positions)
        self.role_mask = None if role_mask is None else tuple(int(item) for item in role_mask)
        self.meta = dict(meta or {})

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(self.tokens[position + 1] for position in self.score_positions)

    def identity(self) -> dict[str, object]:
        return {
            "id": self.id,
            "stratum": self.stratum,
            "length": self.length,
            "scored": len(self.score_positions),
            "tokens_sha256": _u32_sha256(self.tokens),
            "score_positions_sha256": _u32_sha256(self.score_positions),
            "role_mask_sha256": None if self.role_mask is None else _u32_sha256(self.role_mask),
            "half": self.meta.get("half"),
            "reset": "sequence_start",
        }


class KlCorpus:
    """Validated frozen corpus; every sequence resets all recurrent/PLE state at position 0."""

    def __init__(self, name: str, seed: int, sequences: Sequence[CorpusSequence], vocab_size: int = VOCAB_SIZE):
        if not sequences:
            raise FixtureError("corpus has no sequences")
        self.name = name
        self.seed = int(seed)
        self.sequences = tuple(sequences)
        self.vocab_size = int(vocab_size)
        ids = [item.id for item in self.sequences]
        if len(set(ids)) != len(ids):
            raise FixtureError("corpus sequence ids must be unique")
        self.offsets: list[int] = []
        total = 0
        for item in self.sequences:
            self.offsets.append(total)
            total += item.length
        self.input_tokens = total
        self.scored_positions = sum(len(item.score_positions) for item in self.sequences)

    def identity(self) -> dict[str, object]:
        sequences = [item.identity() for item in self.sequences]
        body = {"schema": KL_CORPUS_SCHEMA, "vocab_size": self.vocab_size, "sequences": sequences}
        strata = {name: sum(1 for item in self.sequences if item.stratum == name) for name in STRATA}
        return {
            "schema": KL_CORPUS_SCHEMA,
            "name": self.name,
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "sequence_count": len(self.sequences),
            "input_tokens": self.input_tokens,
            "scored_positions": self.scored_positions,
            "strata_sequences": strata,
            "corpus_sha256": _sha256_hex(_canonical_json(body)),
            "sequences": sequences,
        }

    def score_index(self) -> Any:
        """uint32 [N,3] rows `(sequence_index, input_position, target_token)` in export order."""
        import numpy as np

        rows = [
            (item.index, position, item.tokens[position + 1])
            for item in self.sequences
            for position in item.score_positions
        ]
        return np.asarray(rows, dtype="<u4").reshape(-1, 3)

    def row_offsets(self) -> list[int]:
        """First scored-row index of every sequence."""
        starts, total = [], 0
        for item in self.sequences:
            starts.append(total)
            total += len(item.score_positions)
        return starts


def _score_positions_from(raw: Mapping[str, object], length: int, label: str) -> list[int]:
    given = [key for key in ("score_positions", "score_mask", "score_range") if key in raw]
    if len(given) != 1:
        raise FixtureError(f"{label}: exactly one of score_positions, score_mask, score_range is required")
    key = given[0]
    value = raw[key]
    if key == "score_positions":
        if not isinstance(value, list) or any(isinstance(p, bool) or not isinstance(p, int) for p in value):
            raise FixtureError(f"{label}: score_positions must be a list of integers")
        positions = list(value)
    elif key == "score_mask":
        if not isinstance(value, list) or len(value) != length or any(item not in (0, 1) for item in value):
            raise FixtureError(f"{label}: score_mask must be {length} values in {{0,1}}")
        positions = [index for index, item in enumerate(value) if item]
    else:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(p, bool) or not isinstance(p, int) for p in value)
            or value[0] > value[1]
        ):
            raise FixtureError(f"{label}: score_range must be [start,end) integers")
        positions = list(range(value[0], value[1]))
    if not positions:
        raise FixtureError(f"{label}: sequence has no scored positions")
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise FixtureError(f"{label}: score positions must be strictly increasing")
    if positions[0] < 0 or positions[-1] > length - 2:
        raise FixtureError(
            f"{label}: score positions must lie in [0,{length - 2}] (the last token has no next-token target)"
        )
    return positions


def corpus_from_dict(
    manifest: Mapping[str, object],
    *,
    base_dir: Path | None = None,
    vocab_size: int = VOCAB_SIZE,
) -> KlCorpus:
    """Validate a `hipfire.qwen4.kl_corpus.v1` manifest. See QWEN4_PROFILES.md for the schema."""
    import numpy as np

    if manifest.get("schema") != KL_CORPUS_SCHEMA:
        raise FixtureError(f"corpus schema must be {KL_CORPUS_SCHEMA}")
    raw_sequences = manifest.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise FixtureError("corpus manifest has no sequences")
    payload = None
    payload_spec = manifest.get("token_payload")
    if payload_spec is not None:
        if not isinstance(payload_spec, dict) or not isinstance(payload_spec.get("path"), str):
            raise FixtureError("token_payload requires a relative path")
        relative = Path(payload_spec["path"])
        if relative.is_absolute() or base_dir is None:
            raise FixtureError("token_payload path must be relative to the corpus manifest")
        path = (base_dir / relative).resolve()
        if not path.is_file() or base_dir.resolve() not in path.parents:
            raise FixtureError(f"token payload is missing or escapes the manifest directory: {path}")
        raw = path.read_bytes()
        if len(raw) % 4:
            raise FixtureError("token payload is not u32le")
        if payload_spec.get("sha256") != _sha256_hex(raw):
            raise FixtureError("token payload sha256 mismatch")
        payload = np.frombuffer(raw, dtype="<u4")
    sequences: list[CorpusSequence] = []
    for index, raw_seq in enumerate(raw_sequences):
        if not isinstance(raw_seq, dict):
            raise FixtureError(f"sequence {index} is not an object")
        seq_id = raw_seq.get("id")
        if not isinstance(seq_id, str) or not seq_id:
            raise FixtureError(f"sequence {index} needs a string id")
        label = f"sequence {seq_id!r}"
        stratum = raw_seq.get("stratum")
        if stratum not in STRATA:
            raise FixtureError(f"{label}: stratum must be one of {STRATA}")
        if raw_seq.get("reset", True) is not True:
            raise FixtureError(f"{label}: only independent sequences (reset=true) are supported")
        if "tokens" in raw_seq:
            tokens = raw_seq["tokens"]
            if not isinstance(tokens, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in tokens):
                raise FixtureError(f"{label}: tokens must be a list of integers")
        else:
            if payload is None:
                raise FixtureError(f"{label}: no tokens and no token_payload")
            offset, length = raw_seq.get("offset"), raw_seq.get("length")
            if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 1:
                raise FixtureError(f"{label}: payload sequences need integer offset/length")
            if offset + length > payload.size:
                raise FixtureError(f"{label}: payload range exceeds payload size")
            tokens = payload[offset : offset + length].tolist()
        if len(tokens) < 2:
            raise FixtureError(f"{label}: needs at least two tokens (context + one next-token target)")
        if min(tokens) < 0 or max(tokens) >= vocab_size:
            raise FixtureError(f"{label}: token id outside [0,{vocab_size})")
        positions = _score_positions_from(raw_seq, len(tokens), label)
        role_mask = raw_seq.get("role_mask")
        if role_mask is not None:
            if not isinstance(role_mask, list) or len(role_mask) != len(tokens) or any(v not in (0, 1) for v in role_mask):
                raise FixtureError(f"{label}: role_mask must be {len(tokens)} values in {{0,1}}")
        if stratum == "assistant":
            if role_mask is None:
                raise FixtureError(f"{label}: assistant stratum requires role_mask")
            bad = [p for p in positions if not role_mask[p + 1]]
            if bad:
                raise FixtureError(f"{label}: scored target at position {bad[0] + 1} is not assistant content")
        targets = raw_seq.get("targets")
        if targets is not None:
            expected = [tokens[p + 1] for p in positions]
            if targets != expected:
                raise FixtureError(f"{label}: declared targets do not equal tokens[position+1]")
        meta = raw_seq.get("meta") or {}
        if not isinstance(meta, dict):
            raise FixtureError(f"{label}: meta must be an object")
        sequences.append(CorpusSequence(index, seq_id, str(stratum), tokens, positions, role_mask, meta))
    return KlCorpus(str(manifest.get("name", "unnamed")), int(manifest.get("seed", 0)), sequences, vocab_size)


def load_kl_corpus(path: str | Path, *, vocab_size: int = VOCAB_SIZE) -> KlCorpus:
    requested = Path(path).expanduser()
    try:
        manifest = json.loads(requested.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"invalid corpus manifest {requested}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FixtureError("corpus manifest must be a JSON object")
    return corpus_from_dict(manifest, base_dir=requested.resolve().parent, vocab_size=vocab_size)


def canonical_kl_corpus(tokens: Sequence[int]) -> KlCorpus:
    """The legacy 17-token canonical sequence as a one-sequence corpus (16 scored targets)."""
    sequence = CorpusSequence(0, "canonical-17", "text", tokens, range(len(tokens) - 1), None, {"canonical": True})
    return KlCorpus("canonical-17", 0, [sequence])


# ---------------------------------------------------------- exact KL math


class _ArrayRows:
    """Row-block reader over an in-memory [N,V] float array."""

    def __init__(self, array: Any):
        import numpy as np

        self.array = np.asarray(array)
        if self.array.ndim != 2:
            raise FixtureError("logits must be a [rows, vocab] matrix")
        self.rows, self.vocab = (int(self.array.shape[0]), int(self.array.shape[1]))

    def block(self, start: int, stop: int) -> Any:
        return self.array[start:stop]

    def finish(self) -> str | None:
        return None


class _FileRows:
    """Sequential row-block reader over an exported logits file, hashing raw bytes as it streams.

    `storage="f32"` reads little-endian float32; `storage="bf16"` reads bfloat16 words and
    widens them exactly (`word << 16` reinterpreted as float32), so no precision is lost.
    """

    def __init__(self, path: str | Path, rows: int, vocab: int, expected_sha256: str | None = None, storage: str = "f32"):
        if storage not in ("f32", "bf16"):
            raise FixtureError(f"unknown logits storage {storage!r}")
        self.path = Path(path)
        self.rows, self.vocab = int(rows), int(vocab)
        self.expected = expected_sha256
        self.storage = storage
        self.itemsize = 4 if storage == "f32" else 2
        size = self.path.stat().st_size
        if size != self.rows * self.vocab * self.itemsize:
            raise FixtureError(f"{self.path}: size {size} != rows*vocab*{self.itemsize} = {self.rows * self.vocab * self.itemsize}")
        self.hash = hashlib.sha256()
        self.cursor = 0

    def block(self, start: int, stop: int) -> Any:
        import numpy as np

        if start != self.cursor:
            raise FixtureError("file logits must be streamed sequentially")
        count = (stop - start) * self.vocab
        data = np.fromfile(self.path, dtype="<f4" if self.storage == "f32" else "<u2", count=count, offset=start * self.vocab * self.itemsize)
        if data.size != count:
            raise FixtureError(f"{self.path}: short read")
        self.hash.update(data.tobytes())
        self.cursor = stop
        if self.storage == "bf16":
            data = (data.astype(np.uint32) << 16).view(np.float32)
        return data.reshape(stop - start, self.vocab)

    def finish(self) -> str:
        digest = self.hash.hexdigest()
        if self.cursor == self.rows and self.expected is not None and digest != self.expected:
            raise FixtureError(f"{self.path}: sha256 {digest} != manifest {self.expected}")
        return digest


def _as_rows(source: Any) -> Any:
    return source if hasattr(source, "block") else _ArrayRows(source)


def exact_kl_streamed(
    teacher: Any,
    candidate: Any,
    targets: Sequence[int],
    *,
    block_rows: int = 64,
    block_vocab: int = 16384,
) -> dict[str, Any]:
    """Exact teacher->candidate full-vocabulary KL in F64 from actual logits.

    `KL = sum_v p_T (log p_T - log p_C)` is evaluated with one online pass over
    vocabulary blocks: running maxima/denominators for both rows and the
    `sum_v exp(a_v - m_T) (a_v - b_v)` accumulator (rescaled whenever the
    teacher maximum changes), so no probability is rounded or clipped and no
    row-wide F64 matrix is materialized.  `KL = W/S - (lse_T - lse_C)`.
    Materially negative values raise; tiny negatives (>= -1e-9) are kept raw.
    Returns per-row float64 arrays (kl, lse_*, nll_*) and int top-1 ids.
    """
    import numpy as np

    teacher_rows, candidate_rows = _as_rows(teacher), _as_rows(candidate)
    if (teacher_rows.rows, teacher_rows.vocab) != (candidate_rows.rows, candidate_rows.vocab):
        raise FixtureError("teacher and candidate logits have different shapes")
    rows, vocab = teacher_rows.rows, teacher_rows.vocab
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (rows,):
        raise FixtureError("one target token per logits row is required")
    if rows and (targets.min() < 0 or targets.max() >= vocab):
        raise FixtureError("target token outside the logits vocabulary")
    kl = np.empty(rows, dtype=np.float64)
    lse_t = np.empty(rows, dtype=np.float64)
    lse_c = np.empty(rows, dtype=np.float64)
    target_t = np.empty(rows, dtype=np.float64)
    target_c = np.empty(rows, dtype=np.float64)
    top_t = np.empty(rows, dtype=np.int64)
    top_c = np.empty(rows, dtype=np.int64)
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        count = r1 - r0
        tb, cb = teacher_rows.block(r0, r1), candidate_rows.block(r0, r1)
        mt = np.full(count, -np.inf)
        mc = np.full(count, -np.inf)
        st = np.zeros(count)
        sc = np.zeros(count)
        wt = np.zeros(count)
        best_t = np.full(count, -np.inf)
        best_c = np.full(count, -np.inf)
        arg_t = np.zeros(count, dtype=np.int64)
        arg_c = np.zeros(count, dtype=np.int64)
        tgt = targets[r0:r1]
        for v0 in range(0, vocab, block_vocab):
            v1 = min(vocab, v0 + block_vocab)
            a = np.asarray(tb[:, v0:v1], dtype=np.float64)
            b = np.asarray(cb[:, v0:v1], dtype=np.float64)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise FixtureError("nonfinite logits in exact KL input")
            ia, ib = a.argmax(axis=1), b.argmax(axis=1)
            ma, mb = a[np.arange(count), ia], b[np.arange(count), ib]
            upd = ma > best_t
            best_t = np.where(upd, ma, best_t)
            arg_t = np.where(upd, ia + v0, arg_t)
            upd = mb > best_c
            best_c = np.where(upd, mb, best_c)
            arg_c = np.where(upd, ib + v0, arg_c)
            new_mt = np.maximum(mt, ma)
            scale = np.exp(mt - new_mt)
            e = np.exp(a - new_mt[:, None])
            st = st * scale + e.sum(axis=1)
            wt = wt * scale + (e * (a - b)).sum(axis=1)
            mt = new_mt
            new_mc = np.maximum(mc, mb)
            sc = sc * np.exp(mc - new_mc) + np.exp(b - new_mc[:, None]).sum(axis=1)
            mc = new_mc
            inside = (tgt >= v0) & (tgt < v1)
            if inside.any():
                rr = np.nonzero(inside)[0]
                target_t[r0 + rr] = a[rr, tgt[rr] - v0]
                target_c[r0 + rr] = b[rr, tgt[rr] - v0]
        lt = mt + np.log(st)
        lc = mc + np.log(sc)
        kl[r0:r1] = wt / st - (lt - lc)
        lse_t[r0:r1], lse_c[r0:r1] = lt, lc
        top_t[r0:r1], top_c[r0:r1] = arg_t, arg_c
    teacher_hash, candidate_hash = teacher_rows.finish(), candidate_rows.finish()
    if rows and float(kl.min()) < -NEGATIVE_KL_TOLERANCE:
        raise FixtureError(f"materially negative KL {float(kl.min()):.3e}: inputs or reduction are inconsistent")
    return {
        "kl": kl,
        "lse_teacher": lse_t,
        "lse_candidate": lse_c,
        "nll_teacher": lse_t - target_t,
        "nll_candidate": lse_c - target_c,
        "top1_teacher": top_t,
        "top1_candidate": top_c,
        "teacher_sha256": teacher_hash,
        "candidate_sha256": candidate_hash,
    }


def exact_kl_materialized(teacher: Any, candidate: Any) -> Any:
    """Reference full-matrix F64 KL (log-softmax difference); used to cross-check the streamed reduction."""
    import numpy as np

    a = np.asarray(teacher, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    la = a - (a.max(axis=1, keepdims=True) + np.log(np.exp(a - a.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)))
    lb = b - (b.max(axis=1, keepdims=True) + np.log(np.exp(b - b.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)))
    return (np.exp(la) * (la - lb)).sum(axis=1)


# ------------------------------------------------------- bootstrap / stats


def _stratum_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{name}".encode()).digest()[:8], "little")


def doc_bootstrap(sums: Any, counts: Any, replicates: int, seed: int, name: str) -> Any:
    """Resample documents with replacement; return [replicates, K] position-weighted means.

    `sums[d,k]` is the sum of metric k over document d's scored rows and
    `counts[d]` its scored-row count.  The generator is keyed by `(seed,name)` so
    each stratum draws independently and reproducibly; all K columns share the
    draws (paired metrics/variants resample together).
    """
    import numpy as np

    docs = int(sums.shape[0])
    if docs < 1:
        raise FixtureError("bootstrap needs at least one document")
    rng = np.random.default_rng(_stratum_seed(seed, name))
    out = np.empty((replicates, sums.shape[1]), dtype=np.float64)
    step = max(1, 4_000_000 // max(1, docs * sums.shape[1]))
    for start in range(0, replicates, step):
        stop = min(replicates, start + step)
        idx = rng.integers(0, docs, size=(stop - start, docs))
        out[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
    return out


def weighted_quantile(values: Any, weights: Any, quantile: float) -> float:
    """Inverted-CDF weighted quantile (first value whose cumulative weight reaches `quantile`)."""
    import numpy as np

    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    cutoff = quantile * cumulative[-1]
    position = int(np.searchsorted(cumulative, cutoff - 1e-12 * cumulative[-1], side="left"))
    return float(np.asarray(values)[order][min(position, len(order) - 1)])


def stability_gate(mean: float, half_width: float) -> dict[str, object]:
    """Pilot-size gate: CI half-width <= 0.002 nat, and <= 5% of the mean once mean >= 0.02."""
    relative_applies = mean >= STABILITY_REL_MEAN_FLOOR
    absolute_ok = half_width <= STABILITY_ABS_HALF_WIDTH
    relative_ok = (half_width <= STABILITY_REL_HALF_WIDTH * mean) if relative_applies else True
    return {
        "half_width": half_width,
        "absolute_limit": STABILITY_ABS_HALF_WIDTH,
        "absolute_ok": absolute_ok,
        "relative_applies": relative_applies,
        "relative_limit": STABILITY_REL_HALF_WIDTH * mean if relative_applies else None,
        "relative_ok": relative_ok,
        "stable": bool(absolute_ok and relative_ok),
    }


def _length_bucket(length: int) -> str:
    low = 0
    for edge in CONTEXT_STRATA_EDGES:
        if length <= edge:
            return f"len_{low + 1}-{edge}" if low else f"len_<={edge}"
        low = edge
    return f"len_>{CONTEXT_STRATA_EDGES[-1]}"


def _metric_entry(mean: float, reps: Any, bootstrap: int) -> dict[str, object]:
    import numpy as np

    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    return {"mean": float(mean), "ci95": [lo, hi], "bootstrap_se": float(reps.std(ddof=1)) if bootstrap > 1 else None}


def summarize_columns(
    columns: Mapping[str, Any],
    seq_index: Any,
    sequences: Sequence[Mapping[str, object]],
    *,
    bootstrap: int,
    seed: int,
    primary: str = "kl",
    ppl_pairs: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Per-stratum and equal-weight mixture statistics with independent doc bootstraps.

    `columns` maps metric name -> float64 array over scored rows; `seq_index` maps
    rows to corpus sequences; `sequences` carries `stratum`, `length`, optional
    `half`.  Mixture = 0.5*text + 0.5*assistant for every statistic (means,
    CI replicates and weighted quantiles), never a position-count mixture.
    """
    import numpy as np

    names = list(columns)
    seq_index = np.asarray(seq_index, dtype=np.int64)
    matrix = np.stack([np.asarray(columns[name], dtype=np.float64) for name in names], axis=1)
    seq_count = len(sequences)
    counts_all = np.bincount(seq_index, minlength=seq_count).astype(np.float64)
    sums_all = np.stack(
        [np.bincount(seq_index, weights=matrix[:, k], minlength=seq_count) for k in range(len(names))],
        axis=1,
    )
    primary_k = names.index(primary)
    per_stratum: dict[str, dict[str, object]] = {}
    replicate_sets: dict[str, Any] = {}
    mean_sets: dict[str, Any] = {}
    quantile_inputs: dict[str, Any] = {}
    for stratum in STRATA:
        docs = [i for i, item in enumerate(sequences) if item["stratum"] == stratum and counts_all[i] > 0]
        if not docs:
            continue
        doc_array = np.asarray(docs)
        sums, counts = sums_all[doc_array], counts_all[doc_array]
        means = sums.sum(axis=0) / counts.sum()
        reps = doc_bootstrap(sums, counts, bootstrap, seed, stratum)
        rows = np.isin(seq_index, doc_array)
        replicate_sets[stratum], mean_sets[stratum] = reps, means
        quantile_inputs[stratum] = matrix[rows, primary_k]
        metrics = {name: _metric_entry(means[k], reps[:, k], bootstrap) for k, name in enumerate(names)}
        lo, hi = metrics[primary]["ci95"]  # type: ignore[index]
        entry: dict[str, object] = {
            "n_positions": int(counts.sum()),
            "n_documents": len(docs),
            "metrics": metrics,
            "stability": stability_gate(metrics[primary]["mean"], (hi - lo) / 2.0),  # type: ignore[index]
            "quantiles": {
                f"p{int(q * 100)}": weighted_quantile(quantile_inputs[stratum], np.ones(int(rows.sum())), q)
                for q in (0.5, 0.95, 0.99)
            },
        }
        halves: dict[int, list[int]] = {0: [], 1: []}
        for order, doc in enumerate(docs):
            declared = sequences[doc].get("half")
            halves[int(declared) if declared in (0, 1) else order % 2].append(doc)
        if all(len(v) >= 2 for v in halves.values()):
            stats = []
            for half in (0, 1):
                d = np.asarray(halves[half])
                reps_h = doc_bootstrap(sums_all[d], counts_all[d], bootstrap, seed, f"{stratum}|half{half}")
                stats.append((float(sums_all[d, primary_k].sum() / counts_all[d].sum()), float(reps_h[:, primary_k].std(ddof=1))))
            difference = abs(stats[0][0] - stats[1][0])
            limit = 1.96 * math.sqrt(stats[0][1] ** 2 + stats[1][1] ** 2)
            entry["split_half"] = {
                "means": [stats[0][0], stats[1][0]],
                "se": [stats[0][1], stats[1][1]],
                "abs_difference": difference,
                "limit_1.96se": limit,
                "agree": bool(difference <= limit),
            }
        else:
            entry["split_half"] = None
        buckets: dict[str, dict[str, float]] = {}
        for doc in docs:
            bucket = buckets.setdefault(_length_bucket(int(sequences[doc]["length"])), {"docs": 0.0, "n": 0.0, "sum": 0.0})  # type: ignore[arg-type]
            bucket["docs"] += 1
            bucket["n"] += counts_all[doc]
            bucket["sum"] += sums_all[doc, primary_k]
        entry["context_length_strata"] = {
            key: {"n_documents": int(v["docs"]), "n_positions": int(v["n"]), "mean": v["sum"] / v["n"]}
            for key, v in sorted(buckets.items())
        }
        per_stratum[stratum] = entry
    if len(per_stratum) == len(STRATA):
        reps = 0.5 * (replicate_sets["text"] + replicate_sets["assistant"])
        means = 0.5 * (mean_sets["text"] + mean_sets["assistant"])
        metrics = {name: _metric_entry(means[k], reps[:, k], bootstrap) for k, name in enumerate(names)}
        lo, hi = metrics[primary]["ci95"]  # type: ignore[index]
        values = np.concatenate([quantile_inputs[s] for s in STRATA])
        weights = np.concatenate([np.full(quantile_inputs[s].size, 0.5 / quantile_inputs[s].size) for s in STRATA])
        per_stratum["mixture"] = {
            "weighting": "0.5*text + 0.5*assistant",
            "metrics": metrics,
            "stability": stability_gate(metrics[primary]["mean"], (hi - lo) / 2.0),  # type: ignore[index]
            "quantiles": {f"p{int(q * 100)}": weighted_quantile(values, weights, q) for q in (0.5, 0.95, 0.99)},
        }
    if ppl_pairs:
        for entry in per_stratum.values():
            metrics = entry["metrics"]  # type: ignore[assignment]
            for label, column in ppl_pairs.items():
                if column in metrics:  # type: ignore[operator]
                    m = metrics[column]  # type: ignore[index]
                    metrics[label] = {"mean": math.exp(m["mean"]), "ci95": [math.exp(m["ci95"][0]), math.exp(m["ci95"][1])]}  # type: ignore[index]
    return per_stratum


# -------------------------------------------------------------- runtime


def _resolve_device(torch: Any, device: Any) -> Any:
    """`cpu`, `cuda` or `cuda:N` (ROCm builds also expose `cuda`); CUDA must actually be available."""
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise FixtureError(f"unsupported device {device!r}; use cpu or cuda[:N]")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise FixtureError("--device cuda requested but torch.cuda.is_available() is false")
    return resolved


def device_info(torch: Any, device: Any = "cpu") -> dict[str, object]:
    """Execution identity: device, torch/HIP/CUDA versions and (for GPU) device properties."""
    resolved = _resolve_device(torch, device)
    info: dict[str, object] = {
        "device": str(resolved),
        "type": resolved.type,
        "name": None,
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_threads": int(torch.get_num_threads()),
        "gcn_arch": None,
        "total_memory_bytes": None,
    }
    if resolved.type == "cuda":
        props = torch.cuda.get_device_properties(resolved)
        info["name"] = props.name
        info["gcn_arch"] = getattr(props, "gcnArchName", None)
        info["total_memory_bytes"] = int(props.total_memory)
    return info


def apply_runtime_limits(
    device: str = "cpu",
    threads: int | None = None,
    memory_gib: float | None = None,
    nice: int | None = None,
) -> dict[str, object]:
    """Process limits applied before torch is imported.

    CPU mode: threads <= 16 (default 16), CUDA/HIP hidden, RLIMIT_AS <= 64 GiB (default 64)
    and niceness >= 10.  GPU mode (`cuda[:N]`): no AS limit unless `--memory-limit-gib` > 0
    (GPU mappings need large virtual space), threads/nice only when requested; peak host RSS
    and device memory are recorded instead.
    """
    cpu = str(device).split(":")[0] == "cpu"
    if cpu:
        threads = MAX_THREADS if threads is None else threads
        memory_gib = float(MAX_MEMORY_GIB) if memory_gib is None else memory_gib
        nice = DEFAULT_NICE if nice is None else nice
        if not 0 < float(memory_gib) <= MAX_MEMORY_GIB:
            raise FixtureError(f"CPU mode requires --memory-limit-gib in (0,{MAX_MEMORY_GIB}]")
    else:
        memory_gib = float(memory_gib or 0)
        nice = 0 if nice is None else nice
    if threads is not None and not 1 <= int(threads) <= (MAX_THREADS if cpu else 4096):
        raise FixtureError(f"--threads must be in [1,{MAX_THREADS if cpu else 4096}]")
    if threads is not None:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(int(threads))
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = "-1"
        os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
    limit = None
    if memory_gib > 0:
        import resource

        limit = int(float(memory_gib) * (1 << 30))
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY and hard < limit:
            limit = hard
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    current_nice = os.nice(0)
    if nice > current_nice:
        current_nice = os.nice(nice - current_nice)
    return {
        "device_request": str(device),
        "threads": None if threads is None else int(threads),
        "rlimit_as_bytes": limit,
        "memory_enforcement": "RLIMIT_AS" if limit else "none (peak host RSS and device memory recorded)",
        "nice": current_nice,
        "cuda_disabled": cpu,
    }


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return 0


def _peak_rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _checkpoint_counters(checkpoint: Any) -> dict[str, float]:
    """Flat numeric transfer counters from the shared checkpoint reader (`read_bytes`, `transfer_stats()`)."""
    out: dict[str, float] = {}
    value = getattr(checkpoint, "read_bytes", None)
    if isinstance(value, (int, float)):
        out["read_bytes"] = float(value)
    stats = getattr(checkpoint, "transfer_stats", None)
    if callable(stats):
        try:
            for key, item in stats().items():
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    out[key] = float(item)
        except Exception:  # noqa: BLE001 - counters are diagnostic only
            pass
    return out


def project_runtime(
    timing: Mapping[str, object],
    *,
    corpus_tokens: int = PROJECTED_CORPUS_TOKENS,
    corpus_scored: int = PROJECTED_CORPUS_SCORED,
    pass_tokens: int | None = None,
) -> dict[str, object]:
    """Extrapolate a measured pilot to the full corpus (an estimate; caveats are recorded).

    I/O (weight reads+decode) is charged once per pass, arithmetic per forwarded
    token per variant, LM-head arithmetic per scored row per variant.  With
    `pass_tokens=None` the whole corpus is one layer-major pass (one weight
    traversal).  Attention is superlinear in context, so the projection is only as
    good as the pilot's length mix.
    """
    passes_run = max(1, int(timing["passes"]))  # type: ignore[arg-type]
    variants = max(1, int(timing["variants"]))  # type: ignore[arg-type]
    tokens_run = max(1, int(timing["input_tokens"]))  # type: ignore[arg-type]
    scored_run = max(1, int(timing["scored_rows"]))  # type: ignore[arg-type]
    io_per_pass = (float(timing["layers_io_seconds"]) + float(timing["head_io_seconds"])) / passes_run  # type: ignore[arg-type]
    compute_per_token = (float(timing["layers_compute_seconds"]) + float(timing["embedding_seconds"])) / (tokens_run * variants)  # type: ignore[arg-type]
    head_per_row = float(timing["head_compute_seconds"]) / (scored_run * variants)  # type: ignore[arg-type]
    full_passes = 1 if not pass_tokens else max(1, math.ceil(corpus_tokens / pass_tokens))

    def estimate(n_variants: int) -> float:
        return full_passes * io_per_pass + n_variants * (compute_per_token * corpus_tokens + head_per_row * corpus_scored)

    return {
        "corpus_input_tokens": corpus_tokens,
        "corpus_scored_rows": corpus_scored,
        "full_passes": full_passes,
        "io_seconds_per_pass_measured": io_per_pass,
        "compute_seconds_per_input_token_per_variant_measured": compute_per_token,
        "head_compute_seconds_per_scored_row_per_variant_measured": head_per_row,
        "estimated_seconds_one_variant": estimate(1),
        "estimated_seconds_lockstep_variants_run": estimate(variants),
        "estimated_hours_one_variant": estimate(1) / 3600.0,
        "estimated_hours_lockstep_variants_run": estimate(variants) / 3600.0,
        "variants_in_pilot": variants,
        "basis": f"pilot passes={passes_run} input_tokens={tokens_run} scored_rows={scored_run}",
        "caveats": [
            "estimate only: linear in tokens; attention cost is superlinear in context length",
            "per-pass I/O assumes the pilot read a comparable expert union per layer (see per-layer expert_misses)",
            "cold-network seconds are inside io_seconds; a warm/local run will be faster",
        ],
    }


# --------------------------------------------------------------- engine


class _Engine:
    """Layer-major pinned-source forward over independent sequences and one or more operator variants.

    One decoder layer's weights (and a bounded expert cache) serve every
    sequence and every variant before being released, so a lockstep native +
    reverse-experts run costs one physical weight traversal.  Sequences never
    share attention/convolution/recurrent/PLE state: each operator call is
    per-sequence with position 0 at the sequence start.  Variants keep fully
    independent hidden states.
    """

    def __init__(self, *, torch: Any, operators: Mapping[str, Any], checkpoint: Any, store: "_StreamWeights", cfg: Mapping[str, Any], ple_metadata: Mapping[str, Any], source_provenance: Any, max_layers: int | None):
        self.torch, self.ops, self.checkpoint, self.store = torch, dict(operators), checkpoint, store
        self.device = store.device
        self.head_dtype: dict[str, str] = {}
        self.variant_names = list(self.ops)
        self.cfg, self.ple_metadata, self.source_provenance = cfg, ple_metadata, source_provenance
        layers = int(cfg["num_hidden_layers"])
        if max_layers is not None and not 1 <= max_layers <= layers:
            raise FixtureError(f"--max-layers must be in [1,{layers}]")
        self.layer_count = layers if max_layers is None else int(max_layers)
        self.truncated = self.layer_count != layers
        self.head_stats = {"seconds": 0.0, "io_seconds": 0.0, "rows": 0}

    @classmethod
    def open(
        cls,
        torch: Any,
        cache_dir: str | Path | None,
        local_dir: str | Path | None = None,
        *,
        variants: Sequence[tuple[str, str | None, Any]] = (("default", None, None),),
        expert_cache_bytes: int = DEFAULT_EXPERT_CACHE_BYTES,
        max_layers: int | None = None,
        device: Any = "cpu",
        checkpoint_options: Mapping[str, object] | None = None,
    ) -> "_Engine":
        """`variants` = `(name, reduction_order|None, route_sink|None)` per operator set."""
        import inspect

        source_text, source_provenance, root = upstream.load_pinned_sources(cache_dir)
        operators: dict[str, Any] = {}
        for name, reduction_order, route_sink in variants:
            kwargs: dict[str, Any] = {}
            if route_sink is not None:
                kwargs["route_sink"] = route_sink
            if reduction_order is not None:
                kwargs["reduction_order"] = reduction_order
            operators[name] = upstream.load_pinned_operators(source_text, source_provenance, torch, **kwargs)
        open_kwargs: dict[str, Any] = {}
        accepted = inspect.signature(upstream.open_pinned_checkpoint).parameters
        if local_dir is not None:
            open_kwargs["local_dir"] = local_dir
        for key, value in (checkpoint_options or {}).items():
            if key in accepted and value is not None:
                open_kwargs[key] = value
        checkpoint = upstream.open_pinned_checkpoint(root, **open_kwargs)
        try:
            cfg_raw = checkpoint.config.get("text_config", checkpoint.config)
            if not isinstance(cfg_raw, dict):
                raise FixtureError("pinned config has no text_config")
            cfg = cfg_raw
            expected = {"hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48, "hc_count": 4}
            for key, value in expected.items():
                if int(cfg.get(key, -1)) != value:
                    raise FixtureError(f"pinned source config {key}={cfg.get(key)!r}, expected {value}")
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
            store = _StreamWeights(checkpoint, torch, expert_cache_bytes=expert_cache_bytes, device=_resolve_device(torch, device))
            ple_metadata = {
                "layer_multipliers": store.rows("model.language_model.layers.1.ple.ple_embedding.layer_multipliers", 0, len(NG_SCALE)).reshape(-1),
                "ngram_heads_offsets": store.rows("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets", 0, len(NG_HEAD_OFFSETS)).reshape(-1),
                "ngram_heads_vocab_sizes": store.rows("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes", 0, len(NG_HEAD_VOCAB_SIZES)).reshape(-1),
            }
            return cls(torch=torch, operators=operators, checkpoint=checkpoint, store=store, cfg=cfg, ple_metadata=ple_metadata, source_provenance=source_provenance, max_layers=max_layers)
        except BaseException:
            _close_checkpoint(checkpoint)
            raise

    def close(self) -> None:
        _close_checkpoint(self.checkpoint)

    def provenance(self, variant: str) -> dict[str, object]:
        checkpoint = dict(self.checkpoint.provenance())
        checkpoint.pop("read_bytes", None)
        return {
            "model": upstream.HF_MODEL,
            "revision": upstream.HF_CONFIG_COMMIT,
            "sources": self.source_provenance,
            "operators": self.ops[variant].provenance(),
            "checkpoint": checkpoint,
            "layers_executed": self.layer_count,
            "truncated_layers": self.truncated,
        }

    def _group(self, label: str) -> None:
        begin = getattr(self.checkpoint, "begin_group", None)
        if callable(begin):
            begin(label)

    def _release(self, label: str) -> None:
        release = getattr(self.checkpoint, "release_group", None)
        if callable(release):
            release(label)

    def _layer_block(self, layer: int) -> tuple[dict[str, Any], dict[str, Any], str]:
        store, checkpoint = self.store, self.checkpoint
        base = f"model.language_model.layers.{layer}"
        attn_hc = _load_hc_weights(store, f"{base}.attn_hyper_connection", combine=True)
        if str(self.cfg["layer_types"][layer]) in ("qwen_sparse_attention", "full_attention"):
            block = {
                "index_qk": store.full(f"{base}.self_attn.indexer.index_qk_proj.weight"),
                "index_q_norm": store.full(f"{base}.self_attn.indexer.q_layernorm.weight"),
                "index_k_norm": store.full(f"{base}.self_attn.indexer.k_layernorm.weight"),
                "q": store.full(f"{base}.self_attn.q_proj.weight"),
                "q_norm": store.full(f"{base}.self_attn.q_norm.weight"),
                "k": store.full(f"{base}.self_attn.k_proj.weight"),
                "k_norm": store.full(f"{base}.self_attn.k_norm.weight"),
                "v": store.full(f"{base}.self_attn.v_proj.weight"),
                "out": store.full(f"{base}.self_attn.o_proj.weight"),
            }
            return attn_hc, block, "qsa"
        a_log_name = f"{base}.linear_attn.A_log.weight"
        block = {
            "qkv": store.full(f"{base}.linear_attn.in_proj_qkv.weight"),
            "z": store.full(f"{base}.linear_attn.in_proj_z.weight"),
            "b": store.full(f"{base}.linear_attn.in_proj_b.weight"),
            "a": store.full(f"{base}.linear_attn.in_proj_a.weight"),
            "conv": store.full(f"{base}.linear_attn.conv1d.weight"),
            "a_log": store.full(a_log_name) if a_log_name in checkpoint.weight_map else store.full(f"{base}.linear_attn.A_log"),
            "dt_bias": store.full(f"{base}.linear_attn.dt_bias"),
            "norm": store.full(f"{base}.linear_attn.norm.weight"),
            "out": store.full(f"{base}.linear_attn.out_proj.weight"),
        }
        return attn_hc, block, "gdn"

    def run_pass(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        on_sequence: Any = None,
        capture_states: bool = False,
        pass_index: int = 0,
    ) -> dict[str, Any]:
        """Forward every sequence through all layers (layer-major) for every variant.

        `on_sequence(variant, local_sequence_index, layer)` fires before each
        per-sequence layer step (route recorders key on it).  Returns final hidden per
        variant and sequence plus per-layer timing records.
        """
        torch, cfg, store, checkpoint = self.torch, self.cfg, self.store, self.checkpoint
        sequences = [list(map(int, item)) for item in sequences]
        vocab = int(cfg["vocab_size"])
        if any(not item or min(item) < 0 or max(item) >= vocab for item in sequences):
            raise FixtureError("sequence token outside the pinned vocabulary")
        records: list[dict[str, object]] = []
        states: dict[str, list[list[dict[str, object]]]] = {name: [[] for _ in sequences] for name in self.variant_names}
        positions = sum(len(item) for item in sequences)
        variants = len(self.variant_names)
        with torch.inference_mode():
            self._group("embedding")
            store.begin_layer()
            started, io0 = time.perf_counter(), store.io_seconds
            embedding_name = "model.language_model.embed_tokens.weight"
            unique = sorted({token for item in sequences for token in item})
            rows = store.gather(embedding_name, unique)
            slot = {token: i for i, token in enumerate(unique)}
            base_wides = [rows[torch.as_tensor([slot[t] for t in item], dtype=torch.long, device=self.device)].repeat(1, int(cfg["hc_count"])) for item in sequences]
            del rows
            wides = {name: ([w.clone() for w in base_wides] if i + 1 < variants else base_wides) for i, name in enumerate(self.variant_names)}
            del base_wides
            embed_record = {"pass": pass_index, "seconds": time.perf_counter() - started, "io_seconds": store.io_seconds - io0, "positions": positions}
            self._release("embedding")
            ple_cache: dict[int, Any] = {}

            def read_ple_rows(global_rows: Sequence[int]) -> Any:
                missing: dict[int, list[int]] = {}
                for global_row in global_rows:
                    row = int(global_row)
                    if row not in ple_cache:
                        if row < 0 or row >= PLE_VALID_ROWS:
                            raise FixtureError(f"PLE row {row} is outside the valid source range")
                        missing.setdefault(row // PLE_SHARD_ROWS, [])
                        if row not in missing[row // PLE_SHARD_ROWS]:
                            missing[row // PLE_SHARD_ROWS].append(row)
                for shard, shard_rows in missing.items():
                    name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                    fetched = store.gather(name, [row % PLE_SHARD_ROWS for row in shard_rows])
                    for i, row in enumerate(shard_rows):
                        ple_cache[row] = fetched[i].reshape(-1)
                if global_rows:
                    return torch.stack([ple_cache[int(row)] for row in global_rows])
                return torch.empty((0, int(cfg["ple_embed_dim"]) // 16), dtype=wides[self.variant_names[0]][0].dtype, device=self.device)

            for layer in range(self.layer_count):
                label = f"layer{layer:02d}"
                self._group(label)
                store.begin_layer()
                started, io0, counters0 = time.perf_counter(), store.io_seconds, _checkpoint_counters(checkpoint)
                stats0 = store.snapshot()
                self._reset_peak()
                base = f"model.language_model.layers.{layer}"
                attn_hc, block_weights, block_kind = self._layer_block(layer)
                mlp_hc = _load_hc_weights(store, f"{base}.mlp_hyper_connection", combine=True)
                for index, tokens in enumerate(sequences):
                    for name in self.variant_names:
                        ops = self.ops[name]
                        if on_sequence is not None:
                            on_sequence(name, index, layer)
                        ple_output = None
                        if layer == 1:
                            ple_output = ops.ple(wides[name][index], tokens, cfg, self.ple_metadata, store.full, read_ple_rows)
                            wides[name][index] = wides[name][index] + ple_output
                        mixed, hyper_input, injection = ops.hyperconnection(wides[name][index], attn_hc, cfg, combine=True)
                        if block_kind == "qsa":
                            block_output, qsa_state = ops.qsa(mixed, block_weights, cfg, layer)
                            gdn_state = None
                        else:
                            block_output, gdn_state = ops.gdn(mixed, block_weights, cfg, layer)
                            qsa_state = None
                        wide = ops.inject(hyper_input, block_output, injection)
                        mlp_input, mlp_hyper_input, mlp_injection = ops.hyperconnection(wide, mlp_hc, cfg, combine=True)
                        moe = ops.moe(mlp_input, cfg, store.full, store.expert_rows, layer)
                        wides[name][index] = ops.inject(mlp_hyper_input, moe, mlp_injection)
                        if capture_states:
                            state: dict[str, object] = {"layer": layer, "wide_sha256": hashlib.sha256(_tensor_bytes(wides[name][index])).hexdigest(), "layer_cache_bytes": store.current_bytes}
                            if ple_output is not None:
                                state["ple_output_sha256"] = hashlib.sha256(_tensor_bytes(ple_output)).hexdigest()
                            if qsa_state is not None:
                                state["qsa_attention_sha256"] = hashlib.sha256(_tensor_bytes(qsa_state)).hexdigest()
                            if gdn_state is not None:
                                recurrent, conv = gdn_state.get("recurrent"), gdn_state.get("conv")
                                if recurrent is None or conv is None:
                                    raise FixtureError("pinned GDN cache did not capture recurrent and convolution state")
                                state["gdn_recurrent_sha256"] = hashlib.sha256(_tensor_bytes(recurrent)).hexdigest()
                                state["gdn_conv_sha256"] = hashlib.sha256(_tensor_bytes(conv)).hexdigest()
                            states[name][index].append(state)
                self._sync()
                seconds = time.perf_counter() - started
                counters1 = _checkpoint_counters(checkpoint)
                stats1 = store.snapshot()
                records.append(
                    {
                        "pass": pass_index,
                        "layer": layer,
                        "kind": block_kind,
                        "sequences": len(sequences),
                        "variants": variants,
                        "positions": positions,
                        "seconds": seconds,
                        "io_seconds": store.io_seconds - io0,
                        "compute_seconds": seconds - (store.io_seconds - io0),
                        "tokens_per_second_per_variant": positions / seconds if seconds > 0 else None,
                        "checkpoint_counter_deltas": {k: counters1[k] - counters0.get(k, 0.0) for k in counters1},
                        "store": {k: stats1[k] - stats0[k] for k in stats1},
                        "layer_weight_bytes": store.current_bytes,
                        "rss_bytes": _rss_bytes(),
                        "peak_rss_bytes": _peak_rss_bytes(),
                        **self._device_memory(),
                    }
                )
                del attn_hc, block_weights, mlp_hc
                self._release(label)
            self._group("final")
            store.begin_layer()
            started, io0 = time.perf_counter(), store.io_seconds
            final_weights = _load_hc_weights(store, "model.language_model.hyper_connection_mixer", combine=False)
            hiddens = {
                name: [self.ops[name].hyperconnection(wide, final_weights, cfg, combine=False) for wide in wides[name]]
                for name in self.variant_names
            }
            self._sync()
            final_record = {"pass": pass_index, "seconds": time.perf_counter() - started, "io_seconds": store.io_seconds - io0}
            self._release("final")
            del final_weights, wides
        return {"hidden": hiddens, "layers": records, "embedding": embed_record, "final": final_record, "states": states}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _reset_peak(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def _device_memory(self) -> dict[str, int]:
        if self.device.type != "cuda":
            return {}
        cuda = self.torch.cuda
        return {
            "device_allocated_bytes": int(cuda.memory_allocated(self.device)),
            "device_peak_bytes": int(cuda.max_memory_allocated(self.device)),
            "device_reserved_bytes": int(cuda.memory_reserved(self.device)),
        }

    def head_chunks(self, hidden: Mapping[str, Any], chunk_rows: int) -> Any:
        """Yield `(start, count, {variant: float32 logits [rows,count]})`.

        Each head chunk is read once and applied to every variant's hidden rows;
        the matmul output (bf16 for the source forward) is widened to F32 exactly as in the
        source forward, and its dtype is recorded in `head_dtype`.  Hidden rows must already
        live on the engine device.
        """
        torch, store = self.torch, self.store
        vocab = int(self.cfg["vocab_size"])
        self._group("head")
        try:
            with torch.inference_mode():
                for start in range(0, vocab, chunk_rows):
                    count = min(chunk_rows, vocab - start)
                    began, io0 = time.perf_counter(), store.io_seconds
                    head_rows = store.rows("lm_head.weight", start, count)
                    io = store.io_seconds - io0
                    transposed = head_rows.transpose(0, 1)
                    chunks = {}
                    for name, value in hidden.items():
                        out = torch.matmul(value, transposed)
                        self.head_dtype[name] = str(out.dtype).replace("torch.", "")
                        chunk = out.float()
                        if not bool(torch.isfinite(chunk).all()):
                            raise FixtureError("nonfinite streamed lm_head logits")
                        chunks[name] = chunk
                    self._sync()
                    self.head_stats["io_seconds"] += io
                    self.head_stats["seconds"] += time.perf_counter() - began
                    del head_rows, transposed
                    yield start, count, chunks
            self.head_stats["rows"] += int(next(iter(hidden.values())).shape[0])
        finally:
            self._release("head")


def _close_checkpoint(checkpoint: Any) -> None:
    """Delete any ephemeral stream cache the shared reader staged (idempotent)."""
    for name in ("release_all", "close"):
        method = getattr(checkpoint, name, None)
        if callable(method):
            try:
                method()
            except Exception:  # noqa: BLE001 - cleanup must not mask the real failure
                pass


# ---------------------------------------------------------------- export


class _PwriteFile:
    def __init__(self, path: Path, size: int):
        self.path = path
        self.size = size
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(self.fd, size)

    def write_at(self, offset: int, data: Any) -> None:
        view = memoryview(data).cast("B")
        done = 0
        while done < len(view):
            done += os.pwrite(self.fd, view[done:], offset + done)

    def close(self) -> None:
        os.close(self.fd)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 26)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class _RouteRecorder:
    """Route sink target: all-position ordered expert IDs and F32 scores, layer-major tapes [L,P,K]."""

    def __init__(self, out_dir: Path, corpus: KlCorpus, layers: int, experts: int):
        self.out_dir, self.corpus, self.layers, self.experts = out_dir, corpus, layers, experts
        self.current: int | None = None
        self.calls: dict[tuple[int, int], int] = {}
        self.k: int | None = None
        self.ids_file: _PwriteFile | None = None
        self.scores_file: _PwriteFile | None = None

    def _open(self, k: int) -> None:
        total = self.corpus.input_tokens
        self.k = k
        self.ids_file = _PwriteFile(self.out_dir / "route_ids.u16le", self.layers * total * k * 2)
        self.scores_file = _PwriteFile(self.out_dir / "route_scores.f32le", self.layers * total * k * 4)

    def sink(self, layer: int, logits: Any, scores: Any, indices: Any) -> None:
        import numpy as np

        if self.current is None:
            raise FixtureError("route sink called outside a sequence")
        seq = self.corpus.sequences[self.current]
        ids = indices.detach().cpu().reshape(-1, indices.shape[-1]).numpy().astype(np.int64)
        weights = scores.detach().float().cpu().reshape(-1, scores.shape[-1]).numpy().astype("<f4")
        if ids.shape[0] != seq.length or weights.shape != ids.shape:
            raise FixtureError(f"route capture rows {ids.shape} do not match sequence {seq.id!r} length {seq.length}")
        if ids.min() < 0 or ids.max() >= self.experts:
            raise FixtureError("routed expert id out of range")
        ordered = np.sort(ids, axis=1)
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise FixtureError("duplicate expert id in one route row")
        if not np.isfinite(weights).all():
            raise FixtureError("nonfinite route score")
        if layer < 0 or layer >= self.layers:
            raise FixtureError(f"route sink layer {layer} out of range")
        if self.ids_file is None:
            self._open(int(ids.shape[1]))
        elif ids.shape[1] != self.k:
            raise FixtureError("route top-k changed between calls")
        key = (self.current, layer)
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] != 1:
            raise FixtureError(f"route sink fired {self.calls[key]} times for sequence {seq.id!r} layer {layer}")
        base = layer * self.corpus.input_tokens + self.corpus.offsets[self.current]
        self.ids_file.write_at(base * self.k * 2, ids.astype("<u2"))  # type: ignore[union-attr]
        self.scores_file.write_at(base * self.k * 4, weights)  # type: ignore[union-attr]

    def verify(self, sequence_indices: Sequence[int], layers: int) -> None:
        missing = [(s, l) for s in sequence_indices for l in range(layers) if self.calls.get((s, l)) != 1]
        if missing:
            raise FixtureError(f"route capture incomplete, first missing (sequence,layer)={missing[0]}")

    def close(self) -> None:
        for item in (self.ids_file, self.scores_file):
            if item is not None:
                item.close()


def plan_passes(corpus: KlCorpus, pass_tokens: int | None) -> list[list[int]]:
    """Group consecutive sequences into passes of <= pass_tokens inputs.

    `None`/<=0 puts the whole corpus in ONE layer-major pass (one weight traversal);
    each extra pass re-reads every layer's weights.
    """
    if not pass_tokens or pass_tokens <= 0:
        return [[item.index for item in corpus.sequences]]
    passes: list[list[int]] = []
    current: list[int] = []
    total = 0
    for item in corpus.sequences:
        if current and total + item.length > pass_tokens:
            passes.append(current)
            current, total = [], 0
        current.append(item.index)
        total += item.length
    if current:
        passes.append(current)
    return passes


def _aggregate_timing(
    layer_records: Sequence[Mapping[str, object]],
    embed_records: Sequence[Mapping[str, object]],
    final_records: Sequence[Mapping[str, object]],
    head: Mapping[str, object],
    *,
    passes: int,
    variants: int,
    input_tokens: int,
    scored_rows: int,
) -> dict[str, object]:
    per_layer: dict[int, dict[str, float]] = {}
    for record in layer_records:
        entry = per_layer.setdefault(int(record["layer"]), {"seconds": 0.0, "io_seconds": 0.0, "compute_seconds": 0.0, "positions": 0.0, "peak_rss_bytes": 0.0, "read_bytes": 0.0, "network_seconds": 0.0, "expert_misses": 0.0, "expert_hits": 0.0})
        entry["seconds"] += float(record["seconds"])  # type: ignore[arg-type]
        entry["io_seconds"] += float(record["io_seconds"])  # type: ignore[arg-type]
        entry["compute_seconds"] += float(record["compute_seconds"])  # type: ignore[arg-type]
        entry["positions"] += float(record["positions"])  # type: ignore[arg-type]
        entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], float(record["peak_rss_bytes"]))  # type: ignore[arg-type]
        deltas = record["checkpoint_counter_deltas"]
        entry["read_bytes"] += float(deltas.get("read_bytes", 0.0))  # type: ignore[attr-defined]
        entry["network_seconds"] += float(deltas.get("network_seconds", 0.0))  # type: ignore[attr-defined]
        store = record["store"]
        entry["expert_misses"] += float(store["expert_misses"])  # type: ignore[index]
        entry["expert_hits"] += float(store["expert_hits"])  # type: ignore[index]
    per_layer_rows = []
    for layer in sorted(per_layer):
        entry = per_layer[layer]
        per_layer_rows.append({"layer": layer, **entry, "tokens_per_second_per_variant": entry["positions"] / entry["seconds"] if entry["seconds"] else None})
    layers_seconds = sum(float(r["seconds"]) for r in layer_records)  # type: ignore[arg-type]
    layers_io = sum(float(r["io_seconds"]) for r in layer_records)  # type: ignore[arg-type]
    embed_seconds = sum(float(r["seconds"]) for r in embed_records) + sum(float(r["seconds"]) for r in final_records)  # type: ignore[arg-type]
    timing: dict[str, object] = {
        "passes": passes,
        "variants": variants,
        "input_tokens": input_tokens,
        "scored_rows": scored_rows,
        "per_layer": per_layer_rows,
        "layers_seconds": layers_seconds,
        "layers_io_seconds": layers_io,
        "layers_compute_seconds": layers_seconds - layers_io,
        "embedding_seconds": embed_seconds,
        "head_seconds": float(head["seconds"]),  # type: ignore[arg-type]
        "head_io_seconds": float(head["io_seconds"]),  # type: ignore[arg-type]
        "head_compute_seconds": float(head["seconds"]) - float(head["io_seconds"]),  # type: ignore[arg-type]
    }
    total = layers_seconds + embed_seconds + float(head["seconds"])  # type: ignore[arg-type]
    timing["traversal_seconds"] = total
    timing["input_tokens_per_second_per_variant"] = input_tokens / total if total else None
    timing["peak_rss_bytes"] = max([float(r["peak_rss_bytes"]) for r in layer_records] or [0.0])  # type: ignore[arg-type]
    timing["peak_device_bytes"] = max([float(r.get("device_peak_bytes", 0)) for r in layer_records] or [0.0])  # type: ignore[attr-defined,arg-type]
    return timing


def export_exact(
    corpus: KlCorpus,
    out_dir: str | Path,
    *,
    orders: Sequence[str] = ("native",),
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    pass_tokens: int | None = None,
    head_rows: int = DEFAULT_HEAD_ROWS,
    expert_cache_bytes: int = DEFAULT_EXPERT_CACHE_BYTES,
    hash_files: bool = True,
    export_hidden: bool = False,
    max_layers: int | None = None,
    checkpoint_options: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
    device: Any = "cpu",
    logits_storage: str = "f32",
    torch: Any = None,
) -> dict[str, dict[str, object]]:
    """Run the pinned source layer-major over `corpus`; write one exact export directory per order.

    All `orders` execute in lockstep over ONE physical weight traversal (shared per-layer
    weight/expert cache) with independent hidden states and route capture; one decoder
    layer's weights live on `device` at a time.  Directory `out_dir/<order>/` holds
    `score_index.u32le` [N,3] (sequence, position, target), `logits.f32le` [N,V] (bf16 head
    matmul widened to F32) or, with `logits_storage="bf16"`, `logits.bf16le` (the forward's own
    bf16 words; only accepted when the actual head matmul dtype is bfloat16 and the F32
    widening round-trips bit-exactly), `route_ids.u16le` / `route_scores.f32le` [L,P,K] over
    ALL input positions (P = corpus input tokens, sequences concatenated), optional
    `final_hidden.<dtype>le` [N,H], and `export.json` (deterministic `identity` + measured
    `runtime`).  Returns `{order: manifest}`.
    """
    import numpy as np

    orders = tuple(orders)
    if not orders or len(set(orders)) != len(orders) or any(o not in REDUCTION_ORDERS for o in orders):
        raise FixtureError(f"orders must be distinct values from {REDUCTION_ORDERS}")
    if logits_storage not in ("f32", "bf16"):
        raise FixtureError("logits_storage must be f32 or bf16")
    if sys.byteorder != "little":
        raise FixtureError("exact export requires a little-endian host")
    torch = torch or _import_torch()
    execution = device_info(torch, device)
    root = Path(out_dir)
    outs = {order: root / order for order in orders}
    for path in outs.values():
        path.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    recorders: dict[str, _RouteRecorder] = {}

    def make_sink(order: str) -> Any:
        return lambda layer, logits, scores, indices: recorders[order].sink(layer, logits, scores, indices)

    engine = _Engine.open(
        torch,
        cache_dir,
        local_dir,
        variants=[(order, order, make_sink(order)) for order in orders],
        expert_cache_bytes=expert_cache_bytes,
        max_layers=max_layers,
        checkpoint_options=checkpoint_options,
        device=device,
    )
    logits_name = "logits.f32le" if logits_storage == "f32" else "logits.bf16le"
    logits_itemsize = 4 if logits_storage == "f32" else 2
    logits_files: dict[str, _PwriteFile] = {}
    hidden_files: dict[str, _PwriteFile] = {}
    hidden_dtype: str | None = None
    transfer: dict[str, object] = {}
    try:
        cfg = engine.cfg
        vocab, hidden_size = int(cfg["vocab_size"]), int(cfg["hidden_size"])
        if corpus.vocab_size != vocab:
            raise FixtureError("corpus vocabulary does not match the pinned source")
        score_index = corpus.score_index()
        rows_total = int(score_index.shape[0])
        for order in orders:
            recorders[order] = _RouteRecorder(outs[order], corpus, engine.layer_count, int(cfg["num_experts"]))
            (outs[order] / "score_index.u32le").write_bytes(score_index.astype("<u4").tobytes())
            logits_files[order] = _PwriteFile(outs[order] / logits_name, rows_total * vocab * logits_itemsize)
        row_starts = corpus.row_offsets()
        passes = plan_passes(corpus, pass_tokens)
        layer_records: list[dict[str, object]] = []
        embed_records: list[dict[str, object]] = []
        final_records: list[dict[str, object]] = []
        pass_summaries: list[dict[str, object]] = []
        for pass_index, members in enumerate(passes):
            pass_started = time.perf_counter()

            def on_sequence(order: str, local_index: int, _layer: int, members: list[int] = members) -> None:
                recorders[order].current = members[local_index]

            result = engine.run_pass(
                [corpus.sequences[i].tokens for i in members],
                on_sequence=on_sequence,
                pass_index=pass_index,
            )
            for recorder in recorders.values():
                recorder.current = None
            layer_records.extend(result["layers"])
            embed_records.append(result["embedding"])
            final_records.append(result["final"])
            hidden_rows: dict[str, Any] = {}
            for order in orders:
                selected = []
                for local, seq_idx in enumerate(members):
                    positions = torch.as_tensor(list(corpus.sequences[seq_idx].score_positions), dtype=torch.long, device=engine.device)
                    selected.append(result["hidden"][order][local][positions])
                hidden_rows[order] = torch.cat(selected, dim=0)
            del result
            first_row = row_starts[members[0]]
            if export_hidden:
                for order in orders:
                    rows_t = hidden_rows[order]
                    if order not in hidden_files:
                        hidden_dtype = str(rows_t.dtype).replace("torch.", "")
                        hidden_files[order] = _PwriteFile(outs[order] / f"final_hidden.{hidden_dtype}le", rows_total * hidden_size * int(rows_t.element_size()))
                    hidden_files[order].write_at(first_row * hidden_size * int(rows_t.element_size()), _tensor_bytes(rows_t))
            for start, _count, chunks in engine.head_chunks(hidden_rows, head_rows):
                for order, chunk in chunks.items():
                    if logits_storage == "bf16":
                        if engine.head_dtype.get(order) != "bfloat16":
                            raise FixtureError(f"bf16 logits storage refused: head matmul dtype is {engine.head_dtype.get(order)}")
                        packed = chunk.to(torch.bfloat16)
                        if not bool(torch.equal(packed.float(), chunk)):
                            raise FixtureError("bf16 logits storage refused: forward logits do not round-trip exactly")
                        array = np.ascontiguousarray(packed.view(torch.int16).cpu().numpy())
                    else:
                        array = np.ascontiguousarray(chunk.cpu().numpy(), dtype="<f4")
                    for row in range(array.shape[0]):
                        logits_files[order].write_at(((first_row + row) * vocab + start) * logits_itemsize, array[row])
            pass_summaries.append({"pass": pass_index, "sequences": len(members), "input_tokens": sum(corpus.sequences[i].length for i in members), "scored_rows": int(next(iter(hidden_rows.values())).shape[0]), "wall_seconds": time.perf_counter() - pass_started, "peak_rss_bytes": _peak_rss_bytes()})
            del hidden_rows
        for recorder in recorders.values():
            recorder.verify(range(len(corpus.sequences)), engine.layer_count)
        stats = getattr(engine.checkpoint, "transfer_stats", None)
        coverage = getattr(engine.checkpoint, "coverage_report", None)
        transfer = {
            "transfer_stats": stats() if callable(stats) else None,
            "coverage_report": coverage() if callable(coverage) else None,
        }
        timing = _aggregate_timing(
            layer_records, embed_records, final_records, engine.head_stats,
            passes=len(passes), variants=len(orders), input_tokens=corpus.input_tokens, scored_rows=rows_total,
        )
        provenance = {order: engine.provenance(order) for order in orders}
        truncated, layers_executed = engine.truncated, engine.layer_count
        head_dtypes = dict(engine.head_dtype)
        final_counters = _checkpoint_counters(engine.checkpoint)
    finally:
        for item in (*logits_files.values(), *hidden_files.values()):
            item.close()
        for recorder in recorders.values():
            recorder.close()
        engine.close()
    timing["wall_seconds"] = time.perf_counter() - wall_start
    timing["pass_summaries"] = pass_summaries
    timing["layer_records"] = layer_records
    timing["checkpoint_counters"] = final_counters
    timing["store_stats"] = engine.store.snapshot()
    timing["projection_327680"] = project_runtime(timing, pass_tokens=pass_tokens)
    manifests: dict[str, dict[str, object]] = {}
    for order in orders:
        out, recorder = outs[order], recorders[order]
        files: dict[str, dict[str, object]] = {}

        def register(key: str, name: str, dtype: str, shape: Sequence[int], out: Path = out) -> None:
            path = out / name
            files[key] = {"path": name, "dtype": dtype, "shape": list(shape), "bytes": path.stat().st_size, "sha256": _file_sha256(path) if hash_files else None}

        register("score_index", "score_index.u32le", "uint32", (rows_total, 3))
        register("logits", logits_name, "float32" if logits_storage == "f32" else "bfloat16", (rows_total, vocab))
        if recorder.k is not None:
            register("route_ids", "route_ids.u16le", "uint16", (layers_executed, corpus.input_tokens, recorder.k))
            register("route_scores", "route_scores.f32le", "float32", (layers_executed, corpus.input_tokens, recorder.k))
        if order in hidden_files and hidden_dtype is not None:
            register("final_hidden", f"final_hidden.{hidden_dtype}le", hidden_dtype, (rows_total, hidden_size))
        identity = {
            "schema": EXACT_EXPORT_SCHEMA,
            "estimator_version": EXACT_VERSION,
            "corpus": corpus.identity(),
            "reduction_order": order,
            "source": provenance[order],
            "execution": {key: execution[key] for key in ("type", "name", "torch_version", "torch_hip", "torch_cuda", "gcn_arch")},
            "logits_semantics": "head matmul output widened to float32 (bf16 words stored losslessly when logits_storage=bf16); temperature 1; full vocabulary; no top-k/penalties",
            "logits_storage": logits_storage,
            "forward_head_dtype": head_dtypes.get(order),
            "hidden_dtype": hidden_dtype,
            "pass_tokens": pass_tokens,
            "head_rows": head_rows,
            "reset_semantics": "every sequence starts with fresh GDN/conv/PLE/attention state; no cross-sequence history",
            "vocab_size": vocab,
            "scored_rows": rows_total,
            "layers_executed": layers_executed,
            "truncated_layers": truncated,
            "route_top_k": recorder.k,
            "route_positions": "all input positions of every sequence (sequences concatenated in corpus order)",
            "files": files,
            "mtp": {"status": "unavailable", "reason": "native MTP adapter is not admitted"},
        }
        manifests[order] = {
            "identity": identity,
            "identity_sha256": _sha256_hex(_canonical_json(identity)),
            "runtime": {
                **dict(runtime or {}),
                "execution": execution,
                "expert_cache_bytes": expert_cache_bytes,
                "lockstep_variants": list(orders),
                "timing_shared_across_lockstep_variants": len(orders) > 1,
                "timing": timing,
                "transfer": transfer,
                "deterministic_verified": False,
            },
        }
        (out / "export.json").write_text(json.dumps(manifests[order], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifests


# --------------------------------------------------------------- compare


def _load_export(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    try:
        manifest = json.loads((root / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"{root}: invalid export manifest: {exc}") from exc
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema") != EXACT_EXPORT_SCHEMA:
        raise FixtureError(f"{root}: not a {EXACT_EXPORT_SCHEMA} export")
    if _sha256_hex(_canonical_json(identity)) != manifest.get("identity_sha256"):
        raise FixtureError(f"{root}: identity hash does not match identity block")
    return {"root": root, "manifest": manifest, "identity": identity}


def route_agreement(ids_a: Any, ids_b: Any) -> dict[str, Any]:
    """Per-input-position routing agreement averaged over layers.

    `ids_*` are [L,P,K] ordered expert IDs.  `exact` = fraction of layers whose ordered
    ID vector is identical; `overlap` = mean |A∩B|/K (set agreement).
    """
    import numpy as np

    a, b = np.asarray(ids_a), np.asarray(ids_b)
    if a.shape != b.shape or a.ndim != 3:
        raise FixtureError("route tapes have incompatible shapes")
    layers, positions, k = a.shape
    exact = np.zeros(positions)
    overlap = np.zeros(positions)
    for layer in range(layers):
        exact += (a[layer] == b[layer]).all(axis=1)
        for start in range(0, positions, 4096):
            stop = min(positions, start + 4096)
            same = (a[layer, start:stop, :, None] == b[layer, start:stop, None, :]).any(axis=2).sum(axis=1)
            overlap[start:stop] += same / k
    return {"exact": exact / layers, "overlap": overlap / layers}


def exact_compare(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    out_path: str | Path,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    block_rows: int = 64,
    block_vocab: int = 16384,
) -> dict[str, object]:
    """Exact teacher->candidate F64 KL over two exact exports sharing one corpus; writes report + per-position npz."""
    import numpy as np

    if bootstrap < 100:
        raise FixtureError("--bootstrap must be at least 100 replicates (2000 recommended)")
    ref, cand = _load_export(reference_dir), _load_export(candidate_dir)
    ri, ci = ref["identity"], cand["identity"]
    if ri.get("truncated_layers") or ci.get("truncated_layers"):
        raise FixtureError(
            "refusing to compare a truncated-layer export (--max-layers timing probe): a partial layer prefix is not BF16 teacher quality"
        )
    if ri["corpus"]["corpus_sha256"] != ci["corpus"]["corpus_sha256"]:
        raise FixtureError("exports use different corpora")
    if ri["files"]["score_index"]["sha256"] != ci["files"]["score_index"]["sha256"] or ri["scored_rows"] != ci["scored_rows"]:
        raise FixtureError("exports use different score indices")
    for key in ("vocab_size", "layers_executed"):
        if ri[key] != ci[key]:
            raise FixtureError(f"exports differ in {key}")
    corpus_identity = ri["corpus"]
    rows, vocab = int(ri["scored_rows"]), int(ri["vocab_size"])
    index = np.fromfile(ref["root"] / ri["files"]["score_index"]["path"], dtype="<u4").reshape(-1, 3)
    targets = index[:, 2].astype(np.int64)
    sources = []
    for side in (ref, cand):
        info = side["identity"]["files"]["logits"]
        storage = "bf16" if info["dtype"] == "bfloat16" else "f32"
        sources.append(_FileRows(side["root"] / info["path"], rows, vocab, info.get("sha256"), storage))
    scored = exact_kl_streamed(sources[0], sources[1], targets, block_rows=block_rows, block_vocab=block_vocab)
    sequences = [
        {"stratum": item["stratum"], "length": item["length"], "half": item.get("half")}
        for item in corpus_identity["sequences"]
    ]
    seq_index = index[:, 0].astype(np.int64)
    columns: dict[str, Any] = {
        "kl": scored["kl"],
        "nll_teacher": scored["nll_teacher"],
        "nll_candidate": scored["nll_candidate"],
        "nll_delta": scored["nll_candidate"] - scored["nll_teacher"],
        "top1_agreement": (scored["top1_teacher"] == scored["top1_candidate"]).astype(np.float64),
        "teacher_top1_is_target": (scored["top1_teacher"] == targets).astype(np.float64),
    }
    routing: dict[str, object] = {"status": "not_compared"}
    if "route_ids" in ri["files"] and "route_ids" in ci["files"]:
        shape = tuple(ri["files"]["route_ids"]["shape"])
        if shape != tuple(ci["files"]["route_ids"]["shape"]):
            raise FixtureError("route tapes have different shapes")
        a = np.fromfile(ref["root"] / ri["files"]["route_ids"]["path"], dtype="<u2").reshape(shape)
        b = np.fromfile(cand["root"] / ci["files"]["route_ids"]["path"], dtype="<u2").reshape(shape)
        agreement = route_agreement(a, b)
        offsets = np.concatenate([[0], np.cumsum([item["length"] for item in corpus_identity["sequences"]])[:-1]]).astype(np.int64)
        global_rows = offsets[seq_index] + index[:, 1].astype(np.int64)
        columns["route_exact_ordered"] = agreement["exact"][global_rows]
        columns["route_set_overlap"] = agreement["overlap"][global_rows]
        routing = {
            "status": "compared",
            "all_positions": {"exact_ordered": float(agreement["exact"].mean()), "set_overlap": float(agreement["overlap"].mean()), "positions": int(agreement["exact"].size)},
        }
    summary = summarize_columns(
        columns,
        seq_index,
        sequences,
        bootstrap=bootstrap,
        seed=seed,
        primary="kl",
        ppl_pairs={"teacher_ppl": "nll_teacher", "candidate_ppl": "nll_candidate"},
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    positions_path = out.with_suffix(".positions.npz")
    np.savez_compressed(
        positions_path,
        sequence_index=index[:, 0], position=index[:, 1], target=index[:, 2],
        **{name: np.asarray(values) for name, values in columns.items()},
        top1_teacher=scored["top1_teacher"], top1_candidate=scored["top1_candidate"],
        lse_teacher=scored["lse_teacher"], lse_candidate=scored["lse_candidate"],
    )
    report = {
        "schema": EXACT_COMPARE_SCHEMA,
        "estimator_version": EXACT_VERSION,
        "metric": "KL(p_teacher || p_candidate) in nats per scored next-token position",
        "direction": "teacher_to_candidate",
        "support": "full vocabulary",
        "temperature": 1.0,
        "arithmetic": "float64 stable logsumexp from actual forward float32 logits; no clipping",
        "negative_kl_tolerance": NEGATIVE_KL_TOLERANCE,
        "min_kl": float(scored["kl"].min()),
        "bootstrap": {"replicates": bootstrap, "seed": seed, "unit": "document", "independent_per_stratum": True},
        "reference": {"dir": str(reference_dir), "identity_sha256": ref["manifest"]["identity_sha256"], "reduction_order": ri["reduction_order"], "logits_sha256": scored["teacher_sha256"]},
        "candidate": {"dir": str(candidate_dir), "identity_sha256": cand["manifest"]["identity_sha256"], "reduction_order": ci["reduction_order"], "logits_sha256": scored["candidate_sha256"]},
        "corpus_sha256": corpus_identity["corpus_sha256"],
        "scored_rows": rows,
        "truncated_layers": False,
        "routing": routing,
        "results": summary,
        "per_position_file": str(positions_path),
        "not_additive": "KL is not additive: do not subtract a floor KL from a candidate KL",
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def paired_difference(
    positions_a: str | Path,
    positions_b: str | Path,
    export_dir: str | Path,
    out_path: str | Path,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Paired per-position `KL_A - KL_B` (e.g. ordinary minus forced) with joint document resampling.

    `export_dir` is any export of the shared corpus (supplies strata/lengths).
    """
    import numpy as np

    a, b = np.load(positions_a), np.load(positions_b)
    for key in ("sequence_index", "position", "target"):
        if not np.array_equal(a[key], b[key]):
            raise FixtureError(f"paired runs differ in {key}")
    export = _load_export(export_dir)
    sequences = [{"stratum": s["stratum"], "length": s["length"], "half": s.get("half")} for s in export["identity"]["corpus"]["sequences"]]
    columns = {"kl_diff": a["kl"] - b["kl"], "kl_a": a["kl"], "kl_b": b["kl"]}
    report = {
        "schema": EXACT_COMPARE_SCHEMA,
        "paired": {"a": str(positions_a), "b": str(positions_b), "statistic": "KL_A - KL_B"},
        "bootstrap": {"replicates": bootstrap, "seed": seed},
        "results": summarize_columns(columns, a["sequence_index"], sequences, bootstrap=bootstrap, seed=seed, primary="kl_diff"),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_floor(
    corpus: KlCorpus,
    out_dir: str | Path,
    *,
    self_sanity: bool = True,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    **export_kwargs: Any,
) -> dict[str, object]:
    """Lockstep native + reverse-experts export (one weight traversal), then compare native->reverse (+ native->native sanity)."""
    root = Path(out_dir)
    manifests = export_exact(corpus, root, orders=REDUCTION_ORDERS, **export_kwargs)
    exports = {order: str(root / order) for order in manifests}
    if any(manifest["identity"]["truncated_layers"] for manifest in manifests.values()):
        # --max-layers probe: timings/artifacts only. A partial layer prefix must never yield a quality number.
        return {
            "status": "truncated_timing_probe_no_quality",
            "exports": exports,
            "floor": None,
            "self_sanity": None,
            "reason": "truncated_layers export: floor/self-sanity comparison intentionally not run",
        }
    result: dict[str, object] = {
        "status": "complete",
        "exports": exports,
        "floor": exact_compare(root / "native", root / "reverse-experts", root / "floor_native_vs_reverse.json", bootstrap=bootstrap, seed=seed),
    }
    if self_sanity:
        result["self_sanity"] = exact_compare(root / "native", root / "native", root / "floor_self_native.json", bootstrap=bootstrap, seed=seed)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", action="store_true", help="run the genuine pinned layer-streamed source evaluator (canonical 17-token top-32 path)")
    parser.add_argument("--compare", action="store_true", help="compare two canonical top-32 quality artifacts")
    parser.add_argument("--exact-export", action="store_true", help="export full-vocab F32 logits + all-position routes for --corpus to --out-dir/<reduction-order>/")
    parser.add_argument("--exact-compare", action="store_true", help="exact F64 full-vocab KL between two --exact-export directories")
    parser.add_argument("--exact-floor", "--paired-floor", dest="exact_floor", action="store_true", help="native + reverse-experts in lockstep (one weight traversal), then compare them")
    parser.add_argument("--exact-paired", action="store_true", help="paired KL difference of two per-position npz files")
    parser.add_argument("--tokens", default=CANONICAL_TOKEN_METADATA, help="canonical corpus metadata JSON")
    parser.add_argument("--corpus", help="hipfire.qwen4.kl_corpus.v1 manifest, or 'canonical' for the 17-token corpus")
    parser.add_argument("--reference", help="reference artifact (--compare), export dir (--exact-compare / --exact-paired corpus identity)")
    parser.add_argument("--candidate", help="candidate artifact (--compare) or export dir (--exact-compare)")
    parser.add_argument("--positions-a", help="--exact-paired: first per-position npz")
    parser.add_argument("--positions-b", help="--exact-paired: second per-position npz")
    parser.add_argument("--out", help="quality artifact / report output path")
    parser.add_argument("--out-dir", help="export root (--exact-export, --exact-floor)")
    parser.add_argument("--cache-dir", help="bounded source/checkpoint cache")
    parser.add_argument("--local-dir", help="local source safetensors directory (shared upstream API)")
    parser.add_argument("--reduction-order", choices=REDUCTION_ORDERS, default="native")
    parser.add_argument("--pass-tokens", type=int, default=0, help="max input tokens per layer-major pass; 0 (default) = whole corpus in ONE pass (one weight traversal)")
    parser.add_argument("--head-rows", type=int, default=DEFAULT_HEAD_ROWS, help="lm_head vocabulary rows per streamed chunk")
    parser.add_argument("--expert-cache-gib", type=float, default=DEFAULT_EXPERT_CACHE_BYTES / (1 << 30), help="per-layer expert RAM cache cap (GiB)")
    parser.add_argument("--remote-byte-budget-gb", type=float, help="network byte budget forwarded to the shared reader (when supported)")
    parser.add_argument("--stream-disk-gb", type=float, help="ephemeral stream-cache disk budget forwarded to the shared reader (when supported)")
    parser.add_argument("--max-layers", type=int, help="run only the first N layers: timing/artifact probe only; the export is labeled truncated, --exact-floor then skips every comparison and --exact-compare refuses it")
    parser.add_argument("--export-hidden", action="store_true", help="also export final hidden rows")
    parser.add_argument("--no-hash-files", action="store_true", help="skip sha256 of exported files")
    parser.add_argument("--no-self-sanity", action="store_true", help="--exact-floor: skip native-vs-native zero check")
    parser.add_argument("--device", default="cpu", help="cpu (default; <=16 threads, <=64GiB AS, nice) or cuda[:N] (ROCm uses cuda); one expert layer on device at a time")
    parser.add_argument("--logits-storage", choices=("f32", "bf16"), default="f32", help="bf16 stores the forward's own bf16 logit words (refused unless exact round trip); halves logits size")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (CPU mode default 16, max 16)")
    parser.add_argument("--memory-limit-gib", type=float, default=None, help="RLIMIT_AS bound (CPU default/max 64; GPU default none, 0 disables)")
    parser.add_argument("--nice", type=int, default=None, help="niceness floor (CPU default 10)")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument("--block-vocab", type=int, default=16384)
    return parser


def _load_cli_corpus(args: argparse.Namespace) -> KlCorpus:
    if args.corpus and args.corpus != "canonical":
        return load_kl_corpus(args.corpus)
    tokens, _identity = _load_corpus(args.tokens)
    return canonical_kl_corpus(tokens)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    modes = [args.source_reference, args.compare, args.exact_export, args.exact_compare, args.exact_floor, args.exact_paired]
    try:
        if sum(bool(item) for item in modes) != 1:
            raise FixtureError("choose exactly one of --source-reference, --compare, --exact-export, --exact-compare, --exact-floor, --exact-paired")
        if args.compare or args.source_reference:
            if not args.out:
                raise FixtureError("--out is required")
            tokens, identity = _load_corpus(args.tokens)
            if args.compare:
                if not args.reference or not args.candidate:
                    raise FixtureError("--compare requires --reference and --candidate")
                compare(args.reference, args.candidate, args.out)
            else:
                result = _stream_source(tokens, args.cache_dir, args.local_dir)
                result["corpus"] = identity
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.exact_compare:
            if not (args.reference and args.candidate and args.out):
                raise FixtureError("--exact-compare requires --reference, --candidate and --out")
            exact_compare(args.reference, args.candidate, args.out, bootstrap=args.bootstrap, seed=args.seed, block_rows=args.block_rows, block_vocab=args.block_vocab)
        elif args.exact_paired:
            if not (args.positions_a and args.positions_b and args.reference and args.out):
                raise FixtureError("--exact-paired requires --positions-a, --positions-b, --reference (an export dir for corpus identity) and --out")
            paired_difference(args.positions_a, args.positions_b, args.reference, args.out, bootstrap=args.bootstrap, seed=args.seed)
        else:
            if not args.out_dir:
                raise FixtureError("--exact-export/--exact-floor require --out-dir")
            runtime = apply_runtime_limits(args.device, args.threads, args.memory_limit_gib, args.nice)
            corpus = _load_cli_corpus(args)
            options: dict[str, object] = {}
            if args.remote_byte_budget_gb is not None:
                options["remote_byte_budget"] = int(args.remote_byte_budget_gb * 1e9)
            if args.stream_disk_gb is not None:
                options["stream_disk_budget_bytes"] = int(args.stream_disk_gb * 1e9)
            torch = _import_torch()
            cpu_mode = str(args.device).split(":")[0] == "cpu"
            if args.threads is not None or cpu_mode:
                torch.set_num_threads(args.threads or MAX_THREADS)
            if cpu_mode and int(torch.get_num_threads()) > MAX_THREADS:
                raise FixtureError("torch thread cap exceeds 16")
            common = dict(
                cache_dir=args.cache_dir,
                local_dir=args.local_dir,
                pass_tokens=args.pass_tokens or None,
                head_rows=args.head_rows,
                expert_cache_bytes=int(args.expert_cache_gib * (1 << 30)),
                hash_files=not args.no_hash_files,
                export_hidden=args.export_hidden,
                max_layers=args.max_layers,
                checkpoint_options=options,
                runtime=runtime,
                device=args.device,
                logits_storage=args.logits_storage,
            )
            if args.exact_export:
                export_exact(corpus, args.out_dir, orders=(args.reduction_order,), torch=torch, **common)
            else:
                outcome = run_floor(corpus, args.out_dir, self_sanity=not args.no_self_sanity, bootstrap=args.bootstrap, seed=args.seed, torch=torch, **common)
                print(json.dumps({"status": outcome["status"], "exports": outcome["exports"]}, sort_keys=True))
    except (FixtureError, OSError, ValueError, RuntimeError, MemoryError) as exc:
        print(f"qwen4 quality failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
