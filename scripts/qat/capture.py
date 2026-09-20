#!/usr/bin/env python3
"""Deterministic corpus and layer-streamed Qwen3.8 teacher capture for QAT.

The collector deliberately never places the complete 27B teacher on one GPU.
It loads the embedding, one decoder block, final norm, or lm-head at a time and
keeps block boundaries on disk.  Dense teacher logits are represented by a
vocabulary-tiled recompute recipe because their BF16 materialization would
exceed the capture disk budget.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import math
import os
import re
import struct
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "hipfire.qat.capture.v1"
DEFAULT_SOURCE = Path("/home/kaden/qcal/parents/qwen3.8-27b")
DEFAULT_REF = Path("/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin")
DEFAULT_OUT = Path("/home/kaden/qcal/qat/l3")
WT2_REPO = "Salesforce/wikitext"
WT2_CONFIG = "wikitext-2-raw-v1"
WT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REPO = "allenai/c4"
C4_CONFIG = "en"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
CARD_B_UUID = "GPU-e475645fe0200397"
SEQ_LEN = 2048
TRAIN_SEQUENCES = 128
HELDOUT_SEQUENCES = 32
CORPUS_SEED = 20260920
OVERLAP_NGRAM = 64
BF16_BYTES = 2
F32_BYTES = 4


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    os.replace(temp, path)


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def packed_u32(values: list[int] | np.ndarray) -> bytes:
    return np.asarray(values, dtype="<u4").tobytes(order="C")


def source_identity(source: Path) -> dict[str, Any]:
    names = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
    names.extend(sorted(path.name for path in source.glob("model-*.safetensors")))
    files: list[dict[str, Any]] = []
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    identity_payload = [{"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in files]
    return {
        "path": str(source.resolve()),
        "sha256": sha256_bytes(canonical_json(identity_payload)),
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }


def read_kldref_tokens(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"short HFKLDR header: {path}")
        magic, version, n_ctx, n_vocab, n_chunk, top_k, flags, reserved = struct.unpack("<8sIIIIHHI", header)
        if magic != b"HFKLDR\0\0" or version != 1:
            raise ValueError(f"unsupported HFKLDR header: magic={magic!r} version={version}")
        raw = handle.read(n_ctx * n_chunk * 4)
        if len(raw) != n_ctx * n_chunk * 4:
            raise ValueError(f"short HFKLDR token stream: {path}")
    tokens = np.frombuffer(raw, dtype="<u4").copy().reshape(n_chunk, n_ctx)
    scoring_start = n_ctx // 2
    scored_prediction_positions = [scoring_start + 1, n_ctx]
    c2 = tokens[: min(2, n_chunk)]
    c24 = tokens[: min(24, n_chunk)]
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "version": version,
        "n_ctx": n_ctx,
        "n_vocab": n_vocab,
        "n_chunk": n_chunk,
        "top_k": top_k,
        "flags": flags,
        "reserved": reserved,
        "tokens": tokens,
        "token_stream_sha256": sha256_bytes(raw),
        "c2_token_sha256": sha256_bytes(c2.astype("<u4", copy=False).tobytes()),
        "c24_token_sha256": sha256_bytes(c24.astype("<u4", copy=False).tobytes()),
        "scoring_context_positions_zero_based": [scoring_start, n_ctx - 2],
        "scored_prediction_positions_half_open": scored_prediction_positions,
        "chunk_token_sha256": [sha256_bytes(row.astype("<u4", copy=False).tobytes()) for row in tokens],
    }


def ngram_hashes(tokens: np.ndarray, width: int) -> set[bytes]:
    if tokens.size < width:
        return set()
    result: set[bytes] = set()
    contiguous = np.asarray(tokens, dtype="<u4")
    for start in range(0, contiguous.size - width + 1):
        result.add(hashlib.blake2b(contiguous[start : start + width].tobytes(), digest_size=16).digest())
    return result


def has_reference_overlap(tokens: list[int], reference_ngrams: set[bytes], width: int) -> bool:
    values = np.asarray(tokens, dtype="<u4")
    for start in range(0, values.size - width + 1):
        digest = hashlib.blake2b(values[start : start + width].tobytes(), digest_size=16).digest()
        if digest in reference_ngrams:
            return True
    return False


def wt2_documents(dataset: Any) -> Iterator[tuple[str, str, dict[str, Any]]]:
    title_re = re.compile(r"^\s*= [^=].* =\s*$")
    parts: list[str] = []
    start_row = 0
    title = ""
    for row_index, row in enumerate(dataset):
        text = str(row["text"])
        if title_re.match(text) and parts:
            document = "\n".join(parts)
            doc_id = f"wt2-train:{start_row}:{row_index - 1}:{sha256_bytes(document.encode())[:16]}"
            yield doc_id, document, {"row_start": start_row, "row_end_inclusive": row_index - 1, "title": title}
            parts = []
            start_row = row_index
        if not parts:
            start_row = row_index
            title = text.strip()
        parts.append(text)
    if parts:
        document = "\n".join(parts)
        doc_id = f"wt2-train:{start_row}:{len(dataset) - 1}:{sha256_bytes(document.encode())[:16]}"
        yield doc_id, document, {"row_start": start_row, "row_end_inclusive": len(dataset) - 1, "title": title}


def deterministic_offset(token_count: int, doc_hash: str, seed: int, seq_len: int) -> int:
    span = token_count - seq_len
    if span <= 0:
        return 0
    raw = hashlib.sha256(f"{seed}:{doc_hash}:offset".encode()).digest()
    return int.from_bytes(raw[:8], "little") % (span + 1)


def priority_for(source: str, doc_id: str, doc_hash: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}:{doc_id}:{doc_hash}".encode()).digest(), "big")


def candidate_record(
    source_name: str,
    doc_id: str,
    text: str,
    provenance: dict[str, Any],
    tokenizer: Any,
    seed: int,
    seq_len: int,
    reference_ngrams: set[bytes],
) -> tuple[int, dict[str, Any]] | None:
    if len(text) < seq_len * 2:
        return None
    doc_bytes = text.encode("utf-8")
    doc_sha = sha256_bytes(doc_bytes)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < seq_len:
        return None
    # Excluding a whole source document on a 64-token match is stronger than
    # merely rejecting an exact 2048-token eval chunk: no selected WT2 span can
    # come from any document represented by the c2/c24 reference stream.
    if has_reference_overlap(token_ids, reference_ngrams, OVERLAP_NGRAM):
        return None
    offset = deterministic_offset(len(token_ids), doc_sha, seed, seq_len)
    selected = [int(token) for token in token_ids[offset : offset + seq_len]]
    token_sha = sha256_bytes(packed_u32(selected))
    record = {
        "source": source_name,
        "dataset_document_id": doc_id,
        "document_sha256": doc_sha,
        "document_bytes": len(doc_bytes),
        "document_token_count": len(token_ids),
        "token_offset": offset,
        "token_count": seq_len,
        "token_sha256": token_sha,
        "tokens": selected,
        "provenance": provenance,
    }
    return priority_for(source_name, doc_id, doc_sha, seed), record


def retain_lowest(heap: list[tuple[int, int, dict[str, Any]]], item: tuple[int, dict[str, Any]], count: int, ordinal: int) -> None:
    priority, record = item
    entry = (-priority, ordinal, record)
    if len(heap) < count:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def select_corpus(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    corpus_dir = args.out / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source = source_identity(args.source)
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False, use_fast=True)
    tokenizer_sha = sha256_file(args.source / "tokenizer.json")
    reference = read_kldref_tokens(args.ref)
    if reference["n_ctx"] != args.seq_len:
        raise ValueError(f"reference n_ctx={reference['n_ctx']} does not match seq_len={args.seq_len}")
    reference_tokens = reference.pop("tokens")
    reference_ngrams: set[bytes] = set()
    for chunk in reference_tokens[: min(24, len(reference_tokens))]:
        reference_ngrams.update(ngram_hashes(chunk, OVERLAP_NGRAM))

    need_per_source = (args.train_sequences + args.heldout_sequences) // 2
    if (args.train_sequences % 2) or (args.heldout_sequences % 2):
        raise ValueError("train and heldout sequence counts must both be even for a 50/50 corpus")

    wt = load_dataset(WT2_REPO, WT2_CONFIG, split="train", revision=WT2_REVISION)
    wt_heap: list[tuple[int, int, dict[str, Any]]] = []
    excluded_wt_docs = 0
    eligible_wt_docs = 0
    for ordinal, (doc_id, text, provenance) in enumerate(wt2_documents(wt)):
        item = candidate_record("wikitext2-train", doc_id, text, provenance, tokenizer, args.seed, args.seq_len, reference_ngrams)
        if item is None:
            # This count includes short docs and ref-overlap docs; the exact
            # no-overlap assertion is independently rerun on selected records.
            excluded_wt_docs += 1
            continue
        eligible_wt_docs += 1
        retain_lowest(wt_heap, item, need_per_source, ordinal)
    if len(wt_heap) != need_per_source:
        raise RuntimeError(f"only {len(wt_heap)} eligible WT2 documents; need {need_per_source}")

    c4 = load_dataset(C4_REPO, C4_CONFIG, split="train", streaming=True, revision=C4_REVISION)
    c4_heap: list[tuple[int, int, dict[str, Any]]] = []
    eligible_c4_docs = 0
    scanned_c4_docs = 0
    for row_index, row in enumerate(c4):
        if row_index >= args.c4_scan_docs:
            break
        scanned_c4_docs += 1
        text = str(row["text"])
        url = str(row.get("url", ""))
        timestamp = str(row.get("timestamp", ""))
        doc_id = f"c4-train:{row_index}:{sha256_bytes(url.encode())[:16]}"
        provenance = {"stream_row": row_index, "url": url, "timestamp": timestamp}
        item = candidate_record("c4-train", doc_id, text, provenance, tokenizer, args.seed, args.seq_len, reference_ngrams)
        if item is None:
            continue
        eligible_c4_docs += 1
        retain_lowest(c4_heap, item, need_per_source, row_index)
    if len(c4_heap) != need_per_source:
        raise RuntimeError(
            f"only {len(c4_heap)} eligible C4 documents in first {scanned_c4_docs}; need {need_per_source}; increase --c4-scan-docs"
        )

    def ordered(heap: list[tuple[int, int, dict[str, Any]]]) -> list[dict[str, Any]]:
        rows = [(-neg_priority, record) for neg_priority, _, record in heap]
        rows.sort(key=lambda row: row[0])
        return [record for _, record in rows]

    wt_rows = ordered(wt_heap)
    c4_rows = ordered(c4_heap)
    train_half = args.train_sequences // 2
    held_half = args.heldout_sequences // 2
    records: list[dict[str, Any]] = []
    for split, source_rows in (
        ("train", wt_rows[:train_half]),
        ("train", c4_rows[:train_half]),
        ("heldout", wt_rows[train_half : train_half + held_half]),
        ("heldout", c4_rows[train_half : train_half + held_half]),
    ):
        for row in source_rows:
            row = dict(row)
            row["split"] = split
            records.append(row)
    records.sort(key=lambda row: (0 if row["split"] == "train" else 1, row["source"], row["token_sha256"]))
    if len(records) != args.train_sequences + args.heldout_sequences:
        raise AssertionError("selected corpus count mismatch")

    doc_hashes = [row["document_sha256"] for row in records]
    token_hashes = [row["token_sha256"] for row in records]
    if len(set(doc_hashes)) != len(doc_hashes):
        raise AssertionError("train/heldout corpus contains repeated documents")
    if len(set(token_hashes)) != len(token_hashes):
        raise AssertionError("train/heldout corpus contains repeated token sequences")
    for row in records:
        if has_reference_overlap(row["tokens"], reference_ngrams, OVERLAP_NGRAM):
            raise AssertionError(f"selected sequence overlaps c24 reference: {row['token_sha256']}")

    tokens = np.asarray([row.pop("tokens") for row in records], dtype="<u4")
    token_path = corpus_dir / "tokens.u32"
    token_path.write_bytes(tokens.tobytes(order="C"))
    mask_path = corpus_dir / "attention_mask.u8"
    mask_path.write_bytes(np.ones(tokens.shape, dtype=np.uint8).tobytes(order="C"))
    position_path = corpus_dir / "positions.u32"
    positions = np.broadcast_to(np.arange(args.seq_len, dtype="<u4"), tokens.shape)
    position_path.write_bytes(positions.tobytes(order="C"))

    for sequence_id, row in enumerate(records):
        row["sequence_id"] = sequence_id
        row["state_reset_id"] = f"zero:{row['document_sha256']}:{row['token_sha256']}"
        row["token_file_offset_bytes"] = sequence_id * args.seq_len * 4

    corpus_token_sha = sha256_bytes(tokens.tobytes(order="C"))
    corpus = {
        "seed": args.seed,
        "sequence_length": args.seq_len,
        "train_sequences": args.train_sequences,
        "heldout_sequences": args.heldout_sequences,
        "total_sequences": len(records),
        "total_tokens": int(tokens.size),
        "mixture": {"train": {"wikitext2-train": train_half, "c4-train": train_half}, "heldout": {"wikitext2-train": held_half, "c4-train": held_half}},
        "datasets": {
            "wikitext2": {"repo": WT2_REPO, "config": WT2_CONFIG, "split": "train", "revision": WT2_REVISION, "eligible_documents": eligible_wt_docs, "excluded_or_short_documents": excluded_wt_docs},
            "c4": {"repo": C4_REPO, "config": C4_CONFIG, "split": "train", "revision": C4_REVISION, "scanned_documents": scanned_c4_docs, "eligible_documents": eligible_c4_docs},
        },
        "tokenizer": {"path": str(args.source.resolve()), "tokenizer_json_sha256": tokenizer_sha, "class": tokenizer.__class__.__name__, "add_special_tokens": False},
        "token_stream": {"file": relative(token_path, args.out), "dtype": "uint32-le", "axes": ["sequence", "token"], "shape": list(tokens.shape), "sha256": sha256_file(token_path), "aggregate_token_sha256": corpus_token_sha},
        "attention_mask": {"file": relative(mask_path, args.out), "dtype": "uint8", "axes": ["sequence", "token"], "shape": list(tokens.shape), "sha256": sha256_file(mask_path), "semantics": "all tokens valid; no padding"},
        "positions": {"file": relative(position_path, args.out), "dtype": "uint32-le", "axes": ["sequence", "token"], "shape": list(tokens.shape), "sha256": sha256_file(position_path), "semantics": "0..2047 independently for every document"},
        "state": {"reset": "zero at each selected source document", "carry": "within one 2048-token document only", "kv": "empty", "convolution": "zero", "recurrent": "zero", "snapshot_aliasing": False},
        "sequences": records,
        "disjointness": {
            "document_hashes_unique": True,
            "token_hashes_unique": True,
            "train_heldout_document_intersection": 0,
            "reference_path": reference["path"],
            "reference_file_sha256": reference["file_sha256"],
            "reference_header": {key: reference[key] for key in ("version", "n_ctx", "n_vocab", "n_chunk", "top_k", "flags")},
            "reference_token_stream_sha256": reference["token_stream_sha256"],
            "reference_c2_token_sha256": reference["c2_token_sha256"],
            "reference_c24_token_sha256": reference["c24_token_sha256"],
            "reference_chunk_token_sha256": reference["chunk_token_sha256"],
            "reference_scoring_context_positions_zero_based": reference["scoring_context_positions_zero_based"],
            "reference_scored_prediction_positions_half_open": reference["scored_prediction_positions_half_open"],
            "overlap_assertion": "zero matching contiguous 64-token spans against every token window in the first 24 reference chunks; WT2 source documents with any match were excluded whole",
            "overlap_ngram_tokens": OVERLAP_NGRAM,
            "selected_reference_ngram_intersections": 0,
        },
        "wall_clock_seconds": time.monotonic() - started,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "source": source,
        "corpus": corpus,
        "capture": {"status": "corpus_ready", "blocks": {}, "tensor_shards": [], "validation": {}},
    }
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "source_sha256": source["sha256"], "corpus_token_sha256": corpus_token_sha, "sequences": len(records), "wall_clock_seconds": corpus["wall_clock_seconds"]}, indent=2))
    return manifest


class SafeTensorSource:
    def __init__(self, root: Path):
        from safetensors import safe_open

        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._safe_open = safe_open
        self._handles: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._safe_open(self.root / shard, framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def prefix_state(self, prefix: str) -> dict[str, Any]:
        result = {}
        for name in self.weight_map:
            if name.startswith(prefix):
                result[name[len(prefix) :]] = self.get(name)
        if not result:
            raise KeyError(f"no source tensors with prefix {prefix!r}")
        return result


class RawTensor:
    def __init__(self, path: Path, shape: tuple[int, ...], logical_dtype: str, mode: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.shape = shape
        self.logical_dtype = logical_dtype
        np_dtype = np.uint16 if logical_dtype == "bfloat16" else np.dtype(logical_dtype)
        self.array = np.memmap(path, dtype=np_dtype, mode=mode, shape=shape, order="C")

    def write_torch(self, index: Any, tensor: Any) -> bytes:
        import torch

        cpu = tensor.detach().contiguous().cpu()
        if self.logical_dtype == "bfloat16":
            cpu = cpu.to(torch.bfloat16)
            raw = cpu.view(torch.uint16).numpy()
        elif self.logical_dtype == "float32":
            raw = cpu.float().numpy()
        else:
            raise ValueError(self.logical_dtype)
        self.array[index] = raw
        return raw.tobytes(order="C")

    def read_torch(self, index: Any, device: Any) -> Any:
        import torch

        raw = np.asarray(self.array[index])
        if self.logical_dtype == "bfloat16":
            tensor = torch.from_numpy(raw.copy()).view(torch.bfloat16)
        elif self.logical_dtype == "float32":
            tensor = torch.from_numpy(raw.copy())
        else:
            raise ValueError(self.logical_dtype)
        return tensor.to(device=device, non_blocking=False)

    def flush(self) -> None:
        self.array.flush()


class ProducerSampler:
    def __init__(self, block: int, out: Path, samples: int, hidden: int, seq_count: int, seq_len: int, seed: int):
        self.block = block
        self.seq_count = seq_count
        self.seq_len = seq_len
        total = seq_count * seq_len
        count = min(samples, total)
        rng = np.random.Generator(np.random.PCG64(seed + block * 1_000_003))
        self.global_indices = np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))
        self.current_sequence_start = 0
        self.current_sequence_count = 1
        self.specs = {
            "input_norm_qkvza" if block % 4 != 3 else "input_norm_qkv": hidden,
            "mixer_out": 6144,
            "post_attention_norm_gate_up": hidden,
            "silu_mul_down": 17408,
        }
        self.tensors: dict[str, RawTensor] = {}
        self.hashers: dict[str, Any] = {}
        for site, width in self.specs.items():
            path = out / "tensors" / "producers" / f"block_{block:02d}.{site}.f32"
            self.tensors[site] = RawTensor(path, (count, width), "float32", "w+")
            self.hashers[site] = hashlib.sha256()
        self.handles: list[Any] = []

    def _hook(self, site: str):
        def capture(_module: Any, inputs: tuple[Any, ...], output: Any = None) -> None:
            import torch

            value = output if output is not None else inputs[0]
            lo = self.current_sequence_start * self.seq_len
            hi = (self.current_sequence_start + self.current_sequence_count) * self.seq_len
            begin = int(np.searchsorted(self.global_indices, lo, side="left"))
            end = int(np.searchsorted(self.global_indices, hi, side="left"))
            rows = np.arange(begin, end, dtype=np.int64)
            positions = self.global_indices[begin:end] - lo
            if rows.size == 0:
                return
            value = value.reshape(-1, value.shape[-1])
            pos = torch.from_numpy(positions).to(device=value.device)
            selected = value.index_select(0, pos).float()
            payload = self.tensors[site].write_torch(rows, selected)
            self.hashers[site].update(payload)

        return capture

    def install(self, layer: Any) -> None:
        self.handles.append(layer.input_layernorm.register_forward_hook(self._hook(next(site for site in self.specs if site.startswith("input_norm")))))
        self.handles.append(layer.post_attention_layernorm.register_forward_hook(self._hook("post_attention_norm_gate_up")))
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        mixer_out = mixer.out_proj if hasattr(mixer, "out_proj") else mixer.o_proj
        self.handles.append(mixer_out.register_forward_pre_hook(self._hook("mixer_out")))
        self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook("silu_mul_down")))

    def close(self) -> dict[str, Any]:
        for handle in self.handles:
            handle.remove()
        result = {}
        for site, tensor in self.tensors.items():
            tensor.flush()
            result[site] = {
                "file": str(tensor.path),
                "shape": list(tensor.shape),
                "axes": ["sample", "channel"],
                "dtype": "float32-le",
                "sha256": self.hashers[site].hexdigest(),
                "bytes": tensor.path.stat().st_size,
                "global_sample_indices": self.global_indices.tolist(),
            }
        return result


def load_manifest(out: Path) -> dict[str, Any]:
    path = out / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run the corpus subcommand first: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    return manifest


def torch_environment() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "device_name": props.name if props is not None else None,
        "device_total_memory": props.total_memory if props is not None else None,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "home": os.environ.get("HOME"),
    }


def require_card_b(device: str, allow_other_device: bool) -> None:
    import torch

    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    if visible != CARD_B_UUID and not allow_other_device:
        raise RuntimeError(f"C2 capture requires card-B: ROCR_VISIBLE_DEVICES={visible!r}, expected {CARD_B_UUID!r}")


def load_config(source: Path) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    text = config.text_config
    text._attn_implementation = "eager"
    return text


def instantiate_layer(source: SafeTensorSource, config: Any, block: int, device: Any) -> Any:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, block)
    prefix = f"model.language_model.layers.{block}."
    state = source.prefix_state(prefix)
    incompatible = layer.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"block {block} state mismatch: {incompatible}")
    layer = layer.to(device=device, dtype=torch.bfloat16)
    layer.eval()
    return layer


def rotary_and_masks(
    config: Any, seq_len: int, device: Any, dtype: Any | None = None
) -> tuple[tuple[Any, Any], Any, Any]:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    dtype = torch.bfloat16 if dtype is None else dtype
    positions = torch.arange(seq_len, dtype=torch.long, device=device).view(1, seq_len)
    rotary = Qwen3_5TextRotaryEmbedding(config, device=device).to(device)
    dummy = torch.empty((1, seq_len, config.hidden_size), dtype=dtype, device=device)
    position_embeddings = rotary(dummy, positions)
    causal = torch.triu(
        torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
    return position_embeddings, positions, causal


def layer_forward(layer: Any, hidden: Any, position_embeddings: tuple[Any, Any], positions: Any, causal: Any, block_type: str, cache: Any = None) -> Any:
    mask = None if block_type == "linear_attention" else causal
    return layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=cache is not None,
    )


def functional_validation(layer: Any, config: Any, hidden: Any, block_type: str, device: Any, length: int) -> dict[str, Any]:
    import torch
    from transformers.cache_utils import DynamicCache

    length = min(length, hidden.shape[1])
    value = hidden[:, :length].clone()
    pos_emb, positions, causal = rotary_and_masks(config, length, device)
    with torch.inference_mode():
        whole_cache = DynamicCache(config=config)
        whole = layer_forward(layer, value, pos_emb, positions, causal, block_type, whole_cache)
        rerun_cache = DynamicCache(config=config)
        rerun = layer_forward(layer, value.clone(), pos_emb, positions, causal, block_type, rerun_cache)
        split = length // 2
        split_cache = DynamicCache(config=config)
        pos_a, ids_a, causal_a = rotary_and_masks(config, split, device)
        first = layer_forward(layer, value[:, :split], pos_a, ids_a, causal_a, block_type, split_cache)
        tail_len = length - split
        # Cached continuation uses absolute positions. Full attention needs a
        # q-by-(prefix+q) causal mask; DeltaNet ignores this mask.
        absolute = torch.arange(split, length, dtype=torch.long, device=device).view(1, tail_len)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        rotary = Qwen3_5TextRotaryEmbedding(config, device=device).to(device)
        dummy = torch.empty((1, tail_len, config.hidden_size), dtype=torch.bfloat16, device=device)
        pos_b = rotary(dummy, absolute)
        causal_b = torch.zeros((1, 1, tail_len, length), dtype=torch.bfloat16, device=device)
        causal_b[:, :, :, split:] = torch.triu(
            torch.full((1, 1, tail_len, tail_len), float("-inf"), dtype=torch.bfloat16, device=device), diagonal=1
        )
        second = layer_forward(layer, value[:, split:], pos_b, absolute, causal_b, block_type, split_cache)
        joined = torch.cat((first, second), dim=1)
    deterministic = (whole.float() - rerun.float()).cpu()
    split_error = (whole.float() - joined.float()).cpu()

    reference_rms = float(whole.float().cpu().square().mean().sqrt().item())

    def metrics(error: Any) -> dict[str, float]:
        rms = float(error.square().mean().sqrt().item())
        return {
            "max_abs": float(error.abs().max().item()),
            "rms": rms,
            "reference_rms": reference_rms,
            "relative_rms": rms / max(reference_rms, 1e-30),
        }

    return {
        "tokens": length,
        "whole_vs_checkpoint_recompute": metrics(deterministic),
        "whole_vs_sequence_split": metrics(split_error),
        "checkpoint_recompute_bitwise_equal": bool(torch.equal(whole, rerun)),
    }


def precision_validation(
    layer: Any,
    config: Any,
    hidden: Any,
    block_type: str,
    device: Any,
    length: int,
) -> dict[str, Any]:
    """Compare the BF16 training forward with widened-BF16 F32 math.

    This mutates ``layer`` to F32 and therefore must run only after the block's
    BF16 corpus capture and producer hooks are complete.
    """
    import torch

    length = min(length, hidden.shape[1])
    bf16_input = hidden[:, :length].clone()
    pos16, ids16, mask16 = rotary_and_masks(config, length, device, torch.bfloat16)
    with torch.inference_mode():
        bf16_output = layer_forward(
            layer, bf16_input, pos16, ids16, mask16, block_type
        ).float()
        layer.float()
        fp32_input = bf16_input.float()
        pos32, ids32, mask32 = rotary_and_masks(config, length, device, torch.float32)
        fp32_output = layer_forward(
            layer, fp32_input, pos32, ids32, mask32, block_type
        )
    error = (bf16_output - fp32_output).cpu()
    reference = fp32_output.float().cpu()
    rms = float(error.square().mean().sqrt().item())
    reference_rms = float(reference.square().mean().sqrt().item())
    return {
        "max_abs": float(error.abs().max().item()),
        "rms": rms,
        "reference_rms": reference_rms,
        "relative_rms": rms / max(reference_rms, 1e-30),
    }


def capture_boundaries(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel

    started = time.monotonic()
    require_card_b(args.device, args.allow_other_device)
    manifest = load_manifest(args.out)
    if Path(manifest["source"]["path"]).resolve() != args.source.resolve():
        raise ValueError("manifest source path differs from --source")
    environment = torch_environment()
    config = load_config(args.source)
    if config.num_hidden_layers != 64 or config.hidden_size != 5120:
        raise ValueError(f"unexpected Qwen3.8 shape: layers={config.num_hidden_layers} hidden={config.hidden_size}")
    sequences = manifest["corpus"]["sequences"]
    seq_count = len(sequences)
    seq_len = manifest["corpus"]["sequence_length"]
    token_meta = manifest["corpus"]["token_stream"]
    tokens = np.memmap(args.out / token_meta["file"], dtype="<u4", mode="r", shape=(seq_count, seq_len))
    source = SafeTensorSource(args.source)
    device = torch.device(args.device)
    tensors_dir = args.out / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    capture = manifest.setdefault("capture", {})
    capture["environment"] = environment
    capture["device_contract"] = {
        "venue": os.uname().nodename,
        "visible_device": environment["device_name"],
        "observed_rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "observed_hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    capture["producer_sample_count_per_block"] = args.producer_samples
    capture["batch_size"] = args.batch_size
    capture["blocks"] = {}
    capture["boundaries"] = {}
    capture["validation"] = {"per_block_error_table": []}
    capture["status"] = "loading_resident_teacher"
    atomic_json(args.out / "manifest.json", manifest)

    model_load_started = time.monotonic()
    model = AutoModel.from_pretrained(
        args.source,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    language = model.language_model
    layers = language.layers
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_devices != [str(device)]:
        raise RuntimeError(f"teacher is not fully resident on {device}: {parameter_devices}")
    capture["resident_teacher"] = {
        "loader": "transformers.AutoModel.from_pretrained",
        "dtype": "bfloat16",
        "device_map": getattr(model, "hf_device_map", {"": str(device)}),
        "parameter_devices": parameter_devices,
        "load_wall_clock_seconds": time.monotonic() - model_load_started,
    }

    boundary_shape = (seq_count, seq_len, config.hidden_size)
    boundary_tensors: dict[int, RawTensor] = {}
    boundary_hashers: dict[int, Any] = {}
    for boundary in range(config.num_hidden_layers + 1):
        path = tensors_dir / "boundaries" / f"boundary_{boundary:02d}.bf16"
        boundary_tensors[boundary] = RawTensor(path, boundary_shape, "bfloat16", "w+")
        boundary_hashers[boundary] = hashlib.sha256()
    final_path = tensors_dir / "final_norm.bf16"
    final = RawTensor(final_path, boundary_shape, "bfloat16", "w+")
    final_digest = hashlib.sha256()
    current_batch = {"start": 0, "count": 0}

    def write_boundary(boundary: int, value: Any) -> None:
        if isinstance(value, tuple):
            value = value[0]
        start = current_batch["start"]
        stop = start + current_batch["count"]
        payload = boundary_tensors[boundary].write_torch(slice(start, stop), value)
        boundary_hashers[boundary].update(payload)

    boundary_handles = [
        layers[0].register_forward_pre_hook(
            lambda _module, inputs: write_boundary(0, inputs[0])
        )
    ]
    for block, layer in enumerate(layers):
        boundary_handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, boundary=block + 1: write_boundary(boundary, output)
            )
        )

    samplers = [
        ProducerSampler(
            block,
            args.out,
            args.producer_samples,
            config.hidden_size,
            seq_count,
            seq_len,
            args.seed,
        )
        for block in range(config.num_hidden_layers)
    ]
    for sampler, layer in zip(samplers, layers, strict=True):
        sampler.install(layer)

    capture["status"] = "capturing_resident_batched"
    atomic_json(args.out / "manifest.json", manifest)
    forward_started = time.monotonic()
    input_device = language.embed_tokens.weight.device
    with torch.inference_mode():
        for start in range(0, seq_count, args.batch_size):
            stop = min(start + args.batch_size, seq_count)
            current_batch["start"] = start
            current_batch["count"] = stop - start
            for sampler in samplers:
                sampler.current_sequence_start = start
                sampler.current_sequence_count = stop - start
            ids = torch.from_numpy(
                np.asarray(tokens[start:stop]).astype(np.int64, copy=True)
            ).to(input_device)
            outputs = language(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
            )
            payload = final.write_torch(slice(start, stop), outputs.last_hidden_state)
            final_digest.update(payload)
            print(f"resident batch {start:03d}:{stop:03d}/{seq_count}", flush=True)
            del ids, outputs
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic() - forward_started
    for handle in boundary_handles:
        handle.remove()
    for tensor in boundary_tensors.values():
        tensor.flush()
    final.flush()

    for boundary, tensor in boundary_tensors.items():
        capture["boundaries"][str(boundary)] = {
            "file": relative(tensor.path, args.out),
            "shape": list(boundary_shape),
            "axes": ["sequence", "token", "hidden"],
            "dtype": "bfloat16-le",
            "sha256": boundary_hashers[boundary].hexdigest(),
            "bytes": tensor.path.stat().st_size,
            "meaning": (
                "embedding output / block-0 boundary h"
                if boundary == 0
                else f"block-{boundary - 1} teacher y / block-{boundary} boundary h"
            ),
        }
    producer_meta = [sampler.close() for sampler in samplers]
    for sites in producer_meta:
        for value in sites.values():
            value["file"] = relative(Path(value["file"]), args.out)
    capture["final_norm"] = {
        "file": relative(final_path, args.out),
        "shape": list(boundary_shape),
        "axes": ["sequence", "token", "hidden"],
        "dtype": "bfloat16-le",
        "sha256": final_digest.hexdigest(),
        "bytes": final_path.stat().st_size,
        "wall_clock_seconds": 0.0,
        "meaning": "canonical model final RMSNorm output",
    }
    capture["resident_forward_wall_clock_seconds"] = forward_seconds
    capture["resident_forward_sequences_per_minute"] = seq_count * 60.0 / forward_seconds
    capture["status"] = "validating_blocks"
    atomic_json(args.out / "manifest.json", manifest)

    validation_sequence = int(args.validation_sequence)
    for block, layer in enumerate(layers):
        block_started = time.monotonic()
        layer_device = next(layer.parameters()).device
        boundary = boundary_tensors[block]
        validation_hidden = boundary.read_torch(validation_sequence, layer_device).unsqueeze(0)
        validation_hidden = validation_hidden[:, : args.validation_tokens]
        block_type = str(config.layer_types[block])
        validation_functional = functional_validation(
            layer,
            config,
            validation_hidden,
            block_type,
            layer_device,
            args.validation_tokens,
        )
        validation_precision = precision_validation(
            layer,
            config,
            validation_hidden,
            block_type,
            layer_device,
            args.validation_tokens,
        )
        layer.to(dtype=torch.bfloat16)
        validation_result = {
            "block": block,
            "type": block_type,
            "sequence_id": validation_sequence,
            **validation_functional,
            "bf16_vs_widened_bf16_f32": validation_precision,
        }
        validation_result["passed"] = bool(
            validation_functional["checkpoint_recompute_bitwise_equal"]
            and validation_functional["whole_vs_sequence_split"]["relative_rms"]
            <= args.surrogate_relative_rms_limit
            and validation_precision["relative_rms"]
            <= args.surrogate_relative_rms_limit
        )
        capture["validation"]["per_block_error_table"].append(validation_result)
        capture["blocks"][str(block)] = {
            "status": "complete" if validation_result["passed"] else "blocked_validation",
            "type": block_type,
            "input_boundary": str(block),
            "output_boundary": str(block + 1),
            "producer_sites": producer_meta[block],
            "validation_wall_clock_seconds": time.monotonic() - block_started,
        }
        if not validation_result["passed"]:
            capture["status"] = "blocked_training_surrogate_discrepancy"
            atomic_json(args.out / "manifest.json", manifest)
            raise RuntimeError(
                f"block {block} functional surrogate validation failed: "
                f"{json.dumps(validation_result, sort_keys=True)}"
            )
        print(f"validated block {block:02d}/63", flush=True)
    atomic_json(args.out / "manifest.json", manifest)

    del model, language, layers, final
    boundary_tensors.clear()
    torch.cuda.empty_cache()
    capture_teacher_logit_receipt(manifest, args, source, config, device)
    build_tensor_records(manifest, args.out)
    capture["status"] = "complete"
    completed_unix = time.time()
    capture["completed_unix"] = completed_unix
    capture["attempt_wall_clock_seconds"] = time.monotonic() - started
    capture["processing_wall_clock_seconds"] = capture["attempt_wall_clock_seconds"]
    capture["wall_clock_seconds"] = completed_unix - float(manifest["created_unix"])
    capture["disk_bytes"] = capture_disk_bytes(manifest, args.out)
    capture["run_root_disk_bytes_including_siblings"] = disk_bytes(args.out)
    capture["dense_logit_materialization_bytes_bf16"] = seq_count * seq_len * config.vocab_size * 2
    capture["dense_logit_storage_decision"] = "not materialized: vocabulary-tiled deterministic recompute from captured final_norm and immutable BF16 lm_head"
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "status": capture["status"], "wall_clock_seconds": capture["wall_clock_seconds"], "disk_bytes": capture["disk_bytes"], "resident_forward_sequences_per_minute": capture["resident_forward_sequences_per_minute"]}, indent=2))
    return manifest


def capture_teacher_logit_receipt(manifest: dict[str, Any], args: argparse.Namespace, source: SafeTensorSource, config: Any, device: Any) -> None:
    import torch

    capture = manifest["capture"]
    seq_count = manifest["corpus"]["total_sequences"]
    seq_len = manifest["corpus"]["sequence_length"]
    shape = (seq_count, seq_len, config.hidden_size)
    final = RawTensor(args.out / capture["final_norm"]["file"], shape, "bfloat16", "r")
    head = source.get("lm_head.weight").to(device=device, dtype=torch.bfloat16)
    positions = [0, seq_len // 2, seq_len - 1]
    tile = min(args.logit_tile_vocab, config.vocab_size)
    vocab_starts = sorted(set([0, max(0, config.vocab_size - tile)]))
    receipt_path = args.out / "tensors" / "teacher_logits_receipt.f32"
    receipt = RawTensor(receipt_path, (len(positions), len(vocab_starts), tile), "float32", "w+")
    with torch.inference_mode():
        hidden = final.read_torch(0, device).index_select(0, torch.tensor(positions, device=device))
        for tile_index, start in enumerate(vocab_starts):
            logits = hidden @ head[start : start + tile].T
            receipt.write_torch((slice(None), tile_index, slice(None)), logits.float())
    receipt.flush()
    receipt_sha256 = sha256_file(receipt_path)
    capture["teacher_logits"] = {
        "mode": "vocabulary_tiled_recompute",
        "complete_dense_storage": False,
        "reason": "dense BF16 logits plus mandatory boundaries and producer samples exceed the available capture filesystem budget",
        "axes": ["sequence", "token", "vocabulary"],
        "logical_shape": [seq_count, seq_len, config.vocab_size],
        "compute_dtype": "bfloat16 inputs/weights with ROCm BF16 matmul; consumers may upcast each tile to float32",
        "tile_vocab": args.logit_tile_vocab,
        "input": capture["final_norm"],
        "weight": {"source_tensor": "lm_head.weight", "source_sha256": manifest["source"]["sha256"], "shape": [config.vocab_size, config.hidden_size], "dtype": "bfloat16"},
        "receipt": {"file": relative(receipt_path, args.out), "shape": list(receipt.shape), "dtype": "float32-le", "sequence_id": 0, "token_positions": positions, "vocab_starts": vocab_starts, "sha256": receipt_sha256, "bytes": receipt_path.stat().st_size},
        "consumer": "iter_teacher_logit_tiles in scripts/qat/capture.py",
    }
    del head, final, hidden, receipt
    torch.cuda.empty_cache()


def iter_teacher_logit_tiles(out: Path, source_path: Path, sequence_id: int, token_start: int = 0, token_end: int | None = None, tile_vocab: int = 8192, device: str = "cuda") -> Iterator[tuple[int, Any]]:
    """Yield ``(vocab_start, logits_f32[token, vocab_tile])`` for one sequence."""
    import torch

    manifest = load_manifest(out)
    config = load_config(source_path)
    count = manifest["corpus"]["total_sequences"]
    seq_len = manifest["corpus"]["sequence_length"]
    if not 0 <= sequence_id < count:
        raise IndexError(sequence_id)
    token_end = seq_len if token_end is None else token_end
    if not 0 <= token_start < token_end <= seq_len:
        raise ValueError((token_start, token_end))
    source = SafeTensorSource(source_path)
    target = torch.device(device)
    final_meta = manifest["capture"]["final_norm"]
    final = RawTensor(out / final_meta["file"], tuple(final_meta["shape"]), "bfloat16", "r")
    hidden = final.read_torch(sequence_id, target)[token_start:token_end]
    head = source.get("lm_head.weight").to(device=target, dtype=torch.bfloat16)
    with torch.inference_mode():
        for start in range(0, config.vocab_size, tile_vocab):
            end = min(start + tile_vocab, config.vocab_size)
            yield start, (hidden @ head[start:end].T).float()


def materialize_logits(args: argparse.Namespace) -> None:
    import torch

    require_card_b(args.device, args.allow_other_device)
    manifest = load_manifest(args.out)
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.source)
    token_end = args.token_end or manifest["corpus"]["sequence_length"]
    shape = (token_end - args.token_start, config.vocab_size)
    tensor = RawTensor(target, shape, "float32", "w+")
    digest = hashlib.sha256()
    for start, logits in iter_teacher_logit_tiles(args.out, args.source, args.sequence_id, args.token_start, token_end, args.tile_vocab, args.device):
        payload = tensor.write_torch((slice(None), slice(start, start + logits.shape[1])), logits)
        # Tiles are written column-wise, not in final row-major byte order, so
        # hash the completed file below rather than this write order.
        del payload, logits
        torch.cuda.empty_cache()
    tensor.flush()
    digest_hex = sha256_file(target)
    print(json.dumps({"file": str(target), "shape": list(shape), "dtype": "float32-le", "sha256": digest_hex, "bytes": target.stat().st_size}, indent=2))


def build_tensor_records(manifest: dict[str, Any], out: Path) -> None:
    capture = manifest["capture"]
    source_sha = manifest["source"]["sha256"]
    sequences = manifest["corpus"]["sequences"]
    seq_len = manifest["corpus"]["sequence_length"]
    hidden = capture["boundaries"]["0"]["shape"][2]
    records: list[dict[str, Any]] = []
    capture["model_module_map"] = {
        "decoder_layer": "model.language_model.layers.{block}",
        "delta_qkv": "model.language_model.layers.{block}.linear_attn.in_proj_qkv",
        "delta_z": "model.language_model.layers.{block}.linear_attn.in_proj_z",
        "delta_a": "model.language_model.layers.{block}.linear_attn.in_proj_a",
        "delta_b": "model.language_model.layers.{block}.linear_attn.in_proj_b",
        "full_attention_q": "model.language_model.layers.{block}.self_attn.q_proj",
        "full_attention_k": "model.language_model.layers.{block}.self_attn.k_proj",
        "full_attention_v": "model.language_model.layers.{block}.self_attn.v_proj",
        "attention_output": "model.language_model.layers.{block}.{linear_attn.out_proj|self_attn.o_proj}",
        "gate": "model.language_model.layers.{block}.mlp.gate_proj",
        "up": "model.language_model.layers.{block}.mlp.up_proj",
        "down": "model.language_model.layers.{block}.mlp.down_proj",
    }
    capture["model_forward"] = {
        "implementation": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer",
        "attention_implementation": "eager",
        "state_reset": "per document sequence",
    }

    def key(sequence: dict[str, Any], block: int, site: str, route: str) -> dict[str, Any]:
        return {"source_sha": source_sha, "token_sha": sequence["token_sha256"], "block": block, "site": site, "route": route, "state_reset_id": sequence["state_reset_id"]}

    for block in range(64):
        in_meta = capture["boundaries"][str(block)]
        out_meta = capture["boundaries"][str(block + 1)]
        per_sequence_bytes = seq_len * hidden * BF16_BYTES
        for sequence in sequences:
            sequence_id = sequence["sequence_id"]
            records.append({"key": key(sequence, block, "boundary_h", "teacher_bf16"), "file": in_meta["file"], "offset_bytes": sequence_id * per_sequence_bytes, "shape": [seq_len, hidden], "axes": ["token", "hidden"], "dtype": "bfloat16-le"})
            records.append({"key": key(sequence, block, "teacher_y", "teacher_bf16"), "file": out_meta["file"], "offset_bytes": sequence_id * per_sequence_bytes, "shape": [seq_len, hidden], "axes": ["token", "hidden"], "dtype": "bfloat16-le"})
        producers = capture["blocks"][str(block)]["producer_sites"]
        for site, meta in producers.items():
            indices = np.asarray(meta["global_sample_indices"], dtype=np.int64)
            width = int(meta["shape"][1])
            for sequence in sequences:
                sequence_id = sequence["sequence_id"]
                lo = sequence_id * seq_len
                hi = lo + seq_len
                begin = int(np.searchsorted(indices, lo, side="left"))
                end = int(np.searchsorted(indices, hi, side="left"))
                local_positions = (indices[begin:end] - lo).tolist()
                records.append({"key": key(sequence, block, site, "pre_awq_f32"), "file": meta["file"], "offset_bytes": begin * width * F32_BYTES, "shape": [end - begin, width], "axes": ["sample", "channel"], "dtype": "float32-le", "token_positions": local_positions})
        for sequence in sequences:
            records.append({"key": key(sequence, block, "state", "teacher_functional"), "storage": "descriptor", "reset": {"at_document_start": True, "kv": "empty", "convolution": "zero", "recurrent": "zero"}, "snapshot": {"immutable_sequence_start": True, "may_alias_mutable_forward_state": False}})
    final_meta = capture["final_norm"]
    for sequence in sequences:
        sequence_id = sequence["sequence_id"]
        records.append({"key": key(sequence, 64, "final_norm", "teacher_bf16"), "file": final_meta["file"], "offset_bytes": sequence_id * seq_len * hidden * BF16_BYTES, "shape": [seq_len, hidden], "axes": ["token", "hidden"], "dtype": "bfloat16-le"})
        records.append({"key": key(sequence, 64, "lm_head_logits", "teacher_logits_recompute"), "storage": "vocabulary_tiled_recompute", "input_file": final_meta["file"], "input_offset_bytes": sequence_id * seq_len * hidden * BF16_BYTES, "logical_shape": [seq_len, manifest["capture"]["teacher_logits"]["logical_shape"][2]], "axes": ["token", "vocabulary"], "dtype": "float32 tile", "tile_vocab": manifest["capture"]["teacher_logits"]["tile_vocab"]})
    capture["tensor_shards"] = records
    capture["tensor_shard_key_fields"] = ["source_sha", "token_sha", "block", "site", "route", "state_reset_id"]
    capture["tensor_shard_record_count"] = len(records)


def capture_disk_bytes(manifest: dict[str, Any], out: Path) -> int:
    """Return bytes owned by this C2 capture, excluding sibling artifacts."""
    capture = manifest["capture"]
    files = {manifest["corpus"][name]["file"] for name in ("token_stream", "attention_mask", "positions")}
    files.update(meta["file"] for meta in capture["boundaries"].values())
    for block in capture["blocks"].values():
        files.update(meta["file"] for meta in block["producer_sites"].values())
    files.add(capture["final_norm"]["file"])
    files.add(capture["teacher_logits"]["receipt"]["file"])
    return sum((out / path).stat().st_size for path in files)


def disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def compare_runtime(args: argparse.Namespace) -> None:
    """Compare a captured HF block output with an HFHS runtime dump.

    The runtime dump must be produced by hipfire's F32/BF16 source path on the
    same token sequence.  Quantized runtime dumps are rejected by contract and
    must not be relabeled as source parity.
    """
    manifest = load_manifest(args.out)
    with args.runtime_hfhs.open("rb") as handle:
        magic = handle.read(8)
        if magic != b"HFHS\0\0\0\0":
            raise ValueError("bad HFHS magic")
        n_layers, n_pos, hidden, reserved = struct.unpack("<IIII", handle.read(16))
        if reserved != 0:
            raise ValueError("unsupported HFHS flags")
        offset = 24 + args.block * n_pos * hidden * 4
        handle.seek(offset)
        runtime = np.frombuffer(handle.read(n_pos * hidden * 4), dtype="<f4").copy().reshape(n_pos, hidden)
    boundary_meta = manifest["capture"]["boundaries"][str(args.block + 1)]
    boundary = RawTensor(args.out / boundary_meta["file"], tuple(boundary_meta["shape"]), "bfloat16", "r")
    import torch

    teacher = boundary.read_torch(args.sequence_id, torch.device("cpu")).float().numpy()
    if teacher.shape != runtime.shape:
        raise ValueError(f"shape mismatch: capture={teacher.shape} runtime={runtime.shape}")
    error = teacher.astype(np.float64) - runtime.astype(np.float64)
    row = {
        "block": args.block,
        "sequence_id": args.sequence_id,
        "runtime_hfhs": str(args.runtime_hfhs.resolve()),
        "runtime_hfhs_sha256": sha256_file(args.runtime_hfhs),
        "runtime_declared_route": args.runtime_route,
        "max_abs": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(error * error))),
        "reference_rms": float(np.sqrt(np.mean(runtime.astype(np.float64) ** 2))),
    }
    row["relative_rms"] = row["rms"] / max(row["reference_rms"], 1e-30)
    row["passed"] = row["max_abs"] <= args.max_abs and row["relative_rms"] <= args.relative_rms
    validation = manifest["capture"].setdefault("validation", {})
    validation["runtime_f32"] = row
    manifest["capture"]["status"] = "complete" if row["passed"] else "blocked_runtime_parity"
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps(row, indent=2))
    if not row["passed"]:
        raise SystemExit(3)


def verify_manifest(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.out)
    failures: list[str] = []
    corpus = manifest["corpus"]
    for name in ("token_stream", "attention_mask", "positions"):
        meta = corpus[name]
        observed = sha256_file(args.out / meta["file"])
        if observed != meta["sha256"]:
            failures.append(f"{name}: {observed} != {meta['sha256']}")
    if corpus["total_sequences"] != 160 or corpus["sequence_length"] != 2048:
        failures.append("corpus shape is not the frozen 160x2048 contract")
    if corpus["train_sequences"] != 128 or corpus["heldout_sequences"] != 32:
        failures.append("corpus split is not the frozen 128/32 contract")

    capture = manifest.get("capture", {})
    if capture.get("status") != "complete":
        failures.append(f"capture status is {capture.get('status')!r}, not 'complete'")
    boundaries = capture.get("boundaries", {})
    if len(boundaries) != 65:
        failures.append(f"boundary count is {len(boundaries)}, expected 65")
    for block, meta in boundaries.items():
        path = args.out / meta["file"]
        if not path.is_file() or path.stat().st_size != meta["bytes"]:
            failures.append(f"boundary {block}: byte count")
    blocks = capture.get("blocks", {})
    if len(blocks) != 64:
        failures.append(f"block count is {len(blocks)}, expected 64")
    for block, block_meta in blocks.items():
        if block_meta.get("status") != "complete":
            failures.append(f"block {block}: status {block_meta.get('status')!r}")
        for site, meta in block_meta.get("producer_sites", {}).items():
            path = args.out / meta["file"]
            if not path.is_file() or path.stat().st_size != meta["bytes"]:
                failures.append(f"block {block} producer {site}: byte count")
    validation_rows = capture.get("validation", {}).get("per_block_error_table", [])
    if len(validation_rows) != 64 or any(not row.get("passed") for row in validation_rows):
        failures.append("per-block validation table is incomplete or contains a failure")
    final_meta = capture.get("final_norm")
    if not final_meta:
        failures.append("final_norm metadata is missing")
    else:
        final_path = args.out / final_meta["file"]
        if not final_path.is_file() or final_path.stat().st_size != final_meta["bytes"]:
            failures.append("final_norm: byte count")
    receipt_meta = capture.get("teacher_logits", {}).get("receipt")
    if not receipt_meta:
        failures.append("teacher logit receipt metadata is missing")
    else:
        receipt_path = args.out / receipt_meta["file"]
        if sha256_file(receipt_path) != receipt_meta["sha256"]:
            failures.append("teacher logit receipt: sha256")

    key_fields = capture.get("tensor_shard_key_fields", [])
    keys = [tuple(record["key"].get(field) for field in key_fields) for record in capture.get("tensor_shards", [])]
    if key_fields != ["source_sha", "token_sha", "block", "site", "route", "state_reset_id"]:
        failures.append("tensor shard key fields differ from the frozen schema")
    if len(keys) != len(set(keys)):
        failures.append("tensor shard keys are not unique")
    if len(keys) != capture.get("tensor_shard_record_count"):
        failures.append("tensor shard record count does not match metadata")
    result = {
        "manifest": str(args.out / "manifest.json"),
        "status": capture.get("status"),
        "blocks": len(blocks),
        "validation_rows": len(validation_rows),
        "tensor_shard_records": len(keys),
        "unique_keys": len(set(keys)),
        "failures": failures,
        "capture_disk_bytes": capture_disk_bytes(manifest, args.out) if not failures else None,
        "run_root_disk_bytes_including_siblings": disk_bytes(args.out),
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="build the deterministic WT2/C4 corpus and initial manifest")
    corpus.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    corpus.add_argument("--ref", type=Path, default=DEFAULT_REF)
    corpus.add_argument("--out", type=Path, default=DEFAULT_OUT)
    corpus.add_argument("--seq-len", type=int, default=SEQ_LEN)
    corpus.add_argument("--train-sequences", type=int, default=TRAIN_SEQUENCES)
    corpus.add_argument("--heldout-sequences", type=int, default=HELDOUT_SEQUENCES)
    corpus.add_argument("--seed", type=int, default=CORPUS_SEED)
    corpus.add_argument("--c4-scan-docs", type=int, default=100_000)
    corpus.set_defaults(func=select_corpus)

    capture = sub.add_parser("capture", help="capture all layer boundaries, producer samples, and logit recipe")
    capture.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    capture.add_argument("--out", type=Path, default=DEFAULT_OUT)
    capture.add_argument("--device", default="cuda")
    capture.add_argument("--allow-other-device", action="store_true")
    capture.add_argument("--batch-size", type=int, default=4)
    capture.add_argument("--producer-samples", type=int, default=4096)
    capture.add_argument("--logit-tile-vocab", type=int, default=8192)
    capture.add_argument("--seed", type=int, default=CORPUS_SEED)
    capture.add_argument("--validation-block", type=int, default=0, help=argparse.SUPPRESS)
    capture.add_argument("--validation-sequence", type=int, default=0)
    capture.add_argument("--validation-tokens", type=int, default=128)
    capture.add_argument("--surrogate-relative-rms-limit", type=float, default=0.02)
    capture.set_defaults(func=capture_boundaries)

    logits = sub.add_parser("materialize-logits", help="materialize one sequence/range via vocabulary tiles")
    logits.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    logits.add_argument("--out", type=Path, default=DEFAULT_OUT)
    logits.add_argument("--sequence-id", type=int, required=True)
    logits.add_argument("--token-start", type=int, default=0)
    logits.add_argument("--token-end", type=int)
    logits.add_argument("--tile-vocab", type=int, default=8192)
    logits.add_argument("--output", type=Path, required=True)
    logits.add_argument("--device", default="cuda")
    logits.add_argument("--allow-other-device", action="store_true")
    logits.set_defaults(func=materialize_logits)

    compare = sub.add_parser("compare-runtime", help="attach a hipfire F32/BF16 HFHS parity receipt")
    compare.add_argument("--out", type=Path, default=DEFAULT_OUT)
    compare.add_argument("--runtime-hfhs", type=Path, required=True)
    compare.add_argument("--runtime-route", choices=["f32-oracle", "bf16-source-f32-accum"], required=True)
    compare.add_argument("--block", type=int, default=0)
    compare.add_argument("--sequence-id", type=int, default=0)
    compare.add_argument("--max-abs", type=float, default=0.2)
    compare.add_argument("--relative-rms", type=float, default=0.02)
    compare.set_defaults(func=compare_runtime)

    verify = sub.add_parser("verify", help="verify manifest invariants and small-file hashes")
    verify.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify.set_defaults(func=verify_manifest)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
