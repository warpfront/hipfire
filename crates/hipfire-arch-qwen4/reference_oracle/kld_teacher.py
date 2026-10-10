# SPDX-License-Identifier: Apache-2.0
"""Source-precision Qwen4 KLD teacher over a local copy of the pinned checkpoint.

The engine cannot run Qwen4 at source precision: routed experts are admitted
only as MQ4, and BF16 experts (~245 GB) exceed host memory anyway. This
teacher runs the pinned upstream operators (`upstream.PinnedQwen4Operators`)
with BF16 weights and F32 activations, layer-major over every chunk of an
existing HFKLDR v3 reference, so each checkpoint byte is read once per run.

It writes only raw F32 logits for the reference's scored window, laid out
`[chunk][scored position][vocab]`; `qwen4_kld import` owns top-k, teacher
NLL, the plausibility gate and the HFKLDR format, exactly as for the engine
teacher.

Usage (ROCm/CUDA torch):
  python3 kld_teacher.py --checkpoint DIR --tokens-from REF.kldref \
      --output TEACHER.logits [--max-chunks N] [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import upstream
    from .quality import PLE_SHARD_ROWS, PLE_VALID_ROWS, _load_hc_weights
    from .schema import FixtureError
except ImportError:  # direct execution from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import upstream  # type: ignore
    from quality import PLE_SHARD_ROWS, PLE_VALID_ROWS, _load_hc_weights  # type: ignore
    from schema import FixtureError  # type: ignore

_NP_DTYPES = {"BF16": np.int16, "F32": np.float32, "I64": np.int64, "I32": np.int32}


class LocalCheckpoint:
    """The pinned safetensors checkpoint on a filesystem path, read by row range.

    Refuses a directory whose index/config differ from the pinned revision.
    """

    def __init__(self, root: Path):
        self.root = root
        index_raw = (root / "model.safetensors.index.json").read_bytes()
        config_raw = (root / "config.json").read_bytes()
        for label, raw, pinned in (
            ("safetensors index", index_raw, upstream.MODEL_INDEX_SHA256),
            ("config", config_raw, upstream.MODEL_CONFIG_SHA256),
        ):
            if hashlib.sha256(raw).hexdigest() != pinned:
                raise FixtureError(f"{root}: {label} is not the pinned {upstream.HF_CONFIG_COMMIT} revision")
        self.weight_map: dict[str, str] = json.loads(index_raw)["weight_map"]
        self.config = json.loads(config_raw)
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def _entry(self, name: str) -> tuple[Path, int, Any, list[int]]:
        file = self.weight_map[name]
        if file not in self._headers:
            with open(self.root / file, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self._headers[file] = (8 + length, json.loads(handle.read(length)))
        data_start, header = self._headers[file]
        entry = header[name]
        return self.root / file, data_start + entry["data_offsets"][0], _NP_DTYPES[entry["dtype"]], entry["shape"]

    def rows(self, name: str, row_start: int, row_count: int) -> tuple[np.ndarray, str]:
        path, offset, dtype, shape = self._entry(name)
        row_items = math.prod(shape[1:])
        if row_start < 0 or row_start + row_count > shape[0]:
            raise FixtureError(f"{name}: rows [{row_start}, {row_start + row_count}) outside {shape[0]}")
        start = offset + row_start * row_items * np.dtype(dtype).itemsize
        array = np.fromfile(path, dtype=dtype, count=row_count * row_items, offset=start)
        return array.reshape([row_count, *shape[1:]]), dtype

    def scan_rows(self, name: str, local_rows: np.ndarray) -> np.ndarray:
        """Sorted rows of one tensor in a single sequential pass. The NAS
        backing the checkpoint serves ~250 random reads/s, so scattered rows
        (PLE n-gram lookups) are far cheaper to stream than to seek."""
        shape = self._entry(name)[3]
        block = 1 << 18
        out = np.empty((local_rows.size, *shape[1:]), dtype=self._entry(name)[2])
        for start in range(0, shape[0], block):
            lo, hi = np.searchsorted(local_rows, [start, start + block])
            if lo < hi:
                data, _ = self.rows(name, start, min(block, shape[0] - start))
                out[lo:hi] = data[local_rows[lo:hi] - start]
        return out


class DeviceStore:
    """Torch view of `LocalCheckpoint` with per-layer caching.

    The pinned operators re-request the same tensors (per chunk, per routed
    expert); caching the current layer turns that into one read per byte.
    """

    def __init__(self, checkpoint: LocalCheckpoint, torch: Any, device: Any):
        self.checkpoint = checkpoint
        self.torch = torch
        self.device = device
        self._layer: dict[str, Any] = {}
        self._ple_ids: np.ndarray | None = None  # None: discovery pass
        self._ple_wanted: set[int] = set()
        self._ple_table: np.ndarray | None = None

    def begin_layer(self) -> None:
        self._layer.clear()

    def _tensor(self, array: np.ndarray, dtype: Any) -> Any:
        value = self.torch.from_numpy(array).to(self.device)
        if dtype is np.int16:
            value = value.view(self.torch.bfloat16)
        if value.is_floating_point() and not bool(self.torch.isfinite(value).all()):
            raise FixtureError("nonfinite checkpoint tensor")
        return value

    def rows(self, name: str, row_start: int, row_count: int) -> Any:
        return self._tensor(*self.checkpoint.rows(name, row_start, row_count))

    def full(self, name: str) -> Any:
        if name not in self._layer:
            shape = self.checkpoint._entry(name)[3]
            self._layer[name] = self.rows(name, 0, shape[0])
        return self._layer[name]

    def expert_rows(self, name: str, expert: int, row_start: int, row_count: int) -> Any:
        return self.full(name)[expert, row_start : row_start + row_count].float()

    def ple_rows(self, global_rows: list[int]) -> Any:
        """PLE n-gram rows. Until `load_ple_rows`, records the request and
        returns zeros (discovery pass); afterwards serves the fetched rows."""
        rows = np.asarray(global_rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= PLE_VALID_ROWS):
            raise FixtureError("PLE row outside the valid source range")
        if self._ple_ids is None:
            self._ple_wanted.update(rows.tolist())
            width = self.checkpoint._entry(_ple_shard(0))[3][1]
            return self.torch.zeros((rows.size, width), device=self.device)
        return self._tensor(self._ple_table[np.searchsorted(self._ple_ids, rows)], np.int16).float()

    def load_ple_rows(self) -> int:
        """Fetch every row the discovery pass requested, one scan per shard."""
        wanted = np.array(sorted(self._ple_wanted), dtype=np.int64)
        shards = wanted // PLE_SHARD_ROWS
        table = np.empty((wanted.size, self.checkpoint._entry(_ple_shard(0))[3][1]), dtype=np.int16)
        for shard in np.unique(shards):
            picked = shards == shard
            table[picked] = self.checkpoint.scan_rows(_ple_shard(int(shard)), wanted[picked] % PLE_SHARD_ROWS)
        self._ple_ids, self._ple_table = wanted, table
        return wanted.size


def _ple_shard(shard: int) -> str:
    return f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"


def read_kldref_tokens(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Header and `[n_chunk, n_ctx]` token block of an HFKLDR v3 reference."""
    raw = path.open("rb").read(16)
    if raw[:8] != b"HFKLDR\0\0" or struct.unpack_from("<I", raw, 8)[0] != 3:
        raise FixtureError(f"{path}: not an HFKLDR v3 reference")
    header_len = struct.unpack_from("<I", raw, 12)[0]
    with path.open("rb") as handle:
        handle.seek(16)
        header = json.loads(handle.read(header_len))
        tokens = np.fromfile(handle, dtype="<u4", count=header["n_chunk"] * header["n_ctx"])
    return header, tokens.reshape(header["n_chunk"], header["n_ctx"])


def run(checkpoint_dir: Path, tokens_from: Path, output: Path, max_chunks: int | None, device_name: str) -> None:
    import torch

    header, tokens = read_kldref_tokens(tokens_from)
    tokens = tokens[: max_chunks or len(tokens)]
    n_chunk, n_ctx = tokens.shape
    window = header["window"]
    if window["carry_kv"]:
        raise FixtureError("teacher runs every chunk from a fresh state; carry_kv references are unsupported")
    score_from, score_to = window["score_from"], window["score_to"]

    device = torch.device(device_name)
    checkpoint = LocalCheckpoint(checkpoint_dir)
    source_text, source_provenance, _ = upstream.load_pinned_sources(None)
    ops = upstream.load_pinned_operators(source_text, source_provenance, torch)
    cfg = checkpoint.config.get("text_config", checkpoint.config)
    vocab = int(cfg["vocab_size"])
    if vocab != header["n_vocab"]:
        raise FixtureError(f"reference vocab {header['n_vocab']} != checkpoint vocab {vocab}")
    store = DeviceStore(checkpoint, torch, device)
    started = time.time()

    def spans():
        return [slice(c * n_ctx, (c + 1) * n_ctx) for c in range(n_chunk)]

    def log(message: str) -> None:
        print(f"kld_teacher: {message}  {time.time() - started:.0f}s", file=sys.stderr, flush=True)

    torch.set_default_device(device)
    with torch.inference_mode():
        flat = tokens.reshape(-1)
        embed = store.full("model.language_model.embed_tokens.weight")
        wide = embed[torch.from_numpy(flat.astype(np.int64)).to(device)].float().repeat(1, int(cfg["hc_count"]))
        del embed
        ple_metadata = {
            key: store.full(f"model.language_model.layers.1.ple.ple_embedding.{key}").reshape(-1)
            for key in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
        }
        for layer in range(int(cfg["num_hidden_layers"])):
            store.begin_layer()
            prefix = f"model.language_model.layers.{layer}"
            if layer == 1:
                def ple():
                    return torch.cat([
                        ops.ple(wide[s], flat[s].tolist(), cfg, ple_metadata, store.full, store.ple_rows)
                        for s in spans()
                    ])

                ple()  # discovery pass: records the n-gram rows the corpus touches
                log(f"PLE: fetched {store.load_ple_rows()} unique rows")
                wide = wide + ple()
            mixed, hyper_input, injection = ops.hyperconnection(
                wide, _load_hc_weights(store, f"{prefix}.attn_hyper_connection", combine=True), cfg, combine=True
            )
            if str(cfg["layer_types"][layer]) in ("qwen_sparse_attention", "full_attention"):
                a = f"{prefix}.self_attn"
                weights = {
                    "index_qk": store.full(f"{a}.indexer.index_qk_proj.weight"),
                    "index_q_norm": store.full(f"{a}.indexer.q_layernorm.weight"),
                    "index_k_norm": store.full(f"{a}.indexer.k_layernorm.weight"),
                    "q": store.full(f"{a}.q_proj.weight"),
                    "q_norm": store.full(f"{a}.q_norm.weight"),
                    "k": store.full(f"{a}.k_proj.weight"),
                    "k_norm": store.full(f"{a}.k_norm.weight"),
                    "v": store.full(f"{a}.v_proj.weight"),
                    "out": store.full(f"{a}.o_proj.weight"),
                }
                block = torch.cat([ops.qsa(mixed[s], weights, cfg, layer)[0] for s in spans()])
            else:
                g = f"{prefix}.linear_attn"
                a_log = f"{g}.A_log.weight" if f"{g}.A_log.weight" in checkpoint.weight_map else f"{g}.A_log"
                weights = {
                    "qkv": store.full(f"{g}.in_proj_qkv.weight"),
                    "z": store.full(f"{g}.in_proj_z.weight"),
                    "b": store.full(f"{g}.in_proj_b.weight"),
                    "a": store.full(f"{g}.in_proj_a.weight"),
                    "conv": store.full(f"{g}.conv1d.weight"),
                    "a_log": store.full(a_log),
                    "dt_bias": store.full(f"{g}.dt_bias"),
                    "norm": store.full(f"{g}.norm.weight"),
                    "out": store.full(f"{g}.out_proj.weight"),
                }
                block = torch.cat([ops.gdn(mixed[s], weights, cfg, layer)[0] for s in spans()])
            wide = ops.inject(hyper_input, block, injection)
            mlp_input, mlp_hyper, mlp_injection = ops.hyperconnection(
                wide, _load_hc_weights(store, f"{prefix}.mlp_hyper_connection", combine=True), cfg, combine=True
            )
            moe = ops.moe(mlp_input, cfg, store.full, store.expert_rows, layer)
            wide = ops.inject(mlp_hyper, moe, mlp_injection)
            del mixed, hyper_input, injection, block, mlp_input, mlp_hyper, mlp_injection, moe
            log(f"layer {layer + 1}/{cfg['num_hidden_layers']} {cfg['layer_types'][layer]}")
        store.begin_layer()
        hidden = ops.hyperconnection(
            wide, _load_hc_weights(store, "model.language_model.hyper_connection_mixer", combine=False), cfg, combine=False
        )
        del wide
        head = store.full("lm_head.weight").float()
        nll_sum, scored = 0.0, 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as out:
            for c, s in enumerate(spans()):
                logits = hidden[s][score_from:score_to] @ head.T
                if not bool(torch.isfinite(logits).all()):
                    raise FixtureError(f"nonfinite teacher logits in chunk {c}")
                targets = torch.from_numpy(tokens[c, score_from + 1 : score_to + 1].astype(np.int64)).to(device)
                lse = torch.logsumexp(logits.double(), dim=-1)
                nll_sum += float((lse - logits.double().gather(1, targets[:, None])[:, 0]).sum())
                scored += score_to - score_from
                out.write(logits.cpu().numpy().tobytes())
        log(f"wrote {output} ({n_chunk} chunks x {score_to - score_from} rows x {vocab}); teacher PPL {math.exp(nll_sum / scored):.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        run(args.checkpoint, args.tokens_from, args.output, args.max_chunks, args.device)
    except (FixtureError, OSError, ValueError, KeyError) as exc:
        print(f"kld_teacher failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
