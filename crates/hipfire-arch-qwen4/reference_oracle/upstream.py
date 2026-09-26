#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned Transformers/vLLM parity backend for the Qwen4 oracle.

This module is deliberately an opt-in backend.  The dependency-free synthetic
path remains in :mod:`generate_fixtures`; this backend executes the exact
commit-pinned source files through an explicit adapter and requires only
PyTorch.  Only bounded raw source files, safetensors headers, and small tensor
ranges are cached.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib
import json
import random
import math
import os
import platform
import re
import errno
import socket
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .equations import (
        EOS_TOKEN_ID,
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_PADDED_ROWS,
        NG_SCALE,
        NG_VALID_ROWS,
        bf16_to_f32,
        bf16_tensor,
        gdn_expand_qk,
        gdn_depthwise_conv,
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
        rng_normal,
        rng_uniform,
        rope_half_split,
        tensor,
        transpose2,
    )
    from .schema import FixtureError, Tensor, array_metadata, read_npz, sha256_file, write_json, write_npz
except ImportError:  # direct execution from this directory
    from equations import (  # type: ignore
        EOS_TOKEN_ID,
        NG_HEAD_OFFSETS,
        NG_HEAD_VOCAB_SIZES,
        NG_PADDED_ROWS,
        NG_SCALE,
        NG_VALID_ROWS,
        bf16_to_f32,
        bf16_tensor,
        gdn_expand_qk,
        gdn_depthwise_conv,
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
        rng_normal,
        rng_uniform,
        rope_half_split,
        tensor,
        transpose2,
    )
    from schema import FixtureError, Tensor, array_metadata, read_npz, sha256_file, write_json, write_npz  # type: ignore


HF_TRANSFORMERS_COMMIT = "93c8b7b485963a10800c91f55304db6be211c2bd"
HF_TRANSFORMERS_VERSION = "5.16.1"
VLLM_COMMIT = "e126687a9a828d513c01a07cd69f025f27d63280"
HF_CONFIG_COMMIT = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
HF_MODEL = "Qwen/Qwen3.8-Flash-Next"

# Hashes are intentionally checked in.  A URL containing a commit is not by
# itself sufficient provenance: a proxy, mirror, or cache must not be able to
# silently substitute a different source file.
UPSTREAM_SOURCE_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "transformers_modeling",
        "project": "huggingface/transformers",
        "commit": HF_TRANSFORMERS_COMMIT,
        "path": "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py",
        "sha256": "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459",
    },
    {
        "name": "transformers_modular",
        "project": "huggingface/transformers",
        "commit": HF_TRANSFORMERS_COMMIT,
        "path": "src/transformers/models/qwen4_exp/modular_qwen4_exp.py",
        "sha256": "54b9e147c1e1b95169419a4258c0ea4cefeaf7c37b9af2456e57ebc92a4ecc56",
    },
    {
        "name": "transformers_configuration",
        "project": "huggingface/transformers",
        "commit": HF_TRANSFORMERS_COMMIT,
        "path": "src/transformers/models/qwen4_exp/configuration_qwen4_exp.py",
        "sha256": "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
    },
    {
        "name": "vllm_mtp",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/amd/mtp.py",
        "sha256": "4216bde1d9b7cac1d1dbd8ef114d401887edd727ff85252cc08bf8da576b6006",
    },
    {
        "name": "vllm_model",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/amd/model.py",
        "sha256": "e9ee63d7921d1fba937a3674320225a1f914e668d5a2f91dbde11dfc70e2214e",
    },
    {
        "name": "vllm_hyperconnection",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/amd/hyperconnection.py",
        "sha256": "29e15a7a9ba9d6f186a0b5613d388605b6680ef4bdaf37228057925e44de479c",
    },
    {
        "name": "vllm_indexer_qsa",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/amd/indexer_qsa.py",
        "sha256": "3429bec22e9d1045b3651fd28757384f527efcd071b4f7e35c95476996ef671e",
    },
    {
        "name": "vllm_common_hyperconnection",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/common/hyperconnection.py",
        "sha256": "cf2028bbd7dceb33f78393be62e3c936c29aff7255dc185f211d1e57cb007393",
    },
    {
        "name": "vllm_common_ple",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/common/ple.py",
        "sha256": "4266838d1b39a86c06e63654e80a256dc8515ad4aa99b648d875323b4d7119e0",
    },
    {
        "name": "vllm_common_qsa_cache",
        "project": "vllm-project/vllm",
        "commit": VLLM_COMMIT,
        "path": "vllm/models/qwen4_exp/common/qsa_cache.py",
        "sha256": "9dca1c8d577e636edd3027231f3792b4b26976e2a75cf3875a832dd12d78151a",
    },
)

MODEL_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
MODEL_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
MAX_SOURCE_BYTES = 8 << 20
MAX_HEADER_BYTES = 8 << 20
MAX_TENSOR_SLICE_BYTES = 128 << 10
RANGE_MAX_RETRIES = 4
RANGE_RETRY_BACKOFF_SECONDS = 0.25
RANGE_RETRY_BACKOFF_MAX_SECONDS = 1.0

# Names are intentionally explicit instead of selecting the first lexical
# match.  This prevents a model-index reorder from changing the fixture.
CHECKPOINT_TENSORS: dict[str, tuple[str, ...]] = {
    "ple": (
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
    ),
    "hc": (
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    ),
    "gdn": (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
    ),
    "qsa": (
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
        "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.v_proj.weight",
    ),
    "moe": (
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ),
    "mtp": (
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    ),
}

PLE_EMBEDDING = CHECKPOINT_TENSORS["ple"][2]

TOLERANCES = {
    "exact_integer_or_bytes": {"atol": 0.0, "rtol": 0.0, "policy": "byte_exact"},
    "bf16_input_f32_accumulation": {"atol": 0.015625, "rtol": 0.03125, "policy": "bf16_unit_roundoff_2^-7"},
    "f32_accumulation": {"atol": 1.0e-5, "rtol": 1.0e-5, "policy": "declared_f32_reduction_bound"},
    "f32_state_recurrence": {"atol": 3.0e-5, "rtol": 3.0e-5, "policy": "declared_f32_state_bound"},
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bounded(path: Path, maximum: int) -> bytes:
    size = path.stat().st_size
    if size > maximum:
        raise FixtureError(f"cached object {path} is {size} bytes, exceeds bound {maximum}")
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise FixtureError(f"cached object {path} exceeds bound {maximum}")
    return data


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def default_cache_dir() -> Path:
    override = os.environ.get("HIPFIRE_QWEN4_ORACLE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hipfire" / "datasets" / "qwen4-upstream"


def _validated_cache_dir(path: str | Path | None) -> Path:
    root = (Path(path).expanduser() if path is not None else default_cache_dir()).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    allowed = [Path.home().resolve() / ".hipfire" / "datasets", repo_root / ".codeinsight+research"]
    if not any(_within(root, candidate.resolve()) for candidate in allowed):
        raise FixtureError(
            f"upstream cache {root} is outside the allowed ~/.hipfire/datasets or .codeinsight+research roots"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _git_revision(path: Path) -> str | None:
    candidate = path if path.is_dir() else path.parent
    for parent in (candidate, *candidate.parents):
        if not (parent / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=parent,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()
    return None




def _import_pinned_packages() -> dict[str, Any]:
    """Load the only executable package dependency: PyTorch.

    Transformers and vLLM are intentionally not imported.  Their exact
    commit-pinned source files are fetched, hashed, and AST-executed below;
    this keeps the reference independent of installed package versions.
    """
    extra_paths = [
        os.environ.get("HIPFIRE_TRANSFORMERS_CHECKOUT"),
        os.environ.get("HIPFIRE_VLLM_CHECKOUT"),
    ]
    for value in reversed([item for item in extra_paths if item]):
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FixtureError(f"pinned checkout does not exist: {path}")
        sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise FixtureError(f"upstream reference requires torch: {exc}") from exc
    if not hasattr(torch, "bfloat16"):
        raise FixtureError("PyTorch prerequisite has no bfloat16 support")
    return {
        "modules": {"torch": torch},
        "metadata": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": "source_ast",
            "transformers_revision": HF_TRANSFORMERS_COMMIT,
            "transformers_version": HF_TRANSFORMERS_VERSION,
            "vllm": "source_ast",
            "vllm_revision": VLLM_COMMIT,
        },
    }


def _source_url(spec: Mapping[str, str]) -> str:
    host = "https://raw.githubusercontent.com"
    return f"{host}/{spec['project']}/{spec['commit']}/{spec['path']}"


def _source_cache_paths(root: Path, spec: Mapping[str, str]) -> tuple[Path, Path]:
    key = f"{spec['project'].replace('/', '__')}__{spec['commit']}__{spec['path'].replace('/', '__')}"
    base = root / "sources" / key
    return base.with_suffix(".py"), base.with_suffix(".json")


def _load_cached_source(root: Path, spec: Mapping[str, str]) -> tuple[str, dict[str, object]] | None:
    source_path, meta_path = _source_cache_paths(root, spec)
    if not source_path.exists() and not meta_path.exists():
        return None
    if not source_path.exists() or not meta_path.exists():
        raise FixtureError(f"incomplete cached source for {spec['name']}")
    raw = _read_bounded(source_path, MAX_SOURCE_BYTES)
    digest = _sha256_bytes(raw)
    if digest != spec["sha256"]:
        raise FixtureError(
            f"cached source hash mismatch for {spec['name']}: expected {spec['sha256']}, got {digest}"
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"invalid source cache metadata for {spec['name']}: {exc}") from exc
    expected = {"project": spec["project"], "commit": spec["commit"], "path": spec["path"], "sha256": spec["sha256"]}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise FixtureError(f"cached source provenance mismatch for {spec['name']}")
    return raw.decode("utf-8"), {
        "name": spec["name"],
        "project": spec["project"],
        "commit": spec["commit"],
        "path": spec["path"],
        "url": _source_url(spec),
        "sha256": digest,
        "origin": "bounded_cache",
        "cache_key": source_path.name,
    }


def _local_source(root: Path | None, spec: Mapping[str, str]) -> tuple[str, dict[str, object]] | None:
    if root is None:
        return None
    revision = _git_revision(root)
    if revision != spec["commit"]:
        raise FixtureError(f"local {spec['project']} checkout revision mismatch: expected {spec['commit']}, got {revision!r}")
    path = root / spec["path"]
    if not path.is_file():
        raise FixtureError(f"pinned source file is missing from local checkout: {path}")
    raw = _read_bounded(path, MAX_SOURCE_BYTES)
    digest = _sha256_bytes(raw)
    if digest != spec["sha256"]:
        raise FixtureError(f"local source hash mismatch for {spec['name']}: expected {spec['sha256']}, got {digest}")
    return raw.decode("utf-8"), {
        "name": spec["name"],
        "project": spec["project"],
        "commit": spec["commit"],
        "path": spec["path"],
        "url": _source_url(spec),
        "sha256": digest,
        "origin": "local_checkout",
        "cache_key": None,
    }


def _download_source(root: Path, spec: Mapping[str, str]) -> tuple[str, dict[str, object]]:
    url = _source_url(spec)
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    try:
        response = urllib.request.urlopen(request, timeout=60)
        status = getattr(response, "status", response.getcode())
        if status != 200:
            raise FixtureError(f"source fetch {spec['name']} returned HTTP {status}")
        final_url = response.geturl()
        if final_url != url:
            raise FixtureError(f"source fetch redirected away from pinned URL for {spec['name']}: {final_url}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_SOURCE_BYTES:
            raise FixtureError(f"source {spec['name']} exceeds bounded size {MAX_SOURCE_BYTES}")
        raw = response.read(MAX_SOURCE_BYTES + 1)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise FixtureError(f"unable to fetch pinned source {spec['name']}: {exc}") from exc
    if len(raw) > MAX_SOURCE_BYTES:
        raise FixtureError(f"source {spec['name']} exceeds bounded size {MAX_SOURCE_BYTES}")
    digest = _sha256_bytes(raw)
    if digest != spec["sha256"]:
        raise FixtureError(f"downloaded source hash mismatch for {spec['name']}: expected {spec['sha256']}, got {digest}")
    source_path, meta_path = _source_cache_paths(root, spec)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(raw)
    metadata = {
        "project": spec["project"],
        "commit": spec["commit"],
        "path": spec["path"],
        "sha256": digest,
        "url": url,
    }
    meta_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return raw.decode("utf-8"), {
        "name": spec["name"],
        "project": spec["project"],
        "commit": spec["commit"],
        "path": spec["path"],
        "url": url,
        "sha256": digest,
        "origin": "downloaded_bounded_cache",
        "cache_key": source_path.name,
    }


def load_pinned_sources(cache_dir: str | Path | None = None) -> tuple[dict[str, str], list[dict[str, object]], Path]:
    root = _validated_cache_dir(cache_dir)
    hf_checkout = os.environ.get("HIPFIRE_TRANSFORMERS_CHECKOUT")
    vllm_checkout = os.environ.get("HIPFIRE_VLLM_CHECKOUT")
    checkout_roots = {
        "huggingface/transformers": Path(hf_checkout).expanduser().resolve() if hf_checkout else None,
        "vllm-project/vllm": Path(vllm_checkout).expanduser().resolve() if vllm_checkout else None,
    }
    texts: dict[str, str] = {}
    provenance: list[dict[str, object]] = []
    for spec in UPSTREAM_SOURCE_SPECS:
        local = _local_source(checkout_roots[spec["project"]], spec) if checkout_roots[spec["project"]] else None
        loaded = local or _load_cached_source(root, spec) or _download_source(root, spec)
        texts[spec["name"]], identity = loaded
        provenance.append(identity)
    return texts, provenance, root


_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_RETRYABLE_RANGE_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
        errno.EPIPE,
    }
)


def _retryable_range_status(status: object) -> bool:
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    return code == 429 or 500 <= code < 600


def _retryable_range_transport_error(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, (ConnectionError, BrokenPipeError, TimeoutError, socket.timeout)):
            return True
        if isinstance(current, OSError) and current.errno in _RETRYABLE_RANGE_ERRNOS:
            return True
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        cause = current.__cause__
        if isinstance(cause, BaseException):
            pending.append(cause)
        context = current.__context__
        if isinstance(context, BaseException):
            pending.append(context)
    return False


def _sleep_before_range_retry(retry_index: int) -> None:
    delay = min(
        RANGE_RETRY_BACKOFF_SECONDS * (2**retry_index),
        RANGE_RETRY_BACKOFF_MAX_SECONDS,
    )
    time.sleep(delay)




class _RangeClient:
    """Bounded HTTP range reader with per-object provenance seals.

    Every range is keyed by the exact URL/start/end tuple.  The first response
    for a URL records the available ETag, revision, and total length; later
    ranges (including cached ranges) must agree with every seal that was
    present on the first response.  This prevents a proxy or mutable mirror
    from combining slices from different checkpoint revisions.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "checkpoint_ranges"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seals: dict[str, dict[str, object]] = {}
        self.read_bytes = 0

    def _seal(self, url: str, metadata: Mapping[str, object]) -> None:
        revision = metadata.get("repo_commit")
        if revision not in (None, HF_CONFIG_COMMIT):
            raise FixtureError(f"checkpoint range revision mismatch for {url}: {revision!r}")
        current = {
            "etag": metadata.get("etag"),
            "repo_commit": revision,
            "total": metadata.get("total"),
        }
        previous = self.seals.get(url)
        if previous is None:
            self.seals[url] = current
            return
        for key, old in previous.items():
            new = current.get(key)
            if old is not None and new != old:
                raise FixtureError(
                    f"checkpoint range {key} seal mismatch for {url}: expected {old!r}, got {new!r}"
                )
            if old is None and new is not None:
                previous[key] = new
    @staticmethod
    def _expected_end(start: int, end: int, total: object) -> int:
        if total is None:
            return end
        if not isinstance(total, int) or total <= 0:
            raise FixtureError(f"invalid Content-Range total {total!r}")
        if start >= total:
            raise FixtureError(f"range start {start} is outside object of length {total}")
        return min(end, total - 1)

    def _validate_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        url: str,
        start: int,
        end: int,
        body_length: int,
    ) -> None:
        if metadata.get("url") != url or metadata.get("start") != start or metadata.get("end") != end:
            raise FixtureError(f"range provenance mismatch for {url} bytes {start}-{end}")
        total = metadata.get("total")
        response_end = metadata.get("response_end", self._expected_end(start, end, total))
        expected_end = self._expected_end(start, end, total)
        if response_end != expected_end:
            raise FixtureError(
                f"Content-Range end {response_end!r} does not match requested {start}-{end} for total {total!r}"
            )
        expected_length = expected_end - start + 1
        if body_length != expected_length:
            raise FixtureError(f"range response length {body_length} does not match {expected_length}")
        if metadata.get("length") != body_length:
            raise FixtureError(f"range metadata length {metadata.get('length')!r} does not match {body_length}")


    def read(self, url: str, start: int, end: int, *, maximum: int) -> tuple[bytes, dict[str, object]]:
        if start < 0 or end < start:
            raise FixtureError(f"invalid HTTP range {start}-{end}")
        requested_length = end - start + 1
        if requested_length > maximum:
            raise FixtureError(f"requested HTTP range {start}-{end} exceeds bound {maximum}")
        key = _sha256_bytes(f"{url}\0{start}\0{end}".encode("utf-8"))
        body_path = self.cache_dir / f"{key}.bin"
        meta_path = self.cache_dir / f"{key}.json"
        if body_path.exists() or meta_path.exists():
            if not body_path.exists() or not meta_path.exists():
                raise FixtureError(f"incomplete cached range {key}")
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FixtureError(f"invalid cached range metadata {key}: {exc}") from exc
            body = _read_bounded(body_path, maximum)
            self._validate_metadata(
                metadata,
                url=url,
                start=start,
                end=end,
                body_length=len(body),
            )
            if metadata.get("sha256") != _sha256_bytes(body):
                raise FixtureError(f"cached range digest mismatch {key}")
            self._seal(url, metadata)
            self.read_bytes += len(body)
            return body, metadata

        retry_index = 0
        while True:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            )
            try:
                response = urllib.request.urlopen(request, timeout=90)
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    if _retryable_range_status(status) and retry_index < RANGE_MAX_RETRIES:
                        _sleep_before_range_retry(retry_index)
                        retry_index += 1
                        continue
                    raise FixtureError(f"range request returned HTTP {status}, expected 206")
                content_range = response.headers.get("Content-Range", "")
                match = _RANGE_RE.fullmatch(content_range)
                if match is None:
                    raise FixtureError(f"range response has invalid Content-Range {content_range!r}")
                got_start, got_end, total_text = match.groups()
                got_start_i, got_end_i = int(got_start), int(got_end)
                total = None if total_text == "*" else int(total_text)
                expected_end = self._expected_end(start, end, total)
                if got_start_i != start or got_end_i != expected_end:
                    raise FixtureError(
                        f"range response {content_range!r} does not match requested {start}-{end}"
                    )
                expected_length = got_end_i - start + 1
                body = response.read(maximum + 1)
                if len(body) > maximum or len(body) != expected_length:
                    raise FixtureError(f"range response length {len(body)} does not match {expected_length}")
                metadata = {
                    "url": url,
                    "start": start,
                    "end": end,
                    "response_end": got_end_i,
                    "total": total,
                    "length": len(body),
                    "sha256": _sha256_bytes(body),
                    "etag": response.headers.get("ETag"),
                    "repo_commit": response.headers.get("X-Repo-Commit"),
                }
                self._validate_metadata(
                    metadata,
                    url=url,
                    start=start,
                    end=end,
                    body_length=len(body),
                )
                self._seal(url, metadata)
                break
            except urllib.error.HTTPError as exc:
                if _retryable_range_status(exc.code) and retry_index < RANGE_MAX_RETRIES:
                    _sleep_before_range_retry(retry_index)
                    retry_index += 1
                    continue
                raise FixtureError(f"range request returned HTTP {exc.code}, expected 206") from exc
            except (OSError, urllib.error.URLError, ValueError) as exc:
                if _retryable_range_transport_error(exc) and retry_index < RANGE_MAX_RETRIES:
                    _sleep_before_range_retry(retry_index)
                    retry_index += 1
                    continue
                raise FixtureError(f"unable to range-read {url} bytes {start}-{end}: {exc}") from exc
        body_path.write_bytes(body)
        meta_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        self.read_bytes += len(body)
        return body, metadata


def _read_small_object(client: _RangeClient, url: str, expected_sha256: str, *, label: str) -> tuple[bytes, dict[str, object]]:
    body, metadata = client.read(url, 0, MAX_HEADER_BYTES - 1, maximum=MAX_HEADER_BYTES)
    if metadata.get("total") != len(body):
        raise FixtureError(f"pinned {label} is larger than bounded object limit")
    digest = _sha256_bytes(body)
    if digest != expected_sha256:
        raise FixtureError(f"pinned {label} hash mismatch: expected {expected_sha256}, got {digest}")
    if metadata.get("repo_commit") not in (None, HF_CONFIG_COMMIT):
        raise FixtureError(f"pinned {label} revision mismatch: {metadata.get('repo_commit')!r}")
    return body, metadata


def _dtype_info(dtype: str) -> tuple[int, str]:
    return {
        "BF16": (2, "uint16"),
        "F32": (4, "float32"),
        "F16": (2, "uint16"),
        "I64": (8, "int64"),
        "I32": (4, "int32"),
        "U8": (1, "uint8"),
        "BOOL": (1, "bool"),
    }.get(dtype, (0, ""))
def _tensor_descriptor(
    client: _RangeClient,
    model_index: Mapping[str, str],
    tensor_name: str,
) -> dict[str, object]:
    """Return one validated safetensors descriptor without reading tensor data."""
    shard = model_index.get(tensor_name)
    if shard is None:
        raise FixtureError(f"pinned model index is missing required tensor {tensor_name}")
    url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/{shard}"
    header, header_meta = _safetensor_header(client, url)
    descriptor = header.get(tensor_name)
    if not isinstance(descriptor, dict):
        raise FixtureError(f"safetensors header is missing required tensor {tensor_name}")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not isinstance(offsets, list)
        or len(offsets) != 2
    ):
        raise FixtureError(f"invalid safetensors descriptor for {tensor_name}")
    try:
        shape_tuple = tuple(int(item) for item in shape)
        begin, finish = int(offsets[0]), int(offsets[1])
    except (TypeError, ValueError) as exc:
        raise FixtureError(f"invalid safetensors descriptor for {tensor_name}") from exc
    if not shape_tuple or any(item <= 0 for item in shape_tuple) or begin < 0 or finish < begin:
        raise FixtureError(f"invalid safetensors shape/offsets for {tensor_name}")
    itemsize, storage_dtype = _dtype_info(dtype)
    if not itemsize:
        raise FixtureError(f"unsupported dtype {dtype!r} for {tensor_name}")
    expected_bytes = math.prod(shape_tuple) * itemsize
    if finish - begin != expected_bytes:
        raise FixtureError(
            f"tensor {tensor_name} data extent {finish - begin} does not equal {expected_bytes}"
        )
    header_length = int(header_meta["header_length"])
    data_start = 8 + header_length + begin
    data_end = 8 + header_length + finish
    total = header_meta.get("total")
    if isinstance(total, int) and data_end > total:
        raise FixtureError(f"tensor {tensor_name} extends beyond shard length {total}")
    return {
        "tensor": tensor_name,
        "shard": shard,
        "url": url,
        "dtype": dtype,
        "storage_dtype": storage_dtype,
        "shape": list(shape_tuple),
        "data_offsets": [begin, finish],
        "header_length": header_length,
        "data_start": data_start,
        "data_end": data_end,
        "header_sha256": header_meta["header_sha256"],
        "header_etag": header_meta.get("header_etag"),
        "header_repo_commit": header_meta.get("header_repo_commit"),
    }


def _read_tensor_elements(
    client: _RangeClient,
    descriptor: Mapping[str, object],
    element_start: int,
    element_count: int,
    *,
    output_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Read an arbitrary contiguous element range in bounded HTTP slices."""
    shape = tuple(int(item) for item in descriptor["shape"])  # type: ignore[arg-type]
    itemsize, dtype = _dtype_info(str(descriptor["dtype"]))
    total_items = math.prod(shape)
    if element_start < 0 or element_count < 0 or element_start + element_count > total_items:
        raise FixtureError(
            f"tensor slice {descriptor['tensor']} elements {element_start}+{element_count} exceeds {shape}"
        )
    if output_shape is None:
        output_shape = (element_count,)
    if math.prod(tuple(int(item) for item in output_shape)) != element_count:
        raise FixtureError(f"tensor slice output shape {output_shape} does not contain {element_count} elements")
    max_items = max(1, MAX_TENSOR_SLICE_BYTES // itemsize)
    raw_parts: list[bytes] = []
    ranges: list[dict[str, object]] = []
    remaining = element_count
    cursor = element_start
    while remaining:
        count = min(remaining, max_items)
        absolute_start = int(descriptor["data_start"]) + cursor * itemsize
        absolute_end = absolute_start + count * itemsize - 1
        raw, metadata = client.read(
            str(descriptor["url"]),
            absolute_start,
            absolute_end,
            maximum=count * itemsize,
        )
        if len(raw) != count * itemsize:
            raise FixtureError(
                f"tensor slice {descriptor['tensor']} returned {len(raw)} bytes, expected {count * itemsize}"
            )
        raw_parts.append(raw)
        ranges.append(
            {
                "start": absolute_start,
                "end": absolute_end,
                "length": len(raw),
                "sha256": metadata["sha256"],
                "etag": metadata.get("etag"),
                "repo_commit": metadata.get("repo_commit"),
            }
        )
        cursor += count
        remaining -= count
    raw = b"".join(raw_parts)
    values = _decode_storage(raw, str(descriptor["dtype"]), tuple(int(item) for item in output_shape))
    return {
        "tensor": descriptor["tensor"],
        "shard": descriptor["shard"],
        "url": descriptor["url"],
        "dtype": descriptor["dtype"],
        "shape": list(shape),
        "slice_shape": [int(item) for item in output_shape],
        "element_start": element_start,
        "element_count": element_count,
        "byte_start": int(descriptor["data_start"]) + element_start * itemsize,
        "byte_length": len(raw),
        "sha256": _sha256_bytes(raw),
        "header_sha256": descriptor["header_sha256"],
        "ranges": ranges,
        "values": values,
    }


def _read_tensor_rows(
    client: _RangeClient,
    model_index: Mapping[str, str],
    tensor_name: str,
    row_start: int,
    row_count: int,
) -> dict[str, object]:
    descriptor = _tensor_descriptor(client, model_index, tensor_name)
    shape = tuple(int(item) for item in descriptor["shape"])  # type: ignore[arg-type]
    if row_start < 0 or row_count < 0 or row_start + row_count > shape[0]:
        raise FixtureError(f"tensor row slice {tensor_name} {row_start}+{row_count} exceeds {shape}")
    row_items = math.prod(shape[1:]) if len(shape) > 1 else 1
    return _read_tensor_elements(
        client,
        descriptor,
        row_start * row_items,
        row_count * row_items,
        output_shape=(row_count,) + shape[1:],
    )


def _read_tensor_expert_rows(
    client: _RangeClient,
    model_index: Mapping[str, str],
    tensor_name: str,
    expert: int,
    row_start: int,
    row_count: int,
) -> dict[str, object]:
    """Read one expert's row interval from a fused [experts, rows, cols] tensor."""
    descriptor = _tensor_descriptor(client, model_index, tensor_name)
    shape = tuple(int(item) for item in descriptor["shape"])  # type: ignore[arg-type]
    if len(shape) != 3:
        raise FixtureError(f"expert tensor {tensor_name} is not rank three: {shape}")
    experts, rows, cols = shape
    if expert < 0 or expert >= experts or row_start < 0 or row_count < 0 or row_start + row_count > rows:
        raise FixtureError(f"expert row slice {tensor_name}[{expert}] {row_start}+{row_count} exceeds {shape}")
    start = (expert * rows + row_start) * cols
    result = _read_tensor_elements(
        client,
        descriptor,
        start,
        row_count * cols,
        output_shape=(row_count, cols),
    )
    result["expert"] = expert
    result["row_start"] = row_start
    return result


class PinnedCheckpoint:
    """Metadata/index view whose tensor reads remain bounded and independently sealed."""

    def __init__(
        self,
        client: _RangeClient,
        *,
        index: Mapping[str, object],
        config: Mapping[str, object],
        index_meta: Mapping[str, object],
        config_meta: Mapping[str, object],
        index_sha256: str,
        config_sha256: str,
    ):
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
            raise FixtureError("pinned model index has no valid weight_map")
        self.client = client
        self.index = index
        self.config = config
        self.weight_map: dict[str, str] = dict(weight_map)
        self.index_meta = dict(index_meta)
        self.config_meta = dict(config_meta)
        self.index_sha256 = index_sha256
        self.config_sha256 = config_sha256

    @property
    def read_bytes(self) -> int:
        return self.client.read_bytes

    @property
    def seals(self) -> dict[str, dict[str, object]]:
        return {url: dict(seal) for url, seal in self.client.seals.items()}

    def tensor_rows(self, name: str, row_start: int, row_count: int) -> dict[str, object]:
        return _read_tensor_rows(self.client, self.weight_map, name, row_start, row_count)

    def tensor_expert_rows(
        self,
        name: str,
        expert: int,
        row_start: int,
        row_count: int,
    ) -> dict[str, object]:
        return _read_tensor_expert_rows(self.client, self.weight_map, name, expert, row_start, row_count)

    def provenance(self) -> dict[str, object]:
        return {
            "model": HF_MODEL,
            "revision": HF_CONFIG_COMMIT,
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "index_etag": self.index_meta.get("etag"),
            "config_etag": self.config_meta.get("etag"),
            "range_seals": self.seals,
            "read_bytes": self.read_bytes,
        }


def open_pinned_checkpoint(cache_dir: str | Path | None = None) -> PinnedCheckpoint:
    """Open the pinned index/config; tensor payloads are fetched only on demand."""
    root = _validated_cache_dir(cache_dir)
    client = _RangeClient(root)
    index_url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/model.safetensors.index.json"
    config_url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/config.json"
    index_raw, index_meta = _read_small_object(client, index_url, MODEL_INDEX_SHA256, label="safetensors index")
    config_raw, config_meta = _read_small_object(client, config_url, MODEL_CONFIG_SHA256, label="model config")
    try:
        index = json.loads(index_raw.decode("utf-8"))
        config = json.loads(config_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FixtureError(f"invalid pinned model metadata: {exc}") from exc
    if not isinstance(index, dict) or not isinstance(config, dict):
        raise FixtureError("pinned model metadata must be JSON objects")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict) or text_config.get("model_type") not in ("qwen4_exp_text", "qwen4_exp"):
        raise FixtureError(f"pinned model config has unexpected model_type {text_config!r}")
    return PinnedCheckpoint(
        client,
        index=index,
        config=config,
        index_meta=index_meta,
        config_meta=config_meta,
        index_sha256=_sha256_bytes(index_raw),
        config_sha256=_sha256_bytes(config_raw),
    )




def _decode_storage(raw: bytes, dtype: str, shape: Sequence[int]) -> Tensor:
    itemsize, target_dtype = _dtype_info(dtype)
    if not itemsize:
        raise FixtureError(f"unsupported safetensors dtype {dtype!r}")
    count = math.prod(shape)
    if len(raw) != count * itemsize:
        raise FixtureError(f"source tensor slice has {len(raw)} bytes, expected {count * itemsize}")
    if target_dtype == "bool":
        values: Iterable[object] = (bool(item) for item in raw)
    elif target_dtype == "uint8":
        values = raw
    else:
        code = {"uint16": "H", "float32": "f", "int64": "q", "int32": "i"}[target_dtype]
        values = struct.unpack("<" + code * count, raw)
    return tensor(shape, values, target_dtype)


def _safetensor_header(client: _RangeClient, url: str) -> tuple[dict[str, object], dict[str, object]]:
    prefix, prefix_meta = client.read(url, 0, 7, maximum=8)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 2 or header_length > MAX_HEADER_BYTES:
        raise FixtureError(f"safetensors header length {header_length} exceeds bound")
    header_raw, header_meta = client.read(url, 8, 8 + header_length - 1, maximum=MAX_HEADER_BYTES)
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FixtureError(f"invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict) or "__metadata__" not in header:
        raise FixtureError("safetensors header has no __metadata__")
    return header, {
        "header_length": header_length,
        "header_sha256": _sha256_bytes(header_raw),
        "header_etag": header_meta.get("etag") or prefix_meta.get("etag"),
        "header_repo_commit": header_meta.get("repo_commit") or prefix_meta.get("repo_commit"),
        "total": header_meta.get("total"),
    }


def _tensor_slice(
    client: _RangeClient,
    model_index: Mapping[str, str],
    tensor_name: str,
    *,
    requested_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    shard = model_index.get(tensor_name)
    if shard is None:
        raise FixtureError(f"pinned model index is missing required tensor {tensor_name}")
    url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/{shard}"
    header, header_meta = _safetensor_header(client, url)
    descriptor = header.get(tensor_name)
    if not isinstance(descriptor, dict):
        raise FixtureError(f"safetensors header is missing required tensor {tensor_name}")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
        raise FixtureError(f"invalid safetensors descriptor for {tensor_name}")
    shape_tuple = tuple(int(item) for item in shape)
    begin, finish = int(offsets[0]), int(offsets[1])
    if begin < 0 or finish < begin:
        raise FixtureError(f"invalid data offsets for {tensor_name}")
    itemsize, storage_dtype = _dtype_info(dtype)
    if not itemsize:
        raise FixtureError(f"unsupported dtype {dtype!r} for {tensor_name}")
    if requested_shape is None:
        row_items = math.prod(shape_tuple[1:]) if len(shape_tuple) > 1 else 1
        row_bytes = row_items * itemsize
        if row_bytes <= MAX_TENSOR_SLICE_BYTES:
            rows = max(1, min(shape_tuple[0] if shape_tuple else 1, MAX_TENSOR_SLICE_BYTES // row_bytes))
            slice_shape = (rows,) + shape_tuple[1:] if shape_tuple else ()
        else:
            count = max(1, MAX_TENSOR_SLICE_BYTES // itemsize)
            slice_shape = (count,)
    else:
        slice_shape = tuple(int(item) for item in requested_shape)
        if len(slice_shape) != len(shape_tuple) or any(a > b for a, b in zip(slice_shape, shape_tuple)):
            raise FixtureError(f"requested source slice {slice_shape} exceeds tensor shape {shape_tuple} for {tensor_name}")
    slice_items = math.prod(slice_shape) if slice_shape else 1
    slice_bytes = slice_items * itemsize
    if slice_bytes > MAX_TENSOR_SLICE_BYTES:
        raise FixtureError(f"source tensor slice for {tensor_name} exceeds {MAX_TENSOR_SLICE_BYTES} bytes")
    absolute_begin = 8 + int(header_meta["header_length"]) + begin
    absolute_end = absolute_begin + slice_bytes - 1
    if absolute_end >= 8 + int(header_meta["header_length"]) + finish:
        raise FixtureError(f"source slice for {tensor_name} exceeds its data offsets")
    raw, range_meta = client.read(url, absolute_begin, absolute_end, maximum=MAX_TENSOR_SLICE_BYTES)
    values = _decode_storage(raw, dtype, slice_shape)
    return {
        "tensor": tensor_name,
        "shard": shard,
        "url": url,
        "dtype": dtype,
        "storage_dtype": storage_dtype,
        "shape": list(shape_tuple),
        "slice_shape": list(slice_shape),
        "data_offsets": [begin, finish],
        "byte_offset": absolute_begin,
        "byte_length": slice_bytes,
        "sha256": _sha256_bytes(raw),
        "header_sha256": header_meta["header_sha256"],
        "etag": range_meta.get("etag") or header_meta.get("header_etag"),
        "repo_commit": range_meta.get("repo_commit") or HF_CONFIG_COMMIT,
        "values": values,
    }


def load_checkpoint_slices(cache_dir: Path) -> dict[str, object]:
    client = _RangeClient(cache_dir)
    index_url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/model.safetensors.index.json"
    config_url = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_CONFIG_COMMIT}/config.json"
    index_raw, index_meta = _read_small_object(client, index_url, MODEL_INDEX_SHA256, label="safetensors index")
    config_raw, config_meta = _read_small_object(client, config_url, MODEL_CONFIG_SHA256, label="model config")
    try:
        index = json.loads(index_raw.decode("utf-8"))
        config = json.loads(config_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FixtureError(f"invalid pinned model metadata: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise FixtureError("pinned model index has no weight_map")
    text_config = config.get("text_config", config) if isinstance(config, dict) else {}
    if text_config.get("model_type") not in ("qwen4_exp_text", "qwen4_exp"):
        raise FixtureError(f"pinned model config has unexpected model_type {text_config.get('model_type')!r}")
    tensors: dict[str, dict[str, object]] = {}
    for family, names in CHECKPOINT_TENSORS.items():
        for name in names:
            requested_shape = (208, 160) if name == PLE_EMBEDDING else None
            tensors[name] = _tensor_slice(client, weight_map, name, requested_shape=requested_shape)
    for name, record in tensors.items():
        values = record["values"]
        if not isinstance(values, Tensor):
            raise FixtureError(f"internal source slice type error for {name}")
        if values.dtype in ("float32", "float64") and any(not math.isfinite(float(item)) for item in values.data):
            raise FixtureError(f"nonfinite trained checkpoint slice {name}")
        if values.dtype == "uint16" and record["dtype"] == "BF16":
            if any((int(item) & 0x7F80) == 0x7F80 for item in values.data):
                raise FixtureError(f"nonfinite trained BF16 checkpoint slice {name}")
    return {
        "model": HF_MODEL,
        "revision": HF_CONFIG_COMMIT,
        "index_sha256": _sha256_bytes(index_raw),
        "config_sha256": _sha256_bytes(config_raw),
        "index_etag": index_meta.get("etag"),
        "config_etag": config_meta.get("etag"),
        "tensors": tensors,
    }


def _strip_decorators(node: ast.AST) -> ast.AST:
    """Copy source while retaining decorators with Python binding semantics."""
    cloned = copy.deepcopy(node)
    for item in ast.walk(cloned):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            item.decorator_list = [
                decorator
                for decorator in item.decorator_list
                if isinstance(decorator, ast.Name) and decorator.id in {"staticmethod", "classmethod"}
            ]
    return cloned


def _top_level_nodes(tree: ast.Module, wanted: set[str]) -> list[ast.AST]:
    found: list[ast.AST] = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            found.append(_strip_decorators(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in wanted for target in targets):
                found.append(_strip_decorators(node))
    missing = wanted - {getattr(node, "name", None) for node in tree.body}
    missing -= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if missing:
        raise FixtureError(f"pinned source is missing required definitions: {sorted(missing)}")
    return found


def _compile_source(source: str, names: Sequence[str], globals_extra: Mapping[str, object]) -> dict[str, object]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FixtureError(f"pinned source cannot be parsed: {exc}") from exc
    selected = _top_level_nodes(tree, set(names))
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"__name__": "qwen4_pinned_reference", "__builtins__": __builtins__}
    namespace.update(globals_extra)
    try:
        exec(compile(module, "<pinned-qwen4-source>", "exec"), namespace, namespace)
    except Exception as exc:
        raise FixtureError(f"unable to compile pinned source equations: {exc}") from exc
    return namespace


def _compile_method(source: str, class_name: str, method_name: str, globals_extra: Mapping[str, object]) -> Any:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FixtureError(f"pinned source cannot be parsed: {exc}") from exc
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if class_node is None:
        raise FixtureError(f"pinned source is missing class {class_name}")
    method = next((node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name), None)
    if method is None:
        raise FixtureError(f"pinned source is missing {class_name}.{method_name}")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            _strip_decorators(method),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"__name__": "qwen4_pinned_method", "__builtins__": __builtins__}
    namespace.update(globals_extra)
    try:
        exec(compile(module, "<pinned-qwen4-method>", "exec"), namespace, namespace)
    except Exception as exc:
        raise FixtureError(f"unable to compile pinned {class_name}.{method_name}: {exc}") from exc
    return namespace[method_name]




_PINNED_OPERATOR_SIGNATURES: tuple[dict[str, object], ...] = (
    {"source": "transformers_modeling", "class": "Qwen4ExpTextGatedResidual", "method": "forward", "args": ("self", "hyper_input"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextGatedDeltaNet", "method": "forward", "args": ("self", "hidden_states", "cache_params", "attention_mask"), "vararg": None, "kwarg": "kwargs", "defaults": 2},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextAttention", "method": "forward", "args": ("self", "hidden_states", "position_embeddings", "attention_mask", "past_key_values"), "vararg": None, "kwarg": "kwargs", "defaults": 1},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextQSAIndexer", "method": "forward", "args": ("self", "hidden_states", "position_embeddings", "attention_mask", "past_key_values"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextNGramEmbedding", "method": "forward", "args": ("self", "input_ids", "past_key_values"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextNGramEmbedding", "method": "_shift_right_ignore_eos", "args": ("self", "token_ids", "shift"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextPLELayer", "method": "forward", "args": ("self", "hidden_states", "input_ids", "past_key_values", "conv_mask"), "vararg": None, "kwarg": None, "defaults": 1},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextPLELayer", "method": "_short_conv", "args": ("self", "hidden_states", "past_key_values"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRotaryEmbedding", "method": "forward", "args": ("self", "x", "position_ids"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRMSNorm", "method": "forward", "args": ("self", "x"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRMSNormGated", "method": "forward", "args": ("self", "hidden_states", "gate"), "vararg": None, "kwarg": None, "defaults": 1},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextMLP", "method": "forward", "args": ("self", "x"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextExperts", "method": "forward", "args": ("self", "hidden_states", "top_k_index", "top_k_weights"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextTopKRouter", "method": "forward", "args": ("self", "hidden_states"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextSparseMoeBlock", "method": "forward", "args": ("self", "hidden_states"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": None, "method": "torch_chunk_gated_delta_rule", "args": ("query", "key", "value", "g", "beta", "chunk_size", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"), "vararg": None, "kwarg": "kwargs", "defaults": 4},
    {"source": "transformers_modeling", "class": None, "method": "torch_recurrent_gated_delta_rule", "args": ("query", "key", "value", "g", "beta", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"), "vararg": None, "kwarg": "kwargs", "defaults": 1},
    {"source": "transformers_modeling", "class": None, "method": "causal_conv1d_fn", "args": ("hidden_states", "weight", "bias", "activation"), "vararg": None, "kwarg": "kwargs", "defaults": 2},
    {"source": "transformers_modeling", "class": None, "method": "causal_conv1d_update", "args": ("hidden_states", "conv_state", "weight", "bias", "activation"), "vararg": None, "kwarg": None, "defaults": 2},
    {"source": "transformers_modeling", "class": None, "method": "l2norm", "args": ("x", "dim", "eps"), "vararg": None, "kwarg": None, "defaults": 2},
    {"source": "transformers_modeling", "class": None, "method": "rotate_half", "args": ("x",), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": None, "method": "apply_rotary_pos_emb", "args": ("q", "k", "cos", "sin", "unsqueeze_dim"), "vararg": None, "kwarg": None, "defaults": 4},
    {"source": "transformers_modeling", "class": None, "method": "repeat_kv", "args": ("hidden_states", "n_rep"), "vararg": None, "kwarg": None, "defaults": 0},
    {"source": "transformers_modeling", "class": None, "method": "eager_attention_forward", "args": ("module", "query", "key", "value", "attention_mask", "scaling", "dropout"), "vararg": None, "kwarg": "kwargs", "defaults": 1},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextDecoderLayer", "method": "forward", "args": ("self", "hidden_states", "position_embeddings", "attention_mask", "conv_mask", "past_key_values", "ple_input_ids"), "vararg": None, "kwarg": "kwargs", "defaults": 4},
)
_PINNED_OPERATOR_DIGEST_SPECS: tuple[dict[str, object], ...] = _PINNED_OPERATOR_SIGNATURES + (
    {"source": "transformers_modeling", "class": "Qwen4ExpTextGatedResidual", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextGatedDeltaNet", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextAttention", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextQSAIndexer", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextNGramEmbedding", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextPLELayer", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRotaryEmbedding", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRotaryEmbedding", "method": "compute_default_rope_parameters"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRotaryEmbedding", "method": "apply_interleaved_mrope"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRMSNorm", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRMSNorm", "method": "_norm"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextRMSNormGated", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextMLP", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextExperts", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextTopKRouter", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextSparseMoeBlock", "method": "__init__"},
    {"source": "transformers_modeling", "class": "Qwen4ExpTextDecoderLayer", "method": "__init__"},
    {"source": "transformers_modeling", "class": None, "method": "apply_mask_to_padding_states"},
    {"source": "transformers_modeling", "class": None, "method": "_splitmix64"},
    {"source": "transformers_modeling", "class": None, "method": "_build_layer_multipliers"},
    {"source": "transformers_modeling", "class": None, "method": "_is_prime"},
    {"source": "transformers_modeling", "class": None, "method": "_find_nth_prime_after"},
)

_EXPECTED_OPERATOR_AST_SHA256 = types.MappingProxyType(
    {
        "Qwen4ExpTextGatedResidual.__init__": "6e1d6bc99682dafbca7a9f7710ff5fa233da45f005354d58ea1112b0b48c930e",
        "Qwen4ExpTextGatedResidual.forward": "a1f9ac81b05a64ba879e96c5c62a715c9892f8739f2f2501d16e5f537c589ce8",
        "Qwen4ExpTextGatedDeltaNet.__init__": "dd39e1e905b187afa0868e1c8c62a74dce880a475ecd87eb22a9a616eb5f1c0a",
        "Qwen4ExpTextGatedDeltaNet.forward": "03763c9ca1a426c49b16feed5e40ef7ebccefc4c2ef32e1d50983691df7ba863",
        "Qwen4ExpTextAttention.__init__": "01674c533ba3cf5872cf0c09a6bdd645b5c0ab9c642f4542680fd276ad65af63",
        "Qwen4ExpTextAttention.forward": "c40a96fe210f40e34deafded604e43138bdb1679c6e60ebcae3f981fde01de57",
        "Qwen4ExpTextQSAIndexer.__init__": "1cabf0172d7b7c85e0c42101e96d219df8d0f5d81fab346c37cf138087837c00",
        "Qwen4ExpTextQSAIndexer.forward": "4be4dfa38613822a37295a5bdd497f28e13feb4081ead86c84cc2aa3bfcfa205",
        "Qwen4ExpTextNGramEmbedding.__init__": "1c415b76ba5e9e29bead10d428110618c559a850fe6ac379158a97407401661f",
        "Qwen4ExpTextNGramEmbedding.forward": "bb443aa599841f4e0bdbf4a34839fd168ec0502777280dcefe7e634aed82a2bf",
        "Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos": "766c41743a20992a64a1df57f3c089e43d027c2a39eac4aae4315049940a406c",
        "Qwen4ExpTextPLELayer.__init__": "15891209ca677af19e5459b9706a972324a5cadcae68e7861eea52c8b3367987",
        "Qwen4ExpTextPLELayer.forward": "73802991ec8c2599349914551156692b28b652df230c6f21a98689874d79d7a6",
        "Qwen4ExpTextPLELayer._short_conv": "4776b7a9f9ddb531e406fac22b5c5c1e7372d18c5800b1de94e540d6c8dc375e",
        "Qwen4ExpTextRotaryEmbedding.__init__": "0b28c133b226f80bb2c0f5b10aa8abff1d63ae34bad11b78f6d9da912e1d1de6",
        "Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters": "b64ff8ba6d988ea1b4cb373ff684dcfbcda9685c0520d873aa850863bb579ea3",
        "Qwen4ExpTextRotaryEmbedding.forward": "22cc8862c2da69ece4487bacb5e83794cc50c9a1705ecdc6e893293523e17918",
        "Qwen4ExpTextRotaryEmbedding.apply_interleaved_mrope": "031fdd4fb9c139ce2c2447eb74d6fd5a30442fe7ce4e825f654b27d4ac06e5d4",
        "Qwen4ExpTextRMSNorm.__init__": "a662601e53462088b4eb02c7ebc78fd7979ce1095a6dfe370d9c75294d929d3b",
        "Qwen4ExpTextRMSNorm._norm": "64951ff87b66b040e8650948b58c355486064e90908a85734218c9a646dae9e3",
        "Qwen4ExpTextRMSNorm.forward": "fb25a12e0dd200f06661af8c14b22a98e87c907e88d2a3aa5bd0c85f76e7b68f",
        "Qwen4ExpTextRMSNormGated.__init__": "eb8b1a583ffda82b1c05d3254547cac3aae48f357d5fd099b66f96fd681a35e7",
        "Qwen4ExpTextRMSNormGated.forward": "14f1b1412449675bb3048b948f0c9c4641ec0d28e4af1f0176d4f47a805166f9",
        "Qwen4ExpTextMLP.__init__": "5de5e849fb9fe0ae347a9b6a73a531c6d50c589b3f38ad48d2aead2c0d081ae5",
        "Qwen4ExpTextMLP.forward": "468b4e3f6b1120c589bcf9199034816dce5b637bf01fe5c76646f3729e967d03",
        "Qwen4ExpTextExperts.__init__": "e4909208595b9b0c065440ef30204cdd2b71002bc34fa2257edca66a0dfa50b6",
        "Qwen4ExpTextExperts.forward": "00c4108f96c02c67b7e9e0baea3d166aee3e0a93ea0f4190c8bf79386605fcaf",
        "Qwen4ExpTextTopKRouter.__init__": "bdb78bf81cc1e4db718578f8a6bb58553f15a7f211cd4da82a6d64727146abee",
        "Qwen4ExpTextTopKRouter.forward": "0a0097d78ed871505f54f30ef285939bc25057d0e122f61c5a8ea5f702c65afb",
        "Qwen4ExpTextSparseMoeBlock.__init__": "188cf4926d77bb2f5521d7636a5d544f5ba6105866b5a9739270d0a1d082f0a3",
        "_splitmix64": "97a08c66543de379717684966428caf568b5c56caf3e5e91bdc98399983309a1",
        "_build_layer_multipliers": "a5756debb4036c0e0e1579c739539961ab41955b0e877f85aac4a0550b0412cb",
        "_is_prime": "af6dc0fde5d2273b4055b41cec94b78fcb3f255f26ecb04a14d8dcd293e1ae8e",
        "_find_nth_prime_after": "ed12d037614ef116af3317301bf82d91e6cc59e7df3f3d844acdcc56ef7ea291",
        "Qwen4ExpTextSparseMoeBlock.forward": "8c70c4e5297d16f359a0f244155de792e7412402792f8eeff6226880ade5b07c",
        "Qwen4ExpTextDecoderLayer.__init__": "69d8b7a34e50d134ebb41c378c78c0a0323ab6d859e036640cd365a68606aeb6",
        "Qwen4ExpTextDecoderLayer.forward": "1c754f2c1b49ace9793dcbcd5530babc4f5dd423e3b38cb233c4578f5b6db38a",
        "apply_mask_to_padding_states": "bc47dcff0659508145d9def52a6f0fbed08d207cf7d5562b60eaf4437dfbac46",
        "causal_conv1d_fn": "378eaddeafac2ced194439fc49b75ed85e10c8f3895c262f8627ad107d358c1a",
        "causal_conv1d_update": "0284ada30446410daafd359b936cb815bb873301202c732d9fbddb0ece8c2ba2",
        "l2norm": "4049615a48d1eb41c5c29337e7a34457cb4e4cea85d0cc602bffad20cfd05fb1",
        "torch_chunk_gated_delta_rule": "28c0a0bfbecafe9a5781f7e2817dab1e47da6c796a64a23f2fe649daf5ce0218",
        "torch_recurrent_gated_delta_rule": "9aba4a5e00969de0229f4d6fd6ba6cf2788ee58c08c201f4ca9cafc25c0168ca",
        "rotate_half": "7a26b6ec674072b8eb8553902cec9b2e33c8bf95c2d51f72372503031aeaf3c3",
        "apply_rotary_pos_emb": "a7fed3320acf3edcd986df4c2404944e9667b56e114614fc042ffaedaf793509",
        "repeat_kv": "1e0fd7298a25393f0b9cdff62b94842e42ae90c3f45d3c597e0f35e005ca3766",
        "eager_attention_forward": "2983de7669a3a910b4a812afbfa1353cfc1727aca41fecd3d5e9ab1e4d4bab9e",
    }
)

_PINNED_BASE_CONTRACTS = {
    "Qwen4ExpTextGatedResidual": "nn.Module",
    "Qwen4ExpTextGatedDeltaNet": "nn.Module",
    "Qwen4ExpTextAttention": "nn.Module",
    "Qwen4ExpTextQSAIndexer": "nn.Module",
    "Qwen4ExpTextNGramEmbedding": "nn.Module",
    "Qwen4ExpTextPLELayer": "nn.Module",
    "Qwen4ExpTextRotaryEmbedding": "nn.Module",
    "Qwen4ExpTextRMSNorm": "nn.Module",
    "Qwen4ExpTextRMSNormGated": "nn.Module",
    "Qwen4ExpTextMLP": "nn.Module",
    "Qwen4ExpTextExperts": "nn.Module",
    "Qwen4ExpTextTopKRouter": "nn.Module",
    "Qwen4ExpTextSparseMoeBlock": "nn.Module",
    "Qwen4ExpTextDecoderLayer": "GradientCheckpointingLayer",
}


def _callable_node(tree: ast.Module, spec: Mapping[str, object]) -> ast.AST | None:
    class_name = spec.get("class")
    method_name = str(spec["method"])
    if class_name is None:
        return next(
            (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name),
            None,
        )
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if class_node is None:
        return None
    return next(
        (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name),
        None,
    )


_EXPECTED_DEFAULT_EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "Qwen4ExpTextGatedDeltaNet.forward": ("Constant(value=None)", "Constant(value=None)"),
    "Qwen4ExpTextAttention.forward": ("Constant(value=None)",),
    "Qwen4ExpTextPLELayer.forward": ("Constant(value=None)",),
    "Qwen4ExpTextRMSNormGated.forward": ("Constant(value=None)",),
    "torch_chunk_gated_delta_rule": (
        "Constant(value=64)",
        "Constant(value=None)",
        "Constant(value=False)",
        "Constant(value=False)",
    ),
    "torch_recurrent_gated_delta_rule": ("Constant(value=False)",),
    "causal_conv1d_fn": ("Constant(value=None)", "Constant(value=None)"),
    "causal_conv1d_update": ("Constant(value=None)", "Constant(value=None)"),
    "l2norm": ("UnaryOp(op=USub(), operand=Constant(value=1))", "Constant(value=1e-06)"),
    "apply_rotary_pos_emb": (
        "Constant(value=None)",
        "Constant(value=None)",
        "Constant(value=None)",
        "Constant(value=1)",
    ),
    "eager_attention_forward": ("Constant(value=0.0)",),
    "Qwen4ExpTextDecoderLayer.forward": (
        "Constant(value=None)",
        "Constant(value=None)",
        "Constant(value=None)",
        "Constant(value=None)",
    ),
}


def _callable_label(spec: Mapping[str, object]) -> str:
    return f"{spec.get('class') + '.' if spec.get('class') else ''}{spec['method']}"


def _node_signature(
    node: ast.AST,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None, tuple[str, ...], str | None, tuple[str, ...]]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise FixtureError("pinned operator definition is not a function")
    arguments = node.args
    posonly = tuple(item.arg for item in arguments.posonlyargs)
    positional = tuple(item.arg for item in arguments.args)
    vararg = arguments.vararg.arg if arguments.vararg is not None else None
    kwonly = tuple(item.arg for item in arguments.kwonlyargs)
    kwarg = arguments.kwarg.arg if arguments.kwarg is not None else None
    defaults = tuple(ast.dump(item, annotate_fields=True, include_attributes=False) for item in arguments.defaults)
    return posonly, positional, vararg, kwonly, kwarg, defaults


def _validate_pinned_operator_signatures(
    source_text: Mapping[str, str],
    specs: Sequence[Mapping[str, object]] = _PINNED_OPERATOR_SIGNATURES,
) -> None:
    trees: dict[str, ast.Module] = {}
    for spec in specs:
        source_name = str(spec["source"])
        if source_name not in trees:
            try:
                trees[source_name] = ast.parse(source_text[source_name])
            except (KeyError, SyntaxError) as exc:
                raise FixtureError(f"pinned source cannot satisfy operator signatures for {source_name}: {exc}") from exc
        tree = trees[source_name]
        class_name = spec.get("class")
        if class_name is not None and class_name in _PINNED_BASE_CONTRACTS:
            class_node = next((item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name), None)
            if class_node is None:
                raise FixtureError(f"pinned source is missing required class {class_name}")
            actual_bases = tuple(ast.unparse(base) for base in class_node.bases)
            expected_base = _PINNED_BASE_CONTRACTS[str(class_name)]
            if actual_bases != (expected_base,):
                raise FixtureError(
                    f"pinned base contract changed for {class_name}: expected {(expected_base,)!r}, got {actual_bases!r}"
                )
        node = _callable_node(tree, spec)
        if node is None:
            raise FixtureError(f"pinned source is missing required operator {_callable_label(spec)}")
        actual = _node_signature(node)
        expected = (
            tuple(spec.get("posonly", ())),
            tuple(spec["args"]),
            spec.get("vararg"),
            tuple(spec.get("kwonly", ())),
            spec.get("kwarg"),
            tuple(_EXPECTED_DEFAULT_EXPRESSIONS.get(_callable_label(spec), ())),
        )
        if actual != expected:
            raise FixtureError(
                f"pinned operator signature changed for {_callable_label(spec)}: expected {expected!r}, got {actual!r}"
            )


def _pinned_operator_ast_digests(
    source_text: Mapping[str, str],
    specs: Sequence[Mapping[str, object]] = _PINNED_OPERATOR_DIGEST_SPECS,
) -> dict[str, str]:
    trees: dict[str, ast.Module] = {}
    digests: dict[str, str] = {}
    for spec in specs:
        source_name = str(spec["source"])
        if source_name not in trees:
            try:
                trees[source_name] = ast.parse(source_text[source_name])
            except (KeyError, SyntaxError) as exc:
                raise FixtureError(f"pinned source cannot satisfy callable digest allowlist for {source_name}: {exc}") from exc
        node = _callable_node(trees[source_name], spec)
        label = _callable_label(spec)
        if node is None:
            raise FixtureError(f"pinned source is missing required callable {label}")
        body = ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        expected = _EXPECTED_OPERATOR_AST_SHA256.get(label)
        if expected is None:
            raise FixtureError(f"callable {label} has no immutable AST digest allowlist entry")
        if digest != expected:
            raise FixtureError(f"pinned callable AST changed for {label}: expected {expected}, got {digest}")
        digests[label] = digest
    return digests


def _trusted_operator_provenance(source_text: Mapping[str, str]) -> list[dict[str, object]]:
    trusted: list[dict[str, object]] = []
    for spec in UPSTREAM_SOURCE_SPECS:
        name = spec["name"]
        raw = source_text.get(name)
        if raw is None:
            raise FixtureError(f"pinned source text is missing {name}")
        digest = _sha256_bytes(raw.encode("utf-8"))
        if digest != spec["sha256"]:
            raise FixtureError(f"pinned source hash mismatch for {name}: expected {spec['sha256']}, got {digest}")
        trusted.append(
            {
                "name": name,
                "project": spec["project"],
                "commit": spec["commit"],
                "path": spec["path"],
                "url": _source_url(spec),
                "sha256": digest,
                "origin": "verified_pinned_spec",
            }
        )
    return trusted



class _GdnLayer:
    def __init__(self):
        self.conv_states = [None]
        self.recurrent_states = [None]
        self.record_past = False


class _GdnCache:
    def __init__(self, layer_count: int):
        self.layers = [_GdnLayer() for _ in range(layer_count)]

    def has_previous_state(self, layer_idx: int, state_idx: int = 0) -> bool:
        layer = self.layers[layer_idx]
        return layer.conv_states[state_idx] is not None if state_idx else layer.recurrent_states[0] is not None

    def update_conv_state(self, hidden_states: Any, layer_idx: int, *, conv_kernel_size: int, state_idx: int = 0) -> Any:
        self.layers[layer_idx].conv_states[state_idx] = hidden_states[..., -conv_kernel_size:].detach().clone()
        return hidden_states

    def update_recurrent_state(self, state: Any, layer_idx: int) -> None:
        self.layers[layer_idx].recurrent_states[0] = state.detach().clone() if state is not None else None
class _StreamingExpertParameter:
    def __init__(self, reader: Any):
        self._reader = reader

    def __getitem__(self, index: Any) -> Any:
        if hasattr(index, "item"):
            index = index.item()
        return self._reader(int(index))


class _StreamingNGram(types.SimpleNamespace):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class _StreamingExperts:
    def __init__(self, source_forward: Any, num_experts: int, hidden_dim: int, intermediate_dim: int, gate_reader: Any, down_reader: Any, act_fn: Any):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = _StreamingExpertParameter(gate_reader)
        self.down_proj = _StreamingExpertParameter(down_reader)
        self.act_fn = act_fn
        self.forward = types.MethodType(source_forward, self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class _StreamingEmbedding:
    def __init__(self, torch: Any, embedding_dim: int, reader: Any):
        self._torch = torch
        self._embedding_dim = embedding_dim
        self._reader = reader
        self.weight = types.SimpleNamespace(device=torch.device("cpu"))

    def __call__(self, ids: Any) -> Any:
        shape = tuple(int(item) for item in ids.shape)
        values = self._reader([int(item) for item in ids.reshape(-1).tolist()])
        return values.reshape(*shape, self._embedding_dim)


class PinnedQwen4Operators:
    """Adapters whose arithmetic is executed by the pinned upstream source."""

    def __init__(self, torch: Any, source_text: Mapping[str, str], source_provenance: Sequence[Mapping[str, object]]):
        del source_provenance
        trusted = _trusted_operator_provenance(source_text)
        _validate_pinned_operator_signatures(source_text)
        self._ast_digests = _pinned_operator_ast_digests(source_text)
        self.torch = torch
        self._source_provenance = {str(item["name"]): item for item in trusted}
        nn = torch.nn
        functional = nn.functional

        class _AttentionFunctions:
            @staticmethod
            def get_interface(_name: str, default: Any) -> Any:
                return default

        hf_globals = {
            "torch": torch,
            "nn": nn,
            "math": math,
            "F": functional,
            "ACT2FN": {
                "silu": functional.silu,
                "gelu": functional.gelu,
                "relu": functional.relu,
                "sigmoid": torch.sigmoid,
            },
            "ROPE_INIT_FUNCTIONS": {},
            "ALL_ATTENTION_FUNCTIONS": _AttentionFunctions,
            "maybe_autocast": lambda **_kwargs: nullcontext(),
        }
        names = (
            "_MASK64",
            "_SPLITMIX_GAMMA",
            "_SPLITMIX_M1",
            "_SPLITMIX_M2",
            "_PRIME_1",
            "_splitmix64",
            "_build_layer_multipliers",
            "_is_prime",
            "_find_nth_prime_after",
            "Qwen4ExpTextRotaryEmbedding",
            "Qwen4ExpTextRMSNorm",
            "Qwen4ExpTextRMSNormGated",
            "apply_mask_to_padding_states",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "l2norm",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "Qwen4ExpTextGatedDeltaNet",
            "rotate_half",
            "apply_rotary_pos_emb",
            "repeat_kv",
            "eager_attention_forward",
            "Qwen4ExpTextQSAIndexer",
            "Qwen4ExpTextAttention",
            "Qwen4ExpTextMLP",
            "Qwen4ExpTextExperts",
            "Qwen4ExpTextTopKRouter",
            "Qwen4ExpTextSparseMoeBlock",
            "Qwen4ExpTextGatedResidual",
            "Qwen4ExpTextNGramEmbedding",
            "Qwen4ExpTextPLELayer",
        )
        self._hf = _compile_source(source_text["transformers_modeling"], names, hf_globals)
        self._decoder_forward = _compile_method(
            source_text["transformers_modeling"],
            "Qwen4ExpTextDecoderLayer",
            "forward",
            {"torch": torch},
        )
        self._op_specs = {
            "hyperconnection": (
                ("transformers_modeling", "Qwen4ExpTextGatedResidual.__init__"),
                ("transformers_modeling", "Qwen4ExpTextGatedResidual.forward"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.__init__"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm._norm"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.forward"),
                ("transformers_modeling", "Qwen4ExpTextDecoderLayer.forward"),
            ),
            "gdn": (
                ("transformers_modeling", "Qwen4ExpTextGatedDeltaNet.__init__"),
                ("transformers_modeling", "Qwen4ExpTextGatedDeltaNet.forward"),
                ("transformers_modeling", "Qwen4ExpTextRMSNormGated.__init__"),
                ("transformers_modeling", "Qwen4ExpTextRMSNormGated.forward"),
                ("transformers_modeling", "torch_chunk_gated_delta_rule"),
                ("transformers_modeling", "torch_recurrent_gated_delta_rule"),
                ("transformers_modeling", "causal_conv1d_fn"),
                ("transformers_modeling", "causal_conv1d_update"),
                ("transformers_modeling", "l2norm"),
            ),
            "qsa": (
                ("transformers_modeling", "Qwen4ExpTextAttention.__init__"),
                ("transformers_modeling", "Qwen4ExpTextAttention.forward"),
                ("transformers_modeling", "Qwen4ExpTextQSAIndexer.__init__"),
                ("transformers_modeling", "Qwen4ExpTextQSAIndexer.forward"),
                ("transformers_modeling", "Qwen4ExpTextRotaryEmbedding.__init__"),
                ("transformers_modeling", "Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters"),
                ("transformers_modeling", "Qwen4ExpTextRotaryEmbedding.forward"),
                ("transformers_modeling", "Qwen4ExpTextRotaryEmbedding.apply_interleaved_mrope"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.__init__"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm._norm"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.forward"),
                ("transformers_modeling", "apply_rotary_pos_emb"),
                ("transformers_modeling", "rotate_half"),
                ("transformers_modeling", "repeat_kv"),
                ("transformers_modeling", "eager_attention_forward"),
            ),
            "moe": (
                ("transformers_modeling", "Qwen4ExpTextSparseMoeBlock.forward"),
                ("transformers_modeling", "Qwen4ExpTextTopKRouter.__init__"),
                ("transformers_modeling", "Qwen4ExpTextTopKRouter.forward"),
                ("transformers_modeling", "Qwen4ExpTextExperts.forward"),
                ("transformers_modeling", "Qwen4ExpTextMLP.__init__"),
                ("transformers_modeling", "Qwen4ExpTextMLP.forward"),
            ),
            "ple": (
                ("transformers_modeling", "_splitmix64"),
                ("transformers_modeling", "_build_layer_multipliers"),
                ("transformers_modeling", "_is_prime"),
                ("transformers_modeling", "_find_nth_prime_after"),
                ("transformers_modeling", "Qwen4ExpTextPLELayer.forward"),
                ("transformers_modeling", "Qwen4ExpTextPLELayer._short_conv"),
                ("transformers_modeling", "Qwen4ExpTextNGramEmbedding.forward"),
                ("transformers_modeling", "Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.__init__"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm._norm"),
                ("transformers_modeling", "Qwen4ExpTextRMSNorm.forward"),
            ),
        }
    def provenance(self) -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {}
        for operation, entries in self._op_specs.items():
            operation_rows: list[dict[str, object]] = []
            for source_name, callable_name in entries:
                identity = self._source_provenance.get(source_name)
                if identity is None:
                    raise FixtureError(f"missing source provenance for {source_name}")
                digest = self._ast_digests.get(callable_name)
                if digest is None:
                    raise FixtureError(f"callable {callable_name} has no verified AST digest")
                row = dict(identity)
                row["callable"] = callable_name
                row["ast_sha256"] = digest
                operation_rows.append(row)
            result[operation] = operation_rows
        return result

    def _config(self, raw: Mapping[str, object]) -> Any:
        config = types.SimpleNamespace(**dict(raw))
        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        if not hasattr(config, "attention_dropout"):
            config.attention_dropout = 0.0
        if not hasattr(config, "attention_bias"):
            config.attention_bias = False
        if not hasattr(config, "norm_topk_prob"):
            config.norm_topk_prob = True
        if not hasattr(config, "seed"):
            config.seed = 1234
        if not hasattr(config, "head_dim"):
            config.head_dim = config.hidden_size // config.num_attention_heads
        return config

    def _copy(self, destination: Any, source: Any) -> None:
        with self.torch.no_grad():
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    def _module(self, module: Any, hidden: Any) -> Any:
        return module.to(device=hidden.device, dtype=hidden.dtype).eval()

    def hyperconnection(self, hidden: Any, weights: Mapping[str, Any], cfg: Mapping[str, object], *, combine: bool) -> Any:
        module = self._module(
            self._hf["Qwen4ExpTextGatedResidual"](self._config(cfg), use_combine=combine),
            hidden,
        )
        self._copy(module.hc_norm.weight, weights["hc_norm"])
        self._copy(module.input_mix_weight_down.weight, weights["down"])
        self._copy(module.input_mix_weight_up.weight, weights["up"])
        if combine:
            self._copy(module.block_inject_weight.weight, weights["block"])
        return module(hidden)

    def inject(self, hyper_input: Any, block_output: Any, injection_weights: Any) -> Any:
        squeeze_batch = hyper_input.ndim == 2
        if squeeze_batch:
            hyper_input = hyper_input.unsqueeze(0)
            block_output = block_output.unsqueeze(0)
            injection_weights = injection_weights.unsqueeze(0)
        zero_injection = self.torch.zeros_like(injection_weights)

        def attention_connection(_hidden: Any) -> tuple[Any, Any, Any]:
            return block_output, hyper_input, injection_weights

        def linear_attention(value: Any, **_kwargs: Any) -> Any:
            return value

        def mlp_connection(value: Any) -> tuple[Any, Any, Any]:
            return value, value, zero_injection

        layer = types.SimpleNamespace(
            ple=None,
            layer_type="linear_attention",
            attn_hyper_connection=attention_connection,
            linear_attn=linear_attention,
            mlp_hyper_connection=mlp_connection,
            mlp=lambda _value: self.torch.zeros_like(block_output),
        )
        result = self._decoder_forward(
            layer,
            hyper_input,
            None,
            attention_mask=None,
            conv_mask=None,
            past_key_values=None,
            ple_input_ids=None,
        )
        return result.squeeze(0) if squeeze_batch else result

    def gdn(self, hidden: Any, weights: Mapping[str, Any], cfg: Mapping[str, object], layer: int) -> tuple[Any, Mapping[str, Any]]:
        module = self._module(self._hf["Qwen4ExpTextGatedDeltaNet"](self._config(cfg), layer), hidden)
        self._copy(module.in_proj_qkv.weight, weights["qkv"])
        self._copy(module.in_proj_z.weight, weights["z"])
        self._copy(module.in_proj_b.weight, weights["b"])
        self._copy(module.in_proj_a.weight, weights["a"])
        self._copy(module.conv1d.weight, weights["conv"])
        self._copy(module.A_log, weights["a_log"])
        self._copy(module.dt_bias, weights["dt_bias"])
        self._copy(module.norm.weight, weights["norm"])
        self._copy(module.out_proj.weight, weights["out"])
        cache = _GdnCache(layer + 1)
        output = module(hidden.unsqueeze(0), cache_params=cache, attention_mask=None).squeeze(0)
        state = cache.layers[layer]
        return output, {"recurrent": state.recurrent_states[0], "conv": state.conv_states[0]}

    def qsa(self, hidden: Any, weights: Mapping[str, Any], cfg: Mapping[str, object], layer: int) -> tuple[Any, Any]:
        module = self._module(self._hf["Qwen4ExpTextAttention"](self._config(cfg), layer), hidden)
        self._copy(module.q_proj.weight, weights["q"])
        self._copy(module.q_norm.weight, weights["q_norm"])
        self._copy(module.k_proj.weight, weights["k"])
        self._copy(module.k_norm.weight, weights["k_norm"])
        self._copy(module.v_proj.weight, weights["v"])
        self._copy(module.o_proj.weight, weights["out"])
        self._copy(module.indexer.index_qk_proj.weight, weights["index_qk"])
        self._copy(module.indexer.q_layernorm.weight, weights["index_q_norm"])
        self._copy(module.indexer.k_layernorm.weight, weights["index_k_norm"])
        rotary = self._module(self._hf["Qwen4ExpTextRotaryEmbedding"](self._config(cfg)), hidden)
        positions = self.torch.arange(hidden.shape[0], device=hidden.device, dtype=self.torch.long).unsqueeze(0)
        cos, sin = rotary(hidden.unsqueeze(0), positions)
        size = hidden.shape[0]
        causal = self.torch.triu(
            self.torch.ones((1, 1, size, size), device=hidden.device, dtype=self.torch.bool),
            diagonal=1,
        )
        mask = self.torch.where(
            causal,
            self.torch.full((), self.torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype),
            self.torch.zeros((), device=hidden.device, dtype=hidden.dtype),
        )
        output, attention = module(hidden.unsqueeze(0), (cos, sin), mask, past_key_values=None)
        return output.squeeze(0), attention

    def moe(
        self,
        hidden: Any,
        cfg: Mapping[str, object],
        load_full: Any,
        load_expert: Any,
        layer: int,
    ) -> Any:
        config = self._config(cfg)
        router = self._module(self._hf["Qwen4ExpTextTopKRouter"](config), hidden)
        self._copy(router.weight, load_full(f"model.language_model.layers.{layer}.mlp.gate.weight"))
        shared = self._module(
            self._hf["Qwen4ExpTextMLP"](config, intermediate_size=int(cfg["shared_expert_intermediate_size"])),
            hidden,
        )
        prefix = f"model.language_model.layers.{layer}.mlp"
        self._copy(shared.gate_proj.weight, load_full(f"{prefix}.shared_expert.gate_proj.weight"))
        self._copy(shared.up_proj.weight, load_full(f"{prefix}.shared_expert.up_proj.weight"))
        self._copy(shared.down_proj.weight, load_full(f"{prefix}.shared_expert.down_proj.weight"))
        shared_gate = self.torch.nn.Linear(int(cfg["hidden_size"]), 1, bias=False).to(
            device=hidden.device, dtype=hidden.dtype
        )
        self._copy(shared_gate.weight, load_full(f"{prefix}.shared_expert_gate.weight"))
        expert_forward = self._hf["Qwen4ExpTextExperts"].forward
        experts = _StreamingExperts(
            expert_forward,
            int(cfg["num_experts"]),
            int(cfg["hidden_size"]),
            int(cfg["moe_intermediate_size"]),
            lambda expert: load_expert(
                f"{prefix}.experts.gate_up_proj",
                expert,
                0,
                2 * int(cfg["moe_intermediate_size"]),
            ),
            lambda expert: load_expert(
                f"{prefix}.experts.down_proj",
                expert,
                0,
                int(cfg["hidden_size"]),
            ),
            self.torch.nn.functional.silu,
        )
        sparse = types.SimpleNamespace(
            shared_expert=shared,
            gate=router,
            experts=experts,
            shared_expert_gate=shared_gate,
        )
        return self._hf["Qwen4ExpTextSparseMoeBlock"].forward(sparse, hidden.unsqueeze(0)).squeeze(0)

    def ple(
        self,
        hidden: Any,
        token_ids: Sequence[int],
        cfg: Mapping[str, object],
        metadata: Mapping[str, Any],
        load_full: Any,
        read_rows: Any,
    ) -> Any:
        config = self._config(cfg)
        ngram_class = self._hf["Qwen4ExpTextNGramEmbedding"]
        ple_class = self._hf["Qwen4ExpTextPLELayer"]
        embedding_dim = int(cfg["ple_embed_dim"])
        heads = (int(cfg["ngram_size"]) - 1) * int(cfg["heads_per_ngram"])
        if embedding_dim % heads:
            raise FixtureError("pinned PLE embedding dimension is not divisible by its source head count")
        ngram = _StreamingNGram(
            layer_idx=1,
            ngram_size=int(cfg["ngram_size"]),
            context_len=int(cfg["ngram_size"]) - 1,
            heads_per_ngram=int(cfg["heads_per_ngram"]),
            ngram_heads=heads,
            ple_layer_index=0,
            unigram_vocab_size=int(cfg["vocab_size"]),
            ngram_vocab_size_base=int(cfg["ngram_vocab_size_base"]),
            ple_embed_dim=embedding_dim,
            seed=int(config.seed),
            eos_token_id=(
                int(cfg["eos_token_id"][0])
                if isinstance(cfg["eos_token_id"], list)
                else int(cfg["eos_token_id"])
            ),
            layer_multipliers=metadata["layer_multipliers"],
            ngram_heads_vocab_sizes=metadata["ngram_heads_vocab_sizes"],
            ngram_heads_offsets=metadata["ngram_heads_offsets"],
        )
        ngram.ngram_embedding = _StreamingEmbedding(
            self.torch,
            embedding_dim // heads,
            read_rows,
        )
        ngram._shift_right_ignore_eos = types.MethodType(
            ngram_class._shift_right_ignore_eos,
            ngram,
        )
        ngram.forward = types.MethodType(ngram_class.forward, ngram)

        hidden_size = int(cfg["hidden_size"])
        hc_count = int(cfg["hc_count"])
        hc_hidden_size = hidden_size * hc_count
        ple = types.SimpleNamespace(
            layer_idx=1,
            hidden_size=hidden_size,
            hc_count=hc_count,
            short_conv_state_len=(int(cfg["ple_conv_kernel_size"]) - 1) * int(cfg["ngram_size"]),
            ple_embedding=ngram,
        )
        norm_class = self._hf["Qwen4ExpTextRMSNorm"]
        ple.key_proj = self.torch.nn.Linear(embedding_dim, hc_hidden_size, bias=False)
        ple.value_proj = self.torch.nn.Linear(embedding_dim, hidden_size, bias=False)
        ple.norm_key = norm_class(hc_hidden_size, group_size=hidden_size, eps=float(cfg["rms_norm_eps"]))
        ple.norm_query = norm_class(hc_hidden_size, group_size=hidden_size, eps=float(cfg["rms_norm_eps"]))
        ple.norm_conv = norm_class(hc_hidden_size, group_size=hidden_size, eps=float(cfg["rms_norm_eps"]))
        ple.conv1d = self.torch.nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=int(cfg["ple_conv_kernel_size"]),
            groups=hc_hidden_size,
            dilation=int(cfg["ngram_size"]),
            bias=False,
        )
        ple.key_proj = self._module(ple.key_proj, hidden)
        ple.value_proj = self._module(ple.value_proj, hidden)
        ple.norm_key = self._module(ple.norm_key, hidden)
        ple.norm_query = self._module(ple.norm_query, hidden)
        ple.norm_conv = self._module(ple.norm_conv, hidden)
        ple.conv1d = self._module(ple.conv1d, hidden)
        self._copy(ple.key_proj.weight, load_full("model.language_model.layers.1.ple.key_proj.weight"))
        self._copy(ple.value_proj.weight, load_full("model.language_model.layers.1.ple.value_proj.weight"))
        self._copy(ple.norm_key.weight, load_full("model.language_model.layers.1.ple.norm_key.weight"))
        self._copy(ple.norm_query.weight, load_full("model.language_model.layers.1.ple.norm_query.weight"))
        self._copy(ple.norm_conv.weight, load_full("model.language_model.layers.1.ple.norm_conv.weight"))
        self._copy(ple.conv1d.weight, load_full("model.language_model.layers.1.ple.conv1d.weight"))
        ple._short_conv = types.MethodType(ple_class._short_conv, ple)
        output = ple_class.forward(
            ple,
            hidden.unsqueeze(0),
            self.torch.tensor([list(token_ids)], dtype=self.torch.long, device=hidden.device),
            None,
        )
        return output.squeeze(0)


def load_pinned_operators(
    source_text: Mapping[str, str],
    source_provenance: Sequence[Mapping[str, object]],
    torch: Any,
) -> PinnedQwen4Operators:
    _trusted_operator_provenance(source_text)
    return PinnedQwen4Operators(torch, source_text, source_provenance)

def _from_torch(value: Any, torch: Any) -> Tensor:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        words = value.view(torch.uint16).reshape(-1).tolist()
        return tensor(tuple(int(item) for item in value.shape), (int(item) for item in words), "uint16")
    if value.dtype == torch.float32:
        return tensor(tuple(int(item) for item in value.shape), (float(item) for item in value.reshape(-1).tolist()), "float32")
    if value.dtype == torch.float64:
        return tensor(tuple(int(item) for item in value.shape), (float(item) for item in value.reshape(-1).tolist()), "float64")
    if value.dtype == torch.int64:
        return tensor(tuple(int(item) for item in value.shape), (int(item) for item in value.reshape(-1).tolist()), "int64")
    if value.dtype == torch.int32:
        return tensor(tuple(int(item) for item in value.shape), (int(item) for item in value.reshape(-1).tolist()), "int32")
    if value.dtype == torch.uint16:
        return tensor(tuple(int(item) for item in value.shape), (int(item) for item in value.reshape(-1).tolist()), "uint16")
    if value.dtype == torch.bool:
        return tensor(tuple(int(item) for item in value.shape), (bool(item) for item in value.reshape(-1).tolist()), "bool")
    raise FixtureError(f"unsupported PyTorch output dtype {value.dtype}")


def _assert_finite_arrays(arrays: Mapping[str, Tensor], *, label: str) -> None:
    for name, value in arrays.items():
        if value.dtype in ("float32", "float64") and any(not math.isfinite(float(item)) for item in value.data):
            raise FixtureError(f"nonfinite {label} array {name}")
        if value.dtype == "uint16" and any((int(item) & 0x7F80) == 0x7F80 for item in value.data):
            raise FixtureError(f"nonfinite BF16 {label} array {name}")


def _assert_close(actual: Tensor, expected: Tensor, name: str, *, tolerance: str) -> None:
    if actual.shape != expected.shape:
        raise FixtureError(f"upstream mismatch for {name}: shape {actual.shape} != {expected.shape}")
    if actual.dtype in ("int32", "int64", "bool", "uint16") or expected.dtype in ("int32", "int64", "bool", "uint16"):
        if actual.dtype != expected.dtype or actual.data != expected.data:
            raise FixtureError(f"upstream mismatch for exact {name}")
        return
    bounds = TOLERANCES[tolerance]
    atol = float(bounds["atol"])
    rtol = float(bounds["rtol"])
    for index, (got, want) in enumerate(zip(actual.data, expected.data)):
        difference = abs(float(got) - float(want))
        if difference > atol + rtol * abs(float(want)):
            raise FixtureError(f"upstream mismatch for {name} at {index}: {got!r} != {want!r} ({tolerance})")


def _source_arrays(records: Mapping[str, Mapping[str, object]], family: str) -> dict[str, Tensor]:
    arrays: dict[str, Tensor] = {}
    for index, name in enumerate(CHECKPOINT_TENSORS[family]):
        record = records[name]
        value = record["values"]
        if not isinstance(value, Tensor):
            raise FixtureError(f"invalid checkpoint record {name}")
        arrays[f"source_{family}_{index}"] = value
    return arrays


def _generate_candidate_mtp(seed: int, moe_seed: int) -> dict[str, Tensor]:
    """Build the independent compact F32 MTP candidate used by parity.

    This intentionally mirrors the dependency-free generator's fixed streams
    without importing that module: the generator imports this upstream backend
    when invoked as a script, so a top-level cross-import would recurse.
    """

    rng = random.Random(seed)
    vocab = 32
    hidden = 8
    branches = 4
    tokens = [1, 5, 7, 9]
    embedding_words = bf16_tensor(
        rng_uniform(rng, (vocab, hidden), 0.5).data,
        (vocab, hidden),
    )
    backbone = rng_normal(rng, (len(tokens), branches * hidden), 0.6)
    fc_embedding = rng_normal(rng, (hidden, hidden), 0.08)
    fc_hidden = rng_normal(rng, (hidden, hidden), 0.08)
    projected = mtp_embedding_projection(
        tokens,
        embedding_words,
        backbone,
        fc_embedding,
        fc_hidden,
        branches,
        hidden,
        1e-6,
    )

    raw_q = rng_normal(rng, (len(tokens), 4, 4), 0.6)
    raw_keys = rng_normal(rng, (len(tokens), 2, 4), 0.6)
    index_queries = tensor(
        raw_q.shape,
        (abs(float(value)) + 0.19 + 0.005 * (index % 4) for index, value in enumerate(raw_q.data)),
        "float32",
    )
    raw_keys = tensor(
        raw_keys.shape,
        (abs(float(value)) + 0.13 + 0.009 * (index % 4) for index, value in enumerate(raw_keys.data)),
        "float32",
    )
    mtp_index = qsa_indexer(index_queries, raw_keys, list(range(len(tokens))), 4, 8, 4)
    later_reused = tensor(mtp_index["selected_indices"].shape, mtp_index["selected_indices"].data, "int32")

    queries = rng_normal(rng, (len(tokens), 4, 4), 0.6)
    keys = rng_normal(rng, (len(tokens), 2, 4), 0.6)
    values = rng_normal(rng, (len(tokens), 2, 4), 0.6)
    attention_gate = rng_normal(rng, (len(tokens), 4 * 4), 0.3)
    attention_projection = rng_normal(rng, (hidden, 4 * 4), 0.04)
    attention = qsa_attention(
        queries,
        keys,
        values,
        later_reused,
        attention_gate,
        attention_projection,
        list(range(len(tokens))),
        4,
    )

    norm_weight = rng_uniform(rng, (branches, hidden), 0.05)
    down = rng_normal(rng, (3, branches * hidden), 0.08)
    up = rng_normal(rng, (branches * hidden, 3), 0.08)
    inject_weight = rng_normal(rng, (branches, branches * hidden), 0.08)
    attn_prepared = hc_prepare(projected["residual_input"], down, up, norm_weight, branches, hidden, 1e-6)
    attn_for_inject = dict(attn_prepared)
    attn_for_inject["mixed"] = attention["output"]
    attn_injected = hc_inject(attn_for_inject, projected["residual_input"], inject_weight, branches)

    mlp_prepared = hc_prepare(attn_injected["injected"], down, up, norm_weight, branches, hidden, 1e-6)
    moe_rng = random.Random(moe_seed)
    moe_hidden = mlp_prepared["mixed"]
    router = rng_normal(moe_rng, (512, hidden), 0.07)
    gate_up = rng_normal(moe_rng, (512 * 2 * 8, hidden), 0.045).reshape(512, 2 * 8, hidden)
    down_moe = rng_normal(moe_rng, (512 * hidden, 8), 0.045).reshape(512, hidden, 8)
    shared_gate = rng_uniform(moe_rng, (hidden,), 0.08)
    shared_gate_up = rng_normal(moe_rng, (2 * 8, hidden), 0.045)
    shared_down = rng_normal(moe_rng, (hidden, 8), 0.045)
    moe = moe_top10(
        moe_hidden,
        router,
        gate_up,
        down_moe,
        shared_gate,
        shared_gate_up,
        shared_down,
        10,
    )
    mlp_for_inject = dict(mlp_prepared)
    mlp_for_inject["mixed"] = moe["output"]
    mlp_injected = hc_inject(mlp_for_inject, attn_injected["injected"], inject_weight, branches)

    final_down = rng_normal(rng, (3, branches * hidden), 0.08)
    final_up = rng_normal(rng, (branches * hidden, 3), 0.08)
    final = hc_final_mix(mlp_injected["injected"], norm_weight, branches, hidden, 1e-6, final_down, final_up)
    lm_head = rng_normal(rng, (vocab, hidden), 0.06)
    logits = matmul(final["mixed"], transpose2(lm_head))
    return {
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
        "attention_queries": queries,
        "attention_keys": keys,
        "attention_values": values,
        "attention_gate": attention_gate,
        "attention_projection": attention_projection,
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


def _run_pinned_operations(source_text: Mapping[str, str], checkpoint: Mapping[str, object], seed: int) -> tuple[list[tuple[str, dict[str, Tensor], dict[str, object]]], dict[str, object]]:
    try:
        packages = _import_pinned_packages()
    except FixtureError:
        raise
    torch = packages["modules"]["torch"]
    nn = torch.nn
    functional = nn.functional
    torch.manual_seed(seed)
    hf_globals = {
        "torch": torch,
        "nn": nn,
        "math": math,
        "F": functional,
        "ACT2FN": {"silu": functional.silu, "gelu": functional.gelu, "relu": functional.relu},
    }
    hf_names = (
        "_MASK64",
        "_SPLITMIX_GAMMA",
        "_SPLITMIX_M1",
        "_SPLITMIX_M2",
        "_PRIME_1",
        "_splitmix64",
        "_build_layer_multipliers",
        "_is_prime",
        "_find_nth_prime_after",
        "l2norm",
        "apply_mask_to_padding_states",
        "causal_conv1d_fn",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "rotate_half",
        "apply_rotary_pos_emb",
        "repeat_kv",
        "eager_attention_forward",
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextQSAIndexer",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextTopKRouter",
        "Qwen4ExpTextExperts",
        "Qwen4ExpTextMLP",
        "Qwen4ExpTextNGramEmbedding",
        "Qwen4ExpTextPLELayer",
    )
    hf = _compile_source(source_text["transformers_modeling"], hf_names, hf_globals)

    # PLE source execution uses a capture embedding, so construction never
    # allocates the real 320-million-row table.
    class CaptureEmbedding(nn.Module):
        def __init__(self, _rows: int, embedding_dim: int):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.weight = nn.Parameter(torch.zeros(1, embedding_dim), requires_grad=False)
            self.ids = None

        def forward(self, ids: Any) -> Any:
            self.ids = ids.detach().clone()
            return torch.zeros((*ids.shape, self.embedding_dim), dtype=torch.float32, device=ids.device)

    ple_config = types.SimpleNamespace(
        vocab_size=248320,
        ngram_vocab_size_base=20000000,
        ngram_size=3,
        heads_per_ngram=8,
        seed=1234,
        eos_token_id=EOS_TOKEN_ID,
        make_ngram_vocab_size_divisible_by=128,
    )
    original_embedding = nn.Embedding
    nn.Embedding = CaptureEmbedding
    try:
        ple = hf["Qwen4ExpTextNGramEmbedding"](ple_config, 2560, 1, 0)
    finally:
        nn.Embedding = original_embedding
    ple_tokens = [31, 47, 53, 59, 61, EOS_TOKEN_ID, 71, 73, 79, 83, EOS_TOKEN_ID, 89, 97]
    previous_context = torch.tensor([EOS_TOKEN_ID, EOS_TOKEN_ID], dtype=torch.long)
    token_input = torch.tensor(ple_tokens, dtype=torch.long)
    token_history_source = torch.cat((previous_context, token_input)).unsqueeze(0)
    shifted_source = torch.stack(
        [ple._shift_right_ignore_eos(token_history_source, shift) for shift in range(ple_config.ngram_size)]
    ).squeeze(1)
    source_token_history = _from_torch(token_history_source.squeeze(0), torch)
    source_shifted_tokens = _from_torch(shifted_source, torch)
    ple(torch.tensor([ple_tokens], dtype=torch.long), None)
    capture = ple.ngram_embedding
    if capture.ids is None:
        raise FixtureError("pinned PLE source did not expose n-gram IDs")
    source_ids = _from_torch(capture.ids.squeeze(0), torch)
    expected_ids = ple_hash_history(ple_tokens, [EOS_TOKEN_ID, EOS_TOKEN_ID], EOS_TOKEN_ID)["ple_row_ids"]
    _assert_close(source_ids, expected_ids, "PLE row IDs", tolerance="exact_integer_or_bytes")
    source_multipliers = _from_torch(ple.layer_multipliers, torch)
    source_sizes = _from_torch(ple.ngram_heads_vocab_sizes, torch)
    source_offsets = _from_torch(ple.ngram_heads_offsets, torch)
    _assert_close(source_multipliers, tensor((3,), NG_SCALE, "int64"), "PLE multipliers", tolerance="exact_integer_or_bytes")
    _assert_close(source_sizes, tensor((16,), NG_HEAD_VOCAB_SIZES, "int64"), "PLE vocab sizes", tolerance="exact_integer_or_bytes")
    _assert_close(source_offsets, tensor((16,), NG_HEAD_OFFSETS, "int64"), "PLE offsets", tolerance="exact_integer_or_bytes")

    records = checkpoint["tensors"]
    if not isinstance(records, dict):
        raise FixtureError("invalid checkpoint tensor records")
    ple_source = records[PLE_EMBEDDING]["values"]
    if not isinstance(ple_source, Tensor) or ple_source.dtype != "uint16" or ple_source.shape != (208, 160):
        raise FixtureError("pinned PLE source slice does not have the required BF16 shape")
    ple_rows = tensor((13, 16, 160), ple_source.data, "uint16")
    ple_arrays = {
        "tokens": tensor((len(ple_tokens),), ple_tokens, "int64"),
        "previous_context": tensor((2,), [EOS_TOKEN_ID, EOS_TOKEN_ID], "int64"),
        "token_history": source_token_history,
        "shifted_tokens": source_shifted_tokens,
        "multipliers": source_multipliers,
        "head_vocab_sizes": source_sizes,
        "head_offsets": source_offsets,
        "ple_row_ids": source_ids,
        "ple_rows_bf16": ple_rows,
        "ple_embedding_f32": bf16_to_f32(ple_rows),
        "source_layer_multipliers": source_multipliers,
        "source_head_vocab_sizes": source_sizes,
        "source_head_offsets": source_offsets,
        **_source_arrays(records, "ple"),
    }
    small_ple_config = types.SimpleNamespace(
        vocab_size=32,
        ngram_vocab_size_base=7,
        ngram_size=3,
        heads_per_ngram=8,
        seed=1234,
        eos_token_id=EOS_TOKEN_ID,
        make_ngram_vocab_size_divisible_by=8,
        ple_embed_dim=16,
        ple_conv_kernel_size=4,
        hc_count=4,
        hidden_size=8,
        rms_norm_eps=1e-6,
    )
    small_ple_layer = hf["Qwen4ExpTextPLELayer"](small_ple_config, 1, 0)
    small_ple_ids = torch.tensor([[1, 2, EOS_TOKEN_ID, 3]], dtype=torch.long)
    small_ple_hidden = torch.randn(1, 4, 32, dtype=torch.float32)
    with torch.no_grad():
        small_ple_embedding = small_ple_layer.ple_embedding(small_ple_ids, None)
        small_ple_key_raw = small_ple_layer.key_proj(small_ple_embedding)
        small_ple_key_normed = small_ple_layer.norm_key(small_ple_key_raw).unflatten(-1, (4, 8))
        small_ple_value = small_ple_layer.value_proj(small_ple_embedding)
        small_ple_query_normed = small_ple_layer.norm_query(small_ple_hidden).unflatten(-1, (4, 8))
        small_ple_gate = (
            (small_ple_key_normed * small_ple_query_normed).sum(dim=-1, keepdim=True) / math.sqrt(8.0)
        )
        small_ple_gate = small_ple_gate.abs().clamp_min(1.0e-6).sqrt() * small_ple_gate.sign()
        small_ple_gated = torch.sigmoid(small_ple_gate) * small_ple_value.unsqueeze(-2)
        small_ple_gated = small_ple_gated.flatten(-2)
        small_ple_gated_normed = small_ple_layer.norm_conv(small_ple_gated)
        small_ple_conv = small_ple_layer._short_conv(small_ple_gated_normed, None)
        small_ple_output = small_ple_gated + small_ple_conv
        small_ple_state = torch.zeros((9, 32), dtype=small_ple_gated_normed.dtype)
        small_ple_state[-4:] = small_ple_gated_normed.squeeze(0)
    ple_projection_arrays = {
        "ple_embedding_f32": _from_torch(small_ple_embedding.squeeze(0), torch),
        "hidden_states": _from_torch(small_ple_hidden.squeeze(0), torch),
        "key_proj": _from_torch(small_ple_layer.key_proj.weight, torch),
        "value_proj": _from_torch(small_ple_layer.value_proj.weight, torch),
        "norm_key_weight": _from_torch(small_ple_layer.norm_key.weight.reshape(4, 8), torch),
        "norm_query_weight": _from_torch(small_ple_layer.norm_query.weight.reshape(4, 8), torch),
        "norm_conv_weight": _from_torch(small_ple_layer.norm_conv.weight.reshape(4, 8), torch),
        "conv1d_weight": _from_torch(small_ple_layer.conv1d.weight.squeeze(1), torch),
        "key_normed": _from_torch(small_ple_key_normed.flatten(-2).squeeze(0), torch),
        "query_normed": _from_torch(small_ple_query_normed.flatten(-2).squeeze(0), torch),
        "gate": _from_torch(small_ple_gate.squeeze(0).squeeze(-1), torch),
        "value": _from_torch(small_ple_value.squeeze(0), torch),
        "gated_value": _from_torch(small_ple_gated.squeeze(0), torch),
        "gated_value_normed": _from_torch(small_ple_gated_normed.squeeze(0), torch),
        "conv_history": _from_torch(small_ple_state, torch),
        "conv_output": _from_torch(small_ple_conv.squeeze(0), torch),
        "output": _from_torch(small_ple_output.squeeze(0), torch),
    }
    ple_arrays.update(
        {
            "source_projection_hidden": _from_torch(small_ple_hidden.squeeze(0), torch),
            "source_projection_output": _from_torch(small_ple_output.squeeze(0), torch),
        }
    )

    # GDN recurrent/chunk equations come directly from the pinned HF module.
    tokens, key_heads, value_heads, kdim, vdim = 7, 16, 48, 4, 4
    query_16 = torch.randn(1, tokens, key_heads, kdim, dtype=torch.bfloat16)
    key_16 = torch.randn(1, tokens, key_heads, kdim, dtype=torch.bfloat16)
    value = torch.randn(1, tokens, value_heads, vdim, dtype=torch.bfloat16)
    a_logits = torch.randn(1, tokens, value_heads, dtype=torch.float32) * 0.4
    b_logits = torch.randn(1, tokens, value_heads, dtype=torch.float32) * 0.4
    dt_bias = torch.rand(value_heads, dtype=torch.float32) * 0.15
    a_log = torch.rand(value_heads, dtype=torch.float32) * 0.4
    g = -torch.exp(a_log).reshape(1, 1, value_heads) * functional.softplus(
        a_logits + dt_bias.reshape(1, 1, value_heads)
    )
    beta = torch.sigmoid(b_logits)
    repeat = value_heads // key_heads
    query = query_16.repeat_interleave(repeat, dim=2)
    key = key_16.repeat_interleave(repeat, dim=2)
    source_recurrent, source_state = hf["torch_recurrent_gated_delta_rule"](
        query, key, value, g, beta, None, True, use_qk_l2norm_in_kernel=True
    )
    source_chunk, chunk_state = hf["torch_chunk_gated_delta_rule"](
        query, key, value, g, beta, chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    source_query_l2_torch = hf["l2norm"](query, dim=-1, eps=1.0e-6)
    source_key_l2_torch = hf["l2norm"](key, dim=-1, eps=1.0e-6)
    source_query_scaled_torch = source_query_l2_torch.to(torch.float32) * (1.0 / math.sqrt(kdim))

    query_16_tensor = _from_torch(query_16.squeeze(0).float(), torch)
    key_16_tensor = _from_torch(key_16.squeeze(0).float(), torch)
    value_tensor = _from_torch(value.squeeze(0).float(), torch)
    query_expanded, key_expanded = gdn_expand_qk(query_16_tensor, key_16_tensor, value_heads)
    recurrent_candidate = gdn_recurrent(
        query_expanded,
        key_expanded,
        value_tensor,
        _from_torch(g.squeeze(0), torch),
        _from_torch(beta.squeeze(0), torch),
    )
    source_recurrent_tensor = _from_torch(source_recurrent.squeeze(0).float(), torch)
    source_chunk_tensor = _from_torch(source_chunk.squeeze(0).float(), torch)
    _assert_close(source_recurrent_tensor, recurrent_candidate["output"], "GDN recurrent output", tolerance="bf16_input_f32_accumulation")
    _assert_close(source_chunk_tensor, source_recurrent_tensor, "GDN chunk/recurrent output", tolerance="bf16_input_f32_accumulation")
    conv_channels = key_heads * kdim * 2 + value_heads * vdim
    conv_input = torch.randn(1, conv_channels, tokens, dtype=torch.bfloat16)
    conv_weight = torch.randn(conv_channels, 4, dtype=torch.bfloat16) * 0.025
    source_conv = hf["causal_conv1d_fn"](conv_input, conv_weight, activation="silu")
    conv_candidate = gdn_depthwise_conv(
        tensor((tokens, conv_channels), (float(item) for item in conv_input.squeeze(0).float().transpose(0, 1).reshape(-1).tolist()), "float32"),
        tensor((conv_channels, 4), (float(item) for item in conv_weight.float().reshape(-1).tolist()), "float32"),
        None,
        activation=True,
    )
    source_core_output_torch = source_recurrent.squeeze(0).float()
    candidate_core_output_torch = torch.tensor(
        recurrent_candidate["output"].data,
        dtype=torch.float32,
    ).reshape(tokens, value_heads, vdim)
    # HF casts the recurrent/chunk result back to hidden BF16 before the
    # RMSNormGated boundary.  Keep the raw F32 candidate for core parity, but
    # derive all downstream candidate expectations from an explicit BF16
    # storage roundtrip so the production device boundary is independently
    # measurable.
    candidate_bf16_storage = candidate_core_output_torch.to(torch.bfloat16)
    candidate_bf16_core_tensor = _from_torch(candidate_bf16_storage, torch)
    candidate_bf16_core_f32_tensor = _from_torch(candidate_bf16_storage.float(), torch)
    candidate_bf16_core_output_torch = candidate_bf16_storage.float()
    source_core_norm_raw_torch = source_core_output_torch * torch.rsqrt(
        source_core_output_torch.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    source_core_norm_storage = source_core_norm_raw_torch.to(torch.bfloat16)
    source_core_norm_torch = source_core_norm_storage.float()
    candidate_core_norm_raw_torch = candidate_bf16_core_output_torch * torch.rsqrt(
        candidate_bf16_core_output_torch.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    candidate_core_norm_storage = candidate_core_norm_raw_torch.to(torch.bfloat16)
    candidate_core_norm_torch = candidate_core_norm_storage.float()
    z_torch = torch.randn(tokens, value_heads * vdim, dtype=torch.float32) * 0.4
    source_output_gate_torch = source_core_norm_torch.reshape(tokens, value_heads * vdim) * torch.sigmoid(z_torch)
    # Kernel contract (gated_delta_gate_bf16_f32): z is rounded to BF16, the
    # sigmoid is 1/(1+exp(-z)) in F32, and the gated output is rounded to BF16.
    candidate_z_torch = z_torch.to(torch.bfloat16).float()
    candidate_output_gate_torch = (
        candidate_core_norm_torch.reshape(tokens, value_heads * vdim)
        / (1.0 + torch.exp(-candidate_z_torch))
    ).to(torch.bfloat16).float()
    out_proj_torch = torch.randn(8, value_heads * vdim, dtype=torch.float32) * 0.04
    source_output_torch = source_output_gate_torch @ out_proj_torch.transpose(0, 1)
    candidate_output_torch = candidate_output_gate_torch @ out_proj_torch.transpose(0, 1)
    source_conv_tensor = _from_torch(source_conv.squeeze(0).transpose(0, 1).float(), torch)
    _assert_close(source_conv_tensor, conv_candidate["output"], "GDN causal convolution", tolerance="bf16_input_f32_accumulation")
    gdn_arrays = {
        "query_16x4": _from_torch(query_16.squeeze(0).float(), torch),
        "key_16x4": _from_torch(key_16.squeeze(0).float(), torch),
        "query_expanded_48x4": _from_torch(query.squeeze(0).float(), torch),
        "key_expanded_48x4": _from_torch(key.squeeze(0).float(), torch),
        "source_query_l2": _from_torch(source_query_l2_torch.squeeze(0).float(), torch),
        "source_query_l2_bf16_storage": _from_torch(source_query_l2_torch.squeeze(0), torch),
        "source_key_l2": _from_torch(source_key_l2_torch.squeeze(0).float(), torch),
        "source_key_l2_bf16_storage": _from_torch(source_key_l2_torch.squeeze(0), torch),
        "source_query_scaled": _from_torch(source_query_scaled_torch.squeeze(0), torch),
        "source_value_f32": _from_torch(value.squeeze(0).float(), torch),
        "source_g_decay": _from_torch(g.squeeze(0), torch),
        "source_beta_f32": _from_torch(beta.squeeze(0), torch),
        "value_48x4": _from_torch(value.squeeze(0).float(), torch),
        "mixed_qkv": _from_torch(conv_input.squeeze(0).transpose(0, 1).float(), torch),
        "conv_weight": _from_torch(conv_weight.float(), torch),
        "initial_conv_history": _from_torch(torch.zeros((3, conv_channels), dtype=torch.float32), torch),
        "conv_output": conv_candidate["output"],
        "conv_final_history": conv_candidate["history"],
        "query_after_conv": _from_torch(query.squeeze(0).float(), torch),
        "key_after_conv": _from_torch(key.squeeze(0).float(), torch),
        "value_after_conv": _from_torch(value.squeeze(0).float(), torch),
        "a_logits": _from_torch(a_logits.squeeze(0), torch),
        "b_logits": _from_torch(b_logits.squeeze(0), torch),
        "a_log": _from_torch(a_log, torch),
        "dt_bias": _from_torch(dt_bias, torch),
        "g_decay": _from_torch(g.squeeze(0), torch),
        "beta_sigmoid": _from_torch(beta.squeeze(0), torch),
        "query_l2": recurrent_candidate["query_l2"],
        "key_l2": recurrent_candidate["key_l2"],
        "core_attention_output": source_recurrent_tensor,
        "final_recurrent_state": _from_torch(
            source_state.squeeze(0) if source_state is not None else torch.zeros((value_heads, kdim, vdim)),
            torch,
        ),
        "z_output_gate": _from_torch(z_torch, torch),
        "core_norm": _from_torch(source_core_norm_torch.reshape(tokens, value_heads * vdim), torch),
        "output_gate_sigmoid": _from_torch(source_output_gate_torch, torch),
        "out_proj": _from_torch(out_proj_torch, torch),
        "output": _from_torch(source_output_torch, torch),
        "source_core_attention_output": source_recurrent_tensor,
        "source_final_recurrent_state": _from_torch(
            source_state.squeeze(0) if source_state is not None else torch.zeros((value_heads, kdim, vdim)),
            torch,
        ),
        "source_core_norm": _from_torch(source_core_norm_torch.reshape(tokens, value_heads * vdim), torch),
        "source_core_norm_bf16_storage": _from_torch(source_core_norm_storage, torch),
        "source_output_gate_sigmoid": _from_torch(source_output_gate_torch, torch),
        "source_output": _from_torch(source_output_torch, torch),
        "candidate_core_attention_output": recurrent_candidate["output"],
        "candidate_final_recurrent_state": recurrent_candidate["final_state"],
        "candidate_core_norm": _from_torch(candidate_core_norm_torch.reshape(tokens, value_heads * vdim), torch),
        "candidate_core_norm_bf16_storage": _from_torch(candidate_core_norm_storage, torch),
        "candidate_output_gate_sigmoid": _from_torch(candidate_output_gate_torch, torch),
        "candidate_output": _from_torch(candidate_output_torch, torch),
        "g": _from_torch(g.squeeze(0), torch),
        "beta": _from_torch(beta.squeeze(0), torch),
        "source_recurrent_output": source_recurrent_tensor,
        "source_chunk_output": source_chunk_tensor,
        "source_recurrent_state": _from_torch(
            source_state.squeeze(0) if source_state is not None else torch.zeros((value_heads, kdim, vdim)),
            torch,
        ),
        "candidate_bf16_core_attention_output": candidate_bf16_core_tensor,
        "candidate_bf16_core_attention_output_f32": candidate_bf16_core_f32_tensor,
        "candidate_recurrent_output": recurrent_candidate["output"],
        **_source_arrays(records, "gdn"),
        "mixed_qkv_bf16": _from_torch(conv_input.squeeze(0).transpose(0, 1), torch),
        "conv_weight_bf16": _from_torch(conv_weight, torch),
        "source_conv_output": source_conv_tensor,
        "candidate_conv_output": conv_candidate["output"],
    }

    # HC source class and dependency-free candidate share precisely the same
    # zero-centred RMS/SiLU/sigmoid equation.
    hc_config = types.SimpleNamespace(hc_count=4, hidden_size=8, hc_lowrank=3, rms_norm_eps=1e-6)
    hc_module = hf["Qwen4ExpTextGatedResidual"](hc_config, use_combine=True)
    hyper = torch.randn(6, 32, dtype=torch.float32)
    norm_weight_t = torch.randn(4, 8, dtype=torch.float32) * 0.08
    down_t = torch.randn(3, 32, dtype=torch.float32) * 0.12
    up_t = torch.randn(32, 3, dtype=torch.float32) * 0.12
    inject_t = torch.randn(4, 32, dtype=torch.float32) * 0.12
    with torch.no_grad():
        hc_module.hc_norm.weight.copy_(norm_weight_t.reshape(-1))
        hc_module.input_mix_weight_down.weight.copy_(down_t)
        hc_module.input_mix_weight_up.weight.copy_(up_t)
        hc_module.block_inject_weight.weight.copy_(inject_t)
    source_mix, source_hyper, source_injection = hc_module(hyper)
    hyper_t = _from_torch(hyper, torch)
    hc_candidate = hc_prepare(
        hyper_t,
        _from_torch(down_t, torch),
        _from_torch(up_t, torch),
        _from_torch(norm_weight_t, torch),
        4,
        8,
        1e-6,
    )
    hc_candidate_injected = hc_inject(hc_candidate, hyper_t, _from_torch(inject_t, torch), 4)
    _assert_close(_from_torch(source_mix, torch), hc_candidate["mixed"], "HC mixed input", tolerance="f32_accumulation")
    _assert_close(_from_torch(source_injection, torch), hc_candidate_injected["injection_weight"], "HC injection", tolerance="f32_accumulation")
    final_hc_module = hf["Qwen4ExpTextGatedResidual"](hc_config, use_combine=False)
    with torch.no_grad():
        final_hc_module.hc_norm.weight.copy_(norm_weight_t.reshape(-1))
        final_hc_module.input_mix_weight_down.weight.copy_(down_t)
        final_hc_module.input_mix_weight_up.weight.copy_(up_t)
    source_final_mix = final_hc_module(hyper)
    candidate_final_mix = hc_final_mix(
        hyper_t,
        _from_torch(norm_weight_t, torch),
        4,
        8,
        1e-6,
        _from_torch(down_t, torch),
        _from_torch(up_t, torch),
    )
    _assert_close(_from_torch(source_final_mix, torch), candidate_final_mix["mixed"], "HC final mix", tolerance="f32_accumulation")
    hc_arrays = {
        "hyper_input": hyper_t,
        "hc_norm_weight_zero_centered": _from_torch(norm_weight_t, torch),
        "input_mix_weight_down": _from_torch(down_t, torch),
        "input_mix_weight_up": _from_torch(up_t, torch),
        "block_inject_weight": _from_torch(inject_t, torch),
        "normed": hc_candidate["normed"],
        "lowrank_silu": hc_candidate["lowrank_silu"],
        "mix_gate": hc_candidate["mix_gate"],
        "mixed": hc_candidate["mixed"],
        "injection_weight": hc_candidate_injected["injection_weight"],
        "injected": hc_candidate_injected["injected"],
        "final_input_mix_weight_down": _from_torch(down_t, torch),
        "final_input_mix_weight_up": _from_torch(up_t, torch),
        "final_normed": candidate_final_mix["normed"],
        "final_lowrank_silu": candidate_final_mix["lowrank_silu"],
        "final_mix_gate": candidate_final_mix["mix_gate"],
        "final_mixed": candidate_final_mix["mixed"],
        "source_mixed": _from_torch(source_mix, torch),
        "source_injection_weight": _from_torch(source_injection, torch),
        "candidate_mixed": hc_candidate["mixed"],
        "candidate_injected": hc_candidate_injected["injected"],
        "source_final_mixed": _from_torch(source_final_mix, torch),
        "candidate_final_mixed": candidate_final_mix["mixed"],
        **_source_arrays(records, "hc"),
    }

    # QSA indexer: full position-aware RoPE makes source and compact candidate
    # masks directly comparable while retaining pooling, block starts, and tail.
    qsa_config = types.SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=4,
        hidden_size=8,
        rms_norm_eps=1e-6,
    )
    qsa_module = hf["Qwen4ExpTextQSAIndexer"](qsa_config, 3)
    hidden = torch.rand(1, 13, 8, dtype=torch.float32) * 0.5 + 0.5
    with torch.no_grad():
        qsa_module.index_qk_proj.weight.copy_(torch.rand_like(qsa_module.index_qk_proj.weight) * 0.2 + 0.1)
    qsa_angles = torch.empty(13, 4, dtype=torch.float32)
    for position in range(13):
        for dimension in range(4):
            qsa_angles[position, dimension] = position / (10000000.0 ** (2.0 * (dimension % 2) / 4.0))
    cos = qsa_angles.cos().unsqueeze(0)
    sin = qsa_angles.sin().unsqueeze(0)
    visible = torch.tril(torch.ones(1, 1, 13, 13, dtype=torch.bool))
    source_qsa_mask = qsa_module(hidden, (cos, sin), visible, None)
    qk = qsa_module.index_qk_proj(hidden)
    q_raw, k_raw = qk.split((16, 4), dim=-1)
    q_norm = qsa_module.q_layernorm(q_raw.reshape(1, 13, 4, 4))
    q_indexer_rot = hf["apply_rotary_pos_emb"](q_norm, cos=cos, sin=sin, unsqueeze_dim=2)
    source_block_scores = torch.zeros(13, 4, dtype=torch.float32)
    for query_index in range(13):
        local_visible_indices = torch.nonzero(visible[0, 0, query_index], as_tuple=False).flatten()
        complete = (local_visible_indices.shape[-1] // 4) * 4
        if complete:
            block_tokens = local_visible_indices[:complete].view(-1, 4)
            key_groups = k_raw[0].index_select(0, block_tokens.flatten())
            key_groups = key_groups.view(*block_tokens.shape, 4)
            pooled_keys = key_groups.float().mean(dim=1).to(k_raw.dtype)
            pooled_keys = qsa_module.k_layernorm(pooled_keys)
            block_key_states = hf["apply_rotary_pos_emb"](
                pooled_keys.unsqueeze(1),
                cos=cos[0].index_select(0, block_tokens[:, 0]),
                sin=sin[0].index_select(0, block_tokens[:, 0]),
            ).squeeze(1)
            scores = torch.matmul(
                q_indexer_rot[0, query_index].float(),
                block_key_states.float().transpose(-1, -2),
            ).transpose(-1, -2)
            source_block_scores[query_index, : scores.shape[0]] = torch.relu(scores).sum(dim=-1) / math.sqrt(4)
    candidate_qsa = qsa_indexer(
        _from_torch(q_indexer_rot.squeeze(0), torch),
        _from_torch(k_raw.squeeze(0).unsqueeze(1), torch),
        list(range(13)),
        4,
        8,
        4,
    )
    _assert_close(
        _from_torch(source_block_scores, torch),
        candidate_qsa["block_scores"],
        "QSA block scores",
        tolerance="f32_accumulation",
    )
    rope_positions = [0, 1, 4, 7, 11]
    rope_value = torch.randn(5, 2, 8, dtype=torch.float32)
    rope_angles = torch.empty(5, 8, dtype=torch.float32)
    for row, position in enumerate(rope_positions):
        for dimension in range(8):
            rope_angles[row, dimension] = position / (10000000.0 ** (2.0 * (dimension % 4) / 8.0))
    source_rope = hf["apply_rotary_pos_emb"](
        rope_value,
        cos=rope_angles.cos(),
        sin=rope_angles.sin(),
        unsqueeze_dim=1,
    )
    candidate_rope = rope_half_split(
        _from_torch(rope_value, torch),
        rope_positions,
        8,
    )
    _assert_close(_from_torch(source_rope, torch), candidate_rope, "QSA HalfSplit RoPE", tolerance="f32_accumulation")

    attention_q = torch.randn(1, 4, 13, 4, dtype=torch.float32)
    attention_k = torch.randn(1, 2, 13, 4, dtype=torch.float32)
    attention_v = torch.randn(1, 2, 13, 4, dtype=torch.float32)
    attention_angles = torch.empty(13, 4, dtype=torch.float32)
    for position in range(13):
        for dimension in range(4):
            attention_angles[position, dimension] = position / (10000000.0 ** (2.0 * (dimension % 2) / 4.0))
    attention_cos = attention_angles.cos().unsqueeze(0)
    attention_sin = attention_angles.sin().unsqueeze(0)
    attention_q_rot, attention_k_rot = hf["apply_rotary_pos_emb"](
        attention_q,
        attention_k,
        cos=attention_cos,
        sin=attention_sin,
        unsqueeze_dim=1,
    )
    attention_mask = torch.where(
        visible & source_qsa_mask,
        torch.zeros_like(visible, dtype=torch.float32),
        torch.full_like(visible, torch.finfo(torch.float32).min, dtype=torch.float32),
    )
    attention_module = types.SimpleNamespace(num_key_value_groups=2, training=False)
    source_attention_head, source_attention_weights = hf["eager_attention_forward"](
        attention_module,
        attention_q_rot,
        attention_k_rot,
        attention_v,
        attention_mask,
        scaling=4.0 ** -0.5,
    )
    attention_gate = torch.randn(13, 16, dtype=torch.float32) * 0.3
    attention_projection = torch.randn(8, 16, dtype=torch.float32) * 0.04
    source_attention_gated = source_attention_head * torch.sigmoid(attention_gate).view(1, 13, 4, 4)
    source_attention_output = functional.linear(
        source_attention_gated.reshape(1, 13, 16),
        attention_projection,
    )
    candidate_attention = qsa_attention(
        _from_torch(attention_q.squeeze(0).transpose(0, 1), torch),
        _from_torch(attention_k.squeeze(0).transpose(0, 1), torch),
        _from_torch(attention_v.squeeze(0).transpose(0, 1), torch),
        candidate_qsa["selected_indices"],
        _from_torch(attention_gate, torch),
        _from_torch(attention_projection, torch),
        list(range(13)),
        4,
    )
    _assert_close(
        _from_torch(source_attention_output.squeeze(0), torch),
        candidate_attention["output"],
        "QSA attention and gate",
        tolerance="f32_accumulation",
    )
    source_token_mask = _from_torch(source_qsa_mask[0, 0], torch)
    _assert_close(source_token_mask, candidate_qsa["selected_token_mask"], "QSA selected token mask", tolerance="exact_integer_or_bytes")
    qsa_arrays = {
        "hidden": _from_torch(hidden.squeeze(0), torch),
        "positions": tensor((13,), range(13), "int64"),
        "index_queries": _from_torch(q_indexer_rot.squeeze(0), torch),
        "raw_index_keys": _from_torch(k_raw.squeeze(0), torch),
        "selected_indices": candidate_qsa["selected_indices"],
        "selected_indices_chunked": candidate_qsa["selected_indices"],
        "selected_indices_incremental": candidate_qsa["selected_indices"],
        "selected_mask": candidate_qsa["selected_mask"],
        "selected_token_mask": candidate_qsa["selected_token_mask"],
        "block_scores": candidate_qsa["block_scores"],
        "attention_queries": _from_torch(attention_q.squeeze(0).transpose(0, 1), torch),
        "attention_keys": _from_torch(attention_k.squeeze(0).transpose(0, 1), torch),
        "attention_values": _from_torch(attention_v.squeeze(0).transpose(0, 1), torch),
        "attention_gate": _from_torch(attention_gate, torch),
        "attention_output_projection": _from_torch(attention_projection, torch),
        "query_rope": _from_torch(attention_q_rot.squeeze(0).transpose(0, 1), torch),
        "key_rope": _from_torch(attention_k_rot.squeeze(0).transpose(0, 1), torch),
        "attention_weights": candidate_attention["attention_weights"],
        "attention_head_output": candidate_attention["head_output"],
        "attention_output": candidate_attention["output"],
        "rope_input": _from_torch(rope_value, torch),
        "rope_output": candidate_rope,
        "source_rope": _from_torch(source_rope, torch),
        "candidate_rope": candidate_rope,
        "source_attention_weights": _from_torch(source_attention_weights, torch),
        "source_attention_head": _from_torch(source_attention_head, torch),
        "source_attention_output": _from_torch(source_attention_output.squeeze(0), torch),
        "candidate_attention_output": candidate_attention["output"],
        "source_selected_token_mask": source_token_mask,
        "candidate_selected_indices": candidate_qsa["selected_indices"],
        "candidate_selected_mask": candidate_qsa["selected_mask"],
        "candidate_block_scores": candidate_qsa["block_scores"],
        "source_block_scores": _from_torch(source_block_scores, torch),
        **_source_arrays(records, "qsa"),
    }

    # Router and experts are executed from the pinned HF classes, then compared
    # with the compact top-10 implementation.
    moe_config = types.SimpleNamespace(
        num_experts=512,
        num_experts_per_tok=10,
        norm_topk_prob=True,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        intermediate_size=8,
        hidden_act="silu",
    )
    hidden_moe = torch.randn(5, 8, dtype=torch.float32)
    router_module = hf["Qwen4ExpTextTopKRouter"](moe_config)
    gate_up_t = torch.randn(512, 16, 8, dtype=torch.float32) * 0.045
    down_moe_t = torch.randn(512, 8, 8, dtype=torch.float32) * 0.045
    with torch.no_grad():
        router_weight_t = torch.randn(512, 8, dtype=torch.float32) * 0.07
        router_module.weight.copy_(router_weight_t)
    source_router_logits, source_scores, source_indices = router_module(hidden_moe)
    experts_module = hf["Qwen4ExpTextExperts"](moe_config)
    with torch.no_grad():
        experts_module.gate_up_proj.copy_(gate_up_t)
        experts_module.down_proj.copy_(down_moe_t)
    source_routed = experts_module(hidden_moe, source_indices, source_scores)
    shared_mlp = hf["Qwen4ExpTextMLP"](moe_config, intermediate_size=8)
    shared_gate_up_t = torch.randn(16, 8, dtype=torch.float32) * 0.045
    shared_down_t = torch.randn(8, 8, dtype=torch.float32) * 0.045
    with torch.no_grad():
        shared_mlp.gate_proj.weight.copy_(shared_gate_up_t[:8])
        shared_mlp.up_proj.weight.copy_(shared_gate_up_t[8:])
        shared_mlp.down_proj.weight.copy_(shared_down_t)
    source_shared = shared_mlp(hidden_moe)
    source_shared_gate_t = torch.randn(1, 8, dtype=torch.float32) * 0.08
    source_shared_gate = torch.sigmoid(functional.linear(hidden_moe, source_shared_gate_t))
    source_moe_output = source_routed + source_shared_gate * source_shared
    moe_candidate = moe_top10(
        _from_torch(hidden_moe, torch),
        _from_torch(router_weight_t, torch),
        _from_torch(gate_up_t, torch),
        _from_torch(down_moe_t, torch),
        _from_torch(source_shared_gate_t.squeeze(0), torch),
        _from_torch(shared_gate_up_t, torch),
        _from_torch(shared_down_t, torch),
        10,
    )
    _assert_close(_from_torch(source_router_logits, torch), moe_candidate["router_logits"], "MoE router logits", tolerance="f32_accumulation")
    source_indices_tensor = _from_torch(source_indices.to(torch.int32), torch)
    _assert_close(source_indices_tensor, moe_candidate["selected_experts"], "MoE selected experts", tolerance="exact_integer_or_bytes")
    _assert_close(_from_torch(source_scores, torch), moe_candidate["routing_weights"], "MoE routing weights", tolerance="f32_accumulation")
    _assert_close(_from_torch(source_moe_output, torch), moe_candidate["output"], "MoE output", tolerance="f32_accumulation")
    moe_arrays = {
        "hidden": _from_torch(hidden_moe, torch),
        "router_weight": _from_torch(router_weight_t, torch),
        "gate_up_weight": _from_torch(gate_up_t, torch),
        "down_weight": _from_torch(down_moe_t, torch),
        "shared_gate_weight": _from_torch(source_shared_gate_t.squeeze(0), torch),
        "shared_gate_up_weight": _from_torch(shared_gate_up_t, torch),
        "shared_down_weight": _from_torch(shared_down_t, torch),
        "router_logits": moe_candidate["router_logits"],
        "selected_experts": moe_candidate["selected_experts"],
        "routing_weights": moe_candidate["routing_weights"],
        "routed_output": moe_candidate["routed_output"],
        "shared_gate": moe_candidate["shared_gate"],
        "shared_output": moe_candidate["shared_output"],
        "output": moe_candidate["output"],
        "source_router_logits": _from_torch(source_router_logits, torch),
        "source_selected_experts": source_indices_tensor,
        "source_routing_weights": _from_torch(source_scores, torch),
        "source_output": _from_torch(source_moe_output, torch),
        "candidate_output": moe_candidate["output"],
        **_source_arrays(records, "moe"),
    }

    # Execute the exact vLLM native MTP forward method with a small, explicit
    # decoder adapter.  The method itself (embedding norms, fc_embedding,
    # shared fc_hidden, HC fold, and final mix call) is not reimplemented here.
    mtp_forward = _compile_method(
        source_text["vllm_mtp"],
        "Qwen4ExpMultiTokenPredictor",
        "forward",
        {
            "torch": torch,
            "nn": nn,
            "get_pp_group": lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
        },
    )

    class GroupedRMS(nn.Module):
        def __init__(self, size: int, eps: float):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size, dtype=torch.float32))
            self.eps = eps

        def forward(self, value: Any) -> Any:
            original_dtype = value.dtype
            value_f32 = value.float()
            result = value_f32 * torch.rsqrt(value_f32.square().mean(-1, keepdim=True) + self.eps)
            result = result * self.weight
            return result.to(original_dtype)

    class CaptureLayer:
        def __init__(self):
            self.input = None
            self.hc_mixed = None
            self.hc_injection = None
            self.qsa_mask = None
            self.moe_output = None

        def __call__(self, *, hidden_states: Any, **_kwargs: Any) -> Any:
            self.input = hidden_states.detach().clone()
            # The adapter's synthetic HC weights are F32; the pinned MTP
            # projections remain BF16, so run this captured HC branch in F32.
            mixed, _hyper, injection = hc_module(hidden_states.float())
            self.hc_mixed = mixed.detach().clone()
            self.hc_injection = injection.detach().clone()
            qsa_hidden = mixed.unsqueeze(0)
            qsa_visible = torch.tril(torch.ones(1, 1, hidden_states.shape[0], hidden_states.shape[0], dtype=torch.bool))
            qsa_cos = torch.ones(1, hidden_states.shape[0], 4, dtype=torch.float32)
            qsa_sin = torch.zeros(1, hidden_states.shape[0], 4, dtype=torch.float32)
            self.qsa_mask = qsa_module(qsa_hidden, (qsa_cos, qsa_sin), qsa_visible, None)[0, 0].detach().clone()
            logits, scores, indices = router_module(mixed)
            routed = experts_module(mixed, indices, scores)
            shared = shared_mlp(mixed)
            shared_gate = torch.sigmoid(functional.linear(mixed, source_shared_gate_t))
            self.moe_output = (routed + shared_gate * shared).detach().clone()
            output = self.moe_output.unsqueeze(1).expand(-1, 4, -1).reshape(hidden_states.shape[0], 32)
            return output, torch.zeros_like(output), injection.to(output.dtype)

    class CaptureMixer:
        def combine_and_mix(self, hidden_states: Any, block_output: Any, injection: Any) -> Any:
            streams = hidden_states.view(hidden_states.shape[0], 4, 8)
            return hidden_states, streams.mean(-2), None

    class CaptureMTP:
        pass

    mtp = CaptureMTP()
    mtp.num_mtp_layers = 1
    mtp.hc_count = 4
    mtp.hidden_size = 8
    mtp.pre_fc_norm_embedding = GroupedRMS(8, 1e-6)
    mtp.pre_fc_norm_hidden = GroupedRMS(32, 1e-6)
    mtp.fc_embedding = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    mtp.fc_hidden = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    mtp.layers = [CaptureLayer()]
    mtp.hyper_connection_mixer = CaptureMixer()
    embedding_words_t = torch.randn(32, 8, dtype=torch.bfloat16)
    token_ids = torch.tensor([1, 5, 7, 9], dtype=torch.long)
    backbone_t = torch.randn(4, 32, dtype=torch.bfloat16)
    mtp_inputs = embedding_words_t.index_select(0, token_ids)
    native_sample, native_multi = mtp_forward(
        mtp,
        input_ids=token_ids,
        positions=torch.arange(4, dtype=torch.long),
        hidden_states=backbone_t,
        inputs_embeds=mtp_inputs,
        spec_step_idx=0,
    )
    if mtp.layers[0].input is None:
        raise FixtureError("pinned vLLM MTP forward did not reach decoder layer")
    mtp_layer = mtp.layers[0]
    if mtp_layer.hc_mixed is None or mtp_layer.hc_injection is None or mtp_layer.qsa_mask is None or mtp_layer.moe_output is None:
        raise FixtureError("pinned MTP adapter did not execute HC/QSA/MoE operations")
    mtp_qsa_mask = _from_torch(mtp_layer.qsa_mask, torch)
    mtp_selected_values: list[int] = []
    mtp_mask_width = mtp_qsa_mask.shape[1]
    for row in range(mtp_qsa_mask.shape[0]):
        selected = [
            column
            for column in range(mtp_mask_width)
            if bool(mtp_qsa_mask.data[row * mtp_mask_width + column])
        ]
        if len(selected) > 11:
            raise FixtureError("pinned MTP QSA selection exceeds compact capacity")
        mtp_selected_values.extend(selected + [-1] * (11 - len(selected)))
    mtp_step_indices = tensor((4, 11), mtp_selected_values, "int32")
    local_mtp = mtp_embedding_projection(
        [1, 5, 7, 9],
        _from_torch(embedding_words_t, torch),
        _from_torch(backbone_t.float(), torch),
        _from_torch(mtp.fc_embedding.weight.float(), torch),
        _from_torch(mtp.fc_hidden.weight.float(), torch),
        4,
        8,
        1e-6,
    )
    native_residual = _from_torch(mtp.layers[0].input.float(), torch)
    _assert_close(native_residual, local_mtp["residual_input"], "native MTP residual input", tolerance="bf16_input_f32_accumulation")
    # Keep the pinned vLLM capture as a named source oracle.  The physical
    # runner consumes a separately generated F32 candidate below; preserving
    # both sets prevents the compact implementation fixture from replacing the
    # source expectation.
    source_mtp_arrays = {
        "token_ids": tensor((4,), [1, 5, 7, 9], "int64"),
        "token_embedding_bf16": _from_torch(embedding_words_t, torch),
        "backbone_hidden": _from_torch(backbone_t, torch),
        "native_residual_input": native_residual,
        "native_sample_hidden": _from_torch(native_sample.float(), torch),
        "native_multi_hidden": _from_torch(native_multi.float(), torch),
        "native_hc_mixed": _from_torch(mtp_layer.hc_mixed, torch),
        "native_hc_injection_weight": _from_torch(mtp_layer.hc_injection, torch),
        "native_qsa_selected_token_mask": mtp_qsa_mask,
        "native_mtp_step0_indices": mtp_step_indices,
        "native_mtp_later_reused_indices": mtp_step_indices,
        "native_moe_output": _from_torch(mtp_layer.moe_output, torch),
        "equation_residual_input": local_mtp["residual_input"],
        "equation_embedding_norm": local_mtp["embedding_norm"],
        "equation_hidden_norm": local_mtp["hidden_norm"],
    }
    candidate_mtp_arrays = _generate_candidate_mtp(seed + 107, seed + 89)
    mtp_arrays = {
        # Generic names are the independently generated F32 candidate consumed
        # by the production physical runner.
        **candidate_mtp_arrays,
        **{f"candidate_{name}": value for name, value in candidate_mtp_arrays.items()},
        **{name: value for name, value in source_mtp_arrays.items() if name.startswith("native_")},
        **{f"source_{name}": value for name, value in source_mtp_arrays.items()},
        **_source_arrays(records, "mtp"),
    }

    all_cases = [
        (
            "ple_hash_history",
            ple_arrays,
            {
                "case": "ple_hash_history",
                "equations": ["pinned Transformers NGramEmbedding forward", "signed-I64 hash IDs", "16 heads and exact BF16 source rows"],
                "source_execution": "transformers_modeling.Qwen4ExpTextNGramEmbedding.forward",
            },
        ),
        (
            "hc_prepare_inject_final_mix",
            hc_arrays,
            {
                "case": "hc_prepare_inject_final_mix",
                "equations": ["pinned Transformers GatedResidual forward", "zero-centered grouped RMS", "SiLU low-rank read and sigmoid mix", "four-branch injection"],
                "source_execution": "transformers_modeling.Qwen4ExpTextGatedResidual.forward",
            },
        ),
        (
            "ple_projection_dilated_conv",
            {
                **ple_projection_arrays,
                "source_ple_rows_bf16": ple_rows,
                "source_ple_embedding_f32": bf16_to_f32(ple_rows),
                "source_projection_hidden": ple_arrays["source_projection_hidden"],
                "source_projection_output": ple_arrays["source_projection_output"],
                **_source_arrays(records, "ple"),
            },
            {
                "case": "ple_projection_dilated_conv",
                "equations": ["pinned Transformers PLELayer projection", "key/value gating", "dilated depthwise short convolution", "range-read trained BF16 PLE bytes"],
                "source_execution": "transformers_modeling.Qwen4ExpTextPLELayer.forward",
            },
        ),
        (
            "gdn_recurrence_conv_head_expansion",
            gdn_arrays,
            {
                "case": "gdn_recurrence_conv_head_expansion",
                "equations": ["pinned Transformers recurrent gated-delta rule", "pinned Transformers chunk rule", "16 Q/K heads repeat_interleave to 48 V heads", "causal depthwise conv + SiLU", "upstream 1/sqrt(key_dim) query scale"],
                "source_execution": "transformers_modeling.Qwen4ExpTextGatedDeltaNet forward primitives + repeat_interleave",
                "oracle_sets": {
                    "source": ["source_core_attention_output", "source_final_recurrent_state", "source_core_norm", "source_output_gate_sigmoid", "source_output"],
                    "candidate": ["candidate_core_attention_output", "candidate_bf16_core_attention_output", "candidate_bf16_core_attention_output_f32", "candidate_final_recurrent_state", "candidate_core_norm", "candidate_output_gate_sigmoid", "candidate_output"],
                },
                "candidate_activation_boundary": "BF16 source Q/K raw values use BF16 product/reduction/rsqrt/output in l2norm, then promote to F32; query scale is a separate F32 multiply; BF16 source V promotes to F32; recurrent state remains F32; explicit F32->BF16->F32 cast before RMSNormGated; RMSNormGated rounds z and its gated output to BF16; BF16 storage words, promoted F32 candidate, raw source BF16, and strict raw F32 candidate remain separately reported",
            },
        ),
        (
            "qsa_pool_selection_mask_tail_rope",
            qsa_arrays,
            {
                "case": "qsa_pool_selection_mask_tail_rope",
                "equations": ["pinned Transformers QSAIndexer forward", "mean-pool complete blocks at block starts", "HalfSplit RoPE for current queries and block-start keys", "ReLU score/top-k with non-tied exact indices", "incomplete tail and sentinel selection mask", "eager GQA attention with KV repeat and sigmoid gate"],
                "source_execution": "transformers_modeling.Qwen4ExpTextQSAIndexer.forward + apply_rotary_pos_emb + eager_attention_forward",
            },
        ),
        (
            "moe_top10_normalized_shared",
            moe_arrays,
            {
                "case": "moe_top10_normalized_shared",
                "equations": ["pinned Transformers TopKRouter", "512 experts/top-10 probability renormalization", "shared expert added once"],
                "source_execution": "transformers_modeling.Qwen4ExpTextTopKRouter + TextExperts + TextMLP",
            },
        ),
        (
            "native_mtp_embedding_qsa_moe_hc",
            mtp_arrays,
            {
                "case": "native_mtp_embedding_qsa_moe_hc",
                "equations": ["pinned vLLM residual_linear_shared MTP forward", "BF16 embedding and hidden projections", "native HC read/injection, QSA selection, and MoE output reuse"],
                "source_execution": "vllm_mtp.Qwen4ExpMultiTokenPredictor.forward + pinned HC/QSA/MoE adapter",
                "oracle_sets": {
                    "source": sorted(f"source_{name}" for name in source_mtp_arrays),
                    "native": sorted(name for name in source_mtp_arrays if name.startswith("native_")),
                    "candidate": sorted(f"candidate_{name}" for name in candidate_mtp_arrays),
                },
                "candidate_activation_boundary": "independent deterministic F32 compact MTP equations; pinned vLLM source capture is retained under source_* and native_* arrays",
            },
        ),
    ]
    for _name, arrays, _schema in all_cases:
        _assert_finite_arrays(arrays, label=_name)
    return all_cases, {
        "operations": [schema["source_execution"] for _name, _arrays, schema in all_cases],
        "candidate_comparison": "independent stdlib equations checked only after pinned source execution",
        "tolerances": TOLERANCES,
        "seed": seed,
        "bf16_rationale": "BF16 checkpoint/input storage is promoted to F32 for reductions; output bound is two BF16 unit roundoffs before fixture serialization.",
        "contracts": {
            "ple": "signed-I64 IDs and exact uint16 BF16 rows, 16 heads x 160 width",
            "hc": "zero-centered grouped RMS with read/write/final mix",
            "gdn": "16 Q/K heads repeat_interleave to 48 V heads, causal conv and recurrent/chunk state",
            "qsa": "mean pooling, block-start HalfSplit RoPE, non-tied exact top-k, incomplete tail and sentinel mask, GQA attention and sigmoid gate",
            "moe": "512 experts, top-10 renormalization, shared expert once",
            "mtp": "native vLLM embedding/hidden projections and reused pinned HC/QSA/MoE outputs",
        },
    }


def _canonical_digest(manifest: Mapping[str, object]) -> str:
    fixtures = manifest.get("fixtures", [])
    source = manifest.get("reference_sources", [])
    checkpoint = manifest.get("checkpoint", {})
    payload = {
        "schema": manifest.get("schema"),
        "seed": manifest.get("generator", {}).get("seed") if isinstance(manifest.get("generator"), dict) else None,
        "fixtures": [{"name": item.get("name"), "sha256": item.get("sha256")} for item in fixtures if isinstance(item, dict)],
        "sources": [{"name": item.get("name"), "sha256": item.get("sha256")} for item in source if isinstance(item, dict)],
        "checkpoint": {
            "revision": checkpoint.get("revision") if isinstance(checkpoint, dict) else None,
            "index_sha256": checkpoint.get("index_sha256") if isinstance(checkpoint, dict) else None,
            "config_sha256": checkpoint.get("config_sha256") if isinstance(checkpoint, dict) else None,
            "tensors": {
                key: value.get("sha256")
                for key, value in sorted((checkpoint.get("tensors", {}) if isinstance(checkpoint, dict) else {}).items())
                if isinstance(value, dict)
            },
        },
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _write_upstream_once(
    destination: Path,
    seed: int,
    dependencies: Mapping[str, object],
    sources: Mapping[str, str],
    source_provenance: list[dict[str, object]],
    checkpoint: Mapping[str, object],
    cache_dir: Path,
) -> dict[str, object]:
    cases, execution = _run_pinned_operations(sources, checkpoint, seed)
    fixtures: list[dict[str, object]] = []
    for index, (name, arrays, schema) in enumerate(cases):
        path = destination / f"{name}.npz"
        digest = write_npz(path, arrays)
        fixtures.append({
            "name": name,
            "path": path.name,
            "sha256": digest,
            "seed": seed + (11 + index * 13),
            "schema": schema,
            "arrays": array_metadata(arrays),
        })
    records = checkpoint["tensors"]
    if not isinstance(records, dict):
        raise FixtureError("invalid checkpoint records")
    checkpoint_manifest: dict[str, object] = {
        "model": checkpoint["model"],
        "revision": checkpoint["revision"],
        "index_sha256": checkpoint["index_sha256"],
        "config_sha256": checkpoint["config_sha256"],
        "index_etag": checkpoint.get("index_etag"),
        "config_etag": checkpoint.get("config_etag"),
        "tensors": {},
    }
    tensor_manifest = checkpoint_manifest["tensors"]
    if not isinstance(tensor_manifest, dict):
        raise FixtureError("invalid tensor manifest")
    for family, names in CHECKPOINT_TENSORS.items():
        for name in names:
            record = records[name]
            tensor_manifest[name] = {
                "tensor": record["tensor"],
                "family": family,
                "shard": record["shard"],
                "url": record["url"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "slice_shape": record["slice_shape"],
                "byte_offset": record["byte_offset"],
                "byte_length": record["byte_length"],
                "data_offsets": record["data_offsets"],
                "sha256": record["sha256"],
                "header_sha256": record["header_sha256"],
                "etag": record.get("etag"),
                "source_identity": {
                    "model": HF_MODEL,
                    "revision": HF_CONFIG_COMMIT,
                    "repo_commit": record.get("repo_commit"),
                },
            }
    manifest: dict[str, object] = {
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
        "reference_sources": source_provenance,
        "generator": {
            "mode": "upstream_reference",
            "seed": seed,
            "python": platform.python_version(),
            "dependency_policy": "pinned_source_ast_torch_required",
            "packages": dependencies["metadata"],
            "cache_policy": "bounded_source_and_range_cache",
            "cache_root": "~/.hipfire/datasets/qwen4-upstream (or .codeinsight+research)",
            "deterministic_npz": True,
            "no_full_model_residency": True,
            "source_slice_max_bytes": MAX_TENSOR_SLICE_BYTES,
        },
        "equations": {
            "bf16_storage": "uint16 little-endian BF16 words; F32 promotion before reductions",
            "f32": "IEEE-754 binary32 output records",
            "ple": {"multipliers": list(NG_SCALE), "vocab_sizes": list(NG_HEAD_VOCAB_SIZES), "offsets": list(NG_HEAD_OFFSETS), "valid_rows": NG_VALID_ROWS, "padded_rows": NG_PADDED_ROWS, "eos_token_id": EOS_TOKEN_ID},
            "qsa_ties": "candidate rejects tied block scores; source top-k indices are compared byte-exactly on non-tied fixture scores",
            "tolerances": TOLERANCES,
            "upstream_execution": execution,
        },
        "checkpoint": checkpoint_manifest,
        "fixtures": fixtures,
    }
    manifest["input_output_digests"] = {
        "fixture_npz": {item["name"]: item["sha256"] for item in fixtures},
        "checkpoint_slices": {name: record["sha256"] for name, record in sorted(tensor_manifest.items())},
        "canonical": _canonical_digest(manifest),
    }
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    _validate_output(destination, manifest)
    return manifest


def _validate_output(destination: Path, manifest: Mapping[str, object]) -> None:
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise FixtureError("upstream manifest has no fixtures")
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise FixtureError("upstream manifest has malformed fixture")
        path = destination / str(fixture["path"])
        if sha256_file(path) != fixture["sha256"]:
            raise FixtureError(f"fixture digest changed after writing: {path.name}")
        arrays = read_npz(path)
        _assert_finite_arrays(arrays, label=path.name)
    if manifest.get("input_output_digests", {}).get("canonical") != _canonical_digest(manifest):
        raise FixtureError("upstream canonical content digest mismatch")


def _compare_output_dirs(first: Path, second: Path) -> None:
    first_files = sorted(item.name for item in first.iterdir() if item.is_file())
    second_files = sorted(item.name for item in second.iterdir() if item.is_file())
    if first_files != second_files:
        raise FixtureError("upstream deterministic self-validation changed the file set")
    for name in first_files:
        if sha256_file(first / name) != sha256_file(second / name):
            raise FixtureError(f"upstream deterministic self-validation changed {name}")


def generate_upstream(
    out: str | Path,
    seed: int = 3800,
    cache_dir: str | Path | None = None,
    *,
    self_validate: bool = True,
) -> dict[str, object]:
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    dependencies = _import_pinned_packages()
    sources, provenance, cache_root = load_pinned_sources(cache_dir)
    checkpoint = load_checkpoint_slices(cache_root)
    manifest = _write_upstream_once(destination, seed, dependencies, sources, provenance, checkpoint, cache_root)
    if self_validate:
        verify_dir = Path(tempfile.mkdtemp(prefix=".qwen4-upstream-verify-", dir=str(destination.parent)))
        try:
            _write_upstream_once(verify_dir, seed, dependencies, sources, provenance, checkpoint, cache_root)
            _compare_output_dirs(destination, verify_dir)
        finally:
            shutil.rmtree(verify_dir, ignore_errors=True)
    return manifest


def self_test(cache_dir: str | Path | None = None) -> None:
    """Exercise deterministic synthetic preservation and rejection guards."""
    # Imported lazily so this check also works when torch is intentionally
    # absent.  This is the documented escape hatch for validating the backend
    # plumbing on a clean checkout.
    try:
        from .generate_fixtures import generate
    except ImportError:
        from generate_fixtures import generate  # type: ignore
    root = _validated_cache_dir(cache_dir)
    temp_root = Path(tempfile.mkdtemp(prefix=".qwen4-oracle-self-test-", dir=str(root)))
    try:
        first = temp_root / "first"
        second = temp_root / "second"
        manifest_a = generate(first, 3800)
        manifest_b = generate(second, 3800)
        _compare_output_dirs(first, second)
        if manifest_a.get("schema") != "hipfire.qwen4.reference_oracle.v1" or manifest_b.get("schema") != manifest_a.get("schema"):
            raise FixtureError("synthetic mode changed schema during upstream self-test")
        _assert_finite_arrays({"ok": tensor((1,), [0.0], "float32")}, label="self-test")
        try:
            _assert_finite_arrays({"bad": tensor((1,), [float("nan")], "float32")}, label="self-test")
        except FixtureError:
            pass
        else:
            raise FixtureError("nonfinite rejection guard did not fire")
        fake_spec = {"name": "self-test", "project": "test/project", "commit": "a" * 40, "path": "x.py", "sha256": _sha256_bytes(b"correct")}
        source_path, meta_path = _source_cache_paths(temp_root, fake_spec)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"wrong")
        meta_path.write_text(json.dumps({"project": fake_spec["project"], "commit": fake_spec["commit"], "path": fake_spec["path"], "sha256": fake_spec["sha256"]}), encoding="utf-8")
        try:
            _load_cached_source(temp_root, fake_spec)
        except FixtureError:
            pass
        else:
            raise FixtureError("source hash rejection guard did not fire")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
