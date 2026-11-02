#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Complete tiny Qwen4Exp checkpoint profiles with per-op reference dumps.

Two developer-only profiles are produced on CPU from a *full model stack*
(embedding, GDN + QSA layers, hyper-connections, PLE, MLP/MoE, final HC mixer,
LM head, native MTP head), not from isolated operators:

``tiny-moe``
    ``equation_source = "pinned-upstream"``.  The real
    ``Qwen4ExpForConditionalGeneration`` from the hash-verified Transformers
    ``93c8b7b4`` source, BF16, eager attention, DynamicCache.

``tiny-dense-dev``
    ``equation_source = "non-upstream-dense"``.  The same pinned decoder
    layers (GDN/QSA/HC/PLE untouched) with the MLP slot replaced by
    :func:`equations.dense_ffn_non_upstream` at a declared
    ``intermediate_size``.  No dummy expert exists: the pinned
    ``Qwen4ExpTextSparseMoeBlock`` is swapped out *before* construction, so no
    router/expert/shared-expert tensor is ever created.  Pinned HF has no dense
    decoder (``num_experts <= 0`` is rejected and the decoder layer always
    builds the sparse block); nothing produced here is upstream evidence for
    dense execution.

The native MTP head executes the real ``Qwen4ExpMultiTokenPredictor.forward``
body from hash-verified vLLM ``e126687a`` (``amd/mtp.py``) around the pinned HF
decoder layer and HC mixer.  Pinned HF ignores ``^mtp.*`` checkpoint keys, so
the manifest labels this branch ``pinned-vllm-mtp-forward+hf-operators``.

Source verification: the exact bytes of the three Transformers files that
define the model (and the vLLM files used by MTP) are SHA-256 checked against
``upstream.UPSTREAM_SOURCE_SPECS`` *before* anything is executed, and the
verified bytes themselves (not whatever the import system would find) are what
get executed.  A matching version string is never sufficient.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    from . import equations
    from .schema import FixtureError, Tensor, array_metadata, sha256_file, tensor, write_json, write_npz
    from .upstream import (
        HF_TRANSFORMERS_COMMIT,
        HF_TRANSFORMERS_VERSION,
        UPSTREAM_SOURCE_SPECS,
        VLLM_COMMIT,
        _compile_method,
        _download_source,
        _from_torch,
        _load_cached_source,
        _sha256_bytes,
        _validated_cache_dir,
    )
except ImportError:  # direct execution / unittest discovery from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import equations  # type: ignore
    from schema import FixtureError, Tensor, array_metadata, sha256_file, tensor, write_json, write_npz  # type: ignore
    from upstream import (  # type: ignore
        HF_TRANSFORMERS_COMMIT,
        HF_TRANSFORMERS_VERSION,
        UPSTREAM_SOURCE_SPECS,
        VLLM_COMMIT,
        _compile_method,
        _download_source,
        _from_torch,
        _load_cached_source,
        _sha256_bytes,
        _validated_cache_dir,
    )


TINY_SCHEMA = "hipfire.qwen4.tiny_profile.v1"
PROFILE_IDS = ("tiny-moe", "tiny-dense-dev")
EQUATION_SOURCE_UPSTREAM = "pinned-upstream"
EQUATION_SOURCE_DENSE = equations.DENSE_FFN_EQUATION_SOURCE  # "non-upstream-dense"
MTP_EQUATION_SOURCE = "pinned-vllm-mtp-forward+hf-operators"
FIXED_TEXT = "hi<|endoftext|>to hi."
DEFAULT_ROPE_THETA = 10000.0
DEFAULT_MAX_POSITION_EMBEDDINGS = 64
_ROPE_OPTION_DEFAULTS = {"rope_theta": DEFAULT_ROPE_THETA, "max_position_embeddings": DEFAULT_MAX_POSITION_EMBEDDINGS}
# S1 absence schema.  Without PLE the checkpoint declares `text_config.ple_layer_ids = []` and carries none of these PLE
# scalars (plus no `qwen4_ple` / `hipfire_profile.ple` marker); without MTP the checkpoint has no `text_config.mtp`
# block, no `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings` and no `mtp.*` tensors.
PLE_COMPANION_KEYS = (
    "ple_embed_dim",
    "ple_conv_kernel_size",
    "ngram_size",
    "heads_per_ngram",
    "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by",
    "split_ngram_parts",
)
MTP_CONFIG_KEYS = ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "mtp")
PLE_ABSENT_REASON = "PLE is absent from this profile (with_ple=false): no PLE layer, tensors, buffers, config marker or state slots exist"
MTP_ABSENT_REASON = "native MTP is absent from this profile (with_mtp=false): no MTP head is constructed, no mtp.* tensors, config block or dump exist"
# Option values that are omitted from manifest["options"] (and keep default manifests byte-identical) while at their default.
_MANIFEST_OMITTED_DEFAULTS = {**_ROPE_OPTION_DEFAULTS, "with_ple": True, "with_mtp": True}
MAX_CALLS_PER_MODULE = 128
# Hard bound on instantiated parameter elements.  The 5120-wide knob passes
# because every projection other than the hidden axis is reduced; the two
# [H, H] MTP fusion projections (2 * 5120^2 ~ 52M elements, ~105 MB BF16) are
# the structural floor and dominate.
MAX_PARAMETER_ELEMENTS = 80_000_000

# The Transformers files whose bytes define the executed model.
_HF_SPEC_NAMES = ("transformers_configuration", "transformers_modeling", "transformers_modular")
_VLLM_SPEC_NAMES = ("vllm_mtp", "vllm_model", "vllm_common_hyperconnection")

# `self.<attr>` accesses the pinned vLLM MTP forward body is allowed to make.
# A drift in the pinned file changes this set and is rejected, not adapted.
_MTP_FORWARD_SELF_ATTRS = frozenset(
    {
        "hc_count",
        "hidden_size",
        "embed_input_ids",
        "pre_fc_norm_embedding",
        "fc_embedding",
        "pre_fc_norm_hidden",
        "fc_hidden",
        "num_mtp_layers",
        "layers",
        "hyper_connection_mixer",
    }
)


# ---------------------------------------------------------------------------
# Profile options and the frozen config envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileOptions:
    """Knobs of one tiny profile.  Defaults are the declared gate construction."""

    profile_id: str = "tiny-moe"
    seed: int = 3800
    hidden_size: int = 16
    num_layers: int = 2
    ffn_intermediate_size: int = 8  # dense intermediate_size / MoE moe_intermediate_size
    num_experts: int = 3
    top_k: int = 2
    shared_intermediate_size: int | None = 8
    norm_topk_prob: bool = True
    output_gate_type: str = "sigmoid"
    vocab_size: int = 32
    eos_token_id: int = 2
    # RoPE base and context length shared by the main text config and the cloned MTP config.  Defaults are the declared
    # gate construction; GPU-compatible variants override them (e.g. 10_000_000.0 / 1024) without a new profile id.
    rope_theta: float = DEFAULT_ROPE_THETA
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
    # Optional components.  Pinned HF supports a PLE-free model (`ple_layer_ids` empty); the native MTP head is built here
    # only when requested, so `with_mtp=False` constructs and allocates no MTP module, tensor, config block or dump.
    with_ple: bool = True
    with_mtp: bool = True

    def validate(self) -> None:
        if self.profile_id not in PROFILE_IDS:
            raise FixtureError(f"unknown tiny profile {self.profile_id!r}; expected one of {PROFILE_IDS}")
        if self.hidden_size <= 0 or self.hidden_size % 2:
            raise FixtureError("hidden_size must be a positive even integer (PLE embeds 16 over 2 n-gram heads)")
        if self.num_layers < 1:
            raise FixtureError("num_layers must be >= 1")
        if self.hidden_size > 512 and self.num_layers > 2:
            raise FixtureError(
                f"hidden_size {self.hidden_size} is bounded to 1-2 layers (memory guard); got {self.num_layers}"
            )
        if isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)):
            raise FixtureError("rope_theta must be a number")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise FixtureError("rope_theta must be a positive finite number")
        if isinstance(self.max_position_embeddings, bool) or not isinstance(self.max_position_embeddings, int):
            raise FixtureError("max_position_embeddings must be an integer")
        if self.max_position_embeddings < 1:
            raise FixtureError("max_position_embeddings must be >= 1")
        for knob in ("with_ple", "with_mtp"):
            if not isinstance(getattr(self, knob), bool):
                raise FixtureError(f"{knob} must be a bool")
        if self.ffn_intermediate_size <= 0:
            raise FixtureError("ffn_intermediate_size must be positive")
        if self.output_gate_type not in ("sigmoid", "silu"):
            raise FixtureError("output_gate_type must be sigmoid or silu")
        if self.eos_token_id < 0 or self.eos_token_id >= self.vocab_size:
            raise FixtureError("eos_token_id outside vocabulary")
        if self.profile_id == "tiny-moe":
            if self.shared_intermediate_size is None:
                raise FixtureError(
                    "shared_intermediate_size=null is not constructible from pinned HF: Qwen4ExpTextSparseMoeBlock "
                    "always builds a shared expert and its config requires shared_expert_intermediate_size > 0"
                )
            if self.shared_intermediate_size <= 0:
                raise FixtureError("shared_intermediate_size must be positive")
            if not 0 < self.top_k <= self.num_experts:
                raise FixtureError("top_k must be in [1, num_experts]")
            if self.top_k >= self.num_experts:
                raise FixtureError("top_k must be < num_experts so the router boundary is observable")
        elif self.hidden_size > 512 and self.ffn_intermediate_size > 64:
            raise FixtureError("wide-hidden profiles must keep reduced FFN dimensions (<= 64)")

    def with_overrides(self, **kwargs: Any) -> "ProfileOptions":
        values = {**self.__dict__, **kwargs}
        return ProfileOptions(**values)


def layer_types_for(num_layers: int) -> list[str]:
    """Checkpoint spelling (``full_attention``), interval 2: linear, full, linear, ..."""

    return ["linear_attention" if (index + 1) % 2 else "full_attention" for index in range(num_layers)]


CACHE_UNSUPPORTED_REASON = (
    "pinned DynamicCache.get_seq_length requires at least one attention (QSA) layer, so a linear-attention-only "
    "stack cannot run cached/chunked decode in pinned HF; states.npz then carries only the helper-capture "
    "derived states (captured.*), not DynamicCache states"
)


def has_attention_layer(options: ProfileOptions) -> bool:
    return "full_attention" in layer_types_for(options.num_layers)


def text_config_dict(options: ProfileOptions) -> dict[str, Any]:
    """Raw checkpoint-style ``text_config`` for the declared gate construction."""

    options.validate()
    config: dict[str, Any] = {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_size": options.hidden_size,
        "vocab_size": options.vocab_size,
        "num_hidden_layers": options.num_layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "max_position_embeddings": options.max_position_embeddings,
        "rope_parameters": {"rope_type": "default", "rope_theta": float(options.rope_theta), "partial_rotary_factor": 0.5},
        "layer_types": layer_types_for(options.num_layers),
        "full_attention_interval": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
        "mamba_ssm_dtype": "float32",
        "output_gate_type": options.output_gate_type,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "ple_embed_dim": 16,
        "ple_conv_kernel_size": 2,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ngram_vocab_size_base": 7,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 1,
        "seed": 1234,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "eos_token_id": options.eos_token_id,
        "tie_word_embeddings": False,
        "use_cache": True,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "mtp": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "hybrid": True,
            "rope_theta": float(options.rope_theta),
        },
    }
    if not options.with_ple:
        config["ple_layer_ids"] = []
        for key in PLE_COMPANION_KEYS:
            del config[key]
    if not options.with_mtp:
        for key in MTP_CONFIG_KEYS:
            del config[key]
    if options.profile_id == "tiny-moe":
        config.update(
            {
                "num_experts": options.num_experts,
                "num_experts_per_tok": options.top_k,
                "moe_intermediate_size": options.ffn_intermediate_size,
                "shared_expert_intermediate_size": options.shared_intermediate_size,
                "norm_topk_prob": options.norm_topk_prob,
            }
        )
    else:
        # The dense profile declares ONLY intermediate_size.  The MoE fields
        # HF's strict config validator insists on are supplied separately and
        # only to the in-memory HF config (see `_hf_text_config_dict`).
        config["intermediate_size"] = options.ffn_intermediate_size
    return config


def vision_config_dict(options: ProfileOptions) -> dict[str, Any]:
    """Tiny vision tower: part of the conditional-generation envelope, inventory-flagged, never executed."""

    return {
        "model_type": "qwen4_exp_vision",
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 8,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": options.hidden_size,
        "num_position_embeddings": 4,
    }


def profile_envelope(options: ProfileOptions) -> dict[str, Any]:
    """The frozen ``hipfire_profile`` root object."""

    options.validate()
    if options.profile_id == "tiny-moe":
        ffn: dict[str, Any] = {
            "kind": "moe",
            "num_experts": options.num_experts,
            "top_k": options.top_k,
            "intermediate_size": options.ffn_intermediate_size,
            "shared_intermediate_size": options.shared_intermediate_size,
            "norm_topk_prob": options.norm_topk_prob,
        }
        source = EQUATION_SOURCE_UPSTREAM
    else:
        ffn = {"kind": "dense", "intermediate_size": options.ffn_intermediate_size}
        source = EQUATION_SOURCE_DENSE
    return {"id": options.profile_id, "developer_only": True, "equation_source": source, "ffn": ffn}


def derived_ple_geometry(assembly: "Assembly") -> dict[str, Any] | None:
    """Stored PLE geometry read from the instantiated pinned ``Qwen4ExpTextNGramEmbedding`` (not constants).

    The config marker schema (``qwen4_ple`` + ``hipfire_profile.ple.valid_rows``) describes exactly one PLE table.
    A model without PLE (``with_ple=False``) has no table and no marker: returns ``None``.
    """

    tables = [layer.ple.ple_embedding for layer in assembly.text_model.layers if layer.ple is not None]
    if not assembly.options.with_ple:
        if tables:
            raise FixtureError(f"with_ple=False but the model instantiated {len(tables)} PLE table(s)")
        return None
    if len(tables) != 1:
        raise FixtureError(f"the config PLE marker describes exactly one PLE table; the model has {len(tables)}")
    table = tables[0]
    return {
        "valid_rows": int(table.total_vocab_size),
        "padded_rows": int(table.ngram_embedding.weight.shape[0]),
        "multipliers": [int(value) for value in table.layer_multipliers.tolist()],
        "head_vocab_sizes": [int(value) for value in table.head_vocab_sizes],
        "head_offsets": [int(value) for value in table.head_offsets],
    }


def full_config_dict(options: ProfileOptions, ple_geometry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``config.json``: outer envelope, inner text/vision config, ``hipfire_profile``.

    With ``ple_geometry`` (from :func:`derived_ple_geometry`) the root also carries the S1 PLE stored-geometry marker:
    ``hipfire_profile.ple = {"valid_rows": N}`` and top-level
    ``qwen4_ple = {"version": 1, "multipliers", "head_vocab_sizes", "head_offsets", "padded_rows"}``.  Every other
    PLE quantity (row width, shard count, heads) is derived from ``text_config``.
    """

    envelope = profile_envelope(options)
    config: dict[str, Any] = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "image_token_id": options.vocab_size - 1,
        "video_token_id": options.vocab_size - 2,
        "vision_start_token_id": options.vocab_size - 3,
        "vision_end_token_id": options.vocab_size - 4,
        "text_config": text_config_dict(options),
        "vision_config": vision_config_dict(options),
        "hipfire_profile": envelope,
    }
    if ple_geometry is not None:
        envelope["ple"] = {"valid_rows": int(ple_geometry["valid_rows"])}
        config["qwen4_ple"] = {
            "version": 1,
            "multipliers": list(ple_geometry["multipliers"]),
            "head_vocab_sizes": list(ple_geometry["head_vocab_sizes"]),
            "head_offsets": list(ple_geometry["head_offsets"]),
            "padded_rows": int(ple_geometry["padded_rows"]),
        }
    return config


def _hf_text_config_dict(options: ProfileOptions) -> dict[str, Any]:
    """What pinned HF's strict config validator needs (never written to disk)."""

    config = dict(text_config_dict(options))
    if options.profile_id == "tiny-dense-dev":
        del config["intermediate_size"]
        # Placeholders only satisfy `validate_architecture` (num_experts > 0).
        # The sparse block is replaced before construction: no module, tensor
        # or inventory entry derives from these values.
        config.update(
            {
                "num_experts": 1,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": options.ffn_intermediate_size,
                "shared_expert_intermediate_size": options.ffn_intermediate_size,
            }
        )
    return config


def estimated_parameter_elements(options: ProfileOptions) -> int:
    """Upper-bound parameter count used by the memory guard (hidden axis dominated)."""

    h = options.hidden_size
    hc = 2
    wide = hc * h
    per_layer = (
        4 * wide * 2 + 2 * wide  # two HC modules: down/up/inject + norms
        + 3 * wide + h * 64  # PLE keys/norms/conv/value (bounded)
        + 2 * h * 64  # attention / GDN projections (reduced widths)
        + 3 * h * options.ffn_intermediate_size  # dense or shared expert
        + options.num_experts * 3 * h * options.ffn_intermediate_size
    )
    # MTP: two [H, H] fusion projections (fc_embedding, fc_hidden) are the only
    # H^2 tensors the pinned structure forces; plus one reduced decoder layer.
    mtp = 2 * h * h + per_layer if options.with_mtp else 0
    return options.num_layers * per_layer + mtp + 3 * options.vocab_size * h + 4 * wide


# ---------------------------------------------------------------------------
# Hash-verified source loading
# ---------------------------------------------------------------------------


def _spec(name: str) -> Mapping[str, str]:
    for spec in UPSTREAM_SOURCE_SPECS:
        if spec["name"] == name:
            return spec
    raise FixtureError(f"no pinned source spec named {name}")


@dataclass
class VerifiedSources:
    """Hash-verified executable Transformers modules plus vLLM source texts."""

    config_module: Any
    modeling_module: Any
    vllm_text: dict[str, str]
    records: list[dict[str, object]]
    environment: dict[str, object]


def _verify_bytes(raw: bytes, spec: Mapping[str, str], origin: str) -> dict[str, object]:
    digest = _sha256_bytes(raw)
    if digest != spec["sha256"]:
        raise FixtureError(
            f"source hash mismatch for {spec['name']} at {origin}: expected {spec['sha256']}, got {digest}; "
            "refusing to execute (a matching version/commit string is not sufficient provenance)"
        )
    return {
        "name": spec["name"],
        "project": spec["project"],
        "commit": spec["commit"],
        "path": spec["path"],
        "sha256": digest,
        "pinned_sha256": spec["sha256"],
        "origin": origin,
        "verified_before_use": True,
    }


def _exec_verified_module(name: str, raw: bytes, origin: str) -> Any:
    module = types.ModuleType(name)
    module.__file__ = origin
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=origin)
    module.__hipfire_verified_sha256__ = _sha256_bytes(raw)
    sys.modules[name] = module
    try:
        exec(compile(raw, origin, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FixtureError(f"tiny profiles require PyTorch: {exc}") from exc
    if not hasattr(torch, "bfloat16"):
        raise FixtureError("PyTorch has no bfloat16 support")
    return torch


_SOURCES_CACHE: dict[tuple[str, str], VerifiedSources] = {}


def load_verified_sources(cache_dir: str | Path | None = None) -> VerifiedSources:
    """Verify every source file by SHA-256, then execute the verified bytes.

    * Transformers config/modeling/modular: read from the installed package
      (the same bytes the import system would use), hashed against
      ``UPSTREAM_SOURCE_SPECS``; config and modeling are then ``exec``-ed from
      the hashed bytes under their real module names.  A pre-existing
      unverified module with those names is refused.
    * vLLM ``mtp.py``/``model.py``/``common/hyperconnection.py``: loaded
      through the existing bounded cache (hash-checked on load; downloaded from
      the pinned URL only on a cache miss) and only *parsed/compiled*, never
      imported (vLLM is not installed and not constructible on CPU).
    """

    root = _validated_cache_dir(cache_dir)
    cache_key = (str(root), HF_TRANSFORMERS_COMMIT)
    if cache_key in _SOURCES_CACHE:
        return _SOURCES_CACHE[cache_key]
    torch = _torch()
    try:
        installed_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise FixtureError("transformers is not installed; the pinned model source cannot be located") from exc
    if installed_version != HF_TRANSFORMERS_VERSION:
        raise FixtureError(f"installed transformers {installed_version} != pinned {HF_TRANSFORMERS_VERSION}")
    package_spec = importlib.util.find_spec("transformers")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise FixtureError("transformers package location cannot be resolved")
    package_dir = Path(next(iter(package_spec.submodule_search_locations))) / "models" / "qwen4_exp"
    records: list[dict[str, object]] = []
    raw_by_name: dict[str, tuple[bytes, str]] = {}
    for name in _HF_SPEC_NAMES:
        spec = _spec(name)
        path = package_dir / Path(spec["path"]).name
        if not path.is_file():
            raise FixtureError(f"installed transformers lacks pinned file {path}")
        raw = path.read_bytes()
        records.append(_verify_bytes(raw, spec, f"installed:{path}"))
        raw_by_name[name] = (raw, str(path))
    init_path = package_dir / "__init__.py"
    init_sha = _sha256_bytes(init_path.read_bytes()) if init_path.is_file() else None

    modeling_name = "transformers.models.qwen4_exp.modeling_qwen4_exp"
    config_name = "transformers.models.qwen4_exp.configuration_qwen4_exp"
    for module_name, spec_name in ((config_name, "transformers_configuration"), (modeling_name, "transformers_modeling")):
        existing = sys.modules.get(module_name)
        if existing is not None and getattr(existing, "__hipfire_verified_sha256__", None) != _spec(spec_name)["sha256"]:
            raise FixtureError(
                f"{module_name} is already imported from unverified bytes; run the generator in a fresh interpreter"
            )
    importlib.import_module("transformers")
    importlib.import_module("transformers.models.qwen4_exp")  # lazy package: submodules not imported yet
    if config_name in sys.modules and sys.modules[config_name].__hipfire_verified_sha256__ == _spec("transformers_configuration")["sha256"]:
        config_module = sys.modules[config_name]
    else:
        config_module = _exec_verified_module(config_name, *raw_by_name["transformers_configuration"])
    if modeling_name in sys.modules and sys.modules[modeling_name].__hipfire_verified_sha256__ == _spec("transformers_modeling")["sha256"]:
        modeling_module = sys.modules[modeling_name]
    else:
        modeling_module = _exec_verified_module(modeling_name, *raw_by_name["transformers_modeling"])

    vllm_text: dict[str, str] = {}
    for name in _VLLM_SPEC_NAMES:
        spec = _spec(name)
        loaded = _load_cached_source(root, spec) or _download_source(root, spec)
        text, identity = loaded
        record = _verify_bytes(text.encode("utf-8"), spec, f"bounded_cache:{identity.get('cache_key')}")
        records.append(record)
        vllm_text[name] = text

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers_version": installed_version,
        "transformers_commit": HF_TRANSFORMERS_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "transformers_qwen4_exp_init_sha256": init_sha,
    }
    try:
        import safetensors

        environment["safetensors"] = safetensors.__version__
    except ImportError:  # pragma: no cover
        pass
    result = VerifiedSources(config_module, modeling_module, vllm_text, records, environment)
    _SOURCES_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Pinned vLLM MTP contract checks (parse-only)
# ---------------------------------------------------------------------------


def _function_node(text: str, class_name: str | None, function: str) -> ast.FunctionDef:
    tree = ast.parse(text)
    scope: list[ast.stmt] = tree.body
    if class_name is not None:
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if cls is None:
            raise FixtureError(f"pinned vLLM source is missing class {class_name}")
        scope = cls.body
    node = next((n for n in scope if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if node is None:
        raise FixtureError(f"pinned vLLM source is missing {class_name}.{function}")
    return node


def verify_vllm_mtp_contract(vllm_text: Mapping[str, str]) -> dict[str, object]:
    """Reject drift of the pinned vLLM MTP semantics this adapter relies on."""

    forward = _function_node(vllm_text["vllm_mtp"], "Qwen4ExpMultiTokenPredictor", "forward")
    used = {
        node.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    if used != _MTP_FORWARD_SELF_ATTRS:
        raise FixtureError(
            f"pinned vLLM MTP forward self-attribute set drifted: extra={sorted(used - _MTP_FORWARD_SELF_ATTRS)} "
            f"missing={sorted(_MTP_FORWARD_SELF_ATTRS - used)}"
        )
    body = ast.unparse(forward)
    for needle in (
        "inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)",
        "inputs_embeds = self.fc_embedding(inputs_embeds)",
        "hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(num_tokens, hc_count, hidden_size)",
        "hidden_states = self.fc_hidden(hidden_states)",
        "hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states",
        "self.hyper_connection_mixer.combine_and_mix(hidden_states, block_output, injection)",
        "return (sample_hidden_states, multi_hidden)",
    ):
        if needle not in body:
            raise FixtureError(f"pinned vLLM MTP forward no longer contains: {needle}")
    layer = ast.unparse(_function_node(vllm_text["vllm_model"], "Qwen4ExpDecoderLayer", "forward"))
    if "return (hidden_states, mlp_out, injection)" not in layer:
        raise FixtureError("pinned vLLM decoder layer no longer returns (hidden_states, mlp_out, injection)")
    common = ast.unparse(_function_node(vllm_text["vllm_common_hyperconnection"], "GatedResidual", "combine"))
    if "2.0 * torch.sigmoid(F.linear(hyper_input_normed, self.block_inject_weight.weight) / self.hc_count)" not in common:
        raise FixtureError("pinned vLLM HC combine equation drifted from the HF GatedResidual injection")
    if "output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)" not in common:
        raise FixtureError("pinned vLLM HC combine residual equation drifted")
    return {
        "forward_self_attributes": sorted(used),
        "decoder_layer_return": "(hidden_states, mlp_out, injection)",
        "hc_combine": "residual + block_output * 2*sigmoid(inject(hc_norm(hyper_input))/hc_count)",
    }


# ---------------------------------------------------------------------------
# Deterministic BF16 initialisation
# ---------------------------------------------------------------------------


def _name_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


_NORM_SUFFIXES = (
    "norm.weight",
    "layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "hc_norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def _init_tensor(torch: Any, name: str, shape: tuple[int, ...], seed: int, hidden: int) -> Any:
    """Name-keyed (so adding a tensor never perturbs another) float32 init."""

    generator = torch.Generator().manual_seed(_name_seed(seed, name))

    def normal(std: float) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * std

    if name.endswith("A_log"):
        return (torch.rand(shape, generator=generator, dtype=torch.float32) * 1.9 + 0.1).log()
    if name.endswith("dt_bias"):
        return normal(0.5)
    if name.endswith("linear_attn.norm.weight"):  # gated RMS norm multiplies by `weight` directly
        return 1.0 + normal(0.1)
    if name.endswith(_NORM_SUFFIXES):  # (1 + weight) norms
        return normal(0.1)
    if name.endswith("conv1d.weight"):
        return normal(0.4)
    if name.endswith("embed_tokens.weight") or name.endswith("ngram_embedding.weight"):
        return normal(0.7)
    if name.endswith("lm_head.weight"):
        return normal(1.5 / math.sqrt(hidden))
    if name.endswith("mlp.gate.weight"):  # router: logit std ~2 at any width keeps the top-k boundary and softmax tail observable
        return normal(2.0 / math.sqrt(max(shape[-1], 1)))
    fan_in = shape[-1] if shape else 1
    return normal(1.0 / math.sqrt(max(fan_in, 1)))


def _initialise(torch: Any, module: Any, seed: int, hidden: int, prefix: str = "") -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            full = f"{prefix}{name}"
            value = _init_tensor(torch, full, tuple(parameter.shape), seed, hidden)
            parameter.copy_(value.to(parameter.dtype))


# ---------------------------------------------------------------------------
# Dense FFN adapter (non-upstream) and torch module factories
# ---------------------------------------------------------------------------


def _to_pytensor(value: Any) -> Tensor:
    flat = value.detach().float().cpu().reshape(-1).tolist()
    return tensor(tuple(int(x) for x in value.shape), flat, "float32")


def _from_pytensor(value: Tensor, torch: Any) -> Any:
    return torch.tensor(list(value.data), dtype=torch.float32).reshape(value.shape)


def _make_dense_ffn_cls(torch: Any) -> type:
    nn = torch.nn

    class HipfireDenseFfn(nn.Module):
        """MLP slot for ``tiny-dense-dev``: NON-UPSTREAM, executed by ``equations.dense_ffn_non_upstream``."""

        equation_source = EQUATION_SOURCE_DENSE
        round_bf16 = True

        def __init__(self, config: Any, intermediate_size: int | None = None) -> None:
            super().__init__()
            size = intermediate_size if intermediate_size is not None else config.hipfire_dense_intermediate_size
            self.hidden_size = config.hidden_size
            self.intermediate_size = size
            # Same tensor names as Qwen4ExpTextMLP so the inventory reads naturally.
            self.gate_proj = nn.Linear(self.hidden_size, size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, size, bias=False)
            self.down_proj = nn.Linear(size, self.hidden_size, bias=False)
            self.hipfire_dump: dict[str, Tensor] = {}

        def forward(self, hidden_states: Any) -> Any:
            shape = hidden_states.shape
            flat = hidden_states.reshape(-1, self.hidden_size)
            result = equations.dense_ffn_non_upstream(
                _to_pytensor(flat),
                _to_pytensor(self.gate_proj.weight),
                _to_pytensor(self.up_proj.weight),
                _to_pytensor(self.down_proj.weight),
                self.intermediate_size,
                round_bf16=type(self).round_bf16,
            )
            self.hipfire_dump = result
            return _from_pytensor(result["output"], torch).to(hidden_states.dtype).reshape(shape)

    return HipfireDenseFfn


@contextlib.contextmanager
def _default_dtype(torch: Any, dtype: Any) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class _NeverInstance(type):
    """Metaclass: ``isinstance(x, SlotFactory)`` is always False."""

    def __instancecheck__(cls, instance: object) -> bool:
        return False


def _dense_slot_factory(dense_cls: type) -> type:
    """A callable standing in for ``Qwen4ExpTextSparseMoeBlock`` that builds the dense FFN.

    The pinned decoder layer builds ``Qwen4ExpTextSparseMoeBlock(config)``; the
    pinned ``_init_weights`` also dispatches on ``isinstance(module,
    Qwen4ExpTextSparseMoeBlock)`` and would otherwise treat the dense module as
    a router block.  The factory therefore never matches ``isinstance``.
    """

    class DenseSlot(metaclass=_NeverInstance):
        def __new__(cls, config: Any) -> Any:  # noqa: D102 - returns the dense module, not a DenseSlot
            return dense_cls(config)

    return DenseSlot


@contextlib.contextmanager
def _patched_sparse_block(sources: VerifiedSources, dense_cls: type | None) -> Iterator[None]:
    """Swap the MLP slot in the verified modeling module for the duration of construction."""

    module = sources.modeling_module
    if dense_cls is None:
        yield
        return
    original = module.Qwen4ExpTextSparseMoeBlock
    module.Qwen4ExpTextSparseMoeBlock = _dense_slot_factory(dense_cls)
    try:
        yield
    finally:
        module.Qwen4ExpTextSparseMoeBlock = original


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


@dataclass
class Assembly:
    options: ProfileOptions
    sources: VerifiedSources
    torch: Any
    model: Any
    config: Any
    mtp_config: Any | None  # None when the profile has no MTP head (nothing is constructed)
    mtp_head: Any | None
    effective_seed: int
    dense_cls: type | None

    @property
    def text_model(self) -> Any:
        return self.model.model.language_model


def _make_mtp_head_cls(torch: Any) -> type:
    nn = torch.nn

    class HipfireMtpHead(nn.Module):
        """Parameter container with the checkpoint's ``mtp.*`` names (embedding/LM head are shared with the main model)."""

        def __init__(self, sources: VerifiedSources, config: Any) -> None:
            super().__init__()
            modeling = sources.modeling_module
            hidden = config.hidden_size
            self.fc_embedding = nn.Linear(hidden, hidden, bias=False)
            self.fc_hidden = nn.Linear(hidden, hidden, bias=False)
            self.pre_fc_norm_embedding = modeling.Qwen4ExpTextRMSNorm(hidden, eps=config.rms_norm_eps)
            self.pre_fc_norm_hidden = modeling.Qwen4ExpTextRMSNorm(hidden * config.hc_count, eps=config.rms_norm_eps)
            self.layers = nn.ModuleList([modeling.Qwen4ExpTextDecoderLayer(config, 0)])
            self.hyper_connection_mixer = modeling.Qwen4ExpTextGatedResidual(config, use_combine=False)

    return HipfireMtpHead


def build_assembly(
    options: ProfileOptions,
    sources: VerifiedSources,
    *,
    seed: int | None = None,
    dtype: str = "bfloat16",
    overrides: Mapping[str, Any] | None = None,
) -> Assembly:
    options.validate()
    if estimated_parameter_elements(options) > MAX_PARAMETER_ELEMENTS:
        raise FixtureError("profile exceeds the bounded-memory parameter estimate")
    torch = _torch()
    effective_seed = options.seed if seed is None else seed
    cm = sources.config_module
    modeling = sources.modeling_module
    text_dict = _hf_text_config_dict(options)
    if overrides:
        text_dict.update(overrides)
    vision_dict = vision_config_dict(options)
    dense_cls = _make_dense_ffn_cls(torch) if options.profile_id == "tiny-dense-dev" else None
    config = cm.Qwen4ExpConfig(
        text_config=dict(text_dict),
        vision_config=dict(vision_dict),
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    if dense_cls is not None:
        config.text_config.hipfire_dense_intermediate_size = options.ffn_intermediate_size
    mtp_config = None
    if options.with_mtp:
        mtp_text = dict(text_dict)
        mtp_text.update(
            {
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "ple_layer_ids": [],
                "mtp": None,
            }
        )
        mtp_text.pop("mtp")
        mtp_config = cm.Qwen4ExpTextConfig(**mtp_text)
        mtp_config._attn_implementation = "eager"
        mtp_config._experts_implementation = "eager"
        if dense_cls is not None:
            mtp_config.hipfire_dense_intermediate_size = options.ffn_intermediate_size
    target = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    if dtype != "bfloat16":
        text_dict["dtype"] = "float32"
        config.dtype = "float32"
        config.text_config.dtype = "float32"
        if mtp_config is not None:
            mtp_config.dtype = "float32"
    head = None
    with _default_dtype(torch, target), _patched_sparse_block(sources, dense_cls):
        model = modeling.Qwen4ExpForConditionalGeneration(config)
        if mtp_config is not None:
            head = _make_mtp_head_cls(torch)(sources, mtp_config)
    model.eval()
    hidden = options.hidden_size
    _initialise(torch, model, effective_seed, hidden)
    if head is not None:
        head.eval()
        _initialise(torch, head, effective_seed, hidden, prefix="mtp.")
    ple_layers = [index for index, layer in enumerate(model.model.language_model.layers) if layer.ple is not None]
    if bool(ple_layers) != options.with_ple or bool(config.text_config.ple_layer_ids) != options.with_ple:
        raise FixtureError(
            f"with_ple={options.with_ple} but pinned HF built PLE layers {ple_layers} "
            f"for ple_layer_ids={list(config.text_config.ple_layer_ids)}"
        )
    assembly = Assembly(options, sources, torch, model, config, mtp_config, head, effective_seed, dense_cls)
    rope_evidence(assembly)  # refuses a main/MTP rope or context mismatch before any dump is produced
    return assembly


def rope_evidence(assembly: Assembly) -> dict[str, Any]:
    """RoPE / context values read from the instantiated pinned objects; refuses any main/MTP disagreement.

    The MTP step computes cos/sin from the main ``text_model.rotary_emb`` and feeds the cloned HF decoder layer, so
    the clone's own ``self_attn.config`` and the MTP config are checked against the same knobs, and the rotary
    ``inv_freq`` is recomputed from the stated ``rope_theta`` (not trusted from the JSON ``mtp.rope_theta`` field).
    """

    torch = assembly.torch
    options = assembly.options
    expected_theta = float(options.rope_theta)
    main = assembly.config.text_config
    mtp = assembly.mtp_config
    has_mtp = assembly.mtp_head is not None
    rotary = assembly.text_model.rotary_emb
    observed = {"main_text_config": (main.rope_parameters["rope_theta"], main.max_position_embeddings)}
    if has_mtp:
        layer_config = assembly.mtp_head.layers[0].self_attn.config
        observed["mtp_config"] = (mtp.rope_parameters["rope_theta"], mtp.max_position_embeddings)
        observed["mtp_layer_self_attn_config"] = (layer_config.rope_parameters["rope_theta"], layer_config.max_position_embeddings)
    observed["main_rotary_embedding_config"] = (rotary.config.rope_parameters["rope_theta"], rotary.config.max_position_embeddings)
    for where, (theta, positions) in observed.items():
        if float(theta) != expected_theta or int(positions) != options.max_position_embeddings:
            raise FixtureError(
                f"{where} has rope_theta={theta} max_position_embeddings={positions}; "
                f"expected {expected_theta} / {options.max_position_embeddings}"
            )
    partial = main.rope_parameters.get("partial_rotary_factor", 1.0)
    if has_mtp:
        if mtp.rope_parameters.get("partial_rotary_factor", 1.0) != partial:
            raise FixtureError("MTP config partial_rotary_factor differs from the main text config")
        if float(text_config_dict(options)["mtp"]["rope_theta"]) != expected_theta:
            raise FixtureError("config.json mtp.rope_theta differs from rope_theta")
    dim = int(main.head_dim * partial)
    inv_freq = 1.0 / (expected_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    if not torch.equal(rotary.inv_freq.detach().cpu(), inv_freq):
        raise FixtureError("main rotary inv_freq (shared by the MTP layer) does not match rope_theta")
    evidence = {
        "rope_theta": expected_theta,
        "max_position_embeddings": options.max_position_embeddings,
        "partial_rotary_factor": partial,
        "checked": sorted(observed) + (["config_json_mtp_rope_theta"] if has_mtp else []) + ["rotary_inv_freq_recomputed"],
    }
    if has_mtp:
        evidence["mtp_rotary_source"] = "main text_model.rotary_emb (cos/sin fed to the cloned HF decoder layer)"
    return evidence


# ---------------------------------------------------------------------------
# Per-op capture
# ---------------------------------------------------------------------------

_HOOK_CLASSES = frozenset(
    {
        "Linear",
        "Embedding",
        "Conv1d",
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextRMSNormGated",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextGatedDeltaNet",
        "Qwen4ExpTextQSAIndexer",
        "Qwen4ExpTextAttention",
        "Qwen4ExpTextTopKRouter",
        "Qwen4ExpTextExperts",
        "Qwen4ExpTextMLP",
        "Qwen4ExpTextSparseMoeBlock",
        "Qwen4ExpTextNGramEmbedding",
        "Qwen4ExpTextPLELayer",
        "Qwen4ExpTextDecoderLayer",
        "Qwen4ExpTextRotaryEmbedding",
        "HipfireDenseFfn",
    }
)


_TAPPED_FUNCTIONS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "eager_attention_forward",
)

# Helpers whose final recurrent state is recorded even when the (cache-less) model does not request it.
_FORCED_FINAL_STATE = frozenset({"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"})


class Capture:
    """Forward-hook recorder.  Keys: ``op.<module path>.<in|out><k>`` (``.call<n>`` after the first call)."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.records: dict[str, Any] = {}
        self.calls: dict[str, int] = {}
        self.enabled = True
        self.scope: str | None = None
        self._handles: list[Any] = []

    def _store(self, path: str, kind: str, value: Any, call: int) -> None:
        stem = f"op.{path}" if call == 0 else f"op.{path}.call{call}"
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self._store_one(f"{stem}.{kind}{index}", item)
        else:
            self._store_one(f"{stem}.{kind}0", value)

    def _store_one(self, key: str, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self._store_one(f"{key}.{index}", item)
            return
        if value is None or not hasattr(value, "detach"):
            return
        self.records[key] = value.detach().clone().cpu()

    def _next_call(self, path: str, tag: str) -> int:
        key = f"{path}#{tag}"
        index = self.calls.get(key, 0)
        if index >= MAX_CALLS_PER_MODULE:
            raise FixtureError(f"module {path} called more than {MAX_CALLS_PER_MODULE} times in one capture")
        self.calls[key] = index + 1
        return index

    def attach(self, root: Any, prefix: str, *, skip: Sequence[str] = ()) -> None:
        for name, module in root.named_modules():
            path = f"{prefix}{name}" if name else prefix.rstrip(".")
            if not path or any(fragment in path for fragment in skip):
                continue
            cls_name = type(module).__name__
            if cls_name in _HOOK_CLASSES:
                self._handles.append(module.register_forward_hook(self._hook(path)))
            if cls_name == "Qwen4ExpTextDecoderLayer":
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(path)))
            if cls_name in ("Qwen4ExpTextGatedDeltaNet", "Qwen4ExpTextAttention"):
                self._handles.append(module.register_forward_pre_hook(self._scope_enter(path)))
                self._handles.append(module.register_forward_hook(self._scope_exit))
            if path.endswith("ple_embedding.ngram_embedding") and cls_name == "Embedding":
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(path)))
            if path.endswith("hyper_connection_mixer"):
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(path)))
            if path.endswith(".ple.conv1d") and cls_name == "Conv1d":
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(path)))

    def _hook(self, path: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple, output: Any) -> None:
            if not self.enabled:
                return
            call = self._next_call(path, "out")
            self._store(path, "out", output, call)
            dump = getattr(module, "hipfire_dump", None)
            if dump:
                for name, value in dump.items():
                    stem = f"op.{path}" if call == 0 else f"op.{path}.call{call}"
                    self.records[f"{stem}.{name}"] = _from_pytensor(value, self.torch)

        return hook

    def _pre_hook(self, path: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple) -> None:
            if not self.enabled:
                return
            call = self._next_call(path, "in")
            self._store(path, "in", args[0] if args else None, call)

        return hook

    def _scope_enter(self, path: str) -> Callable[..., None]:
        def hook(module: Any, args: tuple) -> None:
            self.scope = path

        return hook

    def _scope_exit(self, module: Any, args: tuple, output: Any) -> None:
        self.scope = None

    @contextlib.contextmanager
    def tap_functions(self, module: Any, names: Sequence[str]) -> Iterator[None]:
        """Record inputs/outputs of module-level helper functions (GDN conv/recurrence, eager attention).

        The helpers are looked up as globals of the verified modeling module at
        call time, so wrapping them observes the exact arithmetic without
        altering it.  Records are keyed by the enclosing GDN/attention scope.
        """

        originals = {name: getattr(module, name) for name in names}

        def make(name: str) -> Callable[..., Any]:
            original = originals[name]

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Without a cache the model asks for no final recurrent state.  Compute it anyway
                # (``output_final_state`` only selects what is returned, not the arithmetic), record it,
                # and hand the model exactly the tuple it asked for.
                forced = name in _FORCED_FINAL_STATE and not kwargs.get("output_final_state")
                result = original(*args, **({**kwargs, "output_final_state": True} if forced else kwargs))
                recorded = result
                if forced:
                    result = (result[0], None)
                if self.enabled and self.scope is not None:
                    stem = f"{self.scope}.fn.{name}"
                    call = self._next_call(stem, "fn")
                    base = f"op.{stem}" if call == 0 else f"op.{stem}.call{call}"
                    for index, item in enumerate(args):
                        self._store_one(f"{base}.in{index}", item)
                    for key, item in kwargs.items():
                        self._store_one(f"{base}.kw.{key}", item)
                    self._store_one(f"{base}.out", recorded)
                return result

            return wrapper

        for name in names:
            setattr(module, name, make(name))
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(module, name, original)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _state_slots(container: Any) -> list[tuple[int, Any]]:
    """Linear-attention cache states are slot-keyed dicts (0 = GDN, 1 = PLE conv, 2 = PLE token context)."""

    if container is None:
        return []
    items = container.items() if hasattr(container, "items") else enumerate(container)
    return [(int(slot), value) for slot, value in items]


def _cache_state_dump(cache: Any, prefix: str) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for index, layer in enumerate(cache.layers):
        for attribute in ("keys", "values", "indexer_keys"):
            value = getattr(layer, attribute, None)
            if value is not None and hasattr(value, "numel") and value.numel() > 0:
                states[f"{prefix}layer{index}.{attribute}"] = value.detach().clone().cpu()
        for attribute, label in (("conv_states", "conv_state"), ("recurrent_states", "recurrent_state")):
            for slot, value in _state_slots(getattr(layer, attribute, None)):
                if value is not None and hasattr(value, "numel") and value.numel() > 0:
                    states[f"{prefix}layer{index}.{label}{slot}"] = value.detach().clone().cpu()
    return states


# ---------------------------------------------------------------------------
# Forward runs
# ---------------------------------------------------------------------------


@dataclass
class ForwardRun:
    dumps: dict[str, Any]
    logits: Any
    pre_mixer: Any  # [1, T, hc*H] multi-stream before the final HC mixer
    token_ids: list[int]


def run_forward(assembly: Assembly, tokens: Sequence[int]) -> ForwardRun:
    """Stateless full forward with every op captured (reference for ``forward.npz``)."""

    torch = assembly.torch
    capture = Capture(torch)
    capture.attach(assembly.text_model, "model.language_model.", skip=(".visual.",))
    capture.attach(assembly.model.lm_head, "lm_head")
    try:
        with torch.no_grad(), capture.tap_functions(assembly.sources.modeling_module, _TAPPED_FUNCTIONS):
            ids = torch.tensor([list(tokens)], dtype=torch.long)
            output = assembly.model(input_ids=ids, use_cache=False)
    finally:
        capture.detach()
    logits = output.logits.detach().clone().cpu()
    dumps = dict(capture.records)
    dumps["input.token_ids"] = ids.clone()
    dumps["output.logits"] = logits
    mixer_key = "op.model.language_model.hyper_connection_mixer.in0"
    pre_mixer = dumps[mixer_key]
    return ForwardRun(dumps, logits, pre_mixer, list(tokens))


_GDN_RECURRENT_KEY = re.compile(r"^op\.model\.language_model\.layers\.(\d+)\.linear_attn\.fn\.torch_chunk_gated_delta_rule\.out\.1$")
_GDN_CONV_INPUT_KEY = re.compile(r"^op\.model\.language_model\.layers\.(\d+)\.linear_attn\.fn\.causal_conv1d_fn\.in0$")
_PLE_CONV_INPUT_KEY = re.compile(r"^op\.model\.language_model\.layers\.(\d+)\.ple\.conv1d\.in0$")


def _last_columns(torch: Any, value: Any, count: int) -> Any:
    """Last ``count`` columns (left zero-padded), the way the pinned cache stores conv history."""

    if value.shape[-1] >= count:
        return value[..., -count:].clone()
    padding = torch.zeros(*value.shape[:-1], count - value.shape[-1], dtype=value.dtype)
    return torch.cat([padding, value], dim=-1)


def captured_states(assembly: Assembly, forward: ForwardRun) -> dict[str, Any]:
    """Layer states after the fixed tokens, derived from the helper captures of the stateless forward.

    Keys ``captured.layer<L>.<conv_state0|recurrent_state0|conv_state1|conv_state2>`` use the cache's own slot
    numbering (0 = GDN conv / recurrence, 1 = PLE short-conv history, 2 = PLE token context) and shapes, but are
    computed from the captured helper inputs/outputs, not read from a ``DynamicCache``.  They exist for every
    profile, including linear-attention-only stacks where pinned ``DynamicCache`` cannot run
    (:data:`CACHE_UNSUPPORTED_REASON`); when the cache does run, a test asserts equality with the cache states.
    """

    torch = assembly.torch
    config = assembly.config.text_config
    kernel = int(config.linear_conv_kernel_dim)
    ple_active = bool(config.ple_layer_ids)  # PLE-free models have no conv_state1/conv_state2 slots at all
    ple_length = (int(config.ple_conv_kernel_size) - 1) * int(config.ngram_size) if ple_active else 0
    context = int(config.ngram_size) - 1 if ple_active else 0
    eos = _eos_id(assembly)
    states: dict[str, Any] = {}
    for key, value in forward.dumps.items():
        match = _GDN_RECURRENT_KEY.match(key)
        if match:
            states[f"captured.layer{match.group(1)}.recurrent_state0"] = value.clone()
            continue
        match = _GDN_CONV_INPUT_KEY.match(key)
        if match:
            states[f"captured.layer{match.group(1)}.conv_state0"] = _last_columns(torch, value, kernel)
            continue
        match = _PLE_CONV_INPUT_KEY.match(key)
        if match:
            layer = match.group(1)
            states[f"captured.layer{layer}.conv_state1"] = _last_columns(torch, value, ple_length)
            ids = forward.dumps["input.token_ids"]
            if ids.shape[-1] < context:
                ids = torch.cat([torch.full((*ids.shape[:-1], context - ids.shape[-1]), eos, dtype=ids.dtype), ids], dim=-1)
            states[f"captured.layer{layer}.conv_state2"] = ids[..., -context:].clone()
    if not states:
        raise FixtureError("no helper-capture states were recorded")
    return states


def _eos_id(assembly: Assembly) -> int:
    eos = assembly.config.text_config.eos_token_id
    return int(eos[0] if isinstance(eos, (list, tuple)) else eos)


def run_cached(
    assembly: Assembly,
    tokens: Sequence[int],
    chunks: Sequence[int],
    *,
    capture_ple: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run ``tokens`` through a DynamicCache in chunks; returns (state/logit dumps, concatenated logits)."""

    if sum(chunks) != len(tokens):
        raise FixtureError("chunk sizes must sum to the token count")
    torch = assembly.torch
    cache = importlib.import_module("transformers").DynamicCache(config=assembly.config.text_config)
    capture = Capture(torch)
    if capture_ple:
        for name, module in assembly.text_model.named_modules():
            if name.endswith("ple_embedding.ngram_embedding") and type(module).__name__ == "Embedding":
                capture._handles.append(module.register_forward_pre_hook(capture._pre_hook(f"model.language_model.{name}")))
    dumps: dict[str, Any] = {}
    parts = []
    start = 0
    try:
        with torch.no_grad():
            for index, size in enumerate(chunks):
                piece = torch.tensor([list(tokens[start : start + size])], dtype=torch.long)
                out = assembly.model(input_ids=piece, past_key_values=cache, use_cache=True)
                parts.append(out.logits.detach().clone().cpu())
                dumps[f"chunk{index}.logits"] = parts[-1]
                start += size
    finally:
        capture.detach()
    logits = torch.cat(parts, dim=1)
    dumps.update(_cache_state_dump(cache, "state."))
    dumps.update({k.replace("op.", "ple.", 1): v for k, v in capture.records.items()})
    return dumps, logits


# ---------------------------------------------------------------------------
# Native MTP (pinned vLLM forward body around pinned HF operators)
# ---------------------------------------------------------------------------


def _causal_float_mask(torch: Any, length: int, dtype: Any) -> Any:
    minimum = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, length, length), dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1), minimum)
    return mask


def run_mtp_step0(assembly: Assembly, forward: ForwardRun) -> dict[str, Any]:
    """MTP step 0 over the fixed tokens: draft input token ``t+1`` with the backbone's multi-stream hidden at ``t``."""

    if assembly.mtp_head is None:
        raise FixtureError("this profile has no MTP head (with_mtp=false); there is no MTP step to run")
    torch = assembly.torch
    sources = assembly.sources
    verify_vllm_mtp_contract(sources.vllm_text)
    mtp_forward = _compile_method(
        sources.vllm_text["vllm_mtp"],
        "Qwen4ExpMultiTokenPredictor",
        "forward",
        {"torch": torch, "nn": torch.nn, "get_pp_group": lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True)},
    )
    head = assembly.mtp_head
    text_model = assembly.text_model
    config = assembly.config.text_config
    tokens = forward.token_ids
    if len(tokens) < 2:
        raise FixtureError("MTP step 0 needs at least two fixed tokens")
    input_ids = torch.tensor(tokens[1:], dtype=torch.long)
    positions = torch.arange(1, len(tokens), dtype=torch.long)
    backbone = forward.pre_mixer[0, :-1, :].contiguous()  # [T-1, hc*H] pre-final-mixer stream at positions 0..T-2
    layer = head.layers[0]
    capture = Capture(torch)
    capture.attach(head, "mtp.")
    state: dict[str, Any] = {}
    modeling = sources.modeling_module

    class PinnedMtpLayer:
        """vLLM ``Qwen4ExpDecoderLayer.forward`` contract on top of the pinned HF decoder layer.

        Returns ``(hidden_states, mlp_out, injection)`` exactly as the pinned
        vLLM layer does: the multi-stream state *before* the pending MLP-HC
        combine, the MLP output, and the MLP-HC injection (carried here as the
        already-activated ``2*sigmoid(.)`` weights, the quantity both the HF
        and vLLM-common GatedResidual consume).
        """

        def __call__(
            self,
            *,
            hidden_states: Any,
            prev_block_output: Any,
            prev_injection: Any,
            positions: Any,
            input_ids: Any,
            query_start_loc: Any,
            ngram_context: Any,
        ) -> tuple[Any, Any, Any]:
            if prev_block_output is not None or prev_injection is not None or layer.ple is not None:
                raise FixtureError("MTP step 0 has no pending HC state and PLE is forced off")
            sequence = hidden_states.unsqueeze(0)
            length = sequence.shape[1]
            cos, sin = text_model.rotary_emb(sequence, positions.view(1, 1, -1).expand(3, 1, -1))
            mask = _causal_float_mask(torch, length, sequence.dtype)
            mixed, hyper_input, injection_weights = layer.attn_hyper_connection(sequence)
            attn_out, _ = layer.self_attn(mixed, (cos, sin), attention_mask=mask, past_key_values=None)
            after_attn = hyper_input + (attn_out.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)
            mixed2, hyper_input2, injection_weights2 = layer.mlp_hyper_connection(after_attn)
            mlp_out = layer.mlp(mixed2)
            state["layer.input"] = hidden_states.detach().clone()
            state["layer.attn_out"] = attn_out.detach().clone()
            state["layer.hidden_before_mlp_combine"] = hyper_input2.detach().clone()
            state["layer.mlp_out"] = mlp_out.detach().clone()
            state["layer.mlp_injection_weights"] = injection_weights2.detach().clone()
            return hyper_input2[0], mlp_out[0], injection_weights2[0]

    class PinnedMtpMixer:
        """vLLM ``GatedResidual.combine_and_mix`` (``use_combine=False``) via the HF ``GatedResidual``."""

        def combine_and_mix(self, hidden_states: Any, block_output: Any, injection: Any) -> tuple[Any, Any, None]:
            streams = hidden_states.unflatten(-1, (config.hc_count, config.hidden_size))
            combined = streams + block_output.unsqueeze(-2) * injection.unsqueeze(-1)
            multi = combined.flatten(-2)
            mixed = head.hyper_connection_mixer(multi.unsqueeze(0))[0]
            state["multi_hidden"] = multi.detach().clone()
            state["sample_hidden"] = mixed.detach().clone()
            return multi, mixed, None

    host = types.SimpleNamespace(
        hc_count=config.hc_count,
        hidden_size=config.hidden_size,
        embed_input_ids=text_model.embed_tokens,
        pre_fc_norm_embedding=head.pre_fc_norm_embedding,
        fc_embedding=head.fc_embedding,
        pre_fc_norm_hidden=head.pre_fc_norm_hidden,
        fc_hidden=head.fc_hidden,
        num_mtp_layers=1,
        layers=[PinnedMtpLayer()],
        hyper_connection_mixer=PinnedMtpMixer(),
    )
    try:
        with torch.no_grad(), capture.tap_functions(modeling, _TAPPED_FUNCTIONS):
            sample_hidden, multi_hidden = mtp_forward(
                host,
                input_ids=input_ids,
                positions=positions,
                hidden_states=backbone,
                inputs_embeds=None,
                spec_step_idx=0,
            )
            logits = assembly.model.lm_head(sample_hidden)
    finally:
        capture.detach()
    dumps: dict[str, Any] = {f"mtp.{key}": value.cpu() for key, value in state.items()}
    dumps.update({key.replace("op.", "mtp_op.", 1): value for key, value in capture.records.items()})
    dumps["mtp.in.token_ids"] = input_ids.clone()
    dumps["mtp.in.positions"] = positions.clone()
    dumps["mtp.in.backbone_multi_stream"] = backbone.detach().clone().cpu()
    dumps["mtp.out.sample_hidden"] = sample_hidden.detach().clone().cpu()
    dumps["mtp.out.multi_hidden"] = multi_hidden.detach().clone().cpu()
    dumps["mtp.out.logits"] = logits.detach().clone().cpu()
    return dumps


# ---------------------------------------------------------------------------
# Checkpoint (config + tokenizer + BF16 safetensors + inventory)
# ---------------------------------------------------------------------------


def build_tokenizer(options: ProfileOptions) -> tuple[bytes, dict[str, object], list[int]]:
    """Deterministic 32-token word-level tokenizer; returns (tokenizer.json bytes, config, FIXED_TEXT ids)."""

    if options.vocab_size != 32:
        raise FixtureError("the tiny tokenizer is defined for vocab_size 32")
    try:
        from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers
    except ImportError as exc:  # pragma: no cover
        raise FixtureError(f"tiny tokenizer requires the tokenizers package: {exc}") from exc
    specials = ["<|pad|>", "<|bos|>", "<|endoftext|>"]
    if specials[options.eos_token_id if options.eos_token_id < 3 else 2] != "<|endoftext|>" and options.eos_token_id != 2:
        raise FixtureError("tokenizer layout fixes eos at id 2")
    vocab = {token: index for index, token in enumerate(specials)}
    for offset in range(26):
        vocab[chr(ord("a") + offset)] = 3 + offset
    vocab["\u2581"] = 29
    vocab["."] = 30
    vocab[","] = 31
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tokenizer.normalizer = normalizers.Replace(" ", "\u2581")
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in specials])
    ids = tokenizer.encode(FIXED_TEXT).ids
    if len(ids) > options.max_position_embeddings:
        raise FixtureError(f"max_position_embeddings {options.max_position_embeddings} is below the {len(ids)} fixed tokens")
    raw = tokenizer.to_str(pretty=True).encode("utf-8")
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|endoftext|>",
        "model_max_length": options.max_position_embeddings,
        "hipfire_note": "deterministic developer-only word-level tokenizer; no chat template",
    }
    return raw, config, ids


def _tensor_bytes(tensor_value: Any) -> bytes:
    torch = _torch()
    flat = tensor_value.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""


_SAFETENSORS_DTYPE = {"bfloat16": "BF16", "float32": "F32", "int64": "I64"}


def checkpoint_tensors(assembly: Assembly) -> dict[str, Any]:
    """Checkpoint-layout state dict: runtime names, n-gram table sharded as ``shard_<i>.weight``."""

    options = assembly.options
    config = assembly.config.text_config
    parts = int(config.split_ngram_parts)
    tensors: dict[str, Any] = {}
    for name, value in assembly.model.state_dict().items():
        if name.endswith("ngram_embedding.weight"):
            rows = value.shape[0]
            if rows % parts:
                raise FixtureError(f"padded n-gram rows {rows} not divisible by split_ngram_parts {parts}")
            step = rows // parts
            stem = name[: -len("weight")]
            for shard in range(parts):
                tensors[f"{stem}shard_{shard}.weight"] = value[shard * step : (shard + 1) * step].clone()
            continue
        tensors[name] = value.detach().clone()
    if assembly.mtp_head is not None:
        for name, value in assembly.mtp_head.state_dict().items():
            tensors[f"mtp.{name}"] = value.detach().clone()
    del options
    return tensors


def tensor_component(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name == "lm_head.weight":
        return "lm_head"
    if ".ple." in name:
        return "ple"
    if name.startswith("model.language_model."):
        return "language_model"
    return "other"


def write_checkpoint(directory: Path, assembly: Assembly) -> dict[str, object]:
    """Write config/tokenizer/model.safetensors and return their manifest records."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise FixtureError(f"safetensors is required for the checkpoint inventory: {exc}") from exc
    torch = assembly.torch
    options = assembly.options
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    write_json(config_path, full_config_dict(options, ple_geometry=derived_ple_geometry(assembly)))
    tokenizer_raw, tokenizer_config, fixed_ids = build_tokenizer(options)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_raw)
    write_json(directory / "tokenizer_config.json", tokenizer_config)
    tensors = checkpoint_tensors(assembly)
    for name, value in tensors.items():
        if value.dtype not in (torch.bfloat16, torch.int64):
            raise FixtureError(f"{name} has dtype {value.dtype}; the BF16 source checkpoint holds BF16 and I64 only")
    path = directory / "model.safetensors"
    save_file(dict(sorted(tensors.items())), str(path), metadata={"format": "pt"})
    inventory = []
    for name in sorted(tensors):
        value = tensors[name]
        logical = str(value.dtype).replace("torch.", "")
        inventory.append(
            {
                "name": name,
                "dtype": _SAFETENSORS_DTYPE[logical],
                "shape": list(value.shape),
                "elements": int(value.numel()),
                "bytes": int(value.numel() * value.element_size()),
                "component": tensor_component(name),
                "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
            }
        )
    counts: dict[str, int] = {}
    for item in inventory:
        counts[str(item["component"])] = counts.get(str(item["component"]), 0) + 1
    return {
        "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": sha256_file(tokenizer_path),
            "config_path": "tokenizer_config.json",
            "config_sha256": sha256_file(directory / "tokenizer_config.json"),
            "fixed_text": FIXED_TEXT,
            "fixed_text_ids": fixed_ids,
        },
        "safetensors": {"path": path.name, "sha256": sha256_file(path), "tensor_count": len(inventory)},
        "inventory": inventory,
        "inventory_component_counts": dict(sorted(counts.items())),
    }


# ---------------------------------------------------------------------------
# Dump writing
# ---------------------------------------------------------------------------


def _logical_dtype(value: Any) -> str:
    return str(value.dtype).replace("torch.", "")


_MASK_INFINITY_KEY = re.compile(r"\.fn\.eager_attention_forward(\.call\d+)?\.in4$")


def dumps_to_arrays(torch: Any, dumps: Mapping[str, Any]) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Convert dumps to storage tensors.  NaN is always rejected; infinity only in additive attention masks
    (``min + min`` of the combined causal and QSA masks legitimately overflows to ``-inf``)."""

    arrays: dict[str, Tensor] = {}
    logical: dict[str, str] = {}
    for name, value in dumps.items():
        if not hasattr(value, "detach"):
            raise FixtureError(f"dump {name} is not a tensor")
        arrays[name] = _from_torch(value, torch)
        logical[name] = _logical_dtype(value)
        stored = arrays[name]
        allow_infinity = bool(_MASK_INFINITY_KEY.search(name))
        if stored.dtype in ("float32", "float64"):
            for item in stored.data:
                number = float(item)
                if math.isnan(number) or (math.isinf(number) and not allow_infinity):
                    raise FixtureError(f"nonfinite dump {name}")
        elif stored.dtype == "uint16" and logical[name] == "bfloat16":
            for item in stored.data:
                word = int(item)
                if (word & 0x7F80) == 0x7F80 and ((word & 0x007F) != 0 or not allow_infinity):
                    raise FixtureError(f"nonfinite BF16 dump {name}")
    return arrays, logical


def write_dump_file(directory: Path, name: str, torch: Any, dumps: Mapping[str, Any]) -> dict[str, object]:
    arrays, logical = dumps_to_arrays(torch, dumps)
    path = directory / f"{name}.npz"
    digest = write_npz(path, arrays)
    metadata = array_metadata(arrays)
    for item in metadata:
        item["logical_dtype"] = logical[str(item["name"])]
    return {"name": name, "path": path.name, "sha256": digest, "arrays": metadata}


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def changed_keys(a: Mapping[str, Any], b: Mapping[str, Any], torch: Any, *, prefix: str = "") -> list[str]:
    changed = []
    for key in sorted(set(a) | set(b)):
        if not key.startswith(prefix):
            continue
        if key not in a or key not in b:
            changed.append(key)
            continue
        left, right = a[key], b[key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            changed.append(key)
    return changed


def run_negative_controls(
    assembly: Assembly,
    base: ForwardRun,
    tokens: Sequence[int],
    sources: VerifiedSources,
) -> list[dict[str, object]]:
    """Each control perturbs one thing and MUST change the named dumps; ``identity`` MUST change none."""

    torch = assembly.torch
    options = assembly.options
    results: list[dict[str, object]] = []
    base_seed = assembly.effective_seed

    def record(name: str, description: str, expect_prefixes: Sequence[str], changed: list[str], must_equal: bool = False) -> None:
        observed = [key for key in changed if any(key.startswith(prefix) for prefix in expect_prefixes)]
        passed = (not changed) if must_equal else bool(observed)
        results.append(
            {
                "name": name,
                "description": description,
                "expected_to_change": list(expect_prefixes),
                "changed_count": len(changed),
                "changed_expected_examples": observed[:6],
                "passed": passed,
            }
        )
        if not passed:
            raise FixtureError(f"negative control {name} failed: expected {list(expect_prefixes)} to change, got {changed[:6]}")

    identity = run_forward(build_assembly(options, sources, seed=base_seed), tokens)
    record("identity_rebuild", "rebuilding with the same seed reproduces every dump bit-for-bit", [], changed_keys(base.dumps, identity.dumps, torch), must_equal=True)

    flipped = run_forward(
        build_assembly(options, sources, seed=base_seed, overrides={"output_gate_type": "silu" if options.output_gate_type == "sigmoid" else "sigmoid"}),
        tokens,
    )
    record(
        "gdn_output_gate",
        "GDN output-gate activation sigmoid<->silu changes the gated norm output",
        ["op.model.language_model.layers.0.linear_attn.norm", "op.model.language_model.layers.0.linear_attn.out_proj"],
        changed_keys(base.dumps, flipped.dumps, torch),
    )

    if not options.with_ple:
        for name in ("ple_eos_history", "ple_cached_history"):
            results.append({"name": name, "status": "skipped", "passed": None, "reason": PLE_ABSENT_REASON})
    else:
        ple_tokens = list(tokens)
        eos = options.eos_token_id
        if eos not in ple_tokens[:-1]:
            raise FixtureError("fixed tokens must contain an EOS before the last position to exercise PLE history")
        replaced = [tokens[0] if value == eos else value for value in tokens]
        other = run_forward(build_assembly(options, sources, seed=base_seed), replaced)
        record(
            "ple_eos_history",
            "replacing the in-sequence EOS changes the EOS-aware n-gram ids and every downstream PLE dump",
            ["op.model.language_model.layers.0.ple.ple_embedding.ngram_embedding.in0", "op.model.language_model.layers.0.ple.out0"],
            changed_keys(base.dumps, other.dumps, torch),
        )

        # Cached PLE history: same suffix chunk, different prefix => different 2-token context ids.
        if not has_attention_layer(options):
            results.append(
                {
                    "name": "ple_cached_history",
                    "status": "skipped",
                    "passed": None,
                    "reason": CACHE_UNSUPPORTED_REASON,
                }
            )
        else:
            suffix = list(tokens[-3:])
            prefix_a = [tokens[1], tokens[3], tokens[4]]
            prefix_b = [tokens[4], tokens[3], tokens[0]]
            runs = []
            for prefix in (prefix_a, prefix_b):
                run_assembly = build_assembly(options, sources, seed=base_seed)
                dumps, _ = run_cached(run_assembly, prefix + suffix, [len(prefix), len(suffix)], capture_ple=True)
                runs.append(dumps)
            record(
                "ple_cached_history",
                "identical suffix chunk after a different cached prefix changes the PLE n-gram ids (cache history is state)",
                ["ple.model.language_model.layers.0.ple.ple_embedding.ngram_embedding.in0.call1", "ple.model.language_model.layers.0.ple.ple_embedding.ngram_embedding.in0"],
                changed_keys(runs[0], runs[1], torch, prefix="ple."),
            )

    if options.profile_id == "tiny-moe":
        # Router gate: permute router rows of layer 0.
        permuted = build_assembly(options, sources, seed=base_seed)
        with torch.no_grad():
            weight = permuted.text_model.layers[0].mlp.gate.weight
            weight.copy_(weight.flip(0))
        router = run_forward(permuted, tokens)
        record(
            "moe_router_gate",
            "reversing the layer-0 router rows changes router logits, selected experts and the MoE output",
            ["op.model.language_model.layers.0.mlp.gate", "op.model.language_model.layers.0.mlp.experts"],
            changed_keys(base.dumps, router.dumps, torch),
        )
        unnormalised = run_forward(
            build_assembly(options, sources, seed=base_seed, overrides={"norm_topk_prob": not options.norm_topk_prob}),
            tokens,
        )
        record(
            "moe_norm_topk_prob",
            "flipping norm_topk_prob changes routing weights and the routed output",
            ["op.model.language_model.layers.0.mlp.gate.out1", "op.model.language_model.layers.0.mlp.experts"],
            changed_keys(base.dumps, unnormalised.dumps, torch),
        )
        wide = run_forward(build_assembly(options, sources, seed=base_seed, dtype="float32"), tokens)
        record(
            "bf16_vs_f32_rounding",
            "executing the identical weights in F32 instead of BF16 changes the intermediate state and logits",
            ["op.model.language_model.layers.0.mlp.out0", "output.logits"],
            changed_keys(base.dumps, wide.dumps, torch),
        )
    else:
        dense_cls = base_dense_class(assembly)
        previous = dense_cls.round_bf16
        dense_cls.round_bf16 = False
        try:
            f32_run = run_forward(assembly, tokens)
        finally:
            dense_cls.round_bf16 = previous
        record(
            "dense_ffn_rounding",
            "dense FFN with F32 instead of BF16 rounding boundaries changes gate/activation/product/output dumps",
            ["op.model.language_model.layers.0.mlp.activation", "op.model.language_model.layers.0.mlp.product"],
            changed_keys(base.dumps, f32_run.dumps, torch),
        )
        permuted = build_assembly(options, sources, seed=base_seed)
        with torch.no_grad():
            weight = permuted.text_model.layers[0].mlp.gate_proj.weight
            weight.copy_(weight.flip(0))
        gate_run = run_forward(permuted, tokens)
        record(
            "dense_gate_projection",
            "reversing the dense gate_proj rows changes the gated product and FFN output",
            ["op.model.language_model.layers.0.mlp.gate", "op.model.language_model.layers.0.mlp.output"],
            changed_keys(base.dumps, gate_run.dumps, torch),
        )
    return results


def base_dense_class(assembly: Assembly) -> type:
    if assembly.dense_cls is None:
        raise FixtureError("not a dense profile")
    return assembly.dense_cls


# ---------------------------------------------------------------------------
# Equivalence cross-check of the dense equation (labelled evidence, not upstream)
# ---------------------------------------------------------------------------


def dense_equation_cross_check(assembly: Assembly, forward: ForwardRun) -> dict[str, object]:
    """Compare the non-upstream dense equation with ``Qwen4ExpTextMLP`` on the same weights/inputs.

    Pinned ``Qwen4ExpTextMLP`` is reusable *math* only; this records the
    observed max BF16-ulp disagreement so the dense boundary is quantified, not
    asserted.
    """

    torch = assembly.torch
    modeling = assembly.sources.modeling_module
    layer0 = assembly.text_model.layers[0].mlp
    reference = modeling.Qwen4ExpTextMLP(assembly.text_model.config, intermediate_size=layer0.intermediate_size)
    with torch.no_grad():
        reference = reference.to(torch.bfloat16)
        reference.gate_proj.weight.copy_(layer0.gate_proj.weight)
        reference.up_proj.weight.copy_(layer0.up_proj.weight)
        reference.down_proj.weight.copy_(layer0.down_proj.weight)
        hidden = forward.dumps.get("op.model.language_model.layers.0.mlp_hyper_connection.out0")
        if hidden is None:
            raise FixtureError("missing layer-0 MLP-HC output for the dense cross-check")
        expected = reference(hidden.reshape(-1, hidden.shape[-1]))
        actual = forward.dumps["op.model.language_model.layers.0.mlp.out0"].reshape(expected.shape)
    difference = (expected.float() - actual.float()).abs()
    scale = expected.float().abs().clamp_min(1e-12)
    return {
        "compared": "layers.0 mlp output vs Qwen4ExpTextMLP(same weights, same input)",
        "max_abs_difference": float(difference.max()),
        "max_relative_difference": float((difference / scale).max()),
        "elements_not_bit_equal": int((difference > 0).sum()),
        "elements": int(difference.numel()),
        "note": "evidence about arithmetic agreement of the labelled equation with reusable upstream MLP math; NOT upstream dense decoder support",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def shrink_evidence(assembly: Assembly) -> dict[str, object]:
    options = assembly.options
    cfg = assembly.config.text_config
    ple = None
    for layer in assembly.text_model.layers:
        if layer.ple is not None:
            embedding = layer.ple.ple_embedding
            ple = {
                "head_vocab_sizes": list(embedding.head_vocab_sizes),
                "head_offsets": list(embedding.head_offsets),
                "valid_rows": int(embedding.total_vocab_size),
                "padded_rows": int(embedding.ngram_embedding.weight.shape[0]),
                "row_width": int(embedding.ngram_embedding.weight.shape[1]),
                "layer_multipliers": [int(x) for x in embedding.layer_multipliers.tolist()],
                "seed": int(embedding.seed),
                "ngram_vocab_size_base": int(embedding.ngram_vocab_base) if hasattr(embedding, "ngram_vocab_base") else int(embedding.ngram_vocab_size_base),
                "split_ngram_parts": int(cfg.split_ngram_parts),
            }
            break
    parameters = sum(p.numel() for p in assembly.model.parameters())
    if assembly.mtp_head is not None:
        parameters += sum(p.numel() for p in assembly.mtp_head.parameters())
    return {
        "parameter_elements": int(parameters),
        "bounded_memory_estimate": estimated_parameter_elements(options),
        "ple": ple,
        "qsa_layers": [i for i, t in enumerate(cfg.layer_types) if t == "qwen_sparse_attention"],
        "gdn_layers": [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"],
        "qsa_index_budget_tokens": int(cfg.indexer_budget),
        "qsa_compress_ratio": int(cfg.indexer_compress_ratio),
    }


def optional_component_report(assembly: Assembly, checkpoint: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Presence of the optional PLE / MTP components, read from the instantiated model, the written inventory and the
    written ``config.json``.  Raises when any of them disagrees with ``options.with_ple`` / ``options.with_mtp``."""

    options = assembly.options
    text = assembly.config.text_config
    counts = checkpoint["inventory_component_counts"]
    ple_layers = [index for index, layer in enumerate(assembly.text_model.layers) if layer.ple is not None]
    ple_tensors = int(counts.get("ple", 0))
    mtp_tensors = int(counts.get("mtp", 0))
    mtp_built = assembly.mtp_head is not None
    config = json.loads((directory / checkpoint["config"]["path"]).read_text())
    written_text = config["text_config"]
    ple_config_present = {
        "text_config.ple_layer_ids": bool(written_text.get("ple_layer_ids")),
        "qwen4_ple": "qwen4_ple" in config,
        "hipfire_profile.ple": "ple" in config["hipfire_profile"],
        **{f"text_config.{key}": key in written_text for key in PLE_COMPANION_KEYS},
    }
    mtp_config_present = {f"text_config.{key}": key in written_text for key in MTP_CONFIG_KEYS}
    if not (bool(ple_layers) == (ple_tensors > 0) == options.with_ple == bool(text.ple_layer_ids)):
        raise FixtureError(f"with_ple={options.with_ple} disagrees with model layers {ple_layers} / {ple_tensors} PLE tensors")
    if any(present != options.with_ple for present in ple_config_present.values()):
        raise FixtureError(f"with_ple={options.with_ple} disagrees with config.json: {ple_config_present}")
    if not (mtp_built == (mtp_tensors > 0) == options.with_mtp):
        raise FixtureError(f"with_mtp={options.with_mtp} disagrees with the model ({mtp_built}) / {mtp_tensors} mtp.* tensors")
    if any(present != options.with_mtp for present in mtp_config_present.values()):
        raise FixtureError(f"with_mtp={options.with_mtp} disagrees with config.json: {mtp_config_present}")
    report: dict[str, Any] = {
        "ple": {
            "present": bool(ple_layers),
            "layers": ple_layers,
            "inventory_tensors": ple_tensors,
            "text_config_ple_layer_ids": list(text.ple_layer_ids),
            "cache_state_slots": int(text.number_of_conv_states),
        },
        "mtp": {"present": mtp_built, "inventory_tensors": mtp_tensors},
        "derived_from": "instantiated pinned HF layers/config, the written safetensors inventory and the written config.json",
    }
    if not ple_layers:
        report["ple"]["omitted"] = {
            "reason": PLE_ABSENT_REASON,
            "config": ["text_config." + key for key in PLE_COMPANION_KEYS] + ["qwen4_ple", "hipfire_profile.ple"],
            "states": ["captured.layer<L>.conv_state1", "captured.layer<L>.conv_state2", "*.state.layer<L>.conv_state1", "*.state.layer<L>.conv_state2"],
            "negative_controls": ["ple_eos_history", "ple_cached_history"],
        }
    if not mtp_built:
        report["mtp"]["omitted"] = {"reason": MTP_ABSENT_REASON, "config": list(MTP_CONFIG_KEYS), "dumps": ["mtp_step0"]}
    return report


# ---------------------------------------------------------------------------
# Opt-in QSA tie-screened seed search
# ---------------------------------------------------------------------------
#
# The pinned QSA indexer picks `scores.topk(block_topk)` over per-query block scores.  A near-tie at the k-th/(k+1)-th
# boundary makes the selected block set depend on the arithmetic of the executing backend (CPU torch vs a GPU kernel).
# The default profiles do NOT screen this (seed 3800 is used exactly).  `screen_qsa_seed` is the opt-in alternative: it
# walks seeds start, start+1, ... sequentially and accepts the first one whose every selecting QSA row, in the main
# model AND in MTP step 0, has a clear margin.  The scores are the exact pre-topk tensors of the pinned HF indexer
# (captured by wrapping `torch.Tensor.topk` for calls made from `Qwen4ExpTextQSAIndexer.forward`), not a re-implementation.

QSA_SCREEN_RULE = "tie-screened seed"
QSA_SCREEN_MARGIN_ABS = 0.05
QSA_SCREEN_MARGIN_REL = 0.05
QSA_SCREEN_MAX_SEEDS = 4096  # hard search bound; exhausting it is an error, never a fallback seed
QSA_SCORE_SOURCE = (
    "pinned HF Qwen4ExpTextQSAIndexer.forward `scores` tensor (relu(q.k_block).sum/sqrt(index_head_dim) over the complete causally "
    "visible blocks of one query row) captured at its `scores.topk(...)` call"
)


def qsa_margin_row(scores: Sequence[float], block_topk: int) -> dict[str, Any]:
    """Pure predicate: boundary margin of one query row's block scores.

    Requires ``len(scores) > block_topk`` (otherwise nothing is excluded and there is no boundary).  The row passes when the
    k-th best score is positive and ``kth - (k+1)th >= max(0.05, 0.05 * kth)``.
    """

    if isinstance(block_topk, bool) or not isinstance(block_topk, int) or block_topk < 1:
        raise FixtureError("block_topk must be a positive integer")
    values = [float(value) for value in scores]
    if len(values) <= block_topk:
        raise FixtureError(f"a QSA boundary needs more complete blocks ({len(values)}) than block_topk ({block_topk})")
    if any(math.isnan(value) for value in values):
        raise FixtureError("QSA block scores contain NaN")
    ordered = sorted(values, reverse=True)
    kth = ordered[block_topk - 1]
    next_best = ordered[block_topk]
    required = max(QSA_SCREEN_MARGIN_ABS, QSA_SCREEN_MARGIN_REL * kth)
    observed = kth - next_best
    return {
        "kth": kth,
        "next": next_best,
        "required_margin": required,
        "observed_margin": observed,
        "passed": bool(kth > 0.0 and observed >= required),
    }


def evaluate_qsa_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Manifest rows (layer, main/MTP, position, block counts, kth/next/required/observed margin, scores) for captured rows."""

    evaluated: list[dict[str, Any]] = []
    for row in rows:
        verdict = qsa_margin_row(row["scores"], row["block_topk"])
        evaluated.append(
            {
                "source": row["source"],
                "layer": row["layer"],
                "position": row["position"],
                "complete_blocks": row["blocks"],
                "block_topk": row["block_topk"],
                "block_scores": [float(value) for value in row["scores"]],
                **verdict,
            }
        )
    return evaluated


class QsaScoreTap:
    """Records the pre-topk block scores of every selecting QSA row (``num_complete_blocks > block_topk``)."""

    def __init__(self, torch: Any, modeling_module: Any) -> None:
        self.torch = torch
        self.indexer_cls = modeling_module.Qwen4ExpTextQSAIndexer
        self.rows: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def record(self, source: str) -> Iterator[None]:
        """Record rows labelled ``source`` ("main" or "mtp") while the context is active."""

        torch = self.torch
        tensor_cls = torch.Tensor
        had_own = "topk" in tensor_cls.__dict__
        original = tensor_cls.topk
        tap = self

        def topk(scores: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(scores, *args, **kwargs)
            frame = sys._getframe(1)
            owner = frame.f_locals.get("self")
            if frame.f_code.co_name == "forward" and isinstance(owner, tap.indexer_cls):
                local = frame.f_locals
                blocks = int(local["num_complete_blocks"])
                if scores.dim() != 1 or int(scores.shape[0]) != blocks:
                    raise FixtureError("QSA indexer topk input is not the per-block score vector of the visible complete blocks")
                if blocks != int(local["local_visible_indices"].shape[-1]) // owner.compress_ratio:
                    raise FixtureError("QSA block count disagrees with the causal visible-token count")
                if int(local["batch_idx"]) != 0:
                    raise FixtureError("QSA screening supports batch 0 only")
                if blocks > owner.block_topk:
                    values = [float(value) for value in scores.detach().float().cpu().tolist()]
                    if float(result.values[-1]) != sorted(values, reverse=True)[owner.block_topk - 1]:
                        raise FixtureError("captured QSA scores disagree with the k-th value returned by the pinned topk")
                    tap.rows.append(
                        {
                            "source": source,
                            "layer": int(owner.layer_idx),
                            "position": int(local["query_idx"]),
                            "blocks": blocks,
                            "block_topk": int(owner.block_topk),
                            "scores": values,
                        }
                    )
            return result

        tensor_cls.topk = topk
        try:
            yield
        finally:
            if had_own:
                tensor_cls.topk = original
            else:
                del tensor_cls.topk


def qsa_candidate_rows(options: ProfileOptions, sources: VerifiedSources, tokens: Sequence[int], seed: int) -> list[dict[str, Any]]:
    """Run the real main forward and MTP step 0 for ``seed`` and return the evaluated selecting QSA rows."""

    assembly = build_assembly(options, sources, seed=seed)
    tap = QsaScoreTap(assembly.torch, sources.modeling_module)
    with tap.record("main"):
        forward = run_forward(assembly, tokens)
    if options.with_mtp:
        with tap.record("mtp"):
            run_mtp_step0(assembly, forward)
    return evaluate_qsa_rows(tap.rows)


def screen_qsa_seed(
    options: ProfileOptions,
    sources: VerifiedSources,
    tokens: Sequence[int],
    *,
    max_seeds: int = QSA_SCREEN_MAX_SEEDS,
) -> dict[str, Any]:
    """Smallest seed ``>= options.seed`` whose main and MTP-step-0 QSA rows all pass :func:`qsa_margin_row`; no fallback."""

    if isinstance(max_seeds, bool) or not isinstance(max_seeds, int) or max_seeds < 1:
        raise FixtureError("qsa screen max seeds must be a positive integer")
    start = options.seed
    rejected: list[dict[str, Any]] = []
    for seed in range(start, start + max_seeds):
        rows = qsa_candidate_rows(options, sources, tokens, seed)
        present = {row["source"] for row in rows}
        required_sources = {"main", "mtp"} if options.with_mtp else {"main"}
        if not required_sources <= present:
            raise FixtureError(
                f"QSA tie screening needs selecting rows (blocks > block_topk) in {sorted(required_sources)}; "
                f"seed {seed} produced rows only in {sorted(present)} (profile geometry has no QSA boundary to screen)"
            )
        failing = [row for row in rows if not row["passed"]]
        if not failing:
            return {
                "rule": QSA_SCREEN_RULE,
                "start_seed": start,
                "effective_seed": seed,
                "rejected_count": len(rejected),
                "rejected": rejected,
                "predicate": {
                    "rule": "kth selected block score > 0 and kth - (k+1)th >= max(abs, rel * kth)",
                    "margin_abs": QSA_SCREEN_MARGIN_ABS,
                    "margin_rel": QSA_SCREEN_MARGIN_REL,
                    "rows": "every query row of every QSA layer in the main forward and of MTP step 0 with complete_blocks > block_topk",
                    "search": f"sequential seeds start, start+1, ... (bound {max_seeds}); exhausting the bound is an error",
                },
                "score_source": QSA_SCORE_SOURCE,
                "rows": rows,
            }
        rejected.append({"seed": seed, "failing_rows": len(failing), "first_failing_row": {k: v for k, v in failing[0].items() if k != "block_scores"}})
    raise FixtureError(f"no seed in [{start}, {start + max_seeds - 1}] passed the QSA tie screen ({max_seeds} candidates rejected)")


def generate_profile(
    out: str | Path,
    options: ProfileOptions,
    *,
    cache_dir: str | Path | None = None,
    controls: bool = True,
    qsa_tie_screen: bool = False,
    qsa_screen_max_seeds: int = QSA_SCREEN_MAX_SEEDS,
) -> dict[str, object]:
    """Write one tiny profile directory (checkpoint, dumps, manifest) and return the manifest."""

    options.validate()
    torch = _torch()
    sources = load_verified_sources(cache_dir)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # fixed reduction order: byte-stable dumps
    try:
        _tokenizer_raw, _tokenizer_config, tokens = build_tokenizer(options)
        cache_supported = has_attention_layer(options)
        chunk_sizes = [len(tokens) - 3, 1, 1, 1]
        screen: dict[str, Any] | None = None
        if qsa_tie_screen:
            screen = screen_qsa_seed(options, sources, tokens, max_seeds=qsa_screen_max_seeds)
            options = options.with_overrides(seed=screen["effective_seed"])
        assembly = build_assembly(options, sources, seed=options.seed)
        tap = QsaScoreTap(torch, sources.modeling_module) if screen is not None else None
        with tap.record("main") if tap is not None else contextlib.nullcontext():
            forward = run_forward(assembly, tokens)
        mtp_dumps = None
        if options.with_mtp:
            with tap.record("mtp") if tap is not None else contextlib.nullcontext():
                mtp_dumps = run_mtp_step0(assembly, forward)
        if tap is not None:
            regenerated = evaluate_qsa_rows(tap.rows)
            if regenerated != screen["rows"]:
                raise FixtureError("QSA block scores of the written fixture differ from the screened candidate run (non-deterministic run)")
        state_dumps: dict[str, Any] = captured_states(assembly, forward)
        chunk_difference = prefill_difference = None
        if cache_supported:
            cached_dumps, cached_logits = run_cached(assembly, tokens, [len(tokens)])
            chunked_dumps, chunked_logits = run_cached(assembly, tokens, chunk_sizes)
            full_logits = forward.logits
            chunk_difference = float((chunked_logits.float() - full_logits.float()).abs().max())
            prefill_difference = float((cached_logits.float() - full_logits.float()).abs().max())
            state_dumps.update({f"prefill.{k}": v for k, v in cached_dumps.items()})
            state_dumps.update({f"chunked.{k}": v for k, v in chunked_dumps.items()})

        checkpoint = write_checkpoint(destination, assembly)
        components = optional_component_report(assembly, checkpoint, destination)
        dump_files = [
            write_dump_file(destination, "forward", torch, forward.dumps),
            write_dump_file(destination, "states", torch, state_dumps),
        ]
        if mtp_dumps is not None:  # never an empty/fabricated mtp_step0 for an MTP-free profile
            dump_files.append(write_dump_file(destination, "mtp_step0", torch, mtp_dumps))
        control_results = run_negative_controls(assembly, forward, tokens, sources) if controls else []
        dense_check = dense_equation_cross_check(assembly, forward) if options.profile_id == "tiny-dense-dev" else None
        contract = verify_vllm_mtp_contract(sources.vllm_text) if options.with_mtp else None
    finally:
        torch.set_num_threads(previous_threads)

    envelope = profile_envelope(options)
    manifest: dict[str, object] = {
        "schema": TINY_SCHEMA,
        "hipfire_profile": envelope,
        "options": {
            k: v for k, v in options.__dict__.items() if k not in _MANIFEST_OMITTED_DEFAULTS or v != _MANIFEST_OMITTED_DEFAULTS[k]
        },
        "seed": (
            {
                "requested": options.seed,
                "effective": assembly.effective_seed,
                "init": "per-tensor torch.Generator seeded by sha256(seed|tensor name); BF16 parameters; exactly the requested seed",
            }
            if screen is None
            else {
                "requested": screen["start_seed"],
                "effective": assembly.effective_seed,
                "init": "per-tensor torch.Generator seeded by sha256(seed|tensor name); BF16 parameters; effective seed = smallest "
                "seed >= requested that passes the QSA tie screen (see qsa_tie_screen)",
            }
        ),
        "equation_sources": {
            "main_model": envelope["equation_source"],
            "ffn": (
                "Qwen4ExpTextSparseMoeBlock (pinned HF 93c8b7b4)"
                if options.profile_id == "tiny-moe"
                else "equations.dense_ffn_non_upstream (NON-UPSTREAM; pinned HF has no dense decoder)"
            ),
            "mtp": MTP_EQUATION_SOURCE,
        },
        "source_verification": {
            "records": sources.records,
            "environment": sources.environment,
            "policy": "SHA-256 of the exact executed bytes checked against upstream.UPSTREAM_SOURCE_SPECS before any model construction",
        },
        "checkpoint": checkpoint,
        "fixed_tokens": {"text": FIXED_TEXT, "ids": tokens, "eos_token_id": options.eos_token_id},
        "dumps": dump_files,
        "chunked_decode": (
            {
                "supported": True,
                "chunk_sizes": chunk_sizes,
                "max_abs_logit_difference_vs_full_forward": chunk_difference,
                "prefill_cache_max_abs_logit_difference_vs_full_forward": prefill_difference,
            }
            if cache_supported
            else {"supported": False, "reason": CACHE_UNSUPPORTED_REASON}
        ),
        "mtp": {
            "equation_source": MTP_EQUATION_SOURCE,
            "label": "pinned vLLM Qwen4ExpMultiTokenPredictor.forward body executed verbatim; decoder layer and final HC mixer are the pinned HF operators",
            "pinned_hf_mtp_support": False,
            "hf_behaviour": "Qwen4ExpPreTrainedModel._keys_to_ignore_on_load_unexpected = [r'^mtp.*']: pinned HF drops mtp.* keys and has no MTP model",
            "vllm_contract": contract,
            "step": "step 0 (QSA selection recomputed); tokens[1:] with backbone multi-stream hidden at positions 0..T-2",
            "shared_with_main_model": ["embed_tokens", "lm_head"],
            "ple": "off (HF validation limits ple_layer_ids to [1, num_hidden_layers]; MTP layer index is num_hidden_layers)",
            "not_covered": [
                "later draft steps (vLLM skip_topk QSA index reuse and the draft paged KV cache are engine state, not pinned CPU-constructible source)",
                "vLLM Triton HC/QSA kernels (amd/ops/*), which are not pinned; equivalence relies on vllm common/hyperconnection.py torch equations matching HF GatedResidual (asserted by the contract check)",
            ],
        },
        "shrink": shrink_evidence(assembly),
        "negative_controls": control_results,
        "dense_equation_cross_check": dense_check,
        "dense_boundary": (
            "tiny-dense-dev replaces only the MLP slot with a labelled non-upstream equation; every other dump comes from pinned decoder code"
            if options.profile_id == "tiny-dense-dev"
            else None
        ),
        "tolerances": {
            "bf16_dump": "bit-exact on the same torch build and thread count; cross-build comparison uses BF16 unit roundoff 2^-7",
            "integer_dump": "byte exact",
        },
    }
    if any(getattr(options, key) != default for key, default in _ROPE_OPTION_DEFAULTS.items()):
        manifest["rope"] = rope_evidence(assembly)  # only non-default variants add this key: default manifests stay byte-identical
    if not options.with_mtp:
        del manifest["equation_sources"]["mtp"]
        manifest["mtp"] = {
            "present": False,
            "constructed": False,
            "pinned_hf_mtp_support": False,
            "reason": MTP_ABSENT_REASON,
            "omitted": [
                "text_config.mtp, mtp_num_hidden_layers and mtp_use_dedicated_embeddings",
                "mtp.* tensors",
                "mtp_step0 dump (no empty or fabricated MTP dump is written; forward.npz and states.npz are the real backbone forward and layer states)",
            ],
        }
    if not (options.with_ple and options.with_mtp):
        manifest["optional_components"] = components  # only variants add this key: default manifests stay byte-identical
    if screen is not None:  # only screened runs add this key and change seed.requested: unscreened manifests stay byte-identical
        manifest["qsa_tie_screen"] = screen
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
