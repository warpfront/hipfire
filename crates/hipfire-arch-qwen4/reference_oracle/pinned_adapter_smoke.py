# SPDX-License-Identifier: Apache-2.0
"""Deterministic integration smoke for the commit-pinned Qwen4 adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import upstream


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / ".codeinsight+research" / "qwen4" / "upstream-cache"


def _cfg() -> dict[str, object]:
    return {
        "hidden_size": 8,
        "vocab_size": 32,
        "hc_count": 2,
        "hc_lowrank": 3,
        "rms_norm_eps": 1.0e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "head_dim": 4,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 2,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
        "attention_dropout": 0.0,
        "attention_bias": False,
        "_attn_implementation": "eager",
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "intermediate_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ple_embed_dim": 8,
        "ngram_vocab_size_base": 7,
        "eos_token_id": 2,
        "make_ngram_vocab_size_divisible_by": 8,
        "ple_conv_kernel_size": 2,
    }


def _tensor(shape: tuple[int, ...], *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    values = torch.arange(1, 1 + int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
    return (values / 97.0).to(dtype)


def _run() -> None:
    torch.manual_seed(123)
    cfg = _cfg()
    source_text, source_provenance, _ = upstream.load_pinned_sources(CACHE)
    operators = upstream.load_pinned_operators(source_text, source_provenance, torch)
    provenance = operators.provenance()
    required = {"hyperconnection", "gdn", "qsa", "moe", "ple"}
    if set(provenance) != required:
        raise AssertionError(f"adapter operation set changed: {sorted(provenance)}")
    for operation, rows in provenance.items():
        if not rows or any(
            not row.get("path")
            or not row.get("commit")
            or not row.get("sha256")
            or not row.get("callable")
            or not row.get("ast_sha256")
            or row.get("origin") != "verified_pinned_spec"
            for row in rows
        ):
            raise AssertionError(f"incomplete source provenance for {operation}")

    hidden = _tensor((5, 8))
    wide = _tensor((5, 16))
    hc_weights = {
        "hc_norm": _tensor((16,)),
        "down": _tensor((3, 16)),
        "up": _tensor((16, 3)),
        "block": _tensor((2, 16)),
    }
    mixed, hyper_input, injection = operators.hyperconnection(wide, hc_weights, cfg, combine=True)
    if mixed.shape != hidden.shape or hyper_input.shape != wide.shape or injection.shape[-1] != 2:
        raise AssertionError("pinned HC adapter shape contract changed")
    injected = operators.inject(hyper_input, mixed, injection)
    if injected.shape != wide.shape:
        raise AssertionError("pinned HC injection shape contract changed")

    gdn_weights = {
        "qkv": _tensor((16, 8)),
        "z": _tensor((8, 8)),
        "b": _tensor((2, 8)),
        "a": _tensor((2, 8)),
        "conv": _tensor((16, 1, 4)),
        "a_log": _tensor((2,)),
        "dt_bias": _tensor((2,)),
        "norm": _tensor((4,)),
        "out": _tensor((8, 8)),
    }
    gdn_output, _ = operators.gdn(hidden, gdn_weights, cfg, 0)
    if gdn_output.shape != hidden.shape:
        raise AssertionError("pinned GDN adapter shape contract changed")

    rotary = operators._module(operators._hf["Qwen4ExpTextRotaryEmbedding"](operators._config(cfg)), hidden)
    positions = torch.arange(hidden.shape[0], dtype=torch.long).unsqueeze(0)
    rotary_cos, rotary_sin = rotary(hidden.unsqueeze(0), positions)
    if rotary_cos.shape != (1, 5, 2) or rotary_sin.shape != (1, 5, 2):
        raise AssertionError("pinned Rotary constructor/staticmethod binding changed")
    qsa_weights = {
        "index_qk": _tensor((4, 8)),
        "index_q_norm": _tensor((2,)),
        "index_k_norm": _tensor((2,)),
        "q": _tensor((16, 8)),
        "q_norm": _tensor((4,)),
        "k": _tensor((8, 8)),
        "k_norm": _tensor((4,)),
        "v": _tensor((8, 8)),
        "out": _tensor((8, 8)),
    }
    qsa_output, qsa_attention = operators.qsa(hidden, qsa_weights, cfg, 1)
    if qsa_output.shape != hidden.shape or qsa_attention.shape[-2:] != (5, 5):
        raise AssertionError("pinned QSA adapter shape contract changed")

    full_weights = {
        "model.language_model.layers.0.mlp.gate.weight": _tensor((3, 8)),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": _tensor((4, 8)),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": _tensor((4, 8)),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": _tensor((8, 4)),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": _tensor((1, 8)),
    }
    expert_gate = _tensor((8, 8))
    expert_down = _tensor((8, 4))

    def load_full(name: str) -> torch.Tensor:
        return full_weights[name]

    def load_expert(_name: str, _expert: int, _row_start: int, rows: int) -> torch.Tensor:
        return expert_down if "down_proj" in _name else expert_gate

    moe_output = operators.moe(hidden, cfg, load_full, load_expert, 0)
    if moe_output.shape != hidden.shape:
        raise AssertionError("pinned MoE adapter shape contract changed")
    ple_full = {
        "model.language_model.layers.1.ple.key_proj.weight": _tensor((16, 8)),
        "model.language_model.layers.1.ple.value_proj.weight": _tensor((8, 8)),
        "model.language_model.layers.1.ple.norm_key.weight": _tensor((16,)),
        "model.language_model.layers.1.ple.norm_query.weight": _tensor((16,)),
        "model.language_model.layers.1.ple.norm_conv.weight": _tensor((16,)),
        "model.language_model.layers.1.ple.conv1d.weight": _tensor((16, 1, 2)),
    }
    ple_metadata = {
        "layer_multipliers": torch.tensor([10007, 10009, 10037], dtype=torch.long),
        "ngram_heads_vocab_sizes": torch.tensor([7, 11], dtype=torch.long),
        "ngram_heads_offsets": torch.tensor([0, 7], dtype=torch.long),
    }

    def load_ple_full(name: str) -> torch.Tensor:
        return ple_full[name]

    def read_rows(ids: list[int]) -> torch.Tensor:
        return _tensor((len(ids), 4))

    ple_output = operators.ple(hidden.repeat(1, 2), [1, 3, 4, 5, 6], cfg, ple_metadata, load_ple_full, read_rows)
    if ple_output.shape != wide.shape:
        raise AssertionError("pinned PLE adapter shape contract changed")

    repeat = (
        operators.gdn(hidden, gdn_weights, cfg, 0)[0],
        operators.qsa(hidden, qsa_weights, cfg, 1)[0],
        operators.moe(hidden, cfg, load_full, load_expert, 0),
        operators.ple(hidden.repeat(1, 2), [1, 3, 4, 5, 6], cfg, ple_metadata, load_ple_full, read_rows),
    )
    first = (gdn_output, qsa_output, moe_output, ple_output)
    for before, after in zip(first, repeat):
        if not torch.equal(before, after):
            raise AssertionError("pinned AST adapter is not deterministic")
    print("pinned AST adapter smoke: HC/GDN/QSA/MoE/PLE passed deterministically")


if __name__ == "__main__":
    try:
        _run()
    except Exception as exc:
        print(f"pinned adapter smoke failed: {exc}", file=sys.stderr)
        raise
