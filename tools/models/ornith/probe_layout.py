#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Assert Ornith 1.5 35B-A3B matches the layout the quant plan assumes.

Exits 1 on the first violated assumption. Read-only.
"""
import json
import sys
from pathlib import Path

DEFAULT_SRC = "/home/nick/hf/Ornith-1.5-35B-A3B"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(
            "usage: python3 -m tools.models.ornith.probe_layout [model_dir]\n"
            f"  default model_dir: {DEFAULT_SRC}"
        )
        return 0

    src = Path(args[0] if args else DEFAULT_SRC)

    cfg = json.loads((src / "config.json").read_text())
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    tc = cfg["text_config"]

    fail = []

    def check(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f" (want {want!r})"))
        if not ok:
            fail.append(label)

    print("config:")
    check("model_type", cfg["model_type"], "qwen3_5_moe")
    check("architectures", cfg["architectures"], ["Qwen3_5MoeForConditionalGeneration"])
    check("num_hidden_layers", tc["num_hidden_layers"], 40)
    check("hidden_size", tc["hidden_size"], 2048)
    check("num_experts", tc["num_experts"], 256)
    check("num_experts_per_tok", tc["num_experts_per_tok"], 8)
    check("moe_intermediate_size", tc["moe_intermediate_size"], 512)
    check("shared_expert_intermediate_size", tc["shared_expert_intermediate_size"], 512)
    check("head_dim", tc["head_dim"], 256)
    check("num_key_value_heads", tc["num_key_value_heads"], 2)
    check("partial_rotary_factor", tc["partial_rotary_factor"], 0.25)
    check("vocab_size", tc["vocab_size"], 248320)
    check("tie_word_embeddings", tc["tie_word_embeddings"], False)
    check("mrope_section", tc["rope_parameters"]["mrope_section"], [11, 11, 10])
    check("mrope_interleaved", tc["rope_parameters"]["mrope_interleaved"], True)
    check("vision deepstack", cfg["vision_config"]["deepstack_visual_indexes"], [])

    print("tensor layout:")
    # The load-bearing claim: body experts are canonical stacked-3D, so the
    # ORNITH-1.0 un-stacking branch (54e99d9d) is NOT on this model's path.
    body_stacked = [k for k in wm if k.startswith("model.language_model.")
                    and k.endswith(".mlp.experts.gate_up_proj")]
    body_unstacked = [k for k in wm if k.startswith("model.language_model.")
                      and ".mlp.experts." in k and k.endswith(".gate_proj.weight")]
    check("body stacked gate_up_proj count", len(body_stacked), 40)
    check("body UN-stacked expert count (must be 0)", len(body_unstacked), 0)

    shared = [k for k in wm if k.startswith("model.language_model.")
              and ".mlp.shared_expert." in k]
    # Scope to the body namespace, exactly like the `shared` filter above: the
    # MTP module carries its OWN shared_expert_gate, so an unscoped filter
    # returns 41 and the right fix is to tighten the filter, not to expect 41.
    shared_gate = [k for k in wm if k.startswith("model.language_model.")
                   and k.endswith(".mlp.shared_expert_gate.weight")]
    check("shared_expert tensors (3/layer)", len(shared), 120)
    check("shared_expert_gate tensors", len(shared_gate), 40)

    # The MTP module's own shared expert — relevant to Task 8, which must carry it.
    mtp_shared_gate = [k for k in wm if k.startswith("mtp.")
                       and k.endswith(".mlp.shared_expert_gate.weight")]
    check("mtp shared_expert_gate tensors", len(mtp_shared_gate), 1)

    # The inverse claim: the MTP module's experts ARE un-stacked.
    mtp_unstacked = [k for k in wm if k.startswith("mtp.layers.0.mlp.experts.")
                     and k.endswith(".gate_proj.weight")]
    check("mtp UN-stacked expert count", len(mtp_unstacked), 256)

    ns = {"language_model": 0, "mtp": 0, "visual": 0, "other": 0}
    for k in wm:
        if k.startswith("model.language_model."):
            ns["language_model"] += 1
        elif k.startswith("mtp."):
            ns["mtp"] += 1
        elif k.startswith("model.visual."):
            ns["visual"] += 1
        else:
            ns["other"] += 1
    print(f"  namespaces: {ns}  total={len(wm)}")
    check("total tensors", len(wm), 1811)
    check("visual tensors", ns["visual"], 333)

    if fail:
        print(f"\nFAILED {len(fail)} assumption(s): {fail}")
        print("The plan's layout premises do not hold. STOP and re-scope.")
        return 1
    print("\nAll layout assumptions hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
