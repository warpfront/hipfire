#!/usr/bin/env python3
"""Compare safetensors structure between two PARO checkpoints to find
why one admits the batched path and the other falls back to per-token.
"""
import os
import sys
from safetensors import safe_open


def dump_paro(path: str, label: str) -> dict:
    """Inspect a PARO checkpoint dir."""
    print(f"\n=== {label}: {path} ===")
    st_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not st_files:
        print("  (no .safetensors files)")
        return {}
    info = {}
    for fn in sorted(st_files):
        full = os.path.join(path, fn)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  file: {fn}  ({size_mb:.1f} MB)")
        with safe_open(full, framework="numpy") as st:
            keys = list(st.keys())
            info["keys"] = keys
            print(f"    total tensors: {len(keys)}")
            l0e0 = [k for k in keys if "layers.0" in k and "experts.0" in k]
            l0_shared = [
                k for k in keys
                if "layers.0" in k and "shared_expert" in k and "gate" in k.lower()
            ][:8]
            print(f"  layers.0.experts.0 has {len(l0e0)} tensors:")
            for k in sorted(l0e0)[:15]:
                t = st.get_tensor(k)
                print(f"    {k}  shape={list(t.shape)} dtype={t.dtype}")
            print(f"  layers.0.shared_expert.gate first 5 tensors:")
            for k in sorted(l0_shared)[:5]:
                t = st.get_tensor(k)
                print(f"    {k}  shape={list(t.shape)} dtype={t.dtype}")
            has = {
                "pairs": any("pairs" in k for k in keys),
                "theta": any("theta" in k for k in keys),
                "channel_scales": any("channel_scales" in k for k in keys),
                "qweight": any("qweight" in k for k in keys),
                "qzeros": any("qzeros" in k for k in keys),
                "scales": any("scales" in k for k in keys),
            }
            print("  metadata kinds present:", {k: v for k, v in has.items() if v})
    return info


if __name__ == "__main__":
    a = sys.argv[1]
    b = sys.argv[2]
    ai = dump_paro(a, "A (shisa)")
    bi = dump_paro(b, "B (z-lab)")
    print("\n=== diff keys A-only / B-only (sampled) ===")
    a_keys = set(ai.get("keys", []))
    b_keys = set(bi.get("keys", []))
    a_only = sorted(a_keys - b_keys)[:20]
    b_only = sorted(b_keys - a_keys)[:20]
    if a_only:
        print(f"  A-only ({len(a_keys - b_keys)} total, showing 20):")
        for k in a_only:
            print(f"    {k}")
    if b_only:
        print(f"  B-only ({len(b_keys - a_keys)} total, showing 20):")
        for k in b_only:
            print(f"    {k}")
    if not a_only and not b_only:
        print("  (keys identical between A and B)")
