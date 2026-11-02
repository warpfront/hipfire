"""
Spike B0 — can per-row-ternary Maple weights be packed into the MQ2-Lloyd
container (72 B / 256) with a K=3 codebook and NO FWHT, bit-exactly?

Mirrors the tree's byte layout exactly (quant_mq.rs:1598 dequantize_mq2g256_lloyd_to_f32):
  [0..8)  : 4 x fp16 codebook entries, sorted ascending
  [8..72) : 64 bytes of 2-bit indices, 4 per byte, LSB-first
minus the cpu_inv_fwht_256 step.

Run against REAL published Maple tensors, plus a negative control on a
non-ternary tensor so we know the test can fail.
"""
import json, struct, urllib.request
import sys
import numpy as np

URL = "https://huggingface.co/deepgrove/maple-preview/resolve/main/model-00001-of-00009.safetensors"
BASE = 8 + 227136
GROUP, BLOCK_BYTES = 256, 72

# Populated in main() so import is side-effect free.
hdr = None


def fetch(key, maxbytes=8_000_000):
    o = hdr[key]["data_offsets"]
    a, b = BASE + o[0], min(BASE + o[1], BASE + o[0] + maxbytes) - 1
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    ).read()
    u = np.frombuffer(raw, dtype=np.uint16)
    f = np.zeros(len(u), dtype=np.uint32)
    f[:] = u
    f <<= 16
    return f.view(np.float32)


def pack_k3_no_fwht(vals):
    """vals: 1-D f32, len % 256 == 0. Returns (bytes, stats)."""
    n_blocks = len(vals) // GROUP
    out = bytearray(n_blocks * BLOCK_BYTES)
    max_levels = 0
    for b in range(n_blocks):
        g = vals[b * GROUP:(b + 1) * GROUP]
        levels = np.unique(g)              # K=3 codebook == the distinct values
        max_levels = max(max_levels, len(levels))
        if len(levels) > 4:
            # Not ternary: fall back to 4 quantile-seeded centroids. Lossy by
            # construction — this is the negative-control path.
            levels = np.quantile(g, [0.125, 0.375, 0.625, 0.875]).astype(np.float32)
        cb = np.zeros(4, dtype=np.float32)
        cb[:len(levels)] = levels          # already ascending (np.unique sorts)
        if len(levels) < 4:                # slot 3 duplicates the top slot
            cb[len(levels):] = levels[-1]
        cb16 = cb.astype(np.float16)       # kernel reads fp16
        off = b * BLOCK_BYTES
        out[off:off + 8] = cb16.tobytes()
        # nearest-codepoint assignment (exact hit when g's values ARE the levels)
        idx = np.abs(g[:, None] - levels[None, :]).argmin(axis=1).astype(np.uint8)
        packed = (idx[0::4] | (idx[1::4] << 2) | (idx[2::4] << 4) | (idx[3::4] << 6))
        out[off + 8:off + 72] = packed.astype(np.uint8).tobytes()
    return bytes(out), max_levels


def dequant(data, n):
    """Mirror of dequantize_mq2g256_lloyd_to_f32 WITHOUT cpu_inv_fwht_256."""
    n_blocks = n // GROUP
    out = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        blk = data[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        cb = np.frombuffer(blk[0:8], dtype=np.float16).astype(np.float32)
        by = np.frombuffer(blk[8:72], dtype=np.uint8)
        idx = np.empty(GROUP, dtype=np.uint8)
        for j in range(4):
            idx[j::4] = (by >> (2 * j)) & 0x3
        out[b * GROUP:(b + 1) * GROUP] = cb[idx]
    return out


def report(key, expect_exact):
    v = fetch(key)
    v = v[:(len(v) // GROUP) * GROUP]
    data, max_levels = pack_k3_no_fwht(v)
    recon = dequant(data, len(v))

    rb, vb = recon.view(np.uint32), v.view(np.uint32)
    bit_exact = np.array_equal(rb, vb)
    max_err = float(np.max(np.abs(recon - v)))
    # Where bits differ, is it ONLY the sign bit of zero (-0.0 vs +0.0)?
    diff = rb != vb
    n_diff = int(diff.sum())
    signed_zero_only = bool(
        n_diff and np.all((recon[diff] == 0.0) & (v[diff] == 0.0))
    )
    value_exact = bit_exact or (signed_zero_only and max_err == 0.0)
    bpw = BLOCK_BYTES * 8 / GROUP
    verdict = "PASS" if value_exact == expect_exact else "*** UNEXPECTED ***"

    print(f"\n{key}")
    print(f"  shape={hdr[key]['shape']}  weights sampled={len(v)}  ({bpw} bpw)")
    print(f"  max distinct values in any 256-block: {max_levels}")
    print(f"  max |err|: {max_err:.6g}   bitwise-identical: {bit_exact}")
    if n_diff:
        print(f"  differing words: {n_diff} ({n_diff/len(v):.2%})"
              f"  all are signed-zero only: {signed_zero_only}")
    print(f"  VALUE-exact: {value_exact}   expected={expect_exact}  ->  {verdict}")
    return value_exact


def main(argv=None):
    global hdr
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(
            "usage: python3 -m tools.models.maple.b0_ternary_exactness_spike\n"
            "  Reads shard1_header.json from cwd; fetches Maple tensors over HTTP."
        )
        return 0

    hdr = json.load(open("shard1_header.json"))

    print("=" * 70)
    print("ARM B — MQ2-Lloyd container, K=3 codebook, NO FWHT")
    print("=" * 70)

    ok_tern = report("model.layers.0.mlp.experts.0.gate_proj.weight", expect_exact=True)
    ok_down = report("model.layers.0.mlp.experts.0.down_proj.weight", expect_exact=True)
    ok_attn = report("model.layers.0.self_attn.q_proj.weight", expect_exact=True)

    print("\n" + "-" * 70)
    print("NEGATIVE CONTROL — a NON-ternary tensor must NOT round-trip exactly.")
    print("If this came back exact, the test would be vacuous.")
    print("-" * 70)
    ok_neg = report("lm_head.weight", expect_exact=False)

    print("\n" + "=" * 70)
    print("bf16 scale -> fp16 representability (the exactness precondition):")
    q = fetch("model.layers.0.self_attn.q_proj.weight")
    rows = q[:1953 * 2048].reshape(1953, 2048)
    scales = np.array([np.max(np.abs(r)) for r in rows], dtype=np.float32)
    rt = scales.astype(np.float16).astype(np.float32)
    print(f"  {len(scales)} row scales, range {scales.min():.6g}..{scales.max():.6g}")
    print(f"  survive bf16->fp16->f32 unchanged: {np.array_equal(scales, rt)}")

    print("\nB0 VERDICT:", "ARM B IS EXACT" if (ok_tern and ok_down and ok_attn and not ok_neg)
          else "FAILED — see above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
