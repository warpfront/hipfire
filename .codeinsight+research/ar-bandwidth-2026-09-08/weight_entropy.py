#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only lossless-entropy probe of real qt44 (MQ4G256V2) weight bytes.

Reads the actual model file
  /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
  (-> /home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.hfq,
     14980361216 B, md5 e45d15bfe0c9a87132697101d17cbed6)
with streaming File+seek reads only (never rewrites or copies the model).

For each sampled tensor (qt=44 MQ4G256V2: 136 B/group = 8 B header
[s0,z0,s1,z1 fp16 LE at 0..8] + 128 B 4-bit codes at 8..136, 256 weights,
halves 0..128 / 128..256) it reports:
  - tensor name / dims / codec, exact logical header+code bytes
  - sampled nibble histogram + Shannon entropy, explicitly labeled as a
    LOWER-BOUND model (not achievable compression)
  - level-0 nibble fraction and identical-header rates
  - block-local sizes from stock codecs (zlib/lzma/bz2) as FEASIBILITY
    EVIDENCE ONLY (not GPU performance, not a codec proposal), each with
    decompress-equality verification
  - sha256 digests of the sampled bytes

Writes weight_entropy.json next to this script and prints a summary table.
Stdlib only. No GPU, no harness build, no repo edits.
"""

import bz2
import hashlib
import json
import lzma
import math
import os
import struct
import sys
import zlib
from collections import Counter

MODEL_PATH = "/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt"
EXPECTED_SIZE = 14980361216
EXPECTED_MD5 = "e45d15bfe0c9a87132697101d17cbed6"

GROUP_BYTES = 136
HEADER_BYTES = 8
CODE_BYTES = 128
WEIGHTS_PER_GROUP = 256
QT44 = 44

# Bounded stratified sample: (suffix-match rule, groups to sample).
# Suffixes are matched against the full HFQ tensor name.
SAMPLE_PLAN = [
    ("lm_head.weight", 4096),
    ("layers.0.linear_attn.in_proj_qkv.weight", 2048),
    ("layers.0.mlp.gate_proj.weight", 2048),
    ("layers.0.mlp.down_proj.weight", 2048),
    ("layers.F.self_attn.q_proj.weight", 2048),   # F = first full-attn layer
    ("layers.F.self_attn.o_proj.weight", 2048),
    ("layers.32.mlp.up_proj.weight", 2048),
    ("layers.63.mlp.down_proj.weight", 2048),
]
HEAD_GROUPS = 512  # first HEAD_GROUPS groups read densely, rest strided


def u32le(b, o=0):
    return struct.unpack_from("<I", b, o)[0]


def u64le(b, o=0):
    return struct.unpack_from("<Q", b, o)[0]


def parse_hfq_index(path):
    """Replicate hipfire-runtime/src/hfq.rs HfqFile::open_at_offset index
    layout + bench_dflash_verify_shapes.rs: 32 B header, brace-scan for the
    metadata JSON end, then the tensor index. Read-only."""
    canon = os.path.realpath(path)
    with open(canon, "rb") as f:
        hdr = f.read(32)
        assert hdr[:4] == b"HFQM", "not an HFQ container"
        n_tensors = u32le(hdr, 12)
        metadata_offset = u64le(hdr, 16)
        data_offset = u64le(hdr, 24)
        assert metadata_offset <= data_offset, "bad meta/data offsets"
        f.seek(metadata_offset)
        region = f.read(data_offset - metadata_offset)
    depth = 0
    in_str = False
    esc = False
    json_end = 0
    for i, b in enumerate(region):
        if esc:
            esc = False
            continue
        if b == 0x5C and in_str:  # backslash
            esc = True
            continue
        if b == 0x22:  # double quote
            in_str = not in_str
            continue
        if not in_str:
            if b == 0x7B:  # {
                depth += 1
            elif b == 0x7D:  # }
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
    assert json_end > 0, "metadata JSON not brace-terminated"
    meta_json = region[:json_end].decode("utf-8", "replace")
    pos = json_end
    idx_n = u32le(region, pos)
    assert idx_n == n_tensors, "index count != header count"
    pos += 4
    tensors = []
    cum = data_offset
    for _ in range(n_tensors):
        nl = struct.unpack_from("<H", region, pos)[0]
        pos += 2
        name = region[pos:pos + nl].decode()
        pos += nl
        qt = region[pos]
        pos += 1
        nd = region[pos]
        pos += 1
        shape = [u32le(region, pos + 4 * i) for i in range(nd)]
        pos += 4 * nd
        pos += 4  # group_size
        data_len = u64le(region, pos)
        pos += 8
        tensors.append(
            {"name": name, "qt": qt, "shape": shape,
             "data_off": cum, "data_len": data_len}
        )
        cum += data_len
    index_bytes = region[json_end:pos]
    return canon, meta_json, tensors, index_bytes


def shannon_entropy(counts, total):
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def sample_group_indices(n_groups, target, head=HEAD_GROUPS):
    """Deterministic policy: first `head` groups dense, remainder evenly
    strided over [head, n_groups). Returns sorted index list + policy dict."""
    if n_groups <= target:
        idx = list(range(n_groups))
        policy = {"mode": "full", "n_groups": n_groups}
        return idx, policy
    rest = target - head
    span = n_groups - head
    stride = span / rest
    idx = list(range(head))
    idx += [head + int(i * stride) for i in range(rest)]
    idx = sorted(set(idx))
    policy = {"mode": "head+strided", "head_groups": head,
              "stride": stride, "span_start": head, "span_end": n_groups,
              "n_groups": n_groups, "n_sampled": len(idx)}
    return idx, policy


def read_groups(f, data_off, indices):
    """Coalesce sorted group indices into runs; one read per run."""
    out = bytearray()
    order = []  # (group_index, offset_in_out)
    sorted_idx = sorted(indices)
    # build runs of consecutive indices
    runs = []
    start = prev = sorted_idx[0]
    for g in sorted_idx[1:]:
        if g == prev + 1:
            prev = g
        else:
            runs.append((start, prev))
            start = prev = g
    runs.append((start, prev))
    for a, b in runs:
        f.seek(data_off + a * GROUP_BYTES)
        chunk = f.read((b - a + 1) * GROUP_BYTES)
        assert len(chunk) == (b - a + 1) * GROUP_BYTES, "short tensor read"
        base = len(out)
        out += chunk
        for k, g in enumerate(range(a, b + 1)):
            order.append((g, base + k * GROUP_BYTES))
    order.sort()
    return bytes(out), runs


def try_codecs(blob):
    """Feasibility evidence only: stock-codec sizes + round-trip check."""
    res = {}
    variants = {
        "zlib9": (zlib.compress, zlib.decompress, {"level": 9}),
        "lzma6": (lzma.compress, lzma.decompress, {"preset": 6}),
        "bz2_9": (bz2.compress, bz2.decompress, {"compresslevel": 9}),
    }
    for key, (enc, dec, kw) in variants.items():
        try:
            comp = enc(blob, **kw)
        except Exception as e:  # codec unavailable -> record, don't fail
            res[key] = {"error": f"{type(e).__name__}: {e}"}
            continue
        rt = dec(comp)
        res[key] = {
            "compressed_bytes": len(comp),
            "ratio": len(comp) / len(blob) if blob else 0.0,
            "decompress_equals_input": rt == blob,
            "decompress_sha256": hashlib.sha256(rt).hexdigest(),
        }
    return res


def analyze_tensor(f, t, target_groups):
    name = t["name"]
    m, k = t["shape"][0], t["shape"][1]
    assert t["qt"] == QT44, f"{name}: qt={t['qt']} != 44"
    assert k % 256 == 0, f"{name}: K%256 != 0"
    gpr = k // 256
    n_groups = m * gpr
    expected_len = n_groups * GROUP_BYTES
    assert t["data_len"] == expected_len, (
        f"{name}: data_len {t['data_len']} != M*K/256*136 {expected_len}")
    logical_header_bytes = n_groups * HEADER_BYTES
    logical_code_bytes = n_groups * CODE_BYTES

    indices, policy = sample_group_indices(n_groups, target_groups)
    raw, runs = read_groups(f, t["data_off"], indices)
    assert len(raw) == len(indices) * GROUP_BYTES

    headers = bytearray()
    codes = bytearray()
    # raw is in sorted-group order; slice sequentially
    nib_hist = [0] * 16
    byte_hist = [0] * 256
    hdr_byte_hist = [0] * 256
    hdr_set = set()
    half_set = set()
    n_nib = 0
    level0 = 0
    for i in range(len(indices)):
        g = raw[i * GROUP_BYTES:(i + 1) * GROUP_BYTES]
        h, c = g[:HEADER_BYTES], g[HEADER_BYTES:]
        headers += h
        codes += c
        hdr_set.add(bytes(h))
        half_set.add(bytes(h[:4]))
        half_set.add(bytes(h[4:]))
        for b in h:
            hdr_byte_hist[b] += 1
        for b in c:
            byte_hist[b] += 1
            lo, hi = b & 0xF, b >> 4
            nib_hist[lo] += 1
            nib_hist[hi] += 1
            n_nib += 2
            level0 += (lo == 0) + (hi == 0)
    headers, codes = bytes(headers), bytes(codes)

    nib_entropy = shannon_entropy(nib_hist, n_nib)
    code_byte_entropy = shannon_entropy(byte_hist, len(codes))
    hdr_byte_entropy = shannon_entropy(hdr_byte_hist, len(headers))

    region = {
        "tensor": name,
        "codec": "qt44/MQ4G256V2",
        "shape_M_K": [m, k],
        "data_off": t["data_off"],
        "data_len": t["data_len"],
        "groups_total": n_groups,
        "groups_per_row": gpr,
        "logical_header_bytes": logical_header_bytes,
        "logical_code_bytes": logical_code_bytes,
        "logical_total_bytes": expected_len,
        "sample_policy": policy,
        "sample_group_runs": [[a, b] for a, b in runs],
        "sampled_groups": len(indices),
        "sampled_bytes": len(raw),
        "sampled_sha256": hashlib.sha256(raw).hexdigest(),
        "nibble_histogram": nib_hist,
        "nibble_shannon_bits_per_nibble": nib_entropy,
        "nibble_entropy_lower_bound_note": (
            "Shannon H of the sampled nibble histogram is a LOWER-BOUND "
            "model only: it assumes i.i.d. nibbles and ignores all header "
            "cost, codebook overhead and finite-block effects. It is NOT "
            "an achievable compression ratio."),
        "code_byte_histogram": byte_hist,
        "code_byte_shannon_bits_per_byte": code_byte_entropy,
        "header_byte_shannon_bits_per_byte": hdr_byte_entropy,
        "level0_nibble_fraction": level0 / n_nib,
        "distinct_8B_headers": len(hdr_set),
        "identical_header_rate": 1.0 - len(hdr_set) / len(indices),
        "distinct_4B_half_headers": len(half_set),
        "compression_feasibility_only": {
            "note": ("Stock-codec sizes on the SAMPLED bytes as feasibility "
                     "evidence only. Not GPU-decodable, not a codec proposal, "
                     "not a tok/s claim; quantization unchanged."),
            "codes_128B_per_group": try_codecs(codes),
            "headers_8B_per_group": try_codecs(headers),
            "interleaved_groups": try_codecs(raw),
        },
    }
    return region


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "weight_entropy.json")
    size = os.path.getsize(MODEL_PATH)
    assert size == EXPECTED_SIZE, f"model size {size} != {EXPECTED_SIZE}"
    canon, meta_json, tensors, index_bytes = parse_hfq_index(MODEL_PATH)

    # full-file identity hash (streaming, ~15 GB)
    md5 = hashlib.md5()
    with open(canon, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    file_md5 = md5.hexdigest()
    assert file_md5 == EXPECTED_MD5, f"model md5 {file_md5} != expected"

    by_name = {t["name"]: t for t in tensors}
    qt44 = [t for t in tensors if t["qt"] == QT44]
    # first full-attention layer = lowest layer idx with self_attn.q_proj
    import re
    fa_layers = sorted({int(m.group(1)) for t in qt44
                        for m in [re.search(r"layers\.(\d+)\.self_attn\.q_proj",
                                            t["name"])] if m})
    assert fa_layers, "no full-attention q_proj found"
    F = fa_layers[0]

    plan = []
    for suffix, ng in SAMPLE_PLAN:
        plan.append((suffix.replace("layers.F.", f"layers.{F}."), ng))

    regions = []
    with open(canon, "rb") as f:
        for suffix, ng in plan:
            hits = [t for t in tensors if t["name"].endswith(suffix)]
            assert len(hits) == 1, f"tensor suffix {suffix}: {len(hits)} hits"
            regions.append(analyze_tensor(f, hits[0], ng))

    total_qt44 = sum(t["data_len"] for t in qt44)
    sampled = sum(r["sampled_bytes"] for r in regions)
    # Exploratory ceiling: extrapolate the best sampled interleaved ratio
    # (computed properly into per_region_best below).
    per_region_best = {}
    for r in regions:
        ig = r["compression_feasibility_only"]["interleaved_groups"]
        ratios = [v["ratio"] for v in ig.values()
                  if isinstance(v, dict) and v.get("decompress_equals_input")
                  and "ratio" in v]
        per_region_best[r["tensor"]] = min(ratios) if ratios else None
    valid = [v for v in per_region_best.values() if v is not None]
    exploratory = {
        "label": ("EXPLORATORY ceiling only: mean of per-region best sampled "
                  "stock-codec ratios applied to total qt44 bytes. Not a "
                  "measured total, not GPU-decodable, not a performance "
                  "claim."),
        "mean_best_sampled_ratio": sum(valid) / len(valid) if valid else None,
        "total_qt44_bytes": total_qt44,
        "ceiling_bytes": int(total_qt44 * sum(valid) / len(valid))
        if valid else None,
        "per_region_best_ratio": per_region_best,
    }

    nib_all = [0] * 16
    nib_n = 0
    for r in regions:
        for i, c in enumerate(r["nibble_histogram"]):
            nib_all[i] += c
            nib_n += c
    pooled_h = shannon_entropy(nib_all, nib_n)

    result = {
        "input": {
            "path": MODEL_PATH,
            "canonical_path": canon,
            "size_bytes": size,
            "md5": file_md5,
            "hfq_tensors": len(tensors),
            "qt44_tensors": len(qt44),
            "qt_histogram": {str(q): sum(1 for t in tensors if t["qt"] == q)
                             for q in {t["qt"] for t in tensors}},
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "first_full_attn_layer": F,
            "total_qt44_bytes": total_qt44,
        },
        "format": ("qt44/MQ4G256V2: 136 B/group, header [s0,z0,s1,z1] fp16 LE "
                   "at 0..8, 128 B 4-bit codes at 8..136, 256 weights/group, "
                   "halves 0..128/128..256. Stored headers/nibbles preserved "
                   "exactly; DOG/4-acc/reduction order untouched; no "
                   "quantization or FP change."),
        "regions": regions,
        "pooled_sampled_nibble_histogram": nib_all,
        "pooled_sampled_nibble_shannon_bits_per_nibble": pooled_h,
        "exploratory_total_bit_ceiling": exploratory,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"model: {canon} ({size} B, md5 {file_md5})")
    print(f"tensors: {len(tensors)} total, {len(qt44)} qt44, "
          f"{total_qt44} qt44 bytes; first FA layer {F}")
    print(f"pooled nibble H = {pooled_h:.4f} bits/nibble "
          f"(lower-bound model, not achievable)")
    print(f"{'tensor':55s} {'groups':>9s} {'H_nib':>6s} "
          f"{'lvl0%':>6s} {'dupHdr%':>8s} {'bestCodec':>9s}")
    for r in regions:
        best = per_region_best[r["tensor"]]
        print(f"{r['tensor'][-55:]:55s} {r['groups_total']:9d} "
              f"{r['nibble_shannon_bits_per_nibble']:6.3f} "
              f"{100*r['level0_nibble_fraction']:6.2f} "
              f"{100*r['identical_header_rate']:8.2f} "
              f"{best:9.3f}" if best else f"{r['tensor'][-55:]:55s} n/a")
    if exploratory["mean_best_sampled_ratio"] is not None:
        print(f"exploratory ceiling: ratio "
              f"{exploratory['mean_best_sampled_ratio']:.3f} -> "
              f"{exploratory['ceiling_bytes']} bytes total qt44")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} B)")


if __name__ == "__main__":
    sys.exit(main())
