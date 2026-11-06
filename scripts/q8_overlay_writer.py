#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""LAB (measurement branch, not for landing): Q8F16 trunk-overlay payload writer.

Encodes the BF16 QSA attention tensors (per FullAttention layer: q/k/v/o and the indexer
`index_qk_proj`) of a Qwen3.8-Flash-Next checkpoint to the Q8F16 (qt3, `DType::Q8_0`) rows
`hipfire-quantize trunk-overlay --format qt3` packs into a `.trunk-overlay` sidecar:
34 bytes per 32 weights (f16 scale + 32 int8), `m * (k / 32) * 34` bytes per `[m, k]` tensor.

The encoder is a bit-exact numpy port of the producer's `quantize_q8f16`
(crates/hipfire-quantize/src/quant_q4.rs) + `f32_to_f16` (float16.rs, which TRUNCATES rather
than rounding to nearest even), as called for DType::Q8_0 by qwen4.rs on BF16->f32 rows:

  per 32 consecutive weights (K % 32 == 0, so a group never straddles rows):
    max_abs = max|w| (f32); scale = max_abs / 127.0 (f32); inv = 1/scale if scale > 0 else 0
    stored scale = f32_to_f16(scale) (truncating); q = clamp(round_half_away(w * inv), -128, 127)
  (the quantiser divides by the f32 scale, NOT the stored f16 one.)

The port was first written and byte-verified against a verbatim rustc build of the Rust encoder
by MultiCard.QuantScope (mgmtp/quant/tools/q8f16_nmse.py); scripts/test_q8_overlay_writer.py
re-verifies it against scripts/q8_overlay_ref/q8f16_ref.rs when rustc is available.

Usage (needs numpy only; run where the BF16 safetensors live):
  q8_overlay_writer.py encode --src DIR --out DIR [--tier attn|gdn_out|gdn_in|head|full] [--list]
                                                              # DIR has model.safetensors.index.json
  q8_overlay_writer.py suffixes --out DIR                    # the `--suffix` args for hipfire-quantize
Writes <out>/<tensor name>.bin per tensor (the `--qt44-payload-dir` layout) and
<out>/q8_overlay_payloads.json (name, shape, bytes, sha256 of each .bin and of the source bytes).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

GS = 32
BLOCK_BYTES = 34
# The trunk QSA roles the overlay replaces (MTP `mtp.layers.0.self_attn.*` never matches: the
# pattern is anchored on `model.language_model.layers.N`).
QSA_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj)\.weight$"
)
# The remaining MQ6 (qt47) trunk tensors of the shipped artifact: per GDN layer the in-projections
# (qkv, z, a, b) and out_proj, and the head. Routers, shared experts, hyper-connection, conv and PLE are
# source-exact BF16 in the artifact (a Q8F16 swap would LOWER their accuracy) and are never targets.
GDN_IN_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b)\.weight$"
)
GDN_OUT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$")
HEAD_NAME = "lm_head.weight"
# tier -> (target count in the Flash-Next checkpoint, matcher classes). `full` is all 241 MQ6 tensors.
TIERS = {
    "attn": (60, ("attn",)),
    "gdn_out": (36, ("gdn_out",)),
    "gdn_in": (144, ("gdn_in",)),
    "head": (1, ("head",)),
    "full": (241, ("attn", "gdn_in", "gdn_out", "head")),
}
ROW_BLOCK = 4096


def f32_to_f16_trunc(x):
    """Bit-exact vector port of float16.rs f32_to_f16 (truncating). float32 array -> uint16 array."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    sign = (bits >> 31) & 1
    exp = ((bits >> 23) & 0xFF).astype(np.int64)
    frac = bits & 0x7FFFFF
    new_exp = exp - 127 + 15
    f16_frac = np.where(frac == 0, 0, (frac >> 13) | 1).astype(np.uint32)
    special = (sign << 15) | (0x1F << 10) | f16_frac
    over = (sign << 15) | (0x1F << 10)
    shift = np.clip(1 - new_exp + 13, 0, 31).astype(np.uint32)
    sub = (sign << 15) | (((frac | 0x800000) >> shift) & 0xFFFFFFFF)
    under = sign << 15
    normal = (sign << 15) | ((np.clip(new_exp, 0, 31).astype(np.uint32)) << 10) | (frac >> 13)
    out = np.where(
        exp == 0xFF,
        special,
        np.where(
            new_exp >= 31,
            over,
            np.where(new_exp <= 0, np.where(new_exp < -10, under, sub), normal),
        ),
    )
    return (out & 0xFFFF).astype(np.uint16)


def round_half_away(x):
    """Rust f32::round: nearest integer, ties away from zero (exact in f32)."""
    t = np.trunc(x)
    d = x - t
    return np.where(np.abs(d) >= np.float32(0.5), t + np.sign(x), t).astype(np.float32)


def quantize_q8f16_np(v):
    """float32 values (flat, finite) -> (f16 scale bits uint16 [nb], q int8 [nb, 32]); tail zero padded."""
    v = np.ascontiguousarray(v, dtype=np.float32).reshape(-1)
    n = v.size
    nb = -(-n // GS)
    if nb * GS != n:
        v = np.concatenate([v, np.zeros(nb * GS - n, np.float32)])
    g = v.reshape(nb, GS)
    max_abs = np.abs(g).max(axis=1)
    scale = (max_abs / np.float32(127.0)).astype(np.float32)
    with np.errstate(all="ignore"):
        inv = np.where(
            scale > 0,
            np.float32(1.0) / np.where(scale > 0, scale, np.float32(1.0)),
            np.float32(0.0),
        ).astype(np.float32)
        prod = (g * inv[:, None]).astype(np.float32)  # tiny scale: inv=inf, 0*inf=NaN -> Rust NaN.max(-128) == -128
        r = round_half_away(prod)
        q = np.where(np.isnan(r), -128.0, np.clip(r, -128.0, 127.0)).astype(np.int8)
    return f32_to_f16_trunc(scale), q


def encode_bytes(v):
    """The `quantize_q8f16` output bytes of float32 values `v`."""
    s, q = quantize_q8f16_np(v)
    out = np.empty((s.size, BLOCK_BYTES), np.uint8)
    out[:, :2] = s.astype("<u2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1).tobytes()


def expected_bytes(m, k):
    if k <= 0 or k % GS:
        raise ValueError(f"K={k} is not a positive multiple of {GS}")
    return m * (k // GS) * BLOCK_BYTES


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Source:
    """Minimal safetensors reader (BF16 tensors) driven by model.safetensors.index.json."""

    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(self.root / shard, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self._headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self._headers[shard]

    def names(self):
        return sorted(self.weight_map)

    def info(self, name):
        shard = self.weight_map[name]
        header, base = self._header(shard)
        entry = header[name]
        lo, hi = entry["data_offsets"]
        return shard, entry["dtype"], list(entry["shape"]), base + lo, hi - lo

    def rows(self, name, start, count):
        """float32 rows [start, start+count) of a rank-2 BF16 tensor."""
        shard, dtype, shape, off, size = self.info(name)
        if dtype != "BF16" or len(shape) != 2:
            raise ValueError(f"{name}: dtype {dtype} shape {shape}, expected a rank-2 BF16 tensor")
        m, k = shape
        if size != m * k * 2:
            raise ValueError(f"{name}: {size} bytes for [{m}, {k}] BF16")
        count = min(count, m - start)
        with open(self.root / shard, "rb") as f:
            f.seek(off + start * k * 2)
            raw = np.frombuffer(f.read(count * k * 2), dtype="<u2")
        return bf16_bits_to_f32(raw).reshape(count, k)


def qsa_names(names):
    """Trunk QSA overlay targets in (layer, role) order; exactly 5 per FullAttention layer."""
    found = []
    for name in names:
        match = QSA_RE.match(name)
        if match:
            found.append((int(match.group(1)), name))
    found.sort()
    return [name for _, name in found]


def tier_names(names, tier):
    """Overlay targets of `tier` in (class, layer, name) order; the count must be the tier's registered one."""
    want, classes = TIERS[tier]
    found = []
    for name in names:
        for cls in classes:
            if cls == "attn":
                match = QSA_RE.match(name)
            elif cls == "gdn_in":
                match = GDN_IN_RE.match(name)
            elif cls == "gdn_out":
                match = GDN_OUT_RE.match(name)
            else:
                match = name == HEAD_NAME
            if match:
                layer = int(match.group(1)) if match is not True else -1
                found.append((classes.index(cls), layer, name))
                break
    found.sort()
    # `attn` keeps the original "multiple of 5 per FullAttention layer" check (60 on Flash-Next; the unit test
    # checkpoint is smaller); every other tier registers its exact checkpoint count.
    ok = bool(found) and len(found) % 5 == 0 if tier == "attn" else len(found) == want
    if not ok:
        raise SystemExit(f"tier {tier}: found {len(found)} tensors, expected {want}")
    return [name for _, _, name in found]


def encode_tensor(src, name, out_dir):
    """Encode one tensor to <out_dir>/<name>.bin (atomic); returns its manifest record."""
    shard, dtype, shape, _, _ = src.info(name)
    m, k = shape
    want = expected_bytes(m, k)
    final = Path(out_dir) / f"{name}.bin"
    part = final.with_suffix(final.suffix + ".partial")
    h_out, h_src = hashlib.sha256(), hashlib.sha256()
    written = 0
    with open(part, "wb") as f:
        for r0 in range(0, m, ROW_BLOCK):
            rows = src.rows(name, r0, ROW_BLOCK)
            h_src.update(np.ascontiguousarray(rows).tobytes())
            blob = encode_bytes(rows)
            f.write(blob)
            h_out.update(blob)
            written += len(blob)
        f.flush()
        os.fsync(f.fileno())
    if written != want:
        part.unlink()
        raise RuntimeError(f"{name}: wrote {written} bytes, expected {want}")
    os.replace(part, final)
    return {
        "name": name,
        "shape": [m, k],
        "bytes": written,
        "sha256": h_out.hexdigest(),
        "source_f32_sha256": h_src.hexdigest(),
        "source_shard": shard,
    }


def cmd_encode(args):
    src = Source(args.src)
    names = tier_names(src.names(), args.tier)
    if args.list:
        print("\n".join(names))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        records.append(encode_tensor(src, name, out))
        print(f"{name} {records[-1]['bytes']} {records[-1]['sha256'][:12]}", flush=True)
    manifest = {
        "format": "qt3",
        "tier": args.tier,
        "encoder": "quantize_q8f16 + f32_to_f16 (truncating), numpy port",
        "tensors": records,
        "total_bytes": sum(r["bytes"] for r in records),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "q8_overlay_payloads.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(records)} tensors, {manifest['total_bytes']} bytes")
    return 0


def cmd_suffixes(args):
    manifest = json.loads((Path(args.out) / "q8_overlay_payloads.json").read_text())
    for record in manifest["tensors"]:
        # The full tensor name is its own suffix and cannot match the MTP head's twin.
        print("--suffix", record["name"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--src", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tier", choices=sorted(TIERS), default="attn")
    e.add_argument("--list", action="store_true")
    e.set_defaults(fn=cmd_encode)
    s = sub.add_parser("suffixes")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_suffixes)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
