#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
"""Weight-space distortion sweep across the hipfire quant zoo (CPU-only, no GPU).

WHY THIS AND NOT gamma. The centering-inefficiency gamma = 2M/R of
Helcig/Kurtic/Alistarh (arXiv:2605.02404, Def 3.1) compares two *uniform* grids
that differ only in anchoring, under Bennett's *high-rate* approximation. It is
structurally blind to level *placement* and invalid at 2 bits -- empirically it
scores MQ2 (incoherent) and MQ2-Lloyd (coherent) within 0.8% of each other.

What does transfer is the paper's eq. (2): for y = Wx, quantization gives
delta = (Q(W)-W)x with E[||delta||^2] = sigma^2 ||x||^2, and ||x||^2 cancels
across formats. So RELATIVE WEIGHT MSE is the paper's own per-layer output-error
proxy, with none of the high-rate/uniform-grid assumptions.

Calibrated against known outcomes on qwen3.5-0.8b:
    MQ2G256       5.64 dB  -> incoherent
    MQ2G256Lloyd  9.16 dB  -> coherent   (Lloyd-Max 2-bit Gaussian bound: 9.30 dB)

MODES
  --ref R C...   measure each candidate C against reference R. Valid only when
                 R is materially higher-rate than C (a 4-bit ref carries ~1/25
                 the error variance of a 2-bit grid). Same-rate pairs are NOT
                 rankable this way -- use --triangulate.
  --triangulate A B C
                 three same-rate formats, errors assumed mutually uncorrelated:
                     MSE(A,B) = eA + eB, etc.  =>  eA = (AB + AC - BC)/2
                 Recovers each format's ABSOLUTE error with no reference at all.

LAYOUTS (verified against kernels/src/*.hip)
  group-linear, group = 256 weights, offset = tensor_off + g*bpg
    qt=6/13  HFQ4G256 / MQ4G256     136 B  [f32 scale][f32 zero][128 B nib]  w=s*q+z
    qt=17    MQ3G256                104 B  [f32 scale][f32 zero][96 B 3-bit]
    qt=18    MQ2G256                 72 B  [f32 scale][f32 zero][64 B 2-bit]
    qt=19    MQ2G256Lloyd            72 B  [4  x fp16 cb][64 B 2-bit]        w=cb[q]
    qt=20    MQ3G256Lloyd           112 B  [8  x fp16 cb][96 B 3-bit]
    qt=30    MQ4G256Lloyd           160 B  [16 x fp16 cb][128 B nib]
  row-blocked
    qt=34    MFP4G32E8   row = 16 B hdr (fp16 row_scale) + (K/32) x 17 B blocks
             block = [1 B E4M3 scale][4 x u32 E8 codeword];  w = rs*e4m3*0.88*p
  nibbles low-first; 2/3-bit little-endian bit-packed.
"""
import argparse, struct, sys
import numpy as np

G = 256
AFFINE = {6: (136, 4), 13: (136, 4), 17: (104, 3), 18: (72, 2)}
LLOYD = {19: (72, 2, 4), 20: (112, 3, 8), 30: (160, 4, 16)}
ROWBLK = {34}
NAMES = {6: "HFQ4G256", 13: "MQ4G256", 17: "MQ3G256", 18: "MQ2G256",
         19: "MQ2G256Lloyd", 20: "MQ3G256Lloyd", 21: "HFP4G32",
         30: "MQ4G256Lloyd", 34: "MFP4G32E8", 35: "MFP4G32E8SOA"}
SUPPORTED = set(AFFINE) | set(LLOYD) | ROWBLK


# ---------------------------------------------------------------- container
def parse_hfq(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        if hdr[:4] != b"HFQM":
            raise ValueError(f"{path}: not an HFQM container")
        n_t = struct.unpack_from("<I", hdr, 12)[0]
        moff = struct.unpack_from("<Q", hdr, 16)[0]
        doff = struct.unpack_from("<Q", hdr, 24)[0]
        f.seek(moff)
        blob = f.read(doff - moff)
    depth = ins = esc = 0
    ins = False
    esc = False
    jend = 0
    for i, b in enumerate(blob):
        c = chr(b)
        if esc:
            esc = False
            continue
        if c == "\\" and ins:
            esc = True
            continue
        if c == '"':
            ins = not ins
            continue
        if not ins:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    jend = i + 1
                    break
    pos = jend
    if struct.unpack_from("<I", blob, pos)[0] != n_t:
        raise ValueError(f"{path}: tensor index count mismatch")
    pos += 4
    out, cum = {}, doff
    for _ in range(n_t):
        nl = struct.unpack_from("<H", blob, pos)[0]
        pos += 2
        name = blob[pos:pos + nl].decode("utf-8", "replace")
        pos += nl
        qt = blob[pos]
        pos += 1
        nd = blob[pos]
        pos += 1
        shape = list(struct.unpack_from("<%dI" % nd, blob, pos))
        pos += 4 * nd
        gs = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        dsz = struct.unpack_from("<Q", blob, pos)[0]
        pos += 8
        out[name] = dict(qt=qt, shape=shape, gs=gs, off=cum, size=dsz)
        cum += dsz
    return out


# ---------------------------------------------------------------- unpackers
def unpack_codes(payload, bits):
    """(N, bytes) uint8 -> (N, 256) codes, little-endian bit order."""
    n = payload.shape[0]
    if bits == 4:
        q = np.empty((n, payload.shape[1] * 2), dtype=np.uint8)
        q[:, 0::2] = payload & 0x0F
        q[:, 1::2] = payload >> 4
    elif bits == 2:
        q = np.empty((n, payload.shape[1] * 4), dtype=np.uint8)
        for k in range(4):
            q[:, k::4] = (payload >> (2 * k)) & 0x03
    elif bits == 3:
        tri = payload[:, :96].reshape(n, 32, 3).astype(np.uint32)
        pk = tri[:, :, 0] | (tri[:, :, 1] << 8) | (tri[:, :, 2] << 16)
        q = np.stack([((pk >> (3 * i)) & 0x7) for i in range(8)], axis=-1)
        q = q.reshape(n, 256).astype(np.uint8)
    else:
        raise ValueError(bits)
    return q[:, :G]


def e4m3_to_f32(b):
    b = b.astype(np.uint32)
    exp = (b >> 3) & 0xF
    mant = (b & 0x7).astype(np.float32)
    sub = 0.015625 * mant * 0.125
    pow2 = np.exp2((exp.astype(np.float32) - 7.0))
    nrm = pow2 * (1.0 + mant * 0.125)
    out = np.where(exp == 0, sub, nrm)
    return np.where((exp == 0xF) & (mant == 7.0), 448.0, out).astype(np.float32)


def e8_decode(idx):
    """(N,) uint32 -> (N, 8) float32 lattice points (pre-scale)."""
    idx = idx.astype(np.uint32)
    coset = ((idx >> np.uint32(31)) & np.uint32(1)).astype(np.float32)
    e = np.stack([((idx >> np.uint32(4 * i)) & np.uint32(0xF)) for i in range(7)], axis=-1)
    sl = e.sum(axis=-1, dtype=np.uint32)
    e7_high = (idx >> np.uint32(28)) & np.uint32(0x7)
    p7 = e7_high << np.uint32(1)
    lsb = (sl + p7) & np.uint32(1)
    e7 = (p7 | lsb)[:, None]
    coords = np.concatenate([e, e7], axis=-1).astype(np.float32) - 7.0
    return coords + 0.5 * coset[:, None]


# ---------------------------------------------------------------- dequant
def dequant_groups(mm, t, gidx):
    """Return (len(gidx), 256) float32 for the given global 256-weight groups."""
    qt = t["qt"]
    if qt in AFFINE:
        bpg, bits = AFFINE[qt]
        base = t["off"] + gidx.astype(np.int64) * bpg
        raw = np.stack([mm[b:b + bpg] for b in base])
        sc = raw[:, 0:4].copy().view("<f4").reshape(-1, 1).astype(np.float32)
        ze = raw[:, 4:8].copy().view("<f4").reshape(-1, 1).astype(np.float32)
        return unpack_codes(raw[:, 8:], bits).astype(np.float32) * sc + ze
    if qt in LLOYD:
        bpg, bits, nent = LLOYD[qt]
        base = t["off"] + gidx.astype(np.int64) * bpg
        raw = np.stack([mm[b:b + bpg] for b in base])
        cb = raw[:, :2 * nent].copy().view("<f2").astype(np.float32)
        q = unpack_codes(raw[:, 2 * nent:], bits).astype(np.int64)
        return np.take_along_axis(cb, q, axis=1)
    if qt == 34:
        M, K = t["shape"][0], t["shape"][1]
        gpr = K // G                      # 256-groups per row
        rowb = 16 + (K // 32) * 17
        r = (gidx // gpr).astype(np.int64)
        j = (gidx % gpr).astype(np.int64)
        out = np.empty((len(gidx), G), dtype=np.float32)
        for n in range(len(gidx)):
            ro = t["off"] + r[n] * rowb
            rs = np.frombuffer(mm[ro:ro + 2].tobytes(), dtype="<f2")[0].astype(np.float32)
            b0 = ro + 16 + (8 * j[n]) * 17
            blk = np.stack([mm[b0 + k * 17: b0 + (k + 1) * 17] for k in range(8)])
            bs = e4m3_to_f32(blk[:, 0])                                  # (8,)
            cw = blk[:, 1:17].copy().view("<u4")                          # (8, 4)
            pts = e8_decode(cw.reshape(-1)).reshape(8, 4, 8)              # (8,4,8)
            out[n] = (pts * (rs * bs * 0.88)[:, None, None]).reshape(G)
        return out
    raise ValueError(f"qt={qt} unsupported")


# ---------------------------------------------------------------- driver
def pair_mse(pa, pb, groups_per_tensor, seed):
    ia, ib = parse_hfq(pa), parse_hfq(pb)
    ma = np.memmap(pa, dtype=np.uint8, mode="r")
    mb = np.memmap(pb, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    num = den = 0.0
    qa = qb = None
    nt = 0
    for name, ta in ia.items():
        if ta["qt"] not in SUPPORTED:
            continue
        tb = ib.get(name)
        if tb is None or tb["qt"] not in SUPPORTED:
            continue
        if len(ta["shape"]) < 2 or ta["shape"] != tb["shape"]:
            continue
        M, K = ta["shape"][0], ta["shape"][1]
        if K % G:
            continue
        ngrp = (M * K) // G
        if ngrp == 0:
            continue
        qa, qb = ta["qt"], tb["qt"]
        k = min(groups_per_tensor, ngrp)
        gidx = rng.choice(ngrp, size=k, replace=False)
        A = dequant_groups(ma, ta, gidx)
        B = dequant_groups(mb, tb, gidx)
        num += float(np.sum((A - B) ** 2))
        den += float(np.sum(B ** 2))
        nt += 1
    return num / den if den > 0 else float("nan"), qa, qb, nt


def db(x):
    return -10.0 * np.log10(x) if x > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--triangulate", nargs=3, metavar=("A", "B", "C"))
    ap.add_argument("candidates", nargs="*")
    ap.add_argument("--groups", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.triangulate:
        A, B, C = a.triangulate
        ab, qa, qb, _ = pair_mse(A, B, a.groups, a.seed)
        ac, _, qc, _ = pair_mse(A, C, a.groups, a.seed)
        bc, _, _, _ = pair_mse(B, C, a.groups, a.seed)
        eA, eB, eC = (ab + ac - bc) / 2, (ab + bc - ac) / 2, (ac + bc - ab) / 2
        print("triangulation (errors assumed mutually uncorrelated)\n")
        print(f"  MSE(A,B)={ab:.5f}  MSE(A,C)={ac:.5f}  MSE(B,C)={bc:.5f}\n")
        print(f"{'artifact':<30} {'format':<14} {'rel MSE':>10} {'SNR dB':>8}")
        print("-" * 66)
        for p, q, e in ((A, qa, eA), (B, qb, eB), (C, qc, eC)):
            print(f"{p.split('/')[-1]:<30} {NAMES.get(q, q):<14} {e:>10.5f} {db(e):>8.2f}")
        return

    if not a.ref or not a.candidates:
        ap.error("need --ref R with candidates, or --triangulate A B C")
    rows = []
    for c in a.candidates:
        rel, qc, _, nt = pair_mse(c, a.ref, a.groups, a.seed)
        rows.append((c.split("/")[-1], NAMES.get(qc, str(qc)), nt, rel, db(rel)))
    print(f"reference: {a.ref.split('/')[-1]}\n")
    print(f"{'artifact':<30} {'format':<14} {'tensors':>7} {'rel MSE':>10} {'SNR dB':>8}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: -x[3]):
        print(f"{r[0]:<30} {r[1]:<14} {r[2]:>7} {r[3]:>10.5f} {r[4]:>8.2f}")


if __name__ == "__main__":
    main()
