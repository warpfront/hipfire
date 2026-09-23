#!/usr/bin/env python3
"""MQ4E8 gate-1 oracle: independent Python re-derivation of the constrained
qt44 encoder (slice A). Reads the planner's oracle input (48x5120 f32,
AWQ-applied, pre-FWHT), re-derives C0/C1/C2/C3/C4/C4z/C4s with numpy
(float32 DAGs mirroring the Rust encoder op-for-op), and compares md5
against the planner's expected .qt44 files.

f32_to_f16 here is Hipfire's HISTORICAL TRUNCATION (float16.rs:22), NOT
round-to-nearest.
"""
import hashlib
import sys

import numpy as np

ROOT = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study"
IN = f"{ROOT}/oracle_L0_in_proj_a_awq_f32.bin"
M, K = 48, 5120


def gen_signs(seed, n=256):
    state = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


def f32_to_f16_bits(a):
    """Truncating f32->f16 bits (port of float16.rs f32_to_f16)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    sign = (u >> np.uint32(31)) & np.uint32(1)
    exp = ((u >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    frac = u & np.uint32(0x7FFFFF)
    out = np.empty(a.shape, dtype=np.uint16)
    # inf/nan (not expected in oracle data, but port faithfully)
    m_inf = exp == 0xFF
    f16_frac = np.where(
        frac == 0, np.uint32(0), (frac >> np.uint32(13)) | np.uint32(1)
    ).astype(np.uint32)
    out[m_inf] = ((sign[m_inf] << np.uint32(15)) | (np.uint32(0x1F) << np.uint32(10)) | f16_frac[m_inf]).astype(np.uint16)
    new_exp = exp - 127 + 15
    m_over = (~m_inf) & (new_exp >= 31)
    out[m_over] = ((sign[m_over] << np.uint32(15)) | (np.uint32(0x1F) << np.uint32(10))).astype(np.uint16)
    m_sub = (~m_inf) & (~m_over) & (new_exp <= 0)
    tiny = new_exp < -10
    f = frac | np.uint32(0x800000)
    shift = (1 - new_exp + 13).astype(np.uint32)
    sub_val = np.where(tiny, np.uint32(0), f >> shift)
    out[m_sub] = ((sign[m_sub] << np.uint32(15)) | sub_val[m_sub]).astype(np.uint16)
    m_norm = (~m_inf) & (~m_over) & (~m_sub)
    out[m_norm] = (
        (sign[m_norm] << np.uint32(15))
        | (new_exp[m_norm].astype(np.uint32) << np.uint32(10))
        | (frac[m_norm] >> np.uint32(13))
    ).astype(np.uint16)
    return out


def f16_to_f32_arr(bits):
    """Exact f16 bits -> f32 (bit-exact view cast)."""
    return np.ascontiguousarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)

def fwht256(x, s1, s2):
    """In-place FWHT on (...,256) float32, same op order as cpu_fwht_256."""
    lead = x.shape[:-1]
    x = x.reshape(-1, 256)
    x = x * s1
    stride = 1
    while stride < 256:
        for i in range(0, 256, 2 * stride):
            a = x[:, i : i + stride].copy()
            b = x[:, i + stride : i + 2 * stride].copy()
            x[:, i : i + stride] = a + b
            x[:, i + stride : i + 2 * stride] = a - b
        stride <<= 1
    x = x * (np.float32(0.0625) * s2)
    return x.reshape(lead + (256,))
def base_steps(g):
    """Unconstrained minmax f16 half steps (values, not bits)."""
    out = []
    for h in (0, 1):
        s = g[h * 128 : (h + 1) * 128]
        lo = float(s.min())
        hi = float(s.max())
        step = np.float32((hi - lo) / np.float32(15.0)) if hi > lo else np.float32(0.0)
        bits = f32_to_f16_bits(np.array([step], dtype=np.float32))[0]
        if hi == lo:
            bits = np.uint16(0)
        out.append((bits, lo, hi))
    return out


def assign_q(v, z, inv):
    q = np.floor((v - z) * inv + np.float32(0.5))
    return np.clip(q, 0, 15).astype(np.uint8)


def mean_resid(vals, s, q):
    acc = 0.0
    sf = float(s)
    for i in range(len(vals)):
        acc += float(vals[i]) - sf * float(int(q[i]))
    return acc / len(vals)


def lloyd_free(g, s):
    """Fixed-scale free-zero Lloyd. Returns (s_bits, z_bits, q)."""
    f32 = np.float32
    s_bits = f32_to_f16_bits(np.array(s, dtype=np.float32))
    z_bits = np.empty(2, dtype=np.uint16)
    q = np.empty(256, dtype=np.uint8)
    active = [sv != f32(0.0) for sv in s]
    z = [f32(0.0), f32(0.0)]
    for h in (0, 1):
        seg = g[h * 128 : (h + 1) * 128]
        if not active[h]:
            q[h * 128 : (h + 1) * 128] = 0
            m = sum(float(v) for v in seg) / 128.0
            z_bits[h] = f32_to_f16_bits(np.array([np.float32(m)], dtype=np.float32))[0]
        else:
            lo = float(seg.min())
            z_bits[h] = f32_to_f16_bits(np.array([f32(lo)], dtype=np.float32))[0]
            z[h] = f16_to_f32_arr(np.array([z_bits[h]], dtype=np.uint16))[0]
            z[h] = f32(z[h])
    if not any(active):
        return s_bits, z_bits, q
    inv = [f32(1.0) / f32(sv) if a else f32(0.0) for sv, a in zip(s, active)]
    for _ in range(8):
        for h in (0, 1):
            if active[h]:
                q[h * 128 : (h + 1) * 128] = assign_q(
                    g[h * 128 : (h + 1) * 128], z[h], inv[h]
                )
        nz = z_bits.copy()
        for h in (0, 1):
            if active[h]:
                m = mean_resid(g[h * 128 : (h + 1) * 128], f32(s[h]), q[h * 128 : (h + 1) * 128])
                nz[h] = f32_to_f16_bits(np.array([f32(m)], dtype=np.float32))[0]
        if bool((nz == z_bits).all()):
            break
        z_bits = nz
        for h in (0, 1):
            if active[h]:
                z[h] = f32(f16_to_f32_arr(np.array([z_bits[h]], dtype=np.uint16))[0])
    for h in (0, 1):
        if active[h]:
            q[h * 128 : (h + 1) * 128] = assign_q(
                g[h * 128 : (h + 1) * 128], z[h], inv[h]
            )
    return s_bits, z_bits, q


def main():
    f32 = np.float32
    w = np.fromfile(IN, dtype=np.float32).reshape(M, K)
    assert w.shape == (M, K)
    s1 = gen_signs(42)
    s2 = gen_signs(1042)
    # rotate whole tensor once (row path needs full rows; group path reuses)
    g = fwht256(w.copy().reshape(-1, 256), s1, s2).reshape(M, K)
    n_blocks = M * K // 256
    groups = g.reshape(n_blocks, 256)

    # ---- unconstrained base steps for every group ----
    base = [base_steps(groups[b]) for b in range(n_blocks)]
    sh = np.array(
        [[float(f16_to_f32_arr(np.array([bb], dtype=np.uint16))[0]) for bb, _, _ in grp] for grp in base],
        dtype=np.float32,
    )
    lo = np.array([[l for _, l, _ in grp] for grp in base], dtype=np.float32)

    def emit(scales_bits, zeros_bits, codes):
        out = bytearray(n_blocks * 136)
        for b in range(n_blocks):
            o = b * 136
            out[o : o + 2] = int(scales_bits[b, 0]).to_bytes(2, "little")
            out[o + 2 : o + 4] = int(zeros_bits[b, 0]).to_bytes(2, "little")
            out[o + 4 : o + 6] = int(scales_bits[b, 1]).to_bytes(2, "little")
            out[o + 6 : o + 8] = int(zeros_bits[b, 1]).to_bytes(2, "little")
            qq = codes[b]
            for i in range(128):
                out[o + 8 + i] = (int(qq[2 * i]) & 0xF) | ((int(qq[2 * i + 1]) & 0xF) << 4)
        return bytes(out)

    results = {}

    # ---- C0 ----
    sb = np.empty((n_blocks, 2), dtype=np.uint16)
    zb = np.empty((n_blocks, 2), dtype=np.uint16)
    qc = np.empty((n_blocks, 256), dtype=np.uint8)
    for b in range(n_blocks):
        s = [float(sh[b, 0]), float(sh[b, 1])]
        xs, xz, xq = lloyd_free(groups[b].copy(), [f32(s[0]), f32(s[1])])
        sb[b], zb[b], qc[b] = xs, xz, xq
    results["C0"] = emit(sb, zb, qc)

    # ---- C1: nearest exactly-f16 G*2^-d ----
    G = np.maximum(sh[:, 0], sh[:, 1])
    sb = np.empty((n_blocks, 2), dtype=np.uint16)
    zb = np.empty((n_blocks, 2), dtype=np.uint16)
    qc = np.empty((n_blocks, 256), dtype=np.uint8)
    for b in range(n_blocks):
        s = []
        for h in (0, 1):
            best = None  # ((dist32, -cand), cand)
            gb = f32(float(G[b]))
            shb = f32(float(sh[b, h]))
            for d in range(5):
                cand = f32(gb * f32(float(f32(2.0) ** -d))) if d else gb
                # exactness: round-trip + ratio (both f32, mirroring Rust)
                if f16_to_f32_arr(f32_to_f16_bits(np.array([cand])))[0] != float(cand):
                    continue
                if f32(cand * f32(float(f32(2.0) ** d))) != gb:
                    continue
                dist = f32(abs(f32(cand - shb)))
                key = (float(dist), -float(cand))
                if best is None or key < best[0]:
                    best = (key, cand)
            assert best is not None
            s.append(best[1])
        xs, xz, xq = lloyd_free(groups[b].copy(), [f32(s[0]), f32(s[1])])
        # C1 scales are the chosen candidates exactly
        xs = f32_to_f16_bits(np.array([f32(s[0]), f32(s[1])], dtype=np.float32))
        sb[b], zb[b], qc[b] = xs, xz, xq
    results["C1"] = emit(sb, zb, qc)

    # ---- C2 ----
    sb = np.empty((n_blocks, 2), dtype=np.uint16)
    zb = np.empty((n_blocks, 2), dtype=np.uint16)
    qc = np.empty((n_blocks, 256), dtype=np.uint8)
    for b in range(n_blocks):
        gv = float(max(float(sh[b, 0]), float(sh[b, 1])))
        if gv == 0.0:
            s = [f32(0.0), f32(0.0)]
        else:
            bb = int(f32_to_f16_bits(np.array([f32(gv / 16.0)]))[0]) & 0xFFF0
            if bb == 0:
                bb = 0x0010
            bf = float(f16_to_f32_arr(np.array([np.uint16(bb)]))[0])
            s = []
            for h in (0, 1):
                t = float(np.floor(f32(f32(float(sh[b, h])) / f32(bf)) + f32(0.5)))
                m = min(16, max(1, int(t)))
                s.append(f32(f32(bf) * f32(float(m))))
        xs, xz, xq = lloyd_free(groups[b].copy(), s)
        xs = f32_to_f16_bits(np.array(s, dtype=np.float32))
        sb[b], zb[b], qc[b] = xs, xz, xq
    results["C2"] = emit(sb, zb, qc)

    # ---- C3 ----
    sb = np.empty((n_blocks, 2), dtype=np.uint16)
    zb = np.empty((n_blocks, 2), dtype=np.uint16)
    qc = np.empty((n_blocks, 256), dtype=np.uint8)
    for b in range(n_blocks):
        gv = max(float(sh[b, 0]), float(sh[b, 1]))
        xs, xz, xq = lloyd_free(groups[b].copy(), [f32(gv), f32(gv)])
        xs = f32_to_f16_bits(np.array([f32(gv), f32(gv)], dtype=np.float32))
        sb[b], zb[b], qc[b] = xs, xz, xq
    results["C3"] = emit(sb, zb, qc)

    # ---- row-master constraints C4/C4z/C4s ----
    gpr = K // 256
    for tag, mode in (("C4", "free"), ("C4z", "int"), ("C4s", "k8")):
        sb = np.empty((n_blocks, 2), dtype=np.uint16)
        zb = np.empty((n_blocks, 2), dtype=np.uint16)
        qc = np.empty((n_blocks, 256), dtype=np.uint8)
        for r in range(M):
            rmax = float(sh[r * gpr : (r + 1) * gpr].max())
            if rmax == 0.0:
                B = 0.0
            else:
                bb = int(f32_to_f16_bits(np.array([f32(rmax / 256.0)]))[0])
                if mode == "int":
                    bb &= 0xFFF0
                    if bb == 0:
                        bb = 0x0010
                elif bb == 0:
                    bb = 0x0001
                B = float(f16_to_f32_arr(np.array([np.uint16(bb)]))[0])
            for c in range(gpr):
                b = r * gpr + c
                if B == 0.0:
                    xs = np.array([np.uint16(0), np.uint16(0)])
                    if mode == "free":
                        _, xz, xq = lloyd_free(groups[b].copy(), [f32(0.0), f32(0.0)])
                    else:
                        xz = np.array([np.uint16(0), np.uint16(0)])
                        xq = np.zeros(256, dtype=np.uint8)
                    sb[b], zb[b], qc[b] = xs, xz, xq
                    continue
                s = []
                for h in (0, 1):
                    best = None
                    shb = f32(float(sh[b, h]))
                    for e in range(9):
                        cand = f32(f32(B) * f32(float(f32(2.0) ** e)))
                        dist = f32(abs(f32(cand - shb)))
                        key = (float(dist), -float(cand))
                        if best is None or key < best[0]:
                            best = (key, cand, e)
                    s.append(best[1])
                    assert f16_to_f32_arr(
                        f32_to_f16_bits(np.array([best[1]]))
                    )[0] == float(best[1])
                    assert float(f32(best[1] * f32(float(f32(2.0) ** -best[2])))) == float(f32(B))
                if mode == "free":
                    xs, xz, xq = lloyd_free(groups[b].copy(), [f32(s[0]), f32(s[1])])
                    xs = f32_to_f16_bits(np.array([f32(s[0]), f32(s[1])], dtype=np.float32))
                elif mode == "k8":
                    xs = f32_to_f16_bits(np.array([f32(s[0]), f32(s[1])], dtype=np.float32))
                    xz = np.empty(2, dtype=np.uint16)
                    xq = np.empty(256, dtype=np.uint8)
                    for h in (0, 1):
                        z = f32(-8.0 * float(s[h]))
                        xz[h] = f32_to_f16_bits(np.array([z]))[0]
                        xq[h * 128 : (h + 1) * 128] = assign_q(
                            groups[b][h * 128 : (h + 1) * 128], z, f32(1.0) / f32(s[h])
                        )
                else:
                    xs = f32_to_f16_bits(np.array([f32(s[0]), f32(s[1])], dtype=np.float32))
                    xz = np.empty(2, dtype=np.uint16)
                    xq = np.empty(256, dtype=np.uint8)
                    kk = []
                    for h in (0, 1):
                        z0 = f32(float(f16_to_f32_arr(
                            f32_to_f16_bits(np.array([f32(float(lo[b, h]))])))[0]))
                        t = float(np.floor(z0 / f32(s[h]) + f32(0.5)))
                        kk.append(min(0, max(-15, int(t))))
                    z = [f32(float(s[h]) * float(kk[h])) for h in (0, 1)]
                    inv = [f32(1.0) / f32(s[h]) for h in (0, 1)]
                    for _ in range(8):
                        for h in (0, 1):
                            xq[h * 128 : (h + 1) * 128] = assign_q(
                                groups[b][h * 128 : (h + 1) * 128], z[h], inv[h]
                            )
                        nk = []
                        for h in (0, 1):
                            m = mean_resid(
                                groups[b][h * 128 : (h + 1) * 128], f32(s[h]),
                                xq[h * 128 : (h + 1) * 128],
                            )
                            t = int(np.floor(m / float(s[h]) + 0.5))
                            nk.append(min(0, max(-15, t)))
                        if nk == kk:
                            break
                        kk = nk
                        z = [f32(float(s[h]) * float(kk[h])) for h in (0, 1)]
                    for h in (0, 1):
                        z = f32(float(s[h]) * float(kk[h]))
                        xz[h] = f32_to_f16_bits(np.array([z]))[0]
                        xq[h * 128 : (h + 1) * 128] = assign_q(
                            groups[b][h * 128 : (h + 1) * 128], z, inv[h]
                        )
                sb[b], zb[b], qc[b] = xs, xz, xq
        results[tag] = emit(sb, zb, qc)

    ok = True
    for tag in ("C0", "C1", "C2", "C3", "C4", "C4z", "C4s"):
        mine = hashlib.md5(results[tag]).hexdigest()
        exp = open(f"{ROOT}/oracle_L0_in_proj_a_{tag}.qt44", "rb").read()
        exp_md5 = hashlib.md5(exp).hexdigest()
        match = "MATCH" if mine == exp_md5 else "MISMATCH"
        if mine != exp_md5:
            ok = False
            nb = len(exp) // 136
            dg = sum(
                1
                for g in range(nb)
                if results[tag][g * 136 : (g + 1) * 136] != exp[g * 136 : (g + 1) * 136]
            )
            print(f"python oracle {tag}: {match} mine={mine} exp={exp_md5} diffgroups={dg}/{nb}")
        else:
            print(f"python oracle {tag}: {match} md5={mine}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
