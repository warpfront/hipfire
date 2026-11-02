#!/usr/bin/env python3
"""Offline analysis of numerics_probe receipts (raw bits -> bitwise conclusions).

  analyze.py <receipt-dir> identity
  analyze.py <receipt-dir> dot   [--jobs N]
  analyze.py <receipt-dir> wmma  [--jobs N]      (v_wmma_f32/bf16/f16 *16x16x16*, iu8)
  analyze.py <receipt-dir> swmmac
  analyze.py <receipt-dir> div_scale   (suite div_scale: v_div_scale_f32 against the exponent-field chain)
  analyze.py <receipt-dir> scalar_udiv (suite scalar_udiv: s_cvt_f32_u32 / v_s_rcp_f32 / s_mul_f32 / s_cvt_u32_f32)
  analyze.py <receipt-dir> maxmin      (suite maxmin: v_maxmin_num_f32 / v_minmax_num_f32 against the num_select composition)

Each command prints a concise report and writes <receipt-dir>/analysis/<command>.json. The analysis never edits the
receipts. Models are the scalar references in models.py; a mismatch count of 0 over the stated number of words is the
only thing the report calls a match.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M  # noqa: E402


# ------------------------------------------------------------------ receipt access
class Receipt:
    def __init__(self, d):
        self.d = d
        self.meta = {}
        for n in open(os.path.join(d, "INDEX.txt")).read().split():
            self.meta[n] = json.load(open(os.path.join(d, n + ".json")))
        self.ident = self.meta["identity"]
        self.arch = self.ident["compiled_arch"]

    def raw_bytes(self, name):
        """Bytes of <name>.bin, transparently reading <name>.bin.zst (receipts are kept zstd-compressed)."""
        p = os.path.join(self.d, name + ".bin")
        if os.path.exists(p):
            return open(p, "rb").read()
        import subprocess
        return subprocess.run(["zstd", "-dc", p + ".zst"], check=True, capture_output=True).stdout

    def bin(self, name, dtype="<u4"):
        return np.frombuffer(self.raw_bytes(name), dtype=dtype)

    def results(self, kind, **want):
        for n, m in self.meta.items():
            if m.get("kind") == kind and all(m.get(k) == v for k, v in want.items()):
                yield n, m


def write_json(r, name, obj):
    os.makedirs(os.path.join(r.d, "analysis"), exist_ok=True)
    with open(os.path.join(r.d, "analysis", name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def parse_mods(s):
    """'neg_lo:[1,0,0] op_sel_hi:[0,1,1] clamp' -> dict of flag lists (op_sel_hi defaults to all ones)."""
    f = {"neg_lo": [0, 0, 0], "neg_hi": [0, 0, 0], "op_sel": [0, 0, 0], "op_sel_hi": [1, 1, 1], "clamp": 0}
    for tok in re.findall(r"(\w+):\[([01,]+)\]", s):
        f[tok[0]] = [int(x) for x in tok[1].split(",")] + [0] * (3 - len(tok[1].split(",")))
    f["clamp"] = int("clamp" in s.split())
    return f


# ------------------------------------------------------------------ identity
def cmd_identity(r, args):
    out = {"identity": r.ident}
    p = os.path.join(r.d, "identity.host.txt")
    if os.path.exists(p):
        out["host"] = open(p).read()
    print(json.dumps(r.ident, indent=1))
    if "host" in out:
        print(out["host"])
    write_json(r, "identity", out)


# ------------------------------------------------------------------ dot
def _half(x, i):
    return (x >> (16 * i)) & 0xFFFF


def dot2_inputs(a, b, c, f):
    """Effective (a0,a1,b0,b1,c) after neg/op_sel/op_sel_hi hypothesis H_dot2."""
    def sel(x, s, idx):
        return _half(x, f["op_sel_hi"][idx] if s else f["op_sel"][idx])
    a0 = _half(a, f["op_sel"][0]); a1 = _half(a, f["op_sel_hi"][0])
    b0 = _half(b, f["op_sel"][1]); b1 = _half(b, f["op_sel_hi"][1])
    if f["neg_lo"][0]: a0 ^= 0x8000
    if f["neg_hi"][0]: a1 ^= 0x8000
    if f["neg_lo"][1]: b0 ^= 0x8000
    if f["neg_hi"][1]: b1 ^= 0x8000
    cc = c & 0x7FFFFFFF if f["neg_hi"][2] else c  # hypothesis: neg_hi[2] = |C| (src2 is f32: the 'hi' bit is ABS)
    cc ^= 0x80000000 if f["neg_lo"][2] else 0       # neg_lo[2] = -C (applied after ABS)
    return a0, a1, b0, b1, cc


def f32_exact(bits):
    return M.decode_exact(bits, "f32")


def dot2_exact(arch, kind, a0, a1, b0, b1, c, flush_in_f=False, flush_in_c=False):
    """Candidate E: exact real sum, one RNE rounding to f32 (gradual underflow). NaN -> 'nan'."""
    ops = [(M.decode_exact(a0, kind), M.decode_exact(b0, kind)), (M.decode_exact(a1, kind), M.decode_exact(b1, kind))]
    cc = M.decode_exact(c, "f32")
    if cc[0] == "nan" or any(x[0] == "nan" or y[0] == "nan" for x, y in ops):
        return "nan"
    infs = []
    for x, y in ops:
        if x[0] == "inf" or y[0] == "inf":
            other = y if x[0] == "inf" else x
            if other[0] == "fin" and other[2] == 0:
                return "nan"
            infs.append(x[1] ^ y[1])
    if cc[0] == "inf":
        infs.append(cc[1])
    if infs:
        return "nan" if len(set(infs)) > 1 else (0xFF800000 if infs[0] else 0x7F800000)
    s = ops[0][0][2] * ops[0][1][2] + ops[1][0][2] * ops[1][1][2] + cc[2]
    if s == 0:
        # sign of an exact zero: -0 only if all three terms are -0 (RNE rule for sums of zeros)
        negs = [(x[1] ^ y[1]) for x, y in ops] + [cc[1]]
        return 0x80000000 if all(negs) else 0
    return M.encode_rne(s, "f32")


def dot2_step(arch, kind, a0, a1, b0, b1, c):
    """Candidate S: mma.rs dot_step with the two products and C in ONE step (gfx11: 2-term; gfx12: 2-term pair grid)."""
    return M.dot_step([a0, a1], [b0, b1], c, kind == "bf16", arch == "gfx1151")


def _chunk_dot2(args):
    arch, kind, flags, rows = args
    out = []
    for a, b, c, hw in rows:
        a0, a1, b0, b1, cc = dot2_inputs(a, b, c, flags)
        e = dot2_exact(arch, kind, a0, a1, b0, b1, cc)
        s = dot2_step(arch, kind, a0, a1, b0, b1, cc)
        out.append((e, s))
    return out


def classify_f32(bits):
    e = (bits >> 23) & 255
    f = bits & 0x7FFFFF
    if e == 255:
        return "nan" if f else "inf"
    if e == 0:
        return "zero" if f == 0 else "sub"
    return "norm"


def analyze_dot_float(r, op, jobs):
    kind = "f16" if op == "v_dot2_f32_f16" else "bf16"
    rep = {}
    ins = {s: r.bin(f"dot.{op}.{s}.in").reshape(-1, 3) for s in ("full", "red")}
    cache = {}
    for name, m in r.results("dot_result", op=op):
        s = m["set"]
        hw = r.bin(name)
        flags = parse_mods(m["mods"])
        rows = [(int(a), int(b), int(c), int(h)) for (a, b, c), h in zip(ins[s], hw)]
        n = len(rows)
        ck = (s, m["mods"])
        if ck not in cache:
            chunks = [(r.arch, kind, flags, rows[i:i + 20000]) for i in range(0, n, 20000)]
            with mp.get_context("fork").Pool(jobs) as pool:
                cache[ck] = [x for ch in pool.map(_chunk_dot2, chunks) for x in ch]
        res = cache[ck]
        stat = {"cases": n, "mods": m["mods"], "mode_readback": m["mode_readback"], "mode_request": m["mode_request"]}
        for label, idx in (("exact_rne", 0), ("mma_dot_step", 1)):
            bad = Counter()
            ex = []
            for (a, b, c, h), pair in zip(rows, res):
                mdl = pair[idx]
                if mdl == "nan":
                    ok = classify_f32(h) == "nan"
                else:
                    ok = h == mdl
                if flags["clamp"] and not ok:
                    ok = False
                if not ok:
                    key = (classify_f32(h), "nan" if mdl == "nan" else classify_f32(mdl))
                    bad[key] += 1
                    if len(ex) < 6:
                        ex.append(dict(a=hex(a), b=hex(b), c=hex(c), hw=hex(h), model=mdl if mdl == "nan" else hex(mdl)))
            stat[label] = {"mismatches": sum(bad.values()), "by_class(hw,model)": {f"{k[0]}/{k[1]}": v for k, v in bad.items()}, "examples": ex}
        rep[name] = stat
        print(name, {k: (v["mismatches"] if isinstance(v, dict) and "mismatches" in v else v) for k, v in stat.items() if k in ("exact_rne", "mma_dot_step", "cases")})
    return rep


def analyze_dot_int(r, op):
    nbits, nel = (8, 4) if op == "v_dot4_i32_iu8" else (4, 8)
    rep = {}
    ins = {s: r.bin(f"dot.{op}.{s}.in").reshape(-1, 3).astype(np.uint64) for s in ("full", "red")}
    for name, m in r.results("dot_result", op=op):
        hw = r.bin(name).astype(np.uint32)
        a, b, c = ins[m["set"]][:, 0], ins[m["set"]][:, 1], ins[m["set"]][:, 2]
        f = parse_mods(m["mods"])
        best = {}
        for a_s in (0, 1):
            for b_s in (0, 1):
                for clamp in (0, 1):
                    s = np.zeros(len(a), dtype=np.int64)
                    mask = (1 << nbits) - 1
                    for i in range(nel):
                        x = ((a >> np.uint64(nbits * i)) & np.uint64(mask)).astype(np.int64)
                        y = ((b >> np.uint64(nbits * i)) & np.uint64(mask)).astype(np.int64)
                        if a_s: x = np.where(x >= (1 << (nbits - 1)), x - (1 << nbits), x)
                        if b_s: y = np.where(y >= (1 << (nbits - 1)), y - (1 << nbits), y)
                        s += x * y
                    cs = c.astype(np.int64)
                    cs = np.where(cs >= (1 << 31), cs - (1 << 32), cs)
                    t = cs + s
                    t = np.clip(t, -(1 << 31), (1 << 31) - 1) if clamp else ((t + (1 << 31)) % (1 << 32)) - (1 << 31)
                    mm = int(np.count_nonzero((t.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32) != hw))
                    best[(a_s, b_s, clamp)] = mm
        winner = min(best, key=best.get)
        rep[name] = {"cases": len(hw), "mods": m["mods"], "mismatches_by(a_signed,b_signed,clamp)": {str(k): v for k, v in best.items()}, "best": winner, "best_mismatches": best[winner]}
        if winner[2]:
            # order discriminator: saturating after C and after EACH product (in element order) instead of once at the end
            a_s, b_s, _ = winner
            mask = (1 << nbits) - 1
            t = np.where(c.astype(np.int64) >= (1 << 31), c.astype(np.int64) - (1 << 32), c.astype(np.int64))
            for i in range(nel):
                x = ((a >> np.uint64(nbits * i)) & np.uint64(mask)).astype(np.int64)
                y = ((b >> np.uint64(nbits * i)) & np.uint64(mask)).astype(np.int64)
                if a_s: x = np.where(x >= (1 << (nbits - 1)), x - (1 << nbits), x)
                if b_s: y = np.where(y >= (1 << (nbits - 1)), y - (1 << nbits), y)
                t = np.clip(t + x * y, -(1 << 31), (1 << 31) - 1)
            alt = (t & 0xFFFFFFFF).astype(np.uint32)
            rep[name]["per_term_saturation_alt_mismatches"] = int(np.count_nonzero(alt != hw))
        print(name, m["mods"] or "-", "best (a_signed,b_signed,clamp)=", winner, "mismatches", best[winner], "of", len(hw), rep[name].get("per_term_saturation_alt_mismatches", ""))
    return rep


def cmd_dot(r, args):
    rep = {}
    for op in ("v_dot2_f32_f16", "v_dot2_f32_bf16"):
        rep[op] = analyze_dot_float(r, op, args.jobs)
    for op in ("v_dot4_i32_iu8", "v_dot8_i32_iu4"):
        rep[op] = analyze_dot_int(r, op)
    write_json(r, "dot", rep)


# ------------------------------------------------------------------ WMMA layout decode
KINDS = {0: "v_wmma_f32_16x16x16_bf16", 1: "v_wmma_bf16_16x16x16_bf16", 2: "v_wmma_f16_16x16x16_f16", 3: "v_wmma_i32_16x16x16_iu8", 4: "v_wmma_f32_16x16x16_f16"}


def decode_D(raw, arch, out16, variant_hi=False):
    """raw [n][vgpr][lane] u32 -> logical D[n][16][16] (u32; 16-bit values in the low half) plus the 'other half' array."""
    n, nv, _ = raw.shape
    D = np.zeros((n, 16, 16), dtype=np.uint32)
    other = np.zeros((n, 16, 16), dtype=np.uint32)
    lanes = np.arange(32)
    col = lanes & 15
    h = lanes >> 4
    if not out16:
        for j in range(8):
            rows = (8 * h + j) if arch == "gfx1201" else (2 * j + h)
            D[:, rows, col] = raw[:, j, :]
    elif arch == "gfx1201":
        for j in range(4):
            D[:, 8 * h + 2 * j, col] = raw[:, j, :] & 0xFFFF
            D[:, 8 * h + 2 * j + 1, col] = raw[:, j, :] >> 16
    else:
        for j in range(8):
            rows = 2 * j + h
            D[:, rows, col] = (raw[:, j, :] >> (16 if variant_hi else 0)) & 0xFFFF
            other[:, rows, col] = (raw[:, j, :] >> (0 if variant_hi else 16)) & 0xFFFF
    return D, other


# ---- float WMMA model evaluation (per wave: 256 outputs)
def widen16(bits, kind):
    """16-bit accumulator element -> exact f32 bits (bf16: <<16; f16: exact convert, NaN payload shifted)."""
    if kind == "bf16":
        return (bits & 0xFFFF) << 16
    cls, sign, v = M.decode_exact(bits & 0xFFFF, "f16")
    if cls == "nan":
        return (sign << 31) | 0x7F800000 | 0x400000 | ((bits & 0x3FF) << 13)
    if cls == "inf":
        return (sign << 31) | 0x7F800000
    return M.encode_rne(v, "f32", sign_of_zero=sign)


def transform_wave(A, Bt, C, spec, kind, out16):
    """Apply per-flag input transforms: spec = {(src, 'lo'|'hi'): 'neg'|'abs'} to a wave (copies)."""
    A, Bt, C = list(A), list(Bt), list(C)
    # ABS is applied before NEG (v_dot2/WMMA src2: neg_hi=|C| then neg_lo=-C gives -|C|)
    for (src, half), op in sorted(spec.items(), key=lambda kv: 0 if kv[1] == "abs" else 1):
        if op == "none":
            continue
        if src in (0, 1):
            arr = A if src == 0 else Bt
            for i in range(256):
                k = i & 15
                if (k & 1) == (0 if half == "lo" else 1):
                    arr[i] = arr[i] ^ 0x8000 if op == "neg" else arr[i] & 0x7FFF
        else:
            for i in range(256):
                if out16:  # 16-bit C: lo flag -> even rows, hi flag -> odd rows (gfx12 packs two rows per VGPR)
                    if ((i >> 4) & 1) == (0 if half == "lo" else 1):
                        C[i] = (C[i] ^ 0x8000) if op == "neg" else (C[i] & 0x7FFF)
                else:  # f32 C: the flag acts on the whole accumulator word
                    C[i] = (C[i] ^ 0x80000000) if op == "neg" else (C[i] & 0x7FFFFFFF)
    return A, Bt, C


def eval_wave(arch, kind, A, Bt, C, out16, conv):
    """-> list of 256 model results (row*16+col). kind: 'bf16'|'f16'. conv: None (f32 out) or 'rne'/'rtz'."""
    bf = kind == "bf16"
    res = []
    for r in range(16):
        ar = A[r * 16:r * 16 + 16]
        for c in range(16):
            bc = Bt[c * 16:c * 16 + 16]
            acc = widen16(C[r * 16 + c], kind) if out16 else C[r * 16 + c]
            v = M.wmma_f32(arch, ar, bc, acc, bf)
            if out16:
                if conv == "rne":
                    v = M.f32_bits_to_f16_rne(v, kind)
                else:  # rtz: truncate the f32 significand
                    eb, fb, _ = M.fmt_params(kind)
                    cls, sign, val = M.decode_exact(v, "f32")
                    if cls == "fin":
                        v = _trunc_to(val, kind, sign)
                    else:
                        v = M.f32_bits_to_f16_rne(v, kind)
            res.append(v)
    return res


def _trunc_to(val, kind, sign):
    from fractions import Fraction
    eb, fb, bias = M.fmt_params(kind)
    if val == 0:
        return sign << (eb + fb)
    a = -val if val < 0 else val
    emin = 1 - bias
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a: e -= 1
    if Fraction(2) ** (e + 1) <= a: e += 1
    if e < emin:
        q = int(a / (Fraction(2) ** (emin - fb)))
        return (sign << (eb + fb)) | q
    q = int(a / (Fraction(2) ** (e - fb)))
    if e + bias >= (1 << eb) - 1:
        return (sign << (eb + fb)) | (((1 << eb) - 1) << fb) - 1 if False else (sign << (eb + fb)) | ((((1 << eb) - 2) << fb) | ((1 << fb) - 1))
    return (sign << (eb + fb)) | ((e + bias) << fb) | (q & ((1 << fb) - 1))


def _wave_job(args):
    arch, kind, out16, convs, spec, waves = args
    out = {c: [] for c in convs}
    for A, Bt, C in waves:
        if spec:
            A, Bt, C = transform_wave(A, Bt, C, spec, kind, out16)
        for c in convs:
            out[c].append(eval_wave(arch, kind, A, Bt, C, out16, c))
    return out


def run_waves(arch, kind, out16, convs, spec, cs, idx, jobs):
    items = [(cs["A"][i].tolist(), cs["B"][i].tolist(), cs["C"][i].tolist()) for i in idx]
    chunk = max(1, len(items) // (jobs * 4))
    parts = [(arch, kind, out16, convs, spec, items[i:i + chunk]) for i in range(0, len(items), chunk)]
    with mp.get_context("fork").Pool(jobs) as pool:
        rs = pool.map(_wave_job, parts)
    return {c: [w for r_ in rs for w in r_[c]] for c in convs}


def segments(meta_in):
    """Named wave ranges of a float set: controls, F1, F4, F5, F3a(sub end), F2, F3b."""
    return meta_in



def seg_bounds(kind, n_sub, n):
    bf = kind in (0, 1)
    f5 = 3 * 2 * (16 if bf else 2) * 8
    out, pos = {}, 0
    for name, cnt in (("controls", 3), ("F1_single", 16 * 4 * 32), ("F4_subnormal_rows", 32 * 6 * 4), ("F5_c_tail_sweep", f5)):
        out[name] = (pos, pos + cnt); pos += cnt
    out["F3a_random"] = (pos, n_sub); pos = n_sub
    out["F2_two_product"] = (pos, pos + 15360); pos += 15360
    out["F3b_random"] = (pos, n)
    return out


def _cmp_words(hwD, mdl, n_idx):
    """-> mismatch count and examples; NaN results compared bit-exact (the production model emits canonical NaNs)."""
    mism = 0
    ex = []
    for wi, w in enumerate(mdl):
        for z in range(256):
            h = int(hwD[wi][z // 16][z % 16])
            if h != w[z]:
                mism += 1
                if len(ex) < 8:
                    ex.append((n_idx[wi], z // 16, z % 16, hex(h), hex(w[z])))
    return mism, ex


def model_compare(r, fk, kind, out16, cs, D, nn, n_sub, jobs):
    convs = ["rne", "rtz"] if out16 else [None]
    idx = list(range(nn))
    mod = run_waves(r.arch, fk, out16, convs, None, cs, idx, jobs)
    segs = seg_bounds(kind, n_sub, nn)
    res = {"model": "mma.rs dot_step chain (threshold 23)" + (" + f32->16b conversion" if out16 else "")}
    for c in convs:
        per = {}
        for sname, (a, b) in segs.items():
            mm, ex = _cmp_words(D[a:b], mod[c][a:b], list(range(a, b)))
            per[sname] = {"words": (b - a) * 256, "mismatches": mm, "examples": [dict(wave=e[0], row=e[1], col=e[2], hw=e[3], model=e[4]) for e in ex[:4]]}
        res[f"conv={c}" if c else "f32_out"] = per
    return res


def variant_compare(r, fk, kind, out16, cs, D, n_sub, m, spec_by_flag, jobs):
    flags = parse_mods(m["mods"])
    f1 = list(range(*seg_bounds(kind, n_sub, 10 ** 9)["F1_single"]))[:2048]
    todo = [(s, "lo") for s in range(3) if flags["neg_lo"][s]] + [(s, "hi") for s in range(3) if flags["neg_hi"][s]]
    best = {}
    if len(todo) == 1:
        for op in ("none", "neg", "abs"):
            mod = run_waves(r.arch, fk, out16, ["rne"] if out16 else [None], {todo[0]: op}, cs, f1, jobs)
            mm, _ = _cmp_words(D[f1[0]:f1[-1] + 1], mod["rne" if out16 else None], f1)
            best[op] = mm
        win = min(best, key=best.get)
        spec_by_flag[todo[0]] = win
        return {"flag": f"{todo[0]}", "mismatches_by_transform(F1 words)": best, "transform": win}
    spec = {k: spec_by_flag.get(k, "none") for k in todo}
    mod = run_waves(r.arch, fk, out16, ["rne"] if out16 else [None], spec, cs, f1, jobs)
    mm, _ = _cmp_words(D[f1[0]:f1[-1] + 1], mod["rne" if out16 else None], f1)
    return {"composed_transforms": {str(k): v for k, v in spec.items()}, "mismatches(F1 words)": mm}


def iu8_compare(r, cs, D, nn, m):
    f = parse_mods(m["mods"])
    best = {}
    A = cs["A"][:nn].astype(np.int64).reshape(nn, 16, 16) & 255
    Bt = cs["B"][:nn].astype(np.int64).reshape(nn, 16, 16) & 255
    C = cs["C"][:nn].astype(np.int64).reshape(nn, 16, 16)
    C = np.where(C >= (1 << 31), C - (1 << 32), C)
    hw = D.astype(np.uint32)
    for a_s in (0, 1):
        for b_s in (0, 1):
            for clamp in (0, 1):
                Aa = np.where(A >= 128, A - 256, A) if a_s else A
                Bb = np.where(Bt >= 128, Bt - 256, Bt) if b_s else Bt
                t = np.einsum("nrk,nck->nrc", Aa, Bb) + C
                t = np.clip(t, -(1 << 31), (1 << 31) - 1) if clamp else ((t + (1 << 31)) % (1 << 32)) - (1 << 31)
                best[(a_s, b_s, clamp)] = int(np.count_nonzero((t & 0xFFFFFFFF).astype(np.uint32) != hw))
    win = min(best, key=best.get)
    return {"words": nn * 256, "best(a_signed,b_signed,clamp)": win, "mismatches_best": best[win], "mismatches_by": {str(k): v for k, v in best.items()}}


def cmd_wmma(r, args):
    rep = {}
    rec = np.dtype([("A", "<u2", 256), ("B", "<u2", 256), ("C", "<u4", 256)])
    spec_by_flag = {}
    for kind in (0, 4, 1, 2, 3):
        op = KINDS[kind]
        inn = f"wmma.{op}.in"
        if inn not in r.meta:
            continue
        cs = np.frombuffer(r.raw_bytes(inn), dtype=rec)
        out16 = kind in (1, 2)
        fk = "bf16" if kind in (0, 1) else "f16"
        n_sub = r.meta[inn]["n_sub"]
        base = {}
        for name, m in sorted(r.results("wmma_result", op=op), key=lambda x: (x[1]["variant"], x[1]["mode_index"])):
            nn = m["cases"]
            res = r.bin(name).reshape(nn, m["vgprs"], 32)
            hi = "op_sel" in m["mods"]
            D, other = decode_D(res, r.arch, out16, hi)
            stat = {"cases": nn, "mods": m["mods"], "mode_readback": m["mode_readback"], "variant": m["variant"], "mode_index": m["mode_index"]}
            tag = (m["variant"], m["mode_index"])
            if tag == (0, 0):
                base = {"D": D, "other": other}
                wid = (lambda x: x.astype(np.uint32)) if (out16 or kind == 3) else np.vectorize(lambda v: widen16(int(v), fk), otypes=[np.uint32])
                ctl = {"D=B(A=I)": bool(np.array_equal(D[0], wid(cs["B"][0].reshape(16, 16).T))),
                       "D=C": bool(np.array_equal(D[1], cs["C"][1].reshape(16, 16) & (0xFFFF if out16 else 0xFFFFFFFF))),
                       "D=A(B=I)": bool(np.array_equal(D[2], wid(cs["A"][2].reshape(16, 16))))}
                stat["controls"] = ctl
                if out16 and r.arch == "gfx1151":
                    stat["other_half_value_in_case1"] = sorted(set(int(x) for x in other[1].ravel()))[:4]
            elif m["variant"] == 0:  # MODE sweep: bit-identical to default?
                stat["identical_to_default_mode"] = bool(np.array_equal(D, base["D"][:nn]))
                stat["differing_words"] = int(np.count_nonzero(D != base["D"][:nn]))
            if kind != 3 and tag == (0, 0):
                stat.update(model_compare(r, fk, kind, out16, cs, D, nn, n_sub, args.jobs))
            if kind != 3 and m["variant"] > 0 and m["mode_index"] == 0:
                stat.update(variant_compare(r, fk, kind, out16, cs, D, n_sub, m, spec_by_flag, args.jobs))
            if kind == 3:
                stat.update(iu8_compare(r, cs, D, nn, m))
            rep[name] = stat
            print(name, {k: v for k, v in stat.items() if k not in ("examples", "mods", "mode_readback", "variant", "mode_index", "cases")})
        spec_by_flag.clear()
    write_json(r, "wmma", rep)
    return rep


def cmd_swmmac(r, args):
    raise SystemExit("swmmac analysis: see analyze_swmmac.py (written after the discovery data was read)")

# ------------------------------------------------------------------ v_div_scale_f32 / scalar udiv chain
INF_BITS, SIGN_BITS, QUIET_BITS = 0x7F800000, 0x80000000, 0x00400000


def _ldexp64(x):
    """x * 2^64 on raw f32 bits (NaN/inf untouched; overflow to inf; a denormal source normalises exactly)."""
    e = (x >> 23) & 255
    v = x.view(np.float32).astype(np.float64) * 2.0**64
    out = v.astype(np.float32).view(np.uint32)
    return np.where(e == 255, x, out)


def _ldexp_m64(x):
    v = x.view(np.float32).astype(np.float64) * 2.0**-64
    out = v.astype(np.float32).view(np.uint32)
    return np.where(((x >> 23) & 255) == 255, x, out)


def div_scale_chain(s0, s1, s2, open_down_field=False, e2_threshold=23, e1_down=254):
    """Vectorised port of emu/alu.rs::div_scale_chain with every 'open' domain decided by the natural fall-through
    (the hypothesis the hardware receipt then confirms or refutes). Returns (D bits, VCC flag).
    open_down_field: S0 * 2^-64 as raw exponent-field subtraction instead of the exact product (differs only for
    exponent field 64, where the exact product is a denormal)."""
    s0, s1, s2 = (np.asarray(a, dtype=np.uint32) for a in (s0, s1, s2))
    e0, e1, e2 = (((a >> 23) & 255).astype(np.int32) for a in (s0, s1, s2))
    d = e2 - e1
    flag = (d >= 96) | (d <= -96)
    z1, z2 = (s1 & 0x7FFFFFFF) == 0, (s2 & 0x7FFFFFFF) == 0
    up0 = _ldexp64(s0)
    zero0 = (s0 & 0x7FFFFFFF) == 0
    if open_down_field:
        sub = (s0 & SIGN_BITS) | (((e0 - 64).clip(0) << 23).astype(np.uint32)) | (s0 & 0x7FFFFF)
        mid = np.where(e0 == 64, sub, _ldexp_m64(s0))
    else:
        mid = _ldexp_m64(s0)
    down = np.where(e0 == 255, s0, np.where(zero0, s0, np.where(e0 <= 63, (s0 & SIGN_BITS) | INF_BITS, mid)))
    eq01 = s0 == s1
    D = np.where(
        z1 | z2, np.uint32(0xFFC00000),
        np.where(d >= 96, np.where(eq01, up0, s0),
        np.where(e1 == 0, up0,
        np.where(e1 >= e1_down, np.where(flag & ~eq01, s0, down),
        np.where(d <= -96, np.where(eq01, s0, up0),
        np.where(e2 <= e2_threshold, up0, s0))))))
    D = D.astype(np.uint32)
    return D, flag


def cmd_div_scale(r, args):
    ins = r.bin("div_scale.in").reshape(-1, 3)
    out = r.bin("div_scale.out").reshape(-1, 2)
    assert len(ins) == len(out)
    s0, s1, s2 = ins[:, 0], ins[:, 1], ins[:, 2]
    hw_d, hw_f = out[:, 0], out[:, 1].astype(bool)
    e1 = ((s1 >> 23) & 255).astype(np.int32)
    e2 = ((s2 >> 23) & 255).astype(np.int32)
    e0 = ((s0 >> 23) & 255).astype(np.int32)
    rep = {"cases": int(len(ins)), "arch": r.arch}
    for name, kw in [("fallthrough", {}),
                     ("e2<=24,e1>=253", {"e2_threshold": 24, "e1_down": 253}),
                     ("e2<=24,e1>=253,field_sub_for_e0_64", {"e2_threshold": 24, "e1_down": 253, "open_down_field": True})]:
        D, flag = div_scale_chain(s0, s1, s2, **kw)
        if r.arch == "gfx1201":  # gfx1201 quiets every signalling NaN it returns
            snan = ((D & 0x7FFFFFFF) > INF_BITS) & ((D & QUIET_BITS) == 0)
            D = np.where(snan, D | QUIET_BITS, D).astype(np.uint32)
        bad_d, bad_f = D != hw_d, flag != hw_f
        bad = bad_d | bad_f
        d = e2 - e1
        groups = {}
        for label, m in [
            ("e1 228..253", (e1 >= 228) & (e1 <= 253)),
            ("e1 0..227", e1 <= 227),
            ("e1 254..255", e1 >= 254),
            ("d == -96, e1 != 254", (d == -96) & (e1 != 254)),
            ("e2 24..26 (threshold)", (e2 >= 24) & (e2 <= 26)),
            ("e0 64..90", (e0 >= 64) & (e0 <= 90)),
        ]:
            groups[label] = {"cases": int(m.sum()), "mismatching_D": int((bad_d & m).sum()), "mismatching_vcc": int((bad_f & m).sum())}
        first = np.flatnonzero(bad)[:12]
        rep[name] = {
            "mismatching_cases": int(bad.sum()), "mismatching_D": int(bad_d.sum()), "mismatching_vcc": int(bad_f.sum()),
            "by_group": groups,
            "first_mismatches": [
                {"s0": hex(int(s0[i])), "s1": hex(int(s1[i])), "s2": hex(int(s2[i])), "hw_d": hex(int(hw_d[i])),
                 "model_d": hex(int(D[i])), "hw_vcc": int(hw_f[i]), "model_vcc": int(flag[i])} for i in first],
        }
    write_json(r, "div_scale", rep)
    print(json.dumps(rep, indent=1))
    return rep


def cmd_scalar_udiv(r, args):
    rec = r.bin("scalar_udiv.out").reshape(-1, 4)
    n = len(rec)
    den = np.arange(1, n + 1, dtype=np.uint32)
    x, rcp, m, u = (rec[:, i] for i in range(4))
    exact_x = den.astype(np.float32).view(np.uint32)
    mul = np.float32(np.uint32(0x4F7FFFFE).view(np.float32))
    m_model = (rcp.view(np.float32) * mul).view(np.uint32)
    u_model = m.view(np.float32).astype(np.uint64).astype(np.uint32)
    # the rcp stage is read against the exact reciprocal: hardware error in ulps of the f32 result
    exact = 1.0 / den.astype(np.float64)
    ulp = np.abs(rcp.view(np.float32).astype(np.float64) - exact) / (2.0 ** (np.floor(np.log2(exact)) - 23))
    rep = {
        "dens": int(n),
        "s_cvt_f32_u32_mismatches": int((x != exact_x).sum()),
        "s_mul_f32_vs_ieee_mismatches": int((m != m_model).sum()),
        "s_cvt_u32_f32_vs_trunc_mismatches": int((u != u_model).sum()),
        "rcp_max_error_ulp": float(ulp.max()),
        "rcp_inexact_count": int((ulp != 0).sum()),
    }
    write_json(r, "scalar_udiv", rep)
    print(json.dumps(rep, indent=1))
    return rep


def _is_nan(x):
    return (x & 0x7FFFFFFF) > INF_BITS


def num_select(a, b, is_max):
    """maxNum/minNum on raw f32 bits: a NaN loses to a number, two NaNs give the first, quieted; -0 < +0."""
    a, b = np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    fa, fb = a.view(np.float32), b.view(np.float32)
    with np.errstate(invalid="ignore"):
        zero_tie = (fa == 0) & (fb == 0)
        if is_max:
            take_a = (fa > fb) | (zero_tie & ((a & SIGN_BITS) == 0) & ((b & SIGN_BITS) != 0))
        else:
            take_a = (fa < fb) | (zero_tie & ((a & SIGN_BITS) != 0) & ((b & SIGN_BITS) == 0))
    na, nb = _is_nan(a), _is_nan(b)
    out = np.where(take_a, a, b)
    out = np.where(na & ~nb, b, np.where(nb & ~na, a, out))
    return np.where(na & nb, a | QUIET_BITS, out).astype(np.uint32)


def cmd_maxmin(r, args):
    ins = r.bin("maxmin.in").reshape(-1, 3)
    s0, s1, s2 = ins[:, 0], ins[:, 1], ins[:, 2]
    rep = {"cases": int(len(ins)), "arch": r.arch}
    for name, hw_name, outer, inner in [("v_maxmin_num_f32", "maxmin.maxmin.out", False, True),
                                        ("v_minmax_num_f32", "maxmin.minmax.out", True, False)]:
        hw = r.bin(hw_name)
        model = num_select(num_select(s0, s1, inner), s2, outer)
        bad = np.flatnonzero(hw != model)
        rep[name] = {"mismatches": int(len(bad)), "first": [
            {"s0": hex(int(s0[i])), "s1": hex(int(s1[i])), "s2": hex(int(s2[i])), "hw": hex(int(hw[i])), "model": hex(int(model[i]))}
            for i in bad[:12]]}
    write_json(r, "maxmin", rep)
    print(json.dumps(rep, indent=1))
    return rep



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt")
    ap.add_argument("cmd", choices=["identity", "dot", "wmma", "swmmac", "div_scale", "scalar_udiv", "maxmin"])
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    r = Receipt(args.receipt)
    {"identity": cmd_identity, "dot": cmd_dot, "wmma": cmd_wmma, "swmmac": cmd_swmmac, "div_scale": cmd_div_scale, "scalar_udiv": cmd_scalar_udiv, "maxmin": cmd_maxmin}[args.cmd](r, args)


if __name__ == "__main__":
    main()
