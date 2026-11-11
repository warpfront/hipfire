#!/usr/bin/env python3
"""Select the committed gfx1151 regression fixtures from a numerics_probe receipt.

  make_halo_fixtures.py <receipt-dir> <out.bin> [<iu8_clamp-receipt-dir>]

Writes crates/peacemaker-ir/tests/fixtures/halo_numerics_waves.bin (format: see emu/hw_replay.rs). Selection is
deterministic and discriminating: a float wave is taken only where the *gfx1201* chain of models.py (the model this
fixture exists to keep gfx1151 away from) disagrees with the recorded gfx1151 hardware, an integer-clamp wave only
where a single final saturation disagrees, a modifier variant only where it changes the unmodified result, and a
float dot case only where the exact single-rounded sum disagrees. The IU8 CLAMP variants (v4..v7) come from the
`iu8_clamp` suite receipt when one is given (its waves discriminate the saturation points for every signedness
combination; the main set's ladders only do for a signed B). The receipts are not rewritten.
"""
import json
import os
import struct
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M  # noqa: E402

rec_dir, out_path = sys.argv[1], sys.argv[2]
clamp_dir = sys.argv[3] if len(sys.argv) > 3 else rec_dir
WC = np.dtype([("A", "<u2", 256), ("B", "<u2", 256), ("C", "<u4", 256)])
OUT = bytearray()


def parse_mods(text):
    m = [0, 0, 0, 7, 0]
    for tok in text.split():
        if tok == "clamp":
            m[4] = 1
            continue
        key, rest = tok.split(":")
        bits = sum((1 << i) for i, v in enumerate(rest.strip("[]").split(",")) if v == "1")
        m[{"neg_lo": 0, "neg_hi": 1, "op_sel": 2, "op_sel_hi": 3}[key]] = bits
    return m


def header(tag, op, mods):
    return struct.pack("<II", tag, op) + bytes([mods[0], mods[1], mods[2], mods[3], mods[4], 0, 0, 0])


def meta(name, d=None):
    return json.load(open(os.path.join(d or rec_dir, name + ".json")))


def seg_bounds(bf, n_sub, n):
    f5 = 3 * 2 * (16 if bf else 2) * 8
    out, pos = {}, 0
    for name, cnt in (("controls", 3), ("F1", 16 * 4 * 32), ("F4", 32 * 6 * 4), ("F5", f5)):
        out[name] = (pos, pos + cnt)
        pos += cnt
    out["F3a"] = (pos, n_sub)
    pos = n_sub
    out["F2"] = (pos, pos + 15360)
    pos += 15360
    out["F3b"] = (pos, n)
    return out


def wave_model(arch, bf, w):
    res = []
    for r in range(16):
        ar = [int(x) for x in w["A"][r * 16:r * 16 + 16]]
        for c in range(16):
            bc = [int(x) for x in w["B"][c * 16:c * 16 + 16]]
            res.append(M.wmma_f32(arch, ar, bc, int(w["C"][r * 16 + c]), bf))
    return res


def hw_logical(raw, n):
    """raw [n][8][32] -> [n][16][16] (gfx11: row = 2*vgpr + lane//16, col = lane % 16)."""
    d = np.zeros((n, 16, 16), dtype=np.uint32)
    for vg in range(8):
        for lane in range(32):
            d[:, 2 * vg + (lane >> 4), lane & 15] = raw[:, vg, lane]
    return d


def pick_float(kind, op_name, per_seg=3, scan=400):
    bf = kind == 0
    cs = np.fromfile(os.path.join(rec_dir, f"wmma.{op_name}.in.bin"), dtype=WC)
    n = meta(f"wmma.{op_name}.v0.m0.out")["cases"]
    n_sub = meta(f"wmma.{op_name}.in")["n_sub"]
    raw = np.fromfile(os.path.join(rec_dir, f"wmma.{op_name}.v0.m0.out.bin"), dtype="<u4").reshape(n, 8, 32)
    hwl = hw_logical(raw, n)
    chosen = []
    for seg, (lo, hi) in seg_bounds(bf, n_sub, n).items():
        if seg == "controls":
            chosen.append(lo)  # a layout control anchors the register mapping
            continue
        got = 0
        for w in range(lo, min(hi, lo + scan)):
            mdl = wave_model("gfx1201", bf, cs[w])
            if any(int(hwl[w][z // 16][z % 16]) != mdl[z] for z in range(256)):
                chosen.append(w)
                got += 1
                if got == per_seg:
                    break
    for w in chosen:
        OUT.extend(header(1, kind, [0, 0, 0, 7, 0]))
        OUT.extend(cs[w].tobytes())
        OUT.extend(raw[w].tobytes())
    # modifier variants: one wave each where the variant changes the unmodified hardware result
    lo, hi = seg_bounds(bf, n_sub, n)["F1"]
    for v in range(1, 8):
        mm = meta(f"wmma.{op_name}.v{v}.m0.out")
        nv = mm["cases"]
        rv = np.fromfile(os.path.join(rec_dir, f"wmma.{op_name}.v{v}.m0.out.bin"), dtype="<u4").reshape(nv, 8, 32)
        for w in range(lo, min(hi, nv)):
            if (rv[w] != raw[w]).any():
                OUT.extend(header(1, kind, parse_mods(mm["mods"])))
                OUT.extend(cs[w].tobytes())
                OUT.extend(rv[w].tobytes())
                break
        else:
            raise SystemExit(f"{op_name} v{v}: no wave where the modifier changes the result")
    return len(chosen) + 7


def pick_iu8():
    op_name = "v_wmma_i32_16x16x16_iu8"
    cs = np.fromfile(os.path.join(rec_dir, f"wmma.{op_name}.in.bin"), dtype=WC)
    n = meta(f"wmma.{op_name}.v0.m0.out")["cases"]
    raw0 = np.fromfile(os.path.join(rec_dir, f"wmma.{op_name}.v0.m0.out.bin"), dtype="<u4").reshape(n, 8, 32)
    for w in (0, 1, 5000):
        OUT.extend(header(1, 2, [0, 0, 0, 7, 0]))
        OUT.extend(cs[w].tobytes())
        OUT.extend(raw0[w].tobytes())
    count = 3
    for v in range(1, 8):
        mm0 = meta(f"wmma.{op_name}.v{v}.m0.out")
        mods = parse_mods(mm0["mods"])
        a_s, b_s, clamp = bool(mods[0] & 1), bool(mods[0] & 2), bool(mods[4])
        d = clamp_dir if clamp else rec_dir
        mm = meta(f"wmma.{op_name}.v{v}.m0.out", d)
        nv = mm["cases"]
        csd = np.fromfile(os.path.join(d, f"wmma.{op_name}.in.bin"), dtype=WC)
        rv = np.fromfile(os.path.join(d, f"wmma.{op_name}.v{v}.m0.out.bin"), dtype="<u4").reshape(nv, 8, 32)
        if clamp:
            A = csd["A"][:nv].astype(np.int64).reshape(nv, 16, 16) & 255
            B = csd["B"][:nv].astype(np.int64).reshape(nv, 16, 16) & 255
            C = csd["C"][:nv].astype(np.int64).reshape(nv, 16, 16)
            C = np.where(C >= 2 ** 31, C - 2 ** 32, C)
            Aa = np.where(A >= 128, A - 256, A) if a_s else A
            Bb = np.where(B >= 128, B - 256, B) if b_s else B
            final = np.clip(np.einsum("nrk,nck->nrc", Aa, Bb) + C, -2 ** 31, 2 ** 31 - 1)
            hwl = hw_logical(rv, nv).astype(np.int64)
            hwl = np.where(hwl >= 2 ** 31, hwl - 2 ** 32, hwl)
            diff = np.nonzero((final != hwl).reshape(nv, -1).any(axis=1))[0]  # single final saturation disagrees
            take = [int(x) for x in diff[:3]] or [0, 1]
        else:
            diff = np.nonzero((rv != raw0[:nv]).reshape(nv, -1).any(axis=1))[0]
            take = [int(diff[0])]
        for w in take:
            OUT.extend(header(1, 2, mods))
            OUT.extend(csd[w].tobytes())
            OUT.extend(rv[w].tobytes())
            count += 1
    return count


def exact_dot2(bf, mods, a, b, c):
    """Exact single-rounded RNE a0*b0 + a1*b1 + c for finite default-modifier cases, else None."""
    if mods[0] or mods[1] or mods[2] or mods[3] != 7:
        return None
    kind = "bf16" if bf else "f16"
    vals = []
    for x in (a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16):
        cls, _, v = M.decode_exact(x, kind)
        if cls != "fin":
            return None
        vals.append(v)
    cls, _, cv = M.decode_exact(c, "f32")
    if cls != "fin":
        return None
    return M.encode_rne(vals[0] * vals[2] + vals[1] * vals[3] + cv, "f32")


DOT_OPS = ["v_dot2_f32_f16", "v_dot2_f32_bf16", "v_dot4_i32_iu8", "v_dot8_i32_iu4"]


def pick_dot():
    total, per = 0, 24
    for op, name in enumerate(DOT_OPS):
        for setname in ("full", "red"):
            inp = np.fromfile(os.path.join(rec_dir, f"dot.{name}.{setname}.in.bin"), dtype="<u4").reshape(-1, 3)
            for v in range(15 if op < 2 else 10):
                if setname == "full" and v:
                    continue
                path = os.path.join(rec_dir, f"dot.{name}.{setname}.v{v}.m0.out")
                mods = parse_mods(meta(os.path.basename(path))["mods"])
                hw = np.fromfile(path + ".bin", dtype="<u4")
                cand = np.arange(len(hw))[:: max(1, len(hw) // 3000)]
                idx = []
                if op < 2:
                    for i in cand:  # the exact sum is wrong where the measured chain is right
                        ex = exact_dot2(op == 1, mods, int(inp[i, 0]), int(inp[i, 1]), int(inp[i, 2]))
                        if ex is not None and ex != int(hw[i]):
                            idx.append(int(i))
                            if len(idx) == per:
                                break
                idx += [int(i) for i in cand[: per - len(idx)]]
                for i in idx:
                    OUT.extend(header(2, op, mods))
                    OUT.extend(struct.pack("<IIII", int(inp[i, 0]), int(inp[i, 1]), int(inp[i, 2]), int(hw[i])))
                    total += 1
    return total


wm = pick_float(0, "v_wmma_f32_16x16x16_bf16") + pick_float(1, "v_wmma_f32_16x16x16_f16") + pick_iu8()
dt = pick_dot()
open(out_path, "wb").write(OUT)
print(f"wrote {out_path}: {wm} wmma waves, {dt} dot cases, {len(OUT)} bytes")
