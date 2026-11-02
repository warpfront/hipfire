#!/usr/bin/env python3
"""Exact-value validation of the SWMMAC layout/index model for every kind (gfx1201).
usage: swmmac_valid.py <receipt-dir>"""
import json, os, sys, collections
from fractions import Fraction
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M

SK = ['v_swmmac_f32_16x16x32_f16', 'v_swmmac_f32_16x16x32_bf16', 'v_swmmac_f16_16x16x32_f16', 'v_swmmac_bf16_16x16x32_bf16',
      'v_swmmac_i32_16x16x32_iu8', 'v_swmmac_i32_16x16x32_iu4', 'v_swmmac_i32_16x16x64_iu4', 'v_swmmac_f32_16x16x32_fp8_fp8',
      'v_swmmac_f32_16x16x32_fp8_bf8', 'v_swmmac_f32_16x16x32_bf8_fp8', 'v_swmmac_f32_16x16x32_bf8_bf8']
BITS = [16, 16, 16, 16, 8, 4, 4, 8, 8, 8, 8]
SA = [4, 4, 4, 4, 2, 1, 2, 2, 2, 2, 2]; SB = [8, 8, 8, 8, 4, 2, 4, 4, 4, 4, 4]; SD = [8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8]
S = np.dtype([('a', '<u4', (4, 32)), ('b', '<u4', (8, 32)), ('idx', '<u4', 32), ('c', '<u4', (8, 32))])


def RB(rcpt, name):
    p = f'{rcpt}/{name}.bin'
    if os.path.exists(p): return open(p, 'rb').read()
    import subprocess
    return subprocess.run(['zstd', '-dc', p + '.zst'], check=True, capture_output=True).stdout


def fp8_val(bits, e4m3):
    eb, fb, bias = (4, 3, 7) if e4m3 else (5, 2, 15)
    sign = bits >> 7; e = (bits >> fb) & ((1 << eb) - 1); f = bits & ((1 << fb) - 1)
    if e4m3 and e == 15 and f == 7: return None
    if (not e4m3) and e == 31: return None
    v = (Fraction(f, 1 << fb) * Fraction(2) ** (1 - bias)) if e == 0 else (1 + Fraction(f, 1 << fb)) * Fraction(2) ** (e - bias)
    return -v if sign else v


def elem(kind, bits, second, signed):
    if kind <= 3:
        k = 'bf16' if kind in (1, 3) else 'f16'
        c, s, v = M.decode_exact(bits, k); return v if c == 'fin' else None
    if kind in (4, 5, 6):
        nb = BITS[kind]; v = bits & ((1 << nb) - 1)
        return M.sx(v, nb) if signed else v
    e4 = (kind == 7) or (kind == 8 and not second) or (kind == 9 and second)
    return fp8_val(bits, e4)


def getel(regs, lane, e, bits):
    per = 32 // bits
    return (int(regs[e // per][lane]) >> (bits * (e % per))) & ((1 << bits) - 1)


def wave_model(kind, c, key, flags):
    bits = BITS[kind]; nA = SA[kind] * (32 // bits); nB = SB[kind] * (32 // bits)
    a_signed, b_signed, clamp, neg = flags
    Acomp = [[None] * (2 * nA) for _ in range(16)]; sel = [[0] * (2 * nA) for _ in range(16)]
    for l in range(32):
        row, h = l & 15, l >> 4
        for i in range(nA):
            raw = getel(c['a'], l, i, bits)
            v = elem(kind, raw, False, a_signed)
            if neg.get('A_even') and i % 2 == 0 and v is not None: v = -v
            if neg.get('A_odd') and i % 2 == 1 and v is not None: v = -v
            Acomp[row][nA * h + i] = v
            off = (16 * key + 2 * i) if nA == 8 else 2 * i
            sel[row][nA * h + i] = (int(c['idx'][l]) >> off) & 3
    nK = 2 * nB // 1 if False else (64 if kind == 6 else 32)
    Bm = [[None] * 16 for _ in range(nK)]
    for l in range(32):
        col, hb = l & 15, l >> 4
        for e in range(nB):
            v = elem(kind, getel(c['b'], l, e, bits), True, b_signed)
            if neg.get('B_even') and e % 2 == 0 and v is not None: v = -v
            if neg.get('B_odd') and e % 2 == 1 and v is not None: v = -v
            Bm[(nK // 2) * hb + e][col] = v
    return Acomp, sel, Bm, nA, nK


def run(rcpt):
    idx = [n for n in open(f'{rcpt}/INDEX.txt').read().split()]
    out = {}
    for kind in range(11):
        nm = SK[kind]
        inn = f'swmmac.{nm}.valid.in'
        if inn not in idx: continue
        cs = np.frombuffer(RB(rcpt, inn), dtype=S)
        for res in [n for n in idx if n.startswith(f'swmmac.{nm}.valid.v')]:
            meta = json.load(open(f'{rcpt}/{res}.json')); v = meta['variant']
            raw = np.frombuffer(RB(rcpt, res), dtype='<u4').reshape(len(cs), meta['d_regs'], 32)
            key = 1 if v == 1 else 0
            flags = [0, 0, 0, {}]
            if kind in (4, 5, 6):
                if v == 2: flags[0] = 1
                if v == 3: flags[1] = 1
                if v == 4: flags[0] = flags[1] = 1
                if v == 5: flags[2] = 1
                if v == 6: flags[0] = flags[1] = flags[2] = 1
            elif kind <= 3:
                if v == 2: flags[3] = {'A_even': 1}
                if v == 3: flags[3] = {'B_even': 1}
                if v == 4: flags[3] = {'A_odd': 1}
                if v == 5: flags[3] = {'B_odd': 1}
                if v == 6: flags[3] = {'A_even': 1, 'A_odd': 1, 'B_even': 1, 'B_odd': 1}
            stat = collections.Counter(); ex = []
            nsel = 400
            for wi in range(nsel):
                c = cs[wi]
                Acomp, sel, Bm, nA, nK = wave_model(kind, c, key, flags)
                for l in range(32):
                    col, h = l & 15, l >> 4
                    for j in range(SD[kind]) if (kind > 3 or kind <= 1) else range(4):
                        pass
                for row in range(16):
                    for col in range(16):
                        terms = []
                        ok = True
                        for j in range(2 * nA):
                            K = 4 * (j >> 1) + sel[row][j]
                            av, bv = Acomp[row][j], Bm[K][col]
                            if av is None or bv is None: ok = False; break
                            terms.append(av * bv)
                        # C
                        if kind <= 3 and SD[kind] == 4:
                            lane = col + 16 * (row // 8); jj = (row % 8) // 2; half = row % 2
                            cin = (int(c['c'][jj][lane]) >> (16 * half)) & 0xFFFF
                            cv = M.decode_exact(cin, 'bf16' if kind == 3 else 'f16')[2]
                            hwraw = (int(raw[wi][jj][lane]) >> (16 * half)) & 0xFFFF
                        else:
                            lane = col + 16 * (row // 8); jj = row % 8
                            cin = int(c['c'][jj][lane]); hwraw = int(raw[wi][jj][lane])
                            cv = M.sx(cin, 32) if kind in (4, 5, 6) else M.decode_exact(cin, 'f32')[2]
                        if not ok or cv is None: stat['skipped_special'] += 1; continue
                        tot = cv + sum(terms)
                        if kind in (4, 5, 6):
                            t = int(tot)
                            hwv = M.sx(hwraw, 32)
                            mw = M.wrap32(t); ms = M.sat32(t)
                            exp_ = ms if flags[2] else mw
                            stat['words'] += 1
                            if hwraw == (exp_ & 0xFFFFFFFF): stat['match'] += 1
                            else:
                                stat['mismatch'] += 1
                                if len(ex) < 4: ex.append((wi, row, col, hex(hwraw), hex(exp_ & 0xFFFFFFFF), 'sat' if hwraw == (ms & 0xFFFFFFFF) else ''))
                        else:
                            ek = {2: 'f16', 3: 'bf16'}.get(kind, 'f32')
                            if tot == 0: expb = None
                            else: expb = M.encode_rne(tot, ek)
                            stat['words'] += 1
                            if (expb is None and (hwraw & (0x7FFF if ek != 'f32' else 0x7FFFFFFF)) == 0) or expb == hwraw: stat['match'] += 1
                            else:
                                stat['mismatch'] += 1
                                if len(ex) < 4: ex.append((wi, row, col, hex(hwraw), None if expb is None else hex(expb)))
            out[res] = dict(stat); out[res]['examples'] = ex
            print(res, meta['mode_readback'], dict(stat), ex[:2])
    return out


if __name__ == '__main__':
    r = sys.argv[1]
    o = run(r)
    os.makedirs(f'{r}/analysis', exist_ok=True)
    json.dump(o, open(f'{r}/analysis/swmmac_valid.json', 'w'), indent=1, default=str)
