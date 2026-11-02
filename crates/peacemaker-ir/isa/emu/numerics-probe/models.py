"""Reference arithmetic used by analyze.py to read raw hardware bits.

Everything here is an *offline, scalar, readable* model: the production emulator lives in crates/peacemaker-ir/src/emu
and is not imported. The WMMA dot-step model below is a line-for-line port of `mma.rs::dot_step` (gfx11 2-term /
gfx12 4-term datapath) with ONE parameter exposed, ACC_FLOOR_WINDOW, the accumulator-tail rule measured in
`wmma-capture-gfx1201-v2.bf16` (0 mismatches over all 497,794 unique operand triples with 23).
"""
from fractions import Fraction

EMPTY = -1024
FACTOR_NAN = 0xFFC00A3D
INVALID_NAN = 0xFFC00000
ACC_FLOOR_WINDOW = 23  # gfx12 accumulator negative-tail floor applies iff (grid - c.exponent) <= 23


# ------------------------------------------------------------------ format helpers
def fmt_params(kind):
    return {"f16": (5, 10, 15), "bf16": (8, 7, 127), "f32": (8, 23, 127)}[kind]


def decode_exact(bits, kind):
    """-> ('nan'|'inf'|'fin', sign, Fraction|None)."""
    eb, fb, bias = fmt_params(kind)
    sign = (bits >> (eb + fb)) & 1
    e = (bits >> fb) & ((1 << eb) - 1)
    f = bits & ((1 << fb) - 1)
    if e == (1 << eb) - 1:
        return ("inf" if f == 0 else "nan"), sign, None
    if e == 0:
        v = Fraction(f, 1 << fb) * Fraction(2) ** (1 - bias)
    else:
        v = (1 + Fraction(f, 1 << fb)) * Fraction(2) ** (e - bias)
    return "fin", sign, -v if sign else v


def encode_rne(v, kind, flush_out=False, sign_of_zero=0):
    """Fraction -> bits, round-to-nearest-even, IEEE gradual underflow (or flush), overflow -> inf."""
    eb, fb, bias = fmt_params(kind)
    if v == 0:
        return sign_of_zero << (eb + fb)
    sign = 1 if v < 0 else 0
    a = -v if sign else v
    emin = 1 - bias
    # exponent of a
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    if e < emin:
        q = a / (Fraction(2) ** (emin - fb))  # units of the subnormal ulp
        r = _rne_int(q)
        if flush_out:
            return sign << (eb + fb)
        if r >= (1 << fb):
            return (sign << (eb + fb)) | (1 << fb)
        return (sign << (eb + fb)) | r
    q = a / (Fraction(2) ** (e - fb))
    r = _rne_int(q)
    if r >= (1 << (fb + 1)):
        r >>= 1
        e += 1
    if e + bias >= (1 << eb) - 1:
        return (sign << (eb + fb)) | (((1 << eb) - 1) << fb)
    return (sign << (eb + fb)) | ((e + bias) << fb) | (r & ((1 << fb) - 1))


def _rne_int(q):
    fl = q.numerator // q.denominator
    rem = q - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        return fl + 1
    return fl


def f32_bits_to_f16_rne(bits, kind):
    """f32 bits -> f16/bf16 bits, RNE, NaN quieted keeping the payload's top bits."""
    cls, sign, v = decode_exact(bits, "f32")
    eb, fb, _ = fmt_params(kind)
    if cls == "nan":
        return (sign << (eb + fb)) | (((1 << eb) - 1) << fb) | (1 << (fb - 1)) | ((bits >> (23 - fb)) & ((1 << (fb - 1)) - 1))
    if cls == "inf":
        return (sign << (eb + fb)) | (((1 << eb) - 1) << fb)
    return encode_rne(v, kind, sign_of_zero=sign)


# ------------------------------------------------------------------ WMMA dot step (port of mma.rs)
def _factor(bits, bf16, gfx11):
    fb, bias, inf = (7, 127, 0x7F80) if bf16 else (10, 15, 0x7C00)
    mag = bits & 0x7FFF
    e = mag >> fb
    f = mag & ((1 << fb) - 1)
    sig = (f | (1 << fb)) if e else (0 if (bf16 and gfx11) else f)
    return dict(sig=sig, al=max(e, 1) - bias, neg=bool(bits & 0x8000), nan=mag > inf, inf=mag == inf)


def _round_even(v, sh):
    if sh <= 0:
        return v << (-sh)
    if sh >= 64:
        return 0
    tail = v & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    head = v >> sh
    return head + (1 if (tail > half or (tail == half and head & 1)) else 0)


def _pack(units, grid, frame_neg, gfx11):
    if units == 0:
        return 0
    sign = (1 if ((units < 0) != frame_neg) else 0) << 31
    mag = abs(units)
    top = mag.bit_length() - 1
    exp = top + grid
    if not gfx11 and exp < -126:
        fr = _round_even(mag, -149 - grid)
        return 0 if fr == 0 else sign | fr
    sig = _round_even(mag, top - 23)
    if sig == 0x1000000:
        sig >>= 1
        exp += 1
    if exp < -126:
        return 0
    if exp > 127:
        return sign | 0x7F800000
    return sign | ((exp + 127) << 23) | (sig & 0x7FFFFF)


def _lead(sig, exp):
    return EMPTY if sig == 0 else exp + sig.bit_length() - 1


def _mag(sig, exp, grid):
    sh = exp - grid
    if sig == 0 or sh <= -64:
        return 0
    return sig >> (-sh) if sh < 0 else sig << sh


def _afloor(sig, exp, neg, grid, accum, window):
    m = _mag(sig, exp, grid)
    if not neg:
        return m
    sh = exp - grid
    tail = sig != 0 and sh < 0 and (sh <= -64 or (sig & ((1 << (-sh)) - 1)) != 0)
    if accum:
        sub = tail and (m != 0 if window is None else (-sh) <= window)
    else:
        sub = tail
    return -m - (1 if sub else 0)


def dot_step(a, b, acc, bf16, gfx11, window=ACC_FLOOR_WINDOW):
    n = len(a)
    prods, grids = [], []
    pinf = ninf = inval = False
    fb = 7 if bf16 else 10
    for i in range(n):
        l = _factor(a[i], bf16, gfx11)
        r = _factor(b[i], bf16, gfx11)
        if l["nan"] or r["nan"]:
            return FACTOR_NAN
        inval |= (l["inf"] and r["sig"] == 0) or (r["inf"] and l["sig"] == 0)
        if l["inf"] or r["inf"]:
            if l["neg"] != r["neg"]:
                ninf = True
            else:
                pinf = True
        sig = l["sig"] * r["sig"]
        prods.append((sig, l["al"] + r["al"] - 2 * fb, l["neg"] != r["neg"]))
        grids.append(l["al"] + r["al"] - 24 if sig else EMPTY)
    if inval or (pinf and ninf):
        return INVALID_NAN
    if acc & 0x7FFFFFFF > 0x7F800000:
        return acc | 0x400000
    pinf |= acc == 0x7F800000
    ninf |= acc == 0xFF800000
    if pinf or ninf:
        return INVALID_NAN if (pinf and ninf) else (0xFF800000 if ninf else 0x7F800000)
    ae = (acc >> 23) & 255
    csig = ((acc & 0x7FFFFF) | 0x800000) if ae else (0 if gfx11 else acc & 0x7FFFFF)
    cexp = (ae - 150) if gfx11 else (max(ae, 1) - 150)
    cneg = acc >> 31 != 0
    pgrid = max(grids)
    if gfx11:
        if csig == 0 and prods[0][0] == 0 and prods[1][0] == 0:
            return 0
        leading = max(_lead(csig, cexp), _lead(prods[0][0], prods[0][1]), _lead(prods[1][0], prods[1][1]))
        grid = max(pgrid, leading - 26)
        total = 0
        for sig, exp, neg in ((csig, cexp, cneg), prods[0], prods[1]):
            m = _mag(sig, exp, grid)
            total += (-m - 1) if neg != cneg else m
        return _pack(total, grid, cneg, True)
    grid = max(pgrid, max(_lead(csig, cexp), -126) - 26)
    total = _afloor(csig, cexp, cneg, grid, True, window)
    for i in range(0, n, 2):
        pg = max(grids[i], grids[i + 1])
        pair = _afloor(*prods[i], pg, False, window) + _afloor(*prods[i + 1], pg, False, window)
        total += _afloor(abs(pair), pg, pair < 0, grid, False, window)
    return _pack(total, grid, False, False)


def wmma_f32(arch, a_phys, b_phys, c, bf16, window=ACC_FLOOR_WINDOW):
    """16-term chain in *physical* K slots (A[row*16+k] as placed in the harness). gfx12: logical k->phys swaps bits 2/3."""
    if arch == "gfx1201":
        order = [(k & 3) | ((k & 4) << 1) | ((k & 8) >> 1) for k in range(16)]
        step = 4
    else:
        order = list(range(16))
        step = 2
    a = [a_phys[o] for o in order]
    b = [b_phys[o] for o in order]
    for k in range(0, 16, step):
        c = dot_step(a[k:k + step], b[k:k + step], c, bf16, step == 2, window)
    return c


# ------------------------------------------------------------------ integer models
def sx(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def wrap32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


def sat32(v):
    return max(-(1 << 31), min((1 << 31) - 1, v))


def dot_int(a, b, c, nbits, nel, a_signed, b_signed, clamp):
    """v_dot4_i32_iu8 (nbits=8, nel=4) / v_dot8_i32_iu4 (nbits=4, nel=8): c + sum a_i*b_i; single final wrap/saturate."""
    s = 0
    m = (1 << nbits) - 1
    for i in range(nel):
        x = (a >> (nbits * i)) & m
        y = (b >> (nbits * i)) & m
        s += (sx(x, nbits) if a_signed else x) * (sx(y, nbits) if b_signed else y)
    t = sx(c, 32) + s
    return (sat32(t) if clamp else wrap32(t)) & 0xFFFFFFFF
