//! CPU bit-exact references for the gfx1201 A3B exact batch-route lane kernels
//! (`gemv_mq4g256v2_lane_proj_k2048.gfx1201.hip`, wave64, and
//! `gemv_mq4g256v2_residual_lanes.gfx1201.hip`, wave32).
//!
//! Pure std. Every operation is the kernel's documented f32 order; `fma` is
//! `f32::mul_add` (single rounding), `+`/`*` are IEEE f32. f32 denormals are
//! preserved (gfx1201 hipcc default `denormal-fp-math-f32=ieee`).
//!
//! MQ4G256V2 row = K/256 groups x 136 B:
//! `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][128 B nibbles]`. Lane `l` (0..32) of
//! group `g` owns elements `g*256 + l*8 + k` (k = 0..8) = nibble `k` of the
//! little-endian dword at byte `8 + 4*l`; its `(sc, zp)` is header half
//! `(l < 16 ? 0 : 1)`.
//!
//! NaN policy for comparing a kernel output with these references: non-NaN
//! words must be bit-equal; a NaN must meet a NaN (payload/sign differences
//! are tallied separately, see [`compare_words`]).

#![allow(dead_code)]

pub const GROUP_BYTES: usize = 136;

/// Exact IEEE binary16 -> binary32 (denormals, inf, NaN payload preserved).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // denormal half -> normal single
            let lz = man.leading_zeros() - 22; // leading zeros within the 10-bit field
            let m = (man << (lz + 1)) & 0x3FF;
            let e = 127 - 15 - lz;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn rd16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// `u_k = fma(sc, (f32)nibble_k, zp)` for the 8 nibbles of `pv`.
#[inline]
fn dequant8(sc: f32, zp: f32, pv: u32) -> [f32; 8] {
    let mut u = [0f32; 8];
    for (k, slot) in u.iter_mut().enumerate() {
        *slot = sc.mul_add(((pv >> (4 * k)) & 0xF) as f32, zp);
    }
    u
}

/// The singleton's 32-lane `__shfl_down` ladder 16, 8, 4, 2, 1 (lane 0 result).
/// Lanes whose partner is out of range keep their value (they never reach lane 0).
fn tree32(a: &mut [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        for l in 0..32 - off {
            a[l] += a[l + off];
        }
    }
    a[0]
}

/// `lane_proj` reference: up to four weight segments `weights[s]` (rows
/// `rows[s]`, K = 2048 = 8 groups) against `lanes` activation columns
/// `x[b*2048 ..]`. Returns per segment `[lanes][rows_s]` f32 bit patterns
/// (`out[s][b * rows_s + row]`).
pub fn lane_proj_ref(weights: &[&[u8]], rows: &[usize], x: &[f32], lanes: usize) -> Vec<Vec<u32>> {
    const K: usize = 2048;
    const GROUPS: usize = 8;
    assert_eq!(weights.len(), rows.len());
    assert!(x.len() >= lanes * K, "x must hold [lanes][2048] f32");
    let row_bytes = GROUPS * GROUP_BYTES;
    let mut out = Vec::with_capacity(weights.len());
    for (w, &m) in weights.iter().zip(rows) {
        assert!(
            w.len() >= m * row_bytes,
            "weights shorter than rows * 1088 B"
        );
        let mut seg = vec![0u32; lanes * m];
        for row in 0..m {
            let rbase = row * row_bytes;
            for b in 0..lanes {
                let xb = &x[b * K..(b + 1) * K];
                let mut part = [0f32; 32];
                for (l, slot) in part.iter_mut().enumerate() {
                    let mut acc = [0f32; 4];
                    for p in 0..4 {
                        for h in 0..2 {
                            let g = p + 4 * h;
                            let gb = rbase + g * GROUP_BYTES;
                            let hdr = gb + if l < 16 { 0 } else { 4 };
                            let sc = f16_to_f32(rd16(w, hdr));
                            let zp = f16_to_f32(rd16(w, hdr + 2));
                            let pv = rd32(w, gb + 8 + 4 * l);
                            let u = dequant8(sc, zp, pv);
                            let xe = &xb[g * 256 + l * 8..g * 256 + l * 8 + 8];
                            let mut s = u[0].mul_add(xe[0], u[1] * xe[1]);
                            for k in 2..8 {
                                s = u[k].mul_add(xe[k], s);
                            }
                            acc[p] += s;
                        }
                    }
                    *slot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                }
                seg[b * m + row] = tree32(&mut part).to_bits();
            }
        }
        out.push(seg);
    }
    out
}

/// `residual_lanes` reference: `y[b][row] += W[row] . x[b]` for `m` rows,
/// K = 4096 (16 groups), `x` is `[lanes][4096]`, `y` is `[lanes][m]`.
pub fn residual_lanes_ref(w: &[u8], x: &[f32], y: &mut [f32], m: usize, lanes: usize) {
    const K: usize = 4096;
    const GROUPS: usize = 16;
    let row_bytes = GROUPS * GROUP_BYTES;
    assert!(w.len() >= m * row_bytes, "weights shorter than m * 2176 B");
    assert!(x.len() >= lanes * K, "x must hold [lanes][4096] f32");
    assert!(y.len() >= lanes * m, "y must hold [lanes][m] f32");
    for row in 0..m {
        let rbase = row * row_bytes;
        for b in 0..lanes {
            let xb = &x[b * K..(b + 1) * K];
            let mut part = [0f32; 32];
            for (l, slot) in part.iter_mut().enumerate() {
                let mut acc = [0f32; 4];
                for (j, aj) in acc.iter_mut().enumerate() {
                    for q in 0..4 {
                        let g = j + 4 * q;
                        let gb = rbase + g * GROUP_BYTES;
                        let hdr = gb + if l < 16 { 0 } else { 4 };
                        let sc = f16_to_f32(rd16(w, hdr));
                        let zp = f16_to_f32(rd16(w, hdr + 2));
                        let pv = rd32(w, gb + 8 + 4 * l);
                        let wv = dequant8(sc, zp, pv);
                        let xe = &xb[g * 256 + l * 8..g * 256 + l * 8 + 8];
                        let mut t = if row & 1 == 0 {
                            let t = wv[0] * xe[0];
                            wv[1].mul_add(xe[1], t)
                        } else {
                            let t = wv[1] * xe[1];
                            wv[0].mul_add(xe[0], t)
                        };
                        for k in 2..8 {
                            t = wv[k].mul_add(xe[k], t);
                        }
                        *aj += t;
                    }
                }
                *slot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            let s = tree32(&mut part);
            y[b * m + row] += s;
        }
    }
}

/// Word-wise comparison under the lab NaN policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WordCmp {
    /// non-NaN words (or NaN vs non-NaN) that differ: failures
    pub mismatches: usize,
    /// NaN meets NaN with different bits: tallied, not failures
    pub nan_payload_diffs: usize,
    pub words: usize,
}

impl WordCmp {
    pub fn ok(&self) -> bool {
        self.mismatches == 0
    }
}

pub fn compare_words(got: &[u32], want: &[u32]) -> WordCmp {
    assert_eq!(got.len(), want.len());
    let mut c = WordCmp {
        words: got.len(),
        ..Default::default()
    };
    for (&g, &w) in got.iter().zip(want) {
        let (gn, wn) = (f32::from_bits(g).is_nan(), f32::from_bits(w).is_nan());
        if gn && wn {
            if g != w {
                c.nan_payload_diffs += 1;
            }
        } else if g != w {
            c.mismatches += 1;
        }
    }
    c
}

/// Index of the first failing word, if any (for diagnostics).
pub fn first_mismatch(got: &[u32], want: &[u32]) -> Option<usize> {
    got.iter().zip(want).position(|(&g, &w)| {
        let (gn, wn) = (f32::from_bits(g).is_nan(), f32::from_bits(w).is_nan());
        !(gn && wn) && g != w
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: f32) -> u16 {
        // exact for the small values used below
        let b = v.to_bits();
        if v == 0.0 {
            return (b >> 16) as u16 & 0x8000;
        }
        let sign = ((b >> 16) & 0x8000) as u16;
        let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let m = ((b >> 13) & 0x3FF) as u16;
        assert!(
            (1..31).contains(&e) && (b & 0x1FFF) == 0,
            "not exactly representable"
        );
        sign | ((e as u16) << 10) | m
    }

    fn put_group(
        buf: &mut [u8],
        gb: usize,
        sc: [u16; 2],
        zp: [u16; 2],
        nibbles: impl Fn(usize, usize) -> u32,
    ) {
        buf[gb..gb + 2].copy_from_slice(&sc[0].to_le_bytes());
        buf[gb + 2..gb + 4].copy_from_slice(&zp[0].to_le_bytes());
        buf[gb + 4..gb + 6].copy_from_slice(&sc[1].to_le_bytes());
        buf[gb + 6..gb + 8].copy_from_slice(&zp[1].to_le_bytes());
        for l in 0..32 {
            let mut d = 0u32;
            for k in 0..8 {
                d |= (nibbles(l, k) & 0xF) << (4 * k);
            }
            buf[gb + 8 + 4 * l..gb + 12 + 4 * l].copy_from_slice(&d.to_le_bytes());
        }
    }

    #[test]
    fn f16_decode() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E01).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        // exhaustive agreement with a bit-twiddle-free definition for finite values
        for v in 0u16..=0xFFFF {
            let f = f16_to_f32(v);
            let e = ((v >> 10) & 31) as i32;
            let m = (v & 0x3FF) as f32;
            let s = if v & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            if e == 31 {
                assert_eq!(f.is_nan(), v & 0x3FF != 0);
                assert_eq!(f.is_infinite(), v & 0x3FF == 0);
            } else {
                let want = if e == 0 {
                    s * m * 2f32.powi(-24)
                } else {
                    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
                };
                assert_eq!(f.to_bits(), want.to_bits(), "half {v:#06x}");
            }
        }
    }

    /// Hand-computed: row 0, one lane l=0 weights, all others zero nibbles with zp=0,
    /// x nonzero only at lane 0 of group 0 and group 4.
    #[test]
    fn lane_proj_hand_case() {
        // sc=0.5, zp=-1.0 (header half 0); nibbles of lane 0, group g: k -> k
        let mut w = vec![0u8; 8 * GROUP_BYTES];
        for g in 0..8 {
            put_group(
                &mut w,
                g * GROUP_BYTES,
                [h(0.5), h(0.0)],
                [h(-1.0), h(0.0)],
                |l, k| if l == 0 { k as u32 } else { 0 },
            );
        }
        // u_k = 0.5*k - 1 = [-1,-0.5,0,0.5,1,1.5,2,2.5]
        let mut x = vec![0f32; 2048];
        for k in 0..8 {
            x[k] = (k + 1) as f32; // group 0, lane 0
            x[4 * 256 + k] = 2.0; // group 4, lane 0
        }
        // group 0 sum = sum u_k*(k+1) = -1 -1 +0 +2 +5 +9 +14 +20 = 48
        // group 4 sum = 2*sum u_k = 2*6 = 12
        // acc0 = 0 + 48 + 12 = 60. Every other lane has u = zp = 0 (sc*q+zp with lane-1.. hdr half 0
        // for l<16: u = 0.5*0 - 1 = -1!) -> lanes 1..16 contribute -sum(x)=0 since x is zero there.
        let out = lane_proj_ref(&[&w], &[1], &x, 1);
        assert_eq!(f32::from_bits(out[0][0]), 60.0);
        // second lane column of zeros in x gives exactly +0.0 (or -0.0?): 0*u with u<0 -> -0.0 products;
        // acc starts +0.0 so sums stay +0.0.
        let x2 = vec![0f32; 2 * 2048];
        let o2 = lane_proj_ref(&[&w], &[1], &x2, 2);
        assert_eq!(o2[0], vec![0u32, 0u32]);
    }

    #[test]
    fn lane_proj_neg_zero_and_nan() {
        // all-zero weights/headers: u = 0 -> products +-0; acc starts +0.0 so result is +0.0 whatever x sign.
        let w = vec![0u8; 8 * GROUP_BYTES];
        let x = vec![-0.0f32; 2048];
        assert_eq!(lane_proj_ref(&[&w], &[1], &x, 1)[0][0], 0);
        // scale = +inf header with nibble 0: inf*0 = NaN in fma -> NaN propagates to output
        let mut w2 = vec![0u8; 8 * GROUP_BYTES];
        w2[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        let x1 = vec![1.0f32; 2048];
        assert!(f32::from_bits(lane_proj_ref(&[&w2], &[1], &x1, 1)[0][0]).is_nan());
        // NaN activation in lane 31's element -> NaN
        let mut xn = vec![0f32; 2048];
        xn[7 * 256 + 31 * 8 + 7] = f32::NAN;
        assert!(f32::from_bits(lane_proj_ref(&[&w], &[1], &xn, 1)[0][0]).is_nan());
        // +inf and -inf contributions from different lanes cancel to NaN
        let mut wi = vec![0u8; 8 * GROUP_BYTES];
        put_group(&mut wi, 0, [h(1.0), h(1.0)], [h(0.0), h(0.0)], |_, _| 1);
        let mut xi = vec![0f32; 2048];
        xi[0] = f32::INFINITY;
        xi[8] = f32::NEG_INFINITY;
        assert!(f32::from_bits(lane_proj_ref(&[&wi], &[1], &xi, 1)[0][0]).is_nan());
    }

    #[test]
    fn lane_proj_segments_and_layout() {
        // two segments with different row counts; out[s][b*m+row]
        let mut w0 = vec![0u8; 2 * 8 * GROUP_BYTES];
        let mut w1 = vec![0u8; 8 * GROUP_BYTES];
        for r in 0..2 {
            for g in 0..8 {
                put_group(
                    &mut w0,
                    (r * 8 + g) * GROUP_BYTES,
                    [h(1.0), h(1.0)],
                    [h(0.0), h(0.0)],
                    |_, _| 1,
                );
            }
        }
        for g in 0..8 {
            put_group(
                &mut w1,
                g * GROUP_BYTES,
                [h(2.0), h(2.0)],
                [h(0.0), h(0.0)],
                |_, _| 1,
            );
        }
        // u = 1 (seg0) / 2 (seg1) everywhere; x[b] = b+1 everywhere -> dot = 2048*(b+1)*u
        let mut x = vec![0f32; 3 * 2048];
        for b in 0..3 {
            x[b * 2048..(b + 1) * 2048].fill((b + 1) as f32);
        }
        let o = lane_proj_ref(&[&w0, &w1], &[2, 1], &x, 3);
        assert_eq!(o[0].len(), 6);
        for b in 0..3 {
            for r in 0..2 {
                assert_eq!(f32::from_bits(o[0][b * 2 + r]), 2048.0 * (b + 1) as f32);
            }
            assert_eq!(f32::from_bits(o[1][b]), 4096.0 * (b + 1) as f32);
        }
    }

    #[test]
    fn lane_proj_group_and_tree_order_is_visible() {
        // Non-associative data: single lane-0 activations (zero elsewhere) in groups 0 and 4 and 1;
        // acc[0] = 0 + g0, += g4 ; acc[1] = g1 ; result (acc0+acc1)+(acc2+acc3).
        let mut w = vec![0u8; 8 * GROUP_BYTES];
        for g in 0..8 {
            put_group(
                &mut w,
                g * GROUP_BYTES,
                [h(1.0), h(1.0)],
                [h(0.0), h(0.0)],
                |l, k| if l == 0 && k == 0 { 1 } else { 0 },
            );
        }
        // lane 0, k=0 -> u0 = 1; all other u = 0 (nibble 0: fma(1,0,0)=0).
        let mut x = vec![0f32; 2048];
        x[0] = 16777216.0; // group 0
        x[256] = 1.0; // group 1
        x[4 * 256] = 1.0; // group 4
                          // acc0 = (0 + 2^24) + 1 = 2^24 (tie rounds to even); acc1 = 1 ; (acc0+acc1) = 2^24 + 1 -> 2^24
        let got = f32::from_bits(lane_proj_ref(&[&w], &[1], &x, 1)[0][0]);
        assert_eq!(got, 16777216.0);
        // had it been acc0 = 2^24 + 1 + 1 summed as (1+1) first we would have 2^24+2
        assert_ne!(got, 16777218.0);
    }

    #[test]
    fn residual_hand_case_and_parity() {
        let m = 2;
        let mut w = vec![0u8; m * 16 * GROUP_BYTES];
        for r in 0..m {
            for g in 0..16 {
                put_group(
                    &mut w,
                    (r * 16 + g) * GROUP_BYTES,
                    [h(1.0), h(1.0)],
                    [h(0.0), h(0.0)],
                    |_, _| 3,
                );
            }
        }
        // u = 3 everywhere; x[b] = 0.5*(b+1); dot = 4096*3*0.5*(b+1)
        let lanes = 2;
        let mut x = vec![0f32; lanes * 4096];
        for b in 0..lanes {
            x[b * 4096..(b + 1) * 4096].fill(0.5 * (b + 1) as f32);
        }
        let mut y = vec![10.0f32; lanes * m];
        residual_lanes_ref(&w, &x, &mut y, m, lanes);
        for b in 0..lanes {
            for r in 0..m {
                assert_eq!(y[b * m + r], 10.0 + 6144.0 * (b + 1) as f32);
            }
        }
    }

    #[test]
    fn residual_row_parity_contraction_forms_differ() {
        // Group 0, lane 0: w0 = w1 = 1 (u = sc*q + zp with sc=1,zp=0,q=1), x0 = 1 + 2^-23 *..: choose so
        // that fma(w1,x1,w0*x0) != fma(w0,x0,w1*x1): w0*x0 rounds, the other product exact.
        let big = 1.0 + 2f32.powi(-12);
        let mut w = vec![0u8; 2 * 16 * GROUP_BYTES];
        for r in 0..2 {
            for g in 0..16 {
                put_group(
                    &mut w,
                    (r * 16 + g) * GROUP_BYTES,
                    [h(1.0), h(1.0)],
                    [h(0.0), h(0.0)],
                    |l, k| {
                        if l == 0 && g == 0 && k < 2 {
                            1
                        } else {
                            0
                        }
                    },
                );
            }
        }
        // w0 = w1 = 1 -> w0*x0 exact; need rounding: use header scale 1+2^-10 so w = 1.0009765625
        for r in 0..2 {
            let gb = r * 16 * GROUP_BYTES;
            w[gb..gb + 2].copy_from_slice(&0x3C01u16.to_le_bytes()); // sc0 = 1 + 2^-10
        }
        let mut x = vec![0f32; 4096];
        x[0] = big;
        x[1] = -big * (1.0 + 2f32.powi(-10)) * (1.0 + 2f32.powi(-10)) + 2f32.powi(-20);
        let mut y = vec![0f32; 2];
        // lanes=1: y[row] for rows 0 (even form) and 1 (odd form) from identical data must differ here
        // only if the forms differ; compute both forms by hand.
        let w0 = 1.0f32 + 2f32.powi(-10);
        let even = w0.mul_add(x[1], w0 * x[0]);
        let odd = w0.mul_add(x[0], w0 * x[1]);
        residual_lanes_ref(&w, &x, &mut y, 2, 1);
        // both rows share bytes: group sums are `even` and `odd`, combined with exact zeros elsewhere
        // (acc0 starts +0.0, so +0.0 + t == t unless t == -0.0)
        assert_eq!(y[0].to_bits(), (0.0f32 + even).to_bits());
        assert_eq!(y[1].to_bits(), (0.0f32 + odd).to_bits());
    }

    #[test]
    fn residual_residual_add_and_nan() {
        let m = 1;
        let w = vec![0u8; 16 * GROUP_BYTES];
        let x = vec![0f32; 4096];
        let mut y = vec![-0.0f32];
        residual_lanes_ref(&w, &x, &mut y, m, 1);
        // y + (+0.0) = +0.0 : -0.0 + +0.0 = +0.0
        assert_eq!(y[0].to_bits(), 0);
        let mut yn = vec![f32::NAN];
        residual_lanes_ref(&w, &x, &mut yn, m, 1);
        assert!(yn[0].is_nan());
        let mut xn = vec![0f32; 4096];
        xn[15 * 256 + 5] = f32::INFINITY;
        let mut wn = vec![0u8; 16 * GROUP_BYTES];
        put_group(
            &mut wn,
            15 * GROUP_BYTES,
            [h(0.0), h(0.0)],
            [h(0.0), h(0.0)],
            |_, _| 0,
        );
        let mut y2 = vec![1.0f32];
        residual_lanes_ref(&wn, &xn, &mut y2, 1, 1);
        assert!(y2[0].is_nan(), "0 * inf = NaN must reach y");
    }

    #[test]
    fn compare_words_policy() {
        let nan_a = 0x7FC0_0000u32;
        let nan_b = 0xFFC0_0001u32;
        let c = compare_words(&[1, nan_a, 5, nan_a], &[1, nan_b, 5, 7]);
        assert_eq!(c.mismatches, 1);
        assert_eq!(c.nan_payload_diffs, 1);
        assert!(!c.ok());
        assert_eq!(
            first_mismatch(&[1, nan_a, 5, nan_a], &[1, nan_b, 5, 7]),
            Some(3)
        );
    }
}
