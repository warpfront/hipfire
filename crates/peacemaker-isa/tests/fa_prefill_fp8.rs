// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 causal native-fp8 FA2-GQA PeaceMaker twin `fa_prefill_fp8_pm_gfx1201` (H16 / KV2 / D256): ONE export,
//! the direct attention `attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201` (`n_splits = 1`; the Q preconvert is the Q8 module's
//! and is covered by `fa_prefill.rs`). The K / V operands are 516-byte native token rows
//! `[512 head-major OCP E4M3FN code bytes][2 f16 scales]`: code byte `key * 516 + head * 256 + d`, scale
//! `key * 516 + 512 + head * 2`. Both caches are separately and EXACTLY sized (any key read at or past the last
//! live position is an unmapped fault), and the real emitted code runs on the peacemaker CPU interpreter.
//!
//! The reference here is independent of the kernel and of the Q8 test: its own E4M3FN decoder (bias 7, subnormals
//! `m * 2^-9`, signed zero `0x80`, NaN exactly `S.1111.111`, no FNUZ scale fold) and its own staged element
//! `f16_rne(f16_to_f32(scale) * e4m3fn(code))` (the f32 product is exact: <= 4 + 11 significand bits). Attention
//! references are exact by construction, not tolerance-based: scores are separated by far more than the f32 `exp`
//! underflow distance (probabilities are exactly 0 or 1) or tie exactly (scale 0), and V operands are chosen so every
//! tied sum is an exactly representable f32 value. NaN codes (`0x7f` / `0xff`) are classified separately from the
//! finite byte gate. The interpreter is a CPU model of the WMMA / exp / reciprocal arithmetic: it does not replace
//! the GPU byte oracle (`fa_pm_fp8_oracle`, PM vs the production hipcc twin on the same runtime-written bytes).
//!
//! `fa_prefill.rs` additionally runs a shared-codec gate through its original guarded interpreter harness and emitted
//! Q preconvert. This suite keeps the independent full-256 decoder and the FP8-specific cases; neither imports the
//! other's test functions, so Cargo runs each case only once.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::fa_prefill::{self, FaKvCodec, FaPrefillSpec};
use peacemaker_isa::{native, Arch};

const ARCH: Arch = Arch::Gfx1201;
const SYMBOL: &str = "attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201";
const MODULE: &str = "fa_prefill_fp8_pm_gfx1201";
const DIRECT_KERNARG: u32 = 60;
const DIRECT_BLOCK: u32 = 160;
const DIRECT_LDS: u32 = 32_768;
/// Bytes per position of one K or V allocation: 2 KV heads x (256 codes) + 2 f16 scales.
const ROW: usize = 516;
const SCALE_OFF: usize = 512;

fn spec() -> FaPrefillSpec {
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv: FaKvCodec::Fp8E4M3,
    }
}

// ------------------------------------------------------------------------------------------------ f16 / E4M3FN model
/// Independent f32 -> f16 round-to-nearest-even (finite inputs; overflow rounds to infinity).
fn f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = u64::from(m | 0x80_0000);
        let shift = (14 - exp) as u32;
        let mut q = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((exp as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}
fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            let (mut e2, mut m2) = (113u32, m);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
/// OCP E4M3FN: `S.EEEE.MMM`, bias 7, `E = 0` subnormal `M * 2^-9`, `E = 15, M = 7` NaN (`None`), no infinity.
fn e4m3fn(code: u8) -> Option<f32> {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((code >> 3) & 0xf);
    let m = f32::from(code & 7);
    if e == 15 && code & 7 == 7 {
        return None;
    }
    Some(
        sign * if e == 0 {
            m * 2f32.powi(-9)
        } else {
            (1.0 + m / 8.0) * 2f32.powi(e - 7)
        },
    )
}
fn finite_codes() -> Vec<u8> {
    (0..=255u8).filter(|&c| e4m3fn(c).is_some()).collect()
}
/// Finite codes whose value is a multiple of 1/8 (or zero): V operands whose sums are exact in f32 for any tie count.
fn exact_v_codes() -> Vec<u8> {
    finite_codes()
        .into_iter()
        .filter(|&c| {
            let v = e4m3fn(c).unwrap();
            v == 0.0 || (v.abs() >= 0.125 && (v * 8.0).fract() == 0.0)
        })
        .collect()
}
#[test]
fn the_e4m3fn_model_is_pinned_by_known_bit_patterns() {
    for (code, value) in [
        (0x00u8, 0.0f32),
        (0x01, 2f32.powi(-9)),
        (0x07, 7.0 * 2f32.powi(-9)),
        (0x08, 2f32.powi(-6)),
        (0x30, 0.5),
        (0x38, 1.0),
        (0x3c, 1.5),
        (0x40, 2.0),
        (0x68, 64.0),
        (0x77, 240.0),
        (0x78, 256.0),
        (0x7e, 448.0),
        (0xb8, -1.0),
        (0xfe, -448.0),
    ] {
        assert_eq!(e4m3fn(code), Some(value), "{code:#04x}");
    }
    assert_eq!(
        e4m3fn(0x80).map(f32::to_bits),
        Some(0x8000_0000),
        "0x80 is OCP negative zero, never a NaN"
    );
    assert_eq!(e4m3fn(0x7f), None);
    assert_eq!(e4m3fn(0xff), None);
    assert_eq!(
        finite_codes().len(),
        254,
        "exactly the two S.1111.111 codes are NaN"
    );
    // Strictly increasing magnitudes over 0x00..=0x7e, mirrored for the negative half.
    for c in 0u8..0x7e {
        assert!(e4m3fn(c).unwrap() < e4m3fn(c + 1).unwrap(), "{c:#04x}");
        assert_eq!(e4m3fn(c | 0x80).unwrap(), -e4m3fn(c).unwrap(), "{c:#04x}");
    }
    // Every finite value is exactly representable in f16 (so the staged product only rounds once).
    for c in finite_codes() {
        let v = e4m3fn(c).unwrap();
        assert_eq!(f16_to_f32(f16_rne(v)), v, "{c:#04x}");
    }
    assert!(
        exact_v_codes().len() > 100,
        "enough exact V codes for a full-range operand"
    );
    for (x, bits) in [
        (1.0f32, 0x3c00u16),
        (65504.0, 0x7bff),
        (65520.0, 0x7c00),
        (f32::from_bits(0x3380_0000), 0x0001),
        (f32::from_bits(0x3300_0000), 0x0000),
    ] {
        assert_eq!(f16_rne(x), bits, "{x:e}");
    }
}

#[test]
fn all_256_codes_match_an_independent_f32_bit_construction() {
    // Unlike the arithmetic decoder above, construct IEEE f32 words directly. In E=0, the nonzero mantissa's
    // leading bit gives its exponent; normal FP8 exponents and mantissas map straight into the f32 fields.
    for code in 0..=255u8 {
        let sign = u32::from(code & 0x80) << 24;
        let exponent = u32::from((code >> 3) & 15);
        let mantissa = u32::from(code & 7);
        let want = if exponent == 15 && mantissa == 7 {
            None
        } else if exponent == 0 && mantissa == 0 {
            Some(sign)
        } else if exponent == 0 {
            let leading = 31 - mantissa.leading_zeros();
            Some(
                sign | ((127 - 9 + leading) << 23)
                    | ((mantissa - (1 << leading)) << (23 - leading)),
            )
        } else {
            Some(sign | ((exponent + 120) << 23) | (mantissa << 20))
        };
        assert_eq!(
            e4m3fn(code).map(f32::to_bits),
            want,
            "OCP E4M3FN byte {code:#04x}"
        );
    }
}

// ------------------------------------------------------------------------------------------------ cache model
/// The staged f16 element of key `pos`, KV head `g`, dim `d`: f32 product of the f16 scale and the decoded code,
/// rounded once to f16. `None` for a NaN code.
fn stage(cache: &[u8], pos: usize, g: usize, d: usize) -> Option<u16> {
    let scale_at = pos * ROW + SCALE_OFF + 2 * g;
    let scale = u16::from_le_bytes([cache[scale_at], cache[scale_at + 1]]);
    e4m3fn(cache[pos * ROW + g * 256 + d]).map(|v| f16_rne(f16_to_f32(scale) * v))
}
fn build_cache(
    npos: usize,
    code: impl Fn(usize, usize, usize) -> u8,
    scale: impl Fn(usize, usize) -> u16,
) -> Vec<u8> {
    let mut out = vec![0u8; npos * ROW];
    for p in 0..npos {
        for g in 0..2 {
            let at = p * ROW + SCALE_OFF + 2 * g;
            out[at..at + 2].copy_from_slice(&scale(p, g).to_le_bytes());
            for d in 0..256 {
                out[p * ROW + g * 256 + d] = code(p, g, d);
            }
        }
    }
    out
}
fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
#[test]
fn row_layout_is_head_major_codes_then_two_scales() {
    // Every byte of a row has exactly one role: 512 codes (head 0 then head 1), then scale 0, scale 1.
    let npos = 3;
    let cache = build_cache(
        npos,
        |p, g, d| ((p * 7 + g * 131 + d) % 253) as u8,
        |p, g| 0x3c00 + (p * 2 + g) as u16,
    );
    assert_eq!(cache.len(), npos * ROW);
    assert_eq!(ROW, 2 * 256 + 2 * 2);
    for p in 0..npos {
        for g in 0..2 {
            for d in 0..256 {
                assert_eq!(
                    cache[p * 516 + g * 256 + d],
                    ((p * 7 + g * 131 + d) % 253) as u8,
                    "code p{p} g{g} d{d}"
                );
            }
            let s =
                u16::from_le_bytes([cache[p * 516 + 512 + 2 * g], cache[p * 516 + 513 + 2 * g]]);
            assert_eq!(s, 0x3c00 + (p * 2 + g) as u16, "scale p{p} g{g}");
        }
    }
    // The staged element of head 1 reads head 1's scale at +514, never head 0's at +512.
    let c2 = build_cache(
        1,
        |_, _, _| 0x38,
        |_, g| if g == 0 { 0x3c00 } else { 0x4000 },
    );
    assert_eq!(stage(&c2, 0, 0, 7), Some(0x3c00));
    assert_eq!(stage(&c2, 0, 1, 7), Some(0x4000));
}

// ------------------------------------------------------------------------------------------------ attention case
/// A direct-attention case: Q16 `[batch, 16, 256]`, native-fp8 K / V (516-byte rows), per-row positions, `scale_attn`.
#[derive(Clone)]
struct Case {
    batch: usize,
    positions: Vec<i32>,
    q16: Vec<u16>,
    k: Vec<u8>,
    v: Vec<u8>,
    scale: f32,
}
/// Scores are `scale * dot` with large separations between distinct keys (K scales ramp ascending for KV head 0 and
/// descending for KV head 1; the head-1 code plane has opposite signs, so exchanging code planes as well as scale
/// slots changes the winners). Q sign varies per row / head. V are exact-sum codes at scale 1.0. `salt` changes the
/// operands. With `scale == 0` every live key ties.
fn winner_case(batch: usize, positions: Vec<i32>, scale: f32, qmag: f32, salt: usize) -> Case {
    assert_eq!(positions.len(), batch);
    let npos = *positions.iter().max().unwrap() as usize + 1;
    let q16 = (0..batch * 4096)
        .map(|i| {
            let (r, h, d) = (i / 4096, (i / 256) % 16, i % 256);
            let sign = if (r * 5 + h * 3 + salt) % 4 < 2 {
                1.0
            } else {
                -1.0
            };
            f16_rne(sign * qmag * (1 + d % 3) as f32)
        })
        .collect();
    let k = build_cache(
        npos,
        |_, g, d| {
            let code = match d % 61 {
                0 => 0x7e,
                30 => 0xfe,
                _ => 0x68 + ((d * 37 + salt) % 4) as u8,
            };
            code ^ if g == 0 { 0 } else { 0x80 }
        },
        |p, g| {
            f16_rne(if g == 0 {
                0.25 * (1 + p) as f32
            } else {
                0.25 * (200 - p) as f32
            })
        },
    );
    let vcodes = exact_v_codes();
    let v = build_cache(
        npos,
        |p, g, d| vcodes[(p * 29 + d * 7 + g * 61 + salt * 13) % vcodes.len()],
        |_, _| 0x3c00,
    );
    Case {
        batch,
        positions,
        q16,
        k,
        v,
        scale,
    }
}
/// The exact expected output words. Every case must be well posed: all live keys either tie the row's maximum score
/// exactly or sit at least 200 below it (so `exp` underflows to exactly 0 and the online rescale `alpha` is exactly 0),
/// and the tied V sums are exact in f32.
fn expected(c: &Case) -> Vec<u32> {
    let npos = c.k.len() / ROW;
    assert_eq!(c.v.len(), c.k.len());
    let kst: Vec<Vec<u16>> = (0..2 * npos)
        .map(|i| {
            (0..256)
                .map(|d| stage(&c.k, i % npos, i / npos, d).expect("finite K"))
                .collect()
        })
        .collect();
    let vst: Vec<Vec<f32>> = (0..2 * npos)
        .map(|i| {
            (0..256)
                .map(|d| f16_to_f32(stage(&c.v, i % npos, i / npos, d).expect("finite V")))
                .collect()
        })
        .collect();
    let mut out = vec![0u32; c.batch * 4096];
    for r in 0..c.batch {
        let pos = c.positions[r] as usize;
        for h in 0..16 {
            let g = h / 8;
            let q: Vec<f64> = (0..256)
                .map(|d| f64::from(f16_to_f32(c.q16[(r * 16 + h) * 256 + d])))
                .collect();
            let scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let dot: f64 = (0..256)
                        .map(|d| q[d] * f64::from(f16_to_f32(kst[g * npos + j][d])))
                        .sum();
                    (dot as f32) * c.scale
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ties: Vec<usize> = (0..=pos).filter(|&j| scores[j] == m).collect();
            for j in 0..=pos {
                assert!(
                    scores[j] == m || m - scores[j] >= 200.0,
                    "ill-posed case: row {r} head {h} key {j} gap {}",
                    m - scores[j]
                );
            }
            let inv = 1.0f32 / ties.len() as f32;
            for d in 0..256 {
                let (mut sum, mut exact) = (0.0f32, 0.0f64);
                for &j in &ties {
                    let v = vst[g * npos + j][d];
                    sum += v;
                    exact += f64::from(v);
                }
                assert_eq!(
                    f64::from(sum),
                    exact,
                    "inexact V sum: row {r} head {h} dim {d}"
                );
                out[(r * 16 + h) * 256 + d] = (sum * inv).to_bits();
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ emulator rig
const E_KARG: u64 = 0x1_0000;
const E_Q: u64 = 0x1000_0000;
const E_K: u64 = 0x2000_0000;
const E_V: u64 = 0x3000_0000;
const E_OUT: u64 = 0x4000_0000;
const E_POS: u64 = 0x5000_0000;
const GUARD: usize = 256;
const SENTINEL: u8 = 0xa5;
const POISON: u32 = 0x7fc0_beef;

struct Rig {
    program: peacemaker_ir::Program,
}
impl Rig {
    fn new() -> Self {
        let (_, text, _) = fa_prefill::emit_fp8_module(ARCH, &spec()).unwrap();
        let elf = native::assemble(&text, ARCH).unwrap();
        let lifted = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap();
        Rig {
            program: lifted.program,
        }
    }
}
fn grid(batch: usize) -> Vec<[u32; 3]> {
    (0..2u32)
        .flat_map(|y| (0..batch.div_ceil(8) as u32).map(move |x| [x, y, 0]))
        .collect()
}
fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions.iter().find(|r| r.name == name).unwrap()
}
struct Run {
    out: Vec<u8>,
    guards_intact: bool,
    out_written: usize,
    inputs_untouched: bool,
}
enum Initial {
    Poison(u32),
    Bytes(Vec<u8>),
}
/// One launch set on freshly mapped, exactly sized buffers: kernarg exactly 60 bytes, Q16 / K / V / positions exactly
/// their extents (any out-of-range read is an unmapped fault), the output between sentinel guards. `args` overrides
/// `[n_heads, n_kv_heads, head_dim, batch_size]` in the kernarg block.
fn run_direct(
    rig: &Rig,
    c: &Case,
    initial: &Initial,
    args: Option<[i32; 4]>,
    wgs: &[[u32; 3]],
) -> Result<Run, String> {
    let out_bytes = c.batch * 4096 * 4;
    let [nh, nkv, hd, bs] = args.unwrap_or([16, 2, 256, c.batch as i32]);
    let mut karg = Vec::new();
    for ptr in [E_Q, E_K, E_V, E_OUT + GUARD as u64, E_POS] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [nh, nkv, hd, bs] {
        karg.extend(v.to_le_bytes());
    }
    karg.extend(c.scale.to_le_bytes());
    assert_eq!(karg.len(), DIRECT_KERNARG as usize);
    let (q, pos) = (le16(&c.q16), le32(&c.positions));
    let mut out = vec![SENTINEL; out_bytes + 2 * GUARD];
    match initial {
        Initial::Poison(word) => {
            for w in out[GUARD..GUARD + out_bytes].chunks_exact_mut(4) {
                w.copy_from_slice(&word.to_le_bytes());
            }
        }
        Initial::Bytes(b) => {
            assert_eq!(b.len(), out_bytes);
            out[GUARD..GUARD + out_bytes].copy_from_slice(b);
        }
    }
    let before = out.clone();
    let mut mem = Memory::default();
    mem.map("kernarg".into(), E_KARG, karg.clone())?;
    mem.map("q16".into(), E_Q, q.clone())?;
    mem.map("k".into(), E_K, c.k.clone())?;
    mem.map("v".into(), E_V, c.v.clone())?;
    mem.map("pos".into(), E_POS, pos.clone())?;
    mem.map("out".into(), E_OUT, out)?;
    let emu = Emulator::new(&rig.program, SYMBOL)?;
    for wg in wgs {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KARG,
                workgroup: *wg,
                block: [DIRECT_BLOCK, 1, 1],
                dynamic_lds: DIRECT_LDS,
            },
            &mut mem,
        )?;
    }
    let inputs_untouched = [
        ("kernarg", &karg),
        ("q16", &q),
        ("k", &c.k),
        ("v", &c.v),
        ("pos", &pos),
    ]
    .iter()
    .all(|(name, bytes)| {
        let r = region(&mem, name);
        r.bytes == **bytes && r.written.count() == 0
    });
    let o = region(&mem, "out");
    Ok(Run {
        out: o.bytes[GUARD..GUARD + out_bytes].to_vec(),
        guards_intact: o.bytes[..GUARD] == before[..GUARD]
            && o.bytes[GUARD + out_bytes..] == before[GUARD + out_bytes..],
        out_written: o.written.count(),
        inputs_untouched,
    })
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
fn assert_matches(name: &str, run: &Run, want: &[u32], batch: usize) {
    assert!(
        run.guards_intact,
        "{name}: a store escaped the output (tail-row lane or helper wave wrote)"
    );
    assert!(
        run.inputs_untouched,
        "{name}: Q16 / K / V / positions / kernarg changed"
    );
    assert_eq!(
        run.out_written,
        batch * 4096 * 4,
        "{name}: exactly the live output bytes are written"
    );
    assert_eq!(run.out.len(), batch * 4096 * 4, "{name}: output extent");
    assert_eq!(want.len(), batch * 4096, "{name}: reference extent");
    for (i, (g, w)) in words(&run.out).iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{name}: row {} head {} dim {}: got {:#010x} ({}), want {:#010x} ({})",
            i / 4096,
            (i / 256) % 16,
            i % 256,
            g,
            f32::from_bits(*g),
            w,
            f32::from_bits(*w)
        );
    }
}
fn check(rig: &Rig, name: &str, c: &Case) {
    let run = run_direct(rig, c, &Initial::Poison(POISON), None, &grid(c.batch))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_matches(name, &run, &expected(c), c.batch);
}

// ------------------------------------------------------------------------------------------------ 1. emission
fn mnemonics(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| l.trim().split_whitespace().next())
        .collect()
}
#[test]
fn emission_is_deterministic_and_exports_only_the_fp8_direct() {
    let (emitted, text, _) = fa_prefill::emit_fp8_module(ARCH, &spec()).unwrap();
    let (_, again, _) = fa_prefill::emit_fp8_module(ARCH, &spec()).unwrap();
    assert!(text == again, "fp8 emission is not deterministic");
    assert_eq!(emitted.len(), 1, "the fp8 module holds exactly one export");
    assert_eq!(fa_prefill::MODULE_FP8, MODULE);
    assert_eq!(fa_prefill::SYMBOL_FP8_DIRECT, SYMBOL);
    assert_eq!(fa_prefill::FP8_ROW_BYTES as usize, ROW);
    assert_eq!(fa_prefill::FP8_SCALE_OFFSET as usize, SCALE_OFF);
    assert!(text.contains(SYMBOL));
    assert!(
        !text.contains(fa_prefill::SYMBOL_PRECONVERT),
        "the Q16 preconvert stays in the Q8 module"
    );
    assert!(
        !text.contains(fa_prefill::SYMBOL_DIRECT),
        "no Q8 direct export in the fp8 module"
    );
    // The module holds only the fp8 direct: lifting it yields exactly that kernel.
    let rig = Rig::new();
    let symbols: Vec<&str> = rig
        .program
        .kernels
        .iter()
        .map(|k| k.symbol.0.as_str())
        .collect();
    assert_eq!(symbols, [SYMBOL]);
    // Native gfx1201 OCP decode: both pair-convert forms, none of the Q8 sign extension or any other fp8 route.
    let mn = mnemonics(&text);
    assert!(
        mn.iter().any(|m| *m == "v_cvt_pk_f32_fp8_e32"),
        "no low-half native pair convert"
    );
    assert!(
        mn.iter().any(|m| *m == "v_cvt_pk_f32_fp8_e64"),
        "no high-half (op_sel) native pair convert"
    );
    for forbidden in [
        "v_cvt_f32_i32",
        "v_cvt_f32_fp8",
        "v_cvt_f32_bf8",
        "v_cvt_pk_f32_bf8",
        "v_cvt_pk_fp8_f32",
        "v_cvt_pk_bf8_f32",
        "v_cvt_sr_fp8_f32",
        "v_cvt_sr_bf8_f32",
    ] {
        assert!(
            !mn.iter().any(|m| m.starts_with(forbidden)),
            "{forbidden} appears in the fp8 direct"
        );
    }
    // Rows are 516 bytes: the tile stride literal is present, the Q8 544 / 1088 strides are not the fill base.
    assert!(
        text.contains("516") || text.contains("0x204"),
        "the 516-byte fp8 row stride does not appear in the emission"
    );
}

#[test]
fn fp8_direct_resources_match_the_single_stage_layout() {
    // One 32-KiB LDS stage reused sequentially for K then V: four tile
    // rendezvous sites (K publish, K retire, V publish, V retire), VGPR
    // ceiling 224, zero spills/private, unchanged 60-byte ABI and block 160.
    let (emitted, text, _) = fa_prefill::emit_fp8_module(ARCH, &spec()).unwrap();
    assert_eq!(emitted.len(), 1, "the fp8 module holds exactly one export");
    let e = &emitted[0];
    let count = |name: &str| {
        text.lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count()
    };
    assert_eq!(
        (count("s_barrier_signal"), count("s_barrier_wait")),
        (4, 4),
        "fp8 direct: four tile rendezvous sites (K publish/retire, V publish/retire)"
    );
    assert_eq!(e.proof.barriers.len(), 4, "fp8 direct: proof barrier sites");
    // Single LDS stage, base 0 length 32768: the ten sequential-stage labels
    // are all present, in K-ready / K-retire / V-ready / V-retire order.
    let labels = [
        ".Lfapd_tile",
        ".Lfapd_kfill_done",
        ".Lfapd_kready",
        ".Lfapd_qk",
        ".Lfapd_qk_done",
        ".Lfapd_vfill_done",
        ".Lfapd_vready",
        ".Lfapd_turn",
        ".Lfapd_epi",
        ".Lfapd_end",
    ];
    let mut cursor = 0;
    for label in labels {
        let at = text[cursor..]
            .find(label)
            .unwrap_or_else(|| panic!("fp8 direct: missing label {label}"));
        cursor += at + label.len();
    }
    let field = |key: &str| -> Vec<u32> {
        text.lines()
            .filter_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().parse().unwrap())
            .collect()
    };
    let vgprs = field(".amdhsa_next_free_vgpr");
    assert_eq!(vgprs.len(), 1, "{SYMBOL}");
    assert_eq!(vgprs[0], 224, "{SYMBOL}: O96..223 packs to next_free 224");
    assert_eq!(
        u32::from(e.shape.next_free_vgpr),
        vgprs[0],
        "{SYMBOL}: shape/descriptor VGPRs"
    );
    assert!(
        text.contains(&format!(".vgpr_count: {}\n", vgprs[0])),
        "{SYMBOL}: metadata VGPRs"
    );
    let sgprs = field(".amdhsa_next_free_sgpr");
    assert_eq!(sgprs.len(), 1, "{SYMBOL}");
    assert_eq!(
        sgprs[0], 82,
        "{SYMBOL}: SGPR budget unchanged (84 in lifted metadata)"
    );
    assert!(
        sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0],
        "{SYMBOL}: {} SGPRs",
        sgprs[0]
    );
    // Unchanged arithmetic: the QK/softmax/PV opcode census is identical to
    // the Q8 direct body (only K/V fill addressing and decode differ).
    let q8 = fa_prefill::emit(
        ARCH,
        &FaPrefillSpec {
            kv: FaKvCodec::Q8_0,
            ..spec()
        },
    )
    .unwrap();
    // Prologue O init (persistent online state): all 128 FP8 accumulators
    // 96..223 zeroed before the tile loop, never the stale 224..255 window;
    // Q8 keeps 128..255. Scoped to the prologue so fill/reuse zeroes after
    // `.Lfapd_tile` cannot satisfy it.
    let prologue_zeroed = |src: &str| -> Vec<u32> {
        let head = src.split(".Lfapd_tile").next().unwrap_or("");
        head.lines()
            .filter_map(|l| {
                let t = l.split(';').next()?.split("//").next()?.trim();
                let (dst, rhs) = t.split_once(',')?;
                if !dst.starts_with("v_mov_b32_e32 v") {
                    return None;
                }
                if rhs.trim() != "0" {
                    return None;
                }
                dst.strip_prefix("v_mov_b32_e32 v")?.trim().parse().ok()
            })
            .collect()
    };
    let fp8_zeroed = prologue_zeroed(&text);
    for r in 96..224u32 {
        assert!(
            fp8_zeroed.contains(&r),
            "{SYMBOL}: prologue never zeroes O accumulator v{r}"
        );
    }
    assert_eq!(
        fp8_zeroed
            .iter()
            .filter(|r| (96..224u32).contains(r))
            .count(),
        128,
        "{SYMBOL}: prologue O96..223 init count"
    );
    for r in 224..256u32 {
        assert!(
            !fp8_zeroed.contains(&r),
            "{SYMBOL}: prologue zeroes stale v{r} past O96..223"
        );
    }
    let q8_zeroed = prologue_zeroed(&q8.s_text);
    for r in 128..256u32 {
        assert!(
            q8_zeroed.contains(&r),
            "Q8 direct: prologue never zeroes O accumulator v{r}"
        );
    }
    let fp8_contract = fa_prefill::fp8_contract(ARCH, e);
    let q8_contract = fa_prefill::contract(ARCH, &q8);
    for m in [
        "v_wmma_f32_16x16x16_f16",
        "v_fma_mixlo_f16",
        "v_exp_f32_e32",
        "v_max3_num_f32",
        "v_cvt_f16_f32_e32",
        "v_fmac_f32_e32",
        "v_div_scale_f32",
        "v_div_fmas_f32",
        "v_div_fixup_f32",
        "v_rcp_f32_e32",
        "v_mul_f32_e32",
    ] {
        assert_eq!(
            fp8_contract["counts"][m], q8_contract["counts"][m],
            "fp8 direct: arithmetic count {m} differs from the Q8 body"
        );
    }
    // Frozen exact conversion/WMMA census: 64 e32 + 64 e64 packed FP8 pair
    // converts (K 32/32, V 32/32), 256 mixlo dequant products, 128 f16 WMMA.
    assert_eq!(
        fp8_contract["counts"]["v_wmma_f32_16x16x16_f16"].as_u64(),
        Some(128),
        "fp8 direct: WMMA count"
    );
    assert_eq!(
        fp8_contract["counts"]["v_fma_mixlo_f16"].as_u64(),
        Some(256),
        "fp8 direct: mixlo count"
    );
    assert_eq!(
        fp8_contract["counts"]["v_cvt_pk_f32_fp8_e32"].as_u64(),
        Some(64),
        "fp8 direct: low-half pair converts (K 32 + V 32)"
    );
    assert_eq!(
        fp8_contract["counts"]["v_cvt_pk_f32_fp8_e64"].as_u64(),
        Some(64),
        "fp8 direct: high-half pair converts (K 32 + V 32)"
    );
    for (c, symbol, dynamic) in [(&fp8_contract, SYMBOL, DIRECT_LDS)] {
        assert_eq!(c["symbol"], symbol);
        assert_eq!(c["launch_dynamic_lds_bytes"], dynamic, "{symbol}");
        assert_eq!(c["group_segment_fixed_bytes"], 0, "{symbol}");
        assert_eq!(c["require_wave32"], true, "{symbol}");
        assert_eq!(c["require_zero_spills"], true, "{symbol}");
        assert_eq!(c["require_zero_private"], true, "{symbol}");
        assert_eq!(c["vgpr_max"].as_u64(), Some(224), "{symbol}");
        assert!(c["sgpr_max"].as_u64().unwrap() <= 104, "{symbol}");
    }
}
#[test]
fn spec_admits_only_h16_kv2_d256_for_the_fp8_codec() {
    assert!(spec().validate().is_ok());
    for (h, kv, d) in [
        (24, 4, 256),
        (16, 4, 256),
        (16, 2, 128),
        (8, 2, 256),
        (32, 2, 256),
        (16, 1, 256),
    ] {
        let bad = FaPrefillSpec {
            heads: h,
            kv_heads: kv,
            head_dim: d,
            kv: FaKvCodec::Fp8E4M3,
        };
        assert!(bad.validate().is_err(), "{h}/{kv}/{d}");
        assert!(
            fa_prefill::emit_fp8_module(ARCH, &bad).is_err(),
            "{h}/{kv}/{d}"
        );
    }
    assert!(
        fa_prefill::emit_fp8_module(Arch::Gfx1100, &spec()).is_err(),
        "gfx1201 only"
    );
    // The Q8 module refuses an fp8 spec and the fp8 module a Q8 spec: no cross-codec emission.
    assert!(fa_prefill::emit_module(ARCH, &spec()).is_err());
    let q8 = FaPrefillSpec {
        kv: FaKvCodec::Q8_0,
        ..spec()
    };
    assert!(fa_prefill::emit_fp8_module(ARCH, &q8).is_err());
    // Launch validation is codec independent: 1..=8192 rows and one position per row.
    for rows in [1usize, 7, 63, 64, 65, 8192] {
        assert!(fa_prefill::validate(&spec(), rows, rows).is_ok(), "{rows}");
    }
    for (rows, pos) in [(0usize, 0usize), (8193, 8193), (8, 7), (8, 9)] {
        assert!(
            fa_prefill::validate(&spec(), rows, pos).is_err(),
            "{rows}/{pos}"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 2. causal bounds
#[test]
fn causal_bounds_and_tail_masks_on_native_fp8_rows() {
    // Exactly sized caches (any key read at or past `max position + 1` faults), exact-size positions, the output
    // guarded, non-monotone and duplicated groups, every eight-row tail remainder and every subtile class.
    const B: [i32; 10] = [0, 15, 16, 31, 32, 47, 48, 63, 64, 65];
    let rig = Rig::new();
    let cases: Vec<(&str, Vec<i32>)> = vec![
        ("1 row pos 0", vec![0]),
        ("1 row pos 65", vec![65]),
        ("7 rows non-monotone", vec![31, 0, 15, 63, 16, 48, 47]),
        ("8 rows around tiles", vec![64, 0, 65, 32, 15, 16, 31, 48]),
        (
            "9 rows, short first group, lone tail row",
            vec![0, 1, 2, 3, 4, 5, 6, 7, 65],
        ),
        ("63 rows ramp", (0..63).collect()),
        ("64 rows strided", (0..64).map(|r| (r * 5) % 66).collect()),
        (
            "65 rows boundary cycle",
            (0..65).map(|r| B[(r * 7) % 10]).collect(),
        ),
        (
            "duplicate positions in a group",
            vec![9, 9, 9, 9, 2, 2, 2, 2],
        ),
        (
            "unaligned start, same group",
            vec![63, 63, 62, 62, 61, 61, 60, 60],
        ),
    ];
    let cases = cases
        .into_iter()
        .chain((2usize..=6).map(|batch| {
            (
                "remaining row-tail remainder",
                (0..batch).map(|r| B[(r * 7) % B.len()]).collect(),
            )
        }))
        .chain(std::iter::once((
            "third key tile",
            vec![127, 128, 129, 0, 64, 65, 15, 16],
        )));
    for (name, positions) in cases {
        let batch = positions.len();
        check(&rig, name, &winner_case(batch, positions, 1.0, 1.0, 0));
    }
}

// ------------------------------------------------------------------------------------------------ 3. state and layout
/// A single live key (every row position 0) whose V row is built from the finite-code table over the given per-head
/// scales; the output is that key's staged V per KV head. Only products that round to f16 infinity are replaced by
/// the code of 1.0; the emitted cache census below verifies every finite staged product is still present.
fn single_key_case(scales: [u16; 2], shift: usize) -> Case {
    let mut c = winner_case(8, vec![0; 8], 1.0, 1.0, 3 + shift);
    let table = finite_codes();
    let safe = |scale: u16, code: u8| {
        if f16_rne(f16_to_f32(scale) * e4m3fn(code).unwrap()) & 0x7c00 == 0x7c00 {
            0x38
        } else {
            code
        }
    };
    c.v = build_cache(
        1,
        |_, g, d| safe(scales[g], table[(d + 31 * shift + 128 * g) % table.len()]),
        |_, g| scales[g],
    );
    c
}
#[test]
fn single_live_key_covers_every_finite_staged_product_at_f16_scale_boundaries() {
    let rig = Rig::new();
    // Signed zeros, subnormal / normal boundaries, RNE-sensitive scales, large scales and negative scales. All 254
    // finite codes occur on each head whenever all products remain finite; otherwise every non-overflowing code does.
    const SCALES: [u16; 12] = [
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000, 0xbc01,
        0xfbff,
    ];
    for shift in 0..SCALES.len() {
        let scales = [SCALES[shift], SCALES[(shift + 3) % SCALES.len()]];
        let c = single_key_case(scales, shift);
        for g in 0..2 {
            let admitted: std::collections::BTreeSet<u8> = finite_codes()
                .into_iter()
                .filter(|&code| {
                    f16_rne(f16_to_f32(scales[g]) * e4m3fn(code).unwrap()) & 0x7c00 != 0x7c00
                })
                .collect();
            let seen: std::collections::BTreeSet<u8> =
                c.v[g * 256..(g + 1) * 256].iter().copied().collect();
            assert_eq!(
                seen, admitted,
                "head {g} scale {:#06x}: actual emitted operand code coverage",
                scales[g]
            );
        }
        check(
            &rig,
            &format!("single key scales {scales:#06x?} shift {shift}"),
            &c,
        );
    }
    // Distinct K / V scaling across the two heads of one key: swap the scale order, the outputs must follow the head.
    let a = single_key_case([0x3c00, 0x4000], 1);
    let b = single_key_case([0x4000, 0x3c00], 1);
    assert!(expected(&a) != expected(&b));
    check(&rig, "head scales a", &a);
    check(&rig, "head scales b", &b);
}
#[test]
fn the_k_decode_selects_the_winner_per_kv_head_with_separate_code_and_scale_bases() {
    // The scale ramps run in opposite directions and the code planes have opposite signs. Q signs flip the winner
    // per row / head. Swapping either the scale slots or the code planes must therefore change observable outputs.
    let rig = Rig::new();
    let positions: Vec<i32> = vec![40, 3, 17, 16, 5, 31, 32, 9];
    let base = winner_case(8, positions.clone(), 1.0, 1.0, 0);
    check(&rig, "k ramps", &base);
    let mut swapped = base.clone();
    let npos = swapped.k.len() / ROW;
    for p in 0..npos {
        let (a, b) = (p * ROW + SCALE_OFF, p * ROW + SCALE_OFF + 2);
        swapped.k.swap(a, b);
        swapped.k.swap(a + 1, b + 1);
    }
    assert!(
        expected(&base) != expected(&swapped),
        "the reference does not depend on which scale slot belongs to which head"
    );
    check(&rig, "k scale slots swapped", &swapped);
    // Moving the code planes instead (head 0 <-> head 1 codes) is equally observable.
    let mut codes_swapped = base.clone();
    for p in 0..npos {
        for d in 0..256 {
            codes_swapped.k.swap(p * ROW + d, p * ROW + 256 + d);
        }
    }
    assert!(expected(&base) != expected(&codes_swapped));
    check(&rig, "k code planes swapped", &codes_swapped);
}
#[test]
fn emitted_attention_preserves_state_and_layout() {
    let rig = Rig::new();
    // Zero-score / uniform probabilities: every live key ties exactly, probabilities are 1, O = exact V sums.
    check(
        &rig,
        "uniform scale +0",
        &winner_case(9, (0..9).collect(), 0.0, 1.0, 0),
    );
    check(
        &rig,
        "uniform scale -0",
        &winner_case(8, vec![64, 0, 63, 17, 5, 31, 32, 9], -0.0, 1.0, 1),
    );
    // Single live key (every row position 0): the output is that key's dequantized V.
    check(
        &rig,
        "single live key",
        &winner_case(8, vec![0; 8], 1.0, 1.0, 2),
    );
    // `scale_attn` is the argument, never a constant: the sign and magnitude select the winner of each row.
    let positions = vec![3, 0, 17, 16, 5, 31, 32, 9];
    for (scale, qmag) in [
        (1.0f32, 1.0f32),
        (-1.0, 1.0),
        (0.0625, 16.0),
        (-0.0625, 16.0),
        (8.0, 0.125),
        (1.0e20, 1.0),
        (-1.0e20, 1.0),
    ] {
        check(
            &rig,
            &format!("scale {scale:e}"),
            &winner_case(8, positions.clone(), scale, qmag, 0),
        );
    }
    // Replay on a previously used output: changed Q / K / V / positions, the previous result left in place, then a
    // NaN-poisoned output: all three agree byte for byte with the reference (no dependence on retained scratch).
    let first = winner_case(8, positions.clone(), 1.0, 1.0, 0);
    let second = winner_case(8, vec![32, 31, 16, 17, 0, 3, 9, 5], -1.0, 1.0, 5);
    assert!(first.q16 != second.q16 || first.k != second.k || first.v != second.v);
    let run1 = run_direct(&rig, &first, &Initial::Poison(POISON), None, &grid(8)).unwrap();
    assert_matches("replay, first launch", &run1, &expected(&first), 8);
    let want = expected(&second);
    let run2 = run_direct(
        &rig,
        &second,
        &Initial::Bytes(run1.out.clone()),
        None,
        &grid(8),
    )
    .unwrap();
    assert_matches("replay on the previous output", &run2, &want, 8);
    let run3 = run_direct(&rig, &second, &Initial::Poison(POISON), None, &grid(8)).unwrap();
    assert_matches("replay on a poisoned output", &run3, &want, 8);
    assert!(run2.out == run3.out && run2.out != run1.out);
    // Workgroup-uniform launch guards: a wrong compiled shape, an empty / negative batch, a KV head or row group past
    // the launch store nothing, read nothing past the exact buffers and leave every input untouched.
    for args in [
        [15, 2, 256, 8],
        [16, 1, 256, 8],
        [16, 3, 256, 8],
        [16, 2, 128, 8],
        [16, 2, 512, 8],
        [24, 4, 256, 8],
        [16, 2, 256, 0],
        [16, 2, 256, -1],
    ] {
        let run = run_direct(&rig, &first, &Initial::Poison(POISON), Some(args), &grid(8))
            .unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert!(
            run.guards_intact && run.inputs_untouched && run.out_written == 0,
            "{args:?}"
        );
        assert!(
            words(&run.out).iter().all(|&w| w == POISON),
            "{args:?}: output changed"
        );
    }
    for wg in [[1u32, 0, 0], [0, 2, 0], [7, 1, 0], [0, 255, 0]] {
        let run = run_direct(&rig, &first, &Initial::Poison(POISON), None, &[wg])
            .unwrap_or_else(|e| panic!("{wg:?}: {e}"));
        assert!(
            run.guards_intact && run.inputs_untouched && run.out_written == 0,
            "{wg:?}"
        );
        assert!(
            words(&run.out).iter().all(|&w| w == POISON),
            "{wg:?}: output changed"
        );
    }
}
#[test]
fn future_cache_bytes_never_reach_a_row_whose_horizon_precedes_them() {
    // Change V bytes inside the group's mapped horizon but above row 1's own position. Those keys affect other
    // rows, yet row 1 must still return exactly key 0. Exact allocation bounds independently catch over-horizon reads.
    let rig = Rig::new();
    let positions: Vec<i32> = vec![5, 0, 17, 9, 3, 12, 30, 1];
    let base = winner_case(8, positions.clone(), 1.0, 1.0, 4);
    let want = expected(&base);
    let npos = base.k.len() / ROW;
    let mut loud = base.clone();
    for p in 0..npos {
        for g in 0..2 {
            // Row 1 has position 0. Later keys have extreme but finite V at scale 1 and must not affect that row.
            if p > 0 {
                for d in 0..256 {
                    loud.v[p * ROW + g * 256 + d] = 0x7e;
                }
                loud.v[p * ROW + SCALE_OFF + 2 * g..p * ROW + SCALE_OFF + 2 * g + 2]
                    .copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
    }
    let run = run_direct(&rig, &loud, &Initial::Poison(POISON), None, &grid(8)).unwrap();
    assert_matches("loud future V", &run, &expected(&loud), loud.batch);
    let got = words(&run.out);
    // Row 1 has position 0: its output is exactly the unchanged key-0 V, regardless of the loud later keys.
    for h in 0..16 {
        for d in 0..256 {
            let i = (16 + h) * 256 + d;
            assert_eq!(
                got[i], want[i],
                "row 1 head {h} dim {d} saw a key above its horizon"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ 4. NaN classification
fn is_nan_word(w: u32) -> bool {
    f32::from_bits(w).is_nan()
}
#[test]
fn nan_codes_are_classified_separately_from_the_finite_gate() {
    let rig = Rig::new();
    // V NaN at a live key: the NaN lands exactly in that KV head's dim for all of its heads; every other output word
    // stays bit-exact with the finite reference (the NaN code is never an infinity or a finite stand-in).
    for (code, g, d0) in [
        (0x7fu8, 0usize, 5usize),
        (0xff, 1, 200),
        (0x7f, 1, 0),
        (0xff, 0, 255),
    ] {
        let mut c = winner_case(8, vec![0; 8], 1.0, 1.0, 7);
        let clean = expected(&c);
        c.v[g * 256 + d0] = code;
        let run = run_direct(&rig, &c, &Initial::Poison(POISON), None, &grid(8)).unwrap();
        assert!(
            run.guards_intact && run.inputs_untouched,
            "V NaN {code:#04x} g{g} d{d0}"
        );
        assert_eq!(
            run.out_written,
            c.batch * 4096 * 4,
            "V NaN: every output byte must be written"
        );
        let got = words(&run.out);
        for r in 0..8 {
            for h in 0..16 {
                for d in 0..256 {
                    let i = (r * 16 + h) * 256 + d;
                    if h / 8 == g && d == d0 {
                        assert!(
                            is_nan_word(got[i]),
                            "V {code:#04x} g{g}: row {r} head {h} dim {d} is not NaN ({:#010x})",
                            got[i]
                        );
                    } else {
                        assert_eq!(got[i], clean[i], "V NaN leaked: row {r} head {h} dim {d}");
                    }
                }
            }
        }
    }
    // K NaN at the single live key of KV head 1: that group's heads see a NaN score (a NaN somewhere in each of its
    // rows / heads), the other KV head's heads stay bit-exact.
    for code in [0x7fu8, 0xff] {
        let mut c = winner_case(8, vec![0; 8], 1.0, 1.0, 9);
        let clean = expected(&c);
        c.k[256 + 17] = code;
        let run = run_direct(&rig, &c, &Initial::Poison(POISON), None, &grid(8)).unwrap();
        assert!(
            run.guards_intact && run.inputs_untouched,
            "K NaN {code:#04x}"
        );
        assert_eq!(
            run.out_written,
            c.batch * 4096 * 4,
            "K NaN: every output byte must be written"
        );
        let got = words(&run.out);
        for r in 0..8 {
            for h in 0..16 {
                let row = &got[(r * 16 + h) * 256..(r * 16 + h + 1) * 256];
                if h / 8 == 1 {
                    assert!(
                        row.iter().any(|&w| is_nan_word(w)),
                        "K {code:#04x}: row {r} head {h} has no NaN"
                    );
                } else {
                    assert_eq!(
                        row,
                        &clean[(r * 16 + h) * 256..(r * 16 + h + 1) * 256],
                        "K NaN leaked into KV head 0: row {r} head {h}"
                    );
                }
            }
        }
    }
}
// ------------------------------------------------------------------------------------------------ 5. stage rendezvous
#[test]
fn fp8_single_stage_rendezvous_guards_all_five_waves_at_all_four_barriers() {
    // The FP8 direct reuses ONE 32-KiB stage sequentially: K-fill publishes, QK reads, the stage retires, V-fill
    // publishes, softmax/PV reads, the stage retires. All five waves (four compute + helper) join all four
    // barriers; the helper skips K-fill to the K-ready rendezvous and QK to `.Lfapd_qk_done`, V-fill to the V-ready
    // rendezvous and softmax/PV to `.Lfapd_turn`. This pins the proof slot role, the exact barrier transitions,
    // the label order and both helper branch targets using only the public emitted IR/text.
    let (emitted, text, _) = fa_prefill::emit_fp8_module(ARCH, &spec()).unwrap();
    assert_eq!(emitted.len(), 1, "the fp8 module holds exactly one export");
    let e = &emitted[0];
    // Proof slot role: one `stage` slot, id 0, base 0, len 32768.
    let slots = &e.proof.lds_slots;
    assert_eq!(slots.len(), 1, "fp8 direct: one LDS stage");
    assert_eq!(
        (slots[0].name.as_str(), slots[0].base, slots[0].len),
        ("stage", 0, 32_768),
        "fp8 direct: stage slot"
    );
    // Proof barrier lifecycle: Ready(0) publishes K, Retire(0) releases K, Ready(0) publishes V, Retire(0)
    // releases V.
    let transitions: Vec<Vec<String>> = e
        .proof
        .barriers
        .iter()
        .map(|b| b.transitions.clone())
        .collect();
    assert_eq!(
        transitions,
        vec![
            vec!["Ready(0)".to_string()],
            vec!["Retire(0)".to_string()],
            vec!["Ready(0)".to_string()],
            vec!["Retire(0)".to_string()],
        ],
        "fp8 direct: stage publish/retire lifecycle"
    );
    // Label order across the sequential schedule.
    let labels = [
        ".Lfapd_tile",
        ".Lfapd_kfill_done",
        ".Lfapd_kready",
        ".Lfapd_qk",
        ".Lfapd_qk_done",
        ".Lfapd_vfill_done",
        ".Lfapd_vready",
        ".Lfapd_turn",
        ".Lfapd_epi",
        ".Lfapd_end",
    ];
    let mut cursor = 0;
    for label in labels {
        let at = text[cursor..]
            .find(label)
            .unwrap_or_else(|| panic!("fp8 direct: missing label {label}"));
        cursor += at + label.len();
    }
    // Label definitions carry the trailing colon; bare finds would catch earlier branch uses (e.g. the
    // `.Lfapd_qk_done` branch at the K-ready entry precedes its label definition).
    let def = |label: &str| {
        text.find(&format!("{label}:"))
            .unwrap_or_else(|| panic!("fp8 direct: missing label {label}"))
    };
    let tile_def = def(".Lfapd_tile");
    let kfill_done_def = def(".Lfapd_kfill_done");
    let kready_def = def(".Lfapd_kready");
    let qk_done_def = def(".Lfapd_qk_done");
    let vfill_done_def = def(".Lfapd_vfill_done");
    let vready_def = def(".Lfapd_vready");
    let turn_def = def(".Lfapd_turn");
    // Helper rendezvous: the wave>=4 guard at the K-ready entry skips QK straight to `.Lfapd_qk_done` (NOT
    // `.Lfapd_turn`: the helper must still join the K-retire rendezvous before V-fill reuses the stage), and the
    // guard at the V-ready entry skips softmax/PV straight to `.Lfapd_turn`.
    assert!(
        text[kready_def..qk_done_def].contains("s_cbranch_scc1 .Lfapd_qk_done"),
        "fp8 direct: helper wave at the K-ready entry must branch to .Lfapd_qk_done"
    );
    assert!(
        !text[kready_def..qk_done_def].contains("s_cbranch_scc1 .Lfapd_turn"),
        "fp8 direct: helper wave at the K-ready entry must not skip the K-retire rendezvous"
    );
    assert!(
        text[vready_def..turn_def].contains("s_cbranch_scc1 .Lfapd_turn"),
        "fp8 direct: helper wave at the V-ready entry must branch to .Lfapd_turn"
    );
    // Fill skips: the K-fill and V-fill bodies skip to their own fill-done rendezvous, never across stages.
    assert!(
        text[tile_def..kfill_done_def].contains("s_cbranch_scc1 .Lfapd_kfill_done"),
        "fp8 direct: K-fill skip must target .Lfapd_kfill_done"
    );
    assert!(
        text[qk_done_def..vfill_done_def].contains("s_cbranch_scc1 .Lfapd_vfill_done"),
        "fp8 direct: V-fill skip must target .Lfapd_vfill_done"
    );
    // Barrier placement in the text: K publish between kfill_done and kready, K retire between qk_done and
    // vfill_done, V publish between vfill_done and vready, V retire after turn. Exactly one rendezvous per site.
    let mut signals: Vec<usize> = text
        .match_indices("s_barrier_signal")
        .map(|(i, _)| i)
        .collect();
    signals.retain(|&i| i > tile_def);
    assert_eq!(signals.len(), 4, "fp8 direct: four text rendezvous sites");
    let windows = [
        (kfill_done_def, kready_def),
        (qk_done_def, vfill_done_def),
        (vfill_done_def, vready_def),
        (turn_def, text.len()),
    ];
    for (site, (lo, hi)) in windows.iter().enumerate() {
        let inside: Vec<usize> = signals
            .iter()
            .copied()
            .filter(|&i| i > *lo && i < *hi)
            .collect();
        assert_eq!(
            inside.len(),
            1,
            "fp8 direct: rendezvous {site} is not exactly once inside its label window"
        );
    }
    // Bounded ordered path model over the tile region: every branch between the tile entry and the final V
    // retire stays inside the K-ready -> K-retire -> V-ready -> V-retire sequence, so all five waves (four
    // compute falling through, the helper rejoining at each rendezvous) meet the same four barriers in order.
    // Count alone is not the proof: each barrier pair is bracketed by its labels and no branch leaps over one.
    let region = &text[tile_def..turn_def];
    let mut targets: Vec<&str> = Vec::new();
    for l in region.lines() {
        let t = l
            .split(';')
            .next()
            .unwrap_or("")
            .split("//")
            .next()
            .unwrap_or("")
            .trim();
        let mut words = t.split_whitespace();
        match words.next() {
            Some("s_cbranch_scc1") | Some("s_cbranch_scc0") | Some("s_branch") => {
                if let Some(dst) = words.next() {
                    targets.push(dst);
                }
            }
            _ => {}
        }
    }
    let mut allowed: std::collections::BTreeSet<String> = [
        ".Lfapd_kfill_done",
        ".Lfapd_qk_done",
        ".Lfapd_vfill_done",
        ".Lfapd_turn",
        ".Lfapd_tile",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for sub in 0..4 {
        allowed.insert(format!(".Lfapd_full{sub}"));
        allowed.insert(format!(".Lfapd_mask{sub}"));
    }
    for dg in 0..4 {
        allowed.insert(format!(".Lfapd_dg{dg}_end"));
    }
    for dst in &targets {
        assert!(
            allowed.contains(&dst.to_string()),
            "fp8 direct: tile-region branch escapes the ordered barrier sequence -> {dst}"
        );
    }
    // No tile-region branch leaves the loop body early: nothing targets the epilogue, the end label, or any
    // label past the turn; every tile pass (and every wave) reaches the final V-retire rendezvous before the
    // loop-back branch returns to the tile entry.
    for ban in [".Lfapd_epi", ".Lfapd_end", ".Lfapd_vready", ".Lfapd_kready"] {
        assert!(
            !targets.contains(&ban),
            "fp8 direct: tile-region branch skips a rendezvous -> {ban}"
        );
    }
    // Causal / tail subtile early exits target the FINAL V-retire label `.Lfapd_turn`: the three sub>0 skips
    // (remaining-rows compare) plus the one V-ready helper guard make exactly four turn-bound branches in the
    // V-ready window, so a short tail still retires the reused stage instead of stranding it published.
    let sm_region = &text[vready_def..turn_def];
    let mut turn_branches = 0;
    let mut cmp_lt_guarded = 0;
    let mut wave_guarded = 0;
    let sm_lines: Vec<&str> = sm_region.lines().collect();
    for (i, l) in sm_lines.iter().enumerate() {
        let t = l
            .split(';')
            .next()
            .unwrap_or("")
            .split("//")
            .next()
            .unwrap_or("")
            .trim();
        if t == "s_cbranch_scc1 .Lfapd_turn" {
            turn_branches += 1;
            let prev: String = sm_lines[..i]
                .iter()
                .rev()
                .take(2)
                .rev()
                .copied()
                .collect::<Vec<_>>()
                .join("\n");
            if prev.contains("s_cmp_lt_i32") {
                cmp_lt_guarded += 1;
            }
            if prev.contains("s_cmp_ge_u32") {
                wave_guarded += 1;
            }
        }
    }
    assert_eq!(
        turn_branches, 4,
        "fp8 direct: V-ready window turn-bound branches (3 subtile + 1 helper)"
    );
    assert_eq!(
        cmp_lt_guarded, 3,
        "fp8 direct: subtile causal early exits must target .Lfapd_turn"
    );
    assert_eq!(
        wave_guarded, 1,
        "fp8 direct: V-ready helper guard must target .Lfapd_turn"
    );
}
