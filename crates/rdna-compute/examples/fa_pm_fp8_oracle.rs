// SPDX-License-Identifier: Apache-2.0
//! LAB-only GPU byte oracle for the A3B native-fp8 Q-resident FA2-GQA path
//! (`attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` direct and
//! `..._range_a3b_gfx1201` range, H16/KV2/D256/GQA8, native 516-byte E4M3FN K/V
//! rows, upstream-F32 fused FP8-Q prep with F32 row scales). No production
//! dispatch. Separate from the frozen Q8_0 `fa_pm_oracle` (pinned lab source/object
//! untouched); the old PM twin paragraphs below are the ORIGINAL lab record and
//! MUST stay byte-identical (old default goldens/seeds/cells unchanged).
//!
//! ```text
//! CPU only (no HIP init):
//!   fa_pm_fp8_oracle --plan          cell list, launch configs, allocation census
//!   fa_pm_fp8_oracle --self-check    CPU invariants (ABI, E4M3FN model, cache layout, fixtures)
//! GPU (hold only):
//!   fa_pm_fp8_oracle                 legacy byte oracle (unchanged default)
//!   fa_pm_fp8_oracle --filter STR    only cells whose name contains STR
//!   fa_pm_fp8_oracle --filter a3b_native  native reference + actual fused-prep gates
//! ```
//!
//! Native attention arms read the SAME host-built native cache fixtures.
//! Actual production fused-Q-prep execution is a separate mandatory family gate:
//!
//! * CANDIDATE: the new production A3B Q-resident body compiled by the repo JIT
//!   with `-DHIPFIRE_FA2_QRESIDENT_V2_Q8_A3B -DHIPFIRE_FA2_QRESIDENT_V2_Q8
//!   -DHIPFIRE_FA2_FP8 -DHIPFIRE_FA2_KMODE=8`: H16/KV2/GQA8 over caller-owned Raw
//!   Q two-plane `[T*16*256 code bytes][T*16*4 F32 scale bytes]` plus the range twin
//!   (FULL-total-batch planes with offset, output/positions stay segment views),
//!   symbols `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` +
//!   `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201`, block 768,
//!   KV head grid 2, 49,936 B dynamic LDS. Q is upstream-F32 prep output
//!   (row amax / 448 F32 scales, per-dim prepare permute), NOT F16-rounded input.
//! * REFERENCE: oracle-private independent one-query/head wave32 FP8-WMMA
//!   implementation in `attention_fp8_a3b_qresident_independent_ref.gfx1201.hip`.
//!   Canonical global byte gathers, KT48 online P quantization, block32/LDS0;
//!   no candidate body, loaders, head mapper, LDS or transpose helpers.
//!
//! Native gates: EVERY finite Winner/Kt48/Random candidate output must match
//! the quantized independent HIP reference byte-for-byte and be finite.
//! Closed-form winner expectations are secondary; KV-NaN fixtures need the mask.
//! Q-nonfinite `expected` NaNs are INPUT-TAINT markers: BOTH arms strict +0 + equality.
//! Actual `qwen35_fa_prep_batched_gfx1201` F32 and FP8 gated/nogate launches
//! share inputs; CPU per-head amax/448 + E4M3 RNE + permute checks every
//! downloaded code and F32 scale. Normalization/RoPE reuse the exact existing
//! source/ABI, not an independently implemented normalization algorithm.
//! Inputs, guards, direct/range and causal future-cache probes are mandatory.
//! Two native-only finite output poisons (0xA5 and 0x5A) prevent unwritten
//! NaNs from passing; the old global poison and legacy record are unchanged.
//! Requested new-family cases FAIL CLOSED on any module/launch/reference/producer error
//! (never skipped Green). Old f16/old-QP statements below are the untouched original record.
//!
//! ORIGINAL RECORD (frozen, verbatim):
//! Arms (both read the SAME device cache bytes; Runtime cells write them with the production
//! `kv_cache_write_fp8_e4m3_batched`, Synthetic cells upload host-built rows):
//!
//! * HIP twin: the existing production `attention_q8_0_fa2_gqa.gfx1201.hip` compiled by the repo
//!   JIT with `-DHIPFIRE_FA2_KMODE=8 -DHIPFIRE_FA2_Q0_A3B` (no FA2_FP8): the H16/KV2 f16 Q0
//!   body, symbols `attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201` +
//!   `attention_fp8_e4m3_fa2_gqa_f16_a3b_gfx1201`, block 160, 65,536 B LDS.
//! * PM: the shared Q8-module preconvert (`fa_prefill_pm_preconvert`) + the fp8 direct
//!   (`fa_prefill_pm_direct_codec(Fp8E4M3)`), plus (rows >= 64, one segment) the production wrapper
//!   `attention_fp8_e4m3_fa_prefill_pm_gfx1201`.
//!
//! Gates per cell: Q16 bytes (HIP pre vs PM pre), attention bytes (HIP vs PM), wrapper == PM
//! primitive, inputs unchanged, all guards intact, finite non-trivial output, future-cache poison
//! probes (rows at or below a probe horizon byte-stable when every later key is zeroed or made
//! extreme-finite). NaN-code cells (E4M3FN `0x7f`/`0xff`) are classified separately: NaN positions
//! must agree and every other byte must be identical. `all_byte_identical` covers only finite cells;
//! `all_classifications_pass` includes the NaN cells. A sampled f64 CPU reference over f32 Q and
//! independently decoded, f16-staged K/V omits Q16 conversion and the WMMA/reduction DAG: diagnostic only.
use hip_bridge::KernargBlob;
use rdna_compute::fa_prefill_pm::{self, FaPrefillKvCodec};
use rdna_compute::qwen35_fa_batch::FaPrepQOut;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, ffi::c_void};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
const ROW: usize = H * D;
/// Bytes of one native-fp8 cache position: 2 heads x 256 codes + 2 f16 scales.
const POS_BYTES: usize = KV * (D + 2);
const SCALE_OFF: usize = KV * D;
const MAX_ROWS: usize = 8192;
const SCALE: f32 = 1.0 / 16.0;
const GUARD: usize = 4096;
const POISON: i32 = 0xFF;
/// Native-only output poisons: both repeated-byte F32 words are finite.
const NATIVE_POISONS: [i32; 2] = [0xA5, 0x5A];
const PRE_BLOCK: u32 = 128;
const DIRECT_BLOCK: u32 = 160;
const DIRECT_LDS: u32 = 65_536;
/// Native A3B Q-resident consumer: H16/KV2/GQA8, block 768, KV-head grid 2,
/// KT48, dynamic LDS 49,936 (2 double-buffered KT48 planes + 8-byte bounds pair).
const NATIVE_BLOCK: u32 = 768;
const NATIVE_GRID_Y: u32 = 2;
const NATIVE_LDS: u32 = 49_936;
/// Code plane stride of the Raw Q two-plane layout: T*H*D bytes.
const Q_CODES_ROW: usize = H * D;
/// Scale plane stride: T*H F32 words = T*64 bytes.
const Q_SCALES_ROW: usize = H * 4;
/// Full two-plane bytes for T rows: T*(H*D + H*4) = T*4160.
const Q_BYTES_ROW: usize = Q_CODES_ROW + Q_SCALES_ROW;
/// E4M3FN max finite magnitude (OCP): 448. Row scale = amax / 448.
const E4M3_MAX: f32 = 448.0;
/// Native reference module key (private, unique; the JIT/module cache keys on
/// the full source string, so it can never collide with old f16/Q0 symbols).
const NATIVE_REF_MODULE: &str = "fa_pm_fp8_a3b_qresident_independent_ref_gfx1201";
/// Independent reference kernel source: oracle-local `include_str!`, the
/// existing LAB pattern. NOT the candidate keys/registry (no A-file edits).
const NATIVE_REF_BODY: &str =
    include_str!("../../../kernels/src/attention_fp8_a3b_qresident_independent_ref.gfx1201.hip");
/// Source binding for reused normalization/RoPE and actual provider launches.
const NATIVE_PREP_BODY: &str =
    include_str!("../../../kernels/src/qwen35_fa_prep_batched.gfx1201.hip");
const NATIVE_REF_DIRECT: &str = "attention_fp8_a3b_qresident_independent_ref_gfx1201";
const NATIVE_REF_RANGE: &str = "attention_fp8_a3b_qresident_independent_ref_range_gfx1201";
/// Candidate symbols (new A wrappers): direct + range, one module key.
const NATIVE_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201";
const NATIVE_RANGE: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201";

const HIP_BODY: &str = include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip");
/// Distinct source string (the JIT/module cache keys on it): the H16/KV2 Q0 specialization.
fn hip_source() -> String {
    format!("#define HIPFIRE_FA2_KMODE 8\n#define HIPFIRE_FA2_Q0_A3B 1\n{HIP_BODY}")
}
const HIP_PRE: &str = "attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201";
const HIP_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_f16_a3b_gfx1201";

fn sha_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn bytes<T>(x: &[T]) -> &[u8] {
    // SAFETY: only initialized f32/i32/u8 slices are passed; no padding.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
}
/// Independent OCP E4M3FN RNE encode (F32 in, code out), written from the OCP
/// formula (bias 7, 3-bit mantissa RNE, saturating at 448, NaN -> canonical
/// signed code). Explicitly DIFFERENT from the old c273 F16-first path and from
/// any candidate helper: upstream producer quantizes F32 directly (amax/448),
/// never via an F16 rounding step. Matches `__builtin_amdgcn_cvt_pk_fp8_f32`
/// (RNE) for every finite input: subnormal `x*512` RNE ties-to-even, normal
/// top-3-bit RNE with mantissa-carry into the exponent, top-of-range saturation
/// (exp > 15, or exp == 15 with mant >= 7) to 0x7E. Infinities saturate;
/// NaNs retain sign in 0x7F/0xFF. GPU prep proof admits only finite baselines.
fn e4m3_rne(x: f32) -> u8 {
    if x.is_nan() {
        return if x.is_sign_negative() { 0xff } else { 0x7f };
    }
    let bits = x.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let a = f32::from_bits(bits & 0x7fff_ffff);
    if a == 0.0 {
        return sign;
    }
    if a >= E4M3_MAX {
        return sign | 0x7e;
    }
    let fbits = a.to_bits();
    let exp_f = ((fbits >> 23) & 0xff) as i32;
    let mant_f = fbits & 0x007f_ffff;
    if exp_f == 0 {
        return sign;
    }
    let exp_u = exp_f - 127;
    if exp_u < -6 {
        let scaled = a * 512.0;
        let lo = scaled.floor();
        let frac = scaled - lo;
        let mut q = lo as i32;
        if frac > 0.5 || (frac == 0.5 && q & 1 == 1) {
            q += 1;
        }
        if q <= 0 {
            return sign;
        }
        if q >= 8 {
            return sign | (1 << 3);
        }
        return sign | (q as u8);
    }
    let mut exp_b = exp_u + 7;
    const DROP: u32 = 20;
    let mut mant = mant_f >> DROP;
    let rem = mant_f & ((1u32 << DROP) - 1);
    let half = 1u32 << (DROP - 1);
    let tie = rem == half;
    if rem > half || (tie && (mant & 1) == 1) {
        mant += 1;
        if mant == 8 {
            mant = 0;
            exp_b += 1;
        }
    }
    if exp_b > 15 || (exp_b == 15 && mant >= 7) {
        return sign | 0x7e;
    }
    sign | (((exp_b as u8) << 3) | (mant as u8))
}
/// Independent upstream-F32 prep producer for one row of 256 F32: exact
/// order-free amax, `s = (amax == 0) ? 1 : amax / 448` (IEEE division in host
/// order), `code[d] = e4m3_rne(q[d] / s)` (IEEE quotient), permuted layout
/// `byte[kg*128 + f*8 + j]` for `d = 16f + 8kg + j`. Mirrors the settled prep
/// contract (`qwen35_fa_prep_batched.gfx1201.hip:26-31,135-154`): F32 in, no
/// F16 rounding step, scales are exact F32 (not WMMA operands).
fn prep_q_row(q: &[f32]) -> (Vec<u8>, f32) {
    assert_eq!(q.len(), D);
    let mut amax = 0.0f32;
    for &v in q {
        let a = v.abs();
        if a > amax {
            amax = v.abs();
        }
    }
    let s = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
    let mut codes = vec![0u8; D];
    for d in 0..D {
        let f = d >> 4;
        let kg = (d >> 3) & 1;
        let j = d & 7;
        codes[kg * 128 + f * 8 + j] = e4m3_rne(q[d] / s);
    }
    (codes, s)
}
/// Host attention input fixture only; GPU producer proof is separate.
fn prep_q_heads(q: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(q.len() % D, 0);
    let mut codes = Vec::with_capacity(q.len());
    let mut scales = Vec::with_capacity(q.len() / D);
    for head in q.chunks_exact(D) {
        let (c, s) = prep_q_row(head);
        codes.extend_from_slice(&c);
        scales.push(s);
    }
    (codes, scales)
}

// ------------------------------------------------------- independent E4M3FN / cache model
/// OCP E4M3FN: bias 7, `E = 0` subnormal `M * 2^-9`, `E = 15 && M = 7` NaN (`None`), `0x80` = -0.
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
fn f16v(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
/// The staged f16 element value (f32 product of the f16 scale and the decoded code, one RNE to f16).
fn staged(cache: &[u8], pos: usize, head: usize, d: usize) -> f32 {
    let at = pos * POS_BYTES + SCALE_OFF + 2 * head;
    let scale = f16v(u16::from_le_bytes([cache[at], cache[at + 1]]));
    match e4m3fn(cache[pos * POS_BYTES + head * D + d]) {
        Some(v) => half::f16::from_f32(scale * v).to_f32(),
        None => f32::NAN,
    }
}
/// Independent f32 -> f16 round-to-nearest-even, copied faithfully from the independent CPU test
/// (`crates/peacemaker-isa/tests/fa_prefill_fp8.rs:44-77`), not from emitted instructions. Finite
/// inputs; overflow rounds to infinity. Bit-level: signed zeros preserved, subnormal RNE with
/// tie-to-even, normal RNE on the low 13 bits with tie-to-even.
fn ieee_f16_rne(x: f32) -> u16 {
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
/// Independent f16 bit decode (subnormal normalization, signed zeros, NaN payload preserved).
fn f16_to_f32_ieee(h: u16) -> f32 {
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
/// The independent staged f16 element word: decode E4M3FN, multiply by the decoded f16 scale in
/// f32, round once to f16. `None` for a NaN code. Overflow is determined from the rounded f16
/// exponent (as the CPU test does), not from a conservative `abs(product) > 65504` predicate.
fn ieee_stage(cache: &[u8], pos: usize, head: usize, d: usize) -> Option<u16> {
    let at = pos * POS_BYTES + SCALE_OFF + 2 * head;
    let scale = u16::from_le_bytes([cache[at], cache[at + 1]]);
    e4m3fn(cache[pos * POS_BYTES + head * D + d]).map(|v| ieee_f16_rne(f16_to_f32_ieee(scale) * v))
}
/// Single-code staging helper: `None` for NaN codes, `Some(word)` otherwise (possibly infinite
/// when the rounded exponent overflows, which the admission below detects).
fn ieee_stage_one(code: u8, scale: u16) -> Option<u16> {
    e4m3fn(code).map(|v| ieee_f16_rne(f16_to_f32_ieee(scale) * v))
}
/// A finite code admitted at `scale`: the rounded word is finite. Products rounding nonfinite are
/// replaced by the code of 1.0 (`0x38`); every admitted code is independently censused by the
/// frozen-case constructors and the self-check below.
fn ieee_safe_code(scale: u16, code: u8) -> u8 {
    match ieee_stage_one(code, scale) {
        Some(w) if w & 0x7c00 != 0x7c00 => code,
        _ => 0x38,
    }
}
/// Scale boundaries of the f16 staging: smallest subnormal, largest subnormal, smallest normal, 1.0,
/// 1 + ulp, f16 max, two mid-range scales.
const SCALES: [u16; 8] = [
    0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000,
];
/// A finite code whose f16 staged product with `scale` cannot overflow (replaced by 1.0 otherwise).
fn safe_code(scale: u16, code: u8) -> u8 {
    if (f16v(scale) * e4m3fn(code).unwrap()).abs() > 65504.0 {
        0x38
    } else {
        code
    }
}
/// Host-built `[ctx]` rows: every finite code at every scale boundary on both heads (K and V use different
/// rotations), both heads with distinct scales. `nan` injects `0x7f` / `0xff` codes.
fn grid_cache(ctx: usize, salt: usize, nan: bool) -> Vec<u8> {
    let table = finite_codes();
    let mut out = vec![0u8; ctx * POS_BYTES];
    for p in 0..ctx {
        for g in 0..KV {
            let scale = SCALES[(p + 3 * g + salt) % SCALES.len()];
            out[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&scale.to_le_bytes());
            for d in 0..D {
                out[p * POS_BYTES + g * D + d] =
                    safe_code(scale, table[(d + 7 * p + 31 * g + 13 * salt) % table.len()]);
            }
        }
    }
    if nan {
        // A few NaN codes (positions >= 3 so every causal row >= 3 reads them): both heads, both NaN encodings.
        for (p, g, d, code) in [
            (3usize, 0usize, 5usize, 0x7fu8),
            (4, 1, 200, 0xff),
            (5, 0, 0, 0xff),
            (6, 1, 255, 0x7f),
        ] {
            if p < ctx {
                out[p * POS_BYTES + g * D + d] = code;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- fixtures
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fixture {
    Std,
    Reverse8,
    DupPairs8,
    Rotate8,
    SameGroup8,
}
impl Fixture {
    fn name(self) -> &'static str {
        match self {
            Fixture::Std => "std",
            Fixture::Reverse8 => "reverse8",
            Fixture::DupPairs8 => "dup_pairs8",
            Fixture::Rotate8 => "rotate8",
            Fixture::SameGroup8 => "same_group8",
        }
    }
    fn offset(self, r: usize, g: usize, m: usize) -> usize {
        match self {
            Fixture::Std => r,
            Fixture::Reverse8 => m - 1 - r,
            Fixture::DupPairs8 => (r & !1) % m,
            Fixture::Rotate8 => (r * 5 + g) % 8 % m,
            Fixture::SameGroup8 => m - 1,
        }
    }
    fn positions(self, rows: usize, start: usize) -> Vec<i32> {
        (0..rows)
            .map(|row| {
                let (g, r) = (row / 8, row % 8);
                let m = (rows - g * 8).min(8);
                (start + g * 8 + self.offset(r, g, m)) as i32
            })
            .collect()
    }
    fn for_round(self, round: usize) -> Fixture {
        match (round, self) {
            (0, f) => f,
            (_, Fixture::Std) => Fixture::Reverse8,
            _ => Fixture::Std,
        }
    }
}

// ------------------------------------------------------------------ cells
#[derive(Clone, Copy, PartialEq, Debug)]
enum Source {
    /// Random f32 K/V through the production `kv_cache_write_fp8_e4m3_batched`.
    Runtime,
    /// Host-built rows: every finite code x every f16 scale boundary, distinct per-head scales.
    Grid,
    /// `Grid` plus NaN codes (separate classification, not the finite byte gate).
    Nan,
}
impl Source {
    fn name(self) -> &'static str {
        match self {
            Source::Runtime => "runtime",
            Source::Grid => "grid",
            Source::Nan => "nan",
        }
    }
}
/// Native Q-resident qualification cell against the independent quantized HIP
/// reference. Disjoint from the frozen `Cell` matrix: old `oracle_cells()`/
/// `LONG_CELLS` entries are never renamed, re-seeded or re-gated.
#[derive(Clone, Copy, PartialEq, Debug)]
enum NativeKind {
    /// Winner-take-all byte gate: uniform all-448 Q rows, winner K/V
    /// (K 448@1.0, V 448@56.0), every loser +0. expf underflow makes the
    /// chain bit-deterministic with NO tolerance (see `native_winner_kv`).
    Winner,
    /// KT48 boundary sweep: same exact winner fixture with live-key counts
    /// 47/48/49/63/64/65 straddling tile edges. Byte-exact gate.
    Kt48,
    /// General finite per-head/query-varied Q and K/V: mandatory byte gate.
    Random,
    /// Both E4M3 NaN encodings: K key3 and V key0. Independently specified
    /// NaN positions must agree; unaffected words must be finite and exact.
    Nan,
    QNan,
    QInf,
}
impl NativeKind {
    fn name(self) -> &'static str {
        match self {
            NativeKind::Winner => "winner",
            NativeKind::Kt48 => "kt48",
            NativeKind::Random => "random",
            NativeKind::Nan => "nan",
            NativeKind::QNan => "q_nan_probe",
            NativeKind::QInf => "q_inf_probe",
        }
    }
    fn nonfinite(self) -> bool {
        matches!(self, Self::Nan | Self::QNan | Self::QInf)
    }
    fn winner(self) -> bool {
        matches!(self, Self::Winner | Self::Kt48)
    }
}
#[derive(Clone, Copy)]
struct NativeCell {
    rows: usize,
    start: usize,
    kind: NativeKind,
}
impl NativeCell {
    fn ctx(&self) -> usize {
        self.start + self.rows.max(2)
    }
    fn name(&self) -> String {
        format!(
            "a3b_native_{}_n{}_s{}",
            self.kind.name(),
            self.rows,
            self.start
        )
    }
    fn gpu_grid(&self, rows: usize) -> [u32; 3] {
        [
            (rows.min(512) * 8).div_ceil(384) as u32,
            NATIVE_GRID_Y,
            rows.div_ceil(512) as u32,
        ]
    }
    /// Full two-plane Q bytes for `total` rows (codes + F32 scales).
    fn q_bytes(total: usize) -> usize {
        total * Q_BYTES_ROW
    }
    /// Native K/V cache bytes actually accessed: `max_ctx * 516` per side
    /// (mapped-prefix bound, not the reservation; VMM owners use their mapped
    /// prefix via `require_cache_bytes`).
    fn cache_bytes(max_ctx: usize) -> usize {
        max_ctx * POS_BYTES
    }
}
#[derive(Clone, Copy)]
struct Cell {
    rows: usize,
    start: usize,
    segment: usize,
    fixture: Fixture,
    source: Source,
}
impl Cell {
    fn ctx(&self) -> usize {
        self.start + self.rows
    }
    fn name(&self) -> String {
        format!(
            "n{}_s{}_seg{}_{}_{}",
            self.rows,
            self.start,
            self.segment,
            self.fixture.name(),
            self.source.name()
        )
    }
    fn segments(&self) -> Vec<(usize, usize)> {
        (0..self.rows)
            .step_by(self.segment)
            .map(|off| (off, (self.rows - off).min(self.segment)))
            .collect()
    }
    /// The production wrapper admits one whole 64..=8192-row segment.
    fn wrapper(&self) -> bool {
        self.segment == self.rows
            && (fa_prefill_pm::MIN_ROWS..=fa_prefill_pm::MAX_ROWS).contains(&self.rows)
    }
    /// Future-cache poison probes and the sampled CPU reference need standard positions.
    fn diagnostics(&self) -> bool {
        self.fixture == Fixture::Std
            && self.source == Source::Runtime
            && self.start == 0
            && self.segment == self.rows
    }
}
// -------------------------------------------- independent native semantic proof (section N)
/// Independent case descriptor: host-built Q16/K/V inputs plus the independently computed
/// expected output words. The original `Source`, names, seeds and ordinary matrix are unchanged;
/// these cases run before the large ordinary arena is constructed.
#[derive(Clone)]
struct IeeeCase {
    name: String,
    qh: Vec<u16>,
    positions: Vec<i32>,
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    expected: IeeeExpected,
}
/// Independent expected output: finite words, or V-NaN / K-NaN with the clean (NaN-free) finite
/// reference retained so unaffected bytes are checked separately from affected NaN positions.
#[derive(Clone)]
enum IeeeExpected {
    Finite(Vec<u32>),
    VNaN {
        clean_expected_words: Vec<u32>,
        kv_head: usize,
        dimension: usize,
    },
    KNaN {
        clean_expected_words: Vec<u32>,
        kv_head: usize,
    },
}
impl IeeeExpected {
    fn kind(&self) -> &'static str {
        match self {
            IeeeExpected::Finite(_) => "finite_byte_gate",
            IeeeExpected::VNaN { .. } => "v_nan_separate",
            IeeeExpected::KNaN { .. } => "k_nan_separate",
        }
    }
    #[allow(dead_code)]
    fn words(&self) -> &[u32] {
        match self {
            IeeeExpected::Finite(w) => w,
            IeeeExpected::VNaN {
                clean_expected_words,
                ..
            } => clean_expected_words,
            IeeeExpected::KNaN {
                clean_expected_words,
                ..
            } => clean_expected_words,
        }
    }
}
/// Finite codes whose value is a multiple of 1/8 (or zero): V operands whose tied sums are exact
/// in f32 for any tie count. Copied from the CPU test (`exact_v_codes`).
fn ieee_exact_v_codes() -> Vec<u8> {
    finite_codes()
        .into_iter()
        .filter(|&c| {
            let v = e4m3fn(c).unwrap();
            v == 0.0 || (v.abs() >= 0.125 && (v * 8.0).fract() == 0.0)
        })
        .collect()
}
/// Independent `winner_case` arithmetic (CPU test `:249-274`): Q16 sign pattern per row/head,
/// K code plane with opposite head signs and opposing scale ramps, V exact-sum codes at 1.0.
/// Returns `(qh, k_cache, v_cache)` with `npos = max(positions) + 1` allocated contexts.
fn ieee_winner_inputs(
    batch: usize,
    positions: &[i32],
    qmag: f32,
    salt: usize,
) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    let npos = *positions.iter().max().unwrap() as usize + 1;
    let qh = (0..batch * ROW)
        .map(|i| {
            let (r, h, d) = (i / ROW, (i / D) % H, i % D);
            let sign = if (r * 5 + h * 3 + salt) % 4 < 2 {
                1.0
            } else {
                -1.0
            };
            ieee_f16_rne(sign * qmag * (1 + d % 3) as f32)
        })
        .collect();
    let mut k = vec![0u8; npos * POS_BYTES];
    for p in 0..npos {
        for g in 0..KV {
            let scale = ieee_f16_rne(if g == 0 {
                0.25 * (1 + p) as f32
            } else {
                0.25 * (200 - p) as f32
            });
            k[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&scale.to_le_bytes());
            for d in 0..D {
                let code = match d % 61 {
                    0 => 0x7e,
                    30 => 0xfe,
                    _ => 0x68 + ((d * 37 + salt) % 4) as u8,
                };
                k[p * POS_BYTES + g * D + d] = code ^ if g == 0 { 0 } else { 0x80 };
            }
        }
    }
    let vcodes = ieee_exact_v_codes();
    let mut v = vec![0u8; npos * POS_BYTES];
    for p in 0..npos {
        for g in 0..KV {
            v[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
            for d in 0..D {
                v[p * POS_BYTES + g * D + d] =
                    vcodes[(p * 29 + d * 7 + g * 61 + salt * 13) % vcodes.len()];
            }
        }
    }
    (qh, k, v)
}
/// Single-live-key caches at `scales` (CPU test `single_key_case`): every finite code admitted by
/// the rounded-exponent overflow rule, rotated by `shift` per head. Returns `(k, v)` single-row
/// caches; the caller reuses them for both K and V single-key sweeps.
fn single_key_caches(scales: [u16; 2], shift: usize) -> (Vec<u8>, Vec<u8>) {
    let table = finite_codes();
    let mut k = vec![0u8; POS_BYTES];
    let mut v = vec![0u8; POS_BYTES];
    for g in 0..KV {
        k[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2].copy_from_slice(&scales[g].to_le_bytes());
        v[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2].copy_from_slice(&scales[g].to_le_bytes());
        for d in 0..D {
            k[g * D + d] =
                ieee_safe_code(scales[g], table[(d + 31 * shift + 128 * g) % table.len()]);
            v[g * D + d] =
                ieee_safe_code(scales[g], table[(d + 31 * shift + 128 * g) % table.len()]);
        }
    }
    (k, v)
}
/// Independent expected output words (CPU test `expected`, `:278-316`): independently staged K/V,
/// f64 dots, f32 score conversion, exact-max ties, minimum 200 gap for nonwinners,
/// exact-representability assertion for tied V sums, ascending accumulation from +0.0, final f32
/// reciprocal multiplication. Single-key output is therefore NOT a V-bit echo: a staged -0
/// normally becomes output +0. NaN codes propagate: a NaN staged K score (or V element) yields
/// NaN outputs in the affected head lanes.
fn ieee_expected(ic: &IeeeCase) -> Result<Vec<u32>> {
    let npos_k = ic.k_cache.len() / POS_BYTES;
    let npos_v = ic.v_cache.len() / POS_BYTES;
    if npos_k == 0 || npos_v == 0 || npos_k != npos_v {
        return Err("independent caches must be non-empty and K/V matched".into());
    }
    let npos = npos_k;
    let batch = ic.positions.len();
    // K staging keeps NaN lanes: a NaN staged K element makes that key's f64 dot NaN, the f32
    // score NaN, the row max NaN, and every output of the affected KV head NaN (the CPU test's
    // K-NaN assertion). Non-NaN lanes stay exact.
    let kst: Vec<Vec<f32>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| match ieee_stage(&ic.k_cache, i % npos, i / npos, d) {
                    Some(w) => f16_to_f32_ieee(w),
                    None => f32::NAN,
                })
                .collect()
        })
        .collect();
    // V staging keeps NaN lanes (V NaN cases): `None` word 0x7e00 is a placeholder; NaN outputs
    // are decided per element below from the raw codes so the clean reference stays exact.
    let v_is_nan: Vec<Vec<bool>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| e4m3fn(ic.v_cache[i % npos * POS_BYTES + (i / npos) * D + d]).is_none())
                .collect()
        })
        .collect();
    let vst: Vec<Vec<f32>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| match ieee_stage(&ic.v_cache, i % npos, i / npos, d) {
                    Some(w) => f16_to_f32_ieee(w),
                    None => f32::NAN,
                })
                .collect()
        })
        .collect();
    let mut out = vec![0u32; batch * ROW];
    for r in 0..batch {
        let pos = ic.positions[r] as usize;
        if pos >= npos {
            return Err(
                format!("independent position {pos} exceeds allocated context {npos}").into(),
            );
        }
        for h in 0..H {
            let g = h / 8;
            let q: Vec<f64> = (0..D)
                .map(|d| f64::from(f16_to_f32_ieee(ic.qh[(r * H + h) * D + d])))
                .collect();
            let scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let dot: f64 = (0..D).map(|d| q[d] * f64::from(kst[g * npos + j][d])).sum();
                    (dot as f32) * ic.qh_scale()
                })
                .collect();
            // A NaN staged K element makes the dot NaN (`f32::max` would hide it in the fold).
            let nan_score = scores.iter().any(|s| s.is_nan());
            let m = if nan_score {
                f32::NAN
            } else {
                scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };
            let ties: Vec<usize> = (0..=pos).filter(|&j| nan_score || scores[j] == m).collect();
            for j in 0..=pos {
                if !nan_score {
                    assert!(
                        scores[j] == m || m - scores[j] >= 200.0,
                        "ill-posed independent case: row {r} head {h} key {j} gap {}",
                        m - scores[j]
                    );
                }
            }
            let inv = 1.0f32 / ties.len().max(1) as f32;
            for d in 0..D {
                let tied_nan = !nan_score && ties.iter().any(|&j| v_is_nan[g * npos + j][d]);
                if nan_score || tied_nan {
                    out[(r * H + h) * D + d] = f32::NAN.to_bits();
                    continue;
                }
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
                out[(r * H + h) * D + d] = (sum * inv).to_bits();
            }
        }
    }
    Ok(out)
}
impl IeeeCase {
    fn qh_scale(&self) -> f32 {
        // All frozen independent cases use attention scale 1.0 (stored out-of-band would be a
        // second convention; the CPU test passes it per case and every frozen N case uses 1.0).
        1.0
    }
}
/// Frozen independent cases in contract order (CPU test `:489-511,533-555,660-706`):
/// K NaN `0x7f` then `0xff` (single live key, KV head 1, dim 17); empty-masked-subtile positions
/// `[64,0,65,32,15,16,31,48]`; V NaN tuples `(0x7f,0,5)`, `(0xff,1,200)`, `(0x7f,1,0)`,
/// `(0xff,0,255)`; finite scale sweep over the 12 f16 boundary scales (each head a different
/// scale) plus the swapped-head-scale negative control and the swapped-scale-order pair.
/// One-live-key means causal position 0 with finite cache rows allocated only as needed
/// (single-row caches here); eight query rows do not require an eight-key horizon.
fn ieee_cases() -> Result<Vec<IeeeCase>> {
    let mut v = Vec::new();
    // 1. K NaN, cheap first: single live key, KV head 1 / dim 17, `0x7f` then `0xff`.
    for code in [0x7fu8, 0xff] {
        let (qh, mut k, vc) = ieee_winner_inputs(8, &[0; 8], 1.0, 9);
        let clean = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        k[D + 17] = code;
        v.push(IeeeCase {
            name: format!("ieee_k_nan_{code:#04x}_kvhead1_dim17"),
            qh,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::KNaN {
                clean_expected_words: clean,
                kv_head: 1,
            },
        });
    }
    // 2. Empty-masked-subtile case: low-horizon row keeps its finite answer while later subtiles
    // are causally masked (not an invalid empty-batch launch).
    {
        let positions = vec![64, 0, 65, 32, 15, 16, 31, 48];
        let (qh, k, vc) = ieee_winner_inputs(8, &positions, 1.0, 0);
        let words = ieee_expected(&IeeeCase {
            name: "ieee_empty_subtile".into(),
            qh: qh.clone(),
            positions: positions.clone(),
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        v.push(IeeeCase {
            name: "ieee_empty_subtile".into(),
            qh,
            positions,
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::Finite(words),
        });
    }
    // 3. V NaN tuples from the CPU test.
    for (code, g, d0) in [
        (0x7fu8, 0usize, 5usize),
        (0xff, 1, 200),
        (0x7f, 1, 0),
        (0xff, 0, 255),
    ] {
        let (qh, k, mut vc) = ieee_winner_inputs(8, &[0; 8], 1.0, 7);
        let clean = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        vc[g * D + d0] = code;
        v.push(IeeeCase {
            name: format!("ieee_v_nan_{code:#04x}_kvhead{g}_dim{d0}"),
            qh,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::VNaN {
                clean_expected_words: clean,
                kv_head: g,
                dimension: d0,
            },
        });
    }
    // 4. Finite scale sweep over the 12 f16 boundary scales; each head a different scale.
    const IEEE_SWEEP: [u16; 12] = [
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000, 0xbc01,
        0xfbff,
    ];
    for shift in 0..IEEE_SWEEP.len() {
        let scales = [
            IEEE_SWEEP[shift],
            IEEE_SWEEP[(shift + 3) % IEEE_SWEEP.len()],
        ];
        let (k, vc) = single_key_caches(scales, shift);
        let (qh_base, _, _) = ieee_winner_inputs(8, &[0; 8], 1.0, 3 + shift);
        let words = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh_base.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        v.push(IeeeCase {
            name: format!("ieee_finite_scale_{scales:#06x?}_shift{shift}"),
            qh: qh_base,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::Finite(words),
        });
    }
    // Swapped-head-scale negative control: outputs must follow the head, so the swapped pair differs.
    {
        let (ka, va) = single_key_caches([0x3c00, 0x4000], 1);
        let (kb, vb) = single_key_caches([0x4000, 0x3c00], 1);
        let (qh, _, _) = ieee_winner_inputs(8, &[0; 8], 1.0, 4);
        let wa = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: ka.clone(),
            v_cache: va.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        let wb = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: kb.clone(),
            v_cache: vb.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        assert!(
            wa != wb,
            "swapped head scales must change the independent expectation"
        );
        v.push(IeeeCase {
            name: "ieee_head_scales_a".into(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: ka,
            v_cache: va,
            expected: IeeeExpected::Finite(wa),
        });
        v.push(IeeeCase {
            name: "ieee_head_scales_b".into(),
            qh,
            positions: vec![0; 8],
            k_cache: kb,
            v_cache: vb,
            expected: IeeeExpected::Finite(wb),
        });
    }
    Ok(v)
}
/// Native H16/KV2 qualification over KT48 boundaries, ragged/nonzero-past
/// queries, FULL-total-batch plane offsets, odd 516-byte rows and long mapped
/// prefixes. Every finite cell, including general data, gates byte-exactly
/// against the independent quantized HIP reference. Winner closed forms are
/// additional checks only. Counts are PLANNED, not observed GPU results.
const NATIVE_CELLS: [(usize, usize, NativeKind); 30] = [
    // Winner gates, short horizons: tails, one block, ragged rows.
    (1, 0, NativeKind::Winner),
    (7, 0, NativeKind::Winner),
    (8, 0, NativeKind::Winner),
    (16, 0, NativeKind::Winner),
    (48, 1, NativeKind::Winner),
    (64, 1, NativeKind::Winner),
    // KT48 tile edges: live-key counts 47/48/49 and 63/64/65 (winner pinned
    // at `start`, so live keys = start + 1), straddling one-tile, sub-tile
    // and two-tile edges. Odd winner keys (47, 63) sit on odd 516-byte rows
    // (4-aligned, not 8-aligned): the legal-u32 K fill and the 8-byte V
    // gather MUST emit the same fragments as dense H24's 8-byte fast path.
    (3, 46, NativeKind::Kt48),
    (4, 47, NativeKind::Kt48),
    (5, 48, NativeKind::Kt48),
    (15, 62, NativeKind::Kt48),
    (16, 63, NativeKind::Kt48),
    (17, 64, NativeKind::Kt48),
    // Winner gates: KT48-straddling multi-block rows (49 queries = 392 rows >
    // 384 block; 600 queries exercise grid z = 2 with run_rows = 512 + 88
    // tail), all pinned at small horizons.
    (49, 1, NativeKind::Winner),
    (48, 14, NativeKind::Winner),
    (600, 2, NativeKind::Winner),
    // General finite ragged / nonzero-past mandatory byte gates.
    // Varied horizons expose 48-query WG vs obsolete 16-query grouping and
    // the 512-query run reset; uniform winner fixtures cannot detect this.
    (47, 63, NativeKind::Random),
    (48, 63, NativeKind::Random),
    (49, 63, NativeKind::Random),
    (511, 1, NativeKind::Random),
    (513, 1, NativeKind::Random),
    (512, 1, NativeKind::Random),
    (512, 63, NativeKind::Random),
    (64, 64, NativeKind::Random),
    (65, 63, NativeKind::Random),
    // Native cache mapped-bound probes: last row exactly at the mapped prefix
    // (long-context tail at 16383) and a 4-row overhang window whose range
    // segment starts at a nonzero offset (FULL-plane offsets exercised).
    (512, 15872, NativeKind::Random),
    (64, 16320, NativeKind::Random),
    // pp16000 prefill chunks (A3B Qwen3.6, two chunks over a 16000 prompt):
    // 8192 rows from position 0 (16 full 512-query runs) and 7808 rows from
    // position 8192 (15 full runs + one 128-row tail); 11 blocks per run, up to
    // ~333 KT48 tiles per block. Pins the whole causal work distribution.
    (8192, 0, NativeKind::Random),
    (7808, 8192, NativeKind::Random),
    // NaN classification cells (separate classifier, not the finite gate).
    (8, 0, NativeKind::Nan),
    (65, 63, NativeKind::Nan),
];
fn native_cells() -> Vec<NativeCell> {
    let mut cells: Vec<_> = NATIVE_CELLS
        .iter()
        .map(|&(rows, start, kind)| NativeCell { rows, start, kind })
        .collect();
    // Close the identified nonfinite-Q hole only; do not manufacture a
    // one-for-one replacement for the obsolete 31-case F16/KT64 packet set.
    cells.push(NativeCell {
        rows: 8,
        start: 0,
        kind: NativeKind::QNan,
    });
    cells.push(NativeCell {
        rows: 8,
        start: 0,
        kind: NativeKind::QInf,
    });
    cells
}
/// Bounded consumer-visible coverage, not a cardinality-equivalence or reuse
/// of old packet numerical PASS. Primitive reuse is explicitly source-bound.
fn native_coverage_map() -> Value {
    json!({
        "native_cells":32,"finite":28,"nonfinite":4,
        "reference":"all finite direct/range outputs: independent H16 FP8-WMMA KT48 byte gate",
        "shape_groups":{
            "tails_winners":"n1/7/8/16, n48_s1, n64_s1, n49_s1, n48_s14, n600_s2",
            "KT48_live_keys":"47/48/49/63/64/65, including odd 516-byte rows",
            "general_positions":"n47/48/49_s63; n511/512/513_s1; n512_s63; n64_s64; n65_s63",
            "long_active_prefix":"n512_s15872 and n64_s16320 reach key16383",
            "pp16000_chunks":"n8192_s0 (ctx 8192) and n7808_s8192 (ctx 16000): full 512-query runs, causal work distribution",
            "range":"every cell: FULL-total scale plane, offset0 replay and nonzero offset where possible",
            "nonfinite":"KV NaN n8_s0/n65_s63 plus F32-Q NaN/Inf n8_s0; independent affected-head masks"
        },
        "producer":"two finite F32 fixtures x three actual provider modes; full codes/scales, varied heads, non-F16-exact values",
        "legacy_primitive_reuse":"unchanged native codec/writer and Fixture position primitives retain legacy source/ABI; CPU E4M3 signed-zero/tie/subnormal checks remain",
        "not_claimed":"old31 case-for-case native GPU equivalence, F16-first rounding, KT64 recurrence, or old packet attention PASS"
    })
}
/// General finite K/V: rotating E4M3 sweeps on odd/even 516-byte rows,
/// with head/key-dependent F16 scales. No assumed 8-byte alignment.
fn native_cache(ctx: usize, salt: usize) -> Vec<u8> {
    let table = finite_codes();
    let scales = [0x1400u16, 0x1800, 0x1c00, 0x2000];
    let mut out = vec![0u8; ctx * POS_BYTES];
    for p in 0..ctx {
        for g in 0..KV {
            let scale = scales[(p * 3 + g + salt) % scales.len()].to_le_bytes();
            out[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&scale);
            for d in 0..D {
                let shift = if p % 2 == 1 { 127 } else { 0 };
                out[p * POS_BYTES + g * D + d] =
                    table[(d + 11 * p + 37 * g + salt + shift) % table.len()];
            }
        }
    }
    out
}
/// Exact winner-take-all K/V: key `win` carries code 0x7e (448) on every dim
/// of both heads; K scale 1.0, V scale F16 56.0 (`0x5300`). Every other key is
/// +0 (codes 0x00, scales 1.0). With Q rows all +448 (prep scale exactly 1.0,
/// codes 0x7e), the winner score (~3.2M after the 1/16 scale) beats every loser
/// (exactly 0) by thousands, so every loser weight underflows to exactly +0 in
/// both implementations. The closed-form secondary check is the winner V
/// value 448*56 = 25088; the independent quantized HIP byte gate is primary.
fn native_winner_kv(ctx: usize, win: usize, v_scale: u16) -> (Vec<u8>, Vec<u8>) {
    let mut k = vec![0u8; ctx * POS_BYTES];
    let mut v = vec![0u8; ctx * POS_BYTES];
    for p in 0..ctx {
        let live = p == win;
        let code = if live { 0x7e } else { 0x00 };
        for g in 0..KV {
            k[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
            let vs = if live { v_scale } else { 0x3c00u16 };
            v[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&vs.to_le_bytes());
            for d in 0..D {
                k[p * POS_BYTES + g * D + d] = code;
                v[p * POS_BYTES + g * D + d] = code;
            }
        }
    }
    (k, v)
}
/// Additional closed-form check for uniform-Q single-winner fixtures only.
/// General finite and NaN expectations never use this construction.
fn native_expected(k_cache: &[u8], positions: &[i32]) -> Vec<f32> {
    let rows = positions.len();
    let mut out = vec![0f32; rows * ROW];
    // Winner V value: 448 (0x7e) x 56.0 (0x5300), exact in f32.
    let w = 448.0f32 * f16_to_f32_ieee(0x5300);
    for q in 0..rows {
        let win = positions[q] as usize;
        // Sanity: the winner key carries 0x7e on head 0 dim 0 (proves the
        // fixture builder and the reference agree on which key wins; a
        // mismatched fixture FAILS via the byte gate, never silently).
        debug_assert_eq!(k_cache[win * POS_BYTES], 0x7e);
        for h in 0..H {
            for d in 0..D {
                out[(q * H + h) * D + d] = w;
            }
        }
    }
    out
}
/// Runtime-written long-context cells (rows, start, segment): context exactly 16384, uploaded
/// positions include 16383, whole + segmented equality, wrapper admission where applicable, two
/// changed-position rounds and both production K/V writes (executed by the ordinary path).
const LONG_CTX: usize = 16384;
const LONG_CELLS: [(usize, usize, usize); 3] = [
    (512, LONG_CTX - 512, 512),
    (8192, 8192, 8192),
    (8192, 8192, 512),
];
fn long_cells() -> Vec<Cell> {
    LONG_CELLS
        .iter()
        .map(|&(rows, start, segment)| Cell {
            rows,
            start,
            segment,
            fixture: Fixture::Std,
            source: Source::Runtime,
        })
        .collect()
}
const A0_SHAPES: [(usize, usize); 5] = [(512, 0), (4096, 0), (8192, 0), (512, 3584), (4096, 4096)];
const BOUNDARY_SHAPES: [(usize, usize); 3] = [(64, 0), (65, 0), (8191, 0)];
const UNALIGNED_SHAPES: [(usize, usize); 2] = [(512, 1), (512, 63)];
const FIXTURE_SHAPES: [(usize, usize); 4] = [(512, 0), (512, 3584), (65, 63), (4096, 0)];
const FIXTURES: [Fixture; 4] = [
    Fixture::Reverse8,
    Fixture::DupPairs8,
    Fixture::Rotate8,
    Fixture::SameGroup8,
];
/// Exact primitive tails below the production minimum (1..=63 rows).
const TAIL_ROWS: [usize; 12] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63];
const TAIL_STARTS: [usize; 2] = [0, 17];
const GRID_SHAPES: [(usize, usize); 4] = [(64, 0), (65, 63), (512, 0), (512, 3584)];

fn push_modes(v: &mut Vec<Cell>, rows: usize, start: usize, fixture: Fixture, source: Source) {
    v.push(Cell {
        rows,
        start,
        segment: rows,
        fixture,
        source,
    });
    if rows > 512 {
        v.push(Cell {
            rows,
            start,
            segment: 512,
            fixture,
            source,
        });
    }
}
fn oracle_cells() -> Vec<Cell> {
    let mut v = Vec::new();
    for (rows, start) in A0_SHAPES
        .into_iter()
        .chain(BOUNDARY_SHAPES)
        .chain(UNALIGNED_SHAPES)
    {
        push_modes(&mut v, rows, start, Fixture::Std, Source::Runtime);
    }
    for (rows, start) in FIXTURE_SHAPES {
        for f in FIXTURES {
            push_modes(&mut v, rows, start, f, Source::Runtime);
        }
    }
    for rows in TAIL_ROWS {
        for start in TAIL_STARTS {
            v.push(Cell {
                rows,
                start,
                segment: rows,
                fixture: Fixture::Std,
                source: Source::Runtime,
            });
        }
    }
    for (rows, start) in GRID_SHAPES {
        push_modes(&mut v, rows, start, Fixture::Std, Source::Grid);
    }
    v.push(Cell {
        rows: 64,
        start: 0,
        segment: 64,
        fixture: Fixture::Std,
        source: Source::Nan,
    });
    v.push(Cell {
        rows: 65,
        start: 63,
        segment: 65,
        fixture: Fixture::Reverse8,
        source: Source::Nan,
    });
    v
}

// ---------------------------------------------------------- memory census
#[derive(Clone, Copy, Default)]
struct Used {
    q32: usize,
    k32: usize,
    v32: usize,
    allpos: usize,
    kc: usize,
    vc: usize,
    pos: usize,
    q16: usize,
    out: usize,
}
fn used(rows: usize, ctx: usize) -> Used {
    Used {
        q32: rows * ROW * 4,
        k32: ctx * KV * D * 4,
        v32: ctx * KV * D * 4,
        allpos: ctx * 4,
        kc: ctx * POS_BYTES,
        vc: ctx * POS_BYTES,
        pos: rows * 4,
        q16: rows * ROW * 2,
        out: rows * ROW * 4,
    }
}
fn used_of(c: &Cell) -> Used {
    used(c.rows, c.ctx())
}
const BUFFER_NAMES: [&str; 12] = [
    "q32",
    "k32_src",
    "v32_src",
    "allpos",
    "k_cache_fp8",
    "v_cache_fp8",
    "positions",
    "q16_hip",
    "q16_pm",
    "out_hip",
    "out_pm",
    "out_wrapper_or_probe",
];
fn used_list(u: &Used) -> [usize; 12] {
    [
        u.q32, u.k32, u.v32, u.allpos, u.kc, u.vc, u.pos, u.q16, u.q16, u.out, u.out, u.out,
    ]
}
fn arena_sizes(cells: &[Cell]) -> [usize; 12] {
    let mut m = [0usize; 12];
    for c in cells {
        for (slot, b) in m.iter_mut().zip(used_list(&used_of(c))) {
            *slot = (*slot).max(b);
        }
    }
    m
}

// ----------------------------------------------------------- ABI helpers
fn pre_kernargs(q: *mut c_void, q16: *mut c_void, n: usize) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(q);
    b.push_ptr(q16);
    b.push_ptr(std::ptr::null::<c_void>()); // signs1: ignored by the ABI
    b.push_ptr(std::ptr::null::<c_void>()); // signs2: ignored by the ABI
    b.push_i32(n as i32);
    b.push_i32(0); // do_rotate: ignored by the ABI
    b
}
fn direct_kernargs(
    q16: *mut c_void,
    kc: *mut c_void,
    vc: *mut c_void,
    out: *mut c_void,
    pos: *mut c_void,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [q16, kc, vc, out, pos] {
        b.push_ptr(p);
    }
    for x in [H, KV, D, n] {
        b.push_i32(x as i32);
    }
    b.push_f32(SCALE);
    b
}
fn pre_grid(n: usize) -> [u32; 3] {
    [(n * H).div_ceil(4) as u32, 1, 1]
}
fn direct_grid(n: usize) -> [u32; 3] {
    [n.div_ceil(8) as u32, KV as u32, 1]
}
/// Native Q-resident kernargs: the DIRECT dense packed ABI (6 pointers +
/// 4 i32 shape scalars + 1 f32 scale = 68 B packed layout in the LAB arg blob):
/// `q_codes, q_scales, k_cache, v_cache, out, positions, H16, KV2, D256, batch, 1/16`.
/// Range applies FULL-total-batch plane offsets host-side: codes at
/// `row_offset * 4096`, scales at `total * 4096 + row_offset * 64`.
fn native_kernargs(
    q_codes: *mut c_void,
    q_scales: *mut c_void,
    kc: *mut c_void,
    vc: *mut c_void,
    out: *mut c_void,
    pos: *mut c_void,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [q_codes, q_scales, kc, vc, out, pos] {
        b.push_ptr(p);
    }
    for x in [H, KV, D, n] {
        b.push_i32(x as i32);
    }
    b.push_f32(SCALE);
    b
}
/// Admission mirror of the B runtime (`qresident_a3b_launch`): H16/KV2/D256,
/// 1..=32768 rows, positions nonnegative and below the accessed prefix, Raw Q
/// two-plane extent, F32 output extent, strict `require_cache_bytes` style
/// mapped-prefix check per cache side (VMM owners use their mapped prefix, not
/// the reservation). Requested family FAILS CLOSED: `Err` on any violation,
/// never a skipped Green.
fn native_validate(
    rows: usize,
    positions: &[i32],
    max_ctx: usize,
    q_bytes: usize,
    cache_bytes: usize,
) -> std::result::Result<(), String> {
    if rows < 1 || rows > 32768 {
        return Err(format!("native rows {rows} outside 1..=32768"));
    }
    if max_ctx < 1 || max_ctx > 262_144 {
        return Err(format!("native max_ctx {max_ctx} outside 1..=262144"));
    }
    if positions.len() != rows {
        return Err(format!(
            "native positions length {} != rows {rows}",
            positions.len()
        ));
    }
    for &p in positions {
        if p < 0 || (p as usize) >= max_ctx {
            return Err(format!("native position {p} outside 0..{max_ctx}"));
        }
    }
    let need_q = rows * Q_BYTES_ROW;
    if q_bytes < need_q {
        return Err(format!("native Q extent {q_bytes} < {need_q}"));
    }
    let need_cache = max_ctx * POS_BYTES;
    if cache_bytes < need_cache {
        return Err(format!("native cache extent {cache_bytes} < {need_cache}"));
    }
    Ok(())
}

// ------------------------------------------------------------- comparison
#[derive(Clone, Copy)]
enum Elem {
    F32,
    F16,
}
impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 => 2,
        }
    }
    fn at(self, b: &[u8], i: usize) -> (u32, f64) {
        match self {
            Elem::F32 => {
                let bits = u32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
                (bits, f32::from_bits(bits) as f64)
            }
            Elem::F16 => {
                let bits = u16::from_ne_bytes(b[i * 2..i * 2 + 2].try_into().unwrap());
                (bits as u32, half::f16::from_bits(bits).to_f32() as f64)
            }
        }
    }
}
fn coords(idx: usize) -> Value {
    json!({"index":idx,"row":idx / ROW,"head":(idx / D) % H,"dim":idx % D})
}
/// Compare `a` (HIP twin) with `b` (PM) as `elem` arrays of `[row, H, D]`. `nan_class` classifies NaN codes
/// separately: NaN positions must agree and every non-NaN element be bit-identical.
fn compare(a: &[u8], b: &[u8], elem: Elem, nan_class: bool) -> Value {
    if a.len() != b.len() || a.len() % elem.size() != 0 {
        return json!({"pass":false,"byte_identical":false,"error":"length mismatch","len_a":a.len(),"len_b":b.len()});
    }
    let n = a.len() / elem.size();
    let identical = a == b;
    let (mut nf_a, mut nf_b, mut nan_a, mut nan_b) = (0usize, 0usize, 0usize, 0usize);
    let (mut count, mut first) = (0usize, None::<usize>);
    let (mut nan_mismatch, mut value_mismatch) = (0usize, 0usize);
    for i in 0..n {
        let (ba, va) = elem.at(a, i);
        let (bb, vb) = elem.at(b, i);
        nf_a += !va.is_finite() as usize;
        nf_b += !vb.is_finite() as usize;
        nan_a += va.is_nan() as usize;
        nan_b += vb.is_nan() as usize;
        if ba != bb {
            count += 1;
            first.get_or_insert(i);
        }
        if va.is_nan() != vb.is_nan() {
            nan_mismatch += 1;
        } else if !va.is_nan() && ba != bb {
            value_mismatch += 1;
        }
    }
    let finite = nf_a == 0 && nf_b == 0;
    let first_mismatch = first.map(|i| {
        let mut c = coords(i);
        c["hip_bits"] = json!(format!("{:#x}", elem.at(a, i).0));
        c["pm_bits"] = json!(format!("{:#x}", elem.at(b, i).0));
        c
    });
    let pass = if nan_class {
        nan_mismatch == 0 && value_mismatch == 0 && nan_a > 0
    } else {
        identical && finite
    };
    json!({"elements":n,"element_type":match elem {Elem::F32=>"f32",Elem::F16=>"f16"},
        "byte_identical":identical,"mismatched_elements":count,"first_mismatch":first_mismatch,
        "nonfinite_hip":nf_a,"nonfinite_pm":nf_b,"nan_hip":nan_a,"nan_pm":nan_b,
        "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
        "classification":if nan_class {"nan_codes_separate"} else {"finite_byte_gate"},"pass":pass})
}
fn pass(v: &Value) -> bool {
    v["pass"] == true
}

// ------------------------------------------------------------------ rng
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn normal(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
}

// ----------------------------------------------------------- GPU buffers
struct Buf {
    t: GpuTensor,
    es: usize,
    name: &'static str,
}
impl Buf {
    fn new(gpu: &mut Gpu, name: &'static str, bytes: usize, dtype: DType) -> Result<Self> {
        let es = dtype.size();
        assert_eq!(bytes % es, 0);
        let t = gpu.alloc_tensor(&[(bytes + 2 * GUARD) / es], dtype)?;
        Ok(Self { t, es, name })
    }
    fn sub(&self, off: usize, len: usize) -> GpuTensor {
        self.t.sub_offset((GUARD + off) / self.es, len / self.es)
    }
    fn view(&self, len: usize) -> GpuTensor {
        self.sub(0, len)
    }
    fn ptr(&self, off: usize) -> *mut c_void {
        // SAFETY: offset stays inside the allocation (checked by cell sizing).
        unsafe {
            self.t
                .buf
                .as_ptr()
                .cast::<u8>()
                .add(GUARD + off)
                .cast::<c_void>() as *mut c_void
        }
    }
    fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        gpu.hip.memcpy_htod_offset(&self.t.buf, GUARD, data)?;
        Ok(())
    }
    fn download_at(&self, gpu: &Gpu, off: usize, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        gpu.hip.memcpy_dtoh_at(&mut v, &self.t.buf, GUARD + off)?;
        Ok(v)
    }
    fn download(&self, gpu: &Gpu, len: usize) -> Result<Vec<u8>> {
        self.download_at(gpu, 0, len)
    }
    fn poison(&self, gpu: &Gpu, len: usize) -> Result<()> {
        let v = self.view(len);
        gpu.hip.memset(&v.buf, POISON, len)?;
        Ok(())
    }
    fn pattern(&self, side: u8) -> Vec<u8> {
        (0..GUARD)
            .map(|i| (i as u8).wrapping_mul(131) ^ 0xC3 ^ side ^ (self.name.len() as u8))
            .collect()
    }
    fn arm_guards(&self, gpu: &Gpu, used: usize) -> Result<()> {
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, 0, &self.pattern(1))?;
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, GUARD + used, &self.pattern(2))?;
        Ok(())
    }
    fn guards_ok(&self, gpu: &Gpu, used: usize) -> Result<bool> {
        let mut lo = vec![0u8; GUARD];
        let mut hi = vec![0u8; GUARD];
        gpu.hip.memcpy_dtoh_at(&mut lo, &self.t.buf, 0)?;
        gpu.hip.memcpy_dtoh_at(&mut hi, &self.t.buf, GUARD + used)?;
        Ok(lo == self.pattern(1) && hi == self.pattern(2))
    }
}
struct Arena {
    b: Vec<Buf>,
}
impl Arena {
    fn new(gpu: &mut Gpu, cells: &[Cell]) -> Result<Self> {
        let sizes = arena_sizes(cells);
        let dtypes = [
            DType::F32,
            DType::F32,
            DType::F32,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::F32,
            DType::F32,
        ];
        let mut b = Vec::new();
        for i in 0..12 {
            b.push(Buf::new(gpu, BUFFER_NAMES[i], sizes[i], dtypes[i])?);
        }
        Ok(Self { b })
    }
    fn q32(&self) -> &Buf {
        &self.b[0]
    }
    fn k32(&self) -> &Buf {
        &self.b[1]
    }
    fn v32(&self) -> &Buf {
        &self.b[2]
    }
    fn allpos(&self) -> &Buf {
        &self.b[3]
    }
    fn kc(&self) -> &Buf {
        &self.b[4]
    }
    fn vc(&self) -> &Buf {
        &self.b[5]
    }
    fn pos(&self) -> &Buf {
        &self.b[6]
    }
    fn q16_hip(&self) -> &Buf {
        &self.b[7]
    }
    fn q16_pm(&self) -> &Buf {
        &self.b[8]
    }
    fn out_hip(&self) -> &Buf {
        &self.b[9]
    }
    fn out_pm(&self) -> &Buf {
        &self.b[10]
    }
    fn out_extra(&self) -> &Buf {
        &self.b[11]
    }
}

// ----------------------------------------------------------------- arms
fn hip_pre_seg(gpu: &Gpu, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    let mut args = pre_kernargs(a.q32().ptr(off * ROW * 4), q16.ptr(off * ROW * 2), n);
    gpu.launch_kernel_blob(
        HIP_PRE,
        pre_grid(n),
        [PRE_BLOCK, 1, 1],
        0,
        args.as_mut_slice(),
    )?;
    Ok(())
}
/// Native arena, disjoint from legacy `Arena`: Raw Q two-plane, native 516B
/// K/V rows, positions, three F32 outputs and host-fixture F32 Q. All eight
/// buffers retain both-side canaries and are allocated once per family.
struct NativeArena {
    b: Vec<Buf>,
}
const NATIVE_BUFFER_NAMES: [&str; 8] = [
    "native_q_raw",
    "native_k_cache",
    "native_v_cache",
    "native_positions",
    "native_out_cand",
    "native_out_ref",
    "native_out_extra",
    "native_q_f32",
];
impl NativeArena {
    fn new(gpu: &mut Gpu, cells: &[NativeCell]) -> Result<Self> {
        let mut q = 0usize;
        let mut cache = 0usize;
        let mut pos = 0usize;
        let mut out = 0usize;
        for c in cells {
            // FULL two-plane Q includes a defined 4-row overhang to expose
            // tail-local scale-plane bugs. Cache allocation covers exactly
            // the cell's active prefix; future keys are inside that prefix.
            let total = c.rows + 4;
            q = q.max(NativeCell::q_bytes(total));
            cache = cache.max(NativeCell::cache_bytes(c.ctx()));
            pos = pos.max(c.rows * 4);
            out = out.max(c.rows * ROW * 4);
        }
        let q_f32 = cells.iter().map(|c| c.rows * ROW * 4).max().unwrap_or(0);
        let sizes = [q, cache, cache, pos, out, out, out, q_f32];
        let dtypes = [
            DType::Raw,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::F32,
            DType::F32,
            DType::F32,
            DType::F32,
        ];
        let mut b = Vec::new();
        for i in 0..NATIVE_BUFFER_NAMES.len() {
            b.push(Buf::new(gpu, NATIVE_BUFFER_NAMES[i], sizes[i], dtypes[i])?);
        }
        Ok(Self { b })
    }
    fn sizes(&self) -> [usize; 8] {
        // Exact allocation census excludes guards (matches `plan` style).
        let mut m = [0usize; 8];
        for (slot, b) in m.iter_mut().zip(self.b.iter()) {
            *slot = b.t.buf.size() - 2 * GUARD;
        }
        m
    }
    fn q(&self) -> &Buf {
        &self.b[0]
    }
    fn kc(&self) -> &Buf {
        &self.b[1]
    }
    fn vc(&self) -> &Buf {
        &self.b[2]
    }
    fn pos(&self) -> &Buf {
        &self.b[3]
    }
    fn out_cand(&self) -> &Buf {
        &self.b[4]
    }
    fn out_ref(&self) -> &Buf {
        &self.b[5]
    }
    fn out_extra(&self) -> &Buf {
        &self.b[6]
    }
    fn q_f32(&self) -> &Buf {
        &self.b[7]
    }
}
fn hip_direct_seg(gpu: &Gpu, a: &Arena, q16: &Buf, out: &Buf, off: usize, n: usize) -> Result<()> {
    let mut args = direct_kernargs(
        q16.ptr(off * ROW * 2),
        a.kc().ptr(0),
        a.vc().ptr(0),
        out.ptr(off * ROW * 4),
        a.pos().ptr(off * 4),
        n,
    );
    gpu.launch_kernel_blob(
        HIP_DIRECT,
        direct_grid(n),
        [DIRECT_BLOCK, 1, 1],
        DIRECT_LDS,
        args.as_mut_slice(),
    )?;
    Ok(())
}
fn pm_pre_seg(gpu: &mut Gpu, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    gpu.fa_prefill_pm_preconvert(
        &a.q32().sub(off * ROW * 4, n * ROW * 4),
        &q16.sub(off * ROW * 2, n * ROW * 2),
        n,
    )?;
    Ok(())
}
fn pm_direct_seg(
    gpu: &mut Gpu,
    a: &Arena,
    c: &Cell,
    q16: &Buf,
    out: &Buf,
    off: usize,
    n: usize,
) -> Result<()> {
    let u = used_of(c);
    gpu.fa_prefill_pm_direct_codec(
        FaPrefillKvCodec::Fp8E4M3,
        &q16.sub(off * ROW * 2, n * ROW * 2),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &out.sub(off * ROW * 4, n * ROW * 4),
        &a.pos().sub(off * 4, n * 4),
        c.ctx(),
        n,
    )?;
    Ok(())
}
fn run_hip(gpu: &Gpu, a: &Arena, c: &Cell) -> Result<()> {
    for (off, n) in c.segments() {
        hip_pre_seg(gpu, a, a.q16_hip(), off, n)?;
        hip_direct_seg(gpu, a, a.q16_hip(), a.out_hip(), off, n)?;
    }
    Ok(())
}
fn run_pm(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<()> {
    for (off, n) in c.segments() {
        pm_pre_seg(gpu, a, a.q16_pm(), off, n)?;
        pm_direct_seg(gpu, a, c, a.q16_pm(), a.out_pm(), off, n)?;
    }
    Ok(())
}
/// Re-run only the PM direct (on the already converted `q16_pm`) into `out`.
fn run_pm_direct(gpu: &mut Gpu, a: &Arena, c: &Cell, out: &Buf) -> Result<()> {
    for (off, n) in c.segments() {
        pm_direct_seg(gpu, a, c, a.q16_pm(), out, off, n)?;
    }
    Ok(())
}
/// Native candidate direct launch: one segment of `q` (FULL two-plane) via the
/// NEW production wrapper (B owner), strict spans from `a3b_qresident_spans`
/// (codes `row_offset * 4096`, scales `total * 4096 + row_offset * 64`).
/// FAILS CLOSED on any wrapper error: the caller records `pass: false`, never
/// skips Green.
fn native_candidate_seg(
    gpu: &mut Gpu,
    a: &NativeArena,
    cell: &NativeCell,
    out: &Buf,
    off: usize,
    n: usize,
    total: usize,
) -> Result<()> {
    let rows = n;
    let seg_out = out.sub(off * ROW * 4, rows * ROW * 4);
    let seg_pos = a.pos().sub(off * 4, rows * 4);
    if cell.rows == total {
        gpu.attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201(
            &a.q().view(NativeCell::q_bytes(total)),
            &a.kc().view(NativeCell::cache_bytes(cell.ctx())),
            &a.vc().view(NativeCell::cache_bytes(cell.ctx())),
            &seg_out,
            &seg_pos,
            H,
            KV,
            D,
            cell.ctx(),
            rows,
        )?;
    } else {
        gpu.attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201(
            &a.q().view(NativeCell::q_bytes(total)),
            &a.kc().view(NativeCell::cache_bytes(cell.ctx())),
            &a.vc().view(NativeCell::cache_bytes(cell.ctx())),
            &seg_out,
            &seg_pos,
            H,
            KV,
            D,
            cell.ctx(),
            rows,
            total,
            off,
        )?;
    }
    Ok(())
}
/// Independent one-query/head wave32 reference: grid [n*16,1,1], block32,
/// LDS0. Six-pointer/4xi32/F32 (68B) ABI; range offsets use FULL-total planes.
/// Any binding/launch error fails the requested family.
fn native_reference_seg(
    gpu: &Gpu,
    a: &NativeArena,
    cell: &NativeCell,
    out: &Buf,
    off: usize,
    n: usize,
    total: usize,
) -> Result<()> {
    let q_off = off * Q_CODES_ROW;
    let s_off = total * Q_CODES_ROW + off * Q_SCALES_ROW;
    let symbol = if cell.rows == total {
        NATIVE_REF_DIRECT
    } else {
        NATIVE_REF_RANGE
    };
    let grid = [(n * H) as u32, 1, 1];
    let mut args = native_kernargs(
        a.q().ptr(q_off),
        a.q().ptr(s_off),
        a.kc().ptr(0),
        a.vc().ptr(0),
        out.ptr(off * ROW * 4),
        a.pos().ptr(off * 4),
        n,
    );
    gpu.launch_kernel_blob(symbol, grid, [32, 1, 1], 0, args.as_mut_slice())?;
    Ok(())
}
fn run_native_candidate(
    gpu: &mut Gpu,
    a: &NativeArena,
    cell: &NativeCell,
    total: usize,
    segs: &[(usize, usize)],
    out: &Buf,
) -> Result<()> {
    for &(off, n) in segs {
        native_candidate_seg(gpu, a, cell, out, off, n, total)?;
    }
    Ok(())
}
fn run_native_reference(
    gpu: &Gpu,
    a: &NativeArena,
    cell: &NativeCell,
    total: usize,
    segs: &[(usize, usize)],
    out: &Buf,
) -> Result<()> {
    for &(off, n) in segs {
        native_reference_seg(gpu, a, cell, out, off, n, total)?;
    }
    Ok(())
}

// ------------------------------------------------------------ cell data
struct Prepared {
    qh: Vec<f32>,
    pos: Vec<i32>,
    kc: Vec<u8>,
    vc: Vec<u8>,
    used: Used,
    fixture: Fixture,
    seed: u64,
}
/// Pure host admission, before any launch.
fn validate_cell(rows: usize, positions: &[i32], ctx: usize) -> std::result::Result<(), String> {
    if rows < 1 || rows > MAX_ROWS {
        return Err(format!("rows {rows} outside 1..={MAX_ROWS}"));
    }
    if positions.len() != rows {
        return Err(format!(
            "positions length {} != rows {rows}",
            positions.len()
        ));
    }
    fa_prefill_pm::validate_positions(positions, ctx).map_err(|e| e.to_string())?;
    if rows >= fa_prefill_pm::MIN_ROWS {
        fa_prefill_pm::validate_domain(H, KV, D, rows, ctx).map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn prepare(gpu: &mut Gpu, a: &Arena, c: &Cell, round: usize) -> Result<Prepared> {
    let u = used_of(c);
    let ctx = c.ctx();
    let fixture = c.fixture.for_round(round);
    let pos = fixture.positions(c.rows, c.start);
    validate_cell(c.rows, &pos, ctx)?;
    let seed = fnv(&c.name()) ^ (0x41a3b8f8u64 + (round as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let mut rng = Rng(seed);
    let qh: Vec<f32> = (0..c.rows * ROW).map(|_| rng.normal()).collect();
    for (buf, bytes_used) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, bytes_used)?;
    }
    a.q32().upload(gpu, bytes(&qh))?;
    a.pos().upload(gpu, bytes(&pos))?;
    match c.source {
        Source::Runtime => {
            let kf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal()).collect();
            let vf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal() * 0.7).collect();
            let allpos: Vec<i32> = (0..ctx as i32).collect();
            a.k32().upload(gpu, bytes(&kf))?;
            a.v32().upload(gpu, bytes(&vf))?;
            a.allpos().upload(gpu, bytes(&allpos))?;
            gpu.kv_cache_write_fp8_e4m3_batched(
                &a.kc().view(u.kc),
                &a.k32().view(u.k32),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?;
            gpu.kv_cache_write_fp8_e4m3_batched(
                &a.vc().view(u.vc),
                &a.v32().view(u.v32),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?;
            gpu.hip.device_synchronize()?;
        }
        Source::Grid | Source::Nan => {
            let nan = c.source == Source::Nan;
            a.kc().upload(gpu, &grid_cache(ctx, 0, nan))?;
            a.vc().upload(gpu, &grid_cache(ctx, 5, nan))?;
            gpu.hip.device_synchronize()?;
        }
    }
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for b in [
        a.q16_hip(),
        a.q16_pm(),
        a.out_hip(),
        a.out_pm(),
        a.out_extra(),
    ] {
        let len = if b.es == 4 { u.out } else { u.q16 };
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        qh,
        pos,
        kc,
        vc,
        used: u,
        fixture,
        seed,
    })
}
fn integrity(gpu: &Gpu, a: &Arena, p: &Prepared) -> Result<Value> {
    let u = &p.used;
    let inputs = json!({
        "q32": a.q32().download(gpu, u.q32)? == bytes(&p.qh),
        "k_cache": a.kc().download(gpu, u.kc)? == p.kc,
        "v_cache": a.vc().download(gpu, u.vc)? == p.vc,
        "positions": a.pos().download(gpu, u.pos)? == bytes(&p.pos),
    });
    let mut guards = serde_json::Map::new();
    for (b, n) in a.b.iter().zip(used_list(u)) {
        guards.insert(b.name.into(), json!(b.guards_ok(gpu, n)?));
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    Ok(
        json!({"inputs_unchanged":inputs,"guards_intact":guards,"guard_bytes_each_side":GUARD,"pass":ok}),
    )
}
// --------------------------------------- independent round-trip (after prepare, before integrity)
/// Upload independently constructed host inputs, arm every existing guard, poison every
/// output/Q16 buffer, and return the original host bytes in `Prepared`. Reuses the existing
/// launchers and `integrity`; never uses random `prepare` as the source of expected values.
/// The Q16 input is the frozen case's own Q16 words (f32 host upload of the exact words).
fn prepare_ieee(gpu: &mut Gpu, a: &Arena, ic: &IeeeCase) -> Result<Prepared> {
    let rows = ic.positions.len();
    let npos = ic.k_cache.len() / POS_BYTES;
    validate_cell(rows, &ic.positions, npos)?;
    let u = used(rows, npos);
    for (buf, bytes_used) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, bytes_used)?;
    }
    let qh_f32: Vec<f32> = ic.qh.iter().map(|&w| f16_to_f32_ieee(w)).collect();
    // The hardware preconvert reads f32 Q; upload the exact f32 values of the frozen Q16 words.
    a.q32().upload(gpu, bytes(&qh_f32))?;
    a.pos().upload(gpu, bytes(&ic.positions))?;
    a.kc().upload(gpu, &ic.k_cache)?;
    a.vc().upload(gpu, &ic.v_cache)?;
    gpu.hip.device_synchronize()?;
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for b in [
        a.q16_hip(),
        a.q16_pm(),
        a.out_hip(),
        a.out_pm(),
        a.out_extra(),
    ] {
        let len = if b.es == 4 { u.out } else { u.q16 };
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        qh: qh_f32,
        pos: ic.positions.clone(),
        kc,
        vc,
        used: u,
        fixture: Fixture::Std,
        seed: fnv(&ic.name),
    })
}
/// Native prepared host state: upstream-F32 Q rows, the independently produced
/// two-plane Raw Q (codes + F32 scales), host K/V caches, positions, and the
/// independent expected outputs for exact fixtures.
struct NativePrepared {
    q_f32: Vec<f32>,
    q_codes: Vec<u8>,
    q_scales: Vec<f32>,
    positions: Vec<i32>,
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    expected: Vec<f32>,
    total: usize,
    used_q: usize,
    used_cache: usize,
    used_pos: usize,
    used_out: usize,
    seed: u64,
}
/// Native host admission + fixture build (pure host, before any launch).
/// FAILS CLOSED: any invalid shape/range/dtype/span returns `Err` (recorded as
/// `pass: false`, never skipped).
fn native_prepare(cell: &NativeCell, round: usize) -> Result<NativePrepared> {
    let rows = cell.rows;
    let ctx = cell.ctx();
    let total = rows + 4;
    let seed =
        fnv(&cell.name()) ^ (0xa3b00000u64 + (round as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let mut rng = Rng(seed);
    let mut q_f32: Vec<f32> = match cell.kind {
        // Exact byte-gate fixtures use uniform all-448 Q rows: amax is exactly
        // 448, so the prep scale is exactly 1.0 and every code is exactly 0x7e.
        // Scores become exactly (448 x 448 x 256)/16 vs exactly 0: the gap is
        // thousands, every loser weight underflows to exactly +0 in any
        // faithful expf, and the P/V/O chain is bit-exact (see builder docs).
        NativeKind::Kt48 | NativeKind::Winner => {
            vec![448.0; rows * ROW]
        }
        _ => (0..rows * ROW)
            .map(|i| {
                let (r, h) = (i / ROW, (i / D) % H);
                (rng.normal() + (r % 7) as f32 / 37.0) * (1.0 + h as f32 / 11.0)
            })
            .collect(),
    };
    if matches!(cell.kind, NativeKind::QNan | NativeKind::QInf) {
        let (positive, negative) = if cell.kind == NativeKind::QInf {
            (f32::INFINITY, f32::NEG_INFINITY)
        } else {
            (f32::NAN, -f32::NAN)
        };
        q_f32[0] = positive;
        q_f32[D + 7] = negative;
        q_f32[(2 * H + 3) * D + 10] = positive;
        q_f32[(5 * H + 8) * D + 200] = negative;
        q_f32[(5 * H + 8) * D + 201] = positive;
        if cell.kind == NativeKind::QNan {
            for i in 5 * ROW..6 * ROW {
                q_f32[i] = if i % 2 == 0 { positive } else { negative };
            }
        }
    }
    // Positions: exact-gate cells pin every query's horizon at `start` (the
    // winner key; every row sees the same causal prefix).
    // Non-pinned cells use the standard 8-wide fixture pattern; round 1
    // reverses it (second DISTINCT position round, same existing pattern).
    let mut positions = Fixture::Std.for_round(round).positions(rows, cell.start);
    if matches!(cell.kind, NativeKind::Kt48 | NativeKind::Winner) {
        for p in positions.iter_mut() {
            *p = cell.start as i32;
        }
    }
    // Keep future keys available for causal probes, including pinned winners.
    let max_ctx = ctx;
    let (k_cache, v_cache) = match cell.kind {
        NativeKind::Kt48 | NativeKind::Winner => {
            let win = (positions[0] as usize).min(max_ctx.saturating_sub(1));
            native_winner_kv(max_ctx, win, 0x5300)
        }
        NativeKind::Nan => (native_cache(max_ctx, 3), native_cache(max_ctx, 9)),
        _ => (
            native_cache(max_ctx, round * 13),
            native_cache(max_ctx, 7 + round * 13),
        ),
    };
    let mut k_cache = k_cache;
    let mut v_cache = v_cache;
    if cell.kind == NativeKind::Nan {
        // K NaNs at key3 taint all dimensions when causally live.
        // V NaNs at always-live key0 taint only their dimensions, including
        // low-horizon rows. Finite unaffected bytes remain meaningful.
        if max_ctx > 4 {
            k_cache[3 * POS_BYTES + 5] = 0x7f;
            k_cache[3 * POS_BYTES + 1 * D + 200] = 0xff;
            v_cache[0] = 0x7f;
            v_cache[D + 255] = 0xff;
        }
    }
    // Host-built attention inputs, NOT evidence that the fused producer ran.
    // Quantize every D256 head independently, including the FULL-plane tail.
    let (mut q_codes, mut q_scales) = prep_q_heads(&q_f32);
    for r in rows..total {
        let fill: Vec<f32> = (0..ROW)
            .map(|i| {
                (((r * 7 + i * 13) % 251) as f32 / 251.0 - 0.5) * (1.0 + (i / D) as f32 / 13.0)
            })
            .collect();
        let (codes, scales) = prep_q_heads(&fill);
        q_codes.extend_from_slice(&codes);
        q_scales.extend_from_slice(&scales);
    }
    let used_q = NativeCell::q_bytes(total);
    let used_cache = NativeCell::cache_bytes(max_ctx);
    let used_pos = rows * 4;
    let used_out = rows * ROW * 4;
    native_validate(rows, &positions, max_ctx, used_q, used_cache)?;
    let expected = match cell.kind {
        NativeKind::Winner | NativeKind::Kt48 => native_expected(&k_cache, &positions),
        // K key3: all words when live. V key0: dim0 of KVhead0 and
        // dim255 of KVhead1 for every query. Other words must be finite
        // and byte-identical to the independent quantized HIP reference.
        NativeKind::Nan => positions
            .iter()
            .flat_map(|&pos| {
                (0..ROW).map(move |i| {
                    let (h, d) = (i / D, i % D);
                    if pos >= 3 || (h < 8 && d == 0) || (h >= 8 && d == 255) {
                        f32::NAN
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
        // INPUT-TAINT markers for Q heads poisoned BEFORE prep (values kept
        // byte-identical): the scalar DAG derives a strict +0 OUTPUT
        // requirement on finite K/V, not a required-output-NaN. Bytes kept.
        NativeKind::QNan | NativeKind::QInf => q_f32
            .chunks_exact(D)
            .flat_map(|head| {
                let poisoned = head.iter().any(|v| !v.is_finite());
                std::iter::repeat(if poisoned { f32::NAN } else { 0.0 }).take(D)
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok(NativePrepared {
        q_f32,
        q_codes,
        q_scales,
        positions,
        k_cache,
        v_cache,
        expected,
        total,
        used_q,
        used_cache,
        used_pos,
        used_out,
        seed,
    })
}
/// Upload host attention fixtures with both-side guards and native finite
/// output poisons. This upload is not the separate GPU producer proof.
fn native_upload(gpu: &mut Gpu, a: &NativeArena, p: &NativePrepared, round: usize) -> Result<()> {
    for (buf, used) in [
        (a.q(), p.used_q),
        (a.kc(), p.used_cache),
        (a.vc(), p.used_cache),
        (a.pos(), p.used_pos),
        (a.out_cand(), p.used_out),
        (a.out_ref(), p.used_out),
        (a.out_extra(), p.used_out),
        (a.q_f32(), p.q_f32.len() * 4),
    ] {
        buf.arm_guards(gpu, used)?;
    }
    let mut q_plane = p.q_codes.clone();
    q_plane.extend_from_slice(bytes(&p.q_scales));
    debug_assert_eq!(q_plane.len(), p.used_q);
    a.q().upload(gpu, &q_plane)?;
    a.kc().upload(gpu, &p.k_cache)?;
    a.vc().upload(gpu, &p.v_cache)?;
    a.pos().upload(gpu, bytes(&p.positions))?;
    a.q_f32().upload(gpu, bytes(&p.q_f32))?;
    native_output_poison(gpu, a.out_cand(), p.used_out, round)?;
    native_output_poison(gpu, a.out_ref(), p.used_out, 1 - round)?;
    native_output_poison(gpu, a.out_extra(), p.used_out, round)?;
    gpu.hip.device_synchronize()?;
    Ok(())
}
/// Native input/guard integrity: all inputs unchanged, all eight native
/// buffer canaries intact, including after perturbed-cache probe launches.
fn native_integrity(
    gpu: &Gpu,
    a: &NativeArena,
    p: &NativePrepared,
    kc: &[u8],
    vc: &[u8],
) -> Result<Value> {
    let mut q_plane = p.q_codes.clone();
    q_plane.extend_from_slice(bytes(&p.q_scales));
    let inputs = json!({
        "q_two_plane": a.q().download(gpu, p.used_q)? == q_plane,
        "k_cache": a.kc().download(gpu, p.used_cache)? == kc,
        "v_cache": a.vc().download(gpu, p.used_cache)? == vc,
        "positions": a.pos().download(gpu, p.used_pos)? == bytes(&p.positions),
        "q_f32": a.q_f32().download(gpu, p.q_f32.len() * 4)? == bytes(&p.q_f32),
    });
    let mut guards = serde_json::Map::new();
    for (b, n) in [
        (a.q(), p.used_q),
        (a.kc(), p.used_cache),
        (a.vc(), p.used_cache),
        (a.pos(), p.used_pos),
        (a.out_cand(), p.used_out),
        (a.out_ref(), p.used_out),
        (a.out_extra(), p.used_out),
        (a.q_f32(), p.q_f32.len() * 4),
    ] {
        guards.insert(b.name.into(), json!(b.guards_ok(gpu, n)?));
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    Ok(json!({"inputs_unchanged": inputs, "guards_intact": guards,
        "guard_bytes_each_side": GUARD, "pass": ok}))
}
/// Independent classifier over raw output bytes: affected NaN positions must be NaN on both
/// arms with HIP/PM NaN-position agreement, and every unaffected word must match the
/// independent clean expectation byte for byte (separately for HIP and PM downloads).
/// Candidate/HIP downloads never construct expected output. Zero-valued independent fixtures
/// are valid: no `out_nonzero` gate here.
fn compare_ieee_words(expected: &IeeeExpected, got_words: &[u32]) -> Value {
    match expected {
        IeeeExpected::Finite(want) => {
            let mut mism = 0usize;
            let mut first = None::<usize>;
            for (i, (&g, &w)) in got_words.iter().zip(want.iter()).enumerate() {
                if g != w {
                    mism += 1;
                    first.get_or_insert(i);
                }
            }
            let pass = mism == 0 && got_words.len() == want.len();
            json!({"classification":"finite_byte_gate","words":got_words.len(),"mismatched_words":mism,
                "first_mismatch":first,"pass":pass})
        }
        IeeeExpected::VNaN {
            clean_expected_words,
            kv_head,
            dimension,
        } => {
            let (mut nan_mismatch, mut value_mismatch, mut affected_nan) = (0usize, 0usize, 0usize);
            let mut first = None::<usize>;
            for (i, (&g, &w)) in got_words
                .iter()
                .zip(clean_expected_words.iter())
                .enumerate()
            {
                let h = (i / D) % H;
                let d = i % D;
                let affected = h / 8 == *kv_head && d == *dimension;
                let gn = f32::from_bits(g).is_nan();
                if affected {
                    affected_nan += gn as usize;
                    if !gn {
                        nan_mismatch += 1;
                        first.get_or_insert(i);
                    }
                } else if g != w {
                    value_mismatch += 1;
                    first.get_or_insert(i);
                }
            }
            let pass = nan_mismatch == 0 && value_mismatch == 0 && affected_nan > 0;
            json!({"classification":"v_nan_separate","words":got_words.len(),"affected_nan":affected_nan,
                "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
                "first_mismatch":first,"pass":pass})
        }
        IeeeExpected::KNaN {
            clean_expected_words,
            kv_head,
        } => {
            let rows = got_words.len() / ROW;
            let (mut affected_rows_with_nan, mut affected_rows) = (0usize, 0usize);
            let (mut nan_mismatch, mut value_mismatch) = (0usize, 0usize);
            let mut first = None::<usize>;
            for r in 0..rows {
                for h in 0..H {
                    let got = &got_words[(r * H + h) * D..(r * H + h + 1) * D];
                    let want = &clean_expected_words[(r * H + h) * D..(r * H + h + 1) * D];
                    if h / 8 == *kv_head {
                        affected_rows += 1;
                        if got.iter().any(|&w| f32::from_bits(w).is_nan()) {
                            affected_rows_with_nan += 1;
                        } else {
                            nan_mismatch += 1;
                            first.get_or_insert((r * H + h) * D);
                        }
                    } else if got != want {
                        value_mismatch +=
                            got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                        first.get_or_insert((r * H + h) * D);
                    }
                }
            }
            let pass = nan_mismatch == 0
                && value_mismatch == 0
                && affected_rows_with_nan == affected_rows
                && affected_rows > 0;
            json!({"classification":"k_nan_separate","words":got_words.len(),
                "affected_rows_with_nan":affected_rows_with_nan,"affected_rows":affected_rows,
                "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
                "first_mismatch":first,"pass":pass})
        }
    }
}
/// Compare independently expected words against one arm's raw bytes.
#[allow(dead_code)]
fn compare_ieee(expected: &IeeeExpected, got_bytes: &[u8]) -> Result<Value> {
    if got_bytes.len() % 4 != 0 {
        return Err("independent output bytes must be whole f32 words".into());
    }
    let words: Vec<u32> = got_bytes
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    Ok(compare_ieee_words(expected, &words))
}
/// One independent round: upload the frozen case, run HIP and PM primitives against the
/// independent expectation separately, then retain the existing mutual HIP/PM classification
/// and byte comparison. Guards/poison/changed-input/original-matrix semantics are preserved
/// by reusing `integrity` and the guarded arena.
fn ieee_round(gpu: &mut Gpu, a: &Arena, ic: &IeeeCase) -> Result<Value> {
    let rows = ic.positions.len();
    let npos = ic.k_cache.len() / POS_BYTES;
    let p = prepare_ieee(gpu, a, ic)?;
    let u = p.used;
    // HIP primitive on the independent buffers.
    hip_pre_seg(gpu, a, a.q16_hip(), 0, rows)?;
    {
        let mut args = direct_kernargs(
            a.q16_hip().ptr(0),
            a.kc().ptr(0),
            a.vc().ptr(0),
            a.out_hip().ptr(0),
            a.pos().ptr(0),
            rows,
        );
        gpu.launch_kernel_blob(
            HIP_DIRECT,
            direct_grid(rows),
            [DIRECT_BLOCK, 1, 1],
            DIRECT_LDS,
            args.as_mut_slice(),
        )?;
    }
    // PM primitive on the independent buffers (whole: independent rows are <= 8, one segment).
    pm_pre_seg(gpu, a, a.q16_pm(), 0, rows)?;
    gpu.fa_prefill_pm_direct_codec(
        FaPrefillKvCodec::Fp8E4M3,
        &a.q16_pm().view(u.q16),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &a.out_pm().view(u.out),
        &a.pos().view(u.pos),
        npos,
        rows,
    )?;
    gpu.hip.device_synchronize()?;
    let out_hip = a.out_hip().download(gpu, u.out)?;
    let out_pm = a.out_pm().download(gpu, u.out)?;
    // Independent expectation per arm, never constructed from candidate downloads: finite cases
    // use the stored words; NaN cases reuse the stored clean reference (the NaN-carrying full
    // expectation equals `ieee_expected` over the frozen caches, verified in self-check).
    let words_hip: Vec<u32> = out_hip
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    let words_pm: Vec<u32> = out_pm
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    let hip_independent = compare_ieee_words(&ic.expected, &words_hip);
    let pm_independent = compare_ieee_words(&ic.expected, &words_pm);
    // Existing mutual classification/byte comparison retained (NaN-aware for NaN cases).
    let nan_class = !matches!(ic.expected, IeeeExpected::Finite(_));
    let mutual = compare(&out_hip, &out_pm, Elem::F32, nan_class);
    let integ = integrity(gpu, a, &p)?;
    let ok = pass(&hip_independent) && pass(&pm_independent) && pass(&mutual) && pass(&integ);
    Ok(
        json!({"case":ic.name.clone(),"rows":rows,"positions":ic.positions.clone(),"allocated_ctx":npos,
        "assertion":ic.expected.kind(),"hip_vs_independent":hip_independent,"pm_vs_independent":pm_independent,
        "hip_vs_pm":mutual,"integrity":integ,"pass":ok}),
    )
}

/// Compare CPU per-head quantization of the downloaded GPU F32 baseline with
/// the ENTIRE downloaded Raw FP8 plane, including every F32 scale bit.
fn native_q_compare(baseline: &[f32], produced: &[u8], rows: usize) -> Value {
    if rows == 0
        || baseline.len() != rows * ROW
        || produced.len() != NativeCell::q_bytes(rows)
        || !baseline.iter().all(|v| v.is_finite())
    {
        return json!({"pass": false, "error": "nonfinite baseline or Q extent mismatch"});
    }
    let (codes, scales) = prep_q_heads(baseline);
    let actual_codes = &produced[..rows * Q_CODES_ROW];
    let actual_scales = &produced[rows * Q_CODES_ROW..];
    let code_mismatches = codes
        .iter()
        .zip(actual_codes)
        .filter(|(a, b)| a != b)
        .count();
    let scale_mismatches = bytes(&scales)
        .chunks_exact(4)
        .zip(actual_scales.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    json!({"classification": "actual_gpu_prep_quantization_gate",
        "code_bytes": codes.len(), "f32_scale_words": scales.len(),
        "code_mismatches": code_mismatches, "scale_word_mismatches": scale_mismatches,
        "scale_plane_offset": rows * Q_CODES_ROW,
        "pass": code_mismatches == 0 && scale_mismatches == 0})
}

fn native_output_poison(gpu: &Gpu, buf: &Buf, len: usize, round: usize) -> Result<()> {
    gpu.hip
        .memset(&buf.view(len).buf, NATIVE_POISONS[round], len)?;
    Ok(())
}

/// Real provider proof: two finite F32 source fixtures (the two poison
/// rounds), each with 17 QUERY ROWS, x F32 baseline / FP8 gate / FP8 nogate
/// = six launches. One eight-buffer LAB allocation is reused. EVERY provider
/// input is reset before EVERY launch, especially its in-place K input.
/// Norm/RoPE reuse exact source/ABI; CPU quantization/head layout is independent.
fn native_producer_proof(gpu: &mut Gpu) -> Result<Value> {
    const N: usize = 17;
    let qbytes = N * ROW * 4;
    let kbytes = N * KV * D * 4;
    let sizes = [
        2 * qbytes,
        kbytes,
        D * 4,
        D * 4,
        N * 4,
        qbytes,
        NativeCell::q_bytes(N),
        qbytes,
    ];
    let names = [
        "prep_interleaved",
        "prep_k",
        "prep_q_weight",
        "prep_k_weight",
        "prep_positions",
        "prep_baseline_f32",
        "prep_fp8_two_plane",
        "prep_gate",
    ];
    let mut b = Vec::with_capacity(sizes.len());
    for (i, (&name, &size)) in names.iter().zip(sizes.iter()).enumerate() {
        b.push(Buf::new(
            gpu,
            name,
            size,
            if i == 6 { DType::Raw } else { DType::F32 },
        )?);
    }
    let mut rounds = Vec::new();
    for round in 0..2 {
        let mut interleaved = vec![0.0f32; N * ROW * 2];
        let mut expected_gate = vec![0.0f32; N * ROW];
        for r in 0..N {
            for h in 0..H {
                for d in 0..D {
                    let idx = (r * H + h) * D + d;
                    let src = (r * H + h) * D * 2 + d;
                    // Head0/query0 is all-zero (scale1); every other head
                    // has F32-only variation and a different shape/amplitude.
                    interleaved[src] = if r == 0 && h == 0 {
                        0.0
                    } else {
                        (((r * 31 + h * 17 + d * 13 + round * 19) % 257) as f32 / 113.0 - 1.0)
                            * (1.0 + h as f32 / 23.0)
                    };
                    let gate = ((idx * 7 + round * 3) % 127) as f32 / 61.0 - 1.0;
                    interleaved[src + D] = gate;
                    expected_gate[idx] = gate;
                }
            }
        }
        let kin: Vec<f32> = (0..N * KV * D)
            .map(|i| ((i * 23 + round * 11) % 193) as f32 / 79.0 - 1.0)
            .collect();
        let qw: Vec<f32> = (0..D).map(|d| 0.6 + d as f32 / 317.0).collect();
        let kw: Vec<f32> = (0..D).map(|d| 1.1 - d as f32 / 701.0).collect();
        let positions: Vec<i32> = (0..N).map(|r| (63 + (r * 7) % 53) as i32).collect();
        for (buf, &size) in b.iter().zip(&sizes) {
            buf.arm_guards(gpu, size)?;
        }
        let mut baseline = Vec::new();
        let mut baseline_k = Vec::new();
        let mut baseline_gate = Vec::new();
        let mut modes = Vec::new();
        for mode in 0..3 {
            // Reset EVERY provider input from its original host bytes before
            // each arm, especially K (the provider mutates K in place).
            b[0].upload(gpu, bytes(&interleaved))?;
            b[1].upload(gpu, bytes(&kin))?;
            b[2].upload(gpu, bytes(&qw))?;
            b[3].upload(gpu, bytes(&kw))?;
            b[4].upload(gpu, bytes(&positions))?;
            let output = if mode == 0 { 5 } else { 6 };
            native_output_poison(gpu, &b[output], sizes[output], round)?;
            native_output_poison(gpu, &b[7], qbytes, round)?;
            let qt = b[output].view(sizes[output]);
            let gate = b[7].view(qbytes);
            gpu.qwen35_fa_prep_batched_gfx1201(
                &b[0].view(sizes[0]),
                if mode == 0 {
                    FaPrepQOut::F32(&qt)
                } else {
                    FaPrepQOut::Fp8Codes(&qt)
                },
                if mode == 2 { None } else { Some(&gate) },
                &b[1].view(kbytes),
                &b[2].view(D * 4),
                &b[3].view(D * 4),
                &b[4].view(N * 4),
                1.0e-6,
                1_000_000.0,
                11,
                H,
                KV,
                N,
            )?;
            gpu.hip.device_synchronize()?;
            let out = b[output].download(gpu, sizes[output])?;
            let kout = b[1].download(gpu, kbytes)?;
            let gout = b[7].download(gpu, qbytes)?;
            let inputs_ok = b[0].download(gpu, sizes[0])? == bytes(&interleaved)
                && b[2].download(gpu, sizes[2])? == bytes(&qw)
                && b[3].download(gpu, sizes[3])? == bytes(&kw)
                && b[4].download(gpu, sizes[4])? == bytes(&positions);
            let mut guards_ok = true;
            for (buf, &size) in b.iter().zip(&sizes) {
                guards_ok &= buf.guards_ok(gpu, size)?;
            }
            if mode == 0 {
                baseline = f32s(&out);
                baseline_k = kout;
                baseline_gate = gout;
                // Require exercised F32-only precision and varied per-head
                // scales; a trivial/unwritten baseline must fail, not skip.
                let nonhalf = baseline
                    .iter()
                    .filter(|&&x| {
                        x.is_finite() && half::f16::from_f32(x).to_f32().to_bits() != x.to_bits()
                    })
                    .count();
                let (_, scales) = prep_q_heads(&baseline);
                let varied = scales.iter().any(|s| s.to_bits() != scales[0].to_bits());
                let ok = inputs_ok
                    && guards_ok
                    && nonhalf > 0
                    && varied
                    && baseline.iter().all(|x| x.is_finite())
                    && f32s(&baseline_k).iter().all(|x| x.is_finite())
                    && baseline_gate == bytes(&expected_gate);
                modes.push(
                    json!({"mode":"F32_with_gate", "non_f16_exact_words":nonhalf,
                    "head_scales_varied":varied, "inputs_unchanged":inputs_ok,
                    "guards_intact":guards_ok, "pass":ok}),
                );
            } else {
                let quant = native_q_compare(&baseline, &out, N);
                let gate_ok = if mode == 1 {
                    gout == baseline_gate
                } else {
                    gout.iter().all(|&v| v == NATIVE_POISONS[round] as u8)
                };
                let baseline_unchanged = b[5].download(gpu, qbytes)? == bytes(&baseline);
                let ok = pass(&quant)
                    && gate_ok
                    && kout == baseline_k
                    && inputs_ok
                    && guards_ok
                    && baseline_unchanged;
                modes.push(
                    json!({"mode":if mode == 1 {"FP8_with_gate"} else {"FP8_nogate"},
                    "quantization":quant, "gate_copy_or_no_write":gate_ok,
                    "k_matches_f32_baseline":kout == baseline_k,
                    "baseline_unchanged":baseline_unchanged,
                    "inputs_unchanged":inputs_ok, "guards_intact":guards_ok, "pass":ok}),
                );
            }
        }
        let ok = modes.len() == 3 && modes.iter().all(pass);
        rounds.push(json!({"round":round,"fixture":"general_f32_varied_heads",
            "query_rows":N,"heads":H,
            "poison_byte":NATIVE_POISONS[round],"modes":modes,"pass":ok}));
    }
    let ok = rounds.len() == 2 && rounds.iter().all(pass);
    Ok(json!({"provider":"Gpu::qwen35_fa_prep_batched_gfx1201",
        "source":"kernels/src/qwen35_fa_prep_batched.gfx1201.hip",
        "source_sha256":sha_hex(NATIVE_PREP_BODY.as_bytes()),
        "kernarg_bytes":{"F32_gate":80,"FP8_gate":88,"FP8_nogate":80},
        "shape":{"q_heads":H,"kv_heads":KV,"head_dim":D,"n_rot":64,"rows":N},
        "eps":1.0e-6,
        "freq_base":1_000_000.0,"position_offset":11,
        "reset_before_every_launch":["Q_interleaved_including_gates","K_pre_norm","Q_norm_weight","K_norm_weight","positions"],
        "no_v_input":"this production prep ABI has no V argument",
        "normalization_rope":"exact existing source/ABI reuse, not independent algorithm proof",
        "quantizer":"independent CPU per-D256-head amax/448 E4M3 RNE + permute",
        "source_fixtures":2,"poison_rounds":2,
        "modes_per_fixture":["F32_gate","FP8_gate","FP8_nogate"],
        "query_rows_per_fixture":N,"launches":6,"rounds":rounds,"pass":ok}))
}

/// Every finite output gates against the independent quantized HIP reference.
/// Winner closed forms are additional checks; KV-NaN masks come from fixtures,
/// not from candidate output. Every unaffected word must be finite and exact.
/// Q-nonfinite (`QNan`/`QInf`) words carrying an INPUT-TAINT `expected` NaN
/// marker gate differently: the scalar DAG derives strict +0 output on finite
/// K/V (all valid QK scores unordered => fmax stays -Inf => frame never
/// initializes => e/l/P/O stay 0), so BOTH candidate and reference words must
/// be exactly +0 bits, and masked words never bypass candidate/reference
/// equality. `expected_nan_words` counts only true required-output-NaN (KV
/// `Nan`) words; Q taint/required-+0 words are reported separately.
fn native_out_compare(
    kind: NativeKind,
    expected: &[f32],
    cand_bytes: &[u8],
    ref_bytes: &[u8],
) -> Value {
    let words = cand_bytes.len() / 4;
    if words == 0
        || cand_bytes.len() != ref_bytes.len()
        || cand_bytes.len() % 4 != 0
        || ((kind.nonfinite() || kind.winner()) && expected.len() != words)
    {
        return json!({"pass":false, "error":"output/expectation extent mismatch"});
    }
    let is_q = matches!(kind, NativeKind::QNan | NativeKind::QInf);
    let mut mismatches = 0usize;
    let mut nonfinite = 0usize;
    let mut nan_mismatches = 0usize;
    let mut affected_nan = 0usize;
    let mut q_taint = 0usize;
    let mut zero_mismatches = 0usize;
    let mut secondary_mismatches = 0usize;
    let mut first = None;
    for (i, (cb, rb)) in cand_bytes
        .chunks_exact(4)
        .zip(ref_bytes.chunks_exact(4))
        .enumerate()
    {
        let c = f32::from_ne_bytes(cb.try_into().unwrap());
        let r = f32::from_ne_bytes(rb.try_into().unwrap());
        if is_q && expected[i].is_nan() {
            // INPUT-TAINT marker for a Q head poisoned before prep: require
            // strict +0 on BOTH arms plus candidate/reference equality.
            q_taint += 1;
            if c.to_bits() != 0 || r.to_bits() != 0 {
                zero_mismatches += 1;
                first.get_or_insert(i);
            }
            if cb != rb {
                mismatches += 1;
                first.get_or_insert(i);
            }
            if !c.is_finite() || !r.is_finite() {
                nonfinite += 1;
                first.get_or_insert(i);
            }
        } else if kind.nonfinite() && expected[i].is_nan() {
            affected_nan += 1;
            if !c.is_nan() || !r.is_nan() {
                nan_mismatches += 1;
                first.get_or_insert(i);
            }
        } else {
            if !c.is_finite() || !r.is_finite() {
                nonfinite += 1;
                first.get_or_insert(i);
            }
            if cb != rb {
                mismatches += 1;
                first.get_or_insert(i);
            }
            if matches!(kind, NativeKind::Winner | NativeKind::Kt48)
                && (c.to_bits() != expected[i].to_bits() || r.to_bits() != expected[i].to_bits())
            {
                secondary_mismatches += 1;
                first.get_or_insert(i);
            }
        }
    }
    let ok = mismatches == 0
        && nonfinite == 0
        && nan_mismatches == 0
        && zero_mismatches == 0
        && secondary_mismatches == 0
        && (!kind.nonfinite() || affected_nan + q_taint > 0);
    // FNV-1a over the candidate bytes: receipts of two kernel builds on the
    // same fixture are byte-comparable without re-running the reference.
    let cand_fnv64 = cand_bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    json!({"classification":if is_q {"q_taint_strict_positive_zero"} else if kind.nonfinite() {"nonfinite_positions_separate"} else {"finite_byte_gate"},
        "reference":"independent_quantized_HIP", "gate":true, "words":words,
        "candidate_fnv64":format!("{cand_fnv64:016x}"),
        "candidate_reference_word_mismatches":mismatches,
        "unexpected_nonfinite_words":nonfinite, "expected_nan_words":affected_nan,
        "nan_mask_mismatches":nan_mismatches,
        "input_q_taint_words":q_taint,
        "expected_positive_zero_words":q_taint,
        "zero_word_mismatches":zero_mismatches,
        "secondary_closed_winner_mismatches":secondary_mismatches,
        "first_mismatch":first, "pass":ok})
}
/// Consumed-input representation gate for the EXISTING QNan/QInf native
/// rounds: asserts the actual uploaded Q-plane bytes (`NativePrepared`
/// `q_codes`/`q_scales` forming the two-plane buffer) show E4M3 NaN code(s)
/// (`code & 0x7f == 0x7f`) on every input-tainted D256 Q head — finite scales
/// for QNan, +Inf scales for QInf (Inf/Inf quantizes to NaN codes via
/// `prep_q_row`; other dims quantize to signed zero). Counts must equal a
/// NONZERO tainted-head count. Caller couples this to the EXISTING
/// before/after `q_two_plane` device-input integrity equality.
fn native_q_nonfinite_repr(kind: NativeKind, p: &NativePrepared) -> Value {
    let heads = p.q_f32.len() / D;
    let mut tainted = 0usize;
    let mut bad = 0usize;
    let mut first = None;
    for h in 0..heads {
        let poisoned = p.q_f32[h * D..(h + 1) * D].iter().any(|v| !v.is_finite());
        if !poisoned {
            continue;
        }
        tainted += 1;
        let base = h * D;
        let has_nan = p.q_codes[base..base + D].iter().any(|c| c & 0x7f == 0x7f);
        let scale_ok = match kind {
            NativeKind::QInf => p.q_scales[h].is_infinite() && p.q_scales[h].is_sign_positive(),
            _ => p.q_scales[h].is_finite(),
        };
        if !has_nan || !scale_ok {
            bad += 1;
            first.get_or_insert(h);
        }
    }
    let ok = tainted > 0 && bad == 0;
    json!({"classification": "consumed_q_plane_representation",
        "tainted_q_heads": tainted, "representation_mismatches": bad,
        "first_mismatch_head": first, "pass": ok})
}
/// Native future-cache poison probe: rows at or below each probe row's causal
/// horizon stay byte-identical on BOTH arms when every later key is zeroed or
/// made extreme-finite (code 0x7e at scale 1.0, so `0 * value` stays exact in
/// masked lanes). Caches restored from the pristine host snapshot afterwards.
fn native_poison_probe(
    gpu: &mut Gpu,
    a: &NativeArena,
    cell: &NativeCell,
    p: &NativePrepared,
    segs: &[(usize, usize)],
    total: usize,
    round: usize,
) -> Result<Value> {
    let ref_cand = a.out_cand().download(gpu, p.used_out)?;
    let ref_ref = a.out_ref().download(gpu, p.used_out)?;
    let mut results = Vec::new();
    let mut all = true;
    let rows: Vec<usize> = {
        let mut v = vec![0usize];
        if cell.rows > 3 {
            v.push(3);
        }
        if cell.rows > 1 {
            v.push(cell.rows - 1);
        }
        v.dedup();
        v
    };
    for row in rows {
        let horizon = p.positions[row] as usize;
        for kind in ["zero", "extreme"] {
            let mut kc = p.k_cache.clone();
            let mut vc = p.v_cache.clone();
            for t in horizon + 1..p.k_cache.len() / POS_BYTES {
                for buf in [&mut kc, &mut vc] {
                    let rowb = &mut buf[t * POS_BYTES..(t + 1) * POS_BYTES];
                    match kind {
                        "zero" => rowb.fill(0),
                        _ => {
                            rowb[..KV * D].fill(0x7e);
                            for g in 0..KV {
                                rowb[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2]
                                    .copy_from_slice(&0x3c00u16.to_le_bytes());
                            }
                        }
                    }
                }
            }
            a.kc().upload(gpu, &kc)?;
            a.vc().upload(gpu, &vc)?;
            gpu.hip.device_synchronize()?;
            // Compare ALL queries whose actual horizon is unaffected, not
            // a row prefix (round1 positions need not be monotonic).
            native_output_poison(gpu, a.out_extra(), p.used_out, round)?;
            run_native_candidate(gpu, a, cell, total, segs, a.out_extra())?;
            gpu.hip.device_synchronize()?;
            let probe_c = a.out_extra().download(gpu, p.used_out)?;
            let cand_integrity = native_integrity(gpu, a, p, &kc, &vc)?;
            native_output_poison(gpu, a.out_extra(), p.used_out, 1 - round)?;
            run_native_reference(gpu, a, cell, total, segs, a.out_extra())?;
            gpu.hip.device_synchronize()?;
            let probe_r = a.out_extra().download(gpu, p.used_out)?;
            let ref_integrity = native_integrity(gpu, a, p, &kc, &vc)?;
            let unaffected: Vec<usize> = p
                .positions
                .iter()
                .enumerate()
                .filter_map(|(q, &pos)| ((pos as usize) <= horizon).then_some(q))
                .collect();
            let row_matches = |q: usize, probe: &[u8], original: &[u8]| {
                let span = q * ROW * 4..(q + 1) * ROW * 4;
                probe[span.clone()] == original[span]
            };
            let ok_c = !unaffected.is_empty()
                && unaffected
                    .iter()
                    .all(|&q| row_matches(q, &probe_c, &ref_cand));
            let ok_r = !unaffected.is_empty()
                && unaffected
                    .iter()
                    .all(|&q| row_matches(q, &probe_r, &ref_ref));
            let ok = ok_c && ok_r && pass(&cand_integrity) && pass(&ref_integrity);
            all &= ok;
            results.push(json!({"probe_row": row, "horizon": horizon,
                "perturbation": kind, "rows_compared": unaffected.len(),
                "changed_future_keys":p.k_cache.len() / POS_BYTES - horizon - 1,
                "candidate_stable": ok_c, "reference_stable": ok_r,
                "candidate_integrity":cand_integrity, "reference_integrity":ref_integrity,
                "byte_identical": ok}));
        }
    }
    a.kc().upload(gpu, &p.k_cache)?;
    a.vc().upload(gpu, &p.v_cache)?;
    gpu.hip.device_synchronize()?;
    let exercised = results
        .iter()
        .any(|v| v["changed_future_keys"].as_u64().unwrap_or(0) > 0);
    Ok(json!({"probes": results, "changed_future_exercised":exercised, "pass": all && exercised}))
}
/// All finite data gates byte-exactly against the quantized independent HIP
/// reference. NaN masks classify separately. Each arm starts with a different
/// finite poison; the next round exchanges 0xA5/0x5A. No unwritten-output Green.
fn native_round(gpu: &mut Gpu, a: &NativeArena, cell: &NativeCell, round: usize) -> Result<Value> {
    let p = native_prepare(cell, round)?;
    native_upload(gpu, a, &p, round)?;
    let total = p.total;
    let whole = [(0, cell.rows)];
    let range_off = if cell.rows > 2 { 2 } else { 0 };
    let range_n = cell.rows - range_off;
    let range_segs = [(range_off, range_n)];
    // Direct uses a batch-local scale-plane base. Pack the same first rows
    // independently of the overhang; restore FULL-total planes for ranges.
    let mut direct_q = p.q_codes[..cell.rows * Q_CODES_ROW].to_vec();
    direct_q.extend_from_slice(bytes(&p.q_scales[..cell.rows * H]));
    a.q().upload(gpu, &direct_q)?;
    run_native_candidate(gpu, a, cell, cell.rows, &whole, a.out_cand())?;
    gpu.hip.device_synchronize()?;
    let cand_whole = a.out_cand().download(gpu, p.used_out)?;
    let candidate_q_unchanged = a.q().download(gpu, direct_q.len())? == direct_q;
    run_native_reference(gpu, a, cell, cell.rows, &whole, a.out_ref())?;
    gpu.hip.device_synchronize()?;
    let ref_whole = a.out_ref().download(gpu, p.used_out)?;
    let reference_q_unchanged = a.q().download(gpu, direct_q.len())? == direct_q;
    let mut full_q = p.q_codes.clone();
    full_q.extend_from_slice(bytes(&p.q_scales));
    a.q().upload(gpu, &full_q)?;
    // Wrapper == primitive: the RANGE entry with offset 0 over the same FULL
    // allocation reproduces the direct launch bit for bit (exercises the
    // range offset path at the zero corner, exactly like the B range wrapper).
    native_output_poison(gpu, a.out_extra(), p.used_out, round)?;
    run_native_candidate(gpu, a, cell, total, &[(0, cell.rows)], a.out_extra())?;
    gpu.hip.device_synchronize()?;
    let wrap_eq = a.out_extra().download(gpu, p.used_out)? == cand_whole;
    native_output_poison(gpu, a.out_extra(), p.used_out, 1 - round)?;
    run_native_reference(gpu, a, cell, total, &whole, a.out_extra())?;
    gpu.hip.device_synchronize()?;
    let ref_wrap_eq = a.out_extra().download(gpu, p.used_out)? == ref_whole;
    // Range probe (nonzero offset): candidate and reference over the same
    // FULL planes with offset; outputs compared against the independent
    // offset expectation.
    let mut range_gate = Value::Null;
    if range_n >= 1 {
        native_output_poison(gpu, a.out_extra(), p.used_out, round)?;
        run_native_candidate(gpu, a, cell, total, &range_segs, a.out_extra())?;
        gpu.hip.device_synchronize()?;
        let cand_range_all = a.out_extra().download(gpu, p.used_out)?;
        native_output_poison(gpu, a.out_extra(), p.used_out, 1 - round)?;
        run_native_reference(gpu, a, cell, total, &range_segs, a.out_extra())?;
        gpu.hip.device_synchronize()?;
        let ref_range_all = a.out_extra().download(gpu, p.used_out)?;
        let seg_expected = if p.expected.is_empty() {
            &[][..]
        } else {
            &p.expected[range_off * ROW..(range_off + range_n) * ROW]
        };
        range_gate = native_out_compare(
            cell.kind,
            seg_expected,
            &cand_range_all[range_off * ROW * 4..(range_off + range_n) * ROW * 4],
            &ref_range_all[range_off * ROW * 4..(range_off + range_n) * ROW * 4],
        );
        let prefix = range_off * ROW * 4;
        let untouched = cand_range_all[..prefix]
            .iter()
            .all(|&b| b == NATIVE_POISONS[round] as u8)
            && ref_range_all[..prefix]
                .iter()
                .all(|&b| b == NATIVE_POISONS[1 - round] as u8);
        let gate_ok = pass(&range_gate) && untouched;
        range_gate["outside_segment_unchanged"] = json!(untouched);
        range_gate["pass"] = json!(gate_ok);
    }
    let finite = !cell.kind.nonfinite();
    let content_ok = {
        let words: Vec<u32> = cand_whole
            .chunks_exact(4)
            .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
            .collect();
        let any_nonzero = words.iter().any(|&w| w & 0x7FFF_FFFF != 0);
        let nonfinite_ok = if finite {
            words.iter().all(|&w| f32::from_bits(w).is_finite())
        } else {
            true
        };
        any_nonzero && nonfinite_ok
    };
    let attention = native_out_compare(cell.kind, &p.expected, &cand_whole, &ref_whole);
    let initial_integrity = native_integrity(gpu, a, &p, &p.k_cache, &p.v_cache)?;
    let probes = native_poison_probe(gpu, a, cell, &p, &whole, total, round)?;
    let integ = native_integrity(gpu, a, &p, &p.k_cache, &p.v_cache)?;
    let is_q = matches!(cell.kind, NativeKind::QNan | NativeKind::QInf);
    let q_repr = if is_q {
        native_q_nonfinite_repr(cell.kind, &p)
    } else {
        Value::Null
    };
    let ok = pass(&attention)
        && wrap_eq
        && ref_wrap_eq
        && pass(&range_gate)
        && content_ok
        && candidate_q_unchanged
        && reference_q_unchanged
        && pass(&initial_integrity)
        && pass(&probes)
        && pass(&integ)
        && (!is_q || (pass(&q_repr) && pass(&initial_integrity) && pass(&integ)));
    Ok(json!({"round": round, "seed": p.seed, "total_rows": total,
        "q_input":"host fixture; actual GPU producer proof gates the whole native family",
        "attention": attention,
        "consumed_q_representation": q_repr,
        "wrapper_range_offset0_reproduces_direct": wrap_eq,
        "reference_range_offset0_reproduces_direct": ref_wrap_eq,
        "range_nonzero_offset": range_gate,
        "content_sanity_ok": content_ok,
        "direct_q_unchanged":candidate_q_unchanged && reference_q_unchanged,
        "before_probe_integrity":initial_integrity,
        "future_poison_probes": probes, "integrity": integ, "pass": ok}))
}
// ------------------------------------------------------------ diagnostics
const SAMPLE_HEADS: [usize; 2] = [0, H - 1];
fn sample_rows(n: usize) -> Vec<usize> {
    let mut v = vec![0, n / 2, n - 1];
    v.dedup();
    v
}
/// Probe rows: row 3 sits inside the first 8-row group so the horizon bisects a query group.
fn probe_rows(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [0usize, 3, n / 2].into_iter().filter(|&r| r < n).collect();
    v.dedup();
    v
}
/// f64 causal reference over f32 Q and independently decoded, f16-staged fp8 K/V for sampled rows/heads.
/// It omits Q16 conversion and the kernels' WMMA/reduction DAG: diagnostic only.
fn cpu_reference(p: &Prepared, c: &Cell) -> Vec<f32> {
    let mut expected = Vec::new();
    for row in sample_rows(c.rows) {
        for head in SAMPLE_HEADS {
            let kh = head / (H / KV);
            let end = p.pos[row] as usize + 1;
            let mut scores = Vec::with_capacity(end);
            for pos in 0..end {
                let mut s = 0.0f64;
                for d in 0..D {
                    s += p.qh[(row * H + head) * D + d] as f64 * staged(&p.kc, pos, kh, d) as f64;
                }
                scores.push(s / 16.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max).exp();
                    *s
                })
                .sum();
            for d in 0..D {
                let v: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(pos, s)| s * staged(&p.vc, pos, kh, d) as f64)
                    .sum();
                expected.push((v / denom) as f32);
            }
        }
    }
    expected
}
fn sampled_output(gpu: &Gpu, buf: &Buf, c: &Cell) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    for row in sample_rows(c.rows) {
        let f = f32s(&buf.download_at(gpu, row * ROW * 4, ROW * 4)?);
        for head in SAMPLE_HEADS {
            v.extend_from_slice(&f[head * D..(head + 1) * D]);
        }
    }
    Ok(v)
}
fn ref_compare(expected: &[f32], actual: &[f32]) -> Value {
    let (mut max_abs, mut max_rel, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut nonfinite = 0usize;
    for (&e, &a) in expected.iter().zip(actual) {
        if !(e.is_finite() && a.is_finite()) {
            nonfinite += 1;
            continue;
        }
        let (e64, a64) = (e as f64, a as f64);
        let err = (e64 - a64).abs();
        max_abs = max_abs.max(err);
        max_rel = max_rel.max(err / e64.abs().max(1e-6));
        dot += e64 * a64;
        aa += e64 * e64;
        bb += a64 * a64;
    }
    let cosine = if aa == 0.0 || bb == 0.0 {
        Value::Null
    } else {
        json!(dot / (aa * bb).sqrt())
    };
    json!({"samples":expected.len(),"max_abs":max_abs,"max_rel":max_rel,"cosine":cosine,"nonfinite":nonfinite,
        "gate":false,"note":"f64 exact-decode reference without the shared f16 staging; diagnostic only"})
}
/// Replace every key above `horizon` in a pristine cache: `zero` all-zero rows, `extreme` code 0x7e (448) at scale
/// 1.0 (finite, so `0 * value` stays exact in the masked lanes).
fn poison_future(pristine: &[u8], horizon: usize, kind: &str) -> Vec<u8> {
    let mut v = pristine.to_vec();
    let npos = v.len() / POS_BYTES;
    for p in horizon + 1..npos {
        let row = &mut v[p * POS_BYTES..(p + 1) * POS_BYTES];
        match kind {
            "zero" => row.fill(0),
            _ => {
                row[..SCALE_OFF].fill(0x7e);
                for g in 0..KV {
                    row[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2]
                        .copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
        }
    }
    v
}
/// Rows at or below a probe row's causal horizon must stay byte-identical when every later key is poisoned
/// (standard positions). The caches are restored from the pristine host snapshot afterwards.
fn poison_probe(gpu: &mut Gpu, a: &Arena, c: &Cell, p: &Prepared) -> Result<Value> {
    let u = p.used;
    let reference = a.out_pm().download(gpu, u.out)?;
    let mut results = Vec::new();
    let mut all = true;
    for row in probe_rows(c.rows) {
        let horizon = p.pos[row] as usize;
        for kind in ["zero", "extreme"] {
            a.kc().upload(gpu, &poison_future(&p.kc, horizon, kind))?;
            a.vc().upload(gpu, &poison_future(&p.vc, horizon, kind))?;
            gpu.hip.device_synchronize()?;
            a.out_extra().poison(gpu, u.out)?;
            run_pm_direct(gpu, a, c, a.out_extra())?;
            gpu.hip.device_synchronize()?;
            let probe = a.out_extra().download(gpu, u.out)?;
            let live = (row + 1) * ROW * 4;
            let ok = probe[..live] == reference[..live];
            all &= ok;
            results.push(json!({"probe_row":row,"horizon":horizon,"perturbation":kind,"rows_compared":row + 1,"byte_identical":ok}));
        }
    }
    a.kc().upload(gpu, &p.kc)?;
    a.vc().upload(gpu, &p.vc)?;
    gpu.hip.device_synchronize()?;
    Ok(json!({"probes":results,"pass":all}))
}

// ----------------------------------------------------------- oracle round
fn oracle_round(gpu: &mut Gpu, a: &Arena, c: &Cell, round: usize) -> Result<Value> {
    let p = prepare(gpu, a, c, round)?;
    let u = p.used;
    run_hip(gpu, a, c)?;
    run_pm(gpu, a, c)?;
    gpu.hip.device_synchronize()?;
    let nan = c.source == Source::Nan;
    let q16 = compare(
        &a.q16_hip().download(gpu, u.q16)?,
        &a.q16_pm().download(gpu, u.q16)?,
        Elem::F16,
        false,
    );
    let out_hip = a.out_hip().download(gpu, u.out)?;
    let out_pm = a.out_pm().download(gpu, u.out)?;
    let attention = compare(&out_hip, &out_pm, Elem::F32, nan);
    let wrapper = if c.wrapper() {
        a.out_extra().poison(gpu, u.out)?;
        gpu.attention_fp8_e4m3_fa_prefill_pm_gfx1201(
            &a.q32().view(u.q32),
            &a.kc().view(u.kc),
            &a.vc().view(u.vc),
            &a.out_extra().view(u.out),
            &a.pos().view(u.pos),
            H,
            KV,
            D,
            c.ctx(),
            c.rows,
        )?;
        gpu.hip.device_synchronize()?;
        Some(compare(
            &out_pm,
            &a.out_extra().download(gpu, u.out)?,
            Elem::F32,
            nan,
        ))
    } else {
        None
    };
    let q16_bytes = a.q16_pm().download(gpu, u.q16)?;
    let nonzero_q16 = q16_bytes
        .chunks_exact(2)
        .any(|x| u16::from_ne_bytes([x[0], x[1]]) & 0x7FFF != 0);
    let out_nonzero = f32s(&out_pm).iter().any(|&x| x != 0.0);
    let content_ok = nonzero_q16 && out_nonzero;
    let (reference, probes) = if c.diagnostics() && round == 0 && p.fixture == Fixture::Std {
        let r = ref_compare(&cpu_reference(&p, c), &sampled_output(gpu, a.out_pm(), c)?);
        (Some(r), Some(poison_probe(gpu, a, c, &p)?))
    } else {
        (None, None)
    };
    let integ = integrity(gpu, a, &p)?;
    let ok = pass(&q16)
        && pass(&attention)
        && wrapper.as_ref().map_or(true, pass)
        && pass(&integ)
        && content_ok
        && probes.as_ref().map_or(true, pass);
    Ok(
        json!({"round":round,"fixture":p.fixture.name(),"seed":p.seed,"q16":q16,"attention":attention,
        "wrapper_vs_pm_primitive":wrapper,"integrity":integ,"content_sanity_ok":content_ok,
        "future_poison_probes":probes,"cpu_reference_diagnostic":reference,"pass":ok}),
    )
}

// --------------------------------------------------------------- plan
fn plan() {
    let ieee = ieee_cases().expect("independent cases construct on CPU");
    for ic in &ieee {
        println!(
            "{}",
            json!({"ieee_case":ic.name.clone(),"rows":ic.positions.len(),"positions":ic.positions.clone(),
                "allocated_ctx":ic.k_cache.len() / POS_BYTES,"admitted_code_census":"rounded_exponent_overflow",
                "assertion":ic.expected.kind(),
                "hip_vs_pm":"separate","wrapper":false,"launches":"hip_primitive+pm_primitive"})
        );
    }
    // Native qualification plan: exact fixtures first (single/KT48/winner),
    // plus general finite ragged/nonzero-past byte gates, NaNs and ranges.
    // Planned counts only (no GPU execution in SOURCE-ONLY changes).
    let native = native_cells();
    for nc in &native {
        let total = nc.rows + 4;
        println!(
            "{}",
            json!({"native_cell": nc.name(), "rows": nc.rows, "start": nc.start,
                "kind": nc.kind.name(), "context": nc.ctx(),
                "q_two_plane_bytes": NativeCell::q_bytes(total),
                "cache_bytes_per_side": NativeCell::cache_bytes(nc.ctx()),
                "kernarg_bytes": 68, "kernarg_layout": "6ptr+4i32+f32",
                "grid": nc.gpu_grid(nc.rows), "block": [NATIVE_BLOCK, 1, 1],
                "lds": NATIVE_LDS, "kt": 48,
                "reference_grid": [nc.rows * H, 1, 1], "reference_block": [32, 1, 1], "reference_lds": 0,
                "launches_per_round": "both direct + both offset0 range + both nonzero range + causal probes",
                "poison_rounds": ["0xA5", "0x5A"], "finite_byte_gate": !nc.kind.nonfinite(),
                "producer_family_launches": 6, "producer_source_fixtures":2,
                "producer_query_rows_per_fixture": 17})
        );
    }
    println!("{}", json!({"native_coverage_map":native_coverage_map()}));
    let cells = oracle_cells();
    let with_long: Vec<Cell> = cells.iter().copied().chain(long_cells()).collect();
    let sizes = arena_sizes(&with_long);
    let total: usize = sizes.iter().map(|b| b + 2 * GUARD).sum();
    for c in &cells {
        println!(
            "{}",
            json!({"cell":c.name(),"rows":c.rows,"start":c.start,"segment":c.segment,"segments":c.segments().len(),
                "fixture":c.fixture.name(),"source":c.source.name(),"context":c.ctx(),"wrapper":c.wrapper(),
                "pm_pre_grid":pre_grid(c.rows.min(c.segment)),"direct_grid":direct_grid(c.rows.min(c.segment)),
                "launches_per_round":2 * 2 * c.segments().len()})
        );
    }
    for (rows, start, seg) in LONG_CELLS {
        println!(
            "{}",
            json!({"long_cell":format!("n{rows}_s{start}_seg{seg}_std_runtime_long"),"rows":rows,"start":start,"segment":seg,
                "context":LONG_CTX,"includes_last_position_16383":true,"whole_and_segmented":true,"wrapper_where_applicable":true})
        );
    }
    println!(
        "{}",
        json!({"plan":true,"ieee_cases":ieee.len(),"ieee_first":true,"cells":cells.len(),"long_cells":LONG_CELLS.len(),
            "buffers":BUFFER_NAMES,"arena_bytes_each":sizes,"arena_total_bytes":total,
            "hip_symbols":[HIP_PRE,HIP_DIRECT],"hip_defines":["HIPFIRE_FA2_KMODE=8","HIPFIRE_FA2_Q0_A3B=1"],
            "pm_module":fa_prefill_pm::FP8_MODULE,"pm_symbol":fa_prefill_pm::FP8_DIRECT,
            "pm_preconvert_module":fa_prefill_pm::MODULE,"pm_preconvert_symbol":fa_prefill_pm::PRECONVERT,
            "kernarg_bytes":{"pre":40,"direct":60},"direct_block":DIRECT_BLOCK,"direct_lds":DIRECT_LDS,
            "native_cells":native.len(),
            "native_symbols":[NATIVE_DIRECT, NATIVE_RANGE],
            "native_reference_symbols":[NATIVE_REF_DIRECT, NATIVE_REF_RANGE],
            "native_reference_module":NATIVE_REF_MODULE,
            "native_defines":["HIPFIRE_FA2_QRESIDENT_V2_Q8_A3B=1","HIPFIRE_FA2_QRESIDENT_V2_Q8=1","HIPFIRE_FA2_FP8=1","HIPFIRE_FA2_KMODE=8"],
            "native_kernarg_bytes":68,"native_block":NATIVE_BLOCK,"native_grid_y":NATIVE_GRID_Y,
            "native_lds":NATIVE_LDS,"native_buffers":NATIVE_BUFFER_NAMES})
    );
}

// --------------------------------------------------------- self-check
fn self_check() -> usize {
    let mut n = 0usize;
    let mut ok = |c: bool, what: &str| {
        assert!(c, "self-check failed: {what}");
        n += 1;
    };
    // E4M3FN model.
    ok(
        e4m3fn(0x38) == Some(1.0) && e4m3fn(0x7e) == Some(448.0) && e4m3fn(0xfe) == Some(-448.0),
        "known codes",
    );
    ok(
        e4m3fn(0x01) == Some(2f32.powi(-9)) && e4m3fn(0x08) == Some(2f32.powi(-6)),
        "subnormal / smallest normal",
    );
    ok(
        e4m3fn(0x80).map(f32::to_bits) == Some(0x8000_0000),
        "0x80 is -0, not NaN",
    );
    ok(
        e4m3fn(0x7f).is_none() && e4m3fn(0xff).is_none() && finite_codes().len() == 254,
        "NaN codes",
    );
    ok(
        (0u8..0x7e).all(|c| e4m3fn(c).unwrap() < e4m3fn(c + 1).unwrap()),
        "monotone magnitudes",
    );
    // Row layout: 516-byte rows, head-major codes then two scales; ABI blobs.
    ok(
        POS_BYTES == 516
            && SCALE_OFF == 512
            && POS_BYTES == fa_prefill_pm::FaPrefillKvCodec::Fp8E4M3.cache_row_bytes(),
        "row layout",
    );
    let cache = grid_cache(4, 0, false);
    ok(cache.len() == 4 * POS_BYTES, "grid cache extent");
    let table = finite_codes();
    for p in 0..4 {
        for g in 0..KV {
            let scale = SCALES[(p + 3 * g) % SCALES.len()];
            let at = p * POS_BYTES + SCALE_OFF + 2 * g;
            ok(
                u16::from_le_bytes([cache[at], cache[at + 1]]) == scale,
                "scale slot of each head",
            );
            ok(
                (0..D).all(|d| {
                    cache[p * POS_BYTES + g * D + d]
                        == safe_code(scale, table[(d + 7 * p + 31 * g) % table.len()])
                }),
                "code plane of each head",
            );
            ok(
                staged(&cache, p, g, 3).is_finite(),
                "no overflow-invalid artifact in a grid cache",
            );
        }
    }
    ok(
        grid_cache(8, 0, true)
            .iter()
            .zip(grid_cache(8, 0, false))
            .filter(|(a, b)| a != &b)
            .count()
            >= 3,
        "NaN injection",
    );
    ok(
        pre_kernargs(std::ptr::null_mut(), std::ptr::null_mut(), 8).len()
            == fa_prefill_pm::PRE_KERNARG_BYTES,
        "preconvert ABI size",
    );
    let d = direct_kernargs(
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        8,
    );
    ok(
        d.len() == fa_prefill_pm::DIRECT_KERNARG_BYTES,
        "direct ABI size",
    );
    ok(
        DIRECT_BLOCK == fa_prefill_pm::DIRECT_BLOCK[0]
            && DIRECT_LDS == fa_prefill_pm::DIRECT_LDS
            && PRE_BLOCK == fa_prefill_pm::PRE_BLOCK[0],
        "launch config",
    );
    // Fixtures: every position inside the cell, duplicates / nonmonotone where promised.
    for f in [
        Fixture::Std,
        Fixture::Reverse8,
        Fixture::DupPairs8,
        Fixture::Rotate8,
        Fixture::SameGroup8,
    ] {
        for (rows, start) in [(1usize, 0usize), (7, 3), (64, 0), (65, 63), (512, 1)] {
            let pos = f.positions(rows, start);
            ok(
                pos.len() == rows
                    && pos
                        .iter()
                        .all(|&x| (x as usize) >= start && (x as usize) < start + rows),
                "fixture positions in range",
            );
            ok(
                validate_cell(rows, &pos, start + rows).is_ok(),
                "fixture admitted",
            );
        }
    }
    ok(
        Fixture::DupPairs8
            .positions(8, 0)
            .windows(2)
            .any(|w| w[0] == w[1]),
        "duplicate pairs",
    );
    ok(
        Fixture::Reverse8.positions(8, 0) == [7, 6, 5, 4, 3, 2, 1, 0],
        "reverse8",
    );
    ok(
        validate_cell(0, &[], 1).is_err()
            && validate_cell(1, &[1], 1).is_err()
            && validate_cell(1, &[-1], 4).is_err(),
        "domain refusals",
    );
    // Cells: every cell is admitted, tails are primitive-only, the wrapper cells are production rows.
    let cells = oracle_cells();
    for c in &cells {
        ok(
            validate_cell(c.rows, &c.fixture.positions(c.rows, c.start), c.ctx()).is_ok(),
            "cell admitted",
        );
        ok(
            c.wrapper() == (c.rows >= 64 && c.segment == c.rows),
            "wrapper cells",
        );
        ok(
            c.rows >= 64 || c.source == Source::Runtime,
            "tails are runtime-cache primitive cells",
        );
    }
    ok(
        cells.iter().any(|c| c.rows == 1) && cells.iter().any(|c| c.rows == 63),
        "exact tails 1..=63",
    );
    ok(
        cells.iter().any(|c| c.source == Source::Nan)
            && cells.iter().any(|c| c.source == Source::Grid),
        "synthetic sources",
    );
    ok(
        hip_source().starts_with("#define HIPFIRE_FA2_KMODE 8\n#define HIPFIRE_FA2_Q0_A3B 1\n"),
        "hip twin defines",
    );
    ok(
        !hip_source().contains("#define HIPFIRE_FA2_FP8"),
        "hip twin is not FA2_FP8",
    );
    ok(
        HIP_BODY.contains("HIPFIRE_FA2_Q0_A3B")
            && HIP_BODY.contains(HIP_DIRECT)
            && HIP_BODY.contains("attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201"),
        "production hip carries the A3B twin symbols",
    );
    // Comparison semantics (original cells; the independent fixtures use their own classifier).
    let a = [0u8, 0, 0x80, 0x3f];
    ok(pass(&compare(&a, &a, Elem::F32, false)), "identical passes");
    let mut b = a;
    b[0] = 1;
    ok(
        !pass(&compare(&a, &b, Elem::F32, false)),
        "one bit fails the byte gate",
    );
    let nan = f32::NAN.to_ne_bytes();
    ok(
        !pass(&compare(&nan, &nan, Elem::F32, false)),
        "NaN fails the finite gate",
    );
    ok(
        pass(&compare(&nan, &nan, Elem::F32, true)),
        "NaN positions agree under the NaN classification",
    );
    // Independent native semantic proof: IEEE f16 RNE / single-key accumulation / all 256 E4M3
    // encodings / census / overflow / signed zeros / empty-subtile admission / unaffected NaN
    // bytes / negative controls / long-context boundaries. Pure CPU, before any GPU init.
    ok(
        ieee_f16_rne(1.0) == 0x3c00
            && ieee_f16_rne(65504.0) == 0x7bff
            && ieee_f16_rne(65520.0) == 0x7c00,
        "f16 RNE normal/overflow boundaries",
    );
    ok(
        ieee_f16_rne(-0.0) == 0x8000 && ieee_f16_rne(0.0) == 0x0000,
        "f16 RNE signed zero preserved at the bit level",
    );
    ok(
        f16_to_f32_ieee(0x3c00) == 1.0 && f16_to_f32_ieee(0x8000).to_bits() == 0x8000_0000,
        "independent f16 bit decode",
    );
    ok(
        (0..=255u8).all(|code| match e4m3fn(code) {
            Some(_) => ieee_stage_one(code, 0x3c00).is_some(),
            None => code == 0x7f || code == 0xff,
        }),
        "all 256 E4M3 encodings: 254 finite stage, exactly 0x7f/0xff NaN",
    );
    {
        // Staged -0 accumulates from +0 and the independent mean returns +0 (signed-zero canonicalization).
        let mut cache = vec![0u8; POS_BYTES];
        cache[3] = 0x80;
        cache[SCALE_OFF..SCALE_OFF + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for g in 1..KV {
            cache[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let w = ieee_stage(&cache, 0, 0, 3).unwrap();
        let mut acc = 0.0f32;
        acc += f16_to_f32_ieee(w);
        let out = acc;
        ok(
            w == 0x8000 && out.to_bits() == 0x0000_0000,
            "staged -0 accumulates from +0 to output +0",
        );
    }
    ok(
        ieee_cases().map(|v| v.len()).unwrap_or(0) == 21,
        "independent frozen case count: 2 KNaN + 1 empty + 4 VNaN + 14 sweep",
    );
    for ic in ieee_cases().unwrap() {
        let want = ieee_expected(&ic).unwrap();
        ok(
            want.len() == ic.positions.len() * ROW,
            "independent expected extent",
        );
        match &ic.expected {
            IeeeExpected::Finite(_) => ok(true, "finite"),
            IeeeExpected::VNaN {
                clean_expected_words,
                kv_head,
                dimension,
            } => {
                // Precisely the selected KV-head/dimension outputs are NaN; all unaffected words match clean.
                let mut affected_ok = true;
                let mut unaffected_ok = true;
                for (r, _) in ic.positions.iter().enumerate() {
                    for h in 0..H {
                        for d in 0..D {
                            let i = (r * H + h) * D + d;
                            let is_aff = h / 8 == *kv_head && d == *dimension;
                            let word_nan = f32::from_bits(want[i]).is_nan();
                            affected_ok &= !is_aff || word_nan;
                            unaffected_ok &= is_aff || want[i] == clean_expected_words[i];
                        }
                    }
                }
                ok(
                    unaffected_ok,
                    "VNaN unaffected words match clean expectation",
                );
                ok(affected_ok, "VNaN affected words are NaN");
            }
            IeeeExpected::KNaN {
                clean_expected_words,
                kv_head,
            } => {
                // Each query-head row of the affected KV head contains NaN; the other head matches clean.
                let mut affected_ok = true;
                let mut unaffected_ok = true;
                for (r, _) in ic.positions.iter().enumerate() {
                    for h in 0..H {
                        let row = &want[(r * H + h) * D..(r * H + h + 1) * D];
                        let clean = &clean_expected_words[(r * H + h) * D..(r * H + h + 1) * D];
                        if h / 8 == *kv_head {
                            affected_ok &= row.iter().any(|&w| f32::from_bits(w).is_nan());
                        } else {
                            unaffected_ok &= row == clean;
                        }
                    }
                }
                ok(
                    unaffected_ok,
                    "KNaN unaffected words match clean expectation",
                );
                ok(affected_ok, "KNaN affected rows contain NaN");
            }
        }
    }
    {
        // Admitted-code census: recompute it the way the sweep admission does (rounded-exponent
        // overflow) and require every admitted code to be present in the emitted sweep caches.
        const IEEE_SCALES: [u16; 12] = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000, 0xbc01,
            0xfbff,
        ];
        let shifts = [0usize, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 0, 3];
        let mut census_ok = true;
        for shift in shifts {
            let scales = [
                IEEE_SCALES[shift],
                IEEE_SCALES[(shift + 3) % IEEE_SCALES.len()],
            ];
            let (k, v) = single_key_caches(scales, shift);
            for g in 0..KV {
                let admitted: std::collections::BTreeSet<u8> = finite_codes()
                    .into_iter()
                    .filter(|&code| {
                        ieee_stage_one(code, scales[g])
                            .map(|w| w & 0x7c00 != 0x7c00)
                            .unwrap_or(false)
                    })
                    .collect();
                for cache in [&k, &v] {
                    let seen: std::collections::BTreeSet<u8> =
                        cache[g * D..(g + 1) * D].iter().copied().collect();
                    census_ok &= seen == admitted;
                }
            }
        }
        ok(census_ok, "admitted-code census over the sweep caches");
    }
    {
        // Negative controls: a one-bit flip of a finite word, and injecting a NaN code into a
        // finite case, both fail the independent classifier.
        let ic = ieee_cases()
            .unwrap()
            .into_iter()
            .find(|c| matches!(c.expected, IeeeExpected::Finite(_)))
            .unwrap();
        let want = ieee_expected(&ic).unwrap();
        let mut flipped = want.clone();
        flipped[0] ^= 1;
        ok(
            compare_ieee_words(&ic.expected, &flipped)["pass"] == false,
            "one-bit flip fails independent classifier",
        );
        let mut nan_case = ic.clone();
        nan_case.v_cache[5] = 0x7f;
        let nan_got = ieee_expected(&nan_case).unwrap();
        ok(
            compare_ieee_words(&ic.expected, &nan_got)["pass"] == false,
            "NaN injection fails the finite classifier",
        );
    }
    ok(
        LONG_CELLS.iter().all(|&(rows, start, seg)| {
            validate_cell(rows, &Fixture::Std.positions(rows, start), start + rows).is_ok()
                && seg <= rows
        }),
        "long cells admitted",
    );
    ok(
        LONG_CELLS.iter().all(|&(rows, start, _)| {
            start + rows == LONG_CTX
                && Fixture::Std
                    .positions(rows, start)
                    .contains(&(LONG_CTX as i32 - 1))
        }),
        "long cells include last position 16383 at context 16384",
    );
    ok(
        validate_cell(1, &[16383], LONG_CTX).is_ok()
            && validate_cell(1, &[16400], LONG_CTX).is_err(),
        "last position 16383 admitted, 16400 rejected for context 16384",
    );
    // Native A3B qualification CPU checks (pure CPU, no GPU init): independent
    // F32->E4M3 RNE round-trip/saturation, upstream prep contract, two-plane
    // offsets, kernel ABI constants, reference-source hygiene, cell coverage.
    ok(e4m3_rne(1.0) == 0x38, "native e4m3_rne(1.0) == 0x38");
    ok(
        e4m3_rne(448.0) == 0x7e && e4m3_rne(1000.0) == 0x7e && e4m3_rne(-1000.0) == 0xfe,
        "native e4m3 saturates signed at 448",
    );
    ok(
        e4m3_rne(0.0) == 0x00 && e4m3_rne(-0.0) == 0x80,
        "native e4m3 signed zero preserved",
    );
    ok(
        (0..=255u8).all(|c| match e4m3fn(c) {
            Some(v) => e4m3_rne(v) == c,
            None => c == 0x7f || c == 0xff,
        }),
        "native e4m3_rne round-trips all 254 finite codes",
    );
    ok(
        e4m3_rne(f32::NAN) == 0x7f && e4m3_rne(-f32::NAN) == 0xff,
        "native e4m3 NaN collapses to canonical signed codes",
    );
    {
        // Upstream-F32 prep contract: amax/448 F32 scale, per-dim prepare
        // permute, all-zero rows yield zero codes with scale 1 (never NaN).
        let q: Vec<f32> = (0..D).map(|d| (d as f32 - 128.0) / 128.0).collect();
        let (codes, s) = prep_q_row(&q);
        let mut amax = 0.0f32;
        for &v in &q {
            amax = amax.max(v.abs());
        }
        ok(
            (s - amax / 448.0).abs() == 0.0,
            "native prep scale is amax/448",
        );
        ok(
            (0..D).all(|d| {
                let f = d >> 4;
                let kg = (d >> 3) & 1;
                let j = d & 7;
                codes[kg * 128 + f * 8 + j] == e4m3_rne(q[d] / s)
            }),
            "native prep permute matches fused-prep layout",
        );
        let (zcodes, zs) = prep_q_row(&vec![0.0; D]);
        ok(
            zs == 1.0 && zcodes.iter().all(|&c| c == 0x00),
            "native zero row: codes 0, scale 1",
        );
    }
    ok(
        Q_BYTES_ROW == 4160 && Q_CODES_ROW == 4096 && Q_SCALES_ROW == 64,
        "native two-plane row is 4160 bytes (4096 codes + 64 scale)",
    );
    ok(
        NativeCell::q_bytes(10) == 10 * 4160 && NativeCell::cache_bytes(100) == 100 * 516,
        "native byte extents: Q two-plane and 516-byte cache",
    );
    {
        // FULL-plane offsets: codes at row*4096, scales at total*4096 + row*64
        // (never tail-local). Matches the B runtime `a3b_qresident_spans`.
        let (total, off) = (100usize, 7usize);
        ok(
            off * Q_CODES_ROW == 7 * 4096
                && total * Q_CODES_ROW + off * Q_SCALES_ROW == 100 * 4096 + 7 * 64,
            "native FULL-plane codes/scales offsets",
        );
        ok(
            native_validate(
                64,
                &vec![63; 64],
                64,
                NativeCell::q_bytes(64),
                NativeCell::cache_bytes(64),
            )
            .is_ok()
                && native_validate(0, &[], 64, 0, 0).is_err()
                && native_validate(
                    64,
                    &vec![64; 64],
                    64,
                    NativeCell::q_bytes(64),
                    NativeCell::cache_bytes(64),
                )
                .is_err(),
            "native admission: rows/range/extents fail closed",
        );
        // 68-byte packed ABI: 6 pointers + 4 i32 + 1 f32.
        let kb = native_kernargs(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            64,
        );
        ok(kb.len() == 68, "native kernarg blob is 68 bytes");
    }
    ok(
        NATIVE_GRID_Y == 2 && NATIVE_LDS == 49_936 && NATIVE_BLOCK == 768,
        "native launch: block 768, grid Y 2, LDS 49936",
    );
    ok(
        NATIVE_REF_BODY.contains("attention_fp8_a3b_qresident_independent_ref_gfx1201")
            && NATIVE_REF_BODY
                .contains("attention_fp8_a3b_qresident_independent_ref_range_gfx1201")
            && NATIVE_REF_BODY.contains("a3b_ref_body"),
        "native independent reference carries both exports on its own quantized body",
    );
    ok(
        NATIVE_REF_BODY.contains("__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(")
            && !NATIVE_REF_BODY.contains("fa2_qresident_v2_q8_a3b_body(")
            && !NATIVE_REF_BODY.contains("#include \"attention_q8_0_fa2_gqa"),
        "native reference binds FP8 WMMA, not the candidate body/include",
    );
    ok(
        NATIVE_REF_BODY.contains("516ULL")
            && NATIVE_REF_BODY.contains("(unsigned long long)key * 516ULL"),
        "native reference addresses 516-byte rows with byte loads (no alignment assumption)",
    );
    {
        // Independent exactness: the closed-form winner expectation is exactly
        // 448 x 56 = 25,088 on every word, finite everywhere (CPU cross-check
        // of the two independent constructions, not a GPU result).
        let cell = NativeCell {
            rows: 2,
            start: 0,
            kind: NativeKind::Winner,
        };
        let p = native_prepare(&cell, 0).expect("native prepare builds on CPU");
        ok(p.expected.len() == 2 * ROW, "native expected extent");
        ok(
            p.expected
                .iter()
                .all(|&x| x.to_bits() == 25_088f32.to_bits()),
            "native winner expectation is exactly 25088 everywhere",
        );
        ok(
            448.0f32 * f16_to_f32_ieee(0x5300) == 25_088.0,
            "native winner value 448x56 exact",
        );
        // Negative controls: one-bit flip of an expected word, and a scale
        // flip, both fail their gates.
        let mut raw = p.q_codes[..cell.rows * Q_CODES_ROW].to_vec();
        raw.extend_from_slice(bytes(&p.q_scales[..cell.rows * H]));
        ok(
            pass(&native_q_compare(&p.q_f32, &raw, cell.rows)),
            "native CPU quantizer accepts a complete independently packed fixture",
        );
        let mut scale_flip = raw.clone();
        scale_flip[cell.rows * Q_CODES_ROW] ^= 1;
        ok(
            !pass(&native_q_compare(&p.q_f32, &scale_flip, cell.rows)),
            "native scale-plane one-bit flip fails the full prep-Q comparison",
        );
        raw[0] ^= 1;
        ok(
            !pass(&native_q_compare(&p.q_f32, &raw, cell.rows)),
            "native code-plane one-bit flip fails the full prep-Q comparison",
        );
        let mut eflip = p.expected.clone();
        eflip[0] = f32::from_bits(eflip[0].to_bits() ^ 1);
        ok(
            native_out_compare(
                NativeKind::Winner,
                &p.expected,
                bytes(&eflip),
                bytes(&p.expected),
            )["pass"]
                == false,
            "native one-bit output flip fails the byte gate",
        );
        ok(
            !pass(&native_out_compare(
                NativeKind::Random,
                &[],
                bytes(&eflip),
                bytes(&p.expected),
            )),
            "general finite one-bit mismatch fails mandatory HIP-reference byte gate",
        );
        let nf = vec![f32::NAN; ROW];
        ok(
            !pass(&native_out_compare(
                NativeKind::Random,
                &[],
                bytes(&nf),
                bytes(&nf),
            )),
            "matching nonfinite general outputs cannot pass",
        );
        let poison: Vec<f32> = vec![f32::from_bits(0xa5a5_a5a5); ROW];
        ok(
            !pass(&native_out_compare(
                NativeKind::Nan,
                &nf,
                bytes(&poison),
                bytes(&poison),
            )),
            "unwritten finite poison fails an expected NaN mask",
        );
    }
    ok(
        native_cells().len() == 32 && NATIVE_CELLS.len() == 30,
        "native family: 32 planned cells (28 finite byte gates, 4 nonfinite masks)",
    );
    ok(
        native_cells()
            .iter()
            .any(|c| matches!(c.kind, NativeKind::Kt48))
            && native_cells()
                .iter()
                .any(|c| matches!(c.kind, NativeKind::Nan))
            && native_cells()
                .iter()
                .any(|c| matches!(c.kind, NativeKind::Winner))
            && native_cells()
                .iter()
                .any(|c| matches!(c.kind, NativeKind::Random)),
        "native family covers KT48/winner/NaN/ragged kinds",
    );
    ok(
        native_cells()
            .iter()
            .all(|c| native_prepare(c, 0).is_ok() && native_prepare(c, 1).is_ok()),
        "native family: all 32 cells build on CPU for both poison rounds",
    );
    ok(
        NATIVE_REF_BODY.contains("q_scales[") && NATIVE_REF_BODY.contains("q_codes + "),
        "native reference takes separate code/scale pointers (two-plane ABI)",
    );
    let poison_words = NATIVE_POISONS.map(|b| u32::from_ne_bytes([b as u8; 4]));
    ok(
        poison_words[0] != poison_words[1]
            && poison_words.iter().all(|&w| f32::from_bits(w).is_finite()),
        "native poisons A5A5A5A5 and 5A5A5A5A are distinct finite F32 words",
    );
    let varying = NativeCell {
        rows: 49,
        start: 63,
        kind: NativeKind::Random,
    };
    let fixture = native_prepare(&varying, 0).expect("native varied fixture");
    ok(
        fixture.q_scales[..H]
            .windows(2)
            .any(|s| s[0].to_bits() != s[1].to_bits())
            && fixture.positions.windows(2).any(|p| p[0] != p[1])
            && fixture
                .q_f32
                .iter()
                .any(|&x| half::f16::from_f32(x).to_f32().to_bits() != x.to_bits()),
        "general fixture has per-head scales, varied positions and non-F16-exact Q",
    );
    ok(
        [47, 48, 49, 511, 512, 513].iter().all(|&rows| {
            native_cells()
                .iter()
                .any(|c| c.rows == rows && c.kind == NativeKind::Random)
        }),
        "general finite gates cover 48-query groups and the 512-query run reset",
    );
    let mapping = native_coverage_map();
    ok(
        mapping["native_cells"] == 32,
        "bounded native coverage is explicit, not old31 numerical equivalence",
    );
    let inf = native_prepare(
        &NativeCell {
            rows: 8,
            start: 0,
            kind: NativeKind::QInf,
        },
        0,
    )
    .expect("F32 Inf fixture");
    ok(
        inf.q_scales[0].is_infinite()
            && inf.q_codes[0] & 0x7f == 0x7f
            && inf.expected[..D].iter().all(|v| v.is_nan()),
        "F32 Inf fixture: +Inf scale with Inf/Inf NaN code and INPUT-TAINT markers (not expected-output-NaN)",
    );
    // Q-nonfinite comparator regression: taint markers require strict +0 on
    // BOTH arms plus candidate/reference equality (masked words never agree
    // by fiat).
    let taint = vec![f32::NAN; ROW];
    let zeros = vec![0.0f32; ROW];
    ok(
        pass(&native_out_compare(
            NativeKind::QNan,
            &taint,
            bytes(&zeros),
            bytes(&zeros),
        )),
        "Q taint: matching +0 words pass",
    );
    let nonzero = vec![1.0f32; ROW];
    ok(
        !pass(&native_out_compare(
            NativeKind::QNan,
            &taint,
            bytes(&nonzero),
            bytes(&nonzero),
        )),
        "Q taint: matching nonzero finite words fail despite reference agreement",
    );
    let nans = vec![f32::NAN; ROW];
    ok(
        !pass(&native_out_compare(
            NativeKind::QNan,
            &taint,
            bytes(&nans),
            bytes(&nans),
        )),
        "Q taint: matching NaNs fail the required-+0 gate",
    );
    // Consumed-plane representation regression: QNan finite scales, QInf +Inf
    // scales, both with NaN codes on tainted heads and NONZERO taint counts.
    let nan_fixture = native_prepare(
        &NativeCell {
            rows: 8,
            start: 0,
            kind: NativeKind::QNan,
        },
        0,
    )
    .expect("F32 NaN fixture");
    ok(
        pass(&native_q_nonfinite_repr(NativeKind::QNan, &nan_fixture)),
        "QNan consumed plane: NaN codes with finite scales on nonzero tainted heads",
    );
    ok(
        pass(&native_q_nonfinite_repr(NativeKind::QInf, &inf)),
        "QInf consumed plane: NaN codes with +Inf scales on nonzero tainted heads",
    );
    let (nan_codes, nan_scale) = prep_q_row(&vec![f32::NAN; D]);
    ok(
        nan_scale == 1.0 && nan_codes.iter().all(|c| c & 0x7f == 0x7f),
        "all-NaN F32 head retains scale1 and NaN code positions",
    );
    n
}

// ------------------------------------------------------------------ main
struct Opts {
    plan: bool,
    check: bool,
    filter: Option<String>,
}
fn parse() -> Result<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        filter: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--filter" => o.filter = Some(it.next().ok_or("--filter needs a substring")?),
            _ => return Err(format!("unknown argument {a}; usage: fa_pm_fp8_oracle [--plan|--self-check] [--filter SUBSTR]").into()),
        }
    }
    Ok(o)
}
/// Same NEW-kernel controlled dynamic-LDS comparison on the actually loaded FP8 direct
/// symbol (`fa_prefill_pm::FP8_DIRECT`, block 160, static LDS fixed 0): actual 32,768 B
/// dynamic vs 65,536 B dynamic control. Not the old PM driver proof. Reports the real
/// `Gpu::occupancy_max_active_blocks` results; call only after the PM module is loaded
/// (a PM direct launch has run), so the loaded-function lookup resolves.
fn occupancy_control(gpu: &Gpu) -> Value {
    let block = [DIRECT_BLOCK, 1, 1];
    let probe = |dynamic_lds: u32| match gpu.occupancy_max_active_blocks(
        fa_prefill_pm::FP8_DIRECT,
        block,
        dynamic_lds,
    ) {
        Ok(v) => json!({"ok": true, "max_active_blocks": v}),
        Err(e) => json!({"ok": false, "error": e.to_string()}),
    };
    json!({"kind": "occupancy_control",
        "kernel": fa_prefill_pm::FP8_DIRECT, "module": fa_prefill_pm::FP8_MODULE,
        "block": block, "static_lds_bytes": 0,
        "actual_dynamic_lds_32768": probe(32_768),
        "control_dynamic_lds_65536": probe(65_536),
        "note": "same NEW-kernel controlled dynamic-LDS comparison; not the old PM driver proof"})
}

fn run() -> Result<bool> {
    let n = self_check();
    let o = parse()?;
    if o.plan {
        plan();
        return Ok(true);
    }
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":n,"gpu_launched":false})
        );
        return Ok(true);
    }
    // Native qualification is opt-in; legacy no-filter/default behavior is
    // unchanged. An explicitly requested native filter must select real cells.
    let native_requested = o
        .filter
        .as_ref()
        .map_or(false, |f| f.contains("a3b_native"));
    let native_candidates = native_cells();
    let native_total = native_candidates.len();
    let native_selected: Vec<NativeCell> = native_candidates
        .into_iter()
        .filter(|c| {
            native_requested
                && o.filter
                    .as_ref()
                    .map_or(false, |f| c.name().contains(f.as_str()))
        })
        .collect();
    let native_full_selection = native_selected.len() == native_total;
    if native_requested && native_selected.is_empty() {
        return Err("requested a3b_native filter selected zero cases".into());
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let src = hip_source();
    let src_sha = sha_hex(src.as_bytes());
    gpu.ensure_kernel_public(HIP_DIRECT, &src, HIP_DIRECT)?;
    gpu.ensure_kernel_public(HIP_DIRECT, &src, HIP_PRE)?;
    println!(
        "{}",
        json!({"kind":"provenance",
            "hip_twin":{"source_sha256":src_sha,"production_hip_sha256":sha_hex(HIP_BODY.as_bytes()),
                "defines":["HIPFIRE_FA2_KMODE=8","HIPFIRE_FA2_Q0_A3B=1"],"symbols":[HIP_PRE,HIP_DIRECT],
                "compiled_by_this_tool":true,"compiler":"Gpu::ensure_kernel_public (repo hipcc JIT)"},
            "pm_fp8":{"module":fa_prefill_pm::FP8_MODULE,"symbol":fa_prefill_pm::FP8_DIRECT,
                "image_bytes":fa_prefill_pm::FP8_IMAGE.len(),"image_sha256":sha_hex(fa_prefill_pm::FP8_IMAGE)},
            "pm_preconvert":{"module":fa_prefill_pm::MODULE,"symbol":fa_prefill_pm::PRECONVERT,
                "image_bytes":fa_prefill_pm::IMAGE.len(),"image_sha256":sha_hex(fa_prefill_pm::IMAGE),
                "note":"shared with the frozen Q8 PM module; its pinned lab object is untouched"}})
    );
    // Native A3B qualification provenance: candidate module (new production A/B
    // source, JIT) and the oracle-local independent reference (oracle-local
    // `include_str!` HIP source, private unique module key, existing LAB
    // `ensure_kernel_public` pattern). No A-file edits; the reference module
    // key can never collide with old f16/Q0 symbols.
    let native_src = NATIVE_REF_BODY.to_string();
    let native_src_sha = sha_hex(native_src.as_bytes());
    if native_requested {
        gpu.ensure_kernel_public(NATIVE_REF_MODULE, &native_src, NATIVE_REF_DIRECT)?;
        gpu.ensure_kernel_public(NATIVE_REF_MODULE, &native_src, NATIVE_REF_RANGE)?;
        println!(
            "{}",
            json!({"kind":"native_provenance",
            "candidate":{"source_const":"ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_Q8_A3B_GFX1201_SRC",
                "defines":["HIPFIRE_FA2_QRESIDENT_V2_Q8_A3B=1","HIPFIRE_FA2_QRESIDENT_V2_Q8=1","HIPFIRE_FA2_FP8=1","HIPFIRE_FA2_KMODE=8"],
                "symbols":[NATIVE_DIRECT, NATIVE_RANGE],
                "module_key":"attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201",
                "abi":"6ptr(q_codes,q_scales,k,v,out,pos)+4i32(H16,KV2,D256,batch)+f32(1/16)=68B",
                "grid":"[ceil(min(batch,512)*8/384),2,ceil(batch/512)]","block":[NATIVE_BLOCK,1,1],
                "lds":NATIVE_LDS,"heads":"H16/KV2/GQA8","kt":48,
                "compiled_by":"B runtime (qresident_a3b_launch); oracle never touches kernels.rs/registry"},
            "reference":{"source_sha256":native_src_sha,
                "path":"kernels/src/attention_fp8_a3b_qresident_independent_ref.gfx1201.hip",
                "defines":[],"symbols":[NATIVE_REF_DIRECT,NATIVE_REF_RANGE],
                "module_key":NATIVE_REF_MODULE,"compiled_by_this_tool":true,
                "compiler":"Gpu::ensure_kernel_public (repo hipcc JIT, oracle-local include_str)",
                "body":"independent canonical one-query/head FP8-WMMA/KT48 online P-quant; global byte gathers",
                "grid":"[batch*16,1,1]", "block":[32,1,1], "lds":0,
                "independent_of_candidate":true}})
        );
    }
    let mut cells: Vec<Cell> = oracle_cells()
        .into_iter()
        .filter(|c| {
            o.filter
                .as_ref()
                .map_or(true, |f| c.name().contains(f.as_str()))
        })
        .collect();
    // Runtime-written long-context cells last: context exactly 16384, uploaded positions include
    // 16383, `Fixture::Std` with two changed-position rounds and both production K/V writes.
    cells.extend(long_cells().into_iter().filter(|c| {
        o.filter.as_ref().map_or(true, |f| {
            f.as_str() == "long" || c.name().contains(f.as_str())
        })
    }));
    let ieee_selected: Vec<IeeeCase> = ieee_cases()?
        .into_iter()
        .filter(|ic| {
            o.filter
                .as_ref()
                .map_or(true, |f| ic.name.contains(f.as_str()))
        })
        .collect();
    if cells.is_empty() && ieee_selected.is_empty() && native_selected.is_empty() {
        return Err("--filter matched no cell".into());
    }
    // The arena always covers the independent extents (at most 8 rows / 66 contexts): with no
    // ordinary cell selected, a synthetic cover cell sizes it.
    let cover = Cell {
        rows: 8,
        start: 58,
        segment: 8,
        fixture: Fixture::Std,
        source: Source::Runtime,
    };
    let mut sized = cells.clone();
    sized.push(cover);
    let arena = Arena::new(&mut gpu, &sized)?;
    let (mut finite_cells, mut finite_identical, mut nan_cells, mut nan_classified, mut failures) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut ieee_cells, mut ieee_passed) = (0usize, 0usize);
    // Independent native semantic proof first: the first failure returns nonzero before any ordinary
    // or long-context launch.
    let mut ieee_ok = true;
    for ic in &ieee_selected {
        ieee_cells += 1;
        match ieee_round(&mut gpu, &arena, ic) {
            Ok(r) => {
                let ok = r["pass"] == true;
                ieee_passed += ok as usize;
                failures += !ok as usize;
                println!(
                    "{}",
                    json!({"kind":"ieee_case","case":ic.name.clone(),"rows":ic.positions.len(),"positions":ic.positions.clone(),
                        "allocated_ctx":ic.k_cache.len() / POS_BYTES,"assertion":ic.expected.kind(),"rounds":r,"pass":ok})
                );
                if !ok {
                    ieee_ok = false;
                    break;
                }
            }
            Err(e) => {
                failures += 1;
                println!(
                    "{}",
                    json!({"kind":"ieee_case","case":ic.name.clone(),"error":e.to_string(),"pass":false})
                );
                ieee_ok = false;
                break;
            }
        }
    }
    if ieee_ok {
        for c in &cells {
            let mut rounds = Vec::new();
            let mut ok = true;
            for round in 0..2 {
                let r = oracle_round(&mut gpu, &arena, c, round)?;
                ok &= r["pass"] == true;
                rounds.push(r);
            }
            println!(
                "{}",
                json!({"kind":"oracle_cell","cell":c.name(),"rows":c.rows,"start_pos":c.start,"segment_rows":c.segment,
                    "segments":c.segments().len(),"fixture":c.fixture.name(),"source":c.source.name(),
                    "module_key":fa_prefill_pm::FP8_MODULE,"rounds":rounds,
                    "classification":if c.source == Source::Nan {"nan_codes_separate"} else {"finite_byte_gate"},
                    "pass":ok})
            );
            if c.source == Source::Nan {
                nan_cells += 1;
                nan_classified += ok as usize;
            } else {
                finite_cells += 1;
                finite_identical += ok as usize;
            }
            failures += !ok as usize;
            // Fail fast on an actual ordinary-case FAIL: stop immediately so a real GPU
            // divergence is not buried under later cells. NaN expected-classification
            // variation is not a failure here: `ok` already carries the NaN-aware
            // classification gate, and all finite/classification/poison checks are unchanged.
            if !ok {
                break;
            }
        }
    }
    // Every selected case is mandatory. Complete-family runs also require the
    // actual fused-prep proof; cell-only continuations must join its unchanged
    // source-bound prior receipt externally and cannot claim family completion.
    let mut native_producer = Value::Null;
    let (mut native_cells_n, mut native_passed, mut native_finite, mut native_finite_ok) =
        (0usize, 0usize, 0usize, 0usize);
    if !native_selected.is_empty() {
        if native_full_selection {
            native_producer = native_producer_proof(&mut gpu)?;
            println!(
                "{}",
                json!({"kind":"native_gpu_producer", "result":native_producer})
            );
            if !pass(&native_producer) {
                return Err("actual native fused-prep producer gate failed".into());
            }
        }
        let native_arena = NativeArena::new(&mut gpu, &native_selected)?;
        println!(
            "{}",
            json!({"kind": "native_arena",
                "buffers": NATIVE_BUFFER_NAMES,
                "arena_bytes_each": native_arena.sizes(),
                "quirk": "native arena covers direct + one 4-row-overhang range probe; guards excluded"})
        );
        for nc in &native_selected {
            native_cells_n += 1;
            let mut rounds = Vec::new();
            let mut ok = true;
            for round in 0..2 {
                match native_round(&mut gpu, &native_arena, nc, round) {
                    Ok(r) => {
                        ok &= r["pass"] == true;
                        rounds.push(r);
                    }
                    Err(e) => {
                        // Requested family FAILS CLOSED: any module/launch/
                        // reference/producer error is a FAIL, never Green.
                        ok = false;
                        rounds.push(json!({"round": round, "error": e.to_string(), "pass": false}));
                        break;
                    }
                }
            }
            let finite_gate = !nc.kind.nonfinite();
            if finite_gate {
                native_finite += 1;
                native_finite_ok += ok as usize;
            }
            native_passed += ok as usize;
            failures += !ok as usize;
            println!(
                "{}",
                json!({"kind": "native_cell", "cell": nc.name(), "rows": nc.rows,
                    "start": nc.start, "kind": nc.kind.name(), "context": nc.ctx(),
                    "rounds": rounds,
                    "classification":if nc.kind.nonfinite() {"nonfinite_positions_separate"}
                        else {"finite_byte_gate"},
                    "pass": ok})
            );
            if !ok {
                break;
            }
        }
    }
    let native_finite_selected = native_selected
        .iter()
        .filter(|c| !c.kind.nonfinite())
        .count();
    let native_acceptance = native_requested
        && !native_selected.is_empty()
        && (!native_full_selection || pass(&native_producer))
        && native_cells_n == native_selected.len()
        && native_passed == native_selected.len()
        && native_finite == native_finite_selected
        && native_finite_ok == native_finite_selected;
    let all_pass = failures == 0 && (!native_requested || native_acceptance);
    // The ordinary loop above ran at least one PM direct launch, so the actual PM module
    // is loaded and the loaded-function occupancy lookup resolves.
    if ieee_ok && finite_cells + nan_cells > 0 {
        println!("{}", occupancy_control(&gpu));
    }
    println!(
        "{}",
        json!({"summary":true,"mode":"oracle","hip_twin_source_sha256":src_sha,
            "pm_fp8_image_sha256":sha_hex(fa_prefill_pm::FP8_IMAGE),"cells":cells.len(),
            "finite_cells":finite_cells,"finite_byte_identical":finite_identical,
            "nan_cells":nan_cells,"nan_classifications_passed":nan_classified,"failures":failures,
            "ieee_cells":ieee_cells,"ieee_passed":ieee_passed,"independent_failed_before_ordinary":!ieee_ok,
            "native_cells":native_cells_n,"native_passed":native_passed,
            "native_finite_cells":native_finite,"native_finite_byte_identical":native_finite_ok,
            "native_requested":native_requested,"native_selected":native_selected.len(),
            "native_finite_selected":native_finite_selected,"native_gpu_producer":native_producer,
            "native_acceptance":native_acceptance,
            "native_total":native_total,
            "native_producer_executed":native_full_selection && native_requested,
            "native_family_complete_this_run":native_full_selection && native_acceptance,
            "prior_producer_join_required_for_family_closure":native_requested && !native_full_selection,
            "native_coverage":if native_requested {native_coverage_map()} else {Value::Null},
            "native_reference_sha256":native_src_sha})
    );
    println!("all_byte_identical {}", finite_cells == finite_identical);
    if native_requested {
        println!(
            "all_native_byte_identical {}",
            native_acceptance && native_finite > 0 && native_finite == native_finite_ok
        );
    }
    println!("all_classifications_pass {}", all_pass);
    Ok(all_pass)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("fa_pm_fp8_oracle: {e}");
            std::process::exit(1);
        }
    }
}
