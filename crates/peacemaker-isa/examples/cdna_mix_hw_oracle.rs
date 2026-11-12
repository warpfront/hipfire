// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MI300X (gfx942) hardware oracle of the PeaceMaker CDNA interpreter for the promoted `v_fma_mix_f32`.
//!
//! Scope: `FmaMixF32` only, `NEG_HI = 0` (no `|abs|` source modifier), `NEG_LO` (all 8 masks), `CLAMP`, and every
//! source selection (`OP_SEL` 0..7 x `OP_SEL_HI` 0..7 = 64 combinations): 64 hipcc kernels x 16 `asm volatile`
//! instructions = 1024 variants. `v_fma_mix_f32` is the only op; `v_fma_mixlo/hi_f16`, `v_mad_mix*`, `|abs|`
//! and gfx950 are NOT covered and nothing here promotes them.
//!
//! Every variant runs on the card (generated HIP source, real `hipcc` code object, `hipModuleLoadData`) and in the
//! interpreter (`peacemaker_cdna::sim::exec_one` on a `Wave`, one `Inst::Vop3p` per 64 lanes) over the same
//! deterministic adversarial inputs (f32 / f16 lo / f16 hi special words: signed zeros, denormals, infinities,
//! qNaN and sNaN, extrema, fused-vs-unfused discriminators, random edge-exponent rows). The comparison is **strict
//! raw u32 bits**, NaN payload and sign included: any differing word is a mismatch and the exit status is nonzero.
//! NaN, zero, denormal and source-selection classes are only *diagnostics* (counted, sampled per class), never a
//! relaxation.
//!
//! MODE: the kernel writes the requested MODE (`FP_ROUND`=0, `FP_DENORM`, `DX10_CLAMP`, `IEEE`, `FP16_OVFL`) with
//! `s_setreg_b32`, reads it back with `s_getreg_b32` before and after the arithmetic, and the runner demands the
//! exact requested bits from every wave (otherwise the mode is reported as a hard error and not compared). The
//! interpreter `Mode` is built from the same bits. 32 modes: (denorm32, denorm16_64) in {(3,3),(0,0),(1,2),(2,1)} x
//! dx10_clamp x ieee x fp16_ovfl.
//!
//! Output hygiene: each launch runs twice with different byte poisons (0xa5, 0x5a) over output buffers surrounded by
//! 256-byte guard canaries; the two runs must agree word for word (an unwritten word would equal the poison), guards
//! must be intact. Hard errors (compile, load, launch, sync, readback, interpreter error, MODE readback) are recorded
//! in the receipt and fail the run; nothing is suppressed.
//!
//! Negative controls (default mode, whole data set, each MUST be detected): interpreter with flipped op_sel,
//! op_sel_hi, negation, clamp, denormal mode, an abs (`neg_hi`) modifier the hardware kernels never use, and a
//! single flipped hardware bit (exactly 1 mismatch). A control that is not detected fails the run.
//!
//! ```text
//! cdna_mix_hw_oracle --hip-work DIR --receipt FILE.json [--patterns N] [--dry-run]
//! ```
//! * `--patterns N`: rows per variant, multiple of 256, >= 13568 (default 16384).
//! * `--dry-run`: NO hardware, NO hipcc, NO HIP call. Only the generated source/inputs and the interpreter run; the
//!   controls compare mutated interpreters with the baseline interpreter. The receipt says
//!   `"hardware_evidence": false` and `"verdict": "DRY_RUN_NO_HARDWARE_CLAIM"`; it proves nothing about the card.
//!
//! Environment: `HIPCC` (default `hipcc`). Exit 0 iff no error, no strict mismatch, every control detected (dry-run:
//! no error and every control detected). The receipt is rewritten after every mode.

use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};
use peacemaker_cdna::enc::vop3p::{Vop3p, Vop3pOp};
use peacemaker_cdna::float::{f16_to_f32, f32_to_f16_rne};
use peacemaker_cdna::inst::Inst;
use peacemaker_cdna::operand::Src;
use peacemaker_cdna::sim::{exec_one, Config, Memory, Mode, Wave};
use peacemaker_cdna::Arch;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage: cdna_mix_hw_oracle --hip-work DIR --receipt FILE [--patterns N] [--dry-run]\n\
  --hip-work DIR  directory for the generated .hip source and the hipcc code object\n\
  --receipt FILE  receipt JSON (rewritten after every mode)\n\
  --patterns N    rows per variant (multiple of 256, >= 13568; default 16384)\n\
  --dry-run       NO hardware claim: interpreter + controls only, no hipcc, no HIP call";

const EXPECTED_ARCH: &str = "gfx942";
/// 8 op_sel x 8 op_sel_hi kernels; kernel k = op_sel * 8 + op_sel_hi.
const NK: usize = 64;
/// Variants per kernel: v = neg_lo * 2 + clamp.
const NV: usize = 16;
const BLOCK: usize = 256;
const GUARD: usize = 256;
const POISONS: [u8; 2] = [0xa5, 0x5a];
const SIM_POISON: u32 = 0xdead_beef;
const SEED: u64 = 0x6d69_785f_6f72_6163; // "mix_orac"
const MIN_PATTERNS: usize = 12288 + 1024 + 256;
const MAX_SAMPLES_PER_CLASS: usize = 4;
const MAX_FIRST: usize = 8;
const HIST_CAP: usize = 32;

// ----------------------------------------------------------------------------
// Helpers

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b)[..])
}
fn words_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}
fn h32(x: u32) -> String {
    format!("{x:#010x}")
}
fn is_nan(b: u32) -> bool {
    b & 0x7fff_ffff > 0x7f80_0000
}
fn is_snan(b: u32) -> bool {
    is_nan(b) && b & 0x0040_0000 == 0
}
fn is_denorm(b: u32) -> bool {
    b & 0x7f80_0000 == 0 && b & 0x007f_ffff != 0
}

struct Rng(u64);
impl Rng {
    /// splitmix64.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 48) as u16
    }
}

// ----------------------------------------------------------------------------
// Options

struct Opts {
    hip_work: PathBuf,
    receipt: PathBuf,
    dry_run: bool,
    patterns: usize,
}

fn parse_args() -> Result<Option<Opts>, String> {
    let mut hip_work = None;
    let mut receipt = None;
    let mut dry_run = false;
    let mut patterns = 16384usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--dry-run" => dry_run = true,
            "--hip-work" => hip_work = Some(PathBuf::from(it.next().ok_or("--hip-work needs DIR")?)),
            "--receipt" => receipt = Some(PathBuf::from(it.next().ok_or("--receipt needs FILE")?)),
            "--patterns" => {
                patterns = it
                    .next()
                    .ok_or("--patterns needs N")?
                    .parse()
                    .map_err(|e| format!("--patterns: {e}"))?
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if patterns % BLOCK != 0 || patterns < MIN_PATTERNS {
        return Err(format!(
            "--patterns must be a multiple of {BLOCK} and >= {MIN_PATTERNS}, got {patterns}"
        ));
    }
    Ok(Some(Opts {
        hip_work: hip_work.ok_or("--hip-work is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
        dry_run,
        patterns,
    }))
}

// ----------------------------------------------------------------------------
// Deterministic adversarial inputs

const F32S: [u32; 32] = [
    0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007f_ffff, 0x807f_ffff, 0x0040_0000,
    0x0080_0000, // zeros, denormals, min normal
    0x3f80_0000, 0xbf80_0000, 0x4000_0000, 0x3f00_0000, 0x4040_0000, 0x3f80_0001, 0x3f7f_ffff,
    0xbf80_0002, // +-1, 2, .5, 3, 1+-ulp
    0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000, 0x7f80_0001,
    0xffa5_a5a5, // f32 max, inf, qNaN, sNaN (payload)
    0x7fff_ffff, 0x477f_e000, 0x477f_f000, 0x4b80_0000, 0x7f00_0000, 0x3380_0000, 0x3880_0000,
    0x00bf_ffff, // NaN all ones, 65504, 65520, 2^24, 2^127, 2^-24, 2^-14, near-denormal
];
const HALVES: [u16; 12] = [
    0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x7c01,
];

fn special_words() -> [u32; 64] {
    let mut w = [0u32; 64];
    w[..32].copy_from_slice(&F32S);
    for k in 0..32usize {
        let lo = HALVES[k % 12];
        let mut hi = HALVES[(k * 5 + k / 12 + 1) % 12];
        if hi == lo {
            hi = HALVES[(k + 1) % 12];
        }
        w[32 + k] = u32::from(lo) | (u32::from(hi) << 16);
    }
    w
}

struct Inputs {
    n: usize,
    a: Vec<u32>,
    b: Vec<u32>,
    c: Vec<u32>,
    /// (name, start row, rows)
    blocks: Vec<(&'static str, usize, usize)>,
    /// Rows of the discriminator block whose three operands are f32 (valid for the f32/f32/f32 kernel).
    f32_disc_rows: Vec<usize>,
    disc_rows_discriminating: usize,
    disc_rows_exact_product: usize,
    disc_rows_unresolved: usize,
}

fn kind_operand(r: &mut Rng, kind: u8) -> (u32, f32) {
    match kind {
        0 => {
            let w = (u32::from(r.below(2) == 0) << 31)
                | ((100 + r.below(51) as u32) << 23)
                | (r.u32() & 0x7f_ffff);
            (w, f32::from_bits(w))
        }
        _ => {
            let h = (u16::from(r.below(2) == 0) << 15)
                | ((8 + r.below(15) as u16) << 10)
                | (r.u16() & 0x3ff);
            let other = r.u16();
            let w = if kind == 1 {
                (u32::from(other) << 16) | u32::from(h)
            } else {
                (u32::from(h) << 16) | u32::from(other)
            };
            (w, f16_to_f32(h))
        }
    }
}

fn edge_f32(r: &mut Rng) -> u32 {
    const E: [u32; 11] = [0, 1, 2, 3, 126, 127, 128, 150, 253, 254, 255];
    let m = match r.below(4) {
        0 => 0,
        1 => 0x7f_ffff,
        _ => r.u32() & 0x7f_ffff,
    };
    (u32::from(r.below(2) == 0) << 31) | (E[r.below(11) as usize] << 23) | m
}
fn edge_half(r: &mut Rng) -> u16 {
    const E: [u16; 7] = [0, 1, 14, 15, 16, 30, 31];
    let m = match r.below(4) {
        0 => 0,
        1 => 0x3ff,
        2 => 0x200,
        _ => r.u16() & 0x3ff,
    };
    (u16::from(r.below(2) == 0) << 15) | (E[r.below(7) as usize] << 10) | m
}
fn random_operand(r: &mut Rng, w: &[u32; 64]) -> u32 {
    match r.below(4) {
        0 => r.u32(),
        1 => w[r.below(64) as usize],
        2 => edge_f32(r),
        _ => u32::from(edge_half(r)) | (u32::from(edge_half(r)) << 16),
    }
}

fn gen_inputs(n: usize) -> Inputs {
    let w = special_words();
    let mut r = Rng(SEED);
    let (mut a, mut b, mut c) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut blocks = Vec::new();
    let push = |a: &mut Vec<u32>, b: &mut Vec<u32>, c: &mut Vec<u32>, x: u32, y: u32, z: u32| {
        a.push(x);
        b.push(y);
        c.push(z);
    };
    blocks.push(("ab_exhaustive_c_strided", a.len(), 4096));
    for i in 0..64 {
        for j in 0..64 {
            push(&mut a, &mut b, &mut c, w[i], w[j], w[(i * 5 + j * 11 + 1) % 64]);
        }
    }
    blocks.push(("bc_exhaustive_a_strided", a.len(), 4096));
    for j in 0..64 {
        for k in 0..64 {
            push(&mut a, &mut b, &mut c, w[(j * 3 + k * 7 + 2) % 64], w[j], w[k]);
        }
    }
    blocks.push(("ac_exhaustive_b_strided", a.len(), 4096));
    for i in 0..64 {
        for k in 0..64 {
            push(&mut a, &mut b, &mut c, w[i], w[(i * 9 + k * 5 + 3) % 64], w[k]);
        }
    }
    // Fused discriminators: 3 hand-made f32 rows (fused = +-2^-46 / 2^-48, unfused = 0), then 1021 rows over the
    // 27 operand kinds (f32 / f16 lo / f16 hi per source) rejection-sampled so round32(fma) != round32(round32(a*b)+c)
    // (impossible when both a and b are f16: the product is exact in f32; those rows are kept, counted separately).
    let d0 = a.len();
    blocks.push(("fma_discriminators", d0, 1024));
    let mut f32_disc_rows = Vec::new();
    let (mut disc, mut exact, mut unresolved) = (3usize, 0usize, 0usize);
    push(&mut a, &mut b, &mut c, 0x3f80_0001, 0x3f80_0001, 0xbf80_0002);
    push(&mut a, &mut b, &mut c, 0xbf80_0001, 0x3f80_0001, 0x3f80_0002);
    push(&mut a, &mut b, &mut c, 0x3f7f_ffff, 0x3f7f_ffff, 0xbf7f_fffe);
    f32_disc_rows.extend([0, 1, 2]);
    for i in 3..1024usize {
        let q = (i - 3) % 27;
        let (ka, kb, kc) = ((q / 9) as u8, (q / 3 % 3) as u8, (q % 3) as u8);
        let product_exact = ka != 0 && kb != 0;
        let mut last = (0u32, 0u32, 0u32);
        let mut found = false;
        for _ in 0..400 {
            let (wa, fa) = kind_operand(&mut r, ka);
            let (wb, fb) = kind_operand(&mut r, kb);
            let (wc, fc) = if r.below(2) == 0 {
                // cancellation: c = -(a*b) rounded to the c format.
                let p = -(fa * fb);
                if kc == 0 {
                    (p.to_bits(), p)
                } else {
                    let hc = f32_to_f16_rne(p);
                    let other = r.u16();
                    let word = if kc == 1 {
                        (u32::from(other) << 16) | u32::from(hc)
                    } else {
                        (u32::from(hc) << 16) | u32::from(other)
                    };
                    (word, f16_to_f32(hc))
                }
            } else {
                kind_operand(&mut r, kc)
            };
            last = (wa, wb, wc);
            let fused = fa.mul_add(fb, fc);
            let unfused = fa * fb + fc;
            if product_exact {
                found = true;
                break;
            }
            if fused.is_finite() && unfused.is_finite() && fused.to_bits() != unfused.to_bits() {
                found = true;
                break;
            }
        }
        if product_exact {
            exact += 1;
        } else if found {
            disc += 1;
        } else {
            unresolved += 1;
        }
        if ka == 0 && kb == 0 && kc == 0 {
            f32_disc_rows.push(i);
        }
        push(&mut a, &mut b, &mut c, last.0, last.1, last.2);
    }
    blocks.push(("random_edge", a.len(), n - a.len()));
    while a.len() < n {
        let x = random_operand(&mut r, &w);
        let y = random_operand(&mut r, &w);
        let z = random_operand(&mut r, &w);
        push(&mut a, &mut b, &mut c, x, y, z);
    }
    Inputs {
        n,
        a,
        b,
        c,
        blocks,
        f32_disc_rows,
        disc_rows_discriminating: disc,
        disc_rows_exact_product: exact,
        disc_rows_unresolved: unresolved,
    }
}

impl Inputs {
    fn block_of(&self, row: usize) -> &'static str {
        self.blocks
            .iter()
            .find(|(_, s, l)| row >= *s && row < s + l)
            .map(|b| b.0)
            .unwrap_or("?")
    }
}

// ----------------------------------------------------------------------------
// Variants and generated HIP source

fn asm_line(dst: &str, s: [&str; 3], op_sel: u8, op_sel_hi: u8, neg_lo: u8, clamp: bool) -> String {
    let n = |i: u8| if neg_lo >> i & 1 != 0 { "-" } else { "" };
    format!(
        "v_fma_mix_f32 {dst}, {}{}, {}{}, {}{} op_sel:[{},{},{}] op_sel_hi:[{},{},{}]{}",
        n(0),
        s[0],
        n(1),
        s[1],
        n(2),
        s[2],
        op_sel & 1,
        op_sel >> 1 & 1,
        op_sel >> 2 & 1,
        op_sel_hi & 1,
        op_sel_hi >> 1 & 1,
        op_sel_hi >> 2 & 1,
        if clamp { " clamp" } else { "" }
    )
}

/// 0..27: base-3 id of the per-source selection (f32 / f16 lo / f16 hi), matching `sel_signature`.
fn source_class_id(op_sel: u8, op_sel_hi: u8) -> usize {
    (0..3)
        .map(|i| match (op_sel_hi >> i & 1, op_sel >> i & 1) {
            (0, _) => 0usize,
            (_, 0) => 1,
            _ => 2,
        } * 3usize.pow(i as u32))
        .sum()
}

fn sel_signature(op_sel: u8, op_sel_hi: u8) -> String {
    (0..3)
        .map(|i| match (op_sel_hi >> i & 1, op_sel >> i & 1) {
            (0, _) => "f32",
            (_, 0) => "f16lo",
            _ => "f16hi",
        })
        .collect::<Vec<_>>()
        .join(",")
}

const SRC_HEADER: &str = r#"// Generated by cdna_mix_hw_oracle (peacemaker-isa example). One kernel per (op_sel, op_sel_hi);
// 16 volatile v_fma_mix_f32 per kernel = neg_lo(8) x clamp(2): out[v * npat + gid].
#include <hip/hip_runtime.h>

__device__ __forceinline__ void mix_set_mode(unsigned lo10, unsigned ovfl) {
  asm volatile("s_setreg_b32 hwreg(HW_REG_MODE, 0, 10), %0\n\ts_setreg_b32 hwreg(HW_REG_MODE, 23, 1), %1\n\ts_nop 7\n\ts_nop 7"
               :: "s"(lo10), "s"(ovfl) : "memory");
}
__device__ __forceinline__ unsigned mix_get_mode() {
  unsigned lo, hi;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 10)" : "=s"(lo));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 23, 1)" : "=s"(hi));
  return lo | (hi << 23);
}

// Launch-default MODE: [0]=bits 9:0, [1]=bits 22:10, [2]=bit 23.
extern "C" __global__ void mix_mode_probe(unsigned* out) {
  unsigned lo, mid, hi;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 10)" : "=s"(lo));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 10, 13)" : "=s"(mid));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 23, 1)" : "=s"(hi));
  if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = lo; out[1] = mid; out[2] = hi; }
}
"#;

fn gen_source() -> String {
    let mut s = String::from(SRC_HEADER);
    for k in 0..NK {
        let (os, oh) = ((k / 8) as u8, (k % 8) as u8);
        let _ = write!(
            s,
            "\n// op_sel={os} op_sel_hi={oh} sources=({})\nextern \"C\" __global__ void __launch_bounds__({BLOCK}) mix_s{os}_h{oh}(\n    const unsigned* __restrict__ in0, const unsigned* __restrict__ in1, const unsigned* __restrict__ in2,\n    unsigned* __restrict__ out, unsigned* __restrict__ mrb, unsigned npat, unsigned mode_lo, unsigned mode_hi) {{\n  const unsigned gid = blockIdx.x * {BLOCK}u + threadIdx.x;\n  mix_set_mode(mode_lo, mode_hi);\n  const unsigned pre = mix_get_mode();\n  const unsigned a = in0[gid], b = in1[gid], c = in2[gid];\n  unsigned r;\n",
            sel_signature(os, oh)
        );
        for v in 0..NV {
            let (nl, cl) = ((v >> 1) as u8, v & 1 == 1);
            let _ = write!(
                s,
                "  asm volatile(\"{}\" : \"=v\"(r) : \"v\"(a), \"v\"(b), \"v\"(c));\n  out[{v}u * npat + gid] = r;\n",
                asm_line("%0", ["%1", "%2", "%3"], os, oh, nl, cl)
            );
        }
        s.push_str("  const unsigned post = mix_get_mode();\n  if ((threadIdx.x & 63u) == 0u) { mrb[2u * (gid >> 6)] = pre; mrb[2u * (gid >> 6) + 1u] = post; }\n}\n");
    }
    s
}

// ----------------------------------------------------------------------------
// MODE

fn mode_bits(d32: u32, d16: u32, dx10: u32, ieee: u32, ovfl: u32) -> u32 {
    d32 << 4 | d16 << 6 | dx10 << 8 | ieee << 9 | ovfl << 23
}
fn mode_list() -> Vec<u32> {
    let mut v = Vec::new();
    for (d32, d16) in [(3, 3), (0, 0), (1, 2), (2, 1)] {
        for dx10 in [1, 0] {
            for ieee in [1, 0] {
                for ovfl in [0, 1] {
                    v.push(mode_bits(d32, d16, dx10, ieee, ovfl));
                }
            }
        }
    }
    v
}
fn sim_mode(bits: u32) -> Mode {
    Mode {
        round32: (bits & 3) as u8,
        round16_64: (bits >> 2 & 3) as u8,
        denorm32: (bits >> 4 & 3) as u8,
        denorm16_64: (bits >> 6 & 3) as u8,
        dx10_clamp: bits >> 8 & 1 != 0,
        ieee: bits >> 9 & 1 != 0,
        fp16_ovfl: bits >> 23 & 1 != 0,
    }
}
fn mode_json(bits: u32) -> Value {
    json!({
        "bits": h32(bits), "fp_round32": bits & 3, "fp_round16_64": bits >> 2 & 3,
        "fp_denorm32": bits >> 4 & 3, "fp_denorm16_64": bits >> 6 & 3,
        "dx10_clamp": bits >> 8 & 1, "ieee": bits >> 9 & 1, "fp16_ovfl": bits >> 23 & 1,
    })
}

// ----------------------------------------------------------------------------
// Interpreter

#[derive(Clone, Copy, Default)]
struct Mutation {
    xor_op_sel: u8,
    xor_op_sel_hi: u8,
    xor_neg_lo: u8,
    neg_hi: u8,
    flip_clamp: bool,
    flip_denorm: bool,
}

fn sim_kernel(w: &mut Wave, k: usize, mode: Mode, mu: Mutation, inp: &Inputs) -> Result<Vec<u32>, String> {
    let mut mode = mode;
    if mu.flip_denorm {
        mode.denorm32 ^= 3;
        mode.denorm16_64 ^= 3;
    }
    w.mode = mode;
    let cfg = Config::default();
    let n = inp.n;
    let mut out = vec![0u32; NV * n];
    for v in 0..NV {
        let inst = Inst::Vop3p(Vop3p {
            op: Vop3pOp::FmaMixF32,
            dst: Src::Vgpr(4),
            src: [Some(Src::Vgpr(1)), Some(Src::Vgpr(2)), Some(Src::Vgpr(3))],
            op_sel: (k / 8) as u8 ^ mu.xor_op_sel,
            op_sel_hi: (k % 8) as u8 ^ mu.xor_op_sel_hi,
            neg_lo: (v >> 1) as u8 ^ mu.xor_neg_lo,
            neg_hi: mu.neg_hi,
            clamp: (v & 1 == 1) ^ mu.flip_clamp,
        });
        for chunk in 0..n / 64 {
            let r = chunk * 64..chunk * 64 + 64;
            w.v[1].copy_from_slice(&inp.a[r.clone()]);
            w.v[2].copy_from_slice(&inp.b[r.clone()]);
            w.v[3].copy_from_slice(&inp.c[r.clone()]);
            w.v[4] = [SIM_POISON; 64];
            exec_one(w, &inst, &mut [], &mut Memory::default(), &cfg)
                .map_err(|e| format!("interpreter kernel {k} variant {v} chunk {chunk}: {e:?}"))?;
            out[v * n + r.start..v * n + r.end].copy_from_slice(&w.v[4]);
        }
    }
    Ok(out)
}

fn sim_all(mode: Mode, mu: Mutation, inp: &Inputs) -> Result<Vec<Vec<u32>>, String> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || -> Result<Vec<(usize, Vec<u32>)>, String> {
                    let mut w = Wave::new(Arch::Gfx942, 0, SIM_POISON);
                    let mut r = Vec::new();
                    for k in (t..NK).step_by(threads) {
                        r.push((k, sim_kernel(&mut w, k, mode, mu, inp)?));
                    }
                    Ok(r)
                })
            })
            .collect();
        let mut out = vec![Vec::new(); NK];
        let mut err = None;
        for h in hs {
            match h.join() {
                Ok(Ok(v)) => {
                    for (k, o) in v {
                        out[k] = o;
                    }
                }
                Ok(Err(e)) => err = Some(err.unwrap_or(e)),
                Err(p) => err = Some(err.unwrap_or(format!("interpreter thread panic: {}", panic_msg(p)))),
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    })
}

// ----------------------------------------------------------------------------
// Comparator: strict raw u32 bits; classes are diagnostics only.

fn classify(hw: u32, sim: u32) -> &'static str {
    let (hn, sn) = (is_nan(hw), is_nan(sim));
    if hn && sn {
        return "both_nan";
    }
    if hn {
        return "hw_nan_sim_not";
    }
    if sn {
        return "sim_nan_hw_not";
    }
    let (hm, sm) = (hw & 0x7fff_ffff, sim & 0x7fff_ffff);
    if hm == 0 && sm == 0 {
        return "zero_vs_zero_sign";
    }
    if hm == sm {
        return "sign_only_nonzero";
    }
    if is_denorm(hw) || is_denorm(sim) || (hm == 0 && sm < 0x0080_0000) || (sm == 0 && hm < 0x0080_0000) {
        return "denormal_or_zero_vs_denormal";
    }
    if hw >> 31 == sim >> 31 && hw.abs_diff(sim) == 1 {
        return "finite_one_ulp";
    }
    "finite_other"
}

#[derive(Default)]
struct Hist {
    m: BTreeMap<u32, u64>,
    other: u64,
}
impl Hist {
    fn add(&mut self, b: u32) {
        if let Some(c) = self.m.get_mut(&b) {
            *c += 1;
        } else if self.m.len() < HIST_CAP {
            self.m.insert(b, 1);
        } else {
            self.other += 1;
        }
    }
    fn json(&self) -> Value {
        json!({ "distinct_capped": self.m.iter().map(|(k, v)| json!({ "bits": h32(*k), "count": v })).collect::<Vec<_>>(), "other": self.other })
    }
}

struct Cmp {
    rows: u64,
    mismatches: u64,
    by_kernel: Vec<u64>,
    by_class: BTreeMap<&'static str, u64>,
    both_nan_sign_differs: u64,
    both_nan_payload_differs: u64,
    ref_nan: u64,
    ref_snan: u64,
    sim_nan: u64,
    sim_snan: u64,
    ref_nan_hist: Hist,
    sim_nan_hist: Hist,
    first: Vec<Value>,
    samples_by_class: BTreeMap<&'static str, Vec<Value>>,
    samples_by_source_class: BTreeMap<String, Value>,
}

#[allow(clippy::too_many_arguments)]
fn sample(
    inp: &Inputs,
    k: usize,
    v: usize,
    row: usize,
    hw: u32,
    sim: u32,
    cls: &'static str,
    bits: u32,
) -> Value {
    let (os, oh, nl, cl) = ((k / 8) as u8, (k % 8) as u8, (v >> 1) as u8, v & 1 == 1);
    json!({
        "kernel": k, "variant": v, "op_sel": os, "op_sel_hi": oh, "neg_lo": nl, "clamp": cl,
        "source_class": sel_signature(os, oh),
        "asm": asm_line("v0", ["v1", "v2", "v3"], os, oh, nl, cl),
        "row": row, "block": inp.block_of(row), "mode": h32(bits),
        "in0": h32(inp.a[row]), "in1": h32(inp.b[row]), "in2": h32(inp.c[row]),
        "ref": h32(hw), "sim": h32(sim), "class": cls,
    })
}

/// `refv` is the hardware (or, in dry-run controls, the baseline interpreter); `sim` the interpreter under test.
fn compare(refv: &[Vec<u32>], sim: &[Vec<u32>], inp: &Inputs, bits: u32) -> Cmp {
    let n = inp.n;
    let mut c = Cmp {
        rows: 0,
        mismatches: 0,
        by_kernel: vec![0; NK],
        by_class: BTreeMap::new(),
        both_nan_sign_differs: 0,
        both_nan_payload_differs: 0,
        ref_nan: 0,
        ref_snan: 0,
        sim_nan: 0,
        sim_snan: 0,
        ref_nan_hist: Hist::default(),
        sim_nan_hist: Hist::default(),
        first: Vec::new(),
        samples_by_class: BTreeMap::new(),
        samples_by_source_class: BTreeMap::new(),
    };
    let mut seen_src = [false; 54];
    for k in 0..NK {
        let (rk, sk) = (&refv[k], &sim[k]);
        c.rows += rk.len() as u64;
        for (i, (&h, &s)) in rk.iter().zip(sk.iter()).enumerate() {
            if is_nan(h) {
                c.ref_nan += 1;
                c.ref_snan += u64::from(is_snan(h));
                c.ref_nan_hist.add(h);
            }
            if is_nan(s) {
                c.sim_nan += 1;
                c.sim_snan += u64::from(is_snan(s));
                c.sim_nan_hist.add(s);
            }
            if h == s {
                continue;
            }
            c.mismatches += 1;
            c.by_kernel[k] += 1;
            let cls = classify(h, s);
            *c.by_class.entry(cls).or_insert(0) += 1;
            if cls == "both_nan" {
                c.both_nan_sign_differs += u64::from((h ^ s) >> 31 != 0);
                c.both_nan_payload_differs += u64::from((h ^ s) & 0x7fff_ffff != 0);
            }
            let (v, row) = (i / n, i % n);
            let want_first = c.first.len() < MAX_FIRST;
            let by_class = c.samples_by_class.entry(cls).or_default();
            let want_class = by_class.len() < MAX_SAMPLES_PER_CLASS;
            let src_id = source_class_id((k / 8) as u8, (k % 8) as u8) * 2 + usize::from(is_nan(h) || is_nan(s));
            let want_src = !seen_src[src_id];
            if want_first || want_class || want_src {
                let j = sample(inp, k, v, row, h, s, cls, bits);
                if want_first {
                    c.first.push(j.clone());
                }
                if want_class {
                    c.samples_by_class.get_mut(cls).expect("entry").push(j.clone());
                }
                if want_src {
                    seen_src[src_id] = true;
                    c.samples_by_source_class.insert(
                        format!("{}|{}", sel_signature((k / 8) as u8, (k % 8) as u8), if is_nan(h) || is_nan(s) { "nan" } else { "non_nan" }),
                        j,
                    );
                }
            }
        }
    }
    c
}

impl Cmp {
    fn json(&self) -> Value {
        json!({
            "comparator": "strict raw u32 bits (NaN payload and sign included); classes are diagnostics only",
            "rows_compared": self.rows, "mismatches": self.mismatches,
            "mismatches_by_kernel": self.by_kernel,
            "mismatches_by_class": self.by_class,
            "nan": {
                "mismatches_both_nan": self.by_class.get("both_nan").copied().unwrap_or(0),
                "both_nan_sign_differs": self.both_nan_sign_differs,
                "both_nan_payload_differs": self.both_nan_payload_differs,
                "ref_nan_results": self.ref_nan, "ref_snan_results": self.ref_snan,
                "sim_nan_results": self.sim_nan, "sim_snan_results": self.sim_snan,
                "ref_nan_histogram": self.ref_nan_hist.json(), "sim_nan_histogram": self.sim_nan_hist.json(),
            },
            "first_mismatches": self.first,
            "samples_by_class": self.samples_by_class,
            "samples_by_source_class_and_nan": self.samples_by_source_class,
        })
    }
}

fn hash_outputs(o: &[Vec<u32>]) -> String {
    let mut h = Sha256::new();
    for k in o {
        h.update(words_bytes(k));
    }
    hex(&h.finalize()[..])
}

// ----------------------------------------------------------------------------
// Card

fn hip_blob(explicit: &[u8], grid: u32, block: u32) -> Vec<u8> {
    let mut v = explicit.to_vec();
    while v.len() % 8 != 0 {
        v.push(0);
    }
    for x in [grid, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    for x in [block as u16, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    v.resize(v.len().max(256), 0);
    v
}

struct Launched {
    payload: Vec<u32>,
    guards_ok: bool,
    mrb: Vec<u32>,
}

struct HwMode {
    out: Vec<Vec<u32>>,
    json: Value,
    errors: Vec<String>,
}

struct Card {
    rt: HipRuntime,
    _module: Module,
    kernels: Vec<Function>,
    probe: Function,
    d_in: Vec<DeviceBuffer>,
    d_out: DeviceBuffer,
    d_mrb: DeviceBuffer,
    d_probe: DeviceBuffer,
    n: usize,
}

impl Card {
    fn new(rt: HipRuntime, hsaco: &[u8], inp: &Inputs) -> Result<Card, String> {
        let module = rt
            .module_load_data(hsaco)
            .map_err(|e| format!("hipModuleLoadData: {e}"))?;
        let mut kernels = Vec::new();
        for k in 0..NK {
            let name = format!("mix_s{}_h{}", k / 8, k % 8);
            kernels.push(
                rt.module_get_function(&module, &name)
                    .map_err(|e| format!("hipModuleGetFunction({name}): {e}"))?,
            );
        }
        let probe = rt
            .module_get_function(&module, "mix_mode_probe")
            .map_err(|e| format!("hipModuleGetFunction(mix_mode_probe): {e}"))?;
        let up = |w: &[u32]| -> Result<DeviceBuffer, String> {
            let d = rt.malloc(w.len() * 4).map_err(|e| format!("hipMalloc(in): {e}"))?;
            rt.memcpy_htod(&d, &words_bytes(w))
                .map_err(|e| format!("upload in: {e}"))?;
            Ok(d)
        };
        let d_in = vec![up(&inp.a)?, up(&inp.b)?, up(&inp.c)?];
        let d_out = rt
            .malloc(2 * GUARD + NV * inp.n * 4)
            .map_err(|e| format!("hipMalloc(out): {e}"))?;
        let d_mrb = rt
            .malloc(2 * (inp.n / 64) * 4)
            .map_err(|e| format!("hipMalloc(mrb): {e}"))?;
        let d_probe = rt.malloc(16).map_err(|e| format!("hipMalloc(probe): {e}"))?;
        Ok(Card {
            rt,
            _module: module,
            kernels,
            probe,
            d_in,
            d_out,
            d_mrb,
            d_probe,
            n: inp.n,
        })
    }

    fn finish(self) -> Vec<String> {
        let mut errs = Vec::new();
        let rt = self.rt;
        for d in self
            .d_in
            .into_iter()
            .chain([self.d_out, self.d_mrb, self.d_probe])
        {
            if let Err(e) = rt.free(d) {
                errs.push(format!("hipFree: {e}"));
            }
        }
        errs
    }

    fn mode_probe(&self) -> Result<[u32; 3], String> {
        let rt = &self.rt;
        rt.memset(&self.d_probe, 0xa5, 16).map_err(|e| format!("memset(probe): {e}"))?;
        let mut blob = hip_blob(&(self.d_probe.as_ptr() as usize as u64).to_le_bytes(), 1, 64);
        // SAFETY: the blob is the kernel's explicit kernarg image (one pointer to a live 16-byte allocation).
        unsafe { rt.launch_kernel_blob(&self.probe, [1, 1, 1], [64, 1, 1], 0, None, &mut blob) }
            .map_err(|e| format!("launch(mode_probe): {e}"))?;
        rt.device_synchronize().map_err(|e| format!("synchronize(mode_probe): {e}"))?;
        let mut b = vec![0u8; 16];
        rt.memcpy_dtoh(&mut b, &self.d_probe).map_err(|e| format!("readback(mode_probe): {e}"))?;
        let w = bytes_words(&b);
        Ok([w[0], w[1], w[2]])
    }

    fn launch(&self, k: usize, bits: u32, poison: u8) -> Result<Launched, String> {
        let rt = &self.rt;
        let total = 2 * GUARD + NV * self.n * 4;
        let mrb_bytes = 2 * (self.n / 64) * 4;
        rt.memset(&self.d_out, i32::from(poison), total).map_err(|e| format!("memset(out): {e}"))?;
        rt.memset(&self.d_mrb, i32::from(poison), mrb_bytes).map_err(|e| format!("memset(mrb): {e}"))?;
        let p = |d: &DeviceBuffer| d.as_ptr() as usize as u64;
        let mut e = Vec::new();
        for x in [p(&self.d_in[0]), p(&self.d_in[1]), p(&self.d_in[2]), p(&self.d_out) + GUARD as u64, p(&self.d_mrb)] {
            e.extend(x.to_le_bytes());
        }
        e.extend((self.n as u32).to_le_bytes());
        e.extend((bits & 0x3ff).to_le_bytes());
        e.extend((bits >> 23 & 1).to_le_bytes());
        let grid = (self.n / BLOCK) as u32;
        let mut blob = hip_blob(&e, grid, BLOCK as u32);
        // SAFETY: the blob is the kernel's explicit kernarg image (5 pointers to live allocations sized for the
        // launch, then three u32), followed by the zeroed hidden arguments.
        unsafe { rt.launch_kernel_blob(&self.kernels[k], [grid, 1, 1], [BLOCK as u32, 1, 1], 0, None, &mut blob) }
            .map_err(|e| format!("launch(kernel {k}): {e}"))?;
        rt.device_synchronize().map_err(|e| format!("synchronize(kernel {k}): {e}"))?;
        let mut ob = vec![0u8; total];
        rt.memcpy_dtoh(&mut ob, &self.d_out).map_err(|e| format!("readback(out, kernel {k}): {e}"))?;
        let mut mb = vec![0u8; mrb_bytes];
        rt.memcpy_dtoh(&mut mb, &self.d_mrb).map_err(|e| format!("readback(mrb, kernel {k}): {e}"))?;
        let guards_ok = ob[..GUARD].iter().all(|&x| x == poison) && ob[total - GUARD..].iter().all(|&x| x == poison);
        Ok(Launched {
            payload: bytes_words(&ob[GUARD..total - GUARD]),
            guards_ok,
            mrb: bytes_words(&mb),
        })
    }

    fn run_mode(&self, bits: u32) -> Result<HwMode, String> {
        let mut out = Vec::with_capacity(NK);
        let (mut guard_bad, mut poison_dep, mut rb_total, mut rb_bad) = (0u64, 0u64, 0u64, 0u64);
        let mut first_rb_bad: Option<(usize, u32)> = None;
        for k in 0..NK {
            let r0 = self.launch(k, bits, POISONS[0])?;
            let r1 = self.launch(k, bits, POISONS[1])?;
            poison_dep += r0.payload.iter().zip(&r1.payload).filter(|(a, b)| a != b).count() as u64;
            guard_bad += u64::from(!r0.guards_ok) + u64::from(!r1.guards_ok);
            for rb in [&r0.mrb, &r1.mrb] {
                for &x in rb.iter() {
                    rb_total += 1;
                    if x != bits {
                        rb_bad += 1;
                        first_rb_bad.get_or_insert((k, x));
                    }
                }
            }
            out.push(r0.payload);
        }
        let mut errors = Vec::new();
        if guard_bad > 0 {
            errors.push(format!("mode {}: {guard_bad} output guard canaries corrupted", h32(bits)));
        }
        if poison_dep > 0 {
            errors.push(format!("mode {}: {poison_dep} output words depend on the poison pattern (unwritten/nondeterministic)", h32(bits)));
        }
        if rb_bad > 0 {
            errors.push(format!(
                "mode {}: {rb_bad}/{rb_total} MODE readbacks differ from the requested bits (first: kernel {:?}); mode not compared",
                h32(bits),
                first_rb_bad
            ));
        }
        let json = json!({
            "mode_readbacks_checked": rb_total, "mode_readbacks_not_equal_requested": rb_bad,
            "first_bad_readback": first_rb_bad.map(|(k, x)| json!({ "kernel": k, "readback": h32(x) })),
            "poison_bytes": POISONS.iter().map(|p| format!("{p:#04x}")).collect::<Vec<_>>(),
            "poison_dependent_words": poison_dep, "guard_canaries_corrupted_launches": guard_bad,
            "launches": 2 * NK,
        });
        Ok(HwMode { out, json, errors })
    }
}

// ----------------------------------------------------------------------------
// Receipt

struct Receipt {
    path: PathBuf,
    t0: Instant,
    top: Map<String, Value>,
    modes: Vec<Value>,
    controls: Vec<Value>,
    errors: Vec<String>,
    failures: Vec<String>,
}
impl Receipt {
    fn error(&mut self, e: String) {
        eprintln!("ERROR: {e}");
        self.errors.push(e);
    }
    fn failure(&mut self, e: String) {
        eprintln!("FAIL: {e}");
        self.failures.push(e);
    }
    fn ok(&self) -> bool {
        self.errors.is_empty() && self.failures.is_empty()
    }
    fn flush(&self, complete: bool, dry: bool) {
        let mut m = self.top.clone();
        m.insert("modes".into(), Value::Array(self.modes.clone()));
        m.insert("controls".into(), Value::Array(self.controls.clone()));
        m.insert("errors".into(), json!(self.errors));
        m.insert("failures".into(), json!(self.failures));
        let total: u64 = self.modes.iter().filter_map(|x| x["compare"]["rows_compared"].as_u64()).sum();
        let mism: u64 = self.modes.iter().filter_map(|x| x["compare"]["mismatches"].as_u64()).sum();
        m.insert("summary".into(), json!({
            "modes_run": self.modes.len(), "rows_compared": total, "strict_mismatches": mism,
            "controls": self.controls.len(),
            "controls_detected": self.controls.iter().filter(|c| c["detected"] == json!(true)).count(),
        }));
        m.insert("wall_seconds".into(), json!(self.t0.elapsed().as_secs_f64()));
        m.insert("complete".into(), json!(complete));
        let ok = complete && self.ok();
        m.insert("exit_ok".into(), json!(ok));
        m.insert("exit_status".into(), json!(if ok { 0 } else { 1 }));
        let verdict = if dry {
            if ok { "DRY_RUN_NO_HARDWARE_CLAIM" } else { "DRY_RUN_FAILED_NO_HARDWARE_CLAIM" }
        } else if !complete {
            "INCOMPLETE"
        } else if ok {
            "PASS_HARDWARE_EQUALS_INTERPRETER_STRICT_BITS"
        } else {
            "FAIL"
        };
        m.insert("verdict".into(), json!(verdict));
        m.insert("finished_unix".into(), json!(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).ok()));
        match serde_json::to_vec_pretty(&Value::Object(m)) {
            Ok(b) => {
                if let Err(e) = std::fs::write(&self.path, b) {
                    eprintln!("cannot write receipt {}: {e}", self.path.display());
                }
            }
            Err(e) => eprintln!("cannot serialise receipt: {e}"),
        }
    }
}

fn tool_version(path: &str) -> Value {
    match Command::new(path).arg("--version").output() {
        Ok(o) => json!({
            "command": format!("{path} --version"), "ok": o.status.success(), "exit_code": o.status.code(),
            "stdout": String::from_utf8_lossy(&o.stdout), "stderr": String::from_utf8_lossy(&o.stderr),
        }),
        Err(e) => json!({ "command": format!("{path} --version"), "ok": false, "error": e.to_string() }),
    }
}

fn clip(b: &[u8]) -> String {
    let s = String::from_utf8_lossy(b);
    if s.len() > 8192 {
        format!("{}...[truncated, {} bytes]", s.chars().take(8192).collect::<String>(), s.len())
    } else {
        s.into_owned()
    }
}

const CONCERNS: &[&str] = &[
    "[CONCERN] interpreter NaN bits come from host f32::mul_add / f32::from_bits arithmetic (platform-defined payload/sign); the comparator is strict, so any hardware NaN-bit difference is reported as a mismatch (class both_nan) and decided only from hardware evidence",
    "[CONCERN] interpreter clamp_float uses f32::max(0.0).min(1.0): the sign of a clamped -0.0 result is host-defined; watch zero_vs_zero_sign samples",
    "[CONCERN] OP_SEL=1 with OP_SEL_HI=0 is modelled as f32 (same as OP_SEL=0); the oracle spans all 64 selections so hardware decides",
    "[CONCERN] the interpreter ignores MODE.IEEE and MODE.FP16_OVFL for v_fma_mix_f32 (f32 result); both are swept on hardware",
    "[CONCERN] f16 sNaN sources are quieted by float::f16_to_f32 (QNAN32 | payload<<13) in the interpreter; hardware behaviour is measured via the sNaN-half words",
    "[SCOPE] NEG_HI/abs, v_fma_mixlo/hi_f16, v_mad_mix*, gfx950 are not covered and not promoted",
    "[SCOPE] hw asm text is assembled by the ROCm assembler from the generated source; the receipt binds the source and code object hashes but the harness does not disassemble the code object to re-derive the encoded fields",
];

// ----------------------------------------------------------------------------
// Controls

fn run_controls(
    refv: &[Vec<u32>],
    base_sim: &[Vec<u32>],
    bits: u32,
    inp: &Inputs,
    hw: bool,
    rec: &mut Receipt,
) -> Result<(), String> {
    let refname = if hw { "hardware" } else { "baseline interpreter (dry-run: no hardware)" };
    let mode = sim_mode(bits);
    // (name, category, mutation)
    let muts: [(&str, &str, Mutation); 7] = [
        ("interp_op_sel_xor_src0", "source_selection", Mutation { xor_op_sel: 1, ..Default::default() }),
        ("interp_op_sel_hi_xor_src0", "source_selection", Mutation { xor_op_sel_hi: 1, ..Default::default() }),
        ("interp_op_sel_xor_src1_src2", "source_selection", Mutation { xor_op_sel: 6, ..Default::default() }),
        ("interp_neg_lo_xor_src0", "negation", Mutation { xor_neg_lo: 1, ..Default::default() }),
        ("interp_clamp_flipped", "clamp", Mutation { flip_clamp: true, ..Default::default() }),
        ("interp_denorm_mode_flipped", "denormal_mode", Mutation { flip_denorm: true, ..Default::default() }),
        ("interp_neg_hi_abs_src0", "abs_modifier_never_used_by_hw_kernels", Mutation { neg_hi: 1, ..Default::default() }),
    ];
    for (name, cat, mu) in muts {
        let s = sim_all(mode, mu, inp)?;
        let c = compare(refv, &s, inp, bits);
        let detected = c.mismatches > 0;
        if !detected {
            rec.failure(format!("negative control {name} NOT detected (0 mismatches vs {refname})"));
        }
        rec.controls.push(json!({
            "control": name, "category": cat, "reference": refname, "mode": h32(bits),
            "expected": "mismatches > 0", "mismatches": c.mismatches, "detected": detected,
            "mismatches_by_class": c.by_class, "first_mismatches": c.first.iter().take(2).collect::<Vec<_>>(),
        }));
    }
    // corruption: flip bit 0 of one reference word (exactly one mismatch must be reported).
    let (k, i) = (NK / 2, NV * inp.n / 2 + 12345 % inp.n);
    let mut cor = refv.to_vec();
    cor[k][i] ^= 1;
    let c = compare(&cor, base_sim, inp, bits);
    let detected = c.mismatches == 1;
    if !detected {
        rec.failure(format!("negative control ref_word_flip_bit0: expected exactly 1 mismatch, got {}", c.mismatches));
    }
    rec.controls.push(json!({
        "control": "ref_word_flip_bit0", "category": "corruption", "reference": refname, "mode": h32(bits),
        "flipped": { "kernel": k, "index": i }, "expected": "exactly 1 mismatch", "mismatches": c.mismatches,
        "detected": detected, "first_mismatches": c.first,
    }));
    Ok(())
}

/// Diagnostic only: on the f32/f32/f32 discriminator rows of the plain variant (op_sel 0, op_sel_hi 0, no neg, no
/// clamp) does the hardware equal the fused interpreter or the unfused host model round32(round32(a*b)+c)?
fn fused_discrimination(hw0: &[u32], sim0: &[u32], inp: &Inputs) -> Value {
    let d0 = inp.blocks.iter().find(|b| b.0 == "fma_discriminators").map(|b| b.1).unwrap_or(0);
    let (mut rows, mut discr, mut hw_eq_sim, mut hw_eq_unf, mut sim_eq_unf, mut hw_eq_unf_on_discr) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for &r in &inp.f32_disc_rows {
        let row = d0 + r;
        let (a, b, c) = (f32::from_bits(inp.a[row]), f32::from_bits(inp.b[row]), f32::from_bits(inp.c[row]));
        let unf = (a * b + c).to_bits();
        let (h, s) = (hw0[row], sim0[row]);
        rows += 1;
        hw_eq_sim += u64::from(h == s);
        hw_eq_unf += u64::from(h == unf);
        sim_eq_unf += u64::from(s == unf);
        if s != unf && !is_nan(s) && !is_nan(unf) {
            discr += 1;
            hw_eq_unf_on_discr += u64::from(h == unf);
        }
    }
    json!({
        "diagnostic_only": true,
        "note": "unfused host model is NOT the interpreter; it only shows whether the hardware fuses",
        "variant": "kernel 0 (op_sel 0, op_sel_hi 0), neg_lo 0, clamp 0, f32/f32/f32 rows of the discriminator block",
        "rows": rows, "rows_where_unfused_differs_from_interpreter": discr,
        "hw_equals_interpreter": hw_eq_sim, "hw_equals_unfused": hw_eq_unf, "interpreter_equals_unfused": sim_eq_unf,
        "hw_equals_unfused_on_discriminating_rows": hw_eq_unf_on_discr,
    })
}

// ----------------------------------------------------------------------------

fn run(o: &Opts, rec: &mut Receipt) {
    let argv: Vec<String> = std::env::args().collect();
    rec.top.insert("schema".into(), json!("cdna_mix_hw_oracle/1"));
    rec.top.insert("mode".into(), json!(if o.dry_run { "dry-run" } else { "hardware" }));
    rec.top.insert("hardware_evidence".into(), json!(!o.dry_run));
    if o.dry_run {
        rec.top.insert("dry_run_notice".into(), json!("DRY RUN: no HIP call, no hipcc, no hardware. This receipt makes NO claim about the MI300X; the controls compare mutated interpreters with the baseline interpreter."));
        println!("DRY RUN: no hardware, no hipcc; no hardware claim");
    }
    rec.top.insert("args".into(), json!(argv));
    rec.top.insert("scope".into(), json!({
        "op": "v_fma_mix_f32 (Vop3pOp::FmaMixF32)", "neg_hi": 0, "neg_lo_masks": 8, "clamp": [false, true],
        "op_sel": "0..7", "op_sel_hi": "0..7", "kernels": NK, "variants_per_kernel": NV, "variants_total": NK * NV,
        "not_covered": ["v_fma_mixlo_f16", "v_fma_mixhi_f16", "v_mad_mix*", "|abs| / NEG_HI", "gfx950"],
        "round_mode": "RNE only (FP_ROUND = 0)",
    }));
    rec.top.insert("concerns".into(), json!(CONCERNS));
    rec.top.insert("host".into(), json!({
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "hostname": std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|s| s.trim().to_string()),
    }));

    // Source and inputs.
    let src = gen_source();
    let inp = gen_inputs(o.patterns);
    let modes = mode_list();
    if let Err(e) = std::fs::create_dir_all(&o.hip_work) {
        rec.error(format!("create {}: {e}", o.hip_work.display()));
        return;
    }
    let src_path = o.hip_work.join("cdna_mix_hw_oracle.hip");
    let hsaco_path = o.hip_work.join("cdna_mix_hw_oracle.hsaco");
    if let Err(e) = std::fs::write(&src_path, &src) {
        rec.error(format!("write {}: {e}", src_path.display()));
        return;
    }
    let (ab, bb, cb) = (words_bytes(&inp.a), words_bytes(&inp.b), words_bytes(&inp.c));
    let mut all = ab.clone();
    all.extend(&bb);
    all.extend(&cb);
    rec.top.insert("inputs".into(), json!({
        "seed_u64": format!("{SEED:#018x}"), "generator": "splitmix64",
        "patterns": inp.n, "variants_total": NK * NV, "rows_per_mode": NK * NV * inp.n,
        "modes": modes.len(), "rows_total": modes.len() * NK * NV * inp.n,
        "in0_sha256": sha(&ab), "in1_sha256": sha(&bb), "in2_sha256": sha(&cb), "in0_in1_in2_sha256": sha(&all),
        "blocks": inp.blocks.iter().map(|b| json!({ "name": b.0, "start": b.1, "rows": b.2 })).collect::<Vec<_>>(),
        "fma_discriminator_rows_discriminating": inp.disc_rows_discriminating,
        "fma_discriminator_rows_exact_product_f16_x_f16": inp.disc_rows_exact_product,
        "fma_discriminator_rows_unresolved": inp.disc_rows_unresolved,
        "special_words": special_words().iter().map(|x| h32(*x)).collect::<Vec<_>>(),
    }));
    rec.top.insert("hip_source".into(), json!({
        "path": src_path.display().to_string(), "bytes": src.len(), "sha256": sha(src.as_bytes()),
        "example_asm": asm_line("v0", ["v1", "v2", "v3"], 5, 3, 5, true),
    }));
    rec.top.insert("mode_list".into(), json!(modes.iter().map(|&b| mode_json(b)).collect::<Vec<_>>()));
    if inp.disc_rows_unresolved > 0 {
        rec.error(format!("{} fused-discriminator rows could not be made discriminating", inp.disc_rows_unresolved));
    }

    // Hardware setup.
    let mut card: Option<Card> = None;
    if !o.dry_run {
        let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
        rec.top.insert("tools".into(), json!({ "hipcc": tool_version(&hipcc) }));
        let rt = match HipRuntime::load() {
            Ok(r) => r,
            Err(e) => {
                rec.error(format!("HipRuntime::load: {e}"));
                return;
            }
        };
        let mut dev = Map::new();
        let count = match rt.device_count() {
            Ok(c) => c,
            Err(e) => {
                rec.error(format!("hipGetDeviceCount: {e}"));
                return;
            }
        };
        dev.insert("device_count".into(), json!(count));
        if count < 1 {
            rec.top.insert("device".into(), Value::Object(dev));
            rec.error("no HIP device".into());
            return;
        }
        if let Err(e) = rt.set_device(0) {
            rec.error(format!("hipSetDevice(0): {e}"));
            return;
        }
        let arch = match rt.get_arch(0) {
            Ok(a) => a,
            Err(e) => {
                rec.error(format!("hipGetDeviceProperties(0): {e}"));
                return;
            }
        };
        dev.insert("device".into(), json!(0));
        dev.insert("device_arch".into(), json!(arch));
        dev.insert("expected_arch".into(), json!(EXPECTED_ARCH));
        match rt.runtime_version() {
            Ok((a, b)) => {
                dev.insert("hip_runtime_version".into(), json!(format!("{a}.{b}")));
            }
            Err(e) => {
                dev.insert("hip_runtime_version_error".into(), json!(e.to_string()));
            }
        }
        rec.top.insert("device".into(), Value::Object(dev));
        println!("device 0 arch {arch}");
        if arch != EXPECTED_ARCH {
            rec.error(format!("device 0 reports `{arch}`, this oracle runs only on {EXPECTED_ARCH}: refusing"));
            return;
        }
        let args = [
            "--genco", "--offload-arch=gfx942", "-O3", "--no-offload-compress", "-fuse-cuid=none", "-o",
            hsaco_path.to_str().unwrap_or("cdna_mix_hw_oracle.hsaco"),
            src_path.to_str().unwrap_or("cdna_mix_hw_oracle.hip"),
        ];
        let t = Instant::now();
        let out = Command::new(&hipcc).args(args).output();
        let compile = match &out {
            Ok(r) => json!({
                "command": format!("{hipcc} {}", args.join(" ")), "exit_code": r.status.code(), "success": r.status.success(),
                "stdout": clip(&r.stdout), "stderr": clip(&r.stderr), "seconds": t.elapsed().as_secs_f64(),
            }),
            Err(e) => json!({ "command": format!("{hipcc} {}", args.join(" ")), "error": e.to_string() }),
        };
        rec.top.insert("compile".into(), compile);
        match out {
            Ok(r) if r.status.success() => {}
            Ok(r) => {
                rec.error(format!("hipcc failed (exit {:?}): {}", r.status.code(), clip(&r.stderr)));
                return;
            }
            Err(e) => {
                rec.error(format!("cannot run {hipcc}: {e}"));
                return;
            }
        }
        let hsaco = match std::fs::read(&hsaco_path) {
            Ok(b) => b,
            Err(e) => {
                rec.error(format!("read {}: {e}", hsaco_path.display()));
                return;
            }
        };
        rec.top.insert("code_object".into(), json!({ "path": hsaco_path.display().to_string(), "bytes": hsaco.len(), "sha256": sha(&hsaco) }));
        match Card::new(rt, &hsaco, &inp) {
            Ok(c) => card = Some(c),
            Err(e) => {
                rec.error(e);
                return;
            }
        }
        if let Some(c) = &card {
            match c.mode_probe() {
                Ok(p) => {
                    let launch_default = p[0] | (p[2] << 23);
                    rec.top.insert("launch_default_mode_probe".into(), json!({
                        "bits_9_0": h32(p[0]), "bits_22_10": h32(p[1]), "bit_23": p[2],
                        "bits_modelled": h32(launch_default), "interpreter_default_bits": h32(0x3f0),
                        "equals_interpreter_default": launch_default == 0x3f0,
                        "note": "informational; every compared launch sets and reads back its MODE explicitly",
                    }));
                }
                Err(e) => rec.error(e),
            }
        }
    } else {
        rec.top.insert("device".into(), json!({ "device_arch": null, "note": "dry-run: no device" }));
    }
    rec.flush(false, o.dry_run);

    // Modes.
    for (mi, &bits) in modes.iter().enumerate() {
        let t = Instant::now();
        let hw = match &card {
            Some(c) => match c.run_mode(bits) {
                Ok(h) => Some(h),
                Err(e) => {
                    rec.error(format!("mode {}: {e}", h32(bits)));
                    rec.modes.push(json!({ "mode": mode_json(bits), "error": e }));
                    rec.flush(false, o.dry_run);
                    continue;
                }
            },
            None => None,
        };
        let sim = match sim_all(sim_mode(bits), Mutation::default(), &inp) {
            Ok(s) => s,
            Err(e) => {
                rec.error(format!("mode {}: {e}", h32(bits)));
                rec.modes.push(json!({ "mode": mode_json(bits), "error": e }));
                rec.flush(false, o.dry_run);
                continue;
            }
        };
        let mut m = Map::new();
        m.insert("mode".into(), mode_json(bits));
        m.insert("sim_output_sha256".into(), json!(hash_outputs(&sim)));
        m.insert("variants".into(), json!(NK * NV));
        m.insert("inputs_per_variant".into(), json!(inp.n));
        let mut usable = true;
        if let Some(h) = &hw {
            m.insert("hw_output_sha256".into(), json!(hash_outputs(&h.out)));
            m.insert("hw".into(), h.json.clone());
            for e in &h.errors {
                rec.error(e.clone());
                usable = false;
            }
        } else {
            m.insert("hw_output_sha256".into(), Value::Null);
            m.insert("hw".into(), json!({ "note": "dry-run: no hardware" }));
        }
        if let (Some(h), true) = (&hw, usable) {
            let c = compare(&h.out, &sim, &inp, bits);
            if c.mismatches > 0 {
                rec.failure(format!("mode {}: {} strict raw-bit mismatches of {} rows", h32(bits), c.mismatches, c.rows));
            }
            m.insert("compare".into(), c.json());
            if mi == 0 {
                rec.top.insert("fused_discrimination".into(), fused_discrimination(&h.out[0], &sim[0], &inp));
                if let Err(e) = run_controls(&h.out, &sim, bits, &inp, true, rec) {
                    rec.error(format!("controls: {e}"));
                }
            }
        } else if hw.is_none() {
            // dry-run: interpreter against itself (second independent run) proves determinism only.
            match sim_all(sim_mode(bits), Mutation::default(), &inp) {
                Ok(s2) => {
                    let c = compare(&s2, &sim, &inp, bits);
                    if c.mismatches > 0 {
                        rec.failure(format!("mode {}: interpreter is not deterministic ({} rows)", h32(bits), c.mismatches));
                    }
                    m.insert("compare".into(), c.json());
                    m.insert("compare_kind".into(), json!("interpreter run vs second interpreter run (determinism); NOT hardware"));
                    if mi == 0 {
                        if let Err(e) = run_controls(&s2, &sim, bits, &inp, false, rec) {
                            rec.error(format!("controls: {e}"));
                        }
                    }
                }
                Err(e) => rec.error(format!("mode {}: {e}", h32(bits))),
            }
        }
        m.insert("seconds".into(), json!(t.elapsed().as_secs_f64()));
        println!(
            "mode {} ({}/{}): mismatches {}  ({:.1}s)",
            h32(bits),
            mi + 1,
            modes.len(),
            m.get("compare").map(|c| c["mismatches"].to_string()).unwrap_or_else(|| "n/a".into()),
            t.elapsed().as_secs_f64()
        );
        rec.modes.push(Value::Object(m));
        rec.flush(false, o.dry_run);
    }
    if let Some(c) = card {
        for e in c.finish() {
            rec.error(e);
        }
    }
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(Some(o)) => o,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut rec = Receipt {
        path: opts.receipt.clone(),
        t0: Instant::now(),
        top: Map::new(),
        modes: Vec::new(),
        controls: Vec::new(),
        errors: Vec::new(),
        failures: Vec::new(),
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&opts, &mut rec)));
    if let Err(p) = r {
        rec.error(format!("harness panic: {}", panic_msg(p)));
    }
    let complete = rec.modes.len() == mode_list().len();
    rec.flush(complete, opts.dry_run);
    let ok = complete && rec.ok();
    println!(
        "SUMMARY: {} modes, {} errors, {} failures, {:.1}s -> {}{}",
        rec.modes.len(),
        rec.errors.len(),
        rec.failures.len(),
        rec.t0.elapsed().as_secs_f64(),
        if ok { "OK" } else { "FAILED" },
        if opts.dry_run { " (DRY RUN: no hardware claim)" } else { "" }
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
