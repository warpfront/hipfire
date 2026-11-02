// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `fa_prefill` workload: the builder-emitted gfx1201 GQA-fused FA2 prefill attention for the A3B
//! H16 / KV2 / D256 geometry (`peacemaker-isa emit --kernel fa_prefill_pm|fa_prefill_fp8_pm`):
//! the Q8_0 module (Q preconvert + causal direct) and the native-fp8 E4M3 module (direct), against
//! the shipped bundles `kernels/fa_prefill_pm_gfx1201.hxaco` and `kernels/fa_prefill_fp8_pm_gfx1201.hxaco`.
//!
//! # Shape grammar
//! `a3b:<name>` or `N=<rows>[,S=<start>]` (`a3b:` presets take `N`/`S` overrides). `N` query rows
//! at positions `S + r` (the packed positions of one causal prefill chunk), cache context
//! `S + N`. Production admission (`Gpu::fa_prefill_pm_admission`): `64 <= N <= 8192`,
//! `1 <= S + N <= 262144`; the head geometry is fixed at (16, 2, 256). The presets are the five
//! shapes the A3B oracle validates and times (`fa_pm_oracle.rs` `A0_SHAPES`/`TIME_SHAPES`), the chunk
//! sizes and cache offsets of the A3B prefill call sites (the Q8_0 route,
//! `hipfire-dispatch/src/families/attention.rs:2353`, and the native-fp8 route, `:2772`, are the
//! same A3B H16/KV2/D256 call site):
//!
//! ```text
//! a3b:n512-s0   a3b:n4096-s0   a3b:n8192-s0   a3b:n512-s3584   a3b:n4096-s4096
//! ```
//!
//! # Knobs (candidate)
//! `--codec q8_0` (default) | `fp8`; `--export direct` (default, the timed attention kernel) |
//! `preconvert` (the Q8 module's f32 -> f16 Q kernel; `--codec q8_0` only: the fp8 module has no
//! preconvert, the Q16 is the Q8 module's). `--ref-rows full|sample:<n>` (default `sample:32`).
//! The candidate object is the product module (`emit --co`: both Q8_0 exports certified, or the
//! one fp8 export); the launch is exactly the runtime's (`fa_prefill_pm_enqueue_direct` /
//! `_enqueue_pre`): direct grid `[ceil(N/8), 2, 1]`, 160 threads, 65536 B (Q8_0) / 32768 B (fp8)
//! dynamic LDS, 60-byte kernarg `q16 k v out positions` pointers then `16 2 256 N` and
//! `scale_attn = 1/16`; preconvert grid `[ceil(16 N / 4), 1, 1]`, 128 threads, 40-byte kernarg.
//!
//! # Baseline
//! The shipped bundle and the symbol the runtime selects for the codec (`attention_q8_0_fa2_gqa_pm_gfx1201`
//! / `attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201`, or `attention_fa2_q_preconvert_pm_gfx1201`);
//! selection file:line and the bundle sha256 go in `baseline.source`.
//!
//! # Correctness
//! The candidate must be bitwise equal to the shipped baseline. Reference status:
//! * `--export preconvert`: EXACT. `f16(q)` is round-to-nearest-even; every element is compared
//!   bit for bit on all rows.
//! * `--export direct`: APPROXIMATE only. There is no exact CPU reference for the attention at
//!   production size (`v_exp_f32` softmax, f16 probabilities, f32 accumulation order); the record's
//!   reference is an f64 causal attention over the f16-dequantized K/V and the f16-rounded Q, sampled
//!   rows, with an absolute tolerance of `5e-3` (`reference` says `approx`), which checks layout and
//!   semantics, not rounding. `pass` additionally requires bitwise identity with the shipped kernel.
//!   (`tests/fa_prefill.rs` of `peacemaker-isa` holds the exact-by-construction references on
//!   exactly representable inputs.)
//!
//! # Inputs (deterministic, synthetic)
//! Q N(0,1) f32; K N(0,1), V N(0, 0.7) quantized to the codec: Q8_0 blocks `[f16 scale = amax/127][32 i8]`
//! per `(token, head, 32 elements)`, 544-byte rows; fp8 rows `[512 head-major E4M3 codes = x / (amax/224)]
//! [2 f16 scales]`, 516-byte rows. The Q16 the direct kernel reads is the SHIPPED preconvert's output
//! (run once in `setup`, outside every timed interval, shared by both arms).
//!
//! # `--emu`
//! Runs the candidate export on the emulator at N=16, S=0 and compares with the same reference
//! (`approx` for direct, exact for preconvert); an unqualified instruction is the emulator's own refusal.

use crate::common::{
    dev_err, e4m3, e4m3_val, emitted_object, emu_run, f16_bits, f16_val, max_diff, no_leftover,
    put_f32, put_i32, put_ptr, refused, sha256_hex, shipped_object, take_dim, take_dim_or, Elem,
    EmuBuf, ShapeSpec, Xs64,
};
use crate::iu4::{abi_signature, sample_rows};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use peacemaker_isa::kernels::fa_prefill::{self as fa, FaKvCodec, FaPrefillSpec};
use peacemaker_isa::Arch;
use rdna_compute::fa_prefill_pm as pm;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;
use std::path::Path;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
/// Elements of one query row (all heads).
const ROW: usize = H * D;
const MIN_ROWS: usize = 64;
const MAX_ROWS: usize = 8192;
const MAX_CONTEXT: usize = 262_144;
/// Absolute tolerance of the approximate attention reference.
const TOL: f64 = 5e-3;
const SCALE_ATTN: f32 = 1.0 / 16.0;
const Q8_OBJECT: &str = "kernels/fa_prefill_pm_gfx1201.hxaco";
const FP8_OBJECT: &str = "kernels/fa_prefill_fp8_pm_gfx1201.hxaco";

/// `(name, N, S)`: the A3B oracle's validated and timed shapes.
const A3B_SHAPES: [(&str, usize, usize); 5] = [
    ("n512-s0", 512, 0),
    ("n4096-s0", 4096, 0),
    ("n8192-s0", 8192, 0),
    ("n512-s3584", 512, 3584),
    ("n4096-s4096", 4096, 4096),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Export {
    Direct,
    Preconvert,
}

impl Export {
    fn name(self) -> &'static str {
        match self {
            Export::Direct => "direct",
            Export::Preconvert => "preconvert",
        }
    }
}

fn codec_name(c: FaKvCodec) -> &'static str {
    match c {
        FaKvCodec::Q8_0 => "q8_0",
        FaKvCodec::Fp8E4M3 => "fp8",
    }
}

/// Bytes of one token row of one K or V side.
fn row_bytes(c: FaKvCodec) -> usize {
    match c {
        FaKvCodec::Q8_0 => 544,
        FaKvCodec::Fp8E4M3 => 516,
    }
}

fn spec_of(c: FaKvCodec) -> FaPrefillSpec {
    FaPrefillSpec {
        heads: H as u32,
        kv_heads: KV as u32,
        head_dim: D as u32,
        kv: c,
    }
}

fn symbol(c: FaKvCodec, e: Export) -> &'static str {
    match (c, e) {
        (_, Export::Preconvert) => fa::SYMBOL_PRECONVERT,
        (FaKvCodec::Q8_0, Export::Direct) => fa::SYMBOL_DIRECT,
        (FaKvCodec::Fp8E4M3, Export::Direct) => fa::SYMBOL_FP8_DIRECT,
    }
}

fn direct_lds(c: FaKvCodec) -> u32 {
    match c {
        FaKvCodec::Q8_0 => fa::DYNAMIC_LDS_BYTES,
        FaKvCodec::Fp8E4M3 => fa::FP8_DYNAMIC_LDS_BYTES,
    }
}

// ---------------------------------------------------------------------------
// Inputs

fn synth_q(rows: usize) -> Vec<u8> {
    let mut r = Xs64::new(0x51_0e_a3b0_0000_0001);
    (0..rows * ROW)
        .flat_map(|_| r.normal().to_le_bytes())
        .collect()
}

/// `ctx` packed token rows of one K or V side in `codec`'s wire format.
fn synth_kv(c: FaKvCodec, ctx: usize, seed: u64, amp: f32) -> Vec<u8> {
    let mut r = Xs64::new(seed);
    let rb = row_bytes(c);
    let mut out = vec![0u8; ctx * rb];
    let mut x = vec![0f32; D];
    for pos in 0..ctx {
        for g in 0..KV {
            for v in &mut x {
                *v = r.normal() * amp;
            }
            let row = &mut out[pos * rb..(pos + 1) * rb];
            match c {
                FaKvCodec::Q8_0 => {
                    for b in 0..D / 32 {
                        let blk = &x[b * 32..(b + 1) * 32];
                        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
                        let scale = f16_bits(if amax > 0.0 { amax / 127.0 } else { 1.0 });
                        let sf = f16_val(scale);
                        let at = (g * (D / 32) + b) * 34;
                        row[at..at + 2].copy_from_slice(&scale.to_le_bytes());
                        for (i, v) in blk.iter().enumerate() {
                            row[at + 2 + i] = (v / sf).round().clamp(-127.0, 127.0) as i8 as u8;
                        }
                    }
                }
                FaKvCodec::Fp8E4M3 => {
                    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let scale = f16_bits(if amax > 0.0 { amax / 224.0 } else { 1.0 });
                    let sf = f16_val(scale);
                    for (d, v) in x.iter().enumerate() {
                        row[g * D + d] = e4m3(v / sf);
                    }
                    row[512 + g * 2..512 + g * 2 + 2].copy_from_slice(&scale.to_le_bytes());
                }
            }
        }
    }
    out
}

fn positions(rows: usize, start: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|r| ((start + r) as i32).to_le_bytes())
        .collect()
}

/// The f16 values (as f32) the kernels' fill stages from the cache: `f16(scale_f32 * code)`, one rounding.
fn dequant(c: FaKvCodec, cache: &[u8], ctx: usize) -> Vec<f32> {
    let rb = row_bytes(c);
    let mut out = vec![0f32; ctx * KV * D];
    for pos in 0..ctx {
        let row = &cache[pos * rb..(pos + 1) * rb];
        for g in 0..KV {
            for d in 0..D {
                let (scale, code) = match c {
                    FaKvCodec::Q8_0 => {
                        let at = (g * (D / 32) + d / 32) * 34;
                        (
                            f16_val(u16::from_le_bytes([row[at], row[at + 1]])),
                            f32::from(row[at + 2 + d % 32] as i8),
                        )
                    }
                    FaKvCodec::Fp8E4M3 => (
                        f16_val(u16::from_le_bytes([row[512 + g * 2], row[512 + g * 2 + 1]])),
                        e4m3_val(row[g * D + d]),
                    ),
                };
                out[(pos * KV + g) * D + d] = f16_val(f16_bits(scale * code));
            }
        }
    }
    out
}

/// f64 causal attention of the sampled rows `ridx`: f16-rounded Q, f16-dequantized K/V,
/// `scale_attn = 1/16`, keys `0..=start + r`. Returns `[ridx.len()][16][256]` f32.
fn attention_ref(q: &[u8], kd: &[f32], vd: &[f32], start: usize, ridx: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; ridx.len() * ROW];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let tasks = ridx.len() * H;
    let per = tasks.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * D).enumerate() {
            s.spawn(move || {
                let mut qv = [0f64; D];
                let mut sc: Vec<f64> = Vec::new();
                for (ti, o) in chunk.chunks_exact_mut(D).enumerate() {
                    let t = ci * per + ti;
                    let (r, h) = (ridx[t / H], t % H);
                    let kvh = h / (H / KV);
                    let base = (r * H + h) * D * 4;
                    for (d, e) in qv.iter_mut().enumerate() {
                        let b = &q[base + d * 4..base + d * 4 + 4];
                        *e =
                            f64::from(f16_val(f16_bits(f32::from_le_bytes(b.try_into().unwrap()))));
                    }
                    let keys = start + r + 1;
                    sc.clear();
                    for j in 0..keys {
                        let k = &kd[(j * KV + kvh) * D..][..D];
                        let dot: f64 = qv.iter().zip(k).map(|(a, b)| a * f64::from(*b)).sum();
                        sc.push(dot * f64::from(SCALE_ATTN));
                    }
                    let m = sc.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let mut l = 0f64;
                    let mut acc = [0f64; D];
                    for (j, s) in sc.iter().enumerate() {
                        let p = (s - m).exp();
                        l += p;
                        let v = &vd[(j * KV + kvh) * D..][..D];
                        for (a, x) in acc.iter_mut().zip(v) {
                            *a += p * f64::from(*x);
                        }
                    }
                    for (e, a) in o.iter_mut().zip(&acc) {
                        *e = (a / l) as f32;
                    }
                }
            });
        }
    });
    out
}

/// Exact reference of the preconvert: `f16(q)` RNE, all rows.
fn preconvert_ref(q: &[u8]) -> Vec<u8> {
    q.chunks_exact(4)
        .flat_map(|c| f16_bits(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// Kernargs and launch geometry

/// The 60-byte direct kernarg: `q16 k v out positions` then `16 2 256 N` and `scale_attn`.
fn pack_direct(ptrs: [u64; 5], rows: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(60);
    for p in ptrs {
        put_ptr(&mut b, p);
    }
    for v in [H, KV, D, rows] {
        put_i32(&mut b, v as i32);
    }
    put_f32(&mut b, SCALE_ATTN);
    b
}

/// The 40-byte preconvert kernarg: `q q16 signs1 signs2` pointers, `batch_size`, `do_rotate`.
fn pack_pre(q: u64, q16: u64, rows: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(40);
    for p in [q, q16, 0, 0] {
        put_ptr(&mut b, p);
    }
    put_i32(&mut b, rows as i32);
    put_i32(&mut b, 0);
    b
}

fn direct_grid(rows: usize) -> [u32; 3] {
    [rows.div_ceil(8) as u32, KV as u32, 1]
}

fn pre_grid(rows: usize) -> [u32; 3] {
    [(rows * H).div_ceil(4) as u32, 1, 1]
}

// ---------------------------------------------------------------------------
// The workload

struct Host {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    pos: Vec<u8>,
}

struct Dev {
    q: hip_bridge::DeviceBuffer,
    q16: hip_bridge::DeviceBuffer,
    k: hip_bridge::DeviceBuffer,
    v: hip_bridge::DeviceBuffer,
    pos: hip_bridge::DeviceBuffer,
    /// Per arm: the output (`out` f32 for direct, `q16` f16 for preconvert).
    out: [hip_bridge::DeviceBuffer; 2],
}

pub struct Fa {
    rows: usize,
    start: usize,
    codec: FaKvCodec,
    export: Export,
    rows_sel: Vec<usize>,
    rows_desc: String,
    rip: Option<(String, String)>,
    host: Option<Host>,
    dev: Option<Dev>,
    reference: OnceCell<Vec<f32>>,
}

fn parse_shape(shape: &str) -> Result<(usize, usize)> {
    let spec = ShapeSpec::parse(shape)?;
    let defaults: Vec<(&str, usize)> = match &spec.preset {
        None => vec![("S", 0)],
        Some((fam, name)) if fam == "a3b" => {
            let (_, n, s) = A3B_SHAPES
                .iter()
                .find(|(p, _, _)| p == name)
                .ok_or_else(|| {
                    format!(
                        "unknown preset a3b:{name} (one of {})",
                        A3B_SHAPES
                            .iter()
                            .map(|(p, _, _)| format!("a3b:{p}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            vec![("N", *n), ("S", *s)]
        }
        Some((fam, _)) => {
            return Err(format!(
                "shape preset family `{fam}`: only a3b presets exist"
            ))
        }
    };
    let mut m = spec.resolve(&defaults);
    let rows = take_dim(&mut m, "N")?;
    let start = take_dim_or(&mut m, "S", 0)?;
    no_leftover(&m)?;
    if !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
        return Err(format!(
            "shape N={rows}: production admits {MIN_ROWS}..={MAX_ROWS} rows"
        ));
    }
    if rows + start > MAX_CONTEXT {
        return Err(format!(
            "shape S={start},N={rows}: context {} exceeds {MAX_CONTEXT}",
            rows + start
        ));
    }
    Ok((rows, start))
}

fn parse_ref_rows(mode: &str, rows: usize) -> Result<(Vec<usize>, String)> {
    if mode == "full" {
        Ok(((0..rows).collect(), "full".to_owned()))
    } else if let Some(c) = mode.strip_prefix("sample:") {
        let c: usize = c.parse().map_err(|e| format!("--ref-rows {mode}: {e}"))?;
        Ok((sample_rows(rows, c.max(2)), format!("sample:{c}")))
    } else {
        Err(format!("--ref-rows {mode}: expected full|sample:<n>"))
    }
}

pub fn new(shape: &str, knobs: &mut Knobs, rip: Option<&Path>) -> Result<Box<dyn Workload>> {
    let (rows, start) = parse_shape(shape)?;
    let codec = match knobs.take_or("codec", "q8_0").as_str() {
        "q8_0" => FaKvCodec::Q8_0,
        "fp8" => FaKvCodec::Fp8E4M3,
        o => return Err(format!("--codec {o}: expected q8_0|fp8")),
    };
    let export = match knobs.take_or("export", "direct").as_str() {
        "direct" => Export::Direct,
        "preconvert" => Export::Preconvert,
        o => return Err(format!("--export {o}: expected direct|preconvert")),
    };
    if export == Export::Preconvert && codec == FaKvCodec::Fp8E4M3 {
        return Err(
            "--export preconvert needs --codec q8_0: the fp8 module has no preconvert (its Q16 is \
             the Q8 module's `attention_fa2_q_preconvert_pm_gfx1201`)"
                .into(),
        );
    }
    spec_of(codec).validate()?;
    let default_ref = "sample:32";
    let (rows_sel, rows_desc) = parse_ref_rows(&knobs.take_or("ref-rows", default_ref), rows)?;
    let rip = match rip {
        Some(p) => Some((
            p.display().to_string(),
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    Ok(Box::new(Fa {
        rows,
        start,
        codec,
        export,
        rows_sel,
        rows_desc,
        rip,
        host: None,
        dev: None,
        reference: OnceCell::new(),
    }))
}

/// Compile `.rip` source for `(codec, export)`, refuse unless the export's ABI equals the generator's;
/// returns the ELF, symbol and the `candidate.rip` fields.
fn compile_rip(source: &str, codec: FaKvCodec, export: Export) -> Result<(Vec<u8>, String, Value)> {
    let arch = Arch::Gfx1201;
    let spec = spec_of(codec);
    let variant = export.name();
    let compiled = peacemaker_rip::compile(source, arch, variant)?;
    let generator = match export {
        Export::Direct => fa::emit(arch, &spec)?,
        Export::Preconvert => fa::emit_preconvert(arch, &spec)?,
    };
    let sym = symbol(codec, export);
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, arch)?;
    let got = abi_signature(&elf, sym)
        .map_err(|e| format!("compiled kernel does not expose {sym}: {e}"))?;
    let want = abi_signature(&gen_elf, sym)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with fa_prefill::emit({} {variant}):\n  .rip:      {got}\n  generator: {want}",
            codec_name(codec)
        ));
    }
    let info = json!({
        "sha256": sha256_hex(source.as_bytes()),
        "variant": format!("{}.{variant}", codec_name(codec)),
        "identical_to_generator": compiled.s_text == generator.s_text,
    });
    Ok((elf, sym.to_owned(), info))
}

impl Fa {
    fn ctx(&self) -> usize {
        self.start + self.rows
    }

    fn knobs_json(&self) -> Value {
        json!({"codec": codec_name(self.codec), "export": self.export.name()})
    }

    fn out_elem(&self) -> Elem {
        match self.export {
            Export::Direct => Elem::F32,
            Export::Preconvert => Elem::F16,
        }
    }

    fn out_bytes(&self) -> usize {
        match self.export {
            Export::Direct => self.rows * ROW * 4,
            Export::Preconvert => self.rows * ROW * 2,
        }
    }

    fn emitted(&self) -> Result<Object> {
        let arch = Arch::Gfx1201;
        let spec = spec_of(self.codec);
        let (text, exports, what) = match self.codec {
            FaKvCodec::Q8_0 => {
                let (_, text, _) = fa::emit_module(arch, &spec)?;
                (
                    text,
                    vec![fa::SYMBOL_PRECONVERT.to_owned(), fa::SYMBOL_DIRECT.to_owned()],
                    "peacemaker_isa::kernels::fa_prefill::emit_module + native::assemble (peacemaker-isa emit --kernel fa_prefill_pm --co)",
                )
            }
            FaKvCodec::Fp8E4M3 => {
                let (_, text, _) = fa::emit_fp8_module(arch, &spec)?;
                (
                    text,
                    vec![fa::SYMBOL_FP8_DIRECT.to_owned()],
                    "peacemaker_isa::kernels::fa_prefill::emit_fp8_module + native::assemble (peacemaker-isa emit --kernel fa_prefill_fp8_pm --co)",
                )
            }
        };
        emitted_object(
            &text,
            arch,
            symbol(self.codec, self.export),
            &exports,
            what,
            self.knobs_json(),
        )
    }

    fn rip_object(&self, path: &str, text: &str) -> Result<Object> {
        let (elf, sym, mut info) = compile_rip(text, self.codec, self.export)?;
        info["path"] = json!(path);
        Ok(Object {
            source: json!({
                "source": format!("peacemaker_rip::compile({path}) + native::assemble"),
                "variant": info["variant"].clone(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.knobs_json(), "rip": info}),
            exports: vec![sym.clone()],
            launch_symbol: sym,
            bytes: elf,
        })
    }

    /// The reference's per-element comparison over `got`: `(mismatching or out-of-tolerance elements,
    /// worst abs error, covered elements)`.
    fn compare(&self, got: &[u8]) -> Result<(usize, f64, usize)> {
        let host = self.host.as_ref().ok_or("check before setup")?;
        match self.export {
            Export::Preconvert => {
                let want = preconvert_ref(&host.q);
                let (mut bad, mut worst) = (0usize, 0f64);
                for (g, w) in got.chunks_exact(2).zip(want.chunks_exact(2)) {
                    if g != w {
                        bad += 1;
                        worst = worst.max(crate::common::elem_diff(Elem::F16, g, w));
                    }
                }
                Ok((bad, worst, got.len() / 2))
            }
            Export::Direct => {
                let want = self.reference.get_or_init(|| {
                    let kd = dequant(self.codec, &host.k, self.ctx());
                    let vd = dequant(self.codec, &host.v, self.ctx());
                    attention_ref(&host.q, &kd, &vd, self.start, &self.rows_sel)
                });
                let (mut bad, mut worst) = (0usize, 0f64);
                for (ri, &r) in self.rows_sel.iter().enumerate() {
                    for i in 0..ROW {
                        let o = (r * ROW + i) * 4;
                        let g = f32::from_le_bytes(got[o..o + 4].try_into().unwrap());
                        let w = want[ri * ROW + i];
                        let e = f64::from((g - w).abs());
                        if g.is_nan() || e > TOL {
                            bad += 1;
                        }
                        worst = if g.is_nan() {
                            f64::INFINITY
                        } else {
                            worst.max(e)
                        };
                    }
                }
                Ok((bad, worst, self.rows_sel.len() * ROW))
            }
        }
    }

    /// Run the candidate export on the emulator at N=16, S=0.
    fn run_emu(&self) -> Result<Value> {
        let obj = self.candidate()?;
        let (rows, ctx) = (16usize, 16usize);
        let q = synth_q(rows);
        let (k, v) = (
            synth_kv(self.codec, ctx, 0xa11c_e5ee_d001, 1.0),
            synth_kv(self.codec, ctx, 0xa11c_e5ee_d002, 0.7),
        );
        let pos = positions(rows, 0);
        let (qva, q16va, kva, vva, pva, ova) = (
            0x2000_0000u64,
            0x3000_0000u64,
            0x4000_0000u64,
            0x5000_0000u64,
            0x6000_0000u64,
            0x7000_0000u64,
        );
        let (shape, mem, out_va, out_len, elem) = match self.export {
            Export::Preconvert => {
                let mem = emu_run(
                    &obj.bytes,
                    &obj.launch_symbol,
                    pack_pre(qva, q16va, rows),
                    vec![
                        EmuBuf {
                            name: "q",
                            va: qva,
                            bytes: q.clone(),
                        },
                        EmuBuf {
                            name: "q16",
                            va: q16va,
                            bytes: vec![0; rows * ROW * 2],
                        },
                    ],
                    pre_grid(rows),
                    [fa::PRECONVERT_BLOCK, 1, 1],
                    0,
                )?;
                (json!({"N": rows}), mem, q16va, rows * ROW * 2, Elem::F16)
            }
            Export::Direct => {
                let q16 = preconvert_ref(&q);
                let mem = emu_run(
                    &obj.bytes,
                    &obj.launch_symbol,
                    pack_direct([q16va, kva, vva, ova, pva], rows),
                    vec![
                        EmuBuf {
                            name: "q16",
                            va: q16va,
                            bytes: q16,
                        },
                        EmuBuf {
                            name: "k",
                            va: kva,
                            bytes: k.clone(),
                        },
                        EmuBuf {
                            name: "v",
                            va: vva,
                            bytes: v.clone(),
                        },
                        EmuBuf {
                            name: "positions",
                            va: pva,
                            bytes: pos,
                        },
                        EmuBuf {
                            name: "out",
                            va: ova,
                            bytes: vec![0; rows * ROW * 4],
                        },
                    ],
                    direct_grid(rows),
                    [fa::DIRECT_BLOCK, 1, 1],
                    direct_lds(self.codec),
                )?;
                (
                    json!({"N": rows, "S": 0}),
                    mem,
                    ova,
                    rows * ROW * 4,
                    Elem::F32,
                )
            }
        };
        let got = mem.read(out_va, out_len)?;
        let (worst, bad, reference) = match self.export {
            Export::Preconvert => {
                let want = preconvert_ref(&q);
                let bad = got
                    .chunks_exact(2)
                    .zip(want.chunks_exact(2))
                    .filter(|(g, w)| g != w)
                    .count();
                (max_diff(elem, &got[..], &want), bad, "cpu_f16_rne (exact)")
            }
            Export::Direct => {
                let kd = dequant(self.codec, &k, ctx);
                let vd = dequant(self.codec, &v, ctx);
                let ridx: Vec<usize> = (0..rows).collect();
                let want = attention_ref(&q, &kd, &vd, 0, &ridx);
                let mut worst = 0f64;
                let mut bad = 0usize;
                for (i, w) in want.iter().enumerate() {
                    let g = f32::from_le_bytes(got[i * 4..i * 4 + 4].try_into().unwrap());
                    let e = f64::from((g - w).abs());
                    if g.is_nan() || e > TOL {
                        bad += 1;
                    }
                    worst = if g.is_nan() {
                        f64::INFINITY
                    } else {
                        worst.max(e)
                    };
                }
                (worst, bad, "cpu_attention_f64_f16dequant (approx)")
            }
        };
        Ok(json!({
            "ran": true,
            "shape": shape,
            "symbol": obj.launch_symbol,
            "reference": reference,
            "max_abs_err": worst,
            "tolerance": if self.export == Export::Preconvert { 0.0 } else { TOL },
            "out_of_tolerance_elements": bad,
            "within_reference": bad == 0,
        }))
    }
}

impl Workload for Fa {
    fn kernel(&self) -> &'static str {
        "fa_prefill"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        json!({"N": self.rows, "S": self.start, "ctx": self.ctx(), "heads": H, "kv_heads": KV, "head_dim": D})
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.knobs_json();
        v["variant"] = json!(format!("{}.{}", codec_name(self.codec), self.export.name()));
        v
    }
    fn record_extra(&self) -> Vec<(String, Value)> {
        vec![(
            "fa_inputs".to_owned(),
            json!({
                "q": "synthetic N(0,1) f32 [N][16][256]",
                "kv": match self.codec {
                    FaKvCodec::Q8_0 => "synthetic K N(0,1), V N(0,0.7) as Q8_0 blocks [f16 amax/127][32 i8], 544 B rows",
                    FaKvCodec::Fp8E4M3 => "synthetic K N(0,1), V N(0,0.7) as E4M3 rows [512 codes x/(amax/224)][2 f16 scales], 516 B rows",
                },
                "positions": "start + r (full causal chunk)",
                "scale_attn": "1/16",
                "q16": "the shipped preconvert's output, run once in setup",
                "reference": match self.export {
                    Export::Preconvert => "exact (f16 RNE of Q)",
                    Export::Direct => "approximate only: f64 attention over f16-dequantized K/V, tolerance 5e-3; no exact CPU reference exists at production size",
                },
                "ref_rows": self.rows_desc,
                "buffers_bytes": {
                    "q": self.rows * ROW * 4,
                    "q16": self.rows * ROW * 2,
                    "k": self.ctx() * row_bytes(self.codec),
                    "v": self.ctx() * row_bytes(self.codec),
                    "out": self.rows * ROW * 4,
                },
            }),
        )]
    }
    fn requires_baseline_identity(&self) -> bool {
        true
    }

    fn candidate(&self) -> Result<Object> {
        match &self.rip {
            Some((path, text)) => self.rip_object(path, text),
            None => self.emitted(),
        }
    }

    fn baseline(&self) -> Result<Object> {
        let (bytes, artifact, selection, exports): (&[u8], &str, &str, Vec<&str>) =
            match self.codec {
                FaKvCodec::Q8_0 => (
                    pm::IMAGE,
                    Q8_OBJECT,
                    "crates/rdna-compute/src/fa_prefill_pm.rs:18 IMAGE (module fa_prefill_pm_gfx1201); load :284-295 fa_prefill_pm_load, \
                     preconvert launch :296-312 (grid [ceil(16N/4),1,1], 128 threads), direct launch :314-336 (grid [ceil(N/8),2,1], 160 threads, 65536 B LDS), \
                     kernargs :240-270, selection hipfire-dispatch/src/families/attention.rs:2353-2373 (fa_prefill_pm_admitted -> attention_q8_0_fa_prefill_pm_gfx1201), \
                     admission rdna-compute/src/attention.rs:5828-5865 (64..=8192 rows, context 1..=262144, H16/KV2/D256)",
                    vec![fa::SYMBOL_PRECONVERT, fa::SYMBOL_DIRECT],
                ),
                FaKvCodec::Fp8E4M3 => (
                    pm::FP8_IMAGE,
                    FP8_OBJECT,
                    "crates/rdna-compute/src/fa_prefill_pm.rs:42 FP8_IMAGE (module fa_prefill_fp8_pm_gfx1201); load :284-295, \
                     direct launch :314-336 (grid [ceil(N/8),2,1], 160 threads, 32768 B LDS), kernargs :240-270, \
                     selection hipfire-dispatch/src/families/attention.rs:2772-2790 (fa_prefill_pm_admitted -> attention_fp8_e4m3_fa_prefill_pm_gfx1201), \
                     admission rdna-compute/src/attention.rs:5828-5865; the Q16 is the Q8 module's preconvert (fa_prefill_pm.rs:18)",
                    vec![fa::SYMBOL_FP8_DIRECT],
                ),
            };
        let mut o = shipped_object(
            bytes,
            artifact,
            selection,
            symbol(self.codec, self.export),
            &exports,
            self.knobs_json(),
        )?;
        o.source["kernels_const"] = json!(match self.codec {
            FaKvCodec::Q8_0 => "crates/rdna-compute/src/fa_prefill_pm.rs:18 IMAGE",
            FaKvCodec::Fp8E4M3 => "crates/rdna-compute/src/fa_prefill_pm.rs:42 FP8_IMAGE",
        });
        Ok(o)
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let host = Host {
            q: synth_q(self.rows),
            k: synth_kv(self.codec, self.ctx(), 0x7e57_0000_0000_0001, 1.0),
            v: synth_kv(self.codec, self.ctx(), 0x7e57_0000_0000_0002, 0.7),
            pos: positions(self.rows, self.start),
        };
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            Ok(d)
        };
        let alloc = |n: usize| gpu.hip.malloc(n.max(16)).map_err(dev_err);
        let dev = Dev {
            q: up(&host.q)?,
            q16: alloc(self.rows * ROW * 2)?,
            k: up(&host.k)?,
            v: up(&host.v)?,
            pos: up(&host.pos)?,
            out: [alloc(self.out_bytes())?, alloc(self.out_bytes())?],
        };
        // The shipped preconvert, once: the direct kernel's Q16 is read-only to both arms.
        let module = gpu.hip.module_load_data(pm::IMAGE).map_err(dev_err)?;
        let func = gpu
            .hip
            .module_get_function(&module, pm::PRECONVERT)
            .map_err(dev_err)?;
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let mut kernarg = pack_pre(ptr(&dev.q), ptr(&dev.q16), self.rows);
        // SAFETY: the kernarg follows the preconvert's 40-byte ABI and both pointers address live
        // allocations of `rows * 4096` f32 and f16 elements.
        unsafe {
            gpu.hip.launch_kernel_blob(
                &func,
                pre_grid(self.rows),
                [fa::PRECONVERT_BLOCK, 1, 1],
                0,
                None,
                &mut kernarg,
            )
        }
        .map_err(dev_err)?;
        gpu.hip.device_synchronize().map_err(dev_err)?;
        self.host = Some(host);
        self.dev = Some(dev);
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        let dev = self.dev.as_ref().expect("launch before setup");
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let out = ptr(&dev.out[arm_index(arm)]);
        match self.export {
            Export::Direct => Launch {
                kernarg: pack_direct(
                    [ptr(&dev.q16), ptr(&dev.k), ptr(&dev.v), out, ptr(&dev.pos)],
                    self.rows,
                ),
                grid: direct_grid(self.rows),
                block: [fa::DIRECT_BLOCK, 1, 1],
                lds: direct_lds(self.codec),
            },
            Export::Preconvert => Launch {
                kernarg: pack_pre(ptr(&dev.q), out, self.rows),
                grid: pre_grid(self.rows),
                block: [fa::PRECONVERT_BLOCK, 1, 1],
                lds: 0,
            },
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let y = &dev.out[arm_index(arm)];
        gpu.hip.memset(y, 0, y.size()).map_err(dev_err)
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        let y = &dev.out[arm_index(arm)];
        let mut v = vec![0u8; y.size()];
        gpu.hip.memcpy_dtoh(&mut v, y).map_err(dev_err)?;
        v.truncate(self.out_bytes());
        Ok(v)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        if out.len() != self.out_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.out_bytes()
            ));
        }
        let (bad, worst, covered) = self.compare(out)?;
        let (name, tolerance) = match self.export {
            Export::Preconvert => ("cpu_f16_rne(exact,all_rows".to_owned(), 0.0),
            Export::Direct => (
                format!(
                    "cpu_attention_f64_f16dequant(approx,rows={},tolerance={TOL}",
                    self.rows_desc
                ),
                TOL,
            ),
        };
        Ok(RefReport {
            reference: format!("{name},elements={covered},out_of_reference={bad})"),
            max_abs_err: worst,
            tolerance,
            pass: bad == 0,
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        // Bit identity of two all-zero outputs would prove nothing (a failed preconvert or launch).
        if a.iter().all(|&x| x == 0) {
            return Err("baseline output is all zero: the inputs or the launch are invalid".into());
        }
        Ok(max_diff(self.out_elem(), a, b))
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => refused(e),
        }
    }
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect())
    }

    #[test]
    fn shape_grammar_and_presets() {
        for (name, n, s) in A3B_SHAPES {
            assert_eq!(
                parse_shape(&format!("a3b:{name}")).unwrap(),
                (n, s),
                "{name}"
            );
        }
        assert_eq!(parse_shape("N=512").unwrap(), (512, 0));
        assert_eq!(parse_shape("S=7,N=65").unwrap(), (65, 7));
        assert_eq!(parse_shape("a3b:n512-s0,S=64").unwrap(), (512, 64));
        for bad in [
            "a3b:nope",
            "fn:n512-s0",
            "N=63",
            "N=8193",
            "N=512,S=262144",
            "S=1",
            "N=512,Z=1",
            "N=512,N=1024",
            "a3b:n512-s0,a3b:n512-s0",
        ] {
            assert!(parse_shape(bad).is_err(), "{bad}");
        }
        assert!(parse_shape("N=64,S=262080").is_ok());
    }

    #[test]
    fn knob_to_spec_and_symbols() {
        let w = new("a3b:n512-s0", &mut knobs(&[]), None).unwrap();
        assert_eq!(w.kernel(), "fa_prefill");
        assert_eq!(w.candidate_knobs()["variant"], "q8_0.direct");
        assert!(w.requires_baseline_identity());
        let c = w.candidate().unwrap();
        assert_eq!(c.launch_symbol, fa::SYMBOL_DIRECT);
        assert_eq!(c.exports, [fa::SYMBOL_PRECONVERT, fa::SYMBOL_DIRECT]);
        let p = new("N=64", &mut knobs(&[("export", "preconvert")]), None).unwrap();
        assert_eq!(p.candidate().unwrap().launch_symbol, fa::SYMBOL_PRECONVERT);
        let f = new("N=64", &mut knobs(&[("codec", "fp8")]), None).unwrap();
        let c = f.candidate().unwrap();
        assert_eq!(c.launch_symbol, fa::SYMBOL_FP8_DIRECT);
        assert_eq!(c.exports, [fa::SYMBOL_FP8_DIRECT]);
        for (kv, why) in [
            (
                vec![("codec", "fp8"), ("export", "preconvert")],
                "fp8 preconvert",
            ),
            (vec![("codec", "bf16")], "codec"),
            (vec![("export", "split")], "export"),
            (vec![("ref-rows", "some")], "ref-rows"),
        ] {
            assert!(new("N=64", &mut knobs(&kv), None).is_err(), "{why}");
        }
    }

    #[test]
    fn baseline_is_the_shipped_bundle_for_every_codec_and_export() {
        for (kv, sym, artifact) in [
            (vec![], fa::SYMBOL_DIRECT, Q8_OBJECT),
            (
                vec![("export", "preconvert")],
                fa::SYMBOL_PRECONVERT,
                Q8_OBJECT,
            ),
            (vec![("codec", "fp8")], fa::SYMBOL_FP8_DIRECT, FP8_OBJECT),
        ] {
            let w = new("N=64", &mut knobs(&kv), None).unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, sym);
            assert_eq!(b.source["artifact"], artifact);
            assert_eq!(b.source["sha256"].as_str().unwrap().len(), 64);
            assert!(b.source["source"]
                .as_str()
                .unwrap()
                .contains("fa_prefill_pm.rs"));
        }
    }

    #[test]
    fn kernarg_layouts_and_launch_geometry() {
        let b = pack_direct([1, 2, 3, 4, 5], 512);
        assert_eq!(b.len(), 60);
        assert_eq!(u64::from_le_bytes(b[32..40].try_into().unwrap()), 5);
        for (off, v) in [(40, 16), (44, 2), (48, 256), (52, 512)] {
            assert_eq!(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()), v);
        }
        assert_eq!(f32::from_le_bytes(b[56..60].try_into().unwrap()), 0.0625);
        let p = pack_pre(1, 2, 64);
        assert_eq!(p.len(), 40);
        assert_eq!(u64::from_le_bytes(p[8..16].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(p[32..36].try_into().unwrap()), 64);
        assert_eq!(i32::from_le_bytes(p[36..40].try_into().unwrap()), 0);
        assert_eq!(direct_grid(4096), [512, 2, 1]);
        assert_eq!(direct_grid(65), [9, 2, 1]);
        assert_eq!(pre_grid(4096), [16384, 1, 1]);
        // The runtime's constants are the generator's.
        assert_eq!(pm::DIRECT_BLOCK, [fa::DIRECT_BLOCK, 1, 1]);
        assert_eq!(pm::PRE_BLOCK, [fa::PRECONVERT_BLOCK, 1, 1]);
        assert_eq!(pm::DIRECT_LDS, fa::DYNAMIC_LDS_BYTES);
        assert_eq!(pm::DIRECT_KERNARG_BYTES, fa::DIRECT_KERNARG_BYTES as usize);
        assert_eq!(pm::PRE_KERNARG_BYTES, fa::PRECONVERT_KERNARG_BYTES as usize);
        assert_eq!(direct_lds(FaKvCodec::Fp8E4M3), fa::FP8_DYNAMIC_LDS_BYTES);
        for c in [FaKvCodec::Q8_0, FaKvCodec::Fp8E4M3] {
            let rc = match c {
                FaKvCodec::Q8_0 => pm::FaPrefillKvCodec::Q8_0,
                FaKvCodec::Fp8E4M3 => pm::FaPrefillKvCodec::Fp8E4M3,
            };
            assert_eq!(row_bytes(c), rc.cache_row_bytes());
            assert_eq!(symbol(c, Export::Direct), rc.direct_symbol());
        }
    }

    #[test]
    fn synthesis_layouts_are_deterministic_and_finite() {
        let q8 = synth_kv(FaKvCodec::Q8_0, 3, 5, 1.0);
        assert_eq!(q8, synth_kv(FaKvCodec::Q8_0, 3, 5, 1.0));
        assert_eq!(q8.len(), 3 * 544);
        let sc = f16_val(u16::from_le_bytes([q8[0], q8[1]]));
        assert!(sc > 0.0 && sc < 0.1, "{sc}");
        let f8 = synth_kv(FaKvCodec::Fp8E4M3, 3, 5, 1.0);
        assert_eq!(f8.len(), 3 * 516);
        assert!(f8
            .chunks(516)
            .all(|r| r[..512].iter().all(|&b| b & 0x7f != 0x7f)));
        let sc = f16_val(u16::from_le_bytes([f8[512], f8[513]]));
        assert!(sc > 0.0 && sc < 0.1, "{sc}");
        // Dequantization reproduces the source within the codec's step.
        let kd = dequant(FaKvCodec::Fp8E4M3, &f8, 3);
        assert_eq!(kd.len(), 3 * 2 * 256);
        assert!(kd.iter().all(|v| v.is_finite()));
        assert_eq!(positions(3, 10), [10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0]);
    }

    #[test]
    fn the_attention_reference_hand_cases() {
        // Row 0 at position 0 attends to key 0 only: out = dequantized V of its KV head.
        let q = synth_q(2);
        let (k, v) = (
            synth_kv(FaKvCodec::Q8_0, 2, 1, 1.0),
            synth_kv(FaKvCodec::Q8_0, 2, 2, 0.7),
        );
        let (kd, vd) = (
            dequant(FaKvCodec::Q8_0, &k, 2),
            dequant(FaKvCodec::Q8_0, &v, 2),
        );
        let out = attention_ref(&q, &kd, &vd, 0, &[0]);
        for h in 0..H {
            let kvh = h / 8;
            for d in 0..D {
                assert_eq!(out[h * D + d], vd[kvh * D + d], "h{h} d{d}");
            }
        }
        // Row 1 at position 1 attends to keys 0 and 1: a convex combination of their V.
        let out = attention_ref(&q, &kd, &vd, 0, &[1]);
        for h in 0..H {
            let kvh = h / 8;
            for d in 0..D {
                let (a, b) = (vd[kvh * D + d], vd[(KV + kvh) * D + d]);
                let o = out[h * D + d];
                assert!(o >= a.min(b) - 1e-6 && o <= a.max(b) + 1e-6);
            }
        }
        // The preconvert reference is f16 RNE.
        let q16 = preconvert_ref(&1.0f32.to_le_bytes());
        assert_eq!(q16, 0x3c00u16.to_le_bytes());
        assert_eq!(
            preconvert_ref(&(1.0f32 + 2f32.powi(-11)).to_le_bytes()),
            q16,
            "tie to even"
        );
    }

    #[test]
    fn there_is_a_check_and_the_all_zero_guard() {
        let w = new("N=64", &mut knobs(&[]), None).unwrap();
        assert!(
            w.max_abs_diff(&[0; 16], &[0; 16]).is_err(),
            "all-zero is vacuous"
        );
        assert_eq!(
            w.max_abs_diff(&[0, 0, 0x80, 0x3f], &[0, 0, 0x80, 0x3f]),
            Ok(0.0)
        );
        assert!(w.check(&[0; 4]).is_err(), "wrong length is refused");
    }

    const TWIN_Q8: &str = include_str!("../../../peacemaker-isa/src/kernels/fa_prefill.rip");
    const TWIN_FP8: &str = include_str!("../../../peacemaker-isa/src/kernels/fa_prefill_fp8.rip");

    #[test]
    fn rip_twins_are_identical_to_the_generator() {
        for (src, codec, export) in [
            (TWIN_Q8, FaKvCodec::Q8_0, Export::Direct),
            (TWIN_Q8, FaKvCodec::Q8_0, Export::Preconvert),
            (TWIN_FP8, FaKvCodec::Fp8E4M3, Export::Direct),
        ] {
            let (_, sym, info) = compile_rip(src, codec, export).unwrap();
            assert_eq!(sym, symbol(codec, export));
            assert_eq!(info["identical_to_generator"], true, "{codec:?} {export:?}");
            assert_eq!(info["sha256"], sha256_hex(src.as_bytes()));
        }
    }

    #[test]
    fn rip_garbage_and_the_wrong_twin_are_refused() {
        assert!(compile_rip("fn emit(", FaKvCodec::Q8_0, Export::Direct).is_err());
        // The fp8 twin has no preconvert variant, the Q8 twin builds the Q8 kernel under any codec label.
        assert!(compile_rip(TWIN_FP8, FaKvCodec::Fp8E4M3, Export::Preconvert).is_err());
    }
}
