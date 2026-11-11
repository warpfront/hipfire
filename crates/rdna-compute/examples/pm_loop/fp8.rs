// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `fp8_gemm` workload: the builder-emitted MQ4G256V2 x E4M3 ratio-fold prefill
//! GEMM for gfx1201 (`peacemaker-isa emit --kernel fp8_gemm`, the F2 route)
//! against the shipped bundle `kernels/gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco`.
//!
//! # Shape grammar
//! `M=<rows>[+<rows>..],N=<tokens>,K=<k>` (all required, any order),
//! `K % 256 == 0`, every row count `> 0` and `% 4 == 0`, `N >= 1`. The row counts
//! are the runtime's `dims`: `set`/`add` one, `silu` two equal ones (`M=m` is
//! shorthand for `m+m`), `qkv` three, `qkvza` four. The runtime admits the F2 route
//! at `N >= 512` (`fp8_f2_row_active`); `pm_loop` does not enforce that floor.
//! Production-scale: `M=17408,N=512,K=5120` (`--epi silu`, the gate/up pair),
//! `M=5120,N=512,K=17408` (`--epi add`).
//!
//! # Knobs (candidate; names as `peacemaker-isa emit --kernel fp8_gemm`)
//! `--scale row` (default) | `k128` | `both`, `--epi set` (default) | `add` |
//! `silu` | `silu-bf16` | `qkv` | `qkvza`. `--scale both` times the
//! `peacemaker-isa emit --scale both --epi all` module (ten exports, all certified)
//! and launches the one symbol `--launch-scale row|k128` (default `row`) names;
//! it supports `--epi set|add|silu|qkv|qkvza` only (the module's epilogue set).
//! `--epi qkvzagdn` is refused: the 144-byte GDN ABI needs the conv weights, the
//! persistent conv ring and the q/k/v bases (`gemm_qkvza_mq4g256v2_fp8_gdn_prepared`),
//! which this workload does not synthesize.
//!
//! # Launch (exactly the runtime's `fp8_f2_row_launch`)
//! Two kernels per call: the shipped weight repacker
//! `mq4v2_fp8_fragment_repack_gfx1201` (run ONCE in `setup`, outside every timed
//! interval; it builds `Wf`/`Rw`/`Ew` from the raw MQ4G256V2 weights and is shared
//! by both arms), then the GEMM: grid `[ceil(Mt/256), ceil(N/128), 1]`, 256 threads,
//! 19456 B dynamic LDS, kernarg `Wf Rw Ew X8 D Y0 Y1 Y2 Y3` pointers then
//! `M0 M1 M2 M3 K N` (96 bytes; unused `Y` are null, `silu` writes only `Y0`).
//!
//! # Baseline
//! The shipped bundle and the symbol the runtime selects (Row, the same symbol
//! name the generator emits): `gemm_mq4g256v2_fp8_{set,add,silu,qkv,qkvza}_row_b1`
//! and `..._silu_row_b1_bf16`. K128 symbols exist in the bundle but the runtime
//! never launches them; the record says so (`runtime_selected`). Selection
//! file:line and the artifact sha256 go in `baseline.source`.
//!
//! # Correctness
//! The candidate must be bitwise equal to the shipped baseline
//! (`pass_basis: identical_to_baseline (no CPU reference)`). There is NO CPU
//! reference for this kernel: the ratio-fold arithmetic (E4M3 WMMA accumulation
//! order, `Rw`/`Ew` ratio fold, SiLU region) has no committed transcription in the
//! repo (`pm_fp8_verify_oracle` covers the different `mq4_fp8_verify` family), so
//! `check` returns an error and the driver reports `reference: "none"`.
//!
//! # Inputs (deterministic, synthetic)
//! Weights are raw MQ4G256V2 (`[M][K/256][136 B]`: two `(f16 scale, f16 zero)`
//! pairs, then 128 random nibble bytes; scales `0.02..0.035`, symmetric zero) fed
//! through the shipped repacker. Activations are N(0,1)*amp(col) quantized the way
//! the production pre-pass does it: one power-of-two scale `2^ceil(log2(amax/224))`
//! clamped to `[-31, 9]` per row (Row) or per `(token, K128 block)` (K128), codes
//! `E4M3(x / scale)` round-to-nearest-even, `D = [N]` (Row) / `[N][K/128]` (K128).
//!
//! # `--emu`
//! Lifts the candidate and runs one workgroup on the peacemaker emulator at a tiny
//! shape. Dense FP8 WMMA is unqualified in the emulator, so this reports the
//! emulator's own refusal; the instruction is not modelled.

use crate::common::e4m3;
use crate::iu4::{abi_signature, max_diff};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use peacemaker_isa::kernels::fp8_gemm::spec::LDS_BYTES;
use peacemaker_isa::kernels::fp8_gemm::{self, ActScale, Epi, Spec};
use peacemaker_isa::Arch;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const GROUP_BYTES: usize = 136;
const TILE_ROWS: usize = 256;
const TILE_TOKENS: usize = 128;
const THREADS: u32 = 256;
const OBJECT: &str = "kernels/gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco";
const REPACK_OBJECT: &str = "kernels/mq4v2_fp8_fragment_repack_gfx1201.hsaco";
const REPACK_SYMBOL: &str = "mq4v2_fp8_fragment_repack_gfx1201";

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

fn dev_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// `--epi` name (`silu-bf16` is its own spelling, unlike `Epi::name`).
fn epi_name(e: Epi) -> &'static str {
    match e {
        Epi::GateUpSiluBf16 => "silu-bf16",
        other => other.name(),
    }
}

fn is_silu(e: Epi) -> bool {
    matches!(e, Epi::GateUpSilu | Epi::GateUpSiluBf16)
}

/// `M=a[+b..],N=..,K=..` -> (row counts, N, K).
fn parse_shape(s: &str) -> Result<(Vec<usize>, usize, usize)> {
    let (mut m, mut n, mut k) = (None, None, None);
    for part in s.split(',') {
        let (key, v) = part
            .split_once('=')
            .ok_or_else(|| format!("shape `{s}`: expected M=..[+..],N=..,K=.."))?;
        let key = key.trim();
        let num = |t: &str| -> Result<usize> {
            t.trim()
                .parse()
                .map_err(|e| format!("shape `{s}`: {key}: {e}"))
        };
        match key {
            "M" => {
                let dims = v.split('+').map(num).collect::<Result<Vec<_>>>()?;
                if m.replace(dims).is_some() {
                    return Err(format!("shape `{s}`: duplicate M"));
                }
            }
            "N" | "K" => {
                let slot = if key == "N" { &mut n } else { &mut k };
                if slot.replace(num(v)?).is_some() {
                    return Err(format!("shape `{s}`: duplicate {key}"));
                }
            }
            other => return Err(format!("shape `{s}`: unknown dimension `{other}`")),
        }
    }
    match (m, n, k) {
        (Some(m), Some(n), Some(k)) => Ok((m, n, k)),
        _ => Err(format!("shape `{s}`: M, N and K are all required")),
    }
}

/// The runtime's `dims` for `epi` from the shape's row counts.
fn resolve_dims(epi: Epi, ms: Vec<usize>) -> Result<Vec<usize>> {
    let dims = if is_silu(epi) && ms.len() == 1 {
        vec![ms[0], ms[0]]
    } else {
        ms
    };
    if dims.len() != epi.count() {
        return Err(format!(
            "--epi {} takes {} row count(s) in M=a+b.., got {}",
            epi_name(epi),
            epi.count(),
            dims.len()
        ));
    }
    if is_silu(epi) && dims[0] != dims[1] {
        return Err("--epi silu needs gate rows == up rows".into());
    }
    if let Some(m) = dims.iter().find(|&&m| m == 0 || m % 4 != 0) {
        return Err(format!("row count {m}: need > 0 and % 4 == 0"));
    }
    Ok(dims)
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal (Box-Muller, one variate per call).
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn f16_val(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// `m` rows x `k/256` raw MQ4G256V2 groups (the iu4 workload's distribution).
fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for blk in w.chunks_exact_mut(GROUP_BYTES) {
        let sc0 = 0.02 + (rng.next() % 16) as f32 * 0.001;
        let sc1 = 0.03 + (rng.next() % 16) as f32 * 0.001;
        for (i, sc) in [sc0, sc1].into_iter().enumerate() {
            let bits = f16_bits(sc);
            blk[4 * i..4 * i + 2].copy_from_slice(&bits.to_le_bytes());
            // The symmetric repack reads only the scale; keep the asymmetric zero.
            let zp = f16_bits(-8.0 * f16_val(bits));
            blk[4 * i + 2..4 * i + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for q in blk[8..].chunks_exact_mut(8) {
            q.copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    w
}

/// The production pre-pass scale: `2^ceil(log2(amax / 224))` from the f32 bit
/// pattern, clamped to `[-31, 9]`; an all-zero group scales by 1.
fn pow2_scale(amax: f32) -> f32 {
    if !(amax > 0.0) {
        return 1.0;
    }
    let bits = (amax * (1.0 / 224.0)).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((e.clamp(-31, 9) + 127) as u32) << 23)
}

/// `X8 [N][K]` E4M3 codes and `D [N][groups]` power-of-two scales (`groups` = 1
/// for Row, `K/128` for K128).
fn synth_x(n: usize, k: usize, groups: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut x8 = vec![0u8; n * k];
    let mut d = vec![0f32; n * groups];
    let span = k / groups;
    let mut row = vec![0f32; k];
    for t in 0..n {
        let amp = 1.0 + (t % 4) as f64;
        for v in &mut row {
            *v = (rng.normal() * amp) as f32;
        }
        for g in 0..groups {
            let blk = &row[g * span..(g + 1) * span];
            let scale = pow2_scale(blk.iter().fold(0f32, |a, v| a.max(v.abs())));
            d[t * groups + g] = scale;
            for (o, v) in x8[t * k + g * span..][..span].iter_mut().zip(blk) {
                *o = e4m3(v / scale);
            }
        }
    }
    (x8, d)
}

struct Host {
    /// One raw MQ4G256V2 matrix per `dims` entry.
    w: Vec<Vec<u8>>,
    x8: Vec<u8>,
    d: Vec<f32>,
    /// Initial `Y0` (`--epi add` accumulates into it).
    y0: Vec<f32>,
}

impl Host {
    fn synth(dims: &[usize], n: usize, k: usize, groups: usize, add: bool) -> Self {
        let w = dims
            .iter()
            .enumerate()
            .map(|(i, &m)| synth_weights(m, k, 0x9e37_79b9_7f4a_7c15 ^ ((i as u64) << 40)))
            .collect();
        let (x8, d) = synth_x(n, k, groups, 0x1234_5678_9abc_def1);
        let mut rng = Rng::new(0xd1b5_4a32_d192_ed03);
        let y0 = if add {
            (0..n * dims[0])
                .map(|_| (0.25 + rng.unit()) as f32)
                .collect()
        } else {
            Vec::new()
        };
        Self { w, x8, d, y0 }
    }
}

fn put_ptr(b: &mut Vec<u8>, p: u64) {
    b.extend_from_slice(&p.to_le_bytes());
}

fn put_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// The 96-byte GEMM kernarg: `Wf Rw Ew X8 D Y0..Y3` then `M0..M3 K N`.
fn pack_gemm(ptrs: [u64; 9], ms: [i32; 4], k: usize, n: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(96);
    for p in ptrs {
        put_ptr(&mut b, p);
    }
    for v in ms {
        put_i32(&mut b, v);
    }
    put_i32(&mut b, k as i32);
    put_i32(&mut b, n as i32);
    b
}

/// The 80-byte repack kernarg: `W0..W3 Wf Rw Ew M0..M3 K family`.
fn pack_repack(
    ws: [u64; 4],
    wf: u64,
    rw: u64,
    ew: u64,
    ms: [i32; 4],
    k: usize,
    family: i32,
) -> Vec<u8> {
    let mut b = Vec::with_capacity(80);
    for p in ws.into_iter().chain([wf, rw, ew]) {
        put_ptr(&mut b, p);
    }
    for v in ms {
        put_i32(&mut b, v);
    }
    put_i32(&mut b, k as i32);
    put_i32(&mut b, family);
    b
}

/// `(Wf bytes, Rw bytes, Ew bytes)` for `mt` virtual rows
/// (`scratch::fp8_f2_weight_extents`).
fn weight_extents(mt: usize, k: usize) -> (usize, usize, usize) {
    let mp = mt.div_ceil(TILE_ROWS) * TILE_ROWS;
    (mp * k, mp * (k / 128) * 4, mp * 4)
}

struct Dev {
    w: Vec<hip_bridge::DeviceBuffer>,
    wf: hip_bridge::DeviceBuffer,
    rw: hip_bridge::DeviceBuffer,
    ew: hip_bridge::DeviceBuffer,
    x8: hip_bridge::DeviceBuffer,
    d: hip_bridge::DeviceBuffer,
    /// `[arm][output]`.
    y: [Vec<hip_bridge::DeviceBuffer>; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScaleKnob {
    One(ActScale),
    Both,
}

pub struct Fp8 {
    dims: Vec<usize>,
    n: usize,
    k: usize,
    scale: ScaleKnob,
    /// The launched spec (for `--scale both`, `--launch-scale`).
    cand: Spec,
    /// `--rip`: source path and text of the candidate script.
    rip: Option<(String, String)>,
    host: Option<Host>,
    dev: Option<Dev>,
}

fn parse_scale(s: &str) -> Result<ScaleKnob> {
    match s {
        "both" => Ok(ScaleKnob::Both),
        other => Ok(ScaleKnob::One(other.parse().map_err(|e: String| {
            format!("--scale {other}: {e} (row|k128|both)")
        })?)),
    }
}

/// The epilogues of `peacemaker-isa emit --scale both --epi all`.
const BOTH_EPIS: [Epi; 5] = [Epi::Set, Epi::Add, Epi::GateUpSilu, Epi::Qkv, Epi::Qkvza];

fn parse_spec(scale: ScaleKnob, launch: ActScale, epi: &str) -> Result<Spec> {
    let epi: Epi = epi.parse()?;
    if epi == Epi::QkvzaGdn {
        return Err(
            "--epi qkvzagdn is not supported by pm_loop: the 144-byte GDN ABI needs the conv \
             weights, the conv ring and the q/k/v bases that this workload does not synthesize"
                .into(),
        );
    }
    if scale == ScaleKnob::Both && !BOTH_EPIS.contains(&epi) {
        return Err(format!(
            "--scale both emits the module of set|add|silu|qkv|qkvza; --epi {} is not in it",
            epi_name(epi)
        ));
    }
    let act_scale = match scale {
        ScaleKnob::One(s) => s,
        ScaleKnob::Both => launch,
    };
    let spec = Spec {
        arch: Arch::Gfx1201,
        act_scale,
        epi,
    };
    spec.validate()?;
    Ok(spec)
}

pub fn new(shape: &str, knobs: &mut Knobs, rip: Option<&Path>) -> Result<Box<dyn Workload>> {
    let (ms, n, k) = parse_shape(shape)?;
    if n == 0 || k == 0 || k % 256 != 0 {
        return Err(format!("shape N={n},K={k}: need N >= 1, K % 256 == 0"));
    }
    let scale = parse_scale(&knobs.take_or("scale", "row"))?;
    let launch_given = knobs.0.contains_key("launch-scale");
    let launch: ActScale = knobs
        .take_or("launch-scale", "row")
        .parse()
        .map_err(|e: String| format!("--launch-scale: {e} (row|k128)"))?;
    if scale != ScaleKnob::Both && launch_given {
        return Err("--launch-scale only applies to --scale both".into());
    }
    let cand = parse_spec(scale, launch, &knobs.take_or("epi", "set"))?;
    let dims = resolve_dims(cand.epi, ms)?;
    if dims.iter().sum::<usize>() > i32::MAX as usize || n > i32::MAX as usize {
        return Err("shape exceeds the i32 kernarg range".into());
    }
    if rip.is_some() && scale == ScaleKnob::Both {
        return Err(
            "--rip with --scale both: the twin builds one variant; use --scale row|k128".into(),
        );
    }
    let rip = match rip {
        Some(p) => Some((
            p.display().to_string(),
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    Ok(Box::new(Fp8 {
        dims,
        n,
        k,
        scale,
        cand,
        rip,
        host: None,
        dev: None,
    }))
}

/// Compile `.rip` source for `spec`, refuse unless its ABI equals that of
/// `fp8_gemm::emit(spec)`; returns the ELF, symbol and the `candidate.rip`
/// fields (`variant`, `sha256`, `identical_to_generator`).
fn compile_rip(source: &str, spec: Spec) -> Result<(Vec<u8>, String, Value)> {
    let variant = spec.variant();
    let compiled = peacemaker_rip::compile(source, spec.arch, &variant)?;
    let generator = fp8_gemm::emit(spec)?;
    let symbol = spec.symbol();
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, spec.arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, spec.arch)?;
    let got = abi_signature(&elf, &symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = abi_signature(&gen_elf, &symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with fp8_gemm::emit({variant}):\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    let info = json!({
        "sha256": sha256_hex(source.as_bytes()),
        "variant": variant,
        "identical_to_generator": compiled.s_text == generator.s_text,
    });
    Ok((elf, symbol, info))
}

impl Fp8 {
    fn groups(&self) -> usize {
        match self.cand.act_scale {
            ActScale::Row => 1,
            ActScale::K128 => self.k / 128,
        }
    }

    fn silu(&self) -> bool {
        is_silu(self.cand.epi)
    }

    fn mt(&self) -> usize {
        self.dims.iter().sum()
    }

    /// Element width of every output (BF16 only for `silu-bf16`).
    fn width(&self) -> usize {
        if self.cand.epi == Epi::GateUpSiluBf16 {
            2
        } else {
            4
        }
    }

    /// Row counts of the outputs `Y0..` (`silu` writes only `Y0`, gate rows wide).
    fn out_rows(&self) -> Vec<usize> {
        if self.silu() {
            vec![self.dims[0]]
        } else {
            self.dims.clone()
        }
    }

    fn out_bytes(&self) -> usize {
        self.out_rows()
            .iter()
            .map(|r| self.n * r * self.width())
            .sum()
    }

    fn knobs_json(&self) -> Value {
        json!({
            "scale": match self.scale {
                ScaleKnob::One(s) => s.name(),
                ScaleKnob::Both => "both",
            },
            "launch_scale": self.cand.act_scale.name(),
            "epi": epi_name(self.cand.epi),
        })
    }

    fn emitted(&self) -> Result<Object> {
        let arch = self.cand.arch;
        let symbol = self.cand.symbol();
        let (elf, exports, text) = match self.scale {
            ScaleKnob::One(_) => {
                let e = fp8_gemm::emit(self.cand)?;
                let elf = peacemaker_isa::native::assemble(&e.s_text, arch)?;
                (
                    elf,
                    vec![symbol.clone()],
                    "peacemaker_isa::kernels::fp8_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                )
            }
            ScaleKnob::Both => {
                let mut all = Vec::with_capacity(10);
                let mut exports = Vec::with_capacity(10);
                for act_scale in [ActScale::Row, ActScale::K128] {
                    for epi in BOTH_EPIS {
                        let spec = Spec {
                            arch,
                            act_scale,
                            epi,
                        };
                        exports.push(spec.symbol());
                        all.push(fp8_gemm::emit(spec)?);
                    }
                }
                let (text, _proof) = fp8_gemm::module(&all)?;
                let elf = peacemaker_isa::native::assemble(&text, arch)?;
                (
                    elf,
                    exports,
                    "peacemaker_isa::kernels::fp8_gemm::module (emit --scale both --epi all) + native::assemble",
                )
            }
        };
        Ok(Object {
            source: json!({
                "source": text,
                "variant": self.cand.variant(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.knobs_json()}),
            exports,
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn rip_object(&self, path: &str, text: &str) -> Result<Object> {
        let (elf, symbol, mut info) = compile_rip(text, self.cand)?;
        info["path"] = json!(path);
        Ok(Object {
            source: json!({
                "source": format!("peacemaker_rip::compile({path}) + native::assemble"),
                "variant": self.cand.variant(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.knobs_json(), "rip": info}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn y_bufs(&self, gpu: &Gpu) -> Result<Vec<hip_bridge::DeviceBuffer>> {
        self.out_rows()
            .iter()
            .map(|r| {
                gpu.hip
                    .malloc((self.n * r * self.width()).max(16))
                    .map_err(dev_err)
            })
            .collect()
    }

    /// Run the candidate object on the emulator at a tiny shape. Dense FP8
    /// WMMA is unqualified, so this normally reports the emulator's refusal.
    fn run_emu(&self) -> Result<Value> {
        use peacemaker_ir::emu::{Emulator, Launch as ELaunch, Memory};
        let obj = self.candidate()?;
        let lifted = peacemaker_lift::lift_object(
            &obj.bytes,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e:?}"))?;
        let emu = Emulator::new(&lifted.program, &obj.launch_symbol)?.hw_model();
        let (n, k) = (TILE_TOKENS, 256usize);
        let dims: Vec<usize> = if self.silu() {
            vec![TILE_ROWS / 2; 2]
        } else {
            vec![TILE_ROWS / self.dims.len().max(1); self.dims.len()]
        };
        let mt: usize = dims.iter().sum();
        let (wf, rw, ew) = weight_extents(mt, k);
        let va = |i: u64| 0x1000_0000 * (i + 1);
        let mut ms = [0i32; 4];
        for (m, &d) in ms.iter_mut().zip(&dims) {
            *m = d as i32;
        }
        let out_rows = if self.silu() {
            vec![dims[0]]
        } else {
            dims.clone()
        };
        let d_bytes = n
            * 4
            * if self.cand.act_scale == ActScale::K128 {
                k / 128
            } else {
                1
            };
        let mut ptrs = [0u64; 9];
        let mut mem = Memory::default();
        let mut regions: Vec<(String, u64, usize)> = vec![
            ("wf".into(), va(1), wf),
            ("rw".into(), va(2), rw),
            ("ew".into(), va(3), ew),
            ("x8".into(), va(4), n * k),
            ("d".into(), va(5), d_bytes),
        ];
        for (i, r) in out_rows.iter().enumerate() {
            regions.push((format!("y{i}"), va(6 + i as u64), n * r * self.width()));
        }
        for (i, (name, v, len)) in regions.iter().enumerate() {
            ptrs[i] = *v;
            mem.map(name.clone(), *v, vec![0; *len])?;
            mem.map(format!("{name}_tail"), v + *len as u64, vec![0; 4096])?;
        }
        let kva = va(0);
        let mut kern = pack_gemm(ptrs, ms, k, n);
        kern.resize(self.cand.kernargs().size as usize, 0);
        mem.map("kernarg".into(), kva, kern)?;
        emu.run_workgroup(
            &ELaunch {
                kernarg_ptr: kva,
                workgroup: [0, 0, 0],
                block: [THREADS, 1, 1],
                dynamic_lds: LDS_BYTES,
            },
            &mut mem,
        )?;
        Ok(json!({
            "ran": true,
            "shape": {"dims": dims, "N": n, "K": k},
            "symbol": obj.launch_symbol,
            "reference": "none",
            "note": "one workgroup executed; no CPU reference exists to compare it with",
        }))
    }
}

impl Workload for Fp8 {
    fn kernel(&self) -> &'static str {
        "fp8_gemm"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        json!({"dims": self.dims, "Mt": self.mt(), "N": self.n, "K": self.k})
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.knobs_json();
        v["variant"] = json!(self.cand.variant());
        v
    }
    fn record_extra(&self) -> Vec<(String, Value)> {
        let (wf, rw, ew) = weight_extents(self.mt(), self.k);
        vec![(
            "fp8_inputs".to_owned(),
            json!({
                "weights": "synthetic raw MQ4G256V2 [M][K/256][136 B] through the shipped repacker",
                "repack": {
                    "artifact": REPACK_OBJECT,
                    "symbol": REPACK_SYMBOL,
                    "kernels_const": "crates/rdna-compute/src/kernels.rs:5358 MQ4V2_FP8_FRAGMENT_REPACK_GFX1201",
                    "sha256": sha256_hex(rdna_compute::kernels::MQ4V2_FP8_FRAGMENT_REPACK_GFX1201),
                    "family": i32::from(self.silu()),
                    "launch": "gemm.rs:10886 family, :10901-10905 grid [ceil(Mt/256)*4, K/256, 1] x 256; run once in setup, outside every timed interval",
                    "extents_bytes": {"Wf": wf, "Rw": rw, "Ew": ew},
                },
                "activations": "synthetic N(0,1)*amp, pow2 scale 2^ceil(log2(amax/224)) per row (Row) or per K128 block (K128), E4M3 RNE codes",
                "d_layout": match self.cand.act_scale {
                    ActScale::Row => "[N] f32",
                    ActScale::K128 => "[N][K/128] f32",
                },
                "out_rows": self.out_rows(),
                "output_dtype": if self.width() == 2 { "bf16" } else { "f32" },
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
        let bytes = rdna_compute::kernels::GEMM_MQ4G256V2_WMMA_FP8_GFX12_B1;
        let symbol = self.cand.symbol();
        if !contains(bytes, symbol.as_bytes()) {
            return Err(format!(
                "symbol {symbol} is not in the shipped object {OBJECT}"
            ));
        }
        let selected = self.cand.act_scale == ActScale::Row;
        Ok(Object {
            bytes: bytes.to_vec(),
            launch_symbol: symbol.clone(),
            exports: vec![symbol],
            desc: json!({"artifact": OBJECT, "knobs": {
                "scale": self.cand.act_scale.name(),
                "epi": epi_name(self.cand.epi),
            }}),
            source: json!({
                "source": "crates/rdna-compute/src/gemm.rs:10766 fp8_f2_row_launch: symbol choice :10819-10829 (Row symbols only), \
                           embedded-bundle load :10847-10857, admission :10666-10690 (fp8_f2_row_active) and :10785-10803, \
                           repack launch :10901-10921, GEMM launch :10954-10962 (grid [ceil(Mt/256), ceil(N/128), 1], 256 threads, 19456 B LDS)",
                "artifact": OBJECT,
                "kernels_const": "crates/rdna-compute/src/kernels.rs:5330 GEMM_MQ4G256V2_WMMA_FP8_GFX12_B1",
                "sha256": sha256_hex(bytes),
                "runtime_selected": selected,
                "runtime_selected_note": if selected {
                    "the runtime launches this Row symbol"
                } else {
                    "the bundle carries the K128 symbols but no runtime route launches them; the same shipped bundle is timed as the K128 reference"
                },
            }),
        })
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let groups = self.groups();
        let host = Host::synth(
            &self.dims,
            self.n,
            self.k,
            groups,
            self.cand.epi == Epi::Add,
        );
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            Ok(d)
        };
        let (wf_b, rw_b, ew_b) = weight_extents(self.mt(), self.k);
        let d_bytes: Vec<u8> = host.d.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dev = Dev {
            w: host.w.iter().map(|w| up(w)).collect::<Result<Vec<_>>>()?,
            wf: gpu.hip.malloc(wf_b).map_err(dev_err)?,
            rw: gpu.hip.malloc(rw_b).map_err(dev_err)?,
            ew: gpu.hip.malloc(ew_b).map_err(dev_err)?,
            x8: up(&host.x8)?,
            d: up(&d_bytes)?,
            y: [self.y_bufs(gpu)?, self.y_bufs(gpu)?],
        };

        // The shipped repacker, once: Wf/Rw/Ew are read-only to the GEMM arms.
        let module = gpu
            .hip
            .module_load_data(rdna_compute::kernels::MQ4V2_FP8_FRAGMENT_REPACK_GFX1201)
            .map_err(dev_err)?;
        let func = gpu
            .hip
            .module_get_function(&module, REPACK_SYMBOL)
            .map_err(dev_err)?;
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let mut ws = [0u64; 4];
        let mut ms = [0i32; 4];
        for (i, (w, &m)) in dev.w.iter().zip(&self.dims).enumerate() {
            ws[i] = ptr(w);
            ms[i] = m as i32;
        }
        let mut kernarg = pack_repack(
            ws,
            ptr(&dev.wf),
            ptr(&dev.rw),
            ptr(&dev.ew),
            ms,
            self.k,
            i32::from(self.silu()),
        );
        // SAFETY: the kernarg follows the repacker's 80-byte ABI and every
        // pointer addresses a live allocation owned by `dev` sized per
        // `weight_extents` / `m * K/256 * 136`.
        unsafe {
            gpu.hip.launch_kernel_blob(
                &func,
                [
                    (self.mt().div_ceil(TILE_ROWS) * 4) as u32,
                    (self.k / 256) as u32,
                    1,
                ],
                [THREADS, 1, 1],
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
        let ys = &dev.y[arm_index(arm)];
        let mut p = [
            ptr(&dev.wf),
            ptr(&dev.rw),
            ptr(&dev.ew),
            ptr(&dev.x8),
            ptr(&dev.d),
            0,
            0,
            0,
            0,
        ];
        for (i, y) in ys.iter().enumerate() {
            p[5 + i] = ptr(y);
        }
        let mut ms = [0i32; 4];
        for (m, &d) in ms.iter_mut().zip(&self.dims) {
            *m = d as i32;
        }
        Launch {
            kernarg: pack_gemm(p, ms, self.k, self.n),
            grid: [
                self.mt().div_ceil(TILE_ROWS) as u32,
                self.n.div_ceil(TILE_TOKENS) as u32,
                1,
            ],
            block: [THREADS, 1, 1],
            lds: LDS_BYTES,
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let ys = &dev.y[arm_index(arm)];
        for (i, y) in ys.iter().enumerate() {
            if i == 0 && self.cand.epi == Epi::Add {
                let host = self.host.as_ref().ok_or("reset before setup")?;
                let bytes: Vec<u8> = host.y0.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.hip.memcpy_htod(y, &bytes).map_err(dev_err)?;
            } else {
                gpu.hip.memset(y, 0, y.size()).map_err(dev_err)?;
            }
        }
        Ok(())
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        let mut out = Vec::with_capacity(self.out_bytes());
        for (y, rows) in dev.y[arm_index(arm)].iter().zip(self.out_rows()) {
            // Device buffers are padded to 16 bytes; the payload leads.
            let mut v = vec![0u8; y.size()];
            gpu.hip.memcpy_dtoh(&mut v, y).map_err(dev_err)?;
            v.truncate(self.n * rows * self.width());
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    fn check(&self, _out: &[u8]) -> Result<RefReport> {
        Err(
            "no CPU reference for fp8_gemm (the E4M3 WMMA ratio fold has no committed CPU \
             transcription; pm_fp8_verify_oracle covers the different mq4_fp8_verify family); \
             correctness is candidate-vs-baseline only"
                .into(),
        )
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        // Bit identity of two all-zero outputs would prove nothing (a failed repack or launch).
        if a.iter().all(|&x| x == 0) {
            return Err("baseline output is all zero: the inputs or the launch are invalid".into());
        }
        Ok(max_diff(a, b, self.width()))
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => Value::String(format!("refused ({e})")),
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
    fn shape_grammar_and_dims() {
        assert_eq!(
            parse_shape("M=17408,N=512,K=5120").unwrap(),
            (vec![17408], 512, 5120)
        );
        assert_eq!(
            parse_shape("K=256,N=1,M=8+4+4").unwrap(),
            (vec![8, 4, 4], 1, 256)
        );
        assert!(parse_shape("M=4,N=1").is_err());
        assert!(parse_shape("M=4,M=4,N=1,K=256").is_err());
        assert!(parse_shape("M=4,N=1,K=256,Z=1").is_err());
        assert_eq!(resolve_dims(Epi::GateUpSilu, vec![64]).unwrap(), [64, 64]);
        assert!(resolve_dims(Epi::GateUpSilu, vec![64, 32]).is_err());
        assert!(resolve_dims(Epi::Qkv, vec![64, 64]).is_err());
        assert!(resolve_dims(Epi::Set, vec![6]).is_err());
        assert!(resolve_dims(Epi::Set, vec![0]).is_err());
        assert!(new("M=64,N=1,K=128", &mut knobs(&[]), None).is_err());
    }

    #[test]
    fn knob_to_spec_and_symbol() {
        let w = new(
            "M=64,N=128,K=256",
            &mut knobs(&[("epi", "silu-bf16")]),
            None,
        )
        .unwrap();
        assert_eq!(w.kernel(), "fp8_gemm");
        assert_eq!(
            w.candidate_knobs()["variant"],
            "ratio-256x128x8-row.silu-bf16"
        );
        assert_eq!(
            w.candidate().unwrap().launch_symbol,
            "gemm_mq4g256v2_fp8_silu_row_b1_bf16"
        );
        let k = new(
            "M=64,N=128,K=256",
            &mut knobs(&[("scale", "k128"), ("epi", "qkv")]),
            None,
        );
        assert!(k.is_err(), "qkv needs three row counts");
        let k = new(
            "M=64+64+64,N=128,K=256",
            &mut knobs(&[("scale", "k128"), ("epi", "qkv")]),
            None,
        )
        .unwrap();
        assert_eq!(
            k.candidate().unwrap().launch_symbol,
            "gemm_mq4g256v2_fp8_qkv_k128_b1"
        );
        for (kv, why) in [
            (vec![("epi", "qkvzagdn")], "gdn"),
            (vec![("scale", "k128"), ("epi", "silu-bf16")], "k128 bf16"),
            (vec![("scale", "both"), ("epi", "silu-bf16")], "both bf16"),
            (vec![("scale", "x")], "scale"),
            (vec![("launch-scale", "k128")], "launch-scale without both"),
        ] {
            assert!(
                new("M=64,N=128,K=256", &mut knobs(&kv), None).is_err(),
                "{why}"
            );
        }
        let both = new(
            "M=64,N=128,K=256",
            &mut knobs(&[("scale", "both"), ("launch-scale", "k128")]),
            None,
        )
        .unwrap();
        let obj = both.candidate().unwrap();
        assert_eq!(obj.exports.len(), 10);
        assert_eq!(obj.launch_symbol, "gemm_mq4g256v2_fp8_set_k128_b1");
        assert!(obj.exports.contains(&obj.launch_symbol));
    }

    #[test]
    fn baseline_is_the_shipped_bundle_for_every_valid_spec() {
        let mut n = 0;
        for (scale, epis) in [
            (
                "row",
                vec![
                    ("set", "M=64"),
                    ("add", "M=64"),
                    ("silu", "M=64"),
                    ("silu-bf16", "M=64"),
                    ("qkv", "M=64+64+64"),
                    ("qkvza", "M=64+64+64+64"),
                ],
            ),
            (
                "k128",
                vec![
                    ("set", "M=64"),
                    ("add", "M=64"),
                    ("silu", "M=64"),
                    ("qkv", "M=64+64+64"),
                    ("qkvza", "M=64+64+64+64"),
                ],
            ),
        ] {
            for (epi, m) in epis {
                let shape = format!("{m},N=128,K=256");
                let w = new(&shape, &mut knobs(&[("scale", scale), ("epi", epi)]), None).unwrap();
                let b = w.baseline().unwrap();
                assert_eq!(
                    b.launch_symbol,
                    format!(
                        "gemm_mq4g256v2_fp8_{}_{scale}_b1{}",
                        if epi == "silu-bf16" { "silu" } else { epi },
                        if epi == "silu-bf16" { "_bf16" } else { "" }
                    )
                );
                assert_eq!(b.source["artifact"], OBJECT);
                assert_eq!(b.source["runtime_selected"], scale == "row");
                assert_eq!(b.source["sha256"].as_str().unwrap().len(), 64);
                n += 1;
            }
        }
        assert_eq!(n, 11);
    }

    #[test]
    fn kernarg_layouts_and_launch_geometry() {
        let b = pack_gemm([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40], 512, 256);
        assert_eq!(b.len(), 96);
        assert_eq!(u64::from_le_bytes(b[64..72].try_into().unwrap()), 9);
        for (off, v) in [(72, 10), (76, 20), (80, 30), (84, 40), (88, 512), (92, 256)] {
            assert_eq!(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()), v);
        }
        let r = pack_repack([1, 2, 3, 4], 5, 6, 7, [8, 9, 10, 11], 512, 1);
        assert_eq!(r.len(), 80);
        assert_eq!(u64::from_le_bytes(r[48..56].try_into().unwrap()), 7);
        for (off, v) in [(56, 8), (60, 9), (64, 10), (68, 11), (72, 512), (76, 1)] {
            assert_eq!(i32::from_le_bytes(r[off..off + 4].try_into().unwrap()), v);
        }
        // fp8_f2_weight_extents: Mp = 512 for Mt = 300.
        assert_eq!(weight_extents(300, 256), (512 * 256, 512 * 2 * 4, 512 * 4));
        // The kernarg layout the generator declares is the same 96 bytes.
        let spec = Spec {
            arch: Arch::Gfx1201,
            act_scale: ActScale::Row,
            epi: Epi::Set,
        };
        assert_eq!(spec.kernargs().size, 96);
    }

    #[test]
    fn e4m3_codes_and_scales() {
        for (x, code) in [
            (0.0f32, 0x00u8),
            (1.0, 0x38),
            (-1.0, 0xb8),
            (0.5, 0x30),
            (448.0, 0x7e),
            (1000.0, 0x7e),
            (1.0 / 512.0, 0x01),
            (1.0 / 64.0, 0x08),
            (1.5, 0x3c),
            // 1.0625 is the tie between 1.0 and 1.125: even code 0x38.
            (1.0625, 0x38),
            (1.1875, 0x3a),
        ] {
            assert_eq!(e4m3(x), code, "{x}");
        }
        assert_eq!(pow2_scale(0.0), 1.0);
        assert_eq!(pow2_scale(224.0), 1.0);
        assert_eq!(pow2_scale(225.0), 2.0);
        assert_eq!(pow2_scale(1.0), 2f32.powi(-7));
        assert_eq!(pow2_scale(1e30), 2f32.powi(9));
        assert_eq!(pow2_scale(1e-30), 2f32.powi(-31));
        let (x8, d) = synth_x(4, 256, 2, 7);
        assert_eq!((x8.len(), d.len()), (4 * 256, 8));
        assert!(d.iter().all(|s| s.log2().fract() == 0.0));
        let (a, _) = synth_x(4, 256, 2, 7);
        assert_eq!(a, x8, "deterministic");
        assert!(x8.iter().all(|&b| b & 0x7f != 0x7f), "no NaN codes");
    }

    #[test]
    fn synthesis_is_deterministic_and_sized() {
        let a = Host::synth(&[8, 4], 4, 256, 1, true);
        let b = Host::synth(&[8, 4], 4, 256, 1, true);
        assert_eq!(a.w, b.w);
        assert_eq!(a.x8, b.x8);
        assert_eq!(a.w[0].len(), 8 * GROUP_BYTES);
        assert_eq!(a.w[1].len(), 4 * GROUP_BYTES);
        assert_eq!(a.d.len(), 4);
        assert_eq!(a.y0.len(), 4 * 8);
    }

    #[test]
    fn there_is_no_cpu_reference() {
        let w = new("M=64,N=128,K=256", &mut knobs(&[]), None).unwrap();
        assert!(w.check(&[]).unwrap_err().contains("no CPU reference"));
        assert!(
            w.max_abs_diff(&[0; 16], &[0; 16]).is_err(),
            "all-zero is vacuous"
        );
        assert_eq!(
            w.max_abs_diff(&[0, 0, 0x80, 0x3f], &[0, 0, 0x80, 0x3f]),
            Ok(0.0)
        );
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/fp8_gemm.rip");

    #[test]
    fn rip_twin_is_identical_to_generator() {
        for (scale, epi) in [
            ("row", "set"),
            ("row", "silu-bf16"),
            ("row", "qkvza"),
            ("k128", "add"),
            ("k128", "qkv"),
        ] {
            let spec = parse_spec(parse_scale(scale).unwrap(), ActScale::Row, epi).unwrap();
            let (_, symbol, info) = compile_rip(TWIN, spec).unwrap();
            assert_eq!(symbol, spec.symbol());
            assert_eq!(info["identical_to_generator"], true, "{scale} {epi}");
            assert_eq!(info["variant"], spec.variant());
            assert_eq!(info["sha256"], sha256_hex(TWIN.as_bytes()));
        }
    }

    #[test]
    fn rip_garbage_and_both_are_refused() {
        let spec = parse_spec(ScaleKnob::One(ActScale::Row), ActScale::Row, "set").unwrap();
        assert!(compile_rip("fn emit(", spec).is_err());
        let path = std::env::temp_dir().join("pm_loop_fp8_rip_test.rip");
        std::fs::write(&path, TWIN).unwrap();
        let r = new(
            "M=64,N=128,K=256",
            &mut knobs(&[("scale", "both")]),
            Some(&path),
        );
        assert!(r.is_err());
        std::fs::remove_file(&path).ok();
    }
}
