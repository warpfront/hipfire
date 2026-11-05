// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `iu4_gemm --epi qkvzagdn` workload: the gfx1201 27B prefill DeltaNet input projection, QKV + Z +
//! beta + alpha in ONE launch whose QKV row tiles also run `gdn_chunk_prep` in the epilogue
//! (`gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1` for token activations, `..._b1s` for the slab
//! activations a slab producer wrote), judged against the SHIPPED object byte for byte. The product
//! launcher is `Gpu::gemm_qkvza_mq4g256v2_iu4_gdn_prepared` (`crates/rdna-compute/src/gemm.rs`); both it
//! and this workload encode the 120-byte kernarg through the `hipfire-kernel-abi` contract
//! `families::iu4_gdn` (`iu4_gemm.qkvzagdn.b1` / `.b1s`), so the judge reproduces the product ABI.
//!
//! # Shape grammar
//! `N=<tokens>[,K=<k>][,M=10240]` (`K` default 5120, `K % 256 == 0`, `1 <= N <= 104857`: the epilogue's
//! raw-row descriptor is `N * 40960` bytes in a 32-bit scalar). `M`, when given, must be the fused
//! QKV rows 10240: the epilogue hard-codes the Qwen3.5 GDN geometry (q/k/v = 16 + 16 + 48 heads of
//! 128; Z 6144, beta/alpha 48 rows of the Z fold, `Gpu::iu4_gdn_fold_rows`). The launch is
//! `[129, ceil(N/128), 1] x [256]`, 20480 B of dynamic LDS, `q_scale = 1/sqrt(128)`, `eps = 1e-6`.
//!
//! # Knobs
//! The `iu4_gemm` generator knobs of the candidate (`fold|tile|cacc|alayout|ring|wait|fold-op|fetch|idle`,
//! `--rip`) and the `base-` overrides of an emitted baseline, exactly as for the other epilogues; with
//! none the baseline is the committed bundle of the candidate's activation layout (`alayout token` ->
//! `_b1`, `slab` -> `_b1s`). `gdn-inputs` (`mixed` default | `realistic`) selects the input profile.
//!
//! # Inputs (seeded, every buffer the epilogue reads)
//! * Weights: seeded MQ4V2 rows for QKV and the Z fold (zero pad rows as the loader's fold); in `mixed`
//!   a few QKV/Z rows hold only code -8, only code +7 (max-magnitude codes) or zero scales.
//! * Activations: `block_i4_128` records per token (token layout; the slab planes are the same bytes
//!   moved, `iu4::to_slab`). `mixed` token classes by `col % 64`: 5 NaN `d` in one K block, 13 +Inf `d`,
//!   21 all-zero rows, 29 max-magnitude codes (-8/+7 with `d = 30000`), 45 a denormal `d`; the others
//!   realistic (`unit * (1 + col % 4)`, 1/64 x30 outliers).
//! * `ConvW` (`[10240][4]` f32) seeded, `mixed`: zero taps on channels `ch % 509 == 7`; `ConvState`
//!   (`[10240][3]` f32, the first token tile's halo) seeded, `mixed`: NaN at `ch % 701 == 3`, +Inf at
//!   `ch % 997 == 5`, -Inf at `ch % 1009 == 7`, zero at `ch % 64 == 11`.
//! * Every output (`Yz Ybeta Yalpha Q K16 V X`) and both conv inputs are per-arm buffers followed by a
//!   256-byte guard; outputs and guards are poisoned with `0xA5` before every launch pair; the payload
//!   of an arm is every buffer with its guard.
//!
//! # Judgement
//! Candidate and shipped baseline must be byte-identical on EVERY section (`z beta alpha q k v y_qkv
//! conv_weight conv_state`) and every guard (`requires_baseline_identity`); the record lists
//! `correctness.identity_by_buffer`. The CPU reference (`iu4::reference`, symmetric fold) checks both
//! arms on sampled rows of Z/beta/alpha (NaN-aware) and on every raw-row element of `y_qkv` the kernel
//! wrote; q/k/v have no CPU reference here (identity to the shipped object only) and are checked to be
//! written; the conv inputs must be unchanged and the guards intact.

use crate::common::{
    dev_err, emitted_object, i4_record, quantize_i4_block, rip_object, shipped_object, Elem, Xs64,
};
use crate::iu4::{reference, sample_rows, synth_weights, to_slab, Host, RefFold, Strs};
use crate::qkvzaset::{
    fold_ref_rows, fold_weights, ALPHA_ROW, BA, BETA_ROW, DIMS, FOLD_ROWS, QKV_M, QKV_SEED, Z_M,
    Z_SEED,
};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use hipfire_kernel_abi::families::iu4_gdn;
use hipfire_kernel_abi::{
    check_export, eval, validate_invocation, Arch as AbiArch, ArgEvidence, Drift, EvidenceSource,
    ExportEvidence, InvocationV1, KernelAbiV1, OperandBinding,
};
use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Spec, Tune};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;

const DEFAULT_K: usize = 5120;
/// `N * 40960` (the raw-row descriptor) must fit a u32: the contract's `X` domain rule.
const MAX_N: usize = 104_857;
const POISON: u8 = 0xA5;
const GUARD: usize = 256;
const SAMPLE_ROWS: usize = 96;
const REF_TOKENS: usize = 64;
const EPS: f32 = 1e-6;

/// Payload sections, in order; the first seven are the launch's outputs, the last two its read-only
/// conv inputs (proved unchanged).
const SECTIONS: [&str; 9] = [
    "z",
    "beta",
    "alpha",
    "q",
    "k",
    "v",
    "y_qkv",
    "conv_weight",
    "conv_state",
];
const OUTPUTS: usize = 7;
const ELEMS: [Elem; 9] = [
    Elem::F32,
    Elem::F32,
    Elem::F32,
    Elem::F16,
    Elem::F16,
    Elem::F16,
    Elem::F32,
    Elem::F32,
    Elem::F32,
];

fn q_scale() -> f32 {
    1.0 / (128f32).sqrt()
}

/// Logical bytes of each payload section at `n` tokens.
fn section_bytes(n: usize) -> [usize; 9] {
    [
        n * Z_M * 4,
        n * BA * 4,
        n * BA * 4,
        n * 16 * 128 * 2,
        n * 16 * 128 * 2,
        n * 48 * 128 * 2,
        n * QKV_M * 4,
        QKV_M * 4 * 4,
        QKV_M * 3 * 4,
    ]
}

/// Byte offset of every section (logical + guard) in a payload and the payload length.
fn section_offsets(n: usize) -> ([usize; 9], usize) {
    let mut off = [0usize; 9];
    let mut at = 0;
    for (o, b) in off.iter_mut().zip(section_bytes(n)) {
        *o = at;
        at += b + GUARD;
    }
    (off, at)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Mixed,
    Realistic,
}

impl Profile {
    fn name(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Realistic => "realistic",
        }
    }
}

/// Activation class of token `col`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenClass {
    Realistic,
    NanD,
    InfD,
    Zero,
    MaxCode,
    DenormD,
}

fn token_class(profile: Profile, col: usize) -> TokenClass {
    if profile == Profile::Realistic {
        return TokenClass::Realistic;
    }
    match col % 64 {
        5 => TokenClass::NanD,
        13 => TokenClass::InfD,
        21 => TokenClass::Zero,
        29 => TokenClass::MaxCode,
        45 => TokenClass::DenormD,
        _ => TokenClass::Realistic,
    }
}

/// One `block_i4_128` record of K128 block `kb` of token `col`.
fn record(profile: Profile, rng: &mut Xs64, kb: usize, col: usize) -> [u8; 72] {
    let class = token_class(profile, col);
    let mut x = [0f32; 128];
    let amp = 1.0 + (col % 4) as f32;
    for v in &mut x {
        *v = rng.unit() * amp;
        if rng.next() % 64 == 0 {
            *v *= 30.0;
        }
    }
    let q = |code: i32| [code; 128];
    match class {
        TokenClass::Realistic => quantize_i4_block(&x),
        TokenClass::Zero => quantize_i4_block(&[0f32; 128]),
        TokenClass::MaxCode => i4_record(30000.0, &q(if kb % 2 == 0 { -8 } else { 7 })),
        TokenClass::DenormD => {
            let mut rec = quantize_i4_block(&x);
            rec[0..4].copy_from_slice(&1e-40f32.to_le_bytes());
            rec
        }
        TokenClass::NanD | TokenClass::InfD => {
            let mut rec = quantize_i4_block(&x);
            let hit = if class == TokenClass::NanD {
                kb % 7 == 3
            } else {
                kb % 11 == 5
            };
            if hit {
                let d = if class == TokenClass::NanD {
                    f32::NAN
                } else {
                    f32::INFINITY
                };
                rec[0..4].copy_from_slice(&d.to_le_bytes());
            }
            rec
        }
    }
}

/// Token-layout records `[K/128][N]`.
fn synth_recs(profile: Profile, n: usize, k: usize) -> Vec<u8> {
    let mut rng = Xs64::new(0x6d6e_5f67_646e_0001);
    let mut recs = vec![0u8; k / 128 * n * 72];
    for kb in 0..k / 128 {
        for col in 0..n {
            let o = (kb * n + col) * 72;
            recs[o..o + 72].copy_from_slice(&record(profile, &mut rng, kb, col));
        }
    }
    recs
}

/// Max-magnitude / zero-scale rows of the generated weights (`mixed`): rows `r % 128 == 9` hold only code
/// -8 (nibble 0), `== 41` only +7 (nibble 15), `== 73` have zero scales.
fn edge_rows(profile: Profile, w: &mut [u8], rows: usize, k: usize) {
    if profile == Profile::Realistic {
        return;
    }
    let row_bytes = k / 256 * 136;
    for r in 0..rows {
        let kind = r % 128;
        if ![9, 41, 73].contains(&kind) {
            continue;
        }
        for g in w[r * row_bytes..(r + 1) * row_bytes].chunks_exact_mut(136) {
            match kind {
                9 => g[8..].fill(0x00),
                41 => g[8..].fill(0xFF),
                _ => g[0..8].fill(0),
            }
        }
    }
}

fn synth_conv(profile: Profile) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Xs64::new(0x6d6e_5f67_646e_0002);
    let mut w: Vec<f32> = (0..QKV_M * 4).map(|_| rng.unit() * 0.6).collect();
    let mut s: Vec<f32> = (0..QKV_M * 3).map(|_| rng.unit()).collect();
    if profile == Profile::Mixed {
        for ch in 0..QKV_M {
            if ch % 509 == 7 {
                w[ch * 4..ch * 4 + 4].fill(0.0);
            }
            if ch % 701 == 3 {
                s[ch * 3] = f32::NAN;
            }
            if ch % 997 == 5 {
                s[ch * 3 + 1] = f32::INFINITY;
            }
            if ch % 1009 == 7 {
                s[ch * 3 + 2] = f32::NEG_INFINITY;
            }
            if ch % 64 == 11 {
                s[ch * 3..ch * 3 + 3].fill(0.0);
            }
        }
    }
    let bytes = |v: Vec<f32>| v.into_iter().flat_map(f32::to_le_bytes).collect();
    (bytes(w), bytes(s))
}

fn parse_shape(s: &str) -> Result<(usize, usize)> {
    let (mut n, mut k) = (None, None);
    for part in s.split(',') {
        let (key, v) = part
            .split_once('=')
            .ok_or_else(|| format!("shape `{s}`: expected N=..[,K=..]"))?;
        let v: usize = v
            .trim()
            .parse()
            .map_err(|e| format!("shape `{s}`: {key}: {e}"))?;
        let slot = match key.trim() {
            "N" => &mut n,
            "K" => &mut k,
            "M" if v == QKV_M => continue,
            "M" => {
                return Err(format!(
                    "shape `{s}`: M={v}: the fused GDN projection is the 27B geometry, M must be {QKV_M}"
                ))
            }
            other => return Err(format!("shape `{s}`: unknown dimension `{other}`")),
        };
        if slot.replace(v).is_some() {
            return Err(format!("shape `{s}`: duplicate {key}"));
        }
    }
    let n = n.ok_or_else(|| format!("shape `{s}`: N is required"))?;
    let k = k.unwrap_or(DEFAULT_K);
    if n == 0 || n > MAX_N || k == 0 || k % 256 != 0 {
        return Err(format!(
            "shape N={n},K={k}: need 1 <= N <= {MAX_N} and K % 256 == 0"
        ));
    }
    Ok((n, k))
}

/// The contract (token `_b1` / slab `_b1s`) of an arm's activation layout.
fn abi_of(spec: Spec) -> &'static KernelAbiV1 {
    match spec.act {
        ALayout::Token => &iu4_gdn::IU4_QKVZAGDN_B1,
        ALayout::Slab => &iu4_gdn::IU4_QKVZAGDN_B1S,
    }
}

/// The explicit kernarg of `spec`'s launch encoded by its `iu4_gdn` contract, and the launch grid.
///
/// `ops` is `(address, allocation bytes)` of `Wqkv Wz Xq Yz Ybeta Yalpha ConvW ConvState Q K16 V X`.
/// The generator's kernarg layout of `spec` (names, offsets, widths, segment) must agree with the
/// contract and its launch geometry (block, dynamic LDS) must be the contract's, else an error before
/// launch.
fn contract_launch(
    spec: Spec,
    k: usize,
    n: usize,
    ops: [(u64, u64); 12],
) -> Result<(Vec<u8>, [u32; 3], &'static KernelAbiV1)> {
    let abi = abi_of(spec);
    let export = &abi.exports[0];
    let layout = spec.kernargs();
    let evidence = ExportEvidence {
        source: EvidenceSource::Generator,
        arch: Some(AbiArch::Gfx1201),
        symbol: export.symbol.to_owned(),
        args: layout
            .args
            .iter()
            .map(|a| ArgEvidence {
                name: a.name.clone(),
                offset: a.offset,
                size: a.size,
                value_kind: a.value_kind.clone(),
                access: a.actual_access.clone(),
            })
            .collect(),
        segment_bytes: Some(layout.size),
        segment_align: None,
        max_flat_workgroup_size: None,
        threads: Some(spec.tile.threads()),
        dynamic_lds: Some(spec.lds_launch()),
        group_segment_fixed: None,
        wave: None,
        object_sha256: None,
    };
    let drift: Vec<String> = check_export(abi, export, &evidence)
        .into_iter()
        .filter(|d| {
            matches!(
                d,
                Drift::MissingArg { .. }
                    | Drift::ExtraArg { .. }
                    | Drift::ArgName { .. }
                    | Drift::ArgWidth { .. }
                    | Drift::ArgKind { .. }
                    | Drift::ArgAccess { .. }
                    | Drift::Segment { .. }
                    | Drift::Threads { .. }
                    | Drift::DynamicLds { .. }
            )
        })
        .map(|d| d.to_string())
        .collect();
    if !drift.is_empty() {
        return Err(format!(
            "{}: kernarg/launch differs from contract {} ({}): {}",
            spec.variant_name(),
            abi.id,
            export.symbol,
            drift.join("; ")
        ));
    }
    let params = [
        DIMS[0] as u64,
        DIMS[1] as u64,
        DIMS[2] as u64,
        DIMS[3] as u64,
        k as u64,
        n as u64,
        u64::from(q_scale().to_bits()),
        u64::from(EPS.to_bits()),
    ];
    let mut grid = [0u32; 3];
    for (g, e) in grid.iter_mut().zip(&export.grid) {
        let v = eval(e, abi, export, &params).map_err(|e| format!("{}: {e}", abi.id))?;
        *g = u32::try_from(v).map_err(|_| format!("{}: grid {v} exceeds u32", abi.id))?;
    }
    let binds: [OperandBinding; 12] = std::array::from_fn(|i| OperandBinding {
        address: ops[i].0,
        bytes: Some(ops[i].1),
        format: abi.operands[i].format,
    });
    let v = validate_invocation(
        abi,
        &InvocationV1 {
            arch: AbiArch::Gfx1201,
            symbol: export.symbol,
            params: &params,
            operands: &binds,
            grid,
            block: export.block,
            dynamic_lds: export.dynamic_lds,
        },
    )
    .map_err(|e| format!("{}: {e}", abi.id))?;
    Ok((v.explicit().to_vec(), grid, abi))
}

/// Per-arm device buffers.
struct ArmBufs {
    /// `z beta alpha q k v y_qkv`, each `logical + GUARD`.
    out: [hip_bridge::DeviceBuffer; OUTPUTS],
    conv_w: hip_bridge::DeviceBuffer,
    conv_state: hip_bridge::DeviceBuffer,
}

struct Dev {
    w_qkv: hip_bridge::DeviceBuffer,
    w_z: hip_bridge::DeviceBuffer,
    xq_token: hip_bridge::DeviceBuffer,
    xq_slab: hip_bridge::DeviceBuffer,
    arms: [ArmBufs; 2],
}

struct Inputs {
    /// QKV rows and the Z fold (with the shared activations in `recs`).
    hq: Host,
    hz: Host,
    conv_w: Vec<u8>,
    conv_state: Vec<u8>,
}

/// CPU reference rows: `(tokens, qkv rows x tokens, z-fold rows x tokens)`.
struct Refs {
    tokens: Vec<usize>,
    qkv_rows: Vec<usize>,
    qkv: Vec<f32>,
    z: Vec<f32>,
}

pub struct Qkvzagdn {
    n: usize,
    k: usize,
    profile: Profile,
    cand: Spec,
    cand_strs: Strs,
    base: Spec,
    base_strs: Strs,
    base_emitted: bool,
    rip: Option<(String, String)>,
    inputs: Option<Inputs>,
    dev: Option<Dev>,
    refs: OnceCell<Refs>,
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

/// Build the workload from the `iu4_gemm` knobs `iu4::new_with_rip` already resolved.
#[allow(clippy::too_many_arguments)]
pub fn build(
    shape: &str,
    knobs: &mut Knobs,
    cand: Spec,
    cand_strs: Strs,
    base: Spec,
    base_strs: Strs,
    base_emitted: bool,
    rip: Option<(String, String)>,
) -> Result<Box<dyn Workload>> {
    let (n, k) = parse_shape(shape)?;
    let profile = match knobs.take_or("gdn-inputs", "mixed").as_str() {
        "mixed" => Profile::Mixed,
        "realistic" => Profile::Realistic,
        o => return Err(format!("--gdn-inputs {o}: expected mixed|realistic")),
    };
    if cand.mma != iu4_gemm::Mma::Dense || base.mma != iu4_gemm::Mma::Dense {
        return Err("epi qkvzagdn: the fused projection is dense (no --mma / --base-mma)".into());
    }
    // Both arms' generator layouts and launch geometry must be the contract's: refuse GPU-free.
    for spec in [cand, base] {
        let ops = std::array::from_fn(|i| (0x1000_0000 + 0x10_0000 * i as u64, 1u64 << 40));
        contract_launch(spec, k, n, ops)?;
    }
    Ok(Box::new(Qkvzagdn {
        n,
        k,
        profile,
        cand,
        cand_strs,
        base,
        base_strs,
        base_emitted,
        rip,
        inputs: None,
        dev: None,
        refs: OnceCell::new(),
    }))
}

/// Tokens the CPU reference covers: all of them up to `REF_TOKENS`, else the first and last eight,
/// every edge-class token that fits, and an even spread.
fn ref_tokens(profile: Profile, n: usize) -> Vec<usize> {
    if n <= REF_TOKENS {
        return (0..n).collect();
    }
    let mut s = std::collections::BTreeSet::new();
    s.extend(0..8);
    s.extend(n - 8..n);
    for t in 0..n {
        if s.len() >= REF_TOKENS - 16 {
            break;
        }
        if token_class(profile, t) != TokenClass::Realistic {
            s.insert(t);
        }
    }
    let spread = REF_TOKENS - s.len();
    for i in 0..spread {
        s.insert(i * (n - 1) / spread.max(1));
    }
    s.into_iter().collect()
}

/// `a`, `b` equal as bits, or both NaN (the CPU reference does not model NaN payloads).
fn same(got: f32, exp: f32) -> bool {
    got.to_bits() == exp.to_bits() || (got.is_nan() && exp.is_nan())
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
}

impl Qkvzagdn {
    fn spec_of(&self, arm: Arm) -> Spec {
        match arm {
            Arm::Baseline => self.base,
            Arm::Candidate => self.cand,
        }
    }

    fn synth_inputs(&mut self) {
        let mut w_qkv = synth_weights(QKV_M, self.k, QKV_SEED);
        edge_rows(self.profile, &mut w_qkv, QKV_M, self.k);
        let mut w_z = fold_weights(self.k, Z_SEED);
        edge_rows(self.profile, &mut w_z, Z_M, self.k);
        let recs = synth_recs(self.profile, self.n, self.k);
        let (conv_w, conv_state) = synth_conv(self.profile);
        let host = |w: Vec<u8>, recs: Vec<u8>| Host {
            w,
            w_up: Vec::new(),
            recs,
            y0: Vec::new(),
        };
        self.inputs = Some(Inputs {
            hq: host(w_qkv, recs.clone()),
            hz: host(w_z, recs),
            conv_w,
            conv_state,
        });
    }

    fn reference_rows(&self) -> Result<&Refs> {
        let inp = self.inputs.as_ref().ok_or("check before setup")?;
        Ok(self.refs.get_or_init(|| {
            let tokens = ref_tokens(self.profile, self.n);
            let sub = |h: &Host| -> Host {
                let mut recs = Vec::with_capacity(self.k / 128 * tokens.len() * 72);
                for kb in 0..self.k / 128 {
                    for &t in &tokens {
                        recs.extend_from_slice(&h.recs[(kb * self.n + t) * 72..][..72]);
                    }
                }
                Host {
                    w: h.w.clone(),
                    w_up: Vec::new(),
                    recs,
                    y0: Vec::new(),
                }
            };
            let qkv_rows = sample_rows(QKV_M, SAMPLE_ROWS);
            let (hq, hz) = (sub(&inp.hq), sub(&inp.hz));
            let nt = tokens.len();
            let qkv = reference(&hq, QKV_M, nt, self.k, false, &qkv_rows, RefFold::Fma);
            let z = reference(
                &hz,
                FOLD_ROWS,
                nt,
                self.k,
                false,
                &fold_ref_rows(),
                RefFold::Fma,
            );
            Refs {
                tokens,
                qkv_rows,
                qkv,
                z,
            }
        }))
    }

    /// Slice of `payload` for section `i`: `(logical, guard)`.
    fn section<'a>(&self, payload: &'a [u8], i: usize) -> (&'a [u8], &'a [u8]) {
        let (off, _) = section_offsets(self.n);
        let len = section_bytes(self.n)[i];
        (
            &payload[off[i]..off[i] + len],
            &payload[off[i] + len..off[i] + len + GUARD],
        )
    }
}

impl Workload for Qkvzagdn {
    fn kernel(&self) -> &'static str {
        "iu4_gemm"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        json!({
            "M": QKV_M, "N": self.n, "K": self.k, "dims": DIMS, "fold_rows": FOLD_ROWS,
            "launch_rows": QKV_M + FOLD_ROWS, "epi": "qkvzagdn",
            "kernarg_bytes": iu4_gdn::EXPLICIT_BYTES,
            "q_scale_bits": q_scale().to_bits(), "eps_bits": EPS.to_bits(),
            "gdn_inputs": self.profile.name(),
        })
    }
    fn record_extra(&self) -> Vec<(String, Value)> {
        let abi = abi_of(self.cand);
        vec![(
            "qkvzagdn".to_owned(),
            json!({
                "contract": {"id": abi.id, "version": abi.version, "digest": abi.digest(),
                             "explicit_bytes": abi.explicit_bytes, "symbol": abi.exports[0].symbol},
                "baseline_contract": {"id": abi_of(self.base).id, "digest": abi_of(self.base).digest()},
                "sections": SECTIONS, "guard_bytes": GUARD, "poison": format!("0x{POISON:02x}"),
                "inputs": {
                    "profile": self.profile.name(),
                    "edge_token_classes": "col%64: 5 NaN d, 13 +Inf d, 21 zero, 29 max-magnitude codes, 45 denormal d",
                    "edge_conv": "ConvState NaN ch%701==3, +Inf ch%997==5, -Inf ch%1009==7, zero ch%64==11; ConvW zero ch%509==7",
                    "edge_weight_rows": "r%128==9 all code -8, ==41 all code +7, ==73 zero scales",
                },
                "reference": {"tokens": ref_tokens(self.profile, self.n).len(), "sample_rows": SAMPLE_ROWS},
            }),
        )]
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.cand_strs.json();
        v["variant"] = json!(self.cand.variant_name());
        v
    }

    fn candidate(&self) -> Result<Object> {
        let e = iu4_gemm::emit(self.cand)?;
        let symbol = self.cand.symbol();
        if let Some(rip) = &self.rip {
            let base = Spec {
                tune: Tune::default(),
                ..self.cand
            };
            return rip_object(
                rip,
                self.cand.arch,
                &base.variant_name(),
                &symbol,
                &e.s_text,
                self.cand_strs.json(),
            );
        }
        emitted_object(
            &e.s_text,
            self.cand.arch,
            &symbol,
            std::slice::from_ref(&symbol),
            "peacemaker_isa::kernels::iu4_gemm::emit + native::assemble (peacemaker-isa emit --co)",
            self.cand_strs.json(),
        )
    }

    fn baseline(&self) -> Result<Object> {
        let symbol = self.base.symbol();
        if self.base_emitted {
            let e = iu4_gemm::emit(self.base)?;
            return emitted_object(
                &e.s_text,
                self.base.arch,
                &symbol,
                std::slice::from_ref(&symbol),
                "peacemaker_isa::kernels::iu4_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                self.base_strs.json(),
            );
        }
        let (bytes, file) = match self.base.act {
            ALayout::Token => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco",
            ),
            ALayout::Slab => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco",
            ),
        };
        shipped_object(
            bytes,
            file,
            "crates/rdna-compute/src/gemm.rs Gpu::gemm_qkvza_mq4g256v2_iu4_gdn_prepared (symbol `_b1s` when \
             int4_mmq_slab_at(xq), else `_b1`; module via g12_iu4_b1_image / g12_iu4_b1s_image)",
            &symbol,
            &[symbol.as_str()],
            self.base_strs.json(),
        )
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        self.synth_inputs();
        let inp = self.inputs.as_ref().expect("just synthesized");
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            if !b.is_empty() {
                gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            }
            Ok(d)
        };
        let alloc = |len: usize| gpu.hip.malloc((len + GUARD).max(16)).map_err(dev_err);
        let sb = section_bytes(self.n);
        let arm = || -> Result<ArmBufs> {
            Ok(ArmBufs {
                out: [
                    alloc(sb[0])?,
                    alloc(sb[1])?,
                    alloc(sb[2])?,
                    alloc(sb[3])?,
                    alloc(sb[4])?,
                    alloc(sb[5])?,
                    alloc(sb[6])?,
                ],
                conv_w: alloc(sb[7])?,
                conv_state: alloc(sb[8])?,
            })
        };
        let dev = Dev {
            w_qkv: up(&inp.hq.w)?,
            w_z: up(&inp.hz.w)?,
            xq_token: up(&inp.hq.recs)?,
            xq_slab: up(&to_slab(&inp.hq.recs, self.n))?,
            arms: [arm()?, arm()?],
        };
        self.dev = Some(dev);
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        // Both arms' launches must be admitted by the contract (real addresses and sizes).
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.try_launch(arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        self.try_launch(arm)
            .unwrap_or_else(|e| panic!("qkvzagdn launch: {e}"))
    }

    fn null_object(&self) -> Result<Option<Object>> {
        let e = iu4_gemm::emit(self.base)?;
        crate::common::null_object(
            &e.s_text,
            self.base.arch,
            &self.base.symbol(),
            "the baseline spec's emit with the body replaced by s_endpgm (descriptor unchanged)",
        )
        .map(Some)
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let inp = self.inputs.as_ref().ok_or("reset before setup")?;
        let b = &dev.arms[arm_index(arm)];
        for buf in &b.out {
            gpu.hip
                .memset(buf, i32::from(POISON), buf.size())
                .map_err(dev_err)?;
        }
        // The conv inputs are restored to the host bytes with a poisoned guard.
        for (buf, host) in [(&b.conv_w, &inp.conv_w), (&b.conv_state, &inp.conv_state)] {
            let mut bytes = host.clone();
            bytes.resize(buf.size(), POISON);
            gpu.hip.memcpy_htod(buf, &bytes).map_err(dev_err)?;
        }
        Ok(())
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        let b = &dev.arms[arm_index(arm)];
        let (off, total) = section_offsets(self.n);
        let sb = section_bytes(self.n);
        let mut out = vec![0u8; total];
        let bufs = b.out.iter().chain([&b.conv_w, &b.conv_state]);
        for (i, buf) in bufs.enumerate() {
            let mut v = vec![0u8; buf.size()];
            gpu.hip.memcpy_dtoh(&mut v, buf).map_err(dev_err)?;
            let want = sb[i] + GUARD;
            // A 16-byte minimum allocation never applies at these sizes; the payload is exact.
            out[off[i]..off[i] + want].copy_from_slice(&v[..want]);
        }
        Ok(out)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        self.check_arm(Arm::Candidate, out)
    }

    fn check_arm(&self, _arm: Arm, out: &[u8]) -> Result<RefReport> {
        let (_, total) = section_offsets(self.n);
        if out.len() != total {
            return Err(format!("output is {} bytes, expected {total}", out.len()));
        }
        let inp = self.inputs.as_ref().ok_or("check before setup")?;
        let refs = self.reference_rows()?;
        let nt = refs.tokens.len();

        let guards_ok = (0..9).all(|i| self.section(out, i).1.iter().all(|&b| b == POISON));
        let conv_ok = self.section(out, 7).0 == &inp.conv_w[..]
            && self.section(out, 8).0 == &inp.conv_state[..];
        let (mut mismatches, mut worst) = (0usize, 0f64);
        let mut cmp = |got: f32, exp: f32| {
            let ok = same(got, exp);
            mismatches += usize::from(!ok);
            if !ok {
                worst = worst.max(if got.is_nan() != exp.is_nan() {
                    f64::INFINITY
                } else {
                    (f64::from(got) - f64::from(exp)).abs()
                });
            }
        };
        // Z / beta / alpha rows of the fold.
        let (z, beta, alpha) = (
            self.section(out, 0).0,
            self.section(out, 1).0,
            self.section(out, 2).0,
        );
        for (ri, &r) in fold_ref_rows().iter().enumerate() {
            for (j, &t) in refs.tokens.iter().enumerate() {
                let got = if r < Z_M {
                    f32_at(z, t * Z_M + r)
                } else if r < BETA_ROW + BA {
                    f32_at(beta, t * BA + (r - BETA_ROW))
                } else {
                    f32_at(alpha, t * BA + (r - ALPHA_ROW))
                };
                cmp(got, refs.z[ri * nt + j]);
            }
        }
        // The raw QKV rows the kernel wrote.
        let x = self.section(out, 6).0;
        let poison_word = f32::from_bits(u32::from_le_bytes([POISON; 4]));
        let mut x_written = 0usize;
        for (ri, &r) in refs.qkv_rows.iter().enumerate() {
            for (j, &t) in refs.tokens.iter().enumerate() {
                let got = f32_at(x, t * QKV_M + r);
                if got.to_bits() == poison_word.to_bits() {
                    continue;
                }
                x_written += 1;
                cmp(got, refs.qkv[ri * nt + j]);
            }
        }
        // q/k/v: no CPU reference here; they must have been written.
        let written = |i: usize| {
            self.section(out, i)
                .0
                .chunks_exact(2)
                .filter(|c| c != &[POISON, POISON])
                .count()
        };
        let (wq, wk, wv) = (written(3), written(4), written(5));
        let pass =
            guards_ok && conv_ok && mismatches == 0 && x_written > 0 && wq > 0 && wk > 0 && wv > 0;
        Ok(RefReport {
            reference: format!(
                "cpu_iu4_symfold_rn(z rows=sample:192+all beta/alpha (qkvzaset::fold_ref_rows), y_qkv rows=sample:{SAMPLE_ROWS} written raw elements, \
                 tokens={nt}, fold=fma, NaN-aware, mismatching_elements={mismatches}, y_qkv_written={x_written}, \
                 qkv_written_halfwords={wq}/{wk}/{wv}, guards_intact={guards_ok}, conv_inputs_unchanged={conv_ok}; \
                 q/k/v identity-to-shipped only)"
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass,
        })
    }

    fn identity_detail(&self, base: &[u8], cand: &[u8]) -> Value {
        let (_, total) = section_offsets(self.n);
        if base.len() != total || cand.len() != total {
            return json!({"error": format!("payload lengths {} / {}, expected {total}", base.len(), cand.len())});
        }
        let rows: Vec<Value> = (0..9)
            .map(|i| {
                let (bl, bg) = self.section(base, i);
                let (cl, cg) = self.section(cand, i);
                let first = bl.iter().zip(cl).position(|(a, b)| a != b);
                let diffs = bl.iter().zip(cl).filter(|(a, b)| a != b).count();
                json!({
                    "buffer": SECTIONS[i],
                    "bytes": bl.len(),
                    "identical": bl == cl,
                    "differing_bytes": diffs,
                    "first_differing_byte": first,
                    "guard_identical": bg == cg,
                    "guard_intact": cg.iter().all(|&b| b == POISON),
                    "max_abs_diff": crate::common::max_diff(ELEMS[i], bl, cl),
                })
            })
            .collect();
        json!({"buffers": rows, "all_identical": base == cand})
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        let (_, total) = section_offsets(self.n);
        if a.len() != total || b.len() != total {
            return Err(format!(
                "length mismatch {} vs {} (want {total})",
                a.len(),
                b.len()
            ));
        }
        let mut worst = 0f64;
        for i in 0..9 {
            let (la, ga) = self.section(a, i);
            let (lb, gb) = self.section(b, i);
            worst = worst.max(crate::common::max_diff(ELEMS[i], la, lb));
            if ga != gb {
                worst = f64::INFINITY;
            }
        }
        Ok(worst)
    }

    fn requires_baseline_identity(&self) -> bool {
        true
    }

    fn emu(&self) -> Value {
        Value::String(
            "refused (the fused qkvzagdn projection has no emulator harness in pm_loop)".into(),
        )
    }
}

impl Qkvzagdn {
    fn try_launch(&self, arm: Arm) -> Result<Launch> {
        let dev = self.dev.as_ref().ok_or("launch before setup")?;
        let spec = self.spec_of(arm);
        let b = &dev.arms[arm_index(arm)];
        let xq = match spec.act {
            ALayout::Token => &dev.xq_token,
            ALayout::Slab => &dev.xq_slab,
        };
        let op = |d: &hip_bridge::DeviceBuffer| (d.as_ptr() as u64, d.size() as u64);
        let ops = [
            op(&dev.w_qkv),
            op(&dev.w_z),
            op(xq),
            op(&b.out[0]),
            op(&b.out[1]),
            op(&b.out[2]),
            op(&b.conv_w),
            op(&b.conv_state),
            op(&b.out[3]),
            op(&b.out[4]),
            op(&b.out[5]),
            op(&b.out[6]),
        ];
        let (kernarg, grid, abi) = contract_launch(spec, self.k, self.n, ops)?;
        let export = &abi.exports[0];
        Ok(Launch {
            kernarg,
            grid,
            block: export.block,
            lds: export.dynamic_lds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::sha256_hex;

    fn spec(slab: bool) -> Spec {
        Spec {
            act: if slab { ALayout::Slab } else { ALayout::Token },
            ..Spec::control(iu4_gemm::Epi::QkvzaGdn)
        }
    }

    fn strs(s: Spec) -> Strs {
        Strs {
            fold: "k128".into(),
            tile: "128x128x8".into(),
            cacc: "1".into(),
            epi: "qkvzagdn".into(),
            alayout: if s.act == ALayout::Slab {
                "slab"
            } else {
                "token"
            }
            .into(),
            mma: "dense".into(),
            ring: "2".into(),
            wait: "after".into(),
            fold_op: "fma".into(),
            fetch: "split".into(),
            idle: "off".into(),
        }
    }

    fn make(shape: &str, slab: bool, kv: &[(&str, &str)]) -> Result<Box<dyn Workload>> {
        let mut k = Knobs::default();
        for (a, b) in kv {
            k.0.insert((*a).into(), (*b).into());
        }
        let s = spec(slab);
        let w = build(shape, &mut k, s, strs(s), s, strs(s), false, None)?;
        k.finish()?;
        Ok(w)
    }

    #[test]
    fn shape_grammar() {
        assert_eq!(parse_shape("N=128").unwrap(), (128, 5120));
        assert_eq!(parse_shape("K=2560,N=1,M=10240").unwrap(), (1, 2560));
        for bad in [
            "",
            "K=5120",
            "N=0",
            "N=104858",
            "N=8,K=100",
            "N=8,M=4096",
            "N=8,N=9",
            "N=8,Q=1",
        ] {
            assert!(parse_shape(bad).is_err(), "{bad}");
        }
        assert!(parse_shape("N=104857").is_ok());
        assert!(make("N=8", false, &[("gdn-inputs", "weird")]).is_err());
        assert!(make("N=8", false, &[("bogus", "1")]).is_err());
        let w = make("N=512", true, &[("gdn-inputs", "realistic")]).unwrap();
        assert_eq!(w.shape_json()["gdn_inputs"], "realistic");
        assert_eq!(w.shape_json()["kernarg_bytes"], 120);
        assert_eq!(w.shape_json()["launch_rows"], 16512);
    }

    /// The contract encoding of both layouts is the generator's 120-byte layout and the product's.
    #[test]
    fn kernarg_is_the_contract_encoding_of_the_generator_layout() {
        let q = q_scale();
        for slab in [false, true] {
            let ops: [(u64, u64); 12] =
                std::array::from_fn(|i| (0x1000_0000 + 0x10_0000 * i as u64, 1 << 40));
            let (b, grid, abi) = contract_launch(spec(slab), 5120, 300, ops).unwrap();
            assert_eq!(
                abi.id,
                if slab {
                    "iu4_gemm.qkvzagdn.b1s"
                } else {
                    "iu4_gemm.qkvzagdn.b1"
                }
            );
            assert_eq!(b.len(), 120);
            assert_eq!(grid, [129, 3, 1]);
            let ptr = |off: usize| u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
            let int = |off: usize| i32::from_le_bytes(b[off..off + 4].try_into().unwrap());
            // gemm_qkvza_mq4g256v2_iu4_gdn_prepared push order: ptrs(Wqkv, Wz), Xq, outs(Yz, Ybeta,
            // Yalpha), ints(M0, Mz, K, N), gdn ptrs(ConvW, ConvState, Q, K16, V, X), scales.
            for (off, i) in [(0, 0), (8, 1), (16, 2), (24, 3), (32, 4), (40, 5)] {
                assert_eq!(ptr(off), ops[i].0, "ptr@{off}");
            }
            for (off, v) in [(48, 10240), (52, 6144), (56, 5120), (60, 300)] {
                assert_eq!(int(off), v, "int@{off}");
            }
            for (off, i) in [(64, 6), (72, 7), (80, 8), (88, 9), (96, 10), (104, 11)] {
                assert_eq!(ptr(off), ops[i].0, "gdn@{off}");
            }
            assert_eq!(&b[112..116], &q.to_le_bytes());
            assert_eq!(&b[116..120], &EPS.to_le_bytes());
        }
    }

    #[test]
    fn symbols_name_the_shipped_exports_and_both_are_in_the_bundles() {
        for (slab, sym) in [
            (false, "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1"),
            (true, "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s"),
        ] {
            let w = make("N=64", slab, &[]).unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, sym);
            assert_eq!(w.candidate().unwrap().launch_symbol, sym);
            let want = if slab {
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S
            } else {
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1
            };
            assert_eq!(b.source["sha256"], sha256_hex(want));
            assert!(w.requires_baseline_identity());
        }
    }

    /// GPU-free identity: the generator's emit of the shipped schedule has the shipped export's code.
    #[test]
    fn default_candidate_has_the_shipped_exports_code() {
        let code = |o: &Object| {
            let p = crate::lift(&o.bytes).unwrap();
            let r = crate::resources(&p, std::slice::from_ref(&o.launch_symbol)).unwrap();
            (
                r[0]["code_sha256"].as_str().unwrap().to_owned(),
                r[0]["kernarg_bytes"].as_u64().unwrap(),
            )
        };
        for slab in [false, true] {
            let w = make("N=64", slab, &[]).unwrap();
            let (base, cand) = (code(&w.baseline().unwrap()), code(&w.candidate().unwrap()));
            assert_eq!(base, cand, "slab={slab}");
            assert_eq!(base.1, 120 + 256);
        }
    }

    #[test]
    fn input_profiles_are_deterministic_and_carry_the_edge_cases() {
        let (n, k) = (128usize, 256usize);
        let recs = synth_recs(Profile::Mixed, n, k);
        assert_eq!(recs, synth_recs(Profile::Mixed, n, k));
        assert_ne!(recs, synth_recs(Profile::Realistic, n, k));
        let d = |kb: usize, col: usize| {
            f32::from_le_bytes(recs[(kb * n + col) * 72..][..4].try_into().unwrap())
        };
        // K=256: blocks 0 and 1. NaN d needs kb%7==3, so only a longer K carries it.
        assert!(!d(0, 5).is_nan());
        let recs = synth_recs(Profile::Mixed, 64, 5120);
        let d = |kb: usize, col: usize| {
            f32::from_le_bytes(recs[(kb * 64 + col) * 72..][..4].try_into().unwrap())
        };
        assert!(d(3, 5).is_nan());
        assert!(d(5, 13).is_infinite() && d(5, 13) > 0.0);
        assert_eq!(d(7, 21), 0.0);
        assert_eq!(d(2, 29), 30000.0);
        assert!(d(0, 45) > 0.0 && d(0, 45) < f32::MIN_POSITIVE);
        assert!(d(0, 6).is_finite() && d(0, 6) > 0.0);
        let (w, s) = synth_conv(Profile::Mixed);
        let sv =
            |c: usize, j: usize| f32::from_le_bytes(s[(c * 3 + j) * 4..][..4].try_into().unwrap());
        assert!(sv(3, 0).is_nan() && sv(5, 1).is_infinite() && sv(7, 2) == f32::NEG_INFINITY);
        assert_eq!(sv(11, 1), 0.0);
        assert!(w[7 * 16..8 * 16].iter().all(|b| *b == 0));
        let (_, sr) = synth_conv(Profile::Realistic);
        assert!(sr
            .chunks_exact(4)
            .all(|c| f32::from_le_bytes(c.try_into().unwrap()).is_finite()));
        // Max-magnitude / zero-scale weight rows.
        let mut w = synth_weights(128, 256, 7);
        edge_rows(Profile::Mixed, &mut w, 128, 256);
        let row = |r: usize| &w[r * 136..(r + 1) * 136];
        assert!(row(9)[8..].iter().all(|b| *b == 0x00));
        assert!(row(41)[8..].iter().all(|b| *b == 0xff));
        assert!(row(73)[..8].iter().all(|b| *b == 0));
    }

    #[test]
    fn payload_layout_guards_and_identity_detail() {
        let s = spec(false);
        let q = Qkvzagdn {
            n: 64,
            k: 5120,
            profile: Profile::Mixed,
            cand: s,
            cand_strs: strs(s),
            base: s,
            base_strs: strs(s),
            base_emitted: false,
            rip: None,
            inputs: None,
            dev: None,
            refs: OnceCell::new(),
        };
        let (off, total) = section_offsets(64);
        assert_eq!(off[0], 0);
        assert_eq!(total, section_bytes(64).iter().sum::<usize>() + 9 * GUARD);
        let base = vec![POISON; total];
        let mut cand = base.clone();
        let d = q.identity_detail(&base, &base);
        assert_eq!(d["all_identical"], true);
        assert!(d["buffers"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["identical"] == true));
        // One flipped byte in q (section 3) and one in a guard (v, section 5).
        cand[off[3] + 17] ^= 1;
        cand[off[5] + section_bytes(64)[5] + 3] = 0;
        let d = q.identity_detail(&base, &cand);
        assert_eq!(d["all_identical"], false);
        let rows = d["buffers"].as_array().unwrap();
        assert_eq!(rows[3]["buffer"], "q");
        assert_eq!(rows[3]["identical"], false);
        assert_eq!(rows[3]["first_differing_byte"], 17);
        assert_eq!(rows[5]["identical"], true);
        assert_eq!(rows[5]["guard_intact"], false);
        assert_eq!(rows[0]["identical"], true);
        assert!(q.max_abs_diff(&base, &cand).unwrap().is_infinite());
        assert_eq!(q.max_abs_diff(&base, &base).unwrap(), 0.0);
        assert!(q.max_abs_diff(&base, &cand[1..]).is_err());
    }

    #[test]
    fn reference_tokens_cover_the_edge_classes() {
        assert_eq!(ref_tokens(Profile::Mixed, 40), (0..40).collect::<Vec<_>>());
        let t = ref_tokens(Profile::Mixed, 8192);
        assert!(
            t.len() <= REF_TOKENS && t.len() >= REF_TOKENS - 8,
            "{}",
            t.len()
        );
        assert!(t.windows(2).all(|w| w[0] < w[1]));
        assert!(t.contains(&0) && t.contains(&8191));
        for class in [5, 13, 21, 29, 45] {
            assert!(t.contains(&class), "edge token {class}");
        }
    }

    #[test]
    fn null_object_is_the_shipped_descriptor_with_an_empty_body() {
        let w = make("N=64", true, &[]).unwrap();
        let (base, null) = (w.baseline().unwrap(), w.null_object().unwrap().unwrap());
        assert_eq!(null.launch_symbol, base.launch_symbol);
    }
}
