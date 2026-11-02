// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `qkvzaset` workload: the gfx1201 27B verify GDN input projection in ONE launch, the fused
//! QKV + Z|beta|alpha SET export of the slab bundle
//! (`gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s`, and its small-N `..._b1s_v64` twin) against the
//! shipped object. The product launcher is `Gpu::gemm_qkvza_set_g12_iu4_prepared`
//! (`crates/rdna-compute/src/gemm.rs`): admitted for `0 < n <= 128` and the 27B GDN dims
//! `[10240, 6144, 48, 48]`, symbol `_v64` unless `HIPFIRE_VERIFY_V64=0`.
//!
//! # Shape grammar
//! `N=<tokens>[,K=<k>]` (`K` default 5120, `K % 256 == 0`, `1 <= N <= 128`). The row counts are the
//! product's (QKV 10240, Z 6144, beta 48, alpha 48); the Z fold has `Z + 128` rows: Z in rows
//! `0..6144`, beta in the first 48 rows of the extra tile, alpha at its rows `64..112`, zero rows
//! elsewhere (`Gpu::iu4_gdn_fold_rows`). The launch is `[129, ceil(N/128), 1] x [256]`, 20480 B of
//! dynamic LDS.
//!
//! # Kernarg (120 bytes; `Spec::kernargs`, packed like `gemm_qkvza_set_g12_iu4_prepared`)
//! `Wqkv@0 Wz@8 Xq@16 Yz@24 Ybeta@32 Yalpha@40` (pointers), `M0@48 Mz@52 K@56 N@60` (i32), then six
//! GDN pointer slots `ConvW@64 ConvState@72 Q@80 K16@88 V@96 X@104`: the five GDN slots are never
//! read (the product passes `y_z`), `X` is the QKV output `Y[token * 10240 + row]`; `QScale@112`
//! and `Eps@116` (f32, 0).
//!
//! # Knobs
//! `variant` (`v64` default | `b1s`): which shipped export is the baseline and the candidate's
//! default schedule (`iu4_gemm::verify64_tune(QkvzaGdn, true)` for `v64`; the plain
//! `QkvStore::Set` tune for `b1s`). `wait` (`after|mid|before|window`) and `idle`
//! (`off|skip|skiphi`) override the candidate's schedule. The default candidate is the generator's
//! own emit of the shipped export: `identical_to_baseline` must hold (the kernel is a SET, so
//! `requires_baseline_identity`). Candidate-only: no `--rip`.
//!
//! # Reference
//! The shipped SET numerics (`iu4::reference`, symmetric fold) over sampled rows of each section:
//! QKV and Z rows `sample_rows(_, 192)` plus every beta (`6144..6192`) and alpha (`6208..6256`) row
//! of the fold; every token column. Outputs are compared bit for bit.
//!
//! # `--pool` / `--flush-l2` / `--flush`
//! Cold-cache timing (`support/cold.rs`): a pool of distinct seeded `(Wqkv, Wz fold)` sets whose
//! total exceeds 192 MiB, and optionally a device copy after every timed dispatch (2 x L2 per buffer
//! for `--flush-l2`, 2 x 96 MiB for `--flush`). Set 0 is the
//! correctness set the CPU reference reads; the arms are also compared on every other pool member.

use crate::common::{dev_err, put_f32, put_i32, put_ptr, sha256_hex};
use crate::iu4::{
    max_diff, pool_seed, reference, sample_rows, synth_weights, to_slab, Host, RefFold,
};
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use peacemaker_isa::kernels::iu4_gemm::{self, verify64_tune, ALayout, Epi, QkvStore, Spec, Tune};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::cell::OnceCell;

const GROUP_BYTES: usize = 136;
/// The 27B GDN dims the product admits: `[qkv_m, z_m, beta_m, alpha_m]`.
const DIMS: [usize; 4] = [10240, 6144, 48, 48];
const QKV_M: usize = DIMS[0];
const Z_M: usize = DIMS[1];
const BA: usize = DIMS[2];
/// Rows of the Z fold (`Gpu::iu4_gdn_fold_rows`): Z, then one 128-row tile for beta and alpha.
const FOLD_ROWS: usize = Z_M + 128;
const BETA_ROW: usize = Z_M;
const ALPHA_ROW: usize = Z_M + 64;
/// Bytes of the explicit kernarg (`GDN_KERNARG_BYTES`).
pub const KERNARG_BYTES: usize = 120;
/// Dynamic LDS the product launches the kernel with.
const LDS: u32 = 20480;
const MAX_N: usize = 128;
const DEFAULT_K: usize = 5120;
const QKV_SEED: u64 = 0x51ec_7a11_0001_0001;
const Z_SEED: u64 = 0x51ec_7a11_0002_0002;
const X_SEED_M: usize = 4;
const SAMPLE_ROWS: usize = 192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    V64,
    B1s,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::V64 => "v64",
            Self::B1s => "b1s",
        }
    }
    /// The schedule of the shipped export of this variant.
    fn tune(self) -> Tune {
        match self {
            Self::V64 => verify64_tune(Epi::QkvzaGdn, true),
            Self::B1s => Tune {
                qkv: QkvStore::Set,
                ..Tune::default()
            },
        }
    }
}

fn spec_with(tune: Tune) -> Spec {
    Spec {
        act: ALayout::Slab,
        tune,
        ..Spec::control(Epi::QkvzaGdn)
    }
}

/// Explicit kernarg of the fused SET projection, in the product's order.
#[allow(clippy::too_many_arguments)]
fn pack(
    wqkv: u64,
    wz: u64,
    xq: u64,
    y_qkv: u64,
    y_z: u64,
    y_beta: u64,
    y_alpha: u64,
    k: usize,
    n: usize,
) -> Vec<u8> {
    let mut b = Vec::with_capacity(KERNARG_BYTES);
    for p in [wqkv, wz, xq, y_z, y_beta, y_alpha] {
        put_ptr(&mut b, p);
    }
    for v in [QKV_M, Z_M, k, n] {
        put_i32(&mut b, v as i32);
    }
    // The five GDN pointer slots are never read (the product passes `y_z`); `X` is the QKV output.
    for p in [y_z, y_z, y_z, y_z, y_z, y_qkv] {
        put_ptr(&mut b, p);
    }
    for f in [0.0f32, 0.0] {
        put_f32(&mut b, f);
    }
    debug_assert_eq!(b.len(), KERNARG_BYTES);
    b
}

/// The Z fold weights: `FOLD_ROWS` seeded rows, with the rows the fold leaves empty (beta's and
/// alpha's tile padding) zeroed like the loader's fold.
fn fold_weights(k: usize, seed: u64) -> Vec<u8> {
    let row = k / 256 * GROUP_BYTES;
    let mut w = synth_weights(FOLD_ROWS, k, seed);
    for rows in [BETA_ROW + BA..ALPHA_ROW, ALPHA_ROW + BA..FOLD_ROWS] {
        w[rows.start * row..rows.end * row].fill(0);
    }
    w
}

/// Fold rows the reference checks: sampled Z rows, every beta and every alpha row.
fn fold_ref_rows() -> Vec<usize> {
    let mut rows = sample_rows(Z_M, SAMPLE_ROWS);
    rows.extend(BETA_ROW..BETA_ROW + BA);
    rows.extend(ALPHA_ROW..ALPHA_ROW + BA);
    rows
}

/// Byte length of each output section `[Y_qkv, Y_z, Y_beta, Y_alpha]` at `n` tokens.
fn section_bytes(n: usize) -> [usize; 4] {
    [QKV_M, Z_M, BA, BA].map(|m| n * m * 4)
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
            "shape N={n},K={k}: the fused SET projection is launched for 1 <= N <= {MAX_N} and K % 256 == 0"
        ));
    }
    Ok((n, k))
}

struct Dev {
    w_qkv: hip_bridge::DeviceBuffer,
    w_z: hip_bridge::DeviceBuffer,
    xq: hip_bridge::DeviceBuffer,
    /// Per arm: `[Y_qkv, Y_z, Y_beta, Y_alpha]`.
    y: [[hip_bridge::DeviceBuffer; 4]; 2],
    /// `--flush` pool: `(Wqkv, Wz fold)` of sets `1..pool_sets()`.
    pool: Vec<(hip_bridge::DeviceBuffer, hip_bridge::DeviceBuffer)>,
}

pub struct Qkvzaset {
    n: usize,
    k: usize,
    variant: Variant,
    cand: Spec,
    base: Spec,
    cand_knobs: Value,
    /// QKV weights + activations, and the Z fold weights + the same activations.
    hq: Option<Host>,
    hz: Option<Host>,
    dev: Option<Dev>,
    /// CPU reference: `[qkv rows, z-fold rows]` x N, each row-major in the rows' order.
    refs: OnceCell<(Vec<f32>, Vec<f32>)>,
    cold: crate::cold::ColdMode,
}

pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    let (n, k) = parse_shape(shape)?;
    let variant = match knobs.take_or("variant", "v64").as_str() {
        "v64" => Variant::V64,
        "b1s" => Variant::B1s,
        o => return Err(format!("--variant {o}: expected v64|b1s")),
    };
    let base = spec_with(variant.tune());
    let mut tune = variant.tune();
    let mut cand_knobs = json!({"variant": variant.name()});
    if let Some(w) = knobs.take("wait") {
        tune.wait = w.parse()?;
        cand_knobs["wait"] = json!(tune.wait.name());
    }
    if let Some(i) = knobs.take("idle") {
        tune.idle = i.parse()?;
        cand_knobs["idle"] = json!(tune.idle.name());
    }
    let cand = spec_with(tune);
    cand_knobs["symbol"] = json!(cand.symbol());
    cand_knobs["schedule"] = json!({
        "wait": tune.wait.name(), "idle": tune.idle.name(), "fetch": tune.fetch.name(),
        "ring": tune.ring.name(), "fold-op": tune.fold_op.name(),
    });
    Ok(Box::new(Qkvzaset {
        n,
        k,
        variant,
        cand,
        base,
        cand_knobs,
        hq: None,
        hz: None,
        dev: None,
        refs: OnceCell::new(),
        cold: crate::cold::ColdMode::Hot,
    }))
}

impl Qkvzaset {
    /// Bytes of one weight set a dispatch streams (QKV rows + the Z fold).
    fn set_bytes(&self) -> usize {
        (QKV_M + FOLD_ROWS) * (self.k / 256) * GROUP_BYTES
    }

    fn out_bytes(&self) -> usize {
        section_bytes(self.n).iter().sum()
    }

    fn set_weights(
        dev: &Dev,
        set: usize,
    ) -> (&hip_bridge::DeviceBuffer, &hip_bridge::DeviceBuffer) {
        match set {
            0 => (&dev.w_qkv, &dev.w_z),
            s => {
                let (a, b) = &dev.pool[s - 1];
                (a, b)
            }
        }
    }

    fn emit_object(&self) -> Result<Object> {
        let e = iu4_gemm::emit(self.cand)?;
        let elf = peacemaker_isa::native::assemble(&e.s_text, self.cand.arch)?;
        let symbol = self.cand.symbol();
        Ok(Object {
            source: json!({
                "source": "peacemaker_isa::kernels::iu4_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                "variant": self.cand.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.cand_knobs}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    /// CPU reference rows: `(qkv rows, z-fold rows)` x N.
    fn reference_rows(&self) -> Result<&(Vec<f32>, Vec<f32>)> {
        let (hq, hz) = (
            self.hq.as_ref().ok_or("check before setup")?,
            self.hz.as_ref().ok_or("check before setup")?,
        );
        Ok(self.refs.get_or_init(|| {
            let q = reference(
                hq,
                QKV_M,
                self.n,
                self.k,
                false,
                &sample_rows(QKV_M, SAMPLE_ROWS),
                RefFold::Fma,
            );
            let z = reference(
                hz,
                FOLD_ROWS,
                self.n,
                self.k,
                false,
                &fold_ref_rows(),
                RefFold::Fma,
            );
            (q, z)
        }))
    }
}

impl Workload for Qkvzaset {
    fn kernel(&self) -> &'static str {
        "qkvzaset"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        let mut v = json!({
            "N": self.n, "K": self.k, "dims": DIMS, "fold_rows": FOLD_ROWS,
            "launch_rows": QKV_M + FOLD_ROWS, "variant": self.variant.name(),
            "kernarg_bytes": KERNARG_BYTES,
        });
        // Cold-cache timing is part of the shape; a hot run's shape keeps no cold keys.
        crate::cold::annotate_shape(&mut v, self.cold, self.pool_sets(), self.set_bytes());
        v
    }
    fn candidate_knobs(&self) -> Value {
        self.cand_knobs.clone()
    }

    fn candidate(&self) -> Result<Object> {
        self.emit_object()
    }

    fn baseline(&self) -> Result<Object> {
        let symbol = self.base.symbol();
        crate::common::shipped_object(
            rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
            "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco",
            "crates/rdna-compute/src/gemm.rs Gpu::gemm_qkvza_set_g12_iu4_prepared (symbol `_v64` unless \
             HIPFIRE_VERIFY_V64=0; module gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s via g12_iu4_b1s_image)",
            &symbol,
            &[&symbol],
            json!({"variant": self.variant.name(), "symbol": symbol}),
        )
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let x = Host::synth(X_SEED_M, self.n, self.k, false);
        let w_qkv = synth_weights(QKV_M, self.k, QKV_SEED);
        let w_z = fold_weights(self.k, Z_SEED);
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            if !b.is_empty() {
                gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            }
            Ok(d)
        };
        let mut pool = Vec::new();
        if self.cold.pool() {
            for s in 1..self.pool_sets() {
                pool.push((
                    up(&synth_weights(QKV_M, self.k, pool_seed(QKV_SEED, s)))?,
                    up(&fold_weights(self.k, pool_seed(Z_SEED, s)))?,
                ));
            }
        }
        let alloc4 = || -> Result<[hip_bridge::DeviceBuffer; 4]> {
            let [a, b, c, d] = section_bytes(self.n);
            let m = |n: usize| gpu.hip.malloc(n.max(16)).map_err(dev_err);
            Ok([m(a)?, m(b)?, m(c)?, m(d)?])
        };
        let dev = Dev {
            w_qkv: up(&w_qkv)?,
            w_z: up(&w_z)?,
            xq: up(&to_slab(&x.recs, self.n))?,
            y: [alloc4()?, alloc4()?],
            pool,
        };
        self.hq = Some(Host {
            w: w_qkv,
            w_up: Vec::new(),
            recs: x.recs.clone(),
            y0: Vec::new(),
        });
        self.hz = Some(Host {
            w: w_z,
            w_up: Vec::new(),
            recs: x.recs,
            y0: Vec::new(),
        });
        self.dev = Some(dev);
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        self.launch_set(arm, 0)
    }

    fn set_cold(&mut self, mode: crate::cold::ColdMode) -> Result<()> {
        self.cold = mode;
        Ok(())
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

    fn pool_sets(&self) -> usize {
        if self.cold.pool() {
            crate::cold::pool_sets(self.set_bytes())
        } else {
            1
        }
    }

    fn launch_set(&self, arm: Arm, set: usize) -> Launch {
        let dev = self.dev.as_ref().expect("launch before setup");
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let y = &dev.y[match arm {
            Arm::Baseline => 0,
            Arm::Candidate => 1,
        }];
        let (w_qkv, w_z) = Self::set_weights(dev, set);
        Launch {
            kernarg: pack(
                ptr(w_qkv),
                ptr(w_z),
                ptr(&dev.xq),
                ptr(&y[0]),
                ptr(&y[1]),
                ptr(&y[2]),
                ptr(&y[3]),
                self.k,
                self.n,
            ),
            grid: [
                ((QKV_M + FOLD_ROWS) / 128) as u32,
                self.n.div_ceil(128) as u32,
                1,
            ],
            block: [256, 1, 1],
            lds: LDS,
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let y = &dev.y[match arm {
            Arm::Baseline => 0,
            Arm::Candidate => 1,
        }];
        for b in y {
            gpu.hip.memset(b, 0, b.size()).map_err(dev_err)?;
        }
        Ok(())
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        let y = &dev.y[match arm {
            Arm::Baseline => 0,
            Arm::Candidate => 1,
        }];
        let mut out = Vec::with_capacity(self.out_bytes());
        for (b, len) in y.iter().zip(section_bytes(self.n)) {
            let mut v = vec![0u8; b.size()];
            gpu.hip.memcpy_dtoh(&mut v, b).map_err(dev_err)?;
            v.truncate(len);
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        if out.len() != self.out_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.out_bytes()
            ));
        }
        let (rq, rz) = self.reference_rows()?;
        let n = self.n;
        let [qkv_b, z_b, beta_b, _] = section_bytes(n);
        let (o_qkv, o_z) = (0, qkv_b);
        let (o_beta, o_alpha) = (qkv_b + z_b, qkv_b + z_b + beta_b);
        let at = |off: usize| f32::from_le_bytes(out[off..off + 4].try_into().unwrap());
        let (mut mismatches, mut worst) = (0usize, 0f64);
        let mut cmp = |got: f32, exp: f32| {
            mismatches += usize::from(got.to_bits() != exp.to_bits());
            let d = if got.is_nan() != exp.is_nan() {
                f64::INFINITY
            } else if got.is_nan() || got == exp {
                0.0
            } else {
                (f64::from(got) - f64::from(exp)).abs()
            };
            worst = worst.max(d);
        };
        for (ri, &r) in sample_rows(QKV_M, SAMPLE_ROWS).iter().enumerate() {
            for col in 0..n {
                cmp(at(o_qkv + (col * QKV_M + r) * 4), rq[ri * n + col]);
            }
        }
        for (ri, &r) in fold_ref_rows().iter().enumerate() {
            for col in 0..n {
                let off = if r < Z_M {
                    o_z + (col * Z_M + r) * 4
                } else if r < BETA_ROW + BA {
                    o_beta + (col * BA + (r - BETA_ROW)) * 4
                } else {
                    o_alpha + (col * BA + (r - ALPHA_ROW)) * 4
                };
                cmp(at(off), rz[ri * n + col]);
            }
        }
        Ok(RefReport {
            reference: format!(
                "cpu_iu4_symfold_rn(sections=qkv|z|beta|alpha,rows=sample:{SAMPLE_ROWS}+all beta/alpha,fold=fma,mismatching_elements={mismatches})"
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass: mismatches == 0,
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        Ok(max_diff(a, b, 4))
    }

    fn requires_baseline_identity(&self) -> bool {
        true
    }

    fn emu(&self) -> Value {
        Value::String(
            "refused (the fused qkvzaset projection has no emulator harness in pm_loop)".into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(shape: &str, kv: &[(&str, &str)]) -> Result<Box<dyn Workload>> {
        let mut k = Knobs::default();
        for (a, b) in kv {
            k.0.insert((*a).into(), (*b).into());
        }
        let w = new(shape, &mut k)?;
        k.finish()?;
        Ok(w)
    }

    fn le_u64(b: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }
    fn le_i32(b: &[u8], off: usize) -> i32 {
        i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    #[test]
    fn kernarg_is_the_products_120_bytes_and_matches_the_spec_layout() {
        let b = pack(1, 2, 3, 4, 5, 6, 7, 5120, 64);
        assert_eq!(b.len(), KERNARG_BYTES);
        // gemm_qkvza_set_g12_iu4_prepared: ptrs[Wqkv, Wz], xq, outs[Yz, Ybeta, Yalpha], ints, gdn slots.
        for (off, v) in [(0, 1), (8, 2), (16, 3), (24, 5), (32, 6), (40, 7)] {
            assert_eq!(le_u64(&b, off), v, "ptr@{off}");
        }
        for (off, v) in [(48, 10240), (52, 6144), (56, 5120), (60, 64)] {
            assert_eq!(le_i32(&b, off), v, "int@{off}");
        }
        // Five GDN slots point at y_z; the X slot is the QKV output.
        for off in [64, 72, 80, 88, 96] {
            assert_eq!(le_u64(&b, off), 5, "gdn@{off}");
        }
        assert_eq!(le_u64(&b, 104), 4);
        assert_eq!(&b[112..120], &[0u8; 8]);
        // Every name/offset of the generator's layout agrees with the packer.
        let spec = spec_with(Variant::V64.tune());
        let layout = spec.kernargs();
        let want = |name: &str| {
            layout
                .args
                .iter()
                .find(|a| a.name == name)
                .unwrap_or_else(|| panic!("no kernarg {name}"))
                .offset
        };
        for (name, off) in [
            ("Wqkv", 0),
            ("Wz", 8),
            ("Xq", 16),
            ("Yz", 24),
            ("Ybeta", 32),
            ("Yalpha", 40),
            ("M0", 48),
            ("Mz", 52),
            ("K", 56),
            ("N", 60),
            ("X", 104),
            ("QScale", 112),
            ("Eps", 116),
        ] {
            assert_eq!(want(name), off, "{name}");
        }
        assert_eq!(iu4_gemm::spec::GDN_KERNARG_BYTES as usize, KERNARG_BYTES);
    }

    #[test]
    fn launch_geometry_is_the_products() {
        // 129 row tiles (10240 + 6272 rows), ceil(n/128) token tiles, 256 threads, 20480 B LDS.
        assert_eq!((QKV_M + FOLD_ROWS) / 128, 129);
        assert_eq!(FOLD_ROWS, 6272);
        assert_eq!(
            spec_with(Variant::V64.tune()).lds_launch(),
            LDS,
            "the generator's dynamic LDS is the product's 20480"
        );
    }

    #[test]
    fn symbols_name_the_shipped_exports_and_both_are_in_the_bundle() {
        for (v, sym) in [
            ("v64", "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s_v64"),
            ("b1s", "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s"),
        ] {
            let w = make("N=64", &[("variant", v)]).unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, sym);
            let c = w.candidate().unwrap();
            // The default candidate is the generator's own emit of the shipped export.
            assert_eq!(c.launch_symbol, sym);
            assert_eq!(w.candidate_knobs()["variant"], v);
            assert_eq!(
                b.source["sha256"],
                sha256_hex(rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S)
            );
        }
    }

    #[test]
    fn shape_grammar_and_knobs() {
        let w = make("N=64", &[]).unwrap();
        assert_eq!(w.shape_json()["K"], 5120);
        assert_eq!(w.shape_json()["N"], 64);
        assert_eq!(w.shape_json()["launch_rows"], 16512);
        assert!(
            w.shape_json().get("flush").is_none(),
            "hot shape has no cold keys"
        );
        assert_eq!(make("K=2560,N=32", &[]).unwrap().shape_json()["K"], 2560);
        for bad in [
            "",
            "N=0",
            "N=129",
            "N=64,K=100",
            "N=64,N=32",
            "N=64,M=1",
            "K=5120",
        ] {
            assert!(make(bad, &[]).is_err(), "{bad}");
        }
        assert!(make("N=64", &[("variant", "v32")]).is_err());
        assert!(make("N=64", &[("bogus", "1")]).is_err());
        let w = make("N=64", &[("wait", "after"), ("idle", "off")]).unwrap();
        assert_eq!(w.candidate_knobs()["wait"], "after");
        assert_eq!(w.candidate_knobs()["idle"], "off");
        assert!(make("N=64", &[("wait", "later")]).is_err());
    }

    #[test]
    fn cold_pool_exceeds_the_last_level_cache_and_is_recorded() {
        let mut w = make("N=64", &[]).unwrap();
        assert_eq!(w.pool_sets(), 1);
        w.set_cold(crate::cold::ColdMode::Flush).unwrap();
        let set_bytes = (10240 + 6272) * 20 * 136;
        let sets = w.pool_sets();
        assert_eq!(sets % 2, 1);
        assert!(sets * set_bytes > crate::cold::POOL_BYTES);
        let s = w.shape_json();
        assert_eq!(s["flush"], true);
        assert_eq!(s["timing_pool"]["sets"], sets);
        assert_eq!(s["timing_pool"]["bytes_per_set"], set_bytes);
        assert_eq!(s["timing_pool"]["total_bytes"], sets * set_bytes);
        assert_eq!(s["timing_pool"]["dispatch_i_uses_set"], "i % sets");
    }

    #[test]
    fn fold_rows_and_output_sections() {
        let k = 256;
        let w = fold_weights(k, 1);
        assert_eq!(w.len(), FOLD_ROWS * GROUP_BYTES);
        let row = |r: usize| &w[r * GROUP_BYTES..(r + 1) * GROUP_BYTES];
        for r in [BETA_ROW + BA, ALPHA_ROW - 1, ALPHA_ROW + BA, FOLD_ROWS - 1] {
            assert!(row(r).iter().all(|b| *b == 0), "row {r} is a zero pad row");
        }
        for r in [
            0,
            Z_M - 1,
            BETA_ROW,
            BETA_ROW + BA - 1,
            ALPHA_ROW,
            ALPHA_ROW + BA - 1,
        ] {
            assert!(row(r).iter().any(|b| *b != 0), "row {r} carries weights");
        }
        let rows = fold_ref_rows();
        assert!(rows.contains(&BETA_ROW) && rows.contains(&(ALPHA_ROW + BA - 1)));
        assert!(rows.iter().all(|r| !(BETA_ROW + BA..ALPHA_ROW).contains(r)));
        assert_eq!(
            section_bytes(64),
            [64 * 10240 * 4, 64 * 6144 * 4, 64 * 48 * 4, 64 * 48 * 4]
        );
    }

    /// GPU-free identity: the generator's emit of the shipped schedule has the shipped export's code
    /// (`.text` sha256, the incumbent shas of `iu4-roofline/incumbent_shas.json`).
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
        for (variant, sha8) in [("v64", "37c9b735"), ("b1s", "340bcee3")] {
            let w = make("N=64", &[("variant", variant)]).unwrap();
            let (base, cand) = (code(&w.baseline().unwrap()), code(&w.candidate().unwrap()));
            assert_eq!(base, cand, "{variant}");
            assert!(base.0.starts_with(sha8), "{variant}: {}", base.0);
            // 120 explicit bytes + the 256-byte hidden block.
            assert_eq!(base.1 as usize, KERNARG_BYTES + 256);
        }
        // A different schedule is a different object.
        let w = make("N=64", &[("wait", "after")]).unwrap();
        assert_ne!(
            code(&w.baseline().unwrap()).0,
            code(&w.candidate().unwrap()).0
        );
    }

    #[test]
    fn null_object_is_the_shipped_descriptor_with_an_empty_body() {
        let w = make("N=64", &[]).unwrap();
        let (base, null) = (w.baseline().unwrap(), w.null_object().unwrap().unwrap());
        assert_eq!(null.launch_symbol, base.launch_symbol);
        let res = |o: &Object| {
            let p = crate::lift(&o.bytes).unwrap();
            crate::resources(&p, std::slice::from_ref(&o.launch_symbol)).unwrap()[0].clone()
        };
        let (b, n) = (res(&base), res(&null));
        assert!(n["code_bytes"].as_u64().unwrap() <= 8 && b["code_bytes"].as_u64().unwrap() > 9000);
        for key in ["vgprs", "sgprs", "kernarg_bytes", "max_flat_workgroup_size"] {
            assert_eq!(n[key], b[key], "{key}");
        }
    }
}
