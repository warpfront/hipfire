// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Resident device implementations of the host CPU-AR operation order in
//! `hipfire-runtime::spec_sampling` (`kernels/src/spec_sampling_cpu_ar.hip`):
//!
//! * [`Gpu::spec_cpu_ar_rows_enqueue`] — `SparseDist::build_from_logits` for
//!   `row_count` contiguous logit rows (target `p`).
//! * [`Gpu::spec_cpu_ar_candidates_enqueue`] — `SparseDist::build_from_candidates`
//!   for one draft row of at most eight candidates, plus the inverse-CDF draw.
//! * [`Gpu::spec_accept_ordered_enqueue`] — `accept_sampled_prefix` over the
//!   resident rows with the host's pre-drawn uniform tape.
//!
//! The unchanged literal serial kernels are diagnostic-only references of the
//! host operation order (no FMA contraction, plain HIP `expf`), not exact-host
//! probability twins. The CPU probability bit-equality gate was retired after
//! the first gfx1201 synthetic row differed at token 42939: GPU `0x3d66ae67`
//! versus CPU `0x3d66ae68` (1 ULP). No exact-host probability claim is made.
//! Launches are stream-ordered, atomic-free and recorded through
//! `launch_maybe_blob`; no enqueue method reads anything back. Allocate
//! [`SpecDistRows`]/[`SpecDistScratch`]/tapes and call
//! [`Gpu::spec_cpu_ar_precompile`] before capture or the timed window.
//!
//! Production API candidate kernels (twinHW parity still ungated).
//! [`Gpu::spec_cpu_ar_rows_parallel_enqueue`] (the canonical
//! parallel target-row law) and [`Gpu::spec_accept_ordered_enqueue`] run the
//! certified PeaceMaker (builder-emitted) images embedded in `kernels.rs`, with
//! no hipcc fallback, on exactly gfx1151 and gfx1201
//! ([`Gpu::spec_cpu_ar_device_supported`]); any other device is an error.
//! [`Gpu::spec_cpu_ar_precompile`] binds only those two PM kernels and compiles
//! nothing from HIP. The hipcc kernels stay as the explicit oracle twins:
//! `spec_cpu_ar_rows_parallel_hip_twin_*`, `spec_accept_ordered_hip_twin_*`,
//! the serial reference rows and the candidates kernel
//! ([`Gpu::spec_cpu_ar_oracle_precompile`] binds the three serial HIP kernels,
//! [`Gpu::spec_cpu_ar_candidates_precompile`] only the candidates kernel).
//! The canonical PM candidate still requires the twinHW gate against its HIP
//! twins on gfx1151 and gfx1201; the retired CPU comparison does not establish
//! PM/HIP parity. No performance claim is made.
//!
//! The crate has no integer `DType`: `u32` buffers are `F32`-typed tensors of
//! the same byte size (the convention of `sampled_rows_compact`); read them
//! back as raw bytes.
//!
//! Self-contained via `include_str!` (no shared-registry edits).

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for every launcher in this module.
pub const SPEC_SAMPLING_CPU_AR_SRC: &str =
    include_str!("../../../kernels/src/spec_sampling_cpu_ar.hip");
/// Module key.
pub const SPEC_SAMPLING_CPU_AR_MODULE: &str = "spec_sampling_cpu_ar";
/// Device symbol of [`Gpu::spec_cpu_ar_rows_enqueue`].
pub const SPEC_CPU_AR_ROWS_SYMBOL: &str = "spec_cpu_ar_rows_f32";
/// Device symbol of [`Gpu::spec_cpu_ar_candidates_enqueue`].
pub const SPEC_CPU_AR_CANDIDATES_SYMBOL: &str = "spec_cpu_ar_candidates_f32";
/// Device symbol of [`Gpu::spec_accept_ordered_enqueue`].
pub const SPEC_ACCEPT_ORDERED_SYMBOL: &str = "spec_accept_ordered_u32";

/// Fixed entry capacity of one dist row (`SampleSpec::MAX_POOL`).
pub const SPEC_CPU_AR_CAP: usize = 64;
/// Candidates a draft row is built from.
pub const SPEC_CPU_AR_MAX_CANDIDATES: usize = 8;
/// `eos` value meaning "no EOS" and the accept `terminal` of an accepted EOS.
pub const SPEC_CPU_AR_EOS_NONE: u32 = u32::MAX;
/// `picked` value written when the draft row has no finite candidate/mass
/// (the host's `Err`).
pub const SPEC_CPU_AR_PICK_INVALID: u32 = u32::MAX;

/// Words of scratch per row: 64 pool token ids then 64 pool logits.
const SCRATCH_ROW_WORDS: usize = 128;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("spec_sampling_cpu_ar: {}", msg.as_ref()))
}

/// A request's truncation, resolved in Rust from the host `SampleSpec`
/// (`SampleSpec::cpu_ar`); fields are passed to the device unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecCpuArPolicy {
    pub temperature: f32,
    /// Already clamped to `[0, 1]`.
    pub top_p: f32,
    /// Candidate pool, in `1..=64`.
    pub pool: u32,
    /// Candidates kept, in `1..=pool`.
    pub cap: u32,
    /// `0` disables the cut; else in `(0, 1]`.
    pub min_p: f32,
}

impl SpecCpuArPolicy {
    /// Plain field constructor (no clamping: pass `SampleSpec` fields as-is).
    pub fn new(temperature: f32, top_p: f32, pool: u32, cap: u32, min_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            pool,
            cap,
            min_p,
        }
    }

    fn validate(&self, what: &str) -> HipResult<()> {
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return Err(err(format!(
                "{what}: needs a positive finite temperature, got {}",
                self.temperature
            )));
        }
        if self.pool == 0 || self.pool as usize > SPEC_CPU_AR_CAP {
            return Err(err(format!("{what}: pool {} outside 1..=64", self.pool)));
        }
        if self.cap == 0 || self.cap > self.pool {
            return Err(err(format!(
                "{what}: cap {} outside 1..=pool {}",
                self.cap, self.pool
            )));
        }
        Ok(())
    }
}

/// Device distribution rows in `SparseDist::entries()` order (descending
/// probability, not ascending token): row `r` is
/// `ids/probabilities[r*64 .. r*64 + counts[r]]`, slots past the count are
/// zero. `counts`/`ids` are `u32` data in `F32`-typed tensors.
pub struct SpecDistRows {
    /// `u32[capacity_rows]`.
    pub counts: GpuTensor,
    /// `u32[capacity_rows * 64]`.
    pub ids: GpuTensor,
    /// `f32[capacity_rows * 64]`.
    pub probabilities: GpuTensor,
    pub capacity_rows: usize,
}

impl SpecDistRows {
    /// Zero-initialised rows (every count 0).
    pub fn new(gpu: &mut Gpu, capacity_rows: usize) -> HipResult<Self> {
        if capacity_rows == 0 {
            return Err(err("SpecDistRows::new: capacity_rows must be > 0"));
        }
        Ok(Self {
            counts: spec_cpu_ar_alloc_u32(gpu, capacity_rows)?,
            ids: spec_cpu_ar_alloc_u32(gpu, capacity_rows * SPEC_CPU_AR_CAP)?,
            probabilities: spec_cpu_ar_alloc_f32(gpu, capacity_rows * SPEC_CPU_AR_CAP)?,
            capacity_rows,
        })
    }

    /// Release the device buffers.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.free_tensor(self.counts)?;
        gpu.free_tensor(self.ids)?;
        gpu.free_tensor(self.probabilities)
    }
}

/// Reusable per-row working storage of the rows kernel (the pool slots). Sized for `capacity_rows` rows.
pub struct SpecDistScratch {
    pub words: GpuTensor,
    pub capacity_rows: usize,
}

impl SpecDistScratch {
    pub fn new(gpu: &mut Gpu, capacity_rows: usize) -> HipResult<Self> {
        if capacity_rows == 0 {
            return Err(err("SpecDistScratch::new: capacity_rows must be > 0"));
        }
        Ok(Self {
            words: spec_cpu_ar_alloc_f32(gpu, capacity_rows * SCRATCH_ROW_WORDS)?,
            capacity_rows,
        })
    }

    /// Release the device buffer.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.free_tensor(self.words)
    }
}

/// Zeroed `n`-word device buffer holding `u32` data (F32-typed tensor): proposal
/// ids, accept results, `picked`, candidates.
pub fn spec_cpu_ar_alloc_u32(gpu: &mut Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.zeros(&[n.max(1)], DType::F32)
}

/// Zeroed `n`-element f32 device buffer: the uniform tape.
pub fn spec_cpu_ar_alloc_f32(gpu: &mut Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.zeros(&[n.max(1)], DType::F32)
}

fn check_bytes(name: &str, t: &GpuTensor, need: usize) -> HipResult<()> {
    if t.buf.size() < need {
        return Err(err(format!(
            "{name} buffer too small (have {} need {need} bytes)",
            t.buf.size()
        )));
    }
    Ok(())
}

/// The embedded certified builder images of one exact architecture.
struct PmImages {
    rows_module: &'static str,
    rows_image: &'static [u8],
    rows_symbol: &'static str,
    accept_module: &'static str,
    accept_image: &'static [u8],
    accept_symbol: &'static str,
}

/// PM images of `arch` (the device's `gfx...` name, compared exactly), or
/// `None` where no certified image ships.
fn pm_images(arch: &str) -> Option<PmImages> {
    match arch {
        "gfx1151" => Some(PmImages {
            rows_module: "spec_sampling_rows_parallel_pm_gfx1151",
            rows_image: crate::kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1151,
            rows_symbol: "spec_cpu_ar_rows_parallel_f32_pm_gfx1151",
            accept_module: "spec_accept_ordered_pm_gfx1151",
            accept_image: crate::kernels::SPEC_ACCEPT_ORDERED_PM_GFX1151,
            accept_symbol: "spec_accept_ordered_u32_pm_gfx1151",
        }),
        "gfx1201" => Some(PmImages {
            rows_module: "spec_sampling_rows_parallel_pm_gfx1201",
            rows_image: crate::kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1201,
            rows_symbol: "spec_cpu_ar_rows_parallel_f32_pm_gfx1201",
            accept_module: "spec_accept_ordered_pm_gfx1201",
            accept_image: crate::kernels::SPEC_ACCEPT_ORDERED_PM_GFX1201,
            accept_symbol: "spec_accept_ordered_u32_pm_gfx1201",
        }),
        _ => None,
    }
}

/// Device symbol of the PM parallel rows kernel on `arch` (`gfx1151` or
/// `gfx1201`), `None` elsewhere.
pub fn spec_cpu_ar_rows_parallel_pm_symbol(arch: &str) -> Option<&'static str> {
    pm_images(arch).map(|p| p.rows_symbol)
}

/// Device symbol of the PM ordered-accept kernel on `arch` (`gfx1151` or
/// `gfx1201`), `None` elsewhere.
pub fn spec_accept_ordered_pm_symbol(arch: &str) -> Option<&'static str> {
    pm_images(arch).map(|p| p.accept_symbol)
}

impl Gpu {
    /// True exactly on gfx1151 and gfx1201, the architectures with an embedded
    /// certified PM image of both the parallel rows and the ordered-accept
    /// kernels (gfx1200 and every other architecture: false).
    pub fn spec_cpu_ar_device_supported(&self) -> bool {
        pm_images(&self.arch).is_some()
    }

    fn pm_images_required(&self, what: &str) -> HipResult<PmImages> {
        pm_images(&self.arch).ok_or_else(|| {
            err(format!(
                "{what}: no certified PM image for {} (gfx1151 and gfx1201 only; no hipcc fallback)",
                self.arch
            ))
        })
    }

    /// Production setup (call before graph capture and before the timed
    /// window): load the PM modules and bind the parallel rows and
    /// ordered-accept kernels. Nothing is compiled from HIP; the candidates
    /// kernel ([`Gpu::spec_cpu_ar_candidates_precompile`]) and the serial
    /// reference ([`Gpu::spec_cpu_ar_oracle_precompile`]) are not touched.
    /// Fails off gfx1151 / gfx1201.
    pub fn spec_cpu_ar_precompile(&mut self) -> HipResult<()> {
        self.spec_cpu_ar_rows_parallel_precompile()?;
        self.spec_accept_ordered_precompile()
    }

    /// Explicit oracle setup: load the HIP module and bind the serial
    /// reference rows, the candidates and the HIP ordered-accept kernels.
    pub fn spec_cpu_ar_oracle_precompile(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        for symbol in [
            SPEC_CPU_AR_ROWS_SYMBOL,
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
            SPEC_ACCEPT_ORDERED_SYMBOL,
        ] {
            self.ensure_kernel(
                SPEC_SAMPLING_CPU_AR_MODULE,
                SPEC_SAMPLING_CPU_AR_SRC,
                symbol,
            )?;
        }
        Ok(())
    }

    /// Load the HIP module and bind only the candidates kernel.
    pub fn spec_cpu_ar_candidates_precompile(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
        )
    }
    /// Build the target distribution of `row_count` contiguous logit rows
    /// (`logits`: f32 `[row_count * vocab]`, row index = token id) into rows
    /// `0..row_count` of `rows`, exactly as `SparseDist::build_from_logits`.
    /// `scratch` must cover `row_count` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_cpu_ar_rows_enqueue(
        &mut self,
        logits: &GpuTensor,
        rows: &SpecDistRows,
        scratch: &SpecDistScratch,
        vocab: usize,
        row_count: usize,
        policy: SpecCpuArPolicy,
    ) -> HipResult<()> {
        policy.validate("spec_cpu_ar_rows_enqueue")?;
        if row_count == 0 || vocab == 0 {
            return Err(err(format!(
                "rows: row_count {row_count} and vocab {vocab} must be > 0"
            )));
        }
        if vocab > i32::MAX as usize
            || row_count
                .checked_mul(vocab)
                .is_none_or(|n| n.checked_mul(4).is_none())
        {
            return Err(err("rows: row_count * vocab exceeds the addressable range"));
        }
        if row_count > rows.capacity_rows || row_count > scratch.capacity_rows {
            return Err(err(format!(
                "rows: row_count {row_count} exceeds rows capacity {} / scratch capacity {}",
                rows.capacity_rows, scratch.capacity_rows
            )));
        }
        if logits.dtype != DType::F32 {
            return Err(err(format!(
                "rows: logits must be F32, got {:?}",
                logits.dtype
            )));
        }
        check_bytes("logits", logits, row_count * vocab * 4)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_CPU_AR_ROWS_SYMBOL,
        )?;
        let lp = logits.buf.as_ptr();
        let cp = rows.counts.buf.as_ptr();
        let ip = rows.ids.buf.as_ptr();
        let pp = rows.probabilities.buf.as_ptr();
        let sp = scratch.words.buf.as_ptr();
        let vocab_i = vocab as i32;
        let pool_i = policy.pool as i32;
        let cap_i = policy.cap as i32;
        let temp = policy.temperature;
        let top_p = policy.top_p;
        let min_p = policy.min_p;
        let mut params: [*mut c_void; 11] = [
            &lp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &vocab_i as *const _ as *mut c_void,
            &pool_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
            &temp as *const _ as *mut c_void,
            &top_p as *const _ as *mut c_void,
            &min_p as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SPEC_CPU_AR_ROWS_SYMBOL,
            [row_count as u32, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(pp);
                b.push_ptr(sp);
                b.push_i32(vocab_i);
                b.push_i32(pool_i);
                b.push_i32(cap_i);
                b.push_f32(temp);
                b.push_f32(top_p);
                b.push_f32(min_p);
                b
            },
        )
    }

    /// Build draft row `row` of `rows` from up to eight candidates exactly as
    /// `SparseDist::build_from_candidates`, then draw from it with `uniform`
    /// as `SparseDist::sample`.
    ///
    /// `candidates`: 16 `u32` words, `[0, 8)` token ids and `[8, 16)` f32 logit
    /// bits; slots with a non-finite logit are dropped (pad with `-inf`).
    /// `picked`: one `u32` word that receives the drawn token, or
    /// [`SPEC_CPU_AR_PICK_INVALID`] (row count 0) when no candidate is finite
    /// or the mass is not positive finite: this is the host `Err`, which a
    /// kernel cannot raise, so the caller MUST treat it as an error (never as
    /// a distribution) when it reads `picked` back.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_cpu_ar_candidates_enqueue(
        &mut self,
        candidates: &GpuTensor,
        rows: &SpecDistRows,
        row: usize,
        vocab: usize,
        policy: SpecCpuArPolicy,
        uniform: f32,
        picked: &GpuTensor,
    ) -> HipResult<()> {
        policy.validate("spec_cpu_ar_candidates_enqueue")?;
        if vocab == 0 || vocab > u32::MAX as usize {
            return Err(err(format!("candidates: vocab {vocab} out of range")));
        }
        if row >= rows.capacity_rows {
            return Err(err(format!(
                "candidates: row {row} >= rows capacity {}",
                rows.capacity_rows
            )));
        }
        check_bytes("candidates", candidates, 16 * 4)?;
        check_bytes("picked", picked, 4)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
        )?;
        let kp = candidates.buf.as_ptr();
        let cp = rows.counts.buf.as_ptr();
        let ip = rows.ids.buf.as_ptr();
        let pp = rows.probabilities.buf.as_ptr();
        let pk = picked.buf.as_ptr();
        let row_i = row as i32;
        let cap_i = policy.cap as i32;
        let temp = policy.temperature;
        let top_p = policy.top_p;
        let min_p = policy.min_p;
        let mut params: [*mut c_void; 11] = [
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &pk as *const _ as *mut c_void,
            &row_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
            &temp as *const _ as *mut c_void,
            &top_p as *const _ as *mut c_void,
            &min_p as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(kp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(pp);
                b.push_ptr(pk);
                b.push_i32(row_i);
                b.push_i32(cap_i);
                b.push_f32(temp);
                b.push_f32(top_p);
                b.push_f32(min_p);
                b.push_f32(uniform);
                b
            },
        )
    }

    /// `accept_sampled_prefix` over resident rows. `target` rows `0..=k`,
    /// `draft` rows `0..k`; `proposal_ids`: `u32[k]`; `uniforms`: f32 tape of
    /// at least `k + 1` entries (row `j` verdict `uniforms[j]`, its residual
    /// `uniforms[j + 1]`, bonus `uniforms[k]`); `result`: `u32[2]` =
    /// `[accepted, terminal]`, terminal = [`SPEC_CPU_AR_EOS_NONE`] when the last
    /// accepted draft was `eos` (no bonus). `eos = SPEC_CPU_AR_EOS_NONE`
    /// disables the EOS stop. The host's consumed draws are `accepted` after an
    /// accepted EOS, `accepted + 2` after a rejection, `k + 1` when all
    /// accepted.
    ///
    /// Production: runs the certified PM image (gfx1151 / gfx1201 only, no
    /// hipcc fallback); [`Gpu::spec_accept_ordered_hip_twin_enqueue`] is the
    /// explicit hipcc oracle with the same arguments and bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_accept_ordered_enqueue(
        &mut self,
        target: &SpecDistRows,
        draft: &SpecDistRows,
        proposal_ids: &GpuTensor,
        uniforms: &GpuTensor,
        result: &GpuTensor,
        k: usize,
        eos: u32,
    ) -> HipResult<()> {
        let pm = self.pm_images_required("spec_accept_ordered_enqueue")?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(pm.accept_module, pm.accept_image, pm.accept_symbol)?;
        self.spec_accept_ordered_launch(
            pm.accept_symbol,
            target,
            draft,
            proposal_ids,
            uniforms,
            result,
            k,
            eos,
        )
    }

    /// Load the PM ordered-accept module and bind its kernel only (gfx1151 /
    /// gfx1201; error elsewhere).
    pub fn spec_accept_ordered_precompile(&mut self) -> HipResult<()> {
        let pm = self.pm_images_required("spec_accept_ordered_precompile")?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(pm.accept_module, pm.accept_image, pm.accept_symbol)
    }

    /// Load the HIP module and bind the hipcc ordered-accept kernel only (the
    /// explicit oracle twin of [`Gpu::spec_accept_ordered_precompile`]).
    pub fn spec_accept_ordered_hip_twin_precompile(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_ACCEPT_ORDERED_SYMBOL,
        )
    }

    /// The hipcc `accept_sampled_prefix` kernel (explicit oracle twin of
    /// [`Gpu::spec_accept_ordered_enqueue`]; identical arguments).
    #[allow(clippy::too_many_arguments)]
    pub fn spec_accept_ordered_hip_twin_enqueue(
        &mut self,
        target: &SpecDistRows,
        draft: &SpecDistRows,
        proposal_ids: &GpuTensor,
        uniforms: &GpuTensor,
        result: &GpuTensor,
        k: usize,
        eos: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_ACCEPT_ORDERED_SYMBOL,
        )?;
        self.spec_accept_ordered_launch(
            SPEC_ACCEPT_ORDERED_SYMBOL,
            target,
            draft,
            proposal_ids,
            uniforms,
            result,
            k,
            eos,
        )
    }

    /// Validate and launch the (already bound) accept kernel `symbol`.
    #[allow(clippy::too_many_arguments)]
    fn spec_accept_ordered_launch(
        &mut self,
        symbol: &'static str,
        target: &SpecDistRows,
        draft: &SpecDistRows,
        proposal_ids: &GpuTensor,
        uniforms: &GpuTensor,
        result: &GpuTensor,
        k: usize,
        eos: u32,
    ) -> HipResult<()> {
        if k >= target.capacity_rows || k > draft.capacity_rows || k > i32::MAX as usize {
            return Err(err(format!(
                "accept: k {k} needs {} target rows (have {}) and {k} draft rows (have {})",
                k + 1,
                target.capacity_rows,
                draft.capacity_rows
            )));
        }
        check_bytes("proposal_ids", proposal_ids, k * 4)?;
        check_bytes("uniforms", uniforms, (k + 1) * 4)?;
        check_bytes("result", result, 8)?;
        let tc = target.counts.buf.as_ptr();
        let ti = target.ids.buf.as_ptr();
        let tp = target.probabilities.buf.as_ptr();
        let dc = draft.counts.buf.as_ptr();
        let di = draft.ids.buf.as_ptr();
        let dp = draft.probabilities.buf.as_ptr();
        let pr = proposal_ids.buf.as_ptr();
        let un = uniforms.buf.as_ptr();
        let rs = result.buf.as_ptr();
        let k_i = k as i32;
        let mut params: [*mut c_void; 11] = [
            &tc as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &dc as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &pr as *const _ as *mut c_void,
            &un as *const _ as *mut c_void,
            &rs as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
            &eos as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(symbol, [1, 1, 1], [1, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(tc);
            b.push_ptr(ti);
            b.push_ptr(tp);
            b.push_ptr(dc);
            b.push_ptr(di);
            b.push_ptr(dp);
            b.push_ptr(pr);
            b.push_ptr(un);
            b.push_ptr(rs);
            b.push_i32(k_i);
            b.push_u32(eos);
            b
        })
    }
}

/// Kernel source of the parallel target-row builder: the explicit hipcc oracle
/// twin of the production PM kernel. A separate module from
/// [`SPEC_SAMPLING_CPU_AR_SRC`], so compiling it cannot change the reference
/// code object.
pub const SPEC_SAMPLING_CPU_AR_PARALLEL_SRC: &str =
    include_str!("../../../kernels/src/spec_sampling_cpu_ar_parallel.hip");
/// Module key of the hipcc parallel row builder.
pub const SPEC_SAMPLING_CPU_AR_PARALLEL_MODULE: &str = "spec_sampling_cpu_ar_parallel";
/// Device symbol of the hipcc parallel rows kernel
/// ([`Gpu::spec_cpu_ar_rows_parallel_hip_twin_enqueue`]); the production PM
/// symbol is [`spec_cpu_ar_rows_parallel_pm_symbol`].
pub const SPEC_CPU_AR_ROWS_PARALLEL_SYMBOL: &str = "spec_cpu_ar_rows_parallel_f32";

/// Threads per workgroup of the parallel rows kernel (one workgroup per row).
const ROWS_PARALLEL_BLOCK: u32 = 256;

impl Gpu {
    /// Production setup of the parallel rows kernel: load the PM module for
    /// this device and bind its kernel only (gfx1151 / gfx1201; error
    /// elsewhere, no hipcc fallback). The ordered-accept kernel is bound by
    /// [`Gpu::spec_accept_ordered_precompile`]; [`Gpu::spec_cpu_ar_precompile`]
    /// does both.
    pub fn spec_cpu_ar_rows_parallel_precompile(&mut self) -> HipResult<()> {
        let pm = self.pm_images_required("spec_cpu_ar_rows_parallel_precompile")?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(pm.rows_module, pm.rows_image, pm.rows_symbol)
    }

    /// Load the hipcc parallel rows module and bind its kernel only (the
    /// explicit oracle twin of [`Gpu::spec_cpu_ar_rows_parallel_precompile`]).
    pub fn spec_cpu_ar_rows_parallel_hip_twin_precompile(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_PARALLEL_MODULE,
            SPEC_SAMPLING_CPU_AR_PARALLEL_SRC,
            SPEC_CPU_AR_ROWS_PARALLEL_SYMBOL,
        )
    }

    /// Parallel (one 256-thread workgroup per row) target-row builder: the
    /// production PM kernel (certified builder image, gfx1151 / gfx1201 only,
    /// no hipcc fallback). It writes the same [`SpecDistRows`] layout as
    /// [`Gpu::spec_cpu_ar_rows_enqueue`] but follows the canonical law, not the
    /// host CPU-AR law: canonical selection (higher f32 logit first, equal
    /// numeric logits to the lower token id) and fixed-tree aggregate sums.
    /// Output is not byte-equal to the serial reference, and reference tied
    /// supports are not equivalent to it. The PM kernel is the byte twin of the
    /// hipcc kernel run by [`Gpu::spec_cpu_ar_rows_parallel_hip_twin_enqueue`].
    ///
    /// `logits`: f32 `[row_count * vocab]`. `logits` must not alias any `rows`
    /// buffer. There is no global scratch. The kernel has ten arguments in the
    /// order `logits, counts, ids, probabilities, vocab, pool, cap, temperature,
    /// top_p, min_p`, launched with grid `[row_count, 1, 1]`, block `[256, 1, 1]`
    /// and no dynamic shared memory. No readback or synchronization is done;
    /// launches are recorded through `launch_maybe_blob`.
    pub fn spec_cpu_ar_rows_parallel_enqueue(
        &mut self,
        logits: &GpuTensor,
        rows: &SpecDistRows,
        vocab: usize,
        row_count: usize,
        policy: SpecCpuArPolicy,
    ) -> HipResult<()> {
        const NAME: &str = "spec_cpu_ar_rows_parallel_enqueue";
        policy.validate(NAME)?;
        let pm = self.pm_images_required(NAME)?;
        rows_parallel_check(logits, rows, vocab, row_count)?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(pm.rows_module, pm.rows_image, pm.rows_symbol)?;
        self.rows_parallel_launch(pm.rows_symbol, logits, rows, vocab, row_count, policy)
    }

    /// The hipcc parallel rows kernel (`kernels/src/spec_sampling_cpu_ar_parallel.hip`):
    /// the explicit oracle twin of [`Gpu::spec_cpu_ar_rows_parallel_enqueue`],
    /// identical arguments, layout and law.
    pub fn spec_cpu_ar_rows_parallel_hip_twin_enqueue(
        &mut self,
        logits: &GpuTensor,
        rows: &SpecDistRows,
        vocab: usize,
        row_count: usize,
        policy: SpecCpuArPolicy,
    ) -> HipResult<()> {
        policy.validate("spec_cpu_ar_rows_parallel_hip_twin_enqueue")?;
        rows_parallel_check(logits, rows, vocab, row_count)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_PARALLEL_MODULE,
            SPEC_SAMPLING_CPU_AR_PARALLEL_SRC,
            SPEC_CPU_AR_ROWS_PARALLEL_SYMBOL,
        )?;
        self.rows_parallel_launch(
            SPEC_CPU_AR_ROWS_PARALLEL_SYMBOL,
            logits,
            rows,
            vocab,
            row_count,
            policy,
        )
    }

    /// Launch the (already bound) parallel rows kernel `symbol`.
    fn rows_parallel_launch(
        &mut self,
        symbol: &'static str,
        logits: &GpuTensor,
        rows: &SpecDistRows,
        vocab: usize,
        row_count: usize,
        policy: SpecCpuArPolicy,
    ) -> HipResult<()> {
        let lp = logits.buf.as_ptr();
        let cp = rows.counts.buf.as_ptr();
        let ip = rows.ids.buf.as_ptr();
        let pp = rows.probabilities.buf.as_ptr();
        let vocab_i = vocab as i32;
        let pool_i = policy.pool as i32;
        let cap_i = policy.cap as i32;
        let temp = policy.temperature;
        let top_p = policy.top_p;
        let min_p = policy.min_p;
        let mut params: [*mut c_void; 10] = [
            &lp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &vocab_i as *const _ as *mut c_void,
            &pool_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
            &temp as *const _ as *mut c_void,
            &top_p as *const _ as *mut c_void,
            &min_p as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [row_count as u32, 1, 1],
            [ROWS_PARALLEL_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(pp);
                b.push_i32(vocab_i);
                b.push_i32(pool_i);
                b.push_i32(cap_i);
                b.push_f32(temp);
                b.push_f32(top_p);
                b.push_f32(min_p);
                b
            },
        )
    }
}

/// Shape checks shared by the PM and hipcc parallel rows launches.
fn rows_parallel_check(
    logits: &GpuTensor,
    rows: &SpecDistRows,
    vocab: usize,
    row_count: usize,
) -> HipResult<()> {
    if row_count == 0 || vocab == 0 {
        return Err(err(format!(
            "rows_parallel: row_count {row_count} and vocab {vocab} must be > 0"
        )));
    }
    if vocab > i32::MAX as usize || row_count > u32::MAX as usize {
        return Err(err(
            "rows_parallel: vocab or row_count exceeds the launch range",
        ));
    }
    let logit_bytes = row_count
        .checked_mul(vocab)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| err("rows_parallel: row_count * vocab exceeds the addressable range"))?;
    let out_words = row_count
        .checked_mul(SPEC_CPU_AR_CAP)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| err("rows_parallel: row_count * 64 exceeds the addressable range"))?;
    if row_count > rows.capacity_rows {
        return Err(err(format!(
            "rows_parallel: row_count {row_count} exceeds rows capacity {}",
            rows.capacity_rows
        )));
    }
    if logits.dtype != DType::F32 {
        return Err(err(format!(
            "rows_parallel: logits must be F32, got {:?}",
            logits.dtype
        )));
    }
    check_bytes("logits", logits, logit_bytes)?;
    check_bytes("counts", &rows.counts, row_count * 4)?;
    check_bytes("ids", &rows.ids, out_words)?;
    check_bytes("probabilities", &rows.probabilities, out_words)
}
