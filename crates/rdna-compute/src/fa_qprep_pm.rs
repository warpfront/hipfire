// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certified gfx1201 H16/KV2/D256 exact Q-prep fold for the A3B PM prefill attention
//! (`HIPFIRE_FA_PREFILL_PM_QPREP`). One PM launch replaces the incumbent unfused
//! deinterleave + Q-rmsnorm (`deinterleave_q_rmsnorm_f32_batched`) + Q half of
//! `rope_partial_halfsplit_batched_f32` + the PM f32->f16 preconvert: it reads the interleaved
//! projection `[rows, 16 heads, (256 q | 256 gate)]` f32, writes the gate `[rows,16,256]` f32
//! and the final rotated, rms-normalised f16 Q `[rows,16,256]` consumed unchanged by
//! `attention_q8_0_fa2_gqa_pm_gfx1201`.
//!
//! The RoPE angles are not recomputed: the same K-only
//! `rope_partial_halfsplit_batched_f32` launch that rotates K also stores the exact `cosf`/`sinf`
//! floats it rotates K with (optional trailing kernarg, null for every other caller) into a
//! `[rows,64]` f32 table (`cos[0..32]`, `sin[32..64]` per row), which this kernel consumes. Q and K
//! therefore rotate by the identical angle values as the incumbent, with no libm reimplementation.
//! Positions are physical `pbs.positions` plus the cache `compact_offset`, exactly the incumbent's.

use crate::fa_prefill_pm::FaPrefillKvCodec;
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

pub const MODULE: &str = "fa_qprep_pm_gfx1201";
pub const SYMBOL: &str = "attention_fa2_qprep_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/fa_qprep_pm_gfx1201.hxaco");
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 256;
/// Floats per row of the captured angle table: `cos[0..32]` then `sin[32..64]`.
pub const ANGLE_STRIDE: usize = 64;
/// The rotary width the angle table and the kernel are exact for.
pub const N_ROT: usize = 64;
pub const MAX_ROWS: usize = 8192;
pub const KERNARG_BYTES: usize = 48;
/// One wave32 per head, four heads per block.
pub const BLOCK: [u32; 3] = [128, 1, 1];
const HEADS_PER_BLOCK: usize = 4;
/// Floats of one interleaved projection row (per head: 256 q then 256 gate).
const INTERLEAVED_ROW: usize = HEADS * HEAD_DIM * 2;
const ROW: usize = HEADS * HEAD_DIM;

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("fa_qprep_pm: {}", message.as_ref()))
}

/// Why an A3B PM prefill request keeps the incumbent Q prep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaPrefillPmQprepRefusal {
    Disabled,
    PmDisabled,
    InventoryUnavailable,
    Recording,
    Capture,
    /// No PM body serves the cache format (neither Q8_0 nor native fp8-E4M3).
    KvFormat,
    Shape,
    Rotary,
    LegacyRope,
    Rows,
    Context,
}

impl FaPrefillPmQprepRefusal {
    fn log_once(self) {
        let index = match self {
            // The PM attention flag logs its own refusal; an explicit Q-prep 0 is silent.
            Self::Disabled | Self::PmDisabled => return,
            Self::InventoryUnavailable => 0,
            Self::Recording => 1,
            Self::Capture => 2,
            Self::KvFormat => 3,
            Self::Shape => 4,
            Self::Rotary => 5,
            Self::LegacyRope => 6,
            Self::Rows => 7,
            Self::Context => 8,
        };
        static REFUSALS: [std::sync::Once; 9] = [
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
        ];
        REFUSALS[index].call_once(|| {
            eprintln!(
                "[fa-prefill-pm-qprep] HIPFIRE_FA_PREFILL_PM_QPREP fold refused: {self:?}; \
                 keeping incumbent Q prep"
            );
        });
    }
}

/// Launch grid: one wave per (row, head), four heads per block.
pub fn grid(rows: usize) -> [u32; 3] {
    // The launcher validates rows before reaching this arithmetic.
    [((rows * HEADS).div_ceil(HEADS_PER_BLOCK)) as u32, 1, 1]
}

fn kernargs(
    interleaved: *mut c_void,
    q16: *mut c_void,
    gate: *mut c_void,
    q_norm: *mut c_void,
    angles: *mut c_void,
    rows: usize,
    eps: f32,
) -> [u8; KERNARG_BYTES] {
    let mut args = [0; KERNARG_BYTES];
    for (offset, ptr) in [
        (0, interleaved),
        (8, q16),
        (16, gate),
        (24, q_norm),
        (32, angles),
    ] {
        args[offset..offset + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
    }
    args[40..44].copy_from_slice(&(rows as i32).to_le_bytes());
    args[44..48].copy_from_slice(&eps.to_le_bytes());
    args
}

fn require_span(tensor: &GpuTensor, bytes: usize, name: &str) -> HipResult<()> {
    let have = tensor.buf.size();
    if have < bytes {
        return Err(err(format!("{name} needs {bytes} bytes, has {have}")));
    }
    Ok(())
}

impl Gpu {
    /// Pure admission of the Q-prep fold on top of the PM attention route: the Q-prep flag and
    /// the PM flag (both already resolved for the cache codec, see
    /// [`Gpu::fa_prefill_pm_qprep_flag`] / [`Gpu::fa_prefill_pm_flag`]: the explicit value or,
    /// when unset, the codec's qualification; an explicit PM `0` refuses the fold with
    /// `PmDisabled`), raw exact-gfx1201 inventory, eager only, a PM-served KV codec (Q8_0 or
    /// native fp8-E4M3; the fold itself never reads the cache), H16/KV2/D256 with a 64-wide
    /// rotary, non-legacy half-split RoPE, production rows `64..=8192` and host context
    /// `1..=262144`. Callers must additionally exclude tree/window/independent/paired/multi
    /// layouts and any consumer of pre-RoPE Q (triattn tap).
    #[allow(clippy::too_many_arguments)]
    pub fn fa_prefill_pm_qprep_admission(
        qprep_flag: bool,
        pm_flag: bool,
        arch: &str,
        in_recording: bool,
        in_capture: bool,
        rope_legacy: bool,
        kv: Option<FaPrefillKvCodec>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        batch_size: usize,
        max_ctx_len: usize,
    ) -> Result<(), FaPrefillPmQprepRefusal> {
        use crate::fa_prefill_pm::{KV_HEADS, MAX_CONTEXT, MIN_ROWS};
        use crate::prefill_route_capabilities::{raw_inventory, PrefillRoute};
        use FaPrefillPmQprepRefusal::*;
        if !qprep_flag {
            Err(Disabled)
        } else if !pm_flag {
            Err(PmDisabled)
        } else if !raw_inventory(arch, PrefillRoute::FaPrefillPmQprep)
            || !raw_inventory(arch, PrefillRoute::FaPrefillPm)
        {
            Err(InventoryUnavailable)
        } else if in_recording {
            Err(Recording)
        } else if in_capture {
            Err(Capture)
        } else if rope_legacy {
            Err(LegacyRope)
        } else if kv.is_none() {
            Err(KvFormat)
        } else if (n_heads, n_kv_heads, head_dim) != (HEADS, KV_HEADS, HEAD_DIM) {
            Err(Shape)
        } else if n_rot != N_ROT {
            Err(Rotary)
        } else if !(MIN_ROWS..=MAX_ROWS).contains(&batch_size) {
            Err(Rows)
        } else if !(1..=MAX_CONTEXT).contains(&max_ctx_len) {
            Err(Context)
        } else {
            Ok(())
        }
    }

    /// The resolved `HIPFIRE_FA_PREFILL_PM_QPREP` flag for a cache codec (fp8-E4M3 follows its
    /// own qualification on unset, every other codec the Q8_0 resolution).
    pub fn fa_prefill_pm_qprep_flag(&self, kv: Option<FaPrefillKvCodec>) -> bool {
        match kv {
            Some(FaPrefillKvCodec::Fp8E4M3) => self.flags.fa_prefill_pm_qprep_fp8_kv,
            _ => self.flags.fa_prefill_pm_qprep,
        }
    }

    /// Current-device admission of the Q-prep fold. Enabled refusals log once per reason.
    /// The PM attention itself must also be admitted for the same call (this includes the
    /// resolved `fa_prefill_pm` flag of the same codec); callers exclude
    /// tree/window/independent/paired/multi layouts.
    #[allow(clippy::too_many_arguments)]
    pub fn fa_prefill_pm_qprep_admitted(
        &self,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        batch_size: usize,
        max_ctx_len: usize,
        kv: Option<FaPrefillKvCodec>,
    ) -> bool {
        match Self::fa_prefill_pm_qprep_admission(
            self.fa_prefill_pm_qprep_flag(kv),
            self.fa_prefill_pm_flag(kv),
            &self.arch,
            self.replay.is_recording(),
            self.graphs.capture_mode,
            self.flags.rope_interleaved_legacy,
            kv,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rot,
            batch_size,
            max_ctx_len,
        ) {
            Ok(()) => true,
            Err(reason) => {
                reason.log_once();
                false
            }
        }
    }

    fn fa_pm_qprep_require_eager(&self) -> HipResult<()> {
        if self.graphs.capture_mode || self.replay.is_recording() {
            return Err(err(
                "eager only: capture and replay recording are unsupported",
            ));
        }
        if self.arch != "gfx1201" {
            return Err(err("requires gfx1201; no HIP fallback"));
        }
        Ok(())
    }

    /// Exact Q-prep primitive (production uses this same launch). `interleaved` is the f32
    /// projection `[rows,16,512]` (per head 256 q then 256 gate), `q16` the f16 (or Raw)
    /// `[rows,16,256]` output, `gate` the f32 `[rows,16,256]` output, `weight` the f32 `[256]`
    /// q-norm weight, `angles` the f32 `[rows,64]` table captured by
    /// [`Gpu::rope_partial_halfsplit_capture_f32_batched`] (`cos[0..32]`, `sin[32..64]`).
    /// Accepts 1..=8192 rows (production admission is 64..=8192 via
    /// [`Gpu::fa_prefill_pm_qprep_admitted`]). Eager only; no HIP fallback.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn fa_pm_qprep_exact(
        &mut self,
        interleaved: &GpuTensor,
        q16: &GpuTensor,
        gate: &GpuTensor,
        weight: &GpuTensor,
        angles: &GpuTensor,
        rows: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.fa_pm_qprep_require_eager()?;
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(err("rows must be in 1..=8192"));
        }
        if interleaved.dtype != DType::F32
            || gate.dtype != DType::F32
            || weight.dtype != DType::F32
            || angles.dtype != DType::F32
        {
            return Err(err("interleaved, gate, weight and angles must be F32"));
        }
        if !matches!(q16.dtype, DType::F16 | DType::Raw) {
            return Err(err("Q16 must be F16 or Raw"));
        }
        // rows <= 8192, so none of these products can overflow usize.
        require_span(interleaved, rows * INTERLEAVED_ROW * 4, "interleaved")?;
        require_span(q16, rows * ROW * 2, "Q16")?;
        require_span(gate, rows * ROW * 4, "gate")?;
        require_span(weight, HEAD_DIM * 4, "q-norm weight")?;
        require_span(angles, rows * ANGLE_STRIDE * 4, "angles")?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, SYMBOL)?;
        let mut args = kernargs(
            interleaved.buf.as_ptr(),
            q16.buf.as_ptr(),
            gate.buf.as_ptr(),
            weight.buf.as_ptr(),
            angles.buf.as_ptr(),
            rows,
            eps,
        );
        self.launch_blob_recorded(
            SYMBOL,
            grid(rows),
            BLOCK,
            0,
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_constants_match_certified_authoring_contract() {
        use peacemaker_isa::kernels::fa_qprep as author;
        assert_eq!(MODULE, author::MODULE);
        assert_eq!(SYMBOL, author::SYMBOL);
        assert_eq!(KERNARG_BYTES as u32, author::KERNARG_BYTES);
        assert_eq!(BLOCK[0], author::BLOCK);
        assert_eq!(HEADS as u32, author::HEADS);
        assert_eq!(HEAD_DIM as u32, author::HEAD_DIM);
        assert_eq!(N_ROT as u32, author::N_ROT);
        assert_eq!(ANGLE_STRIDE, N_ROT);
        assert_eq!(HEADS_PER_BLOCK as u32, author::BLOCK / 32);
        assert_eq!(author::BLOCKS_PER_ROW as usize * HEADS_PER_BLOCK, HEADS);
    }

    #[test]
    fn frozen_kernarg_abi_and_grid() {
        let args = kernargs(
            0x1000usize as *mut c_void,
            0x2000usize as *mut c_void,
            0x3000usize as *mut c_void,
            0x4000usize as *mut c_void,
            0x5000usize as *mut c_void,
            8191,
            1e-6,
        );
        assert_eq!(args.len(), 48);
        for (i, value) in [0x1000u64, 0x2000, 0x3000, 0x4000, 0x5000]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                u64::from_le_bytes(args[i * 8..i * 8 + 8].try_into().unwrap()),
                value
            );
        }
        assert_eq!(i32::from_le_bytes(args[40..44].try_into().unwrap()), 8191);
        assert_eq!(f32::from_le_bytes(args[44..48].try_into().unwrap()), 1e-6);
        assert_eq!(BLOCK, [128, 1, 1]);
        for rows in [1, 7, 8, 9, 63, 64, 65, 8191, 8192] {
            assert_eq!(grid(rows), [(rows * 4) as u32, 1, 1]);
        }
    }

    #[test]
    fn admission_requires_every_gate_in_order() {
        use FaPrefillPmQprepRefusal as R;
        let ok = |q: bool,
                  pm: bool,
                  arch: &str,
                  rec: bool,
                  cap: bool,
                  legacy: bool,
                  q8: bool,
                  shape: (usize, usize, usize),
                  rot: usize,
                  rows: usize,
                  ctx: usize| {
            Gpu::fa_prefill_pm_qprep_admission(
                q,
                pm,
                arch,
                rec,
                cap,
                legacy,
                q8.then_some(FaPrefillKvCodec::Q8_0),
                shape.0,
                shape.1,
                shape.2,
                rot,
                rows,
                ctx,
            )
        };
        let shape = (16, 2, 256);
        assert_eq!(
            ok(true, true, "gfx1201", false, false, false, true, shape, 64, 64, 64),
            Ok(())
        );
        assert_eq!(
            ok(true, true, "gfx1201", false, false, false, true, shape, 64, 8192, 262_144),
            Ok(())
        );
        assert_eq!(
            ok(false, true, "gfx1201", false, false, false, true, shape, 64, 64, 64),
            Err(R::Disabled)
        );
        assert_eq!(
            ok(true, false, "gfx1201", false, false, false, true, shape, 64, 64, 64),
            Err(R::PmDisabled)
        );
        for arch in ["gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown"] {
            assert_eq!(
                ok(true, true, arch, false, false, false, true, shape, 64, 64, 64),
                Err(R::InventoryUnavailable)
            );
        }
        assert_eq!(
            ok(true, true, "gfx1201", true, false, false, true, shape, 64, 64, 64),
            Err(R::Recording)
        );
        assert_eq!(
            ok(true, true, "gfx1201", false, true, false, true, shape, 64, 64, 64),
            Err(R::Capture)
        );
        assert_eq!(
            ok(true, true, "gfx1201", false, false, true, true, shape, 64, 64, 64),
            Err(R::LegacyRope)
        );
        assert_eq!(
            ok(true, true, "gfx1201", false, false, false, false, shape, 64, 64, 64),
            Err(R::KvFormat)
        );
        for bad in [(24, 4, 256), (16, 4, 256), (16, 2, 128)] {
            assert_eq!(
                ok(true, true, "gfx1201", false, false, false, true, bad, 64, 64, 64),
                Err(R::Shape)
            );
        }
        for rot in [0, 32, 128] {
            assert_eq!(
                ok(true, true, "gfx1201", false, false, false, true, shape, rot, 64, 64),
                Err(R::Rotary)
            );
        }
        for rows in [0, 1, 63, 8193] {
            assert_eq!(
                ok(true, true, "gfx1201", false, false, false, true, shape, 64, rows, 64),
                Err(R::Rows)
            );
        }
        for ctx in [0, 262_145] {
            assert_eq!(
                ok(true, true, "gfx1201", false, false, false, true, shape, 64, 64, ctx),
                Err(R::Context)
            );
        }
    }

    #[test]
    fn admission_follows_the_resolved_pm_flag_not_an_explicit_one() {
        use crate::prefill_route_capabilities::{enabled, PrefillRoute};
        use FaPrefillPmQprepRefusal as R;
        let shape = (16, 2, 256);
        let admit = |arch: &str, pm: Option<bool>, qprep: Option<bool>| {
            Gpu::fa_prefill_pm_qprep_admission(
                enabled(qprep, arch, PrefillRoute::FaPrefillPmQprep),
                enabled(pm, arch, PrefillRoute::FaPrefillPm),
                arch,
                false,
                false,
                false,
                Some(FaPrefillKvCodec::Q8_0),
                shape.0,
                shape.1,
                shape.2,
                64,
                4096,
                4096,
            )
        };
        // gfx1201: both unset resolve on (qualified), so the all-unset default is admitted.
        assert_eq!(admit("gfx1201", None, None), Ok(()));
        assert_eq!(admit("gfx1201", None, Some(true)), Ok(()));
        // PM 1 + Q-prep 1 is admitted, PM 1 + Q-prep unset follows qualification.
        assert_eq!(admit("gfx1201", Some(true), Some(true)), Ok(()));
        assert_eq!(admit("gfx1201", Some(true), None), Ok(()));
        // The fold nests beneath PM: an explicit PM 0 refuses it whatever Q-prep says.
        assert_eq!(admit("gfx1201", Some(false), None), Err(R::PmDisabled));
        assert_eq!(
            admit("gfx1201", Some(false), Some(true)),
            Err(R::PmDisabled)
        );
        // An explicit Q-prep 0 kills the fold whatever PM says.
        assert_eq!(admit("gfx1201", None, Some(false)), Err(R::Disabled));
        assert_eq!(admit("gfx1201", Some(true), Some(false)), Err(R::Disabled));
        // Other architectures: unset is off for both flags; an explicit pair lacks the inventory.
        for arch in ["gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown"] {
            assert_eq!(admit(arch, None, None), Err(R::Disabled), "{arch}");
            assert_eq!(admit(arch, None, Some(true)), Err(R::PmDisabled), "{arch}");
            assert_eq!(
                admit(arch, Some(true), Some(true)),
                Err(R::InventoryUnavailable),
                "{arch}"
            );
        }
    }

    #[test]
    fn fp8_codec_resolves_its_own_flags_and_defaults_on_where_qualified() {
        use crate::prefill_route_capabilities::{enabled_fp8_kv, PrefillRoute};
        use FaPrefillPmQprepRefusal as R;
        let shape = (16, 2, 256);
        let fp8 = Some(FaPrefillKvCodec::Fp8E4M3);
        let admit = |arch: &str, pm: Option<bool>, qprep: Option<bool>| {
            Gpu::fa_prefill_pm_qprep_admission(
                enabled_fp8_kv(qprep, arch, PrefillRoute::FaPrefillPmQprep),
                enabled_fp8_kv(pm, arch, PrefillRoute::FaPrefillPm),
                arch,
                false,
                false,
                false,
                fp8,
                shape.0,
                shape.1,
                shape.2,
                64,
                4096,
                4096,
            )
        };
        // Unset follows the fp8 qualification: on at exact gfx1201 for both keys.
        assert_eq!(admit("gfx1201", None, None), Ok(()));
        assert_eq!(admit("gfx1201", None, Some(true)), Ok(()));
        assert_eq!(admit("gfx1201", Some(false), None), Err(R::PmDisabled));
        assert_eq!(admit("gfx1201", Some(true), None), Ok(()));
        assert_eq!(admit("gfx1201", None, Some(false)), Err(R::Disabled));
        // Explicit opt-in of both keys admits; an explicit PM 0 or Q-prep 0 refuses.
        assert_eq!(admit("gfx1201", Some(true), Some(true)), Ok(()));
        assert_eq!(
            admit("gfx1201", Some(false), Some(true)),
            Err(R::PmDisabled)
        );
        assert_eq!(admit("gfx1201", Some(true), Some(false)), Err(R::Disabled));
        // Other architectures: unset is off for both flags; an explicit pair lacks the inventory.
        for arch in ["gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown"] {
            assert_eq!(admit(arch, None, None), Err(R::Disabled), "{arch}");
            assert_eq!(admit(arch, None, Some(true)), Err(R::PmDisabled), "{arch}");
            assert_eq!(
                admit(arch, Some(true), Some(true)),
                Err(R::InventoryUnavailable),
                "{arch}"
            );
        }
        let gate = |rec: bool,
                    cap: bool,
                    kv: Option<FaPrefillKvCodec>,
                    shape: (usize, usize, usize),
                    rot: usize,
                    rows: usize,
                    ctx: usize| {
            Gpu::fa_prefill_pm_qprep_admission(
                true, true, "gfx1201", rec, cap, false, kv, shape.0, shape.1, shape.2, rot, rows,
                ctx,
            )
        };
        assert_eq!(gate(false, false, fp8, shape, 64, 64, 64), Ok(()));
        assert_eq!(gate(true, false, fp8, shape, 64, 64, 64), Err(R::Recording));
        assert_eq!(gate(false, true, fp8, shape, 64, 64, 64), Err(R::Capture));
        assert_eq!(
            gate(false, false, None, shape, 64, 64, 64),
            Err(R::KvFormat)
        );
        assert_eq!(
            gate(false, false, fp8, (24, 4, 256), 64, 64, 64),
            Err(R::Shape)
        );
        assert_eq!(gate(false, false, fp8, shape, 32, 64, 64), Err(R::Rotary));
        assert_eq!(gate(false, false, fp8, shape, 64, 63, 64), Err(R::Rows));
        assert_eq!(
            gate(false, false, fp8, shape, 64, 64, 262_145),
            Err(R::Context)
        );
    }
}
