// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The building blocks of the `fa_prefill` generators (the Q8_0 direct and preconvert exports and the
//! native-fp8 direct export), at the granularity of their own emit functions.
//!
//! [`super::emit`] and [`super::emit_preconvert`] are written over exactly these blocks, so a script that
//! calls them in the same order re-emits the production kernel's bytes. The generator is an
//! untyped-`Builder` kernel: the LDS slots (`k_plane`/`v_plane`, or the single reused `stage`), every
//! barrier, every branch and the tile loop are `Builder` calls the caller makes itself, checked by the
//! builder's slot machine, wait ledger and loop fixpoint. The blocks are everything between them:
//!
//! ```text
//! slots:     Gen::lds_slots()            -> Builder::lds_slot
//! prologue:  prologue                    (entry block, guards, group bounds, Q fragments, online state)
//! tile loop: loop_(LOOP) {
//!   tile_setup
//!   Q8_0:  fill ; label FILL_DONE ; barrier [Ready K, Ready V] ; label QK
//!          skip_if_helper(TURN) ; qk ; softmax_sub 0..4 ; label TURN
//!          barrier [Retire K, Retire V] ; next_tile
//!   FP8:   fill_k ; label KFILL_DONE ; barrier [Ready stage] ; label KREADY
//!          skip_if_helper(QK_DONE) ; qk ; label QK_DONE ; barrier [Retire stage]
//!          fill_v ; label VFILL_DONE ; barrier [Ready stage] ; label VREADY
//!          skip_if_helper(TURN) ; softmax_sub 0..4 ; label TURN ; barrier [Retire stage] ; next_tile }
//! epilogue:  epilogue                    (label EPI .. label END, s_endpgm)
//! ```
//!
//! The preconvert export is one block, [`preconvert`].

use super::{
    admit, FaKvCodec, FaPrefillSpec, KernargLayout, KernelSpec, DIRECT_BLOCK, DIRECT_KERNARG_BYTES,
    KERNEL_ID, KERNEL_ID_FP8, PRECONVERT_BLOCK, PRECONVERT_KERNARG_BYTES, R, SYMBOL_DIRECT,
    SYMBOL_FP8_DIRECT, SYMBOL_PRECONVERT,
};
use crate::kernels::common::sop;
use crate::plan::Access;
use crate::{Arch, Builder, RegPlan};

pub const LOOP: &str = super::D_LOOP;
pub const FILL_DONE: &str = super::D_FILL_DONE;
pub const KFILL_DONE: &str = super::D_KFILL_DONE;
pub const KREADY: &str = super::D_KREADY;
pub const QK: &str = super::D_QK;
pub const QK_DONE: &str = super::D_QK_DONE;
pub const VFILL_DONE: &str = super::D_VFILL_DONE;
pub const VREADY: &str = super::D_VREADY;
pub const TURN: &str = super::D_TURN;
pub const EPI: &str = super::D_EPI;
pub const END: &str = super::D_END;
/// The preconvert export's end label.
pub const PRECONVERT_END: &str = super::P_END;

/// LDS slot ids of the Q8_0 direct kernel (`k_plane`, `v_plane`) and of the fp8 one (`stage`).
pub const SLOT_K: usize = super::L_K;
pub const SLOT_V: usize = super::L_V;
pub const SLOT_STAGE: usize = super::L_STAGE;

/// The validated generator of one spec on one arch: what every block reads.
#[derive(Clone, Copy, Debug)]
pub struct Gen {
    arch: Arch,
    spec: FaPrefillSpec,
}

impl Gen {
    /// The gate of `emit`: gfx1201 and exactly `(16, 2, 256)`.
    pub fn new(arch: Arch, spec: FaPrefillSpec) -> Result<Gen, String> {
        admit(arch, &spec)?;
        Ok(Gen { arch, spec })
    }

    pub fn spec(&self) -> FaPrefillSpec {
        self.spec
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    pub fn fp8(&self) -> bool {
        self.spec.kv == FaKvCodec::Fp8E4M3
    }

    /// The direct export's `KernelSpec` (60-byte ABI, 160 threads).
    pub fn kernel_spec(&self) -> KernelSpec {
        let kernargs = KernargLayout::new(DIRECT_KERNARG_BYTES)
            .pointer_access("q16", 0, Access::ReadOnly)
            .pointer_access("k_cache", 8, Access::ReadOnly)
            .pointer_access("v_cache", 16, Access::ReadOnly)
            .pointer_access("out", 24, Access::WriteOnly)
            .pointer_access("positions", 32, Access::ReadOnly)
            .hidden("n_heads", 40, 4, "by_value")
            .hidden("n_kv_heads", 44, 4, "by_value")
            .hidden("head_dim", 48, 4, "by_value")
            .hidden("batch_size", 52, 4, "by_value")
            .hidden("scale_attn", 56, 4, "by_value");
        KernelSpec {
            kernel_id: (if self.fp8() { KERNEL_ID_FP8 } else { KERNEL_ID }).into(),
            variant: "direct".into(),
            arch: self.arch,
            symbol: (if self.fp8() {
                SYMBOL_FP8_DIRECT
            } else {
                SYMBOL_DIRECT
            })
            .into(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: DIRECT_BLOCK as u16,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        }
    }

    /// The direct export's register plan.
    pub fn plan(&self) -> Result<RegPlan, String> {
        super::plan_direct(self.fp8())
    }

    /// The preconvert export's `KernelSpec` (40-byte ABI, 128 threads).
    pub fn preconvert_kernel_spec(&self) -> KernelSpec {
        let kernargs = KernargLayout::new(PRECONVERT_KERNARG_BYTES)
            .pointer_access("q", 0, Access::ReadOnly)
            .pointer_access("q16", 8, Access::WriteOnly)
            .pointer_access("signs1", 16, Access::ReadOnly)
            .pointer_access("signs2", 24, Access::ReadOnly)
            .hidden("batch_size", 32, 4, "by_value")
            .hidden("do_rotate", 36, 4, "by_value");
        KernelSpec {
            kernel_id: KERNEL_ID.into(),
            variant: "preconvert".into(),
            arch: self.arch,
            symbol: SYMBOL_PRECONVERT.into(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: PRECONVERT_BLOCK as u16,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        }
    }

    /// The preconvert export's register plan.
    pub fn preconvert_plan(&self) -> Result<RegPlan, String> {
        super::plan_preconvert()
    }

    /// The LDS slots in id order: `(name, base, bytes)`. Q8_0: the K plane then the V plane (32 KiB
    /// each); fp8: one 32-KiB stage reused sequentially for K then V.
    pub fn lds_slots(&self) -> Vec<(&'static str, u32, u32)> {
        if self.fp8() {
            vec![("stage", 0, 32_768)]
        } else {
            vec![("k_plane", 0, 32_768), ("v_plane", 32_768, 32_768)]
        }
    }
}

fn q8(what: &str, g: &Gen) -> R {
    if g.fp8() {
        Err(format!(
            "{what}: only the Q8_0 kernel fills K and V jointly"
        ))
    } else {
        Ok(())
    }
}

fn fp8(what: &str, g: &Gen) -> R {
    if g.fp8() {
        Ok(())
    } else {
        Err(format!(
            "{what}: only the fp8 kernel fills K and V separately"
        ))
    }
}

/// Entry block: guards, scalar launch state, the query row and head of this lane, the group bounds,
/// the Q fragments and the fresh online state. Branches to [`END`] on a launch the guards refuse and
/// to [`EPI`] when no tile is live.
pub fn prologue(b: &mut Builder, g: &Gen) -> R {
    super::prologue(b, g.fp8())
}

/// Per-tile scalar set-up: the remaining-key counts and the K/V tile bases (`ktile * 544` for Q8_0,
/// `ktile * 516` for fp8).
pub fn tile_setup(b: &mut Builder, g: &Gen) -> R {
    super::tile_setup(b, g.fp8())
}

/// Q8_0: the cooperative K and V fill of one KT64 tile (waves 0..3; wave 4 branches to [`FILL_DONE`]).
pub fn fill(b: &mut Builder, g: &Gen) -> R {
    q8("fill", g)?;
    super::fill(b, false)
}

/// fp8: the cooperative K fill into the single stage.
pub fn fill_k(b: &mut Builder, g: &Gen) -> R {
    fp8("fill_k", g)?;
    super::fill_k_fp8(b)
}

/// fp8: the cooperative V fill into the single stage (after the K stage was retired).
pub fn fill_v(b: &mut Builder, g: &Gen) -> R {
    fp8("fill_v", g)?;
    super::fill_v_fp8(b)
}

/// The QK product of one tile: fragment loads from the published K and the four score accumulators.
pub fn qk(b: &mut Builder, g: &Gen) -> R {
    super::qk(b, g.fp8())
}

/// One KT64 key subtile's softmax and PV (`sub` in `0..4`).
pub fn softmax_sub(b: &mut Builder, g: &Gen, sub: u8) -> R {
    if sub >= 4 {
        return Err(format!("softmax_sub: subtile {sub} is outside 0..4"));
    }
    super::softmax_sub(b, sub, g.fp8())
}

/// `s_cmp_ge_u32 wave, 4` + `s_cbranch_scc1 label`: the helper wave (wave 4) skips to `label`.
pub fn skip_if_helper(b: &mut Builder, label: &str) -> R {
    sop(
        b,
        format!("s_cmp_ge_u32 s{}, 4", super::WAVE),
        &[],
        &[super::WAVE],
    )?;
    super::op(b, format!("s_cbranch_scc1 {label}"), &[], &[])
}

/// The tile advance and the loop back edge: `ktile += 64`, then back to [`LOOP`] while `ktile <= gmax`.
pub fn next_tile(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_add_co_i32 s{}, s{}, 64", super::KT, super::KT),
        &[super::KT],
        &[super::KT],
    )?;
    sop(
        b,
        format!("s_cmp_gt_i32 s{}, s{}", super::KT, super::GMAX),
        &[],
        &[super::KT, super::GMAX],
    )?;
    super::op(b, format!("s_cbranch_scc0 {LOOP}"), &[], &[])
}

/// Everything from label [`EPI`]: `out = O * (l > 0 ? 1/l : 0)` for live queries, the stores, the
/// wait, label [`END`] and `s_endpgm`.
pub fn epilogue(b: &mut Builder, g: &Gen) -> R {
    super::epilogue(b, g.fp8())
}

/// The whole preconvert export body (f32 Q to f16 Q16, one wave per `(row, head)`), through `s_endpgm`.
pub fn preconvert(b: &mut Builder, g: &Gen) -> R {
    let _ = g;
    super::preconvert_body(b)
}
