// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `fp8_gemm` generator as public building blocks.
//!
//! [`super::emit`], [`super::kloop::emit`], [`super::prologue::emit`] and [`super::fold::emit`] are
//! written over exactly these functions, so a composition that mirrors them (the `.rip` twin
//! `kernels/fp8_gemm.rip`) re-emits their bytes, and a composition that differs (a barrier moved,
//! the fold placed elsewhere, another weight-ring refill point) is checked by the same builder as
//! the production kernel.
//!
//! Unlike `iu4_gemm` and `mq6_i8_gemm`, this generator is an **untyped-builder** kernel (as
//! shipped): LDS slots, barriers, branches and the loop are `Builder` calls, and the epilogue is a
//! jump network of labels and branches. The split here follows that:
//!
//! - **LDS traffic stays with the caller.** Every block that touches LDS only *returns*
//!   [`Instruction`]s ([`publish_a_insns`], [`fragment_insns`], [`ratio_store_insn`],
//!   [`d_store_insn`], [`fold_row_group_load_insns`], [`fold_k128_token_load_insns`],
//!   [`fold_k128_pair_load_insns`]); the caller issues them with `Builder::ds_store(slot, insn)` /
//!   `ds_load(slot, insn)`, so every slot access, barrier transition and wait is the caller's
//!   statement, checked by the builder's slot machine and wait ledger.
//! - **Everything else emits straight into the builder** (VMEM fetches, address set-up, the WMMA
//!   steps, the fold arithmetic, the whole epilogue including the GDN-fused one).
//!
//! The kernel order, as `emit` composes it:
//!
//! ```text
//! lds slots A0 A1 R0 R1 D0 D1 (ids 0..6);  enable_delay_alu
//! prologue_front; s_cbranch_scc1 END; prologue_lanes
//! fetch_a(0, 0); stores of publish_a_insns(0) into slot 0; barrier [Ready(0)]
//! label BEGIN; block(next, first)             -- see below
//! trips_setup; s_cbranch_scc1 LOOP_END
//! loop LOOP { block; block; trip_end; s_cbranch_scc1 LOOP }
//! label LOOP_END; block(!next); wait_all; s_branch EPILOGUE
//! epilogue (label EPILOGUE ..); label END; s_endpgm
//! ```
//!
//! and one block (`first`: the loop's first block; plane `p = (first + 1) & 1`):
//!
//! ```text
//! weight_base; [first: fetch_w 0, 1]; fetch_a(0, 1); [next: fetch_next_ratio(1)]
//! units 0,1: steps (fragment loads of slot 0, 16 WMMAs each), fetch_w(u + 2)
//! stores of publish_a_insns(1) into A1; [next: ratio_store_insn(p) into R<p>,
//!     K128: d_prepare, d_store_insn(p) into D<p>, d_restore]
//! barrier [Retire(0), Ready(1), [next: Ready(2+p), K128: Ready(4+p)]]
//! [next: fetch_a(1, 0)]
//! units 2,3: steps (slot 1), [next: fetch_w(u + 14)]
//! [next: stores of publish_a_insns(0) into A0; fold (waits and loads per fold_*);
//!     barrier_signal [Retire(1), Retire(2+p), Ready(0), K128: Retire(4+p)]; barrier_wait]
//! [!next: barrier [Retire(1)]]
//! advance_block; wait_all
//! ```
use super::{epilogue as epi, fold, kloop, prologue, publish, Builder};
use crate::{insn::Instruction, KernelSpec, RegPlan};

pub use super::{ActScale, Epi, Spec, BEGIN, END, EPILOGUE, LOOP, LOOP_END};

/// The six LDS slots, in builder id order: `(name, base, bytes)`. Two A payload buffers (slot
/// `0`, `1`), two row-ratio planes `R` (`2`, `3`) and two per-token D-ratio planes `D` (`4`,
/// `5`; K128 only).
pub const LDS_SLOTS: [(&str, u32, u32); 6] = [
    ("A0", 0, 8192),
    ("A1", 8192, 8192),
    ("R0", 16384, 1024),
    ("R1", 17408, 1024),
    ("D0", 18432, 512),
    ("D1", 18944, 512),
];

/// A validated generator: the spec every block reads. The register map is fixed (the plan in
/// `mod.rs`), as are the tile geometry (256 rows x 128 tokens x 8 waves) and the LDS layout.
#[derive(Clone, Copy, Debug)]
pub struct Gen {
    spec: Spec,
}

impl Gen {
    /// The generator for `spec`. `Err` when `spec` has no generator (`Spec::validate`).
    pub fn new(spec: Spec) -> Result<Self, String> {
        spec.validate()?;
        Ok(Self { spec })
    }
    pub fn spec(&self) -> Spec {
        self.spec
    }
    /// The kernel header (symbol, kernargs, workgroup size, static LDS) of this spec.
    pub fn kernel_spec(&self) -> KernelSpec {
        let spec = self.spec;
        KernelSpec {
            kernel_id: "fp8_gemm".into(),
            variant: spec.variant(),
            arch: spec.arch,
            symbol: spec.symbol(),
            kernargs: spec.kernargs(),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: 256,
            group_segment_fixed_size: if spec.epi == Epi::QkvzaGdn {
                super::gdn_epilogue::FIXED_LDS
            } else {
                0
            },
            wave32: true,
            cu_mode: false,
        }
    }
    /// The kernel's register plan.
    pub fn plan(&self) -> Result<RegPlan, String> {
        super::plan()
    }
    /// `(base, bytes)` of the LDS slot named `name` (`A0 A1 R0 R1 D0 D1`).
    pub fn slot(&self, name: &str) -> Option<(u32, u32)> {
        LDS_SLOTS
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|&(_, base, bytes)| (base, bytes))
    }
}

/// Kernel arguments (`s_load_b256` x3), the banded CTA raster, the tile origin and the trip
/// scalars; ends with the compare whose SCC the caller branches on (`s_cbranch_scc1 END`) when the
/// tile starts at or past `M`.
pub fn prologue_front(b: &mut Builder, g: &Gen) -> Result<(), String> {
    prologue::front(b, g.spec)
}

/// Hoisted lane addresses, the block scalars and the weight/ratio bases (after the exit branch).
pub fn prologue_lanes(b: &mut Builder, _g: &Gen) -> Result<(), String> {
    prologue::lanes(b)
}

/// The scalar offset and the four `buffer_load_b64`s of an A slab: `kb_offset` K128 blocks ahead
/// of the current one, slab `slab` (`0` or `1`), into the prefetch registers `v168..v175`.
pub fn fetch_a(b: &mut Builder, _g: &Gen, kb_offset: u32, slab: u32) -> Result<(), String> {
    publish::fetch_a(b, kb_offset, slab)
}

/// The four `ds_store_b64`s of the prefetched A payload into slot `slot` (`0` or `1`).
pub fn publish_a_insns(_g: &Gen, slot: usize) -> Result<Vec<Instruction>, String> {
    slot_of("publish_a_insns", slot, 2)?;
    publish::publish_a_insns(slot)
}

/// The weight base `s90` of the current block.
pub fn weight_base(b: &mut Builder, _g: &Gen) -> Result<(), String> {
    publish::weight_base(b)
}

/// Weight ring unit `unit` (`0..8`; the ring holds two, selected by `unit & 1`): four
/// `buffer_load_b128`, one per row16 tile. Units `u + 2` refill after unit `u` is consumed; units
/// `u + 14` are the next block's, `0x8000` bytes on.
pub fn fetch_w(b: &mut Builder, _g: &Gen, unit: usize) -> Result<(), String> {
    publish::fetch_w(b, unit)
}

/// The next block's row-ratio word (`s92` base, `v184`).
pub fn fetch_next_ratio(b: &mut Builder, _g: &Gen, next_kb: u32) -> Result<(), String> {
    publish::fetch_next_ratio(b, next_kb)
}

/// The row-ratio store of plane `plane` (`0` or `1`; into slot `2 + plane`).
pub fn ratio_store_insn(_g: &Gen, plane: usize) -> Result<Instruction, String> {
    slot_of("ratio_store_insn", plane, 2)?;
    publish::ratio_store_insn(plane)
}

/// K128 only: the next block's per-token D ratio, from its loads to the narrowed exec mask. The
/// caller issues [`d_store_insn`] and then [`d_restore`].
pub fn d_prepare(b: &mut Builder, g: &Gen, next_kb: u32) -> Result<(), String> {
    k128("d_prepare", g)?;
    publish::d_prepare(b, next_kb)
}

/// K128 only: the D-ratio store of plane `plane` (into slot `4 + plane`).
pub fn d_store_insn(g: &Gen, plane: usize) -> Result<Instruction, String> {
    k128("d_store_insn", g)?;
    slot_of("d_store_insn", plane, 2)?;
    publish::d_store_insn(plane)
}

/// K128 only: restore the exec mask [`d_prepare`] narrowed.
pub fn d_restore(b: &mut Builder, g: &Gen) -> Result<(), String> {
    k128("d_restore", g)?;
    publish::d_restore(b)
}

/// The four A fragment loads (`ds_load_b64`, one per 16-token tile) of K16 step `step` (`0..4`:
/// `2 * (unit % 2) + st`) from slot `slot`.
pub fn fragment_insns(_g: &Gen, slot: usize, step: usize) -> Result<Vec<Instruction>, String> {
    slot_of("fragment_insns", slot, 2)?;
    if step >= 4 {
        return Err(format!("fragment_insns: step {step} is not 0..4"));
    }
    publish::fragment_insns(slot, step)
}

/// The `i`-th (`0..16`: row tile `i / 4`, token tile `i % 4`) `v_wmma_f32_16x16x16_fp8_fp8` of K16
/// step `st` (`0..2`) of K32 unit `unit` (`0..4`). With `first` the accumulator starts from the
/// instruction's inline zero.
pub fn mma_insn(
    g: &Gen,
    unit: usize,
    st: usize,
    first: bool,
    i: usize,
) -> Result<Instruction, String> {
    if unit >= 4 || st >= 2 || i >= 16 {
        return Err(format!(
            "mma_insn: unit {unit}, step {st}, index {i} are outside 0..4, 0..2, 0..16"
        ));
    }
    kloop::mma_insn(g.spec, unit, st, first, i)
}

/// Issue [`mma_insn`].
pub fn mma(
    b: &mut Builder,
    g: &Gen,
    unit: usize,
    st: usize,
    first: bool,
    i: usize,
) -> Result<(), String> {
    b.push(mma_insn(g, unit, st, first, i)?)
}

/// Row fold: the two `ds_load_b128`s of row group `i`'s (`0..4`) eight ratios from plane `plane`
/// (slot `2 + plane`).
pub fn fold_row_group_load_insns(
    g: &Gen,
    plane: usize,
    i: usize,
) -> Result<[Instruction; 2], String> {
    row("fold_row_group_load_insns", g)?;
    slot_of("fold_row_group_load_insns", plane, 2)?;
    group("fold_row_group_load_insns", i)?;
    fold::row_group_load_insns(plane, i)
}

/// Row fold: the 16 dual-mul packets of row group `i` (after its loads have landed).
pub fn fold_row_group_scale(b: &mut Builder, g: &Gen, i: usize) -> Result<(), String> {
    row("fold_row_group_scale", g)?;
    group("fold_row_group_scale", i)?;
    fold::row_group_scale(b, i)
}

/// K128 fold: the four per-token D-ratio loads from plane `plane` (slot `4 + plane`).
pub fn fold_k128_token_load_insns(g: &Gen, plane: usize) -> Result<Vec<Instruction>, String> {
    k128("fold_k128_token_load_insns", g)?;
    slot_of("fold_k128_token_load_insns", plane, 2)?;
    fold::k128_token_load_insns(plane)
}

/// K128 fold: the two row-ratio loads of pair `(i, e)` (`i` in `0..4`, `e` even in `0..8`) from
/// plane `plane` (slot `2 + plane`).
pub fn fold_k128_pair_load_insns(
    g: &Gen,
    plane: usize,
    i: usize,
    e: usize,
) -> Result<[Instruction; 2], String> {
    k128("fold_k128_pair_load_insns", g)?;
    slot_of("fold_k128_pair_load_insns", plane, 2)?;
    pair("fold_k128_pair_load_insns", i, e)?;
    fold::k128_pair_load_insns(plane, i, e)
}

/// K128 fold: the four token chains of pair `(i, e)` (after its loads have landed).
pub fn fold_k128_pair_scale(b: &mut Builder, g: &Gen, i: usize, e: usize) -> Result<(), String> {
    k128("fold_k128_pair_scale", g)?;
    pair("fold_k128_pair_scale", i, e)?;
    fold::k128_pair_scale(b, i, e)
}

/// The block counter (`s85`) moves on (the last statement of a block before its `wait_all`).
pub fn advance_block(b: &mut Builder, _g: &Gen) -> Result<(), String> {
    kloop::advance_block(b)
}

/// Retire every outstanding memory operation (the builder's ledger-wide wait).
pub fn wait_all(b: &mut Builder) -> Result<(), String> {
    b.wait_all()
}

/// The rolled loop's trip count `(K/128 - 2) / 2` into `s86` and the compare with zero; the
/// caller branches on it (`s_cbranch_scc1 LOOP_END`) when there is no full trip.
pub fn trips_setup(b: &mut Builder, _g: &Gen) -> Result<(), String> {
    kloop::trips_setup(b)
}

/// The loop's trip end: the trip counter drops by one and the compare with zero; the caller
/// branches on it (`s_cbranch_scc1 LOOP`) back to the loop head.
pub fn trip_end(b: &mut Builder, _g: &Gen) -> Result<(), String> {
    kloop::trip_end(b)
}

/// Everything from the [`EPILOGUE`] label (this places it) to the kernel exit: the ratio loads
/// and scaling, the per-epilogue stores and, for `Epi::QkvzaGdn`, the GDN preparation after a
/// branch to [`END`]'s predecessor. The epilogue is one jump network of labels and branches, so
/// it stays a single block. The caller places [`END`] and `s_endpgm` after it.
pub fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    epi::emit(b, g.spec)
}

fn slot_of(what: &str, v: usize, n: usize) -> Result<(), String> {
    if v < n {
        Ok(())
    } else {
        Err(format!("{what}: {v} is not 0..{n}"))
    }
}
fn group(what: &str, i: usize) -> Result<(), String> {
    slot_of(what, i, 4)
}
fn pair(what: &str, i: usize, e: usize) -> Result<(), String> {
    group(what, i)?;
    if e < 8 && e % 2 == 0 {
        Ok(())
    } else {
        Err(format!("{what}: pair element {e} is not even in 0..8"))
    }
}
fn k128(what: &str, g: &Gen) -> Result<(), String> {
    if g.spec.act_scale == ActScale::K128 {
        Ok(())
    } else {
        Err(format!("{what}: only the K128-scaled kernel has a D ratio"))
    }
}
fn row(what: &str, g: &Gen) -> Result<(), String> {
    if g.spec.act_scale == ActScale::Row {
        Ok(())
    } else {
        Err(format!(
            "{what}: only the Row-scaled kernel folds row groups"
        ))
    }
}
