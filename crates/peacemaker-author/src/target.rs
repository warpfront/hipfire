// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Targets are types. A kernel names its target once (`Workgroup<Gfx1151, _>`)
//! and every capability it uses is a trait bound that only targets with a
//! certified lowering implement, so a missing capability is a rustc error
//! rather than a run-time `Err` from the builder.
//!
//! M0 covers the three bench architectures (RDNA3 gfx1100, RDNA3.5 gfx1151,
//! RDNA4 gfx1201). gfx942 and its FNUZ fp8 type are a documented future
//! extension (design §4.1, §4.4), not part of this core.

mod sealed {
    pub trait Sealed {}
}

/// The memory counter that retires LDS (DS) operations on a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdsCounter {
    /// gfx11 `LGKMcnt` (shared with SMEM and GDS).
    Lgkm,
    /// gfx12 `DScnt`.
    Ds,
}

/// A target's wait-counter model.
pub trait WaitModel: sealed::Sealed + Copy + 'static {
    /// The counter an LDS store drain waits on.
    const LDS: LdsCounter;
}
/// gfx11 counters: VMcnt, LGKMcnt, EXPcnt, VScnt.
#[derive(Clone, Copy, Debug)]
pub enum Gfx11Waits {}
/// gfx12 counters: LOADcnt, STOREcnt, DScnt, KMcnt.
#[derive(Clone, Copy, Debug)]
pub enum Gfx12Waits {}
impl sealed::Sealed for Gfx11Waits {}
impl sealed::Sealed for Gfx12Waits {}
impl WaitModel for Gfx11Waits {
    const LDS: LdsCounter = LdsCounter::Lgkm;
}
impl WaitModel for Gfx12Waits {
    const LDS: LdsCounter = LdsCounter::Ds;
}

/// A target's workgroup barrier.
pub trait BarrierModel: sealed::Sealed + Copy + 'static {}
/// One `s_barrier` (gfx11).
#[derive(Clone, Copy, Debug)]
pub enum Full {}
/// `s_barrier_signal -1` / `s_barrier_wait 0xffff` (gfx12).
#[derive(Clone, Copy, Debug)]
pub enum Split {}
impl sealed::Sealed for Full {}
impl sealed::Sealed for Split {}
impl BarrierModel for Full {}
impl BarrierModel for Split {}

/// A GPU target. Sealed: targets enter only with a certified lowering table.
pub trait Target: sealed::Sealed + Copy + 'static {
    /// The backend's architecture name (`gfx1100`, …); `Workgroup::new`
    /// refuses a backend built for another architecture.
    const NAME: &'static str;
    /// LDS bytes one workgroup can address.
    const LDS_BYTES: u32;
    type Waits: WaitModel;
    type Barrier: BarrierModel;
}

/// RDNA3 (Navi 31).
#[derive(Clone, Copy, Debug)]
pub struct Gfx1100;
/// RDNA3.5 (Strix Halo).
#[derive(Clone, Copy, Debug)]
pub struct Gfx1151;
/// RDNA4 (Navi 48).
#[derive(Clone, Copy, Debug)]
pub struct Gfx1201;
impl sealed::Sealed for Gfx1100 {}
impl sealed::Sealed for Gfx1151 {}
impl sealed::Sealed for Gfx1201 {}
impl Target for Gfx1100 {
    const NAME: &'static str = "gfx1100";
    const LDS_BYTES: u32 = 65536;
    type Waits = Gfx11Waits;
    type Barrier = Full;
}
impl Target for Gfx1151 {
    const NAME: &'static str = "gfx1151";
    const LDS_BYTES: u32 = 65536;
    type Waits = Gfx11Waits;
    type Barrier = Full;
}
impl Target for Gfx1201 {
    const NAME: &'static str = "gfx1201";
    const LDS_BYTES: u32 = 65536;
    type Waits = Gfx12Waits;
    type Barrier = Split;
}

/// Split-barrier capability: `Workgroup::signal` / `Workgroup::wait_arrived`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no split barrier",
    note = "gfx11 has one `s_barrier`; use `Workgroup::barrier`, which lowers to signal+wait where the target splits it"
)]
pub trait SplitBarrier: Target {}
impl<T: Target<Barrier = Split>> SplitBarrier for T {}

/// Native signed-nibble integer WMMA (`iu4`, i32 accumulate), 16×16 output:
/// `v_wmma_i32_16x16x16_iu4` (gfx11) or `v_wmma_i32_16x16x32_iu4` (gfx12).
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no certified lowering for MMA iu4 x iu4 -> i32",
    note = "add a row to the {Self} lowering table, or bound the kernel on a different target"
)]
pub trait MmaIu4: Target {}
impl MmaIu4 for Gfx1100 {}
impl MmaIu4 for Gfx1151 {}
impl MmaIu4 for Gfx1201 {}
