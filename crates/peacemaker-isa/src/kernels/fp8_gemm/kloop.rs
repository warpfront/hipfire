// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::{fold, publish, Builder, Instruction, Spec, BEGIN, EPILOGUE, LOOP, LOOP_END};
use crate::kernels::common::{op, sop};
use crate::{insn::Wmma, lds::Transition, reg::V};

/// The `i`-th (`0..16`: row tile `i / 4`, token tile `i % 4`) `v_wmma_f32_16x16x16_fp8_fp8` of K16
/// step `st` (`0..2`) of K32 unit `unit` (`0..4`; the weight ring holds two). `first` starts the
/// accumulator from the instruction's inline zero instead of the accumulator itself.
pub(super) fn mma_insn(
    spec: Spec,
    unit: usize,
    st: usize,
    first: bool,
    i: usize,
) -> Result<Instruction, String> {
    let (rt, tt) = (i / 4, i % 4);
    let dst = (rt * 4 + tt) * 8;
    let w = 128 + ((unit & 1) * 16 + rt * 4 + st * 2) as u8;
    let x = 160 + (tt * 2) as u8;
    Wmma::fp8(
        spec.arch,
        V::<8>(dst as u8),
        V::<2>(w),
        V::<2>(x),
        if first { None } else { Some(V::<8>(dst as u8)) },
    )
}

fn step(
    b: &mut Builder,
    spec: Spec,
    unit: usize,
    st: usize,
    first: bool,
    slot: usize,
) -> Result<(), String> {
    publish::fragments(b, slot, 2 * (unit % 2) + st)?;
    for i in 0..16 {
        b.push(mma_insn(spec, unit, st, first, i)?)?;
    }
    Ok(())
}

/// The barrier transitions of one block's B1: retire A0, publish A1 and (with a next block) the
/// ratio planes of plane `p`.
pub(super) fn b1_transitions(spec: Spec, next: bool, p: usize) -> Vec<Transition> {
    let mut b1 = vec![Transition::Retire(0), Transition::Ready(1)];
    if next {
        b1.push(Transition::Ready(2 + p));
        if spec.act_scale == super::ActScale::K128 {
            b1.push(Transition::Ready(4 + p));
        }
    }
    b1
}

/// The transitions of one block's B2 (signal + wait): retire A1 and the ratio planes, publish the
/// next A0.
pub(super) fn b2_transitions(spec: Spec, p: usize) -> Vec<Transition> {
    let mut b2 = vec![
        Transition::Retire(1),
        Transition::Retire(2 + p),
        Transition::Ready(0),
    ];
    if spec.act_scale == super::ActScale::K128 {
        b2.push(Transition::Retire(4 + p));
    }
    b2
}

/// On entry A0 is Published. A1's VMEM payload is fetched before slab 0,
/// published after all slab-0 fragment readers, then a uniform barrier retires
/// A0. Metadata for b+1 is published in that same barrier. The final block
/// omits next-block prefetch and fold. No packed weight decode is performed.
fn block(b: &mut Builder, spec: Spec, next: bool, first: bool) -> Result<(), String> {
    publish::weight_base(b)?;
    if first {
        publish::fetch_w(b, 0)?;
        publish::fetch_w(b, 1)?;
    }
    publish::fetch_a(b, 0, 1)?;
    if next {
        publish::fetch_next_ratio(b, 1)?;
    }
    for u in 0..2 {
        for st in 0..2 {
            step(b, spec, u, st, first && u == 0 && st == 0, 0)?;
        }
        // Two-unit 16-VGPR ring: the unit just consumed is dead.
        publish::fetch_w(b, u + 2)?;
    }
    publish::publish_a(b, 1)?;
    if next {
        publish::publish_next_ratios(b, spec, 1, ((first as usize) + 1) & 1)?;
    }
    let p = ((first as usize) + 1) & 1;
    b.barrier(&b1_transitions(spec, next, p))?;
    if next {
        publish::fetch_a(b, 1, 0)?;
    }
    for u in 2..4 {
        for st in 0..2 {
            step(b, spec, u, st, false, 1)?;
        }
        // The retired K32 unit is free while the other unit still feeds WMMA.
        // W for the next K128 is 0x8000 bytes beyond this block's base.
        if next {
            publish::fetch_w(b, u + 14)?;
        }
    }
    if next {
        // The next A0 payload was prefetched before slab 1. Its stores are
        // deliberately delayed until all A1 reads have issued.
        publish::publish_a(b, 0)?;
        fold::emit(b, spec, p)?;
        b.barrier_signal(&b2_transitions(spec, p))?;
        b.barrier_wait()?;
    } else {
        b.barrier(&[Transition::Retire(1)])?;
    }
    advance_block(b)?;
    b.wait_all()?;
    Ok(())
}

/// The block counter (`s85`) moves on.
pub(super) fn advance_block(b: &mut Builder) -> Result<(), String> {
    sop(b, "s_add_co_i32 s85, s85, 1", &[85], &[85])
}

/// Trip count `(K/128 - 2) / 2` of the rolled loop into `s86`, and the compare with zero (no full
/// trip). The caller branches on it to the loop end.
pub(super) fn trips_setup(b: &mut Builder) -> Result<(), String> {
    sop(b, "s_add_co_i32 s86, s84, -2", &[86], &[84])?;
    sop(b, "s_lshr_b32 s86, s86, 1", &[86], &[86])?;
    sop(b, "s_cmp_eq_u32 s86, 0", &[], &[86])
}

/// The loop's trip end: the trip counter drops by one and the compare with zero. The caller
/// branches on it back to the loop head.
pub(super) fn trip_end(b: &mut Builder) -> Result<(), String> {
    sop(b, "s_add_co_i32 s86, s86, -1", &[86], &[86])?;
    sop(b, "s_cmp_lg_u32 s86, 0", &[], &[86])
}

pub(super) fn emit(b: &mut Builder, spec: Spec) -> Result<(), String> {
    b.label(BEGIN)?;
    // C=0 is the hardware WMMA's inline operand only for the first K16;
    // subsequent instructions carry the same 128 VGPRs as their SrcC.
    block(b, spec, true, true)?;
    trips_setup(b)?;
    op(b, format!("s_cbranch_scc1 {LOOP_END}"), &[], &[])?;
    b.loop_(LOOP, |b| {
        block(b, spec, true, false)?;
        block(b, spec, true, false)?;
        trip_end(b)?;
        op(b, format!("s_cbranch_scc1 {LOOP}"), &[], &[])
    })?;
    b.label(LOOP_END)?;
    block(b, spec, false, false)?;
    // Final global values are read only after all WMMA users of the fragment
    // ring have retired; no descriptor or LDS payload survives the epilogue.
    b.wait_all()?;
    op(b, format!("s_branch {EPILOGUE}"), &[], &[])
}
