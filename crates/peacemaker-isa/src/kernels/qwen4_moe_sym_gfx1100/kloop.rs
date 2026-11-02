// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 one-epoch K loop (B4-B baseline).
//!
//! Per iteration: scalar epoch offsets from the epoch index, bounded loads of
//! exactly this epoch, compute/fold ([`super::fold::compute`]), then a full
//! drain of the wait ledger before the back edge. No prefetch, no double
//! buffering, no load past the last epoch. The descriptors never move: every
//! load is `voffset + immediate` against a descriptor whose `num_records` is
//! the exact tensor extent, so each address is bounds-checked by the hardware
//! as well as proven by the admission check.
//!
//! The index `e` runs `0..E` ascending (`E = K/128`). `Negative::ReversedEpoch`
//! runs `E-1..0` with the matching operands and scales, which changes only the
//! F32 fold order.
use super::*;
use peacemaker_author::{lds::Lowering, Transitions};

/// A workgroup barrier that carries no LDS transitions (this kernel owns no LDS region): a pure
/// `s_barrier`, so the two waves of a CTA enter every epoch together. No LDS instruction is
/// emitted for it, so there is no LDS access of any kind to bound.
struct Lockstep;
impl Transitions for Lockstep {
    type After = ();
    fn lower(&self, _l: &mut Lowering) -> Result<(), String> {
        Ok(())
    }
    fn after(self) {}
}

fn cmp<T: Gfx11>(wg: &mut Wg<'_, T>, text: String, uses: &[u8]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

/// Scalar byte offsets of epoch `EIDX` (code, scale, activation) and the
/// three per-epoch VGPR offsets they feed.
pub(super) fn epoch_offsets(b: &mut Builder, g: &Gen) -> Result<(), String> {
    match (g.spec.codec, g.spec.kind) {
        (Codec::W3, _) => {
            sop(
                b,
                format!("s_mul_i32 s{COFF}, s{EIDX}, 48"),
                &[COFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{SOFF}, s{EIDX}, 1"),
                &[SOFF],
                &[EIDX],
            )?;
        }
        (Codec::W2 | Codec::E8p, _) => {
            sop(
                b,
                format!("s_lshl_b32 s{COFF}, s{EIDX}, 5"),
                &[COFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{SOFF}, s{EIDX}, 1"),
                &[SOFF],
                &[EIDX],
            )?;
        }
        (Codec::Mq4, Kind::Down) => {
            // QT53 group `[f16 sc, f16 zp][64 nibble bytes]`: codes e*68+4, scale e*68.
            sop(
                b,
                format!("s_mul_i32 s{SOFF}, s{EIDX}, 0x44"),
                &[SOFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{SOFF}, 4"),
                &[COFF],
                &[SOFF],
            )?;
        }
        (Codec::Mq4, Kind::GateUp) => {
            // QT44 group `[sc0, zp0, sc1, zp1][128 nibble bytes]` per two epochs:
            // codes (e>>1)*136 + 8 + (e&1)*64, scale (e>>1)*136 + (e&1)*4.
            sop(b, format!("s_lshr_b32 s{T0}, s{EIDX}, 1"), &[T0], &[EIDX])?;
            sop(b, format!("s_mul_i32 s{T0}, s{T0}, 0x88"), &[T0], &[T0])?;
            sop(b, format!("s_and_b32 s{T1}, s{EIDX}, 1"), &[T1], &[EIDX])?;
            sop(b, format!("s_lshl_b32 s{COFF}, s{T1}, 6"), &[COFF], &[T1])?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{COFF}, s{T0}"),
                &[COFF],
                &[COFF, T0],
            )?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{COFF}, 8"),
                &[COFF],
                &[COFF],
            )?;
            sop(b, format!("s_lshl_b32 s{SOFF}, s{T1}, 2"), &[SOFF], &[T1])?;
            sop(
                b,
                format!("s_add_i32 s{SOFF}, s{SOFF}, s{T0}"),
                &[SOFF],
                &[SOFF, T0],
            )?;
        }
    }
    sop(
        b,
        format!("s_mul_i32 s{XEO}, s{EIDX}, s{XSTR}"),
        &[XEO],
        &[EIDX, XSTR],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{WE}, s{COFF}, v{WOFF}"),
        &[v(WE)],
        &[s(COFF), v(WOFF)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{HE}, s{SOFF}, v{HOFF}"),
        &[v(HE)],
        &[s(SOFF), v(HOFF)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XE}, s{XEO}, v{XOFF}"),
        &[v(XE)],
        &[s(XEO), v(XOFF)],
    )?;
    // A padding slot stays out of range at every epoch (no wrap of OOB + epoch).
    op(
        b,
        format!(
            "v_cndmask_b32_e64 v{XE}, {}, v{XE}, s{LIVE}",
            lit(GATHER_OOB)
        ),
        &[v(XE)],
        &[v(XE), s(LIVE)],
    )
}

/// The 16-bit scale load at byte `imm` past the epoch's scale offset:
/// `buffer_load_u16` zero-extends into the low half.
pub(super) fn scale_load_at(b: &mut Builder, dst: u8, srd: u8, imm: u32) -> Result<(), String> {
    let off = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    mem(
        b,
        format!(
            "buffer_load_u16 v{dst}, v{HE}, s[{srd}:{}], 0 offen{off}",
            srd + 3
        ),
        &[v(dst)],
        &[v(HE), sr(srd, 4)],
        MemoryClass::VmemLoad,
    )
}

pub(super) fn scale_load(b: &mut Builder, dst: u8, srd: u8) -> Result<(), String> {
    scale_load_at(b, dst, srd, 0)
}

pub(super) fn weight_loads(b: &mut Builder, g: &Gen, t: u8) -> Result<(), String> {
    let mut dst = g.a(t);
    for (off, width) in code_loads(g.spec.codec) {
        bload(b, width, dst, WE, g.code_srd(t), off)?;
        dst += width;
    }
    Ok(())
}

/// Scale dword of the trip's second epoch, tensor `t`: scratch pair 0 (MQ4 never unpacks
/// through scratch).
fn hs2(g: &Gen, t: u8) -> u8 {
    g.scratch(0).0 + t
}

/// Position of an epoch inside a prefetching two-epoch trip.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    /// One epoch loads and broadcasts its own scale.
    Plain,
    /// First epoch of the pair: loads both epochs' scales (the odd epoch's is the next
    /// 16-bit field of the same QT44 group header, `+4`) and broadcasts the second one
    /// under its own fold.
    First,
    /// Second epoch: scale already broadcast.
    Second,
}

/// Issue exactly epoch `EIDX`'s loads: scales and the first tensor's codes,
/// the second tensor's codes (GateUp), then the activation block.
fn load_epoch(b: &mut Builder, g: &Gen, phase: Phase) -> Result<(), String> {
    match phase {
        Phase::Plain | Phase::First => b.clause(|b| {
            for t in 0..g.tn() {
                scale_load(b, g.hs(t), g.scale_srd(t))?;
            }
            if phase == Phase::First {
                for t in 0..g.tn() {
                    scale_load_at(b, hs2(g, t), g.scale_srd(t), 4)?;
                }
            }
            weight_loads(b, g, 0)
        })?,
        Phase::Second => b.clause(|b| weight_loads(b, g, 0))?,
    }
    if g.tn() == 2 {
        b.clause(|b| weight_loads(b, g, 1))?;
    }
    b.clause(|b| {
        for q in 0..4u8 {
            bload(b, 4, g.x() + 4 * q, XE, SRD_X, 8 + 16 * u32::from(q))?;
        }
        bload(b, 1, g.d(), XE, SRD_X, 0)
    })
}

/// The exact-gfx1100 BF16 MQ4 GateUp entry carries the lockstep barrier and the prefetched
/// second-epoch scale; the `_acc` entries and gfx1151 keep their previous code.
fn lockstep_entry(g: &Gen) -> bool {
    g.spec.arch == Arch::Gfx1100
        && g.spec.codec == Codec::Mq4
        && g.spec.kind == Kind::GateUp
        && !g.spec.acc
}

/// Where the per-epoch lockstep barrier sits in the epoch.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum Place {
    /// After the fold and the drain: the rendezvous that precedes the next
    /// epoch's load issue.
    End,
    /// Between the last WMMA of the chain and the F32 fold.
    PostWmma,
}

/// No lockstep barrier in this candidate.
pub(super) const LOCK: Option<Place> = None;

/// Epochs per loop trip of the lockstep GateUp entry (its epoch count is even: `K % 256 == 0`).
const TRIP: u8 = 2;

/// Marker of the two 4-byte LDS regions whose ring rotation carries the
/// lockstep barrier (gfx11 has only the full `s_barrier`, which the typed
/// core lowers only with an LDS transition).
pub(super) struct Lock;
pub(super) type LockRing = Ring<Lock, Published, Free, 2>;
pub(super) type LockFree = Ring<Lock, Free, Free, 2>;
type LockStored = (
    Ring<Lock, Published, peacemaker_author::Writing, 2>,
    peacemaker_author::Pending<Gfx11Waits, peacemaker_author::LdsWrite<Lock>>,
);

/// A one-dword LDS store into the ring's next buffer: the address and data
/// are the zeroed scratch register (MQ4 never uses scratch pair 0).
fn lock_store<T: Gfx11, C: peacemaker_author::State>(
    wg: &mut Wg<'_, T>,
    g: &Gen,
    ring: Ring<Lock, C, Free, 2>,
) -> Result<
    (
        Ring<Lock, C, peacemaker_author::Writing, 2>,
        peacemaker_author::Pending<Gfx11Waits, peacemaker_author::LdsWrite<Lock>>,
    ),
    String,
> {
    let r = g.scratch(0).0;
    let store = Instruction::new(format!("ds_store_b32 v{r}, v{r}"), vec![], vec![v(r)])
        .memory(MemoryClass::DsStore);
    wg.ds_store(ring, store)
}

/// A one-dword read of the ring's current buffer (the barrier that retires
/// the buffer needs a read of it); the data lands in scratch `y`, unused.
fn lock_load<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen, ring: &LockRing) -> Result<(), String> {
    let (r, y) = g.scratch(0);
    let load = Instruction::new(format!("ds_load_b32 v{y}, v{r}"), vec![v(y)], vec![v(r)])
        .memory(MemoryClass::DsLoad);
    wg.ds_load_cur(ring, load)
}

/// The pre-loop rendezvous: publish the first buffer.
fn lock_prime<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen, ring: LockFree) -> Result<LockRing, String> {
    let r = g.scratch(0).0;
    op(wg.isa(), format!("v_mov_b32_e32 v{r}, 0"), &[v(r)], &[])?;
    let (ring, pending) = lock_store(wg, g, ring)?;
    let drained = wg.wait(pending)?;
    let (ring,) = wg.barrier((prime(ring, drained),))?;
    Ok(ring)
}

/// Drain the epoch's store and rotate the ring at one full barrier.
fn lock_barrier<T: Gfx11>(wg: &mut Wg<'_, T>, stored: LockStored) -> Result<LockRing, String> {
    let (ring, pending) = stored;
    let drained = wg.wait(pending)?;
    let (ring,) = wg.barrier((rotate(ring, drained),))?;
    Ok(ring)
}

/// One K128 epoch: offsets, loads, chain, fold, drain, epoch counter step,
/// with the lockstep rendezvous at `LOCK` when `ring` is present. The ring's
/// current buffer is read at the top of the epoch (retire needs a read) and
/// the next buffer's store is issued after the chain, so both LDS ops
/// complete under the fold's own `lgkmcnt` wait.
fn epoch<T: Gfx11>(
    wg: &mut Wg<'_, T>,
    g: &Gen,
    ring: Option<LockRing>,
    phase: Phase,
) -> Result<Option<LockRing>, String> {
    if let Some(ring) = &ring {
        lock_load(wg, g, ring)?;
    }
    if lockstep_entry(g) {
        // Lockstep: both waves issue this epoch's weight and activation gathers together.
        // M7 refuses a barrier with DS ops pending: the second epoch's prefetched scale
        // broadcast (16 `ds_swizzle`) is drained first.
        if phase == Phase::Second {
            wg.isa().wait_all()?;
        }
        wg.barrier(Lockstep)?;
    }
    let b = wg.isa();
    epoch_offsets(b, g)?;
    load_epoch(b, g, phase)?;
    fold::chain(b, g, phase != Phase::Second)?;
    let mut stored = match ring {
        Some(ring) => Some(lock_store(wg, g, ring)?),
        None => None,
    };
    let mut ring = None;
    if LOCK == Some(Place::PostWmma) {
        if let Some(s) = stored.take() {
            ring = Some(lock_barrier(wg, s)?);
        }
    }
    let b = wg.isa();
    fold::fold(b, g)?;
    if phase == Phase::First {
        // The second epoch's scales were loaded with this epoch's: broadcast them now, off
        // the next epoch's pre-WMMA path. The 16 `ds_swizzle` stay in flight across the
        // epoch step (nothing else is pending: every load fed this epoch's fold).
        fold::broadcast(b, g, |t| hs2(g, t))?;
    } else {
        // Drain before the next epoch (and the epilogue aliases): an empty ledger.
        b.wait_all()?;
    }
    let step = if g.spec.negative == Negative::ReversedEpoch {
        "-1"
    } else {
        "1"
    };
    sop(
        b,
        format!("s_add_i32 s{EIDX}, s{EIDX}, {step}"),
        &[EIDX],
        &[EIDX],
    )?;
    if LOCK == Some(Place::End) {
        if let Some(s) = stored.take() {
            ring = Some(lock_barrier(wg, s)?);
        }
    }
    Ok(ring)
}
/// The loop compare after the trip's last epoch step.
fn back_edge<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen) -> Result<WgUniform<Scc>, String> {
    if g.spec.negative == Negative::ReversedEpoch {
        cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, -1"), &[EIDX])
    } else {
        cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, s{ECNT}"), &[EIDX, ECNT])
    }
}

/// Phase of epoch `k` of a trip: the MQ4 GateUp pair prefetches its second scale.
fn phase_of(g: &Gen, k: u8) -> Phase {
    if lockstep_entry(g) {
        if k == 0 {
            Phase::First
        } else {
            Phase::Second
        }
    } else {
        Phase::Plain
    }
}

/// The epoch loop. Places `k_begin`; leaves the builder at the fall-through
/// into the caller's `epilogue` label with an empty wait ledger.
///
/// The BF16 MQ4 Down entry takes the unrolled software-pipelined path when `K = 640`
/// (`ECNT == 5`); every other launch runs the incumbent loop ([`general`]).
pub(super) fn emit<T: Gfx11>(
    wg: &mut Wg<'_, T>,
    g: &Gen,
    lock: Option<LockFree>,
) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    wg.isa().wait_all()?;
    if pipe::applies(g) {
        let not_k640 = cmp(
            wg,
            format!("s_cmp_lg_u32 s{ECNT}, {}", pipe::EPOCHS),
            &[ECNT],
        )?;
        wg.wg_if_else(
            not_k640,
            &g.label("k_general"),
            &g.label("k_join"),
            (),
            |wg, ()| pipe::emit(wg, g),
            |wg, ()| general(wg, g, lock),
        )?;
    } else {
        general(wg, g, lock)?;
    }
    wg.isa().wait_all()
}

/// The incumbent one-epoch loop (runtime epoch count), entered with an empty ledger.
fn general<T: Gfx11>(wg: &mut Wg<'_, T>, g: &Gen, lock: Option<LockFree>) -> Result<(), String> {
    let trip = if lockstep_entry(g) { TRIP } else { 1 };
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let ring = match lock {
        Some(free) => Some(lock_prime(wg, g, free)?),
        None => None,
    };
    let entry = wg.isa().ledger().shape();
    let none = cmp(wg, format!("s_cmp_eq_u32 s{ECNT}, 0"), &[ECNT])?;
    let ring = wg.wg_skip_if(none, &done, ring, |wg, ring| {
        let ring = match ring {
            None => {
                wg.loop_carried(&head, (), |wg, ()| {
                    for k in 0..trip {
                        epoch(wg, g, None, phase_of(g, k))?;
                    }
                    Ok(((), back_edge(wg, g)?))
                })?;
                None
            }
            Some(ring) => {
                let ring = wg.loop_carried(&head, ring, |wg, ring| {
                    let mut ring = Some(ring);
                    for k in 0..trip {
                        ring = epoch(wg, g, ring, phase_of(g, k))?;
                    }
                    let ring = ring.ok_or("lock ring lost")?;
                    Ok((ring, back_edge(wg, g)?))
                })?;
                Some(ring)
            }
        };
        if wg.isa().ledger().shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(ring)
    })?;
    if let Some(ring) = ring {
        // Every wave is past the last epoch: read, then retire the published buffer.
        lock_load(wg, g, &ring)?;
        let (_free,) = wg.barrier((retire_cur(ring),))?;
    }
    wg.isa().wait_all()
}
