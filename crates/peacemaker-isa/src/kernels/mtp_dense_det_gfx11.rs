// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 MTP-append DET dense GEMM (the `fc_hidden` projection), emitted with
//! the checked ISA builder: `Y[N, M] = X16[N, K] . W16[M, K]^T`, F16 operands,
//! F32 accumulation and F32 output. Entry `mtp_dense_det_pm_gfx1151_w8`, module
//! `mtp_dense_det_pm_gfx1151`.
//!
//! Kernargs (36 bytes): `A16` (F16 `[M, K]`, offset 0), `X16` (F16 `[N, K]`, 8),
//! `Y32` (F32 `[N, M]`, 16), `M` (i32, 24), `K` (i32, 28), `N` (i32, 32). The
//! host (the runtime route, not the kernel) guarantees `M % 16 == 0`, `K % 64 == 0`,
//! `K >= 64`, `N >= 1`, 16-byte aligned non-aliasing matrices and byte extents
//! (`M * K * 2`, `N * K * 2`, `N * M * 4`) below `2^31`. `M`, `K` or `N <= 0` and
//! `K < 64` return without writing.
//!
//! Deterministic by construction (DET, NOT byte-identical to the incumbent
//! BF16/F32 multirow GEMM): operands are the host's conversions (weights: BF16 ->
//! F32 exact widening -> F16 RNE, cached immutable shadow; X: F32 -> F16 RNE, per
//! call), whose narrower exponent range can under/overflow. Every output element
//! is the chain of `v_wmma_f32_16x16x16_f16` over ascending 16-K steps with C
//! chained from +0.0 (exactly `K / 16` WMMAs, no split-K, no atomics), whatever
//! `M`, `K` and `N` are: row `n` of `Y` depends only on row `n` of `X16`, never on
//! the batch, its tiling or its padding. The epilogue is the incumbent GEMM's
//! `0.0f + acc`, then an F32 store.
//!
//! Tiling and scheduling (all of it free of numerical effect):
//!  * a workgroup is 8 waves of 16 channels = 128 channels x 256 projection rows
//!    (`N` counts rows, including any flattened branches), grid
//!    `[ceil(M / 128), ceil(N / 256), 1]`, block 256, static LDS 40960 bytes;
//!    each wave owns 16 channels x 256 rows = 16 accumulator tiles (128 VGPRs);
//!  * X is staged in 32-K chunks, double-buffered in LDS (two 20480-byte slots,
//!    80-byte row pitch), one barrier per chunk; the weights are read straight
//!    from global memory (a wave's rows are its own), one chunk (two K-steps = 32
//!    VGPRs) ahead in two alternating register sets; the B fragments (two
//!    `ds_load_b128`) rotate through a ring of four; the 64-K loop body holds
//!    both chunks (64 WMMAs) so slot and set parity return to the loop entry;
//!  * rows past `N` read zeros (X descriptor bounded at `N * K * 2`) and are never
//!    stored; channel rows past `M` (idle waves of the last workgroup column) are
//!    clamped to row `M - 1` and never stored. Every lane stays active through
//!    the WMMAs and barriers; only the stores are masked;
//!  * the speculative loads past the last chunk re-read the last chunk (offsets
//!    clamped), never wrapping into another row; they never enter a WMMA.
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, smem, sop, sr, v, vr};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The one entry (exact gfx1151).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
}
impl Spec {
    pub fn symbol(&self) -> String {
        format!("mtp_dense_det_pm_{}_w8", self.arch.name())
    }
    pub fn module(arch: Arch) -> String {
        format!("mtp_dense_det_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("mtp_dense_det_gfx11 builder is exact-gfx1151 only".into());
        }
        Ok(())
    }
}

/// Channels per workgroup (8 waves x 16) and projection rows per workgroup (16 tiles x 16).
pub const CHANNELS: u32 = 128;
pub const TOK: u32 = 256;
pub const THREADS: u16 = 256;
pub const KERNARG_BYTES: u32 = 36;
const PITCH: u32 = 80; // 32 K f16 + 8 pad per row
pub const SLOT_BYTES: u32 = TOK * PITCH; // 20480
pub const GROUP_BYTES: u32 = 2 * SLOT_BYTES; // 40960
/// Accumulator tiles per wave; WMMAs per 32-K chunk (two K-steps x 16 tiles); B ring depth.
const BT: u8 = 16;
const WPC: u8 = 2 * BT;
const RING: u8 = 4;
/// WMMA after which the chunk barrier sits: the last ring load of this chunk's
/// slot (fragment `i + RING` after WMMA `i`, `i = WPC - RING - 1`) has issued.
const BAR_AT: u8 = WPC - RING - 1;
/// X uint4 per thread per chunk (256 rows x 4 uint4 over 256 threads) and the row step between them.
const XPT: u8 = 4;
const XROW_STEP: u32 = 64;

fn lbl(name: &str) -> String {
    format!(".Lmtpdet_{name}")
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2, s3 workgroup ids (after the two user SGPRs).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
const SRD_A: u8 = 4;
const SRD_X: u8 = 8;
const SRD_Y: u8 = 12;
// Raw kernargs: s[16:23] A X Y M K (one s_load_b256), s24 N.
const PA: u8 = 16;
const PX: u8 = 18;
const PY: u8 = 20;
const SM: u8 = 22;
const SK: u8 = 23;
const SN: u8 = 24;
const ITERS: u8 = 25;
const ITERSM1: u8 = 26;
const IT: u8 = 27;
const SA: u8 = 28;
const SANX: u8 = 29;
const C2: u8 = 30;
const SXOFF: u8 = 31;
const NCHM1: u8 = 32;
const ST0: u8 = 33;
const ST1: u8 = 34;
const SK2: u8 = 35;
const SZERO: u8 = 36;
const SNB: u8 = 37;
const YSTEP: u8 = 38;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
const MASK0: u8 = 40;
const MASK1: u8 = 42;
/// Lane-half mask (`lh != 0`), live from the prologue to the end.
const LHM: u8 = 44;
const PSEL: u8 = 46;
const SGPR_BUDGET: u16 = 64;

// ----------------------------------------------------------------------- VGPRs
// v0 tid; v1 B fragment LDS read row; v2 X LDS stage address; v3 A row byte
// offset; v4..7 X global byte offsets; then the 16 accumulator tiles, the B
// ring (4 x 8), two sets of A fragments (per set: K-step 8 each) and the X staging.
const TID: u8 = 0;
const LDRD: u8 = 1;
const LDST: u8 = 2;
const VAOFF: u8 = 3;
const XOFF: u8 = 4;
const ACC: u8 = XOFF + XPT; // 8
const RINGB: u8 = ACC + 128; // 136
const XR: u8 = RINGB + 64; // 200
const VGPRS: u16 = XR as u16 + 4 * XPT as u16; // 216: a multiple of the descriptor's granule of 8
fn acc(t: u8) -> u8 {
    ACC + 8 * t
}
fn ring(f: u8) -> u8 {
    RINGB + 8 * (f % RING)
}
fn abuf(set: usize, j: u8) -> u8 {
    RINGB + 32 + 16 * set as u8 + 8 * j
}

/// Epilogue VGPRs (live only from `epi`): they alias the dead main loop's ring.
#[derive(Clone, Copy)]
struct Ep {
    tmp: u8,
    vtok: u8,
    vrow0: u8,
    vyoff: u8,
    offs: u8,
}
const EP: Ep = Ep {
    tmp: RINGB,
    vtok: RINGB + 6,
    vrow0: RINGB + 7,
    vyoff: RINGB + 8,
    offs: RINGB + 16,
};

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPRS, SGPR_BUDGET)?;
    let w = || Live::Whole;
    let dead = || Live::Between("entry".into(), lbl("epi"));
    let epi = || Live::Between(lbl("epi"), lbl("end"));
    p.v::<1>("tid", TID, w())?;
    for n in [LDRD, LDST, VAOFF] {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in 0..XPT {
        p.v::<1>("x_offset", XOFF + n, w())?;
    }
    for t in 0..BT {
        p.v::<8>("acc", acc(t), w())?;
    }
    for r in 0..RING {
        p.v::<8>("b_ring", RINGB + 8 * r, dead())?;
    }
    for set in 0..2 {
        for j in 0..2 {
            p.v::<8>("a_frag", abuf(set, j), dead())?;
        }
    }
    for i in 0..XPT {
        p.v::<4>("x_stage", XR + 4 * i, dead())?;
    }
    for n in 0..6 {
        p.v::<1>("epi_temp", EP.tmp + n, epi())?;
    }
    for n in [EP.vtok, EP.vrow0, EP.vyoff] {
        p.v::<1>("epi_lane", n, epi())?;
    }
    for n in 0..BT {
        p.v::<1>("epi_offset", EP.offs + n, epi())?;
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<1>("wg_x", WGX, w())?;
    p.s::<1>("wg_y", WGY, w())?;
    for srd in [SRD_A, SRD_X, SRD_Y] {
        p.s::<4>("srd", srd, w())?;
    }
    p.s::<8>("kernargs", PA, w())?;
    for n in SN..=YSTEP {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1, LHM] {
        p.s::<2>("lane_mask", m, w())?;
    }
    p.s::<1>("permlane_sel", PSEL, w())?;
    Ok(p)
}

fn rs(base: u8, len: u8) -> String {
    if len == 1 {
        format!("v{base}")
    } else {
        format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1)
    }
}
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}

/// buffer_load_b{32,64,128} dst, v{voff}, s[srd], s{soff} offen offset:imm
fn bload(b: &mut Builder, dwords: u8, dst: u8, voff: u8, srd: u8, soff: u8, imm: u32) -> R {
    let w = match dwords {
        1 => 32,
        2 => 64,
        4 => 128,
        _ => return Err("bload width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "buffer_load_b{w} {}, v{voff}, s[{srd}:{}], s{soff} offen{}",
                rs(dst, dwords),
                srd + 3,
                off(imm)
            ),
            vec![vr(dst, dwords)],
            vec![v(voff), sr(srd, 4), s(soff)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
/// buffer_store_b128.
fn bstore(b: &mut Builder, data: u8, voff: u8, srd: u8, soff: u8, imm: u32) -> R {
    b.push(
        Instruction::new(
            format!(
                "buffer_store_b128 {}, v{voff}, s[{srd}:{}], s{soff} offen{}",
                rs(data, 4),
                srd + 3,
                off(imm)
            ),
            vec![],
            vec![vr(data, 4), v(voff), sr(srd, 4), s(soff)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
/// Accumulator tile `t` += A * B.
fn wmma(b: &mut Builder, t: u8, a: u8, bb: u8) -> R {
    let c = acc(t);
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_f16 {}, {}, {}, {}",
            rs(c, 8),
            rs(a, 8),
            rs(bb, 8),
            rs(c, 8)
        ),
        &[vr(c, 8)],
        &[vr(a, 8), vr(bb, 8), vr(c, 8)],
    )
}
fn slot_base(slot: usize) -> u32 {
    slot as u32 * SLOT_BYTES
}

/// One scheduled operation interleaved between WMMAs.
#[derive(Clone)]
enum Fill {
    /// B fragment `f` of a chunk (K-step `f / 16`, row tile `f % 16`) from `slot` into its ring register.
    BLoad { slot: usize, f: u8 },
    /// Store X staging uint4 `i` into `slot`.
    XStore { slot: usize, i: u8 },
    /// 16 B of an A fragment half: `dst` <- A rows at `VAOFF + soff + imm`.
    ALoad { dst: u8, soff: u8, imm: u32 },
}
fn emit_fill(b: &mut Builder, f: &Fill) -> R {
    match f {
        Fill::BLoad { slot, f } => {
            let (j, t) = (u32::from(f / BT), u32::from(f % BT));
            let dst = ring(*f);
            let imm = slot_base(*slot) + t * 16 * PITCH + j * 32;
            for half in 0..2u8 {
                let d = dst + 4 * half;
                b.ds_load(
                    *slot,
                    Instruction::new(
                        format!(
                            "ds_load_b128 {}, v{LDRD}{}",
                            rs(d, 4),
                            off(imm + 16 * u32::from(half))
                        ),
                        vec![vr(d, 4)],
                        vec![v(LDRD)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            Ok(())
        }
        Fill::XStore { slot, i } => {
            let data = XR + 4 * i;
            let imm = u32::from(*i) * XROW_STEP * PITCH;
            b.ds_store(
                *slot,
                Instruction::new(
                    format!(
                        "ds_store_b128 v{LDST}, {}{}",
                        rs(data, 4),
                        off(slot_base(*slot) + imm)
                    ),
                    vec![],
                    vec![v(LDST), vr(data, 4)],
                )
                .memory(MemoryClass::DsStore),
            )
        }
        Fill::ALoad { dst, soff, imm } => bload(b, 4, *dst, VAOFF, SRD_A, *soff, *imm),
    }
}

/// The four 16-byte loads of the A set `set` (K-steps 0 and 1 of one 32-K chunk,
/// two halves each) at `soff + imm`, imm = `base + 32 j + 16 h`.
fn a_loads(set: usize, soff: u8, base: u32) -> Vec<Fill> {
    let mut out = Vec::new();
    for j in 0..2u8 {
        for h in 0..2u8 {
            out.push(Fill::ALoad {
                dst: abuf(set, j) + 4 * h,
                soff,
                imm: base + 32 * u32::from(j) + 16 * u32::from(h),
            });
        }
    }
    out
}

/// Spread `fills` evenly over the WMMA slots `from..to`.
fn spread(slots: &mut [Vec<Fill>], fills: Vec<Fill>, from: usize, to: usize) {
    let n = fills.len().max(1);
    for (k, f) in fills.into_iter().enumerate() {
        slots[from + k * (to - from) / n].push(f);
    }
}

/// One 32-K chunk `cc` (0 or 1) of the 64-K loop body: 32 WMMAs. WMMA `w` is
/// K-step `w / 16`, row tile `w % 16` of the chunk; B fragment `f = w` feeds it
/// (K offset `32 cc + 16 (w / 16)` within the iteration, ascending).
fn chunk(b: &mut Builder, cc: u8) -> R {
    let slot = usize::from(cc % 2);
    let other = 1 - slot;
    // X chunk (2 * iteration + cc + 2), clamped: its loads follow this chunk's barrier.
    sop(
        b,
        format!("s_add_i32 s{SXOFF}, s{C2}, {}", cc + 2),
        &[SXOFF],
        &[C2],
    )?;
    sop(
        b,
        format!("s_min_i32 s{SXOFF}, s{SXOFF}, s{NCHM1}"),
        &[SXOFF],
        &[SXOFF, NCHM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXOFF}, s{SXOFF}, 6"),
        &[SXOFF],
        &[SXOFF],
    )?;
    let mut slots: Vec<Vec<Fill>> = vec![Vec::new(); usize::from(WPC)];
    // The next chunk's A set (chunk 1 of this iteration, or chunk 0 of the
    // next with its offset clamped to the last iteration), issued early.
    let (soff, base) = if cc == 0 { (SA, 64) } else { (SANX, 0) };
    spread(&mut slots, a_loads(1 - slot, soff, base), 0, 16);
    // B ring: WMMA i loads WMMA i + RING's fragment into the register it just read.
    for i in 0..WPC {
        if i + RING < WPC {
            slots[usize::from(i)].push(Fill::BLoad { slot, f: i + RING });
        }
    }
    // Stage chunk c+1's X into the other slot, well before the barrier.
    let stores: Vec<Fill> = (0..XPT).map(|i| Fill::XStore { slot: other, i }).collect();
    spread(&mut slots, stores, 18, 26);
    for i in 0..WPC {
        let (j, t) = (i / BT, i % BT);
        wmma(b, t, abuf(slot, j), ring(i))?;
        for fill in &slots[usize::from(i)] {
            emit_fill(b, fill)?;
        }
        if i == BAR_AT {
            // This wave's reads of `slot` have landed (the last ring loads of
            // this chunk); publish the other slot, retire this one.
            b.wait(Counter::Lgkm, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in 0..XPT {
                bload(b, 4, XR + 4 * x, XOFF + x, SRD_X, SXOFF, 0)?;
            }
        } else if i > BAR_AT {
            // The next chunk's first fragments, from the published slot.
            emit_fill(
                b,
                &Fill::BLoad {
                    slot: other,
                    f: i + RING - WPC,
                },
            )?;
        }
    }
    Ok(())
}

fn prologue(b: &mut Builder) -> R {
    let t = |n: u8| RINGB + n; // prologue temporaries alias the (not yet loaded) ring
    let (t0, t1, t2, t3, t4) = (t(0), t(1), t(2), t(3), t(4));
    smem(b, PA, 8, KARG, 0)?;
    smem(b, SN, 1, KARG, 0x20)?;
    // Lane constants that need no kernargs (the entry block: the LDS bound
    // check derives every DS address from these).
    op(
        b,
        format!("v_and_b32_e32 v{t0}, 15, v{TID}"),
        &[v(t0)],
        &[v(TID)],
    )?; // ml
    op(
        b,
        format!("v_bfe_u32 v{t1}, v{TID}, 4, 1"),
        &[v(t1)],
        &[v(TID)],
    )?; // lh
    op(
        b,
        format!("v_mul_u32_u24_e32 v{LDRD}, {}, v{t0}", lit(PITCH)),
        &[v(LDRD)],
        &[v(t0)],
    )?; // B rows: no lane-half term
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t3}, 2, v{TID}"),
        &[v(t3)],
        &[v(TID)],
    )?; // tid/4
    op(
        b,
        format!("v_and_b32_e32 v{t4}, 3, v{TID}"),
        &[v(t4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{t4}, 4, v{t4}"),
        &[v(t4)],
        &[v(t4)],
    )?; // (tid%4)*16
    op(
        b,
        format!("v_mad_u32_u24 v{LDST}, v{t3}, {}, v{t4}", lit(PITCH)),
        &[v(LDST)],
        &[v(t3), v(t4)],
    )?;
    op(
        b,
        format!("v_cmp_ne_u32_e64 s{LHM}, 0, v{t1}"),
        &[s(LHM)],
        &[v(t1)],
    )?;
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    for n in 0..128u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", ACC + n),
            &[v(ACC + n)],
            &[],
        )?;
    }
    b.wait(Counter::Lgkm, 0)?;
    // M, K, N > 0 else exit; K < 64: no chunks, nothing stored.
    sop(
        b,
        format!("s_min_i32 s{ST0}, s{SM}, s{SK}"),
        &[ST0],
        &[SM, SK],
    )?;
    sop(
        b,
        format!("s_min_i32 s{ST0}, s{ST0}, s{SN}"),
        &[ST0],
        &[ST0, SN],
    )?;
    sop(b, format!("s_cmp_lt_i32 s{ST0}, 1"), &[], &[ST0])?;
    op(b, format!("s_cbranch_scc1 {}", lbl("end")), &[], &[])?;
    sop(b, format!("s_cmp_lt_i32 s{SK}, 64"), &[], &[SK])?;
    op(b, format!("s_cbranch_scc1 {}", lbl("end")), &[], &[])?;
    sop(b, format!("s_lshr_b32 s{ITERS}, s{SK}, 6"), &[ITERS], &[SK])?;
    sop(
        b,
        format!("s_add_i32 s{ITERSM1}, s{ITERS}, -1"),
        &[ITERSM1],
        &[ITERS],
    )?;
    sop(b, format!("s_lshr_b32 s{NCHM1}, s{SK}, 5"), &[NCHM1], &[SK])?;
    sop(
        b,
        format!("s_add_i32 s{NCHM1}, s{NCHM1}, -1"),
        &[NCHM1],
        &[NCHM1],
    )?;
    sop(b, format!("s_lshl_b32 s{SK2}, s{SK}, 1"), &[SK2], &[SK])?;
    // Buffer descriptors, each bounded at its byte extent: A at M*K*2 (its
    // rows are clamped, never past), X at N*K*2 (rows past N read zeros), Y at N*M*4.
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{SM}, s{SK2}"),
        &[ST0],
        &[SM, SK2],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{SN}, s{SK2}"),
        &[ST1],
        &[SN, SK2],
    )?;
    sop(b, format!("s_lshl_b32 s{YSTEP}, s{SM}, 2"), &[YSTEP], &[SM])?;
    sop(
        b,
        format!("s_mul_i32 s{YSTEP}, s{YSTEP}, s{SN}"),
        &[YSTEP],
        &[YSTEP, SN],
    )?;
    for (srd, p, records) in [(SRD_A, PA, ST0), (SRD_X, PX, ST1), (SRD_Y, PY, YSTEP)] {
        sop(b, format!("s_mov_b32 s{srd}, s{p}"), &[srd], &[p])?;
        sop(
            b,
            format!("s_and_b32 s{}, s{}, 0xffff", srd + 1, p + 1),
            &[srd + 1],
            &[p + 1],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{}, s{records}", srd + 2),
            &[srd + 2],
            &[records],
        )?;
        sop(
            b,
            format!(
                "s_mov_b32 s{}, {}",
                srd + 3,
                lit(crate::kernels::common::SRD_WORD3)
            ),
            &[srd + 3],
            &[],
        )?;
    }
    // Weight row byte offset of lane (workgroup channel start + 16 wave + ml), clamped to the last row.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t3}, 1, v{TID}"),
        &[v(t3)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t3}, {}, v{t3}", lit(CHANNELS - 16)),
        &[v(t3)],
        &[v(t3)],
    )?; // 16 * wave
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(CHANNELS)),
        &[ST0],
        &[WGX],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{t3}, s{ST0}, v{t3}"),
        &[v(t3)],
        &[s(ST0), v(t3)],
    )?; // channel start
    op(
        b,
        format!("v_add_nc_u32_e32 v{t2}, v{t3}, v{t0}"),
        &[v(t2)],
        &[v(t3), v(t0)],
    )?; // row
        // Clamp to the last row with the shared_gu compare/select pair (the emulator has no `v_min_i32_e32` semantics).
    let t5 = t(5);
    sop(b, format!("s_add_i32 s{ST1}, s{SM}, -1"), &[ST1], &[SM])?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{t2}"),
        &[s(MASK1)],
        &[s(SM), v(t2)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{t5}, s{ST1}"),
        &[v(t5)],
        &[s(ST1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t2}, v{t5}, v{t2}, s{MASK1}"),
        &[v(t2)],
        &[v(t5), v(t2), s(MASK1)],
    )?; // clamped row
    op(
        b,
        format!("v_mul_lo_u32 v{VAOFF}, v{t2}, s{SK2}"),
        &[v(VAOFF)],
        &[v(t2), s(SK2)],
    )?;
    // X rows: row wgy*256 + i*64 + tid/4, byte (tid%4)*16 of a 32-K chunk.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 8"), &[ST0], &[WGY])?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t2}, 2, v{TID}"),
        &[v(t2)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{t2}, s{ST0}, v{t2}"),
        &[v(t2)],
        &[s(ST0), v(t2)],
    )?;
    for i in 0..XPT {
        let x = XOFF + i;
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{x}, {}, v{t2}",
                lit(u32::from(i) * XROW_STEP)
            ),
            &[v(x)],
            &[v(t2)],
        )?;
        op(
            b,
            format!("v_mul_lo_u32 v{x}, v{x}, s{SK2}"),
            &[v(x)],
            &[v(x), s(SK2)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{x}, v{x}, v{t4}"),
            &[v(x)],
            &[v(x), v(t4)],
        )?;
    }
    // Chunk 0: X to slot 0, its A set loads in flight (issued right behind
    // the X loads: the stores wait for X only), then the loop's entry state:
    // chunk 1's X loads and chunk 0's first four B fragments in flight.
    sop(b, format!("s_mov_b32 s{IT}, 0"), &[IT], &[])?;
    sop(b, format!("s_mov_b32 s{SA}, 0"), &[SA], &[])?;
    sop(b, format!("s_mov_b32 s{C2}, 0"), &[C2], &[])?;
    sop(b, format!("s_mov_b32 s{SXOFF}, 0"), &[SXOFF], &[])?;
    for i in 0..XPT {
        bload(b, 4, XR + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in a_loads(0, SA, 0) {
        emit_fill(b, &f)?;
    }
    for i in 0..XPT {
        emit_fill(b, &Fill::XStore { slot: 0, i })?;
    }
    b.barrier(&[Transition::Ready(0)])?;
    // K >= 64: chunk 1 exists (min(1, K/32 - 1) = 1).
    sop(b, format!("s_mov_b32 s{SXOFF}, 64"), &[SXOFF], &[])?;
    for i in 0..XPT {
        bload(b, 4, XR + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in 0..RING {
        emit_fill(b, &Fill::BLoad { slot: 0, f })?;
    }
    Ok(())
}

fn body(b: &mut Builder) -> R {
    // sanx = min(it + 1, iters - 1) * 128 (the last iteration re-reads itself).
    sop(b, format!("s_add_i32 s{SANX}, s{IT}, 1"), &[SANX], &[IT])?;
    sop(
        b,
        format!("s_min_i32 s{SANX}, s{SANX}, s{ITERSM1}"),
        &[SANX],
        &[SANX, ITERSM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SANX}, s{SANX}, 7"),
        &[SANX],
        &[SANX],
    )?;
    for cc in 0..2 {
        chunk(b, cc)?;
    }
    sop(b, format!("s_add_i32 s{IT}, s{IT}, 1"), &[IT], &[IT])?;
    sop(
        b,
        format!("s_add_i32 s{SA}, s{SA}, {}", lit(128)),
        &[SA],
        &[SA],
    )?;
    sop(b, format!("s_add_i32 s{C2}, s{C2}, 2"), &[C2], &[C2])?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{IT}, s{ITERS}"),
        &[],
        &[IT, ITERS],
    )?;
    op(b, format!("s_cbranch_scc1 {}", lbl("loop")), &[], &[])
}

// ------------------------------------------------------------------- epilogue

/// `v_add_f32 0, acc` over every accumulator tile (the incumbent GEMM's `0.0f + acc`).
fn add_zero(b: &mut Builder) -> R {
    for n in 0..8 * BT {
        let a = ACC + n;
        op(b, format!("v_add_f32_e32 v{a}, 0, v{a}"), &[v(a)], &[v(a)])?;
    }
    Ok(())
}

/// The cross-half swap in place over accumulator tile `t`: before it,
/// `acc[e]` is channel `2e + lh`, row `ml` of the tile; afterwards `acc[0..7]`
/// are the eight consecutive channels `8 lh ..` of row `ml`.
/// `give_i = lh ? acc[i] : acc[4+i]`, `got_i = permlanex16(give_i)`,
/// `r[2i] = lh ? got_i : acc[i]`, `r[2i+1] = lh ? acc[4+i] : got_i`. The
/// outputs permute the inputs in two 3-cycles (positions 1,2,4 and 3,5,6),
/// each broken by one saved input.
fn swap_tile(b: &mut Builder, t: u8) -> R {
    let a = acc(t);
    let g = EP.tmp;
    let (c1, c3) = (EP.tmp + 4, EP.tmp + 5);
    for i in 0..4u8 {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{LHM}",
                g + i,
                a + 4 + i,
                a + i
            ),
            &[v(g + i)],
            &[v(a + 4 + i), v(a + i), s(LHM)],
        )?;
        op(
            b,
            format!("v_permlanex16_b32 v{0}, v{0}, s{PSEL}, 0xfedcba98", g + i),
            &[v(g + i)],
            &[v(g + i), s(PSEL)],
        )?;
    }
    op(
        b,
        format!("v_mov_b32_e32 v{c1}, v{}", a + 1),
        &[v(c1)],
        &[v(a + 1)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{c3}, v{}", a + 3),
        &[v(c3)],
        &[v(a + 3)],
    )?;
    // (dst position, lh-false source, lh-true source)
    for (q, f, tr) in [
        (0, a, g),
        (7, g + 3, a + 7),
        (1, g, a + 4),
        (4, a + 2, g + 2),
        (2, c1, g + 1),
        (3, g + 1, a + 5),
        (5, g + 2, a + 6),
        (6, c3, g + 3),
    ] {
        op(
            b,
            format!("v_cndmask_b32_e64 v{}, v{f}, v{tr}, s{LHM}", a + q),
            &[v(a + q)],
            &[v(f), v(tr), s(LHM)],
        )?;
    }
    Ok(())
}

fn epilogue(b: &mut Builder) -> R {
    b.wait_all()?;
    b.label(&lbl("epi"))?;
    let e = EP;
    let (ml, lh, rstart) = (e.tmp, e.tmp + 1, e.tmp + 2);
    // Lane addresses: channel group start 128 wgx + 16 wave (+ 8 lh), row wgy*256 + ml,
    // Y byte offset (row * M + channel) * 4.
    op(
        b,
        format!("v_and_b32_e32 v{ml}, 15, v{TID}"),
        &[v(ml)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_bfe_u32 v{lh}, v{TID}, 4, 1"),
        &[v(lh)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{rstart}, 1, v{TID}"),
        &[v(rstart)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{rstart}, {}, v{rstart}", lit(CHANNELS - 16)),
        &[v(rstart)],
        &[v(rstart)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(CHANNELS)),
        &[ST0],
        &[WGX],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{rstart}, s{ST0}, v{rstart}"),
        &[v(rstart)],
        &[s(ST0), v(rstart)],
    )?;
    op(
        b,
        format!("v_lshl_or_b32 v{}, v{lh}, 3, v{rstart}", e.vrow0),
        &[v(e.vrow0)],
        &[v(lh), v(rstart)],
    )?;
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 8"), &[ST0], &[WGY])?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{}, s{ST0}, v{ml}", e.vtok),
        &[v(e.vtok)],
        &[s(ST0), v(ml)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{}, v{}, s{SM}", e.vyoff, e.vtok),
        &[v(e.vyoff)],
        &[v(e.vtok), s(SM)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.vyoff, e.vrow0),
        &[v(e.vyoff)],
        &[v(e.vyoff), v(e.vrow0)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", e.vyoff),
        &[v(e.vyoff)],
        &[v(e.vyoff)],
    )?;
    sop(b, format!("s_lshl_b32 s{YSTEP}, s{SM}, 6"), &[YSTEP], &[SM])?;
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    sop(b, format!("s_mov_b32 s{SNB}, s{SN}"), &[SNB], &[SN])?;
    // Tile `t` is 16 rows (16 * M * 4 bytes) after tile `t - 1`.
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", e.offs, e.vyoff),
        &[v(e.offs)],
        &[v(e.vyoff)],
    )?;
    for t in 1..BT {
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{}, s{YSTEP}, v{}",
                e.offs + t,
                e.offs + t - 1
            ),
            &[v(e.offs + t)],
            &[s(YSTEP), v(e.offs + t - 1)],
        )?;
    }
    // Idle waves (channels >= M): M % 16 == 0, so a wave's 16 channels are all in or all out.
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{}", e.vrow0),
        &[s(MASK1)],
        &[s(SM), v(e.vrow0)],
    )?;
    add_zero(b)?;
    for t in 0..BT {
        swap_tile(b, t)?;
        let a = acc(t);
        // Row tile `t`: row wgy*256 + 16 t + ml < N  <=>  vtok < N - 16 t.
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{}", e.vtok),
            &[s(MASK0)],
            &[s(SNB), v(e.vtok)],
        )?;
        op(
            b,
            format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
            &[],
            &[s(MASK0), s(MASK1)],
        )?;
        bstore(b, a, e.offs + t, SRD_Y, SZERO, 0)?;
        bstore(b, a + 4, e.offs + t, SRD_Y, SZERO, 16)?;
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        sop(b, format!("s_add_i32 s{SNB}, s{SNB}, -16"), &[SNB], &[SNB])?;
    }
    b.label(&lbl("end"))?;
    b.control(Sop::End.encode(Arch::Gfx1151)?)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer("A", 0)
        .pointer("X", 8)
        .pointer("Y", 16)
        .hidden("M", 24, 4, "by_value")
        .hidden("K", 28, 4, "by_value")
        .hidden("N", 32, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mtp_dense_det".into(),
        variant: "w8".into(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: THREADS,
        group_segment_fixed_size: GROUP_BYTES,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    for (id, (name, base)) in [("x_slot0", 0u32), ("x_slot1", SLOT_BYTES)]
        .into_iter()
        .enumerate()
    {
        if b.lds_slot(name, base, SLOT_BYTES)? != id {
            return Err("LDS slot order".into());
        }
    }
    prologue(&mut b)?;
    b.loop_(&lbl("loop"), body)?;
    epilogue(&mut b)?;
    b.finish()
}

/// The entry as its code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(Spec { arch })?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}
