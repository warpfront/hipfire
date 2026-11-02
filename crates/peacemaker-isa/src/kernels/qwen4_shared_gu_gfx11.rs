// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 Flash-Next shared-expert gate/up, emitted with the checked ISA
//! builder: one launch replaces the incumbent four-kernel chain
//!
//!  1. `gemm_wmma_lds_128_128_32_64_k64_p` (gate), 2. the same (up),
//!  3. `shared_expert_activation_bf16_f32` (hidden F32 + selector sigmoid),
//!  4. `convert_f32_to_f16` (the shared down GEMM's input conversion),
//!
//! and leaves the same F16 hidden `[N, I]` (row pitch `I`) and the same
//! post-sigmoid selector `[N]` (rewritten in place) in memory.
//!
//! Kernargs (52 bytes): `Wg` (F16 `[I, K]`), `Wu` (F16 `[I, K]`), `X` (F16
//! `[N, K]`), `Y` (F16 hidden `[N, I]`), `Sel` (F32 `[N]`, pre-sigmoid in,
//! post-sigmoid out), `I`, `K`, `N` (i32 by value). Requirements (checked by
//! the runtime route, not by the kernel): `I % 16 == 0`, `K % 64 == 0`,
//! `K >= 64`, `N >= 1`, `N * K * 2 < 2^31`, `I * K * 2 < 2^31`. `K < 64`,
//! `I <= 0` and `N <= 0` return without writing.
//!
//! Exactness (byte-identical to the chain). Every output element is
//! `h = R(R(u) * (v / (1 + expf(-v))))` with `v = R(g)`, `R` hipcc's
//! `silu_mul_bf16_rt` and `(g, u)` the chain's GEMM results:
//!  * `g` / `u` are the same chain of `v_wmma_f32_16x16x16_f16` over
//!    ascending 16-K steps, C chained from +0.0, A = the weight row's 16 F16
//!    (all lanes `l` and `l + 16` hold row `l & 15`), B = the token's 16 F16
//!    (X row `l & 15`) at the same K offsets as the incumbent's
//!    `wlds_wmma_stage`; a 16x16 tile's accumulator element `e` of lane `l`
//!    is channel `16w + 2e + (l >> 4)`, token `l & 15`. No output depends on
//!    the tiling. The epilogue starts with the GEMM epilogue's `0.0f + acc`
//!    (`v_add_f32 0, acc`);
//!  * `R` is the op sequence of the activation kernel's ISA (`v_bfe_u32 16,
//!    1`, `v_cmp_class_f32 0x1f8`, `v_add3_u32 0x7fff`, `v_and_b32
//!    0xffff0000`, `v_cndmask`); `expf` and the IEEE divide are
//!    [`Epilogue::silu_dense_with`]'s expansion, the DAG of that ISA; the
//!    final product has hipcc's operand order `R(u) * q`;
//!  * the selector is `1.0f / (1.0f + expf(-R(sel)))`: the same expansion with
//!    the constant numerator (`v_div_scale_f32 .., 1.0`, `v_div_fixup_f32 ..,
//!    1.0`), applied to token `n` by the workgroup column `blockIdx.x == 0`
//!    of token block `n / 128` (each selector is read and rewritten once);
//!  * the F16 store is `v_cvt_f16_f32` of the finite-rounded `h` (non-finite
//!    values pass `R` unchanged, as in the chain); all later moves are bit
//!    moves (`v_cndmask`, `v_perm_b32`, `v_permlanex16_b32`).
//!
//! Freedom used (tiling, layout and scheduling only):
//!  * a wave owns 16 channels x 128 tokens for BOTH matrices (16
//!    accumulator tiles: gate tiles 0..7, up tiles 0..7, 128 VGPRs), so one B
//!    fragment feeds two WMMAs and the epilogue combines gate and up of the
//!    same channel and token in registers; a workgroup is `wr` waves
//!    (16 * `wr` channels) x 128 tokens, grid `[ceil(I / (16 wr)), ceil(N /
//!    128), 1]`;
//!  * X is staged in 32-K chunks, double-buffered in LDS (two 10240-byte
//!    slots, 80-byte token pitch), one barrier per chunk; the weights are read
//!    straight from global memory (a wave's rows are its own), one chunk
//!    (two K-steps x gate/up = 32 VGPRs) ahead in two alternating register
//!    sets; the B fragments (two `ds_load_b128`) rotate through a ring of four;
//!  * tokens past N read zeros (descriptor bounded at `N*K*2`) and are never
//!    stored; channel rows past `I` are clamped to the last row and never
//!    stored (a workgroup's waves past `I` still run the loop for its
//!    barriers);
//!  * the hidden tile leaves through `v_permlanex16_b32` pair exchanges
//!    (16 B per lane: eight consecutive channels of one token).
//!
//! Launch: block `32 * wr`, grid as above, static LDS 20480 bytes.
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, smem, sop, sr, v, vr};
use crate::kernels::gemm_uk::epilogue::{DivNum, Epilogue, Finish};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::Live;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// `wr` waves (16 channels each) per workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub wr: u8,
}
impl Spec {
    /// The two PM entries.
    pub const ALL: [u8; 2] = [8, 4];
    pub fn symbol(&self) -> String {
        format!("qwen4_shared_gu_pm_{}_w{}", self.arch.name(), self.wr)
    }
    pub fn module(arch: Arch) -> String {
        format!("qwen4_shared_gu_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("qwen4_shared_gu_gfx11 builder is exact-gfx1151 only".into());
        }
        if !Self::ALL.contains(&self.wr) {
            return Err(format!("qwen4_shared_gu_gfx11: no entry w{}", self.wr));
        }
        Ok(())
    }
    /// Channels per workgroup.
    pub fn channels(&self) -> u32 {
        16 * u32::from(self.wr)
    }
    fn threads(&self) -> u16 {
        32 * u16::from(self.wr)
    }
    /// X uint4 per thread per chunk (TOK tokens x 4 uint4).
    fn xpt(&self) -> u8 {
        (TOK * 4 / u32::from(self.threads())) as u8
    }
    /// Token rows between a thread's X uint4s.
    fn xrow_step(&self) -> u32 {
        u32::from(self.threads()) / 4
    }

    // VGPRs. v0 tid; v1..v3 lane constants; v4.. X buffer offsets; then the 16
    // accumulator tiles (gate 0..7, up 8..15), the B ring (4 x 8), two sets of
    // A fragments (per set: K-step x matrix, 8 each) and the X staging.
    fn acc(&self) -> u8 {
        XOFF + self.xpt()
    }
    fn accg(&self, t: u8) -> u8 {
        self.acc() + 8 * t
    }
    fn accu(&self, t: u8) -> u8 {
        self.acc() + 8 * (8 + t)
    }
    fn ringb(&self) -> u8 {
        self.acc() + 128
    }
    fn ring(&self, f: u8) -> u8 {
        self.ringb() + 8 * (f % RING)
    }
    fn abuf(&self, set: usize, j: u8, m: u8) -> u8 {
        self.ringb() + 32 + 32 * set as u8 + 16 * j + 8 * m
    }
    fn xr(&self) -> u8 {
        self.ringb() + 96
    }
    /// Highest VGPR used, plus one.
    fn vgprs_used(&self) -> u16 {
        u16::from(self.xr()) + 4 * u16::from(self.xpt())
    }
    /// The descriptor allocates VGPRs in granules of 8: the plan ends on one.
    fn vgprs(&self) -> u16 {
        self.vgprs_used().next_multiple_of(8)
    }
}

/// Tokens per workgroup (and per wave): 8 tiles of 16.
pub const TOK: u32 = 128;
const PITCH: u32 = 80; // 32 K f16 + 8 pad per token row
pub const SLOT_BYTES: u32 = TOK * PITCH; // 10240
pub const GROUP_BYTES: u32 = 2 * SLOT_BYTES;
/// WMMAs per 32-K chunk (two K-steps x 8 token tiles x gate/up); B ring depth.
const WPC: u8 = 32;
const RING: u8 = 4;
/// WMMA after which the chunk barrier sits: the last ring load of this
/// chunk's slot (fragment `f + RING` after WMMA `2f + 1`, `f = 11`) has issued.
const BAR_AT: u8 = 2 * (16 - RING) - 1;

/// Local labels, qualified by entry so every entry shares one module.
fn lbl(spec: &Spec, name: &str) -> String {
    format!(".Lshgu{}_{name}", spec.wr)
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2, s3 workgroup ids (after the two user SGPRs).
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
const SRD_G: u8 = 4;
const SRD_U: u8 = 8;
const SRD_X: u8 = 12;
const SRD_Y: u8 = 16;
const SRD_S: u8 = 20;
// Raw kernargs: s[24:31] Wg Wu X Y, s[32:33] Sel, s34 I, s35 K, s36 N.
const PG: u8 = 24;
const PU: u8 = 26;
const PX: u8 = 28;
const PY: u8 = 30;
const PS: u8 = 32;
const SI: u8 = 34;
const SK: u8 = 35;
const SN: u8 = 36;
const ITERS: u8 = 37;
const ITERSM1: u8 = 38;
const IT: u8 = 39;
const SA: u8 = 40;
const SANX: u8 = 41;
const C2: u8 = 42;
const SXOFF: u8 = 43;
const NCHM1: u8 = 44;
const ST0: u8 = 45;
const ST1: u8 = 46;
const SK2: u8 = 47;
const SZERO: u8 = 48;
const SNB: u8 = 49;
const SYSTEP: u8 = 50;
const SI2: u8 = 51;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
const MASK0: u8 = 52;
const MASK1: u8 = 54;
const MASKX: u8 = 56;
const LHM: u8 = 58;
const PSEL: u8 = 60;
/// Four mask pairs of the BF16 rounding (`s[64:71]`) and the twelve of the
/// SiLU expansion (`s[72:95]`).
const RMASK: u8 = 64;
const SMASK: u8 = 72;

const TID: u8 = 0;
const LDRD: u8 = 1;
const LDST: u8 = 2;
const VAOFF: u8 = 3;
const XOFF: u8 = 4;

/// Epilogue VGPRs (live only from `epi`): they alias the dead main loop's
/// ring, A sets and X staging.
#[derive(Clone, Copy)]
struct Ep {
    tmp: u8,
    rt: u8,
    sele: u8,
    selo: u8,
    vml: u8,
    vcs: u8,
    offs: u8,
    o: u8,
    sv: u8,
    out: u8,
}
fn ep(spec: &Spec) -> Ep {
    let e = spec.ringb();
    Ep {
        tmp: e,
        rt: e + 24,
        sele: e + 28,
        selo: e + 29,
        vml: e + 30,
        vcs: e + 31,
        offs: e + 32,
        o: e + 40,
        sv: e + 44,
        out: e + 48,
    }
}

fn plan(spec: &Spec) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(spec.vgprs(), 104)?;
    let w = || Live::Whole;
    let dead = || Live::Between("entry".into(), lbl(spec, "epi"));
    let epi = || Live::Between(lbl(spec, "epi"), lbl(spec, "end"));
    p.v::<1>("tid", TID, w())?;
    for n in [LDRD, LDST, VAOFF] {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in 0..spec.xpt() {
        p.v::<1>("x_offset", XOFF + n, w())?;
    }
    for t in 0..16 {
        p.v::<8>("acc", spec.acc() + 8 * t, w())?;
    }
    for r in 0..RING {
        p.v::<8>("b_ring", spec.ringb() + 8 * r, dead())?;
    }
    for set in 0..2 {
        for j in 0..2 {
            for m in 0..2 {
                p.v::<8>("a_frag", spec.abuf(set, j, m), dead())?;
            }
        }
    }
    for i in 0..spec.xpt() {
        p.v::<4>("x_stage", spec.xr() + 4 * i, dead())?;
    }
    if spec.vgprs() > spec.vgprs_used() {
        p.v::<1>("granule_pad", (spec.vgprs() - 1) as u8, w())?;
    }
    let e = ep(spec);
    for n in 0..3 {
        p.v::<8>("epi_silu_tmp", e.tmp + 8 * n, epi())?;
    }
    p.v::<4>("epi_round_tmp", e.rt, epi())?;
    for n in [e.sele, e.selo, e.vml, e.vcs] {
        p.v::<1>("epi_const", n, epi())?;
    }
    p.v::<8>("epi_offset", e.offs, epi())?;
    p.v::<4>("epi_o", e.o, epi())?;
    p.v::<4>("epi_s", e.sv, epi())?;
    for n in 0..4 {
        p.v::<8>("epi_out", e.out + 8 * n, epi())?;
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<1>("wg_x", WGX, w())?;
    p.s::<1>("wg_y", WGY, w())?;
    for srd in [SRD_G, SRD_U, SRD_X, SRD_Y, SRD_S] {
        p.s::<4>("srd", srd, w())?;
    }
    p.s::<8>("kernarg_ptrs", PG, w())?;
    p.s::<2>("kernarg_sel", PS, w())?;
    p.s::<2>("kernarg_ik", SI, w())?;
    for n in SN..=SI2 {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1, MASKX, LHM] {
        p.s::<2>("lane_mask", m, w())?;
    }
    p.s::<1>("permlane_sel", PSEL, w())?;
    for i in 0..2 {
        p.s::<4>("round_mask", RMASK + 4 * i, w())?;
    }
    for i in 0..6 {
        p.s::<4>("silu_mask", SMASK + 4 * i, w())?;
    }
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
/// buffer_store_b{32,128}.
fn bstore(b: &mut Builder, dwords: u8, data: u8, voff: u8, srd: u8, soff: u8, imm: u32) -> R {
    let w = match dwords {
        1 => 32,
        4 => 128,
        _ => return Err("bstore width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "buffer_store_b{w} {}, v{voff}, s[{srd}:{}], s{soff} offen{}",
                rs(data, dwords),
                srd + 3,
                off(imm)
            ),
            vec![],
            vec![vr(data, dwords), v(voff), sr(srd, 4), s(soff)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
/// Accumulator tile `t` of the 16 (gate 0..7, up 8..15) += A * B.
fn wmma(b: &mut Builder, spec: &Spec, t: u8, a: u8, bb: u8) -> R {
    let acc = spec.acc() + 8 * t;
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_f16 {}, {}, {}, {}",
            rs(acc, 8),
            rs(a, 8),
            rs(bb, 8),
            rs(acc, 8)
        ),
        &[vr(acc, 8)],
        &[vr(a, 8), vr(bb, 8), vr(acc, 8)],
    )
}
fn slot_base(slot: usize) -> u32 {
    slot as u32 * SLOT_BYTES
}

/// One scheduled operation interleaved between WMMAs.
#[derive(Clone)]
enum Fill {
    /// B fragment `f` of a chunk (K-step `f / 8`, token tile `f % 8`) from `slot` into its ring register.
    BLoad { slot: usize, f: u8 },
    /// Store X staging uint4 `i` into `slot`.
    XStore { slot: usize, i: u8 },
    /// 16 B of an A fragment half: `dst` <- `srd` rows at `VAOFF + soff + imm`.
    ALoad {
        dst: u8,
        srd: u8,
        soff: u8,
        imm: u32,
    },
}
fn emit_fill(b: &mut Builder, spec: &Spec, f: &Fill) -> R {
    match f {
        Fill::BLoad { slot, f } => {
            let (j, t) = (u32::from(f / 8), u32::from(f % 8));
            let dst = spec.ring(*f);
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
            let data = spec.xr() + 4 * i;
            let imm = u32::from(*i) * spec.xrow_step() * PITCH;
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
        Fill::ALoad {
            dst,
            srd,
            soff,
            imm,
        } => bload(b, 4, *dst, VAOFF, *srd, *soff, *imm),
    }
}

/// The eight 16-byte loads of the A set `set` (K-steps 0 and 1 of one 32-K
/// chunk, gate then up, two halves each) at `soff + imm`, imm = `base + 32 j +
/// 16 h`.
fn a_loads(spec: &Spec, set: usize, soff: u8, base: u32) -> Vec<Fill> {
    let mut out = Vec::new();
    for j in 0..2u8 {
        for m in 0..2u8 {
            for h in 0..2u8 {
                out.push(Fill::ALoad {
                    dst: spec.abuf(set, j, m) + 4 * h,
                    srd: if m == 0 { SRD_G } else { SRD_U },
                    soff,
                    imm: base + 32 * u32::from(j) + 16 * u32::from(h),
                });
            }
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

/// One 32-K chunk `cc` (0 or 1) of the 64-K loop body: 32 WMMAs. WMMA `w`
/// is K-step `w / 16`, token tile `(w % 16) / 2`, matrix `w % 2` (0 gate, 1
/// up) of the chunk; B fragment `f = w / 2` feeds WMMAs `2f` and `2f + 1`.
fn chunk(b: &mut Builder, spec: &Spec, cc: u8) -> R {
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
    spread(&mut slots, a_loads(spec, 1 - slot, soff, base), 0, 16);
    // B ring: after the second WMMA of fragment f its register takes fragment f + RING.
    for w in (1..WPC).step_by(2) {
        let f = w / 2;
        if f + RING < 16 {
            slots[usize::from(w)].push(Fill::BLoad { slot, f: f + RING });
        }
    }
    // Stage chunk c+1's X into the other slot, well before the barrier.
    let stores: Vec<Fill> = (0..spec.xpt())
        .map(|i| Fill::XStore { slot: other, i })
        .collect();
    spread(&mut slots, stores, 12, 23);
    for w in 0..WPC {
        let (f, m) = (w / 2, w % 2);
        let (j, t) = (f / 8, f % 8);
        wmma(b, spec, m * 8 + t, spec.abuf(slot, j, m), spec.ring(f))?;
        for fill in &slots[usize::from(w)] {
            emit_fill(b, spec, fill)?;
        }
        if w == BAR_AT {
            // This wave's reads of `slot` have landed (the last ring loads of
            // this chunk); publish the other slot, retire this one.
            b.wait(Counter::Lgkm, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in 0..spec.xpt() {
                bload(b, 4, spec.xr() + 4 * x, XOFF + x, SRD_X, SXOFF, 0)?;
            }
        } else if w > BAR_AT && w % 2 == 1 {
            // The next chunk's first fragments, from the published slot.
            emit_fill(
                b,
                spec,
                &Fill::BLoad {
                    slot: other,
                    f: w / 2 + RING - 16,
                },
            )?;
        }
    }
    Ok(())
}

fn prologue(b: &mut Builder, spec: &Spec) -> R {
    let t = |n: u8| spec.ringb() + n; // prologue temporaries alias the (not yet loaded) ring
    let (t0, t1, t2, t3, t4, t5) = (t(0), t(1), t(2), t(3), t(4), t(5));
    let rows = spec.channels();
    smem(b, PG, 8, KARG, 0)?;
    smem(b, PS, 2, KARG, 0x20)?;
    smem(b, SI, 2, KARG, 0x28)?;
    smem(b, SN, 1, KARG, 0x30)?;
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
            format!("v_mov_b32_e32 v{}, 0", spec.acc() + n),
            &[v(spec.acc() + n)],
            &[],
        )?;
    }
    b.wait(Counter::Lgkm, 0)?;
    // I, K, N > 0 else exit; K < 64: no chunks, nothing stored; workgroup
    // column past the channels: exit.
    sop(
        b,
        format!("s_min_i32 s{ST0}, s{SI}, s{SK}"),
        &[ST0],
        &[SI, SK],
    )?;
    sop(
        b,
        format!("s_min_i32 s{ST0}, s{ST0}, s{SN}"),
        &[ST0],
        &[ST0, SN],
    )?;
    sop(b, format!("s_cmp_lt_i32 s{ST0}, 1"), &[], &[ST0])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(b, format!("s_cmp_lt_i32 s{SK}, 64"), &[], &[SK])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(rows)),
        &[ST0],
        &[WGX],
    )?;
    sop(b, format!("s_cmp_ge_i32 s{ST0}, s{SI}"), &[], &[ST0, SI])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
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
    // Buffer descriptors: weights, Y and the selector raw unbounded
    // (0x7fffffff), X bounded at N*K*2 (tokens past N read zeros).
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{SN}, s{SK2}"),
        &[ST1],
        &[SN, SK2],
    )?;
    for (srd, p) in [
        (SRD_G, PG),
        (SRD_U, PU),
        (SRD_X, PX),
        (SRD_Y, PY),
        (SRD_S, PS),
    ] {
        sop(b, format!("s_mov_b32 s{srd}, s{p}"), &[srd], &[p])?;
        sop(
            b,
            format!("s_and_b32 s{}, s{}, 0xffff", srd + 1, p + 1),
            &[srd + 1],
            &[p + 1],
        )?;
        if srd == SRD_X {
            sop(
                b,
                format!("s_mov_b32 s{}, s{ST1}", srd + 2),
                &[srd + 2],
                &[ST1],
            )?;
        } else {
            sop(
                b,
                format!("s_mov_b32 s{}, 0x7fffffff", srd + 2),
                &[srd + 2],
                &[],
            )?;
        }
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
    // Weight row byte offset of lane (wave channel start + ml), clamped to the last row.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t3}, 1, v{TID}"),
        &[v(t3)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t3}, {}, v{t3}", lit(rows - 16)),
        &[v(t3)],
        &[v(t3)],
    )?; // 16 * wave
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(rows)),
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
    sop(b, format!("s_add_i32 s{ST1}, s{SI}, -1"), &[ST1], &[SI])?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASKX}, s{SI}, v{t2}"),
        &[s(MASKX)],
        &[s(SI), v(t2)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{t5}, s{ST1}"),
        &[v(t5)],
        &[s(ST1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t2}, v{t5}, v{t2}, s{MASKX}"),
        &[v(t2)],
        &[v(t5), v(t2), s(MASKX)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{VAOFF}, v{t2}, s{SK2}"),
        &[v(VAOFF)],
        &[v(t2), s(SK2)],
    )?;
    // X rows: token wgy*128 + i*step + tid/4, byte (tid%4)*16 of a 32-K chunk.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 7"), &[ST0], &[WGY])?;
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
    for i in 0..spec.xpt() {
        let x = XOFF + i;
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{x}, {}, v{t2}",
                lit(u32::from(i) * spec.xrow_step())
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
    for i in 0..spec.xpt() {
        bload(b, 4, spec.xr() + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in a_loads(spec, 0, SA, 0) {
        emit_fill(b, spec, &f)?;
    }
    for i in 0..spec.xpt() {
        emit_fill(b, spec, &Fill::XStore { slot: 0, i })?;
    }
    b.barrier(&[Transition::Ready(0)])?;
    // K >= 64: chunk 1 exists (min(1, K/32 - 1) = 1).
    sop(b, format!("s_mov_b32 s{SXOFF}, 64"), &[SXOFF], &[])?;
    for i in 0..spec.xpt() {
        bload(b, 4, spec.xr() + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in 0..RING {
        emit_fill(b, spec, &Fill::BLoad { slot: 0, f })?;
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Spec) -> R {
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
        chunk(b, spec, cc)?;
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
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

// ------------------------------------------------------------------- epilogue

/// hipcc's `silu_mul_bf16_rt` (finite-only BF16 RNE, non-finite unchanged) in
/// place over `x[k]`, the ops of the activation kernel's ISA, interleaved op
/// by op over the elements (`rt[k]`: temporary, `RMASK + 2k`: class mask).
fn round_bf16(b: &mut Builder, x: &[u8], rt: u8) -> R {
    let m = |k: usize| RMASK + 2 * k as u8;
    for k in 0..x.len() {
        op(
            b,
            format!("v_bfe_u32 v{}, v{}, 16, 1", rt + k as u8, x[k]),
            &[v(rt + k as u8)],
            &[v(x[k])],
        )?;
    }
    for k in 0..x.len() {
        op(
            b,
            format!("v_cmp_class_f32_e64 s{}, v{}, 0x1f8", m(k), x[k]),
            &[s(m(k))],
            &[v(x[k])],
        )?;
    }
    for k in 0..x.len() {
        op(
            b,
            format!("v_add3_u32 v{0}, v{1}, v{0}, 0x7fff", rt + k as u8, x[k]),
            &[v(rt + k as u8)],
            &[v(x[k]), v(rt + k as u8)],
        )?;
    }
    for k in 0..x.len() {
        op(
            b,
            format!("v_and_b32_e32 v{0}, 0xffff0000, v{0}", rt + k as u8),
            &[v(rt + k as u8)],
            &[v(rt + k as u8)],
        )?;
    }
    for k in 0..x.len() {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{0}, v{0}, v{1}, s{2}",
                x[k],
                rt + k as u8,
                m(k)
            ),
            &[v(x[k])],
            &[v(x[k]), v(rt + k as u8), s(m(k))],
        )?;
    }
    Ok(())
}

/// The epilogue of the 16 channel x 16 token tile pair `t`: GEMM epilogue
/// `0.0f + acc`, the activation, F16 conversion and the exchange that leaves
/// lane `(ml, lh)` holding the eight consecutive channels `8 lh ..` of token
/// `ml` as four packed dwords at `e.out + 4 t`.
fn tile(b: &mut Builder, spec: &Spec, e: &Ep, t: u8) -> R {
    let (g, u) = (spec.accg(t), spec.accu(t));
    for n in 0..8u8 {
        op(
            b,
            format!("v_add_f32_e32 v{0}, 0, v{0}", g + n),
            &[v(g + n)],
            &[v(g + n)],
        )?;
        op(
            b,
            format!("v_add_f32_e32 v{0}, 0, v{0}", u + n),
            &[v(u + n)],
            &[v(u + n)],
        )?;
    }
    for grp in 0..2u8 {
        let xs = |base: u8| {
            [
                base + 4 * grp,
                base + 4 * grp + 1,
                base + 4 * grp + 2,
                base + 4 * grp + 3,
            ]
        };
        round_bf16(b, &xs(g), e.rt)?;
        round_bf16(b, &xs(u), e.rt)?;
        Epilogue::silu_dense_with(
            b,
            g + 4 * grp,
            u + 4 * grp,
            e.tmp,
            SMASK,
            4,
            DivNum::Gate,
            Finish::MulUpFirst,
        )?;
        round_bf16(b, &xs(g), e.rt)?;
    }
    for n in 0..8u8 {
        op(
            b,
            format!("v_cvt_f16_f32_e32 v{0}.l, v{0}", g + n),
            &[v(g + n)],
            &[v(g + n)],
        )?;
    }
    // Lane lh sends h[4 (1-lh) + j] (j = 0..3), keeps h[4 lh + j]; the pair
    // (lane, lane ^ 16) then holds channel 2e + lh of token ml in lane lh's
    // register e.
    for j in 0..4u8 {
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{LHM}",
                e.o + j,
                g + j,
                g + 4 + j
            ),
            &[v(e.o + j)],
            &[v(g + j), v(g + 4 + j), s(LHM)],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{LHM}",
                e.sv + j,
                g + 4 + j,
                g + j
            ),
            &[v(e.sv + j)],
            &[v(g + 4 + j), v(g + j), s(LHM)],
        )?;
    }
    for p in [0u8, 2] {
        op(
            b,
            format!(
                "v_perm_b32 v{0}, v{1}, v{0}, 0x5040100",
                e.sv + p,
                e.sv + p + 1
            ),
            &[v(e.sv + p)],
            &[v(e.sv + p + 1), v(e.sv + p)],
        )?;
        op(
            b,
            format!(
                "v_permlanex16_b32 v{0}, v{0}, s{PSEL}, 0xfedcba98",
                e.sv + p
            ),
            &[v(e.sv + p)],
            &[v(e.sv + p), s(PSEL)],
        )?;
    }
    for j in 0..4u8 {
        let (sel, tp) = (if j % 2 == 0 { e.sele } else { e.selo }, e.sv + 2 * (j / 2));
        op(
            b,
            format!(
                "v_perm_b32 v{}, v{}, v{}, v{}",
                e.out + 4 * t + j,
                e.o + j,
                tp,
                sel
            ),
            &[v(e.out + 4 * t + j)],
            &[v(e.o + j), v(tp), v(sel)],
        )?;
    }
    Ok(())
}

fn tiles(b: &mut Builder, spec: &Spec) -> R {
    let e = ep(spec);
    let rows = spec.channels();
    // v_perm selectors of the pair exchange: lane half 0 packs (own, received), lane half 1 (received, own).
    op(
        b,
        format!("v_mov_b32_e32 v{}, 0x1000504", e.sele),
        &[v(e.sele)],
        &[],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{0}, v{0}, 0x5040100, s{LHM}", e.sele),
        &[v(e.sele)],
        &[v(e.sele), s(LHM)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, 0x3020504", e.selo),
        &[v(e.selo)],
        &[],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{0}, v{0}, 0x5040302, s{LHM}", e.selo),
        &[v(e.selo)],
        &[v(e.selo), s(LHM)],
    )?;
    // Store addresses: byte ((tok0 + ml) * I + channel start + 8 lh) * 2, tiles 16 tokens apart.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 7"), &[ST0], &[WGY])?;
    sop(b, format!("s_lshl_b32 s{SI2}, s{SI}, 1"), &[SI2], &[SI])?;
    sop(
        b,
        format!("s_lshl_b32 s{SYSTEP}, s{SI}, 5"),
        &[SYSTEP],
        &[SI],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{SNB}, s{SN}, s{ST0}"),
        &[SNB],
        &[SN, ST0],
    )?;
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    op(
        b,
        format!("v_and_b32_e32 v{}, 15, v{TID}", e.vml),
        &[v(e.vml)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{}, s{ST0}, v{}", e.offs, e.vml),
        &[v(e.offs)],
        &[s(ST0), v(e.vml)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{0}, v{0}, s{SI2}", e.offs),
        &[v(e.offs)],
        &[v(e.offs), s(SI2)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{}, 1, v{TID}", e.vcs),
        &[v(e.vcs)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{0}, {1}, v{0}", e.vcs, lit(rows - 16)),
        &[v(e.vcs)],
        &[v(e.vcs)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{WGX}, {}", lit(rows)),
        &[ST1],
        &[WGX],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, s{ST1}, v{0}", e.vcs),
        &[v(e.vcs)],
        &[s(ST1), v(e.vcs)],
    )?; // channel start
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SI}, v{}", e.vcs),
        &[s(MASK1)],
        &[s(SI), v(e.vcs)],
    )?; // channels < I
    op(
        b,
        format!("v_and_b32_e32 v{}, 16, v{TID}", e.o),
        &[v(e.o)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{0}, v{0}, 1, v{1}", e.vcs, e.o),
        &[v(e.vcs)],
        &[v(e.vcs), v(e.o)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.offs, e.vcs),
        &[v(e.offs)],
        &[v(e.offs), v(e.vcs)],
    )?;
    for t in 1..8u8 {
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{}, s{SYSTEP}, v{}",
                e.offs + t,
                e.offs + t - 1
            ),
            &[v(e.offs + t)],
            &[s(SYSTEP), v(e.offs + t - 1)],
        )?;
    }
    for t in 0..8u8 {
        tile(b, spec, &e, t)?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{}", e.vml),
            &[s(MASK0)],
            &[s(SNB), v(e.vml)],
        )?;
        op(
            b,
            format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
            &[],
            &[s(MASK0), s(MASK1)],
        )?;
        bstore(b, 4, e.out + 4 * t, e.offs + t, SRD_Y, SZERO, 0)?;
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        sop(b, format!("s_add_i32 s{SNB}, s{SNB}, -16"), &[SNB], &[SNB])?;
    }
    Ok(())
}

/// `selector[n] = 1.0f / (1.0f + expf(-R(selector[n])))` for the 128 tokens
/// of the token block, one thread each, by the workgroup column 0 only.
fn selector(b: &mut Builder, spec: &Spec) -> R {
    let e = ep(spec);
    sop(b, format!("s_cmp_lg_u32 s{WGX}, 0"), &[], &[WGX])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    let (sx, soff, stok) = (e.sele, e.selo, e.vcs);
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 7"), &[ST0], &[WGY])?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{stok}, s{ST0}, v{TID}"),
        &[v(stok)],
        &[s(ST0), v(TID)],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASKX}, s{SN}, v{stok}"),
        &[s(MASKX)],
        &[s(SN), v(stok)],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{MASK0}, 0x80, v{TID}"),
        &[s(MASK0)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, s{MASKX}, s{MASK0}"),
        &[],
        &[s(MASKX), s(MASK0)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{soff}, 2, v{stok}"),
        &[v(soff)],
        &[v(stok)],
    )?;
    bload(b, 1, sx, soff, SRD_S, SZERO, 0)?;
    round_bf16(b, &[sx], e.rt)?;
    Epilogue::silu_dense_with(b, sx, sx, e.tmp, SMASK, 1, DivNum::One, Finish::Quotient)?;
    bstore(b, 1, e.tmp + 5, soff, SRD_S, SZERO, 0)?;
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

fn epilogue(b: &mut Builder, spec: &Spec) -> R {
    b.wait_all()?;
    b.label(&lbl(spec, "epi"))?;
    tiles(b, spec)?;
    selector(b, spec)?;
    b.label(&lbl(spec, "end"))?;
    b.control(Sop::End.encode(Arch::Gfx1151)?)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(52)
        .pointer("Wg", 0)
        .pointer("Wu", 8)
        .pointer("X", 16)
        .pointer("Y", 24)
        .pointer("Sel", 32)
        .hidden("I", 40, 4, "by_value")
        .hidden("K", 44, 4, "by_value")
        .hidden("N", 48, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "qwen4_shared_gu".into(),
        variant: format!("w{}", spec.wr),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads(),
        group_segment_fixed_size: GROUP_BYTES,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&spec)?);
    b.enable_delay_alu();
    for (id, (name, base)) in [("x_slot0", 0u32), ("x_slot1", SLOT_BYTES)]
        .into_iter()
        .enumerate()
    {
        if b.lds_slot(name, base, SLOT_BYTES)? != id {
            return Err("LDS slot order".into());
        }
    }
    prologue(&mut b, &spec)?;
    b.loop_(&lbl(&spec, "loop"), |b| body(b, &spec))?;
    epilogue(&mut b, &spec)?;
    b.finish()
}

/// Every entry as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Spec::ALL
        .into_iter()
        .map(|wr| emit(Spec { arch, wr }))
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}
