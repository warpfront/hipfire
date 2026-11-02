// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 MQ6 trunk GEMM (`gemm_mq6g256v2_wmma_gfx12_bt8_x4`), emitted with
//! the checked ISA builder as an exact twin of the hipcc kernel in
//! `kernels/src/qwen4_gemm_mq6g256v2_wmma_gfx12_x4.hip` (overwrite F32 entry).
//!
//! Exactness (byte-identical outputs): every 16x16 output tile runs the same
//! chain of `v_wmma_f32_16x16x16_f16` over ascending 16-K steps with the
//! same A and B operands, so no output depends on the tiling. A has the same
//! f16 values as hipcc's dequantization, element for element (`sc*q + zp`,
//! one RNE rounding, with the group header's f16 scale / zero point: hA for
//! K-steps 0..7 of a 256-group, hB for 8..15), but two codes at a time (the
//! gfx1151 twin's construction): the 6-bit codes of a pair are placed in the
//! mantissas of two f16 with a constant exponent (`1024 + q`, `64 + q` or
//! `256 + q`, all exact for q < 64), `v_pk_add_f16` of the bias recovers `q`
//! exactly (the same f16 as `v_cvt_f32_ubyte0` + `v_cvt_f16_f32`), and one
//! `v_pk_fma_f16` (operands sc, q, zp; sc and zp read from the header halves
//! with `op_sel`) is the same fused expression as hipcc's `v_fma_f16`. The
//! epilogue stores `0.0f + acc` (`v_add_f32 0, acc`), as hipcc's overwrite
//! path.
//!
//! Freedom used (tiling, layout and scheduling only):
//!  * a wave owns 16 rows x 256 tokens (16 accumulator tiles), so one A
//!    dequant feeds 16 WMMAs (hipcc: 8); a workgroup is `WR` waves (64, 128
//!    or 256 rows) x 256 tokens;
//!  * X is staged in 32-K chunks, double-buffered in LDS (two 20480-byte
//!    slots, 80-byte token pitch), one barrier per chunk;
//!  * B fragments rotate through an 8-deep register ring loaded eight WMMAs
//!    ahead (no per-WMMA `s_wait_dscnt 0`); the next K-step's A dequant is
//!    interleaved with the current WMMAs;
//!  * each lane loads only its own 48 code bits per K-step (two dwords at
//!    `12j + 4*lane_half`) and gathers them with four `v_perm_b32`;
//!  * X loads go through a descriptor bounded at `N*K*2` bytes, so tokens
//!    past N read zeros (their outputs are never stored);
//!  * W8/W16 stage the output (`M % 8 == 0`) through LDS in `WR/2` passes of
//!    `512/WR` tokens, so each store writes contiguous rows of one token
//!    (whole 128-byte lines); W4 and ragged M store straight from the tiles.
//!
//! Launch: block `32*WR`, grid `[ceil(M/(16*WR)), ceil(N/256), 1]`, kernargs
//! `(A, X, Y, M, K, N)` (36 bytes), no dynamic LDS. K is a multiple of 256
//! (a ragged K tail is ignored, as in hipcc).
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, sop, sr, v, vr};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// `wr` waves (16 rows each) per workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub wr: u8,
}
impl Spec {
    pub const ALL: [u8; 3] = [4, 8, 16];
    pub fn symbol(&self) -> String {
        format!("qwen4_mq6_x4_pm_gfx1201_w{}", self.wr)
    }
    pub fn module(arch: Arch) -> String {
        format!("qwen4_mq6_x4_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("qwen4_mq6_x4 builder is exact-gfx1201 only".into());
        }
        if !Self::ALL.contains(&self.wr) {
            return Err(format!(
                "qwen4_mq6_x4: wr {} not in {:?}",
                self.wr,
                Self::ALL
            ));
        }
        Ok(())
    }
    /// Rows per workgroup.
    pub fn rows(&self) -> u32 {
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
    /// W8/W16 stage the `M % 8 == 0` output through LDS for coalesced
    /// stores; W4 keeps the direct per-tile stores.
    fn staged_epilogue(&self) -> bool {
        self.wr >= 8
    }
}

/// Tokens per workgroup (and per wave): 16 tiles of 16.
pub const TOK: u32 = 256;
const BT: u8 = 16;
const PITCH: u32 = 80; // 32 K f16 + 8 pad per token row
pub const SLOT_BYTES: u32 = TOK * PITCH; // 20480
pub const GROUP_BYTES: u32 = 2 * SLOT_BYTES;
/// 32-K chunks per 256-K group; WMMAs per chunk; B ring depth.
const CHUNKS: u8 = 8;
const WPC: u8 = 2 * BT;
const RING: u8 = 8;
/// WMMA after which the chunk barrier sits (its ring loads stop at RING before).
const BAR_AT: u8 = WPC - 5;
const L_X0: usize = 0;

/// Local labels, qualified by variant so both entries share one module.
fn lbl(spec: &Spec, name: &str) -> String {
    format!(".Lmq6w{}_{name}", spec.wr)
}

// ------------------------------------------------------------------ registers
// VGPRs. v0 tid; v1..v15 lane constants; v16..v23 X buffer offsets;
// v24..v151 accumulators (16 token tiles x 8); v152..v183 B ring (8 x 4);
// v184..v191 A fragments (two); v192..v207 codes (four chunks: loaded two
// chunks ahead); v208..v211 group headers (current, next); v212..v215
// dequant temporaries (the four code pairs); v220..v221 two more lane
// constants (code-pair selectors); v224.. X staging (4*xpt).
const TID: u8 = 0;
/// `v_perm_b32` byte selectors of the four code pairs (this lane half's window).
const SEL: [u8; 4] = [1, 2, 220, 221];
const ACODE: u8 = 3;
const AHDR: u8 = 4;
const LDST: u8 = 5;
const LDRD: u8 = 6;
const VTOK: u8 = 7;
const VYOFF: u8 = 8;
const VROW0: u8 = 9;
const T0: u8 = 10;
const T1: u8 = 11;
const T2: u8 = 12;
const T3: u8 = 13;
const T4: u8 = 14;
const T5: u8 = 15;
const XOFF: u8 = 16;
const ACC: u8 = 24;
const RINGB: u8 = 152;
const AF: [u8; 2] = [184, 188];
const CODE: [u8; 4] = [192, 196, 200, 204];
const HDR: [u8; 2] = [208, 210];
const Q: u8 = 212;
const XR: u8 = 224;

// SGPRs.
const KARG: u8 = 0;
const SRD_A: u8 = 4;
const SRD_X: u8 = 8;
const SRD_Y: u8 = 12;
const PA: u8 = 16;
const PX: u8 = 18;
const PY: u8 = 20;
const SM: u8 = 22;
const SK: u8 = 23;
const SN: u8 = 24;
const GPR: u8 = 25;
const NCHM1: u8 = 26;
const GPRM1: u8 = 27;
const WGX: u8 = 28;
const WGY: u8 = 29;
const G: u8 = 30;
const GOFF: u8 = 31;
const NGOFF: u8 = 32;
const SXOFF: u8 = 33;
const C8: u8 = 34;
const ST0: u8 = 35;
const ST1: u8 = 36;
const SZERO: u8 = 37;
const YSTEP: u8 = 40;
const SNB: u8 = 41;
const SMJ: u8 = 42;
const SK2: u8 = 43;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
const MASK0: u8 = 44;
const MASK1: u8 = 46;
/// Packed-dequant constants (live from the prologue to the end): field masks of
/// the even / odd code pairs and the `v_pk_add_f16` constants that remove the
/// exponent bias (`-1024` / `-64` and `-256` per half).
const PK_MASKE: u8 = 48;
const PK_MASKO: u8 = 49;
const PK_CE: u8 = 50;
const PK_CO: u8 = 51;

fn plan(spec: &Spec) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(XR) + 4 * u16::from(spec.xpt()), 52)?;
    let w = || Live::Whole;
    for n in 0..16u8 {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in SEL[2]..=SEL[3] {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in 0..spec.xpt() {
        p.v::<1>("x_offset", XOFF + n, w())?;
    }
    for t in 0..BT {
        p.v::<8>("acc", ACC + 8 * t, w())?;
    }
    for r in 0..RING {
        p.v::<4>("b_ring", RINGB + 4 * r, w())?;
    }
    for a in AF {
        p.v::<4>("a_frag", a, w())?;
    }
    for c in CODE {
        for j in 0..2u8 {
            p.v::<2>("codes", c + 2 * j, w())?;
        }
    }
    for h in HDR {
        p.v::<2>("header", h, w())?;
    }
    for n in Q..Q + 4 {
        p.v::<1>("dequant_temp", n, w())?;
    }
    for i in 0..spec.xpt() {
        p.v::<4>("x_stage", XR + 4 * i, w())?;
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    for srd in [SRD_A, SRD_X, SRD_Y] {
        p.s::<4>("srd", srd, w())?;
    }
    p.s::<8>("kernargs", PA, w())?;
    for n in SN..=SZERO {
        p.s::<1>("scalar", n, w())?;
    }
    for n in YSTEP..=SK2 {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1] {
        p.s::<2>("lane_mask", m, w())?;
    }
    for n in PK_MASKE..=PK_CO {
        p.s::<1>("pk_const", n, w())?;
    }
    Ok(p)
}

fn wait_alu(b: &mut Builder, what: &str) -> R {
    op(b, format!("s_wait_alu {what}"), &[], &[])
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
fn wmma(b: &mut Builder, t: u8, a: u8, bb: u8) -> R {
    let acc = ACC + 8 * t;
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_f16 {}, {}, {}, {}",
            rs(acc, 8),
            rs(a, 4),
            rs(bb, 4),
            rs(acc, 8)
        ),
        &[vr(acc, 8)],
        &[vr(a, 4), vr(bb, 4), vr(acc, 8)],
    )
}
fn slot_base(slot: usize) -> u32 {
    slot as u32 * SLOT_BYTES
}
fn ring(i: u8) -> u8 {
    RINGB + 4 * (i % RING)
}
/// Code buffer of chunk `cc` (mod 4; a group's 8 chunks keep the rotation).
fn code(cc: u8) -> u8 {
    CODE[usize::from(cc % 4)]
}

/// One scheduled operation interleaved between WMMAs.
#[derive(Clone)]
enum Fill {
    /// B fragment of chunk WMMA `k` (K-step k/16, token tile k%16) from `slot`.
    BLoad { slot: usize, k: u8 },
    /// Store X staging uint4 `i` into `slot` at slot-relative `imm`.
    XStore { slot: usize, i: u8, imm: u32 },
    /// Codes of K-step `j` of chunk `cc` (of the group at `soff`) into `dst`.
    CodeLoad { dst: u8, soff: u8, cc: u8, j: u8 },
    /// Group header (hA, hB) of the group at `soff` into `dst`.
    HdrLoad { dst: u8, soff: u8 },
    Valu {
        text: String,
        defs: Vec<RegRef>,
        uses: Vec<RegRef>,
    },
}
fn emit_fill(b: &mut Builder, f: &Fill) -> R {
    match f {
        Fill::BLoad { slot, k } => {
            let (j, t) = (u32::from(k / BT), u32::from(k % BT));
            let dst = ring(*k);
            let imm = slot_base(*slot) + t * 16 * PITCH + j * 32;
            b.ds_load(
                *slot,
                Instruction::new(
                    format!("ds_load_b128 {}, v{LDRD}{}", rs(dst, 4), off(imm)),
                    vec![vr(dst, 4)],
                    vec![v(LDRD)],
                )
                .memory(MemoryClass::DsLoad),
            )
        }
        Fill::XStore { slot, i, imm } => {
            let data = XR + 4 * i;
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
        Fill::CodeLoad { dst, soff, cc, j } => bload(
            b,
            2,
            *dst,
            ACODE,
            SRD_A,
            *soff,
            8 + u32::from(*cc) * 24 + u32::from(*j) * 12,
        ),
        Fill::HdrLoad { dst, soff } => bload(b, 2, *dst, AHDR, SRD_A, *soff, 0),
        Fill::Valu { text, defs, uses } => op(b, text.clone(), defs, uses),
    }
}
fn xstore(spec: &Spec, slot: usize, i: u8) -> Fill {
    Fill::XStore {
        slot,
        i,
        imm: u32::from(i) * spec.xrow_step() * PITCH,
    }
}

/// `v_perm_b32` selectors (lane half 0, lane half 1) of the four code pairs:
/// the lane's 48-bit window is bytes `2*lh ..` of its 8 loaded bytes. Pair `p`
/// (codes 2p and 2p+1, window bits 12p..12p+11) is gathered into a dword whose
/// two 16-bit halves each hold one six-bit field within the low five bits of
/// the half (the other bytes are zeroed or ignored by the mask):
///  * pairs 0 and 2 (byte-aligned codes 2p at offset 0, 2p+1 at offset 6):
///    `u = [b0, b1, 0, 0]` of the pair's two bytes; `(u << 10) | u` moves
///    code 2p+1 to bit 16 (code 2p stays at bit 0);
///  * pairs 1 and 3 (code 2p at byte offset 4, code 2p+1 at offset 2 of the
///    next byte): `[b0, b1, b1, 0]` already puts code 2p at bit 4 of the low
///    half and code 2p+1 at bit 18 of the high half.
const PAIR_SEL: [(u32, u32); 4] = [
    (0x0c0c_0100, 0x0c0c_0302),
    (0x0c02_0201, 0x0c04_0403),
    (0x0c0c_0403, 0x0c0c_0605),
    (0x0c05_0504, 0x0c07_0706),
];

/// Dequantize one K-step: codes at `w` (two dwords: this lane half's 48 bits
/// at bit 0 for lane half 0, bit 16 for lane half 1), header dword `h`
/// (sc in .l, zp in .h) into A fragment `a` (codes 2p, 2p+1 in `a+p` .l/.h).
/// Packed pairs (exact: same f16 values as `sc*q + zp` per code with one
/// `v_fma_f16` rounding): the code pair is placed in the mantissas of two f16
/// with a constant exponent (`v_and_or_b32` against the pair's field mask and
/// exponent bits: `0x6400 | q` is `1024 + q` for a field at bit 0,
/// `0x5400 | q << 4` is `64 + q` for a field at bit 4, `0x5c00 | q << 2` is
/// `256 + q` for a field at bit 2), `v_pk_add_f16` of the bias recovers `q`
/// exactly, and `v_pk_fma_f16` (sc and zp straight from the header halves
/// through `op_sel`, operand order sc, q, zp) writes the fragment pair.
/// Breadth-first order: four independent per-pair chains.
fn dequant(w: u8, h: u8, a: u8) -> Vec<Fill> {
    let mut out = Vec::new();
    let mut f = |text: String, defs: Vec<RegRef>, uses: Vec<RegRef>| {
        out.push(Fill::Valu { text, defs, uses })
    };
    for p in 0..4u8 {
        let sel = SEL[usize::from(p)];
        f(
            format!("v_perm_b32 v{}, v{}, v{w}, v{sel}", Q + p, w + 1),
            vec![v(Q + p)],
            vec![v(w), v(w + 1), v(sel)],
        );
    }
    for p in [0u8, 2] {
        f(
            format!("v_lshl_or_b32 v{0}, v{0}, 10, v{0}", Q + p),
            vec![v(Q + p)],
            vec![v(Q + p)],
        );
    }
    for p in 0..4u8 {
        let (mask, exp) = if p % 2 == 0 {
            (PK_MASKE, 0x6400_6400u32)
        } else {
            (PK_MASKO, 0x5c00_5400)
        };
        f(
            format!("v_and_or_b32 v{0}, v{0}, s{mask}, {1}", Q + p, lit(exp)),
            vec![v(Q + p)],
            vec![v(Q + p), s(mask)],
        );
    }
    for p in 0..4u8 {
        let bias = if p % 2 == 0 { PK_CE } else { PK_CO };
        f(
            format!("v_pk_add_f16 v{0}, v{0}, s{bias}", Q + p),
            vec![v(Q + p)],
            vec![v(Q + p), s(bias)],
        );
    }
    for p in 0..4u8 {
        f(
            format!(
                "v_pk_fma_f16 v{}, v{h}, v{}, v{h} op_sel:[0,0,1] op_sel_hi:[0,1,1]",
                a + p,
                Q + p
            ),
            vec![v(a + p)],
            vec![v(h), v(Q + p)],
        );
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

/// Header dword of chunk `cc` of a group: hA for K-steps 0..7, hB for 8..15.
fn hdr_dword(cc: u8) -> u8 {
    HDR[0] + u8::from(cc >= CHUNKS / 2)
}

/// One 32-K chunk `cc` (0..7) of the group loop body: 32 WMMAs.
fn chunk(b: &mut Builder, spec: &Spec, cc: u8) -> R {
    let slot = usize::from(cc % 2);
    let other = 1 - slot;
    let cb = code(cc);
    let ncb = code(cc + 1);
    // X chunk 8g+cc+2 (clamped): its loads follow this chunk's barrier.
    sop(
        b,
        format!("s_add_co_i32 s{SXOFF}, s{C8}, {}", cc + 2),
        &[SXOFF],
        &[C8],
    )?;
    sop(
        b,
        format!("s_min_u32 s{SXOFF}, s{SXOFF}, s{NCHM1}"),
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
    // Codes two chunks ahead (+ the next group's header two chunks before
    // its first use: the last chunk's next-K-step dequant).
    let (psoff, pcc) = if cc + 2 < CHUNKS {
        (GOFF, cc + 2)
    } else {
        (NGOFF, cc + 2 - CHUNKS)
    };
    for j in 0..2u8 {
        slots[0].push(Fill::CodeLoad {
            dst: code(cc + 2) + 2 * j,
            soff: psoff,
            cc: pcc,
            j,
        });
    }
    if cc + 2 == CHUNKS {
        slots[0].push(Fill::HdrLoad {
            dst: HDR[1],
            soff: NGOFF,
        });
    }
    // B ring: WMMA i loads WMMA i+8's fragment into the register it just read.
    for i in 0..WPC {
        if i + RING < WPC && i < BAR_AT {
            slots[usize::from(i)].push(Fill::BLoad { slot, k: i + RING });
        }
    }
    // Stage chunk c+1's X into the other slot, well before the barrier.
    let stores: Vec<Fill> = (0..spec.xpt()).map(|i| xstore(spec, other, i)).collect();
    spread(&mut slots, stores, 18, 26);
    // A(j=1) under K-step 0; A of the next chunk's K-step 0 under K-step 1.
    spread(
        &mut slots,
        dequant(cb + 2, hdr_dword(cc), AF[1]),
        0,
        usize::from(BT) - 1,
    );
    let nh = if cc + 1 == CHUNKS {
        HDR[1]
    } else {
        hdr_dword(cc + 1)
    };
    spread(
        &mut slots,
        dequant(ncb, nh, AF[0]),
        usize::from(BT),
        usize::from(WPC) - 1,
    );
    for i in 0..WPC {
        let (j, t) = (i / BT, i % BT);
        wmma(b, t, AF[usize::from(j)], ring(i))?;
        for f in &slots[usize::from(i)] {
            emit_fill(b, f)?;
        }
        if i == BAR_AT {
            // This wave's reads of `slot` have landed (the last ring loads of
            // this chunk); publish the other slot, retire this one.
            b.wait(Counter::Ds, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in 0..spec.xpt() {
                bload(b, 4, XR + 4 * x, XOFF + x, SRD_X, SXOFF, 0)?;
            }
            // The next chunk's first fragments, from the published slot.
            for k in 0..(WPC - 1 - BAR_AT) {
                emit_fill(b, &Fill::BLoad { slot: other, k })?;
            }
        } else if i > BAR_AT {
            emit_fill(
                b,
                &Fill::BLoad {
                    slot: other,
                    k: i + RING - WPC,
                },
            )?;
        }
    }
    Ok(())
}

fn prologue(b: &mut Builder, spec: &Spec) -> R {
    b.push(
        Instruction::new(
            "s_load_b256 s[16:23], s[0:1], 0x0",
            vec![sr(PA, 8)],
            vec![sr(KARG, 2)],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(
        Instruction::new(
            "s_load_b32 s24, s[0:1], 0x20",
            vec![s(SN)],
            vec![sr(KARG, 2)],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    // Lane constants that need no kernargs.
    op(
        b,
        format!("v_and_b32_e32 v{T0}, 15, v{TID}"),
        &[v(T0)],
        &[v(TID)],
    )?; // ml
    op(
        b,
        format!("v_bfe_u32 v{T1}, v{TID}, 4, 1"),
        &[v(T1)],
        &[v(TID)],
    )?; // lh
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T2}, 4, v{T1}"),
        &[v(T2)],
        &[v(T1)],
    )?; // 16*lh
    op(
        b,
        format!("v_mad_u32_u24 v{LDRD}, v{T0}, {}, v{T2}", lit(PITCH)),
        &[v(LDRD)],
        &[v(T0), v(T2)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T3}, 2, v{TID}"),
        &[v(T3)],
        &[v(TID)],
    )?; // tid/4
    op(
        b,
        format!("v_and_b32_e32 v{T4}, 3, v{TID}"),
        &[v(T4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T4}, 4, v{T4}"),
        &[v(T4)],
        &[v(T4)],
    )?; // (tid%4)*16
    op(
        b,
        format!("v_mad_u32_u24 v{LDST}, v{T3}, {}, v{T4}", lit(PITCH)),
        &[v(LDST)],
        &[v(T3), v(T4)],
    )?;
    op(
        b,
        format!("v_cmp_ne_u32_e64 s{MASK0}, 0, v{T1}"),
        &[s(MASK0)],
        &[v(T1)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    for (p, (l0, l1)) in PAIR_SEL.into_iter().enumerate() {
        let sel = SEL[p];
        op(
            b,
            format!("v_mov_b32_e32 v{T5}, {}", lit(l0)),
            &[v(T5)],
            &[],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{sel}, v{T5}, {}, s{MASK0}", lit(l1)),
            &[v(sel)],
            &[v(T5), s(MASK0)],
        )?;
    }
    for (n, c) in [
        (PK_MASKE, 0x003f_003fu32),
        (PK_MASKO, 0x00fc_03f0),
        (PK_CE, 0xe400_e400),
        (PK_CO, 0xdc00_d400),
    ] {
        sop(b, format!("s_mov_b32 s{n}, {}", lit(c)), &[n], &[])?;
    }
    for n in 0..8 * BT {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", ACC + n),
            &[v(ACC + n)],
            &[],
        )?;
    }
    b.wait(Counter::Km, 0)?;
    // M, K, N > 0 else exit; K < 256: no groups, store zeros.
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
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    op(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[s(WGX)], &[])?;
    op(
        b,
        format!("s_and_b32 s{WGY}, ttmp7, 0xffff"),
        &[s(WGY)],
        &[],
    )?;
    sop(b, format!("s_lshr_b32 s{GPR}, s{SK}, 8"), &[GPR], &[SK])?;
    sop(b, format!("s_lshr_b32 s{NCHM1}, s{SK}, 5"), &[NCHM1], &[SK])?;
    sop(
        b,
        format!("s_add_co_i32 s{NCHM1}, s{NCHM1}, -1"),
        &[NCHM1],
        &[NCHM1],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{GPRM1}, s{GPR}, -1"),
        &[GPRM1],
        &[GPR],
    )?;
    sop(b, format!("s_lshl_b32 s{SK2}, s{SK}, 1"), &[SK2], &[SK])?;
    // Buffer descriptors: A and Y raw unbounded (0x7fffffff), X bounded at N*K*2.
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{SN}, s{SK2}"),
        &[ST1],
        &[SN, SK2],
    )?;
    for (srd, p) in [(SRD_A, PA), (SRD_X, PX), (SRD_Y, PY)] {
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
            sop(b, format!("s_brev_b32 s{}, -2", srd + 2), &[srd + 2], &[])?;
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
    // Rows: row_start = (wgx*wr + wave)*16; row = row_start + ml.
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T3}, 1, v{TID}"),
        &[v(T3)],
        &[v(TID)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(
        b,
        format!(
            "v_and_or_b32 v{T3}, v{T3}, {}, s{ST0}",
            lit(spec.rows() - 16)
        ),
        &[v(T3)],
        &[v(T3), s(ST0)],
    )?; // row_start
    op(
        b,
        format!("v_lshl_or_b32 v{VROW0}, v{T1}, 3, v{T3}"),
        &[v(VROW0)],
        &[v(T1), v(T3)],
    )?; // row_start + 8*lh
    op(
        b,
        format!("v_or_b32_e32 v{T3}, v{T3}, v{T0}"),
        &[v(T3)],
        &[v(T3), v(T0)],
    )?; // row
    sop(b, format!("s_add_co_i32 s{ST0}, s{SM}, -1"), &[ST0], &[SM])?;
    sop(b, format!("s_mul_i32 s{ST1}, s{GPR}, 0xc8"), &[ST1], &[GPR])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(
        b,
        format!("v_min_i32_e32 v{T3}, s{ST0}, v{T3}"),
        &[v(T3)],
        &[s(ST0), v(T3)],
    )?; // safe_row
    op(
        b,
        format!("v_mul_lo_u32 v{AHDR}, v{T3}, s{ST1}"),
        &[v(AHDR)],
        &[v(T3), s(ST1)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{ACODE}, v{T1}, 2, v{AHDR}"),
        &[v(ACODE)],
        &[v(T1), v(AHDR)],
    )?;
    // Tokens: epilogue token wgy*256 + ml; X rows wgy*256 + i*step + tid/4.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 8"), &[ST0], &[WGY])?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VTOK}, s{ST0}, v{T0}"),
        &[v(VTOK)],
        &[s(ST0), v(T0)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T2}, 2, v{TID}"),
        &[v(T2)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T2}, s{ST0}, v{T2}"),
        &[v(T2)],
        &[s(ST0), v(T2)],
    )?;
    for i in 0..spec.xpt() {
        let x = XOFF + i;
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{x}, {}, v{T2}",
                lit(u32::from(i) * spec.xrow_step())
            ),
            &[v(x)],
            &[v(T2)],
        )?;
        op(
            b,
            format!("v_mul_lo_u32 v{x}, v{x}, s{SK2}"),
            &[v(x)],
            &[v(x), s(SK2)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{x}, v{x}, v{T4}"),
            &[v(x)],
            &[v(x), v(T4)],
        )?;
    }
    // Y byte offset of (token wgy*256+ml, row row_start+8*lh).
    op(
        b,
        format!("v_mul_lo_u32 v{VYOFF}, v{VTOK}, s{SM}"),
        &[v(VYOFF)],
        &[v(VTOK), s(SM)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VYOFF}, v{VYOFF}, v{VROW0}"),
        &[v(VYOFF)],
        &[v(VYOFF), v(VROW0)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VYOFF}, 2, v{VYOFF}"),
        &[v(VYOFF)],
        &[v(VYOFF)],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{GPR}, 0"), &[], &[GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "epi")), &[], &[])?;
    // Group 0: header, chunk 0's codes and X; stage X(0) in slot 0.
    sop(b, format!("s_mov_b32 s{G}, 0"), &[G], &[])?;
    sop(b, format!("s_mov_b32 s{GOFF}, 0"), &[GOFF], &[])?;
    sop(b, format!("s_mov_b32 s{C8}, 0"), &[C8], &[])?;
    sop(b, format!("s_mov_b32 s{SXOFF}, 0"), &[SXOFF], &[])?;
    bload(b, 2, HDR[0], AHDR, SRD_A, GOFF, 0)?;
    for j in 0..2u8 {
        emit_fill(
            b,
            &Fill::CodeLoad {
                dst: CODE[0] + 2 * j,
                soff: GOFF,
                cc: 0,
                j,
            },
        )?;
    }
    for i in 0..spec.xpt() {
        bload(b, 4, XR + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for i in 0..spec.xpt() {
        emit_fill(b, &xstore(spec, L_X0, i))?;
    }
    b.barrier(&[Transition::Ready(L_X0)])?;
    // Chunk 1's codes, X chunk min(1, nch-1) and the ring's first fragments
    // in flight into the loop, as the body's back edge leaves them.
    for j in 0..2u8 {
        emit_fill(
            b,
            &Fill::CodeLoad {
                dst: code(1) + 2 * j,
                soff: GOFF,
                cc: 1,
                j,
            },
        )?;
    }
    sop(
        b,
        format!("s_min_u32 s{SXOFF}, s{NCHM1}, 1"),
        &[SXOFF],
        &[NCHM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXOFF}, s{SXOFF}, 6"),
        &[SXOFF],
        &[SXOFF],
    )?;
    for i in 0..spec.xpt() {
        bload(b, 4, XR + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in dequant(CODE[0], HDR[0], AF[0]) {
        emit_fill(b, &f)?;
    }
    for k in 0..RING {
        emit_fill(b, &Fill::BLoad { slot: L_X0, k })?;
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Spec) -> R {
    // ngoff = min(g+1, gpr-1)*200 (the last group re-reads itself).
    sop(b, format!("s_add_co_i32 s{NGOFF}, s{G}, 1"), &[NGOFF], &[G])?;
    sop(
        b,
        format!("s_min_u32 s{NGOFF}, s{NGOFF}, s{GPRM1}"),
        &[NGOFF],
        &[NGOFF, GPRM1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{NGOFF}, s{NGOFF}, 0xc8"),
        &[NGOFF],
        &[NGOFF],
    )?;
    for cc in 0..CHUNKS {
        chunk(b, spec, cc)?;
    }
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", HDR[0], HDR[1]),
        &[v(HDR[0])],
        &[v(HDR[1])],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", HDR[0] + 1, HDR[1] + 1),
        &[v(HDR[0] + 1)],
        &[v(HDR[1] + 1)],
    )?;
    sop(b, format!("s_add_co_i32 s{G}, s{G}, 1"), &[G], &[G])?;
    sop(
        b,
        format!("s_add_co_i32 s{GOFF}, s{GOFF}, 0xc8"),
        &[GOFF],
        &[GOFF],
    )?;
    sop(b, format!("s_add_co_i32 s{C8}, s{C8}, 8"), &[C8], &[C8])?;
    sop(b, format!("s_cmp_lt_u32 s{G}, s{GPR}"), &[], &[G, GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

fn epilogue(b: &mut Builder, spec: &Spec) -> R {
    b.wait_all()?;
    if spec.staged_epilogue() {
        // Retire the last X slot on the loop-exit path, so both paths reach
        // `epi` with every slot Free (K < 256 branches before any slot is touched).
        b.barrier(&[Transition::Retire(0)])?;
    }
    b.label(&lbl(spec, "epi"))?;
    for n in 0..8 * BT {
        op(
            b,
            format!("v_add_f32_e32 v{0}, 0, v{0}", ACC + n),
            &[v(ACC + n)],
            &[v(ACC + n)],
        )?;
    }
    if spec.staged_epilogue() {
        sop(b, format!("s_and_b32 s{ST0}, s{SM}, 7"), &[ST0], &[SM])?;
        sop(b, format!("s_cmp_lg_u32 s{ST0}, 0"), &[], &[ST0])?;
        op(
            b,
            format!("s_cbranch_scc1 {}", lbl(spec, "scalar")),
            &[],
            &[],
        )?;
        staged_stores(b, spec)?;
        // The staged stores' sources (B ring offsets) are reused by the ragged
        // path below: release them so its offsets need no store-count wait.
        b.release_store_sources()?;
        op(b, format!("s_branch {}", lbl(spec, "end")), &[], &[])?;
        b.label(&lbl(spec, "scalar"))?;
        tile_offsets(b)?;
        sop(b, format!("s_mov_b32 s{SNB}, s{SN}"), &[SNB], &[SN])?;
    } else {
        tile_offsets(b)?;
        sop(b, format!("s_mov_b32 s{SNB}, s{SN}"), &[SNB], &[SN])?;
        sop(b, format!("s_and_b32 s{ST0}, s{SM}, 7"), &[ST0], &[SM])?;
        sop(b, format!("s_cmp_lg_u32 s{ST0}, 0"), &[], &[ST0])?;
        op(
            b,
            format!("s_cbranch_scc1 {}", lbl(spec, "scalar")),
            &[],
            &[],
        )?;
        // M % 8 == 0: whole 8-row runs (row0 < M implies row0+8 <= M).
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{VROW0}"),
            &[s(MASK1)],
            &[s(SM), v(VROW0)],
        )?;
        for t in 0..BT {
            op(
                b,
                format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
                &[s(MASK0)],
                &[s(SNB), v(VTOK)],
            )?;
            wait_alu(b, "depctr_va_sdst(0)")?;
            op(
                b,
                format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                &[],
                &[s(MASK0), s(MASK1)],
            )?;
            let acc = ACC + 8 * t;
            bstore(b, 4, acc, RINGB + t, SRD_Y, SZERO, 0)?;
            bstore(b, 4, acc + 4, RINGB + t, SRD_Y, SZERO, 16)?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            sop(
                b,
                format!("s_add_co_i32 s{SNB}, s{SNB}, -16"),
                &[SNB],
                &[SNB],
            )?;
        }
        op(b, format!("s_branch {}", lbl(spec, "end")), &[], &[])?;
        b.label(&lbl(spec, "scalar"))?;
    }
    // Ragged M: per element, row0 + e < M.
    for t in 0..BT {
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
            &[s(MASK0)],
            &[s(SNB), v(VTOK)],
        )?;
        sop(b, format!("s_mov_b32 s{SMJ}, s{SM}"), &[SMJ], &[SM])?;
        for e in 0..8u8 {
            op(
                b,
                format!("v_cmp_gt_i32_e64 s{MASK1}, s{SMJ}, v{VROW0}"),
                &[s(MASK1)],
                &[s(SMJ), v(VROW0)],
            )?;
            wait_alu(b, "depctr_va_sdst(0)")?;
            op(
                b,
                format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                &[],
                &[s(MASK0), s(MASK1)],
            )?;
            bstore(
                b,
                1,
                ACC + 8 * t + e,
                RINGB + t,
                SRD_Y,
                SZERO,
                4 * u32::from(e),
            )?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            sop(
                b,
                format!("s_add_co_i32 s{SMJ}, s{SMJ}, -1"),
                &[SMJ],
                &[SMJ],
            )?;
        }
        sop(
            b,
            format!("s_add_co_i32 s{SNB}, s{SNB}, -16"),
            &[SNB],
            &[SNB],
        )?;
    }
    b.label(&lbl(spec, "end"))?;
    b.release_store_sources()?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

/// Tile t: token + 16t, Y offset + t*16*M*4 in v[RINGB+t] (the B ring is
/// dead here), so no store source is redefined while stores are in flight.
/// Also zeroes the stores' soffset.
fn tile_offsets(b: &mut Builder) -> R {
    sop(b, format!("s_lshl_b32 s{YSTEP}, s{SM}, 6"), &[YSTEP], &[SM])?;
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    op(
        b,
        format!("v_mov_b32_e32 v{RINGB}, v{VYOFF}"),
        &[v(RINGB)],
        &[v(VYOFF)],
    )?;
    for t in 1..BT {
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{}, s{YSTEP}, v{}",
                RINGB + t,
                RINGB + t - 1
            ),
            &[v(RINGB + t)],
            &[s(YSTEP), v(RINGB + t - 1)],
        )?;
    }
    Ok(())
}

/// Coalesced `M % 8 == 0` stores of the W8/W16 entries, staged through LDS.
/// The 16*WR x 256 output tile goes out in `WR/2` passes of `P = 512/WR`
/// tokens. Each pass: every wave writes its accumulator tiles of those tokens
/// to `y_stage`, a barrier publishes them, and each thread reads back `64/WR`
/// uint4 at (token `8i + tid/R`, rows `4*(tid % R) ..+3`, `R = 4*WR` uint4 per
/// token) into the accumulators the pass just staged, then stores each to Y:
/// per token, the workgroup's `16*WR` rows are contiguous, so a store touches
/// 4 (W8) or 8 (W16) whole 128-byte lines instead of 16 partial ones. Same
/// values and same (token < N, row < M) element set as the unstaged path
/// (M % 8 == 0 makes the 4-row and 8-row run masks agree).
///
/// `y_stage` is 32 KiB, token-major (`16*R` bytes per token, no pad); uint4
/// `c` of token `t` sits at uint4 `c ^ (t & 15)` of its row, so the 16 tokens
/// of a tile land in 16 distinct bank groups on the write side and a wave's
/// read stays one row. Every stage address is precomputed once into its own
/// VGPR as `v_and_b32 vA, 0x7ff0, vX` (the static LDS bound's masked form; all
/// addresses are 16-aligned and below 32768, so the mask keeps them exact) and
/// no store source is redefined while stores are in flight: store `i` of a
/// pass uses its own offset VGPR (`8i` tokens on, shared by every pass) and
/// pass `p` its own soffset SGPR, all computed before the first store.
fn staged_stores(b: &mut Builder, spec: &Spec) -> R {
    const STAGE_BYTES: u32 = 32768;
    const MASK: u32 = STAGE_BYTES - 16;
    let wr = u32::from(spec.wr);
    let r = 4 * wr; // uint4 per token row
    let row = 16 * r; // bytes per token row
    let tok_pass = 512 / wr;
    debug_assert_eq!(tok_pass * row, STAGE_BYTES);
    let tiles_pass = (tok_pass / 16) as u8;
    let loads_pass = (64 / wr) as u8;
    let passes = spec.wr / 2;
    // Per-lane registers (lane constants, temps, codes and the B ring are dead
    // after the loop; the scalar path rebuilds its offsets in the B ring).
    let (vtok, t0, t1, t2, t4, t5) = (T3, T0, T1, T2, T4, T5);
    let voff = |i: u8| RINGB + i;
    let waddr = |j: u8, h: u8| CODE[0] + 2 * j + h;
    let raddr = |i: u8| CODE[2] + i;
    // Scalars of the dead group loop: one Y token-group offset per pass.
    const SOFF: [u8; 8] = [G, GOFF, NGOFF, SXOFF, C8, GPR, NCHM1, GPRM1];
    if usize::from(passes) > SOFF.len() {
        return Err("qwen4_mq6_x4: too many epilogue passes".into());
    }
    b.lds_relayout()?;
    let stage = b.lds_slot("y_stage", 0, STAGE_BYTES)?;
    let masked = |b: &mut Builder, dst: u8, src: u8| {
        op(
            b,
            format!("v_and_b32_e32 v{dst}, {}, v{src}", lit(MASK)),
            &[v(dst)],
            &[v(src)],
        )
    };
    // Write side: lane (ml, lh) of wave w holds token ml of a tile, uint4
    // c = 2*(tid >> 4) + h (rows 16w + 8lh + 4h); tile j of the pass at
    // (16j + ml) * row + ((c ^ ml) << 4).
    op(
        b,
        format!("v_and_b32_e32 v{t4}, 15, v{TID}"),
        &[v(t4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t5}, 4, v{TID}"),
        &[v(t5)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{t5}, 1, v{t5}"),
        &[v(t5)],
        &[v(t5)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{t0}, {}, v{t4}", row.trailing_zeros()),
        &[v(t0)],
        &[v(t4)],
    )?;
    for h in 0..2u8 {
        if h == 0 {
            op(
                b,
                format!("v_xor_b32_e32 v{t1}, v{t5}, v{t4}"),
                &[v(t1)],
                &[v(t5), v(t4)],
            )?;
        } else {
            op(
                b,
                format!("v_or_b32_e32 v{t1}, 1, v{t5}"),
                &[v(t1)],
                &[v(t5)],
            )?;
            op(
                b,
                format!("v_xor_b32_e32 v{t1}, v{t1}, v{t4}"),
                &[v(t1)],
                &[v(t1), v(t4)],
            )?;
        }
        op(
            b,
            format!("v_lshl_add_u32 v{t1}, v{t1}, 4, v{t0}"),
            &[v(t1)],
            &[v(t1), v(t0)],
        )?;
        for j in 0..tiles_pass {
            if j == 0 {
                masked(b, waddr(j, h), t1)?;
                continue;
            }
            op(
                b,
                format!(
                    "v_add_nc_u32_e32 v{t2}, {}, v{t1}",
                    lit(u32::from(j) * 16 * row)
                ),
                &[v(t2)],
                &[v(t1)],
            )?;
            masked(b, waddr(j, h), t2)?;
        }
    }
    // Read side: token t0 = tid / R of each 8-token group, uint4 c = tid % R;
    // load i at (8i + t0) * row + ((c ^ t0 ^ 8*(i & 1)) << 4).
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t5}, {}, v{TID}", r.trailing_zeros()),
        &[v(t5)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t4}, {}, v{TID}", r - 1),
        &[v(t4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_xor_b32_e32 v{t1}, v{t4}, v{t5}"),
        &[v(t1)],
        &[v(t4), v(t5)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{t0}, {}, v{t5}", row.trailing_zeros()),
        &[v(t0)],
        &[v(t5)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{t1}, v{t1}, 4, v{t0}"),
        &[v(t1)],
        &[v(t1), v(t0)],
    )?;
    op(
        b,
        format!("v_xor_b32_e32 v{t2}, {}, v{t1}", lit(128)),
        &[v(t2)],
        &[v(t1)],
    )?;
    for i in 0..loads_pass {
        let base = if i % 2 == 0 { t1 } else { t2 };
        if i == 0 {
            masked(b, raddr(i), base)?;
            continue;
        }
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{t0}, {}, v{base}",
                lit(u32::from(i) * 8 * row)
            ),
            &[v(t0)],
            &[v(base)],
        )?;
        masked(b, raddr(i), t0)?;
    }
    // Token of this lane's first store: wgy*256 + tid / R.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 8"), &[ST0], &[WGY])?;
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{WGX}, {}", lit(spec.rows())),
        &[ST1],
        &[WGX],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{vtok}, s{ST0}, v{t5}"),
        &[v(vtok)],
        &[s(ST0), v(t5)],
    )?;
    // Row = wgx*rows + 4*(tid % R); store i's Y byte offset within its pass
    // ((tid / R + 8i) * M + row) * 4; the pass's token offset rides in soffset.
    op(
        b,
        format!("v_lshl_add_u32 v{}, v{t4}, 2, s{ST1}", voff(0)),
        &[v(voff(0))],
        &[v(t4), s(ST1)],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{}", voff(0)),
        &[s(MASK1)],
        &[s(SM), v(voff(0))],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{t5}, v{t5}, s{SM}"),
        &[v(t5)],
        &[v(t5), s(SM)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{t5}", voff(0)),
        &[v(voff(0))],
        &[v(voff(0)), v(t5)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", voff(0)),
        &[v(voff(0))],
        &[v(voff(0))],
    )?;
    sop(b, format!("s_lshl_b32 s{ST1}, s{SM}, 5"), &[ST1], &[SM])?; // 8 tokens of Y
    for i in 1..loads_pass {
        op(
            b,
            format!("v_add_nc_u32_e32 v{}, s{ST1}, v{}", voff(i), voff(i - 1)),
            &[v(voff(i))],
            &[s(ST1), v(voff(i - 1))],
        )?;
    }
    // Pass p's token-group offset: (wgy*256 + P*p) * M * 4.
    sop(
        b,
        format!("s_mul_i32 s{}, s{ST0}, s{SM}", SOFF[0]),
        &[SOFF[0]],
        &[ST0, SM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{0}, s{0}, 2", SOFF[0]),
        &[SOFF[0]],
        &[SOFF[0]],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{SM}, {}", lit(4 * tok_pass)),
        &[ST1],
        &[SM],
    )?;
    for p in 1..usize::from(passes) {
        sop(
            b,
            format!("s_add_co_i32 s{}, s{}, s{ST1}", SOFF[p], SOFF[p - 1]),
            &[SOFF[p]],
            &[SOFF[p - 1], ST1],
        )?;
    }
    sop(b, format!("s_mov_b32 s{SNB}, s{SN}"), &[SNB], &[SN])?;
    for p in 0..passes {
        for j in 0..tiles_pass {
            let acc = ACC + 8 * (p * tiles_pass + j);
            for h in 0..2u8 {
                let (addr, data) = (waddr(j, h), acc + 4 * h);
                b.ds_store(
                    stage,
                    Instruction::new(
                        format!("ds_store_b128 v{addr}, {}", rs(data, 4)),
                        vec![],
                        vec![v(addr), vr(data, 4)],
                    )
                    .memory(MemoryClass::DsStore),
                )?;
            }
        }
        b.barrier(&[Transition::Ready(stage)])?;
        // Read back into this pass's (now staged) accumulators.
        let dst = |i: u8| ACC + 8 * p * tiles_pass + 4 * i;
        for i in 0..loads_pass {
            b.ds_load(
                stage,
                Instruction::new(
                    format!("ds_load_b128 {}, v{}", rs(dst(i), 4), raddr(i)),
                    vec![vr(dst(i), 4)],
                    vec![v(raddr(i))],
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for i in 0..loads_pass {
            op(
                b,
                format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{vtok}"),
                &[s(MASK0)],
                &[s(SNB), v(vtok)],
            )?;
            sop(
                b,
                format!("s_add_co_i32 s{SNB}, s{SNB}, -8"),
                &[SNB],
                &[SNB],
            )?;
            wait_alu(b, "depctr_va_sdst(0)")?;
            op(
                b,
                format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                &[],
                &[s(MASK0), s(MASK1)],
            )?;
            bstore(b, 4, dst(i), voff(i), SRD_Y, SOFF[usize::from(p)], 0)?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        }
        if p + 1 < passes {
            b.barrier(&[Transition::Retire(stage)])?;
        }
    }
    Ok(())
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(36)
        .pointer("A", 0)
        .pointer("X", 8)
        .pointer("Y", 16)
        .hidden("M", 24, 4, "by_value")
        .hidden("K", 28, 4, "by_value")
        .hidden("N", 32, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "qwen4_mq6_x4".into(),
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

/// Every entry of the arch as one code object (`Spec::module`).
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
