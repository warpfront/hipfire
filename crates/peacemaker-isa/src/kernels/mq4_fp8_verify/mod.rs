// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 MQ4G256V2 (symmetric) FP8 verify GEMMs for 1 <= N <= 128 activation
//! rows, emitted with the checked ISA builder: exact E4M3 dequantization of
//! `q - 8` in registers, E4M3 activations and `v_wmma_f32_16x16x16_fp8_fp8`.
//! Normative contract: `qcal/release-0.4.2/cb/pmfp8/KERNEL-SPEC.md`.
//!
//! # Exactness (row-count independence by construction)
//!
//! ISA text has no compiler, so the DAG is pinned by construction: every
//! multiply and add of the contract is one explicit instruction in a stated
//! order, and no instruction sequence of the arithmetic depends on `N`, the
//! tile (`BT`), the wave count (`W`), the wave's position in the grid or the
//! template. See the FP-contract hazard note in `pm_decode/mod.rs`: a
//! compiler is free to contract or reorder, this text is not. For the output
//! at (token `t`, weight row `r`), K in 256-K groups `g`, 128-K halves `h`:
//!
//!  1. Fragment `(g, h, tt)`, `tt = 0..7`: the 16 K values
//!     `256g + 128h + 64kg + 8tt + e` (`kg` = lane >> 4, `e = 0..7`). The
//!     lane's 8 codes are the one dword of the group at byte `8 + 64h + 32kg +
//!     4tt`; nibble `e` becomes the E4M3 byte of `(q - 8)` (`D0 CE CC CA C8 C4
//!     C0 B8 | 00 38 40 44 48 4A 4C 4E`, exact, no subnormals) — 17 VALU per
//!     fragment: `and, lshr, and` (nibble spread), two `v_perm_b32` (nibbles
//!     0..3 / 4..7 in byte order) and per 4 nibbles `perm(NEG) / xor / perm(POS)
//!     / add / perm(0,0,.) / bfi`. The activation fragment is the raw E4M3
//!     bytes `X[t*K + 256g + 128h + 64kg + 8tt .. +8)`.
//!  2. Per half the partial `P = WMMA(A_0, B_0, 0)`, `P = WMMA(A_tt, B_tt, P)`
//!     for `tt = 1..7`: operand A (SRC0) is the activation fragment (row = token),
//!     operand B (SRC1) the weight fragment (column = the lane's own row), so
//!     `P` VGPR `j` of lane `l` is the output (token `8*(l>>4) + j`, row
//!     `l & 15`) of the tile. The same instruction, operands and chain order in
//!     every entry; each `P` is an exact sum for exactly representable operands
//!     and is otherwise fixed by the hardware's single-instruction numerics.
//!  3. Fold: `T` starts at `+0` and takes, per half in ascending `(g, h)`,
//!     `T = v_fma_f32(sc, P, T)` with `sc = v_cvt_f32_f16(header.l)` (exact; one
//!     scale per lane = per weight row). The first fold of an entry is emitted as
//!     a fold of an all-zero `P` with a zero scale into `T = +0`
//!     (`fma(0, 0, +0) = +0`, bit-identical to starting at `+0`).
//!  4. Epilogue: `a = S[t]`; overwrite ops `Y = v_mul_f32(a, T)`; residual
//!     `Y = v_fma_f32(a, T, Y_old)`.
//!
//! Row-count independence: the sequence of (1)–(4) executed for one output
//! reads only the weight row, the token's activation row and scale; tokens
//! share nothing but the instruction that happens to execute them in other
//! lanes/accumulators. Padding (tokens >= N read 0 through a bounded buffer
//! descriptor; rows >= total_m clamp to the last row) is never stored.
//!
//! # Templates
//!  * `bt1w1` (`direct`): one wave, no LDS, no barriers; per half 4 b128
//!    activation loads, 4 b64 code loads; codes are prefetched two halves ahead
//!    (>= 2 KB of weights per wave in flight), dequantization of the next half
//!    is interleaved with the dependent WMMA chain, the fold of a half is
//!    interleaved with the next chain.
//!  * `bt{2,4,8}w{4,8}` (`staged`): X staged per 32-K slab in LDS (48 B pitch),
//!    two slots, one barrier per slab, codes three slabs ahead, next slab's
//!    dequant and the half-boundary folds interleaved with the WMMAs.
//!
//! Launch: block `[32*W, 1, 1]`, grid `[ceil(total_m / (16*W)), ceil(N / (16*BT)), 1]`,
//! no dynamic LDS, wave32. Domain: `K > 0`, `K % 256 == 0`, `N >= 1`,
//! `N*K < 2^31`, every `m_i >= 1`. Kernargs (HIP order): `A_0..A_{n-1}`, `X`, `S`,
//! `Y_0..Y_{n-1}`, `m_0..m_{n-1}`, `K`, `N` (`20n + 24` bytes).
//!
//! Per-fragment VALU (dequant): 17 (`and, lshr, and, perm, perm`, then twice
//! `perm, xor, perm, add, perm, bfi`). Per output element-group the fold is one
//! `v_fma_f32` per half. Register and LDS budgets per entry (asserted by
//! `register_budgets_are_the_documented_ones`, recorded in the contracts and
//! M7 receipts):
//!
//! | tile | VGPR high water | static LDS |
//! |---|---|---|
//! | `bt1w1` | 147 | 0 |
//! | `bt2w4` / `bt2w8` | 103 / 101 | 3072 |
//! | `bt4w4` / `bt4w8` | 147 / 143 | 6144 |
//! | `bt8w4` / `bt8w8` | 235 / 227 | 12288 |
//!
//! SGPR high water (every tile): `qkvza` 74, `qkv` 69, `gate_up` 64, `residual`
//! / `set` 59 (the kernarg segment image dominates; the contract ceiling is 80).
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, smem, sop, sr, v, vr, SRD_WORD3};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

mod direct;
mod staged;
#[cfg(test)]
mod tests;

pub(crate) type R = Result<(), String>;

/// The five wrapper families (ABI and epilogue).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Op {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
    Set,
}
impl Fp8Op {
    pub const ALL: [Fp8Op; 5] = [
        Fp8Op::Qkvza,
        Fp8Op::Qkv,
        Fp8Op::GateUp,
        Fp8Op::Residual,
        Fp8Op::Set,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Qkvza => "qkvza",
            Self::Qkv => "qkv",
            Self::GateUp => "gate_up",
            Self::Residual => "residual",
            Self::Set => "set",
        }
    }
    /// Projection matrices (weight and output pairs).
    pub fn matrices(self) -> u8 {
        match self {
            Self::Qkvza => 4,
            Self::Qkv => 3,
            Self::GateUp => 2,
            Self::Residual | Self::Set => 1,
        }
    }
    /// Residual (`Y += ...`) instead of overwrite (`Y = ...`).
    pub fn accumulates(self) -> bool {
        self == Self::Residual
    }
    /// Kernarg segment bytes: n weight pointers, X, S, n output pointers, n row counts, K, N.
    pub fn kernarg_bytes(self) -> u32 {
        20 * u32::from(self.matrices()) + 24
    }
    fn names(
        self,
    ) -> (
        &'static [&'static str],
        &'static [&'static str],
        &'static [&'static str],
    ) {
        match self {
            Self::Qkvza => (
                &["A_qkv", "A_z", "A_beta", "A_alpha"],
                &["Y_qkv", "Y_z", "Y_beta", "Y_alpha"],
                &["qkv_m", "z_m", "beta_m", "alpha_m"],
            ),
            Self::Qkv => (
                &["A_q", "A_k", "A_v"],
                &["Y_q", "Y_k", "Y_v"],
                &["q_m", "k_m", "v_m"],
            ),
            Self::GateUp => (
                &["A_gate", "A_up"],
                &["Y_gate", "Y_up"],
                &["gate_m", "up_m"],
            ),
            Self::Residual | Self::Set => (&["A"], &["Y"], &["M"]),
        }
    }
}
impl std::str::FromStr for Fp8Op {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|o| o.name() == s)
            .ok_or_else(|| format!("mq4_fp8_verify op {s} (qkvza|qkv|gate_up|residual|set)"))
    }
}

/// One entry: `bt` 16-token tiles per block, `waves` waves (16 weight rows each) per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub bt: u8,
    pub waves: u8,
    pub op: Fp8Op,
}
impl Spec {
    /// (BT, W) in the frozen order: the direct tile, then the staged tiles.
    pub const TILES: [(u8, u8); 7] = [(1, 1), (2, 4), (2, 8), (4, 4), (4, 8), (8, 4), (8, 8)];
    /// Every entry of the module in its frozen order (op, then tile).
    pub fn all(arch: Arch) -> Vec<Spec> {
        let mut out = Vec::with_capacity(35);
        for op in Fp8Op::ALL {
            for (bt, waves) in Self::TILES {
                out.push(Spec {
                    arch,
                    bt,
                    waves,
                    op,
                });
            }
        }
        out
    }
    pub fn symbol(&self) -> String {
        format!(
            "mq4_fp8_verify_{}_pm_gfx1201_bt{}w{}",
            self.op.name(),
            self.bt,
            self.waves
        )
    }
    /// Contract file tag: `<op>_bt<bt>w<w>`.
    pub fn tag(&self) -> String {
        format!("{}_bt{}w{}", self.op.name(), self.bt, self.waves)
    }
    pub fn module(arch: Arch) -> String {
        format!("mq4_fp8_verify_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("mq4_fp8_verify builder is exact-gfx1201 only".into());
        }
        if !Self::TILES.contains(&(self.bt, self.waves)) {
            return Err(format!(
                "mq4_fp8_verify: tile bt{}w{} not in {:?}",
                self.bt,
                self.waves,
                Self::TILES
            ));
        }
        Ok(())
    }
    /// The direct (no LDS, one wave) template.
    pub fn direct(&self) -> bool {
        self.bt == 1
    }
    fn threads(&self) -> u32 {
        32 * u32::from(self.waves)
    }
    /// Token rows per block.
    fn tokens(&self) -> u32 {
        16 * u32::from(self.bt)
    }
    /// Weight rows per block.
    fn rows(&self) -> u32 {
        16 * u32::from(self.waves)
    }
    /// One LDS X slot: `16*BT` token rows x 48 bytes (staged only).
    pub fn slot_bytes(&self) -> u32 {
        if self.direct() {
            0
        } else {
            self.tokens() * staged::PITCH
        }
    }
    /// Static LDS: two slots (none for the direct template).
    pub fn lds_bytes(&self) -> u32 {
        2 * self.slot_bytes()
    }
}

// ------------------------------------------------------------------ registers
// VGPRs: v0 tid; lane constants; 64-bit row pointers; then the template's own.
pub(crate) const TID: u8 = 0;
pub(crate) const LDRD: u8 = 1; // staged: LDS read address of the lane's A row
pub(crate) const LDST: u8 = 2; // staged: LDS store address of the thread's X piece
pub(crate) const XOFF: u8 = 3; // X buffer voffset (token row + K piece)
pub(crate) const VROW: u8 = 4; // output row rs + ml (unclamped)
pub(crate) const NL: u8 = 5; // N - bs - 8*kg: tokens 16b+j < NL are valid
pub(crate) const VSO: u8 = 6; // 4 * (bs + 8*kg): S buffer voffset
pub(crate) const T0: u8 = 7;
pub(crate) const T1: u8 = 8;
pub(crate) const T2: u8 = 9;
pub(crate) const T3: u8 = 10;
pub(crate) const T4: u8 = 11;
pub(crate) const T5: u8 = 12;
/// 64-bit per-lane pointers: row base (header), row base + 8 + 32*kg (codes),
/// this group's codes, the next group's codes and header (the direct template
/// also keeps the group-after-next's codes in `PHN` neighbours; see `direct`).
pub(crate) const BASEH: u8 = 13;
pub(crate) const BASEC: u8 = 15;
pub(crate) const PC: u8 = 17;
pub(crate) const PCN: u8 = 19;
pub(crate) const PHN: u8 = 21;
/// First VGPR of the template-specific layout.
pub(crate) const V0: u8 = 23;
/// Epilogue aliases of dead loop registers.
pub(crate) const PTR: u8 = PC; // output pointer pair
pub(crate) const LPTR: u8 = PCN; // residual load pointer pair
pub(crate) const M4: u8 = PHN; // 4*m of the lane's projection
pub(crate) const M9: u8 = PHN + 1; // 9*4*m

// SGPRs.
pub(crate) const KARG: u8 = 0;
pub(crate) const SRD_X: u8 = 4;
pub(crate) const SRD_S: u8 = 8;
pub(crate) const WGX: u8 = 12;
pub(crate) const WGY: u8 = 13;
pub(crate) const TOTM: u8 = 14;
pub(crate) const TOTM1: u8 = 15;
pub(crate) const GPR: u8 = 16;
pub(crate) const STRIDE: u8 = 17;
pub(crate) const GPRM1: u8 = 18;
pub(crate) const BS: u8 = 19;
pub(crate) const NMB: u8 = 20;
pub(crate) const ST0: u8 = 21;
pub(crate) const G: u8 = 22;
pub(crate) const GX0: u8 = 23;
pub(crate) const GX1: u8 = 24;
pub(crate) const NGO: u8 = 25;
pub(crate) const GO: u8 = 26;
pub(crate) const NIDX: u8 = 27;
/// Cumulative projection row bounds m0, m0+m1, m0+m1+m2.
pub(crate) const E: [u8; 3] = [28, 29, 30];
pub(crate) const NGO2: u8 = 31;
/// Dequant tables: NEG_LO, NEG_HI, POS_LO, POS_HI.
pub(crate) const TNL: u8 = 32;
pub(crate) const TNH: u8 = 33;
pub(crate) const TPL: u8 = 34;
pub(crate) const TPH: u8 = 35;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
pub(crate) const MASK0: u8 = 36;
pub(crate) const MASK1: u8 = 38;
pub(crate) const MASKR: u8 = 40;
/// Kernarg segment image (up to 26 dwords; 8-aligned for `s_load_b256`).
pub(crate) const KA: u8 = 48;

pub(crate) const GROUP: u32 = 136; // bytes per 256-K weight group: h0, h1, 128 code bytes
const NEG_LO: u32 = 0xcacc_ced0;
const NEG_HI: u32 = 0xb8c0_c4c8;
const POS_LO: u32 = 0x4440_3800;
const POS_HI: u32 = 0x4e4c_4a48;

pub(crate) fn n_of(spec: &Spec) -> u8 {
    spec.op.matrices()
}
/// Kernarg SGPRs: weight pointer i, X, S, output pointer i, row count i, K, N.
pub(crate) fn ka_a(i: u8) -> u8 {
    KA + 2 * i
}
pub(crate) fn ka_x(spec: &Spec) -> u8 {
    KA + 2 * n_of(spec)
}
pub(crate) fn ka_s(spec: &Spec) -> u8 {
    KA + 2 * n_of(spec) + 2
}
pub(crate) fn ka_y(spec: &Spec, i: u8) -> u8 {
    KA + 2 * (n_of(spec) + 2 + i)
}
pub(crate) fn ka_m(spec: &Spec, i: u8) -> u8 {
    KA + 4 * n_of(spec) + 4 + i
}
pub(crate) fn ka_k(spec: &Spec) -> u8 {
    KA + 5 * n_of(spec) + 4
}
pub(crate) fn ka_n(spec: &Spec) -> u8 {
    KA + 5 * n_of(spec) + 5
}
pub(crate) fn sgprs(spec: &Spec) -> u16 {
    u16::from(KA) + u16::from(5 * n_of(spec)) + 6
}
/// Exact SMEM loads covering the kernarg segment: (byte offset, dwords).
fn kernarg_loads(spec: &Spec) -> Vec<(u32, u8)> {
    let mut out = Vec::new();
    let (mut off, size) = (0u32, spec.op.kernarg_bytes());
    for d in [8u8, 4, 2, 1] {
        while size - off >= 4 * u32::from(d) {
            out.push((off, d));
            off += 4 * u32::from(d);
        }
    }
    out
}

/// Declare the registers every template shares (VGPR budget `vgprs`).
pub(crate) fn plan_common(spec: &Spec, vgprs: u16) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(vgprs, sgprs(spec))?;
    let w = || Live::Whole;
    p.v::<1>("tid", TID, w())?;
    for n in LDRD..=T5 {
        p.v::<1>("lane_constant", n, w())?;
    }
    for ptr in [BASEH, BASEC, PC, PCN, PHN] {
        p.v::<2>("row_pointer", ptr, w())?;
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<4>("srd_x", SRD_X, w())?;
    p.s::<4>("srd_s", SRD_S, w())?;
    for n in WGX..=TPH {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1, MASKR] {
        p.s::<2>("lane_mask", m, w())?;
    }
    for (off, d) in kernarg_loads(spec) {
        let base = KA + (off / 4) as u8;
        match d {
            8 => {
                p.s::<8>("kernargs", base, w())?;
            }
            4 => {
                p.s::<4>("kernargs", base, w())?;
            }
            2 => {
                p.s::<2>("kernargs", base, w())?;
            }
            _ => {
                p.s::<1>("kernargs", base, w())?;
            }
        }
    }
    Ok(p)
}

pub(crate) fn wait_alu(b: &mut Builder, what: &str) -> R {
    op(b, format!("s_wait_alu {what}"), &[], &[])
}
pub(crate) fn rs(base: u8, len: u8) -> String {
    if len == 1 {
        format!("v{base}")
    } else {
        format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1)
    }
}
pub(crate) fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
pub(crate) fn lbl(spec: &Spec, name: &str) -> String {
    format!(".Lmq4f8_{}_{name}", spec.tag())
}

/// `v[dst:dst+1] = src0 * src1 + v[base:base+1]` (64-bit, no carry out).
pub(crate) fn mad64(b: &mut Builder, dst: u8, src0: RegRef, src1: &str, base: u8) -> R {
    op(
        b,
        format!(
            "v_mad_co_u64_u32 {}, null, {src0}, {src1}, {}",
            rs(dst, 2),
            rs(base, 2)
        ),
        &[vr(dst, 2)],
        &[src0, vr(base, 2)],
    )
}
/// `global_load_b{32,64,128} dst, v[ptr:ptr+1], off offset:imm`.
pub(crate) fn gload(b: &mut Builder, dwords: u8, dst: u8, ptr: u8, imm: u32) -> R {
    let w = match dwords {
        1 => 32,
        2 => 64,
        4 => 128,
        _ => return Err("gload width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "global_load_b{w} {}, {}, off{}",
                rs(dst, dwords),
                rs(ptr, 2),
                off(imm)
            ),
            vec![vr(dst, dwords)],
            vec![vr(ptr, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
/// `global_store_b32 v[ptr:ptr+1], data, off`.
pub(crate) fn gstore(b: &mut Builder, ptr: u8, data: u8) -> R {
    b.push(
        Instruction::new(
            format!("global_store_b32 {}, v{data}, off", rs(ptr, 2)),
            vec![],
            vec![vr(ptr, 2), v(data)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
/// `buffer_load_b{32,64,128} dst, v{voff}, s[srd:srd+3], s{soff}|null offen offset:imm`.
pub(crate) fn bload(
    b: &mut Builder,
    dwords: u8,
    dst: u8,
    voff: u8,
    srd: u8,
    soff: Option<u8>,
    imm: u32,
) -> R {
    let w = match dwords {
        1 => 32,
        2 => 64,
        4 => 128,
        _ => return Err("bload width".into()),
    };
    let so = soff.map_or_else(|| "null".to_string(), |n| format!("s{n}"));
    let mut uses = vec![v(voff), sr(srd, 4)];
    if let Some(n) = soff {
        uses.push(s(n))
    }
    b.push(
        Instruction::new(
            format!(
                "buffer_load_b{w} {}, v{voff}, s[{srd}:{}], {so} offen{}",
                rs(dst, dwords),
                srd + 3,
                off(imm)
            ),
            vec![vr(dst, dwords)],
            uses,
        )
        .memory(MemoryClass::VmemLoad),
    )
}

/// One scheduled operation interleaved between WMMAs.
#[derive(Clone)]
pub(crate) enum Fill {
    Valu {
        text: String,
        defs: Vec<RegRef>,
        uses: Vec<RegRef>,
    },
    /// `dwords` of codes / headers at `v[ptr] + imm`.
    GLoad {
        dwords: u8,
        dst: u8,
        ptr: u8,
        imm: u32,
    },
    /// `dwords` of X at `v{XOFF} + s{soff} + imm`.
    XLoad {
        dwords: u8,
        dst: u8,
        soff: u8,
        imm: u32,
    },
    /// Staged: `ds_load_b128 dst, v{LDRD} offset:imm` from LDS slot `slot`.
    DsLoad { slot: usize, dst: u8, imm: u32 },
    /// Staged: `ds_store_b{32,64,128} v{LDST}, src offset:imm` into LDS slot `slot`.
    DsStore {
        slot: usize,
        dwords: u8,
        src: u8,
        imm: u32,
    },
}
pub(crate) fn valu(text: String, defs: Vec<RegRef>, uses: Vec<RegRef>) -> Fill {
    Fill::Valu { text, defs, uses }
}
pub(crate) fn emit_fill(b: &mut Builder, f: &Fill) -> R {
    match f {
        Fill::Valu { text, defs, uses } => op(b, text.clone(), defs, uses),
        Fill::GLoad {
            dwords,
            dst,
            ptr,
            imm,
        } => gload(b, *dwords, *dst, *ptr, *imm),
        Fill::XLoad {
            dwords,
            dst,
            soff,
            imm,
        } => bload(b, *dwords, *dst, XOFF, SRD_X, Some(*soff), *imm),
        Fill::DsLoad { slot, dst, imm } => b.ds_load(
            *slot,
            Instruction::new(
                format!("ds_load_b128 {}, v{LDRD}{}", rs(*dst, 4), off(*imm)),
                vec![vr(*dst, 4)],
                vec![v(LDRD)],
            )
            .memory(MemoryClass::DsLoad),
        ),
        Fill::DsStore {
            slot,
            dwords,
            src,
            imm,
        } => {
            let w = match dwords {
                1 => 32,
                2 => 64,
                4 => 128,
                _ => return Err("ds_store width".into()),
            };
            b.ds_store(
                *slot,
                Instruction::new(
                    format!("ds_store_b{w} v{LDST}, {}{}", rs(*src, *dwords), off(*imm)),
                    vec![],
                    vec![v(LDST), vr(*src, *dwords)],
                )
                .memory(MemoryClass::DsStore),
            )
        }
    }
}

/// `v_wmma_f32_16x16x16_fp8_fp8 P, A, B, P|0`: A (SRC0) activation fragment,
/// B (SRC1) weight fragment. `chain` false starts the half's chain from inline zero.
pub(crate) fn wmma(b: &mut Builder, p: u8, a: u8, bb: u8, chain: bool) -> R {
    let c = if chain { rs(p, 8) } else { "0".to_string() };
    let mut uses = vec![vr(a, 2), vr(bb, 2)];
    if chain {
        uses.push(vr(p, 8))
    }
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_fp8_fp8 {}, {}, {}, {c}",
            rs(p, 8),
            rs(a, 2),
            rs(bb, 2)
        ),
        &[vr(p, 8)],
        &uses,
    )
}

/// Dequantize the codes dword `w` into the E4M3 fragment `out` (pair): 17 VALU
/// over eight temporaries `t..t+8`. The two independent halves of the dword
/// (nibbles 0..3, 4..7) are interleaved instruction by instruction.
pub(crate) fn dequant(w: u8, out: u8, t: u8) -> Vec<Fill> {
    let (lo, hi, qa, qb, r1a, sa, r1b, sb) = (t, t + 1, t + 2, t + 3, t + 4, t + 5, t + 6, t + 7);
    let mut o = Vec::with_capacity(17);
    let (nibbles, sel0, sel1, x08, k05) = (
        lit(0x0f0f_0f0f),
        lit(0x0501_0400),
        lit(0x0703_0602),
        lit(0x0808_0808),
        lit(0x0505_0505),
    );
    o.push(valu(
        format!("v_and_b32_e32 v{lo}, {nibbles}, v{w}"),
        vec![v(lo)],
        vec![v(w)],
    ));
    o.push(valu(
        format!("v_lshrrev_b32_e32 v{hi}, 4, v{w}"),
        vec![v(hi)],
        vec![v(w)],
    ));
    o.push(valu(
        format!("v_and_b32_e32 v{hi}, {nibbles}, v{hi}"),
        vec![v(hi)],
        vec![v(hi)],
    ));
    o.push(valu(
        format!("v_perm_b32 v{qa}, v{hi}, v{lo}, {sel0}"),
        vec![v(qa)],
        vec![v(hi), v(lo)],
    ));
    o.push(valu(
        format!("v_perm_b32 v{qb}, v{hi}, v{lo}, {sel1}"),
        vec![v(qb)],
        vec![v(hi), v(lo)],
    ));
    // R1 = perm(NEG_HI, NEG_LO, q): valid for q < 8.
    for (q, r1) in [(qa, r1a), (qb, r1b)] {
        o.push(valu(
            format!("v_perm_b32 v{r1}, s{TNH}, s{TNL}, v{q}"),
            vec![v(r1)],
            vec![s(TNH), s(TNL), v(q)],
        ));
    }
    // R2 = perm(POS_HI, POS_LO, q ^ 8): valid for q >= 8 (in lo/hi, dead after the perms).
    for (q, r2) in [(qa, lo), (qb, hi)] {
        o.push(valu(
            format!("v_xor_b32_e32 v{r2}, {x08}, v{q}"),
            vec![v(r2)],
            vec![v(q)],
        ));
    }
    for r2 in [lo, hi] {
        o.push(valu(
            format!("v_perm_b32 v{r2}, s{TPH}, s{TPL}, v{r2}"),
            vec![v(r2)],
            vec![s(TPH), s(TPL), v(r2)],
        ));
    }
    // m = 0xFF per byte iff q >= 8: perm(0, 0, q + 5) (selectors >= 13 give 0xFF).
    for (q, m) in [(qa, sa), (qb, sb)] {
        o.push(valu(
            format!("v_add_nc_u32_e32 v{m}, {k05}, v{q}"),
            vec![v(m)],
            vec![v(q)],
        ));
    }
    for m in [sa, sb] {
        o.push(valu(
            format!("v_perm_b32 v{m}, 0, 0, v{m}"),
            vec![v(m)],
            vec![v(m)],
        ));
    }
    for (k, (m, r2, r1)) in [(sa, lo, r1a), (sb, hi, r1b)].into_iter().enumerate() {
        let d = out + k as u8;
        o.push(valu(
            format!("v_bfi_b32 v{d}, v{m}, v{r2}, v{r1}"),
            vec![v(d)],
            vec![v(m), v(r2), v(r1)],
        ));
    }
    debug_assert_eq!(o.len(), 17);
    o
}
/// Fold of one tile: `T[j] = v_fma_f32(sc, P[j], T[j])`, j = 0..7.
pub(crate) fn fold(t: u8, p: u8, sc: u8) -> Vec<Fill> {
    (0..8u8)
        .map(|j| {
            valu(
                format!("v_fma_f32 v{}, v{sc}, v{}, v{}", t + j, p + j, t + j),
                vec![v(t + j)],
                vec![v(sc), v(p + j), v(t + j)],
            )
        })
        .collect()
}
/// `sc = v_cvt_f32_f16(header dword d .l)`.
pub(crate) fn cvt_scale(sc: u8, hdr: u8, d: u8) -> Fill {
    valu(
        format!("v_cvt_f32_f16_e64 v{sc}, v{}.l", hdr + d),
        vec![v(sc)],
        vec![v(hdr + d)],
    )
}
/// Spread `fills` evenly over the slots `from..to` (at least one).
pub(crate) fn spread(slots: &mut [Vec<Fill>], fills: Vec<Fill>, from: usize, to: usize) {
    let n = fills.len().max(1);
    let span = to.max(from + 1) - from;
    for (k, f) in fills.into_iter().enumerate() {
        slots[from + k * span / n].push(f);
    }
}

/// Per-lane projection select for row `row` (VGPR): `dst` (pair) = pointer
/// `ptrs[i]`, `offv` = first row of projection i, `m` (optional) = its row
/// count, for the last i with `E[i-1] <= row`.
pub(crate) fn select(
    b: &mut Builder,
    spec: &Spec,
    row: u8,
    ptrs: &[u8],
    dst: u8,
    offv: u8,
    m: Option<u8>,
) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{dst}, s{}", ptrs[0]),
        &[v(dst)],
        &[s(ptrs[0])],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{}", dst + 1, ptrs[0] + 1),
        &[v(dst + 1)],
        &[s(ptrs[0] + 1)],
    )?;
    op(b, format!("v_mov_b32_e32 v{offv}, 0"), &[v(offv)], &[])?;
    if let Some(m) = m {
        op(
            b,
            format!("v_mov_b32_e32 v{m}, s{}", ka_m(spec, 0)),
            &[v(m)],
            &[s(ka_m(spec, 0))],
        )?;
    }
    for k in 1..n_of(spec) {
        let e = E[usize::from(k - 1)];
        op(
            b,
            format!("v_cmp_le_i32_e64 s{MASK1}, s{e}, v{row}"),
            &[s(MASK1)],
            &[s(e), v(row)],
        )?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        let p = ptrs[usize::from(k)];
        for (d, src) in [(dst, p), (dst + 1, p + 1), (offv, e)] {
            op(
                b,
                format!("v_cndmask_b32_e64 v{d}, v{d}, s{src}, s{MASK1}"),
                &[v(d)],
                &[v(d), s(src), s(MASK1)],
            )?;
        }
        if let Some(m) = m {
            let mk = ka_m(spec, k);
            op(
                b,
                format!("v_cndmask_b32_e64 v{m}, v{m}, s{mk}, s{MASK1}"),
                &[v(m)],
                &[v(m), s(mk), s(MASK1)],
            )?;
        }
    }
    Ok(())
}

/// Everything before the template-specific setup: kernargs, total rows, the
/// block-uniform early exits, the X and S descriptors, dequant tables, the
/// lane's row, projection and weight pointers (`BASEH`, `BASEC`), `VROW`, `NL`, `VSO`.
pub(crate) fn common_prologue(b: &mut Builder, spec: &Spec) -> R {
    let n = n_of(spec);
    for (o, d) in kernarg_loads(spec) {
        smem(b, KA + (o / 4) as u8, d, KARG, o)?;
    }
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
    )?; // kg
    b.wait(Counter::Km, 0)?;
    // total_m and the cumulative projection bounds.
    sop(
        b,
        format!("s_mov_b32 s{TOTM}, s{}", ka_m(spec, 0)),
        &[TOTM],
        &[ka_m(spec, 0)],
    )?;
    for k in 1..n {
        sop(
            b,
            format!("s_mov_b32 s{}, s{TOTM}", E[usize::from(k - 1)]),
            &[E[usize::from(k - 1)]],
            &[TOTM],
        )?;
        sop(
            b,
            format!("s_add_co_i32 s{TOTM}, s{TOTM}, s{}", ka_m(spec, k)),
            &[TOTM],
            &[TOTM, ka_m(spec, k)],
        )?;
    }
    // Block-uniform early exit: no weight rows, or no tokens, for this block.
    op(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[s(WGX)], &[])?;
    op(
        b,
        format!("s_and_b32 s{WGY}, ttmp7, 0xffff"),
        &[s(WGY)],
        &[],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{ST0}, s{TOTM}"),
        &[],
        &[ST0, TOTM],
    )?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{BS}, s{WGY}, {}", lit(spec.tokens())),
        &[BS],
        &[WGY],
    )?;
    let (sk, sn) = (ka_k(spec), ka_n(spec));
    sop(b, format!("s_cmp_ge_i32 s{BS}, s{sn}"), &[], &[BS, sn])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(b, format!("s_ashr_i32 s{GPR}, s{sk}, 8"), &[GPR], &[sk])?;
    sop(
        b,
        format!("s_mul_i32 s{STRIDE}, s{GPR}, {}", lit(GROUP)),
        &[STRIDE],
        &[GPR],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{GPRM1}, s{GPR}, -1"),
        &[GPRM1],
        &[GPR],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{TOTM1}, s{TOTM}, -1"),
        &[TOTM1],
        &[TOTM],
    )?;
    sop(
        b,
        format!("s_sub_co_i32 s{NMB}, s{sn}, s{BS}"),
        &[NMB],
        &[sn, BS],
    )?;
    // X descriptor bounded at N*K bytes and S at 4*N: tokens past N read 0.
    let (x, sv) = (ka_x(spec), ka_s(spec));
    sop(b, format!("s_mov_b32 s{SRD_X}, s{x}"), &[SRD_X], &[x])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD_X + 1, x + 1),
        &[SRD_X + 1],
        &[x + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{sn}, s{sk}", SRD_X + 2),
        &[SRD_X + 2],
        &[sn, sk],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_X + 3, lit(SRD_WORD3)),
        &[SRD_X + 3],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s{SRD_S}, s{sv}"), &[SRD_S], &[sv])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD_S + 1, sv + 1),
        &[SRD_S + 1],
        &[sv + 1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{sn}, 2", SRD_S + 2),
        &[SRD_S + 2],
        &[sn],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    for (r, c) in [(TNL, NEG_LO), (TNH, NEG_HI), (TPL, POS_LO), (TPH, POS_HI)] {
        sop(b, format!("s_mov_b32 s{r}, {}", lit(c)), &[r], &[])?;
    }
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    // Rows: rs = (wgx*W + wave)*16; the lane's output row rs + ml (VROW) and
    // its weight row min(VROW, total_m - 1).
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T5}, 5, v{TID}"),
        &[v(T5)],
        &[v(TID)],
    )?; // wave
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T5}, 4, v{T5}"),
        &[v(T5)],
        &[v(T5)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T5}, s{ST0}, v{T5}"),
        &[v(T5)],
        &[s(ST0), v(T5)],
    )?; // rs
    op(
        b,
        format!("v_add_nc_u32_e32 v{VROW}, v{T5}, v{T0}"),
        &[v(VROW)],
        &[v(T5), v(T0)],
    )?;
    op(
        b,
        format!("v_min_i32_e32 v{T5}, s{TOTM1}, v{VROW}"),
        &[v(T5)],
        &[s(TOTM1), v(VROW)],
    )?;
    // Its projection: weight pointer (into PC, free until the loop) and first row.
    let a_ptrs: Vec<u8> = (0..n).map(ka_a).collect();
    select(b, spec, T5, &a_ptrs, PC, T2, None)?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{T5}, v{T5}, v{T2}"),
        &[v(T5)],
        &[v(T5), v(T2)],
    )?;
    mad64(b, BASEH, v(T5), &format!("s{STRIDE}"), PC)?;
    op(
        b,
        format!("v_lshl_add_u32 v{T2}, v{T1}, 5, 8"),
        &[v(T2)],
        &[v(T1)],
    )?; // 8 + 32*kg
    mad64(b, BASEC, v(T2), "1", BASEH)?;
    // Epilogue token constants: NL = N - bs - 8*kg, VSO = 4*(bs + 8*kg).
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T3}, 3, v{T1}"),
        &[v(T3)],
        &[v(T1)],
    )?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{NL}, s{NMB}, v{T3}"),
        &[v(NL)],
        &[s(NMB), v(T3)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T3}, s{BS}, v{T3}"),
        &[v(T3)],
        &[s(BS), v(T3)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VSO}, 2, v{T3}"),
        &[v(VSO)],
        &[v(T3)],
    )
}

/// The per-group scalars of group `G`: `GX0` / `GX1` = K byte offset of this /
/// the next (clamped) group, `GO` / `NGO` = this / the next (clamped) group's
/// code-pointer offset, and, with `second`, `NGO2` for the group after next
/// (clamped). Callers turn them into `PC` / `PCN` / `PHN` pointers.
pub(crate) fn group_scalars(b: &mut Builder, second: bool) -> R {
    sop(b, format!("s_add_co_i32 s{NIDX}, s{G}, 1"), &[NIDX], &[G])?;
    sop(
        b,
        format!("s_min_u32 s{NIDX}, s{NIDX}, s{GPRM1}"),
        &[NIDX],
        &[NIDX, GPRM1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{NGO}, s{NIDX}, {}", lit(GROUP)),
        &[NGO],
        &[NIDX],
    )?;
    sop(b, format!("s_lshl_b32 s{GX1}, s{NIDX}, 8"), &[GX1], &[NIDX])?;
    sop(b, format!("s_lshl_b32 s{GX0}, s{G}, 8"), &[GX0], &[G])?;
    sop(
        b,
        format!("s_mul_i32 s{GO}, s{G}, {}", lit(GROUP)),
        &[GO],
        &[G],
    )?;
    if second {
        sop(b, format!("s_add_co_i32 s{NIDX}, s{G}, 2"), &[NIDX], &[G])?;
        sop(
            b,
            format!("s_min_u32 s{NIDX}, s{NIDX}, s{GPRM1}"),
            &[NIDX],
            &[NIDX, GPRM1],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{NGO2}, s{NIDX}, {}", lit(GROUP)),
            &[NGO2],
            &[NIDX],
        )?;
    }
    wait_alu(b, "depctr_sa_sdst(0)")
}

/// Registers of the shared epilogue.
pub(crate) struct Epi {
    /// First T accumulator (8 per tile).
    pub t: u8,
    /// First P (the last half's partial) and its scale register.
    pub p: u8,
    pub sc: u8,
    /// `8*BT` dead registers receiving the S values (the P registers themselves
    /// once folded; the pool entries of tile `b`, row `j` at index `8b + j`).
    pub s_pool: Vec<u8>,
    /// `8*BT` dead registers receiving Y_old (residual only).
    pub y_pool: Vec<u8>,
}

/// `exec = (16*b + j < NL) & MASKR` for the lane's token `bs + 16b + 8kg + j`.
fn token_exec(b: &mut Builder, k: u32) -> R {
    op(
        b,
        format!("v_cmp_lt_i32_e64 s{MASK0}, {}, v{NL}", lit(k)),
        &[s(MASK0)],
        &[v(NL)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("s_and_b32 exec_lo, s{MASK0}, s{MASKR}"),
        &[],
        &[s(MASK0), s(MASKR)],
    )
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
/// `ptr += stride` (bytes) after the store/load of (tile b, row j) `idx`: one
/// token ahead, or nine (8 -> next tile's 0) at a tile boundary.
fn bump(b: &mut Builder, ptr: u8, j: u32) -> R {
    mad64(b, ptr, v(if j == 7 { M9 } else { M4 }), "1", ptr)
}

/// The shared epilogue after the K loop: `P` holds the last half's partial
/// and `e.sc` its scale; every tile is folded, scaled and stored.
pub(crate) fn epilogue(b: &mut Builder, spec: &Spec, e: &Epi) -> R {
    let bt = u32::from(spec.bt);
    let acc = spec.op.accumulates();
    b.wait_all()?;
    // Output pointer of (token bs + 8kg, row VROW) and the byte strides.
    let y_ptrs: Vec<u8> = (0..n_of(spec)).map(|i| ka_y(spec, i)).collect();
    select(b, spec, VROW, &y_ptrs, PTR, T3, Some(T4))?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T0}, 2, v{VSO}"),
        &[v(T0)],
        &[v(VSO)],
    )?; // bs + 8*kg
    op(
        b,
        format!("v_mul_lo_u32 v{T2}, v{T0}, v{T4}"),
        &[v(T2)],
        &[v(T0), v(T4)],
    )?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{T5}, v{VROW}, v{T3}"),
        &[v(T5)],
        &[v(VROW), v(T3)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T2}, v{T2}, v{T5}"),
        &[v(T2)],
        &[v(T2), v(T5)],
    )?;
    mad64(b, PTR, v(T2), "4", PTR)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{M4}, 2, v{T4}"),
        &[v(M4)],
        &[v(T4)],
    )?;
    op(
        b,
        format!("v_mul_u32_u24_e32 v{M9}, 9, v{M4}"),
        &[v(M9)],
        &[v(M4)],
    )?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASKR}, s{TOTM}, v{VROW}"),
        &[s(MASKR)],
        &[s(TOTM), v(VROW)],
    )?;
    if acc {
        // Y_old in flight under the final folds.
        for k in 0..2 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, v{}", LPTR + k, PTR + k),
                &[v(LPTR + k)],
                &[v(PTR + k)],
            )?;
        }
        for bi in 0..bt {
            for j in 0..8u32 {
                token_exec(b, 16 * bi + j)?;
                gload(b, 1, e.y_pool[(8 * bi + j) as usize], LPTR, 0)?;
                exec_all(b)?;
                bump(b, LPTR, j)?;
            }
        }
    }
    // Last half's fold, then S = the tokens' scales into the dead P registers.
    for bi in 0..bt as u8 {
        for f in fold(e.t + 8 * bi, e.p + 8 * bi, e.sc) {
            emit_fill(b, &f)?;
        }
    }
    for bi in 0..bt {
        for j in 0..8u32 {
            bload(
                b,
                1,
                e.s_pool[(8 * bi + j) as usize],
                VSO,
                SRD_S,
                None,
                4 * (16 * bi + j),
            )?;
        }
    }
    for bi in 0..bt {
        for j in 0..8u32 {
            let k = (8 * bi + j) as usize;
            let (t, a) = (e.t + (8 * bi + j) as u8, e.s_pool[k]);
            if acc {
                op(
                    b,
                    format!("v_fma_f32 v{t}, v{a}, v{t}, v{}", e.y_pool[k]),
                    &[v(t)],
                    &[v(a), v(t), v(e.y_pool[k])],
                )?;
            } else {
                op(
                    b,
                    format!("v_mul_f32_e32 v{t}, v{a}, v{t}"),
                    &[v(t)],
                    &[v(a), v(t)],
                )?;
            }
            token_exec(b, 16 * bi + j)?;
            gstore(b, PTR, t)?;
            exec_all(b)?;
            b.release_store_sources()?;
            bump(b, PTR, j)?;
        }
    }
    b.release_store_sources()?;
    b.label(&lbl(spec, "end"))?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

fn kernargs(spec: &Spec) -> KernargLayout {
    let (a, y, m) = spec.op.names();
    let n = u32::from(spec.op.matrices());
    let mut k = KernargLayout::new(spec.op.kernarg_bytes());
    for (i, name) in a.iter().enumerate() {
        k = k.pointer(name, 8 * i as u32);
    }
    k = k.pointer("X", 8 * n).pointer("S", 8 * n + 8);
    for (i, name) in y.iter().enumerate() {
        k = k.pointer(name, 8 * (n + 2 + i as u32));
    }
    for (i, name) in m.iter().enumerate() {
        k = k.hidden(name, 16 * n + 16 + 4 * i as u32, 4, "by_value");
    }
    k.hidden("K", 20 * n + 16, 4, "by_value")
        .hidden("N", 20 * n + 20, 4, "by_value")
}

pub(crate) fn kernel_spec(spec: &Spec) -> KernelSpec {
    KernelSpec {
        kernel_id: "mq4_fp8_verify".into(),
        variant: spec.tag(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs: kernargs(spec),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: spec.lds_bytes(),
        wave32: true,
        cu_mode: false,
    }
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.direct() {
        direct::emit(spec)
    } else {
        staged::emit(spec)
    }
}

/// Every entry as one code object (`Spec::module`), in `Spec::all` order.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Spec::all(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Mnemonics a contract pins by exact count (when present).
const COUNTED: [&str; 27] = [
    "v_wmma_f32_16x16x16_fp8_fp8",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_perm_b32",
    "v_bfi_b32",
    "v_xor_b32_e32",
    "v_fma_f32",
    "v_mul_f32_e32",
    "v_cvt_f32_f16_e64",
    "v_mad_co_u64_u32",
    "buffer_load_b32",
    "buffer_load_b64",
    "buffer_load_b128",
    "global_load_b32",
    "global_load_b64",
    "global_load_b128",
    "global_store_b32",
    "ds_load_b128",
    "ds_store_b32",
    "ds_store_b64",
    "ds_store_b128",
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "s_endpgm",
    "v_nop",
];

/// The shape contract of one entry (`toolchain::IsaShapeContract`): exact
/// counts of its arithmetic, memory and barrier instructions, the forbidden
/// families (every other WMMA/convert/FMA-f16 route, flat/scratch, the DS
/// widths a template does not use), register ceilings, wave32, no spills or
/// private memory, the static LDS and no dynamic LDS.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    let mut forbidden: Vec<&str> = vec![
        "scratch_*",
        "flat_*",
        "ds_load_2addr*",
        "ds_store_2addr*",
        "v_readlane*",
        "v_writelane*",
        "s_setpc*",
        "s_swappc*",
        "s_waitcnt*",
        "v_wmma_f32_16x16x16_f16*",
        "v_wmma_f32_16x16x16_bf16*",
        "v_wmma_i32_*",
        "v_swmmac_*",
        "v_wmma_f32_16x16x16_bf8*",
        "v_cvt_f16_f32*",
        "v_fma_f16*",
        "v_pk_*",
        "v_dot*",
        "v_cvt_pk_f32_fp8*",
        "v_cvt_f32_ubyte*",
        "buffer_store*",
    ];
    if spec.direct() {
        forbidden.extend(["ds_*", "s_barrier*"]);
    } else {
        forbidden.extend([
            "ds_load_b32",
            "ds_load_b64",
            "ds_load_b96",
            "ds_load_u*",
            "ds_load_i*",
        ]);
    }
    serde_json::json!({
        "symbol": spec.symbol(), "counts": counts, "forbidden": forbidden,
        "vgpr_max": if spec.direct() { 192 } else { 256 }, "sgpr_max": 80,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": spec.lds_bytes(),
    })
}
