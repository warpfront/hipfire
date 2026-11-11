// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 MQ6 trunk GEMM, emitted with the checked ISA builder as an exact
//! twin of hipcc's `gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p{1,2}` family
//! (`kernels/src/qwen4_gemm_mqv2_wmma_gfx11_bt.hip`, `resid_bt_xlds<6, 8, 8,
//! true, ...>`): plain F32 overwrite, BF16 out, three-region and HC-write
//! (`_hcw`) entries.
//!
//! Exactness (byte-identical outputs): every 16x16 output tile runs the same
//! chain of `v_wmma_f32_16x16x16_f16` over ascending 16-K steps, C chained
//! from +0.0, with the same A and B operands as hipcc, so no output depends
//! on the tiling. A has the same f16 values as hipcc's dequantization, element
//! for element (`sc*q + zp`, one RNE rounding, with the group header's f16
//! scale / zero point: hA for K-steps 0..7 of a 256-group, hB for 8..15), but
//! two codes at a time: the 6-bit codes of a pair are placed in the mantissas
//! of two f16 with a constant exponent (`1024 + q`, `64 + q` or `256 + q`,
//! all exact for q < 64), `v_pk_add_f16` of the bias recovers `q` exactly
//! (the same f16 as `v_cvt_f32_ubyte0` + `v_cvt_f16_f32`), and one
//! `v_pk_fma_f16` (operands sc, q, zp; sc and zp read from the header halves
//! with `op_sel`) is the same fused expression as hipcc's `v_fma_f16`. Lane
//! `l` and `l+16` hold the same weight row (own eight values,
//! `v_permlanex16` for the other half's, `v_cndmask` into the 8-VGPR
//! fragment, hipcc's construction), so every fragment element has hipcc's
//! bits and lane placement. The epilogue swaps accumulator halves exactly as
//! hipcc (pure bit moves) and stores `0.0f + acc` (`v_add_f32 0, acc`).
//!
//! Freedom used (tiling, layout and scheduling only):
//!  * a wave owns 16 rows x 256 tokens (16 accumulator tiles, 128 VGPRs), so
//!    one A dequant feeds 16 WMMAs (hipcc: 8); a workgroup is `wr` waves
//!    (64 or 128 rows) x 256 tokens;
//!  * X is staged in 32-K chunks, double-buffered in LDS (two 20480-byte
//!    slots, 80-byte token pitch), one barrier per chunk;
//!  * B fragments (8 VGPRs: two `ds_load_b128`) rotate through a ring of
//!    four loaded four WMMAs ahead; the next K-step's A dequant is
//!    interleaved with the current WMMAs;
//!  * each lane loads only its own 48 code bits per K-step (a b64 at
//!    `12j + 4*lane_half`) and gathers them with `v_perm_b32` (MQ6: four, one per code pair);
//!  * X loads go through a descriptor bounded at `N*K*2` bytes, so tokens
//!    past N read zeros (their outputs are never stored).
//!
//! HC-write (`_hcw`): stage 1 per 128-token half parks `bf16(0 + acc)` in LDS
//! (256-byte token rows, 32768 bytes, over the X slots) and mirrors the F32
//! values when `Y != 0`; stage 2 (a runtime loop of 8 items per thread)
//! applies `hcw_apply8` to the streams with the arithmetic of
//! `qwen4_mq6_x4_gfx11_hcw`.
//!
//! Launch: block `32*wr`, grid `[ceil(M/(16*wr)), ceil(N/256), 1]` (regions:
//! x = sum of the regions' row blocks), static LDS 40960 bytes. K is a
//! multiple of 256 (a ragged K tail is ignored, as in hipcc).
//!
//! gfx1100 residual twin (`Kind::Resid`, `Cfg::RESID` wr 4 / 8): an exact twin of hipcc's
//! `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds` (`gemm_mqv2_wmma_gfx11_mw_lds.hip`,
//! `Y[out_col * M + out_row] += acc[j]`, F32). The A operand, the B operand and the WMMA chain are
//! the ones above (hipcc's mw kernel decodes with the same contracted `v_fma_f16` of
//! `sc_h * (f16)q + zp_h`, one rounding; both run the ascending-K chain from +0.0), so only the
//! epilogue differs: after the cross-half swap, Y is loaded (two b128 per tile under the
//! token < N, row < M mask), `v_add_f32 acc, acc, y` (hipcc: `v_add_f32_e32 v1, v1, v14`, acc is
//! src0) and stored; no `0 + acc`. Tails: rows >= M and tokens >= N are neither loaded nor
//! stored; `M % 8 != 0` takes a per-element path. Launch: block `32*wr`, grid
//! `[ceil(M/(16*wr)), ceil(N/256), 1]`, static LDS 40960; the symbols are
//! `gemm_mq6g256v2_residual_pm_<arch>_w<wr>` (`Cfg::symbol`), both in one object (`emit_resid_module`), kernargs `A X Y M K N` (36 bytes). Residual entries
//! assemble for gfx1100 (the product) and gfx1151 (the emulator differential: the instruction
//! streams are identical); every other arch refuses. A K < 256 call adds an all-zero
//! accumulator to Y exactly as hipcc does; K <= 0 exits without touching Y.
//!
//! MQ4 (`Fmt::Mq4`, `qwen4_mq4_x4_gfx11`): the MQ4G256V2 entries are exact
//! twins of the current hipcc MQ4 route (`gemm_mq4g256v2_residual_wmma_gfx11_bt4`,
//! contract C reference): the same A operands (`v_cvt_f32_ubyte0`,
//! `v_cvt_f16_f32`, one `v_fma_f16` with the group's f16 scale / zero point:
//! single rounding), the same B operands (F16 rotated X) and the same WMMA
//! chain. They differ from MQ6 only in the weight format: 136-byte groups
//! (8 header bytes, then 128 nibble bytes), one `buffer_load_b32` of eight
//! nibbles per lane per 16-K step at group byte `8 + 8*kt + 4*lh` and a
//! 4-bit `v_bfe_u32` per value (no `v_perm_b32`). The module
//! `qwen4_mq4_x4_pm_gfx1151` exports `qwen4_mq4_x4_pm_gfx1151_{w8, w4,
//! w8_bf16out, w4_bf16out, w8_regions, w8_hcw}` with the ABIs of the MQ6 entries.
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, smem, sop, sr, v, vr};
use crate::kernels::qwen4_mq6_x4_gfx11_hcw as hcw;
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// Entry kind: plain F32 overwrite, BF16 out, three regions, HC-write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Bf16,
    Regions,
    Hcw,
    /// gfx1100 residual: `Y += acc` in F32 (the `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds` twin).
    Resid,
}

/// Weight format of the A operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Mq6,
    Mq4,
}
impl Fmt {
    /// Lower-case format tag used in symbols, modules and labels.
    fn tag(self) -> &'static str {
        match self {
            Fmt::Mq6 => "mq6",
            Fmt::Mq4 => "mq4",
        }
    }
    fn kernel_id(self) -> &'static str {
        match self {
            Fmt::Mq6 => "qwen4_mq6_x4",
            Fmt::Mq4 => "qwen4_mq4_x4",
        }
    }
    fn builder(self) -> &'static str {
        match self {
            Fmt::Mq6 => "qwen4_mq6_x4_gfx11",
            Fmt::Mq4 => "qwen4_mq4_x4_gfx11",
        }
    }
}

/// `wr` waves (16 rows each) per workgroup, of one MQ6 entry kind (the public
/// MQ6 API; the MQ4 twin is `qwen4_mq4_x4_gfx11::Spec`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub wr: u8,
    pub kind: Kind,
}
impl Spec {
    /// The six PM entries.
    pub const ALL: [(u8, Kind); 6] = Cfg::ALL;
    pub fn cfg(self) -> Cfg {
        Cfg {
            arch: self.arch,
            wr: self.wr,
            kind: self.kind,
            fmt: Fmt::Mq6,
        }
    }
    pub fn symbol(&self) -> String {
        self.cfg().symbol()
    }
    pub fn module(arch: Arch) -> String {
        Cfg::module(Fmt::Mq6, arch)
    }
    pub fn validate(&self) -> R {
        self.cfg().validate()
    }
    /// Rows per workgroup.
    pub fn rows(&self) -> u32 {
        self.cfg().rows()
    }
}

/// One entry of one weight format: `Spec` plus the format. Every
/// format-dependent decision (group byte stride, code load, dequant) reads
/// `fmt`; `Fmt::Mq6` selects the original code everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cfg {
    pub arch: Arch,
    pub wr: u8,
    pub kind: Kind,
    pub fmt: Fmt,
}
impl Cfg {
    /// The six PM entries of either format.
    pub const ALL: [(u8, Kind); 6] = [
        (8, Kind::Plain),
        (4, Kind::Plain),
        (8, Kind::Bf16),
        (4, Kind::Bf16),
        (8, Kind::Regions),
        (8, Kind::Hcw),
    ];
    /// Waves per workgroup of the gfx1100 residual twin entries.
    pub const RESID: [u8; 2] = [4, 8];
    fn suffix(&self) -> &'static str {
        match self.kind {
            Kind::Plain => "",
            Kind::Bf16 => "_bf16out",
            Kind::Regions => "_regions",
            Kind::Hcw => "_hcw",
            Kind::Resid => "_resid",
        }
    }
    /// Exported symbol. The residual twin of hipcc's `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds`
    /// is `gemm_mq6g256v2_residual_pm_<arch>_w<wr>`: distinct from hipcc's so both can be
    /// loaded in one product process.
    pub fn symbol(&self) -> String {
        if self.kind == Kind::Resid {
            return format!(
                "gemm_mq6g256v2_residual_pm_{}_w{}",
                self.arch.name(),
                self.wr
            );
        }
        format!(
            "qwen4_{}_x4_pm_{}_w{}{}",
            self.fmt.tag(),
            self.arch.name(),
            self.wr,
            self.suffix()
        )
    }
    pub fn module(fmt: Fmt, arch: Arch) -> String {
        format!("qwen4_{}_x4_pm_{}", fmt.tag(), arch.name())
    }
    /// Module name of the residual twin object (both `wr` entries).
    pub fn resid_module(arch: Arch) -> String {
        format!("gemm_mq6g256v2_residual_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.kind == Kind::Resid {
            if !matches!(self.arch, Arch::Gfx1100 | Arch::Gfx1151) || self.fmt != Fmt::Mq6 {
                return Err(format!(
                    "{} residual (Y += acc) entry is MQ6 gfx1100 / gfx1151 only",
                    self.fmt.builder()
                ));
            }
            if !Self::RESID.contains(&self.wr) {
                return Err(format!(
                    "{}: no residual entry w{}",
                    self.fmt.builder(),
                    self.wr
                ));
            }
            return Ok(());
        }
        if self.arch != Arch::Gfx1151 {
            return Err(format!(
                "{} builder is exact-gfx1151 only",
                self.fmt.builder()
            ));
        }
        if !Self::ALL.contains(&(self.wr, self.kind)) {
            return Err(format!(
                "{}: no entry w{} {:?}",
                self.fmt.builder(),
                self.wr,
                self.kind
            ));
        }
        Ok(())
    }
    /// Bytes of one 256-K weight group of the row: MQ6 200 (8 header + 192
    /// code), MQ4 136 (8 header + 128 code).
    fn stride(&self) -> u32 {
        match self.fmt {
            Fmt::Mq6 => 200,
            Fmt::Mq4 => 136,
        }
    }
    /// Dwords of codes a lane loads per 16-K step (the register slot keeps two).
    fn code_dwords(&self) -> u8 {
        match self.fmt {
            Fmt::Mq6 => 2,
            Fmt::Mq4 => 1,
        }
    }
    /// Group byte offset (plus the lane-half term in ACODE) of the codes of
    /// K-step `j` of chunk `cc`: MQ6 a b64 of the 12-byte step, MQ4 the b32 at
    /// `8 + 8 * kt` with `kt = 2 * cc + j`.
    fn code_offset(&self, cc: u8, j: u8) -> u32 {
        match self.fmt {
            Fmt::Mq6 => 8 + u32::from(cc) * 24 + u32::from(j) * 12,
            Fmt::Mq4 => 8 + u32::from(cc) * 16 + u32::from(j) * 8,
        }
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
    fn hcw(&self) -> bool {
        self.kind == Kind::Hcw
    }
    fn bf16(&self) -> bool {
        self.kind == Kind::Bf16
    }

    // VGPRs. v0 tid; v1..v9 lane constants; v10.. X buffer offsets; then the
    // 16 token-tile accumulators (8 each), the B ring (4 x 8), two A
    // fragments (8 each), codes (four chunks: loaded two chunks ahead),
    // group headers (current, next), the dequant temporaries (q0..q7, the
    // packed values m[0..3] alias q0..q3 and the swapped halves o[0..3] alias
    // q4..q7), X staging (4*xpt) and, for HCW, the LDS parking address.
    fn acc(&self) -> u8 {
        10 + self.xpt()
    }
    fn ringb(&self) -> u8 {
        self.acc() + 128
    }
    fn ring(&self, i: u8) -> u8 {
        self.ringb() + 8 * (i % RING)
    }
    fn af(&self, i: usize) -> u8 {
        self.ringb() + 32 + 8 * i as u8
    }
    fn code_base(&self) -> u8 {
        self.ringb() + 48
    }
    /// Code buffer of chunk `cc` (mod 4; a group's 8 chunks keep the rotation).
    fn code(&self, cc: u8) -> u8 {
        self.code_base() + 4 * (cc % 4)
    }
    fn hdr(&self, i: usize) -> u8 {
        self.code_base() + 16 + 2 * i as u8
    }
    fn q(&self) -> u8 {
        self.code_base() + 20
    }
    fn lo(&self) -> u8 {
        self.q() + 8
    }
    fn d1(&self) -> u8 {
        self.q() + 9
    }
    fn xr(&self) -> u8 {
        self.q() + 10
    }
    /// MQ6 packed dequant: the `v_perm_b32` selector VGPR of code pair `p`.
    /// Pairs 0 and 1 take the lane-constant slots the scalar dequant's
    /// selectors used; pairs 2 and 3 take its `lo` / `d1` temporaries (live
    /// over the main loop only, which is where the selectors are read).
    fn pair_sel(&self, p: u8) -> u8 {
        match p {
            0 => SELLO,
            1 => SELD1,
            2 => self.lo(),
            _ => self.d1(),
        }
    }
    fn park(&self) -> u8 {
        self.xr() + 4 * self.xpt()
    }
    /// Highest VGPR used, plus one.
    fn vgprs_used(&self) -> u16 {
        u16::from(self.xr()) + 4 * u16::from(self.xpt()) + u16::from(self.hcw())
    }
    /// The descriptor allocates VGPRs in granules of 8: the plan ends on one.
    fn vgprs(&self) -> u16 {
        self.vgprs_used().next_multiple_of(8)
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
const RING: u8 = 4;
/// WMMA after which the chunk barrier sits: its ring loads (WMMA i loads the
/// fragment of WMMA i + RING) end with this chunk's last fragment.
const BAR_AT: u8 = WPC - RING - 1;
const L_X0: usize = 0;
/// HC parking: 128 tokens x 256 bytes (128 BF16 channels) per half.
const PARK_ROW: u32 = 256;
const PARK_BYTES: u32 = 128 * PARK_ROW;
/// Mask of an item's parking address: `item * 16` below 32768, 16-aligned.
const PARK_MASK: u32 = 0x7ff0;

/// Local labels, qualified by variant so every entry shares one module.
fn lbl(spec: &Cfg, name: &str) -> String {
    let k = match spec.kind {
        Kind::Plain => "p",
        Kind::Bf16 => "b",
        Kind::Regions => "r",
        Kind::Hcw => "h",
        Kind::Resid => "y",
    };
    format!(".L{}g11w{}{k}_{name}", spec.fmt.tag(), spec.wr)
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; the workgroup ids follow the DECLARED user SGPRs: s2, s3 on gfx1151, but on
// gfx1100 `FeatureUserSGPRInit16Bug` (LLVM) the kernarg pointer is padded with dead user SGPRs up to
// `user + system (workgroup ids x, y) = 16`, i.e. 14 declared user SGPRs: ids s14, s15 (and the output
// descriptor `SRD_Y`, which would overlap them, moves out of s[12:15]).
/// Real user SGPRs (the kernarg pointer) and enabled system SGPRs (workgroup ids x, y).
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 2;
/// `.amdhsa_user_sgpr_count` of `arch` (`Arch::declared_user_sgprs`: user + system >= 16 on gfx1100).
const fn declared_user_sgprs(arch: Arch) -> u8 {
    arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS)
}
const KARG: u8 = 0;
/// Workgroup id x / y: the first two system SGPRs, after the declared user SGPRs.
const fn wgx_in(arch: Arch) -> u8 {
    declared_user_sgprs(arch)
}
const fn wgy_in(arch: Arch) -> u8 {
    declared_user_sgprs(arch) + 1
}
const SRD_A: u8 = 4;
const SRD_X: u8 = 8;
/// Output (Y) buffer descriptor `s[SRD_Y:SRD_Y+3]`: s[12:15] on gfx1151; on gfx1100 s[56:59] (s14 / s15
/// are the workgroup ids; the residual entries use none of the region / HCW scalars at s[56:79]).
const fn srd_y(arch: Arch) -> u8 {
    if wgy_in(arch) >= SRD_Y_GFX1151 {
        56
    } else {
        SRD_Y_GFX1151
    }
}
const SRD_Y_GFX1151: u8 = 12;
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
const SSEL: u8 = 38;
const YSTEP: u8 = 40;
const SNB: u8 = 41;
const SMJ: u8 = 42;
const SK2: u8 = 43;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
const MASK0: u8 = 44;
const MASK1: u8 = 46;
const MASKB: u8 = 48;
const PSEL: u8 = 50;
const MASKY: u8 = 52;
/// Lane-half mask (`lh != 0`), live from the prologue to the end.
const LHM: u8 = 54;
// Regions: raw kernarg copies. s[56:63] A0 A1 A2 X; s[64:71] Y0 Y1 Y2 M0 M1; s72 M2; s74 K, s75 N.
const R_A0: u8 = 56;
const R_A1: u8 = 58;
const R_A2: u8 = 60;
const R_X: u8 = 62;
const R_Y0: u8 = 64;
const R_Y1: u8 = 66;
const R_Y2: u8 = 68;
const R_M0: u8 = 70;
const R_M1: u8 = 71;
const R_M2: u8 = 72;
const R_K: u8 = 74;
const R_N: u8 = 75;
// HCW: s[56:57] streams, s[58:59] gates, s60 BF16 flag, stage-2 scalars, stream/gate descriptors.
const HS: u8 = 56;
const HG: u8 = 58;
const HBF: u8 = 60;
const SM4: u8 = 61;
const SBR1: u8 = 62;
const SBR2: u8 = 63;
const SBR3: u8 = 64;
const SSTEPS: u8 = 65;
const SIT: u8 = 66;
const STOK: u8 = 67;
const SROWB: u8 = 68;
const SRD_S: u8 = 72;
const SRD_G: u8 = 76;
/// MQ6 packed-dequant constants (live from the prologue to the end): field
/// masks of the even / odd code pairs and the `v_pk_add_f16` constants that
/// remove the exponent bias (`-1024` / `-64` and `-256` per half).
const PK_MASKE: u8 = 80;
const PK_MASKO: u8 = 81;
const PK_CE: u8 = 82;
const PK_CO: u8 = 83;

/// Epilogue VGPRs (live only from `epi`): they alias the dead main loop's ring
/// and everything after it.
#[derive(Clone, Copy)]
struct Ep {
    tmp: u8,
    offs: u8,
    pset: u8,
    strm: u8,
    mq: u8,
    g: u8,
    h: u8,
    vt: u8,
    voffs: u8,
    voffg: u8,
    vla: u8,
    vlb: u8,
    vc: u8,
    /// Residual epilogue: Y-load registers (`YG` tiles x 8), after the 16 tile offsets.
    yb: u8,
}
fn ep(spec: &Cfg) -> Ep {
    let e = spec.ringb();
    Ep {
        tmp: e,
        offs: e + 6,
        pset: e + 14,
        strm: e + 30,
        mq: e + 62,
        g: e + 66,
        h: e + 70,
        vt: e + 79,
        voffs: e + 80,
        voffg: e + 81,
        vla: e + 82,
        vlb: e + 83,
        vc: e + 84,
        yb: e + 22,
    }
}

const TID: u8 = 0;
const SELLO: u8 = 1;
const SELD1: u8 = 2;
const ACODE: u8 = 3;
const AHDR: u8 = 4;
const LDST: u8 = 5;
const LDRD: u8 = 6;
const VTOK: u8 = 7;
const VYOFF: u8 = 8;
const VROW0: u8 = 9;
const XOFF: u8 = 10;

fn plan(spec: &Cfg) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(spec.vgprs(), 104)?;
    let w = || Live::Whole;
    let dead = || Live::Between("entry".into(), lbl(spec, "epi"));
    let epi = || Live::Between(lbl(spec, "epi"), lbl(spec, "end"));
    p.v::<1>("tid", TID, w())?;
    for n in SELLO..=VROW0 {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in 0..spec.xpt() {
        p.v::<1>("x_offset", XOFF + n, w())?;
    }
    for t in 0..BT {
        p.v::<8>("acc", spec.acc() + 8 * t, w())?;
    }
    for r in 0..RING {
        p.v::<8>("b_ring", spec.ringb() + 8 * r, dead())?;
    }
    for i in 0..2 {
        p.v::<8>("a_frag", spec.af(i), dead())?;
    }
    for c in 0..4u8 {
        for j in 0..2u8 {
            p.v::<2>("codes", spec.code(c) + 2 * j, dead())?;
        }
    }
    for h in 0..2 {
        p.v::<2>("header", spec.hdr(h), dead())?;
    }
    p.v::<8>("dequant_q", spec.q(), dead())?;
    if spec.fmt == Fmt::Mq6 {
        p.v::<1>("pair_sel", spec.lo(), dead())?;
        p.v::<1>("pair_sel", spec.d1(), dead())?;
    } else {
        p.v::<1>("dequant_temp", spec.lo(), dead())?;
        p.v::<1>("dequant_temp", spec.d1(), dead())?;
    }
    for i in 0..spec.xpt() {
        p.v::<4>("x_stage", spec.xr() + 4 * i, dead())?;
    }
    if spec.hcw() {
        p.v::<1>("park_addr", spec.park(), w())?;
    }
    if spec.vgprs() > spec.vgprs_used() {
        p.v::<1>("granule_pad", (spec.vgprs() - 1) as u8, w())?;
    }
    let e = ep(spec);
    for n in 0..6 {
        p.v::<1>("epi_temp", e.tmp + n, epi())?;
    }
    if spec.hcw() {
        for n in 0..8 {
            p.v::<1>("epi_offset", e.offs + n, epi())?;
        }
        for n in 0..4 {
            p.v::<4>("park_stage", e.pset + 4 * n, epi())?;
        }
        for n in 0..4 {
            p.v::<8>("stream", e.strm + 8 * n, epi())?;
        }
        p.v::<4>("parked", e.mq, epi())?;
        p.v::<4>("gates", e.g, epi())?;
        for n in 0..hcw::TEMPS {
            p.v::<1>("hcw_temp", e.h + n, epi())?;
        }
        for n in [e.vt, e.voffs, e.voffg, e.vla, e.vlb, e.vc] {
            p.v::<1>("item", n, epi())?;
        }
    } else {
        for n in 0..16 {
            p.v::<1>("epi_offset", e.offs + n, epi())?;
        }
        if spec.kind == Kind::Resid {
            for n in 0..YG {
                p.v::<8>("y_load", e.yb + 8 * n, epi())?;
            }
        }
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<1>("wg_x_in", wgx_in(spec.arch), w())?;
    p.s::<1>("wg_y_in", wgy_in(spec.arch), w())?;
    for srd in [SRD_A, SRD_X, srd_y(spec.arch)] {
        p.s::<4>("srd", srd, w())?;
    }
    p.s::<8>("kernargs", PA, w())?;
    for n in SN..=SSEL {
        p.s::<1>("scalar", n, w())?;
    }
    for n in YSTEP..=SK2 {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1, MASKB, LHM] {
        p.s::<2>("lane_mask", m, w())?;
    }
    p.s::<1>("permlane_sel", PSEL, w())?;
    if spec.fmt == Fmt::Mq6 {
        for n in PK_MASKE..=PK_CO {
            p.s::<1>("pk_const", n, w())?;
        }
    }
    match spec.kind {
        Kind::Regions => {
            p.s::<8>("raw_a", R_A0, w())?;
            p.s::<8>("raw_y", R_Y0, w())?;
            p.s::<1>("raw_m2", R_M2, w())?;
            p.s::<2>("raw_kn", R_K, w())?;
        }
        Kind::Hcw => {
            p.s::<2>("lane_mask", MASKY, w())?;
            p.s::<4>("hc_ptrs", HS, w())?;
            for n in HBF..=SROWB {
                p.s::<1>("hc_scalar", n, w())?;
            }
            for srd in [SRD_S, SRD_G] {
                p.s::<4>("srd", srd, w())?;
            }
        }
        _ => {}
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
/// buffer_store_b{16,32,128}: `dwords` 0 stores the low 16 bits of `data`.
fn bstore(b: &mut Builder, dwords: u8, data: u8, voff: u8, srd: u8, soff: u8, imm: u32) -> R {
    let (w, n) = match dwords {
        0 => ("b16".to_string(), 1),
        1 => ("b32".to_string(), 1),
        4 => ("b128".to_string(), 4),
        _ => return Err("bstore width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "buffer_store_{w} {}, v{voff}, s[{srd}:{}], s{soff} offen{}",
                rs(data, n),
                srd + 3,
                off(imm)
            ),
            vec![],
            vec![vr(data, n), v(voff), sr(srd, 4), s(soff)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
fn wmma(b: &mut Builder, spec: &Cfg, t: u8, a: u8, bb: u8) -> R {
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
fn emit_fill(b: &mut Builder, spec: &Cfg, f: &Fill) -> R {
    match f {
        Fill::BLoad { slot, k } => {
            let (j, t) = (u32::from(k / BT), u32::from(k % BT));
            let dst = spec.ring(*k);
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
        Fill::XStore { slot, i, imm } => {
            let data = spec.xr() + 4 * i;
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
            spec.code_dwords(),
            *dst,
            ACODE,
            SRD_A,
            *soff,
            spec.code_offset(*cc, *j),
        ),
        Fill::HdrLoad { dst, soff } => bload(b, 2, *dst, AHDR, SRD_A, *soff, 0),
        Fill::Valu { text, defs, uses } => op(b, text.clone(), defs, uses),
    }
}
fn xstore(spec: &Cfg, slot: usize, i: u8) -> Fill {
    Fill::XStore {
        slot,
        i,
        imm: u32::from(i) * spec.xrow_step() * PITCH,
    }
}

/// MQ6 `v_perm_b32` selectors (lane half 0, lane half 1) of the four code
/// pairs: the lane's 48-bit window is bytes `2*lh ..` of its 8 loaded bytes.
/// Pair `p` (codes 2p and 2p+1, window bits 12p..12p+11) is gathered into a
/// dword whose two 16-bit halves each hold one six-bit field within the low
/// five bits of the half (the other bytes are zeroed or ignored by the mask):
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

/// Dequantize one K-step into A fragment `a` (8 VGPRs, both lane halves hold
/// the full row) from header dword `h` (sc in .l, zp in .h) and the codes at
/// `w`. MQ6: two dwords, this lane half's 48 bits at bit 0 for lane half 0,
/// bit 16 for lane half 1. MQ4: one dword, this lane half's eight nibbles.
///
/// MQ6 runs packed pairs (exact: the same f16 values as `sc*q + zp` per code
/// with one `v_fma_f16` rounding): the code pair is placed in the mantissas
/// of two f16 with a constant exponent (`v_and_or_b32` against the pair's
/// field mask and exponent bits: `0x6400 | q` is `1024 + q` for a field at
/// bit 0, `0x5400 | q << 4` is `64 + q` for a field at bit 4,
/// `0x5c00 | q << 2` is `256 + q` for a field at bit 2), `v_pk_add_f16` of
/// the bias recovers `q` exactly, and `v_pk_fma_f16` (sc and zp straight from
/// the header halves through `op_sel`, operand order sc, q, zp) gives the
/// packed values m[k] (codes 2k, 2k+1) in q0..q3. MQ4 runs eight per-value
/// chains (`v_bfe_u32`, `v_cvt_f32_ubyte0`, `v_cvt_f16_f32`, `v_fma_f16`)
/// into m[0..3]. Then, breadth-first, their `permlanex16` o[0..3] (q4..q7)
/// and hipcc's selects `A[k] = lh ? o[k] : m[k]`, `A[k+4] = lh ? m[k] : o[k]`.
fn dequant(spec: &Cfg, w: u8, h: u8, a: u8) -> Vec<Fill> {
    let q = spec.q();
    let mut out = Vec::new();
    let mut f = |text: String, defs: Vec<RegRef>, uses: Vec<RegRef>| {
        out.push(Fill::Valu { text, defs, uses })
    };
    match spec.fmt {
        Fmt::Mq6 => {
            for p in 0..4u8 {
                let sel = spec.pair_sel(p);
                f(
                    format!("v_perm_b32 v{}, v{}, v{w}, v{sel}", q + p, w + 1),
                    vec![v(q + p)],
                    vec![v(w), v(w + 1), v(sel)],
                );
            }
            for p in [0u8, 2] {
                f(
                    format!("v_lshl_or_b32 v{0}, v{0}, 10, v{0}", q + p),
                    vec![v(q + p)],
                    vec![v(q + p)],
                );
            }
            for p in 0..4u8 {
                let (mask, exp) = if p % 2 == 0 {
                    (PK_MASKE, 0x6400_6400u32)
                } else {
                    (PK_MASKO, 0x5c00_5400)
                };
                f(
                    format!("v_and_or_b32 v{0}, v{0}, s{mask}, {1}", q + p, lit(exp)),
                    vec![v(q + p)],
                    vec![v(q + p), s(mask)],
                );
            }
            for p in 0..4u8 {
                let bias = if p % 2 == 0 { PK_CE } else { PK_CO };
                f(
                    format!("v_pk_add_f16 v{0}, v{0}, s{bias}", q + p),
                    vec![v(q + p)],
                    vec![v(q + p), s(bias)],
                );
            }
            for p in 0..4u8 {
                f(
                    format!(
                        "v_pk_fma_f16 v{0}, v{h}, v{0}, v{h} op_sel:[0,0,1] op_sel_hi:[0,1,1]",
                        q + p
                    ),
                    vec![v(q + p)],
                    vec![v(h), v(q + p)],
                );
            }
        }
        Fmt::Mq4 => {
            // Code i of the lane half's dword sits at bits 4i..4i+3.
            for i in 0..8u8 {
                f(
                    format!("v_bfe_u32 v{}, v{w}, {}, 4", q + i, 4 * i),
                    vec![v(q + i)],
                    vec![v(w)],
                );
            }
            for i in 0..8u8 {
                f(
                    format!("v_cvt_f32_ubyte0_e32 v{0}, v{0}", q + i),
                    vec![v(q + i)],
                    vec![v(q + i)],
                );
            }
            for i in 0..8u8 {
                f(
                    format!("v_cvt_f16_f32_e64 v{0}.l, v{0}", q + i),
                    vec![v(q + i)],
                    vec![v(q + i)],
                );
            }
            for i in 0..8u8 {
                let d = q + i / 2;
                let (half, sel) = if i % 2 == 0 {
                    ("l", "op_sel:[0,0,1,0]")
                } else {
                    ("h", "op_sel:[0,0,1,1]")
                };
                f(
                    format!("v_fma_f16 v{d}.{half}, v{h}.l, v{}.l, v{h}.h {sel}", q + i),
                    vec![v(d)],
                    vec![v(d), v(h), v(q + i)],
                );
            }
        }
    }
    for k in 0..4u8 {
        f(
            format!(
                "v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98",
                q + 4 + k,
                q + k
            ),
            vec![v(q + 4 + k)],
            vec![v(q + k), s(PSEL)],
        );
    }
    for k in 0..4u8 {
        f(
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{LHM}",
                a + k,
                q + k,
                q + 4 + k
            ),
            vec![v(a + k)],
            vec![v(q + k), v(q + 4 + k), s(LHM)],
        );
        f(
            format!(
                "v_cndmask_b32_e64 v{}, v{}, v{}, s{LHM}",
                a + 4 + k,
                q + 4 + k,
                q + k
            ),
            vec![v(a + 4 + k)],
            vec![v(q + k), v(q + 4 + k), s(LHM)],
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
fn hdr_dword(spec: &Cfg, cc: u8) -> u8 {
    spec.hdr(0) + u8::from(cc >= CHUNKS / 2)
}

/// One 32-K chunk `cc` (0..7) of the group loop body: 32 WMMAs.
fn chunk(b: &mut Builder, spec: &Cfg, cc: u8) -> R {
    let slot = usize::from(cc % 2);
    let other = 1 - slot;
    let cb = spec.code(cc);
    let ncb = spec.code(cc + 1);
    // X chunk 8g+cc+2 (clamped): its loads follow this chunk's barrier.
    sop(
        b,
        format!("s_add_i32 s{SXOFF}, s{C8}, {}", cc + 2),
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
            dst: spec.code(cc + 2) + 2 * j,
            soff: psoff,
            cc: pcc,
            j,
        });
    }
    if cc + 2 == CHUNKS {
        slots[0].push(Fill::HdrLoad {
            dst: spec.hdr(1),
            soff: NGOFF,
        });
    }
    // B ring: WMMA i loads WMMA i+RING's fragment into the register it just read.
    for i in 0..WPC {
        if i + RING < WPC {
            slots[usize::from(i)].push(Fill::BLoad { slot, k: i + RING });
        }
    }
    // Stage chunk c+1's X into the other slot, well before the barrier.
    let stores: Vec<Fill> = (0..spec.xpt()).map(|i| xstore(spec, other, i)).collect();
    spread(&mut slots, stores, 18, 26);
    // A(j=1) under K-step 0; A of the next chunk's K-step 0 under K-step 1.
    spread(
        &mut slots,
        dequant(spec, cb + 2, hdr_dword(spec, cc), spec.af(1)),
        0,
        usize::from(BT) - 1,
    );
    let nh = if cc + 1 == CHUNKS {
        spec.hdr(1)
    } else {
        hdr_dword(spec, cc + 1)
    };
    spread(
        &mut slots,
        dequant(spec, ncb, nh, spec.af(0)),
        usize::from(BT),
        usize::from(WPC) - 1,
    );
    for i in 0..WPC {
        let (j, t) = (i / BT, i % BT);
        wmma(b, spec, t, spec.af(usize::from(j)), spec.ring(i))?;
        for f in &slots[usize::from(i)] {
            emit_fill(b, spec, f)?;
        }
        if i == BAR_AT {
            // This wave's reads of `slot` have landed (the last ring loads of
            // this chunk); publish the other slot, retire this one.
            b.wait(Counter::Lgkm, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in 0..spec.xpt() {
                bload(b, 4, spec.xr() + 4 * x, XOFF + x, SRD_X, SXOFF, 0)?;
            }
        } else if i > BAR_AT {
            // The next chunk's first fragments, from the published slot.
            emit_fill(
                b,
                spec,
                &Fill::BLoad {
                    slot: other,
                    k: i + RING - WPC,
                },
            )?;
        }
    }
    Ok(())
}

/// Canonical scalar kernargs (A, X, Y, M, K, N), the workgroup row block and
/// the early exits; per-kind kernarg loads and region selection first.
fn kernargs(b: &mut Builder, spec: &Cfg) -> R {
    match spec.kind {
        Kind::Regions => {
            smem(b, R_A0, 8, KARG, 0)?;
            smem(b, R_Y0, 8, KARG, 32)?;
            smem(b, R_M2, 1, KARG, 64)?;
            smem(b, R_K, 2, KARG, 68)?;
        }
        _ => {
            smem(b, PA, 8, KARG, 0)?;
            smem(b, SN, 1, KARG, 0x20)?;
            if spec.hcw() {
                smem(b, HS, 4, KARG, 0x28)?;
                smem(b, HBF, 1, KARG, 0x38)?;
            }
        }
    }
    Ok(())
}

fn select_region(b: &mut Builder, spec: &Cfg) -> R {
    let rows_m1 = spec.rows() - 1;
    let shift = spec.rows().trailing_zeros();
    // N, K > 0 else return.
    sop(
        b,
        format!("s_min_i32 s{ST0}, s{R_K}, s{R_N}"),
        &[ST0],
        &[R_K, R_N],
    )?;
    sop(b, format!("s_cmp_lt_i32 s{ST0}, 1"), &[], &[ST0])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    // r0, r1: row blocks of regions 0 and 1; SSEL = r0 + r1.
    for (r, m) in [(ST0, R_M0), (ST1, R_M1)] {
        sop(b, format!("s_max_i32 s{r}, s{m}, 0"), &[r], &[m])?;
        sop(
            b,
            format!("s_add_i32 s{r}, s{r}, {}", lit(rows_m1)),
            &[r],
            &[r],
        )?;
        sop(b, format!("s_lshr_b32 s{r}, s{r}, {shift}"), &[r], &[r])?;
    }
    sop(
        b,
        format!("s_add_i32 s{SSEL}, s{ST0}, s{ST1}"),
        &[SSEL],
        &[ST0, ST1],
    )?;
    // rb < r0 + r1 ? region 1 : region 2, then rb < r0 ? region 0. WGX = the
    // region's first row block.
    sop(
        b,
        format!("s_cmp_lt_i32 s{}, s{SSEL}", wgx_in(spec.arch)),
        &[],
        &[wgx_in(spec.arch), SSEL],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{PA}:{}], s[{R_A1}:{}], s[{R_A2}:{}]",
            PA + 1,
            R_A1 + 1,
            R_A2 + 1
        ),
        &[sr(PA, 2)],
        &[sr(R_A1, 2), sr(R_A2, 2)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{PY}:{}], s[{R_Y1}:{}], s[{R_Y2}:{}]",
            PY + 1,
            R_Y1 + 1,
            R_Y2 + 1
        ),
        &[sr(PY, 2)],
        &[sr(R_Y1, 2), sr(R_Y2, 2)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{R_M1}, s{R_M2}"),
        &[SM],
        &[R_M1, R_M2],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{WGX}, s{ST0}, s{SSEL}"),
        &[WGX],
        &[ST0, SSEL],
    )?;
    sop(
        b,
        format!("s_cmp_lt_i32 s{}, s{ST0}", wgx_in(spec.arch)),
        &[],
        &[wgx_in(spec.arch), ST0],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{PA}:{}], s[{R_A0}:{}], s[{PA}:{}]",
            PA + 1,
            R_A0 + 1,
            PA + 1
        ),
        &[sr(PA, 2)],
        &[sr(R_A0, 2), sr(PA, 2)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{PY}:{}], s[{R_Y0}:{}], s[{PY}:{}]",
            PY + 1,
            R_Y0 + 1,
            PY + 1
        ),
        &[sr(PY, 2)],
        &[sr(R_Y0, 2), sr(PY, 2)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{R_M0}, s{SM}"),
        &[SM],
        &[R_M0, SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{WGX}, 0, s{WGX}"),
        &[WGX],
        &[WGX],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{WGX}, s{}, s{WGX}", wgx_in(spec.arch)),
        &[WGX],
        &[wgx_in(spec.arch), WGX],
    )?;
    op(
        b,
        format!("s_mov_b64 s[{PX}:{}], s[{R_X}:{}]", PX + 1, R_X + 1),
        &[sr(PX, 2)],
        &[sr(R_X, 2)],
    )?;
    sop(b, format!("s_mov_b32 s{SK}, s{R_K}"), &[SK], &[R_K])?;
    sop(b, format!("s_mov_b32 s{SN}, s{R_N}"), &[SN], &[R_N])?;
    // M <= 0 or the local row block past the region: return.
    sop(b, format!("s_cmp_lt_i32 s{SM}, 1"), &[], &[SM])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(
        b,
        format!("s_add_i32 s{ST0}, s{SM}, {}", lit(rows_m1)),
        &[ST0],
        &[SM],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{ST0}, s{ST0}, {shift}"),
        &[ST0],
        &[ST0],
    )?;
    sop(b, format!("s_cmp_ge_i32 s{WGX}, s{ST0}"), &[], &[WGX, ST0])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])
}

fn prologue(b: &mut Builder, spec: &Cfg) -> R {
    let t = |n: u8| spec.ringb() + n; // prologue temporaries alias the (not yet loaded) ring
    let (t0, t1, t2, t3, t4, t5) = (t(0), t(1), t(2), t(3), t(4), t(5));
    kernargs(b, spec)?;
    // Lane constants that need no kernargs.
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
    if spec.hcw() {
        // Parking address of (token ml, byte 32*wave + 16*lh): ml * 256 + (tid & ~15).
        op(
            b,
            format!("v_and_b32_e32 v{t5}, -16, v{TID}"),
            &[v(t5)],
            &[v(TID)],
        )?;
        op(
            b,
            format!(
                "v_mad_u32_u24 v{}, v{t0}, {}, v{t5}",
                spec.park(),
                lit(PARK_ROW)
            ),
            &[v(spec.park())],
            &[v(t0), v(t5)],
        )?;
    }
    op(
        b,
        format!("v_cmp_ne_u32_e64 s{LHM}, 0, v{t1}"),
        &[s(LHM)],
        &[v(t1)],
    )?;
    if spec.fmt == Fmt::Mq6 {
        for (p, (l0, l1)) in PAIR_SEL.into_iter().enumerate() {
            let sel = spec.pair_sel(p as u8);
            op(
                b,
                format!("v_mov_b32_e32 v{t5}, {}", lit(l0)),
                &[v(t5)],
                &[],
            )?;
            op(
                b,
                format!("v_cndmask_b32_e64 v{sel}, v{t5}, {}, s{LHM}", lit(l1)),
                &[v(sel)],
                &[v(t5), s(LHM)],
            )?;
        }
        sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
        for (n, c) in [
            (PK_MASKE, 0x003f_003fu32),
            (PK_MASKO, 0x00fc_03f0),
            (PK_CE, 0xe400_e400),
            (PK_CO, 0xdc00_d400),
        ] {
            sop(b, format!("s_mov_b32 s{n}, {}", lit(c)), &[n], &[])?;
        }
    } else {
        op(
            b,
            format!("v_mov_b32_e32 v{t5}, {}", lit(0x0302_0100)),
            &[v(t5)],
            &[],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{SELLO}, v{t5}, {}, s{LHM}",
                lit(0x0504_0302)
            ),
            &[v(SELLO)],
            &[v(t5), s(LHM)],
        )?;
        op(
            b,
            format!("v_mov_b32_e32 v{t5}, {}", lit(0x0c05_0403)),
            &[v(t5)],
            &[],
        )?;
        op(
            b,
            format!(
                "v_cndmask_b32_e64 v{SELD1}, v{t5}, {}, s{LHM}",
                lit(0x0c07_0605)
            ),
            &[v(SELD1)],
            &[v(t5), s(LHM)],
        )?;
        sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    }
    for n in 0..8 * BT {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", spec.acc() + n),
            &[v(spec.acc() + n)],
            &[],
        )?;
    }
    b.wait(Counter::Lgkm, 0)?;
    if spec.kind == Kind::Regions {
        select_region(b, spec)?;
    } else {
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
        sop(
            b,
            format!("s_mov_b32 s{WGX}, s{}", wgx_in(spec.arch)),
            &[WGX],
            &[wgx_in(spec.arch)],
        )?;
    }
    sop(
        b,
        format!("s_mov_b32 s{WGY}, s{}", wgy_in(spec.arch)),
        &[WGY],
        &[wgy_in(spec.arch)],
    )?;
    sop(b, format!("s_lshr_b32 s{GPR}, s{SK}, 8"), &[GPR], &[SK])?;
    sop(b, format!("s_lshr_b32 s{NCHM1}, s{SK}, 5"), &[NCHM1], &[SK])?;
    sop(
        b,
        format!("s_add_i32 s{NCHM1}, s{NCHM1}, -1"),
        &[NCHM1],
        &[NCHM1],
    )?;
    sop(
        b,
        format!("s_add_i32 s{GPRM1}, s{GPR}, -1"),
        &[GPRM1],
        &[GPR],
    )?;
    sop(b, format!("s_lshl_b32 s{SK2}, s{SK}, 1"), &[SK2], &[SK])?;
    // Buffer descriptors: A, Y and the HC streams / gates raw unbounded
    // (0x7fffffff), X bounded at N*K*2.
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{SN}, s{SK2}"),
        &[ST1],
        &[SN, SK2],
    )?;
    let mut srds = vec![(SRD_A, PA), (SRD_X, PX), (srd_y(spec.arch), PY)];
    if spec.hcw() {
        srds.extend([(SRD_S, HS), (SRD_G, HG)]);
    }
    for (srd, p) in srds {
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
        format!("v_lshrrev_b32_e32 v{t3}, 1, v{TID}"),
        &[v(t3)],
        &[v(TID)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    op(
        b,
        format!(
            "v_and_or_b32 v{t3}, v{t3}, {}, s{ST0}",
            lit(spec.rows() - 16)
        ),
        &[v(t3)],
        &[v(t3), s(ST0)],
    )?; // row_start
    op(
        b,
        format!("v_lshl_or_b32 v{VROW0}, v{t1}, 3, v{t3}"),
        &[v(VROW0)],
        &[v(t1), v(t3)],
    )?; // row_start + 8*lh
    op(
        b,
        format!("v_or_b32_e32 v{t3}, v{t3}, v{t0}"),
        &[v(t3)],
        &[v(t3), v(t0)],
    )?; // row
    sop(b, format!("s_add_i32 s{ST0}, s{SM}, -1"), &[ST0], &[SM])?;
    sop(
        b,
        format!("s_mul_i32 s{ST1}, s{GPR}, {:#x}", spec.stride()),
        &[ST1],
        &[GPR],
    )?;
    op(
        b,
        format!("v_min_i32_e32 v{t3}, s{ST0}, v{t3}"),
        &[v(t3)],
        &[s(ST0), v(t3)],
    )?; // safe_row
    op(
        b,
        format!("v_mul_lo_u32 v{AHDR}, v{t3}, s{ST1}"),
        &[v(AHDR)],
        &[v(t3), s(ST1)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{ACODE}, v{t1}, 2, v{AHDR}"),
        &[v(ACODE)],
        &[v(t1), v(AHDR)],
    )?;
    // Tokens: epilogue token wgy*256 + ml; X rows wgy*256 + i*step + tid/4.
    sop(b, format!("s_lshl_b32 s{ST0}, s{WGY}, 8"), &[ST0], &[WGY])?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VTOK}, s{ST0}, v{t0}"),
        &[v(VTOK)],
        &[s(ST0), v(t0)],
    )?;
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
        format!(
            "v_lshlrev_b32_e32 v{VYOFF}, {}, v{VYOFF}",
            if spec.bf16() { 1 } else { 2 }
        ),
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
    bload(b, 2, spec.hdr(0), AHDR, SRD_A, GOFF, 0)?;
    for j in 0..2u8 {
        emit_fill(
            b,
            spec,
            &Fill::CodeLoad {
                dst: spec.code(0) + 2 * j,
                soff: GOFF,
                cc: 0,
                j,
            },
        )?;
    }
    for i in 0..spec.xpt() {
        bload(b, 4, spec.xr() + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for i in 0..spec.xpt() {
        emit_fill(b, spec, &xstore(spec, L_X0, i))?;
    }
    b.barrier(&[Transition::Ready(L_X0)])?;
    // Chunk 1's codes, X chunk min(1, nch-1) and the ring's first fragments
    // in flight into the loop, as the body's back edge leaves them.
    for j in 0..2u8 {
        emit_fill(
            b,
            spec,
            &Fill::CodeLoad {
                dst: spec.code(1) + 2 * j,
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
        bload(b, 4, spec.xr() + 4 * i, XOFF + i, SRD_X, SXOFF, 0)?;
    }
    for f in dequant(spec, spec.code(0), spec.hdr(0), spec.af(0)) {
        emit_fill(b, spec, &f)?;
    }
    for k in 0..RING {
        emit_fill(b, spec, &Fill::BLoad { slot: L_X0, k })?;
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Cfg) -> R {
    // ngoff = min(g+1, gpr-1)*group bytes (the last group re-reads itself).
    sop(b, format!("s_add_i32 s{NGOFF}, s{G}, 1"), &[NGOFF], &[G])?;
    sop(
        b,
        format!("s_min_u32 s{NGOFF}, s{NGOFF}, s{GPRM1}"),
        &[NGOFF],
        &[NGOFF, GPRM1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{NGOFF}, s{NGOFF}, {:#x}", spec.stride()),
        &[NGOFF],
        &[NGOFF],
    )?;
    for cc in 0..CHUNKS {
        chunk(b, spec, cc)?;
    }
    let (h0, h1) = (spec.hdr(0), spec.hdr(1));
    op(b, format!("v_mov_b32_e32 v{h0}, v{h1}"), &[v(h0)], &[v(h1)])?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", h0 + 1, h1 + 1),
        &[v(h0 + 1)],
        &[v(h1 + 1)],
    )?;
    sop(b, format!("s_add_i32 s{G}, s{G}, 1"), &[G], &[G])?;
    sop(
        b,
        format!("s_add_i32 s{GOFF}, s{GOFF}, {:#x}", spec.stride()),
        &[GOFF],
        &[GOFF],
    )?;
    sop(b, format!("s_add_i32 s{C8}, s{C8}, 8"), &[C8], &[C8])?;
    sop(b, format!("s_cmp_lt_u32 s{G}, s{GPR}"), &[], &[G, GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

// ------------------------------------------------------------------- epilogue

/// `v_add_f32 0, acc` over `tiles` accumulator tiles from `first`.
fn add_zero(b: &mut Builder, spec: &Cfg, first: u8, tiles: u8) -> R {
    for n in 8 * first..8 * (first + tiles) {
        let a = spec.acc() + n;
        op(b, format!("v_add_f32_e32 v{a}, 0, v{a}"), &[v(a)], &[v(a)])?;
    }
    Ok(())
}

/// hipcc's cross-half swap (lines 913-924), in place over accumulator tile
/// `t`: afterwards v[acc+0..7] = `r[0..7]`, rows `8*lh .. 8*lh+7` of the tile.
/// `give_i = lh ? acc[i] : acc[4+i]`, `got_i = permlanex16(give_i)`,
/// `r[2i] = lh ? got_i : acc[i]`, `r[2i+1] = lh ? acc[4+i] : got_i`. The
/// outputs permute the inputs in two 3-cycles (positions 1,2,4 and 3,5,6),
/// each broken by one saved input.
fn swap_tile(b: &mut Builder, spec: &Cfg, e: &Ep, t: u8) -> R {
    let a = spec.acc() + 8 * t;
    let g = e.tmp;
    let (c1, c3) = (e.tmp + 4, e.tmp + 5);
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

/// BF16 of every value of tile `t` (RNE, non-finite truncated: hipcc lines
/// 944-946) in place: afterwards the BF16 bits are the HIGH half of each
/// register (low half dirty).
fn round_tile(b: &mut Builder, spec: &Cfg, e: &Ep, t: u8) -> R {
    let (a, c) = (e.tmp + 4, e.tmp + 5);
    for n in 0..8u8 {
        let x = spec.acc() + 8 * t + n;
        op(b, format!("v_bfe_u32 v{a}, v{x}, 16, 1"), &[v(a)], &[v(x)])?;
        op(
            b,
            format!("v_add3_u32 v{a}, v{x}, v{a}, 0x7fff"),
            &[v(a)],
            &[v(x), v(a)],
        )?;
        op(
            b,
            format!("v_and_b32_e32 v{c}, 0x7f800000, v{x}"),
            &[v(c)],
            &[v(x)],
        )?;
        op(
            b,
            format!("v_cmp_ne_u32_e64 s{MASKB}, 0x7f800000, v{c}"),
            &[s(MASKB)],
            &[v(c)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{x}, v{x}, v{a}, s{MASKB}"),
            &[v(x)],
            &[v(x), v(a), s(MASKB)],
        )?;
    }
    Ok(())
}

/// Y byte offsets of the tiles: `offs + n` = `base + n * ystep`.
fn tile_offsets(b: &mut Builder, offs: u8, n: u8, base: u8, ystep: u8) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{offs}, v{base}"),
        &[v(offs)],
        &[v(base)],
    )?;
    for t in 1..n {
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{}, s{ystep}, v{}",
                offs + t,
                offs + t - 1
            ),
            &[v(offs + t)],
            &[s(ystep), v(offs + t - 1)],
        )?;
    }
    Ok(())
}

fn plain_epilogue(b: &mut Builder, spec: &Cfg) -> R {
    let e = ep(spec);
    let bf16 = spec.bf16();
    add_zero(b, spec, 0, BT)?;
    for t in 0..BT {
        swap_tile(b, spec, &e, t)?;
        if bf16 {
            round_tile(b, spec, &e, t)?;
        }
    }
    sop(
        b,
        format!("s_lshl_b32 s{YSTEP}, s{SM}, {}", if bf16 { 5 } else { 6 }),
        &[YSTEP],
        &[SM],
    )?;
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    tile_offsets(b, e.offs, BT, VYOFF, YSTEP)?;
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
        let a = spec.acc() + 8 * t;
        if bf16 {
            // Pack the BF16 high halves: word p = (r[2p+1] & 0xffff0000) | (r[2p] >> 16).
            for p in 0..4u8 {
                op(
                    b,
                    format!(
                        "v_perm_b32 v{}, v{}, v{}, 0x7060302",
                        a + p,
                        a + 2 * p + 1,
                        a + 2 * p
                    ),
                    &[v(a + p)],
                    &[v(a + 2 * p + 1), v(a + 2 * p)],
                )?;
            }
        }
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
            &[s(MASK0)],
            &[s(SNB), v(VTOK)],
        )?;
        op(
            b,
            format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
            &[],
            &[s(MASK0), s(MASK1)],
        )?;
        bstore(b, 4, a, e.offs + t, srd_y(spec.arch), SZERO, 0)?;
        if !bf16 {
            bstore(b, 4, a + 4, e.offs + t, srd_y(spec.arch), SZERO, 16)?;
        }
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        sop(b, format!("s_add_i32 s{SNB}, s{SNB}, -16"), &[SNB], &[SNB])?;
    }
    // The ragged path is entered by the branch above, before these stores: drain
    // them here so the linear ledger at its label models only its own stores.
    b.wait_all()?;
    op(b, format!("s_branch {}", lbl(spec, "end")), &[], &[])?;
    // Ragged M: per element, row0 + e < M.
    b.label(&lbl(spec, "scalar"))?;
    for t in 0..BT {
        let a = spec.acc() + 8 * t;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
            &[s(MASK0)],
            &[s(SNB), v(VTOK)],
        )?;
        sop(b, format!("s_mov_b32 s{SMJ}, s{SM}"), &[SMJ], &[SM])?;
        for n in 0..8u8 {
            // BF16 sits in the high half: shift it down in place, once, for the
            // 16-bit store. Each accumulator register is stored from exactly once
            // and never redefined afterwards, so no store needs a source wait.
            if bf16 {
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{0}, 16, v{0}", a + n),
                    &[v(a + n)],
                    &[v(a + n)],
                )?;
            }
            op(
                b,
                format!("v_cmp_gt_i32_e64 s{MASK1}, s{SMJ}, v{VROW0}"),
                &[s(MASK1)],
                &[s(SMJ), v(VROW0)],
            )?;
            op(
                b,
                format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                &[],
                &[s(MASK0), s(MASK1)],
            )?;
            if bf16 {
                bstore(
                    b,
                    0,
                    a + n,
                    e.offs + t,
                    srd_y(spec.arch),
                    SZERO,
                    2 * u32::from(n),
                )?;
            } else {
                bstore(
                    b,
                    1,
                    a + n,
                    e.offs + t,
                    srd_y(spec.arch),
                    SZERO,
                    4 * u32::from(n),
                )?;
            }
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            sop(b, format!("s_add_i32 s{SMJ}, s{SMJ}, -1"), &[SMJ], &[SMJ])?;
        }
        sop(b, format!("s_add_i32 s{SNB}, s{SNB}, -16"), &[SNB], &[SNB])?;
    }
    Ok(())
}

/// HC-write stage 2 of one 128-token half over the parked BF16 values: a
/// runtime loop of 8 items (token t, 8 channels c0..c0+7) per thread.
fn hcw_stage2(b: &mut Builder, spec: &Cfg, h: u8, bf16: bool) -> R {
    let e = ep(spec);
    let es_shift = if bf16 { 1 } else { 2 };
    // Per-branch stream strides, the per-iteration stream step (16 tokens x 4
    // branches x M x element size) and the item's token / channel.
    sop(b, format!("s_lshl_b32 s{SM4}, s{SM}, 2"), &[SM4], &[SM])?;
    sop(
        b,
        format!("s_lshl_b32 s{SBR1}, s{SM}, {es_shift}"),
        &[SBR1],
        &[SM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SBR2}, s{SM}, {}", es_shift + 1),
        &[SBR2],
        &[SM],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{SBR3}, s{SBR1}, 3"),
        &[SBR3],
        &[SBR1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SSTEPS}, s{SM}, {}", 6 + es_shift),
        &[SSTEPS],
        &[SM],
    )?;
    sop(b, format!("s_lshl_b32 s{STOK}, s{WGY}, 8"), &[STOK], &[WGY])?;
    sop(
        b,
        format!("s_add_i32 s{STOK}, s{STOK}, {}", lit(128 * u32::from(h))),
        &[STOK],
        &[STOK],
    )?;
    let (t0, t1) = (e.tmp, e.tmp + 1);
    op(
        b,
        format!("v_lshrrev_b32_e32 v{t0}, 4, v{TID}"),
        &[v(t0)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{}, s{STOK}, v{t0}", e.vt),
        &[v(e.vt)],
        &[s(STOK), v(t0)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t1}, 15, v{TID}"),
        &[v(t1)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{}, v{t1}, 3, s{SROWB}", e.vc),
        &[v(e.vc)],
        &[v(t1), s(SROWB)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{}, v{}, s{SM4}", e.voffs, e.vt),
        &[v(e.voffs)],
        &[v(e.vt), s(SM4)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{0}, v{0}, v{1}", e.voffs, e.vc),
        &[v(e.voffs)],
        &[v(e.voffs), v(e.vc)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{0}, {es_shift}, v{0}", e.voffs),
        &[v(e.voffs)],
        &[v(e.voffs)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{}, 4, v{}", e.voffg, e.vt),
        &[v(e.voffg)],
        &[v(e.vt)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{}, 4, v{TID}", e.vla),
        &[v(e.vla)],
        &[v(TID)],
    )?;
    sop(b, format!("s_mov_b32 s{SIT}, 0"), &[SIT], &[])?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASKB}, s{SM}, v{}", e.vc),
        &[s(MASKB)],
        &[s(SM), v(e.vc)],
    )?;
    b.wait_all()?;
    let name = format!("s2_h{h}_{}", if bf16 { "b" } else { "f" });
    b.loop_(&lbl(spec, &name), |b| {
        let sbr = [SZERO, SBR1, SBR2, SBR3];
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SN}, v{}", e.vt),
            &[s(MASK0)],
            &[s(SN), v(e.vt)],
        )?;
        op(
            b,
            format!("s_and_b32 exec_lo, s{MASK0}, s{MASKB}"),
            &[],
            &[s(MASK0), s(MASKB)],
        )?;
        bload(b, 4, e.g, e.voffg, SRD_G, SZERO, 0)?;
        for (br, &so) in sbr.iter().enumerate() {
            let d = e.strm + 8 * br as u8;
            bload(b, 4, d, e.voffs, SRD_S, so, 0)?;
            if !bf16 {
                bload(b, 4, d + 4, e.voffs, SRD_S, so, 16)?;
            }
        }
        op(
            b,
            format!("v_and_b32_e32 v{}, {PARK_MASK:#x}, v{}", e.vlb, e.vla),
            &[v(e.vlb)],
            &[v(e.vla)],
        )?;
        b.ds_load(
            0,
            Instruction::new(
                format!("ds_load_b128 {}, v{}", rs(e.mq, 4), e.vlb),
                vec![vr(e.mq, 4)],
                vec![v(e.vlb)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        for i in 0..4u8 {
            hcw::gate(b, e.g + i, e.g + i, e.h)?;
        }
        for br in 0..4u8 {
            let d = e.strm + 8 * br;
            if bf16 {
                hcw::apply_bf16(b, d, d, e.mq, e.g + br, e.h)?;
            } else {
                hcw::apply_f32(b, d, d, e.mq, e.g + br, e.h)?;
            }
        }
        for (br, &so) in sbr.iter().enumerate() {
            let d = e.strm + 8 * br as u8;
            bstore(b, 4, d, e.voffs, SRD_S, so, 0)?;
            if !bf16 {
                bstore(b, 4, d + 4, e.voffs, SRD_S, so, 16)?;
            }
        }
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{0}, 16, v{0}", e.vt),
            &[v(e.vt)],
            &[v(e.vt)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{0}, s{SSTEPS}, v{0}", e.voffs),
            &[v(e.voffs)],
            &[s(SSTEPS), v(e.voffs)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{0}, 0x100, v{0}", e.voffg),
            &[v(e.voffg)],
            &[v(e.voffg)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{0}, 0x1000, v{0}", e.vla),
            &[v(e.vla)],
            &[v(e.vla)],
        )?;
        b.wait_all()?;
        sop(b, format!("s_add_i32 s{SIT}, s{SIT}, 1"), &[SIT], &[SIT])?;
        sop(b, format!("s_cmp_lt_u32 s{SIT}, 8"), &[], &[SIT])?;
        op(b, format!("s_cbranch_scc1 {}", lbl(spec, &name)), &[], &[])
    })
}

/// HC-write epilogue: per 128-token half, stage 1 (swap, mirror, park) then
/// stage 2 (apply to the streams).
fn hcw_epilogue(b: &mut Builder, spec: &Cfg) -> R {
    let e = ep(spec);
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    // The mirror is off when Y is null: MASKY is all ones iff Y != 0.
    sop(
        b,
        format!("s_or_b32 s{ST0}, s{PY}, s{}", PY + 1),
        &[ST0],
        &[PY, PY + 1],
    )?;
    sop(b, format!("s_cmp_lg_u32 s{ST0}, 0"), &[], &[ST0])?;
    sop(b, format!("s_cselect_b32 s{MASKY}, -1, 0"), &[MASKY], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{SROWB}, s{WGX}, {}", lit(spec.rows())),
        &[SROWB],
        &[WGX],
    )?;
    sop(b, format!("s_lshl_b32 s{YSTEP}, s{SM}, 6"), &[YSTEP], &[SM])?;
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{VROW0}"),
        &[s(MASK1)],
        &[s(SM), v(VROW0)],
    )?;
    sop(
        b,
        format!("s_and_b32 s{MASK1}, s{MASK1}, s{MASKY}"),
        &[MASK1],
        &[MASK1, MASKY],
    )?;
    // Retire the last X slot (no further reads), then the parking slot
    // replaces the X layout.
    b.barrier(&[Transition::Retire(0)])?;
    b.lds_relayout()?;
    if b.lds_slot("park", 0, PARK_BYTES)? != 0 {
        return Err("LDS park slot order".into());
    }
    for h in 0..2u8 {
        let first = 8 * h;
        add_zero(b, spec, first, 8)?;
        for t in first..first + 8 {
            swap_tile(b, spec, &e, t)?;
        }
        // Stage 1: F32 mirror (`Y != 0`) and BF16 parking of this half's 8 tiles.
        if h == 0 {
            sop(b, format!("s_mov_b32 s{SNB}, s{SN}"), &[SNB], &[SN])?;
        } else {
            sop(
                b,
                format!("s_add_i32 s{SNB}, s{SN}, {}", lit((-128i32) as u32)),
                &[SNB],
                &[SN],
            )?;
        }
        if h == 0 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, v{VYOFF}", e.offs),
                &[v(e.offs)],
                &[v(VYOFF)],
            )?;
        } else {
            sop(b, format!("s_lshl_b32 s{ST1}, s{SM}, 9"), &[ST1], &[SM])?;
            op(
                b,
                format!("v_add_nc_u32_e32 v{}, s{ST1}, v{VYOFF}", e.offs),
                &[v(e.offs)],
                &[s(ST1), v(VYOFF)],
            )?;
        }
        for t in 1..8u8 {
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
        for bt in 0..8u8 {
            let a = spec.acc() + 8 * (first + bt);
            op(
                b,
                format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
                &[s(MASK0)],
                &[s(SNB), v(VTOK)],
            )?;
            op(
                b,
                format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                &[],
                &[s(MASK0), s(MASK1)],
            )?;
            bstore(b, 4, a, e.offs + bt, srd_y(spec.arch), SZERO, 0)?;
            bstore(b, 4, a + 4, e.offs + bt, srd_y(spec.arch), SZERO, 16)?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
            sop(b, format!("s_add_i32 s{SNB}, s{SNB}, -16"), &[SNB], &[SNB])?;
            // hcw_pack8: word p = bf16(r[2p]) | bf16(r[2p+1]) << 16.
            let pset = e.pset + 4 * (bt % 4);
            let (lo, hi) = (e.tmp, e.tmp + 1);
            for p in 0..4u8 {
                hcw::bf16_bits(b, lo, a + 2 * p, e.h)?;
                hcw::bf16_bits(b, hi, a + 2 * p + 1, e.h)?;
                op(
                    b,
                    format!("v_lshl_or_b32 v{}, v{hi}, 16, v{lo}", pset + p),
                    &[v(pset + p)],
                    &[v(hi), v(lo)],
                )?;
            }
            b.ds_store(
                0,
                Instruction::new(
                    format!(
                        "ds_store_b128 v{}, {}{}",
                        spec.park(),
                        rs(pset, 4),
                        off(u32::from(bt) * 16 * PARK_ROW)
                    ),
                    vec![],
                    vec![v(spec.park()), vr(pset, 4)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.barrier(&[Transition::Ready(0)])?;
        // Stage 2 over the state kind (uniform branch on `hc_bf16`).
        sop(b, format!("s_cmp_lg_u32 s{HBF}, 0"), &[], &[HBF])?;
        let fl = lbl(spec, &format!("s2f_h{h}"));
        let jl = lbl(spec, &format!("s2j_h{h}"));
        op(b, format!("s_cbranch_scc0 {fl}"), &[], &[])?;
        hcw_stage2(b, spec, h, true)?;
        op(b, format!("s_branch {jl}"), &[], &[])?;
        b.label(&fl)?;
        hcw_stage2(b, spec, h, false)?;
        b.label(&jl)?;
        b.barrier(&[Transition::Retire(0)])?;
    }
    Ok(())
}

/// Tiles per Y-load batch of the residual epilogue (8 VGPRs per tile).
const YG: u8 = 8;

/// Residual epilogue (`Y += acc`, F32), hipcc's `Y[(long long)out_col * M + out_row] += acc[j]`
/// for `out_col < N`, `out_row < M`: after the cross-half swap each lane owns rows
/// `8*lh .. 8*lh+7` of its token's tile column, so the stores are two b128 runs; the value
/// stored is `v_add_f32 acc, acc, y` (hipcc's `v_add_f32_e32 v1, v1, v14`: acc is src0, the
/// loaded Y src1; no `0 + acc` canonicalisation, the accumulator is added to Y as is). Y is
/// loaded and stored under the same exec mask (token < N, row < M), so nothing outside the
/// output is read. Ragged M (`M % 8 != 0`) loads, adds and stores element by element.
fn resid_epilogue(b: &mut Builder, spec: &Cfg) -> R {
    let e = ep(spec);
    for t in 0..BT {
        swap_tile(b, spec, &e, t)?;
    }
    sop(b, format!("s_lshl_b32 s{YSTEP}, s{SM}, 6"), &[YSTEP], &[SM])?;
    sop(b, format!("s_mov_b32 s{SZERO}, 0"), &[SZERO], &[])?;
    tile_offsets(b, e.offs, BT, VYOFF, YSTEP)?;
    sop(b, format!("s_and_b32 s{ST0}, s{SM}, 7"), &[ST0], &[SM])?;
    sop(b, format!("s_cmp_lg_u32 s{ST0}, 0"), &[], &[ST0])?;
    op(
        b,
        format!("s_cbranch_scc1 {}", lbl(spec, "scalar")),
        &[],
        &[],
    )?;
    // M % 8 == 0: whole 8-row runs (row0 < M implies row0 + 8 <= M).
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK1}, s{SM}, v{VROW0}"),
        &[s(MASK1)],
        &[s(SM), v(VROW0)],
    )?;
    // The token mask of tile `t` is recomputed under full exec each time: exec is
    // restored to all ones after every tile, since a VOPC under a partial exec clears
    // the result bits of the inactive lanes.
    let tile_exec = |b: &mut Builder, t: u8| -> R {
        sop(
            b,
            format!("s_sub_i32 s{SNB}, s{SN}, {}", lit(16 * u32::from(t))),
            &[SNB],
            &[SN],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
            &[s(MASK0)],
            &[s(SNB), v(VTOK)],
        )?;
        op(
            b,
            format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
            &[],
            &[s(MASK0), s(MASK1)],
        )
    };
    for g in 0..BT / YG {
        for k in 0..YG {
            let t = g * YG + k;
            tile_exec(b, t)?;
            bload(b, 4, e.yb + 8 * k, e.offs + t, srd_y(spec.arch), SZERO, 0)?;
            bload(
                b,
                4,
                e.yb + 8 * k + 4,
                e.offs + t,
                srd_y(spec.arch),
                SZERO,
                16,
            )?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        }
        for k in 0..YG {
            let t = g * YG + k;
            let a = spec.acc() + 8 * t;
            tile_exec(b, t)?;
            for i in 0..8u8 {
                op(
                    b,
                    format!("v_add_f32_e32 v{0}, v{0}, v{1}", a + i, e.yb + 8 * k + i),
                    &[v(a + i)],
                    &[v(a + i), v(e.yb + 8 * k + i)],
                )?;
            }
            bstore(b, 4, a, e.offs + t, srd_y(spec.arch), SZERO, 0)?;
            bstore(b, 4, a + 4, e.offs + t, srd_y(spec.arch), SZERO, 16)?;
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        }
    }
    b.wait_all()?;
    op(b, format!("s_branch {}", lbl(spec, "end")), &[], &[])?;
    // Ragged M: per element, row0 + n < M.
    b.label(&lbl(spec, "scalar"))?;
    for t in 0..BT {
        let a = spec.acc() + 8 * t;
        sop(
            b,
            format!("s_sub_i32 s{SNB}, s{SN}, {}", lit(16 * u32::from(t))),
            &[SNB],
            &[SN],
        )?;
        op(
            b,
            format!("v_cmp_gt_i32_e64 s{MASK0}, s{SNB}, v{VTOK}"),
            &[s(MASK0)],
            &[s(SNB), v(VTOK)],
        )?;
        for phase in 0..2u8 {
            sop(b, format!("s_mov_b32 s{SMJ}, s{SM}"), &[SMJ], &[SM])?;
            for n in 0..8u8 {
                op(
                    b,
                    format!("v_cmp_gt_i32_e64 s{MASK1}, s{SMJ}, v{VROW0}"),
                    &[s(MASK1)],
                    &[s(SMJ), v(VROW0)],
                )?;
                op(
                    b,
                    format!("s_and_b32 exec_lo, s{MASK0}, s{MASK1}"),
                    &[],
                    &[s(MASK0), s(MASK1)],
                )?;
                if phase == 0 {
                    bload(
                        b,
                        1,
                        e.yb + n,
                        e.offs + t,
                        srd_y(spec.arch),
                        SZERO,
                        4 * u32::from(n),
                    )?;
                } else {
                    op(
                        b,
                        format!("v_add_f32_e32 v{0}, v{0}, v{1}", a + n, e.yb + n),
                        &[v(a + n)],
                        &[v(a + n), v(e.yb + n)],
                    )?;
                    bstore(
                        b,
                        1,
                        a + n,
                        e.offs + t,
                        srd_y(spec.arch),
                        SZERO,
                        4 * u32::from(n),
                    )?;
                }
                op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
                sop(b, format!("s_add_i32 s{SMJ}, s{SMJ}, -1"), &[SMJ], &[SMJ])?;
            }
        }
    }
    Ok(())
}

fn epilogue(b: &mut Builder, spec: &Cfg) -> R {
    b.wait_all()?;
    b.label(&lbl(spec, "epi"))?;
    if spec.hcw() {
        hcw_epilogue(b, spec)?;
    } else if spec.kind == Kind::Resid {
        resid_epilogue(b, spec)?;
    } else {
        plain_epilogue(b, spec)?;
    }
    b.label(&lbl(spec, "end"))?;
    b.control(Sop::End.encode(spec.arch)?)
}

/// One MQ6 entry (`Cfg` with `Fmt::Mq6`).
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_cfg(spec.cfg())
}

/// One entry of either weight format.
pub fn emit_cfg(spec: Cfg) -> Result<Emitted, String> {
    spec.validate()?;
    let by_value = |k: KernargLayout, names: &[(&str, u32)]| {
        names
            .iter()
            .fold(k, |k, (n, o)| k.hidden(n, *o, 4, "by_value"))
    };
    let kernargs = match spec.kind {
        Kind::Plain | Kind::Bf16 | Kind::Resid => by_value(
            KernargLayout::new(36)
                .pointer("A", 0)
                .pointer("X", 8)
                .pointer("Y", 16),
            &[("M", 24), ("K", 28), ("N", 32)],
        ),
        Kind::Regions => by_value(
            KernargLayout::new(76)
                .pointer("A0", 0)
                .pointer("A1", 8)
                .pointer("A2", 16)
                .pointer("X", 24)
                .pointer("Y0", 32)
                .pointer("Y1", 40)
                .pointer("Y2", 48),
            &[("M0", 56), ("M1", 60), ("M2", 64), ("K", 68), ("N", 72)],
        ),
        Kind::Hcw => by_value(
            KernargLayout::new(60)
                .pointer("A", 0)
                .pointer("X", 8)
                .pointer("Y", 16)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value")
                .hidden("N", 32, 4, "by_value")
                .pointer("hc_streams", 40)
                .pointer("hc_gates", 48),
            &[("hc_bf16", 56)],
        ),
    };
    let variant = format!("w{}{}", spec.wr, spec.suffix());
    let kspec = KernelSpec {
        kernel_id: spec.fmt.kernel_id().into(),
        variant,
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: USER_SGPRS,
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

/// Both residual twin entries (`Cfg::RESID`) as one code object (`Cfg::resid_module`).
pub fn emit_resid_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Cfg::RESID
        .into_iter()
        .map(|wr| {
            emit_cfg(Cfg {
                arch,
                wr,
                kind: Kind::Resid,
                fmt: Fmt::Mq6,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Cfg::resid_module(arch))?;
    Ok((emitted, text, proof))
}

/// Every MQ6 entry as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    emit_module_fmt(Fmt::Mq6, arch)
}

/// Every entry of weight format `fmt` as one code object (`Cfg::module`).
pub fn emit_module_fmt(
    fmt: Fmt,
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Cfg::ALL
        .into_iter()
        .map(|(wr, kind)| {
            emit_cfg(Cfg {
                arch,
                wr,
                kind,
                fmt,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Cfg::module(fmt, arch))?;
    Ok((emitted, text, proof))
}
