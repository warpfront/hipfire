// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact one-row-per-workgroup decode GEMV over MQ3-I4 / MQ2-I4 planar tensors
//! (HFQ qt=131..=134), emitted with the checked ISA builder for gfx1201 only:
//! four symbols `mq_i4_rows_{w3,w2}_{r256,r128}` in the module
//! `mq_i4_rows_pm_gfx1201` (committed image `kernels/mq_i4_rows_pm_gfx1201.hxaco`).
//!
//! ABI (32-byte kernarg segment, identical for all four symbols): `codes@0`
//! (read-only pointer to the planar tensor), `x@8` (read-only pointer, `K` f32,
//! the rotated activation), `y@16` (write-only pointer, `M` f32), `M@24` (u32),
//! `K@28` (u32). The scale plane is not an argument: it starts at
//! `codes + M*K*b/8` (`b` = 3 for W3, 2 for W2).
//!
//! Launch: grid `[M, 1, 1]`, block `[32, 1, 1]` (one wave32 per workgroup, one
//! output row per workgroup, no LDS). Workgroups with `row >= M` return.
//!
//! Domain (host-validated, the kernel relies on it and does not re-check):
//! `K > 0`, `K % 128 == 0`, `K % R == 0` (`R` = 256 for R256, 128 for R128),
//! `M > 0`. Every 32-bit scalar quantity must not wrap: the whole tensor
//! (`M*K*b/8 + 2*M*K/128` bytes, hence also the codes plane and every row/scale
//! offset), `4*K` (`x`) and `4*M` (`y`) are at most `u32::MAX` bytes. Codes are
//! the planar bitstream of `mq_i4` (row `r` at byte `r*K*b/8`, code `j` at bits
//! `[b*j, b*(j+1))` LSB first, scale of row `r` group `g` the little-endian f16
//! at `codes_bytes + 2*(r*(K/128) + g)`). Every b-bit pattern is a legal code and
//! scales are finite f16 (the loader's validation); the kernel never inspects
//! them. The rotation `R` selects the reduction tree only, never `K`.
//!
//! Exactness: each output is bitwise `mq_i4_reference::decode_gemv_f32` for the
//! row (the frozen F4 tree). Lane `l` (0..32) of the wave owns elements
//! `g*128 + 4l ..= g*128 + 4l + 3` of every 128-column group `g`, in ascending
//! `g`:
//!  * `w_i = RN(float(d) * float(2c_i - L))`, `L = 2^b - 1`, `d` the group's f16
//!    scale converted exactly to f32 (`v_cvt_f32_f16`), `2c_i - L` a small exact
//!    integer converted exactly (`v_cvt_f32_i32`), one `v_mul_f32` rounding;
//!  * `p = RN(w_0 * x_0)` (`v_mul_f32`), then `p = fma(w_i, x_i, p)` for
//!    `i = 1, 2, 3` (`v_fmac_f32`, one rounding each), left associated;
//!  * the group partial is added separately: R256 into `acc[g & 3]` (four chains,
//!    `acc[s] = acc[s] + p`), R128 into the single `acc0`; accumulators start at
//!    `+0`;
//!  * lane value: R256 `RN(RN(a0 + a1) + RN(a2 + a3))`, R128 `a0`;
//!  * cross-lane snapshot shuffle-down at offsets 16, 8, 4, 2, 1 (`ds_bpermute_b32`
//!    reads `lane + off`, or the lane's own value when out of range; each step reads
//!    every lane's pre-step value): `sum = RN(acc + shuffled)` into a scratch
//!    register, then `v_cndmask_b32` selects `sum` for lanes with `lane + off <= 31`
//!    (the step's `v_cmp` mask, still live in vcc) and the old `acc` otherwise, so
//!    out-of-range lanes stay unchanged exactly as in the reference; lane 0's value
//!    is the result and lane 0 stores `y[row]`.
//!
//! Unpack (simple network, no optimization): a lane's four codes of a group are
//! one packet. W3: 12 bits at bit `384g + 12l` of the row, i.e. group byte
//! `(3l) >> 1` shifted right by `4*(l & 1)`; read as two byte loads
//! (`global_load_u8` at `+0` and `+1`, never straddling past the group's 48
//! bytes and never over-reading the plane), combined `b0 | b1 << 8`, shifted,
//! and each code `v_bfe_u32(packet, 3i, 3)`. W2: one byte (group byte `l`, 32
//! bytes per group), code `v_bfe_u32(byte, 2i, 2)`. The signed weight integer is
//! `2c - L` (no MQ4 XOR rebias).

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, s_add_i32, smem, sop};
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 32;
/// Workgroup size (threads): one wave32.
pub const WORKGROUP_SIZE: u32 = 32;
/// Scale / activation group width in columns.
pub const GROUP: u32 = 128;

/// Code width of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// 3-bit codes, `2c - 7`; a lane's four codes are a 12-bit packet.
    W3,
    /// 2-bit codes, `2c - 3`; a lane's four codes are one byte.
    W2,
}
impl Codec {
    pub const ALL: [Codec; 2] = [Codec::W3, Codec::W2];
    /// Bits per code.
    pub fn bits(self) -> u32 {
        match self {
            Self::W3 => 3,
            Self::W2 => 2,
        }
    }
    /// `L = 2^b - 1`.
    pub fn levels(self) -> u32 {
        (1 << self.bits()) - 1
    }
    /// Code bytes of one 128-column group of a row: `128*b/8`.
    pub fn group_bytes(self) -> u32 {
        GROUP * self.bits() / 8
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::W3 => "w3",
            Self::W2 => "w2",
        }
    }
}

/// Reduction tree selector (the profile's rotation width).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// Four group chains `acc[g & 3]`, folded `(a0 + a1) + (a2 + a3)`.
    R256,
    /// One chain.
    R128,
}
impl Rotation {
    pub const ALL: [Rotation; 2] = [Rotation::R256, Rotation::R128];
    /// Rotation width in columns (`K % width == 0`).
    pub fn width(self) -> u32 {
        match self {
            Self::R256 => 256,
            Self::R128 => 128,
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::R256 => "r256",
            Self::R128 => "r128",
        }
    }
}

/// One entry: `(arch, codec, rotation)`; gfx1201 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub codec: Codec,
    pub rotation: Rotation,
}
impl Spec {
    /// The exported symbol, `mq_i4_rows_<codec>_<rotation>`.
    pub fn symbol(&self) -> String {
        format!("mq_i4_rows_{}", self.tag())
    }
    /// `<codec>_<rotation>`: the `--epi` name, the label prefix and the contract file tag.
    pub fn tag(&self) -> String {
        format!("{}_{}", self.codec.tag(), self.rotation.tag())
    }
    pub fn module(arch: Arch) -> String {
        format!("mq_i4_rows_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("mq_i4_rows builder is exact-gfx1201 only".into());
        }
        Ok(())
    }
}

/// Every entry of one architecture in module order: W3 R256, W3 R128, W2 R256, W2 R128.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    Codec::ALL
        .into_iter()
        .flat_map(|codec| {
            Rotation::ALL.into_iter().map(move |rotation| Spec {
                arch,
                codec,
                rotation,
            })
        })
        .collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; the workgroup id is ttmp9 (gfx12).
const KARG: u8 = 0;
/// Kernargs (one 8-aligned `s_load_b256`): s[8:9] codes, s[10:11] x, s[12:13] y, s14 M, s15 K.
const CODES: u8 = 8;
const XP: u8 = 10;
const YP: u8 = 12;
const SM: u8 = 14;
const SK: u8 = 15;
const ROW: u8 = 16;
const GROUPS: u8 = 17;
const ROW_BYTES: u8 = 18;
const CODES_BYTES: u8 = 19;
/// The row's code pointer and scale pointer (even-aligned pairs).
const ROW_PTR: u8 = 20;
const TMP: u8 = 22;
const TMP2: u8 = 23;
const SCALE_PTR: u8 = 24;
const GI: u8 = 26;
const QUADS: u8 = 27;
const G_CODE: u8 = 28;
const G_X: u8 = 29;
const G_SCALE: u8 = 30;
const Y_OFF: u8 = 31;

// ----------------------------------------------------------------------- VGPRs
const LANE: u8 = 0;
const CODE_ADDR: u8 = 1;
const SCALE_ADDR: u8 = 2;
const X_LANE: u8 = 3;
const SHUFFLE: u8 = 4;
/// Per-lane code byte offset within a group, and (W3) the 0/4 bit shift of its packet.
const CODE_LANE: u8 = 5;
const SHIFT: u8 = 6;
const X_ADDR: u8 = 7;
const PKT0: u8 = 8;
const PKT1: u8 = 9;
const SCALE: u8 = 10;
const DOT: u8 = 11;
const W: u8 = 12;
const X: u8 = 16;
const ACC: u8 = 20;
const EXCH: u8 = 24;
const STORE_ADDR: u8 = 25;

fn lbl(spec: Spec, name: &str) -> String {
    format!(".Lmqi4_{}_{name}", spec.tag())
}

fn plan(spec: Spec) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(32, 40)?;
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("kernel_arguments", CODES, 8),
        ("row", ROW, 1),
        ("groups", GROUPS, 1),
        ("row_bytes", ROW_BYTES, 1),
        ("codes_bytes", CODES_BYTES, 1),
        ("row_code_pointer", ROW_PTR, 2),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("scale_pointer", SCALE_PTR, 2),
        ("group_index", GI, 1),
        ("quad_end", QUADS, 1),
        ("group_code_offset", G_CODE, 1),
        ("group_x_offset", G_X, 1),
        ("group_scale_offset", G_SCALE, 1),
        ("y_offset", Y_OFF, 1),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    let mut v: Vec<(&str, u8, u8)> = vec![
        ("lane", LANE, 1),
        ("code_address", CODE_ADDR, 1),
        ("scale_address", SCALE_ADDR, 1),
        ("x_lane_offset", X_LANE, 1),
        ("shuffle_address", SHUFFLE, 1),
        ("code_lane_offset", CODE_LANE, 1),
        ("x_address", X_ADDR, 1),
        ("packet0", PKT0, 1),
        ("scale", SCALE, 1),
        ("dot", DOT, 1),
        ("weights", W, 4),
        ("x_values", X, 4),
        ("acc", ACC, 4),
        ("exchange", EXCH, 1),
        ("store_address", STORE_ADDR, 1),
    ];
    if spec.codec == Codec::W3 {
        v.push(("packet_shift", SHIFT, 1));
        v.push(("packet1", PKT1, 1));
    }
    for (name, base, len) in v {
        regs.add_range(name, Kind::V, base, len, Live::Whole)?;
    }
    // The code object's VGPR reservation is whole 8-register granules: declare the unused
    // tail so the metadata count equals the descriptor's reservation (as in every other family).
    regs.add_range("granule_pad_lo", Kind::V, 26, 2, Live::Whole)?;
    regs.add_range("granule_pad_hi", Kind::V, 28, 4, Live::Whole)?;
    Ok(regs)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("codes", 0, Access::ReadOnly)
        .pointer_access("x", 8, Access::ReadOnly)
        .pointer_access("y", 16, Access::WriteOnly)
        .hidden("M", 24, 4, "by_value")
        .hidden("K", 28, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq_i4_rows".into(),
        variant: spec.tag(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(spec)?);
    b.enable_delay_alu();
    prologue(&mut b, spec)?;
    match spec.rotation {
        Rotation::R256 => {
            // Groups strictly ascending; group g feeds chain g & 3: whole quads in the
            // loop, then up to three tail groups on chains 0, 1, 2.
            sop(
                &mut b,
                format!("s_and_b32 s{QUADS}, s{GROUPS}, -4"),
                &[QUADS],
                &[GROUPS],
            )?;
            b.wait_all()?;
            sop(&mut b, format!("s_cmp_eq_u32 s{QUADS}, 0"), &[], &[QUADS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "tail")))?;
            b.loop_(&lbl(spec, "quad"), |b| {
                for chain in 0..4 {
                    group(b, spec, chain)?;
                }
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{QUADS}"),
                    &[],
                    &[GI, QUADS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "quad")))
            })?;
            b.label(&lbl(spec, "tail"))?;
            for chain in 0..3 {
                sop(
                    &mut b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(&mut b, &format!("s_cbranch_scc0 {}", lbl(spec, "reduce")))?;
                group(&mut b, spec, chain)?;
            }
        }
        Rotation::R128 => {
            b.wait_all()?;
            sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "reduce")))?;
            b.loop_(&lbl(spec, "group"), |b| {
                group(b, spec, 0)?;
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "group")))
            })?;
        }
    }
    b.label(&lbl(spec, "reduce"))?;
    reduce(&mut b, spec)?;
    store(&mut b, spec)?;
    b.finish()
}

/// The four entries as one code object (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Kernarg loads, the row (workgroup id), its code and scale pointers, per-lane
/// address constants and zeroed accumulators.
fn prologue(b: &mut Builder, spec: Spec) -> R {
    let a = b.spec.arch;
    smem(b, CODES, 8, KARG, 0)?;
    sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "exit")))?;
    // Row bytes K*b/8 (K % 128 == 0, so K >> 3 is exact), the codes plane M*row_bytes,
    // the row's code pointer, and the row's scale pointer
    // codes + codes_bytes + 2*row*(K/128).
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{SK}, 7"),
        &[GROUPS],
        &[SK],
    )?;
    match spec.codec {
        Codec::W3 => {
            sop(
                b,
                format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 3"),
                &[ROW_BYTES],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{ROW_BYTES}, 3"),
                &[ROW_BYTES],
                &[ROW_BYTES],
            )?;
        }
        Codec::W2 => sop(
            b,
            format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 2"),
            &[ROW_BYTES],
            &[SK],
        )?,
    }
    sop(
        b,
        format!("s_mul_i32 s{CODES_BYTES}, s{SM}, s{ROW_BYTES}"),
        &[CODES_BYTES],
        &[SM, ROW_BYTES],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
        &[TMP],
        &[ROW, ROW_BYTES],
    )?;
    add64(b, ROW_PTR, CODES, TMP)?;
    sop(
        b,
        format!("s_mul_i32 s{TMP2}, s{ROW}, s{GROUPS}"),
        &[TMP2],
        &[ROW, GROUPS],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TMP2}, s{TMP2}, 1"),
        &[TMP2],
        &[TMP2],
    )?;
    sop(
        b,
        format!("{} s{TMP2}, s{TMP2}, s{CODES_BYTES}", s_add_i32(a)),
        &[TMP2],
        &[TMP2, CODES_BYTES],
    )?;
    add64(b, SCALE_PTR, CODES, TMP2)?;
    sop(
        b,
        format!("s_lshl_b32 s{Y_OFF}, s{ROW}, 2"),
        &[Y_OFF],
        &[ROW],
    )?;
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    // Per-lane constants: x byte offset 16*l within a group; the code byte offset of the
    // lane's packet within a group ((3l) >> 1 for W3, l for W2) and the W3 packet's bit shift 4*(l & 1).
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{X_LANE}, 4, v{LANE}"),
        &[X_LANE],
        &[LANE],
        &[],
    )?;
    match spec.codec {
        Codec::W3 => {
            valu(
                b,
                &format!("v_mul_u32_u24_e32 v{CODE_LANE}, 3, v{LANE}"),
                &[CODE_LANE],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{CODE_LANE}, 1, v{CODE_LANE}"),
                &[CODE_LANE],
                &[CODE_LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_and_b32_e32 v{SHIFT}, 1, v{LANE}"),
                &[SHIFT],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{SHIFT}, 2, v{SHIFT}"),
                &[SHIFT],
                &[SHIFT],
                &[],
            )?;
        }
        Codec::W2 => valu(
            b,
            &format!("v_mov_b32_e32 v{CODE_LANE}, v{LANE}"),
            &[CODE_LANE],
            &[LANE],
            &[],
        )?,
    }
    let chains = match spec.rotation {
        Rotation::R256 => 4,
        Rotation::R128 => 1,
    };
    for c in 0..chains {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", ACC + c),
            &[ACC + c],
            &[],
            &[],
        )?;
    }
    Ok(())
}

/// One 128-column group of the row, accumulated into chain `chain` (always 0 for R128).
fn group(b: &mut Builder, spec: Spec, chain: u8) -> R {
    let arch = b.spec.arch;
    let bytes = spec.codec.group_bytes();
    sop(
        b,
        format!("s_mul_i32 s{G_CODE}, s{GI}, {bytes}"),
        &[G_CODE],
        &[GI],
    )?;
    sop(b, format!("s_lshl_b32 s{G_X}, s{GI}, 9"), &[G_X], &[GI])?;
    sop(
        b,
        format!("s_lshl_b32 s{G_SCALE}, s{GI}, 1"),
        &[G_SCALE],
        &[GI],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{CODE_ADDR}, s{G_CODE}, v{CODE_LANE}"),
        &[CODE_ADDR],
        &[CODE_LANE],
        &[G_CODE],
    )?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{SCALE_ADDR}, s{G_SCALE}"),
        &[SCALE_ADDR],
        &[],
        &[G_SCALE],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{X_ADDR}, s{G_X}, v{X_LANE}"),
        &[X_ADDR],
        &[X_LANE],
        &[G_X],
    )?;
    gload(b, "global_load_u8", PKT0, CODE_ADDR, ROW_PTR, "")?;
    if spec.codec == Codec::W3 {
        gload(b, "global_load_u8", PKT1, CODE_ADDR, ROW_PTR, " offset:1")?;
    }
    gload(b, "global_load_u16", SCALE, SCALE_ADDR, SCALE_PTR, "")?;
    let x: Vec<RegRef> = (X..X + 4).map(|n| vreg(n, 1)).collect();
    b.push(
        Instruction::new(
            format!(
                "global_load_b128 v[{X}:{}], v{X_ADDR}, s[{XP}:{}]",
                X + 3,
                XP + 1
            ),
            x,
            vec![vreg(X_ADDR, 1), sreg(XP, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )?;
    // d = float(f16 scale), exact.
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{SCALE}, v{SCALE}.l"),
        &[SCALE],
        &[SCALE],
        &[],
    )?;
    // The packet of four codes at bit 0 of PKT0.
    let (bits, levels) = (spec.codec.bits(), spec.codec.levels());
    if spec.codec == Codec::W3 {
        valu(
            b,
            &format!("v_lshl_or_b32 v{PKT0}, v{PKT1}, 8, v{PKT0}"),
            &[PKT0],
            &[PKT1, PKT0],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{PKT0}, v{SHIFT}, v{PKT0}"),
            &[PKT0],
            &[SHIFT, PKT0],
            &[],
        )?;
    }
    for i in 0..4u8 {
        let w = W + i;
        valu(
            b,
            &format!("v_bfe_u32 v{w}, v{PKT0}, {}, {bits}", u32::from(i) * bits),
            &[w],
            &[PKT0],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{w}, 1, v{w}"),
            &[w],
            &[w],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{w}, -{levels}, v{w}"),
            &[w],
            &[w],
            &[],
        )?;
        valu(b, &format!("v_cvt_f32_i32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
        valu(
            b,
            &format!("v_mul_f32_e32 v{w}, v{SCALE}, v{w}"),
            &[w],
            &[SCALE, w],
            &[],
        )?;
    }
    // p = RN(w0*x0), then three left-associated fused steps.
    valu(
        b,
        &format!("v_mul_f32_e32 v{DOT}, v{W}, v{X}"),
        &[DOT],
        &[W, X],
        &[],
    )?;
    for i in 1..4u8 {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{DOT}, v{}, v{}", W + i, X + i),
            &[DOT],
            &[DOT, W + i, X + i],
            &[],
        )?;
    }
    let acc = ACC + chain;
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{DOT}"),
        &[acc],
        &[acc, DOT],
        &[],
    )?;
    sop(
        b,
        format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
        &[GI],
        &[GI],
    )?;
    b.wait_all()
}

/// R256 fold `RN(RN(a0+a1)+RN(a2+a3))`, then the snapshot shuffle-down tree.
fn reduce(b: &mut Builder, spec: Spec) -> R {
    if spec.rotation == Rotation::R256 {
        valu(
            b,
            &format!("v_add_f32_e32 v{ACC}, v{ACC}, v{}", ACC + 1),
            &[ACC],
            &[ACC, ACC + 1],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_f32_e32 v{}, v{}, v{}", ACC + 2, ACC + 2, ACC + 3),
            &[ACC + 2],
            &[ACC + 2, ACC + 3],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_f32_e32 v{ACC}, v{ACC}, v{}", ACC + 2),
            &[ACC],
            &[ACC, ACC + 2],
            &[],
        )?;
    }
    for off in [16u32, 8, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - off),
            &[],
            &[LANE],
            &[],
        )?;
        b.push(Instruction::new(
            "s_wait_alu depctr_va_vcc(0)",
            vec![],
            vec![],
        ))?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{SHUFFLE}, 0, {off}, vcc_lo"),
            &[SHUFFLE],
            &[],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{SHUFFLE}, v{SHUFFLE}, v{LANE}, 2"),
            &[SHUFFLE],
            &[SHUFFLE, LANE],
            &[],
        )?;
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v{EXCH}, v{SHUFFLE}, v{ACC}"),
                vec![vreg(EXCH, 1)],
                vec![vreg(SHUFFLE, 1), vreg(ACC, 1)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        // sum = acc + acc[lane + off]; lanes with lane + off > 31 (vcc clear) keep acc unchanged.
        valu(
            b,
            &format!("v_add_f32_e32 v{EXCH}, v{ACC}, v{EXCH}"),
            &[EXCH],
            &[ACC, EXCH],
            &[],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{ACC}, v{ACC}, v{EXCH}, vcc_lo"),
            &[ACC],
            &[ACC, EXCH],
            &[],
        )?;
    }
    Ok(())
}

/// Lane 0 stores `y[row]`.
fn store(b: &mut Builder, spec: Spec) -> R {
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{STORE_ADDR}, s{Y_OFF}"),
        &[STORE_ADDR],
        &[],
        &[Y_OFF],
    )?;
    b.push(
        Instruction::new(
            format!("global_store_b32 v{STORE_ADDR}, v{ACC}, s[{YP}:{}]", YP + 1),
            vec![],
            vec![vreg(STORE_ADDR, 1), vreg(ACC, 1), sreg(YP, 2)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.wait_all()?;
    b.label(&lbl(spec, "exit"))?;
    branch(b, "s_endpgm")
}

/// The shape contract of one emitted entry: counts of its characteristic
/// instructions, the forbidden families, register ceilings, wave32, no spills,
/// no private or LDS memory. Committed as
/// `kernels/mq_i4_rows.gfx1201.<tag>.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    serde_json::json!({
        "symbol": spec.symbol(), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "buffer_*", "ds_load*", "ds_store*", "ds_swizzle*", "ds_read*", "ds_write*", "s_barrier*",
                      "v_readlane*", "v_writelane*", "v_wmma*", "v_swmmac*", "v_fma_mix*", "s_waitcnt*", "s_setpc*", "s_swappc*"],
        "vgpr_max": 32, "sgpr_max": 64,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": 0,
    })
}
const COUNTED: [&str; 17] = [
    "s_load_b256",
    "global_load_u8",
    "global_load_u16",
    "global_load_b128",
    "global_store_b32",
    "ds_bpermute_b32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_bfe_u32",
    "v_lshl_or_b32",
    "v_mul_f32_e32",
    "v_fmac_f32_e32",
    "v_add_f32_e32",
    "v_cmpx_eq_u32_e32",
    "v_cndmask_b32_e64",
    "v_add_lshl_u32",
    "s_endpgm",
];

fn vreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len,
    }
}
fn sreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base,
        len,
    }
}
fn refs(kind: Kind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
/// `mnemonic v{dst}, v{voff}, s[{base}:{base+1}]{suffix}` of one dword (or sub-dword) load.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8, suffix: &str) -> R {
    b.push(
        Instruction::new(
            format!("{mnemonic} v{dst}, v{voff}, s[{base}:{}]{suffix}", base + 1),
            vec![vreg(dst, 1)],
            vec![vreg(voff, 1), sreg(base, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
