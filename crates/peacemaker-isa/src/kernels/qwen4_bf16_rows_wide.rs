// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exact fixed-eight rowtile twins of the ordinary BF16 GEMV
//! (`gemv_bf16_xf32_x4`, its row extension `gemv_bf16_xf32_x4_rows_r<R>`) and
//! the HC k4 projection: `gemv_bf16_xf32_rows_x4_t8_pm_<arch>` and
//! `gemv_bf16_xf32_k4_rows_t8_pm_<arch>`, module `bf16_rows_wide_pm_<arch>`.
//! Native gfx1151 only: the Halo singleton DAG transcribed here; any other
//! `arch` is refused. Total logical rows `R = 1..64` in the `R` kernarg; grid
//! y selects one independent eight-row tile.
//!
//! Ordinary S1 schedule: one output per wave (block32, no LDS); HC keeps four
//! fixed quarter owners (block128, 128-byte partial LDS). The per-row
//! arithmetic is the singleton DAG: `v_fmac_f32 acc, X, W` chains
//! (`a[j & 3]`), `(a1+a0)+(a3+a2)` fold, 16/8/4/2/1 swizzle/bpermute
//! own+other tree, lane-0 stores; HC appends the exact scalar
//! `(q0+q1)+(q2+q3)` fold with BF16 round trips, scale and `v / (1+exp(-v))`
//! from `qwen4_bf16_hc_epilogue`, exp/divide emitted exactly. K domain: K256
//! vector groups for `K % 8 == 0`, K32 all-scalar domain otherwise; K % 32
//! != 0 HC exits uniformly. See the `rowtiles` module.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, sr, v, SRD_WORD3,
};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size (the narrow MQ4 entry's).
pub const KERNARG_BYTES: u32 = 96;
/// HC k4 kernarg segment: 44 logical bytes padded to eight-byte alignment.
pub const HC_KERNARG_BYTES: u32 = 48;
/// Physical activation-row tile; independent of the logical window width.
pub const ROW_TILE: u8 = 8;
/// Maximum logical row count of the rowtile entries.
pub const MAX_ROWS: u8 = 64;
/// The module `bf16_rows_wide_pm_<arch>` (frozen name; the source kernel id stays `qwen4_bf16_rows_wide`).
pub fn module(arch: Arch) -> String {
    format!("bf16_rows_wide_pm_{}", arch.name())
}

// ----------------------------------------------------------------------- SGPRs
const KARG: u8 = 0;
const WGIN: u8 = 2;
const SRD: u8 = 4;
const PA: u8 = 8;
const PXY: u8 = 16;
const Y3M: u8 = 24;
const M2KB: u8 = 28;
const XP: u8 = 16;
const Y0: u8 = 18;
const M0: u8 = 26;
const M1: u8 = 27;
const M2: u8 = 28;
const M3: u8 = 29;
const SK: u8 = 30;
const SR: u8 = 31;
const SY: u8 = 32;
const SM: u8 = 34;
const TMP: u8 = 45;
/// Weight row byte offset (64-bit: low / high), added to the descriptor base.
const WRL: u8 = 46;
const WRH: u8 = 47;
/// Active-lane EXEC mask of the current tile, and the K remaining at its start.
const EM: u8 = 48;
const KREM: u8 = 49;
const GOFF: u8 = 50;
const XG: u8 = 51;
const GI: u8 = 52;
const KB: u8 = 53;
const MB: u8 = 54;
const STMP: u8 = 55;
/// `K & 7`: nonzero selects the scalar domain.
const SPATH: u8 = 58;

// ----------------------------------------------------------------------- VGPRs
const TID: u8 = 0;
const HOFF: u8 = 1;
const XOFF: u8 = 3;
const SHUF: u8 = 4;
const LANE: u8 = 5;
/// LDS producer store address and the consumer read base.
const LDST: u8 = 9;
const VT: u8 = 13;
/// Widened weight slots W0..W7.
const WS: u8 = 16;
/// X slots X0..X7 (the producer's staging and the consumer's row vector).
const XS: u8 = 24;
const ACC0: u16 = 40;

fn acc(row: u32, stream: u32) -> u8 {
    let n = usize::from(ACC0) + 4 * row as usize + stream as usize;
    u8::try_from(n).expect("accumulator index validated by the rowtile plan")
}

fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}

/// The module `bf16_rows_wide_pm_<arch>`: the actual two rowtile entries,
/// ordinary x4 then HC k4. The shared module combiner keeps exactly one
/// end-of-text guard (`emit::CODE_END_TAIL`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit_rowtiles(arch, false)?, emit_rowtiles(arch, true)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

/// Emit the fixed8, total-R rowtile twin for the Halo singleton DAG.
///
/// Ordinary single projections use the x4 ABI with the other Mi set to zero.
/// This deliberately refuses other architectures: their singleton objects are
/// not interchangeable with the Halo object transcribed here.
pub fn emit_rowtiles(arch: Arch, hc: bool) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 {
        return Err("BF16 Halo rowtiles require the gfx1151 singleton DAG".into());
    }
    rowtiles::emit(arch, hc, false)
}

/// Lab-only HC observation, never included in the production module.
/// Same 48-byte HC ABI. Y owns five planes of 64*M f32 words: final in
/// plane 0, raw fused quarter q in plane q+1, at row*M+column.
/// Only active rows are written; the caller must provide the expanded owner.
pub fn emit_hc_quarter_diagnostic(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 {
        return Err("BF16 Halo rowtiles require the gfx1151 singleton DAG".into());
    }
    rowtiles::emit(arch, true, true)
}

fn exec_set(b: &mut Builder) -> R {
    op(b, format!("s_mov_b32 exec_lo, s{EM}"), &[], &[s(EM)])
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
fn exec_lane0(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}

/// bf16 -> f32 is bitwise: low half `<< 16`, high half `& 0xffff0000`. The
/// dwords sit in `WS..WS+3` (one `buffer_load_b128`); widen in place, dword 3
/// first, so no dword is overwritten before it is read.
fn widen_weights(b: &mut Builder) -> R {
    let w = |i: u8| WS + i;
    for d in (0..4u8).rev() {
        valu(
            b,
            &format!("v_and_b32_e32 v{}, 0xffff0000, v{}", w(2 * d + 1), w(d)),
            &[w(2 * d + 1)],
            &[w(d)],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 16, v{}", w(2 * d), w(d)),
            &[w(2 * d)],
            &[w(d)],
            &[],
        )?;
    }
    Ok(())
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
pub(super) fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
pub(super) fn vmem(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
    store: bool,
) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
pub(super) fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}

mod rowtiles {
    use super::*;
    use crate::kernels::qwen4_bf16_hc_epilogue::{self as epi, Regs};

    const COUNT: u8 = 40;
    const BASE: u8 = 41;
    const PART: u8 = 42;
    const QUARTER: u8 = 43;
    const SCALE: u8 = 44;
    const ACT: u8 = 61;
    const QX: u8 = 57;
    const NG: u8 = 59;
    const COL: u8 = 60;

    fn label(hc: bool, name: &str) -> String {
        format!(".Lbf16t8_{}_{name}", if hc { "hc" } else { "x4" })
    }

    fn jump(b: &mut Builder, hc: bool, cond: &str, name: &str) -> R {
        branch(b, &format!("s_{cond} {}", label(hc, name)))
    }

    fn mov(b: &mut Builder, dst: u8, src: u8) -> R {
        sop(b, format!("s_mov_b32 s{dst}, s{src}"), &[dst], &[src])
    }

    pub(super) fn emit(arch: Arch, hc: bool, diagnostic: bool) -> Result<Emitted, String> {
        let args = if hc {
            KernargLayout::new(HC_KERNARG_BYTES)
                .pointer_access("W", 0, Access::ReadOnly)
                .pointer_access("X", 8, Access::ReadOnly)
                .pointer_access("Y", 16, Access::WriteOnly)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value")
                .hidden("scale", 32, 4, "by_value")
                .hidden("hc_act", 36, 4, "by_value")
                .hidden("R", 40, 4, "by_value")
        } else {
            KernargLayout::new(KERNARG_BYTES)
                .pointer_access("A0", 0, Access::ReadOnly)
                .pointer_access("A1", 8, Access::ReadOnly)
                .pointer_access("A2", 16, Access::ReadOnly)
                .pointer_access("A3", 24, Access::ReadOnly)
                .pointer_access("x", 32, Access::ReadOnly)
                .pointer_access("y0", 40, Access::WriteOnly)
                .pointer_access("y1", 48, Access::WriteOnly)
                .pointer_access("y2", 56, Access::WriteOnly)
                .pointer_access("y3", 64, Access::WriteOnly)
                .hidden("M0", 72, 4, "by_value")
                .hidden("M1", 76, 4, "by_value")
                .hidden("M2", 80, 4, "by_value")
                .hidden("M3", 84, 4, "by_value")
                .hidden("K", 88, 4, "by_value")
                .hidden("R", 92, 4, "by_value")
        };
        let mut regs = RegPlan::new(80, 80)?;
        // s3 is the enabled workgroup-y input; s64..79 belong to the
        // unchanged HC expf/division expansion. Eight rows use v40..71.
        for base in (0..80u8).step_by(8) {
            regs.add_range(&format!("s{base}"), Kind::S, base, 8, Live::Whole)?;
            regs.add_range(&format!("v{base}"), Kind::V, base, 8, Live::Whole)?;
        }
        let mut b = Builder::new(
            KernelSpec {
                kernel_id: "qwen4_bf16_rows_wide".into(),
                variant: if hc { "t8_hc_k4_s1" } else { "t8_x4_s1" }.into(),
                arch,
                symbol: if diagnostic {
                    format!("gemv_bf16_xf32_k4_rows_t8_quarters_pm_{}", arch.name())
                } else {
                    format!(
                        "gemv_bf16_xf32_{}_t8_pm_{}",
                        if hc { "k4_rows" } else { "rows_x4" },
                        arch.name()
                    )
                },
                kernargs: args,
                user_sgpr_count: 2,
                system_sgpr_workgroup_id_y: true,
                workgroup_size: if hc { 128 } else { 32 },
                group_segment_fixed_size: if hc { 128 } else { 0 },
                wave32: true,
                cu_mode: false,
            },
            regs,
        );
        b.enable_delay_alu();
        let partials = if hc {
            Some(b.lds_slot("quarters", 0, 128)?)
        } else {
            None
        };
        prologue(&mut b, hc)?;
        // Inactive rows do not execute any floating-point instruction.
        for r in 0..8 {
            sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
            jump(&mut b, hc, "cbranch_scc1", "initialized")?;
            for st in 0..4 {
                let a = acc(r, st);
                valu(&mut b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
        b.label(&label(hc, "initialized"))?;
        sop(&mut b, format!("s_cmp_lg_u32 s{SPATH}, 0"), &[], &[SPATH])?;
        jump(&mut b, hc, "cbranch_scc1", "scalar")?;
        sweep(&mut b, hc, false)?;
        jump(&mut b, hc, "branch", "reduce")?;
        b.label(&label(hc, "scalar"))?;
        sweep(&mut b, hc, true)?;
        b.label(&label(hc, "reduce"))?;
        for r in 0..8 {
            sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
            jump(&mut b, hc, "cbranch_scc1", "reduced")?;
            reduce(&mut b, r)?;
            if let Some(slot) = partials {
                sop(
                    &mut b,
                    format!("s_lshl_b32 s{TMP}, s{QUARTER}, 2"),
                    &[TMP],
                    &[QUARTER],
                )?;
                valu(
                    &mut b,
                    &format!("v_mov_b32_e32 v{LDST}, s{TMP}"),
                    &[LDST],
                    &[],
                    &[TMP],
                )?;
                exec_lane0(&mut b)?;
                b.ds_store(
                    slot,
                    Instruction::new(
                        format!("ds_store_b32 v{LDST}, v{}{}", acc(r, 3), off(16 * r)),
                        vec![],
                        vec![v(LDST), v(acc(r, 3))],
                    )
                    .memory(MemoryClass::DsStore),
                )?;
                b.wait_all()?;
                exec_all(&mut b)?;
            } else {
                store(&mut b, hc, r, acc(r, 3))?;
            }
        }
        b.label(&label(hc, "reduced"))?;
        if let Some(slot) = partials {
            b.barrier(&[Transition::Ready(slot)])?;
            // Only quarter zero consumes the partials; all four owners have
            // reached the one barrier, and there is no later LDS overwrite.
            sop(
                &mut b,
                format!("s_cmp_lg_u32 s{QUARTER}, 0"),
                &[],
                &[QUARTER],
            )?;
            jump(&mut b, hc, "cbranch_scc1", "end")?;
            let mut st = [0u8; 11];
            for (i, x) in st.iter_mut().enumerate() {
                *x = 69 + i as u8;
            }
            let mut vt = [0u8; 16];
            for (i, x) in vt.iter_mut().enumerate() {
                *x = XS + i as u8;
            }
            let er = Regs {
                q: [64, 65, 66, 67],
                scale: SCALE,
                act: ACT,
                out: 68,
                s_tmp: st,
                v_tmp: vt,
            };
            for r in 0..8 {
                sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
                jump(&mut b, hc, "cbranch_scc1", "end")?;
                valu(
                    &mut b,
                    &format!("v_mov_b32_e32 v{LDST}, {}", lit(16 * r)),
                    &[LDST],
                    &[],
                    &[],
                )?;
                b.ds_load(
                    slot,
                    Instruction::new(
                        format!("ds_load_b128 v[{WS}:{}], v{LDST}", WS + 3),
                        (0..4).map(|i| v(WS + i)).collect(),
                        vec![v(LDST)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
                for i in 0..4 {
                    op(
                        &mut b,
                        format!("v_readfirstlane_b32 s{}, v{}", 64 + i, WS + i),
                        &[s(64 + i)],
                        &[v(WS + i)],
                    )?;
                }
                if diagnostic {
                    for q in 0..4 {
                        store_plane(&mut b, r, WS + q, u32::from(q) + 1)?;
                    }
                }
                epi::emit_value(&mut b, &label(hc, &format!("act{r}")), &er)?;
                valu(
                    &mut b,
                    &format!("v_mov_b32_e32 v{XS}, s68"),
                    &[XS],
                    &[],
                    &[68],
                )?;
                store(&mut b, hc, r, XS)?;
            }
        }
        b.label(&label(hc, "end"))?;
        branch(&mut b, "s_endpgm")?;
        b.finish()
    }

    fn prologue(b: &mut Builder, hc: bool) -> R {
        if hc {
            smem(b, 8, 8, KARG, 0)?;
            smem(b, 16, 4, KARG, 32)?;
            b.wait_all()?;
            for (d, x) in [
                (SCALE, 16),
                (ACT, 17),
                (SR, 18),
                (XP, 10),
                (XP + 1, 11),
                (SY, 12),
                (SY + 1, 13),
                (SM, 14),
                (SK, 15),
            ] {
                mov(b, d, x)?;
            }
            mov(b, COL, WGIN)?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
                &[VT],
                &[TID],
                &[],
            )?;
            op(
                b,
                format!("v_readfirstlane_b32 s{QUARTER}, v{VT}"),
                &[s(QUARTER)],
                &[v(VT)],
            )?;
            mov(b, SRD, 8)?;
            mov(b, SRD + 1, 9)?;
        } else {
            smem(b, PA, 8, KARG, 0)?;
            smem(b, PXY, 8, KARG, 32)?;
            smem(b, Y3M, 4, KARG, 64)?;
            smem(b, M2KB, 4, KARG, 80)?;
            b.wait_all()?;
            // S1 grid-x is sum Mi, with no padding between matrices.
            mov(b, COL, WGIN)?;
            for i in 0..4u8 {
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{COL}, s{}", M0 + i),
                    &[],
                    &[COL, M0 + i],
                )?;
                jump(b, hc, "cbranch_scc1", &format!("matrix{i}"))?;
                sop(
                    b,
                    format!("s_sub_u32 s{COL}, s{COL}, s{}", M0 + i),
                    &[COL],
                    &[COL, M0 + i],
                )?;
            }
            jump(b, hc, "branch", "end")?;
            for i in 0..4u8 {
                b.label(&label(hc, &format!("matrix{i}")))?;
                for (d, x) in [
                    (SRD, PA + 2 * i),
                    (SRD + 1, PA + 2 * i + 1),
                    (SY, Y0 + 2 * i),
                    (SY + 1, Y0 + 2 * i + 1),
                    (SM, M0 + i),
                ] {
                    mov(b, d, x)?;
                }
                jump(b, hc, "branch", "selected")?;
            }
            b.label(&label(hc, "selected"))?;
            sop(b, format!("s_mov_b32 s{QUARTER}, 0"), &[QUARTER], &[])?;
        }
        sop(b, format!("s_cmp_ge_u32 s{COL}, s{SM}"), &[], &[COL, SM])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        sop(b, format!("s_cmp_gt_u32 s{SR}, 64"), &[], &[SR])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        sop(b, format!("s_cmp_le_i32 s{SK}, 0"), &[], &[SK])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        sop(b, format!("s_lshl_b32 s{BASE}, s3, 3"), &[BASE], &[3])?;
        sop(b, format!("s_cmp_ge_u32 s{BASE}, s{SR}"), &[], &[BASE, SR])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        // Single-definition clamp: COUNT's ONE def (the min) dominates every
        // `s_cmp_le_u32 COUNT, r` user (cf. lane_rows `rowtile_domain`); TMP is
        // dead here on both paths and redefined before any later read.
        sop(
            b,
            format!("s_sub_u32 s{TMP}, s{SR}, s{BASE}"),
            &[TMP],
            &[SR, BASE],
        )?;
        sop(
            b,
            format!("s_min_u32 s{COUNT}, s{TMP}, 8"),
            &[COUNT],
            &[TMP],
        )?;
        if hc {
            sop(b, format!("s_and_b32 s{TMP}, s{SK}, 31"), &[TMP], &[SK])?;
            sop(b, format!("s_cmp_lg_u32 s{TMP}, 0"), &[], &[TMP])?;
            jump(b, hc, "cbranch_scc1", "end")?;
            sop(b, format!("s_lshr_b32 s{PART}, s{SK}, 2"), &[PART], &[SK])?;
        } else {
            mov(b, PART, SK)?;
        }
        sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
        sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{SK}, 1"), &[TMP], &[SK])?;
        sop(
            b,
            format!("s_mul_i32 s{WRL}, s{COL}, s{TMP}"),
            &[WRL],
            &[COL, TMP],
        )?;
        sop(
            b,
            format!("s_mul_hi_u32 s{WRH}, s{COL}, s{TMP}"),
            &[WRH],
            &[COL, TMP],
        )?;
        sop(
            b,
            format!("{} s{SRD}, s{SRD}, s{WRL}", s_add_u32(b.spec.arch)),
            &[SRD],
            &[SRD, WRL],
        )?;
        sop(
            b,
            format!(
                "{} s{}, s{}, s{WRH}",
                s_addc_u32(b.spec.arch),
                SRD + 1,
                SRD + 1
            ),
            &[SRD + 1],
            &[SRD + 1, WRH],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{QX}, s{QUARTER}, s{PART}"),
            &[QX],
            &[QUARTER, PART],
        )?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{QX}, 1"), &[TMP], &[QX])?;
        sop(
            b,
            format!("{} s{SRD}, s{SRD}, s{TMP}", s_add_u32(b.spec.arch)),
            &[SRD],
            &[SRD, TMP],
        )?;
        sop(
            b,
            format!("{} s{}, s{}, 0", s_addc_u32(b.spec.arch), SRD + 1, SRD + 1),
            &[SRD + 1],
            &[SRD + 1],
        )?;
        sop(b, format!("s_lshl_b32 s{QX}, s{QX}, 2"), &[QX], &[QX])?;
        sop(
            b,
            format!("s_and_b32 s{}, s{}, 0xffff", SRD + 1, SRD + 1),
            &[SRD + 1],
            &[SRD + 1],
        )?;
        sop(b, format!("s_mov_b32 s{}, -1", SRD + 2), &[SRD + 2], &[])?;
        sop(
            b,
            format!("s_mov_b32 s{}, {}", SRD + 3, lit(SRD_WORD3)),
            &[SRD + 3],
            &[],
        )?;
        sop(
            b,
            format!("s_and_b32 s{SPATH}, s{PART}, 7"),
            &[SPATH],
            &[PART],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
            &[LANE],
            &[TID],
            &[],
        )
    }

    fn sweep(b: &mut Builder, hc: bool, scalar: bool) -> R {
        let (shift, size, n) = if scalar { (5, 32, 1) } else { (8, 256, 8) };
        let tag = if scalar { "scalar" } else { "fast" };
        sop(
            b,
            format!(
                "{} s{NG}, s{PART}, {}",
                s_add_i32(b.spec.arch),
                lit(size - 1)
            ),
            &[NG],
            &[PART],
        )?;
        sop(b, format!("s_lshr_b32 s{NG}, s{NG}, {shift}"), &[NG], &[NG])?;
        sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
        b.loop_(&label(hc, &format!("{tag}_loop")), |b| {
            sop(
                b,
                format!("s_lshl_b32 s{TMP}, s{GI}, {shift}"),
                &[TMP],
                &[GI],
            )?;
            sop(
                b,
                format!("s_sub_u32 s{KREM}, s{PART}, s{TMP}"),
                &[KREM],
                &[PART, TMP],
            )?;
            if scalar {
                mov(b, EM, KREM)?;
            } else {
                sop(b, format!("s_lshr_b32 s{EM}, s{KREM}, 3"), &[EM], &[KREM])?;
            }
            sop(b, format!("s_lshl_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
            sop(b, format!("s_not_b32 s{EM}, s{EM}"), &[EM], &[EM])?;
            sop(
                b,
                format!("s_cmp_ge_u32 s{KREM}, {}", lit(size)),
                &[],
                &[KREM],
            )?;
            sop(b, format!("s_cselect_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
            sop(b, format!("s_lshl_b32 s{GOFF}, s{TMP}, 1"), &[GOFF], &[TMP])?;
            sop(b, format!("s_lshl_b32 s{XG}, s{TMP}, 2"), &[XG], &[TMP])?;
            valu(
                b,
                &format!(
                    "v_lshlrev_b32_e32 v{HOFF}, {}, v{LANE}",
                    if scalar { 1 } else { 4 }
                ),
                &[HOFF],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
                &[HOFF],
                &[HOFF],
                &[GOFF],
            )?;
            exec_set(b)?;
            let load = if scalar {
                format!(
                    "buffer_load_u16 v{WS}, v{HOFF}, s[{SRD}:{}], 0 offen",
                    SRD + 3
                )
            } else {
                format!(
                    "buffer_load_b128 v[{WS}:{}], v{HOFF}, s[{SRD}:{}], 0 offen",
                    WS + 3,
                    SRD + 3
                )
            };
            let defs: Vec<u8> = (0..if scalar { 1 } else { 4 }).map(|i| WS + i).collect();
            vmem(
                b,
                &load,
                &defs,
                &[HOFF],
                &[SRD, SRD + 1, SRD + 2, SRD + 3],
                false,
            )?;
            b.wait_all()?;
            if scalar {
                valu(
                    b,
                    &format!("v_lshlrev_b32_e32 v{WS}, 16, v{WS}"),
                    &[WS],
                    &[WS],
                    &[],
                )?;
            } else {
                widen_weights(b)?;
            }
            exec_all(b)?;
            for r in 0..8 {
                sop(b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
                jump(b, hc, "cbranch_scc1", &format!("{tag}_advanced"))?;
                sop(
                    b,
                    format!("{} s{TMP}, s{BASE}, {r}", s_add_i32(b.spec.arch)),
                    &[TMP],
                    &[BASE],
                )?;
                sop(
                    b,
                    format!("s_mul_i32 s{TMP}, s{TMP}, s{KB}"),
                    &[TMP],
                    &[TMP, KB],
                )?;
                sop(
                    b,
                    format!("{} s{TMP}, s{TMP}, s{QX}", s_add_i32(b.spec.arch)),
                    &[TMP],
                    &[TMP, QX],
                )?;
                sop(
                    b,
                    format!("{} s{TMP}, s{TMP}, s{XG}", s_add_i32(b.spec.arch)),
                    &[TMP],
                    &[TMP, XG],
                )?;
                valu(
                    b,
                    &format!(
                        "v_lshlrev_b32_e32 v{XOFF}, {}, v{LANE}",
                        if scalar { 2 } else { 5 }
                    ),
                    &[XOFF],
                    &[LANE],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_add_nc_u32_e32 v{XOFF}, s{TMP}, v{XOFF}"),
                    &[XOFF],
                    &[XOFF],
                    &[TMP],
                )?;
                exec_set(b)?;
                for j in 0..n {
                    vmem(
                        b,
                        &format!(
                            "global_load_b32 v{}, v{XOFF}, s[{XP}:{}]{}",
                            XS + j as u8,
                            XP + 1,
                            off(4 * j)
                        ),
                        &[XS + j as u8],
                        &[XOFF],
                        &[XP, XP + 1],
                        false,
                    )?;
                }
                b.wait_all()?;
                for j in 0..n {
                    let a = acc(r, j & 3);
                    valu(
                        b,
                        &format!("v_fmac_f32_e32 v{a}, v{}, v{}", XS + j as u8, WS + j as u8),
                        &[a],
                        &[a, XS + j as u8, WS + j as u8],
                        &[],
                    )?;
                }
                exec_all(b)?;
            }
            b.label(&label(hc, &format!("{tag}_advanced")))?;
            sop(
                b,
                format!("{} s{GI}, s{GI}, 1", s_add_i32(b.spec.arch)),
                &[GI],
                &[GI],
            )?;
            sop(b, format!("s_cmp_lt_u32 s{GI}, s{NG}"), &[], &[GI, NG])?;
            jump(b, hc, "cbranch_scc1", &format!("{tag}_loop"))
        })
    }

    fn reduce(b: &mut Builder, row: u32) -> R {
        for (d, x) in [(1, 0), (3, 2), (3, 1)] {
            let (d, x) = (acc(row, d), acc(row, x));
            valu(
                b,
                &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
                &[d],
                &[d, x],
                &[],
            )?;
        }
        let a = acc(row, 3);
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v{XS}, v{a} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                vec![v(XS)],
                vec![v(a)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        valu(
            b,
            &format!("v_add_f32_e32 v{a}, v{a}, v{XS}"),
            &[a],
            &[a, XS],
            &[],
        )?;
        for d in [8, 4, 2, 1] {
            valu(
                b,
                &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - d),
                &[],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_cndmask_b32_e64 v{SHUF}, 0, {d}, vcc_lo"),
                &[SHUF],
                &[],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"),
                &[SHUF],
                &[SHUF, LANE],
                &[],
            )?;
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{XS}, v{SHUF}, v{a}"),
                    vec![v(XS)],
                    vec![v(SHUF), v(a)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{a}, v{a}, v{XS}"),
                &[a],
                &[a, XS],
                &[],
            )?;
        }
        Ok(())
    }

    fn store(b: &mut Builder, _hc: bool, r: u32, val: u8) -> R {
        store_plane(b, r, val, 0)
    }

    fn store_plane(b: &mut Builder, r: u32, val: u8, plane: u32) -> R {
        let row_offset = r + plane * 64;
        if row_offset > 64 {
            // Diagnostic planes exceed the signed-add inline-constant range.
            // STMP is dead here and is redefined for the column below.
            sop(
                b,
                format!("s_mov_b32 s{STMP}, {}", lit(row_offset)),
                &[STMP],
                &[],
            )?;
            sop(
                b,
                format!("{} s{TMP}, s{BASE}, s{STMP}", s_add_i32(b.spec.arch)),
                &[TMP],
                &[BASE, STMP],
            )?;
        } else {
            sop(
                b,
                format!("{} s{TMP}, s{BASE}, {}", s_add_i32(b.spec.arch), row_offset),
                &[TMP],
                &[BASE],
            )?;
        }
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{TMP}, s{MB}"),
            &[TMP],
            &[TMP, MB],
        )?;
        sop(b, format!("s_lshl_b32 s{STMP}, s{COL}, 2"), &[STMP], &[COL])?;
        sop(
            b,
            format!("{} s{TMP}, s{TMP}, s{STMP}", s_add_i32(b.spec.arch)),
            &[TMP],
            &[TMP, STMP],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{XOFF}, s{TMP}"),
            &[XOFF],
            &[],
            &[TMP],
        )?;
        exec_lane0(b)?;
        vmem(
            b,
            &format!("global_store_b32 v{XOFF}, v{val}, s[{SY}:{}]", SY + 1),
            &[],
            &[XOFF, val],
            &[SY, SY + 1],
            true,
        )?;
        b.wait_all()?;
        exec_all(b)
    }
}
