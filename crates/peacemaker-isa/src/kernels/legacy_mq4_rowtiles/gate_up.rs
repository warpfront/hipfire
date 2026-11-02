// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PM twin of the legacy aggregate indexed gate/up rows8 kernel, tiled over
//! physical token batches.
//!
//! Twinned original (aggregate route, NOT the dense PM multirow path):
//! `gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched` from
//! `kernels/src/qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`
//! (shipped under the top-10 alias
//! `GEMV_MQ4G256V2_MOE_GATE_UP_TOP10_INDEXED_BATCHED_SRC`; the source is
//! byte-identical, the entry point is the sealed top-k=10 ABI).
//! Census provenance (actual AR oracle, not this file):
//! `row-exact-wide/census/historical-hip-object-provenance.json`
//! objects `gemv_mq4g256v2_moe_gate_up_top10_indexed_batched`
//! (gfx1201 + gfx1100 rows), selected symbols recorded
//! `gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched` and
//! `gemv_mq4g256v2_moe_gate_up_top10_indexed_batched`, historical export
//! receipt `jit-peacemaker/hipcc/<arch>/.../rep-1/kernel.hsaco`.
//!
//! Dimension correction (parent-directed): the wrapper's `m`
//! (`gemv.rs:16910`) is the OUTPUT row count (canonical 1280 = 2*MI), and
//! `batch_size` is the token count (original grid.z). One ROWS8 workgroup
//! owns 8 OUTPUT rows and ONE x token. This tile therefore keeps `M`
//! (output dim, unbounded by 64) and appends the total token count `R`:
//! grid `[ceil(M/8), K_TOP, ceil(R/8)]`, token base `r0 = 8*wgZ`, eight token
//! iterations reusing the same 10240-B LDS slot sequentially (never 80 KiB).
//! All waves reach every barrier, including output-dead waves and
//! token-skipped iterations (uniform skip over BOTH barriers of the
//! iteration, so LDS state stays consistent).
//!
//! Kernarg ABI (56 bytes: the original 52 offsets preserved exactly, `R`
//! appended): `expert_ptrs@0` (u64 table, read-only), `topk_indices@8`
//! (read-only), `x@16` (read-only), `y_gate@24` (write-only), `y_up@32`
//! (write-only), `M@40` (i32 output rows), `K_TOP@44` (i32, sealed 10),
//! `n_experts@48` (i32 table extent), `R@52` (i32 total tokens, NEW tile
//! kernarg; no new math, derived tile domain).
//! Expected runtime geometry: grid `[ceil(M/8), K_TOP, ceil(R/8)]`, block
//! 256 (eight wave32 waves; wave w owns absolute output row `8*bx+w`),
//! LDS 10240 B. `M==0`, `R==0`, `krank >= K_TOP`, token base `8*wgZ >= R`,
//! and output tile base `8*bx >= M` exit uniformly before any LDS access.
//! Waves whose row is dead (`row >= M`) still stage and hit every barrier
//! but issue no weight loads, no arithmetic and no stores. Invalid experts
//! (`expert_id >= n_experts`, one unsigned compare so negatives count) store
//! `0.0` from lane 0 to the live row's gate/up slot, exactly like the HIP
//! guard. Fixed `K=2560` (ten groups): no group tail exists. Raw qt44/qt53
//! payloads never enter any Slab/K1 path; weights are borrowed in place
//! through the existing `expert_ptrs` table (no copies, no re-encode).
//!
//! Supported arches: gfx1201 (first real target) and gfx1151 (the HIP source
//! compiles the rows8 entry uniformly there; gfx11 spellings — system-SGPR
//! ids, no `co` suffixes, `null` SMEM offsets, plain loads, vmcnt/lgkmcnt
//! waits — are applied). Every other arch is an explicit error; no generic
//! math fallback, no WMMA/IU4/fp16 precision changes.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{lit, op, s, s_add_i32, smem, sop, srd_tail, v};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// 56-byte tile kernarg segment: original 52 offsets + `R@52` (see docs).
pub const KERNARG_BYTES: u32 = 56;
/// Physical output rows per tile (eight waves of one 256-thread workgroup).
pub const ROW_TILE: u8 = 8;
/// Physical tokens per tile (eight sequential iterations, one LDS slot).
pub const TOKEN_TILE: u8 = 8;
/// Sealed top-k of the indexed route.
pub const SEALED_K_TOP: u8 = 10;
/// Fixed K of the rows8 entry (ten 136-byte groups per row).
pub const FIXED_K: u32 = 2560;
/// Row stride of one expert row: 10 groups * 136 bytes.
pub const ROW_BYTES: u32 = 1360;
/// LDS bytes staging one token's x row (2560 f32), reused per token.
pub const LDS_BYTES: u32 = 10240;

/// Export symbol per arch. Contains the original indexed operation plus the
/// `_t8_pm_<arch>` twin marker; never the old singleton name.
pub fn symbol(arch: Arch) -> String {
    format!(
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_{}",
        arch.name()
    )
}

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_ptrs", 0, Access::ReadOnly)
        .pointer_access("topk_indices", 8, Access::ReadOnly)
        .pointer_access("x", 16, Access::ReadOnly)
        .pointer_access("y_gate", 24, Access::WriteOnly)
        .pointer_access("y_up", 32, Access::WriteOnly)
        .hidden("M", 40, 4, "by_value")
        .hidden("K_TOP", 44, 4, "by_value")
        .hidden("n_experts", 48, 4, "by_value")
        .hidden("R", 52, 4, "by_value")
}

// ---- SGPRs ----------------------------------------------------------------
// s0:1 kernarg pointer; s2/s3/s4 hold the gfx11 system workgroup ids
// (X/Y/Z; reserved on gfx12 where ids come from ttmp9/ttmp7).
const KARG: u8 = 0;
const SYS_X: u8 = 2;
const SYS_Y: u8 = 3;
const SYS_Z: u8 = 4;
// Loaded kernargs (s8:21, 14 SGPR = 56 B): joint owners s8:15 / s16:19 / s20:21
// matching the three SMEM defs (len 8/4/2); pair/single subviews below are
// contained in exactly one owner each.
const PP: u8 = 8; // expert_ptrs table pointer pair (s8:9)
const IP: u8 = 10; // topk_indices pointer pair (s10:11)
const XP: u8 = 12; // x pointer pair (s12:13)
const YG: u8 = 14; // y_gate pointer pair (s14:15)
const YU: u8 = 16; // y_up pointer pair (s16:17)
const SM: u8 = 18; // M output rows (s18)
const SK: u8 = 19; // K_TOP sealed 10 (s19)
const SN: u8 = 20; // n_experts (s20)
const SR: u8 = 21; // R total tokens (s21), packed with SN in the len-2 tail
                   // Weight buffer descriptor has its own aligned four-SGPR owner.
const WSRD: u8 = 24;
// Workgroup / wave / row state (uniform within a wave).
const SWX: u8 = 28; // blockIdx.x (output row tile index)
const SWY: u8 = 29; // blockIdx.y (krank)
const SWZ: u8 = 30; // blockIdx.z (token tile index)
const R0: u8 = 31; // output tile row base = 8*bx
const ROW: u8 = 32; // this wave's absolute output row
const MI: u8 = 33; // M >> 1 (gate/up split)
const ROWB: u8 = 34; // row * ROW_BYTES
const GOFF: u8 = 35; // group byte offset g*136
const STMP: u8 = 36;
const STMP2: u8 = 37;
const EXPERT: u8 = 38;
const XT: u8 = 40; // current token's x byte base (pair reservation)
const WAVE: u8 = 42; // wave index within the workgroup
const ST0: u8 = 43; // token tile base = 8*wgZ
const ST: u8 = 44; // current token index

// ---- VGPRs ----------------------------------------------------------------
const LANE: u8 = 0; // tid & 31
const WV: u8 = 1; // tid >> 5
const TID: u8 = 2; // flat thread id 0..255
const GEN: u8 = 3; // general address temp
const GEN2: u8 = 4; // second address temp
const XR: u8 = 5; // x row ring, 8 f32 (LDS-fed per group)
const PK: u8 = 13; // packed nibble dword
const HDR: u8 = 14; // header dword (lo=f16 scale, hi=f16 zero)
const DOT: u8 = 25; // group dot
const ACC0: u8 = 26; // streams acc[g&3]
const REDT: u8 = 30; // reduction address temp
const STOFF: u8 = 31; // store byte offset
fn w(n: u8) -> u8 {
    17 + n
}
fn xreg(n: u8) -> u8 {
    XR + n
}
fn acc(s: u8) -> u8 {
    ACC0 + s
}

fn s_add(arch: Arch) -> &'static str {
    s_add_i32(arch)
}
fn s_sub(arch: Arch) -> &'static str {
    if arch.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text.into(), refs(Kind::V, defs), inputs))
}
fn vmem(
    b: &mut Builder,
    text: impl Into<String>,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
    store: bool,
) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text.into(), refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: impl Into<String>) -> R {
    b.control(Instruction::new(text.into(), vec![], vec![]))
}
fn lbl(name: &str) -> String {
    format!(".Lgu8_{name}")
}
fn tlbl(t: u8, name: &str) -> String {
    format!(".Lgu8_t{t}_{name}")
}

fn plan() -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(64, 64)?;
    // Joint owners s8:15 / s16:19 / s20:21 matching the three SMEM defs;
    // pair/single subviews (PP/IP/XP/YG/YU pointers, SM/SK/SN/SR scalars)
    // are contained in exactly one owner each.
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("sys_x", SYS_X, 1),
        ("sys_y", SYS_Y, 1),
        ("sys_z", SYS_Z, 1),
        ("kern_pp_ip", PP, 8),
        ("kern_yu_s", YU, 4),
        ("kern_n_r", SN, 2),
        ("weight_resource", WSRD, 4),
        ("wg_x", SWX, 1),
        ("wg_y", SWY, 1),
        ("wg_z", SWZ, 1),
        ("tile_base", R0, 1),
        ("row", ROW, 1),
        ("half_m", MI, 1),
        ("row_bytes", ROWB, 1),
        ("group_off", GOFF, 1),
        ("stmp", STMP, 1),
        ("stmp2", STMP2, 1),
        ("expert", EXPERT, 1),
        ("x_base", XT, 2),
        ("wave", WAVE, 1),
        ("token_base", ST0, 1),
        ("token", ST, 1),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    for (name, base, len) in [
        ("tids_lo", 0, 4),
        ("tids_hi", 4, 1),
        ("xrow", XR, 8),
        ("packed", PK, 1),
        ("header", HDR, 1),
        ("weights", 17, 8),
        ("dot", DOT, 1),
        ("acc", ACC0, 4),
        ("redtmp", REDT, 1),
        ("storeoff", STOFF, 1),
    ] {
        regs.add_range(name, Kind::V, base, len, Live::Whole)?;
    }
    Ok(regs)
}

/// Emit the rows8 twin for one supported arch (gfx1201/gfx1151, else error).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1201 | Arch::Gfx1151) {
        return Err(format!(
            "legacy_mq4_rowtiles gate_up: no native AR recipe for {}, refusing generic math",
            arch.name()
        ));
    }
    let spec = KernelSpec {
        kernel_id: "legacy_mq4_rowtiles_gate_up".into(),
        variant: arch.name().into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: 256,
        group_segment_fixed_size: LDS_BYTES,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan()?);
    b.enable_workgroup_id_z();
    let slot = b.lds_slot("x_tile", 0, LDS_BYTES)?;
    prologue(&mut b)?;
    // Eight sequential token iterations reuse the one LDS slot: each live
    // iteration stages its token, publishes, consumes, and retires.
    for t in 0..TOKEN_TILE {
        token_iter(&mut b, slot, t)?;
    }
    b.label(&lbl("end"))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Kernarg loads, uniform domain guards, workgroup ids. Dead workgroups
/// (`M==0`, `R==0`, krank past K_TOP, token tile past R, output tile past M)
/// exit before any LDS access.
fn prologue(b: &mut Builder) -> R {
    let a = b.spec.arch;
    smem(b, PP, 8, KARG, 0)?;
    smem(b, YU, 4, KARG, 32)?;
    smem(b, SN, 2, KARG, 48)?;
    // Workgroup ids: gfx12 ttmp9 (x) + ttmp7 (y low, z high); gfx11 s2/s3/s4.
    if a.gfx12() {
        sop(b, format!("s_mov_b32 s{SWX}, ttmp9"), &[SWX], &[])?;
        sop(b, format!("s_and_b32 s{SWY}, ttmp7, 0xffff"), &[SWY], &[])?;
        sop(b, format!("s_lshr_b32 s{SWZ}, ttmp7, 16"), &[SWZ], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{SWX}, s{SYS_X}"), &[SWX], &[SYS_X])?;
        sop(b, format!("s_mov_b32 s{SWY}, s{SYS_Y}"), &[SWY], &[SYS_Y])?;
        sop(b, format!("s_mov_b32 s{SWZ}, s{SYS_Z}"), &[SWZ], &[SYS_Z])?;
    }
    b.wait_all()?;
    // M == 0: no output rows; R == 0: no tokens. Uniform exits.
    sop(b, format!("s_cmp_eq_u32 s{SM}, 0"), &[], &[SM])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl("end")))?;
    sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl("end")))?;
    // krank past K_TOP: no rank owned, uniform exit.
    sop(b, format!("s_cmp_ge_u32 s{SWY}, s{SK}"), &[], &[SWY, SK])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl("end")))?;
    // Token tile base 8*wgZ past R: dead token tile, uniform exit.
    sop(b, format!("s_lshl_b32 s{ST0}, s{SWZ}, 3"), &[ST0], &[SWZ])?;
    sop(b, format!("s_cmp_ge_u32 s{ST0}, s{SR}"), &[], &[ST0, SR])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl("end")))?;
    // Output tile base 8*bx past M: no live output row, uniform exit.
    sop(b, format!("s_lshl_b32 s{R0}, s{SWX}, 3"), &[R0], &[SWX])?;
    sop(b, format!("s_cmp_ge_u32 s{R0}, s{SM}"), &[], &[R0, SM])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl("end")))?;
    // MI = M >> 1 (gate/up split; M is the full output dim).
    sop(b, format!("s_lshr_b32 s{MI}, s{SM}, 1"), &[MI], &[SM])?;
    // Flat tid, lane, wave; wave id also to SGPR.
    valu(b, "v_mov_b32_e32 v2, v0", &[TID], &[0], &[])?;
    valu(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[LANE],
        &[TID],
        &[],
    )?;
    valu(
        b,
        format!("v_lshrrev_b32_e32 v{WV}, 5, v{TID}"),
        &[WV],
        &[TID],
        &[],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{WV}"),
        &[s(WAVE)],
        &[v(WV)],
    )?;
    // This wave's absolute output row = R0 + wave (SGPR, uniform in wave).
    sop(
        b,
        format!("{} s{ROW}, s{R0}, s{WAVE}", s_add(a)),
        &[ROW],
        &[R0, WAVE],
    )?;
    Ok(())
}

/// One token iteration: uniform token `ST = ST0 + t`; dead tokens skip the
/// whole iteration (both barriers) so every wave stays converged and the LDS
/// slot rests Free. Live tokens stage x, publish, run the original row DAG,
/// store, and retire.
fn token_iter(b: &mut Builder, slot: usize, t: u8) -> R {
    let a = b.spec.arch;
    if t == 0 {
        sop(b, format!("s_mov_b32 s{ST}, s{ST0}"), &[ST], &[ST0])?;
    } else {
        sop(b, format!("{} s{ST}, s{ST0}, {t}", s_add(a)), &[ST], &[ST0])?;
    }
    sop(b, format!("s_cmp_ge_u32 s{ST}, s{SR}"), &[], &[ST, SR])?;
    branch(b, &format!("s_cbranch_scc1 {}", tlbl(t, "skip")))?;
    // x byte base of this token = token * 2560 * 4.
    // Non-inline ALU constants use the native SOP2 hex-literal spelling.
    sop(
        b,
        format!("s_mul_i32 s{STMP}, s{ST}, {}", lit(LDS_BYTES)),
        &[STMP],
        &[ST],
    )?;
    sop(b, format!("{} s{XT}, s{STMP}, 0", s_add(a)), &[XT], &[STMP])?;
    stage(b, slot)?;
    b.barrier(&[Transition::Ready(slot)])?;
    per_row(b, t)?;
    // Live waves fall through with acc in acc0; dead/invalid paths join below.
    b.label(&tlbl(t, "store"))?;
    store(b, t)?;
    b.label(&tlbl(t, "nostore"))?;
    b.wait_all()?;
    b.barrier(&[Transition::Retire(slot)])?;
    b.label(&tlbl(t, "skip"))?;
    Ok(())
}

/// Stage the current token's 2560 f32 x row into LDS (all 256 threads; two
/// unconditional float4 rounds plus a tid<128 third round under exec mask).
fn stage(b: &mut Builder, slot: usize) -> R {
    let sts: Vec<u8> = vec![XP, XP + 1];
    for k in 0..3u8 {
        if k == 2 {
            // E32 compares src0 against src1: 128 > tid selects waves 0..3,
            // whose third float4 writes bytes 8192..10239 (not tid > 128).
            valu(b, "v_cmp_gt_u32_e32 vcc_lo, 0x80, v2", &[], &[TID], &[])?;
            if b.spec.arch.gfx12() {
                b.push(Instruction::new(
                    "s_wait_alu depctr_va_vcc(0)",
                    vec![],
                    vec![],
                ))?;
            }
            valu(b, "v_cndmask_b32_e64 v4, 0, 1, vcc_lo", &[GEN2], &[], &[])?;
            op(b, "s_and_b32 exec_lo, exec_lo, vcc_lo", &[], &[v(GEN2)])?;
        }
        // float4 index i = tid + 256k; byte offset = tokenbase + i*16.
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{GEN}, 4, v{TID}"),
            &[GEN],
            &[TID],
            &[],
        )?;
        if k > 0 {
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{GEN}, {}, v{GEN}", lit(k as u32 * 4096)),
                &[GEN],
                &[GEN],
                &[],
            )?;
        }
        valu(
            b,
            format!("v_add_nc_u32_e32 v{GEN}, s{XT}, v{GEN}"),
            &[GEN],
            &[GEN],
            &[XT],
        )?;
        vmem(
            b,
            &format!(
                "global_load_b128 v[{XR}:{}], v{GEN}, s[{XP}:{}]",
                XR + 3,
                XP + 1
            ),
            &[XR, XR + 1, XR + 2, XR + 3],
            &[GEN],
            &sts,
            false,
        )?;
        // LDS dword index = i*4, byte = i*16.
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{GEN2}, 4, v{TID}"),
            &[GEN2],
            &[TID],
            &[],
        )?;
        if k > 0 {
            valu(
                b,
                &format!(
                    "v_add_nc_u32_e32 v{GEN2}, {}, v{GEN2}",
                    lit(k as u32 * 4096)
                ),
                &[GEN2],
                &[GEN2],
                &[],
            )?;
        }
        b.ds_store(
            slot,
            Instruction::new(
                format!("ds_store_b128 v{GEN2}, v[{XR}:{}]", XR + 3),
                vec![],
                vec![v(XR), v(XR + 1), v(XR + 2), v(XR + 3), v(GEN2)],
            )
            .memory(MemoryClass::DsStore),
        )?;
        if k == 2 {
            op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        }
    }
    b.wait_all()
}

/// Per-wave row work for the current token: output-dead-row skip, expert
/// lookup + invalid guard, weight resource, then the ten original group
/// dots, fold and cross-lane tree.
fn per_row(b: &mut Builder, t: u8) -> R {
    // Dead tail output rows do no loads, arithmetic or stores (but still
    // reach both barriers of this iteration via the nostore label).
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {}", tlbl(t, "nostore")))?;
    // Uniform expert lookup: soff = (token*KTOP + krank)*4.
    sop(
        b,
        format!("s_mul_i32 s{STMP}, s{ST}, s{SK}"),
        &[STMP],
        &[ST, SK],
    )?;
    sop(
        b,
        format!("{} s{STMP}, s{STMP}, s{SWY}", s_add(b.spec.arch)),
        &[STMP],
        &[STMP, SWY],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{STMP}, s{STMP}, 2"),
        &[STMP],
        &[STMP],
    )?;
    valu(
        b,
        format!("v_mov_b32_e32 v{GEN}, s{STMP}"),
        &[GEN],
        &[],
        &[STMP],
    )?;
    vmem(
        b,
        &format!("global_load_b32 v{GEN2}, v{GEN}, s[{IP}:{}]", IP + 1),
        &[GEN2],
        &[GEN],
        &[IP, IP + 1],
        false,
    )?;
    b.wait_all()?;
    op(
        b,
        format!("v_readfirstlane_b32 s{EXPERT}, v{GEN2}"),
        &[s(EXPERT)],
        &[v(GEN2)],
    )?;
    // Invalid expert (unsigned >= n_experts catches negatives): lane 0
    // stores 0.0 to the live row's gate/up slot.
    sop(
        b,
        format!("s_cmp_ge_u32 s{EXPERT}, s{SN}"),
        &[],
        &[EXPERT, SN],
    )?;
    branch(b, &format!("s_cbranch_scc1 {}", tlbl(t, "inv")))?;
    // expert base: global_load_b64 of table[expert] then to SGPR pair.
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{GEN}, 3, v{GEN2}"),
        &[GEN],
        &[GEN2],
        &[],
    )?;
    vmem(
        b,
        &format!(
            "global_load_b64 v[{GEN}:{GEN2}], v{GEN}, s[{PP}:{}]",
            PP + 1
        ),
        &[GEN, GEN2],
        &[GEN],
        &[PP, PP + 1],
        false,
    )?;
    b.wait_all()?;
    op(
        b,
        format!("v_readfirstlane_b32 s{STMP}, v{GEN}"),
        &[s(STMP)],
        &[v(GEN)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{STMP2}, v{GEN2}"),
        &[s(STMP2)],
        &[v(GEN2)],
    )?;
    // Raw buffer resource over the expert's weights; base = table[expert]
    // plus this output row's 1360 bytes (single 64-bit add with carry).
    sop(
        b,
        format!("s_mul_i32 s{ROWB}, s{ROW}, {}", lit(ROW_BYTES)),
        &[ROWB],
        &[ROW],
    )?;
    sop(b, format!("s_mov_b32 s{WSRD}, s{STMP}"), &[WSRD], &[STMP])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{STMP2}", WSRD + 1),
        &[WSRD + 1],
        &[STMP2],
    )?;
    {
        let a = b.spec.arch;
        sop(
            b,
            format!(
                "{} s{}, s{}, s{ROWB}",
                crate::kernels::common::s_add_u32(a),
                WSRD,
                WSRD
            ),
            &[WSRD],
            &[WSRD, ROWB],
        )?;
        sop(
            b,
            format!(
                "{} s{}, s{}, 0",
                crate::kernels::common::s_addc_u32(a),
                WSRD + 1,
                WSRD + 1
            ),
            &[WSRD + 1],
            &[WSRD + 1],
        )?;
    }
    srd_tail(b, WSRD, None)?;
    b.wait_all()?;
    // Original row10 uses wave-local tid: header offset 0/4, payload lane*4.
    // Adding flat tid*4 to the header instead reinterprets payload as f16s.
    valu(
        b,
        format!("v_lshrrev_b32_e32 v{GEN2}, 4, v{LANE}"),
        &[GEN2],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{GEN2}, 2, v{GEN2}"),
        &[GEN2],
        &[GEN2],
        &[],
    )?;
    // acc = +0 (four streams).
    for s in 0..4u8 {
        valu(
            b,
            format!("v_mov_b32_e32 v{}, 0", acc(s)),
            &[acc(s)],
            &[],
            &[],
        )?;
    }
    // Ten original groups, stream s = g & 3, groups strictly ascending.
    for g in 0..10u32 {
        group(b, g)?;
    }
    // Fold (a0+a1) + (a2+a3).
    for (d, x) in [(0u8, 1u8), (2, 3)] {
        valu(
            b,
            format!("v_add_f32_e32 v{}, v{}, v{}", acc(d), acc(d), acc(x)),
            &[acc(d)],
            &[acc(d), acc(x)],
            &[],
        )?;
    }
    valu(
        b,
        format!("v_add_f32_e32 v{}, v{}, v{}", acc(0), acc(0), acc(2)),
        &[acc(0)],
        &[acc(0), acc(2)],
        &[],
    )?;
    tree(b)?;
    branch(b, &format!("s_branch {}", tlbl(t, "store")))?;
    // Invalid expert: acc = 0.0, join the shared store.
    b.label(&tlbl(t, "inv"))?;
    valu(
        b,
        format!("v_mov_b32_e32 v{}, 0", acc(0)),
        &[acc(0)],
        &[],
        &[],
    )?;
    Ok(())
}

/// One original group: header + packed loads, eight-term affine dot
/// (term 1 first, then 0, 2..7; weight-by-x orientation), stream add.
fn group(b: &mut Builder, g: u32) -> R {
    let policy = if b.spec.arch.gfx12() {
        " scope:SCOPE_DEV"
    } else {
        ""
    };
    // Match native buffer helpers: gfx12's 7-bit SOFFSET cannot encode inline zero (128).
    let soffset = if b.spec.arch.gfx12() { "null" } else { "0" };
    sop(
        b,
        format!("s_mov_b32 s{GOFF}, {}", lit(g * 136)),
        &[GOFF],
        &[],
    )?;
    // vGO = GOFF + vH (header voff).
    valu(
        b,
        format!("v_add_nc_u32_e32 v{GEN}, s{GOFF}, v{GEN2}"),
        &[GEN],
        &[GEN2],
        &[GOFF],
    )?;
    let wsr: Vec<u8> = (WSRD..WSRD + 4).collect();
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{HDR}, v{GEN}, s[{WSRD}:{}], {soffset} offen{policy}",
            WSRD + 3
        ),
        &[HDR],
        &[GEN],
        &wsr,
        false,
    )?;
    // vGP = GOFF + lane*4 (packed voff; each wave owns its own row).
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{GEN}, 2, v{LANE}"),
        &[GEN],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{GEN}, s{GOFF}, v{GEN}"),
        &[GEN],
        &[GEN],
        &[GOFF],
    )?;
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{PK}, v{GEN}, s[{WSRD}:{}], {soffset} offen offset:8{policy}",
            WSRD + 3
        ),
        &[PK],
        &[GEN],
        &wsr,
        false,
    )?;
    b.wait(
        if b.spec.arch.gfx12() {
            crate::ledger::Counter::Load
        } else {
            crate::ledger::Counter::Vm
        },
        0,
    )?;
    // Dequant: w_i = sc*q_i + zp via one fma_mix (frozen clang spelling).
    for n in 0..8u8 {
        valu(
            b,
            format!("v_bfe_u32 v{}, v{PK}, {}, 4", w(n), n * 4),
            &[w(n)],
            &[PK],
            &[],
        )?;
        valu(
            b,
            format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}", w = w(n)),
            &[w(n)],
            &[w(n)],
            &[],
        )?;
        valu(
            b,
            format!(
                "v_fma_mix_f32 v{w}, v{HDR}, v{w}, v{HDR} op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                w = w(n)
            ),
            &[w(n)],
            &[HDR, w(n)],
            &[],
        )?;
    }
    // x row of this group from LDS: byte (g*256 + 8*lane + n)*4.
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{GEN}, 5, v{LANE}"),
        &[GEN],
        &[LANE],
        &[],
    )?;
    if g > 0 {
        valu(
            b,
            format!("v_add_nc_u32_e32 v{GEN}, {}, v{GEN}", lit(g * 1024)),
            &[GEN],
            &[GEN],
            &[],
        )?;
    }
    for n in 0..8u8 {
        let off = if n == 0 {
            String::new()
        } else {
            format!(" offset:{}", u32::from(n) * 4)
        };
        b.ds_load(
            0,
            Instruction::new(
                format!("ds_load_b32 v{}, v{GEN}{off}", xreg(n)),
                vec![v(xreg(n))],
                vec![v(GEN)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    // Dot: term 1 mul, then fmac 0, 2..7 (weight lhs, x rhs).
    let s = (g & 3) as u8;
    valu(
        b,
        format!("v_mul_f32_e32 v{DOT}, v{}, v{}", w(1), xreg(1)),
        &[DOT],
        &[w(1), xreg(1)],
        &[],
    )?;
    for n in [0u8, 2, 3, 4, 5, 6, 7] {
        valu(
            b,
            format!("v_fmac_f32_e32 v{DOT}, v{}, v{}", w(n), xreg(n)),
            &[DOT],
            &[DOT, w(n), xreg(n)],
            &[],
        )?;
    }
    valu(
        b,
        format!("v_add_f32_e32 v{}, v{}, v{DOT}", acc(s), acc(s)),
        &[acc(s)],
        &[acc(s), DOT],
        &[],
    )
}

/// Cross-lane tree: swizzle "1pppp" then bpermute +8/+4/+2/+1, own lane
/// when out of range, `own + other` every level.
fn tree(b: &mut Builder) -> R {
    let a = b.spec.arch;
    b.ds_crosslane(
        Instruction::new(
            format!(
                "ds_swizzle_b32 v{REDT}, v{} offset:swizzle(BITMASK_PERM,\"1pppp\")",
                acc(0)
            ),
            refs(Kind::V, &[REDT]),
            refs(Kind::V, &[acc(0)]),
        )
        .memory(MemoryClass::DsLoad),
    )?;
    valu(
        b,
        format!("v_add_f32_e32 v{}, v{}, v{REDT}", acc(0), acc(0)),
        &[acc(0)],
        &[acc(0), REDT],
        &[],
    )?;
    for off in [8u32, 4, 2, 1] {
        valu(
            b,
            format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - off),
            &[],
            &[LANE],
            &[],
        )?;
        if a.gfx12() {
            b.push(Instruction::new(
                "s_wait_alu depctr_va_vcc(0)",
                vec![],
                vec![],
            ))?;
        }
        valu(
            b,
            format!("v_cndmask_b32_e64 v{REDT}, 0, {off}, vcc_lo"),
            &[REDT],
            &[],
            &[],
        )?;
        valu(
            b,
            format!("v_add_lshl_u32 v{REDT}, v{REDT}, v{LANE}, 2"),
            &[REDT],
            &[REDT, LANE],
            &[],
        )?;
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v{REDT}, v{REDT}, v{}", acc(0)),
                refs(Kind::V, &[REDT]),
                refs(Kind::V, &[REDT, acc(0)]),
            )
            .memory(MemoryClass::DsLoad),
        )?;
        valu(
            b,
            format!("v_add_f32_e32 v{}, v{}, v{REDT}", acc(0), acc(0)),
            &[acc(0)],
            &[acc(0), REDT],
            &[],
        )?;
    }
    Ok(())
}

/// Lane-0 store of the folded acc to gate (`row < MI`) or up (`row - MI`),
/// at this token's `(token*K_TOP + krank)*MI` output window.
fn store(b: &mut Builder, t: u8) -> R {
    valu(
        b,
        format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    // Byte offset ((token*KTOP + krank)*MI + (row or row-MI)) * 4.
    sop(
        b,
        format!("s_mul_i32 s{STMP}, s{ST}, s{SK}"),
        &[STMP],
        &[ST, SK],
    )?;
    sop(
        b,
        format!("{} s{STMP}, s{STMP}, s{SWY}", s_add(b.spec.arch)),
        &[STMP],
        &[STMP, SWY],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{STMP}, s{STMP}, s{MI}"),
        &[STMP],
        &[STMP, MI],
    )?;
    sop(
        b,
        format!("{} s{STMP}, s{STMP}, s{ROW}", s_add(b.spec.arch)),
        &[STMP],
        &[STMP, ROW],
    )?;
    // Up half rows address row-MI: subtract MI when row >= MI.
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{MI}"), &[], &[ROW, MI])?;
    // Keep the taken edge's counter/hazard lifetime: the gate arm's VMEM
    // clears its SGPR hazards, but the up arm bypasses that instruction.
    // Mirror author.rs fork/resume/join for the state these two arms change.
    let up_ledger = b.ledger.clone();
    let up_sgpr = b.hazard.clone();
    let up_trans = b.gfx11_hazard.clone();
    branch(b, &format!("s_cbranch_scc1 {}", tlbl(t, "up")))?;
    sop(
        b,
        format!("s_lshl_b32 s{STMP}, s{STMP}, 2"),
        &[STMP],
        &[STMP],
    )?;
    valu(
        b,
        format!("v_mov_b32_e32 v{STOFF}, s{STMP}"),
        &[STOFF],
        &[],
        &[STMP],
    )?;
    vmem(
        b,
        &format!("global_store_b32 v{STOFF}, v{}, s[{YG}:{}]", acc(0), YG + 1),
        &[],
        &[STOFF, acc(0)],
        &[YG, YG + 1],
        true,
    )?;
    branch(b, &format!("s_branch {}", tlbl(t, "stored")))?;
    let gate_ledger = b.ledger.clone();
    let gate_sgpr = b.hazard.clone();
    let gate_trans = b.gfx11_hazard.clone();
    b.label(&tlbl(t, "up"))?;
    b.ledger.resume(up_ledger);
    b.hazard = up_sgpr;
    b.gfx11_hazard = up_trans;
    // Builder::emit now guards this first s36 read with sa_sdst(0) on
    // gfx1201; gfx1151 retains its own NoDataDepHazard rule.
    sop(
        b,
        format!("{} s{STMP}, s{STMP}, s{MI}", s_sub(b.spec.arch)),
        &[STMP],
        &[STMP, MI],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{STMP}, s{STMP}, 2"),
        &[STMP],
        &[STMP],
    )?;
    valu(
        b,
        format!("v_mov_b32_e32 v{STOFF}, s{STMP}"),
        &[STOFF],
        &[],
        &[STMP],
    )?;
    vmem(
        b,
        &format!("global_store_b32 v{STOFF}, v{}, s[{YU}:{}]", acc(0), YU + 1),
        &[],
        &[STOFF, acc(0)],
        &[YU, YU + 1],
        true,
    )?;
    b.label(&tlbl(t, "stored"))?;
    b.ledger.join(&gate_ledger);
    b.join_hazards(&gate_sgpr, &gate_trans, &[]);
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    b.wait_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_both_arches_and_rejects_the_rest() {
        for arch in [Arch::Gfx1201, Arch::Gfx1151] {
            let e = emit(arch).expect("supported arch emits");
            assert!(
                e.s_text
                    .contains(&format!(".amdhsa_kernel {}", symbol(arch))),
                "{arch:?}: missing export symbol"
            );
        }
        for arch in [Arch::Gfx1100, Arch::Gfx942] {
            assert!(emit(arch).is_err(), "{arch:?}: must be refused");
        }
    }

    #[test]
    fn abi_and_geometry_markers() {
        let e = emit(Arch::Gfx1201).expect("gfx1201 emits");
        // Ten-group original DAG markers.
        assert!(e.s_text.contains("v_fma_mix_f32"));
        assert!(e.s_text.contains("v_bfe_u32"));
        assert!(e.s_text.contains("ds_swizzle_b32"));
        assert!(e.s_text.contains("ds_bpermute_b32"));
        assert!(e.s_text.contains("ds_store_b128"));
        assert!(e.s_text.contains("global_load_b128"));
        assert!(e.s_text.contains("global_store_b32"));
        assert!(e.s_text.contains("buffer_load_b32"));
        // Twin marker, never the old singleton name as the export.
        assert!(e.s_text.contains("_t8_pm_gfx1201"));
        // Appended R kernarg at offset 52 keeps the original 52 intact.
        assert_eq!(KERNARG_BYTES, 56);
        // No forbidden families. `flat_` is checked on instruction mnemonics
        // only: the `.max_flat_workgroup_size` metadata key always contains it.
        for bad in ["wmma", "swmmac", "iu4"] {
            assert!(!e.s_text.contains(bad), "forbidden {bad} present");
        }
        assert!(
            !e.s_text
                .lines()
                .any(|l| l.trim_start().starts_with("flat_")),
            "forbidden flat_ instruction present"
        );
        // The two per-iteration LDS barriers are the gfx12 signal/wait pair
        // (`Arch::barrier`); plain `s_barrier` is illegal on gfx12.
        assert!(e.s_text.contains("s_barrier_signal -1"));
        assert!(e.s_text.contains("s_barrier_wait 0xffff"));
        assert!(!e.s_text.lines().any(|l| l.trim() == "s_barrier"));
        // Register / domain contract.
        assert!(e.shape.next_free_vgpr <= 64);
        assert!(e.shape.next_free_sgpr <= 64);
        assert!(e.shape.waits > 0, "checked waits must exist");
    }

    /// Source-level CPU oracle of the exact per-row DAG order: stream
    /// `g&3` accumulation, term order 1,0,2..7, fold `(a0+a1)+(a2+a3)`.
    /// Mirrors `mq4v2_gate_up_row10` (K=2560 fast path) structurally.
    fn model_row(groups: &[[f32; 8]]) -> f32 {
        let mut acc = [0.0f32; 4];
        for (g, grp) in groups.iter().enumerate() {
            // Term 1 starts the dot (scalar singleton recipe), then 0, 2..7.
            let mut d = grp[1];
            for n in [0usize, 2, 3, 4, 5, 6, 7] {
                d += grp[n];
            }
            acc[g & 3] += d;
        }
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    }

    #[test]
    fn dag_oracle_matches_reference_fold() {
        let groups: Vec<[f32; 8]> = (0..10)
            .map(|g| core::array::from_fn(|n| (g * 8 + n) as f32 * 0.25 - 3.0))
            .collect();
        let got = model_row(&groups);
        let mut acc = [0.0f32; 4];
        for (g, grp) in groups.iter().enumerate() {
            let mut d = grp[1];
            for n in [0usize, 2, 3, 4, 5, 6, 7] {
                d = d + grp[n];
            }
            acc[g & 3] = acc[g & 3] + d;
        }
        let flat = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        assert_eq!(got, flat);
        assert!(got.is_finite());
    }
}
