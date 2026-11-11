// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-eight-token-row twin of the aggregate indexed unweighted down
//! singleton (`gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`).
//!
//! Original singleton recipe (admitted AR oracle):
//! `kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip:15-90`,
//! launched by `crates/rdna-compute/src/moe.rs:2878-2963`
//! (`gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`), grid
//! `[M, k_top, batch]`, block `[32, 1, 1]`, dynamic LDS 0, no scratch.
//!
//! Per original workgroup `(row, rank, token)`: `flat = token * k_top + rank`,
//! `expert = topk_indices[flat]`. A dead domain returns without a store.
//! A live token with an id outside `[0, n_exp)` stores positive zero without
//! weight/activation loads. A valid expert streams K
//! ascending in 128-wide groups: group bytes are `scale:2 (f16, HIP
//! `__half2float`), `zero:2 (f16)`, then 64 payload bytes; lane `tid` owns K
//! `4*tid .. 4*tid+3` of the group with `w = scale * nibble + zero` per
//! nibble (`v_fma_f32` affine: one fused rounding), `d = x1*w1`
//! (`v_mul_f32`) then `d = fma(x0, w0, d)`, followed by terms 2 and 3,
//! added into one accumulator, then the `__shfl_down` 16/8/4/2/1 tree with
//! lane 0 storing `expert_outputs[flat * M + row]` (F32, `[batch, k_top, M]`
//! logical; the indexed combine `moe_down_combine_top10_batched` alone owns
//! route weighting). With `(K & 127) == 0` the whole groups take unguarded x
//! loads; otherwise each x is guarded (`i < K ? x[i] : 0.0`). Aggregate qt44
//! (G256, 136-byte rows) / qt53 (G128, 68-byte rows) wire rows are borrowed
//! in place through the existing A pointers/strides; no slab reinterpretation,
//! no WMMA/IU4, no fp16 precision change, no across-row reduction.
//!
//! Twin ABI: byte-identical to the original HIP by-value kernarg segment
//! (52 bytes; N@40 is batch/token rows = the logical total R, 1..=64):
//! expert_ptrs ptr@0 (read_only), topk_indices ptr@8 (read_only),
//! hidden_batch ptr@16 (read_only), expert_outputs ptr@24 (write_only),
//! M i32@32 (actual output dim, 2560), K i32@36, N i32@40 (total R),
//! n_exp i32@44, k_top i32@48. Geometry: grid `[M, k_top, ceil(N/8)]`,
//! block `[32,1,1]`, LDS 0.
//! Workgroup `(row, rank, tile)` owns 8 physical token rows
//! `t = 8*tile + s` for `s in 0..8`; slot `s` has `flat(s) = t * k_top + rank`
//! and the original output coordinate `flat(s) * M + row`. A tile whose start
//! is past N exits uniformly before any data-memory access (uniformly dead
//! WG). The eight token bodies execute serially with reused registers, each
//! streaming ascending K groups with the original per-token DAG. Dead tokens
//! skip before index loads and perform no arithmetic or stores. Live tokens
//! with invalid experts store +0 without weight/X loads or dot arithmetic.
//!
//! The actual Native gfx1201 singleton's loaded raw ELF is SHA-256
//! `8ec3f2d218dc18c74031ad0c933d4ec2cb83f33211c579f1dd99ffccef29db55`.
//! Its unrolled whole-group body at 0x1f4c multiplies x1*w1; 0x1fc4
//! contracts x0*w0 into that product, followed by x2 and x3 at
//! 0x1ff0/0x2008 and a separate accumulator add at 0x2040. Whole-group
//! remainder (0x223c/0x224c) and ragged (0x1afc/0x1b28) agree. The
//! source-level `p0 + p1` must not become `fma(x1, w1, x0*w0)`:
//! that changes finite model output bits.
//! Historical singleton export receipts remain in
//! `row-exact-wide/census/historical-route-objects.json`; they are not the
//! loaded Native identity above. No new native ISA is required: the emitter
//! spells the actual AR body (scalar address arithmetic, GFN
//! `v_bfe_u32`/`v_cvt_f32_u32` dequant, `v_fma_f32` affine weights,
//! mul-then-fma dot, `ds_bpermute` shuffle-down lowering of `__shfl_down`,
//! lane-0 `global_store_b32`) with the architecture's system-register and
//! wait spellings (gfx12 `ttmp9`/`ttmp7`, `s_wait_loadcnt_dscnt`,
//! `global_inv`; gfx11 system SGPRs, `s_waitcnt`, `buffer_gl0_inv`).
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::lit;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{arch::WgIdSource, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// Physical token rows per workgroup (tile width in the z dimension).
pub const TILE_ROWS: u8 = 8;
/// Maximum logical token rows (the existing N argument domain).
pub const MAX_ROWS: u8 = 64;
/// Kernarg segment size: byte-identical to the original HIP segment.
pub const KERNARG_BYTES: u32 = 52;

fn symbol(arch: Arch) -> String {
    format!(
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
        arch.name()
    )
}
fn module(arch: Arch) -> String {
    format!("legacy_mq4_rowtiles_pm_{}", arch.name())
}

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_ptrs", 0, Access::ReadOnly)
        .pointer_access("topk_indices", 8, Access::ReadOnly)
        .pointer_access("hidden_batch", 16, Access::ReadOnly)
        .pointer_access("expert_outputs", 24, Access::WriteOnly)
        .hidden("M", 32, 4, "by_value")
        .hidden("K", 36, 4, "by_value")
        .hidden("N", 40, 4, "by_value")
        .hidden("n_exp", 44, 4, "by_value")
        .hidden("k_top", 48, 4, "by_value")
}

// ---- SGPRs ----------------------------------------------------------------
// s0:1 kernarg pointer. gfx12 workgroup ids: ttmp9 (x) / ttmp7 low/high
// (y/z). gfx11: system SGPRs s[wg_x_in] (x), +1 (y), +2 (z), after the DECLARED user SGPRs.
const KARG: u8 = 0;
/// Real user SGPRs (the kernarg pointer) and enabled system SGPRs (workgroup ids X, Y, Z).
const USER_SGPRS: u8 = 2;
const SYSTEM_SGPRS: u8 = 3;
/// gfx11 workgroup id x: after the DECLARED user SGPRs. gfx1100 pads the descriptor to user + system >= 16
/// (`Arch::declared_user_sgprs`, LLVM FeatureUserSGPRInit16Bug), so the ids are s13, s14, s15 there.
fn wg_x_in(arch: Arch) -> Result<u8, String> {
    match arch
        .workgroup_ids(arch.declared_user_sgprs(USER_SGPRS, SYSTEM_SGPRS))?
        .x
    {
        WgIdSource::Sgpr(n) => Ok(n),
        other => Err(format!(
            "legacy_mq4 down t8: no workgroup id SGPR on {}: {other:?}",
            arch.name()
        )),
    }
}
// Kernarg words -> s8..s20 (the pointer block s[8:15]; on gfx1100 s13..s15 are the workgroup ids, so the
// block moves to `KARG_BLOCK_GFX1100`).
struct Karg {
    pptr: u8, // expert_ptrs
    iptr: u8, // topk_indices
    xptr: u8, // hidden_batch
    yptr: u8, // expert_outputs
}
const KARG_BLOCK_GFX1100: u8 = 48;
fn karg(arch: Arch) -> Karg {
    let base = if arch.min_user_system_sgprs() > 0 {
        KARG_BLOCK_GFX1100
    } else {
        8
    };
    Karg {
        pptr: base,
        iptr: base + 2,
        xptr: base + 4,
        yptr: base + 6,
    }
}
/// SGPR budget of the plan: `S_END`, or past the relocated pointer block on gfx1100.
fn s_end(arch: Arch) -> u8 {
    if arch.min_user_system_sgprs() > 0 {
        KARG_BLOCK_GFX1100 + 8
    } else {
        S_END
    }
}
const SM: u8 = 16; // M (actual output dim, 2560)
const SK: u8 = 17; // K
const SN: u8 = 18; // N (batch/token rows: the logical total R, 1..=64)
const SNEXP: u8 = 19; // n_exp
const SKT: u8 = 20; // k_top
                    // Derived.
const ROW: u8 = 21; // x id (output row)
const RANK: u8 = 22; // y id (rank)
const TILE: u8 = 23; // z id (8-row tile)
const TBASE: u8 = 24; // 8*tile (first physical token)
const GROUPS: u8 = 25;
const GTAIL: u8 = 26; // K & 127 remainder class: 0 whole, else ragged
const GI: u8 = 27; // group cursor
const TMP: u8 = 28;
const TMP2: u8 = 29;
const EB0: u8 = 30; // expert base pointer, low (per-slot broadcast)
const EB1: u8 = 31; // expert base pointer, high
const SHM: u8 = 32; // 5 shuffle lane-mask pairs (32..41)
const S_END: u8 = 42;

// ---- VGPRs ----------------------------------------------------------------
const LANE: u8 = 0; // tid = lane id (also v0 hardware state)
const VO: u8 = 1; // weight byte offset = tid*2
const VA: u8 = 2; // address temp
const VB: u8 = 3; // loaded word temp (VB+1 = pair high)
const HD: u8 = 5; // scale (f32, v_cvt from header .l)
const HZ: u8 = 6; // zero  (f32)
const W0: u8 = 7; // dequant weights w0..w3
const X0: u8 = 11; // x values x0..x3
const DOT: u8 = 15; // group dot
const ACC: u8 = 16; // one token accumulator, reused across the physical tile
const FLAT: u8 = 17;
const EID: u8 = 18;
const SHUF: u8 = 19; // five shuffle addresses
const EXCH: u8 = 24;
const KIDX: u8 = 25; // lane-local group element index (ragged guard)
const V_END: u8 = 26;

fn plan(arch: Arch) -> Result<RegPlan, String> {
    // gfx1100: s[2:16) are the (padded) user SGPRs and the workgroup ids (s13..s15); the kernarg pointer block
    // is relocated by `karg`.
    let mut p = RegPlan::new(u16::from(V_END), u16::from(s_end(arch)))?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    if !arch.gfx12() {
        let wgx = wg_x_in(arch)?;
        p.s::<1>("workgroup_id_x", wgx, Live::Whole)?;
        p.s::<1>("workgroup_id_y", wgx + 1, Live::Whole)?;
        p.s::<1>("workgroup_id_z", wgx + 2, Live::Whole)?;
    }
    p.add_range("kernarg_ptr0", Kind::S, karg(arch).pptr, 8, Live::Whole)?;
    p.add_range("kernarg_rest", Kind::S, SM, 4, Live::Whole)?;
    p.add_range("kernarg_kt", Kind::S, SKT, 1, Live::Whole)?;
    p.add_range("derived_row", Kind::S, ROW, 1, Live::Whole)?;
    p.add_range("derived_rank_tile", Kind::S, RANK, 2, Live::Whole)?;
    p.add_range("derived_groups", Kind::S, TBASE, 4, Live::Whole)?;
    p.add_range("derived_tmp", Kind::S, TMP, 1, Live::Whole)?;
    p.add_range("derived_b", Kind::S, TMP2, 1, Live::Whole)?;
    p.add_range("expert_base", Kind::S, EB0, 2, Live::Whole)?;
    p.add_range("shuffle_masks_a", Kind::S, SHM, 8, Live::Whole)?;
    p.add_range("shuffle_masks_b", Kind::S, SHM + 8, 2, Live::Whole)?;
    p.add_range("vgpr_file_a", Kind::V, 0, 8, Live::Whole)?;
    p.add_range("vgpr_file_b", Kind::V, 8, 8, Live::Whole)?;
    p.add_range("vgpr_file_c", Kind::V, 16, 8, Live::Whole)?;
    p.add_range("vgpr_file_d", Kind::V, 24, 2, Live::Whole)?;
    Ok(p)
}

fn sreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base,
        len,
    }
}
fn vreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len,
    }
}
fn refs(kind: Kind, ids: &[u8]) -> Vec<RegRef> {
    ids.iter()
        .map(|&b| RegRef {
            kind,
            base: b,
            len: 1,
        })
        .collect()
}
fn sop(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8]) -> R {
    b.push(Instruction::new(
        text,
        refs(Kind::S, defs),
        refs(Kind::S, uses),
    ))
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
fn raw(b: &mut Builder, text: &str) -> R {
    b.push(Instruction::new(text, vec![], vec![]))
}
fn smem(b: &mut Builder, dst: u8, len: u8, base: u8, off: u32) -> R {
    let name = match len {
        1 => "s_load_b32",
        2 => "s_load_b64",
        4 => "s_load_b128",
        8 => "s_load_b256",
        _ => return Err("SMEM width".into()),
    };
    let d = if len == 1 {
        sreg(dst, 1)
    } else {
        sreg(dst, len)
    };
    let text = if b.spec.arch.gfx12() {
        format!("{name} {d}, s[{base}:{}], {off:#x}", base + 1)
    } else if off == 0 {
        format!("{name} {d}, s[{base}:{}], null", base + 1)
    } else {
        format!("{name} {d}, s[{base}:{}], {off:#x}", base + 1)
    };
    b.push(Instruction::new(text, vec![d], vec![sreg(base, 2)]).memory(MemoryClass::SmemLoad))
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
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
fn gload_b128(b: &mut Builder, dst: u8, voff: u8, base: u8) -> R {
    b.push(
        Instruction::new(
            format!(
                "global_load_b128 v[{dst}:{}], v{voff}, s[{base}:{}]",
                dst + 3,
                base + 1
            ),
            vec![vreg(dst, 4)],
            vec![vreg(voff, 1), sreg(base, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn s_add_i32(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_add_co_i32"
    } else {
        "s_add_i32"
    }
}

/// The builder-emitted twin (gfx1201, gfx1151, gfx1100; every other target refused).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if !matches!(arch, Arch::Gfx1201 | Arch::Gfx1151 | Arch::Gfx1100) {
        return Err(format!(
            "legacy_mq4 down t8: built for gfx1201/gfx1151/gfx1100 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: "legacy_mq4_down_t8".into(),
        variant: "indexed_expanded".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(arch)?);
    b.enable_delay_alu();
    // Tile Z is an entry input on every target: s4 on gfx11, ttmp7.hi on gfx12.
    b.enable_workgroup_id_z();
    kernel(&mut b, arch)?;
    b.finish()
}

/// The single-symbol module `legacy_mq4_rowtiles_pm_<arch>` order is owned by
/// the parent root (`legacy_mq4_rowtiles.rs`); this helper is unused by it.
#[allow(dead_code)]
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String), String> {
    Ok((vec![emit(arch)?], module(arch)))
}

// ---- kernel ---------------------------------------------------------------
fn kernel(b: &mut Builder, arch: Arch) -> R {
    let end = ".Ldown_t8_end";
    // Kernarg words (original 52-byte segment: 32B pointers + 20B scalars).
    smem(b, karg(arch).pptr, 8, KARG, 0x00)?;
    smem(b, SM, 4, KARG, 0x20)?;
    smem(b, SKT, 1, KARG, 0x30)?;
    b.wait_all()?;
    // Workgroup ids: x = row, y = rank, z = 8-row tile.
    if arch.gfx12() {
        sop(b, format!("s_mov_b32 s{ROW}, ttmp9"), &[ROW], &[])?;
        sop(b, format!("s_and_b32 s{RANK}, ttmp7, 0xffff"), &[RANK], &[])?;
        sop(b, format!("s_lshr_b32 s{TILE}, ttmp7, 16"), &[TILE], &[])?;
    } else {
        let wgx = wg_x_in(arch)?;
        sop(b, format!("s_mov_b32 s{ROW}, s{wgx}"), &[ROW], &[wgx])?;
        sop(
            b,
            format!("s_mov_b32 s{RANK}, s{}", wgx + 1),
            &[RANK],
            &[wgx + 1],
        )?;
        sop(
            b,
            format!("s_mov_b32 s{TILE}, s{}", wgx + 2),
            &[TILE],
            &[wgx + 2],
        )?;
    }
    // Domain guards before any data-memory access (uniform exit).
    // row >= M, k_top <= 0, rank >= k_top, K <= 0, N out of 1..=64.
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_le_i32 s{SKT}, 0"), &[], &[SKT])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{RANK}, s{SKT}"),
        &[],
        &[RANK, SKT],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_le_i32 s{SK}, 0"), &[], &[SK])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_le_i32 s{SN}, 0"), &[], &[SN])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{SN}, {}", u32::from(MAX_ROWS)),
        &[],
        &[SN],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("s_lshl_b32 s{TBASE}, s{TILE}, 3"),
        &[TBASE],
        &[TILE],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{TBASE}, s{SN}"),
        &[],
        &[TBASE, SN],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    // groups = ceil(K/128); ragged class = K & 127 (0: whole-K fast path).
    sop(
        b,
        format!("{} s{TMP}, s{SK}, {}", s_add_i32(arch), lit(127)),
        &[TMP],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{TMP}, 7"),
        &[GROUPS],
        &[TMP],
    )?;
    sop(
        b,
        format!("s_and_b32 s{GTAIL}, s{SK}, {}", lit(127)),
        &[GTAIL],
        &[SK],
    )?;
    // Per-lane constants: tid, weight byte offset (tid*2).
    valu(
        b,
        &format!("v_and_b32_e32 v{LANE}, 31, v{LANE}"),
        &[LANE],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{VO}, 1, v{LANE}"),
        &[VO],
        &[LANE],
        &[],
    )?;
    // Each physical token executes the original one-token body uniformly.
    // A dead token branches before even the index load; an invalid expert
    // branches directly to reduction/store with the initial positive zero.
    for slot in 0..TILE_ROWS {
        let next = format!(".Ldown_t8_next_{slot}");
        let reduce = format!(".Ldown_t8_reduce_{slot}");
        let whole = format!(".Ldown_t8_whole_{slot}");
        let tail = format!(".Ldown_t8_ragged_{slot}");
        let group_w = format!(".Ldown_t8_group_w_{slot}");
        let group_r = format!(".Ldown_t8_group_r_{slot}");
        sop(
            b,
            format!("{} s{TMP}, s{TBASE}, {slot}", s_add_i32(arch)),
            &[TMP],
            &[TBASE],
        )?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SN}"), &[], &[TMP, SN])?;
        let dead_ledger = b.ledger.clone();
        let dead_sgpr = b.hazard.clone();
        let dead_trans = b.gfx11_hazard.clone();
        branch(b, &format!("s_cbranch_scc1 {next}"))?;
        valu(b, &format!("v_mov_b32_e32 v{ACC}, 0"), &[ACC], &[], &[])?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{FLAT}, s{TMP}"),
            &[FLAT],
            &[],
            &[TMP],
        )?;
        valu(
            b,
            &format!("v_mul_lo_u32 v{FLAT}, v{FLAT}, s{SKT}"),
            &[FLAT],
            &[FLAT],
            &[SKT],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{FLAT}, s{RANK}, v{FLAT}"),
            &[FLAT],
            &[FLAT],
            &[RANK],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{VA}, 2, v{FLAT}"),
            &[VA],
            &[FLAT],
            &[],
        )?;
        gload(b, "global_load_b32", EID, VA, karg(arch).iptr, "")?;
        b.wait_all()?;
        b.push(Instruction::new(
            format!("v_readfirstlane_b32 s{TMP}, v{EID}"),
            vec![sreg(TMP, 1)],
            vec![vreg(EID, 1)],
        ))?;
        sop(b, format!("s_cmp_lt_i32 s{TMP}, 0"), &[], &[TMP])?;
        let negative_ledger = b.ledger.clone();
        let negative_sgpr = b.hazard.clone();
        let negative_trans = b.gfx11_hazard.clone();
        branch(b, &format!("s_cbranch_scc1 {reduce}"))?;
        sop(
            b,
            format!("s_cmp_ge_i32 s{TMP}, s{SNEXP}"),
            &[],
            &[TMP, SNEXP],
        )?;
        let invalid_ledger = b.ledger.clone();
        let invalid_sgpr = b.hazard.clone();
        let invalid_trans = b.gfx11_hazard.clone();
        branch(b, &format!("s_cbranch_scc1 {reduce}"))?;
        broadcast_expert(b, EID)?;
        sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
        sop(b, format!("s_cmp_eq_u32 s{GTAIL}, 0"), &[], &[GTAIL])?;
        // Both K-class arms inherit GI's initialization. Emitting the whole
        // arm must not clear the producer hazard on the ragged entry edge.
        let ragged_ledger = b.ledger.clone();
        let ragged_sgpr = b.hazard.clone();
        let ragged_trans = b.gfx11_hazard.clone();
        branch(b, &format!("s_cbranch_scc1 {whole}"))?;
        branch(b, &format!("s_branch {tail}"))?;
        b.label(&whole)?;
        b.loop_(&group_w, |b| {
            group_body(b, arch, false)?;
            sop(
                b,
                format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
                &[GI],
                &[GI],
            )?;
            sop(
                b,
                format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                &[],
                &[GI, GROUPS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {group_w}"))
        })?;
        branch(b, &format!("s_branch {reduce}"))?;
        let whole_ledger = b.ledger.clone();
        let whole_sgpr = b.hazard.clone();
        let whole_trans = b.gfx11_hazard.clone();
        b.label(&tail)?;
        b.ledger.resume(ragged_ledger);
        b.hazard = ragged_sgpr;
        b.gfx11_hazard = ragged_trans;
        b.loop_(&group_r, |b| {
            group_body(b, arch, true)?;
            sop(
                b,
                format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
                &[GI],
                &[GI],
            )?;
            sop(
                b,
                format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                &[],
                &[GI, GROUPS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {group_r}"))
        })?;
        b.label(&reduce)?;
        // Reduction is also reached directly by either invalid-expert guard.
        b.ledger.join(&whole_ledger);
        b.join_hazards(&whole_sgpr, &whole_trans, &[]);
        b.ledger.join(&negative_ledger);
        b.join_hazards(&negative_sgpr, &negative_trans, &[]);
        b.ledger.join(&invalid_ledger);
        b.join_hazards(&invalid_sgpr, &invalid_trans, &[]);
        for (k, off) in [16u32, 8, 4, 2, 1].into_iter().enumerate() {
            let (m, sh) = (SHM + 2 * k as u8, SHUF + k as u8);
            b.push(Instruction::new(
                format!("v_cmp_gt_u32_e64 s{m}, {}, v{LANE}", 32 - off),
                vec![sreg(m, 2)],
                vec![vreg(LANE, 1)],
            ))?;
            valu(
                b,
                &format!("v_cndmask_b32_e64 v{sh}, 0, {off}, s{m}"),
                &[sh],
                &[],
                &[m],
            )?;
            valu(
                b,
                &format!("v_add_lshl_u32 v{sh}, v{sh}, v{LANE}, 2"),
                &[sh],
                &[sh, LANE],
                &[],
            )?;
        }
        let acc = ACC;
        for k in 0..5u8 {
            let (m, sh) = (SHM + 2 * k, SHUF + k);
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{EXCH}, v{sh}, v{acc}"),
                    vec![vreg(EXCH, 1)],
                    vec![vreg(sh, 1), vreg(acc, 1)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{EXCH}, v{acc}, v{EXCH}"),
                &[EXCH],
                &[acc, EXCH],
                &[],
            )?;
            valu(
                b,
                &format!("v_cndmask_b32_e64 v{acc}, v{acc}, v{EXCH}, s{m}"),
                &[acc],
                &[acc, EXCH],
                &[m],
            )?;
        }
        valu(
            b,
            &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
            &[],
            &[LANE],
            &[],
        )?;
        let fl = FLAT;
        // yoff = flat * M + row (bytes).
        valu(
            b,
            &format!("v_mul_lo_u32 v{VA}, v{fl}, s{SM}"),
            &[VA],
            &[fl],
            &[SM],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{VA}, s{ROW}, v{VA}"),
            &[VA],
            &[VA],
            &[ROW],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{VA}, 2, v{VA}"),
            &[VA],
            &[VA],
            &[],
        )?;
        let yptr = karg(arch).yptr;
        b.push(
            Instruction::new(
                format!("global_store_b32 v{VA}, v{acc}, s[{yptr}:{}]", yptr + 1),
                vec![],
                vec![vreg(VA, 1), vreg(acc, 1), sreg(yptr, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
        raw(b, "s_mov_b32 exec_lo, -1")?;
        b.label(&next)?;
        // A dead token bypasses all loads, reduction, and the store.
        b.ledger.join(&dead_ledger);
        b.join_hazards(&dead_sgpr, &dead_trans, &[]);
    }
    b.wait_all()?;
    b.label(end)?;
    branch(b, "s_endpgm")
}

/// One ascending group for the current live token and valid expert.
/// Header halves widen from f16; ragged X loads are individually EXEC-guarded.
fn group_body(b: &mut Builder, arch: Arch, ragged: bool) -> R {
    let (acc, fl) = (ACC, FLAT);
    // Non-inline group strides use the native SOP2 hex-literal spelling.
    sop(
        b,
        format!("s_mul_i32 s{TMP2}, s{GI}, {}", lit(68)),
        &[TMP2],
        &[GI],
    )?;
    // row_ptr = base + row * groups * 68; group_ptr = row_ptr + GI * 68.
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{GROUPS}, {}", lit(68)),
        &[TMP],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{TMP}, s{ROW}"),
        &[TMP],
        &[TMP, ROW],
    )?;
    sop(
        b,
        format!("{} s{TMP}, s{TMP}, s{TMP2}", s_add_i32(arch)),
        &[TMP],
        &[TMP, TMP2],
    )?;
    // header scale/zero (f16 halves, HIP `__half2float`, exact widening).
    valu(
        b,
        &format!("v_mov_b32_e32 v{VA}, s{TMP}"),
        &[VA],
        &[],
        &[TMP],
    )?;
    gload(b, "global_load_b32", VB, VA, EB0, "")?;
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{HD}, v{VB}.l"),
        &[HD],
        &[VB],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{HZ}, 16, v{VB}"),
        &[HZ],
        &[VB],
        &[],
    )?;
    valu(
        b,
        &format!("v_cvt_f32_f16_e64 v{HZ}, v{HZ}.l"),
        &[HZ],
        &[HZ],
        &[],
    )?;
    // payload bytes b0 = payload[tid*2], b1 = payload[tid*2+1].
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{VA}, 4, v{VA}"),
        &[VA],
        &[VA],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{VA}, v{VA}, v{VO}"),
        &[VA],
        &[VA, VO],
        &[],
    )?;
    gload(b, "global_load_u16", VB, VA, EB0, "")?;
    // nibbles -> weights w = fma(scale, nibble_f32, zero).
    for i in 0..4u8 {
        let w = W0 + i;
        valu(
            b,
            &format!("v_bfe_u32 v{w}, v{VB}, {}, 4", u32::from(i) * 4),
            &[w],
            &[VB],
            &[],
        )?;
        valu(b, &format!("v_cvt_f32_u32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
        valu(
            b,
            &format!("v_fma_f32 v{w}, v{HD}, v{w}, v{HZ}"),
            &[w],
            &[HD, w, HZ],
            &[],
        )?;
    }
    // x base = flat * K + GI * 128; lanes load x[base + 4*tid .. +3].
    valu(
        b,
        &format!("v_mul_lo_u32 v{VB}, v{fl}, s{SK}"),
        &[VB],
        &[fl],
        &[SK],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e64 v{VA}, 7, s{GI}"),
        &[VA],
        &[],
        &[GI],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{VA}, v{VB}, v{VA}"),
        &[VA],
        &[VB, VA],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{VB}, 2, v{LANE}"),
        &[VB],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{VA}, v{VB}, v{VA}"),
        &[VA],
        &[VA, VB],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{VA}, 2, v{VA}"),
        &[VA],
        &[VA],
        &[],
    )?;
    if ragged {
        // Compare group-local K, not flat*K + the group-local coordinate.
        valu(
            b,
            &format!("v_lshlrev_b32_e64 v{KIDX}, 7, s{GI}"),
            &[KIDX],
            &[],
            &[GI],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{KIDX}, v{KIDX}, v{VB}"),
            &[KIDX],
            &[KIDX, VB],
            &[],
        )?;
        for i in 0..4u8 {
            let x = X0 + i;
            valu(b, &format!("v_mov_b32_e32 v{x}, 0"), &[x], &[], &[])?;
            valu(
                b,
                &format!("v_cmp_gt_u32_e32 vcc_lo, s{SK}, v{KIDX}"),
                &[],
                &[KIDX],
                &[SK],
            )?;
            raw(b, "s_and_b32 exec_lo, exec_lo, vcc_lo")?;
            gload(b, "global_load_b32", x, VA, karg(b.spec.arch).xptr, "")?;
            b.wait_all()?;
            raw(b, "s_mov_b32 exec_lo, -1")?;
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{KIDX}, 1, v{KIDX}"),
                &[KIDX],
                &[KIDX],
                &[],
            )?;
            if i < 3 {
                valu(
                    b,
                    &format!("v_add_nc_u32_e32 v{VA}, 4, v{VA}"),
                    &[VA],
                    &[VA],
                    &[],
                )?;
            }
        }
    } else {
        gload_b128(b, X0, VA, karg(b.spec.arch).xptr)?;
    }
    // Actual indexed AR contracts the first product into the second:
    // d = x1*w1; d = fma(x0, w0, d); then fma terms 2 and 3.
    valu(
        b,
        &format!("v_mul_f32_e32 v{DOT}, v{}, v{}", X0 + 1, W0 + 1),
        &[DOT],
        &[X0 + 1, W0 + 1],
        &[],
    )?;
    for i in [0u8, 2, 3] {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{DOT}, v{}, v{}", X0 + i, W0 + i),
            &[DOT],
            &[DOT, X0 + i, W0 + i],
            &[],
        )?;
    }
    valu(
        b,
        &format!("v_add_f32_e32 v{acc}, v{acc}, v{DOT}"),
        &[acc],
        &[acc, DOT],
        &[],
    )?;
    Ok(())
}

/// Broadcast one slot's expert row pointer to SGPRs: all lanes hold the same
/// expert id, so the table load is uniform and the two pointer words are read
/// back with `v_readfirstlane_b32`.
fn broadcast_expert(b: &mut Builder, eid: u8) -> R {
    let pptr = karg(b.spec.arch).pptr;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{VA}, 3, v{eid}"),
        &[VA],
        &[eid],
        &[],
    )?;
    b.push(
        Instruction::new(
            format!(
                "global_load_b64 v[{VB}:{}], v{VA}, s[{pptr}:{}]",
                VB + 1,
                pptr + 1
            ),
            vec![vreg(VB, 2)],
            vec![vreg(VA, 1), sreg(pptr, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )?;
    b.wait_all()?;
    b.push(Instruction::new(
        format!("v_readfirstlane_b32 s{EB0}, v{VB}"),
        vec![sreg(EB0, 1)],
        vec![vreg(VB, 1)],
    ))?;
    b.push(Instruction::new(
        format!("v_readfirstlane_b32 s{EB1}, v{}", VB + 1),
        vec![sreg(EB1, 1)],
        vec![vreg(VB + 1, 1)],
    ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn down_t8_source_oracle_abi() {
        // Source-only oracle: ABI/geometry contract the consumer binds.
        assert_eq!(KERNARG_BYTES, 52);
        let l = kernargs();
        let names: Vec<(&str, u32)> = l.args.iter().map(|a| (a.name.as_str(), a.offset)).collect();
        assert_eq!(
            names,
            [
                ("expert_ptrs", 0),
                ("topk_indices", 8),
                ("hidden_batch", 16),
                ("expert_outputs", 24),
                ("M", 32),
                ("K", 36),
                ("N", 40),
                ("n_exp", 44),
                ("k_top", 48),
            ]
        );
        assert_eq!(l.args[3].value_kind, "global_buffer");
        assert_eq!(l.args[3].actual_access.as_deref(), Some("write_only"));
        assert_eq!(TILE_ROWS, 8);
        assert_eq!(MAX_ROWS, 64);
        for arch in [Arch::Gfx1201, Arch::Gfx1151, Arch::Gfx1100] {
            assert_eq!(
                symbol(arch),
                format!(
                    "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
                    arch.name()
                )
            );
            assert_eq!(
                module(arch),
                format!("legacy_mq4_rowtiles_pm_{}", arch.name())
            );
        }
    }
    #[test]
    fn down_t8_domain_contract() {
        // Logical R domain 1..=64, physical tile 8; M stays the output dim.
        assert_eq!(u32::from(MAX_ROWS), 64);
        assert_eq!(u32::from(TILE_ROWS), 8);
        assert_eq!((u32::from(MAX_ROWS) + 7) / 8, 8);
        // Geometry-only assertions, not an execution/rounding oracle.
        for n in 1..=u32::from(MAX_ROWS) {
            let mut live = Vec::new();
            for tile in 0..(n + 7) / 8 {
                for slot in 0..8 {
                    let token = 8 * tile + slot;
                    if token < n {
                        live.push(token);
                    }
                }
            }
            assert_eq!(live, (0..n).collect::<Vec<_>>());
            let m = 2560;
            let rank = 9;
            assert_eq!(((n - 1) * 10 + rank) * m + (m - 1), n * 10 * m - 1);
        }
    }
}
