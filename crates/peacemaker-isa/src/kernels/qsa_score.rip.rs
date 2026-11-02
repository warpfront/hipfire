// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1151 QSA selector scores
//! (`indexed_attention_select_scores_rows16_f32_pm_gfx1151`): the kernarg ABI,
//! grid (`[ceil(block_count/256), ceil(rows/16), 1]`), block (256, eight
//! waves) and output bytes of the hipcc twin
//! `indexed_attention_select_scores_rows16_f32` in `kernels/src/tensor_ops.hip`
//! (host contract: `index_heads = 4`, `index_dim = 128`):
//! The BF16-pooled variant (`indexed_attention_select_scores_rows16_bf16_pm_gfx1151`)
//! has the same ABI and arithmetic, with exact BF16-to-F32 bit widening.
//! The R32 twins (`indexed_attention_select_scores_rows32_{f32,bf16}_pm_gfx1151`)
//! are the same builder with the row tile parameterized: the workgroup's
//! first row is `32 * wg_y` (not `16 * wg_y`), its row count clamps to 32
//! (not 16), and the grid is `[ceil(block_count/256), ceil(rows/32), 1]`.
//! The key load, four-chain `row()` body, division sequence, query lookahead,
//! store descriptor and resources are unchanged, so a row's score bits do not
//! depend on the tile size.
//!
//! ```text
//! scores[(row0 + r) * score_stride + block] =
//!     (((0 + max(0, dot[r][0] / sqrtf(128))) + max(0, dot[r][1] / sqrtf(128)))
//!       + max(0, dot[r][2] / sqrtf(128))) + max(0, dot[r][3] / sqrtf(128))
//! dot[r][h] = fma(q[r][h][127], key[127], .. fma(q[r][h][1], key[1], fma(q[r][h][0], key[0], +0.0)) ..)
//! ```
//!
//! for `block = 256 * wg_x + tid < min(block_count, (position_start + row0 +
//! r + 1) / compress)` and `r < min(R, rows - row0)` (`R` = 16, or 32 for the
//! rows32 symbols); every other score is never touched.
//!
//! Schedule (the incumbent is LDS-bound: one broadcast `ds_load_b128` per four
//! FMAs and 188 VGPRs). This port keeps no LDS at all:
//! - every lane loads its whole pooled key row (128 F32, 32 `buffer_load_b128`
//!   in one clause) once into 128 VGPRs. The descriptor is based at the
//!   workgroup's first block with `num_records = min(block_count - 256 *
//!   wg_x, 256) * 512`, so a lane with `block >= block_count` reads 0 and
//!   touches no memory;
//!   BF16 uses 256-byte rows and 16 loads into the upper 64 key VGPRs;
//!   after `vmcnt(0)`, ascending in-place unpack shifts each low half by 16
//!   and masks each high half with `0xffff0000`, giving the same 128 F32 keys;
//! - the query is workgroup-uniform, so it streams through SGPRs: a row is 16
//!   stages of 8 dims, each four `s_load_b256` (one per head, 32 SGPRs) into a
//!   double-buffered 64-SGPR ring. SMEM loads return out of order, so every
//!   wait is `lgkmcnt(0)`: a stage's loads are issued right after the previous
//!   stage's wait and land while that stage's 32 FMAs run; the next row's
//!   first stage is fetched before the row's store drains. Every FMA is one
//!   `v_fmac_f32 acc, s_q, v_key` (the first of a chain `v_fma_f32 acc, s_q,
//!   v_key, 0`), four independent head chains interleaved per dim;
//! - per row the four chains give `dot[r][0..4]` in 4 accumulator VGPRs (rows
//!   are independent, so no row's accumulators outlive its epilogue); the
//!   epilogue is hipcc's: four IEEE divisions by `sqrtf(128)` (two interleaved
//!   `v_div_scale` / `v_rcp_f32` / fma / `v_div_fmas` / `v_div_fixup`
//!   sequences, the divisor in a VGPR as in hipcc), `v_max_f32 x, 0, x`, then
//!   the in-order sum starting `v_add_f32 s, 0, m0`;
//! - the store goes through a per-row raw buffer descriptor based at
//!   `scores + ((row0 + r) * score_stride + 256 * wg_x) * 4` with
//!   `num_records = 4 * max(0, min(block_count, visible_blocks(r)) - 256 *
//!   wg_x)`, so a lane at or past the row's visible range stores nothing and
//!   no lane masks or branches are needed. `visible_blocks(r)` is tracked
//!   incrementally from one scalar `n / compress` at entry (`n % compress`
//!   plus one per row), exact for `n >= 1`, `compress >= 1`;
//! - rows are a wave-uniform loop over `nrows = min(R, rows - row0)` (no
//!   unrolling: ragged row tiles read no query memory beyond their rows; the
//!   one-row-ahead stage fetch of the last row re-reads the row it is on). A
//!   wave whose first block is at or past `block_count` returns before any
//!   load.
//!
//! Numerics, identical to the hipcc kernel: each `(r, h)` chain is the same
//! `d = 0..127` in-order sequence of single-rounding fused multiply-adds from
//! `+0.0` (no split-K, no reassociation); the divisor is `sqrtf(128.0f)`, the
//! correctly rounded `0x413504f3` (hipcc computes the same value at run time
//! from `index_dim` with `v_sqrt_f32` and its residual correction); the
//! division, `v_max_f32` operand order and the sum order are hipcc's
//! instruction sequence.
//!
//! Resources: 152 VGPRs (128 key + 4 accumulators + 10 division temporaries +
//! address/constant registers, padded to the 8-register granule), 103 SGPRs
//! plus VCC, zero LDS, zero scratch, no atomics.
use super::common::{add64, bload, lit, mem, op, s, smem, sop, sr, v, SRD_WORD3};
use crate::{
    insn::{Instruction, MemoryClass},
    ledger::Counter,
    reg::Live,
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1151, Wave, Workgroup};

pub fn symbol(arch: Arch) -> String {
    format!(
        "indexed_attention_select_scores_rows16_f32_pm_{}",
        arch.name()
    )
}
pub fn symbol_bf16(arch: Arch) -> String {
    format!(
        "indexed_attention_select_scores_rows16_bf16_pm_{}",
        arch.name()
    )
}
pub fn symbol32(arch: Arch) -> String {
    format!(
        "indexed_attention_select_scores_rows32_f32_pm_{}",
        arch.name()
    )
}
pub fn symbol_bf16_rows32(arch: Arch) -> String {
    format!(
        "indexed_attention_select_scores_rows32_bf16_pm_{}",
        arch.name()
    )
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Pooled {
    F32,
    Bf16,
}
impl Pooled {
    fn row_shift(self) -> u32 {
        match self {
            Self::F32 => 9,
            Self::Bf16 => 8,
        }
    }
    fn load_count(self) -> u8 {
        match self {
            Self::F32 => 32,
            Self::Bf16 => 16,
        }
    }
    fn packed_base(self) -> u8 {
        match self {
            Self::F32 => KEYS,
            Self::Bf16 => KEYS + 64,
        }
    }
}

/// `sqrtf(128.0f)` correctly rounded (`128f32.sqrt().to_bits()`).
const SQRT_INDEX_DIM: u32 = 0x4135_04f3;
/// Query rows per workgroup: the only geometry that differs between the R16
/// and R32 symbols.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Tile {
    R16,
    R32,
}
impl Tile {
    fn rows(self) -> i32 {
        match self {
            Self::R16 => 16,
            Self::R32 => 32,
        }
    }
    /// `log2(rows)`: the shift from `wg_y` to the tile's first row.
    fn shift(self) -> u32 {
        match self {
            Self::R16 => 4,
            Self::R32 => 5,
        }
    }
    /// Assembler labels are module-global: the R16 labels are the incumbent's,
    /// the R32 kernels (same module) carry their own.
    fn end_label(self) -> &'static str {
        match self {
            Self::R16 => ".Lqsc_end",
            Self::R32 => ".Lqsc32_end",
        }
    }
    fn rows_label(self) -> &'static str {
        match self {
            Self::R16 => ".Lqsc_rows",
            Self::R32 => ".Lqsc32_rows",
        }
    }
    fn done_label(self) -> &'static str {
        match self {
            Self::R16 => ".Lqsc_rows_done",
            Self::R32 => ".Lqsc32_rows_done",
        }
    }
}
/// Index heads and dims per head (the host contract).
const HEADS: usize = 4;
const DIM: usize = 128;
/// Dims per query stage (`s_load_b256`) and stages per row.
const CHUNK: usize = 8;
const STAGES: usize = DIM / CHUNK;
/// Blocks per workgroup tile.
const TILE_BLOCKS: u32 = 256;

// SGPRs.
const KARG: u8 = 0;
const WGX: u8 = 2;
const WGY: u8 = 3;
const SRD_S: u8 = 4; // scores descriptor 4:7
const A0: u8 = 8; // query 8:9, pooled 10:11, scores 12:13, rows 14, query_row_stride 15
const QUERY: u8 = 8;
const POOLED: u8 = 10;
const SCORES: u8 = 12;
const ROWS_ARG: u8 = 14;
const QSTRIDE: u8 = 15;
const A1: u8 = 16; // block_count 16, index_dim 17 (unused), compress 18, position_start 19
const BLOCK_COUNT: u8 = 16;
const COMPRESS: u8 = 18;
const POSITION: u8 = 19;
const SRD_K: u8 = 20; // pooled key descriptor 20:23
const QCUR: u8 = 24; // query row cursor 24:25
const SCSTRIDE: u8 = 26;
const ROW0: u8 = 27;
const NROWS: u8 = 28;
const ROW: u8 = 29;
const QUOT: u8 = 30;
const REM: u8 = 31;
const RING: u8 = 32; // 2 stages x 4 heads x 8 SGPRs, 32:95
const T0: u8 = 96;
const T1: u8 = 97;
const T2: u8 = 98;
const T3: u8 = 99;
const MASK: u8 = 100;
const BLK0: u8 = 101;
const WAVE: u8 = 102;

// VGPRs.
const TID: u8 = 0;
const SOFF: u8 = 1;
const DIV: u8 = 2;
const KOFF: u8 = 3;
const ACC: u8 = 4; // dot[h], h = 0..4
const KEYS: u8 = 8; // key[d] = v(8 + d)
const DT: u8 = 136; // division temporaries: ns, ds, rc, e, q per interleaved pair half

fn kernargs() -> KernargLayout {
    KernargLayout::new(52)
        .pointer("query", 0)
        .pointer("pooled", 8)
        .pointer("scores", 16)
        .hidden("rows", 24, 4, "by_value")
        .hidden("query_row_stride", 28, 4, "by_value")
        .hidden("block_count", 32, 4, "by_value")
        .hidden("index_dim", 36, 4, "by_value")
        .hidden("compress", 40, 4, "by_value")
        .hidden("position_start", 44, 4, "by_value")
        .hidden("score_stride", 48, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<1>("wg_x_in", WGX, Live::Whole)?;
    p.s::<1>("wg_y_in", WGY, Live::Whole)?;
    p.s::<4>("srd_scores", SRD_S, Live::Whole)?;
    p.s::<8>("kernargs_0x00", A0, Live::Whole)?;
    p.s::<4>("kernargs_0x20", A1, Live::Whole)?;
    p.s::<4>("srd_keys", SRD_K, Live::Whole)?;
    p.s::<2>("query_cursor", QCUR, Live::Whole)?;
    for (name, r) in [
        ("score_stride", SCSTRIDE),
        ("row0", ROW0),
        ("nrows", NROWS),
        ("row", ROW),
        ("visible_quot", QUOT),
        ("visible_rem", REM),
        ("t0", T0),
        ("t1", T1),
        ("t2", T2),
        ("t3", T3),
        ("div_mask", MASK),
        ("blk0", BLK0),
        ("wave", WAVE),
    ] {
        p.s::<1>(name, r, Live::Whole)?;
    }
    for i in 0..8u8 {
        p.s::<8>("query_ring", RING + 8 * i, Live::Whole)?;
    }
    for (name, r) in [
        ("tid", TID),
        ("score_off", SOFF),
        ("sqrt_dim", DIV),
        ("key_off", KOFF),
    ] {
        p.v::<1>(name, r, Live::Whole)?;
    }
    p.v::<4>("dot", ACC, Live::Whole)?;
    for i in 0..16u8 {
        p.v::<8>("key", KEYS + 8 * i, Live::Whole)?;
    }
    p.v::<8>("div_t0", DT, Live::Whole)?;
    p.v::<2>("div_t1", DT + 8, Live::Whole)?;
    let top = p.next_free_vgpr();
    if top % 8 != 0 {
        p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
    }
    Ok(p)
}

pub fn emit(arch: Arch) -> Result<Emitted, String> {
    emit_pooled(arch, Pooled::F32, Tile::R16)
}

pub fn emit_bf16(arch: Arch) -> Result<Emitted, String> {
    emit_pooled(arch, Pooled::Bf16, Tile::R16)
}

/// The rows32 F32-pooled twin of [`emit`].
pub fn emit32(arch: Arch) -> Result<Emitted, String> {
    emit_pooled(arch, Pooled::F32, Tile::R32)
}

/// The rows32 BF16-pooled twin of [`emit_bf16`].
pub fn emit_bf16_rows32(arch: Arch) -> Result<Emitted, String> {
    emit_pooled(arch, Pooled::Bf16, Tile::R32)
}

fn emit_pooled(arch: Arch, pooled: Pooled, tile: Tile) -> Result<Emitted, String> {
    if arch != Arch::Gfx1151 {
        return Err("qsa_score: built for gfx1151 only".into());
    }
    let (variant, symbol) = match (pooled, tile) {
        (Pooled::F32, Tile::R16) => ("score", symbol(arch)),
        (Pooled::Bf16, Tile::R16) => ("score-bf16", symbol_bf16(arch)),
        (Pooled::F32, Tile::R32) => ("score32", symbol32(arch)),
        (Pooled::Bf16, Tile::R32) => ("score-bf16-rows32", symbol_bf16_rows32(arch)),
    };
    let spec = KernelSpec {
        kernel_id: "qsa_select".into(),
        variant: variant.into(),
        arch,
        symbol,
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: 256,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(spec, plan()?);
    author(
        &mut Workgroup::<Gfx1151, Builder>::new(&mut b)?,
        pooled,
        tile,
    )?;
    b.finish()
}

type Wg<'a, 'b> = &'a mut Workgroup<'b, Gfx1151, Builder>;

fn scmp(
    w: &mut Wave<'_, Gfx1151, Builder>,
    text: String,
    uses: &[u8],
) -> Result<peacemaker_author::Uniform<peacemaker_author::Scc>, String> {
    w.scmp(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

fn author(wg: Wg, pooled: Pooled, tile: Tile) -> Result<(), String> {
    let end = wg.exit(tile.end_label())?;
    let b = wg.isa();
    smem(b, A0, 8, KARG, 0)?;
    smem(b, A1, 4, KARG, 0x20)?;
    smem(b, SCSTRIDE, 1, KARG, 0x30)?;
    // Lane constants: key row byte offset, score byte offset, the divisor and this wave's index.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{KOFF}, {}, v{TID}", pooled.row_shift()),
        &[v(KOFF)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{SOFF}, 2, v{TID}"),
        &[v(SOFF)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{DIV}, {}", lit(SQRT_INDEX_DIM)),
        &[v(DIV)],
        &[],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{ACC}, 5, v{TID}"),
        &[v(ACC)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v{ACC}"),
        &[s(WAVE)],
        &[v(ACC)],
    )?;
    sop(b, format!("s_lshl_b32 s{BLK0}, s{WGX}, 8"), &[BLK0], &[WGX])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 5"), &[T0], &[WAVE])?;
    sop(
        b,
        format!("s_add_i32 s{T0}, s{T0}, s{BLK0}"),
        &[T0],
        &[T0, BLK0],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{ROW0}, s{WGY}, {}", tile.shift()),
        &[ROW0],
        &[WGY],
    )?;
    // A wave entirely past block_count, or a row tile past the rows, returns.
    let live = scmp(
        wg,
        format!("s_cmp_lt_i32 s{T0}, s{BLOCK_COUNT}"),
        &[T0, BLOCK_COUNT],
    )?;
    Wave::<Gfx1151, Builder>::exit_unless(wg, live, &end)?;
    let b = wg.isa();
    sop(
        b,
        format!("s_sub_i32 s{NROWS}, s{ROWS_ARG}, s{ROW0}"),
        &[NROWS],
        &[ROWS_ARG, ROW0],
    )?;
    sop(
        b,
        format!("s_min_i32 s{NROWS}, s{NROWS}, {}", tile.rows()),
        &[NROWS],
        &[NROWS],
    )?;
    let some = scmp(wg, format!("s_cmp_gt_i32 s{NROWS}, 0"), &[NROWS])?;
    Wave::<Gfx1151, Builder>::exit_unless(wg, some, &end)?;
    let b = wg.isa();
    // Pooled key descriptor: this workgroup's first block, at most 256 blocks of records.
    sop(
        b,
        format!("s_lshl_b32 s{T1}, s{WGX}, {}", pooled.row_shift() + 8),
        &[T1],
        &[WGX],
    )?;
    sop(
        b,
        format!("s_add_u32 s{SRD_K}, s{POOLED}, s{T1}"),
        &[SRD_K],
        &[POOLED, T1],
    )?;
    sop(
        b,
        format!("s_addc_u32 s{}, s{}, 0", SRD_K + 1, POOLED + 1),
        &[SRD_K + 1],
        &[POOLED + 1],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{T1}, s{BLOCK_COUNT}, s{BLK0}"),
        &[T1],
        &[BLOCK_COUNT, BLK0],
    )?;
    sop(
        b,
        format!("s_min_i32 s{T1}, s{T1}, {}", lit(TILE_BLOCKS)),
        &[T1],
        &[T1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{T1}, {}", SRD_K + 2, pooled.row_shift()),
        &[SRD_K + 2],
        &[T1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_K + 3, lit(SRD_WORD3)),
        &[SRD_K + 3],
        &[],
    )?;
    // Every lane's whole key row, in one clause.
    b.clause(|b| {
        for i in 0..pooled.load_count() {
            bload(
                b,
                4,
                pooled.packed_base() + 4 * i,
                KOFF,
                SRD_K,
                16 * u32::from(i),
            )?;
        }
        Ok(())
    })?;
    // Query row cursor and the scores descriptor, based at row0 and this tile's first block.
    sop(
        b,
        format!("s_lshl_b32 s{QSTRIDE}, s{QSTRIDE}, 2"),
        &[QSTRIDE],
        &[QSTRIDE],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{ROW0}, s{QSTRIDE}"),
        &[T1],
        &[ROW0, QSTRIDE],
    )?;
    sop(
        b,
        format!("s_add_u32 s{QCUR}, s{QUERY}, s{T1}"),
        &[QCUR],
        &[QUERY, T1],
    )?;
    sop(
        b,
        format!("s_addc_u32 s{}, s{}, 0", QCUR + 1, QUERY + 1),
        &[QCUR + 1],
        &[QUERY + 1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SCSTRIDE}, s{SCSTRIDE}, 2"),
        &[SCSTRIDE],
        &[SCSTRIDE],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{ROW0}, s{SCSTRIDE}"),
        &[T1],
        &[ROW0, SCSTRIDE],
    )?;
    sop(b, format!("s_lshl_b32 s{T2}, s{BLK0}, 2"), &[T2], &[BLK0])?;
    sop(
        b,
        format!("s_add_i32 s{T1}, s{T1}, s{T2}"),
        &[T1],
        &[T1, T2],
    )?;
    sop(
        b,
        format!("s_add_u32 s{SRD_S}, s{SCORES}, s{T1}"),
        &[SRD_S],
        &[SCORES, T1],
    )?;
    sop(
        b,
        format!("s_addc_u32 s{}, s{}, 0", SRD_S + 1, SCORES + 1),
        &[SRD_S + 1],
        &[SCORES + 1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    // The first stage of row 0 (the loop keeps one stage fetched ahead).
    issue_stage(b, 0)?;
    // Visible blocks of row 0: (position_start + row0 + 1) / compress and its remainder.
    sop(
        b,
        format!("s_add_i32 s{T0}, s{POSITION}, s{ROW0}"),
        &[T0],
        &[POSITION, ROW0],
    )?;
    sop(b, format!("s_add_i32 s{T0}, s{T0}, 1"), &[T0], &[T0])?;
    udiv(b, QUOT, T0, COMPRESS, [ACC, ACC + 1, ACC + 2], [T1, T2, T3])?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{QUOT}, s{COMPRESS}"),
        &[T1],
        &[QUOT, COMPRESS],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{REM}, s{T0}, s{T1}"),
        &[REM],
        &[T0, T1],
    )?;
    set_visible(b)?;
    sop(b, format!("s_mov_b32 s{ROW}, 0"), &[ROW], &[])?;
    b.wait(Counter::Vm, 0)?;
    if pooled == Pooled::Bf16 {
        // Ascending unpack cannot overwrite a future packed source. The last
        // high destination aliases its source; emit low before high.
        for i in 0..64u8 {
            let packed = pooled.packed_base() + i;
            let lo = KEYS + 2 * i;
            let hi = lo + 1;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{lo}, 16, v{packed}"),
                &[v(lo)],
                &[v(packed)],
            )?;
            op(
                b,
                format!("v_and_b32_e32 v{hi}, 0xffff0000, v{packed}"),
                &[v(hi)],
                &[v(packed)],
            )?;
        }
    }
    wg.loop_until(tile.rows_label(), tile.done_label(), |w, exit| {
        row(w.isa())?;
        let b = w.isa();
        // Visible blocks of the next row: remainder + 1, carrying into the quotient.
        sop(b, format!("s_add_i32 s{REM}, s{REM}, 1"), &[REM], &[REM])?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{REM}, s{COMPRESS}"),
            &[],
            &[REM, COMPRESS],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{T0}, s{COMPRESS}, 0"),
            &[T0],
            &[COMPRESS],
        )?;
        sop(b, format!("s_cselect_b32 s{T1}, 1, 0"), &[T1], &[])?;
        sop(
            b,
            format!("s_sub_i32 s{REM}, s{REM}, s{T0}"),
            &[REM],
            &[REM, T0],
        )?;
        sop(
            b,
            format!("s_add_i32 s{QUOT}, s{QUOT}, s{T1}"),
            &[QUOT],
            &[QUOT, T1],
        )?;
        // The query cursor moves to the next row; the last row's lookahead stays on it.
        sop(b, format!("s_add_i32 s{ROW}, s{ROW}, 1"), &[ROW], &[ROW])?;
        sop(
            b,
            format!("s_cmp_lt_u32 s{ROW}, s{NROWS}"),
            &[],
            &[ROW, NROWS],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{T0}, s{QSTRIDE}, 0"),
            &[T0],
            &[QSTRIDE],
        )?;
        add64(b, QCUR, QCUR, T0)?;
        issue_stage(b, 0)?;
        // The store must have read its descriptor and data before they change.
        b.wait(Counter::Vs, 0)?;
        add64(b, SRD_S, SRD_S, SCSTRIDE)?;
        set_visible(b)?;
        let done = scmp(w, format!("s_cmp_ge_u32 s{ROW}, s{NROWS}"), &[ROW, NROWS])?;
        w.break_if(done, exit)
    })?;
    wg.isa().wait(Counter::Lgkm, 0)?;
    wg.end(end)
}

/// Fetch stage `stage` of the current query row into its ring half: head `h`'s
/// dims `8 * stage .. 8 * stage + 8` into eight SGPRs.
fn issue_stage(b: &mut Builder, stage: usize) -> Result<(), String> {
    for h in 0..HEADS {
        smem(
            b,
            q_buf(stage, h),
            8,
            QCUR,
            (h * DIM * 4 + stage * CHUNK * 4) as u32,
        )?;
    }
    Ok(())
}

/// First SGPR of head `h` of stage `stage`'s ring half.
fn q_buf(stage: usize, h: usize) -> u8 {
    RING + 32 * (stage % 2) as u8 + 8 * h as u8
}

/// `num_records` of the scores descriptor for the row whose visible block
/// count is `visible_quot`: `4 * max(0, min(block_count, visible) - blk0)`.
fn set_visible(b: &mut Builder) -> Result<(), String> {
    sop(
        b,
        format!("s_min_i32 s{T0}, s{BLOCK_COUNT}, s{QUOT}"),
        &[T0],
        &[BLOCK_COUNT, QUOT],
    )?;
    sop(
        b,
        format!("s_sub_i32 s{T0}, s{T0}, s{BLK0}"),
        &[T0],
        &[T0, BLK0],
    )?;
    sop(b, format!("s_max_i32 s{T0}, s{T0}, 0"), &[T0], &[T0])?;
    sop(
        b,
        format!("s_lshl_b32 s{}, s{T0}, 2", SRD_S + 2),
        &[SRD_S + 2],
        &[T0],
    )
}

/// One query row: 16 stages of eight dims, four chains per dim, then the epilogue and the store.
fn row(b: &mut Builder) -> Result<(), String> {
    for st in 0..STAGES {
        // Retire this stage's fetch (issued during the previous stage), then fetch the next into the other half.
        b.wait(Counter::Lgkm, 0)?;
        if st + 1 < STAGES {
            issue_stage(b, st + 1)?;
        }
        for dd in 0..CHUNK {
            let d = st * CHUNK + dd;
            let key = KEYS + d as u8;
            for h in 0..HEADS {
                let (acc, q) = (ACC + h as u8, q_buf(st, h) + dd as u8);
                if d == 0 {
                    op(
                        b,
                        format!("v_fma_f32 v{acc}, s{q}, v{key}, 0"),
                        &[v(acc)],
                        &[s(q), v(key)],
                    )?;
                } else {
                    op(
                        b,
                        format!("v_fmac_f32_e32 v{acc}, s{q}, v{key}"),
                        &[v(acc)],
                        &[v(acc), s(q), v(key)],
                    )?;
                }
            }
        }
    }
    b.enable_delay_alu();
    divide_pair(
        b,
        ACC,
        ACC + 1,
        [DT, DT + 1, DT + 2, DT + 3, DT + 4],
        [DT + 5, DT + 6, DT + 7, DT + 8, DT + 9],
    )?;
    divide_pair(
        b,
        ACC + 2,
        ACC + 3,
        [DT, DT + 1, DT + 2, DT + 3, DT + 4],
        [DT + 5, DT + 6, DT + 7, DT + 8, DT + 9],
    )?;
    for h in 0..HEADS as u8 {
        op(
            b,
            format!("v_max_f32_e32 v{}, 0, v{}", ACC + h, ACC + h),
            &[v(ACC + h)],
            &[v(ACC + h)],
        )?;
    }
    op(
        b,
        format!("v_add_f32_e32 v{ACC}, 0, v{ACC}"),
        &[v(ACC)],
        &[v(ACC)],
    )?;
    for h in 1..HEADS as u8 {
        op(
            b,
            format!("v_add_f32_e32 v{ACC}, v{}, v{ACC}", ACC + h),
            &[v(ACC)],
            &[v(ACC + h), v(ACC)],
        )?;
    }
    mem(
        b,
        format!(
            "buffer_store_b32 v{ACC}, v{SOFF}, s[{SRD_S}:{}], 0 offen",
            SRD_S + 3
        ),
        &[],
        &[v(ACC), v(SOFF), sr(SRD_S, 4)],
        MemoryClass::VmemStore,
    )
}

/// IEEE `n / sqrtf(128)` in place over two accumulators, the two hipcc
/// `v_div_scale` / `v_div_fmas` / `v_div_fixup` sequences interleaved. `ta`
/// and `tb` are `[ns, ds, rc, e, q]`; the first uses VCC for its scale flag
/// and the second `s{MASK}`, moved to VCC for its `v_div_fmas` as hipcc does.
fn divide_pair(b: &mut Builder, na: u8, nb: u8, ta: [u8; 5], tb: [u8; 5]) -> Result<(), String> {
    let (n, t) = ([na, nb], [ta, tb]);
    for i in 0..2 {
        let [_, ds, ..] = t[i];
        op(
            b,
            format!("v_div_scale_f32 v{ds}, null, v{DIV}, v{DIV}, v{}", n[i]),
            &[v(ds)],
            &[v(DIV), v(n[i])],
        )?;
    }
    op(
        b,
        format!("v_div_scale_f32 v{}, vcc_lo, v{na}, v{DIV}, v{na}", ta[0]),
        &[v(ta[0])],
        &[v(na), v(DIV)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{}, s{MASK}, v{nb}, v{DIV}, v{nb}", tb[0]),
        &[v(tb[0]), s(MASK)],
        &[v(nb), v(DIV)],
    )?;
    for i in 0..2 {
        let [_, ds, rc, ..] = t[i];
        op(b, format!("v_rcp_f32_e32 v{rc}, v{ds}"), &[v(rc)], &[v(ds)])?;
    }
    for i in 0..2 {
        let [_, ds, rc, e, _] = t[i];
        op(
            b,
            format!("v_fma_f32 v{e}, -v{ds}, v{rc}, 1.0"),
            &[v(e)],
            &[v(ds), v(rc)],
        )?;
    }
    for i in 0..2 {
        let [_, _, rc, e, _] = t[i];
        op(
            b,
            format!("v_fmac_f32_e32 v{rc}, v{e}, v{rc}"),
            &[v(rc)],
            &[v(rc), v(e)],
        )?;
    }
    for i in 0..2 {
        let [ns, _, rc, _, q] = t[i];
        op(
            b,
            format!("v_mul_f32_e32 v{q}, v{ns}, v{rc}"),
            &[v(q)],
            &[v(ns), v(rc)],
        )?;
    }
    for i in 0..2 {
        let [ns, ds, _, e, q] = t[i];
        op(
            b,
            format!("v_fma_f32 v{e}, -v{ds}, v{q}, v{ns}"),
            &[v(e)],
            &[v(ds), v(q), v(ns)],
        )?;
    }
    for i in 0..2 {
        let [_, _, rc, e, q] = t[i];
        op(
            b,
            format!("v_fmac_f32_e32 v{q}, v{e}, v{rc}"),
            &[v(q)],
            &[v(q), v(e), v(rc)],
        )?;
    }
    for i in 0..2 {
        let [ns, ds, _, e, q] = t[i];
        op(
            b,
            format!("v_fma_f32 v{e}, -v{ds}, v{q}, v{ns}"),
            &[v(e)],
            &[v(ds), v(q), v(ns)],
        )?;
    }
    let [_, _, rc, e, q] = ta;
    op(
        b,
        format!("v_div_fmas_f32 v{e}, v{e}, v{rc}, v{q}"),
        &[v(e)],
        &[v(e), v(rc), v(q)],
    )?;
    sop(b, format!("s_mov_b32 vcc_lo, s{MASK}"), &[], &[MASK])?;
    let [_, _, rc, e, q] = tb;
    op(
        b,
        format!("v_div_fmas_f32 v{e}, v{e}, v{rc}, v{q}"),
        &[v(e)],
        &[v(e), v(rc), v(q)],
    )?;
    for i in 0..2 {
        let [_, _, _, e, _] = t[i];
        op(
            b,
            format!("v_div_fixup_f32 v{}, v{e}, v{DIV}, v{}", n[i], n[i]),
            &[v(n[i])],
            &[v(e), v(DIV), v(n[i])],
        )?;
    }
    Ok(())
}

/// Exact `q = n / d` over unsigned 32-bit SGPRs (`d != 0`): the LLVM
/// reciprocal estimate on lane-uniform VGPRs, then two remainder corrections.
fn udiv(b: &mut Builder, q: u8, n: u8, d: u8, vt: [u8; 3], t: [u8; 3]) -> Result<(), String> {
    let [f, r, h] = vt;
    let [neg, rem, alt] = t;
    op(b, format!("v_cvt_f32_u32_e32 v{f}, s{d}"), &[v(f)], &[s(d)])?;
    op(
        b,
        format!("v_rcp_iflag_f32_e32 v{f}, v{f}"),
        &[v(f)],
        &[v(f)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{f}, 0x4f7ffffe, v{f}"),
        &[v(f)],
        &[v(f)],
    )?;
    op(b, format!("v_cvt_u32_f32_e32 v{r}, v{f}"), &[v(r)], &[v(f)])?;
    sop(b, format!("s_sub_i32 s{neg}, 0, s{d}"), &[neg], &[d])?;
    op(
        b,
        format!("v_mul_lo_u32 v{h}, s{neg}, v{r}"),
        &[v(h)],
        &[s(neg), v(r)],
    )?;
    op(
        b,
        format!("v_mul_hi_u32 v{h}, v{r}, v{h}"),
        &[v(h)],
        &[v(r), v(h)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{r}, v{r}, v{h}"),
        &[v(r)],
        &[v(r), v(h)],
    )?;
    op(
        b,
        format!("v_mul_hi_u32 v{h}, s{n}, v{r}"),
        &[v(h)],
        &[s(n), v(r)],
    )?;
    op(
        b,
        format!("v_readfirstlane_b32 s{q}, v{h}"),
        &[s(q)],
        &[v(h)],
    )?;
    for _ in 0..2 {
        sop(b, format!("s_mul_i32 s{rem}, s{q}, s{d}"), &[rem], &[q, d])?;
        sop(
            b,
            format!("s_sub_i32 s{rem}, s{n}, s{rem}"),
            &[rem],
            &[n, rem],
        )?;
        sop(b, format!("s_add_i32 s{alt}, s{q}, 1"), &[alt], &[q])?;
        sop(b, format!("s_cmp_ge_u32 s{rem}, s{d}"), &[], &[rem, d])?;
        sop(
            b,
            format!("s_cselect_b32 s{q}, s{alt}, s{q}"),
            &[q],
            &[alt, q],
        )?;
    }
    Ok(())
}
