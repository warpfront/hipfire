// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave-3 C8 MoE glue-fold entries of qwen4 Flash-Next on exact gfx1201
//! (`HIPFIRE_QWEN4_MOE_GLUE_FOLD`, default on; folds a/b/c). Four hipcc-built entries in two modules, three
//! kernarg layouts:
//!
//! | contract | entries | launcher |
//! |---|---|---|
//! | [`MOE_SCATTER_FUSED_TOP10_PMASK`] | `moe_scatter_fused_top10_pmask` (X8a) | `Gpu::moe_scatter_fused_top10_pmask` (`rdna-compute/src/moe.rs`) |
//! | [`MOE_ROUTE_SCATTER_TOP10_WIDE_F32`] | `moe_route_scatter_top10_wide_f32` (X8b) | `Gpu::moe_route_scatter_top10_wide_f32` |
//! | [`QWEN4_MOE_EP_A8_GATE_UP_ROT_PLANAR`] | `qwen4_moe_ep_a8_gate_up_rot_planar_nt{1,2}` (X8c) | `Gpu::gemm_qwen4_moe_gate_up_silu_a8_rot_sym` (`rdna-compute/src/qwen4_moe_ep_a8.rs`) |
//!
//! The first two live in module `moe_route_scatter_wide` (`kernels/src/moe_route_scatter_wide.gfx1201.hip`, compiled after
//! the fast router and the qwen4 scatter units); the third in `qwen4_moe_ep_a8_rot_gfx12`
//! (`kernels/src/qwen4_moe_ep_a8_rot.gfx12.hip`, the planar A8 gate/up chain of `qwen4_moe_ep_a8.gfx12.hip` plus the
//! rotate128 epilogue). The Route-A PeaceMaker twins of the two modules
//! (`kernels/moe_route_scatter_wide_pm_gfx1201.hxaco`, `kernels/qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco`) are whole-ELF
//! byte-identical proof artifacts of these hipcc objects (same kernarg metadata); dispatch stays HIP.
//!
//! The two scatter entries carry the runtime's hidden block as the object metadata lists it (including the
//! dynamic-LDS word); the A8 rot entries have no hidden words (84 B segment, explicit args only).
//!
//! Admission that is not a function of the launch params stays in the launcher (`moe_glue_fold_mask_admitted`,
//! `moe_glue_fold_route_admitted`, `qwen4_moe_gu_rot_fold_applies`: exact gfx1201, eager, planar experts, the
//! block-wide-scan scatter, 9..=64 rows for X8b / X8c, `!is_recording() && !capture_mode`).

use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::qwen4_moe_sym::{
    FORMAT_EXPERT_PTRS_MQ4_PLANAR, FORMAT_EXPERT_TILE_IDS, FORMAT_SORTED_SLOT_INDEX,
};

// ---- numeric contracts / formats -------------------------------------------------------------

/// The shipped scatter body UNCHANGED (histogram, padded exclusive prefix, bucket fill) followed by the owner tile
/// mask `tile_ids_owned[t] = owner_table[id] == rank && 0 <= id < E ? id : -1`: integer only. The tables equal the
/// unfused `moe_scatter_fused_top10` + `ep_tile_mask` as sets; the order of slots inside one expert's bucket is the
/// scatter's atomic arrival race, unchanged.
pub const NUMERIC_SCATTER_MASK_I32: NumericContractId =
    NumericContractId("qwen4_moe_glue_fold.scatter_mask.i32.v1");

/// The fast top-10 router (`router_fast_token`, the very text of `moe_router_softmax_top10_f32_fast`) on eight tokens
/// per workgroup, then the last-arriving workgroup runs the scatter and the mask of
/// [`NUMERIC_SCATTER_MASK_I32`]: every output byte is the unfused router + scatter (+ mask) chain's, up to the order
/// inside one expert's bucket.
pub const NUMERIC_ROUTE_SCATTER_WIDE: NumericContractId =
    NumericContractId("qwen4_moe_glue_fold.route_scatter_wide.top10_f32.v1");

/// Planar drop-zero MQ4 expert weights x `block_i8_128` activations: the gate/up A8 chain of
/// `qwen4_moe_ep_a8_gate_up_planar_nt{1,2}` (exact int32 per K128 epoch, `fma` fold, BF16 round-trip SwiGLU
/// epilogue, padding rows `+0`), then per 128-column group the `qwen4_moe_rotate128_i8` body (signs1, FWHT128,
/// `* 0.0883883476f`, signs2, `block_i8_128` quantization) over the staged BF16 SwiGLU rows. The BF16 grouped rows are
/// not written; the output is the down sidecar.
pub const NUMERIC_EP_A8_GATE_UP_ROT: NumericContractId =
    NumericContractId("qwen4_moe_ep_a8.planar_gate_up_swiglu_rot128_i8.v1");

/// Router logits `[tokens, 512]` F32, row-major.
pub const FORMAT_F32_LOGITS_ROWS_512: FormatId = FormatId("qwen4_moe.router_logits.f32.rows_x_512");
/// Router top-10 expert ids `[tokens, 10]` i32 (`-1` marks a failed row).
pub const FORMAT_I32_TOPK_IDX: FormatId = FormatId("qwen4_moe.topk_idx.i32.rows_x_10");
/// Router top-10 weights `[tokens, 10]` F32.
pub const FORMAT_F32_TOPK_W: FormatId = FormatId("qwen4_moe.topk_w.f32.rows_x_10");
/// Per-expert i32 table: padded counts `[E]`, owner table `[E]`, offsets `[E + 1]`.
pub const FORMAT_I32_PER_EXPERT: FormatId = FormatId("qwen4_moe.scatter_table.i32.per_expert");
/// Inverse permutation `[total_slots]` i32.
pub const FORMAT_I32_INVERSE_PERM: FormatId = FormatId("qwen4_moe.inverse_perm.i32.per_slot");
/// Owned tile ids `[tiles]` i32 (`-1` for a tile another rank owns).
pub const FORMAT_I32_TILE_IDS_OWNED: FormatId = FormatId("qwen4_moe.tile_ids_owned.i32.per_tile16");
/// Device-visible router fault word (host-mapped u64).
pub const FORMAT_U64_ROUTER_FAULT: FormatId = FormatId("qwen4_moe.router_fault.u64.word");
/// Persistent arrival counter of the wide router + scatter launch: one u32, zero between launches.
pub const FORMAT_U32_ARRIVAL_COUNTER: FormatId = FormatId("qwen4_moe.route_sync.u32.single");
/// Signed `block_i8_128` records (`float d; int s; i8 qs[128]`, 136 B), epoch-major `[K/128][x_src_rows]`.
pub const FORMAT_BLOCK_I8_128_EPOCH_ROWS: FormatId =
    FormatId("qwen4_moe.xq.block_i8_128.epoch_major.token_rows");
/// Down sidecar of the rot entries: `block_i8_128` records, block index `group * L + flat_slot`,
/// `group < M / 2 / 128`, `L = x_src_rows * x_row_div`.
pub const FORMAT_BLOCK_I8_128_DOWN_FLAT_SLOTS: FormatId =
    FormatId("qwen4_moe.xdown.block_i8_128.group_major.flat_slots");
/// Rotate128 sign plane (128 F32).
pub const FORMAT_F32_SIGNS_128: FormatId = FormatId("qwen4_moe.mq_signs.f32.128");

// ---- shared helpers --------------------------------------------------------------------------

const fn param(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}

const fn ptr(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand: name },
    }
}

const fn scalar(name: &'static str, offset: u32, value: Expr) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::I32,
            value,
        },
    }
}

const fn hid(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
    HiddenV1 {
        offset,
        width,
        kind,
    }
}

const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}

const fn op(
    name: &'static str,
    format: FormatId,
    access: Access,
    align: u32,
    extent: Expr,
    nullable: bool,
    note: &'static str,
) -> OperandV1 {
    OperandV1 {
        name,
        format,
        access,
        align,
        extent: Some(extent),
        nullable,
        note,
    }
}

const I32_MAX: u64 = i32::MAX as u64;
const E512: u64 = 512;

// ---- X8a / X8b: module moe_route_scatter_wide ------------------------------------------------

const OBJ_WIDE: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "moe_route_scatter_wide",
    source: "kernels/src/moe_route_scatter_wide.gfx1201.hip",
};

const P_TOTAL_SLOTS: Expr = Expr::Param("total_slots");
const P_NUM_EXPERTS: Expr = Expr::Param("num_experts");
const P_M_TOTAL_MAX: Expr = Expr::Param("m_total_max");
const P_BLOCK_M: Expr = Expr::Param("block_m");
const P_TILES: Expr = Expr::Param("tiles");
const P_RANK: Expr = Expr::Param("rank");
const P_TOKENS: Expr = Expr::Param("tokens");

const EXT_TOPK_SLOTS: Expr = Expr::Mul(&[P_TOTAL_SLOTS, Expr::Const(4)]);
const EXT_PER_EXPERT: Expr = Expr::Mul(&[P_NUM_EXPERTS, Expr::Const(4)]);
const EXT_OFFSETS: Expr = Expr::Mul(&[Expr::Add(&[P_NUM_EXPERTS, Expr::Const(1)]), Expr::Const(4)]);
const EXT_SORTED_MAX: Expr = Expr::Mul(&[P_M_TOTAL_MAX, Expr::Const(4)]);
/// `m_total_max / block_m` tile ids (`block_m` is the grouped-GEMM tile, 16).
const EXT_TILE_IDS_MAX: Expr = Expr::Mul(&[Expr::DivCeil(&P_M_TOTAL_MAX, 16), Expr::Const(4)]);
const EXT_TILES_I32: Expr = Expr::Mul(&[P_TILES, Expr::Const(4)]);

const NOTE_TABLE_OVERWRITE: &str =
    "overwritten whole by the scatter body (padded counts / prefix offsets)";

// X8a ------------------------------------------------------------------------------------------

const PARAMS_PMASK: &[ParamV1] = &[
    param("total_slots", "routed slots = tokens * 10"),
    param(
        "num_experts",
        "experts E (512); also the dynamic LDS histogram words",
    ),
    param(
        "m_total_max",
        "upper bound of grouped rows (the -1 init span), a multiple of 16",
    ),
    param("block_m", "grouped tile rows (16)"),
    param("tiles", "owner mask length: grouped_rows / 16"),
    param("rank", "this rank's owner id"),
];

const OPERANDS_PMASK: &[OperandV1] = &[
    op(
        "topk_indices",
        FORMAT_I32_TOPK_IDX,
        Access::ReadOnly,
        4,
        EXT_TOPK_SLOTS,
        false,
        "an id outside [0, num_experts) is not counted",
    ),
    op(
        "expert_token_counts",
        FORMAT_I32_PER_EXPERT,
        Access::WriteOnly,
        4,
        EXT_PER_EXPERT,
        false,
        NOTE_TABLE_OVERWRITE,
    ),
    op(
        "expert_offsets",
        FORMAT_I32_PER_EXPERT,
        Access::WriteOnly,
        4,
        EXT_OFFSETS,
        false,
        NOTE_TABLE_OVERWRITE,
    ),
    op(
        "sorted_slot_index",
        FORMAT_SORTED_SLOT_INDEX,
        Access::WriteOnly,
        4,
        EXT_SORTED_MAX,
        false,
        "initialised to -1 over m_total_max, live slots then stored per expert bucket",
    ),
    op(
        "expert_tile_ids",
        FORMAT_EXPERT_TILE_IDS,
        Access::ReadWrite,
        4,
        EXT_TILE_IDS_MAX,
        false,
        "initialised to -1 over m_total_max / block_m, live tiles stored, then read back by the mask pass",
    ),
    op(
        "inverse_perm",
        FORMAT_I32_INVERSE_PERM,
        Access::WriteOnly,
        4,
        EXT_TOPK_SLOTS,
        false,
        "one entry per routed slot",
    ),
    op(
        "tile_ids_owned",
        FORMAT_I32_TILE_IDS_OWNED,
        Access::WriteOnly,
        4,
        EXT_TILES_I32,
        false,
        "never aliases expert_tile_ids; tiles <= m_total_max / 16 is the launcher's guard",
    ),
    op(
        "owner_table",
        FORMAT_I32_PER_EXPERT,
        Access::ReadOnly,
        4,
        EXT_PER_EXPERT,
        false,
        "owner rank of each expert (the sealed e % ranks table)",
    ),
];

const FIELDS_PMASK: &[FieldV1] = &[
    ptr("topk_indices", 0),
    ptr("expert_token_counts", 8),
    ptr("expert_offsets", 16),
    ptr("sorted_slot_index", 24),
    ptr("expert_tile_ids", 32),
    ptr("inverse_perm", 40),
    scalar("total_slots", 48, P_TOTAL_SLOTS),
    scalar("num_experts", 52, P_NUM_EXPERTS),
    scalar("m_total_max", 56, P_M_TOTAL_MAX),
    scalar("block_m", 60, P_BLOCK_M),
    ptr("tile_ids_owned", 64),
    ptr("owner_table", 72),
    scalar("tiles", 80, P_TILES),
    scalar("rank", 84, P_RANK),
];

const HIDDEN_PMASK: &[HiddenV1] = &[
    hid(88, 4, HiddenKind::BlockCount(Axis::X)),
    hid(92, 4, HiddenKind::BlockCount(Axis::Y)),
    hid(96, 4, HiddenKind::BlockCount(Axis::Z)),
    hid(100, 2, HiddenKind::GroupSize(Axis::X)),
    hid(102, 2, HiddenKind::GroupSize(Axis::Y)),
    hid(104, 2, HiddenKind::GroupSize(Axis::Z)),
    hid(106, 2, HiddenKind::Remainder(Axis::X)),
    hid(108, 2, HiddenKind::Remainder(Axis::Y)),
    hid(110, 2, HiddenKind::Remainder(Axis::Z)),
    hid(128, 8, HiddenKind::GlobalOffset(Axis::X)),
    hid(136, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hid(144, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hid(152, 2, HiddenKind::GridDims),
    hid(208, 4, HiddenKind::DynamicLdsSize),
];

const DOMAIN_PMASK: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_TOTAL_SLOTS), "no routed slots"),
    rule(
        Rule::Equal(P_NUM_EXPERTS, Expr::Const(E512)),
        "Flash-Next E512: the launch passes dynamic LDS = 512 * 4",
    ),
    rule(
        Rule::Equal(P_BLOCK_M, Expr::Const(16)),
        "MOE_GROUPED_BLOCK_M: the tile extents divide by 16",
    ),
    rule(
        Rule::MultipleOf(P_M_TOTAL_MAX, 16),
        "whole 16-slot tiles over the -1 init span",
    ),
    rule(Rule::Positive(P_M_TOTAL_MAX), "no grouped rows"),
    rule(
        Rule::Positive(P_TILES),
        "owner mask covers at least one tile",
    ),
    rule(Rule::AtMost(P_TOTAL_SLOTS, I32_MAX), "slots passed as i32"),
    rule(Rule::AtMost(P_M_TOTAL_MAX, I32_MAX), "rows passed as i32"),
];

const EXPORTS_PMASK: &[ExportV1] = &[ExportV1 {
    symbol: "moe_scatter_fused_top10_pmask",
    arch: Arch::Gfx1201,
    object: OBJ_WIDE,
    block: [256, 1, 1],
    dynamic_lds: 2048,
    group_segment_fixed: 128,
    grid: [Expr::Const(1), Expr::Const(1), Expr::Const(1)],
    consts: &[],
}];

/// X8a: the shipped scatter plus the owner tile mask in ONE single-CTA launch: 6 pointers + 4 i32 + 2 pointers + 2
/// i32 (88 explicit bytes), hidden block at 88 with the dynamic-LDS word at 208, segment 344.
pub static MOE_SCATTER_FUSED_TOP10_PMASK: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_glue_fold.hipcc.gfx12.scatter_fused_top10_pmask",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SCATTER_MASK_I32,
    params: PARAMS_PMASK,
    operands: OPERANDS_PMASK,
    fields: FIELDS_PMASK,
    hidden: HIDDEN_PMASK,
    explicit_bytes: 88,
    segment_bytes: 344,
    segment_align: 8,
    domain: DOMAIN_PMASK,
    exports: EXPORTS_PMASK,
};

// X8b ------------------------------------------------------------------------------------------

const PARAMS_WIDE: &[ParamV1] = &[
    param("normalize_topk_prob", "0/1: renormalise the top-10 weights"),
    param("round_logits", "0/1: BF16 round trip of the logits"),
    param("tokens", "router rows, 9..=64 (launcher admission)"),
    param("layer", "sealed call layer index (fault record)"),
    param(
        "row_base",
        "global row of this launch's row 0 (fault record)",
    ),
    param(
        "m_total_max",
        "upper bound of grouped rows (the -1 init span), a multiple of 16",
    ),
    param("block_m", "grouped tile rows (16)"),
    param("tiles", "owner mask length (0 with no mask)"),
    param("rank", "this rank's owner id"),
];

const EXT_TOPK_TOKENS: Expr = Expr::Mul(&[P_TOKENS, Expr::Const(10 * 4)]);
const EXT_COUNTS_512: Expr = Expr::Const(E512 * 4);
const EXT_OFFSETS_512: Expr = Expr::Const((E512 + 1) * 4);

const OPERANDS_WIDE: &[OperandV1] = &[
    op(
        "logits",
        FORMAT_F32_LOGITS_ROWS_512,
        Access::ReadOnly,
        4,
        Expr::Mul(&[P_TOKENS, Expr::Const(E512 * 4)]),
        false,
        "one 512-logit row per token",
    ),
    op(
        "topk_idx",
        FORMAT_I32_TOPK_IDX,
        Access::ReadWrite,
        4,
        EXT_TOPK_TOKENS,
        false,
        "router output written by the row's wave, read back by the last workgroup's scatter",
    ),
    op(
        "topk_w",
        FORMAT_F32_TOPK_W,
        Access::WriteOnly,
        4,
        EXT_TOPK_TOKENS,
        false,
        "router weights, one per routed slot",
    ),
    op(
        "fault",
        FORMAT_U64_ROUTER_FAULT,
        Access::ReadWrite,
        8,
        Expr::Const(8),
        true,
        "host-mapped router fault word; a null pointer records no fault",
    ),
    op(
        "expert_token_counts",
        FORMAT_I32_PER_EXPERT,
        Access::WriteOnly,
        4,
        EXT_COUNTS_512,
        false,
        NOTE_TABLE_OVERWRITE,
    ),
    op(
        "expert_offsets",
        FORMAT_I32_PER_EXPERT,
        Access::WriteOnly,
        4,
        EXT_OFFSETS_512,
        false,
        NOTE_TABLE_OVERWRITE,
    ),
    op(
        "sorted_slot_index",
        FORMAT_SORTED_SLOT_INDEX,
        Access::WriteOnly,
        4,
        EXT_SORTED_MAX,
        false,
        "initialised to -1 over m_total_max, live slots then stored per expert bucket",
    ),
    op(
        "expert_tile_ids",
        FORMAT_EXPERT_TILE_IDS,
        Access::ReadWrite,
        4,
        EXT_TILE_IDS_MAX,
        false,
        "initialised to -1 over m_total_max / block_m, live tiles stored, then read back by the mask pass",
    ),
    op(
        "inverse_perm",
        FORMAT_I32_INVERSE_PERM,
        Access::WriteOnly,
        4,
        EXT_TOPK_TOKENS,
        false,
        "one entry per routed slot (tokens * 10)",
    ),
    op(
        "tile_ids_owned",
        FORMAT_I32_TILE_IDS_OWNED,
        Access::WriteOnly,
        4,
        EXT_TILES_I32,
        true,
        "null with no mask (then tiles = 0, rank = 0); never aliases expert_tile_ids",
    ),
    op(
        "owner_table",
        FORMAT_I32_PER_EXPERT,
        Access::ReadOnly,
        4,
        EXT_COUNTS_512,
        true,
        "null = no mask: the mask pass is skipped and tile_ids_owned is not written",
    ),
    op(
        "sync",
        FORMAT_U32_ARRIVAL_COUNTER,
        Access::ReadWrite,
        4,
        Expr::Const(4),
        false,
        "the Gpu's persistent arrival word: zero on entry, re-zeroed by the last workgroup; one launch in flight",
    ),
];

const FIELDS_WIDE: &[FieldV1] = &[
    ptr("logits", 0),
    ptr("topk_idx", 8),
    ptr("topk_w", 16),
    scalar(
        "normalize_topk_prob",
        24,
        Expr::Param("normalize_topk_prob"),
    ),
    scalar("round_logits", 28, Expr::Param("round_logits")),
    scalar("tokens", 32, P_TOKENS),
    ptr("fault", 40),
    scalar("layer", 48, Expr::Param("layer")),
    scalar("row_base", 52, Expr::Param("row_base")),
    ptr("expert_token_counts", 56),
    ptr("expert_offsets", 64),
    ptr("sorted_slot_index", 72),
    ptr("expert_tile_ids", 80),
    ptr("inverse_perm", 88),
    scalar("m_total_max", 96, P_M_TOTAL_MAX),
    scalar("block_m", 100, P_BLOCK_M),
    ptr("tile_ids_owned", 104),
    ptr("owner_table", 112),
    scalar("tiles", 120, P_TILES),
    scalar("rank", 124, P_RANK),
    ptr("sync", 128),
];

const HIDDEN_WIDE: &[HiddenV1] = &[
    hid(136, 4, HiddenKind::BlockCount(Axis::X)),
    hid(140, 4, HiddenKind::BlockCount(Axis::Y)),
    hid(144, 4, HiddenKind::BlockCount(Axis::Z)),
    hid(148, 2, HiddenKind::GroupSize(Axis::X)),
    hid(150, 2, HiddenKind::GroupSize(Axis::Y)),
    hid(152, 2, HiddenKind::GroupSize(Axis::Z)),
    hid(154, 2, HiddenKind::Remainder(Axis::X)),
    hid(156, 2, HiddenKind::Remainder(Axis::Y)),
    hid(158, 2, HiddenKind::Remainder(Axis::Z)),
    hid(176, 8, HiddenKind::GlobalOffset(Axis::X)),
    hid(184, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hid(192, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hid(200, 2, HiddenKind::GridDims),
    hid(256, 4, HiddenKind::DynamicLdsSize),
];

const DOMAIN_WIDE: &[DomainRuleV1] = &[
    rule(
        Rule::AtLeast(P_TOKENS, 1),
        "the kernel serves 1..=64 rows (the launcher admits 9..=64: 1..=8 keep the unfused path)",
    ),
    rule(
        Rule::AtMost(P_TOKENS, 64),
        "MOE_GLUE_FOLD_MAX_ROWS: the scatter's slot span and the LDS histogram",
    ),
    rule(
        Rule::Equal(P_BLOCK_M, Expr::Const(16)),
        "MOE_GROUPED_BLOCK_M: the tile extents divide by 16",
    ),
    rule(
        Rule::MultipleOf(P_M_TOTAL_MAX, 16),
        "whole 16-slot tiles over the -1 init span",
    ),
    rule(Rule::Positive(P_M_TOTAL_MAX), "no grouped rows"),
    rule(Rule::AtMost(P_M_TOTAL_MAX, I32_MAX), "rows passed as i32"),
    rule(Rule::AtMost(P_TILES, I32_MAX), "tiles passed as i32"),
];

const EXPORTS_WIDE: &[ExportV1] = &[ExportV1 {
    symbol: "moe_route_scatter_top10_wide_f32",
    arch: Arch::Gfx1201,
    object: OBJ_WIDE,
    block: [256, 1, 1],
    dynamic_lds: 2048,
    group_segment_fixed: 16516,
    grid: [Expr::DivCeil(&P_TOKENS, 8), Expr::Const(1), Expr::Const(1)],
    consts: &[],
}];

/// X8b: the E512 / top-10 router (eight tokens per 256-thread workgroup) + last-arriving-workgroup scatter (+ mask):
/// 136 explicit bytes (pointers and i32 through kernelParams), hidden block at 136 with the dynamic-LDS word at 256,
/// segment 392. Static LDS = 8 * 512 * 4 router rows + the arrival flag + the scatter's wave totals.
pub static MOE_ROUTE_SCATTER_TOP10_WIDE_F32: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_glue_fold.hipcc.gfx12.route_scatter_top10_wide_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_ROUTE_SCATTER_WIDE,
    params: PARAMS_WIDE,
    operands: OPERANDS_WIDE,
    fields: FIELDS_WIDE,
    hidden: HIDDEN_WIDE,
    explicit_bytes: 136,
    segment_bytes: 392,
    segment_align: 8,
    domain: DOMAIN_WIDE,
    exports: EXPORTS_WIDE,
};

// ---- X8c: module qwen4_moe_ep_a8_rot_gfx12 ---------------------------------------------------

const OBJ_ROT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "qwen4_moe_ep_a8_rot_gfx12",
    source: "kernels/src/qwen4_moe_ep_a8_rot.gfx12.hip",
};

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_X_ROW_DIV: Expr = Expr::Param("x_row_div");
const P_M_TOTAL: Expr = Expr::Param("m_total");
const P_X_SRC_ROWS: Expr = Expr::Param("x_src_rows");
const P_N_EXPERTS: Expr = Expr::Param("n_experts");
/// Flat routed slots `L = x_src_rows * x_row_div` (the down sidecar's group stride).
const SLOTS_L: Expr = Expr::Mul(&[P_X_ROW_DIV, P_X_SRC_ROWS]);

const PARAMS_ROT: &[ParamV1] = &[
    param("M", "gate+up weight rows per expert, a multiple of 256 (the CTA covers one 128-column rotate group)"),
    param("K", "reduction width, a multiple of 256"),
    param("x_row_div", "sidecar row of a grouped slot: slot / x_row_div (top-k)"),
    param("m_total", "grouped rows (tile-padded), a multiple of 16; grid y = m_total / 16"),
    param("x_src_rows", "activation records per K128 epoch plane (token rows)"),
    param("n_experts", "expert pointer table entries"),
];

const OPERANDS_ROT: &[OperandV1] = &[
    op(
        "expert_weight_ptrs",
        FORMAT_EXPERT_PTRS_MQ4_PLANAR,
        Access::ReadOnly,
        8,
        Expr::Mul(&[P_N_EXPERTS, Expr::Const(8)]),
        false,
        "u64 per expert; entries point at drop-zero planar MQ4 tensors [M, K] (gate rows then up rows; the scale plane \
         starts at byte M*K/2) whose extent is the loader's contract, not checked here; the kernel indexes only \
         e < n_experts",
    ),
    op(
        "expert_tile_ids",
        FORMAT_EXPERT_TILE_IDS,
        Access::ReadOnly,
        4,
        Expr::Mul(&[Expr::DivExact(&P_M_TOTAL, 16), Expr::Const(4)]),
        false,
        "an id outside [0, n_experts) is a dead tile (no load, no store); runs of one expert are contiguous",
    ),
    op(
        "sorted_slot_index",
        FORMAT_SORTED_SLOT_INDEX,
        Access::ReadOnly,
        4,
        Expr::Mul(&[P_M_TOTAL, Expr::Const(4)]),
        false,
        "negative = padding row (zero activation, no sidecar store)",
    ),
    op(
        "Xq",
        FORMAT_BLOCK_I8_128_EPOCH_ROWS,
        Access::ReadOnly,
        4,
        Expr::Mul(&[Expr::DivExact(&P_K, 128), P_X_SRC_ROWS, Expr::Const(136)]),
        false,
        "signed block_i8_128 records, epoch-major [K/128][x_src_rows]; slot / x_row_div < x_src_rows is the \
         caller's contract",
    ),
    op(
        "Xdown",
        FORMAT_BLOCK_I8_128_DOWN_FLAT_SLOTS,
        Access::WriteOnly,
        4,
        Expr::Mul(&[Expr::DivExact(&P_M, 256), SLOTS_L, Expr::Const(136)]),
        false,
        "down sidecar [M/2/128][L] of the live slots of the tiles run; padding and dead tiles are untouched",
    ),
    op(
        "signs1",
        FORMAT_F32_SIGNS_128,
        Access::ReadOnly,
        4,
        Expr::Const(128 * 4),
        false,
        "rotate128 sign plane 1",
    ),
    op(
        "signs2",
        FORMAT_F32_SIGNS_128,
        Access::ReadOnly,
        4,
        Expr::Const(128 * 4),
        false,
        "rotate128 sign plane 2",
    ),
];

const FIELDS_ROT: &[FieldV1] = &[
    ptr("expert_weight_ptrs", 0),
    ptr("expert_tile_ids", 8),
    ptr("sorted_slot_index", 16),
    ptr("Xq", 24),
    ptr("Xdown", 32),
    ptr("signs1", 40),
    ptr("signs2", 48),
    scalar("M", 56, P_M),
    scalar("K", 60, P_K),
    scalar("x_row_div", 64, P_X_ROW_DIV),
    scalar("m_total", 68, P_M_TOTAL),
    scalar("x_src_rows", 72, P_X_SRC_ROWS),
    scalar("n_experts", 76, P_N_EXPERTS),
    scalar("L", 80, SLOTS_L),
];

/// 32-bit slot-index guard of the launcher: `max(grouped_rows, total_slots, M, K) <= i32::MAX / 2`.
const ROT_INDEX_MAX: u64 = I32_MAX / 2;

const DOMAIN_ROT: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(
        Rule::MultipleOf(P_M, 256),
        "gate and up halves of whole 128-column rotate groups (ep_a8_gate_up_rot_shape_ok)",
    ),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 256),
        "whole 136 B qt44-style K256 groups (two K128 epochs each)",
    ),
    rule(Rule::Positive(P_M_TOTAL), "no grouped rows"),
    rule(
        Rule::MultipleOf(P_M_TOTAL, 16),
        "16-slot tiles; the tile-run leader test indexes expert_tile_ids[tile + i] below m_total / 16",
    ),
    rule(Rule::AtLeast(P_X_ROW_DIV, 1), "slot / x_row_div"),
    rule(Rule::Positive(P_X_SRC_ROWS), "empty activation sidecar"),
    rule(Rule::Positive(P_N_EXPERTS), "empty expert table"),
    rule(
        Rule::AtMost(P_M_TOTAL, ROT_INDEX_MAX),
        "32-bit slot indices (launcher guard)",
    ),
    rule(
        Rule::AtMost(SLOTS_L, ROT_INDEX_MAX),
        "32-bit slot indices (launcher guard)",
    ),
    rule(Rule::AtMost(P_M, ROT_INDEX_MAX), "32-bit slot indices (launcher guard)"),
    rule(Rule::AtMost(P_K, ROT_INDEX_MAX), "32-bit slot indices (launcher guard)"),
];

/// `[M / 2 / 128, m_total / 16, 1]`, block 256 (eight wave32 waves, 16 weight rows each).
const GRID_ROT: [Expr; 3] = [
    Expr::DivExact(&P_M, 256),
    Expr::DivExact(&P_M_TOTAL, 16),
    Expr::Const(1),
];

const EXPORTS_ROT: &[ExportV1] = &[
    ExportV1 {
        symbol: "qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
        arch: Arch::Gfx1201,
        object: OBJ_ROT,
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 4352,
        grid: GRID_ROT,
        consts: &[("NT", 1)],
    },
    ExportV1 {
        symbol: "qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
        arch: Arch::Gfx1201,
        object: OBJ_ROT,
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 8704,
        grid: GRID_ROT,
        consts: &[("NT", 2)],
    },
];

/// Planar A8 gate/up GEMM with the rotate128 down producer in the epilogue, `NT` = 1 or 2 tile-run width: 7 pointers +
/// 7 i32 through kernelParams (84 explicit bytes), no hidden words (the kernel reads no grid/block dims), segment 84.
/// Static LDS `NT * 16 * 136 * 2` (the BF16 SwiGLU staging).
pub static QWEN4_MOE_EP_A8_GATE_UP_ROT_PLANAR: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_ep_a8.hipcc.gfx12.gate_up_rot_planar",
    version: 1,
    wave: 32,
    numeric: NUMERIC_EP_A8_GATE_UP_ROT,
    params: PARAMS_ROT,
    operands: OPERANDS_ROT,
    fields: FIELDS_ROT,
    hidden: &[],
    explicit_bytes: 84,
    segment_bytes: 84,
    segment_align: 8,
    domain: DOMAIN_ROT,
    exports: EXPORTS_ROT,
};

/// Every contract of this family.
pub static ABIS: &[&KernelAbiV1] = &[
    &MOE_SCATTER_FUSED_TOP10_PMASK,
    &MOE_ROUTE_SCATTER_TOP10_WIDE_F32,
    &QWEN4_MOE_EP_A8_GATE_UP_ROT_PLANAR,
];
