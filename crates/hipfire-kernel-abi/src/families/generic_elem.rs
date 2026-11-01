// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic-op elementwise/glue launchers (hipcc): converts, rounding, MoE route/scatter/combine, activations, rotations.
//!
//! Eleven contracts, one per symbol, each with a [`WorkSpecV1`] sidecar:
//! `convert_f32_to_f16`, `bf16_round_trip_f32_strided`, `ffn_round_bf16_f32`, `ffn_moe_combine_f32`,
//! `ffn_situ_grouped_f32`, `ffn_situ_rows_f32`, `ffn_moe_router_topk_f32`, `moe_scatter_fused_k8`,
//! `ffn_single_expert_tables`, `mq_rotate_x`, `rotate_x_mq_awq`.
//!
//! Evidence the numbers below come from:
//!
//! * Launchers (grid/block/kernarg order): `crates/rdna-compute/src/ffn_activation.rs` (every `ffn_*`
//!   symbol), `scratch.rs::ensure_fp16_x` (`convert_f32_to_f16`), `norm.rs::bf16_round_trip_f32_strided`,
//!   `moe.rs::moe_scatter_fused_k8`, `gemv.rs::rotate_x_mq{,_batched}` and `scratch.rs::rotate_x_mq_awq{,_batched}`
//!   (the batch is folded into grid x: `batch * K/256`).
//! * Objects: the JIT-cached hsaco of each module in `~/.hipfire_kernels/gfx1201` and `gfx1100`, unbundled
//!   with `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--<arch>` and read with
//!   `llvm-readelf --notes`. The cached object whose sibling `.hip` equals the current `kernels/src` source
//!   was used: gfx1201 `convert_f32_to_f16.0139907fed8c4a42`, `qwen4_bf16_round_trip.bab23a3c74e34080`,
//!   `ffn_situ.f7c0f8dbdef660ea`, `ffn_grouped_activation_pad.9b5dd35cbaf47545`,
//!   `ffn_router_renorm.82c629831b241523`, `ffn_moe_combine_policy.a6637b1dea47425a`,
//!   `moe_scatter_fused_k8.b36646a9826bcd8e`, `mq_rotate_x.2505e379af401ae8`,
//!   `rotate_x_mq_awq.219a506f24e16e4f`; gfx1100 `convert_f32_to_f16.0b96e50235df4e62`,
//!   `qwen4_bf16_round_trip.67877571a7c0371f`, `moe_scatter_fused_k8.916ea23517082d6a`,
//!   `mq_rotate_x.362a4d71a7c5378f`, `rotate_x_mq_awq.d858502456b9487b`. No `ffn_*` object is cached for
//!   gfx1100: those four modules were compiled locally (`hipcc --offload-arch=gfx1100 --offload-device-only -O3`)
//!   from the current sources and their metadata read the same way (identical kernarg layout, LDS 0 except the
//!   router's 10336 B).
//! * Layout facts: no kernel names its arguments in the metadata (only `convert_f32_to_f16` and the scatter /
//!   rotate objects carry `actual_access`); the field names are the kernel source's. Only
//!   `bf16_round_trip_f32_strided` (segment 280) and `moe_scatter_fused_k8` (segment 320) carry hidden words
//!   (the others have none and a segment equal to the explicit bytes).
//! * `moe_scatter_fused_k8` takes `num_experts * 4` bytes of dynamic LDS, a function of a param, while
//!   [`ExportV1::dynamic_lds`] is a constant: the contract is therefore pinned to 128 experts (LDS 512, the
//!   Raon observation) and 16-row blocks (`MOE_GROUPED_BLOCK_M`); another expert count needs its own contract.
//!   gfx1151 compiles a different source for this symbol (`qwen4_moe_scatter_fused_k8.hip`) and is not exported.
//! * `p0`/`p1`/`eps`/`scale` kernargs are `float`s: the contract params carry their IEEE-754 bit patterns
//!   (`*_bits`, `u32`).
//! * Work specs bind only the params their expressions use; flag/stride/bit-pattern params that no expression
//!   reads are deliberately left unbound.

use crate::work::{
    op_class, BindSource, BindV1, Pipe, Stacking, WeightAxis, WorkSpecV1, WorkV1, ALL_ROWS,
};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_convert::{FORMAT_F16_FLAT, FORMAT_F32_FLAT};
use super::qwen4_moe_sym::{FORMAT_EXPERT_TILE_IDS, FORMAT_SORTED_SLOT_INDEX};

// ---- formats ---------------------------------------------------------------

/// int32 flat tensor (`topk_indices`, `row_of_slot`, `expert_token_counts`, `expert_offsets`, `inverse_perm`).
pub const FORMAT_I32_FLAT: FormatId = FormatId("i32.flat");
/// u64 device-pointer table `[n_experts]`, entry e -> `F32[phys_n]` AWQ scale of expert e.
pub const FORMAT_U64_AWQ_PTR_TABLE: FormatId = FormatId("elem.u64.awq_ptr_table.f32_phys_n");
/// The `awq_a` operand of `ffn_situ_rows_f32`: an F32 vector `[phys_n]` (awq_mode 1) or the u64 pointer table
/// `[n_experts]` (awq_mode 2).
pub const FORMAT_AWQ_VECTOR_OR_TABLE: FormatId = FormatId("elem.awq_f32_vector_or_u64_ptr_table");
/// The single-expert grouping tables: three u64 weight pointers, then `rows_pad` expert indices, `rows_pad`
/// sorted slot indices and `rows_pad / 16` tile ids (all int32 words).
pub const FORMAT_SINGLE_EXPERT_TABLES: FormatId = FormatId("elem.single_expert_tables.words");
/// MagnumQuant FWHT256 sign vector: 256 f32 (`gen_fwht_signs`).
pub const FORMAT_MQ_FWHT_SIGNS_256: FormatId = FormatId("mq.fwht_signs.f32.256");

// ---- const builders ---------------------------------------------------------

const fn p(name: &'static str) -> Expr {
    Expr::Param(name)
}
const fn c(v: u64) -> Expr {
    Expr::Const(v)
}
const fn param(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}
const fn param_u32(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::U32,
        doc,
    }
}
const fn param_u64(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::U64,
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
/// An int32 by-value kernarg carrying param `name` (kernel argument name = param name).
const fn i32f(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::I32,
            value: Expr::Param(name),
        },
    }
}
/// A float by-value kernarg carried as the u32 bit pattern of param `bits`.
const fn f32f(name: &'static str, offset: u32, bits: &'static str) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::U32,
            value: Expr::Param(bits),
        },
    }
}
const fn u64f(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Scalar {
            ty: ScalarTy::U64,
            value: Expr::Param(name),
        },
    }
}
const fn operand(
    name: &'static str,
    format: FormatId,
    access: Access,
    align: u32,
    extent: Option<Expr>,
    nullable: bool,
    note: &'static str,
) -> OperandV1 {
    OperandV1 {
        name,
        format,
        access,
        align,
        extent,
        nullable,
        note,
    }
}
const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}
const fn hid(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
    HiddenV1 {
        offset,
        width,
        kind,
    }
}
#[allow(clippy::too_many_arguments)]
const fn export(
    symbol: &'static str,
    arch: Arch,
    object: ObjectRefV1,
    block_x: u32,
    dynamic_lds: u32,
    group_segment_fixed: u32,
    grid: [Expr; 3],
) -> ExportV1 {
    ExportV1 {
        symbol,
        arch,
        object,
        block: [block_x, 1, 1],
        dynamic_lds,
        group_segment_fixed,
        grid,
        consts: &[],
    }
}

const ONE: Expr = c(1);
const C2: Expr = c(2);
const C4: Expr = c(4);
const C8: Expr = c(8);
const I32_MAX: u64 = i32::MAX as u64;
/// Column tiles of 256 ride grid.y, which caps at 65535 blocks (`ffn_activation.rs::MAX_COL_TILES`).
const MAX_COL_TILES: u64 = 65535;
/// `MOE_GROUPED_BLOCK_M` (`crates/hipfire-dispatch/src/families/moe.rs`): rows per expert tile.
const BLOCK_M: u64 = 16;
/// Experts of the pinned `moe_scatter_fused_k8` contract (dynamic LDS `4 * 128` = 512 B).
const SCATTER_EXPERTS: u64 = 128;

const FLAG_01: &[u64] = &[0, 1];
const FLAG_012: &[u64] = &[0, 1, 2];

/// Config keys: experts per token (top-k), routed experts, hidden width, per-expert FFN width.
const KEYS_TOPK: &[&str] = &[
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "top_k",
];
const KEYS_EXPERTS: &[&str] = &[
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "n_experts",
    "moe_num_experts",
];
const KEYS_HIDDEN: &[&str] = &["hidden_size", "hidden_dim", "d_model", "n_embd"];
const KEYS_MOE_INTERMEDIATE: &[&str] = &[
    "moe_intermediate_size",
    "moe_intermediate_dim",
    "intermediate_size",
];

// ===========================================================================
// 1. convert_f32_to_f16
// ===========================================================================

const N: Expr = p("n");

const OBJ_CONVERT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "convert_f32_to_f16",
    source: "kernels/src/gemm_hfq4g256_residual_fp16.hip",
};
const PARAMS_CONVERT: &[ParamV1] = &[param("n", "f32 elements converted")];
const OPS_CONVERT: &[OperandV1] = &[
    operand(
        "in",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[N, C4])),
        false,
        "",
    ),
    operand(
        "out",
        FORMAT_F16_FLAT,
        Access::WriteOnly,
        2,
        Some(Expr::Mul(&[N, C2])),
        false,
        "",
    ),
];
const FIELDS_CONVERT: &[FieldV1] = &[ptr("in", 0), ptr("out", 8), i32f("n", 16)];
const DOMAIN_CONVERT: &[DomainRuleV1] = &[
    rule(Rule::Positive(N), "n > 0"),
    rule(
        Rule::AtMost(N, I32_MAX),
        "the thread index and the bound are int",
    ),
];
const GRID_CONVERT: [Expr; 3] = [Expr::DivCeil(&N, 256), ONE, ONE];
const EXPORTS_CONVERT: &[ExportV1] = &[
    export(
        "convert_f32_to_f16",
        Arch::Gfx1201,
        OBJ_CONVERT,
        256,
        0,
        0,
        GRID_CONVERT,
    ),
    export(
        "convert_f32_to_f16",
        Arch::Gfx1100,
        OBJ_CONVERT,
        256,
        0,
        0,
        GRID_CONVERT,
    ),
];

/// `convert_f32_to_f16(in, out, n)`: `out[i] = (half)in[i]` for `i < n` (round-to-nearest-even), grid
/// `ceil(n / 256)`, block 256, segment 20/8, no hidden words (the module also holds
/// `gemm_hfq4g256_residual_fp16`, a different contract's business). `n` is `rows * k` of the caller; the
/// grid gives it to within one 256-element tile.
pub static CONVERT_F32_TO_F16: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.convert_f32_to_f16",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.f32_to_f16.rne"),
    params: PARAMS_CONVERT,
    operands: OPS_CONVERT,
    fields: FIELDS_CONVERT,
    hidden: &[],
    explicit_bytes: 20,
    segment_bytes: 20,
    segment_align: 8,
    domain: DOMAIN_CONVERT,
    exports: EXPORTS_CONVERT,
};

pub static WORK_CONVERT_F32_TO_F16: WorkSpecV1 = WorkSpecV1 {
    abi: &CONVERT_F32_TO_F16,
    work: WorkV1 {
        op_class: op_class::CONVERT,
        codec: "f32>f16",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Mul(&[N, c(6)])),
        rows: None,
        admit: ALL_ROWS,
        bind: &[],
        note: "n is elements (rows * k), not rows, so no rows param; the grid bounds n to a 256-element tile (exact when n % 256 == 0); 4 B read + 2 B written per element",
    },
};

// ===========================================================================
// 2. bf16_round_trip_f32_strided
// ===========================================================================

const ROWS: Expr = p("rows");
const ROW_OFFSET: Expr = p("row_offset");
const ROW_STRIDE: Expr = p("row_stride");
const WIDTH: Expr = p("width");
const ROWS_X_WIDTH: Expr = Expr::Mul(&[ROWS, WIDTH]);
const ROWS_MINUS_1: Expr = Expr::Sub(&ROWS, &ONE);

const OBJ_BF16_RT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "qwen4_bf16_round_trip",
    source: "kernels/src/qwen4_bf16_round_trip.hip",
};
const PARAMS_BF16_RT: &[ParamV1] = &[
    param("rows", "rows rounded"),
    param("row_offset", "first rounded column of each row"),
    param("row_stride", "elements between row starts"),
    param("width", "columns rounded per row"),
];
const OPS_BF16_RT: &[OperandV1] = &[operand(
    "x",
    FORMAT_F32_FLAT,
    Access::ReadWrite,
    4,
    Some(Expr::Mul(&[
        Expr::Add(&[Expr::Mul(&[ROWS_MINUS_1, ROW_STRIDE]), ROW_OFFSET, WIDTH]),
        C4,
    ])),
    false,
    "in place; non-finite elements are left untouched",
)];
const FIELDS_BF16_RT: &[FieldV1] = &[
    ptr("x", 0),
    i32f("rows", 8),
    i32f("row_offset", 12),
    i32f("row_stride", 16),
    i32f("width", 20),
];
const HIDDEN_BF16_RT: &[HiddenV1] = &[
    hid(24, 4, HiddenKind::BlockCount(Axis::X)),
    hid(28, 4, HiddenKind::BlockCount(Axis::Y)),
    hid(32, 4, HiddenKind::BlockCount(Axis::Z)),
    hid(36, 2, HiddenKind::GroupSize(Axis::X)),
    hid(38, 2, HiddenKind::GroupSize(Axis::Y)),
    hid(40, 2, HiddenKind::GroupSize(Axis::Z)),
    hid(42, 2, HiddenKind::Remainder(Axis::X)),
    hid(44, 2, HiddenKind::Remainder(Axis::Y)),
    hid(46, 2, HiddenKind::Remainder(Axis::Z)),
    hid(64, 8, HiddenKind::GlobalOffset(Axis::X)),
    hid(72, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hid(80, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hid(88, 2, HiddenKind::GridDims),
];
const DOMAIN_BF16_RT: &[DomainRuleV1] = &[
    rule(Rule::Positive(ROWS), "rows > 0"),
    rule(Rule::Positive(WIDTH), "width > 0"),
    rule(
        Rule::AtLeast(Expr::Sub(&Expr::Sub(&ROW_STRIDE, &ROW_OFFSET), &WIDTH), 0),
        "row_offset + width <= row_stride (a negative difference is rejected by the evaluation)",
    ),
    rule(
        Rule::AtMost(ROWS_X_WIDTH, I32_MAX),
        "rows * width is computed in int",
    ),
];
const GRID_BF16_RT: [Expr; 3] = [Expr::DivCeil(&ROWS_X_WIDTH, 256), ONE, ONE];
const EXPORTS_BF16_RT: &[ExportV1] = &[
    export(
        "bf16_round_trip_f32_strided",
        Arch::Gfx1201,
        OBJ_BF16_RT,
        256,
        0,
        0,
        GRID_BF16_RT,
    ),
    export(
        "bf16_round_trip_f32_strided",
        Arch::Gfx1100,
        OBJ_BF16_RT,
        256,
        0,
        0,
        GRID_BF16_RT,
    ),
];

/// `bf16_round_trip_f32_strided(x, rows, row_offset, row_stride, width)`: in-place bit-level RNE rounding
/// to bfloat16 (stored widened) of `x[r * row_stride + row_offset + c]`, `r < rows`, `c < width`; one thread
/// per element, grid `ceil(rows * width / 256)`, block 256. Segment 280/8 (24 explicit B + hidden block).
pub static BF16_ROUND_TRIP_F32_STRIDED: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.bf16_round_trip_f32_strided",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.bf16_round_trip.rne_nonfinite_skip"),
    params: PARAMS_BF16_RT,
    operands: OPS_BF16_RT,
    fields: FIELDS_BF16_RT,
    hidden: HIDDEN_BF16_RT,
    explicit_bytes: 24,
    segment_bytes: 280,
    segment_align: 8,
    domain: DOMAIN_BF16_RT,
    exports: EXPORTS_BF16_RT,
};

pub static WORK_BF16_ROUND_TRIP_F32_STRIDED: WorkSpecV1 = WorkSpecV1 {
    abi: &BF16_ROUND_TRIP_F32_STRIDED,
    work: WorkV1 {
        op_class: op_class::CONVERT,
        codec: "f32>bf16>f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Mul(&[ROWS_X_WIDTH, C8])),
        rows: Some("rows"),
        admit: ALL_ROWS,
        bind: &[BindV1 {
            param: "rows",
            source: BindSource::ChunkRows,
        }],
        note: "grid gives rows * width only; rows binds to the chunk rows (QSA index-K prefill) so width follows; 4 B read + 4 B written per element (writes skipped for non-finite values)",
    },
};

// ===========================================================================
// 3-4. ffn_moe_combine_policy: ffn_round_bf16_f32, ffn_moe_combine_f32
// ===========================================================================

const OBJ_COMBINE: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "ffn_moe_combine_policy",
    source: "kernels/src/ffn_moe_combine_policy.hip",
};

const N_ELEMS: Expr = p("n_elems");
const PARAMS_ROUND_BF16: &[ParamV1] = &[param("n_elems", "leading f32 elements of x rounded")];
const OPS_ROUND_BF16: &[OperandV1] = &[operand(
    "x",
    FORMAT_F32_FLAT,
    Access::ReadWrite,
    4,
    Some(Expr::Mul(&[N_ELEMS, C4])),
    false,
    "in place; NaN passes through, a finite value past the bf16 range rounds to inf",
)];
const FIELDS_ROUND_BF16: &[FieldV1] = &[ptr("x", 0), i32f("n_elems", 8)];
const DOMAIN_ROUND_BF16: &[DomainRuleV1] = &[
    rule(Rule::Positive(N_ELEMS), "n_elems > 0"),
    rule(Rule::AtMost(N_ELEMS, I32_MAX), "the bound `i < n` is int"),
];
const GRID_ROUND_BF16: [Expr; 3] = [Expr::DivCeil(&N_ELEMS, 256), ONE, ONE];
const EXPORTS_ROUND_BF16: &[ExportV1] = &[
    export(
        "ffn_round_bf16_f32",
        Arch::Gfx1201,
        OBJ_COMBINE,
        256,
        0,
        0,
        GRID_ROUND_BF16,
    ),
    export(
        "ffn_round_bf16_f32",
        Arch::Gfx1100,
        OBJ_COMBINE,
        256,
        0,
        0,
        GRID_ROUND_BF16,
    ),
];

/// `ffn_round_bf16_f32(x, n_elems)`: in-place bf16 RNE of `x[0..n_elems)`, grid `ceil(n_elems / 256)`,
/// block 256, segment 12/8, no hidden words. gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_ROUND_BF16_F32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_round_bf16_f32",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.ffn_round_bf16.rne_nan_passthrough"),
    params: PARAMS_ROUND_BF16,
    operands: OPS_ROUND_BF16,
    fields: FIELDS_ROUND_BF16,
    hidden: &[],
    explicit_bytes: 12,
    segment_bytes: 12,
    segment_align: 8,
    domain: DOMAIN_ROUND_BF16,
    exports: EXPORTS_ROUND_BF16,
};

pub static WORK_FFN_ROUND_BF16_F32: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_ROUND_BF16_F32,
    work: WorkV1 {
        op_class: op_class::CONVERT,
        codec: "f32>bf16>f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Mul(&[N_ELEMS, C8])),
        rows: None,
        admit: ALL_ROWS,
        bind: &[],
        note: "n_elems is elements, not rows; the grid bounds it to a 256-element tile; 4 B read + 4 B written per element",
    },
};

// ---- ffn_moe_combine_f32 ----------------------------------------------------

const HIDDEN: Expr = p("hidden");
const TOP_K: Expr = p("top_k");
const N_EXPERTS: Expr = p("n_experts");
const N_DOWN_ROWS: Expr = p("n_down_rows");
const ROWS_X_TOPK: Expr = Expr::Mul(&[ROWS, TOP_K]);

const PARAMS_COMBINE: &[ParamV1] = &[
    param("rows", "output rows (tokens), grid x"),
    param("hidden", "output columns; grid y is ceil(hidden / 256)"),
    param("top_k", "routed ranks per row, 1..=8"),
    param("n_experts", "expert ids outside [0, n_experts) are skipped"),
    param("order", "0 = rank order, 1 = ascending expert id"),
    param("round_expert_outputs", "1: bf16-round each expert row value"),
    param("round_routed_sum", "1: bf16-round the routed sum"),
    param("round_final", "1: bf16-round routed + shared"),
    param(
        "n_down_rows",
        "rows of down_rows (the launcher derives it from the tensor); gathered rows outside it are skipped",
    ),
];
const OPS_COMBINE: &[OperandV1] = &[
    operand(
        "down_rows",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[N_DOWN_ROWS, HIDDEN, C4])),
        false,
        "F32 [n_down_rows, hidden]: expanded layout (row = flat slot) or grouped (row_of_slot maps slots to rows)",
    ),
    operand(
        "row_of_slot",
        FORMAT_I32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS_X_TOPK, C4])),
        true,
        "null: row = flat slot (down_rows then holds >= rows * top_k rows); a negative entry is an unrouted slot",
    ),
    operand(
        "topk_idx",
        FORMAT_I32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS_X_TOPK, C4])),
        false,
        "[rows, top_k] expert ids",
    ),
    operand(
        "topk_w",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS_X_TOPK, C4])),
        false,
        "[rows, top_k] routing weights",
    ),
    operand(
        "shared",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS, HIDDEN, C4])),
        true,
        "added after the routed sum is rounded",
    ),
    operand(
        "routed_out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[ROWS, HIDDEN, C4])),
        true,
        "optional diagnostic: routed sum after round_routed_sum",
    ),
    operand(
        "out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[ROWS, HIDDEN, C4])),
        false,
        "",
    ),
];
const FIELDS_COMBINE: &[FieldV1] = &[
    ptr("down_rows", 0),
    ptr("row_of_slot", 8),
    ptr("topk_idx", 16),
    ptr("topk_w", 24),
    ptr("shared", 32),
    ptr("routed_out", 40),
    ptr("out", 48),
    i32f("rows", 56),
    i32f("hidden", 60),
    i32f("top_k", 64),
    i32f("n_experts", 68),
    i32f("order", 72),
    i32f("round_expert_outputs", 76),
    i32f("round_routed_sum", 80),
    i32f("round_final", 84),
    i32f("n_down_rows", 88),
];
const DOMAIN_COMBINE: &[DomainRuleV1] = &[
    rule(Rule::Positive(ROWS), "rows > 0"),
    rule(Rule::Positive(HIDDEN), "hidden > 0"),
    rule(Rule::AtLeast(TOP_K, 1), "top_k >= 1"),
    rule(
        Rule::AtMost(TOP_K, 8),
        "the kernel's rank arrays hold eight entries",
    ),
    rule(Rule::Positive(N_EXPERTS), "n_experts > 0"),
    rule(Rule::OneOf(p("order"), FLAG_01), "order is 0 or 1"),
    rule(
        Rule::OneOf(p("round_expert_outputs"), FLAG_01),
        "flag is 0 or 1",
    ),
    rule(
        Rule::OneOf(p("round_routed_sum"), FLAG_01),
        "flag is 0 or 1",
    ),
    rule(Rule::OneOf(p("round_final"), FLAG_01), "flag is 0 or 1"),
    rule(
        Rule::AtMost(Expr::DivCeil(&HIDDEN, 256), MAX_COL_TILES),
        "column tiles ride grid.y (<= 65535)",
    ),
];
const GRID_COMBINE: [Expr; 3] = [ROWS, Expr::DivCeil(&HIDDEN, 256), ONE];
const EXPORTS_COMBINE: &[ExportV1] = &[
    export(
        "ffn_moe_combine_f32",
        Arch::Gfx1201,
        OBJ_COMBINE,
        256,
        0,
        0,
        GRID_COMBINE,
    ),
    export(
        "ffn_moe_combine_f32",
        Arch::Gfx1100,
        OBJ_COMBINE,
        256,
        0,
        0,
        GRID_COMBINE,
    ),
];

/// `ffn_moe_combine_f32`: routed expert-row combine with an explicit rounding policy (no fma: separate
/// rounded multiply and add), one thread per output column, grid `(rows, ceil(hidden / 256))`, block 256,
/// segment 92/8, no hidden words. gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_MOE_COMBINE_F32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_moe_combine_f32",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.ffn_moe_combine.policy_nofma"),
    params: PARAMS_COMBINE,
    operands: OPS_COMBINE,
    fields: FIELDS_COMBINE,
    hidden: &[],
    explicit_bytes: 92,
    segment_bytes: 92,
    segment_align: 8,
    domain: DOMAIN_COMBINE,
    exports: EXPORTS_COMBINE,
};

pub static WORK_FFN_MOE_COMBINE_F32: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_MOE_COMBINE_F32,
    work: WorkV1 {
        op_class: op_class::MOE_COMBINE,
        codec: "f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: Some(Expr::Mul(&[ROWS, HIDDEN, TOP_K, C2])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[ROWS, TOP_K, HIDDEN, C4]),
            Expr::Mul(&[ROWS_X_TOPK, C8]),
            Expr::Mul(&[ROWS, HIDDEN, C4]),
        ])),
        rows: Some("rows"),
        admit: ALL_ROWS,
        bind: &[
            BindV1 {
                param: "hidden",
                source: BindSource::Config(KEYS_HIDDEN),
            },
            BindV1 {
                param: "top_k",
                source: BindSource::Config(KEYS_TOPK),
            },
            BindV1 {
                param: "n_experts",
                source: BindSource::Config(KEYS_EXPERTS),
            },
        ],
        note: "bytes: gathered expert rows (rows*top_k*hidden) + idx/weights + out; excludes the optional shared/routed_out planes; n_down_rows (a tensor extent, >= rows*top_k) and the policy flags are unbound and unused; hidden is cross-checked against the grid.y tile interval",
    },
};

// ===========================================================================
// 5-6. SiTU activation: ffn_situ_rows_f32, ffn_situ_grouped_f32
// ===========================================================================

/// Approximate VALU work per activated element of SiTU-GLU
/// (`((p0*tanhf(g/p0)) * sigmoid(g)) * (p1*tanhf(u/p1))`): two `tanhf`, one `expf`, three divides and a few
/// multiplies, counted in FLOP-equivalents. An estimate, not a measured instruction count.
const SITU_FLOP_PER_ELEM: u64 = 64;

const LOGICAL_N: Expr = p("logical_n");
const PHYS_N: Expr = p("phys_n");
const M_TOTAL: Expr = p("m_total");
const TOTAL_SLOTS: Expr = p("total_slots");

const NUMERIC_SITU: NumericContractId = NumericContractId("elem.ffn_situ.f32_expf_tanhf");

// ---- ffn_situ_rows_f32 ------------------------------------------------------

const OBJ_SITU_ROWS: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "ffn_situ",
    source: "kernels/src/ffn_situ.hip",
};
const GATE_LD: Expr = p("gate_ld");
const UP_LD: Expr = p("up_ld");
const AWQ_MODE: Expr = p("awq_mode");

const PARAMS_SITU_ROWS: &[ParamV1] = &[
    param("rows", "activation rows, grid x"),
    param("logical_n", "activation columns computed"),
    param(
        "phys_n",
        "output columns, >= logical_n; the pad is +0.0; grid y is ceil(phys_n / 256)",
    ),
    param("gate_ld", "elements between gate row starts"),
    param("up_ld", "elements between up row starts"),
    param(
        "awq_mode",
        "0 none, 1 F32 vector [phys_n], 2 per-expert pointer table + row_expert",
    ),
    param("n_experts", "per-expert AWQ table entries (awq_mode 2)"),
    param(
        "round_inputs",
        "1: bf16-round g and u before the activation",
    ),
    param("round_output", "1: bf16-round the activation result"),
    param_u32("p0_bits", "f32 bits of gate_beta"),
    param_u32("p1_bits", "f32 bits of value_beta"),
];
const OPS_SITU_ROWS: &[OperandV1] = &[
    operand(
        "gate",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[
            Expr::Add(&[Expr::Mul(&[ROWS_MINUS_1, GATE_LD]), LOGICAL_N]),
            C4,
        ])),
        false,
        "base already includes the caller's element offset",
    ),
    operand(
        "up",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[
            Expr::Add(&[Expr::Mul(&[ROWS_MINUS_1, UP_LD]), LOGICAL_N]),
            C4,
        ])),
        false,
        "base already includes the caller's element offset",
    ),
    operand(
        "out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[ROWS, PHYS_N, C4])),
        false,
        "F32 [rows, phys_n] row-major",
    ),
    operand(
        "awq_a",
        FORMAT_AWQ_VECTOR_OR_TABLE,
        Access::ReadOnly,
        8,
        None,
        true,
        "null for awq_mode 0; F32 [phys_n] for mode 1 (4*phys_n B); u64 table [n_experts] (8*n_experts B) for mode 2: the reach depends on awq_mode, so no single extent is stated",
    ),
    operand(
        "row_expert",
        FORMAT_I32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS, C4])),
        true,
        "expert of each row; read only for awq_mode 2",
    ),
];
const FIELDS_SITU_ROWS: &[FieldV1] = &[
    ptr("gate", 0),
    ptr("up", 8),
    ptr("out", 16),
    ptr("awq_a", 24),
    ptr("row_expert", 32),
    i32f("rows", 40),
    i32f("logical_n", 44),
    i32f("phys_n", 48),
    i32f("gate_ld", 52),
    i32f("up_ld", 56),
    i32f("awq_mode", 60),
    i32f("n_experts", 64),
    i32f("round_inputs", 68),
    i32f("round_output", 72),
    f32f("p0", 76, "p0_bits"),
    f32f("p1", 80, "p1_bits"),
];
const DOMAIN_SITU_ROWS: &[DomainRuleV1] = &[
    rule(Rule::Positive(ROWS), "rows > 0"),
    rule(
        Rule::Positive(LOGICAL_N),
        "logical_n > 0 (extents of gate/up)",
    ),
    rule(
        Rule::AtLeast(Expr::Sub(&PHYS_N, &LOGICAL_N), 0),
        "logical_n <= phys_n (a negative difference is rejected by the evaluation)",
    ),
    rule(
        Rule::AtLeast(Expr::Sub(&GATE_LD, &LOGICAL_N), 0),
        "gate_ld >= logical_n",
    ),
    rule(
        Rule::AtLeast(Expr::Sub(&UP_LD, &LOGICAL_N), 0),
        "up_ld >= logical_n",
    ),
    rule(
        Rule::AtMost(Expr::DivCeil(&PHYS_N, 256), MAX_COL_TILES),
        "column tiles ride grid.y (<= 65535)",
    ),
    rule(Rule::OneOf(AWQ_MODE, FLAG_012), "awq_mode is 0, 1 or 2"),
    rule(Rule::OneOf(p("round_inputs"), FLAG_01), "flag is 0 or 1"),
    rule(Rule::OneOf(p("round_output"), FLAG_01), "flag is 0 or 1"),
];
const GRID_SITU_ROWS: [Expr; 3] = [ROWS, Expr::DivCeil(&PHYS_N, 256), ONE];
const EXPORTS_SITU_ROWS: &[ExportV1] = &[
    export(
        "ffn_situ_rows_f32",
        Arch::Gfx1201,
        OBJ_SITU_ROWS,
        256,
        0,
        0,
        GRID_SITU_ROWS,
    ),
    export(
        "ffn_situ_rows_f32",
        Arch::Gfx1100,
        OBJ_SITU_ROWS,
        256,
        0,
        0,
        GRID_SITU_ROWS,
    ),
];

/// `ffn_situ_rows_f32`: SiTU-GLU gate/up activation over dense rows with optional bf16 casts, zero pad and
/// AWQ divide; grid `(rows, ceil(phys_n / 256))`, block 256, segment 84/8, no hidden words. The swiglu
/// siblings of the module (`ffn_swiglu_rows_f32`, `ffn_swiglu_clamp_rows_f32`) are separate contracts' business.
/// gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_SITU_ROWS_F32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_situ_rows_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SITU,
    params: PARAMS_SITU_ROWS,
    operands: OPS_SITU_ROWS,
    fields: FIELDS_SITU_ROWS,
    hidden: &[],
    explicit_bytes: 84,
    segment_bytes: 84,
    segment_align: 8,
    domain: DOMAIN_SITU_ROWS,
    exports: EXPORTS_SITU_ROWS,
};

pub static WORK_FFN_SITU_ROWS_F32: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_SITU_ROWS_F32,
    work: WorkV1 {
        op_class: op_class::ACTIVATION,
        codec: "f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: Some(Expr::Mul(&[ROWS, LOGICAL_N, c(SITU_FLOP_PER_ELEM)])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[ROWS, LOGICAL_N, C8]),
            Expr::Mul(&[ROWS, PHYS_N, C4]),
        ])),
        rows: Some("rows"),
        admit: ALL_ROWS,
        bind: &[
            // phys_n (grid.y tile interval) is the output width of the gate/up weights that feed this
            // launch: matching it against weight rows binds phys_n exactly; the by-product (the weights'
            // K) is a pure work param no expression reads.
            BindV1 {
                param: "gate_up_k",
                source: BindSource::Weight {
                    by: "phys_n",
                    by_axis: WeightAxis::Rows,
                    codecs: &[],
                    stacking: Stacking::Either,
                    fuse_siblings: false,
                },
            },
            BindV1 {
                param: "logical_n",
                source: BindSource::Derived(PHYS_N),
            },
        ],
        note: "rows is exact from grid.x and may be rows*top_k on the expanded MoE path; phys_n comes from the grid.y interval narrowed by a weight-rows match; logical_n = phys_n unless the output is zero-padded; flops use an estimated 64 FLOP-equivalents per element (SITU_FLOP_PER_ELEM); AWQ planes (<= 4*phys_n B) not charged",
    },
};

// ---- ffn_situ_grouped_f32 ---------------------------------------------------

const OBJ_GROUPED: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "ffn_grouped_activation_pad",
    source: "kernels/src/ffn_grouped_activation_pad.hip",
};
const M_TOTAL_TILES: Expr = Expr::DivCeil(&M_TOTAL, BLOCK_M);

const PARAMS_SITU_GROUPED: &[ParamV1] = &[
    param("m_total", "grouped (expert-sorted, padded) rows, grid x"),
    param("block_m", "rows per expert tile; pinned to 16 (MOE_GROUPED_BLOCK_M)"),
    param("logical_n", "activation columns computed (gate and up are each logical_n wide)"),
    param("phys_n", "output columns, >= logical_n; the pad is +0.0; grid y is ceil(phys_n / 256)"),
    param("n_experts", "AWQ table entries (0 without a table)"),
    param(
        "total_slots",
        "rows of out (flat slots); the launcher takes it from the out tensor; a slot outside it is skipped",
    ),
    param("round_inputs", "1: bf16-round g and u before the activation"),
    param("round_output", "1: bf16-round the activation result"),
    param_u32("p0_bits", "f32 bits of gate_beta"),
    param_u32("p1_bits", "f32 bits of value_beta"),
];
const OPS_SITU_GROUPED: &[OperandV1] = &[
    operand(
        "y_grouped",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[M_TOTAL, C2, LOGICAL_N, C4])),
        false,
        "F32 [m_total, 2*logical_n]: gate in cols 0..logical_n, up in cols logical_n..2*logical_n",
    ),
    operand(
        "sorted_slot_index",
        FORMAT_SORTED_SLOT_INDEX,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[M_TOTAL, C4])),
        false,
        "flat slot (token*top_k + rank) of grouped row g, or -1 for a padding row (skipped, nothing written)",
    ),
    operand(
        "expert_tile_ids",
        FORMAT_EXPERT_TILE_IDS,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[M_TOTAL_TILES, C4])),
        false,
        "read only when awq_table is non-null; the launcher checks the extent only then",
    ),
    operand(
        "awq_table",
        FORMAT_U64_AWQ_PTR_TABLE,
        Access::ReadOnly,
        8,
        Some(Expr::Mul(&[N_EXPERTS, C8])),
        true,
        "null: no AWQ divide; entries point at F32[phys_n] (their extent is the loader's contract)",
    ),
    operand(
        "out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[TOTAL_SLOTS, PHYS_N, C4])),
        false,
        "F32 [total_slots, phys_n]: row = flat slot (the un-scatter is fused here)",
    ),
];
const FIELDS_SITU_GROUPED: &[FieldV1] = &[
    ptr("y_grouped", 0),
    ptr("sorted_slot_index", 8),
    ptr("expert_tile_ids", 16),
    ptr("awq_table", 24),
    ptr("out", 32),
    i32f("m_total", 40),
    i32f("block_m", 44),
    i32f("logical_n", 48),
    i32f("phys_n", 52),
    i32f("n_experts", 56),
    i32f("total_slots", 60),
    i32f("round_inputs", 64),
    i32f("round_output", 68),
    f32f("p0", 72, "p0_bits"),
    f32f("p1", 76, "p1_bits"),
];
const DOMAIN_SITU_GROUPED: &[DomainRuleV1] = &[
    rule(Rule::Positive(M_TOTAL), "m_total > 0"),
    rule(
        Rule::Equal(p("block_m"), c(BLOCK_M)),
        "the contract fixes the 16-row expert tile (expert_tile_ids extent)",
    ),
    rule(
        Rule::Positive(LOGICAL_N),
        "logical_n > 0 (y_grouped extent)",
    ),
    rule(
        Rule::AtLeast(Expr::Sub(&PHYS_N, &LOGICAL_N), 0),
        "logical_n <= phys_n (a negative difference is rejected by the evaluation)",
    ),
    rule(Rule::Positive(TOTAL_SLOTS), "total_slots > 0"),
    rule(
        Rule::AtMost(Expr::DivCeil(&PHYS_N, 256), MAX_COL_TILES),
        "column tiles ride grid.y (<= 65535)",
    ),
    rule(Rule::OneOf(p("round_inputs"), FLAG_01), "flag is 0 or 1"),
    rule(Rule::OneOf(p("round_output"), FLAG_01), "flag is 0 or 1"),
];
const GRID_SITU_GROUPED: [Expr; 3] = [M_TOTAL, Expr::DivCeil(&PHYS_N, 256), ONE];
const EXPORTS_SITU_GROUPED: &[ExportV1] = &[
    export(
        "ffn_situ_grouped_f32",
        Arch::Gfx1201,
        OBJ_GROUPED,
        256,
        0,
        0,
        GRID_SITU_GROUPED,
    ),
    export(
        "ffn_situ_grouped_f32",
        Arch::Gfx1100,
        OBJ_GROUPED,
        256,
        0,
        0,
        GRID_SITU_GROUPED,
    ),
];

/// `ffn_situ_grouped_f32`: SiTU-GLU over expert-sorted grouped rows with the un-scatter fused in (row g
/// writes out row `sorted_slot_index[g]`); grid `(m_total, ceil(phys_n / 256))`, block 256, segment 80/8, no
/// hidden words. Contract fixes `block_m = 16`. gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_SITU_GROUPED_F32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_situ_grouped_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SITU,
    params: PARAMS_SITU_GROUPED,
    operands: OPS_SITU_GROUPED,
    fields: FIELDS_SITU_GROUPED,
    hidden: &[],
    explicit_bytes: 80,
    segment_bytes: 80,
    segment_align: 8,
    domain: DOMAIN_SITU_GROUPED,
    exports: EXPORTS_SITU_GROUPED,
};

pub static WORK_FFN_SITU_GROUPED_F32: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_SITU_GROUPED_F32,
    work: WorkV1 {
        op_class: op_class::ACTIVATION,
        codec: "f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: Some(Expr::Mul(&[M_TOTAL, LOGICAL_N, c(SITU_FLOP_PER_ELEM)])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[M_TOTAL, C2, LOGICAL_N, C4]),
            Expr::Mul(&[M_TOTAL, C4]),
            Expr::Mul(&[M_TOTAL, PHYS_N, C4]),
        ])),
        rows: None,
        admit: ALL_ROWS,
        bind: &[
            BindV1 {
                param: "phys_n",
                source: BindSource::Config(KEYS_MOE_INTERMEDIATE),
            },
            BindV1 {
                param: "logical_n",
                source: BindSource::Derived(PHYS_N),
            },
        ],
        note: "m_total is the padded routed-row count (grid.x, exact), not chunk rows, so no rows param; bytes count every grouped row (padding rows write nothing: upper bound); phys_n from config must lie in the grid.y tile interval; logical_n = phys_n unless down_k is padded; flops estimated (SITU_FLOP_PER_ELEM)",
    },
};

// ===========================================================================
// 7. ffn_moe_router_topk_f32
// ===========================================================================

const OBJ_ROUTER: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "ffn_router_renorm",
    source: "kernels/src/ffn_router_renorm.hip",
};
const KIND: Expr = p("kind");

const PARAMS_ROUTER: &[ParamV1] = &[
    param(
        "rows",
        "token rows, one 256-thread block each (grid x; not a kernarg)",
    ),
    param("n_experts", "router logits per row, 1..=1024"),
    param("top_k", "experts selected per row, 1..=8, <= n_experts"),
    param(
        "kind",
        "0 softmax + renorm, 1 sigmoid + optional bias + renorm",
    ),
    param_u32("eps_bits", "f32 bits of eps (kind 1)"),
    param_u32("scale_bits", "f32 bits of scale (kind 1)"),
];
const OPS_ROUTER: &[OperandV1] = &[
    operand(
        "logits",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[ROWS, N_EXPERTS, C4])),
        false,
        "F32 [rows, n_experts] raw router linear output",
    ),
    operand(
        "bias",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[N_EXPERTS, C4])),
        true,
        "F32 [n_experts] correction bias, selection only (kind 1); null = 0",
    ),
    operand(
        "idx_out",
        FORMAT_I32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[ROWS, TOP_K, C4])),
        false,
        "[rows, top_k] selected expert ids, descending selection score, ties to the lower id",
    ),
    operand(
        "w_out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[ROWS, TOP_K, C4])),
        false,
        "[rows, top_k] renormalised weights in selection order",
    ),
];
const FIELDS_ROUTER: &[FieldV1] = &[
    ptr("logits", 0),
    ptr("bias", 8),
    ptr("idx_out", 16),
    ptr("w_out", 24),
    i32f("n_experts", 32),
    i32f("top_k", 36),
    i32f("kind", 40),
    f32f("eps", 44, "eps_bits"),
    f32f("scale", 48, "scale_bits"),
];
const DOMAIN_ROUTER: &[DomainRuleV1] = &[
    rule(Rule::Positive(ROWS), "rows > 0"),
    rule(Rule::AtLeast(N_EXPERTS, 1), "n_experts >= 1"),
    rule(
        Rule::AtMost(N_EXPERTS, 1024),
        "LDS holds 1024 scores per row (FFN_ROUTER_MAX_E)",
    ),
    rule(Rule::AtLeast(TOP_K, 1), "top_k >= 1"),
    rule(
        Rule::AtMost(TOP_K, 8),
        "LDS holds eight picks (FFN_ROUTER_MAX_K)",
    ),
    rule(
        Rule::AtLeast(Expr::Sub(&N_EXPERTS, &TOP_K), 0),
        "top_k <= n_experts (a negative difference is rejected by the evaluation)",
    ),
    rule(Rule::OneOf(KIND, FLAG_01), "kind is 0 or 1"),
];
const GRID_ROUTER: [Expr; 3] = [ROWS, ONE, ONE];
const EXPORTS_ROUTER: &[ExportV1] = &[
    export(
        "ffn_moe_router_topk_f32",
        Arch::Gfx1201,
        OBJ_ROUTER,
        256,
        0,
        10336,
        GRID_ROUTER,
    ),
    export(
        "ffn_moe_router_topk_f32",
        Arch::Gfx1100,
        OBJ_ROUTER,
        256,
        0,
        10336,
        GRID_ROUTER,
    ),
];

/// `ffn_moe_router_topk_f32`: per-row router top-k with renormalised weights, one 256-thread block per row,
/// grid `(rows)`, static LDS 10336 B (`2*1024*4 + 2*256*4 + 8*4*3 + 8*4` as the compiler lays it out),
/// segment 52/8, no hidden words. gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_MOE_ROUTER_TOPK_F32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_moe_router_topk_f32",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.ffn_moe_router_topk.f32_lds_tree"),
    params: PARAMS_ROUTER,
    operands: OPS_ROUTER,
    fields: FIELDS_ROUTER,
    hidden: &[],
    explicit_bytes: 52,
    segment_bytes: 52,
    segment_align: 8,
    domain: DOMAIN_ROUTER,
    exports: EXPORTS_ROUTER,
};

pub static WORK_FFN_MOE_ROUTER_TOPK_F32: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_MOE_ROUTER_TOPK_F32,
    work: WorkV1 {
        op_class: op_class::MOE_ROUTE,
        codec: "f32",
        pipe: Pipe::ValuF32,
        macs: None,
        flops: Some(Expr::Mul(&[ROWS, N_EXPERTS, Expr::Add(&[c(12), TOP_K])])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[ROWS, N_EXPERTS, C4]),
            Expr::Mul(&[ROWS, TOP_K, C8]),
        ])),
        rows: Some("rows"),
        admit: ALL_ROWS,
        bind: &[
            BindV1 {
                param: "n_experts",
                source: BindSource::Config(KEYS_EXPERTS),
            },
            BindV1 {
                param: "top_k",
                source: BindSource::Config(KEYS_TOPK),
            },
        ],
        note: "rows exact from grid.x (one block per row); flops estimate one expf + divide + compare per expert (~12 FLOP-equivalents) plus top_k selection compares per expert; bias plane (4*n_experts B) not charged",
    },
};

// ===========================================================================
// 8. moe_scatter_fused_k8
// ===========================================================================

const OBJ_SCATTER: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "moe_scatter_fused_k8",
    source: "kernels/src/moe_scatter_fused_k8.hip",
};
const M_TOTAL_MAX: Expr = p("m_total_max");

const PARAMS_SCATTER: &[ParamV1] = &[
    param("total_slots", "routed slots (tokens * top_k)"),
    param("num_experts", "experts; pinned to 128 by the LDS size"),
    param(
        "m_total_max",
        "upper bound of the padded grouped rows (sentinel-fill extent)",
    ),
    param("block_m", "rows per expert tile; pinned to 16"),
];
const OPS_SCATTER: &[OperandV1] = &[
    operand(
        "topk_indices",
        FORMAT_I32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[TOTAL_SLOTS, C4])),
        false,
        "[total_slots] expert ids; an id outside [0, num_experts) is an unrouted slot",
    ),
    operand(
        "expert_token_counts",
        FORMAT_I32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[p("num_experts"), C4])),
        false,
        "[E] padded counts",
    ),
    operand(
        "expert_offsets",
        FORMAT_I32_FLAT,
        Access::ReadWrite,
        4,
        Some(Expr::Mul(&[Expr::Add(&[p("num_experts"), ONE]), C4])),
        false,
        "[E+1] exclusive scan of the padded counts; written, then read back by the same workgroup",
    ),
    operand(
        "sorted_slot_index",
        FORMAT_SORTED_SLOT_INDEX,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[M_TOTAL_MAX, C4])),
        false,
        "[m_total_max] flat slot per grouped row, -1 padding over the whole bound",
    ),
    operand(
        "expert_tile_ids",
        FORMAT_EXPERT_TILE_IDS,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[Expr::DivExact(&M_TOTAL_MAX, BLOCK_M), C4])),
        false,
        "[m_total_max / 16] expert of each tile, -1 beyond the live tiles",
    ),
    operand(
        "inverse_perm",
        FORMAT_I32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[TOTAL_SLOTS, C4])),
        false,
        "[total_slots] flat slot -> sorted position, -1 for an unrouted slot",
    ),
];
const FIELDS_SCATTER: &[FieldV1] = &[
    ptr("topk_indices", 0),
    ptr("expert_token_counts", 8),
    ptr("expert_offsets", 16),
    ptr("sorted_slot_index", 24),
    ptr("expert_tile_ids", 32),
    ptr("inverse_perm", 40),
    i32f("total_slots", 48),
    i32f("num_experts", 52),
    i32f("m_total_max", 56),
    i32f("block_m", 60),
];
/// Hidden words of the scatter (it reads `blockDim`, so hipcc keeps the implicit-arg block and the dynamic
/// LDS size word); same layout on gfx1201 and gfx1100, segment 320.
const HIDDEN_SCATTER: &[HiddenV1] = &[
    hid(64, 4, HiddenKind::BlockCount(Axis::X)),
    hid(68, 4, HiddenKind::BlockCount(Axis::Y)),
    hid(72, 4, HiddenKind::BlockCount(Axis::Z)),
    hid(76, 2, HiddenKind::GroupSize(Axis::X)),
    hid(78, 2, HiddenKind::GroupSize(Axis::Y)),
    hid(80, 2, HiddenKind::GroupSize(Axis::Z)),
    hid(82, 2, HiddenKind::Remainder(Axis::X)),
    hid(84, 2, HiddenKind::Remainder(Axis::Y)),
    hid(86, 2, HiddenKind::Remainder(Axis::Z)),
    hid(104, 8, HiddenKind::GlobalOffset(Axis::X)),
    hid(112, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hid(120, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hid(128, 2, HiddenKind::GridDims),
    hid(184, 4, HiddenKind::DynamicLdsSize),
];
const DOMAIN_SCATTER: &[DomainRuleV1] = &[
    rule(Rule::Positive(TOTAL_SLOTS), "total_slots > 0"),
    rule(
        Rule::Equal(p("num_experts"), c(SCATTER_EXPERTS)),
        "dynamic LDS is 4 * num_experts bytes and the export's dynamic_lds is a constant (512)",
    ),
    rule(
        Rule::Equal(p("block_m"), c(BLOCK_M)),
        "MOE_GROUPED_BLOCK_M: the tile-id extent divides by 16",
    ),
    rule(
        Rule::MultipleOf(M_TOTAL_MAX, BLOCK_M),
        "m_total_max / block_m tile ids are initialised",
    ),
    rule(
        Rule::Positive(M_TOTAL_MAX),
        "m_total_max > 0 (the padded rows must fit it; not checkable here: it depends on the routing)",
    ),
];
const GRID_SCATTER: [Expr; 3] = [ONE, ONE, ONE];
const EXPORTS_SCATTER: &[ExportV1] = &[
    export(
        "moe_scatter_fused_k8",
        Arch::Gfx1201,
        OBJ_SCATTER,
        256,
        SCATTER_EXPERTS as u32 * 4,
        0,
        GRID_SCATTER,
    ),
    export(
        "moe_scatter_fused_k8",
        Arch::Gfx1100,
        OBJ_SCATTER,
        256,
        SCATTER_EXPERTS as u32 * 4,
        0,
        GRID_SCATTER,
    ),
];

/// `moe_scatter_fused_k8`: single-workgroup histogram + padded prefix sum + permutation producing the six
/// grouped-GEMM tables; grid `[1,1,1]`, block 256, dynamic LDS `4 * num_experts` (pinned: 128 experts, 512 B),
/// segment 320/8 (64 explicit B + hidden block). Slots land in LDS-atomic arrival order. On exact gfx1201
/// with `>= 4` chunks of 1024 slots the launcher selects the multi-workgroup stable twin instead (not this
/// symbol).
pub static MOE_SCATTER_FUSED_K8: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.moe_scatter_fused_k8.e128",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.moe_scatter_fused_k8.lds_atomic_arrival_order"),
    params: PARAMS_SCATTER,
    operands: OPS_SCATTER,
    fields: FIELDS_SCATTER,
    hidden: HIDDEN_SCATTER,
    explicit_bytes: 64,
    segment_bytes: 320,
    segment_align: 8,
    domain: DOMAIN_SCATTER,
    exports: EXPORTS_SCATTER,
};

pub static WORK_MOE_SCATTER_FUSED_K8: WorkSpecV1 = WorkSpecV1 {
    abi: &MOE_SCATTER_FUSED_K8,
    work: WorkV1 {
        op_class: op_class::MOE_SCATTER,
        codec: "i32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[TOTAL_SLOTS, C8]),
            Expr::Mul(&[p("num_experts"), c(12)]),
            Expr::Mul(&[M_TOTAL_MAX, C4]),
            Expr::Mul(&[Expr::DivExact(&M_TOTAL_MAX, BLOCK_M), C4]),
        ])),
        rows: None,
        admit: ALL_ROWS,
        bind: &[
            BindV1 {
                param: "rows",
                source: BindSource::ChunkRows,
            },
            BindV1 {
                param: "topk",
                source: BindSource::Config(KEYS_TOPK),
            },
            BindV1 {
                param: "total_slots",
                source: BindSource::Derived(Expr::Mul(&[p("rows"), p("topk")])),
            },
            BindV1 {
                param: "num_experts",
                source: BindSource::Config(KEYS_EXPERTS),
            },
            // dispatch `grouped_moe_m_total_bound`: ceil16(slots + min(slots, E) * 15); slots >= E assumed.
            BindV1 {
                param: "m_total_max",
                source: BindSource::Derived(Expr::Mul(&[
                    Expr::DivCeil(
                        &Expr::Add(&[TOTAL_SLOTS, Expr::Mul(&[p("num_experts"), c(15)])]),
                        BLOCK_M,
                    ),
                    c(BLOCK_M),
                ])),
            },
        ],
        note: "single workgroup: the grid carries no rows or experts, everything binds from the chunk rows and config; m_total_max follows dispatch's grouped_moe_m_total_bound assuming total_slots >= num_experts [INFERENCE: caller-sized bound, not visible in the trace]; bytes: slots read once + inverse_perm written + counts/offsets + sorted index + tile ids (LDS traffic free)",
    },
};

// ===========================================================================
// 9. ffn_single_expert_tables
// ===========================================================================

const ROWS_PAD: Expr = p("rows_pad");

const PARAMS_TABLES: &[ParamV1] = &[
    param(
        "rows",
        "dense rows driven through the grouped core as expert 0",
    ),
    param("rows_pad", "rows rounded up to a multiple of 16"),
    param_u64("w0", "device address of the first dense weight"),
    param_u64("w1", "device address of the second dense weight"),
    param_u64("w2", "device address of the third dense weight"),
];
const OPS_TABLES: &[OperandV1] = &[operand(
    "tables",
    FORMAT_SINGLE_EXPERT_TABLES,
    Access::WriteOnly,
    8,
    Some(Expr::Mul(&[
        Expr::Add(&[
            c(6),
            Expr::Mul(&[C2, ROWS_PAD]),
            Expr::DivExact(&ROWS_PAD, 16),
        ]),
        C4,
    ])),
    false,
    "words [0,6): three u64 weight pointers; [6, 6+rows_pad): expert index (all 0); then sorted_slot_index (i for i < rows, else -1); then rows_pad/16 tile ids (all 0)",
)];
const FIELDS_TABLES: &[FieldV1] = &[
    ptr("tables", 0),
    i32f("rows", 8),
    i32f("rows_pad", 12),
    u64f("w0", 16),
    u64f("w1", 24),
    u64f("w2", 32),
];
const DOMAIN_TABLES: &[DomainRuleV1] = &[
    rule(Rule::Positive(ROWS), "rows > 0"),
    rule(
        Rule::Equal(ROWS_PAD, Expr::Mul(&[Expr::DivCeil(&ROWS, 16), c(16)])),
        "rows_pad is rows rounded up to a multiple of 16",
    ),
];
const GRID_TABLES: [Expr; 3] = [Expr::DivCeil(&ROWS_PAD, 256), ONE, ONE];
const EXPORTS_TABLES: &[ExportV1] = &[
    export(
        "ffn_single_expert_tables",
        Arch::Gfx1201,
        OBJ_GROUPED,
        256,
        0,
        0,
        GRID_TABLES,
    ),
    export(
        "ffn_single_expert_tables",
        Arch::Gfx1100,
        OBJ_GROUPED,
        256,
        0,
        0,
        GRID_TABLES,
    ),
];

/// `ffn_single_expert_tables(tables, rows, rows_pad, w0, w1, w2)`: fills the grouping tables that let one
/// dense weight run through the grouped WMMA core / indexed GEMV as expert 0; grid `ceil(rows_pad / 256)`
/// (the launcher's `.max(1)` never binds: `rows_pad >= 16`), block 256, segment 40/8, no hidden words; the
/// weight addresses are by-value u64 kernargs. gfx1100 layout from a local hipcc build (see the module docs).
pub static FFN_SINGLE_EXPERT_TABLES: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.ffn_single_expert_tables",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.single_expert_tables.words"),
    params: PARAMS_TABLES,
    operands: OPS_TABLES,
    fields: FIELDS_TABLES,
    hidden: &[],
    explicit_bytes: 40,
    segment_bytes: 40,
    segment_align: 8,
    domain: DOMAIN_TABLES,
    exports: EXPORTS_TABLES,
};

pub static WORK_FFN_SINGLE_EXPERT_TABLES: WorkSpecV1 = WorkSpecV1 {
    abi: &FFN_SINGLE_EXPERT_TABLES,
    work: WorkV1 {
        op_class: op_class::MOE_SCATTER,
        codec: "i32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Mul(&[
            Expr::Add(&[
                c(6),
                Expr::Mul(&[C2, ROWS_PAD]),
                Expr::DivExact(&ROWS_PAD, 16),
            ]),
            C4,
        ])),
        rows: Some("rows"),
        admit: ALL_ROWS,
        bind: &[
            BindV1 {
                param: "rows",
                source: BindSource::ChunkRows,
            },
            BindV1 {
                param: "rows_pad",
                source: BindSource::Derived(Expr::Mul(&[Expr::DivCeil(&ROWS, 16), c(16)])),
            },
        ],
        note: "slot planning for a dense weight; the grid gives rows_pad only to a 256-row tile, so rows binds to the chunk rows (dense single-expert path runs on the chunk)",
    },
};

// ===========================================================================
// 10-11. FWHT256 rotations: mq_rotate_x, rotate_x_mq_awq
// ===========================================================================

const K: Expr = p("K");
const BATCH: Expr = p("batch");
const GROUPS_PER_ROW: Expr = Expr::DivExact(&K, 256);
const BATCH_X_K: Expr = Expr::Mul(&[BATCH, K]);
const SIGNS_EXTENT: Expr = c(256 * 4);

/// FLOP per rotated element: eight butterfly stages, the sign multiply, and the final normalisation.
const MQ_ROTATE_FLOP_PER_ELEM: u64 = 10;

const OBJ_MQ_ROTATE: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "mq_rotate_x",
    source: "kernels/src/gemv_mq4g256.hip",
};
const OBJ_ROTATE_AWQ: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "rotate_x_mq_awq",
    source: "kernels/src/rotate_x_mq_awq.hip",
};

const PARAMS_ROTATE: &[ParamV1] = &[
    param("K", "row length, a multiple of 256 (groups of 256 per row)"),
    param("batch", "rows rotated; folded into grid x as batch * K/256"),
];
const DOMAIN_ROTATE: &[DomainRuleV1] = &[
    rule(Rule::Positive(K), "K > 0"),
    rule(Rule::MultipleOf(K, 256), "whole 256-element FWHT groups"),
    rule(Rule::Positive(BATCH), "batch > 0"),
    rule(
        Rule::AtMost(Expr::Mul(&[BATCH, GROUPS_PER_ROW]), I32_MAX),
        "the folded one-dimensional grid.x limit",
    ),
];
const GRID_ROTATE: [Expr; 3] = [Expr::Mul(&[BATCH, GROUPS_PER_ROW]), ONE, ONE];

// ---- mq_rotate_x ------------------------------------------------------------

const OPS_MQ_ROTATE: &[OperandV1] = &[
    operand(
        "x_in",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[BATCH_X_K, C4])),
        false,
        "F32 [batch, K] row-major",
    ),
    operand(
        "x_out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[BATCH_X_K, C4])),
        false,
        "F32 [batch, K]; must not alias x_in (__restrict__)",
    ),
    operand(
        "signs1",
        FORMAT_MQ_FWHT_SIGNS_256,
        Access::ReadOnly,
        4,
        Some(SIGNS_EXTENT),
        false,
        "",
    ),
    operand(
        "signs2",
        FORMAT_MQ_FWHT_SIGNS_256,
        Access::ReadOnly,
        4,
        Some(SIGNS_EXTENT),
        false,
        "",
    ),
];
const FIELDS_MQ_ROTATE: &[FieldV1] = &[
    ptr("x_in", 0),
    ptr("x_out", 8),
    ptr("signs1", 16),
    ptr("signs2", 24),
    i32f("K", 32),
];
const EXPORTS_MQ_ROTATE: &[ExportV1] = &[
    export(
        "mq_rotate_x",
        Arch::Gfx1201,
        OBJ_MQ_ROTATE,
        32,
        0,
        0,
        GRID_ROTATE,
    ),
    export(
        "mq_rotate_x",
        Arch::Gfx1100,
        OBJ_MQ_ROTATE,
        32,
        0,
        0,
        GRID_ROTATE,
    ),
];

/// `mq_rotate_x(x_in, x_out, signs1, signs2, K)`: per 256-element group a sign multiply and a Walsh-Hadamard
/// butterfly (the MagnumQuant input rotation), one 32-lane workgroup per group, grid `batch * K/256` (the
/// launcher folds the batch into x), segment 36/8, no hidden words. The gfx942 build
/// (`mq_rotate_x_gfx942`, block 64) is a different symbol and not covered. `gemv_mq4g256` in the same
/// source is another contract's business.
pub static MQ_ROTATE_X: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.mq_rotate_x",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.mq_rotate_x.fwht256_f32"),
    params: PARAMS_ROTATE,
    operands: OPS_MQ_ROTATE,
    fields: FIELDS_MQ_ROTATE,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_ROTATE,
    exports: EXPORTS_MQ_ROTATE,
};

pub static WORK_MQ_ROTATE_X: WorkSpecV1 = WorkSpecV1 {
    abi: &MQ_ROTATE_X,
    work: WorkV1 {
        op_class: op_class::ROTATE,
        codec: "f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: Some(Expr::Mul(&[BATCH_X_K, c(MQ_ROTATE_FLOP_PER_ELEM)])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[BATCH_X_K, C8]),
            Expr::Mul(&[C2, SIGNS_EXTENT]),
        ])),
        rows: Some("batch"),
        admit: ALL_ROWS,
        bind: &[],
        note: "the grid carries batch * K/256 only and batch is not the chunk rows on the grouped-MoE path (batch = rows*top_k slots with K = down_k; Raon: 15360 groups = 3072 slots x K 1280), so batch and K stay a joint interval here and the binder must split them from a second source; signs planes counted once per launch",
    },
};

// ---- rotate_x_mq_awq --------------------------------------------------------

const OPS_ROTATE_AWQ: &[OperandV1] = &[
    operand(
        "x_in",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[BATCH_X_K, C4])),
        false,
        "F32 [batch, K] row-major",
    ),
    operand(
        "x_out",
        FORMAT_F32_FLAT,
        Access::WriteOnly,
        4,
        Some(Expr::Mul(&[BATCH_X_K, C4])),
        false,
        "F32 [batch, K]; must not alias x_in (__restrict__)",
    ),
    operand(
        "awq_scale",
        FORMAT_F32_FLAT,
        Access::ReadOnly,
        4,
        Some(Expr::Mul(&[K, C4])),
        false,
        "F32 [K] per-channel AWQ scale; each input element is divided by it before the rotation",
    ),
    operand(
        "signs1",
        FORMAT_MQ_FWHT_SIGNS_256,
        Access::ReadOnly,
        4,
        Some(SIGNS_EXTENT),
        false,
        "",
    ),
    operand(
        "signs2",
        FORMAT_MQ_FWHT_SIGNS_256,
        Access::ReadOnly,
        4,
        Some(SIGNS_EXTENT),
        false,
        "",
    ),
];
const FIELDS_ROTATE_AWQ: &[FieldV1] = &[
    ptr("x_in", 0),
    ptr("x_out", 8),
    ptr("awq_scale", 16),
    ptr("signs1", 24),
    ptr("signs2", 32),
    i32f("K", 40),
];
const EXPORTS_ROTATE_AWQ: &[ExportV1] = &[
    export(
        "rotate_x_mq_awq",
        Arch::Gfx1201,
        OBJ_ROTATE_AWQ,
        32,
        0,
        0,
        GRID_ROTATE,
    ),
    export(
        "rotate_x_mq_awq",
        Arch::Gfx1100,
        OBJ_ROTATE_AWQ,
        32,
        0,
        0,
        GRID_ROTATE,
    ),
];

/// `rotate_x_mq_awq(x_in, x_out, awq_scale, signs1, signs2, K)`: `mq_rotate_x` with a per-channel divide by
/// `awq_scale[K]` before the sign multiply and butterfly; grid `batch * K/256` (folded one-dimensional grid,
/// also for the unbatched launcher with `batch = 1`), block 32, segment 44/8, no hidden words.
pub static ROTATE_X_MQ_AWQ: KernelAbiV1 = KernelAbiV1 {
    id: "generic_elem.rotate_x_mq_awq",
    version: 1,
    wave: 32,
    numeric: NumericContractId("elem.rotate_x_mq_awq.div_then_fwht256_f32"),
    params: PARAMS_ROTATE,
    operands: OPS_ROTATE_AWQ,
    fields: FIELDS_ROTATE_AWQ,
    hidden: &[],
    explicit_bytes: 44,
    segment_bytes: 44,
    segment_align: 8,
    domain: DOMAIN_ROTATE,
    exports: EXPORTS_ROTATE_AWQ,
};

pub static WORK_ROTATE_X_MQ_AWQ: WorkSpecV1 = WorkSpecV1 {
    abi: &ROTATE_X_MQ_AWQ,
    work: WorkV1 {
        op_class: op_class::ROTATE,
        codec: "f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: Some(Expr::Mul(&[BATCH_X_K, c(MQ_ROTATE_FLOP_PER_ELEM + 1)])),
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[BATCH_X_K, C8]),
            Expr::Mul(&[K, C4]),
            Expr::Mul(&[C2, SIGNS_EXTENT]),
        ])),
        rows: Some("batch"),
        admit: ALL_ROWS,
        bind: &[BindV1 {
            param: "batch",
            source: BindSource::ChunkRows,
        }],
        note: "the grid carries batch * K/256 only; batch binds to the chunk rows (the post-projection input rotation runs on the chunk) and K follows from the grid (Raon: 5120 groups = 512 x K 2560); the AWQ divide counts one extra FLOP per element; scale and signs planes counted once per launch",
    },
};

// ---------------------------------------------------------------------------

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &CONVERT_F32_TO_F16,
    &BF16_ROUND_TRIP_F32_STRIDED,
    &FFN_ROUND_BF16_F32,
    &FFN_MOE_COMBINE_F32,
    &FFN_SITU_GROUPED_F32,
    &FFN_SITU_ROWS_F32,
    &FFN_MOE_ROUTER_TOPK_F32,
    &MOE_SCATTER_FUSED_K8,
    &FFN_SINGLE_EXPERT_TABLES,
    &MQ_ROTATE_X,
    &ROTATE_X_MQ_AWQ,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &WORK_CONVERT_F32_TO_F16,
    &WORK_BF16_ROUND_TRIP_F32_STRIDED,
    &WORK_FFN_ROUND_BF16_F32,
    &WORK_FFN_MOE_COMBINE_F32,
    &WORK_FFN_SITU_GROUPED_F32,
    &WORK_FFN_SITU_ROWS_F32,
    &WORK_FFN_MOE_ROUTER_TOPK_F32,
    &WORK_MOE_SCATTER_FUSED_K8,
    &WORK_FFN_SINGLE_EXPERT_TABLES,
    &WORK_MQ_ROTATE_X,
    &WORK_ROTATE_X_MQ_AWQ,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval;
    use crate::families::{find_export, work_for};

    fn ev(abi: &KernelAbiV1, arch: Arch, symbol: &str, e: &Expr, params: &[u64]) -> u64 {
        let export = abi.export(arch, symbol).expect("export");
        eval(e, abi, export, params).expect("evaluates")
    }

    /// Grid x/y in rocprof work-items (workgroups x block) for the observed Raon dispatch.
    fn items(abi: &KernelAbiV1, arch: Arch, symbol: &str, params: &[u64]) -> (u64, u64) {
        let export = abi.export(arch, symbol).expect("export");
        let g: Vec<u64> = export
            .grid
            .iter()
            .map(|e| eval(e, abi, export, params).expect("grid"))
            .collect();
        (
            g[0] * u64::from(export.block[0]),
            g[1] * u64::from(export.block[1]),
        )
    }

    #[test]
    fn every_contract_self_checks() {
        assert_eq!(ABIS.len(), 11);
        for abi in ABIS {
            if let Err(e) = abi.self_check() {
                panic!("{}: {e:#?}", abi.id);
            }
        }
    }

    #[test]
    fn every_work_spec_self_checks_and_pairs_one_to_one() {
        assert_eq!(WORK.len(), ABIS.len());
        for (w, a) in WORK.iter().zip(ABIS.iter()) {
            assert!(std::ptr::eq(w.abi, *a), "{} paired out of order", a.id);
            w.self_check().unwrap_or_else(|e| panic!("{e}"));
            assert!(work_for(a.id).is_some());
        }
    }

    #[test]
    fn every_symbol_resolves_on_both_arches() {
        for abi in ABIS {
            for e in abi.exports {
                let (found, ex) = find_export(e.arch, e.symbol).expect("registered");
                assert!(std::ptr::eq(found, *abi));
                assert_eq!(ex.symbol, e.symbol);
            }
            for arch in [Arch::Gfx1201, Arch::Gfx1100] {
                assert!(
                    abi.exports.iter().any(|e| e.arch == arch),
                    "{} lacks {}",
                    abi.id,
                    arch.name()
                );
            }
        }
    }

    #[test]
    fn work_expressions_evaluate_for_a_worked_example() {
        // combine at the Raon geometry (rows 512, hidden 2560, top-k 6).
        let w = &WORK_FFN_MOE_COMBINE_F32.work;
        let abi = &FFN_MOE_COMBINE_F32;
        let params = [512, 2560, 6, 128, 0, 0, 0, 0, 3072];
        let bytes = ev(
            abi,
            Arch::Gfx1201,
            "ffn_moe_combine_f32",
            &w.bytes.unwrap(),
            &params,
        );
        assert_eq!(bytes, 512 * 6 * 2560 * 4 + 512 * 6 * 8 + 512 * 2560 * 4);
        // convert: 6 B per element.
        let w = &WORK_CONVERT_F32_TO_F16.work;
        let bytes = ev(
            &CONVERT_F32_TO_F16,
            Arch::Gfx1201,
            "convert_f32_to_f16",
            &w.bytes.unwrap(),
            &[1_310_720],
        );
        assert_eq!(bytes, 7_864_320);
    }

    // ---- grid inversions: the observed Raon pp4096 dispatches ------------------------------------

    #[test]
    fn grid_convert_f32_to_f16_raon() {
        // (1310720, 1)/256: 5120 workgroups; n = 512 rows x K 2560 is the upper end of the tile interval.
        let (x, y) = items(
            &CONVERT_F32_TO_F16,
            Arch::Gfx1201,
            "convert_f32_to_f16",
            &[1_310_720],
        );
        assert_eq!((x, y), (1_310_720, 1));
        // n = 1310465 is the lowest n that still needs 5120 workgroups.
        let (x, _) = items(
            &CONVERT_F32_TO_F16,
            Arch::Gfx1201,
            "convert_f32_to_f16",
            &[1_310_465],
        );
        assert_eq!(x, 1_310_720);
        let (x, _) = items(
            &CONVERT_F32_TO_F16,
            Arch::Gfx1201,
            "convert_f32_to_f16",
            &[1_310_464],
        );
        assert_eq!(x, 1_310_464);
    }

    #[test]
    fn grid_ffn_moe_combine_f32_raon() {
        // (131072, 10)/256 = 512 rows x hidden in [2305, 2560]; hidden 2560.
        let (x, y) = items(
            &FFN_MOE_COMBINE_F32,
            Arch::Gfx1201,
            "ffn_moe_combine_f32",
            &[512, 2560, 6, 128, 0, 0, 0, 0, 3072],
        );
        assert_eq!((x, y), (131_072, 10));
    }

    #[test]
    fn grid_ffn_situ_grouped_f32_raon() {
        // (1277952, 5)/256 = 4992 grouped rows x phys_n in [1025, 1280]; m_total 4992, phys_n 1280.
        let params = [4992, 16, 1280, 1280, 128, 3072, 1, 1, 0, 0];
        let (x, y) = items(
            &FFN_SITU_GROUPED_F32,
            Arch::Gfx1201,
            "ffn_situ_grouped_f32",
            &params,
        );
        assert_eq!((x, y), (1_277_952, 5));
        // 4992 = grouped_moe_m_total_bound(512 * 6 slots, 128 experts).
        assert_eq!(4992, (3072u64 + 128 * 15).div_ceil(16) * 16);
    }

    #[test]
    fn grid_ffn_situ_rows_f32_tiles_columns_on_y() {
        // rows 512, phys_n 1088 (a shared-expert width): 5 column tiles.
        let params = [512, 1088, 1088, 1088, 1088, 0, 0, 0, 0, 0, 0];
        let (x, y) = items(
            &FFN_SITU_ROWS_F32,
            Arch::Gfx1100,
            "ffn_situ_rows_f32",
            &params,
        );
        assert_eq!((x, y), (512 * 256, 5));
    }

    #[test]
    fn grid_ffn_moe_router_topk_f32_raon() {
        // (131072, 1)/256 = 512 rows, static LDS 10336.
        let params = [512, 128, 6, 0, 0, 0];
        let (x, y) = items(
            &FFN_MOE_ROUTER_TOPK_F32,
            Arch::Gfx1201,
            "ffn_moe_router_topk_f32",
            &params,
        );
        assert_eq!((x, y), (131_072, 1));
        let ex = FFN_MOE_ROUTER_TOPK_F32
            .export(Arch::Gfx1201, "ffn_moe_router_topk_f32")
            .unwrap();
        assert_eq!(ex.group_segment_fixed, 10336);
    }

    #[test]
    fn grid_mq_rotate_x_raon() {
        // (491520, 1)/32 = 15360 groups = 3072 slots x K 1280 (K/256 = 5).
        let (x, _) = items(&MQ_ROTATE_X, Arch::Gfx1201, "mq_rotate_x", &[1280, 3072]);
        assert_eq!(x, 491_520);
        // the unbatched launcher is batch 1.
        let (x, _) = items(&MQ_ROTATE_X, Arch::Gfx1201, "mq_rotate_x", &[2560, 1]);
        assert_eq!(x, 10 * 32);
    }

    #[test]
    fn grid_rotate_x_mq_awq_raon() {
        // (163840, 1)/32 = 5120 groups = 512 rows x K 2560.
        let (x, _) = items(
            &ROTATE_X_MQ_AWQ,
            Arch::Gfx1201,
            "rotate_x_mq_awq",
            &[2560, 512],
        );
        assert_eq!(x, 163_840);
        let (x, _) = items(
            &ROTATE_X_MQ_AWQ,
            Arch::Gfx1100,
            "rotate_x_mq_awq",
            &[2560, 512],
        );
        assert_eq!(x, 163_840);
    }

    #[test]
    fn grid_moe_scatter_fused_k8_raon() {
        // (256, 1)/256, dynamic LDS 512 (128 experts x 4 B).
        let params = [3072, 128, 4992, 16];
        let (x, y) = items(
            &MOE_SCATTER_FUSED_K8,
            Arch::Gfx1201,
            "moe_scatter_fused_k8",
            &params,
        );
        assert_eq!((x, y), (256, 1));
        let ex = MOE_SCATTER_FUSED_K8
            .export(Arch::Gfx1201, "moe_scatter_fused_k8")
            .unwrap();
        assert_eq!(ex.dynamic_lds, 512);
        // the dispatch bound the work spec derives for m_total_max.
        let w = &WORK_MOE_SCATTER_FUSED_K8.work;
        let BindSource::Derived(e) = w.bind[4].source else {
            panic!("m_total_max binding is Derived");
        };
        assert_eq!(
            ev(
                &MOE_SCATTER_FUSED_K8,
                Arch::Gfx1201,
                "moe_scatter_fused_k8",
                &e,
                &[3072, 128, 0, 16]
            ),
            4992
        );
    }

    #[test]
    fn grid_ffn_single_expert_tables_and_round_and_bf16() {
        // rows 512: rows_pad 512, grid 2.
        let (x, _) = items(
            &FFN_SINGLE_EXPERT_TABLES,
            Arch::Gfx1201,
            "ffn_single_expert_tables",
            &[512, 512, 1, 2, 3],
        );
        assert_eq!(x, 2 * 256);
        // rows 1: rows_pad 16, still one workgroup.
        let (x, _) = items(
            &FFN_SINGLE_EXPERT_TABLES,
            Arch::Gfx1201,
            "ffn_single_expert_tables",
            &[1, 16, 1, 2, 3],
        );
        assert_eq!(x, 256);
        let (x, _) = items(
            &FFN_ROUND_BF16_F32,
            Arch::Gfx1201,
            "ffn_round_bf16_f32",
            &[1000],
        );
        assert_eq!(x, 4 * 256);
        // rows 512 x width 128: 256 blocks.
        let (x, _) = items(
            &BF16_ROUND_TRIP_F32_STRIDED,
            Arch::Gfx1100,
            "bf16_round_trip_f32_strided",
            &[512, 0, 128, 128],
        );
        assert_eq!(x, 256 * 256);
    }

    #[test]
    fn convert_invocation_validates_and_encodes() {
        use crate::{encode_into, validate_invocation, InvocationV1, OperandBinding};
        let n = 1_310_720u64;
        let ops = [
            OperandBinding {
                address: 0x1000,
                bytes: Some(n * 4),
                format: FORMAT_F32_FLAT,
            },
            OperandBinding {
                address: 0x2_0000,
                bytes: Some(n * 2),
                format: FORMAT_F16_FLAT,
            },
        ];
        let inv = InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: "convert_f32_to_f16",
            params: &[n],
            operands: &ops,
            grid: [5120, 1, 1],
            block: [256, 1, 1],
            dynamic_lds: 0,
        };
        let v = validate_invocation(&CONVERT_F32_TO_F16, &inv).expect("valid");
        let mut buf = [0u8; 20];
        encode_into(&v, &mut buf).expect("encodes");
        assert_eq!(u64::from_le_bytes(buf[0..8].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(buf[8..16].try_into().unwrap()), 0x2_0000);
        assert_eq!(
            u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            n as u32
        );
        // a wrong grid is rejected before enqueue.
        let bad = InvocationV1 {
            grid: [5119, 1, 1],
            ..inv
        };
        assert!(validate_invocation(&CONVERT_F32_TO_F16, &bad).is_err());
    }

    #[test]
    fn scatter_invocation_pins_dynamic_lds_and_experts() {
        use crate::{validate_invocation, InvocationV1, OperandBinding};
        let b = |bytes: u64, format| OperandBinding {
            address: 0x10_000,
            bytes: Some(bytes),
            format,
        };
        let (slots, e, mtm) = (3072u64, 128u64, 4992u64);
        let ops = [
            b(slots * 4, FORMAT_I32_FLAT),
            b(e * 4, FORMAT_I32_FLAT),
            b((e + 1) * 4, FORMAT_I32_FLAT),
            b(mtm * 4, FORMAT_SORTED_SLOT_INDEX),
            b(mtm / 16 * 4, FORMAT_EXPERT_TILE_IDS),
            b(slots * 4, FORMAT_I32_FLAT),
        ];
        let inv = InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: "moe_scatter_fused_k8",
            params: &[slots, e, mtm, 16],
            operands: &ops,
            grid: [1, 1, 1],
            block: [256, 1, 1],
            dynamic_lds: 512,
        };
        assert!(validate_invocation(&MOE_SCATTER_FUSED_K8, &inv).is_ok());
        let wrong_lds = InvocationV1 {
            dynamic_lds: 1024,
            ..inv
        };
        assert!(validate_invocation(&MOE_SCATTER_FUSED_K8, &wrong_lds).is_err());
        let wrong_experts = InvocationV1 {
            params: &[slots, 256, mtm, 16],
            ..inv
        };
        assert!(validate_invocation(&MOE_SCATTER_FUSED_K8, &wrong_experts).is_err());
    }
}
