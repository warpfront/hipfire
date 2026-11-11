// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-part lane-WMMA GEMM entries of qwen4 on exact gfx1201
//! (`crates/rdna-compute/src/lane_wmma.rs` `lane_wmma_gemm_multi`,
//! `kernels/src/qwen4_lane_wmma.gfx1201.hip` `LWMMA_GEMM_MULTI_ENTRY`):
//! `lwmma_gemm_bf16_t2_m{1,2,3,4}_n1_multi`, up to four BF16-weight projections `Y_p = X W_p^T`
//! of the SAME `X [rows, K]` (F32) in ONE launch, each part with its own `M_p` and split-K `S_p`.
//!
//! One 128-byte explicit kernarg image for every entry (hipcc-built, `HipccSource`):
//!
//! | off | field | kind |
//! |---|---|---|
//! | 0/8/16/24 | `W0..W3` | pointers, BF16 `[M_p][K]` |
//! | 32/40/48/56 | `Y0..Y3` | pointers, F32 `[rows][M_p]`, overwritten |
//! | 64 | `X` | pointer, F32 `[rows][K]` |
//! | 72 | `partial` | pointer, F32 split-K scratch, read/write |
//! | 80 | `counters` | pointer, u32 tile counters, read/write (left zeroed) |
//! | 88/92/96/100 | `M0..M3` | i32 (0 = empty part) |
//! | 104/108/112/116 | `S0..S3` | i32 split-K per part |
//! | 120 | `K` | i32 |
//! | 124 | `rows` | i32 |
//!
//! The kernel reads `gridDim.x`, so the object carries the runtime's hidden block: block counts
//! `128/132/136`, group sizes `140/142/144`, remainders `146/148/150`, global offsets
//! `168/176/184`, grid dims `192`, `.kernarg_segment_size` 384 / align 8 (no dynamic-LDS word: the
//! kernel has no dynamic LDS). Launch: grid `[sum_p ceil(M_p/16), max_p S_p, ceil(rows/64)]`,
//! block 128, dynamic LDS 0; the entry `m<MT>` (`MT = min(ceil(rows/16), 4)`) has static LDS
//! `4096 * MT + 4`. A part's pointers are never null: the launcher repeats the last real part's
//! pointers into the empty slots (`M = 0`, `S = 1`).
//!
//! Every output word of part `p` is bitwise what the plain `lwmma_gemm_bf16_t2_m<MT>_n1` launch
//! of that part alone writes: the blocks run the same body with the part's own `(W, Y, M,
//! split_k)`, a part-private slice of `partial` and the counter slot of the block's flat tile.
//!
//! Admission that is not a function of the launch params stays in the launcher
//! (`lane_wmma_multi_plan`: 2..=4 non-empty parts, each `S_p` the auto split-K of its part, scratch
//! fit, `HIPFIRE_QWEN4_LWMMA_MULTI_PART`, `lane_wmma_applies`).

use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind, HiddenV1,
    KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- numeric contract / formats --------------------------------------------------------------

/// Per output word: exactly the per-part plain bf16 two-term lane-WMMA entry's value (same body,
/// same split-K fold from `0.0f`), F32 store.
pub const NUMERIC_BF16_T2_MULTI_F32: NumericContractId =
    NumericContractId("qwen4_lane_wmma.bf16_t2_multi.f32.v1");

/// BF16 weights `[M][K]`, row-major, 2 bytes per element.
pub const FORMAT_BF16_ROWS_K: FormatId = FormatId("bf16.rows_x_k");
/// F32 activations `[rows][K]`.
pub const FORMAT_F32_ROWS_K: FormatId = FormatId("f32.rows_x_k");
/// F32 outputs `[rows][M]`.
pub const FORMAT_F32_ROWS_M: FormatId = FormatId("f32.rows_x_m");
/// F32 split-K partial scratch: per part with `S_p > 1`, `S_p * rows * M_p` floats back to back.
pub const FORMAT_F32_SPLITK_PARTIAL: FormatId = FormatId("f32.splitk_partial.parts_back_to_back");
/// u32 split-K arrival counters, one per `(z, flat tile)`; zero on entry and on exit.
pub const FORMAT_U32_TILE_COUNTERS: FormatId = FormatId("u32.tile_counters.z_x_tiles");

// ---- params ----------------------------------------------------------------------------------

const fn param(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}

const PARAMS: &[ParamV1] = &[
    param("M0", "part 0 output columns (non-empty)"),
    param("M1", "part 1 output columns (non-empty)"),
    param("M2", "part 2 output columns, 0 = empty"),
    param("M3", "part 3 output columns, 0 = empty"),
    param("S0", "part 0 split-K"),
    param("S1", "part 1 split-K"),
    param("S2", "part 2 split-K (1 for an empty part)"),
    param("S3", "part 3 split-K (1 for an empty part)"),
    param("K", "reduction length, a multiple of 128 x every S_p"),
    param("rows", "activation rows (tokens)"),
    param(
        "grid_y",
        "launch grid y = max(S0, S1, S2, S3) (not a kernarg word)",
    ),
];

const P_M0: Expr = Expr::Param("M0");
const P_M1: Expr = Expr::Param("M1");
const P_M2: Expr = Expr::Param("M2");
const P_M3: Expr = Expr::Param("M3");
const P_S0: Expr = Expr::Param("S0");
const P_S1: Expr = Expr::Param("S1");
const P_S2: Expr = Expr::Param("S2");
const P_S3: Expr = Expr::Param("S3");
const P_K: Expr = Expr::Param("K");
const P_ROWS: Expr = Expr::Param("rows");
const P_GRID_Y: Expr = Expr::Param("grid_y");

// ---- operands --------------------------------------------------------------------------------

const X_BYTES: Expr = Expr::Mul(&[P_ROWS, P_K, Expr::Const(4)]);

const TILES: Expr = Expr::Add(&[
    Expr::DivCeil(&P_M0, 16),
    Expr::DivCeil(&P_M1, 16),
    Expr::DivCeil(&P_M2, 16),
    Expr::DivCeil(&P_M3, 16),
]);

/// Upper bound of the partial scratch the parts take (parts with `S_p == 1` take none).
const PARTIAL_BYTES: Expr = Expr::Add(&[
    Expr::Mul(&[P_S0, P_ROWS, P_M0, Expr::Const(4)]),
    Expr::Mul(&[P_S1, P_ROWS, P_M1, Expr::Const(4)]),
    Expr::Mul(&[P_S2, P_ROWS, P_M2, Expr::Const(4)]),
    Expr::Mul(&[P_S3, P_ROWS, P_M3, Expr::Const(4)]),
]);

/// `counters[blockIdx.z * gridDim.x + blockIdx.x]`: `ceil(rows/64) * sum_p ceil(M_p/16)` words.
const COUNTER_BYTES: Expr = Expr::Mul(&[Expr::DivCeil(&P_ROWS, 64), TILES, Expr::Const(4)]);

macro_rules! op_w {
    ($name:literal, $m:expr) => {
        OperandV1 {
            name: $name,
            format: FORMAT_BF16_ROWS_K,
            access: Access::ReadOnly,
            align: 2,
            extent: Some(Expr::Mul(&[$m, P_K, Expr::Const(2)])),
            nullable: false,
            note: "alignment = element size; an empty part (M = 0) carries the last real part's pointer",
        }
    };
}

macro_rules! op_y {
    ($name:literal, $m:expr) => {
        OperandV1 {
            name: $name,
            format: FORMAT_F32_ROWS_M,
            access: Access::WriteOnly,
            align: 4,
            extent: Some(Expr::Mul(&[P_ROWS, $m, Expr::Const(4)])),
            nullable: false,
            note: "alignment = element size; an empty part carries the last real part's pointer",
        }
    };
}

const OPERANDS: &[OperandV1] = &[
    op_w!("W0", P_M0),
    op_w!("W1", P_M1),
    op_w!("W2", P_M2),
    op_w!("W3", P_M3),
    op_y!("Y0", P_M0),
    op_y!("Y1", P_M1),
    op_y!("Y2", P_M2),
    op_y!("Y3", P_M3),
    OperandV1 {
        name: "X",
        format: FORMAT_F32_ROWS_K,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(X_BYTES),
        nullable: false,
        note: "alignment = element size",
    },
    OperandV1 {
        name: "partial",
        format: FORMAT_F32_SPLITK_PARTIAL,
        access: Access::ReadWrite,
        align: 4,
        extent: Some(PARTIAL_BYTES),
        nullable: false,
        note: "bound is the sum over parts of S_p * rows * M_p floats (parts with S_p == 1 take \
               none); the launcher passes the fixed Gpu-owned multi scratch",
    },
    OperandV1 {
        name: "counters",
        format: FORMAT_U32_TILE_COUNTERS,
        access: Access::ReadWrite,
        align: 4,
        extent: Some(COUNTER_BYTES),
        nullable: false,
        note: "must be zero on entry; the last block of a split-K tile re-zeroes its slot",
    },
];

// ---- fields ----------------------------------------------------------------------------------

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

const FIELDS: &[FieldV1] = &[
    ptr("W0", 0),
    ptr("W1", 8),
    ptr("W2", 16),
    ptr("W3", 24),
    ptr("Y0", 32),
    ptr("Y1", 40),
    ptr("Y2", 48),
    ptr("Y3", 56),
    ptr("X", 64),
    ptr("partial", 72),
    ptr("counters", 80),
    scalar("M0", 88, P_M0),
    scalar("M1", 92, P_M1),
    scalar("M2", 96, P_M2),
    scalar("M3", 100, P_M3),
    scalar("S0", 104, P_S0),
    scalar("S1", 108, P_S1),
    scalar("S2", 112, P_S2),
    scalar("S3", 116, P_S3),
    scalar("K", 120, P_K),
    scalar("rows", 124, P_ROWS),
];

const fn hid(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
    HiddenV1 {
        offset,
        width,
        kind,
    }
}

/// The runtime's hidden block as the object metadata lists it for a kernel that reads `gridDim`
/// (base 128, no dynamic-LDS word).
const HIDDEN: &[HiddenV1] = &[
    hid(128, 4, HiddenKind::BlockCount(crate::Axis::X)),
    hid(132, 4, HiddenKind::BlockCount(crate::Axis::Y)),
    hid(136, 4, HiddenKind::BlockCount(crate::Axis::Z)),
    hid(140, 2, HiddenKind::GroupSize(crate::Axis::X)),
    hid(142, 2, HiddenKind::GroupSize(crate::Axis::Y)),
    hid(144, 2, HiddenKind::GroupSize(crate::Axis::Z)),
    hid(146, 2, HiddenKind::Remainder(crate::Axis::X)),
    hid(148, 2, HiddenKind::Remainder(crate::Axis::Y)),
    hid(150, 2, HiddenKind::Remainder(crate::Axis::Z)),
    hid(168, 8, HiddenKind::GlobalOffset(crate::Axis::X)),
    hid(176, 8, HiddenKind::GlobalOffset(crate::Axis::Y)),
    hid(184, 8, HiddenKind::GlobalOffset(crate::Axis::Z)),
    hid(192, 2, HiddenKind::GridDims),
];

// ---- domain ----------------------------------------------------------------------------------

const I32_MAX: u64 = i32::MAX as u64;

const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}

const DOMAIN: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_K), "lane_wmma_multi_plan: k > 0"),
    rule(
        Rule::MultipleOf(P_K, 128),
        "lane_wmma_multi_plan: k % 128 == 0 (kernel: K % (128 * split_k) == 0)",
    ),
    rule(Rule::Positive(P_ROWS), "lane_wmma_multi_plan: rows > 0"),
    rule(
        Rule::Positive(P_M0),
        "parts 0 and 1 are never empty (2..=4 non-empty parts)",
    ),
    rule(
        Rule::Positive(P_M1),
        "parts 0 and 1 are never empty (2..=4 non-empty parts)",
    ),
    rule(Rule::Positive(P_S0), "split_k >= 1"),
    rule(Rule::Positive(P_S1), "split_k >= 1"),
    rule(Rule::Positive(P_S2), "split_k >= 1 (1 for an empty part)"),
    rule(Rule::Positive(P_S3), "split_k >= 1 (1 for an empty part)"),
    rule(
        Rule::Positive(P_GRID_Y),
        "grid y = max_p S_p >= 1 (launcher invariant, not an Expr)",
    ),
    rule(Rule::AtMost(P_M0, I32_MAX), "m passed as i32"),
    rule(Rule::AtMost(P_M1, I32_MAX), "m passed as i32"),
    rule(Rule::AtMost(P_M2, I32_MAX), "m passed as i32"),
    rule(Rule::AtMost(P_M3, I32_MAX), "m passed as i32"),
    rule(Rule::AtMost(P_S0, I32_MAX), "split_k passed as i32"),
    rule(Rule::AtMost(P_S1, I32_MAX), "split_k passed as i32"),
    rule(Rule::AtMost(P_S2, I32_MAX), "split_k passed as i32"),
    rule(Rule::AtMost(P_S3, I32_MAX), "split_k passed as i32"),
    rule(Rule::AtMost(P_K, I32_MAX), "k passed as i32"),
    rule(Rule::AtMost(P_ROWS, I32_MAX), "rows passed as i32"),
];

// ---- exports ---------------------------------------------------------------------------------

const OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "qwen4_lane_wmma",
    source: "kernels/src/qwen4_lane_wmma.gfx1201.hip",
};

/// `[sum_p ceil(M_p/16), max_p S_p, ceil(rows/64)]`.
const GRID: [Expr; 3] = [TILES, P_GRID_Y, Expr::DivCeil(&P_ROWS, 64)];

const CONSTS: &[(&str, u64)] = &[];

macro_rules! export {
    ($sym:literal, $lds:expr) => {
        ExportV1 {
            symbol: $sym,
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [128, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: $lds,
            grid: GRID,
            consts: CONSTS,
        }
    };
}

const EXPORTS: &[ExportV1] = &[
    export!("lwmma_gemm_bf16_t2_m1_n1_multi", 4100),
    export!("lwmma_gemm_bf16_t2_m2_n1_multi", 8196),
    export!("lwmma_gemm_bf16_t2_m3_n1_multi", 12292),
    export!("lwmma_gemm_bf16_t2_m4_n1_multi", 16388),
];

/// BF16-weight two-term lane-WMMA multi-part GEMM, `MT` = 1..=4 row tiles: 11 pointers + 10 i32
/// through kernelParams (128 explicit bytes), hidden block at 128, segment 384.
pub static LWMMA_GEMM_BF16_T2_MULTI: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_lane_wmma.hipcc.gfx12.bf16_t2_n1_multi",
    version: 1,
    wave: 32,
    numeric: NUMERIC_BF16_T2_MULTI_F32,
    params: PARAMS,
    operands: OPERANDS,
    fields: FIELDS,
    hidden: HIDDEN,
    explicit_bytes: 128,
    segment_bytes: 384,
    segment_align: 8,
    domain: DOMAIN,
    exports: EXPORTS,
};

/// Every contract of this family.
pub static ABIS: &[&KernelAbiV1] = &[&LWMMA_GEMM_BF16_T2_MULTI];
