// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-WMMA prefill tile GEMM entries of qwen4 on exact gfx1201
//! (`crates/rdna-compute/src/lane_wmma_tile.rs` `lane_wmma_tile_gemm_cfg`,
//! `kernels/src/qwen4_lwmma_tile.gfx1201.hip` `LWMMA_TILE_ENTRY_NAMED`):
//! `lwmma_tile_bf16_t{1,2}_b<BM>x<BN>w<WM>x<WN>[g4]`, `Y[rows, M] = epi(X W^T)` for a BF16 weight
//! `W [M][K]` and an F32 `X [rows][K]` with `terms` (1 | 2) BF16 activation split terms, bitwise
//! the plain lane-WMMA entry's value at the same split-K, in ONE launch (no partial scratch, no
//! counters). 10 entries (exactly the tiles `lane_wmma_tile_pick` can select): 3 one-term
//! (`g4` 32x32, 32x64w2x2, 64x64w4x1) and 7 two-term (`g4` 16x32, 32x64w2x2, 64x64w4x1,
//! 64x64w4x2, 32x128w2x2, 64x128w4x2, 64x128w2x4).
//!
//! One 48-byte explicit kernarg image for every entry (hipcc-built, `HipccSource`), no hidden
//! block (the kernel never reads `gridDim` / the implicit arguments: `.kernarg_segment_size` 48,
//! align 8, no dynamic-LDS word):
//!
//! | off | field | kind |
//! |---|---|---|
//! | 0 | `W` | pointer, BF16 `[M][K]` |
//! | 8 | `X` | pointer, F32 `[rows][K]` |
//! | 16 | `Y` | pointer, F32 `[rows][M]`, overwritten |
//! | 24 | `M` | i32 |
//! | 28 | `K` | i32 |
//! | 32 | `rows` | i32 |
//! | 36 | `split_k` | i32 |
//! | 40 | `scale_bits` | the `hc_act` epilogue scale, an `f32` passed as its bit pattern (u32) |
//! | 44 | `hc_act` | i32, 0 = no epilogue, 1 = HC epilogue |
//!
//! Launch: grid `[ceil(M / BN), ceil(rows / BM), 1]`, block `32 * WM * WN * QG`, dynamic LDS 0;
//! the static LDS is `QG * 2 * (terms * BM + BN) * 40 * 2` bytes plus, for the quarter-group
//! (`g4`) tiles, the `3 * WM * WN * MT * NTN * 8 * 32` float fold exchange. Per-entry numbers
//! are the formula above (checked against the hipcc anchor `kernels/qwen4_lwmma_tile_hipcc_gfx1201.co`).
//!
//! Admission that is not a function of the launch params stays in the launcher
//! (`lane_wmma_tile_gemm_cfg`: `terms` in 1..=2, `k % (128 * split_k) == 0`, the configuration
//! instantiated for `terms`, operand sizes).

use crate::families::qwen4_lane_wmma::{FORMAT_BF16_ROWS_K, FORMAT_F32_ROWS_K, FORMAT_F32_ROWS_M};
use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, KernelAbiV1, NumericContractId,
    ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- numeric contract ------------------------------------------------------------------------

/// Per output word: exactly the plain bf16 lane-WMMA entry's value at the same split-K (four
/// quarter chains folded `(c0 + c1) + (c2 + c3)`, split-K chunks accumulated ascending from
/// `0.0f`), optional `hc_act` epilogue, F32 store.
pub const NUMERIC_BF16_TILE_F32: NumericContractId =
    NumericContractId("qwen4_lwmma_tile.bf16_tile.f32.v1");

// ---- params ----------------------------------------------------------------------------------

const fn param(name: &'static str, ty: ScalarTy, doc: &'static str) -> ParamV1 {
    ParamV1 { name, ty, doc }
}

const PARAMS: &[ParamV1] = &[
    param("M", ScalarTy::I32, "output columns (weight rows)"),
    param(
        "K",
        ScalarTy::I32,
        "reduction length, a multiple of 128 x split_k",
    ),
    param("rows", ScalarTy::I32, "activation rows (tokens)"),
    param("split_k", ScalarTy::I32, "split-K chunks"),
    param(
        "scale_bits",
        ScalarTy::U32,
        "hc_act epilogue scale, f32 bit pattern (0 when hc_act = 0)",
    ),
    param("hc_act", ScalarTy::I32, "0 = plain store, 1 = HC epilogue"),
];

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_ROWS: Expr = Expr::Param("rows");
const P_SPLIT_K: Expr = Expr::Param("split_k");
const P_SCALE_BITS: Expr = Expr::Param("scale_bits");
const P_HC_ACT: Expr = Expr::Param("hc_act");

// ---- operands --------------------------------------------------------------------------------

const OPERANDS: &[OperandV1] = &[
    OperandV1 {
        name: "W",
        format: FORMAT_BF16_ROWS_K,
        access: Access::ReadOnly,
        align: 2,
        extent: Some(Expr::Mul(&[P_M, P_K, Expr::Const(2)])),
        nullable: false,
        note: "alignment = element size; the kernel clamps tile rows to M - 1",
    },
    OperandV1 {
        name: "X",
        format: FORMAT_F32_ROWS_K,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_ROWS, P_K, Expr::Const(4)])),
        nullable: false,
        note: "alignment = element size; the kernel clamps tile rows to rows - 1",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_ROWS_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_ROWS, P_M, Expr::Const(4)])),
        nullable: false,
        note: "alignment = element size",
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

const fn scalar(name: &'static str, offset: u32, ty: ScalarTy, value: Expr) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar { ty, value },
    }
}

const FIELDS: &[FieldV1] = &[
    ptr("W", 0),
    ptr("X", 8),
    ptr("Y", 16),
    scalar("M", 24, ScalarTy::I32, P_M),
    scalar("K", 28, ScalarTy::I32, P_K),
    scalar("rows", 32, ScalarTy::I32, P_ROWS),
    scalar("split_k", 36, ScalarTy::I32, P_SPLIT_K),
    scalar("scale_bits", 40, ScalarTy::U32, P_SCALE_BITS),
    scalar("hc_act", 44, ScalarTy::I32, P_HC_ACT),
];

// ---- domain ----------------------------------------------------------------------------------

const I32_MAX: u64 = i32::MAX as u64;

const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}

const DOMAIN: &[DomainRuleV1] = &[
    rule(
        Rule::Positive(P_M),
        "lane_wmma_tile_gemm_cfg: m and rows must be non-zero",
    ),
    rule(
        Rule::Positive(P_ROWS),
        "lane_wmma_tile_gemm_cfg: m and rows must be non-zero",
    ),
    rule(Rule::Positive(P_K), "lane_wmma_tile_gemm_cfg: k > 0"),
    rule(
        Rule::MultipleOf(P_K, 128),
        "lane_wmma_tile_gemm_cfg: k % (128 * split_k) == 0 (32-wide K steps, four quarter chains)",
    ),
    rule(
        Rule::Positive(P_SPLIT_K),
        "lane_wmma_tile_gemm_cfg: split_k >= 1",
    ),
    rule(
        Rule::OneOf(P_HC_ACT, &[0, 1]),
        "kernel: hc_act is a flag (0 plain store, 1 HC epilogue)",
    ),
    rule(Rule::AtMost(P_M, I32_MAX), "m passed as i32"),
    rule(Rule::AtMost(P_K, I32_MAX), "k passed as i32"),
    rule(Rule::AtMost(P_ROWS, I32_MAX), "rows passed as i32"),
    rule(Rule::AtMost(P_SPLIT_K, I32_MAX), "split_k passed as i32"),
];

// ---- exports ---------------------------------------------------------------------------------

const OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "qwen4_lwmma_tile",
    source: "kernels/src/qwen4_lwmma_tile.gfx1201.hip",
};

const CONSTS: &[(&str, u64)] = &[];

/// `export!(symbol, threads, BM, BN, static LDS bytes)`; grid `[ceil(M / BN), ceil(rows / BM), 1]`.
macro_rules! export {
    ($sym:literal, $threads:literal, $bm:literal, $bn:literal, $lds:literal) => {
        ExportV1 {
            symbol: $sym,
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [$threads, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: $lds,
            grid: [
                Expr::DivCeil(&P_M, $bn),
                Expr::DivCeil(&P_ROWS, $bm),
                Expr::Const(1),
            ],
            consts: CONSTS,
        }
    };
}

const EXPORTS: &[ExportV1] = &[
    export!("lwmma_tile_bf16_t1_b32x32w2x1g4", 256, 32, 32, 53248),
    export!("lwmma_tile_bf16_t1_b32x64w2x2", 128, 32, 64, 15360),
    export!("lwmma_tile_bf16_t1_b64x64w4x1", 128, 64, 64, 20480),
    export!("lwmma_tile_bf16_t2_b16x32w1x1g4", 128, 16, 32, 47104),
    export!("lwmma_tile_bf16_t2_b32x64w2x2", 128, 32, 64, 20480),
    export!("lwmma_tile_bf16_t2_b64x64w4x1", 128, 64, 64, 30720),
    export!("lwmma_tile_bf16_t2_b64x64w4x2", 256, 64, 64, 30720),
    export!("lwmma_tile_bf16_t2_b32x128w2x2", 128, 32, 128, 30720),
    export!("lwmma_tile_bf16_t2_b64x128w4x2", 256, 64, 128, 40960),
    export!("lwmma_tile_bf16_t2_b64x128w2x4", 256, 64, 128, 40960),
];

/// BF16-weight lane-WMMA prefill tile GEMM, 1..=2 activation terms: 3 pointers + 6 scalars
/// through kernelParams (48 explicit bytes), no hidden block, segment 48.
pub static LWMMA_TILE_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_lwmma_tile.hipcc.gfx12.bf16_tile",
    version: 1,
    wave: 32,
    numeric: NUMERIC_BF16_TILE_F32,
    params: PARAMS,
    operands: OPERANDS,
    fields: FIELDS,
    hidden: &[],
    explicit_bytes: 48,
    segment_bytes: 48,
    segment_align: 8,
    domain: DOMAIN,
    exports: EXPORTS,
};

/// Every contract of this family.
pub static ABIS: &[&KernelAbiV1] = &[&LWMMA_TILE_BF16];
