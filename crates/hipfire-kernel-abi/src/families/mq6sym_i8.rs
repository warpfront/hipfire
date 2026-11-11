// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the symmetric-MQ6 x A8 trunk route for qwen4 Flash-Next on exact gfx1201
//! (`crates/rdna-compute/src/gemm_mq6sym_i8.rs`): the T0 signed dot4 GEMV
//! (`kernels/src/gemv_mq6sym_i8.gfx1201.hip`, N = 1..8) and the T1 signed iu8 WMMA GEMM
//! (`kernels/src/gemm_mq6sym_i8.gfx1201.hip`: narrow verify tiles and the 128x128 prefill tile).
//!
//! Every entry shares the ABI `(const u8* W, const block_i8_128* Xq, void* Y, i32 M, i32 K,
//! i32 N)`: 36 explicit bytes through ordinary kernelParams, the weights are qt47
//! (`FORMAT_MQ6G256V2_WEIGHTS`, 200 B per 256 weights, symmetric: `z_h == -32 * s_h`), the
//! activations `block_i8_128` `[K/128][N]`, `Y` `[N][M]` F32 or BF16 bits, overwritten. All
//! entries are hipcc-built (`HipccSource`); their kernarg metadata is checked against compiled
//! objects by `crates/rdna-compute/tests/kernel_abi_drift.rs`. The arithmetic of every entry is
//! one definition (exact i32 per 128-K block, `acc = fmaf(RN(s_h * d), (float)C, acc)`), so the
//! F32 outputs share [`NUMERIC_SYM_SET_F32`] and the BF16 outputs [`NUMERIC_SYM_SET_BF16_RNE`].

use super::mq6_i8::{
    DOMAIN_HIPCC, FIELDS_HIPCC, FORMAT_BLOCK_I8_128_KB_N, FORMAT_F32_N_M, FORMAT_MQ6G256V2_WEIGHTS,
    HIDDEN, OPERANDS_HIPCC_BF16, OPERANDS_HIPCC_F32, PARAMS_MKN, P_M, P_N,
};
use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, KernelAbiV1, NumericContractId,
    ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

/// F32 store of the symmetric A8 route (T0 dot4 and T1 WMMA entries, bytewise equal).
pub const NUMERIC_SYM_SET_F32: NumericContractId = NumericContractId("mq6sym_i8.a8.set_f32.v1");
/// BF16 store of the symmetric A8 route: `isfinite(v) ? (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16`,
/// `u = bits(0.0f + v)`.
pub const NUMERIC_SYM_SET_BF16_RNE: NumericContractId =
    NumericContractId("mq6sym_i8.a8.set_bf16_rne.v1");

const GEMV_OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "gemv_mq6sym_i8_gfx1201",
    source: "kernels/src/gemv_mq6sym_i8.gfx1201.hip",
};
const GEMM_OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "gemm_mq6sym_i8_gfx1201",
    source: "kernels/src/gemm_mq6sym_i8.gfx1201.hip",
};

/// T0: `[ceil(M/32), 1, 1]`, block 128 (four waves, eight rows per wave).
const GRID_R32: [Expr; 3] = [Expr::DivCeil(&P_M, 32), Expr::Const(1), Expr::Const(1)];
/// Narrow tiles of 64 rows: `[ceil(M/64), 1, 1]`.
const GRID_R64: [Expr; 3] = [Expr::DivCeil(&P_M, 64), Expr::Const(1), Expr::Const(1)];
/// Token-split twin `m64n16y`: `[ceil(M/64), ceil(N/16), 1]`.
const GRID_R64_Y: [Expr; 3] = [
    Expr::DivCeil(&P_M, 64),
    Expr::DivCeil(&P_N, 16),
    Expr::Const(1),
];
/// Prefill 128x128: `[ceil(M/128), ceil(N/128), 1]`, block 256, dynamic LDS 36864.
const GRID_128X128: [Expr; 3] = [
    Expr::DivCeil(&P_M, 128),
    Expr::DivCeil(&P_N, 128),
    Expr::Const(1),
];

const CONSTS_T0: &[(&str, u64)] = &[("tile_rows", 32)];
const CONSTS_R32: &[(&str, u64)] = &[("tile_rows", 32)];
const CONSTS_R64: &[(&str, u64)] = &[("tile_rows", 64)];
const CONSTS_128X128: &[(&str, u64)] = &[("tile_rows", 128), ("tile_tokens", 128)];

macro_rules! export {
    ($sym:literal, $obj:expr, $block:expr, $lds:expr, $fixed:expr, $grid:expr, $consts:expr) => {
        ExportV1 {
            symbol: $sym,
            arch: Arch::Gfx1201,
            object: $obj,
            block: [$block, 1, 1],
            dynamic_lds: $lds,
            group_segment_fixed: $fixed,
            grid: $grid,
            consts: $consts,
        }
    };
}

/// F32 T0 (N = 1..8) and narrow-tile entries: no hidden kernargs (kernarg segment = the 36
/// explicit bytes), static LDS per tile.
const EXPORTS_SYM_NARROW_F32: &[ExportV1] = &[
    export!(
        "gemv_mq6sym_i8_rows_r1_gfx1201",
        GEMV_OBJECT,
        128,
        1168,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r2_gfx1201",
        GEMV_OBJECT,
        128,
        1760,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r3_gfx1201",
        GEMV_OBJECT,
        128,
        2352,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r4_gfx1201",
        GEMV_OBJECT,
        128,
        2944,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r5_gfx1201",
        GEMV_OBJECT,
        128,
        3536,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r6_gfx1201",
        GEMV_OBJECT,
        128,
        4128,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r7_gfx1201",
        GEMV_OBJECT,
        128,
        5296,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r8_gfx1201",
        GEMV_OBJECT,
        128,
        5312,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemm_mq6sym_i8_set_m32n16_gfx1201",
        GEMM_OBJECT,
        64,
        0,
        5120,
        GRID_R32,
        CONSTS_R32
    ),
    export!(
        "gemm_mq6sym_i8_set_m32n32_gfx1201",
        GEMM_OBJECT,
        64,
        0,
        9216,
        GRID_R32,
        CONSTS_R32
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n16_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n32_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        10240,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n48_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        14336,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n16y_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_R64_Y,
        CONSTS_R64
    ),
];

/// F32 128x128 prefill entry: the runtime hidden block (kernarg segment 296), dynamic LDS only.
const EXPORTS_SYM_PREFILL_F32: &[ExportV1] = &[export!(
    "gemm_mq6sym_i8_set_gfx1201",
    GEMM_OBJECT,
    256,
    36864,
    0,
    GRID_128X128,
    CONSTS_128X128
)];

/// Symmetric A8 route, 128x128 prefill entry, F32 `Y`: `W Xq Y M K N` through kernelParams (36 explicit bytes).
pub static MQ6SYM_I8_SET_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6sym_i8.hipcc.gfx12.set_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SYM_SET_F32,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_F32,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_SYM_PREFILL_F32,
};

/// Symmetric A8 route, T0 GEMV and narrow tiles, F32 `Y`: `W Xq Y M K N` through kernelParams (36 explicit bytes).
pub static MQ6SYM_I8_NARROW_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6sym_i8.hipcc.gfx12.narrow_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SYM_SET_F32,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_F32,
    fields: FIELDS_HIPCC,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_SYM_NARROW_F32,
};

/// BF16 T0 (N = 1..8) and narrow-tile entries: no hidden kernargs (kernarg segment = the 36
/// explicit bytes), static LDS per tile.
const EXPORTS_SYM_NARROW_BF16: &[ExportV1] = &[
    export!(
        "gemv_mq6sym_i8_rows_r1_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        1168,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r2_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        1760,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r3_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        2352,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r4_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        2944,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r5_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        3536,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r6_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        4128,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r7_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        5296,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemv_mq6sym_i8_rows_r8_bf16_gfx1201",
        GEMV_OBJECT,
        128,
        5312,
        0,
        GRID_R32,
        CONSTS_T0
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        GEMM_OBJECT,
        64,
        0,
        5120,
        GRID_R32,
        CONSTS_R32
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        GEMM_OBJECT,
        64,
        0,
        9216,
        GRID_R32,
        CONSTS_R32
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        10240,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        14336,
        GRID_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_R64_Y,
        CONSTS_R64
    ),
];

/// BF16 128x128 prefill entry: the runtime hidden block (kernarg segment 296), dynamic LDS only.
const EXPORTS_SYM_PREFILL_BF16: &[ExportV1] = &[export!(
    "gemm_mq6sym_i8_set_bf16_gfx1201",
    GEMM_OBJECT,
    256,
    36864,
    0,
    GRID_128X128,
    CONSTS_128X128
)];

/// Symmetric A8 route, 128x128 prefill entry, BF16 `Y`: `W Xq Y M K N` through kernelParams (36 explicit bytes).
pub static MQ6SYM_I8_SET_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "mq6sym_i8.hipcc.gfx12.set_bf16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SYM_SET_BF16_RNE,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_BF16,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_SYM_PREFILL_BF16,
};

/// Symmetric A8 route, T0 GEMV and narrow tiles, BF16 `Y`: `W Xq Y M K N` through kernelParams (36 explicit bytes).
pub static MQ6SYM_I8_NARROW_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "mq6sym_i8.hipcc.gfx12.narrow_bf16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SYM_SET_BF16_RNE,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_BF16,
    fields: FIELDS_HIPCC,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_SYM_NARROW_BF16,
};

// ---- multi-part (x4) narrow tiles, F32 `Y` ------------------------------------------------------

const P_M0: Expr = Expr::Param("M0");
const P_M1: Expr = Expr::Param("M1");
const P_M2: Expr = Expr::Param("M2");
const P_M3: Expr = Expr::Param("M3");
const P_X4_K: Expr = Expr::Param("K");
const I32_MAX_X4: u64 = i32::MAX as u64;

const PARAMS_X4: &[ParamV1] = &[
    ParamV1 {
        name: "M0",
        ty: ScalarTy::I32,
        doc: "weight rows of part 0 (> 0)",
    },
    ParamV1 {
        name: "M1",
        ty: ScalarTy::I32,
        doc: "weight rows of part 1 (0 = part absent)",
    },
    ParamV1 {
        name: "M2",
        ty: ScalarTy::I32,
        doc: "weight rows of part 2 (0 = part absent)",
    },
    ParamV1 {
        name: "M3",
        ty: ScalarTy::I32,
        doc: "weight rows of part 3 (0 = part absent)",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction length (shared)",
    },
    ParamV1 {
        name: "N",
        ty: ScalarTy::I32,
        doc: "tokens (Y rows, shared)",
    },
];

macro_rules! x4_weight {
    ($name:literal, $m:expr, $nullable:expr) => {
        OperandV1 {
            name: $name,
            format: FORMAT_MQ6G256V2_WEIGHTS,
            access: Access::ReadOnly,
            align: 4,
            extent: Some(Expr::Mul(&[$m, Expr::DivExact(&P_X4_K, 256), Expr::Const(200)])),
            nullable: $nullable,
            note: "alignment 4: the launcher checks no base alignment of W; a part with M_i = 0 passes null",
        }
    };
}
macro_rules! x4_out {
    ($name:literal, $m:expr, $nullable:expr) => {
        OperandV1 {
            name: $name,
            format: FORMAT_F32_N_M,
            access: Access::WriteOnly,
            align: 4,
            extent: Some(Expr::Mul(&[P_N, $m, Expr::Const(4)])),
            nullable: $nullable,
            note: "[N][M_i] F32, overwritten; a part with M_i = 0 passes null",
        }
    };
}

const OPERANDS_X4_F32: &[OperandV1] = &[
    x4_weight!("W0", P_M0, false),
    x4_weight!("W1", P_M1, true),
    x4_weight!("W2", P_M2, true),
    x4_weight!("W3", P_M3, true),
    x4_out!("Y0", P_M0, false),
    x4_out!("Y1", P_M1, true),
    x4_out!("Y2", P_M2, true),
    x4_out!("Y3", P_M3, true),
    OperandV1 {
        name: "Xq",
        format: FORMAT_BLOCK_I8_128_KB_N,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            P_N,
            Expr::DivExact(&P_X4_K, 128),
            Expr::Const(136),
        ])),
        nullable: false,
        note: "alignment 4: records start with a float; the launcher checks only non-null",
    },
];

macro_rules! x4_ptr {
    ($name:literal, $off:literal) => {
        FieldV1 {
            name: $name,
            offset: $off,
            width: 8,
            kind: FieldKind::Pointer { operand: $name },
        }
    };
}
macro_rules! x4_i32 {
    ($name:literal, $off:literal, $p:expr) => {
        FieldV1 {
            name: $name,
            offset: $off,
            width: 4,
            kind: FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: $p,
            },
        }
    };
}

/// `W0..W3 @0, Y0..Y3 @32, Xq @64, M0..M3 @72, K @88, N @92` (kernel parameter order of
/// `SYM_NARROW_X4_ENTRY`).
const FIELDS_X4: &[FieldV1] = &[
    x4_ptr!("W0", 0),
    x4_ptr!("W1", 8),
    x4_ptr!("W2", 16),
    x4_ptr!("W3", 24),
    x4_ptr!("Y0", 32),
    x4_ptr!("Y1", 40),
    x4_ptr!("Y2", 48),
    x4_ptr!("Y3", 56),
    x4_ptr!("Xq", 64),
    x4_i32!("M0", 72, P_M0),
    x4_i32!("M1", 76, P_M1),
    x4_i32!("M2", 80, P_M2),
    x4_i32!("M3", 84, P_M3),
    x4_i32!("K", 88, P_X4_K),
    x4_i32!("N", 92, P_N),
];

const DOMAIN_X4: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_X4_K),
        why: "gemm_mq6sym_i8_prequant_x4: k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_X4_K, 256),
        why: "gemm_mq6sym_i8_prequant_x4: MQ6 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M0),
        why: "gemm_mq6sym_i8_prequant_x4: part 0 present",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_mq6sym_i8_prequant_x4: n > 0",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M0, I32_MAX_X4),
        why: "m passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M1, I32_MAX_X4),
        why: "m passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M2, I32_MAX_X4),
        why: "m passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M3, I32_MAX_X4),
        why: "m passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_X4_K, I32_MAX_X4),
        why: "k passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX_X4),
        why: "n passed as i32",
    },
];

/// x4 grid, `[sum_i ceil(M_i/64), 1, 1]`.
const GRID_X4_R64: [Expr; 3] = [
    Expr::Add(&[
        Expr::DivCeil(&P_M0, 64),
        Expr::DivCeil(&P_M1, 64),
        Expr::DivCeil(&P_M2, 64),
        Expr::DivCeil(&P_M3, 64),
    ]),
    Expr::Const(1),
    Expr::Const(1),
];
/// x4 grid of the 32-row tile, `[sum_i ceil(M_i/32), 1, 1]`.
const GRID_X4_R32: [Expr; 3] = [
    Expr::Add(&[
        Expr::DivCeil(&P_M0, 32),
        Expr::DivCeil(&P_M1, 32),
        Expr::DivCeil(&P_M2, 32),
        Expr::DivCeil(&P_M3, 32),
    ]),
    Expr::Const(1),
    Expr::Const(1),
];
/// x4 token-split grid, `[sum_i ceil(M_i/64), ceil(N/16), 1]`.
const GRID_X4_R64_Y: [Expr; 3] = [
    Expr::Add(&[
        Expr::DivCeil(&P_M0, 64),
        Expr::DivCeil(&P_M1, 64),
        Expr::DivCeil(&P_M2, 64),
        Expr::DivCeil(&P_M3, 64),
    ]),
    Expr::DivCeil(&P_N, 16),
    Expr::Const(1),
];

/// x4 entries: no hidden kernargs (kernarg segment = the 96 explicit bytes), static LDS = the
/// single-matrix tile's.
const EXPORTS_SYM_X4_F32: &[ExportV1] = &[
    export!(
        "gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_X4_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
        GEMM_OBJECT,
        64,
        0,
        9216,
        GRID_X4_R32,
        CONSTS_R32
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        14336,
        GRID_X4_R64,
        CONSTS_R64
    ),
    export!(
        "gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
        GEMM_OBJECT,
        128,
        0,
        6144,
        GRID_X4_R64_Y,
        CONSTS_R64
    ),
];

/// Symmetric A8 route, multi-part narrow tiles, F32 `Y`: `W0..W3 Y0..Y3 Xq M0..M3 K N` through
/// kernelParams (96 explicit bytes). Same numeric contract as the single-matrix entries: every
/// output word is bytewise the one the single-matrix entry of the same tile writes.
pub static MQ6SYM_I8_NARROW_X4_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6sym_i8.hipcc.gfx12.narrow_x4_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SYM_SET_F32,
    params: PARAMS_X4,
    operands: OPERANDS_X4_F32,
    fields: FIELDS_X4,
    hidden: &[],
    explicit_bytes: 96,
    segment_bytes: 96,
    segment_align: 8,
    domain: DOMAIN_X4,
    exports: EXPORTS_SYM_X4_F32,
};

/// Every contract of this family.
pub static ABIS: &[&KernelAbiV1] = &[
    &MQ6SYM_I8_SET_F32,
    &MQ6SYM_I8_SET_BF16,
    &MQ6SYM_I8_NARROW_F32,
    &MQ6SYM_I8_NARROW_BF16,
    &MQ6SYM_I8_NARROW_X4_F32,
];
