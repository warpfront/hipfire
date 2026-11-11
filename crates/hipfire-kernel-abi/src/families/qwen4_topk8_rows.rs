// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-batched draft-head top-8 re-score entries of qwen4 on exact gfx1201
//! (`HIPFIRE_QWEN4_DRAFT_TOPK8_ROWS`, default on; module `topk8_rows_mq6g256v2`, source
//! `kernels/src/topk8_rows.gfx1201.hip` compiled after `topk8_select.hip` and
//! `qwen4_gemv_mq6g256v2.hip`). Two hipcc-built entries, each the body of its one-row kernel on
//! `blockIdx` rows, so a row's bytes do not depend on which launch computed it:
//!
//! | contract | entry | launcher |
//! |---|---|---|
//! | [`TOPK8_PARTIAL_F32_ROWS`] | `topk8_partial_f32_rows` (body of `topk8_partial_f32`) | `Gpu::topk8_partial_rows` (`rdna-compute/src/gemv.rs`) |
//! | [`TOPK8_RESCORE_MQ6G256V2_K2560_ROWS`] | `topk8_rescore_mq6g256v2_k2560_rows` (body of `topk8_rescore_mq6g256v2_k2560`, m = 1024) | `Gpu::topk8_rescore_k2560_rows` |
//!
//! `topk8_partial_f32_rows` takes a 40-byte explicit kernarg image: `v` @0, `n` @8, `chunk` @12,
//! `partial` @16, `front` @24, `special` @28, `tail` @32, `v_stride` @36 (floats). Launch grid
//! `[128, rows, 1]`, block 256, static LDS 64. Row `y` reads `v + y * v_stride` and writes the
//! 8192-byte slot `partial + y * 8192` (`pv` 1024 f32, then `pi` 1024 i32; only the first 128 * 8
//! words of each half are written).
//!
//! `topk8_rescore_mq6g256v2_k2560_rows` takes a 40-byte explicit image: `partial` @0, `head` @8,
//! `x` @16, `out` @24, `x_stride` @32 (floats), `out_stride` @36 (ints). Launch grid `[rows, 1, 1]`,
//! block 256, static LDS 128. Block `r` reads the slot `partial + r * 8192`, `x + r * x_stride`
//! (K = 2560, the head's FWHT basis) and writes the 18-word slot `out + r * out_stride`.
//!
//! Both objects carry the runtime's hidden block as the metadata lists it (base 40: block counts
//! `40/44/48`, group sizes `52/54/56`, remainders `58/60/62`, global offsets `80/88/96`, grid dims
//! `104`; no dynamic-LDS word), `.kernarg_segment_size` 296 / align 8. The rescore candidates read
//! head rows chosen by the partial slots; the head extent is the `vocab` rows of 2000 B.
//!
//! Admission that is not a function of the launch params stays in the launcher (`draft_rows_enqueue`
//! / `HIPFIRE_QWEN4_DRAFT_TOPK8_ROWS`: exact gfx1201, MQ6G256V2 head, K = 2560, `chunk * 128 >= n`,
//! `v_stride >= n`, partial slots written by the matching partial launch).

use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::mq6_i8::FORMAT_MQ6G256V2_WEIGHTS;

// ---- numeric contracts / formats -------------------------------------------------------------

/// Per row: exactly the one-row `topk8_partial_f32` launch's slot (128 groups of the 8 best
/// `(value, index)` pairs under `topk8_better`, indices mapped to token ids by
/// `front/special/tail`): integer and f32 selection only, no arithmetic.
pub const NUMERIC_TOPK8_PARTIAL_ROWS: NumericContractId =
    NumericContractId("qwen4_topk8_rows.partial_f32.v1");

/// Per row: exactly the one-row `topk8_rescore_mq6g256v2_k2560` launch's 18 output words (the
/// top-8 of the row's 1024 partial candidates scored with `mq6g256v2_row_dot`, K = 2560, then
/// `topk8_finish`).
pub const NUMERIC_TOPK8_RESCORE_ROWS: NumericContractId =
    NumericContractId("qwen4_topk8_rows.rescore_mq6g256v2_k2560.v1");

/// F32 `[rows][v_stride]` draft logits (vocabulary row `j` at `y * v_stride + j`).
pub const FORMAT_F32_ROWS_STRIDED: FormatId = FormatId("f32.rows_x_stride");
/// Per row an 8192-byte slot: `pv` 1024 f32 then `pi` 1024 i32 (128 groups x 8 candidates used).
pub const FORMAT_TOPK8_PARTIAL_SLOTS: FormatId = FormatId("topk8.partial_slots.8192b");
/// F32 head-basis activations `[rows][x_stride]`, K = 2560 live words per row.
pub const FORMAT_F32_X_K2560_STRIDED: FormatId = FormatId("f32.rows_x_stride.k2560");
/// Per row an 18-word i32 slot: token, margin bits, 8 candidates, 8 exact score bits.
pub const FORMAT_TOPK8_OUT_SLOTS: FormatId = FormatId("topk8.out_slots.18w");

// ---- shared pieces ---------------------------------------------------------------------------

const fn param(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}

const P_ROWS: Expr = Expr::Param("rows");
const P_N: Expr = Expr::Param("n");
const P_CHUNK: Expr = Expr::Param("chunk");
const P_FRONT: Expr = Expr::Param("front");
const P_SPECIAL: Expr = Expr::Param("special");
const P_TAIL: Expr = Expr::Param("tail");
const P_V_STRIDE: Expr = Expr::Param("v_stride");
const P_VOCAB: Expr = Expr::Param("vocab");
const P_X_STRIDE: Expr = Expr::Param("x_stride");
const P_OUT_STRIDE: Expr = Expr::Param("out_stride");

const ONE: Expr = Expr::Const(1);
const ROWS_M1: Expr = Expr::Sub(&P_ROWS, &ONE);
const PARTIAL_SLOT_BYTES: u64 = 8192;
const PARTIAL_BYTES: Expr = Expr::Mul(&[P_ROWS, Expr::Const(PARTIAL_SLOT_BYTES)]);

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

/// The runtime's hidden block as the object metadata lists it for a kernel that reads `gridDim`
/// (base 40 after the 40 explicit bytes, no dynamic-LDS word).
const HIDDEN: &[HiddenV1] = &[
    hid(40, 4, HiddenKind::BlockCount(Axis::X)),
    hid(44, 4, HiddenKind::BlockCount(Axis::Y)),
    hid(48, 4, HiddenKind::BlockCount(Axis::Z)),
    hid(52, 2, HiddenKind::GroupSize(Axis::X)),
    hid(54, 2, HiddenKind::GroupSize(Axis::Y)),
    hid(56, 2, HiddenKind::GroupSize(Axis::Z)),
    hid(58, 2, HiddenKind::Remainder(Axis::X)),
    hid(60, 2, HiddenKind::Remainder(Axis::Y)),
    hid(62, 2, HiddenKind::Remainder(Axis::Z)),
    hid(80, 8, HiddenKind::GlobalOffset(Axis::X)),
    hid(88, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hid(96, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hid(104, 2, HiddenKind::GridDims),
];

const I32_MAX: u64 = i32::MAX as u64;

const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}

const OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "topk8_rows_mq6g256v2",
    source: "kernels/src/topk8_rows.gfx1201.hip",
};

const CONSTS: &[(&str, u64)] = &[];

// ---- topk8_partial_f32_rows ------------------------------------------------------------------

const PARTIAL_PARAMS: &[ParamV1] = &[
    param("rows", "draft rows of the launch (grid y), 1..=8"),
    param("n", "vocabulary entries per row (logits scored)"),
    param("chunk", "entries per group (grid x = 128 groups), <= 2048"),
    param(
        "front",
        "head reorder: first `front` rows are tokens 0..front",
    ),
    param("special", "head reorder: first token id of the `tail` run"),
    param(
        "tail",
        "head reorder: length of the `[special, special + tail)` run",
    ),
    param("v_stride", "floats between consecutive rows of v"),
];

const PARTIAL_OPERANDS: &[OperandV1] = &[
    OperandV1 {
        name: "v",
        format: FORMAT_F32_ROWS_STRIDED,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::Add(&[Expr::Mul(&[ROWS_M1, P_V_STRIDE]), P_N]),
            Expr::Const(4),
        ])),
        nullable: false,
        note: "alignment = element size; row y reads v[y * v_stride .. y * v_stride + n)",
    },
    OperandV1 {
        name: "partial",
        format: FORMAT_TOPK8_PARTIAL_SLOTS,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(PARTIAL_BYTES),
        nullable: false,
        note: "alignment = element size; row y writes the slot at partial + y * 8192",
    },
];

const PARTIAL_FIELDS: &[FieldV1] = &[
    ptr("v", 0),
    scalar("n", 8, P_N),
    scalar("chunk", 12, P_CHUNK),
    ptr("partial", 16),
    scalar("front", 24, P_FRONT),
    scalar("special", 28, P_SPECIAL),
    scalar("tail", 32, P_TAIL),
    scalar("v_stride", 36, P_V_STRIDE),
];

const PARTIAL_DOMAIN: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_ROWS), "topk8_partial_rows: rows > 0"),
    rule(
        Rule::AtMost(P_ROWS, 8),
        "topk8_partial_rows: rows <= 8 (draft row cap)",
    ),
    rule(Rule::Positive(P_N), "topk8_partial_rows: n > 0"),
    rule(Rule::Positive(P_CHUNK), "chunk > 0"),
    rule(
        Rule::AtMost(P_CHUNK, 256 * 8),
        "kernel: each of the 256 threads holds 8 values, chunk <= 256 * 8",
    ),
    rule(Rule::Positive(P_V_STRIDE), "v_stride > 0"),
    rule(Rule::AtMost(P_N, I32_MAX), "n passed as i32"),
    rule(Rule::AtMost(P_CHUNK, I32_MAX), "chunk passed as i32"),
    rule(Rule::AtMost(P_FRONT, I32_MAX), "front passed as i32"),
    rule(Rule::AtMost(P_SPECIAL, I32_MAX), "special passed as i32"),
    rule(Rule::AtMost(P_TAIL, I32_MAX), "tail passed as i32"),
    rule(Rule::AtMost(P_V_STRIDE, I32_MAX), "v_stride passed as i32"),
];

/// `[128, rows, 1]`.
const PARTIAL_GRID: [Expr; 3] = [Expr::Const(128), P_ROWS, Expr::Const(1)];

const PARTIAL_EXPORTS: &[ExportV1] = &[ExportV1 {
    symbol: "topk8_partial_f32_rows",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [256, 1, 1],
    dynamic_lds: 0,
    group_segment_fixed: 64,
    grid: PARTIAL_GRID,
    consts: CONSTS,
}];

/// Row-batched `topk8_partial_f32`: 2 pointers + 6 i32 through kernelParams (40 explicit bytes),
/// hidden block at 40, segment 296.
pub static TOPK8_PARTIAL_F32_ROWS: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_topk8_rows.hipcc.gfx12.partial_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_TOPK8_PARTIAL_ROWS,
    params: PARTIAL_PARAMS,
    operands: PARTIAL_OPERANDS,
    fields: PARTIAL_FIELDS,
    hidden: HIDDEN,
    explicit_bytes: 40,
    segment_bytes: 296,
    segment_align: 8,
    domain: PARTIAL_DOMAIN,
    exports: PARTIAL_EXPORTS,
};

// ---- topk8_rescore_mq6g256v2_k2560_rows ------------------------------------------------------

const RESCORE_PARAMS: &[ParamV1] = &[
    param("rows", "draft rows of the launch (grid x), 1..=8"),
    param("vocab", "head rows (candidate token ids are < vocab)"),
    param("x_stride", "floats between consecutive rows of x"),
    param("out_stride", "ints between consecutive out slots"),
];

/// K = 2560 -> 10 groups of 256 x 200 B.
const HEAD_ROW_BYTES: u64 = 2000;

const RESCORE_OPERANDS: &[OperandV1] = &[
    OperandV1 {
        name: "partial",
        format: FORMAT_TOPK8_PARTIAL_SLOTS,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(PARTIAL_BYTES),
        nullable: false,
        note: "alignment = element size; block r reads the slot at partial + r * 8192 (1024 candidates)",
    },
    OperandV1 {
        name: "head",
        format: FORMAT_MQ6G256V2_WEIGHTS,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_VOCAB, Expr::Const(HEAD_ROW_BYTES)])),
        nullable: false,
        note: "MQ6G256V2 head, K = 2560; candidate rows come from the partial slots (token ids < vocab)",
    },
    OperandV1 {
        name: "x",
        format: FORMAT_F32_X_K2560_STRIDED,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::Add(&[Expr::Mul(&[ROWS_M1, P_X_STRIDE]), Expr::Const(2560)]),
            Expr::Const(4),
        ])),
        nullable: false,
        note: "alignment = element size; row r reads x[r * x_stride .. r * x_stride + 2560)",
    },
    OperandV1 {
        name: "out",
        format: FORMAT_TOPK8_OUT_SLOTS,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::Add(&[Expr::Mul(&[ROWS_M1, P_OUT_STRIDE]), Expr::Const(18)]),
            Expr::Const(4),
        ])),
        nullable: false,
        note: "alignment = element size; block r writes 18 words at out[r * out_stride]",
    },
];

const RESCORE_FIELDS: &[FieldV1] = &[
    ptr("partial", 0),
    ptr("head", 8),
    ptr("x", 16),
    ptr("out", 24),
    scalar("x_stride", 32, P_X_STRIDE),
    scalar("out_stride", 36, P_OUT_STRIDE),
];

const RESCORE_DOMAIN: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_ROWS), "topk8_rescore_k2560_rows: rows > 0"),
    rule(
        Rule::AtMost(P_ROWS, 8),
        "topk8_rescore_k2560_rows: rows <= 8 (draft row cap)",
    ),
    rule(Rule::Positive(P_VOCAB), "vocab > 0"),
    rule(Rule::AtMost(P_VOCAB, I32_MAX), "token ids are i32"),
    rule(
        Rule::AtLeast(P_X_STRIDE, 2560),
        "a row of x is K = 2560 floats; a smaller stride overlaps rows",
    ),
    rule(Rule::AtMost(P_X_STRIDE, I32_MAX), "x_stride passed as i32"),
    rule(
        Rule::AtLeast(P_OUT_STRIDE, 18),
        "an out slot is 18 words; a smaller stride overlaps slots",
    ),
    rule(
        Rule::AtMost(P_OUT_STRIDE, I32_MAX),
        "out_stride passed as i32",
    ),
];

/// `[rows, 1, 1]`.
const RESCORE_GRID: [Expr; 3] = [P_ROWS, Expr::Const(1), Expr::Const(1)];

const RESCORE_EXPORTS: &[ExportV1] = &[ExportV1 {
    symbol: "topk8_rescore_mq6g256v2_k2560_rows",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [256, 1, 1],
    dynamic_lds: 0,
    group_segment_fixed: 128,
    grid: RESCORE_GRID,
    consts: CONSTS,
}];

/// Row-batched `topk8_rescore_mq6g256v2_k2560`: 4 pointers + 2 i32 through kernelParams (40
/// explicit bytes), hidden block at 40, segment 296.
pub static TOPK8_RESCORE_MQ6G256V2_K2560_ROWS: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_topk8_rows.hipcc.gfx12.rescore_mq6g256v2_k2560",
    version: 1,
    wave: 32,
    numeric: NUMERIC_TOPK8_RESCORE_ROWS,
    params: RESCORE_PARAMS,
    operands: RESCORE_OPERANDS,
    fields: RESCORE_FIELDS,
    hidden: HIDDEN,
    explicit_bytes: 40,
    segment_bytes: 296,
    segment_align: 8,
    domain: RESCORE_DOMAIN,
    exports: RESCORE_EXPORTS,
};

/// Every contract of this family.
pub static ABIS: &[&KernelAbiV1] = &[&TOPK8_PARTIAL_F32_ROWS, &TOPK8_RESCORE_MQ6G256V2_K2560_ROWS];
