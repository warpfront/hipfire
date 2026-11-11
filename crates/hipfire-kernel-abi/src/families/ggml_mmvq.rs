// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the GGUF-native decode GEMV shim for gfx1201
//! (`kernels/src/ggml_mmvq_shim.hip`, object `kernels/ggml_mmvq_shim_gfx1201.co`,
//! built by `scripts/build-ggml-mmvq-shim.sh`): the `q8_1` activation producer
//! and the `iq4_xs` / `q4_k` / `q6_k` / `q8_0` quantised-weight GEMVs, all
//! transcribed from the pinned upstream snapshot in `third_party/ggml/`.
//!
//! Evidence the numbers below come from:
//!
//! * Object: `llvm-readelf --notes` of the gfx1201 ELF unbundled with
//!   `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201`
//!   from the committed `.co` (19064 B, `--no-offload-compress` bundle). Every
//!   entry: `.kernarg_segment_align` 8, `.group_segment_fixed_size` as stated
//!   per export, `.private_segment_fixed_size` 0, `.wavefront_size` 32,
//!   `.max_flat_workgroup_size` 256. Producer: args `x@0 vy@8` (`global_buffer`,
//!   8 B), `ne00@16 s01@20 ne0@24` (`by_value`, 4 B), hidden words
//!   `block_count_{x,y,z}@32/36/40` (4 B), `group_size_{x,y,z}@44/46/48`,
//!   `remainder_{x,y,z}@50/52/54` (2 B), `global_offset_{x,y,z}@72/80/88` (8 B),
//!   `grid_dims@96` (2 B), `.kernarg_segment_size` 288, LDS 0. mmvq entries:
//!   args `vx@0 vy@8 dst@16` (`global_buffer`) and `K@24` (`by_value`), NO
//!   hidden args, `.kernarg_segment_size` 28, static LDS 896 (= 7 x 32 x 4, the
//!   `tmp_shared[NWARPS-1][WARP]` partials). The metadata names no argument and
//!   carries no `.actual_access`, so the field names below are the shim header's.
//! * Launch geometry: shim header comment (`ggml_mmvq_shim.hip:20-47`): producer
//!   block (256,1,1), grid (ceil(ne0/256), n_rows, 1); mmvq block
//!   (32,8,1) = (warp, nwarps), grid (M,1,1), one weight row per workgroup,
//!   dynamic LDS 0. M is the grid, not a kernarg.
//! * Weight block sizes (`ggml-common.h`): `block_iq4_xs` 136 B / 256 elems,
//!   `block_q4_K` 144 B / 256, `block_q6_K` 210 B / 256, `block_q8_0` 34 B / 32;
//!   activations `block_q8_1` 36 B / 32 elems (half2 `ds` + int8 `qs[32]`).
//! * Alignment, from `vecdotq.cuh` / `common.cuh`: `get_int_b4` is
//!   `((const int *) x)[i]` ("assume at least 4 byte alignment"), `get_int_b2`
//!   is two `uint16_t` loads ("assume at least 2 byte alignment"). `iq4_xs`
//!   (`get_int_b4(bq4->qs)`, `qs` at block offset 8, block 136) and `q4_k`
//!   (`(const int *)(bq4_K->qs + ...)`, `qs` at offset 16, block 144) read
//!   ints at 4-aligned offsets inside blocks whose size is a multiple of 4, so
//!   `vx` needs 4. `q6_k` (`get_int_b2` of `ql`/`qh`, block 210) and `q8_0`
//!   (`get_int_b2(bq8_0->qs)`, `qs` at offset 2, block 34) only ever load
//!   16-bit pairs and their block sizes are 2 mod 4, so 2 is all the code
//!   needs (a 4-byte base would not even give 4-byte rows); the shim header's
//!   "4 for q8_0" is therefore stricter than the code. `vy` (`get_int_b4` of
//!   `qs` at offset 4 of 36 B blocks) and f32 operands need 4.
//! * Address math: the kernel computes `row0 * blocks_per_row + kbx` in `int`,
//!   so `M * (K / qk) <= i32::MAX` is a domain rule; the block pointer
//!   arithmetic after that is 64-bit. The producer indexes
//!   `i1 * ne0 + i0` and `i1 * s01 + i0` in `int`: `NROWS * NE0` and the x
//!   element extent `(NROWS - 1) * S01 + K` are bounded by `i32::MAX`.
//! * Numeric: fixed reduction order (per-thread ascending-`kbx` fold, 7 LDS
//!   partials added l = 0..6, `__shfl_xor` butterfly), fp32 FMA contraction as
//!   compiled, no atomics; exact int32 dp4a block dots (`vec_dot_*_q8_1`).

use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind, HiddenV1,
    KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- formats ---------------------------------------------------------------

/// `block_iq4_xs` rows: `M x K/256` blocks of 136 B.
pub const FORMAT_IQ4_XS_136: FormatId = FormatId("ggml.iq4_xs.136");
/// `block_q4_K` rows: `M x K/256` blocks of 144 B.
pub const FORMAT_Q4_K_144: FormatId = FormatId("ggml.q4_k.144");
/// `block_q6_K` rows: `M x K/256` blocks of 210 B.
pub const FORMAT_Q6_K_210: FormatId = FormatId("ggml.q6_k.210");
/// `block_q8_0` rows: `M x K/32` blocks of 34 B.
pub const FORMAT_Q8_0_34: FormatId = FormatId("ggml.q8_0.34");
/// `block_q8_1` activation rows in the mmvq layout: `NROWS x NE0/32` blocks of
/// 36 B (`half2 ds {d, sum}; int8 qs[32]`).
pub const FORMAT_Q8_1_MMVQ: FormatId = FormatId("ggml.q8_1.mmvq");
/// F32 activation rows: row `i` at element `i * S01`, `K` valid floats.
pub const FORMAT_F32_ROWS_S01: FormatId = FormatId("f32.rows_x_s01");
/// F32 GEMV output `[M]`.
pub const FORMAT_F32_M: FormatId = FormatId("f32.m");

// ---- numeric contracts -------------------------------------------------------

pub const NUMERIC_IQ4_XS: NumericContractId = NumericContractId("ggml.a8q81.iq4_xs");
pub const NUMERIC_Q4_K: NumericContractId = NumericContractId("ggml.a8q81.q4_k");
pub const NUMERIC_Q6_K: NumericContractId = NumericContractId("ggml.a8q81.q6_k");
pub const NUMERIC_Q8_0: NumericContractId = NumericContractId("ggml.a8q81.q8_0");
pub const NUMERIC_Q8_1_PRODUCER: NumericContractId = NumericContractId("ggml.a8q81.q8_1_producer");

// ---- shared pieces -----------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_S01: Expr = Expr::Param("S01");
const P_NE0: Expr = Expr::Param("NE0");
const P_NROWS: Expr = Expr::Param("NROWS");
const ONE: Expr = Expr::Const(1);
const I32_MAX: u64 = i32::MAX as u64;

/// Weight blocks, 256-element formats: `M * (K / 256)`.
const BLOCKS_256: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]);
/// Weight blocks, `q8_0`: `M * (K / 32)`.
const BLOCKS_32: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, 32)]);
const IQ4_XS_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(136)]);
const Q4_K_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(144)]);
const Q6_K_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(210)]);
const Q8_0_BYTES: Expr = Expr::Mul(&[BLOCKS_32, Expr::Const(34)]);
/// mmvq activation extent: one row, `K / 32` blocks of 36 B.
const MMVQ_VY_BYTES: Expr = Expr::Mul(&[Expr::DivExact(&P_K, 32), Expr::Const(36)]);
/// mmvq output extent: `M * 4`.
const MMVQ_DST_BYTES: Expr = Expr::Mul(&[P_M, Expr::Const(4)]);
/// Producer input extent in elements: `(NROWS - 1) * S01 + K`.
const X_ELEMS: Expr = Expr::Add(&[Expr::Mul(&[Expr::Sub(&P_NROWS, &ONE), P_S01]), P_K]);
const X_BYTES: Expr = Expr::Mul(&[X_ELEMS, Expr::Const(4)]);
/// Producer output extent: `NROWS * (NE0 / 32) * 36`.
const PRODUCER_VY_BYTES: Expr = Expr::Mul(&[P_NROWS, Expr::DivExact(&P_NE0, 32), Expr::Const(36)]);
/// Producer index bound: `NROWS * NE0` (`i1 * ne0 + i0` in `int`).
const PRODUCER_ELEMS: Expr = Expr::Mul(&[P_NROWS, P_NE0]);

const PARAMS_MMVQ: &[ParamV1] = &[
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "weight rows (the grid; not a kernarg)",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction length",
    },
];

const PARAMS_PRODUCER: &[ParamV1] = &[
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "valid elements per row (`ne00`)",
    },
    ParamV1 {
        name: "S01",
        ty: ScalarTy::I32,
        doc: "row stride of the f32 input, in floats",
    },
    ParamV1 {
        name: "NE0",
        ty: ScalarTy::I32,
        doc: "padded elements per row, multiple of 32, >= K",
    },
    ParamV1 {
        name: "NROWS",
        ty: ScalarTy::I32,
        doc: "activation rows (grid y)",
    },
];

// ---- operands -----------------------------------------------------------------

const OP_VX_IQ4_XS: OperandV1 = OperandV1 {
    name: "vx",
    format: FORMAT_IQ4_XS_136,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(IQ4_XS_BYTES),
    nullable: false,
    note:
        "alignment 4: get_int_b4 reads ints at 4-aligned offsets of 4-multiple blocks (vecdotq.cuh)",
};
const OP_VX_Q4_K: OperandV1 = OperandV1 {
    name: "vx",
    format: FORMAT_Q4_K_144,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(Q4_K_BYTES),
    nullable: false,
    note:
        "alignment 4: get_int_b4 reads ints at 4-aligned offsets of 4-multiple blocks (vecdotq.cuh)",
};
const OP_VX_Q6_K: OperandV1 = OperandV1 {
    name: "vx",
    format: FORMAT_Q6_K_210,
    access: Access::ReadOnly,
    align: 2,
    extent: Some(Q6_K_BYTES),
    nullable: false,
    note: "alignment 2: only get_int_b2 (16-bit pairs) reads weights; blocks are 2 mod 4 bytes (vecdotq.cuh)",
};
const OP_VX_Q8_0: OperandV1 = OperandV1 {
    name: "vx",
    format: FORMAT_Q8_0_34,
    access: Access::ReadOnly,
    align: 2,
    extent: Some(Q8_0_BYTES),
    nullable: false,
    note: "alignment 2: only get_int_b2 (16-bit pairs) reads weights; blocks are 2 mod 4 bytes (vecdotq.cuh)",
};
const OP_VY_MMVQ: OperandV1 = OperandV1 {
    name: "vy",
    format: FORMAT_Q8_1_MMVQ,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(MMVQ_VY_BYTES),
    nullable: false,
    note: "alignment 4: get_int_b4 of block_q8_1.qs (offset 4 of 36 B blocks)",
};
const OP_DST: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F32_M,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(MMVQ_DST_BYTES),
    nullable: false,
    note: "alignment 4: f32 store, one per weight row",
};
const OP_X: OperandV1 = OperandV1 {
    name: "x",
    format: FORMAT_F32_ROWS_S01,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(X_BYTES),
    nullable: false,
    note: "alignment 4: f32 loads",
};
const OP_VY_PRODUCER: OperandV1 = OperandV1 {
    name: "vy",
    format: FORMAT_Q8_1_MMVQ,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(PRODUCER_VY_BYTES),
    nullable: false,
    note: "alignment 4: block_q8_1 has a half2 member; byte stores to qs, 4-byte store of ds",
};

const OPERANDS_IQ4_XS: &[OperandV1] = &[OP_VX_IQ4_XS, OP_VY_MMVQ, OP_DST];
const OPERANDS_Q4_K: &[OperandV1] = &[OP_VX_Q4_K, OP_VY_MMVQ, OP_DST];
const OPERANDS_Q6_K: &[OperandV1] = &[OP_VX_Q6_K, OP_VY_MMVQ, OP_DST];
const OPERANDS_Q8_0: &[OperandV1] = &[OP_VX_Q8_0, OP_VY_MMVQ, OP_DST];
const OPERANDS_PRODUCER: &[OperandV1] = &[OP_X, OP_VY_PRODUCER];

// ---- fields --------------------------------------------------------------------

const F_VX: FieldV1 = FieldV1 {
    name: "vx",
    offset: 0,
    width: 8,
    kind: FieldKind::Pointer { operand: "vx" },
};
const F_VY: FieldV1 = FieldV1 {
    name: "vy",
    offset: 8,
    width: 8,
    kind: FieldKind::Pointer { operand: "vy" },
};
const F_DST: FieldV1 = FieldV1 {
    name: "dst",
    offset: 16,
    width: 8,
    kind: FieldKind::Pointer { operand: "dst" },
};
const F_K_MMVQ: FieldV1 = FieldV1 {
    name: "K",
    offset: 24,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};
const F_X: FieldV1 = FieldV1 {
    name: "x",
    offset: 0,
    width: 8,
    kind: FieldKind::Pointer { operand: "x" },
};
const F_VY_PRODUCER: FieldV1 = FieldV1 {
    name: "vy",
    offset: 8,
    width: 8,
    kind: FieldKind::Pointer { operand: "vy" },
};
const F_NE00: FieldV1 = FieldV1 {
    name: "ne00",
    offset: 16,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};
const F_S01: FieldV1 = FieldV1 {
    name: "s01",
    offset: 20,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_S01,
    },
};
const F_NE0: FieldV1 = FieldV1 {
    name: "ne0",
    offset: 24,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_NE0,
    },
};

const FIELDS_MMVQ: &[FieldV1] = &[F_VX, F_VY, F_DST, F_K_MMVQ];
const FIELDS_PRODUCER: &[FieldV1] = &[F_X, F_VY_PRODUCER, F_NE00, F_S01, F_NE0];

/// The mmvq entries have no hidden arguments (`.kernarg_segment_size` 28).
const HIDDEN_MMVQ: &[HiddenV1] = &[];

/// Producer hidden block, exactly the object metadata's offsets/widths.
const HIDDEN_PRODUCER: &[HiddenV1] = &[
    HiddenV1 {
        offset: 32,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::X),
    },
    HiddenV1 {
        offset: 36,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 40,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 44,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::X),
    },
    HiddenV1 {
        offset: 46,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 48,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 50,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::X),
    },
    HiddenV1 {
        offset: 52,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 54,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 72,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::X),
    },
    HiddenV1 {
        offset: 80,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 88,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 96,
        width: 2,
        kind: HiddenKind::GridDims,
    },
];

// ---- domains -------------------------------------------------------------------

/// mmvq domain for the 256-element formats (`iq4_xs`, `q4_k`, `q6_k`).
const DOMAIN_K256: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "K > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "shim header: K % 256 == 0 (whole weight blocks per row)",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "M >= 1 (grid x)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "K is an i32 kernarg",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "M is the grid and a row index in int",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(BLOCKS_256, I32_MAX),
        why: "`row0 * blocks_per_row + kbx` is computed in int",
    },
];

/// mmvq domain for `q8_0` (32-element blocks).
const DOMAIN_Q8_0: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "K > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 32),
        why: "shim header: K % 32 == 0 (whole weight blocks per row)",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "M >= 1 (grid x)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "K is an i32 kernarg",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "M is the grid and a row index in int",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(BLOCKS_32, I32_MAX),
        why: "`row0 * blocks_per_row + kbx` is computed in int",
    },
];

/// Producer domain (shim header: `ne0 % 32 == 0`, `ne0 >= ne00 > 0`,
/// `n_rows * ne0 <= INT32_MAX`). `ne0 >= ne00` is expressed as
/// `NE0 - K >= 0`: a negative difference is an expression error, which
/// `validate_invocation` rejects like a failed rule.
const DOMAIN_PRODUCER: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "ne00 > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_NE0, 32),
        why: "ne0 % 32 == 0 (whole q8_1 blocks)",
    },
    DomainRuleV1 {
        rule: Rule::AtLeast(Expr::Sub(&P_NE0, &P_K), 0),
        why: "ne0 >= ne00 (negative NE0 - K is an expression error)",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_NROWS),
        why: "n_rows >= 1 (grid y)",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_S01),
        why: "s01 > 0: rows are distinct strided rows of x",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(PRODUCER_ELEMS, I32_MAX),
        why: "`i1 * ne0 + i0` is computed in int",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(X_ELEMS, I32_MAX),
        why: "`i1 * s01 + i0` is computed in int",
    },
];

// ---- exports -------------------------------------------------------------------

const OBJECT: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "ggml_mmvq_shim_gfx1201",
    artifact: "kernels/ggml_mmvq_shim_gfx1201.co",
    // `sha256sum` of the committed offload bundle (build is deterministic).
    sha256: "8b50bc03ef6f719478cba6d22aa5f6c734339dab9907d6bceaf217c9492e1a4c",
};

const GRID_MMVQ: [Expr; 3] = [P_M, Expr::Const(1), Expr::Const(1)];
const GRID_PRODUCER: [Expr; 3] = [Expr::DivCeil(&P_NE0, 256), P_NROWS, Expr::Const(1)];
const CONSTS_MMVQ: &[(&str, u64)] = &[("nwarps", 8), ("rows_per_block", 1)];
const CONSTS_PRODUCER: &[(&str, u64)] = &[];

const EXPORTS_IQ4_XS: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_mmvq_iq4_xs_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [32, 8, 1],
    dynamic_lds: 0,
    group_segment_fixed: 896,
    grid: GRID_MMVQ,
    consts: CONSTS_MMVQ,
}];
const EXPORTS_Q4_K: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_mmvq_q4_k_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [32, 8, 1],
    dynamic_lds: 0,
    group_segment_fixed: 896,
    grid: GRID_MMVQ,
    consts: CONSTS_MMVQ,
}];
const EXPORTS_Q6_K: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_mmvq_q6_k_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [32, 8, 1],
    dynamic_lds: 0,
    group_segment_fixed: 896,
    grid: GRID_MMVQ,
    consts: CONSTS_MMVQ,
}];
const EXPORTS_Q8_0: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_mmvq_q8_0_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [32, 8, 1],
    dynamic_lds: 0,
    group_segment_fixed: 896,
    grid: GRID_MMVQ,
    consts: CONSTS_MMVQ,
}];
const EXPORTS_PRODUCER: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_q8_1_quantize_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [256, 1, 1],
    dynamic_lds: 0,
    group_segment_fixed: 0,
    grid: GRID_PRODUCER,
    consts: CONSTS_PRODUCER,
}];

// ---- contracts -----------------------------------------------------------------

/// `q8_1` activation producer: `x vy ne00 s01 ne0`, explicit 28 B, segment 288/8.
pub static GGML_Q8_1_PRODUCER: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq.q8_1_producer.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_1_PRODUCER,
    params: PARAMS_PRODUCER,
    operands: OPERANDS_PRODUCER,
    fields: FIELDS_PRODUCER,
    hidden: HIDDEN_PRODUCER,
    explicit_bytes: 28,
    segment_bytes: 288,
    segment_align: 8,
    domain: DOMAIN_PRODUCER,
    exports: EXPORTS_PRODUCER,
};

/// `iq4_xs` decode GEMV: `vx vy dst K`, no hidden args, segment 28/8, static LDS 896.
pub static GGML_MMVQ_IQ4_XS: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq.iq4_xs.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_IQ4_XS,
    params: PARAMS_MMVQ,
    operands: OPERANDS_IQ4_XS,
    fields: FIELDS_MMVQ,
    hidden: HIDDEN_MMVQ,
    explicit_bytes: 28,
    segment_bytes: 28,
    segment_align: 8,
    domain: DOMAIN_K256,
    exports: EXPORTS_IQ4_XS,
};

/// `q4_k` decode GEMV: `vx vy dst K`, no hidden args, segment 28/8, static LDS 896.
pub static GGML_MMVQ_Q4_K: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq.q4_k.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q4_K,
    params: PARAMS_MMVQ,
    operands: OPERANDS_Q4_K,
    fields: FIELDS_MMVQ,
    hidden: HIDDEN_MMVQ,
    explicit_bytes: 28,
    segment_bytes: 28,
    segment_align: 8,
    domain: DOMAIN_K256,
    exports: EXPORTS_Q4_K,
};

/// `q6_k` decode GEMV: `vx vy dst K`, no hidden args, segment 28/8, static LDS 896.
pub static GGML_MMVQ_Q6_K: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq.q6_k.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q6_K,
    params: PARAMS_MMVQ,
    operands: OPERANDS_Q6_K,
    fields: FIELDS_MMVQ,
    hidden: HIDDEN_MMVQ,
    explicit_bytes: 28,
    segment_bytes: 28,
    segment_align: 8,
    domain: DOMAIN_K256,
    exports: EXPORTS_Q6_K,
};

/// `q8_0` decode GEMV: `vx vy dst K`, no hidden args, segment 28/8, static LDS 896.
pub static GGML_MMVQ_Q8_0: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq.q8_0.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_0,
    params: PARAMS_MMVQ,
    operands: OPERANDS_Q8_0,
    fields: FIELDS_MMVQ,
    hidden: HIDDEN_MMVQ,
    explicit_bytes: 28,
    segment_bytes: 28,
    segment_align: 8,
    domain: DOMAIN_Q8_0,
    exports: EXPORTS_Q8_0,
};

/// Every contract of this family, producer first.
pub static ABIS: &[&KernelAbiV1] = &[
    &GGML_Q8_1_PRODUCER,
    &GGML_MMVQ_IQ4_XS,
    &GGML_MMVQ_Q4_K,
    &GGML_MMVQ_Q6_K,
    &GGML_MMVQ_Q8_0,
];
