// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the S3 additions to the GGUF-native decode GEMV set for gfx1201
//! (`kernels/src/ggml_mmvq_fused_shim.hip`, object
//! `kernels/ggml_mmvq_fused_shim_gfx1201.co`, built by
//! `scripts/build-ggml-mmvq-fused-shim.sh`): the `q5_k` GEMV, the
//! `dst = dst + dot` (`_res`) variants of the `iq4_xs` / `q4_k` / `q5_k` /
//! `q6_k` / `q8_0` GEMVs, and the block-gather `q8_1` producer. Companion of
//! [`super::ggml_mmvq`] (S1; its formats, `iq4_xs`/`q4_k`/`q6_k`/`q8_0` weight
//! formats and the `q8_1` activation format are reused by name); everything
//! the S1 module records about the object metadata, the launch geometry, the
//! alignment rules, the address math and the fixed reduction order holds here
//! unchanged, plus:
//!
//! * `q5_k`: `block_q5_K` 176 B / 256 elems (`ggml-common.h`: `dm` half2 at 0,
//!   `scales[12]` at 4, `qh[32]` at 16, `qs[128]` at 48). `vec_dot_q5_K_q8_1`
//!   (`vecdotq.cuh`) reads `(const int *)(bq5_K->qs + ...)` and
//!   `(const int *)(bq5_K->qh + ...)` at 4-aligned offsets of a 4-multiple
//!   block, and `uint16_t` scale pairs, so `vx` needs 4. Upstream constants:
//!   `VDR_Q5_K_Q8_1_MMVQ` 2, `QI5_K` 32, nwarps 8 on RDNA4 (S0 census 3.6).
//! * `_res`: after the unchanged reduction, lane 0 of warp 0 stores
//!   `dst[row] = dst[row] + tmp` (one f32 add of the previous value). `dst` is
//!   therefore a READ-WRITE operand (`Access::ReadWrite`): the residual
//!   stream. One workgroup owns one row, so there is no concurrent writer and
//!   no atomic. Numeric contract `ggml.a8q81.<type>.res`.
//! * gather producer: kernargs `x@0 vy@8 blk_map@16` (pointers) `ne0@24`
//!   (int32), grid (ceil(ne0/256),1,1), block (256,1,1), no LDS; output block
//!   `j` is quantised from `x[blk_map[j]*32 .. +32]`. Domain: `ne0 % 32 == 0`;
//!   every `blk_map` entry in `[0, ne0/32)` (data-dependent, so it is the
//!   caller's obligation and cannot be an [`Expr`] rule). With the identity map
//!   the bytes equal the S1 producer's (one row, `ne00 = s01 = ne0`).

use super::ggml_mmvq::{
    FORMAT_F32_M, FORMAT_F32_ROWS_S01, FORMAT_IQ4_XS_136, FORMAT_Q4_K_144, FORMAT_Q6_K_210,
    FORMAT_Q8_0_34, FORMAT_Q8_1_MMVQ,
};
use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind, HiddenV1,
    KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- formats ---------------------------------------------------------------

/// `block_q5_K` rows: `M x K/256` blocks of 176 B.
pub const FORMAT_Q5_K_176: FormatId = FormatId("ggml.q5_k.176");
/// Gather map: `K/32` little-endian i32 source-block indices.
pub const FORMAT_I32_BLK_MAP: FormatId = FormatId("i32.blk_map");

// ---- numeric contracts -------------------------------------------------------

pub const NUMERIC_Q5_K: NumericContractId = NumericContractId("ggml.a8q81.q5_k");
pub const NUMERIC_IQ4_XS_RES: NumericContractId = NumericContractId("ggml.a8q81.iq4_xs.res");
pub const NUMERIC_Q4_K_RES: NumericContractId = NumericContractId("ggml.a8q81.q4_k.res");
pub const NUMERIC_Q5_K_RES: NumericContractId = NumericContractId("ggml.a8q81.q5_k.res");
pub const NUMERIC_Q6_K_RES: NumericContractId = NumericContractId("ggml.a8q81.q6_k.res");
pub const NUMERIC_Q8_0_RES: NumericContractId = NumericContractId("ggml.a8q81.q8_0.res");
pub const NUMERIC_Q8_1_GATHER: NumericContractId =
    NumericContractId("ggml.a8q81.q8_1_producer.gather");

// ---- shared pieces -----------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const I32_MAX: u64 = i32::MAX as u64;

const BLOCKS_256: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]);
const BLOCKS_32: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, 32)]);
const IQ4_XS_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(136)]);
const Q4_K_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(144)]);
const Q5_K_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(176)]);
const Q6_K_BYTES: Expr = Expr::Mul(&[BLOCKS_256, Expr::Const(210)]);
const Q8_0_BYTES: Expr = Expr::Mul(&[BLOCKS_32, Expr::Const(34)]);
/// mmvq activation extent: one row, `K / 32` blocks of 36 B.
const MMVQ_VY_BYTES: Expr = Expr::Mul(&[Expr::DivExact(&P_K, 32), Expr::Const(36)]);
/// mmvq output extent: `M * 4`.
const MMVQ_DST_BYTES: Expr = Expr::Mul(&[P_M, Expr::Const(4)]);
/// Gather producer input extent: `K` floats.
const GATHER_X_BYTES: Expr = Expr::Mul(&[P_K, Expr::Const(4)]);
/// Gather map extent: `K / 32` i32.
const GATHER_MAP_BYTES: Expr = Expr::Mul(&[Expr::DivExact(&P_K, 32), Expr::Const(4)]);

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

const PARAMS_GATHER: &[ParamV1] = &[ParamV1 {
    name: "K",
    ty: ScalarTy::I32,
    doc: "elements of the activation row (`ne0`), multiple of 32",
}];

// ---- operands -----------------------------------------------------------------

const NOTE_VX4: &str =
    "alignment 4: get_int_b4 / int loads at 4-aligned offsets of 4-multiple blocks (vecdotq.cuh)";
const NOTE_VX2: &str =
    "alignment 2: only get_int_b2 (16-bit pairs) reads weights; blocks are 2 mod 4 bytes (vecdotq.cuh)";

const fn vx(format: FormatId, align: u32, extent: Expr, note: &'static str) -> OperandV1 {
    OperandV1 {
        name: "vx",
        format,
        access: Access::ReadOnly,
        align,
        extent: Some(extent),
        nullable: false,
        note,
    }
}

const OP_VX_IQ4_XS: OperandV1 = vx(FORMAT_IQ4_XS_136, 4, IQ4_XS_BYTES, NOTE_VX4);
const OP_VX_Q4_K: OperandV1 = vx(FORMAT_Q4_K_144, 4, Q4_K_BYTES, NOTE_VX4);
const OP_VX_Q5_K: OperandV1 = vx(
    FORMAT_Q5_K_176,
    4,
    Q5_K_BYTES,
    "alignment 4: int loads of qs / qh and u16 scale pairs at 4-aligned offsets of 176 B blocks (vecdotq.cuh vec_dot_q5_K_q8_1)",
);
const OP_VX_Q6_K: OperandV1 = vx(FORMAT_Q6_K_210, 2, Q6_K_BYTES, NOTE_VX2);
const OP_VX_Q8_0: OperandV1 = vx(FORMAT_Q8_0_34, 2, Q8_0_BYTES, NOTE_VX2);

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
const OP_DST_RES: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F32_M,
    access: Access::ReadWrite,
    align: 4,
    extent: Some(MMVQ_DST_BYTES),
    nullable: false,
    note: "alignment 4: the residual stream; read then stored once per weight row (dst[row] = dst[row] + dot)",
};
const OP_X_GATHER: OperandV1 = OperandV1 {
    name: "x",
    format: FORMAT_F32_ROWS_S01,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(GATHER_X_BYTES),
    nullable: false,
    note: "alignment 4: f32 loads, one row of K floats, read through blk_map",
};
const OP_VY_GATHER: OperandV1 = OperandV1 {
    name: "vy",
    format: FORMAT_Q8_1_MMVQ,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(MMVQ_VY_BYTES),
    nullable: false,
    note: "alignment 4: block_q8_1 has a half2 member; byte stores to qs, 4-byte store of ds",
};
const OP_BLK_MAP: OperandV1 = OperandV1 {
    name: "blk_map",
    format: FORMAT_I32_BLK_MAP,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(GATHER_MAP_BYTES),
    nullable: false,
    note: "alignment 4: i32 loads; every entry must lie in [0, K/32) (data-dependent: the caller's obligation)",
};

const OPERANDS_IQ4_XS: &[OperandV1] = &[OP_VX_IQ4_XS, OP_VY_MMVQ, OP_DST_RES];
const OPERANDS_Q4_K: &[OperandV1] = &[OP_VX_Q4_K, OP_VY_MMVQ, OP_DST_RES];
const OPERANDS_Q5_K: &[OperandV1] = &[OP_VX_Q5_K, OP_VY_MMVQ, OP_DST];
const OPERANDS_Q5_K_RES: &[OperandV1] = &[OP_VX_Q5_K, OP_VY_MMVQ, OP_DST_RES];
const OPERANDS_Q6_K: &[OperandV1] = &[OP_VX_Q6_K, OP_VY_MMVQ, OP_DST_RES];
const OPERANDS_Q8_0: &[OperandV1] = &[OP_VX_Q8_0, OP_VY_MMVQ, OP_DST_RES];
const OPERANDS_GATHER: &[OperandV1] = &[OP_X_GATHER, OP_VY_GATHER, OP_BLK_MAP];

// ---- fields --------------------------------------------------------------------

const fn ptr(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand: name },
    }
}
const F_K: FieldV1 = FieldV1 {
    name: "K",
    offset: 24,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};
const F_NE0: FieldV1 = FieldV1 {
    name: "ne0",
    offset: 24,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};

const FIELDS_MMVQ: &[FieldV1] = &[ptr("vx", 0), ptr("vy", 8), ptr("dst", 16), F_K];
const FIELDS_GATHER: &[FieldV1] = &[ptr("x", 0), ptr("vy", 8), ptr("blk_map", 16), F_NE0];

/// The mmvq entries have no hidden arguments (`.kernarg_segment_size` 28).
const HIDDEN_MMVQ: &[HiddenV1] = &[];

/// Gather producer hidden block, exactly the object metadata's offsets/widths (the
/// same block as the S1 producer's).
const HIDDEN_GATHER: &[HiddenV1] = &[
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

/// mmvq domain for the 256-element formats (`iq4_xs`, `q4_k`, `q5_k`, `q6_k`).
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

/// Gather producer domain (shim header: `ne0 % 32 == 0`, `ne0 > 0`).
const DOMAIN_GATHER: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "ne0 > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 32),
        why: "ne0 % 32 == 0 (whole q8_1 blocks; every lane of a 32-group takes the same early return)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "ne0 is an i32 kernarg",
    },
];

// ---- exports -------------------------------------------------------------------

const OBJECT: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "ggml_mmvq_fused_shim_gfx1201",
    artifact: "kernels/ggml_mmvq_fused_shim_gfx1201.co",
    // `sha256sum` of the committed offload bundle (build is deterministic).
    sha256: "8cdc691ba2c97e24eea87bd5fd5991036101bcad398ca3d626ff8341ef10f91b",
};

const GRID_MMVQ: [Expr; 3] = [P_M, Expr::Const(1), Expr::Const(1)];
const GRID_GATHER: [Expr; 3] = [Expr::DivCeil(&P_K, 256), Expr::Const(1), Expr::Const(1)];
const CONSTS_MMVQ: &[(&str, u64)] = &[("nwarps", 8), ("rows_per_block", 1)];
const CONSTS_GATHER: &[(&str, u64)] = &[];

const fn mmvq_export(symbol: &'static str) -> ExportV1 {
    ExportV1 {
        symbol,
        arch: Arch::Gfx1201,
        object: OBJECT,
        block: [32, 8, 1],
        dynamic_lds: 0,
        group_segment_fixed: 896,
        grid: GRID_MMVQ,
        consts: CONSTS_MMVQ,
    }
}

const EXPORTS_Q5_K: &[ExportV1] = &[mmvq_export("ggml_mmvq_q5_k_gfx1201")];
const EXPORTS_IQ4_XS_RES: &[ExportV1] = &[mmvq_export("ggml_mmvq_iq4_xs_res_gfx1201")];
const EXPORTS_Q4_K_RES: &[ExportV1] = &[mmvq_export("ggml_mmvq_q4_k_res_gfx1201")];
const EXPORTS_Q5_K_RES: &[ExportV1] = &[mmvq_export("ggml_mmvq_q5_k_res_gfx1201")];
const EXPORTS_Q6_K_RES: &[ExportV1] = &[mmvq_export("ggml_mmvq_q6_k_res_gfx1201")];
const EXPORTS_Q8_0_RES: &[ExportV1] = &[mmvq_export("ggml_mmvq_q8_0_res_gfx1201")];
const EXPORTS_GATHER: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_q8_1_quantize_gather_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [256, 1, 1],
    dynamic_lds: 0,
    group_segment_fixed: 0,
    grid: GRID_GATHER,
    consts: CONSTS_GATHER,
}];

// ---- contracts -----------------------------------------------------------------

/// A decode GEMV contract: `vx vy dst K`, no hidden args, segment 28/8, static LDS 896.
const fn mmvq_abi(
    id: &'static str,
    numeric: NumericContractId,
    operands: &'static [OperandV1],
    domain: &'static [DomainRuleV1],
    exports: &'static [ExportV1],
) -> KernelAbiV1 {
    KernelAbiV1 {
        id,
        version: 1,
        wave: 32,
        numeric,
        params: PARAMS_MMVQ,
        operands,
        fields: FIELDS_MMVQ,
        hidden: HIDDEN_MMVQ,
        explicit_bytes: 28,
        segment_bytes: 28,
        segment_align: 8,
        domain,
        exports,
    }
}

/// `q5_k` decode GEMV, `dst[row] = dot`.
pub static GGML_MMVQ_Q5_K: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.q5_k.gfx1201",
    NUMERIC_Q5_K,
    OPERANDS_Q5_K,
    DOMAIN_K256,
    EXPORTS_Q5_K,
);
/// `iq4_xs` decode GEMV, `dst[row] = dst[row] + dot`.
pub static GGML_MMVQ_IQ4_XS_RES: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.iq4_xs_res.gfx1201",
    NUMERIC_IQ4_XS_RES,
    OPERANDS_IQ4_XS,
    DOMAIN_K256,
    EXPORTS_IQ4_XS_RES,
);
/// `q4_k` decode GEMV, `dst[row] = dst[row] + dot`.
pub static GGML_MMVQ_Q4_K_RES: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.q4_k_res.gfx1201",
    NUMERIC_Q4_K_RES,
    OPERANDS_Q4_K,
    DOMAIN_K256,
    EXPORTS_Q4_K_RES,
);
/// `q5_k` decode GEMV, `dst[row] = dst[row] + dot`.
pub static GGML_MMVQ_Q5_K_RES: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.q5_k_res.gfx1201",
    NUMERIC_Q5_K_RES,
    OPERANDS_Q5_K_RES,
    DOMAIN_K256,
    EXPORTS_Q5_K_RES,
);
/// `q6_k` decode GEMV, `dst[row] = dst[row] + dot`.
pub static GGML_MMVQ_Q6_K_RES: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.q6_k_res.gfx1201",
    NUMERIC_Q6_K_RES,
    OPERANDS_Q6_K,
    DOMAIN_K256,
    EXPORTS_Q6_K_RES,
);
/// `q8_0` decode GEMV, `dst[row] = dst[row] + dot`.
pub static GGML_MMVQ_Q8_0_RES: KernelAbiV1 = mmvq_abi(
    "ggml_mmvq_fused.q8_0_res.gfx1201",
    NUMERIC_Q8_0_RES,
    OPERANDS_Q8_0,
    DOMAIN_Q8_0,
    EXPORTS_Q8_0_RES,
);

/// Single-row `q8_1` producer with a block permutation: `x vy blk_map ne0`, explicit
/// 28 B, segment 288/8.
pub static GGML_Q8_1_GATHER: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_mmvq_fused.q8_1_gather.gfx1201",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_1_GATHER,
    params: PARAMS_GATHER,
    operands: OPERANDS_GATHER,
    fields: FIELDS_GATHER,
    hidden: HIDDEN_GATHER,
    explicit_bytes: 28,
    segment_bytes: 288,
    segment_align: 8,
    domain: DOMAIN_GATHER,
    exports: EXPORTS_GATHER,
};

/// Every contract of this family, gather producer first.
pub static ABIS: &[&KernelAbiV1] = &[
    &GGML_Q8_1_GATHER,
    &GGML_MMVQ_Q5_K,
    &GGML_MMVQ_IQ4_XS_RES,
    &GGML_MMVQ_Q4_K_RES,
    &GGML_MMVQ_Q5_K_RES,
    &GGML_MMVQ_Q6_K_RES,
    &GGML_MMVQ_Q8_0_RES,
];
