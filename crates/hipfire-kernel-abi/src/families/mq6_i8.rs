// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the exact-INT8 MQ6G256V2 prefill GEMM
//! (`crates/rdna-compute/src/gemm_mq6_i8.rs`): the PeaceMaker gfx1201 image
//! (`kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco`, five entries) and the
//! hipcc twins it is the exact twin of (gfx1201) or merely a sibling of
//! (gfx1151).
//!
//! Evidence the numbers below come from:
//!
//! * PM image: `llvm-readelf --notes` of the gfx1201 object unbundled from the
//!   committed `.hxaco`: every entry has args `A@0 Xq@8 Y@16` (`global_buffer`,
//!   8 B), `M@24 K@28 N@32 unused@36` (`by_value`, 4 B; the NR entry names the
//!   word at 36 `head_mask`), hidden words `block_count_{x,y,z}@40/44/48` (4 B),
//!   `group_size_{x,y,z}@52/54/56`, `remainder_{x,y,z}@58/60/62` (2 B),
//!   `global_offset_{x,y,z}@80/88/96` (8 B), `grid_dims@104` (2 B),
//!   `dynamic_lds_size@160` (4 B), `.kernarg_segment_size` 296 (NR: 312, with
//!   `norm@296 rope@304` `global_buffer`), `.kernarg_segment_align` 8,
//!   `.group_segment_fixed_size` 0, `.wavefront_size` 32, and
//!   `.max_flat_workgroup_size` 256 (128x128 entries) / 512 (t128x256 and NR
//!   entries). The metadata carries no `.actual_access` for these entries.
//! * hipcc twins: one object per source compiled with the runtime's flags
//!   (`--genco -O3 -fuse-cuid=none -mcode-object-version=6`, the `block_i8_128`
//!   prelude with `HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE`, `I8_TILE=256` for
//!   the `_t256` build) show the same 296/8 segment, same hidden layout, args
//!   `A Xq Y` (`A`/`Xq` `read_only`; `Y` `write_only` on gfx1151, absent on
//!   gfx12), `M K N` at 24/28/32, `.group_segment_fixed_size` 0 and
//!   `.max_flat_workgroup_size` 256 (gfx12), 512 (gfx11 128-token), 1024
//!   (gfx11 t256). hipcc contracts carry no object hash (`HipccSource`), so
//!   no drift evidence is checked against them.

use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind, HiddenV1,
    KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- formats ---------------------------------------------------------------

/// MQ6G256V2 weights: rows x K/256 groups x 200 B (8 B header + 192 B 6-bit
/// payload) (`gemm_mq6_i8.rs:40-41`, weight bytes at `:374-377`).
pub const FORMAT_MQ6G256V2_WEIGHTS: FormatId = FormatId("mq6g256v2.rows_x_groups.200b");
/// A8 `block_i8_128` activations laid out `[K/128][N]`, 136 B records
/// (`float d; int s; signed char qs[128]`) (`gemm_mq6_i8.rs:42-43`, `:446`).
pub const FORMAT_BLOCK_I8_128_KB_N: FormatId = FormatId("block_i8_128.kb_x_n.136b");
/// F32 output `[N][M]` (`gemm_mq6_i8.rs:344-345`, hipcc store
/// `gemm_mq6g256v2_mmq_i8.gfx12.hip:615`: `Y[col * M + row]`).
pub const FORMAT_F32_N_M: FormatId = FormatId("f32.n_x_m");
/// BF16 bits output `[N][M]` (`gemm_mq6_i8.rs:344-345`).
pub const FORMAT_BF16_N_M: FormatId = FormatId("bf16.n_x_m");
/// BF16 RMSNorm weight `[256]`, applied as `1 + w` (`gemm_mq6_i8.rs:511-513`,
/// `QSA_NORM_EPI_HEAD_DIM` at `:78`).
pub const FORMAT_BF16_NORM_256: FormatId = FormatId("bf16.norm_weight.256");
/// F32 RoPE table `[N][64]`: cosines of bases 0..32 then sines
/// (`gemm_mq6_i8.rs:514-515`; `peacemaker-isa mq6_i8_gemm/spec.rs:313-314`).
pub const FORMAT_F32_ROPE_N_64: FormatId = FormatId("f32.rope_table.n_x_64");

// ---- numeric contracts -------------------------------------------------------

/// SET, F32 store: exact INT8 core folded in K128 order, `y = acc`
/// (`gemm_mq6g256v2_mmq_i8.gfx12.hip:30-47`). The PM image is the exact twin of
/// the gfx1201 hipcc kernel (`gemm_mq6_i8.rs:33`: "bit-identical"), so both
/// share this id.
pub const NUMERIC_SET_F32: NumericContractId = NumericContractId("mq6_i8.a8.set_f32.v1");
/// SET, BF16 store: `isfinite(v) ? (u + 0x7FFF + ((u >> 16) & 1)) >> 16 : u >> 16`,
/// `u = bits(0.0f + y)` (`gemm_mq6g256v2_mmq_i8.gfx12.hip:42-43`, `:210-213`).
/// Shared by the PM image and the gfx1201 hipcc kernel like [`NUMERIC_SET_F32`].
pub const NUMERIC_SET_BF16_RNE: NumericContractId = NumericContractId("mq6_i8.a8.set_bf16_rne.v1");
/// SET + QSA norm/RoPE epilogue: F32 GEMM output, head-aligned 256-row tiles
/// RMS-normed, `1 + w` weighted and partially RoPE'd; byte-identical to the
/// plain GEMM followed by `indexed_attention_norm_rope_batch`
/// (`gemm_mq6_i8.rs:511-520`).
pub const NUMERIC_SET_NR_QSA: NumericContractId = NumericContractId("mq6_i8.a8.set_nr_qsa.v1");
/// gfx11 (iu8 WMMA) hipcc SET, F32 store. The source documents the same
/// integer core and fold (`gemm_mq6g256v2_mmq_i8.gfx11.hip:9-20`) but no
/// byte-identity with the gfx12 kernel, so it has its own id.
pub const NUMERIC_GFX11_SET_F32: NumericContractId =
    NumericContractId("mq6_i8.a8.gfx11.set_f32.v1");
/// gfx11 hipcc SET, BF16 store (`gemm_mq6g256v2_mmq_i8.gfx11.hip:169-176`);
/// own id like [`NUMERIC_GFX11_SET_F32`].
pub const NUMERIC_GFX11_SET_BF16_RNE: NumericContractId =
    NumericContractId("mq6_i8.a8.gfx11.set_bf16_rne.v1");

// ---- shared pieces -----------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_N: Expr = Expr::Param("N");
const P_HSR: Expr = Expr::Param("head_stride_rows");
const ONE: Expr = Expr::Const(1);

/// `gemm_mq6_i8.rs:374-377`: `m * (k / 256) * 200`.
const WEIGHT_BYTES: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256), Expr::Const(200)]);
/// `gemm_mq6_i8.rs:446`: `n * (k / 128) * 136`.
const XQ_BYTES: Expr = Expr::Mul(&[P_N, Expr::DivExact(&P_K, 128), Expr::Const(136)]);
/// `gemm_mq6_i8.rs:381`: `n * m * 4`.
const Y_F32_BYTES: Expr = Expr::Mul(&[P_N, P_M, Expr::Const(4)]);
/// `gemm_mq6_i8.rs:381`: `n * m * 2`.
const Y_BF16_BYTES: Expr = Expr::Mul(&[P_N, P_M, Expr::Const(2)]);
/// `gemm_mq6_i8.rs:543-547`: `n * 64` F32 words.
const ROPE_BYTES: Expr = Expr::Mul(&[P_N, Expr::Const(64), Expr::Const(4)]);

/// `i32::MAX` = `crate::gemm::PM_OUTPUT_EXTENT_LIMIT_BYTES`
/// (`rdna-compute/src/gemm.rs:1454`).
const I32_MAX: u64 = i32::MAX as u64;

const PARAMS_MKN: &[ParamV1] = &[
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "weight rows (Y columns)",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction length",
    },
    ParamV1 {
        name: "N",
        ty: ScalarTy::I32,
        doc: "tokens (Y rows)",
    },
];

const PARAMS_NR: &[ParamV1] = &[
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "weight rows (Y columns)",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction length",
    },
    ParamV1 {
        name: "N",
        ty: ScalarTy::I32,
        doc: "tokens (Y rows)",
    },
    ParamV1 {
        name: "head_stride_rows",
        ty: ScalarTy::I32,
        doc: "rows per head stride: 512 interleaved q-gate rows, 256 plain k rows",
    },
];

const OP_A: OperandV1 = OperandV1 {
    name: "A",
    format: FORMAT_MQ6G256V2_WEIGHTS,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(WEIGHT_BYTES),
    nullable: false,
    note: "alignment 4: the launcher checks no base alignment of A (gemm_mq6_i8.rs:374-387)",
};
const OP_XQ: OperandV1 = OperandV1 {
    name: "Xq",
    format: FORMAT_BLOCK_I8_128_KB_N,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(XQ_BYTES),
    nullable: false,
    note: "alignment 4: records start with a float; the launcher checks only non-null (gemm_mq6_i8.rs:363)",
};
const OP_Y_F32_PM: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_F32_N_M,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Y_F32_BYTES),
    nullable: false,
    note: "alignment 4 per mq6_i8_pm_admits (gemm_mq6_i8.rs:170)",
};
const OP_Y_BF16_PM: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_BF16_N_M,
    access: Access::WriteOnly,
    align: 8,
    extent: Some(Y_BF16_BYTES),
    nullable: false,
    note: "alignment 8 per mq6_i8_pm_admits (gemm_mq6_i8.rs:170): the quad stores are buffer_store_b64",
};
const OP_Y_F32_HIPCC: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_F32_N_M,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Y_F32_BYTES),
    nullable: false,
    note: "alignment = element size; the hipcc launch path checks none (gemm_mq6_i8.rs:378-387)",
};
const OP_Y_BF16_HIPCC: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_BF16_N_M,
    access: Access::WriteOnly,
    align: 2,
    extent: Some(Y_BF16_BYTES),
    nullable: false,
    note: "alignment = element size; the hipcc launch path checks none (gemm_mq6_i8.rs:378-387)",
};
const OP_NORM: OperandV1 = OperandV1 {
    name: "norm",
    format: FORMAT_BF16_NORM_256,
    access: Access::ReadOnly,
    align: 2,
    extent: Some(Expr::Const(512)),
    nullable: false,
    note: "256 BF16 words (gemm_mq6_i8.rs:543, :78)",
};
const OP_ROPE: OperandV1 = OperandV1 {
    name: "rope",
    format: FORMAT_F32_ROPE_N_64,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(ROPE_BYTES),
    nullable: false,
    note: "N rows of 64 F32 words (gemm_mq6_i8.rs:544-547)",
};

const OPERANDS_PM_F32: &[OperandV1] = &[OP_A, OP_XQ, OP_Y_F32_PM];
const OPERANDS_PM_BF16: &[OperandV1] = &[OP_A, OP_XQ, OP_Y_BF16_PM];
const OPERANDS_PM_NR: &[OperandV1] = &[OP_A, OP_XQ, OP_Y_F32_PM, OP_NORM, OP_ROPE];
const OPERANDS_HIPCC_F32: &[OperandV1] = &[OP_A, OP_XQ, OP_Y_F32_HIPCC];
const OPERANDS_HIPCC_BF16: &[OperandV1] = &[OP_A, OP_XQ, OP_Y_BF16_HIPCC];

const F_A: FieldV1 = FieldV1 {
    name: "A",
    offset: 0,
    width: 8,
    kind: FieldKind::Pointer { operand: "A" },
};
const F_XQ: FieldV1 = FieldV1 {
    name: "Xq",
    offset: 8,
    width: 8,
    kind: FieldKind::Pointer { operand: "Xq" },
};
const F_Y: FieldV1 = FieldV1 {
    name: "Y",
    offset: 16,
    width: 8,
    kind: FieldKind::Pointer { operand: "Y" },
};
const F_M: FieldV1 = FieldV1 {
    name: "M",
    offset: 24,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_M,
    },
};
const F_K: FieldV1 = FieldV1 {
    name: "K",
    offset: 28,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};
const F_N: FieldV1 = FieldV1 {
    name: "N",
    offset: 32,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_N,
    },
};
/// The PM image keeps K1's FullSet layout: a seventh by-value word `unused` at
/// 36 (`gemm_mq6_i8.rs:440-445`, `kernels/iu4_k1.rs Variant::kernargs`).
const F_UNUSED: FieldV1 = FieldV1 {
    name: "unused",
    offset: 36,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: Expr::Const(0),
    },
};
/// NR entry: the word at 36 is `head_mask = head_stride_rows - 1`
/// (`gemm_mq6_i8.rs:582`, `:619`; `mq6_i8_gemm/spec.rs:281-285`).
const F_HEAD_MASK: FieldV1 = FieldV1 {
    name: "head_mask",
    offset: 36,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: Expr::Sub(&P_HSR, &ONE),
    },
};
/// `gemm_mq6_i8.rs:73`, `:621-625`; `mq6_i8_gemm/spec.rs:315`.
const F_NORM: FieldV1 = FieldV1 {
    name: "norm",
    offset: 296,
    width: 8,
    kind: FieldKind::Pointer { operand: "norm" },
};
/// `mq6_i8_gemm/spec.rs:316`.
const F_ROPE: FieldV1 = FieldV1 {
    name: "rope",
    offset: 304,
    width: 8,
    kind: FieldKind::Pointer { operand: "rope" },
};

const FIELDS_PM_SET: &[FieldV1] = &[F_A, F_XQ, F_Y, F_M, F_K, F_N, F_UNUSED];
const FIELDS_PM_NR: &[FieldV1] = &[F_A, F_XQ, F_Y, F_M, F_K, F_N, F_HEAD_MASK, F_NORM, F_ROPE];
const FIELDS_HIPCC: &[FieldV1] = &[F_A, F_XQ, F_Y, F_M, F_K, F_N];

/// Hidden block of every entry, exactly the object metadata's offsets/widths.
const HIDDEN: &[HiddenV1] = &[
    HiddenV1 {
        offset: 40,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::X),
    },
    HiddenV1 {
        offset: 44,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 48,
        width: 4,
        kind: HiddenKind::BlockCount(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 52,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::X),
    },
    HiddenV1 {
        offset: 54,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 56,
        width: 2,
        kind: HiddenKind::GroupSize(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 58,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::X),
    },
    HiddenV1 {
        offset: 60,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 62,
        width: 2,
        kind: HiddenKind::Remainder(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 80,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::X),
    },
    HiddenV1 {
        offset: 88,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::Y),
    },
    HiddenV1 {
        offset: 96,
        width: 8,
        kind: HiddenKind::GlobalOffset(crate::Axis::Z),
    },
    HiddenV1 {
        offset: 104,
        width: 2,
        kind: HiddenKind::GridDims,
    },
    HiddenV1 {
        offset: 160,
        width: 4,
        kind: HiddenKind::DynamicLdsSize,
    },
];

// ---- domains -------------------------------------------------------------------

/// `mq6_i8_pm_admits` (`gemm_mq6_i8.rs:154-176`), F32 output. The `y` pointer
/// alignment is an operand alignment; the flag/arch tests are export identity.
const DOMAIN_PM_F32: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "gemm_mq6_i8.rs:165 k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "gemm_mq6_i8.rs:166 MQ6 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "gemm_mq6_i8.rs:167 m > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_M, 4),
        why: "gemm_mq6_i8.rs:168 row quads are stored whole",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_mq6_i8.rs:169 n > 0",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(Y_F32_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:172 output extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(WEIGHT_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:173-175 weight extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
];

/// As [`DOMAIN_PM_F32`] with the 2-byte output extent.
const DOMAIN_PM_BF16: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "gemm_mq6_i8.rs:165 k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "gemm_mq6_i8.rs:166 MQ6 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "gemm_mq6_i8.rs:167 m > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_M, 4),
        why: "gemm_mq6_i8.rs:168 row quads are stored whole",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_mq6_i8.rs:169 n > 0",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(Y_BF16_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:172 output extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(WEIGHT_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:173-175 weight extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
];

/// [`DOMAIN_PM_F32`] plus `gemm_mq6g256v2_i8_norm_rope_applies`
/// (`gemm_mq6_i8.rs:496-498`). NOT expressible: `m % head_stride_rows == 0`
/// (`:497`; given `m % 256 == 0` it only bites for `head_stride_rows == 512`
/// with `m % 512 != 0`); the launcher keeps checking it.
const DOMAIN_PM_NR: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "gemm_mq6_i8.rs:165 k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "gemm_mq6_i8.rs:166 MQ6 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "gemm_mq6_i8.rs:167 m > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_M, 4),
        why: "gemm_mq6_i8.rs:168 row quads are stored whole",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_M, 256),
        why: "gemm_mq6_i8.rs:498 the entry's CTA covers 256 rows",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_mq6_i8.rs:169 n > 0",
    },
    DomainRuleV1 {
        rule: Rule::OneOf(P_HSR, &[256, 512]),
        why: "gemm_mq6_i8.rs:496 head_mask = head_stride_rows - 1 selects 256 or 512-row strides",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX),
        why: "gemm_mq6_i8.rs:171 32-bit address contract",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(Y_F32_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:172 output extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(WEIGHT_BYTES, I32_MAX),
        why: "gemm_mq6_i8.rs:173-175 weight extent <= PM_OUTPUT_EXTENT_LIMIT_BYTES",
    },
];

/// hipcc route (`gemm_mq6_i8.rs:363`): `m, n, k > 0`, `k % 256 == 0`; the
/// kernargs are `i32` (`:429-431`), so dims are bounded by the I32 param type.
/// The hipcc path has no 2^31-byte extent rule and none is imposed here.
const DOMAIN_HIPCC: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "gemm_mq6_i8.rs:363 k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "gemm_mq6_i8.rs:363 MQ6 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_M),
        why: "gemm_mq6_i8.rs:363 m > 0",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_mq6_i8.rs:363 n > 0",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_M, I32_MAX),
        why: "gemm_mq6_i8.rs:429 m passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "gemm_mq6_i8.rs:430 k passed as i32",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX),
        why: "gemm_mq6_i8.rs:431 n passed as i32",
    },
];

// ---- exports -------------------------------------------------------------------

const PM_OBJECT: ObjectRefV1 = ObjectRefV1::Embedded {
    // `gemm_mq6_i8.rs:58` (MQ6_I8_PM_MODULE)
    module: "gemm_mq6g256v2_mmq_i8_pm_gfx1201",
    artifact: "kernels/gemm_mq6g256v2_mmq_i8_pm_gfx1201.hxaco",
    // `sha256sum` of the committed object. (The doc comment of
    // `kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201` quotes an older digest.)
    sha256: "c1458584a1b8ab53e613c982c9a3e87c530c80e83e76223ed4f677d594bad300",
};

/// 128 x 128 grid: `[ceil(M/128), ceil(N/128), 1]` (`gemm_mq6_i8.rs:451-454`
/// with `tile_rows = tile_tokens = 128`).
const GRID_128X128: [Expr; 3] = [
    Expr::DivCeil(&P_M, 128),
    Expr::DivCeil(&P_N, 128),
    Expr::Const(1),
];
/// 128 x 256 grid: `[ceil(M/128), ceil(N/256), 1]` (`gemm_mq6_i8.rs:451-454`).
const GRID_128X256: [Expr; 3] = [
    Expr::DivCeil(&P_M, 128),
    Expr::DivCeil(&P_N, 256),
    Expr::Const(1),
];
/// NR grid: `[M/256, ceil(N/128), 1]` (`gemm_mq6_i8.rs:603-607`).
const GRID_NR: [Expr; 3] = [
    Expr::DivExact(&P_M, 256),
    Expr::DivCeil(&P_N, 128),
    Expr::Const(1),
];
/// gfx11 t256 grid: `[ceil(M/256), ceil(N/256), 1]`
/// (`gemm_mq6g256v2_mmq_i8.gfx11.hip:69` with `I8_TILE = 256`).
const GRID_256X256: [Expr; 3] = [
    Expr::DivCeil(&P_M, 256),
    Expr::DivCeil(&P_N, 256),
    Expr::Const(1),
];

const CONSTS_128X128: &[(&str, u64)] = &[("tile_rows", 128), ("tile_tokens", 128)];
const CONSTS_128X256: &[(&str, u64)] = &[("tile_rows", 128), ("tile_tokens", 256)];
const CONSTS_NR: &[(&str, u64)] = &[("tile_rows", 256), ("tile_tokens", 128)];
const CONSTS_256X256: &[(&str, u64)] = &[("tile_rows", 256), ("tile_tokens", 256)];

/// PM SET F32 entries (`gemm_mq6_i8.rs:59`, `:62`; launch `:122-139`):
/// 128 x 128 (block 256, LDS 36864 = `MQ6_I8_GFX12_LDS_BYTES` `:51`) and
/// t128x256 (block 512, LDS 55296 `:86-87`). `.max_flat_workgroup_size` 256 / 512.
const EXPORTS_PM_SET_F32: &[ExportV1] = &[
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201",
        arch: Arch::Gfx1201,
        object: PM_OBJECT,
        block: [256, 1, 1],
        dynamic_lds: 36864,
        group_segment_fixed: 0,
        grid: GRID_128X128,
        consts: CONSTS_128X128,
    },
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256",
        arch: Arch::Gfx1201,
        object: PM_OBJECT,
        block: [512, 1, 1],
        dynamic_lds: 55296,
        group_segment_fixed: 0,
        grid: GRID_128X256,
        consts: CONSTS_128X256,
    },
];

/// PM SET BF16 entries (`gemm_mq6_i8.rs:60`, `:63`), same launch geometry as F32.
const EXPORTS_PM_SET_BF16: &[ExportV1] = &[
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201",
        arch: Arch::Gfx1201,
        object: PM_OBJECT,
        block: [256, 1, 1],
        dynamic_lds: 36864,
        group_segment_fixed: 0,
        grid: GRID_128X128,
        consts: CONSTS_128X128,
    },
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201_t128x256",
        arch: Arch::Gfx1201,
        object: PM_OBJECT,
        block: [512, 1, 1],
        dynamic_lds: 55296,
        group_segment_fixed: 0,
        grid: GRID_128X256,
        consts: CONSTS_128X256,
    },
];

/// PM NR entry (`gemm_mq6_i8.rs:67-70`, launch `:601-610`): block 512, LDS 55296.
const EXPORTS_PM_NR: &[ExportV1] = &[ExportV1 {
    symbol: "gemm_mq6g256v2_mmq_i8_pm_set_nr_gfx1201_t256x128",
    arch: Arch::Gfx1201,
    object: PM_OBJECT,
    block: [512, 1, 1],
    dynamic_lds: 55296,
    group_segment_fixed: 0,
    grid: GRID_NR,
    consts: CONSTS_NR,
}];

const HIPCC_GFX12_OBJECT_F32: ObjectRefV1 = ObjectRefV1::HipccSource {
    // `gemm_mq6_i8.rs:211-212`
    module: "gemm_mq6g256v2_mmq_i8_gfx12",
    source: "kernels/src/gemm_mq6g256v2_mmq_i8.gfx12.hip",
};
const HIPCC_GFX11_OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    // `gemm_mq6_i8.rs:225-226`
    module: "gemm_mq6g256v2_mmq_i8_gfx11",
    source: "kernels/src/gemm_mq6g256v2_mmq_i8.gfx11.hip",
};
const HIPCC_GFX11_T256_OBJECT: ObjectRefV1 = ObjectRefV1::HipccSource {
    // `gemm_mq6_i8.rs:218-219`; the source is built with `#define I8_TILE 256`.
    module: "gemm_mq6g256v2_mmq_i8_gfx11_t256",
    source: "kernels/src/gemm_mq6g256v2_mmq_i8.gfx11.hip",
};

/// gfx1201 hipcc SET F32 (`gemm_mq6_i8.rs:238`; block 256, LDS 36864 `:213-215`).
const EXPORTS_HIPCC_GFX12_F32: &[ExportV1] = &[ExportV1 {
    symbol: "gemm_mq6g256v2_mmq_i8_set_gfx12",
    arch: Arch::Gfx1201,
    object: HIPCC_GFX12_OBJECT_F32,
    block: [256, 1, 1],
    dynamic_lds: 36864,
    group_segment_fixed: 0,
    grid: GRID_128X128,
    consts: CONSTS_128X128,
}];
/// gfx1201 hipcc SET BF16 (`gemm_mq6_i8.rs:239`).
const EXPORTS_HIPCC_GFX12_BF16: &[ExportV1] = &[ExportV1 {
    symbol: "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12",
    arch: Arch::Gfx1201,
    object: HIPCC_GFX12_OBJECT_F32,
    block: [256, 1, 1],
    dynamic_lds: 36864,
    group_segment_fixed: 0,
    grid: GRID_128X128,
    consts: CONSTS_128X128,
}];
/// gfx1151 hipcc SET F32: 128-token build (block 512, LDS 32768,
/// `gemm_mq6_i8.rs:224-229`, `:242`) and t256 build (block 1024, LDS 65536,
/// `:217-223`, `:240`).
const EXPORTS_HIPCC_GFX11_F32: &[ExportV1] = &[
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_set_gfx11",
        arch: Arch::Gfx1151,
        object: HIPCC_GFX11_OBJECT,
        block: [512, 1, 1],
        dynamic_lds: 32768,
        group_segment_fixed: 0,
        grid: GRID_128X128,
        consts: CONSTS_128X128,
    },
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_set_gfx11_t256",
        arch: Arch::Gfx1151,
        object: HIPCC_GFX11_T256_OBJECT,
        block: [1024, 1, 1],
        dynamic_lds: 65536,
        group_segment_fixed: 0,
        grid: GRID_256X256,
        consts: CONSTS_256X256,
    },
];
/// gfx1151 hipcc SET BF16 (`gemm_mq6_i8.rs:241`, `:243`).
const EXPORTS_HIPCC_GFX11_BF16: &[ExportV1] = &[
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11",
        arch: Arch::Gfx1151,
        object: HIPCC_GFX11_OBJECT,
        block: [512, 1, 1],
        dynamic_lds: 32768,
        group_segment_fixed: 0,
        grid: GRID_128X128,
        consts: CONSTS_128X128,
    },
    ExportV1 {
        symbol: "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256",
        arch: Arch::Gfx1151,
        object: HIPCC_GFX11_T256_OBJECT,
        block: [1024, 1, 1],
        dynamic_lds: 65536,
        group_segment_fixed: 0,
        grid: GRID_256X256,
        consts: CONSTS_256X256,
    },
];

// ---- contracts -----------------------------------------------------------------

/// PM image, F32 SET: `A Xq Y M K N unused`, explicit prefix 40 B
/// (`gemm_mq6_i8.rs:440-445`, `:459-469`), segment 296/8.
pub static MQ6_I8_PM_SET_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.pm.set_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SET_F32,
    params: PARAMS_MKN,
    operands: OPERANDS_PM_F32,
    fields: FIELDS_PM_SET,
    hidden: HIDDEN,
    explicit_bytes: 40,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_PM_F32,
    exports: EXPORTS_PM_SET_F32,
};

/// PM image, BF16 SET (`Y` 8-byte aligned).
pub static MQ6_I8_PM_SET_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.pm.set_bf16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SET_BF16_RNE,
    params: PARAMS_MKN,
    operands: OPERANDS_PM_BF16,
    fields: FIELDS_PM_SET,
    hidden: HIDDEN,
    explicit_bytes: 40,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_PM_BF16,
    exports: EXPORTS_PM_SET_BF16,
};

/// PM image, SET + QSA norm/RoPE epilogue: explicit prefix 40 B, hidden block
/// 40..296 (runtime's), then `norm@296 rope@304`; 312 B (`gemm_mq6_i8.rs:612-626`).
pub static MQ6_I8_PM_SET_NR: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.pm.set_nr",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SET_NR_QSA,
    params: PARAMS_NR,
    operands: OPERANDS_PM_NR,
    fields: FIELDS_PM_NR,
    hidden: HIDDEN,
    explicit_bytes: 312,
    segment_bytes: 312,
    segment_align: 8,
    domain: DOMAIN_PM_NR,
    exports: EXPORTS_PM_NR,
};

/// hipcc gfx1201 SET F32: `A Xq Y M K N` through ordinary kernelParams
/// (`gemm_mq6_i8.rs:432-439`, blob `:459-465`: 36 B), segment 296/8. Shares
/// [`NUMERIC_SET_F32`] with the PM image (its documented exact twin).
pub static MQ6_I8_HIPCC_SET_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.hipcc.gfx12.set_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SET_F32,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_F32,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_HIPCC_GFX12_F32,
};

/// hipcc gfx1201 SET BF16; shares [`NUMERIC_SET_BF16_RNE`] with the PM image.
pub static MQ6_I8_HIPCC_SET_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.hipcc.gfx12.set_bf16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_SET_BF16_RNE,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_BF16,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_HIPCC_GFX12_BF16,
};

/// hipcc gfx1151 SET F32 (128-token and t256 builds); own numeric id: the
/// gfx11 source is not documented byte-identical to the gfx12 kernel.
pub static MQ6_I8_HIPCC_GFX11_SET_F32: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.hipcc.gfx11.set_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GFX11_SET_F32,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_F32,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_HIPCC_GFX11_F32,
};

/// hipcc gfx1151 SET BF16 (128-token and t256 builds).
pub static MQ6_I8_HIPCC_GFX11_SET_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "mq6_i8_gemm.hipcc.gfx11.set_bf16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GFX11_SET_BF16_RNE,
    params: PARAMS_MKN,
    operands: OPERANDS_HIPCC_BF16,
    fields: FIELDS_HIPCC,
    hidden: HIDDEN,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_HIPCC,
    exports: EXPORTS_HIPCC_GFX11_BF16,
};

/// Every contract of this family, PM first.
pub static ABIS: &[&KernelAbiV1] = &[
    &MQ6_I8_PM_SET_F32,
    &MQ6_I8_PM_SET_BF16,
    &MQ6_I8_PM_SET_NR,
    &MQ6_I8_HIPCC_SET_F32,
    &MQ6_I8_HIPCC_SET_BF16,
    &MQ6_I8_HIPCC_GFX11_SET_F32,
    &MQ6_I8_HIPCC_GFX11_SET_BF16,
];
