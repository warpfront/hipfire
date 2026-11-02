// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the GGUF-native convert shim for gfx1201
//! (`kernels/src/ggml_convert_shim.hip`, object `kernels/ggml_convert_shim_gfx1201.co`,
//! built by `scripts/build-ggml-convert-shim.sh`): block -> f16 / f32 (optionally bf16-rounded)
//! dequantisation with an optional 128-element column-chunk permutation, the embedding row gather
//! (token ids from memory, and one token by value) and the in-place A8 activation round trip, for
//! `iq4_xs`, `q4_k`, `q5_k`, `q6_k` and `q8_0`, transcribed from the pinned upstream snapshot in
//! `third_party/ggml/`. 21 exports, one contract each.
//!
//! Evidence the numbers below come from:
//!
//! * Object: `llvm-readelf --notes` of the gfx1201 ELF unbundled with
//!   `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201` from the committed
//!   `.co` (63544 B, `--no-offload-compress` bundle). Every entry: `.kernarg_segment_align` 8,
//!   `.group_segment_fixed_size` 0 (no LDS), `.private_segment_fixed_size` 0, `.wavefront_size` 32, NO
//!   hidden kernel arguments (the shim uses constant work-group sizes, so the kernarg segment is exactly
//!   the explicit arguments): dequant f16 36 B (`vx@0 dst@8 colmap@16` buffers, `nb@24 K@28 chunk@32`
//!   by value), dequant f32 40 B (`+ bf16_round@36`), gather 28 B (`vx@0 tok_ids@8 dst@16` buffers,
//!   `K@24`), by-value gather 28 B (`vx@0` buffer, `tok@8` by value, `dst@16` buffer, `K@24`), round
//!   trip 12 B (`x@0` buffer, `ne@8`). The metadata names no argument; the field names are the shim
//!   header's. `.max_flat_workgroup_size` = the launch block (`__launch_bounds__`).
//! * Launch geometry (shim header): dequant blocks `iq4_xs`/`q4_k` 32, `q5_k`/`q6_k` 64, `q8_0` 256
//!   threads; grid x = `nb` super-blocks (k/i-quants) or `ceil(nb * 32 / 512)` (`q8_0`). Gather: same
//!   blocks, grid `(K/256 | ceil(K/512), n_tokens, 1)`. Round trip: block 256, grid `ceil(ne/256)`.
//! * Weight block sizes (`ggml-common.h`): `block_iq4_xs` 136 B / 256, `block_q4_K` 144 / 256,
//!   `block_q5_K` 176 / 256, `block_q6_K` 210 / 256, `block_q8_0` 34 / 32. Base alignment = `alignof`
//!   of the block struct, static-asserted in the shim: iq4_xs 4 (its `qs` is read as 32-bit words for the
//!   register table select, offset 8 of a 136 B block), q4_k 4, q5_k 4 (`half2 dm`), q6_k 2, q8_0 2.
//! * Address math (domain rules): the dequant index math is 32-bit, so `M * K <= i32::MAX`;
//!   `colmap` additionally needs `K % chunk == 0` and `chunk % 128 == 0` (chunk is fixed to 128 by the
//!   contract) because each thread's writes stay inside one 128-element window of its block; the
//!   gather computes `token * row_bytes` and `out_row * K` in 64 bits, `K <= i32::MAX`; the token is
//!   not checked on the device (as upstream), the contract states `TOK < ROWS` for the by-value
//!   launcher and the extent of the table for the id launcher. Grid y of the gather is the token count
//!   (`<= 65535`, the HIP y-grid limit).
//! * Numeric: every output element is one evaluation of upstream's per-element expression
//!   (`d * scale * code`, or `d * code - m`), all products exact in f32, followed by the destination
//!   conversion (f16 round-to-nearest-even, or bf16 round-to-nearest-even with NaN quietened as
//!   `ggml_compute_fp32_to_bf16`, widened back to f32); no reduction, no atomics, independent of the
//!   launch geometry. The round trip is the S1 producer's quantiser followed by `f32(half(d)) * q`.

use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, KernelAbiV1,
    NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_mmvq::{FORMAT_IQ4_XS_136, FORMAT_Q4_K_144, FORMAT_Q6_K_210, FORMAT_Q8_0_34};

// ---- formats ---------------------------------------------------------------

/// `block_q5_K` rows: `M x K/256` blocks of 176 B.
pub const FORMAT_Q5_K_176: FormatId = FormatId("ggml.q5_k.176");
/// F16 flat dequantised tensor `[M * K]`, row-major.
pub const FORMAT_F16_FLAT: FormatId = FormatId("f16.flat");
/// F32 flat tensor `[N]`, row-major (dequantised weights, or the A8 activation plane).
pub const FORMAT_F32_FLAT: FormatId = FormatId("f32.flat");
/// int32 `[K / 128]` destination-chunk permutation (`colmap[g]` = destination chunk of source chunk `g`).
pub const FORMAT_I32_CHUNK_PERM: FormatId = FormatId("i32.chunk_perm");
/// int32 `[N]` token ids.
pub const FORMAT_I32_TOKEN_IDS: FormatId = FormatId("i32.token_ids");
/// F32 gathered rows `[N, K]`, row-major.
pub const FORMAT_F32_ROWS_NK: FormatId = FormatId("f32.rows_n_k");

// ---- shared pieces -----------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_COLMAP: Expr = Expr::Param("COLMAP");
const P_BF16: Expr = Expr::Param("BF16");
const P_ROWS: Expr = Expr::Param("ROWS");
const P_N: Expr = Expr::Param("N");
const P_TOK: Expr = Expr::Param("TOK");
const P_NE: Expr = Expr::Param("NE");
const ONE: Expr = Expr::Const(1);
const I32_MAX: u64 = i32::MAX as u64;
/// Largest `ne` of the round trip whose `256 * blockIdx.x + threadIdx.x` index stays in `int`
/// (`ne <= INT32_MAX - 255`, rounded down to the 32-element multiple).
const ROUNDTRIP_MAX_NE: u64 = 2_147_483_392;

/// Elements of the dequantised tensor: `M * K`.
const MK: Expr = Expr::Mul(&[P_M, P_K]);
const MK_F16_BYTES: Expr = Expr::Mul(&[P_M, P_K, Expr::Const(2)]);
const MK_F32_BYTES: Expr = Expr::Mul(&[P_M, P_K, Expr::Const(4)]);
/// `colmap` extent: `K / 128` int32 when present (`COLMAP = 1`), 0 bytes otherwise.
const COLMAP_BYTES: Expr = Expr::Mul(&[P_COLMAP, Expr::Div(&P_K, 128), Expr::Const(4)]);
/// `K * COLMAP`: the chunk-permutation rows need `K % 128 == 0` only when a colmap is passed.
const K_IF_COLMAP: Expr = Expr::Mul(&[P_K, P_COLMAP]);
/// `TOK + 1` (the by-value token is a row index: `TOK < ROWS`).
const TOK_PLUS_ONE: Expr = Expr::Add(&[P_TOK, ONE]);

const PARAMS_DQ_F16: &[ParamV1] = &[
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "weight rows",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "row length in elements",
    },
    ParamV1 {
        name: "COLMAP",
        ty: ScalarTy::I32,
        doc: "1 when a column-chunk permutation is passed, 0 for a null colmap",
    },
];
const PARAMS_DQ_F32: &[ParamV1] = &[
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "weight rows",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "row length in elements",
    },
    ParamV1 {
        name: "COLMAP",
        ty: ScalarTy::I32,
        doc: "1 when a column-chunk permutation is passed, 0 for a null colmap",
    },
    ParamV1 {
        name: "BF16",
        ty: ScalarTy::I32,
        doc: "1: round the f32 result to bfloat16 (RNE) and store it widened; 0: plain f32",
    },
];
const PARAMS_ROW: &[ParamV1] = &[
    ParamV1 {
        name: "ROWS",
        ty: ScalarTy::I32,
        doc: "rows of the weight (embedding) table",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "row length in elements",
    },
    ParamV1 {
        name: "N",
        ty: ScalarTy::I32,
        doc: "tokens gathered (grid y)",
    },
];
const PARAMS_ROW1: &[ParamV1] = &[
    ParamV1 {
        name: "ROWS",
        ty: ScalarTy::I32,
        doc: "rows of the weight (embedding) table",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "row length in elements",
    },
    ParamV1 {
        name: "TOK",
        ty: ScalarTy::I32,
        doc: "token id (row of the table), by value",
    },
];
const PARAMS_ROUNDTRIP: &[ParamV1] = &[ParamV1 {
    name: "NE",
    ty: ScalarTy::I32,
    doc: "f32 elements, multiple of 32",
}];

const OP_COLMAP: OperandV1 = OperandV1 {
    name: "colmap",
    format: FORMAT_I32_CHUNK_PERM,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(COLMAP_BYTES),
    nullable: true,
    note: "null = identity layout (COLMAP = 0, extent 0); otherwise K/128 int32, a permutation of 0..K/128-1",
};
const OP_DST_F16: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F16_FLAT,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(MK_F16_BYTES),
    nullable: false,
    note: "alignment 4 (the q8_0 path stores element pairs); every element is written exactly once",
};
const OP_DST_F32: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F32_FLAT,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(MK_F32_BYTES),
    nullable: false,
    note: "alignment 4; every element is written exactly once",
};
const OP_TOK_IDS: OperandV1 = OperandV1 {
    name: "tok_ids",
    format: FORMAT_I32_TOKEN_IDS,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(Expr::Mul(&[P_N, Expr::Const(4)])),
    nullable: false,
    note: "token values are not range-checked on the device (as upstream): each must be < ROWS, the loader/caller's contract",
};
const OP_ROWS_OUT: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F32_ROWS_NK,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
    nullable: false,
    note: "alignment 4; row i holds the dequantised table row tok_ids[i]",
};
const OP_ROW1_OUT: OperandV1 = OperandV1 {
    name: "dst",
    format: FORMAT_F32_ROWS_NK,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Expr::Mul(&[P_K, Expr::Const(4)])),
    nullable: false,
    note: "alignment 4; one dequantised table row",
};

// ---- fields --------------------------------------------------------------------

const F_VX: FieldV1 = FieldV1 {
    name: "vx",
    offset: 0,
    width: 8,
    kind: FieldKind::Pointer { operand: "vx" },
};
const F_DST8: FieldV1 = FieldV1 {
    name: "dst",
    offset: 8,
    width: 8,
    kind: FieldKind::Pointer { operand: "dst" },
};
const F_COLMAP: FieldV1 = FieldV1 {
    name: "colmap",
    offset: 16,
    width: 8,
    kind: FieldKind::Pointer { operand: "colmap" },
};
const F_K28: FieldV1 = FieldV1 {
    name: "K",
    offset: 28,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_K,
    },
};
const F_CHUNK: FieldV1 = FieldV1 {
    name: "chunk",
    offset: 32,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: Expr::Const(128),
    },
};
const F_BF16: FieldV1 = FieldV1 {
    name: "bf16_round",
    offset: 36,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_BF16,
    },
};
const F_TOK_IDS: FieldV1 = FieldV1 {
    name: "tok_ids",
    offset: 8,
    width: 8,
    kind: FieldKind::Pointer { operand: "tok_ids" },
};
const F_TOK: FieldV1 = FieldV1 {
    name: "tok",
    offset: 8,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_TOK,
    },
};
const F_DST16: FieldV1 = FieldV1 {
    name: "dst",
    offset: 16,
    width: 8,
    kind: FieldKind::Pointer { operand: "dst" },
};
const F_K24: FieldV1 = FieldV1 {
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
const F_NE: FieldV1 = FieldV1 {
    name: "ne",
    offset: 8,
    width: 4,
    kind: FieldKind::Scalar {
        ty: ScalarTy::I32,
        value: P_NE,
    },
};

// ---- object ----------------------------------------------------------------------

const OBJECT: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "ggml_convert_shim_gfx1201",
    artifact: "kernels/ggml_convert_shim_gfx1201.co",
    // `sha256sum` of the committed offload bundle (build is deterministic).
    sha256: "50510764a546a3552d43278e9d7e3cd8b37aeb6554eb8a79bb8888bb0d613074",
};

// ---- per-type contracts ----------------------------------------------------------

/// Contracts of one weight type. `$qk` elements and `$bb` bytes per block, `$threads` the launch block
/// of the dequant and gather exports, `$dq_grid` / `$row_grid` the dequant and gather grid x over the
/// params, `$valign` the base alignment of the weight operand.
macro_rules! convert_type {
    ($tag:literal, $qk:literal, $bb:literal, $fmt:expr, $valign:literal, $threads:literal,
     $dq_grid:expr, $row_grid:expr) => {
        use super::*;

        /// Weight blocks of the dequantised tensor: `M * (K / qk)`.
        const NB: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, $qk)]);
        const VX_BYTES_DQ: Expr = Expr::Mul(&[P_M, Expr::DivExact(&P_K, $qk), Expr::Const($bb)]);
        const VX_BYTES_ROW: Expr =
            Expr::Mul(&[P_ROWS, Expr::DivExact(&P_K, $qk), Expr::Const($bb)]);

        const OP_VX_DQ: OperandV1 = OperandV1 {
            name: "vx",
            format: $fmt,
            access: Access::ReadOnly,
            align: $valign,
            extent: Some(VX_BYTES_DQ),
            nullable: false,
            note: "base alignment = alignof(block struct), static-asserted in the shim",
        };
        const OP_VX_ROW: OperandV1 = OperandV1 {
            name: "vx",
            format: $fmt,
            access: Access::ReadOnly,
            align: $valign,
            extent: Some(VX_BYTES_ROW),
            nullable: false,
            note:
                "the whole table; the rows touched are the token ids' (not checked on the device)",
        };

        const F_NB: FieldV1 = FieldV1 {
            name: "nb",
            offset: 24,
            width: 4,
            kind: FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: NB,
            },
        };

        const OPERANDS_DQ_F16: &[OperandV1] = &[OP_VX_DQ, OP_DST_F16, OP_COLMAP];
        const OPERANDS_DQ_F32: &[OperandV1] = &[OP_VX_DQ, OP_DST_F32, OP_COLMAP];
        const OPERANDS_ROW: &[OperandV1] = &[OP_VX_ROW, OP_TOK_IDS, OP_ROWS_OUT];
        const OPERANDS_ROW1: &[OperandV1] = &[OP_VX_ROW, OP_ROW1_OUT];

        const FIELDS_DQ_F16: &[FieldV1] = &[F_VX, F_DST8, F_COLMAP, F_NB, F_K28, F_CHUNK];
        const FIELDS_DQ_F32: &[FieldV1] = &[F_VX, F_DST8, F_COLMAP, F_NB, F_K28, F_CHUNK, F_BF16];
        const FIELDS_ROW: &[FieldV1] = &[F_VX, F_TOK_IDS, F_DST16, F_K24];
        const FIELDS_ROW1: &[FieldV1] = &[F_VX, F_TOK, F_DST16, F_K24];

        const DOMAIN_DQ_COMMON: [DomainRuleV1; 7] = [
            DomainRuleV1 {
                rule: Rule::Positive(P_K),
                why: "K > 0",
            },
            DomainRuleV1 {
                rule: Rule::MultipleOf(P_K, $qk),
                why: "K % qk == 0 (whole weight blocks per row)",
            },
            DomainRuleV1 {
                rule: Rule::Positive(P_M),
                why: "M >= 1",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(MK, I32_MAX),
                why: "block and element indices are computed in 32 bits (nb * qk <= INT32_MAX)",
            },
            DomainRuleV1 {
                rule: Rule::OneOf(P_COLMAP, &[0, 1]),
                why: "COLMAP is a presence flag",
            },
            DomainRuleV1 {
                rule: Rule::MultipleOf(K_IF_COLMAP, 128),
                why: "a colmap permutes whole 128-element chunks of a row: K % 128 == 0",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_K, I32_MAX),
                why: "K is an i32 kernarg",
            },
        ];
        const DOMAIN_DQ_F16: &[DomainRuleV1] = &DOMAIN_DQ_COMMON;
        const DOMAIN_DQ_F32: &[DomainRuleV1] = &[
            DOMAIN_DQ_COMMON[0],
            DOMAIN_DQ_COMMON[1],
            DOMAIN_DQ_COMMON[2],
            DOMAIN_DQ_COMMON[3],
            DOMAIN_DQ_COMMON[4],
            DOMAIN_DQ_COMMON[5],
            DOMAIN_DQ_COMMON[6],
            DomainRuleV1 {
                rule: Rule::OneOf(P_BF16, &[0, 1]),
                why: "bf16_round is a flag",
            },
        ];
        const DOMAIN_ROW: &[DomainRuleV1] = &[
            DomainRuleV1 {
                rule: Rule::Positive(P_K),
                why: "K > 0",
            },
            DomainRuleV1 {
                rule: Rule::MultipleOf(P_K, $qk),
                why: "K % qk == 0 (whole weight blocks per row)",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_K, I32_MAX),
                why: "K is an i32 kernarg",
            },
            DomainRuleV1 {
                rule: Rule::Positive(P_ROWS),
                why: "ROWS >= 1",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_ROWS, I32_MAX),
                why: "token ids are int32 row indices",
            },
            DomainRuleV1 {
                rule: Rule::Positive(P_N),
                why: "N >= 1 (grid y)",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_N, 65535),
                why: "tokens ride grid y (HIP limit 65535)",
            },
        ];
        const DOMAIN_ROW1: &[DomainRuleV1] = &[
            DomainRuleV1 {
                rule: Rule::Positive(P_K),
                why: "K > 0",
            },
            DomainRuleV1 {
                rule: Rule::MultipleOf(P_K, $qk),
                why: "K % qk == 0 (whole weight blocks per row)",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_K, I32_MAX),
                why: "K is an i32 kernarg",
            },
            DomainRuleV1 {
                rule: Rule::Positive(P_ROWS),
                why: "ROWS >= 1",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(P_ROWS, I32_MAX),
                why: "token ids are int32 row indices",
            },
            DomainRuleV1 {
                rule: Rule::AtLeast(Expr::Sub(&P_ROWS, &TOK_PLUS_ONE), 0),
                why: "TOK < ROWS (a negative ROWS - (TOK + 1) is an expression error)",
            },
        ];

        const GRID_DQ: [Expr; 3] = [$dq_grid, Expr::Const(1), Expr::Const(1)];
        const GRID_ROW: [Expr; 3] = [$row_grid, P_N, Expr::Const(1)];
        const GRID_ROW1: [Expr; 3] = [$row_grid, Expr::Const(1), Expr::Const(1)];

        const EXPORTS_DQ_F16: &[ExportV1] = &[ExportV1 {
            symbol: concat!("ggml_dequant_", $tag, "_f16_gfx1201"),
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [$threads, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_DQ,
            consts: &[],
        }];
        const EXPORTS_DQ_F32: &[ExportV1] = &[ExportV1 {
            symbol: concat!("ggml_dequant_", $tag, "_f32_gfx1201"),
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [$threads, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_DQ,
            consts: &[],
        }];
        const EXPORTS_ROW: &[ExportV1] = &[ExportV1 {
            symbol: concat!("ggml_get_row_", $tag, "_f32_gfx1201"),
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [$threads, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_ROW,
            consts: &[],
        }];
        const EXPORTS_ROW1: &[ExportV1] = &[ExportV1 {
            symbol: concat!("ggml_get_row1_", $tag, "_f32_gfx1201"),
            arch: Arch::Gfx1201,
            object: OBJECT,
            block: [$threads, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_ROW1,
            consts: &[],
        }];

        /// `ggml_dequant_<t>_f16_gfx1201`: `vx dst colmap nb K chunk`, no hidden args, segment 36/8.
        pub static DEQUANT_F16: KernelAbiV1 = KernelAbiV1 {
            id: concat!("ggml_convert.dequant_", $tag, "_f16.gfx1201"),
            version: 1,
            wave: 32,
            numeric: NumericContractId(concat!("ggml.dequant.", $tag, ".f16")),
            params: PARAMS_DQ_F16,
            operands: OPERANDS_DQ_F16,
            fields: FIELDS_DQ_F16,
            hidden: &[],
            explicit_bytes: 36,
            segment_bytes: 36,
            segment_align: 8,
            domain: DOMAIN_DQ_F16,
            exports: EXPORTS_DQ_F16,
        };
        /// `ggml_dequant_<t>_f32_gfx1201`: `vx dst colmap nb K chunk bf16_round`, segment 40/8.
        pub static DEQUANT_F32: KernelAbiV1 = KernelAbiV1 {
            id: concat!("ggml_convert.dequant_", $tag, "_f32.gfx1201"),
            version: 1,
            wave: 32,
            numeric: NumericContractId(concat!("ggml.dequant.", $tag, ".f32")),
            params: PARAMS_DQ_F32,
            operands: OPERANDS_DQ_F32,
            fields: FIELDS_DQ_F32,
            hidden: &[],
            explicit_bytes: 40,
            segment_bytes: 40,
            segment_align: 8,
            domain: DOMAIN_DQ_F32,
            exports: EXPORTS_DQ_F32,
        };
        /// `ggml_get_row_<t>_f32_gfx1201`: `vx tok_ids dst K`, segment 28/8.
        pub static GET_ROW: KernelAbiV1 = KernelAbiV1 {
            id: concat!("ggml_convert.get_row_", $tag, "_f32.gfx1201"),
            version: 1,
            wave: 32,
            numeric: NumericContractId(concat!("ggml.get_row.", $tag, ".f32")),
            params: PARAMS_ROW,
            operands: OPERANDS_ROW,
            fields: FIELDS_ROW,
            hidden: &[],
            explicit_bytes: 28,
            segment_bytes: 28,
            segment_align: 8,
            domain: DOMAIN_ROW,
            exports: EXPORTS_ROW,
        };
        /// `ggml_get_row1_<t>_f32_gfx1201`: `vx tok dst K` (token by value), segment 28/8.
        pub static GET_ROW1: KernelAbiV1 = KernelAbiV1 {
            id: concat!("ggml_convert.get_row1_", $tag, "_f32.gfx1201"),
            version: 1,
            wave: 32,
            numeric: NumericContractId(concat!("ggml.get_row.", $tag, ".f32")),
            params: PARAMS_ROW1,
            operands: OPERANDS_ROW1,
            fields: FIELDS_ROW1,
            hidden: &[],
            explicit_bytes: 28,
            segment_bytes: 28,
            segment_align: 8,
            domain: DOMAIN_ROW1,
            exports: EXPORTS_ROW1,
        };
    };
}

/// `iq4_xs`: 136 B / 256, 32-thread blocks, one super-block per workgroup.
pub mod iq4_xs {
    convert_type!(
        "iq4_xs",
        256,
        136,
        FORMAT_IQ4_XS_136,
        4,
        32,
        Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]),
        Expr::DivExact(&P_K, 256)
    );
}
/// `q4_k`: 144 B / 256, 32-thread blocks.
pub mod q4_k {
    convert_type!(
        "q4_k",
        256,
        144,
        FORMAT_Q4_K_144,
        4,
        32,
        Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]),
        Expr::DivExact(&P_K, 256)
    );
}
/// `q5_k`: 176 B / 256, 64-thread blocks.
pub mod q5_k {
    convert_type!(
        "q5_k",
        256,
        176,
        FORMAT_Q5_K_176,
        4,
        64,
        Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]),
        Expr::DivExact(&P_K, 256)
    );
}
/// `q6_k`: 210 B / 256, 64-thread blocks.
pub mod q6_k {
    convert_type!(
        "q6_k",
        256,
        210,
        FORMAT_Q6_K_210,
        2,
        64,
        Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256)]),
        Expr::DivExact(&P_K, 256)
    );
}
/// `q8_0`: 34 B / 32, 256-thread blocks, 512 elements per workgroup.
pub mod q8_0 {
    convert_type!(
        "q8_0",
        32,
        34,
        FORMAT_Q8_0_34,
        2,
        256,
        Expr::DivCeil(&Expr::Mul(&[P_M, P_K]), 512),
        Expr::DivCeil(&P_K, 512)
    );
}

// ---- in-place A8 round trip --------------------------------------------------------

const OP_X_INPLACE: OperandV1 = OperandV1 {
    name: "x",
    format: FORMAT_F32_FLAT,
    access: Access::ReadWrite,
    align: 4,
    extent: Some(Expr::Mul(&[P_NE, Expr::Const(4)])),
    nullable: false,
    note: "read and written in place, element i only by its own thread (the 32-lane block reductions are register shuffles)",
};
const OPERANDS_ROUNDTRIP: &[OperandV1] = &[OP_X_INPLACE];
const FIELDS_ROUNDTRIP: &[FieldV1] = &[F_X, F_NE];
const DOMAIN_ROUNDTRIP: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Positive(P_NE),
        why: "ne > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_NE, 32),
        why: "ne % 32 == 0 (whole q8_1 blocks: the block reductions span 32 lanes)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_NE, ROUNDTRIP_MAX_NE),
        why: "256 * blockIdx.x + threadIdx.x is computed in int",
    },
];
const EXPORTS_ROUNDTRIP: &[ExportV1] = &[ExportV1 {
    symbol: "ggml_q81_roundtrip_gfx1201",
    arch: Arch::Gfx1201,
    object: OBJECT,
    block: [256, 1, 1],
    dynamic_lds: 0,
    group_segment_fixed: 0,
    grid: [Expr::DivCeil(&P_NE, 256), Expr::Const(1), Expr::Const(1)],
    consts: &[],
}];

/// `ggml_q81_roundtrip_gfx1201`: `x ne`, in place, segment 12/8. Domain note (not expressible as a
/// rule): every 32-element block needs `amax < 127 * 2^-149`-free input (`d = amax / 127` must not
/// underflow to 0 in f32, else `x / d` is infinite) and `amax <= 65504 * 127` (`half(d)` finite);
/// activations are far inside both.
pub static Q81_ROUNDTRIP: KernelAbiV1 = KernelAbiV1 {
    id: "ggml_convert.q81_roundtrip.gfx1201",
    version: 1,
    wave: 32,
    numeric: NumericContractId("ggml.q81_roundtrip"),
    params: PARAMS_ROUNDTRIP,
    operands: OPERANDS_ROUNDTRIP,
    fields: FIELDS_ROUNDTRIP,
    hidden: &[],
    explicit_bytes: 12,
    segment_bytes: 12,
    segment_align: 8,
    domain: DOMAIN_ROUNDTRIP,
    exports: EXPORTS_ROUNDTRIP,
};

/// Every contract of this family: per type (iq4_xs, q4_k, q5_k, q6_k, q8_0) the f16 dequant, f32
/// dequant, id gather and by-value gather, then the round trip.
pub static ABIS: &[&KernelAbiV1] = &[
    &iq4_xs::DEQUANT_F16,
    &iq4_xs::DEQUANT_F32,
    &iq4_xs::GET_ROW,
    &iq4_xs::GET_ROW1,
    &q4_k::DEQUANT_F16,
    &q4_k::DEQUANT_F32,
    &q4_k::GET_ROW,
    &q4_k::GET_ROW1,
    &q5_k::DEQUANT_F16,
    &q5_k::DEQUANT_F32,
    &q5_k::GET_ROW,
    &q5_k::GET_ROW1,
    &q6_k::DEQUANT_F16,
    &q6_k::DEQUANT_F32,
    &q6_k::GET_ROW,
    &q6_k::GET_ROW1,
    &q8_0::DEQUANT_F16,
    &q8_0::DEQUANT_F32,
    &q8_0::GET_ROW,
    &q8_0::GET_ROW1,
    &Q81_ROUNDTRIP,
];
