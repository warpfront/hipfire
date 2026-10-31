// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic-op GEMM/GEMV launchers (hipcc): dense Q8_0/F32 GEMMs, grouped MoE GEMMs, embedding.
//!
//! Ten contracts, each with a [`WorkSpecV1`] sidecar:
//!
//! | contract | exports | launcher |
//! |---|---|---|
//! | [`GEMM_Q8_0_WMMA`] | `gemm_q8_0_wmma_gfx12` (gfx1201), `gemm_q8_0_wmma` (gfx1100) | `Gpu::gemm_q8_0_wmma` (`crates/rdna-compute/src/gemm.rs`) |
//! | [`GEMM_MQ4G256V2_MOE_GROUPED_WMMA`] | `gemm_mq4g256v2_moe_grouped_wmma_gfx12` (gfx1201), `gemm_mq4g256v2_moe_grouped_wmma_k2` (gfx1100) | `Gpu::gemm_mq4g256v2_moe_grouped_wmma_k2_impl` (`gemm.rs`) |
//! | [`GEMM_F32_BATCHED`] | `gemm_f32_batched` (gfx1201, gfx1100) | `Gpu::gemm_f32_batched` (`gemm.rs`) |
//! | [`GEMM_BF16_XF32_BATCHED`] | `gemm_bf16_xf32_batched` (gfx1201, gfx1100) | `Gpu::gemm_bf16_xf32_batched` (`gemm.rs`) |
//! | [`GEMM_BF16_XF32_TILED_ROWS`] / [`GEMM_BF16_XF32_TILED_ROWS_MP4`] | `gemm_bf16_xf32_tiled_rows`, `gemm_bf16_xf32_tiled_rows_mp4` (gfx1201) | `Gpu::gemm_bf16_xf32_tiled_rows` (`gemm.rs`), selector `rows_gemm` (`crates/hipfire-dispatch/src/pipeline/stream_mix.rs`) |
//! | [`GEMV_Q8_0`] | `gemv_q8_0` (gfx1201, gfx1100) | `Gpu::gemv_q8_0` (`crates/rdna-compute/src/gemv.rs`) |
//! | [`EMBEDDING_Q8_BATCHED`] | `embedding_q8_batched` (gfx1201, gfx1100) | `Gpu::embedding_lookup_q8_batched` (`crates/rdna-compute/src/embedding.rs`) |
//! | [`MOE_GROUPED_QT44_A8_SYM`] | `moe_grouped_qt44_a8_sym_gfx1201_nt{1,2}r{1,2}` (gfx1201) | `Gpu::gemm_mq4g256v2_moe_grouped_a8_sym` (`crates/rdna-compute/src/moe_grouped_qt44_iu4.rs`) |
//! | [`MOE_GROUPED_QT44_IU4_SYM`] | `moe_grouped_qt44_iu4_sym_gfx1201_nt{1,2,4}r{1,2}` (gfx1201) | `Gpu::gemm_mq4g256v2_moe_grouped_iu4_sym` (same file) |
//!
//! Evidence for every layout: the JIT-cached objects in `~/.hipfire_kernels/<arch>/` (clang offload
//! bundles; unbundled with `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--<arch>`,
//! read with `llvm-readelf --notes`), each chosen because the sibling `<module>.<hash>.hip` in that
//! directory is byte-identical to the current `kernels/src` source. The metadata names no argument:
//! field names are the kernels' parameter names. Hash used per object is cited at each contract.
//! Every kernel is built for wave 32; `.private_segment_fixed_size` is 0 everywhere.
//! hipcc contracts carry no object hash (`HipccSource`), so no drift evidence is checked against them.

use crate::work::{
    op_class, BindSource, BindV1, Pipe, RowDomainV1, Stacking, WeightAxis, WorkSpecV1, WorkV1,
    ALL_ROWS, STACK_PARAM,
};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_convert::{FORMAT_F16_FLAT, FORMAT_F32_FLAT, FORMAT_I32_TOKEN_IDS};
use super::ggml_mmvq::FORMAT_Q8_0_34;
use super::mq6_i8::{FORMAT_BLOCK_I8_128_KB_N, FORMAT_F32_N_M};

// ---- formats -----------------------------------------------------------------

/// BF16 flat tensor `[M, K]` row-major (u16 words, 2 bytes per element, widened exactly to f32 in registers).
pub const FORMAT_BF16_FLAT: FormatId = FormatId("bf16.flat");

/// `u64 [E]` table of expert weight slab base addresses (MQ4G256V2: `M x K/256` groups of 136 B).
pub const FORMAT_EXPERT_PTR_TABLE_U64: FormatId = FormatId("moe_grouped.expert_ptr_table.u64");
/// `i32 [ceil(m_total / 16)]`: expert id of each 16-slot tile, `-1` for sentinel tiles.
pub const FORMAT_EXPERT_TILE_IDS_I32: FormatId =
    FormatId("moe_grouped.expert_tile_ids.i32.per_tile16");
/// `i32 [m_total]`: flat routed slot of each grouped position, `-1` padding.
pub const FORMAT_SORTED_SLOT_INDEX_I32: FormatId =
    FormatId("moe_grouped.sorted_slot_index.i32.minus1_padding");
/// `i32 [E + 1]` 16-aligned slot prefix of the scatter (`expert_offsets[e] >> 4` = first tile of `e`).
pub const FORMAT_EXPERT_OFFSETS_I32: FormatId = FormatId("moe_grouped.expert_offsets.i32.e_plus_1");
/// A4 `block_i4_128` activations laid out `[K/128][x_src_rows]`, 72 B records
/// (`float d; int s; unsigned char qs[64]`, `moe_grouped_qt44_iu4_sym.gfx12.hip:39-43`).
pub const FORMAT_BLOCK_I4_128_KB_N: FormatId = FormatId("block_i4_128.kb_x_n.72b");

// ---- numeric contracts -----------------------------------------------------------

/// Q8_0 weight x F16 activation, per-element `half(scale) * half(float(i8))` dequant, f32-accumulating
/// `wmma_f32_16x16x16_f16`, K order ascending in 32-blocks, plain f32 store.
pub const NUMERIC_Q8_0_WMMA_F16: NumericContractId =
    NumericContractId("generic_gemm.q8_0_x_f16.wmma_f32acc");
/// MQ4G256V2 weight (dual affine grid per 256) x F16 activation, f32-accumulating F16 WMMA.
pub const NUMERIC_MQ4_GROUPED_WMMA_F16: NumericContractId =
    NumericContractId("generic_gemm.mq4g256v2_grouped_x_f16.wmma_f32acc");
/// F32 x F32, lane-strided FMA over K (lane `l` takes `k = l, l+32, ...`) then a width-32 `shfl_down` tree.
/// `gemm_f32_batched`, `gemm_bf16_xf32_batched` and `gemm_bf16_xf32_tiled_rows*` share it (BF16 operand
/// A widens exactly, so the BF16 kernels are bit-identical to the F32 kernel on the widened weight).
pub const NUMERIC_F32_LANE_STRIDED: NumericContractId =
    NumericContractId("generic_gemm.f32.lane_strided_fma.shfl_tree32");
/// Q8_0 weight x F32 activation GEMV, 8-block unrolled lane-strided FMA, `shfl_down` tree.
pub const NUMERIC_GEMV_Q8_0: NumericContractId =
    NumericContractId("generic_gemm.gemv_q8_0.f32_lane_strided.shfl_tree32");
/// `f32(half(scale)) * f32(i8)` exact product stored as f32.
pub const NUMERIC_EMBEDDING_Q8_0: NumericContractId =
    NumericContractId("generic_gemm.embedding_q8_0.f32_dequant");
/// Symmetric qt44 weight x `block_i8_128` activation: exact int32 per K128 epoch (`wmma_i32_16x16x16_iu8`),
/// `sum = fma(RN(sc * d), float(C), sum)` ascending epochs, F32 store.
pub const NUMERIC_QT44_A8: NumericContractId =
    NumericContractId("generic_gemm.qt44_sym_x_i8_128.wmma_iu8.epoch_fold_f32");
/// Symmetric qt44 weight x `block_i4_128` activation: exact int32 per K128 epoch
/// (`wmma_i32_16x16x32_iu4`), same fold as [`NUMERIC_QT44_A8`].
pub const NUMERIC_QT44_IU4: NumericContractId =
    NumericContractId("generic_gemm.qt44_sym_x_i4_128.wmma_iu4.epoch_fold_f32");

// ---- shared pieces ------------------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_N: Expr = Expr::Param("N");
const P_DIM: Expr = Expr::Param("dim");
const P_VOCAB: Expr = Expr::Param("vocab");
const P_X_ROW_DIV: Expr = Expr::Param("x_row_div");
const P_M_TOTAL: Expr = Expr::Param("m_total");
const P_N_EXPERTS: Expr = Expr::Param("n_experts");
const P_X_SRC_ROWS: Expr = Expr::Param("x_src_rows");
const P_ROWS: Expr = Expr::Param("rows");
const ONE: Expr = Expr::Const(1);
/// HIP grid-y limit (65535 workgroups): launchers chunk beyond it.
const GRID_Y_MAX: u64 = 65535;

const fn param(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}

const fn ptr_field(name: &'static str, offset: u32, operand: &'static str) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand },
    }
}

const fn i32_field(name: &'static str, offset: u32, value: Expr) -> FieldV1 {
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

const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}

const fn hip_hidden(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
    HiddenV1 {
        offset,
        width,
        kind,
    }
}

/// `u64 [E]` pointer table operand (same in the F16 grouped GEMM and both qt44 kernels).
const OP_EXPERT_PTRS: OperandV1 = OperandV1 {
    name: "expert_weight_ptrs",
    format: FORMAT_EXPERT_PTR_TABLE_U64,
    access: Access::ReadOnly,
    align: 8,
    extent: Some(Expr::Mul(&[P_N_EXPERTS, Expr::Const(8)])),
    nullable: false,
    note: "entries point at MQ4G256V2 expert slabs `[M, K/256] x 136 B`; the slab extent is the loader's contract (the \
           kernel indexes the table only after `0 <= expert_id < n_experts`)",
};

// ================================================================================================
// gemm_q8_0_wmma (gfx12 + gfx11 siblings)
// ================================================================================================

const PARAMS_Q8_WMMA: &[ParamV1] = &[
    param("M", "weight rows (output features)"),
    param("K", "row length in elements, a multiple of 32"),
    param("N", "activation rows (grid y x 16, tail masked)"),
];

const OPS_Q8_WMMA: &[OperandV1] = &[
    OperandV1 {
        name: "A_q8",
        format: FORMAT_Q8_0_34,
        access: Access::ReadOnly,
        align: 2,
        extent: Some(Expr::Mul(&[P_M, Expr::DivExact(&P_K, 32), Expr::Const(34)])),
        nullable: false,
        note: "Q8_0 rows `[M, K/32] x 34 B` (f16 scale + 32 i8); read with 8-byte memcpy, alignment 2",
    },
    OperandV1 {
        name: "X",
        format: FORMAT_F16_FLAT,
        access: Access::ReadOnly,
        align: 32,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(2)])),
        nullable: false,
        note: "F16 activations `[N, K]`; the gfx12 kernel loads 16 B vectors, the gfx11 kernel `half16_t` (32 B)",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[N, M]`; tail rows/columns are masked, every in-range element is written once",
    },
];

const FIELDS_Q8_WMMA: &[FieldV1] = &[
    ptr_field("A_q8", 0, "A_q8"),
    ptr_field("X", 8, "X"),
    ptr_field("Y", 16, "Y"),
    i32_field("M", 24, P_M),
    i32_field("K", 28, P_K),
    i32_field("N", 32, P_N),
];

const DOMAIN_Q8_WMMA: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 32),
        "the kernel iterates K/32 Q8_0 blocks and silently drops a tail",
    ),
    rule(
        Rule::AtMost(Expr::DivCeil(&P_N, 16), GRID_Y_MAX),
        "grid y (ceil(N/16)) must fit the HIP y-grid limit",
    ),
];

const GRID_Q8_WMMA: [Expr; 3] = [Expr::DivCeil(&P_M, 16), Expr::DivCeil(&P_N, 16), ONE];

/// `gemm_q8_0_wmma_gfx12`: hsaco `~/.hipfire_kernels/gfx1201/gemm_q8_0_wmma_gfx12.289b06a6b8e3da41.hsaco`
/// (source byte-identical to `kernels/src/gemm_q8_0_wmma.gfx12.hip`; also the unhashed
/// `gemm_q8_0_wmma_gfx12.hsaco`): args `A_q8@0 X@8 Y@16` (`read_only, read_only, write_only` buffers),
/// `M@24 K@28 N@32` by value, `.kernarg_segment_size` 36 / align 8, NO hidden words,
/// `.group_segment_fixed_size` 0, `.max_flat_workgroup_size` 32.
/// `gemm_q8_0_wmma` (gfx1100): `gfx1100/gemm_q8_0_wmma.2e91ad0f5a9fa2dd.hsaco` (source identical to
/// `kernels/src/gemm_q8_0_wmma.hip`), same layout (36 B, no hidden, LDS 0, max flat 32).
/// gfx1151 launches `gemm_q8_0_wmma` too (and `_gfx1151_xlds`); no gfx1151 object was inspected, so it
/// is not an export here.
pub static GEMM_Q8_0_WMMA: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.q8_0_wmma.f16x.f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_0_WMMA_F16,
    params: PARAMS_Q8_WMMA,
    operands: OPS_Q8_WMMA,
    fields: FIELDS_Q8_WMMA,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_Q8_WMMA,
    exports: &[
        ExportV1 {
            symbol: "gemm_q8_0_wmma_gfx12",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "gemm_q8_0_wmma_gfx12",
                source: "kernels/src/gemm_q8_0_wmma.gfx12.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_Q8_WMMA,
            consts: &[],
        },
        ExportV1 {
            symbol: "gemm_q8_0_wmma",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "gemm_q8_0_wmma",
                source: "kernels/src/gemm_q8_0_wmma.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_Q8_WMMA,
            consts: &[],
        },
    ],
};

const BIND_Q8_WMMA: &[BindV1] = &[
    BindV1 {
        param: "N",
        source: BindSource::ChunkRows,
    },
    BindV1 {
        param: "K",
        source: BindSource::Weight {
            by: "M",
            by_axis: WeightAxis::Rows,
            codecs: &["q8_0"],
            stacking: Stacking::Single,
            fuse_siblings: false,
        },
    },
];

pub static WORK_GEMM_Q8_0_WMMA: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_Q8_0_WMMA,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "q8_0",
        pipe: Pipe::WmmaF16,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: RowDomainV1 {
            min_rows: 1,
            max_rows: None,
            predicate: "crates/rdna-compute/src/gemm.rs::gemm_q8_0_batched_chunked",
            kill_env: Some("HIPFIRE_Q8_BATCHED_LEGACY"),
        },
        bind: BIND_Q8_WMMA,
        note: "MACs = M*N*K (tile padding excluded). X is the F16 conversion of the activation (the separate \
               convert kernel is its own node). Bytes = weights + X + Y extents. gfx1151 routes m>=4096 \
               m%64==0 n%64==0 to gemm_q8_0_wmma_4w and 1..=16 rows to the xlds twin: not this contract. \
               `HIPFIRE_Q8_BATCHED_LEGACY=1` routes the scalar batched kernel instead.",
    },
};

// ================================================================================================
// gemm_mq4g256v2_moe_grouped_wmma (gfx12 + gfx11 siblings)
// ================================================================================================

const PARAMS_MOE_GROUPED_WMMA: &[ParamV1] = &[
    param("M", "weight rows per expert (output features)"),
    param("K", "row length in elements, a multiple of 256"),
    param(
        "x_row_div",
        "`sorted_slot_index` -> X_src row: `flat / x_row_div` (gate/up: top-k; down: 1)",
    ),
    param(
        "m_total",
        "grouped positions launched (a multiple of 16 expected; the dispatcher may pass an upper bound with sentinel tiles)",
    ),
    param("n_experts", "length of the expert pointer table"),
    param(
        "x_src_rows",
        "rows of `X_src` (extent only; not a kernel argument)",
    ),
];

const OPS_MOE_GROUPED_WMMA: &[OperandV1] = &[
    OP_EXPERT_PTRS,
    OperandV1 {
        name: "expert_tile_ids",
        format: FORMAT_EXPERT_TILE_IDS_I32,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivCeil(&P_M_TOTAL, 16), Expr::Const(4)])),
        nullable: false,
        note: "expert of each 16-slot tile; `-1` (and any id >= n_experts) makes the tile return",
    },
    OperandV1 {
        name: "sorted_slot_index",
        format: FORMAT_SORTED_SLOT_INDEX_I32,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M_TOTAL, Expr::Const(4)])),
        nullable: false,
        note: "flat routed slot per grouped position, `-1` = padding lane (contributes a zero B row); \
               `slot / x_row_div < x_src_rows` is the caller's contract (not checked on the device)",
    },
    OperandV1 {
        name: "X_src",
        format: FORMAT_F16_FLAT,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_X_SRC_ROWS, P_K, Expr::Const(2)])),
        nullable: false,
        note: "F16 activations `[x_src_rows, K]`, 16 B vector loads",
    },
    OperandV1 {
        name: "Y_grouped",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M_TOTAL, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 `[m_total, M]` in grouped order; sentinel tiles and padding positions are not stored",
    },
];

const FIELDS_MOE_GROUPED_WMMA: &[FieldV1] = &[
    ptr_field("expert_weight_ptrs", 0, "expert_weight_ptrs"),
    ptr_field("expert_tile_ids", 8, "expert_tile_ids"),
    ptr_field("sorted_slot_index", 16, "sorted_slot_index"),
    ptr_field("X_src", 24, "X_src"),
    ptr_field("Y_grouped", 32, "Y_grouped"),
    i32_field("M", 40, P_M),
    i32_field("K", 44, P_K),
    i32_field("x_row_div", 48, P_X_ROW_DIV),
    i32_field("m_total", 52, P_M_TOTAL),
    i32_field("n_experts", 56, P_N_EXPERTS),
];

const DOMAIN_MOE_GROUPED_WMMA: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(Rule::Positive(P_M_TOTAL), "no grouped positions"),
    rule(Rule::Positive(P_N_EXPERTS), "empty expert table"),
    rule(Rule::Positive(P_X_SRC_ROWS), "empty activation source"),
    rule(
        Rule::AtLeast(P_X_ROW_DIV, 1),
        "`flat / x_row_div` (the kernel treats <= 1 as the identity)",
    ),
    rule(
        Rule::MultipleOf(P_K, 256),
        "the kernel iterates K/256 groups of 136 B and reads whole groups",
    ),
    rule(
        Rule::AtMost(Expr::DivCeil(&P_M_TOTAL, 16), GRID_Y_MAX),
        "grid y (ceil(m_total/16)) must fit the HIP y-grid limit",
    ),
];

const GRID_MOE_GROUPED_WMMA: [Expr; 3] =
    [Expr::DivCeil(&P_M, 16), Expr::DivCeil(&P_M_TOTAL, 16), ONE];

/// `gemm_mq4g256v2_moe_grouped_wmma_gfx12`: hsaco
/// `~/.hipfire_kernels/gfx1201/gemm_mq4g256v2_moe_grouped_wmma_gfx12.419d93e3d433eed7.hsaco` (source
/// byte-identical to `kernels/src/gemm_mq4g256v2_moe_grouped_wmma.gfx12.hip`; the unhashed
/// `.hsaco` of the same module is an older build with ONE argument fewer: 4 scalars, segment 56, and is
/// NOT used): args `expert_weight_ptrs@0 expert_tile_ids@8 sorted_slot_index@16 X_src@24` (read_only)
/// `Y_grouped@32` (write_only), `M@40 K@44 x_row_div@48 m_total@52 n_experts@56` by value,
/// `.kernarg_segment_size` 60 / align 8, no hidden words, LDS 0, `.max_flat_workgroup_size` 32.
/// gfx1100 `gemm_mq4g256v2_moe_grouped_wmma_k2`:
/// `gfx1100/gemm_mq4g256v2_moe_grouped_wmma_k2.f945b1c7d0b15147.hsaco` (source identical to
/// `kernels/src/gemm_mq4g256v2_moe_grouped_wmma_k2.hip`), the identical 60 B layout. The gfx1100
/// launcher may instead take the PM twin (`moe_grouped_k2_pm`, 64 rows per 128-lane block) or, on
/// gfx1151, a different source: those symbols are not exports of this contract.
pub static GEMM_MQ4G256V2_MOE_GROUPED_WMMA: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.mq4g256v2_moe_grouped_wmma.f16x.f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_MQ4_GROUPED_WMMA_F16,
    params: PARAMS_MOE_GROUPED_WMMA,
    operands: OPS_MOE_GROUPED_WMMA,
    fields: FIELDS_MOE_GROUPED_WMMA,
    hidden: &[],
    explicit_bytes: 60,
    segment_bytes: 60,
    segment_align: 8,
    domain: DOMAIN_MOE_GROUPED_WMMA,
    exports: &[
        ExportV1 {
            symbol: "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
                source: "kernels/src/gemm_mq4g256v2_moe_grouped_wmma.gfx12.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_MOE_GROUPED_WMMA,
            consts: &[],
        },
        ExportV1 {
            symbol: "gemm_mq4g256v2_moe_grouped_wmma_k2",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "gemm_mq4g256v2_moe_grouped_wmma_k2",
                source: "kernels/src/gemm_mq4g256v2_moe_grouped_wmma_k2.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: GRID_MOE_GROUPED_WMMA,
            consts: &[],
        },
    ],
};

const MOE_TOPK_KEYS: &[&str] = &[
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "top_k",
];

const BIND_MOE_GROUPED_WMMA: &[BindV1] = &[
    BindV1 {
        param: "K",
        source: BindSource::Weight {
            by: "M",
            by_axis: WeightAxis::Rows,
            codecs: &["mq4g256v2"],
            stacking: Stacking::Either,
            fuse_siblings: true,
        },
    },
    BindV1 {
        param: "rows",
        source: BindSource::RoutedRows {
            topk: MOE_TOPK_KEYS,
            dense: "m_total",
        },
    },
    BindV1 {
        param: "tokens",
        source: BindSource::RoutedTokens { dense: "m_total" },
    },
];

pub static WORK_GEMM_MQ4G256V2_MOE_GROUPED_WMMA: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
    work: WorkV1 {
        op_class: op_class::GROUPED_GEMM,
        codec: "mq4g256v2",
        pipe: Pipe::WmmaF16,
        macs: Some(Expr::Mul(&[P_ROWS, P_M, P_K])),
        flops: None,
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[
                Expr::Param(STACK_PARAM),
                P_M,
                Expr::DivExact(&P_K, 256),
                Expr::Const(136),
            ]),
            Expr::Mul(&[P_ROWS, P_K, Expr::Const(2)]),
            Expr::Mul(&[P_ROWS, P_M, Expr::Const(4)]),
        ])),
        rows: Some("tokens"),
        admit: ALL_ROWS,
        bind: BIND_MOE_GROUPED_WMMA,
        note: "MACs = useful routed rows * M * K; `m_total` is a launch bound (sentinel tiles return early) so it \
               is not the row count. Bytes: `stack` expert slabs `M*K/256*136` (the weight binding's stack size: \
               every expert of the layer for a routed launch, an upper bound when routing leaves experts cold; 1 \
               for the shared-expert and dense launches) + gathered X + Y. Gate/up M is the fused gate+up of one \
               expert (sibling fusion).",
    },
};

// ================================================================================================
// gemm_f32_batched / gemm_bf16_xf32_batched / gemm_bf16_xf32_tiled_rows / gemm_bf16_xf32_tiled_rows_mp4
// ================================================================================================

const PARAMS_F32_GEMM: &[ParamV1] = &[
    param("M", "weight rows (outputs per activation row)"),
    param("K", "row length in elements"),
    param(
        "N",
        "activation rows of THIS launch (the launcher chunks grid y at 65535 tiles)",
    ),
];

const OPS_F32_GEMM: &[OperandV1] = &[
    OperandV1 {
        name: "A",
        format: FORMAT_F32_FLAT,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 weight `[M, K]` (the launcher also accepts an untyped `Raw` view whose logical dtype is F32)",
    },
    OperandV1 {
        name: "B",
        format: FORMAT_F32_FLAT,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[N, K]`",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[N, M]`; every element is written once (lane 0 of its wave)",
    },
];

const OPS_BF16_XF32_GEMM: &[OperandV1] = &[
    OperandV1 {
        name: "A",
        format: FORMAT_BF16_FLAT,
        access: Access::ReadOnly,
        align: 2,
        extent: Some(Expr::Mul(&[P_M, P_K, Expr::Const(2)])),
        nullable: false,
        note: "BF16 weight `[M, K]` row-major (u16 words, 2-byte aligned), widened exactly to f32 in registers",
    },
    OperandV1 {
        name: "B",
        format: FORMAT_F32_FLAT,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[N, K]`",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[N, M]`; every element is written once (lane 0 of its wave)",
    },
];

const FIELDS_F32_GEMM: &[FieldV1] = &[
    ptr_field("A", 0, "A"),
    ptr_field("B", 8, "B"),
    ptr_field("Y", 16, "Y"),
    i32_field("M", 24, P_M),
    i32_field("K", 28, P_K),
    i32_field("N", 32, P_N),
];

const DOMAIN_F32_BATCHED: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::AtMost(P_N, GRID_Y_MAX),
        "grid y = N must fit the HIP y-grid limit (the launcher chunks)",
    ),
];

/// `gemm_f32_batched`, one wave per output: hsaco
/// `~/.hipfire_kernels/gfx1201/gemm_f32_batched.0304a82e699eeca2.hsaco` (source byte-identical to
/// `kernels/src/gemm_f32.hip`; `85fd231db96b7c96` is the same source) and
/// `gfx1100/gemm_f32_batched.fc3453b930547691.hsaco`: args `A@0` (read_only) `B@8` (read_only) `Y@16`
/// (write_only), `M@24 K@28 N@32` by value, segment 36 / align 8, no hidden words, LDS 0,
/// `.max_flat_workgroup_size` 32 (both archs).
pub static GEMM_F32_BATCHED: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.f32_batched.wave_per_output",
    version: 1,
    wave: 32,
    numeric: NUMERIC_F32_LANE_STRIDED,
    params: PARAMS_F32_GEMM,
    operands: OPS_F32_GEMM,
    fields: FIELDS_F32_GEMM,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_F32_BATCHED,
    exports: &[
        ExportV1 {
            symbol: "gemm_f32_batched",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "gemm_f32_batched",
                source: "kernels/src/gemm_f32.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, P_N, ONE],
            consts: &[],
        },
        ExportV1 {
            symbol: "gemm_f32_batched",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "gemm_f32_batched",
                source: "kernels/src/gemm_f32.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, P_N, ONE],
            consts: &[],
        },
    ],
};

const FIELDS_BF16_XF32_BATCHED: &[FieldV1] = &[
    ptr_field("W", 0, "A"),
    ptr_field("X", 8, "B"),
    ptr_field("Y", 16, "Y"),
    i32_field("M", 24, P_M),
    i32_field("K", 28, P_K),
    i32_field("N", 32, P_N),
];

/// `gemm_bf16_xf32_batched`, one wave per output, BF16 weight: hipcc-JIT'd from
/// `kernels/src/gemm_bf16_xf32_batched.hip`; same layout as [`GEMM_F32_BATCHED`] (the hsaco evidence
/// cited there belongs to the F32 kernel; no BF16 hsaco hash is pinned): args `W@0` (read_only) `X@8`
/// (read_only) `Y@16` (write_only), `M@24 K@28 N@32` by value, segment 36 / align 8, no hidden words,
/// LDS 0, block 32. Lane-strided numerics identical to `gemm_f32_batched`.
pub static GEMM_BF16_XF32_BATCHED: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.bf16_xf32_batched.wave_per_output",
    version: 1,
    wave: 32,
    numeric: NUMERIC_F32_LANE_STRIDED,
    params: PARAMS_F32_GEMM,
    operands: OPS_BF16_XF32_GEMM,
    fields: FIELDS_BF16_XF32_BATCHED,
    hidden: &[],
    explicit_bytes: 36,
    segment_bytes: 36,
    segment_align: 8,
    domain: DOMAIN_F32_BATCHED,
    exports: &[
        ExportV1 {
            symbol: "gemm_bf16_xf32_batched",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "gemm_bf16_xf32_batched",
                source: "kernels/src/gemm_bf16_xf32_batched.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, P_N, ONE],
            consts: &[],
        },
        ExportV1 {
            symbol: "gemm_bf16_xf32_batched",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "gemm_bf16_xf32_batched",
                source: "kernels/src/gemm_bf16_xf32_batched.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, P_N, ONE],
            consts: &[],
        },
    ],
};

/// Hidden block of the hipcc `gemm_bf16_xf32_tiled_rows*` objects (they read `blockDim`): offsets from the
/// metadata of the F32 sibling `gfx1201/gemm_f32_tiled_rows.e23c665b964e8f70.hsaco` (both entries); the
/// BF16 object is hipcc-JIT'd with the same entry layout.
const HIDDEN_BF16_TILED: &[HiddenV1] = &[
    hip_hidden(40, 4, HiddenKind::BlockCount(Axis::X)),
    hip_hidden(44, 4, HiddenKind::BlockCount(Axis::Y)),
    hip_hidden(48, 4, HiddenKind::BlockCount(Axis::Z)),
    hip_hidden(52, 2, HiddenKind::GroupSize(Axis::X)),
    hip_hidden(54, 2, HiddenKind::GroupSize(Axis::Y)),
    hip_hidden(56, 2, HiddenKind::GroupSize(Axis::Z)),
    hip_hidden(58, 2, HiddenKind::Remainder(Axis::X)),
    hip_hidden(60, 2, HiddenKind::Remainder(Axis::Y)),
    hip_hidden(62, 2, HiddenKind::Remainder(Axis::Z)),
    hip_hidden(80, 8, HiddenKind::GlobalOffset(Axis::X)),
    hip_hidden(88, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hip_hidden(96, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hip_hidden(104, 2, HiddenKind::GridDims),
];

const DOMAIN_BF16_TILED: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::AtMost(P_N, GRID_Y_MAX * 4),
        "grid y = ceil(N/4) must fit the HIP y-grid limit (the launcher chunks at 65535 * 4 rows)",
    ),
];

const DOMAIN_BF16_TILED_MP4: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::AtMost(P_N, GRID_Y_MAX * 4),
        "grid y = ceil(N/4) must fit the HIP y-grid limit (the launcher chunks at 65535 * 4 rows)",
    ),
    rule(
        Rule::AtMost(P_M, 32),
        "the MPW=4 entry covers 8 waves x 4 outputs = 32 outputs per CTA and its grid x is ceil(M/32) with M <= 32",
    ),
];

/// `gemm_bf16_xf32_tiled_rows` (MPW=16, any M; grid x = ceil(M/128)), BF16 weight: hipcc-JIT'd from
/// `kernels/src/gemm_bf16_xf32_tiled_rows.hip` with the entry layout of its F32 sibling (the hsaco
/// `gemm_f32_tiled_rows.e23c665b964e8f70` evidence belongs to that F32 kernel; no BF16 hsaco hash is
/// pinned): args `A@0` (read_only)
/// `X@8` (read_only) `Y@16` (write_only), `M@24 K@28 N@32` by value, then the 13 hidden words at
/// 40..106, `.kernarg_segment_size` 296 / align 8, `.group_segment_fixed_size` 16384
/// (`xs[4][1024]` f32), `.max_flat_workgroup_size` 256. The launcher chunks the grid-y extent.
pub static GEMM_BF16_XF32_TILED_ROWS: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.bf16_xf32_tiled_rows.mpw16",
    version: 1,
    wave: 32,
    numeric: NUMERIC_F32_LANE_STRIDED,
    params: PARAMS_F32_GEMM,
    operands: OPS_BF16_XF32_GEMM,
    fields: FIELDS_F32_GEMM,
    hidden: HIDDEN_BF16_TILED,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_BF16_TILED,
    exports: &[ExportV1 {
        symbol: "gemm_bf16_xf32_tiled_rows",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gemm_bf16_xf32_tiled_rows",
            source: "kernels/src/gemm_bf16_xf32_tiled_rows.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 16384,
        grid: [Expr::DivCeil(&P_M, 128), Expr::DivCeil(&P_N, 4), ONE],
        consts: &[("NT", 4), ("MPW", 16)],
    }],
};

/// `gemm_bf16_xf32_tiled_rows_mp4` (MPW=4, `M <= 32`; grid x = ceil(M/32)): same object and layout as
/// [`GEMM_BF16_XF32_TILED_ROWS`] (F32-sibling evidence `gemm_f32_tiled_rows.e23c665b964e8f70.hsaco`, entry
/// `_mp4`: segment 296, LDS 16384, max flat 256, identical hidden block).
pub static GEMM_BF16_XF32_TILED_ROWS_MP4: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.bf16_xf32_tiled_rows.mpw4",
    version: 1,
    wave: 32,
    numeric: NUMERIC_F32_LANE_STRIDED,
    params: PARAMS_F32_GEMM,
    operands: OPS_BF16_XF32_GEMM,
    fields: FIELDS_F32_GEMM,
    hidden: HIDDEN_BF16_TILED,
    explicit_bytes: 36,
    segment_bytes: 296,
    segment_align: 8,
    domain: DOMAIN_BF16_TILED_MP4,
    exports: &[ExportV1 {
        symbol: "gemm_bf16_xf32_tiled_rows_mp4",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gemm_bf16_xf32_tiled_rows",
            source: "kernels/src/gemm_bf16_xf32_tiled_rows.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 16384,
        grid: [Expr::DivCeil(&P_M, 32), Expr::DivCeil(&P_N, 4), ONE],
        consts: &[("NT", 4), ("MPW", 4)],
    }],
};

/// Weight binding of the F32 GEMMs: any codec (the runtime holds F32 whatever the file codec), fused
/// sibling rows allowed (mHC phi_pre/post/z fused into one 17-row weight).
const BIND_F32_K: BindV1 = BindV1 {
    param: "K",
    source: BindSource::Weight {
        by: "M",
        by_axis: WeightAxis::Rows,
        codecs: &[],
        stacking: Stacking::Single,
        fuse_siblings: true,
    },
};

/// Admission of `gemm_f32_batched`: the F32 fallback for tensors the loader kept F32 (the selector
/// `rows_gemm` routes BF16-resident weights to the `gemm_bf16_xf32_*` kernels). It is not row-limited:
/// other launchers call `Gpu::gemm_f32_batched` directly at any row count (e.g. `hipfire-dispatch/src/families/gemm.rs`).
const ADMIT_F32_BATCHED: RowDomainV1 = RowDomainV1 {
    min_rows: 1,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/stream_mix.rs::BF16_TILED_ROWS_MIN_N",
    kill_env: None,
};

/// Admission of `gemm_bf16_xf32_batched`: the selector `rows_gemm` sends BF16-resident weights with
/// `n < BF16_TILED_ROWS_MIN_N` (512) here, and every row count on archs without the tiled kernel
/// (gfx1100) or with `HIPFIRE_BF16W_TILED_ROWS=0`; Raon's router calls it directly.
const ADMIT_BF16_XF32_BATCHED: RowDomainV1 = RowDomainV1 {
    min_rows: 1,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/stream_mix.rs::BF16_TILED_ROWS_MIN_N",
    kill_env: None,
};

const ADMIT_BF16_TILED: RowDomainV1 = RowDomainV1 {
    min_rows: 512,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/stream_mix.rs::BF16_TILED_ROWS_MIN_N",
    kill_env: Some("HIPFIRE_BF16W_TILED_ROWS"),
};

/// Weight binding of the BF16 GEMMs: the weight is BF16-resident, fused sibling rows allowed.
const BIND_BF16_K: BindV1 = BindV1 {
    param: "K",
    source: BindSource::Weight {
        by: "M",
        by_axis: WeightAxis::Rows,
        codecs: &[],
        stacking: Stacking::Single,
        fuse_siblings: true,
    },
};

pub static WORK_GEMM_F32_BATCHED: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_F32_BATCHED,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "f32",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_F32_BATCHED,
        bind: &[BIND_F32_K],
        note: "One wave per output: grid (M, N) gives M and N exactly; K from the weight. Bytes = A + B + Y extents. \
               Every output re-reads its activation row (reads are L2-served), so the true traffic exceeds the \
               operand extents: the memory floor is a lower bound. See the admit note: this is the \
               selector's n < 512 (or non-gfx1201 / kill-switch) route AND the direct-call route at any rows.",
    },
};

pub static WORK_GEMM_BF16_XF32_BATCHED: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_BF16_XF32_BATCHED,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "bf16",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_BF16_XF32_BATCHED,
        bind: &[BIND_BF16_K],
        note: "One wave per output: grid (M, N) gives M and N exactly; K from the BF16 weight. Bytes = A + B + Y \
               extents (A is 2 bytes per element). Every output re-reads its activation row (L2-served), so the \
               true traffic exceeds the operand extents. Selector `rows_gemm`'s n < 512 / non-gfx1201 / \
               kill-switch route for BF16-resident weights, and Raon's router.",
    },
};

pub static WORK_GEMM_BF16_XF32_TILED_ROWS: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_BF16_XF32_TILED_ROWS,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "bf16",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_BF16_TILED,
        bind: &[
            BindV1 {
                param: "N",
                source: BindSource::ChunkRows,
            },
            BIND_BF16_K,
        ],
        note: "Grid y = ceil(N/4) leaves N in an interval of 4: ChunkRows narrows it. M in (32, ..] takes this entry, \
               M <= 32 the mp4 entry. Admitted on gfx1201 only, at n >= 512 rows, `HIPFIRE_BF16W_TILED_ROWS=0` \
               restores gemm_bf16_xf32_batched. Bytes = A + X + Y extents.",
    },
};

pub static WORK_GEMM_BF16_XF32_TILED_ROWS_MP4: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_BF16_XF32_TILED_ROWS_MP4,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "bf16",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_BF16_TILED,
        bind: &[
            BindV1 {
                param: "N",
                source: BindSource::ChunkRows,
            },
            BIND_BF16_K,
        ],
        note: "The MPW=4 entry of the tiled kernel, taken when M <= 32 (the mHC selector, M = 17). Grid x = ceil(M/32) \
               is 1 for every admitted M, so the grid only bounds M to [1, 32]; the weight match (by M, interval \
               [1, 32]) then binds M exactly (e.g. the 17-row fused mHC phi) and K. Same row domain as \
               gemm_bf16_xf32_tiled_rows.",
    },
};

// ================================================================================================
// gemv_q8_0
// ================================================================================================

const PARAMS_GEMV_Q8: &[ParamV1] = &[
    param("M", "weight rows (grid x, one 32-lane block per row)"),
    param("K", "row length in elements, a multiple of 32"),
];

const OPS_GEMV_Q8: &[OperandV1] = &[
    OperandV1 {
        name: "A_q8",
        format: FORMAT_Q8_0_34,
        access: Access::ReadOnly,
        align: 2,
        extent: Some(Expr::Mul(&[P_M, Expr::DivExact(&P_K, 32), Expr::Const(34)])),
        nullable: false,
        note: "Q8_0 rows `[M, K/32] x 34 B`; scale read as an f16 at the block start (alignment 2)",
    },
    OperandV1 {
        name: "x",
        format: FORMAT_F32_FLAT,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activation vector `[K]`",
    },
    OperandV1 {
        name: "y",
        format: FORMAT_F32_FLAT,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[M]`, lane 0 of each block stores its row",
    },
];

const FIELDS_GEMV_Q8: &[FieldV1] = &[
    ptr_field("A_q8", 0, "A_q8"),
    ptr_field("x", 8, "x"),
    ptr_field("y", 16, "y"),
    i32_field("M", 24, P_M),
    i32_field("K", 28, P_K),
];

const DOMAIN_GEMV_Q8: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 32),
        "lane `tid` reads byte `2 + tid` of each 34 B block: K is whole Q8_0 blocks",
    ),
];

/// `gemv_q8_0`: hsaco `~/.hipfire_kernels/gfx1201/gemv_q8_0.06f3399fa85eb2e5.hsaco` (source
/// byte-identical to `kernels/src/gemv_q8_0.hip`; `7d231e0abcb9131d`, `9ae7447cd40a09a1`,
/// `e9aef808866d3fc9` and the unhashed object are the same source) and
/// `gfx1100/gemv_q8_0.8b7c06ea0d472edb.hsaco` (same source): args `A_q8@0` (read_only) `x@8` (read_only)
/// `y@16` (write_only), `M@24 K@28` by value, `.kernarg_segment_size` 32 / align 8, no hidden words,
/// LDS 0, `.max_flat_workgroup_size` 32.
pub static GEMV_Q8_0: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.gemv_q8_0.f32x",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GEMV_Q8_0,
    params: PARAMS_GEMV_Q8,
    operands: OPS_GEMV_Q8,
    fields: FIELDS_GEMV_Q8,
    hidden: &[],
    explicit_bytes: 32,
    segment_bytes: 32,
    segment_align: 8,
    domain: DOMAIN_GEMV_Q8,
    exports: &[
        ExportV1 {
            symbol: "gemv_q8_0",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "gemv_q8_0",
                source: "kernels/src/gemv_q8_0.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, ONE, ONE],
            consts: &[],
        },
        ExportV1 {
            symbol: "gemv_q8_0",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "gemv_q8_0",
                source: "kernels/src/gemv_q8_0.hip",
            },
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_M, ONE, ONE],
            consts: &[],
        },
    ],
};

pub static WORK_GEMV_Q8_0: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMV_Q8_0,
    work: WorkV1 {
        op_class: op_class::GEMV,
        codec: "q8_0",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_M, P_K])),
        flops: None,
        bytes: None,
        rows: None,
        admit: ALL_ROWS,
        bind: &[BindV1 {
            param: "K",
            source: BindSource::Weight {
                by: "M",
                by_axis: WeightAxis::Rows,
                codecs: &["q8_0"],
                stacking: Stacking::Single,
                fuse_siblings: false,
            },
        }],
        note: "One activation row. Gpu::gemv_q8_0 launches this symbol only for k > 1536 (k <= 1536 takes \
               gemv_q8_0_wide) and, on gfx1151, not at k = 320/640/2560: a K-domain predicate, no row predicate. \
               Multiply-accumulate in f32 (dequant * x), so the VALU pipe. Bytes = weight + x + y extents.",
    },
};

// ================================================================================================
// embedding_q8_batched
// ================================================================================================

const PARAMS_EMBEDDING: &[ParamV1] = &[
    param(
        "N",
        "token rows gathered (grid x, one 256-lane block per token)",
    ),
    param("dim", "row length in elements, a multiple of 32"),
    param(
        "vocab",
        "rows of the Q8_0 table (extent only; not a kernel argument)",
    ),
];

const OPS_EMBEDDING: &[OperandV1] = &[
    OperandV1 {
        name: "table",
        format: FORMAT_Q8_0_34,
        access: Access::ReadOnly,
        align: 2,
        extent: Some(Expr::Mul(&[
            P_VOCAB,
            Expr::DivExact(&P_DIM, 32),
            Expr::Const(34),
        ])),
        nullable: false,
        note: "Q8_0 table `[vocab, dim/32] x 34 B`; token ids are not range-checked on the device (each must be < vocab)",
    },
    OperandV1 {
        name: "output",
        format: FORMAT_F32_FLAT,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_DIM, Expr::Const(4)])),
        nullable: false,
        note: "F32 rows `[N, dim]`",
    },
    OperandV1 {
        name: "token_ids",
        format: FORMAT_I32_TOKEN_IDS,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, Expr::Const(4)])),
        nullable: false,
        note: "device-resident ids (hipGraph-capturable)",
    },
];

const FIELDS_EMBEDDING: &[FieldV1] = &[
    ptr_field("table", 0, "table"),
    ptr_field("output", 8, "output"),
    ptr_field("token_ids", 16, "token_ids"),
    i32_field("dim", 24, P_DIM),
];

const DOMAIN_EMBEDDING: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_N), "no tokens"),
    rule(Rule::Positive(P_DIM), "empty rows"),
    rule(Rule::Positive(P_VOCAB), "empty table"),
    rule(
        Rule::MultipleOf(P_DIM, 32),
        "the kernel decodes `dim/32` whole 34 B blocks per row",
    ),
];

/// Hidden block of `embedding_q8_batched` (explicit args end at 28, hidden words from 32): offsets from
/// the metadata of `gfx1201/embedding_q8_batched.10e558b73b1fb525.hsaco` and
/// `gfx1100/embedding_q8_batched.073248c59ae6c599.hsaco` (identical).
const HIDDEN_EMBEDDING: &[HiddenV1] = &[
    hip_hidden(32, 4, HiddenKind::BlockCount(Axis::X)),
    hip_hidden(36, 4, HiddenKind::BlockCount(Axis::Y)),
    hip_hidden(40, 4, HiddenKind::BlockCount(Axis::Z)),
    hip_hidden(44, 2, HiddenKind::GroupSize(Axis::X)),
    hip_hidden(46, 2, HiddenKind::GroupSize(Axis::Y)),
    hip_hidden(48, 2, HiddenKind::GroupSize(Axis::Z)),
    hip_hidden(50, 2, HiddenKind::Remainder(Axis::X)),
    hip_hidden(52, 2, HiddenKind::Remainder(Axis::Y)),
    hip_hidden(54, 2, HiddenKind::Remainder(Axis::Z)),
    hip_hidden(72, 8, HiddenKind::GlobalOffset(Axis::X)),
    hip_hidden(80, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hip_hidden(88, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hip_hidden(96, 2, HiddenKind::GridDims),
];

/// `embedding_q8_batched`: hsaco `~/.hipfire_kernels/gfx1201/embedding_q8_batched.10e558b73b1fb525.hsaco`
/// (source byte-identical to `kernels/src/embedding_q8_batched.hip`; `3a140ce96d9da398`,
/// `4db658ba7b86cb7c`, `83ebe7be1ba6f938` and the unhashed object are the same source) and
/// `gfx1100/embedding_q8_batched.073248c59ae6c599.hsaco`: args `table@0` (read_only) `output@8`
/// (write_only) `token_ids@16` (read_only), `dim@24` by value, 13 hidden words at 32..98,
/// `.kernarg_segment_size` 288 / align 8, LDS 0, `.max_flat_workgroup_size` 1024 (hipcc default; the launch
/// block is 256).
pub static EMBEDDING_Q8_BATCHED: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.embedding_q8_batched.f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_EMBEDDING_Q8_0,
    params: PARAMS_EMBEDDING,
    operands: OPS_EMBEDDING,
    fields: FIELDS_EMBEDDING,
    hidden: HIDDEN_EMBEDDING,
    explicit_bytes: 28,
    segment_bytes: 288,
    segment_align: 8,
    domain: DOMAIN_EMBEDDING,
    exports: &[
        ExportV1 {
            symbol: "embedding_q8_batched",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "embedding_q8_batched",
                source: "kernels/src/embedding_q8_batched.hip",
            },
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_N, ONE, ONE],
            consts: &[],
        },
        ExportV1 {
            symbol: "embedding_q8_batched",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "embedding_q8_batched",
                source: "kernels/src/embedding_q8_batched.hip",
            },
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_N, ONE, ONE],
            consts: &[],
        },
    ],
};

pub static WORK_EMBEDDING_Q8_BATCHED: WorkSpecV1 = WorkSpecV1 {
    abi: &EMBEDDING_Q8_BATCHED,
    work: WorkV1 {
        op_class: op_class::EMBEDDING,
        codec: "q8_0>f32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[P_N, Expr::DivExact(&P_DIM, 32), Expr::Const(34)]),
            Expr::Mul(&[P_N, P_DIM, Expr::Const(4)]),
            Expr::Mul(&[P_N, Expr::Const(4)]),
        ])),
        rows: Some("N"),
        admit: ALL_ROWS,
        bind: &[BindV1 {
            param: "dim",
            source: BindSource::Config(&["hidden_size", "n_embd", "d_model", "dim"]),
        }],
        note: "Rows are exact from the grid (x = N workgroups of 256). Pass/chunk anchor of the binder. Bytes = N \
               gathered table rows + F32 output + ids (the whole table is not read).",
    },
};

// ================================================================================================
// moe_grouped_qt44_{a8,iu4}_sym (gfx1201 W4A8 / W4A4 grouped routed experts)
// ================================================================================================

const PARAMS_QT44: &[ParamV1] = &[
    param(
        "M",
        "weight rows per expert; a multiple of 64 * RF (rf=1 entries: 64, rf=2 entries: 128)",
    ),
    param("K", "row length in elements, a multiple of 256"),
    param(
        "x_row_div",
        "sidecar row of a grouped position: `slot / x_row_div` (gate/up: top-k; down: 1)",
    ),
    param(
        "m_total",
        "grouped positions launched, a multiple of 16 (upper bound with sentinel tiles)",
    ),
    param(
        "x_src_rows",
        "rows of the activation sidecar (its epoch stride)",
    ),
    param(
        "n_experts",
        "length of the expert pointer table / offsets - 1",
    ),
];

const OPS_QT44_HEAD: [OperandV1; 4] = [
    OP_EXPERT_PTRS,
    OperandV1 {
        name: "expert_tile_ids",
        format: FORMAT_EXPERT_TILE_IDS_I32,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivExact(&P_M_TOTAL, 16), Expr::Const(4)])),
        nullable: false,
        note: "expert of each 16-slot tile; `-1` or `>= n_experts` makes the tile return",
    },
    OperandV1 {
        name: "expert_offsets",
        format: FORMAT_EXPERT_OFFSETS_I32,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::Add(&[P_N_EXPERTS, ONE]),
            Expr::Const(4),
        ])),
        nullable: false,
        note: "the scatter's `[E + 1]` slot prefix, every entry 16-aligned; the NT-aligned tile-run leader test reads `expert_offsets[e] >> 4`",
    },
    OperandV1 {
        name: "sorted_slot_index",
        format: FORMAT_SORTED_SLOT_INDEX_I32,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M_TOTAL, Expr::Const(4)])),
        nullable: false,
        note: "flat routed slot per grouped position, `-1` = padding lane (zero activation, +0 store)",
    },
];

const OP_QT44_Y: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_F32_N_M,
    access: Access::WriteOnly,
    align: 16,
    extent: Some(Expr::Mul(&[P_M_TOTAL, P_M, Expr::Const(4)])),
    nullable: false,
    note: "F32 `[m_total, M]` in grouped order, 16 B vector stores (M % 64 keeps each row 16 B aligned); positions >= m_total and sentinel tiles are not stored",
};

const OPS_QT44_A8: &[OperandV1] = &[
    OPS_QT44_HEAD[0],
    OPS_QT44_HEAD[1],
    OPS_QT44_HEAD[2],
    OPS_QT44_HEAD[3],
    OperandV1 {
        name: "Xq",
        format: FORMAT_BLOCK_I8_128_KB_N,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::DivExact(&P_K, 128),
            P_X_SRC_ROWS,
            Expr::Const(136),
        ])),
        nullable: false,
        note: "signed `block_i8_128` records (`float d; int s; i8 qs[128]`, 136 B), epoch-major `[K/128][x_src_rows]`; `slot / x_row_div < x_src_rows` is the caller's contract",
    },
    OP_QT44_Y,
];

const OPS_QT44_IU4: &[OperandV1] = &[
    OPS_QT44_HEAD[0],
    OPS_QT44_HEAD[1],
    OPS_QT44_HEAD[2],
    OPS_QT44_HEAD[3],
    OperandV1 {
        name: "Xq",
        format: FORMAT_BLOCK_I4_128_KB_N,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[
            Expr::DivExact(&P_K, 128),
            P_X_SRC_ROWS,
            Expr::Const(72),
        ])),
        nullable: false,
        note: "signed `block_i4_128` records (`float d; int s; u8 qs[64]`, 72 B), epoch-major `[K/128][x_src_rows]`; `slot / x_row_div < x_src_rows` is the caller's contract",
    },
    OP_QT44_Y,
];

const FIELDS_QT44: &[FieldV1] = &[
    ptr_field("expert_weight_ptrs", 0, "expert_weight_ptrs"),
    ptr_field("expert_tile_ids", 8, "expert_tile_ids"),
    ptr_field("expert_offsets", 16, "expert_offsets"),
    ptr_field("sorted_slot_index", 24, "sorted_slot_index"),
    ptr_field("Xq", 32, "Xq"),
    ptr_field("Y", 40, "Y"),
    i32_field("M", 48, P_M),
    i32_field("K", 52, P_K),
    i32_field("x_row_div", 56, P_X_ROW_DIV),
    i32_field("m_total", 60, P_M_TOTAL),
    i32_field("x_src_rows", 64, P_X_SRC_ROWS),
    i32_field("n_experts", 68, P_N_EXPERTS),
];

/// 32-bit slot-index guard of the launcher: `max(m_total, x_src_rows, M) <= i32::MAX / 2`.
const QT44_INDEX_MAX: u64 = (i32::MAX as u64) / 2;

const DOMAIN_QT44: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(Rule::Positive(P_M_TOTAL), "no grouped positions"),
    rule(Rule::Positive(P_N_EXPERTS), "empty expert table"),
    rule(Rule::Positive(P_X_SRC_ROWS), "empty activation sidecar"),
    rule(Rule::AtLeast(P_X_ROW_DIV, 1), "`slot / x_row_div`"),
    rule(
        Rule::MultipleOf(P_M, 64),
        "4 waves x 16 rows per CTA (rf=2 entries need M % 128, enforced by their exact grid division)",
    ),
    rule(
        Rule::MultipleOf(P_K, 256),
        "whole 136 B qt44 groups (two K128 epochs each)",
    ),
    rule(
        Rule::MultipleOf(P_M_TOTAL, 16),
        "16-slot tiles; the tile-run leader test indexes `expert_tile_ids[tile + i]` below `m_total / 16`",
    ),
    rule(
        Rule::AtMost(Expr::DivExact(&P_M_TOTAL, 16), GRID_Y_MAX),
        "grid y (m_total/16) must fit the HIP y-grid limit",
    ),
    rule(
        Rule::AtMost(P_M_TOTAL, QT44_INDEX_MAX),
        "32-bit slot indices (launcher guard)",
    ),
    rule(
        Rule::AtMost(P_X_SRC_ROWS, QT44_INDEX_MAX),
        "32-bit slot indices (launcher guard)",
    ),
    rule(
        Rule::AtMost(P_M, QT44_INDEX_MAX),
        "32-bit slot indices (launcher guard)",
    ),
];

const GRID_Y_QT44: Expr = Expr::DivExact(&P_M_TOTAL, 16);

const fn qt44_export(
    symbol: &'static str,
    module: &'static str,
    source: &'static str,
    grid_x: Expr,
    nt: u64,
    rf: u64,
) -> ExportV1 {
    ExportV1 {
        symbol,
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource { module, source },
        block: [128, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [grid_x, GRID_Y_QT44, ONE],
        consts: match (nt, rf) {
            (1, 1) => &[("NT", 1), ("RF", 1)],
            (2, 1) => &[("NT", 2), ("RF", 1)],
            (4, 1) => &[("NT", 4), ("RF", 1)],
            (1, 2) => &[("NT", 1), ("RF", 2)],
            (2, 2) => &[("NT", 2), ("RF", 2)],
            _ => &[],
        },
    }
}

const QT44_GRID_X_RF1: Expr = Expr::DivExact(&P_M, 64);
const QT44_GRID_X_RF2: Expr = Expr::DivExact(&P_M, 128);
const A8_SRC: &str = "kernels/src/moe_grouped_qt44_a8_sym.gfx12.hip";
const IU4_SRC: &str = "kernels/src/moe_grouped_qt44_iu4_sym.gfx12.hip";
const A8_MODULE: &str = "moe_grouped_qt44_a8_sym_gfx12";
const IU4_MODULE: &str = "moe_grouped_qt44_iu4_sym_gfx12";

/// W4A8 grouped routed experts (`Gpu::gemm_mq4g256v2_moe_grouped_a8_sym`, gfx1201 only): hsaco
/// `~/.hipfire_kernels/gfx1201/moe_grouped_qt44_a8_sym_gfx12.5ba21441111b0f59.hsaco` (source
/// byte-identical to `kernels/src/moe_grouped_qt44_a8_sym.gfx12.hip`; `746d1eeb6d8b9dd7` is another
/// build). The module holds four entries `moe_grouped_qt44_a8_sym_gfx1201_nt{1,2}r{1,2}` (`nt4r1` spills and
/// is not built), identical layout: args `expert_weight_ptrs@0 expert_tile_ids@8 expert_offsets@16
/// sorted_slot_index@24 Xq@32` (read_only) `Y@40` (write_only), `M@48 K@52 x_row_div@56 m_total@60
/// x_src_rows@64 n_experts@68` by value, `.kernarg_segment_size` 72 / align 8, no hidden words,
/// LDS 0, `.max_flat_workgroup_size` 128. Grid `(M / (64*RF), m_total / 16)`, block 128.
pub static MOE_GROUPED_QT44_A8_SYM: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.moe_grouped_qt44_a8_sym",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QT44_A8,
    params: PARAMS_QT44,
    operands: OPS_QT44_A8,
    fields: FIELDS_QT44,
    hidden: &[],
    explicit_bytes: 72,
    segment_bytes: 72,
    segment_align: 8,
    domain: DOMAIN_QT44,
    exports: &[
        qt44_export(
            "moe_grouped_qt44_a8_sym_gfx1201_nt1r1",
            A8_MODULE,
            A8_SRC,
            QT44_GRID_X_RF1,
            1,
            1,
        ),
        qt44_export(
            "moe_grouped_qt44_a8_sym_gfx1201_nt2r1",
            A8_MODULE,
            A8_SRC,
            QT44_GRID_X_RF1,
            2,
            1,
        ),
        qt44_export(
            "moe_grouped_qt44_a8_sym_gfx1201_nt1r2",
            A8_MODULE,
            A8_SRC,
            QT44_GRID_X_RF2,
            1,
            2,
        ),
        qt44_export(
            "moe_grouped_qt44_a8_sym_gfx1201_nt2r2",
            A8_MODULE,
            A8_SRC,
            QT44_GRID_X_RF2,
            2,
            2,
        ),
    ],
};

/// W4A4 grouped routed experts (`Gpu::gemm_mq4g256v2_moe_grouped_iu4_sym`, gfx1201 only): hsaco
/// `~/.hipfire_kernels/gfx1201/moe_grouped_qt44_iu4_sym_gfx12.21162c0c83b89777.hsaco` (source
/// byte-identical to `kernels/src/moe_grouped_qt44_iu4_sym.gfx12.hip`); five entries
/// `moe_grouped_qt44_iu4_sym_gfx1201_nt{1,2}r1`, `nt4r1`, `nt{1,2}r2`, each with the 72 B layout of
/// [`MOE_GROUPED_QT44_A8_SYM`] (same offsets, no hidden words, LDS 0, max flat 128).
pub static MOE_GROUPED_QT44_IU4_SYM: KernelAbiV1 = KernelAbiV1 {
    id: "generic_gemm.moe_grouped_qt44_iu4_sym",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QT44_IU4,
    params: PARAMS_QT44,
    operands: OPS_QT44_IU4,
    fields: FIELDS_QT44,
    hidden: &[],
    explicit_bytes: 72,
    segment_bytes: 72,
    segment_align: 8,
    domain: DOMAIN_QT44,
    exports: &[
        qt44_export(
            "moe_grouped_qt44_iu4_sym_gfx1201_nt1r1",
            IU4_MODULE,
            IU4_SRC,
            QT44_GRID_X_RF1,
            1,
            1,
        ),
        qt44_export(
            "moe_grouped_qt44_iu4_sym_gfx1201_nt2r1",
            IU4_MODULE,
            IU4_SRC,
            QT44_GRID_X_RF1,
            2,
            1,
        ),
        qt44_export(
            "moe_grouped_qt44_iu4_sym_gfx1201_nt4r1",
            IU4_MODULE,
            IU4_SRC,
            QT44_GRID_X_RF1,
            4,
            1,
        ),
        qt44_export(
            "moe_grouped_qt44_iu4_sym_gfx1201_nt1r2",
            IU4_MODULE,
            IU4_SRC,
            QT44_GRID_X_RF2,
            1,
            2,
        ),
        qt44_export(
            "moe_grouped_qt44_iu4_sym_gfx1201_nt2r2",
            IU4_MODULE,
            IU4_SRC,
            QT44_GRID_X_RF2,
            2,
            2,
        ),
    ],
};

const BIND_QT44: &[BindV1] = &[
    BindV1 {
        param: "K",
        source: BindSource::Weight {
            by: "M",
            by_axis: WeightAxis::Rows,
            codecs: &["mq4g256v2"],
            stacking: Stacking::Either,
            fuse_siblings: true,
        },
    },
    BindV1 {
        param: "rows",
        source: BindSource::RoutedRows {
            topk: MOE_TOPK_KEYS,
            dense: "m_total",
        },
    },
    BindV1 {
        param: "tokens",
        source: BindSource::RoutedTokens { dense: "m_total" },
    },
];

/// Row floor of the grouped symmetric quantized-activation core: every quantized prefill tier is
/// admitted at `PREFILL_QUANT_MIN_ROWS` (64) rows or more on gfx1201 (`prefill_quant_admitted`;
/// `GROUPED_MOE_IU4_MIN_ROWS` aliases it). The tier per site is a model policy (Raon:
/// `HIPFIRE_RAON_PREFILL_PRECISION`), not a generic switch, so no kill env is named here.
const ADMIT_QT44: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/q8_prefill.rs::PREFILL_QUANT_MIN_ROWS",
    kill_env: None,
};

pub static WORK_MOE_GROUPED_QT44_A8_SYM: WorkSpecV1 = WorkSpecV1 {
    abi: &MOE_GROUPED_QT44_A8_SYM,
    work: WorkV1 {
        op_class: op_class::GROUPED_GEMM,
        codec: "mq4g256v2",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_ROWS, P_M, P_K])),
        flops: None,
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256), Expr::Const(136)]),
            Expr::Mul(&[P_ROWS, Expr::DivExact(&P_K, 128), Expr::Const(136)]),
            Expr::Mul(&[P_ROWS, P_M, Expr::Const(4)]),
        ])),
        rows: Some("tokens"),
        admit: ADMIT_QT44,
        bind: BIND_QT44,
        note: "Symmetric-header qt44 experts only (`zp == -8*sc`), gfx1201. MACs = useful routed rows * M * K \
               (`m_total` is the sentinel-padded launch bound). Bytes are a LOWER bound: one expert slab + the \
               int8 sidecar rows + Y; a routed launch streams every touched expert slab. The model's per-site \
               prefill precision policy selects this tier (Raon gfx1201 default: A8 at every site). The entry \
               (nt, rf) follows `select_entry_a8`: rf=2 at m_total >= 12288 && M % 128 == 0.",
    },
};

pub static WORK_MOE_GROUPED_QT44_IU4_SYM: WorkSpecV1 = WorkSpecV1 {
    abi: &MOE_GROUPED_QT44_IU4_SYM,
    work: WorkV1 {
        op_class: op_class::GROUPED_GEMM,
        codec: "mq4g256v2",
        pipe: Pipe::WmmaIu4,
        macs: Some(Expr::Mul(&[P_ROWS, P_M, P_K])),
        flops: None,
        bytes: Some(Expr::Add(&[
            Expr::Mul(&[P_M, Expr::DivExact(&P_K, 256), Expr::Const(136)]),
            Expr::Mul(&[P_ROWS, Expr::DivExact(&P_K, 128), Expr::Const(72)]),
            Expr::Mul(&[P_ROWS, P_M, Expr::Const(4)]),
        ])),
        rows: Some("tokens"),
        admit: ADMIT_QT44,
        bind: BIND_QT44,
        note: "W4A4 twin of the A8 route (`wmma_i32_16x16x32_iu4`, `block_i4_128` sidecar, 72 B per K128 row). \
               Same binding and lower-bound byte model as the A8 contract. The entry follows `select_entry`: \
               rf=2 at m_total >= 12288 && M % 128 == 0, else nt4r1.",
    },
};

// ================================================================================================

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &GEMM_Q8_0_WMMA,
    &GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
    &GEMM_F32_BATCHED,
    &GEMM_BF16_XF32_BATCHED,
    &GEMM_BF16_XF32_TILED_ROWS,
    &GEMM_BF16_XF32_TILED_ROWS_MP4,
    &GEMV_Q8_0,
    &EMBEDDING_Q8_BATCHED,
    &MOE_GROUPED_QT44_A8_SYM,
    &MOE_GROUPED_QT44_IU4_SYM,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &WORK_GEMM_Q8_0_WMMA,
    &WORK_GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
    &WORK_GEMM_F32_BATCHED,
    &WORK_GEMM_BF16_XF32_BATCHED,
    &WORK_GEMM_BF16_XF32_TILED_ROWS,
    &WORK_GEMM_BF16_XF32_TILED_ROWS_MP4,
    &WORK_GEMV_Q8_0,
    &WORK_EMBEDDING_Q8_BATCHED,
    &WORK_MOE_GROUPED_QT44_A8_SYM,
    &WORK_MOE_GROUPED_QT44_IU4_SYM,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval;
    use std::path::Path;

    fn repo_root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives in <root>/crates/<name>")
    }

    fn export<'a>(abi: &'a KernelAbiV1, arch: Arch, symbol: &str) -> &'a ExportV1 {
        abi.export(arch, symbol)
            .unwrap_or_else(|| panic!("{} has no export {symbol}", abi.id))
    }

    /// Workgroups the rocprof work-item grid `(x, y)` of `block` stands for.
    fn workgroups(items: [u64; 2], block: [u32; 3]) -> [u64; 3] {
        assert_eq!(
            items[0] % u64::from(block[0]),
            0,
            "x items not a block multiple"
        );
        assert_eq!(
            items[1] % u64::from(block[1]),
            0,
            "y items not a block multiple"
        );
        [
            items[0] / u64::from(block[0]),
            items[1] / u64::from(block[1]),
            1,
        ]
    }

    /// Interval of the one param a grid expression pins at workgroup count `g`:
    /// `Param p -> [g, g]`, `DivCeil(p, d) -> [(g-1)d+1, g*d]`, `DivExact(p, d) -> [g*d, g*d]`,
    /// `Div(p, d) -> [g*d, g*d+d-1]`.
    fn invert(e: &Expr, g: u64) -> Option<(&'static str, u64, u64)> {
        match e {
            Expr::Param(p) => Some((p, g, g)),
            Expr::DivCeil(inner, d) => match **inner {
                Expr::Param(p) => Some((p, (g - 1) * d + 1, g * d)),
                _ => None,
            },
            Expr::DivExact(inner, d) => match **inner {
                Expr::Param(p) => Some((p, g * d, g * d)),
                _ => None,
            },
            Expr::Div(inner, d) => match **inner {
                Expr::Param(p) => Some((p, g * d, g * d + d - 1)),
                _ => None,
            },
            _ => None,
        }
    }

    fn interval(e: &Expr, g: u64, name: &str) -> (u64, u64) {
        let (p, lo, hi) = invert(e, g).unwrap_or_else(|| panic!("grid expr {e:?} not invertible"));
        assert_eq!(p, name, "grid expression pins {p}, not {name}");
        (lo, hi)
    }

    fn eval_grid(abi: &KernelAbiV1, ex: &ExportV1, params: &[(&str, u64)]) -> [u64; 3] {
        let vals: Vec<u64> = abi
            .params
            .iter()
            .map(|p| {
                params
                    .iter()
                    .find(|(n, _)| *n == p.name)
                    .unwrap_or_else(|| panic!("param {} not given", p.name))
                    .1
            })
            .collect();
        let mut out = [0u64; 3];
        for (i, g) in ex.grid.iter().enumerate() {
            out[i] = eval(g, abi, ex, &vals).unwrap();
        }
        out
    }

    /// The Raon grid inverts to an interval holding the stated param, and the stated params
    /// re-evaluate to the same workgroup counts and work-items.
    fn check_grid(
        abi: &KernelAbiV1,
        arch: Arch,
        symbol: &str,
        items: [u64; 2],
        stated: &[(&str, u64)],
        axes: [Option<&str>; 2],
    ) {
        let ex = export(abi, arch, symbol);
        let wg = workgroups(items, ex.block);
        for (axis, name) in axes.iter().enumerate() {
            if let Some(name) = name {
                let (lo, hi) = interval(&ex.grid[axis], wg[axis], name);
                let v = stated.iter().find(|(n, _)| n == name).unwrap().1;
                assert!(
                    (lo..=hi).contains(&v),
                    "{symbol}: {name} = {v} outside the inverted [{lo}, {hi}] of grid {items:?}"
                );
            }
        }
        let got = eval_grid(abi, ex, stated);
        assert_eq!(
            got, wg,
            "{symbol}: stated params do not give grid {items:?}"
        );
        assert_eq!(
            [
                got[0] * u64::from(ex.block[0]),
                got[1] * u64::from(ex.block[1])
            ],
            items
        );
    }

    #[test]
    fn every_contract_self_checks() {
        assert_eq!(ABIS.len(), 10);
        for abi in ABIS {
            if let Err(e) = abi.self_check() {
                panic!("{}: {e:#?}", abi.id);
            }
        }
    }

    #[test]
    fn every_work_spec_self_checks_and_matches_its_contract() {
        assert_eq!(WORK.len(), ABIS.len());
        for (w, a) in WORK.iter().zip(ABIS) {
            assert!(
                std::ptr::eq(w.abi, *a),
                "{}: WORK is out of order with ABIS",
                a.id
            );
            if let Err(e) = w.self_check() {
                panic!("{e}");
            }
        }
    }

    #[test]
    fn sources_exist_and_contain_their_symbols() {
        let root = repo_root();
        for abi in ABIS {
            for e in abi.exports {
                let ObjectRefV1::HipccSource { source, .. } = e.object else {
                    panic!("{}: expected a hipcc source", abi.id);
                };
                let text = std::fs::read_to_string(root.join(source))
                    .unwrap_or_else(|err| panic!("{}: {source}: {err}", abi.id));
                // The qt44 entries are macro-generated: `..._nt##NT##r##RF`.
                let needle = match e.symbol.find("_nt") {
                    Some(i) if e.symbol.starts_with("moe_grouped_qt44") => &e.symbol[..i + 3],
                    _ => e.symbol,
                };
                assert!(text.contains(needle), "{source} lacks {needle}");
            }
        }
    }

    #[test]
    fn admission_predicates_name_real_constants() {
        let root = repo_root();
        for w in WORK {
            let p = w.work.admit.predicate;
            if p.is_empty() {
                continue;
            }
            let (path, name) = p.split_once("::").expect("path::NAME");
            let text = std::fs::read_to_string(root.join(path))
                .unwrap_or_else(|e| panic!("{}: {path}: {e}", w.abi.id));
            assert!(text.contains(name), "{path} has no {name}");
            if let Some(env) = w.work.admit.kill_env {
                // The switch is read by the launcher/selector named, or the flag table it uses.
                let hit = text.contains(env)
                    || std::fs::read_to_string(
                        root.join("crates/rdna-compute/src/feature_flags.rs"),
                    )
                    .unwrap()
                    .contains(env);
                assert!(hit, "{env} not found near {path}");
            }
        }
    }

    // ---- observed Raon pp4096 grids (rocprof work-items, block) ---------------------------------

    #[test]
    fn raon_gemm_q8_0_grids_invert() {
        // (10240,32): q/k/v/g [5120, 2560] and f_b [5120, 128] (M in [5105, 5120]); N = 512 rows.
        // (96,32): wg 3 -> M in [33, 48]; (256,32): wg 8 -> M in [113, 128].
        for (items, m) in [([10240, 32], 5120), ([96, 32], 48), ([256, 32], 128)] {
            for k in [2560u64, 128] {
                check_grid(
                    &GEMM_Q8_0_WMMA,
                    Arch::Gfx1201,
                    "gemm_q8_0_wmma_gfx12",
                    items,
                    &[("M", m), ("K", k), ("N", 512)],
                    [Some("M"), Some("N")],
                );
            }
        }
        // N interval of the (x,32) rows: [497, 512].
        let ex = export(&GEMM_Q8_0_WMMA, Arch::Gfx1201, "gemm_q8_0_wmma_gfx12");
        assert_eq!(interval(&ex.grid[1], 32, "N"), (497, 512));
        assert_eq!(interval(&ex.grid[0], 320, "M"), (5105, 5120));
        // gfx1100 sibling launches the same grid under its own symbol.
        check_grid(
            &GEMM_Q8_0_WMMA,
            Arch::Gfx1100,
            "gemm_q8_0_wmma",
            [10240, 32],
            &[("M", 5120), ("K", 2560), ("N", 512)],
            [Some("M"), Some("N")],
        );
    }

    #[test]
    fn raon_moe_grouped_grids_invert() {
        let cases: [([u64; 2], u64, u64); 5] = [
            // routed gate_up: M = 2176 (fused gate+up), m_total bound in [4977, 4992]
            ([4352, 312], 2176, 4992),
            // routed down: M = 2560
            ([5120, 312], 2560, 4992),
            // shared gate & up separately: M = 1088, dense 512 rows
            ([2176, 32], 1088, 512),
            // shared down / dense down: M = 2560
            ([5120, 32], 2560, 512),
            // dense gate/up: M = 12032
            ([24064, 32], 12032, 512),
        ];
        for (items, m, m_total) in cases {
            check_grid(
                &GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
                Arch::Gfx1201,
                "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
                items,
                &[
                    ("M", m),
                    ("K", 2560),
                    ("x_row_div", 1),
                    ("m_total", m_total),
                    ("n_experts", 128),
                    ("x_src_rows", 512),
                ],
                [Some("M"), Some("m_total")],
            );
        }
        let ex = export(
            &GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
            Arch::Gfx1201,
            "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
        );
        assert_eq!(interval(&ex.grid[0], 136, "M"), (2161, 2176));
        assert_eq!(interval(&ex.grid[1], 312, "m_total"), (4977, 4992));
        assert_eq!(interval(&ex.grid[1], 32, "m_total"), (497, 512));
        // gfx1100 `_k2` sibling: same grid and block.
        check_grid(
            &GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
            Arch::Gfx1100,
            "gemm_mq4g256v2_moe_grouped_wmma_k2",
            [4352, 312],
            &[
                ("M", 2176),
                ("K", 2560),
                ("x_row_div", 6),
                ("m_total", 4992),
                ("n_experts", 128),
                ("x_src_rows", 512),
            ],
            [Some("M"), Some("m_total")],
        );
    }

    #[test]
    fn raon_f32_embedding_and_gemv_grids_invert() {
        // gemm_f32_batched one wave per output: (544, 512) = 17 x 512, (4096, 512) = 128 x 512.
        for (items, m) in [([544u64, 512], 17u64), ([4096, 512], 128)] {
            for arch in [Arch::Gfx1201, Arch::Gfx1100] {
                check_grid(
                    &GEMM_F32_BATCHED,
                    arch,
                    "gemm_f32_batched",
                    items,
                    &[("M", m), ("K", 10240), ("N", 512)],
                    [Some("M"), Some("N")],
                );
            }
        }
        for arch in [Arch::Gfx1201, Arch::Gfx1100] {
            check_grid(
                &GEMM_BF16_XF32_BATCHED,
                arch,
                "gemm_bf16_xf32_batched",
                [544, 512],
                &[("M", 17), ("K", 10240), ("N", 512)],
                [Some("M"), Some("N")],
            );
        }
        // Tiled entries on the same shapes: router M = 128 (MPW16), selector M = 17 (MPW4), N = 512.
        check_grid(
            &GEMM_BF16_XF32_TILED_ROWS,
            Arch::Gfx1201,
            "gemm_bf16_xf32_tiled_rows",
            [256, 128],
            &[("M", 128), ("K", 2560), ("N", 512)],
            [Some("M"), Some("N")],
        );
        check_grid(
            &GEMM_BF16_XF32_TILED_ROWS_MP4,
            Arch::Gfx1201,
            "gemm_bf16_xf32_tiled_rows_mp4",
            [256, 128],
            &[("M", 17), ("K", 10240), ("N", 512)],
            [Some("M"), Some("N")],
        );
        // embedding (131072, 1) block 256 -> 512 rows exactly, on both archs.
        for arch in [Arch::Gfx1201, Arch::Gfx1100] {
            check_grid(
                &EMBEDDING_Q8_BATCHED,
                arch,
                "embedding_q8_batched",
                [131072, 1],
                &[("N", 512), ("dim", 2560), ("vocab", 152064)],
                [Some("N"), None],
            );
        }
        let ex = export(&EMBEDDING_Q8_BATCHED, Arch::Gfx1201, "embedding_q8_batched");
        assert_eq!(interval(&ex.grid[0], 512, "N"), (512, 512));
        // gemv_q8_0 (M workgroups of 32).
        check_grid(
            &GEMV_Q8_0,
            Arch::Gfx1201,
            "gemv_q8_0",
            [248832 * 32, 1],
            &[("M", 248832), ("K", 2560)],
            [Some("M"), None],
        );
    }

    #[test]
    fn qt44_grids_invert_per_entry() {
        // gate_up M = 2176 and down M = 2560, m_total 25552 + sentinel padding = 25600.
        let base = |m: u64| {
            [
                ("M", m),
                ("K", 2560),
                ("x_row_div", 1),
                ("m_total", 25600),
                ("x_src_rows", 4096),
                ("n_experts", 128),
            ]
        };
        for abi in [&MOE_GROUPED_QT44_A8_SYM, &MOE_GROUPED_QT44_IU4_SYM] {
            for e in abi.exports {
                let rf: u64 = e.consts.iter().find(|(n, _)| *n == "RF").unwrap().1;
                for m in [2176u64, 2560] {
                    if m % (64 * rf) != 0 {
                        continue;
                    }
                    let wg = eval_grid(abi, e, &base(m));
                    assert_eq!(wg, [m / (64 * rf), 1600, 1], "{}", e.symbol);
                    assert_eq!(e.block, [128, 1, 1]);
                    // inverts exactly (DivExact).
                    assert_eq!(interval(&e.grid[0], wg[0], "M"), (m, m));
                    assert_eq!(interval(&e.grid[1], wg[1], "m_total"), (25600, 25600));
                }
                // rf = 2 entries reject an M that is only a multiple of 64 (exact division).
                if rf == 2 {
                    let vals = [2112u64, 2560, 1, 25600, 4096, 128];
                    assert!(eval(&e.grid[0], abi, e, &vals).is_err());
                }
            }
        }
        assert_eq!(MOE_GROUPED_QT44_A8_SYM.exports.len(), 4);
        assert_eq!(MOE_GROUPED_QT44_IU4_SYM.exports.len(), 5);
    }

    // ---- layouts equal the inspected metadata --------------------------------------------------

    #[test]
    fn kernarg_layouts_match_inspected_metadata() {
        // (contract, explicit bytes, segment bytes, field (offset, width) list), from llvm-readelf.
        let ptr_scalar = |np: u32, ns: u32| -> Vec<(u32, u32)> {
            (0..np)
                .map(|i| (8 * i, 8))
                .chain((0..ns).map(|i| (8 * np + 4 * i, 4)))
                .collect()
        };
        let cases: [(&KernelAbiV1, u32, u32, Vec<(u32, u32)>); 10] = [
            (&GEMM_Q8_0_WMMA, 36, 36, ptr_scalar(3, 3)),
            (&GEMM_MQ4G256V2_MOE_GROUPED_WMMA, 60, 60, ptr_scalar(5, 5)),
            (&GEMM_F32_BATCHED, 36, 36, ptr_scalar(3, 3)),
            (&GEMM_BF16_XF32_BATCHED, 36, 36, ptr_scalar(3, 3)),
            (&GEMM_BF16_XF32_TILED_ROWS, 36, 296, ptr_scalar(3, 3)),
            (&GEMM_BF16_XF32_TILED_ROWS_MP4, 36, 296, ptr_scalar(3, 3)),
            (&GEMV_Q8_0, 32, 32, ptr_scalar(3, 2)),
            (&EMBEDDING_Q8_BATCHED, 28, 288, ptr_scalar(3, 1)),
            (&MOE_GROUPED_QT44_A8_SYM, 72, 72, ptr_scalar(6, 6)),
            (&MOE_GROUPED_QT44_IU4_SYM, 72, 72, ptr_scalar(6, 6)),
        ];
        for (abi, explicit, segment, fields) in cases {
            assert_eq!(abi.explicit_bytes, explicit, "{}", abi.id);
            assert_eq!(abi.segment_bytes, segment, "{}", abi.id);
            assert_eq!(abi.segment_align, 8, "{}", abi.id);
            let got: Vec<(u32, u32)> = abi.fields.iter().map(|f| (f.offset, f.width)).collect();
            assert_eq!(got, fields, "{}", abi.id);
        }
        // Hidden blocks: 13 words of the hipcc implicit-arg block, as the metadata lists them.
        assert!(GEMM_Q8_0_WMMA.hidden.is_empty() && GEMV_Q8_0.hidden.is_empty());
        assert_eq!(GEMM_BF16_XF32_TILED_ROWS.hidden.len(), 13);
        assert_eq!(GEMM_BF16_XF32_TILED_ROWS.hidden[0].offset, 40);
        assert_eq!(GEMM_BF16_XF32_TILED_ROWS.hidden[12].offset, 104);
        assert_eq!(EMBEDDING_Q8_BATCHED.hidden.len(), 13);
        assert_eq!(EMBEDDING_Q8_BATCHED.hidden[0].offset, 32);
        assert_eq!(EMBEDDING_Q8_BATCHED.hidden[12].offset, 96);
        for e in GEMM_BF16_XF32_TILED_ROWS
            .exports
            .iter()
            .chain(GEMM_BF16_XF32_TILED_ROWS_MP4.exports)
        {
            assert_eq!(e.group_segment_fixed, 16384, "{}", e.symbol);
            assert_eq!(e.block, [256, 1, 1]);
        }
    }

    // ---- work specs ----------------------------------------------------------------------------

    #[test]
    fn bf16_tiled_admission_is_the_selector_floor() {
        for w in [
            &WORK_GEMM_BF16_XF32_TILED_ROWS,
            &WORK_GEMM_BF16_XF32_TILED_ROWS_MP4,
        ] {
            assert_eq!(w.work.admit.min_rows, 512);
            assert_eq!(w.work.admit.max_rows, None);
            assert_eq!(w.work.admit.kill_env, Some("HIPFIRE_BF16W_TILED_ROWS"));
            assert!(w
                .work
                .admit
                .predicate
                .ends_with("stream_mix.rs::BF16_TILED_ROWS_MIN_N"));
        }
        // The batched kernel is admitted at every row count (direct callers), the selector named.
        assert_eq!(WORK_GEMM_F32_BATCHED.work.admit.min_rows, 1);
        assert_eq!(WORK_GEMM_F32_BATCHED.work.admit.max_rows, None);
        assert_eq!(WORK_GEMM_BF16_XF32_BATCHED.work.admit.min_rows, 1);
        assert_eq!(WORK_GEMM_BF16_XF32_BATCHED.work.admit.max_rows, None);
        // The source constant the contract mirrors.
        let text = std::fs::read_to_string(
            repo_root().join("crates/hipfire-dispatch/src/pipeline/stream_mix.rs"),
        )
        .unwrap();
        assert!(text.contains("BF16_TILED_ROWS_MIN_N: usize = 512;"));
    }

    #[test]
    fn work_macs_and_bytes_evaluate_on_the_raon_shapes() {
        let ev =
            |w: &WorkSpecV1, e: Expr, ex: &ExportV1, p: &[u64]| eval(&e, w.abi, ex, p).unwrap();
        // gemm_q8_0 q-proj: M=5120 K=2560 N=512.
        let w = &WORK_GEMM_Q8_0_WMMA;
        let ex = &w.abi.exports[0];
        assert_eq!(
            ev(w, w.work.macs.unwrap(), ex, &[5120, 2560, 512]),
            5120 * 2560 * 512
        );
        // grouped: rows is a bound work param, so evaluate with the extended parameter vector the
        // binder supplies (params in declaration order + bound params): use a Derived-free check of names.
        let w = &WORK_GEMM_MQ4G256V2_MOE_GROUPED_WMMA;
        // The row class is the token rows; the MACs count the routed slots (`rows`).
        assert_eq!(w.work.rows, Some("tokens"));
        assert!(w
            .work
            .bind
            .iter()
            .any(|b| matches!(b.source, BindSource::RoutedTokens { dense: "m_total" })));
        assert!(w.work.bind.iter().any(|b| matches!(
            b.source,
            BindSource::RoutedRows {
                dense: "m_total",
                ..
            }
        )));
        // embedding bytes: 512 rows x dim 2560 -> 512*(80*34) + 512*2560*4 + 512*4.
        let w = &WORK_EMBEDDING_Q8_BATCHED;
        let ex = &w.abi.exports[0];
        assert_eq!(
            ev(w, w.work.bytes.unwrap(), ex, &[512, 2560, 152064]),
            512 * 80 * 34 + 512 * 2560 * 4 + 512 * 4
        );
        assert_eq!(w.work.op_class, op_class::EMBEDDING);
        assert_eq!(w.work.rows, Some("N"));
        // gemv macs.
        let w = &WORK_GEMV_Q8_0;
        let ex = &w.abi.exports[0];
        assert_eq!(
            ev(w, w.work.macs.unwrap(), ex, &[248832, 2560]),
            248832 * 2560
        );
    }

    #[test]
    fn grouped_and_w4a8_bindings_follow_the_spec() {
        for w in [
            &WORK_GEMM_MQ4G256V2_MOE_GROUPED_WMMA,
            &WORK_MOE_GROUPED_QT44_A8_SYM,
            &WORK_MOE_GROUPED_QT44_IU4_SYM,
        ] {
            assert_eq!(w.work.op_class, op_class::GROUPED_GEMM);
            assert_eq!(w.work.codec, "mq4g256v2");
            let Some(BindSource::Weight {
                by: "M",
                by_axis: WeightAxis::Rows,
                codecs: &["mq4g256v2"],
                stacking: Stacking::Either,
                fuse_siblings: true,
            }) = w.work.bind.first().map(|b| b.source)
            else {
                panic!("{}: first bind is not the grouped weight match", w.abi.id);
            };
            assert_eq!(w.work.bind[0].param, "K");
            let BindSource::RoutedRows { topk, dense } = w.work.bind[1].source else {
                panic!("{}: second bind is not RoutedRows", w.abi.id);
            };
            assert_eq!(dense, "m_total");
            assert!(topk.contains(&"num_experts_per_tok") && topk.contains(&"top_k"));
        }
        assert_eq!(WORK_MOE_GROUPED_QT44_A8_SYM.work.pipe, Pipe::WmmaIu8);
        assert_eq!(WORK_MOE_GROUPED_QT44_IU4_SYM.work.pipe, Pipe::WmmaIu4);
        assert_eq!(
            WORK_GEMM_MQ4G256V2_MOE_GROUPED_WMMA.work.pipe,
            Pipe::WmmaF16
        );
        assert_eq!(WORK_GEMM_Q8_0_WMMA.work.pipe, Pipe::WmmaF16);
        assert_eq!(WORK_GEMM_F32_BATCHED.work.pipe, Pipe::ValuF32);
        // fused phi (17 rows, mixed codecs) and the router match any codec with sibling fusion.
        let BindSource::Weight {
            codecs,
            fuse_siblings,
            ..
        } = WORK_GEMM_F32_BATCHED.work.bind[0].source
        else {
            panic!("f32 batched binds the weight");
        };
        assert!(codecs.is_empty() && fuse_siblings);
    }
}
