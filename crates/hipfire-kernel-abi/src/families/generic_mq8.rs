// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ8G256 (qt14) prefill: the batched activation producer and the int8 MMQ GEMMs on gfx1201 and
//! gfx1100 (hipcc).
//!
//! Seven contracts, each with a [`WorkSpecV1`] sidecar:
//!
//! | contract | exports | launcher |
//! |---|---|---|
//! | [`MQ8_ROTATE_QUANTIZE_X_BATCHED`] | `mq8_rotate_quantize_x_batched` (gfx1201, gfx1100) | `Gpu::mq8_quantize_x` (`crates/rdna-compute/src/gemm_mq8g256.rs`) |
//! | [`GEMM_MQ8G256_MMQ_I8_WA`] | `gemm_mq8g256_mmq_i8_t{128x128,128x64,64x64}_wa` (gfx1201), `gemm_mq8g256_mmq_i8_gfx11_t{128x128,64x128,64x64}_wa` (gfx1100) | `Gpu::gemm_mq8g256_gemm*` with `Mq8WLayout::A`, `bf16_out = false` |
//! | [`GEMM_MQ8G256_MMQ_I8_WA_BF16`] | same symbols with a `_bf16` suffix | same, `bf16_out = true` |
//! | [`GEMM_MQ8G256_MMQ_I8_WB`] | `..._wb` | `Mq8WLayout::B`, `bf16_out = false` |
//! | [`GEMM_MQ8G256_MMQ_I8_WB_BF16`] | `..._wb_bf16` | `Mq8WLayout::B`, `bf16_out = true` |
//! | [`GEMV_MQ8G256_B`] | `gemv_mq8g256_b_r{1..=16}_u1_w4_p1` (gfx1201, gfx1100) | `Gpu::gemv_mq8g256_b`, `Gpu::gemv_mq8g256_rows_exact_b` (`crates/rdna-compute/src/gemv_mq8_rows.rs`) |
//! | [`GEMV_MQ8G256_BX`] | `gemv_mq8g256_bx_r{1,2}_u{1,5,10}_w{10,24,20}_p1` (gfx1201, gfx1100) | `Gpu::gemv_mq8g256_b_x` (`crates/rdna-compute/src/gemv_mq8_rows.rs`) |
//!
//! Layouts:
//! * **Layout A** ([`FORMAT_MQ8G256_Q14_258`]): the artifact's qt=14 rows `[M, K/256]` of 258 B groups
//!   (`f16 scale, int8 q[256]`) read in place, 2-byte aligned base.
//! * **Layout B** ([`FORMAT_MQ8G256_B_PLANES`]): the aligned repack (`mq8g256_repack_b`): f16 scale plane
//!   `[M, K/256]`, zero padding to the next 256 B boundary, then the row-major int8 code plane `[M, K]`
//!   (`round_up(M*K/256*2, 256) + M*K` bytes), 16-byte aligned base.
//! * **X plane** (producer output = GEMM input): int8 codes `[N, K]` ([`FORMAT_I8_N_K`], codes in
//!   `[-128, 127]`) plus one F32 scale per (token, 256-group), `[K/256, N]` ([`FORMAT_F32_KG256_N_SCALES`],
//!   `Xs[group * N + token]`).
//!
//! The `_bf16` GEMMs do NOT write BF16 words: they round every output to BF16 in the epilogue and still store
//! an F32 plane (`y.dtype == F32`, `n * m * 4` bytes), so they are a different numeric contract with the same
//! operands. Layouts A and B differ in the weight operand only (format, alignment, extent) and the kernels share
//! one fold; a contract has exactly one numeric id and one weight operand, hence one contract per (layout,
//! epilogue).
//!
//! Evidence: no offline object exists (hipcc-built at load); `rdna-compute/tests/kernel_abi_drift.rs::
//! hipcc_exports_match_compiled_metadata` compiles every export and checks explicit args, hidden words,
//! `.group_segment_fixed_size` and `.kernarg_segment_size` against the compiled metadata. Every kernel is built for wave 32; each GEMM contract lists 3 tiles per arch (6 exports).
//! gfx1151 builds the gfx11 source too (the launcher admits it) but is not an export here: no gfx1151 object
//! has been checked against these contracts.

use crate::work::{
    op_class, BindSource, BindV1, Pipe, RowDomainV1, Stacking, WeightAxis, WorkSpecV1, WorkV1,
};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::generic_elem::FORMAT_MQ_FWHT_SIGNS_256;
use super::generic_mmq::FORMAT_I8_N_K;
use super::ggml_convert::FORMAT_F32_ROWS_NK;
use super::mq6_i8::FORMAT_F32_N_M;

// ---- formats -----------------------------------------------------------------

/// MQ8G256 (qt=14) weight rows `[M, K/256]`, 258 B per 256-group: f16 scale then int8 `q[256]` (codes in
/// `[-128, 127]`), row stride `K/256 * 258`, read in place ("layout A").
pub const FORMAT_MQ8G256_Q14_258: FormatId = FormatId("mq8g256.rows_x_groups.258b");
/// MQ8G256 aligned repack ("layout B"): f16 scale plane `[M, K/256]` (row-major, the stored f16 bits verbatim),
/// zero padding to `round_up(M*K/256*2, 256)`, then the int8 code plane `[M, K]` row-major.
pub const FORMAT_MQ8G256_B_PLANES: FormatId =
    FormatId("mq8g256.layout_b.f16_scale_plane.i8_code_plane");
/// F32 per-(256-group, token) activation scales `[K/256, N]`, group-major (`Xs[group * N + token]`):
/// `amax / 127` of the rotated group, 1.0 for an all-zero group.
pub const FORMAT_F32_KG256_N_SCALES: FormatId = FormatId("f32.kg256_x_n.mq8_x_scales");

// ---- numeric contracts ---------------------------------------------------------

/// MQ8G256 weight x int8 per-(token, 256-group) activation: per K128 epoch `e` (ascending) an exact int32 dot
/// of the stored int8 codes (`|dot128| <= 2^21`; WMMA/MMQ iu8, signed x signed, accumulator seeded `0x4B400000`
/// and recovered by the exact float subtract of `12582912.0f`), folded as
/// `acc = fmaf(RN(sw[m, e/2] * dx[n, e/2]), (float)dot128_e, acc)` from `acc = +0.0f`: `sw` = the f16 group
/// scale widened to F32, `dx` = `Xs[(e/2) * N + n]`, the product its own F32 rounding, ONE scale pair feeding
/// the two K128 epochs of a 256-group. One plain F32 store per output; identical for every tile and for both
/// weight layouts.
pub const NUMERIC_MQ8G256_X_I8G256_MMQ_F32: NumericContractId = NumericContractId(
    "generic_mq8.mq8g256_x_i8grp256.mmq_iu8.exact_dot128_fma_fold_rn_scale_pair.f32",
);
/// [`NUMERIC_MQ8G256_X_I8G256_MMQ_F32`] followed, per output, by RN-even rounding to BF16 (non-finite values
/// unchanged), stored widened as F32.
pub const NUMERIC_MQ8G256_X_I8G256_MMQ_BF16_RNE: NumericContractId = NumericContractId(
    "generic_mq8.mq8g256_x_i8grp256.mmq_iu8.exact_dot128_fma_fold_rn_scale_pair.bf16_rne_in_f32",
);
/// F32 -> int8 per token per 256-group, `mq8_rotate_quantize_x` operation for operation: `v = x * signs1`,
/// FWHT-256 (local butterflies over strides 1/2/4 of a lane's 8 elements, `ds_swizzle` butterflies over lane
/// strides 1..16), `v *= 0.0625f * signs2`, `amax = max |v|` (`fmaxf`), `scale = amax > 0 ? amax / 127.0f : 1.0f`
/// and `inv = amax > 0 ? 127.0f / amax : 0.0f` (IEEE divides), `q = clamp(rintf(v * inv), -128, 127)` (ties to
/// even); codes stored in K order, scale to `[K/256, N]`.
pub const NUMERIC_MQ8_ROTATE_QUANTIZE_X_BATCHED: NumericContractId = NumericContractId(
    "generic_mq8.mq8_rotate_quantize_x_batched.f32_signfwht256_to_i8.amax127_rint_clamp128",
);
/// MQ8G256 layout-B decode GEMV (R = 1) and verify rows-exact (R = 2..=16) over pre-quantized int8 X rows: per
/// (lane, group pair) an exact int32 `sdot4` x4 over 16 contiguous code bytes, `acc = fmaf(RN(ws * xs),
/// (float)dot, acc)`, pairs ascending, ONE chain per lane; 16-lane xor butterfly 1, 2, 4, 8 per half, then
/// `T0 + T1` at lane 0. Shared by all 32 symbols (bit-identical per row by construction).
pub const NUMERIC_GEMV_MQ8G256_B: NumericContractId =
    NumericContractId("generic_mq8.gemv_mq8g256_b.i8_pair_lane_fma.half_butterfly16_t0_t1");
/// MQ8G256 fused-producer layout-B GEMV (R = 1..=2): the F32 activation rows are rotated and quantized in-kernel
/// with [`NUMERIC_MQ8_ROTATE_QUANTIZE_X_BATCHED`]'s math (sign-FWHT256 via DPP / `v_permlanex16` lane butterflies
/// written as `p + (+-v)`, the same IEEE results; bit-identical codes and scales), then reduced exactly as
/// [`NUMERIC_GEMV_MQ8G256_B`].
pub const NUMERIC_GEMV_MQ8G256_BX: NumericContractId =
    NumericContractId("generic_mq8.gemv_mq8g256_bx.fused_rotq256_then_pair_lane_fma");

// ---- shared pieces ---------------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_N: Expr = Expr::Param("N");
const P_BLOCKS: Expr = Expr::Param("blocks");
const ONE: Expr = Expr::Const(1);
/// HIP grid-y limit (65535 workgroups).
const GRID_Y_MAX: u64 = 65535;
/// Narrowest token tile of the family (`BN` of the 128x64 / 64x64 entries): the grid-y bound of the contract.
const BN_MIN: u64 = 64;

const MODULE_PRODUCER: &str = "mq8_rotate_quantize_x_batched";
const SOURCE_PRODUCER: &str = "kernels/src/mq8_rotate_quantize_x_batched.hip";
const MODULE_GFX12: &str = "gemm_mq8g256_mmq_i8_gfx12";
const SOURCE_GFX12: &str = "kernels/src/gemm_mq8g256_mmq_i8.gfx12.hip";
const MODULE_GFX11: &str = "gemm_mq8g256_mmq_i8_gfx11";
const SOURCE_GFX11: &str = "kernels/src/gemm_mq8g256_mmq_i8.gfx11.hip";

const OBJ_PRODUCER: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: MODULE_PRODUCER,
    source: SOURCE_PRODUCER,
};
const OBJ_GFX12: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: MODULE_GFX12,
    source: SOURCE_GFX12,
};
const OBJ_GFX11: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: MODULE_GFX11,
    source: SOURCE_GFX11,
};

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

/// Weight binding of the MQ8G256 GEMMs: the grid pins `M` to a BM-row interval, the match against the qt14
/// weight rows binds `M` exactly and `K` to the weight's columns. The tensor directory names qt14 `mq8g256`
/// (`pm-census::model::codec_name(14)`), for both on-GPU layouts (layout B is a load-time repack of the same
/// tensor).
const BIND_MQ8_K: BindV1 = BindV1 {
    param: "K",
    source: BindSource::Weight {
        by: "M",
        by_axis: WeightAxis::Rows,
        codecs: &["mq8g256"],
        stacking: Stacking::Single,
        fuse_siblings: false,
    },
};

/// Row floor of every quantized prefill tier (`prefill_quant_admitted`: `rows >= PREFILL_QUANT_MIN_ROWS` = 64 and
/// no graph capture), the same floor as the W8A8 Q8_0 route this one replaces for the Raon attention class.
const ADMIT_MQ8: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/q8_prefill.rs::PREFILL_QUANT_MIN_ROWS",
    kill_env: None,
};

// ================================================================================================
// mq8_rotate_quantize_x_batched
// ================================================================================================

const PARAMS_PRODUCER: &[ParamV1] = &[
    param("K", "row length in elements, a multiple of 256"),
    param("N", "activation rows (folded into grid x with K)"),
    param_u32(
        "blocks",
        "grid x (workgroups): ceil(N * (K/256) / 8) (not a kernarg; the folded 1-D launch is the only thing the grid determines)",
    ),
];

const N_GROUPS: Expr = Expr::Mul(&[P_N, Expr::DivExact(&P_K, 256)]);
const PRODUCER_BLOCKS: Expr = Expr::DivCeil(&N_GROUPS, 8);
const SIGNS_EXTENT: Expr = Expr::Const(256 * 4);

const OPS_PRODUCER: &[OperandV1] = &[
    OperandV1 {
        name: "X",
        format: FORMAT_F32_ROWS_NK,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[N, K]`, one wave32 reads one (token, 256-group) as 8 contiguous elements per lane (two float4); waves past N*K/256 read nothing",
    },
    OperandV1 {
        name: "Xq",
        format: FORMAT_I8_N_K,
        access: Access::WriteOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_N, P_K])),
        nullable: false,
        note: "int8 codes `[N, K]` in `[-128, 127]`, one uint2 (8 codes) stored per lane; the GEMMs read them as 16 B vectors",
    },
    OperandV1 {
        name: "Xs",
        format: FORMAT_F32_KG256_N_SCALES,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivExact(&P_K, 256), P_N, Expr::Const(4)])),
        nullable: false,
        note: "F32 scales `[K/256, N]` (group-major, `Xs[group * N + token]`), written by lane 0 of each wave; zero group -> 1.0",
    },
    OperandV1 {
        name: "signs1",
        format: FORMAT_MQ_FWHT_SIGNS_256,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(SIGNS_EXTENT),
        nullable: false,
        note: "seed-42 FWHT256 sign table (`scratch.mq_signs1`), read as float4",
    },
    OperandV1 {
        name: "signs2",
        format: FORMAT_MQ_FWHT_SIGNS_256,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(SIGNS_EXTENT),
        nullable: false,
        note: "seed-1042 FWHT256 sign table (`scratch.mq_signs2`), read as float4",
    },
];

const FIELDS_PRODUCER: &[FieldV1] = &[
    ptr_field("X", 0, "X"),
    ptr_field("Xq", 8, "Xq"),
    ptr_field("Xs", 16, "Xs"),
    ptr_field("signs1", 24, "signs1"),
    ptr_field("signs2", 32, "signs2"),
    i32_field("K", 40, P_K),
    i32_field("N", 44, P_N),
];

/// The kernel reads `blockDim` (hidden words 48..114): the hipcc layout after 48 explicit bytes (block counts
/// at the end of the explicit args, then group sizes, remainders, a 16 B gap, global offsets, grid dims), the
/// same shape as [`super::generic_mmq::QUANTIZE_X_Q8_BLK32`] shifted by the 16 extra explicit bytes. Checked
/// against the compiled metadata by `hipcc_exports_match_compiled_metadata`.
const HIDDEN_PRODUCER: &[HiddenV1] = &[
    hip_hidden(48, 4, HiddenKind::BlockCount(Axis::X)),
    hip_hidden(52, 4, HiddenKind::BlockCount(Axis::Y)),
    hip_hidden(56, 4, HiddenKind::BlockCount(Axis::Z)),
    hip_hidden(60, 2, HiddenKind::GroupSize(Axis::X)),
    hip_hidden(62, 2, HiddenKind::GroupSize(Axis::Y)),
    hip_hidden(64, 2, HiddenKind::GroupSize(Axis::Z)),
    hip_hidden(66, 2, HiddenKind::Remainder(Axis::X)),
    hip_hidden(68, 2, HiddenKind::Remainder(Axis::Y)),
    hip_hidden(70, 2, HiddenKind::Remainder(Axis::Z)),
    hip_hidden(88, 8, HiddenKind::GlobalOffset(Axis::X)),
    hip_hidden(96, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hip_hidden(104, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hip_hidden(112, 2, HiddenKind::GridDims),
];

const DOMAIN_PRODUCER: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_K), "empty rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::MultipleOf(P_K, 256),
        "one wave32 is one whole 256-group (8 elements per lane): K is whole groups",
    ),
    rule(
        Rule::Equal(P_BLOCKS, PRODUCER_BLOCKS),
        "launcher: grid x = ceil(n * (k/256) / 8), one wave32 per (token, 256-group), 8 waves per block of 256",
    ),
    rule(
        Rule::AtMost(N_GROUPS, i32::MAX as u64),
        "the kernel indexes (token, group) waves with 32-bit group arithmetic: N*(K/256) < 2^31 (the launcher errors beyond it)",
    ),
    rule(
        Rule::AtMost(P_BLOCKS, u32::MAX as u64),
        "grid x is a u32 (the launcher errors beyond it)",
    ),
];

/// `mq8_rotate_quantize_x_batched`: the activation producer of the MQ8G256 prefill GEMMs. Source
/// `kernels/src/mq8_rotate_quantize_x_batched.hip` (one portable wave32 source: the same symbol and layout on
/// gfx1201 and gfx1100): args `X@0` (`read_only`) `Xq@8 Xs@16` (`write_only`) `signs1@24 signs2@32` (`read_only`),
/// `K@40 N@44` by value, hidden words at 48..114 (`GridDims` at 112), LDS 0, launch block 256, grid
/// `[ceil(N*(K/256)/8), 1, 1]` (`Gpu::mq8_quantize_x`).
pub static MQ8_ROTATE_QUANTIZE_X_BATCHED: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mq8.mq8_rotate_quantize_x_batched.f32_to_i8_f32scales",
    version: 1,
    wave: 32,
    numeric: NUMERIC_MQ8_ROTATE_QUANTIZE_X_BATCHED,
    params: PARAMS_PRODUCER,
    operands: OPS_PRODUCER,
    fields: FIELDS_PRODUCER,
    hidden: HIDDEN_PRODUCER,
    explicit_bytes: 48,
    segment_bytes: 304,
    segment_align: 8,
    domain: DOMAIN_PRODUCER,
    exports: &[
        ExportV1 {
            symbol: "mq8_rotate_quantize_x_batched",
            arch: Arch::Gfx1201,
            object: OBJ_PRODUCER,
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_BLOCKS, ONE, ONE],
            consts: &[],
        },
        ExportV1 {
            symbol: "mq8_rotate_quantize_x_batched",
            arch: Arch::Gfx1100,
            object: OBJ_PRODUCER,
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [P_BLOCKS, ONE, ONE],
            consts: &[],
        },
    ],
};

pub static WORK_MQ8_ROTATE_QUANTIZE_X_BATCHED: WorkSpecV1 = WorkSpecV1 {
    abi: &MQ8_ROTATE_QUANTIZE_X_BATCHED,
    work: WorkV1 {
        op_class: op_class::ROTATE,
        codec: "f32>i8_g256",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        // Per workgroup: 8 waves x one 256-group = 2048 elements: F32 read 8192 B + int8 write 2048 B + 8 F32 scales
        // 32 B (the two 1 KiB sign tables stay in cache and are not charged).
        bytes: Some(Expr::Mul(&[P_BLOCKS, Expr::Const(10272)])),
        rows: None,
        admit: ADMIT_MQ8,
        bind: &[],
        note: "Memory-bound producer: bytes = blocks * 10272 (per 2048-element workgroup: 8192 B F32 in + 2048 B int8 out + 32 B \
               scales out; the sign tables are cache-resident), exact when N*K % 2048 == 0 (every Raon grid: 5120 = 4096x2560, \
               10240 = 4096x5120, 128 = 512x512, 1280 = 4096x512) and otherwise an over-count of less than one tile. The VALU \
               work (sign multiply, FWHT-256 butterflies, abs/max, two IEEE divides + rint + clamp per element) is not charged. The grid \
               fixes only N*K/256, never N alone (the 512-row MLA tile launches are not the 4096-row chunk), so no rows param is bound; \
               admission follows the quantized prefill row floor (>= 64 rows).",
    },
};

// ================================================================================================
// gemm_mq8g256_mmq_i8_*_{wa,wb}[_bf16]
// ================================================================================================

const PARAMS_GEMM: &[ParamV1] = &[
    param(
        "M",
        "weight rows (output features; grid x x BM, tail rows re-read row M-1 and are masked)",
    ),
    param(
        "K",
        "row length in elements, a multiple of 256 (whole MQ8G256 groups)",
    ),
    param(
        "N",
        "activation rows / tokens (grid y x BN, tail tokens re-read token N-1 with scale 0)",
    ),
];

const K_GROUPS: Expr = Expr::DivExact(&P_K, 256);
const A_BYTES: Expr = Expr::Mul(&[P_M, K_GROUPS, Expr::Const(258)]);
const B_SCALE_PLANE: Expr = Expr::Mul(&[P_M, K_GROUPS, Expr::Const(2)]);
const B_CODE_OFFSET: Expr = Expr::Mul(&[Expr::DivCeil(&B_SCALE_PLANE, 256), Expr::Const(256)]);
const B_BYTES: Expr = Expr::Add(&[B_CODE_OFFSET, Expr::Mul(&[P_M, P_K])]);

const OP_WQ_A: OperandV1 = OperandV1 {
    name: "Wq",
    format: FORMAT_MQ8G256_Q14_258,
    access: Access::ReadOnly,
    align: 2,
    extent: Some(A_BYTES),
    nullable: false,
    note: "layout A: qt14 rows `[M, K/256] x 258 B` read in place (row stride K/256*258; a group's codes sit 2 B off a dword boundary in even groups of a 4-aligned row): the stager realigns dwords with a funnel shift and reads the 17th dword of a shifted slab with an exactly-in-bounds 16-bit load, so the base needs only 2-byte alignment; rows past M are re-read from row M-1",
};

const OP_WQ_B: OperandV1 = OperandV1 {
    name: "Wq",
    format: FORMAT_MQ8G256_B_PLANES,
    access: Access::ReadOnly,
    align: 16,
    extent: Some(B_BYTES),
    nullable: false,
    note: "layout B: f16 scale plane `[M, K/256]` (2 B each), zero padding to a 256 B boundary (`mq8g256_b_code_offset`), then the row-major int8 code plane `[M, K]` read as 16 B vectors: base 16-byte aligned; extent = round_up(M*K/256*2, 256) + M*K; rows past M are re-read from row M-1",
};

const OP_XQ: OperandV1 = OperandV1 {
    name: "Xq",
    format: FORMAT_I8_N_K,
    access: Access::ReadOnly,
    align: 16,
    extent: Some(Expr::Mul(&[P_N, P_K])),
    nullable: false,
    note: "int8 activations `[N, K]` from mq8_rotate_quantize_x_batched (codes in [-128, 127]); read as 16 B vectors; tokens past N are re-read from row N-1",
};

const OP_XS: OperandV1 = OperandV1 {
    name: "Xs",
    format: FORMAT_F32_KG256_N_SCALES,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(Expr::Mul(&[K_GROUPS, P_N, Expr::Const(4)])),
    nullable: false,
    note: "F32 activation scales `[K/256, N]` (group-major); one scale feeds the two K128 epochs of a group; tokens past N read as 0 (zero contribution)",
};

const OP_Y_F32: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_F32_N_M,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
    nullable: false,
    note: "F32 output `[N, M]` (Y[token * M + row]); tail rows/tokens are masked, every in-range element is written once; float4 stores when M % 8 == 0 and the base is 16 B aligned, scalar stores otherwise",
};

const OP_Y_BF16: OperandV1 = OperandV1 {
    name: "Y",
    format: FORMAT_F32_N_M,
    access: Access::WriteOnly,
    align: 4,
    extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
    nullable: false,
    note: "F32 output `[N, M]` holding BF16-rounded values (`y.dtype == F32`, `n*m*4` bytes in the launcher): the epilogue rounds each output to BF16 (RN-even, non-finite passthrough) and widens it back; tail masked, every in-range element written once",
};

const OPS_GEMM_A_F32: &[OperandV1] = &[OP_WQ_A, OP_XQ, OP_XS, OP_Y_F32];
const OPS_GEMM_A_BF16: &[OperandV1] = &[OP_WQ_A, OP_XQ, OP_XS, OP_Y_BF16];
const OPS_GEMM_B_F32: &[OperandV1] = &[OP_WQ_B, OP_XQ, OP_XS, OP_Y_F32];
const OPS_GEMM_B_BF16: &[OperandV1] = &[OP_WQ_B, OP_XQ, OP_XS, OP_Y_BF16];

const FIELDS_GEMM: &[FieldV1] = &[
    ptr_field("Wq", 0, "Wq"),
    ptr_field("Xq", 8, "Xq"),
    ptr_field("Xs", 16, "Xs"),
    ptr_field("Y", 24, "Y"),
    i32_field("M", 32, P_M),
    i32_field("K", 36, P_K),
    i32_field("N", 40, P_N),
];

const DOMAIN_GEMM: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 256),
        "a 256-group carries one W scale and one X scale feeding two K128 epochs: K is whole groups",
    ),
    rule(
        Rule::AtMost(Expr::DivCeil(&P_N, BN_MIN), GRID_Y_MAX),
        "grid y (ceil(N/BN)) must fit the HIP y-grid limit; stated for the narrowest token tile (BN = 64), the 128-wide tiles admit more",
    ),
];

/// One tile entry of an arch: block `2*BM`, LDS fixed per tile (`W_BYTES + X_BYTES + WS_BYTES + XS_BYTES`),
/// grid `[ceil(M/BM), ceil(N/BN)]`.
macro_rules! gemm_export {
    ($symbol:expr, $arch:expr, $object:expr, $bm:literal, $bn:literal, $lds:literal) => {
        ExportV1 {
            symbol: $symbol,
            arch: $arch,
            object: $object,
            block: [2 * $bm, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: $lds,
            grid: [Expr::DivCeil(&P_M, $bm), Expr::DivCeil(&P_N, $bn), ONE],
            consts: &[("BM", $bm), ("BN", $bn)],
        }
    };
}

/// All six exports (3 tiles x 2 arches) of one (layout, epilogue) suffix (`wa`, `wa_bf16`, `wb`, `wb_bf16`).
macro_rules! gemm_exports {
    ($sfx:literal) => {
        &[
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_t128x128_", $sfx),
                Arch::Gfx1201,
                OBJ_GFX12,
                128,
                128,
                37888
            ),
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_t128x64_", $sfx),
                Arch::Gfx1201,
                OBJ_GFX12,
                128,
                64,
                28416
            ),
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_t64x64_", $sfx),
                Arch::Gfx1201,
                OBJ_GFX12,
                64,
                64,
                18944
            ),
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_gfx11_t128x128_", $sfx),
                Arch::Gfx1100,
                OBJ_GFX11,
                128,
                128,
                37888
            ),
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_gfx11_t64x128_", $sfx),
                Arch::Gfx1100,
                OBJ_GFX11,
                64,
                128,
                28416
            ),
            gemm_export!(
                concat!("gemm_mq8g256_mmq_i8_gfx11_t64x64_", $sfx),
                Arch::Gfx1100,
                OBJ_GFX11,
                64,
                64,
                18944
            ),
        ]
    };
}

macro_rules! gemm_abi {
    ($(#[$doc:meta])* $name:ident, $id:literal, $numeric:expr, $ops:expr, $sfx:literal) => {
        $(#[$doc])*
        pub static $name: KernelAbiV1 = KernelAbiV1 {
            id: $id,
            version: 1,
            wave: 32,
            numeric: $numeric,
            params: PARAMS_GEMM,
            operands: $ops,
            fields: FIELDS_GEMM,
            hidden: &[],
            explicit_bytes: 44,
            segment_bytes: 44,
            segment_align: 8,
            domain: DOMAIN_GEMM,
            exports: gemm_exports!($sfx),
        };
    };
}

gemm_abi!(
    /// `gemm_mq8g256_mmq_i8_*_wa`: layout A weights, F32 output. Source
    /// `kernels/src/gemm_mq8g256_mmq_i8.gfx12.hip` (gfx1201, module `gemm_mq8g256_mmq_i8_gfx12`, tiles 128x128 /
    /// 128x64 / 64x64) and `.gfx11.hip` (gfx1100, module `gemm_mq8g256_mmq_i8_gfx11`, tiles 128x128 / 64x128 /
    /// 64x64, symbols carry `_gfx11_`): args `Wq@0 Xq@8 Xs@16` (`read_only`) `Y@24` (`write_only`), `M@32 K@36 N@40`
    /// by value, `.kernarg_segment_size` 44 / align 8, NO hidden words, one LDS buffer (`group_segment_fixed`
    /// 37888 / 28416 / 18944 for the 128x128 / 128x64 or 64x128 / 64x64 tiles). Launch: block `2*BM`, grid
    /// `(ceil(M/BM), ceil(N/BN))`. `mq8_pick_tile` is a performance choice only (all tiles are bit-identical).
    GEMM_MQ8G256_MMQ_I8_WA,
    "generic_mq8.gemm_mq8g256_mmq_i8.wa.f32",
    NUMERIC_MQ8G256_X_I8G256_MMQ_F32,
    OPS_GEMM_A_F32,
    "wa"
);
gemm_abi!(
    /// `gemm_mq8g256_mmq_i8_*_wa_bf16`: layout A weights, BF16-rounded F32 output (`Gpu::gemm_mq8g256_gemm_bf16`);
    /// kernarg layout, LDS and launch of [`GEMM_MQ8G256_MMQ_I8_WA`].
    GEMM_MQ8G256_MMQ_I8_WA_BF16,
    "generic_mq8.gemm_mq8g256_mmq_i8.wa.bf16_in_f32",
    NUMERIC_MQ8G256_X_I8G256_MMQ_BF16_RNE,
    OPS_GEMM_A_BF16,
    "wa_bf16"
);
gemm_abi!(
    /// `gemm_mq8g256_mmq_i8_*_wb`: layout B (aligned repack) weights, F32 output; kernarg layout, LDS and launch of
    /// [`GEMM_MQ8G256_MMQ_I8_WA`].
    GEMM_MQ8G256_MMQ_I8_WB,
    "generic_mq8.gemm_mq8g256_mmq_i8.wb.f32",
    NUMERIC_MQ8G256_X_I8G256_MMQ_F32,
    OPS_GEMM_B_F32,
    "wb"
);
gemm_abi!(
    /// `gemm_mq8g256_mmq_i8_*_wb_bf16`: layout B weights, BF16-rounded F32 output; kernarg layout, LDS and launch of
    /// [`GEMM_MQ8G256_MMQ_I8_WA`].
    GEMM_MQ8G256_MMQ_I8_WB_BF16,
    "generic_mq8.gemm_mq8g256_mmq_i8.wb.bf16_in_f32",
    NUMERIC_MQ8G256_X_I8G256_MMQ_BF16_RNE,
    OPS_GEMM_B_BF16,
    "wb_bf16"
);

/// F32-output GEMMs (`gemm_mq8g256_gemm`): the Raon launches of this epilogue mirror the F32 Q8 `ps` GEMM, which
/// is the MLA `kv_b` expansion, sub-chunked at `MLA_TILE_ROWS` = 512 rows, so `N` binds to the tile height 512
/// (MACs are strict, so a bare interval would leave the leaf without a floor); the binder verifies it against
/// the grid interval (every tile of the family covers 512: y in {4, 8} -> N in [385, 512] / [449, 512]).
/// [INFERENCE] The M3 projection arms are not wired yet: this follows the routing of the Q8 chain this replaces.
const BIND_GEMM_F32: &[BindV1] = &[
    BindV1 {
        param: "N",
        source: BindSource::Derived(Expr::Const(512)),
    },
    BIND_MQ8_K,
];

/// BF16-rounded GEMMs: the Raon launches run on the whole pass chunk (like `gemm_q8_0_a8_gemm_bf16`).
const BIND_GEMM_BF16: &[BindV1] = &[
    BindV1 {
        param: "N",
        source: BindSource::ChunkRows,
    },
    BIND_MQ8_K,
];

pub static WORK_GEMM_MQ8G256_MMQ_I8_WA: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_MQ8G256_MMQ_I8_WA,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "mq8g256",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_MQ8,
        bind: BIND_GEMM_F32,
        note: concat!(
            "Layout A (258 B groups in place), F32 Y. N binds to 512 = the MLA tile height. ",
            "MACs = M*N*K (tile padding excluded): iu8 WMMA over int8 weight codes x int8 activation codes, the per-epoch scale ",
            "folds (`fmaf` per output per K128 epoch) are VALU work not charged. Bytes = sum of the operand extents: weights once ",
            "(M*K/256*258) + int8 X (N*K) + F32 scales (K/256*N*4) + F32 Y (N*M*4). K comes from the qt14 (`mq8g256`) weight match ",
            "on M (grid x interval of BM). The tile selector (`mq8_pick_tile`) is a grid predicate, not a row one."
        ),
    },
};

pub static WORK_GEMM_MQ8G256_MMQ_I8_WA_BF16: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_MQ8G256_MMQ_I8_WA_BF16,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "mq8g256",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_MQ8,
        bind: BIND_GEMM_BF16,
        note: concat!(
            "Layout A (258 B groups in place), BF16-rounded F32 Y. N binds to the chunk rows. ",
            "MACs = M*N*K (tile padding excluded) on the iu8 WMMA pipe; the scale folds and the BF16 round are VALU work not ",
            "charged. Bytes = sum of the operand extents: weights once (M*K/256*258) + int8 X (N*K) + F32 scales (K/256*N*4) + F32 Y ",
            "(N*M*4, the plane is F32). K comes from the qt14 (`mq8g256`) weight match on M. The 5120-row node (q/k/v/g) may share M ",
            "with other tensors: the weight-set selection by call count may report several fitting sets."
        ),
    },
};

pub static WORK_GEMM_MQ8G256_MMQ_I8_WB: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_MQ8G256_MMQ_I8_WB,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "mq8g256",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_MQ8,
        bind: BIND_GEMM_F32,
        note: concat!(
            "Layout B (aligned repack), F32 Y. N binds to 512 = the MLA tile height. ",
            "MACs = M*N*K (tile padding excluded): iu8 WMMA over int8 weight codes x int8 activation codes, the per-epoch scale ",
            "folds are VALU work not charged. Bytes = sum of the operand extents: weights once (round_up(M*K/256*2, 256) + M*K) + int8 X ",
            "(N*K) + F32 scales (K/256*N*4) + F32 Y (N*M*4). K comes from the qt14 (`mq8g256`) weight match on M (the tensor directory ",
            "names the tensor by its file codec; the repack is a load-time layout)."
        ),
    },
};

pub static WORK_GEMM_MQ8G256_MMQ_I8_WB_BF16: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_MQ8G256_MMQ_I8_WB_BF16,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "mq8g256",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_MQ8,
        bind: BIND_GEMM_BF16,
        note: concat!(
            "Layout B (aligned repack), BF16-rounded F32 Y. N binds to the chunk rows. ",
            "MACs = M*N*K (tile padding excluded) on the iu8 WMMA pipe; the scale folds and the BF16 round are VALU work not ",
            "charged. Bytes = sum of the operand extents: weights once (round_up(M*K/256*2, 256) + M*K) + int8 X (N*K) + F32 scales ",
            "(K/256*N*4) + F32 Y (N*M*4, the plane is F32). K comes from the qt14 (`mq8g256`) weight match on M."
        ),
    },
};

// ================================================================================================
// gemv_mq8g256_b_r{R}_u{U}_w4_p1
// ================================================================================================

const P_R: Expr = Expr::Param("R");
const P_LDXS: Expr = Expr::Param("ldxs");

const PARAMS_GEMV_B: &[ParamV1] = &[
    param(
        "M",
        "weight rows (grid x = ceil(M / 4), one wave32 per row)",
    ),
    param("K", "row length in elements, a multiple of 256"),
    param(
        "ldxs",
        "row stride of the X scale plane `[K/256][ldxs]` (the producer's N), >= R",
    ),
];

const OPS_GEMV_B: &[OperandV1] = &[
    OperandV1 {
        name: "W",
        format: FORMAT_MQ8G256_B_PLANES,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(B_BYTES),
        nullable: false,
        note: "layout B: f16 scale plane `[M, K/256]` (uniform dword scalar loads), zero padding to a 256 B boundary, then the row-major int8 code plane `[M, K]` read as aligned 16 B vectors: base 16-byte aligned; extent = round_up(M*K/256*2, 256) + M*K",
    },
    OperandV1 {
        name: "xq",
        format: FORMAT_I8_N_K,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_R, P_K])),
        nullable: false,
        note: "int8 activation codes `[R][K]` of the launch (the launcher passes a pointer already offset to the first X row; row r at byte r*K), read as 16 B vectors",
    },
    OperandV1 {
        name: "xs",
        format: FORMAT_F32_KG256_N_SCALES,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[K_GROUPS, P_LDXS, Expr::Const(4)])),
        nullable: false,
        note: "F32 scales `[K/256][ldxs]` (group-major); the scale of row r, group g is `xs[g * ldxs + r]`, ldxs >= R",
    },
    OperandV1 {
        name: "y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_R, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[R][M]`: lane 0 of the wave owning `row` stores `y[r * M + row]` for every r",
    },
];

const FIELDS_GEMV_B: &[FieldV1] = &[
    ptr_field("W", 0, "W"),
    ptr_field("xq", 8, "xq"),
    ptr_field("xs", 16, "xs"),
    ptr_field("y", 24, "y"),
    i32_field("M", 32, P_M),
    i32_field("K", 36, P_K),
    i32_field("ldxs", 40, P_LDXS),
];

const DOMAIN_GEMV_B: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 256),
        "a lane owns 16 code bytes of a whole 256-group; the kernel iterates K/256 groups and drops a tail",
    ),
    rule(
        Rule::Positive(P_LDXS),
        "the scale plane stride must cover the launch's X rows: ldxs >= R (R is a per-export constant, not expressible as an AtLeast limit; the launcher guarantees it)",
    ),
    rule(
        Rule::AtMost(Expr::Mul(&[P_M, P_K]), (1u64 << 40) - 1),
        "the launcher bounds M*K < 2^40 (byte offsets are formed in size_t, the weight extent stays addressable)",
    ),
];

const OBJ_GEMV_B: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "gemv_mq8g256_b",
    source: "kernels/src/gemv_mq8g256_b.hip",
};

/// One export of an arch: block 128 (4 waves, one weight row per wave), grid `[ceil(M/4), 1, 1]`, no LDS.
macro_rules! gemv_b_export {
    ($arch:expr, $r:literal) => {
        ExportV1 {
            symbol: concat!("gemv_mq8g256_b_r", $r, "_u1_w4_p1"),
            arch: $arch,
            object: OBJ_GEMV_B,
            block: [128, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [Expr::DivCeil(&P_M, 4), ONE, ONE],
            consts: &[("R", $r)],
        }
    };
}

macro_rules! gemv_b_exports {
    ($($r:literal),* $(,)?) => {
        &[
            $(gemv_b_export!(Arch::Gfx1201, $r),)*
            $(gemv_b_export!(Arch::Gfx1100, $r),)*
        ]
    };
}

/// `gemv_mq8g256_b_r{R}_u1_w4_p1` (`kernels/src/gemv_mq8g256_b.hip`, module `gemv_mq8g256_b`), R = 1..=16,
/// gfx1201 + gfx1100 (32 exports): args `W@0 xq@8 xs@16 y@24`, `M@32 K@36
/// ldxs@40` by value, kernarg segment 48 / align 8, NO hidden words, LDS 0, block 128, grid `[ceil(M/4), 1, 1]`
/// (`Gpu::gemv_mq8g256_b` / `gemv_mq8g256_rows_exact_b`). Row r of an R-row launch is bit-identical to the R = 1
/// kernel on X row r.
pub static GEMV_MQ8G256_B: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mq8.gemv_mq8g256_b.i8x",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GEMV_MQ8G256_B,
    params: PARAMS_GEMV_B,
    operands: OPS_GEMV_B,
    fields: FIELDS_GEMV_B,
    hidden: &[],
    explicit_bytes: 44,
    segment_bytes: 44,
    segment_align: 8,
    domain: DOMAIN_GEMV_B,
    exports: gemv_b_exports!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
};

pub static WORK_GEMV_MQ8G256_B: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMV_MQ8G256_B,
    work: WorkV1 {
        op_class: op_class::GEMV,
        codec: "mq8g256",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_R, P_M, P_K])),
        flops: None,
        bytes: None,
        rows: Some("R"),
        admit: RowDomainV1 {
            min_rows: 1,
            max_rows: Some(16),
            predicate: "crates/rdna-compute/src/gemv_mq8_rows.rs::MQ8G256_ROWS_EXACT_KERNELS",
            kill_env: None,
        },
        bind: &[BIND_MQ8_K],
        note: "R = the symbol's X row count (1..=16 per launch; larger blocks run in groups of 16). Integer sdot4 MACs with an f32 group fold, charged to the VALU pipe. Weights are read once for all R rows. Bytes = layout-B weight + R x int8 X + scale plane + R x F32 y extents.",
    },
};

// ================================================================================================
// gemv_mq8g256_bx_r{R}_u{U}_w{W}_p1 (fused rotate+quantize producer)
// ================================================================================================

const PARAMS_GEMV_BX: &[ParamV1] = &[
    param(
        "M",
        "weight rows (grid x = ceil(M / W), one wave32 per row)",
    ),
    param(
        "K",
        "row length in elements, a multiple of 256, at most 5120 (LDS X staging)",
    ),
];

const OPS_GEMV_BX: &[OperandV1] = &[
    OperandV1 {
        name: "W",
        format: FORMAT_MQ8G256_B_PLANES,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(B_BYTES),
        nullable: false,
        note: "layout B: f16 scale plane `[M, K/256]`, zero padding to a 256 B boundary, then the row-major int8 code plane `[M, K]` read as aligned 16 B vectors: base 16-byte aligned; extent = round_up(M*K/256*2, 256) + M*K",
    },
    OperandV1 {
        name: "x",
        format: FORMAT_F32_ROWS_NK,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_R, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[R][K]` (row r at byte r*K*4); rotated and quantized in-kernel",
    },
    OperandV1 {
        name: "signs1",
        format: FORMAT_MQ_FWHT_SIGNS_256,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(SIGNS_EXTENT),
        nullable: false,
        note: "pre-rotation +-1 signs, 256 F32",
    },
    OperandV1 {
        name: "signs2",
        format: FORMAT_MQ_FWHT_SIGNS_256,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(SIGNS_EXTENT),
        nullable: false,
        note: "post-rotation +-1 signs, 256 F32",
    },
    OperandV1 {
        name: "y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_R, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[R][M]`: lane 0 of the wave owning `row` stores `y[r * M + row]` for every r",
    },
];

const FIELDS_GEMV_BX: &[FieldV1] = &[
    ptr_field("W", 0, "W"),
    ptr_field("x", 8, "x"),
    ptr_field("signs1", 16, "signs1"),
    ptr_field("signs2", 24, "signs2"),
    ptr_field("y", 32, "y"),
    i32_field("M", 40, P_M),
    i32_field("K", 44, P_K),
];

const DOMAIN_GEMV_BX: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 256),
        "a lane owns 16 code bytes of a whole 256-group; the kernel iterates K/256 groups and drops a tail",
    ),
    rule(
        Rule::AtMost(P_K, 5120),
        "the fused kernels stage the rotated X rows in LDS: K <= MQ8B_FX_KMAX = 5120",
    ),
    rule(
        Rule::Positive(P_K),
        "K <= 512*U for the export's U (U in {1, 5, 10}; R is a per-export constant): not expressible as an AtMost limit, the launcher's route (`mq8g256_b_fused_route`) guarantees it",
    ),
    rule(
        Rule::AtMost(Expr::Mul(&[P_M, P_K]), (1u64 << 40) - 1),
        "the launcher bounds M*K < 2^40 (byte offsets are formed in size_t, the weight extent stays addressable)",
    ),
];

/// One export of an arch: block `32*W` (W waves, one weight row per wave), grid `[ceil(M/W), 1, 1]`, static LDS
/// `R * 5200` (R * 5120 code bytes + R * 20 F32 scales).
macro_rules! gemv_bx_export {
    ($arch:expr, $r:literal, $u:literal, $w:literal, $block:literal, $lds:literal) => {
        ExportV1 {
            symbol: concat!("gemv_mq8g256_bx_r", $r, "_u", $u, "_w", $w, "_p1"),
            arch: $arch,
            object: OBJ_GEMV_B,
            block: [$block, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: $lds,
            grid: [Expr::DivCeil(&P_M, $w), ONE, ONE],
            consts: &[("R", $r), ("U", $u), ("W", $w)],
        }
    };
}

/// `gemv_mq8g256_bx_r{R}_u{U}_w{W}_p1` (`kernels/src/gemv_mq8g256_b.hip`, module `gemv_mq8g256_b`), R in {1, 2},
/// (U, W) in {(1, 10), (5, 24), (10, 20)}, gfx1201 + gfx1100 (12 exports): the fused-producer GEMV. X rows are
/// rotated and quantized in-kernel (no producer launch); args `W@0 x@8 signs1@16 signs2@24 y@32`, `M@40 K@44`
/// by value, kernarg segment 48 / align 8, NO hidden words, static LDS `R * 5200`, block `32*W`, grid
/// `[ceil(M/W), 1, 1]` (`Gpu::gemv_mq8g256_b_x`).
pub static GEMV_MQ8G256_BX: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mq8.gemv_mq8g256_bx.f32x",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GEMV_MQ8G256_BX,
    params: PARAMS_GEMV_BX,
    operands: OPS_GEMV_BX,
    fields: FIELDS_GEMV_BX,
    hidden: &[],
    explicit_bytes: 48,
    segment_bytes: 48,
    segment_align: 8,
    domain: DOMAIN_GEMV_BX,
    exports: &[
        gemv_bx_export!(Arch::Gfx1201, 1, 1, 10, 320, 5200),
        gemv_bx_export!(Arch::Gfx1201, 1, 5, 24, 768, 5200),
        gemv_bx_export!(Arch::Gfx1201, 1, 10, 20, 640, 5200),
        gemv_bx_export!(Arch::Gfx1201, 2, 1, 10, 320, 10400),
        gemv_bx_export!(Arch::Gfx1201, 2, 5, 24, 768, 10400),
        gemv_bx_export!(Arch::Gfx1201, 2, 10, 20, 640, 10400),
        gemv_bx_export!(Arch::Gfx1100, 1, 1, 10, 320, 5200),
        gemv_bx_export!(Arch::Gfx1100, 1, 5, 24, 768, 5200),
        gemv_bx_export!(Arch::Gfx1100, 1, 10, 20, 640, 5200),
        gemv_bx_export!(Arch::Gfx1100, 2, 1, 10, 320, 10400),
        gemv_bx_export!(Arch::Gfx1100, 2, 5, 24, 768, 10400),
        gemv_bx_export!(Arch::Gfx1100, 2, 10, 20, 640, 10400),
    ],
};

pub static WORK_GEMV_MQ8G256_BX: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMV_MQ8G256_BX,
    work: WorkV1 {
        op_class: op_class::GEMV,
        codec: "mq8g256",
        pipe: Pipe::ValuF32,
        macs: Some(Expr::Mul(&[P_R, P_M, P_K])),
        flops: None,
        bytes: None,
        rows: Some("R"),
        admit: RowDomainV1 {
            min_rows: 1,
            max_rows: Some(2),
            predicate: "crates/rdna-compute/src/gemv_mq8_rows.rs::MQ8G256_FUSED_N_CAP",
            kill_env: None,
        },
        bind: &[BIND_MQ8_K],
        note: "R = X rows (1..=2); X is quantized in-kernel, no producer launch. Integer sdot4 MACs with an f32 group fold, charged to the VALU pipe. Weights are read once for all R rows.",
    },
};

// ================================================================================================

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &MQ8_ROTATE_QUANTIZE_X_BATCHED,
    &GEMM_MQ8G256_MMQ_I8_WA,
    &GEMM_MQ8G256_MMQ_I8_WA_BF16,
    &GEMM_MQ8G256_MMQ_I8_WB,
    &GEMM_MQ8G256_MMQ_I8_WB_BF16,
    &GEMV_MQ8G256_B,
    &GEMV_MQ8G256_BX,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &WORK_MQ8_ROTATE_QUANTIZE_X_BATCHED,
    &WORK_GEMM_MQ8G256_MMQ_I8_WA,
    &WORK_GEMM_MQ8G256_MMQ_I8_WA_BF16,
    &WORK_GEMM_MQ8G256_MMQ_I8_WB,
    &WORK_GEMM_MQ8G256_MMQ_I8_WB_BF16,
    &WORK_GEMV_MQ8G256_B,
    &WORK_GEMV_MQ8G256_BX,
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

    const GEMMS: [&KernelAbiV1; 4] = [
        &GEMM_MQ8G256_MMQ_I8_WA,
        &GEMM_MQ8G256_MMQ_I8_WA_BF16,
        &GEMM_MQ8G256_MMQ_I8_WB,
        &GEMM_MQ8G256_MMQ_I8_WB_BF16,
    ];

    #[test]
    fn every_contract_self_checks() {
        assert_eq!(ABIS.len(), 7);
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
            let want =
                if std::ptr::eq(w.abi, &GEMV_MQ8G256_B) || std::ptr::eq(w.abi, &GEMV_MQ8G256_BX) {
                    1
                } else {
                    64
                };
            assert_eq!(w.work.admit.min_rows, want, "{}", a.id);
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
                if abi.exports.len() == 2 {
                    assert!(text.contains(e.symbol), "{source} lacks {}", e.symbol);
                    continue;
                }
                if e.symbol.starts_with("gemv_mq8g256_bx_r") {
                    // Macro-generated: `gemv_mq8g256_bx_r##R##_u##U##_w##WPB##_p##P`.
                    assert!(
                        text.contains("gemv_mq8g256_bx_r##R##_u##U##_w##WPB##_p##P"),
                        "{source} lacks the gemv_bx macro"
                    );
                    continue;
                }
                if e.symbol.starts_with("gemv_mq8g256_b_r") {
                    // Macro-generated: `gemv_mq8g256_b_r##R##_u##U##_w##WPB##_p##P`.
                    assert!(
                        text.contains("gemv_mq8g256_b_r##R##_u##U##_w##WPB##_p##P"),
                        "{source} lacks the gemv_b macro"
                    );
                    continue;
                }
                // The GEMM symbols are instantiated through `NAME##WL(##_bf16)` macros: compare the pieces.
                let tile = ["128x128", "128x64", "64x128", "64x64"]
                    .into_iter()
                    .find(|t| e.symbol.contains(&format!("_t{t}_")))
                    .unwrap_or_else(|| panic!("{}: no tile in the symbol", e.symbol));
                let marker = format!("_t{tile}_");
                let (stem, wl) = e.symbol.split_once(&marker).unwrap();
                let (wl, bf16) = match wl.strip_suffix("_bf16") {
                    Some(w) => (w, true),
                    None => (wl, false),
                };
                let frag = format!("({stem}{marker}##WL{}", if bf16 { "##_bf16" } else { "," });
                assert!(text.contains(&frag), "{source} lacks {frag} ({})", e.symbol);
                assert!(
                    text.contains(&format!("TILES({wl},")),
                    "{source}: no {wl} set"
                );
            }
        }
    }

    #[test]
    fn admission_predicates_name_real_constants() {
        let root = repo_root();
        for w in WORK {
            let p = w.work.admit.predicate;
            let (path, name) = p.split_once("::").expect("path::NAME");
            let text = std::fs::read_to_string(root.join(path))
                .unwrap_or_else(|e| panic!("{}: {path}: {e}", w.abi.id));
            assert!(text.contains(name), "{path} has no {name}");
        }
    }

    #[test]
    fn layouts_match_the_spec() {
        // Producer: 48 explicit bytes (5 pointers + K + N), 13 hidden words ending at the grid dims, 304 segment.
        let p = &MQ8_ROTATE_QUANTIZE_X_BATCHED;
        assert_eq!(
            (p.explicit_bytes, p.segment_bytes, p.segment_align),
            (48, 304, 8)
        );
        assert_eq!(p.hidden.len(), 13);
        let last = p.hidden.last().unwrap();
        assert_eq!((last.offset, last.width), (112, 2));
        assert_eq!(p.exports.len(), 2);
        for e in p.exports {
            assert_eq!(e.block, [256, 1, 1]);
            assert_eq!((e.dynamic_lds, e.group_segment_fixed), (0, 0));
        }
        for abi in GEMMS {
            assert_eq!(
                (abi.explicit_bytes, abi.segment_bytes, abi.segment_align),
                (44, 44, 8),
                "{}",
                abi.id
            );
            assert!(abi.hidden.is_empty(), "{}", abi.id);
            assert_eq!(abi.exports.len(), 6, "{}", abi.id);
            for e in abi.exports {
                let (_, rest) = e.symbol.split_once("_t").unwrap();
                let tile = rest.split('_').next().unwrap();
                let (bm, bn): (u32, u32) = match tile {
                    "128x128" => (128, 128),
                    "128x64" => (128, 64),
                    "64x128" => (64, 128),
                    "64x64" => (64, 64),
                    other => panic!("tile {other}"),
                };
                let lds = match (bm, bn) {
                    (128, 128) => 37888,
                    (128, 64) | (64, 128) => 28416,
                    _ => 18944,
                };
                assert_eq!(e.block, [2 * bm, 1, 1], "{}", e.symbol);
                assert_eq!(e.group_segment_fixed, lds, "{}", e.symbol);
                assert_eq!(e.dynamic_lds, 0);
                assert_eq!(e.consts, &[("BM", u64::from(bm)), ("BN", u64::from(bn))]);
                // gfx12 owns 128x64, gfx11 owns 64x128, the symbol carries `_gfx11_` on gfx1100 only.
                match e.arch {
                    Arch::Gfx1201 => {
                        assert!(bn != 128 || bm == 128, "{}", e.symbol);
                        assert!(!e.symbol.contains("gfx11"), "{}", e.symbol);
                        assert_eq!(
                            e.object,
                            ObjectRefV1::HipccSource {
                                module: MODULE_GFX12,
                                source: SOURCE_GFX12
                            }
                        );
                    }
                    Arch::Gfx1100 => {
                        assert!(bm != 128 || bn == 128, "{}", e.symbol);
                        assert!(e.symbol.contains("_gfx11_t"), "{}", e.symbol);
                        assert_eq!(
                            e.object,
                            ObjectRefV1::HipccSource {
                                module: MODULE_GFX11,
                                source: SOURCE_GFX11
                            }
                        );
                    }
                    Arch::Gfx1151 => panic!("{}: gfx1151 is not an export", e.symbol),
                }
            }
        }
    }

    #[test]
    fn contracts_differ_where_they_must() {
        // Layouts differ in the weight operand only; epilogues in numeric id only.
        let [a, a_bf16, b, b_bf16] = GEMMS;
        assert_ne!(a.operands[0].format, b.operands[0].format);
        assert_eq!((a.operands[0].align, b.operands[0].align), (2, 16));
        assert_eq!(a.operands[1..3], b.operands[1..3]);
        assert_ne!(a.numeric, a_bf16.numeric);
        assert_eq!(a.numeric, b.numeric);
        assert_eq!(a_bf16.numeric, b_bf16.numeric);
        let mut ids: Vec<_> = GEMMS.iter().map(|g| g.id).collect();
        ids.push(MQ8_ROTATE_QUANTIZE_X_BATCHED.id);
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 5);
        // The producer's X plane operands are what the GEMMs read.
        let pq = MQ8_ROTATE_QUANTIZE_X_BATCHED.operands[1];
        let ps = MQ8_ROTATE_QUANTIZE_X_BATCHED.operands[2];
        for g in GEMMS {
            assert_eq!(g.operands[1].format, pq.format);
            assert_eq!(g.operands[1].align, pq.align);
            assert_eq!(g.operands[2].format, ps.format);
        }
    }

    fn vals(abi: &KernelAbiV1, given: &[(&str, u64)]) -> Vec<u64> {
        abi.params
            .iter()
            .map(|p| given.iter().find(|(n, _)| *n == p.name).unwrap().1)
            .collect()
    }

    #[test]
    fn extents_grids_and_work_on_the_raon_shapes() {
        // (M, K): RDA q/k/v/g + MLA g, RDA o + MLA o, MLA q_a / kv_a, MLA kv_b.
        for (m, k) in [(5120u64, 2560u64), (2560, 5120), (576, 2560), (10240, 512)] {
            for n in [512u64, 4096] {
                for abi in GEMMS {
                    let given = [("M", m), ("K", k), ("N", n)];
                    let v = vals(abi, &given);
                    for e in abi.exports {
                        let (bm, bn) = (e.consts[0].1, e.consts[1].1);
                        let g = |i: usize| eval(&e.grid[i], abi, e, &v).unwrap();
                        assert_eq!(g(0), m.div_ceil(bm));
                        assert_eq!(g(1), n.div_ceil(bn));
                        assert_eq!(g(2), 1);
                    }
                    let e = &abi.exports[0];
                    let ex = |i: usize| eval(&abi.operands[i].extent.unwrap(), abi, e, &v).unwrap();
                    let a_layout = abi.operands[0].format == FORMAT_MQ8G256_Q14_258;
                    let want_w = if a_layout {
                        m * (k / 256) * 258
                    } else {
                        (m * (k / 256) * 2).next_multiple_of(256) + m * k
                    };
                    assert_eq!(ex(0), want_w, "{} M {m} K {k}", abi.id);
                    assert_eq!(ex(1), n * k);
                    assert_eq!(ex(2), (k / 256) * n * 4);
                    assert_eq!(ex(3), n * m * 4);
                }
            }
        }
        // MACs and bytes (default = sum of extents) on M=5120 K=2560 N=4096.
        let w = &WORK_GEMM_MQ8G256_MMQ_I8_WA_BF16;
        let e = &w.abi.exports[0];
        let v = [5120u64, 2560, 4096];
        assert_eq!(
            eval(&w.work.macs.unwrap(), w.abi, e, &v).unwrap(),
            5120 * 2560 * 4096
        );
        assert!(w.work.bytes.is_none());
        // Producer: K=2560 N=4096 -> 5120 blocks; bytes exact = N*K*5 + K/256*N*4.
        let w = &WORK_MQ8_ROTATE_QUANTIZE_X_BATCHED;
        let e = &w.abi.exports[0];
        let v = [2560u64, 4096, 5120];
        assert_eq!(
            eval(&w.work.bytes.unwrap(), w.abi, e, &v).unwrap(),
            4096 * 2560 * 5 + 10 * 4096 * 4
        );
        for (n, k, blocks) in [
            (4096u64, 2560u64, 5120u64),
            (4096, 5120, 10240),
            (512, 512, 128),
            (4096, 512, 1024),
        ] {
            let abi = &MQ8_ROTATE_QUANTIZE_X_BATCHED;
            let v = vals(abi, &[("K", k), ("N", n), ("blocks", blocks)]);
            for e in abi.exports {
                assert_eq!(eval(&e.grid[0], abi, e, &v).unwrap(), blocks);
                assert_eq!(eval(&PRODUCER_BLOCKS, abi, e, &v).unwrap(), blocks);
            }
            assert_eq!(n * k, blocks * 2048);
            for d in abi.domain {
                if let Rule::AtMost(e, lim) = d.rule {
                    assert!(
                        eval(&e, abi, &abi.exports[0], &v).unwrap() <= lim,
                        "{}",
                        d.why
                    );
                }
            }
        }
        // The 32-bit group index bound is a domain rule: N*(K/256) = 2^31 is outside it.
        assert!(MQ8_ROTATE_QUANTIZE_X_BATCHED
            .domain
            .iter()
            .any(|d| matches!(
                d.rule,
                Rule::AtMost(Expr::Mul(&[Expr::Param("N"), Expr::DivExact(_, 256)]), l)
                    if l == i32::MAX as u64
            )));
    }

    #[test]
    fn bindings_and_pipes_follow_the_spec() {
        for w in [
            &WORK_GEMM_MQ8G256_MMQ_I8_WA,
            &WORK_GEMM_MQ8G256_MMQ_I8_WA_BF16,
            &WORK_GEMM_MQ8G256_MMQ_I8_WB,
            &WORK_GEMM_MQ8G256_MMQ_I8_WB_BF16,
        ] {
            assert_eq!(w.work.op_class, op_class::GEMM);
            assert_eq!(w.work.codec, "mq8g256");
            assert_eq!(w.work.pipe, Pipe::WmmaIu8);
            assert_eq!(w.work.rows, Some("N"));
            let Some(BindSource::Weight {
                by: "M",
                by_axis: WeightAxis::Rows,
                codecs: &["mq8g256"],
                stacking: Stacking::Single,
                fuse_siblings: false,
            }) = w.work.bind.last().map(|b| b.source)
            else {
                panic!("{}: last bind is not the qt14 weight match", w.abi.id);
            };
            assert_eq!(w.work.bind.last().unwrap().param, "K");
            assert_eq!(w.work.bind[0].param, "N");
        }
        assert!(matches!(
            WORK_GEMM_MQ8G256_MMQ_I8_WA.work.bind[0].source,
            BindSource::Derived(Expr::Const(512))
        ));
        assert!(matches!(
            WORK_GEMM_MQ8G256_MMQ_I8_WB_BF16.work.bind[0].source,
            BindSource::ChunkRows
        ));
        let w = &WORK_MQ8_ROTATE_QUANTIZE_X_BATCHED;
        assert_eq!(w.work.op_class, op_class::ROTATE);
        assert_eq!(w.work.pipe, Pipe::Memory);
        assert!(w.work.macs.is_none() && w.work.bind.is_empty());
    }
}
