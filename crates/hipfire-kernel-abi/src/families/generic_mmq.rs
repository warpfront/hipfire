// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 W8A8 prefill GEMM (pipelined 128x128 schedule) and the two activation quantizers of the
//! int8 prefill tiers (hipcc).
//!
//! Four contracts, each with a [`WorkSpecV1`] sidecar:
//!
//! | contract | export (gfx1201) | launcher |
//! |---|---|---|
//! | [`GEMM_Q8_0_MMQ_I8_T128X128_PS`] | `gemm_q8_0_mmq_i8_t128x128_ps` | `Gpu::gemm_q8_0_a8_gemm*` (`crates/rdna-compute/src/gemm_q8_a8.rs`) |
//! | [`GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16`] | `gemm_q8_0_mmq_i8_t128x128_ps_bf16` | same launcher, `bf16_out = true` |
//! | [`QUANTIZE_X_Q8_BLK32`] | `quantize_x_q8_blk32` | `Gpu::q8_a8_quantize_x` (same file) |
//! | [`QUANTIZE_INT8_MMQ_DS128`] | `quantize_int8_mmq_ds128` | `Scratch::ensure_int8_mmq_x` (`crates/rdna-compute/src/scratch.rs`) |
//!
//! The `ps` and `ps_bf16` GEMMs share one kernarg layout and one operand set: the BF16 variant does NOT
//! write BF16 words, it rounds every output to BF16 in the epilogue and still stores an F32 plane
//! (`gemm_q8_a8.rs`: `y.dtype == F32`, `n * m * 4` bytes). They differ only in numerics, hence two
//! contracts (a contract has exactly one numeric id).
//!
//! Evidence: the JIT-cached objects in `~/.hipfire_kernels/gfx1201/` (clang offload bundles, unbundled
//! with `clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201`, read with
//! `llvm-readelf --notes`), each chosen because the sibling `<module>.<hash>.hip` is byte-identical to
//! the current source:
//! * `gemm_q8_0_mmq_i8_gfx12.36fe2ca22ee066a0` (== `kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`; the module
//!   the launcher loads with `kernels::GEMM_Q8_0_MMQ_I8_GFX12_SRC`, no prelude): both `_ps` entries and
//!   `quantize_x_q8_blk32`;
//! * `gemm_mq4g256v2_residual_mmq_i8_gfx12.63383e7e3ba89ce2` (== `block_i8_128_quant.hip` ++
//!   `gemm_mq4g256v2_residual_mmq_i8.gfx12.hip`, the runtime's `GEMM_MQ4G256V2_RESIDUAL_MMQ_I8_GFX12_SRC`
//!   concatenation): `quantize_int8_mmq_ds128`.
//!
//! Every kernel is built for wave 32; `.private_segment_fixed_size` is 0 everywhere.
//! hipcc contracts carry no object hash (`HipccSource`), so no drift evidence is checked against them
//! other than the compiled-metadata test in `rdna-compute`.

use crate::work::{
    op_class, BindSource, BindV1, Pipe, RowDomainV1, Stacking, WeightAxis, WorkSpecV1, WorkV1,
};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_convert::FORMAT_F32_ROWS_NK;
use super::ggml_mmvq::FORMAT_Q8_0_34;
use super::mq6_i8::{FORMAT_BLOCK_I8_128_KB_N, FORMAT_F32_N_M};

// ---- formats -----------------------------------------------------------------

/// int8 activation plane `[N, K]` row-major, one `quantize_x_q8_blk32` code per element (`|q| <= 127`).
pub const FORMAT_I8_N_K: FormatId = FormatId("i8.n_x_k");
/// F32 per-(32-K block, row) scales `[K/32, N]`, block-major (`Xs[b * N + token]`), `d = RN(amax / 127)`.
pub const FORMAT_F32_KB_N_SCALES: FormatId = FormatId("f32.kb_x_n.q8_blk32_scales");

// ---- numeric contracts ---------------------------------------------------------

/// Q8_0 weight x int8 per-32-K activation: per 32-K block an exact int32 dot of the stored int8 codes
/// (two chained `v_wmma_i32_16x16x16_iu8` per block, signed x signed, accumulator seeded `0x4B400000` and
/// recovered by the exact float subtract of `12582912.0f`), folded as
/// `acc = fmaf(float(half(scale_w)) * scale_x, dot, acc)` (the product is its own F32 rounding) over
/// K blocks ascending, one plain F32 store per output. The pipelined (`_ps`) schedule issues the same
/// per-element operations in the same order (`gemm_q8_0_mmq_i8.gfx12.hip`, "bit-identical to the plain kernel").
pub const NUMERIC_Q8_0_A8_MMQ_F32: NumericContractId =
    NumericContractId("generic_mmq.q8_0_x_i8blk32.wmma_iu8.exact_dot_fma_fold.f32");
/// [`NUMERIC_Q8_0_A8_MMQ_F32`] followed, per output, by `q8a8_bf16_rn`: round-to-nearest-even to BF16
/// (`(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`), non-finite values unchanged, stored widened as F32.
pub const NUMERIC_Q8_0_A8_MMQ_BF16_RNE: NumericContractId =
    NumericContractId("generic_mmq.q8_0_x_i8blk32.wmma_iu8.exact_dot_fma_fold.bf16_rne_in_f32");
/// F32 -> int8 per 32-K block: `amax = max |x|` over the block (8 lanes x 4 elements, `fmaxf`), `d = RN(amax / 127)`
/// (IEEE divide), `q = clamp(rintf(RN(x / d)), -127, 127)` (ties to even), zero block -> `d = 0`, `q = 0`; codes
/// stored in K order, scale to `[K/32, N]`.
pub const NUMERIC_QUANTIZE_X_Q8_BLK32: NumericContractId =
    NumericContractId("generic_mmq.quantize_x_q8_blk32.f32_to_i8.amax127_rne");
/// F32 -> `block_i8_128` per 128-K group: `amax` over the group (wave32 `shfl_xor` max), `d = RN(amax / 127)`,
/// `q = clamp(rintf(RN(x / d)), -127, 127)`, `s = sum of the 128 q` (exact int32), zero group -> `d = 0`, `q = 0`,
/// `s = 0`; `qs` in the nibble-pair byte order of `block_i8_128_quant.hip` (8-element group `g`: dword `2g` =
/// `q[8g+0,2,4,6]`, dword `2g+1` = `q[8g+1,3,5,7]`).
pub const NUMERIC_QUANTIZE_INT8_MMQ_DS128: NumericContractId = NumericContractId(
    "generic_mmq.quantize_int8_mmq_ds128.f32_to_block_i8_128.amax127_rne_pairorder",
);

// ---- shared pieces ---------------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_N: Expr = Expr::Param("N");
const P_BLOCKS: Expr = Expr::Param("blocks");
const ONE: Expr = Expr::Const(1);
/// HIP grid-y limit (65535 workgroups).
const GRID_Y_MAX: u64 = 65535;

const MODULE_Q8_MMQ: &str = "gemm_q8_0_mmq_i8_gfx12";
const SOURCE_Q8_MMQ: &str = "kernels/src/gemm_q8_0_mmq_i8.gfx12.hip";
const MODULE_I8_128: &str = "gemm_mq4g256v2_residual_mmq_i8_gfx12";
const SOURCE_I8_128: &str = "kernels/src/block_i8_128_quant.hip";

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

/// Weight binding of the Q8_0 GEMMs: the grid pins `M` to a 128-row interval, the match against the Q8_0
/// weight rows binds `M` exactly and `K` to the weight's columns.
const BIND_Q8_K: BindV1 = BindV1 {
    param: "K",
    source: BindSource::Weight {
        by: "M",
        by_axis: WeightAxis::Rows,
        codecs: &["q8_0"],
        stacking: Stacking::Single,
        fuse_siblings: false,
    },
};

/// Row floor of every quantized prefill tier (`prefill_quant_admitted`: `rows >= PREFILL_QUANT_MIN_ROWS` = 64 and
/// no graph capture); the W8A8 Q8_0 route and its producers run only behind it (`q8_0_prefill_runs`).
const ADMIT_Q8_A8: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/q8_prefill.rs::PREFILL_QUANT_MIN_ROWS",
    kill_env: None,
};

// ================================================================================================
// gemm_q8_0_mmq_i8_t128x128_ps / _ps_bf16
// ================================================================================================

const PARAMS_MMQ_GEMM: &[ParamV1] = &[
    param(
        "M",
        "weight rows (output features; grid x x 128, tail masked)",
    ),
    param(
        "K",
        "row length in elements, a multiple of 64 (FAST weight path)",
    ),
    param("N", "activation rows / tokens (grid y x 128, tail masked)"),
];

const OPS_MMQ_GEMM_F32: &[OperandV1] = &[
    OperandV1 {
        name: "Wq",
        format: FORMAT_Q8_0_34,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_M, Expr::DivExact(&P_K, 32), Expr::Const(34)])),
        nullable: false,
        note: "Q8_0 rows `[M, K/32] x 34 B` read in place; the FAST kernels load the 68 B K64 slab as dwords: base 4-byte aligned (row stride 34*K/32 is a multiple of 4 since K % 64 == 0); rows past M are re-read from row M-1",
    },
    OperandV1 {
        name: "Xq",
        format: FORMAT_I8_N_K,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_N, P_K])),
        nullable: false,
        note: "int8 activations `[N, K]` from quantize_x_q8_blk32; read as 16 B vectors (K % 16 == 0); tokens past N are re-read from row N-1",
    },
    OperandV1 {
        name: "Xs",
        format: FORMAT_F32_KB_N_SCALES,
        access: Access::ReadOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivExact(&P_K, 32), P_N, Expr::Const(4)])),
        nullable: false,
        note: "F32 activation scales `[K/32, N]` (block-major); tokens past N read as 0 (zero contribution)",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[N, M]` (Y[token * M + row]); tail rows/tokens are masked, every in-range element is written once; float4 stores when M % 8 == 0 and the base is 16 B aligned, scalar stores otherwise",
    },
];

/// BF16-epilogue sibling: the same operands; the plane is F32 (BF16-rounded values), not BF16 words.
const OPS_MMQ_GEMM_BF16: &[OperandV1] = &[
    OPS_MMQ_GEMM_F32[0],
    OPS_MMQ_GEMM_F32[1],
    OPS_MMQ_GEMM_F32[2],
    OperandV1 {
        name: "Y",
        format: FORMAT_F32_N_M,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_M, Expr::Const(4)])),
        nullable: false,
        note: "F32 output `[N, M]` holding BF16-rounded values (`y.dtype == F32`, `n*m*4` bytes in the launcher): the epilogue rounds each output to BF16 and widens it back; tail masked, every in-range element written once",
    },
];

const FIELDS_MMQ_GEMM: &[FieldV1] = &[
    ptr_field("Wq", 0, "Wq"),
    ptr_field("Xq", 8, "Xq"),
    ptr_field("Xs", 16, "Xs"),
    ptr_field("Y", 24, "Y"),
    i32_field("M", 32, P_M),
    i32_field("K", 36, P_K),
    i32_field("N", 40, P_N),
];

const DOMAIN_MMQ_GEMM: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_M), "no output rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(Rule::Positive(P_K), "empty reduction"),
    rule(
        Rule::MultipleOf(P_K, 64),
        "the pipelined schedule exists for the FAST weight path only: 68 B K64 slabs read as dwords",
    ),
    rule(
        Rule::AtMost(Expr::DivCeil(&P_N, 128), GRID_Y_MAX),
        "grid y (ceil(N/128)) must fit the HIP y-grid limit",
    ),
];

const GRID_MMQ_GEMM: [Expr; 3] = [Expr::DivCeil(&P_M, 128), Expr::DivCeil(&P_N, 128), ONE];

/// `gemm_q8_0_mmq_i8_t128x128_ps`: hsaco
/// `~/.hipfire_kernels/gfx1201/gemm_q8_0_mmq_i8_gfx12.36fe2ca22ee066a0.hsaco` (source byte-identical to
/// `kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`): args `Wq@0 Xq@8 Xs@16` (`read_only`) `Y@24` (`write_only`)
/// buffers, `M@32 K@36 N@40` by value, `.kernarg_segment_size` 44 / align 8, NO hidden words,
/// `.group_segment_fixed_size` 22528 (one LDS buffer: `128*80 + 128*80 + 2*128*4 + 2*128*4`),
/// `.max_flat_workgroup_size` 256. Launch: block 256 (`2 * BM`), grid `(ceil(M/128), ceil(N/128))`.
/// `Gpu::q8_a8_pick_tile` selects the 128x128 tile when the grid holds at least `Q8A8_FILL_BLOCKS` (128)
/// CTAs, FAST needs `K % 64 == 0` and a 4-byte aligned weight base, and `HIPFIRE_Q8A8_PS=0` keeps the plain
/// (two-LDS-buffer, 45056 B) schedule instead.
pub static GEMM_Q8_0_MMQ_I8_T128X128_PS: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mmq.q8_0_mmq_i8_t128x128_ps.f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_0_A8_MMQ_F32,
    params: PARAMS_MMQ_GEMM,
    operands: OPS_MMQ_GEMM_F32,
    fields: FIELDS_MMQ_GEMM,
    hidden: &[],
    explicit_bytes: 44,
    segment_bytes: 44,
    segment_align: 8,
    domain: DOMAIN_MMQ_GEMM,
    exports: &[ExportV1 {
        symbol: "gemm_q8_0_mmq_i8_t128x128_ps",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: MODULE_Q8_MMQ,
            source: SOURCE_Q8_MMQ,
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 22528,
        grid: GRID_MMQ_GEMM,
        consts: &[],
    }],
};

/// `gemm_q8_0_mmq_i8_t128x128_ps_bf16`: same object, kernarg layout (44 B / align 8, no hidden words), LDS
/// (22528) and launch as [`GEMM_Q8_0_MMQ_I8_T128X128_PS`]; the epilogue rounds every output to BF16
/// (`Gpu::gemm_q8_0_a8_gemm_bf16`, bit-identical to the F32 GEMM followed by `bf16_round_trip_f32_strided`).
pub static GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mmq.q8_0_mmq_i8_t128x128_ps.bf16_in_f32",
    version: 1,
    wave: 32,
    numeric: NUMERIC_Q8_0_A8_MMQ_BF16_RNE,
    params: PARAMS_MMQ_GEMM,
    operands: OPS_MMQ_GEMM_BF16,
    fields: FIELDS_MMQ_GEMM,
    hidden: &[],
    explicit_bytes: 44,
    segment_bytes: 44,
    segment_align: 8,
    domain: DOMAIN_MMQ_GEMM,
    exports: &[ExportV1 {
        symbol: "gemm_q8_0_mmq_i8_t128x128_ps_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: MODULE_Q8_MMQ,
            source: SOURCE_Q8_MMQ,
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 22528,
        grid: GRID_MMQ_GEMM,
        consts: &[],
    }],
};

/// `ps` (F32 output): the Raon launches of this symbol are the MLA `kv_b` expansion, sub-chunked at
/// `MLA_TILE_ROWS` = 512 rows (grid y = 4 -> N in [385, 512], while the pass chunk is 4096), so `N` is NOT
/// bound to the chunk rows: it binds to the tile height 512 (MACs are strict, so a bare interval would leave
/// the leaf without a floor), which the binder verifies against the grid interval.
const BIND_MMQ_PS: &[BindV1] = &[
    BindV1 {
        param: "N",
        source: BindSource::Derived(Expr::Const(512)),
    },
    BIND_Q8_K,
];

/// `ps_bf16`: the Raon launches run on the whole 4096-row chunk (grid y = 32 -> N in [3969, 4096]).
const BIND_MMQ_PS_BF16: &[BindV1] = &[
    BindV1 {
        param: "N",
        source: BindSource::ChunkRows,
    },
    BIND_Q8_K,
];

pub static WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_Q8_0_MMQ_I8_T128X128_PS,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "q8_0",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_Q8_A8,
        bind: BIND_MMQ_PS,
        note: "MACs = M*N*K (128x128 tile padding excluded): iu8 WMMA over int8 weight codes x int8 activation codes, \
               the per-block scale folds (`fmaf` per output per 32-K block) are VALU work not charged. Bytes = sum of \
               the operand extents: weights once (M*K/32*34) + int8 X (N*K) + F32 scales (K/32*N*4) + F32 Y (N*M*4). \
               N binds to 512 = the MLA tile height (`crates/hipfire-arch-raon/src/state.rs::MLA_TILE_ROWS`: the Raon \
               `kv_b` launches are 512-row sub-chunks of the 4096-row pass, not the pass chunk); the grid (y = 4) only \
               gives N in [385, 512] and the binder checks the Derived value against it (a launch outside the interval \
               is reported inconsistent). Selector: \
               `q8_a8_pick_tile` takes 128x128 when ceil(M/128)*ceil(N/128) >= 128 CTAs (a grid predicate, not a row \
               one); `HIPFIRE_Q8A8_PS=0` routes the plain `gemm_q8_0_mmq_i8_t128x128` instead; K % 64 != 0 takes the \
               `_slow` kernels.",
    },
};

pub static WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16: WorkSpecV1 = WorkSpecV1 {
    abi: &GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
    work: WorkV1 {
        op_class: op_class::GEMM,
        codec: "q8_0",
        pipe: Pipe::WmmaIu8,
        macs: Some(Expr::Mul(&[P_M, P_N, P_K])),
        flops: None,
        bytes: None,
        rows: Some("N"),
        admit: ADMIT_Q8_A8,
        bind: BIND_MMQ_PS_BF16,
        note: "MACs = M*N*K (128x128 tile padding excluded) on the iu8 WMMA pipe; the scale folds and the BF16 \
               round are VALU work not charged. Bytes = sum of the operand extents: weights once (M*K/32*34) + int8 \
               X (N*K) + F32 scales (K/32*N*4) + F32 Y (N*M*4; the plane is F32, BF16-rounded values). N binds to \
               the chunk rows (Raon: grid y = 32 -> N in [3969, 4096], the whole 4096-row chunk); K from the Q8_0 \
               weight match on M (grid x interval of 128). `gemm_q8_0_a8_gemm_bf16` is called by the \
               prepared-activation projection path when the caller wants BF16-rounded output; `HIPFIRE_Q8A8_PS=0` \
               routes the plain `gemm_q8_0_mmq_i8_t128x128_bf16`. The 5120-row node (g/q/k/v) shares M with f_b \
               [5120,128]: the weight-set selection by call count may report several fitting sets.",
    },
};

// ================================================================================================
// quantize_x_q8_blk32
// ================================================================================================

const PARAMS_QUANT_X: &[ParamV1] = &[
    param("K", "row length in elements, a multiple of 32"),
    param("N", "activation rows (folded into grid x with K)"),
    param_u32(
        "blocks",
        "grid x (workgroups): ceil(N * (K/4) / 256) (not a kernarg; the folded 1-D launch is the only thing the grid determines)",
    ),
];

const NK4: Expr = Expr::Mul(&[P_N, Expr::DivExact(&P_K, 4)]);
const QUANT_X_BLOCKS: Expr = Expr::DivCeil(&NK4, 256);

const OPS_QUANT_X: &[OperandV1] = &[
    OperandV1 {
        name: "X",
        format: FORMAT_F32_ROWS_NK,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[N, K]`, read as float4 (4 elements per thread); lanes past N*K/4 read nothing",
    },
    OperandV1 {
        name: "Xq",
        format: FORMAT_I8_N_K,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[P_N, P_K])),
        nullable: false,
        note: "int8 codes `[N, K]`, one dword (4 codes) stored per thread",
    },
    OperandV1 {
        name: "Xs",
        format: FORMAT_F32_KB_N_SCALES,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivExact(&P_K, 32), P_N, Expr::Const(4)])),
        nullable: false,
        note: "F32 scales `[K/32, N]`, written by the first of the 8 lanes of each 32-K block",
    },
];

const FIELDS_QUANT_X: &[FieldV1] = &[
    ptr_field("X", 0, "X"),
    ptr_field("Xq", 8, "Xq"),
    ptr_field("Xs", 16, "Xs"),
    i32_field("K", 24, P_K),
    i32_field("N", 28, P_N),
];

/// The kernel reads `blockDim` (hidden words 32..98): offsets from the metadata of
/// `gemm_q8_0_mmq_i8_gfx12.36fe2ca22ee066a0` (same layout as the other hipcc kernels of this ROCm).
const HIDDEN_QUANT_X: &[HiddenV1] = &[
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

const DOMAIN_QUANT_X: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_K), "empty rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::MultipleOf(P_K, 32),
        "8 consecutive lanes (4 elements each) are one 32-K block: K is whole blocks (quads_per_row % 8 == 0)",
    ),
    rule(
        Rule::Equal(P_BLOCKS, QUANT_X_BLOCKS),
        "launcher: groups = ceil(n * (k/4) / 256), a folded 1-D grid",
    ),
    rule(
        Rule::AtMost(P_BLOCKS, u32::MAX as u64),
        "grid x is a u32 (the launcher errors beyond it)",
    ),
];

/// `quantize_x_q8_blk32`: the activation producer of the W8A8 Q8_0 route. hsaco
/// `~/.hipfire_kernels/gfx1201/gemm_q8_0_mmq_i8_gfx12.36fe2ca22ee066a0.hsaco` (source byte-identical to
/// `kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`): args `X@0` (`read_only`) `Xq@8 Xs@16` (`write_only`), `K@24 N@28` by
/// value, 13 hidden words at 32..98, `.kernarg_segment_size` 288 / align 8, LDS 0, `.max_flat_workgroup_size`
/// 1024 (hipcc default; the launch block is 256). Grid `[ceil(N*(K/4)/256), 1, 1]` (`Gpu::q8_a8_quantize_x`).
pub static QUANTIZE_X_Q8_BLK32: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mmq.quantize_x_q8_blk32.f32_to_i8_f32scales",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QUANTIZE_X_Q8_BLK32,
    params: PARAMS_QUANT_X,
    operands: OPS_QUANT_X,
    fields: FIELDS_QUANT_X,
    hidden: HIDDEN_QUANT_X,
    explicit_bytes: 32,
    segment_bytes: 288,
    segment_align: 8,
    domain: DOMAIN_QUANT_X,
    exports: &[ExportV1 {
        symbol: "quantize_x_q8_blk32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: MODULE_Q8_MMQ,
            source: SOURCE_Q8_MMQ,
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [P_BLOCKS, ONE, ONE],
        consts: &[],
    }],
};

pub static WORK_QUANTIZE_X_Q8_BLK32: WorkSpecV1 = WorkSpecV1 {
    abi: &QUANTIZE_X_Q8_BLK32,
    work: WorkV1 {
        op_class: op_class::ROTATE,
        codec: "f32>i8_blk32",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        // Per workgroup: 256 threads x 4 elements = 1024 elements: F32 read 4096 B + int8 write 1024 B + F32 scales
        // 1024/32*4 = 128 B.
        bytes: Some(Expr::Mul(&[P_BLOCKS, Expr::Const(5248)])),
        rows: None,
        admit: ADMIT_Q8_A8,
        bind: &[],
        note: "Memory-bound producer: bytes = blocks * 5248 (per 1024-element workgroup: 4096 B F32 in + 1024 B int8 out + 128 B \
               scales out), exact when N*K % 1024 == 0 (every observed Raon grid: 20480 = 4096x5120, 10240 = 4096x2560, \
               2304 = 4096x576, 256 = 512x512) and otherwise an over-count of less than one tile. The VALU work \
               (abs/max, one IEEE divide + rint + clamp per element) is not charged: it is a few % of the memory time. The grid \
               fixes only N*K, never N alone (the 256-block launches are the 512-row kv_b sub-chunks, not the 4096-row chunk), so no rows \
               param is bound; admission follows the W8A8 route (>= 64 rows).",
    },
};

// ================================================================================================
// quantize_int8_mmq_ds128
// ================================================================================================

const PARAMS_QUANT_DS128: &[ParamV1] = &[
    param("K", "row length in elements, a multiple of 128"),
    param(
        "N",
        "activation rows (folded into grid x: ceil(K/1024) per row)",
    ),
    param_u32(
        "blocks",
        "grid x (workgroups): ceil(K/1024) * N (not a kernarg; the folded 1-D launch)",
    ),
];

const DS128_BLOCKS: Expr = Expr::Mul(&[Expr::DivCeil(&P_K, 1024), P_N]);

const OPS_QUANT_DS128: &[OperandV1] = &[
    OperandV1 {
        name: "X",
        format: FORMAT_F32_ROWS_NK,
        access: Access::ReadOnly,
        align: 16,
        extent: Some(Expr::Mul(&[P_N, P_K, Expr::Const(4)])),
        nullable: false,
        note: "F32 activations `[N, K]` (the rotated x), read as float4; threads past K/4 per row return",
    },
    OperandV1 {
        name: "Y",
        format: FORMAT_BLOCK_I8_128_KB_N,
        access: Access::WriteOnly,
        align: 4,
        extent: Some(Expr::Mul(&[Expr::DivExact(&P_K, 128), P_N, Expr::Const(136)])),
        nullable: false,
        note: "`block_i8_128` records `[K/128][N]`, 136 B each (`float d; int s; signed char qs[128]`); one wave32 writes one record: Y[kblock * N + token]",
    },
];

const FIELDS_QUANT_DS128: &[FieldV1] = &[
    ptr_field("X", 0, "X"),
    ptr_field("Y", 8, "Y"),
    i32_field("K", 16, P_K),
    i32_field("N", 20, P_N),
];

/// Hidden block (the kernel reads `blockDim`): offsets from the metadata of
/// `gemm_mq4g256v2_residual_mmq_i8_gfx12.63383e7e3ba89ce2`, entry `quantize_int8_mmq_ds128`. The explicit args
/// end at 24, so the words sit 8 bytes below those of [`HIDDEN_QUANT_X`].
const HIDDEN_QUANT_DS128: &[HiddenV1] = &[
    hip_hidden(24, 4, HiddenKind::BlockCount(Axis::X)),
    hip_hidden(28, 4, HiddenKind::BlockCount(Axis::Y)),
    hip_hidden(32, 4, HiddenKind::BlockCount(Axis::Z)),
    hip_hidden(36, 2, HiddenKind::GroupSize(Axis::X)),
    hip_hidden(38, 2, HiddenKind::GroupSize(Axis::Y)),
    hip_hidden(40, 2, HiddenKind::GroupSize(Axis::Z)),
    hip_hidden(42, 2, HiddenKind::Remainder(Axis::X)),
    hip_hidden(44, 2, HiddenKind::Remainder(Axis::Y)),
    hip_hidden(46, 2, HiddenKind::Remainder(Axis::Z)),
    hip_hidden(64, 8, HiddenKind::GlobalOffset(Axis::X)),
    hip_hidden(72, 8, HiddenKind::GlobalOffset(Axis::Y)),
    hip_hidden(80, 8, HiddenKind::GlobalOffset(Axis::Z)),
    hip_hidden(88, 2, HiddenKind::GridDims),
];

const DOMAIN_QUANT_DS128: &[DomainRuleV1] = &[
    rule(Rule::Positive(P_K), "empty rows"),
    rule(Rule::Positive(P_N), "no activation rows"),
    rule(
        Rule::MultipleOf(P_K, 128),
        "one wave32 quantizes whole 128-K groups (block_i8_128 records)",
    ),
    rule(
        Rule::Equal(P_BLOCKS, DS128_BLOCKS),
        "launcher: grid = fold_rows_into_x(ceil(K/1024), N) = ceil(K/1024) * N workgroups of 256 threads x 4 elements",
    ),
    rule(
        Rule::AtMost(P_BLOCKS, i32::MAX as u64),
        "fold_rows_into_x refuses a folded grid.x above i32::MAX",
    ),
];

/// `quantize_int8_mmq_ds128`: the standalone K128 int8 activation producer (`block_i8_128`) of the A8 MMQ routes.
/// The runtime loads it from the module `gemm_mq4g256v2_residual_mmq_i8_gfx12`
/// (`GEMM_MQ4G256V2_RESIDUAL_MMQ_I8_GFX12_SRC` = `block_i8_128_quant.hip` ++ the gemm source); the symbol and its
/// layout come from the first file alone, which compiles standalone and is what `source` names. hsaco
/// `~/.hipfire_kernels/gfx1201/gemm_mq4g256v2_residual_mmq_i8_gfx12.63383e7e3ba89ce2.hsaco` (sibling `.hip`
/// byte-identical to that concatenation): args `X@0` (`read_only`) `Y@8` (`write_only`), `K@16 N@20` by value,
/// 13 hidden words at 24..90, `.kernarg_segment_size` 280 / align 8, LDS 0, `.max_flat_workgroup_size` 1024 (launch
/// block 256). gfx1100 launches the same symbol from `gemm_mq4g256v2_residual_mmq_i8_gfx11`; no gfx1100 object was
/// inspected, so it is not an export here.
pub static QUANTIZE_INT8_MMQ_DS128: KernelAbiV1 = KernelAbiV1 {
    id: "generic_mmq.quantize_int8_mmq_ds128.f32_to_block_i8_128",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QUANTIZE_INT8_MMQ_DS128,
    params: PARAMS_QUANT_DS128,
    operands: OPS_QUANT_DS128,
    fields: FIELDS_QUANT_DS128,
    hidden: HIDDEN_QUANT_DS128,
    explicit_bytes: 24,
    segment_bytes: 280,
    segment_align: 8,
    domain: DOMAIN_QUANT_DS128,
    exports: &[ExportV1 {
        symbol: "quantize_int8_mmq_ds128",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: MODULE_I8_128,
            source: SOURCE_I8_128,
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [P_BLOCKS, ONE, ONE],
        consts: &[],
    }],
};

pub static WORK_QUANTIZE_INT8_MMQ_DS128: WorkSpecV1 = WorkSpecV1 {
    abi: &QUANTIZE_INT8_MMQ_DS128,
    work: WorkV1 {
        op_class: op_class::ROTATE,
        codec: "f32>block_i8_128",
        pipe: Pipe::Memory,
        macs: None,
        flops: None,
        // Per workgroup: 256 threads x 4 elements = 1024 K elements = 8 block_i8_128 records: 4096 B in + 8*136 B out.
        bytes: Some(Expr::Mul(&[P_BLOCKS, Expr::Const(5184)])),
        rows: Some("N"),
        admit: ADMIT_Q8_A8,
        bind: &[BindV1 {
            param: "N",
            source: BindSource::ChunkRows,
        }],
        note: "Memory-bound producer: bytes = blocks * 5184 (per 1024-K workgroup: 4096 B F32 in + 8 x 136 B records out). This is an \
               UPPER BOUND when K % 1024 != 0 (the last workgroup of a row is partial): Raon K = 2560 -> 3 workgroups \
               per row, 12288 = 3 x 4096, so the bytes over-count by 1024*3/2560 - 1 = 20%. The grid fixes only ceil(K/1024)*N, \
               never K, so K is not bound. N binds to the chunk rows (the MoE input rows of the prefill chunk; the observed 12288-block \
               grid is consistent with N = 4096), a guess the grid cannot confirm. The VALU work (abs/max shuffles, IEEE divide + rint \
               per element) is not charged. Admission follows the quantized prefill tiers (>= 64 rows).",
    },
};

// ================================================================================================

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &GEMM_Q8_0_MMQ_I8_T128X128_PS,
    &GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
    &QUANTIZE_X_Q8_BLK32,
    &QUANTIZE_INT8_MMQ_DS128,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS,
    &WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
    &WORK_QUANTIZE_X_Q8_BLK32,
    &WORK_QUANTIZE_INT8_MMQ_DS128,
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

    /// Interval of the one param a grid expression pins at workgroup count `g`.
    fn invert(e: &Expr, g: u64) -> Option<(&'static str, u64, u64)> {
        match e {
            Expr::Param(p) => Some((p, g, g)),
            Expr::DivCeil(inner, d) => match **inner {
                Expr::Param(p) => Some((p, (g - 1) * d + 1, g * d)),
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

    fn params_vec(abi: &KernelAbiV1, given: &[(&str, u64)]) -> Vec<u64> {
        abi.params
            .iter()
            .map(|p| {
                given
                    .iter()
                    .find(|(n, _)| *n == p.name)
                    .unwrap_or_else(|| panic!("param {} not given", p.name))
                    .1
            })
            .collect()
    }

    fn eval_grid(abi: &KernelAbiV1, ex: &ExportV1, given: &[(&str, u64)]) -> [u64; 3] {
        let vals = params_vec(abi, given);
        let mut out = [0u64; 3];
        for (i, g) in ex.grid.iter().enumerate() {
            out[i] = eval(g, abi, ex, &vals).unwrap();
        }
        out
    }

    /// The observed workgroup grid inverts to an interval holding the stated param, and the stated params
    /// re-evaluate to the same workgroup counts.
    fn check_grid(
        abi: &KernelAbiV1,
        symbol: &str,
        wg: [u64; 3],
        stated: &[(&str, u64)],
        axes: [Option<&str>; 2],
    ) {
        let ex = export(abi, Arch::Gfx1201, symbol);
        for (axis, name) in axes.iter().enumerate() {
            if let Some(name) = name {
                let (lo, hi) = interval(&ex.grid[axis], wg[axis], name);
                let v = stated.iter().find(|(n, _)| n == name).unwrap().1;
                assert!(
                    (lo..=hi).contains(&v),
                    "{symbol}: {name} = {v} outside the inverted [{lo}, {hi}] of grid {wg:?}"
                );
            }
        }
        assert_eq!(
            eval_grid(abi, ex, stated),
            wg,
            "{symbol}: stated params do not give grid {wg:?}"
        );
    }

    /// Every `Rule::Equal` and the other domain rules of the contract hold on `given`.
    fn domain_holds(abi: &KernelAbiV1, symbol: &str, given: &[(&str, u64)]) {
        let ex = export(abi, Arch::Gfx1201, symbol);
        let vals = params_vec(abi, given);
        for d in abi.domain {
            let ok = match d.rule {
                Rule::Positive(e) => eval(&e, abi, ex, &vals).unwrap() > 0,
                Rule::MultipleOf(e, m) => eval(&e, abi, ex, &vals).unwrap() % m == 0,
                Rule::AtMost(e, l) => eval(&e, abi, ex, &vals).unwrap() <= l,
                Rule::AtLeast(e, l) => eval(&e, abi, ex, &vals).unwrap() >= l,
                Rule::Equal(a, b) => {
                    eval(&a, abi, ex, &vals).unwrap() == eval(&b, abi, ex, &vals).unwrap()
                }
                Rule::OneOf(e, set) => set.contains(&eval(&e, abi, ex, &vals).unwrap()),
            };
            assert!(
                ok,
                "{}: domain rule {:?} fails on {given:?} ({})",
                abi.id, d.rule, d.why
            );
        }
    }

    #[test]
    fn every_contract_self_checks() {
        assert_eq!(ABIS.len(), 4);
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
                assert!(text.contains(e.symbol), "{source} lacks {}", e.symbol);
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
            assert_eq!(w.work.admit.min_rows, 64, "{}", w.abi.id);
        }
    }

    #[test]
    fn layouts_match_the_compiled_metadata() {
        // (abi, explicit, segment, hidden words, static LDS, max block)
        for (abi, explicit, segment, hidden, lds) in [
            (&GEMM_Q8_0_MMQ_I8_T128X128_PS, 44, 44, 0, 22528),
            (&GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16, 44, 44, 0, 22528),
            (&QUANTIZE_X_Q8_BLK32, 32, 288, 13, 0),
            (&QUANTIZE_INT8_MMQ_DS128, 24, 280, 13, 0),
        ] {
            assert_eq!(abi.explicit_bytes, explicit, "{}", abi.id);
            assert_eq!(abi.segment_bytes, segment, "{}", abi.id);
            assert_eq!(abi.segment_align, 8, "{}", abi.id);
            assert_eq!(abi.hidden.len(), hidden, "{}", abi.id);
            assert_eq!(abi.exports.len(), 1, "{}", abi.id);
            let e = &abi.exports[0];
            assert_eq!(e.arch, Arch::Gfx1201);
            assert_eq!(e.group_segment_fixed, lds, "{}", abi.id);
            assert_eq!(e.dynamic_lds, 0);
            assert_eq!(e.block, [256, 1, 1]);
        }
        // Hidden words end at the grid-dims word, 2 bytes wide, inside the segment.
        let last = QUANTIZE_X_Q8_BLK32.hidden.last().unwrap();
        assert_eq!((last.offset, last.width), (96, 2));
        let last = QUANTIZE_INT8_MMQ_DS128.hidden.last().unwrap();
        assert_eq!((last.offset, last.width), (88, 2));
    }

    // ---- observed Raon pp4096 grids (workgroups, from pmrip/ev/moe1/run1/nodes.json) -------------

    #[test]
    fn raon_ps_bf16_grids_invert() {
        // x = ceil(M/128): 40 -> M in [4993, 5120] (g/q/k/v [5120, 2560]); 20 -> [2433, 2560] (o_proj [2560, 5120]);
        // 60 -> [7553, 7680] (q_b [7680, 576]); 5 -> [513, 640] (q_a / kv_a [576, 2560]); y = 32 -> N in [3969, 4096].
        for (gx, m, k) in [
            (40u64, 5120u64, 2560u64),
            (20, 2560, 5120),
            (60, 7680, 576),
            (5, 576, 2560),
        ] {
            check_grid(
                &GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
                "gemm_q8_0_mmq_i8_t128x128_ps_bf16",
                [gx, 32, 1],
                &[("M", m), ("K", k), ("N", 4096)],
                [Some("M"), Some("N")],
            );
            domain_holds(
                &GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
                "gemm_q8_0_mmq_i8_t128x128_ps_bf16",
                &[("M", m), ("K", k), ("N", 4096)],
            );
        }
        let ex = export(
            &GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
            Arch::Gfx1201,
            "gemm_q8_0_mmq_i8_t128x128_ps_bf16",
        );
        assert_eq!(interval(&ex.grid[1], 32, "N"), (3969, 4096));
        assert_eq!(interval(&ex.grid[0], 40, "M"), (4993, 5120));
        assert_eq!(interval(&ex.grid[0], 20, "M"), (2433, 2560));
        assert_eq!(interval(&ex.grid[0], 60, "M"), (7553, 7680));
        assert_eq!(interval(&ex.grid[0], 5, "M"), (513, 640));
    }

    #[test]
    fn raon_ps_grid_inverts_to_the_kv_b_subchunk() {
        // (80, 4): M in [10113, 10240] = kv_b [10240, 512]; N in [385, 512] = the 512-row sub-chunk (pass chunk 4096).
        let abi = &GEMM_Q8_0_MMQ_I8_T128X128_PS;
        check_grid(
            abi,
            "gemm_q8_0_mmq_i8_t128x128_ps",
            [80, 4, 1],
            &[("M", 10240), ("K", 512), ("N", 512)],
            [Some("M"), Some("N")],
        );
        domain_holds(
            abi,
            "gemm_q8_0_mmq_i8_t128x128_ps",
            &[("M", 10240), ("K", 512), ("N", 512)],
        );
        let ex = export(abi, Arch::Gfx1201, "gemm_q8_0_mmq_i8_t128x128_ps");
        assert_eq!(interval(&ex.grid[0], 80, "M"), (10113, 10240));
        assert_eq!(interval(&ex.grid[1], 4, "N"), (385, 512));
    }

    #[test]
    fn raon_quantize_x_grids_invert() {
        // (blocks, N, K): the folded grid is the exact flat launch; N*K is pinned to a 1024-element tile.
        let abi = &QUANTIZE_X_Q8_BLK32;
        for (blocks, n, k) in [
            (20480u64, 4096u64, 5120u64), // o_proj input
            (10240, 4096, 2560),          // g/q/k/v and q_a/kv_a input
            (2304, 4096, 576),            // q_b input
            (256, 512, 512),              // kv_b input, 512-row sub-chunk
        ] {
            let given = [("K", k), ("N", n), ("blocks", blocks)];
            check_grid(
                abi,
                "quantize_x_q8_blk32",
                [blocks, 1, 1],
                &given,
                [Some("blocks"), None],
            );
            domain_holds(abi, "quantize_x_q8_blk32", &given);
            // The Raon shapes are exact multiples of one workgroup's 1024 elements.
            assert_eq!(n * k, blocks * 1024);
        }
    }

    #[test]
    fn raon_quantize_int8_mmq_ds128_grid_inverts() {
        // 12288 = ceil(2560/1024) * 4096.
        let abi = &QUANTIZE_INT8_MMQ_DS128;
        let given = [("K", 2560u64), ("N", 4096), ("blocks", 12288)];
        check_grid(
            abi,
            "quantize_int8_mmq_ds128",
            [12288, 1, 1],
            &given,
            [Some("blocks"), None],
        );
        domain_holds(abi, "quantize_int8_mmq_ds128", &given);
    }

    #[test]
    fn work_macs_and_bytes_evaluate_on_the_raon_shapes() {
        // q/k/v/g: M=5120 K=2560 N=4096.
        let w = &WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16;
        let ex = &w.abi.exports[0];
        let vals = [5120u64, 2560, 4096];
        assert_eq!(
            eval(&w.work.macs.unwrap(), w.abi, ex, &vals).unwrap(),
            5120 * 2560 * 4096
        );
        // Bytes default to the operand extents: weights once + int8 X + F32 scales + F32 Y.
        let sum: u64 = w
            .abi
            .operands
            .iter()
            .map(|o| eval(&o.extent.unwrap(), w.abi, ex, &vals).unwrap())
            .sum();
        assert_eq!(
            sum,
            5120 * 80 * 34 + 4096 * 2560 + 80 * 4096 * 4 + 4096 * 5120 * 4
        );
        assert!(w.work.bytes.is_none());
        // quantize_x (K=2560, N=4096): exact bytes = N*K*(4 + 1) + K/32*N*4.
        let w = &WORK_QUANTIZE_X_Q8_BLK32;
        let ex = &w.abi.exports[0];
        let got = eval(&w.work.bytes.unwrap(), w.abi, ex, &[2560, 4096, 10240]).unwrap();
        assert_eq!(got, 4096 * 2560 * 5 + 80 * 4096 * 4);
        // ds128 (K=2560, N=4096, 12288 blocks): upper bound over the exact N*K*4 + K/128*N*136.
        let w = &WORK_QUANTIZE_INT8_MMQ_DS128;
        let ex = &w.abi.exports[0];
        let got = eval(&w.work.bytes.unwrap(), w.abi, ex, &[2560, 4096, 12288]).unwrap();
        assert_eq!(got, 12288 * 5184);
        assert!(got >= 4096 * 2560 * 4 + 20 * 4096 * 136);
    }

    #[test]
    fn bindings_and_pipes_follow_the_spec() {
        for w in [
            &WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS,
            &WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16,
        ] {
            assert_eq!(w.work.op_class, op_class::GEMM);
            assert_eq!(w.work.codec, "q8_0");
            assert_eq!(w.work.pipe, Pipe::WmmaIu8);
            assert_eq!(w.work.rows, Some("N"));
            let Some(BindSource::Weight {
                by: "M",
                by_axis: WeightAxis::Rows,
                codecs: &["q8_0"],
                stacking: Stacking::Single,
                fuse_siblings: false,
            }) = w.work.bind.last().map(|b| b.source)
            else {
                panic!("{}: last bind is not the Q8_0 weight match", w.abi.id);
            };
            assert_eq!(w.work.bind.last().unwrap().param, "K");
        }
        // bf16 binds N to the chunk rows; the F32 `ps` (512-row kv_b sub-chunks) binds N to the tile height
        // (checked against the grid interval by the binder).
        assert!(matches!(
            WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16.work.bind[0].source,
            BindSource::ChunkRows
        ));
        assert_eq!(
            WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16.work.bind[0].param,
            "N"
        );
        assert_eq!(WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS.work.bind[0].param, "N");
        assert!(matches!(
            WORK_GEMM_Q8_0_MMQ_I8_T128X128_PS.work.bind[0].source,
            BindSource::Derived(Expr::Const(512))
        ));
        for w in [&WORK_QUANTIZE_X_Q8_BLK32, &WORK_QUANTIZE_INT8_MMQ_DS128] {
            assert_eq!(w.work.op_class, op_class::ROTATE);
            assert_eq!(w.work.pipe, Pipe::Memory);
            assert!(w.work.macs.is_none());
        }
        // The two GEMMs share every operand but carry distinct numerics.
        assert_ne!(
            GEMM_Q8_0_MMQ_I8_T128X128_PS.numeric,
            GEMM_Q8_0_MMQ_I8_T128X128_PS_BF16.numeric
        );
    }
}
