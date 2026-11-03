// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raon RDA pp4096 kernels with no contract in `generic_seq`: the chunked (C64) gated delta rule
//! (`gated_delta_vector_chunk_prep_gfx12`, `gated_delta_vector_chunk_scan_gfx12`), the fused
//! conv + prepare producer (`raon_rda_conv_prepare_w128_f32`) and the one-wave-per-(row, head) gated
//! RMSNorm (`gated_rmsnorm_sigmoid_rows_w128_f32`). gfx1201 only (every launcher refuses other archs:
//! `chunked_supported` / `rda_wave_kernels_arch_supported`); the conv + prepare producer is also admitted on exact
//! gfx1100 (`rda_conv_prepare_arch_supported`).
//!
//! Evidence.
//!
//! * Kernarg layouts and static LDS are read from the JIT-cached gfx1201 objects whose sibling `.hip`
//!   equals the current `kernels/src` source (`cmp`): `gated_delta_vector_chunk_gfx1201.3ad47537f75319f0`,
//!   `raon_rda_conv_prepare.d3c6fd13e1de2ba1`, `gated_rmsnorm_sigmoid_rows_w128.5dcbfc98afd340a9`
//!   (clang offload bundle unbundled with `clang-offload-bundler --type=o
//!   --targets=hipv4-amdgcn-amd-amdhsa--gfx1201`, notes via `llvm-readelf --notes`). Measured:
//!   prep `.kernarg_segment_size` 352 (explicit 96, hidden block at 96), `.group_segment_fixed_size` 58624;
//!   scan 48 / 0 (no hidden block); conv_prepare 208 / 0; rmsnorm_w128 64 / 0. Modules compile from the
//!   bare `.hip` text (no prelude), exactly what `kernel_abi_drift::runtime_source` hands hipcc.
//! * The chunk prep kernel changes often: the pp4096 trace in `pmrip/ev/moe1/run1` was recorded with an
//!   earlier build whose static LDS was 58880; the contract states the current source's 58624 (the binder
//!   records the difference as an `lds:` note, not a mismatch).
//! * Launch geometry: `crates/rdna-compute/src/gated_delta_vector_chunk.rs` (prep grid
//!   `min(2 * multiprocessors, chunks * n_heads)`, scan grid `n_heads * 8`) and
//!   `crates/rdna-compute/src/gated_delta_vector.rs` (conv_prepare / rmsnorm_w128: one wave32 per tile, 8
//!   waves per block).
//! * Launch rows: the Raon RDA layer runs segmented (`rda.rs::RDA_SEGMENT_ROWS` = 256 rows per launch,
//!   `min(chunk scratch rows 512, 256)`), so every kernel here sees 256-row launches (18 RDA layers x
//!   16 segments = 288 calls per 4096-row pass, the observed call count) while the pass chunk is 4096
//!   rows. The grid of the chunk kernels does not carry rows, so the work specs bind the launch rows as a
//!   `Derived` constant (256 = `RDA_SEGMENT_ROWS`), NOT `ChunkRows`; conv_prepare and rmsnorm_w128 cross-check it
//!   against the observed grid through the `waves` param (a bind that falls outside the grid interval is
//!   reported `inconsistent` by the binder).
//!
//! Conventions as in `generic_seq`: f32 kernel scalars are `U32` params named `*_bits`; the contracts pin
//! `head_dim == 128` (the kernels are specialised: `CK_D`, `block.x = 256 = 8 waves x 32` etc.).

use crate::work::{op_class, BindSource, BindV1, Pipe, RowDomainV1, WorkSpecV1, WorkV1};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_convert::FORMAT_F32_FLAT;

// ---- formats ---------------------------------------------------------------------

/// Chunk workspace of the chunked RDA route: `[chunk][head]` blocks of 123392 bytes (layout in the kernel
/// header: `-W'` hi, `Q'`, `Kend^T` hi/lo, `U'^T` f32, `M`, `exp(ell_T)`, `-W'` lo).
pub const FORMAT_RDA_CHUNK_SCRATCH: FormatId = FormatId("rda.chunk_scratch.c64.d128.v1");
/// Debug timestamp words of the chunk prep kernel: 8 `u64` per (chunk, head).
pub const FORMAT_U64_PROF_STAMPS: FormatId = FormatId("u64.prof_stamps.8_per_chunk_head");

// ---- numeric contracts -------------------------------------------------------------

/// `gated_delta_vector_chunk_prep_gfx12`. Per (C64 chunk, head): chunk-relative cumulative log decay `ell`
/// in f32, per-token decay clamped to `[-10, 0]`, summed in the fixed order (8 groups of 8 tokens, each
/// group scanned sequentially, then the group totals added in group order); every exponent a causal
/// relative one (`<= 0`, diagonal 16-row blocks bounded by `exp(8 * 10)`); pair matrices `G` (k.k) and `M`
/// (q.k) from bf16 hi+lo operand splits (`hi*hi + hi*lo + lo*hi`, f32 WMMA accumulation over the 128
/// channels in 8 steps of 16, ~2^-16 relative); `L = diag(beta) G` stored as hi+lo f16; `T' = (I + L)^-1`
/// by a scalar 16x16 block forward-substitution, f16 WMMA merges, then one Newton-Schulz step against the
/// exact `L` (hi+lo f16, f32 accumulation); `W'`, `Kend` and `U'` from hi+lo f16 splits with f32
/// accumulation (`-W'` and `Kend^T` stored as hi and lo f16 planes, `U'^T` as f32); `M` and `Q'` stored as
/// plain f16. Deterministic for a fixed shape (no atomics, fixed reduction order, persistent grid striding
/// over (chunk, head) pairs only changes which workgroup computes a pair, not its result).
pub const NUMERIC_GATED_DELTA_CHUNK_PREP: NumericContractId =
    NumericContractId("gated_delta.vector_chunk_c64.prep.f16x2_bf16x2_f32acc");

/// `gated_delta_vector_chunk_scan_gfx12`. One wave per (head, 16 value rows) carries the FP32 state
/// `[16 v][128 k]` in registers across chunks in order; per chunk four WMMA stages with f32 accumulation:
/// `Et = U' - W' S0^T` (state split hi+lo f16, `-W'` hi+lo), `Y = M U' + Q' S0^T` (plain f16 `M`, `U'`, `Q'`
/// and the high f16 half of the state), `S <- S diag(exp(ell_T)) + Et^T Kend` (`Et` and `Kend` hi+lo f16,
/// three products `hi*hi + hi*lo + lo*hi`). The output is f32, rounded to bf16 (RNE, NaN passthrough) when
/// `round_out_bf16`. Deterministic; chunk order is the only reduction across chunks.
pub const NUMERIC_GATED_DELTA_CHUNK_SCAN: NumericContractId =
    NumericContractId("gated_delta.vector_chunk_c64.scan.f16x2_f32state");

/// `raon_rda_conv_prepare_w128_f32`: bit-identical to the three `causal_conv_depthwise_silu_rows_f32` +
/// `raon_rda_prepare_f32` launches (kernel header): conv tap chain `(((0 + w0*a0) + w1*a1) + w2*a2) + w3*x` with
/// explicit `__fmul_rn/__fadd_rn` (no FMA contraction), `silu = acc / (1 + expf(-acc))`, RNE bf16 round
/// when `round_bf16`; q/k sums of squares by the incumbent's fixed tree (stages 64..1 over the 128
/// channels, evaluated as wave-shuffle offsets 16/8/4/2/1 then in-lane), `n = max(sqrtf(sum), eps)`,
/// `q = bf16(bf16(c / n) * q_scale)`, `k = bf16(c / n)`, gate `u = exp(a_log) * (f + dt_bias)`,
/// `log_alpha = lower_bound * (1 / (1 + expf(-u)))`, `beta = bf16(1 / (1 + expf(-b)))`,
/// `ratio = expf(logsigmoid(b + delta) - logsigmoid(b))`, `vprime = bf16(conv_v * ratio)`.
pub const NUMERIC_RAON_RDA_CONV_PREPARE: NumericContractId =
    NumericContractId("raon_rda.conv_prepare.f32");

/// `gated_rmsnorm_sigmoid_rows_w128_f32`: bit-identical to `gated_rmsnorm_sigmoid_rows_f32` (same
/// string id as that contract): `red` = sum of squares by the fixed tree (stage 64..1; shuffle offsets
/// 16/8/4/2/1 then in-lane), `inv = 1 / sqrtf(red / 128 + eps)`, `y = x * inv * gamma * (1 / (1 + expf(-gate)))`
/// multiplied left to right, RNE bf16 round when `round_bf16`.
pub const NUMERIC_GATED_RMSNORM_SIGMOID_ROWS_W128: NumericContractId =
    NumericContractId("gated_rmsnorm.sigmoid_rows.f32");

// ---- expression / layout helpers ---------------------------------------------------

macro_rules! p {
    ($n:literal) => {
        Expr::Param($n)
    };
}
macro_rules! c {
    ($v:expr) => {
        Expr::Const($v)
    };
}
macro_rules! mul {
    ($($x:expr),+ $(,)?) => { Expr::Mul(&[$($x),+]) };
}
macro_rules! add {
    ($($x:expr),+ $(,)?) => { Expr::Add(&[$($x),+]) };
}
macro_rules! sub {
    ($a:expr, $b:expr) => {
        Expr::Sub(&$a, &$b)
    };
}
/// `(rows - 1) * stride + width`.
macro_rules! strided {
    ($rows:expr, $stride:expr, $width:expr) => {
        add!(mul!(sub!($rows, c!(1)), $stride), $width)
    };
}
/// `4 * elems`.
macro_rules! f32b {
    ($e:expr) => {
        mul!($e, c!(4))
    };
}
/// `ceil(seg_rows / 64)` C64 chunks of a launch.
macro_rules! chunks {
    () => {
        Expr::DivCeil(&p!("seg_rows"), 64)
    };
}

/// Hidden block of an `amdgpu` kernel whose first hidden word is at `$b` (no dynamic-LDS word).
macro_rules! hidden_std {
    ($b:expr) => {
        &[
            HiddenV1 {
                offset: $b,
                width: 4,
                kind: HiddenKind::BlockCount(Axis::X),
            },
            HiddenV1 {
                offset: $b + 4,
                width: 4,
                kind: HiddenKind::BlockCount(Axis::Y),
            },
            HiddenV1 {
                offset: $b + 8,
                width: 4,
                kind: HiddenKind::BlockCount(Axis::Z),
            },
            HiddenV1 {
                offset: $b + 12,
                width: 2,
                kind: HiddenKind::GroupSize(Axis::X),
            },
            HiddenV1 {
                offset: $b + 14,
                width: 2,
                kind: HiddenKind::GroupSize(Axis::Y),
            },
            HiddenV1 {
                offset: $b + 16,
                width: 2,
                kind: HiddenKind::GroupSize(Axis::Z),
            },
            HiddenV1 {
                offset: $b + 18,
                width: 2,
                kind: HiddenKind::Remainder(Axis::X),
            },
            HiddenV1 {
                offset: $b + 20,
                width: 2,
                kind: HiddenKind::Remainder(Axis::Y),
            },
            HiddenV1 {
                offset: $b + 22,
                width: 2,
                kind: HiddenKind::Remainder(Axis::Z),
            },
            HiddenV1 {
                offset: $b + 40,
                width: 8,
                kind: HiddenKind::GlobalOffset(Axis::X),
            },
            HiddenV1 {
                offset: $b + 48,
                width: 8,
                kind: HiddenKind::GlobalOffset(Axis::Y),
            },
            HiddenV1 {
                offset: $b + 56,
                width: 8,
                kind: HiddenKind::GlobalOffset(Axis::Z),
            },
            HiddenV1 {
                offset: $b + 64,
                width: 2,
                kind: HiddenKind::GridDims,
            },
        ]
    };
}

const fn ip(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::I32,
        doc,
    }
}
const fn up(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::U32,
        doc,
    }
}
const fn f_ptr(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand: name },
    }
}
/// `i32` field `name` carrying param `param`.
const fn f_i32(name: &'static str, offset: u32, param: &'static str) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::I32,
            value: Expr::Param(param),
        },
    }
}
/// `f32` field `name`, carried as the `U32` bit-pattern param `param`.
const fn f_f32(name: &'static str, offset: u32, param: &'static str) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::U32,
            value: Expr::Param(param),
        },
    }
}
const fn opd(
    name: &'static str,
    format: FormatId,
    access: Access,
    align: u32,
    extent: Expr,
    note: &'static str,
) -> OperandV1 {
    OperandV1 {
        name,
        format,
        access,
        align,
        extent: Some(extent),
        nullable: false,
        note,
    }
}
const fn opd_null(
    name: &'static str,
    format: FormatId,
    access: Access,
    align: u32,
    extent: Expr,
    note: &'static str,
) -> OperandV1 {
    OperandV1 {
        name,
        format,
        access,
        align,
        extent: Some(extent),
        nullable: true,
        note,
    }
}
const fn rule(rule: Rule, why: &'static str) -> DomainRuleV1 {
    DomainRuleV1 { rule, why }
}
const fn bind(param: &'static str, source: BindSource) -> BindV1 {
    BindV1 { param, source }
}

const BOOL: &[u64] = &[0, 1];

/// Value heads (same synonyms as `generic_seq::CFG_HEADS`; that const is private).
const CFG_HEADS: BindSource = BindSource::Config(&[
    "rda_config.num_heads",
    "linear_num_value_heads",
    "num_attention_heads",
    "n_head",
    "num_heads",
]);
/// RDA head dim (same synonyms as `generic_seq::CFG_RDA_HEAD_DIM`).
const CFG_RDA_HEAD_DIM: BindSource =
    BindSource::Config(&["rda_config.head_dim", "linear_value_head_dim", "head_dim"]);

/// Rows of one RDA launch: `crates/hipfire-dispatch/src/pipeline/rda.rs::RDA_SEGMENT_ROWS` (the layer runs as 256-row
/// segments whenever the chunked core is on and the chunk has more rows).
const SEGMENT_ROWS: BindSource = BindSource::Derived(Expr::Const(256));

const CHUNK_ADMIT: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate:
        "crates/rdna-compute/src/gated_delta_vector_chunk.rs::GATED_DELTA_VECTOR_CHUNK_MIN_ROWS",
    kill_env: Some("HIPFIRE_GATED_DELTA_VECTOR_CHUNKED"),
};

const WAVE_ADMIT: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/rda.rs::RDA_WAVE_MIN_ROWS",
    kill_env: Some("HIPFIRE_RAON_RDA_FUSED_PREP"),
};

// ================================================================================
// gated_delta_vector_chunk_prep_gfx12
// ================================================================================

mod prep {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip(
            "seg_rows",
            "rows of this launch (one scratch segment); chunks = ceil(seg_rows / 64)",
        ),
        ip("n_heads", "value heads"),
        ip("qk_group", "value heads per q/k head (1 = no repeat)"),
        ip("decay_per_key", "1: log_decay is [T, H, K]; 0: [T, H]"),
        ip("q_rs", "query row stride, floats (multiple of 4)"),
        ip("k_rs", "key row stride, floats (multiple of 4)"),
        ip("v_rs", "value row stride, floats"),
        ip("a_rs", "log_decay row stride, floats (multiple of 4 when per key)"),
        ip("b_rs", "beta row stride, floats"),
        ip(
            "prep_grid",
            "grid x in workgroups = min(2 * multiprocessors, chunks * n_heads) (not a kernarg; persistent grid)",
        ),
        ip("n_qk_heads", "q/k heads = n_heads / qk_group (not a kernarg)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd("query", FORMAT_F32_FLAT, Access::ReadOnly, 16,
            f32b!(strided!(p!("seg_rows"), p!("q_rs"), mul!(p!("n_qk_heads"), p!("D")))),
            "[rows, Hqk, K] at q_rs; 16-byte loads"),
        opd("key", FORMAT_F32_FLAT, Access::ReadOnly, 16,
            f32b!(strided!(p!("seg_rows"), p!("k_rs"), mul!(p!("n_qk_heads"), p!("D")))),
            "[rows, Hqk, K] at k_rs; 16-byte loads"),
        opd("value", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("seg_rows"), p!("v_rs"), mul!(p!("n_heads"), p!("D")))),
            "[rows, H, V] at v_rs"),
        opd("log_decay", FORMAT_F32_FLAT, Access::ReadOnly, 16,
            f32b!(strided!(
                p!("seg_rows"),
                p!("a_rs"),
                mul!(p!("n_heads"), add!(c!(1), mul!(p!("decay_per_key"), c!(127))))
            )),
            "per key [rows, H, K] (16-byte loads) or per head [rows, H]"),
        opd("beta", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("seg_rows"), p!("b_rs"), p!("n_heads"))),
            "[rows, H]"),
        opd("scratch", FORMAT_RDA_CHUNK_SCRATCH, Access::ReadWrite, 16,
            mul!(chunks!(), p!("n_heads"), c!(123392)),
            "[chunk][head] blocks of 123392 B; the kernel writes every byte of a block and re-reads M from it"),
        opd_null("prof", FORMAT_U64_PROF_STAMPS, Access::WriteOnly, 8,
            mul!(chunks!(), p!("n_heads"), c!(64)),
            "nullable debug timestamps, 8 u64 per (chunk, head)"),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("query", 0),
        f_ptr("key", 8),
        f_ptr("value", 16),
        f_ptr("log_decay", 24),
        f_ptr("beta", 32),
        f_ptr("scratch", 40),
        f_i32("seg_rows", 48, "seg_rows"),
        f_i32("n_heads", 52, "n_heads"),
        f_i32("qk_group", 56, "qk_group"),
        f_i32("decay_per_key", 60, "decay_per_key"),
        f_i32("q_rs", 64, "q_rs"),
        f_i32("k_rs", 68, "k_rs"),
        f_i32("v_rs", 72, "v_rs"),
        f_i32("a_rs", 76, "a_rs"),
        f_i32("b_rs", 80, "b_rs"),
        f_ptr("prof", 88),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("seg_rows")), "seg_rows > 0"),
        rule(Rule::Positive(p!("n_heads")), "at least one head"),
        rule(Rule::Positive(p!("qk_group")), "qk_group >= 1"),
        rule(
            Rule::Equal(mul!(p!("n_qk_heads"), p!("qk_group")), p!("n_heads")),
            "n_heads % qk_group == 0, n_qk_heads = n_heads / qk_group",
        ),
        rule(
            Rule::OneOf(p!("decay_per_key"), BOOL),
            "decay_per_key is a flag",
        ),
        rule(
            Rule::MultipleOf(p!("q_rs"), 4),
            "16-byte query loads (launcher check)",
        ),
        rule(
            Rule::MultipleOf(p!("k_rs"), 4),
            "16-byte key loads (launcher check)",
        ),
        rule(
            Rule::MultipleOf(mul!(p!("decay_per_key"), p!("a_rs")), 4),
            "16-byte per-key log_decay loads (launcher check, per-key only)",
        ),
        rule(Rule::Positive(p!("prep_grid")), "at least one workgroup"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "gated_delta_vector_chunk_prep_gfx12",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gated_delta_vector_chunk_gfx1201",
            source: "kernels/src/gated_delta_vector_chunk.gfx1201.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        // struct CkLds: ell[64][132] f32 33792 + Lh/Ll/Th/Tl 4 x 5120 + union(2304 | 4096) + beta[64] f32 256.
        group_segment_fixed: 58624,
        grid: [p!("prep_grid"), c!(1), c!(1)],
        consts: &[("D", 128)],
    }];

    /// `gated_delta_vector_chunk_prep_gfx12`: chunk-parallel prepare of the C64 gated delta rule; a persistent
    /// grid of `min(2 * multiprocessors, chunks * n_heads)` workgroups of 256 threads (8 waves) strides over the
    /// (chunk, head) pairs. `.kernarg_segment_size` 352 (explicit 96, hidden block at 96), static LDS 58624,
    /// 256 VGPRs. Compiled object: `gated_delta_vector_chunk_gfx1201.3ad47537f75319f0`.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_rda.gated_delta_vector_chunk_prep_gfx12",
        version: 1,
        wave: 32,
        numeric: NUMERIC_GATED_DELTA_CHUNK_PREP,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(96),
        explicit_bytes: 96,
        segment_bytes: 352,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("n_heads", CFG_HEADS), bind("seg_rows", SEGMENT_ROWS)];

    /// Per (chunk, head), counted from the kernel source as issued WMMA 16x16x16 instructions (4096 MAC each):
    /// P1 pair matrices: 10 lower block pairs x 8 channel steps x 6 bf16 WMMA (G and M, hi*hi + hi*lo + lo*hi)
    /// = 480 bf16; P2 `T'`: 4 + 8 + 8 merges, residual `R` 20 (m, n, kk) triples x 2 (hi, lo of `L`) = 40,
    /// refinement `T1` 20 = 80 f16; P3 per channel-tile wave (8 waves): `W'` 10 (m, kk) x 3 = 30, `Q'` 10,
    /// `U'` 30 = 70 -> 560 f16. Total 1120 WMMA = 640 f16 + 480 bf16 (both peak at the same rate on every
    /// supported arch, so the whole count is charged to the f16 pipe).
    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::RECURRENCE,
            codec: "f32",
            pipe: Pipe::WmmaF16,
            macs: Some(mul!(c!(1120 * 4096), chunks!(), p!("n_heads"))),
            flops: Some(mul!(c!(660_000), chunks!(), p!("n_heads"))),
            bytes: Some(add!(
                f32b!(add!(
                    mul!(c!(4), p!("seg_rows"), p!("n_heads"), p!("D")),
                    mul!(p!("seg_rows"), p!("n_heads"))
                )),
                mul!(chunks!(), p!("n_heads"), c!(123392))
            )),
            rows: Some("seg_rows"),
            admit: CHUNK_ADMIT,
            bind: BIND,
            note: "WMMA: 1120 issued 16x16x16 per (chunk, head) = 480 bf16 (pair matrices) + 640 f16 (T' solve/merge/refine, W', Q', U'), counted from the source loops; triangular tiles are charged as issued (they are half zero), so this is the implementation's pipe floor, not the algebraic minimum; tail chunks are charged whole. flops: VALU op count per (chunk, head) = P0 ell prefix ~25k + P1 operand exp/scale/bf16 split ~340k + P2 scalar 16x16 solve 15k + P3 W'/Kend/U'/Q' operand build ~280k, instruction-level estimate (+-30%, [INFERENCE] beyond the 15k solve which is exact: 120 FMA x 16 lanes x 4 waves x 2). bytes: q, k, v, log_decay (per key, upper bound; q/k counted at n_heads, exact when qk_group = 1) + beta read once, plus the 123392 B per-(chunk, head) scratch block written in full (the scratch round trip is counted here as the write, in the scan as the read). rows = launch rows, a Derived constant (RDA_SEGMENT_ROWS 256): the persistent grid carries no rows. n_heads from config (grid is the persistent prep_grid).",
        },
    };
}

// ================================================================================
// gated_delta_vector_chunk_scan_gfx12
// ================================================================================

mod scan {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip(
            "seg_rows",
            "rows of this launch (one scratch segment); chunks = ceil(seg_rows / 64)",
        ),
        ip("n_heads", "value heads (grid x = n_heads * 8 value tiles)"),
        ip("round_out_bf16", "1: round the output to bf16"),
        ip("out_row_stride", "output row stride, floats"),
        ip(
            "state_head_stride",
            "state head stride, floats (multiple of 4)",
        ),
        ip(
            "state_row_stride",
            "state value-row stride, floats (multiple of 4, >= 128)",
        ),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd("scratch", FORMAT_RDA_CHUNK_SCRATCH, Access::ReadOnly, 16,
            mul!(chunks!(), p!("n_heads"), c!(123392)),
            "[chunk][head] blocks written by the prep kernel"),
        opd("state", FORMAT_F32_FLAT, Access::ReadWrite, 16,
            f32b!(add!(
                mul!(sub!(p!("n_heads"), c!(1)), p!("state_head_stride")),
                mul!(c!(127), p!("state_row_stride")),
                c!(128)
            )),
            "value-major [H, V, K] at slot base via head/row strides, read at entry and written at exit; 16-byte tiles"),
        opd("output", FORMAT_F32_FLAT, Access::WriteOnly, 4,
            f32b!(strided!(
                p!("seg_rows"),
                p!("out_row_stride"),
                mul!(p!("n_heads"), p!("D"))
            )),
            "[rows, H, V] at out_row_stride, segment-relative"),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("scratch", 0),
        f_ptr("state", 8),
        f_ptr("output", 16),
        f_i32("seg_rows", 24, "seg_rows"),
        f_i32("n_heads", 28, "n_heads"),
        f_i32("round_out_bf16", 32, "round_out_bf16"),
        f_i32("out_row_stride", 36, "out_row_stride"),
        f_i32("state_head_stride", 40, "state_head_stride"),
        f_i32("state_row_stride", 44, "state_row_stride"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("seg_rows")), "seg_rows > 0"),
        rule(Rule::Positive(p!("n_heads")), "at least one head"),
        rule(
            Rule::OneOf(p!("round_out_bf16"), BOOL),
            "round_out_bf16 is a flag",
        ),
        rule(
            Rule::AtLeast(p!("state_row_stride"), 128),
            "state rows must not overlap (>= head_dim)",
        ),
        rule(
            Rule::MultipleOf(p!("state_row_stride"), 4),
            "16-byte state tiles (launcher check)",
        ),
        rule(
            Rule::MultipleOf(p!("state_head_stride"), 4),
            "16-byte state tiles (launcher check)",
        ),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "gated_delta_vector_chunk_scan_gfx12",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gated_delta_vector_chunk_gfx1201",
            source: "kernels/src/gated_delta_vector_chunk.gfx1201.hip",
        },
        block: [32, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        // SCAN_VTILES = 8 value tiles of 16 rows per head; blockIdx.x = head * 8 + value tile.
        grid: [mul!(p!("n_heads"), c!(8)), c!(1), c!(1)],
        consts: &[("D", 128)],
    }];

    /// `gated_delta_vector_chunk_scan_gfx12`: one wave32 per (head, 16 value rows) walks the chunks keeping the
    /// FP32 state in registers. `.kernarg_segment_size` 48, no hidden arguments, 0 LDS, 247 VGPRs.
    /// Same module (and object `3ad47537f75319f0`) as the prep kernel.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_rda.gated_delta_vector_chunk_scan_gfx12",
        version: 1,
        wave: 32,
        numeric: NUMERIC_GATED_DELTA_CHUNK_SCAN,
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

    const BIND: &[BindV1] = &[bind("n_heads", CFG_HEADS), bind("seg_rows", SEGMENT_ROWS)];

    /// Per (wave = head x 16-value tile, chunk), issued WMMA: `Et = U' - W' S0^T` 4 m x 8 kk x 3 = 96;
    /// `Y`: `M U'` (m + 1 per token tile m: 1 + 2 + 3 + 4 = 10) + `Q' S0^T` 4 x 8 = 32 -> 42; state update
    /// 8 nt x 4 jt x 3 = 96; total 234 f16 WMMA. x 8 waves per head = 1872 per (chunk, head).
    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::RECURRENCE,
            codec: "f32",
            pipe: Pipe::WmmaF16,
            macs: Some(mul!(c!(8 * 234 * 4096), chunks!(), p!("n_heads"))),
            flops: Some(mul!(c!(8 * 12_800), chunks!(), p!("n_heads"))),
            bytes: Some(add!(
                mul!(chunks!(), p!("n_heads"), c!(123392)),
                f32b!(add!(
                    mul!(p!("seg_rows"), p!("n_heads"), p!("D")),
                    mul!(c!(2), p!("n_heads"), p!("D"), p!("D"))
                ))
            )),
            rows: Some("seg_rows"),
            admit: CHUNK_ADMIT,
            bind: BIND,
            note: "WMMA: 234 issued f16 16x16x16 per (wave, chunk) x 8 waves per head, counted from the source loops (E 96, Y 42, state update 96); triangular M tiles charged as issued; tail chunks charged whole. flops: per wave-chunk ~400 VALU ops per lane (state hi/lo f16 split 64 elements x 3, E split 32 x 3, decay multiply 64, output rounding) x 32 lanes = 12.8k, instruction-level estimate [INFERENCE] (~1% of the WMMA work). bytes: the whole scratch (123392 B per (chunk, head)) read once (the prep kernel counts the write), y written, FP32 state read + written once (2 * H * D * D * 4). The scratch was just written by the prep launch and can be cache-resident (a 256-row segment is 19.7 MB), so measured time may beat the HBM floor. grid x = n_heads * 8 exactly; rows = launch rows, a Derived constant (RDA_SEGMENT_ROWS 256).",
        },
    };
}

// ================================================================================
// raon_rda_conv_prepare_w128_f32
// ================================================================================

mod conv_prepare {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("head_dim", "head dim (pinned to 128: lane l holds channels 4l..4l+3)"),
        ip("n_heads", "heads"),
        ip("n_rows", "rows of this launch"),
        ip("rows_per_wave", "rows one wave walks, >= 3 (launcher heuristic)"),
        ip("raw_row_stride", "q/k/v raw row stride, floats (multiple of 4)"),
        ip("f_row_stride", "f row stride, floats (multiple of 4)"),
        ip("bd_row_stride", "b/delta row stride, floats"),
        ip("out_row_stride", "q/k/vprime/log_alpha row stride, floats (multiple of 4)"),
        ip("beta_row_stride", "beta_out row stride, floats"),
        ip("hist_tap_stride", "history tap stride, floats (multiple of 4)"),
        up("q_scale_bits", "f32 bits of q_scale"),
        up("norm_eps_bits", "f32 bits of norm_eps"),
        up("lower_bound_bits", "f32 bits of the gate lower bound"),
        ip("round_bf16", "1: bf16 model-dtype boundaries"),
        ip(
            "waves",
            "waves launched = n_heads * ceil(n_rows / rows_per_wave); grid x = ceil(waves / 8) (not a kernarg)",
        ),
    ];

    macro_rules! plane {
        ($name:literal, $acc:expr, $stride:literal, $note:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                $acc,
                16,
                f32b!(strided!(
                    p!("n_rows"),
                    p!($stride),
                    mul!(p!("n_heads"), p!("head_dim"))
                )),
                $note,
            )
        };
    }
    macro_rules! per_head_plane {
        ($name:literal, $acc:expr, $stride:literal, $note:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                $acc,
                4,
                f32b!(strided!(p!("n_rows"), p!($stride), p!("n_heads"))),
                $note,
            )
        };
    }
    macro_rules! conv_w {
        ($name:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                Access::ReadOnly,
                16,
                f32b!(mul!(p!("n_heads"), p!("head_dim"), c!(4))),
                "[H*128, 4] (channel stride 4, tap stride 1)",
            )
        };
    }
    macro_rules! hist {
        ($name:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                Access::ReadWrite,
                16,
                f32b!(strided!(
                    c!(3),
                    p!("hist_tap_stride"),
                    mul!(p!("n_heads"), p!("head_dim"))
                )),
                "raw conv history [3, H*128] oldest -> newest, read and updated in place",
            )
        };
    }

    const OPERANDS: &[OperandV1] = &[
        plane!("q_out", Access::WriteOnly, "out_row_stride", "[N, H*K]"),
        plane!("k_out", Access::WriteOnly, "out_row_stride", "[N, H*K]"),
        plane!(
            "vprime_out",
            Access::WriteOnly,
            "out_row_stride",
            "[N, H*K]"
        ),
        plane!(
            "log_alpha_out",
            Access::WriteOnly,
            "out_row_stride",
            "[N, H*K]"
        ),
        per_head_plane!("beta_out", Access::WriteOnly, "beta_row_stride", "[N, H]"),
        plane!(
            "q_raw",
            Access::ReadOnly,
            "raw_row_stride",
            "pre-conv projection [N, H*K]"
        ),
        plane!(
            "k_raw",
            Access::ReadOnly,
            "raw_row_stride",
            "pre-conv projection [N, H*K]"
        ),
        plane!(
            "v_raw",
            Access::ReadOnly,
            "raw_row_stride",
            "pre-conv projection [N, H*K]"
        ),
        conv_w!("w_q"),
        conv_w!("w_k"),
        conv_w!("w_v"),
        hist!("hist_q"),
        hist!("hist_k"),
        hist!("hist_v"),
        plane!(
            "f",
            Access::ReadOnly,
            "f_row_stride",
            "gate projection [N, H*K]"
        ),
        per_head_plane!("b", Access::ReadOnly, "bd_row_stride", "[N, H]"),
        per_head_plane!("delta", Access::ReadOnly, "bd_row_stride", "[N, H]"),
        opd(
            "a_log",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(p!("n_heads")),
            "[H]",
        ),
        opd(
            "dt_bias",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(mul!(p!("n_heads"), p!("head_dim"))),
            "[H*K]",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("q_out", 0),
        f_ptr("k_out", 8),
        f_ptr("vprime_out", 16),
        f_ptr("log_alpha_out", 24),
        f_ptr("beta_out", 32),
        f_ptr("q_raw", 40),
        f_ptr("k_raw", 48),
        f_ptr("v_raw", 56),
        f_ptr("w_q", 64),
        f_ptr("w_k", 72),
        f_ptr("w_v", 80),
        f_ptr("hist_q", 88),
        f_ptr("hist_k", 96),
        f_ptr("hist_v", 104),
        f_ptr("f", 112),
        f_ptr("b", 120),
        f_ptr("delta", 128),
        f_ptr("a_log", 136),
        f_ptr("dt_bias", 144),
        f_i32("head_dim", 152, "head_dim"),
        f_i32("n_heads", 156, "n_heads"),
        f_i32("n_rows", 160, "n_rows"),
        f_i32("rows_per_wave", 164, "rows_per_wave"),
        f_i32("raw_row_stride", 168, "raw_row_stride"),
        f_i32("f_row_stride", 172, "f_row_stride"),
        f_i32("bd_row_stride", 176, "bd_row_stride"),
        f_i32("out_row_stride", 180, "out_row_stride"),
        f_i32("beta_row_stride", 184, "beta_row_stride"),
        f_i32("hist_tap_stride", 188, "hist_tap_stride"),
        f_f32("q_scale", 192, "q_scale_bits"),
        f_f32("norm_eps", 196, "norm_eps_bits"),
        f_f32("lower_bound", 200, "lower_bound_bits"),
        f_i32("round_bf16", 204, "round_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Equal(p!("head_dim"), c!(128)),
            "a wave owns one head: lane l holds channels 4l..4l+3 (head_dim 128); conv kernel width is fixed at 4",
        ),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0 (the launcher returns early otherwise)"),
        rule(
            Rule::AtLeast(p!("rows_per_wave"), 3),
            "the history is read and updated in place by the first wave only (launcher check)",
        ),
        rule(Rule::OneOf(p!("round_bf16"), BOOL), "round_bf16 is a flag"),
        rule(Rule::MultipleOf(p!("raw_row_stride"), 4), "16-byte accesses (launcher check)"),
        rule(Rule::MultipleOf(p!("f_row_stride"), 4), "16-byte accesses (launcher check)"),
        rule(Rule::MultipleOf(p!("out_row_stride"), 4), "16-byte accesses (launcher check)"),
        rule(Rule::MultipleOf(p!("hist_tap_stride"), 4), "16-byte accesses (launcher check)"),
    ];

    const EXPORTS: &[ExportV1] = &[
        ExportV1 {
            symbol: "raon_rda_conv_prepare_w128_f32",
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "raon_rda_conv_prepare",
                source: "kernels/src/raon_rda_conv_prepare.hip",
            },
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            // WAVES_PER_BLOCK = 8 (one (head, row tile) per wave32).
            grid: [Expr::DivCeil(&p!("waves"), 8), c!(1), c!(1)],
            consts: &[],
        },
        // gfx1100 (7900 XTX): same layout from a local hipcc build (`--offload-arch=gfx1100`: kernarg 208,
        // 0 LDS, 175 VGPRs, max flat workgroup 256, wave32).
        ExportV1 {
            symbol: "raon_rda_conv_prepare_w128_f32",
            arch: Arch::Gfx1100,
            object: ObjectRefV1::HipccSource {
                module: "raon_rda_conv_prepare",
                source: "kernels/src/raon_rda_conv_prepare.hip",
            },
            block: [256, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [Expr::DivCeil(&p!("waves"), 8), c!(1), c!(1)],
            consts: &[],
        },
    ];

    /// `raon_rda_conv_prepare_w128_f32`: the three depthwise width-4 SiLU convs (q, k, v) fused with
    /// `raon_rda_prepare_f32` in one launch; one wave32 per (head, `rows_per_wave` rows), 8 waves per block.
    /// `.kernarg_segment_size` 208, no hidden arguments, 0 LDS, 175 VGPRs. Compiled object:
    /// `raon_rda_conv_prepare.d3c6fd13e1de2ba1`.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_rda.raon_rda_conv_prepare_w128_f32",
        version: 1,
        wave: 32,
        numeric: NUMERIC_RAON_RDA_CONV_PREPARE,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 208,
        segment_bytes: 208,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    /// Order matters: `waves` is checked against the grid interval after n_heads / n_rows / rows_per_wave.
    /// `rows_per_wave` 16 is `rda_conv_prepare_rows_per_wave(n_rows)` for `n_rows >= 256` (a launcher heuristic: 8
    /// below 256 rows; it only sets the wave count).
    const BIND: &[BindV1] = &[
        bind("n_heads", CFG_HEADS),
        bind("head_dim", CFG_RDA_HEAD_DIM),
        bind("n_rows", SEGMENT_ROWS),
        bind("rows_per_wave", BindSource::Derived(Expr::Const(16))),
        bind(
            "waves",
            BindSource::Derived(Expr::Mul(&[
                Expr::Param("n_heads"),
                Expr::DivCeil(&Expr::Param("n_rows"), 16),
            ])),
        ),
    ];

    /// Elementwise arithmetic per channel-element (row, channel), counted from the kernel source with one op per
    /// instruction class and transcendentals as one op: 3 conv tap chains x (4 mul + 4 add) = 24; 3 SiLU x
    /// (exp, add, div) = 9; q/k sum of squares 2 x (mul + reduction add) = 4; q/k normalise + `q_scale` 3; gate
    /// `exp(a_log) * (f + dt)`, sigmoid (exp, add, rcp), `lower_bound *` = 6; `vprime = conv_v * ratio` 1:
    /// 47, plus the per-(row, head) scalars (two logsigmoid, exp, sigmoid ~ 20 ops / 128) ~ 0.2 -> 48.
    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::CONV,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: None,
            flops: Some(mul!(c!(48), p!("n_rows"), p!("n_heads"), p!("head_dim"))),
            bytes: Some(f32b!(add!(
                mul!(c!(8), p!("n_rows"), p!("n_heads"), p!("head_dim")),
                mul!(c!(3), p!("n_rows"), p!("n_heads")),
                mul!(c!(31), p!("n_heads"), p!("head_dim")),
                p!("n_heads")
            ))),
            rows: Some("n_rows"),
            admit: WAVE_ADMIT,
            bind: BIND,
            note: "op class conv: the three conv chains + SiLU are 33 of the ~48 ops per element (the rest is the prepare epilogue) and the kernel is memory bound anyway. flops = 48 * rows * H * D, instruction-class count per channel-element (see the derivation above), integer bf16 rounding and shuffle traffic not charged, approximate [INFERENCE: +-15%]. bytes: 4 planes read (q_raw, k_raw, v_raw, f) + 4 written (q, k, vprime, log_alpha), b/delta read + beta written (3 per row-head), conv weights 12 and history 9 read + 9 written per channel (31 per channel with dt_bias), a_log per head; halo re-reads of extra waves are not minimal and not counted. rows = launch rows, a Derived constant (RDA_SEGMENT_ROWS 256); rows_per_wave = 16 (launcher heuristic for >= 256 rows); `waves` = n_heads * ceil(n_rows / 16) = 640 is cross-checked against the observed grid (80 workgroups x 8 waves).",
        },
    };
}

// ================================================================================
// gated_rmsnorm_sigmoid_rows_w128_f32
// ================================================================================

mod gated_rms_w128 {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip(
            "head_dim",
            "head dim (pinned to 128: lane l holds channels 4l..4l+3)",
        ),
        ip("n_heads", "heads"),
        ip("n_rows", "rows of this launch"),
        ip("x_row_stride", "x row stride, floats (multiple of 4)"),
        ip("gate_row_stride", "gate row stride, floats (multiple of 4)"),
        ip("out_row_stride", "out row stride, floats (multiple of 4)"),
        up("eps_bits", "f32 bits of eps"),
        ip("round_bf16", "1: round the output to bf16"),
        ip(
            "waves",
            "waves launched = n_rows * n_heads; grid x = ceil(waves / 8) (not a kernarg)",
        ),
    ];

    macro_rules! plane {
        ($name:literal, $acc:expr, $stride:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                $acc,
                16,
                f32b!(strided!(
                    p!("n_rows"),
                    p!($stride),
                    mul!(p!("n_heads"), p!("head_dim"))
                )),
                "[N, H*D]",
            )
        };
    }

    const OPERANDS: &[OperandV1] = &[
        plane!("out", Access::WriteOnly, "out_row_stride"),
        plane!("x", Access::ReadOnly, "x_row_stride"),
        plane!("gate", Access::ReadOnly, "gate_row_stride"),
        opd(
            "gamma",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(p!("head_dim")),
            "[D] plain gamma",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("out", 0),
        f_ptr("x", 8),
        f_ptr("gate", 16),
        f_ptr("gamma", 24),
        f_i32("head_dim", 32, "head_dim"),
        f_i32("n_heads", 36, "n_heads"),
        f_i32("n_rows", 40, "n_rows"),
        f_i32("x_row_stride", 44, "x_row_stride"),
        f_i32("gate_row_stride", 48, "gate_row_stride"),
        f_i32("out_row_stride", 52, "out_row_stride"),
        f_f32("eps", 56, "eps_bits"),
        f_i32("round_bf16", 60, "round_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Equal(p!("head_dim"), c!(128)),
            "a wave owns one (row, head): lane l holds channels 4l..4l+3 (head_dim 128)",
        ),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(
            Rule::Positive(p!("n_rows")),
            "n_rows > 0 (the launcher returns early otherwise)",
        ),
        rule(Rule::OneOf(p!("round_bf16"), BOOL), "round_bf16 is a flag"),
        rule(
            Rule::MultipleOf(p!("x_row_stride"), 4),
            "16-byte accesses (launcher check)",
        ),
        rule(
            Rule::MultipleOf(p!("gate_row_stride"), 4),
            "16-byte accesses (launcher check)",
        ),
        rule(
            Rule::MultipleOf(p!("out_row_stride"), 4),
            "16-byte accesses (launcher check)",
        ),
        rule(
            Rule::Equal(p!("waves"), mul!(p!("n_rows"), p!("n_heads"))),
            "one wave32 per (row, head)",
        ),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "gated_rmsnorm_sigmoid_rows_w128_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gated_rmsnorm_sigmoid_rows_w128",
            source: "kernels/src/gated_rmsnorm_sigmoid_rows_w128.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        // WAVES_PER_BLOCK = 8.
        grid: [Expr::DivCeil(&p!("waves"), 8), c!(1), c!(1)],
        consts: &[],
    }];

    /// `gated_rmsnorm_sigmoid_rows_w128_f32`: per-(row, head) RMSNorm times sigmoid(gate), one wave32 per
    /// (row, head), no LDS, no barrier. `.kernarg_segment_size` 64, no hidden arguments. Compiled object:
    /// `gated_rmsnorm_sigmoid_rows_w128.5dcbfc98afd340a9`.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_rda.gated_rmsnorm_sigmoid_rows_w128_f32",
        version: 1,
        wave: 32,
        numeric: NUMERIC_GATED_RMSNORM_SIGMOID_ROWS_W128,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 64,
        segment_bytes: 64,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_heads", CFG_HEADS),
        bind("head_dim", CFG_RDA_HEAD_DIM),
        bind("n_rows", SEGMENT_ROWS),
        bind(
            "waves",
            BindSource::Derived(Expr::Mul(&[Expr::Param("n_rows"), Expr::Param("n_heads")])),
        ),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::NORM,
            codec: "f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(f32b!(add!(
                mul!(c!(3), p!("n_rows"), p!("n_heads"), p!("head_dim")),
                p!("head_dim")
            ))),
            rows: Some("n_rows"),
            admit: WAVE_ADMIT,
            bind: BIND,
            note: "mirrors generic_seq.gated_rmsnorm_sigmoid_rows_f32: x and gate read, out written (3 planes) + gamma; arithmetic not charged. The Raon launch is a 256-row segment whose planes the previous kernel just produced, so the segment is last-level-cache resident and the measured time can beat the HBM floor. rows = launch rows, a Derived constant (RDA_SEGMENT_ROWS 256); `waves` = n_rows * n_heads is cross-checked against the observed grid (1280 workgroups x 8 waves).",
        },
    };
}

// ================================================================================

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &prep::ABI,
    &scan::ABI,
    &conv_prepare::ABI,
    &gated_rms_w128::ABI,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &prep::WORK,
    &scan::WORK,
    &conv_prepare::WORK,
    &gated_rms_w128::WORK,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval;

    fn params_of(abi: &KernelAbiV1, given: &[(&str, u64)]) -> Vec<u64> {
        abi.params
            .iter()
            .map(|p| {
                given
                    .iter()
                    .find(|(n, _)| *n == p.name)
                    .map_or(1, |(_, v)| *v)
            })
            .collect()
    }

    /// Workgroup grid of the single export of `abi` for `given` params.
    fn grid_of(abi: &KernelAbiV1, given: &[(&str, u64)]) -> [u64; 3] {
        let e = &abi.exports[0];
        let ps = params_of(abi, given);
        let g: Vec<u64> = e
            .grid
            .iter()
            .map(|x| eval(x, abi, e, &ps).expect("grid evaluates"))
            .collect();
        [g[0], g[1], g[2]]
    }

    /// A work expression over `given` params (export constants resolve by name).
    fn work_value(spec: &WorkSpecV1, e: &Expr, given: &[(&str, u64)]) -> u64 {
        let abi = spec.abi;
        let ps = params_of(abi, given);
        eval(e, abi, &abi.exports[0], &ps).expect("work expression evaluates")
    }

    #[test]
    fn every_contract_self_checks() {
        for abi in ABIS {
            if let Err(e) = abi.self_check() {
                panic!("{}: {e:#?}", abi.id);
            }
        }
    }

    #[test]
    fn every_work_spec_self_checks_and_pairs_with_its_contract() {
        assert_eq!(ABIS.len(), WORK.len());
        for (abi, w) in ABIS.iter().zip(WORK) {
            assert!(std::ptr::eq(*abi, w.abi), "{}", abi.id);
            if let Err(e) = w.self_check() {
                panic!("{e}");
            }
        }
    }

    #[test]
    fn every_export_symbol_resolves_through_the_registry() {
        for abi in ABIS {
            for e in abi.exports {
                let (found, _) = crate::families::find_export(e.arch, e.symbol).unwrap();
                assert_eq!(found.id, abi.id);
            }
        }
    }

    #[test]
    fn kernarg_layouts_match_the_compiled_objects() {
        // Measured with llvm-readelf --notes on the objects named in the module docs.
        assert_eq!(
            (
                prep::ABI.explicit_bytes,
                prep::ABI.segment_bytes,
                prep::ABI.exports[0].group_segment_fixed
            ),
            (96, 352, 58624)
        );
        assert_eq!(prep::ABI.hidden.first().map(|h| h.offset), Some(96));
        assert_eq!(
            (scan::ABI.explicit_bytes, scan::ABI.segment_bytes),
            (48, 48)
        );
        assert!(scan::ABI.hidden.is_empty());
        assert_eq!(
            (
                conv_prepare::ABI.explicit_bytes,
                conv_prepare::ABI.segment_bytes
            ),
            (208, 208)
        );
        assert_eq!(
            (
                gated_rms_w128::ABI.explicit_bytes,
                gated_rms_w128::ABI.segment_bytes
            ),
            (64, 64)
        );
        // 19 pointers = 5 outputs + 3 raw + 3 weights + 3 histories + f, b, delta, a_log, dt_bias.
        assert_eq!(
            conv_prepare::ABI
                .fields
                .iter()
                .filter(|f| matches!(f.kind, FieldKind::Pointer { .. }))
                .count(),
            19
        );
    }

    #[test]
    fn raon_chunk_prep_grid_is_the_persistent_grid() {
        // Observed (64, 1, 1) workgroups, block 256, rocprof work-items 16384; LDS 58880 in the older traced
        // build, 58624 in the current source.
        let g = grid_of(&prep::ABI, &[("prep_grid", 64)]);
        assert_eq!(g, [64, 1, 1]);
        assert_eq!(g[0] * 256, 16384);
        assert_eq!(prep::ABI.exports[0].block, [256, 1, 1]);
        // 2 * 32 multiprocessors = 64 < 4 chunks * 40 heads = 160 pairs at 256 rows.
        assert_eq!((2u64 * 32).min(4 * 40), 64);
    }

    #[test]
    fn raon_chunk_scan_grid_inverts_to_40_heads() {
        // Observed (320, 1, 1) workgroups, block 32: 40 heads x 8 value tiles.
        let g = grid_of(&scan::ABI, &[("n_heads", 40)]);
        assert_eq!(g, [320, 1, 1]);
        assert_eq!(320 / 8, 40);
        assert_eq!(scan::ABI.exports[0].block, [32, 1, 1]);
    }

    #[test]
    fn raon_conv_prepare_grid_inverts_to_640_waves_of_256_rows() {
        // Observed (80, 1, 1) workgroups, block 256: 640 waves = 40 heads x ceil(256 / 16) row tiles.
        let g = grid_of(&conv_prepare::ABI, &[("waves", 640)]);
        assert_eq!(g, [80, 1, 1]);
        // The grid admits waves in [(80 - 1) * 8 + 1, 80 * 8].
        assert!((79 * 8 + 1..=640).contains(&(40 * 256u64.div_ceil(16))));
        // 512 rows at 16 rows per wave would be 1280 waves = 160 workgroups: not the observed launch.
        assert_eq!(grid_of(&conv_prepare::ABI, &[("waves", 40 * 32)])[0], 160);
        // The `waves` bind (n_heads * ceil(n_rows / 16)) lies inside the grid interval of the observed launch.
        let derived = conv_prepare::WORK
            .work
            .bind
            .iter()
            .find(|b| b.param == "waves")
            .expect("waves bind");
        let BindSource::Derived(expr) = derived.source else {
            panic!("waves must be derived")
        };
        let abi = &conv_prepare::ABI;
        let ps = params_of(abi, &[("n_heads", 40), ("n_rows", 256)]);
        let waves = eval(&expr, abi, &abi.exports[0], &ps).unwrap();
        assert_eq!(waves, 640);
        assert!((79 * 8 + 1..=80 * 8).contains(&waves));
    }

    #[test]
    fn raon_gated_rms_w128_grid_inverts_to_256_rows_x_40_heads() {
        // Observed (1280, 1, 1) workgroups, block 256: 10240 waves = 256 rows x 40 heads.
        let g = grid_of(&gated_rms_w128::ABI, &[("waves", 256 * 40)]);
        assert_eq!(g, [1280, 1, 1]);
        assert_eq!(gated_rms_w128::ABI.exports[0].block, [256, 1, 1]);
    }

    #[test]
    fn raon_work_counts_at_256_rows_40_heads() {
        let given = [("seg_rows", 256), ("n_heads", 40)];
        // prep: 4 chunks x 40 heads x 1120 WMMA x 4096 MAC.
        let w = &prep::WORK;
        assert_eq!(
            work_value(w, &w.work.macs.unwrap(), &given),
            4 * 40 * 1120 * 4096
        );
        // scan: 4 chunks x 40 heads x 8 waves x 234 WMMA x 4096 MAC.
        let w = &scan::WORK;
        assert_eq!(
            work_value(w, &w.work.macs.unwrap(), &given),
            4 * 40 * 8 * 234 * 4096
        );
        // scan scratch read = 160 blocks x 123392 B = 19.7 MB, plus y and the state round trip.
        assert_eq!(
            work_value(w, &w.work.bytes.unwrap(), &given),
            160 * 123392 + 4 * (256 * 40 * 128 + 2 * 40 * 128 * 128)
        );
        // conv_prepare: flops 48 per channel-element, 8 planes + side data of bytes.
        let w = &conv_prepare::WORK;
        let given = [("n_rows", 256), ("n_heads", 40), ("head_dim", 128)];
        assert_eq!(
            work_value(w, &w.work.flops.unwrap(), &given),
            48 * 256 * 5120
        );
        assert_eq!(
            work_value(w, &w.work.bytes.unwrap(), &given),
            4 * (8 * 256 * 5120 + 3 * 256 * 40 + 31 * 5120 + 40)
        );
        // gated rmsnorm: 3 planes + gamma.
        let w = &gated_rms_w128::WORK;
        assert_eq!(
            work_value(w, &w.work.bytes.unwrap(), &given),
            4 * (3 * 256 * 5120 + 128)
        );
    }

    #[test]
    fn rows_are_launch_rows_and_admission_matches_the_selectors() {
        // The launch rows bind as a constant (RDA_SEGMENT_ROWS), never as the 4096-row pass chunk.
        for w in WORK {
            assert!(
                w.work
                    .bind
                    .iter()
                    .all(|b| !matches!(b.source, BindSource::ChunkRows | BindSource::PassTokens)),
                "{}",
                w.abi.id
            );
        }
        for w in [&prep::WORK, &scan::WORK] {
            assert_eq!(w.work.admit.min_rows, 64);
            assert_eq!(
                w.work.admit.predicate,
                "crates/rdna-compute/src/gated_delta_vector_chunk.rs::GATED_DELTA_VECTOR_CHUNK_MIN_ROWS"
            );
            assert_eq!(
                w.work.admit.kill_env,
                Some("HIPFIRE_GATED_DELTA_VECTOR_CHUNKED")
            );
            assert_eq!(w.work.pipe, Pipe::WmmaF16);
            assert_eq!(w.work.op_class, op_class::RECURRENCE);
        }
        for w in [&conv_prepare::WORK, &gated_rms_w128::WORK] {
            assert_eq!(w.work.admit.min_rows, 64);
            assert_eq!(
                w.work.admit.predicate,
                "crates/hipfire-dispatch/src/pipeline/rda.rs::RDA_WAVE_MIN_ROWS"
            );
            assert_eq!(w.work.admit.kill_env, Some("HIPFIRE_RAON_RDA_FUSED_PREP"));
        }
        assert_eq!(conv_prepare::WORK.work.op_class, op_class::CONV);
        assert_eq!(conv_prepare::WORK.work.pipe, Pipe::ValuF32);
        assert_eq!(gated_rms_w128::WORK.work.op_class, op_class::NORM);
    }

    #[test]
    fn raon_work_params_bind_from_the_raon_config_keys() {
        let keys = |w: &WorkSpecV1, p: &str| -> Vec<&'static str> {
            match w.work.bind.iter().find(|b| b.param == p).unwrap().source {
                BindSource::Config(k) => k.to_vec(),
                _ => panic!("unexpected source"),
            }
        };
        for w in [
            &prep::WORK,
            &scan::WORK,
            &conv_prepare::WORK,
            &gated_rms_w128::WORK,
        ] {
            assert!(keys(w, "n_heads").contains(&"rda_config.num_heads"));
        }
        assert!(keys(&conv_prepare::WORK, "head_dim").contains(&"rda_config.head_dim"));
        // Raon: 40 heads x 128, 18 RDA layers x 16 segments of 256 rows = the observed 288 calls per pass.
        assert_eq!(18 * (4096 / 256), 288);
    }
}
