// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic-op fused/producer launchers (hipcc) of the Raon pp4096 debt list: the BF16-stream postmix and
//! route+aggregate of the mHC stream mixer, the fused SiTU-GLU + FWHT256 + `block_i8_128` down producers
//! (grouped-MoE and dense-rows entries), and the Q8 expanded-mirror scatter of the latent attention.
//!
//! Evidence. Kernarg layouts, static LDS and segment sizes are read from the JIT-cached gfx1201 objects in
//! `~/.hipfire_kernels/gfx1201` (clang offload bundle unbundled with `clang-offload-bundler`, notes read with
//! `llvm-readelf --notes`, argument table from the `.radiowave.json` sidecar). Where a module has several
//! hashes, the hash whose `.hip` equals the current source was used:
//!
//! * `stream_mix_postmix.94c9c85e7486bbdd` (`stream_mix_postmix_bf16`: 8 explicit args, `.kernarg_segment_size`
//!   52, LDS 0, no hidden words);
//! * `stream_mix_route_aggregate.0cba07c7cff3679d` (`stream_mix_route_aggregate_n4_k10_bf16`: 15 explicit args,
//!   segment 92, `.group_segment_fixed_size` 1584, no hidden words);
//! * `ffn_grouped_activation_rotate_i8_gfx1201.e26fdc48c9f67f9a` (byte-identical to the module text the runtime
//!   builds: the two `#define`s, `block_i8_128_quant.hip`, `block_i8_128_emit.hip`, `mq_fwht256.h`,
//!   `ffn_grouped_activation_pad.hip`, then the producer source, in this order, see
//!   `rdna_compute::kernels::FFN_GROUPED_ACTIVATION_ROTATE_I8_SRC`): grouped entry segment 104, rows entry
//!   segment 92, LDS 0, `max_flat_workgroup_size` 32, no hidden words;
//! * `latent_attention_decompress.3c44cbae4c12bf08` (`latent_attention_scatter_expanded_q8`: segment 56,
//!   LDS 0, `max_flat_workgroup_size` 64, no hidden words).
//!
//! Scope and conventions of this file:
//!
//! * `Arch::Gfx1201` is exported for every contract here; `ffn_situ_grouped_rotate_i8_gfx1201` is also exported on
//!   `Arch::Gfx1100` (layout from a local `hipcc --offload-arch=gfx1100` build: kernarg 104, 0 LDS). Every other
//!   launcher is exact-arch gfx1201 or its trace is gfx1201-only; no gfx1100 object of those symbols exists.
//! * f32 kernel scalars (`post_factor`, `eps`, `p0`, ...) are `U32` params named `*_bits` carrying the IEEE-754
//!   bit pattern, because `ScalarTy` has no float.
//! * The `ffn_swiglu_*` / `ffn_swiglu_clamp_*` siblings of the rotate-i8 module have the same kernarg layout
//!   but a different numeric contract (a [`KernelAbiV1`] has exactly one), and the f32-stream
//!   `stream_mix_route_aggregate_n4_k10` has an F32 `x` plane (another operand format); they are not
//!   covered. Raon launches none of them.
//! * Work specs charge flops to the VALU pipe at nominal one FLOP per add/mul/min/max/abs/rint/div and the
//!   same 64-FLOP estimate per SiTU element as `generic_elem`'s `ffn_situ_*_f32`; those counts are estimates,
//!   not instruction counts, and each `note` says what is and is not charged.

use crate::work::{
    op_class, BindSource, BindV1, Pipe, RowDomainV1, Stacking, WeightAxis, WorkSpecV1, WorkV1,
    ALL_ROWS,
};
use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, KernelAbiV1,
    NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::generic_elem::{FORMAT_MQ_FWHT_SIGNS_256, FORMAT_U64_AWQ_PTR_TABLE};
use super::generic_seq::{
    FORMAT_BF16_FLAT, FORMAT_GAMMA_F32_OR_BF16, FORMAT_I32_DEVICE_SCALAR,
    FORMAT_I8_MIRROR_PLANE_VIEW,
};
use super::ggml_convert::FORMAT_F32_FLAT;
use super::mq6_i8::FORMAT_BLOCK_I8_128_KB_N;
use super::qwen4_moe_sym::{FORMAT_EXPERT_TILE_IDS, FORMAT_SORTED_SLOT_INDEX};

// ---- numeric contracts -----------------------------------------------------------

/// `stream_mix_postmix_bf16`. Per output element `(i, c)` of a row: `a = 0.0f`; then for `j = 0..n` ascending
/// `a += res[i, j] * x[j, c]` with `x` the BF16 bits widened exactly (`bits << 16`); then
/// `a += post[i] * branch[c]`; then `a` is rounded to bfloat16 round-to-nearest-even (a NaN keeps its payload
/// and gets the quiet bit) and stored as 16 bits. The multiply-adds are contracted to `v_fma_f32` by the
/// compiler (gfx1201 disassembly of the cached object shows 16 `v_fma_f32` and no `v_mul_f32`/`v_add_f32` in the
/// kernel). The result is the bit-for-bit BF16 image of `stream_mix_postmix` on the widened inputs. The
/// `bf16_unused` argument is ignored.
pub const NUMERIC_STREAM_MIX_POSTMIX_BF16: NumericContractId =
    NumericContractId("stream_mix.postmix.bf16_stream");

/// `stream_mix_route_aggregate_n4_k10_bf16`. Streams are BF16 bits widened exactly. Thread `tid` of the
/// 256-thread row workgroup loads `x[d]` for `d = tid + 256 j`, `j = 0..N*KPER` ascending (stream-major), keeps
/// them in registers and forms `acc = sum_j x_j^2` sequentially per thread; the 256 partial sums are reduced by
/// the LDS halving tree `red[t] += red[t + off]`, `off = 128, 64, ..., 1`. `inv = rsqrtf(red[0] / (N*WIDTH) +
/// selector_eps)`. For `tid < zdim = 2N + (N-1)^2`: `zv = z_raw[row, tid]`, bf16-rounded when `bf16 != 0`, then
/// `zv *= inv`; `pre[s] = sigmoid(alpha_pre * zv + bias[s])`, `post[s] = post_factor * sigmoid(alpha_post * zv +
/// bias[N + s])`, `sigmoid(v) = 1.0f / (1.0f + expf(-v))`; the remaining lanes form the Helmert-chart `res`
/// exactly as `stream_mix_route_helmert` (runtime stream count `n_rt`, `E_ij = sum_k (sum_m H[i,m] Z[m,k])
/// H[j,k]`, `res = (1 - g) I + g phi`, `g = sigmoid(eta)`). The aggregate `a[c] = sum_s pre[s] * x[s, c]`
/// accumulates `s` ascending from `0.0f`, rounded to bf16 when `bf16 != 0`. With `has_norm`:
/// `r = rsqrtf(tree(sum_k a^2) / WIDTH + eps)` (same per-thread-then-halving-tree order), `out = a * r * gamma[c]`
/// (left to right), rounded to bf16 when `bf16 != 0`; `out` is F32 carrying those values. Reductions and
/// expressions are copied from `stream_mix_route_helmert.hip` / `stream_mix_aggregate.hip`, so `pre`, `post`,
/// `res` and `out` equal those two kernels run in sequence (kernel source header).
pub const NUMERIC_STREAM_MIX_ROUTE_AGGREGATE_BF16: NumericContractId =
    NumericContractId("stream_mix.route_aggregate.bf16_stream");

/// `ffn_situ_{grouped,rows}_rotate_i8_gfx1201`. One wave owns one (row, 256-column group); lane `l` owns columns
/// `8l..8l+7`. Activation per column (`ffn_act<2>`, SiTU-GLU): when `round_inputs`, `g` and `u` are bf16-RNE
/// rounded (`ffn_bf16_rne`); `sig = 1.0f / (1.0f + expf(-g))`, `gate = (p0 * tanhf(g / p0)) * sig`,
/// `value = p1 * tanhf(u / p1)`, `r = gate * value`; when `round_output`, `r` is bf16-RNE rounded. A column
/// at or beyond `logical_n` is `+0.0f`. An optional AWQ divide `r = r / scale[col]` (IEEE division, after the
/// cast and the pad; an expert outside `[0, n_experts)` or a null table entry makes the whole grouped row
/// `+0.0f` with no divide). Rotation (`mq_rotate_x`): `v = r * signs1[col]`; signed FWHT-256 as eight butterfly
/// stages (in-register strides 1, 2, 4 as `(a + b, a - b)`, then lane strides 1..16 through `ds_swizzle` with
/// `x = peer - x` on the high side of the pair, `x + peer` on the low side); `f = v * 0.0625f * signs2[col]`
/// (multiplied left to right). Quantization (`block_i8_128`, 128 columns = 16 lanes): `amax = max |f|` (exact),
/// `d = amax == 0 ? 0 : __fdiv_rn(amax, 127.0f)`, `q = amax == 0 ? 0 : clamp(rintf(__fdiv_rn(f, d)), -127, 127)`,
/// `s = sum q` (integer); block bytes `{ f32 d, i32 s, int8 qs[128] }` (136 B). The stored blocks are
/// byte-identical to the chain `ffn_*_f32 -> mq_rotate_x_batched -> quantize_int8_mmq_ds128` (kernel source
/// header; byte oracle `tests/ffn_grouped_i8_producer_oracle.rs`); the optional `x_out` F32 plane carries `f`.
pub const NUMERIC_FFN_SITU_ROTATE_I8: NumericContractId =
    NumericContractId("fused.ffn_situ_rotate_i8.f32_expf_tanhf_fwht256_block_i8_128");

/// `latent_attention_scatter_expanded_q8`. Per (row, head, plane) vector `x[0..width)` of the F32 projector
/// output (no intermediate BF16 rounding): `m = max |x[e]|` (exact, order-independent), `sc = min(m / 127.0f,
/// 65504.0f)` (IEEE division), `s16 = (_Float16) sc` (round-to-nearest-even), `sf = (float) s16`;
/// `q[e] = sf > 0 ? clamp(rintf(x[e] / sf), -127, 127) : 0` (IEEE division, round-half-even); codes are stored
/// as int8 at byte `h * width + e` of the plane row and `s16` as F16 at byte `n_heads * width + 2 h`.
pub const NUMERIC_LATENT_SCATTER_EXPANDED_Q8: NumericContractId =
    NumericContractId("latent_attention.scatter_expanded.q8_f16_scale");

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
/// `(rows - 1) * stride + width`: elements reached by `rows` rows of `width` at `stride`.
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
/// `2 * elems`.
macro_rules! bf16b {
    ($e:expr) => {
        mul!($e, c!(2))
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
const I32_MAX: u64 = i32::MAX as u64;
const U32_MAX: u64 = u32::MAX as u64;
/// `MOE_GROUPED_BLOCK_M` (`crates/hipfire-dispatch/src/families/moe.rs`): rows per expert tile.
const BLOCK_M: u64 = 16;

/// Config synonyms shared by the work specs.
const CFG_STREAMS: BindSource = BindSource::Config(&[
    "raon_mhc_streams",
    "mhc_streams",
    "num_residual_streams",
    "n_streams",
]);
const CFG_HIDDEN: BindSource = BindSource::Config(&["hidden_size", "n_embd", "d_model", "dim"]);
const CFG_NOPE: BindSource = BindSource::Config(&["qk_nope_head_dim"]);
const CFG_V: BindSource = BindSource::Config(&["v_head_dim"]);
/// Experts per token (top-k).
const KEYS_TOPK: &[&str] = &[
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_topk",
    "top_k",
];
/// Per-expert FFN width (the logical `n` of the gate/up activation).
const KEYS_MOE_INTERMEDIATE: &[&str] = &[
    "moe_intermediate_size",
    "moe_intermediate_dim",
    "intermediate_size",
];
/// Weight codecs whose activation input is the FWHT-rotated A8 sidecar (same codec the A8 symmetric grouped
/// GEMM contracts of `generic_gemm` bind).
const MQ_ROTATED_CODECS: &[&str] = &["mq4g256v2"];

// ================================================================================
// stream_mix_postmix_bf16
// ================================================================================

mod postmix_bf16 {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams, 1..=8 (SM_MAX_N)"),
        ip(
            "width",
            "row width, even (each thread owns two columns); grid x = ceil(width / 512)",
        ),
        ip(
            "bf16_unused",
            "kernel argument `bf16_unused`: ignored (the BF16 variant always rounds); the launcher passes the rounding flag, 1",
        ),
        ip("rows", "rows (grid y, not a kernarg)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_BF16_FLAT,
            Access::ReadOnly,
            4,
            bf16b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width] old streams, BF16 bits; read as 32-bit words of two adjacent columns",
        ),
        opd(
            "branch",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            8,
            f32b!(mul!(p!("rows"), p!("width"))),
            "[rows, width] F32; read as float2",
        ),
        opd(
            "post",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"))),
            "[rows, n]",
        ),
        opd(
            "res",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("n"))),
            "[rows, n, n] destination/source",
        ),
        opd(
            "out",
            FORMAT_BF16_FLAT,
            Access::WriteOnly,
            4,
            bf16b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width] BF16 bits; must not alias x (ping-pong)",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("branch", 8),
        f_ptr("post", 16),
        f_ptr("res", 24),
        f_ptr("out", 32),
        f_i32("n", 40, "n"),
        f_i32("width", 44, "width"),
        f_i32("bf16_unused", 48, "bf16_unused"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::AtLeast(p!("n"), 1), "n >= 1"),
        rule(Rule::AtMost(p!("n"), 8), "SM_MAX_N = 8 register window"),
        rule(Rule::Positive(p!("width")), "width > 0"),
        rule(
            Rule::MultipleOf(p!("width"), 2),
            "each thread owns two adjacent columns (32-bit access per stream)",
        ),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(
            Rule::AtMost(mul!(p!("rows"), p!("n"), p!("width")), I32_MAX),
            "int index",
        ),
        rule(
            Rule::OneOf(p!("bf16_unused"), BOOL),
            "the launcher passes the rounding flag (BF16 streams need Bf16 rounding)",
        ),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_postmix_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_postmix",
            source: "kernels/src/stream_mix_mix.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&p!("width"), 512), p!("rows"), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_postmix_bf16`: `out[i,c] = bf16(sum_j res[i,j] x[j,c] + post[i] branch[c])` over BF16 stream
    /// planes, two columns per thread. `.kernarg_segment_size` 52, no hidden arguments, 0 LDS.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_fused.stream_mix_postmix_bf16",
        version: 1,
        wave: 32,
        numeric: NUMERIC_STREAM_MIX_POSTMIX_BF16,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 52,
        segment_bytes: 52,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("n", CFG_STREAMS), bind("width", CFG_HIDDEN)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "bf16",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(p!("rows"), p!("n"), add!(p!("n"), c!(1)), p!("width"))),
            flops: None,
            bytes: Some(f32b!(mul!(
                p!("rows"),
                add!(
                    mul!(p!("n"), p!("width")),
                    p!("width"),
                    p!("n"),
                    mul!(p!("n"), p!("n"))
                )
            ))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "macs = rows*n*(n+1)*width: per output element n FMAs of res*x plus one post*branch FMA (kernel source; contracted to v_fma_f32), the bf16 round (~4 integer ops) not charged; bytes = BF16 x read + BF16 out write (2*2*n*width B) + F32 branch (4*width) + post (4*n) + res (4*n*n) per row; rows exact from grid y; width = hidden size, grid x = ceil(width/512) is an interval of 512; n from the stream-count config",
        },
    };
}

// ================================================================================
// stream_mix_route_aggregate_n4_k10_bf16
// ================================================================================

mod route_aggregate_bf16 {
    use super::*;

    /// `zdim = 2n + (n-1)^2`.
    const ZDIM: Expr = add!(
        mul!(c!(2), p!("n")),
        mul!(sub!(p!("n"), c!(1)), sub!(p!("n"), c!(1)))
    );

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams (pinned to 4 by the instantiation)"),
        ip(
            "width",
            "row width (pinned to 2560 = KPER 10 x 256 by the instantiation)",
        ),
        up("post_factor_bits", "f32 bits of post_factor"),
        up("selector_eps_bits", "f32 bits of selector_eps"),
        up("eps_bits", "f32 bits of the norm eps"),
        ip(
            "bf16",
            "1: round z, the aggregate and the norm output to bf16",
        ),
        ip("has_norm", "1: plain-gamma RMSNorm after the aggregate"),
        ip("gamma_kind", "0: gamma F32, 1: gamma BF16 bits"),
        ip("rows", "rows (grid x)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_BF16_FLAT,
            Access::ReadOnly,
            2,
            bf16b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n*width] old streams, BF16 bits",
        ),
        opd(
            "z_raw",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), ZDIM)),
            "[rows, zdim] raw selector projection",
        ),
        opd(
            "params",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(add!(c!(4), ZDIM)),
            "[alpha_pre, alpha_post, alpha_z, eta, b_pre[n], b_post[n], b_z[(n-1)^2]]",
        ),
        opd(
            "pre",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"))),
            "[rows, n]",
        ),
        opd(
            "post",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"))),
            "[rows, n]",
        ),
        opd(
            "res",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("n"))),
            "[rows, n, n] destination/source",
        ),
        opd_null(
            "gamma",
            FORMAT_GAMMA_F32_OR_BF16,
            Access::ReadOnly,
            4,
            mul!(
                p!("has_norm"),
                p!("width"),
                sub!(c!(4), mul!(c!(2), p!("gamma_kind")))
            ),
            "[width] F32 or BF16 bits; null when has_norm == 0",
        ),
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("width"))),
            "[rows, width] F32 carrying the rounded aggregate; must not alias x",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("z_raw", 8),
        f_ptr("params", 16),
        f_ptr("pre", 24),
        f_ptr("post", 32),
        f_ptr("res", 40),
        f_ptr("gamma", 48),
        f_ptr("out", 56),
        f_f32("post_factor", 64, "post_factor_bits"),
        f_f32("selector_eps", 68, "selector_eps_bits"),
        f_f32("eps", 72, "eps_bits"),
        f_i32("bf16", 76, "bf16"),
        f_i32("has_norm", 80, "has_norm"),
        f_i32("gamma_kind", 84, "gamma_kind"),
        f_i32("n_rt", 88, "n"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Equal(p!("n"), c!(4)),
            "the symbol is the (n = 4, KPER = 10) instantiation (`stream_mix_route_aggregate_supported`)",
        ),
        rule(
            Rule::Equal(p!("width"), c!(2560)),
            "KPER * 256 = 2560 columns per row (`stream_mix_route_aggregate_supported`)",
        ),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(
            Rule::AtMost(mul!(p!("rows"), p!("n"), p!("width")), I32_MAX),
            "int index (check_geometry)",
        ),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
        rule(Rule::OneOf(p!("has_norm"), BOOL), "has_norm is a flag"),
        rule(Rule::OneOf(p!("gamma_kind"), BOOL), "gamma_kind is 0 or 1"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_route_aggregate_n4_k10_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_route_aggregate",
            source: "kernels/src/stream_mix_route_aggregate.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 1584,
        grid: [p!("rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_route_aggregate_n4_k10_bf16`: fused Helmert route finalize + aggregate (+ plain-gamma RMSNorm)
    /// over BF16 streams, one workgroup per row. `.kernarg_segment_size` 92, no hidden arguments, static LDS
    /// 1584 (the trace reports 2048: [INFERENCE] the allocation rounded up to 512 B).
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_fused.stream_mix_route_aggregate_n4_k10_bf16",
        version: 1,
        wave: 32,
        numeric: NUMERIC_STREAM_MIX_ROUTE_AGGREGATE_BF16,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 92,
        segment_bytes: 92,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("n", CFG_STREAMS), bind("width", CFG_HIDDEN)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "bf16",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(c!(2), p!("rows"), p!("n"), p!("width"))),
            flops: None,
            bytes: Some(add!(
                bf16b!(mul!(p!("rows"), p!("n"), p!("width"))),
                f32b!(mul!(
                    p!("rows"),
                    add!(ZDIM, mul!(c!(2), p!("n")), mul!(p!("n"), p!("n")), p!("width"))
                ))
            )),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "macs = 2*rows*n*width: n*width squares of the RMS accumulation plus n*width pre*x FMAs of the aggregate (kernel source); not charged: the norm (~3 FLOP per column), the zdim sigmoids/expf and the Helmert res chain (O(n^4) per row), bf16 rounds; bytes = BF16 x read once + F32 z_raw (4*zdim) + pre/post/res writes (4*(2n+n^2)) + F32 out (4*width) per row, the gamma plane (width*4 B, L2-resident) and params (4*(4+zdim) B per launch) not charged; rows exact from grid x; BF16 streams take this kernel at every row count (no admission predicate)",
        },
    };
}

// ================================================================================
// ffn_situ_grouped_rotate_i8_gfx1201
// ================================================================================

/// FLOP-equivalents per SiTU-GLU element: the same estimate as `generic_elem`'s `ffn_situ_*_f32` (two `tanhf`,
/// one `expf`, three divides, a few multiplies). An estimate, not a measured instruction count.
const SITU_FLOP_PER_ELEM: u64 = 64;
/// FLOP per rotated + quantized element: 8 FWHT butterfly add/sub stages, 1 sign multiply, 2 scale multiplies
/// (`v * 0.0625f * signs2`), then the quantizer's `|x|`, running max, divide, `rintf`, clamp min and max (6).
/// The per-block cross-lane max/sum reductions and the int accumulate are not charged.
const ROTATE_QUANT_FLOP_PER_ELEM: u64 = 8 + 1 + 2 + 6;
/// Sign planes `signs1`, `signs2` (256 F32 each), counted once per launch (re-reads are L2 hits).
const SIGNS_BYTES: u64 = 2 * 256 * 4;
/// Bytes of one `block_i8_128`.
const BLOCK_I8_128_BYTES: u64 = 136;

/// Row floor of the A8 prefill tier the fused producer rides: admitted at `PREFILL_QUANT_MIN_ROWS` (64) rows or
/// more by the caller (`down_producer_fused_admits` documents that the caller's rows floor already ran), and
/// turned off by `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED=0` (`Flags::gfx12_producer_quant_fused_enabled`).
const ADMIT_A8_PRODUCER: RowDomainV1 = RowDomainV1 {
    min_rows: 64,
    max_rows: None,
    predicate: "crates/hipfire-dispatch/src/pipeline/q8_prefill.rs::PREFILL_QUANT_MIN_ROWS",
    kill_env: Some("HIPFIRE_GFX12_PRODUCER_QUANT_FUSED"),
};

const MODULE_FFN_ROTATE_I8: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "ffn_grouped_activation_rotate_i8_gfx1201",
    source: "kernels/src/ffn_grouped_activation_rotate_i8.gfx1201.hip",
};

mod ffn_grouped {
    use super::*;

    const M_TOTAL: Expr = p!("m_total");
    const LOGICAL_N: Expr = p!("logical_n");
    const PHYS_N: Expr = p!("phys_n");
    const SLOTS: Expr = p!("total_slots");
    /// `block_i8_128` blocks per row: `phys_n / 128`.
    const I8_BLOCKS: Expr = Expr::DivExact(&PHYS_N, 128);
    /// Workgroups: one per (grouped row, 256-column group).
    const GRID_X: Expr = mul!(M_TOTAL, Expr::DivExact(&PHYS_N, 256));

    const PARAMS: &[ParamV1] = &[
        ip("m_total", "grouped (expert-sorted, padded) rows"),
        ip(
            "block_m",
            "rows per expert tile; pinned to 16 (MOE_GROUPED_BLOCK_M)",
        ),
        ip(
            "logical_n",
            "activation columns computed (gate and up are each logical_n wide); multiple of 8",
        ),
        ip(
            "phys_n",
            "rotated/quantized columns = the down GEMM's K, >= logical_n, multiple of 256; grid x = m_total * phys_n / 256",
        ),
        ip("n_experts", "AWQ table entries (0 without a table)"),
        ip(
            "total_slots",
            "rows of the sidecar (flat slots = rows * top_k); a flat slot outside it is skipped",
        ),
        ip("round_inputs", "1: bf16-round g and u before the activation"),
        ip("round_output", "1: bf16-round the activation result"),
        up("p0_bits", "f32 bits of gate_beta"),
        up("p1_bits", "f32 bits of value_beta"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "y_grouped",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(mul!(M_TOTAL, c!(2), LOGICAL_N)),
            "F32 [m_total, 2*logical_n]: gate in cols 0..logical_n, up in cols logical_n..2*logical_n; read as float4",
        ),
        opd(
            "sorted_slot_index",
            FORMAT_SORTED_SLOT_INDEX,
            Access::ReadOnly,
            4,
            f32b!(M_TOTAL),
            "flat slot of grouped row g, or -1 for a padding row (skipped, nothing written)",
        ),
        opd(
            "expert_tile_ids",
            FORMAT_EXPERT_TILE_IDS,
            Access::ReadOnly,
            4,
            f32b!(Expr::DivCeil(&M_TOTAL, BLOCK_M)),
            "read only when awq_table is non-null",
        ),
        opd_null(
            "awq_table",
            FORMAT_U64_AWQ_PTR_TABLE,
            Access::ReadOnly,
            8,
            mul!(p!("n_experts"), c!(8)),
            "null: no AWQ divide; entries point at F32[phys_n]",
        ),
        opd_null(
            "x_out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            16,
            f32b!(mul!(SLOTS, PHYS_N)),
            "null in production; else the rotated F32 rows [total_slots, phys_n] (row = flat slot), written as float4",
        ),
        opd(
            "signs1",
            FORMAT_MQ_FWHT_SIGNS_256,
            Access::ReadOnly,
            4,
            c!(256 * 4),
            "",
        ),
        opd(
            "signs2",
            FORMAT_MQ_FWHT_SIGNS_256,
            Access::ReadOnly,
            4,
            c!(256 * 4),
            "",
        ),
        opd(
            "x_i8",
            FORMAT_BLOCK_I8_128_KB_N,
            Access::WriteOnly,
            8,
            mul!(I8_BLOCKS, SLOTS, c!(BLOCK_I8_128_BYTES)),
            "[phys_n/128 x total_slots] block_i8_128; block index (2*group + half) * total_slots + flat_slot",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("y_grouped", 0),
        f_ptr("sorted_slot_index", 8),
        f_ptr("expert_tile_ids", 16),
        f_ptr("awq_table", 24),
        f_ptr("x_out", 32),
        f_ptr("signs1", 40),
        f_ptr("signs2", 48),
        f_ptr("x_i8", 56),
        f_i32("m_total", 64, "m_total"),
        f_i32("block_m", 68, "block_m"),
        f_i32("logical_n", 72, "logical_n"),
        f_i32("phys_n", 76, "phys_n"),
        f_i32("n_experts", 80, "n_experts"),
        f_i32("total_slots", 84, "total_slots"),
        f_i32("round_inputs", 88, "round_inputs"),
        f_i32("round_output", 92, "round_output"),
        f_f32("p0", 96, "p0_bits"),
        f_f32("p1", 100, "p1_bits"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(M_TOTAL), "m_total > 0"),
        rule(
            Rule::Equal(p!("block_m"), c!(BLOCK_M)),
            "the contract fixes the 16-row expert tile (expert_tile_ids extent)",
        ),
        rule(Rule::Positive(LOGICAL_N), "logical_n > 0"),
        rule(
            Rule::MultipleOf(LOGICAL_N, 8),
            "a lane's eight columns are all live or all padding (16-byte aligned gate/up reads)",
        ),
        rule(
            Rule::AtLeast(sub!(PHYS_N, LOGICAL_N), 0),
            "logical_n <= phys_n (a negative difference is rejected by the evaluation)",
        ),
        rule(
            Rule::MultipleOf(PHYS_N, 256),
            "whole 256-wide rotation groups",
        ),
        rule(Rule::Positive(SLOTS), "total_slots > 0"),
        rule(
            Rule::AtMost(GRID_X, U32_MAX),
            "grid x = m_total * (phys_n / 256) must fit u32 (the launcher refuses beyond)",
        ),
        rule(Rule::AtMost(M_TOTAL, I32_MAX), "int index"),
        rule(Rule::OneOf(p!("round_inputs"), BOOL), "flag is 0 or 1"),
        rule(Rule::OneOf(p!("round_output"), BOOL), "flag is 0 or 1"),
    ];

    const EXPORTS: &[ExportV1] = &[
        ExportV1 {
            symbol: "ffn_situ_grouped_rotate_i8_gfx1201",
            arch: Arch::Gfx1201,
            object: MODULE_FFN_ROTATE_I8,
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [GRID_X, c!(1), c!(1)],
            consts: &[],
        },
        // gfx1100 (7900 XTX) runs the same module text (symbol keeps its gfx1201 name); local hipcc build
        // `--offload-arch=gfx1100`: kernarg 104, 0 LDS, max flat workgroup 32.
        ExportV1 {
            symbol: "ffn_situ_grouped_rotate_i8_gfx1201",
            arch: Arch::Gfx1100,
            object: MODULE_FFN_ROTATE_I8,
            block: [32, 1, 1],
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid: [GRID_X, c!(1), c!(1)],
            consts: &[],
        },
    ];

    /// `ffn_situ_grouped_rotate_i8_gfx1201`: grouped gate/up SiTU-GLU activation + per-expert AWQ divide +
    /// un-scatter + FWHT256 rotation + `block_i8_128` quantization in one launch, one wave per (grouped row,
    /// 256-column group). `.kernarg_segment_size` 104, no hidden arguments, 0 LDS. The module text is
    /// `FFN_GROUPED_ACTIVATION_ROTATE_I8_SRC` (prelude + producer source), see the module docs.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_fused.ffn_situ_grouped_rotate_i8",
        version: 1,
        wave: 32,
        numeric: NUMERIC_FFN_SITU_ROTATE_I8,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 104,
        segment_bytes: 104,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("hidden", CFG_HIDDEN),
        // phys_n is the down GEMM's K: the cols of the expert-stacked down weight whose output rows are the
        // hidden size. The launch reads no weight; this only names the K the producer pads to.
        BindV1 {
            param: "phys_n",
            source: BindSource::Weight {
                by: "hidden",
                by_axis: WeightAxis::Rows,
                codecs: MQ_ROTATED_CODECS,
                stacking: Stacking::Stack,
                fuse_siblings: false,
            },
        },
        bind("logical_n", BindSource::Config(KEYS_MOE_INTERMEDIATE)),
        bind(
            "total_slots",
            BindSource::RoutedRows {
                topk: KEYS_TOPK,
                dense: "m_total",
            },
        ),
        bind("tokens", BindSource::RoutedTokens { dense: "m_total" }),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::ROTATE,
            codec: "f32>i8_128",
            pipe: Pipe::Memory,
            macs: None,
            flops: Some(mul!(
                SLOTS,
                add!(
                    mul!(c!(SITU_FLOP_PER_ELEM), LOGICAL_N),
                    mul!(c!(ROTATE_QUANT_FLOP_PER_ELEM), PHYS_N)
                )
            )),
            bytes: Some(add!(
                mul!(SLOTS, add!(mul!(c!(8), LOGICAL_N), c!(4))),
                mul!(SLOTS, I8_BLOCKS, c!(BLOCK_I8_128_BYTES)),
                c!(SIGNS_BYTES)
            )),
            rows: Some("tokens"),
            admit: ADMIT_A8_PRODUCER,
            bind: BIND,
            note: "useful work only: total_slots (= tokens*top-k) routed rows, excluding the m_total - total_slots padding rows (which read one sorted_slot_index word and exit); grid x = m_total*phys_n/256 has two unknowns so it constrains nothing, phys_n binds from the down weight (cols of the expert-stacked mq4 weight with hidden rows) and logical_n from the per-expert FFN width config; flops = total_slots*(64*logical_n activation [estimate] + 17*phys_n: 8 FWHT stages + 3 sign/scale multiplies + 6 quantize ops, nominal); bytes = gate/up F32 reads 8*logical_n + sorted_slot_index 4 per slot + (phys_n/128)*136 sidecar writes per slot + 2 KB sign planes once; x_out is null in production and the optional AWQ table (n_experts*phys_n*4 B) is not charged",
        },
    };
}

// ================================================================================
// ffn_situ_rows_rotate_i8_gfx1201
// ================================================================================

mod ffn_rows {
    use super::*;

    const ROWS: Expr = p!("rows");
    const LOGICAL_N: Expr = p!("logical_n");
    const PHYS_N: Expr = p!("phys_n");
    const GATE_LD: Expr = p!("gate_ld");
    const UP_LD: Expr = p!("up_ld");
    const I8_BLOCKS: Expr = Expr::DivExact(&PHYS_N, 128);
    /// Workgroups: one per (row, 256-column group).
    const GRID_X: Expr = mul!(ROWS, Expr::DivExact(&PHYS_N, 256));

    const PARAMS: &[ParamV1] = &[
        ip("rows", "activation rows"),
        ip(
            "logical_n",
            "activation columns computed; multiple of 8, <= gate_ld and up_ld",
        ),
        ip(
            "phys_n",
            "rotated/quantized columns = the down GEMM's K, >= logical_n, multiple of 256; grid x = rows * phys_n / 256",
        ),
        ip("gate_ld", "elements between gate row starts, multiple of 4"),
        ip("up_ld", "elements between up row starts, multiple of 4"),
        ip("round_inputs", "1: bf16-round g and u before the activation"),
        ip("round_output", "1: bf16-round the activation result"),
        up("p0_bits", "f32 bits of gate_beta"),
        up("p1_bits", "f32 bits of value_beta"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "gate",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(strided!(ROWS, GATE_LD, LOGICAL_N)),
            "F32 [rows, logical_n] at element stride gate_ld; base already includes the caller's element offset; read as float4",
        ),
        opd(
            "up",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(strided!(ROWS, UP_LD, LOGICAL_N)),
            "F32 [rows, logical_n] at element stride up_ld; read as float4",
        ),
        opd_null(
            "x_out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            16,
            f32b!(mul!(ROWS, PHYS_N)),
            "null in production; else the rotated F32 rows [rows, phys_n], written as float4",
        ),
        opd_null(
            "awq",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(PHYS_N),
            "null: no AWQ divide; else one F32 [phys_n] vector shared by every row",
        ),
        opd(
            "signs1",
            FORMAT_MQ_FWHT_SIGNS_256,
            Access::ReadOnly,
            4,
            c!(256 * 4),
            "",
        ),
        opd(
            "signs2",
            FORMAT_MQ_FWHT_SIGNS_256,
            Access::ReadOnly,
            4,
            c!(256 * 4),
            "",
        ),
        opd(
            "x_i8",
            FORMAT_BLOCK_I8_128_KB_N,
            Access::WriteOnly,
            8,
            mul!(I8_BLOCKS, ROWS, c!(BLOCK_I8_128_BYTES)),
            "[phys_n/128 x rows] block_i8_128; block index (2*group + half) * rows + row",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("gate", 0),
        f_ptr("up", 8),
        f_ptr("x_out", 16),
        f_ptr("awq", 24),
        f_ptr("signs1", 32),
        f_ptr("signs2", 40),
        f_ptr("x_i8", 48),
        f_i32("rows", 56, "rows"),
        f_i32("logical_n", 60, "logical_n"),
        f_i32("phys_n", 64, "phys_n"),
        f_i32("gate_ld", 68, "gate_ld"),
        f_i32("up_ld", 72, "up_ld"),
        f_i32("round_inputs", 76, "round_inputs"),
        f_i32("round_output", 80, "round_output"),
        f_f32("p0", 84, "p0_bits"),
        f_f32("p1", 88, "p1_bits"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(ROWS), "rows > 0"),
        rule(Rule::Positive(LOGICAL_N), "logical_n > 0"),
        rule(
            Rule::MultipleOf(LOGICAL_N, 8),
            "a lane's eight columns are all live or all padding (16-byte aligned gate/up reads)",
        ),
        rule(
            Rule::AtLeast(sub!(PHYS_N, LOGICAL_N), 0),
            "logical_n <= phys_n (a negative difference is rejected by the evaluation)",
        ),
        rule(
            Rule::MultipleOf(PHYS_N, 256),
            "whole 256-wide rotation groups",
        ),
        rule(Rule::MultipleOf(GATE_LD, 4), "float4 reads of gate rows"),
        rule(Rule::MultipleOf(UP_LD, 4), "float4 reads of up rows"),
        rule(
            Rule::AtLeast(sub!(GATE_LD, LOGICAL_N), 0),
            "gate_ld >= logical_n",
        ),
        rule(
            Rule::AtLeast(sub!(UP_LD, LOGICAL_N), 0),
            "up_ld >= logical_n",
        ),
        rule(
            Rule::AtMost(GRID_X, U32_MAX),
            "grid x = rows * (phys_n / 256) must fit u32 (the launcher refuses beyond)",
        ),
        rule(Rule::AtMost(ROWS, I32_MAX), "int index"),
        rule(Rule::OneOf(p!("round_inputs"), BOOL), "flag is 0 or 1"),
        rule(Rule::OneOf(p!("round_output"), BOOL), "flag is 0 or 1"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "ffn_situ_rows_rotate_i8_gfx1201",
        arch: Arch::Gfx1201,
        object: MODULE_FFN_ROTATE_I8,
        block: [32, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [GRID_X, c!(1), c!(1)],
        consts: &[],
    }];

    /// `ffn_situ_rows_rotate_i8_gfx1201`: dense-rows gate/up SiTU-GLU activation + optional vector AWQ divide +
    /// FWHT256 rotation + `block_i8_128` quantization in one launch, one wave per (row, 256-column group). Used
    /// for the shared expert and the dense FFN. `.kernarg_segment_size` 92, no hidden arguments, 0 LDS.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_fused.ffn_situ_rows_rotate_i8",
        version: 1,
        wave: 32,
        numeric: NUMERIC_FFN_SITU_ROTATE_I8,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 92,
        segment_bytes: 92,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("hidden", CFG_HIDDEN),
        // phys_n is the down GEMM's K: the cols of the single (unstacked) down weight whose output rows are
        // the hidden size. The node's call count selects the shared-expert (every MoE layer) or dense-FFN
        // (first layers) down weight.
        BindV1 {
            param: "phys_n",
            source: BindSource::Weight {
                by: "hidden",
                by_axis: WeightAxis::Rows,
                codecs: MQ_ROTATED_CODECS,
                stacking: Stacking::Single,
                fuse_siblings: false,
            },
        },
        bind("logical_n", BindSource::Derived(PHYS_N)),
        bind("rows", BindSource::ChunkRows),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::ROTATE,
            codec: "f32>i8_128",
            pipe: Pipe::Memory,
            macs: None,
            flops: Some(mul!(
                ROWS,
                add!(
                    mul!(c!(SITU_FLOP_PER_ELEM), LOGICAL_N),
                    mul!(c!(ROTATE_QUANT_FLOP_PER_ELEM), PHYS_N)
                )
            )),
            bytes: Some(add!(
                mul!(ROWS, mul!(c!(8), LOGICAL_N)),
                mul!(ROWS, I8_BLOCKS, c!(BLOCK_I8_128_BYTES)),
                c!(SIGNS_BYTES)
            )),
            rows: Some("rows"),
            admit: ADMIT_A8_PRODUCER,
            bind: BIND,
            note: "rows = chunk rows (the shared-expert and dense-FFN down sites see every token of the chunk; checked against the grid: 4096*(1280/256) = 20480 and 4096*(12032/256) = 192512 on the Raon trace) because grid x = rows*phys_n/256 has two unknowns and constrains nothing; phys_n binds from the down weight (cols of the unstacked mq4 weight with hidden rows, selected by the node's call count); logical_n is bound = phys_n, an UPPER BOUND of the gate/up reads and activation flops where the FFN width is padded (Raon shared expert: 1088 live of 1280 columns, the dense FFN 12032 = 47*256 is exact); flops = rows*(64*logical_n activation [estimate] + 17*phys_n rotate+quantize, nominal); bytes = gate/up F32 reads 8*logical_n per row + (phys_n/128)*136 sidecar writes per row + 2 KB sign planes once; x_out is null in production and the optional AWQ vector (phys_n*4 B) is not charged",
        },
    };
}

// ================================================================================
// latent_attention_scatter_expanded_q8
// ================================================================================

mod scatter_q8 {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("row0", "first row offset of this slice"),
        ip("n_rows", "projector rows of this launch"),
        ip("n_heads", "mirror heads"),
        ip("nope", "K_nope width, multiple of 4"),
        ip("v", "V width, multiple of 4"),
        ip("cap", "mirror plane rows (rows past it are skipped)"),
        ip(
            "blocks",
            "(row, head) pairs = n_rows * n_heads (grid x, not a kernarg)",
        ),
    ];

    /// Projector elements `blocks * (nope + v)`.
    const TOTAL: Expr = mul!(p!("blocks"), add!(p!("nope"), p!("v")));

    const OPERANDS: &[OperandV1] = &[
        opd(
            "proj",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(TOTAL),
            "[n_rows, H*(nope+v)] F32; read as float4",
        ),
        opd(
            "knope",
            FORMAT_I8_MIRROR_PLANE_VIEW,
            Access::WriteOnly,
            4,
            mul!(p!("cap"), p!("n_heads"), add!(p!("nope"), c!(2))),
            "[cap][H*(nope+2)] bytes: int8 codes then F16 scales",
        ),
        opd(
            "vmir",
            FORMAT_I8_MIRROR_PLANE_VIEW,
            Access::WriteOnly,
            4,
            mul!(p!("cap"), p!("n_heads"), add!(p!("v"), c!(2))),
            "[cap][H*(v+2)] bytes: int8 codes then F16 scales",
        ),
        opd_null(
            "base_ptr",
            FORMAT_I32_DEVICE_SCALAR,
            Access::ReadOnly,
            4,
            c!(4),
            "nullable device row base",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("proj", 0),
        f_ptr("knope", 8),
        f_ptr("vmir", 16),
        f_ptr("base_ptr", 24),
        f_i32("row0", 32, "row0"),
        f_i32("n_rows", 36, "n_rows"),
        f_i32("n_heads", 40, "n_heads"),
        f_i32("nope", 44, "nope"),
        f_i32("v", 48, "v"),
        f_i32("cap", 52, "cap"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Positive(p!("n_rows")),
            "n_rows > 0 (the launcher returns early at 0)",
        ),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::Positive(p!("nope")), "nope > 0"),
        rule(Rule::Positive(p!("v")), "v > 0"),
        rule(Rule::Positive(p!("cap")), "cap > 0"),
        rule(
            Rule::MultipleOf(p!("nope"), 4),
            "16-byte vector loads of the F32 projector rows",
        ),
        rule(
            Rule::MultipleOf(p!("v"), 4),
            "16-byte vector loads of the F32 projector rows",
        ),
        rule(
            Rule::MultipleOf(mul!(p!("n_heads"), add!(p!("nope"), c!(2))), 4),
            "4-byte code stores: K plane rows are whole words",
        ),
        rule(
            Rule::MultipleOf(mul!(p!("n_heads"), add!(p!("v"), c!(2))), 4),
            "4-byte code stores: V plane rows are whole words",
        ),
        rule(
            Rule::Equal(p!("blocks"), mul!(p!("n_rows"), p!("n_heads"))),
            "the launcher's grid: one 64-thread block per (row, head)",
        ),
        rule(
            Rule::AtMost(p!("blocks"), I32_MAX),
            "the launcher refuses grids beyond i32",
        ),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_scatter_expanded_q8",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_decompress",
            source: "kernels/src/latent_attention_decompress.hip",
        },
        block: [64, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [p!("blocks"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_scatter_expanded_q8`: projector output -> int8 K_nope/V mirrors with one F16 scale per
    /// (row, head, plane); one wave32 per (row, head, plane), the two planes of a (row, head) share a 64-thread
    /// block. `.kernarg_segment_size` 56, no hidden arguments, 0 LDS.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_fused.latent_attention_scatter_expanded_q8",
        version: 1,
        wave: 32,
        numeric: NUMERIC_LATENT_SCATTER_EXPANDED_Q8,
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 56,
        segment_bytes: 56,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("nope", CFG_NOPE), bind("v", CFG_V)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::GATHER,
            codec: "f32>i8",
            pipe: Pipe::Memory,
            macs: None,
            flops: Some(mul!(TOTAL, c!(6))),
            bytes: Some(add!(
                f32b!(TOTAL),
                mul!(p!("blocks"), add!(p!("nope"), p!("v"), c!(4)))
            )),
            rows: None,
            admit: ALL_ROWS,
            bind: BIND,
            note: "blocks = n_rows*n_heads is exact from the grid (n_rows is the mirror-fill slice, e.g. 512 of a 4096-row chunk, so it is not the chunk rows and no rows param is bound); flops = 6 nominal per projector element (|x|, running max, divide, rintf, clamp min and max); bytes = F32 projector reads + int8 codes and F16 scales written ((nope+2)+(v+2) per block); the projector plane was just written by the preceding GEMM, so a cache-resident read can make the measured time beat the HBM floor; the latent-cache base scalar (4 B) not charged; the Q8 mirror is opt-in (HIPFIRE_RAON_MIRROR_Q8, mirror.q8), so no kill env is named",
        },
    };
}

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &postmix_bf16::ABI,
    &route_aggregate_bf16::ABI,
    &ffn_grouped::ABI,
    &ffn_rows::ABI,
    &scatter_q8::ABI,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &postmix_bf16::WORK,
    &route_aggregate_bf16::WORK,
    &ffn_grouped::WORK,
    &ffn_rows::WORK,
    &scatter_q8::WORK,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval;

    /// Params of `abi` in declaration order; unnamed ones take the value 1.
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

    fn ev(abi: &KernelAbiV1, x: &Expr, given: &[(&str, u64)]) -> u64 {
        eval(x, abi, &abi.exports[0], &params_of(abi, given)).expect("expression evaluates")
    }

    #[test]
    fn every_contract_self_checks() {
        assert_eq!(ABIS.len(), 5);
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
                // gfx1201 everywhere; the grouped G5 down producer is also exported on gfx1100.
                assert!(
                    e.arch == Arch::Gfx1201
                        || (e.arch == Arch::Gfx1100
                            && e.symbol == "ffn_situ_grouped_rotate_i8_gfx1201")
                );
                let (found, _) = crate::families::find_export(e.arch, e.symbol).unwrap();
                assert_eq!(found.id, abi.id);
            }
        }
    }

    #[test]
    fn raon_stream_mix_bf16_postmix_grid_and_work() {
        // Observed (5, 4096, 1) workgroups, block 256: ceil(2560 / 512) = 5 column tiles, 4096 rows.
        let given = [("n", 4), ("width", 2560), ("rows", 4096)];
        assert_eq!(grid_of(&postmix_bf16::ABI, &given), [5, 4096, 1]);
        assert_eq!(postmix_bf16::ABI.exports[0].block, [256, 1, 1]);
        // width 2049..=2560 all need 5 tiles; 2048 needs 4.
        assert_eq!(
            grid_of(&postmix_bf16::ABI, &[("width", 2049), ("rows", 1)])[0],
            5
        );
        assert_eq!(
            grid_of(&postmix_bf16::ABI, &[("width", 2048), ("rows", 1)])[0],
            4
        );
        let w = &postmix_bf16::WORK.work;
        assert_eq!(
            ev(&postmix_bf16::ABI, &w.macs.unwrap(), &given),
            4096 * 4 * 5 * 2560
        );
        // x read + out write (BF16) + branch + post + res (F32) per call.
        assert_eq!(
            ev(&postmix_bf16::ABI, &w.bytes.unwrap(), &given),
            4096 * (2 * 2 * 4 * 2560 + 4 * 2560 + 4 * 4 + 4 * 16)
        );
        assert_eq!(w.rows, Some("rows"));
        assert_eq!(w.pipe, Pipe::ValuF32);
    }

    #[test]
    fn raon_stream_mix_route_aggregate_bf16_grid_lds_and_work() {
        // Observed (4096, 1, 1), block 256, trace LDS 2048 (1584 static rounded to 512 B).
        let given = [("n", 4), ("width", 2560), ("rows", 4096)];
        assert_eq!(grid_of(&route_aggregate_bf16::ABI, &given), [4096, 1, 1]);
        let e = &route_aggregate_bf16::ABI.exports[0];
        assert_eq!(e.block, [256, 1, 1]);
        assert_eq!(e.dynamic_lds, 0);
        assert_eq!(e.group_segment_fixed, 1584);
        assert_eq!(1584u32.next_multiple_of(512), 2048);
        // The 1584 B of static LDS is the object's `.group_segment_fixed_size` (cached gfx1201 hsaco).
        assert_eq!(route_aggregate_bf16::ABI.fields.len(), 15);
        let w = &route_aggregate_bf16::WORK.work;
        assert_eq!(
            ev(&route_aggregate_bf16::ABI, &w.macs.unwrap(), &given),
            2 * 4096 * 4 * 2560
        );
        // zdim = 2n + (n-1)^2 = 17 for n = 4.
        assert_eq!(
            ev(&route_aggregate_bf16::ABI, &w.bytes.unwrap(), &given),
            4096 * 2 * 4 * 2560 + 4096 * 4 * (17 + 8 + 16 + 2560)
        );
        assert_eq!(w.admit, ALL_ROWS);
    }

    #[test]
    fn raon_ffn_grouped_rotate_grid_binds_and_work() {
        // Observed (132480, 1, 1), block 32: m_total 26496 grouped rows x phys_n 1280 / 256 = 5 groups.
        let slots = 4096u64 * 6;
        let m_total = (slots + 128 * 15).div_ceil(16) * 16;
        assert_eq!(m_total, 26496);
        let given = [
            ("m_total", m_total),
            ("block_m", 16),
            ("logical_n", 1088),
            ("phys_n", 1280),
            ("n_experts", 128),
            ("total_slots", slots),
        ];
        assert_eq!(grid_of(&ffn_grouped::ABI, &given), [132_480, 1, 1]);
        assert_eq!(ffn_grouped::ABI.exports[0].block, [32, 1, 1]);
        assert_eq!(ffn_grouped::ABI.fields.len(), 18);
        let w = &ffn_grouped::WORK.work;
        assert_eq!(
            ev(&ffn_grouped::ABI, &w.flops.unwrap(), &given),
            slots * (64 * 1088 + 17 * 1280)
        );
        assert_eq!(
            ev(&ffn_grouped::ABI, &w.bytes.unwrap(), &given),
            slots * (8 * 1088 + 4) + slots * 10 * 136 + 2048
        );
        // The sidecar the producer writes is exactly the extent the contract states.
        let x_i8 = ffn_grouped::ABI.operands.last().unwrap();
        assert_eq!(x_i8.name, "x_i8");
        assert_eq!(
            ev(&ffn_grouped::ABI, &x_i8.extent.unwrap(), &given),
            10 * slots * 136
        );
        assert_eq!(w.op_class, op_class::ROTATE);
        assert_eq!(w.rows, Some("tokens"));
        assert_eq!(w.admit.min_rows, 64);
        assert_eq!(
            w.admit.predicate,
            "crates/hipfire-dispatch/src/pipeline/q8_prefill.rs::PREFILL_QUANT_MIN_ROWS"
        );
        assert_eq!(w.admit.kill_env, Some("HIPFIRE_GFX12_PRODUCER_QUANT_FUSED"));
    }

    #[test]
    fn raon_ffn_rows_rotate_grids_for_the_shared_expert_and_the_dense_ffn() {
        // Observed (20480, 1, 1): shared expert, rows 4096 x phys_n 1280 / 256 = 5.
        let shared = [
            ("rows", 4096),
            ("logical_n", 1280),
            ("phys_n", 1280),
            ("gate_ld", 1088),
            ("up_ld", 1088),
        ];
        assert_eq!(grid_of(&ffn_rows::ABI, &shared), [20_480, 1, 1]);
        // Observed (192512, 1, 1): dense layer, rows 4096 x phys_n 12032 / 256 = 47.
        let dense = [
            ("rows", 4096),
            ("logical_n", 12032),
            ("phys_n", 12032),
            ("gate_ld", 12032),
            ("up_ld", 12032),
        ];
        assert_eq!(grid_of(&ffn_rows::ABI, &dense), [192_512, 1, 1]);
        assert_eq!(12032 % 256, 0);
        assert_eq!(ffn_rows::ABI.exports[0].block, [32, 1, 1]);
        assert_eq!(ffn_rows::ABI.fields.len(), 16);
        let w = &ffn_rows::WORK.work;
        assert_eq!(
            ev(&ffn_rows::ABI, &w.flops.unwrap(), &dense),
            4096 * (64 * 12032 + 17 * 12032)
        );
        assert_eq!(
            ev(&ffn_rows::ABI, &w.bytes.unwrap(), &dense),
            4096 * 8 * 12032 + 4096 * 94 * 136 + 2048
        );
        assert_eq!(w.rows, Some("rows"));
        assert!(matches!(
            w.bind.iter().find(|b| b.param == "rows").unwrap().source,
            BindSource::ChunkRows
        ));
    }

    #[test]
    fn ffn_producer_binds_resolve_on_config_keys_and_weight_shapes_only() {
        for w in [&ffn_grouped::WORK, &ffn_rows::WORK] {
            assert_eq!(w.work.bind[0].param, "hidden");
            let BindSource::Config(keys) = w.work.bind[0].source else {
                panic!("{}: first bind is not the hidden size", w.abi.id);
            };
            assert!(keys.contains(&"hidden_size"));
            let BindSource::Weight {
                by,
                by_axis,
                codecs,
                fuse_siblings,
                ..
            } = w.work.bind[1].source
            else {
                panic!("{}: second bind is not the down-weight match", w.abi.id);
            };
            assert_eq!(w.work.bind[1].param, "phys_n");
            assert_eq!(by, "hidden");
            assert_eq!(by_axis, WeightAxis::Rows);
            assert_eq!(codecs, MQ_ROTATED_CODECS);
            assert!(!fuse_siblings);
        }
        let BindSource::Weight { stacking, .. } = ffn_grouped::WORK.work.bind[1].source else {
            unreachable!()
        };
        assert_eq!(stacking, Stacking::Stack);
        let BindSource::Weight { stacking, .. } = ffn_rows::WORK.work.bind[1].source else {
            unreachable!()
        };
        assert_eq!(stacking, Stacking::Single);
        let BindSource::Config(keys) = ffn_grouped::WORK
            .work
            .bind
            .iter()
            .find(|b| b.param == "logical_n")
            .unwrap()
            .source
        else {
            panic!("grouped logical_n is a config bind");
        };
        assert_eq!(keys[0], "moe_intermediate_size");
        let BindSource::RoutedRows { topk, dense } = ffn_grouped::WORK
            .work
            .bind
            .iter()
            .find(|b| b.param == "total_slots")
            .unwrap()
            .source
        else {
            panic!("grouped total_slots is the routed row count");
        };
        assert_eq!(dense, "m_total");
        assert!(topk.contains(&"num_experts_per_token"));
    }

    #[test]
    fn raon_scatter_q8_grid_inverts_to_blocks_and_512_rows() {
        // Observed (20480, 1, 1), block 64: n_rows 512 x n_heads 40.
        let given = [
            ("row0", 0),
            ("n_rows", 512),
            ("n_heads", 40),
            ("nope", 128),
            ("v", 128),
            ("cap", 4096),
            ("blocks", 20_480),
        ];
        assert_eq!(grid_of(&scatter_q8::ABI, &given), [20_480, 1, 1]);
        assert_eq!(512 * 40, 20_480);
        assert_eq!(scatter_q8::ABI.exports[0].block, [64, 1, 1]);
        assert_eq!(scatter_q8::ABI.exports[0].group_segment_fixed, 0);
        let w = &scatter_q8::WORK.work;
        assert_eq!(
            ev(&scatter_q8::ABI, &w.flops.unwrap(), &given),
            20_480 * 256 * 6
        );
        assert_eq!(
            ev(&scatter_q8::ABI, &w.bytes.unwrap(), &given),
            4 * 20_480 * 256 + 20_480 * 260
        );
        // The proj extent of the contract equals n_rows * H * (nope + v) F32.
        let proj = &scatter_q8::ABI.operands[0];
        assert_eq!(
            ev(&scatter_q8::ABI, &proj.extent.unwrap(), &given),
            4 * 512 * 40 * 256
        );
        // Calls: 7 MLA layers x 8 slices of 512 rows = 56 launches per 4096-row pass.
        assert_eq!(7 * (4096 / 512), 56);
        assert_eq!(w.op_class, op_class::GATHER);
        assert_eq!(w.rows, None);
    }

    #[test]
    fn contract_ids_are_unique_and_numeric_ids_are_distinct_per_semantics() {
        let mut ids: Vec<&str> = ABIS.iter().map(|a| a.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ABIS.len());
        assert!(ABIS.iter().all(|a| a.id.starts_with("generic_fused.")));
        // grouped and rows entries compute the same chain: one numeric contract.
        assert_eq!(ffn_grouped::ABI.numeric, ffn_rows::ABI.numeric);
        assert_ne!(postmix_bf16::ABI.numeric, route_aggregate_bf16::ABI.numeric);
    }

    #[test]
    fn kernarg_layouts_match_the_cached_gfx1201_objects() {
        // (segment bytes, explicit bytes, static LDS) read from the `llvm-readelf --notes` of the cached objects.
        for (abi, segment, lds) in [
            (&postmix_bf16::ABI, 52, 0),
            (&route_aggregate_bf16::ABI, 92, 1584),
            (&ffn_grouped::ABI, 104, 0),
            (&ffn_rows::ABI, 92, 0),
            (&scatter_q8::ABI, 56, 0),
        ] {
            assert_eq!(abi.segment_bytes, segment, "{}", abi.id);
            assert_eq!(abi.explicit_bytes, segment, "{}", abi.id);
            assert!(abi.hidden.is_empty(), "{}", abi.id);
            assert_eq!(abi.exports[0].group_segment_fixed, lds, "{}", abi.id);
            assert_eq!(abi.exports[0].dynamic_lds, 0, "{}", abi.id);
        }
    }
}
