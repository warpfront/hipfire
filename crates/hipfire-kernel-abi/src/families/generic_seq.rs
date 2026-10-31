// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic-op sequence-mixer launchers (hipcc): recurrences, causal conv, latent attention, stream mixing, norms.
//!
//! Evidence. Every kernarg layout below is read from the JIT-cached gfx1201 objects in
//! `~/.hipfire_kernels/gfx1201` (clang offload bundle unbundled with `clang-offload-bundler`, notes read
//! with `llvm-readelf --notes`; the `.radiowave.json` sidecars carry the same argument table). Where a
//! module has several hashes, the hash whose `.hip` equals the current `kernels/src` source was used:
//! `gated_delta_net_vector_serial.f142f905…`, `causal_conv_depthwise_rows.d124a0cd…`,
//! `latent_attention_flash.59b467b4…`, `raon_rda_prepare.7f81cdb1…` (`.kernarg_segment_size` 148; the
//! other hash is an older source), `gated_rmsnorm_sigmoid_rows.314b1139…`,
//! `latent_attention_decompress.3c44cbae…`, `latent_attention_gate.65e618a3…`,
//! `latent_attention_split.12a4195b…`, `latent_attention_absorb.cb9897ad…`, and the five single-hash
//! `stream_mix_*` modules.
//!
//! Scope. Only `Arch::Gfx1201` is exported: no gfx1100 object of these modules exists in
//! `~/.hipfire_kernels/gfx1100`, so a gfx1100 export would be unverified. The gfx1100/gfx1201 PM twins of the
//! gated-delta and conv kernels (`gated_delta_vector_pm`) are separate embedded objects with their own
//! symbols and are not covered here.
//!
//! Two conventions of this file:
//!
//! * f32 kernel scalars (`eps`, `scale`, ...) are `U32` params named `*_bits` carrying the IEEE-754 bit
//!   pattern, because [`ScalarTy`] has no float.
//! * [`ExportV1::dynamic_lds`] and [`ExportV1::block`] are constants, while the launchers derive them from
//!   the geometry (`head_dim` = block.x of the per-head kernels; `width * 4` dynamic LDS of the stream
//!   aggregate/mean; the flash prefill LDS from `dq`/`dv`). Each such contract therefore pins the geometry
//!   by a domain rule (`head_dim == 128`, `width == 2560`, `wa/wb/dv == 128/64/128`): other geometries need
//!   another contract. The pins are Raon's, the only model these generic launchers are exercised with.

use crate::work::{op_class, BindSource, BindV1, Pipe, WorkSpecV1, WorkV1, ALL_ROWS};
use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

use super::ggml_convert::FORMAT_F32_FLAT;

// ---- formats -------------------------------------------------------------------

/// BF16 flat tensor, row-major (latent cache, expanded K/V mirrors, packed absorb weight).
pub const FORMAT_BF16_FLAT: FormatId = FormatId("bf16.flat");
/// BF16 token/head strided view of the flash prefill (`(t * tok_stride + h * head_stride + c)`).
pub const FORMAT_BF16_TOKEN_HEAD_VIEW: FormatId = FormatId("bf16.token_head_view");
/// Device-resident `i32` scalar (row position / slot / token count).
pub const FORMAT_I32_DEVICE_SCALAR: FormatId = FormatId("i32.device_scalar");
/// Norm gamma `[n]`, F32 (`gamma_kind`/`gamma_bf16` = 0) or BF16 bits (= 1).
pub const FORMAT_GAMMA_F32_OR_BF16: FormatId = FormatId("norm.gamma.f32_or_bf16");

// ---- expression / layout helpers ------------------------------------------------

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

/// The standard hidden block of an `amdgpu` kernel whose first hidden word is at `$b`.
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
    ($b:expr, dyn_lds $d:expr) => {
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
            HiddenV1 {
                offset: $d,
                width: 4,
                kind: HiddenKind::DynamicLdsSize,
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
const fn u64p(name: &'static str, doc: &'static str) -> ParamV1 {
    ParamV1 {
        name,
        ty: ScalarTy::U64,
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
const fn f_u64(name: &'static str, offset: u32, param: &'static str) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Scalar {
            ty: ScalarTy::U64,
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

/// Config synonyms shared by the work specs.
const CFG_STREAMS: BindSource = BindSource::Config(&[
    "raon_mhc_streams",
    "mhc_streams",
    "num_residual_streams",
    "n_streams",
]);
const CFG_HIDDEN: BindSource = BindSource::Config(&["hidden_size", "n_embd", "d_model", "dim"]);
const CFG_HEADS: BindSource = BindSource::Config(&[
    "rda_config.num_heads",
    "linear_num_value_heads",
    "num_attention_heads",
    "n_head",
    "num_heads",
]);
const CFG_RDA_HEAD_DIM: BindSource =
    BindSource::Config(&["rda_config.head_dim", "linear_value_head_dim", "head_dim"]);
const CFG_RDA_KEY_DIM: BindSource =
    BindSource::Config(&["rda_config.head_dim", "linear_key_head_dim", "head_dim"]);
const CFG_CONV_K: BindSource = BindSource::Config(&[
    "rda_config.conv_kernel_size",
    "conv_kernel_size",
    "linear_conv_kernel_dim",
]);
const CFG_NOPE: BindSource = BindSource::Config(&["qk_nope_head_dim"]);
const CFG_ROPE: BindSource = BindSource::Config(&["qk_rope_head_dim"]);
const CFG_V: BindSource = BindSource::Config(&["v_head_dim"]);
const CFG_LATENT: BindSource = BindSource::Config(&["kv_lora_rank"]);

// ================================================================================
// gated_delta_net_vector_serial_f32
// ================================================================================

mod gdn {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip(
            "n_tokens",
            "tokens of this launch (upper bound when n_dev is passed)",
        ),
        ip("n_heads", "value heads (grid x)"),
        ip("qk_group", "value heads per q/k head (1 = no repeat)"),
        ip("decay_per_key", "1: log_decay is [T, H, K]; 0: [T, H]"),
        ip("round_out_bf16", "1: round the output to bf16"),
        ip(
            "slot",
            "baked state slot (overridden by slot_idx when passed)",
        ),
        ip("q_row_stride", "query row stride, floats"),
        ip("k_row_stride", "key row stride, floats"),
        ip("v_row_stride", "value row stride, floats"),
        ip("decay_row_stride", "log_decay row stride, floats"),
        ip("beta_row_stride", "beta row stride, floats"),
        ip("out_row_stride", "output row stride, floats"),
        ip("state_head_stride", "state head stride, floats"),
        ip("state_row_stride", "state value-row stride, floats"),
        u64p("state_slot_stride", "state slot stride, floats"),
        ip(
            "n_qk_heads",
            "q/k heads = n_heads / qk_group (not a kernarg)",
        ),
    ];

    /// Elements of one `[n_tokens, width]` plane at `stride`.
    const OPERANDS: &[OperandV1] = &[
        opd("query", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("n_tokens"), p!("q_row_stride"), mul!(p!("n_qk_heads"), p!("D")))), "[T, Hqk, K]"),
        opd("key", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("n_tokens"), p!("k_row_stride"), mul!(p!("n_qk_heads"), p!("D")))), "[T, Hqk, K]"),
        opd("value", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("n_tokens"), p!("v_row_stride"), mul!(p!("n_heads"), p!("D")))), "[T, H, V]"),
        opd("log_decay", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(
                p!("n_tokens"),
                p!("decay_row_stride"),
                mul!(p!("n_heads"), add!(c!(1), mul!(p!("decay_per_key"), c!(127))))
            )),
            "per-key [T, H, K] or per-head [T, H]"),
        opd("beta", FORMAT_F32_FLAT, Access::ReadOnly, 4,
            f32b!(strided!(p!("n_tokens"), p!("beta_row_stride"), p!("n_heads"))), "[T, H]"),
        opd("state", FORMAT_F32_FLAT, Access::ReadWrite, 4,
            f32b!(add!(
                mul!(p!("slot"), p!("state_slot_stride")),
                mul!(sub!(p!("n_heads"), c!(1)), p!("state_head_stride")),
                mul!(c!(127), p!("state_row_stride")),
                c!(128)
            )),
            "value-major [H, V, K] via head/row strides at slot*state_slot_stride; only the baked slot is bounded (slot_idx overrides it on device)"),
        opd("output", FORMAT_F32_FLAT, Access::WriteOnly, 4,
            f32b!(strided!(p!("n_tokens"), p!("out_row_stride"), mul!(p!("n_heads"), p!("D")))), "[T, H, V]"),
        opd_null("slot_idx", FORMAT_I32_DEVICE_SCALAR, Access::ReadOnly, 4, c!(4),
            "nullable device slot index"),
        opd_null("n_dev", FORMAT_I32_DEVICE_SCALAR, Access::ReadOnly, 4, c!(4),
            "nullable device token count (<= n_tokens)"),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("query", 0),
        f_ptr("key", 8),
        f_ptr("value", 16),
        f_ptr("log_decay", 24),
        f_ptr("beta", 32),
        f_ptr("state", 40),
        f_ptr("output", 48),
        f_ptr("slot_idx", 56),
        f_ptr("n_dev", 64),
        f_i32("n_tokens", 72, "n_tokens"),
        f_i32("n_heads", 76, "n_heads"),
        f_i32("qk_group", 80, "qk_group"),
        f_i32("decay_per_key", 84, "decay_per_key"),
        f_i32("round_out_bf16", 88, "round_out_bf16"),
        f_i32("slot", 92, "slot"),
        f_i32("q_row_stride", 96, "q_row_stride"),
        f_i32("k_row_stride", 100, "k_row_stride"),
        f_i32("v_row_stride", 104, "v_row_stride"),
        f_i32("decay_row_stride", 108, "decay_row_stride"),
        f_i32("beta_row_stride", 112, "beta_row_stride"),
        f_i32("out_row_stride", 116, "out_row_stride"),
        f_i32("state_head_stride", 120, "state_head_stride"),
        f_i32("state_row_stride", 124, "state_row_stride"),
        f_u64("state_slot_stride", 128, "state_slot_stride"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Positive(p!("n_tokens")),
            "n_tokens > 0 (the launcher returns early otherwise)",
        ),
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
            Rule::OneOf(p!("round_out_bf16"), BOOL),
            "round_out_bf16 is a flag",
        ),
        rule(
            Rule::AtLeast(p!("state_row_stride"), 128),
            "state rows must not overlap (>= head_dim)",
        ),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "gated_delta_net_vector_serial_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gated_delta_net_vector_serial",
            source: "kernels/src/gated_delta_net_vector_serial.hip",
        },
        block: [32, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        // GDV_TV = 8 value rows per workgroup; GDV_D = 128 (grid y = D / 8).
        grid: [p!("n_heads"), Expr::DivExact(&p!("D"), 8), c!(1)],
        consts: &[("D", 128)],
    }];

    /// `gated_delta_net_vector_serial_f32`: gated delta rule, head_dim 128 (GDV_D), one wave32 workgroup
    /// per (head, 8-row value tile); `.kernarg_segment_size` 136, no hidden arguments, 0 LDS.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.gated_delta_net_vector_serial_f32",
        version: 1,
        wave: 32,
        numeric: NumericContractId("gated_delta.vector_serial.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 136,
        segment_bytes: 136,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_tokens", BindSource::ChunkRows),
        bind("dk", CFG_RDA_KEY_DIM),
        bind("dv", CFG_RDA_HEAD_DIM),
    ];

    /// 4 MAC per state element per token (decay, `S.k`, rank-1 update, `S.q`; PHASE1's 8 FLOP).
    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::RECURRENCE,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(c!(4), p!("n_tokens"), p!("n_heads"), p!("dk"), p!("dv"))),
            flops: None,
            bytes: Some(f32b!(add!(
                mul!(p!("n_tokens"), p!("n_heads"), add!(mul!(c!(5), p!("D")), c!(1))),
                mul!(c!(2), p!("n_heads"), p!("D"), p!("D"))
            ))),
            rows: Some("n_tokens"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid x = n_heads exactly; rows = chunk rows; dk/dv from config (kernel is specialised to 128). Bytes: q,k counted at n_heads (an upper bound when qk_group > 1), decay per key channel, state read + written once.",
        },
    };
}

// ================================================================================
// causal_conv_depthwise_silu_rows_f32
// ================================================================================

mod conv {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_rows", "rows"),
        ip("n_channels", "channels (grid x * 128, interval)"),
        ip("ksize", "kernel width, 2..=8"),
        ip(
            "rows_per_chunk",
            "rows per workgroup row-slice (n_rows when hist_in aliases hist_out)",
        ),
        ip("in_row_stride", "input row stride, floats"),
        ip("out_row_stride", "output row stride, floats"),
        ip("w_ch_stride", "weight channel stride, floats"),
        ip("w_tap_stride", "weight tap stride, floats"),
        ip("hist_tap_stride", "history tap stride, floats"),
        ip("apply_silu", "1: SiLU on the conv output"),
        ip("round_bf16", "1: round the output to bf16"),
        ip(
            "row_chunks",
            "grid y = ceil(n_rows / rows_per_chunk) (not a kernarg)",
        ),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(strided!(
                p!("n_rows"),
                p!("out_row_stride"),
                p!("n_channels")
            )),
            "[N, C]",
        ),
        opd(
            "in",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(
                p!("n_rows"),
                p!("in_row_stride"),
                p!("n_channels")
            )),
            "[N, C]",
        ),
        opd(
            "weight",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(add!(
                mul!(sub!(p!("n_channels"), c!(1)), p!("w_ch_stride")),
                mul!(sub!(p!("ksize"), c!(1)), p!("w_tap_stride")),
                c!(1)
            )),
            "[C, taps] with explicit strides",
        ),
        opd(
            "hist_in",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(
                sub!(p!("ksize"), c!(1)),
                p!("hist_tap_stride"),
                p!("n_channels")
            )),
            "[K-1, C] oldest -> newest; may alias hist_out only when row_chunks == 1",
        ),
        opd(
            "hist_out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(strided!(
                sub!(p!("ksize"), c!(1)),
                p!("hist_tap_stride"),
                p!("n_channels")
            )),
            "[K-1, C] raw (pre-conv) values of the last K-1 rows",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("out", 0),
        f_ptr("in", 8),
        f_ptr("weight", 16),
        f_ptr("hist_in", 24),
        f_ptr("hist_out", 32),
        f_i32("n_rows", 40, "n_rows"),
        f_i32("n_channels", 44, "n_channels"),
        f_i32("ksize", 48, "ksize"),
        f_i32("rows_per_chunk", 52, "rows_per_chunk"),
        f_i32("in_row_stride", 56, "in_row_stride"),
        f_i32("out_row_stride", 60, "out_row_stride"),
        f_i32("w_ch_stride", 64, "w_ch_stride"),
        f_i32("w_tap_stride", 68, "w_tap_stride"),
        f_i32("hist_tap_stride", 72, "hist_tap_stride"),
        f_i32("apply_silu", 76, "apply_silu"),
        f_i32("round_bf16", 80, "round_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("n_channels")), "n_channels > 0"),
        rule(
            Rule::AtLeast(p!("ksize"), 2),
            "ksize >= 2 (the launcher refuses 1)",
        ),
        rule(Rule::AtMost(p!("ksize"), 8), "CCD_KMAX = 8 register window"),
        rule(Rule::Positive(p!("rows_per_chunk")), "rows_per_chunk > 0"),
        rule(
            Rule::AtLeast(
                sub!(mul!(p!("rows_per_chunk"), p!("row_chunks")), p!("n_rows")),
                0,
            ),
            "row_chunks * rows_per_chunk >= n_rows (every row covered)",
        ),
        rule(
            Rule::AtLeast(
                sub!(
                    p!("n_rows"),
                    mul!(p!("rows_per_chunk"), sub!(p!("row_chunks"), c!(1)))
                ),
                1,
            ),
            "row_chunks == ceil(n_rows / rows_per_chunk) (no empty trailing slice)",
        ),
        rule(Rule::OneOf(p!("apply_silu"), BOOL), "apply_silu is a flag"),
        rule(Rule::OneOf(p!("round_bf16"), BOOL), "round_bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "causal_conv_depthwise_silu_rows_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "causal_conv_depthwise_rows",
            source: "kernels/src/causal_conv_depthwise_rows.hip",
        },
        block: [128, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [
            Expr::DivCeil(&p!("n_channels"), 128),
            p!("row_chunks"),
            c!(1),
        ],
        consts: &[],
    }];

    /// `causal_conv_depthwise_silu_rows_f32`: depthwise causal conv, one thread per channel; grid
    /// `[ceil(C/128), ceil(N/rows_per_chunk)]`. Hidden block at 88, `.kernarg_segment_size` 344.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.causal_conv_depthwise_silu_rows_f32",
        version: 1,
        wave: 32,
        numeric: NumericContractId("causal_conv.depthwise_silu_rows.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(88),
        explicit_bytes: 84,
        segment_bytes: 344,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_rows", BindSource::ChunkRows),
        bind("ksize", CFG_CONV_K),
        bind("heads", CFG_HEADS),
        bind("head_dim", CFG_RDA_HEAD_DIM),
        bind(
            "n_channels",
            BindSource::Derived(mul!(p!("heads"), p!("head_dim"))),
        ),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::CONV,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: None,
            flops: Some(mul!(c!(2), p!("n_rows"), p!("n_channels"), p!("ksize"))),
            bytes: Some(f32b!(add!(
                mul!(c!(2), p!("n_rows"), p!("n_channels")),
                mul!(p!("n_channels"), p!("ksize")),
                mul!(c!(2), sub!(p!("ksize"), c!(1)), p!("n_channels"))
            ))),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid x only bounds channels to an interval of 128; n_channels is derived as heads * head_dim (one conv plane per q/k/v; a fused-qkv model with a different channel count must narrow it to the grid interval or stay partial). rows = chunk rows (grid y is the row-slice count, 1 when in place).",
        },
    };
}

// ================================================================================
// latent_attention_flash_prefill
// ================================================================================

mod flash {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_rows", "query rows of this call"),
        ip("n_heads", "heads (grid y)"),
        ip("wa", "k_a width (nope)"),
        ip("wb", "k_b width (rope tail)"),
        ip("dv", "value width"),
        ip("pos_offset", "added to *pos_ptr"),
        ip("max_rows", "keys visible at most (rows < max_rows)"),
        ip("ka_tok", "k_a token stride, elements"),
        ip("ka_head", "k_a head stride, elements"),
        ip("kb_tok", "k_b token stride, elements"),
        ip("kb_head", "k_b head stride, elements"),
        ip("v_tok", "v token stride, elements"),
        ip("v_head", "v head stride, elements"),
        up("scale_bits", "f32 bits of the score scale"),
        ip("round_y", "1: round y to bf16"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "q",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            16,
            f32b!(mul!(p!("n_rows"), p!("n_heads"), add!(p!("wa"), p!("wb")))),
            "[n_rows, H, dq] f32, rounded to bf16 at LDS staging",
        ),
        opd(
            "y",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            16,
            f32b!(mul!(p!("n_rows"), p!("n_heads"), p!("dv"))),
            "[n_rows, H, dv] f32",
        ),
        opd(
            "ka",
            FORMAT_BF16_TOKEN_HEAD_VIEW,
            Access::ReadOnly,
            16,
            bf16b!(add!(
                mul!(sub!(p!("max_rows"), c!(1)), p!("ka_tok")),
                mul!(sub!(p!("n_heads"), c!(1)), p!("ka_head")),
                p!("wa")
            )),
            "base pointer has the view offset applied",
        ),
        opd(
            "kb",
            FORMAT_BF16_TOKEN_HEAD_VIEW,
            Access::ReadOnly,
            16,
            bf16b!(add!(
                mul!(sub!(p!("max_rows"), c!(1)), p!("kb_tok")),
                mul!(sub!(p!("n_heads"), c!(1)), p!("kb_head")),
                p!("wb")
            )),
            "shared latent tail (head stride may be 0)",
        ),
        opd(
            "vv",
            FORMAT_BF16_TOKEN_HEAD_VIEW,
            Access::ReadOnly,
            16,
            bf16b!(add!(
                mul!(sub!(p!("max_rows"), c!(1)), p!("v_tok")),
                mul!(sub!(p!("n_heads"), c!(1)), p!("v_head")),
                p!("dv")
            )),
            "base pointer has the view offset applied",
        ),
        opd(
            "pos_ptr",
            FORMAT_I32_DEVICE_SCALAR,
            Access::ReadOnly,
            4,
            c!(4),
            "rows cached before this call",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("q", 0),
        f_ptr("y", 8),
        f_ptr("ka", 16),
        f_ptr("kb", 24),
        f_ptr("vv", 32),
        f_ptr("pos_ptr", 40),
        f_i32("n_rows", 48, "n_rows"),
        f_i32("n_heads", 52, "n_heads"),
        f_i32("wa", 56, "wa"),
        f_i32("wb", 60, "wb"),
        f_i32("dv", 64, "dv"),
        f_i32("pos_offset", 68, "pos_offset"),
        f_i32("max_rows", 72, "max_rows"),
        f_i32("ka_tok", 76, "ka_tok"),
        f_i32("ka_head", 80, "ka_head"),
        f_i32("kb_tok", 84, "kb_tok"),
        f_i32("kb_head", 88, "kb_head"),
        f_i32("v_tok", 92, "v_tok"),
        f_i32("v_head", 96, "v_head"),
        f_f32("scale", 100, "scale_bits"),
        f_i32("round_y", 104, "round_y"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::AtMost(p!("n_heads"), 65535), "grid y limit"),
        rule(Rule::Positive(p!("max_rows")), "max_rows > 0"),
        rule(Rule::Equal(p!("wa"), c!(128)),
            "dynamic LDS is a constant of the export: 4 waves * 16 * (dq + 8) * 2 + min(dv,128) * 144 = 44032 for wa 128, wb 64, dv 128"),
        rule(Rule::Equal(p!("wb"), c!(64)), "see wa"),
        rule(Rule::Equal(p!("dv"), c!(128)), "see wa; also grid z = ceil(dv / 128) = 1"),
        rule(Rule::MultipleOf(p!("ka_tok"), 8), "16-byte aligned operand loads"),
        rule(Rule::MultipleOf(p!("ka_head"), 8), "16-byte aligned operand loads"),
        rule(Rule::MultipleOf(p!("kb_tok"), 8), "16-byte aligned operand loads"),
        rule(Rule::MultipleOf(p!("kb_head"), 8), "16-byte aligned operand loads"),
        rule(Rule::MultipleOf(p!("v_tok"), 8), "16-byte aligned operand loads"),
        rule(Rule::MultipleOf(p!("v_head"), 8), "16-byte aligned operand loads"),
        rule(Rule::OneOf(p!("round_y"), BOOL), "round_y is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_flash_prefill",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_flash",
            source: "kernels/src/latent_attention_flash.hip",
        },
        block: [128, 1, 1],
        dynamic_lds: 44032,
        group_segment_fixed: 0,
        // 4 waves x 16 query rows per workgroup; z = value-column chunks of 128.
        grid: [
            Expr::DivCeil(&p!("n_rows"), 64),
            p!("n_heads"),
            Expr::DivCeil(&p!("dv"), 128),
        ],
        consts: &[],
    }];

    /// `latent_attention_flash_prefill` (4-wave variant; the `_wide` and `_q8` entries are other
    /// symbols). Hidden block at 112 plus `hidden_dynamic_lds_size@232`, `.kernarg_segment_size` 368.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_flash_prefill",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.flash_prefill.bf16_wmma_f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(112, dyn_lds 232),
        explicit_bytes: 108,
        segment_bytes: 368,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_rows", BindSource::ChunkRows),
        bind("prefix", BindSource::CausalPrefix),
        bind("wa", CFG_NOPE),
        bind("wb", CFG_ROPE),
        bind("dv", CFG_V),
    ];

    /// `H * (dqk + dv) * (q_rows * prefix + q_rows * (q_rows + 1) / 2)` (causal triangle).
    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::ATTENTION,
            codec: "bf16",
            pipe: Pipe::WmmaBf16,
            macs: Some(mul!(
                p!("n_heads"),
                add!(p!("wa"), p!("wb"), p!("dv")),
                add!(
                    mul!(p!("n_rows"), p!("prefix")),
                    Expr::DivExact(&mul!(p!("n_rows"), add!(p!("n_rows"), c!(1))), 2)
                )
            )),
            flops: None,
            bytes: Some(add!(
                f32b!(mul!(p!("n_rows"), p!("n_heads"), add!(p!("wa"), p!("wb"), p!("dv")))),
                bf16b!(mul!(
                    add!(p!("prefix"), p!("n_rows")),
                    p!("n_heads"),
                    add!(p!("wa"), p!("wb"), p!("dv"))
                ))
            )),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "n_heads exact from grid y; grid x only bounds n_rows (chunk rows binds it). prefix = rows already processed in the pass. Bytes: q read + y written once, K/V (prefix + rows) rows read once per head (an upper bound when the rope tail is shared across heads).",
        },
    };
}

// ================================================================================
// raon_rda_prepare_f32
// ================================================================================

mod rda_prepare {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("head_dim", "head dim (= block.x, pinned to 128)"),
        ip("n_heads", "heads (grid x)"),
        ip("n_rows", "rows (grid y)"),
        ip("conv_row_stride", "qconv/kconv/vconv row stride, floats"),
        ip("f_row_stride", "f row stride, floats"),
        ip("bd_row_stride", "b/delta row stride, floats"),
        ip("out_row_stride", "q/k/vprime/log_alpha row stride, floats"),
        ip("beta_row_stride", "beta_out row stride, floats"),
        up("q_scale_bits", "f32 bits of q_scale"),
        up("norm_eps_bits", "f32 bits of norm_eps"),
        up("lower_bound_bits", "f32 bits of the gate lower bound"),
        ip("round_bf16", "1: bf16 model-dtype boundaries"),
    ];

    macro_rules! plane_ro {
        ($name:literal, $stride:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                Access::ReadOnly,
                4,
                f32b!(strided!(
                    p!("n_rows"),
                    p!($stride),
                    mul!(p!("n_heads"), p!("head_dim"))
                )),
                "[N, H*K]",
            )
        };
    }
    macro_rules! plane_wo {
        ($name:literal) => {
            opd(
                $name,
                FORMAT_F32_FLAT,
                Access::WriteOnly,
                4,
                f32b!(strided!(
                    p!("n_rows"),
                    p!("out_row_stride"),
                    mul!(p!("n_heads"), p!("head_dim"))
                )),
                "[N, H*K]",
            )
        };
    }

    const OPERANDS: &[OperandV1] = &[
        plane_wo!("q_out"),
        plane_wo!("k_out"),
        plane_wo!("vprime_out"),
        plane_wo!("log_alpha_out"),
        opd(
            "beta_out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(strided!(p!("n_rows"), p!("beta_row_stride"), p!("n_heads"))),
            "[N, H]",
        ),
        plane_ro!("qconv", "conv_row_stride"),
        plane_ro!("kconv", "conv_row_stride"),
        plane_ro!("vconv", "conv_row_stride"),
        plane_ro!("f", "f_row_stride"),
        opd(
            "b",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(p!("n_rows"), p!("bd_row_stride"), p!("n_heads"))),
            "[N, H]",
        ),
        opd(
            "delta",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(p!("n_rows"), p!("bd_row_stride"), p!("n_heads"))),
            "[N, H]",
        ),
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
            4,
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
        f_ptr("qconv", 40),
        f_ptr("kconv", 48),
        f_ptr("vconv", 56),
        f_ptr("f", 64),
        f_ptr("b", 72),
        f_ptr("delta", 80),
        f_ptr("a_log", 88),
        f_ptr("dt_bias", 96),
        f_i32("head_dim", 104, "head_dim"),
        f_i32("n_heads", 108, "n_heads"),
        f_i32("conv_row_stride", 112, "conv_row_stride"),
        f_i32("f_row_stride", 116, "f_row_stride"),
        f_i32("bd_row_stride", 120, "bd_row_stride"),
        f_i32("out_row_stride", 124, "out_row_stride"),
        f_i32("beta_row_stride", 128, "beta_row_stride"),
        f_f32("q_scale", 132, "q_scale_bits"),
        f_f32("norm_eps", 136, "norm_eps_bits"),
        f_f32("lower_bound", 140, "lower_bound_bits"),
        f_i32("round_bf16", 144, "round_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Equal(p!("head_dim"), c!(128)),
            "block.x = head_dim is a constant of the export",
        ),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::OneOf(p!("round_bf16"), BOOL), "round_bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "raon_rda_prepare_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "raon_rda_prepare",
            source: "kernels/src/raon_rda_prepare.hip",
        },
        block: [128, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 2048,
        grid: [p!("n_heads"), p!("n_rows"), c!(1)],
        consts: &[],
    }];

    /// `raon_rda_prepare_f32`: q/k normalisation, decay gate, beta, write ratio; one workgroup per
    /// (head, row). `.kernarg_segment_size` 148, no hidden arguments, static LDS 2048 (`rq[256]`, `rk[256]`).
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.raon_rda_prepare_f32",
        version: 1,
        wave: 32,
        numeric: NumericContractId("raon_rda.prepare.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 148,
        segment_bytes: 148,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("head_dim", CFG_RDA_HEAD_DIM)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::NORM,
            codec: "f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(f32b!(add!(
                mul!(c!(8), p!("n_rows"), p!("n_heads"), p!("head_dim")),
                mul!(c!(3), p!("n_rows"), p!("n_heads")),
                mul!(p!("n_heads"), p!("head_dim")),
                p!("n_heads")
            ))),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid (n_heads, n_rows) exact; 4 planes read (qconv, kconv, vconv, f) and 4 written, plus per-head b/delta/beta and dt_bias/a_log; transcendental ALU not charged.",
        },
    };
}

// ================================================================================
// gated_rmsnorm_sigmoid_rows_f32
// ================================================================================

mod gated_rms {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("head_dim", "head dim (= block.x, pinned to 128)"),
        ip("n_heads", "heads (grid x)"),
        ip("n_rows", "rows (grid y)"),
        ip("x_row_stride", "x row stride, floats"),
        ip("gate_row_stride", "gate row stride, floats"),
        ip("out_row_stride", "out row stride, floats"),
        up("eps_bits", "f32 bits of eps"),
        ip("round_bf16", "1: round the output to bf16"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(strided!(
                p!("n_rows"),
                p!("out_row_stride"),
                mul!(p!("n_heads"), p!("head_dim"))
            )),
            "[N, H*D]",
        ),
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(
                p!("n_rows"),
                p!("x_row_stride"),
                mul!(p!("n_heads"), p!("head_dim"))
            )),
            "[N, H*D]",
        ),
        opd(
            "gate",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(
                p!("n_rows"),
                p!("gate_row_stride"),
                mul!(p!("n_heads"), p!("head_dim"))
            )),
            "[N, H*D]",
        ),
        opd(
            "gamma",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
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
        f_i32("x_row_stride", 40, "x_row_stride"),
        f_i32("gate_row_stride", 44, "gate_row_stride"),
        f_i32("out_row_stride", 48, "out_row_stride"),
        f_f32("eps", 52, "eps_bits"),
        f_i32("round_bf16", 56, "round_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(
            Rule::Equal(p!("head_dim"), c!(128)),
            "block.x = head_dim is a constant of the export",
        ),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::OneOf(p!("round_bf16"), BOOL), "round_bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "gated_rmsnorm_sigmoid_rows_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "gated_rmsnorm_sigmoid_rows",
            source: "kernels/src/gated_rmsnorm_sigmoid_rows.hip",
        },
        block: [128, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 4096,
        grid: [p!("n_heads"), p!("n_rows"), c!(1)],
        consts: &[],
    }];

    /// `gated_rmsnorm_sigmoid_rows_f32`: per-(row, head) RMSNorm times sigmoid(gate).
    /// `.kernarg_segment_size` 60, no hidden arguments, static LDS 4096 (`red[1024]`).
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.gated_rmsnorm_sigmoid_rows_f32",
        version: 1,
        wave: 32,
        numeric: NumericContractId("gated_rmsnorm.sigmoid_rows.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 60,
        segment_bytes: 60,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("head_dim", CFG_RDA_HEAD_DIM)];

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
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid (n_heads, n_rows) exact; x and gate read, out written.",
        },
    };
}

// ================================================================================
// stream_mix_*
// ================================================================================

mod postmix {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams, 1..=8"),
        ip("width", "row width (grid x * 256, interval)"),
        ip("bf16", "1: round to bf16"),
        ip("rows", "rows (grid y, not a kernarg)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width] old streams",
        ),
        opd(
            "branch",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("width"))),
            "[rows, width]",
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
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width]; must not alias x (ping-pong)",
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
        f_i32("bf16", 48, "bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::AtLeast(p!("n"), 1), "n >= 1"),
        rule(Rule::AtMost(p!("n"), 8), "SM_MAX_N = 8 register window"),
        rule(Rule::Positive(p!("width")), "width > 0"),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(
            Rule::AtMost(mul!(p!("rows"), p!("n"), p!("width")), I32_MAX),
            "int index",
        ),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_postmix",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_postmix",
            source: "kernels/src/stream_mix_mix.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&p!("width"), 256), p!("rows"), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_postmix`: `out[i,c] = round(sum_j res[i,j] x[j,c] + post[i] branch[c])`.
    /// `.kernarg_segment_size` 52, no hidden arguments, 0 LDS.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.stream_mix_postmix",
        version: 1,
        wave: 32,
        numeric: NumericContractId("stream_mix.postmix.f32"),
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
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(p!("rows"), p!("n"), add!(p!("n"), c!(1)), p!("width"))),
            flops: None,
            bytes: Some(f32b!(mul!(
                p!("rows"),
                add!(
                    mul!(c!(2), p!("n"), p!("width")),
                    p!("width"),
                    p!("n"),
                    mul!(p!("n"), p!("n"))
                )
            ))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid y; width = hidden size (grid x is an interval of 256).",
        },
    };
}

mod aggregate {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams"),
        ip(
            "width",
            "row width (pinned to 2560 by the constant dynamic LDS)",
        ),
        up("eps_bits", "f32 bits of eps"),
        ip("bf16", "1: round to bf16"),
        ip("has_norm", "1: plain-gamma RMSNorm after the sum"),
        ip("gamma_kind", "0: gamma F32, 1: gamma BF16 bits"),
        ip("rows", "rows (grid x)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width] old streams",
        ),
        opd(
            "pre",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"))),
            "[rows, n]",
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
            "[rows, width]; must not alias x",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("pre", 8),
        f_ptr("gamma", 16),
        f_ptr("out", 24),
        f_i32("n", 32, "n"),
        f_i32("width", 36, "width"),
        f_f32("eps", 40, "eps_bits"),
        f_i32("bf16", 44, "bf16"),
        f_i32("has_norm", 48, "has_norm"),
        f_i32("gamma_kind", 52, "gamma_kind"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n")), "n > 0"),
        rule(
            Rule::Equal(p!("width"), c!(2560)),
            "dynamic LDS = width * 4 = 10240 is a constant of the export",
        ),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
        rule(Rule::OneOf(p!("has_norm"), BOOL), "has_norm is a flag"),
        rule(Rule::OneOf(p!("gamma_kind"), BOOL), "gamma_kind is 0 or 1"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_aggregate",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_aggregate",
            source: "kernels/src/stream_mix_aggregate.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 10240,
        group_segment_fixed: 1024,
        grid: [p!("rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_aggregate`: `out = [norm](round(sum_s pre[s] x[s]))`, one workgroup per row.
    /// `.kernarg_segment_size` 56, no hidden arguments, static LDS 1024 + dynamic `width * 4`.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.stream_mix_aggregate",
        version: 1,
        wave: 32,
        numeric: NumericContractId("stream_mix.aggregate.f32"),
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

    const BIND: &[BindV1] = &[bind("n", CFG_STREAMS), bind("width", CFG_HIDDEN)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(p!("rows"), p!("n"), p!("width"))),
            flops: None,
            bytes: Some(f32b!(mul!(
                p!("rows"),
                add!(mul!(p!("n"), p!("width")), p!("n"), p!("width"))
            ))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid x (one workgroup per row). The optional RMSNorm (3 FLOP/element) and gamma read are not charged.",
        },
    };
}

mod helmert {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams, 2..=8"),
        ip("x_dim", "n * width"),
        up("post_factor_bits", "f32 bits of post_factor"),
        up("selector_eps_bits", "f32 bits of selector_eps"),
        ip("bf16", "1: round z to bf16"),
        ip("rows", "rows (grid x)"),
    ];

    /// `zdim = 2n + (n-1)^2`.
    const ZDIM: Expr = add!(
        mul!(c!(2), p!("n")),
        mul!(sub!(p!("n"), c!(1)), sub!(p!("n"), c!(1)))
    );

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("x_dim"))),
            "[rows, x_dim] old streams",
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
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("z_raw", 8),
        f_ptr("params", 16),
        f_ptr("pre", 24),
        f_ptr("post", 32),
        f_ptr("res", 40),
        f_i32("n", 48, "n"),
        f_i32("x_dim", 52, "x_dim"),
        f_f32("post_factor", 56, "post_factor_bits"),
        f_f32("selector_eps", 60, "selector_eps_bits"),
        f_i32("bf16", 64, "bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::AtLeast(p!("n"), 2), "the Helmert chart needs n >= 2"),
        rule(Rule::AtMost(p!("n"), 8), "SM_MAX_N = 8"),
        rule(Rule::Positive(p!("x_dim")), "x_dim > 0"),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_route_helmert",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_route_helmert",
            source: "kernels/src/stream_mix_route_helmert.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 1552,
        grid: [p!("rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_route_helmert`: selector finalize + Helmert chart, one workgroup per row.
    /// `.kernarg_segment_size` 68, no hidden arguments, static LDS 1552.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.stream_mix_route_helmert",
        version: 1,
        wave: 32,
        numeric: NumericContractId("stream_mix.route_helmert.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 68,
        segment_bytes: 68,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n", CFG_STREAMS),
        bind("width", CFG_HIDDEN),
        bind("x_dim", BindSource::Derived(mul!(p!("n"), p!("width")))),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: Some(mul!(p!("rows"), p!("x_dim"))),
            flops: None,
            bytes: Some(f32b!(add!(
                mul!(p!("rows"), add!(p!("x_dim"), ZDIM, mul!(c!(2), p!("n")), mul!(p!("n"), p!("n")))),
                c!(4),
                ZDIM
            ))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid x. MACs are the weightless RMS sum of squares over x_dim; the O(n^3) chart arithmetic per row is not charged.",
        },
    };
}

mod broadcast {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_streams", "streams"),
        ip("width", "row width (grid x * 256, interval)"),
        ip("src_stride", "src row stride, floats (>= width)"),
        ip("bf16", "1: round to bf16"),
        ip("rows", "rows (grid y, not a kernarg)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "src",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(strided!(p!("rows"), p!("src_stride"), p!("width"))),
            "[rows, src_stride]",
        ),
        opd(
            "dst",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n_streams"), p!("width"))),
            "[rows, n_streams, width]; must not alias src",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("src", 0),
        f_ptr("dst", 8),
        f_i32("n_streams", 16, "n_streams"),
        f_i32("width", 20, "width"),
        f_i32("src_stride", 24, "src_stride"),
        f_i32("bf16", 28, "bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_streams")), "n_streams > 0"),
        rule(Rule::Positive(p!("width")), "width > 0"),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(
            Rule::AtMost(mul!(p!("rows"), p!("n_streams"), p!("width")), I32_MAX),
            "int index",
        ),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_broadcast",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_broadcast",
            source: "kernels/src/stream_mix_broadcast.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&p!("width"), 256), p!("rows"), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_broadcast`: replicate one row into every stream. `.kernarg_segment_size` 32.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.stream_mix_broadcast",
        version: 1,
        wave: 32,
        numeric: NumericContractId("stream_mix.broadcast.f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 32,
        segment_bytes: 32,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[bind("n_streams", CFG_STREAMS), bind("width", CFG_HIDDEN)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(f32b!(mul!(
                p!("rows"),
                p!("width"),
                add!(c!(1), p!("n_streams"))
            ))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid y; one source row read, n_streams rows written.",
        },
    };
}

mod mean {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n", "streams"),
        ip(
            "width",
            "row width (pinned to 2560 by the constant dynamic LDS)",
        ),
        up("eps_bits", "f32 bits of eps"),
        ip("bf16", "1: round to bf16"),
        ip("has_norm", "1: plain-gamma RMSNorm after the mean"),
        ip("gamma_kind", "0: gamma F32, 1: gamma BF16 bits"),
        ip("rows", "rows (grid x)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("rows"), p!("n"), p!("width"))),
            "[rows, n, width]",
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
            "[rows, width]; must not alias x",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("gamma", 8),
        f_ptr("out", 16),
        f_i32("n", 24, "n"),
        f_i32("width", 28, "width"),
        f_f32("eps", 32, "eps_bits"),
        f_i32("bf16", 36, "bf16"),
        f_i32("has_norm", 40, "has_norm"),
        f_i32("gamma_kind", 44, "gamma_kind"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n")), "n > 0"),
        rule(
            Rule::Equal(p!("width"), c!(2560)),
            "dynamic LDS = width * 4 = 10240 is a constant of the export",
        ),
        rule(Rule::Positive(p!("rows")), "rows > 0"),
        rule(Rule::OneOf(p!("bf16"), BOOL), "bf16 is a flag"),
        rule(Rule::OneOf(p!("has_norm"), BOOL), "has_norm is a flag"),
        rule(Rule::OneOf(p!("gamma_kind"), BOOL), "gamma_kind is 0 or 1"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "stream_mix_mean",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "stream_mix_mean",
            source: "kernels/src/stream_mix_mean.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 10240,
        group_segment_fixed: 1024,
        grid: [p!("rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `stream_mix_mean`: `out = [norm](round(mean_s x[s]))`, one workgroup per row.
    /// `.kernarg_segment_size` 48, static LDS 1024 + dynamic `width * 4`.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.stream_mix_mean",
        version: 1,
        wave: 32,
        numeric: NumericContractId("stream_mix.mean.f32"),
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

    const BIND: &[BindV1] = &[bind("n", CFG_STREAMS), bind("width", CFG_HIDDEN)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::STREAM_MIX,
            codec: "f32",
            pipe: Pipe::ValuF32,
            macs: None,
            flops: Some(mul!(p!("rows"), p!("n"), p!("width"))),
            bytes: Some(f32b!(mul!(p!("rows"), p!("width"), add!(p!("n"), c!(1))))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid x. The optional RMSNorm and gamma read are not charged.",
        },
    };
}

// ================================================================================
// latent_attention helpers
// ================================================================================

mod scatter {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("row0", "first row offset"),
        ip("n_rows", "rows"),
        ip("n_heads", "heads"),
        ip("nope", "K_nope width"),
        ip("v", "V width"),
        ip("cap", "mirror capacity (rows)"),
    ];

    /// `n_rows * n_heads * (nope + v)` projector elements.
    const TOTAL: Expr = mul!(p!("n_rows"), p!("n_heads"), add!(p!("nope"), p!("v")));

    const OPERANDS: &[OperandV1] = &[
        opd(
            "proj",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(TOTAL),
            "[n_rows, H*(nope+v)]",
        ),
        opd(
            "knope",
            FORMAT_BF16_FLAT,
            Access::WriteOnly,
            2,
            bf16b!(mul!(p!("cap"), p!("n_heads"), p!("nope"))),
            "[cap, H, nope]",
        ),
        opd(
            "vmir",
            FORMAT_BF16_FLAT,
            Access::WriteOnly,
            2,
            bf16b!(mul!(p!("cap"), p!("n_heads"), p!("v"))),
            "[cap, H, v]",
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
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::Positive(p!("nope")), "nope > 0"),
        rule(Rule::Positive(p!("v")), "v > 0"),
        rule(Rule::Positive(p!("cap")), "cap > 0"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_scatter_expanded_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_decompress",
            source: "kernels/src/latent_attention_decompress.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&TOTAL, 256), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_scatter_expanded_bf16`: projector output -> BF16 K_nope/V mirrors.
    /// Hidden block at 56, `.kernarg_segment_size` 312.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_scatter_expanded_bf16",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.scatter_expanded.bf16_rne"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(56),
        explicit_bytes: 56,
        segment_bytes: 312,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_rows", BindSource::ChunkRows),
        bind("n_heads", CFG_HEADS),
        bind("nope", CFG_NOPE),
        bind("v", CFG_V),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::GATHER,
            codec: "f32>bf16",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(mul!(TOTAL, c!(6))),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid x is total/256 (an interval); rows = chunk rows; heads from config num_attention_heads [INFERENCE: the mirror head count is the attention head count].",
        },
    };
}

mod gate {
    use super::*;

    const PARAMS: &[ParamV1] = &[ip("n", "elements")];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "y",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(p!("n")),
            "[n] bf16-valued",
        ),
        opd(
            "g",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(p!("n")),
            "[n] bf16-valued",
        ),
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(p!("n")),
            "[n]; may alias y",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("y", 0),
        f_ptr("g", 8),
        f_ptr("out", 16),
        f_i32("n", 24, "n"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[rule(Rule::Positive(p!("n")), "n > 0")];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_gate_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_gate",
            source: "kernels/src/latent_attention_gate.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&p!("n"), 256), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_gate_bf16`: `out = bf16(y * bf16(sigmoid(g)))`. Hidden block at 32,
    /// `.kernarg_segment_size` 288.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_gate_bf16",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.gate.bf16_sigmoid"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(32),
        explicit_bytes: 28,
        segment_bytes: 288,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("rows", BindSource::ChunkRows),
        bind("heads", CFG_HEADS),
        bind("v", CFG_V),
        bind(
            "n",
            BindSource::Derived(mul!(p!("rows"), p!("heads"), p!("v"))),
        ),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::ACTIVATION,
            codec: "f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(f32b!(mul!(c!(3), p!("n")))),
            rows: Some("rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "n = rows * heads * v_head_dim [INFERENCE: the gate covers the attention output plane [rows, H, v]]; grid x is an interval of 256 that the derived value must fall in.",
        },
    };
}

mod split_kv {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_rows", "rows (grid x)"),
        ip("latent", "latent channels (normalised)"),
        ip("tail", "tail channels (copied)"),
        up("eps_bits", "f32 bits of eps"),
        ip("gamma_bf16", "1: gamma is BF16 bits"),
        ip("cache_rows", "cache capacity (rows)"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "kv_a",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("n_rows"), add!(p!("latent"), p!("tail")))),
            "[n_rows, latent+tail] bf16-valued f32",
        ),
        opd(
            "gamma",
            FORMAT_GAMMA_F32_OR_BF16,
            Access::ReadOnly,
            4,
            mul!(p!("latent"), sub!(c!(4), mul!(c!(2), p!("gamma_bf16")))),
            "[latent] F32 or BF16 bits",
        ),
        opd(
            "cache",
            FORMAT_BF16_FLAT,
            Access::WriteOnly,
            2,
            bf16b!(mul!(p!("cache_rows"), add!(p!("latent"), p!("tail")))),
            "[cache_rows, latent+tail]; rows *pos_ptr + i written",
        ),
        opd(
            "pos_ptr",
            FORMAT_I32_DEVICE_SCALAR,
            Access::ReadOnly,
            4,
            c!(4),
            "rows already in the cache",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("kv_a", 0),
        f_ptr("gamma", 8),
        f_ptr("cache", 16),
        f_ptr("pos_ptr", 24),
        f_i32("n_rows", 32, "n_rows"),
        f_i32("latent", 36, "latent"),
        f_i32("tail", 40, "tail"),
        f_f32("eps", 44, "eps_bits"),
        f_i32("gamma_bf16", 48, "gamma_bf16"),
        f_i32("cache_rows", 52, "cache_rows"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("latent")), "latent > 0"),
        rule(Rule::Positive(p!("cache_rows")), "cache_rows > 0"),
        rule(Rule::OneOf(p!("gamma_bf16"), BOOL), "gamma_bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_split_kv_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_split",
            source: "kernels/src/latent_attention_split.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 1024,
        grid: [p!("n_rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_split_kv_bf16`: RMSNorm over the latent channels, BF16 round, tail copy,
    /// into cache row `*pos_ptr + i`. `.kernarg_segment_size` 56, no hidden arguments, static LDS 1024.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_split_kv_bf16",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.split_kv.rms_bf16_rne"),
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

    const BIND: &[BindV1] = &[bind("latent", CFG_LATENT), bind("tail", CFG_ROPE)];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::GATHER,
            codec: "f32>bf16",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(add!(
                mul!(c!(6), p!("n_rows"), add!(p!("latent"), p!("tail"))),
                f32b!(p!("latent"))
            )),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "rows exact from grid x; latent = kv_lora_rank, tail = qk_rope_head_dim.",
        },
    };
}

mod rms_rows {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_rows", "rows (grid x)"),
        ip("width", "row width"),
        up("eps_bits", "f32 bits of eps"),
        ip("gamma_bf16", "1: gamma is BF16 bits"),
    ];

    const OPERANDS: &[OperandV1] = &[
        opd(
            "x",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(mul!(p!("n_rows"), p!("width"))),
            "[n_rows, width]",
        ),
        opd(
            "gamma",
            FORMAT_GAMMA_F32_OR_BF16,
            Access::ReadOnly,
            4,
            mul!(p!("width"), sub!(c!(4), mul!(c!(2), p!("gamma_bf16")))),
            "[width] F32 or BF16 bits",
        ),
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(mul!(p!("n_rows"), p!("width"))),
            "[n_rows, width] bf16-valued f32; may alias x",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("x", 0),
        f_ptr("gamma", 8),
        f_ptr("out", 16),
        f_i32("n_rows", 24, "n_rows"),
        f_i32("width", 28, "width"),
        f_f32("eps", 32, "eps_bits"),
        f_i32("gamma_bf16", 36, "gamma_bf16"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("width")), "width > 0"),
        rule(Rule::OneOf(p!("gamma_bf16"), BOOL), "gamma_bf16 is a flag"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_rms_rows_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_split",
            source: "kernels/src/latent_attention_split.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 1024,
        grid: [p!("n_rows"), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_rms_rows_bf16`: row RMSNorm, BF16 result kept as F32.
    /// `.kernarg_segment_size` 40, no hidden arguments, static LDS 1024.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_rms_rows_bf16",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.rms_rows.bf16_rne"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: &[],
        explicit_bytes: 40,
        segment_bytes: 40,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::NORM,
            codec: "f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(f32b!(mul!(p!("width"), add!(mul!(c!(2), p!("n_rows")), c!(1))))),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: &[bind("n_rows", BindSource::ChunkRows)],
            note: "width is not carried by the grid (one workgroup per row) and no config key is grounded to it (q_a / kv_a norm widths differ per model), so a leaf with this node stays partial until a model binds it.",
        },
    };
}

mod gather_latent {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("row0", "first row offset"),
        ip("n_rows", "rows"),
        ip("latent", "latent channels"),
        ip("tail", "tail channels (skipped)"),
        ip("cap", "cache capacity (rows)"),
    ];

    const TOTAL: Expr = mul!(p!("n_rows"), p!("latent"));

    const OPERANDS: &[OperandV1] = &[
        opd(
            "cache",
            FORMAT_BF16_FLAT,
            Access::ReadOnly,
            2,
            bf16b!(mul!(p!("cap"), add!(p!("latent"), p!("tail")))),
            "[cap, latent+tail]",
        ),
        opd_null(
            "base_ptr",
            FORMAT_I32_DEVICE_SCALAR,
            Access::ReadOnly,
            4,
            c!(4),
            "nullable device row base",
        ),
        opd(
            "out",
            FORMAT_F32_FLAT,
            Access::WriteOnly,
            4,
            f32b!(TOTAL),
            "[n_rows, latent]",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("cache", 0),
        f_ptr("base_ptr", 8),
        f_ptr("out", 16),
        f_i32("row0", 24, "row0"),
        f_i32("n_rows", 28, "n_rows"),
        f_i32("latent", 32, "latent"),
        f_i32("tail", 36, "tail"),
        f_i32("cap", 40, "cap"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_rows")), "n_rows > 0"),
        rule(Rule::Positive(p!("latent")), "latent > 0"),
        rule(Rule::Positive(p!("cap")), "cap > 0"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_gather_latent_f32",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_decompress",
            source: "kernels/src/latent_attention_decompress.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&TOTAL, 256), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_gather_latent_f32`: cache rows -> F32 plane `[n_rows, latent]`.
    /// Hidden block at 48, `.kernarg_segment_size` 304.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_gather_latent_f32",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.gather_latent.bf16_to_f32"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(48),
        explicit_bytes: 44,
        segment_bytes: 304,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_rows", BindSource::ChunkRows),
        bind("latent", CFG_LATENT),
        bind("tail", CFG_ROPE),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::GATHER,
            codec: "bf16>f32",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(mul!(TOTAL, c!(6))),
            rows: Some("n_rows"),
            admit: ALL_ROWS,
            bind: BIND,
            note: "grid x = n_rows * latent / 256 (an interval); rows = chunk rows.",
        },
    };
}

mod absorb_pack {
    use super::*;

    const PARAMS: &[ParamV1] = &[
        ip("n_heads", "heads"),
        ip("nope", "K_nope width"),
        ip("v", "V width"),
        ip("latent", "latent width"),
    ];

    /// `n_heads * latent * (nope + v)` packed elements.
    const TOTAL: Expr = mul!(p!("n_heads"), p!("latent"), add!(p!("nope"), p!("v")));

    const OPERANDS: &[OperandV1] = &[
        opd(
            "w_t",
            FORMAT_F32_FLAT,
            Access::ReadOnly,
            4,
            f32b!(TOTAL),
            "[latent, H*(nope+v)] column plane",
        ),
        opd(
            "packed",
            FORMAT_BF16_FLAT,
            Access::WriteOnly,
            16,
            bf16b!(TOTAL),
            "private packed layout read by the absorb gemm; 16-byte aligned base",
        ),
    ];

    const FIELDS: &[FieldV1] = &[
        f_ptr("w_t", 0),
        f_ptr("packed", 8),
        f_i32("n_heads", 16, "n_heads"),
        f_i32("nope", 20, "nope"),
        f_i32("v", 24, "v"),
        f_i32("latent", 28, "latent"),
    ];

    const DOMAIN: &[DomainRuleV1] = &[
        rule(Rule::Positive(p!("n_heads")), "n_heads > 0"),
        rule(Rule::AtMost(p!("n_heads"), 65535), "launcher grid limit"),
        rule(
            Rule::MultipleOf(p!("nope"), 8),
            "launcher requires multiples of 8",
        ),
        rule(
            Rule::MultipleOf(p!("v"), 8),
            "launcher requires multiples of 8",
        ),
        rule(
            Rule::MultipleOf(p!("latent"), 8),
            "launcher requires multiples of 8",
        ),
        rule(Rule::Positive(p!("nope")), "nope > 0"),
        rule(Rule::Positive(p!("v")), "v > 0"),
        rule(Rule::Positive(p!("latent")), "latent > 0"),
        rule(Rule::AtMost(TOTAL, I32_MAX), "extent fits i32"),
    ];

    const EXPORTS: &[ExportV1] = &[ExportV1 {
        symbol: "latent_attention_absorb_pack_bf16",
        arch: Arch::Gfx1201,
        object: ObjectRefV1::HipccSource {
            module: "latent_attention_absorb",
            source: "kernels/src/latent_attention_absorb.hip",
        },
        block: [256, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed: 0,
        grid: [Expr::DivCeil(&TOTAL, 256), c!(1), c!(1)],
        consts: &[],
    }];

    /// `latent_attention_absorb_pack_bf16`: F32 column plane -> packed BF16 absorb weight (load time).
    /// Hidden block at 32, `.kernarg_segment_size` 288.
    pub static ABI: KernelAbiV1 = KernelAbiV1 {
        id: "generic_seq.latent_attention_absorb_pack_bf16",
        version: 1,
        wave: 32,
        numeric: NumericContractId("latent_attention.absorb_pack.bf16_rne"),
        params: PARAMS,
        operands: OPERANDS,
        fields: FIELDS,
        hidden: hidden_std!(32),
        explicit_bytes: 32,
        segment_bytes: 288,
        segment_align: 8,
        domain: DOMAIN,
        exports: EXPORTS,
    };

    const BIND: &[BindV1] = &[
        bind("n_heads", CFG_HEADS),
        bind("nope", CFG_NOPE),
        bind("v", CFG_V),
        bind("latent", CFG_LATENT),
    ];

    pub static WORK: WorkSpecV1 = WorkSpecV1 {
        abi: &ABI,
        work: WorkV1 {
            op_class: op_class::GATHER,
            codec: "f32>bf16",
            pipe: Pipe::Memory,
            macs: None,
            flops: None,
            bytes: Some(mul!(TOTAL, c!(6))),
            rows: None,
            admit: ALL_ROWS,
            bind: BIND,
            note: "one-shot load-time pack; grid x is total/256 (an interval) that the config dims must fall in.",
        },
    };
}

/// Every contract of this module.
pub static ABIS: &[&KernelAbiV1] = &[
    &gdn::ABI,
    &conv::ABI,
    &flash::ABI,
    &rda_prepare::ABI,
    &gated_rms::ABI,
    &postmix::ABI,
    &aggregate::ABI,
    &helmert::ABI,
    &broadcast::ABI,
    &mean::ABI,
    &scatter::ABI,
    &gate::ABI,
    &split_kv::ABI,
    &rms_rows::ABI,
    &gather_latent::ABI,
    &absorb_pack::ABI,
];

/// Every work spec of this module (one per contract in [`ABIS`]).
pub static WORK: &[&WorkSpecV1] = &[
    &gdn::WORK,
    &conv::WORK,
    &flash::WORK,
    &rda_prepare::WORK,
    &gated_rms::WORK,
    &postmix::WORK,
    &aggregate::WORK,
    &helmert::WORK,
    &broadcast::WORK,
    &mean::WORK,
    &scatter::WORK,
    &gate::WORK,
    &split_kv::WORK,
    &rms_rows::WORK,
    &gather_latent::WORK,
    &absorb_pack::WORK,
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

    /// Rocprof work-items `(x, y)` of the single export of `abi` for `given` params.
    fn work_items(abi: &KernelAbiV1, given: &[(&str, u64)]) -> (u64, u64) {
        let e = &abi.exports[0];
        let ps = params_of(abi, given);
        let g: Vec<u64> = e
            .grid
            .iter()
            .map(|x| eval(x, abi, e, &ps).expect("grid evaluates"))
            .collect();
        (g[0] * u64::from(e.block[0]), g[1] * u64::from(e.block[1]))
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
    fn raon_gated_delta_grid_inverts_to_40_heads() {
        // Observed (1280, 16) work-items, block 32.
        let (x, y) = work_items(&gdn::ABI, &[("n_heads", 40)]);
        assert_eq!((x, y), (1280, 16));
        assert_eq!(gdn::ABI.exports[0].block[0], 32);
    }

    #[test]
    fn raon_causal_conv_grid_5120_channels_single_chunk() {
        // Observed (5120, 1) / 128: channels in (4992, 5120], one row slice.
        let (x, y) = work_items(&conv::ABI, &[("n_channels", 5120), ("row_chunks", 1)]);
        assert_eq!((x, y), (5120, 1));
        let (x, _) = work_items(&conv::ABI, &[("n_channels", 4993), ("row_chunks", 1)]);
        assert_eq!(x, 5120);
        let (x, _) = work_items(&conv::ABI, &[("n_channels", 4992), ("row_chunks", 1)]);
        assert_eq!(x, 4992);
        // 40 heads * 128 head_dim.
        assert_eq!(40 * 128, 5120);
    }

    #[test]
    fn raon_flash_prefill_grid_and_lds() {
        // Observed (1024, 40) / 128, LDS 44032: 512 rows / 64 = 8 tiles, 40 heads.
        let (x, y) = work_items(
            &flash::ABI,
            &[
                ("n_rows", 512),
                ("n_heads", 40),
                ("wa", 128),
                ("wb", 64),
                ("dv", 128),
            ],
        );
        assert_eq!((x, y), (1024, 40));
        let e = &flash::ABI.exports[0];
        let (dq, dv) = (128u32 + 64, 128u32);
        assert_eq!(e.dynamic_lds, 4 * 16 * (dq + 8) * 2 + dv.min(128) * 36 * 4);
        assert_eq!(e.dynamic_lds, 44032);
    }

    #[test]
    fn raon_rda_prepare_and_gated_rmsnorm_grids() {
        for abi in [&rda_prepare::ABI, &gated_rms::ABI] {
            let (x, y) = work_items(abi, &[("n_heads", 40), ("n_rows", 512), ("head_dim", 128)]);
            assert_eq!((x, y), (5120, 512), "{}", abi.id);
            assert_eq!(abi.exports[0].block, [128, 1, 1]);
        }
        assert_eq!(rda_prepare::ABI.exports[0].group_segment_fixed, 2048);
    }

    #[test]
    fn raon_stream_mix_grids() {
        // postmix (2560, 512)/256: width in (2304, 2560], 512 rows.
        let (x, y) = work_items(&postmix::ABI, &[("width", 2560), ("rows", 512), ("n", 4)]);
        assert_eq!((x, y), (2560, 512));
        // aggregate (131072, 1)/256: 512 rows; LDS 11264 = 1024 static + 2560 * 4.
        let (x, y) = work_items(&aggregate::ABI, &[("rows", 512)]);
        assert_eq!((x, y), (131072, 1));
        let e = &aggregate::ABI.exports[0];
        assert_eq!(e.dynamic_lds + e.group_segment_fixed, 11264);
        assert_eq!(e.dynamic_lds, 2560 * 4);
        assert_eq!(mean::ABI.exports[0].dynamic_lds, 2560 * 4);
    }

    #[test]
    fn raon_work_params_bind_from_the_raon_config_keys() {
        // hidden 2560, 4 streams, 40 heads x 128, rows 512 come from these binds.
        let keys = |w: &WorkSpecV1, p: &str| -> Vec<&'static str> {
            match w.work.bind.iter().find(|b| b.param == p).unwrap().source {
                BindSource::Config(k) => k.to_vec(),
                BindSource::Derived(_) | BindSource::ChunkRows | BindSource::CausalPrefix => vec![],
                _ => panic!("unexpected source"),
            }
        };
        assert!(keys(&postmix::WORK, "n").contains(&"raon_mhc_streams"));
        assert!(keys(&postmix::WORK, "width").contains(&"hidden_size"));
        assert!(keys(&gdn::WORK, "dk").contains(&"rda_config.head_dim"));
        assert!(keys(&gdn::WORK, "dk").contains(&"linear_key_head_dim"));
        assert!(keys(&gdn::WORK, "dv").contains(&"linear_value_head_dim"));
        assert!(keys(&conv::WORK, "ksize").contains(&"rda_config.conv_kernel_size"));
        assert!(keys(&conv::WORK, "ksize").contains(&"linear_conv_kernel_dim"));
        assert!(keys(&conv::WORK, "heads").contains(&"linear_num_value_heads"));
        assert!(keys(&flash::WORK, "wa").contains(&"qk_nope_head_dim"));
        assert!(keys(&flash::WORK, "wb").contains(&"qk_rope_head_dim"));
        assert!(keys(&flash::WORK, "dv").contains(&"v_head_dim"));
        assert!(matches!(
            flash::WORK
                .work
                .bind
                .iter()
                .find(|b| b.param == "prefix")
                .unwrap()
                .source,
            BindSource::CausalPrefix
        ));
        // Raon numbers: 40 heads x 128 = 5120 conv channels; 4 x 2560 stream plane.
        assert_eq!(40 * 128, 5120);
        assert_eq!(4 * 2560, 10240);
    }

    #[test]
    fn op_classes_and_pipes() {
        assert_eq!(gdn::WORK.work.op_class, op_class::RECURRENCE);
        assert_eq!(conv::WORK.work.op_class, op_class::CONV);
        assert_eq!(flash::WORK.work.op_class, op_class::ATTENTION);
        assert_eq!(flash::WORK.work.pipe, Pipe::WmmaBf16);
        assert_eq!(rda_prepare::WORK.work.op_class, op_class::NORM);
        assert_eq!(gated_rms::WORK.work.op_class, op_class::NORM);
        for w in [
            &postmix::WORK,
            &aggregate::WORK,
            &helmert::WORK,
            &broadcast::WORK,
            &mean::WORK,
        ] {
            assert_eq!(w.work.op_class, op_class::STREAM_MIX);
        }
        for w in [
            &scatter::WORK,
            &split_kv::WORK,
            &gather_latent::WORK,
            &absorb_pack::WORK,
        ] {
            assert_eq!(w.work.op_class, op_class::GATHER);
        }
    }
}
