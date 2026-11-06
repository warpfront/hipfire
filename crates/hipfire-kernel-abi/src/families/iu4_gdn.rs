// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contracts of the gfx1201 A4 fused GDN input projection
//! (`gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1` / `..._b1s`): one launch over the QKV weights and
//! the loader's Z fold (Z, beta, alpha) whose QKV row tiles also run `gdn_chunk_prep` in the epilogue
//! (`peacemaker-isa` `iu4_gemm::Epi::QkvzaGdn`, `kernels/iu4_gemm/gdn_epilogue.rs`).
//!
//! Launcher: `Gpu::gemm_qkvza_mq4g256v2_iu4_gdn_prepared` (`crates/rdna-compute/src/gemm.rs`); judge:
//! `crates/rdna-compute/examples/pm_loop/qkvzagdn.rs`. Both encode the 120 explicit kernarg bytes
//! through the contracts below.
//!
//! Evidence the numbers come from:
//!
//! * Generator `iu4_gemm::Spec::kernargs` (`peacemaker-isa/src/kernels/iu4_gemm/spec.rs`, the
//!   `Epi::QkvzaGdn` arm; `GDN_KERNARG_BYTES = 120`): pointers `Wqkv@0 Wz@8 Xq@16 Yz@24 Ybeta@32
//!   Yalpha@40`, by-value words `M0@48 Mz@52 K@56 N@60`, pointers `ConvW@64 ConvState@72 Q@80 K16@88
//!   V@96 X@104`, by-value `QScale@112 Eps@116` (f32 bit patterns), then K1's hidden block from
//!   offset 120 (block counts 120/124/128, group sizes 132/134/136, remainders 138/140/142, global
//!   offsets 160/168/176, grid dims 184, dynamic LDS size 240; segment 376 = 120 + 256).
//! * The committed bundles `kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco` (`_b1`, token
//!   activations) and `..._b1s.hxaco` (`_b1s`, slab activations): `kernels::GEMM_MQ4G256V2_RESIDUAL_
//!   MMQ_IU4_GFX12_B1` / `_B1S`, selected by `gemm_qkvza_mq4g256v2_iu4_gdn_prepared`
//!   (`Int4MmqScratch::int4_mmq_slab_at`). The object metadata is checked by
//!   `crates/rdna-compute/tests/kernel_abi_drift.rs`.
//! * Launch (`gemm.rs`): grid `[ceil((M0 + Mz + 128) / 128), ceil(N / 128), 1]`, block 256, dynamic
//!   LDS 20480.
//!
//! Formats: the token-layout activations are generic_gemm's `block_i4_128.kb_x_n.72b` (record
//! `kb * N + token`), the F32 `[N][M]` outputs mq6_i8's `f32.n_x_m`.
//!
//! The `N <= 104857` bound (32-bit raw-row / q,k,v descriptor sizes) is inferred from `gdn_epilogue.rs`, not hardware-observed.
//!
//! Domain: the Qwen3.5 GDN geometry the epilogue hard-codes (QKV 10240 = 16 + 16 + 48 heads of 128,
//! Z 6144, beta/alpha 48), `K % 256 == 0`, `0 < N`. `QScale` and `Eps` are passed as their f32 bit
//! patterns (`f32::to_bits`), which the contract's integer expressions carry as `U32` params.

use super::generic_gemm::FORMAT_BLOCK_I4_128_KB_N;
use super::mq6_i8::FORMAT_F32_N_M;
use crate::{
    Access, Arch, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind, HiddenV1,
    KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---- formats ---------------------------------------------------------------

/// MQ4G256V2 symmetric weights: rows x K/256 groups x 136 B (`GROUP_BYTES`, `MQ4V2_GROUP_BYTES`).
pub const FORMAT_MQ4G256V2_WEIGHTS: FormatId = FormatId("mq4g256v2.sym.rows_x_groups.136b");
/// The loader's Z fold (`Gpu::iu4_gdn_fold_rows`): Z rows, then one 128-row tile holding beta in
/// rows 0..47 and alpha in rows 64..111 (zero elsewhere), same 136 B groups.
pub const FORMAT_MQ4G256V2_Z_FOLD: FormatId =
    FormatId("mq4g256v2.sym.gdn_z_fold.rows_x_groups.136b");
/// Slab-layout A4 activations: per K128 block `[d 4N][s 4N][qs 0..31: 32N][qs 32..63: 32N]`
/// (`_b1s`, written only by the slab producer twins).
pub const FORMAT_BLOCK_I4_128_SLAB: FormatId = FormatId("block_i4_128.slab.kb_x_planes.72b");
/// Depthwise conv taps `[10240][4]` F32.
pub const FORMAT_F32_CONV_WEIGHT_4: FormatId = FormatId("f32.gdn_conv_weight.channels_x_4");
/// Persistent conv ring `[10240][3]` F32 (`conv_state[ch * 3 + k]` is `x[-1-k]`).
pub const FORMAT_F32_CONV_STATE_3: FormatId = FormatId("f32.gdn_conv_state.channels_x_3");
/// `gdn_chunk_prep` FP16 q/k/v `[N][heads][128]`.
pub const FORMAT_F16_TOKEN_HEADS_128: FormatId = FormatId("f16.gdn_prep.n_x_heads_x_128");
/// Raw F32 QKV rows `[N][10240]` of which only the rows `gdn_chunk_prep_fixup` reads are written
/// (tile positions 0..2 and 125..127 and the last three tokens).
pub const FORMAT_F32_QKV_RAW_ROWS: FormatId = FormatId("f32.gdn_qkv_raw_rows.n_x_10240");

// ---- numeric contracts -------------------------------------------------------

/// Token-activation `_b1`: symmetric MQ4 x A4 fold (`acc = fma(RN(sc * d), float(dot), acc)`, K128
/// order) stored F32 for Z/beta/alpha, QKV tiles turned into `gdn_chunk_prep` FP16 q/k/v (conv1d
/// width 4 + SiLU + per-head RMS norm and `q_scale` for q/k) plus the raw F32 rows the fixup reads.
pub const NUMERIC_QKVZAGDN_B1: NumericContractId =
    NumericContractId("iu4_gemm.qkvzagdn.b1.set_f32.gdn_prep_f16.v1");
/// Slab-activation `_b1s`: the arithmetic of [`NUMERIC_QKVZAGDN_B1`] reading the slab planes.
pub const NUMERIC_QKVZAGDN_B1S: NumericContractId =
    NumericContractId("iu4_gemm.qkvzagdn.b1s.set_f32.gdn_prep_f16.v1");

// ---- geometry -----------------------------------------------------------------

/// QKV rows (16 q + 16 k + 48 v heads of 128).
pub const QKV_ROWS: u64 = 10240;
/// Z rows.
pub const Z_ROWS: u64 = 6144;
/// Beta and alpha rows (each).
pub const BETA_ALPHA_ROWS: u64 = 48;
/// Extra rows of the Z fold (one 128-row tile for beta and alpha).
pub const FOLD_EXTRA_ROWS: u64 = 128;
/// Bytes of the explicit kernarg (`iu4_gemm::spec::GDN_KERNARG_BYTES`).
pub const EXPLICIT_BYTES: u32 = 120;
/// Dynamic LDS the launch passes.
pub const DYNAMIC_LDS: u32 = 20480;

const P_M0: Expr = Expr::Param("M0");
const P_MZ: Expr = Expr::Param("Mz");
const P_MBETA: Expr = Expr::Param("Mbeta");
const P_MALPHA: Expr = Expr::Param("Malpha");
const P_K: Expr = Expr::Param("K");
const P_N: Expr = Expr::Param("N");
const P_QSCALE: Expr = Expr::Param("QScaleBits");
const P_EPS: Expr = Expr::Param("EpsBits");

const I32_MAX: u64 = i32::MAX as u64;
const U32_MAX: u64 = u32::MAX as u64;

/// `M0 x K/256 x 136`.
const WQKV_BYTES: Expr = Expr::Mul(&[P_M0, Expr::DivExact(&P_K, 256), Expr::Const(136)]);
/// `(Mz + 128) x K/256 x 136`.
const FOLD_ROWS: Expr = Expr::Add(&[P_MZ, Expr::Const(FOLD_EXTRA_ROWS)]);
const WZ_BYTES: Expr = Expr::Mul(&[FOLD_ROWS, Expr::DivExact(&P_K, 256), Expr::Const(136)]);
/// `K/128 x N x 72`.
const XQ_BYTES: Expr = Expr::Mul(&[P_N, Expr::DivExact(&P_K, 128), Expr::Const(72)]);
const YZ_BYTES: Expr = Expr::Mul(&[P_N, P_MZ, Expr::Const(4)]);
const YBETA_BYTES: Expr = Expr::Mul(&[P_N, P_MBETA, Expr::Const(4)]);
const YALPHA_BYTES: Expr = Expr::Mul(&[P_N, P_MALPHA, Expr::Const(4)]);
/// `M0 x 4 x 4`.
const CONVW_BYTES: Expr = Expr::Mul(&[P_M0, Expr::Const(16)]);
/// `M0 x 3 x 4`.
const CONVSTATE_BYTES: Expr = Expr::Mul(&[P_M0, Expr::Const(12)]);
/// `N x 16 heads x 128 x 2` (q and k).
const QK_BYTES: Expr = Expr::Mul(&[P_N, Expr::Const(16 * 128 * 2)]);
/// `N x 48 heads x 128 x 2`.
const V_BYTES: Expr = Expr::Mul(&[P_N, Expr::Const(48 * 128 * 2)]);
/// `N x M0 x 4`: the raw-row buffer (`RAWD` num_records `N * 0xa000`).
const X_BYTES: Expr = Expr::Mul(&[P_N, P_M0, Expr::Const(4)]);
/// `N x 48 x 256`: the selected q/k/v descriptor span (`OUTD` num_records, v heads).
const V_DESC_BYTES: Expr = Expr::Mul(&[P_N, Expr::Const(48 * 256)]);

const ROW_TILES: Expr = Expr::DivCeil(&Expr::Add(&[P_M0, FOLD_ROWS]), 128);
const GRID: [Expr; 3] = [ROW_TILES, Expr::DivCeil(&P_N, 128), Expr::Const(1)];
const CONSTS: &[(&str, u64)] = &[("tile_rows", 128), ("tile_tokens", 128)];

const PARAMS: &[ParamV1] = &[
    ParamV1 {
        name: "M0",
        ty: ScalarTy::I32,
        doc: "QKV rows (10240)",
    },
    ParamV1 {
        name: "Mz",
        ty: ScalarTy::I32,
        doc: "Z rows (6144)",
    },
    ParamV1 {
        name: "Mbeta",
        ty: ScalarTy::I32,
        doc: "beta rows (48); not a kernarg, the beta output extent",
    },
    ParamV1 {
        name: "Malpha",
        ty: ScalarTy::I32,
        doc: "alpha rows (48); not a kernarg, the alpha output extent",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction length",
    },
    ParamV1 {
        name: "N",
        ty: ScalarTy::I32,
        doc: "tokens",
    },
    ParamV1 {
        name: "QScaleBits",
        ty: ScalarTy::U32,
        doc: "f32 bit pattern of q_scale (1 / sqrt(head_dim))",
    },
    ParamV1 {
        name: "EpsBits",
        ty: ScalarTy::U32,
        doc: "f32 bit pattern of the RMS-norm epsilon",
    },
];

const fn operand(
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

const NOTE_ALIGN16: &str =
    "alignment 16: gemm_qkvza_mq4g256v2_iu4_gdn_prepared admits only 16-byte aligned buffers here";
const NOTE_ALIGN4: &str = "alignment 4 (F32 words): the launcher checks no base alignment";

macro_rules! operands {
    ($xq_format:expr, $xq_note:expr) => {
        &[
            operand(
                "Wqkv",
                FORMAT_MQ4G256V2_WEIGHTS,
                Access::ReadOnly,
                16,
                WQKV_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "Wz",
                FORMAT_MQ4G256V2_Z_FOLD,
                Access::ReadOnly,
                16,
                WZ_BYTES,
                "alignment 16 per the launcher; the launcher additionally requires the buffer to be exactly (Mz + 128) rows",
            ),
            operand(
                "Xq",
                $xq_format,
                Access::ReadOnly,
                4,
                XQ_BYTES,
                $xq_note,
            ),
            operand(
                "Yz",
                FORMAT_F32_N_M,
                Access::WriteOnly,
                16,
                YZ_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "Ybeta",
                FORMAT_F32_N_M,
                Access::WriteOnly,
                16,
                YBETA_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "Yalpha",
                FORMAT_F32_N_M,
                Access::WriteOnly,
                16,
                YALPHA_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "ConvW",
                FORMAT_F32_CONV_WEIGHT_4,
                Access::ReadOnly,
                4,
                CONVW_BYTES,
                NOTE_ALIGN4,
            ),
            operand(
                "ConvState",
                FORMAT_F32_CONV_STATE_3,
                Access::ReadOnly,
                4,
                CONVSTATE_BYTES,
                "read only by this launch (the first token tile's halo); the ring is advanced by gdn_chunk_prep_fixup. Alignment 4: the launcher checks none",
            ),
            operand(
                "Q",
                FORMAT_F16_TOKEN_HEADS_128,
                Access::WriteOnly,
                16,
                QK_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "K16",
                FORMAT_F16_TOKEN_HEADS_128,
                Access::WriteOnly,
                16,
                QK_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "V",
                FORMAT_F16_TOKEN_HEADS_128,
                Access::WriteOnly,
                16,
                V_BYTES,
                NOTE_ALIGN16,
            ),
            operand(
                "X",
                FORMAT_F32_QKV_RAW_ROWS,
                Access::WriteOnly,
                16,
                X_BYTES,
                "only the raw rows gdn_chunk_prep_fixup reads are written; the rest of the buffer is untouched",
            ),
        ]
    };
}

const OPERANDS_B1: &[OperandV1] = operands!(
    FORMAT_BLOCK_I4_128_KB_N,
    "alignment 4: records start with a float; the launcher checks the live token-layout scratch only"
);
const OPERANDS_B1S: &[OperandV1] = operands!(
    FORMAT_BLOCK_I4_128_SLAB,
    "alignment 4: planes start with the F32 d plane; the launcher checks the live slab scratch only"
);

const fn ptr(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand: name },
    }
}

const fn word(name: &'static str, offset: u32, ty: ScalarTy, value: Expr) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar { ty, value },
    }
}

/// The 120 explicit bytes, in kernarg order.
const FIELDS: &[FieldV1] = &[
    ptr("Wqkv", 0),
    ptr("Wz", 8),
    ptr("Xq", 16),
    ptr("Yz", 24),
    ptr("Ybeta", 32),
    ptr("Yalpha", 40),
    word("M0", 48, ScalarTy::I32, P_M0),
    word("Mz", 52, ScalarTy::I32, P_MZ),
    word("K", 56, ScalarTy::I32, P_K),
    word("N", 60, ScalarTy::I32, P_N),
    ptr("ConvW", 64),
    ptr("ConvState", 72),
    ptr("Q", 80),
    ptr("K16", 88),
    ptr("V", 96),
    ptr("X", 104),
    word("QScale", 112, ScalarTy::U32, P_QSCALE),
    word("Eps", 116, ScalarTy::U32, P_EPS),
];

const fn hidden(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
    HiddenV1 {
        offset,
        width,
        kind,
    }
}

/// K1's hidden block after the 120 explicit bytes (`Spec::kernargs`).
const HIDDEN: &[HiddenV1] = &[
    hidden(120, 4, HiddenKind::BlockCount(crate::Axis::X)),
    hidden(124, 4, HiddenKind::BlockCount(crate::Axis::Y)),
    hidden(128, 4, HiddenKind::BlockCount(crate::Axis::Z)),
    hidden(132, 2, HiddenKind::GroupSize(crate::Axis::X)),
    hidden(134, 2, HiddenKind::GroupSize(crate::Axis::Y)),
    hidden(136, 2, HiddenKind::GroupSize(crate::Axis::Z)),
    hidden(138, 2, HiddenKind::Remainder(crate::Axis::X)),
    hidden(140, 2, HiddenKind::Remainder(crate::Axis::Y)),
    hidden(142, 2, HiddenKind::Remainder(crate::Axis::Z)),
    hidden(160, 8, HiddenKind::GlobalOffset(crate::Axis::X)),
    hidden(168, 8, HiddenKind::GlobalOffset(crate::Axis::Y)),
    hidden(176, 8, HiddenKind::GlobalOffset(crate::Axis::Z)),
    hidden(184, 2, HiddenKind::GridDims),
    hidden(240, 4, HiddenKind::DynamicLdsSize),
];

const DOMAIN: &[DomainRuleV1] = &[
    DomainRuleV1 {
        rule: Rule::Equal(P_M0, Expr::Const(QKV_ROWS)),
        why: "iu4_gdn_fold_active dims[0]: the epilogue hard-codes 16 + 16 + 48 heads of 128",
    },
    DomainRuleV1 {
        rule: Rule::Equal(P_MZ, Expr::Const(Z_ROWS)),
        why: "iu4_gdn_fold_active dims[1]: Z rows of the fold",
    },
    DomainRuleV1 {
        rule: Rule::Equal(P_MBETA, Expr::Const(BETA_ALPHA_ROWS)),
        why: "iu4_gdn_fold_active dims[2]: beta rows of the fold",
    },
    DomainRuleV1 {
        rule: Rule::Equal(P_MALPHA, Expr::Const(BETA_ALPHA_ROWS)),
        why: "iu4_gdn_fold_active dims[3]: alpha rows of the fold",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_K),
        why: "iu4_gdn_fold_active k > 0",
    },
    DomainRuleV1 {
        rule: Rule::MultipleOf(P_K, 256),
        why: "iu4_gdn_fold_active: MQ4 groups span 256 K",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_K, I32_MAX),
        why: "iu4_gdn_fold_active k <= i32::MAX (kernarg word)",
    },
    DomainRuleV1 {
        rule: Rule::Positive(P_N),
        why: "gemm_qkvza_mq4g256v2_iu4_gdn_prepared n > 0",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(P_N, I32_MAX),
        why: "gemm_qkvza_mq4g256v2_iu4_gdn_prepared n <= i32::MAX (kernarg word)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(X_BYTES, U32_MAX),
        why: "gdn_epilogue.rs RAWD num_records = N * 0xa000 is a 32-bit s_mul_i32 (inferred from the generator source, not hardware-observed)",
    },
    DomainRuleV1 {
        rule: Rule::AtMost(V_DESC_BYTES, U32_MAX),
        why:
            "gdn_epilogue.rs OUTD num_records = N * heads * 256 (v: 48 heads) is a 32-bit s_mul_i32",
    },
];

const OBJ_B1: ObjectRefV1 = ObjectRefV1::Embedded {
    // `gemm.rs` g12_iu4_b1_image: kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1
    module: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1",
    artifact: "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco",
    sha256: "49cc18eedd9507b868ecce6b05bd2993688354ed4180e1df456cafe45f34c297",
};
const OBJ_B1S: ObjectRefV1 = ObjectRefV1::Embedded {
    // `gemm.rs` g12_iu4_b1s_image: kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S
    module: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s",
    artifact: "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco",
    sha256: "b351ce70156444a657f31165c704b02faf58c4e81acc3de2d1ff35ffa428eefa",
};

const fn export(symbol: &'static str, object: ObjectRefV1) -> ExportV1 {
    ExportV1 {
        symbol,
        arch: Arch::Gfx1201,
        object,
        block: [256, 1, 1],
        dynamic_lds: DYNAMIC_LDS,
        group_segment_fixed: 0,
        grid: GRID,
        consts: CONSTS,
    }
}

const EXPORTS_B1: &[ExportV1] = &[export(
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1",
    OBJ_B1,
)];
const EXPORTS_B1S: &[ExportV1] = &[export(
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s",
    OBJ_B1S,
)];

/// `_b1`: token-layout activations.
pub static IU4_QKVZAGDN_B1: KernelAbiV1 = KernelAbiV1 {
    id: "iu4_gemm.qkvzagdn.b1",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QKVZAGDN_B1,
    params: PARAMS,
    operands: OPERANDS_B1,
    fields: FIELDS,
    hidden: HIDDEN,
    explicit_bytes: EXPLICIT_BYTES,
    segment_bytes: 376,
    segment_align: 8,
    domain: DOMAIN,
    exports: EXPORTS_B1,
};

/// `_b1s`: slab-layout activations.
pub static IU4_QKVZAGDN_B1S: KernelAbiV1 = KernelAbiV1 {
    id: "iu4_gemm.qkvzagdn.b1s",
    version: 1,
    wave: 32,
    numeric: NUMERIC_QKVZAGDN_B1S,
    params: PARAMS,
    operands: OPERANDS_B1S,
    fields: FIELDS,
    hidden: HIDDEN,
    explicit_bytes: EXPLICIT_BYTES,
    segment_bytes: 376,
    segment_align: 8,
    domain: DOMAIN,
    exports: EXPORTS_B1S,
};

pub const ABIS: &[&KernelAbiV1] = &[&IU4_QKVZAGDN_B1, &IU4_QKVZAGDN_B1S];
