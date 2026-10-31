// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build-time HIP source inventory for admitted Qwen3.5 routes, for the
//! JIT modules of the admitted default Redline programs, and for railgun's
//! own copy kernels. This is a packaging inventory, not a replacement for the
//! runtime's dispatch policy. Every source below is the Rust source
//! expression supplied to `ensure_kernel` (for
//! railgun, the source its lowering will JIT); never infer module names from
//! HIP filenames.

use std::borrow::Cow;

use crate::{sampling, KernelCompiler};

// The Qwen3.5 runtime is feature-gated, but the offline source inventory and
// its byte oracle must also compile under `cargo test --lib kernel_registry`
// without --features deltanet. With the feature enabled these resolve to the
// runtime's exact constants; otherwise include the same literal bodies.
mod kernels {
    pub use crate::kernels::*;
    #[cfg(not(feature = "deltanet"))]
    pub const SIGMOID_SRC: &str = include_str!("../../../kernels/src/sigmoid.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ALPHA_GATE_SRC: &str = include_str!("../../../kernels/src/alpha_gate.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const CONV1D_SILU_SRC: &str = include_str!("../../../kernels/src/conv1d_silu.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const L2_NORM_SRC: &str = include_str!("../../../kernels/src/l2_norm.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const FUSED_QK_L2_NORM_SCALE_SRC: &str =
        include_str!("../../../kernels/src/fused_qk_l2_norm_scale.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const FUSED_SIGMOID_ALPHA_GATE_SRC: &str =
        include_str!("../../../kernels/src/fused_sigmoid_alpha_gate.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const CONV1D_SILU_SPLIT_SRC: &str =
        include_str!("../../../kernels/src/conv1d_silu_split.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const CONV1D_SILU_SPLIT_TREE_SRC: &str =
        include_str!("../../../kernels/src/conv1d_silu_split_tree.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_DELTA_NET_Q8_TREE_SRC: &str =
        include_str!("../../../kernels/src/gated_delta_net_q8_tree.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const SCALE_F32_SRC: &str = include_str!("../../../kernels/src/scale_f32.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_NORM_SRC: &str = include_str!("../../../kernels/src/gated_norm.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_INTERLEAVED_SRC: &str =
        include_str!("../../../kernels/src/rope_partial_interleaved.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_DELTA_NET_Q8_SRC: &str =
        include_str!("../../../kernels/src/gated_delta_net_q8.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_SRC: &str =
        include_str!("../../../kernels/src/rope_partial_halfsplit.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC: &str =
        include_str!("../../../kernels/src/rope_partial_halfsplit_headgrid.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC: &str =
        include_str!("../../../kernels/src/rope_partial_halfsplit_batched.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const QWEN35_FA_PREP_GFX1100_SRC: &str =
        include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const QWEN35_FA_PREP_GFX1151_SRC: &str = concat!(
        "#define HIPFIRE_QWEN35_FA_PREP_KERNEL qwen35_fa_prep_gfx1151\n",
        include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip")
    );
    #[cfg(not(feature = "deltanet"))]
    pub const QWEN35_FA_PREP_GFX1201_SRC: &str = concat!(
        "#define HIPFIRE_QWEN35_FA_PREP_KERNEL qwen35_fa_prep_gfx1201\n",
        include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip")
    );
    /// `railgun::copy::SOURCE` (`crates/railgun/src/copy.rs`) includes the same file.
    pub const RAILGUN_COPY_SRC: &str = include_str!("../../../kernels/src/railgun_copy.hip");
    pub const RAILGUN_FILL_SRC: &str = include_str!("../../../kernels/src/railgun_fill.hip");
}

pub const SUPPORTED_ARCHES: &[&str] = &["gfx1201", "gfx1100", "gfx1151", "gfx906", "gfx942"];

#[derive(Debug, Clone)]
pub struct KernelEntry {
    pub arch: &'static str,
    pub module: &'static str,
    pub symbols: &'static [&'static str],
    pub source: Cow<'static, str>,
    /// Core hipcc arguments, including arch, module and source-dependent flags.
    /// Selected ROCm root/include paths and output/input paths are appended by
    /// the build host; never include a host-specific path in a package index.
    pub flags: Vec<String>,
    pub scheduler_profile: Option<String>,
}

impl KernelEntry {
    pub fn source(&self) -> &str {
        &self.source
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    UnsupportedArch(String),
    UnsupportedModule {
        arch: String,
        module: String,
    },
    /// The header/policy is outside every closed route: the caller keeps
    /// lazy loading for it. Never a partial plan.
    UnsupportedRoute(String),
    /// Two selected images claim one public entry, or one module name
    /// resolves to two different recipes.
    ConflictingPlan(String),
}

fn prepend_kv_slot_desc(body: &str) -> String {
    // Same assembly as attention.rs `ensure_givens4_kernel`.
    format!(
        "{}\n{}",
        kernels::KV_SLOT_DESC_H,
        body.replace("#include \"kv_slot_desc.h\"", "")
    )
}

fn q8_flash_prefill_default_source() -> String {
    // attention.rs:3482-3504 uses NTHREADS=256; the default dispatcher
    // selects BR=8, BC=16 (hipfire-dispatch/src/families/attention.rs:2236-2258).
    // Explicit HIPFIRE_FLASH_PREFILL_BR/BC overrides require separate entries.
    format!(
        "#define BR 8\n#define BC 16\n#define NTHREADS 256\n{}\n{}",
        kernels::KV_SLOT_DESC_H,
        kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC.replace("#include \"kv_slot_desc.h\"", "")
    )
}

fn q8_fa2_gqa_gfx1100_source() -> String {
    // attention.rs gfx11 FA2 GQA prefill on gfx1100 (r3): KT32, f16 Q tile,
    // fill on, -mcumode recipe comment and the gfx1100 define.
    format!(
        "// HIPFIRE_COMPILER_FLAGS: -mcumode\n#define HIPFIRE_FA2_KT 32\n#define HIPFIRE_FA2_Q16 1\n#define HIPFIRE_FA2_FILL 1\n#define HIPFIRE_FA2_GFX1100 1\n{}",
        kernels::ATTENTION_Q8_0_FA2_GQA_GFX11_SRC
    )
}

fn q8_flash_prefill_wmma_gfx11_hd256_source() -> String {
    // attention.rs WMMA flash prefill defaults (no split Q, fixed head_dim,
    // V prefetch on) for head_dim 256 on gfx11.
    format!(
        "#define SPLIT_Q 0\n#define FIXED_HEAD_DIM 256\n#define PREFETCH_V 1\n{}\n{}",
        kernels::KV_SLOT_DESC_H,
        kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC.replace("#include \"kv_slot_desc.h\"", "")
    )
}

fn assemble_asym(body: &str) -> String {
    // The slot descriptor appears in asym3 but not all the other asym bodies.
    let needs_slot = body.contains("#include \"kv_slot_desc.h\"");
    let stripped = body
        .replace("#include \"turbo_common.h\"", "")
        .replace("#include \"givens_common.h\"", "")
        .replace("#include \"kv_slot_desc.h\"", "");
    if needs_slot {
        format!(
            "{}\n{}\n{}\n{}",
            kernels::TURBO_COMMON_H,
            kernels::GIVENS_COMMON_SRC,
            kernels::KV_SLOT_DESC_H,
            stripped
        )
    } else {
        format!(
            "{}\n{}\n{}",
            kernels::TURBO_COMMON_H,
            kernels::GIVENS_COMMON_SRC,
            stripped
        )
    }
}

/// Sources covering the gfx1201 H2 FP8-KV/MQ4v2 and 0.8B Q8-KV/MQ4 routes,
/// plus the union of the daemon's Qwen3.5 `{mq4,mq6,hfq4,hfq6,q8} ×
/// {asym3,q8}` default precompile matrix, and the H2 modules traced on
/// gfx1201, gfx1100 and gfx1151. Other arches advertise this installer
/// inventory plus the default Q8 scalar-flash specialization, but not
/// unobserved model-specific routes; unsupported pairs fail explicitly.
///
/// `extra_flags` must be the same process-config value used by KernelCompiler.
/// Non-default runtime selector flags require a separately admitted registry.
pub fn entries(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let arch: &'static str = SUPPORTED_ARCHES
        .iter()
        .copied()
        .find(|&candidate| candidate == arch)
        .ok_or_else(|| RegistryError::UnsupportedArch(arch.to_owned()))?;
    let mut entries = Vec::new();
    macro_rules! add {
        ($module:literal, $src:expr, [$($symbol:literal),+ $(,)?]) => {
            entries.push(entry(arch, $module, &[$($symbol),+], ($src).into(), extra_flags))
        };
    }

    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        add!(
            "qsa_filter_select",
            kernels::QSA_FILTER_SELECT_SRC,
            [
                "qsa_filter_threshold",
                "qsa_filter_compact",
                "qsa_filter_finish"
            ]
        );
        add!(
            "qsa_bf16_scores",
            kernels::QSA_BF16_SCORES_SRC,
            [
                "indexed_attention_select_scores_rows8_bf16",
                "indexed_attention_select_scores_rows16_bf16",
                "indexed_attention_select_scores_rows32_bf16"
            ]
        );
    }

    // Common Qwen3.5 kernels (historical whole-model precompile set).
    add!("rmsnorm", kernels::RMSNORM_SRC, ["rmsnorm_f32"]);
    add!("add_inplace", kernels::ADD_INPLACE_SRC, ["add_inplace_f32"]);
    add!("mul", kernels::MUL_SRC, ["mul_f32"]);
    add!("silu_mul", kernels::SILU_MUL_SRC, ["silu_mul_f32"]);
    add!("sigmoid", kernels::SIGMOID_SRC, ["sigmoid_f32"]);
    add!("alpha_gate", kernels::ALPHA_GATE_SRC, ["alpha_gate_f32"]);
    add!("conv1d_silu", kernels::CONV1D_SILU_SRC, ["conv1d_silu_f32"]);
    add!("l2_norm", kernels::L2_NORM_SRC, ["l2_norm_f32"]);
    add!(
        "fused_qk_l2_norm_scale",
        kernels::FUSED_QK_L2_NORM_SCALE_SRC,
        ["fused_qk_l2_norm_scale_f32"]
    );
    add!(
        "fused_sigmoid_alpha_gate",
        kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
        ["fused_sigmoid_alpha_gate_f32"]
    );
    add!(
        "conv1d_silu_split",
        kernels::CONV1D_SILU_SPLIT_SRC,
        ["conv1d_silu_split_f32", "conv1d_silu_split_f32_masked",]
    );
    add!(
        "conv1d_silu_split_tree",
        kernels::CONV1D_SILU_SPLIT_TREE_SRC,
        ["conv1d_silu_split_tree_f32"]
    );
    add!(
        "gated_delta_net_q8_tree",
        kernels::GATED_DELTA_NET_Q8_TREE_SRC,
        ["gated_delta_net_q8_tree"]
    );
    add!("sigmoid_mul", kernels::SIGMOID_MUL_SRC, ["sigmoid_mul_f32"]);
    add!("topk_logits", kernels::TOPK_LOGITS_SRC, ["topk_logits_f32"]);
    add!("scale_f32", kernels::SCALE_F32_SRC, ["scale_f32"]);
    add!("gated_norm", kernels::GATED_NORM_SRC, ["gated_norm_f32"]);
    add!(
        "rope_partial_interleaved",
        kernels::ROPE_PARTIAL_INTERLEAVED_SRC,
        ["rope_partial_interleaved_f32"]
    );
    add!(
        "deinterleave",
        kernels::DEINTERLEAVE_SRC,
        ["deinterleave_f32"]
    );
    add!(
        "repeat_interleave_qk",
        kernels::REPEAT_INTERLEAVE_QK_SRC,
        ["repeat_interleave_qk_f32"]
    );
    add!(
        "embedding_q8",
        kernels::EMBEDDING_Q8_SRC,
        ["embedding_q8", "embedding_q8_buf_broadcast"]
    );
    add!(
        "embedding_q8_batched",
        kernels::EMBEDDING_Q8_BATCHED_SRC,
        ["embedding_q8_batched"]
    );
    add!(
        "embedding_hfq4g128",
        kernels::EMBEDDING_HFQ4G128_SRC,
        ["embedding_hfq4g128"]
    );
    add!(
        "embedding_hfq4g128_batched",
        kernels::EMBEDDING_HFQ4G128_BATCHED_SRC,
        ["embedding_hfq4g128_batched"]
    );
    add!(
        "embedding_hfq4g256",
        kernels::EMBEDDING_HFQ4G256_SRC,
        ["embedding_hfq4g256"]
    );
    add!(
        "embedding_hfq4g256_batched",
        kernels::EMBEDDING_HFQ4G256_BATCHED_SRC,
        ["embedding_hfq4g256_batched"]
    );
    add!(
        "gated_delta_net_q8",
        kernels::GATED_DELTA_NET_Q8_SRC,
        [
            "gated_delta_net_q8",
            "gated_delta_net_q8_independent_masked",
        ]
    );
    // DFlash verify tape capture (`Gpu::tape_copy_batched`, tape_copy.rs): the
    // kernel source has no arch guard and the launcher no arch gate
    // (HIPFIRE_VERIFY_TAPE_COPY_BATCH defaults on), so every inventory carries
    // it; module key == symbol, source is the launcher's own constant.
    add!(
        "tape_copy_batched",
        crate::tape_copy::TAPE_COPY_BATCHED_SRC,
        ["tape_copy_batched"]
    );
    // `Gpu::bf16_round_trip_f32` (norm.rs): module `bf16_round_trip`, symbol
    // `bf16_round_trip_f32`, source `kernels::BF16_ROUND_TRIP_SRC`; no arch
    // guard in the kernel or launcher (Qwen4 storage boundaries, dots.ocr).
    add!(
        "bf16_round_trip",
        kernels::BF16_ROUND_TRIP_SRC,
        ["bf16_round_trip_f32"]
    );
    // Lane-table conv of the lane-batched CB DFlash verify (`Gpu::
    // conv1d_silu_split_f32_lanes`, lane_gdn.rs). The kernel has no arch
    // guard; its only caller is admitted by `lane_gdn_batch_admitted`, which
    // requires exact gfx1201, so only that inventory carries it.
    if arch == "gfx1201" {
        add!(
            "conv1d_silu_split_f32_lanes",
            crate::lane_gdn::CONV1D_SILU_SPLIT_LANES_SRC,
            ["conv1d_silu_split_f32_lanes"]
        );
    }

    // Matrix-dependent default precompile modules; each name and source is
    // taken from the same constants as dispatch.rs:4900-5174.
    add!(
        "gemv_hfq6g256",
        kernels::GEMV_HFQ6G256_SRC,
        ["gemv_hfq6g256"]
    );
    add!("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC, ["gemv_mq6g256"]);
    add!("gemm_mq6g256", kernels::GEMM_MQ6G256_SRC, ["gemm_mq6g256"]);
    add!("gemv_q8_0", kernels::GEMV_Q8_0_SRC, ["gemv_q8_0"]);
    add!(
        "gemv_mq4g256",
        kernels::GEMV_MQ4G256_SRC,
        ["gemv_mq4g256", "mq_rotate_x"]
    );
    add!(
        "gemv_hfq4g256_wide",
        kernels::GEMV_HFQ4G256_WIDE_SRC,
        ["gemv_hfq4g256_wide"]
    );
    add!(
        "fused_qkvza_hfq4g256",
        kernels::FUSED_QKVZA_HFQ4G256_SRC,
        ["fused_qkvza_hfq4g256"]
    );
    add!(
        "fused_qkv_hfq4g256",
        kernels::FUSED_QKV_HFQ4G256_SRC,
        ["fused_qkv_hfq4g256"]
    );
    add!(
        "fused_gate_up_hfq4g256",
        kernels::FUSED_GATE_UP_HFQ4G256_SRC,
        ["fused_gate_up_hfq4g256"]
    );
    add!(
        "fused_rmsnorm_mq_rotate",
        kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
        ["fused_rmsnorm_mq_rotate"]
    );
    add!(
        "fused_silu_mul_mq_rotate",
        kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
        ["fused_silu_mul_mq_rotate"]
    );
    match arch {
        "gfx1100" => add!(
            "gemv_hfq4g256_rdna3",
            kernels::GEMV_HFQ4G256_GFX1100_SRC,
            ["gemv_hfq4g256"]
        ),
        _ => add!(
            "gemv_hfq4g256",
            kernels::GEMV_HFQ4G256_SRC,
            ["gemv_hfq4g256"]
        ),
    }
    if arch == "gfx1201" {
        add!(
            "fused_gate_up_mq4g256v2_k5120_gfx1201",
            kernels::fused_gate_up_mq4g256v2_k5120_gfx1201_src(),
            ["fused_gate_up_mq4g256v2_k5120_gfx1201"]
        );
    }
    if arch == "gfx1100" {
        add!(
            "fused_gate_up_mq4g256v2_k5120_gfx1100",
            kernels::fused_gate_up_mq4g256v2_k5120_gfx1100_src(),
            ["fused_gate_up_mq4g256v2_k5120_gfx1100"]
        );
        // Default gfx1100 precompile branches (dispatch.rs:4944-4962,5304-5308).
        add!(
            "fused_qkvza_hfq4g256_k2048_gfx1100",
            kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC,
            ["fused_qkvza_hfq4g256_k2048"]
        );
        add!(
            "fused_gate_up_hfq4g256_stage_x32_gfx1100",
            kernels::FUSED_GATE_UP_HFQ4G256_STAGE_X32_GFX1100_SRC,
            ["fused_gate_up_hfq4g256_stage_x32_gfx1100"]
        );
        add!(
            "kv_cache_write_asym3_q8_pair_gfx1100",
            assemble_asym(kernels::KV_CACHE_WRITE_ASYM3_Q8_PAIR_GFX1100_SRC),
            ["kv_cache_write_asym3_q8_pair_gfx1100"]
        );
        // Default-on int8 small-batch verify GEMV (`gemv_q8x.rs`): the
        // activation quantizer and every v4/v3 batch x residual variant, each
        // module with its `_multi` twin, so the pack carries them precompiled.
        add!(
            "mq4v2_q8x_quant",
            kernels::GEMV_MQ4G256V2_Q8X_QUANT_SRC,
            ["mq4v2_q8x_quant"]
        );
        add!(
            "mq4v2_q8x_v4_b2",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B2_SRC,
            ["mq4v2_q8x_v4_b2", "mq4v2_q8x_v4_b2_multi"]
        );
        add!(
            "mq4v2_q8x_v4_b2_res",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B2_RES_SRC,
            ["mq4v2_q8x_v4_b2_res", "mq4v2_q8x_v4_b2_res_multi"]
        );
        add!(
            "mq4v2_q8x_v4_b3",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B3_SRC,
            ["mq4v2_q8x_v4_b3", "mq4v2_q8x_v4_b3_multi"]
        );
        add!(
            "mq4v2_q8x_v4_b3_res",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B3_RES_SRC,
            ["mq4v2_q8x_v4_b3_res", "mq4v2_q8x_v4_b3_res_multi"]
        );
        add!(
            "mq4v2_q8x_v4_b4",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B4_SRC,
            ["mq4v2_q8x_v4_b4", "mq4v2_q8x_v4_b4_multi"]
        );
        add!(
            "mq4v2_q8x_v4_b4_res",
            kernels::GEMV_MQ4G256V2_Q8X_V4_B4_RES_SRC,
            ["mq4v2_q8x_v4_b4_res", "mq4v2_q8x_v4_b4_res_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b2",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B2_SRC,
            ["mq4v2_q8x_v3_b2", "mq4v2_q8x_v3_b2_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b2_res",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B2_RES_SRC,
            ["mq4v2_q8x_v3_b2_res", "mq4v2_q8x_v3_b2_res_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b3",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B3_SRC,
            ["mq4v2_q8x_v3_b3", "mq4v2_q8x_v3_b3_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b3_res",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B3_RES_SRC,
            ["mq4v2_q8x_v3_b3_res", "mq4v2_q8x_v3_b3_res_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b4",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B4_SRC,
            ["mq4v2_q8x_v3_b4", "mq4v2_q8x_v3_b4_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b4_res",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B4_RES_SRC,
            ["mq4v2_q8x_v3_b4_res", "mq4v2_q8x_v3_b4_res_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b5",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B5_SRC,
            ["mq4v2_q8x_v3_b5", "mq4v2_q8x_v3_b5_multi"]
        );
        add!(
            "mq4v2_q8x_v3_b5_res",
            kernels::GEMV_MQ4G256V2_Q8X_V3_B5_RES_SRC,
            ["mq4v2_q8x_v3_b5_res", "mq4v2_q8x_v3_b5_res_multi"]
        );
    }
    if matches!(arch, "gfx906" | "gfx942") {
        add!(
            "fused_qkvza_hfq4g256_wave64",
            kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC,
            ["fused_qkvza_hfq4g256_wave64"]
        );
        add!(
            "fused_qkv_hfq4g256_wave64",
            kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC,
            ["fused_qkv_hfq4g256_wave64"]
        );
        add!(
            "fused_gate_up_hfq4g256_wave64",
            kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC,
            ["fused_gate_up_hfq4g256_wave64"]
        );
        add!(
            "gemv_hfq4g256_moe_gate_up_indexed_wave64",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC,
            ["gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"]
        );
        add!(
            "gemv_hfq4g256_moe_down_indexed_wave64",
            kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC,
            ["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64"]
        );
        add!(
            "gemm_qkvza_hfq4g256_wave64",
            kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC,
            ["gemm_qkvza_hfq4g256_wave64"]
        );
        add!(
            "gemm_qkv_hfq4g256_wave64",
            kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC,
            ["gemm_qkv_hfq4g256_wave64"]
        );
        add!(
            "gemm_hfq4g256_wave64",
            kernels::GEMM_HFQ4G256_WAVE64_SRC,
            ["gemm_hfq4g256_wave64"]
        );
        add!(
            "gemm_hfq4g256_residual_wave64",
            kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC,
            ["gemm_hfq4g256_residual_wave64"]
        );
        add!(
            "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC,
            ["gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64"]
        );
        add!(
            "gemv_hfq4g256_moe_down_indexed_batched_wave64",
            kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC,
            ["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64"]
        );
    }

    // KV paths: precompile's asym3/q8 assembly mirrors the runtime header
    // stripping; FP8 substitutions are only admitted for observed gfx1201.
    add!(
        "kv_cache_write_asym_k_givens3",
        assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC),
        ["kv_cache_write_asym_k_givens3"]
    );
    add!(
        "kv_cache_write_asym_k_givens3_batched",
        assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC),
        ["kv_cache_write_asym_k_givens3_batched"]
    );
    add!(
        "attention_flash_asym3_tile",
        assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_SRC),
        ["attention_flash_asym3_tile"]
    );
    add!(
        "attention_flash_asym3_tile_batched",
        assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC),
        ["attention_flash_asym3_tile_batched"]
    );
    add!(
        "attention_flash_asym_reduce_batched",
        kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC,
        ["attention_flash_asym_reduce_batched"]
    );
    add!(
        "kv_cache_write_q8_0",
        kernels::KV_CACHE_WRITE_Q8_0_SRC,
        ["kv_cache_write_q8_0", "kv_cache_write_q8_0_ring",]
    );
    add!(
        "attention_q8_0_kv",
        kernels::ATTENTION_Q8_0_KV_SRC,
        ["attention_q8_0_kv"]
    );
    add!(
        "attention_q8_0_kv_batched",
        prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_KV_BATCHED_SRC),
        ["attention_q8_0_kv_batched"]
    );
    add!(
        "attention_q8_0_kv_independent_masked_windowed",
        prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_KV_BATCHED_SRC),
        ["attention_q8_0_kv_independent_masked_windowed"]
    );
    add!(
        "attention_q8_0_flash_prefill",
        prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC),
        ["attention_q8_0_flash_prefill"]
    );
    // The installer-only unspecialized module above cannot satisfy this
    // runtime's default scalar-prefill module or its BR/BC-specialized source.
    add!(
        "attention_q8_0_flash_prefill_br8_bc16",
        q8_flash_prefill_default_source(),
        ["attention_q8_0_flash_prefill"]
    );
    add!(
        "kv_cache_write_q8_0_batched",
        prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC),
        ["kv_cache_write_q8_0_batched"]
    );
    add!(
        "kv_cache_write_q8_0_independent",
        prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC),
        ["kv_cache_write_q8_0_independent"]
    );
    add!(
        "kv_cache_write_q8_0_independent_masked",
        prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC),
        ["kv_cache_write_q8_0_independent_masked"]
    );
    add!(
        "attention_flash_q8_0_tile",
        kernels::ATTENTION_FLASH_Q8_0_TILE_SRC,
        ["attention_flash_q8_0_tile"]
    );
    add!(
        "attention_flash_q8_0_tile_hd64",
        kernels::ATTENTION_FLASH_Q8_0_TILE_HD64_SRC,
        ["attention_flash_q8_0_tile_hd64"]
    );
    add!(
        "attention_flash_q8_0_reduce",
        kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
        ["attention_flash_q8_0_reduce"]
    );
    // Gemma 4 decode twins of the two modules above (attention.rs
    // `attention_flash_q8_0_windowed_impl` with `gemma_tiles`).
    add!(
        "attention_flash_q8_0_tile_swa",
        kernels::ATTENTION_FLASH_Q8_0_TILE_SWA_SRC,
        ["attention_flash_q8_0_tile_swa"]
    );
    add!(
        "attention_flash_q8_0_reduce_skip",
        kernels::ATTENTION_FLASH_Q8_0_REDUCE_SKIP_SRC,
        ["attention_flash_q8_0_reduce_skip"]
    );
    // Multi-slot (slot-descriptor) launches of the q8 and asym3 KV routes run
    // separately named `*_paged` modules (attention.rs `ensure_kv_slot_kernel`,
    // `ensure_givens4_kv_slot_kernel`, `attention_q8_0_flash_prefill_wmma_slots`).
    add!(
        "kv_cache_write_q8_0_batched_paged",
        kernels::kv_slot_desc_paged_source(
            kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC,
            "kv_cache_write_q8_0_batched",
            "kv_cache_write_q8_0_batched_paged"
        ),
        ["kv_cache_write_q8_0_batched_paged"]
    );
    add!(
        "attention_q8_0_kv_batched_paged",
        kernels::kv_slot_desc_paged_source(
            kernels::ATTENTION_Q8_0_KV_BATCHED_SRC,
            "attention_q8_0_kv_batched",
            "attention_q8_0_kv_batched_paged"
        ),
        ["attention_q8_0_kv_batched_paged"]
    );
    add!(
        "attention_flash_q8_0_tile_batched_paged",
        kernels::kv_slot_givens4_paged_source(
            kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC,
            "attention_flash_q8_0_tile_batched",
            "attention_flash_q8_0_tile_batched_paged"
        ),
        ["attention_flash_q8_0_tile_batched_paged"]
    );
    add!(
        "kv_cache_write_asym_k_givens3_batched_paged",
        kernels::kv_slot_givens4_paged_source(
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC,
            "kv_cache_write_asym_k_givens3_batched",
            "kv_cache_write_asym_k_givens3_batched_paged"
        ),
        ["kv_cache_write_asym_k_givens3_batched_paged"]
    );
    add!(
        "attention_flash_asym3_tile_batched_paged",
        kernels::kv_slot_givens4_paged_source(
            kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC,
            "attention_flash_asym3_tile_batched",
            "attention_flash_asym3_tile_batched_paged"
        ),
        ["attention_flash_asym3_tile_batched_paged"]
    );
    if matches!(arch, "gfx1100" | "gfx1151") {
        // gfx11 admits the multi-slot WMMA prefill tile by default
        // (forward_slots `q8_flash_prefill_wmma_eligible`); Qwen3.5 head_dim
        // 256 with the default SPLIT_Q=0, fixed head dim and V prefetch.
        add!(
            "attention_q8_0_flash_prefill_wmma_gfx11_hd256_paged",
            format!(
                "#define SPLIT_Q 0\n#define FIXED_HEAD_DIM 256\n#define PREFETCH_V 1\n{}",
                kernels::kv_slot_desc_paged_source(
                    kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC,
                    "attention_q8_0_flash_prefill_wmma",
                    "attention_q8_0_flash_prefill_wmma_paged"
                )
            ),
            ["attention_q8_0_flash_prefill_wmma_paged"]
        );
    }
    add!(
        "sample_top_p_parallel",
        sampling::sample_top_p_parallel_src(),
        [
            "sample_apply_repeat_penalty",
            "sample_topk_partial",
            "sample_topk_finalize"
        ]
    );
    add!(
        "sample_top_p_parallel_w64",
        sampling::sample_top_p_parallel_w64_src(),
        [
            "sample_apply_repeat_penalty_w64",
            "sample_topk_partial_w64",
            "sample_topk_finalize_w64"
        ]
    );
    add!(
        "sample_top_p_parallel_fast21",
        sampling::sample_top_p_parallel_fast_src(21, "fast21"),
        [
            "sample_apply_repeat_penalty_fast21",
            "sample_topk_partial_fast21",
            "sample_topk_finalize_fast21"
        ]
    );
    add!(
        "sample_top_p_parallel_fast65",
        sampling::sample_top_p_parallel_fast_src(65, "fast65"),
        [
            "sample_apply_repeat_penalty_fast65",
            "sample_topk_partial_fast65",
            "sample_topk_finalize_fast65"
        ]
    );
    // Qwen4 logit-policy penalty prepass (AR greedy rows, sampled-MTP verify
    // rows; `Gpu::apply_penalty_table`).
    add!(
        "logit_penalty_table",
        kernels::LOGIT_PENALTY_TABLE_SRC,
        ["logit_penalty_table_rows"]
    );

    if arch == "gfx1201" {
        // H2 and small-model first-token JIT: use the actual module key, even
        // when it differs from the kernel symbol or the source filename.
        add!(
            "attention_flash_fp8_e4m3_tile",
            kernels::ATTENTION_FLASH_FP8_E4M3_TILE_SRC,
            ["attention_flash_fp8_e4m3_tile"]
        );
        add!(
            "attention_fp8_e4m3_kv_batched",
            prepend_kv_slot_desc(kernels::ATTENTION_FP8_E4M3_KV_BATCHED_SRC),
            ["attention_fp8_e4m3_kv_batched"]
        );
        add!(
            "conv1d_silu_split_qknorm_b256",
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
            ["conv1d_silu_split_qknorm_b256"]
        );
        add!(
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            ["convert_f32_to_f16", "gemm_hfq4g256_residual_fp16",]
        );
        add!(
            "fused_gate_up_hfq4g256_mq4v2",
            kernels::FUSED_GATE_UP_MQ4G256V2_SRC,
            ["fused_gate_up_mq4g256v2"]
        );
        add!(
            "fused_qkv_hfq4g256_mq4v2",
            kernels::FUSED_QKV_MQ4G256V2_SRC,
            ["fused_qkv_mq4g256v2"]
        );
        add!(
            "fused_qkvza_hfq4g256_mq4v2",
            kernels::FUSED_QKVZA_MQ4G256V2_SRC,
            ["fused_qkvza_mq4g256v2"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            ["fused_rmsnorm_mq_rotate_awq"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_g12dec",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_G12DEC_SRC,
            ["fused_rmsnorm_mq_rotate_awq_g12dec"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            ["fused_silu_mul_mq_rotate_awq"]
        );
        add!(
            "gated_delta_net_q8_fast",
            kernels::GATED_DELTA_NET_Q8_FAST_SRC,
            [
                "gated_delta_net_q8_fast",
                "gated_delta_net_q8_fast_independent_masked",
            ]
        );
        add!(
            "gdn_pre_batched_gfx1201",
            include_str!("../../../kernels/src/gdn_pre_batched.gfx1201.hip"),
            ["gdn_pre_batched_gfx1201"]
        );
        add!(
            "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC,
            [
                "gemm_gate_up_mq4g256v2_wmma_gfx12",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_ldsstage",
            ]
        );
        add!(
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC,
            [
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w4",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w8",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w4",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w8",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4_k32",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8_k32",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w4_k32",
                "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w8_k32"
            ]
        );
        add!(
            "gemm_mq4g256v2_overwrite_wmma_gfx12",
            kernels::GEMM_MQ4G256V2_OVERWRITE_WMMA_GFX12_SRC,
            ["gemm_mq4g256v2_overwrite_wmma_gfx12"]
        );
        add!(
            "gemm_mq4g256v2_overwrite_wmma_gfx12_vt",
            kernels::GEMM_MQ4G256V2_OVERWRITE_WMMA_GFX12_VT_SRC,
            [
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt2w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt2w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt3w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt3w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w4_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w8_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt8w4_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt8w8_k32"
            ]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_gfx12_mq4v2",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC,
            [
                "gemm_mq4g256v2_residual_wmma_gfx12",
                "gemm_mq4g256v2_residual_wmma_gfx12_ldsstage",
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_wmma_gfx12_vt",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC,
            [
                "gemm_mq4g256v2_residual_wmma_gfx12_vt2w4",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt2w8",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt3w4",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt3w8",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4_k32",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8_k32",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt8w4_k32",
                "gemm_mq4g256v2_residual_wmma_gfx12_vt8w8_k32"
            ]
        );
        add!(
            "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC,
            ["gemm_qkv_mq4g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC,
            [
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w4",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w8",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w4",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w8",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4_k32",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8_k32",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w4_k32",
                "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w8_k32"
            ]
        );
        add!(
            "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC,
            ["gemm_qkvza_mq4g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC,
            [
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w4",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w8",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w4",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w8",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4_k32",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8_k32",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w4_k32",
                "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w8_k32"
            ]
        );
        add!(
            "gemv_hfq4g256_multirow_default_mq4v2",
            kernels::GEMV_MQ4G256V2_MULTIROW_SRC,
            [
                "gemv_mq4g256v2_multirow_r2",
                "gemv_mq4g256v2_multirow_r4",
                "gemv_mq4g256v2_multirow_r8"
            ]
        );
        add!(
            "gemv_hfq4g256_residual_mq4v2",
            kernels::GEMV_MQ4G256V2_RESIDUAL_SRC,
            ["gemv_mq4g256v2_residual"]
        );
        add!(
            "gemv_mq4g256v2_mq4v2",
            kernels::GEMV_MQ4G256V2_SRC,
            ["gemv_mq4g256v2"]
        );
        add!(
            "kv_cache_write_fp8_e4m3",
            kernels::KV_CACHE_WRITE_FP8_E4M3_SRC,
            ["kv_cache_write_fp8_e4m3"]
        );
        add!(
            "kv_cache_write_fp8_e4m3_batched",
            prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_FP8_E4M3_BATCHED_SRC),
            ["kv_cache_write_fp8_e4m3_batched"]
        );
        add!("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, ["mq_rotate_x"]);
        add!(
            "qwen35_fa_prep_batched_gfx1201",
            include_str!("../../../kernels/src/qwen35_fa_prep_batched.gfx1201.hip"),
            ["qwen35_fa_prep_batched_gfx1201"]
        );
        add!("rmsnorm_f32", kernels::RMSNORM_SRC, ["rmsnorm_f32"]);
        add!(
            "rmsnorm_f32_rowsplit",
            kernels::RMSNORM_ROWSPLIT_SRC,
            ["rmsnorm_f32_rowsplit"]
        );
        add!(
            "rope_partial_halfsplit",
            kernels::ROPE_PARTIAL_HALFSPLIT_SRC,
            ["rope_partial_halfsplit_f32"]
        );
        add!(
            "rope_partial_halfsplit_f32_headgrid",
            kernels::ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC,
            ["rope_partial_halfsplit_f32_headgrid"]
        );
        add!(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            ["rotate_x_mq_awq"]
        );
        add!(
            "deinterleave_q_rmsnorm_f32_batched",
            kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC,
            ["deinterleave_q_rmsnorm_f32_batched"]
        );
        add!(
            "fused_gate_up_hfq4g256_k1024_gfx1201",
            kernels::FUSED_GATE_UP_HFQ4G256_K1024_GFX1201_SRC,
            ["fused_gate_up_hfq4g256_k1024_gfx1201"]
        );
        add!(
            "gemm_gate_up_hfq4g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC,
            [
                "gemm_gate_up_hfq4g256_wmma_gfx12",
                "gemm_gate_up_hfq4g256_wmma_gfx12_ldsstage",
            ]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC,
            [
                "gemm_hfq4g256_residual_wmma_gfx12",
                "gemm_hfq4g256_residual_wmma_gfx12_ldsstage",
            ]
        );
        add!(
            "gemm_qkv_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC,
            ["gemm_qkv_hfq4g256_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_hfq4g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC,
            ["gemm_qkvza_hfq4g256_wmma_gfx12"]
        );
        add!(
            "gemv_hfq4g256_residual",
            kernels::GEMV_HFQ4G256_RESIDUAL_SRC,
            ["gemv_hfq4g256_residual"]
        );
        add!(
            "rope_partial_halfsplit_batched",
            kernels::ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC,
            ["rope_partial_halfsplit_batched_f32"]
        );
        // H2 (Qwen3.8-27B MQ4v2, fp8 KV) decode, observed JIT-compiled by
        // `hipfire run` greedy AR/MTP/DFlash on the installer inventory above.
        add!(
            "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
            kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC,
            [
                "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
                "attention_flash_fp8_e4m3_tile_gqa8_gfx1201"
            ]
        );
        add!(
            "attention_flash_reduce_dsplit_gfx1201",
            kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC,
            ["attention_flash_reduce_dsplit_gfx1201"]
        );
        add!(
            "dflash_state_bulk_copy_gfx1100",
            crate::dflash_state_copy::DFLASH_STATE_BULK_COPY_GFX1100_SRC,
            ["dflash_state_bulk_copy_gfx1100"]
        );
        add!(
            "gated_delta_net_q8_compact3_b2",
            kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC,
            ["gated_delta_net_q8_compact3_b2"]
        );
        add!(
            "gated_norm_mq_rotate_awq_k6144_gfx1201",
            kernels::gated_norm_mq_rotate_awq_k6144_gfx1201_src(),
            ["gated_norm_mq_rotate_awq_k6144_gfx1201"]
        );
        add!(
            "qwen36_27b_fa_prep_gfx1201",
            kernels::qwen36_27b_fa_prep_gfx1201_src(),
            ["qwen36_27b_fa_prep_gfx1201"]
        );
        // Qwen3.6-27B CbVmm exact route (forward_slots/vmm/exact.rs) and the
        // cross-request batched MTP verify (verify_twins.rs), observed
        // JIT-compiled on the shipped gfx1201 pack (v0.4.4 27B AR/MTP). Each
        // source is the launcher's own module assembly (the shared helpers
        // below), never a divergent copy. gfx1201 only: the VMM fp8 KV write
        // and the GQA fp8 flash tile `#error` off gfx1201, and `ensure_rows_twin`
        // / `ensure_seg_twin` refuse every other arch, so no gfx1100/gfx1151
        // launcher can load any of these modules.
        {
            use crate::attention::{
                kv_slot_desc_vmm_source, vmm_rows_source, KV_CACHE_WRITE_FP8_E4M3_VMM_SRC,
            };
            let recipe_err = |e: hip_bridge::HipError| {
                RegistryError::ConflictingPlan(format!("vmm exact route source: {e}"))
            };
            // attention.rs `kv_cache_write_batched_vmm` (Fp8E4M3 arm).
            add!(
                "kv_cache_write_fp8_e4m3_batched_vmm",
                kv_slot_desc_vmm_source(
                    KV_CACHE_WRITE_FP8_E4M3_VMM_SRC,
                    "kv_cache_write_fp8_e4m3_batched_vmm",
                    "kv_cache_write_fp8_e4m3_batched_vmm"
                ),
                ["kv_cache_write_fp8_e4m3_batched_vmm"]
            );
            // attention.rs `attention_flash_decode_vmm` (Fp8GqaGfx1201 route):
            // the define drops the gqa8 entry, so each module exports one kernel.
            add!(
                "attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm",
                vmm_rows_source(
                    kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC,
                    "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
                    "VMM_ROWS_FP8_GQA_TILE"
                )
                .map_err(recipe_err)?,
                ["attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm"]
            );
            add!(
                "attention_flash_reduce_dsplit_gfx1201_vmm",
                vmm_rows_source(
                    kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC,
                    "attention_flash_reduce_dsplit_gfx1201",
                    "VMM_ROWS_DSPLIT_REDUCE"
                )
                .map_err(recipe_err)?,
                ["attention_flash_reduce_dsplit_gfx1201_vmm"]
            );
            // A3B (H16/KV2) lane-batched FullAttention on the exact route
            // (attention.rs `attention_flash_decode_vmm{,_gated_mq_rotate}`,
            // rows_batched.rs `qwen35_a3b_fa_prep_rows`): the group-8 tile
            // twin, the fused reduce + gate + rotate epilogue twin and the
            // 16Q/2K FA prep rows twin.
            add!(
                "attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm",
                crate::attention::vmm_rows_source_from(
                    crate::attention::ATTENTION_FLASH_DECODE_VMM_ROWS_A3B_SRC,
                    kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC,
                    "attention_flash_fp8_e4m3_tile_gqa8_gfx1201",
                    "VMM_ROWS_FP8_GQA8_TILE"
                )
                .map_err(recipe_err)?,
                ["attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm"]
            );
            add!(
                "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201_vmm",
                crate::attention::vmm_rows_source_from(
                    crate::attention::ATTENTION_FLASH_DECODE_VMM_ROWS_A3B_SRC,
                    kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1201_SRC,
                    "HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL",
                    "VMM_ROWS_ATTN_REDUCE_GATED_MQ"
                )
                .map_err(recipe_err)?,
                ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201_vmm"]
            );
            add!(
                "qwen35_fa_prep_gfx1201_rows",
                crate::attention::rows_batched::rows_twin_module_source(
                    crate::attention::rows_batched::QWEN35_A3B_FA_PREP_ROWS
                )
                .map_err(recipe_err)?,
                ["qwen35_fa_prep_gfx1201_rows"]
            );
            // rows_batched.rs twins (`rows_twin_module_source`).
            for &(module, symbols) in QWEN36_27B_GFX1201_ROWS_TWINS_HIP
                .iter()
                .chain(A3B_GFX1201_DN_ROWS_TWINS_HIP)
            {
                entries.push(entry(
                    arch,
                    module,
                    gfx1201_twin_exports(module, symbols),
                    crate::attention::rows_batched::rows_twin_module_source(module)
                        .map_err(recipe_err)?
                        .into(),
                    extra_flags,
                ));
            }
            // verify_twins.rs segment twin (`seg_twin_module_source`).
            for &(module, symbols) in QWEN36_27B_GFX1201_SPEC_CB_HIP {
                entries.push(entry(
                    arch,
                    module,
                    gfx1201_twin_exports(module, symbols),
                    crate::verify_twins::seg_twin_module_source(module)
                        .map_err(recipe_err)?
                        .into(),
                    extra_flags,
                ));
            }
        }
    }
    // Modules the shared runtime launchers request on every arch the pack
    // ships: the plain Q8_0 GEMV (K <= 1536), and the LFM2 decode path's
    // generic RoPE, gated short-conv and sampler kernels. A host without hipcc
    // can only run what the pack carries, so each is packaged for all three
    // (`pack_carries_every_shared_launcher_module_on_every_arch`).
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!(
            "gemv_q8_0_wide",
            kernels::GEMV_Q8_0_WIDE_SRC,
            ["gemv_q8_0_wide"]
        );
        add!("rope", kernels::ROPE_SRC, ["rope_f32"]);
        add!(
            "conv1d_gated_decode",
            kernels::CONV1D_GATED_DECODE_SRC,
            ["conv1d_gated_decode_f32"]
        );
        add!("sample_top_p", kernels::SAMPLE_TOP_P_SRC, ["sample_top_p"]);
        add!(
            "sample_rows_pf",
            kernels::SAMPLE_ROWS_PF_SRC,
            ["sample_rows_pf_f32"]
        );
    }
    // H2 (Qwen3.8-27B MQ4v2) prefill, MTP and DFlash modules JIT-compiled by
    // `hipfire serve`/`run` beyond the inventory above: gfx1201 fp8 KV, and
    // gfx1100/gfx1151 Q8 KV. Symbols are every kernel each object defines.
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!(
            "attention_dflash_sliding_f32",
            kernels::ATTENTION_DFLASH_SLIDING_SRC,
            ["attention_dflash_sliding_f32"]
        );
        add!(
            "deinterleave_batched",
            kernels::DEINTERLEAVE_BATCHED_SRC,
            ["deinterleave_f32_batched"]
        );
        add!(
            "dynamic_conv_f32",
            kernels::DYNAMIC_CONV_F32_SRC,
            ["dynamic_causal_conv_f32", "dynamic_conv_f32"]
        );
        add!(
            "fused_qk_l2_norm_scale_interleave_f32_batched",
            kernels::FUSED_QK_L2_NORM_SCALE_INTERLEAVE_F32_BATCHED_SRC,
            ["fused_qk_l2_norm_scale_interleave_f32_batched"]
        );
        add!(
            "gemm_hfq4g256",
            kernels::GEMM_HFQ4G256_SRC,
            ["gemm_hfq4g256"]
        );
        add!(
            "rope_batched",
            kernels::ROPE_BATCHED_SRC,
            ["rope_batched_f32"]
        );
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!(
            "add",
            kernels::ADD_SRC,
            ["add_f32", "broadcast_add_rows_f32"]
        );
        add!(
            "argmax_token_chain",
            kernels::ARGMAX_TOKEN_CHAIN_SRC,
            ["argmax_token_chain_f32"]
        );
        add!(
            "gdn_chunk_kkt_solve",
            kernels::GDN_CHUNK_KKT_SOLVE_SRC,
            ["gdn_chunk_kkt_solve"]
        );
        add!(
            "gemv_hfq4g256_multirow_default",
            kernels::GEMV_HFQ4G256_MULTIROW_SRC,
            [
                "gemv_hfq4g256_multirow_r2",
                "gemv_hfq4g256_multirow_r4",
                "gemv_hfq4g256_multirow_r8",
                "gemv_hfq4g256_multirow_r2_x4",
                "gemv_hfq4g256_multirow_r2_x8"
            ]
        );
        add!(
            "greedy_accept",
            kernels::GREEDY_ACCEPT_SRC,
            ["greedy_accept_from_argmax_i32"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "argmax_f32_batched",
            kernels::ARGMAX_BATCHED_SRC,
            ["argmax_f32_batched"]
        );
        add!(
            "attention_q8_0_flash_prefill_wmma_gfx11_hd256",
            q8_flash_prefill_wmma_gfx11_hd256_source(),
            ["attention_q8_0_flash_prefill_wmma"]
        );
        add!(
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            ["convert_f32_to_f16", "gemm_hfq4g256_residual_fp16"]
        );
        add!(
            "fused_gate_up_hfq4g256_mq4v2",
            kernels::FUSED_GATE_UP_MQ4G256V2_SRC,
            ["fused_gate_up_mq4g256v2"]
        );
        add!(
            "fused_qkv_hfq4g256_mq4v2",
            kernels::FUSED_QKV_MQ4G256V2_SRC,
            ["fused_qkv_mq4g256v2"]
        );
        add!(
            "fused_qkvza_hfq4g256_mq4v2",
            kernels::FUSED_QKVZA_MQ4G256V2_SRC,
            ["fused_qkvza_mq4g256v2"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC,
            ["fused_rmsnorm_mq_rotate_awq"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_g12dec",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_G12DEC_SRC,
            ["fused_rmsnorm_mq_rotate_awq_g12dec"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC,
            ["fused_silu_mul_mq_rotate_awq"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq_i4",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_SRC,
            ["fused_silu_mul_mq_rotate_awq_i4"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq_i4_hin",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_SRC,
            ["fused_silu_mul_mq_rotate_awq_i4_hin"]
        );
        add!(
            "gated_delta_net_q8_fast",
            kernels::GATED_DELTA_NET_Q8_FAST_SRC,
            [
                "gated_delta_net_q8_fast",
                "gated_delta_net_q8_fast_independent_masked"
            ]
        );
        add!(
            "gdn_chunk_prep_gfx11",
            kernels::GDN_CHUNK_PREP_GFX11_SRC,
            ["gdn_chunk_prep_gfx11"]
        );
        add!(
            "gemm_gate_up_mq4g256v2_wmma",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_SRC,
            ["gemm_gate_up_mq4g256v2_wmma"]
        );
        // HIPFIRE_IU4_PREFILL=0 uses the Q8_1 MMQ source unchanged; both
        // X128 and per-32 exports share this object and compiler recipe.
        add!(
            "gemm_mq4g256v2_residual_mmq",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_SRC,
            [
                "gemm_mq4g256v2_residual_mmq",
                "gemm_mq4g256v2_residual_mmq_full_add",
                "gemm_mq4g256v2_residual_mmq_full_set",
                "gemm_mq4g256v2_residual_mmq_x128",
                "gemm_mq4g256v2_residual_mmq_full_add_x128",
                "gemm_mq4g256v2_residual_mmq_full_set_x128",
                "quantize_q8_1_mmq_ds4",
                "quantize_q8_1_mmq_ds4_x128",
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_wmma",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_SRC,
            ["gemm_mq4g256v2_residual_wmma"]
        );
        add!(
            "gemm_qkv_mq4g256v2_wmma",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_SRC,
            ["gemm_qkv_mq4g256v2_wmma"]
        );
        add!(
            "gemm_qkvza_mq4g256v2_wmma",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_SRC,
            ["gemm_qkvza_mq4g256v2_wmma"]
        );
        add!(
            "mq_rotate_x",
            kernels::GEMV_MQ4G256_SRC,
            ["gemv_mq4g256", "mq_rotate_x"]
        );
        add!(
            "rmsnorm_f32_rowsplit",
            kernels::RMSNORM_ROWSPLIT_SRC,
            ["rmsnorm_f32_rowsplit"]
        );
        add!(
            "rope_partial_halfsplit_batched",
            kernels::ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC,
            ["rope_partial_halfsplit_batched_f32"]
        );
        add!(
            "rotate_x_mq_awq",
            kernels::ROTATE_X_MQ_AWQ_SRC,
            ["rotate_x_mq_awq"]
        );
        add!(
            "sigmoid_mul_rotate_x_mq_awq_i4_gfx11",
            kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX11_SRC,
            ["sigmoid_mul_rotate_x_mq_awq_i4_gfx11"]
        );
        add!(
            "softmax_temp_topp_batched",
            kernels::SOFTMAX_TEMP_BATCHED_SRC,
            ["softmax_temp_batched_f32", "softmax_temp_topp_batched_f32"]
        );
        add!(
            "split_mq4v2_z_betaalpha",
            kernels::SPLIT_MQ4V2_Z_BETAALPHA_SRC,
            ["split_mq4v2_z_betaalpha", "split_mq4v2_z_betaalpha_g12"]
        );
        add!(
            "topk_logsumexp_batched",
            kernels::TOPK_LOGSUMEXP_BATCHED_SRC,
            ["topk_logsumexp_batched_f32"]
        );
        // Qwen4 QSA split-KV decode for q8 K/V on exact gfx1151 and gfx1100 (same HIPFIRE_QWEN4_QSA_DECODE_SPLITK switch; KLD-gated; tensor_ops.rs qsa_decode_splitkv_q8_launch).
        add!(
            "qsa_decode_splitkv_q8",
            kernels::QSA_DECODE_SPLITKV_Q8_SRC,
            [
                "qsa_decode_splitkv_partial_q8",
                "qsa_decode_splitkv_merge_f32"
            ]
        );
    }
    if arch == "gfx1201" {
        // Z/beta/alpha fold split of the non-fused IU4 verify GDN projection
        // (`gemm_zba_fold_g12_iu4_prepared`); shares the halo split's source.
        add!(
            "split_mq4v2_z_betaalpha",
            kernels::SPLIT_MQ4V2_Z_BETAALPHA_SRC,
            ["split_mq4v2_z_betaalpha_g12", "split_mq4v2_z_betaalpha"]
        );
        // Dense 27B DFlash cold helpers; retain runtime source preambles and
        // historical gfx1100 module/symbol names on gfx1201.
        add!(
            "dflash_hidden_commit5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "dflash_hidden_scatter5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "gated_norm_mq_rotate_f16",
            crate::mq_f16_residual_producers::GATED_NORM_F16_SRC,
            [
                "gated_norm_mq_rotate_awq_f16_batched_gfx1100",
                "gated_norm_mq_rotate_f16_batched_gfx1100"
            ]
        );
        add!(
            "fused_silu_mul_mq_rotate_f16",
            crate::mq_f16_residual_producers::FUSED_SILU_F16_SRC,
            [
                "fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100",
                "fused_silu_mul_mq_rotate_f16_batched_gfx1100"
            ]
        );
        add!(
            "sigmoid_mul_mq_rotate_f16",
            crate::mq_f16_residual_producers::SIGMOID_MUL_F16_SRC,
            [
                "sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100",
                "sigmoid_mul_mq_rotate_f16_batched_gfx1100"
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_f16",
            crate::mq_f16_producers::FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            [
                "fused_rmsnorm_mq_rotate_awq_f16",
                "fused_rmsnorm_mq_rotate_f16"
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_f16",
            crate::mq_f16_producers::FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            [
                "fused_rmsnorm_mq_rotate_awq_f16",
                "fused_rmsnorm_mq_rotate_f16"
            ]
        );
        add!(
            "dflash_gdn_pre_gfx1100",
            crate::dflash_gdn_pre::DFLASH_GDN_PRE_GFX1100_SRC,
            [
                "dflash_gdn_pre_capture_gfx1100",
                "dflash_gdn_pre_replay_gfx1100"
            ]
        );
        add!(
            "qwen35_fa_prep_batched_gfx1100",
            crate::qwen35_fa_batch::FA_PREP_BATCHED_SRC,
            ["qwen35_fa_prep_batched_gfx1100"]
        );
        add!(
            "kv_cache_write_q8_0_pair_batched_gfx1100",
            crate::qwen35_fa_batch::KV_PAIR_BATCHED_SRC,
            ["kv_cache_write_q8_0_pair_batched_gfx1100"]
        );
        add!(
            "dflash_gdn_replay_pre_ml_from",
            crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC,
            ["dflash_gdn_replay_pre_ml_from"]
        );
        add!(
            "gated_delta_net_q8_fast_ml_from",
            crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC,
            ["gated_delta_net_q8_fast_ml_from"]
        );
        add!(
            "attention_verify_gqa_gfx1201",
            kernels::ATTENTION_VERIFY_GQA_GFX1201_SRC,
            [
                "attention_verify_gqa_q8_gfx1201",
                "attention_verify_gqa_fp8_gfx1201",
                "attention_verify_gqa_fp8_gqa8_gfx1201",
                "attention_verify_gqa_fwht4_gfx1201",
                "attention_verify_gqa_fwht3_gfx1201",
                "attention_verify_reduce_gfx1201"
            ]
        );
        add!(
            "attention_flash_fp8_e4m3_tile_batched",
            assemble_asym(kernels::ATTENTION_FLASH_FP8_E4M3_TILE_BATCHED_SRC),
            [
                "attention_flash_fp8_e4m3_tile_batched",
                "attention_flash_q8_0_tile_batched"
            ]
        );
        add!(
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201",
            kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_GFX1201_SRC,
            [
                "attention_fp8_e4m3_fa2_gqa_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201"
            ]
        );
        add!(
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201",
            kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_Q8_A4EPI_GFX1201_SRC,
            [
                "attention_fp8_e4m3_fa2_gqa_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201"
            ]
        );
        add!(
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
            kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_Q8_GFX1201_SRC,
            [
                "attention_fp8_e4m3_fa2_gqa_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201"
            ]
        );
        add!(
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201",
            kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_Q8_A3B_GFX1201_SRC,
            [
                "attention_fp8_e4m3_fa2_gqa_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201"
            ]
        );
        // Native-fp8 split-KV FA2 DFlash verifier (`Gpu::attention_fp8_e4m3_fa2_gqa_split_gfx1201`,
        // HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8). One module named for the partial
        // symbol, the name the launcher requests; the merge and Q pre-convert
        // entries bind out of the same object. The direct `_f16_` entry is
        // defined by the same source and declared here but never bound (the
        // Q0 module owns the launcher that requests it).
        add!(
            "attention_fp8_e4m3_fa2_gqa_partial_lse_gfx1201",
            kernels::ATTENTION_FP8_E4M3_FA2_GQA_SPLIT_LSE_GFX1201_SRC,
            [
                "attention_fp8_e4m3_fa2_gqa_f16_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_lse_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_lse_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_lse_gfx1201"
            ]
        );
        add!(
            "dflash_gdn_replay_pre_ml",
            crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_SRC,
            ["dflash_gdn_replay_pre_ml"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SLAB_FDIV_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SLAB_TOKFAST_SRC,
            ["fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast"]
        );
        add!(
            "gated_delta_net_q8_fast_ml",
            crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_SRC,
            ["gated_delta_net_q8_fast_ml"]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx12_v2_slab",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SLAB_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx12_v2_slab"]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_XBF16_SLAB_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab"]
        );
        add!(
            "gdn_chunk_kkt_solve_batched",
            kernels::GDN_CHUNK_KKT_SOLVE_BATCHED_SRC,
            ["gdn_chunk_kkt_solve_batched"]
        );
        add!(
            "gdn_chunk_prep",
            kernels::GDN_CHUNK_PREP_SRC,
            ["gdn_chunk_prep"]
        );
        add!(
            "gdn_chunk_prep_fixup",
            kernels::GDN_CHUNK_PREP_FIXUP_SRC,
            ["gdn_chunk_prep_fixup"]
        );
        add!(
            "gdn_chunk_scan_bf16",
            kernels::GDN_CHUNK_SCAN_BF16_SRC,
            ["gdn_chunk_scan_bf16"]
        );
        add!(
            "gdn_chunk_scan_bf16_mseg",
            kernels::GDN_CHUNK_SCAN_BF16_MSEG_SRC,
            ["gdn_chunk_scan_bf16_mseg"]
        );
        add!(
            "fn_gdn_dense",
            kernels::FN_GDN_DENSE_SRC,
            [
                "fn_gdn_dense_producer",
                "fn_gdn_dense_producer_v2_f32",
                "fn_gdn_dense_producer_v2_bf16",
                "fn_gdn_dense_gate_bf16in",
                "fn_gdn_dense_gate_i8",
                "fn_gdn_dense_gate_rotate_bf16in"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "quantize_int4_mmq_ds128"
            ]
        );
        add!(
            "qwen35_fa_prep_fp8q_nogate_batched_gfx1201",
            crate::qwen35_fa_batch::FA_PREP_BATCHED_GFX1201_SRC,
            [
                "qwen35_fa_prep_batched_gfx1201",
                "qwen35_fa_prep_fp8q_batched_gfx1201",
                "qwen35_fa_prep_fp8q_nogate_batched_gfx1201"
            ]
        );
        // select_regrid / topk_values_fixup_f32 live in the gfx1201|gfx1151 block below.
        add!(
            "sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab",
            kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SLAB_SRC,
            ["sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab"]
        );
        add!(
            "sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab",
            kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GIL_GFX12_SLAB_SRC,
            ["sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab"]
        );
    }
    // `Gpu::select_regrid_enabled` routes the row selectors through these on
    // gfx1201 and gfx1151 (measured: Halo 27B DFlash/MTP request them first
    // on a host without hipcc).
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!(
            "select_regrid",
            crate::select_regrid::SELECT_REGRID_SRC,
            ["argmax_f32_batched_regrid", "topk_values_regrid_f32"]
        );
        add!(
            "topk_values_fixup_f32",
            crate::select_regrid::TOPK_VALUES_FIXUP_SRC,
            ["topk_values_fixup_f32"]
        );
    }
    if arch == "gfx1100" {
        add!(
            "attention_verify_gqa_gfx1100",
            kernels::ATTENTION_VERIFY_GQA_GFX1100_SRC,
            [
                "attention_verify_gqa_q8_gfx1100",
                "attention_verify_gqa_fwht4_gfx1100",
                "attention_verify_gqa_fwht3_gfx1100",
                "attention_verify_gqa_q8_rows_gfx1100",
                "attention_verify_reduce_gfx1100"
            ]
        );
        // Default-on WMMA split-K verify attention (`attention_wmma_verify.rs`).
        add!(
            "attention_verify_wmma_gfx1100",
            kernels::ATTENTION_VERIFY_WMMA_GFX1100_SRC,
            [
                "attention_verify_wmma_q8_qt1_gfx1100",
                "attention_verify_wmma_q8_qt2_gfx1100",
                "attention_verify_wmma_q8_qt3_gfx1100",
                "attention_verify_wmma_reduce_gfx1100"
            ]
        );
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_DEC_GFX1100_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100"]
        );
        add!(
            "attention_flash_q8_0_tile_batched",
            assemble_asym(kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC),
            ["attention_flash_q8_0_tile_batched"]
        );
        add!(
            "attention_flash_q8_0_tile_gqa_gfx1100",
            kernels::ATTENTION_FLASH_Q8_0_TILE_GQA_GFX1100_SRC,
            ["attention_flash_q8_0_tile_gqa_gfx1100"]
        );
        add!(
            "attention_q8_0_fa2_gqa_gfx1100",
            q8_fa2_gqa_gfx1100_source(),
            [
                "attention_fa2_q_preconvert_gfx1100",
                "attention_q8_0_fa2_gqa_gfx1100",
                "attention_q8_0_fa2_gqa_partial_gfx1100",
                "attention_q8_0_fa2_gqa_merge_gfx1100"
            ]
        );
        add!(
            "conv1d_silu_split_qknorm_b256_scalar_prep",
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SCALAR_PREP_SRC,
            ["conv1d_silu_split_qknorm_b256_scalar_prep"]
        );
        add!(
            "dflash_gdn_pre_gfx1100",
            crate::dflash_gdn_pre::DFLASH_GDN_PRE_GFX1100_SRC,
            [
                "dflash_gdn_pre_capture_gfx1100",
                "dflash_gdn_pre_replay_gfx1100"
            ]
        );
        add!(
            "dflash_hidden_commit5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "dflash_hidden_scatter5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "dflash_state_bulk_copy_gfx1100",
            crate::dflash_state_copy::DFLASH_STATE_BULK_COPY_GFX1100_SRC,
            ["dflash_state_bulk_copy_gfx1100"]
        );
        add!(
            "dynamic_conv_residual_gfx1100",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            [
                "dynamic_conv_residual_gfx1100",
                "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100",
                "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8",
                "mq_rotate_x_f16_dflash_gfx1100",
                "rmsnorm_residual_dual_gfx1100"
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_f16",
            crate::mq_f16_producers::FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            [
                "fused_rmsnorm_mq_rotate_awq_f16",
                "fused_rmsnorm_mq_rotate_f16"
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_b8",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_B8_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_b8"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_f16",
            crate::mq_f16_producers::FUSED_RMSNORM_MQ_ROTATE_F16_SRC,
            [
                "fused_rmsnorm_mq_rotate_awq_f16",
                "fused_rmsnorm_mq_rotate_f16"
            ]
        );
        add!(
            "fused_silu_mul_mq_rotate_f16",
            crate::mq_f16_residual_producers::FUSED_SILU_F16_SRC,
            [
                "fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100",
                "fused_silu_mul_mq_rotate_f16_batched_gfx1100"
            ]
        );
        add!(
            "gated_delta_net_q8_compact3_b2",
            kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC,
            [
                "gated_delta_net_q8_compact3_b2",
                "gated_delta_net_q8_fast_independent_masked"
            ]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx1100_v2",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX1100_V2_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx1100_v2"]
        );
        add!(
            "gated_norm_mq_rotate_awq_k6144_gfx1100",
            kernels::gated_norm_mq_rotate_awq_k6144_gfx1100_src(),
            ["gated_norm_mq_rotate_awq_k6144_gfx1100"]
        );
        add!(
            "gated_norm_mq_rotate_f16",
            crate::mq_f16_residual_producers::GATED_NORM_F16_SRC,
            [
                "gated_norm_mq_rotate_awq_f16_batched_gfx1100",
                "gated_norm_mq_rotate_f16_batched_gfx1100"
            ]
        );
        add!(
            "gdn_chunk_kkt_solve_gfx1100",
            kernels::GDN_CHUNK_KKT_SOLVE_GFX1100_SRC,
            ["gdn_chunk_kkt_solve_gfx1100"]
        );
        add!(
            "gdn_chunk_scan",
            kernels::GDN_CHUNK_SCAN_SRC,
            ["gdn_chunk_scan"]
        );
        add!(
            "gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX1100_LDSSTAGE_SRC,
            ["gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage"]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "quantize_int4_mmq_ds128"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gridspec_symfold",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SYMFOLD_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
                "quantize_int4_mmq_ds128"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX1100_LDSSTAGE_SRC,
            ["gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage"]
        );
        add!(
            "gemm_mq6g256v2_residual_gfx1100_port",
            kernels::GEMM_MQ6G256V2_RESIDUAL_GFX1100_PORT_SRC,
            [
                "gemm_mq6g256v2_residual_gfx1100_port_w4n128",
                "gemm_mq6g256v2_residual_gfx1100_port_w8n256"
            ]
        );
        add!(
            "gemm_mqv2_wmma_gfx1100_mw_lds",
            kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC,
            [
                "gemm_gate_up_mq3g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_mw8_lds",
                "gemm_mq3g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq3g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq4g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq5g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq5g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
            ]
        );
        add!(
            "gemv_hfq4g256_residual_rdna3_mq4v2",
            kernels::GEMV_MQ4G256V2_RESIDUAL_SRC,
            ["gemv_mq4g256v2_residual"]
        );
        add!(
            "gemv_mq4g256v2_rdna3_mq4v2",
            kernels::GEMV_MQ4G256V2_SRC,
            ["gemv_mq4g256v2"]
        );
        add!(
            "kv_cache_write_q8_0_pair",
            kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC,
            ["kv_cache_write_q8_0_pair"]
        );
        add!(
            "kv_cache_write_q8_0_pair_batched_gfx1100",
            crate::qwen35_fa_batch::KV_PAIR_BATCHED_SRC,
            ["kv_cache_write_q8_0_pair_batched_gfx1100"]
        );
        add!(
            "qwen35_fa_prep_batched_gfx1100",
            crate::qwen35_fa_batch::FA_PREP_BATCHED_SRC,
            ["qwen35_fa_prep_batched_gfx1100"]
        );
        add!(
            "qwen36_27b_fa_prep_gfx1100",
            kernels::qwen36_27b_fa_prep_gfx1100_src(),
            ["qwen36_27b_fa_prep_gfx1100"]
        );
        add!(
            "rmsnorm_residual_dual_gfx1100",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            [
                "dynamic_conv_residual_gfx1100",
                "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100",
                "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4",
                "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8",
                "mq_rotate_x_f16_dflash_gfx1100",
                "rmsnorm_residual_dual_gfx1100"
            ]
        );
        add!(
            "rope_partial_halfsplit",
            kernels::ROPE_PARTIAL_HALFSPLIT_SRC,
            ["rope_partial_halfsplit_f32"]
        );
        add!(
            "sigmoid_mul_mq_rotate_f16",
            crate::mq_f16_residual_producers::SIGMOID_MUL_F16_SRC,
            [
                "sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100",
                "sigmoid_mul_mq_rotate_f16_batched_gfx1100"
            ]
        );
    }
    if arch == "gfx1151" {
        // Qwen4 HC input_mix_down (320 x 10240) + activation, one launch
        // (HIPFIRE_QWEN4_HC_DOWN_FUSED; gemm.rs gemm_bf16_xf16_hc_down_fused).
        add!(
            "qwen4_hc_down_fused_gfx1151",
            kernels::QWEN4_HC_DOWN_FUSED_GFX1151_SRC,
            ["qwen4_hc_down_fused_gfx1151"]
        );
        // Qwen4 HC read up+mix, persistent W-stationary (HIPFIRE_QWEN4_HC_UP_V2;
        // tensor_ops.rs hyper_read_up_wmma_launch).
        add!(
            "hyper_read_up_wmma_gfx1151_v2",
            crate::tensor_ops::HYPER_READ_UP_WMMA_GFX1151_V2_SRC,
            ["hyper_read_up_wmma_bf16_gfx1151_v2"]
        );
        // QSA gathered F16 WMMA attention (`tensor_ops::qsa_gathered_wmma_launch_with`
        // and the QSA F16 arena producer): the route is gfx1151-only for F32/Q8
        // K/V (`qsa_gathered_wmma_enabled`); the Q8 converter stays HIP even when
        // the PM image serves the attend kernel.
        add!(
            "indexed_attention_gathered_wmma",
            crate::tensor_ops::INDEXED_ATTENTION_GATHERED_WMMA_SRC,
            [
                "indexed_attention_gathered_wmma_f16",
                "indexed_attention_kv_f16vb",
                "indexed_attention_kv_f16vb_q8",
            ]
        );
        add!(
            "attention_flash_q8_0_tile_gqa_gfx1151",
            kernels::ATTENTION_FLASH_Q8_0_TILE_GQA_GFX1151_SRC,
            ["attention_flash_q8_0_tile_gqa_gfx1151"]
        );
        add!(
            "attention_flash_q8_0_tile_gqa_gemma_gfx1151",
            kernels::ATTENTION_FLASH_Q8_0_TILE_GQA_GEMMA_GFX1151_SRC,
            [
                "attention_flash_q8_0_tile_gqa_d256g2_gfx1151",
                "attention_flash_q8_0_tile_gqa_d512g8_gfx1151"
            ]
        );
        add!(
            "attention_flash_reduce_dsplit_gfx1151",
            kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1151_SRC,
            ["attention_flash_reduce_dsplit_gfx1151"]
        );
        add!(
            "attention_q8_0_fa2_gqa_gfx1151",
            kernels::ATTENTION_Q8_0_FA2_GQA_GFX1151_SRC,
            [
                "attention_fa2_q_preconvert_gfx1151",
                "attention_q8_0_fa2_gqa_gfx1151"
            ]
        );
        add!(
            "conv1d_silu_split_qknorm_b256",
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
            ["conv1d_silu_split_qknorm_b256"]
        );
        add!(
            "deinterleave_q_rmsnorm_f32_batched",
            kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC,
            ["deinterleave_q_rmsnorm_f32_batched"]
        );
        // Exact-gfx1151 pure byte-copy DFlash/DeltaNet helpers; same module and
        // symbol names, source and recipe as gfx1100/gfx1201 (HIPFIRE_DN_SNAPSHOT_BULK_OFF,
        // HIPFIRE_HIDDEN_SCATTER_FUSE_OFF kill switches unchanged).
        add!(
            "dflash_hidden_commit5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "dflash_hidden_scatter5_gfx1100",
            crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC,
            [
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_scatter5_gfx1100"
            ]
        );
        add!(
            "dflash_state_bulk_copy_gfx1100",
            crate::dflash_state_copy::DFLASH_STATE_BULK_COPY_GFX1100_SRC,
            ["dflash_state_bulk_copy_gfx1100"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_b8_gfx1151",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_B8_GFX1151_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_b8_gfx1151"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_fold_b8",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_FOLD_B8_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_fold_b8"]
        );
        // Qwen4 chunked-GDN inline-Q8 recurrence (opt-in `HIPFIRE_QWEN4_GDN_Q8_INLINE`, exact gfx1151; tensor_ops.rs gated_delta_step_gate_wmma_arms).
        add!(
            "gated_delta_chunk_q8_wmma",
            crate::tensor_ops::GATED_DELTA_CHUNK_Q8_WMMA_SRC,
            ["gated_delta_chunk_gate_q8_wmma"]
        );
        add!(
            "gated_delta_chunk_wmma_arena",
            crate::tensor_ops::npu_arena::CHUNK_ARENA_SRC,
            ["gated_delta_chunk_gate_wmma_arena"]
        );
        add!(
            "gated_delta_chunk_q8_wmma_arena",
            crate::tensor_ops::npu_arena::CHUNK_Q8_ARENA_SRC,
            ["gated_delta_chunk_gate_q8_wmma_arena"]
        );
        add!(
            "npu_trunk_consumers",
            crate::tensor_ops::npu_arena::ARENA_SRC,
            [
                "gated_delta_conv_batched_arena",
                "gated_delta_conv_qknorm_batched_arena",
                "gated_delta_params_batched_arena",
                "gated_delta_gate_arena",
                "gated_delta_gate_batched_arena",
                "indexed_attention_norm_rope_batch_arena",
                "gated_delta_gate_batched_rotate_arena",
                "indexed_attention_cache_append_q8_batch_arena",
            ]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx1151_v2",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX1151_V2_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx1151_v2"]
        );
        add!(
            "gdn_chunk_scan_gfx1151",
            kernels::GDN_CHUNK_SCAN_GFX1151_SRC,
            ["gdn_chunk_scan_gfx1151"]
        );
        add!(
            "gemm_mq4g256v2_residual_iu4_v2b_gfx11",
            kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_V2B_GFX11_SRC,
            [
                "gemm_mq4g256v2_gate_up_silu_iu4_v2b_gfx11",
                "gemm_mq4g256v2_residual_iu4_v2b_add_gfx11",
                "gemm_mq4g256v2_residual_iu4_v2b_add_touch_gfx11",
                "gemm_mq4g256v2_residual_iu4_v2b_add_touch_swz_gfx11",
                "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11",
                "gemm_mq4g256v2_residual_iu4_v2b_set_zba_gfx11"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "quantize_int4_mmq_ds128"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX11_X5_SYMFOLD_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_x5_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_x5_col_gfx1151_symfold",
                "quantize_int4_mmq_ds128"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gridspec_symfold",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SYMFOLD_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
                "quantize_int4_mmq_ds128"
            ]
        );
        // Exact-gfx1151 1..=16-row MQ4V2 + HFQ4G256/MQ4G256 v1 WMMA (HIPFIRE_GFX1151_XLDS; =0 restores the base kernels).
        add!(
            "gemm_mq4g256v2_wmma_gfx1151_xlds",
            kernels::GEMM_MQ4G256V2_WMMA_GFX1151_XLDS_SRC,
            [
                "gemm_gate_up_hfq4g256_wmma_gfx1151_xlds",
                "gemm_gate_up_mq4g256v2_wmma_gfx1151_xlds",
                "gemm_hfq4g256_residual_wmma_gfx1151_xlds",
                "gemm_mq4g256v2_residual_wmma_gfx1151_xlds",
                "gemm_qkv_hfq4g256_wmma_gfx1151_xlds",
                "gemm_qkv_mq4g256v2_wmma_gfx1151_xlds",
                "gemm_qkvza_hfq4g256_wmma_gfx1151_xlds",
                "gemm_qkvza_mq4g256v2_wmma_gfx1151_xlds"
            ]
        );
        // Exact-gfx1151 1..=16-row Q8_0 WMMA (HIPFIRE_GFX1151_XLDS; =0 restores the base kernel).
        add!(
            "gemm_q8_0_wmma_gfx1151_xlds",
            kernels::GEMM_Q8_0_WMMA_GFX1151_XLDS_SRC,
            ["gemm_q8_0_wmma_gfx1151_xlds"]
        );
        add!(
            "gemm_mqv2_wmma_gfx1151_mw_lds",
            kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC,
            [
                "gemm_gate_up_mq3g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_mw8_lds",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_mw4_lds",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_mw8_lds",
                "gemm_mq3g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq3g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq4g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq5g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq5g256v2_residual_wmma_gfx11_mw8_lds",
                "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds",
                "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
            ]
        );
        add!(
            "gemv_hfq4g256_multirow_default_mq4v2",
            kernels::GEMV_MQ4G256V2_MULTIROW_SRC,
            [
                "gemv_mq4g256v2_multirow_r2",
                "gemv_mq4g256v2_multirow_r4",
                "gemv_mq4g256v2_multirow_r8"
            ]
        );
        add!(
            "gemv_mq4g256v2_multirow_r2_x3",
            kernels::GEMV_MQ4G256V2_MULTIROW_R2_X3_SRC,
            ["gemv_mq4g256v2_multirow_r2_x3"]
        );
        add!(
            "gemv_mq4g256v2_mq4v2",
            kernels::GEMV_MQ4G256V2_SRC,
            ["gemv_mq4g256v2"]
        );
        add!(
            "gemv_mq4g256v2_residual_row_serial_gfx1151",
            kernels::GEMV_MQ4G256V2_RESIDUAL_ROW_SERIAL_GFX1151_SRC,
            ["gemv_mq4g256v2_residual_row_serial_gfx1151"]
        );
        add!(
            "qwen35_fa_prep_batched_gfx1151",
            crate::qwen35_fa_batch::FA_PREP_BATCHED_GFX1151_SRC,
            ["qwen35_fa_prep_batched_gfx1151"]
        );
        // Exact-gfx1151 Qwen4 decode MoE head (HIPFIRE_QWEN4_DECODE_HEAD_FUSE; =0 keeps the x4 + Q8_0 pair launches).
        add!(
            "qwen4_decode_head_fuse",
            kernels::QWEN4_DECODE_HEAD_FUSE_SRC,
            ["gemv_bf16_xf32_x2_q8_0_k2560_staged_pair"]
        );
        // Exact-gfx1151 derived BF16 mirror of the F32 pooled index keys (HIPFIRE_QWEN4_QSA_POOLED_BF16, default off).
        add!(
            "qsa_pooled_bf16_mirror",
            kernels::QSA_POOLED_BF16_MIRROR_SRC,
            [
                "qsa_pooled_f32_to_bf16_mirror",
                "qsa_pooled_f32_low_nonzero_count"
            ]
        );
        // Exact-gfx1151 Qwen4 QSA split-K decode (default on (exact gfx1151; =0 kill switch) HIPFIRE_QWEN4_QSA_DECODE_SPLITK; KLD-gated; tensor_ops.rs qsa_decode_splitk_launch).
        add!(
            "qsa_decode_splitk",
            kernels::QSA_DECODE_SPLITK_SRC,
            [
                "qsa_decode_splitk_partial_f32",
                "qsa_decode_splitk_combine_f32"
            ]
        );
        add!(
            "repeat_interleave_qk_batched",
            kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC,
            ["repeat_interleave_qk_f32_batched"]
        );
        add!(
            "rope_partial_halfsplit_f32_headgrid",
            kernels::ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC,
            ["rope_partial_halfsplit_f32_headgrid"]
        );
        // Exact-gfx1151 multi-lane fn-lanes GDN decode core (default on; =0 kill switch HIPFIRE_QWEN4_GDN_LANES; tensor_ops.rs gated_delta_core_lanes).
        add!(
            "gdn_lanes",
            crate::tensor_ops::GDN_LANES_SRC,
            [
                // GDN_LANES_SRC also compiles the complete tensor_ops translation unit.
                "argmax_f32",
                "bf16_roundtrip_f32",
                "bf16_scaled_add_batched_f32",
                "bf16_scaled_add_f32",
                "copy_regions_u32",
                "gated_delta_conv_bf16_f32",
                "gated_delta_conv_bf16_f32_batched_k4",
                "gated_delta_conv_params_bf16_f32",
                "gated_delta_conv_qknorm_bf16_f32_batched_k4",
                "gated_delta_gate_bf16_f32",
                "gated_delta_gate_bf16_f32_batched",
                "gated_delta_gate_rotate_bf16_f32_batched",
                "gated_delta_params_bf16_f32",
                "gated_delta_params_bf16_f32_batched",
                "gated_delta_params_f32",
                "gated_delta_qk_norm_bf16_batched",
                "gated_delta_rollback_layers_f32",
                "gated_delta_rollback_layers_q8",
                "gated_delta_step_f32",
                "gated_delta_step_gate_norm128_gfx1151",
                "gated_delta_step_gate_norm128_q8",
                "gated_delta_step_gate_norm128_q8ef",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "gated_delta_step_halves_state128_persistent256_capture_f32",
                "gated_delta_step_halves_state128_persistent256_capture_q8",
                "gated_delta_step_halves_state128_persistent256_capture_q8ef",
                "gated_delta_step_halves_state128_persistent256_f32",
                "gated_delta_step_halves_state128_persistent256_q8",
                "gated_delta_step_halves_state128_persistent256_q8ef",
                "gated_delta_step_norm128_q8",
                "gated_delta_step_norm128_q8ef",
                "gated_delta_step_shared_norm128_gfx1151",
                "gdn_q8ef_residual_zero",
                "gdn_state_f32_to_q8",
                "gdn_state_f32_to_q8ef",
                "gdn_state_q8_to_f32",
                "gdn_state_q8ef_to_f32",
                "hc_activation_fused_f32",
                "hc_state_bf16_add_f32",
                "hc_state_bf16_to_f32",
                "hyper_norm_f32",
                "hyper_norm_gate_f32",
                "hyper_norm_gate_outputs",
                "hyper_norm_gate_outputs_f32",
                "hyper_read_f32",
                "hyper_read_projected_f32",
                "hyper_read_up_fused_f32",
                "hyper_write_bf16x2",
                "hyper_write_f32",
                "indexed_attention_attention_f32",
                "indexed_attention_attention_f32_batched",
                "indexed_attention_attention_f32_batched_hg12",
                "indexed_attention_attention_f32_batched_hg4",
                "indexed_attention_attention_f32_batched_serial",
                "indexed_attention_attention_f32_serial",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_f32",
                "indexed_attention_cache_append_f32_batched",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_f32",
                "indexed_attention_decode_prologue_q8",
                "indexed_attention_index_key_append_bf16_batched",
                "indexed_attention_norm_rope_f32",
                "indexed_attention_norm_rope_f32_batched",
                "indexed_attention_pool_rope_bf16",
                "indexed_attention_pool_rope_f32",
                "indexed_attention_reuse_selection",
                "indexed_attention_rope_table_f32",
                "indexed_attention_select_bf16_batched",
                "indexed_attention_select_bf16_batched_serial",
                "indexed_attention_select_f32",
                "indexed_attention_select_f32_batched",
                "indexed_attention_select_f32_batched_serial",
                "indexed_attention_select_f32_serial",
                "indexed_attention_select_from_scores",
                "indexed_attention_select_scores_rows16_f32",
                "indexed_attention_select_scores_rows32_f32",
                "indexed_attention_select_scores_rows8_f32",
                "scale_f32",
                "gated_delta_conv_params_bf16_f32_lanes",
                "gated_delta_step_gate_norm128_q8_lanes"
            ]
        );
    }
    // Qwen3.8-Flash-Next (Qwen4) load, AR prefill/decode and native MTP:
    // every module a kernel-load trace (tests/fixtures/kernel-trace-qwen4-
    // flash-next.tsv) JIT-compiled on gfx1151 and gfx1201 beyond the
    // inventory above. gfx1100 cannot hold Flash-Next on the trace hosts; its
    // rows are the same callsites resolved through gfx1100's arch predicates
    // (no sym-IU4 MoE, F16 WMMA MoE at >= 512 rows, `_rdna3` GEMV modules,
    // `qwen4_gemm_wmma_lds256` beside the shipped LDS256 source). Sources are
    // the exact expressions the callsites pass to `ensure_kernel`; symbols are
    // every kernel the arch's object defines.
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!(
            "copy_f32_buffer",
            kernels::COPY_F32_BUFFER_SRC,
            ["copy_f32_buffer", "copy_f32_strided_slot_buffer"]
        );
        add!(
            "ep_gather_rows_by_owner",
            kernels::EP_GATHER_ROWS_BY_OWNER_SRC,
            ["ep_gather_rows_by_owner"]
        );
        add!(
            "ep_gather_rows_by_slot_owner",
            kernels::EP_GATHER_ROWS_BY_SLOT_OWNER_SRC,
            ["ep_gather_rows_by_slot_owner"]
        );
        add!(
            "gemm_bf16_xf32_multirow",
            kernels::GEMM_BF16_XF32_MULTIROW_SRC,
            [
                "convert_bf16_to_f16",
                "gemm_bf16_xf32_multirow",
                "gemm_bf16_xf32_multirow_pto2",
                "gemm_bf16_xf32_multirow_rows3",
            ]
        );
        add!(
            "gemm_bf16_xf32_multirow_r16w4_gfx1151",
            kernels::GEMM_BF16_XF32_MULTIROW_R16_GFX1151_SRC,
            [
                "gemm_bf16_xf32_multirow_r16_gfx1151",
                "gemm_bf16_xf32_multirow_r16w4_gfx1151",
                "gemm_bf16_xf32_multirow_r16w4t_gfx1151",
            ]
        );
        add!(
            "gemm_bf16_xf32_multirow_r16w4t_gfx1151",
            kernels::GEMM_BF16_XF32_MULTIROW_R16_GFX1151_SRC,
            [
                "gemm_bf16_xf32_multirow_r16_gfx1151",
                "gemm_bf16_xf32_multirow_r16w4_gfx1151",
                "gemm_bf16_xf32_multirow_r16w4t_gfx1151",
            ]
        );
        add!(
            "gemm_bf16_xf32_multirow_rows3",
            kernels::GEMM_BF16_XF32_MULTIROW_SRC,
            [
                "convert_bf16_to_f16",
                "gemm_bf16_xf32_multirow",
                "gemm_bf16_xf32_multirow_pto2",
                "gemm_bf16_xf32_multirow_rows3",
            ]
        );
        // Native MTP batched prompt fill, HC input_mix_up (K = 320).
        add!(
            "gemm_bf16_xf32_multirow_pto2",
            kernels::GEMM_BF16_XF32_MULTIROW_SRC,
            [
                "convert_bf16_to_f16",
                "gemm_bf16_xf32_multirow",
                "gemm_bf16_xf32_multirow_pto2",
                "gemm_bf16_xf32_multirow_rows3",
            ]
        );
        // A3B mq4rs on gfx1100/gfx1151 with the explicit HIPFIRE_MOE_SYM_IU4=0
        // opt-in (qt53 F32 prefill fallback) launches the generic multirow
        // MoE down kernel (`MoeDownKernel::Multirow`); measured as the only
        // hipcc-compiled module of that route on both archs.
        if matches!(arch, "gfx1100" | "gfx1151") {
            add!(
                "gemm_mq4g128v2_moe_grouped_top10_multirow",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_MULTIROW_SRC,
                ["gemm_mq4g128v2_moe_grouped_top10_multirow"]
            );
        }
        add!(
            "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
            kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_GFX1151_SRC,
            ["gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",]
        );
        add!(
            "gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs",
            kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_XS_GFX1151_SRC,
            ["gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs4_gfx1151"]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4",
            kernels::GEMM_MQ4G256V2_MOE_GROUPED_TOP10_O4_R8_X4_GFX1151_SRC,
            [
                "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4",
                "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse",
                "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse_k2560",
                "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4",
            ]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe",
            kernels::GEMM_MQ4G256V2_MOE_GROUPED_TOP10_O4_R8_PIPE_GFX1151_SRC,
            ["gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe_gfx1151",]
        );
        add!(
            "gemm_mq6g256v2_f32_rows",
            kernels::GEMM_MQ6G256V2_F32_ROWS_SRC,
            [
                "gemm_mq6g256v2_f32_rows_r2",
                "gemm_mq6g256v2_f32_rows_r3",
                "gemm_mq6g256v2_f32_rows_r4",
                "gemm_mq6g256v2_f32_rows_r5",
                "gemm_mq6g256v2_f32_rows_r6",
                "gemm_mq6g256v2_f32_rows_r7",
                "gemm_mq6g256v2_f32_rows_r8",
                "gemm_mq6g256v2_f32_rows_x4_r2",
                "gemm_mq6g256v2_f32_rows_x4_r3",
                "gemm_mq6g256v2_f32_rows_x4_r4",
                "gemm_mq6g256v2_f32_rows_x4_r5",
                "gemm_mq6g256v2_f32_rows_x4_r6",
                "gemm_mq6g256v2_f32_rows_x4_r7",
                "gemm_mq6g256v2_f32_rows_x4_r8",
                "gemm_mq6g256v2_f32_rows_split_r3_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r3_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r3_w4_s1",
                "gemm_mq6g256v2_f32_rows_split_r4_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r4_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r4_w4_s1",
                "gemm_mq6g256v2_f32_rows_split_r5_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r5_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r5_w4_s1",
                "gemm_mq6g256v2_f32_rows_split_r6_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r6_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r6_w4_s1",
                "gemm_mq6g256v2_f32_rows_split_r7_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r7_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r7_w4_s1",
                "gemm_mq6g256v2_f32_rows_split_r8_w8_s4",
                "gemm_mq6g256v2_f32_rows_split_r8_w4_s4",
                "gemm_mq6g256v2_f32_rows_split_r8_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r3_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r3_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r3_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r4_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r4_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r4_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r5_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r5_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r5_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r6_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r6_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r6_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r7_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r7_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r7_w4_s1",
                "gemm_mq6g256v2_f32_rows_x4_split_r8_w8_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r8_w4_s4",
                "gemm_mq6g256v2_f32_rows_x4_split_r8_w4_s1",
            ]
        );
        add!(
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
            kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC,
            [
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4",
            ]
        );
        // Match the exact gfx1151-only launcher admission in moe_few.rs.
        if arch == "gfx1151" {
            add!(
                "gemv_mq4g128v2_moe_down_top10_k640",
                kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_K640_SRC,
                ["gemv_mq4g128v2_moe_down_top10_k640",]
            );
        }
        add!(
            "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
            kernels::GEMV_MQ4G256V2_MOE_GATE_UP_TOP10_INDEXED_BATCHED_SRC,
            [
                "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
                "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
            ]
        );
        add!(
            "grouped_ops",
            crate::grouped_ops::GROUPED_KERNEL_SRC,
            [
                "grouped_depthwise_conv_silu_add_bf16",
                "grouped_depthwise_conv_silu_add_f32",
                "grouped_gate_bf16",
                "grouped_gate_f32",
                "grouped_gather_convert_bf16",
                "grouped_gather_convert_bf16_f16",
                "grouped_linear_f32",
                "grouped_norm_bf16",
                "grouped_norm_f32",
                "ple_conv_add_bf16s",
                "ple_conv_add_bf16s_k4d3",
                "ple_gate_rows_bf16s",
                "ple_norm_inv",
            ]
        );
        add!(
            "hc_streams_init_from_embed_batched",
            kernels::HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC,
            ["hc_streams_init_from_embed_batched"]
        );
        add!(
            "hyper_read_up_fused_twin",
            crate::tensor_ops::HYPER_READ_UP_FUSED_TWIN_SRC,
            ["hyper_read_up_fused_f32_sx"]
        );
        add!(
            "moe_down_combine_grouped_top10",
            kernels::MOE_DOWN_COMBINE_GROUPED_TOP10_SRC,
            [
                "moe_combine_order_top10",
                "moe_down_combine_grouped_top10",
                "moe_down_combine_grouped_top10_bf16in",
                "moe_down_combine_grouped_top10_bf16in_zinit",
            ]
        );
        // S1 slow-link layer-major prefill kernels: their own module and source, so the
        // legacy `moe_down_combine_grouped_top10` source/identity above is unchanged.
        add!(
            "slowlink_prefill",
            kernels::SLOWLINK_PREFILL_SRC,
            [
                "slowlink_combine_pad",
                "slowlink_combine_panel",
                "slowlink_fold_k1_counts",
                "slowlink_mask_ids",
                "slowlink_merge_rows",
                "slowlink_remap_sorted",
            ]
        );
        add!(
            "moe_down_combine_top10_batched",
            kernels::MOE_DOWN_COMBINE_TOP10_BATCHED_SRC,
            ["moe_down_combine_top10_batched"]
        );
        add!(
            "moe_gate_up_unscatter_silu_top10",
            kernels::MOE_GATE_UP_UNSCATTER_SILU_TOP10_SRC,
            [
                "moe_gate_up_unscatter_silu_rotate128_top10",
                "moe_gate_up_unscatter_silu_top10",
                "moe_gate_up_unscatter_silu_top10_bf16in",
                "moe_unscatter_rotate128_f16",
            ]
        );
        add!(
            "moe_router_softmax_top10_f32",
            kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_SRC,
            ["moe_router_softmax_top10_f32"]
        );
        add!(
            "moe_router_softmax_top10_masked_f32",
            kernels::MOE_ROUTER_SOFTMAX_TOP10_MASKED_F32_SRC,
            ["moe_router_softmax_top10_masked_f32"]
        );
        add!(
            "moe_scatter_fused_top10",
            kernels::MOE_SCATTER_FUSED_TOP10_SRC,
            ["moe_scatter_fused_top10"]
        );
        add!(
            "mq_rotate_x_128_v2",
            kernels::MQ_ROTATE_X_128_V2_SRC,
            [
                "mq_rotate_x_128_v2",
                "mq_rotate_x_128_v2_f16",
                "mq_rotate_x_128_v2_silu_bf16"
            ]
        );
        add!(
            "qwen4_gemv_bf16_xf32",
            kernels::QWEN4_GEMV_BF16_XF32_SRC,
            [
                "gemv_bf16_xf32",
                "gemv_bf16_xf32_bf16_scaled_add",
                "gemv_bf16_xf32_k4",
                "gemv_bf16_xf32_k4_rows_r2",
                "gemv_bf16_xf32_k4_rows_r3",
                "gemv_bf16_xf32_k4_rows_r4",
                "gemv_bf16_xf32_k4_rows_r5",
                "gemv_bf16_xf32_k4_rows_r6",
                "gemv_bf16_xf32_k4_rows_r7",
                "gemv_bf16_xf32_k4_rows_r8",
                "gemv_bf16_xf32_k4_rows_tiled_r2",
                "gemv_bf16_xf32_k4_rows_tiled_r3",
                "gemv_bf16_xf32_k4_rows_tiled_r4",
                "gemv_bf16_xf32_k4_rows_tiled_r5",
                "gemv_bf16_xf32_k4_rows_tiled_r6",
                "gemv_bf16_xf32_k4_rows_tiled_r7",
                "gemv_bf16_xf32_k4_rows_tiled_r8",
                "gemv_bf16_xf32_x4",
                "gemv_bf16_xf32_x4_rows_r2",
                "gemv_bf16_xf32_x4_rows_r3",
                "gemv_bf16_xf32_x4_rows_r4",
                "gemv_bf16_xf32_x4_rows_r5",
                "gemv_bf16_xf32_x4_rows_r6",
                "gemv_bf16_xf32_x4_rows_r7",
                "gemv_bf16_xf32_x4_rows_r8",
                "hyper_write_norm_f32",
            ]
        );
        add!(
            "qwen4_gemv_mq4g256",
            kernels::QWEN4_GEMV_MQ4G256_SRC,
            [
                "hyper_read_projected_rotate_f32",
                "mq_rotate_x_bf16_f16",
                "mq_rotate_x_f16"
            ]
        );
        add!(
            "qwen4_gemv_mq6g256v2",
            kernels::QWEN4_GEMV_MQ6G256V2_SRC,
            ["gemv_mq6g256v2", "gemv_mq6g256v2_x4"]
        );
        add!(
            "qwen4_gemv_q8_0",
            kernels::QWEN4_GEMV_Q8_0_SRC,
            [
                "gemv_q8_0_k2560_staged",
                "gemv_q8_0_k2560_staged_pair",
                "gemv_q8_0_k2560_staged_rows",
                "gemv_q8_0_k2560_staged_rows_tiled",
                "gemv_q8_0_k2560_staged_rows_x2",
                "gemv_q8_0_k320_staged",
                "gemv_q8_0_k320_staged_rows",
                "gemv_q8_0_k8",
                "gemv_q8_0_k8_rows_r2",
                "gemv_q8_0_k8_rows_r3",
                "gemv_q8_0_k8_rows_r4",
                "gemv_q8_0_k8_rows_r5",
                "gemv_q8_0_k8_rows_r6",
                "gemv_q8_0_k8_rows_r7",
                "gemv_q8_0_k8_rows_r8",
                "quantize_bf16_q8_0",
                "topk8_partial_f32",
                "topk8_rescore_q8_0_k2560",
            ]
        );
        add!(
            "qwen4_hc_streams_init_from_embed_batched",
            kernels::QWEN4_HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC,
            ["hc_streams_init_from_embed_batched_bf16"]
        );
        add!(
            "qwen4_silu_mul",
            kernels::QWEN4_SILU_MUL_SRC,
            ["shared_expert_activation_bf16_f32", "silu_mul_bf16_rt_f32"]
        );
        add!(
            "requant_g256",
            kernels::REQUANT_G256_SRC,
            [
                "requant_bf16_to_f32",
                "requant_mqg256v2_to_f32",
                "requant_pack_mqg256v2",
                "requant_q8_0_to_f32"
            ]
        );
        add!(
            "topk8_rescore_mq6g256v2",
            kernels::TOPK8_RESCORE_MQ6G256V2_SRC,
            [
                "gemv_mq6g256v2",
                "gemv_mq6g256v2_x4",
                "topk8_rescore_mq6g256v2_k2560"
            ]
        );
        add!("zero_f32", kernels::ZERO_F32_SRC, ["zero_f32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1100") {
        add!(
            "gemv_bf16_xf32",
            kernels::gemv_bf16_xf32_src(arch == "gfx1151"),
            ["gemv_bf16_xf32"]
        );
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!(
            "gemv_mq2g256v2",
            kernels::GEMV_MQ2G256V2_SRC,
            ["gemv_mq2g256v2"]
        );
        add!(
            "gemv_mq6g256v2_mq6v2",
            kernels::GEMV_MQ6G256V2_SRC,
            ["gemv_mq6g256v2"]
        );
        if arch == "gfx1151" {
            add!(
                "gemv_mq6g256v2_k2560_mq6v2",
                kernels::GEMV_MQ6G256V2_K2560_SRC,
                ["gemv_mq6g256v2_k2560"]
            );
            add!(
                "gemv_mq6g256v2_k2560_u2_mq6v2",
                kernels::GEMV_MQ6G256V2_K2560_U2_SRC,
                ["gemv_mq6g256v2_k2560_u2"]
            );
        }

        // Default-on `HIPFIRE_QWEN4_ROUTER_FAST`: every eager multi-row router call.
        add!(
            "moe_router_softmax_top10_f32_fast",
            kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_FAST_SRC,
            ["moe_router_softmax_top10_f32_fast"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "copy_rows_strided_f32",
            kernels::COPY_ROWS_STRIDED_F32_SRC,
            ["copy_rows_strided_f32"]
        );
        add!(
            "gated_delta_chunk_wmma",
            crate::tensor_ops::GATED_DELTA_CHUNK_WMMA_SRC,
            ["gated_delta_chunk_gate_wmma"]
        );
        add!(
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151",
            kernels::GEMM_MQ4G128V2_MOE_GROUPED_WMMA_GFX1151_SRC,
            [
                "gemm_mq4g128v2_moe_grouped_wmma_gfx1151",
                "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out",
                "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out",
            ]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
            [
                "gemm_mq4g256v2_moe_grouped_wmma_k2",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            ]
        );
        add!(
            "gemm_mq6g256v2_residual_wmma",
            kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_SRC,
            ["gemm_mq6g256v2_residual_wmma"]
        );
        add!(
            "hyper_read_up_wmma",
            crate::tensor_ops::HYPER_READ_UP_WMMA_SRC,
            ["hyper_read_up_wmma_bf16", "hyper_read_up_wmma_bf16_swap"]
        );
        add!(
            "indexed_attention_dense_wmma",
            crate::tensor_ops::INDEXED_ATTENTION_DENSE_WMMA_SRC,
            [
                "indexed_attention_dense_wmma_f16",
                "indexed_attention_kv_f16",
            ]
        );
        add!(
            "qwen4_bf16_round_trip",
            kernels::QWEN4_BF16_ROUND_TRIP_SRC,
            ["bf16_round_trip_f32_strided"]
        );
        add!(
            "qwen4_gemm_mqv2_wmma_gfx11_bt",
            kernels::QWEN4_GEMM_MQV2_WMMA_GFX11_BT_SRC,
            [
                "gemm_gate_up_mq2g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt6",
                "gemm_mq2g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt8",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt8",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt8",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt8_x4",
                "gemm_mq6g256v2_wmma_gfx11_bt8_x4",
                "gemm_mq6g256v2_wmma_gfx11_bt8_x4_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw",
                "gemm_mq6g256v2_wmma_gfx11_bt8_x4_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_dq",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_dq_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_dq",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_dq_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_regions",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_dq",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_dq_bf16out",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_hcw",
                "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_regions",
                "gemm_qkv_mq2g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq5g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq5g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq2g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq5g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq5g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt4",
            ]
        );
        // One module, a different symbol per launcher: the installed object's
        // index must list every kernel the source defines for the arch (the Q8
        // QSA exports included); `tensor_ops_declares_exactly_the_kernels_its_source_defines`.
        add!(
            "tensor_ops",
            crate::tensor_ops::TENSOR_OPS_SRC,
            [
                "argmax_f32",
                "bf16_roundtrip_f32",
                "bf16_scaled_add_batched_f32",
                "bf16_scaled_add_f32",
                "copy_regions_u32",
                "gated_delta_conv_bf16_f32",
                "gated_delta_conv_bf16_f32_batched_k4",
                "gated_delta_conv_params_bf16_f32",
                "gated_delta_conv_qknorm_bf16_f32_batched_k4",
                "gated_delta_gate_bf16_f32",
                "gated_delta_gate_bf16_f32_batched",
                "gated_delta_gate_rotate_bf16_f32_batched",
                "gated_delta_params_bf16_f32",
                "gated_delta_params_bf16_f32_batched",
                "gated_delta_params_f32",
                "gated_delta_qk_norm_bf16_batched",
                "gated_delta_rollback_layers_f32",
                "gated_delta_rollback_layers_q8",
                "gated_delta_step_f32",
                "gated_delta_step_gate_norm128_gfx1151",
                "gated_delta_step_gate_norm128_q8",
                "gated_delta_step_gate_norm128_q8ef",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "gated_delta_step_halves_state128_persistent256_capture_f32",
                "gated_delta_step_halves_state128_persistent256_capture_q8",
                "gated_delta_step_halves_state128_persistent256_capture_q8ef",
                "gated_delta_step_halves_state128_persistent256_f32",
                "gated_delta_step_halves_state128_persistent256_q8",
                "gated_delta_step_halves_state128_persistent256_q8ef",
                "gated_delta_step_norm128_q8",
                "gated_delta_step_norm128_q8ef",
                "gated_delta_step_shared_norm128_gfx1151",
                "gdn_q8ef_residual_zero",
                "gdn_state_f32_to_q8",
                "gdn_state_f32_to_q8ef",
                "gdn_state_q8_to_f32",
                "gdn_state_q8ef_to_f32",
                "hc_activation_fused_f32",
                "hc_state_bf16_add_f32",
                "hc_state_bf16_to_f32",
                "hyper_norm_f32",
                "hyper_norm_gate_f32",
                "hyper_norm_gate_outputs",
                "hyper_norm_gate_outputs_f32",
                "hyper_read_f32",
                "hyper_read_projected_f32",
                "hyper_read_up_fused_f32",
                "hyper_write_bf16x2",
                "hyper_write_f32",
                "indexed_attention_attention_f32",
                "indexed_attention_attention_f32_batched",
                "indexed_attention_attention_f32_batched_hg12",
                "indexed_attention_attention_f32_batched_hg4",
                "indexed_attention_attention_f32_batched_serial",
                "indexed_attention_attention_f32_serial",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_f32",
                "indexed_attention_cache_append_f32_batched",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_f32",
                "indexed_attention_decode_prologue_q8",
                "indexed_attention_index_key_append_bf16_batched",
                "indexed_attention_norm_rope_f32",
                "indexed_attention_norm_rope_f32_batched",
                "indexed_attention_pool_rope_bf16",
                "indexed_attention_pool_rope_f32",
                "indexed_attention_reuse_selection",
                "indexed_attention_rope_table_f32",
                "indexed_attention_select_bf16_batched",
                "indexed_attention_select_bf16_batched_serial",
                "indexed_attention_select_f32",
                "indexed_attention_select_f32_batched",
                "indexed_attention_select_f32_batched_serial",
                "indexed_attention_select_f32_serial",
                "indexed_attention_select_from_scores",
                "indexed_attention_select_scores_rows16_f32",
                "indexed_attention_select_scores_rows32_f32",
                "indexed_attention_select_scores_rows8_f32",
                "scale_f32",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_q8",
            ]
        );
    }
    if arch == "gfx1201" {
        // Keep the actual Q8 singleton and narrow-row HIP oracle exports.
        add!(
            "qwen4_gemv_q8_0_wide",
            kernels::QWEN4_GEMV_Q8_0_WIDE_SRC,
            [
                "gemv_q8_0_wide_k640_staged",
                "gemv_q8_0_wide_k640_staged_bf16_scaled_add",
                "gemv_q8_0_wide_rows",
            ]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
            kernels::GEMM_MQ4G256V2_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_mq4g256v2_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_mq6g256v2_residual_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            [
                "gemm_mq6g256v2_residual_wmma_gfx12_bt12",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
                "gemm_mq6g256v2_wmma_gfx12_bt12_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt4_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
            ]
        );
        add!(
            "gemm_mq6g256v2_residual_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            [
                "gemm_mq6g256v2_residual_wmma_gfx12_bt12",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
                "gemm_mq6g256v2_wmma_gfx12_bt12_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt4_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
            ]
        );
        add!(
            "gemm_mq6g256v2_residual_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            [
                "gemm_mq6g256v2_residual_wmma_gfx12_bt12",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
                "gemm_mq6g256v2_wmma_gfx12_bt12_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt4_hcw",
                "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
            ]
        );
        add!(
            "gemm_mq6g256v2_residual_wmma_gfx12_mq5v2",
            kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_SRC,
            ["gemm_mq6g256v2_residual_wmma_gfx12"]
        );
        // Exact-INT8 MQ6G256V2 prefill (default on gfx1201 in the Qwen4 forward,
        // `gemm_mq6_i8.rs`): the GEMM module and the A8 `block_i8_128` quantizer
        // (`Gpu::ensure_int8_mmq_x`) it consumes.
        add!(
            "gemm_mq6g256v2_mmq_i8_gfx12",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC,
            [
                "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12",
                "gemm_mq6g256v2_mmq_i8_set_gfx12"
            ]
        );
        // Qwen4 shared-expert INT8 route (`HIPFIRE_QWEN4_SHARED_INT8`, `shared_i8.rs`): the
        // int8 x int8 GEMM and the BF16 -> FWHT -> A8 producers (weight shadows, hidden).
        add!(
            "mq_rotate_x_i4_f16out_i8",
            kernels::MQ_ROTATE_X_I4_F16OUT_I8_SRC,
            ["mq_rotate_x_i4_f16out_i8"]
        );
        add!(
            "qwen4_shared_i8_gemm_gfx12",
            kernels::QWEN4_SHARED_I8_GEMM_GFX12_SRC,
            [
                "qwen4_shared_i8_down_set_f32",
                "qwen4_shared_i8_gate_up_silu_bf16"
            ]
        );
        add!(
            "qwen4_shared_i8_quant_gfx12",
            kernels::QWEN4_SHARED_I8_QUANT_GFX12_SRC,
            ["qwen4_shared_rot128_i8", "qwen4_shared_rot256_i8_w"]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_i8_gfx12",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_I8_GFX12_SRC,
            [
                "gemm_mq4g256v2_gate_up_silu_mmq_i8",
                "gemm_mq4g256v2_residual_mmq_i8_full_add",
                "gemm_mq4g256v2_residual_mmq_i8_full_set",
                "quantize_int8_mmq_ds128"
            ]
        );
        // Symmetric qt49 MQ3V25 A4 (default on gfx1201 for qt49 artifacts) and
        // the opt-in A8 consumer (`HIPFIRE_MQ3V25_A8=1`).
        add!(
            "gemm_mq3v25_residual_mmq_iu4_gfx1201",
            kernels::GEMM_MQ3V25_RESIDUAL_MMQ_IU4_GFX12_SRC,
            [
                "gemm_mq3v25_gate_up_silu_mmq_iu4",
                "gemm_mq3v25_residual_mmq_iu4"
            ]
        );
        add!(
            "gemm_mq3v25_residual_mmq_iu8_gfx1201",
            kernels::GEMM_MQ3V25_RESIDUAL_MMQ_IU8_GFX12_SRC,
            ["gemm_mq3v25_residual_mmq_iu8"]
        );
        // One-pass plain rotate + A8 `block_i8_128` producer of the same route
        // (`Gpu::rotate_x_mq_i8_batched`, `Gpu::mq6_i8_prepare_x`).
        add!(
            "mq_rotate_x_i8_gfx12",
            kernels::MQ_ROTATE_X_I8_GFX12_SRC,
            ["mq_rotate_x_i8_gfx12"]
        );
        add!(
            "qwen4_gemm_mq6g256v2_wmma_gfx12_x4",
            kernels::QWEN4_GEMM_MQ6G256V2_WMMA_GFX12_X4_SRC,
            [
                "gemm_mq6g256v2_residual_wmma_gfx12_bt8_x4",
                "gemm_mq6g256v2_wmma_gfx12_bt8_x4",
                "gemm_mq6g256v2_wmma_gfx12_bt8_x4_bf16out",
                "gemm_mq6g256v2_wmma_gfx12_bt8_x4_regions",
            ]
        );
        // One module, a different symbol per launcher: the installed object's
        // index must list every kernel the source defines for the arch (the Q8
        // QSA exports included); `tensor_ops_declares_exactly_the_kernels_its_source_defines`.
        add!(
            "tensor_ops",
            crate::tensor_ops::TENSOR_OPS_SRC,
            [
                "argmax_f32",
                "bf16_roundtrip_f32",
                "bf16_scaled_add_batched_f32",
                "bf16_scaled_add_f32",
                "copy_regions_u32",
                "gated_delta_conv_bf16_f32",
                "gated_delta_conv_bf16_f32_batched_k4",
                "gated_delta_conv_params_bf16_f32",
                "gated_delta_conv_qknorm_bf16_f32_batched_k4",
                "gated_delta_gate_bf16_f32",
                "gated_delta_gate_bf16_f32_batched",
                "gated_delta_gate_rotate_bf16_f32_batched",
                "gated_delta_params_bf16_f32",
                "gated_delta_params_bf16_f32_batched",
                "gated_delta_params_f32",
                "gated_delta_qk_norm_bf16_batched",
                "gated_delta_rollback_layers_f32",
                "gated_delta_rollback_layers_q8",
                "gated_delta_step_f32",
                "gated_delta_step_gate_norm128_gfx1151",
                "gated_delta_step_gate_norm128_q8",
                "gated_delta_step_gate_norm128_q8ef",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "gated_delta_step_halves_state128_persistent256_capture_f32",
                "gated_delta_step_halves_state128_persistent256_capture_q8",
                "gated_delta_step_halves_state128_persistent256_capture_q8ef",
                "gated_delta_step_halves_state128_persistent256_f32",
                "gated_delta_step_halves_state128_persistent256_q8",
                "gated_delta_step_halves_state128_persistent256_q8ef",
                "gated_delta_step_norm128_q8",
                "gated_delta_step_norm128_q8ef",
                "gated_delta_step_pipe128_f32",
                "gated_delta_step_pipe128_q8",
                "gated_delta_step_pipe128_q8ef",
                "gated_delta_step_shared_norm128_gfx1151",
                "gdn_q8ef_residual_zero",
                "gdn_state_f32_to_q8",
                "gdn_state_f32_to_q8ef",
                "gdn_state_q8_to_f32",
                "gdn_state_q8ef_to_f32",
                "hc_activation_fused_f32",
                "hc_state_bf16_add_f32",
                "hc_state_bf16_to_f32",
                "hyper_norm_f32",
                "hyper_norm_gate_f32",
                "hyper_norm_gate_outputs",
                "hyper_norm_gate_outputs_f32",
                "hyper_read_f32",
                "hyper_read_projected_f32",
                "hyper_read_up_fused_f32",
                "hyper_write_bf16x2",
                "hyper_write_bf16x8_gfx1201",
                "hyper_write_f32",
                "indexed_attention_attention_f32",
                "indexed_attention_attention_f32_batched",
                "indexed_attention_attention_f32_batched_hg12",
                "indexed_attention_attention_f32_batched_hg4",
                "indexed_attention_attention_f32_batched_serial",
                "indexed_attention_attention_f32_serial",
                "indexed_attention_attention_fp8",
                "indexed_attention_attention_fp8_batched",
                "indexed_attention_attention_fp8_batched_hg4",
                "indexed_attention_attention_fp8_batched_serial",
                "indexed_attention_attention_fp8_serial",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_f32",
                "indexed_attention_cache_append_f32_batched",
                "indexed_attention_cache_append_fp8_batched",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_f32",
                "indexed_attention_decode_prologue_fp8",
                "indexed_attention_decode_prologue_q8",
                "indexed_attention_index_key_append_bf16_batched",
                "indexed_attention_norm_rope_f32",
                "indexed_attention_norm_rope_f32_batched",
                "indexed_attention_pool_rope_bf16",
                "indexed_attention_pool_rope_f32",
                "indexed_attention_reuse_selection",
                "indexed_attention_rope_table_f32",
                "indexed_attention_select_bf16_batched",
                "indexed_attention_select_bf16_batched_serial",
                "indexed_attention_select_f32",
                "indexed_attention_select_f32_batched",
                "indexed_attention_select_f32_batched_serial",
                "indexed_attention_select_f32_serial",
                "indexed_attention_select_from_scores",
                "indexed_attention_select_scores_rows16_f32",
                "indexed_attention_select_scores_rows32_f32",
                "indexed_attention_select_scores_rows8_f32",
                "scale_f32",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_q8",
            ]
        );
        // Default >= 512-row prefill: BF16 -> F16 WMMA projections (HC
        // input_mix_down 160 x 80 split-K tile + reducer, router/shared 128 x
        // 128, wider projections 128 x 256), the HC read-up tile, and the
        // symmetric-IU4 MoE load check, A4 producers and stable scatter.
        add!(
            "gemm_wmma_lds_splitk",
            kernels::GEMM_F16_X_F16_WMMA_LDS_SPLITK_SRC,
            [
                "qwen4_wmma_lds_128_128_32_64_k64",
                "qwen4_wmma_lds_128_256_32_64_k64",
                "qwen4_wmma_lds_128_256_32_64_k64_bsr",
                "qwen4_wmma_lds_160_80_32_80_k32_p_s4",
                "qwen4_wmma_lds_64_128_32_32_k64_p_s4",
                "qwen4_wmma_lds_64_128_32_64_k64_p",
                "qwen4_wmma_lds_64_128_32_64_k64_p_s2",
                "qwen4_wmma_lds_64_128_32_64_k64_p_s4",
                "qwen4_wmma_lds_64_128_32_64_k64_p_s8",
                "qwen4_wmma_lds_64_128_32_64_k64_s4",
                "qwen4_wmma_lds_64_64_32_32_k64_p_s4",
                "qwen4_wmma_lds_64_64_32_64_k64_p",
                "qwen4_wmma_lds_64_64_32_64_k64_p_s2",
                "qwen4_wmma_lds_64_64_32_64_k64_p_s4",
                "qwen4_wmma_lds_reduce",
                // Arch17 pitched projections: gfx1201-only, unsplit ascending K.
                "qwen4_wmma_lds_64_64_32_64_k64_p_ld",
                "qwen4_wmma_lds_64_128_32_64_k64_p_ld",
                "qwen4_wmma_lds_128_128_32_64_k64_ld",
                "qwen4_wmma_lds_128_256_32_64_k64_ld",
            ]
        );
        add!(
            "qwen4_hc_down_fused",
            kernels::QWEN4_HC_DOWN_FUSED_SRC,
            ["qwen4_hc_down_fused_b64_k32"]
        );
        add!(
            "hyper_read_up_wmma_gfx1201",
            crate::tensor_ops::HYPER_READ_UP_WMMA_GFX1201_SRC,
            [
                "hyper_read_up_wmma_bf16_gfx1201",
                "hyper_read_up_wmma_bf16_gfx1201_t128",
                "hyper_read_up_wmma_bf16_gfx1201_v2",
            ]
        );
        add!(
            "mq_rotate_x_i4",
            kernels::MQ_ROTATE_X_I4_SRC,
            ["mq_rotate_x_i4"]
        );
        add!(
            "mq_rotate_x_i4_f16out",
            kernels::MQ_ROTATE_X_I4_F16OUT_SRC,
            ["mq_rotate_x_i4_f16out"]
        );
        add!(
            "qwen4_moe_rotate128_i4",
            kernels::QWEN4_MOE_ROTATE128_I4_SRC,
            ["qwen4_moe_rotate128_i4"]
        );
        add!(
            "qwen4_moe_scatter_stable_top10",
            kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_SRC,
            [
                "qwen4_moe_group_prefix",
                "qwen4_moe_group_ranks",
                "qwen4_moe_group_scatter",
            ]
        );
        add!(
            "qwen4_moe_sym_check_gfx1201",
            kernels::QWEN4_MOE_SYM_CHECK_GFX1201_SRC,
            ["qwen4_moe_sym_check_gfx1201"]
        );
        add!(
            "qwen4_slab_union_plan",
            kernels::QWEN4_SLAB_UNION_PLAN_SRC,
            ["qwen4_slab_union_plan"]
        );
    }
    if arch == "gfx1100" {
        // Symmetric qt49 MQ3V25 A4 full-tile consumer (`HIPFIRE_MQ3V25_A4=1`,
        // `Gpu::gemm_mq3v25_residual_iu4`).
        add!(
            "gemm_mq3v25_iu4_v2c_gfx1100",
            kernels::GEMM_MQ3V25_RESIDUAL_IU4_V2C_GFX11_SRC,
            ["gemm_mq3v25_residual_iu4_v2c"]
        );
        add!(
            "qwen4_moe_scatter_stable_top10_gfx1100",
            kernels::qwen4_moe_scatter_stable_top10_gfx1100_src(),
            [
                "qwen4_moe_group_ranks",
                "qwen4_moe_group_prefix",
                "qwen4_moe_group_scatter",
            ]
        );
        add!(
            "qwen4_moe_sym_check_gfx1100",
            kernels::qwen4_moe_sym_check_gfx1100_src(),
            ["qwen4_moe_sym_check_gfx1100"]
        );
        add!(
            "qwen4_slab_union_plan_gfx1100",
            kernels::qwen4_slab_union_plan_gfx1100_src(),
            ["qwen4_slab_union_plan"]
        );
        add!(
            "add",
            kernels::ADD_SRC,
            ["add_f32", "broadcast_add_rows_f32"]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
            kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
            [
                "gemm_mq4g256v2_moe_grouped_wmma_k2",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            ]
        );
        add!(
            "gemm_mqv2_wmma_gfx1100_bt",
            kernels::gemm_mqv2_wmma_gfx11_bt_src(arch == "gfx1151"),
            [
                "gemm_gate_up_mq2g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq5g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt6",
                "gemm_mq2g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt8",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq5g256v2_residual_wmma_gfx11_bt8",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt8",
                "gemm_qkv_mq2g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq5g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq5g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq2g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq5g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq5g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt12",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt4",
            ]
        );
        add!(
            "gemm_wmma_lds256",
            kernels::gemm_f16_x_f16_wmma_lds256_src(arch == "gfx1151"),
            [
                "gemm_wmma_lds_128_128_32_64_k64",
                "gemm_wmma_lds_128_128_32_64_k64_a",
                "gemm_wmma_lds_128_128_32_64_k64_gr",
                "gemm_wmma_lds_128_128_32_64_k64_gra",
                "gemm_wmma_lds_128_128_32_64_k64_o16",
                "gemm_wmma_lds_128_128_32_64_k64_o16g",
                "gemm_wmma_lds_128_128_32_64_k64_p",
                "gemm_wmma_lds_128_128_32_64_k64_p_a",
                "gemm_wmma_lds_128_128_32_64_k64_p_gr",
                "gemm_wmma_lds_128_128_32_64_k64_p_gra",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_128_64_64_k64",
                "gemm_wmma_lds_128_128_64_64_k64_a",
                "gemm_wmma_lds_128_128_64_64_k64_gr",
                "gemm_wmma_lds_128_128_64_64_k64_gra",
                "gemm_wmma_lds_128_128_64_64_k64_o16",
                "gemm_wmma_lds_128_128_64_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64",
                "gemm_wmma_lds_128_256_32_64_k64_a",
                "gemm_wmma_lds_128_256_32_64_k64_gr",
                "gemm_wmma_lds_128_256_32_64_k64_gra",
                "gemm_wmma_lds_128_256_32_64_k64_o16",
                "gemm_wmma_lds_128_256_32_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64_p",
                "gemm_wmma_lds_128_256_32_64_k64_p_a",
                "gemm_wmma_lds_128_256_32_64_k64_p_gr",
                "gemm_wmma_lds_128_256_32_64_k64_p_gra",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k32",
                "gemm_wmma_lds_128_256_64_64_k32_a",
                "gemm_wmma_lds_128_256_64_64_k32_gr",
                "gemm_wmma_lds_128_256_64_64_k32_gra",
                "gemm_wmma_lds_128_256_64_64_k32_o16",
                "gemm_wmma_lds_128_256_64_64_k32_o16g",
                "gemm_wmma_lds_128_256_64_64_k32_p",
                "gemm_wmma_lds_128_256_64_64_k32_p_a",
                "gemm_wmma_lds_128_256_64_64_k32_p_gr",
                "gemm_wmma_lds_128_256_64_64_k32_p_gra",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k64",
                "gemm_wmma_lds_128_256_64_64_k64_sw",
                "gemm_wmma_lds_128_512_32_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32_sw",
                "gemm_wmma_lds_256_128_32_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64_sw",
                "gemm_wmma_lds_256_256_32_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64_a",
                "gemm_wmma_lds_256_256_64_64_k64_gr",
                "gemm_wmma_lds_256_256_64_64_k64_gra",
                "gemm_wmma_lds_256_256_64_64_k64_o16",
                "gemm_wmma_lds_256_256_64_64_k64_o16g",
                "gemm_wmma_lds_256_256_64_64_k64_p",
                "gemm_wmma_lds_256_256_64_64_k64_p_a",
                "gemm_wmma_lds_256_256_64_64_k64_p_gr",
                "gemm_wmma_lds_256_256_64_64_k64_p_gra",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16g",
                "gemm_wmma_lds_64_512_64_64_k32",
            ]
        );
        add!(
            "gemv_mq2g256v2_rdna3",
            kernels::GEMV_MQ2G256V2_SRC,
            ["gemv_mq2g256v2"]
        );
        add!(
            "gemv_mq6g256v2_rdna3_mq6v2",
            kernels::GEMV_MQ6G256V2_SRC,
            ["gemv_mq6g256v2"]
        );
        add!(
            "qwen4_gemm_wmma_lds256",
            kernels::QWEN4_GEMM_F16_X_F16_WMMA_LDS256_SRC,
            [
                "gemm_f16_x_f16_wmma_lds_regions_128_128_32_64_k64_p",
                "gemm_wmma_lds_128_128_32_64_k64",
                "gemm_wmma_lds_128_128_32_64_k64_a",
                "gemm_wmma_lds_128_128_32_64_k64_bsr",
                "gemm_wmma_lds_128_128_32_64_k64_gr",
                "gemm_wmma_lds_128_128_32_64_k64_gra",
                "gemm_wmma_lds_128_128_32_64_k64_hcsd",
                "gemm_wmma_lds_128_128_32_64_k64_o16",
                "gemm_wmma_lds_128_128_32_64_k64_o16g",
                "gemm_wmma_lds_128_128_32_64_k64_p",
                "gemm_wmma_lds_128_128_32_64_k64_p_a",
                "gemm_wmma_lds_128_128_32_64_k64_p_bsr",
                "gemm_wmma_lds_128_128_32_64_k64_p_gr",
                "gemm_wmma_lds_128_128_32_64_k64_p_gra",
                "gemm_wmma_lds_128_128_32_64_k64_p_hcsd",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_128_64_64_k64",
                "gemm_wmma_lds_128_128_64_64_k64_a",
                "gemm_wmma_lds_128_128_64_64_k64_bsr",
                "gemm_wmma_lds_128_128_64_64_k64_gr",
                "gemm_wmma_lds_128_128_64_64_k64_gra",
                "gemm_wmma_lds_128_128_64_64_k64_o16",
                "gemm_wmma_lds_128_128_64_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64",
                "gemm_wmma_lds_128_256_32_64_k64_a",
                "gemm_wmma_lds_128_256_32_64_k64_bf16st",
                "gemm_wmma_lds_128_256_32_64_k64_bsr",
                "gemm_wmma_lds_128_256_32_64_k64_gr",
                "gemm_wmma_lds_128_256_32_64_k64_gra",
                "gemm_wmma_lds_128_256_32_64_k64_hcsd",
                "gemm_wmma_lds_128_256_32_64_k64_o16",
                "gemm_wmma_lds_128_256_32_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64_p",
                "gemm_wmma_lds_128_256_32_64_k64_p_a",
                "gemm_wmma_lds_128_256_32_64_k64_p_bsr",
                "gemm_wmma_lds_128_256_32_64_k64_p_gr",
                "gemm_wmma_lds_128_256_32_64_k64_p_gra",
                "gemm_wmma_lds_128_256_32_64_k64_p_hcsd",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k32",
                "gemm_wmma_lds_128_256_64_64_k32_a",
                "gemm_wmma_lds_128_256_64_64_k32_bsr",
                "gemm_wmma_lds_128_256_64_64_k32_gr",
                "gemm_wmma_lds_128_256_64_64_k32_gra",
                "gemm_wmma_lds_128_256_64_64_k32_o16",
                "gemm_wmma_lds_128_256_64_64_k32_o16g",
                "gemm_wmma_lds_128_256_64_64_k32_p",
                "gemm_wmma_lds_128_256_64_64_k32_p_a",
                "gemm_wmma_lds_128_256_64_64_k32_p_bsr",
                "gemm_wmma_lds_128_256_64_64_k32_p_gr",
                "gemm_wmma_lds_128_256_64_64_k32_p_gra",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k64",
                "gemm_wmma_lds_128_256_64_64_k64_sw",
                "gemm_wmma_lds_128_512_32_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32_sw",
                "gemm_wmma_lds_160_64_32_64_k64",
                "gemm_wmma_lds_160_64_32_64_k64_p",
                "gemm_wmma_lds_256_128_32_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64_sw",
                "gemm_wmma_lds_256_256_32_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64_a",
                "gemm_wmma_lds_256_256_64_64_k64_bsr",
                "gemm_wmma_lds_256_256_64_64_k64_gr",
                "gemm_wmma_lds_256_256_64_64_k64_gra",
                "gemm_wmma_lds_256_256_64_64_k64_hcsd",
                "gemm_wmma_lds_256_256_64_64_k64_o16",
                "gemm_wmma_lds_256_256_64_64_k64_o16g",
                "gemm_wmma_lds_256_256_64_64_k64_p",
                "gemm_wmma_lds_256_256_64_64_k64_p_a",
                "gemm_wmma_lds_256_256_64_64_k64_p_bsr",
                "gemm_wmma_lds_256_256_64_64_k64_p_gr",
                "gemm_wmma_lds_256_256_64_64_k64_p_gra",
                "gemm_wmma_lds_256_256_64_64_k64_p_hcsd",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16g",
                "gemm_wmma_lds_64_512_64_64_k32",
                "gemm_wmma_lds_64_64_32_64_k64",
                "gemm_wmma_lds_64_64_32_64_k64_p",
                "gemm_wmma_lds_80_32_80_16_k64_p",
            ]
        );
    }
    if arch == "gfx1151" {
        add!(
            "gemm_wmma_lds256",
            kernels::gemm_f16_x_f16_wmma_lds256_src(arch == "gfx1151"),
            [
                "gemm_f16_x_f16_wmma_lds_regions_128_128_32_64_k64_p",
                "gemm_wmma_lds_128_128_32_64_k64",
                "gemm_wmma_lds_128_128_32_64_k64_a",
                "gemm_wmma_lds_128_128_32_64_k64_bsr",
                "gemm_wmma_lds_128_128_32_64_k64_gr",
                "gemm_wmma_lds_128_128_32_64_k64_gra",
                "gemm_wmma_lds_128_128_32_64_k64_hcsd",
                "gemm_wmma_lds_128_128_32_64_k64_o16",
                "gemm_wmma_lds_128_128_32_64_k64_o16g",
                "gemm_wmma_lds_128_128_32_64_k64_p",
                "gemm_wmma_lds_128_128_32_64_k64_p_a",
                "gemm_wmma_lds_128_128_32_64_k64_p_bsr",
                "gemm_wmma_lds_128_128_32_64_k64_p_gr",
                "gemm_wmma_lds_128_128_32_64_k64_p_gra",
                "gemm_wmma_lds_128_128_32_64_k64_p_hcsd",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16",
                "gemm_wmma_lds_128_128_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_128_64_64_k64",
                "gemm_wmma_lds_128_128_64_64_k64_a",
                "gemm_wmma_lds_128_128_64_64_k64_bsr",
                "gemm_wmma_lds_128_128_64_64_k64_gr",
                "gemm_wmma_lds_128_128_64_64_k64_gra",
                "gemm_wmma_lds_128_128_64_64_k64_o16",
                "gemm_wmma_lds_128_128_64_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64",
                "gemm_wmma_lds_128_256_32_64_k64_a",
                "gemm_wmma_lds_128_256_32_64_k64_bf16st",
                "gemm_wmma_lds_128_256_32_64_k64_bsr",
                "gemm_wmma_lds_128_256_32_64_k64_gr",
                "gemm_wmma_lds_128_256_32_64_k64_gra",
                "gemm_wmma_lds_128_256_32_64_k64_hcsd",
                "gemm_wmma_lds_128_256_32_64_k64_o16",
                "gemm_wmma_lds_128_256_32_64_k64_o16g",
                "gemm_wmma_lds_128_256_32_64_k64_p",
                "gemm_wmma_lds_128_256_32_64_k64_p_a",
                "gemm_wmma_lds_128_256_32_64_k64_p_bsr",
                "gemm_wmma_lds_128_256_32_64_k64_p_gr",
                "gemm_wmma_lds_128_256_32_64_k64_p_gra",
                "gemm_wmma_lds_128_256_32_64_k64_p_hcsd",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16",
                "gemm_wmma_lds_128_256_32_64_k64_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k32",
                "gemm_wmma_lds_128_256_64_64_k32_a",
                "gemm_wmma_lds_128_256_64_64_k32_bsr",
                "gemm_wmma_lds_128_256_64_64_k32_gr",
                "gemm_wmma_lds_128_256_64_64_k32_gra",
                "gemm_wmma_lds_128_256_64_64_k32_o16",
                "gemm_wmma_lds_128_256_64_64_k32_o16g",
                "gemm_wmma_lds_128_256_64_64_k32_p",
                "gemm_wmma_lds_128_256_64_64_k32_p_a",
                "gemm_wmma_lds_128_256_64_64_k32_p_bsr",
                "gemm_wmma_lds_128_256_64_64_k32_p_gr",
                "gemm_wmma_lds_128_256_64_64_k32_p_gra",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16",
                "gemm_wmma_lds_128_256_64_64_k32_p_o16g",
                "gemm_wmma_lds_128_256_64_64_k64",
                "gemm_wmma_lds_128_256_64_64_k64_sw",
                "gemm_wmma_lds_128_512_32_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32",
                "gemm_wmma_lds_128_512_64_64_k32_sw",
                "gemm_wmma_lds_160_64_32_64_k64",
                "gemm_wmma_lds_160_64_32_64_k64_p",
                "gemm_wmma_lds_256_128_32_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64",
                "gemm_wmma_lds_256_128_64_64_k64_sw",
                "gemm_wmma_lds_256_256_32_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64",
                "gemm_wmma_lds_256_256_64_64_k64_a",
                "gemm_wmma_lds_256_256_64_64_k64_bsr",
                "gemm_wmma_lds_256_256_64_64_k64_gr",
                "gemm_wmma_lds_256_256_64_64_k64_gra",
                "gemm_wmma_lds_256_256_64_64_k64_hcsd",
                "gemm_wmma_lds_256_256_64_64_k64_o16",
                "gemm_wmma_lds_256_256_64_64_k64_o16g",
                "gemm_wmma_lds_256_256_64_64_k64_p",
                "gemm_wmma_lds_256_256_64_64_k64_p_a",
                "gemm_wmma_lds_256_256_64_64_k64_p_bsr",
                "gemm_wmma_lds_256_256_64_64_k64_p_gr",
                "gemm_wmma_lds_256_256_64_64_k64_p_gra",
                "gemm_wmma_lds_256_256_64_64_k64_p_hcsd",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16",
                "gemm_wmma_lds_256_256_64_64_k64_p_o16g",
                "gemm_wmma_lds_64_512_64_64_k32",
                "gemm_wmma_lds_64_64_32_64_k64",
                "gemm_wmma_lds_64_64_32_64_k64_p",
            ]
        );
        add!(
            "gemv_bf16_xf32",
            kernels::gemv_bf16_xf32_src(arch == "gfx1151"),
            [
                "gemv_bf16_xf32",
                "gemv_bf16_xf32_bf16_scaled_add",
                "gemv_bf16_xf32_k4",
                "gemv_bf16_xf32_k4_rows_r2",
                "gemv_bf16_xf32_k4_rows_r3",
                "gemv_bf16_xf32_k4_rows_r4",
                "gemv_bf16_xf32_k4_rows_r5",
                "gemv_bf16_xf32_k4_rows_r6",
                "gemv_bf16_xf32_k4_rows_r7",
                "gemv_bf16_xf32_k4_rows_r8",
                "gemv_bf16_xf32_k4_rows_tiled_r2",
                "gemv_bf16_xf32_k4_rows_tiled_r3",
                "gemv_bf16_xf32_k4_rows_tiled_r4",
                "gemv_bf16_xf32_k4_rows_tiled_r5",
                "gemv_bf16_xf32_k4_rows_tiled_r6",
                "gemv_bf16_xf32_k4_rows_tiled_r7",
                "gemv_bf16_xf32_k4_rows_tiled_r8",
                "gemv_bf16_xf32_x4",
                "gemv_bf16_xf32_x4_rows_r2",
                "gemv_bf16_xf32_x4_rows_r3",
                "gemv_bf16_xf32_x4_rows_r4",
                "gemv_bf16_xf32_x4_rows_r5",
                "gemv_bf16_xf32_x4_rows_r6",
                "gemv_bf16_xf32_x4_rows_r7",
                "gemv_bf16_xf32_x4_rows_r8",
                "hyper_write_norm_f32",
            ]
        );
        add!(
            "mq_rotate_x_i4",
            kernels::MQ_ROTATE_X_I4_SRC,
            ["mq_rotate_x_i4"]
        );
        // Qwen4 MoE front producer (HIPFIRE_QWEN4_MOE_FRONT_FUSE): the F16-output twin.
        add!(
            "mq_rotate_x_i4_f16out",
            kernels::MQ_ROTATE_X_I4_F16OUT_SRC,
            ["mq_rotate_x_i4_f16out"]
        );
        add!(
            "qwen4_gemv_q8_0_wide",
            kernels::QWEN4_GEMV_Q8_0_WIDE_SRC,
            [
                "gemv_q8_0_wide_k640_staged",
                "gemv_q8_0_wide_k640_staged_bf16_scaled_add",
                "gemv_q8_0_wide_rows",
            ]
        );
        add!(
            "qwen4_moe_iu4_sym_gfx1151",
            kernels::QWEN4_MOE_IU4_SYM_GFX1151_SRC,
            [
                "qwen4_moe_down_iu4_sym_gfx1151",
                "qwen4_moe_down_iu4_sym_gfx1151_nt4",
                "qwen4_moe_gate_up_silu_iu4_sym_gfx1151",
                "qwen4_moe_gate_up_silu_iu4_sym_gfx1151_nt4",
                "qwen4_moe_sym_check_gfx1151",
            ]
        );
        add!(
            "qwen4_moe_rotate128_i4",
            kernels::QWEN4_MOE_ROTATE128_I4_SRC,
            ["qwen4_moe_rotate128_i4"]
        );
        add!(
            "qwen4_moe_scatter_stable_top10",
            kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_SRC,
            [
                "qwen4_moe_group_prefix",
                "qwen4_moe_group_ranks",
                "qwen4_moe_group_scatter",
            ]
        );
        add!(
            "qwen4_slab_union_plan",
            kernels::QWEN4_SLAB_UNION_PLAN_SRC,
            ["qwen4_slab_union_plan"]
        );
        add!(
            "qwen4_slab_tile_mask",
            kernels::QWEN4_SLAB_TILE_MASK_SRC,
            ["qwen4_slab_tile_mask"]
        );
        // Default-on `HIPFIRE_QWEN4_HC_ROW_FOLD`: grouped MoE prefill fold.
        add!(
            "hc_row_fold",
            crate::hc_row_fold::SRC,
            ["hc_row_fold_norm_gate"]
        );
    }
    // Default-on `HIPFIRE_QWEN4_HC_ROW_FOLD` on gfx1201: the MoE-tail and mixer-tail row
    // folds, in the same module (the `__GFX12__` entries are compiled for gfx12 only).
    if arch == "gfx1201" {
        add!(
            "hc_row_fold",
            crate::hc_row_fold::SRC,
            [
                "hc_row_fold_norm_gate",
                "hc_row_fold_moe_gfx1201",
                "hc_row_write_norm_gate_gfx1201",
            ]
        );
    }
    // Qwen3.5-MoE (ornith-1.5-35b-a3b) load, AR prefill/decode and native MTP,
    // plus Qwen3.5 dense (H2): every module a kernel-load trace
    // (tests/fixtures/kernel-trace-qwen35.tsv) JIT-compiled on gfx1100,
    // gfx1151 and gfx1201 beyond the inventory above. Sources are the exact
    // expressions the callsites pass to `ensure_kernel`; symbols are every
    // kernel the arch's object defines.
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!(
            "gated_delta_net_q8_compact2_b2",
            kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC,
            [
                "gated_delta_net_q8_compact2_b2",
                "gated_delta_net_q8_fast_independent_masked",
            ]
        );
        add!(
            "gemm_qkv_hfq6g256",
            kernels::GEMM_QKV_HFQ6G256_SRC,
            ["gemm_qkv_hfq6g256"]
        );
        add!(
            "gemm_qkvza_hfq6g256",
            kernels::GEMM_QKVZA_HFQ6G256_SRC,
            ["gemm_qkvza_hfq6g256"]
        );
        add!(
            "gemv_hfq6g256_residual",
            kernels::GEMV_HFQ6G256_RESIDUAL_SRC,
            ["gemv_hfq6g256_residual"]
        );
        add!(
            "gemv_hfq6g256_residual_sigmoid_scaled",
            kernels::GEMV_HFQ6G256_RESIDUAL_SIGMOID_SCALED_SRC,
            ["gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",]
        );
        add!(
            "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_MQ4G256V2_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",]
        );
        add!(
            "moe_down_combine_grouped_k8",
            kernels::MOE_DOWN_COMBINE_GROUPED_K8_SRC,
            ["moe_down_combine_grouped_k8"]
        );
        add!(
            "moe_down_combine_k8_batched",
            kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC,
            ["moe_down_combine_k8_batched"]
        );
        add!(
            "moe_gate_up_unscatter_k8",
            kernels::MOE_GATE_UP_UNSCATTER_K8_SRC,
            ["moe_gate_up_unscatter_k8"]
        );
        add!(
            "moe_scatter_fused_k8",
            kernels::moe_scatter_fused_k8_src(arch == "gfx1151"),
            ["moe_scatter_fused_k8"]
        );
        if arch == "gfx1201" {
            add!(
                "moe_scatter_stable_k8",
                kernels::MOE_SCATTER_STABLE_K8_SRC,
                [
                    "moe_scatter_stable_hist_k8",
                    "moe_scatter_stable_inverse_k8",
                    "moe_scatter_stable_place_k8"
                ]
            );
        }
        add!(
            "moe_topk_renorm_k8",
            kernels::MOE_TOPK_RENORM_K8_SRC,
            ["moe_topk_renorm_k8"]
        );
        add!(
            "moe_topk_renorm_k8_batched",
            kernels::MOE_TOPK_RENORM_K8_BATCHED_SRC,
            [
                "moe_router_softmax_topk_k8_wave",
                "moe_topk_renorm_k8_batched"
            ]
        );
        add!(
            "scaled_add_inplace",
            kernels::SCALED_ADD_INPLACE_SRC,
            [
                "scaled_add_inplace_cpu_scalar_f32",
                "scaled_add_inplace_gpu_scalar_f32"
            ]
        );
        add!(
            "sigmoid_scaled_residual_add_batched",
            kernels::SIGMOID_SCALED_RESIDUAL_ADD_BATCHED_SRC,
            ["sigmoid_scaled_residual_add_batched_f32"]
        );
        add!("softmax", kernels::SOFTMAX_SRC, ["softmax_f32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1100") {
        add!(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            kernels::GEMV_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_K512_SRC,
            ["gemv_mq4g256v2_residual_sigmoid_scaled_k512",]
        );
        add!(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2",
            kernels::GEMV_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_K512_R2_SRC,
            ["gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2"]
        );
        add!(
            "repeat_interleave_qk_batched",
            kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC,
            ["repeat_interleave_qk_f32_batched"]
        );
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!(
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            kernels::GEMV_MQ4G256V2_MOE_GATE_UP_K8_INDEXED_SRC,
            ["gemv_mq4g256v2_moe_gate_up_k8_indexed"]
        );
        add!(
            "gemv_mq4g256v2_moe_ninepath_d4",
            kernels::GEMV_MQ4G256V2_MOE_NINEPATH_D4_SRC,
            ["gemv_mq4g256v2_moe_ninepath_d4"]
        );
        add!(
            "kv_cache_write_q8_0_pair",
            kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC,
            ["kv_cache_write_q8_0_pair"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "fused_rmsnorm_mq_rotate_vecsum",
            kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
            ["fused_rmsnorm_mq_rotate_vecsum"]
        );
        add!(
            "gemm_gate_up_hfq6g256_wmma",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_SRC,
            ["gemm_gate_up_hfq6g256_wmma"]
        );
        add!(
            "gemm_hfq6g256_residual_wmma_k2",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC,
            ["gemm_hfq6g256_residual_wmma_k2"]
        );
        add!(
            "gemm_q8_0_wmma",
            kernels::GEMM_Q8_0_WMMA_SRC,
            ["gemm_q8_0_wmma"]
        );
        add!(
            "gemm_qkv_hfq6g256_wmma",
            kernels::GEMM_QKV_HFQ6G256_WMMA_SRC,
            ["gemm_qkv_hfq6g256_wmma"]
        );
        add!(
            "gemm_qkvza_hfq6g256_wmma",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_SRC,
            ["gemm_qkvza_hfq6g256_wmma"]
        );
        add!(
            "moe_router_softmax_topk_k8_wave64_exact",
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC,
            ["moe_router_softmax_topk_k8_wave64_exact"]
        );
    }
    if arch == "gfx1201" {
        // A3B decode norm fold (`HIPFIRE_A3B_NORM_FOLD`): RMSNorm + FWHT rotate
        // inside the MQ4G256V2 projection GEMV that consumes it.
        add!(
            "fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201",
            kernels::FUSED_RMSNORM_PROJ4_MQ4G256V2_FOLD_GFX1201_SRC,
            ["fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201"]
        );
        // A3B VMM exact batch route: lane-batched multi-weight projection GEMV
        // (`lane_proj.rs`, `HIPFIRE_A3B_LANE_PROJ`).
        add!(
            "gemv_mq4g256v2_lane_proj_k2048_gfx1201",
            kernels::GEMV_MQ4G256V2_LANE_PROJ_K2048_GFX1201_SRC,
            [
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n1",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n2",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n3",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n4",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n5",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n6",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n7",
                "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n8",
            ]
        );
        // A3B VMM exact batch route: lane-batched twins of the MoE block's
        // router top-k, shared down, routed gate/up and routed down/combine
        // (`moe_decode_lanes.rs`; the shared silu launch reuses
        // `fused_silu_mul_mq_rotate`).
        for &(module, symbols) in A3B_GFX1201_MOE_LANES_HIP {
            entries.push(entry(
                arch,
                module,
                symbols,
                crate::moe_decode_lanes::moe_lane_module_source(module)
                    .map_err(|e| {
                        RegistryError::ConflictingPlan(format!("moe lane twin source: {e}"))
                    })?
                    .into(),
                extra_flags,
            ));
        }
        // A3B VMM exact batch route lm_head: the one-weight-read lane-batched
        // twin of `gemv_mq4g256v2_multirow_r2` (`head_lanes.rs`).
        entries.push(entry(
            arch,
            crate::head_lanes::HEAD_LANES_MODULE,
            &crate::head_lanes::HEAD_LANES_SYMBOLS,
            crate::head_lanes::HEAD_LANES_SRC.into(),
            extra_flags,
        ));
        // A3B VMM exact batch route: lane-batched residual GEMV of the wo
        // projection (`residual_lanes.rs`).
        add!(
            "gemv_mq4g256v2_residual_lanes_gfx1201",
            kernels::GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC,
            ["gemv_mq4g256v2_residual_lanes_gfx1201"]
        );
        // A3B VMM exact batch route expert-deduplicating routed gate/up and
        // down + combine (`moe_decode_lanes.rs`).
        entries.push(entry(
            arch,
            crate::moe_decode_lanes::MOE_DEDUP_MODULE,
            &crate::moe_decode_lanes::MOE_DEDUP_SYMBOLS,
            crate::moe_decode_lanes::MOE_DEDUP_SRC.into(),
            extra_flags,
        ));
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1201_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201",]
        );
        add!(
            "attention_q8_0_flash_prefill_wmma_gfx12_hd256",
            format!(
                "#define SPLIT_Q 0\n#define FIXED_HEAD_DIM 256\n#define PREFETCH_V 1\n{}",
                kernels::kv_slot_desc_source(
                    kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC,
                    false
                )
            ),
            ["attention_q8_0_flash_prefill_wmma",]
        );
        add!(
            "gated_norm_mq_rotate_gfx1201",
            kernels::GATED_NORM_MQ_ROTATE_GFX1201_SRC,
            ["gated_norm_mq_rotate_gfx1201"]
        );
        add!(
            "gated_norm_mq_rotate_i4_gfx12_v2",
            kernels::GATED_NORM_MQ_ROTATE_I4_GFX12_V2_SRC,
            ["gated_norm_mq_rotate_i4_gfx12_v2"]
        );
        add!(
            "gemm_gate_up_hfq6g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC,
            ["gemm_gate_up_hfq6g256_wmma_gfx12"]
        );
        add!(
            "gemm_hfq6g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_GFX12_SRC,
            ["gemm_hfq6g256_residual_wmma_gfx12"]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gfx12_g12r",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_G12R_SRC,
            [
                "gemm_mq4g256v2_gate_up_silu_mmq_iu4_g12r",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_g12r",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_g12r",
                "gemm_mq4g256v2_residual_mmq_iu4_g12r",
                "quantize_int4_mmq_ds128",
            ]
        );
        add!(
            "gemm_q8_0_wmma_gfx12",
            kernels::GEMM_Q8_0_WMMA_GFX12_SRC,
            ["gemm_q8_0_wmma_gfx12"]
        );
        add!(
            "gemm_q8_0_mmq_i8_gfx12",
            kernels::GEMM_Q8_0_MMQ_I8_GFX12_SRC,
            [
                "gemm_q8_0_mmq_i8_t128x128",
                "gemm_q8_0_mmq_i8_t128x128_slow",
                "gemm_q8_0_mmq_i8_t128x64",
                "gemm_q8_0_mmq_i8_t128x64_slow",
                "gemm_q8_0_mmq_i8_t64x64",
                "gemm_q8_0_mmq_i8_t64x64_slow",
                "quantize_x_q8_blk32",
            ]
        );
        add!(
            "gemm_q8_0_mmq_fp8_gfx12",
            kernels::GEMM_Q8_0_MMQ_FP8_GFX12_SRC,
            [
                "gemm_q8_0_mmq_fp8_t128x128",
                "gemm_q8_0_mmq_fp8_t128x128_slow",
                "gemm_q8_0_mmq_fp8_t128x64",
                "gemm_q8_0_mmq_fp8_t128x64_slow",
                "gemm_q8_0_mmq_fp8_t64x64",
                "gemm_q8_0_mmq_fp8_t64x64_slow",
                "quantize_x_fp8_blk32",
            ]
        );
        add!(
            "gemm_qkv_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC,
            ["gemm_qkv_hfq6g256_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_hfq6g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC,
            ["gemm_qkvza_hfq6g256_wmma_gfx12"]
        );
        add!(
            "moe_router_softmax_topk_k8_wave64",
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC,
            ["moe_router_softmax_topk_k8_wave64"]
        );
        add!(
            "qwen35_fa_prep_gfx1201",
            kernels::QWEN35_FA_PREP_GFX1201_SRC,
            ["qwen35_fa_prep_gfx1201"]
        );
    }
    if arch == "gfx1100" {
        add!(
            "argmax_token_chain",
            kernels::ARGMAX_TOKEN_CHAIN_SRC,
            ["argmax_token_chain_f32"]
        );
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_GFX1100_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100",]
        );
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",]
        );
        add!(
            "conv1d_silu_split_qknorm_b256",
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
            ["conv1d_silu_split_qknorm_b256"]
        );
        add!(
            "deinterleave_q_rmsnorm_f32_batched",
            kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC,
            ["deinterleave_q_rmsnorm_f32_batched"]
        );
        add!(
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
            kernels::FUSED_QKV_MQ4G256V2_K2048_X_BUFFER_GFX1100_SRC,
            ["fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",]
        );
        add!(
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
            kernels::FUSED_QKVZA_MQ4G256V2_K2048_HOIST_X32_GFX1100_SRC,
            ["fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",]
        );
        add!(
            "gated_norm_mq_rotate_gfx1100",
            kernels::GATED_NORM_MQ_ROTATE_GFX1100_SRC,
            ["gated_norm_mq_rotate_gfx1100"]
        );
        add!(
            "gated_norm_mq_rotate_i4_gfx1100_v2",
            kernels::GATED_NORM_MQ_ROTATE_I4_GFX1100_V2_SRC,
            ["gated_norm_mq_rotate_i4_gfx1100_v2"]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_k2",
            kernels::GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_mq4g256v2_moe_grouped_wmma_k2"]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gridspec",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
                "quantize_int4_mmq_ds128",
            ]
        );
        add!(
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100",
            kernels::GEMV_MQ4G256V2_MOE_GATE_UP_K8_INDEXED_K2048_NOLDS_GFX1100_SRC,
            ["gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100",]
        );
        add!(
            "gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100",
            kernels::GEMV_MQ4G256V2_MOE_NINEPATH_RPB8_GFX1100_SRC,
            ["gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100"]
        );
        add!(
            "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
            kernels::GEMV_MQ4G256V2_RESIDUAL_R1_K4096_GFX1100_NOSCRATCH_SRC,
            ["gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",]
        );
        add!(
            "greedy_accept",
            kernels::GREEDY_ACCEPT_SRC,
            ["greedy_accept_from_argmax_i32"]
        );
        add!(
            "mq_rotate_x_i4",
            kernels::MQ_ROTATE_X_I4_SRC,
            ["mq_rotate_x_i4"]
        );
        add!(
            "qwen35_fa_prep_gfx1100",
            kernels::QWEN35_FA_PREP_GFX1100_SRC,
            ["qwen35_fa_prep_gfx1100"]
        );
    }
    if arch == "gfx1151" {
        add!(
            "attention_dflash_wmma_kvsplit_gfx1151",
            kernels::ATTENTION_DFLASH_WMMA_KVSPLIT_GFX1151_SRC,
            ["attention_dflash_wmma_kvsplit_gfx1151"]
        );
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",]
        );
        add!(
            "attention_verify_wmma_gfx1151",
            kernels::ATTENTION_VERIFY_WMMA_GFX1151_SRC,
            [
                "attention_verify_wmma_pv1s_d4_gfx1151",
                "attention_verify_wmma_pv1s_d4_hd128_gfx1151",
                "attention_verify_wmma_pv2_d2_gfx1151",
                "attention_verify_wmma_pv2_d2_hd128_gfx1151",
                "attention_verify_wmma_qk_gfx1151",
                "attention_verify_wmma_qk_hd128_gfx1151",
            ]
        );
        add!(
            "gated_norm_mq_rotate_gfx1151",
            kernels::GATED_NORM_MQ_ROTATE_GFX1151_SRC,
            ["gated_norm_mq_rotate_gfx1151"]
        );
        add!(
            "gated_norm_mq_rotate_i4_gfx1151_v2",
            kernels::GATED_NORM_MQ_ROTATE_I4_GFX1151_V2_SRC,
            ["gated_norm_mq_rotate_i4_gfx1151_v2"]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_k2",
            kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
            [
                "gemm_mq4g256v2_moe_grouped_wmma_k2",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
            ]
        );
        add!(
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4",
            kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC,
            [
                "gemm_mq4g256v2_moe_grouped_wmma_k2",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
                "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4",
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gridspec",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
                "quantize_int4_mmq_ds128",
            ]
        );
        add!(
            "qwen35_fa_prep_gfx1151",
            kernels::QWEN35_FA_PREP_GFX1151_SRC,
            ["qwen35_fa_prep_gfx1151"]
        );
    }

    // FirstRequestJit discovery batch 1 (cell R4, qwen3.6:35b-a3b mq4p on
    // gfx1201, hold frj-disc-r4-v2-01): modules the first requests compiled
    // after `ready`. Each source is the exact constant/expression the launcher
    // passes to `ensure_kernel`; the gfx1201 sources' sha256 equal the observed
    // kcache `.hip` files. Siblings the same launcher selects on gfx1100 /
    // gfx1151 under defaults are packaged with them.

    // Decode 3-/4-way fused Q8_0 projections (`Gpu::fused_qkv_q8_0` /
    // `fused_qkvza_q8_0`, gemm.rs): scalar kernels, no arch gate (the
    // dispatch tables route Q8_0 decode here "cross-arch").
    add!(
        "fused_qkv_q8_0",
        kernels::FUSED_QKV_Q8_0_SRC,
        ["fused_qkv_q8_0"]
    );
    add!(
        "fused_qkvza_q8_0",
        kernels::FUSED_QKVZA_Q8_0_SRC,
        ["fused_qkvza_q8_0"]
    );
    // ParoQ4G64 scalar projections and its group-sized out-of-place rotation;
    // these launchers have no architecture gate.
    add!("gemm_hfq4g64", kernels::GEMM_HFQ4G64_SRC, ["gemm_hfq4g64"]);
    add!("gemv_hfq4g64", kernels::GEMV_HFQ4G64_SRC, ["gemv_hfq4g64"]);
    add!(
        "givens_rotate_to_g64_f32",
        kernels::GIVENS_ROTATE_TO_G64_SRC,
        ["givens_rotate_to_g64_f32"]
    );
    // DFlash2's HC-state mean capture uses the same scalar source on all arches.
    add!(
        "hc_mean_capture",
        crate::tensor_ops::HC_MEAN_CAPTURE_SRC,
        ["hc_mean_capture_f32"]
    );
    // Coverage batch A (no arch gate in the launchers, embedding.rs /
    // gemm.rs `fused_gate_up_q8_0`): default resident-codec embeddings
    // (pipeline/layer_ops.rs MQ4V2 and BF16 `EmbeddingOp`) and the Q8_0
    // decode fused gate/up (families/fused_qkv.rs).
    add!(
        "embedding_mq4v2_batched",
        kernels::EMBEDDING_MQ4V2_BATCHED_SRC,
        ["embedding_mq4v2_batched"]
    );
    add!(
        "embedding_bf16_batched",
        kernels::EMBEDDING_BF16_BATCHED_SRC,
        ["embedding_bf16_batched"]
    );
    add!(
        "fused_gate_up_q8_0",
        kernels::FUSED_GATE_UP_Q8_0_SRC,
        ["fused_gate_up_q8_0"]
    );
    // Coverage batch B: HFQ6 gate/up scalar fallback
    // (`gemm_gate_up_hfq6g256`: the decode `None` arm on every non-dp4a arch
    // and the prefill tail; the WMMA twins are registered per arch); no arch gates.
    add!(
        "gemm_gate_up_hfq6g256",
        kernels::GEMM_GATE_UP_HFQ6G256_SRC,
        ["gemm_gate_up_hfq6g256"]
    );
    // The `gemv_hfq4g256_residual_scaled` module is shared by the routed
    // `_cpu`/`_gpu` accumulators, the shared-expert sigmoid decode launcher
    // (plain-module fallback) and its batched prefill twin
    // (`gemv.rs` `gemv_hfq4g256_residual_*`); the batched launcher has no
    // arch gate on the RDNA wave32 MoE routes.
    // `default_route_entries` carries the module without the batched symbol
    // (gfx1151/gfx1201); the same-source twin here supersedes it in the corpus.
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        add!(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            [
                "gemv_hfq4g256_residual_scaled_cpu",
                "gemv_hfq4g256_residual_scaled_gpu",
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched"
            ]
        );
        // Mixed-precision (dtype-tag) MoE batched decode/prefill GEMVs
        // (`gemv_mixed_moe_gate_up_k8_indexed_batched`,
        // `gemv_mixed_moe_down_k8_indexed_batched_expanded`; wave32 RDNA only).
        add!(
            "gemv_mixed_moe_gate_up_indexed_batched",
            kernels::GEMV_MIXED_MOE_GATE_UP_INDEXED_BATCHED_SRC,
            ["gemv_mixed_moe_gate_up_k8_indexed_batched"]
        );
        add!(
            "gemv_mixed_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_MIXED_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_mixed_moe_down_k8_indexed_batched_expanded"]
        );
        // Uniform-dtype Qwen3.5-MoE decode GEMVs (hipfire-dispatch
        // `pipeline/mod.rs` decode gate/up + down stages, no arch branch in
        // the gemv.rs launchers): HFQ6 (qt15) gate/up and expanded down,
        // MQ6V2 (qt47) gate/up and expanded down, and the default-on
        // single-token nine-path d4 fused decode for uniform MQ4G256 and
        // MQ6G256V2 experts (`moe_program.rs` ninepath_d4; `_d3` and
        // `_last_combine` are developer opt-ins).
        add!(
            "gemv_hfq6g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ6G256_MOE_GATE_UP_INDEXED_SRC,
            ["gemv_hfq6g256_moe_gate_up_k8_indexed"]
        );
        add!(
            "gemv_hfq6g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ6G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_hfq6g256_moe_down_k8_indexed_batched_expanded"]
        );
        add!(
            "gemv_hfq4g256_moe_ninepath_d4",
            kernels::GEMV_HFQ4G256_MOE_NINEPATH_D4_SRC,
            ["gemv_hfq4g256_moe_ninepath_d4"]
        );
        // qt53 (MQ4G128V2) routed-down nine-path (`moe_program.rs`
        // ninepath_mq4g128v2): fused silu*up + FWHT128 + down + combine.
        add!(
            "gemv_mq4g128v2_moe_ninepath_d4",
            kernels::GEMV_MQ4G128V2_MOE_NINEPATH_D4_SRC,
            ["gemv_mq4g128v2_moe_ninepath_d4"]
        );
        add!(
            "gemv_mq6g256v2_moe_ninepath_d4",
            kernels::GEMV_MQ6G256V2_MOE_NINEPATH_D4_SRC,
            ["gemv_mq6g256v2_moe_ninepath_d4"]
        );
        add!(
            "gemv_mq6g256v2_moe_gate_up_k8_indexed",
            kernels::GEMV_MQ6G256V2_MOE_GATE_UP_K8_INDEXED_SRC,
            ["gemv_mq6g256v2_moe_gate_up_k8_indexed"]
        );
        add!(
            "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_MQ6G256V2_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded"]
        );
    }
    // Coverage batch D (ClassifyOther / ClassifyGemm; every module's launcher
    // sink, source constant and symbol are the ones named below, symbol ==
    // module unless noted). RDNA-core arches only.
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        // Sampled DFlash (`HIPFIRE_DFLASH_FAST_SAMPLE` default on): sampling.rs
        // `batched_categorical_sample_f32` / `chain_accept_spec_f32`.
        add!(
            "batched_categorical_sample",
            kernels::BATCHED_CATEGORICAL_SAMPLE_SRC,
            ["batched_categorical_sample_f32"]
        );
        add!(
            "chain_accept_spec",
            kernels::CHAIN_ACCEPT_SPEC_SRC,
            ["chain_accept_spec_f32"]
        );
        // Decode fused projections of MQ6G256V2 / MQ3G256V2 (`ArchPredicate::Always`).
        add!(
            "fused_qkv_mq6g256v2",
            kernels::FUSED_QKV_MQ6G256V2_SRC,
            ["fused_qkv_mq6g256v2"]
        );
        add!(
            "fused_qkvza_mq6g256v2",
            kernels::FUSED_QKVZA_MQ6G256V2_SRC,
            ["fused_qkvza_mq6g256v2"]
        );
        add!(
            "fused_gate_up_mq6g256v2",
            kernels::FUSED_GATE_UP_MQ6G256V2_SRC,
            ["fused_gate_up_mq6g256v2"]
        );
        add!(
            "fused_qkv_mq3g256v2",
            kernels::FUSED_QKV_MQ3G256V2_SRC,
            ["fused_qkv_mq3g256v2"]
        );
        add!(
            "fused_qkvza_mq3g256v2",
            kernels::FUSED_QKVZA_MQ3G256V2_SRC,
            ["fused_qkvza_mq3g256v2"]
        );
        add!(
            "fused_gate_up_mq3g256v2",
            kernels::FUSED_GATE_UP_MQ3G256V2_SRC,
            ["fused_gate_up_mq3g256v2"]
        );
        // HFQ4 A3B MoE decode gate/up (`gemv_hfq4g256_moe_gate_up_k8_indexed`,
        // gemv.rs plain-module arm; the gfx1100/gfx1151 tuned variants are
        // developer envs). `default_route_entries` carries the same source.
        add!(
            "gemv_hfq4g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
            ["gemv_hfq4g256_moe_gate_up_k8_indexed"]
        );
        // Q8_0 lm_head / spec-verify batched GEMM (`gemm_q8_0_batched_f32_chunked`).
        add!(
            "gemm_q8_0_batched",
            kernels::GEMM_Q8_0_BATCHED_SRC,
            ["gemm_q8_0_batched"]
        );
        // HFQ4 scalar batch-1 / `HIPFIRE_FP16=0` prefill fallbacks.
        add!(
            "gemm_qkvza_hfq4g256",
            kernels::GEMM_QKVZA_HFQ4G256_SRC,
            ["gemm_qkvza_hfq4g256"]
        );
        add!(
            "gemm_qkv_hfq4g256",
            kernels::GEMM_QKV_HFQ4G256_SRC,
            ["gemm_qkv_hfq4g256"]
        );
        add!(
            "gemm_gate_up_hfq4g256",
            kernels::GEMM_GATE_UP_HFQ4G256_SRC,
            ["gemm_gate_up_hfq4g256"]
        );
        add!(
            "gemm_hfq4g256_residual",
            kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
            ["gemm_hfq4g256_residual"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        // MQ3G256 v1 prefill QKV / QKVZA / residual: base WMMA plus the MB4
        // twins (`flags.mq3_mb4` None: batch >= 128 and large M).
        add!(
            "gemm_qkv_hfq3g256_wmma",
            kernels::GEMM_QKV_HFQ3G256_WMMA_SRC,
            ["gemm_qkv_hfq3g256_wmma"]
        );
        add!(
            "gemm_qkv_hfq3g256_wmma_mb4",
            kernels::GEMM_QKV_HFQ3G256_WMMA_MB4_SRC,
            ["gemm_qkv_hfq3g256_wmma_mb4"]
        );
        add!(
            "gemm_qkvza_hfq3g256_wmma",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_SRC,
            ["gemm_qkvza_hfq3g256_wmma"]
        );
        add!(
            "gemm_qkvza_hfq3g256_wmma_mb4",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_MB4_SRC,
            ["gemm_qkvza_hfq3g256_wmma_mb4"]
        );
        add!(
            "gemm_hfq3g256_residual_wmma",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_SRC,
            ["gemm_hfq3g256_residual_wmma"]
        );
        add!(
            "gemm_hfq3g256_residual_wmma_mb4",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_MB4_SRC,
            ["gemm_hfq3g256_residual_wmma_mb4"]
        );
        // MQ4G256 v1 gfx11 WMMA prefill (batch < 128, no MMQ).
        add!(
            "gemm_qkvza_hfq4g256_wmma",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_SRC,
            ["gemm_qkvza_hfq4g256_wmma"]
        );
        add!(
            "gemm_qkv_hfq4g256_wmma",
            kernels::GEMM_QKV_HFQ4G256_WMMA_SRC,
            ["gemm_qkv_hfq4g256_wmma"]
        );
        // F16 lm_head (`gemm_f16_batched_lmhead`; gfx12 takes `gemm_f16_wmma_mb8`).
        add!(
            "gemm_mw16_residual_wmma",
            kernels::GEMM_MW16_RESIDUAL_WMMA_SRC,
            [
                "gemm_mw16_residual_wmma",
                "gemm_mw16_residual_wmma_gfx1151_xlds"
            ]
        );
        // Sparse sampled DFlash host-accept target rows (speculative.rs
        // `dflash_target_rows_into`; the device accept path is gfx1201-only).
        // Module name == `sampled_rows_compact::SAMPLED_ROWS_COMPACT_MODULE`.
        add!(
            "sampled_rows_compact",
            crate::sampled_rows_compact::SAMPLED_ROWS_COMPACT_SRC,
            ["sampled_rows_compact_f32"]
        );
        // gfx11 sibling of the HFQ4 grouped MoE WMMA (same launcher's non-gfx12 arm).
        add!(
            "gemm_hfq4g256_moe_grouped_wmma_k2",
            kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_hfq4g256_moe_grouped_wmma_k2"]
        );
    }
    if arch == "gfx1201" {
        // Discovery batch R11 (cell R11, hold frj-disc-r11-v2-01, 27B mq4
        // DFlash) and R13 (frj-disc-r13-v2-01, 27B mq4 AR): the non-slab AWQ
        // IU4 prefill producers (`HIPFIRE_G12_NORM` default on, slab off), the
        // k6144 gated norm and the fp8q FA prep (module name == symbol).
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2"]
        );
        add!(
            "fused_silu_mul_mq_rotate_awq_i4_gfx12",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_GFX12_SRC,
            ["fused_silu_mul_mq_rotate_awq_i4_gfx12"]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx12_v2",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx12_v2"]
        );
        add!(
            "gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16",
            kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_XBF16_SRC,
            ["gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16"]
        );
        add!(
            "sigmoid_mul_rotate_x_mq_awq_i4_gfx12",
            kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SRC,
            ["sigmoid_mul_rotate_x_mq_awq_i4_gfx12"]
        );
        add!(
            "sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12",
            kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GIL_GFX12_SRC,
            ["sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12"]
        );
        add!(
            "gated_norm_mq_rotate_k6144_gfx1201",
            kernels::gated_norm_mq_rotate_k6144_gfx1201_src(),
            ["gated_norm_mq_rotate_k6144_gfx1201"]
        );
        add!(
            "qwen35_fa_prep_fp8q_batched_gfx1201",
            crate::qwen35_fa_batch::FA_PREP_BATCHED_GFX1201_SRC,
            ["qwen35_fa_prep_fp8q_batched_gfx1201"]
        );
        // Q8 decode fused epilogue with an AWQ scale on gfx1201 (A3B).
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1201",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_GFX1201_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1201"]
        );
        add!(
            "attention_flash_q8_0_rows4_d8",
            kernels::ATTENTION_FLASH_Q8_0_TILE_ROWS_SRC,
            [
                "attention_flash_q8_0_rows4_d8",
                "attention_flash_q8_0_rows4_d4",
                "attention_flash_q8_0_rows8_d4",
            ]
        );
        add!(
            "attention_flash_q8_0_rows8_d8",
            kernels::ATTENTION_FLASH_Q8_0_TILE_ROWS_SRC,
            ["attention_flash_q8_0_rows8_d8"]
        );
        // Dynamic MQ6V2/MQ3V2 gfx12 GEMM families (`gfx12_bt_tile`: base for
        // N < 64, bt4/bt8/bt12 above; shared `_mq5v2` module suffix, symbol is
        // the unsuffixed kname).
        add!(
            "gemm_qkvza_mq6g256v2_wmma_gfx12_mq5v2",
            kernels::GEMM_QKVZA_MQ6G256V2_WMMA_GFX12_SRC,
            ["gemm_qkvza_mq6g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_mq6g256v2_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_QKVZA_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq6g256v2_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_qkvza_mq6g256v2_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_QKVZA_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq6g256v2_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_qkvza_mq6g256v2_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_QKVZA_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq6g256v2_wmma_gfx12_bt12"]
        );
        add!(
            "gemm_gate_up_mq6g256v2_wmma_gfx12_mq5v2",
            kernels::GEMM_GATE_UP_MQ6G256V2_WMMA_GFX12_SRC,
            ["gemm_gate_up_mq6g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_gate_up_mq6g256v2_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_GATE_UP_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq6g256v2_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_gate_up_mq6g256v2_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_GATE_UP_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq6g256v2_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_gate_up_mq6g256v2_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_GATE_UP_MQ6G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq6g256v2_wmma_gfx12_bt12"]
        );
        add!(
            "gemm_qkvza_mq3g256v2_wmma_gfx12_mq5v2",
            kernels::GEMM_QKVZA_MQ3G256V2_WMMA_GFX12_SRC,
            ["gemm_qkvza_mq3g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_mq3g256v2_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_QKVZA_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq3g256v2_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_qkvza_mq3g256v2_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_QKVZA_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq3g256v2_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_qkvza_mq3g256v2_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_QKVZA_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_mq3g256v2_wmma_gfx12_bt12"]
        );
        add!(
            "gemm_gate_up_mq3g256v2_wmma_gfx12_mq5v2",
            kernels::GEMM_GATE_UP_MQ3G256V2_WMMA_GFX12_SRC,
            ["gemm_gate_up_mq3g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_gate_up_mq3g256v2_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_GATE_UP_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq3g256v2_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_gate_up_mq3g256v2_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_GATE_UP_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq3g256v2_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_gate_up_mq3g256v2_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_GATE_UP_MQ3G256V2_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_mq3g256v2_wmma_gfx12_bt12"]
        );
        add!(
            "gemm_mq3g256v2_residual_wmma_gfx12_mq5v2",
            kernels::GEMM_MQ3G256V2_RESIDUAL_WMMA_GFX12_SRC,
            ["gemm_mq3g256v2_residual_wmma_gfx12"]
        );
        add!(
            "gemm_mq3g256v2_residual_wmma_gfx12_bt4_mq5v2",
            kernels::GEMM_MQ3G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_mq3g256v2_residual_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_mq3g256v2_residual_wmma_gfx12_bt8_mq5v2",
            kernels::GEMM_MQ3G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_mq3g256v2_residual_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_mq3g256v2_residual_wmma_gfx12_bt12_mq5v2",
            kernels::GEMM_MQ3G256V2_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_mq3g256v2_residual_wmma_gfx12_bt12"]
        );
        // Discovery batch R7 (cell R7, hold frj-disc-r7-v2-01, mq4r A3B):
        // `Gpu::gemm_hfq4g256_moe_grouped_wmma*` picks the gfx12 module on
        // RDNA4 (`HIPFIRE_MOE_GROUPED_M2` developer opt-in is not registered);
        // the residual and QKVZA launchers pick bt4/bt8/bt12 via
        // `gfx12_bt_tile` (one source each, symbol == module). The QKV
        // launcher has no bt family.
        add!(
            "gemm_hfq4g256_moe_grouped_wmma_gfx12",
            kernels::GEMM_HFQ4G256_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_hfq4g256_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_gfx12_bt4",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_hfq4g256_residual_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_gfx12_bt8",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_hfq4g256_residual_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_gfx12_bt12",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_BT_SRC,
            ["gemm_hfq4g256_residual_wmma_gfx12_bt12"]
        );
        add!(
            "gemm_qkvza_hfq4g256_wmma_gfx12_bt4",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_hfq4g256_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_qkvza_hfq4g256_wmma_gfx12_bt8",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_hfq4g256_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_qkvza_hfq4g256_wmma_gfx12_bt12",
            kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_qkvza_hfq4g256_wmma_gfx12_bt12"]
        );
        // gfx1201 default MQ4V2 FP8 prefill pre-pass (`Gpu::prepare_mq4v2_fp8_x`,
        // scratch.rs): one module per entry, both over the same TU.
        add!(
            "pack_f16_to_fp8_mq4v2_gfx12",
            kernels::PACK_F16_TO_FP8_MQ4V2_GFX12_SRC,
            ["pack_f16_to_fp8_mq4v2_gfx12"]
        );
        add!(
            "pack_f32_to_fp8_mq4v2_gfx12",
            kernels::PACK_F32_TO_FP8_MQ4V2_GFX12_SRC,
            ["pack_f32_to_fp8_mq4v2_gfx12"]
        );
        // HFQ3 gfx12 QKV / QKVZA / residual WMMA (gate/up twin is above).
        add!(
            "gemm_qkv_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKV_HFQ3G256_WMMA_GFX12_SRC,
            ["gemm_qkv_hfq3g256_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_hfq3g256_wmma_gfx12",
            kernels::GEMM_QKVZA_HFQ3G256_WMMA_GFX12_SRC,
            ["gemm_qkvza_hfq3g256_wmma_gfx12"]
        );
        add!(
            "gemm_hfq3g256_residual_wmma_gfx12",
            kernels::GEMM_HFQ3G256_RESIDUAL_WMMA_GFX12_SRC,
            ["gemm_hfq3g256_residual_wmma_gfx12"]
        );
        // MTP/DFlash temperature verify softmax (norm.rs
        // `softmax_temp_topp_batched_into_f32`), same module and source the
        // gfx11 inventory carries; formerly pushed only by `route_corpus`.
        add!(
            "softmax_temp_topp_batched",
            kernels::SOFTMAX_TEMP_BATCHED_SRC,
            ["softmax_temp_batched_f32", "softmax_temp_topp_batched_f32"]
        );
    }
    if matches!(arch, "gfx1151" | "gfx1201") {
        add!(
            "gemv_hfq4g256_residual_mq6v2",
            kernels::GEMV_MQ6G256V2_RESIDUAL_SRC,
            ["gemv_mq6g256v2_residual"]
        );
        add!(
            "gemv_hfq4g256_residual_mq3g256v2",
            kernels::GEMV_MQ3G256V2_RESIDUAL_SRC,
            ["gemv_mq3g256v2_residual"]
        );
        add!(
            "gemv_mq3g256v2_mq3v2",
            kernels::GEMV_MQ3G256V2_SRC,
            ["gemv_mq3g256v2"]
        );
    }
    if arch == "gfx1151" {
        // Q8_0 4-wave WMMA batched GEMM (`gemm_q8_0_batched_chunked`: M >= 4096,
        // M and N multiples of 64) and the 248320x2048 lm_head r1 hybrid-buffer
        // GEMV (`gemv_hfq4g256`, and `gemv_mq4g256v2` over the same module
        // plus `_mq4v2`; shape-gated, no env).
        add!(
            "gemm_q8_0_wmma_4w",
            kernels::GEMM_Q8_0_WMMA_4W_SRC,
            ["gemm_q8_0_wmma_4w"]
        );
        add!(
            "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151",
            kernels::GEMV_HFQ4G256_LM_HEAD_R1_HYBRID_BUFFER_GFX1151_SRC,
            ["gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"]
        );
        add!(
            "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151_mq4v2",
            kernels::GEMV_MQ4G256V2_SRC,
            ["gemv_mq4g256v2"]
        );
    }
    if arch == "gfx1100" {
        // DFlash draft collapse modules (dflash_draft_fusion.rs: module name ==
        // symbol, all over `COLLAPSE_SRC`); previously only exported as
        // symbols of `dynamic_conv_residual_gfx1100`, never as module keys.
        add!(
            "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2"]
        );
        add!(
            "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4"]
        );
        add!(
            "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8"]
        );
        add!(
            "mq_rotate_x_f16_dflash_gfx1100",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["mq_rotate_x_f16_dflash_gfx1100"]
        );
        add!(
            "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100"]
        );
        // gfx11-dGPU arch-suffixed GEMV module names of the MQ6V2/MQ3V2
        // dense GEMV launchers (`gemv_hfq4g256_residual_for_arch` /
        // `gemv_mq3g256v2_for_arch` return the `_rdna3` module name on
        // gfx1100; the symbols are the plain ones); gfx1151/gfx1201 take the
        // unsuffixed names above.
        add!(
            "gemv_hfq4g256_residual_rdna3_mq6v2",
            kernels::GEMV_MQ6G256V2_RESIDUAL_SRC,
            ["gemv_mq6g256v2_residual"]
        );
        add!(
            "gemv_hfq4g256_residual_rdna3_mq3g256v2",
            kernels::GEMV_MQ3G256V2_RESIDUAL_SRC,
            ["gemv_mq3g256v2_residual"]
        );
        add!(
            "gemv_mq3g256v2_rdna3_mq3v2",
            kernels::GEMV_MQ3G256V2_SRC,
            ["gemv_mq3g256v2"]
        );
        // Q8 KV multi-row flash tile (`attention_flash_q8_0_rows_masked_logical`:
        // N 4..=32 at ctx > 4096, default on gfx1100/gfx1201; rows4 for N 4..7,
        // rows8 for N >= 8; D=256 heads use the d8 entries; d4 is head_dim 128).
        add!(
            "attention_flash_q8_0_rows4_d8",
            kernels::ATTENTION_FLASH_Q8_0_TILE_ROWS_SRC,
            [
                "attention_flash_q8_0_rows4_d8",
                "attention_flash_q8_0_rows4_d4",
                "attention_flash_q8_0_rows8_d4",
            ]
        );
        add!(
            "attention_flash_q8_0_rows8_d8",
            kernels::ATTENTION_FLASH_Q8_0_TILE_ROWS_SRC,
            ["attention_flash_q8_0_rows8_d8"]
        );
        // DFlash draft MQ4G256 deterministic K-split collapse
        // (`gemm_hfq4g256_overwrite_ksplit_det_dflash`, exact gfx1100): the
        // K-split GEMM plus its finalize kernel from the collapse source.
        add!(
            "gemm_hfq4g256_residual_wmma_ksplit_det",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_KSPLIT_DET_SRC,
            ["gemm_hfq4g256_residual_wmma_ksplit_det"]
        );
        add!(
            "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100",
            crate::dflash_draft_fusion::COLLAPSE_SRC,
            ["gemm_ksplit_det_overwrite_finalize_dflash_gfx1100"]
        );
        // The K-split finalize `gemm_hfq4g256_residual_wmma`'s `ksplit_det`
        // variant pairs with the partial GEMM above (gemm.rs `fin_name`).
        add!(
            "gemm_ksplit_det_finalize",
            kernels::GEMM_KSPLIT_DET_FINALIZE_SRC,
            ["gemm_ksplit_det_finalize"]
        );
    }
    // Coverage batch E: MQ6 v1 plain residual (`gemm_hfq6g256_residual`, the
    // n=1 / `HIPFIRE_FP16=0` fallback of `K::GemmHfq6G256Residual`) and the
    // DFlash F32 draft-weight batched GEMM (`gemm_f32_batched`,
    // hipfire-runtime dflash.rs `DType::F32` arm); no arch gate.
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        add!(
            "gemm_hfq6g256_residual",
            kernels::GEMM_HFQ6G256_RESIDUAL_SRC,
            ["gemm_hfq6g256_residual"]
        );
        add!(
            "gemm_f32_batched",
            kernels::GEMM_F32_SRC,
            ["gemm_f32_batched"]
        );
        // Raon mHC selector and grouped-MoE router: BF16-weight x F32-activation
        // GEMMs, batched (decode / verify / chunks) and x-stationary tiled (gfx1201
        // prefill).
        add!(
            "gemm_bf16_xf32_batched",
            kernels::GEMM_BF16_XF32_BATCHED_SRC,
            ["gemm_bf16_xf32_batched"]
        );
        add!(
            "gemm_bf16_xf32_tiled_rows",
            kernels::GEMM_BF16_XF32_TILED_ROWS_SRC,
            ["gemm_bf16_xf32_tiled_rows", "gemm_bf16_xf32_tiled_rows_mp4"]
        );
    }
    // lm_head rm2 muse tile (`gemm_hfq4g256_residual_muse_gfx1100_rm` rm=2):
    // gfx1151 at batch % 96 == 0 and (m/32)*(batch/96) >= 256 (default DFlash
    // FC projection), gfx1100 at the exact Glimmer M19968 K6656 B192 shape.
    // Module name is the file, the symbol is the rm2_bv6 entry (rm 3/4/6 are
    // never requested).
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm_bt",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX1100_MUSE_RM_BT_SRC,
            [
                "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm2_bv6",
                "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm3_bv4",
                "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm4_bv3",
                "gemm_hfq4g256_residual_wmma_gfx1100_muse_rm6_bv2",
            ]
        );
    }
    if arch == "gfx1151" {
        // Q8 decode fused gate/up epilogue with an AWQ scale (A3B on gfx1151,
        // `attention_flash_reduce_gated_mq_rotate_gfx1100`).
        add!(
            "attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1151",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_GFX1151_SRC,
            ["attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1151"]
        );
        // Flash-Next NPU trunk G80 activation producer / BF16 consume and
        // epilogue kernels (npu_trunk.rs `ensure_kernel_public("npu_trunk",
        // NPU_TRUNK_SRC, ..)`; exact gfx1151, needs the XDNA NPU; `_pm`
        // consumers and the i8 GEMM are embedded images, not this source).
        add!(
            "npu_trunk",
            kernels::NPU_TRUNK_SRC,
            [
                "npu_trunk_consume_bf16_fast",
                "npu_trunk_epi_bf16",
                "npu_trunk_pack_a_bf16",
                "npu_trunk_pack_a_bf16_k5120",
                "npu_trunk_pack_a_bf16_old",
            ]
        );
    }
    // Batched Q8_0 flash tile (`attention_flash_q8_0_tile_batched` /
    // `_masked` / windowed callers, attention.rs `launch_asym_flash_batched`);
    // gfx1100 carries it above. Source goes through `ensure_givens4_kernel`'s
    // strip-and-prepend, i.e. `assemble_asym`.
    if matches!(arch, "gfx1151" | "gfx1201") {
        add!(
            "attention_flash_q8_0_tile_batched",
            assemble_asym(kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC),
            ["attention_flash_q8_0_tile_batched"]
        );
    }
    // gfx11 WMMA Q8_0 prefill twins of the gfx12 projections below
    // (`gemm_qkv_q8_0_wmma` / `gemm_qkvza_q8_0_wmma` /
    // `gemm_q8_0_residual_wmma`, gemm.rs; RDNA4 delegates to the gfx12 twins)
    // and the gfx11 mixed-dtype grouped MoE WMMA (`gemm_mixed_moe_grouped_wmma_impl`
    // non-gfx12, non-`HIPFIRE_MOE_GROUPED_4W` arm).
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "gemm_qkv_q8_0_wmma",
            kernels::GEMM_QKV_Q8_0_WMMA_SRC,
            ["gemm_qkv_q8_0_wmma"]
        );
        add!(
            "gemm_qkvza_q8_0_wmma",
            kernels::GEMM_QKVZA_Q8_0_WMMA_SRC,
            ["gemm_qkvza_q8_0_wmma"]
        );
        add!(
            "gemm_q8_0_residual_wmma",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_SRC,
            ["gemm_q8_0_residual_wmma"]
        );
        add!(
            "gemm_gate_up_q8_0_wmma",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_SRC,
            ["gemm_gate_up_q8_0_wmma"]
        );
        // HFQ3 prefill gate/up (`gemm_gate_up_hfq3g256_wmma`: base WMMA, or
        // the MB4 twin at N>=128 and gate+up M>=4096 on RDNA3 dGPU/3.5).
        add!(
            "gemm_gate_up_hfq3g256_wmma",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_SRC,
            ["gemm_gate_up_hfq3g256_wmma"]
        );
        add!(
            "gemm_gate_up_hfq3g256_wmma_mb4",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_MB4_SRC,
            ["gemm_gate_up_hfq3g256_wmma_mb4"]
        );
        // DFlash draft full-attention WMMA tiles (hipfire-dispatch
        // `AttnFullF32`, HasWmmaW32 = RDNA3/3.5): B=8 `attention_dflash_wmma_f32`
        // and the B=16/head_dim=128 N64 tile `attention_dflash_wmma_n64_f32`.
        add!(
            "attention_dflash_wmma_f32",
            kernels::ATTENTION_DFLASH_WMMA_SRC,
            ["attention_dflash_wmma_f32"]
        );
        add!(
            "attention_dflash_wmma_n64_f32",
            kernels::ATTENTION_DFLASH_WMMA_N64_SRC,
            ["attention_dflash_wmma_n64_f32"]
        );
        add!(
            "gemm_mixed_moe_grouped_wmma_k2",
            kernels::GEMM_MIXED_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_mixed_moe_grouped_wmma_k2"]
        );
    }
    if arch == "gfx1201" {
        add!(
            "gemm_qkv_q8_0_wmma_gfx12",
            kernels::GEMM_QKV_Q8_0_WMMA_GFX12_SRC,
            ["gemm_qkv_q8_0_wmma_gfx12"]
        );
        add!(
            "gemm_qkvza_q8_0_wmma_gfx12",
            kernels::GEMM_QKVZA_Q8_0_WMMA_GFX12_SRC,
            ["gemm_qkvza_q8_0_wmma_gfx12"]
        );
        add!(
            "gemm_q8_0_residual_wmma_gfx12",
            kernels::GEMM_Q8_0_RESIDUAL_WMMA_GFX12_SRC,
            ["gemm_q8_0_residual_wmma_gfx12"]
        );
        add!(
            "gemm_gate_up_q8_0_wmma_gfx12",
            kernels::GEMM_GATE_UP_Q8_0_WMMA_GFX12_SRC,
            ["gemm_gate_up_q8_0_wmma_gfx12"]
        );
        // HFQ3 prefill gate/up on gfx12 (`gemm_gate_up_hfq3g256_wmma`
        // delegates here on RDNA4; the MB4 twin is gfx11-only).
        add!(
            "gemm_gate_up_hfq3g256_wmma_gfx12",
            kernels::GEMM_GATE_UP_HFQ3G256_WMMA_GFX12_SRC,
            ["gemm_gate_up_hfq3g256_wmma_gfx12"]
        );
        // DFlash draft full-attention WMMA tile, gfx12 lowering
        // (`attention_dflash_wmma_f32`, attention.rs: `has_wmma_w32_gfx12`).
        add!(
            "attention_dflash_wmma_f32_gfx12",
            kernels::ATTENTION_DFLASH_WMMA_GFX12_SRC,
            ["attention_dflash_wmma_f32_gfx12"]
        );

        // Mixed-dtype grouped MoE WMMA: `gemm_mixed_moe_grouped_wmma_impl`
        // picks the gfx12 module, or the coalesced twin (`..._coalesced`:
        // speculative-verify windows over host-mapped routed projections).
        add!(
            "gemm_mixed_moe_grouped_wmma_gfx12",
            kernels::GEMM_MIXED_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_mixed_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_mixed_moe_grouped_wmma_coalesced_gfx1201",
            kernels::GEMM_MIXED_MOE_GROUPED_WMMA_COALESCED_GFX1201_SRC,
            ["gemm_mixed_moe_grouped_wmma_coalesced_gfx1201"]
        );
        // `Gpu::gemm_gate_up_hfq4g256_wmma_gfx12` batch-tile family
        // (`gfx12_bt_tile`: bt4 for N in [64,128), bt8 at multiples of 128,
        // bt12 at multiples of 192 / N >= 192): one source, symbol == module.
        // `HIPFIRE_GATE_UP_BT=0` and the `_ldsstage` module are developer-only.
        add!(
            "gemm_gate_up_hfq4g256_wmma_gfx12_bt4",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_hfq4g256_wmma_gfx12_bt4"]
        );
        add!(
            "gemm_gate_up_hfq4g256_wmma_gfx12_bt8",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_hfq4g256_wmma_gfx12_bt8"]
        );
        add!(
            "gemm_gate_up_hfq4g256_wmma_gfx12_bt12",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_BT_SRC,
            ["gemm_gate_up_hfq4g256_wmma_gfx12_bt12"]
        );
    }
    // FirstRequestJit coverage batch G (ClassifyGemmTuples / ClassifyGemvTuples
    // holes): each module name, source expression and symbol is the one its
    // launcher passes to `ensure_kernel` on the default route of that arch.
    if arch == "gfx1151" {
        add!(
            "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151",
            kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC,
            ["fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"]
        );
        add!(
            "gemm_gate_up_hfq4g256_wmma_ldscoop_nosync",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_LDSCOOP_NOSYNC_SRC,
            ["gemm_gate_up_hfq4g256_wmma_ldscoop_nosync"]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_k2",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2_SRC,
            ["gemm_hfq4g256_residual_wmma_k2"]
        );
        add!(
            "gemm_hfq4g256_residual_wmma_k2x32",
            kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_K2X32_SRC,
            ["gemm_hfq4g256_residual_wmma_k2x32"]
        );
        add!(
            "gemm_hfq4g256_moe_grouped_mmq_gfx1151",
            kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX1151_SRC,
            ["gemm_hfq4g256_moe_grouped_mmq_gfx1151"]
        );
        add!(
            "gemm_hfq6g256_moe_grouped_wmma_gfx1151",
            kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX1151_SRC,
            ["gemm_hfq6g256_moe_grouped_wmma_gfx1151"]
        );
        add!(
            "gemm_mqv2_wmma_gfx1151_bt",
            kernels::gemm_mqv2_wmma_gfx11_bt_src(true),
            [
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq3g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq3g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq3g256v2_wmma_gfx11_bt6",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq3g256v2_residual_wmma_gfx11_bt8",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq6g256v2_wmma_gfx11_bt12",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq6g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt12",
                "gemm_gate_up_mq6g256v2_wmma_gfx11_bt6",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq6g256v2_residual_wmma_gfx11_bt8"
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_i4_b8_gfx1151",
            kernels::FUSED_RMSNORM_MQ_ROTATE_I4_B8_GFX1151_SRC,
            ["fused_rmsnorm_mq_rotate_i4_b8_gfx1151"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_i4_fold_b8",
            kernels::FUSED_RMSNORM_MQ_ROTATE_I4_FOLD_B8_SRC,
            ["fused_rmsnorm_mq_rotate_i4_fold_b8"]
        );
        add!(
            "gemv_hfq4g256_residual_rt_low_gfx1151",
            kernels::GEMV_HFQ4G256_RESIDUAL_RT_LOW_GFX1151_SRC,
            ["gemv_hfq4g256_residual_rt_low_gfx1151"]
        );
    }
    if arch == "gfx1201" {
        add!(
            "gemm_f16_wmma_mb8_gfx12",
            kernels::GEMM_F16_WMMA_MB8_GFX12_SRC,
            ["gemm_f16_wmma_mb8_gfx12"]
        );
        add!(
            "gemm_hfq6g256_moe_grouped_wmma_gfx12",
            kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_hfq6g256_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_mq6g256v2_moe_grouped_wmma_gfx12",
            kernels::GEMM_MQ6G256V2_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_mq6g256v2_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12",
            kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_GFX12_SRC,
            ["gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12"]
        );
        add!(
            "gemm_qkv_hfq4g256_wmma_gfx12_mq3v2",
            kernels::GEMM_QKV_MQ3G256V2_WMMA_GFX12_SRC,
            ["gemm_qkv_mq3g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkv_hfq4g256_wmma_gfx12_mq6v2",
            kernels::GEMM_QKV_MQ6G256V2_WMMA_GFX12_SRC,
            ["gemm_qkv_mq6g256v2_wmma_gfx12"]
        );
        add!(
            "gemm_qkv_mqv2_wmma_gfx1201_bt",
            kernels::GEMM_QKV_MQV2_WMMA_GFX1201_BT_SRC,
            [
                "gemm_qkv_mq6g256v2_wmma_gfx1201_bt8",
                "gemm_qkv_mq2g256v2_wmma_gfx1201_bt8",
                "gemm_qkv_mq3g256v2_wmma_gfx1201_bt8",
                "gemm_qkv_mq5g256v2_wmma_gfx1201_bt8",
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gfx12_v3",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_V3_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_v3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_v3",
                "quantize_int4_mmq_ds128",
                "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3",
            ]
        );
        add!(
            "fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab",
            kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SLAB_SRC,
            ["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
            kernels::FUSED_RMSNORM_MQ_ROTATE_I4_GFX12_V2_SRC,
            ["fused_rmsnorm_mq_rotate_i4_gfx12_v2"]
        );
        add!(
            "fused_silu_mul_mq_rotate_i4_gfx12",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_I4_GFX12_SRC,
            ["fused_silu_mul_mq_rotate_i4_gfx12"]
        );
        add!(
            "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
            kernels::GATED_NORM_MQ_ROTATE_I4_GFX12_V2_XBF16_SRC,
            ["gated_norm_mq_rotate_i4_gfx12_v2_xbf16"]
        );
        add!(
            "mq_rotate_x_awq_i4_gfx12",
            kernels::MQ_ROTATE_X_AWQ_I4_GFX12_SRC,
            ["rotate_x_mq_awq_i4_gfx12"]
        );
        add!(
            "mq_rotate_x_i4_gfx12",
            kernels::MQ_ROTATE_X_I4_GFX12_SRC,
            ["mq_rotate_x_i4_gfx12"]
        );
    }
    if arch == "gfx1100" {
        add!(
            "gemm_gate_up_hfq4g256_wmma",
            kernels::GEMM_GATE_UP_HFQ4G256_WMMA_SRC,
            [
                "gemm_gate_up_hfq4g256_wmma",
                "gemm_gate_up_hfq4g256_wmma_ldsstage",
            ]
        );
        add!(
            "gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu",
            kernels::GEMM_HFQ4G256_MOE_GROUPED_MMQ_GFX11_DGPU_SRC,
            ["gemm_hfq4g256_moe_grouped_mmq_gfx11_dgpu"]
        );
        add!(
            "gemm_hfq6g256_moe_grouped_wmma_k2",
            kernels::GEMM_HFQ6G256_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_hfq6g256_moe_grouped_wmma_k2"]
        );
        add!(
            "fused_rmsnorm_mq_rotate_i4_b8",
            kernels::FUSED_RMSNORM_MQ_ROTATE_I4_B8_SRC,
            ["fused_rmsnorm_mq_rotate_i4_b8"]
        );
        add!(
            "gated_norm_mq_rotate_k6144_gfx1100",
            kernels::gated_norm_mq_rotate_k6144_gfx1100_src(),
            ["gated_norm_mq_rotate_k6144_gfx1100"]
        );
        add!(
            "gemv_hfq3g256_rdna3",
            kernels::GEMV_HFQ3G256_GFX1100_SRC,
            ["gemv_hfq3g256"]
        );
        add!(
            "gemv_hfq3g256_residual_rdna3",
            kernels::GEMV_HFQ3G256_RESIDUAL_GFX1100_SRC,
            ["gemv_hfq3g256_residual"]
        );
        add!(
            "gemv_hfq4g256_residual_stage_x32_gfx1100",
            kernels::GEMV_HFQ4G256_RESIDUAL_STAGE_X32_GFX1100_SRC,
            ["gemv_hfq4g256_residual"]
        );
        add!(
            "gemv_hfq4g256_residual_sigmoid_buffer_gfx1100",
            kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_BUFFER_GFX1100_SRC,
            ["gemv_hfq4g256_residual_sigmoid_scaled_gpu"]
        );
        add!(
            "gemv_hfq4g256_moe_gate_up_indexed_cpol_slc",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_SLC_GFX1100_SRC,
            ["gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!(
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            [
                "gemm_hfq4g256_residual_mmq",
                "gemm_hfq4g256_residual_mmq_full_add",
                "quantize_q8_1_mmq_ds4",
                "gemm_hfq4g256_residual_mmq_full_set",
            ]
        );
        add!(
            "gemm_mq6g256v2_moe_grouped_wmma_k2",
            kernels::GEMM_MQ6G256V2_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_mq6g256v2_moe_grouped_wmma_k2"]
        );
        add!(
            "gemm_mq2g256_lloyd_moe_grouped_wmma_k2",
            kernels::GEMM_MQ2G256_LLOYD_MOE_GROUPED_WMMA_K2_SRC,
            ["gemm_mq2g256_lloyd_moe_grouped_wmma_k2"]
        );
        add!(
            "gemm_gate_up_mq3g256v2_wmma",
            kernels::GEMM_GATE_UP_MQ3G256V2_WMMA_SRC,
            ["gemm_gate_up_mq3g256v2_wmma"]
        );
        add!(
            "gemm_qkv_mq3g256v2_wmma",
            kernels::GEMM_QKV_MQ3G256V2_WMMA_SRC,
            ["gemm_qkv_mq3g256v2_wmma"]
        );
        add!(
            "gemm_qkvza_mq3g256v2_wmma",
            kernels::GEMM_QKVZA_MQ3G256V2_WMMA_SRC,
            ["gemm_qkvza_mq3g256v2_wmma"]
        );
        add!(
            "gemm_mq3g256v2_residual_wmma",
            kernels::GEMM_MQ3G256V2_RESIDUAL_WMMA_SRC,
            ["gemm_mq3g256v2_residual_wmma"]
        );
        add!(
            "gemm_gate_up_mq6g256v2_wmma",
            kernels::GEMM_GATE_UP_MQ6G256V2_WMMA_SRC,
            ["gemm_gate_up_mq6g256v2_wmma"]
        );
        add!(
            "gemm_qkv_mq6g256v2_wmma",
            kernels::GEMM_QKV_MQ6G256V2_WMMA_SRC,
            ["gemm_qkv_mq6g256v2_wmma"]
        );
        add!(
            "gemm_qkvza_mq6g256v2_wmma",
            kernels::GEMM_QKVZA_MQ6G256V2_WMMA_SRC,
            ["gemm_qkvza_mq6g256v2_wmma"]
        );
        add!(
            "fused_silu_mul_mq_rotate_i4",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_I4_SRC,
            ["fused_silu_mul_mq_rotate_i4"]
        );
        add!(
            "mq_rotate_x_awq_i4",
            kernels::MQ_ROTATE_X_AWQ_I4_SRC,
            ["rotate_x_mq_awq_i4"]
        );
    }
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        add!(
            "gemm_mq4g128v2_multirow_gfx1151",
            kernels::GEMM_MQ4G128V2_MULTIROW_GFX1151_SRC,
            ["gemm_mq4g128v2_multirow_gfx1151"]
        );
        add!(
            "gemv_mq4g128v2",
            kernels::GEMV_MQ4G128V2_SRC,
            ["gemv_mq4g128v2"]
        );
        add!(
            "moe_topk_renorm_top10_batched",
            kernels::MOE_TOPK_RENORM_TOP10_BATCHED_SRC,
            [
                "moe_topk_renorm_top10_batched",
                "moe_router_logits_guard_top10"
            ]
        );
        add!(
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"]
        );
        // `k8all` is requested only by the exact-gfx1151 DeepSeek4 K=2048 route
        // (gemv.rs:21118-21125, `deepseek4_gfx1151_route && is_gfx1151() && k == 2048`).
        if matches!(arch, "gfx1100" | "gfx1201") {
            add!(
                "gemv_mq2g256_lloyd_moe_down_indexed",
                kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
                ["gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"]
            );
        }
        if arch == "gfx1151" {
            add!(
                "gemv_mq2g256_lloyd_moe_down_indexed",
                kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
                [
                    "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
                    "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
                ]
            );
        }
        add!(
            "gemv_mq2g256_lloyd_moe_gate_up_indexed",
            kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
            ["gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"]
        );
        add!(
            "gemv_mq3g256_lloyd_moe_down_indexed",
            kernels::GEMV_MQ3G256_LLOYD_MOE_DOWN_INDEXED_SRC,
            ["gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed"]
        );
    }
    if matches!(arch, "gfx1151" | "gfx1201") {
        add!(
            "gemv_hfq3g256",
            kernels::GEMV_HFQ3G256_SRC,
            ["gemv_hfq3g256"]
        );
        add!(
            "gemv_hfq3g256_residual",
            kernels::GEMV_HFQ3G256_RESIDUAL_SRC,
            ["gemv_hfq3g256_residual"]
        );
        add!(
            "gemv_hfq4g256_multirow_default_mq3v2",
            kernels::GEMV_MQ3G256V2_MULTIROW_SRC,
            [
                "gemv_mq3g256v2_multirow_r2",
                "gemv_mq3g256v2_multirow_r4",
                "gemv_mq3g256v2_multirow_r8",
            ]
        );
    }
    if arch == "gfx1100" {
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gfx11_symfold",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX11_SYMFOLD_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold",
                "quantize_int4_mmq_ds128",
            ]
        );
    }
    if arch == "gfx1151" {
        add!(
            "gemm_mq4g256v2_residual_mmq_iu4_gfx11_symfold",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX11_SYMFOLD_SRC,
            [
                "gemm_mq4g256v2_residual_mmq_iu4",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold",
                "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold",
                "quantize_int4_mmq_ds128",
            ]
        );
    }
    // `moe_down_combine_rmsnorm_mq_rotate_vecsum_*` (moe.rs): the exact-arch
    // down+combine+RMSNorm+rotate tails, previously only in
    // `default_route_entries`; the symbols are those launchers' own.
    if arch == "gfx1100" {
        add!(
            "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100",
            kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
            ["moe_down_combine_rmsnorm_mq_rotate_vecsum"]
        );
    }
    if arch == "gfx1151" {
        add!(
            "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
            kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC,
            ["moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151"]
        );
    }
    // Sampled DFlash host-accept fallback on gfx1201 (the device accept path
    // is the default there; `dflash_target_rows_into` is the host fallback).
    if arch == "gfx1201" {
        add!(
            "sampled_rows_compact",
            crate::sampled_rows_compact::SAMPLED_ROWS_COMPACT_SRC,
            ["sampled_rows_compact_f32"]
        );
    }
    // MQ4V2 weight-reuse batch-tile modules (`mqv2_prefill_batch_tile`): under
    // the gfx1100/gfx1151 batch >= 64 IU4/A8/Q8_1 MMQ guard they are reached
    // for N < 64 launches whose `route_rows` is >= 96 (Qwen4 prefill-invariant
    // scope, default Flash-Next), and on gfx1201 the QKV BT8 tile when the IU4
    // route does not take the launch. Module name per arch as each launcher
    // names it; symbols are every arm the launcher can select.
    if arch == "gfx1100" {
        add!(
            "gemm_qkvza_mq4g256v2_wmma_gfx11_bt",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_qkvza_mq4g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_qkv_mq4g256v2_wmma_gfx11_bt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_qkv_mq4g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_gate_up_mq4g256v2_wmma_gfx11_bt",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_gate_up_mq4g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_wmma_gfx11_bt",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX11_BT_SRC,
            [
                "gemm_mq4g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq4g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            ]
        );
    }
    if arch == "gfx1151" {
        add!(
            "gemm_qkvza_mq4g256v2_wmma_gfx1151_bt",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_qkvza_mq4g256v2_wmma_gfx11_bt4",
                "gemm_qkvza_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_qkv_mq4g256v2_wmma_gfx1151_bt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_qkv_mq4g256v2_wmma_gfx11_bt4",
                "gemm_qkv_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_gate_up_mq4g256v2_wmma_gfx1151_bt",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX11_BT_SRC,
            [
                "gemm_gate_up_mq4g256v2_wmma_gfx11_bt6",
                "gemm_gate_up_mq4g256v2_wmma_gfx11_bt12"
            ]
        );
        add!(
            "gemm_mq4g256v2_residual_wmma_gfx1151_bt",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX11_BT_SRC,
            [
                "gemm_mq4g256v2_residual_wmma_gfx11_bt4",
                "gemm_mq4g256v2_residual_wmma_gfx11_bt6",
                "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            ]
        );
    }
    if arch == "gfx1201" {
        add!(
            "gemm_qkv_mq4g256v2_wmma_gfx1201_bt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX1201_BT_SRC,
            [
                "gemm_qkv_mq4g256v2_wmma_gfx1201_bt4",
                "gemm_qkv_mq4g256v2_wmma_gfx1201_bt8",
                "gemm_qkv_mq4g256v2_wmma_gfx1201_bt12"
            ]
        );
    }
    Ok(entries)
}

fn entry(
    arch: &'static str,
    module: &'static str,
    symbols: &'static [&'static str],
    source: Cow<'static, str>,
    extra_flags: &str,
) -> KernelEntry {
    let recipe = KernelCompiler::recipe_for_source(arch, module, &source, extra_flags);
    KernelEntry {
        arch,
        module,
        symbols,
        flags: recipe.flags,
        scheduler_profile: recipe.scheduler_profile,
        source,
    }
}

/// JIT modules that the admitted default Redline programs load
/// (`hipfire-runtime` `retained_redline_default`: MQ4R on gfx1100/gfx1151/
/// gfx1201, Qwen3.5 dense on gfx1201, DeepSeek4 MQ2R on gfx1151) and that the
/// installer inventory [`entries`] does not carry. Module names are the ones
/// the programs' captures record; each source is the expression the cited
/// launcher passes to `ensure_kernel` on the default route. railgun's JIT
/// receipt corpus compiles [`corpus_entries`], never a hand-picked source.
pub fn default_route_entries(
    arch: &str,
    extra_flags: &str,
) -> Result<Vec<KernelEntry>, RegistryError> {
    let arch: &'static str = SUPPORTED_ARCHES
        .iter()
        .copied()
        .find(|&candidate| candidate == arch)
        .ok_or_else(|| RegistryError::UnsupportedArch(arch.to_owned()))?;
    let mut entries = Vec::new();
    macro_rules! add {
        ($module:literal, $src:expr, [$($symbol:literal),+ $(,)?]) => {
            entries.push(entry(arch, $module, &[$($symbol),+], ($src).into(), extra_flags))
        };
    }
    let rdna3 = matches!(arch, "gfx1100" | "gfx1151");
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        // MoE routing and expert GEMVs shared by the three MQ4R programs.
        add!(
            "gated_delta_net_q8_compact2_b2",
            kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC,
            ["gated_delta_net_q8_compact2_b2"]
        ); // norm.rs:2994
        add!(
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
            kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC,
            ["gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"]
        ); // gemv.rs:14599
        add!(
            "moe_down_combine_k8_batched",
            kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC,
            ["moe_down_combine_k8_batched"]
        ); // moe.rs:41
    }
    if matches!(arch, "gfx1151" | "gfx1201") {
        add!(
            "gemv_hfq4g256_moe_gate_up_indexed",
            kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC,
            ["gemv_hfq4g256_moe_gate_up_k8_indexed"]
        ); // gemv.rs:13888
           // gemv.rs:12202,12266,12379: three launchers share the module.
        add!(
            "gemv_hfq4g256_residual_scaled",
            kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC,
            [
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_residual_scaled_gpu",
                "gemv_hfq4g256_residual_scaled_cpu"
            ]
        );
    }
    if rdna3 {
        add!(
            "conv1d_silu_split_qknorm_b256",
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
            ["conv1d_silu_split_qknorm_b256"]
        ); // norm.rs:4459
        add!(
            "fused_rmsnorm_mq_rotate_vecsum",
            kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
            ["fused_rmsnorm_mq_rotate_vecsum"]
        ); // gemv.rs:2477
        add!(
            "kv_cache_write_q8_0_pair",
            kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC,
            ["kv_cache_write_q8_0_pair"]
        ); // attention.rs:2502
        add!(
            "moe_router_softmax_topk_k8_wave64_exact",
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC,
            ["moe_router_softmax_topk_k8_wave64_exact"]
        ); // gemv.rs:13029
    }
    match arch {
        "gfx1100" => {
            add!(
                "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
                kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC,
                ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"]
            ); // attention.rs:10626
            add!(
                "gated_norm_mq_rotate_gfx1100",
                kernels::GATED_NORM_MQ_ROTATE_GFX1100_SRC,
                ["gated_norm_mq_rotate_gfx1100"]
            ); // gemv.rs:4796
            add!(
                "gemv_hfq4g256_moe_gate_up_indexed_cpol_slc",
                kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_SLC_GFX1100_SRC,
                ["gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"]
            ); // gemv.rs:13770
            add!(
                "gemv_hfq4g256_residual_sigmoid_buffer_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_BUFFER_GFX1100_SRC,
                ["gemv_hfq4g256_residual_sigmoid_scaled_gpu"]
            ); // gemv.rs:12369
            add!(
                "gemv_hfq4g256_residual_stage_x32_gfx1100",
                kernels::GEMV_HFQ4G256_RESIDUAL_STAGE_X32_GFX1100_SRC,
                ["gemv_hfq4g256_residual"]
            ); // gemv.rs:10262
            add!(
                "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100",
                kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
                ["moe_down_combine_rmsnorm_mq_rotate_vecsum"]
            ); // moe.rs:119
            add!(
                "qwen35_fa_prep_gfx1100",
                kernels::QWEN35_FA_PREP_GFX1100_SRC,
                ["qwen35_fa_prep_gfx1100"]
            ); // norm.rs:1152
        }
        "gfx1151" => {
            add!(
                "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
                kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC,
                ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"]
            ); // attention.rs:10620
            add!(
                "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC,
                ["fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"]
            ); // gemm.rs:3811
            add!(
                "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151",
                kernels::FUSED_QKVZA_HFQ4G256_K2048_HYBRID_BUFFER_GFX1151_SRC,
                ["fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"]
            ); // gemm.rs:3800
            add!(
                "gated_norm_mq_rotate_gfx1151",
                kernels::GATED_NORM_MQ_ROTATE_GFX1151_SRC,
                ["gated_norm_mq_rotate_gfx1151"]
            ); // gemv.rs:4783
            add!(
                "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151",
                kernels::GEMV_HFQ4G256_LM_HEAD_R1_HYBRID_BUFFER_GFX1151_SRC,
                ["gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"]
            ); // gemv.rs:9892
            add!(
                "gemv_hfq4g256_residual_rt_low_gfx1151",
                kernels::GEMV_HFQ4G256_RESIDUAL_RT_LOW_GFX1151_SRC,
                ["gemv_hfq4g256_residual_rt_low_gfx1151"]
            ); // gemv.rs:10226
            add!(
                "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
                kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC,
                ["moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151"]
            ); // moe.rs:115
            add!(
                "qwen35_fa_prep_gfx1151",
                kernels::QWEN35_FA_PREP_GFX1151_SRC,
                ["qwen35_fa_prep_gfx1151"]
            ); // norm.rs:1146
               // DeepSeek4 MQ2R gfx1151 v2 route (registry/deepseek4-mq2r-gfx1151-v2.json).
            add!(
                "compressor_add_ape_buf",
                kernels::COMPRESSOR_ADD_APE_BATCHED_SRC,
                ["compressor_add_ape_f32_buf"]
            ); // attention.rs:14675
            add!(
                "compressor_overlap_concat",
                kernels::COMPRESSOR_OVERLAP_CONCAT_SRC,
                ["compressor_overlap_concat_f32"]
            ); // attention.rs:14720
            add!(
                "compressor_softmax_pool_f32_buf",
                kernels::COMPRESSOR_SOFTMAX_POOL_BUF_SRC,
                ["compressor_softmax_pool_f32_buf"]
            ); // attention.rs:14852
            add!(
                "deepseek4_attn_swa_buf",
                kernels::V4F_ATTN_SWA_BUF_SRC,
                ["deepseek4_attn_swa_buf"]
            ); // attention.rs:18544
            add!(
                "deepseek4_attn_swa_topk_scoregrid_f32_buf",
                kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
                ["deepseek4_attn_swa_topk_scoregrid_f32_buf"]
            ); // attention.rs:19068
            add!(
                "deepseek4_fused_silu_mul_clamp_mq_rotate",
                kernels::V4F_FUSED_SILU_MUL_CLAMP_MQ_ROTATE_SRC,
                ["deepseek4_fused_silu_mul_clamp_mq_rotate"]
            ); // norm.rs:6165
            add!(
                "deepseek4_moe_topk_bias_aware",
                kernels::V4F_MOE_TOPK_BIAS_AWARE_SRC,
                ["deepseek4_moe_topk_bias_aware_f32"]
            ); // moe.rs:878
            add!(
                "deepseek4_silu_mul_clamp",
                kernels::V4F_SILU_MUL_CLAMP_SRC,
                ["deepseek4_silu_mul_clamp_f32"]
            ); // norm.rs:6224
            add!(
                "deepseek4_topk_kv_gather_f32_buf",
                kernels::V4F_TOPK_KV_GATHER_BUF_SRC,
                ["deepseek4_topk_kv_gather_f32_buf"]
            ); // moe.rs:1159
            add!(
                "deepseek4_topk_kv_gather_identity_f32_buf",
                kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC,
                ["deepseek4_topk_kv_gather_identity_f32_buf"]
            ); // moe.rs:1410
            add!(
                "fused_rmsnorm_mq_rotate_plain_nox",
                kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC,
                ["fused_rmsnorm_mq_rotate_plain_nox"]
            ); // norm.rs:5946
            add!(
                "gemv_mfp4g32_e8_soa_grouped_gfx1151",
                kernels::GEMV_MFP4G32_E8_SOA_GROUPED_GFX1151_SRC,
                ["gemv_mfp4g32_e8_soa_grouped_gfx1151"]
            ); // gemv.rs:7564,8025
            add!(
                "gemv_mfp4g32_e8_soa_u4",
                kernels::GEMV_MFP4G32_E8_SOA_U4_SRC,
                ["gemv_mfp4g32_e8_soa_u4"]
            ); // gemv.rs:8075,7448
            add!(
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed",
                kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC,
                ["gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"]
            ); // gemv.rs:16927
            add!(
                "gemv_mq2g256_lloyd_moe_gate_up_indexed",
                kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC,
                ["gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"]
            ); // gemv.rs:17258
            add!(
                "hash_router_normalize_f32_buf",
                kernels::HASH_ROUTER_NORMALIZE_BUF_SRC,
                ["hash_router_normalize_f32_buf"]
            ); // moe.rs:770
            add!(
                "hc_compute_control_vec4_finalize",
                kernels::HC_COMPUTE_CONTROL_SRC,
                ["hc_compute_control_vec4_finalize"]
            ); // attention.rs:15414
            add!(
                "hc_head_compute_pre",
                kernels::HC_HEAD_COMPUTE_PRE_SRC,
                ["hc_head_compute_pre"]
            ); // attention.rs:15697
            add!(
                "hc_input_map_4stream",
                kernels::HC_INPUT_MAP_SRC,
                ["hc_input_map_4stream"]
            ); // attention.rs:15751
            add!(
                "hc_mix_4stream",
                kernels::HC_MIX_4STREAM_SRC,
                ["hc_mix_4stream"]
            ); // attention.rs:15835
            add!(
                "indexer_relu_score_f32_buf",
                kernels::INDEXER_RELU_SCORE_BUF_SRC,
                ["indexer_relu_score_f32_buf"]
            ); // attention.rs:16916
            add!(
                "indexer_top_k_buf_parallel",
                kernels::INDEXER_TOP_K_BUF_PARALLEL_GFX1151_SRC,
                ["indexer_top_k_buf_parallel"]
            ); // attention.rs:17320
            add!(
                "rmsnorm_f32_at_slot_buf",
                kernels::RMSNORM_AT_SLOT_BUF_SRC,
                ["rmsnorm_f32_at_slot_buf"]
            ); // norm.rs:6089
            add!(
                "rope_tail_interleaved",
                kernels::ROPE_TAIL_INTERLEAVED_SRC,
                ["rope_tail_interleaved_f32"]
            ); // attention.rs:17562
            add!(
                "rope_tail_yarn_interleaved_at_slot_buf",
                kernels::ROPE_TAIL_YARN_INTERLEAVED_AT_SLOT_BUF_SRC,
                ["rope_tail_yarn_interleaved_at_slot_buf_f32"]
            ); // attention.rs:17838
            add!(
                "rope_tail_yarn_interleaved_wide",
                kernels::ROPE_TAIL_YARN_INTERLEAVED_SRC,
                ["rope_tail_yarn_interleaved_wide_f32"]
            ); // attention.rs:17768
            add!(
                "sqrt_softplus_f32",
                kernels::SQRT_SOFTPLUS_F32_SRC,
                ["sqrt_softplus_f32"]
            ); // norm.rs:6128
            add!(
                "state_overlap_shift_f32_buf",
                kernels::STATE_OVERLAP_SHIFT_F32_BUF_SRC,
                ["state_overlap_shift_f32_buf"]
            ); // attention.rs:17988
            add!(
                "state_ring_write_f32_buf",
                kernels::STATE_RING_WRITE_F32_BUF_SRC,
                ["state_ring_write_f32_buf"]
            ); // attention.rs:18031
            add!(
                "swa_ring_write_f32_buf",
                kernels::SWA_RING_WRITE_BUF_SRC,
                ["swa_ring_write_f32_buf"]
            ); // attention.rs:18171
        }
        "gfx1201" => {
            add!(
                "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
                kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC,
                [
                    "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
                    "attention_flash_fp8_e4m3_tile_gqa8_gfx1201"
                ]
            ); // attention.rs:7783
            add!(
                "attention_flash_reduce_dsplit_gfx1201",
                kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC,
                ["attention_flash_reduce_dsplit_gfx1201"]
            ); // attention.rs:7784
            add!(
                "gemv_hfq4g256_multirow_default",
                kernels::GEMV_HFQ4G256_MULTIROW_SRC,
                ["gemv_hfq4g256_multirow_r2"]
            ); // gemv.rs:10031
               // Qwen3.6-27B decode fusions on exact gfx1201 (3be1cdbe9): the H2 tape's compact-3 GDN,
               // 48-head AWQ gated-norm/MQ rotation and 24Q/4K FA prep.
            add!(
                "gated_delta_net_q8_compact3_b2",
                kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC,
                ["gated_delta_net_q8_compact3_b2"]
            ); // norm.rs:3045
            add!(
                "gated_norm_mq_rotate_awq_k6144_gfx1201",
                kernels::gated_norm_mq_rotate_awq_k6144_gfx1201_src(),
                ["gated_norm_mq_rotate_awq_k6144_gfx1201"]
            ); // gemv.rs:4751
            #[cfg(feature = "deltanet")]
            add!(
                "qwen36_27b_fa_prep_gfx1201",
                kernels::qwen36_27b_fa_prep_gfx1201_src(),
                ["qwen36_27b_fa_prep_gfx1201"]
            ); // norm.rs:1149
            add!(
                "moe_router_softmax_topk_k8_wave64",
                kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC,
                ["moe_router_softmax_topk_k8_wave64"]
            ); // gemv.rs:12992
        }
        _ => {}
    }
    Ok(entries)
}

/// Modules railgun's own lowering launches (design §1.1 `Node::Copy` and
/// `Node::CopyBatch`): the hipfire-owned copy kernels, JIT-compiled as module
/// `railgun::copy::MODULE` from `railgun::copy::SOURCE`, on every arch
/// railgun certifies, the typed fill, and on exact gfx1201 the DFlash window's
/// device selector. No Redline program loads them.
pub fn railgun_entries(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let arch: &'static str = SUPPORTED_ARCHES
        .iter()
        .copied()
        .find(|&candidate| candidate == arch)
        .ok_or_else(|| RegistryError::UnsupportedArch(arch.to_owned()))?;
    let mut entries = Vec::new();
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        entries.push(entry(
            arch,
            "railgun_copy",
            &["railgun_copy", "railgun_copy_batch"],
            kernels::RAILGUN_COPY_SRC.into(),
            extra_flags,
        ));
        entries.push(entry(
            arch,
            "railgun_fill",
            &["railgun_fill_u32"],
            kernels::RAILGUN_FILL_SRC.into(),
            extra_flags,
        ));
    }
    // The certified DFlash window's (design A, exact gfx1201) device greedy
    // selector: a node of the window program, JIT-loaded by
    // `Gpu::ensure_dflash_selector_greedy_exact`. The window refuses any node
    // whose exact object has no corpus record, so it is inventoried here.
    if arch == "gfx1201" {
        entries.push(entry(
            arch,
            crate::dflash_selector::DFLASH_SELECTOR_GREEDY_EXACT_MODULE,
            &[crate::dflash_selector::DFLASH_SELECTOR_GREEDY_EXACT_SYMBOL],
            crate::dflash_selector::DFLASH_SELECTOR_GREEDY_EXACT_SRC.into(),
            extra_flags,
        ));
    }
    Ok(entries)
}

/// Every module railgun's JIT receipt corpus may compile for `arch`: the
/// installer inventory followed by [`default_route_entries`] and
/// [`railgun_entries`]. A module name appears once per arch: a default-route
/// module the installer inventory already carries with the same source is
/// listed once (a differing source stays and fails the uniqueness test).
pub fn corpus_entries(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let mut all = entries(arch, extra_flags)?;
    for entry in default_route_entries(arch, extra_flags)? {
        if !all
            .iter()
            .any(|e| e.module == entry.module && e.source == entry.source)
        {
            all.push(entry);
        }
    }
    all.extend(railgun_entries(arch, extra_flags)?);
    Ok(all)
}

pub fn lookup(arch: &str, module: &str, extra_flags: &str) -> Result<KernelEntry, RegistryError> {
    entries(arch, extra_flags)?
        .into_iter()
        .find(|entry| entry.module == module)
        .ok_or_else(|| RegistryError::UnsupportedModule {
            arch: arch.to_owned(),
            module: module.to_owned(),
        })
}

/// gfx1100 slab single-symbol modules (K3 A4 producer, codec tile mask).
static SLAB_GFX1100_K3_SYMBOLS: [&str; 1] = [crate::slab_moe::MQ_I4_A4_GFX1100_SYMBOL];
static SLAB_GFX1100_MASK_SYMBOLS: [&str; 1] = [crate::slab_moe::EP_TILE_MASK_GFX1100_MODULE];

/// Narrow and t8 exports share the admitted gfx1100 K1 images.
static SLAB_GFX1100_K1_SYMBOLS: std::sync::LazyLock<Vec<&str>> = std::sync::LazyLock::new(|| {
    let k1 = &crate::slab_moe::slab_arch_kernels("gfx1100")
        .unwrap()
        .mq4_k1;
    let tiled = crate::slab_moe::slab_rowtile_k1(k1).unwrap();
    k1.symbols.into_iter().chain(tiled.symbols).collect()
});
static SLAB_GFX1100_E8P_K1_SYMBOLS: std::sync::LazyLock<Vec<&str>> =
    std::sync::LazyLock::new(|| {
        let k1 = &crate::slab_moe::slab_cold_k1_metadata("gfx1100", 3)
            .unwrap()
            .k1;
        let tiled = crate::slab_moe::slab_rowtile_k1(k1).unwrap();
        k1.symbols.into_iter().chain(tiled.symbols).collect()
    });

/// Embedded certified-builder images of the sealed expert-slab serving route on `arch` for a
/// layer whose cold bucket is codec `cold_codec` (K1 decode GEMVs, K2 NT1 grouped GEMMs, K3 down
/// A4 producer and the codec tile mask), as closed-route [`PlannedKernel::Embedded`] entries that
/// [`validate_route_plan`] accepts. gfx1100 includes narrow/t8 K1 exports in
/// each admitted image; gfx1201/gfx1151 keep their runtime-managed preload.
/// Qualification twins are never listed. Unknown arches are refused.
///
/// gfx1100 `cold_codec == 0` (no cold bucket, MQ4-only) lists the hot K1, the MQ4 K2, K3 and the
/// mask only: no E8P module. `cold_codec == 3` (E8P) additionally lists the cold K1 and the E8P
/// K2 once each. Registry presence is not permission: the cold entries come only from the
/// admission-gated [`crate::slab_moe::slab_cold_kernels`] and `slab_k2_entry`, so every other
/// codec (W2, gfx1100 W3, unknown) is an [`RegistryError::UnsupportedRoute`].
pub fn slab_embedded_entries(
    arch: &str,
    cold_codec: u8,
) -> Result<Vec<PlannedKernel>, RegistryError> {
    if !SUPPORTED_ARCHES.contains(&arch) {
        return Err(RegistryError::UnsupportedArch(arch.to_owned()));
    }
    let embedded =
        |module: &'static str, image: &'static [u8], symbols: &'static [&'static str]| {
            PlannedKernel::Embedded {
                module,
                image,
                radiowave_json: None,
                symbols,
            }
        };
    match arch {
        "gfx1100" => {
            let slab = crate::slab_moe::slab_arch_kernels(arch).expect("gfx1100 slab table");
            let k2_mq4 = crate::gemm::slab_k2_entry(arch, 0).expect("gfx1100 MQ4 K2 entry");
            let (k3_module, k3_image, _) = slab.k3.expect("gfx1100 K3 entry");
            let crate::slab_moe::SlabTileMask::Embedded {
                module: mask_module,
                image: mask_image,
                ..
            } = slab.tile_mask
            else {
                panic!("gfx1100 embedded tile mask");
            };
            let mut plan = vec![
                embedded(
                    slab.mq4_k1.module,
                    slab.mq4_k1.image,
                    &SLAB_GFX1100_K1_SYMBOLS,
                ),
                embedded(k2_mq4.module, k2_mq4.image, &k2_mq4.entries),
                embedded(k3_module, k3_image, &SLAB_GFX1100_K3_SYMBOLS),
                embedded(mask_module, mask_image, &SLAB_GFX1100_MASK_SYMBOLS),
            ];
            if cold_codec != 0 {
                let refused = || {
                    RegistryError::UnsupportedRoute(format!(
                        "gfx1100 expert-slab cold codec {cold_codec} has no admitted K1 + K2 entries"
                    ))
                };
                let cold =
                    crate::slab_moe::slab_cold_kernels(arch, cold_codec).ok_or_else(refused)?;
                let k2_cold = crate::gemm::slab_k2_entry(arch, cold_codec).ok_or_else(refused)?;
                plan.push(embedded(
                    cold.k1.module,
                    cold.k1.image,
                    &SLAB_GFX1100_E8P_K1_SYMBOLS,
                ));
                plan.push(embedded(k2_cold.module, k2_cold.image, &k2_cold.entries));
            }
            if WideTwinFamily::K1.enabled() {
                plan.push(rowtile_planner_entry(arch).expect("gfx1100 rowtile planner"));
            }
            Ok(plan)
        }
        // gfx1201's slab images keep their historical preload; the other arches have none.
        _ => Ok(Vec::new()),
    }
}

// ── A3: header-selected route closure ──────────────────────────────────
//
// `route_entries` maps an already-resolved load (header tensor metadata,
// arch, KV/state axes, spec route, process-frozen feature flags) to every
// kernel module that route can request from load through prefill, decode,
// sampling, retained replay and (for native MTP) draft/verify/rollback, so a
// loader can compile and load the whole closure before the first dispatch.
// It is CPU-only and pure: no weight payload, compiler, or HIP call.
//
// Two kinds of routes are admitted. The Qwen3.6-27B closure was walked from
// source and cross-checked against a module-load audit of every request
// family. Walked routes (`route_plans/*.tsv`) are generated, not written:
// `scripts/route_plan_walk.py` drives the real dispatch through a route's
// request axes under `HIPFIRE_MODULE_LOAD_AUDIT`, and
// `scripts/route_plan_build.py` records every (module, symbol) it bound
// with its walk-time source or image digest. That digest is historical
// provenance only: an admitted load emits the current registry
// module/source/cache recipe for every walked row (unique module, walked
// symbols, current flags/profile/header/ABI/capture/shape), so drift between
// the capture and the current Beta registry fails as a real recipe
// incompatibility, never as a repinned literal. Everything else (other
// explicit drafter routes, non-default feature flags or policy axes) is
// `UnsupportedRoute`, and the caller keeps lazy loading. Closures assume the
// default developer environment (no `HIPFIRE_*` developer override of a
// launcher selector); such overrides are outside this plan and are caught by
// the loader's post-ready barrier, not here.

/// Borrowed header tensor metadata: name, storage dtype and shape.
#[derive(Clone, Copy, Debug)]
pub struct KernelTensorMeta<'a> {
    pub name: &'a str,
    pub dtype: crate::DType,
    pub shape: &'a [usize],
}

/// The upper layer's already-resolved speculative route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelSpecRoute {
    Ar,
    NativeMtp,
    DFlash,
    DDTree,
    DSpark,
}

/// Everything a route closure depends on. `tensors` lists the primary model
/// header plus, for a sidecar native-MTP head, the head's tensors.
pub struct RouteKernelInput<'a> {
    pub arch: &'a str,
    pub model_arch: u32,
    pub tensors: &'a [KernelTensorMeta<'a>],
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub prefill_chunk_rows: usize,
    pub kv_k: &'a str,
    pub kv_v: &'a str,
    pub state_quant: &'a str,
    pub spec: KernelSpecRoute,
    pub host_mapped_experts: bool,
    pub retained_decode: bool,
    pub hipcc_extra_flags: &'a str,
    pub flags: &'a crate::feature_flags::FeatureFlags,
    /// Model-specific policy axes a walked route plan must match exactly
    /// (`#! axis` lines of `route_plans/*.tsv`), e.g. `("qsa", "fp8")`.
    /// Empty for routes without a walked plan.
    pub route_axes: &'a [(&'a str, String)],
}

#[derive(Debug)]
pub enum PlannedKernel {
    Hip(KernelEntry),
    Embedded {
        module: &'static str,
        image: &'static [u8],
        /// Radiowave manifest verified against `image` at admission.
        radiowave_json: Option<&'static str>,
        symbols: &'static [&'static str],
    },
}

impl PlannedKernel {
    pub fn module(&self) -> &'static str {
        match self {
            Self::Hip(entry) => entry.module,
            Self::Embedded { module, .. } => module,
        }
    }

    pub fn symbols(&self) -> &'static [&'static str] {
        match self {
            Self::Hip(entry) => entry.symbols,
            Self::Embedded { symbols, .. } => symbols,
        }
    }
}

/// Whether the loader may seal a resolved route plan. A walked plan is
/// [`RouteSeal::Open`] (off-plan loads stay lazy and warn) unless its file
/// names the spec route in `#! sealed`, which `scripts/route_plan_build.py`
/// writes only from a sealed replay of the checked-in plan; the hand-closed
/// Qwen3.6-27B gfx1201 route is [`RouteSeal::Sealed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteSeal {
    Sealed,
    /// The walked family of the open plan, for the loader's log label.
    Open(&'static str),
}

#[derive(Debug)]
pub struct RouteKernelPlan {
    pub entries: Vec<PlannedKernel>,
}

/// One HIP module of a closed route: the module name the launcher passes to
/// `ensure_kernel` and exactly the symbols the route requests from it. The
/// source is the registry's launcher-identical expression for that module.
type RouteHip = (&'static str, &'static [&'static str]);

/// A3B (`qwen3_5_moe` mq4rs) lane-batched MoE block twins of the VMM exact
/// batch route (`moe_decode_lanes`; module name = entry name, gfx1201 only).
/// The route also launches the batched `fused_silu_mul_mq_rotate`, which the
/// A3B AR route already loads.
const A3B_GFX1201_MOE_LANES_HIP: &[RouteHip] = &[
    (
        "moe_router_softmax_topk_k8_wave64_rows",
        &["moe_router_softmax_topk_k8_wave64_rows"],
    ),
    (
        "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows",
        &["gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows"],
    ),
    (
        "gemv_mq4g256v2_moe_gate_up_k8_indexed_rows",
        &["gemv_mq4g256v2_moe_gate_up_k8_indexed_rows"],
    ),
    (
        "gemv_mq4g128v2_moe_ninepath_d4_rows",
        &["gemv_mq4g128v2_moe_ninepath_d4_rows"],
    ),
];

/// Qwen3.6-27B dense MQ4G256V2 XTS on exact gfx1201, AR route: native fp8
/// K/V, Q8 DeltaNet state, widened 8192-row prefill and its 512-row legacy
/// fallback, lowered single-token decode (direct and retained replay), and
/// the AR sampler for every request option (`sample_top_p_pf`: the 20-wide
/// parallel module including its penalty prepass, the >20 top-k 64-wide
/// module, and the tie-safe fast21/fast65 reducers). Prefill attention spans
/// every rows/context class of `AttnFp8E4m3KvBatchedMasked`: Q-resident FA2
/// (64..=512 rows or 512-multiples, with and without the A4 epilogue),
/// VerifyAttn (<= 32 rows past the 4096-context crossover), the flash tile
/// with its batched asym reduce (33..63 rows past it), and the scalar kernel
/// below it.
const QWEN36_27B_GFX1201_AR_HIP: &[RouteHip] = &[
    (
        "attention_flash_asym_reduce_batched",
        &["attention_flash_asym_reduce_batched"],
    ),
    (
        "attention_flash_fp8_e4m3_tile_batched",
        &["attention_flash_fp8_e4m3_tile_batched"],
    ),
    (
        "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
        &["attention_flash_fp8_e4m3_tile_gqa_gfx1201"],
    ),
    (
        "attention_flash_reduce_dsplit_gfx1201",
        &["attention_flash_reduce_dsplit_gfx1201"],
    ),
    // Native-fp8 split-KV FA2 verifier (`Gpu::attention_fp8_e4m3_fa2_gqa_split_gfx1201`),
    // on by default in `attention_flash_fp8_e4m3_tile_batched` for 2..=16-row
    // calls at >= 8192 live context: every AR prefill tail chunk of 2..=16
    // rows past 8192 positions and every MTP verify block. The module is
    // named for its partial symbol; the launcher binds the partial,
    // Q pre-convert and merge entries out of the one object.
    (
        "attention_fp8_e4m3_fa2_gqa_partial_lse_gfx1201",
        &[
            "attention_fp8_e4m3_fa2_gqa_partial_lse_gfx1201",
            "attention_fp8_e4m3_fa2_gqa_merge_lse_gfx1201",
            "attention_fp8_e4m3_fa2_q_preconvert_lse_gfx1201",
        ],
    ),
    (
        "attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201",
        &["attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201"],
    ),
    (
        "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201",
        &["attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201"],
    ),
    (
        "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
        &["attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201"],
    ),
    (
        "attention_fp8_e4m3_kv_batched",
        &["attention_fp8_e4m3_kv_batched"],
    ),
    (
        "attention_verify_gqa_gfx1201",
        &[
            "attention_verify_gqa_fp8_gfx1201",
            "attention_verify_reduce_gfx1201",
        ],
    ),
    (
        "conv1d_silu_split_qknorm_b256",
        &["conv1d_silu_split_qknorm_b256"],
    ),
    ("convert_f32_to_f16", &["convert_f32_to_f16"]),
    (
        "dflash_state_bulk_copy_gfx1100",
        &["dflash_state_bulk_copy_gfx1100"],
    ),
    ("embedding_q8", &["embedding_q8"]),
    ("embedding_q8_batched", &["embedding_q8_batched"]),
    ("fused_gate_up_hfq4g256_mq4v2", &["fused_gate_up_mq4g256v2"]),
    ("fused_qkv_hfq4g256_mq4v2", &["fused_qkv_mq4g256v2"]),
    ("fused_qkvza_hfq4g256_mq4v2", &["fused_qkvza_mq4g256v2"]),
    (
        "fused_rmsnorm_mq_rotate_awq",
        &["fused_rmsnorm_mq_rotate_awq"],
    ),
    (
        "fused_rmsnorm_mq_rotate_awq_g12dec",
        &["fused_rmsnorm_mq_rotate_awq_g12dec"],
    ),
    (
        "fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv",
        &["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv"],
    ),
    (
        "fused_sigmoid_alpha_gate",
        &["fused_sigmoid_alpha_gate_f32"],
    ),
    (
        "fused_silu_mul_mq_rotate_awq",
        &["fused_silu_mul_mq_rotate_awq"],
    ),
    (
        "fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast",
        &["fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast"],
    ),
    (
        "gated_delta_net_q8_compact3_b2",
        &["gated_delta_net_q8_compact3_b2"],
    ),
    ("gated_delta_net_q8_fast", &["gated_delta_net_q8_fast"]),
    ("gated_norm", &["gated_norm_f32"]),
    (
        "gated_norm_mq_rotate_awq_i4_gfx12_v2_slab",
        &["gated_norm_mq_rotate_awq_i4_gfx12_v2_slab"],
    ),
    (
        "gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab",
        &["gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab"],
    ),
    (
        "gated_norm_mq_rotate_awq_k6144_gfx1201",
        &["gated_norm_mq_rotate_awq_k6144_gfx1201"],
    ),
    ("gdn_chunk_kkt_solve", &["gdn_chunk_kkt_solve"]),
    (
        "gdn_chunk_kkt_solve_batched",
        &["gdn_chunk_kkt_solve_batched"],
    ),
    ("gdn_chunk_prep", &["gdn_chunk_prep"]),
    ("gdn_chunk_prep_fixup", &["gdn_chunk_prep_fixup"]),
    ("gdn_chunk_scan_bf16", &["gdn_chunk_scan_bf16"]),
    ("gdn_chunk_scan_bf16_mseg", &["gdn_chunk_scan_bf16_mseg"]),
    ("gdn_pre_batched_gfx1201", &["gdn_pre_batched_gfx1201"]),
    (
        "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2",
        &["gemm_gate_up_mq4g256v2_wmma_gfx12"],
    ),
    (
        "gemm_gate_up_mq4g256v2_wmma_gfx12_vt",
        &[
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w4",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w8",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w4",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w8",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4_k32",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8_k32",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w4_k32",
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w8_k32",
        ],
    ),
    (
        "gemm_hfq4g256_residual_wmma_gfx12_mq4v2",
        &["gemm_mq4g256v2_residual_wmma_gfx12"],
    ),
    (
        "gemm_mq4g256v2_residual_wmma_gfx12_vt",
        &[
            "gemm_mq4g256v2_residual_wmma_gfx12_vt2w4",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt2w8",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt3w4",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt3w8",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4_k32",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8_k32",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt8w4_k32",
            "gemm_mq4g256v2_residual_wmma_gfx12_vt8w8_k32",
        ],
    ),
    (
        "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2",
        &["gemm_qkv_mq4g256v2_wmma_gfx12"],
    ),
    (
        "gemm_qkv_mq4g256v2_wmma_gfx12_vt",
        &[
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w4",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w8",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w4",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w8",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4_k32",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8_k32",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w4_k32",
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w8_k32",
        ],
    ),
    (
        "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2",
        &["gemm_qkvza_mq4g256v2_wmma_gfx12"],
    ),
    (
        "gemm_qkvza_mq4g256v2_wmma_gfx12_vt",
        &[
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w4",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w8",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w4",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w8",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4_k32",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8_k32",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w4_k32",
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w8_k32",
        ],
    ),
    (
        "gemv_hfq4g256_multirow_default_mq4v2",
        &["gemv_mq4g256v2_multirow_r2"],
    ),
    ("gemv_hfq4g256_residual_mq4v2", &["gemv_mq4g256v2_residual"]),
    ("gemv_mq4g256v2_mq4v2", &["gemv_mq4g256v2"]),
    ("kv_cache_write_fp8_e4m3", &["kv_cache_write_fp8_e4m3"]),
    (
        "kv_cache_write_fp8_e4m3_batched",
        &["kv_cache_write_fp8_e4m3_batched"],
    ),
    ("mq_rotate_x", &["mq_rotate_x"]),
    (
        "qwen35_fa_prep_batched_gfx1201",
        &["qwen35_fa_prep_batched_gfx1201"],
    ),
    (
        "qwen35_fa_prep_fp8q_nogate_batched_gfx1201",
        &["qwen35_fa_prep_fp8q_nogate_batched_gfx1201"],
    ),
    (
        "qwen36_27b_fa_prep_gfx1201",
        &["qwen36_27b_fa_prep_gfx1201"],
    ),
    ("rmsnorm_f32_rowsplit", &["rmsnorm_f32_rowsplit"]),
    ("rotate_x_mq_awq", &["rotate_x_mq_awq"]),
    (
        "sample_top_p_parallel",
        &[
            "sample_apply_repeat_penalty",
            "sample_topk_partial",
            "sample_topk_finalize",
        ],
    ),
    (
        "sample_top_p_parallel_fast21",
        &["sample_topk_partial_fast21", "sample_topk_finalize_fast21"],
    ),
    (
        "sample_top_p_parallel_fast65",
        &["sample_topk_partial_fast65", "sample_topk_finalize_fast65"],
    ),
    (
        "sample_top_p_parallel_w64",
        &[
            "sample_apply_repeat_penalty_w64",
            "sample_topk_partial_w64",
            "sample_topk_finalize_w64",
        ],
    ),
    ("sigmoid_mul", &["sigmoid_mul_f32"]),
    (
        "sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab",
        &["sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab"],
    ),
    (
        "sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab",
        &["sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab"],
    ),
];

/// Native-MTP additions to [`QWEN36_27B_GFX1201_AR_HIP`] for the sidecar
/// MQ4G256 head (compressed-serial speculator): the head's prompt fill,
/// Q8-KV head attention, greedy/temperature verify and accept, and the
/// DeltaNet snapshot/replay rollback. The AR closure stays included because
/// the MTP route falls back to AR for requests that do not fit the draft
/// context (`hipfire-generate` `spec_ctx_request_fits`).
const QWEN36_27B_GFX1201_MTP_HIP: &[RouteHip] = &[
    ("add_inplace", &["add_inplace_f32"]),
    ("argmax_token_chain", &["argmax_token_chain_f32"]),
    (
        "attention_flash_q8_0_reduce",
        &["attention_flash_q8_0_reduce"],
    ),
    ("attention_flash_q8_0_tile", &["attention_flash_q8_0_tile"]),
    ("conv1d_silu_split", &["conv1d_silu_split_f32"]),
    ("deinterleave", &["deinterleave_f32"]),
    ("deinterleave_batched", &["deinterleave_f32_batched"]),
    ("dflash_gdn_replay_pre_ml", &["dflash_gdn_replay_pre_ml"]),
    // The MQ4-v1 head runs the typed layer program's norm/activation
    // producers and residual projections even when the trunk is all AWQ.
    ("fused_gate_up_hfq4g256", &["fused_gate_up_hfq4g256"]),
    ("fused_qkv_hfq4g256", &["fused_qkv_hfq4g256"]),
    ("fused_rmsnorm_mq_rotate", &["fused_rmsnorm_mq_rotate"]),
    ("fused_silu_mul_mq_rotate", &["fused_silu_mul_mq_rotate"]),
    (
        "fused_qk_l2_norm_scale_interleave_f32_batched",
        &["fused_qk_l2_norm_scale_interleave_f32_batched"],
    ),
    (
        "gated_delta_net_q8_fast_ml",
        &["gated_delta_net_q8_fast_ml"],
    ),
    ("gemm_hfq4g256", &["gemm_hfq4g256"]),
    ("gemv_hfq4g256", &["gemv_hfq4g256"]),
    (
        "gemv_hfq4g256_multirow_default",
        &["gemv_hfq4g256_multirow_r2"],
    ),
    ("gemv_hfq4g256_residual", &["gemv_hfq4g256_residual"]),
    ("greedy_accept", &["greedy_accept_from_argmax_i32"]),
    ("kv_cache_write_q8_0", &["kv_cache_write_q8_0"]),
    (
        "kv_cache_write_q8_0_batched",
        &["kv_cache_write_q8_0_batched"],
    ),
    ("rmsnorm", &["rmsnorm_f32"]),
    (
        "rope_partial_halfsplit_batched",
        &["rope_partial_halfsplit_batched_f32"],
    ),
    (
        "rope_partial_halfsplit_f32_headgrid",
        &["rope_partial_halfsplit_f32_headgrid"],
    ),
    ("select_regrid", &["argmax_f32_batched_regrid"]),
    ("silu_mul", &["silu_mul_f32"]),
    (
        "softmax_temp_topp_batched",
        &["softmax_temp_topp_batched_f32"],
    ),
];

/// CbVmm's rows-batched twins of the singleton decode kernels
/// (`attention::rows_batched`; module name = entry name).
const QWEN36_27B_GFX1201_ROWS_TWINS_HIP: &[RouteHip] = &[
    (
        "gated_delta_net_q8_compact3_b2_rows",
        &["gated_delta_net_q8_compact3_b2_rows"],
    ),
    (
        "conv1d_silu_split_qknorm_b256_rows",
        &["conv1d_silu_split_qknorm_b256_rows"],
    ),
    (
        "gated_norm_mq_rotate_awq_k6144_gfx1201_rows",
        &["gated_norm_mq_rotate_awq_k6144_gfx1201_rows"],
    ),
    (
        "qwen36_27b_fa_prep_gfx1201_rows",
        &["qwen36_27b_fa_prep_gfx1201_rows"],
    ),
];

/// Exact-A3B CbVmm DeltaNet rows twins (`rows_batched.rs`; module name =
/// entry name). The conv twin is the shared 27B module.
const A3B_GFX1201_DN_ROWS_TWINS_HIP: &[RouteHip] = &[
    (
        "gated_delta_net_q8_compact2_b2_rows",
        &["gated_delta_net_q8_compact2_b2_rows"],
    ),
    (
        "gated_norm_mq_rotate_gfx1201_rows",
        &["gated_norm_mq_rotate_gfx1201_rows"],
    ),
];

/// VMM continuous-batching exact route additions to
/// [`QWEN36_27B_GFX1201_AR_HIP`] (`forward_slots/vmm/exact.rs`): batched
/// VMM KV write, row-batched flash decode, and the rows twins.
const QWEN36_27B_GFX1201_VMM_EXACT_HIP: &[RouteHip] = &[
    (
        "kv_cache_write_fp8_e4m3_batched_vmm",
        &["kv_cache_write_fp8_e4m3_batched_vmm"],
    ),
    (
        "attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm",
        &["attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm"],
    ),
    (
        "attention_flash_reduce_dsplit_gfx1201_vmm",
        &["attention_flash_reduce_dsplit_gfx1201_vmm"],
    ),
    (
        "gated_delta_net_q8_compact3_b2_rows",
        &["gated_delta_net_q8_compact3_b2_rows"],
    ),
    (
        "conv1d_silu_split_qknorm_b256_rows",
        &["conv1d_silu_split_qknorm_b256_rows"],
    ),
    (
        "gated_norm_mq_rotate_awq_k6144_gfx1201_rows",
        &["gated_norm_mq_rotate_awq_k6144_gfx1201_rows"],
    ),
    (
        "qwen36_27b_fa_prep_gfx1201_rows",
        &["qwen36_27b_fa_prep_gfx1201_rows"],
    ),
];

/// The hipcc x-batch GEMV the PM multi-column launchers fall back to when
/// `kernel.pm_decode` is off.
const QWEN36_27B_GFX1201_XBATCH_HIP: &[RouteHip] =
    &[("gemv_mq4g256v2_xbatch", &["gemv_mq4g256v2_xbatch"])];

/// Cross-request batched MTP verify additions to the native-MTP route
/// (`verify_twins`; module name = entry name).
const QWEN36_27B_GFX1201_SPEC_CB_HIP: &[RouteHip] = &[(
    "attention_fp8_e4m3_kv_batched_segs",
    &["attention_fp8_e4m3_kv_batched_segs"],
)];

/// Every kernel a gfx1201 twin module's generated source defines, for the
/// packaged index only. The route tables above stay the measured required
/// symbols (route binding is unchanged); the shared TUs also define these.
fn gfx1201_twin_exports(
    module: &str,
    required: &'static [&'static str],
) -> &'static [&'static str] {
    match module {
        "gated_delta_net_q8_compact3_b2_rows" => &[
            "gated_delta_net_q8_compact3_b2_rows",
            "gated_delta_net_q8_fast_independent_masked",
        ],
        "gated_delta_net_q8_compact2_b2_rows" => &[
            "gated_delta_net_q8_compact2_b2_rows",
            "gated_delta_net_q8_fast_independent_masked",
        ],
        "attention_fp8_e4m3_kv_batched_segs" => &[
            "attention_fp8_e4m3_kv_batched_segs",
            "attention_q8_0_kv_batched",
            "attention_q8_0_kv_independent_masked_windowed",
        ],
        _ => required,
    }
}

/// The gfx1201 slab IU4 MMQ bundle every 27B prefill projection selects
/// (`gemm.rs` `g12_iu4_b1s_image`, default A4 slab route).
const QWEN36_27B_GFX1201_B1S_SYMBOLS: &[&str] = &[
    "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s",
    "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s_v64",
];
/// [`QWEN36_27B_GFX1201_B1S_SYMBOLS`] minus the gate_up export a PM tune
/// variant replaces (`pm_tune`); the small-N `_v64` gate_up export stays.
const QWEN36_27B_GFX1201_B1S_SYMBOLS_NO_GATE_UP: &[&str] = &[
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s",
    "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s_v64",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s_v64",
];
const QWEN36_27B_GFX1201_B1S_MODULE: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s";
const QWEN36_27B_GATE_UP_B1S_SYMBOLS: &[&str] = &[crate::pm_tune::GATE_UP_B1S_SYMBOL];

/// The PM tune variant for the 27B gate_up b1s symbol, if a table is
/// configured and selects one. The planner knows the gate_up shape: the
/// route fingerprint pins `layers.0.mlp.gate_proj.weight` at `[17408, 5120]`
/// and every dense layer (and the MTP head's `ffn_gate`) shares it, so the
/// table key is `(arch, symbol, gate rows, hidden)`. A dispatch at any other
/// shape then fails loudly (one image per symbol per process).
fn gate_up_pm_tune_variant(
    input: &RouteKernelInput<'_>,
) -> Result<Option<&'static crate::pm_tune::PmVariant>, RegistryError> {
    if !crate::pm_tune::enabled() {
        return Ok(None);
    }
    let Some(gate) = input
        .tensors
        .iter()
        .find(|t| t.name == "model.language_model.layers.0.mlp.gate_proj.weight")
    else {
        return Ok(None);
    };
    let [m, k] = *gate.shape else {
        return Ok(None);
    };
    crate::pm_tune::lookup(
        input.arch,
        crate::pm_tune::GATE_UP_B1S_SYMBOL,
        m,
        k,
        crate::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
    )
    .map_err(|e| RegistryError::UnsupportedRoute(format!("qwen3.6-27b: {e}")))
}

/// Move the gate_up export of the shipped b1s bundle to `variant`'s module:
/// the plan keeps one image per symbol.
fn apply_gate_up_pm_tune_variant(
    embedded: &mut Vec<PlannedKernel>,
    variant: &'static crate::pm_tune::PmVariant,
) {
    for entry in embedded.iter_mut() {
        if let PlannedKernel::Embedded {
            module, symbols, ..
        } = entry
        {
            if *module == QWEN36_27B_GFX1201_B1S_MODULE {
                *symbols = QWEN36_27B_GFX1201_B1S_SYMBOLS_NO_GATE_UP;
            }
        }
    }
    embedded.push(PlannedKernel::Embedded {
        module: variant.module.as_str(),
        image: variant.bytes.as_slice(),
        radiowave_json: None,
        symbols: QWEN36_27B_GATE_UP_B1S_SYMBOLS,
    });
}

/// No-AWQ plain rotate producers of the 27B route, planned only for the
/// families whose projection weights lack a sidecar (`Qwen36PlainRotate`).
/// Decode and batched fallback launch `fused_rmsnorm_mq_rotate`,
/// `fused_silu_mul_mq_rotate` and the 48-head `gated_norm_mq_rotate_k6144`
/// (`gemv.rs`); gfx1201 prefill launches the non-slab `_i4_gfx12` producers
/// (`awq == None` never selects a slab twin; the gated norm also has its
/// `_xbf16` GDN-plane twin). `mq_rotate_x` is already in the AR closure.
const QWEN36_27B_GFX1201_PLAIN_RMSNORM_HIP: &[RouteHip] = &[
    ("fused_rmsnorm_mq_rotate", &["fused_rmsnorm_mq_rotate"]),
    (
        "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
        &["fused_rmsnorm_mq_rotate_i4_gfx12_v2"],
    ),
];
const QWEN36_27B_GFX1201_PLAIN_SILU_HIP: &[RouteHip] = &[
    ("fused_silu_mul_mq_rotate", &["fused_silu_mul_mq_rotate"]),
    (
        "fused_silu_mul_mq_rotate_i4_gfx12",
        &["fused_silu_mul_mq_rotate_i4_gfx12"],
    ),
];
const QWEN36_27B_GFX1201_PLAIN_GATED_NORM_HIP: &[RouteHip] = &[
    (
        "gated_norm_mq_rotate_k6144_gfx1201",
        &["gated_norm_mq_rotate_k6144_gfx1201"],
    ),
    (
        "gated_norm_mq_rotate_i4_gfx12_v2",
        &["gated_norm_mq_rotate_i4_gfx12_v2"],
    ),
    (
        "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
        &["gated_norm_mq_rotate_i4_gfx12_v2_xbf16"],
    ),
];
const QWEN36_27B_GFX1201_PLAIN_ROTATE_HIP: &[RouteHip] =
    &[("mq_rotate_x_i4_gfx12", &["mq_rotate_x_i4_gfx12"])];

/// The non-slab `_b1` GEMM symbols that read the plain producers' records
/// (`gemm.rs` `g12_iu4_b1_image`): residual add/set and the fused GDN input
/// projection, plus the F1-lite gate/up + SwiGLU entry (planned whenever an
/// RMSNorm producer is plain; see `route_entries`).
const QWEN36_27B_GFX1201_B1_SYMBOLS: &[&str] = &[
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1",
];
const QWEN36_27B_GFX1201_B1_SILU_SYMBOLS: &[&str] = &[
    "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1",
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1",
];

/// Header-selected kernel closure for one resolved load. Refuses (never
/// returns a partial plan) when the model, arch, policy axes, spec route or
/// feature flags are outside a closed route.
pub fn route_entries(input: &RouteKernelInput<'_>) -> Result<RouteKernelPlan, RegistryError> {
    route_entries_sealed(input).map(|(plan, _)| plan)
}

/// [`route_entries`] plus the plan's [`RouteSeal`]: the loader seals only
/// [`RouteSeal::Sealed`] plans.
pub fn route_entries_sealed(
    input: &RouteKernelInput<'_>,
) -> Result<(RouteKernelPlan, RouteSeal), RegistryError> {
    let refuse = |why: String| Err(RegistryError::UnsupportedRoute(why));
    match input.spec {
        KernelSpecRoute::Ar | KernelSpecRoute::NativeMtp => {}
        other => {
            return refuse(format!(
                "{other:?} needs a separately admitted drafter header; no closed route"
            ));
        }
    }
    if input.host_mapped_experts {
        return refuse("host-mapped experts have no closed route".into());
    }
    // `pm_decode` is a closed route axis (accepted twins replace their HIP
    // entries below); every other flag must be the builtin default.
    let mut default_flags = default_feature_flags(input.arch)?;
    default_flags.pm_decode = input.flags.pm_decode;
    if format!("{:?}", input.flags) != format!("{default_flags:?}") {
        return refuse(format!("non-default feature flags on {}", input.arch));
    }
    let (hip, embedded): (Vec<&[RouteHip]>, _) = match qwen36_27b_route(input)? {
        Some(mtp) => {
            let mut hip = vec![QWEN36_27B_GFX1201_AR_HIP, QWEN36_27B_GFX1201_VMM_EXACT_HIP];
            if mtp {
                hip.push(QWEN36_27B_GFX1201_MTP_HIP);
                hip.push(QWEN36_27B_GFX1201_SPEC_CB_HIP);
            }
            let b1s = PlannedKernel::Embedded {
                module: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s",
                image: crate::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
                radiowave_json: None,
                symbols: QWEN36_27B_GFX1201_B1S_SYMBOLS,
            };
            let mut embedded = vec![b1s];
            // PM tune table (`HIPFIRE_PM_TUNE_TABLE`): preload the selected
            // gate_up variant instead of the shipped image, so the dispatch
            // selector finds the symbol already bound to that same image.
            // Unset (the default): nothing here runs and the plan is unchanged.
            if let Some(variant) = gate_up_pm_tune_variant(input)? {
                apply_gate_up_pm_tune_variant(&mut embedded, variant);
            }
            if mtp {
                let (module, image, symbols) = crate::pm_mtp_gemm::route_object();
                embedded.push(PlannedKernel::Embedded {
                    module,
                    image,
                    radiowave_json: None,
                    symbols,
                });
            }
            // VMM exact route projections/lm_head: the accepted PM
            // multi-column objects under pm_decode, else the hipcc x-batch
            // GEMV their launchers fall back to (residual/multirow fall back
            // to singleton modules already in the AR route).
            if input.flags.pm_decode {
                for (module, image, symbols) in crate::pm_xbatch::route_objects() {
                    embedded.push(PlannedKernel::Embedded {
                        module,
                        image,
                        radiowave_json: None,
                        symbols,
                    });
                }
            } else {
                hip.push(QWEN36_27B_GFX1201_XBATCH_HIP);
            }
            // No-AWQ projections: the AWQ closure above is kept whole and the
            // plain producer families their sidecar-less weights launch are
            // appended. Slab producer twins need an AWQ scale, so a plain
            // producer writes token-order A4 records and its GEMM consumers are
            // the `_b1` symbols of the non-slab bundle.
            let plain = qwen36_27b_plain_rotate(input);
            if plain.rmsnorm {
                hip.push(QWEN36_27B_GFX1201_PLAIN_RMSNORM_HIP);
            }
            if plain.silu {
                hip.push(QWEN36_27B_GFX1201_PLAIN_SILU_HIP);
            }
            if plain.gated_norm {
                hip.push(QWEN36_27B_GFX1201_PLAIN_GATED_NORM_HIP);
            }
            if plain.rotate {
                hip.push(QWEN36_27B_GFX1201_PLAIN_ROTATE_HIP);
            }
            if plain.any() {
                embedded.push(PlannedKernel::Embedded {
                    module: "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1",
                    image: crate::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1,
                    radiowave_json: None,
                    // F1-lite gate/up + SwiGLU (a layer with a plain RMSNorm
                    // producer and an AWQ down scale) reads the non-slab
                    // `gate_up_silu` symbol; planned whenever any RMSNorm
                    // producer is plain, since the families are aggregated
                    // across layers.
                    symbols: if plain.rmsnorm {
                        QWEN36_27B_GFX1201_B1_SILU_SYMBOLS
                    } else {
                        QWEN36_27B_GFX1201_B1_SYMBOLS
                    },
                });
            }
            (hip, embedded)
        }
        None => {
            if let Some((plan, seal)) = walked_route_entries(input)? {
                validate_route_plan(&plan)?;
                return Ok((plan, seal));
            }
            return refuse(format!(
                "model arch {} on {} has no closed route",
                input.model_arch, input.arch
            ));
        }
    };
    let corpus = route_corpus(input.arch, input.hipcc_extra_flags)?;
    let mut entries = embedded;
    for &(module, symbols) in hip.into_iter().flatten() {
        // The MQ4 head and a no-AWQ trunk share plain producer entries.
        if entries
            .iter()
            .any(|entry| entry.module() == module && entry.symbols() == symbols)
        {
            continue;
        }
        if let Some(twin) =
            crate::pm_decode_twins::pm_decode_module(input.arch, input.flags.pm_decode, module)
        {
            if let Some(symbol) = symbols.iter().find(|s| !twin.symbols.contains(s)) {
                return Err(RegistryError::ConflictingPlan(format!(
                    "module {module} symbol {symbol} has no accepted pm_decode export"
                )));
            }
            if !entries.iter().any(|e| e.module() == module) {
                entries.push(PlannedKernel::Embedded {
                    module: twin.module,
                    image: twin.image,
                    radiowave_json: Some(twin.radiowave_json),
                    symbols,
                });
            }
            continue;
        }
        let mut matches = corpus.iter().filter(|entry| entry.module == module);
        let found = matches
            .next()
            .ok_or_else(|| RegistryError::UnsupportedModule {
                arch: input.arch.to_owned(),
                module: module.to_owned(),
            })?;
        if matches.any(|other| other.source != found.source || other.flags != found.flags) {
            return Err(RegistryError::ConflictingPlan(format!(
                "module {module} has two registry recipes"
            )));
        }
        entries.push(PlannedKernel::Hip(KernelEntry {
            arch: found.arch,
            module,
            symbols,
            source: found.source.clone(),
            flags: found.flags.clone(),
            scheduler_profile: found.scheduler_profile.clone(),
        }));
    }
    let plan = RouteKernelPlan { entries };
    validate_route_plan(&plan)?;
    Ok((plan, RouteSeal::Sealed))
}

/// One selected image per public entry and one recipe per module name. Any
/// other plan is refused before loading: the function cache is keyed by
/// entry name, so mutually exclusive images exporting one name cannot both
/// be preloaded.
pub fn validate_route_plan(plan: &RouteKernelPlan) -> Result<(), RegistryError> {
    let mut owner: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (index, entry) in plan.entries.iter().enumerate() {
        let module = entry.module();
        if let Some(previous) = plan.entries[..index].iter().find(|e| e.module() == module) {
            let same_recipe = match (previous, entry) {
                (PlannedKernel::Hip(a), PlannedKernel::Hip(b)) => {
                    a.source == b.source
                        && a.flags == b.flags
                        && a.scheduler_profile == b.scheduler_profile
                }
                _ => false,
            };
            if !same_recipe {
                return Err(RegistryError::ConflictingPlan(format!(
                    "module {module} selected with two different images"
                )));
            }
        }
        for &symbol in entry.symbols() {
            if let Some(&first) = owner.get(symbol) {
                if plan.entries[first].module() != module {
                    return Err(RegistryError::ConflictingPlan(format!(
                        "entry {symbol} exported by {} and {module}",
                        plan.entries[first].module()
                    )));
                }
            } else {
                owner.insert(symbol, index);
            }
        }
    }
    Ok(())
}

// ── Predicted family kernels ────────────────────────────────────────────
//
// A route without a closed plan (a refused option, a layout no walk covers,
// an open walked plan whose walks missed a rung) still has a predictable
// kernel set: the arch's whole installer inventory. [`predicted_kernels`]
// resolves it to the exact registry recipes (never another arch's source) so
// a loader can compile it before ready instead of at the first request.
// Compiling is always safe; binding a symbol into the loader's function
// cache is not: that cache is keyed by symbol name alone, so a symbol two
// registry modules export (or one an embedded image exports) could return
// the wrong image to a launcher that asked for the other. Only symbols a
// family trace proved, with exactly one registry owner, are bound.

/// One kernel-load trace fixture and the model archs it describes.
struct FamilyTrace {
    family: &'static str,
    model_archs: &'static [u32],
    /// `arch\tmodule\tsymbols` rows (`tests/fixtures/kernel-trace-*.tsv`).
    fixture: &'static str,
}

const FAMILY_TRACES: &[FamilyTrace] = &[
    FamilyTrace {
        family: "qwen3.8-flash-next",
        model_archs: &[16],
        fixture: include_str!("../tests/fixtures/kernel-trace-qwen4-flash-next.tsv"),
    },
    FamilyTrace {
        family: "qwen3.5",
        model_archs: &[5, 6],
        fixture: include_str!("../tests/fixtures/kernel-trace-qwen35.tsv"),
    },
];

/// How a predicted module set was chosen. Both compile the arch's whole
/// installer inventory; they differ in what binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionBasis {
    /// The loader recognised an exact plan of a family whose trace has rows
    /// for this arch: the symbols the trace and matching walked rows proved
    /// bind eagerly (conflict-free subset), everything else compile only.
    ArchInventoryFamilyBindings,
    /// No proof for this load (unrecognised or refused route, unknown
    /// family, or no trace rows for this arch): nothing is bound.
    ArchInventoryCompileOnly,
}

impl PredictionBasis {
    pub fn label(self) -> &'static str {
        match self {
            Self::ArchInventoryFamilyBindings => "arch-inventory (family-proven bindings)",
            Self::ArchInventoryCompileOnly => "arch-inventory compile-only",
        }
    }
}

/// What [`predicted_kernels`] needs from a load.
pub struct PredictKernelInput<'a> {
    pub arch: &'a str,
    /// Header `arch_id` when the source is an HFQ file.
    pub model_arch: Option<u32>,
    pub hipcc_extra_flags: &'a str,
    /// `FeatureFlags::pm_decode`: accepted twins replace their HIP modules,
    /// so those modules are neither compiled nor bound.
    pub pm_decode: bool,
    /// True only when the loader recognised this exact header layout, arch
    /// and policy axes as a closed or walked route plan (`planned` is that
    /// plan). Only then may the family trace choose which symbols bind
    /// eagerly; the compile set is the arch's whole inventory either way.
    pub exact_plan: bool,
}

/// One predicted HIP module: its exact registry recipe (compiled with every
/// symbol the registry lists) and the proven conflict-free subset to bind.
#[derive(Debug, Clone)]
pub struct PredictedModule {
    pub entry: KernelEntry,
    pub bind: Vec<&'static str>,
}

#[derive(Debug)]
pub struct PredictedKernels {
    pub family: &'static str,
    pub basis: PredictionBasis,
    pub modules: Vec<PredictedModule>,
    /// Inventory modules the set leaves out, with the reason
    /// (already in the route plan, accepted pm_decode twin, ambiguous or
    /// stale recipe).
    pub omitted: Vec<(&'static str, &'static str)>,
}

impl PredictedKernels {
    /// Symbols the route preload binds.
    pub fn bound_symbols(&self) -> usize {
        self.modules.iter().map(|m| m.bind.len()).sum()
    }

    /// Modules with at least one bound symbol; the rest are compile-only.
    pub fn bound_modules(&self) -> usize {
        self.modules.iter().filter(|m| !m.bind.is_empty()).count()
    }
}

/// Modules the fixture's rows record for `arch`, with the symbols each run
/// resolved from them.
fn traced_symbols(
    fixture: &'static str,
    arch: &str,
) -> std::collections::BTreeMap<&'static str, std::collections::BTreeSet<&'static str>> {
    let mut modules: std::collections::BTreeMap<
        &'static str,
        std::collections::BTreeSet<&'static str>,
    > = std::collections::BTreeMap::new();
    for line in fixture
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
    {
        let mut columns = line.split('\t');
        let (Some(row_arch), Some(module), Some(symbols)) =
            (columns.next(), columns.next(), columns.next())
        else {
            continue;
        };
        if row_arch == arch {
            modules
                .entry(module)
                .or_default()
                .extend(symbols.split(','));
        }
    }
    modules
}

/// The kernel set a load of this model on this arch will predictably need,
/// resolved to exact registry recipes and minus everything `planned` already
/// covers. Pure CPU: no weight payload, compiler or HIP call.
///
/// Compile set: always the arch's whole installer inventory. An open plan is
/// open exactly because its walks were incomplete, and a family trace covers
/// the layouts it was walked on, not the family, so no subset of the
/// inventory is trusted to be enough (another codec, DFlash, vision, an AR
/// plan's MTP fallback, a kernel registered after the walk).
///
/// Bind set: the trace is trusted only to choose which symbols bind eagerly.
/// Only for an exact plan (`exact_plan`) of a traced family with rows for
/// `arch`: the symbols those rows and the matching walked HIP rows observed
/// (a walked row whose source hash no longer matches proves nothing), and
/// only when every registry corpus entry that exports a symbol belongs to
/// its module and no embedded image or `planned` entry of another module
/// exports it. Everything else compiles only. Modules whose recipe differs
/// between corpus entries are omitted outright.
pub fn predicted_kernels(
    input: &PredictKernelInput<'_>,
    planned: &RouteKernelPlan,
) -> Result<PredictedKernels, RegistryError> {
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeSet, HashMap, HashSet};
    let inventory = entries(input.arch, input.hipcc_extra_flags)?;
    let by_name: HashMap<&'static str, &KernelEntry> =
        inventory.iter().rev().map(|e| (e.module, e)).collect();
    let mut omitted: Vec<(&'static str, &'static str)> = Vec::new();
    let trace = input
        .model_arch
        .and_then(|id| FAMILY_TRACES.iter().find(|t| t.model_archs.contains(&id)));
    // Compile set: always the arch's whole installer inventory. Binding set:
    // only symbols a family trace row or a matching walked row proved.
    let names: BTreeSet<&'static str> = inventory.iter().map(|e| e.module).collect();
    let mut proven: HashMap<&'static str, BTreeSet<&'static str>> = HashMap::new();
    let family = trace.map_or("unknown-family", |t| t.family);
    if let (Some(trace), true) = (trace, input.exact_plan) {
        for (module, symbols) in traced_symbols(trace.fixture, input.arch) {
            proven.entry(module).or_default().extend(symbols);
        }
        // A walked row proves its symbols only while its recorded source
        // still matches the registry; a stale row proves nothing.
        if !proven.is_empty() {
            if let Some(model_arch) = input.model_arch {
                for plan in walked_route_plans()?
                    .iter()
                    .filter(|p| p.model_arch == model_arch && p.arch == input.arch)
                {
                    for row in plan.rows.iter().filter(|row| !row.embedded) {
                        let Some(found) = by_name.get(row.module) else {
                            continue;
                        };
                        if format!("{:x}", Sha256::digest(found.source.as_bytes())) == row.sha256 {
                            proven
                                .entry(found.module)
                                .or_default()
                                .extend(row.symbols.iter().copied());
                        }
                    }
                }
            }
        }
    }
    let basis = if proven.is_empty() {
        PredictionBasis::ArchInventoryCompileOnly
    } else {
        PredictionBasis::ArchInventoryFamilyBindings
    };

    // Symbol ownership across every launcher source the registry knows, and
    // the recipe uniqueness of each module name.
    let mut corpus = route_corpus(input.arch, input.hipcc_extra_flags)?;
    corpus.extend(railgun_entries(input.arch, input.hipcc_extra_flags)?);
    let mut owners: HashMap<&'static str, HashSet<&'static str>> = HashMap::new();
    let mut first: HashMap<&'static str, &KernelEntry> = HashMap::new();
    let mut ambiguous: HashSet<&'static str> = HashSet::new();
    for entry in &corpus {
        for &symbol in entry.symbols {
            owners.entry(symbol).or_default().insert(entry.module);
        }
        match first.get(entry.module) {
            Some(prev)
                if prev.source != entry.source
                    || prev.flags != entry.flags
                    || prev.scheduler_profile != entry.scheduler_profile =>
            {
                ambiguous.insert(entry.module);
            }
            Some(_) => {}
            None => {
                first.insert(entry.module, entry);
            }
        }
    }
    // Names embedded images bind into the same function cache.
    let mut foreign: HashSet<&'static str> = HashSet::new();
    if !proven.is_empty() {
        foreign.extend(
            crate::pm_decode_twins::GFX1201_TWINS
                .iter()
                .flat_map(|twin| twin.symbols.iter().copied()),
        );
        foreign.extend(crate::pm_mtp_gemm::route_object().2.iter().copied());
        for (_, _, symbols) in crate::pm_xbatch::route_objects() {
            foreign.extend(symbols.iter().copied());
        }
        if let Ok(slab) = slab_embedded_entries(input.arch, 3) {
            foreign.extend(slab.iter().flat_map(|e| e.symbols().iter().copied()));
        }
        for plan in walked_route_plans()?
            .iter()
            .filter(|p| p.arch == input.arch)
        {
            for row in plan.rows.iter().filter(|row| row.embedded) {
                foreign.extend(row.symbols.iter().copied());
            }
        }
    }
    let planned_modules: HashSet<&str> =
        planned.entries.iter().map(PlannedKernel::module).collect();

    let mut modules = Vec::with_capacity(names.len());
    for module in names {
        if planned_modules.contains(module) {
            omitted.push((module, "already in the route plan"));
            continue;
        }
        if crate::pm_decode_twins::pm_decode_module(input.arch, input.pm_decode, module).is_some() {
            omitted.push((module, "accepted pm_decode twin"));
            continue;
        }
        if ambiguous.contains(module) {
            omitted.push((module, "two registry recipes"));
            continue;
        }
        let Some(entry) = by_name.get(module) else {
            omitted.push((module, "missing from the registry"));
            continue;
        };
        let mut bind: Vec<&'static str> = Vec::new();
        for &symbol in proven.get(module).into_iter().flatten() {
            let exported = inventory
                .iter()
                .any(|e| e.module == module && e.symbols.contains(&symbol));
            let sole_owner = owners.get(symbol).is_some_and(|set| set.len() == 1);
            let planned_elsewhere = planned
                .entries
                .iter()
                .any(|p| p.module() != module && p.symbols().contains(&symbol));
            if exported && sole_owner && !planned_elsewhere && !foreign.contains(symbol) {
                bind.push(symbol);
            }
        }
        modules.push(PredictedModule {
            entry: (*entry).clone(),
            bind,
        });
    }
    Ok(PredictedKernels {
        family,
        basis,
        modules,
        omitted,
    })
}

/// Builtin-default feature flags for `arch`, independent of the process
/// environment, as the reference a closed route was walked against.
fn default_feature_flags(arch: &str) -> Result<crate::feature_flags::FeatureFlags, RegistryError> {
    let resolved = hipfire_config::resolve(std::iter::empty())
        .map_err(|e| RegistryError::UnsupportedRoute(format!("builtin config: {e}")))?;
    let config = hipfire_config::ProcessConfig::from_resolved(&resolved)
        .map_err(|e| RegistryError::UnsupportedRoute(format!("builtin config: {e}")))?;
    Ok(crate::feature_flags::FeatureFlags::from_process_config(
        arch, &config,
    ))
}

/// Registry sources a route may name: the installer inventory, the admitted
/// default-route corpus, and route-only launcher sources neither carries.
fn route_corpus(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let mut all = entries(arch, extra_flags)?;
    all.extend(default_route_entries(arch, extra_flags)?);
    let arch = all.first().map(|e| e.arch).unwrap_or("");
    if arch == "gfx1201" {
        // `softmax_temp_topp_batched` (norm.rs `softmax_temp_topp_batched_into_f32`,
        // MTP temperature verify), the VMM exact route's KV write / row-batched
        // flash decode, the rows twins and the segment twin are all in the
        // installer inventory (`entries`).
        all.push(entry(
            arch,
            "gemv_mq4g256v2_xbatch",
            &["gemv_mq4g256v2_xbatch"],
            kernels::GEMV_MQ4G256V2_XBATCH_SRC.into(),
            extra_flags,
        ));
        // No-AWQ plain rotate producers (`QWEN36_27B_GFX1201_PLAIN_*`): the
        // launchers' own sources for the modules neither inventory carries.
        all.push(entry(
            arch,
            "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
            &["fused_rmsnorm_mq_rotate_i4_gfx12_v2"],
            kernels::FUSED_RMSNORM_MQ_ROTATE_I4_GFX12_V2_SRC.into(),
            extra_flags,
        ));
        all.push(entry(
            arch,
            "fused_silu_mul_mq_rotate_i4_gfx12",
            &["fused_silu_mul_mq_rotate_i4_gfx12"],
            kernels::FUSED_SILU_MUL_MQ_ROTATE_I4_GFX12_SRC.into(),
            extra_flags,
        ));
        all.push(entry(
            arch,
            "gated_norm_mq_rotate_k6144_gfx1201",
            &["gated_norm_mq_rotate_k6144_gfx1201"],
            kernels::gated_norm_mq_rotate_k6144_gfx1201_src().into(),
            extra_flags,
        ));
        all.push(entry(
            arch,
            "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
            &["gated_norm_mq_rotate_i4_gfx12_v2_xbf16"],
            kernels::GATED_NORM_MQ_ROTATE_I4_GFX12_V2_XBF16_SRC.into(),
            extra_flags,
        ));
        all.push(entry(
            arch,
            "mq_rotate_x_i4_gfx12",
            &["mq_rotate_x_i4_gfx12"],
            kernels::MQ_ROTATE_X_I4_GFX12_SRC.into(),
            extra_flags,
        ));
    }
    Ok(all)
}

/// Qwen3.6-27B dense XTS fingerprint on exact gfx1201. `Ok(Some(mtp))` when
/// the closed route applies (`mtp` = native-MTP route), `Ok(None)` when the
/// header is not this model, `Err` when it is but a policy axis is outside
/// the closure.
fn qwen36_27b_route(input: &RouteKernelInput<'_>) -> Result<Option<bool>, RegistryError> {
    use crate::DType;
    let refuse = |why: &str| {
        Err(RegistryError::UnsupportedRoute(format!(
            "qwen3.6-27b: {why}"
        )))
    };
    if input.arch != "gfx1201" || input.model_arch != 5 {
        return Ok(None);
    }
    if (input.n_heads, input.n_kv_heads, input.head_dim) != (24, 4, 256) {
        return Ok(None);
    }
    let trunk: Vec<_> = input
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("model.language_model.") || t.name == "lm_head.weight")
        .collect();
    let layer_of = |name: &str| -> Option<usize> {
        name.strip_prefix("model.language_model.layers.")?
            .split('.')
            .next()?
            .parse()
            .ok()
    };
    let layers = trunk
        .iter()
        .filter_map(|t| layer_of(t.name))
        .max()
        .map_or(0, |l| l + 1);
    let shape_of = |name: &str| {
        trunk
            .iter()
            .find(|t| t.name == name)
            .map(|t| (t.dtype, t.shape))
    };
    let projections_mq4v2 = trunk.iter().filter(|t| {
        t.name.ends_with("_proj.weight")
            || t.name.ends_with("in_proj_qkv.weight")
            || t.name.ends_with("in_proj_z.weight")
            || t.name.ends_with("in_proj_a.weight")
            || t.name.ends_with("in_proj_b.weight")
    });
    let mut n_projections = 0;
    for tensor in projections_mq4v2 {
        if tensor.dtype != DType::MQ4G256V2 {
            return Ok(None);
        }
        n_projections += 1;
    }
    let fingerprint = layers == 64
        && n_projections == 48 * 5 + 16 * 4 + 64 * 3
        && shape_of("lm_head.weight") == Some((DType::MQ4G256V2, &[248_320, 5_120][..]))
        && shape_of("model.language_model.embed_tokens.weight")
            == Some((DType::Q8_0, &[248_320, 5_120][..]))
        && shape_of("model.language_model.layers.0.mlp.gate_proj.weight")
            == Some((DType::MQ4G256V2, &[17_408, 5_120][..]))
        && shape_of("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
            == Some((DType::MQ4G256V2, &[10_240, 5_120][..]))
        && shape_of("model.language_model.layers.3.self_attn.q_proj.weight")
            == Some((DType::MQ4G256V2, &[12_288, 5_120][..]));
    if !fingerprint {
        return Ok(None);
    }
    if (input.kv_k, input.kv_v) != ("fp8", "fp8") {
        return refuse("KV must resolve to native fp8 K/V");
    }
    if input.state_quant != "q8" {
        return refuse("DeltaNet state must be Q8");
    }
    if !(2..=262_144).contains(&input.max_seq) {
        return refuse("max_seq outside the model's 262144 positions");
    }
    if !(512..=8_192).contains(&input.prefill_chunk_rows) {
        return refuse("prefill chunk ceiling outside the walked 512..=8192 rungs");
    }
    // The sidecar head (`<trunk>.mtp`): one MQ4G256 layer at the trunk width.
    let head = |name: &str, dtype: DType, shape: &[usize]| {
        input
            .tensors
            .iter()
            .any(|t| t.name == name && t.dtype == dtype && t.shape == shape)
    };
    let mtp_head = head("eh_proj", DType::MQ4G256, &[5_120, 10_240])
        && head("wq", DType::MQ4G256, &[12_288, 5_120])
        && head("wk", DType::MQ4G256, &[1_024, 5_120])
        && head("wv", DType::MQ4G256, &[1_024, 5_120])
        && head("wo", DType::MQ4G256, &[5_120, 6_144])
        && head("ffn_gate", DType::MQ4G256, &[17_408, 5_120])
        && head("ffn_up", DType::MQ4G256, &[17_408, 5_120])
        && head("ffn_down", DType::MQ4G256, &[5_120, 17_408]);
    match input.spec {
        KernelSpecRoute::NativeMtp if !mtp_head => {
            refuse("native MTP needs the MQ4G256 sidecar head tensors")
        }
        KernelSpecRoute::NativeMtp => Ok(Some(true)),
        _ => Ok(Some(false)),
    }
}

/// Which plain (no-AWQ) rotate producer families a 27B load needs on top of
/// the AWQ closure. Each flag is set when a projection weight the family
/// feeds is present in the header without its `<stem>.awq_scale.weight`
/// sidecar; weights the model does not have never set it.
#[derive(Clone, Copy)]
struct Qwen36PlainRotate {
    /// RMSNorm producers: DeltaNet `in_proj_qkv`, attention `q_proj`, `mlp.gate_proj`.
    rmsnorm: bool,
    /// SwiGLU producer: `mlp.down_proj`.
    silu: bool,
    /// Gated-norm producer: `linear_attn.out_proj`.
    gated_norm: bool,
    /// Standalone rotate producer: `self_attn.o_proj`.
    rotate: bool,
}

impl Qwen36PlainRotate {
    fn any(self) -> bool {
        self.rmsnorm || self.silu || self.gated_norm || self.rotate
    }
}

fn qwen36_27b_plain_rotate(input: &RouteKernelInput<'_>) -> Qwen36PlainRotate {
    let awq_stems: std::collections::HashSet<&str> = input
        .tensors
        .iter()
        .filter_map(|t| t.name.strip_suffix(".awq_scale.weight"))
        .collect();
    let lacks_sidecar = |stems: &[&str]| {
        input
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("model.language_model.layers."))
            .filter_map(|t| t.name.strip_suffix(".weight"))
            .filter(|stem| stems.iter().any(|s| stem.ends_with(*s)))
            .any(|stem| !awq_stems.contains(stem))
    };
    Qwen36PlainRotate {
        rmsnorm: lacks_sidecar(&[
            ".linear_attn.in_proj_qkv",
            ".self_attn.q_proj",
            ".mlp.gate_proj",
        ]),
        silu: lacks_sidecar(&[".mlp.down_proj"]),
        gated_norm: lacks_sidecar(&[".linear_attn.out_proj"]),
        rotate: lacks_sidecar(&[".self_attn.o_proj"]),
    }
}

// ── Walked route plans ──────────────────────────────────────────────────

/// Historical walked coverage/provenance, one file per model family and exact
/// arch: the union of (module, symbol) bindings recorded by
/// `scripts/route_plan_walk.py` + `scripts/route_plan_build.py`. Never edit by
/// hand; record new walks after dispatch coverage or admission changes.
/// The constructor rebinds HIP rows to current registry recipes; recorded
/// embedded-binary identity remains strict.
const WALKED_ROUTE_PLAN_FILES: &[&str] = &[
    include_str!("route_plans/qwen3.8-flash-next.gfx1201.tsv"),
    include_str!("route_plans/qwen3.8-flash-next-base.mq4-xts.gfx1201.tsv"),
    include_str!("route_plans/qwen3.8-flash-next-base.mq2-xts.gfx1201.tsv"),
    include_str!("route_plans/qwen3.8-flash-next.gfx1100.tsv"),
];

/// One parsed `route_plans/*.tsv`.
#[derive(Debug)]
pub struct WalkedRoutePlan {
    pub family: &'static str,
    pub model_arch: u32,
    pub arch: &'static str,
    /// [`route_layout_sha256`] of the walked model header. For a family whose
    /// native-MTP head is a separate file (Qwen3.5) this is the trunk header
    /// alone, the AR route's layout.
    pub layout: &'static str,
    /// `#! mtp_layout`: [`route_layout_sha256`] of the trunk plus the sidecar
    /// head tensors a native-MTP load passes. `None` when the walked header
    /// carries its own head (Flash-Next); the native-MTP route then matches
    /// [`Self::layout`].
    pub mtp_layout: Option<&'static str>,
    /// `#! heads H,KV,D`: the `(n_heads, n_kv_heads, head_dim)` the walks
    /// resolved, when the walk log recorded them. `None` admits any
    /// (Flash-Next plans, which predate the line).
    pub heads: Option<(usize, usize, usize)>,
    /// Largest admitted `max_seq`: the walks' context ceiling.
    pub max_seq: usize,
    /// Policy axis name and every value the walks covered.
    pub axes: Vec<(&'static str, Vec<&'static str>)>,
    /// `#! sealed`: spec routes (`ar`, `native_mtp`) whose whole walked
    /// closure was replayed against this plan and found closed. Empty: the
    /// loader keeps the plan open.
    pub sealed: Vec<&'static str>,
    pub rows: Vec<WalkedRouteRow>,
}

/// One walked module of one spec route and the symbols the dispatch bound.
/// For HIP rows, `sha256` is historical effective-source provenance; current
/// plans bind the registry's source, flags and scheduler profile after checking
/// the walked symbols and admission policy. Embedded rows retain strict
/// equality with the recorded binary image SHA-256.
#[derive(Debug)]
pub struct WalkedRouteRow {
    pub native_mtp: bool,
    pub embedded: bool,
    pub module: &'static str,
    pub symbols: Vec<&'static str>,
    pub sha256: &'static str,
}

fn parse_walked_route_plan(text: &'static str) -> Result<WalkedRoutePlan, String> {
    let mut directive = std::collections::HashMap::new();
    let mut axes = Vec::new();
    let mut sealed = Vec::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#! ") {
            let mut words = rest.split(' ');
            match (words.next(), words.next()) {
                (Some("axis"), Some(name)) => axes.push((name, words.collect::<Vec<_>>())),
                (Some("sealed"), Some(first)) => {
                    for route in std::iter::once(first).chain(words) {
                        if !matches!(route, "ar" | "native_mtp") {
                            return Err(format!("unknown sealed route in {line:?}"));
                        }
                        sealed.push(route);
                    }
                }
                (Some(key), Some(value)) => {
                    directive.insert(key, value);
                }
                _ => return Err(format!("malformed directive {line:?}")),
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let [route, kind, module, symbols, sha256] = line.split('\t').collect::<Vec<_>>()[..]
        else {
            return Err(format!("malformed row {line:?}"));
        };
        rows.push(WalkedRouteRow {
            native_mtp: match route {
                "ar" => false,
                "native_mtp" => true,
                _ => return Err(format!("unknown route in {line:?}")),
            },
            embedded: match kind {
                "hip" => false,
                "embedded" => true,
                _ => return Err(format!("unknown kind in {line:?}")),
            },
            module,
            symbols: symbols.split(',').collect(),
            sha256,
        });
    }
    let get = |key: &str| {
        directive
            .get(key)
            .copied()
            .ok_or_else(|| format!("missing #! {key}"))
    };
    let heads = directive
        .get("heads")
        .map(|value| {
            let numbers = value
                .split(',')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("heads: {e}"))?;
            let [heads, kv_heads, head_dim] = numbers[..] else {
                return Err(format!("heads {value:?}: expected H,KV,D"));
            };
            Ok((heads, kv_heads, head_dim))
        })
        .transpose()?;
    Ok(WalkedRoutePlan {
        family: get("family")?,
        model_arch: get("model_arch")?
            .parse()
            .map_err(|e| format!("model_arch: {e}"))?,
        arch: get("arch")?,
        layout: get("layout")?,
        mtp_layout: directive.get("mtp_layout").copied(),
        heads,
        max_seq: get("max_seq")?
            .parse()
            .map_err(|e| format!("max_seq: {e}"))?,
        axes,
        sealed,
        rows,
    })
}

static WALKED_ROUTE_PLANS: std::sync::LazyLock<Result<Vec<WalkedRoutePlan>, String>> =
    std::sync::LazyLock::new(|| {
        WALKED_ROUTE_PLAN_FILES
            .iter()
            .map(|text| parse_walked_route_plan(text))
            .collect()
    });

/// Every checked-in walked route plan, parsed once.
pub fn walked_route_plans() -> Result<&'static [WalkedRoutePlan], RegistryError> {
    WALKED_ROUTE_PLANS
        .as_deref()
        .map_err(|e| RegistryError::ConflictingPlan(format!("route plan file: {e}")))
}

/// Model-header layout fingerprint of a walked plan: SHA-256 over the sorted
/// `name\tdtype\tshape` lines of every header tensor (`route_plan_build.py`
/// computes the same digest from the HFQM header).
pub fn route_layout_sha256(tensors: &[KernelTensorMeta<'_>]) -> String {
    use sha2::{Digest, Sha256};
    let mut lines: Vec<String> = tensors
        .iter()
        .map(|t| {
            let shape: Vec<String> = t.shape.iter().map(usize::to_string).collect();
            format!("{}\t{:?}\t{}\n", t.name, t.dtype, shape.join(","))
        })
        .collect();
    lines.sort();
    format!("{:x}", Sha256::digest(lines.concat().as_bytes()))
}

/// Embedded certified-builder images by exact GPU arch: every `.hxaco` image
/// the dispatch can bind through `ensure_embedded_kernel` on that arch. A
/// walked embedded row is resolved here by the SHA-256 it recorded, so the
/// image that ran in the walk (not another image under the same module name)
/// is the one planned, and an image of another arch never resolves. The
/// module name and symbols stay those the audit recorded.
fn walked_arch_images(arch: &str) -> &'static [&'static [u8]] {
    match arch {
        "gfx1100" => &[
            kernels::QWEN4_MOE_IU4_SYM_PM_GFX1100,
            kernels::QWEN4_MOE_IU4_SYM_E8P_PM_GFX1100,
            kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
            kernels::E8P_I4_MOE_GEMV_PM_GFX1100,
            kernels::MQ_I4_A4_PM_GFX1100,
            kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_GFX1100,
            kernels::GEMM_MQ6G256V2_RESIDUAL_PM_GFX1100,
            kernels::GATED_DELTA_VECTOR_PM_GFX1100,
            kernels::GEMM_MQ4G256V2_MOE_GROUPED_K2_PM_GFX1100,
            kernels::EP_TILE_MASK_PM_GFX1100,
            kernels::EP_TILE_PULL_PM_GFX1100,
            kernels::EP_TILE_PULL_N8_PM_GFX1100,
        ],
        "gfx1151" => &[
            kernels::NPU_TRUNK_PM_GFX1151,
            kernels::NPU_TRUNK_ARENA_PM_GFX1151,
            kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151,
            kernels::QWEN4_MOE_IU4_SYM_E8P_PM_GFX1151,
            kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1151,
            kernels::QWEN4_MOE_IU4_SYM_W3_NT4_PM_GFX1151,
            kernels::QSA_GATHER_PM_GFX1151,
            kernels::QSA_SELECT_PM_GFX1151,
            kernels::QSA_SELECT_BF16_PM_GFX1151,
            kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1151,
            kernels::SPEC_ACCEPT_ORDERED_PM_GFX1151,
            kernels::QWEN4_MQ6_X4_PM_GFX1151,
            kernels::QWEN4_MQ4_X4_PM_GFX1151,
            kernels::QWEN4_LANE_ROWS_PM_GFX1151,
            kernels::MQ4_ROWS_X4_PM_GFX1151,
            kernels::BF16_ROWS_WIDE_PM_GFX1151,
            kernels::QWEN4_SHARED_GU_PM_GFX1151,
            kernels::MTP_DENSE_DET_PM_GFX1151,
            kernels::MTP_Q8_DET_PM_GFX1151,
            kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
            kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151,
            kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
            kernels::E8P_I4_MOE_GEMV_PM_GFX1151,
            kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_V2B_GFX1151,
            kernels::GEMM_MQ4G256V2_GATE_UP_SILU_A4_IU4_PM_V2B_GFX1151,
        ],
        "gfx1201" => &[
            kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201,
            kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1201,
            kernels::QWEN4_MOE_IU4_SYM_E8P_PM_GFX1201,
            kernels::QSA_GATHER_PM_GFX1201,
            kernels::DFLASH_SAMPLED_ACCEPT_PM_GFX1201,
            kernels::SPEC_SAMPLING_ROWS_PARALLEL_PM_GFX1201,
            kernels::SPEC_ACCEPT_ORDERED_PM_GFX1201,
            kernels::QWEN4_MQ6_X4_PM_GFX1201,
            kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201,
            kernels::MQ4_ROWS_X4_PM_GFX1201,
            kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
            kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1,
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1_CONTROL,
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
            kernels::GEMM_MQ4G256V2_WMMA_FP8_GFX12_B1,
            kernels::MQ4_VERIFY_PM_GFX1201,
            kernels::MQ4_FP8_VERIFY_PM_GFX1201,
            kernels::MQ4V2_FP8_FRAGMENT_REPACK_GFX1201,
            kernels::QWEN4_EP_OWNER_ASSIGN_PM_GFX1201,
            kernels::EP_TILE_MASK_PM_GFX1201,
            kernels::EP_TILE_PULL_PM_GFX1201,
            kernels::EP_TILE_PULL_N8_PM_GFX1201,
            kernels::GATED_DELTA_VECTOR_PM_GFX1201,
        ],
        _ => &[],
    }
}

/// `(arch, SHA-256 hex, image)` of every [`walked_arch_images`] entry,
/// hashed once on the first embedded row that needs it.
static WALKED_ARCH_IMAGE_SHA: std::sync::LazyLock<Vec<(&'static str, String, &'static [u8])>> =
    std::sync::LazyLock::new(|| {
        use sha2::{Digest, Sha256};
        ["gfx1100", "gfx1151", "gfx1201"]
            .into_iter()
            .flat_map(|arch| {
                walked_arch_images(arch)
                    .iter()
                    .map(move |image| (arch, format!("{:x}", Sha256::digest(image)), *image))
            })
            .collect()
    });

/// The embedded image of exact `arch` whose SHA-256 is `sha256`.
fn walked_arch_image(arch: &str, sha256: &str) -> Option<&'static [u8]> {
    WALKED_ARCH_IMAGE_SHA
        .iter()
        .find(|(a, digest, _)| *a == arch && digest == sha256)
        .map(|(_, _, image)| *image)
}

/// Runtime image of a walked embedded module, with its Radiowave manifest
/// when the module has one: accepted `pm_decode` twins and the native
/// objects the dispatch admits as in-memory code objects under these module
/// names (`ensure_embedded_kernel` callsites), then the exact-arch image
/// catalog ([`walked_arch_images`]) by the walked image's SHA-256. A walked
/// module with no runtime image refuses the plan.
fn walked_embedded_image(
    arch: &str,
    module: &'static str,
    sha256: &str,
) -> Option<(&'static str, &'static [u8], Option<&'static str>)> {
    if let Some((_, name, image, _, _)) = legacy_mq4_rowtile_image(arch) {
        if name == module {
            return Some((name, image, None));
        }
    }
    if let Some((name, image)) = crate::slab_moe::slab_rowtile_planner(arch) {
        if name == module {
            return Some((name, image, None));
        }
    }
    if arch == "gfx1151" {
        for (name, image) in [
            ("qwen4_lane_rows_pm", kernels::QWEN4_LANE_ROWS_PM_GFX1151),
            (
                crate::gemv::BF16_ROWS_WIDE_PM_MODULE,
                kernels::BF16_ROWS_WIDE_PM_GFX1151,
            ),
            ("mq4_rows_x4_pm_gfx1151", kernels::MQ4_ROWS_X4_PM_GFX1151),
        ] {
            if name == module {
                return Some((name, image, None));
            }
        }
    }
    if arch == "gfx1100" {
        for (name, image) in [
            ("mq4_rows_x4_pm_gfx1100", kernels::MQ4_ROWS_X4_PM_GFX1100),
            (
                "bf16_rows_wide_pm_gfx1100",
                kernels::BF16_ROWS_WIDE_PM_GFX1100,
            ),
            (
                "qwen4_lane_rows_pm_gfx1100",
                kernels::QWEN4_LANE_ROWS_PM_GFX1100,
            ),
        ] {
            if name == module {
                return Some((name, image, None));
            }
        }
    }
    if arch == "gfx1201" {
        if let Some(twin) = crate::pm_decode_twins::GFX1201_TWINS
            .iter()
            .find(|t| t.module == module)
        {
            return Some((twin.module, twin.image, Some(twin.radiowave_json)));
        }
        let (name, image, _) = crate::pm_mtp_gemm::route_object();
        if name == module {
            return Some((name, image, None));
        }
        if let Some((name, image, _)) = crate::pm_xbatch::route_objects()
            .into_iter()
            .find(|o| o.0 == module)
        {
            return Some((name, image, None));
        }
        // Flash-Next (Qwen4) native objects: `gemm.rs` MQ6 x4 and symmetric
        // IU4 MoE, `tensor_ops.rs` QSA gathered attention.
        const QWEN4_GFX1201: &[(&str, &[u8])] = &[
            ("qwen4_mq6_x4_pm_gfx1201", kernels::QWEN4_MQ6_X4_PM_GFX1201),
            (
                "qwen4_moe_iu4_sym_pm_gfx1201",
                kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201,
            ),
            (
                "qwen4_moe_iu4_sym_w3_pm_gfx1201",
                kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1201,
            ),
            (
                "qwen4_moe_down_gp_pm_gfx1201",
                kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201,
            ),
            (
                "moe_rotate_i4_pm_gfx1201",
                kernels::MOE_ROTATE_I4_PM_GFX1201,
            ),
            ("qsa_gather_pm_gfx1201", kernels::QSA_GATHER_PM_GFX1201),
            ("mq4_rows_x4_pm_gfx1201", kernels::MQ4_ROWS_X4_PM_GFX1201),
            (
                "lane_rows_mq6_pm_gfx1201",
                kernels::LANE_ROWS_MQ6_PM_GFX1201,
            ),
            (
                "lane_rows_bf16_pm_gfx1201",
                kernels::LANE_ROWS_BF16_PM_GFX1201,
            ),
        ];
        if let Some(&(name, image)) = QWEN4_GFX1201.iter().find(|(name, _)| *name == module) {
            return Some((name, image, None));
        }
    }
    walked_arch_image(arch, sha256).map(|image| (module, image, None))
}

/// The walked plan for this load, when one exists for its model arch and
/// exact GPU arch. `Ok(None)` when no plan names the pair; `Err` when one
/// does but the header layout, a policy axis, `max_seq` or the feature
/// flags are outside the walks. Once admitted, the constructor emits the
/// current registry module/source/cache recipe for every walked row after
/// validating uniqueness and walked symbols/flags/profile/header/ABI/capture/shape
/// bindings; the checked-in row digest is historical provenance only and is
/// never compared against the current registry. A recipe-incompatible or
/// symbol-missing row is a hard `ConflictingPlan`, never a fallback; scalar,
/// admission and lazy-loading ceilings are unchanged by this binding.
fn walked_route_entries(
    input: &RouteKernelInput<'_>,
) -> Result<Option<(RouteKernelPlan, RouteSeal)>, RegistryError> {
    walked_route_entries_in(walked_route_plans()?, input)
}

fn walked_route_entries_in(
    plans: &'static [WalkedRoutePlan],
    input: &RouteKernelInput<'_>,
) -> Result<Option<(RouteKernelPlan, RouteSeal)>, RegistryError> {
    use sha2::{Digest, Sha256};
    let candidates: Vec<&'static WalkedRoutePlan> = plans
        .iter()
        .filter(|plan| plan.model_arch == input.model_arch && plan.arch == input.arch)
        .collect();
    let Some(family) = candidates.first().map(|plan| plan.family) else {
        return Ok(None);
    };
    let refuse = |why: String| Err(RegistryError::UnsupportedRoute(format!("{family}: {why}")));
    let layout = route_layout_sha256(input.tensors);
    // A sidecar-head family (Qwen3.5) walks its native-MTP route over the
    // trunk plus head tensors, whose layout `mtp_layout` records; every
    // other load matches the walked trunk layout.
    let walked_layout = |plan: &WalkedRoutePlan| match (input.spec, plan.mtp_layout) {
        (KernelSpecRoute::NativeMtp, Some(mtp_layout)) => mtp_layout,
        _ => plan.layout,
    };
    let Some(plan) = candidates
        .into_iter()
        .find(|plan| walked_layout(plan) == layout)
    else {
        return refuse(format!(
            "header layout {layout} has no walked route on {} ({:?})",
            input.arch, input.spec
        ));
    };
    if input.max_seq > plan.max_seq {
        return refuse(format!(
            "max_seq {} above the walked {}",
            input.max_seq, plan.max_seq
        ));
    }
    if let Some(heads) = plan.heads {
        let resolved = (input.n_heads, input.n_kv_heads, input.head_dim);
        if resolved != heads {
            return refuse(format!(
                "heads/kv heads/head dim {resolved:?} differ from the walked {heads:?}"
            ));
        }
    }
    if input.flags.pm_decode != default_feature_flags(input.arch)?.pm_decode {
        return refuse("non-default kernel.pm_decode".into());
    }
    for (name, admitted) in &plan.axes {
        match input.route_axes.iter().find(|(axis, _)| axis == name) {
            Some((_, value)) if admitted.contains(&value.as_str()) => {}
            Some((_, value)) => {
                return refuse(format!(
                    "{name} = {value} is outside the walked {admitted:?}"
                ));
            }
            None => return refuse(format!("the loader did not resolve the walked axis {name}")),
        }
    }
    if let Some((name, _)) = input
        .route_axes
        .iter()
        .find(|(axis, _)| !plan.axes.iter().any(|(n, _)| n == axis))
    {
        return refuse(format!("axis {name} was not walked"));
    }
    let native_mtp = input.spec == KernelSpecRoute::NativeMtp;
    // The checked-in row digests are the historical walk-time capture
    // identities: kept for provenance, but the admitted plan is bound to the
    // current registry recipe below. Regeneration runs the existing
    // `scripts/route_plan_walk.py` walks plus `scripts/route_plan_build.py`
    // (same argv/inputs/outputs as the checked-in plans) against the current
    // Beta registry; no hash alias or fallback.
    let stale = |module: &str, what: &str| {
        RegistryError::ConflictingPlan(format!(
            "{family} route plan: {module} {what} differs from the walked dispatch; regenerate route_plans/"
        ))
    };
    let inventory = entries(input.arch, input.hipcc_extra_flags)?;
    let mut planned = Vec::new();
    for row in plan.rows.iter().filter(|row| row.native_mtp == native_mtp) {
        if row.embedded {
            let (module, image, radiowave_json) =
                walked_embedded_image(input.arch, row.module, row.sha256).ok_or_else(|| {
                    RegistryError::ConflictingPlan(format!(
                        "{family} route plan: no runtime image for embedded {} on {}",
                        row.module, input.arch
                    ))
                })?;
            if format!("{:x}", Sha256::digest(image)) != row.sha256 {
                return Err(stale(row.module, "image"));
            }
            planned.push(PlannedKernel::Embedded {
                module,
                image,
                radiowave_json,
                symbols: &row.symbols,
            });
            continue;
        }
        // HIP rows come from the installer inventory, so a walked plan is
        // always served by the release kernel pack.
        let mut matches = inventory.iter().filter(|entry| entry.module == row.module);
        let found = matches
            .next()
            .ok_or_else(|| RegistryError::UnsupportedModule {
                arch: input.arch.to_owned(),
                module: row.module.to_owned(),
            })?;
        if matches.next().is_some() {
            return Err(RegistryError::ConflictingPlan(format!(
                "module {} has two registry recipes",
                row.module
            )));
        }
        // Bind the current registry recipe: no whole-source digest comparison
        // against the historical walked capture. The emitted entry carries the
        // current registry module/source/cache identity (`found` below), after
        // the uniqueness and walked-symbol checks that keep the strict
        // actual-current plan-to-pack matching.
        if let Some(symbol) = row
            .symbols
            .iter()
            .find(|symbol| !found.symbols.contains(symbol))
        {
            return Err(RegistryError::ConflictingPlan(format!(
                "{family} route plan: packaged {} does not export walked symbol {symbol}",
                row.module
            )));
        }
        planned.push(PlannedKernel::Hip(KernelEntry {
            arch: found.arch,
            module: found.module,
            symbols: &row.symbols,
            source: found.source.clone(),
            flags: found.flags.clone(),
            scheduler_profile: found.scheduler_profile.clone(),
        }));
    }
    add_wide_twin_entries(input.arch, &mut planned);
    let seal = if plan
        .sealed
        .contains(&if native_mtp { "native_mtp" } else { "ar" })
    {
        RouteSeal::Sealed
    } else {
        RouteSeal::Open(family)
    };
    Ok(Some((RouteKernelPlan { entries: planned }, seal)))
}

// ── Exact-wide fixed-eight-row PM tiles ─────────────────────────────────
//
// Walks record the narrow route. Developer-family switches add only the
// fixed-t8 PM exports below; singleton and narrow HIP recipes stay intact.
// K1 tiles are projected by `slab_embedded_entries` and `slab_moe.rs`.

use crate::gemv::WideTwinFamily;

const MQ6_GFX1201_ROWTILE_SYMBOLS: [&str; 2] = [
    "gemv_mq6g256v2_rows_t8_pm_gfx1201",
    "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201",
];
const BF16_GFX1201_ROWTILE_SYMBOLS: [&str; 2] = [
    "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201",
    "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201",
];
const MQ6_GFX1100_ROWTILE_SYMBOLS: [&str; 2] = [
    "gemv_mq6g256v2_rows_t8_pm_gfx1100",
    "gemv_mq6g256v2_rows_x4_t8_pm_gfx1100",
];
const BF16_GFX1100_ROWTILE_SYMBOLS: [&str; 2] = [
    "gemv_bf16_xf32_rows_x4_t8_pm_gfx1100",
    "gemv_bf16_xf32_k4_rows_t8_pm_gfx1100",
];
const Q8_GFX1100_ROWTILE_SYMBOLS: [&str; 5] = [
    "gemv_q8_0_k8_rows_t8_pm_gfx1100",
    "gemv_q8_0_k320_rows_t8_pm_gfx1100",
    "gemv_q8_0_k2560_rows_t8_pm_gfx1100",
    "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100",
    "gemv_q8_0_wide_rows_t8_pm_gfx1100",
];

/// The MQ4 PM image of `arch`: merge-cell slot, module, image and wide symbols.
fn wide_twin_mq4_image(
    arch: &str,
) -> Option<(usize, &'static str, &'static [u8], &'static [&'static str])> {
    match arch {
        "gfx1151" => Some((
            0,
            "mq4_rows_x4_pm_gfx1151",
            kernels::MQ4_ROWS_X4_PM_GFX1151,
            &["gemv_mq4g256v2_rows_x4_t8_pm_gfx1151"],
        )),
        "gfx1201" => Some((
            1,
            "mq4_rows_x4_pm_gfx1201",
            kernels::MQ4_ROWS_X4_PM_GFX1201,
            &["gemv_mq4g256v2_rows_x4_t8_pm_gfx1201"],
        )),
        "gfx1100" => Some((
            2,
            "mq4_rows_x4_pm_gfx1100",
            kernels::MQ4_ROWS_X4_PM_GFX1100,
            &["gemv_mq4g256v2_rows_x4_t8_pm_gfx1100"],
        )),
        _ => None,
    }
}

/// The representation-specific legacy GU/Down image and its actual exports.
/// This does not admit a shape, change AR, or grant a qualified width.
pub(crate) fn legacy_mq4_rowtile_image(
    arch: &str,
) -> Option<(
    usize,
    &'static str,
    &'static [u8],
    &'static str,
    &'static [&'static str],
)> {
    match arch {
        "gfx1151" => Some((
            0,
            "legacy_mq4_rowtiles_pm_gfx1151",
            kernels::LEGACY_MQ4_ROWTILES_PM_GFX1151,
            kernels::LEGACY_MQ4_ROWTILES_PM_GFX1151_SHA256,
            &[
                "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151",
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151",
            ],
        )),
        "gfx1201" => Some((
            1,
            "legacy_mq4_rowtiles_pm_gfx1201",
            kernels::LEGACY_MQ4_ROWTILES_PM_GFX1201,
            kernels::LEGACY_MQ4_ROWTILES_PM_GFX1201_SHA256,
            &[
                "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201",
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201",
            ],
        )),
        _ => None,
    }
}

/// `base` plus the `extra` symbols it lacks (`base` itself when it has them
/// all). The merged list is owned by `cell` (one per planned module; the
/// switches and the walked plan are process constants).
fn with_wide_twin_symbols(
    cell: &'static std::sync::OnceLock<Vec<&'static str>>,
    base: &'static [&'static str],
    extra: &'static [&'static str],
) -> &'static [&'static str] {
    if extra.iter().all(|symbol| base.contains(symbol)) {
        return base;
    }
    cell.get_or_init(|| {
        base.iter()
            .chain(extra.iter().filter(|symbol| !base.contains(symbol)))
            .copied()
            .collect()
    })
}

/// Merged symbols retain every narrow export already selected by the walk.
static WIDE_TWIN_MQ4_MERGED: [std::sync::OnceLock<Vec<&'static str>>; 3] =
    [const { std::sync::OnceLock::new() }; 3];
/// gfx1151 lane/BF16, gfx1201 MQ6/BF16 and gfx1100 lane/BF16 PM modules.
static WIDE_TWIN_PM_MERGED: [std::sync::OnceLock<Vec<&'static str>>; 6] =
    [const { std::sync::OnceLock::new() }; 6];
/// Initialization merges the runtime walked symbol list, not a fixed list.
static LEGACY_MQ4_ROWTILE_MERGED: [std::sync::OnceLock<Vec<&'static str>>; 2] =
    [const { std::sync::OnceLock::new() }; 2];

/// Add `extra` to the embedded `module`'s planned symbols, or plan the module.
fn plan_wide_embedded(
    planned: &mut Vec<PlannedKernel>,
    cell: &'static std::sync::OnceLock<Vec<&'static str>>,
    module: &'static str,
    image: &'static [u8],
    extra: &'static [&'static str],
) {
    let present = planned.iter().position(
        |entry| matches!(entry, PlannedKernel::Embedded { module: m, .. } if *m == module),
    );
    match present {
        Some(index) => {
            if let PlannedKernel::Embedded { symbols, .. } = &mut planned[index] {
                *symbols = with_wide_twin_symbols(cell, *symbols, extra);
            }
        }
        None => planned.push(PlannedKernel::Embedded {
            module,
            image,
            radiowave_json: None,
            symbols: extra,
        }),
    }
}

/// Add the PM row tiles of every switched-on family to a walked route.
fn add_wide_twin_entries(arch: &str, planned: &mut Vec<PlannedKernel>) {
    add_rowtile_entries(arch, planned, |family| family.enabled());
}

fn add_rowtile_entries(
    arch: &str,
    planned: &mut Vec<PlannedKernel>,
    enabled: impl Fn(WideTwinFamily) -> bool,
) {
    if enabled(WideTwinFamily::K1) {
        if let Some(entry) = rowtile_planner_entry(arch) {
            if !planned
                .iter()
                .any(|existing| existing.module() == entry.module())
            {
                planned.push(entry);
            }
        }
    }
    if enabled(WideTwinFamily::Mq4) {
        if let Some((slot, module, image, extra)) = wide_twin_mq4_image(arch) {
            plan_wide_embedded(planned, &WIDE_TWIN_MQ4_MERGED[slot], module, image, extra);
        }
        if let Some((slot, module, image, _, symbols)) = legacy_mq4_rowtile_image(arch) {
            plan_wide_embedded(
                planned,
                &LEGACY_MQ4_ROWTILE_MERGED[slot],
                module,
                image,
                symbols,
            );
        }
    }
    if arch == "gfx1151" {
        // The two families share one image; cache each enabled combination.
        static LANE_SYMBOLS: std::sync::LazyLock<[Vec<&str>; 4]> = std::sync::LazyLock::new(|| {
            [
                Vec::new(),
                crate::lane_rows_pm::MQ6_WIDE_SYMBOLS.to_vec(),
                crate::lane_rows_pm::Q8_WIDE_SYMBOLS.to_vec(),
                crate::lane_rows_pm::MQ6_WIDE_SYMBOLS
                    .into_iter()
                    .chain(crate::lane_rows_pm::Q8_WIDE_SYMBOLS)
                    .collect(),
            ]
        });
        let mq6 = enabled(WideTwinFamily::Mq6);
        let q8 = enabled(WideTwinFamily::Q8);
        let lane = LANE_SYMBOLS[usize::from(mq6) | (usize::from(q8) << 1)].as_slice();
        if !lane.is_empty() {
            plan_wide_embedded(
                planned,
                &WIDE_TWIN_PM_MERGED[0],
                "qwen4_lane_rows_pm",
                kernels::QWEN4_LANE_ROWS_PM_GFX1151,
                lane,
            );
        }
        if enabled(WideTwinFamily::Bf16) {
            plan_wide_embedded(
                planned,
                &WIDE_TWIN_PM_MERGED[1],
                crate::gemv::BF16_ROWS_WIDE_PM_MODULE,
                kernels::BF16_ROWS_WIDE_PM_GFX1151,
                &crate::gemv::BF16_ROWS_WIDE_PM_SYMBOLS,
            );
        }
    } else if arch == "gfx1201" {
        for (family, slot, module, image, symbols) in [
            (
                WideTwinFamily::Mq6,
                2,
                "lane_rows_mq6_pm_gfx1201",
                kernels::LANE_ROWS_MQ6_PM_GFX1201,
                MQ6_GFX1201_ROWTILE_SYMBOLS.as_slice(),
            ),
            (
                WideTwinFamily::Bf16,
                3,
                "lane_rows_bf16_pm_gfx1201",
                kernels::LANE_ROWS_BF16_PM_GFX1201,
                BF16_GFX1201_ROWTILE_SYMBOLS.as_slice(),
            ),
        ] {
            if enabled(family) {
                plan_wide_embedded(planned, &WIDE_TWIN_PM_MERGED[slot], module, image, symbols);
            }
        }
    } else if arch == "gfx1100" {
        static LANE_SYMBOLS: std::sync::LazyLock<[Vec<&str>; 4]> = std::sync::LazyLock::new(|| {
            [
                Vec::new(),
                MQ6_GFX1100_ROWTILE_SYMBOLS.to_vec(),
                Q8_GFX1100_ROWTILE_SYMBOLS.to_vec(),
                MQ6_GFX1100_ROWTILE_SYMBOLS
                    .into_iter()
                    .chain(Q8_GFX1100_ROWTILE_SYMBOLS)
                    .collect(),
            ]
        });
        let mq6 = enabled(WideTwinFamily::Mq6);
        let q8 = enabled(WideTwinFamily::Q8);
        let lane = LANE_SYMBOLS[usize::from(mq6) | (usize::from(q8) << 1)].as_slice();
        if !lane.is_empty() {
            plan_wide_embedded(
                planned,
                &WIDE_TWIN_PM_MERGED[4],
                "qwen4_lane_rows_pm_gfx1100",
                kernels::QWEN4_LANE_ROWS_PM_GFX1100,
                lane,
            );
        }
        if enabled(WideTwinFamily::Bf16) {
            plan_wide_embedded(
                planned,
                &WIDE_TWIN_PM_MERGED[5],
                "bf16_rows_wide_pm_gfx1100",
                kernels::BF16_ROWS_WIDE_PM_GFX1100,
                &BF16_GFX1100_ROWTILE_SYMBOLS,
            );
        }
    }
}

fn rowtile_planner_entry(arch: &str) -> Option<PlannedKernel> {
    static SYMBOLS: [&[&str]; 3] = [
        &["qwen4_slab_union_plan_t8_pm_gfx1151"],
        &["qwen4_slab_union_plan_t8_pm_gfx1201"],
        &["qwen4_slab_union_plan_t8_pm_gfx1100"],
    ];
    let slot = match arch {
        "gfx1151" => 0,
        "gfx1201" => 1,
        "gfx1100" => 2,
        _ => return None,
    };
    let (module, image) = crate::slab_moe::slab_rowtile_planner(arch)?;
    Some(PlannedKernel::Embedded {
        module,
        image,
        radiowave_json: None,
        symbols: SYMBOLS[slot],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::{HashMap, HashSet};

    #[test]
    fn a3b_qresident_q8_has_isolated_exports_and_shared_dense_body() {
        let direct = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201";
        let range = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_gfx1201";
        let entry = lookup("gfx1201", direct, "").unwrap();
        assert_eq!(
            entry.symbols,
            [
                "attention_fp8_e4m3_fa2_gqa_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_merge_gfx1201",
                "attention_fp8_e4m3_fa2_gqa_partial_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
                "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201",
                direct,
                range,
            ]
        );
        let source = entry.source();
        let body = source
            .strip_prefix(concat!(
                "#define HIPFIRE_FA2_QRESIDENT_V2_Q8_A3B 1\n",
                "#define HIPFIRE_FA2_QRESIDENT_V2_Q8 1\n",
                "#define HIPFIRE_FA2_FP8 1\n",
                "#define HIPFIRE_FA2_KMODE 8\n",
            ))
            .unwrap();
        let dense = lookup(
            "gfx1201",
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
            "",
        )
        .unwrap();
        assert_eq!(
            body,
            dense
                .source()
                .strip_prefix(concat!(
                    "#define HIPFIRE_FA2_QRESIDENT_V2_Q8 1\n",
                    "#define HIPFIRE_FA2_FP8 1\n",
                    "#define HIPFIRE_FA2_KMODE 8\n",
                ))
                .unwrap()
        );
        for contract in [
            "QR_HEADS = 16, QR_KV_HEADS = 2, QR_GQA = 8",
            "QR_HEADS = 24, QR_KV_HEADS = 4, QR_GQA = 6",
            "row_base + 383 < QR_GQA * run_rows",
            "v2i_t Qfr[16]",
            "constexpr int TK = 48",
            "LDS + 49920",
            "const uint32_t* p = (const uint32_t*)(k_cache",
            "min(kb + r, seq_len - 1)",
            "vp + (unsigned long long)key * QR_KV_STRIDE",
            "__builtin_amdgcn_perm(w1, w0, 0x05010400u)",
        ] {
            assert!(
                body.contains(contract),
                "missing source contract: {contract}"
            );
        }
        assert!(!dense.symbols.contains(&direct));
        assert!(!dense.symbols.contains(&range));
    }

    #[test]
    fn k640_registry_matches_gfx1151_only_launcher_admission() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let registry = entries(arch, "").unwrap();
            assert_eq!(
                registry
                    .iter()
                    .any(|entry| entry.module == "gemv_mq4g128v2_moe_down_top10_k640"),
                arch == "gfx1151",
                "{arch}: K640 registry must match the launchers architecture guard",
            );
        }
    }

    #[test]
    fn native_qwen4_gfx1151_route_entries_refuses_partial_closed_plan() {
        let flags = default_feature_flags("gfx1151").unwrap();
        for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
            let input = RouteKernelInput {
                arch: "gfx1151",
                model_arch: 16,
                tensors: &[],
                n_heads: 32,
                n_kv_heads: 4,
                head_dim: 128,
                max_seq: 32768,
                prefill_chunk_rows: 8192,
                kv_k: "fp8",
                kv_v: "fp8",
                state_quant: "q8",
                spec,
                host_mapped_experts: false,
                retained_decode: true,
                hipcc_extra_flags: "",
                flags: &flags,
                route_axes: &[],
            };
            match route_entries(&input) {
                Err(RegistryError::UnsupportedRoute(reason)) => {
                    assert_eq!(reason, "model arch 16 on gfx1151 has no closed route",)
                }
                other => panic!("gfx1151 must keep lazy loading, not a partial plan: {other:?}"),
            }
        }
    }

    #[test]
    fn mq4v2_k5120_inventory_preserves_generic_and_limits_arch() {
        let module = "fused_gate_up_mq4g256v2_k5120_gfx1100";
        let registry = entries("gfx1100", "").unwrap();
        let candidate = registry
            .iter()
            .find(|entry| entry.module == module)
            .unwrap();
        assert_eq!(candidate.symbols, [module]);
        let body = candidate
            .source()
            .strip_prefix(
                "#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_mq4g256v2_k5120_gfx1100\n",
            )
            .unwrap();
        assert_eq!(
            body.replace(
                "const int groups_per_row = 20;",
                "const int groups_per_row = K / 256;"
            ),
            kernels::FUSED_GATE_UP_MQ4G256V2_SRC
        );
        for arch in ["gfx1151", "gfx1201", "gfx906", "gfx942"] {
            assert!(!entries(arch, "")
                .unwrap()
                .iter()
                .any(|entry| entry.module == module));
        }
    }

    #[test]
    fn gfx1100_slab_planner_is_the_gfx1201_source_under_an_arch_key() {
        let entry = lookup("gfx1100", "qwen4_slab_union_plan_gfx1100", "").unwrap();
        assert_eq!(entry.symbols, ["qwen4_slab_union_plan"]);
        assert_eq!(entry.source(), kernels::qwen4_slab_union_plan_gfx1100_src());
        // Registry and launcher share one assembled expression: the shared planner file minus ONLY
        // its exact arch guard (the runtime table test pins the body and the forbidden constructs).
        assert_eq!(
            kernels::qwen4_slab_union_plan_gfx1100_src(),
            kernels::QWEN4_SLAB_UNION_PLAN_SRC.replacen(
                kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD,
                "",
                1
            )
        );
        // The gfx1201 entry keeps its historical key; neither arch carries the other's.
        assert!(entries("gfx1201", "")
            .unwrap()
            .iter()
            .any(|e| e.module == "qwen4_slab_union_plan"));
        assert!(!entries("gfx1201", "")
            .unwrap()
            .iter()
            .any(|e| e.module == "qwen4_slab_union_plan_gfx1100"));
        assert!(!entries("gfx1100", "")
            .unwrap()
            .iter()
            .any(|e| e.module == "qwen4_slab_union_plan"));
        for arch in ["gfx906", "gfx942"] {
            assert!(!entries(arch, "")
                .unwrap()
                .iter()
                .any(|e| e.module.starts_with("qwen4_slab_union_plan")));
        }
        // gfx1151 carries the shared planner under the same historical key and the hipcc tile mask.
        let gfx1151 = entries("gfx1151", "").unwrap();
        assert!(gfx1151.iter().any(|e| e.module == "qwen4_slab_union_plan"));
        assert!(gfx1151.iter().any(|e| e.module == "qwen4_slab_tile_mask"));
        assert!(!gfx1151
            .iter()
            .any(|e| e.module == "qwen4_slab_union_plan_gfx1100"));
    }

    /// The gfx1100 header checker is the gfx1201 source with exactly its arch guard removed and
    /// exactly its entry renamed (a genuine gfx1100 compile, no fake arch macro), registered
    /// only in the gfx1100 arm under its own module key.
    #[test]
    fn gfx1100_sym_check_is_the_gfx1201_source_with_two_exact_edits() {
        let original = kernels::QWEN4_MOE_SYM_CHECK_GFX1201_SRC;
        let assembled = kernels::qwen4_moe_sym_check_gfx1100_src();
        assert_eq!(
            original
                .matches(kernels::QWEN4_MOE_SYM_CHECK_GFX1201_ARCH_GUARD)
                .count(),
            1
        );
        assert_eq!(
            original
                .matches(kernels::QWEN4_MOE_SYM_CHECK_GFX1201_ENTRY_DECL)
                .count(),
            1
        );
        assert_eq!(original.matches("qwen4_moe_sym_check_gfx1201").count(), 1);
        assert_eq!(
            assembled,
            original
                .replacen(kernels::QWEN4_MOE_SYM_CHECK_GFX1201_ARCH_GUARD, "", 1)
                .replacen(
                    kernels::QWEN4_MOE_SYM_CHECK_GFX1201_ENTRY_DECL,
                    kernels::QWEN4_MOE_SYM_CHECK_GFX1100_ENTRY_DECL,
                    1
                )
        );
        assert_eq!(assembled.matches("qwen4_moe_sym_check_gfx1100(").count(), 1);
        assert!(!assembled.contains("qwen4_moe_sym_check_gfx1201"));
        assert!(!assembled.contains("#error"));
        for forbidden in [
            "#define __gfx",
            "#undef __gfx",
            "__gfx",
            "__builtin_amdgcn",
            "__shfl",
            "__ballot",
            "__activemask",
            "warpSize",
        ] {
            assert!(
                !assembled.contains(forbidden),
                "{forbidden} in the assembled gfx1100 checker source"
            );
        }
        let gfx1100 = entries("gfx1100", "").unwrap();
        let check = gfx1100
            .iter()
            .find(|e| e.module == "qwen4_moe_sym_check_gfx1100")
            .expect("gfx1100 checker module");
        assert_eq!(check.symbols, ["qwen4_moe_sym_check_gfx1100"]);
        assert_eq!(check.source(), assembled);
        for arch in ["gfx1151", "gfx1201", "gfx906", "gfx942"] {
            assert!(
                !entries(arch, "")
                    .unwrap()
                    .iter()
                    .any(|e| e.module == "qwen4_moe_sym_check_gfx1100"),
                "{arch}"
            );
        }
        assert!(!gfx1100
            .iter()
            .any(|e| e.module == "qwen4_moe_sym_check_gfx1201"));
    }

    #[test]
    fn gfx1100_slab_serving_inventory_is_complete_and_acc_free() {
        // hipcc entries inside the gfx1100 registry arm: planner, stable grouping, A4 rotate.
        let registry = entries("gfx1100", "").unwrap();
        let scatter = registry
            .iter()
            .find(|e| e.module == "qwen4_moe_scatter_stable_top10_gfx1100")
            .unwrap();
        assert_eq!(
            scatter.symbols,
            [
                "qwen4_moe_group_ranks",
                "qwen4_moe_group_prefix",
                "qwen4_moe_group_scatter"
            ]
        );
        assert_eq!(
            scatter.source(),
            kernels::qwen4_moe_scatter_stable_top10_gfx1100_src()
        );
        // The grouping source is the gfx1201/gfx1151/gfx942 file minus ONLY its exact arch guard:
        // no fake compiler arch macro, and no wave-sensitive construct remains in the body.
        let original = kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_SRC;
        let assembled = kernels::qwen4_moe_scatter_stable_top10_gfx1100_src();
        assert_eq!(
            assembled,
            original.replacen(kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_ARCH_GUARD, "", 1)
        );
        assert_eq!(
            original.len() - assembled.len(),
            kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_ARCH_GUARD.len()
        );
        for forbidden in [
            "#define __gfx",
            "#undef __gfx",
            "__shfl",
            "__ballot",
            "__activemask",
            "warpSize",
            "__builtin_amdgcn",
            "__gfx",
        ] {
            assert!(
                !assembled.contains(forbidden),
                "{forbidden} in the assembled grouping source"
            );
        }
        assert!(registry
            .iter()
            .any(|e| e.module == "mq_rotate_x_i4" && e.symbols == ["mq_rotate_x_i4"]));
        assert!(!entries("gfx1201", "")
            .unwrap()
            .iter()
            .any(|e| e.module.ends_with("_gfx1100") && e.module.starts_with("qwen4_moe_scatter")));
        // Embedded images of an MQ4-only layer (cold codec 0): K1 (hot MQ4-only module), K2 (MQ4,
        // serving symbols only), K3, tile mask. No E8P module is listed or loaded.
        let mq4_only = RouteKernelPlan {
            entries: slab_embedded_entries("gfx1100", 0).unwrap(),
        };
        // An E8P layer (cold codec 3) adds the cold K1 and the E8P K2, each exactly once.
        let e8p = RouteKernelPlan {
            entries: slab_embedded_entries("gfx1100", 3).unwrap(),
        };
        let mut mq4_modules = vec![
            "mq_i4_moe_gemv_mq4only_pm_gfx1100",
            "qwen4_moe_iu4_sym_pm_gfx1100",
            "mq_i4_a4_pm_gfx1100",
            "ep_tile_mask_pm_gfx1100",
        ];
        let planner_count = usize::from(WideTwinFamily::K1.enabled());
        if planner_count != 0 {
            mq4_modules.push("qwen4_slab_union_plan_t8_pm_gfx1100");
        }
        for (plan, expected, expected_symbols) in [
            (
                &mq4_only,
                mq4_modules.clone(),
                4 + 2 + 1 + 1 + planner_count,
            ),
            (
                &e8p,
                [
                    &mq4_modules[..4],
                    &[
                        "e8p_i4_moe_gemv_pm_gfx1100",
                        "qwen4_moe_iu4_sym_e8p_pm_gfx1100",
                    ],
                    &mq4_modules[4..],
                ]
                .concat(),
                4 + 2 + 1 + 1 + 4 + 2 + planner_count,
            ),
        ] {
            validate_route_plan(plan).unwrap();
            let modules: Vec<_> = plan.entries.iter().map(PlannedKernel::module).collect();
            assert_eq!(modules, expected);
            // Neither the retired mixed K1 module nor any module twice.
            assert!(!modules.contains(&"mq_i4_moe_gemv_pm_gfx1100"));
            let mut unique = modules.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), modules.len(), "a module is ensured twice");
            // No W2 / codec-2 entry anywhere in the inventory.
            for planned in &plan.entries {
                assert!(!planned.module().contains("w2") && !planned.module().contains("mq2i4"));
                assert!(planned
                    .symbols()
                    .iter()
                    .all(|s| !s.contains("w2") && !s.contains("mq2i4")));
            }
            let mut seen = Vec::new();
            for planned in &plan.entries {
                let PlannedKernel::Embedded { image, symbols, .. } = planned else {
                    panic!("hip entry in the embedded slab inventory")
                };
                for &symbol in *symbols {
                    assert!(!seen.contains(&symbol), "{symbol} listed twice");
                    seen.push(symbol);
                    assert!(
                        !symbol.ends_with("_acc"),
                        "{symbol}: qualification twin must not be served"
                    );
                    assert!(
                        image
                            .windows(symbol.len() + 3)
                            .any(|w| w.starts_with(symbol.as_bytes()) && w.ends_with(b".kd")),
                        "{symbol}.kd missing"
                    );
                }
            }
            assert_eq!(seen.len(), expected_symbols);
        }
        // The MQ4-only preload never loads an E8P module or entry.
        assert!(
            mq4_only
                .entries
                .iter()
                .all(|p| !p.module().contains("e8p")
                    && p.symbols().iter().all(|s| !s.contains("e8p")))
        );
        // The E8P layer's exact cold entries: K1 (R256 gate/up, R128 down) and the K2 NT1 pair.
        let cold_symbols: Vec<_> = e8p.entries[4..]
            .iter()
            .take(2)
            .flat_map(|p| p.symbols().iter().copied())
            .collect();
        assert_eq!(
            cold_symbols,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1100",
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8",
                "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1100",
                "qwen4_moe_down_iu4_sym_e8p_pm_gfx1100",
            ]
        );
        // Registry presence is not permission: W3 (1) on gfx1100, W2 (2) and unknown codecs are refused.
        for codec in [1u8, 2, 4, 255] {
            assert!(
                matches!(
                    slab_embedded_entries("gfx1100", codec),
                    Err(RegistryError::UnsupportedRoute(_))
                ),
                "gfx1100 codec {codec}"
            );
        }
        // gfx1201's inventory is untouched; unknown arches are refused.
        for codec in [0u8, 1, 3] {
            assert!(slab_embedded_entries("gfx1201", codec).unwrap().is_empty());
            assert!(slab_embedded_entries("gfx1151", codec).unwrap().is_empty());
        }
        assert!(matches!(
            slab_embedded_entries("gfx1101", 0),
            Err(RegistryError::UnsupportedArch(_))
        ));
    }

    // ---- Sealed-slab K1/K2 kernarg ABI: the uniform invalid-expert skip contract ----------------------------------
    //
    // Every embedded K1 image the slab route serves takes a trailing `n_experts@88` plus a zero `reserved@92`
    // (96-byte kernarg) and every NT1 K2 image a trailing `n_experts@60` (64-byte kernarg): the shared gfx1151 /
    // gfx1201 MQ4/W3 images, the gfx1100 MQ4-only K1, and the gfx1100 / gfx1201 E8P K1 and K2 images. The kernels
    // compare the routing expert id unsigned against `n_experts` and skip the workgroup (no load, no store) before
    // any pointer-table index, so the host-visible contract is exactly this descriptor/metadata layout.

    /// Minimal MessagePack value: enough to read the `NT_AMDGPU_METADATA` note of a code object.
    #[derive(Debug)]
    enum Mp {
        Uint(u64),
        Str(String),
        Arr(Vec<Mp>),
        Map(Vec<(Mp, Mp)>),
        Other,
    }

    impl Mp {
        fn get(&self, key: &str) -> &Mp {
            let Mp::Map(entries) = self else {
                panic!("not a map: {self:?}")
            };
            entries
                .iter()
                .find(|(k, _)| matches!(k, Mp::Str(s) if s == key))
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("metadata key {key} missing"))
        }
        fn uint(&self) -> u64 {
            match self {
                Mp::Uint(v) => *v,
                other => panic!("not an unsigned integer: {other:?}"),
            }
        }
        fn str(&self) -> &str {
            match self {
                Mp::Str(s) => s,
                other => panic!("not a string: {other:?}"),
            }
        }
        fn arr(&self) -> &[Mp] {
            match self {
                Mp::Arr(v) => v,
                other => panic!("not an array: {other:?}"),
            }
        }
    }

    fn mp_be(b: &[u8], at: &mut usize, n: usize) -> u64 {
        let v = b[*at..*at + n]
            .iter()
            .fold(0u64, |acc, &x| (acc << 8) | u64::from(x));
        *at += n;
        v
    }

    fn mp_str(b: &[u8], at: &mut usize, n: usize) -> Mp {
        let s = std::str::from_utf8(&b[*at..*at + n]).unwrap().to_owned();
        *at += n;
        Mp::Str(s)
    }

    fn mp_arr(b: &[u8], at: &mut usize, n: usize) -> Mp {
        Mp::Arr((0..n).map(|_| mp_parse(b, at)).collect())
    }

    fn mp_map(b: &[u8], at: &mut usize, n: usize) -> Mp {
        Mp::Map(
            (0..n)
                .map(|_| {
                    let key = mp_parse(b, at);
                    (key, mp_parse(b, at))
                })
                .collect(),
        )
    }

    fn mp_parse(b: &[u8], at: &mut usize) -> Mp {
        let tag = b[*at];
        *at += 1;
        match tag {
            0x00..=0x7f => Mp::Uint(u64::from(tag)),
            0x80..=0x8f => mp_map(b, at, usize::from(tag & 0x0f)),
            0x90..=0x9f => mp_arr(b, at, usize::from(tag & 0x0f)),
            0xa0..=0xbf => mp_str(b, at, usize::from(tag & 0x1f)),
            0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => Mp::Other,
            0xc4..=0xc6 => {
                let n = mp_be(b, at, 1 << (tag - 0xc4)) as usize;
                *at += n;
                Mp::Other
            }
            0xca => {
                *at += 4;
                Mp::Other
            }
            0xcb => {
                *at += 8;
                Mp::Other
            }
            0xcc..=0xcf => Mp::Uint(mp_be(b, at, 1 << (tag - 0xcc))),
            0xd0..=0xd3 => {
                *at += 1 << (tag - 0xd0);
                Mp::Other
            }
            0xd9..=0xdb => {
                let n = mp_be(b, at, 1 << (tag - 0xd9)) as usize;
                mp_str(b, at, n)
            }
            0xdc | 0xdd => {
                let n = mp_be(b, at, 2 << (tag - 0xdc)) as usize;
                mp_arr(b, at, n)
            }
            0xde | 0xdf => {
                let n = mp_be(b, at, 2 << (tag - 0xde)) as usize;
                mp_map(b, at, n)
            }
            _ => panic!("unsupported MessagePack tag {tag:#x}"),
        }
    }

    /// The amdgcn code object of a clang offload bundle.
    fn amdgcn_elf(image: &[u8]) -> &[u8] {
        let u64_at = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap()) as usize;
        assert!(image.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"));
        let (count, mut p) = (u64_at(24), 32);
        let mut elf = None;
        for _ in 0..count {
            let (off, size, triple_len) = (u64_at(p), u64_at(p + 8), u64_at(p + 16));
            p += 24;
            if image[p..p + triple_len].windows(6).any(|w| w == b"amdgcn") {
                elf = Some(&image[off..off + size]);
            }
            p += triple_len;
        }
        elf.expect("no amdgcn code object in the bundle")
    }

    /// The decoded `NT_AMDGPU_METADATA` (note type 32, owner `AMDGPU`) of a code object.
    fn amdgpu_metadata(elf: &[u8]) -> Mp {
        let u16_e = |o: usize| u16::from_le_bytes(elf[o..o + 2].try_into().unwrap()) as usize;
        let u32_e = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
        let u64_e = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
        let (phoff, phentsize, phnum) = (u64_e(0x20), u16_e(0x36), u16_e(0x38));
        for i in 0..phnum {
            let ph = phoff + i * phentsize;
            if u32_e(ph) != 4 {
                continue; // not PT_NOTE
            }
            let (off, size) = (u64_e(ph + 8), u64_e(ph + 32));
            let mut q = off;
            while q < off + size {
                let (namesz, descsz, ty) = (u32_e(q), u32_e(q + 4), u32_e(q + 8));
                q += 12;
                let name = &elf[q..q + namesz];
                q += (namesz + 3) & !3;
                let desc = &elf[q..q + descsz];
                q += (descsz + 3) & !3;
                if ty == 32 && name == b"AMDGPU\0" {
                    return mp_parse(desc, &mut 0);
                }
            }
        }
        panic!("no NT_AMDGPU_METADATA note");
    }

    /// `[group_segment_fixed_size, private_segment_fixed_size, kernarg_segment_size]` words of the kernel
    /// descriptor `<symbol>.kd` (the bytes the loader reads), from the code object's symbol table.
    fn kernel_descriptor_words(elf: &[u8], symbol: &str) -> [u32; 3] {
        let u16_e = |o: usize| u16::from_le_bytes(elf[o..o + 2].try_into().unwrap()) as usize;
        let u32_e = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
        let u64_e = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
        let (shoff, shentsize, shnum) = (u64_e(0x28), u16_e(0x3a), u16_e(0x3c));
        let sh = |i: usize| shoff + i * shentsize;
        let symtab = (0..shnum)
            .find(|&i| u32_e(sh(i) + 4) == 2)
            .expect("no symbol table");
        let (sym_off, sym_size, strtab) = (
            u64_e(sh(symtab) + 0x18),
            u64_e(sh(symtab) + 0x20),
            u32_e(sh(symtab) + 0x28),
        );
        let str_off = u64_e(sh(strtab) + 0x18);
        let want = format!("{symbol}.kd");
        for sym in (sym_off..sym_off + sym_size).step_by(0x18) {
            let name_at = str_off + u32_e(sym);
            let name_end = name_at + elf[name_at..].iter().position(|&b| b == 0).unwrap();
            if &elf[name_at..name_end] != want.as_bytes() {
                continue;
            }
            let (shndx, value) = (u16_e(sym + 6), u64_e(sym + 8));
            let at = value - u64_e(sh(shndx) + 0x10) + u64_e(sh(shndx) + 0x18);
            let word = |o: usize| u32::from_le_bytes(elf[at + o..at + o + 4].try_into().unwrap());
            return [word(0), word(4), word(8)];
        }
        panic!("{want} missing from the symbol table");
    }

    /// One kernel of an embedded image: descriptor words plus the metadata `.args` as
    /// `(name, offset, size, value_kind)`.
    struct KernelAbi {
        group: u32,
        private: u32,
        kernarg: u32,
        block: u32,
        wave: u32,
        args: Vec<(String, u64, u64, String)>,
    }

    /// Every kernel of `image`, with the `.kd` descriptor words cross-checked against the metadata.
    fn image_kernel_abis(image: &[u8]) -> Vec<(String, KernelAbi)> {
        let elf = amdgcn_elf(image);
        let meta = amdgpu_metadata(elf);
        meta.get("amdhsa.kernels")
            .arr()
            .iter()
            .map(|k| {
                let name = k.get(".name").str().to_owned();
                let abi = KernelAbi {
                    group: k.get(".group_segment_fixed_size").uint() as u32,
                    private: k.get(".private_segment_fixed_size").uint() as u32,
                    kernarg: k.get(".kernarg_segment_size").uint() as u32,
                    block: k.get(".max_flat_workgroup_size").uint() as u32,
                    wave: k.get(".wavefront_size").uint() as u32,
                    args: k
                        .get(".args")
                        .arr()
                        .iter()
                        .map(|a| {
                            (
                                a.get(".name").str().to_owned(),
                                a.get(".offset").uint(),
                                a.get(".size").uint(),
                                a.get(".value_kind").str().to_owned(),
                            )
                        })
                        .collect(),
                };
                assert_eq!(
                    kernel_descriptor_words(elf, &name),
                    [abi.group, abi.private, abi.kernarg],
                    "{name}: .kd words vs metadata"
                );
                (name, abi)
            })
            .collect()
    }

    /// K1 (`mq_i4_moe_gemv`, `e8p_i4_moe_gemv`) kernarg: seven pointers, eight u32 words, then the trailing
    /// `n_experts@88` and the zero `reserved@92`: 96 bytes.
    const K1_KERNARG: [(&str, u64, u64, &str); 17] = [
        ("expert_weight_ptrs", 0, 8, "global_buffer"),
        ("expert_ids", 8, 8, "global_buffer"),
        ("expert_offsets", 16, 8, "global_buffer"),
        ("slot_ids", 24, 8, "global_buffer"),
        ("Xrot", 32, 8, "global_buffer"),
        ("Y0", 40, 8, "global_buffer"),
        ("Y1", 48, 8, "global_buffer"),
        ("M", 56, 4, "by_value"),
        ("K", 60, 4, "by_value"),
        ("U", 64, 4, "by_value"),
        ("C", 68, 4, "by_value"),
        ("x_row_div", 72, 4, "by_value"),
        ("x_src_rows", 76, 4, "by_value"),
        ("x_row_stride", 80, 4, "by_value"),
        ("y_slot_stride", 84, 4, "by_value"),
        ("n_experts", 88, 4, "by_value"),
        ("reserved", 92, 4, "by_value"),
    ];

    /// NT1 K2 (`qwen4_moe_iu4_sym*`) kernarg: five pointers, five u32 words, then the trailing `n_experts@60`:
    /// 64 bytes.
    const K2_KERNARG: [(&str, u64, u64, &str); 11] = [
        ("expert_weight_ptrs", 0, 8, "global_buffer"),
        ("expert_tile_ids", 8, 8, "global_buffer"),
        ("sorted_slot_index", 16, 8, "global_buffer"),
        ("Xq", 24, 8, "global_buffer"),
        ("Y_grouped", 32, 8, "global_buffer"),
        ("M", 40, 4, "by_value"),
        ("K", 44, 4, "by_value"),
        ("x_row_div", 48, 4, "by_value"),
        ("m_total", 52, 4, "by_value"),
        ("x_src_rows", 56, 4, "by_value"),
        ("n_experts", 60, 4, "by_value"),
    ];

    /// Asserts every kernel of `image` (served symbols and, where present, qualification twins) has exactly the
    /// `expected` kernarg layout, `bytes` long, a zero private segment and `.kd` words agreeing with the metadata
    /// (checked by [`image_kernel_abis`]). Returns the kernels for the caller's symbol, count and LDS checks.
    fn assert_kernarg_abi(
        label: &str,
        image: &[u8],
        bytes: u32,
        expected: &[(&str, u64, u64, &str)],
    ) -> Vec<(String, KernelAbi)> {
        let abis = image_kernel_abis(image);
        assert!(!abis.is_empty(), "{label}: no kernels");
        for (name, abi) in &abis {
            assert_eq!(abi.kernarg, bytes, "{label}/{name}: kernarg_segment_size");
            assert_eq!(abi.private, 0, "{label}/{name}: private segment");
            let got: Vec<_> = abi
                .args
                .iter()
                .map(|(n, o, s, k)| (n.as_str(), *o, *s, k.as_str()))
                .collect();
            assert_eq!(got, expected, "{label}/{name}: kernarg layout");
        }
        abis
    }

    fn sorted(names: impl IntoIterator<Item = String>) -> Vec<String> {
        let mut names: Vec<_> = names.into_iter().collect();
        names.sort_unstable();
        names
    }

    fn kernel_names(abis: &[(String, KernelAbi)]) -> Vec<String> {
        sorted(abis.iter().map(|(name, _)| name.clone()))
    }

    const ROWTILE_X4_KERNARG: [(&str, u64, u64, &str); 15] = [
        ("A0", 0, 8, "global_buffer"),
        ("A1", 8, 8, "global_buffer"),
        ("A2", 16, 8, "global_buffer"),
        ("A3", 24, 8, "global_buffer"),
        ("x", 32, 8, "global_buffer"),
        ("y0", 40, 8, "global_buffer"),
        ("y1", 48, 8, "global_buffer"),
        ("y2", 56, 8, "global_buffer"),
        ("y3", 64, 8, "global_buffer"),
        ("M0", 72, 4, "by_value"),
        ("M1", 76, 4, "by_value"),
        ("M2", 80, 4, "by_value"),
        ("M3", 84, 4, "by_value"),
        ("K", 88, 4, "by_value"),
        ("R", 92, 4, "by_value"),
    ];
    const ROWTILE_SINGLE_KERNARG: [(&str, u64, u64, &str); 6] = [
        ("A", 0, 8, "global_buffer"),
        ("x", 8, 8, "global_buffer"),
        ("y", 16, 8, "global_buffer"),
        ("M", 24, 4, "by_value"),
        ("K", 28, 4, "by_value"),
        ("R", 32, 4, "by_value"),
    ];
    const ROWTILE_HC_KERNARG: [(&str, u64, u64, &str); 8] = [
        ("W", 0, 8, "global_buffer"),
        ("X", 8, 8, "global_buffer"),
        ("Y", 16, 8, "global_buffer"),
        ("M", 24, 4, "by_value"),
        ("K", 28, 4, "by_value"),
        ("scale", 32, 4, "by_value"),
        ("hc_act", 36, 4, "by_value"),
        ("R", 40, 4, "by_value"),
    ];

    fn assert_rowtile_abi(
        image: &[u8],
        symbol: &str,
        bytes: u32,
        expected: &[(&str, u64, u64, &str)],
        block: u32,
        lds: u32,
    ) {
        let abis = image_kernel_abis(image);
        let (_, abi) = abis
            .iter()
            .find(|(name, _)| name == symbol)
            .unwrap_or_else(|| panic!("{symbol}: missing PM export"));
        assert_eq!(
            (abi.kernarg, abi.private, abi.block, abi.wave, abi.group),
            (bytes, 0, block, 32, lds),
            "{symbol}: ABI/launch geometry"
        );
        let got: Vec<_> = abi
            .args
            .iter()
            .map(|(n, o, s, k)| (n.as_str(), *o, *s, k.as_str()))
            .collect();
        assert_eq!(got, expected, "{symbol}: exact kernarg layout");
    }

    #[test]
    fn rowtile_projection_images_pin_abi_exports_and_physical_geometry() {
        for (arch, mq6, mq6_symbols, bf16, bf16_symbols, mq4) in [
            (
                "gfx1151",
                kernels::QWEN4_LANE_ROWS_PM_GFX1151,
                &crate::lane_rows_pm::MQ6_WIDE_SYMBOLS,
                kernels::BF16_ROWS_WIDE_PM_GFX1151,
                &crate::gemv::BF16_ROWS_WIDE_PM_SYMBOLS,
                kernels::MQ4_ROWS_X4_PM_GFX1151,
            ),
            (
                "gfx1201",
                kernels::LANE_ROWS_MQ6_PM_GFX1201,
                &MQ6_GFX1201_ROWTILE_SYMBOLS,
                kernels::LANE_ROWS_BF16_PM_GFX1201,
                &BF16_GFX1201_ROWTILE_SYMBOLS,
                kernels::MQ4_ROWS_X4_PM_GFX1201,
            ),
        ] {
            // The runtime count's metadata name is historically C on gfx1201.
            let mut single = ROWTILE_SINGLE_KERNARG;
            let mut x4 = ROWTILE_X4_KERNARG;
            let mut hc = ROWTILE_HC_KERNARG;
            if arch == "gfx1201" {
                single[5].0 = "C";
                x4[14].0 = "C";
                hc[7].0 = "C";
            }
            assert_rowtile_abi(mq6, mq6_symbols[0], 36, &single, 256, 8192);
            assert_rowtile_abi(mq6, mq6_symbols[1], 96, &x4, 256, 8192);
            assert_rowtile_abi(bf16, bf16_symbols[0], 96, &x4, 32, 0);
            assert_rowtile_abi(bf16, bf16_symbols[1], 48, &hc, 128, 128);
            let (_, _, _, symbols) = wide_twin_mq4_image(arch).unwrap();
            assert_eq!(symbols.len(), 1);
            assert_rowtile_abi(mq4, symbols[0], 96, &ROWTILE_X4_KERNARG, 32, 0);
            let mq4_names = kernel_names(&image_kernel_abis(mq4));
            assert_eq!(
                mq4_names,
                sorted([
                    format!("gemv_mq4g256v2_rows_x4_pm_{arch}"),
                    format!("gemv_mq4g256v2_rows_x4_t8_pm_{arch}"),
                ])
            );
            for image in [mq6, bf16, mq4] {
                let meta = amdgpu_metadata(amdgcn_elf(image));
                assert!(meta.get("amdhsa.target").str().ends_with(arch));
                assert!(image_kernel_abis(image)
                    .iter()
                    .all(|(name, _)| !name.contains("wide_r16") && !name.contains("wide_r32")));
            }
        }
        // Narrow PM entries remain available alongside Halo's fixed tiles.
        let halo = kernel_names(&image_kernel_abis(kernels::QWEN4_LANE_ROWS_PM_GFX1151));
        for narrow in [
            "gemv_mq6g256v2_xbatch_pm",
            "gemv_mq6g256v2_x4_xbatch_pm",
            "gemv_q8_0_k8_xbatch_pm",
            "gemv_q8_0_k320_xbatch_pm",
        ] {
            assert!(
                halo.iter().any(|name| name == narrow),
                "{narrow}: narrow export lost"
            );
        }
    }

    #[test]
    fn legacy_mq4_rowtile_images_pin_each_projection_abi() {
        let gu = [
            ("expert_ptrs", 0, 8, "global_buffer"),
            ("topk_indices", 8, 8, "global_buffer"),
            ("x", 16, 8, "global_buffer"),
            ("y_gate", 24, 8, "global_buffer"),
            ("y_up", 32, 8, "global_buffer"),
            ("M", 40, 4, "by_value"),
            ("K_TOP", 44, 4, "by_value"),
            ("n_experts", 48, 4, "by_value"),
            ("R", 52, 4, "by_value"),
        ];
        let down = [
            ("expert_ptrs", 0, 8, "global_buffer"),
            ("topk_indices", 8, 8, "global_buffer"),
            ("hidden_batch", 16, 8, "global_buffer"),
            ("expert_outputs", 24, 8, "global_buffer"),
            ("M", 32, 4, "by_value"),
            ("K", 36, 4, "by_value"),
            ("N", 40, 4, "by_value"),
            ("n_exp", 44, 4, "by_value"),
            ("k_top", 48, 4, "by_value"),
        ];
        for arch in ["gfx1151", "gfx1201"] {
            let (_, _, image, _, symbols) = legacy_mq4_rowtile_image(arch).unwrap();
            assert_eq!(symbols.len(), 2);
            assert_rowtile_abi(image, symbols[0], 56, &gu, 256, 10240);
            assert_rowtile_abi(image, symbols[1], 52, &down, 32, 0);
            assert_eq!(
                kernel_names(&image_kernel_abis(image)),
                sorted(symbols.iter().map(|symbol| symbol.to_string()))
            );
            assert!(amdgpu_metadata(amdgcn_elf(image))
                .get("amdhsa.target")
                .str()
                .ends_with(arch));
        }
    }

    #[test]
    fn gfx1100_mq4_rowtile_image_is_the_scalar_only_recipe() {
        let (_, module, image, symbols) = wide_twin_mq4_image("gfx1100").unwrap();
        assert_eq!(module, "mq4_rows_x4_pm_gfx1100");
        assert_eq!(symbols, ["gemv_mq4g256v2_rows_x4_t8_pm_gfx1100"]);
        assert_eq!(
            kernel_names(&image_kernel_abis(image)),
            symbols
                .iter()
                .map(|symbol| symbol.to_string())
                .collect::<Vec<_>>()
        );
        assert_rowtile_abi(image, symbols[0], 96, &ROWTILE_X4_KERNARG, 32, 0);
        assert!(amdgpu_metadata(amdgcn_elf(image))
            .get("amdhsa.target")
            .str()
            .ends_with("gfx1100"));
    }

    #[test]
    fn gfx1100_mq6_and_bf16_rowtile_images_pin_their_actual_abis() {
        let mq6 = kernels::QWEN4_LANE_ROWS_PM_GFX1100;
        assert_rowtile_abi(
            mq6,
            MQ6_GFX1100_ROWTILE_SYMBOLS[0],
            36,
            &ROWTILE_SINGLE_KERNARG,
            256,
            8192,
        );
        assert_rowtile_abi(
            mq6,
            MQ6_GFX1100_ROWTILE_SYMBOLS[1],
            96,
            &ROWTILE_X4_KERNARG,
            256,
            8192,
        );
        let bf16 = kernels::BF16_ROWS_WIDE_PM_GFX1100;
        assert_rowtile_abi(
            bf16,
            BF16_GFX1100_ROWTILE_SYMBOLS[0],
            96,
            &ROWTILE_X4_KERNARG,
            32,
            0,
        );
        assert_rowtile_abi(
            bf16,
            BF16_GFX1100_ROWTILE_SYMBOLS[1],
            48,
            &ROWTILE_HC_KERNARG,
            128,
            128,
        );
        assert_eq!(
            kernel_names(&image_kernel_abis(bf16)),
            sorted(BF16_GFX1100_ROWTILE_SYMBOLS.into_iter().map(str::to_owned))
        );
        for image in [mq6, bf16] {
            assert!(amdgpu_metadata(amdgcn_elf(image))
                .get("amdhsa.target")
                .str()
                .ends_with("gfx1100"));
            assert!(image_kernel_abis(image)
                .iter()
                .all(|(name, _)| !name.contains("wide_r16") && !name.contains("wide_r32")));
        }
        assert_ne!(mq6, kernels::QWEN4_LANE_ROWS_PM_GFX1151);
        assert_ne!(bf16, kernels::BF16_ROWS_WIDE_PM_GFX1151);
    }

    #[test]
    fn q8_rowtile_images_preserve_all_five_recipe_abis() {
        let halo_symbols = [
            crate::lane_rows_pm::Q8_K8_ROWTILE,
            crate::lane_rows_pm::Q8_K320_ROWTILE,
            crate::lane_rows_pm::Q8_K2560_ROWTILE,
            crate::lane_rows_pm::Q8_K2560_X2_ROWTILE,
            crate::lane_rows_pm::Q8_SMALL_ROWTILE,
        ];
        assert_eq!(crate::lane_rows_pm::Q8_WIDE_SYMBOLS.len(), 5);
        let mut k8 = ROWTILE_SINGLE_KERNARG.to_vec();
        k8.extend([("scale", 36, 4, "by_value"), ("hc_act", 40, 4, "by_value")]);
        let x2 = [
            ("A0", 0, 8, "global_buffer"),
            ("A1", 8, 8, "global_buffer"),
            ("x", 16, 8, "global_buffer"),
            ("y0", 24, 8, "global_buffer"),
            ("y1", 32, 8, "global_buffer"),
            ("M0", 40, 4, "by_value"),
            ("M1", 44, 4, "by_value"),
            ("K", 48, 4, "by_value"),
            ("R", 52, 4, "by_value"),
        ];
        for (arch, image, symbols) in [
            ("gfx1151", kernels::QWEN4_LANE_ROWS_PM_GFX1151, halo_symbols),
            (
                "gfx1100",
                kernels::QWEN4_LANE_ROWS_PM_GFX1100,
                Q8_GFX1100_ROWTILE_SYMBOLS,
            ),
        ] {
            assert_rowtile_abi(image, symbols[0], 44, &k8, 512, 16384);
            for symbol in [symbols[1], symbols[2], symbols[4]] {
                assert_rowtile_abi(image, symbol, 36, &ROWTILE_SINGLE_KERNARG, 256, 8192);
            }
            assert_rowtile_abi(image, symbols[3], 56, &x2, 256, 8192);
            if arch == "gfx1100" {
                // The supplemental image contains only genuine tile exports.
                assert_eq!(
                    kernel_names(&image_kernel_abis(image)),
                    sorted(
                        MQ6_GFX1100_ROWTILE_SYMBOLS
                            .into_iter()
                            .chain(Q8_GFX1100_ROWTILE_SYMBOLS)
                            .map(str::to_owned)
                    )
                );
            }
        }
    }

    #[test]
    fn rowtile_planner_images_pin_logical36_segment40_and_geometry() {
        let args = [
            ("topk", 0, 8, "global_buffer"),
            ("codec", 8, 8, "global_buffer"),
            ("out", 16, 8, "global_buffer"),
            ("rows", 24, 4, "by_value"),
            ("uniform_bucket", 28, 4, "by_value"),
            ("n_experts", 32, 4, "by_value"),
        ];
        let mut images = Vec::new();
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            let PlannedKernel::Embedded {
                module,
                image,
                symbols,
                ..
            } = rowtile_planner_entry(arch).unwrap()
            else {
                panic!("HIP planner")
            };
            assert_eq!(symbols, [module]);
            assert_eq!(module, format!("qwen4_slab_union_plan_t8_pm_{arch}"));
            let abis = assert_kernarg_abi(module, image, 40, &args);
            assert_eq!(kernel_names(&abis), [module.to_owned()]);
            assert_rowtile_abi(image, module, 40, &args, 128, 1280);
            let last = abis[0].1.args.last().unwrap();
            assert_eq!(last.1 + last.2, 36, "{arch}: logical kernarg size");
            assert!(amdgpu_metadata(amdgcn_elf(image))
                .get("amdhsa.target")
                .str()
                .ends_with(arch));
            assert_eq!(walked_embedded_image(arch, module, "").unwrap().1, image);
            assert!(
                images.iter().all(|prior| *prior != image),
                "retagged planner on {arch}"
            );
            images.push(image);
        }
        for arch in ["gfx906", "gfx942"] {
            assert!(rowtile_planner_entry(arch).is_none());
        }
    }

    #[test]
    fn rowtile_images_match_bare_sha256_sidecars() {
        for (image, expected) in [
            (
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1151,
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1151_SHA256,
            ),
            (
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1201,
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1201_SHA256,
            ),
            (
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1100,
                kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1100_SHA256,
            ),
            (
                kernels::MQ4_ROWS_X4_PM_GFX1151,
                kernels::MQ4_ROWS_X4_PM_GFX1151_SHA256,
            ),
            (
                kernels::MQ4_ROWS_X4_PM_GFX1201,
                kernels::MQ4_ROWS_X4_PM_GFX1201_SHA256,
            ),
            (
                kernels::MQ4_ROWS_X4_PM_GFX1100,
                kernels::MQ4_ROWS_X4_PM_GFX1100_SHA256,
            ),
            (
                kernels::QWEN4_LANE_ROWS_PM_GFX1151,
                kernels::QWEN4_LANE_ROWS_PM_GFX1151_SHA256,
            ),
            (
                kernels::BF16_ROWS_WIDE_PM_GFX1151,
                kernels::QWEN4_BF16_ROWS_WIDE_PM_GFX1151_SHA256,
            ),
            (
                kernels::QWEN4_LANE_ROWS_PM_GFX1100,
                kernels::QWEN4_LANE_ROWS_PM_GFX1100_SHA256,
            ),
            (
                kernels::BF16_ROWS_WIDE_PM_GFX1100,
                kernels::BF16_ROWS_WIDE_PM_GFX1100_SHA256,
            ),
            (
                kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
                kernels::MQ_I4_MOE_GEMV_PM_GFX1201_SHA256,
            ),
            (
                kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
                kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151_SHA256,
            ),
            (
                kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
                kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100_SHA256,
            ),
            (
                kernels::E8P_I4_MOE_GEMV_PM_GFX1100,
                kernels::E8P_I4_MOE_GEMV_PM_GFX1100_SHA256,
            ),
            (
                kernels::E8P_I4_MOE_GEMV_PM_GFX1151,
                kernels::E8P_I4_MOE_GEMV_PM_GFX1151_SHA256,
            ),
            (
                kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
                kernels::E8P_I4_MOE_GEMV_PM_GFX1201_SHA256,
            ),
            (
                kernels::LANE_ROWS_MQ6_PM_GFX1201,
                kernels::LANE_ROWS_MQ6_PM_GFX1201_SHA256,
            ),
            (
                kernels::LANE_ROWS_BF16_PM_GFX1201,
                kernels::LANE_ROWS_BF16_PM_GFX1201_SHA256,
            ),
            (
                kernels::LEGACY_MQ4_ROWTILES_PM_GFX1151,
                kernels::LEGACY_MQ4_ROWTILES_PM_GFX1151_SHA256,
            ),
            (
                kernels::LEGACY_MQ4_ROWTILES_PM_GFX1201,
                kernels::LEGACY_MQ4_ROWTILES_PM_GFX1201_SHA256,
            ),
        ] {
            assert_eq!(expected.len(), 64);
            assert!(expected
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
            assert_eq!(format!("{:x}", Sha256::digest(image)), expected);
        }
        assert_ne!(
            kernels::LANE_ROWS_MQ6_PM_GFX1201,
            kernels::QWEN4_LANE_ROWS_PM_GFX1151
        );
        assert_ne!(
            kernels::LANE_ROWS_BF16_PM_GFX1201,
            kernels::BF16_ROWS_WIDE_PM_GFX1151
        );
    }

    #[test]
    fn rowtile_route_prewarm_is_pm_only_closed_and_idempotent() {
        for (arch, expected_modules, expected_symbols) in
            [("gfx1151", 5, 13), ("gfx1201", 5, 8), ("gfx1100", 4, 11)]
        {
            let mut entries = Vec::new();
            add_rowtile_entries(arch, &mut entries, |_| true);
            add_rowtile_entries(arch, &mut entries, |_| true);
            let plan = RouteKernelPlan { entries };
            validate_route_plan(&plan).unwrap();
            assert_eq!(plan.entries.len(), expected_modules, "{arch}");
            assert_eq!(
                plan.entries
                    .iter()
                    .map(|entry| entry.symbols().len())
                    .sum::<usize>(),
                expected_symbols,
                "{arch}: all fixed-t8 exports"
            );
            for entry in &plan.entries {
                let PlannedKernel::Embedded {
                    module,
                    image,
                    symbols,
                    ..
                } = entry
                else {
                    panic!("{arch}: obsolete HIP wide registration")
                };
                assert_eq!(walked_embedded_image(arch, module, "").unwrap().1, *image);
                let exports = kernel_names(&image_kernel_abis(image));
                for symbol in *symbols {
                    assert!(symbol.contains("t8"), "{symbol}: non-tiled wide export");
                    assert!(
                        exports.iter().any(|name| name == symbol),
                        "{symbol}: missing export"
                    );
                }
            }
            let mut disabled = Vec::new();
            add_rowtile_entries(arch, &mut disabled, |_| false);
            assert!(disabled.is_empty());
        }
        for arch in ["gfx1151", "gfx1201"] {
            let inventory = entries(arch, "").unwrap();
            assert!(inventory
                .iter()
                .flat_map(|entry| entry.symbols)
                .all(|symbol| !symbol.contains("wide_r16") && !symbol.contains("wide_r32")));
            let bf16 = lookup(arch, "qwen4_gemv_bf16_xf32", "").unwrap();
            assert!(bf16.symbols.contains(&"gemv_bf16_xf32_x4"));
            assert!(bf16.symbols.contains(&"gemv_bf16_xf32_x4_rows_r8"));
            let q8 = lookup(arch, "qwen4_gemv_q8_0_wide", "").unwrap();
            assert!(q8.symbols.contains(&"gemv_q8_0_wide_k640_staged"));
            assert!(q8.symbols.contains(&"gemv_q8_0_wide_rows"));
        }
    }

    #[test]
    fn rowtile_symbol_merge_preserves_narrow_exports() {
        static MERGED: std::sync::OnceLock<Vec<&str>> = std::sync::OnceLock::new();
        let mut planned = vec![PlannedKernel::Embedded {
            module: "mq4_rows_x4_pm_gfx1201",
            image: kernels::MQ4_ROWS_X4_PM_GFX1201,
            radiowave_json: None,
            symbols: &["gemv_mq4g256v2_rows_x4_pm_gfx1201"],
        }];
        let (_, module, image, tiles) = wide_twin_mq4_image("gfx1201").unwrap();
        plan_wide_embedded(&mut planned, &MERGED, module, image, tiles);
        plan_wide_embedded(&mut planned, &MERGED, module, image, tiles);
        assert_eq!(planned.len(), 1);
        assert_eq!(
            planned[0].symbols(),
            [
                "gemv_mq4g256v2_rows_x4_pm_gfx1201",
                "gemv_mq4g256v2_rows_x4_t8_pm_gfx1201",
            ]
        );
        validate_route_plan(&RouteKernelPlan { entries: planned }).unwrap();
    }

    #[test]
    fn rowtile_family_prewarm_keeps_arch_and_family_boundaries() {
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            let mut mq4 = Vec::new();
            add_rowtile_entries(arch, &mut mq4, |family| {
                matches!(family, WideTwinFamily::Mq4)
            });
            assert_eq!(mq4.len(), if arch == "gfx1100" { 1 } else { 2 });
            if let Some((_, module, image, _, symbols)) = legacy_mq4_rowtile_image(arch) {
                let legacy = mq4.iter().find(|entry| entry.module() == module).unwrap();
                assert_eq!(legacy.symbols(), symbols);
                assert_eq!(walked_embedded_image(arch, module, "").unwrap().1, image);
            } else {
                assert!(mq4
                    .iter()
                    .all(|entry| !entry.module().starts_with("legacy_mq4")));
            }
            let mut mq6 = Vec::new();
            add_rowtile_entries(arch, &mut mq6, |family| {
                matches!(family, WideTwinFamily::Mq6)
            });
            assert_eq!(mq6.len(), 1);
            assert_eq!(mq6[0].symbols().len(), 2);
            assert!(mq6[0]
                .symbols()
                .iter()
                .all(|symbol| symbol.starts_with("gemv_mq6")));
            let mut q8 = Vec::new();
            add_rowtile_entries(arch, &mut q8, |family| matches!(family, WideTwinFamily::Q8));
            if arch == "gfx1151" {
                assert_eq!(q8.len(), 1);
                assert_eq!(q8[0].symbols(), crate::lane_rows_pm::Q8_WIDE_SYMBOLS);
            } else if arch == "gfx1100" {
                assert_eq!(q8.len(), 1);
                assert_eq!(q8[0].symbols(), Q8_GFX1100_ROWTILE_SYMBOLS);
            } else {
                // gfx1201 has no admitted Q8 PM object; never revive HIP twins.
                assert!(q8.is_empty());
            }
        }
        for arch in ["gfx906", "gfx942"] {
            let mut entries = Vec::new();
            add_rowtile_entries(arch, &mut entries, |_| true);
            assert!(entries.is_empty());
        }
    }

    #[test]
    fn slab_k1_images_carry_the_96_byte_n_experts_kernarg() {
        use crate::slab_moe::{
            slab_arch_kernels, slab_cold_k1_metadata, slab_rowtile_k1,
            MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS, MQ_I4_MOE_GEMV_CN_SYMBOLS,
            MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS, MQ_I4_MOE_GEMV_SYMBOLS,
        };
        let narrow = sorted(
            MQ_I4_MOE_GEMV_SYMBOLS
                .iter()
                .flatten()
                .map(|s| s.to_string()),
        );
        for (arch, image) in [
            ("gfx1201", kernels::MQ_I4_MOE_GEMV_PM_GFX1201),
            ("gfx1151", kernels::MQ_I4_MOE_GEMV_PM_GFX1151),
        ] {
            assert_eq!(slab_arch_kernels(arch).unwrap().mq4_k1.image, image);
            let abis = assert_kernarg_abi(arch, image, 96, &K1_KERNARG);
            let mut expected = narrow.clone();
            if arch == "gfx1201" {
                expected.extend(
                    MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS
                        .iter()
                        .flatten()
                        .map(|s| s.to_string()),
                );
                for (bucket, symbols) in MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS.iter().enumerate() {
                    for (kind, symbol) in symbols.iter().enumerate() {
                        let lds = if bucket == 1 && kind == 0 { 8192 } else { 4096 };
                        assert_rowtile_abi(image, symbol, 96, &K1_KERNARG, 256, lds);
                    }
                }
                assert!(abis
                    .iter()
                    .all(|(n, _)| !n.contains("_c16") && !n.contains("_c32")));
            } else {
                // The unchanged Halo reference image keeps genuine legacy
                // diagnostics, but production rowtiles live only in Cn.
                expected.extend(
                    narrow
                        .iter()
                        .flat_map(|name| [format!("{name}_c16"), format!("{name}_c32")]),
                );
                assert_eq!(expected.len(), 12);
                assert!(abis.iter().all(|(name, _)| !name.ends_with("_t8")));
            }
            assert_eq!(kernel_names(&abis), sorted(expected), "{arch}: K1 exports");
        }
        let cn = assert_kernarg_abi(
            "gfx1151 Cn",
            kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
            96,
            &K1_KERNARG,
        );
        assert_eq!(
            kernel_names(&cn),
            sorted(
                MQ_I4_MOE_GEMV_CN_SYMBOLS
                    .iter()
                    .flatten()
                    .chain(MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS.iter().flatten())
                    .map(|s| s.to_string())
            ),
        );
        for symbols in MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS {
            for (kind, symbol) in symbols.into_iter().enumerate() {
                assert_rowtile_abi(
                    kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
                    symbol,
                    96,
                    &K1_KERNARG,
                    256,
                    if kind == 0 { 40960 } else { 20480 },
                );
            }
        }
        let hot = &slab_arch_kernels("gfx1100").unwrap().mq4_k1;
        let tiled = slab_rowtile_k1(hot).unwrap();
        for (narrow, symbol) in hot.symbols.into_iter().zip(tiled.symbols) {
            assert_eq!(symbol, format!("{narrow}_t8"));
        }
        let abis = assert_kernarg_abi("gfx1100 MQ4 K1", hot.image, 96, &K1_KERNARG);
        assert_eq!(
            kernel_names(&abis),
            sorted(
                hot.symbols
                    .into_iter()
                    .chain(tiled.symbols)
                    .map(str::to_owned)
            ),
        );
        assert!(abis
            .iter()
            .all(|(n, abi)| abi.group == 0 && !n.ends_with("_acc") && !n.contains("mq2i4")));
        for symbol in tiled.symbols {
            assert_rowtile_abi(tiled.image, symbol, 96, &K1_KERNARG, 256, 0);
        }
        // Inventory includes all arch-specific E8P images, without changing admission.
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let e8p = slab_cold_k1_metadata(arch, 3).unwrap();
            let tiled = slab_rowtile_k1(&e8p.k1).unwrap();
            assert_eq!((tiled.module, tiled.image), (e8p.k1.module, e8p.k1.image));
            let abis = assert_kernarg_abi(arch, e8p.k1.image, 96, &K1_KERNARG);
            assert_eq!(
                kernel_names(&abis),
                sorted(
                    e8p.k1
                        .symbols
                        .into_iter()
                        .chain(tiled.symbols)
                        .map(str::to_owned)
                ),
            );
            assert!(abis
                .iter()
                .all(|(n, abi)| abi.group == 1536 && !n.ends_with("_acc")));
            for (narrow, symbol) in e8p.k1.symbols.into_iter().zip(tiled.symbols) {
                assert_eq!(symbol, format!("{narrow}_t8"));
                assert_rowtile_abi(tiled.image, symbol, 96, &K1_KERNARG, 256, 1536);
            }
        }
    }

    #[test]
    fn slab_k2_images_carry_the_64_byte_n_experts_kernarg() {
        use crate::gemm::slab_k2_entry;
        // Beta's shared K2 images keep their kernel counts (served entries, NT4/NT8 widths, row repeats and the
        // qualification twins) at the 64-byte ABI.
        for (label, image, count) in [
            ("gfx1151 MQ4 K2", kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151, 6),
            (
                "gfx1151 streamed GU K2",
                kernels::QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151,
                1,
            ),
            ("gfx1201 MQ4 K2", kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201, 6),
            ("gfx1201 W3 K2", kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1201, 4),
            (
                "gfx1151 W3 NT1 K2",
                kernels::QWEN4_MOE_IU4_SYM_W3_PM_GFX1151,
                4,
            ),
            (
                "gfx1151 W3 NT4 K2",
                kernels::QWEN4_MOE_IU4_SYM_W3_NT4_PM_GFX1151,
                6,
            ),
        ] {
            assert_eq!(
                assert_kernarg_abi(label, image, 64, &K2_KERNARG).len(),
                count,
                "{label}: kernel count"
            );
        }
        // The sealed-slab NT1 K2 tables (gfx1100 MQ4, gfx1100 E8P, gfx1201 E8P): the two served entries plus
        // their `_acc` qualification twins in the image, every one at 64 bytes, static LDS 0 (MQ4) / 1,536 (E8P).
        for (arch, codec, lds) in [
            ("gfx1100", 0u8, 0u32),
            ("gfx1100", 3, 1536),
            ("gfx1201", 3, 1536),
        ] {
            let k2 = slab_k2_entry(arch, codec).unwrap();
            let abis = assert_kernarg_abi(
                &format!("{arch} codec {codec} K2"),
                k2.image,
                64,
                &K2_KERNARG,
            );
            let expected = sorted(
                k2.entries
                    .iter()
                    .flat_map(|e| [e.to_string(), format!("{e}_acc")]),
            );
            assert_eq!(
                kernel_names(&abis),
                expected,
                "{arch} codec {codec}: K2 symbols"
            );
            assert!(
                abis.iter().all(|(_, abi)| abi.group == lds),
                "{arch} codec {codec}: K2 static LDS"
            );
            assert!(
                k2.entries.iter().all(|e| !e.ends_with("_acc")),
                "{arch} codec {codec}: a twin is served"
            );
        }
    }

    /// The E8P slab image matrix: one K1 module and one K2 module per E8P arch (gfx1100, gfx1151,
    /// gfx1201), each with its own image and symbols, conflict-free as a route plan, and the
    /// serving selection of every (arch, codec) pair equals table presence AND the shared
    /// admission matrix (registry presence is not permission).
    #[test]
    fn slab_e8p_image_matrix_is_arch_keyed_conflict_free_and_admission_gated() {
        let embedded =
            |module: &'static str, image: &'static [u8], symbols: &'static [&'static str]| {
                PlannedKernel::Embedded {
                    module,
                    image,
                    radiowave_json: None,
                    symbols,
                }
            };
        let contains = |haystack: &[u8], needle: &str| {
            haystack
                .windows(needle.len())
                .any(|w| w == needle.as_bytes())
        };
        // (arch, K1 module, K1 [gate/up, down], K2 module, K2 [gate/up, down])
        let rows: [(&str, &str, [&str; 2], &str, [&str; 2]); 3] = [
            (
                "gfx1100",
                "e8p_i4_moe_gemv_pm_gfx1100",
                [
                    "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
                    "e8p_i4_moe_gemv_r128_down_pm_gfx1100",
                ],
                "qwen4_moe_iu4_sym_e8p_pm_gfx1100",
                [
                    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1100",
                    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1100",
                ],
            ),
            (
                "gfx1151",
                "e8p_i4_moe_gemv_pm_gfx1151",
                [
                    "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151",
                    "e8p_i4_moe_gemv_r128_down_pm_gfx1151",
                ],
                "qwen4_moe_iu4_sym_e8p_pm_gfx1151",
                [
                    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1151",
                    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1151",
                ],
            ),
            (
                "gfx1201",
                "e8p_i4_moe_gemv_pm_gfx1201",
                ["e8p_i4_moe_gemv_r256_gate_up", "e8p_i4_moe_gemv_r128_down"],
                "qwen4_moe_iu4_sym_e8p_pm_gfx1201",
                [
                    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201",
                    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1201",
                ],
            ),
        ];
        let mut images: Vec<(&str, &'static [u8])> = Vec::new();
        for (arch, k1_module, k1_symbols, k2_module, k2_entries) in rows {
            let k1 = crate::slab_moe::slab_cold_k1_metadata(arch, 3).unwrap();
            let k2 = crate::gemm::slab_k2_metadata(arch, 3).unwrap();
            assert_eq!(k1.cold_codec, 3, "{arch}");
            assert_eq!(
                (k1.k1.module, k1.k1.symbols),
                (k1_module, k1_symbols),
                "{arch} K1"
            );
            assert_eq!(
                (k2.module, k2.entries),
                (k2_module, k2_entries),
                "{arch} K2"
            );
            // The arch-suffixed spellings: every gfx1100/gfx1151 symbol carries its arch, K2 modules and
            // entries always do, the historical gfx1201 K1 symbols stay unsuffixed.
            for module in [k1_module, k2_module] {
                assert!(module.ends_with(&format!("_pm_{arch}")), "{module}");
            }
            let k1_checked: &[&str] = if arch == "gfx1201" { &[] } else { &k1_symbols };
            for symbol in k2_entries.into_iter().chain(k1_checked.iter().copied()) {
                assert!(symbol.ends_with(&format!("_pm_{arch}")), "{symbol}");
                assert!(
                    !symbol.ends_with("_acc"),
                    "{symbol}: qualification twin must not be served"
                );
            }
            let plan = RouteKernelPlan {
                entries: vec![
                    embedded(k1.k1.module, k1.k1.image, &k1.k1.symbols),
                    embedded(k2.module, k2.image, &k2.entries),
                ],
            };
            validate_route_plan(&plan).unwrap();
            for planned in &plan.entries {
                let PlannedKernel::Embedded { image, symbols, .. } = planned else {
                    panic!("hip entry")
                };
                for &symbol in *symbols {
                    assert!(
                        contains(*image, &format!("{symbol}.kd")),
                        "{arch} {symbol}.kd missing"
                    );
                }
            }
            images.push((arch, k1.k1.image));
            images.push((arch, k2.image));
        }
        // Six distinct images; the gfx1100 and gfx1151 pairs each name only their own arch.
        for (i, (arch_a, a)) in images.iter().enumerate() {
            for (arch_b, b) in &images[i + 1..] {
                assert_ne!(a, b, "{arch_a} and {arch_b} share an image");
            }
        }
        for ((arch, k1_module, k1_symbols, _, k2_entries), other) in
            [(rows[0], "gfx1151"), (rows[1], "gfx1100")]
        {
            for (other_arch, image) in &images {
                if *other_arch != other {
                    continue;
                }
                for symbol in k1_symbols.into_iter().chain(k2_entries) {
                    assert!(
                        !contains(*image, symbol),
                        "{arch} symbol {symbol} inside the {other_arch} image"
                    );
                }
                assert!(
                    !contains(*image, k1_module),
                    "{k1_module} inside the {other_arch} image"
                );
            }
        }
        // Matrix: serving = table presence AND admission, for the K1 cold table, the K2 table and the
        // registry inventory. Every admitted E8P arch has both its K1 and its K2 table.
        for &arch in SUPPORTED_ARCHES {
            for codec in 0u8..=4 {
                let admitted = crate::qwen4_slab_codec_admitted(arch, codec);
                let k1_meta = crate::slab_moe::slab_cold_k1_metadata(arch, codec).is_some();
                let k2_meta = crate::gemm::slab_k2_metadata(arch, codec).is_some();
                let k1_served = crate::slab_moe::slab_cold_kernels(arch, codec).is_some();
                let k2_served = crate::gemm::slab_k2_entry(arch, codec).is_some();
                assert_eq!(
                    k1_served,
                    k1_meta && codec != 0 && admitted,
                    "{arch} codec {codec} K1"
                );
                assert_eq!(k2_served, k2_meta && admitted, "{arch} codec {codec} K2");
                if codec == 3 {
                    assert_eq!(
                        k1_meta, k2_meta,
                        "{arch}: E8P K1 and K2 tables come as a pair"
                    );
                    if admitted {
                        assert!(
                            k1_meta && k2_meta,
                            "{arch}: admitted E8P without a K1/K2 table"
                        );
                    }
                }
                let plan = slab_embedded_entries(arch, codec);
                match arch {
                    "gfx1100" if codec == 0 || (k1_served && k2_served) => {
                        let modules: Vec<_> =
                            plan.unwrap().iter().map(PlannedKernel::module).collect();
                        assert_eq!(modules.contains(&"e8p_i4_moe_gemv_pm_gfx1100"), codec == 3);
                        assert_eq!(
                            modules.contains(&"qwen4_moe_iu4_sym_e8p_pm_gfx1100"),
                            codec == 3
                        );
                        assert!(
                            !modules.iter().any(|m| m.contains("gfx1151")),
                            "gfx1100 plan names gfx1151"
                        );
                    }
                    "gfx1100" => assert!(
                        matches!(plan, Err(RegistryError::UnsupportedRoute(_))),
                        "{arch} codec {codec}"
                    ),
                    // The other supported arches keep their historical (non-registry) preload: the
                    // registry lists nothing, whatever the table or admission says.
                    _ => assert!(plan.unwrap().is_empty(), "{arch} codec {codec}"),
                }
            }
        }
    }

    #[test]
    fn gfx1100_fa2_split_verifier_is_compiler_free_packaged() {
        let entry = lookup("gfx1100", "attention_q8_0_fa2_gqa_gfx1100", "").unwrap();
        for symbol in [
            "attention_fa2_q_preconvert_gfx1100",
            "attention_q8_0_fa2_gqa_gfx1100",
            "attention_q8_0_fa2_gqa_partial_gfx1100",
            "attention_q8_0_fa2_gqa_merge_gfx1100",
        ] {
            assert!(entry.symbols.contains(&symbol), "missing {symbol}");
        }
        for arch in ["gfx1151", "gfx1201"] {
            let symbols = entries(arch, "")
                .unwrap()
                .into_iter()
                .find(|candidate| candidate.module == "attention_q8_0_fa2_gqa_gfx1100")
                .map(|candidate| candidate.symbols);
            assert!(symbols.is_none(), "{arch} packaged gfx1100 split verifier");
        }
    }

    #[test]
    fn gfx11_residual_mmq_fallback_preserves_runtime_identity() {
        let module = "gemm_mq4g256v2_residual_mmq";
        for arch in ["gfx1100", "gfx1151"] {
            for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
                let entry = lookup(arch, module, extra_flags).unwrap();
                assert_eq!(entry.source(), kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_SRC);
                assert_eq!(
                    entry.symbols,
                    [
                        module,
                        "gemm_mq4g256v2_residual_mmq_full_add",
                        "gemm_mq4g256v2_residual_mmq_full_set",
                        "gemm_mq4g256v2_residual_mmq_x128",
                        "gemm_mq4g256v2_residual_mmq_full_add_x128",
                        "gemm_mq4g256v2_residual_mmq_full_set_x128",
                        "quantize_q8_1_mmq_ds4",
                        "quantize_q8_1_mmq_ds4_x128",
                    ]
                );
                let recipe =
                    KernelCompiler::recipe_for_source(arch, module, entry.source(), extra_flags);
                assert_eq!(entry.flags, recipe.flags);
                assert_eq!(entry.scheduler_profile, recipe.scheduler_profile);
                assert_eq!(entry.scheduler_profile.as_deref(), Some("default"));
            }
        }
        for arch in ["gfx1201", "gfx906", "gfx942"] {
            assert!(
                lookup(arch, module, "").is_err(),
                "{arch}: admitted RDNA3-only MMQ"
            );
        }
    }

    #[test]
    fn mq4v2_k5120_gfx1201_inventory_preserves_cache_policy() {
        let module = "fused_gate_up_mq4g256v2_k5120_gfx1201";
        let registry = entries("gfx1201", "").unwrap();
        let candidate = registry.iter().find(|e| e.module == module).unwrap();
        assert_eq!(candidate.symbols, [module]);
        let body = candidate
            .source()
            .strip_prefix(
                "#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_mq4g256v2_k5120_gfx1201\n",
            )
            .unwrap();
        assert_eq!(
            body.replace(
                "const int groups_per_row = 20;",
                "const int groups_per_row = K / 256;"
            ),
            kernels::FUSED_GATE_UP_MQ4G256V2_SRC
        );
        assert!(registry
            .iter()
            .any(|e| e.symbols.contains(&"fused_gate_up_mq4g256v2")));
        for arch in ["gfx1100", "gfx1151", "gfx906", "gfx942"] {
            assert!(!entries(arch, "")
                .unwrap()
                .iter()
                .any(|e| e.module == module));
        }
        for arch in ["gfx1101", "gfx1200"] {
            assert!(matches!(
                entries(arch, ""),
                Err(RegistryError::UnsupportedArch(_))
            ));
        }
    }

    #[test]
    fn kernel_registry_p0_inventory_membership() {
        // The gfx1201 runtime appends this default via FeatureFlags at init;
        // it appears in every P0 hipcc argv, even without an env override.
        let registry = entries("gfx1201", "-DIU4_A4_CANDIDATES=2").unwrap();
        let by_name: HashMap<_, _> = registry.iter().map(|e| (e.module, e)).collect();
        assert_eq!(by_name.len(), registry.len(), "duplicate module identities");
        // P0 provenance fixture: membership only. Its digests record the P0
        // capture's effective sources, not the current beta sources, so they
        // are parsed but never compared against the live registry.
        let expected = include_str!("../tests/fixtures/p0-gfx1201-sha256.tsv");
        let mut count = 0;
        for line in expected.lines().filter(|line| !line.starts_with('#')) {
            let (module, _digest) = line.split_once('\t').unwrap();
            assert!(by_name.contains_key(module), "missing P0 module {module}");
            count += 1;
        }
        assert_eq!(count, 92);
        // Modules added since the P0 capture: the scalar-prefill runtime's BR/BC
        // specialization, the `HIPFIRE_G12_DEC_NORM` decode twins, the five
        // multi-slot `*_paged` modules of the q8/asym3 routes, the H2 decode /
        // prefill / MTP / DFlash modules for compiler-free packs, the Qwen3.8
        // Flash-Next and Qwen3.5-MoE trace modules, the Qwen4 penalty prepass
        // `logit_penalty_table`, the dense 27B DFlash cold helpers, the gfx12
        // MQ4V2 verify-tile (`*_vt`) GEMM modules, the expert-parallel row
        // gathers, the 27B AR route's plain q8 attention and slab sigmoid
        // producer, plus the opt-in twins, decode twins, QSA modules, slab
        // planner, pruned router, Q8 wide module and DFlash-draft overwrite
        // twins. Those 158 are the pre-S1 inventory; the S1 slow-link kernels
        // add exactly the modules in `S1_ADDED_MODULES` (not P0 modules, not
        // in the provenance fixture), each its own isolated source.
        const PRE_S1_ADDED_SINCE_P0: usize = 158;
        const S1_ADDED_MODULES: [&str; 1] = ["slowlink_prefill"];
        for module in S1_ADDED_MODULES {
            assert!(by_name.contains_key(module), "missing S1 module {module}");
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The native A3B port is a new module, not a historical P0 or S1 row.
        // Keep the earlier inventory cohorts and their membership guards intact.
        // The exact-gfx1201 multi-workgroup MoE scatter is likewise its own module.
        const NATIVE_A3B_ADDED_MODULES: [&str; 2] = [
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201",
            "moe_scatter_stable_k8",
        ];
        for module in NATIVE_A3B_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing native A3B module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // FirstRequestJit adds 119 modules beyond the pre-S1 cohort:
        // tape/lane copies, the storage-boundary round trip and 116 discovery
        // and static-coverage modules. Keep the later beta cohorts above.
        const FIRST_REQUEST_JIT_ADDED_MODULES: usize = 119;
        // The exact-INT8 MQ6 prefill default adds its GEMM module, the A8
        // `block_i8_128` quantizer module and the one-pass rotate producer.
        const MQ6_I8_ADDED_MODULES: [&str; 3] = [
            "gemm_mq6g256v2_mmq_i8_gfx12",
            "gemm_mq4g256v2_residual_mmq_i8_gfx12",
            "mq_rotate_x_i8_gfx12",
        ];
        // The fused HC-down + activation kernel (`Gpu::gemm_bf16_xf16_hc_down_fused`).
        const HC_DOWN_FUSED_ADDED_MODULES: [&str; 1] = ["qwen4_hc_down_fused"];
        for module in HC_DOWN_FUSED_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing HC-down fused module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        for module in MQ6_I8_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing MQ6 i8 module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The Qwen4 shared-expert INT8 route (`HIPFIRE_QWEN4_SHARED_INT8`, `shared_i8.rs`).
        const SHARED_I8_ADDED_MODULES: [&str; 3] = [
            "qwen4_shared_i8_gemm_gfx12",
            "qwen4_shared_i8_quant_gfx12",
            "mq_rotate_x_i4_f16out_i8",
        ];
        for module in SHARED_I8_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing shared-expert INT8 module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // Later ParoQ4G64 and DFlash2 launchers add their own modules, not P0 rows.
        const G64_AND_HC_CAPTURE_ADDED_MODULES: [&str; 4] = [
            "gemm_hfq4g64",
            "gemv_hfq4g64",
            "givens_rotate_to_g64_f32",
            "hc_mean_capture",
        ];
        for module in G64_AND_HC_CAPTURE_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing G64/HC capture module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The gfx1201 HC row folds (`HIPFIRE_QWEN4_HC_ROW_FOLD`) live in `hc_row_fold`,
        // which gfx1201 lists with its three entries (the gfx1151 module has one).
        const HC_ROW_FOLD_ADDED_MODULES: [&str; 1] = ["hc_row_fold"];
        for module in HC_ROW_FOLD_ADDED_MODULES {
            let entry = by_name.get(module).unwrap_or_else(|| {
                panic!("missing gfx1201 HC row fold module {module}");
            });
            assert_eq!(
                entry.symbols,
                [
                    "hc_row_fold_norm_gate",
                    "hc_row_fold_moe_gfx1201",
                    "hc_row_write_norm_gate_gfx1201"
                ]
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The Qwen4 MoE front producer (`HIPFIRE_QWEN4_MOE_FRONT_FUSE`) adds the
        // `mq_rotate_x_i4` source's F16-output twin.
        const MOE_FRONT_ADDED_MODULES: [&str; 1] = ["mq_rotate_x_i4_f16out"];
        for module in MOE_FRONT_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing MoE front module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The 0.4.4 mq4rs (A3B MQ4G128V2) decode kernels: the fused qt53
        // nine-path down and the two-row shared-expert k512 GEMV, the norm
        // fold, and the lane-batched MoE block twins of the VMM exact route.
        const MQ4RS_DECODE_ADDED_MODULES: [&str; 7] = [
            "gemv_mq4g128v2_moe_ninepath_d4",
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2",
            "fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201",
            "moe_router_softmax_topk_k8_wave64_rows",
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_rows",
            "gemv_mq4g128v2_moe_ninepath_d4_rows",
        ];
        for module in MQ4RS_DECODE_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing mq4rs decode module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The A3B lane-batched lm_head of the exact CbVmm route (gfx1201).
        const A3B_HEAD_LANES_ADDED_MODULES: [&str; 1] = ["gemv_mq4g256v2_head_lanes_gfx1201"];
        for module in A3B_HEAD_LANES_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B head lanes module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The A3B expert-deduplicating routed MoE kernels (gfx1201).
        const A3B_MOE_DEDUP_ADDED_MODULES: [&str; 1] = ["moe_dedup_mq4rs_gfx1201"];
        for module in A3B_MOE_DEDUP_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B moe dedup module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The head_dim 64 Q8 flash decode tile (HIPF-124).
        const HD64_FLASH_TILE_ADDED_MODULES: [&str; 1] = ["attention_flash_q8_0_tile_hd64"];
        for module in HD64_FLASH_TILE_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing hd64 flash tile module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The native-fp8 split-KV FA2 DFlash verifier (HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8).
        const FP8_SPLIT_LSE_ADDED_MODULES: [&str; 1] =
            ["attention_fp8_e4m3_fa2_gqa_partial_lse_gfx1201"];
        for module in FP8_SPLIT_LSE_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing fp8 split LSE module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // Shared-launcher modules the packs lacked (LFM2 decode, plain Q8 GEMV).
        const SHARED_LAUNCHER_ADDED_MODULES: [&str; 4] = [
            "rope",
            "conv1d_gated_decode",
            "sample_top_p",
            "sample_rows_pf",
        ];
        for module in SHARED_LAUNCHER_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing shared launcher module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // Qwen3.6-27B CbVmm exact route and cross-request MTP verify segment
        // twin (gfx1201 only), measured JIT-compiled on the shipped pack.
        const VMM_EXACT_MTP_SEGS_ADDED_MODULES: [&str; 8] = [
            "kv_cache_write_fp8_e4m3_batched_vmm",
            "attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm",
            "attention_flash_reduce_dsplit_gfx1201_vmm",
            "gated_delta_net_q8_compact3_b2_rows",
            "conv1d_silu_split_qknorm_b256_rows",
            "gated_norm_mq_rotate_awq_k6144_gfx1201_rows",
            "qwen36_27b_fa_prep_gfx1201_rows",
            "attention_fp8_e4m3_kv_batched_segs",
        ];
        for module in VMM_EXACT_MTP_SEGS_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing VMM exact/segs module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // The gfx1201 Z/beta/alpha fold split of the non-fused IU4 verify GDN projection
        // (`gemm_zba_fold_g12_iu4_prepared`) is its own module on gfx1201 (the gfx1100/gfx1151
        // halo split of the same name is a P0 module there).
        const ZBA_FOLD_SPLIT_ADDED_MODULES: [&str; 1] = ["split_mq4v2_z_betaalpha"];
        for module in ZBA_FOLD_SPLIT_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing gfx1201 ZBA fold split module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // Exact-A3B CbVmm DeltaNet rows twins (gfx1201 only; the conv twin is
        // the shared 27B module above).
        const A3B_DN_ROWS_ADDED_MODULES: [&str; 2] = [
            "gated_delta_net_q8_compact2_b2_rows",
            "gated_norm_mq_rotate_gfx1201_rows",
        ];
        for module in A3B_DN_ROWS_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B DN rows module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // A3B lane-batched FullAttention on the exact CbVmm route (gfx1201).
        const A3B_FA_LANES_ADDED_MODULES: [&str; 3] = [
            "attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm",
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201_vmm",
            "qwen35_fa_prep_gfx1201_rows",
        ];
        for module in A3B_FA_LANES_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B FA lanes module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // gfx12 W8A8 / W8-FP8 Q8_0 prefill GEMMs (quantized-activation tiers).
        const Q8_0_MMQ_ACT_ADDED_MODULES: [&str; 2] =
            ["gemm_q8_0_mmq_i8_gfx12", "gemm_q8_0_mmq_fp8_gfx12"];
        for module in Q8_0_MMQ_ACT_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing Q8_0 MMQ module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // BF16-weight GEMMs (batched + x-stationary tiled) for the Raon mHC selector
        // and grouped-MoE router.
        const BF16_XF32_ROWS_ADDED_MODULES: [&str; 2] =
            ["gemm_bf16_xf32_batched", "gemm_bf16_xf32_tiled_rows"];
        for module in BF16_XF32_ROWS_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing BF16-weight rows module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // A3B VMM exact batch route: lane-batched multi-weight projection GEMV (gfx1201 only).
        const A3B_LANE_PROJ_ADDED_MODULES: [&str; 1] = ["gemv_mq4g256v2_lane_proj_k2048_gfx1201"];
        for module in A3B_LANE_PROJ_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B lane proj module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // A3B VMM exact batch route: lane-batched wo residual GEMV (gfx1201 only).
        const A3B_LANE_WO_ADDED_MODULES: [&str; 1] = ["gemv_mq4g256v2_residual_lanes_gfx1201"];
        for module in A3B_LANE_WO_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing A3B lane wo module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        // Symmetric qt49 MQ3V25 A4/A8 prefill consumers (gfx1201).
        const MQ3V25_ADDED_MODULES: [&str; 2] = [
            "gemm_mq3v25_residual_mmq_iu4_gfx1201",
            "gemm_mq3v25_residual_mmq_iu8_gfx1201",
        ];
        for module in MQ3V25_ADDED_MODULES {
            assert!(
                by_name.contains_key(module),
                "missing MQ3V25 module {module}"
            );
            assert!(
                !expected
                    .lines()
                    .any(|l| l.split('\t').next() == Some(module)),
                "{module} must not be a P0 module"
            );
        }
        assert_eq!(
            registry.len(),
            count
                + MQ3V25_ADDED_MODULES.len()
                + PRE_S1_ADDED_SINCE_P0
                + S1_ADDED_MODULES.len()
                + NATIVE_A3B_ADDED_MODULES.len()
                + FIRST_REQUEST_JIT_ADDED_MODULES
                + MQ6_I8_ADDED_MODULES.len()
                + HC_DOWN_FUSED_ADDED_MODULES.len()
                + SHARED_I8_ADDED_MODULES.len()
                + G64_AND_HC_CAPTURE_ADDED_MODULES.len()
                + HC_ROW_FOLD_ADDED_MODULES.len()
                + MOE_FRONT_ADDED_MODULES.len()
                + MQ4RS_DECODE_ADDED_MODULES.len()
                + HD64_FLASH_TILE_ADDED_MODULES.len()
                + FP8_SPLIT_LSE_ADDED_MODULES.len()
                + SHARED_LAUNCHER_ADDED_MODULES.len()
                + VMM_EXACT_MTP_SEGS_ADDED_MODULES.len()
                + ZBA_FOLD_SPLIT_ADDED_MODULES.len()
                + A3B_DN_ROWS_ADDED_MODULES.len()
                + A3B_FA_LANES_ADDED_MODULES.len()
                + A3B_HEAD_LANES_ADDED_MODULES.len()
                + A3B_LANE_WO_ADDED_MODULES.len()
                + A3B_MOE_DEDUP_ADDED_MODULES.len()
                + A3B_LANE_PROJ_ADDED_MODULES.len()
                + BF16_XF32_ROWS_ADDED_MODULES.len()
                + Q8_0_MMQ_ACT_ADDED_MODULES.len(),
            "unexpected gfx1201 inventory size"
        );
        let default_prefill = by_name
            .get("attention_q8_0_flash_prefill_br8_bc16")
            .unwrap();
        assert_eq!(default_prefill.symbols, ["attention_q8_0_flash_prefill"]);
        assert!(default_prefill
            .source()
            .starts_with("#define BR 8\n#define BC 16\n#define NTHREADS 256\n"));
        assert_ne!(
            default_prefill.source(),
            by_name
                .get("attention_q8_0_flash_prefill")
                .unwrap()
                .source()
        );
    }

    /// Modules the shared launchers request on every arch (the plain Q8_0 GEMV
    /// and the LFM2 decode path's RoPE, gated short-conv and samplers) are in
    /// every arch's registry, hence in every release kernel pack, with the
    /// runtime's exact source. A pack that lacks one makes a host without hipcc
    /// fail on first use. Each row is checked against the launcher's own
    /// `ensure_kernel` call text so the table cannot drift from the runtime.
    /// Not a general "every requested module is packaged" check: ~470 modules
    /// the runtime can request are deliberately JIT-only (lab, vision and
    /// older-family paths), so a blanket scrape needs an allowlist as large as
    /// the gap; this table is the set known to be hit on supported models.
    #[test]
    fn pack_carries_every_shared_launcher_module_on_every_arch() {
        fn squeeze(text: &str) -> String {
            text.chars().filter(|c| !c.is_whitespace()).collect()
        }
        let gemv = squeeze(include_str!("gemv.rs"));
        let norm = squeeze(include_str!("norm.rs"));
        let sampling = squeeze(include_str!("sampling.rs"));
        let rows: [(&str, &str, &str, &str, &str); 5] = [
            (
                "gemv_q8_0_wide",
                "GEMV_Q8_0_WIDE_SRC",
                "gemv_q8_0_wide",
                "gemv",
                kernels::GEMV_Q8_0_WIDE_SRC,
            ),
            ("rope", "ROPE_SRC", "rope_f32", "norm", kernels::ROPE_SRC),
            (
                "conv1d_gated_decode",
                "CONV1D_GATED_DECODE_SRC",
                "conv1d_gated_decode_f32",
                "norm",
                kernels::CONV1D_GATED_DECODE_SRC,
            ),
            (
                "sample_top_p",
                "SAMPLE_TOP_P_SRC",
                "sample_top_p",
                "sampling",
                kernels::SAMPLE_TOP_P_SRC,
            ),
            (
                "sample_rows_pf",
                "SAMPLE_ROWS_PF_SRC",
                "sample_rows_pf_f32",
                "sampling",
                kernels::SAMPLE_ROWS_PF_SRC,
            ),
        ];
        for (module, src_const, symbol, file, source) in rows {
            let call = squeeze(&format!(
                "ensure_kernel(\"{module}\",kernels::{src_const},\"{symbol}\""
            ));
            let text = match file {
                "gemv" => &gemv,
                "norm" => &norm,
                _ => &sampling,
            };
            assert!(
                text.contains(&call),
                "launcher no longer requests {module}/{symbol}"
            );
            for arch in ["gfx1100", "gfx1151", "gfx1201"] {
                let entry = lookup(arch, module, "")
                    .unwrap_or_else(|e| panic!("{arch} pack lacks module {module}: {e:?}"));
                assert!(
                    entry.symbols.contains(&symbol),
                    "{arch} {module} lacks {symbol}"
                );
                assert_eq!(entry.source(), source, "{arch} {module} source");
            }
        }
    }
    #[test]
    fn corpus_entries_name_each_module_once_per_arch() {
        for arch in SUPPORTED_ARCHES {
            let entries = corpus_entries(arch, "").unwrap();
            let mut seen = std::collections::HashSet::new();
            for entry in &entries {
                assert!(
                    seen.insert(entry.module),
                    "{arch}: module {} inventoried twice",
                    entry.module
                );
            }
        }
    }

    #[test]
    fn kernel_registry_explicit_arch_and_module_rejection() {
        assert!(matches!(
            lookup("gfx1200", "rmsnorm", ""),
            Err(RegistryError::UnsupportedArch(_))
        ));
        assert!(matches!(
            lookup("gfx942", "qwen35_fa_prep_batched_gfx1201", ""),
            Err(RegistryError::UnsupportedModule { .. })
        ));
        assert!(lookup("gfx942", "fused_qkv_hfq4g256_wave64", "").is_ok());
        assert!(lookup("gfx906", "gemm_qkv_hfq4g256_wmma_gfx12", "").is_err());
        for arch in SUPPORTED_ARCHES {
            assert!(lookup(arch, "attention_q8_0_flash_prefill_br8_bc16", "").is_ok());
            assert!(lookup(arch, "attention_q8_0_flash_prefill_br16_bc32", "").is_err());
        }
    }

    /// The opt-in Hyper inline-Q8 chunk module is inventoried on exact gfx1151
    /// only, carries the source `gated_delta_step_gate_wmma` compiles, and
    /// exports the symbol it launches.
    #[test]
    fn hyper_q8_chunk_module_is_exact_gfx1151_only() {
        let entry = lookup("gfx1151", "gated_delta_chunk_q8_wmma", "").unwrap();
        assert_eq!(entry.symbols, ["gated_delta_chunk_gate_q8_wmma"]);
        assert_eq!(
            entry.source(),
            crate::tensor_ops::GATED_DELTA_CHUNK_Q8_WMMA_SRC
        );
        assert!(entry
            .source()
            .contains("void gated_delta_chunk_gate_q8_wmma("));
        for arch in SUPPORTED_ARCHES.iter().filter(|arch| **arch != "gfx1151") {
            assert!(
                lookup(arch, "gated_delta_chunk_q8_wmma", "").is_err(),
                "{arch} must not inventory the gfx1151 module"
            );
        }
    }

    /// Every Q8 GDN entry the `tensor_ops` inventory carries has its Q8Ef twin
    /// and the residual zero kernel, and each exported symbol is defined by
    /// the source; no inline-WY EF module is inventoried.
    #[test]
    fn gdn_q8ef_twins_are_inventoried() {
        for arch in ["gfx1151", "gfx1201"] {
            let entry = lookup(arch, "tensor_ops", "").unwrap();
            assert!(entry.symbols.contains(&"gdn_q8ef_residual_zero"), "{arch}");
            for symbol in entry
                .symbols
                .iter()
                .filter(|s| s.starts_with("gdn_state_") || s.starts_with("gated_delta_step_"))
            {
                if let Some(base) = symbol.strip_suffix("_q8") {
                    let twin = format!("{base}_q8ef");
                    assert!(entry.symbols.iter().any(|s| *s == twin), "{arch}: {twin}");
                    assert!(
                        entry.source().contains(&format!("void {twin}(")),
                        "{arch}: {twin}"
                    );
                }
            }
            assert!(!entry
                .symbols
                .iter()
                .any(|s| s.starts_with("gated_delta_rollback_layers_q8ef")));
        }
        // The Q8Ef inline modules were removed until they have a GPU gate: no arch (gfx1151
        // included) inventories them.
        for arch in SUPPORTED_ARCHES {
            assert!(
                lookup(arch, "gated_delta_chunk_q8ef_wmma", "").is_err(),
                "{arch}"
            );
            assert!(
                lookup(arch, "gated_delta_chunk_q8ef_wmma_arena", "").is_err(),
                "{arch}"
            );
        }
        let plain = lookup("gfx1151", "gated_delta_chunk_q8_wmma", "").unwrap();
        let plain_arena = lookup("gfx1151", "gated_delta_chunk_q8_wmma_arena", "").unwrap();
        assert!(!plain.source().contains("q8ef"));
        assert!(!plain_arena.source().contains("q8ef"));
    }

    /// Kernels `kernels/src/tensor_ops.hip` (as the registry compiles it) exports
    /// on `arch`, read from the source's own preprocessor guards: `__GFX11__`
    /// (gfx1100, gfx1151), `__GFX12__` (gfx1201), `__gfx942__`, and the
    /// `QSA_HAS_*` capability macros the source defines from them. Covers the
    /// plain `extern "C" __global__` kernels and the QSA kernel macros.
    fn source_exports(src: &str, arch: &str) -> std::collections::BTreeSet<String> {
        let mut defined: HashSet<String> = HashSet::new();
        match arch {
            "gfx1100" | "gfx1151" => {
                defined.insert("__GFX11__".to_owned());
            }
            "gfx1201" => {
                defined.insert("__GFX12__".to_owned());
            }
            "gfx942" => {
                defined.insert("__gfx942__".to_owned());
            }
            _ => {}
        }
        let eval = |condition: &str, defined: &HashSet<String>| -> bool {
            condition.split("||").any(|term| {
                let term = term.trim();
                let (negate, term) = match term.strip_prefix('!') {
                    Some(inner) => (true, inner.trim()),
                    None => (false, term),
                };
                let name = term
                    .strip_prefix("defined(")
                    .and_then(|inner| inner.strip_suffix(')'))
                    .unwrap_or_else(|| panic!("unsupported preprocessor condition {condition:?}"));
                defined.contains(name.trim()) != negate
            })
        };
        let lines: Vec<&str> = src.lines().collect();
        // (this branch is live, some branch of the group was taken)
        let mut stack: Vec<(bool, bool)> = Vec::new();
        let mut active_text = String::new();
        let mut index = 0;
        while index < lines.len() {
            let line = lines[index];
            index += 1;
            let Some(directive) = line.trim().strip_prefix('#') else {
                if stack.iter().all(|(live, _)| *live) {
                    active_text.push_str(line);
                    active_text.push('\n');
                }
                continue;
            };
            let mut directive = directive.trim_start().to_owned();
            while directive.ends_with('\\') {
                directive.pop();
                directive.push_str(lines[index].trim());
                index += 1;
            }
            let (keyword, rest) = directive
                .split_once(char::is_whitespace)
                .unwrap_or((directive.as_str(), ""));
            let rest = rest.split("//").next().unwrap_or("").trim();
            let live = stack.iter().all(|(live, _)| *live);
            match keyword {
                "if" => {
                    let value = eval(rest, &defined);
                    stack.push((value, value));
                }
                "ifdef" => {
                    let value = defined.contains(rest);
                    stack.push((value, value));
                }
                "ifndef" => {
                    let value = !defined.contains(rest);
                    stack.push((value, value));
                }
                "elif" => {
                    let (_, taken) = stack.pop().expect("#elif without #if");
                    let value = !taken && eval(rest, &defined);
                    stack.push((value, taken || value));
                }
                "else" => {
                    let (_, taken) = stack.pop().expect("#else without #if");
                    stack.push((!taken, true));
                }
                "endif" => {
                    stack.pop().expect("#endif without #if");
                }
                "define" if live => {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if name.starts_with("QSA_HAS_") {
                        defined.insert(name);
                    }
                }
                _ => {}
            }
        }
        assert!(stack.is_empty(), "unbalanced preprocessor conditionals");

        let mut names = std::collections::BTreeSet::new();
        let identifier = |text: &str| -> String {
            text.trim_start()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect()
        };
        let mut rest = active_text.as_str();
        while let Some(at) = rest.find("extern \"C\" __global__") {
            let mut tail = rest[at + "extern \"C\" __global__".len()..].trim_start();
            rest = tail;
            if let Some(after) = tail.strip_prefix("__launch_bounds__") {
                let close = after.find(')').expect("unterminated __launch_bounds__");
                tail = after[close + 1..].trim_start();
            }
            if let Some(after) = tail.strip_prefix("void") {
                names.insert(identifier(after));
            }
        }
        for line in active_text.lines() {
            let line = line.trim();
            for prefix in [
                "QSA_SELECT_SERIAL(",
                "QSA_SELECT_PARALLEL(",
                "QSA_DECODE_PROLOGUE(",
                "QSA_ATTENTION_BATCHED(",
            ] {
                if let Some(args) = line.strip_prefix(prefix) {
                    names.insert(identifier(args));
                }
            }
            if let Some(args) = line.strip_prefix("QSA_ATTENTION_KERNELS(") {
                let tag = identifier(args);
                for suffix in ["_batched_serial", "_batched", "_batched_hg4", "_serial", ""] {
                    names.insert(format!("indexed_attention_attention_{tag}{suffix}"));
                }
            }
        }
        names
    }

    fn tensor_ops_source_exports(arch: &str) -> std::collections::BTreeSet<String> {
        source_exports(crate::tensor_ops::TENSOR_OPS_SRC, arch)
    }

    /// The legacy `moe_down_combine_grouped_top10` source is byte-for-byte the
    /// one the G0 fixtures and the walked gfx1201 plans recorded, and the S1
    /// slow-link kernels live in a module of their own, `slowlink_prefill`.
    /// Both modules take the same registry path as every other: each index is
    /// its source's export set on every qwen4 arch, and every kernel a
    /// `slowlink_*` wrapper requests (read from `slab_moe.rs` itself, no
    /// allowlist here) is in the new module's index, so the official pack of
    /// each arch carries them with no arch-specific gate. (S1 stays default-off;
    /// the Flash-Next walked plans are open and unedited, so these load lazily
    /// from the same index.)
    #[test]
    fn s1_d_kernels_are_in_their_own_packaged_module_on_every_qwen4_arch() {
        use std::collections::BTreeSet;
        const OLD: &str = "moe_down_combine_grouped_top10";
        const NEW: &str = "slowlink_prefill";
        const OLD_SHA: &str = "b5714d81799dee960acd90adb2aba167df2ccacbe2a2907bd58e98d041cba334";
        let slab = include_str!("slab_moe.rs");
        let wrappers: BTreeSet<&str> = slab
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix("const FUNC: &str = \"slowlink_")?
                    .strip_suffix("\";")
            })
            .collect();
        assert!(wrappers.len() >= 6, "{wrappers:?}");
        // Every slowlink wrapper loads the new module and source; none the legacy one.
        let loads = |needle: &str| slab.lines().filter(|l| l.contains(needle)).count();
        assert_eq!(loads("kernels::SLOWLINK_PREFILL_SRC"), wrappers.len());
        assert_eq!(loads("MOE_DOWN_COMBINE_GROUPED_TOP10_SRC"), 0);
        assert_eq!(
            loads(&format!("const MODULE: &str = \"{NEW}\";")),
            wrappers.len()
        );
        assert_eq!(loads(&format!("\"{OLD}\"")), 0);
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let old = lookup(arch, OLD, "").unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(old.source().as_bytes())),
                OLD_SHA,
                "{arch}"
            );
            let new = lookup(arch, NEW, "").unwrap();
            let index = |entry: &KernelEntry| -> BTreeSet<String> {
                entry.symbols.iter().map(|s| (*s).to_owned()).collect()
            };
            assert_eq!(
                index(&old),
                source_exports(old.source(), arch),
                "{arch} {OLD}"
            );
            assert_eq!(
                index(&new),
                source_exports(new.source(), arch),
                "{arch} {NEW}"
            );
            assert!(
                index(&old).iter().all(|s| !s.starts_with("slowlink_")),
                "{arch}"
            );
            assert!(index(&old).is_disjoint(&index(&new)), "{arch}");
            for name in &wrappers {
                let symbol = format!("slowlink_{name}");
                assert!(index(&new).contains(&symbol), "{arch}: {symbol}");
            }
            assert_eq!(
                index(&new).len(),
                wrappers.len(),
                "{arch}: unrequested export"
            );
        }
        // The walked plans still name the legacy module at its recorded identity.
        let old = lookup("gfx1201", OLD, "").unwrap();
        let live = format!("{:x}", Sha256::digest(old.source().as_bytes()));
        for plan in walked_route_plans().unwrap() {
            for row in plan.rows.iter().filter(|r| !r.embedded && r.module == OLD) {
                assert_eq!(row.sha256, live);
            }
        }
    }

    const SEALED_J1778_GFX1100: &str =
        include_str!("../tests/fixtures/pack-closure/tensor_ops-gfx1100-j1778.symbols.txt");

    /// Preserve the measured sealed gfx1100 object's historical export/index
    /// gap; the later gfx1201-only x8 addition does not alter gfx11 exports. The
    /// current source adds two gfx11 exports the sealed object predates: the
    /// rope table and the q8 hg4 low-LDS twin (`..._hg4_lite`).
    #[test]
    fn source_export_evaluator_reproduces_the_sealed_gfx1100_object_and_index_gap() {
        use std::collections::BTreeSet;
        let rows = |tag: &str| -> BTreeSet<String> {
            SEALED_J1778_GFX1100
                .lines()
                .filter_map(|line| line.strip_prefix(tag))
                .map(str::to_owned)
                .collect()
        };
        let (index, elf) = (rows("index "), rows("export "));
        assert_eq!((index.len(), elf.len()), (74, 83));
        assert!(
            index.is_subset(&elf),
            "the sealed index named a symbol its object lacks"
        );
        let current: BTreeSet<String> = elf
            .iter()
            .cloned()
            .chain([
                "indexed_attention_attention_q8_batched_hg4_lite".to_owned(),
                "indexed_attention_rope_table_f32".to_owned(),
            ])
            .collect();
        assert_eq!(tensor_ops_source_exports("gfx1100"), current);
        let omitted: Vec<&str> = elf.difference(&index).map(String::as_str).collect();
        assert_eq!(
            omitted,
            [
                "gated_delta_step_halves_state128_persistent256_canon_q8",
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                "indexed_attention_attention_q8",
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_serial",
                "indexed_attention_attention_q8_serial",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_q8",
            ]
        );
        // The current gfx11 registry contains exactly the measured source exports.
        for arch in ["gfx1100", "gfx1151"] {
            let registered: BTreeSet<String> = lookup(arch, "tensor_ops", "")
                .unwrap()
                .symbols
                .iter()
                .map(|s| (*s).to_owned())
                .collect();
            assert_eq!(registered, current, "{arch}");
        }
    }

    /// Every supported arch's `tensor_ops` index is exactly the source's
    /// export set under that arch's capability macros: no supported export
    /// (Q8 KV, canonical Q8 prefill, ...) is missing from the index, and no
    /// export another arch's guards define (gfx12 FP8 KV, pipe128) is
    /// declared where the source does not compile it.
    #[test]
    fn tensor_ops_index_is_the_source_export_closure_per_arch() {
        use std::collections::BTreeSet;
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let exported = tensor_ops_source_exports(arch);
            let registered: BTreeSet<String> = lookup(arch, "tensor_ops", "")
                .unwrap()
                .symbols
                .iter()
                .map(|s| (*s).to_owned())
                .collect();
            let omitted: Vec<_> = exported.difference(&registered).collect();
            let phantom: Vec<_> = registered.difference(&exported).collect();
            assert!(
                omitted.is_empty(),
                "{arch}: exports missing from the index: {omitted:?}"
            );
            assert!(
                phantom.is_empty(),
                "{arch}: indexed but not compiled: {phantom:?}"
            );
        }
        let has = |arch: &str, symbol: &str| tensor_ops_source_exports(arch).contains(symbol);
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            for q8 in [
                "indexed_attention_decode_prologue_q8",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_attention_q8_batched",
                "gated_delta_step_halves_state128_persistent256_canon_q8",
            ] {
                assert!(has(arch, q8), "{arch}: {q8}");
            }
        }
        for (symbol, gfx11, gfx12) in [
            ("hyper_write_bf16x2", true, true),
            ("hyper_write_bf16x8_gfx1201", false, true),
            ("indexed_attention_decode_prologue_fp8", false, true),
            ("gated_delta_step_pipe128_q8", false, true),
        ] {
            assert_eq!(has("gfx1100", symbol), gfx11, "gfx1100 {symbol}");
            assert_eq!(has("gfx1151", symbol), gfx11, "gfx1151 {symbol}");
            assert_eq!(has("gfx1201", symbol), gfx12, "gfx1201 {symbol}");
        }
        // gfx942 compiles no Q8 KV (and the registry packages no `tensor_ops` there).
        assert!(!has("gfx942", "indexed_attention_decode_prologue_q8"));
        for arch in ["gfx906", "gfx942"] {
            assert!(lookup(arch, "tensor_ops", "").is_err(), "{arch}");
        }
    }

    /// Every symbol a closed route plan can request of a HIP module is in that
    /// module's index on its arch: the 27B plan tables (corpus) and every
    /// walked plan (installer inventory, which is the packaged index).
    #[test]
    fn closed_route_plan_symbols_are_registry_indexed() {
        let flags = default_feature_flags("gfx1201").unwrap().hipcc_extra_flags;
        let corpus = route_corpus("gfx1201", &flags).unwrap();
        let tables: [(&str, &[RouteHip]); 10] = [
            ("AR", QWEN36_27B_GFX1201_AR_HIP),
            ("MTP", QWEN36_27B_GFX1201_MTP_HIP),
            ("ROWS_TWINS", QWEN36_27B_GFX1201_ROWS_TWINS_HIP),
            ("VMM_EXACT", QWEN36_27B_GFX1201_VMM_EXACT_HIP),
            ("XBATCH", QWEN36_27B_GFX1201_XBATCH_HIP),
            ("SPEC_CB", QWEN36_27B_GFX1201_SPEC_CB_HIP),
            ("PLAIN_RMSNORM", QWEN36_27B_GFX1201_PLAIN_RMSNORM_HIP),
            ("PLAIN_SILU", QWEN36_27B_GFX1201_PLAIN_SILU_HIP),
            ("PLAIN_GATED_NORM", QWEN36_27B_GFX1201_PLAIN_GATED_NORM_HIP),
            ("PLAIN_ROTATE", QWEN36_27B_GFX1201_PLAIN_ROTATE_HIP),
        ];
        for (label, table) in tables {
            for &(module, symbols) in table {
                let entry = corpus
                    .iter()
                    .find(|e| e.module == module)
                    .unwrap_or_else(|| panic!("{label}: {module} is not in the gfx1201 registry"));
                for symbol in symbols {
                    assert!(
                        entry.symbols.contains(symbol),
                        "{label}: {module} plans {symbol}, which its index omits"
                    );
                }
            }
        }
        for plan in walked_route_plans().unwrap() {
            let flags = default_feature_flags(plan.arch).unwrap().hipcc_extra_flags;
            let inventory = entries(plan.arch, &flags).unwrap();
            for row in plan.rows.iter().filter(|row| !row.embedded) {
                let entry = inventory
                    .iter()
                    .find(|e| e.module == row.module)
                    .unwrap_or_else(|| panic!("{}: {} is not packaged", plan.family, row.module));
                for symbol in &row.symbols {
                    assert!(
                        entry.symbols.contains(symbol),
                        "{}: {} walks {symbol}, which its index omits",
                        plan.family,
                        row.module
                    );
                }
            }
        }
    }
    #[test]
    fn dense_dflash_cold_helpers_preserve_runtime_identity() {
        let cross_admitted = [
            "dflash_hidden_commit5_gfx1100",
            "dflash_hidden_scatter5_gfx1100",
            "gated_norm_mq_rotate_f16",
            "fused_silu_mul_mq_rotate_f16",
            "sigmoid_mul_mq_rotate_f16",
            "fused_rmsnorm_mq_rotate_awq_f16",
            "fused_rmsnorm_mq_rotate_f16",
            "dflash_gdn_pre_gfx1100",
            "qwen35_fa_prep_batched_gfx1100",
            "kv_cache_write_q8_0_pair_batched_gfx1100",
        ];
        for module in cross_admitted {
            let entry = lookup("gfx1201", module, "").unwrap();
            let original = lookup("gfx1100", module, "").unwrap();
            assert_eq!(entry.source(), original.source(), "{module}");
            assert_eq!(entry.symbols, original.symbols, "{module}");
        }
        let exact = [
            (
                "gfx1201",
                "dflash_gdn_replay_pre_ml_from",
                crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC,
                &["dflash_gdn_replay_pre_ml_from"][..],
            ),
            (
                "gfx1201",
                "gated_delta_net_q8_fast_ml_from",
                crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC,
                &["gated_delta_net_q8_fast_ml_from"][..],
            ),
            (
                "gfx1201",
                "attention_verify_gqa_gfx1201",
                kernels::ATTENTION_VERIFY_GQA_GFX1201_SRC,
                &[
                    "attention_verify_gqa_q8_gfx1201",
                    "attention_verify_gqa_fp8_gfx1201",
                    "attention_verify_gqa_fp8_gqa8_gfx1201",
                    "attention_verify_gqa_fwht4_gfx1201",
                    "attention_verify_gqa_fwht3_gfx1201",
                    "attention_verify_reduce_gfx1201",
                ][..],
            ),
            (
                "gfx1100",
                "attention_verify_gqa_gfx1100",
                kernels::ATTENTION_VERIFY_GQA_GFX1100_SRC,
                &[
                    "attention_verify_gqa_q8_gfx1100",
                    "attention_verify_gqa_fwht4_gfx1100",
                    "attention_verify_gqa_fwht3_gfx1100",
                    "attention_verify_gqa_q8_rows_gfx1100",
                    "attention_verify_reduce_gfx1100",
                ][..],
            ),
        ];
        for (arch, module, source, symbols) in exact {
            let entry = lookup(arch, module, "").unwrap();
            assert_eq!(entry.source(), source, "{module}");
            assert_eq!(entry.symbols, symbols, "{module}");
            let other = if arch == "gfx1201" {
                "gfx1100"
            } else {
                "gfx1201"
            };
            assert!(lookup(other, module, "").is_err());
        }
        // Pure byte-copy DFlash helpers are also admitted on exact gfx1151 with
        // the identical module name, source, symbols and compile recipe.
        let gfx1151_byte_copy = [
            "dflash_hidden_commit5_gfx1100",
            "dflash_hidden_scatter5_gfx1100",
            "dflash_state_bulk_copy_gfx1100",
        ];
        for module in gfx1151_byte_copy {
            let entry = lookup("gfx1151", module, "").unwrap();
            let original = lookup("gfx1100", module, "").unwrap();
            assert_eq!(entry.source(), original.source(), "{module}");
            assert_eq!(entry.symbols, original.symbols, "{module}");
            let recipe = KernelCompiler::recipe_for_source("gfx1151", module, entry.source(), "");
            assert_eq!(entry.flags, recipe.flags, "{module}");
            assert_eq!(
                entry.scheduler_profile, recipe.scheduler_profile,
                "{module}"
            );
            for arch in ["gfx906", "gfx942"] {
                assert!(lookup(arch, module, "").is_err(), "{arch}: leaked {module}");
            }
        }
        let modules = cross_admitted
            .into_iter()
            .chain(exact.into_iter().map(|(_, module, _, _)| module));
        for module in modules {
            for arch in ["gfx1151", "gfx906", "gfx942"] {
                if arch == "gfx1151" && gfx1151_byte_copy.contains(&module) {
                    continue;
                }
                assert!(lookup(arch, module, "").is_err(), "{arch}: leaked {module}");
            }
            let arch = if module == "attention_verify_gqa_gfx1100" {
                "gfx1100"
            } else {
                "gfx1201"
            };
            for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
                let entry = lookup(arch, module, extra_flags).unwrap();
                let recipe =
                    KernelCompiler::recipe_for_source(arch, module, entry.source(), extra_flags);
                assert_eq!(entry.flags, recipe.flags, "{module}");
                assert_eq!(
                    entry.scheduler_profile, recipe.scheduler_profile,
                    "{module}"
                );
                assert_eq!(
                    entry.scheduler_profile.as_deref(),
                    Some("default"),
                    "{module}"
                );
            }
        }
    }

    /// The gfx1201 DFlash-draft MQ4G256V2 overwrite WMMA twins
    /// (`gemm_mq4g256v2_overwrite_gfx12_dflash`, default 27B draft route) are
    /// packaged under the exact module keys and symbols their launchers pass to
    /// `ensure_kernel` / `launch_gemm_vt` (the one-tile kernel, and every
    /// `_vt{2,3,4}w{4,8}` symbol of `vt_symbols!`), with the launchers' sources,
    /// on exact gfx1201 only.
    #[test]
    fn gfx1201_draft_overwrite_twins_are_packaged() {
        let one_tile = lookup("gfx1201", "gemm_mq4g256v2_overwrite_wmma_gfx12", "").unwrap();
        assert_eq!(one_tile.symbols, ["gemm_mq4g256v2_overwrite_wmma_gfx12"]);
        assert_eq!(
            one_tile.source(),
            kernels::GEMM_MQ4G256V2_OVERWRITE_WMMA_GFX12_SRC
        );
        let vt = lookup("gfx1201", "gemm_mq4g256v2_overwrite_wmma_gfx12_vt", "").unwrap();
        assert_eq!(
            vt.source(),
            kernels::GEMM_MQ4G256V2_OVERWRITE_WMMA_GFX12_VT_SRC
        );
        assert_eq!(
            vt.symbols,
            [
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt2w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt2w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt3w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt3w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w4",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w8",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w4_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt4w8_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt8w4_k32",
                "gemm_mq4g256v2_overwrite_wmma_gfx12_vt8w8_k32",
            ]
        );
        // The vt symbol set is the residual twin's, symbol for symbol.
        let residual = lookup("gfx1201", "gemm_mq4g256v2_residual_wmma_gfx12_vt", "").unwrap();
        let rename = |symbol: &&str| symbol.replace("_residual_", "_overwrite_");
        assert_eq!(
            residual.symbols.iter().map(rename).collect::<Vec<_>>(),
            vt.symbols
        );
        for module in [
            "gemm_mq4g256v2_overwrite_wmma_gfx12",
            "gemm_mq4g256v2_overwrite_wmma_gfx12_vt",
        ] {
            for arch in ["gfx1100", "gfx1151", "gfx906", "gfx942"] {
                assert!(lookup(arch, module, "").is_err(), "{arch}: leaked {module}");
            }
        }
    }

    /// The DFlash verify tape-capture copy and the lane-batched conv are
    /// packaged under the exact module key and symbol their launchers pass to
    /// `ensure_kernel`, with the launchers' own source constants, and each
    /// entry's recipe is the compiler's recipe for that source. The copy kernel
    /// has no arch guard or launcher gate, so every inventory carries it; the
    /// lane conv is only admitted on exact gfx1201 (`lane_gdn_batch_admitted`).
    #[test]
    fn dflash_tape_copy_and_lane_conv_match_launchers_and_compiler() {
        use crate::lane_gdn::{CONV1D_SILU_SPLIT_LANES_SRC, CONV1D_SILU_SPLIT_LANES_SYMBOL};
        use crate::tape_copy::{TAPE_COPY_BATCHED_SRC, TAPE_COPY_BATCHED_SYMBOL};
        for arch in SUPPORTED_ARCHES {
            for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
                let copy = lookup(arch, TAPE_COPY_BATCHED_SYMBOL, extra_flags)
                    .unwrap_or_else(|e| panic!("{arch}: tape copy not packaged: {e:?}"));
                assert_eq!(copy.module, TAPE_COPY_BATCHED_SYMBOL, "{arch}");
                assert_eq!(copy.symbols, [TAPE_COPY_BATCHED_SYMBOL], "{arch}");
                assert_eq!(copy.source(), TAPE_COPY_BATCHED_SRC, "{arch}");
                let recipe = KernelCompiler::recipe_for_source(
                    arch,
                    TAPE_COPY_BATCHED_SYMBOL,
                    TAPE_COPY_BATCHED_SRC,
                    extra_flags,
                );
                assert_eq!(copy.flags, recipe.flags, "{arch}: {extra_flags}");
                assert_eq!(
                    copy.scheduler_profile, recipe.scheduler_profile,
                    "{arch}: {extra_flags}"
                );

                let conv = lookup(arch, CONV1D_SILU_SPLIT_LANES_SYMBOL, extra_flags);
                if *arch != "gfx1201" {
                    assert!(conv.is_err(), "{arch}: leaked lane conv");
                    continue;
                }
                let conv = conv.unwrap();
                assert_eq!(conv.module, CONV1D_SILU_SPLIT_LANES_SYMBOL);
                assert_eq!(conv.symbols, [CONV1D_SILU_SPLIT_LANES_SYMBOL]);
                assert_eq!(conv.source(), CONV1D_SILU_SPLIT_LANES_SRC);
                let recipe = KernelCompiler::recipe_for_source(
                    arch,
                    CONV1D_SILU_SPLIT_LANES_SYMBOL,
                    CONV1D_SILU_SPLIT_LANES_SRC,
                    extra_flags,
                );
                assert_eq!(conv.flags, recipe.flags, "{extra_flags}");
                assert_eq!(
                    conv.scheduler_profile, recipe.scheduler_profile,
                    "{extra_flags}"
                );
            }
        }
    }

    /// `Gpu::bf16_round_trip_f32` (norm.rs) passes module `bf16_round_trip`,
    /// source `kernels::BF16_ROUND_TRIP_SRC` and symbol `bf16_round_trip_f32`
    /// to `ensure_kernel` on every call (Qwen4 MoE / dense-FFN / QSA storage
    /// boundaries, dots.ocr). The kernel has no arch guard and the launcher no
    /// arch gate, so every inventory carries it, with the compiler's recipe.
    #[test]
    fn bf16_round_trip_matches_launcher_and_compiler() {
        for arch in SUPPORTED_ARCHES {
            for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
                let entry = lookup(arch, "bf16_round_trip", extra_flags)
                    .unwrap_or_else(|e| panic!("{arch}: bf16_round_trip not packaged: {e:?}"));
                assert_eq!(entry.symbols, ["bf16_round_trip_f32"], "{arch}");
                assert_eq!(entry.source(), kernels::BF16_ROUND_TRIP_SRC, "{arch}");
                let recipe = KernelCompiler::recipe_for_source(
                    arch,
                    "bf16_round_trip",
                    kernels::BF16_ROUND_TRIP_SRC,
                    extra_flags,
                );
                assert_eq!(entry.flags, recipe.flags, "{arch}: {extra_flags}");
                assert_eq!(
                    entry.scheduler_profile, recipe.scheduler_profile,
                    "{arch}: {extra_flags}"
                );
            }
            // The strided Qwen4 sibling is a different module and source.
            assert_ne!(
                kernels::BF16_ROUND_TRIP_SRC,
                kernels::QWEN4_BF16_ROUND_TRIP_SRC
            );
        }
    }

    /// The QSA gathered F16 WMMA module (`tensor_ops::qsa_gathered_wmma_launch_with`,
    /// the F16 arena producer) is requested on gfx1151, its only route arch
    /// (`qsa_gathered_wmma_enabled`: F32/Q8 K/V); it was compiled after `ready`
    /// by the first request of a Halo Flash-Next MQ4 load. Its gfx1151 source,
    /// recipe and every launched symbol are packaged, and no other arch carries it.
    #[test]
    fn qsa_gathered_wmma_is_packaged_on_gfx1151() {
        for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
            let entry = lookup("gfx1151", "indexed_attention_gathered_wmma", extra_flags)
                .unwrap_or_else(|e| panic!("gathered QSA module not packaged: {e:?}"));
            assert_eq!(
                entry.symbols,
                [
                    "indexed_attention_gathered_wmma_f16",
                    "indexed_attention_kv_f16vb",
                    "indexed_attention_kv_f16vb_q8",
                ]
            );
            assert_eq!(
                entry.source(),
                crate::tensor_ops::INDEXED_ATTENTION_GATHERED_WMMA_SRC
            );
            let recipe = KernelCompiler::recipe_for_source(
                "gfx1151",
                "indexed_attention_gathered_wmma",
                crate::tensor_ops::INDEXED_ATTENTION_GATHERED_WMMA_SRC,
                extra_flags,
            );
            assert_eq!(entry.flags, recipe.flags, "{extra_flags}");
            assert_eq!(
                entry.scheduler_profile, recipe.scheduler_profile,
                "{extra_flags}"
            );
        }
        for arch in SUPPORTED_ARCHES.iter().filter(|arch| **arch != "gfx1151") {
            assert!(
                lookup(arch, "indexed_attention_gathered_wmma", "").is_err(),
                "{arch}: gathered QSA module leaked"
            );
        }
    }

    /// Every module a kernel-load trace loaded (rows: arch, module, symbols the
    /// run resolved) is in that arch's installer inventory and exports every
    /// resolved symbol, so a compiler-free pack serves the whole traced run.
    fn assert_trace_packaged(fixture: &str) {
        let mut by_arch: HashMap<&str, Vec<KernelEntry>> = HashMap::new();
        let mut rows = 0;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let [arch, module, symbols] = line.split('\t').collect::<Vec<_>>()[..] else {
                panic!("malformed trace row {line:?}");
            };
            let inventory = by_arch
                .entry(arch)
                .or_insert_with(|| entries(arch, "").unwrap());
            let entry = inventory
                .iter()
                .find(|entry| entry.module == module)
                .unwrap_or_else(|| panic!("{arch}: traced module {module} is not packaged"));
            for symbol in symbols.split(',') {
                assert!(
                    entry.symbols.contains(&symbol),
                    "{arch} {module}: packaged symbols lack {symbol}"
                );
            }
            rows += 1;
        }
        assert!(rows > 0, "empty trace fixture");
    }

    #[test]
    fn qwen4_flash_next_trace_is_packaged() {
        assert_trace_packaged(include_str!(
            "../tests/fixtures/kernel-trace-qwen4-flash-next.tsv"
        ));
    }

    #[test]
    fn qwen35_trace_is_packaged() {
        assert_trace_packaged(include_str!("../tests/fixtures/kernel-trace-qwen35.tsv"));
    }

    /// Every module the FirstRequestJit discovery holds (fresh kernel cache,
    /// product defaults, diverse first requests) compiled after `ready`.
    #[test]
    fn first_request_discovery_is_packaged() {
        assert_trace_packaged(include_str!(
            "../tests/fixtures/kernel-trace-first-request.tsv"
        ));
    }

    // ── Declared-symbol completeness ────────────────────────────────────────
    // The packaged object's index records the symbols its recipe declares, and
    // `KernelCompiler::check_indexed_object` rejects the object when a requested
    // symbol is not among them. So every `extern "C" __global__` kernel a source
    // defines for an arch must be declared by its entry; this scanner is a
    // minimal C preprocessor (comments, `#if`/`#ifdef`/`#elif`/`#else`,
    // object- and function-like `#define`) plus a kernel-name extractor
    // (`__global__ ... void NAME(`, macro-resolved names, `##`-pasted names of
    // function-like generator macros expanded at their active invocations).

    fn strip_c_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out = String::with_capacity(src.len());
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'"' | b'\'' => {
                    let q = b[i];
                    out.push(q as char);
                    i += 1;
                    while i < b.len() && b[i] != q {
                        if b[i] == b'\\' && i + 1 < b.len() {
                            out.push(b[i] as char);
                            i += 1;
                        }
                        out.push(b[i] as char);
                        i += 1;
                    }
                    if i < b.len() {
                        out.push(q as char);
                        i += 1;
                    }
                }
                b'/' if b.get(i + 1) == Some(&b'/') => {
                    while i < b.len() && b[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if b.get(i + 1) == Some(&b'*') => {
                    i += 2;
                    while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                        if b[i] == b'\n' {
                            out.push('\n');
                        }
                        i += 1;
                    }
                    i += 2;
                    out.push(' ');
                }
                c => {
                    out.push(c as char);
                    i += 1;
                }
            }
        }
        out
    }

    fn c_idents(s: &str) -> Vec<(usize, usize)> {
        let b = s.as_bytes();
        let mut v = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_alphabetic() || b[i] == b'_' {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                v.push((st, i));
            } else if b[i].is_ascii_digit() {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        v
    }

    fn eval_pp_cond(expr: &str, defs: &HashMap<String, String>) -> bool {
        // Tokens.
        #[derive(Clone, Debug, PartialEq)]
        enum T {
            Num(i64),
            Id(String),
            Op(&'static str),
        }
        let b = expr.as_bytes();
        let mut toks = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c.is_ascii_whitespace() {
                i += 1;
            } else if c.is_ascii_digit() {
                let st = i;
                while i < b.len() && b[i].is_ascii_alphanumeric() {
                    i += 1;
                }
                let t = expr[st..i]
                    .trim_end_matches(|c: char| c == 'u' || c == 'U' || c == 'l' || c == 'L');
                toks.push(T::Num(t.parse::<i64>().unwrap_or(0)));
            } else if c.is_ascii_alphabetic() || c == b'_' {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                toks.push(T::Id(expr[st..i].to_owned()));
            } else {
                let two = expr.get(i..i + 2).unwrap_or("");
                let op = ["&&", "||", "==", "!=", "<=", ">="]
                    .into_iter()
                    .find(|o| *o == two);
                if let Some(o) = op {
                    toks.push(T::Op(o));
                    i += 2;
                } else {
                    let o = ["(", ")", "!", "<", ">", "+", "-", "*", "/", "%"]
                        .into_iter()
                        .find(|o| o.as_bytes()[0] == c)
                        .unwrap_or_else(|| panic!("unsupported #if token {c:?} in {expr:?}"));
                    toks.push(T::Op(o));
                    i += 1;
                }
            }
        }
        struct P<'a> {
            t: Vec<T>,
            i: usize,
            defs: &'a HashMap<String, String>,
        }
        impl P<'_> {
            fn peek(&self) -> Option<&T> {
                self.t.get(self.i)
            }
            fn eat(&mut self, op: &str) -> bool {
                if matches!(self.peek(), Some(T::Op(o)) if *o == op) {
                    self.i += 1;
                    true
                } else {
                    false
                }
            }
            fn ident_value(&self, name: &str, depth: u32) -> i64 {
                match self.defs.get(name) {
                    Some(v) if depth < 8 => {
                        let v = v.trim();
                        if let Ok(n) = v.parse::<i64>() {
                            n
                        } else if v.is_empty() {
                            0
                        } else if v.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                            self.ident_value(v, depth + 1)
                        } else {
                            // A parenthesized/compound macro body (`(A && B)`).
                            i64::from(eval_pp_cond(v, self.defs))
                        }
                    }
                    _ => 0,
                }
            }
            fn primary(&mut self) -> i64 {
                match self.t.get(self.i).cloned() {
                    Some(T::Num(n)) => {
                        self.i += 1;
                        n
                    }
                    Some(T::Id(id)) if id == "defined" => {
                        self.i += 1;
                        let paren = self.eat("(");
                        let name = match self.t.get(self.i).cloned() {
                            Some(T::Id(n)) => n,
                            other => panic!("defined without identifier: {other:?}"),
                        };
                        self.i += 1;
                        if paren {
                            assert!(self.eat(")"), "unbalanced defined(");
                        }
                        i64::from(self.defs.contains_key(&name))
                    }
                    Some(T::Id(id)) => {
                        self.i += 1;
                        // Function-like macro call in a condition: no kernel
                        // source here uses one; treat as 0.
                        let v = self.ident_value(&id, 0);
                        if self.peek() == Some(&T::Op("(")) {
                            let mut depth = 0;
                            while let Some(t) = self.t.get(self.i) {
                                match t {
                                    T::Op("(") => depth += 1,
                                    T::Op(")") => {
                                        depth -= 1;
                                        if depth == 0 {
                                            self.i += 1;
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                                self.i += 1;
                            }
                            return 0;
                        }
                        v
                    }
                    Some(T::Op("(")) => {
                        self.i += 1;
                        let v = self.or();
                        assert!(self.eat(")"), "unbalanced (");
                        v
                    }
                    Some(T::Op("!")) => {
                        self.i += 1;
                        i64::from(self.primary() == 0)
                    }
                    Some(T::Op("-")) => {
                        self.i += 1;
                        -self.primary()
                    }
                    other => panic!("bad #if expression token {other:?}"),
                }
            }
            fn mul(&mut self) -> i64 {
                let mut v = self.primary();
                loop {
                    if self.eat("*") {
                        v *= self.primary();
                    } else if self.eat("/") {
                        let d = self.primary();
                        v = if d == 0 { 0 } else { v / d };
                    } else if self.eat("%") {
                        let d = self.primary();
                        v = if d == 0 { 0 } else { v % d };
                    } else {
                        return v;
                    }
                }
            }
            fn add(&mut self) -> i64 {
                let mut v = self.mul();
                loop {
                    if self.eat("+") {
                        v += self.mul();
                    } else if self.eat("-") {
                        v -= self.mul();
                    } else {
                        return v;
                    }
                }
            }
            fn rel(&mut self) -> i64 {
                let mut v = self.add();
                loop {
                    if self.eat("<=") {
                        v = i64::from(v <= self.add());
                    } else if self.eat(">=") {
                        v = i64::from(v >= self.add());
                    } else if self.eat("<") {
                        v = i64::from(v < self.add());
                    } else if self.eat(">") {
                        v = i64::from(v > self.add());
                    } else {
                        return v;
                    }
                }
            }
            fn eq(&mut self) -> i64 {
                let mut v = self.rel();
                loop {
                    if self.eat("==") {
                        v = i64::from(v == self.rel());
                    } else if self.eat("!=") {
                        v = i64::from(v != self.rel());
                    } else {
                        return v;
                    }
                }
            }
            fn and(&mut self) -> i64 {
                let mut v = self.eq();
                while self.eat("&&") {
                    let r = self.eq();
                    v = i64::from(v != 0 && r != 0);
                }
                v
            }
            fn or(&mut self) -> i64 {
                let mut v = self.and();
                while self.eat("||") {
                    let r = self.and();
                    v = i64::from(v != 0 || r != 0);
                }
                v
            }
        }
        let mut p = P {
            t: toks,
            i: 0,
            defs,
        };
        let v = p.or();
        assert_eq!(p.i, p.t.len(), "trailing tokens in #if {expr:?}");
        v != 0
    }

    /// Names of the `__global__` kernels `src` defines after preprocessing with
    /// `predefs` (`-D` style `(name, value)` pairs).
    fn active_kernel_names(
        src: &str,
        predefs: &[(String, String)],
    ) -> std::collections::BTreeSet<String> {
        let stripped = strip_c_comments(src);
        // Logical lines (backslash continuation).
        let mut lines: Vec<String> = Vec::new();
        let mut cur = String::new();
        for raw in stripped.lines() {
            if let Some(head) = raw.strip_suffix('\\') {
                cur.push_str(head);
                cur.push(' ');
            } else {
                cur.push_str(raw);
                lines.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
        let mut defs: HashMap<String, String> = predefs.iter().cloned().collect();
        let mut fdefs: HashMap<String, (Vec<String>, String)> = HashMap::new();
        let mut stack: Vec<(bool, bool, bool)> = Vec::new(); // parent, taken, active
        let mut text = String::new();
        let mut out = std::collections::BTreeSet::new();
        let active = |st: &Vec<(bool, bool, bool)>| st.last().map_or(true, |s| s.2);
        for line in &lines {
            let t = line.trim_start();
            let Some(d) = t.strip_prefix('#') else {
                if active(&stack) {
                    text.push_str(line);
                    text.push('\n');
                }
                continue;
            };
            let d = d.trim_start();
            let name_end = d
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(d.len());
            let (kw, rest) = (&d[..name_end], d[name_end..].trim());
            let parent = active(&stack);
            if parent && matches!(kw, "define" | "undef") && !text.is_empty() {
                // Expand/resolve pending text under the macros as they stand
                // before this directive changes them.
                scan_kernels(&text, &defs, &fdefs, &mut out, 0);
                text.clear();
            }
            match kw {
                "if" => {
                    let a = parent && eval_pp_cond(rest, &defs);
                    stack.push((parent, a, a));
                }
                "ifdef" => {
                    let a =
                        parent && defs.contains_key(rest.split_whitespace().next().unwrap_or(""));
                    stack.push((parent, a, a));
                }
                "ifndef" => {
                    let a =
                        parent && !defs.contains_key(rest.split_whitespace().next().unwrap_or(""));
                    stack.push((parent, a, a));
                }
                "elif" => {
                    let top = stack.last_mut().expect("#elif without #if");
                    if top.0 && !top.1 && eval_pp_cond(rest, &defs) {
                        top.2 = true;
                        top.1 = true;
                    } else {
                        top.2 = false;
                    }
                }
                "else" => {
                    let top = stack.last_mut().expect("#else without #if");
                    top.2 = top.0 && !top.1;
                    top.1 = true;
                }
                "endif" => {
                    stack.pop().expect("#endif without #if");
                }
                "define" if parent => {
                    let nend = rest
                        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .unwrap_or(rest.len());
                    let name = rest[..nend].to_owned();
                    let after = &rest[nend..];
                    if let Some(args) = after.strip_prefix('(') {
                        let close = args.find(')').expect("unterminated macro parameter list");
                        let params = args[..close]
                            .split(',')
                            .map(|p| p.trim().to_owned())
                            .filter(|p| !p.is_empty())
                            .collect();
                        fdefs.insert(name, (params, args[close + 1..].trim().to_owned()));
                    } else {
                        defs.insert(name, after.trim().to_owned());
                    }
                }
                "undef" if parent => {
                    defs.remove(rest.split_whitespace().next().unwrap_or(""));
                    fdefs.remove(rest.split_whitespace().next().unwrap_or(""));
                }
                _ => {}
            }
        }
        assert!(stack.is_empty(), "unbalanced preprocessor conditionals");
        scan_kernels(&text, &defs, &fdefs, &mut out, 0);
        out
    }

    fn scan_kernels(
        text: &str,
        defs: &HashMap<String, String>,
        fdefs: &HashMap<String, (Vec<String>, String)>,
        out: &mut std::collections::BTreeSet<String>,
        depth: u32,
    ) {
        let ids = c_idents(text);
        for (n, &(s, e)) in ids.iter().enumerate() {
            let word = &text[s..e];
            if word == "__global__" {
                // `void NAME (`
                let mut j = n + 1;
                while j < ids.len() && &text[ids[j].0..ids[j].1] != "void" {
                    j += 1;
                    if j > n + 12 {
                        break;
                    }
                }
                let Some(&(ns, ne)) = ids.get(j + 1) else {
                    continue;
                };
                if ids.get(j).map_or(true, |&(a, b)| &text[a..b] != "void") {
                    continue;
                }
                if !text[ne..].trim_start().starts_with('(') {
                    continue;
                }
                let mut name = text[ns..ne].to_owned();
                for _ in 0..6 {
                    match defs.get(&name) {
                        Some(v) if !v.trim().is_empty() => name = v.trim().to_owned(),
                        _ => break,
                    }
                }
                out.insert(name);
            } else if let Some((params, body)) = fdefs.get(word) {
                // Only generator macros matter: a body that defines a kernel or
                // invokes another function-like macro (which may).
                let nests = c_idents(body)
                    .iter()
                    .any(|&(a, b)| fdefs.contains_key(&body[a..b]));
                if depth > 4 || !(body.contains("__global__") || nests) {
                    continue;
                }
                let rest = &text[e..];
                if !rest.trim_start().starts_with('(') {
                    continue;
                }
                let open = e + rest.find('(').unwrap();
                let mut d = 0;
                let mut args: Vec<String> = Vec::new();
                let mut cur = String::new();
                let mut close = None;
                for (k, ch) in text[open..].char_indices() {
                    match ch {
                        '(' => {
                            d += 1;
                            if d > 1 {
                                cur.push(ch);
                            }
                        }
                        ')' => {
                            d -= 1;
                            if d == 0 {
                                close = Some(open + k);
                                break;
                            }
                            cur.push(ch);
                        }
                        ',' if d == 1 => args.push(std::mem::take(&mut cur)),
                        _ => cur.push(ch),
                    }
                }
                if close.is_none() {
                    continue;
                }
                args.push(cur);
                let args: Vec<String> = args.iter().map(|a| a.trim().to_owned()).collect();
                if args.len() != params.len() {
                    continue;
                }
                // Substitute parameters token-wise, then paste `##`.
                let mut exp = String::new();
                let mut last = 0;
                for (a, b) in c_idents(body) {
                    exp.push_str(&body[last..a]);
                    match params.iter().position(|p| p == &body[a..b]) {
                        Some(pi) => exp.push_str(&args[pi]),
                        None => exp.push_str(&body[a..b]),
                    }
                    last = b;
                }
                exp.push_str(&body[last..]);
                let exp = exp.split("##").map(str::trim).collect::<Vec<_>>().join("");
                scan_kernels(&exp, defs, fdefs, out, depth + 1);
            }
        }
    }

    fn entry_predefs(arch: &str, flags: &[String]) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = vec![("__HIP_DEVICE_COMPILE__".into(), "1".into())];
        let (iso, fam) = match arch {
            "gfx1100" | "gfx1151" | "gfx1201" => (
                arch,
                if arch == "gfx1201" {
                    "__GFX12__"
                } else {
                    "__GFX11__"
                },
            ),
            _ => (arch, "__GFX9__"),
        };
        v.push((format!("__{iso}__"), "1".into()));
        v.push((fam.into(), "1".into()));
        for f in flags {
            if let Some(d) = f.strip_prefix("-D") {
                let (n, val) = d.split_once('=').unwrap_or((d, "1"));
                v.push((n.to_owned(), val.to_owned()));
            }
        }
        v
    }

    /// Kernels a shared source defines on an arch that no entry of its source
    /// group declares (`arch\tmodule\tsymbol\treason`, `#` comments): variants
    /// the launchers reach only through developer opt-ins or through a module
    /// name another entry carries. Shrink-only: a row must still be a real gap.
    const UNDECLARED_KERNEL_CENSUS: &str =
        include_str!("../tests/fixtures/undeclared-kernel-census.tsv");

    /// Gaps keyed `(arch, first module of the source group, symbol)`. Entries
    /// that share one source text on an arch (the repo convention of one module
    /// per symbol over a shared translation unit) form a group: a kernel is
    /// covered when any entry of its group declares it, because each module's
    /// index only has to contain the symbols requested through that module.
    fn undeclared_kernels() -> std::collections::BTreeMap<(String, String, String), ()> {
        let mut gaps = std::collections::BTreeMap::new();
        for arch in SUPPORTED_ARCHES {
            let mut groups: std::collections::BTreeMap<String, Vec<KernelEntry>> =
                Default::default();
            for entry in entries(arch, "").unwrap() {
                groups
                    .entry(entry.source().to_owned())
                    .or_default()
                    .push(entry);
            }
            for group in groups.values() {
                let first = group.iter().map(|e| e.module).min().unwrap();
                let defined =
                    active_kernel_names(group[0].source(), &entry_predefs(arch, &group[0].flags));
                for name in defined {
                    if !group.iter().any(|e| e.symbols.contains(&name.as_str())) {
                        gaps.insert((arch.to_string(), first.to_string(), name), ());
                    }
                }
            }
        }
        gaps
    }

    /// The converse: a declared symbol the source does not define for the arch
    /// would put a phantom export in the packaged index.
    #[test]
    fn every_declared_kernel_is_defined_by_its_source() {
        let mut bad = Vec::new();
        for arch in SUPPORTED_ARCHES {
            for e in entries(arch, "").unwrap() {
                let defined = active_kernel_names(e.source(), &entry_predefs(arch, &e.flags));
                for sym in e.symbols {
                    if !defined.contains(*sym) {
                        bad.push((arch.to_string(), e.module, sym.to_string()));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "declared but not defined (arch, module, symbol): {bad:#?}"
        );
    }

    #[test]
    fn every_defined_kernel_is_declared_by_its_entry() {
        let gaps = undeclared_kernels();
        let mut allowed = std::collections::BTreeSet::new();
        for line in UNDECLARED_KERNEL_CENSUS
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let [arch, module, symbol, reason] = line.split('\t').collect::<Vec<_>>()[..] else {
                panic!("malformed census row {line:?}");
            };
            assert!(
                !reason.trim().is_empty(),
                "census row without reason: {line:?}"
            );
            assert!(
                allowed.insert((arch.to_owned(), module.to_owned(), symbol.to_owned())),
                "duplicate census row {line:?}"
            );
        }
        let missing: Vec<_> = gaps.keys().filter(|g| !allowed.contains(*g)).collect();
        assert!(
            missing.is_empty(),
            "kernels defined by an entry's source but declared by no entry of its source group \
             (arch, module, symbol): {missing:#?}"
        );
        let stale: Vec<_> = allowed.iter().filter(|g| !gaps.contains_key(*g)).collect();
        assert!(
            stale.is_empty(),
            "stale undeclared-kernel census rows (the gap no longer exists): {stale:#?}"
        );
    }

    /// `tensor_ops` is one module requested with a different symbol per
    /// launcher (`ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)`),
    /// and the installed object is only accepted when every requested symbol is
    /// in its index, so its declaration is exactly the set of kernels the
    /// source defines for the arch (arch/flag-guarded exports are compiled out).
    #[test]
    fn tensor_ops_declares_exactly_the_kernels_its_source_defines() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let entry = lookup(arch, "tensor_ops", "").unwrap();
            let defined = active_kernel_names(entry.source(), &entry_predefs(arch, &entry.flags));
            let declared: std::collections::BTreeSet<String> =
                entry.symbols.iter().map(|s| (*s).to_owned()).collect();
            assert_eq!(
                defined.difference(&declared).collect::<Vec<_>>(),
                Vec::<&String>::new(),
                "{arch}: tensor_ops source defines kernels the entry does not declare"
            );
            assert_eq!(
                declared.difference(&defined).collect::<Vec<_>>(),
                Vec::<&String>::new(),
                "{arch}: tensor_ops declares kernels its source does not define"
            );
            for symbol in [
                "indexed_attention_attention_q8_batched",
                "indexed_attention_attention_q8_batched_hg4",
                "indexed_attention_attention_q8_batched_hg4_lite",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_decode_prologue_q8",
            ] {
                assert!(declared.contains(symbol), "{arch}: {symbol}");
            }
        }
    }

    /// The scanner itself: guards, macro-resolved names and `##` generators.
    #[test]
    fn kernel_scanner_handles_guards_macros_and_generators() {
        let src = r#"
#define GEN(BV) \
extern "C" __global__ void tile_bt##BV(int* p) {}
#define NAMED my_kernel
#if defined(__GFX12__) && !defined(NO_X)
extern "C" __global__ void only12(int* p) {}
GEN(4)
GEN(8)
#elif defined(__GFX11__)
extern "C" __global__ void only11(int* p) {}
#else
extern "C" __global__ void other(int* p) {}
#endif
extern "C" __global__ void NAMED(int* p) {}
// extern "C" __global__ void commented(int* p) {}
"#;
        let g12 = active_kernel_names(src, &entry_predefs("gfx1201", &[]));
        assert_eq!(
            g12.iter().map(String::as_str).collect::<Vec<_>>(),
            ["my_kernel", "only12", "tile_bt4", "tile_bt8"]
        );
        let g11 = active_kernel_names(src, &entry_predefs("gfx1151", &[]));
        assert_eq!(
            g11.iter().map(String::as_str).collect::<Vec<_>>(),
            ["my_kernel", "only11"]
        );
        let flagged = active_kernel_names(src, &entry_predefs("gfx1201", &["-DNO_X".to_string()]));
        assert_eq!(
            flagged.iter().map(String::as_str).collect::<Vec<_>>(),
            ["my_kernel", "other"]
        );
    }

    const FN_TRACE: &str = include_str!("../tests/fixtures/kernel-trace-qwen4-flash-next.tsv");
    const Q35_TRACE: &str = include_str!("../tests/fixtures/kernel-trace-qwen35.tsv");

    fn predict(
        arch: &str,
        model_arch: Option<u32>,
        pm_decode: bool,
        planned: &RouteKernelPlan,
    ) -> PredictedKernels {
        predicted_kernels(
            &PredictKernelInput {
                arch,
                model_arch,
                hipcc_extra_flags: "",
                pm_decode,
                exact_plan: true,
            },
            planned,
        )
        .unwrap()
    }

    fn no_plan() -> RouteKernelPlan {
        RouteKernelPlan {
            entries: Vec::new(),
        }
    }

    /// An exact open plan of a traced family compiles the arch's WHOLE
    /// inventory minus the plan: the open plan's walks are incomplete by
    /// definition, so no traced subset is trusted to be enough. Every recipe
    /// is the registry's own for that arch, modules the trace and walks never
    /// saw (another codec, a newly registered kernel) are in, and only
    /// trace- or walk-proven symbols bind.
    #[test]
    fn exact_open_plan_compiles_whole_inventory_and_binds_only_proven() {
        for (arch, model_arch, fixture) in [
            ("gfx1100", 16, FN_TRACE),
            ("gfx1151", 16, FN_TRACE),
            ("gfx1201", 16, FN_TRACE),
            ("gfx1100", 5, Q35_TRACE),
            ("gfx1151", 5, Q35_TRACE),
            ("gfx1201", 5, Q35_TRACE),
        ] {
            let predicted = predict(arch, Some(model_arch), false, &no_plan());
            assert_eq!(
                predicted.basis,
                PredictionBasis::ArchInventoryFamilyBindings,
                "{arch} {model_arch}"
            );
            assert_eq!(
                predicted.basis.label(),
                "arch-inventory (family-proven bindings)"
            );
            let inventory = entries(arch, "").unwrap();
            let mut distinct: Vec<_> = inventory.iter().map(|e| e.module).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                predicted.modules.len() + predicted.omitted.len(),
                distinct.len(),
                "{arch} {model_arch}: every inventory module is predicted or omitted with a reason"
            );
            let traced = traced_symbols(fixture, arch);
            assert!(
                predicted
                    .modules
                    .iter()
                    .any(|m| !traced.contains_key(m.entry.module)),
                "{arch} {model_arch}: modules the trace never saw must still compile"
            );
            for module in &predicted.modules {
                let registry = inventory
                    .iter()
                    .find(|e| e.module == module.entry.module)
                    .unwrap();
                let name = module.entry.module;
                assert_eq!(module.entry.arch, arch);
                assert_eq!(module.entry.source, registry.source, "{arch} {name}");
                assert_eq!(module.entry.flags, registry.flags, "{arch} {name}");
                assert_eq!(module.entry.symbols, registry.symbols, "{arch} {name}");
                assert_eq!(module.entry.scheduler_profile, registry.scheduler_profile);
                // Modules the trace never saw bind nothing.
                let walked = walked_route_plans()
                    .unwrap()
                    .iter()
                    .filter(|p| p.model_arch == model_arch && p.arch == arch)
                    .any(|p| p.rows.iter().any(|r| r.module == name));
                if !traced.contains_key(name) && !walked {
                    assert!(module.bind.is_empty(), "{arch} {name}: unproven bind");
                }
            }
            let mut names: Vec<_> = predicted.modules.iter().map(|m| m.entry.module).collect();
            let len = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), len, "{arch}: duplicate predicted module");
        }
    }

    /// Every HIP row of the walked plans for the family and arch is
    /// predicted (or omitted with a reason) even when the trace lacks it.
    #[test]
    fn predicted_family_includes_walked_hip_rows() {
        let predicted = predict("gfx1201", Some(16), false, &no_plan());
        let mut checked = 0;
        for plan in walked_route_plans()
            .unwrap()
            .iter()
            .filter(|p| p.model_arch == 16 && p.arch == "gfx1201")
        {
            for row in plan.rows.iter().filter(|row| !row.embedded) {
                assert!(
                    predicted
                        .modules
                        .iter()
                        .any(|m| m.entry.module == row.module)
                        || predicted.omitted.iter().any(|(m, _)| *m == row.module),
                    "walked module {} missing from the prediction",
                    row.module
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no walked HIP rows checked");
    }

    /// Every row of every checked-in walked plan (each layout, arch and spec
    /// route, including plans with no header fixture such as the split mq2
    /// Flash-Next plan) resolves the way `walked_route_entries_in` resolves
    /// it at load: a HIP row to exactly one inventory recipe with the walked
    /// source SHA-256 and every walked symbol (so the release pack serves
    /// it), an embedded row to a runtime image with the walked image SHA-256.
    #[test]
    fn every_walked_plan_row_resolves_with_its_walked_identity() {
        let plans = walked_route_plans().unwrap();
        assert!(plans.len() >= 3, "walked plans missing");
        for plan in plans {
            let flags = default_feature_flags(plan.arch).unwrap();
            let inventory = entries(plan.arch, &flags.hipcc_extra_flags).unwrap();
            for native_mtp in [false, true] {
                let rows: Vec<_> = plan
                    .rows
                    .iter()
                    .filter(|row| row.native_mtp == native_mtp)
                    .collect();
                let what = format!("{} {} native_mtp={native_mtp}", plan.layout, plan.arch);
                assert!(rows.iter().any(|row| !row.embedded), "{what}: no HIP rows");
                for row in rows {
                    if row.embedded {
                        let (_, image, _) =
                            walked_embedded_image(plan.arch, row.module, row.sha256)
                                .unwrap_or_else(|| panic!("{what}: no image for {}", row.module));
                        assert_eq!(
                            format!("{:x}", Sha256::digest(image)),
                            row.sha256,
                            "{what}: {} image",
                            row.module
                        );
                        continue;
                    }
                    let found: Vec<_> = inventory
                        .iter()
                        .filter(|e| e.module == row.module)
                        .collect();
                    assert_eq!(found.len(), 1, "{what}: {} recipes", row.module);
                    // Beta binds HIP rows to the current registry recipe, not
                    // the historical walk-time source digest. Preserve that
                    // provenance in the fixture; only embedded images require
                    // their recorded bytes at admission (checked above).
                    for symbol in &row.symbols {
                        assert!(
                            found[0].symbols.contains(symbol),
                            "{what}: {} lacks {symbol}",
                            row.module
                        );
                    }
                }
            }
        }
    }

    /// The two gfx1100 kernels of the MQ6-residual PM twin and the q8 HG4 low-LDS QSA twin are
    /// installable without a device compiler. The Flash-Next gfx1100 route has no closed plan (it
    /// loads as a predicted fallback: the arch's whole inventory is compiled before ready and
    /// nothing binds eagerly, so every first dispatch binds lazily with one warning, planned or
    /// not), so what a strict / pack-only install needs is: `tensor_ops` is in the gfx1100
    /// pack (registry entry exporting the lite symbol, present in the predicted compile set,
    /// kill-switched twin and incumbent in one module), and the PM image is an embedded
    /// gfx1100 image that resolves by its recorded SHA-256 and carries both entry symbols.
    #[test]
    fn gfx1100_pack_ships_mq6_resid_pm_and_qsa_hg4_lite() {
        use sha2::{Digest, Sha256};
        const LITE: &str = "indexed_attention_attention_q8_batched_hg4_lite";
        let inventory = entries("gfx1100", "").unwrap();
        let tensor_ops = inventory
            .iter()
            .find(|e| e.module == "tensor_ops")
            .expect("tensor_ops is a gfx1100 pack module");
        assert!(tensor_ops.symbols.contains(&LITE));
        assert!(tensor_ops
            .symbols
            .contains(&"indexed_attention_attention_q8_batched_hg4"));
        assert!(
            tensor_ops.source().contains(LITE),
            "pack source lacks the lite kernel"
        );
        let predicted = predict("gfx1100", Some(16), false, &no_plan());
        assert!(
            predicted
                .modules
                .iter()
                .any(|m| m.entry.module == "tensor_ops" && m.entry.symbols.contains(&LITE)),
            "the Flash-Next gfx1100 predicted set must compile tensor_ops with the lite symbol"
        );
        // The PM image is embedded: nothing to compile, so it is in no HIP module list.
        const PM_MODULE: &str = "gemm_mq6g256v2_residual_pm_gfx1100";
        assert!(!inventory.iter().any(|e| e.module == PM_MODULE));
        let image = kernels::GEMM_MQ6G256V2_RESIDUAL_PM_GFX1100;
        let sha = format!("{:x}", Sha256::digest(image));
        assert_eq!(
            sha,
            kernels::GEMM_MQ6G256V2_RESIDUAL_PM_GFX1100_SHA256.trim()
        );
        assert_eq!(walked_arch_image("gfx1100", &sha), Some(image));
        for other in ["gfx1151", "gfx1201"] {
            assert!(walked_arch_image(other, &sha).is_none(), "{other}");
        }
        for symbol in [
            "gemm_mq6g256v2_residual_pm_gfx1100_w4",
            "gemm_mq6g256v2_residual_pm_gfx1100_w8",
        ] {
            assert!(
                image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                "{symbol} missing from the embedded image"
            );
        }
    }

    /// The two hipcc MQ6-residual "pk twin" ports are a registry module of the exact-gfx1100 pack
    /// only: one module exporting both launch symbols, source carrying both kernels, in the
    /// predicted compile set of the gfx1100 Flash-Next route, and on no other arch (other arches'
    /// inventories and bytes are unchanged). Their incumbent (`mw4_lds` / `mw8_lds`) stays in its
    /// own module as the kill-switch fallback.
    #[test]
    fn gfx1100_pack_ships_mq6_resid_port() {
        const MODULE: &str = "gemm_mq6g256v2_residual_gfx1100_port";
        const SYMBOLS: [&str; 2] = [
            "gemm_mq6g256v2_residual_gfx1100_port_w4n128",
            "gemm_mq6g256v2_residual_gfx1100_port_w8n256",
        ];
        let inventory = entries("gfx1100", "").unwrap();
        let port = inventory
            .iter()
            .find(|e| e.module == MODULE)
            .expect("the port module is a gfx1100 pack module");
        assert_eq!(port.symbols, SYMBOLS.as_slice());
        for symbol in SYMBOLS {
            assert!(
                port.source().contains(&format!("{symbol}(")),
                "pack source lacks {symbol}"
            );
        }
        let mw = inventory
            .iter()
            .find(|e| e.module == "gemm_mqv2_wmma_gfx1100_mw_lds")
            .expect("the hipcc MW module stays in the gfx1100 pack (kill-switch fallback)");
        assert!(mw
            .symbols
            .contains(&"gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"));
        let predicted = predict("gfx1100", Some(16), false, &no_plan());
        assert!(
            predicted
                .modules
                .iter()
                .any(|m| m.entry.module == MODULE && m.entry.symbols == SYMBOLS.as_slice()),
            "the Flash-Next gfx1100 predicted set must compile the port module with both symbols"
        );
        for other in ["gfx1151", "gfx1201", "gfx1101", "gfx942"] {
            if let Ok(other_entries) = entries(other, "") {
                assert!(
                    !other_entries.iter().any(
                        |e| e.module == MODULE || e.symbols.iter().any(|s| SYMBOLS.contains(s))
                    ),
                    "{other} must not carry the gfx1100 port"
                );
            }
        }
    }

    /// The gfx1100 HC `input_mix_down` fill tile (`gemm_wmma_lds_80_32_80_16_k64_p`) is an export of
    /// the `qwen4_gemm_wmma_lds256` module of the exact-gfx1100 pack, next to its incumbent
    /// (`gemm_wmma_lds_64_64_32_64_k64_p`, the kill-switch fallback), compiled from the module
    /// source under a `__gfx1100__` guard so the gfx1151 object (same source) is unchanged; no
    /// other arch's inventory lists it.
    #[test]
    fn gfx1100_pack_ships_hc_mix_fill80() {
        const MODULE: &str = "qwen4_gemm_wmma_lds256";
        const FILL: &str = "gemm_wmma_lds_80_32_80_16_k64_p";
        const INCUMBENT: &str = "gemm_wmma_lds_64_64_32_64_k64_p";
        let inventory = entries("gfx1100", "").unwrap();
        let lds = inventory
            .iter()
            .find(|e| e.module == MODULE)
            .expect("qwen4_gemm_wmma_lds256 is a gfx1100 pack module");
        assert!(lds.symbols.contains(&FILL));
        assert!(lds.symbols.contains(&INCUMBENT));
        assert!(
            lds.source().contains(&format!("WLDS_KERNEL({FILL},")),
            "pack source lacks the fill kernel instantiation"
        );
        assert!(
            lds.source().contains("#if defined(__gfx1100__)"),
            "the fill kernel must stay gfx1100-only"
        );
        let predicted = predict("gfx1100", Some(16), false, &no_plan());
        assert!(
            predicted
                .modules
                .iter()
                .any(|m| m.entry.module == MODULE && m.entry.symbols.contains(&FILL)),
            "the Flash-Next gfx1100 predicted set must compile the module with the fill symbol"
        );
        for other in ["gfx1151", "gfx1201", "gfx1101", "gfx942"] {
            if let Ok(other_entries) = entries(other, "") {
                assert!(
                    !other_entries.iter().any(|e| e.symbols.contains(&FILL)),
                    "{other} must not list the gfx1100 fill tile"
                );
            }
        }
    }

    /// A traced family on an arch its fixture has no rows for is never
    /// completed from another arch's names: the whole inventory, compile-only.
    #[test]
    fn traced_family_without_arch_rows_compiles_whole_inventory_without_borrowing() {
        assert!(
            traced_symbols(FN_TRACE, "gfx906").is_empty(),
            "fixture grew gfx906 rows"
        );
        let predicted = predict("gfx906", Some(16), false, &no_plan());
        assert_eq!(predicted.basis, PredictionBasis::ArchInventoryCompileOnly);
        assert_eq!(predicted.family, "qwen3.8-flash-next");
        assert_eq!(predicted.bound_symbols(), 0);
        let inventory = entries("gfx906", "").unwrap();
        for module in &predicted.modules {
            let registry = inventory
                .iter()
                .find(|e| e.module == module.entry.module)
                .unwrap_or_else(|| panic!("{} is not a gfx906 module", module.entry.module));
            assert_eq!(module.entry.arch, "gfx906");
            assert_eq!(module.entry.source, registry.source);
        }
    }

    /// A family trace proves only the layouts it was walked on: without an
    /// exact recognised plan (`exact_plan` false) every family, arch and spec
    /// is the arch's whole inventory, every recipe, compile-only. No
    /// family-based assumption is universal.
    #[test]
    fn no_exact_plan_is_whole_inventory_for_every_family() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let mut distinct: Vec<_> = entries(arch, "")
                .unwrap()
                .iter()
                .map(|e| e.module)
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            for model_arch in [Some(5), Some(6), Some(16), Some(99), None] {
                let predicted = predicted_kernels(
                    &PredictKernelInput {
                        arch,
                        model_arch,
                        hipcc_extra_flags: "",
                        pm_decode: false,
                        exact_plan: false,
                    },
                    &no_plan(),
                )
                .unwrap();
                assert_eq!(
                    predicted.basis,
                    PredictionBasis::ArchInventoryCompileOnly,
                    "{arch} {model_arch:?}"
                );
                assert_eq!(predicted.bound_symbols(), 0, "{arch} {model_arch:?}");
                assert_eq!(
                    predicted.modules.len() + predicted.omitted.len(),
                    distinct.len(),
                    "{arch} {model_arch:?}: not every inventory recipe"
                );
                // A module the family trace never saw (another codec's) is in.
                let traced = traced_symbols(FN_TRACE, arch);
                assert!(
                    predicted.modules.len() > traced.len(),
                    "{arch} {model_arch:?}: the set must exceed the traced union"
                );
            }
        }
    }

    /// No trustworthy trace: the arch's whole inventory, compile-only.
    #[test]
    fn unknown_family_is_conservative_compile_only() {
        for model_arch in [None, Some(99)] {
            let predicted = predict("gfx1100", model_arch, false, &no_plan());
            assert_eq!(predicted.basis, PredictionBasis::ArchInventoryCompileOnly);
            assert_eq!(predicted.basis.label(), "arch-inventory compile-only");
            let mut distinct: Vec<_> = entries("gfx1100", "")
                .unwrap()
                .iter()
                .map(|e| e.module)
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                predicted.modules.len() + predicted.omitted.len(),
                distinct.len()
            );
            assert_eq!(predicted.bound_symbols(), 0);
            assert!(predicted.modules.iter().all(|m| m.bind.is_empty()));
        }
        assert!(matches!(
            predicted_kernels(
                &PredictKernelInput {
                    arch: "gfx000",
                    model_arch: None,
                    hipcc_extra_flags: "",
                    pm_decode: false,
                    exact_plan: true
                },
                &no_plan()
            ),
            Err(RegistryError::UnsupportedArch(_))
        ));
    }

    /// Bound symbols have exactly one owning module in the whole registry
    /// corpus, and no accepted embedded twin exports them.
    #[test]
    fn predicted_binds_are_conflict_free() {
        for (arch, model_arch) in [
            ("gfx1201", 16),
            ("gfx1201", 5),
            ("gfx1151", 16),
            ("gfx1100", 5),
        ] {
            let predicted = predict(arch, Some(model_arch), false, &no_plan());
            let mut corpus = route_corpus(arch, "").unwrap();
            corpus.extend(railgun_entries(arch, "").unwrap());
            let mut bound = 0;
            for module in &predicted.modules {
                for &symbol in &module.bind {
                    let owners: std::collections::BTreeSet<_> = corpus
                        .iter()
                        .filter(|e| e.symbols.contains(&symbol))
                        .map(|e| e.module)
                        .collect();
                    assert_eq!(
                        owners.iter().copied().collect::<Vec<_>>(),
                        vec![module.entry.module],
                        "{arch} {model_arch}: {symbol} has several owners"
                    );
                    assert!(module.entry.symbols.contains(&symbol));
                    // Only symbols a trace row or a walked row observed.
                    let fixture = if model_arch == 16 {
                        FN_TRACE
                    } else {
                        Q35_TRACE
                    };
                    let traced = traced_symbols(fixture, arch)
                        .get(module.entry.module)
                        .is_some_and(|set| set.contains(symbol));
                    let walked = walked_route_plans()
                        .unwrap()
                        .iter()
                        .filter(|p| p.model_arch == model_arch && p.arch == arch)
                        .flat_map(|p| p.rows.iter())
                        .any(|row| {
                            row.module == module.entry.module && row.symbols.contains(&symbol)
                        });
                    assert!(
                        traced || walked,
                        "{arch} {model_arch}: {symbol} was never observed"
                    );
                    assert!(
                        !crate::pm_decode_twins::GFX1201_TWINS
                            .iter()
                            .any(|t| t.symbols.contains(&symbol)),
                        "{arch}: {symbol} is also an embedded twin export"
                    );
                    bound += 1;
                }
            }
            assert_eq!(bound, predicted.bound_symbols());
            assert!(bound > 0, "{arch} {model_arch}: nothing bound");
        }
    }

    /// A module the route plan already carries is not predicted again, and a
    /// symbol the plan binds from another module is never bound from the
    /// predicted one.
    #[test]
    fn predicted_binds_never_collide_with_the_route_plan() {
        let baseline = predict("gfx1151", Some(5), false, &no_plan());
        let victim = baseline
            .modules
            .iter()
            .find(|m| !m.bind.is_empty())
            .expect("a bindable module");
        let victim_module = victim.entry.module;
        let victim_symbols = victim.entry.symbols;

        let planned = RouteKernelPlan {
            entries: vec![PlannedKernel::Hip(victim.entry.clone())],
        };
        let with_module = predict("gfx1151", Some(5), false, &planned);
        assert!(with_module
            .modules
            .iter()
            .all(|m| m.entry.module != victim_module));
        assert!(with_module
            .omitted
            .contains(&(victim_module, "already in the route plan")));
        assert_eq!(with_module.modules.len() + 1, baseline.modules.len());

        let mut foreign = victim.entry.clone();
        foreign.module = "route_plan_foreign_owner";
        let planned = RouteKernelPlan {
            entries: vec![PlannedKernel::Hip(foreign)],
        };
        let with_symbol = predict("gfx1151", Some(5), false, &planned);
        let kept = with_symbol
            .modules
            .iter()
            .find(|m| m.entry.module == victim_module)
            .expect("the module still compiles");
        assert!(
            kept.bind.iter().all(|s| !victim_symbols.contains(s)),
            "a symbol the plan binds elsewhere stayed bound"
        );
    }

    /// An accepted pm_decode twin replaces its HIP module: it is neither
    /// compiled nor bound as a predicted module.
    #[test]
    fn predicted_set_skips_accepted_pm_decode_twins() {
        let with_twins = predict("gfx1201", Some(16), true, &no_plan());
        for twin in crate::pm_decode_twins::GFX1201_TWINS.iter() {
            assert!(
                with_twins
                    .modules
                    .iter()
                    .all(|m| m.entry.module != twin.module),
                "twin {} predicted as a HIP module",
                twin.module
            );
        }
        let without = predict("gfx1201", Some(16), false, &no_plan());
        let skipped = with_twins
            .omitted
            .iter()
            .filter(|(_, why)| *why == "accepted pm_decode twin")
            .count();
        assert_eq!(with_twins.modules.len() + skipped, without.modules.len());
    }

    /// Tensor table of the dense 27B plus its sidecar native-MTP head.
    fn qwen36_27b_native_mtp_fixture() -> Vec<(String, crate::DType, Vec<usize>)> {
        use crate::DType;
        let mut owned: Vec<(String, DType, Vec<usize>)> = vec![
            (
                "lm_head.weight".into(),
                DType::MQ4G256V2,
                vec![248320, 5120],
            ),
            (
                "model.language_model.embed_tokens.weight".into(),
                DType::Q8_0,
                vec![248320, 5120],
            ),
        ];
        for layer in 0..64 {
            let p = format!("model.language_model.layers.{layer}");
            for name in ["gate", "up", "down"] {
                owned.push((
                    format!("{p}.mlp.{name}_proj.weight"),
                    DType::MQ4G256V2,
                    vec![17408, 5120],
                ));
            }
            if layer % 4 == 3 {
                for name in ["q", "k", "v", "o"] {
                    owned.push((
                        format!("{p}.self_attn.{name}_proj.weight"),
                        DType::MQ4G256V2,
                        vec![12288, 5120],
                    ));
                }
            } else {
                for name in ["qkv", "z", "a", "b"] {
                    owned.push((
                        format!("{p}.linear_attn.in_proj_{name}.weight"),
                        DType::MQ4G256V2,
                        vec![10240, 5120],
                    ));
                }
                owned.push((
                    format!("{p}.linear_attn.out_proj.weight"),
                    DType::MQ4G256V2,
                    vec![5120, 5120],
                ));
            }
        }
        for (name, shape) in [
            ("eh_proj", [5120, 10240]),
            ("wq", [12288, 5120]),
            ("wk", [1024, 5120]),
            ("wv", [1024, 5120]),
            ("wo", [5120, 6144]),
            ("ffn_gate", [17408, 5120]),
            ("ffn_up", [17408, 5120]),
            ("ffn_down", [5120, 17408]),
        ] {
            owned.push((name.into(), DType::MQ4G256, shape.to_vec()));
        }
        owned
    }

    #[test]
    fn native_mtp_plans_pm_fill_before_ready() {
        let owned = qwen36_27b_native_mtp_fixture();
        let tensors: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape)| KernelTensorMeta {
                name,
                dtype: *dtype,
                shape,
            })
            .collect();
        for pm_decode in [false, true] {
            let mut flags = default_feature_flags("gfx1201").unwrap();
            flags.pm_decode = pm_decode;
            let mut input = RouteKernelInput {
                arch: "gfx1201",
                model_arch: 5,
                tensors: &tensors,
                n_heads: 24,
                n_kv_heads: 4,
                head_dim: 256,
                max_seq: 16448,
                prefill_chunk_rows: 512,
                kv_k: "fp8",
                kv_v: "fp8",
                state_quant: "q8",
                spec: KernelSpecRoute::NativeMtp,
                host_mapped_experts: false,
                retained_decode: false,
                hipcc_extra_flags: "",
                flags: &flags,
                route_axes: &[],
            };
            let (module, image, symbols) = crate::pm_mtp_gemm::route_object();
            let plan = route_entries(&input).expect("native MTP closed route");
            assert_eq!(
                route_entries_sealed(&input).unwrap().1,
                RouteSeal::Sealed,
                "the hand-closed 27B route seals"
            );
            assert!(plan
                .entries
                .iter()
                .any(|e| matches!(e, PlannedKernel::Embedded {
                module: m, image: i, symbols: s, ..
            } if *m == module && *i == image && *s == symbols)));
            input.spec = KernelSpecRoute::Ar;
            assert!(!route_entries(&input)
                .unwrap()
                .entries
                .iter()
                .any(|e| e.module() == module));
        }
    }

    /// Resolve the 27B closed route for one (arch, spec, KV pair).
    fn route_27b(
        arch: &str,
        spec: KernelSpecRoute,
        kv: (&str, &str),
    ) -> Result<RouteKernelPlan, RegistryError> {
        let owned = qwen36_27b_native_mtp_fixture();
        let tensors: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape)| KernelTensorMeta {
                name,
                dtype: *dtype,
                shape,
            })
            .collect();
        let flags = default_feature_flags("gfx1201").unwrap();
        route_entries(&RouteKernelInput {
            arch,
            model_arch: 5,
            tensors: &tensors,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            max_seq: 262_144,
            prefill_chunk_rows: 8192,
            kv_k: kv.0,
            kv_v: kv.1,
            state_quant: "q8",
            spec,
            host_mapped_experts: false,
            retained_decode: false,
            hipcc_extra_flags: "",
            flags: &flags,
            route_axes: &[],
        })
    }

    /// Regression: the v0.4.4 sealed 27B AR/MTP plan omitted the native-fp8
    /// split-KV FA2 verifier, so `hipfire bench --spec off` died in the 12-row
    /// tail chunk of a 65,548 / 131,084-token prompt and `--spec mtp` died in
    /// `spec_step` at >= 8192 live context ("outside this load's planned
    /// kernel route"). Every (rows, live context) the dispatch admits to the
    /// split route must find all three launcher entries planned, for both
    /// closed spec routes, under the default route and the explicit-split
    /// developer override (2..=32 rows, no context floor).
    #[test]
    fn closed_27b_plans_cover_the_fp8_split_verify_route() {
        use crate::attention::{
            fa2_gqa_split_gfx1201_symbols, fp8_fa2_split_verify_admission,
            fp8_fa2_split_verify_route_for,
        };
        let (partial, merge, precon, _) = fa2_gqa_split_gfx1201_symbols(true);
        let ctxs = [1, 8191, 8192, 8193, 8204, 65_548, 131_084, 260_000, 262_144];
        let routes = [
            fp8_fa2_split_verify_route_for(None),
            fp8_fa2_split_verify_route_for(Some("8")),
        ];
        let mut admitted = 0;
        for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
            let plan = route_27b("gfx1201", spec, ("fp8", "fp8")).expect("closed 27B route");
            for route in routes {
                for rows in 1..=32 {
                    for ctx in ctxs {
                        let Some(splits) = fp8_fa2_split_verify_admission(
                            route,
                            "gfx1201",
                            ctx,
                            24,
                            4,
                            256,
                            rows,
                            1 << 30,
                        ) else {
                            continue;
                        };
                        admitted += 1;
                        assert!(splits >= 2, "{spec:?} rows {rows} ctx {ctx}");
                        let entry = plan
                            .entries
                            .iter()
                            .find(|e| e.module() == partial)
                            .unwrap_or_else(|| {
                                panic!(
                                    "{spec:?}: rows {rows} at ctx {ctx} dispatch the fp8 \
                                     split-KV FA2 verifier but its module {partial} is not planned"
                                )
                            });
                        for symbol in [partial, merge, precon] {
                            assert!(
                                entry.symbols().contains(&symbol),
                                "{spec:?}: {symbol} missing from the planned {partial} module"
                            );
                        }
                    }
                }
            }
        }
        assert!(admitted > 0, "the sweep never reached the split route");
        // The default route covers exactly the tail/verify rows 2..=16 from
        // 8192 live context (the failing 8192+12 prompt included).
        let default = fp8_fa2_split_verify_route_for(None);
        for rows in 2..=16 {
            assert!(
                fp8_fa2_split_verify_admission(default, "gfx1201", 8204, 24, 4, 256, rows, 1 << 30)
                    .is_some(),
                "rows {rows} at ctx 8204 must reach the split route"
            );
        }
    }

    /// The sealed 27B plan exists only for native fp8 K/V on exact gfx1201;
    /// every other KV mode or arch resolves to no closed route (lazy loading),
    /// so none of them can dispatch the split verifier against a sealed plan.
    #[test]
    fn closed_27b_route_is_refused_outside_native_fp8_gfx1201() {
        for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
            for kv in [
                ("q8", "q8"),
                ("q8_0", "q8_0"),
                ("fp8", "q8"),
                ("q8", "fp8"),
                ("asym4", "asym4"),
                ("asym3", "asym3"),
                ("fwht4", "fwht4"),
                ("fp16", "fp16"),
            ] {
                assert!(
                    matches!(
                        route_27b("gfx1201", spec, kv),
                        Err(RegistryError::UnsupportedRoute(_))
                    ),
                    "{spec:?} {kv:?} must not resolve to a sealed 27B route"
                );
            }
            for arch in [
                "gfx1100", "gfx1151", "gfx1150", "gfx1101", "gfx1102", "gfx1200",
            ] {
                assert!(
                    route_27b(arch, spec, ("fp8", "fp8")).is_err(),
                    "{spec:?} {arch} must not resolve to a sealed 27B route"
                );
            }
        }
    }

    /// With no table configured (the product default) the 27B plan keeps the
    /// whole shipped b1s bundle and plans no `pm_tune:` module: the planner
    /// output is unchanged.
    #[test]
    fn pm_tune_unset_leaves_the_27b_b1s_plan_unchanged() {
        if crate::pm_tune::enabled() {
            return; // a developer table is configured in this environment
        }
        let owned = qwen36_27b_native_mtp_fixture();
        let tensors: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape)| KernelTensorMeta {
                name,
                dtype: *dtype,
                shape,
            })
            .collect();
        let flags = default_feature_flags("gfx1201").unwrap();
        for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
            let input = RouteKernelInput {
                arch: "gfx1201",
                model_arch: 5,
                tensors: &tensors,
                n_heads: 24,
                n_kv_heads: 4,
                head_dim: 256,
                max_seq: 16448,
                prefill_chunk_rows: 512,
                kv_k: "fp8",
                kv_v: "fp8",
                state_quant: "q8",
                spec,
                host_mapped_experts: false,
                retained_decode: false,
                hipcc_extra_flags: "",
                flags: &flags,
                route_axes: &[],
            };
            let plan = route_entries(&input).unwrap();
            assert!(
                !plan
                    .entries
                    .iter()
                    .any(|e| e.module().starts_with("pm_tune:")),
                "{spec:?}: no pm_tune module without a table"
            );
            let b1s = plan
                .entries
                .iter()
                .find(|e| e.module() == QWEN36_27B_GFX1201_B1S_MODULE)
                .expect("b1s bundle planned");
            assert_eq!(b1s.symbols(), QWEN36_27B_GFX1201_B1S_SYMBOLS, "{spec:?}");
            assert!(matches!(
                b1s,
                PlannedKernel::Embedded { image, .. }
                    if *image == crate::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S
            ));
        }
    }

    /// A selected variant takes over exactly the gate_up export: the shipped
    /// bundle keeps its other three symbols and the plan stays valid (one
    /// image per symbol).
    #[test]
    fn pm_tune_variant_takes_only_the_gate_up_export_in_the_plan() {
        static VARIANT: std::sync::LazyLock<crate::pm_tune::PmVariant> =
            std::sync::LazyLock::new(|| crate::pm_tune::PmVariant {
                module: "pm_tune:0123456789abcdef".to_owned(),
                bytes: b"variant image".to_vec(),
                row_id: "rows[0]:test".to_owned(),
            });
        let owned = qwen36_27b_native_mtp_fixture();
        let tensors: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape)| KernelTensorMeta {
                name,
                dtype: *dtype,
                shape,
            })
            .collect();
        let flags = default_feature_flags("gfx1201").unwrap();
        let input = RouteKernelInput {
            arch: "gfx1201",
            model_arch: 5,
            tensors: &tensors,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            max_seq: 16448,
            prefill_chunk_rows: 512,
            kv_k: "fp8",
            kv_v: "fp8",
            state_quant: "q8",
            spec: KernelSpecRoute::Ar,
            host_mapped_experts: false,
            retained_decode: false,
            hipcc_extra_flags: "",
            flags: &flags,
            route_axes: &[],
        };
        let mut plan = route_entries(&input).unwrap();
        apply_gate_up_pm_tune_variant(&mut plan.entries, &VARIANT);
        validate_route_plan(&plan).expect("variant plan is valid");
        let owner_of = |symbol: &str| {
            plan.entries
                .iter()
                .filter(|e| e.symbols().contains(&symbol))
                .map(|e| e.module())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            owner_of(crate::pm_tune::GATE_UP_B1S_SYMBOL),
            ["pm_tune:0123456789abcdef"]
        );
        for symbol in QWEN36_27B_GFX1201_B1S_SYMBOLS_NO_GATE_UP {
            assert_eq!(owner_of(symbol), [QWEN36_27B_GFX1201_B1S_MODULE]);
        }
        assert!(matches!(
            plan.entries.last().unwrap(),
            PlannedKernel::Embedded { image, .. } if *image == b"variant image"
        ));
    }

    // ── Generic (Qwen3.5 A3B/27B) walked-plan matching ──────────────────
    //
    // Synthetic fixtures only: nothing below is a recorded walk. They drive
    // `parse_walked_route_plan` / `walked_route_entries_in` over generated
    // plan text to pin the matcher: layout (trunk vs trunk+head), heads,
    // axes, max_seq, seal, and arch-exact embedded identity.

    type SyntheticHeader = Vec<(String, crate::DType, Vec<usize>)>;

    fn synthetic_trunk() -> SyntheticHeader {
        let mut out = vec![(
            "model.language_model.embed_tokens.weight".to_owned(),
            crate::DType::Q8_0,
            vec![1024, 256],
        )];
        for layer in 0..4 {
            out.push((
                format!("model.language_model.layers.{layer}.mlp.gate_proj.weight"),
                crate::DType::MQ4G256V2,
                vec![512, 256],
            ));
        }
        out
    }

    fn synthetic_with_head() -> SyntheticHeader {
        let mut header = synthetic_trunk();
        header.push(("eh_proj".to_owned(), crate::DType::MQ4G256, vec![256, 512]));
        header
    }

    fn synthetic_metas(header: &SyntheticHeader) -> Vec<KernelTensorMeta<'_>> {
        header
            .iter()
            .map(|(name, dtype, shape)| KernelTensorMeta {
                name,
                dtype: *dtype,
                shape,
            })
            .collect()
    }

    fn synthetic_layout(header: &SyntheticHeader) -> String {
        route_layout_sha256(&synthetic_metas(header))
    }

    fn synthetic_plan_text(
        arch: &str,
        layout: &str,
        mtp_layout: &str,
        sealed: &str,
        rows: &[String],
    ) -> String {
        let mut text = format!(
            "#! family synthetic-qwen35\n#! model_arch 5\n#! arch {arch}\n#! layout {layout}\n\
             #! mtp_layout {mtp_layout}\n#! heads 16,2,256\n#! max_seq 8192\n\
             #! axis kv_backend vmm\n#! axis kv_k q8\n#! axis kv_v q8\n\
             #! axis prefill_chunk_rows 512\n#! axis state_quant q8\n"
        );
        if !sealed.is_empty() {
            text.push_str(&format!("#! sealed {sealed}\n"));
        }
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        text
    }

    /// Plan text of the gfx1100 fixture: AR hip + embedded rows, the same
    /// plus one extra hip row on native MTP, sealed for `ar` only.
    fn main_plan_text() -> String {
        let inventory = entries("gfx1100", "").unwrap();
        let hip = &inventory[0];
        let extra = inventory
            .iter()
            .find(|entry| entry.module != hip.module)
            .unwrap();
        let hip_row = |route: &str, entry: &KernelEntry| {
            format!(
                "{route}\thip\t{}\t{}\t{:x}",
                entry.module,
                entry.symbols.join(","),
                Sha256::digest(entry.source().as_bytes())
            )
        };
        let image_row = |route: &str| {
            format!(
                "{route}\tembedded\tgemm_mq4g256v2_residual_iu4_pm_gfx1100\t\
                 gemm_mq4g256v2_residual_iu4_pm_gfx1100\t{:x}",
                Sha256::digest(kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_GFX1100)
            )
        };
        let rows = [
            hip_row("ar", hip),
            image_row("ar"),
            hip_row("native_mtp", hip),
            image_row("native_mtp"),
            hip_row("native_mtp", extra),
        ];
        synthetic_plan_text(
            "gfx1100",
            &synthetic_layout(&synthetic_trunk()),
            &synthetic_layout(&synthetic_with_head()),
            "ar",
            &rows,
        )
    }

    /// A trunk-layout plan on `arch` whose only row is one embedded image.
    fn embedded_plan_text(arch: &str, module: &str, image: &[u8]) -> String {
        let layout = synthetic_layout(&synthetic_trunk());
        synthetic_plan_text(
            arch,
            &layout,
            &layout,
            "",
            &[format!(
                "ar\tembedded\t{module}\t{module}\t{:x}",
                Sha256::digest(image)
            )],
        )
    }

    /// Parse generated plan text once into a `'static` plan slice (the
    /// planner hands out borrows of it as `'static` symbol lists).
    macro_rules! walked_fixture {
        ($text:expr) => {{
            static TEXT: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| $text);
            static PLANS: std::sync::LazyLock<Vec<WalkedRoutePlan>> =
                std::sync::LazyLock::new(|| vec![parse_walked_route_plan(TEXT.as_str()).unwrap()]);
            PLANS.as_slice()
        }};
    }

    fn synthetic_axes() -> Vec<(&'static str, String)> {
        vec![
            ("kv_backend", "vmm".to_owned()),
            ("kv_k", "q8".to_owned()),
            ("kv_v", "q8".to_owned()),
            ("prefill_chunk_rows", "512".to_owned()),
            ("state_quant", "q8".to_owned()),
        ]
    }

    fn synthetic_input<'a>(
        arch: &'a str,
        tensors: &'a [KernelTensorMeta<'a>],
        spec: KernelSpecRoute,
        flags: &'a crate::feature_flags::FeatureFlags,
        axes: &'a [(&'a str, String)],
    ) -> RouteKernelInput<'a> {
        RouteKernelInput {
            arch,
            model_arch: 5,
            tensors,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            max_seq: 8192,
            prefill_chunk_rows: 512,
            kv_k: "q8",
            kv_v: "q8",
            state_quant: "q8",
            spec,
            host_mapped_experts: false,
            retained_decode: true,
            hipcc_extra_flags: "",
            flags,
            route_axes: axes,
        }
    }

    fn is_unsupported(result: Result<Option<(RouteKernelPlan, RouteSeal)>, RegistryError>) -> bool {
        matches!(result, Err(RegistryError::UnsupportedRoute(_)))
    }

    #[test]
    fn generic_walked_plan_matches_layout_heads_axes_max_seq_and_seal() {
        let plans = walked_fixture!(main_plan_text());
        let trunk = synthetic_trunk();
        let with_head = synthetic_with_head();
        assert_ne!(synthetic_layout(&trunk), synthetic_layout(&with_head));
        let image = kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_GFX1100;
        assert_eq!(plans[0].sealed, ["ar"]);
        assert_eq!(plans[0].heads, Some((16, 2, 256)));
        assert_eq!(plans[0].mtp_layout, Some(&*synthetic_layout(&with_head)));

        let flags = default_feature_flags("gfx1100").unwrap();
        let axes = synthetic_axes();
        let trunk_metas = synthetic_metas(&trunk);
        let head_metas = synthetic_metas(&with_head);

        // AR over the trunk: the sealed AR route, exactly the AR rows.
        let ar = synthetic_input("gfx1100", &trunk_metas, KernelSpecRoute::Ar, &flags, &axes);
        let (plan, seal) = walked_route_entries_in(plans, &ar).unwrap().unwrap();
        assert_eq!(seal, RouteSeal::Sealed);
        assert_eq!(plan.entries.len(), 2);
        assert!(matches!(
            &plan.entries[1],
            PlannedKernel::Embedded { image: planned, .. } if *planned == image
        ));

        // Native MTP over trunk + head: its own layout, every native_mtp row,
        // and open because only `ar` was replayed sealed.
        let mtp = synthetic_input(
            "gfx1100",
            &head_metas,
            KernelSpecRoute::NativeMtp,
            &flags,
            &axes,
        );
        let (plan, seal) = walked_route_entries_in(plans, &mtp).unwrap().unwrap();
        assert_eq!(seal, RouteSeal::Open("synthetic-qwen35"));
        assert_eq!(plan.entries.len(), 3);

        // The layouts do not cross routes.
        let wrong = synthetic_input(
            "gfx1100",
            &trunk_metas,
            KernelSpecRoute::NativeMtp,
            &flags,
            &axes,
        );
        assert!(is_unsupported(walked_route_entries_in(plans, &wrong)));
        let wrong = synthetic_input("gfx1100", &head_metas, KernelSpecRoute::Ar, &flags, &axes);
        assert!(is_unsupported(walked_route_entries_in(plans, &wrong)));

        // max_seq ceiling, resolved heads, every axis value, a missing and an
        // extra axis, and a non-default pm_decode flag.
        let mut over = synthetic_input("gfx1100", &trunk_metas, KernelSpecRoute::Ar, &flags, &axes);
        over.max_seq = 8193;
        assert!(is_unsupported(walked_route_entries_in(plans, &over)));
        for heads in [(24, 2, 256), (16, 4, 256), (16, 2, 128)] {
            let mut other =
                synthetic_input("gfx1100", &trunk_metas, KernelSpecRoute::Ar, &flags, &axes);
            (other.n_heads, other.n_kv_heads, other.head_dim) = heads;
            assert!(
                is_unsupported(walked_route_entries_in(plans, &other)),
                "{heads:?}"
            );
        }
        for index in 0..axes.len() {
            let mut changed = axes.clone();
            changed[index].1 = "unwalked".to_owned();
            let other = synthetic_input(
                "gfx1100",
                &trunk_metas,
                KernelSpecRoute::Ar,
                &flags,
                &changed,
            );
            assert!(
                is_unsupported(walked_route_entries_in(plans, &other)),
                "{}",
                axes[index].0
            );
            let mut missing = axes.clone();
            missing.remove(index);
            let other = synthetic_input(
                "gfx1100",
                &trunk_metas,
                KernelSpecRoute::Ar,
                &flags,
                &missing,
            );
            assert!(
                is_unsupported(walked_route_entries_in(plans, &other)),
                "missing {}",
                axes[index].0
            );
        }
        let mut extra_axis = axes.clone();
        extra_axis.push(("unwalked_axis", "on".to_owned()));
        let other = synthetic_input(
            "gfx1100",
            &trunk_metas,
            KernelSpecRoute::Ar,
            &flags,
            &extra_axis,
        );
        assert!(is_unsupported(walked_route_entries_in(plans, &other)));
        let mut pm = flags.clone();
        pm.pm_decode = !pm.pm_decode;
        let other = synthetic_input("gfx1100", &trunk_metas, KernelSpecRoute::Ar, &pm, &axes);
        assert!(is_unsupported(walked_route_entries_in(plans, &other)));

        // A header that is not the walked layout, and an arch the plan does not name.
        let mut other_header = trunk.clone();
        other_header.pop();
        let other_metas = synthetic_metas(&other_header);
        let other = synthetic_input("gfx1100", &other_metas, KernelSpecRoute::Ar, &flags, &axes);
        assert!(is_unsupported(walked_route_entries_in(plans, &other)));
        let flags_1151 = default_feature_flags("gfx1151").unwrap();
        let other = synthetic_input(
            "gfx1151",
            &trunk_metas,
            KernelSpecRoute::Ar,
            &flags_1151,
            &axes,
        );
        assert!(walked_route_entries_in(plans, &other).unwrap().is_none());
    }

    #[test]
    fn walked_embedded_rows_resolve_only_on_their_own_arch() {
        let trunk = synthetic_trunk();
        let trunk_metas = synthetic_metas(&trunk);
        let flags_1100 = default_feature_flags("gfx1100").unwrap();
        let flags_1151 = default_feature_flags("gfx1151").unwrap();
        let axes = synthetic_axes();

        // gfx1151's own image resolves; the recorded module name is kept.
        let v2b = kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_V2B_GFX1151;
        let plans = walked_fixture!(embedded_plan_text(
            "gfx1151",
            "gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151",
            kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_V2B_GFX1151
        ));
        let input = synthetic_input(
            "gfx1151",
            &trunk_metas,
            KernelSpecRoute::Ar,
            &flags_1151,
            &axes,
        );
        let (plan, seal) = walked_route_entries_in(plans, &input).unwrap().unwrap();
        assert_eq!(seal, RouteSeal::Open("synthetic-qwen35"));
        assert_eq!(plan.entries.len(), 1);
        assert_eq!(
            plan.entries[0].module(),
            "gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151"
        );
        assert!(matches!(
            &plan.entries[0],
            PlannedKernel::Embedded { image, .. } if *image == v2b
        ));

        // The gfx1100 image recorded under a gfx1151 plan has no runtime image.
        let plans = walked_fixture!(embedded_plan_text(
            "gfx1151",
            "gemm_mq4g256v2_residual_iu4_pm_gfx1100",
            kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_PM_GFX1100
        ));
        let input = synthetic_input(
            "gfx1151",
            &trunk_metas,
            KernelSpecRoute::Ar,
            &flags_1151,
            &axes,
        );
        match walked_route_entries_in(plans, &input) {
            Err(RegistryError::ConflictingPlan(why)) => {
                assert!(why.contains("no runtime image"), "{why}")
            }
            other => panic!(
                "expected a missing runtime image, got {:?}",
                other.map(|p| p.map(|p| p.1))
            ),
        }

        // A recorded SHA no image of the arch matches is refused, not guessed.
        let plans = walked_fixture!(embedded_plan_text(
            "gfx1100",
            "gemm_mq4g256v2_residual_iu4_pm_gfx1100",
            b"not an embedded image"
        ));
        let input = synthetic_input(
            "gfx1100",
            &trunk_metas,
            KernelSpecRoute::Ar,
            &flags_1100,
            &axes,
        );
        assert!(matches!(
            walked_route_entries_in(plans, &input),
            Err(RegistryError::ConflictingPlan(_))
        ));
    }

    #[test]
    fn walked_arch_images_are_exact_arch_and_sha_unique() {
        let archs = ["gfx1100", "gfx1151", "gfx1201"];
        for arch in archs {
            let images = walked_arch_images(arch);
            assert!(!images.is_empty(), "{arch}");
            let shas: Vec<String> = images
                .iter()
                .map(|image| format!("{:x}", Sha256::digest(image)))
                .collect();
            let unique: HashSet<&String> = shas.iter().collect();
            assert_eq!(unique.len(), shas.len(), "{arch}: an image is listed twice");
            for (image, sha) in images.iter().zip(&shas) {
                assert_eq!(walked_arch_image(arch, sha), Some(*image), "{arch}");
                for other in archs.into_iter().filter(|other| *other != arch) {
                    assert!(
                        walked_arch_image(other, sha).is_none(),
                        "a {arch} image resolves under {other}"
                    );
                }
            }
        }
        assert!(walked_arch_image("gfx906", "00").is_none());
    }

    #[test]
    fn walked_plan_directives_reject_unknown_seal_routes_and_bad_heads() {
        macro_rules! base {
            () => {
                "#! family f\n#! model_arch 5\n#! arch gfx1100\n#! layout a\n#! max_seq 1\n"
            };
        }
        let plan = parse_walked_route_plan(concat!(
            base!(),
            "#! sealed ar native_mtp\n#! heads 24,4,256\n#! mtp_layout b\n"
        ))
        .unwrap();
        assert_eq!(plan.sealed, ["ar", "native_mtp"]);
        assert_eq!(plan.heads, Some((24, 4, 256)));
        assert_eq!(plan.mtp_layout, Some("b"));
        let plain = parse_walked_route_plan(base!()).unwrap();
        assert!(plain.sealed.is_empty() && plain.heads.is_none() && plain.mtp_layout.is_none());
        assert!(parse_walked_route_plan(concat!(base!(), "#! sealed turbo\n")).is_err());
        assert!(parse_walked_route_plan(concat!(base!(), "#! heads 24,4\n")).is_err());
        assert!(parse_walked_route_plan(concat!(base!(), "#! heads 24,x,256\n")).is_err());
    }

    /// Every node kernel of the certified DFlash window that is not a default
    /// Redline route kernel has a corpus record on exact gfx1201: the window
    /// refuses a node whose object the corpus lacks.
    #[test]
    fn corpus_entries_cover_the_dflash_window_node_kernels_on_gfx1201() {
        let entries = corpus_entries("gfx1201", "").unwrap();
        let has = |module: &str, symbol: &str| {
            entries
                .iter()
                .any(|e| e.module == module && e.symbols.contains(&symbol))
        };
        for (module, symbol) in [
            (
                "dflash_selector_greedy_exact",
                "dflash_selector_greedy_exact",
            ),
            ("railgun_copy", "railgun_copy"),
            ("railgun_fill", "railgun_fill_u32"),
            ("select_regrid", "topk_values_regrid_f32"),
            ("topk_values_fixup_f32", "topk_values_fixup_f32"),
            ("select_regrid", "argmax_f32_batched_regrid"),
            (
                "attention_dflash_sliding_f32",
                "attention_dflash_sliding_f32",
            ),
            (
                "dflash_hidden_commit5_gfx1100",
                "dflash_hidden_commit5_gfx1100",
            ),
            ("embedding_q8_batched", "embedding_q8_batched"),
        ] {
            assert!(
                has(module, symbol),
                "gfx1201 corpus lacks {module}/{symbol}"
            );
        }
        for arch in ["gfx1100", "gfx1151"] {
            assert!(
                !corpus_entries(arch, "")
                    .unwrap()
                    .iter()
                    .any(|e| e.module == "dflash_selector_greedy_exact"),
                "{arch} must not inventory the gfx1201 window selector"
            );
        }
    }
}
