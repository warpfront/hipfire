// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Build-time HIP source inventory for admitted Qwen3.5 routes. This is a
//! packaging inventory, not a replacement for the runtime's dispatch policy.
//! Every source below is the Rust source expression supplied to `ensure_kernel`
//! or to `precompile_qwen35`; never infer module names from HIP filenames.

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
    pub const FUSED_QK_L2_NORM_SCALE_SRC: &str = include_str!("../../../kernels/src/fused_qk_l2_norm_scale.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const FUSED_SIGMOID_ALPHA_GATE_SRC: &str = include_str!("../../../kernels/src/fused_sigmoid_alpha_gate.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const CONV1D_SILU_SPLIT_SRC: &str = include_str!("../../../kernels/src/conv1d_silu_split.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const CONV1D_SILU_SPLIT_TREE_SRC: &str = include_str!("../../../kernels/src/conv1d_silu_split_tree.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_DELTA_NET_Q8_TREE_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q8_tree.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const SCALE_F32_SRC: &str = include_str!("../../../kernels/src/scale_f32.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_NORM_SRC: &str = include_str!("../../../kernels/src/gated_norm.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_INTERLEAVED_SRC: &str = include_str!("../../../kernels/src/rope_partial_interleaved.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const GATED_DELTA_NET_Q8_SRC: &str = include_str!("../../../kernels/src/gated_delta_net_q8.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_SRC: &str = include_str!("../../../kernels/src/rope_partial_halfsplit.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC: &str = include_str!("../../../kernels/src/rope_partial_halfsplit_batched.hip");
}

pub const SUPPORTED_ARCHES: &[&str] = &["gfx1201", "gfx1100", "gfx1151", "gfx906", "gfx942"];

#[derive(Debug)]
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
    UnsupportedModule { arch: String, module: String },
}

fn prepend_kv_slot_desc(body: &str) -> String {
    // Same assembly as attention.rs and dispatch.rs precompile_qwen35.
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
/// {asym3,q8}` default precompile matrix. Other arches currently advertise
/// only the architecture-independent precompile entries below; rejected pairs
/// are explicit rather than compiled from a guessed basename.
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
        ($module:literal, $src:expr, [$($symbol:literal),+ $(,)?]) => {{
            let source: Cow<'static, str> = ($src).into();
            let recipe = KernelCompiler::recipe_for_source(arch, $module, &source, extra_flags);
            entries.push(KernelEntry {
                arch,
                module: $module,
                symbols: &[$($symbol),+],
                flags: recipe.flags,
                scheduler_profile: recipe.scheduler_profile,
                source,
            });
        }};
    }

    // precompile_qwen35 common kernels (dispatch.rs:4852-4897,5176-5203).
    add!("rmsnorm", kernels::RMSNORM_SRC, ["rmsnorm_f32"]);
    add!("add_inplace", kernels::ADD_INPLACE_SRC, ["add_inplace_f32"]);
    add!("mul", kernels::MUL_SRC, ["mul_f32"]);
    add!("silu_mul", kernels::SILU_MUL_SRC, ["silu_mul_f32"]);
    add!("sigmoid", kernels::SIGMOID_SRC, ["sigmoid_f32"]);
    add!("alpha_gate", kernels::ALPHA_GATE_SRC, ["alpha_gate_f32"]);
    add!("conv1d_silu", kernels::CONV1D_SILU_SRC, ["conv1d_silu_f32"]);
    add!("l2_norm", kernels::L2_NORM_SRC, ["l2_norm_f32"]);
    add!("fused_qk_l2_norm_scale", kernels::FUSED_QK_L2_NORM_SCALE_SRC, ["fused_qk_l2_norm_scale_f32"]);
    add!("fused_sigmoid_alpha_gate", kernels::FUSED_SIGMOID_ALPHA_GATE_SRC, ["fused_sigmoid_alpha_gate_f32"]);
    add!("conv1d_silu_split", kernels::CONV1D_SILU_SPLIT_SRC, ["conv1d_silu_split_f32"]);
    add!("conv1d_silu_split_tree", kernels::CONV1D_SILU_SPLIT_TREE_SRC, ["conv1d_silu_split_tree_f32"]);
    add!("gated_delta_net_q8_tree", kernels::GATED_DELTA_NET_Q8_TREE_SRC, ["gated_delta_net_q8_tree"]);
    add!("sigmoid_mul", kernels::SIGMOID_MUL_SRC, ["sigmoid_mul_f32"]);
    add!("topk_logits", kernels::TOPK_LOGITS_SRC, ["topk_logits_f32"]);
    add!("scale_f32", kernels::SCALE_F32_SRC, ["scale_f32"]);
    add!("gated_norm", kernels::GATED_NORM_SRC, ["gated_norm_f32"]);
    add!("rope_partial_interleaved", kernels::ROPE_PARTIAL_INTERLEAVED_SRC, ["rope_partial_interleaved_f32"]);
    add!("deinterleave", kernels::DEINTERLEAVE_SRC, ["deinterleave_f32"]);
    add!("repeat_interleave_qk", kernels::REPEAT_INTERLEAVE_QK_SRC, ["repeat_interleave_qk_f32"]);
    add!("embedding_q8", kernels::EMBEDDING_Q8_SRC, ["embedding_q8"]);
    add!("embedding_q8_batched", kernels::EMBEDDING_Q8_BATCHED_SRC, ["embedding_q8_batched"]);
    add!("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC, ["embedding_hfq4g128"]);
    add!("embedding_hfq4g128_batched", kernels::EMBEDDING_HFQ4G128_BATCHED_SRC, ["embedding_hfq4g128_batched"]);
    add!("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC, ["embedding_hfq4g256"]);
    add!("embedding_hfq4g256_batched", kernels::EMBEDDING_HFQ4G256_BATCHED_SRC, ["embedding_hfq4g256_batched"]);
    add!("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC, ["gated_delta_net_q8"]);

    // Matrix-dependent default precompile modules; each name and source is
    // taken from the same constants as dispatch.rs:4900-5174.
    add!("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC, ["gemv_hfq6g256"]);
    add!("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC, ["gemv_mq6g256"]);
    add!("gemm_mq6g256", kernels::GEMM_MQ6G256_SRC, ["gemm_mq6g256"]);
    add!("gemv_q8_0", kernels::GEMV_Q8_0_SRC, ["gemv_q8_0"]);
    add!("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, ["gemv_mq4g256", "mq_rotate_x"]);
    add!("gemv_hfq4g256_wide", kernels::GEMV_HFQ4G256_WIDE_SRC, ["gemv_hfq4g256_wide"]);
    add!("fused_qkvza_hfq4g256", kernels::FUSED_QKVZA_HFQ4G256_SRC, ["fused_qkvza_hfq4g256"]);
    add!("fused_qkv_hfq4g256", kernels::FUSED_QKV_HFQ4G256_SRC, ["fused_qkv_hfq4g256"]);
    add!("fused_gate_up_hfq4g256", kernels::FUSED_GATE_UP_HFQ4G256_SRC, ["fused_gate_up_hfq4g256"]);
    add!("fused_rmsnorm_mq_rotate", kernels::FUSED_RMSNORM_MQ_ROTATE_SRC, ["fused_rmsnorm_mq_rotate"]);
    add!("fused_silu_mul_mq_rotate", kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC, ["fused_silu_mul_mq_rotate"]);
    match arch {
        "gfx1100" => add!("gemv_hfq4g256_rdna3", kernels::GEMV_HFQ4G256_GFX1100_SRC, ["gemv_hfq4g256"]),
        _ => add!("gemv_hfq4g256", kernels::GEMV_HFQ4G256_SRC, ["gemv_hfq4g256"]),
    }
    if arch == "gfx1100" {
        // Default gfx1100 precompile branches (dispatch.rs:4944-4962,5304-5308).
        add!("fused_qkvza_hfq4g256_k2048_gfx1100", kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC, ["fused_qkvza_hfq4g256_k2048"]);
        add!("fused_gate_up_hfq4g256_stage_x32_gfx1100", kernels::FUSED_GATE_UP_HFQ4G256_STAGE_X32_GFX1100_SRC, ["fused_gate_up_hfq4g256_stage_x32_gfx1100"]);
        add!("kv_cache_write_asym3_q8_pair_gfx1100", assemble_asym(kernels::KV_CACHE_WRITE_ASYM3_Q8_PAIR_GFX1100_SRC), ["kv_cache_write_asym3_q8_pair_gfx1100"]);
    }
    if matches!(arch, "gfx906" | "gfx942") {
        add!("fused_qkvza_hfq4g256_wave64", kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC, ["fused_qkvza_hfq4g256_wave64"]);
        add!("fused_qkv_hfq4g256_wave64", kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC, ["fused_qkv_hfq4g256_wave64"]);
        add!("fused_gate_up_hfq4g256_wave64", kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC, ["fused_gate_up_hfq4g256_wave64"]);
        add!("gemv_hfq4g256_moe_gate_up_indexed_wave64", kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC, ["gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"]);
        add!("gemv_hfq4g256_moe_down_indexed_wave64", kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC, ["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64"]);
        add!("gemm_qkvza_hfq4g256_wave64", kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC, ["gemm_qkvza_hfq4g256_wave64"]);
        add!("gemm_qkv_hfq4g256_wave64", kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC, ["gemm_qkv_hfq4g256_wave64"]);
        add!("gemm_hfq4g256_wave64", kernels::GEMM_HFQ4G256_WAVE64_SRC, ["gemm_hfq4g256_wave64"]);
        add!("gemm_hfq4g256_residual_wave64", kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC, ["gemm_hfq4g256_residual_wave64"]);
        add!("gemv_hfq4g256_moe_gate_up_indexed_batched_wave64", kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC, ["gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64"]);
        add!("gemv_hfq4g256_moe_down_indexed_batched_wave64", kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC, ["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64"]);
    }

    // KV paths: precompile's asym3/q8 assembly mirrors the runtime header
    // stripping; FP8 substitutions are only admitted for observed gfx1201.
    add!("kv_cache_write_asym_k_givens3", assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC), ["kv_cache_write_asym_k_givens3"]);
    add!("kv_cache_write_asym_k_givens3_batched", assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC), ["kv_cache_write_asym_k_givens3_batched"]);
    add!("attention_flash_asym3_tile", assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_SRC), ["attention_flash_asym3_tile"]);
    add!("attention_flash_asym3_tile_batched", assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC), ["attention_flash_asym3_tile_batched"]);
    add!("attention_flash_asym_reduce_batched", kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC, ["attention_flash_asym_reduce_batched"]);
    add!("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC, ["kv_cache_write_q8_0"]);
    add!("attention_q8_0_kv", kernels::ATTENTION_Q8_0_KV_SRC, ["attention_q8_0_kv"]);
    add!("attention_q8_0_kv_batched", prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_KV_BATCHED_SRC), ["attention_q8_0_kv_batched"]);
    add!("attention_q8_0_kv_independent_masked_windowed", prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_KV_BATCHED_SRC), ["attention_q8_0_kv_independent_masked_windowed"]);
    add!("attention_q8_0_flash_prefill", prepend_kv_slot_desc(kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC), ["attention_q8_0_flash_prefill"]);
    // The installer-only unspecialized module above cannot satisfy this
    // runtime's default scalar-prefill module or its BR/BC-specialized source.
    add!("attention_q8_0_flash_prefill_br8_bc16", q8_flash_prefill_default_source(), ["attention_q8_0_flash_prefill"]);
    add!("kv_cache_write_q8_0_batched", prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC), ["kv_cache_write_q8_0_batched"]);
    add!("kv_cache_write_q8_0_independent", prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC), ["kv_cache_write_q8_0_independent"]);
    add!("kv_cache_write_q8_0_independent_masked", prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC), ["kv_cache_write_q8_0_independent_masked"]);
    add!("attention_flash_q8_0_tile", kernels::ATTENTION_FLASH_Q8_0_TILE_SRC, ["attention_flash_q8_0_tile"]);
    add!("attention_flash_q8_0_reduce", kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC, ["attention_flash_q8_0_reduce"]);
    add!("sample_top_p_parallel", sampling::sample_top_p_parallel_src(), ["sample_apply_repeat_penalty", "sample_topk_partial", "sample_topk_finalize"]);
    add!("sample_top_p_parallel_w64", sampling::sample_top_p_parallel_w64_src(), ["sample_apply_repeat_penalty_w64", "sample_topk_partial_w64", "sample_topk_finalize_w64"]);
    add!("sample_top_p_parallel_fast21", sampling::sample_top_p_parallel_fast_src(21, "fast21"), ["sample_apply_repeat_penalty_fast21", "sample_topk_partial_fast21", "sample_topk_finalize_fast21"]);
    add!("sample_top_p_parallel_fast65", sampling::sample_top_p_parallel_fast_src(65, "fast65"), ["sample_apply_repeat_penalty_fast65", "sample_topk_partial_fast65", "sample_topk_finalize_fast65"]);

    if arch == "gfx1201" {
        // H2 and small-model first-token JIT: use the actual module key, even
        // when it differs from the kernel symbol or the source filename.
        add!("attention_flash_fp8_e4m3_tile", kernels::ATTENTION_FLASH_FP8_E4M3_TILE_SRC, ["attention_flash_fp8_e4m3_tile"]);
        add!("attention_fp8_e4m3_kv_batched", prepend_kv_slot_desc(kernels::ATTENTION_FP8_E4M3_KV_BATCHED_SRC), ["attention_fp8_e4m3_kv_batched"]);
        add!("conv1d_silu_split_qknorm_b256", kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC, ["conv1d_silu_split_qknorm_b256"]);
        add!("convert_f32_to_f16", kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC, ["convert_f32_to_f16"]);
        add!("fused_gate_up_hfq4g256_mq4v2", kernels::FUSED_GATE_UP_MQ4G256V2_SRC, ["fused_gate_up_mq4g256v2"]);
        add!("fused_qkv_hfq4g256_mq4v2", kernels::FUSED_QKV_MQ4G256V2_SRC, ["fused_qkv_mq4g256v2"]);
        add!("fused_qkvza_hfq4g256_mq4v2", kernels::FUSED_QKVZA_MQ4G256V2_SRC, ["fused_qkvza_mq4g256v2"]);
        add!("fused_rmsnorm_mq_rotate_awq", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC, ["fused_rmsnorm_mq_rotate_awq"]);
        add!("fused_silu_mul_mq_rotate_awq", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC, ["fused_silu_mul_mq_rotate_awq"]);
        add!("gated_delta_net_q8_fast", kernels::GATED_DELTA_NET_Q8_FAST_SRC, ["gated_delta_net_q8_fast"]);
        add!("gdn_pre_batched_gfx1201", include_str!("../../../kernels/src/gdn_pre_batched.gfx1201.hip"), ["gdn_pre_batched_gfx1201"]);
        add!("gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_gate_up_mq4g256v2_wmma_gfx12"]);
        add!("gemm_hfq4g256_residual_wmma_gfx12_mq4v2", kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC, ["gemm_mq4g256v2_residual_wmma_gfx12"]);
        add!("gemm_qkv_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_qkv_mq4g256v2_wmma_gfx12"]);
        add!("gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_qkvza_mq4g256v2_wmma_gfx12"]);
        add!("gemv_hfq4g256_multirow_default_mq4v2", kernels::GEMV_MQ4G256V2_MULTIROW_SRC, ["gemv_mq4g256v2_multirow_r2", "gemv_mq4g256v2_multirow_r4", "gemv_mq4g256v2_multirow_r8"]);
        add!("gemv_hfq4g256_residual_mq4v2", kernels::GEMV_MQ4G256V2_RESIDUAL_SRC, ["gemv_mq4g256v2_residual"]);
        add!("gemv_mq4g256v2_mq4v2", kernels::GEMV_MQ4G256V2_SRC, ["gemv_mq4g256v2"]);
        add!("kv_cache_write_fp8_e4m3", kernels::KV_CACHE_WRITE_FP8_E4M3_SRC, ["kv_cache_write_fp8_e4m3"]);
        add!("kv_cache_write_fp8_e4m3_batched", prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_FP8_E4M3_BATCHED_SRC), ["kv_cache_write_fp8_e4m3_batched"]);
        add!("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, ["mq_rotate_x"]);
        add!("qwen35_fa_prep_batched_gfx1201", include_str!("../../../kernels/src/qwen35_fa_prep_batched.gfx1201.hip"), ["qwen35_fa_prep_batched_gfx1201"]);
        add!("rmsnorm_f32", kernels::RMSNORM_SRC, ["rmsnorm_f32"]);
        add!("rope_partial_halfsplit", kernels::ROPE_PARTIAL_HALFSPLIT_SRC, ["rope_partial_halfsplit_f32"]);
        add!("rotate_x_mq_awq", kernels::ROTATE_X_MQ_AWQ_SRC, ["rotate_x_mq_awq"]);
        add!("deinterleave_q_rmsnorm_f32_batched", kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC, ["deinterleave_q_rmsnorm_f32_batched"]);
        add!("fused_gate_up_hfq4g256_k1024_gfx1201", kernels::FUSED_GATE_UP_HFQ4G256_K1024_GFX1201_SRC, ["fused_gate_up_hfq4g256_k1024_gfx1201"]);
        add!("gemm_gate_up_hfq4g256_wmma_gfx12", kernels::GEMM_GATE_UP_HFQ4G256_WMMA_GFX12_SRC, ["gemm_gate_up_hfq4g256_wmma_gfx12"]);
        add!("gemm_hfq4g256_residual_wmma_gfx12", kernels::GEMM_HFQ4G256_RESIDUAL_WMMA_GFX12_SRC, ["gemm_hfq4g256_residual_wmma_gfx12"]);
        add!("gemm_qkv_hfq4g256_wmma_gfx12", kernels::GEMM_QKV_HFQ4G256_WMMA_GFX12_SRC, ["gemm_qkv_hfq4g256_wmma_gfx12"]);
        add!("gemm_qkvza_hfq4g256_wmma_gfx12", kernels::GEMM_QKVZA_HFQ4G256_WMMA_GFX12_SRC, ["gemm_qkvza_hfq4g256_wmma_gfx12"]);
        add!("gemv_hfq4g256_residual", kernels::GEMV_HFQ4G256_RESIDUAL_SRC, ["gemv_hfq4g256_residual"]);
        add!("gemv_q8_0_wide", kernels::GEMV_Q8_0_WIDE_SRC, ["gemv_q8_0_wide"]);
        add!("rope_partial_halfsplit_batched", kernels::ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC, ["rope_partial_halfsplit_batched_f32"]);
    }
    Ok(entries)
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

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn kernel_registry_p0_sources_are_byte_identical() {
        let root = Path::new("/home/kaden/qcal/boundary/p0/raw");
        // The gfx1201 runtime appends this default via FeatureFlags at init;
        // it appears in every P0 hipcc argv, even without an env override.
        let registry = entries("gfx1201", "-DIU4_A4_CANDIDATES=2").unwrap();
        let by_name: HashMap<_, _> = registry.iter().map(|e| (e.module, e)).collect();
        assert_eq!(by_name.len(), registry.len(), "duplicate module identities");
        // Portable copy of P0's full effective-source fingerprints: CI can
        // check every entry without depending on this machine's scratch cache.
        let expected = include_str!("../tests/fixtures/p0-gfx1201-sha256.tsv");
        let mut count = 0;
        for line in expected.lines().filter(|line| !line.starts_with('#')) {
            let (module, digest) = line.split_once('\t').unwrap();
            let entry = by_name.get(module).unwrap_or_else(|| panic!("missing P0 module {module}"));
            assert_eq!(format!("{:x}", Sha256::digest(entry.source().as_bytes())), digest, "{module}");
            count += 1;
        }
        assert_eq!(count, 92);
        // The installer trace predates the scalar-prefill runtime's BR/BC
        // specialization. Its default key/source is additional to P0's 92.
        assert_eq!(registry.len(), count + 1, "unexpected gfx1201 inventory size");
        let default_prefill = by_name.get("attention_q8_0_flash_prefill_br8_bc16").unwrap();
        assert_eq!(default_prefill.symbols, ["attention_q8_0_flash_prefill"]);
        assert!(default_prefill.source().starts_with(
            "#define BR 8\n#define BC 16\n#define NTHREADS 256\n"
        ));
        assert_eq!(
            format!("{:x}", Sha256::digest(default_prefill.source().as_bytes())),
            "680c37bf2f4f978d0361bd5c1b48c1fbd6430d1777663d31b45c9b6ad510bb95",
            "default Q8 flash prefill source changed"
        );
        assert_ne!(
            default_prefill.source(),
            by_name.get("attention_q8_0_flash_prefill").unwrap().source()
        );
        if !root.is_dir() {
            // P0 cache paths are machine-local. The 92 digests above are
            // checked everywhere; on the capture machine, also compare bytes
            // and per-invocation argv directly to all three raw trace files.
            return;
        }
        for (trace, expected_count) in [("cold1", 35), ("smokecold1", 35), ("preinstall", 58)] {
            let records = std::fs::read_to_string(root.join(format!("{trace}.hipcc.jsonl"))).unwrap();
            let mut observed = 0;
            for line in records.lines() {
                let record: serde_json::Value = serde_json::from_str(line).unwrap();
                let Some(module) = record["module"].as_str() else { continue };
                let source_path = record["source"].as_str().unwrap();
                let expected = std::fs::read(source_path).unwrap();
                let entry = by_name.get(module).unwrap_or_else(|| panic!("missing {trace} module {module}"));
                assert_eq!(entry.source().as_bytes(), expected, "{trace}: {module} source differs");
                let argv = record["argv"].as_array().unwrap();
                let expected_flags = argv.iter().map(|arg| arg.as_str().unwrap())
                    .take_while(|arg| *arg != "-o")
                    .filter(|arg| !arg.starts_with("--rocm-path=")
                        && !arg.starts_with("--hip-path=") && !arg.starts_with("-I"))
                    .collect::<Vec<_>>();
                assert_eq!(entry.flags.iter().map(String::as_str).collect::<Vec<_>>(),
                    expected_flags, "{trace}: {module} core hipcc flags differ");
                observed += 1;
            }
            assert_eq!(observed, expected_count, "{trace}: incomplete compiler trace");
        }
    }

    #[test]
    fn kernel_registry_explicit_arch_and_module_rejection() {
        assert!(matches!(lookup("gfx1200", "rmsnorm", ""), Err(RegistryError::UnsupportedArch(_))));
        assert!(matches!(lookup("gfx942", "qwen35_fa_prep_batched_gfx1201", ""), Err(RegistryError::UnsupportedModule { .. })));
        assert!(lookup("gfx942", "fused_qkv_hfq4g256_wave64", "").is_ok());
        assert!(lookup("gfx906", "gemm_qkv_hfq4g256_wmma_gfx12", "").is_err());
        for arch in SUPPORTED_ARCHES {
            assert!(lookup(arch, "attention_q8_0_flash_prefill_br8_bc16", "").is_ok());
            assert!(lookup(arch, "attention_q8_0_flash_prefill_br16_bc32", "").is_err());
        }
    }
}
