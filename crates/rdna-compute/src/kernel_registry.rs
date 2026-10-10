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
    pub const ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC: &str = include_str!("../../../kernels/src/rope_partial_halfsplit_headgrid.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC: &str = include_str!("../../../kernels/src/rope_partial_halfsplit_batched.hip");
    #[cfg(not(feature = "deltanet"))]
    pub const QWEN35_FA_PREP_GFX1100_SRC: &str = include_str!("../../../kernels/src/qwen35_fa_prep.gfx1100.hip");
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

    // Common Qwen3.5 kernels (historical whole-model precompile set).
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
        add!("fused_gate_up_mq4g256v2_k5120_gfx1100", kernels::fused_gate_up_mq4g256v2_k5120_gfx1100_src(), ["fused_gate_up_mq4g256v2_k5120_gfx1100"]);
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
    // Multi-slot (slot-descriptor) launches of the q8 and asym3 KV routes run
    // separately named `*_paged` modules (attention.rs `ensure_kv_slot_kernel`,
    // `ensure_givens4_kv_slot_kernel`, `attention_q8_0_flash_prefill_wmma_slots`).
    add!("kv_cache_write_q8_0_batched_paged", kernels::kv_slot_desc_paged_source(kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC, "kv_cache_write_q8_0_batched", "kv_cache_write_q8_0_batched_paged"), ["kv_cache_write_q8_0_batched_paged"]);
    add!("attention_q8_0_kv_batched_paged", kernels::kv_slot_desc_paged_source(kernels::ATTENTION_Q8_0_KV_BATCHED_SRC, "attention_q8_0_kv_batched", "attention_q8_0_kv_batched_paged"), ["attention_q8_0_kv_batched_paged"]);
    add!("attention_flash_q8_0_tile_batched_paged", kernels::kv_slot_givens4_paged_source(kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC, "attention_flash_q8_0_tile_batched", "attention_flash_q8_0_tile_batched_paged"), ["attention_flash_q8_0_tile_batched_paged"]);
    add!("kv_cache_write_asym_k_givens3_batched_paged", kernels::kv_slot_givens4_paged_source(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC, "kv_cache_write_asym_k_givens3_batched", "kv_cache_write_asym_k_givens3_batched_paged"), ["kv_cache_write_asym_k_givens3_batched_paged"]);
    add!("attention_flash_asym3_tile_batched_paged", kernels::kv_slot_givens4_paged_source(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC, "attention_flash_asym3_tile_batched", "attention_flash_asym3_tile_batched_paged"), ["attention_flash_asym3_tile_batched_paged"]);
    if matches!(arch, "gfx1100" | "gfx1151") {
        // gfx11 admits the multi-slot WMMA prefill tile by default
        // (forward_slots `q8_flash_prefill_wmma_eligible`); Qwen3.5 head_dim
        // 256 with the default SPLIT_Q=0, fixed head dim and V prefetch.
        add!("attention_q8_0_flash_prefill_wmma_gfx11_hd256_paged", format!("#define SPLIT_Q 0\n#define FIXED_HEAD_DIM 256\n#define PREFETCH_V 1\n{}", kernels::kv_slot_desc_paged_source(kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC, "attention_q8_0_flash_prefill_wmma", "attention_q8_0_flash_prefill_wmma_paged")), ["attention_q8_0_flash_prefill_wmma_paged"]);
    }
    add!("sample_top_p_parallel", sampling::sample_top_p_parallel_src(), ["sample_apply_repeat_penalty", "sample_topk_partial", "sample_topk_finalize"]);
    add!("sample_top_p_parallel_w64", sampling::sample_top_p_parallel_w64_src(), ["sample_apply_repeat_penalty_w64", "sample_topk_partial_w64", "sample_topk_finalize_w64"]);
    add!("sample_top_p_parallel_fast21", sampling::sample_top_p_parallel_fast_src(21, "fast21"), ["sample_apply_repeat_penalty_fast21", "sample_topk_partial_fast21", "sample_topk_finalize_fast21"]);
    add!("sample_top_p_parallel_fast65", sampling::sample_top_p_parallel_fast_src(65, "fast65"), ["sample_apply_repeat_penalty_fast65", "sample_topk_partial_fast65", "sample_topk_finalize_fast65"]);
    // Qwen4 logit-policy penalty prepass (AR greedy rows, sampled-MTP verify
    // rows; `Gpu::apply_penalty_table`).
    add!("logit_penalty_table", kernels::LOGIT_PENALTY_TABLE_SRC, ["logit_penalty_table_rows"]);

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
        add!("fused_rmsnorm_mq_rotate_awq_g12dec", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_G12DEC_SRC, ["fused_rmsnorm_mq_rotate_awq_g12dec"]);
        add!("fused_silu_mul_mq_rotate_awq", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC, ["fused_silu_mul_mq_rotate_awq"]);
        add!("gated_delta_net_q8_fast", kernels::GATED_DELTA_NET_Q8_FAST_SRC, ["gated_delta_net_q8_fast"]);
        add!("gdn_pre_batched_gfx1201", include_str!("../../../kernels/src/gdn_pre_batched.gfx1201.hip"), ["gdn_pre_batched_gfx1201"]);
        add!("gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_gate_up_mq4g256v2_wmma_gfx12"]);
        add!("gemm_gate_up_mq4g256v2_wmma_gfx12_vt", kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC, ["gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w8_k32"]);
        add!("gemm_hfq4g256_residual_wmma_gfx12_mq4v2", kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC, ["gemm_mq4g256v2_residual_wmma_gfx12"]);
        add!("gemm_mq4g256v2_residual_wmma_gfx12_vt", kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC, ["gemm_mq4g256v2_residual_wmma_gfx12_vt2w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt2w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt3w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt3w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt8w4_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt8w8_k32"]);
        add!("gemm_qkv_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_qkv_mq4g256v2_wmma_gfx12"]);
        add!("gemm_qkv_mq4g256v2_wmma_gfx12_vt", kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC, ["gemm_qkv_mq4g256v2_wmma_gfx12_vt2w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w8_k32"]);
        add!("gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2", kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC, ["gemm_qkvza_mq4g256v2_wmma_gfx12"]);
        add!("gemm_qkvza_mq4g256v2_wmma_gfx12_vt", kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC, ["gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w8_k32"]);
        add!("gemv_hfq4g256_multirow_default_mq4v2", kernels::GEMV_MQ4G256V2_MULTIROW_SRC, ["gemv_mq4g256v2_multirow_r2", "gemv_mq4g256v2_multirow_r4", "gemv_mq4g256v2_multirow_r8"]);
        add!("gemv_hfq4g256_residual_mq4v2", kernels::GEMV_MQ4G256V2_RESIDUAL_SRC, ["gemv_mq4g256v2_residual"]);
        add!("gemv_mq4g256v2_mq4v2", kernels::GEMV_MQ4G256V2_SRC, ["gemv_mq4g256v2"]);
        add!("kv_cache_write_fp8_e4m3", kernels::KV_CACHE_WRITE_FP8_E4M3_SRC, ["kv_cache_write_fp8_e4m3"]);
        add!("kv_cache_write_fp8_e4m3_batched", prepend_kv_slot_desc(kernels::KV_CACHE_WRITE_FP8_E4M3_BATCHED_SRC), ["kv_cache_write_fp8_e4m3_batched"]);
        add!("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, ["mq_rotate_x"]);
        add!("qwen35_fa_prep_batched_gfx1201", include_str!("../../../kernels/src/qwen35_fa_prep_batched.gfx1201.hip"), ["qwen35_fa_prep_batched_gfx1201"]);
        add!("rmsnorm_f32", kernels::RMSNORM_SRC, ["rmsnorm_f32"]);
        add!("rmsnorm_f32_rowsplit", kernels::RMSNORM_ROWSPLIT_SRC, ["rmsnorm_f32_rowsplit"]);
        add!("rope_partial_halfsplit", kernels::ROPE_PARTIAL_HALFSPLIT_SRC, ["rope_partial_halfsplit_f32"]);
        add!("rope_partial_halfsplit_f32_headgrid", kernels::ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC, ["rope_partial_halfsplit_f32_headgrid"]);
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
        // H2 (Qwen3.8-27B MQ4v2, fp8 KV) decode, observed JIT-compiled by
        // `hipfire run` greedy AR/MTP/DFlash on the installer inventory above.
        add!("attention_flash_fp8_e4m3_tile_gqa_gfx1201", kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC, ["attention_flash_fp8_e4m3_tile_gqa_gfx1201"]);
        add!("attention_flash_reduce_dsplit_gfx1201", kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC, ["attention_flash_reduce_dsplit_gfx1201"]);
        add!("dflash_state_bulk_copy_gfx1100", crate::dflash_state_copy::DFLASH_STATE_BULK_COPY_GFX1100_SRC, ["dflash_state_bulk_copy_gfx1100"]);
        add!("gated_delta_net_q8_compact3_b2", kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC, ["gated_delta_net_q8_compact3_b2"]);
        add!("gated_norm_mq_rotate_awq_k6144_gfx1201", kernels::gated_norm_mq_rotate_awq_k6144_gfx1201_src(), ["gated_norm_mq_rotate_awq_k6144_gfx1201"]);
        add!("qwen36_27b_fa_prep_gfx1201", kernels::qwen36_27b_fa_prep_gfx1201_src(), ["qwen36_27b_fa_prep_gfx1201"]);
    }
    // H2 (Qwen3.8-27B MQ4v2) prefill, MTP and DFlash modules JIT-compiled by
    // `hipfire serve`/`run` beyond the inventory above: gfx1201 fp8 KV, and
    // gfx1100/gfx1151 Q8 KV. Symbols are every kernel each object defines.
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!("attention_dflash_sliding_f32", kernels::ATTENTION_DFLASH_SLIDING_SRC, ["attention_dflash_sliding_f32"]);
        add!("deinterleave_batched", kernels::DEINTERLEAVE_BATCHED_SRC, ["deinterleave_f32_batched"]);
        add!("dynamic_conv_f32", kernels::DYNAMIC_CONV_F32_SRC, ["dynamic_causal_conv_f32", "dynamic_conv_f32"]);
        add!("fused_qk_l2_norm_scale_interleave_f32_batched", kernels::FUSED_QK_L2_NORM_SCALE_INTERLEAVE_F32_BATCHED_SRC, ["fused_qk_l2_norm_scale_interleave_f32_batched"]);
        add!("gemm_hfq4g256", kernels::GEMM_HFQ4G256_SRC, ["gemm_hfq4g256"]);
        add!("rope_batched", kernels::ROPE_BATCHED_SRC, ["rope_batched_f32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!("add", kernels::ADD_SRC, ["add_f32", "broadcast_add_rows_f32"]);
        add!("argmax_token_chain", kernels::ARGMAX_TOKEN_CHAIN_SRC, ["argmax_token_chain_f32"]);
        add!("gdn_chunk_kkt_solve", kernels::GDN_CHUNK_KKT_SOLVE_SRC, ["gdn_chunk_kkt_solve"]);
        add!("gemv_hfq4g256_multirow_default", kernels::GEMV_HFQ4G256_MULTIROW_SRC, ["gemv_hfq4g256_multirow_r2", "gemv_hfq4g256_multirow_r4", "gemv_hfq4g256_multirow_r8"]);
        add!("greedy_accept", kernels::GREEDY_ACCEPT_SRC, ["greedy_accept_from_argmax_i32"]);
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!("argmax_f32_batched", kernels::ARGMAX_BATCHED_SRC, ["argmax_f32_batched"]);
        add!("attention_q8_0_flash_prefill_wmma_gfx11_hd256", q8_flash_prefill_wmma_gfx11_hd256_source(), ["attention_q8_0_flash_prefill_wmma"]);
        add!("convert_f32_to_f16", kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC, ["convert_f32_to_f16", "gemm_hfq4g256_residual_fp16"]);
        add!("fused_gate_up_hfq4g256_mq4v2", kernels::FUSED_GATE_UP_MQ4G256V2_SRC, ["fused_gate_up_mq4g256v2"]);
        add!("fused_qkv_hfq4g256_mq4v2", kernels::FUSED_QKV_MQ4G256V2_SRC, ["fused_qkv_mq4g256v2"]);
        add!("fused_qkvza_hfq4g256_mq4v2", kernels::FUSED_QKVZA_MQ4G256V2_SRC, ["fused_qkvza_mq4g256v2"]);
        add!("fused_rmsnorm_mq_rotate_awq", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_SRC, ["fused_rmsnorm_mq_rotate_awq"]);
        add!("fused_rmsnorm_mq_rotate_awq_g12dec", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_G12DEC_SRC, ["fused_rmsnorm_mq_rotate_awq_g12dec"]);
        add!("fused_silu_mul_mq_rotate_awq", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC, ["fused_silu_mul_mq_rotate_awq"]);
        add!("fused_silu_mul_mq_rotate_awq_i4", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_SRC, ["fused_silu_mul_mq_rotate_awq_i4"]);
        add!("fused_silu_mul_mq_rotate_awq_i4_hin", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_SRC, ["fused_silu_mul_mq_rotate_awq_i4_hin"]);
        add!("gated_delta_net_q8_fast", kernels::GATED_DELTA_NET_Q8_FAST_SRC, ["gated_delta_net_q8_fast", "gated_delta_net_q8_fast_independent_masked"]);
        add!("gdn_chunk_prep_gfx11", kernels::GDN_CHUNK_PREP_GFX11_SRC, ["gdn_chunk_prep_gfx11"]);
        add!("gemm_gate_up_mq4g256v2_wmma", kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_SRC, ["gemm_gate_up_mq4g256v2_wmma"]);
        // HIPFIRE_IU4_PREFILL=0 uses the Q8_1 MMQ source unchanged; both
        // X128 and per-32 exports share this object and compiler recipe.
        add!("gemm_mq4g256v2_residual_mmq", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_SRC, [
            "gemm_mq4g256v2_residual_mmq", "gemm_mq4g256v2_residual_mmq_full_add",
            "gemm_mq4g256v2_residual_mmq_full_set", "gemm_mq4g256v2_residual_mmq_x128",
            "gemm_mq4g256v2_residual_mmq_full_add_x128", "gemm_mq4g256v2_residual_mmq_full_set_x128",
            "quantize_q8_1_mmq_ds4", "quantize_q8_1_mmq_ds4_x128",
        ]);
        add!("gemm_mq4g256v2_residual_wmma", kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_SRC, ["gemm_mq4g256v2_residual_wmma"]);
        add!("gemm_qkv_mq4g256v2_wmma", kernels::GEMM_QKV_MQ4G256V2_WMMA_SRC, ["gemm_qkv_mq4g256v2_wmma"]);
        add!("gemm_qkvza_mq4g256v2_wmma", kernels::GEMM_QKVZA_MQ4G256V2_WMMA_SRC, ["gemm_qkvza_mq4g256v2_wmma"]);
        add!("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, ["gemv_mq4g256", "mq_rotate_x"]);
        add!("rmsnorm_f32_rowsplit", kernels::RMSNORM_ROWSPLIT_SRC, ["rmsnorm_f32_rowsplit"]);
        add!("rope_partial_halfsplit_batched", kernels::ROPE_PARTIAL_HALFSPLIT_BATCHED_SRC, ["rope_partial_halfsplit_batched_f32"]);
        add!("rotate_x_mq_awq", kernels::ROTATE_X_MQ_AWQ_SRC, ["rotate_x_mq_awq"]);
        add!("sigmoid_mul_rotate_x_mq_awq_i4_gfx11", kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX11_SRC, ["sigmoid_mul_rotate_x_mq_awq_i4_gfx11"]);
        add!("softmax_temp_topp_batched", kernels::SOFTMAX_TEMP_BATCHED_SRC, ["softmax_temp_batched_f32", "softmax_temp_topp_batched_f32"]);
        add!("split_mq4v2_z_betaalpha", kernels::SPLIT_MQ4V2_Z_BETAALPHA_SRC, ["split_mq4v2_z_betaalpha"]);
        add!("topk_logsumexp_batched", kernels::TOPK_LOGSUMEXP_BATCHED_SRC, ["topk_logsumexp_batched_f32"]);
    }
    if arch == "gfx1201" {
        // Dense 27B DFlash cold helpers; retain runtime source preambles and
        // historical gfx1100 module/symbol names on gfx1201.
        add!("dflash_hidden_commit5_gfx1100", crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC, ["dflash_hidden_commit5_gfx1100", "dflash_hidden_scatter5_gfx1100"]);
        add!("dflash_hidden_scatter5_gfx1100", crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC, ["dflash_hidden_commit5_gfx1100", "dflash_hidden_scatter5_gfx1100"]);
        add!("gated_norm_mq_rotate_f16", crate::mq_f16_residual_producers::GATED_NORM_F16_SRC, ["gated_norm_mq_rotate_awq_f16_batched_gfx1100", "gated_norm_mq_rotate_f16_batched_gfx1100"]);
        add!("fused_silu_mul_mq_rotate_f16", crate::mq_f16_residual_producers::FUSED_SILU_F16_SRC, ["fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100", "fused_silu_mul_mq_rotate_f16_batched_gfx1100"]);
        add!("sigmoid_mul_mq_rotate_f16", crate::mq_f16_residual_producers::SIGMOID_MUL_F16_SRC, ["sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100", "sigmoid_mul_mq_rotate_f16_batched_gfx1100"]);
        add!("dflash_gdn_replay_pre_ml_from", crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC, ["dflash_gdn_replay_pre_ml_from"]);
        add!("gated_delta_net_q8_fast_ml_from", crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC, ["gated_delta_net_q8_fast_ml_from"]);
        add!("attention_verify_gqa_gfx1201", kernels::ATTENTION_VERIFY_GQA_GFX1201_SRC, ["attention_verify_gqa_q8_gfx1201", "attention_verify_gqa_fp8_gfx1201", "attention_verify_reduce_gfx1201"]);
        add!("attention_flash_fp8_e4m3_tile_batched", assemble_asym(kernels::ATTENTION_FLASH_FP8_E4M3_TILE_BATCHED_SRC), ["attention_flash_fp8_e4m3_tile_batched", "attention_flash_q8_0_tile_batched"]);
        add!("attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201", kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_GFX1201_SRC, ["attention_fp8_e4m3_fa2_gqa_gfx1201", "attention_fp8_e4m3_fa2_gqa_merge_gfx1201", "attention_fp8_e4m3_fa2_gqa_partial_gfx1201", "attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201", "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201", "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201"]);
        add!("attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201", kernels::ATTENTION_FP8_E4M3_FA2_GQA_QRESIDENT_V2_Q8_A4EPI_GFX1201_SRC, ["attention_fp8_e4m3_fa2_gqa_gfx1201", "attention_fp8_e4m3_fa2_gqa_merge_gfx1201", "attention_fp8_e4m3_fa2_gqa_partial_gfx1201", "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201", "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201", "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201"]);
        add!("dflash_gdn_replay_pre_ml", crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_SRC, ["dflash_gdn_replay_pre_ml"]);
        add!("fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SLAB_FDIV_SRC, ["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv"]);
        add!("fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast", kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_GFX12_SLAB_TOKFAST_SRC, ["fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast"]);
        add!("gated_delta_net_q8_fast_ml", crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_SRC, ["gated_delta_net_q8_fast_ml"]);
        add!("gated_norm_mq_rotate_awq_i4_gfx12_v2_slab", kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_SLAB_SRC, ["gated_norm_mq_rotate_awq_i4_gfx12_v2_slab"]);
        add!("gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab", kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX12_V2_XBF16_SLAB_SRC, ["gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab"]);
        add!("gdn_chunk_kkt_solve_batched", kernels::GDN_CHUNK_KKT_SOLVE_BATCHED_SRC, ["gdn_chunk_kkt_solve_batched"]);
        add!("gdn_chunk_prep", kernels::GDN_CHUNK_PREP_SRC, ["gdn_chunk_prep"]);
        add!("gdn_chunk_prep_fixup", kernels::GDN_CHUNK_PREP_FIXUP_SRC, ["gdn_chunk_prep_fixup"]);
        add!("gdn_chunk_scan_bf16", kernels::GDN_CHUNK_SCAN_BF16_SRC, ["gdn_chunk_scan_bf16"]);
        add!("gdn_chunk_scan_bf16_mseg", kernels::GDN_CHUNK_SCAN_BF16_MSEG_SRC, ["gdn_chunk_scan_bf16_mseg"]);
        add!("fn_gdn_dense", kernels::FN_GDN_DENSE_SRC, ["fn_gdn_dense_producer", "fn_gdn_dense_gate_bf16in", "fn_gdn_dense_gate_rotate_bf16in"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "quantize_int4_mmq_ds128"]);
        add!("qwen35_fa_prep_fp8q_nogate_batched_gfx1201", crate::qwen35_fa_batch::FA_PREP_BATCHED_GFX1201_SRC, ["qwen35_fa_prep_batched_gfx1201", "qwen35_fa_prep_fp8q_batched_gfx1201", "qwen35_fa_prep_fp8q_nogate_batched_gfx1201"]);
        add!("select_regrid", crate::select_regrid::SELECT_REGRID_SRC, ["argmax_f32_batched_regrid", "topk_values_regrid_f32"]);
        add!("sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab", kernels::SIGMOID_MUL_MQ_ROTATE_X_AWQ_I4_GFX12_SLAB_SRC, ["sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab"]);
        add!("topk_values_fixup_f32", crate::select_regrid::TOPK_VALUES_FIXUP_SRC, ["topk_values_fixup_f32"]);
    }
    if arch == "gfx1100" {
        add!("attention_verify_gqa_gfx1100", kernels::ATTENTION_VERIFY_GQA_GFX1100_SRC, ["attention_verify_gqa_q8_gfx1100", "attention_verify_gqa_q8_rows_gfx1100", "attention_verify_reduce_gfx1100"]);
        add!("attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_DEC_GFX1100_SRC, ["attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100"]);
        add!("attention_flash_q8_0_tile_batched", assemble_asym(kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC), ["attention_flash_q8_0_tile_batched"]);
        add!("attention_flash_q8_0_tile_gqa_gfx1100", kernels::ATTENTION_FLASH_Q8_0_TILE_GQA_GFX1100_SRC, ["attention_flash_q8_0_tile_gqa_gfx1100"]);
        add!("attention_q8_0_fa2_gqa_gfx1100", q8_fa2_gqa_gfx1100_source(), ["attention_fa2_q_preconvert_gfx1100", "attention_q8_0_fa2_gqa_gfx1100", "attention_q8_0_fa2_gqa_partial_gfx1100", "attention_q8_0_fa2_gqa_merge_gfx1100"]);
        add!("conv1d_silu_split_qknorm_b256_scalar_prep", kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SCALAR_PREP_SRC, ["conv1d_silu_split_qknorm_b256_scalar_prep"]);
        add!("dflash_gdn_pre_gfx1100", crate::dflash_gdn_pre::DFLASH_GDN_PRE_GFX1100_SRC, ["dflash_gdn_pre_capture_gfx1100", "dflash_gdn_pre_replay_gfx1100"]);
        add!("dflash_hidden_commit5_gfx1100", crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC, ["dflash_hidden_commit5_gfx1100", "dflash_hidden_scatter5_gfx1100"]);
        add!("dflash_hidden_scatter5_gfx1100", crate::dflash_hidden_scatter::DFLASH_HIDDEN_SCATTER_SRC, ["dflash_hidden_commit5_gfx1100", "dflash_hidden_scatter5_gfx1100"]);
        add!("dflash_state_bulk_copy_gfx1100", crate::dflash_state_copy::DFLASH_STATE_BULK_COPY_GFX1100_SRC, ["dflash_state_bulk_copy_gfx1100"]);
        add!("dynamic_conv_residual_gfx1100", crate::dflash_draft_fusion::COLLAPSE_SRC, ["dynamic_conv_residual_gfx1100", "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100", "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8", "mq_rotate_x_f16_dflash_gfx1100", "rmsnorm_residual_dual_gfx1100"]);
        add!("fused_rmsnorm_mq_rotate_awq_f16", crate::mq_f16_producers::FUSED_RMSNORM_MQ_ROTATE_F16_SRC, ["fused_rmsnorm_mq_rotate_awq_f16", "fused_rmsnorm_mq_rotate_f16"]);
        add!("fused_rmsnorm_mq_rotate_awq_i4_b8", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_B8_SRC, ["fused_rmsnorm_mq_rotate_awq_i4_b8"]);
        add!("fused_silu_mul_mq_rotate_f16", crate::mq_f16_residual_producers::FUSED_SILU_F16_SRC, ["fused_silu_mul_mq_rotate_awq_f16_batched_gfx1100", "fused_silu_mul_mq_rotate_f16_batched_gfx1100"]);
        add!("gated_delta_net_q8_compact3_b2", kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC, ["gated_delta_net_q8_compact3_b2", "gated_delta_net_q8_fast_independent_masked"]);
        add!("gated_norm_mq_rotate_awq_i4_gfx1100_v2", kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX1100_V2_SRC, ["gated_norm_mq_rotate_awq_i4_gfx1100_v2"]);
        add!("gated_norm_mq_rotate_awq_k6144_gfx1100", kernels::gated_norm_mq_rotate_awq_k6144_gfx1100_src(), ["gated_norm_mq_rotate_awq_k6144_gfx1100"]);
        add!("gated_norm_mq_rotate_f16", crate::mq_f16_residual_producers::GATED_NORM_F16_SRC, ["gated_norm_mq_rotate_awq_f16_batched_gfx1100", "gated_norm_mq_rotate_f16_batched_gfx1100"]);
        add!("gdn_chunk_kkt_solve_gfx1100", kernels::GDN_CHUNK_KKT_SOLVE_GFX1100_SRC, ["gdn_chunk_kkt_solve_gfx1100"]);
        add!("gdn_chunk_scan", kernels::GDN_CHUNK_SCAN_SRC, ["gdn_chunk_scan"]);
        add!("gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage", kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX1100_LDSSTAGE_SRC, ["gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "quantize_int4_mmq_ds128"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gridspec_symfold", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SYMFOLD_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold", "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec", "quantize_int4_mmq_ds128"]);
        add!("gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage", kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX1100_LDSSTAGE_SRC, ["gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage"]);
        add!("gemm_mqv2_wmma_gfx1100_mw_lds", kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC, ["gemm_gate_up_mq3g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq3g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq4g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq4g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq5g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq5g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq6g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq6g256v2_wmma_gfx11_mw8_lds", "gemm_mq3g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq3g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq4g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq5g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq5g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"]);
        add!("gemv_hfq4g256_residual_rdna3_mq4v2", kernels::GEMV_MQ4G256V2_RESIDUAL_SRC, ["gemv_mq4g256v2_residual"]);
        add!("gemv_mq4g256v2_rdna3_mq4v2", kernels::GEMV_MQ4G256V2_SRC, ["gemv_mq4g256v2"]);
        add!("kv_cache_write_q8_0_pair", kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC, ["kv_cache_write_q8_0_pair"]);
        add!("kv_cache_write_q8_0_pair_batched_gfx1100", crate::qwen35_fa_batch::KV_PAIR_BATCHED_SRC, ["kv_cache_write_q8_0_pair_batched_gfx1100"]);
        add!("qwen35_fa_prep_batched_gfx1100", crate::qwen35_fa_batch::FA_PREP_BATCHED_SRC, ["qwen35_fa_prep_batched_gfx1100"]);
        add!("qwen36_27b_fa_prep_gfx1100", kernels::qwen36_27b_fa_prep_gfx1100_src(), ["qwen36_27b_fa_prep_gfx1100"]);
        add!("rmsnorm_residual_dual_gfx1100", crate::dflash_draft_fusion::COLLAPSE_SRC, ["dynamic_conv_residual_gfx1100", "gemm_hfq4g256_overwrite_wmma_k2_dflash_gfx1100", "gemm_ksplit_det_overwrite_finalize_dflash_gfx1100", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks2", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks4", "gemm_mq4g256v2_overwrite_ksplit_lds_dflash_gfx1100_ks8", "mq_rotate_x_f16_dflash_gfx1100", "rmsnorm_residual_dual_gfx1100"]);
        add!("rope_partial_halfsplit", kernels::ROPE_PARTIAL_HALFSPLIT_SRC, ["rope_partial_halfsplit_f32"]);
        add!("sigmoid_mul_mq_rotate_f16", crate::mq_f16_residual_producers::SIGMOID_MUL_F16_SRC, ["sigmoid_mul_mq_rotate_awq_f16_batched_gfx1100", "sigmoid_mul_mq_rotate_f16_batched_gfx1100"]);
    }
    if arch == "gfx1151" {
        add!("attention_flash_q8_0_tile_gqa_gfx1151", kernels::ATTENTION_FLASH_Q8_0_TILE_GQA_GFX1151_SRC, ["attention_flash_q8_0_tile_gqa_gfx1151"]);
        add!("attention_flash_reduce_dsplit_gfx1151", kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1151_SRC, ["attention_flash_reduce_dsplit_gfx1151"]);
        add!("attention_q8_0_fa2_gqa_gfx1151", kernels::ATTENTION_Q8_0_FA2_GQA_GFX1151_SRC, ["attention_fa2_q_preconvert_gfx1151", "attention_q8_0_fa2_gqa_gfx1151"]);
        add!("conv1d_silu_split_qknorm_b256", kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC, ["conv1d_silu_split_qknorm_b256"]);
        add!("deinterleave_q_rmsnorm_f32_batched", kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC, ["deinterleave_q_rmsnorm_f32_batched"]);
        add!("fused_rmsnorm_mq_rotate_awq_i4_b8_gfx1151", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_B8_GFX1151_SRC, ["fused_rmsnorm_mq_rotate_awq_i4_b8_gfx1151"]);
        add!("fused_rmsnorm_mq_rotate_awq_i4_fold_b8", kernels::FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_FOLD_B8_SRC, ["fused_rmsnorm_mq_rotate_awq_i4_fold_b8"]);
        // Qwen4 chunked-GDN inline-Q8 recurrence (opt-in `HIPFIRE_QWEN4_GDN_Q8_INLINE`, exact gfx1151; tensor_ops.rs gated_delta_step_gate_wmma_arms).
        add!("gated_delta_chunk_q8_wmma", crate::tensor_ops::GATED_DELTA_CHUNK_Q8_WMMA_SRC, ["gated_delta_chunk_gate_q8_wmma"]);
        add!("gated_norm_mq_rotate_awq_i4_gfx1151_v2", kernels::GATED_NORM_MQ_ROTATE_AWQ_I4_GFX1151_V2_SRC, ["gated_norm_mq_rotate_awq_i4_gfx1151_v2"]);
        add!("gdn_chunk_scan_gfx1151", kernels::GDN_CHUNK_SCAN_GFX1151_SRC, ["gdn_chunk_scan_gfx1151"]);
        add!("gemm_mq4g256v2_residual_iu4_v2b_gfx11", kernels::GEMM_MQ4G256V2_RESIDUAL_IU4_V2B_GFX11_SRC, ["gemm_mq4g256v2_gate_up_silu_iu4_v2b_gfx11", "gemm_mq4g256v2_residual_iu4_v2b_add_gfx11", "gemm_mq4g256v2_residual_iu4_v2b_add_touch_gfx11", "gemm_mq4g256v2_residual_iu4_v2b_add_touch_swz_gfx11", "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11", "gemm_mq4g256v2_residual_iu4_v2b_set_zba_gfx11"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "quantize_int4_mmq_ds128"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gfx11_x5_symfold", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX11_X5_SYMFOLD_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_x5_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_x5_col_gfx1151_symfold", "quantize_int4_mmq_ds128"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gridspec_symfold", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SYMFOLD_SRC, ["gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec", "gemm_mq4g256v2_residual_mmq_iu4_full_add", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_symfold", "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold", "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec", "quantize_int4_mmq_ds128"]);
        add!("gemm_mqv2_wmma_gfx1151_mw_lds", kernels::GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC, ["gemm_gate_up_mq3g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq3g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq4g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq4g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq5g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq5g256v2_wmma_gfx11_mw8_lds", "gemm_gate_up_mq6g256v2_wmma_gfx11_mw4_lds", "gemm_gate_up_mq6g256v2_wmma_gfx11_mw8_lds", "gemm_mq3g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq3g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq4g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq5g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq5g256v2_residual_wmma_gfx11_mw8_lds", "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds", "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"]);
        add!("gemv_hfq4g256_multirow_default_mq4v2", kernels::GEMV_MQ4G256V2_MULTIROW_SRC, ["gemv_mq4g256v2_multirow_r2", "gemv_mq4g256v2_multirow_r4", "gemv_mq4g256v2_multirow_r8"]);
        add!("gemv_mq4g256v2_mq4v2", kernels::GEMV_MQ4G256V2_SRC, ["gemv_mq4g256v2"]);
        add!("gemv_mq4g256v2_residual_row_serial_gfx1151", kernels::GEMV_MQ4G256V2_RESIDUAL_ROW_SERIAL_GFX1151_SRC, ["gemv_mq4g256v2_residual_row_serial_gfx1151"]);
        add!("qwen35_fa_prep_batched_gfx1151", crate::qwen35_fa_batch::FA_PREP_BATCHED_GFX1151_SRC, ["qwen35_fa_prep_batched_gfx1151"]);
        add!("repeat_interleave_qk_batched", kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC, ["repeat_interleave_qk_f32_batched"]);
        add!("rope_partial_halfsplit_f32_headgrid", kernels::ROPE_PARTIAL_HALFSPLIT_HEADGRID_SRC, ["rope_partial_halfsplit_f32_headgrid"]);
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
        add!("copy_f32_buffer", kernels::COPY_F32_BUFFER_SRC, ["copy_f32_buffer", "copy_f32_strided_slot_buffer"]);
        add!("gemm_bf16_xf32_multirow", kernels::GEMM_BF16_XF32_MULTIROW_SRC, [
            "convert_bf16_to_f16", "gemm_bf16_xf32_multirow", "gemm_bf16_xf32_multirow_pto2", "gemm_bf16_xf32_multirow_rows3",
        ]);
        add!("gemm_bf16_xf32_multirow_r16w4_gfx1151", kernels::GEMM_BF16_XF32_MULTIROW_R16_GFX1151_SRC, [
            "gemm_bf16_xf32_multirow_r16_gfx1151", "gemm_bf16_xf32_multirow_r16w4_gfx1151", "gemm_bf16_xf32_multirow_r16w4t_gfx1151",
        ]);
        add!("gemm_bf16_xf32_multirow_r16w4t_gfx1151", kernels::GEMM_BF16_XF32_MULTIROW_R16_GFX1151_SRC, [
            "gemm_bf16_xf32_multirow_r16_gfx1151", "gemm_bf16_xf32_multirow_r16w4_gfx1151", "gemm_bf16_xf32_multirow_r16w4t_gfx1151",
        ]);
        add!("gemm_bf16_xf32_multirow_rows3", kernels::GEMM_BF16_XF32_MULTIROW_SRC, [
            "convert_bf16_to_f16", "gemm_bf16_xf32_multirow", "gemm_bf16_xf32_multirow_pto2", "gemm_bf16_xf32_multirow_rows3",
        ]);
        // Native MTP batched prompt fill, HC input_mix_up (K = 320).
        add!("gemm_bf16_xf32_multirow_pto2", kernels::GEMM_BF16_XF32_MULTIROW_SRC, [
            "convert_bf16_to_f16", "gemm_bf16_xf32_multirow", "gemm_bf16_xf32_multirow_pto2", "gemm_bf16_xf32_multirow_rows3",
        ]);
        add!("gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151", kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_GFX1151_SRC, [
            "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
        ]);
        add!("gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4", kernels::GEMM_MQ4G256V2_MOE_GROUPED_TOP10_O4_R8_X4_GFX1151_SRC, [
            "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4", "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4",
        ]);
        add!("gemm_mq6g256v2_f32_rows", kernels::GEMM_MQ6G256V2_F32_ROWS_SRC, [
            "gemm_mq6g256v2_f32_rows_r2", "gemm_mq6g256v2_f32_rows_r3", "gemm_mq6g256v2_f32_rows_r4", "gemm_mq6g256v2_f32_rows_r5",
            "gemm_mq6g256v2_f32_rows_r6", "gemm_mq6g256v2_f32_rows_r7", "gemm_mq6g256v2_f32_rows_r8", "gemm_mq6g256v2_f32_rows_x4_r2",
            "gemm_mq6g256v2_f32_rows_x4_r3", "gemm_mq6g256v2_f32_rows_x4_r4", "gemm_mq6g256v2_f32_rows_x4_r5", "gemm_mq6g256v2_f32_rows_x4_r6",
            "gemm_mq6g256v2_f32_rows_x4_r7", "gemm_mq6g256v2_f32_rows_x4_r8",
        ]);
        add!("gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded", kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC, [
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
        ]);
        add!("gemv_mq4g256v2_moe_gate_up_top10_indexed_batched", kernels::GEMV_MQ4G256V2_MOE_GATE_UP_TOP10_INDEXED_BATCHED_SRC, [
            "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched", "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
        ]);
        add!("grouped_ops", crate::grouped_ops::GROUPED_KERNEL_SRC, [
            "grouped_depthwise_conv_silu_add_bf16", "grouped_depthwise_conv_silu_add_f32", "grouped_gate_bf16", "grouped_gate_f32",
            "grouped_gather_convert_bf16", "grouped_gather_convert_bf16_f16", "grouped_linear_f32", "grouped_norm_bf16", "grouped_norm_f32",
            "ple_conv_add_bf16s", "ple_conv_add_bf16s_k4d3", "ple_gate_rows_bf16s", "ple_norm_inv",
        ]);
        add!("hc_streams_init_from_embed_batched", kernels::HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC, ["hc_streams_init_from_embed_batched"]);
        add!("moe_down_combine_grouped_top10", kernels::MOE_DOWN_COMBINE_GROUPED_TOP10_SRC, [
            "moe_combine_order_top10", "moe_down_combine_grouped_top10", "moe_down_combine_grouped_top10_bf16in",
            "moe_down_combine_grouped_top10_bf16in_zinit",
        ]);
        add!("moe_down_combine_top10_batched", kernels::MOE_DOWN_COMBINE_TOP10_BATCHED_SRC, ["moe_down_combine_top10_batched"]);
        add!("moe_gate_up_unscatter_silu_top10", kernels::MOE_GATE_UP_UNSCATTER_SILU_TOP10_SRC, [
            "moe_gate_up_unscatter_silu_rotate128_top10", "moe_gate_up_unscatter_silu_top10", "moe_gate_up_unscatter_silu_top10_bf16in",
            "moe_unscatter_rotate128_f16",
        ]);
        add!("moe_router_softmax_top10_f32", kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_SRC, ["moe_router_softmax_top10_f32"]);
        add!("moe_scatter_fused_top10", kernels::MOE_SCATTER_FUSED_TOP10_SRC, ["moe_scatter_fused_top10"]);
        add!("mq_rotate_x_128_v2", kernels::MQ_ROTATE_X_128_V2_SRC, ["mq_rotate_x_128_v2", "mq_rotate_x_128_v2_f16", "mq_rotate_x_128_v2_silu_bf16"]);
        add!("qwen4_gemv_bf16_xf32", kernels::QWEN4_GEMV_BF16_XF32_SRC, [
            "gemv_bf16_xf32", "gemv_bf16_xf32_bf16_scaled_add", "gemv_bf16_xf32_k4", "gemv_bf16_xf32_k4_rows_r2", "gemv_bf16_xf32_k4_rows_r3",
            "gemv_bf16_xf32_k4_rows_r4", "gemv_bf16_xf32_k4_rows_r5", "gemv_bf16_xf32_k4_rows_r6", "gemv_bf16_xf32_k4_rows_r7",
            "gemv_bf16_xf32_k4_rows_r8", "gemv_bf16_xf32_k4_rows_tiled_r2", "gemv_bf16_xf32_k4_rows_tiled_r3",
            "gemv_bf16_xf32_k4_rows_tiled_r4", "gemv_bf16_xf32_k4_rows_tiled_r5", "gemv_bf16_xf32_k4_rows_tiled_r6",
            "gemv_bf16_xf32_k4_rows_tiled_r7", "gemv_bf16_xf32_k4_rows_tiled_r8", "gemv_bf16_xf32_x4", "gemv_bf16_xf32_x4_rows_r2",
            "gemv_bf16_xf32_x4_rows_r3",
            "gemv_bf16_xf32_x4_rows_r4", "gemv_bf16_xf32_x4_rows_r5", "gemv_bf16_xf32_x4_rows_r6", "gemv_bf16_xf32_x4_rows_r7",
            "gemv_bf16_xf32_x4_rows_r8", "hyper_write_norm_f32",
        ]);
        add!("qwen4_gemv_mq4g256", kernels::QWEN4_GEMV_MQ4G256_SRC, ["hyper_read_projected_rotate_f32", "mq_rotate_x_bf16_f16", "mq_rotate_x_f16"]);
        add!("qwen4_gemv_mq6g256v2", kernels::QWEN4_GEMV_MQ6G256V2_SRC, ["gemv_mq6g256v2", "gemv_mq6g256v2_x4"]);
        add!("qwen4_gemv_q8_0", kernels::QWEN4_GEMV_Q8_0_SRC, [
            "gemv_q8_0_k2560_staged", "gemv_q8_0_k2560_staged_pair", "gemv_q8_0_k2560_staged_rows", "gemv_q8_0_k2560_staged_rows_tiled",
            "gemv_q8_0_k320_staged",
            "gemv_q8_0_k320_staged_rows", "gemv_q8_0_k8", "gemv_q8_0_k8_rows_r2", "gemv_q8_0_k8_rows_r3", "gemv_q8_0_k8_rows_r4",
            "gemv_q8_0_k8_rows_r5", "gemv_q8_0_k8_rows_r6", "gemv_q8_0_k8_rows_r7", "gemv_q8_0_k8_rows_r8", "quantize_bf16_q8_0",
            "topk8_partial_f32", "topk8_rescore_q8_0_k2560",
        ]);
        add!("qwen4_hc_streams_init_from_embed_batched", kernels::QWEN4_HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC, ["hc_streams_init_from_embed_batched_bf16"]);
        add!("qwen4_silu_mul", kernels::QWEN4_SILU_MUL_SRC, ["shared_expert_activation_bf16_f32", "silu_mul_bf16_rt_f32"]);
        add!("requant_g256", kernels::REQUANT_G256_SRC, ["requant_bf16_to_f32", "requant_mqg256v2_to_f32", "requant_pack_mqg256v2", "requant_q8_0_to_f32"]);
        add!("topk8_rescore_mq6g256v2", kernels::TOPK8_RESCORE_MQ6G256V2_SRC, ["gemv_mq6g256v2", "gemv_mq6g256v2_x4", "topk8_rescore_mq6g256v2_k2560"]);
        add!("zero_f32", kernels::ZERO_F32_SRC, ["zero_f32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1100") {
        add!("gemv_bf16_xf32", kernels::gemv_bf16_xf32_src(arch == "gfx1151"), ["gemv_bf16_xf32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!("gemv_mq2g256v2", kernels::GEMV_MQ2G256V2_SRC, ["gemv_mq2g256v2"]);
        add!("gemv_mq6g256v2_mq6v2", kernels::GEMV_MQ6G256V2_SRC, ["gemv_mq6g256v2"]);
        // Default-on `HIPFIRE_QWEN4_ROUTER_FAST`: every eager multi-row router call.
        add!("moe_router_softmax_top10_f32_fast", kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_FAST_SRC, ["moe_router_softmax_top10_f32_fast"]);
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!("copy_rows_strided_f32", kernels::COPY_ROWS_STRIDED_F32_SRC, ["copy_rows_strided_f32"]);
        add!("gated_delta_chunk_wmma", crate::tensor_ops::GATED_DELTA_CHUNK_WMMA_SRC, ["gated_delta_chunk_gate_wmma"]);
        add!("gemm_mq4g128v2_moe_grouped_wmma_gfx1151", kernels::GEMM_MQ4G128V2_MOE_GROUPED_WMMA_GFX1151_SRC, [
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151", "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out",
        ]);
        add!("gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out", kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC, [
            "gemm_mq4g256v2_moe_grouped_wmma_k2", "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
        ]);
        add!("gemm_mq6g256v2_residual_wmma", kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_SRC, ["gemm_mq6g256v2_residual_wmma"]);
        add!("hyper_read_up_wmma", crate::tensor_ops::HYPER_READ_UP_WMMA_SRC, ["hyper_read_up_wmma_bf16", "hyper_read_up_wmma_bf16_swap"]);
        add!("indexed_attention_dense_wmma", crate::tensor_ops::INDEXED_ATTENTION_DENSE_WMMA_SRC, [
            "indexed_attention_dense_wmma_f16", "indexed_attention_kv_f16",
        ]);
        add!("qwen4_bf16_round_trip", kernels::QWEN4_BF16_ROUND_TRIP_SRC, ["bf16_round_trip_f32_strided"]);
        add!("qwen4_gemm_mqv2_wmma_gfx11_bt", kernels::QWEN4_GEMM_MQV2_WMMA_GFX11_BT_SRC, [
            "gemm_gate_up_mq2g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq3g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq3g256v2_wmma_gfx11_bt6",
            "gemm_gate_up_mq5g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq5g256v2_wmma_gfx11_bt6", "gemm_gate_up_mq6g256v2_wmma_gfx11_bt12",
            "gemm_gate_up_mq6g256v2_wmma_gfx11_bt6", "gemm_mq2g256v2_residual_wmma_gfx11_bt4", "gemm_mq3g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq3g256v2_residual_wmma_gfx11_bt6", "gemm_mq3g256v2_residual_wmma_gfx11_bt8", "gemm_mq5g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq5g256v2_residual_wmma_gfx11_bt6", "gemm_mq5g256v2_residual_wmma_gfx11_bt8", "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt6", "gemm_mq6g256v2_residual_wmma_gfx11_bt8", "gemm_mq6g256v2_residual_wmma_gfx11_bt8_x4",
            "gemm_mq6g256v2_wmma_gfx11_bt8_x4", "gemm_mq6g256v2_wmma_gfx11_bt8_x4_bf16out", "gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw",
            "gemm_mq6g256v2_wmma_gfx11_bt8_x4_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b12_r4_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b12_r8_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p1_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r4_p2_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p1_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b16_r8_p2_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_dq_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p1_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_dq",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_dq_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r4_p2_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_dq",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_dq_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p1_regions", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_dq",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_dq_bf16out", "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_hcw",
            "gemm_mq6g256v2_wmma_gfx11_u3_b8_r8_p2_regions", "gemm_qkv_mq2g256v2_wmma_gfx11_bt4", "gemm_qkv_mq3g256v2_wmma_gfx11_bt12",
            "gemm_qkv_mq3g256v2_wmma_gfx11_bt4", "gemm_qkv_mq5g256v2_wmma_gfx11_bt12", "gemm_qkv_mq5g256v2_wmma_gfx11_bt4",
            "gemm_qkv_mq6g256v2_wmma_gfx11_bt12", "gemm_qkv_mq6g256v2_wmma_gfx11_bt4", "gemm_qkvza_mq2g256v2_wmma_gfx11_bt4",
            "gemm_qkvza_mq3g256v2_wmma_gfx11_bt12", "gemm_qkvza_mq3g256v2_wmma_gfx11_bt4", "gemm_qkvza_mq5g256v2_wmma_gfx11_bt12",
            "gemm_qkvza_mq5g256v2_wmma_gfx11_bt4", "gemm_qkvza_mq6g256v2_wmma_gfx11_bt12", "gemm_qkvza_mq6g256v2_wmma_gfx11_bt4",
        ]);
        add!("tensor_ops", crate::tensor_ops::TENSOR_OPS_SRC, [
            "argmax_f32", "bf16_roundtrip_f32", "bf16_scaled_add_batched_f32", "bf16_scaled_add_f32", "copy_regions_u32",
            "gated_delta_conv_bf16_f32", "gated_delta_conv_bf16_f32_batched_k4", "gated_delta_conv_params_bf16_f32",
            "gated_delta_conv_qknorm_bf16_f32_batched_k4", "gated_delta_gate_bf16_f32", "gated_delta_gate_bf16_f32_batched",
            "gated_delta_gate_rotate_bf16_f32_batched", "gated_delta_params_bf16_f32", "gated_delta_params_bf16_f32_batched",
            "gated_delta_params_f32", "gated_delta_qk_norm_bf16_batched", "gated_delta_rollback_layers_f32", "gated_delta_rollback_layers_q8",
            "gated_delta_step_f32", "gated_delta_step_gate_norm128_gfx1151", "gated_delta_step_gate_norm128_q8",
            "gated_delta_step_halves_state128_persistent256_capture_f32", "gated_delta_step_halves_state128_persistent256_capture_q8",
            "gated_delta_step_halves_state128_persistent256_f32", "gated_delta_step_halves_state128_persistent256_q8",
            "gated_delta_step_norm128_q8", "gated_delta_step_shared_norm128_gfx1151", "gdn_state_f32_to_q8", "gdn_state_q8_to_f32",
            "hc_activation_fused_f32", "hc_state_bf16_add_f32", "hc_state_bf16_to_f32", "hyper_norm_f32", "hyper_norm_gate_f32",
            "hyper_norm_gate_outputs", "hyper_norm_gate_outputs_f32", "hyper_read_f32", "hyper_read_projected_f32", "hyper_read_up_fused_f32",
            "hyper_write_bf16x2", "hyper_write_f32", "indexed_attention_attention_f32", "indexed_attention_attention_f32_batched",
            "indexed_attention_attention_f32_batched_hg12", "indexed_attention_attention_f32_batched_hg4",
            "indexed_attention_attention_f32_batched_serial",
            "indexed_attention_attention_f32_serial", "indexed_attention_cache_append_f32", "indexed_attention_cache_append_f32_batched",
            "indexed_attention_decode_prologue_f32", "indexed_attention_index_key_append_bf16_batched", "indexed_attention_norm_rope_f32",
            "indexed_attention_norm_rope_f32_batched", "indexed_attention_pool_rope_bf16", "indexed_attention_pool_rope_f32",
            "indexed_attention_reuse_selection", "indexed_attention_select_bf16_batched", "indexed_attention_select_bf16_batched_serial",
            "indexed_attention_select_f32", "indexed_attention_select_f32_batched", "indexed_attention_select_f32_batched_serial",
            "indexed_attention_select_f32_serial", "indexed_attention_select_from_scores", "indexed_attention_select_scores_rows16_f32",
            "indexed_attention_select_scores_rows8_f32", "scale_f32",
        ]);
    }
    if arch == "gfx1201" {
        add!("gemm_mq4g256v2_moe_grouped_wmma_gfx12", kernels::GEMM_MQ4G256V2_MOE_GROUPED_WMMA_GFX12_SRC, ["gemm_mq4g256v2_moe_grouped_wmma_gfx12"]);
        add!("gemm_mq6g256v2_residual_wmma_gfx12_bt12_mq5v2", kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC, [
            "gemm_mq6g256v2_residual_wmma_gfx12_bt12", "gemm_mq6g256v2_residual_wmma_gfx12_bt4", "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
            "gemm_mq6g256v2_wmma_gfx12_bt12_hcw", "gemm_mq6g256v2_wmma_gfx12_bt4_hcw", "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
        ]);
        add!("gemm_mq6g256v2_residual_wmma_gfx12_bt4_mq5v2", kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC, [
            "gemm_mq6g256v2_residual_wmma_gfx12_bt12", "gemm_mq6g256v2_residual_wmma_gfx12_bt4", "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
            "gemm_mq6g256v2_wmma_gfx12_bt12_hcw", "gemm_mq6g256v2_wmma_gfx12_bt4_hcw", "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
        ]);
        add!("gemm_mq6g256v2_residual_wmma_gfx12_bt8_mq5v2", kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_BT_SRC, [
            "gemm_mq6g256v2_residual_wmma_gfx12_bt12", "gemm_mq6g256v2_residual_wmma_gfx12_bt4", "gemm_mq6g256v2_residual_wmma_gfx12_bt8",
            "gemm_mq6g256v2_wmma_gfx12_bt12_hcw", "gemm_mq6g256v2_wmma_gfx12_bt4_hcw", "gemm_mq6g256v2_wmma_gfx12_bt8_hcw",
        ]);
        add!("gemm_mq6g256v2_residual_wmma_gfx12_mq5v2", kernels::GEMM_MQ6G256V2_RESIDUAL_WMMA_GFX12_SRC, ["gemm_mq6g256v2_residual_wmma_gfx12"]);
        add!("qwen4_gemm_mq6g256v2_wmma_gfx12_x4", kernels::QWEN4_GEMM_MQ6G256V2_WMMA_GFX12_X4_SRC, [
            "gemm_mq6g256v2_residual_wmma_gfx12_bt8_x4", "gemm_mq6g256v2_wmma_gfx12_bt8_x4", "gemm_mq6g256v2_wmma_gfx12_bt8_x4_bf16out",
            "gemm_mq6g256v2_wmma_gfx12_bt8_x4_regions",
        ]);
        add!("tensor_ops", crate::tensor_ops::TENSOR_OPS_SRC, [
            "argmax_f32", "bf16_roundtrip_f32", "bf16_scaled_add_batched_f32", "bf16_scaled_add_f32", "copy_regions_u32",
            "gated_delta_conv_bf16_f32", "gated_delta_conv_bf16_f32_batched_k4", "gated_delta_conv_params_bf16_f32",
            "gated_delta_conv_qknorm_bf16_f32_batched_k4", "gated_delta_gate_bf16_f32", "gated_delta_gate_bf16_f32_batched",
            "gated_delta_gate_rotate_bf16_f32_batched", "gated_delta_params_bf16_f32", "gated_delta_params_bf16_f32_batched",
            "gated_delta_params_f32", "gated_delta_qk_norm_bf16_batched", "gated_delta_rollback_layers_f32", "gated_delta_rollback_layers_q8",
            "gated_delta_step_f32", "gated_delta_step_gate_norm128_gfx1151", "gated_delta_step_gate_norm128_q8",
            "gated_delta_step_halves_state128_persistent256_capture_f32", "gated_delta_step_halves_state128_persistent256_capture_q8",
            "gated_delta_step_halves_state128_persistent256_f32", "gated_delta_step_halves_state128_persistent256_q8",
            "gated_delta_step_norm128_q8", "gated_delta_step_pipe128_f32", "gated_delta_step_pipe128_q8",
            "gated_delta_step_shared_norm128_gfx1151", "gdn_state_f32_to_q8", "gdn_state_q8_to_f32",
            "hc_activation_fused_f32", "hc_state_bf16_add_f32", "hc_state_bf16_to_f32", "hyper_norm_f32", "hyper_norm_gate_f32",
            "hyper_norm_gate_outputs", "hyper_norm_gate_outputs_f32", "hyper_read_f32", "hyper_read_projected_f32", "hyper_read_up_fused_f32",
            "hyper_write_bf16x2", "hyper_write_f32", "indexed_attention_attention_f32", "indexed_attention_attention_f32_batched",
            "indexed_attention_attention_f32_batched_hg12", "indexed_attention_attention_f32_batched_hg4",
            "indexed_attention_attention_f32_batched_serial",
            "indexed_attention_attention_f32_serial", "indexed_attention_attention_fp8", "indexed_attention_attention_fp8_batched",
            "indexed_attention_attention_fp8_batched_hg4", "indexed_attention_attention_fp8_batched_serial",
            "indexed_attention_attention_fp8_serial", "indexed_attention_cache_append_f32", "indexed_attention_cache_append_f32_batched",
            "indexed_attention_cache_append_fp8_batched", "indexed_attention_decode_prologue_f32", "indexed_attention_decode_prologue_fp8",
            "indexed_attention_index_key_append_bf16_batched", "indexed_attention_norm_rope_f32", "indexed_attention_norm_rope_f32_batched",
            "indexed_attention_pool_rope_bf16", "indexed_attention_pool_rope_f32", "indexed_attention_reuse_selection",
            "indexed_attention_select_bf16_batched", "indexed_attention_select_bf16_batched_serial", "indexed_attention_select_f32",
            "indexed_attention_select_f32_batched", "indexed_attention_select_f32_batched_serial", "indexed_attention_select_f32_serial",
            "indexed_attention_select_from_scores", "indexed_attention_select_scores_rows16_f32",
            "indexed_attention_select_scores_rows8_f32",
            "scale_f32",
        ]);
        // Default >= 512-row prefill: BF16 -> F16 WMMA projections (HC
        // input_mix_down 160 x 80 split-K tile + reducer, router/shared 128 x
        // 128, wider projections 128 x 256), the HC read-up tile, and the
        // symmetric-IU4 MoE load check, A4 producers and stable scatter.
        add!("gemm_wmma_lds_splitk", kernels::GEMM_F16_X_F16_WMMA_LDS_SPLITK_SRC, [
            "qwen4_wmma_lds_128_128_32_64_k64", "qwen4_wmma_lds_128_256_32_64_k64", "qwen4_wmma_lds_128_256_32_64_k64_bsr",
            "qwen4_wmma_lds_160_80_32_80_k32_p_s4", "qwen4_wmma_lds_64_128_32_32_k64_p_s4", "qwen4_wmma_lds_64_128_32_64_k64_p",
            "qwen4_wmma_lds_64_128_32_64_k64_p_s2", "qwen4_wmma_lds_64_128_32_64_k64_p_s4", "qwen4_wmma_lds_64_128_32_64_k64_p_s8",
            "qwen4_wmma_lds_64_128_32_64_k64_s4", "qwen4_wmma_lds_64_64_32_32_k64_p_s4", "qwen4_wmma_lds_64_64_32_64_k64_p",
            "qwen4_wmma_lds_64_64_32_64_k64_p_s2", "qwen4_wmma_lds_64_64_32_64_k64_p_s4", "qwen4_wmma_lds_reduce",
        ]);
        add!("hyper_read_up_wmma_gfx1201", crate::tensor_ops::HYPER_READ_UP_WMMA_GFX1201_SRC, [
            "hyper_read_up_wmma_bf16_gfx1201", "hyper_read_up_wmma_bf16_gfx1201_t128",
        ]);
        add!("mq_rotate_x_i4", kernels::MQ_ROTATE_X_I4_SRC, ["mq_rotate_x_i4"]);
        add!("qwen4_moe_rotate128_i4", kernels::QWEN4_MOE_ROTATE128_I4_SRC, ["qwen4_moe_rotate128_i4"]);
        add!("qwen4_moe_scatter_stable_top10", kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_SRC, [
            "qwen4_moe_group_prefix", "qwen4_moe_group_ranks", "qwen4_moe_group_scatter",
        ]);
        add!("qwen4_moe_sym_check_gfx1201", kernels::QWEN4_MOE_SYM_CHECK_GFX1201_SRC, ["qwen4_moe_sym_check_gfx1201"]);
    }
    if arch == "gfx1100" {
        add!("add", kernels::ADD_SRC, ["add_f32", "broadcast_add_rows_f32"]);
        add!("gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out", kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC, [
            "gemm_mq4g256v2_moe_grouped_wmma_k2", "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
        ]);
        add!("gemm_mqv2_wmma_gfx1100_bt", kernels::gemm_mqv2_wmma_gfx11_bt_src(arch == "gfx1151"), [
            "gemm_gate_up_mq2g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq3g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq3g256v2_wmma_gfx11_bt6",
            "gemm_gate_up_mq5g256v2_wmma_gfx11_bt12", "gemm_gate_up_mq5g256v2_wmma_gfx11_bt6", "gemm_gate_up_mq6g256v2_wmma_gfx11_bt12",
            "gemm_gate_up_mq6g256v2_wmma_gfx11_bt6", "gemm_mq2g256v2_residual_wmma_gfx11_bt4", "gemm_mq3g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq3g256v2_residual_wmma_gfx11_bt6", "gemm_mq3g256v2_residual_wmma_gfx11_bt8", "gemm_mq5g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq5g256v2_residual_wmma_gfx11_bt6", "gemm_mq5g256v2_residual_wmma_gfx11_bt8", "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt6", "gemm_mq6g256v2_residual_wmma_gfx11_bt8", "gemm_qkv_mq2g256v2_wmma_gfx11_bt4",
            "gemm_qkv_mq3g256v2_wmma_gfx11_bt12", "gemm_qkv_mq3g256v2_wmma_gfx11_bt4", "gemm_qkv_mq5g256v2_wmma_gfx11_bt12",
            "gemm_qkv_mq5g256v2_wmma_gfx11_bt4", "gemm_qkv_mq6g256v2_wmma_gfx11_bt12", "gemm_qkv_mq6g256v2_wmma_gfx11_bt4",
            "gemm_qkvza_mq2g256v2_wmma_gfx11_bt4", "gemm_qkvza_mq3g256v2_wmma_gfx11_bt12", "gemm_qkvza_mq3g256v2_wmma_gfx11_bt4",
            "gemm_qkvza_mq5g256v2_wmma_gfx11_bt12", "gemm_qkvza_mq5g256v2_wmma_gfx11_bt4", "gemm_qkvza_mq6g256v2_wmma_gfx11_bt12",
            "gemm_qkvza_mq6g256v2_wmma_gfx11_bt4",
        ]);
        add!("gemm_wmma_lds256", kernels::gemm_f16_x_f16_wmma_lds256_src(arch == "gfx1151"), [
            "gemm_wmma_lds_128_128_32_64_k64", "gemm_wmma_lds_128_128_32_64_k64_a", "gemm_wmma_lds_128_128_32_64_k64_gr",
            "gemm_wmma_lds_128_128_32_64_k64_gra", "gemm_wmma_lds_128_128_32_64_k64_o16", "gemm_wmma_lds_128_128_32_64_k64_o16g",
            "gemm_wmma_lds_128_128_32_64_k64_p", "gemm_wmma_lds_128_128_32_64_k64_p_a", "gemm_wmma_lds_128_128_32_64_k64_p_gr",
            "gemm_wmma_lds_128_128_32_64_k64_p_gra", "gemm_wmma_lds_128_128_32_64_k64_p_o16", "gemm_wmma_lds_128_128_32_64_k64_p_o16g",
            "gemm_wmma_lds_128_128_64_64_k64", "gemm_wmma_lds_128_128_64_64_k64_a", "gemm_wmma_lds_128_128_64_64_k64_gr",
            "gemm_wmma_lds_128_128_64_64_k64_gra", "gemm_wmma_lds_128_128_64_64_k64_o16", "gemm_wmma_lds_128_128_64_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64", "gemm_wmma_lds_128_256_32_64_k64_a", "gemm_wmma_lds_128_256_32_64_k64_gr",
            "gemm_wmma_lds_128_256_32_64_k64_gra", "gemm_wmma_lds_128_256_32_64_k64_o16", "gemm_wmma_lds_128_256_32_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64_p", "gemm_wmma_lds_128_256_32_64_k64_p_a", "gemm_wmma_lds_128_256_32_64_k64_p_gr",
            "gemm_wmma_lds_128_256_32_64_k64_p_gra", "gemm_wmma_lds_128_256_32_64_k64_p_o16", "gemm_wmma_lds_128_256_32_64_k64_p_o16g",
            "gemm_wmma_lds_128_256_64_64_k32", "gemm_wmma_lds_128_256_64_64_k32_a", "gemm_wmma_lds_128_256_64_64_k32_gr",
            "gemm_wmma_lds_128_256_64_64_k32_gra", "gemm_wmma_lds_128_256_64_64_k32_o16", "gemm_wmma_lds_128_256_64_64_k32_o16g",
            "gemm_wmma_lds_128_256_64_64_k32_p", "gemm_wmma_lds_128_256_64_64_k32_p_a", "gemm_wmma_lds_128_256_64_64_k32_p_gr",
            "gemm_wmma_lds_128_256_64_64_k32_p_gra", "gemm_wmma_lds_128_256_64_64_k32_p_o16", "gemm_wmma_lds_128_256_64_64_k32_p_o16g",
            "gemm_wmma_lds_128_256_64_64_k64", "gemm_wmma_lds_128_256_64_64_k64_sw", "gemm_wmma_lds_128_512_32_64_k32",
            "gemm_wmma_lds_128_512_64_64_k32", "gemm_wmma_lds_128_512_64_64_k32_sw", "gemm_wmma_lds_256_128_32_64_k64",
            "gemm_wmma_lds_256_128_64_64_k64", "gemm_wmma_lds_256_128_64_64_k64_sw", "gemm_wmma_lds_256_256_32_64_k64",
            "gemm_wmma_lds_256_256_64_64_k64", "gemm_wmma_lds_256_256_64_64_k64_a", "gemm_wmma_lds_256_256_64_64_k64_gr",
            "gemm_wmma_lds_256_256_64_64_k64_gra", "gemm_wmma_lds_256_256_64_64_k64_o16", "gemm_wmma_lds_256_256_64_64_k64_o16g",
            "gemm_wmma_lds_256_256_64_64_k64_p", "gemm_wmma_lds_256_256_64_64_k64_p_a", "gemm_wmma_lds_256_256_64_64_k64_p_gr",
            "gemm_wmma_lds_256_256_64_64_k64_p_gra", "gemm_wmma_lds_256_256_64_64_k64_p_o16", "gemm_wmma_lds_256_256_64_64_k64_p_o16g",
            "gemm_wmma_lds_64_512_64_64_k32",
        ]);
        add!("gemv_mq2g256v2_rdna3", kernels::GEMV_MQ2G256V2_SRC, ["gemv_mq2g256v2"]);
        add!("gemv_mq6g256v2_rdna3_mq6v2", kernels::GEMV_MQ6G256V2_SRC, ["gemv_mq6g256v2"]);
        add!("qwen4_gemm_wmma_lds256", kernels::QWEN4_GEMM_F16_X_F16_WMMA_LDS256_SRC, [
            "gemm_f16_x_f16_wmma_lds_regions_128_128_32_64_k64_p", "gemm_wmma_lds_128_128_32_64_k64", "gemm_wmma_lds_128_128_32_64_k64_a",
            "gemm_wmma_lds_128_128_32_64_k64_bsr", "gemm_wmma_lds_128_128_32_64_k64_gr", "gemm_wmma_lds_128_128_32_64_k64_gra",
            "gemm_wmma_lds_128_128_32_64_k64_hcsd", "gemm_wmma_lds_128_128_32_64_k64_o16", "gemm_wmma_lds_128_128_32_64_k64_o16g",
            "gemm_wmma_lds_128_128_32_64_k64_p", "gemm_wmma_lds_128_128_32_64_k64_p_a", "gemm_wmma_lds_128_128_32_64_k64_p_bsr",
            "gemm_wmma_lds_128_128_32_64_k64_p_gr", "gemm_wmma_lds_128_128_32_64_k64_p_gra", "gemm_wmma_lds_128_128_32_64_k64_p_hcsd",
            "gemm_wmma_lds_128_128_32_64_k64_p_o16", "gemm_wmma_lds_128_128_32_64_k64_p_o16g", "gemm_wmma_lds_128_128_64_64_k64",
            "gemm_wmma_lds_128_128_64_64_k64_a", "gemm_wmma_lds_128_128_64_64_k64_bsr", "gemm_wmma_lds_128_128_64_64_k64_gr",
            "gemm_wmma_lds_128_128_64_64_k64_gra", "gemm_wmma_lds_128_128_64_64_k64_o16", "gemm_wmma_lds_128_128_64_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64", "gemm_wmma_lds_128_256_32_64_k64_a", "gemm_wmma_lds_128_256_32_64_k64_bf16st",
            "gemm_wmma_lds_128_256_32_64_k64_bsr", "gemm_wmma_lds_128_256_32_64_k64_gr", "gemm_wmma_lds_128_256_32_64_k64_gra",
            "gemm_wmma_lds_128_256_32_64_k64_hcsd", "gemm_wmma_lds_128_256_32_64_k64_o16", "gemm_wmma_lds_128_256_32_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64_p", "gemm_wmma_lds_128_256_32_64_k64_p_a", "gemm_wmma_lds_128_256_32_64_k64_p_bsr",
            "gemm_wmma_lds_128_256_32_64_k64_p_gr", "gemm_wmma_lds_128_256_32_64_k64_p_gra", "gemm_wmma_lds_128_256_32_64_k64_p_hcsd",
            "gemm_wmma_lds_128_256_32_64_k64_p_o16", "gemm_wmma_lds_128_256_32_64_k64_p_o16g", "gemm_wmma_lds_128_256_64_64_k32",
            "gemm_wmma_lds_128_256_64_64_k32_a", "gemm_wmma_lds_128_256_64_64_k32_bsr", "gemm_wmma_lds_128_256_64_64_k32_gr",
            "gemm_wmma_lds_128_256_64_64_k32_gra", "gemm_wmma_lds_128_256_64_64_k32_o16", "gemm_wmma_lds_128_256_64_64_k32_o16g",
            "gemm_wmma_lds_128_256_64_64_k32_p", "gemm_wmma_lds_128_256_64_64_k32_p_a", "gemm_wmma_lds_128_256_64_64_k32_p_bsr",
            "gemm_wmma_lds_128_256_64_64_k32_p_gr", "gemm_wmma_lds_128_256_64_64_k32_p_gra", "gemm_wmma_lds_128_256_64_64_k32_p_o16",
            "gemm_wmma_lds_128_256_64_64_k32_p_o16g", "gemm_wmma_lds_128_256_64_64_k64", "gemm_wmma_lds_128_256_64_64_k64_sw",
            "gemm_wmma_lds_128_512_32_64_k32", "gemm_wmma_lds_128_512_64_64_k32", "gemm_wmma_lds_128_512_64_64_k32_sw",
            "gemm_wmma_lds_160_64_32_64_k64", "gemm_wmma_lds_160_64_32_64_k64_p", "gemm_wmma_lds_256_128_32_64_k64",
            "gemm_wmma_lds_256_128_64_64_k64", "gemm_wmma_lds_256_128_64_64_k64_sw", "gemm_wmma_lds_256_256_32_64_k64",
            "gemm_wmma_lds_256_256_64_64_k64", "gemm_wmma_lds_256_256_64_64_k64_a", "gemm_wmma_lds_256_256_64_64_k64_bsr",
            "gemm_wmma_lds_256_256_64_64_k64_gr", "gemm_wmma_lds_256_256_64_64_k64_gra", "gemm_wmma_lds_256_256_64_64_k64_hcsd",
            "gemm_wmma_lds_256_256_64_64_k64_o16", "gemm_wmma_lds_256_256_64_64_k64_o16g", "gemm_wmma_lds_256_256_64_64_k64_p",
            "gemm_wmma_lds_256_256_64_64_k64_p_a", "gemm_wmma_lds_256_256_64_64_k64_p_bsr", "gemm_wmma_lds_256_256_64_64_k64_p_gr",
            "gemm_wmma_lds_256_256_64_64_k64_p_gra", "gemm_wmma_lds_256_256_64_64_k64_p_hcsd", "gemm_wmma_lds_256_256_64_64_k64_p_o16",
            "gemm_wmma_lds_256_256_64_64_k64_p_o16g", "gemm_wmma_lds_64_512_64_64_k32", "gemm_wmma_lds_64_64_32_64_k64",
            "gemm_wmma_lds_64_64_32_64_k64_p",
        ]);
    }
    if arch == "gfx1151" {
        add!("gemm_wmma_lds256", kernels::gemm_f16_x_f16_wmma_lds256_src(arch == "gfx1151"), [
            "gemm_f16_x_f16_wmma_lds_regions_128_128_32_64_k64_p", "gemm_wmma_lds_128_128_32_64_k64", "gemm_wmma_lds_128_128_32_64_k64_a",
            "gemm_wmma_lds_128_128_32_64_k64_bsr", "gemm_wmma_lds_128_128_32_64_k64_gr", "gemm_wmma_lds_128_128_32_64_k64_gra",
            "gemm_wmma_lds_128_128_32_64_k64_hcsd", "gemm_wmma_lds_128_128_32_64_k64_o16", "gemm_wmma_lds_128_128_32_64_k64_o16g",
            "gemm_wmma_lds_128_128_32_64_k64_p", "gemm_wmma_lds_128_128_32_64_k64_p_a", "gemm_wmma_lds_128_128_32_64_k64_p_bsr",
            "gemm_wmma_lds_128_128_32_64_k64_p_gr", "gemm_wmma_lds_128_128_32_64_k64_p_gra", "gemm_wmma_lds_128_128_32_64_k64_p_hcsd",
            "gemm_wmma_lds_128_128_32_64_k64_p_o16", "gemm_wmma_lds_128_128_32_64_k64_p_o16g", "gemm_wmma_lds_128_128_64_64_k64",
            "gemm_wmma_lds_128_128_64_64_k64_a", "gemm_wmma_lds_128_128_64_64_k64_bsr", "gemm_wmma_lds_128_128_64_64_k64_gr",
            "gemm_wmma_lds_128_128_64_64_k64_gra", "gemm_wmma_lds_128_128_64_64_k64_o16", "gemm_wmma_lds_128_128_64_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64", "gemm_wmma_lds_128_256_32_64_k64_a", "gemm_wmma_lds_128_256_32_64_k64_bf16st",
            "gemm_wmma_lds_128_256_32_64_k64_bsr", "gemm_wmma_lds_128_256_32_64_k64_gr", "gemm_wmma_lds_128_256_32_64_k64_gra",
            "gemm_wmma_lds_128_256_32_64_k64_hcsd", "gemm_wmma_lds_128_256_32_64_k64_o16", "gemm_wmma_lds_128_256_32_64_k64_o16g",
            "gemm_wmma_lds_128_256_32_64_k64_p", "gemm_wmma_lds_128_256_32_64_k64_p_a", "gemm_wmma_lds_128_256_32_64_k64_p_bsr",
            "gemm_wmma_lds_128_256_32_64_k64_p_gr", "gemm_wmma_lds_128_256_32_64_k64_p_gra", "gemm_wmma_lds_128_256_32_64_k64_p_hcsd",
            "gemm_wmma_lds_128_256_32_64_k64_p_o16", "gemm_wmma_lds_128_256_32_64_k64_p_o16g", "gemm_wmma_lds_128_256_64_64_k32",
            "gemm_wmma_lds_128_256_64_64_k32_a", "gemm_wmma_lds_128_256_64_64_k32_bsr", "gemm_wmma_lds_128_256_64_64_k32_gr",
            "gemm_wmma_lds_128_256_64_64_k32_gra", "gemm_wmma_lds_128_256_64_64_k32_o16", "gemm_wmma_lds_128_256_64_64_k32_o16g",
            "gemm_wmma_lds_128_256_64_64_k32_p", "gemm_wmma_lds_128_256_64_64_k32_p_a", "gemm_wmma_lds_128_256_64_64_k32_p_bsr",
            "gemm_wmma_lds_128_256_64_64_k32_p_gr", "gemm_wmma_lds_128_256_64_64_k32_p_gra", "gemm_wmma_lds_128_256_64_64_k32_p_o16",
            "gemm_wmma_lds_128_256_64_64_k32_p_o16g", "gemm_wmma_lds_128_256_64_64_k64", "gemm_wmma_lds_128_256_64_64_k64_sw",
            "gemm_wmma_lds_128_512_32_64_k32", "gemm_wmma_lds_128_512_64_64_k32", "gemm_wmma_lds_128_512_64_64_k32_sw",
            "gemm_wmma_lds_160_64_32_64_k64", "gemm_wmma_lds_160_64_32_64_k64_p", "gemm_wmma_lds_256_128_32_64_k64",
            "gemm_wmma_lds_256_128_64_64_k64", "gemm_wmma_lds_256_128_64_64_k64_sw", "gemm_wmma_lds_256_256_32_64_k64",
            "gemm_wmma_lds_256_256_64_64_k64", "gemm_wmma_lds_256_256_64_64_k64_a", "gemm_wmma_lds_256_256_64_64_k64_bsr",
            "gemm_wmma_lds_256_256_64_64_k64_gr", "gemm_wmma_lds_256_256_64_64_k64_gra", "gemm_wmma_lds_256_256_64_64_k64_hcsd",
            "gemm_wmma_lds_256_256_64_64_k64_o16", "gemm_wmma_lds_256_256_64_64_k64_o16g", "gemm_wmma_lds_256_256_64_64_k64_p",
            "gemm_wmma_lds_256_256_64_64_k64_p_a", "gemm_wmma_lds_256_256_64_64_k64_p_bsr", "gemm_wmma_lds_256_256_64_64_k64_p_gr",
            "gemm_wmma_lds_256_256_64_64_k64_p_gra", "gemm_wmma_lds_256_256_64_64_k64_p_hcsd", "gemm_wmma_lds_256_256_64_64_k64_p_o16",
            "gemm_wmma_lds_256_256_64_64_k64_p_o16g", "gemm_wmma_lds_64_512_64_64_k32", "gemm_wmma_lds_64_64_32_64_k64",
            "gemm_wmma_lds_64_64_32_64_k64_p",
        ]);
        add!("gemv_bf16_xf32", kernels::gemv_bf16_xf32_src(arch == "gfx1151"), [
            "gemv_bf16_xf32", "gemv_bf16_xf32_bf16_scaled_add", "gemv_bf16_xf32_k4", "gemv_bf16_xf32_k4_rows_r2", "gemv_bf16_xf32_k4_rows_r3",
            "gemv_bf16_xf32_k4_rows_r4", "gemv_bf16_xf32_k4_rows_r5", "gemv_bf16_xf32_k4_rows_r6", "gemv_bf16_xf32_k4_rows_r7",
            "gemv_bf16_xf32_k4_rows_r8", "gemv_bf16_xf32_k4_rows_tiled_r2", "gemv_bf16_xf32_k4_rows_tiled_r3",
            "gemv_bf16_xf32_k4_rows_tiled_r4", "gemv_bf16_xf32_k4_rows_tiled_r5", "gemv_bf16_xf32_k4_rows_tiled_r6",
            "gemv_bf16_xf32_k4_rows_tiled_r7", "gemv_bf16_xf32_k4_rows_tiled_r8", "gemv_bf16_xf32_x4", "gemv_bf16_xf32_x4_rows_r2",
            "gemv_bf16_xf32_x4_rows_r3",
            "gemv_bf16_xf32_x4_rows_r4", "gemv_bf16_xf32_x4_rows_r5", "gemv_bf16_xf32_x4_rows_r6", "gemv_bf16_xf32_x4_rows_r7",
            "gemv_bf16_xf32_x4_rows_r8", "hyper_write_norm_f32",
        ]);
        add!("mq_rotate_x_i4", kernels::MQ_ROTATE_X_I4_SRC, ["mq_rotate_x_i4"]);
        add!("qwen4_gemv_q8_0_wide", kernels::QWEN4_GEMV_Q8_0_WIDE_SRC, ["gemv_q8_0_wide_k640_staged", "gemv_q8_0_wide_rows"]);
        add!("qwen4_moe_iu4_sym_gfx1151", kernels::QWEN4_MOE_IU4_SYM_GFX1151_SRC, [
            "qwen4_moe_down_iu4_sym_gfx1151", "qwen4_moe_down_iu4_sym_gfx1151_nt4", "qwen4_moe_gate_up_silu_iu4_sym_gfx1151",
            "qwen4_moe_gate_up_silu_iu4_sym_gfx1151_nt4", "qwen4_moe_sym_check_gfx1151",
        ]);
        add!("qwen4_moe_rotate128_i4", kernels::QWEN4_MOE_ROTATE128_I4_SRC, ["qwen4_moe_rotate128_i4"]);
        add!("qwen4_moe_scatter_stable_top10", kernels::QWEN4_MOE_SCATTER_STABLE_TOP10_SRC, [
            "qwen4_moe_group_prefix", "qwen4_moe_group_ranks", "qwen4_moe_group_scatter",
        ]);
        // Default-on `HIPFIRE_QWEN4_HC_ROW_FOLD`: grouped MoE prefill fold.
        add!("hc_row_fold", crate::hc_row_fold::SRC, ["hc_row_fold_norm_gate"]);
    }
    // Qwen3.5-MoE (ornith-1.5-35b-a3b) load, AR prefill/decode and native MTP,
    // plus Qwen3.5 dense (H2): every module a kernel-load trace
    // (tests/fixtures/kernel-trace-qwen35.tsv) JIT-compiled on gfx1100,
    // gfx1151 and gfx1201 beyond the inventory above. Sources are the exact
    // expressions the callsites pass to `ensure_kernel`; symbols are every
    // kernel the arch's object defines.
    if matches!(arch, "gfx1201" | "gfx1100" | "gfx1151") {
        add!("gated_delta_net_q8_compact2_b2", kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC, [
            "gated_delta_net_q8_compact2_b2", "gated_delta_net_q8_fast_independent_masked",
        ]);
        add!("gemm_qkv_hfq6g256", kernels::GEMM_QKV_HFQ6G256_SRC, ["gemm_qkv_hfq6g256"]);
        add!("gemm_qkvza_hfq6g256", kernels::GEMM_QKVZA_HFQ6G256_SRC, ["gemm_qkvza_hfq6g256"]);
        add!("gemv_hfq6g256_residual", kernels::GEMV_HFQ6G256_RESIDUAL_SRC, ["gemv_hfq6g256_residual"]);
        add!("gemv_hfq6g256_residual_sigmoid_scaled", kernels::GEMV_HFQ6G256_RESIDUAL_SIGMOID_SCALED_SRC, [
            "gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched",
        ]);
        add!("gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded", kernels::GEMV_MQ4G256V2_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC, [
            "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",
        ]);
        add!("moe_down_combine_grouped_k8", kernels::MOE_DOWN_COMBINE_GROUPED_K8_SRC, ["moe_down_combine_grouped_k8"]);
        add!("moe_down_combine_k8_batched", kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC, ["moe_down_combine_k8_batched"]);
        add!("moe_gate_up_unscatter_k8", kernels::MOE_GATE_UP_UNSCATTER_K8_SRC, ["moe_gate_up_unscatter_k8"]);
        add!("moe_scatter_fused_k8", kernels::moe_scatter_fused_k8_src(arch == "gfx1151"), ["moe_scatter_fused_k8"]);
        add!("moe_topk_renorm_k8", kernels::MOE_TOPK_RENORM_K8_SRC, ["moe_topk_renorm_k8"]);
        add!("moe_topk_renorm_k8_batched", kernels::MOE_TOPK_RENORM_K8_BATCHED_SRC, ["moe_topk_renorm_k8_batched"]);
        add!("scaled_add_inplace", kernels::SCALED_ADD_INPLACE_SRC, ["scaled_add_inplace_cpu_scalar_f32", "scaled_add_inplace_gpu_scalar_f32"]);
        add!("sigmoid_scaled_residual_add_batched", kernels::SIGMOID_SCALED_RESIDUAL_ADD_BATCHED_SRC, ["sigmoid_scaled_residual_add_batched_f32"]);
        add!("softmax", kernels::SOFTMAX_SRC, ["softmax_f32"]);
    }
    if matches!(arch, "gfx1201" | "gfx1100") {
        add!("gemv_mq4g256v2_residual_sigmoid_scaled_k512", kernels::GEMV_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_K512_SRC, [
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
        ]);
        add!("repeat_interleave_qk_batched", kernels::REPEAT_INTERLEAVE_QK_BATCHED_SRC, ["repeat_interleave_qk_f32_batched"]);
    }
    if matches!(arch, "gfx1201" | "gfx1151") {
        add!("gemv_mq4g256v2_moe_gate_up_k8_indexed", kernels::GEMV_MQ4G256V2_MOE_GATE_UP_K8_INDEXED_SRC, ["gemv_mq4g256v2_moe_gate_up_k8_indexed"]);
        add!("gemv_mq4g256v2_moe_ninepath_d4", kernels::GEMV_MQ4G256V2_MOE_NINEPATH_D4_SRC, ["gemv_mq4g256v2_moe_ninepath_d4"]);
        add!("kv_cache_write_q8_0_pair", kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC, ["kv_cache_write_q8_0_pair"]);
    }
    if matches!(arch, "gfx1100" | "gfx1151") {
        add!("fused_rmsnorm_mq_rotate_vecsum", kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC, ["fused_rmsnorm_mq_rotate_vecsum"]);
        add!("gemm_gate_up_hfq6g256_wmma", kernels::GEMM_GATE_UP_HFQ6G256_WMMA_SRC, ["gemm_gate_up_hfq6g256_wmma"]);
        add!("gemm_hfq6g256_residual_wmma_k2", kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_K2_SRC, ["gemm_hfq6g256_residual_wmma_k2"]);
        add!("gemm_q8_0_wmma", kernels::GEMM_Q8_0_WMMA_SRC, ["gemm_q8_0_wmma"]);
        add!("gemm_qkv_hfq6g256_wmma", kernels::GEMM_QKV_HFQ6G256_WMMA_SRC, ["gemm_qkv_hfq6g256_wmma"]);
        add!("gemm_qkvza_hfq6g256_wmma", kernels::GEMM_QKVZA_HFQ6G256_WMMA_SRC, ["gemm_qkvza_hfq6g256_wmma"]);
        add!("moe_router_softmax_topk_k8_wave64_exact", kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC, ["moe_router_softmax_topk_k8_wave64_exact"]);
    }
    if arch == "gfx1201" {
        add!("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1201_SRC, [
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201",
        ]);
        add!("attention_q8_0_flash_prefill_wmma_gfx12_hd256", format!("#define SPLIT_Q 0\n#define FIXED_HEAD_DIM 256\n#define PREFETCH_V 1\n{}", kernels::kv_slot_desc_source(kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC, false)), [
            "attention_q8_0_flash_prefill_wmma",
        ]);
        add!("gated_norm_mq_rotate_gfx1201", kernels::GATED_NORM_MQ_ROTATE_GFX1201_SRC, ["gated_norm_mq_rotate_gfx1201"]);
        add!("gated_norm_mq_rotate_i4_gfx12_v2", kernels::GATED_NORM_MQ_ROTATE_I4_GFX12_V2_SRC, ["gated_norm_mq_rotate_i4_gfx12_v2"]);
        add!("gemm_gate_up_hfq6g256_wmma_gfx12", kernels::GEMM_GATE_UP_HFQ6G256_WMMA_GFX12_SRC, ["gemm_gate_up_hfq6g256_wmma_gfx12"]);
        add!("gemm_hfq6g256_residual_wmma_gfx12", kernels::GEMM_HFQ6G256_RESIDUAL_WMMA_GFX12_SRC, ["gemm_hfq6g256_residual_wmma_gfx12"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gfx12_g12r", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_G12R_SRC, [
            "gemm_mq4g256v2_gate_up_silu_mmq_iu4_g12r", "gemm_mq4g256v2_residual_mmq_iu4_full_add_g12r",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_g12r", "gemm_mq4g256v2_residual_mmq_iu4_g12r", "quantize_int4_mmq_ds128",
        ]);
        add!("gemm_q8_0_wmma_gfx12", kernels::GEMM_Q8_0_WMMA_GFX12_SRC, ["gemm_q8_0_wmma_gfx12"]);
        add!("gemm_qkv_hfq6g256_wmma_gfx12", kernels::GEMM_QKV_HFQ6G256_WMMA_GFX12_SRC, ["gemm_qkv_hfq6g256_wmma_gfx12"]);
        add!("gemm_qkvza_hfq6g256_wmma_gfx12", kernels::GEMM_QKVZA_HFQ6G256_WMMA_GFX12_SRC, ["gemm_qkvza_hfq6g256_wmma_gfx12"]);
        add!("moe_router_softmax_topk_k8_wave64", kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC, ["moe_router_softmax_topk_k8_wave64"]);
        add!("qwen35_fa_prep_gfx1201", kernels::QWEN35_FA_PREP_GFX1201_SRC, ["qwen35_fa_prep_gfx1201"]);
    }
    if arch == "gfx1100" {
        add!("argmax_token_chain", kernels::ARGMAX_TOKEN_CHAIN_SRC, ["argmax_token_chain_f32"]);
        add!("attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_AWQ_GFX1100_SRC, [
            "attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100",
        ]);
        add!("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC, [
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
        ]);
        add!("conv1d_silu_split_qknorm_b256", kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC, ["conv1d_silu_split_qknorm_b256"]);
        add!("deinterleave_q_rmsnorm_f32_batched", kernels::DEINTERLEAVE_Q_RMSNORM_BATCHED_SRC, ["deinterleave_q_rmsnorm_f32_batched"]);
        add!("fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100", kernels::FUSED_QKV_MQ4G256V2_K2048_X_BUFFER_GFX1100_SRC, [
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        ]);
        add!("fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100", kernels::FUSED_QKVZA_MQ4G256V2_K2048_HOIST_X32_GFX1100_SRC, [
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        ]);
        add!("gated_norm_mq_rotate_gfx1100", kernels::GATED_NORM_MQ_ROTATE_GFX1100_SRC, ["gated_norm_mq_rotate_gfx1100"]);
        add!("gated_norm_mq_rotate_i4_gfx1100_v2", kernels::GATED_NORM_MQ_ROTATE_I4_GFX1100_V2_SRC, ["gated_norm_mq_rotate_i4_gfx1100_v2"]);
        add!("gemm_mq4g256v2_moe_grouped_wmma_k2", kernels::GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC, ["gemm_mq4g256v2_moe_grouped_wmma_k2"]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gridspec", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SRC, [
            "gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec", "gemm_mq4g256v2_residual_mmq_iu4_full_add",
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_gfx1100",
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set", "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
            "quantize_int4_mmq_ds128",
        ]);
        add!("gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100", kernels::GEMV_MQ4G256V2_MOE_GATE_UP_K8_INDEXED_K2048_NOLDS_GFX1100_SRC, [
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100",
        ]);
        add!("gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100", kernels::GEMV_MQ4G256V2_MOE_NINEPATH_RPB8_GFX1100_SRC, ["gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100"]);
        add!("gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch", kernels::GEMV_MQ4G256V2_RESIDUAL_R1_K4096_GFX1100_NOSCRATCH_SRC, [
            "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
        ]);
        add!("greedy_accept", kernels::GREEDY_ACCEPT_SRC, ["greedy_accept_from_argmax_i32"]);
        add!("mq_rotate_x_i4", kernels::MQ_ROTATE_X_I4_SRC, ["mq_rotate_x_i4"]);
        add!("qwen35_fa_prep_gfx1100", kernels::QWEN35_FA_PREP_GFX1100_SRC, ["qwen35_fa_prep_gfx1100"]);
    }
    if arch == "gfx1151" {
        add!("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC, [
            "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
        ]);
        add!("attention_verify_wmma_gfx1151", kernels::ATTENTION_VERIFY_WMMA_GFX1151_SRC, [
            "attention_verify_wmma_pv1s_d4_gfx1151", "attention_verify_wmma_pv2_d2_gfx1151", "attention_verify_wmma_qk_gfx1151",
        ]);
        add!("gated_norm_mq_rotate_gfx1151", kernels::GATED_NORM_MQ_ROTATE_GFX1151_SRC, ["gated_norm_mq_rotate_gfx1151"]);
        add!("gated_norm_mq_rotate_i4_gfx1151_v2", kernels::GATED_NORM_MQ_ROTATE_I4_GFX1151_V2_SRC, ["gated_norm_mq_rotate_i4_gfx1151_v2"]);
        add!("gemm_mq4g256v2_moe_grouped_wmma_k2", kernels::QWEN4_GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC, [
            "gemm_mq4g256v2_moe_grouped_wmma_k2", "gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out",
            "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out",
        ]);
        add!("gemm_mq4g256v2_residual_mmq_iu4_gridspec", kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GRIDSPEC_SRC, [
            "gemm_mq4g256v2_residual_mmq_iu4", "gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec", "gemm_mq4g256v2_residual_mmq_iu4_full_add",
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3",
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151", "gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec",
            "quantize_int4_mmq_ds128",
        ]);
        add!("qwen35_fa_prep_gfx1151", kernels::QWEN35_FA_PREP_GFX1151_SRC, ["qwen35_fa_prep_gfx1151"]);
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
pub fn default_route_entries(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
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
        add!("gated_delta_net_q8_compact2_b2", kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC, ["gated_delta_net_q8_compact2_b2"]); // norm.rs:2994
        add!("gemv_hfq4g256_moe_down_k8_indexed_batched_expanded", kernels::GEMV_HFQ4G256_MOE_DOWN_K8_INDEXED_BATCHED_EXPANDED_SRC, ["gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"]); // gemv.rs:14599
        add!("moe_down_combine_k8_batched", kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC, ["moe_down_combine_k8_batched"]); // moe.rs:41
    }
    if matches!(arch, "gfx1151" | "gfx1201") {
        add!("gemv_hfq4g256_moe_gate_up_indexed", kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_SRC, ["gemv_hfq4g256_moe_gate_up_k8_indexed"]); // gemv.rs:13888
        // gemv.rs:12202,12266,12379: three launchers share the module.
        add!("gemv_hfq4g256_residual_scaled", kernels::GEMV_HFQ4G256_RESIDUAL_SCALED_SRC, [
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu", "gemv_hfq4g256_residual_scaled_gpu", "gemv_hfq4g256_residual_scaled_cpu"]);
    }
    if rdna3 {
        add!("conv1d_silu_split_qknorm_b256", kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC, ["conv1d_silu_split_qknorm_b256"]); // norm.rs:4459
        add!("fused_rmsnorm_mq_rotate_vecsum", kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC, ["fused_rmsnorm_mq_rotate_vecsum"]); // gemv.rs:2477
        add!("kv_cache_write_q8_0_pair", kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC, ["kv_cache_write_q8_0_pair"]); // attention.rs:2502
        add!("moe_router_softmax_topk_k8_wave64_exact", kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_EXACT_SRC, ["moe_router_softmax_topk_k8_wave64_exact"]); // gemv.rs:13029
    }
    match arch {
        "gfx1100" => {
            add!("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC, ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"]); // attention.rs:10626
            add!("gated_norm_mq_rotate_gfx1100", kernels::GATED_NORM_MQ_ROTATE_GFX1100_SRC, ["gated_norm_mq_rotate_gfx1100"]); // gemv.rs:4796
            add!("gemv_hfq4g256_moe_gate_up_indexed_cpol_slc", kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_CPOL_SLC_GFX1100_SRC, ["gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"]); // gemv.rs:13770
            add!("gemv_hfq4g256_residual_sigmoid_buffer_gfx1100", kernels::GEMV_HFQ4G256_RESIDUAL_SIGMOID_BUFFER_GFX1100_SRC, ["gemv_hfq4g256_residual_sigmoid_scaled_gpu"]); // gemv.rs:12369
            add!("gemv_hfq4g256_residual_stage_x32_gfx1100", kernels::GEMV_HFQ4G256_RESIDUAL_STAGE_X32_GFX1100_SRC, ["gemv_hfq4g256_residual"]); // gemv.rs:10262
            add!("moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100", kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC, ["moe_down_combine_rmsnorm_mq_rotate_vecsum"]); // moe.rs:119
            add!("qwen35_fa_prep_gfx1100", kernels::QWEN35_FA_PREP_GFX1100_SRC, ["qwen35_fa_prep_gfx1100"]); // norm.rs:1152
        }
        "gfx1151" => {
            add!("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151", kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC, ["attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"]); // attention.rs:10620
            add!("fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151", kernels::FUSED_QKVZA_HFQ4G256_K2048_ALL_BUFFER_GFX1151_SRC, ["fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"]); // gemm.rs:3811
            add!("fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151", kernels::FUSED_QKVZA_HFQ4G256_K2048_HYBRID_BUFFER_GFX1151_SRC, ["fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"]); // gemm.rs:3800
            add!("gated_norm_mq_rotate_gfx1151", kernels::GATED_NORM_MQ_ROTATE_GFX1151_SRC, ["gated_norm_mq_rotate_gfx1151"]); // gemv.rs:4783
            add!("gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151", kernels::GEMV_HFQ4G256_LM_HEAD_R1_HYBRID_BUFFER_GFX1151_SRC, ["gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"]); // gemv.rs:9892
            add!("gemv_hfq4g256_residual_rt_low_gfx1151", kernels::GEMV_HFQ4G256_RESIDUAL_RT_LOW_GFX1151_SRC, ["gemv_hfq4g256_residual_rt_low_gfx1151"]); // gemv.rs:10226
            add!("moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151", kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC, ["moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151"]); // moe.rs:115
            add!("qwen35_fa_prep_gfx1151", kernels::QWEN35_FA_PREP_GFX1151_SRC, ["qwen35_fa_prep_gfx1151"]); // norm.rs:1146
            // DeepSeek4 MQ2R gfx1151 v2 route (registry/deepseek4-mq2r-gfx1151-v2.json).
            add!("compressor_add_ape_buf", kernels::COMPRESSOR_ADD_APE_BATCHED_SRC, ["compressor_add_ape_f32_buf"]); // attention.rs:14675
            add!("compressor_overlap_concat", kernels::COMPRESSOR_OVERLAP_CONCAT_SRC, ["compressor_overlap_concat_f32"]); // attention.rs:14720
            add!("compressor_softmax_pool_f32_buf", kernels::COMPRESSOR_SOFTMAX_POOL_BUF_SRC, ["compressor_softmax_pool_f32_buf"]); // attention.rs:14852
            add!("deepseek4_attn_swa_buf", kernels::V4F_ATTN_SWA_BUF_SRC, ["deepseek4_attn_swa_buf"]); // attention.rs:18544
            add!("deepseek4_attn_swa_topk_scoregrid_f32_buf", kernels::V4F_ATTN_SWA_TOPK_BUF_SRC, ["deepseek4_attn_swa_topk_scoregrid_f32_buf"]); // attention.rs:19068
            add!("deepseek4_fused_silu_mul_clamp_mq_rotate", kernels::V4F_FUSED_SILU_MUL_CLAMP_MQ_ROTATE_SRC, ["deepseek4_fused_silu_mul_clamp_mq_rotate"]); // norm.rs:6165
            add!("deepseek4_moe_topk_bias_aware", kernels::V4F_MOE_TOPK_BIAS_AWARE_SRC, ["deepseek4_moe_topk_bias_aware_f32"]); // moe.rs:878
            add!("deepseek4_silu_mul_clamp", kernels::V4F_SILU_MUL_CLAMP_SRC, ["deepseek4_silu_mul_clamp_f32"]); // norm.rs:6224
            add!("deepseek4_topk_kv_gather_f32_buf", kernels::V4F_TOPK_KV_GATHER_BUF_SRC, ["deepseek4_topk_kv_gather_f32_buf"]); // moe.rs:1159
            add!("deepseek4_topk_kv_gather_identity_f32_buf", kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC, ["deepseek4_topk_kv_gather_identity_f32_buf"]); // moe.rs:1410
            add!("fused_rmsnorm_mq_rotate_plain_nox", kernels::FUSED_RMSNORM_MQ_ROTATE_PLAIN_SRC, ["fused_rmsnorm_mq_rotate_plain_nox"]); // norm.rs:5946
            add!("gemv_mfp4g32_e8_soa_grouped_gfx1151", kernels::GEMV_MFP4G32_E8_SOA_GROUPED_GFX1151_SRC, ["gemv_mfp4g32_e8_soa_grouped_gfx1151"]); // gemv.rs:7564,8025
            add!("gemv_mfp4g32_e8_soa_u4", kernels::GEMV_MFP4G32_E8_SOA_U4_SRC, ["gemv_mfp4g32_e8_soa_u4"]); // gemv.rs:8075,7448
            add!("gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed", kernels::GEMV_MQ2G256_LLOYD_MOE_DOWN_INDEXED_SRC, ["gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"]); // gemv.rs:16927
            add!("gemv_mq2g256_lloyd_moe_gate_up_indexed", kernels::GEMV_MQ2G256_LLOYD_MOE_GATE_UP_INDEXED_SRC, ["gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"]); // gemv.rs:17258
            add!("hash_router_normalize_f32_buf", kernels::HASH_ROUTER_NORMALIZE_BUF_SRC, ["hash_router_normalize_f32_buf"]); // moe.rs:770
            add!("hc_compute_control_vec4_finalize", kernels::HC_COMPUTE_CONTROL_SRC, ["hc_compute_control_vec4_finalize"]); // attention.rs:15414
            add!("hc_head_compute_pre", kernels::HC_HEAD_COMPUTE_PRE_SRC, ["hc_head_compute_pre"]); // attention.rs:15697
            add!("hc_input_map_4stream", kernels::HC_INPUT_MAP_SRC, ["hc_input_map_4stream"]); // attention.rs:15751
            add!("hc_mix_4stream", kernels::HC_MIX_4STREAM_SRC, ["hc_mix_4stream"]); // attention.rs:15835
            add!("indexer_relu_score_f32_buf", kernels::INDEXER_RELU_SCORE_BUF_SRC, ["indexer_relu_score_f32_buf"]); // attention.rs:16916
            add!("indexer_top_k_buf_parallel", kernels::INDEXER_TOP_K_BUF_PARALLEL_GFX1151_SRC, ["indexer_top_k_buf_parallel"]); // attention.rs:17320
            add!("rmsnorm_f32_at_slot_buf", kernels::RMSNORM_AT_SLOT_BUF_SRC, ["rmsnorm_f32_at_slot_buf"]); // norm.rs:6089
            add!("rope_tail_interleaved", kernels::ROPE_TAIL_INTERLEAVED_SRC, ["rope_tail_interleaved_f32"]); // attention.rs:17562
            add!("rope_tail_yarn_interleaved_at_slot_buf", kernels::ROPE_TAIL_YARN_INTERLEAVED_AT_SLOT_BUF_SRC, ["rope_tail_yarn_interleaved_at_slot_buf_f32"]); // attention.rs:17838
            add!("rope_tail_yarn_interleaved_wide", kernels::ROPE_TAIL_YARN_INTERLEAVED_SRC, ["rope_tail_yarn_interleaved_wide_f32"]); // attention.rs:17768
            add!("sqrt_softplus_f32", kernels::SQRT_SOFTPLUS_F32_SRC, ["sqrt_softplus_f32"]); // norm.rs:6128
            add!("state_overlap_shift_f32_buf", kernels::STATE_OVERLAP_SHIFT_F32_BUF_SRC, ["state_overlap_shift_f32_buf"]); // attention.rs:17988
            add!("state_ring_write_f32_buf", kernels::STATE_RING_WRITE_F32_BUF_SRC, ["state_ring_write_f32_buf"]); // attention.rs:18031
            add!("swa_ring_write_f32_buf", kernels::SWA_RING_WRITE_BUF_SRC, ["swa_ring_write_f32_buf"]); // attention.rs:18171
        }
        "gfx1201" => {
            add!("attention_flash_fp8_e4m3_tile_gqa_gfx1201", kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC, ["attention_flash_fp8_e4m3_tile_gqa_gfx1201"]); // attention.rs:7783
            add!("attention_flash_reduce_dsplit_gfx1201", kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC, ["attention_flash_reduce_dsplit_gfx1201"]); // attention.rs:7784
            add!("gemv_hfq4g256_multirow_default", kernels::GEMV_HFQ4G256_MULTIROW_SRC, ["gemv_hfq4g256_multirow_r2"]); // gemv.rs:10031
            // Qwen3.6-27B decode fusions on exact gfx1201 (3be1cdbe9): the H2 tape's compact-3 GDN,
            // 48-head AWQ gated-norm/MQ rotation and 24Q/4K FA prep.
            add!("gated_delta_net_q8_compact3_b2", kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC, ["gated_delta_net_q8_compact3_b2"]); // norm.rs:3045
            add!("gated_norm_mq_rotate_awq_k6144_gfx1201", kernels::gated_norm_mq_rotate_awq_k6144_gfx1201_src(), ["gated_norm_mq_rotate_awq_k6144_gfx1201"]); // gemv.rs:4751
            #[cfg(feature = "deltanet")]
            add!("qwen36_27b_fa_prep_gfx1201", kernels::qwen36_27b_fa_prep_gfx1201_src(), ["qwen36_27b_fa_prep_gfx1201"]); // norm.rs:1149
            add!("moe_router_softmax_topk_k8_wave64", kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC, ["moe_router_softmax_topk_k8_wave64"]); // gemv.rs:12992
        }
        _ => {}
    }
    Ok(entries)
}

/// Modules railgun's own lowering launches (design §1.1 `Node::Copy` and
/// `Node::CopyBatch`): the hipfire-owned copy kernels, JIT-compiled as module
/// `railgun::copy::MODULE` from `railgun::copy::SOURCE`, on every arch
/// railgun certifies. No Redline program loads them.
pub fn railgun_entries(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let arch: &'static str = SUPPORTED_ARCHES
        .iter()
        .copied()
        .find(|&candidate| candidate == arch)
        .ok_or_else(|| RegistryError::UnsupportedArch(arch.to_owned()))?;
    let mut entries = Vec::new();
    if matches!(arch, "gfx1100" | "gfx1151" | "gfx1201") {
        entries.push(entry(arch, "railgun_copy", &["railgun_copy", "railgun_copy_batch"], kernels::RAILGUN_COPY_SRC.into(), extra_flags));
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
        if !all.iter().any(|e| e.module == entry.module && e.source == entry.source) {
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

// ── A3: header-selected route closure ──────────────────────────────────
//
// `route_entries` maps an already-resolved load (header tensor metadata,
// arch, KV/state axes, spec route, process-frozen feature flags) to every
// kernel module that route can request from load through prefill, decode,
// sampling, retained replay and (for native MTP) draft/verify/rollback, so a
// loader can compile and load the whole closure before the first dispatch.
// It is CPU-only and pure: no weight payload, compiler, or HIP call.
//
// Only routes whose closure was walked from source and cross-checked against
// a module-load audit of every request family are admitted; everything else
// (other models/arches, explicit drafter routes, non-default feature flags or
// policy axes) is `UnsupportedRoute`, and the caller keeps lazy loading.
// Closures assume the default developer environment (no `HIPFIRE_*`
// developer override of a launcher selector); such overrides are outside
// this plan and are caught by the loader's post-ready barrier, not here.

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

#[derive(Debug)]
pub struct RouteKernelPlan {
    pub entries: Vec<PlannedKernel>,
}

/// One HIP module of a closed route: the module name the launcher passes to
/// `ensure_kernel` and exactly the symbols the route requests from it. The
/// source is the registry's launcher-identical expression for that module.
type RouteHip = (&'static str, &'static [&'static str]);

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
    ("attention_flash_asym_reduce_batched", &["attention_flash_asym_reduce_batched"]),
    ("attention_flash_fp8_e4m3_tile_batched", &["attention_flash_fp8_e4m3_tile_batched"]),
    ("attention_flash_fp8_e4m3_tile_gqa_gfx1201", &["attention_flash_fp8_e4m3_tile_gqa_gfx1201"]),
    ("attention_flash_reduce_dsplit_gfx1201", &["attention_flash_reduce_dsplit_gfx1201"]),
    ("attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201", &["attention_fp8_e4m3_fa2_gqa_qresident_v2_gfx1201"]),
    ("attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201", &["attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201"]),
    ("attention_fp8_e4m3_kv_batched", &["attention_fp8_e4m3_kv_batched"]),
    ("attention_verify_gqa_gfx1201", &["attention_verify_gqa_fp8_gfx1201", "attention_verify_reduce_gfx1201"]),
    ("conv1d_silu_split_qknorm_b256", &["conv1d_silu_split_qknorm_b256"]),
    ("convert_f32_to_f16", &["convert_f32_to_f16"]),
    ("dflash_state_bulk_copy_gfx1100", &["dflash_state_bulk_copy_gfx1100"]),
    ("embedding_q8", &["embedding_q8"]),
    ("embedding_q8_batched", &["embedding_q8_batched"]),
    ("fused_gate_up_hfq4g256_mq4v2", &["fused_gate_up_mq4g256v2"]),
    ("fused_qkv_hfq4g256_mq4v2", &["fused_qkv_mq4g256v2"]),
    ("fused_qkvza_hfq4g256_mq4v2", &["fused_qkvza_mq4g256v2"]),
    ("fused_rmsnorm_mq_rotate_awq", &["fused_rmsnorm_mq_rotate_awq"]),
    ("fused_rmsnorm_mq_rotate_awq_g12dec", &["fused_rmsnorm_mq_rotate_awq_g12dec"]),
    ("fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv", &["fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv"]),
    ("fused_sigmoid_alpha_gate", &["fused_sigmoid_alpha_gate_f32"]),
    ("fused_silu_mul_mq_rotate_awq", &["fused_silu_mul_mq_rotate_awq"]),
    ("fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast", &["fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast"]),
    ("gated_delta_net_q8_compact3_b2", &["gated_delta_net_q8_compact3_b2"]),
    ("gated_delta_net_q8_fast", &["gated_delta_net_q8_fast"]),
    ("gated_norm", &["gated_norm_f32"]),
    ("gated_norm_mq_rotate_awq_i4_gfx12_v2_slab", &["gated_norm_mq_rotate_awq_i4_gfx12_v2_slab"]),
    ("gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab", &["gated_norm_mq_rotate_awq_i4_gfx12_v2_xbf16_slab"]),
    ("gated_norm_mq_rotate_awq_k6144_gfx1201", &["gated_norm_mq_rotate_awq_k6144_gfx1201"]),
    ("gdn_chunk_kkt_solve", &["gdn_chunk_kkt_solve"]),
    ("gdn_chunk_kkt_solve_batched", &["gdn_chunk_kkt_solve_batched"]),
    ("gdn_chunk_prep", &["gdn_chunk_prep"]),
    ("gdn_chunk_prep_fixup", &["gdn_chunk_prep_fixup"]),
    ("gdn_chunk_scan_bf16", &["gdn_chunk_scan_bf16"]),
    ("gdn_chunk_scan_bf16_mseg", &["gdn_chunk_scan_bf16_mseg"]),
    ("gdn_pre_batched_gfx1201", &["gdn_pre_batched_gfx1201"]),
    ("gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2", &["gemm_gate_up_mq4g256v2_wmma_gfx12"]),
    ("gemm_gate_up_mq4g256v2_wmma_gfx12_vt", &["gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt2w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt3w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_gate_up_mq4g256v2_wmma_gfx12_vt8w8_k32"]),
    ("gemm_hfq4g256_residual_wmma_gfx12_mq4v2", &["gemm_mq4g256v2_residual_wmma_gfx12"]),
    ("gemm_mq4g256v2_residual_wmma_gfx12_vt", &["gemm_mq4g256v2_residual_wmma_gfx12_vt2w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt2w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt3w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt3w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w4_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt4w8_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt8w4_k32", "gemm_mq4g256v2_residual_wmma_gfx12_vt8w8_k32"]),
    ("gemm_qkv_hfq4g256_wmma_gfx12_mq4v2", &["gemm_qkv_mq4g256v2_wmma_gfx12"]),
    ("gemm_qkv_mq4g256v2_wmma_gfx12_vt", &["gemm_qkv_mq4g256v2_wmma_gfx12_vt2w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt2w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt3w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_qkv_mq4g256v2_wmma_gfx12_vt8w8_k32"]),
    ("gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2", &["gemm_qkvza_mq4g256v2_wmma_gfx12"]),
    ("gemm_qkvza_mq4g256v2_wmma_gfx12_vt", &["gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt2w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt3w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w4_k32", "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w8_k32"]),
    ("gemv_hfq4g256_multirow_default_mq4v2", &["gemv_mq4g256v2_multirow_r2"]),
    ("gemv_hfq4g256_residual_mq4v2", &["gemv_mq4g256v2_residual"]),
    ("gemv_mq4g256v2_mq4v2", &["gemv_mq4g256v2"]),
    ("kv_cache_write_fp8_e4m3", &["kv_cache_write_fp8_e4m3"]),
    ("kv_cache_write_fp8_e4m3_batched", &["kv_cache_write_fp8_e4m3_batched"]),
    ("mq_rotate_x", &["mq_rotate_x"]),
    ("qwen35_fa_prep_batched_gfx1201", &["qwen35_fa_prep_batched_gfx1201"]),
    ("qwen35_fa_prep_fp8q_nogate_batched_gfx1201", &["qwen35_fa_prep_fp8q_nogate_batched_gfx1201"]),
    ("qwen36_27b_fa_prep_gfx1201", &["qwen36_27b_fa_prep_gfx1201"]),
    ("rmsnorm_f32_rowsplit", &["rmsnorm_f32_rowsplit"]),
    ("rotate_x_mq_awq", &["rotate_x_mq_awq"]),
    ("sample_top_p_parallel", &["sample_apply_repeat_penalty", "sample_topk_partial", "sample_topk_finalize"]),
    ("sample_top_p_parallel_fast21", &["sample_topk_partial_fast21", "sample_topk_finalize_fast21"]),
    ("sample_top_p_parallel_fast65", &["sample_topk_partial_fast65", "sample_topk_finalize_fast65"]),
    ("sample_top_p_parallel_w64", &["sample_apply_repeat_penalty_w64", "sample_topk_partial_w64", "sample_topk_finalize_w64"]),
    ("sigmoid_mul", &["sigmoid_mul_f32"]),
    ("sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab", &["sigmoid_mul_rotate_x_mq_awq_i4_gfx12_slab"]),
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
    ("attention_flash_q8_0_reduce", &["attention_flash_q8_0_reduce"]),
    ("attention_flash_q8_0_tile", &["attention_flash_q8_0_tile"]),
    ("conv1d_silu_split", &["conv1d_silu_split_f32"]),
    ("deinterleave", &["deinterleave_f32"]),
    ("deinterleave_batched", &["deinterleave_f32_batched"]),
    ("dflash_gdn_replay_pre_ml", &["dflash_gdn_replay_pre_ml"]),
    ("fused_qk_l2_norm_scale_interleave_f32_batched", &["fused_qk_l2_norm_scale_interleave_f32_batched"]),
    ("gated_delta_net_q8_fast_ml", &["gated_delta_net_q8_fast_ml"]),
    ("gemm_hfq4g256", &["gemm_hfq4g256"]),
    ("gemv_hfq4g256", &["gemv_hfq4g256"]),
    ("gemv_hfq4g256_multirow_default", &["gemv_hfq4g256_multirow_r2"]),
    ("greedy_accept", &["greedy_accept_from_argmax_i32"]),
    ("kv_cache_write_q8_0", &["kv_cache_write_q8_0"]),
    ("kv_cache_write_q8_0_batched", &["kv_cache_write_q8_0_batched"]),
    ("rmsnorm", &["rmsnorm_f32"]),
    ("rope_partial_halfsplit_batched", &["rope_partial_halfsplit_batched_f32"]),
    ("rope_partial_halfsplit_f32_headgrid", &["rope_partial_halfsplit_f32_headgrid"]),
    ("select_regrid", &["argmax_f32_batched_regrid"]),
    ("silu_mul", &["silu_mul_f32"]),
    ("softmax_temp_topp_batched", &["softmax_temp_topp_batched_f32"]),
];

/// CbVmm's rows-batched twins of the singleton decode kernels
/// (`attention::rows_batched`; module name = entry name).
const QWEN36_27B_GFX1201_ROWS_TWINS_HIP: &[RouteHip] = &[
    ("gated_delta_net_q8_compact3_b2_rows", &["gated_delta_net_q8_compact3_b2_rows"]),
    ("conv1d_silu_split_qknorm_b256_rows", &["conv1d_silu_split_qknorm_b256_rows"]),
    ("gated_norm_mq_rotate_awq_k6144_gfx1201_rows", &["gated_norm_mq_rotate_awq_k6144_gfx1201_rows"]),
    ("qwen36_27b_fa_prep_gfx1201_rows", &["qwen36_27b_fa_prep_gfx1201_rows"]),
];

/// VMM continuous-batching exact route additions to
/// [`QWEN36_27B_GFX1201_AR_HIP`] (`forward_slots/vmm/exact.rs`): batched
/// VMM KV write, row-batched flash decode, and the rows twins.
const QWEN36_27B_GFX1201_VMM_EXACT_HIP: &[RouteHip] = &[
    ("kv_cache_write_fp8_e4m3_batched_vmm", &["kv_cache_write_fp8_e4m3_batched_vmm"]),
    ("attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm", &["attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm"]),
    ("attention_flash_reduce_dsplit_gfx1201_vmm", &["attention_flash_reduce_dsplit_gfx1201_vmm"]),
    ("gated_delta_net_q8_compact3_b2_rows", &["gated_delta_net_q8_compact3_b2_rows"]),
    ("conv1d_silu_split_qknorm_b256_rows", &["conv1d_silu_split_qknorm_b256_rows"]),
    ("gated_norm_mq_rotate_awq_k6144_gfx1201_rows", &["gated_norm_mq_rotate_awq_k6144_gfx1201_rows"]),
    ("qwen36_27b_fa_prep_gfx1201_rows", &["qwen36_27b_fa_prep_gfx1201_rows"]),
];

/// The hipcc x-batch GEMV the PM multi-column launchers fall back to when
/// `kernel.pm_decode` is off.
const QWEN36_27B_GFX1201_XBATCH_HIP: &[RouteHip] = &[("gemv_mq4g256v2_xbatch", &["gemv_mq4g256v2_xbatch"])];

/// Cross-request batched MTP verify additions to the native-MTP route
/// (`verify_twins`; module name = entry name).
const QWEN36_27B_GFX1201_SPEC_CB_HIP: &[RouteHip] =
    &[("attention_fp8_e4m3_kv_batched_segs", &["attention_fp8_e4m3_kv_batched_segs"])];

/// The gfx1201 slab IU4 MMQ bundle every 27B prefill projection selects
/// (`gemm.rs` `g12_iu4_b1s_image`, default A4 slab route).
const QWEN36_27B_GFX1201_B1S_SYMBOLS: &[&str] = &[
    "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s",
    "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s",
];

/// Header-selected kernel closure for one resolved load. Refuses (never
/// returns a partial plan) when the model, arch, policy axes, spec route or
/// feature flags are outside a closed route.
pub fn route_entries(input: &RouteKernelInput<'_>) -> Result<RouteKernelPlan, RegistryError> {
    let refuse = |why: String| Err(RegistryError::UnsupportedRoute(why));
    match input.spec {
        KernelSpecRoute::Ar | KernelSpecRoute::NativeMtp => {}
        other => {
            return refuse(format!(
                "{other:?} needs a separately admitted drafter header; no closed route"
            ))
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
            // VMM exact route projections/lm_head: the accepted PM
            // multi-column objects under pm_decode, else the hipcc x-batch
            // GEMV their launchers fall back to (residual/multirow fall back
            // to singleton modules already in the AR route).
            if input.flags.pm_decode {
                for (module, image, symbols) in crate::pm_xbatch::route_objects() {
                    embedded.push(PlannedKernel::Embedded { module, image, radiowave_json: None, symbols });
                }
            } else {
                hip.push(QWEN36_27B_GFX1201_XBATCH_HIP);
            }
            (hip, embedded)
        }
        None => {
            return refuse(format!(
                "model arch {} on {} has no closed route",
                input.model_arch, input.arch
            ))
        }
    };
    let corpus = route_corpus(input.arch, input.hipcc_extra_flags)?;
    let mut entries = embedded;
    for &(module, symbols) in hip.into_iter().flatten() {
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
        let found = matches.next().ok_or_else(|| RegistryError::UnsupportedModule {
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
    Ok(plan)
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
                    a.source == b.source && a.flags == b.flags && a.scheduler_profile == b.scheduler_profile
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

/// Builtin-default feature flags for `arch`, independent of the process
/// environment, as the reference a closed route was walked against.
fn default_feature_flags(arch: &str) -> Result<crate::feature_flags::FeatureFlags, RegistryError> {
    let resolved = hipfire_config::resolve(std::iter::empty())
        .map_err(|e| RegistryError::UnsupportedRoute(format!("builtin config: {e}")))?;
    let config = hipfire_config::ProcessConfig::from_resolved(&resolved)
        .map_err(|e| RegistryError::UnsupportedRoute(format!("builtin config: {e}")))?;
    Ok(crate::feature_flags::FeatureFlags::from_process_config(arch, &config))
}

/// Registry sources a route may name: the installer inventory, the admitted
/// default-route corpus, and route-only launcher sources neither carries.
fn route_corpus(arch: &str, extra_flags: &str) -> Result<Vec<KernelEntry>, RegistryError> {
    let mut all = entries(arch, extra_flags)?;
    all.extend(default_route_entries(arch, extra_flags)?);
    let arch = all.first().map(|e| e.arch).unwrap_or("");
    if arch == "gfx1201" {
        // norm.rs `softmax_temp_topp_batched_into_f32` (MTP temperature verify).
        all.push(entry(arch, "softmax_temp_topp_batched", &["softmax_temp_topp_batched_f32"],
            kernels::SOFTMAX_TEMP_BATCHED_SRC.into(), extra_flags));
        // VMM continuous batching exact route (forward_slots/vmm/exact.rs):
        // the batched VMM KV write and row-batched flash decode, and the
        // rows-batched singleton twins — the launchers' own module sources.
        use crate::attention::{kv_slot_desc_vmm_source, vmm_rows_source, KV_CACHE_WRITE_FP8_E4M3_VMM_SRC};
        let recipe_err = |e: hip_bridge::HipError| RegistryError::ConflictingPlan(format!("vmm exact route source: {e}"));
        all.push(entry(arch, "kv_cache_write_fp8_e4m3_batched_vmm", &["kv_cache_write_fp8_e4m3_batched_vmm"],
            kv_slot_desc_vmm_source(KV_CACHE_WRITE_FP8_E4M3_VMM_SRC, "kv_cache_write_fp8_e4m3_batched_vmm",
                "kv_cache_write_fp8_e4m3_batched_vmm").into(), extra_flags));
        all.push(entry(arch, "attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm", &["attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm"],
            vmm_rows_source(kernels::ATTENTION_FLASH_FP8_E4M3_TILE_GQA_GFX1201_SRC,
                "attention_flash_fp8_e4m3_tile_gqa_gfx1201", "VMM_ROWS_FP8_GQA_TILE").map_err(recipe_err)?.into(),
            extra_flags));
        all.push(entry(arch, "attention_flash_reduce_dsplit_gfx1201_vmm", &["attention_flash_reduce_dsplit_gfx1201_vmm"],
            vmm_rows_source(kernels::ATTENTION_FLASH_REDUCE_DSPLIT_GFX1201_SRC,
                "attention_flash_reduce_dsplit_gfx1201", "VMM_ROWS_DSPLIT_REDUCE").map_err(recipe_err)?.into(),
            extra_flags));
        for &(module, symbols) in QWEN36_27B_GFX1201_ROWS_TWINS_HIP {
            all.push(entry(arch, module, symbols,
                crate::attention::rows_batched::rows_twin_module_source(module).map_err(recipe_err)?.into(),
                extra_flags));
        }
        for &(module, symbols) in QWEN36_27B_GFX1201_SPEC_CB_HIP {
            all.push(entry(arch, module, symbols,
                crate::verify_twins::seg_twin_module_source(module).map_err(recipe_err)?.into(),
                extra_flags));
        }
        all.push(entry(arch, "gemv_mq4g256v2_xbatch", &["gemv_mq4g256v2_xbatch"],
            kernels::GEMV_MQ4G256V2_XBATCH_SRC.into(), extra_flags));
    }
    Ok(all)
}

/// Qwen3.6-27B dense XTS fingerprint on exact gfx1201. `Ok(Some(mtp))` when
/// the closed route applies (`mtp` = native-MTP route), `Ok(None)` when the
/// header is not this model, `Err` when it is but a policy axis is outside
/// the closure.
fn qwen36_27b_route(input: &RouteKernelInput<'_>) -> Result<Option<bool>, RegistryError> {
    use crate::DType;
    let refuse = |why: &str| Err(RegistryError::UnsupportedRoute(format!("qwen3.6-27b: {why}")));
    if input.arch != "gfx1201" || input.model_arch != 5 {
        return Ok(None);
    }
    if (input.n_heads, input.n_kv_heads, input.head_dim) != (24, 4, 256) {
        return Ok(None);
    }
    let trunk: Vec<_> = input.tensors.iter().filter(|t| t.name.starts_with("model.language_model.")
        || t.name == "lm_head.weight").collect();
    let layer_of = |name: &str| -> Option<usize> {
        name.strip_prefix("model.language_model.layers.")?.split('.').next()?.parse().ok()
    };
    let layers = trunk.iter().filter_map(|t| layer_of(t.name)).max().map_or(0, |l| l + 1);
    let shape_of = |name: &str| trunk.iter().find(|t| t.name == name).map(|t| (t.dtype, t.shape));
    let projections_mq4v2 = trunk.iter().filter(|t| t.name.ends_with("_proj.weight")
        || t.name.ends_with("in_proj_qkv.weight") || t.name.ends_with("in_proj_z.weight")
        || t.name.ends_with("in_proj_a.weight") || t.name.ends_with("in_proj_b.weight"));
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
        input.tensors.iter().any(|t| t.name == name && t.dtype == dtype && t.shape == shape)
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
        KernelSpecRoute::NativeMtp if !mtp_head => refuse("native MTP needs the MQ4G256 sidecar head tensors"),
        KernelSpecRoute::NativeMtp => Ok(Some(true)),
        _ => Ok(Some(false)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::{HashMap, HashSet};
    use std::path::Path;

    #[test]
    fn mq4v2_k5120_inventory_preserves_generic_and_limits_arch() {
        let module = "fused_gate_up_mq4g256v2_k5120_gfx1100";
        let registry = entries("gfx1100", "").unwrap();
        let candidate = registry.iter().find(|entry| entry.module == module).unwrap();
        assert_eq!(candidate.symbols, [module]);
        let body = candidate.source().strip_prefix(
            "#define HIPFIRE_FUSED_GATE_UP_KERNEL fused_gate_up_mq4g256v2_k5120_gfx1100\n"
        ).unwrap();
        assert_eq!(
            body.replace("const int groups_per_row = 20;", "const int groups_per_row = K / 256;"),
            kernels::FUSED_GATE_UP_MQ4G256V2_SRC
        );
        for arch in ["gfx1151", "gfx1201", "gfx906", "gfx942"] {
            assert!(!entries(arch, "").unwrap().iter().any(|entry| entry.module == module));
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
                assert_eq!(entry.symbols, [
                    module, "gemm_mq4g256v2_residual_mmq_full_add",
                    "gemm_mq4g256v2_residual_mmq_full_set", "gemm_mq4g256v2_residual_mmq_x128",
                    "gemm_mq4g256v2_residual_mmq_full_add_x128",
                    "gemm_mq4g256v2_residual_mmq_full_set_x128",
                    "quantize_q8_1_mmq_ds4", "quantize_q8_1_mmq_ds4_x128",
                ]);
                let recipe = KernelCompiler::recipe_for_source(arch, module, entry.source(), extra_flags);
                assert_eq!(entry.flags, recipe.flags);
                assert_eq!(entry.scheduler_profile, recipe.scheduler_profile);
                assert_eq!(entry.scheduler_profile.as_deref(), Some("default"));
            }
        }
        for arch in ["gfx1201", "gfx906", "gfx942"] {
            assert!(lookup(arch, module, "").is_err(), "{arch}: admitted RDNA3-only MMQ");
        }
    }

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
        // specialization, the `HIPFIRE_G12_DEC_NORM` decode twins (fused
        // RMSNorm+FWHT group grid, rmsnorm row split, RoPE head grid), the
        // five multi-slot `*_paged` modules of the q8/asym3 routes, the
        // 36 H2 decode, prefill, MTP and DFlash modules added for
        // compiler-free packs, the 46 Qwen3.8-Flash-Next modules (including
        // the gfx1201 `fn_gdn_dense` adapter)
        // (tests/fixtures/kernel-trace-qwen4-flash-next.tsv, including the
        // 8K-128K prefill, MTP and serve rows), the 32 Qwen3.5-MoE modules
        // (tests/fixtures/kernel-trace-qwen35.tsv) and the Qwen4 penalty
        // prepass `logit_penalty_table`, plus eight dense 27B DFlash cold
        // helpers and the four gfx12 MQ4V2 verify-tile (`*_vt`) dense GEMM
        // modules. Those 136 keys are additional to P0's 92.
        assert_eq!(registry.len(), count + 136, "unexpected gfx1201 inventory size");
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
        // Deliberate changes that landed after the P0 capture (kernel cache
        // ABI 4). The traces keep the captured bytes and argv; the portable
        // digests above are re-pinned to the current source.
        // The five KV-write modules gained paged-only code behind
        // `#ifdef HIPFIRE_KV_SLOT_PAGED` (fold/scs); their preprocessed source
        // and gfx1100/gfx1151/gfx1201 `.text` are unchanged. The four
        // grid.y row-fold modules (qk-L2 norm, sigmoid-alpha gate, SiLU-mul
        // rotate, AWQ rotate) moved their row axis onto grid.x for the
        // gfx1201 65536-workgroup limit. The two RMSNorm/FWHT modules' source
        // gained the fold's batched Phase-1a behind `HIPFIRE_RMSNORM_FOLD` +
        // `HIPFIRE_RMSNORM_P1A_BATCHED` (gfx1151 `_fold_b8`); their
        // preprocessed source is unchanged on gfx1100/gfx1151/gfx1201.
        const REPINNED_SINCE_P0: [&str; 14] = [
            "conv1d_silu_split_qknorm_b256",
            "fused_qk_l2_norm_scale",
            "fused_rmsnorm_mq_rotate",
            "fused_rmsnorm_mq_rotate_awq",
            "fused_sigmoid_alpha_gate",
            "fused_silu_mul_mq_rotate",
            "fused_silu_mul_mq_rotate_awq",
            "gated_delta_net_q8_fast",
            "kv_cache_write_asym_k_givens3_batched",
            "kv_cache_write_fp8_e4m3_batched",
            "kv_cache_write_q8_0_batched",
            "kv_cache_write_q8_0_independent",
            "kv_cache_write_q8_0_independent_masked",
            "rotate_x_mq_awq",
        ];
        const FLAGS_ADDED_SINCE_P0: [&str; 2] = ["-fuse-cuid=none", "-mcode-object-version=6"];
        let mut repins_seen = HashSet::new();
        for (trace, expected_count) in [("cold1", 35), ("smokecold1", 35), ("preinstall", 58)] {
            let records = std::fs::read_to_string(root.join(format!("{trace}.hipcc.jsonl"))).unwrap();
            let mut observed = 0;
            for line in records.lines() {
                let record: serde_json::Value = serde_json::from_str(line).unwrap();
                let Some(module) = record["module"].as_str() else { continue };
                let source_path = record["source"].as_str().unwrap();
                let expected = std::fs::read(source_path).unwrap();
                let entry = by_name.get(module).unwrap_or_else(|| panic!("missing {trace} module {module}"));
                if let Some(&repinned) = REPINNED_SINCE_P0.iter().find(|name| **name == module) {
                    assert_ne!(entry.source().as_bytes(), expected,
                        "{trace}: {module} matches the trace again; drop it from REPINNED_SINCE_P0");
                    repins_seen.insert(repinned);
                } else {
                    assert_eq!(entry.source().as_bytes(), expected, "{trace}: {module} source differs");
                }
                let argv = record["argv"].as_array().unwrap();
                let expected_flags = argv.iter().map(|arg| arg.as_str().unwrap())
                    .take_while(|arg| *arg != "-o")
                    .filter(|arg| !arg.starts_with("--rocm-path=")
                        && !arg.starts_with("--hip-path=") && !arg.starts_with("-I"))
                    .collect::<Vec<_>>();
                let flags = entry.flags.iter().map(String::as_str)
                    .filter(|flag| !FLAGS_ADDED_SINCE_P0.contains(flag))
                    .collect::<Vec<_>>();
                assert_eq!(flags, expected_flags, "{trace}: {module} core hipcc flags differ");
                observed += 1;
            }
            assert_eq!(observed, expected_count, "{trace}: incomplete compiler trace");
        }
        assert_eq!(repins_seen.len(), REPINNED_SINCE_P0.len(),
            "REPINNED_SINCE_P0 names a module no trace compiled: {repins_seen:?}");
    }

    #[test]
    fn corpus_entries_name_each_module_once_per_arch() {
        for arch in SUPPORTED_ARCHES {
            let entries = corpus_entries(arch, "").unwrap();
            let mut seen = std::collections::HashSet::new();
            for entry in &entries {
                assert!(seen.insert(entry.module), "{arch}: module {} inventoried twice", entry.module);
            }
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

    /// The opt-in Hyper inline-Q8 chunk module is inventoried on exact gfx1151
    /// only, carries the source `gated_delta_step_gate_wmma` compiles, and
    /// exports the symbol it launches.
    #[test]
    fn hyper_q8_chunk_module_is_exact_gfx1151_only() {
        let entry = lookup("gfx1151", "gated_delta_chunk_q8_wmma", "").unwrap();
        assert_eq!(entry.symbols, ["gated_delta_chunk_gate_q8_wmma"]);
        assert_eq!(entry.source(), crate::tensor_ops::GATED_DELTA_CHUNK_Q8_WMMA_SRC);
        assert!(entry.source().contains("void gated_delta_chunk_gate_q8_wmma("));
        for arch in SUPPORTED_ARCHES.iter().filter(|arch| **arch != "gfx1151") {
            assert!(
                lookup(arch, "gated_delta_chunk_q8_wmma", "").is_err(),
                "{arch} must not inventory the gfx1151 module"
            );
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
        ];
        for module in cross_admitted {
            let entry = lookup("gfx1201", module, "").unwrap();
            let original = lookup("gfx1100", module, "").unwrap();
            assert_eq!(entry.source(), original.source(), "{module}");
            assert_eq!(entry.symbols, original.symbols, "{module}");
        }
        let exact = [
            ("gfx1201", "dflash_gdn_replay_pre_ml_from",
                crate::dflash_gdn_replay::DFLASH_GDN_REPLAY_PRE_ML_FROM_SRC,
                &["dflash_gdn_replay_pre_ml_from"][..]),
            ("gfx1201", "gated_delta_net_q8_fast_ml_from",
                crate::dflash_gdn_replay::GATED_DELTA_NET_Q8_FAST_ML_FROM_SRC,
                &["gated_delta_net_q8_fast_ml_from"][..]),
            ("gfx1201", "attention_verify_gqa_gfx1201",
                kernels::ATTENTION_VERIFY_GQA_GFX1201_SRC,
                &["attention_verify_gqa_q8_gfx1201", "attention_verify_gqa_fp8_gfx1201", "attention_verify_reduce_gfx1201"][..]),
            ("gfx1100", "attention_verify_gqa_gfx1100",
                kernels::ATTENTION_VERIFY_GQA_GFX1100_SRC,
                &["attention_verify_gqa_q8_gfx1100", "attention_verify_gqa_q8_rows_gfx1100", "attention_verify_reduce_gfx1100"][..]),
        ];
        for (arch, module, source, symbols) in exact {
            let entry = lookup(arch, module, "").unwrap();
            assert_eq!(entry.source(), source, "{module}");
            assert_eq!(entry.symbols, symbols, "{module}");
            let other = if arch == "gfx1201" { "gfx1100" } else { "gfx1201" };
            assert!(lookup(other, module, "").is_err());
        }
        let modules = cross_admitted.into_iter()
            .chain(exact.into_iter().map(|(_, module, _, _)| module));
        for module in modules {
            for arch in ["gfx1151", "gfx906", "gfx942"] {
                assert!(lookup(arch, module, "").is_err(), "{arch}: leaked {module}");
            }
            let arch = if module == "attention_verify_gqa_gfx1100" { "gfx1100" } else { "gfx1201" };
            for extra_flags in ["", "-DIU4_A4_CANDIDATES=2"] {
                let entry = lookup(arch, module, extra_flags).unwrap();
                let recipe = KernelCompiler::recipe_for_source(arch, module, entry.source(), extra_flags);
                assert_eq!(entry.flags, recipe.flags, "{module}");
                assert_eq!(entry.scheduler_profile, recipe.scheduler_profile, "{module}");
                assert_eq!(entry.scheduler_profile.as_deref(), Some("default"), "{module}");
            }
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
            let inventory = by_arch.entry(arch).or_insert_with(|| entries(arch, "").unwrap());
            let entry = inventory
                .iter()
                .find(|entry| entry.module == module)
                .unwrap_or_else(|| panic!("{arch}: traced module {module} is not packaged"));
            for symbol in symbols.split(',') {
                assert!(entry.symbols.contains(&symbol), "{arch} {module}: packaged symbols lack {symbol}");
            }
            rows += 1;
        }
        assert!(rows > 0, "empty trace fixture");
    }

    #[test]
    fn qwen4_flash_next_trace_is_packaged() {
        assert_trace_packaged(include_str!("../tests/fixtures/kernel-trace-qwen4-flash-next.tsv"));
    }

    #[test]
    fn qwen35_trace_is_packaged() {
        assert_trace_packaged(include_str!("../tests/fixtures/kernel-trace-qwen35.tsv"));
    }
}
