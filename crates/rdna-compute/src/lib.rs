// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
pub mod attention;
pub mod attention_wmma_verify;
pub mod byte_arena;
pub mod cdna;
pub mod code_object;
pub mod compile_jobs;
mod compiler;
pub mod device_counter;
pub mod device_wait;
pub mod dflash_draft_fusion;
pub mod dflash_gdn_pre;
pub mod dflash_gdn_replay;
pub mod dflash_hidden_scatter;
pub mod dflash_sampled_accept;
pub mod dflash_selector;
pub mod dflash_state_copy;
mod dispatch;
pub mod draft_lane_conv;
pub mod draft_lane_rows;
pub mod e8p_moe_decode;
pub mod embedding;
pub mod emu;
pub mod emu_validate;
pub mod expert_capabilities;
pub mod fa_prefill_pm;
pub mod fa_qprep_pm;
pub mod feature_flags;
pub mod ffn_activation;
#[cfg(feature = "flash-attn-ck")]
pub mod flash_attn_ck;
pub mod flux_fused;
#[cfg(feature = "deltanet")]
pub mod fn_gdn_dense;
pub mod fp8_e4m3;
pub mod gap_timing;
#[cfg(feature = "deltanet")]
pub mod gated_delta_vector;
#[cfg(feature = "deltanet")]
pub mod gated_delta_vector_pm;
pub mod gdn_chunk_prep_pm;
#[cfg(test)]
mod gdn_q8ef_reference;
pub mod gemm;
mod gemm_fp8_verify;
mod gemm_mq6_fp8;
mod gemm_mq6_i8;
mod gemm_q8_a8;
mod gemm_q8_fp8;
mod gemm_vt;
mod gemma4_ext;
mod gemma4_ops;
pub mod gemv;
mod gemv_q8x;
pub mod ggml_convert;
pub mod ggml_mmvq;
pub mod ggml_ops;
#[cfg(test)]
mod gpu_test;
pub mod graph;
pub mod grouped_ops;
pub mod hc_row_fold;
pub mod head_lanes;
pub mod iu4_c8_pm;
mod kernel_abi;
pub mod kernel_pack;
pub mod kernel_registry;
pub mod kernels;
pub mod kv_slots;
pub mod lane_fa;
pub mod lane_gdn;
pub mod lane_proj;
pub mod lane_rows_pm;
pub mod latent_attention;
pub mod latent_attention_absorb;
pub mod latent_attention_decode;
pub mod latent_attention_flash;
pub mod latent_attention_tiled;
pub mod load_phases;
mod loaded_code_check;
pub mod moe;
pub mod moe_decode_lanes;
mod moe_few;
pub mod moe_grouped_k2_pm;
pub mod moe_grouped_qt44_iu4;
pub mod moe_rotate_i4_pm;
pub mod moe_slot_ops;
pub mod norm_fold;
pub mod residual_lanes;
pub mod slab_admission;
pub mod stream_mix;
pub use slab_admission::{
    qwen4_slab_codec_admitted, qwen4_slab_default_cold_codec, qwen4_slab_layer_cold_codec,
    SlabLayerCodecError,
};
pub mod mq6_railgun;
pub mod mq_f16_producers;
pub mod mq_f16_residual_producers;
pub mod norm;
pub mod npu_trunk;
mod packed_mq4;
pub mod page_pool;
pub mod plan_sink;
pub mod pm4_ledger;
pub mod pm_decode_twins;
pub mod pm_mtp_gemm;
pub mod pm_tune;
pub mod pm_xbatch;
pub mod pool;
pub mod prefill_invariant;
pub mod prefill_route_capabilities;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;
pub mod qsa_pooled_mirror;
pub mod qwen35_fa_batch;
pub mod railgun_check;
pub mod railgun_window;
pub mod rdna;
pub mod replay;
pub mod rope;
pub mod router_fault;
pub mod sampled_rows_compact;
pub mod sampling;
pub mod scratch;
pub mod select_regrid;
pub mod shadow_borrow;
pub mod shared_i8;
pub mod slab_moe;
pub mod spec_sampling_cpu_ar;
pub mod tape_copy;
pub mod verify_twins;
pub use gemm::F2GdnTargets;
pub use gemm_fp8_verify::{
    fp8_verify_symbol, fp8_verify_tile, Fp8VerifyOp, Fp8VerifyReceipt, Fp8VerifyTarget,
    Fp8VerifyTile, Fp8VerifyX,
};
pub use gemm_mq6_fp8::Mq6Fp8Arm;
pub use gemm_mq6_i8::{QSA_NORM_EPI_HEAD_DIM, QSA_NORM_EPI_ROTARY_DIM};
pub use gemm_q8_a8::{Q8A8Activations, Q8A8Tile};
pub use gemm_q8_fp8::Q8Fp8Activations;
pub use scratch::{
    Int4MmqDownPrepared, Int4MmqDownReservation, Int4MmqPrepared, Int4MmqReservation,
    Int8MmqPrepared, Int8MmqReservation, Mq4v2Fp8Prepared,
};
pub mod slot_pool;
pub mod tensor_ops;
pub mod text_encoder;
pub mod tp_peer_push;
pub mod trunk_mask;
pub mod uma_memory;
pub mod vae;
pub use trunk_mask::{
    qwen4_trunk_iu4_route, Qwen4TrunkIu4Route, Qwen4TrunkMask, TrunkFamily, TrunkIu4Default,
    TrunkIu4MaskSource, TRUNK_ALL_BITS, TRUNK_GDN_BITS, TRUNK_IU4_DEFAULT_KEY, TRUNK_IU4_ENV,
    TRUNK_QSA_BITS,
};

pub use compiler::KernelCompiler;
#[cfg(feature = "lab")]
pub use dispatch::LabKernelFaultReceipt;
pub use dispatch::{
    gen_fwht_signs, ActivationCapture, BlockHessianAcc, DType, Gpu, GpuTensor, HessianCapture,
    HostAllocLease, KvFitFloor, QsaExtentDepth, VmmPrefixSet, VmmPrefixSpec, VmmResourceMove,
    GL_CB2, GL_CB3, GL_GROUP_SCALE_BYTES, GL_MQ2_GROUP_IDX_BYTES, GL_MQ3_GROUP_IDX_BYTES,
    LLOYD_MQ3_GROUP_BYTES, LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER, MQ2G256V2_GROUP_BYTES,
    MQ3G256V2_GROUP_BYTES, MQ4C_GROUP_BYTES, MQ4G128V2_GROUP_BYTES, MQ4V2_GROUP_BYTES,
    MQ5G256V2_GROUP_BYTES, MQ6G256V2_GROUP_BYTES,
};
#[cfg(feature = "lab")]
pub use dispatch::{
    lab_extract_device_bytes, LabG1DownBorrow, LabG1GateUpBorrow, LabG1WindowSeal, LabG1Witness,
};
#[cfg(target_os = "linux")]
pub use dispatch::{
    ExportedWeightAllocation, ImportedWeightAllocationDesc, EXPORT_WEIGHT_TAIL_PAD_BYTES,
};
#[cfg(feature = "lab")]
pub use dispatch::{LabLaunchWitness, LabProjectionLaunch};
pub use feature_flags::FeatureFlags;
pub use hip_bridge::{HipError, HipResult};
use hipfire_config::developer_bool;
pub use plan_sink::{PlanSink, PlannedKernel, PlannedOp};

/// Calibration-only override that forces native-BF16 teachers to stay BF16.
///
/// When `HIPFIRE_CALIB_BF16=1`, a qt=16 / BF16-source weight is uploaded as
/// raw 2-byte BF16 and tagged `DType::BF16` instead of being widened to F32,
/// and activations staged via `Gpu::convert_f32_to_bf16` reach
/// `KernelKey::GemmBf16Mfma` on gfx942 (CDNA3 MFMA `v_mfma_f32_16x16x16bf16_1k`)
/// instead of the scalar F32 GEMM (`GemmF32RegisterTiled` / `GemmF32Batched`
/// in `kernels/src/gemm_f32.hip` — one warp per output element, zero data
/// reuse, no LDS, no MFMA). OFF by default so shipped inference is
/// unaffected.
pub fn calib_force_bf16() -> bool {
    // Snapshot-backed (mirrors `FeatureFlags::calib_force_bf16`): no
    // process-global cache here, the versioned `ProcessConfig` snapshot is
    // itself resolved once. Arch callers keep calling this free function.
    developer_bool("HIPFIRE_CALIB_BF16", false)
}

/// Startup alias/baseline eligibility probe for the `qwen4_qsa_code_check`
/// registry: runs the same CPU sample + report as the once-only HIP 700
/// capture over the currently registered modules, labelled
/// `hipfire: code-check self-test: ...`. Never consumes or changes the
/// once-only HIP 700 slot (even if it is already consumed); uses `try_lock`
/// (busy/poisoned registry yields a "skipped" line) and performs no HIP/HSA
/// call. Returns the one-line report; the caller prints it.
#[doc(hidden)]
pub fn qsa_code_check_self_test() -> String {
    loaded_code_check::self_test_report()
}
pub use kernels::GEMV_SRC;
/// whose gfx11 and gfx12 kernels are separate translation units. Exported so
/// no-GPU tests can assert the launcher resolves to a real entry point on both
/// arch legs (the `kernels` module itself stays private).
pub use kernels::{mq2g256_lloyd_moe_grouped_wmma_source, mq3g256_lloyd_moe_grouped_wmma_source};

#[cfg(test)]
mod tests {
    use super::calib_force_bf16;

    #[test]
    fn calib_force_bf16_false_when_unset() {
        // Spec requires false when HIPFIRE_CALIB_BF16 is not "1".
        // In CI the var is unset; if a prior test or env sets it to "1"
        // we skip rather than flake — the contract is OFF by default.
        if hipfire_config::developer_var("HIPFIRE_CALIB_BF16").as_deref() == Ok("1") {
            return;
        }
        // Ensure the var is not "1" for this assertion path. The snapshot
        // is resolved once per process; in CI the var is unset at process
        // start, which is the acceptance condition.
        assert!(
            !calib_force_bf16(),
            "calib_force_bf16() must be false when HIPFIRE_CALIB_BF16 != \"1\" (OFF by default)"
        );
    }
}
