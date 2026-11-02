// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx942 operations.
//!
//! Gfx942Device is the architecture proof: its field and constructor are
//! private, it does not dereference to Gpu, and it exposes only operations
//! implemented by gfx942-owned sources. This keeps model code from reaching a
//! CDNA kernel through a string/feature flag alone.

use std::ffi::c_void;

use hip_bridge::{HipResult, KernargBlob};

use crate::{DType, Gpu, GpuTensor};

const GROUPED_OLORA_E8_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mfp4g32_e8_soa_grouped.gfx942.hip");
const GROUPED_OLORA_E8_KERNEL: &str = "gemv_mfp4g32_e8_soa_grouped_gfx942";
const GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mfp4g32_e8_soa_grouped_wave64x4.gfx942.hip");
const GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL: &str =
    "gemv_mfp4g32_e8_soa_grouped_wave64x4_gfx942_candidate";
const INDEXER_TOP_K_BUF_PARALLEL_SRC: &str =
    include_str!("../../../../kernels/src/indexer_top_k_buf_parallel.gfx942.hip");
const INDEXER_TOP_K_BUF_PARALLEL_KERNEL: &str = "indexer_top_k_buf_parallel_gfx942";
// Re-export of the frozen kernels.rs contract const (single include_str! site;
// all other gfx942 sources in this file predate the kernels.rs re-export
// convention and keep their local includes).
const INDEXER_TOP_K_BUF_BOUNDED_SRC: &str = crate::kernels::INDEXER_TOP_K_BUF_BOUNDED_GFX942_SRC;
const INDEXER_TOP_K_BUF_BOUNDED_KERNEL: &str = "indexer_top_k_buf_parallel_gfx942_bounded";
const MQ2_LLOYD_GATE_UP_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.gfx942.hip");
const MQ2_LLOYD_GATE_UP_WAVE64_KERNEL: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_gfx942";
const MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_wave64x8.gfx942.hip");
const MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_gate_up_wave64x8_gfx942_candidate";
const MQ_ROTATE_X_WAVE64_SRC: &str = include_str!("../../../../kernels/src/mq_rotate_x.gfx942.hip");
const MQ_ROTATE_X_WAVE64_KERNEL: &str = "mq_rotate_x_gfx942";
const MQ2_LLOYD_DOWN_EXPANDED_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4.gfx942.hip");
const MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_expanded_k4_gfx942";
const MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_SRC: &str =
    include_str!("../../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.gfx942.hip");
const MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL: &str =
    "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_gfx942";
const FP8B128_MOE_GROUPED_SRC: &str =
    include_str!("../../../../kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip");
const FP8B128_MOE_GROUPED_MODULE: &str = "gemm_fp8b128_moe_grouped_bf16mfma";
const FP8B128_MOE_GROUPED_KERNEL: &str = "gemm_fp8b128_moe_grouped_bf16mfma";
const FP8B128_MOE_GROUPED_XBF16_KERNEL: &str = "gemm_fp8b128_moe_grouped_bf16mfma_xbf16";
/// Test-only retained P1 baseline: its own JIT module and symbols, never
/// selected by a production route. Compiled from an immutable standalone
/// snapshot of the P1 (cdc37659a) kernel (SHA-256 pinned by a unit test), not
/// from the production source, so edits to the production kernel cannot move it.
const FP8B128_MOE_GROUPED_REFERENCE_SRC: &str =
    include_str!("../../tests/fixtures/fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip");
const FP8B128_MOE_GROUPED_REFERENCE_MODULE: &str = "gemm_fp8b128_moe_grouped_bf16mfma_reference";
const FP8B128_MOE_GROUPED_REFERENCE_KERNEL: &str = "gemm_fp8b128_moe_grouped_bf16mfma_reference";
const FP8B128_MOE_GROUPED_REFERENCE_XBF16_KERNEL: &str =
    "gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16";
/// Output rows per workgroup of the retained baseline: frozen at P1's 64, so
/// a candidate geometry change never moves the reference launch (pinned
/// against the snapshot's `FP8REF_NSUB` by a unit test below).
const FP8B128_MOE_REFERENCE_ROWS_PER_WG: usize = 64;
/// Output rows per workgroup of the FP8 grouped kernel: `64 * FP8MOE_NSUB`
/// (pinned against the source by a unit test below).
const FP8B128_MOE_ROWS_PER_WG: usize = 64;

/// Build identity of one FP8 grouped GEMM JIT module as this process loads it
/// (benchmark provenance; see `test_fp8_moe_gfx942 --time`).
#[doc(hidden)]
pub struct Fp8MoeBuildIdentity {
    pub module: &'static str,
    pub source_sha256: String,
    /// The compiler's content key: source + flags + arch + toolchain + ABI.
    pub jit_cache_key: String,
    pub toolchain_id: String,
    pub object_path: String,
    pub object_sha256: String,
}
const DEQUANT_FP8B128_TO_BF16_SRC: &str =
    include_str!("../../../../kernels/src/dequant_fp8b128_to_bf16.gfx942.hip");
const DEQUANT_FP8B128_TO_BF16_KERNEL: &str = "dequant_fp8b128_to_bf16_gfx942";
const FP8B128_MOE_UNSCATTER_SILU_SRC: &str =
    include_str!("../../../../kernels/src/fp8b128_moe_unscatter_silu.gfx942.hip");
const FP8B128_MOE_UNSCATTER_SILU_KERNEL: &str = "fp8b128_moe_unscatter_silu_gfx942";
const FP8B128_MOE_SHARED_ACTIVATION_SRC: &str =
    include_str!("../../../../kernels/src/fp8b128_moe_shared_activation.gfx942.hip");
const FP8B128_MOE_SHARED_ACTIVATION_KERNEL: &str = "fp8b128_moe_shared_activation_gfx942";

/// A mutable GPU borrow proven to target exact gfx942.
///
/// The constructor is intentionally available only through Gpu::try_gfx942.
/// There is no Deref or escape hatch to the underlying generic context.
pub struct Gfx942Device<'gpu> {
    gpu: &'gpu mut Gpu,
}

impl Gpu {
    /// Borrow this context as an exact-gfx942 device.
    ///
    /// gfx940/gfx941 are deliberately rejected even though they are CDNA3.
    pub fn try_gfx942(&mut self) -> Option<Gfx942Device<'_>> {
        self.arch_caps
            .is_gfx942()
            .then_some(Gfx942Device { gpu: self })
    }
}

impl Gfx942Device<'_> {
    /// Native wave64 MQ2-Lloyd gate/up baseline for DS4 routed experts.
    pub fn mq2_lloyd_moe_gate_up_wave64(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 gate/up requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            MQ2_LLOYD_GATE_UP_WAVE64_SRC,
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let n_experts_val = (expert_ptrs.buf.size() / 8) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &n_experts_val as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_WAVE64_KERNEL,
            [m as u32, k_top as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(n_experts_val);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Exact-shape eight-wave gate/up candidate with workgroup-shared x tiles.
    pub fn mq2_lloyd_moe_gate_up_wave64x8_candidate(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        x_rot: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert_eq!(m, 4096, "gfx942 wave64x8 MQ2 gate/up requires M=4096");
        assert_eq!(k, 4096, "gfx942 wave64x8 MQ2 gate/up requires K=4096");
        assert_eq!(k_top, 6, "gfx942 wave64x8 MQ2 gate/up requires top-k=6");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_SRC,
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let x_ptr = x_rot.buf.as_ptr();
        let gate_ptr = y_gate.buf.as_ptr();
        let up_ptr = y_up.buf.as_ptr();
        let n_experts_val = (expert_ptrs.buf.size() / 8) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &n_experts_val as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            MQ2_LLOYD_GATE_UP_WAVE64X8_CANDIDATE_KERNEL,
            [512, 6, 1],
            [512, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(gate_ptr);
                blob.push_ptr(up_ptr);
                blob.push_i32(n_experts_val);
                blob
            },
        )
    }

    /// Exact-gfx942 wave64 MQ G256 rotation for a batch of routed experts.
    ///
    /// The source and symbol are private to this sealed device so the DS4
    /// backend cannot accidentally select them through a generic arch string.
    pub fn mq_rotate_x_wave64_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ rotate requires K%256=0");
        self.gpu.ensure_kernel(
            MQ_ROTATE_X_WAVE64_KERNEL,
            MQ_ROTATE_X_WAVE64_SRC,
            MQ_ROTATE_X_WAVE64_KERNEL,
        )?;
        self.gpu.ensure_mq_signs()?;
        let x_ptr = x.buf.as_ptr();
        let x_rot_ptr = x_rot.buf.as_ptr();
        let signs1_ptr = self
            .gpu
            .scratch
            .mq_signs1
            .as_ref()
            .expect("ensure_mq_signs populated signs1")
            .buf
            .as_ptr();
        let signs2_ptr = self
            .gpu
            .scratch
            .mq_signs2
            .as_ref()
            .expect("ensure_mq_signs populated signs2")
            .buf
            .as_ptr();
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &x_rot_ptr as *const _ as *mut c_void,
            &signs1_ptr as *const _ as *mut c_void,
            &signs2_ptr as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.gpu.hip, "fwht", MQ_ROTATE_X_WAVE64_KERNEL, bytes);
        let result = self.gpu.launch_maybe_blob(
            MQ_ROTATE_X_WAVE64_KERNEL,
            [((k / 256) * batch_size) as u32, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(x_rot_ptr);
                blob.push_ptr(signs1_ptr);
                blob.push_ptr(signs2_ptr);
                blob.push_i32(k_i32);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        self.gpu.invalidate_x_caches_for(x_rot_ptr);
        result
    }

    /// Exact-gfx942 deterministic MQ2-Lloyd down projection.
    ///
    /// This preserves the incumbent expanded per-rank output and the caller's
    /// fixed-order combine. It never selects the atomic benchmark-only route.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_expanded_wave64(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        rot_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_SRC,
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let rot_batch_ptr = rot_batch.buf.as_ptr();
        let expert_outputs_ptr = expert_outputs.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let k_top_i32 = k_top as i32;
        let n_experts_val = (expert_ptrs.buf.size() / 8) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &rot_batch_ptr as *const _ as *mut c_void,
            &expert_outputs_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &k_top_i32 as *const _ as *mut c_void,
            &n_experts_val as *const _ as *mut c_void,
        ];
        let bytes = batch_size * k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_EXPANDED_WAVE64_KERNEL,
            [m.div_ceil(2) as u32, k_top as u32, batch_size as u32],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(rot_batch_ptr);
                blob.push_ptr(expert_outputs_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(k_top_i32);
                blob.push_i32(n_experts_val);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Exact-gfx942 atomic residual MQ2-Lloyd down projection used by the
    /// model's four static hash-routed layers. This restores the pre-backend
    /// wave64 route without exposing it through the generic `Gpu` surface.
    #[allow(clippy::too_many_arguments)]
    pub fn mq2_lloyd_moe_down_residual_wave64(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        rot_batch: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        k_top: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(k.is_multiple_of(256), "gfx942 MQ2 down requires K%256=0");
        self.gpu.ensure_kernel(
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_SRC,
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
        )?;
        let expert_ptrs_ptr = expert_ptrs.buf.as_ptr();
        let topk_indices_ptr = topk_indices.buf.as_ptr();
        let topk_weights_ptr = topk_weights.buf.as_ptr();
        let rot_batch_ptr = rot_batch.buf.as_ptr();
        let residual_ptr = residual.buf.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let n_experts_val = (expert_ptrs.buf.size() / 8) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &expert_ptrs_ptr as *const _ as *mut c_void,
            &topk_indices_ptr as *const _ as *mut c_void,
            &topk_weights_ptr as *const _ as *mut c_void,
            &rot_batch_ptr as *const _ as *mut c_void,
            &residual_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &n_experts_val as *const _ as *mut c_void,
        ];
        let bytes = k_top * (m * (k / 256) * 72 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemv",
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            MQ2_LLOYD_DOWN_RESIDUAL_WAVE64_KERNEL,
            [m as u32, k_top as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(expert_ptrs_ptr);
                blob.push_ptr(topk_indices_ptr);
                blob.push_ptr(topk_weights_ptr);
                blob.push_ptr(rot_batch_ptr);
                blob.push_ptr(residual_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(n_experts_val);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Exact-gfx942 deterministic indexer top-K.
    ///
    /// `bounded == false` selects the O(N^2) rank-count reference
    /// (`indexer_top_k_buf_parallel_gfx942`). `bounded == true` selects the
    /// opt-in O(N log^2 K) tile-merge bitonic port
    /// (`indexer_top_k_buf_parallel_gfx942_bounded`): a drop-in with
    /// identical dispatch shape and byte-identical output. The selected
    /// symbol keys both `ensure_kernel` and the launch, so the two modules
    /// never share a cache entry.
    ///
    /// This is intentionally not exposed through the generic `Gpu` surface:
    /// DeepSeek4 must hold a verified model-owned gfx942 backend before it can
    /// select the CDNA code object.
    pub fn indexer_top_k_buf_parallel(
        &mut self,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_k: i32,
        bounded: bool,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        let (src, kernel) = if bounded {
            (
                INDEXER_TOP_K_BUF_BOUNDED_SRC,
                INDEXER_TOP_K_BUF_BOUNDED_KERNEL,
            )
        } else {
            (
                INDEXER_TOP_K_BUF_PARALLEL_SRC,
                INDEXER_TOP_K_BUF_PARALLEL_KERNEL,
            )
        };
        self.gpu.ensure_kernel(kernel, src, kernel)?;
        let scores_ptr = scores.buf.as_ptr();
        let top_indices_ptr = top_indices.buf.as_ptr();
        let n_compressed_ptr = n_compressed_buf.buf.as_ptr();
        let k_ptr = k_buf.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &scores_ptr as *const _ as *mut c_void,
            &top_indices_ptr as *const _ as *mut c_void,
            &n_compressed_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &n_idx_heads as *const _ as *mut c_void,
            &max_k as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            kernel,
            [n_idx_heads as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(scores_ptr);
                blob.push_ptr(top_indices_ptr);
                blob.push_ptr(n_compressed_ptr);
                blob.push_ptr(k_ptr);
                blob.push_i32(n_idx_heads);
                blob.push_i32(max_k);
                blob
            },
        )
    }

    /// One-dispatch block-diagonal qt35 projection for DeepSeek4 O-LoRA:
    /// A[G,M,K] @ x[G,K] -> y[G,M].
    ///
    /// This is the retained baseline while a full-wave64 persistent projection
    /// engine is built. It is owned here so no RDNA/model route can select it.
    pub fn grouped_olora_e8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert!(
            k % 256 == 0,
            "grouped E8 gfx942 kernel requires K%256==0, got K={k}"
        );
        self.gpu.ensure_kernel(
            GROUPED_OLORA_E8_KERNEL,
            GROUPED_OLORA_E8_SRC,
            GROUPED_OLORA_E8_KERNEL,
        )?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let groups_i32 = groups as i32;
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &groups_i32 as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            GROUPED_OLORA_E8_KERNEL,
            [m.div_ceil(2) as u32, groups as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob.push_i32(groups_i32);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Exact-shape native-wave64 candidate for the DS4 O-LoRA projection.
    ///
    /// Four wave64s compute four rows in one 256-thread workgroup. This is a
    /// channel-only candidate: no product dispatch calls it.
    pub fn grouped_olora_e8_wave64x4_candidate(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        groups: usize,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gpu.bind_thread()?;
        assert_eq!(groups, 8, "gfx942 wave64x4 candidate requires G=8");
        assert_eq!(m, 1024, "gfx942 wave64x4 candidate requires M=1024");
        assert_eq!(k, 4096, "gfx942 wave64x4 candidate requires K=4096");
        self.gpu.ensure_kernel(
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_SRC,
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
        )?;
        let a_ptr = a.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            GROUPED_OLORA_E8_WAVE64X4_CANDIDATE_KERNEL,
            [256, 8, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(x_ptr);
                blob.push_ptr(y_ptr);
                blob
            },
        )
    }

    /// Reserve the stable-scatter grouping scratch (`Gpu::moe_scatter_stable_top10`)
    /// for the FP8 MoE route: prefills of up to `max_rows` tokens with `k_top`
    /// routed slots each over `n_exp` experts. Call at model load, before any
    /// graph capture, so no forward grows (and invalidates) scratch. Only the
    /// grouping scratch: the FP8 grouped GEMM itself needs none.
    ///
    /// ```text
    /// reserve_fp8b128_moe_group(max_rows: usize, k_top: usize, n_exp: usize) -> HipResult<()>
    /// ```
    pub fn reserve_fp8b128_moe_group(
        &mut self,
        max_rows: usize,
        k_top: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        let slots = max_rows.checked_mul(k_top).ok_or_else(|| {
            hip_bridge::HipError::new(0, "reserve_fp8b128_moe_group: max_rows * k_top overflows")
        })?;
        let bytes = crate::scratch::qwen4_moe_group_layout(slots, n_exp).bytes;
        if crate::scratch::scratch_will_grow(
            self.gpu.scratch.qwen4_moe_group_scratch_bytes,
            self.gpu.scratch.qwen4_moe_group_scratch.is_some(),
            bytes,
        ) {
            self.gpu.invalidate_for_scratch_growth();
        }
        self.gpu.bind_thread()?;
        self.gpu
            .scratch
            .reserve_qwen4_moe_group(&self.gpu.hip, slots, n_exp)
    }

    /// Grouped MoE GEMM over qt=55 `FP8E4M3B128` expert blocks (BF16 MFMA,
    /// F32 accumulate) for the gfx942 FP8 Qwen4 route: stable scatter → this
    /// (gate_up) → unscatter+SiLU → this (down) → slot-order combine. Prefill
    /// and decode use this one kernel (decode = one tile per selected expert
    /// with `nrows = 1`).
    ///
    /// ```text
    /// gemm_fp8b128_moe_grouped_bf16mfma(
    ///     w:             qt=55 payload base; expert e's block at e * expert_stride:
    ///                    codes u8[r*k] at +0, s_inv bf16 [r/128, k/128] at +r*k
    ///     expert_stride: bytes between expert blocks (= fp8_e4m3::b128_block_stride(r, k)
    ///                    for a contiguous tensor; % 16 == 0, >= r*k + 2*scales)
    ///     n_exp:         experts in `w` (span check)
    ///     tile_desc:     [n_tiles, 4] i32 {expert (< 0 = skip tile), row0 (index into
    ///                    sorted_rows), nrows (1..=16), 0}
    ///     n_tiles
    ///     sorted_rows:   [n_sorted] i32 flat slot id (token*top_k + j); -1 = padding
    ///                    (its Y row is written +0)
    ///     x:             [x_rows, k] DType::BF16, or DType::F32 holding BF16 values
    ///                    (RNE'd to BF16 in-kernel; identity on BF16-exact input)
    ///     x_row_div:     X row = sorted_rows[i] / x_row_div (10: gate_up, X = token
    ///                    rows; 1: down, X = slot rows)
    ///     x_rows:        span check
    ///     y:             [n_sorted, r] F32; rows row0..row0+nrows of every live tile are
    ///                    overwritten, nothing else is written
    ///     n_sorted:      span check
    ///     r, k:          one expert's output rows / input width (r % 16 == 0,
    ///                    k % 128 == 0); Qwen4 gate_up r=1280 k=2560, down r=2560 k=640
    /// )
    /// ```
    ///
    /// Grid `[ceil(r / 64), n_tiles]`, 256 threads. The MQ4 padded grouped
    /// layout is the special case `row0 = 16 * t, nrows = 16`.
    ///
    /// Numerics (frozen contract): `W = bf16_rne(f32(e4m3fn(q)) * f32(s_inv))`
    /// bit-identical to `fp8_teacher.dequant_e4m3_block`, BF16 x, F32 MFMA
    /// accumulation in one fixed K order for every launch shape (documented in
    /// the kernel header), no split-K, no atomics: a row's output bytes depend
    /// only on its X row and its expert's block — not on the other rows of the
    /// launch, its position inside a tile, or the tile index. Tile descriptors
    /// and row ids are not range-checked on the device.
    ///
    /// Any kernel change is gated by exact F32-byte A/B against the retained P1
    /// baseline (`test_fp8_moe_gfx942 --byte-identity`, hidden
    /// `gemm_fp8b128_moe_grouped_bf16mfma_reference`, compiled from the
    /// SHA-pinned standalone snapshot under `tests/fixtures/`); the baseline's
    /// bytes are never re-pinned to a candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8b128_moe_grouped_bf16mfma(
        &mut self,
        w: &GpuTensor,
        expert_stride: usize,
        n_exp: usize,
        tile_desc: &GpuTensor,
        n_tiles: usize,
        sorted_rows: &GpuTensor,
        x: &GpuTensor,
        x_row_div: usize,
        x_rows: usize,
        y: &GpuTensor,
        n_sorted: usize,
        r: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemm_fp8b128_moe_grouped_launch(
            false,
            w,
            expert_stride,
            n_exp,
            tile_desc,
            n_tiles,
            sorted_rows,
            x,
            x_row_div,
            x_rows,
            y,
            n_sorted,
            r,
            k,
        )
    }

    /// Hidden, diagnostic-only: the retained P1 baseline of
    /// [`Self::gemm_fp8b128_moe_grouped_bf16mfma`], same arguments and contract,
    /// for the exact-byte A/B gate (`test_fp8_moe_gfx942 --byte-identity`).
    /// It is compiled from an immutable standalone source snapshot
    /// (`tests/fixtures/fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip`, no
    /// shared helper or flag with the production kernel) as its own JIT module
    /// (`gemm_fp8b128_moe_grouped_bf16mfma_reference`), with its own frozen
    /// launch geometry (64 rows / 256 threads per workgroup). Production
    /// source, module name, flags and geometry are untouched, and nothing on a
    /// model route calls this. Reachable only through the exact-gfx942 device
    /// proof.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8b128_moe_grouped_bf16mfma_reference(
        &mut self,
        w: &GpuTensor,
        expert_stride: usize,
        n_exp: usize,
        tile_desc: &GpuTensor,
        n_tiles: usize,
        sorted_rows: &GpuTensor,
        x: &GpuTensor,
        x_row_div: usize,
        x_rows: usize,
        y: &GpuTensor,
        n_sorted: usize,
        r: usize,
        k: usize,
    ) -> HipResult<()> {
        self.gemm_fp8b128_moe_grouped_launch(
            true,
            w,
            expert_stride,
            n_exp,
            tile_desc,
            n_tiles,
            sorted_rows,
            x,
            x_row_div,
            x_rows,
            y,
            n_sorted,
            r,
            k,
        )
    }

    /// Hidden, diagnostic-only: provenance of the loaded FP8 grouped GEMM
    /// module (`reference`: the retained P1 snapshot; otherwise production):
    /// source SHA-256, compiler content key, toolchain id and the SHA-256 of
    /// the code object the compiler resolves for it. Compiles (or hits the
    /// kernel cache) as a side effect.
    #[doc(hidden)]
    pub fn fp8b128_moe_grouped_build_identity(
        &mut self,
        reference: bool,
    ) -> HipResult<Fp8MoeBuildIdentity> {
        use sha2::{Digest, Sha256};
        let (module, src) = if reference {
            (
                FP8B128_MOE_GROUPED_REFERENCE_MODULE,
                FP8B128_MOE_GROUPED_REFERENCE_SRC,
            )
        } else {
            (FP8B128_MOE_GROUPED_MODULE, FP8B128_MOE_GROUPED_SRC)
        };
        self.gpu.bind_thread()?;
        let object = self.gpu.compiler.compile(module, src)?.to_path_buf();
        let bytes = std::fs::read(&object).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("{module}: read {}: {e}", object.display()))
        })?;
        Ok(Fp8MoeBuildIdentity {
            module,
            source_sha256: format!("{:x}", Sha256::digest(src.as_bytes())),
            jit_cache_key: self.gpu.compiler.jit_cache_key(module, src),
            toolchain_id: self.gpu.compiler.toolchain_id().to_owned(),
            object_path: object.display().to_string(),
            object_sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8b128_moe_grouped_launch(
        &mut self,
        reference: bool,
        w: &GpuTensor,
        expert_stride: usize,
        n_exp: usize,
        tile_desc: &GpuTensor,
        n_tiles: usize,
        sorted_rows: &GpuTensor,
        x: &GpuTensor,
        x_row_div: usize,
        x_rows: usize,
        y: &GpuTensor,
        n_sorted: usize,
        r: usize,
        k: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "gemm_fp8b128_moe_grouped_bf16mfma";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &format!("{WHAT}: {msg}")));
        if r == 0 || r % 16 != 0 || k == 0 || k % 128 != 0 || x_row_div == 0 {
            return refuse(format!("needs r % 16 == 0, k % 128 == 0, x_row_div >= 1 (r={r} k={k} x_row_div={x_row_div})"));
        }
        if r.checked_mul(k).is_none_or(|rk| rk > i32::MAX as usize) || n_tiles > u32::MAX as usize {
            return refuse(format!(
                "r*k={r}*{k} exceeds i32 or n_tiles={n_tiles} exceeds the grid"
            ));
        }
        let Some(min_stride) =
            crate::fp8_e4m3::b128_scale_count(r, k).and_then(|s| (r * k).checked_add(2 * s))
        else {
            return refuse(format!("block geometry overflows (r={r} k={k})"));
        };
        if expert_stride % 16 != 0 || expert_stride < min_stride {
            return refuse(format!(
                "expert_stride {expert_stride} must be a multiple of 16 and >= {min_stride}"
            ));
        }
        let (func, x_elem) = match (x.dtype, reference) {
            (DType::F32, false) => (FP8B128_MOE_GROUPED_KERNEL, 4),
            (DType::BF16, false) => (FP8B128_MOE_GROUPED_XBF16_KERNEL, 2),
            (DType::F32, true) => (FP8B128_MOE_GROUPED_REFERENCE_KERNEL, 4),
            (DType::BF16, true) => (FP8B128_MOE_GROUPED_REFERENCE_XBF16_KERNEL, 2),
            (other, _) => return refuse(format!("x must be F32 or BF16, got {other:?}")),
        };
        crate::moe::require_span(
            w,
            &[n_exp, expert_stride],
            "gemm_fp8b128_moe_grouped_bf16mfma w",
        )?;
        crate::moe::require_span(
            tile_desc,
            &[n_tiles, 16],
            "gemm_fp8b128_moe_grouped_bf16mfma tile_desc",
        )?;
        crate::moe::require_span(
            sorted_rows,
            &[n_sorted, 4],
            "gemm_fp8b128_moe_grouped_bf16mfma sorted_rows",
        )?;
        crate::moe::require_span(
            x,
            &[x_rows, k, x_elem],
            "gemm_fp8b128_moe_grouped_bf16mfma x",
        )?;
        crate::moe::require_span(y, &[n_sorted, r, 4], "gemm_fp8b128_moe_grouped_bf16mfma y")?;
        if n_tiles == 0 {
            return Ok(());
        }
        self.gpu.bind_thread()?;
        let rows_per_wg = if reference {
            self.gpu.ensure_kernel(
                FP8B128_MOE_GROUPED_REFERENCE_MODULE,
                FP8B128_MOE_GROUPED_REFERENCE_SRC,
                func,
            )?;
            FP8B128_MOE_REFERENCE_ROWS_PER_WG
        } else {
            self.gpu
                .ensure_kernel(FP8B128_MOE_GROUPED_MODULE, FP8B128_MOE_GROUPED_SRC, func)?;
            FP8B128_MOE_ROWS_PER_WG
        };
        let wp = w.buf.as_ptr();
        let stride = expert_stride as u64;
        let tp = tile_desc.buf.as_ptr();
        let sp = sorted_rows.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let rv = r as i32;
        let kv = k as i32;
        let rd = x_row_div as i32;
        let nev = i32::try_from(n_exp).map_err(|_| {
            hip_bridge::HipError::new(0, &format!("{WHAT}: n_exp {n_exp} exceeds i32"))
        })?;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &stride as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rd as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        // Weight bytes streamed per tile + X + Y (upper bound: 16 rows per tile).
        let bytes = n_tiles.saturating_mul(r * k + 16 * k * x_elem + 16 * r * 4);
        let timer = crate::profile::begin_timer(&self.gpu.hip, "gemm", func, bytes);
        let result = self.gpu.launch_maybe_blob(
            func,
            [r.div_ceil(rows_per_wg) as u32, n_tiles as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(wp);
                blob.push_u64(stride);
                blob.push_ptr(tp);
                blob.push_ptr(sp);
                blob.push_ptr(xp);
                blob.push_ptr(yp);
                blob.push_i32(rv);
                blob.push_i32(kv);
                blob.push_i32(rd);
                blob.push_i32(nev);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Build `tile_desc` for [`Self::gemm_fp8b128_moe_grouped_bf16mfma`] from a
    /// padded-16 grouping (the stable scatter's `tiles` output):
    /// `tile_desc[t] = {tile_ids[t], 16 * t, 16, 0}` for `t < n_tiles`; a `-1`
    /// tile id stays a skipped tile.
    ///
    /// ```text
    /// fp8b128_moe_tile_desc_pad16(
    ///     tile_ids:  [n_tiles] i32 expert per 16-row tile (-1 = none)
    ///     tile_desc: [n_tiles, 4] i32, overwritten
    ///     n_tiles)
    /// ```
    pub fn fp8b128_moe_tile_desc_pad16(
        &mut self,
        tile_ids: &GpuTensor,
        tile_desc: &GpuTensor,
        n_tiles: usize,
    ) -> HipResult<()> {
        const FUNC: &str = "gemm_fp8b128_moe_tile_desc_pad16";
        if n_tiles
            .checked_mul(16)
            .is_none_or(|rows| rows > i32::MAX as usize)
        {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{FUNC}: n_tiles={n_tiles} exceeds i32 rows"),
            ));
        }
        crate::moe::require_span(
            tile_ids,
            &[n_tiles, 4],
            "fp8b128_moe_tile_desc_pad16 tile_ids",
        )?;
        crate::moe::require_span(
            tile_desc,
            &[n_tiles, 16],
            "fp8b128_moe_tile_desc_pad16 tile_desc",
        )?;
        if n_tiles == 0 {
            return Ok(());
        }
        self.gpu.bind_thread()?;
        self.gpu
            .ensure_kernel(FP8B128_MOE_GROUPED_MODULE, FP8B128_MOE_GROUPED_SRC, FUNC)?;
        let ip = tile_ids.buf.as_ptr();
        let dp = tile_desc.buf.as_ptr();
        let nv = n_tiles as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
        ];
        self.gpu.launch_maybe_blob(
            FUNC,
            [n_tiles.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(ip);
                blob.push_ptr(dp);
                blob.push_i32(nv);
                blob
            },
        )
    }

    /// Model-lifetime decode of `experts` consecutive qt=55 `FP8E4M3B128`
    /// expert blocks to BF16 bits.
    ///
    /// ```text
    /// dequant_fp8b128_to_bf16(
    ///     blocks: qt=55 payload, expert e at e * b128_block_stride(r, k)
    ///     out:    [experts, r, k] BF16 bits (DType::BF16 or any 2-byte view)
    ///     experts, r, k   (k % 16 == 0)
    /// )
    /// ```
    ///
    /// `out = bf16_rne(f32(e4m3fn(q)) * f32(s_inv[r/128][k/128]))`, the decode
    /// the grouped kernel feeds its MFMA (same FNUZ-fold body, 0x80 -> 0x8000).
    pub fn dequant_fp8b128_to_bf16(
        &mut self,
        blocks: &GpuTensor,
        out: &GpuTensor,
        experts: usize,
        r: usize,
        k: usize,
    ) -> HipResult<()> {
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        if r == 0
            || k == 0
            || k % 16 != 0
            || r.checked_mul(k).is_none_or(|rk| rk > i32::MAX as usize)
        {
            return refuse(format!(
                "dequant_fp8b128_to_bf16: needs k % 16 == 0 and r*k <= i32::MAX (r={r} k={k})"
            ));
        }
        let Some(stride) = crate::fp8_e4m3::b128_block_stride(r, k) else {
            return refuse(format!(
                "dequant_fp8b128_to_bf16: block stride overflows (r={r} k={k})"
            ));
        };
        crate::moe::require_span(blocks, &[experts, stride], "dequant_fp8b128_to_bf16 blocks")?;
        crate::moe::require_span(out, &[experts, r, k, 2], "dequant_fp8b128_to_bf16 out")?;
        if experts == 0 {
            return Ok(());
        }
        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(
            DEQUANT_FP8B128_TO_BF16_KERNEL,
            DEQUANT_FP8B128_TO_BF16_SRC,
            DEQUANT_FP8B128_TO_BF16_KERNEL,
        )?;
        let bp = blocks.buf.as_ptr();
        let op = out.buf.as_ptr();
        let rv = r as i32;
        let kv = k as i32;
        let sv = stride as u64;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
        ];
        let bytes = experts.saturating_mul(stride + r * k * 2);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            DEQUANT_FP8B128_TO_BF16_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            DEQUANT_FP8B128_TO_BF16_KERNEL,
            [(r * k / 16).div_ceil(256) as u32, experts as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(bp);
                blob.push_ptr(op);
                blob.push_i32(rv);
                blob.push_i32(kv);
                blob.push_u64(sv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Grouped gate/up unscatter fused with the det teacher's BF16 SwiGLU
    /// (`fastfwd.py:476-479`; precise `expf`).
    ///
    /// ```text
    /// fp8b128_moe_unscatter_silu(
    ///     gate_up:    F32 [n_sorted, 2*mi]  grouped logits, gate | up
    ///     sorted:     i32 [n_sorted]        flat slot per grouped row, -1 = skip
    ///     activation: F32 [slots, mi]       row `sorted[i]` written
    ///     mi, n_sorted, slots
    /// )
    /// ```
    ///
    /// Per element `g = bf16(gate)`, `u = bf16(up)`, `s = bf16(g / (1 + expf(-g)))`,
    /// `h = bf16(s * u)`, stored as F32 holding the BF16 value. Rows with
    /// `sorted[i] < 0` or `>= slots` are skipped; each live flat slot is written
    /// by exactly one thread with a plain store (no atomics). Grid
    /// `[n_sorted, ceil(mi/256)]`, block 256. `gate_up`, `sorted` and
    /// `activation` must be distinct buffers.
    pub fn fp8b128_moe_unscatter_silu(
        &mut self,
        gate_up: &GpuTensor,
        sorted: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        n_sorted: usize,
        slots: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "fp8b128_moe_unscatter_silu";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let max_dim = i32::MAX as usize;
        if mi > max_dim || n_sorted > max_dim || slots > max_dim {
            return refuse(format!("{WHAT}: scalar dimension exceeds i32::MAX (mi={mi} n_sorted={n_sorted} slots={slots})"));
        }
        let grid_y = mi.div_ceil(256);
        if grid_y > u16::MAX as usize {
            return refuse(format!(
                "{WHAT}: mi={mi} needs grid.y={grid_y} > {}",
                u16::MAX
            ));
        }
        // `sorted` is i32 data; the sealed route carries the stable scatter's index scratch as
        // `DType::Raw` (1 B/elem in byte_size), so accept a 4-byte dtype or Raw — the byte-span check
        // below is what bounds the read.
        let sorted_ok = sorted.dtype.size() == 4 || sorted.dtype == DType::Raw;
        if gate_up.dtype != DType::F32 || activation.dtype != DType::F32 || !sorted_ok {
            return refuse(format!(
                "{WHAT}: needs F32 gate_up/activation and a 4-byte sorted (got {:?}/{:?}/{:?})",
                gate_up.dtype, activation.dtype, sorted.dtype
            ));
        }
        crate::moe::require_span(
            gate_up,
            &[n_sorted, 2, mi, 4],
            "fp8b128_moe_unscatter_silu gate_up",
        )?;
        crate::moe::require_span(sorted, &[n_sorted, 4], "fp8b128_moe_unscatter_silu sorted")?;
        crate::moe::require_span(
            activation,
            &[slots, mi, 4],
            "fp8b128_moe_unscatter_silu activation",
        )?;
        let (yp, sp, ap) = (
            gate_up.buf.as_ptr(),
            sorted.buf.as_ptr(),
            activation.buf.as_ptr(),
        );
        if yp == ap || sp == ap || yp == sp {
            return refuse(format!(
                "{WHAT}: gate_up, sorted and activation must be distinct buffers"
            ));
        }
        if mi == 0 || n_sorted == 0 || slots == 0 {
            return Ok(());
        }
        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(
            FP8B128_MOE_UNSCATTER_SILU_KERNEL,
            FP8B128_MOE_UNSCATTER_SILU_SRC,
            FP8B128_MOE_UNSCATTER_SILU_KERNEL,
        )?;
        let (miv, nv, sv) = (mi as i32, n_sorted as i32, slots as i32);
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &miv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
        ];
        let bytes = n_sorted.saturating_mul(mi).saturating_mul(12);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            FP8B128_MOE_UNSCATTER_SILU_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            FP8B128_MOE_UNSCATTER_SILU_KERNEL,
            [n_sorted as u32, grid_y as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(yp);
                blob.push_ptr(sp);
                blob.push_ptr(ap);
                blob.push_i32(miv);
                blob.push_i32(nv);
                blob.push_i32(sv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Shared-expert BF16 SwiGLU plus in-place selector sigmoid, matching the
    /// det teacher's BF16 boundaries (`fastfwd.py:476-479`; precise `expf`).
    ///
    /// ```text
    /// fp8b128_moe_shared_activation(
    ///     gate, up:   F32 [rows, mi]
    ///     activation: F32 [rows, mi]   h = bf16(bf16(silu(bf16 g)) * bf16 u)
    ///     selector:   F32 [rows]       in place: bf16(sigmoid(bf16(logit))), once
    ///     rows, mi
    /// )
    /// ```
    ///
    /// One thread per element, plain stores, no atomics (thread `i < rows` also
    /// owns `selector[i]`). Grid `[ceil(rows*mi/256)]`, block 256. All four
    /// buffers must be distinct.
    pub fn fp8b128_moe_shared_activation(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        activation: &GpuTensor,
        selector: &GpuTensor,
        rows: usize,
        mi: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "fp8b128_moe_shared_activation";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let max_dim = i32::MAX as usize;
        if rows > max_dim || mi > max_dim {
            return refuse(format!(
                "{WHAT}: scalar dimension exceeds i32::MAX (rows={rows} mi={mi})"
            ));
        }
        let Some(n) = rows.checked_mul(mi) else {
            return refuse(format!("{WHAT}: rows*mi overflows (rows={rows} mi={mi})"));
        };
        let grid_x = n.div_ceil(256);
        if grid_x > u32::MAX as usize {
            return refuse(format!(
                "{WHAT}: rows*mi={n} needs grid.x={grid_x} > u32::MAX"
            ));
        }
        for (name, t) in [
            ("gate", gate),
            ("up", up),
            ("activation", activation),
            ("selector", selector),
        ] {
            if t.dtype != DType::F32 {
                return refuse(format!("{WHAT}: {name} must be F32 (got {:?})", t.dtype));
            }
        }
        crate::moe::require_span(gate, &[rows, mi, 4], "fp8b128_moe_shared_activation gate")?;
        crate::moe::require_span(up, &[rows, mi, 4], "fp8b128_moe_shared_activation up")?;
        crate::moe::require_span(
            activation,
            &[rows, mi, 4],
            "fp8b128_moe_shared_activation activation",
        )?;
        crate::moe::require_span(
            selector,
            &[rows, 4],
            "fp8b128_moe_shared_activation selector",
        )?;
        let (gp, up_p, ap, sp) = (
            gate.buf.as_ptr(),
            up.buf.as_ptr(),
            activation.buf.as_ptr(),
            selector.buf.as_ptr(),
        );
        if [gp, up_p, ap, sp]
            .iter()
            .enumerate()
            .any(|(i, a)| [gp, up_p, ap, sp][..i].contains(a))
        {
            return refuse(format!(
                "{WHAT}: gate, up, activation and selector must be distinct buffers"
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.gpu.bind_thread()?;
        self.gpu.ensure_kernel(
            FP8B128_MOE_SHARED_ACTIVATION_KERNEL,
            FP8B128_MOE_SHARED_ACTIVATION_SRC,
            FP8B128_MOE_SHARED_ACTIVATION_KERNEL,
        )?;
        let (rv, miv) = (rows as i32, mi as i32);
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &miv as *const _ as *mut c_void,
        ];
        let bytes = n.saturating_mul(12).saturating_add(rows.saturating_mul(8));
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            FP8B128_MOE_SHARED_ACTIVATION_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            FP8B128_MOE_SHARED_ACTIVATION_KERNEL,
            [grid_x as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gp);
                blob.push_ptr(up_p);
                blob.push_ptr(ap);
                blob.push_ptr(sp);
                blob.push_i32(rv);
                blob.push_i32(miv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::arch_caps::ArchCaps;
    use crate::FeatureFlags;
    use std::sync::Arc;

    #[test]
    fn exact_gfx942_cap_is_not_a_cdna3_family_alias() {
        let caps = |arch| ArchCaps::new(arch, Arc::new(FeatureFlags::for_test(arch)));
        assert!(caps("gfx942").is_gfx942());
        for arch in ["gfx940", "gfx941", "gfx1151", "gfx1201"] {
            assert!(!caps(arch).is_gfx942(), "{arch} admitted as gfx942");
        }
    }

    #[test]
    fn fp8b128_moe_rows_per_wg_matches_kernel_source() {
        let nsub = super::FP8B128_MOE_GROUPED_SRC
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define FP8MOE_NSUB "))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .expect("FP8MOE_NSUB define in the FP8 grouped kernel source");
        assert_eq!(super::FP8B128_MOE_ROWS_PER_WG, 64 * nsub);
    }

    /// The retained P1 oracle is an immutable standalone source: any byte change
    /// is an oracle repin (forbidden) and fails here.
    #[test]
    fn fp8b128_moe_reference_snapshot_is_pinned() {
        use sha2::{Digest, Sha256};
        const PIN: &str = "98edc4e17a855270f2b1c74bf1d5f34b3f743a93179bdfe3d7252b274920a0cc";
        let got = format!(
            "{:x}",
            Sha256::digest(super::FP8B128_MOE_GROUPED_REFERENCE_SRC.as_bytes())
        );
        assert_eq!(
            got, PIN,
            "tests/fixtures/fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip changed: the retained \
             P1 oracle must never be edited or re-pinned to a candidate"
        );
    }

    #[test]
    fn fp8b128_moe_reference_is_independent_of_production_source() {
        let src = super::FP8B128_MOE_GROUPED_REFERENCE_SRC;
        // Geometry pin: 64 * FP8REF_NSUB output rows per workgroup, 256 threads.
        let nsub = src
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define FP8REF_NSUB "))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .expect("FP8REF_NSUB define in the reference snapshot");
        assert_eq!(super::FP8B128_MOE_REFERENCE_ROWS_PER_WG, 64 * nsub);
        assert_eq!(src.matches("__launch_bounds__(256)").count(), 2);
        // Same exports as the launcher selects; nothing from the production
        // kernel's helpers, flags or module.
        for k in [
            super::FP8B128_MOE_GROUPED_REFERENCE_KERNEL,
            super::FP8B128_MOE_GROUPED_REFERENCE_XBF16_KERNEL,
        ] {
            assert!(
                src.contains(&format!("void {k}(")),
                "reference snapshot lacks export {k}"
            );
        }
        assert!(!src.contains("fp8moe_") && !src.contains("FP8MOE_"));
        assert!(!src.contains("HIPFIRE_COMPILER_FLAGS"));
        assert!(src.contains("#error"));
    }
}
