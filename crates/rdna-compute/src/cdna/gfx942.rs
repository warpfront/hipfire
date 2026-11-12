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

use hip_bridge::{DeviceBuffer, HipResult, KernargBlob};

use crate::{DType, Gpu, GpuTensor};

// Batched QSA decode core of many lanes (`HIPFIRE_QWEN4_QSA_LANES`): a child module so it
// reaches the private `Gfx942Device` borrow.
#[path = "gfx942_qsa_lanes.rs"]
mod qsa_lanes;
pub use qsa_lanes::{
    Qwen4QsaLanesPlan, QWEN4_QSA_LANES_ATTENTION_F32, QWEN4_QSA_LANES_ATTENTION_FP8_FNUZ,
    QWEN4_QSA_LANES_MAX, QWEN4_QSA_LANES_MODULE, QWEN4_QSA_LANES_POOL, QWEN4_QSA_LANES_PROLOGUE_F32,
    QWEN4_QSA_LANES_PROLOGUE_FP8_FNUZ, QWEN4_QSA_LANES_SCORES, QWEN4_QSA_LANES_SELECT,
    QWEN4_QSA_LANE_ROWS,
};
pub(crate) use qsa_lanes::{Qwen4QsaLaneDesc, Qwen4QsaLanesShape, QsaLanesTableCache};

// The GDN Core of a whole lane group (verify windows and AR lanes, ragged rows) in three
// launches: a child module so it reaches the private `Gfx942Device` borrow and the table cache.
#[path = "gfx942_gdn_lanes_verify.rs"]
mod gdn_lanes_verify;
pub use gdn_lanes_verify::{
    qwen4_gdn_lanes_verify_admitted,
    Qwen4GdnLanesVerifyLane, Qwen4GdnLanesVerifyShape, QWEN4_G942_GDN_LANES_CONV_PARAMS,
    QWEN4_G942_GDN_LANES_GATE, QWEN4_G942_GDN_LANES_STEP, QWEN4_G942_GDN_LANES_VERIFY_MAX,
    QWEN4_G942_GDN_LANES_VERIFY_MODULE, QWEN4_G942_GDN_LANES_VERIFY_ROWS,
};

// x-batched MQ2G256V2 GEMV of the MTP draft head's ranking (one weight pass for up to eight
// rows, bit-equal to the one-row GEMV): a child module so it reaches the private
// `Gfx942Device` borrow.
#[path = "gfx942_mq2_xbatch.rs"]
mod mq2_xbatch;
pub use mq2_xbatch::{
    qwen4_g942_mq2_xbatch_covers, qwen4_g942_mq2_xbatch_entry, QWEN4_G942_MQ2_XBATCH_MAX_ROWS,
    QWEN4_G942_MQ2_XBATCH_MODULE,
};

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
const GEMM_BF16_MFMA_WIDE_SRC: &str =
    include_str!("../../../../kernels/src/gemm_bf16_mfma_wide.gfx942.hip");
const GEMM_BF16_MFMA_WIDE_MODULE: &str = "gemm_bf16_mfma_wide_gfx942";
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
/// C1 stock-FP8 MoE glue folds (`HIPFIRE_QWEN4_G942_FOLD_MOE`): one source, three entries.
const QWEN4_G942_GLUE_MOE_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_glue_moe.hip");
const QWEN4_G942_GLUE_MOE_MODULE: &str = "qwen4_g942_glue_moe";
const QWEN4_G942_MOE_GROUP_SCATTER_DESC: &str = "qwen4_g942_moe_group_scatter_desc";
const QWEN4_G942_MOE_UNSCATTER_ACT: &str = "qwen4_g942_moe_unscatter_act";
const QWEN4_G942_MOE_COMBINE_SHARED_ADD: &str = "qwen4_g942_moe_combine_shared_add";
/// Capacity of the single-workgroup scatter fold (`G942_FOLD_MAXL` / `G942_FOLD_MAXE` in the
/// kernel source): flat slots (four 256-slot chunks) and experts.
pub const G942_FOLD_MOE_MAX_SLOTS: usize = 1024;
pub const G942_FOLD_MOE_MAX_EXPERTS: usize = 1024;

/// One stderr line per process the first time a fold launches: the proof, in a served or benched
/// run's log, that the lever ran (lessons-0.4.5 §4.7).
fn g942_fold_note(once: &std::sync::Once, what: &'static str) {
    once.call_once(|| eprintln!("[g942-fold-moe] {what} active (HIPFIRE_QWEN4_G942_FOLD_MOE)"));
}

/// Bind `func` of the fold module: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_FOLD_MOE_IMAGE=/absolute/device.elf` names one, that explicitly supplied
/// code object (the Route-A twin proof: same module, same entry names, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_fold_ensure(gpu: &mut Gpu, func: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_FOLD_MOE_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_FOLD_MOE_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_G942_GLUE_MOE_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                func,
            )
        }
        _ => gpu.ensure_kernel(QWEN4_G942_GLUE_MOE_MODULE, QWEN4_G942_GLUE_MOE_SRC, func),
    }
}
/// Exact-gfx942 wide-lane (>= [`G942_MOE_WIDE_MIN_SLOTS`] routed slots) stock-FP8 MoE glue: the
/// router, the grouping + tile descriptors, the routed unscatter + shared activation and the
/// combine + shared add of the C1 folds, bitwise, with the latency structure of the 8-lane verify
/// window (`kernels/src/qwen4_g942_moe_wide.gfx942.hip`; its own cache key and twin seat).
const QWEN4_G942_MOE_WIDE_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_moe_wide.gfx942.hip");
pub const QWEN4_G942_MOE_WIDE_MODULE: &str = "qwen4_g942_moe_wide";
/// The four entries of [`QWEN4_G942_MOE_WIDE_MODULE`], in source order.
pub const QWEN4_G942_MOE_WIDE_ENTRIES: [&str; 4] = [
    "qwen4_g942_moe_wide_router",
    "qwen4_g942_moe_wide_group_scatter_desc",
    "qwen4_g942_moe_wide_unscatter_act",
    "qwen4_g942_moe_wide_combine_shared_add",
];
const QWEN4_G942_MOE_WIDE_ROUTER: &str = QWEN4_G942_MOE_WIDE_ENTRIES[0];
const QWEN4_G942_MOE_WIDE_SCATTER_DESC: &str = QWEN4_G942_MOE_WIDE_ENTRIES[1];
const QWEN4_G942_MOE_WIDE_UNSCATTER_ACT: &str = QWEN4_G942_MOE_WIDE_ENTRIES[2];
const QWEN4_G942_MOE_WIDE_COMBINE_SHARED_ADD: &str = QWEN4_G942_MOE_WIDE_ENTRIES[3];
/// Fewest routed slots (eight top-10 rows: the lane width) that take the wide glue; below it the
/// C1 folds' kernels run unchanged.
// G12 input: executor-local route predicate; the row count is the one the sealed MoE call
// already carries, no scheduler or engine interface is reshaped.
pub const G942_MOE_WIDE_MIN_SLOTS: usize = 80;
/// Capacity of the wide scatter workgroup (`MW_MAXL` / `MW_MAXE` in the kernel source): flat
/// slots and experts.
pub const G942_MOE_WIDE_MAX_SLOTS: usize = 640;
pub const G942_MOE_WIDE_MAX_EXPERTS: usize = 512;
/// Tokens per router workgroup (`MW_ROUTER_WAVES`: one wave64 per token).
const G942_MOE_WIDE_ROUTER_TOKENS_PER_BLOCK: usize = 4;

/// `HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE=/absolute/device.elf` (developer, read once): the explicitly
/// supplied code object of [`QWEN4_G942_MOE_WIDE_MODULE`] (the Route-A twin proof). Unset: the
/// hipcc source, unchanged.
fn g942_moe_wide_image() -> HipResult<Option<&'static std::sync::Arc<[u8]>>> {
    static IMAGE: std::sync::LazyLock<Result<Option<std::sync::Arc<[u8]>>, String>> =
        std::sync::LazyLock::new(|| {
            match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE") {
                Ok(path) if !path.is_empty() => std::fs::read(&path)
                    .map(|bytes| Some(std::sync::Arc::<[u8]>::from(bytes)))
                    .map_err(|error| {
                        format!("HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE {path}: {error}")
                    }),
                _ => Ok(None),
            }
        });
    let loaded: &'static Result<Option<std::sync::Arc<[u8]>>, String> = &IMAGE;
    match loaded {
        Ok(image) => Ok(image.as_ref()),
        Err(message) => Err(hip_bridge::HipError::new(0, message)),
    }
}

/// Bind `func` of the wide glue module: the hipcc source (the anchor object) or, only when
/// `HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE` names one, that code object (same module, same entry
/// names, loaded through `ensure_embedded_kernel`).
fn g942_moe_wide_ensure(gpu: &mut Gpu, func: &str) -> HipResult<()> {
    match g942_moe_wide_image()? {
        Some(image) => {
            gpu.ensure_embedded_kernel(QWEN4_G942_MOE_WIDE_MODULE, image.clone(), func)
        }
        None => gpu.ensure_kernel(QWEN4_G942_MOE_WIDE_MODULE, QWEN4_G942_MOE_WIDE_SRC, func),
    }
}
const QWEN4_GDN_LANES_F32_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_gdn_lanes_f32.gfx942.hip");
const QWEN4_GDN_LANES_F32_KERNEL: &str = "qwen4_gdn_lanes_step_f32_gfx942";
/// Most lanes one [`Gfx942Device::qwen4_gdn_lanes_step_f32`] launch covers (grid.y;
/// `serve.continuous_batch_size` tops out at 256).
pub const QWEN4_GDN_LANES_F32_MAX: usize = 256;
/// Lane tables the cache keeps before it is dropped whole (one table per layer and
/// lane set: 36 layers x about a hundred distinct lane sets).
const GDN_LANES_TABLE_CACHE_MAX: usize = 4096;

const QWEN4_GDN_LANES_CAPTURE_F32_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_gdn_lanes_capture_f32.gfx942.hip");
const QWEN4_GDN_LANES_CAPTURE_F32_KERNEL: &str = "qwen4_gdn_lanes_step_capture_f32_gfx942";

/// One lane of the device lane table: the five pointers of one lane's recurrence
/// (`struct Qwen4GdnLaneDesc` in `kernels/src/qwen4_gdn_lanes_f32.gfx942.hip`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Qwen4GdnLaneDesc {
    projection: u64,
    gate: u64,
    beta: u64,
    state: u64,
    output: u64,
}
const _: () = assert!(std::mem::size_of::<Qwen4GdnLaneDesc>() == 40);

/// One lane of the capture lanes' device table: the six pointers of one lane's captured
/// recurrence (`struct Qwen4GdnLaneCaptureDesc` in
/// `kernels/src/qwen4_gdn_lanes_capture_f32.gfx942.hip`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Qwen4GdnLaneCaptureDesc {
    projection: u64,
    gate: u64,
    beta: u64,
    state: u64,
    final_state: u64,
    output: u64,
}
const _: () = assert!(std::mem::size_of::<Qwen4GdnLaneCaptureDesc>() == 48);

/// [`GdnLanesTableKey`] of the capture lanes.
type GdnLanesCaptureTableKey = ([usize; 3], Vec<Qwen4GdnLaneCaptureDesc>);

/// `[rows, key_heads, value_heads]` and the lane descriptors a cached table was built from.
type GdnLanesTableKey = ([usize; 3], Vec<Qwen4GdnLaneDesc>);

/// Device lane tables of [`Gfx942Device::qwen4_gdn_lanes_step_f32`], keyed by content (shape
/// and every lane's five pointers): a table is a pure function of its key, so a hit is always
/// current and a lane set that changes simply misses once per layer. Each entry keeps its host
/// bytes until the table is dropped, so its stream-ordered upload never races the source.
#[derive(Default)]
pub(crate) struct GdnLanesTableCache {
    tables: std::collections::HashMap<GdnLanesTableKey, (hip_bridge::DeviceBuffer, Vec<u8>)>,
    /// The capture lanes' tables ([`Gfx942Device::qwen4_gdn_lanes_step_capture_f32`]), keyed
    /// the same way over the six-pointer descriptors.
    capture_tables:
        std::collections::HashMap<GdnLanesCaptureTableKey, (hip_bridge::DeviceBuffer, Vec<u8>)>,
    /// The GDN Core lanes' tables ([`Gfx942Device::qwen4_gdn_lanes_verify`], one seventeen-pointer
    /// descriptor per lane).
    verify_tables: std::collections::HashMap<
        gdn_lanes_verify::GdnLanesVerifyTableKey,
        (hip_bridge::DeviceBuffer, Vec<u8>),
    >,
}

/// One lane of a [`Gfx942Device::qwen4_gdn_lanes_step_f32`] launch: the tensors the
/// per-lane `gated_delta_step_batched` launch of that lane names.
pub struct Qwen4GdnLanesF32Lane<'a> {
    /// F32 `[rows, qkv_width]` convolution output (q | k | v per row).
    pub projection: &'a GpuTensor,
    /// F32 `[rows, value_heads]` decay gate.
    pub gate: &'a GpuTensor,
    /// F32 `[rows, value_heads]` beta.
    pub beta: &'a GpuTensor,
    /// F32 `[value_heads, 128, 128]` recurrent state, read then updated in place.
    pub state: &'a GpuTensor,
    /// F32 `[rows, value_heads * 128]` recurrent output.
    pub output: &'a GpuTensor,
}

/// Shape shared by every lane of one [`Gfx942Device::qwen4_gdn_lanes_step_f32`] launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4GdnLanesF32Shape {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

/// One lane of a [`Gfx942Device::qwen4_gdn_lanes_step_capture_f32`] launch: the tensors the
/// per-lane capture launch of that lane (`gated_delta_step_batched` with `row_states`) names.
pub struct Qwen4GdnLanesCaptureF32Lane<'a> {
    /// F32 `[rows, qkv_width]` convolution output (q | k | v per row).
    pub projection: &'a GpuTensor,
    /// F32 `[rows, value_heads]` decay gate.
    pub gate: &'a GpuTensor,
    /// F32 `[rows, value_heads]` beta.
    pub beta: &'a GpuTensor,
    /// F32 `[value_heads, 128, 128]` recurrent state: read only (the pre-window state a
    /// rollback replays from).
    pub state: &'a GpuTensor,
    /// The lane's capture ring (`GdnRowCapture::states`): F32 slots of `[value_heads, 128, 128]`,
    /// at least `rows` of them; the state after the last row lands in slot `rows - 1`.
    pub ring: &'a GpuTensor,
    /// F32 `[rows, value_heads * 128]` recurrent output.
    pub output: &'a GpuTensor,
}

/// Bind the lanes kernel: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_GDN_LANES_IMAGE=/absolute/device.elf` names one, that explicitly supplied
/// code object (the Route-A twin proof: same module, same entry name, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_gdn_lanes_ensure(gpu: &mut Gpu) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_GDN_LANES_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_GDN_LANES_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_GDN_LANES_F32_KERNEL,
                std::sync::Arc::<[u8]>::from(bytes),
                QWEN4_GDN_LANES_F32_KERNEL,
            )
        }
        _ => gpu.ensure_kernel(
            QWEN4_GDN_LANES_F32_KERNEL,
            QWEN4_GDN_LANES_F32_SRC,
            QWEN4_GDN_LANES_F32_KERNEL,
        ),
    }
}

/// Bind the capture lanes kernel (`qwen4_gdn_lanes_step_capture_f32_gfx942`, its own module):
/// the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_GDN_LANES_CAPTURE_IMAGE=/absolute/device.elf` names one, that explicitly
/// supplied code object (the Route-A twin proof: same module, same entry name, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_gdn_lanes_capture_ensure(gpu: &mut Gpu) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_GDN_LANES_CAPTURE_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_GDN_LANES_CAPTURE_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
                std::sync::Arc::<[u8]>::from(bytes),
                QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
            )
        }
        _ => gpu.ensure_kernel(
            QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
            QWEN4_GDN_LANES_CAPTURE_F32_SRC,
            QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
        ),
    }
}

const QWEN4_G942_GDN_VERIFY_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_gdn_verify.gfx942.hip");
/// The exact-gfx942 GDN verify-window module (its own cache key and twin seat).
pub const QWEN4_G942_GDN_VERIFY_MODULE: &str = "qwen4_g942_gdn_verify";
/// `(tensor_ops entry, gfx942 verify module entry)`: the four entries whose gfx942 launches live in
/// [`QWEN4_G942_GDN_VERIFY_MODULE`]. The names differ (the per-context function table is keyed by
/// entry name, so the module cannot reuse the tensor_ops symbols); every body is a verbatim copy.
pub const QWEN4_G942_GDN_VERIFY_ENTRIES: [(&str, &str); 4] = [
    (
        "gated_delta_step_halves_state128_persistent256_capture_f32",
        "qwen4_g942_gdn_verify_capture_f32",
    ),
    (
        "gated_delta_rollback_layers_f32",
        "qwen4_g942_gdn_verify_rollback_f32",
    ),
    (
        "gated_delta_conv_bf16_f32_batched_k4",
        "qwen4_g942_gdn_verify_conv_params_k4",
    ),
    (
        "gated_delta_gate_bf16_f32_batched",
        "qwen4_g942_gdn_verify_gate_batched",
    ),
];

/// `HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE=/absolute/device.elf` (developer, read once): the
/// explicitly supplied code object of [`QWEN4_G942_GDN_VERIFY_MODULE`] (the Route-A twin proof).
/// Unset: the hipcc source, unchanged.
fn g942_gdn_verify_image() -> HipResult<Option<&'static std::sync::Arc<[u8]>>> {
    static IMAGE: std::sync::LazyLock<Result<Option<std::sync::Arc<[u8]>>, String>> =
        std::sync::LazyLock::new(|| {
            match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE") {
                Ok(path) if !path.is_empty() => std::fs::read(&path)
                    .map(|bytes| Some(std::sync::Arc::<[u8]>::from(bytes)))
                    .map_err(|error| {
                        format!("HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE {path}: {error}")
                    }),
                _ => Ok(None),
            }
        });
    let loaded: &'static Result<Option<std::sync::Arc<[u8]>>, String> = &IMAGE;
    match loaded {
        Ok(image) => Ok(image.as_ref()),
        Err(message) => Err(hip_bridge::HipError::new(0, message)),
    }
}

/// Load the entry `kernel` of the tensor_ops GDN verify launches and return the symbol to launch.
/// On exact gfx942 the four [`QWEN4_G942_GDN_VERIFY_ENTRIES`] resolve to the gfx942-only module
/// (hipcc source, or the `HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE` code object) and the module's own
/// symbol is returned; every other arch, and every other entry, loads from `tensor_ops` as
/// before and returns `kernel`.
pub(crate) fn ensure_gdn_verify_kernel(gpu: &mut Gpu, kernel: &'static str) -> HipResult<&'static str> {
    let entry = gpu
        .arch_caps
        .is_gfx942()
        .then(|| {
            QWEN4_G942_GDN_VERIFY_ENTRIES
                .iter()
                .find(|(tensor_ops, _)| *tensor_ops == kernel)
        })
        .flatten();
    let Some(&(_, own)) = entry else {
        gpu.ensure_kernel_public("tensor_ops", crate::tensor_ops::TENSOR_OPS_SRC, kernel)?;
        return Ok(kernel);
    };
    gpu.bind_thread()?;
    match g942_gdn_verify_image()? {
        Some(image) => gpu.ensure_embedded_kernel(QWEN4_G942_GDN_VERIFY_MODULE, image.clone(), own)?,
        None => gpu.ensure_kernel(QWEN4_G942_GDN_VERIFY_MODULE, QWEN4_G942_GDN_VERIFY_SRC, own)?,
    }
    Ok(own)
}

const QWEN4_G942_QSA_ROW_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_qsa_row_attention.gfx942.hip");
/// The exact-gfx942 single-row parallel QSA attention module (its own cache key and twin seat).
pub const QWEN4_G942_QSA_ROW_MODULE: &str = "qwen4_g942_qsa_row_attention";
/// `(tensor_ops entry, gfx942 row module entry)`: the two parallel single-row QSA attention
/// entries exact gfx942 can select (K/V formats F32 and E4M3 FNUZ; `fp8` is gfx12-only and `q8`
/// gfx11/gfx12-only). The serial entries stay in `tensor_ops`. The names differ (the per-context
/// function table is keyed by entry name); the device body is `tensor_ops.hip`'s, byte for byte.
pub const QWEN4_G942_QSA_ROW_ENTRIES: [(&str, &str); 2] = [
    (
        "indexed_attention_attention_f32",
        "qwen4_g942_qsa_row_f32",
    ),
    (
        "indexed_attention_attention_fp8_fnuz",
        "qwen4_g942_qsa_row_fp8_fnuz",
    ),
];

/// `HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE=/absolute/device.elf` (developer, read once): the explicitly
/// supplied code object of [`QWEN4_G942_QSA_ROW_MODULE`] (the Route-A twin proof). Unset: the
/// hipcc source, unchanged.
fn g942_qsa_row_image() -> HipResult<Option<&'static std::sync::Arc<[u8]>>> {
    static IMAGE: std::sync::LazyLock<Result<Option<std::sync::Arc<[u8]>>, String>> =
        std::sync::LazyLock::new(|| {
            match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE") {
                Ok(path) if !path.is_empty() => std::fs::read(&path)
                    .map(|bytes| Some(std::sync::Arc::<[u8]>::from(bytes)))
                    .map_err(|error| {
                        format!("HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE {path}: {error}")
                    }),
                _ => Ok(None),
            }
        });
    let loaded: &'static Result<Option<std::sync::Arc<[u8]>>, String> = &IMAGE;
    match loaded {
        Ok(image) => Ok(image.as_ref()),
        Err(message) => Err(hip_bridge::HipError::new(0, message)),
    }
}

/// Load the single-row QSA attention entry `kernel` and return the symbol to launch. On exact
/// gfx942 the two [`QWEN4_G942_QSA_ROW_ENTRIES`] resolve to the gfx942-only module (hipcc source,
/// or the `HIPFIRE_QWEN4_G942_QSA_ROW_IMAGE` code object) and the module's own symbol is
/// returned; every other arch, and every other entry (the serial row kernels), loads from
/// `tensor_ops` as before and returns `kernel`.
pub(crate) fn ensure_qsa_row_kernel(gpu: &mut Gpu, kernel: &'static str) -> HipResult<&'static str> {
    let entry = gpu
        .arch_caps
        .is_gfx942()
        .then(|| {
            QWEN4_G942_QSA_ROW_ENTRIES
                .iter()
                .find(|(tensor_ops, _)| *tensor_ops == kernel)
        })
        .flatten();
    let Some(&(_, own)) = entry else {
        gpu.ensure_kernel_public("tensor_ops", crate::tensor_ops::TENSOR_OPS_SRC, kernel)?;
        return Ok(kernel);
    };
    gpu.bind_thread()?;
    match g942_qsa_row_image()? {
        Some(image) => gpu.ensure_embedded_kernel(QWEN4_G942_QSA_ROW_MODULE, image.clone(), own)?,
        None => gpu.ensure_kernel(QWEN4_G942_QSA_ROW_MODULE, QWEN4_G942_QSA_ROW_SRC, own)?,
    }
    Ok(own)
}

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

const QWEN4_G942_GLUE_ELEM_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_glue_elem.hip");
const QWEN4_G942_GLUE_ELEM_MODULE: &str = "qwen4_g942_glue_elem";
const QWEN4_G942_HC_DOWN_ACT_KERNEL: &str = "qwen4_g942_gemm_bf16_rows16_hc_act";
const QWEN4_G942_HC_DOWN_ACT_X2_KERNEL: &str = "qwen4_g942_gemm_bf16_rows16_hc_act_x2";
/// LDS row pitch (BF16 elements) of the few-row GEMM bodies: a 256-K chunk plus a
/// 4-element pad (`R16_LD` in `qwen4_g942_glue_elem.hip`).
const QWEN4_G942_ROWS16_LDS_PITCH: usize = 256 + 4;

/// Bind `func` of the elementwise-fold module: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_FOLD_ELEM_IMAGE=/absolute/device.elf` names one, that explicitly supplied
/// code object (the Route-A twin proof: same module, same entry names, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_fold_elem_ensure(gpu: &mut Gpu, func: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_FOLD_ELEM_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_FOLD_ELEM_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_G942_GLUE_ELEM_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                func,
            )
        }
        _ => gpu.ensure_kernel(QWEN4_G942_GLUE_ELEM_MODULE, QWEN4_G942_GLUE_ELEM_SRC, func),
    }
}

const QWEN4_G942_HC_GATES_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_hc_gates.gfx942.hip");
const QWEN4_G942_HC_GATES_MODULE: &str = "qwen4_g942_hc_gates";
/// The seven `gemv_bf16_xf32_k4_rows_tiled_r<R>` entries of the module, R = 2..=8.
pub(crate) const QWEN4_G942_HC_GATES_TILED: [&str; 7] = [
    "gemv_bf16_xf32_k4_rows_tiled_r2",
    "gemv_bf16_xf32_k4_rows_tiled_r3",
    "gemv_bf16_xf32_k4_rows_tiled_r4",
    "gemv_bf16_xf32_k4_rows_tiled_r5",
    "gemv_bf16_xf32_k4_rows_tiled_r6",
    "gemv_bf16_xf32_k4_rows_tiled_r7",
    "gemv_bf16_xf32_k4_rows_tiled_r8",
];

/// Bind `func` (one of [`QWEN4_G942_HC_GATES_TILED`]) of the gfx942 HC-gates module: the hipcc
/// source of this subset module (the anchor object, `kernels/qwen4_g942_hc_gates_hipcc_gfx942.co`),
/// or, only when `HIPFIRE_QWEN4_G942_HC_GATES_IMAGE=/absolute/device.elf` names one, that
/// explicitly supplied code object (the Route-A twin proof: same module, same entry names,
/// loaded through `ensure_embedded_kernel`). Unset: the source, unchanged. Only this module is
/// affected: the one-row `gemv_bf16_xf32_k4` and every other `qwen4_gemv_bf16_xf32` entry keep
/// their own module and cache entry.
pub(crate) fn g942_hc_gates_ensure(gpu: &mut Gpu, func: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_HC_GATES_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_HC_GATES_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_G942_HC_GATES_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                func,
            )
        }
        _ => gpu.ensure_kernel(QWEN4_G942_HC_GATES_MODULE, QWEN4_G942_HC_GATES_SRC, func),
    }
}

impl Gfx942Device<'_> {
    /// The HC read's down projection and its activation as ONE launch (C1 fold,
    /// kill switch `HIPFIRE_QWEN4_G942_FOLD_ELEM`):
    /// `out[b, m] = hc_act(Σ_k a[m, k] · x[b, k])`, `hc_act(v) = bf16(silu(bf16(bf16(v) · scale)))`
    /// in the unfused chain's expressions (see `kernels/src/qwen4_g942_glue_elem.hip`).
    ///
    /// Bitwise `gemm_bf16_mfma_gfx942_rows16` (F32 `out`) followed by
    /// `bf16_roundtrip_f32`, `scale_f32`, `bf16_roundtrip_f32`, `silu_f32`,
    /// `bf16_roundtrip_f32` over `out`: the GEMM body is a single-member copy of the
    /// rows16 entry and only its output store differs. `a` is the BF16 weight
    /// `[m, k]`, `x` the F32 (rounded to BF16 in-kernel) or BF16 `[batch, k]`
    /// activation, `out` F32 `[batch, m]`.
    ///
    /// Returns `Ok(false)` without launching when the entry does not apply (`batch`
    /// outside 1..=16, `k % 8 != 0`, `a` or `x` not 16-byte aligned, a dtype outside
    /// F32 / BF16 for `x` or not F32 for `out`, or `out` overlapping `x`); the caller
    /// keeps the unfused launches. Block `[576,1,1]`, grid `[ceil(m/16),1,1]`, dynamic
    /// LDS `2 × (16 + batch) × 260 × 2 B`; the `_x2` entry takes an F32 `x` of more
    /// than 8 rows.
    pub fn qwen4_hc_down_act_rows16(
        &mut self,
        a: &DeviceBuffer,
        x: &GpuTensor,
        out: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
        scale: f32,
    ) -> HipResult<bool> {
        const WHAT: &str = "qwen4_hc_down_act_rows16";
        let (x_f32, x_size) = match x.dtype {
            DType::F32 => (true, 4usize),
            DType::BF16 => (false, 2usize),
            _ => return Ok(false),
        };
        if out.dtype != DType::F32
            || m == 0
            || !(1..=16).contains(&batch)
            || k == 0
            || k % 8 != 0
            || m > i32::MAX as usize
            || k > i32::MAX as usize
            || (a.as_ptr() as usize) % 16 != 0
            || (x.buf.as_ptr() as usize) % 16 != 0
        {
            return Ok(false);
        }
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let (Some(a_bytes), Some(x_bytes), Some(out_bytes)) = (
            m.checked_mul(k).and_then(|e| e.checked_mul(2)),
            batch.checked_mul(k).and_then(|e| e.checked_mul(x_size)),
            batch.checked_mul(m).and_then(|e| e.checked_mul(4)),
        ) else {
            return refuse(format!(
                "{WHAT}: extent overflow (m={m} k={k} batch={batch})"
            ));
        };
        if a.size() < a_bytes || x.buf.size() < x_bytes || out.buf.size() < out_bytes {
            return refuse(format!(
                "{WHAT}: buffer too small (a {}/{a_bytes}, x {}/{x_bytes}, out {}/{out_bytes})",
                a.size(),
                x.buf.size(),
                out.buf.size()
            ));
        }
        // Every workgroup reads x for the whole launch: an output that overlaps it
        // keeps the unfused route (which writes `out` after its own read of x).
        let (x_start, out_start) = (x.buf.as_ptr() as usize, out.buf.as_ptr() as usize);
        if out_start < x_start + x_bytes && x_start < out_start + out_bytes {
            return Ok(false);
        }
        let kernel = if x_f32 && batch > 8 {
            QWEN4_G942_HC_DOWN_ACT_X2_KERNEL
        } else {
            QWEN4_G942_HC_DOWN_ACT_KERNEL
        };
        self.gpu.bind_thread()?;
        g942_fold_elem_ensure(&mut *self.gpu, kernel)?;
        let (x_ptr, a_ptr, d_ptr) = (x.buf.as_ptr(), a.as_ptr(), out.buf.as_ptr());
        let (m_i32, k_i32, batch_i32, x_f32_i32) =
            (m as i32, k as i32, batch as i32, i32::from(x_f32));
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &a_ptr as *const _ as *mut c_void,
            &d_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
            &batch_i32 as *const _ as *mut c_void,
            &x_f32_i32 as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        let lds = (2 * (16 + batch) * QWEN4_G942_ROWS16_LDS_PITCH * 2) as u32;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "gemm",
            kernel,
            a_bytes + x_bytes + out_bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            kernel,
            [m.div_ceil(16) as u32, 1, 1],
            [576, 1, 1],
            lds,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(a_ptr);
                blob.push_ptr(d_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob.push_i32(batch_i32);
                blob.push_i32(x_f32_i32);
                blob.push_f32(scale);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result.map(|()| true)
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

    /// The one-row (or few-row) F32 GDN recurrence of up to
    /// [`QWEN4_GDN_LANES_F32_MAX`] lanes in ONE launch
    /// (`qwen4_gdn_lanes_step_f32_gfx942`, grid `[value_heads, lanes]`, block 128).
    ///
    /// Lane `i` runs `gated_delta_step_pipe128_f32`'s body (the kernel
    /// `gated_delta_step_batched` launches for an F32 state on gfx942) on lane `i`'s
    /// own tensors, so its state and output bytes equal that lane's per-lane launch.
    /// The lanes' pointers ride in a device lane table, built once per distinct lane set
    /// and layer and reused until the set changes (the cache is keyed by content).
    ///
    /// Refused (`Err`, nothing launched) unless every tensor is F32 with exactly the
    /// extent the singleton launch validates, the shape is the exact 128 x 128 GDN one,
    /// no lane's written range (state, output) overlaps another lane's written range or
    /// any lane's read range (projection, gate, beta), and, during hipGraph capture, the
    /// table is already cached.
    pub fn qwen4_gdn_lanes_step_f32(
        &mut self,
        lanes: &[Qwen4GdnLanesF32Lane<'_>],
        shape: Qwen4GdnLanesF32Shape,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_gdn_lanes_step_f32";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let Qwen4GdnLanesF32Shape {
            rows,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = shape;
        if lanes.is_empty() || lanes.len() > QWEN4_GDN_LANES_F32_MAX {
            return refuse(format!(
                "{WHAT}: {} lanes (1..={QWEN4_GDN_LANES_F32_MAX})",
                lanes.len()
            ));
        }
        if rows == 0
            || key_heads == 0
            || value_heads == 0
            || value_heads % key_heads != 0
            || key_dim != 128
            || value_dim != 128
        {
            return refuse(format!(
                "{WHAT}: needs the exact 128x128 GDN shape (rows={rows} key_heads={key_heads} \
                 value_heads={value_heads} key_dim={key_dim} value_dim={value_dim})"
            ));
        }
        let qk = key_heads * 128;
        let value = value_heads * 128;
        let qkv = 2 * qk + value;
        let (Some(proj_elems), Some(gate_elems), Some(out_elems)) = (
            rows.checked_mul(qkv),
            rows.checked_mul(value_heads),
            rows.checked_mul(value),
        ) else {
            return refuse(format!("{WHAT}: lane extent overflows"));
        };
        let state_elems = value * 128;
        // The kernel indexes rows with 32-bit ints.
        if proj_elems.max(out_elems) > i32::MAX as usize {
            return refuse(format!(
                "{WHAT}: rows={rows} exceeds the kernel's 32-bit row index"
            ));
        }
        let mut descs = Vec::with_capacity(lanes.len());
        for (i, lane) in lanes.iter().enumerate() {
            for (name, tensor, elems) in [
                ("projection", lane.projection, proj_elems),
                ("gate", lane.gate, gate_elems),
                ("beta", lane.beta, gate_elems),
                ("state", lane.state, state_elems),
                ("output", lane.output, out_elems),
            ] {
                if tensor.dtype != DType::F32
                    || tensor.numel() != elems
                    || tensor.buf.size() < elems * 4
                {
                    return refuse(format!(
                        "{WHAT}: lane {i} {name} must be F32 with {elems} elements (got {:?}, {} \
                         elements, {} bytes)",
                        tensor.dtype,
                        tensor.numel(),
                        tensor.buf.size()
                    ));
                }
            }
            descs.push(Qwen4GdnLaneDesc {
                projection: lane.projection.buf.as_ptr() as u64,
                gate: lane.gate.buf.as_ptr() as u64,
                beta: lane.beta.buf.as_ptr() as u64,
                state: lane.state.buf.as_ptr() as u64,
                output: lane.output.buf.as_ptr() as u64,
            });
        }
        let key: GdnLanesTableKey = ([rows, key_heads, value_heads], descs);
        if !self.gpu.gdn_lanes_f32_tables.tables.contains_key(&key) {
            // Validated once per distinct lane set: the table is a pure function of the key.
            let descs = &key.1;
            let mut writes: Vec<(u64, u64)> = Vec::with_capacity(2 * descs.len());
            for d in descs {
                writes.push((d.state, d.state + (state_elems * 4) as u64));
                writes.push((d.output, d.output + (out_elems * 4) as u64));
            }
            writes.sort_unstable();
            if writes.windows(2).any(|w| w[0].1 > w[1].0) {
                return refuse(format!(
                    "{WHAT}: two lanes write overlapping state or output ranges"
                ));
            }
            let hits_write = |start: u64, bytes: usize| {
                let end = start + bytes as u64;
                // The last write starting before `end` is the only one that can reach back over
                // `start` (the writes are disjoint and sorted).
                let at = writes.partition_point(|w| w.0 < end);
                at > 0 && writes[at - 1].1 > start
            };
            for d in descs {
                if hits_write(d.projection, proj_elems * 4)
                    || hits_write(d.gate, gate_elems * 4)
                    || hits_write(d.beta, gate_elems * 4)
                {
                    return refuse(format!(
                        "{WHAT}: a lane reads a range another lane (or itself) writes"
                    ));
                }
            }
            if self.gpu.graphs.capture_mode {
                return refuse(format!("{WHAT}: lane table staging during graph capture"));
            }
            self.gpu.bind_thread()?;
            if self.gpu.gdn_lanes_f32_tables.tables.len() >= GDN_LANES_TABLE_CACHE_MAX {
                // Queued launches may still read a table: drain before freeing any.
                self.gpu.hip.device_synchronize()?;
                self.gpu.gdn_lanes_f32_tables.tables.clear();
            }
            let bytes: Vec<u8> = key
                .1
                .iter()
                .flat_map(|d| {
                    [d.projection, d.gate, d.beta, d.state, d.output]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                })
                .collect();
            let buf = self.gpu.hip.malloc(bytes.len().next_multiple_of(256))?;
            // Stream-ordered before this launch; `bytes` lives in the cache entry, so the
            // (possibly staged) copy never reads a dropped source and nothing synchronizes.
            match self.gpu.active_stream.as_ref() {
                Some(stream) => self.gpu.hip.memcpy_htod_async(&buf, &bytes, stream)?,
                None => self.gpu.hip.memcpy_htod(&buf, &bytes)?,
            }
            self.gpu
                .gdn_lanes_f32_tables
                .tables
                .insert(key.clone(), (buf, bytes));
        }
        let table = self.gpu.gdn_lanes_f32_tables.tables[&key].0.as_ptr();
        self.gpu.bind_thread()?;
        g942_gdn_lanes_ensure(&mut *self.gpu)?;
        let (rv, qv, khv, vhv, kdv, vdv) = (
            rows as i32,
            qkv as i32,
            key_heads as i32,
            value_heads as i32,
            key_dim as i32,
            value_dim as i32,
        );
        let scale = (key_dim as f32).sqrt().recip();
        let mut params: Vec<*mut c_void> = vec![
            &table as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &qv as *const _ as *mut c_void,
            &khv as *const _ as *mut c_void,
            &vhv as *const _ as *mut c_void,
            &kdv as *const _ as *mut c_void,
            &vdv as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        let bytes = lanes.len().saturating_mul(
            2 * state_elems * 4 + (proj_elems + out_elems + 2 * gate_elems) * 4,
        );
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "deltanet",
            QWEN4_GDN_LANES_F32_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_GDN_LANES_F32_KERNEL,
            [value_heads as u32, lanes.len() as u32, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(table);
                blob.push_i32(rv);
                blob.push_i32(qv);
                blob.push_i32(khv);
                blob.push_i32(vhv);
                blob.push_i32(kdv);
                blob.push_i32(vdv);
                blob.push_f32(scale);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// The captured F32 GDN verify-window recurrence (`rows` rows per lane, uniform) of up to
    /// [`QWEN4_GDN_LANES_F32_MAX`] lanes in ONE launch
    /// (`qwen4_gdn_lanes_step_capture_f32_gfx942`, grid `[value_heads, lanes]`, block 128).
    ///
    /// Lane `i` runs `gated_delta_step_pipe128_f32`'s body (the AR singleton's F32 recurrence)
    /// over its `rows` rows from its `state`, which is NOT written: the state after the last
    /// row is stored to slot `rows - 1` of its `ring`, the layout the per-lane capture launch
    /// (`gated_delta_step_batched` with `row_states`) leaves, so the lane's rollback
    /// (`gated_delta_rollback_layers`) and commit read it unchanged. The lane pointers ride in a
    /// device lane table built once per distinct lane set and layer and reused until the set
    /// changes (content-keyed, as [`Self::qwen4_gdn_lanes_step_f32`]).
    ///
    /// Refused (`Err`, nothing launched) unless every tensor is F32 with exactly the extent the
    /// per-lane launch validates, the shape is the exact 128 x 128 GDN one, no lane's written
    /// range (final-state slot, output) overlaps another lane's written range or any lane's
    /// read range (projection, gate, beta, state), and, during hipGraph capture, the table is
    /// already cached.
    pub fn qwen4_gdn_lanes_step_capture_f32(
        &mut self,
        lanes: &[Qwen4GdnLanesCaptureF32Lane<'_>],
        shape: Qwen4GdnLanesF32Shape,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_gdn_lanes_step_capture_f32";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let Qwen4GdnLanesF32Shape {
            rows,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = shape;
        if lanes.is_empty() || lanes.len() > QWEN4_GDN_LANES_F32_MAX {
            return refuse(format!(
                "{WHAT}: {} lanes (1..={QWEN4_GDN_LANES_F32_MAX})",
                lanes.len()
            ));
        }
        if rows == 0
            || key_heads == 0
            || value_heads == 0
            || value_heads % key_heads != 0
            || key_dim != 128
            || value_dim != 128
        {
            return refuse(format!(
                "{WHAT}: needs the exact 128x128 GDN shape (rows={rows} key_heads={key_heads} \
                 value_heads={value_heads} key_dim={key_dim} value_dim={value_dim})"
            ));
        }
        let qk = key_heads * 128;
        let value = value_heads * 128;
        let qkv = 2 * qk + value;
        let (Some(proj_elems), Some(gate_elems), Some(out_elems), Some(ring_elems)) = (
            rows.checked_mul(qkv),
            rows.checked_mul(value_heads),
            rows.checked_mul(value),
            rows.checked_mul(value * 128),
        ) else {
            return refuse(format!("{WHAT}: lane extent overflows"));
        };
        let state_elems = value * 128;
        // The kernel indexes rows with 32-bit ints.
        if proj_elems.max(out_elems) > i32::MAX as usize {
            return refuse(format!(
                "{WHAT}: rows={rows} exceeds the kernel's 32-bit row index"
            ));
        }
        let mut descs = Vec::with_capacity(lanes.len());
        for (i, lane) in lanes.iter().enumerate() {
            // The ring only has to HOLD `rows` slots (a compact capture view may be larger).
            for (name, tensor, elems, exact) in [
                ("projection", lane.projection, proj_elems, true),
                ("gate", lane.gate, gate_elems, true),
                ("beta", lane.beta, gate_elems, true),
                ("state", lane.state, state_elems, true),
                ("ring", lane.ring, ring_elems, false),
                ("output", lane.output, out_elems, true),
            ] {
                let extent_ok = if exact {
                    tensor.numel() == elems
                } else {
                    tensor.numel() >= elems
                };
                if tensor.dtype != DType::F32 || !extent_ok || tensor.buf.size() < elems * 4 {
                    return refuse(format!(
                        "{WHAT}: lane {i} {name} must be F32 with {}{elems} elements (got {:?}, \
                         {} elements, {} bytes)",
                        if exact { "" } else { "at least " },
                        tensor.dtype,
                        tensor.numel(),
                        tensor.buf.size()
                    ));
                }
            }
            descs.push(Qwen4GdnLaneCaptureDesc {
                projection: lane.projection.buf.as_ptr() as u64,
                gate: lane.gate.buf.as_ptr() as u64,
                beta: lane.beta.buf.as_ptr() as u64,
                state: lane.state.buf.as_ptr() as u64,
                final_state: lane.ring.buf.as_ptr() as u64 + ((rows - 1) * state_elems * 4) as u64,
                output: lane.output.buf.as_ptr() as u64,
            });
        }
        let key: GdnLanesCaptureTableKey = ([rows, key_heads, value_heads], descs);
        if !self.gpu.gdn_lanes_f32_tables.capture_tables.contains_key(&key) {
            // Validated once per distinct lane set: the table is a pure function of the key.
            let descs = &key.1;
            let mut writes: Vec<(u64, u64)> = Vec::with_capacity(2 * descs.len());
            for d in descs {
                writes.push((d.final_state, d.final_state + (state_elems * 4) as u64));
                writes.push((d.output, d.output + (out_elems * 4) as u64));
            }
            writes.sort_unstable();
            if writes.windows(2).any(|w| w[0].1 > w[1].0) {
                return refuse(format!(
                    "{WHAT}: two lanes write overlapping final-state or output ranges"
                ));
            }
            let hits_write = |start: u64, bytes: usize| {
                let end = start + bytes as u64;
                // The last write starting before `end` is the only one that can reach back over
                // `start` (the writes are disjoint and sorted).
                let at = writes.partition_point(|w| w.0 < end);
                at > 0 && writes[at - 1].1 > start
            };
            for d in descs {
                if hits_write(d.projection, proj_elems * 4)
                    || hits_write(d.gate, gate_elems * 4)
                    || hits_write(d.beta, gate_elems * 4)
                    || hits_write(d.state, state_elems * 4)
                {
                    return refuse(format!(
                        "{WHAT}: a lane reads a range another lane (or itself) writes"
                    ));
                }
            }
            if self.gpu.graphs.capture_mode {
                return refuse(format!("{WHAT}: lane table staging during graph capture"));
            }
            self.gpu.bind_thread()?;
            if self.gpu.gdn_lanes_f32_tables.capture_tables.len() >= GDN_LANES_TABLE_CACHE_MAX {
                // Queued launches may still read a table: drain before freeing any.
                self.gpu.hip.device_synchronize()?;
                self.gpu.gdn_lanes_f32_tables.capture_tables.clear();
            }
            let bytes: Vec<u8> = key
                .1
                .iter()
                .flat_map(|d| {
                    [d.projection, d.gate, d.beta, d.state, d.final_state, d.output]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                })
                .collect();
            let buf = self.gpu.hip.malloc(bytes.len().next_multiple_of(256))?;
            // Stream-ordered before this launch; `bytes` lives in the cache entry, so the
            // (possibly staged) copy never reads a dropped source and nothing synchronizes.
            match self.gpu.active_stream.as_ref() {
                Some(stream) => self.gpu.hip.memcpy_htod_async(&buf, &bytes, stream)?,
                None => self.gpu.hip.memcpy_htod(&buf, &bytes)?,
            }
            self.gpu
                .gdn_lanes_f32_tables
                .capture_tables
                .insert(key.clone(), (buf, bytes));
        }
        let table = self.gpu.gdn_lanes_f32_tables.capture_tables[&key].0.as_ptr();
        self.gpu.bind_thread()?;
        g942_gdn_lanes_capture_ensure(&mut *self.gpu)?;
        let (rv, qv, khv, vhv, kdv, vdv) = (
            rows as i32,
            qkv as i32,
            key_heads as i32,
            value_heads as i32,
            key_dim as i32,
            value_dim as i32,
        );
        let scale = (key_dim as f32).sqrt().recip();
        let mut params: Vec<*mut c_void> = vec![
            &table as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &qv as *const _ as *mut c_void,
            &khv as *const _ as *mut c_void,
            &vhv as *const _ as *mut c_void,
            &kdv as *const _ as *mut c_void,
            &vdv as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        let bytes = lanes
            .len()
            .saturating_mul(2 * state_elems * 4 + (proj_elems + out_elems + 2 * gate_elems) * 4);
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "deltanet",
            QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_GDN_LANES_CAPTURE_F32_KERNEL,
            [value_heads as u32, lanes.len() as u32, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(table);
                blob.push_i32(rv);
                blob.push_i32(qv);
                blob.push_i32(khv);
                blob.push_i32(vhv);
                blob.push_i32(kdv);
                blob.push_i32(vdv);
                blob.push_f32(scale);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }
}

/// Most batch rows one `gemm_bf16_mfma_gfx942_rows16` launch takes; a chain
/// launches it once per this many rows.
const ROWS16_CHAIN_STEP: usize = 16;

/// Most batch rows one `gemm_bf16_mfma_gfx942_wide` launch takes (the
/// `gemm_bf16_mfma_gfx942_wide64` entries: four 16-column MFMA chains).
// selector-receipt: receipts/selectors/gfx942_wide_max_rows.json
pub const WIDE_MAX_ROWS: usize = 64;

/// X rows staged in LDS per chunk (`WD_XROWS` in
/// `gemm_bf16_mfma_wide.gfx942.hip`; a unit test pins it to the source).
const WIDE_XROWS: usize = WIDE_MAX_ROWS;

/// The gfx942 workgroup LDS limit the wide entries' dynamic LDS must fit.
const WIDE_LDS_LIMIT: usize = 65536;

/// Wide-entry variants: `(kernel, weight rows per workgroup, K elements per
/// LDS chunk)`. The tile height is how many weight rows one workgroup streams
/// once for all of its ≤64 batch rows; a taller tile re-reads the activation
/// less per weight byte but launches fewer workgroups. The chunk is the
/// kernel's `KC`: the 64-row tile stages 128 rows (64 weight + 64 X) and its
/// double buffer at KC = 128 would be 67,584 B, so it takes KC = 64. A unit
/// test pins the table to the `WD_ENTRY` lines of the kernel source.
const WIDE_ENTRIES: [(&str, usize, usize); 3] = [
    ("gemm_bf16_mfma_gfx942_wide64", 16, 128),
    ("gemm_bf16_mfma_gfx942_wide64_m32", 32, 128),
    ("gemm_bf16_mfma_gfx942_wide64_m64", 64, 64),
];

/// Dynamic LDS bytes of a wide entry: a double buffer of `tile_rows` weight
/// rows plus [`WIDE_XROWS`] X rows, each `kc` K elements of BF16 plus the
/// 4-element pad (`LD = KC + 4`).
const fn wide_lds_bytes(tile_rows: usize, kc: usize) -> usize {
    2 * (tile_rows + WIDE_XROWS) * (kc + 4) * 2
}

/// Bind `func` of the wide module: the hipcc source, or, only when
/// `HIPFIRE_G942_BF16_WIDE_IMAGE=/absolute/device.elf` names one, that
/// explicitly supplied code object (the Route-A twin proof: same module, same
/// entry names, loaded through `ensure_embedded_kernel`). Unset: the source,
/// unchanged.
fn g942_wide_ensure(gpu: &mut Gpu, func: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_G942_BF16_WIDE_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                hip_bridge::HipError::new(
                    0,
                    &format!("HIPFIRE_G942_BF16_WIDE_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                GEMM_BF16_MFMA_WIDE_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                func,
            )
        }
        _ => gpu.ensure_kernel(GEMM_BF16_MFMA_WIDE_MODULE, GEMM_BF16_MFMA_WIDE_SRC, func),
    }
}

/// The most batch rows at which ⌈rows/16⌉ chained few-row launches of a BF16
/// `[m, k]` weight beat one 32-row-tile launch on MI300X (P-K table,
/// `mi300x/beat/P/P-K.md` (a), every Flash-Next trunk shape at 17 / 31 / 57 /
/// 145 / 456 / 912 rows). The tile entry has a ~120-130 µs floor on small `m`
/// and a chain launch costs ~6-20 µs, so the crossover is per shape:
/// - `k < 512` (hc_up 10240x320): chain/tile 0.88 at 57, 1.22 at 145.
/// - `k < 1024` (sh_down 2560x640): 0.78 at 145, 1.27 at 456.
/// - `m >= 2048` (qkv, z, out_proj, qsa_q, PLE, at k >= 1024): ≤ 0.56 up to
///   912 rows; the prefill chunk bound (1,536 rows) is the cap.
/// - `m < 64` (gdn_a/b, sh_egate, hc_inject): crossover ~210-310 rows.
/// - other `m` (router, qsa_k/v/idx, shared gate/up, hc_down): 0.51-0.54 at
///   145, crossover ~300 rows (hc_down, k = 10240: ~600).
pub fn rows16_chain_max_rows(m: usize, k: usize) -> usize {
    if k < 512 {
        48
    } else if k < 1024 {
        256
    } else if m >= 2048 {
        usize::MAX
    } else if m < 64 {
        192
    } else if k > 4096 {
        512
    } else {
        256
    }
}

/// The most batch rows at which ⌈rows/64⌉ launches of the wide entry beat both
/// the 32-row-tile entry and the rows16 chain for a BF16 `[m, k]` weight with
/// a BF16 activation on MI300X (P1b round 2, LDS-staged X and weights,
/// `mi300x/beat/P/p1b/bench/d7.log`: every Flash-Next trunk shape at 1 / 8 /
/// 16 / 17 / 32 / 48 / 57 / 64 / 145 / 257 / 456 / 1025 / 1536 rows; the route
/// stages an F32 activation to BF16 once past 16 rows).
/// - `m * k >= 2^20` (every shape but the three below: qkv, z, qsa_q, out_proj,
///   hc_down, hc_up, sh_down, qsa_k / idx, shared gate, PLE): the wide entry
///   is the fastest exact route at every measured row count from 17 through
///   the prefill chunk bound (1,536 rows): 0.20-0.85x of the chain from 17
///   rows up (0.21-0.52x at 64 and more), 0.05-0.83x of the tile entry.
/// - smaller weights (gdn a / b 48x2560, hc_inject 4x10240, shared expert gate
///   1x2560): the tile entry's ~85-340 µs launch is flat in `rows` while the
///   wide entry's cost grows with the row slices; wide beats the tile entry up
///   to ~590 rows (sh_egate), ~690 (hc_inject) and ~830 (gdn_a) (fine grid
///   `crossover.log`), still wins at 456 rows (0.07-0.85x of the tile) and
///   beats the chain 0.36-0.85x at 17..=456 rows, so it takes them up to 456.
pub fn wide_max_rows(m: usize, k: usize) -> usize {
    if m.saturating_mul(k) >= 1 << 20 {
        usize::MAX
    } else {
        456
    }
}

/// Whether the wide entry takes a launch: 1..=[`WIDE_MAX_ROWS`] batch rows,
/// `K % 8 == 0` and 16-byte aligned weight and activation bases (the
/// entry's 16-byte weight loads and 8/16-byte activation loads).
fn wide_admits(a: *const c_void, x: *const c_void, k: usize, batch: usize) -> bool {
    (1..=WIDE_MAX_ROWS).contains(&batch)
        && k % 8 == 0
        && (a as usize) % 16 == 0
        && (x as usize) % 16 == 0
}

/// Workgroups below which a wide launch under-fills MI300X (304 CUs): the
/// tallest tile that still launches this many workgroups is taken ([P1b]
/// measured: 64-row tiles at M = 10240 / 12288 (160 / 192 workgroups), 32-row
/// at M = 6144 (192), 16-row at M = 2560 (160) were each the fastest entry).
const WIDE_MIN_WORKGROUPS: usize = 144;

/// The wide entry for a launch whose members have the given weight rows `m`:
/// the tallest variant (`WIDE_ENTRIES`) that still launches at least
/// [`WIDE_MIN_WORKGROUPS`] workgroups, the shortest otherwise.
fn wide_entry(members: &[usize]) -> (&'static str, usize, usize) {
    WIDE_ENTRIES
        .iter()
        .rev()
        .find(|&&(_, tile, _)| {
            members.iter().map(|m| m.div_ceil(tile)).sum::<usize>() >= WIDE_MIN_WORKGROUPS
        })
        .copied()
        .unwrap_or(WIDE_ENTRIES[0])
}

impl Gpu {
    /// [`Gpu::gemm_bf16_mfma_gfx942_rows16`] for any `batch`: up to 16 rows
    /// it is that one launch; past 16 rows, with `wide` set and any
    /// member's `batch <= `[`wide_max_rows`] (a small member rides in the launch
    /// of a large one at no cost), it is
    /// [`Gpu::gemm_bf16_mfma_gfx942_wide_slices`] (one weight pass per 64
    /// rows); otherwise, with `chain` set and every member's `batch <=
    /// `[`rows16_chain_max_rows`], it launches the few-row entry once per
    /// 16-row slice of `x` and of each `y_j`, last slice partial. Each output
    /// element is the one ascending-K MFMA chain of its own activation row in
    /// every entry of both modules, so the bytes equal the 32-row-tile
    /// route's (P-K: 114/114 shape × rows cells, 0 differing words; P1b: see
    /// the wide entry). Returns `Ok(false)` without launching when the first
    /// slice declines or a member's shape keeps the tile entry; a later
    /// slice's refusal (an output overlapping that slice of `x`) also
    /// returns `Ok(false)` and the caller's tile route rewrites every row.
    pub fn gemm_bf16_mfma_gfx942_rows16_chain(
        &mut self,
        x: &GpuTensor,
        members: &[(&hip_bridge::DeviceBuffer, &GpuTensor, usize)],
        k: usize,
        batch: usize,
        chain: bool,
        wide: bool,
    ) -> HipResult<bool> {
        const MAX: usize = crate::gemm::GEMM_BF16_MFMA_ROWS16_MAX_MEMBERS;
        if batch <= ROWS16_CHAIN_STEP {
            return self.gemm_bf16_mfma_gfx942_rows16(x, members, k, batch);
        }
        if members.is_empty() || members.len() > MAX {
            return Ok(false);
        }
        if wide
            && members
                .iter()
                .any(|&(_, _, m)| batch <= wide_max_rows(m, k))
        {
            return self.gemm_bf16_mfma_gfx942_wide_slices(x, members, k, batch);
        }
        if !chain
            || members
                .iter()
                .any(|&(_, _, m)| batch > rows16_chain_max_rows(m, k))
        {
            return Ok(false);
        }
        let mut ys: [Option<GpuTensor>; MAX] = Default::default();
        for row in (0..batch).step_by(ROWS16_CHAIN_STEP) {
            let rows = (batch - row).min(ROWS16_CHAIN_STEP);
            let xs = x.sub_offset(row * k, rows * k);
            for (slot, &(_, y, m)) in ys.iter_mut().zip(members) {
                *slot = Some(y.sub_offset(row * m, rows * m));
            }
            let mut slice = [(members[0].0, x, 0usize); MAX];
            for (j, &(a, _, m)) in members.iter().enumerate() {
                slice[j] = (a, ys[j].as_ref().expect("slice view"), m);
            }
            if !self.gemm_bf16_mfma_gfx942_rows16(&xs, &slice[..members.len()], k, rows)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// [`Gpu::gemm_bf16_mfma_gfx942_wide`] for any `batch`: one launch per
    /// 64-row slice of `x` and of each `y_j`, last slice partial. No row
    /// threshold; the route's [`wide_max_rows`] gate is the caller's. A later
    /// slice's refusal returns `Ok(false)` as in the rows16 chain.
    pub fn gemm_bf16_mfma_gfx942_wide_slices(
        &mut self,
        x: &GpuTensor,
        members: &[(&hip_bridge::DeviceBuffer, &GpuTensor, usize)],
        k: usize,
        batch: usize,
    ) -> HipResult<bool> {
        const MAX: usize = crate::gemm::GEMM_BF16_MFMA_ROWS16_MAX_MEMBERS;
        if members.is_empty() || members.len() > MAX {
            return Ok(false);
        }
        let mut ys: [Option<GpuTensor>; MAX] = Default::default();
        for row in (0..batch).step_by(WIDE_MAX_ROWS) {
            let rows = (batch - row).min(WIDE_MAX_ROWS);
            let xs = x.sub_offset(row * k, rows * k);
            for (slot, &(_, y, m)) in ys.iter_mut().zip(members) {
                *slot = Some(y.sub_offset(row * m, rows * m));
            }
            let mut slice = [(members[0].0, x, 0usize); MAX];
            for (j, &(a, _, m)) in members.iter().enumerate() {
                slice[j] = (a, ys[j].as_ref().expect("slice view"), m);
            }
            if !self.gemm_bf16_mfma_gfx942_wide(&xs, &slice[..members.len()], k, rows)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The wide entry `gemm_bf16_mfma_gfx942_wide64*` for a projection group:
    /// up to four BF16 weights `[m_j, K]` that read the same `x` (F32 or BF16
    /// `[batch, K]`) as ONE launch of 1..=[`WIDE_MAX_ROWS`] batch rows, member
    /// j writing its `y_j` (F32 or BF16 `[batch, m_j]`). Every weight is read
    /// from HBM once for all `batch` rows (the 32-row tile entry re-reads it
    /// per 32 rows, the rows16 chain per 16). Same contract and same bytes as
    /// [`Gpu::gemm_bf16_mfma_gfx942_rows16`]: an F32 `x` is rounded to BF16
    /// in-kernel and a BF16 `y_j` is rounded from the F32 accumulator, both by
    /// `convert_f32_to_bf16`'s RNE rule, and each output element is the tile
    /// entry's ascending-K MFMA chain. Returns `Ok(false)` without launching
    /// when the entry does not apply (no member or more than four, `batch`
    /// outside 1..=64, `K % 8 != 0`, an operand not 16-byte aligned, a dtype
    /// outside F32 / BF16, or an output overlapping `x`).
    pub fn gemm_bf16_mfma_gfx942_wide(
        &mut self,
        x: &GpuTensor,
        members: &[(&hip_bridge::DeviceBuffer, &GpuTensor, usize)],
        k: usize,
        batch: usize,
    ) -> HipResult<bool> {
        const MAX: usize = crate::gemm::GEMM_BF16_MFMA_ROWS16_MAX_MEMBERS;
        let (x_f32, x_size) = match x.dtype {
            DType::F32 => (true, 4),
            DType::BF16 => (false, 2),
            _ => return Ok(false),
        };
        if members.is_empty() || members.len() > MAX {
            return Ok(false);
        }
        let x_start = x.buf.as_ptr() as usize;
        let x_end = x_start + batch * k * x_size;
        let mut a_ptr = [std::ptr::null_mut::<c_void>(); MAX];
        let mut d_ptr = [std::ptr::null_mut::<c_void>(); MAX];
        let mut m_i32 = [0i32; MAX];
        let mut d_bf16_mask = 0i32;
        let mut ms = [0usize; MAX];
        for (j, &(a, y, m)) in members.iter().enumerate() {
            // Every workgroup reads x for the whole launch: an output that
            // overlaps it keeps the staged route (which reads a staged copy).
            let y_start = y.buf.as_ptr() as usize;
            if y_start < x_end && x_start < y_start + batch * m * y.dtype.size() {
                return Ok(false);
            }
            let (d_bf16, d_size) = match y.dtype {
                DType::F32 => (false, 4),
                DType::BF16 => (true, 2),
                _ => return Ok(false),
            };
            if !wide_admits(a.as_ptr(), x.buf.as_ptr(), k, batch) {
                return Ok(false);
            }
            self.gemm_bf16_mfma_gfx942_check(a, &x.buf, x_size, &y.buf, d_size, m, k, batch)?;
            a_ptr[j] = a.as_ptr();
            d_ptr[j] = y.buf.as_ptr();
            m_i32[j] = m as i32;
            d_bf16_mask |= i32::from(d_bf16) << j;
            ms[j] = m;
        }
        let n = members.len();
        let (kernel, tile_rows, kc) = wide_entry(&ms[..n]);
        g942_wide_ensure(self, kernel)?;

        let tiles: usize = ms[..n].iter().map(|m| m.div_ceil(tile_rows)).sum();
        let bytes = members
            .iter()
            .map(|&(_, y, m)| m * k * 2 + batch * m * y.dtype.size())
            .sum::<usize>()
            + batch * k * x_size;
        let x_ptr = x.buf.as_ptr();
        let k_i32 = k as i32;
        let batch_i32 = batch as i32;
        let x_f32_i32 = i32::from(x_f32);
        let mut params: Vec<*mut c_void> = vec![&x_ptr as *const _ as *mut c_void];
        params.extend(a_ptr.iter().map(|p| p as *const _ as *mut c_void));
        params.extend(d_ptr.iter().map(|p| p as *const _ as *mut c_void));
        params.extend(m_i32.iter().map(|v| v as *const _ as *mut c_void));
        params.extend([
            &k_i32 as *const _ as *mut c_void,
            &batch_i32 as *const _ as *mut c_void,
            &x_f32_i32 as *const _ as *mut c_void,
            &d_bf16_mask as *const _ as *mut c_void,
        ]);
        let lds = wide_lds_bytes(tile_rows, kc);
        assert!(
            lds <= WIDE_LDS_LIMIT,
            "{kernel}: {lds} B of LDS exceeds the gfx942 workgroup limit"
        );
        let lds = lds as u32;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            [tiles as u32, 1, 1],
            [512, 1, 1],
            lds,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(x_ptr);
                for p in a_ptr.iter().chain(d_ptr.iter()) {
                    blob.push_ptr(*p);
                }
                for v in m_i32 {
                    blob.push_i32(v);
                }
                blob.push_i32(k_i32);
                blob.push_i32(batch_i32);
                blob.push_i32(x_f32_i32);
                blob.push_i32(d_bf16_mask);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(true)
    }
}

impl Gfx942Device<'_> {
    /// `HIPFIRE_QWEN4_G942_FOLD_MOE` (default off until the Route-A twin is sealed): whether the
    /// stock-FP8 MoE glue folds are selected for this device.
    pub fn g942_fold_moe_enabled(&self) -> bool {
        self.gpu.flags.qwen4_g942_fold_moe_enabled()
    }

    /// Whether [`Self::g942_fold_moe_group_scatter_desc`] covers this grouping: the fold is one
    /// workgroup, so it holds at most [`G942_FOLD_MOE_MAX_SLOTS`] flat slots and
    /// [`G942_FOLD_MOE_MAX_EXPERTS`] experts.
    pub fn g942_fold_moe_scatter_admitted(
        &self,
        total_slots: usize,
        experts: usize,
        grouped_rows: usize,
    ) -> bool {
        total_slots > 0
            && total_slots <= G942_FOLD_MOE_MAX_SLOTS
            && experts > 0
            && experts <= G942_FOLD_MOE_MAX_EXPERTS
            && grouped_rows % 16 == 0
            && grouped_rows >= total_slots
            && i32::try_from(grouped_rows).is_ok()
    }

    /// Stable pad16 grouping plus the FP8 tile descriptors in ONE launch: bitwise
    /// `Gpu::moe_scatter_stable_top10` (`qwen4_moe_group_ranks`, `_prefix`, `_scatter`) followed by
    /// [`Self::fp8b128_moe_tile_desc_pad16`] over the tile ids it wrote.
    ///
    /// ```text
    /// qwen4_g942_moe_group_scatter_desc(
    ///     topk:      i32 [slots]            routed expert per flat slot
    ///     counts:    i32 [experts]          pad16 counts
    ///     offsets:   i32 [experts + 1]      pad16 exclusive offsets
    ///     sorted:    i32 [grouped_rows]     flat slot per grouped row, -1 padding
    ///     tiles:     i32 [grouped_rows/16]  expert per tile, -1 tail
    ///     inverse:   i32 [slots]            grouped row per flat slot, -1 unrouted
    ///     tile_desc: [grouped_rows/16, 4] i32 `{tile_id, 16 t, 16, 0}`
    ///     slots, experts, grouped_rows)
    /// ```
    ///
    /// One workgroup of 1024 threads; the grouping scratch lives in LDS. Refused outside
    /// [`Self::g942_fold_moe_scatter_admitted`].
    #[allow(clippy::too_many_arguments)]
    pub fn g942_fold_moe_group_scatter_desc(
        &mut self,
        topk: &GpuTensor,
        counts: &GpuTensor,
        offsets: &GpuTensor,
        sorted: &GpuTensor,
        tiles: &GpuTensor,
        inverse: &GpuTensor,
        tile_desc: &GpuTensor,
        total_slots: usize,
        experts: usize,
        grouped_rows: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_group_scatter_desc";
        if !self.g942_fold_moe_scatter_admitted(total_slots, experts, grouped_rows) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "{WHAT}: need 0 < slots <= {G942_FOLD_MOE_MAX_SLOTS}, 0 < experts <= \
                     {G942_FOLD_MOE_MAX_EXPERTS}, grouped_rows >= slots and a multiple of 16 \
                     (slots={total_slots} experts={experts} grouped_rows={grouped_rows})"
                ),
            ));
        }
        let n_tiles = grouped_rows / 16;
        crate::moe::require_span(
            topk,
            &[total_slots, 4],
            "qwen4_g942_moe_group_scatter_desc topk",
        )?;
        crate::moe::require_span(
            counts,
            &[experts, 4],
            "qwen4_g942_moe_group_scatter_desc counts",
        )?;
        crate::moe::require_span(
            offsets,
            &[experts + 1, 4],
            "qwen4_g942_moe_group_scatter_desc offsets",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_g942_moe_group_scatter_desc sorted",
        )?;
        crate::moe::require_span(
            tiles,
            &[n_tiles, 4],
            "qwen4_g942_moe_group_scatter_desc tiles",
        )?;
        crate::moe::require_span(
            inverse,
            &[total_slots, 4],
            "qwen4_g942_moe_group_scatter_desc inverse",
        )?;
        crate::moe::require_span(
            tile_desc,
            &[n_tiles, 16],
            "qwen4_g942_moe_group_scatter_desc tile_desc",
        )?;
        self.gpu.bind_thread()?;
        g942_fold_ensure(&mut *self.gpu, QWEN4_G942_MOE_GROUP_SCATTER_DESC)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(
            &NOTE,
            "fold 1 (ranks + prefix + scatter + tile descriptors)",
        );
        let tp = topk.buf.as_ptr();
        let cp = counts.buf.as_ptr();
        let op = offsets.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let tip = tiles.buf.as_ptr();
        let ip = inverse.buf.as_ptr();
        let dp = tile_desc.buf.as_ptr();
        let (lv, ev, pv) = (total_slots as i32, experts as i32, grouped_rows as i32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tip as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
            &ev as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_GROUP_SCATTER_DESC,
            (3 * total_slots + 2 * experts + grouped_rows + 5 * n_tiles) * 4,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_GROUP_SCATTER_DESC,
            [1, 1, 1],
            [1024, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(tp);
                blob.push_ptr(cp);
                blob.push_ptr(op);
                blob.push_ptr(sp);
                blob.push_ptr(tip);
                blob.push_ptr(ip);
                blob.push_ptr(dp);
                blob.push_i32(lv);
                blob.push_i32(ev);
                blob.push_i32(pv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// The routed SwiGLU unscatter and the shared expert's activation in ONE launch: bitwise
    /// [`Self::fp8b128_moe_unscatter_silu`] followed by [`Self::fp8b128_moe_shared_activation`]
    /// (the two touch disjoint buffers).
    ///
    /// ```text
    /// qwen4_g942_moe_unscatter_act(
    ///     gate_up, sorted, activation, mi, n_sorted, slots      (fp8b128_moe_unscatter_silu)
    ///     shared_gate, shared_up, shared_act, selector, rows, shared_mi
    ///                                                           (fp8b128_moe_shared_activation)
    /// )
    /// ```
    ///
    /// Grid `[n_sorted + ceil(ceil(rows*shared_mi/256) / ceil(mi/256)), ceil(mi/256)]`, block 256.
    #[allow(clippy::too_many_arguments)]
    pub fn g942_fold_moe_unscatter_act(
        &mut self,
        gate_up: &GpuTensor,
        sorted: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        n_sorted: usize,
        slots: usize,
        shared_gate: &GpuTensor,
        shared_up: &GpuTensor,
        shared_act: &GpuTensor,
        selector: &GpuTensor,
        rows: usize,
        shared_mi: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_unscatter_act";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let max_dim = i32::MAX as usize;
        if mi > max_dim
            || n_sorted > max_dim
            || slots > max_dim
            || rows > max_dim
            || shared_mi > max_dim
        {
            return refuse(format!("{WHAT}: scalar dimension exceeds i32::MAX"));
        }
        let Some(shared_n) = rows.checked_mul(shared_mi) else {
            return refuse(format!("{WHAT}: rows*shared_mi overflows"));
        };
        let grid_y = mi.div_ceil(256);
        if grid_y == 0 || grid_y > u16::MAX as usize {
            return refuse(format!(
                "{WHAT}: mi={mi} needs grid.y={grid_y} (1..={})",
                u16::MAX
            ));
        }
        let act_blocks = shared_n.div_ceil(256);
        let grid_x = n_sorted + act_blocks.div_ceil(grid_y);
        if grid_x > i32::MAX as usize || act_blocks > i32::MAX as usize {
            return refuse(format!("{WHAT}: grid.x={grid_x} exceeds i32::MAX"));
        }
        for (name, t) in [
            ("gate_up", gate_up),
            ("activation", activation),
            ("shared_gate", shared_gate),
            ("shared_up", shared_up),
            ("shared_act", shared_act),
            ("selector", selector),
        ] {
            if t.dtype != DType::F32 {
                return refuse(format!("{WHAT}: {name} must be F32 (got {:?})", t.dtype));
            }
        }
        // `sorted` is i32 data; the sealed route carries the stable scatter's index scratch as
        // `DType::Raw`, so accept a 4-byte dtype or Raw (the byte-span check bounds the read).
        if !(sorted.dtype.size() == 4 || sorted.dtype == DType::Raw) {
            return refuse(format!(
                "{WHAT}: sorted needs a 4-byte dtype (got {:?})",
                sorted.dtype
            ));
        }
        crate::moe::require_span(
            gate_up,
            &[n_sorted, 2, mi, 4],
            "qwen4_g942_moe_unscatter_act gate_up",
        )?;
        crate::moe::require_span(
            sorted,
            &[n_sorted, 4],
            "qwen4_g942_moe_unscatter_act sorted",
        )?;
        crate::moe::require_span(
            activation,
            &[slots, mi, 4],
            "qwen4_g942_moe_unscatter_act activation",
        )?;
        crate::moe::require_span(
            shared_gate,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_unscatter_act shared gate",
        )?;
        crate::moe::require_span(
            shared_up,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_unscatter_act shared up",
        )?;
        crate::moe::require_span(
            shared_act,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_unscatter_act shared activation",
        )?;
        crate::moe::require_span(
            selector,
            &[rows, 4],
            "qwen4_g942_moe_unscatter_act selector",
        )?;
        let (yp, sp, ap) = (
            gate_up.buf.as_ptr(),
            sorted.buf.as_ptr(),
            activation.buf.as_ptr(),
        );
        let (gp, up_p, hp, lp) = (
            shared_gate.buf.as_ptr(),
            shared_up.buf.as_ptr(),
            shared_act.buf.as_ptr(),
            selector.buf.as_ptr(),
        );
        let all = [yp, sp, ap, gp, up_p, hp, lp];
        if all.iter().enumerate().any(|(i, a)| all[..i].contains(a)) {
            return refuse(format!(
                "{WHAT}: gate_up, sorted, activation and the four shared buffers must be distinct"
            ));
        }
        if mi == 0 || n_sorted == 0 || slots == 0 {
            return refuse(format!(
                "{WHAT}: empty routed extent (mi={mi} n_sorted={n_sorted} slots={slots})"
            ));
        }
        self.gpu.bind_thread()?;
        g942_fold_ensure(&mut *self.gpu, QWEN4_G942_MOE_UNSCATTER_ACT)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "fold 2 (unscatter-silu + shared activation)");
        let (miv, nv, sv) = (mi as i32, n_sorted as i32, slots as i32);
        let (rv, smv, abv) = (rows as i32, shared_mi as i32, act_blocks as i32);
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &miv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &smv as *const _ as *mut c_void,
            &abv as *const _ as *mut c_void,
        ];
        let bytes = n_sorted
            .saturating_mul(mi)
            .saturating_mul(12)
            .saturating_add(shared_n.saturating_mul(12));
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_UNSCATTER_ACT,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_UNSCATTER_ACT,
            [grid_x as u32, grid_y as u32, 1],
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
                blob.push_ptr(gp);
                blob.push_ptr(up_p);
                blob.push_ptr(hp);
                blob.push_ptr(lp);
                blob.push_i32(rv);
                blob.push_i32(smv);
                blob.push_i32(abv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// `Gpu::moe_down_combine_grouped_top10` with the shared expert's `bf16_scaled_add_batched`
    /// as its epilogue, in ONE launch: bitwise the combine into `residual` followed by
    /// `residual += sigmoid-selector * shared_out` (`bf16_scaled_add_batched_f32`).
    ///
    /// `shared_out` is the shared down's F32 rows `[tokens, hidden]`, `selector` the in-place
    /// selector the shared activation left (`[tokens]`). Grid `[ceil(hidden/256), tokens]`.
    #[allow(clippy::too_many_arguments)]
    pub fn g942_fold_moe_combine_shared_add(
        &mut self,
        grouped_down: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        shared_out: &GpuTensor,
        selector: &GpuTensor,
        hidden: usize,
        grouped_rows: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_combine_shared_add";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        if tokens == 0
            || hidden == 0
            || grouped_rows > i32::MAX as usize
            || hidden > i32::MAX as usize
            || tokens > u16::MAX as usize
            || n_experts > i32::MAX as usize
        {
            return refuse(format!(
                "{WHAT}: bad extent (hidden={hidden} grouped_rows={grouped_rows} tokens={tokens})"
            ));
        }
        for (name, t) in [
            ("grouped_down", grouped_down),
            ("residual", residual),
            ("shared_out", shared_out),
            ("selector", selector),
            ("topk_weights", topk_weights),
        ] {
            if t.dtype != DType::F32 {
                return refuse(format!("{WHAT}: {name} must be F32 (got {:?})", t.dtype));
            }
        }
        crate::moe::require_span(
            grouped_down,
            &[grouped_rows, hidden, 4],
            "qwen4_g942_moe_combine_shared_add grouped rows",
        )?;
        crate::moe::require_span(
            inverse_perm,
            &[tokens, 10, 4],
            "qwen4_g942_moe_combine_shared_add inverse permutation",
        )?;
        crate::moe::require_span(
            topk_indices,
            &[tokens, 10, 4],
            "qwen4_g942_moe_combine_shared_add topk indices",
        )?;
        crate::moe::require_span(
            topk_weights,
            &[tokens, 10, 4],
            "qwen4_g942_moe_combine_shared_add topk weights",
        )?;
        crate::moe::require_span(
            residual,
            &[tokens, hidden, 4],
            "qwen4_g942_moe_combine_shared_add target",
        )?;
        crate::moe::require_span(
            shared_out,
            &[tokens, hidden, 4],
            "qwen4_g942_moe_combine_shared_add shared rows",
        )?;
        crate::moe::require_span(
            selector,
            &[tokens, 4],
            "qwen4_g942_moe_combine_shared_add selector",
        )?;
        let (gp, ip, tp, wp, rp, hp, sp) = (
            grouped_down.buf.as_ptr(),
            inverse_perm.buf.as_ptr(),
            topk_indices.buf.as_ptr(),
            topk_weights.buf.as_ptr(),
            residual.buf.as_ptr(),
            shared_out.buf.as_ptr(),
            selector.buf.as_ptr(),
        );
        // `residual` is read and written in place; nothing else may alias it (the kernel reads
        // `shared_out` and `selector` per element after the combine).
        if [gp, ip, tp, wp, hp, sp].contains(&rp) {
            return refuse(format!("{WHAT}: the target must not alias another operand"));
        }
        self.gpu.bind_thread()?;
        g942_fold_ensure(&mut *self.gpu, QWEN4_G942_MOE_COMBINE_SHARED_ADD)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "fold 3 (combine + shared add)");
        let (hv, gr, tv, nx) = (
            hidden as i32,
            grouped_rows as i32,
            tokens as i32,
            n_experts as i32,
        );
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
        ];
        let grid_x = (hidden as u32).div_ceil(256);
        let bytes = (grouped_rows * hidden + 3 * tokens * 10 + 3 * tokens * hidden) * 4;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_COMBINE_SHARED_ADD,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_COMBINE_SHARED_ADD,
            [grid_x, tokens as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gp);
                blob.push_ptr(ip);
                blob.push_ptr(tp);
                blob.push_ptr(wp);
                blob.push_ptr(rp);
                blob.push_i32(hv);
                blob.push_i32(gr);
                blob.push_i32(tv);
                blob.push_i32(nx);
                blob.push_ptr(hp);
                blob.push_ptr(sp);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// Whether the wide router serves a call of `rows` routed tokens at this `n_exp` / `k_top`:
    /// the C1 glue folds are on and the slot count lies in
    /// [`G942_MOE_WIDE_MIN_SLOTS`]`..=`[`G942_MOE_WIDE_MAX_SLOTS`] (the wide scatter's capacity,
    /// so a call takes the whole wide chain or none of it). The caller adds the BF16-recipe
    /// precondition of [`Self::g942_moe_wide_router`].
    pub fn g942_moe_wide_router_admitted(&self, rows: usize, n_exp: usize, k_top: usize) -> bool {
        let Some(slots) = rows.checked_mul(k_top) else {
            return false;
        };
        self.g942_fold_moe_enabled()
            && k_top == 10
            && n_exp == 512
            && (G942_MOE_WIDE_MIN_SLOTS..=G942_MOE_WIDE_MAX_SLOTS).contains(&slots)
    }

    /// Whether [`Self::g942_moe_wide_group_scatter_desc`] covers this grouping.
    pub fn g942_moe_wide_scatter_admitted(
        &self,
        total_slots: usize,
        experts: usize,
        grouped_rows: usize,
    ) -> bool {
        self.g942_fold_moe_enabled()
            && (G942_MOE_WIDE_MIN_SLOTS..=G942_MOE_WIDE_MAX_SLOTS).contains(&total_slots)
            && experts > 0
            && experts <= G942_MOE_WIDE_MAX_EXPERTS
            && grouped_rows % 16 == 0
            && grouped_rows >= total_slots
            && i32::try_from(grouped_rows).is_ok()
    }

    /// Whether the wide unscatter and combine serve a call of `slots` routed slots: both are
    /// bitwise the fold kernels at every size, so only the lane-width floor applies.
    pub fn g942_moe_wide_glue_admitted(&self, slots: usize) -> bool {
        self.g942_fold_moe_enabled() && slots >= G942_MOE_WIDE_MIN_SLOTS
    }

    /// `Gpu::moe_router_softmax_top10_f32` for the BF16 recipe (`round_logits`), wave64 per
    /// token: bitwise the incumbent's ids, weights and fault record.
    ///
    /// ```text
    /// qwen4_g942_moe_wide_router(
    ///     logits: f32 [tokens, 512], topk_idx: i32 [tokens, 10], topk_w: f32 [tokens, 10],
    ///     n_exp = 512, top_k = 10, normalize, round_logits = 1, tokens, fault, layer, row_base)
    /// ```
    ///
    /// Grid `[ceil(tokens / 4)]`, block 256 (four tokens, one wave each).
    #[allow(clippy::too_many_arguments)]
    pub fn g942_moe_wide_router(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        tokens: usize,
        normalize_topk_prob: bool,
        layer: usize,
        row_base: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_wide_router";
        if !self.g942_moe_wide_router_admitted(tokens, 512, 10) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "{WHAT}: need {G942_MOE_WIDE_MIN_SLOTS} <= 10 * tokens <= \
                     {G942_MOE_WIDE_MAX_SLOTS} (tokens={tokens})"
                ),
            ));
        }
        crate::moe::require_span(logits, &[tokens, 512, 4], "qwen4_g942_moe_wide_router logits")?;
        crate::moe::require_span(topk_idx, &[tokens, 10, 4], "qwen4_g942_moe_wide_router ids")?;
        crate::moe::require_span(topk_w, &[tokens, 10, 4], "qwen4_g942_moe_wide_router weights")?;
        self.gpu.bind_thread()?;
        g942_moe_wide_ensure(&mut *self.gpu, QWEN4_G942_MOE_WIDE_ROUTER)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "wide router (wave64 per token)");
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let (ne, kt) = (512i32, 10i32);
        let norm = i32::from(normalize_topk_prob);
        let round = 1i32;
        let tv = tokens as i32;
        let fault = self.gpu.router_fault_ptr();
        let (layer_arg, row_base_arg) = (layer as i32, row_base as i32);
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &kt as *const _ as *mut c_void,
            &norm as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &fault as *const _ as *mut c_void,
            &layer_arg as *const _ as *mut c_void,
            &row_base_arg as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_WIDE_ROUTER,
            (tokens * 512 + tokens * 10 * 2) * 4,
        );
        let grid = tokens.div_ceil(G942_MOE_WIDE_ROUTER_TOKENS_PER_BLOCK) as u32;
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_WIDE_ROUTER,
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(lp);
                blob.push_ptr(ip);
                blob.push_ptr(wp);
                blob.push_i32(ne);
                blob.push_i32(kt);
                blob.push_i32(norm);
                blob.push_i32(round);
                blob.push_i32(tv);
                blob.push_ptr(fault as *const c_void);
                blob.push_i32(layer_arg);
                blob.push_i32(row_base_arg);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// [`Self::g942_fold_moe_group_scatter_desc`], same arguments and outputs bit for bit, as the
    /// wide module's mask-and-popcount kernel (`qwen4_g942_moe_wide_group_scatter_desc`, one
    /// 1024-thread workgroup). Refused outside [`Self::g942_moe_wide_scatter_admitted`].
    #[allow(clippy::too_many_arguments)]
    pub fn g942_moe_wide_group_scatter_desc(
        &mut self,
        topk: &GpuTensor,
        counts: &GpuTensor,
        offsets: &GpuTensor,
        sorted: &GpuTensor,
        tiles: &GpuTensor,
        inverse: &GpuTensor,
        tile_desc: &GpuTensor,
        total_slots: usize,
        experts: usize,
        grouped_rows: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_wide_group_scatter_desc";
        if !self.g942_moe_wide_scatter_admitted(total_slots, experts, grouped_rows) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "{WHAT}: need {G942_MOE_WIDE_MIN_SLOTS} <= slots <= {G942_MOE_WIDE_MAX_SLOTS}, \
                     0 < experts <= {G942_MOE_WIDE_MAX_EXPERTS}, grouped_rows >= slots and a \
                     multiple of 16 (slots={total_slots} experts={experts} \
                     grouped_rows={grouped_rows})"
                ),
            ));
        }
        let n_tiles = grouped_rows / 16;
        crate::moe::require_span(topk, &[total_slots, 4], "qwen4_g942_moe_wide_group_scatter_desc topk")?;
        crate::moe::require_span(counts, &[experts, 4], "qwen4_g942_moe_wide_group_scatter_desc counts")?;
        crate::moe::require_span(
            offsets,
            &[experts + 1, 4],
            "qwen4_g942_moe_wide_group_scatter_desc offsets",
        )?;
        crate::moe::require_span(sorted, &[grouped_rows, 4], "qwen4_g942_moe_wide_group_scatter_desc sorted")?;
        crate::moe::require_span(tiles, &[n_tiles, 4], "qwen4_g942_moe_wide_group_scatter_desc tiles")?;
        crate::moe::require_span(
            inverse,
            &[total_slots, 4],
            "qwen4_g942_moe_wide_group_scatter_desc inverse",
        )?;
        crate::moe::require_span(
            tile_desc,
            &[n_tiles, 16],
            "qwen4_g942_moe_wide_group_scatter_desc tile_desc",
        )?;
        self.gpu.bind_thread()?;
        g942_moe_wide_ensure(&mut *self.gpu, QWEN4_G942_MOE_WIDE_SCATTER_DESC)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "wide scatter (mask + popcount grouping + tile descriptors)");
        let tp = topk.buf.as_ptr();
        let cp = counts.buf.as_ptr();
        let op = offsets.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let tip = tiles.buf.as_ptr();
        let ip = inverse.buf.as_ptr();
        let dp = tile_desc.buf.as_ptr();
        let (lv, ev, pv) = (total_slots as i32, experts as i32, grouped_rows as i32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tip as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
            &ev as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_WIDE_SCATTER_DESC,
            (3 * total_slots + 2 * experts + grouped_rows + 5 * n_tiles) * 4,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_WIDE_SCATTER_DESC,
            [1, 1, 1],
            [1024, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(tp);
                blob.push_ptr(cp);
                blob.push_ptr(op);
                blob.push_ptr(sp);
                blob.push_ptr(tip);
                blob.push_ptr(ip);
                blob.push_ptr(dp);
                blob.push_i32(lv);
                blob.push_i32(ev);
                blob.push_i32(pv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// [`Self::g942_fold_moe_unscatter_act`] over the flat routed slots: the routed activation of
    /// slot `s` is read from grouped row `inverse[s]` (unrouted slots, `-1`, write nothing), so the
    /// grid is `slots` routed blocks instead of one per padded grouped row; the shared
    /// activation part is unchanged. Bitwise the fold kernel's `activation`, `shared_act` and
    /// `selector`.
    ///
    /// Grid `[slots + ceil(ceil(rows*shared_mi/256) / ceil(mi/256)), ceil(mi/256)]`, block 256.
    #[allow(clippy::too_many_arguments)]
    pub fn g942_moe_wide_unscatter_act(
        &mut self,
        gate_up: &GpuTensor,
        inverse: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        n_sorted: usize,
        slots: usize,
        shared_gate: &GpuTensor,
        shared_up: &GpuTensor,
        shared_act: &GpuTensor,
        selector: &GpuTensor,
        rows: usize,
        shared_mi: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_wide_unscatter_act";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        let max_dim = i32::MAX as usize;
        if mi > max_dim
            || n_sorted > max_dim
            || slots > max_dim
            || rows > max_dim
            || shared_mi > max_dim
        {
            return refuse(format!("{WHAT}: scalar dimension exceeds i32::MAX"));
        }
        let Some(shared_n) = rows.checked_mul(shared_mi) else {
            return refuse(format!("{WHAT}: rows*shared_mi overflows"));
        };
        let grid_y = mi.div_ceil(256);
        if grid_y == 0 || grid_y > u16::MAX as usize {
            return refuse(format!(
                "{WHAT}: mi={mi} needs grid.y={grid_y} (1..={})",
                u16::MAX
            ));
        }
        let act_blocks = shared_n.div_ceil(256);
        let grid_x = slots + act_blocks.div_ceil(grid_y);
        if grid_x > i32::MAX as usize || act_blocks > i32::MAX as usize {
            return refuse(format!("{WHAT}: grid.x={grid_x} exceeds i32::MAX"));
        }
        for (name, t) in [
            ("gate_up", gate_up),
            ("activation", activation),
            ("shared_gate", shared_gate),
            ("shared_up", shared_up),
            ("shared_act", shared_act),
            ("selector", selector),
        ] {
            if t.dtype != DType::F32 {
                return refuse(format!("{WHAT}: {name} must be F32 (got {:?})", t.dtype));
            }
        }
        if !(inverse.dtype.size() == 4 || inverse.dtype == DType::Raw) {
            return refuse(format!(
                "{WHAT}: inverse needs a 4-byte dtype (got {:?})",
                inverse.dtype
            ));
        }
        crate::moe::require_span(gate_up, &[n_sorted, 2, mi, 4], "qwen4_g942_moe_wide_unscatter_act gate_up")?;
        crate::moe::require_span(inverse, &[slots, 4], "qwen4_g942_moe_wide_unscatter_act inverse")?;
        crate::moe::require_span(
            activation,
            &[slots, mi, 4],
            "qwen4_g942_moe_wide_unscatter_act activation",
        )?;
        crate::moe::require_span(
            shared_gate,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_wide_unscatter_act shared gate",
        )?;
        crate::moe::require_span(
            shared_up,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_wide_unscatter_act shared up",
        )?;
        crate::moe::require_span(
            shared_act,
            &[rows, shared_mi, 4],
            "qwen4_g942_moe_wide_unscatter_act shared activation",
        )?;
        crate::moe::require_span(selector, &[rows, 4], "qwen4_g942_moe_wide_unscatter_act selector")?;
        let (yp, ip, ap) = (
            gate_up.buf.as_ptr(),
            inverse.buf.as_ptr(),
            activation.buf.as_ptr(),
        );
        let (gp, up_p, hp, lp) = (
            shared_gate.buf.as_ptr(),
            shared_up.buf.as_ptr(),
            shared_act.buf.as_ptr(),
            selector.buf.as_ptr(),
        );
        let all = [yp, ip, ap, gp, up_p, hp, lp];
        if all.iter().enumerate().any(|(i, a)| all[..i].contains(a)) {
            return refuse(format!(
                "{WHAT}: gate_up, inverse, activation and the four shared buffers must be distinct"
            ));
        }
        if mi == 0 || n_sorted == 0 || slots == 0 {
            return refuse(format!(
                "{WHAT}: empty routed extent (mi={mi} n_sorted={n_sorted} slots={slots})"
            ));
        }
        self.gpu.bind_thread()?;
        g942_moe_wide_ensure(&mut *self.gpu, QWEN4_G942_MOE_WIDE_UNSCATTER_ACT)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "wide unscatter (slot-major unscatter-silu + shared activation)");
        let (miv, nv, sv) = (mi as i32, n_sorted as i32, slots as i32);
        let (rv, smv, abv) = (rows as i32, shared_mi as i32, act_blocks as i32);
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &miv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up_p as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
            &smv as *const _ as *mut c_void,
            &abv as *const _ as *mut c_void,
        ];
        let bytes = slots
            .saturating_mul(mi)
            .saturating_mul(12)
            .saturating_add(shared_n.saturating_mul(12));
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_WIDE_UNSCATTER_ACT,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_WIDE_UNSCATTER_ACT,
            [grid_x as u32, grid_y as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(yp);
                blob.push_ptr(ip);
                blob.push_ptr(ap);
                blob.push_i32(miv);
                blob.push_i32(nv);
                blob.push_i32(sv);
                blob.push_ptr(gp);
                blob.push_ptr(up_p);
                blob.push_ptr(hp);
                blob.push_ptr(lp);
                blob.push_i32(rv);
                blob.push_i32(smv);
                blob.push_i32(abv);
                blob
            },
        );
        if let Some(timer) = timer {
            timer.finish(&self.gpu.hip);
        }
        result
    }

    /// [`Self::g942_fold_moe_combine_shared_add`], same arguments and output bit for bit, with
    /// the ten rank gathers issued as two independent load batches
    /// (`qwen4_g942_moe_wide_combine_shared_add`). Grid `[ceil(hidden/256), tokens]`.
    #[allow(clippy::too_many_arguments)]
    pub fn g942_moe_wide_combine_shared_add(
        &mut self,
        grouped_down: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        shared_out: &GpuTensor,
        selector: &GpuTensor,
        hidden: usize,
        grouped_rows: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_g942_moe_wide_combine_shared_add";
        let refuse = |msg: String| Err(hip_bridge::HipError::new(0, &msg));
        if tokens == 0
            || hidden == 0
            || grouped_rows > i32::MAX as usize
            || hidden > i32::MAX as usize
            || tokens > u16::MAX as usize
            || n_experts > i32::MAX as usize
        {
            return refuse(format!(
                "{WHAT}: bad extent (hidden={hidden} grouped_rows={grouped_rows} tokens={tokens})"
            ));
        }
        for (name, t) in [
            ("grouped_down", grouped_down),
            ("residual", residual),
            ("shared_out", shared_out),
            ("selector", selector),
            ("topk_weights", topk_weights),
        ] {
            if t.dtype != DType::F32 {
                return refuse(format!("{WHAT}: {name} must be F32 (got {:?})", t.dtype));
            }
        }
        crate::moe::require_span(
            grouped_down,
            &[grouped_rows, hidden, 4],
            "qwen4_g942_moe_wide_combine_shared_add grouped rows",
        )?;
        crate::moe::require_span(
            inverse_perm,
            &[tokens, 10, 4],
            "qwen4_g942_moe_wide_combine_shared_add inverse permutation",
        )?;
        crate::moe::require_span(
            topk_indices,
            &[tokens, 10, 4],
            "qwen4_g942_moe_wide_combine_shared_add topk indices",
        )?;
        crate::moe::require_span(
            topk_weights,
            &[tokens, 10, 4],
            "qwen4_g942_moe_wide_combine_shared_add topk weights",
        )?;
        crate::moe::require_span(
            residual,
            &[tokens, hidden, 4],
            "qwen4_g942_moe_wide_combine_shared_add target",
        )?;
        crate::moe::require_span(
            shared_out,
            &[tokens, hidden, 4],
            "qwen4_g942_moe_wide_combine_shared_add shared rows",
        )?;
        crate::moe::require_span(selector, &[tokens, 4], "qwen4_g942_moe_wide_combine_shared_add selector")?;
        let (gp, ip, tp, wp, rp, hp, sp) = (
            grouped_down.buf.as_ptr(),
            inverse_perm.buf.as_ptr(),
            topk_indices.buf.as_ptr(),
            topk_weights.buf.as_ptr(),
            residual.buf.as_ptr(),
            shared_out.buf.as_ptr(),
            selector.buf.as_ptr(),
        );
        // `residual` is read and written in place; nothing else may alias it.
        if [gp, ip, tp, wp, hp, sp].contains(&rp) {
            return refuse(format!("{WHAT}: the target must not alias another operand"));
        }
        self.gpu.bind_thread()?;
        g942_moe_wide_ensure(&mut *self.gpu, QWEN4_G942_MOE_WIDE_COMBINE_SHARED_ADD)?;
        static NOTE: std::sync::Once = std::sync::Once::new();
        g942_fold_note(&NOTE, "wide combine (batched rank gathers + shared add)");
        let (hv, gr, tv, nx) = (
            hidden as i32,
            grouped_rows as i32,
            tokens as i32,
            n_experts as i32,
        );
        let mut params: Vec<*mut c_void> = vec![
            &gp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
        ];
        let grid_x = (hidden as u32).div_ceil(256);
        let bytes = (grouped_rows * hidden + 3 * tokens * 10 + 3 * tokens * hidden) * 4;
        let timer = crate::profile::begin_timer(
            &self.gpu.hip,
            "elementwise",
            QWEN4_G942_MOE_WIDE_COMBINE_SHARED_ADD,
            bytes,
        );
        let result = self.gpu.launch_maybe_blob(
            QWEN4_G942_MOE_WIDE_COMBINE_SHARED_ADD,
            [grid_x, tokens as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gp);
                blob.push_ptr(ip);
                blob.push_ptr(tp);
                blob.push_ptr(wp);
                blob.push_ptr(rp);
                blob.push_i32(hv);
                blob.push_i32(gr);
                blob.push_i32(tv);
                blob.push_i32(nx);
                blob.push_ptr(hp);
                blob.push_ptr(sp);
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

    /// The wide MoE glue's launcher constants are the kernel source's own: capacity, router
    /// tokens per workgroup and the four entry names the module exports.
    #[test]
    fn g942_moe_wide_launcher_matches_kernel_source() {
        let define = |name: &str| -> usize {
            super::QWEN4_G942_MOE_WIDE_SRC
                .lines()
                .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or_else(|| panic!("{name} define in the wide MoE glue source"))
        };
        assert_eq!(define("MW_MAXL"), super::G942_MOE_WIDE_MAX_SLOTS);
        assert_eq!(define("MW_MAXE"), super::G942_MOE_WIDE_MAX_EXPERTS);
        assert_eq!(
            define("MW_ROUTER_WAVES"),
            super::G942_MOE_WIDE_ROUTER_TOKENS_PER_BLOCK
        );
        assert_eq!(define("MW_THREADS"), 1024);
        for entry in super::QWEN4_G942_MOE_WIDE_ENTRIES {
            assert!(
                super::QWEN4_G942_MOE_WIDE_SRC.contains(&format!("void {entry}(")),
                "{entry} is not an entry of the wide MoE glue source"
            );
        }
        // The wide scatter's capacity is a whole number of 32-slot mask words per expert.
        assert_eq!(super::G942_MOE_WIDE_MAX_SLOTS % 32, 0);
        assert!(super::G942_MOE_WIDE_MIN_SLOTS <= super::G942_MOE_WIDE_MAX_SLOTS);
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

    #[test]
    fn wide_constants_match_kernel_source() {
        let src = super::GEMM_BF16_MFMA_WIDE_SRC;
        let define = |name: &str| {
            src.lines()
                .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
                .and_then(|v| v.split("//").next())
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or_else(|| panic!("#define {name} <integer> in the wide kernel source"))
        };
        assert_eq!(super::WIDE_XROWS, define("WD_XROWS"));
        assert_eq!(super::WIDE_MAX_ROWS, define("WD_XROWS"));
        // Each entry is `WD_ENTRY(name, MS, KC, min_blocks)`: MS 16-row groups
        // per workgroup, KC K elements per LDS chunk.
        for (kernel, tile_rows, kc) in super::WIDE_ENTRIES {
            let args: Vec<usize> = src
                .lines()
                .find_map(|l| l.strip_prefix(&format!("WD_ENTRY({kernel}, ")))
                .map(|r| {
                    r.trim_end_matches(')')
                        .split(',')
                        .map(|v| v.trim().parse().expect("integer WD_ENTRY argument"))
                        .collect()
                })
                .unwrap_or_else(|| panic!("WD_ENTRY({kernel}, ..) in the wide kernel source"));
            assert_eq!(tile_rows, 16 * args[0], "{kernel}");
            assert_eq!(kc, args[1], "{kernel}");
            assert!(
                super::wide_lds_bytes(tile_rows, kc) <= super::WIDE_LDS_LIMIT,
                "{kernel}: {} B of LDS",
                super::wide_lds_bytes(tile_rows, kc)
            );
        }
    }

    #[test]
    fn wide_lds_sizes_fit_the_workgroup() {
        use super::wide_lds_bytes as l;
        assert_eq!(l(16, 128), 42_240);
        assert_eq!(l(32, 128), 50_688);
        assert_eq!(l(64, 64), 34_816);
        // The 64-row tile does not fit as KC = 128 (the reason it takes KC = 64).
        assert!(l(64, 128) > super::WIDE_LDS_LIMIT);
    }

    #[test]
    fn wide_thresholds_follow_the_p1b_table() {
        use super::wide_max_rows as t;
        // m * k >= 2^20: the wide entry is the fastest exact route through the
        // prefill chunk bound (1,536 rows), every Flash-Next trunk shape.
        for (m, k) in [
            (10240, 2560),
            (6144, 2560),
            (12288, 2560),
            (10240, 320),
            (2560, 640),
            (2560, 6144),
            (320, 10240),
            (512, 2560),
            (640, 2560),
            (2560, 2560),
        ] {
            assert!(
                t(m, k) >= 1536,
                "{m}x{k} takes the wide entry through the prefill chunk"
            );
        }
        // Small weights: the tile entry's flat ~85-135 us launch wins past
        // ~480-730 rows; wide is admitted through the measured 456.
        for (m, k) in [(48, 2560), (4, 10240), (1, 2560)] {
            assert_eq!(t(m, k), 456, "{m}x{k} takes the wide entry up to 456 rows");
        }
    }

    #[test]
    fn wide_entry_takes_the_tallest_tile_that_fills_the_device() {
        use super::wide_entry as e;
        // (member rows, expected kernel tile rows): 160 / 192 / 192 / 160
        // workgroups at the tallest tile with >= 144 workgroups.
        for (ms, tile) in [
            (&[10240usize][..], 64),
            (&[12288][..], 64),
            (&[6144][..], 32),
            (&[2560][..], 16),
            (&[48][..], 16),
            (&[10240, 6144, 48, 48][..], 64),
        ] {
            assert_eq!(e(ms).1, tile, "{ms:?}");
        }
    }

    #[test]
    fn rows16_chain_thresholds_follow_the_p_k_table() {
        use super::rows16_chain_max_rows as t;
        // (m, k, rows where chain/tile < 1 was measured, rows where it lost)
        for (m, k, win, lose) in [
            (10240, 320, 31, 145),
            (2560, 640, 145, 456),
            (48, 2560, 145, 456),
            (1, 2560, 145, 456),
            (4, 10240, 145, 456),
            (512, 2560, 145, 456),
            (640, 2560, 145, 456),
            (320, 10240, 456, 912),
        ] {
            assert!(win <= t(m, k), "{m}x{k} must chain at {win}");
            assert!(lose > t(m, k), "{m}x{k} must keep the tile at {lose}");
        }
        for (m, k) in [
            (10240, 2560),
            (6144, 2560),
            (2560, 6144),
            (12288, 2560),
            (2560, 2560),
        ] {
            assert!(t(m, k) >= 1536, "{m}x{k} chains through the prefill chunk");
        }
    }

    /// The lanes kernel's pipe128 body is `tensor_ops.hip`'s, byte for byte: every block between
    /// the `BEGIN` / `END verbatim tensor_ops.hip` markers must occur in the production source.
    #[test]
    fn gdn_lanes_f32_gfx942_source_is_tensor_ops_verbatim() {
        const BEGIN: &str = "// BEGIN verbatim tensor_ops.hip\n";
        const END: &str = "// END verbatim tensor_ops.hip";
        let src = super::QWEN4_GDN_LANES_F32_SRC;
        let production = crate::tensor_ops::TENSOR_OPS_SRC;
        let mut blocks = 0;
        let mut rest = src;
        while let Some(at) = rest.find(BEGIN) {
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_gdn_lanes_f32.gfx942.hip is no longer a \
                 substring of tensor_ops.hip: mirror the production edit into the lanes kernel\n\
                 block starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        assert_eq!(blocks, 4, "bf16 helpers, Q8 helpers, barrier, pipe128 helpers + body");
        // The one-row production kernel is the body this wrapper batches.
        assert!(production.contains(
            "gated_delta_step_pipe128_body<false>(projection, gate, beta, state, output, rows,"
        ));
        assert_eq!(src.matches("gated_delta_step_pipe128_body<false>(").count(), 1);
        assert!(src.contains(&format!("void {}(", super::QWEN4_GDN_LANES_F32_KERNEL)));
        assert_eq!(src.matches("__launch_bounds__(128)").count(), 1);
    }

    /// The lane table entry is five 8-byte pointers on both sides of the launch.
    #[test]
    fn gdn_lanes_f32_gfx942_lane_desc_matches_kernel_struct() {
        assert_eq!(std::mem::size_of::<super::Qwen4GdnLaneDesc>(), 40);
        let src = super::QWEN4_GDN_LANES_F32_SRC;
        let at = src
            .find("struct Qwen4GdnLaneDesc {")
            .expect("kernel lane descriptor");
        let body = &src[at..src[at..].find("};").unwrap() + at];
        assert_eq!(body.matches("QWEN4_GDN_GLOBAL*").count(), 5);
        assert_eq!(body.matches(';').count(), 5);
        for field in ["projection", "gate", "beta", "state", "output"] {
            assert!(body.contains(&format!("* {field};")), "field {field}");
        }
    }

    /// The capture lanes kernel's helpers are `tensor_ops.hip`'s, byte for byte (the same
    /// `BEGIN` / `END verbatim` blocks as the uncaptured lanes kernel, the pipe128 body split
    /// out), and its body is the production pipe128 body with exactly the two documented edits
    /// (the `final_out` parameter and the final-state store).
    #[test]
    fn gdn_lanes_capture_f32_gfx942_source_is_tensor_ops_with_the_final_store_edit() {
        const BEGIN: &str = "// BEGIN verbatim tensor_ops.hip\n";
        const END: &str = "// END verbatim tensor_ops.hip";
        const BEGIN_BODY: &str = "// BEGIN derived tensor_ops.hip pipe128 body (final_out)\n";
        const END_BODY: &str = "// END derived tensor_ops.hip pipe128 body (final_out)";
        let src = super::QWEN4_GDN_LANES_CAPTURE_F32_SRC;
        let production = crate::tensor_ops::TENSOR_OPS_SRC;
        let mut blocks = 0;
        let mut rest = src;
        while let Some(at) = rest.find(BEGIN) {
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_gdn_lanes_capture_f32.gfx942.hip is no longer a \
                 substring of tensor_ops.hip: mirror the production edit\nblock starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        assert_eq!(blocks, 4, "bf16 helpers, Q8 helpers, barrier, pipe128 helpers");
        // The derived body: the production pipe128 body with the two edits applied.
        let derived_at = src.find(BEGIN_BODY).expect("derived body marker") + BEGIN_BODY.len();
        let derived = &src[derived_at..derived_at + src[derived_at..].find(END_BODY).unwrap()];
        let derived = derived.trim_end_matches('\n');
        let head = "template <bool kQ8, bool kEf = false>\n__device__ __forceinline__ void gated_delta_step_pipe128_body(";
        let from = production.find(head).expect("production pipe128 body");
        let production_body = &production[from..];
        let to = production_body
            .find("\n}\n")
            .expect("end of the production pipe128 body")
            + 2;
        let production_body = &production_body[..to];
        let signature = "    unsigned short* ef = nullptr) {";
        let store = "    float* const final_state = reinterpret_cast<float*>(state);";
        assert_eq!(production_body.matches(signature).count(), 1);
        assert_eq!(production_body.matches(store).count(), 1);
        let expected = production_body
            .replace(
                signature,
                "    unsigned short* ef = nullptr,\n    float* final_out = nullptr) {",
            )
            .replace(store, "    float* const final_state = final_out;");
        assert_eq!(
            derived,
            expected.trim_end_matches('\n'),
            "the capture lanes body drifted from tensor_ops.hip's pipe128 body"
        );
        assert_eq!(src.matches("gated_delta_step_pipe128_body<false>(").count(), 1);
        assert!(src.contains(&format!(
            "void {}(",
            super::QWEN4_GDN_LANES_CAPTURE_F32_KERNEL
        )));
        assert_eq!(src.matches("__launch_bounds__(128)").count(), 1);
    }

    /// The capture lane table entry is six 8-byte pointers on both sides of the launch.
    #[test]
    fn gdn_lanes_capture_f32_gfx942_lane_desc_matches_kernel_struct() {
        assert_eq!(std::mem::size_of::<super::Qwen4GdnLaneCaptureDesc>(), 48);
        let src = super::QWEN4_GDN_LANES_CAPTURE_F32_SRC;
        let at = src
            .find("struct Qwen4GdnLaneCaptureDesc {")
            .expect("kernel lane descriptor");
        let body = &src[at..src[at..].find("};").unwrap() + at];
        assert_eq!(body.matches("QWEN4_GDN_GLOBAL*").count(), 6);
        assert_eq!(body.matches(';').count(), 6);
        let order: Vec<usize> = ["projection", "gate", "beta", "state", "final_state", "output"]
            .iter()
            .map(|field| body.find(&format!("* {field};")).expect(field))
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "field order");
    }

    /// The gfx942 GDN verify module is `tensor_ops.hip`'s bodies byte for byte: with the four
    /// entry names mapped back, every `BEGIN` / `END verbatim` block occurs in the production
    /// source (the bf16 / Q8 helpers, the barrier, the Q8 peak exchange, the persistent body,
    /// the capture and rollback wrappers, the convolution + parameters kernel, the gated norm).
    #[test]
    fn g942_gdn_verify_source_is_tensor_ops_verbatim() {
        const BEGIN: &str = "// BEGIN verbatim tensor_ops.hip\n";
        const END: &str = "// END verbatim tensor_ops.hip";
        let production = crate::tensor_ops::TENSOR_OPS_SRC;
        let mut src = super::QWEN4_G942_GDN_VERIFY_SRC.to_string();
        for (old, new) in super::QWEN4_G942_GDN_VERIFY_ENTRIES {
            assert!(
                src.contains(&format!("void {new}(")),
                "entry {new} is defined by the module"
            );
            assert!(
                production.contains(&format!("void {old}(")),
                "entry {old} is a tensor_ops kernel"
            );
            src = src.replace(new, old);
        }
        // Mapped back, the module names no tensor_ops entry it does not carry: every
        // `extern "C" __global__` of the module is one of the four.
        assert_eq!(
            src.matches("extern \"C\" __global__").count(),
            super::QWEN4_G942_GDN_VERIFY_ENTRIES.len()
        );
        let mut blocks = 0;
        let mut rest = src.as_str();
        while let Some(at) = rest.find(BEGIN) {
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_g942_gdn_verify.gfx942.hip is no longer a \
                 substring of tensor_ops.hip: mirror the production edit\nblock starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        assert_eq!(
            blocks, 8,
            "bf16 helpers, Q8 helpers, barrier, Q8 peak, persistent body, capture wrapper, \
             convolution + parameters, gated norm"
        );
        // The rollback wrapper is the production wrapper with exactly one edit: the layer table
        // and the pointers loaded from it go through address space 1 (global).
        const BEGIN_WRAP: &str = "// BEGIN derived tensor_ops.hip rollback wrapper (global lane table)\n";
        const END_WRAP: &str = "// END derived tensor_ops.hip rollback wrapper (global lane table)";
        let at = src.find(BEGIN_WRAP).expect("derived wrapper marker") + BEGIN_WRAP.len();
        let derived = &src[at..at + src[at..].find(END_WRAP).unwrap()];
        let derived = &derived[derived.find("#define QWEN4_G942_GDN_GLOBAL").unwrap()..];
        let head = "__launch_bounds__(256)\nextern \"C\" __global__ void gated_delta_rollback_layers_f32(";
        let from = production.find(head).expect("production rollback wrapper");
        let wrapper = &production[from..];
        let wrapper = &wrapper[..wrapper.find("\n}\n").unwrap() + 2];
        let loads = "    const float* recurrence = reinterpret_cast<const float*>(table[2 * blockIdx.y]);\n    float* ring = reinterpret_cast<float*>(table[2 * blockIdx.y + 1]);\n";
        assert_eq!(wrapper.matches(loads).count(), 1);
        let expected = format!(
            "#define QWEN4_G942_GDN_GLOBAL __attribute__((address_space(1)))\n{}",
            wrapper.replace(
                loads,
                "    const unsigned long long QWEN4_G942_GDN_GLOBAL* table_g =\n        (const unsigned long long QWEN4_G942_GDN_GLOBAL*)table;\n    const float* recurrence =\n        (const float*)(const float QWEN4_G942_GDN_GLOBAL*)table_g[2 * blockIdx.y];\n    float* ring = (float*)(float QWEN4_G942_GDN_GLOBAL*)table_g[2 * blockIdx.y + 1];\n"
            )
        );
        assert_eq!(
            derived.trim_end_matches('\n'),
            expected.trim_end_matches('\n'),
            "the rollback wrapper drifted from tensor_ops.hip's"
        );
        // gfx942 never defines the gfx12-only staged parameter pass.
        assert!(!src.contains("#define GDN_PERSISTENT_STAGED"));
    }

    /// The HC-gates module's blocks are `qwen4_gemv_bf16_xf32.hip`'s, byte for byte: every block
    /// between the `BEGIN` / `END verbatim qwen4_gemv_bf16_xf32.hip` markers must occur in the
    /// production source, and the module carries exactly the seven tiled entries.
    #[test]
    fn g942_hc_gates_source_is_qwen4_gemv_bf16_xf32_verbatim() {
        const BEGIN: &str = "// BEGIN verbatim qwen4_gemv_bf16_xf32.hip\n";
        const END: &str = "// END verbatim qwen4_gemv_bf16_xf32.hip";
        let src = super::QWEN4_G942_HC_GATES_SRC;
        let production = crate::kernels::QWEN4_GEMV_BF16_XF32_SRC;
        let mut blocks = 0;
        let mut rest = src;
        while let Some(at) = rest.find(BEGIN) {
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_g942_hc_gates.gfx942.hip is no longer a \
                 substring of qwen4_gemv_bf16_xf32.hip: mirror the production edit into the \
                 HC-gates module\nblock starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        assert_eq!(blocks, 4, "bf16 macros, row_multi, k4 rows body, tiled entry macro");
        // Nothing outside the markers but the preamble (comments, includes).
        let stripped: String = {
            let mut out = String::new();
            let mut rest = src;
            while let Some(at) = rest.find(BEGIN) {
                out.push_str(&rest[..at]);
                let body = &rest[at + BEGIN.len()..];
                let end = body.find(END).unwrap();
                rest = &body[end + END.len()..];
            }
            out.push_str(rest);
            out
        };
        for line in stripped.lines() {
            let t = line.trim();
            assert!(
                t.is_empty() || t.starts_with("//") || t.starts_with("#include"),
                "unmarked code in the HC-gates module: {t:?}"
            );
        }
        // Exactly the seven tiled instantiations, none else.
        for r in 2..=8 {
            assert!(src.contains(&format!("GEMV_BF16_XF32_K4_ROWS_TILED({r})\n")), "R={r}");
        }
        assert_eq!(src.matches("\nGEMV_BF16_XF32_K4_ROWS_TILED(").count(), 7);
        assert_eq!(src.matches("extern \"C\" __global__").count(), 1);
        assert_eq!(super::QWEN4_G942_HC_GATES_TILED.len(), 7);
        for (i, name) in super::QWEN4_G942_HC_GATES_TILED.iter().enumerate() {
            assert_eq!(*name, format!("gemv_bf16_xf32_k4_rows_tiled_r{}", i + 2));
        }
        // The production module still defines the same entries (the non-gfx942 route).
        assert!(production.contains("gemv_bf16_xf32_k4_rows_tiled_r##R"));
    }

    /// The frozen hipcc anchor of the HC-gates module: `hipcc --genco --offload-arch=gfx942 -O3
    /// --no-offload-compress -fuse-cuid=none -mcode-object-version=6` (`core_hipcc_flags`) of the
    /// pinned source, built twice on MI300X (ROCm 7.16, AMD clang 24.0.0git d6f6cb6918) with the
    /// same bytes. Re-seal both hashes together when the source changes.
    #[test]
    fn g942_hc_gates_anchor_and_source_are_pinned() {
        use sha2::{Digest, Sha256};
        const SOURCE_SHA256: &str =
            "4f94fe4e929fcbb22a8ec115220b4e21d154093403129f7aeea045d975416740";
        const ANCHOR_SHA256: &str =
            "00be6bc34e992331dd241f520e7659ebfd0ff28515895f83839bebe0aefe9287";
        let hex = |bytes: &[u8]| -> String {
            Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        assert_eq!(hex(super::QWEN4_G942_HC_GATES_SRC.as_bytes()), SOURCE_SHA256);
        let anchor: &[u8] =
            include_bytes!("../../../../kernels/qwen4_g942_hc_gates_hipcc_gfx942.co");
        assert_eq!(hex(anchor), ANCHOR_SHA256);
        // Every entry the module serves is exported by the anchor.
        for name in super::QWEN4_G942_HC_GATES_TILED {
            assert!(
                anchor.windows(name.len()).any(|w| w == name.as_bytes()),
                "{name} missing from the anchor"
            );
        }
    }

    /// The single-row QSA attention module is `tensor_ops.hip`'s body byte for byte: with the two
    /// entry names mapped back, every `BEGIN` / `END verbatim` block occurs in the production
    /// source (format helpers + key dots, the gfx942 rounding pins + `qsa_attention_row_parallel`,
    /// the `qsa_attention_single` wrapper, the `QSA_ATTENTION_SINGLE` entry macro), and nothing
    /// else but the two instantiations is code.
    #[test]
    fn g942_qsa_row_source_is_tensor_ops_verbatim() {
        const BEGIN: &str = "// BEGIN verbatim tensor_ops.hip\n";
        const END: &str = "// END verbatim tensor_ops.hip";
        let production = crate::tensor_ops::TENSOR_OPS_SRC;
        let mut src = super::QWEN4_G942_QSA_ROW_SRC.to_string();
        for (old, new) in super::QWEN4_G942_QSA_ROW_ENTRIES {
            assert!(
                src.contains(&format!("QSA_ATTENTION_SINGLE({new},")),
                "entry {new} is instantiated by the module"
            );
            assert!(
                production.contains("QSA_ATTENTION_SINGLE(indexed_attention_attention_##TAG, qsa_attention_single,"),
                "entry {old} is the tensor_ops single-row instantiation"
            );
            src = src.replace(new, old);
        }
        assert_eq!(src.matches("QSA_ATTENTION_SINGLE(indexed_attention_attention_").count(), 2);
        let mut blocks = 0;
        let mut rest = src.as_str();
        let mut outside = String::new();
        while let Some(at) = rest.find(BEGIN) {
            outside.push_str(&rest[..at]);
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_g942_qsa_row_attention.gfx942.hip is no longer \
                 a substring of tensor_ops.hip: mirror the production edit\nblock starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        outside.push_str(rest);
        assert_eq!(
            blocks, 4,
            "format helpers + key dots, rounding pins + row body, single wrapper, entry macro"
        );
        // The production source instantiates both entries under the arch guards the module
        // relies on: F32 unguarded, FNUZ gfx942-only (the only formats gfx942 compiles).
        assert!(production.contains("QSA_ATTENTION_KERNELS(f32, QSA_KV_F32, float)"));
        assert!(production
            .contains("#if defined(__gfx942__)\nQSA_ATTENTION_KERNELS(fp8_fnuz, QSA_KV_FP8_FNUZ, unsigned char)"));
        // Outside the markers: comments, includes, the bf16 alias and the two instantiations.
        let code: Vec<&str> = outside
            .lines()
            .map(str::trim)
            .filter(|t| {
                !(t.is_empty()
                    || t.starts_with("//")
                    || t.starts_with("#include")
                    || t.starts_with("using __hip_bfloat16"))
            })
            .collect();
        assert_eq!(
            code,
            [
                "QSA_ATTENTION_SINGLE(indexed_attention_attention_f32, qsa_attention_single, __launch_bounds__(256),",
                "QSA_KV_F32, float)",
                "QSA_ATTENTION_SINGLE(indexed_attention_attention_fp8_fnuz, qsa_attention_single, __launch_bounds__(256),",
                "QSA_KV_FP8_FNUZ, unsigned char)",
            ],
            "unmarked code in the QSA row module"
        );
    }

    /// The frozen hipcc anchor of the QSA row module: `hipcc --genco --offload-arch=gfx942 -O3
    /// --no-offload-compress -fuse-cuid=none -mcode-object-version=6` (`core_hipcc_flags`) of the
    /// pinned source (ROCm 7.16, AMD clang 24.0.0git d6f6cb6918). Re-seal both hashes together
    /// when the source changes.
    #[test]
    fn g942_qsa_row_anchor_and_source_are_pinned() {
        use sha2::{Digest, Sha256};
        const SOURCE_SHA256: &str =
            "e95e554a7f939feaa2d9a80393edb9a4d1df0596149afb3b4cc54e9599a06b6e";
        const ANCHOR_SHA256: &str =
            "bbc9fbcfe6a2cd5ebc830fc82ffc9a6a360d7efe3c15c5824d93eda5513e4d9c";
        let hex = |bytes: &[u8]| -> String {
            Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        };
        assert_eq!(hex(super::QWEN4_G942_QSA_ROW_SRC.as_bytes()), SOURCE_SHA256);
        let anchor: &[u8] =
            include_bytes!("../../../../kernels/qwen4_g942_qsa_row_attention_hipcc_gfx942.co");
        assert_eq!(hex(anchor), ANCHOR_SHA256);
        // Exactly the entries gfx942 can select are exported by the anchor, and no serial or
        // batched sibling is.
        for (_, name) in super::QWEN4_G942_QSA_ROW_ENTRIES {
            assert!(
                anchor.windows(name.len()).any(|w| w == name.as_bytes()),
                "{name} missing from the anchor"
            );
        }
        for absent in ["_serial", "_batched", "_hg4", "_hg12"] {
            assert!(
                !anchor.windows(absent.len()).any(|w| w == absent.as_bytes()),
                "{absent} entry in the QSA row anchor"
            );
        }
    }
}
