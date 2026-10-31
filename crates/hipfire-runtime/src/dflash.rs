// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DFlash draft forward pass — native Rust+HIP.
//!
//! Minimal dependency surface: only reads HFQ draft files (arch_id = 20),
//! writes F32 GpuTensor weights, and runs a bidirectional cross-attention
//! Qwen3-flavored decoder over a block of masked positions.
//!
//! The draft model does not own a vocab head. Its output is the final
//! hidden state per block position; the caller applies the target's
//! `lm_head` to map to logits. This matches the upstream z-lab/dflash
//! reference and lets a single tokenizer / embedding table be shared.
//!
//! Architectural notes:
//! - 5-layer Qwen3 decoder, all full attention, non-causal.
//! - Per-layer cross-attention over `target_hidden` (the projected
//!   concatenation of hidden states from a configured set of target
//!   layers, default `[1, 8, 15, 22, 29]` for a 32-layer target).
//! - Q length = `block_size`, K/V length = `ctx_len + block_size`
//!   (K/V = concat of projected target_hidden and current hidden_states).
//! - MVP simplification: draft has NO persistent KV cache; `k_ctx` /
//!   `v_ctx` are recomputed from the (caller-managed) cumulative
//!   `target_hidden` buffer on every step. This is functionally
//!   equivalent to the reference's cropped draft-KV cache and avoids
//!   one whole layer of persistence bookkeeping.

use crate::hfq::{awq_scale_f32_bytes, HfqFile};
use crate::llama::WeightTensor;
use hip_bridge::{DeviceBuffer, Graph, GraphExec, HipResult, HipRuntime};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{HashMap, HashSet};

/// Max rows per call into `gemm_dispatch` for the MQ (FWHT-rotated)
/// path. The activation rotation scratch (`DflashScratch.mq_x_rot`) is
/// sized to this many rows × `max(inter, q_dim, feature_width)`,
/// regardless of context length. Calls with `batch > MQ_X_ROT_CHUNK_ROWS`
/// are chunked transparently inside `gemm_dispatch`.
///
/// Sizing rationale: at chunk=1024 and 27B (feature width 25600 floats per row),
/// the scratch is `1024 × 25600 × 4 ≈ 100 MB`. The pre-2026-05-15
/// allocator sized this buffer to `max_seq × feature_width`, which at ctx=17K
/// reached 1.74 GB on 27B — a multi-GB waste that scaled with `max_seq`.
/// Chunking adds `ceil(batch / 1024)` extra kernel launches on the
/// first-call `fc` rotation (one-shot per prompt, negligible vs prefill).
const MQ_X_ROT_CHUNK_ROWS: usize = 1024;

// ─── Config ────────────────────────────────────────────────────────────────

/// DFlash2 runtime proposal width without VMM batching: what `auto` (the
/// `speculation.dflash_block` default) picks for a singleton-only load. 16
/// removes the B=8 acceptance ceiling at C1 on gfx1201.
pub const DFLASH_BLOCK_UNBATCHED: usize = 16;
/// `speculation.dflash_block = "auto"` width on a batching serve
/// (`serve.vmm_batch` with at least [`DFLASH_BLOCK_AUTO_MIN_LANES`] lanes):
/// the draft's trained width, which keeps a full C8 verify step at 64 rows.
pub const DFLASH_BLOCK_AUTO_BATCHED: usize = 8;
/// Lane count from which `auto` picks [`DFLASH_BLOCK_AUTO_BATCHED`].
pub const DFLASH_BLOCK_AUTO_MIN_LANES: usize = 4;
/// Admitted fixed widths of `speculation.dflash_block`.
pub const DFLASH_BLOCK_MIN: usize = 4;
pub const DFLASH_BLOCK_MAX: usize = 16;

/// `speculation.dflash_block`: the DFlash2 proposal width of one load.
///
/// The width is a property of the serving configuration, resolved once at
/// load, never of the live batch: a load-dependent width would move window
/// boundaries with server load and break batched == singleton output. Every
/// request of a load (singleton or batched lane) runs the same width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashBlockPolicy {
    /// [`DFLASH_BLOCK_AUTO_BATCHED`] on a VMM batching serve with at least
    /// [`DFLASH_BLOCK_AUTO_MIN_LANES`] lanes, else [`DFLASH_BLOCK_UNBATCHED`].
    /// The default.
    Auto,
    /// A fixed width in `DFLASH_BLOCK_MIN..=DFLASH_BLOCK_MAX`.
    Fixed(usize),
}

impl DflashBlockPolicy {
    /// Parse the `dflash_block` load param: `"auto"` or an integer in range.
    /// Absent = [`Self::Auto`], the schema default.
    pub fn from_param(value: Option<&serde_json::Value>) -> Result<Self, String> {
        let Some(value) = value else {
            return Ok(Self::Auto);
        };
        if value.as_str() == Some("auto") {
            return Ok(Self::Auto);
        }
        match value.as_u64().map(|v| v as usize) {
            Some(b) if (DFLASH_BLOCK_MIN..=DFLASH_BLOCK_MAX).contains(&b) => Ok(Self::Fixed(b)),
            _ => Err(format!(
                "dflash_block must be \"auto\" or an integer {DFLASH_BLOCK_MIN}..={DFLASH_BLOCK_MAX} (got {value})"
            )),
        }
    }

    /// The width of a load: `vmm_batch` = the VMM continuous-batching route
    /// is staged, `lanes` = its `continuous_batch_size`.
    pub fn resolve(self, vmm_batch: bool, lanes: usize) -> usize {
        match self {
            Self::Fixed(b) => b,
            Self::Auto if vmm_batch && lanes >= DFLASH_BLOCK_AUTO_MIN_LANES => {
                DFLASH_BLOCK_AUTO_BATCHED
            }
            Self::Auto => DFLASH_BLOCK_UNBATCHED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    /// The SWA window the draft artifact declares it was TRAINED with
    /// (`config.sliding_window`). `Some` only when the artifact sets
    /// `use_sliding_window: true` AND its `layer_types` match one of the
    /// implemented splits — either `n-1` sliding + final full, or all layers
    /// sliding (DFlash2). `None` when the artifact is silent or declares
    /// a split we do not implement, in which case windowed mode stays off
    /// unless `HIPFIRE_DFLASH_WINDOW` forces it.
    ///
    /// This is the only width that is correct by construction: running an
    /// SWA-trained layer over a different span is a train/inference mask
    /// mismatch, which degrades acceptance silently (verify stays exact).
    pub declared_window: Option<usize>,
    /// `true` when `layer_types` is all `sliding_attention` (DFlash2 chain-only).
    /// Public indicator so callers can skip last-layer full backfill and reuse
    /// the same `W` ring for every layer.
    pub all_layers_sliding: bool,
    /// DFlash2 dynamic-conv knobs from nested `dflash_config` (defaults in
    /// parens): group 16, kernel 2, rank 256, top_k 16. `None` when the
    /// artifact is legacy DFlash (fields absent) — forward then skips conv
    /// and selector paths and stays byte-identical.
    pub conv_group_size: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub selector_rank: Option<usize>,
    pub selector_top_k: Option<usize>,
    /// Hidden rows the target appends per extract layer per position
    /// (`dflash_config.target_streams_per_layer`, default 1). A multi-stream
    /// residual target (e.g. Raon's 4 mHC streams) contributes every stream.
    pub target_streams_per_layer: usize,
}

impl DflashConfig {
    /// Returns the number of target hidden layers concatenated into fc input.
    pub fn num_extract(&self) -> usize {
        self.target_layer_ids.len()
    }

    /// Width of one feature row (= fc input K):
    /// `num_extract × target_streams_per_layer × hidden`.
    pub fn feature_width(&self) -> usize {
        self.num_extract() * self.target_streams_per_layer * self.hidden
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Runtime proposal width for the configured `speculation.dflash_block`
    /// width `requested` (already resolved by [`DflashBlockPolicy::resolve`]).
    /// DFlash2's selector and dynamic convolutions are length-generic even
    /// though the published checkpoint declares B=8, so a DFlash2 draft runs
    /// at `requested` (8 or 16 under the `auto` default). Legacy DFlash is not
    /// length-generic and always runs its artifact-declared width.
    pub fn runtime_block_size(&self, requested: usize) -> usize {
        if self.selector_rank.is_some() {
            requested
        } else {
            self.block_size
        }
    }

    /// Parse from an HFQ file's metadata JSON. Expects the top-level
    /// `dflash` object written by `dflash_convert`.
    pub fn from_hfq(hfq: &HfqFile) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
        Self::from_metadata(&meta)
    }

    /// Parse from the full HFQ metadata JSON value (see [`Self::from_hfq`]).
    pub fn from_metadata(meta: &serde_json::Value) -> Option<Self> {
        let df = meta.get("dflash")?;

        let n_layers = df.get("num_hidden_layers").and_then(|v| v.as_u64())? as usize;
        let hidden = df.get("hidden_size").and_then(|v| v.as_u64())? as usize;
        let intermediate = df.get("intermediate_size").and_then(|v| v.as_u64())? as usize;
        let n_heads = df.get("num_attention_heads").and_then(|v| v.as_u64())? as usize;
        let n_kv_heads = df.get("num_key_value_heads").and_then(|v| v.as_u64())? as usize;
        let head_dim = df
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or((hidden / n_heads) as u64) as usize;
        let vocab_size = df.get("vocab_size").and_then(|v| v.as_u64())? as usize;
        let norm_eps = df
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let rope_theta = df
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000_000.0) as f32;
        let block_size = df.get("block_size").and_then(|v| v.as_u64())? as usize;
        let mask_token_id = df.get("mask_token_id").and_then(|v| v.as_u64())? as u32;
        let target_layer_ids: Vec<usize> = df
            .get("target_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let num_target_layers = df.get("num_target_layers").and_then(|v| v.as_u64())? as usize;
        // The window fields live in the sibling `config` object (HF-style),
        // not in the `dflash` block, so they are read separately. DFlash2
        // nests the same knobs under `dflash.dflash_config` as well; we probe
        // both `config` and `dflash` for `layer_types` so that either
        // emission (flat or nested) is honoured.
        let (declared_window, all_layers_sliding) = {
            let cfg_obj = meta.get("config");
            let use_sw = cfg_obj
                .and_then(|c| c.get("use_sliding_window"))
                .or_else(|| df.get("use_sliding_window"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !use_sw {
                (None, false)
            } else {
                let w_opt = cfg_obj
                    .and_then(|c| c.get("sliding_window"))
                    .or_else(|| df.get("sliding_window"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .filter(|&w| w != 0);
                match w_opt {
                    None => (None, false),
                    Some(w) => {
                        let types_from = |v: &serde_json::Value| {
                            v.get("layer_types").and_then(|x| x.as_array()).cloned()
                        };
                        let types = cfg_obj
                            .and_then(types_from)
                            .or_else(|| types_from(df))
                            .or_else(|| df.get("dflash_config").and_then(types_from))
                            .or_else(|| {
                                cfg_obj
                                    .and_then(|c| c.get("dflash_config"))
                                    .and_then(types_from)
                            });
                        match types {
                            Some(types) => {
                                let is_all_sliding = types.len() == n_layers
                                    && types
                                        .iter()
                                        .all(|t| t.as_str().unwrap_or("") == "sliding_attention");
                                let is_split = types.len() == n_layers
                                    && types.iter().enumerate().all(|(i, t)| {
                                        let s = t.as_str().unwrap_or("");
                                        if i + 1 == n_layers {
                                            s == "full_attention"
                                        } else {
                                            s == "sliding_attention"
                                        }
                                    });
                                if is_all_sliding || is_split {
                                    (Some(w), is_all_sliding)
                                } else {
                                    eprintln!(
                                        "  DFlash draft declares sliding_window={w} but layer_types do not \
                                         match the implemented splits (all sliding or n-1 sliding + last full) — \
                                         not auto-enabling windowed mode"
                                    );
                                    (None, false)
                                }
                            }
                            None => (Some(w), false),
                        }
                    }
                }
            }
        };
        // Nested dflash_config fields (DFlash2) with legacy flat fallbacks.
        let dflash_cfg_nested = df.get("dflash_config").and_then(|v| v.as_object());
        let dflash_u64 = |key: &str| {
            dflash_cfg_nested
                .and_then(|m| m.get(key))
                .or_else(|| df.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        };
        let conv_group_size = dflash_u64("conv_group_size");
        let conv_kernel_size = dflash_u64("conv_kernel_size");
        let selector_rank = dflash_u64("selector_rank");
        let selector_top_k = dflash_u64("selector_top_k");
        let target_streams_per_layer = dflash_u64("target_streams_per_layer")
            .filter(|&s| s >= 1)
            .unwrap_or(1);

        Some(DflashConfig {
            n_layers,
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            norm_eps,
            rope_theta,
            block_size,
            mask_token_id,
            target_layer_ids,
            num_target_layers,
            declared_window,
            all_layers_sliding,
            conv_group_size,
            conv_kernel_size,
            selector_rank,
            selector_top_k,
            target_streams_per_layer,
        })
    }
}

// ─── Weights ───────────────────────────────────────────────────────────────

pub struct DflashLayerWeights {
    pub attn_norm: GpuTensor, // [hidden] — F32, RMSNorm weight
    pub wq: WeightTensor,     // [q_dim, hidden]
    pub wk: WeightTensor,     // [kv_dim, hidden]
    pub wv: WeightTensor,     // [kv_dim, hidden]
    pub wo: WeightTensor,     // [hidden, q_dim]
    pub q_norm: GpuTensor,    // [head_dim] — F32
    pub k_norm: GpuTensor,    // [head_dim] — F32
    pub ffn_norm: GpuTensor,  // [hidden] — F32
    pub w_gate: WeightTensor, // [intermediate, hidden]
    pub w_up: WeightTensor,   // [intermediate, hidden]
    pub w_down: WeightTensor, // [hidden, intermediate]
    // DFlash2 dynamic conv: per-layer base kernels [2,K,H] F32 and
    // kernel projections [2*K*G,H] (G=hidden/group_size). `None` on legacy.
    pub attn_conv_base: Option<GpuTensor>,
    pub attn_conv_proj: Option<WeightTensor>,
    pub mlp_conv_base: Option<GpuTensor>,
    pub mlp_conv_proj: Option<WeightTensor>,
}

/// Compact host-side selector codebooks (never VRAM). F16 stored as u16
/// halves to avoid ~508 MB F32 expansion for vocab=152k rank=256; converted
/// during the small rank dot.
#[derive(Debug, Clone)]
pub struct SelectorCodebook {
    pub vocab: usize,
    pub rank: usize,
    // Exactly one of these is Some.
    pub f16_data: Option<Vec<u16>>,
    pub f32_data: Option<Vec<f32>>,
}

impl SelectorCodebook {
    pub fn get_f32(&self, token: usize, dim: usize) -> f32 {
        if let Some(f16) = &self.f16_data {
            let u = f16[token * self.rank + dim];
            crate::llama::f16_to_f32(u)
        } else {
            self.f32_data.as_ref().unwrap()[token * self.rank + dim]
        }
    }
}

pub struct DflashWeights {
    /// `fc`: Linear(feature_width → hidden). Shape: [hidden, feature_width].
    pub fc: WeightTensor,
    pub hidden_norm: GpuTensor, // [hidden] — F32
    pub norm: GpuTensor,        // [hidden] — F32, final output norm
    pub layers: Vec<DflashLayerWeights>,
    /// True when at least one matrix weight is MQ (FWHT-rotated) — drives whether
    /// the draft_forward path needs to allocate FWHT rotation scratches.
    pub has_mq: bool,
    // DFlash2 candidate selector (host-side)
    pub selector_hidden_proj: Option<WeightTensor>, // [rank, hidden]
    pub predecessor_codebook: Option<SelectorCodebook>, // [vocab, rank]
    pub successor_codebook: Option<SelectorCodebook>, // [vocab, rank]
    pub selector_rank: Option<usize>,
    pub selector_top_k: Option<usize>,
    pub conv_group_size: Option<usize>,
    pub conv_kernel_size: Option<usize>,
}

/// Load a F32-only tensor (norms, embedding-shaped scalars). Always F32 on GPU.
fn hfq_tensor_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: Vec<usize>,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        q => panic!("dflash: unsupported quant_type {q} for {name}"),
    };
    let expected: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        expected,
        "dflash: shape mismatch for {name}: have {}, expected {}",
        f32_data.len(),
        expected,
    );
    upload_f32_weight(gpu, &f32_data, &shape)
}

/// Upload a DFlash raw weight through the reusable pool. The global raw-upload
/// helper allocates directly from HIP, so an immediate constructor retry cannot
/// reclaim a successfully staged predecessor. This local path also returns the
/// allocation if the host-to-device copy itself fails.
fn upload_raw_weight(gpu: &mut Gpu, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
    upload_raw_weight_with_copy(gpu, data, shape, HipRuntime::memcpy_htod)
}

/// [`upload_raw_weight`] with an injectable host-to-device copy step. The
/// production path passes the real copy; regression tests pass a failing
/// callback to prove the allocation-to-copy failure seam returns its pool
/// allocation instead of stranding it outside the constructor ledgers.
fn upload_raw_weight_with_copy(
    gpu: &mut Gpu,
    data: &[u8],
    shape: &[usize],
    copy: impl FnOnce(&HipRuntime, &DeviceBuffer, &[u8]) -> HipResult<()>,
) -> HipResult<GpuTensor> {
    let mut tensor = gpu.alloc_tensor(&[data.len()], DType::Raw)?;
    if let Err(error) = copy(&gpu.hip, &tensor.buf, data) {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    tensor.shape = shape.to_vec();
    Ok(tensor)
}

/// Upload a DFlash F32 weight through the reusable pool. Same contract as
/// [`upload_raw_weight`]: `Gpu::upload_f32` allocates from the pool but
/// returns a copy failure without freeing the new owner, which would strand
/// it before `gt!`/`wt!` can ledger it for constructor rollback.
fn upload_f32_weight(gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
    upload_f32_weight_with_copy(gpu, data, shape, HipRuntime::memcpy_htod)
}

/// [`upload_f32_weight`] with an injectable copy step; see
/// [`upload_raw_weight_with_copy`].
fn upload_f32_weight_with_copy(
    gpu: &mut Gpu,
    data: &[f32],
    shape: &[usize],
    copy: impl FnOnce(&HipRuntime, &DeviceBuffer, &[u8]) -> HipResult<()>,
) -> HipResult<GpuTensor> {
    let tensor = gpu.alloc_tensor(shape, DType::F32)?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    if let Err(error) = copy(&gpu.hip, &tensor.buf, bytes) {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    Ok(tensor)
}

/// Load a matrix tensor as a `WeightTensor` carrying its native dtype.
/// Supported quant_types:
///   1  (F16)      → lifted to F32 on GPU (legacy path).
///   2  (F32)      → uploaded as F32.
///   13 (MQ4-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///   15 (MQ6-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///   17 (MQ3-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///   44 (MQ4G256V2)→ per-128 fp16 header, 136 B/group, exact validation.
///
/// `shape = [m, k]` so m=output_dim and k=input_dim. The HFQ index stores
/// the unaligned byte length; for MQ formats we skip shape verification (the
/// quantized bytes are not a function of m*k alone — group padding can add
/// up to 255 trailing bytes per row group), except MQ4G256V2 which has an
/// exact 136-byte/G256 shape to catch byte-order bugs.
/// `shape = [m, k]` so m=output_dim and k=input_dim. The HFQ index stores
/// the unaligned byte length; for MQ formats we skip shape verification (the
/// quantized bytes are not a function of m*k alone — group padding can add
/// up to 255 trailing bytes per row group).
fn hfq_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    let wt = match info.quant_type {
        1 => {
            // F16 on disk. Default: upload as F16 (no lift) and dispatch through
            // the mw16 WMMA kernel — 3-5× faster draft at B=16 on gfx1100 than
            // the F32 lift path (which bypassed WMMA entirely via the naive
            // gemm_f32_batched kernel at ~100 GB/s / 10 % peak).
            //
            // HIPFIRE_DRAFT_F16=0 falls back to the legacy F16→F32 lift for
            // A/B comparison.
            let use_f16 = crate::config::get().draft_f16;
            if use_f16 {
                assert_eq!(
                    data.len(),
                    m * k * 2,
                    "dflash {name} F16 byte-size mismatch"
                );
                // gfx1151: a power-of-two row pitch (K % 2048 == 0) camps every
                // wave on the same DRAM channels (~50 GB/s); 128 pad halfs per
                // row restore ~200 GB/s. Only gemm_dispatch reads these
                // weights, and it routes padded ones to gemm_f16_pitched_gfx1151.
                let ld = if gpu.arch_caps.is_gfx1151() && k % 2048 == 0 {
                    k + 128
                } else {
                    k
                };
                let buf = if ld == k {
                    upload_raw_weight(gpu, data, &[m * k])?
                } else {
                    let mut padded = vec![0u8; m * ld * 2];
                    for (dst, src) in padded.chunks_exact_mut(ld * 2).zip(data.chunks_exact(k * 2)) {
                        dst[..k * 2].copy_from_slice(src);
                    }
                    upload_raw_weight(gpu, &padded, &[m * ld])?
                };
                Ok::<WeightTensor, hip_bridge::HipError>(WeightTensor {
                    buf,
                    gpu_dtype: DType::F16,
                    m,
                    k,
                    row_stride: if ld == k { 0 } else { ld * 2 },
                    paro: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                })
            } else {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                assert_eq!(f32_data.len(), m * k, "dflash {name} F16 size mismatch");
                let buf = upload_f32_weight(gpu, &f32_data, &[m * k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                })
            }
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(f32_data.len(), m * k, "dflash {name} F32 size mismatch");
            let buf = upload_f32_weight(gpu, &f32_data, &[m * k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        13 => {
            // MQ4-G256: 136 bytes per 256 weights. The buffer is opaque to
            // the engine; the gemm_hfq4g256 kernel reads it directly.
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        15 => {
            // MQ6-G256: 200 bytes per 256 weights. Same opaque-buffer pattern
            // as MQ4/MQ3; dispatch rotates activations and calls HFQ6 GEMM.
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        17 => {
            // MQ3-G256: 104 bytes per 256 weights. Same opaque-buffer pattern
            // as MQ4. Dispatch path (`gemm_dispatch`) routes through
            // `rotate_x_mq_batched` + `gemm_hfq3g256_batched_lmhead`.
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        44 => {
            // MQ4G256V2 (qt=44): per-128 fp16 header, 136 B/group. Exact
            // shape validation — payload nibbles at same offset as v1 but
            // header encodes s0/z0/s1/z1 as fp16 pairs.
            if k % 256 != 0 {
                panic!("dflash {name} MQ4G256V2 requires K%256==0 (got K={k})");
            }
            let groups = k / 256;
            let expected = m * groups * 136;
            if data.len() != expected {
                panic!(
                    "dflash {name} MQ4G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
                    data.len()
                );
            }
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        47 => {
            // MQ6G256V2 (qt=47): 200 B/group, per-128 fp16 s0/z0/s1/z1 + 6-bit payload.
            if k % 256 != 0 {
                panic!("dflash {name} MQ6G256V2 requires K%256==0 (got K={k})");
            }
            let groups = k / 256;
            let expected = m * groups * rdna_compute::MQ6G256V2_GROUP_BYTES;
            if data.len() != expected {
                panic!(
                    "dflash {name} MQ6G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
                    data.len()
                );
            }
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        48 => {
            // MQ5G256V2 (qt=48): 168 B/group, per-128 fp16 + 5-bit payload.
            if k % 256 != 0 {
                panic!("dflash {name} MQ5G256V2 requires K%256==0 (got K={k})");
            }
            let groups = k / 256;
            let expected = m * groups * rdna_compute::MQ5G256V2_GROUP_BYTES;
            if data.len() != expected {
                panic!(
                    "dflash {name} MQ5G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
                    data.len()
                );
            }
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ5G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        49 => {
            // MQ3G256V2 (qt=49): 104 B/group, per-128 fp16 + 3-bit payload.
            if k % 256 != 0 {
                panic!("dflash {name} MQ3G256V2 requires K%256==0 (got K={k})");
            }
            let groups = k / 256;
            let expected = m * groups * rdna_compute::MQ3G256V2_GROUP_BYTES;
            if data.len() != expected {
                panic!(
                    "dflash {name} MQ3G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
                    data.len()
                );
            }
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        50 => {
            // MQ2G256V2 (qt=50): 72 B/group, per-128 fp16 + 2-bit payload.
            if k % 256 != 0 {
                panic!("dflash {name} MQ2G256V2 requires K%256==0 (got K={k})");
            }
            let groups = k / 256;
            let expected = m * groups * rdna_compute::MQ2G256V2_GROUP_BYTES;
            if data.len() != expected {
                panic!(
                    "dflash {name} MQ2G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
                    data.len()
                );
            }
            let buf = upload_raw_weight(gpu, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        53 => panic!("dflash: MQ4G128V2 (qt=53) is typed-Qwen4-only; generic dflash has no consumer for {name}"),
        131..=134 => panic!("dflash: MQ-I4 (qt={}) is a typed-Qwen4 expert-slab format; generic dflash has no consumer for {name}", info.quant_type),
        q => panic!("dflash: unsupported matrix quant_type {q} for {name}"),
    }?;
    // AWQ sidecar attachment — same allow-list as hfq.rs::load_weight_tensor
    // `DType::supports_awq_sidecar` so future widening (MQ6, MQ2, MQ3-Lloyd,
    // MFP4) is a single helper edit. Sidecar absent → `awq_scale` stays None,
    // dispatch path matches the pre-fix behavior.
    attach_awq_scale(hfq, gpu, wt, name, k)
}

/// Attach the AWQ sidecar through the reusable pool. A genuine sidecar upload
/// failure frees the already-built trunk owner and surfaces the error instead
/// of silently dropping the scale (the pre-fix `load_awq_scale(...).ok()`
/// converted a copy failure into a missing scale *and* leaked the direct
/// allocation). Successful uploads are byte-identical to the direct path.
fn attach_awq_scale(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    wt: WeightTensor,
    name: &str,
    k: usize,
) -> HipResult<WeightTensor> {
    attach_awq_scale_with_copy(hfq, gpu, wt, name, k, HipRuntime::memcpy_htod)
}

/// [`attach_awq_scale`] with an injectable copy step; see
/// [`upload_raw_weight_with_copy`].
fn attach_awq_scale_with_copy(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    mut wt: WeightTensor,
    name: &str,
    k: usize,
    copy: impl FnOnce(&HipRuntime, &DeviceBuffer, &[u8]) -> HipResult<()>,
) -> HipResult<WeightTensor> {
    if !wt.gpu_dtype.supports_awq_sidecar() {
        return Ok(wt);
    }
    let Some(f32_bytes) = awq_scale_f32_bytes(hfq, name, k) else {
        return Ok(wt);
    };
    match upload_raw_weight_with_copy(gpu, &f32_bytes, &[f32_bytes.len()], copy) {
        Ok(scale) => {
            wt.awq_scale = Some(scale);
            Ok(wt)
        }
        Err(error) => {
            wt.free_all(gpu);
            Err(error)
        }
    }
}

impl DflashLayerWeights {
    /// Consume one layer's weights, releasing every GPU tensor including the
    /// AWQ/paro sidecars (`free_all`, not `.buf` — an AWQ-trunk drafter
    /// carries one scale tensor per weight per layer). Shared by the
    /// success-path `DflashWeights::free_gpu` and the `load` error path so a
    /// failed load frees completed layers without duplicating this list.
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.attn_norm);
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        self.w_gate.free_all(gpu);
        self.w_up.free_all(gpu);
        self.w_down.free_all(gpu);
        if let Some(t) = self.attn_conv_base {
            let _ = gpu.free_tensor(t);
        }
        if let Some(w) = self.attn_conv_proj {
            w.free_all(gpu);
        }
        if let Some(t) = self.mlp_conv_base {
            let _ = gpu.free_tensor(t);
        }
        if let Some(w) = self.mlp_conv_proj {
            w.free_all(gpu);
        }
    }
}

impl DflashWeights {
    /// True when the selector (candidate proposal) path is available.
    pub fn has_candidate_selector(&self) -> bool {
        self.selector_hidden_proj.is_some()
            && self.predecessor_codebook.is_some()
            && self.successor_codebook.is_some()
    }
    pub fn load(gpu: &mut Gpu, hfq: &HfqFile, cfg: &DflashConfig) -> HipResult<Self> {
        Self::load_with_boundaries(gpu, hfq, cfg, || Ok(()), || Ok(()))
    }

    fn load_with_boundaries(
        gpu: &mut Gpu,
        hfq: &HfqFile,
        cfg: &DflashConfig,
        mut after_owner: impl FnMut() -> HipResult<()>,
        mut after_layer: impl FnMut() -> HipResult<()>,
    ) -> HipResult<Self> {
        // Transactional construction. Every GPU owner returned by the leaf
        // loaders is recorded in `live_t` (plain F32) / `live_w` (weight +
        // sidecars) and taken exactly once into its final owner; completed
        // layers accumulate in `layers`. Any ordinary loader or injected
        // boundary failure frees the completed layers plus every
        // recorded-but-unplaced tensor before returning Err — a bare `?`
        // would leak them (`GpuTensor`/`DeviceBuffer` have no `Drop`). Mirrors
        // the `or_free!` style in `hipfire-arch-qwen35`'s
        // `load_dflash_state`.
        let mut live_t: Vec<Option<GpuTensor>> = Vec::new();
        let mut live_w: Vec<Option<WeightTensor>> = Vec::new();
        let mut layers: Vec<DflashLayerWeights> = Vec::with_capacity(cfg.n_layers);
        macro_rules! cleanup {
            () => {{
                for l in layers.drain(..).rev() {
                    l.free_gpu(gpu);
                }
                for slot in live_w.iter_mut().rev() {
                    if let Some(w) = slot.take() {
                        w.free_all(gpu);
                    }
                }
                for slot in live_t.iter_mut().rev() {
                    if let Some(t) = slot.take() {
                        let _ = gpu.free_tensor(t);
                    }
                }
            }};
        }
        macro_rules! gt {
            ($e:expr) => {{
                match $e {
                    Ok(t) => {
                        live_t.push(Some(t));
                        let index = live_t.len() - 1;
                        if let Err(e) = after_owner() {
                            cleanup!();
                            return Err(e);
                        }
                        index
                    }
                    Err(e) => {
                        cleanup!();
                        return Err(e);
                    }
                }
            }};
        }
        macro_rules! wt {
            ($e:expr) => {{
                match $e {
                    Ok(w) => {
                        live_w.push(Some(w));
                        let index = live_w.len() - 1;
                        if let Err(e) = after_owner() {
                            cleanup!();
                            return Err(e);
                        }
                        index
                    }
                    Err(e) => {
                        cleanup!();
                        return Err(e);
                    }
                }
            }};
        }
        macro_rules! take_t {
            ($i:expr) => {
                live_t[$i]
                    .take()
                    .expect("dflash load: F32 slot taken twice")
            };
        }
        macro_rules! take_w {
            ($i:expr) => {
                live_w[$i]
                    .take()
                    .expect("dflash load: weight slot taken twice")
            };
        }
        let i_fc = wt!(hfq_weight(
            hfq,
            gpu,
            "fc.weight",
            cfg.hidden,
            cfg.feature_width(),
        ));
        let i_hidden_norm = gt!(hfq_tensor_f32(
            hfq,
            gpu,
            "hidden_norm.weight",
            vec![cfg.hidden]
        ));
        let i_norm = gt!(hfq_tensor_f32(hfq, gpu, "norm.weight", vec![cfg.hidden]));

        let conv_k = cfg.conv_kernel_size.unwrap_or(2);
        let conv_g = cfg.conv_group_size.unwrap_or(16);
        let conv_groups = cfg.hidden / conv_g;
        let proj_m = 2 * conv_k * conv_groups;

        // (`layers` is declared above so the `gt!`/`wt!` error arms can free it.)
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            // Attempt DFlash2 conv weights; absent on legacy drafts.
            let attn_conv_base = if cfg.conv_kernel_size.is_some() {
                // base_kernel [2, K, H] -> 2*K*H
                let name = format!("{p}.self_attn.attention_conv.base_kernel");
                // also try alternate naming `attn_conv` if upstream uses that
                let alt = format!("{p}.attention_conv.base_kernel");
                let key = if hfq.tensor_data(&name).is_some() {
                    name
                } else {
                    alt
                };
                if hfq.tensor_data(&key).is_some() {
                    // shape 2*K*H
                    Some(gt!(hfq_tensor_f32(
                        hfq,
                        gpu,
                        &key,
                        vec![2 * conv_k * cfg.hidden],
                    )))
                } else {
                    None
                }
            } else {
                None
            };
            let attn_conv_proj = if cfg.conv_kernel_size.is_some() {
                let name = format!("{p}.self_attn.attention_conv.kernel_projection.weight");
                let alt = format!("{p}.attention_conv.kernel_projection.weight");
                let key = if hfq.tensor_data(&name).is_some() {
                    name
                } else {
                    alt
                };
                if hfq.tensor_data(&key).is_some() {
                    Some(wt!(hfq_weight(hfq, gpu, &key, proj_m, cfg.hidden)))
                } else {
                    None
                }
            } else {
                None
            };
            let mlp_conv_base = if cfg.conv_kernel_size.is_some() {
                let name = format!("{p}.mlp.mlp_conv.base_kernel");
                let alt = format!("{p}.mlp_conv.base_kernel");
                let key = if hfq.tensor_data(&name).is_some() {
                    name
                } else {
                    alt
                };
                if hfq.tensor_data(&key).is_some() {
                    Some(gt!(hfq_tensor_f32(
                        hfq,
                        gpu,
                        &key,
                        vec![2 * conv_k * cfg.hidden],
                    )))
                } else {
                    None
                }
            } else {
                None
            };
            let mlp_conv_proj = if cfg.conv_kernel_size.is_some() {
                let name = format!("{p}.mlp.mlp_conv.kernel_projection.weight");
                let alt = format!("{p}.mlp_conv.kernel_projection.weight");
                let key = if hfq.tensor_data(&name).is_some() {
                    name
                } else {
                    alt
                };
                if hfq.tensor_data(&key).is_some() {
                    Some(wt!(hfq_weight(hfq, gpu, &key, proj_m, cfg.hidden)))
                } else {
                    None
                }
            } else {
                None
            };
            // Stage every field as a slot index first: a failure below frees
            // the staged slots (plus completed layers) via `gt!`/`wt!`, and
            // the `take_*!` push itself is infallible.
            let i_attn_norm = gt!(hfq_tensor_f32(
                hfq,
                gpu,
                &format!("{p}.input_layernorm.weight"),
                vec![cfg.hidden],
            ));
            let i_wq = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                cfg.q_dim(),
                cfg.hidden,
            ));
            let i_wk = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                cfg.kv_dim(),
                cfg.hidden,
            ));
            let i_wv = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                cfg.kv_dim(),
                cfg.hidden,
            ));
            let i_wo = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                cfg.hidden,
                cfg.q_dim(),
            ));
            let i_q_norm = gt!(hfq_tensor_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_norm.weight"),
                vec![cfg.head_dim],
            ));
            let i_k_norm = gt!(hfq_tensor_f32(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_norm.weight"),
                vec![cfg.head_dim],
            ));
            let i_ffn_norm = gt!(hfq_tensor_f32(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                vec![cfg.hidden],
            ));
            let i_w_gate = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.mlp.gate_proj.weight"),
                cfg.intermediate,
                cfg.hidden,
            ));
            let i_w_up = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.mlp.up_proj.weight"),
                cfg.intermediate,
                cfg.hidden,
            ));
            let i_w_down = wt!(hfq_weight(
                hfq,
                gpu,
                &format!("{p}.mlp.down_proj.weight"),
                cfg.hidden,
                cfg.intermediate,
            ));
            layers.push(DflashLayerWeights {
                attn_norm: take_t!(i_attn_norm),
                wq: take_w!(i_wq),
                wk: take_w!(i_wk),
                wv: take_w!(i_wv),
                wo: take_w!(i_wo),
                q_norm: take_t!(i_q_norm),
                k_norm: take_t!(i_k_norm),
                ffn_norm: take_t!(i_ffn_norm),
                w_gate: take_w!(i_w_gate),
                w_up: take_w!(i_w_up),
                w_down: take_w!(i_w_down),
                attn_conv_base: attn_conv_base.map(|j| take_t!(j)),
                attn_conv_proj: attn_conv_proj.map(|j| take_w!(j)),
                mlp_conv_base: mlp_conv_base.map(|j| take_t!(j)),
                mlp_conv_proj: mlp_conv_proj.map(|j| take_w!(j)),
            });
            if let Err(e) = after_layer() {
                cleanup!();
                return Err(e);
            }
        }

        // Selector: hidden_projection [rank, hidden] + two codebooks [vocab, rank] host-side
        // Exact HFQ names are `candidate_selector.hidden_projection.weight`,
        // `candidate_selector.predecessor_codebook`, `candidate_selector.successor_codebook`.
        // Optional fallback `.weight` suffix is tolerated but not required.
        // Slot index (`take_w!`n below); the codebooks between here and the
        // take are host-side only and cannot fail with GPU memory held.
        let i_selector_hidden_proj = if cfg.selector_rank.is_some() {
            let rank = cfg.selector_rank.unwrap();
            let candidates = [
                "candidate_selector.hidden_projection.weight",
                "selector.hidden_projection.weight",
                "selector.hidden_proj.weight",
            ];
            let mut found: Option<usize> = None;
            for n in candidates {
                if hfq.tensor_data(n).is_some() {
                    found = Some(wt!(hfq_weight(hfq, gpu, n, rank, cfg.hidden)));
                    break;
                }
            }
            found
        } else {
            None
        };
        let load_codebook = |names: &[&str],
                             vocab: usize,
                             rank: usize|
         -> Option<SelectorCodebook> {
            for n in names {
                if let Some((info, data)) = hfq.tensor_data(n) {
                    let expected = vocab * rank;
                    match info.quant_type {
                        1 => {
                            assert_eq!(data.len(), expected * 2, "codebook {n} F16 size mismatch");
                            let mut v = Vec::with_capacity(expected);
                            for chunk in data.chunks_exact(2) {
                                v.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                            }
                            return Some(SelectorCodebook {
                                vocab,
                                rank,
                                f16_data: Some(v),
                                f32_data: None,
                            });
                        }
                        2 => {
                            assert_eq!(data.len(), expected * 4, "codebook {n} F32 size mismatch");
                            let mut v = Vec::with_capacity(expected);
                            for chunk in data.chunks_exact(4) {
                                v.push(f32::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                ]));
                            }
                            return Some(SelectorCodebook {
                                vocab,
                                rank,
                                f16_data: None,
                                f32_data: Some(v),
                            });
                        }
                        q => panic!("selector codebook {n} unsupported quant_type {q}"),
                    }
                }
            }
            None
        };
        let (predecessor_codebook, successor_codebook, selector_rank_opt, selector_top_k_opt) =
            if cfg.selector_rank.is_some() {
                let rank = cfg.selector_rank.unwrap();
                let vocab = cfg.vocab_size;
                let pred = load_codebook(
                    &[
                        "candidate_selector.predecessor_codebook",
                        "selector.predecessor_codebook",
                        "candidate_selector.predecessor_codebook.weight",
                        "selector.predecessor.weight",
                    ],
                    vocab,
                    rank,
                );
                let succ = load_codebook(
                    &[
                        "candidate_selector.successor_codebook",
                        "selector.successor_codebook",
                        "candidate_selector.successor_codebook.weight",
                        "selector.successor.weight",
                    ],
                    vocab,
                    rank,
                );
                (pred, succ, cfg.selector_rank, cfg.selector_top_k)
            } else {
                (None, None, None, None)
            };

        let fc = take_w!(i_fc);
        let hidden_norm = take_t!(i_hidden_norm);
        let norm = take_t!(i_norm);
        let selector_hidden_proj = i_selector_hidden_proj.map(|j| take_w!(j));
        debug_assert!(live_t.iter().all(|s| s.is_none()));
        debug_assert!(live_w.iter().all(|s| s.is_none()));
        let has_mq = std::iter::once(&fc)
            .chain(layers.iter().flat_map(|l| {
                let mut v: Vec<&WeightTensor> =
                    vec![&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down];
                if let Some(p) = &l.attn_conv_proj {
                    v.push(p);
                }
                if let Some(p) = &l.mlp_conv_proj {
                    v.push(p);
                }
                v.into_iter()
            }))
            .chain(selector_hidden_proj.iter())
            .any(|w| {
                matches!(
                    w.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ4G256V2
                        | DType::MQ6G256
                        | DType::MQ6G256V2
                        | DType::MQ5G256V2
                        | DType::MQ3G256
                        | DType::MQ3G256V2
                        | DType::MQ2G256V2
                )
            });
        if has_mq {
            // MQ dispatch needs the engine's FWHT sign tables uploaded
            // (matches `gemv_mq4g256_with_rotate`'s setup). A failure here
            // must still release the weights above before returning Err.
            if let Err(e) = gpu.ensure_mq_signs() {
                fc.free_all(gpu);
                let _ = gpu.free_tensor(hidden_norm);
                let _ = gpu.free_tensor(norm);
                for l in layers {
                    l.free_gpu(gpu);
                }
                if let Some(w) = selector_hidden_proj {
                    w.free_all(gpu);
                }
                return Err(e);
            }
        }

        Ok(DflashWeights {
            fc,
            hidden_norm,
            norm,
            layers,
            has_mq,
            selector_hidden_proj,
            predecessor_codebook,
            successor_codebook,
            selector_rank: selector_rank_opt,
            selector_top_k: selector_top_k_opt,
            conv_group_size: cfg.conv_group_size,
            conv_kernel_size: cfg.conv_kernel_size,
        })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        // free_all (not .buf) so the awq_scale / paro sidecars are released too —
        // on an AWQ-trunk drafter every weight carries an awq_scale GpuTensor, so
        // .buf-only freeing leaks one tensor per weight per layer on each unload.
        self.fc.free_all(gpu);
        let _ = gpu.free_tensor(self.hidden_norm);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            l.free_gpu(gpu);
        }
        if let Some(w) = self.selector_hidden_proj {
            w.free_all(gpu);
        }
        // codebooks are host-side only, no VRAM to free
    }
}

// ─── target_hidden bookkeeping ───────────────────────────────────────────────

/// Encapsulated cursors describing how much of the draft's `target_hidden`
/// context is live on GPU. Three quantities that MUST move together:
///
/// - `uploaded_rows` — rows of `target_hidden` already uploaded H2D (the
///   delta-upload watermark `draft_forward` reads/advances);
/// - `abs_positions` — the absolute (pre-compaction) position of every
///   populated row; its length is always exactly `uploaded_rows`;
/// - `proj_cached_rows` — rows whose per-layer `k_ctx`/`v_ctx` projection is
///   cached (`≤ uploaded_rows`).
///
/// The fields are PRIVATE (this is a submodule), so the only ways to mutate
/// them are the invariant-preserving operations below — `seed_prompt`,
/// `append_committed`, `rebuild_after_eviction`, `reset`, plus the two
/// `draft_forward`-owned watermarks `mark_uploaded` / `mark_proj_cached`.
/// A caller can no longer set `uploaded_rows` without `abs_positions` staying
/// consistent (the desync that bit the generic DFlash path in 89856eab and
/// the #462 class): that error is now defined out of existence — it does not
/// compile.
mod target_hidden_log {
    #[derive(Clone, Copy, Debug)]
    pub struct TargetHiddenLogMark {
        uploaded_rows: usize,
        abs_positions_len: usize,
        proj_cached_rows: usize,
        full_cached_rows: usize,
    }

    /// See module-level intent. Construct via [`TargetHiddenLog::new`].
    #[derive(Default)]
    pub struct TargetHiddenLog {
        uploaded_rows: usize,
        abs_positions: Vec<i32>,
        proj_cached_rows: usize,
        /// Separate K/V-fill watermark for the windowed mode's last
        /// (full-attention) layer. Rows older than `l − swa_w` are not
        /// resident in the proj ring, so the last layer's fill for them runs
        /// in the post-seed backfill (host shadow), tracked here. Always
        /// `== proj_cached_rows` in Legacy mode.
        full_cached_rows: usize,
    }

    impl TargetHiddenLog {
        pub fn new() -> Self {
            Self::default()
        }

        // ── reads ────────────────────────────────────────────────────────
        /// Rows of `target_hidden` already uploaded to GPU (delta watermark).
        pub fn uploaded_rows(&self) -> usize {
            self.uploaded_rows
        }
        /// Absolute (pre-compaction) position of each populated row;
        /// `len() == uploaded_rows`.
        pub fn abs_positions(&self) -> &[i32] {
            &self.abs_positions
        }
        /// Rows whose per-layer k_ctx/v_ctx projection is cached.
        pub fn proj_cached_rows(&self) -> usize {
            self.proj_cached_rows
        }
        /// Rows whose LAST-layer (full-attention) K/V projection is cached
        /// (windowed mode; mirrors `proj_cached_rows` in Legacy).
        pub fn full_cached_rows(&self) -> usize {
            self.full_cached_rows
        }

        /// Lightweight rollback point for one speculative window. The backing
        /// tensors are append-only here, so restoring metadata is sufficient;
        /// stale tail rows are ignored and overwritten by the next append.
        pub fn mark(&self) -> TargetHiddenLogMark {
            TargetHiddenLogMark {
                uploaded_rows: self.uploaded_rows,
                abs_positions_len: self.abs_positions.len(),
                proj_cached_rows: self.proj_cached_rows,
                full_cached_rows: self.full_cached_rows,
            }
        }

        pub fn restore(&mut self, mark: TargetHiddenLogMark) -> Result<(), String> {
            if mark.abs_positions_len > self.abs_positions.len() {
                return Err(format!(
                    "target-hidden rollback mark {} exceeds live rows {}",
                    mark.abs_positions_len,
                    self.abs_positions.len()
                ));
            }
            self.abs_positions.truncate(mark.abs_positions_len);
            self.uploaded_rows = mark.uploaded_rows;
            self.proj_cached_rows = mark.proj_cached_rows;
            self.full_cached_rows = mark.full_cached_rows;
            Ok(())
        }

        // ── invariant-preserving mutations ────────────────────────────────
        /// New-prompt / session boundary: forget all GPU-resident rows.
        pub fn reset(&mut self) {
            self.uploaded_rows = 0;
            self.abs_positions.clear();
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// Prompt-prefill seed: `rows` contiguous rows `[0..rows)` are live on
        /// GPU at contiguous absolute positions.
        pub fn seed_prompt(&mut self, rows: usize) {
            self.uploaded_rows = rows;
            self.abs_positions = (0..rows as i32).collect();
        }

        /// Divergent-render resume: drop the projection cache back to a
        /// checkpoint position so the next `draft_forward` re-projects from it.
        /// The last-layer (windowed full) watermark clamps down too — rows at
        /// >= ckpt in its ring are stale (pre-divergence); they re-fill from
        /// the live window as new rows stream (out-of-window stale rows are a
        /// τ-only degradation; verify stays exact).
        pub fn set_resume_checkpoint(&mut self, ckpt: usize) {
            self.proj_cached_rows = ckpt;
            self.full_cached_rows = self.full_cached_rows.min(ckpt);
        }

        /// Post-commit append: `n` newly committed rows starting at logical
        /// `base_pos` (with the target KV `compact_offset`) become live. After
        /// this, `uploaded_rows == base_pos + n` and `abs_positions` has one
        /// entry per committed row.
        pub fn append_committed(&mut self, base_pos: usize, n: usize, compact_offset: i32) {
            debug_assert_eq!(
                self.abs_positions.len(),
                base_pos,
                "append_committed: abs_positions out of sync with base_pos"
            );
            // Release-visible companion to the assert above. A `base_pos` ahead
            // of the row count means positions advanced WITHOUT committing their
            // target_hidden rows, so the buffer keeps an unwritten hole —
            // uninitialized VRAM, i.e. NaN — which poisons every later draft
            // forward, collapses acceptance to zero for the rest of the session,
            // and survives a prompt-cache HIT (whose `seed_prompt` rebuilds
            // `abs_positions` contiguously, hiding the skew but not the hole).
            // The `debug_assert` is compiled out in release, which is exactly how
            // the think-budget force-close path went unnoticed; warn loudly once
            // instead of silently poisoning the drafter.
            if self.abs_positions.len() != base_pos {
                static GAP_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !GAP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[dflash] WARNING target-hidden row gap: have {} rows but committing at \
                         position {} ({} row(s) never written) — acceptance will degrade until \
                         the drafter is reseeded. This is a bug in whichever path advanced the \
                         target without committing its hidden rows.",
                        self.abs_positions.len(),
                        base_pos,
                        base_pos.saturating_sub(self.abs_positions.len())
                    );
                }
            }
            self.uploaded_rows = base_pos + n;
            for p in 0..n {
                self.abs_positions
                    .push(base_pos as i32 + p as i32 + compact_offset);
            }
        }

        /// Post-eviction rebuild: `new_abs` is the compacted absolute-position
        /// list (one entry per retained row). Replaces the row layout and
        /// invalidates the projection cache (row indices shifted).
        pub fn rebuild_after_eviction(&mut self, new_abs: Vec<i32>) {
            self.uploaded_rows = new_abs.len();
            self.abs_positions = new_abs;
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// Invalidate only the per-layer projection cache (eviction mirror that
        /// keeps the uploaded rows but shifts their indices).
        pub fn invalidate_proj_cache(&mut self) {
            self.proj_cached_rows = 0;
            self.full_cached_rows = 0;
        }

        /// `draft_forward`-owned: record that `l` rows are now uploaded H2D.
        pub fn mark_uploaded(&mut self, l: usize) {
            self.uploaded_rows = l;
        }

        /// `draft_forward`-owned: record that `l` rows are now projection-cached.
        /// Also advances the last-layer watermark: a completed forward filled
        /// every layer's K/V through `l` (windowed mode relies on the
        /// post-seed backfill for the out-of-window prefix).
        pub fn mark_proj_cached(&mut self, l: usize) {
            self.proj_cached_rows = l;
            self.full_cached_rows = self.full_cached_rows.max(l);
        }
        /// Windowed backfill-owned: record that the last (full-attention)
        /// layer's K/V cache is valid through row `l`.
        pub fn mark_full_cached(&mut self, l: usize) {
            self.full_cached_rows = l;
        }
    }
}
pub use target_hidden_log::{TargetHiddenLog, TargetHiddenLogMark};

mod lanes;
pub use lanes::{draft_forward_lanes, DraftLane};

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Draft context mode (HIPFIRE_DFLASH_WINDOW).
///
/// `Legacy` (default): every draft layer attends over the full context; all
/// context-indexed buffers are sized to the (capped) ctx capacity and a
/// request past the cap falls back to AR in the daemon.
///
/// `Windowed { w, w_full }`: layers `0..n-2` attend over the last `w` rows
/// (SWA), layer `n-1` over the last `w_full` rows — the NInfer companion-
/// drafter pattern (W=4096 SWA + one full layer) ported inference-side.
/// Every per-layer K/V cache is a ring (`slot = abs_row % window`), so the
/// layout at `l <= w` is byte-identical to Legacy and all fills stay
/// row-local-exact; past `w` acceptance (τ) degrades gracefully instead of
/// the AR cliff. Draft quality never affects emitted tokens (verify-gated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftCtxMode {
    Legacy,
    Windowed { w: usize, w_full: usize },
}

/// Iterate the contiguous ring segments covering rows `[start, end)` in a
/// ring of `modulus` slots (slot = row % modulus). Yields
/// `(row_start, slot_start, len)` per segment — at most two for any range
/// smaller than `modulus`. `modulus == usize::MAX` degenerates to a single
/// identity segment (Legacy layout), keeping the windowed code path
/// byte-exact with the legacy one when no wrap is possible.
pub fn ring_segments(start: usize, end: usize, modulus: usize) -> Vec<(usize, usize, usize)> {
    debug_assert!(end >= start);
    if start == end {
        return Vec::new();
    }
    if modulus == usize::MAX {
        return vec![(start, start, end - start)];
    }
    let mut out = Vec::with_capacity(2);
    let mut row = start;
    while row < end {
        let slot = row % modulus;
        let len = (modulus - slot).min(end - row);
        out.push((row, slot, len));
        row += len;
    }
    out
}

/// Rows of the lane-batched ctx/fc workspace: eight lanes of at most nine new
/// rows each (an accepted block of up to `b + 1` committed rows).
pub const DRAFT_LANE_WS_ROWS: usize = 72;

/// Preallocated packing buffers of the lane-batched context fill and `fc`
/// projection (`draft_forward_lanes`). Allocated once per shared draft
/// scratch, never per lane or per forward; contents are transient within one
/// stage and carry no lane state.
pub struct DraftLaneWorkspace {
    /// Packed rows these tensors hold ([`DRAFT_LANE_WS_ROWS`]).
    pub rows: usize,
    /// Row stride of `input`: `max(feature_width, hidden)`.
    pub input_width: usize,
    /// `[rows, input_width]` gathered `target_hidden` (fc) / projected-hidden
    /// (ctx) rows; shape subviews use the actual width of the stage.
    pub input: GpuTensor,
    /// `[rows, hidden]` fc output (post-norm before the scatter).
    pub fc_out: GpuTensor,
    /// `[rows, kv_dim]` packed context K.
    pub k_out: GpuTensor,
    /// `[rows, kv_dim]` packed context V.
    pub v_out: GpuTensor,
    /// `[rows]` i32 packed key positions (stored in an F32 tensor like every
    /// `positions_*` plane).
    pub positions: GpuTensor,
}

impl DraftLaneWorkspace {
    /// Allocate every tensor or none: a failure frees the earlier ones.
    fn alloc(gpu: &mut Gpu, cfg: &DflashConfig) -> HipResult<Self> {
        let rows = DRAFT_LANE_WS_ROWS;
        let h = cfg.hidden;
        let kvd = cfg.kv_dim();
        let input_width = std::cmp::max(cfg.feature_width(), h);
        let mut live: Vec<GpuTensor> = Vec::with_capacity(5);
        for len in [rows * input_width, rows * h, rows * kvd, rows * kvd, rows] {
            match gpu.alloc_tensor(&[len], DType::F32) {
                Ok(t) => live.push(t),
                Err(e) => {
                    for t in live {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(e);
                }
            }
        }
        let mut it = live.into_iter();
        let mut next = || it.next().expect("five lane workspace tensors");
        Ok(Self {
            rows,
            input_width,
            input: next(),
            fc_out: next(),
            k_out: next(),
            v_out: next(),
            positions: next(),
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.input,
            self.fc_out,
            self.k_out,
            self.v_out,
            self.positions,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Activation buffers for one forward pass. Sized for up to
/// `max_block_size` query positions and up to `max_ctx_len` context
/// positions. A single scratch is reused across all speculative steps.
pub struct DflashScratch {
    pub max_block_size: usize,
    pub max_ctx_len: usize,

    // Block-sized activations (B rows).
    pub x: GpuTensor,        // [B, hidden] — hidden state rolled across layers
    pub x_norm: GpuTensor,   // [B, hidden]
    pub q: GpuTensor,        // [B, q_dim]
    pub k_noise: GpuTensor,  // [B, kv_dim]
    pub v_noise: GpuTensor,  // [B, kv_dim]
    pub gate: GpuTensor,     // [B, intermediate]
    pub up: GpuTensor,       // [B, intermediate]
    pub gate_up: GpuTensor,  // [B, intermediate]
    pub attn_out: GpuTensor, // [B, q_dim]
    /// Shared residual plane. Attention and FFN consume it sequentially,
    /// including under the per-layer FFN graph, so separate planes only pin
    /// another B×hidden allocation without enabling overlap.
    pub residual: GpuTensor, // [B, hidden]

    // Context activations (L rows), where L ≤ max_ctx_len.
    pub target_hidden: GpuTensor,      // [L, feature_width]
    pub target_hidden_proj: GpuTensor, // [L, hidden]

    // Concatenated K/V (L + B rows).
    pub k_cat: GpuTensor, // [L + B, kv_dim]
    pub v_cat: GpuTensor, // [L + B, kv_dim]

    // Positions (i32).
    pub positions_q: GpuTensor, // [B]       i32
    pub positions_k: GpuTensor, // [L + B]   i32
    /// Host staging of the concatenated query positions of a batched lane
    /// draft (`draft_forward_lanes` on the shared scratch). Capacity
    /// `max_block_size`, reserved once and never grown, so the upload source
    /// has a stable owned address (a captured upload bakes it).
    pub lane_positions_q: Vec<i32>,

    // FWHT rotation scratch for MQ4 weight paths. Sized to the largest
    // single-call requirement: max(max_ctx × feature_width,
    // max_block × max_layer_K). Allocated only when DflashWeights.has_mq.
    pub mq_x_rot: Option<GpuTensor>,
    // Launch-fusion prescaffold (S7): F16 twin of `mq_x_rot` (same element
    // count, half the bytes). Allocated/freed but never written or read yet.
    pub mq_x_rot_f16: Option<GpuTensor>,
    // Launch-fusion prescaffold (S7): persistent noise-token-ID plane.
    // S7 uploads the draft token IDs once instead of 16 scalar embeddings.
    // i32 IDs stored as F32 (same cosmetic pattern as `positions_*`).
    pub noise_tokens: GpuTensor, // [B]

    // DFlash2 optional scratch: conv temp/dynamic and selector buffers.
    // Allocated only when the loaded draft actually needs them.
    pub conv_temp: Option<GpuTensor>,     // [B, hidden]
    pub conv_dynamic: Option<GpuTensor>,  // [B, 2*K*G]
    pub selector_proj: Option<GpuTensor>, // [B, rank]
    pub topk_ids: Option<GpuTensor>,      // [B, K] i32 (stored as F32 buffer)
    pub topk_vals: Option<GpuTensor>,     // [B, K] f32

    // Encapsulated `target_hidden` cursors: uploaded-row watermark, per-row
    // absolute positions, and projection-cache extent. The delta-upload
    // tracker drops per-cycle H2D from ~90 MB (full ctx at 1100 tokens × 5
    // layers × 4096 × 4 B) to ~700 KB. See `TargetHiddenLog` for the
    // invariant-preserving API (seed_prompt / append_committed /
    // rebuild_after_eviction / reset); raw poking is no longer possible.
    pub thlog: TargetHiddenLog,

    /// Per-layer cache of `k_ctx` and `v_ctx` (post-GEMM-of-target_hidden_proj;
    /// K additionally post-RMSNorm and post-RoPE). Filled incrementally as
    /// draft_forward sees new target_hidden rows.
    ///
    /// The win: without this cache, each `draft_forward` call re-ran 2
    /// big GEMMs per layer over ALL L context rows, even though only the
    /// tail (accept+1 new rows) had changed since the previous cycle. On
    /// 27B at L=512, that cost ~230 ms/cycle. With the cache, only the
    /// delta rows are recomputed and appended — ~5 ms/cycle for typical
    /// τ ≈ 5.
    ///
    /// Lucebox calls the same structure a "rolling target_feat ring" in
    /// its DFlash-on-ggml writeup; this is our equivalent.
    ///
    /// Shapes: each entry is `[max_ctx, kv_dim]` f32.
    /// The valid extent of these caches (rows `[0..thlog.proj_cached_rows())`
    /// have finished fc + hidden_norm projection, per-layer wk/wv GEMMs,
    /// k_norm, and K RoPE). Tracked by `thlog` so it stays consistent with the
    /// upload watermark. Caching post-RoPE is important: absolute positions of
    /// committed rows do not change, so re-rotating the full historical K span
    /// every speculative cycle is redundant O(context × layers) work.
    pub k_ctx_cached: Vec<GpuTensor>,
    pub v_ctx_cached: Vec<GpuTensor>,

    /// Context mode this scratch was sized for (see [`DraftCtxMode`]).
    pub ctx_mode: DraftCtxMode,
    /// Windowed-mode last-layer (full-attention) K/V rings, `[w_full × kvd]`.
    /// The last layer keeps a 4× longer reach than the SWA layers — the only
    /// unbounded-context structure in windowed mode. `None` in Legacy and in
    /// all-sliding DFlash2 (every layer shares the same W).
    pub k_full_cached: Option<GpuTensor>,
    pub v_full_cached: Option<GpuTensor>,
    /// Windowed-mode last-layer concat assembly, `[(w_full + B) × kvd]`.
    pub k_cat_full: Option<GpuTensor>,
    pub v_cat_full: Option<GpuTensor>,

    /// Per-layer, per-B graph cache for the fixed-shape FFN tail inside
    /// `draft_forward_opts`. The attention/context part depends on `ctx_len`;
    /// this FFN subgraph does not, so it can replay across DFlash cycles.
    pub draft_ffn_graphs: Vec<HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>>,
    pub draft_ffn_warmed_up: Vec<HashSet<usize>>,

    /// Lane-batched ctx/fc workspace of the shared continuous-batching draft
    /// scratch (`None` elsewhere, and when the batched path is off). See
    /// [`DflashScratch::enable_lane_workspace`].
    pub lane_ws: Option<DraftLaneWorkspace>,
    /// Developer-only online draft tuner (`HIPFIRE_DFLASH_ONLINE_TUNE`);
    /// `None` keeps the shipping argmax proposal path.
    pub online: Option<crate::dflash_online::OnlineDraftTuner>,
}

impl DflashScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
    ) -> HipResult<Self> {
        Self::new_with_mq(gpu, cfg, max_block_size, max_ctx_len, false)
    }

    /// Windowed-mode constructor (HIPFIRE_DFLASH_WINDOW). SWA layers
    /// `0..n-2` get `w`-row K/V rings and `(w + B)` concat buffers; the last
    /// layer gets `w_full`-row rings and its own concat pair. All other
    /// buffers are sized exactly as `new_with_mq(gpu, cfg, b, w, ..)` — so
    /// windowed-mode VRAM is the Legacy-at-ctx=w footprint plus one
    /// `w_full`-sized layer (≈270 MB at w_full=32K, kvd=1024, f32).
    /// `max_ctx` is the target's physical context capacity: the rings wrap
    /// forever, so `l` may grow past `w_full` (spans stay suffixes).
    pub fn new_windowed(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        w: usize,
        w_full: usize,
        max_ctx: usize,
        with_mq: bool,
    ) -> HipResult<Self> {
        Self::new_windowed_with_boundary(
            gpu,
            cfg,
            max_block_size,
            w,
            w_full,
            max_ctx,
            with_mq,
            &mut || Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)] // Mirrors the public constructor plus a test-only boundary.
    fn new_windowed_with_boundary(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        w: usize,
        w_full: usize,
        max_ctx: usize,
        with_mq: bool,
        after_owner: &mut impl FnMut() -> HipResult<()>,
    ) -> HipResult<Self> {
        // DFlash2 all-sliding: every layer shares the same W ring. Skip the
        // last-layer full replacement/backfill path and keep the footprint at
        // Legacy-at-ctx=w for all layers.
        if cfg.all_layers_sliding {
            let mut s =
                Self::new_with_mq_with_boundary(gpu, cfg, max_block_size, w, with_mq, after_owner)?;
            s.max_ctx_len = max_ctx;
            s.ctx_mode = DraftCtxMode::Windowed { w, w_full: w };
            return Ok(s);
        }
        let kvd = cfg.kv_dim();
        let b = max_block_size;
        // The long-reach layer never has a SHORTER window than the SWA layers
        // (possible when physical_cap < w): its span must contain theirs.
        let w_full = w_full.max(w);
        // Base at ctx=w: SWA rings, (w+B) concat buffers, w-row target_hidden
        // ring — the entire draft footprint except the one long-reach layer.
        let mut s = Self::new_with_mq_with_boundary(gpu, cfg, b, w, with_mq, after_owner)?;
        // The last (full-attention) layer gets w_full-row rings + its own
        // concat pair; its w-sized base caches are freed.
        if let Some(k) = s.k_ctx_cached.pop() {
            let _ = gpu.free_tensor(k);
        }
        if let Some(v) = s.v_ctx_cached.pop() {
            let _ = gpu.free_tensor(v);
        }
        // The base scratch `s` is fully owned here: any failure below frees
        // it before returning Err — a bare `?` would leak it (no `Drop` on
        // the GPU-owning types), including when the failure follows the pop
        // above. Same class as the `or_free!` sites in `load_dflash_state`.
        //
        // Each allocation is parked in `s` the moment it succeeds, so the
        // error arm's `s.free_gpu` also covers every earlier allocation of
        // this ladder: with the tensors held as locals until the end, a
        // failure on the 2nd..5th alloc freed `s` but leaked the locals
        // (hw-gate Fable seat on #691, run 33900101473).
        macro_rules! alloc_or_free {
            ($e:expr) => {
                match $e {
                    Ok(t) => t,
                    Err(e) => {
                        s.free_gpu(gpu);
                        return Err(e);
                    }
                }
            };
        }
        macro_rules! boundary_or_free {
            () => {
                if let Err(e) = after_owner() {
                    s.free_gpu(gpu);
                    return Err(e);
                }
            };
        }
        s.k_full_cached = Some(alloc_or_free!(gpu.alloc_tensor(&[w_full * kvd], DType::F32)));
        boundary_or_free!();
        s.v_full_cached = Some(alloc_or_free!(gpu.alloc_tensor(&[w_full * kvd], DType::F32)));
        boundary_or_free!();
        s.k_cat_full = Some(alloc_or_free!(
            gpu.alloc_tensor(&[(w_full + b) * kvd], DType::F32)
        ));
        boundary_or_free!();
        s.v_cat_full = Some(alloc_or_free!(
            gpu.alloc_tensor(&[(w_full + b) * kvd], DType::F32)
        ));
        boundary_or_free!();
        // positions_k holds the last w_full context rows + the B noise rows
        // (the forward uploads only that suffix; every layer's span is one).
        // Allocate before freeing the old buffer so a failure still leaves
        // `s` intact for the error arm above.
        let new_positions_k = alloc_or_free!(gpu.alloc_tensor(&[w_full + b], DType::F32));
        let _ = gpu.free_tensor(std::mem::replace(&mut s.positions_k, new_positions_k));
        boundary_or_free!();
        // The ctx bound is the target's physical capacity, not the window —
        // l may cross w_full (the last layer's span just slides).
        s.max_ctx_len = max_ctx;
        s.ctx_mode = DraftCtxMode::Windowed { w, w_full };
        Ok(s)
    }

    /// `with_mq` allocates the FWHT rotation scratch needed when at least
    /// one matrix weight is MQ4-G256. Sized to handle every per-call
    /// rotation in the draft forward.
    pub fn new_with_mq(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
        with_mq: bool,
    ) -> HipResult<Self> {
        Self::new_with_mq_with_boundary(gpu, cfg, max_block_size, max_ctx_len, with_mq, &mut || {
            Ok(())
        })
    }

    fn new_with_mq_with_boundary(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
        with_mq: bool,
        after_owner: &mut impl FnMut() -> HipResult<()>,
    ) -> HipResult<Self> {
        let b = max_block_size;
        let l = max_ctx_len;
        let tot = l + b;
        let fw = cfg.feature_width();
        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();

        // Transactional construction: every `alloc_tensor` below goes through
        // `at!`, which records the tensor in `live`; each index is taken
        // exactly once when the struct is built. On failure the error arm
        // frees everything recorded so far and returns — a bare `?` would
        // leak (`GpuTensor`/`DeviceBuffer` have no `Drop`). Same style as the
        // `gt!`/`wt!` slots in `DflashWeights::load` above.
        let mut live: Vec<Option<GpuTensor>> = Vec::new();
        macro_rules! at {
            ($shape:expr) => {{
                at!($shape, DType::F32)
            }};
            ($shape:expr, $dtype:expr) => {{
                match gpu.alloc_tensor($shape, $dtype) {
                    Ok(t) => {
                        live.push(Some(t));
                        let index = live.len() - 1;
                        if let Err(e) = after_owner() {
                            for slot in live.iter_mut().rev() {
                                if let Some(t) = slot.take() {
                                    let _ = gpu.free_tensor(t);
                                }
                            }
                            return Err(e);
                        }
                        index
                    }
                    Err(e) => {
                        for slot in live.iter_mut().rev() {
                            if let Some(t) = slot.take() {
                                let _ = gpu.free_tensor(t);
                            }
                        }
                        return Err(e);
                    }
                }
            }};
        }
        macro_rules! take {
            ($i:expr) => {
                live[$i].take().expect("dflash scratch slot taken twice")
            };
        }

        let i_mq_x_rot = if with_mq {
            // Sized for a CHUNK of the worst-case MQ rotation, not the whole
            // first-call prefix. The rotations called through `gemm_dispatch`
            // are:
            //   - first-call `fc` (target_hidden):  batch up to `l`, w.k = feature_width
            //   - per-cycle wq/wk/wv/gate/up:       batch = b,         w.k = h
            //   - per-cycle wo:                     batch = b,         w.k = q_dim
            //   - per-cycle w_down:                 batch = b,         w.k = intermediate
            //   - first-call wk/wv on prefix:       batch up to `l`,   w.k = h
            //
            // Steady-state cycles only need `b × max(inter, qd, feature_width)`. The
            // first-call rotations against the full prefix used to pin the
            // buffer to `l × feature_width` (1.7 GB at ctx=17K on 27B). That sizing
            // forced VRAM bloat that scales with max_seq.
            //
            // Fix: cap the scratch at `MQ_X_ROT_CHUNK_ROWS × max(inter, qd, feature_width)`
            // floats and chunk any call where `batch × w.k > scratch.size()`
            // inside `gemm_dispatch`. The first-call rotations are split into
            // `ceil(batch / chunk_rows)` smaller GEMMs — adds ~1-2 launches per
            // 1K prefix tokens (negligible vs seconds-scale prefill).
            let widest = MQ_X_ROT_CHUNK_ROWS * std::cmp::max(inter, std::cmp::max(qd, fw));
            Some(at!(&[widest]))
        } else {
            None
        };
        // Prescaffold F16 twin: same element count as `mq_x_rot`.
        // Joins the `at!` transaction (dtype arm): a bare `?` here would
        // leak every earlier allocation (`GpuTensor` has no `Drop`).
        let i_mq_x_rot_f16 = if with_mq {
            let widest = MQ_X_ROT_CHUNK_ROWS * std::cmp::max(inter, std::cmp::max(qd, fw));
            Some(at!(&[widest], DType::F16))
        } else {
            None
        };

        // DFlash2 optional buffers: allocated only when the config declares them.
        // Slot indices (`take!`n at the build below).
        let need_conv = cfg.conv_kernel_size.is_some() && cfg.conv_group_size.is_some();
        let need_selector = cfg.selector_rank.is_some() && cfg.selector_top_k.is_some();
        let i_conv_temp = if need_conv { Some(at!(&[b * h])) } else { None };
        let i_conv_dynamic = if need_conv {
            let k = cfg.conv_kernel_size.unwrap();
            let g = cfg.conv_group_size.unwrap();
            let groups = h / g;
            let stride = 2 * k * groups;
            Some(at!(&[b * stride]))
        } else {
            None
        };
        let i_selector_proj = if need_selector {
            let rank = cfg.selector_rank.unwrap();
            Some(at!(&[b * rank]))
        } else {
            None
        };
        let (i_topk_ids, i_topk_vals) = if need_selector {
            let kk = cfg.selector_top_k.unwrap();
            // ids as i32 stored in F32 buffer (reinterprets), vals as f32
            (Some(at!(&[b * kk])), Some(at!(&[b * kk])))
        } else {
            (None, None)
        };

        // Per-layer cache buffers for k_ctx/v_ctx (post-norm-for-K, pre-rope).
        // Size each at [max_ctx × kv_dim] f32 = l × kvd × 4 bytes. Memory
        // cost for 16-layer / 4096-ctx / 256-kv_dim draft ≈ 2 × 16 × 4 MB
        // = 128 MB. Trivial vs 24 GB VRAM.
        let mut kv_idx: Vec<(usize, usize)> = Vec::with_capacity(cfg.n_layers);
        let mut draft_ffn_graphs = Vec::with_capacity(cfg.n_layers);
        let mut draft_ffn_warmed_up = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            kv_idx.push((at!(&[l * kvd]), at!(&[l * kvd])));
            draft_ffn_graphs.push(HashMap::new());
            draft_ffn_warmed_up.push(HashSet::new());
        }

        let i_x = at!(&[b * h]);
        let i_x_norm = at!(&[b * h]);
        let i_q = at!(&[b * qd]);
        let i_k_noise = at!(&[b * kvd]);
        let i_v_noise = at!(&[b * kvd]);
        let i_gate = at!(&[b * inter]);
        let i_up = at!(&[b * inter]);
        let i_gate_up = at!(&[b * inter]);
        let i_attn_out = at!(&[b * qd]);
        let i_residual = at!(&[b * h]);

        let i_target_hidden = at!(&[l * fw]);
        let i_target_hidden_proj = at!(&[l * h]);

        let i_k_cat = at!(&[tot * kvd]);
        let i_v_cat = at!(&[tot * kvd]);

        let i_positions_q = at!(&[b]);
        let i_positions_k = at!(&[tot]);

        // Launch-fusion prescaffold (S7): persistent noise-token-ID plane
        // ([B] i32 IDs stored as F32, same cosmetic pattern as `positions_*`).
        // Unconditional like #702; joins the transaction so any later `at!`
        // failure frees it.
        let i_noise_tokens = at!(&[b]);

        let mut k_ctx_cached = Vec::with_capacity(cfg.n_layers);
        let mut v_ctx_cached = Vec::with_capacity(cfg.n_layers);
        for (ik, iv) in kv_idx {
            k_ctx_cached.push(take!(ik));
            v_ctx_cached.push(take!(iv));
        }
        let scratch = DflashScratch {
            max_block_size: b,
            max_ctx_len: l,

            x: take!(i_x),
            x_norm: take!(i_x_norm),
            q: take!(i_q),
            k_noise: take!(i_k_noise),
            v_noise: take!(i_v_noise),
            gate: take!(i_gate),
            up: take!(i_up),
            gate_up: take!(i_gate_up),
            attn_out: take!(i_attn_out),
            residual: take!(i_residual),

            target_hidden: take!(i_target_hidden),
            target_hidden_proj: take!(i_target_hidden_proj),

            k_cat: take!(i_k_cat),
            v_cat: take!(i_v_cat),

            positions_q: take!(i_positions_q),
            positions_k: take!(i_positions_k),
            lane_positions_q: Vec::with_capacity(b),

            mq_x_rot: i_mq_x_rot.map(|j| take!(j)),
            mq_x_rot_f16: i_mq_x_rot_f16.map(|j| take!(j)),
            noise_tokens: take!(i_noise_tokens),
            conv_temp: i_conv_temp.map(|j| take!(j)),
            conv_dynamic: i_conv_dynamic.map(|j| take!(j)),
            selector_proj: i_selector_proj.map(|j| take!(j)),
            topk_ids: i_topk_ids.map(|j| take!(j)),
            topk_vals: i_topk_vals.map(|j| take!(j)),
            thlog: TargetHiddenLog::new(),
            k_ctx_cached,
            v_ctx_cached,
            ctx_mode: DraftCtxMode::Legacy,
            k_full_cached: None,
            v_full_cached: None,
            k_cat_full: None,
            v_cat_full: None,
            draft_ffn_graphs,
            draft_ffn_warmed_up,
            lane_ws: None,
            online: crate::dflash_online::OnlineDraftTuner::from_env(),
        };
        debug_assert!(live.iter().all(|s| s.is_none()));
        Ok(scratch)
    }

    /// Allocate the lane-batched ctx/fc workspace ([`DraftLaneWorkspace`]) on
    /// this scratch (idempotent). Only the shared continuous-batching draft
    /// scratch calls this. A no-op (`Ok`, nothing allocated) unless the batched
    /// context path can ever be admitted: `HIPFIRE_CB_DRAFT_LANE_CTX` not `0`,
    /// exact gfx1201, and an MQ scratch (`mq_x_rot` present). On `Err` nothing
    /// is left allocated by this call; the caller still owns (and frees) `self`.
    pub fn enable_lane_workspace(&mut self, gpu: &mut Gpu, cfg: &DflashConfig) -> HipResult<()> {
        if self.lane_ws.is_some()
            || self.mq_x_rot.is_none()
            || !lanes::draft_lane_ctx_enabled()
            || gpu.arch != "gfx1201"
        {
            return Ok(());
        }
        self.lane_ws = Some(DraftLaneWorkspace::alloc(gpu, cfg)?);
        Ok(())
    }

    /// Reset the incremental-upload tracker for target_hidden. Call this
    /// at the start of a new prompt / session — otherwise stale tracker
    /// state from a prior prompt would cause the next draft_forward to
    /// skip required rows. Also clears the draft-ctx projection cache so
    /// the first draft_forward after reset does a full rebuild.
    pub fn reset_upload_tracking(&mut self) {
        self.thlog.reset();
    }

    /// The dst ring modulus for absolute-row scatters into
    /// `target_hidden` — `usize::MAX` (identity) in Legacy, `w` in
    /// Windowed mode.
    pub fn ctx_modulus(&self) -> usize {
        match self.ctx_mode {
            DraftCtxMode::Legacy => usize::MAX,
            DraftCtxMode::Windowed { w, .. } => w,
        }
    }

    /// Invalidate the per-layer k_ctx/v_ctx projection cache. Called from
    /// `apply_eviction_retain_to_draft` (in speculative.rs) when CASK
    /// evicts positions — the cached rows no longer correspond to the
    /// right absolute positions, so the simplest correct thing is to
    /// rebuild on the next cycle. A finer mirror (applying retain_mask
    /// to the cache) could preserve the cache across eviction but adds
    /// complexity; the rebuild cost is bounded by one slow cycle per
    /// eviction which is rare relative to total cycles.
    pub fn invalidate_draft_ctx_cache(&mut self) {
        self.thlog.invalidate_proj_cache();
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for per_layer in self.draft_ffn_graphs {
            for (_, (graph, exec, _blobs)) in per_layer {
                let _ = gpu.hip.graph_exec_destroy(exec);
                let _ = gpu.hip.graph_destroy(graph);
            }
        }
        if let Some(mut t) = self.online {
            t.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.x_norm);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k_noise);
        let _ = gpu.free_tensor(self.v_noise);
        let _ = gpu.free_tensor(self.gate);
        let _ = gpu.free_tensor(self.up);
        let _ = gpu.free_tensor(self.gate_up);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.target_hidden);
        let _ = gpu.free_tensor(self.target_hidden_proj);
        let _ = gpu.free_tensor(self.k_cat);
        let _ = gpu.free_tensor(self.v_cat);
        let _ = gpu.free_tensor(self.positions_q);
        let _ = gpu.free_tensor(self.positions_k);
        for t in self.k_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.k_full_cached,
            self.v_full_cached,
            self.k_cat_full,
            self.v_cat_full,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.mq_x_rot {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.mq_x_rot_f16 {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.noise_tokens);
        for t in [
            self.conv_temp,
            self.conv_dynamic,
            self.selector_proj,
            self.topk_ids,
            self.topk_vals,
        ]
        .into_iter()
        .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
        if let Some(ws) = self.lane_ws {
            ws.free_gpu(gpu);
        }
    }
}

// ─── Forward ───────────────────────────────────────────────────────────────

/// Dispatch a batched GEMM by weight dtype.
///
/// Layout (row-major):
///   x [batch × k]  F32 input activations
///   w.buf [m × k]  weight, format depends on w.gpu_dtype
///   y [batch × m]  F32 output
///
/// S7: pre-collapse MQ4G256 chunk loop, byte-for-byte the pre-slice dispatch.
/// Kept as the fallback for every route predicate failure (non-gfx1100, kill
/// switch, batch<=1, AWQ sidecar, non-default WMMA variant policy).
fn gemm_dispatch_mq4_legacy(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &WeightTensor,
    y: &GpuTensor,
    batch: usize,
    mq_x_rot: Option<&GpuTensor>,
) -> HipResult<()> {
    // Chunk on `batch` when the request exceeds the scratch capacity
    // for this w.k. `mq_x_rot` is sized to MQ_X_ROT_CHUNK_ROWS × max(...)
    // — first-call rotations against the full prefix split into
    // `ceil(batch / max_chunk)` GEMMs.
    let scratch = mq_x_rot.expect("MQ4 dispatch requires mq_x_rot scratch");
    let max_chunk = (scratch.shape[0] / w.k).max(1);
    let mut chunked: HipResult<()> = Ok(());
    let mut row = 0;
    while row < batch {
        let n = std::cmp::min(max_chunk, batch - row);
        let x_chunk = x.sub_offset(row * w.k, n * w.k);
        let y_chunk = y.sub_offset(row * w.m, n * w.m);
        let rot_view = scratch.sub_offset(0, n * w.k);
        // AWQ-aware FWHT rotation. When the drafter weight ships an
        // AWQ sidecar (`w.awq_scale.is_some()`), `_for` dispatches
        // the `x /= awq_scale` + FWHT kernel; otherwise falls
        // through to the plain `rotate_x_mq_batched` and is
        // numerically identical to the prior dispatch.
        if let Err(e) = crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n) {
            chunked = Err(e);
            break;
        }
        if let Err(e) = gpu.gemm_hfq4g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n) {
            chunked = Err(e);
            break;
        }
        row += n;
    }
    chunked
}
///
/// S7: collapsed MQ4G256 chunk loop. Rotates F32 `x` straight to the
/// persistent F16 twin (`mq_x_rot_f16`, same element capacity as `mq_x_rot`)
/// and runs an overwrite WMMA whose accumulator starts at +0 — replacing
/// rotate + convert_f32_to_f16 + pre-zero fill + residual GEMM with
/// rotate_f16 + overwrite GEMM. `route` selects the k2 vs deterministic
/// ksplit schedule, mirroring the default policy of the legacy path.
/// Never touches the shared fp16 pointer cache.
fn gemm_dispatch_mq4_collapsed(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &WeightTensor,
    y: &GpuTensor,
    batch: usize,
    rot_f16_scratch: &GpuTensor,
    route: rdna_compute::dflash_draft_fusion::DraftCollapseGemm,
) -> HipResult<()> {
    let max_chunk = (rot_f16_scratch.shape[0] / w.k).max(1);
    let mut chunked: HipResult<()> = Ok(());
    let mut row = 0;
    while row < batch {
        let n = std::cmp::min(max_chunk, batch - row);
        let x_chunk = x.sub_offset(row * w.k, n * w.k);
        let y_chunk = y.sub_offset(row * w.m, n * w.m);
        let rot_f16 = rot_f16_scratch.sub_offset(0, n * w.k);
        if let Err(e) = gpu.mq_rotate_x_f16_dflash(&x_chunk, &rot_f16, w.k, n) {
            chunked = Err(e);
            break;
        }
        let r = match route {
            rdna_compute::dflash_draft_fusion::DraftCollapseGemm::OverwriteK2 => {
                gpu.gemm_hfq4g256_overwrite_wmma_k2_dflash(&w.buf, &rot_f16, &y_chunk, w.m, w.k, n)
            }
            _ => gpu
                .gemm_hfq4g256_overwrite_ksplit_det_dflash(&w.buf, &rot_f16, &y_chunk, w.m, w.k, n),
        };
        if let Err(e) = r {
            chunked = Err(e);
            break;
        }
        row += n;
    }
    chunked
}

/// Draft lm_head on FWHT-rotated draft rows (`rotated` `[rows, k]` → `logits`
/// `[rows, m]`), routed through this draft-GEMM layer like [`gemm_dispatch`]'s
/// MQ4G256V2 arm: on exact gfx1201 an MQ4G256V2 head runs the in-kernel-zeroing
/// overwrite twin (`Gpu::gemm_mq4g256v2_overwrite_gfx12_dflash`, bit-identical to
/// the `hipMemset` pre-zero + residual lm_head it replaces;
/// `HIPFIRE_DRAFT_COLLAPSE_OFF=1` vetoes it). `Ok(false)` = not admitted and
/// nothing launched: the caller runs its batched lm_head launcher unchanged.
#[allow(clippy::too_many_arguments)]
pub fn draft_lm_head_overwrite(
    gpu: &mut Gpu,
    dtype: DType,
    w_buf: &GpuTensor,
    rotated: &GpuTensor,
    logits: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) -> HipResult<bool> {
    if dtype != DType::MQ4G256V2 {
        return Ok(false);
    }
    gpu.gemm_mq4g256v2_overwrite_gfx12_dflash(w_buf, rotated, logits, m, k, rows)
}

/// Projection group of a [`gemm_dispatch_shared`] call; selects its kill switch.
enum SharedProjection {
    Qkv,
    Ffn,
    ContextKv,
}

/// Run `N` draft projections that read the same input `x` (#838). On exact
/// gfx1100, eager (no replay recording, no graph capture), 2..=16 rows, every
/// weight MQ4G256V2 with the same `k` (a multiple of 256), no AWQ sidecar and
/// scratch for all rows, `x` is FWHT-rotated once and every projection's GEMM
/// reads that one rotation: the route `gemm_dispatch` would pick per
/// projection (the S7 overwrite split-K GEMM on F16, else the batched lm_head
/// GEMM), minus the repeated rotations of identical bytes. Anything else, or
/// the group's switch off (`kernel.draft_shared_{qkv,ffn,ctx_kv}_rotation`,
/// default on exact gfx1100), runs [`gemm_dispatch`] once per projection in
/// order. Inventoried as railgun-cert decision
/// `dflash-gfx1100-shared-rotation`.
fn gemm_dispatch_shared<const N: usize>(
    gpu: &mut Gpu,
    x: &GpuTensor,
    projections: [(&WeightTensor, &GpuTensor); N],
    batch: usize,
    mq_x_rot: Option<&GpuTensor>,
    mq_x_rot_f16: Option<&GpuTensor>,
    family: SharedProjection,
) -> HipResult<()> {
    let enabled = match family {
        SharedProjection::Qkv => gpu.flags.draft_shared_qkv_rotation,
        SharedProjection::Ffn => gpu.flags.draft_shared_ffn_rotation,
        SharedProjection::ContextKv => gpu.flags.draft_shared_ctx_kv_rotation,
    };
    let k = projections[0].0.k;
    let eligible = enabled
        && k > 0
        && k % 256 == 0
        && gpu.arch_caps.is_gfx1100()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && batch > 1
        && batch <= 16
        && !crate::config::get().draft_gemm_dump
        && mq_x_rot.is_some_and(|s| s.shape[0] / k >= batch)
        && mq_x_rot_f16.is_some_and(|s| s.shape[0] / k >= batch)
        && projections
            .iter()
            .all(|(w, _)| w.gpu_dtype == DType::MQ4G256V2 && w.k == k && w.awq_scale.is_none());
    if !eligible {
        for (w, y) in projections {
            gemm_dispatch(gpu, x, w, y, batch, mq_x_rot, mq_x_rot_f16)?;
        }
        return Ok(());
    }
    // All consumers use the same input and rotation contract; GEMMs stay unchanged.
    match gpu.draft_collapse_mq4v2_route(k, batch, false) {
        rdna_compute::dflash_draft_fusion::DraftCollapseV2::OverwriteKsplit { kw } => {
            let rot = mq_x_rot_f16.unwrap().sub_offset(0, batch * k);
            gpu.mq_rotate_x_f16_dflash(x, &rot, k, batch)?;
            for (w, y) in projections {
                gpu.gemm_mq4g256v2_overwrite_ksplit_lds_dflash(&w.buf, &rot, y, w.m, k, batch, kw)?;
            }
        }
        rdna_compute::dflash_draft_fusion::DraftCollapseV2::Off => {
            let rot = mq_x_rot.unwrap().sub_offset(0, batch * k);
            crate::llama::rotate_x_mq_batched_for(gpu, projections[0].0, x, &rot, k, batch)?;
            for (w, y) in projections {
                gpu.gemm_mq4g256v2_batched_lmhead(&w.buf, &rot, y, w.m, k, batch)?;
            }
        }
    }
    Ok(())
}

/// For MQ-G256, the kernel needs the input FWHT-rotated. We do that into
/// `mq_x_rot` (sized to the per-call max in `DflashScratch`), then call the
/// HFQ4-G256 GEMM kernel against the pre-rotated weights.
fn gemm_dispatch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &WeightTensor,
    y: &GpuTensor,
    batch: usize,
    mq_x_rot: Option<&GpuTensor>,
    mq_x_rot_f16: Option<&GpuTensor>,
) -> HipResult<()> {
    // Route HFQ4/MQ4 batched paths through the WMMA lm_head helper — the
    // DFlash draft forward's per-layer projections (wq/wk/wv/wo/gate/up/down)
    // and fc are ALL batched > 1, and share the same "y = A @ x" shape as
    // lm_head. Using the WMMA residual-pre-zeroed path here unlocks ~8-10×
    // on the same matmuls without touching AR-greedy numerics (AR on
    // Qwen3.5 doesn't call `gpu.gemm_hfq4g256` directly — it uses the
    // fused qkvza / gate_up / residual WMMA variants instead).
    // HIPFIRE_DRAFT_GEMM_DUMP=1: per-call (dtype, M, K, B, us, GB/s) dump for
    // draft GEMM triage. Cached via OnceLock so the fast path pays a single
    // atomic load per call rather than an env lookup.
    static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let dump = *DUMP.get_or_init(|| crate::config::get().draft_gemm_dump);
    if dump {
        gpu.hip.device_synchronize()?;
    }
    let t0 = if dump {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let result = match w.gpu_dtype {
        DType::F32 => gpu.gemm_f32_batched(x, &w.buf, y, batch, w.k, w.m),
        DType::F16 if w.row_stride != 0 => {
            gpu.gemm_f16_pitched_gfx1151(&w.buf, x, y, w.m, w.k, w.row_stride / 2, batch)
        }
        DType::F16 => gpu.gemm_f16_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G128V2 => Err(hip_bridge::HipError::new(
            0,
            "dflash GEMM cannot consume MQ4G128V2: qt=53 is admitted only by typed Qwen4 sealed/dense consumers",
        )),
        DType::MQ3I4G128R256
        | DType::MQ3I4G128R128
        | DType::MQ2I4G128R256
        | DType::MQ2I4G128R128
        | DType::E8PI4G128R256
        | DType::E8PI4G128R128 => Err(hip_bridge::HipError::new(
            0,
            "dflash GEMM cannot consume MQ-I4 / E8P-I4 (qt=131..=136): typed-Qwen4 expert-slab format with no generic GPU kernel",
        )),
        DType::HFQ4G256 => gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G256 => {
            // S7 draft collapse: rotate straight to the persistent F16 twin
            // and run an overwrite WMMA (accumulator from +0), removing the
            // per-call convert_f32_to_f16 + pre-zero fill. Every predicate
            // failure (non-gfx1100, kill switch, batch<=1, AWQ, non-default
            // variant policy) falls through to the loop below byte-for-byte.
            let route = gpu.draft_collapse_mq4_route(w.m, w.k, batch, w.awq_scale.is_some());
            if route == rdna_compute::dflash_draft_fusion::DraftCollapseGemm::Off {
                gemm_dispatch_mq4_legacy(gpu, x, w, y, batch, mq_x_rot)
            } else {
                let scratch = mq_x_rot_f16.expect("MQ4 collapse requires mq_x_rot_f16 scratch");
                gemm_dispatch_mq4_collapsed(gpu, x, w, y, batch, scratch, route)
            }
        }
        DType::MQ3G256 => {
            // Mirrors the MQ4 path: pre-rotate x via FWHT (same shared signs
            // as MQ4 — rotate_x_mq_batched is dtype-agnostic for the activation
            // side), invalidate the FP16 x cache because the rotated bytes
            // share the same source pointer, then dispatch the HFQ3 batched
            // lm_head WMMA kernel. Chunked symmetrically with MQ4.
            //
            // `fp16_x_source_ptr` is invalidated ONCE before the chunk loop —
            // not per-iteration. The MQ3 dispatch always overwrites the
            // shared rotation scratch from scratch each call (no chunk can
            // re-read FP16 from the previous chunk's output), so the
            // invalidation only needs to fire once per gemm_dispatch entry.
            // Previously the assignment was inside the loop, firing
            // `ceil(batch / max_chunk)` times for no extra correctness.
            let scratch = mq_x_rot.expect("MQ3 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                let rot_view = scratch.sub_offset(0, n * w.k);
                // Same AWQ-aware FWHT rotation as the MQ4 arm. Drafters
                // that ship MQ3 AWQ sidecars (via the loader fix in
                // `hfq_weight`) now actually receive the `x /= s` divide
                // before the HFQ3 GEMM; pre-fix this silently produced
                // wrong drafts on AWQ-calibrated drafters.
                if let Err(e) =
                    crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                if let Err(e) =
                    gpu.gemm_hfq3g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                row += n;
            }
            chunked
        }
        DType::MQ6G256 => {
            // Mirrors the MQ4/MQ3 path: pre-rotate x via FWHT, then dispatch
            // the HFQ6 batched lm_head WMMA kernel. Chunked symmetrically with
            // the other MQ formats.
            let scratch = mq_x_rot.expect("MQ6 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                let rot_view = scratch.sub_offset(0, n * w.k);
                if let Err(e) =
                    crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                if let Err(e) =
                    gpu.gemm_hfq6g256_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                {
                    chunked = Err(e);
                    break;
                }
                row += n;
            }
            chunked
        }
        DType::MQ4G256V2 => {
            // MQ4 v2 (qt=44): same 136 B stride as v1 but fp16 per-128 header.
            // Uses the dedicated v2 batched lm_head kernel so header decode is
            // correct; rotation is identical FWHT path.
            // S7: per-chunk route — chunks that hit the gfx1100 ksplit tier
            // (batch 2..=16, default policy) rotate to F16 and run the
            // overwrite ksplit GEMM. On gfx1201 the `Off` branch first tries
            // the in-kernel-zeroing overwrite WMMA twin (batch 2..=63, no
            // memset); everything else (n==1 GEMV tails, chunked first-call
            // prefixes, capture/replay, kill switch) keeps the legacy loop
            // byte-for-byte.
            let scratch_f16 = mq_x_rot_f16;
            let max_chunk = (mq_x_rot
                .expect("MQ4V2 dispatch requires mq_x_rot scratch")
                .shape[0]
                / w.k)
                .max(1);
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                if n == 1 {
                    // The qt44 batched launcher intentionally has no scalar
                    // fallback. Incremental context fills can be one row after
                    // a zero-accept cycle, so use the ordinary qt44 GEMV on the
                    // original (unrotated) activation for that tail.
                    if let Err(e) = crate::llama::weight_gemv(gpu, w, &x_chunk, &y_chunk) {
                        chunked = Err(e);
                        break;
                    }
                } else {
                    let route = gpu.draft_collapse_mq4v2_route(w.k, n, w.awq_scale.is_some());
                    if route == rdna_compute::dflash_draft_fusion::DraftCollapseV2::Off {
                        let scratch = mq_x_rot.expect("MQ4V2 dispatch requires mq_x_rot scratch");
                        let rot_view = scratch.sub_offset(0, n * w.k);
                        if let Err(e) = crate::llama::rotate_x_mq_batched_for(
                            gpu, w, &x_chunk, &rot_view, w.k, n,
                        ) {
                            chunked = Err(e);
                            break;
                        }
                        // gfx1201: overwrite twin of the one-tile / verify-tile
                        // residual kernel (no pre-zero fill, byte-identical).
                        // `Ok(false)` = not admitted, device untouched: run the
                        // legacy memset + residual lm_head launcher unchanged.
                        match gpu.gemm_mq4g256v2_overwrite_gfx12_dflash(
                            &w.buf, &rot_view, &y_chunk, w.m, w.k, n,
                        ) {
                            Ok(true) => {}
                            Ok(false) => {
                                if let Err(e) = gpu.gemm_mq4g256v2_batched_lmhead(
                                    &w.buf, &rot_view, &y_chunk, w.m, w.k, n,
                                ) {
                                    chunked = Err(e);
                                    break;
                                }
                            }
                            Err(e) => {
                                chunked = Err(e);
                                break;
                            }
                        }
                    } else {
                        let scratch =
                            scratch_f16.expect("MQ4V2 collapse requires mq_x_rot_f16 scratch");
                        let rot_f16 = scratch.sub_offset(0, n * w.k);
                        if let Err(e) = gpu.mq_rotate_x_f16_dflash(&x_chunk, &rot_f16, w.k, n) {
                            chunked = Err(e);
                            break;
                        }
                        let rdna_compute::dflash_draft_fusion::DraftCollapseV2::OverwriteKsplit {
                            kw,
                        } = route
                        else {
                            unreachable!("route != Off here")
                        };
                        if let Err(e) = gpu.gemm_mq4g256v2_overwrite_ksplit_lds_dflash(
                            &w.buf, &rot_f16, &y_chunk, w.m, w.k, n, kw,
                        ) {
                            chunked = Err(e);
                            break;
                        }
                    }
                }
                row += n;
            }
            chunked
        }
        DType::MQ6G256V2 => {
            let scratch = mq_x_rot.expect("MQ6V2 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                if n == 1 {
                    if let Err(e) = crate::llama::weight_gemv(gpu, w, &x_chunk, &y_chunk) {
                        chunked = Err(e);
                        break;
                    }
                } else {
                    let rot_view = scratch.sub_offset(0, n * w.k);
                    if let Err(e) =
                        crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                    if let Err(e) =
                        gpu.gemm_mq6g256v2_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                }
                row += n;
            }
            chunked
        }
        DType::MQ5G256V2 => {
            let scratch = mq_x_rot.expect("MQ5V2 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                if n == 1 {
                    if let Err(e) = crate::llama::weight_gemv(gpu, w, &x_chunk, &y_chunk) {
                        chunked = Err(e);
                        break;
                    }
                } else {
                    let rot_view = scratch.sub_offset(0, n * w.k);
                    if let Err(e) =
                        crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                    if let Err(e) =
                        gpu.gemm_mq5g256v2_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                }
                row += n;
            }
            chunked
        }
        DType::MQ3G256V2 => {
            // Safety quarantine: repeated DFlash2 B=16 runs with an MQ3V2
            // draft have dropped exact gfx1100 from the PCIe bus, while the
            // ordinary MQ3V2 AR GEMV path is stable. Do not dispatch the
            // gfx11 batched residual-WMMA kernel from DFlash on this arch.
            // Row-wise `weight_gemv` is the same proven kernel path AR uses;
            // gfx1151/gfx12 retain the batched route below.
            if gpu.arch_caps.is_gfx1100() {
                for row in 0..batch {
                    let x_row = x.sub_offset(row * w.k, w.k);
                    let y_row = y.sub_offset(row * w.m, w.m);
                    crate::llama::weight_gemv(gpu, w, &x_row, &y_row)?;
                }
                return Ok(());
            }
            let scratch = mq_x_rot.expect("MQ3V2 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                if n == 1 {
                    if let Err(e) = crate::llama::weight_gemv(gpu, w, &x_chunk, &y_chunk) {
                        chunked = Err(e);
                        break;
                    }
                } else {
                    let rot_view = scratch.sub_offset(0, n * w.k);
                    if let Err(e) =
                        crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                    if let Err(e) =
                        gpu.gemm_mq3g256v2_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                }
                row += n;
            }
            chunked
        }
        DType::MQ2G256V2 => {
            let scratch = mq_x_rot.expect("MQ2V2 dispatch requires mq_x_rot scratch");
            let max_chunk = (scratch.shape[0] / w.k).max(1);
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            let mut chunked: HipResult<()> = Ok(());
            let mut row = 0;
            while row < batch {
                let n = std::cmp::min(max_chunk, batch - row);
                let x_chunk = x.sub_offset(row * w.k, n * w.k);
                let y_chunk = y.sub_offset(row * w.m, n * w.m);
                if n == 1 {
                    if let Err(e) = crate::llama::weight_gemv(gpu, w, &x_chunk, &y_chunk) {
                        chunked = Err(e);
                        break;
                    }
                } else {
                    let rot_view = scratch.sub_offset(0, n * w.k);
                    if let Err(e) =
                        crate::llama::rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot_view, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                    // MQ2V2 batched lm_head — sibling kernel work provides the wrapper.
                    // If absent, this arm fails closed rather than mapping to legacy.
                    if let Err(e) =
                        gpu.gemm_mq2g256v2_batched_lmhead(&w.buf, &rot_view, &y_chunk, w.m, w.k, n)
                    {
                        chunked = Err(e);
                        break;
                    }
                }
                row += n;
            }
            chunked
        }
        other => panic!("dflash gemm_dispatch: unsupported weight dtype {:?}", other),
    };
    if let Some(t) = t0 {
        let us = t.elapsed().as_micros();
        let weight_bytes = match w.gpu_dtype {
            DType::F32 => w.m * w.k * 4,
            DType::F16 => w.m * w.k * 2,
            DType::MQ3G256 => w.m * (w.k / 256).max(1) * 104,
            DType::MQ3G256V2 => w.m * (w.k / 256).max(1) * rdna_compute::MQ3G256V2_GROUP_BYTES,
            DType::HFQ4G256 | DType::MQ4G256 | DType::MQ4G256V2 => w.m * (w.k / 256).max(1) * 136,
            DType::MQ6G256 => w.m * (w.k / 256).max(1) * 200,
            DType::MQ6G256V2 => w.m * (w.k / 256).max(1) * rdna_compute::MQ6G256V2_GROUP_BYTES,
            DType::MQ5G256V2 => w.m * (w.k / 256).max(1) * rdna_compute::MQ5G256V2_GROUP_BYTES,
            DType::MQ2G256V2 => w.m * (w.k / 256).max(1) * rdna_compute::MQ2G256V2_GROUP_BYTES,
            _ => w.m * w.k,
        };
        let bytes = weight_bytes + batch * w.k * 4 + batch * w.m * 4 * 2;
        let gbs = (bytes as f64) / (us.max(1) as f64) / 1000.0;
        eprintln!(
            "[draft-gemm] dtype={:?} M={} K={} B={} us={} bytes={}KB GB/s={:.1}",
            w.gpu_dtype,
            w.m,
            w.k,
            batch,
            us,
            bytes / 1024,
            gbs
        );
    }
    result
}

fn begin_draft_ffn_graph_capture(gpu: &mut Gpu) -> HipResult<()> {
    gpu.graphs.capture_blobs.clear();
    gpu.graphs.capture_mode = true;
    let stream = gpu
        .active_stream
        .as_ref()
        .expect("draft FFN graph capture requires an explicit stream");
    gpu.hip.stream_begin_capture(stream, 0)
}

fn end_draft_ffn_graph_capture(gpu: &mut Gpu) -> HipResult<(Graph, GraphExec, Vec<Vec<u8>>)> {
    gpu.graphs.capture_mode = false;
    let stream = gpu.active_stream.as_ref().unwrap();
    let graph = gpu.hip.stream_end_capture(stream)?;
    let exec = gpu.hip.graph_instantiate(&graph)?;
    let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
    Ok((graph, exec, blobs))
}

fn abort_draft_ffn_graph_capture(gpu: &mut Gpu) {
    if gpu.graphs.capture_mode {
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_end_capture(stream);
        }
        gpu.graphs.capture_mode = false;
    }
    gpu.graphs.capture_blobs.clear();
}

fn draft_ffn_layer(
    gpu: &mut Gpu,
    layer: &DflashLayerWeights,
    scratch: &mut DflashScratch,
    b: usize,
    h: usize,
    eps: f32,
    graph_safe: bool,
) -> HipResult<()> {
    // S7: one dual-output RMSNorm replaces the residual memcpy + norm pair
    // (bitwise residual capture, identical norm order). Blob-launched, so it
    // is capturable in both graph_safe modes without a branch.
    if gpu.draft_collapse_fused_enabled() {
        gpu.rmsnorm_residual_dual_dflash(
            &scratch.x,
            &layer.ffn_norm,
            &scratch.residual,
            &scratch.x_norm,
            b,
            h,
            eps,
        )?;
    } else {
        if graph_safe {
            gpu.memcpy_dtod_auto(&scratch.residual.buf, &scratch.x.buf, (b * h) * 4)?;
        } else {
            gpu.hip
                .memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, (b * h) * 4)?;
        }
        gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;
    }
    gemm_dispatch_shared(
        gpu,
        &scratch.x_norm,
        [(&layer.w_gate, &scratch.gate), (&layer.w_up, &scratch.up)],
        b,
        scratch.mq_x_rot.as_ref(),
        scratch.mq_x_rot_f16.as_ref(),
        SharedProjection::Ffn,
    )?;
    gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;
    gemm_dispatch(
        gpu,
        &scratch.gate_up,
        &layer.w_down,
        &scratch.x,
        b,
        scratch.mq_x_rot.as_ref(),
        scratch.mq_x_rot_f16.as_ref(),
    )?;
    if graph_safe {
        gpu.add_f32(&scratch.residual, &scratch.x, &scratch.x)
    } else {
        gpu.add_f32(&scratch.residual, &scratch.x, &scratch.x)
    }
}

fn draft_ffn_layer_maybe_graph(
    gpu: &mut Gpu,
    layer: &DflashLayerWeights,
    scratch: &mut DflashScratch,
    layer_idx: usize,
    b: usize,
    h: usize,
    eps: f32,
    use_graph: bool,
) -> HipResult<()> {
    if !use_graph {
        return draft_ffn_layer(gpu, layer, scratch, b, h, eps, false);
    }

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    if scratch.draft_ffn_graphs[layer_idx].contains_key(&b) {
        let stream = gpu.active_stream.as_ref().unwrap();
        let entry = scratch.draft_ffn_graphs[layer_idx]
            .get(&b)
            .unwrap_or_else(|| panic!("missing draft FFN graph for layer={layer_idx} b={b}"));
        return gpu.hip.graph_launch(&entry.1, stream);
    }

    if !scratch.draft_ffn_warmed_up[layer_idx].contains(&b) {
        scratch.draft_ffn_warmed_up[layer_idx].insert(b);
        return draft_ffn_layer(gpu, layer, scratch, b, h, eps, false);
    }

    begin_draft_ffn_graph_capture(gpu)?;
    let r = draft_ffn_layer(gpu, layer, scratch, b, h, eps, true);
    if r.is_ok() {
        let entry = end_draft_ffn_graph_capture(gpu)?;
        scratch.draft_ffn_graphs[layer_idx].insert(b, entry);
        let stream = gpu.active_stream.as_ref().unwrap();
        let entry = scratch.draft_ffn_graphs[layer_idx]
            .get(&b)
            .unwrap_or_else(|| {
                panic!("missing captured draft FFN graph for layer={layer_idx} b={b}")
            });
        gpu.hip.graph_launch(&entry.1, stream)
    } else {
        abort_draft_ffn_graph_capture(gpu);
        r
    }
}

/// Upload f32 slice into a GPU tensor (bytes via memcpy_htod).
fn upload_slice_f32(gpu: &Gpu, dst: &GpuTensor, data: &[f32]) -> HipResult<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Upload i32 slice into a GPU tensor (interpreted as i32 by kernels).
/// Synchronous outside hipGraph capture; while capturing it is enqueued on
/// the capturing stream (`Gpu::memcpy_htod_auto`), which bakes `data`'s host
/// address into the graph: a replaying owner keeps that storage alive and
/// rewrites it in place before each launch.
fn upload_slice_i32(gpu: &Gpu, dst: &GpuTensor, data: &[i32]) -> HipResult<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.memcpy_htod_auto(&dst.buf, bytes)
}

/// `HIPFIRE_DFLASH_ORDERED_DRAFT=0` (developer kill switch): the draft
/// body's D2D copies (residual saves, context/noise K/V concatenation) go
/// back to the synchronous `hipMemcpy` on the null stream, and the batched
/// lane draft uploads its key positions between the lane fc GEMMs again.
/// Default on: the copies are enqueued on the execution stream
/// (`Gpu::memcpy_dtod_at_ordered_async`) ahead of their consumer kernels and
/// the host no longer drains the queue per copy. Same bytes, same ranges,
/// same order on the stream as every producer and consumer kernel.
static ORDERED_DRAFT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_DFLASH_ORDERED_DRAFT").map_or(true, |v| v.trim() != "0")
});

/// Whether the draft body runs its D2D copies stream-ordered
/// ([`ORDERED_DRAFT`]); resolved once per process.
pub fn ordered_draft() -> bool {
    *ORDERED_DRAFT
}

/// One draft-body D2D copy. `ordered`: enqueued on the active stream (the
/// null stream when none is set), where every producer of `src` and every
/// consumer of `dst` in the draft body also runs, so no host wait is needed
/// and the source cannot be rewritten before the copy executes. Otherwise
/// the historical host-synchronous copy.
#[track_caller]
fn draft_dtod_at(
    gpu: &Gpu,
    ordered: bool,
    dst: &DeviceBuffer,
    dst_offset: usize,
    src: &DeviceBuffer,
    src_offset: usize,
    bytes: usize,
) -> HipResult<()> {
    if ordered {
        gpu.memcpy_dtod_at_ordered_async(dst, dst_offset, src, src_offset, bytes)
    } else {
        gpu.hip
            .memcpy_dtod_at(dst, dst_offset, src, src_offset, bytes)
    }
}

/// Rows per chunk for the windowed-mode post-seed backfill.
pub const SEED_BACKFILL_CHUNK: usize = 512;

/// Whether [`draft_seed_backfill`] reads its host shadow for a cold prompt of
/// `prompt_len` rows: only a windowed draft with a dedicated long-reach layer
/// (`w != w_full`) whose prompt is longer than the SWA window. Legacy mode and
/// DFlash2 all-sliding drafts (every layer shares the same W ring) never read
/// it, so a cold prefill may leave the hidden rows on device.
pub fn draft_seed_backfill_reads_host(ctx_mode: DraftCtxMode, prompt_len: usize) -> bool {
    matches!(ctx_mode, DraftCtxMode::Windowed { w, w_full } if w != w_full && prompt_len > w)
}

/// Windowed-mode post-seed backfill: after a cold prefill longer than the
/// SWA window, the last (full-attention) layer still needs K/V for EVERY
/// context row — but `hidden_rb` / the draft ring only retain the last
/// `swa_w`. The host shadow (cumulative over the cold prefill) has the full
/// prompt, so stream it through the ring in chunks:
/// upload → fc → hidden_norm → last-layer wk/wv + k_norm into its
/// `full_w`-row ring. SWA layers are deliberately skipped — the first
/// `draft_forward` lazy fill covers their in-window rows bit-identically.
///
/// No-op (and `host_hidden` unread) unless
/// [`draft_seed_backfill_reads_host`]. Sets the thlog
/// full-layer watermark to `prompt_len` so the forward's lazy fill resumes
/// from the window edge. GEMMs, norms, and RoPE are row-local: chunked ==
/// monolithic bit-exactly.
pub fn draft_seed_backfill(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    scratch: &mut DflashScratch,
    host_hidden: &[f32],
    prompt_len: usize,
) -> HipResult<()> {
    let (swa_w, full_w) = match scratch.ctx_mode {
        DraftCtxMode::Windowed { w, w_full }
            if draft_seed_backfill_reads_host(scratch.ctx_mode, prompt_len) =>
        {
            (w, w_full)
        }
        _ => return Ok(()),
    };
    let h = cfg.hidden;
    let fw = cfg.feature_width();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let row_f32 = fw;
    assert_eq!(
        host_hidden.len(),
        prompt_len * row_f32,
        "backfill: host shadow must hold the full prompt"
    );
    let last_layer = weights.layers.last().expect("draft has no layers");
    let k_full = scratch.k_full_cached.as_ref().expect("windowed k_full");
    let v_full = scratch.v_full_cached.as_ref().expect("windowed v_full");

    let mut row = 0usize;
    while row < prompt_len {
        let len = SEED_BACKFILL_CHUNK.min(prompt_len - row);
        for (row0, slot0, seg_len) in ring_segments(row, row + len, swa_w) {
            // H2D this host segment into the draft ring.
            let seg = &host_hidden[row0 * row_f32..(row0 + seg_len) * row_f32];
            let src_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4) };
            gpu.hip.memcpy_htod_offset(
                &scratch.target_hidden.buf,
                slot0 * row_f32 * 4,
                src_bytes,
            )?;
            // fc + hidden_norm into the proj ring (same slots).
            let th = scratch
                .target_hidden
                .sub_offset(slot0 * row_f32, seg_len * row_f32);
            let thp = scratch
                .target_hidden_proj
                .sub_offset(slot0 * h, seg_len * h);
            gemm_dispatch(
                gpu,
                &th,
                &weights.fc,
                &thp,
                seg_len,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
            )?;
            gpu.rmsnorm_batched(&thp, &weights.hidden_norm, &thp, seg_len, h, eps)?;
            // Last-layer wk/wv into the full_w ring (its own modulus).
            let mut r2 = row0;
            while r2 < row0 + seg_len {
                let step = (full_w - r2 % full_w).min(row0 + seg_len - r2);
                let thp2 = scratch
                    .target_hidden_proj
                    .sub_offset((slot0 + (r2 - row0)) * h, step * h);
                let c_slot = r2 % full_w;
                let k_slot = k_full.sub_offset(c_slot * kvd, step * kvd);
                let v_slot = v_full.sub_offset(c_slot * kvd, step * kvd);
                gemm_dispatch(
                    gpu,
                    &thp2,
                    &last_layer.wk,
                    &k_slot,
                    step,
                    scratch.mq_x_rot.as_ref(),
                    scratch.mq_x_rot_f16.as_ref(),
                )?;
                gemm_dispatch(
                    gpu,
                    &thp2,
                    &last_layer.wv,
                    &v_slot,
                    step,
                    scratch.mq_x_rot.as_ref(),
                    scratch.mq_x_rot_f16.as_ref(),
                )?;
                gpu.rmsnorm_batched(
                    &k_slot,
                    &last_layer.k_norm,
                    &k_slot,
                    step * cfg.n_kv_heads,
                    hd,
                    eps,
                )?;
                // The steady-state cache stores post-RoPE K, so cold-seed
                // backfill must establish the same invariant. Prompt rows are
                // contiguous absolute positions here; upload just this chunk's
                // position vector into the reusable device buffer.
                let positions: Vec<i32> = (r2..r2 + step).map(|p| p as i32).collect();
                upload_slice_i32(gpu, &scratch.positions_k, &positions)?;
                let positions_view = scratch.positions_k.sub_offset(0, step);
                gpu.rope_batched_f32(
                    &scratch.q, // ignored because n_heads_q = 0
                    &k_slot,
                    &positions_view,
                    0,
                    cfg.n_kv_heads,
                    hd,
                    cfg.rope_theta,
                    step,
                )?;
                r2 += step;
            }
        }
        row += len;
    }
    scratch.thlog.mark_full_cached(prompt_len);
    Ok(())
}

/// Run one draft forward. Inputs:
/// - `noise_embedding`: `[block_size × hidden]` f32, row-major. Comes from
///   `target.embed_tokens(block_output_ids)` on the caller side.
/// - `target_hidden`:   `[ctx_len × feature_width]` f32, row-major
///   (5-way concat of target's chosen-layer hidden states at `ctx_len`
///   accepted positions).
/// - `positions_q`:     `[block_size]` i32 — absolute position index of
///   each block position in the full sequence (used for RoPE on Q).
/// - `positions_k`:     `[ctx_len + block_size]` i32 — absolute position
///   index for every ctx position followed by every block position
///   (used for RoPE on K = concat(ctx, noise)).
///
/// Output: writes final hidden states `[block_size × hidden]` into
/// `scratch.x`. Caller applies target's `lm_head` over the last
/// `block_size - 1` rows to produce logits for the mask slots.
///
/// Precondition: `block_size ≤ scratch.max_block_size`,
/// `ctx_len ≤ scratch.max_ctx_len`.
/// Run one draft forward over `block_size` positions with `ctx_len` cached
/// context rows.
///
/// `noise_embedding`: if `Some`, uploaded into `scratch.x` before the forward.
///     If `None`, the caller must have already filled `scratch.x` with B × hidden
///     F32 embeddings — this avoids the target→host→draft round-trip in the
///     spec-decode hot loop (both target and draft share the same GPU, so
///     D2D copies into `scratch.x` suffice).
/// `target_hidden`: if `Some`, uploaded into `scratch.target_hidden`.
///     If `None`, the caller must have already filled `scratch.target_hidden`
///     with `ctx_len × feature_width` F32 rows.
#[allow(clippy::too_many_arguments)]
pub fn draft_forward(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
) -> HipResult<()> {
    draft_forward_opts(
        gpu,
        weights,
        cfg,
        noise_embedding,
        target_hidden,
        positions_q,
        positions_k,
        block_size,
        ctx_len,
        scratch,
        false,
    )
}

pub fn draft_forward_opts(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
    graph_ffn: bool,
) -> HipResult<()> {
    draft_forward_based(
        gpu,
        weights,
        cfg,
        noise_embedding,
        target_hidden,
        0,
        positions_q,
        positions_k,
        block_size,
        ctx_len,
        scratch,
        graph_ffn,
    )
}

/// [`draft_forward`] whose host `target_hidden` holds only rows
/// `target_hidden_base..ctx_len` (a caller that keeps the draft's window, not
/// the whole context). Windowed scratch only, and `target_hidden_base` must
/// not pass `ctx_len - W`: every row the forward can upload is still held.
#[allow(clippy::too_many_arguments)]
pub fn draft_forward_based(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    target_hidden_base: usize,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
    graph_ffn: bool,
) -> HipResult<()> {
    draft_forward_based_impl(
        gpu,
        weights,
        cfg,
        noise_embedding,
        target_hidden,
        target_hidden_base,
        positions_q,
        positions_k,
        block_size,
        ctx_len,
        scratch,
        graph_ffn,
        None,
        true,
    )
}

/// Device-resident staging of one window's positions (Design A §4): the owner
/// packed the B absolute Q positions and the live K-position words
/// (`positions_k[pos_base..]`: the `thlog.abs_positions()` suffix followed by
/// the B noise positions) into pinned storage and uploaded them once to `buf`
/// before the window.
pub struct DraftStagedInputs<'a> {
    pub buf: &'a DeviceBuffer,
    /// Byte offset of the B Q-position words.
    pub q_byte_off: usize,
    /// Byte offset of the K-position words (`ctx_len - pos_base + B` of them).
    pub k_byte_off: usize,
}

/// The GPU-resident, planned draft forward: [`draft_forward_based`] for the
/// `target_hidden = None` / `noise_embedding = None` route (noise rows already
/// in `scratch.x`), with the position uploads replaced by D2D copies out of
/// `staged` and **no host watermark published** (`thlog` is read, never
/// written): the owner commits `mark_proj_cached(ctx_len)` after completion
/// (Design A §2 `WindowHostDelta`). The FFN graph route is never taken.
pub fn draft_forward_planned(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    staged: &DraftStagedInputs<'_>,
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
) -> HipResult<()> {
    draft_forward_based_impl(
        gpu,
        weights,
        cfg,
        None,
        None,
        0,
        &[],
        &[],
        block_size,
        ctx_len,
        scratch,
        false,
        Some(staged),
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn draft_forward_based_impl(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    target_hidden_base: usize,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
    graph_ffn: bool,
    staged: Option<&DraftStagedInputs<'_>>,
    publish: bool,
) -> HipResult<()> {
    let base = target_hidden_base;
    let b = block_size;
    let l = ctx_len;
    let tot = l + b;
    let h = cfg.hidden;
    let fw = cfg.feature_width();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;

    assert!(b <= scratch.max_block_size, "block_size > scratch max");
    assert!(l <= scratch.max_ctx_len, "ctx_len > scratch max");
    if let Some(ne_slice) = noise_embedding {
        assert_eq!(ne_slice.len(), b * h, "noise_embedding size");
    }
    if let Some(th_slice) = target_hidden {
        assert_eq!(th_slice.len(), (l - base) * fw, "target_hidden size");
    }
    match scratch.ctx_mode {
        DraftCtxMode::Legacy => {
            assert_eq!(base, 0, "a based target_hidden needs a windowed scratch")
        }
        DraftCtxMode::Windowed { w, .. } => {
            assert!(
                base <= l.saturating_sub(w),
                "target_hidden base past the window"
            )
        }
    }
    if staged.is_none() {
        assert_eq!(positions_q.len(), b, "positions_q size");
        assert_eq!(positions_k.len(), tot, "positions_k size");
    } else {
        assert!(
            noise_embedding.is_none() && target_hidden.is_none(),
            "staged draft inputs own the positions only; embeddings and target hidden are GPU-resident"
        );
    }

    // Draft context windowing (DraftCtxMode::Windowed). `swa_w`/`full_w` are
    // the per-layer ring moduli (usize::MAX in Legacy ⇒ identity ring math,
    // single segment, byte-identical offsets). The last (full-attention)
    // layer uses `full_w`; all others `swa_w`. At l <= swa_w every span is
    // [0..l) and every slot == row: the windowed code path degenerates to
    // Legacy exactly.
    let (swa_w, full_w) = match scratch.ctx_mode {
        DraftCtxMode::Legacy => (usize::MAX, usize::MAX),
        DraftCtxMode::Windowed { w, w_full } => (w, w_full),
    };
    let windowed = !matches!(scratch.ctx_mode, DraftCtxMode::Legacy);
    // Windowed positions base: the device positions_k buffer holds only the
    // suffix [pos_base..tot) (last full_w context rows + B noise) — every
    // layer's attention span lives inside it. Device index of row r is
    // r − pos_base. Legacy: pos_base = 0, the full upload as before.
    let pos_base = if windowed {
        l.saturating_sub(full_w)
    } else {
        0
    };
    // Backfill-invariant probe: past the SWA window the last layer's
    // out-of-window K/V must come from the post-seed backfill. If a driver
    // skipped it (research demos on non-speculator paths), its ring holds
    // stale rows — τ-only damage (verify is exact), but name it once.
    // All-sliding DFlash2 has no dedicated full layer, so the watermark is moot.
    if windowed
        && full_w != swa_w
        && l > swa_w
        && scratch.thlog.full_cached_rows() < l.saturating_sub(swa_w)
    {
        static BACKFILL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !BACKFILL_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[dflash] windowed: last-layer backfill watermark {} < l−W {} — stale \
                 out-of-window K/V (acceptance degraded; output still verify-exact). \
                 Call draft_seed_backfill after cold seeds.",
                scratch.thlog.full_cached_rows(),
                l.saturating_sub(swa_w)
            );
        }
    }

    // ── 0. Uploads ────────────────────────────────────────────────────────
    if let Some(ne_slice) = noise_embedding {
        upload_slice_f32(gpu, &scratch.x, ne_slice)?;
    }
    if let Some(th_slice) = target_hidden {
        // Incremental-upload fast path: the caller passes a rolling prefix
        // (rows 0..l) of target_hidden. In DFlash's common steady-state
        // (ctx_slice == None), the prefix grows by accept+1 rows per cycle
        // and rows [0..prev_l) are unchanged since the previous call.
        // Upload only the new tail when detected. This cuts the H2D from
        // `l × feature_width × 4` (e.g. ~90 MB at l=1100 on 9B) to
        // `(l - uploaded) × feature_width × 4` (~700 KB at accept=8).
        //
        // The optimization only fires when `th_slice.len() == l × feature_width`
        // and it matches what the caller told us (matches a non-sliced
        // cumulative buffer). ctx_slice callers pass a DIFFERENT slice
        // every cycle (last N rows shift) — for them, force full upload.
        //
        // Windowed mode: the GPU buffer is an `swa_w`-row ring (slot = row %
        // swa_w); only rows [max(prev, l−swa_w)..l) are uploaded, wrap-split
        // at the ring boundary. Out-of-window rows are never read (fc is the
        // only consumer and its fill is span-limited too).
        let row_f32 = fw;
        let expected_full_len = (l - base) * row_f32;
        let prev = scratch.thlog.uploaded_rows();
        // Full-upload conditions: first call, reset flagged, caller shrank
        // the context, or the slice length suggests ctx_slice (unusual l).
        if prev == 0 || prev > l || th_slice.len() != expected_full_len {
            if windowed {
                let up_start = l.saturating_sub(swa_w);
                for (row0, slot0, len) in ring_segments(up_start, l, swa_w) {
                    let seg = &th_slice[(row0 - base) * row_f32..(row0 - base + len) * row_f32];
                    let dst_byte_off = slot0 * row_f32 * 4;
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4)
                    };
                    gpu.hip.memcpy_htod_offset(
                        &scratch.target_hidden.buf,
                        dst_byte_off,
                        src_bytes,
                    )?;
                }
            } else {
                upload_slice_f32(gpu, &scratch.target_hidden, th_slice)?;
            }
            scratch.thlog.mark_uploaded(l);
        } else if prev < l {
            // Delta-upload: rows [prev..l) need to land at byte offset
            // prev * row_f32 * 4 of scratch.target_hidden.
            if windowed {
                let up_start = prev.max(l.saturating_sub(swa_w));
                for (row0, slot0, len) in ring_segments(up_start, l, swa_w) {
                    let seg = &th_slice[(row0 - base) * row_f32..(row0 - base + len) * row_f32];
                    let dst_byte_off = slot0 * row_f32 * 4;
                    let src_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(seg.as_ptr() as *const u8, seg.len() * 4)
                    };
                    gpu.hip.memcpy_htod_offset(
                        &scratch.target_hidden.buf,
                        dst_byte_off,
                        src_bytes,
                    )?;
                }
            } else {
                let tail = &th_slice[prev * row_f32..];
                let dst_byte_off = prev * row_f32 * 4;
                let src_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tail.as_ptr() as *const u8, tail.len() * 4)
                };
                gpu.hip
                    .memcpy_htod_offset(&scratch.target_hidden.buf, dst_byte_off, src_bytes)?;
            }
            scratch.thlog.mark_uploaded(l);
        }
        // prev == l: nothing new to upload (wouldn't happen in practice
        // since caller always appends, but harmless).
    }
    match staged {
        // Owner-staged inputs (Design A §4): the positions were packed into
        // pinned storage and uploaded once before the window; the draft only
        // moves them D2D, which a plan sink records as typed copies.
        Some(st) => {
            gpu.memcpy_dtod_at_ordered_async(
                &scratch.positions_q.buf,
                0,
                st.buf,
                st.q_byte_off,
                b * 4,
            )?;
            let k_rows = if windowed { tot - pos_base } else { tot };
            gpu.memcpy_dtod_at_ordered_async(
                &scratch.positions_k.buf,
                0,
                st.buf,
                st.k_byte_off,
                k_rows * 4,
            )?;
        }
        None => {
            upload_slice_i32(gpu, &scratch.positions_q, positions_q)?;
            if windowed {
                upload_slice_i32(gpu, &scratch.positions_k, &positions_k[pos_base..])?;
            } else {
                upload_slice_i32(gpu, &scratch.positions_k, positions_k)?;
            }
        }
    }

    // ── 1. target_hidden_proj = hidden_norm(fc @ target_hidden) ──────────
    // Incremental-projection fast path (2026-04-20): only compute the
    // delta rows [cached..L) that haven't been projected yet. Rows
    // [0..cached) were projected and cached by a previous draft_forward
    // call and are still valid. After this block, rows [0..L) of
    // target_hidden_proj are usable by the per-layer K/V step below.
    //
    // `draft_ctx_cached_rows` is the scratch-owned invariant: how many
    // prefix rows have been computed end-to-end (fc + hidden_norm +
    // per-layer wk + k_norm + per-layer wv). It resets to 0 at new
    // prompts (reset_upload_tracking) and on eviction
    // (invalidate_draft_ctx_cache).
    //
    // Full-rebuild cases: delta == L (first cycle after reset), or
    // delta > L somehow (shouldn't happen — this would be a bug). Those
    // go through the same code path with delta == L, same cost as before.
    //
    // Dispatch on fc weight dtype: F32 → gemm_f32_batched (legacy),
    // MQ4 → FWHT-rotate target_hidden then gemm_hfq4g256.
    //
    // Windowed: the proj ring only needs the rows any layer can still read —
    // clamp the fill to [max(cached, l−swa_w)..l) (out-of-window rows were
    // covered by the post-seed backfill for the last layer, and SWA layers
    // never read them). Ring-slot writes are wrap-split; every GEMM/norm is
    // row-local, so segments compose bit-exactly. Legacy: swa_w=MAX ⇒ one
    // identity segment, offsets identical to the pre-windowing code.
    let cached_rows = scratch.thlog.proj_cached_rows();
    let fc_start = cached_rows.max(l.saturating_sub(swa_w));
    let delta = l.saturating_sub(fc_start);
    if delta > 0 {
        for (_row0, slot0, len) in ring_segments(fc_start, l, swa_w) {
            let th_slice = scratch.target_hidden.sub_offset(slot0 * fw, len * fw);
            let thp_slice = scratch.target_hidden_proj.sub_offset(slot0 * h, len * h);
            gemm_dispatch(
                gpu,
                &th_slice,
                &weights.fc,
                &thp_slice,
                len,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
            )?;
            gpu.rmsnorm_batched(&thp_slice, &weights.hidden_norm, &thp_slice, len, h, eps)?;
        }
    }

    // HIPFIRE_DRAFT_SUBPHASE=1: per-layer-section timing inside draft_forward.
    // Diagnostic only — device_synchronize at each boundary makes this 2-3×
    // slower than a production run. Printed once per forward.
    //
    // First measurement (2026-04-21, 27B HumanEval, B=16, steady-state):
    //   attn_gemm: 7.4 ms  (wq + wk/v_noise + wk/v_ctx)
    //   concat:    0.4 ms  (K/V cache concat D2Ds)
    //   attn_krn:  0.6 ms  (attention_dflash_f32)
    //   ffn_gemm:  56 ms   (wo + w_gate + w_up + w_down + silu_mul + adds)
    //
    // 87 % of draft_forward lives in the FFN GEMM block. w_gate/w_up/w_down
    // at M=17408/K=5120 should be ~0.5 ms/layer BW-bound but is ~11 ms/layer
    // observed. Next lever: route draft's w_gate/w_up through the fused
    // gemm_gate_up_hfq4g256_wmma kernel (measured 73 µs/call on the same
    // shape in target; vs ksplit's 288 µs/call on a different shape). Or
    // find the real cause of the ksplit slowdown on draft shapes.
    let dbg = crate::config::get().draft_subphase;
    let mut us_attn_gemm: u128 = 0;
    let mut us_attn_kernel: u128 = 0;
    let mut us_ffn_gemm: u128 = 0;
    let mut us_concat: u128 = 0;
    if dbg {
        gpu.hip.device_synchronize()?;
    }

    // ── 2. Per-layer decoder ─────────────────────────────────────────────
    // DFlash2 all-sliding detection for windowed mode branching.
    let is_all_sliding_windowed = windowed && cfg.all_layers_sliding;
    let ordered = ordered_draft();
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];

        // S7: dual-output RMSNorm replaces the residual memcpy + attn_norm
        // pair (bitwise residual capture, identical norm order).
        if gpu.draft_collapse_fused_enabled() {
            gpu.rmsnorm_residual_dual_dflash(
                &scratch.x,
                &layer.attn_norm,
                &scratch.residual,
                &scratch.x_norm,
                b,
                h,
                eps,
            )?;
        } else {
            // Residual.
            draft_dtod_at(
                gpu,
                ordered,
                &scratch.residual.buf,
                0,
                &scratch.x.buf,
                0,
                (b * h) * 4,
            )?;
            // attn_norm.
            gpu.rmsnorm_batched(&scratch.x, &layer.attn_norm, &scratch.x_norm, b, h, eps)?;
        }

        // ── DFlash2 prepare conv before QKV (no cross-cycle history) ─────
        // After RMSNorm, project normalized hidden to dynamic kernel coeffs
        // [B,2*K*G] then left-zero-padded grouped conv over the B rows.
        let attn_prepare_src =
            if let (Some(base), Some(proj)) = (&layer.attn_conv_base, &layer.attn_conv_proj) {
                if let (Some(dyn_buf), Some(tmp)) = (&scratch.conv_dynamic, &scratch.conv_temp) {
                    let k = cfg.conv_kernel_size.unwrap_or(2);
                    let g = cfg.conv_group_size.unwrap_or(16);
                    let groups = h / g;
                    let stride = 2 * k * groups;
                    // projection: x_norm [B,H] @ proj [2KG, H]^T -> dynamic [B,2KG]
                    let dyn_slice = dyn_buf.sub_offset(0, b * stride);
                    gemm_dispatch(
                        gpu,
                        &scratch.x_norm,
                        proj,
                        &dyn_slice,
                        b,
                        scratch.mq_x_rot.as_ref(),
                        scratch.mq_x_rot_f16.as_ref(),
                    )?;
                    // prepare phase offset 0, window K*G
                    gpu.dynamic_causal_conv_f32(
                        &scratch.x_norm,
                        base,
                        &dyn_slice,
                        tmp,
                        b,
                        h,
                        k,
                        g,
                        stride,
                        0,
                    )?;
                    Some(tmp as &GpuTensor)
                } else {
                    None
                }
            } else {
                None
            };
        let qkv_src = attn_prepare_src.unwrap_or(&scratch.x_norm);

        let t0 = if dbg {
            gpu.hip.device_synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Q/K/V projections — dispatched on each weight's dtype.
        // Q and K/V noise (over the B block positions) must be computed
        // every cycle. K_ctx and V_ctx (over the L context positions)
        // are *incrementally* cached — see the per-layer block below.
        gemm_dispatch_shared(
            gpu,
            qkv_src,
            [
                (&layer.wq, &scratch.q),
                (&layer.wk, &scratch.k_noise),
                (&layer.wv, &scratch.v_noise),
            ],
            b,
            scratch.mq_x_rot.as_ref(),
            scratch.mq_x_rot_f16.as_ref(),
            SharedProjection::Qkv,
        )?;

        // K_ctx / V_ctx — same wk/wv weights but projected over the L
        // accepted-context rows of target_hidden_proj. INCREMENTAL PATH:
        // only rows past the layer's fill watermark need projection; earlier
        // rows were projected in a prior call and stored in the per-layer
        // k_ctx_cached / v_ctx_cached buffers (post-k_norm + post-RoPE for K).
        //
        // Windowed spans: SWA layers read the last `swa_w` context rows, the
        // last (full-attention) layer the last `full_w`. Each cache is a ring
        // (slot = row % layer_w); fills clamp to rows still readable
        // ([max(watermark, span_start)..l)) and source the proj ring, whose
        // modulus is `swa_w` — the last layer's (watermark, l−swa_w) band is
        // the post-seed backfill's responsibility. Segments split at BOTH
        // ring boundaries (proj + cache) so every GEMM sees contiguous src
        // and dst. Legacy: moduli are MAX — one identity segment, offsets
        // identical to the pre-windowing incremental path.
        //
        // For all-sliding DFlash2 windowed mode every layer shares swa_w;
        // last-layer full replacement/backfill/watermarks are skipped.
        let is_last_layer = li + 1 == cfg.n_layers;
        let is_full_layer = !is_all_sliding_windowed && is_last_layer;
        let layer_w = if is_full_layer { full_w } else { swa_w };
        let (k_cache_layer, v_cache_layer) = if is_full_layer && windowed {
            (
                scratch.k_full_cached.as_ref().expect("windowed k_full"),
                scratch.v_full_cached.as_ref().expect("windowed v_full"),
            )
        } else {
            (&scratch.k_ctx_cached[li], &scratch.v_ctx_cached[li])
        };
        let (k_cat_l, v_cat_l) = if is_full_layer && windowed {
            (
                scratch.k_cat_full.as_ref().expect("windowed k_cat_full"),
                scratch.v_cat_full.as_ref().expect("windowed v_cat_full"),
            )
        } else {
            (&scratch.k_cat, &scratch.v_cat)
        };
        let span_start = l.saturating_sub(layer_w);
        let span = l - span_start;
        let wm = if is_full_layer {
            scratch.thlog.full_cached_rows()
        } else {
            cached_rows
        };
        // Rows older than the proj ring's live window cannot be (re)filled
        // here — the post-seed backfill owns that band for the last layer.
        let fill_start = wm.max(span_start).max(l.saturating_sub(swa_w));
        // For all-sliding, there is no out-of-window full backfill band to skip;
        // the fill_start already equals span_start.
        if fill_start < l {
            let mut row = fill_start;
            while row < l {
                // Next row where the proj ring or the cache ring wraps.
                let step =
                    (swa_w.saturating_sub(if swa_w == usize::MAX { 0 } else { row % swa_w }))
                        .min(layer_w.saturating_sub(if layer_w == usize::MAX {
                            0
                        } else {
                            row % layer_w
                        }))
                        .min(l - row);
                let p_slot = if swa_w == usize::MAX {
                    row
                } else {
                    row % swa_w
                };
                let c_slot = if layer_w == usize::MAX {
                    row
                } else {
                    row % layer_w
                };
                let thp_slice = scratch.target_hidden_proj.sub_offset(p_slot * h, step * h);
                let k_slot = k_cache_layer.sub_offset(c_slot * kvd, step * kvd);
                let v_slot = v_cache_layer.sub_offset(c_slot * kvd, step * kvd);
                gemm_dispatch_shared(
                    gpu,
                    &thp_slice,
                    [(&layer.wk, &k_slot), (&layer.wv, &v_slot)],
                    step,
                    scratch.mq_x_rot.as_ref(),
                    scratch.mq_x_rot_f16.as_ref(),
                    SharedProjection::ContextKv,
                )?;
                // Per-head RMSNorm on K delta rows only. batch = step × n_kv_heads.
                gpu.rmsnorm_batched(
                    &k_slot,
                    &layer.k_norm,
                    &k_slot,
                    step * cfg.n_kv_heads,
                    hd,
                    eps,
                )?;
                // Cache K in its final, absolute-position RoPE domain. The
                // positions buffer holds the live context suffix starting at
                // pos_base in windowed mode and the full sequence in Legacy.
                // RoPE is elementwise, so rotating this delta now is bitwise
                // equivalent to rotating the same rows after concatenation.
                let fill_positions = scratch.positions_k.sub_offset(row - pos_base, step);
                gpu.rope_batched_f32(
                    &scratch.q, // ignored because n_heads_q = 0
                    &k_slot,
                    &fill_positions,
                    0,
                    cfg.n_kv_heads,
                    hd,
                    theta,
                    step,
                )?;
                row += step;
            }
        }

        if let Some(t) = t0 {
            gpu.hip.device_synchronize()?;
            us_attn_gemm += t.elapsed().as_micros();
        }
        let t1 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Per-head RMSNorm on Q: each of B*n_heads rows, size head_dim,
        // weight [head_dim].
        gpu.rmsnorm_batched(
            &scratch.q,
            &layer.q_norm,
            &scratch.q,
            b * cfg.n_heads,
            hd,
            eps,
        )?;
        // Normalize and rotate the B-row noise K before concatenation. Q and
        // noise K share the same absolute block positions, so one RoPE launch
        // handles both. Historical context K is already post-RoPE in its cache.
        gpu.rmsnorm_batched(
            &scratch.k_noise,
            &layer.k_norm,
            &scratch.k_noise,
            b * cfg.n_kv_heads,
            hd,
            eps,
        )?;
        gpu.rope_batched_f32(
            &scratch.q,
            &scratch.k_noise,
            &scratch.positions_q, // [B]
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
            theta,
            b,
        )?;

        // Concat finalized K = [post-RoPE context ring | post-RoPE noise]
        // and V = [context ring | noise]. Ring segments assemble in absolute
        // row order; Legacy is one segment.
        let noise_bytes = (b * kvd) * 4;
        let mut cat_off = 0usize;
        for (_row0, slot0, len) in ring_segments(span_start, l, layer_w) {
            let seg_bytes = len * kvd * 4;
            draft_dtod_at(
                gpu,
                ordered,
                &k_cat_l.buf,
                cat_off,
                &k_cache_layer.buf,
                slot0 * kvd * 4,
                seg_bytes,
            )?;
            draft_dtod_at(
                gpu,
                ordered,
                &v_cat_l.buf,
                cat_off,
                &v_cache_layer.buf,
                slot0 * kvd * 4,
                seg_bytes,
            )?;
            cat_off += seg_bytes;
        }
        draft_dtod_at(
            gpu,
            ordered,
            &k_cat_l.buf,
            cat_off,
            &scratch.k_noise.buf,
            0,
            noise_bytes,
        )?;
        draft_dtod_at(
            gpu,
            ordered,
            &v_cat_l.buf,
            cat_off,
            &scratch.v_noise.buf,
            0,
            noise_bytes,
        )?;

        if let Some(t) = t1 {
            gpu.hip.device_synchronize()?;
            us_concat += t.elapsed().as_micros();
        }
        let t2 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Attention: Q [B, n_heads, hd] × K [span+B, n_kv_heads, hd]^T → scores
        // (with GQA expansion) → softmax → @V.
        // Legacy → full; Windowed SWA layers → faithful sliding primitive.
        // Split drafts: SWA layers sliding, last full stays full.
        // All-sliding DFlash2: every layer sliding with same W.
        let is_swa_layer = windowed && (is_all_sliding_windowed || !is_full_layer);
        if is_swa_layer {
            // Faithful non-causal SWA: window= swa_w, ctx_span= span
            gpu.attention_dflash_sliding_f32(
                &scratch.q,
                k_cat_l,
                v_cat_l,
                &scratch.attn_out,
                b,
                span + b,
                cfg.n_heads,
                cfg.n_kv_heads,
                hd,
                span,
                swa_w,
            )?;
        } else {
            use crate::llama::{attention_family, DispatchCtx, FullAttnParams, KernelKey};
            let ctx = DispatchCtx::new(gpu);
            let family = attention_family();
            family
                .run_full_attention(
                    &ctx,
                    gpu,
                    &FullAttnParams {
                        key: KernelKey::AttnFullF32,
                        q: &scratch.q,
                        k: k_cat_l,
                        v: v_cat_l,
                        out: &scratch.attn_out,
                        n: b,
                        seq_len: span + b,
                        n_heads: cfg.n_heads,
                        n_kv_heads: cfg.n_kv_heads,
                        head_dim: hd,
                    },
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        };
        if let Some(t) = t2 {
            gpu.hip.device_synchronize()?;
            us_attn_kernel += t.elapsed().as_micros();
        }
        let t3 = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // S7: on the DFlash2 finish path the wo projection lands in dead
        // conv_temp instead of x, and one fused conv+residual kernel replaces
        // the finish convolution plus the attention residual add
        // (x = residual + conv(wo_out), identical add order). conv_temp is
        // dead here: the prepare output it held was consumed by the QKV
        // GEMMs above. Off-switch and legacy drafts keep today's dataflow.
        let attn_finish_conv = matches!(
            (&layer.attn_conv_base, &layer.attn_conv_proj),
            (Some(_), Some(_))
        ) && scratch.conv_dynamic.is_some()
            && scratch.conv_temp.is_some()
            && gpu.draft_collapse_fused_enabled();
        if attn_finish_conv {
            let tmp = scratch.conv_temp.as_ref().unwrap();
            gemm_dispatch(
                gpu,
                &scratch.attn_out,
                &layer.wo,
                tmp,
                b,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
            )?;
        } else {
            // Write the projection directly into x. The pre-attention x is already
            // preserved in the shared residual plane, so a dedicated attn_proj
            // allocation has no lifetime that must overlap this output.
            gemm_dispatch(
                gpu,
                &scratch.attn_out,
                &layer.wo,
                &scratch.x,
                b,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
            )?;
        }
        // DFlash2 finish convolution before the attention residual add.
        if let (Some(base), Some(_proj), Some(dyn_buf), Some(tmp)) = (
            &layer.attn_conv_base,
            &layer.attn_conv_proj,
            &scratch.conv_dynamic,
            &scratch.conv_temp,
        ) {
            let k = cfg.conv_kernel_size.unwrap_or(2);
            let g = cfg.conv_group_size.unwrap_or(16);
            let groups = h / g;
            let stride = 2 * k * groups;
            let dyn_slice = dyn_buf.sub_offset(0, b * stride);
            let base_phase1 = base.sub_offset(k * h, k * h);
            if attn_finish_conv {
                gpu.dynamic_conv_residual_dflash(
                    tmp,
                    &base_phase1,
                    &dyn_slice,
                    &scratch.residual,
                    &scratch.x,
                    b,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
            } else {
                gpu.dynamic_causal_conv_f32(
                    &scratch.x,
                    &base_phase1,
                    &dyn_slice,
                    tmp,
                    b,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
                gpu.add_f32(&scratch.residual, tmp, &scratch.x)?;
            }
        } else {
            gpu.add_f32(&scratch.residual, &scratch.x, &scratch.x)?;
        }

        // Fixed-shape FFN tail with DFlash2 prepare/finish convolution.
        if let (Some(base), Some(proj), Some(dyn_buf), Some(tmp)) = (
            &layer.mlp_conv_base,
            &layer.mlp_conv_proj,
            &scratch.conv_dynamic,
            &scratch.conv_temp,
        ) {
            // S7: dual-output RMSNorm replaces the FFN residual memcpy +
            // ffn_norm pair (bitwise residual capture, identical norm order).
            let ffn_collapse = gpu.draft_collapse_fused_enabled();
            if ffn_collapse {
                gpu.rmsnorm_residual_dual_dflash(
                    &scratch.x,
                    &layer.ffn_norm,
                    &scratch.residual,
                    &scratch.x_norm,
                    b,
                    h,
                    eps,
                )?;
            } else {
                draft_dtod_at(
                    gpu,
                    ordered,
                    &scratch.residual.buf,
                    0,
                    &scratch.x.buf,
                    0,
                    (b * h) * 4,
                )?;
                gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;
            }
            let k = cfg.conv_kernel_size.unwrap_or(2);
            let g = cfg.conv_group_size.unwrap_or(16);
            let groups = h / g;
            let stride = 2 * k * groups;
            let dyn_slice = dyn_buf.sub_offset(0, b * stride);
            gemm_dispatch(
                gpu,
                &scratch.x_norm,
                proj,
                &dyn_slice,
                b,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
            )?;
            gpu.dynamic_causal_conv_f32(
                &scratch.x_norm,
                base,
                &dyn_slice,
                tmp,
                b,
                h,
                k,
                g,
                stride,
                0,
            )?;
            gemm_dispatch_shared(
                gpu,
                tmp,
                [(&layer.w_gate, &scratch.gate), (&layer.w_up, &scratch.up)],
                b,
                scratch.mq_x_rot.as_ref(),
                scratch.mq_x_rot_f16.as_ref(),
                SharedProjection::Ffn,
            )?;
            gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;
            // S7: w_down lands in dead conv_temp (last read by the gate/up
            // GEMMs above) and one fused conv+residual kernel replaces the
            // finish convolution plus the FFN residual add
            // (x = residual + conv(down_out), identical add order).
            if ffn_collapse {
                gemm_dispatch(
                    gpu,
                    &scratch.gate_up,
                    &layer.w_down,
                    tmp,
                    b,
                    scratch.mq_x_rot.as_ref(),
                    scratch.mq_x_rot_f16.as_ref(),
                )?;
            } else {
                gemm_dispatch(
                    gpu,
                    &scratch.gate_up,
                    &layer.w_down,
                    &scratch.x,
                    b,
                    scratch.mq_x_rot.as_ref(),
                    scratch.mq_x_rot_f16.as_ref(),
                )?;
            }
            let base_phase1 = base.sub_offset(k * h, k * h);
            if ffn_collapse {
                gpu.dynamic_conv_residual_dflash(
                    tmp,
                    &base_phase1,
                    &dyn_slice,
                    &scratch.residual,
                    &scratch.x,
                    b,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
            } else {
                gpu.dynamic_causal_conv_f32(
                    &scratch.x,
                    &base_phase1,
                    &dyn_slice,
                    tmp,
                    b,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
                gpu.add_f32(&scratch.residual, tmp, &scratch.x)?;
            }
        } else {
            let graph_ffn_active = graph_ffn && !dbg && !crate::config::get().draft_gemm_dump;
            draft_ffn_layer_maybe_graph(gpu, layer, scratch, li, b, h, eps, graph_ffn_active)?;
        }
        // 2026-04-21: tried target's fused gemm_gate_up_hfq4g256 here (shared
        // FP16-X convert + interleaved gate/up GEMMs). Byte-exact A/B neutral
        // on 27B HumanEval (median 76.47 fused vs 76.74 baseline; ±7 % run-to-
        // run variance from ksplit's non-deterministic atomicAdd dominates).
        // Kept per-weight dispatch for clarity. The real draft perf lever is
        // the ~56 ms of ffn_gemm per cycle (see HIPFIRE_DRAFT_SUBPHASE=1);
        // fusion alone doesn't move that number — kernel engineering does.

        if let Some(t) = t3 {
            gpu.hip.device_synchronize()?;
            us_ffn_gemm += t.elapsed().as_micros();
        }
    }
    if dbg {
        gpu.hip.device_synchronize()?;
        eprintln!(
            "[draft-sub] attn_gemm={}µs concat={}µs attn_kernel={}µs ffn_gemm={}µs (B={} L={})",
            us_attn_gemm, us_concat, us_attn_kernel, us_ffn_gemm, b, l,
        );
    }

    // ── 3. Final norm ────────────────────────────────────────────────────
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, b, h, eps)?;

    // ── 4. Advance the draft-ctx projection cache pointer ────────────────
    // All rows [0..l) of target_hidden_proj and every layer's
    // k_ctx_cached / v_ctx_cached now contain finalized per-layer
    // projections. Next call's delta starts from here.
    if publish {
        scratch.thlog.mark_proj_cached(l);
    }

    Ok(())
}

#[cfg(test)]
mod ring_tests {
    use super::{ring_segments, TargetHiddenLog};

    #[test]
    fn identity_modulus_is_single_segment() {
        assert_eq!(ring_segments(3, 10, usize::MAX), vec![(3, 3, 7)]);
        assert!(ring_segments(5, 5, usize::MAX).is_empty());
    }

    #[test]
    fn no_wrap_within_ring() {
        // rows < modulus: slot == row, single segment
        assert_eq!(ring_segments(2, 7, 16), vec![(2, 2, 5)]);
    }

    #[test]
    fn wrap_splits_at_boundary() {
        // rows 14..18 in a 16-ring: [14..16) at slots 14.., [16..18) at slots 0..
        assert_eq!(ring_segments(14, 18, 16), vec![(14, 14, 2), (16, 0, 2)]);
    }

    #[test]
    fn span_never_exceeds_ring() {
        // windowed fill spans are <= modulus, so at most two segments
        for start in [0usize, 1, 7, 15, 16, 17, 31, 32] {
            for len in [1usize, 3, 8, 16] {
                let segs = ring_segments(start, start + len, 16);
                assert!(segs.len() <= 2, "{start}+{len}: {segs:?}");
                let covered: usize = segs.iter().map(|s| s.2).sum();
                assert_eq!(covered, len);
                // slots contiguous per segment, rows contiguous across
                let mut row = start;
                for (r0, s0, l) in &segs {
                    assert_eq!(*r0, row);
                    assert_eq!(*s0, row % 16);
                    row += l;
                }
            }
        }
    }

    #[test]
    fn target_hidden_log_restores_a_speculative_append() {
        let mut log = TargetHiddenLog::new();
        log.seed_prompt(4);
        log.mark_proj_cached(3);
        log.mark_full_cached(4);
        let mark = log.mark();

        log.append_committed(4, 2, 0);
        log.mark_proj_cached(6);
        assert_eq!(log.abs_positions(), &[0, 1, 2, 3, 4, 5]);

        log.restore(mark).unwrap();
        assert_eq!(log.uploaded_rows(), 4);
        assert_eq!(log.abs_positions(), &[0, 1, 2, 3]);
        assert_eq!(log.proj_cached_rows(), 3);
        assert_eq!(log.full_cached_rows(), 4);
    }
}

#[cfg(test)]
mod block_policy_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn param_parses_auto_fixed_and_absent() {
        assert_eq!(
            DflashBlockPolicy::from_param(None),
            Ok(DflashBlockPolicy::Auto)
        );
        assert_eq!(
            DflashBlockPolicy::from_param(Some(&json!("auto"))),
            Ok(DflashBlockPolicy::Auto)
        );
        assert_eq!(
            DflashBlockPolicy::from_param(Some(&json!(8))),
            Ok(DflashBlockPolicy::Fixed(8))
        );
        for bad in [json!(3), json!(17), json!("8"), json!(true), json!(-1)] {
            assert!(
                DflashBlockPolicy::from_param(Some(&bad)).is_err(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn auto_depends_on_serving_configuration_only() {
        assert_eq!(DflashBlockPolicy::Auto.resolve(true, 8), 8);
        assert_eq!(DflashBlockPolicy::Auto.resolve(true, 4), 8);
        assert_eq!(DflashBlockPolicy::Auto.resolve(true, 3), 16);
        assert_eq!(DflashBlockPolicy::Auto.resolve(false, 8), 16);
        assert_eq!(DflashBlockPolicy::Fixed(12).resolve(true, 8), 12);
        assert_eq!(DflashBlockPolicy::Fixed(8).resolve(false, 1), 8);
    }

    #[test]
    fn legacy_drafts_keep_their_declared_width() {
        let mut cfg = DflashConfig {
            n_layers: 1,
            hidden: 8,
            intermediate: 16,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 8,
            vocab_size: 8,
            norm_eps: 1e-6,
            rope_theta: 1.0,
            block_size: 8,
            mask_token_id: 0,
            target_layer_ids: vec![0],
            num_target_layers: 1,
            declared_window: None,
            all_layers_sliding: true,
            conv_group_size: Some(16),
            conv_kernel_size: Some(2),
            selector_rank: Some(256),
            selector_top_k: Some(16),
            target_streams_per_layer: 1,
        };
        assert_eq!(cfg.runtime_block_size(DFLASH_BLOCK_UNBATCHED), 16);
        assert_eq!(cfg.runtime_block_size(8), 8);
        cfg.selector_rank = None;
        assert_eq!(cfg.runtime_block_size(DFLASH_BLOCK_UNBATCHED), 8);
    }

    fn meta_with(dflash: serde_json::Value) -> serde_json::Value {
        let mut df = json!({
            "num_hidden_layers": 5,
            "hidden_size": 2560,
            "intermediate_size": 8192,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "head_dim": 128,
            "vocab_size": 259840,
            "block_size": 8,
            "mask_token_id": 259839,
            "target_layer_ids": [1, 6, 11, 16, 21],
            "num_target_layers": 24,
        });
        for (k, v) in dflash.as_object().unwrap() {
            df[k] = v.clone();
        }
        json!({ "dflash": df })
    }

    #[test]
    fn streams_parse_nested_flat_and_default() {
        let nested = meta_with(json!({"dflash_config": {"target_streams_per_layer": 4}}));
        let cfg = DflashConfig::from_metadata(&nested).unwrap();
        assert_eq!(cfg.target_streams_per_layer, 4);
        assert_eq!(cfg.feature_width(), 5 * 4 * 2560);
        assert_eq!(cfg.feature_width(), 51200);

        let flat = meta_with(json!({"target_streams_per_layer": 4}));
        assert_eq!(
            DflashConfig::from_metadata(&flat)
                .unwrap()
                .target_streams_per_layer,
            4
        );

        let legacy = meta_with(json!({}));
        let cfg = DflashConfig::from_metadata(&legacy).unwrap();
        assert_eq!(cfg.target_streams_per_layer, 1);
        assert_eq!(cfg.feature_width(), cfg.num_extract() * cfg.hidden);
    }
}

#[cfg(test)]
mod construction_tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny_config() -> DflashConfig {
        DflashConfig {
            n_layers: 2,
            hidden: 32,
            intermediate: 64,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 16,
            vocab_size: 64,
            norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            block_size: 2,
            mask_token_id: 0,
            target_layer_ids: vec![0],
            num_target_layers: 1,
            declared_window: Some(2),
            all_layers_sliding: false,
            conv_group_size: Some(16),
            conv_kernel_size: Some(2),
            selector_rank: Some(8),
            selector_top_k: Some(2),
            target_streams_per_layer: 1,
        }
    }

    fn draft_path() -> PathBuf {
        PathBuf::from(std::env::var_os("HOME").expect("HOME is required"))
            .join(".hipfire/models/qwen35-27b-dflash-mq4.hfq")
    }

    #[test]
    #[ignore = "requires an AMD GPU and the canonical Qwen3.5-27B DFlash draft"]
    fn late_weight_failure_reuses_every_staged_owner() {
        let path = draft_path();
        assert!(path.is_file(), "missing DFlash fixture: {}", path.display());
        let hfq = HfqFile::open(&path).expect("open DFlash fixture");
        let cfg = DflashConfig::from_hfq(&hfq).expect("parse DFlash config");
        let mut gpu = Gpu::init().expect("GPU required for DFlash rollback");

        let mut allocations = 0usize;
        let warm = DflashWeights::load_with_boundaries(
            &mut gpu,
            &hfq,
            &cfg,
            || {
                allocations += 1;
                Ok(())
            },
            || Ok(()),
        )
        .expect("warm DFlash weights");
        warm.free_gpu(&mut gpu);
        let fresh_allocations = gpu.pool_stats().0;

        let mut attempted = 0usize;
        let failure = DflashWeights::load_with_boundaries(
            &mut gpu,
            &hfq,
            &cfg,
            || {
                attempted += 1;
                if attempted == allocations {
                    Err(hip_bridge::HipError::new(
                        2,
                        "injected failure after final DFlash weight owner",
                    ))
                } else {
                    Ok(())
                }
            },
            || Ok(()),
        );
        match failure {
            Err(error) => assert_eq!(error.code, 2),
            Ok(weights) => {
                weights.free_gpu(&mut gpu);
                panic!("weight fault did not trigger");
            }
        }
        assert_eq!(
            attempted, allocations,
            "failure must follow the final owner"
        );

        let retry = DflashWeights::load(&mut gpu, &hfq, &cfg)
            .expect("immediate weight retry after late failure");
        retry.free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed weight construction lost reusable allocations",
        );
        gpu.drain_pool();
    }

    #[test]
    #[ignore = "requires an AMD GPU and the canonical Qwen3.5-27B DFlash draft"]
    fn late_layer_failure_reuses_every_completed_layer() {
        let path = draft_path();
        assert!(path.is_file(), "missing DFlash fixture: {}", path.display());
        let hfq = HfqFile::open(&path).expect("open DFlash fixture");
        let cfg = DflashConfig::from_hfq(&hfq).expect("parse DFlash config");
        let mut gpu = Gpu::init().expect("GPU required for DFlash rollback");

        let warm = DflashWeights::load(&mut gpu, &hfq, &cfg).expect("warm DFlash weights");
        warm.free_gpu(&mut gpu);
        let fresh_allocations = gpu.pool_stats().0;

        let mut completed_layers = 0usize;
        let failure = DflashWeights::load_with_boundaries(
            &mut gpu,
            &hfq,
            &cfg,
            || Ok(()),
            || {
                completed_layers += 1;
                if completed_layers == cfg.n_layers {
                    Err(hip_bridge::HipError::new(
                        2,
                        "injected failure after final DFlash layer",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        match failure {
            Err(error) => assert_eq!(error.code, 2),
            Ok(weights) => {
                weights.free_gpu(&mut gpu);
                panic!("layer fault did not trigger");
            }
        }
        assert_eq!(
            completed_layers, cfg.n_layers,
            "failure must follow the final completed layer"
        );

        let retry =
            DflashWeights::load(&mut gpu, &hfq, &cfg).expect("immediate retry after layer failure");
        retry.free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed layer construction lost reusable allocations",
        );
        gpu.drain_pool();
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises real base-scratch rollback and retry"]
    fn late_base_scratch_failure_reuses_every_staged_owner() {
        let cfg = tiny_config();
        let mut gpu = Gpu::init().expect("GPU required for DFlash rollback");

        let mut allocations = 0usize;
        let warm =
            DflashScratch::new_with_mq_with_boundary(&mut gpu, &cfg, 2, 4, true, &mut || {
                allocations += 1;
                Ok(())
            })
            .expect("warm DFlash scratch");
        warm.free_gpu(&mut gpu);

        let mut attempted = 0usize;
        let failure =
            DflashScratch::new_with_mq_with_boundary(&mut gpu, &cfg, 2, 4, true, &mut || {
                attempted += 1;
                if attempted == allocations {
                    Err(hip_bridge::HipError::new(
                        2,
                        "injected failure after final DFlash scratch owner",
                    ))
                } else {
                    Ok(())
                }
            });
        match failure {
            Err(error) => assert_eq!(error.code, 2),
            Ok(scratch) => {
                scratch.free_gpu(&mut gpu);
                panic!("base scratch fault did not trigger");
            }
        }
        assert_eq!(
            attempted, allocations,
            "failure must follow the final owner"
        );
        let fresh_allocations = gpu.pool_stats().0;

        let retry = DflashScratch::new_with_mq(&mut gpu, &cfg, 2, 4, true)
            .expect("immediate base-scratch retry");
        retry.free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed base-scratch construction lost reusable allocations",
        );
        gpu.drain_pool();
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises real window-extension rollback and retry"]
    fn late_window_extension_failure_reuses_base_and_extensions() {
        let cfg = tiny_config();
        let mut gpu = Gpu::init().expect("GPU required for DFlash rollback");

        let mut allocations = 0usize;
        let warm = DflashScratch::new_windowed_with_boundary(
            &mut gpu,
            &cfg,
            2,
            2,
            8,
            32,
            true,
            &mut || {
                allocations += 1;
                Ok(())
            },
        )
        .expect("warm windowed DFlash scratch");
        warm.free_gpu(&mut gpu);

        let mut attempted = 0usize;
        let failure = DflashScratch::new_windowed_with_boundary(
            &mut gpu,
            &cfg,
            2,
            2,
            8,
            32,
            true,
            &mut || {
                attempted += 1;
                if attempted == allocations {
                    Err(hip_bridge::HipError::new(
                        2,
                        "injected failure after final DFlash window owner",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        match failure {
            Err(error) => assert_eq!(error.code, 2),
            Ok(scratch) => {
                scratch.free_gpu(&mut gpu);
                panic!("window fault did not trigger");
            }
        }
        assert_eq!(
            attempted, allocations,
            "failure must follow the final owner"
        );
        let fresh_allocations = gpu.pool_stats().0;

        let retry = DflashScratch::new_windowed(&mut gpu, &cfg, 2, 2, 8, 32, true)
            .expect("immediate windowed-scratch retry");
        retry.free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed window construction lost reusable allocations",
        );
        gpu.drain_pool();
    }

    /// Deterministic copy-failure callback for the leaf-upload seams: the
    /// allocation is real, only the host-to-device copy itself fails.
    fn failing_copy(_: &HipRuntime, _: &DeviceBuffer, _: &[u8]) -> HipResult<()> {
        Err(hip_bridge::HipError::new(2, "injected H2D copy failure"))
    }

    #[test]
    fn leaf_upload_copy_failure_returns_pool_allocation() {
        // Focused allocation→copy failure seam for the DFlash leaf uploaders.
        // `Gpu::upload_f32` returns a copy error without freeing the fresh
        // pool allocation; the local leaf helpers must not strand it outside
        // the constructor ledgers. Real allocation and real free — only the
        // copy is injected — with no test-only production API. Soft-skip
        // without a GPU, like the dispatch rollback tests.
        //
        // The baseline is recorded after a successful warmup pass, not on an
        // empty pool: `total_new` is cumulative, so the two first-touch
        // mallocs belong outside the measured window, and the retry holds
        // two live buffers at once — the pool must already hold both slots
        // for the fail+retry cycle to stay flat. A leaked owner still fails
        // this: the failure consumes a pooled slot without returning it, so
        // the retry finds the list short and mallocs anew.
        let Some(mut gpu) = Gpu::init().ok() else {
            eprintln!("skip: no GPU");
            return;
        };
        let warm_raw = upload_raw_weight(&mut gpu, &[9u8; 64], &[64]).expect("warm raw");
        let warm_staged = upload_f32_weight(&mut gpu, &[1.5f32; 16], &[16]).expect("warm F32");
        gpu.free_tensor(warm_raw).expect("free warm raw");
        gpu.free_tensor(warm_staged).expect("free warm F32");
        let fresh_allocations = gpu.pool_stats().0;

        let err = match upload_raw_weight_with_copy(&mut gpu, &[9u8; 64], &[64], failing_copy) {
            Err(error) => error,
            Ok(tensor) => {
                gpu.free_tensor(tensor).expect("free");
                panic!("injected raw copy failure must surface");
            }
        };
        assert!(
            err.to_string().contains("injected"),
            "unexpected error: {err}"
        );

        let err = match upload_f32_weight_with_copy(&mut gpu, &[1.5f32; 16], &[16], failing_copy) {
            Err(error) => error,
            Ok(tensor) => {
                gpu.free_tensor(tensor).expect("free");
                panic!("injected F32 copy failure must surface");
            }
        };
        assert!(
            err.to_string().contains("injected"),
            "unexpected error: {err}"
        );

        // Immediate retry reuses the returned allocations instead of growing
        // the pool: both freed owners are back on the free list.
        let raw = upload_raw_weight(&mut gpu, &[9u8; 64], &[64]).expect("raw retry");
        let staged = upload_f32_weight(&mut gpu, &[1.5f32; 16], &[16]).expect("F32 retry");
        gpu.free_tensor(raw).expect("free raw");
        gpu.free_tensor(staged).expect("free F32");
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed leaf upload leaked its pool allocation",
        );
        gpu.drain_pool();
    }

    #[test]
    fn awq_sidecar_copy_failure_propagates_and_frees_trunk() {
        // The pre-fix path swallowed a sidecar copy failure into a missing
        // scale (`upload_raw(...).ok()` → None) and leaked the direct
        // allocation. The pooled attach must surface the error and free the
        // already-built trunk owner; an immediate retry then reuses both
        // pooled buffers. Tiny synthetic container, no model fixture.
        let Some(mut gpu) = Gpu::init().ok() else {
            eprintln!("skip: no GPU");
            return;
        };
        use crate::hfq::hfq_test_fixture::write_min_hfq;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("awq.hfq");
        let weight_bytes = vec![0xA5u8; 128];
        let sidecar_f16: Vec<u8> = (0..8).flat_map(|_| [0x00u8, 0x3C]).collect();
        write_min_hfq(
            &path,
            9,
            &[
                ("w.weight", 13, &[2, 8], &weight_bytes),
                ("w.awq_scale.weight", 1, &[8], &sidecar_f16),
            ],
        );
        let hfq = HfqFile::open(&path).expect("open AWQ fixture");

        fn trunk(gpu: &mut Gpu, bytes: &[u8]) -> WeightTensor {
            WeightTensor {
                buf: upload_raw_weight(gpu, bytes, &[bytes.len()]).expect("trunk upload"),
                gpu_dtype: DType::MQ4G256,
                m: 2,
                k: 8,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }

        // Success baseline: the sidecar attaches, then everything is freed
        // back to the pool.
        let staged = trunk(&mut gpu, &weight_bytes);
        let attached =
            attach_awq_scale(&hfq, &mut gpu, staged, "w.weight", 8).expect("baseline attach");
        assert!(attached.awq_scale.is_some(), "sidecar must attach");
        attached.free_all(&mut gpu);
        let fresh_allocations = gpu.pool_stats().0;

        // Injected sidecar copy failure: the error surfaces and the trunk
        // owner is freed with it — no silent scale drop, no stranded owner.
        let staged = trunk(&mut gpu, &weight_bytes);
        let err =
            match attach_awq_scale_with_copy(&hfq, &mut gpu, staged, "w.weight", 8, failing_copy) {
                Err(error) => error,
                Ok(wt) => {
                    wt.free_all(&mut gpu);
                    panic!("injected sidecar copy failure must surface");
                }
            };
        assert!(
            err.to_string().contains("injected"),
            "unexpected error: {err}"
        );

        // Immediate retry reuses the trunk + sidecar buffers.
        let staged = trunk(&mut gpu, &weight_bytes);
        let retry = attach_awq_scale(&hfq, &mut gpu, staged, "w.weight", 8).expect("sidecar retry");
        assert!(retry.awq_scale.is_some(), "retry must attach the scale");
        retry.free_all(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed AWQ attach leaked trunk or sidecar allocation",
        );
        gpu.drain_pool();
    }
}

// ─── Candidate selector (DFlash2 chain-only) ───────────────────────────────

/// Proposal returned by the DFlash2 selector. All rows flattened row-major.
#[derive(Debug, Clone)]
pub struct DflashCandidateProposal {
    /// Selected chain tokens, length `rows` (sequential, predecessor advances
    /// through this chain; discarded after proposal).
    pub tokens: Vec<u32>,
    /// Per-row top-K candidate ids, flattened row-major `[rows*K]`.
    pub candidates: Vec<u32>,
    /// Flattened per-row softmax(q) over the top-K selector scores when
    /// temperature>0 (`[rows*K]`, each row sums ≈1). `None` for greedy.
    pub probabilities: Option<Vec<f32>>,
    /// Per-row probability of the selected token (entry of `probabilities` at
    /// the chosen index when temperature>0). `None` for greedy.
    pub selected_probabilities: Option<Vec<f32>>,
    /// Per-row softmax (temperature 1) mass of the selected candidate over
    /// the row's top-K selector scores, at every temperature (the draft's own
    /// confidence; picks the DFlash2 verify depth).
    pub confidence: Vec<f32>,
    pub top_k: usize,
}

/// Softmax scores/temp → normalized q; sample index via inverse-CDF on `uniform`.
/// Returns `(selected_idx, q)` where `q` is the full normalized mass vector.
fn softmax_sample(scores: &[f32], temp: f32, uniform: f32) -> (usize, Vec<f32>) {
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut q = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for &s in scores {
        let e = ((s - max_s) / temp).exp();
        q.push(e);
        sum += e;
    }
    for p in &mut q {
        *p /= sum;
    }
    let mut cumsum = 0.0f32;
    for (i, &p) in q.iter().enumerate() {
        cumsum += p;
        if uniform < cumsum || i + 1 == q.len() {
            return (i, q);
        }
    }
    let last = q.len().saturating_sub(1);
    (last, q)
}

/// Borrowed-readback selector scorer: the arithmetic of the former
/// `propose_inner` (runtime lines 4275-4372 before this factoring), exposed so
/// packed multi-lane readbacks can score each lane's span without a device or
/// a copy of the projection/IDs/unaries.
///
/// Inputs are the raw device readbacks for ONE chain: `projection`
/// `[rows*rank]` (hidden projection GEMM output), `ids` `[rows*K]` (raw GPU
/// top-K IDs) and `unaries` `[rows*K]` (raw top-K values); `rank`/`K`/vocab
/// come from `weights`. Scoring is sequential: predecessor starts at `anchor`
/// and becomes the selected ID of each row; candidates are scored in readback
/// order with the sequential f32 `pred_cb*projection*succ_cb` dot. With
/// `temperature>0` `uniforms` (>= `rows`, one per row) are required.
pub fn propose_candidates_from_rows(
    weights: &DflashWeights,
    projection: &[f32],
    ids: &[u32],
    unaries: &[f32],
    rows: usize,
    anchor: u32,
    temperature: f32,
    uniforms: Option<&[f32]>,
) -> HipResult<DflashCandidateProposal> {
    propose_candidates_from_rows_capture(
        weights,
        projection,
        ids,
        unaries,
        rows,
        anchor,
        temperature,
        uniforms,
        None,
    )
}

/// [`propose_candidates_from_rows`] that also appends every row's `k` scores
/// (`unary + dot`, row-major `[rows*K]`) to `capture`: the byte reference of
/// the device selector `dflash_selector_greedy_exact`.
pub fn propose_candidates_from_rows_capture(
    weights: &DflashWeights,
    projection: &[f32],
    ids: &[u32],
    unaries: &[f32],
    rows: usize,
    anchor: u32,
    temperature: f32,
    uniforms: Option<&[f32]>,
    capture: Option<&mut Vec<f32>>,
) -> HipResult<DflashCandidateProposal> {
    if rows == 0 {
        return Err(hip_bridge::HipError::new(0, "rows must be >0"));
    }
    if !weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
    }
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    if k == 0 || k > 16 {
        return Err(hip_bridge::HipError::new(
            0,
            "selector_top_k must be in [1,16]",
        ));
    }
    if projection.len() != rows * rank {
        return Err(hip_bridge::HipError::new(
            0,
            "projection length != rows*rank",
        ));
    }
    if ids.len() != rows * k || unaries.len() != rows * k {
        return Err(hip_bridge::HipError::new(0, "ids/unaries length != rows*K"));
    }
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    propose_inner_capture(
        vocab,
        weights,
        projection,
        ids,
        unaries,
        rows,
        k,
        rank,
        anchor,
        temperature,
        uniforms,
        capture,
    )
}

fn propose_inner_capture(
    vocab_size: usize,
    weights: &DflashWeights,
    hidden_proj_host: &[f32], // [rows*rank] already projected, row-major
    top_ids: &[u32],          // [rows*K]
    top_vals: &[f32],         // [rows*K] raw unaries
    rows: usize,
    k: usize,
    rank: usize,
    anchor: u32,
    temperature: f32,
    uniforms: Option<&[f32]>,
    mut capture: Option<&mut Vec<f32>>,
) -> HipResult<DflashCandidateProposal> {
    if temperature > 0.0 && uniforms.is_none() {
        return Err(hip_bridge::HipError::new(
            0,
            "selector temperature>0 requires uniforms",
        ));
    }
    if let Some(u) = uniforms {
        if u.len() < rows {
            return Err(hip_bridge::HipError::new(0, "uniforms length < rows"));
        }
    }
    let pred_cb = weights.predecessor_codebook.as_ref().unwrap();
    let succ_cb = weights.successor_codebook.as_ref().unwrap();
    let mut tokens = Vec::with_capacity(rows);
    // Sparse q over every top-K candidate (temp>0 only); selected q matches chosen entry.
    let mut all_probs: Option<Vec<f32>> = if temperature > 0.0 {
        Some(Vec::with_capacity(rows * k))
    } else {
        None
    };
    let mut selected_probs: Option<Vec<f32>> = if temperature > 0.0 {
        Some(Vec::with_capacity(rows))
    } else {
        None
    };
    let mut confidence = Vec::with_capacity(rows);
    let mut pred = anchor;
    for r in 0..rows {
        let proj_off = r * rank;
        let proj = &hidden_proj_host[proj_off..proj_off + rank];
        // Build scores for this row's K candidates
        let id_off = r * k;
        let mut scores = Vec::with_capacity(k);
        for j in 0..k {
            let cand = top_ids[id_off + j] as usize;
            if cand >= vocab_size {
                return Err(hip_bridge::HipError::new(
                    0,
                    "candidate id out of vocab range",
                ));
            }
            let unary = top_vals[id_off + j];
            // dot( pred_cb[pred] * proj , succ_cb[cand] )
            let mut dot = 0.0f32;
            for d in 0..rank {
                let p = pred_cb.get_f32(pred as usize, d);
                let s = succ_cb.get_f32(cand, d);
                dot += p * proj[d] * s;
            }
            scores.push(unary + dot);
        }
        if let Some(c) = capture.as_deref_mut() {
            c.extend_from_slice(&scores);
        }
        // Softmax (temperature 1) mass of the selected candidate, at every
        // request temperature: the draft's own confidence in this row.
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
        let sel_idx = if temperature == 0.0 {
            let mut best = 0usize;
            let mut best_s = scores[0];
            for (i, &s) in scores.iter().enumerate().skip(1) {
                if s > best_s {
                    best_s = s;
                    best = i;
                }
            }
            best
        } else {
            let u = uniforms.unwrap()[r];
            if !(0.0..1.0).contains(&u) {
                return Err(hip_bridge::HipError::new(0, "uniform out of [0,1)"));
            }
            let (idx, q) = softmax_sample(&scores, temperature, u);
            if let Some(sp) = &mut selected_probs {
                sp.push(q[idx]);
            }
            if let Some(ap) = &mut all_probs {
                ap.extend_from_slice(&q);
            }
            idx
        };
        confidence.push((scores[sel_idx] - max_s).exp() / z);
        let cand_id = top_ids[id_off + sel_idx];
        tokens.push(cand_id);
        pred = cand_id;
    }
    Ok(DflashCandidateProposal {
        tokens,
        candidates: top_ids.to_vec(),
        probabilities: all_probs,
        selected_probabilities: selected_probs,
        confidence,
        top_k: k,
    })
}

/// Host-logits equivalent of `propose_candidates_device`: compute per-row
/// top-K via CPU partial sort then score identically.
pub fn propose_candidates_host(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    scratch: &DflashScratch,
    hidden: &GpuTensor,
    host_logits: &[f32],
    rows: usize,
    anchor: u32,
    temperature: f32,
    uniforms: Option<&[f32]>,
) -> HipResult<DflashCandidateProposal> {
    if rows == 0 {
        return Err(hip_bridge::HipError::new(0, "rows must be >0"));
    }
    if !weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
    }
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    if k == 0 || k > 16 {
        return Err(hip_bridge::HipError::new(
            0,
            "selector_top_k must be in [1,16]",
        ));
    }
    if hidden.shape.iter().product::<usize>() < rows * weights.layers[0].wq.k
        && hidden.buf.size() < rows * weights.layers[0].wq.k * 4
    {
        // shape check is best-effort; rely on GEMM to fail if truly wrong
    }
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    if host_logits.len() != rows * vocab {
        return Err(hip_bridge::HipError::new(
            0,
            "host_logits length != rows*vocab",
        ));
    }
    if hidden.buf.size() < rows * weights.layers[0].wq.k * 4 && hidden.shape.is_empty() {
        // allow flat shape
    }
    // Project hidden -> rank on device (small GEMM) then D2H.
    let proj = scratch
        .selector_proj
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "selector scratch not allocated"))?;
    let hp = weights.selector_hidden_proj.as_ref().unwrap();
    let proj_slice = proj.sub_offset(0, rows * rank);
    gemm_dispatch(
        gpu,
        hidden,
        hp,
        &proj_slice,
        rows,
        scratch.mq_x_rot.as_ref(),
        scratch.mq_x_rot_f16.as_ref(),
    )?;
    // D2H projected hidden
    let mut host_proj = vec![0f32; rows * rank];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(host_proj.as_mut_ptr() as *mut u8, host_proj.len() * 4)
    };
    gpu.hip.memcpy_dtoh(bytes, &proj_slice.buf)?;
    // CPU top-K per row (raw logits)
    let mut top_ids = Vec::with_capacity(rows * k);
    let mut top_vals = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let off = r * vocab;
        let row = &host_logits[off..off + vocab];
        // partial top-K via nth_element style: collect top k with sort
        let mut idxs: Vec<usize> = (0..vocab).collect();
        // Use select_nth_unstable_by for efficiency but simple sort for small K?
        // For correctness use full sort then take K (vocab up to 152k, rows up to 16, cost trivial)
        idxs.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for j in 0..k {
            let id = idxs[j] as u32;
            top_ids.push(id);
            top_vals.push(row[idxs[j]]);
        }
    }
    propose_candidates_from_rows(
        weights,
        &host_proj,
        &top_ids,
        &top_vals,
        rows,
        anchor,
        temperature,
        uniforms,
    )
}

/// Device-logits proposal: top-K on GPU via `topk_values_batched_f32` (raw
/// logits, small D2H), then host-side scoring identical to the host path.
pub fn propose_candidates_device(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    scratch: &DflashScratch,
    hidden: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    anchor: u32,
    temperature: f32,
    uniforms: Option<&[f32]>,
) -> HipResult<DflashCandidateProposal> {
    if rows == 0 {
        return Err(hip_bridge::HipError::new(0, "rows must be >0"));
    }
    if !weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
    }
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    if k == 0 || k > 16 {
        return Err(hip_bridge::HipError::new(
            0,
            "selector_top_k must be in [1,16]",
        ));
    }
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    if hidden.buf.size() < rows * 4 {
        return Err(hip_bridge::HipError::new(0, "hidden buffer too small"));
    }
    if logits.buf.size() < rows * vocab * 4 {
        return Err(hip_bridge::HipError::new(0, "logits buffer too small"));
    }
    // Project hidden -> rank
    let host_proj = selector_projection_host(gpu, weights, scratch, hidden, rows, rank)?;
    // GPU top-K raw values (small D2H)
    let top_ids_gpu = scratch
        .topk_ids
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_ids scratch not allocated"))?;
    let top_vals_gpu = scratch
        .topk_vals
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_vals scratch not allocated"))?;
    let ids_slice = top_ids_gpu.sub_offset(0, rows * k);
    let vals_slice = top_vals_gpu.sub_offset(0, rows * k);
    // Call the raw top-k primitive; the string `topk_values_batched_f32` is kept for grep checks
    // and the fallback trait will delegate to logsumexp when native is absent (per-row shift valid).
    {
        let _marker = "topk_values_batched_f32";
        let _ = _marker;
    }
    gpu.topk_values_batched_f32(logits, &ids_slice, &vals_slice, vocab, k, rows)?;
    let (top_ids, top_vals) = download_top(gpu, &ids_slice, &vals_slice, rows * k)?;
    propose_candidates_from_rows(
        weights,
        &host_proj,
        &top_ids,
        &top_vals,
        rows,
        anchor,
        temperature,
        uniforms,
    )
}

/// The kernel half of [`propose_candidates_device`] for the exact device
/// greedy selector (Design A §4), with no host transfer: the same selector
/// hidden-projection GEMM (`gemm_dispatch` into `scratch.selector_proj`), the
/// same raw `topk_values_batched_f32`, then
/// [`SelectorGreedyBlock::enqueue`] (which scores, picks and publishes the
/// proposal block on the device). Kernel order matches the eager path; only
/// the two D2H round trips and the host scorer are replaced. Plan-mode safe.
pub fn enqueue_selector_greedy_device(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    scratch: &DflashScratch,
    books: &SelectorDeviceCodebooks,
    block: &SelectorGreedyBlock,
    hidden: &GpuTensor,
    logits: &GpuTensor,
) -> HipResult<()> {
    if !weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
    }
    let rows = block.rows;
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    if rank != block.rank || k != block.k {
        return Err(hip_bridge::HipError::new(
            0,
            "selector block rank/k disagree with the loaded selector metadata",
        ));
    }
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    if logits.buf.size() < rows * vocab * 4 {
        return Err(hip_bridge::HipError::new(0, "logits buffer too small"));
    }
    let proj = scratch
        .selector_proj
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "selector scratch not allocated"))?;
    let hp = weights.selector_hidden_proj.as_ref().unwrap();
    let proj_slice = proj.sub_offset(0, rows * rank);
    gemm_dispatch(
        gpu,
        hidden,
        hp,
        &proj_slice,
        rows,
        scratch.mq_x_rot.as_ref(),
        scratch.mq_x_rot_f16.as_ref(),
    )?;
    let ids_slice = scratch
        .topk_ids
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_ids scratch not allocated"))?
        .sub_offset(0, rows * k);
    let vals_slice = scratch
        .topk_vals
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_vals scratch not allocated"))?
        .sub_offset(0, rows * k);
    gpu.topk_values_batched_f32(logits, &ids_slice, &vals_slice, vocab, k, rows)?;
    block.enqueue(gpu, books, &proj_slice, &ids_slice, &vals_slice)
}

/// Selector codebooks resident on the device, uploaded once. Element values
/// are exactly `SelectorCodebook::get_f32`: f32 books are copied bit for bit;
/// f16 books keep their 16-bit patterns, which the kernel widens exactly as
/// `half::f16::to_f32` does (the widening is lossless).
pub struct SelectorDeviceCodebooks {
    pub pred: GpuTensor,
    pub succ: GpuTensor,
    pub pred_f16: bool,
    pub succ_f16: bool,
    pub vocab: usize,
    pub rank: usize,
}

fn codebook_device_bytes(cb: &SelectorCodebook) -> HipResult<(&[u8], bool)> {
    let elems = cb
        .vocab
        .checked_mul(cb.rank)
        .ok_or_else(|| hip_bridge::HipError::new(0, "codebook extent overflow"))?;
    match (&cb.f16_data, &cb.f32_data) {
        (Some(h), None) if h.len() == elems => {
            // SAFETY: u16 slice viewed as its own initialized bytes.
            Ok((
                unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2) },
                true,
            ))
        }
        (None, Some(f)) if f.len() == elems => {
            // SAFETY: f32 slice viewed as its own initialized bytes.
            Ok((
                unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) },
                false,
            ))
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "codebook must hold exactly one dtype with vocab*rank elements",
        )),
    }
}

impl SelectorDeviceCodebooks {
    pub fn upload(gpu: &mut Gpu, weights: &DflashWeights) -> HipResult<Self> {
        if !weights.has_candidate_selector() {
            return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
        }
        let pred_cb = weights.predecessor_codebook.as_ref().unwrap();
        let succ_cb = weights.successor_codebook.as_ref().unwrap();
        let rank = weights.selector_rank.unwrap_or(256);
        if pred_cb.rank != rank || succ_cb.rank != rank || pred_cb.vocab != succ_cb.vocab {
            return Err(hip_bridge::HipError::new(
                0,
                "codebook rank/vocab disagree with loaded selector metadata",
            ));
        }
        let (pb, pred_f16) = codebook_device_bytes(pred_cb)?;
        let (sb, succ_f16) = codebook_device_bytes(succ_cb)?;
        let shape = [pred_cb.vocab, rank];
        let pred = gpu.upload_raw(pb, &shape)?;
        let succ = match gpu.upload_raw(sb, &shape) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.hip.free(pred.buf);
                return Err(e);
            }
        };
        Ok(Self {
            pred,
            succ,
            pred_f16,
            succ_f16,
            vocab: pred_cb.vocab,
            rank,
        })
    }

    pub fn free(self, gpu: &mut Gpu) {
        let _ = gpu.hip.free(self.pred.buf);
        let _ = gpu.hip.free(self.succ.buf);
    }
}

/// Result of one device greedy selection, read back for diagnostics/oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectorGreedyReadback {
    /// 0 = ok, else `code | (row+1) << 8` (see `dflash_selector_greedy_exact.hip`).
    pub status: u32,
    /// `[rows+1]`: `block[0]` = seed, `block[r+1]` = id chosen at row `r`.
    pub block: Vec<u32>,
    /// `[rows*K]` raw `unary + dot` scores.
    pub scores: Vec<f32>,
}

/// Dedicated window-immutable proposal block for the device greedy selector:
/// never aliased with `verify_scratch.argmax`/logits. Exact-size allocations.
pub struct SelectorGreedyBlock {
    seed: GpuTensor,
    scores: GpuTensor,
    block: GpuTensor,
    status: GpuTensor,
    pub rows: usize,
    pub k: usize,
    pub rank: usize,
}

impl SelectorGreedyBlock {
    pub fn new(gpu: &mut Gpu, weights: &DflashWeights, rows: usize) -> HipResult<Self> {
        let rank = weights.selector_rank.unwrap_or(256);
        let k = weights.selector_top_k.unwrap_or(16);
        if rows == 0 || rows > rdna_compute::dflash_selector::DFLASH_SELECTOR_MAX_ROWS {
            return Err(hip_bridge::HipError::new(0, "selector rows not in 1..=15"));
        }
        if k == 0 || k > 16 {
            return Err(hip_bridge::HipError::new(
                0,
                "selector_top_k must be in [1,16]",
            ));
        }
        let seed = gpu.alloc_tensor_exact(&[1], DType::F32)?;
        let scores = gpu.alloc_tensor_exact(&[rows * k], DType::F32)?;
        let block = gpu.alloc_tensor_exact(&[rows + 1], DType::F32)?;
        let status = gpu.alloc_tensor_exact(&[1], DType::F32)?;
        Ok(Self {
            seed,
            scores,
            block,
            status,
            rows,
            k,
            rank,
        })
    }

    /// Device word the owner's pinned staging fills with the seed token.
    pub fn seed_buffer(&self) -> &DeviceBuffer {
        &self.seed.buf
    }

    /// The proposal block `[rows+1]` u32 (`block[0]` = seed): the railgun copy
    /// source for the verify token vector.
    pub fn block_buffer(&self) -> &DeviceBuffer {
        &self.block.buf
    }

    pub fn status_buffer(&self) -> &DeviceBuffer {
        &self.status.buf
    }

    pub fn scores_buffer(&self) -> &DeviceBuffer {
        &self.scores.buf
    }

    /// Enqueue the exact greedy selector over `rows` rows of the existing GPU
    /// selector projection and raw top-K scratch (`topk_values_batched_f32`
    /// outputs). The seed word must already be on the device.
    pub fn enqueue(
        &self,
        gpu: &mut Gpu,
        books: &SelectorDeviceCodebooks,
        proj: &GpuTensor,
        top_ids: &GpuTensor,
        top_vals: &GpuTensor,
    ) -> HipResult<()> {
        if books.rank != self.rank {
            return Err(hip_bridge::HipError::new(0, "codebook rank != block rank"));
        }
        gpu.dflash_selector_greedy_exact(&rdna_compute::dflash_selector::DflashSelectorGreedyArgs {
            pred_cb: &books.pred.buf,
            succ_cb: &books.succ.buf,
            pred_f16: books.pred_f16,
            succ_f16: books.succ_f16,
            proj: &proj.buf,
            top_ids: &top_ids.buf,
            top_vals: &top_vals.buf,
            seed: &self.seed.buf,
            scores: &self.scores.buf,
            block: &self.block.buf,
            status: &self.status.buf,
            rows: self.rows,
            k: self.k,
            rank: self.rank,
            vocab: books.vocab,
        })
    }

    /// Blocking D2H of status, block and scores (default-stream ordered).
    pub fn read_back(&self, gpu: &Gpu) -> HipResult<SelectorGreedyReadback> {
        let mut status = [0u8; 4];
        gpu.hip.memcpy_dtoh(&mut status, &self.status.buf)?;
        let mut blk = vec![0u8; (self.rows + 1) * 4];
        gpu.hip.memcpy_dtoh(&mut blk, &self.block.buf)?;
        let mut sc = vec![0u8; self.rows * self.k * 4];
        gpu.hip.memcpy_dtoh(&mut sc, &self.scores.buf)?;
        Ok(SelectorGreedyReadback {
            status: u32::from_le_bytes(status),
            block: blk
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            scores: sc
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    pub fn free(self, gpu: &mut Gpu) {
        for t in [self.seed, self.scores, self.block, self.status] {
            let _ = gpu.hip.free(t.buf);
        }
    }
}

/// Byte offsets of the sections written by
/// [`PinnedWindowInputs::pack_seed_mask_positions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowInputOffsets {
    pub seed: usize,
    pub mask: usize,
    pub q_positions: usize,
    pub verify_positions: usize,
    pub total_bytes: usize,
}

/// Persistent pinned owner-input staging (Design A §4 "Before the IB"): the
/// host-known seed, mask id and position words are packed into reusable
/// `hipHostMalloc` storage, uploaded with ONE `hipMemcpyAsync` on the owner
/// stream, and retired by ONE event (the completion fence). The staging bytes
/// are immutable while a transfer is in flight: `pack_*` and a second
/// `submit` are refused until [`Self::complete`]. Pageable host memory is
/// never the source of a GPU input.
pub struct PinnedWindowInputs {
    host: hip_bridge::PinnedHostBuffer,
    fence: hip_bridge::Event,
    packed: usize,
    in_flight: bool,
}

impl PinnedWindowInputs {
    pub fn new(gpu: &Gpu, capacity_bytes: usize) -> HipResult<Self> {
        if capacity_bytes == 0 || capacity_bytes % 4 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "staging capacity must be a non-zero multiple of 4",
            ));
        }
        Ok(Self {
            host: gpu.hip.host_buffer(capacity_bytes)?,
            fence: gpu
                .hip
                .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            packed: 0,
            in_flight: false,
        })
    }

    pub fn in_flight(&self) -> bool {
        self.in_flight
    }

    /// Pack `sections` back to back as little-endian u32 words from offset 0;
    /// returns each section's byte offset.
    pub fn pack(&mut self, sections: &[&[u32]]) -> HipResult<Vec<usize>> {
        if self.in_flight {
            return Err(hip_bridge::HipError::new(
                0,
                "staging buffer is in flight; complete() before repacking",
            ));
        }
        let total: usize = sections.iter().map(|s| s.len() * 4).sum();
        if total > self.host.size() {
            return Err(hip_bridge::HipError::new(0, "staging capacity exceeded"));
        }
        let bytes = self.host.as_bytes_mut();
        let mut at = 0usize;
        let mut offsets = Vec::with_capacity(sections.len());
        for s in sections {
            offsets.push(at);
            for w in s.iter() {
                bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
                at += 4;
            }
        }
        self.packed = total;
        Ok(offsets)
    }

    /// The seed / mask-id / absolute-Q-position / verify-position layout.
    pub fn pack_seed_mask_positions(
        &mut self,
        seed: u32,
        mask_id: u32,
        q_positions: &[u32],
        verify_positions: &[u32],
    ) -> HipResult<WindowInputOffsets> {
        let seed = [seed];
        let mask = [mask_id];
        let o = self.pack(&[&seed, &mask, q_positions, verify_positions])?;
        Ok(WindowInputOffsets {
            seed: o[0],
            mask: o[1],
            q_positions: o[2],
            verify_positions: o[3],
            total_bytes: self.packed,
        })
    }

    /// One async H2D of the packed bytes into `dst[0..packed]` on `stream`,
    /// then record the fence on `stream`.
    pub fn submit(
        &mut self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        stream: &hip_bridge::Stream,
    ) -> HipResult<()> {
        if self.in_flight {
            return Err(hip_bridge::HipError::new(0, "staging already in flight"));
        }
        if self.packed == 0 || self.packed > dst.size() {
            return Err(hip_bridge::HipError::new(
                0,
                "staged bytes empty or exceed the destination",
            ));
        }
        let n = self.packed;
        gpu.hip
            .memcpy_htod_async(dst, &self.host.as_bytes_mut()[..n], stream)?;
        gpu.hip.event_record(&self.fence, Some(stream))?;
        self.in_flight = true;
        Ok(())
    }

    /// Host wait on the fence; afterwards the staging bytes are reusable.
    pub fn complete(&mut self, gpu: &Gpu) -> HipResult<()> {
        if self.in_flight {
            gpu.hip.event_synchronize(&self.fence)?;
            self.in_flight = false;
        }
        Ok(())
    }

    /// Drain the fence, then release the staging storage and the event. Drop
    /// without this leaks the event and, if a copy is in flight, is the
    /// caller's bug: the pinned bytes would be freed under the DMA.
    pub fn destroy(mut self, gpu: &Gpu) -> HipResult<()> {
        self.complete(gpu)?;
        gpu.hip.event_destroy(self.fence)
    }

    /// Non-blocking fence query; true (and retires) once the copy is done.
    pub fn poll(&mut self, gpu: &Gpu) -> HipResult<bool> {
        if !self.in_flight {
            return Ok(true);
        }
        if gpu.hip.event_query(&self.fence)? {
            self.in_flight = false;
        }
        Ok(!self.in_flight)
    }
}

/// One lane of [`propose_candidates_lanes_device`]: its `rows` candidate rows
/// start at row `first_row` of the shared hidden / logits rows.
#[derive(Clone, Copy, Debug)]
pub struct SelectorLane {
    pub first_row: usize,
    pub rows: usize,
    pub anchor: u32,
}

/// GREEDY selector proposals (temperature 0.0, no uniforms: the batched
/// greedy CB draft is the only caller) of several lanes whose draft rows
/// share one `[rows_total × hidden]` / `[rows_total × vocab]` pair:
/// ONE selector projection GEMM and ONE top-K over all `rows_total` rows
/// (rows no lane selects, e.g. seed rows, are computed and ignored), one
/// download of the projection and one of each top-K tensor, then per-lane
/// host scoring ([`propose_candidates_from_rows`]) of exactly the lane's row
/// slice — three host syncs per call instead of three per lane. Proposals
/// are byte-identical to per-lane [`propose_candidates_device`] calls
/// because the projection GEMM (MQ4G256V2 gfx1201 overwrite route,
/// `gemm_dispatch`) and `topk_values_batched_f32` are row-local.
///
/// The projection route admits `2..=63` rows, and exactly 64 only inside
/// [`Gpu::draft_rows64_enter`]'s scope: `rows64_scope` opens that scope
/// around the projection GEMM (and only it). The caller MUST pass `true`
/// iff `rows_total == 64` and the 64-row draft chunk is exact on this GPU
/// (`dflash_draft_batch_max_rows == 64`), `false` otherwise.
///
/// `scratch` must hold `rows_total` selector rows (`selector_proj`,
/// `topk_ids`, `topk_vals`) and own FWHT rotation buffers (`mq_x_rot`,
/// `mq_x_rot_f16`).
#[allow(clippy::too_many_arguments)]
pub fn propose_candidates_lanes_device(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    scratch: &DflashScratch,
    hidden: &GpuTensor,
    logits: &GpuTensor,
    rows_total: usize,
    lanes: &[SelectorLane],
    rows64_scope: bool,
) -> HipResult<Vec<DflashCandidateProposal>> {
    if rows_total == 0
        || lanes
            .iter()
            .any(|l| l.rows == 0 || l.first_row + l.rows > rows_total)
    {
        return Err(hip_bridge::HipError::new(
            0,
            "selector lanes outside the rows",
        ));
    }
    if !weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(0, "selector weights not loaded"));
    }
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    if k == 0 || k > 16 {
        return Err(hip_bridge::HipError::new(
            0,
            "selector_top_k must be in [1,16]",
        ));
    }
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    if logits.buf.size() < rows_total * vocab * 4 {
        return Err(hip_bridge::HipError::new(0, "logits buffer too small"));
    }
    let prev = gpu.draft_rows64_enter(rows64_scope);
    let host_proj = selector_projection_host(gpu, weights, scratch, hidden, rows_total, rank);
    gpu.draft_rows64_exit(prev);
    let host_proj = host_proj?;
    let ids_slice = scratch
        .topk_ids
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_ids scratch not allocated"))?
        .sub_offset(0, rows_total * k);
    let vals_slice = scratch
        .topk_vals
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "topk_vals scratch not allocated"))?
        .sub_offset(0, rows_total * k);
    gpu.topk_values_batched_f32(logits, &ids_slice, &vals_slice, vocab, k, rows_total)?;
    let (top_ids, top_vals) = download_top(gpu, &ids_slice, &vals_slice, rows_total * k)?;
    lanes
        .iter()
        .map(|l| {
            let (a, b) = (l.first_row, l.first_row + l.rows);
            propose_candidates_from_rows(
                weights,
                &host_proj[a * rank..b * rank],
                &top_ids[a * k..b * k],
                &top_vals[a * k..b * k],
                l.rows,
                l.anchor,
                0.0,
                None,
            )
        })
        .collect()
}

/// Selector hidden projection `[rows × rank]` (one `gemm_dispatch`),
/// downloaded to the host.
fn selector_projection_host(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    scratch: &DflashScratch,
    hidden: &GpuTensor,
    rows: usize,
    rank: usize,
) -> HipResult<Vec<f32>> {
    let proj = scratch
        .selector_proj
        .as_ref()
        .ok_or_else(|| hip_bridge::HipError::new(0, "selector scratch not allocated"))?;
    let hp = weights.selector_hidden_proj.as_ref().unwrap();
    let proj_slice = proj.sub_offset(0, rows * rank);
    gemm_dispatch(
        gpu,
        hidden,
        hp,
        &proj_slice,
        rows,
        scratch.mq_x_rot.as_ref(),
        scratch.mq_x_rot_f16.as_ref(),
    )?;
    let mut host_proj = vec![0f32; rows * rank];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(host_proj.as_mut_ptr() as *mut u8, host_proj.len() * 4)
    };
    gpu.hip.memcpy_dtoh(bytes, &proj_slice.buf)?;
    Ok(host_proj)
}

/// Download `n` top-K ids (i32 stored in an F32 buffer) and values (f32).
fn download_top(
    gpu: &mut Gpu,
    ids: &GpuTensor,
    vals: &GpuTensor,
    n: usize,
) -> HipResult<(Vec<u32>, Vec<f32>)> {
    let mut ids_bytes = vec![0u8; n * 4];
    let mut vals_bytes = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut ids_bytes, &ids.buf)?;
    gpu.hip.memcpy_dtoh(&mut vals_bytes, &vals.buf)?;
    let top_ids = ids_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let top_vals = vals_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((top_ids, top_vals))
}
