// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5-VL vision encoder: SigLIP-2 ViT + spatial merger.
//! GPU path: gemm_f16 (9 VGPRs), layernorm (13), gelu (8), vit_attention, transpose.

use hip_bridge::{
    DeviceBuffer, Event, HipError, HipResult, HipRuntime, PinnedHostBuffer, Stream,
    HIP_EVENT_DISABLE_TIMING,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use hipfire_runtime::model_source::SourceRangeDescriptor;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::mem::ManuallyDrop;
use std::sync::Mutex;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub mlp_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    /// Number of entries in the learned `pos_embed` table. The table is laid
    /// out as a square grid of side `sqrt(num_position_embeddings)` and
    /// bilinearly interpolated to each image's `(grid_h, grid_w)` at forward
    /// time — see `fast_pos_embed_interpolate`.
    pub num_position_embeddings: usize,
    /// Base θ for the 2D vision rotary frequencies (HF default 10000.0).
    /// Read from `vision_config.rope_theta` if present, else defaulted —
    /// future variants that override `Qwen3_5VisionRotaryEmbedding(theta=…)`
    /// will pick up the correct value automatically.
    pub rope_theta: f32,
    pub norm_eps: f32,
}

pub fn vision_config_from_hfq(hfq: &HfqFile) -> Option<VisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let vc = config.get("vision_config")?;

    let hidden_size = vc.get("hidden_size")?.as_u64()? as usize;
    let num_heads = vc.get("num_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let num_layers = vc.get("depth").and_then(|v| v.as_u64()).unwrap_or(27) as usize;
    let mlp_dim = vc
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4304) as usize;
    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let temporal_patch_size = vc
        .get("temporal_patch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let out_hidden_size = vc
        .get("out_hidden_size")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("hidden_size"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(4096) as usize;
    let spatial_merge_size = vc
        .get("spatial_merge_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let num_position_embeddings = vc
        .get("num_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2304) as usize;
    let rope_theta = vc
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    Some(VisionConfig {
        hidden_size,
        num_heads,
        head_dim: hidden_size / num_heads,
        num_layers,
        mlp_dim,
        patch_size,
        temporal_patch_size,
        out_hidden_size,
        spatial_merge_size,
        num_position_embeddings,
        rope_theta,
        norm_eps: 1e-6,
    })
}

// ─── GPU-side weights ────────────────────────────────────────────────────────

/// Where the tower's large F16 matrices live between image encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TowerResidency {
    /// Every matrix is uploaded to VRAM at load.
    Resident,
    /// Large matrices stay in GPU-addressable host RAM and are read directly.
    /// Small tensors and the patch projection retain their resident placement.
    HostMapped,
    /// The matrices stay on host/disk. Each block streams through two fixed
    /// VRAM staging slots while an image encodes ([`TowerStream`]), so no
    /// tower weights hold VRAM between encodes.
    Streamed,
}

impl TowerResidency {
    /// Resolve the existing vision mode. The developer host-mapped opt-in
    /// only applies to auto/on; explicit resident and off retain their policy.
    pub fn from_vision_mode(mode: &str) -> Self {
        Self::with_host_mapped(
            mode,
            hipfire_config::developer_bool("HIPFIRE_VISION_HOST_MAPPED", false),
        )
    }

    fn with_host_mapped(mode: &str, host_mapped: bool) -> Self {
        match mode {
            "resident" => Self::Resident,
            "auto" | "on" if host_mapped => Self::HostMapped,
            _ => Self::Streamed,
        }
    }
}

/// The small per-block tensors, VRAM-resident under every [`TowerResidency`].
/// The block's four large matrices are [`VisionLayerMats`].
pub struct VisionLayerWeights {
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    pub qkv_b: GpuTensor,
    pub proj_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_b: GpuTensor,
    /// Row pitch (elements) of the block's `fc2_w`: `mlp_dim` rounded up to 64
    /// on the LDS route when `mlp_dim % 64 != 0`, else `mlp_dim`. Fixed at load
    /// by the same predicate (`vit_lds_route`) `linear_f16_ld` re-evaluates.
    pub fc2_ld: usize,
}

/// One block's large F16 matrices, row-major `[out, in]`.
pub struct VisionLayerMats {
    pub qkv_w: GpuTensor,
    pub proj_w: GpuTensor,
    pub fc1_w: GpuTensor,
    /// MLP down-projection weight, F16 `[hidden, fc2_ld]` row-major. When the
    /// LDS GEMM route is active and `mlp_dim % 64 != 0` (4304 on the 27B
    /// tower) each row carries `fc2_ld - mlp_dim` trailing zero columns;
    /// otherwise `fc2_ld == mlp_dim` and the matrix is packed.
    pub fc2_w: GpuTensor,
}

/// The tower's large F16 matrices (every block's [`VisionLayerMats`] and the
/// two merger linears), held per [`TowerResidency`].
pub enum TowerMats {
    Resident {
        layers: Vec<VisionLayerMats>,
        merger_fc1_w: GpuTensor,
        merger_fc2_w: GpuTensor,
    },
    Streamed(TowerStream),
}

impl TowerMats {
    fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Self::Resident {
                layers,
                merger_fc1_w,
                merger_fc2_w,
            } => {
                for m in layers {
                    free_layer_mats(gpu, m);
                }
                free_all(gpu, [merger_fc1_w, merger_fc2_w]);
            }
            Self::Streamed(stream) => stream.free_gpu(gpu),
        }
    }
}

pub struct VisionWeights {
    pub patch_embed_w: GpuTensor,
    pub patch_embed_b: GpuTensor,
    /// Learned position-embedding table `(num_position_embeddings, hidden)`,
    /// resident on CPU because every image bilinearly interpolates a different
    /// `(grid_h, grid_w)` slice out of it (`fast_pos_embed_interpolate`). Cost
    /// is small enough (~10 MB at 2304×1152 F32) that keeping it on host is
    /// strictly simpler than re-uploading shards per image.
    pub pos_embed: Vec<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub merger_norm_w: GpuTensor,
    pub merger_norm_b: GpuTensor,
    pub merger_fc1_b: GpuTensor,
    pub merger_fc2_b: GpuTensor,
    /// The large matrices; see [`TowerMats`].
    pub mats: TowerMats,
}

/// Which merger linear [`VisionWeights::with_merger_w`] hands out; the
/// discriminant is the offset from the first merger streaming unit.
#[derive(Clone, Copy)]
enum MergerFc {
    Fc1 = 0,
    Fc2 = 1,
}

impl VisionWeights {
    pub fn residency(&self) -> TowerResidency {
        match &self.mats {
            TowerMats::Resident { layers, .. } => {
                if layers.first().is_some_and(|l| l.qkv_w.buf.is_host_mapped()) {
                    TowerResidency::HostMapped
                } else {
                    TowerResidency::Resident
                }
            }
            TowerMats::Streamed(_) => TowerResidency::Streamed,
        }
    }

    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        free_all(gpu, [self.patch_embed_w, self.patch_embed_b]);
        for l in self.layers {
            free_layer_small(gpu, l);
        }
        free_all(
            gpu,
            [
                self.merger_norm_w,
                self.merger_norm_b,
                self.merger_fc1_b,
                self.merger_fc2_b,
            ],
        );
        self.mats.free_gpu(gpu);
    }

    /// VRAM bytes of the always-resident tensors (everything but the large
    /// matrices; `pos_embed` lives on the host).
    fn small_tensor_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                [
                    &l.norm1_w, &l.norm1_b, &l.qkv_b, &l.proj_b, &l.norm2_w, &l.norm2_b, &l.fc1_b,
                    &l.fc2_b,
                ]
                .iter()
                .map(|t| t.buf.size())
                .sum::<usize>()
            })
            .sum();
        let rest: usize = [
            &self.patch_embed_w,
            &self.patch_embed_b,
            &self.merger_norm_w,
            &self.merger_norm_b,
            &self.merger_fc1_b,
            &self.merger_fc2_b,
        ]
        .iter()
        .map(|t| t.buf.size())
        .sum();
        layers + rest
    }

    /// Streamed residency: queue the copy of `unit` ahead of its use (no-op
    /// when resident).
    fn prefetch_unit(&self, gpu: &Gpu, unit: usize) {
        if let TowerMats::Streamed(stream) = &self.mats {
            stream.prefetch(gpu, unit);
        }
    }

    /// Run `f` with a merger matrix: the resident tensor, or the staged view
    /// of its streaming unit (released when `f` returns, on success or not).
    fn with_merger_w<R>(
        &self,
        gpu: &mut Gpu,
        fc: MergerFc,
        f: impl FnOnce(&mut Gpu, &GpuTensor) -> HipResult<R>,
    ) -> HipResult<R> {
        match &self.mats {
            TowerMats::Resident {
                merger_fc1_w,
                merger_fc2_w,
                ..
            } => f(
                gpu,
                match fc {
                    MergerFc::Fc1 => merger_fc1_w,
                    MergerFc::Fc2 => merger_fc2_w,
                },
            ),
            TowerMats::Streamed(stream) => stream.with_unit(
                gpu,
                self.layers.len() + fc as usize,
                |gpu, staged| match staged {
                    [w] => f(gpu, w),
                    _ => Err(HipError::new(
                        1,
                        "vision tower stream: merger unit is not one matrix",
                    )),
                },
            ),
        }
    }
}

fn free_all(gpu: &mut Gpu, tensors: impl IntoIterator<Item = GpuTensor>) {
    for t in tensors {
        let _ = gpu.free_tensor(t);
    }
}

fn free_layer_small(gpu: &mut Gpu, l: VisionLayerWeights) {
    free_all(
        gpu,
        [
            l.norm1_w, l.norm1_b, l.qkv_b, l.proj_b, l.norm2_w, l.norm2_b, l.fc1_b, l.fc2_b,
        ],
    );
}

fn free_layer_mats(gpu: &mut Gpu, m: VisionLayerMats) {
    free_all(gpu, [m.qkv_w, m.proj_w, m.fc1_w, m.fc2_w]);
}

// ─── Weight loading ──────────────────────────────────────────────────────────

fn load_f32_cpu(hfq: &HfqFile, name: &str, n: usize) -> Vec<f32> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let mut vals: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        6 | 7 => dequant_hfq4(data, n, info.group_size as usize),
        _ => panic!(
            "expected F16/F32/HFQ4 for {name}, got qt={}",
            info.quant_type
        ),
    };
    vals.truncate(n);
    vals
}

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let vals = load_f32_cpu(hfq, name, n);
    gpu.upload_f32(&vals, &[n])
}

/// Finite F16 range: ±65504 is the largest finite half-precision value.
const F16_FINITE_MAX: f32 = 65504.0;

/// Validate a single F32 oracle element before narrowing to F16.
///
/// Rejects non-finite values (NaN, ±Inf) and finite magnitudes outside the
/// finite IEEE 754 binary16 range. Returns a descriptive `HipError` (invalid
/// value) naming the tensor, element index, and offending value so the loader
/// caller can surface the exact source tensor.
fn checked_f32_to_f16(value: f32, tensor: &str, index: usize) -> HipResult<u16> {
    if !value.is_finite() {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: element {index} is non-finite ({value}) — cannot narrow F32 oracle to F16 without producing Inf/NaN"
            ),
        ));
    }
    if value.abs() > F16_FINITE_MAX {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: element {index} value {value} exceeds finite F16 range (±{F16_FINITE_MAX})"
            ),
        ));
    }
    Ok(f32_to_f16(value))
}

/// Pure bulk helper: validate every F32 element then narrow to little-endian F16 bytes.
/// Operates on the raw little-endian F32 byte payload (`data`) and the expected
/// element count `n` so the production loader and focused tests share the same
/// truncation/length validation. Rejects payloads with fewer than `n` complete
/// F32 elements or a non-multiple-of-4 length before narrowing each element
/// through [`checked_f32_to_f16`].
fn f32_bytes_to_f16_bytes_checked(data: &[u8], n: usize, tensor: &str) -> HipResult<Vec<u8>> {
    let need = n.checked_mul(4).ok_or_else(|| {
        HipError::new(
            1,
            &format!("{tensor}: tensor length overflow for {n} F32 elements"),
        )
    })?;
    if data.len() < need {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: F32 payload too short: need {need} bytes for {n} elements, got {}",
                data.len()
            ),
        ));
    }
    if !data.len().is_multiple_of(4) {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: F32 payload length {} is not a multiple of 4 (trailing malformed bytes)",
                data.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(n * 2);
    for idx in 0..n {
        let off = idx * 4;
        let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let bits = checked_f32_to_f16(v, tensor, idx)?;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

/// Raw F16 bytes of a vision weight (F16 / F32-oracle / HFQ4 sources) and its
/// element count. Borrowed for native F16, owned when a conversion ran.
fn load_f16_bytes<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> HipResult<(std::borrow::Cow<'a, [u8]>, usize)> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        // F16 — use directly. Shape records element count, not byte count.
        1 => Ok((std::borrow::Cow::Borrowed(data), n)),
        2 => {
            // F32 oracle — the vision kernels consume F16 matrices, so narrow
            // the lossless decoder-oracle container at load time just as the
            // ordinary --include-vision ingest does before writing F16.
            // Every element is validated before narrowing so non-finite or
            // out-of-range values surface as a load error instead of silent Inf/NaN.
            // The shared `f32_bytes_to_f16_bytes_checked` helper owns both
            // truncation/length checks and the per-element F16-range validation
            // so the production path is mechanically tied to focused tests.
            let f16_bytes = f32_bytes_to_f16_bytes_checked(data, n, name)?;
            Ok((std::borrow::Cow::Owned(f16_bytes), n))
        }
        6 | 7 => {
            // HFQ4 — dequantize to F32, then convert to F16 for gemm_f16.
            // Shape records element count, not byte count.
            let f32_data = dequant_hfq4(data, n, info.group_size as usize);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            Ok((std::borrow::Cow::Owned(f16_bytes), n))
        }
        other => {
            panic!(
                "{name}: unsupported vision quant_type={other} \
                 (expected F16=1, F32=2, HFQ4=6/7)"
            )
        }
    }
}

fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    load_f16_gpu_at(hfq, gpu, name, false)
}

fn load_f16_gpu_at(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    host_mapped: bool,
) -> HipResult<GpuTensor> {
    let (bytes, n) = load_f16_bytes(hfq, name)?;
    if host_mapped {
        gpu.upload_raw_host_mapped(&bytes, &[n])
    } else {
        gpu.upload_raw(&bytes, &[n])
    }
}

/// Upload a `[rows, cols]` F16 weight at row pitch `pitch >= cols`, the
/// `pitch - cols` trailing columns of every row being exact zeros. Shape is
/// recorded as the element count `rows * pitch`, like every other vision
/// weight.
fn load_f16_gpu_padded(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    rows: usize,
    cols: usize,
    pitch: usize,
    host_mapped: bool,
) -> HipResult<GpuTensor> {
    let (bytes, n) = load_f16_bytes(hfq, name)?;
    if n != rows * cols || bytes.len() < n * 2 || pitch < cols {
        return Err(HipError::new(
            1,
            &format!("{name}: cannot pad {n} elements to [{rows}, {cols}] at pitch {pitch}"),
        ));
    }
    let mut padded = vec![0u8; rows * pitch * 2];
    for r in 0..rows {
        padded[r * pitch * 2..(r * pitch + cols) * 2]
            .copy_from_slice(&bytes[r * cols * 2..(r + 1) * cols * 2]);
    }
    // Typed F16: the pitched LDS entries assert their operand dtype.
    let mut t = if host_mapped {
        gpu.upload_raw_host_mapped(&padded, &[rows * pitch])?
    } else {
        gpu.upload_raw(&padded, &[rows * pitch])?
    };
    t.dtype = DType::F16;
    Ok(t)
}

/// Dequantize HFQ4 blocks to F32, using actual group_size (128 or 256).
/// G256 block: [scale:f32, zero:f32, 128 bytes nibbles] = 136 bytes per 256 values.
/// G128 block: [scale:f32, zero:f32, 64 bytes nibbles] = 72 bytes per 128 values.
fn dequant_hfq4(data: &[u8], n: usize, group_size: usize) -> Vec<f32> {
    let nibble_bytes = group_size / 2;
    let block_size = 8 + nibble_bytes; // 4+4 scale/zero + nibbles
    let mut out = Vec::with_capacity(n);
    let n_groups = n.div_ceil(group_size);
    for g in 0..n_groups {
        let off = g * block_size;
        if off + 8 > data.len() {
            break;
        }
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let zero = f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let nibbles = &data[off + 8..(off + block_size).min(data.len())];
        let base = g * group_size;
        for i in 0..group_size.min(n.saturating_sub(base)) {
            let byte_idx = i / 2;
            if byte_idx >= nibbles.len() {
                break;
            }
            let nibble = if i % 2 == 0 {
                nibbles[byte_idx] & 0xF
            } else {
                nibbles[byte_idx] >> 4
            };
            out.push(scale * nibble as f32 + zero);
        }
    }
    out.truncate(n);
    out
}

/// One deferred tensor upload of a [`load_group`].
type Upload<'a> = &'a dyn Fn(&mut Gpu) -> HipResult<GpuTensor>;

/// Upload a fixed group of tensors in order. A failure frees the members
/// already uploaded (dropping a `GpuTensor` leaks its VRAM) before it
/// propagates.
fn load_group<const N: usize>(gpu: &mut Gpu, loads: [Upload<'_>; N]) -> HipResult<[GpuTensor; N]> {
    let mut got = Vec::with_capacity(N);
    for load in loads {
        match load(gpu) {
            Ok(t) => got.push(t),
            Err(e) => {
                free_all(gpu, got);
                return Err(e);
            }
        }
    }
    let Ok(group) = <[GpuTensor; N]>::try_from(got) else {
        unreachable!("one uploaded tensor per load");
    };
    Ok(group)
}

fn load_layer_small(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
    i: usize,
    fc2_ld: usize,
) -> HipResult<VisionLayerWeights> {
    let h = config.hidden_size;
    let p = format!("model.visual.blocks.{i}");
    let [norm1_w, norm1_b, qkv_b, proj_b, norm2_w, norm2_b, fc1_b, fc2_b] = load_group(
        gpu,
        [
            &|g| load_f32_gpu(hfq, g, &format!("{p}.norm1.weight"), h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.norm1.bias"), h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.attn.qkv.bias"), 3 * h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.attn.proj.bias"), h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.norm2.weight"), h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.norm2.bias"), h),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.mlp.linear_fc1.bias"), config.mlp_dim),
            &|g| load_f32_gpu(hfq, g, &format!("{p}.mlp.linear_fc2.bias"), h),
        ],
    )?;
    Ok(VisionLayerWeights {
        norm1_w,
        norm1_b,
        qkv_b,
        proj_b,
        norm2_w,
        norm2_b,
        fc1_b,
        fc2_b,
        fc2_ld,
    })
}

/// The tensors resident under every [`TowerResidency`]: all but the large
/// matrices.
struct ResidentTensors {
    patch_embed_w: GpuTensor,
    patch_embed_b: GpuTensor,
    layers: Vec<VisionLayerWeights>,
    merger_norm_w: GpuTensor,
    merger_norm_b: GpuTensor,
    merger_fc1_b: GpuTensor,
    merger_fc2_b: GpuTensor,
}

impl ResidentTensors {
    fn free(self, gpu: &mut Gpu) {
        for l in self.layers {
            free_layer_small(gpu, l);
        }
        free_all(
            gpu,
            [
                self.patch_embed_w,
                self.patch_embed_b,
                self.merger_norm_w,
                self.merger_norm_b,
                self.merger_fc1_b,
                self.merger_fc2_b,
            ],
        );
    }
}

fn load_resident_tensors(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
    fc2_ld: usize,
) -> HipResult<ResidentTensors> {
    let h = config.hidden_size;
    let merge_dim = h * config.spatial_merge_size * config.spatial_merge_size;
    let [patch_embed_w, patch_embed_b] = load_group(
        gpu,
        [
            &|g| load_f16_gpu(hfq, g, "model.visual.patch_embed.proj.weight"),
            &|g| load_f32_gpu(hfq, g, "model.visual.patch_embed.proj.bias", h),
        ],
    )?;
    let mut layers = Vec::with_capacity(config.num_layers);
    let mut failure = None;
    for i in 0..config.num_layers {
        match load_layer_small(hfq, config, gpu, i, fc2_ld) {
            Ok(l) => layers.push(l),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    let tail = match failure {
        Some(e) => Err(e),
        None => load_group(
            gpu,
            [
                &|g| load_f32_gpu(hfq, g, "model.visual.merger.norm.weight", h),
                &|g| load_f32_gpu(hfq, g, "model.visual.merger.norm.bias", h),
                &|g| load_f32_gpu(hfq, g, "model.visual.merger.linear_fc1.bias", merge_dim),
                &|g| {
                    load_f32_gpu(
                        hfq,
                        g,
                        "model.visual.merger.linear_fc2.bias",
                        config.out_hidden_size,
                    )
                },
            ],
        ),
    };
    match tail {
        Ok([merger_norm_w, merger_norm_b, merger_fc1_b, merger_fc2_b]) => Ok(ResidentTensors {
            patch_embed_w,
            patch_embed_b,
            layers,
            merger_norm_w,
            merger_norm_b,
            merger_fc1_b,
            merger_fc2_b,
        }),
        Err(e) => {
            for l in layers {
                free_layer_small(gpu, l);
            }
            free_all(gpu, [patch_embed_w, patch_embed_b]);
            Err(e)
        }
    }
}

/// Persistent matrices use the same conversion, pitch and transactional
/// ownership whether their allocation lives in VRAM or mapped host RAM.
fn load_persistent_mats(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
    fc2_ld: usize,
    host_mapped: bool,
) -> HipResult<TowerMats> {
    let h = config.hidden_size;
    let mut layers: Vec<VisionLayerMats> = Vec::with_capacity(config.num_layers);
    let mut failure = None;
    for i in 0..config.num_layers {
        if i % 9 == 0 {
            eprintln!("  loading vision block {i}/{}...", config.num_layers);
        }
        let p = format!("model.visual.blocks.{i}");
        let fc2_name = format!("{p}.mlp.linear_fc2.weight");
        let group = load_group(
            gpu,
            [
                &|g| load_f16_gpu_at(hfq, g, &format!("{p}.attn.qkv.weight"), host_mapped),
                &|g| load_f16_gpu_at(hfq, g, &format!("{p}.attn.proj.weight"), host_mapped),
                &|g| load_f16_gpu_at(hfq, g, &format!("{p}.mlp.linear_fc1.weight"), host_mapped),
                &|g| {
                    if fc2_ld != config.mlp_dim {
                        load_f16_gpu_padded(
                            hfq,
                            g,
                            &fc2_name,
                            h,
                            config.mlp_dim,
                            fc2_ld,
                            host_mapped,
                        )
                    } else {
                        load_f16_gpu_at(hfq, g, &fc2_name, host_mapped)
                    }
                },
            ],
        );
        match group {
            Ok([qkv_w, proj_w, fc1_w, fc2_w]) => layers.push(VisionLayerMats {
                qkv_w,
                proj_w,
                fc1_w,
                fc2_w,
            }),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    let merger = match failure {
        Some(e) => Err(e),
        None => {
            eprintln!("  loading vision merger...");
            load_group(
                gpu,
                [
                    &|g| {
                        load_f16_gpu_at(
                            hfq,
                            g,
                            "model.visual.merger.linear_fc1.weight",
                            host_mapped,
                        )
                    },
                    &|g| {
                        load_f16_gpu_at(
                            hfq,
                            g,
                            "model.visual.merger.linear_fc2.weight",
                            host_mapped,
                        )
                    },
                ],
            )
        }
    };
    match merger {
        Ok([merger_fc1_w, merger_fc2_w]) => Ok(TowerMats::Resident {
            layers,
            merger_fc1_w,
            merger_fc2_w,
        }),
        Err(e) => {
            for m in layers {
                free_layer_mats(gpu, m);
            }
            Err(e)
        }
    }
}

/// Load the vision tower. `residency` decides where the large F16 matrices
/// live ([`TowerResidency`]); the small tensors and the host-side position
/// table are the same either way, and on any error everything already
/// uploaded is released.
pub fn load_vision_weights(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
    residency: TowerResidency,
) -> HipResult<VisionWeights> {
    // Load-time geometry validation: the learned position table is a square
    // grid indexed by `sqrt(num_position_embeddings)`. A non-square value
    // makes `fast_pos_embed_interpolate` index the wrong rows — the tower then
    // produces garbage tokens. This is a model-config malformation, so it is
    // refused HERE (model load) rather than per image request, where it used
    // to be an engine-thread assert that poisoned every slot.
    let side = (config.num_position_embeddings as f64).sqrt().round() as usize;
    if side == 0 || side * side != config.num_position_embeddings {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen35-vl: num_position_embeddings ({}) must be a perfect square",
                config.num_position_embeddings
            ),
        ));
    }
    // Same class as the num_position_embeddings check above: a `.vl` sidecar
    // whose `depth`/`hidden`/`heads` is 0 loads "successfully" (the layer loop
    // is empty) and then panics the engine thread on the FIRST image request
    // (`step_layer`'s `next_layer < num_layers` assert) — the serve-hang class.
    // Refuse at load, where the operator sees it.
    if config.num_layers == 0 || config.hidden_size == 0 || config.num_heads == 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "qwen35-vl: vision_config must have positive depth/hidden/heads \
                 (got depth={} hidden={} heads={})",
                config.num_layers, config.hidden_size, config.num_heads
            ),
        ));
    }
    let h = config.hidden_size;

    // Arch advisory. The vision tower kernels (gemm_f16, layernorm_batched,
    // vit_attention_f32, apply_rope_2d_vision_f32, gelu_tanh_f32, etc.) are
    // arch-neutral HIP and should COMPILE on any ROCm-supported GPU, but the
    // 12/12 quality bench against llama.cpp was only run on gfx1100 (RX 7900 XT).
    // Other archs are unvalidated — we don't refuse, but we warn so users can
    // distinguish "tested" from "should work in theory."
    // Tracked validation: gfx1100, gfx1101, gfx1102 (all RDNA3 wave32).
    match gpu.arch.as_str() {
        "gfx1100" | "gfx1101" | "gfx1102" => {}
        other => {
            eprintln!(
                "  ⚠ vision tower not yet validated on {other}; \
                       results may differ from gfx1100 reference. See \
                       benchmarks/vision/comparison-2026-05-23.md for the \
                       gfx1100 baseline."
            );
        }
    }

    // Detect vision weight format (F16 direct vs HFQ4 auto-dequant) and log once.
    // HFQ4 vision weights (qt=6 G256, qt=7 G128) are dequantized to F16 at load
    // time for the gemm_f16 path — there is no GPU HFQ4 kernel for vision yet.
    // See CHANGELOG.md "v0.1.7-alpha.4 / Vision" for details.
    if let Some((info, _)) = hfq.tensor_data("model.visual.patch_embed.proj.weight") {
        let fmt = match info.quant_type {
            1 => "F16 (direct)",
            2 => "F32 oracle (narrowing to F16 on load)",
            6 => "HFQ4-G256 (dequanting to F16 on load)",
            7 => "HFQ4-G128 (dequanting to F16 on load)",
            other => &format!("qt={other}"),
        };
        eprintln!("  vision weight format: {fmt}");
    }
    match residency {
        TowerResidency::Resident => eprintln!("  loading vision weights (GPU)..."),
        TowerResidency::HostMapped => {
            eprintln!("  loading vision weights (host-mapped direct reads)...")
        }
        TowerResidency::Streamed => {}
    }
    // Host-only; read before any upload so a missing table strands no VRAM.
    let pos_embed = load_f32_cpu(
        hfq,
        "model.visual.pos_embed.weight",
        config.num_position_embeddings * h,
    );

    // fc2's contraction dim (mlp_dim = 4304 on the 27B tower) is not a
    // multiple of 64, which the LDS GEMM requires. On the LDS route the
    // weight is stored zero-padded to K_pad = round_up(mlp_dim, 64) so the
    // whole tower stays on that route; the pad is decided here and
    // re-derived by `linear_f16_ld` from the same predicate.
    let fc2_ld = if vit_lds_route(gpu, h) {
        config.mlp_dim.next_multiple_of(VIT_LDS_K_ALIGN)
    } else {
        config.mlp_dim
    };
    let small = load_resident_tensors(hfq, config, gpu, fc2_ld)?;
    let mats = match residency {
        TowerResidency::Resident => load_persistent_mats(hfq, config, gpu, fc2_ld, false),
        TowerResidency::HostMapped => load_persistent_mats(hfq, config, gpu, fc2_ld, true),
        TowerResidency::Streamed => {
            TowerStream::new(hfq, config, gpu, fc2_ld).map(TowerMats::Streamed)
        }
    };
    let mats = match mats {
        Ok(mats) => mats,
        Err(e) => {
            small.free(gpu);
            return Err(e);
        }
    };
    let weights = VisionWeights {
        patch_embed_w: small.patch_embed_w,
        patch_embed_b: small.patch_embed_b,
        pos_embed,
        layers: small.layers,
        merger_norm_w: small.merger_norm_w,
        merger_norm_b: small.merger_norm_b,
        merger_fc1_b: small.merger_fc1_b,
        merger_fc2_b: small.merger_fc2_b,
        mats,
    };
    if let TowerMats::Streamed(stream) = &weights.mats {
        const MIB: f64 = (1 << 20) as f64;
        eprintln!(
            "  vision tower: streamed from host ({} units, 2 x {:.1} MiB VRAM staging, \
             2 x {:.1} MiB pinned host, {:.1} MiB resident small tensors)",
            stream.units.len(),
            stream.slot_bytes as f64 / MIB,
            stream.slot_bytes as f64 / MIB,
            weights.small_tensor_bytes() as f64 / MIB,
        );
    }
    Ok(weights)
}

// ─── Streamed tower matrices ─────────────────────────────────────────────────
//
// `TowerResidency::Streamed` keeps the large F16 matrices on host/disk and
// moves one *unit* at a time through two fixed VRAM staging slots: unit
// `i < num_layers` is block `i`'s qkv/proj/fc1/fc2, unit `num_layers` the
// merger fc1, unit `num_layers + 1` the merger fc2. A private non-blocking
// copy stream does the H2D copies, double-buffered: while the kernels of
// one slot run on the launch stream, the next unit is read into a pinned
// bounce buffer and copied into the other slot. Per slot, `ready` (recorded
// on the copy stream after the copy) orders the kernels after the copy, and
// `free` (recorded on the launch stream after the kernels that read the
// slot) orders the next copy after those kernels.

/// Bytes per F16 element.
const F16_BYTES: usize = 2;

/// Alignment (bytes) of every matrix inside a staging slot.
const STAGE_ALIGN: usize = 4096;

/// Host source of one matrix's PACKED F16 bytes (row pitch == `cols`).
enum HostSrc {
    /// Native F16 tensor: positional reads straight from the container
    /// (identity-checked; works with or without an mmap). The tower owns no
    /// host copy between encodes; the kernel page cache serves repeats.
    Range(SourceRangeDescriptor),
    /// F32-oracle / HFQ4 tensor, converted to F16 once at load.
    Owned(Vec<u8>),
}

impl HostSrc {
    fn open(hfq: &HfqFile, name: &str, elems: usize) -> HipResult<Self> {
        let bad = |what: String| HipError::new(1, &format!("{name}: {what}"));
        let info = hfq
            .find_tensor_info(name)
            .ok_or_else(|| bad("vision tensor not found".into()))?;
        let n: usize = info.shape.iter().map(|&s| s as usize).product();
        if n != elems {
            return Err(bad(format!(
                "{n} elements, the tower geometry needs {elems}"
            )));
        }
        if info.quant_type != 1 {
            // F32 oracle / HFQ4: convert exactly as the resident loader does.
            let (bytes, _) = load_f16_bytes(hfq, name)?;
            if bytes.len() != elems * F16_BYTES {
                return Err(bad(format!(
                    "converted F16 payload is {} bytes, expected {}",
                    bytes.len(),
                    elems * F16_BYTES
                )));
            }
            return Ok(Self::Owned(bytes.into_owned()));
        }
        let range = hfq
            .tensor_range(name)
            .map_err(|e| bad(format!("source range: {e}")))?
            .ok_or_else(|| bad("vision tensor not found".into()))?;
        if range.length != (elems * F16_BYTES) as u64 {
            return Err(bad(format!(
                "F16 payload is {} bytes, expected {}",
                range.length,
                elems * F16_BYTES
            )));
        }
        Ok(Self::Range(range))
    }

    /// Fill `dst` (exactly the packed byte count) with the matrix. A container
    /// range is read by up to [`READ_THREADS`] concurrent positional reads of
    /// [`READ_CHUNK`]-byte pieces: one `pread` stream copies the page cache
    /// into the pinned buffer at roughly half the H2D rate, which would make
    /// the host read, not the copy, the per-block floor.
    fn read_into(&self, name: &str, dst: &mut [u8]) -> HipResult<()> {
        match self {
            Self::Range(range) => read_range_parallel(range, dst)
                .map_err(|e| HipError::new(1, &format!("{name}: read from container: {e}"))),
            Self::Owned(bytes) => {
                dst.copy_from_slice(bytes);
                Ok(())
            }
        }
    }
}

/// Bytes per concurrent positional read of a streamed matrix.
const READ_CHUNK: usize = 2 << 20;
/// Concurrent positional reads per streamed matrix.
const READ_THREADS: usize = 4;

/// `range.read_exact(dst)` split over up to [`READ_THREADS`] scoped threads,
/// each reading every `READ_THREADS`-th [`READ_CHUNK`] piece.
fn read_range_parallel(
    range: &SourceRangeDescriptor,
    dst: &mut [u8],
) -> Result<(), hipfire_runtime::model_source::SourceError> {
    if dst.len() as u64 != range.length || dst.len() <= READ_CHUNK {
        return range.read_exact(dst);
    }
    let mut lanes: Vec<Vec<(u64, &mut [u8])>> = (0..READ_THREADS).map(|_| Vec::new()).collect();
    for (i, piece) in dst.chunks_mut(READ_CHUNK).enumerate() {
        lanes[i % READ_THREADS].push((range.offset + (i * READ_CHUNK) as u64, piece));
    }
    std::thread::scope(|s| {
        let workers: Vec<_> = lanes
            .into_iter()
            .filter(|lane| !lane.is_empty())
            .map(|lane| {
                s.spawn(move || {
                    lane.into_iter()
                        .try_for_each(|(offset, piece)| range.read_exact_at(offset, piece))
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().expect("tower range reader panicked"))
    })
}

/// Pull the streamed matrices' container bytes into the page cache in the
/// background, so the first image after a load reads RAM rather than disk
/// (the resident load reads the same bytes at load). Detached: the reader
/// clones keep the file open until it finishes; nothing waits on it, and a
/// read error only means the first encode goes to disk.
fn warm_page_cache(units: &[UnitPlan]) {
    let ranges: Vec<SourceRangeDescriptor> = units
        .iter()
        .flat_map(|u| &u.parts)
        .filter_map(|p| match &p.src {
            HostSrc::Range(range) => Some(range.clone()),
            HostSrc::Owned(_) => None,
        })
        .collect();
    if ranges.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("vl-tower-warm".into())
        .spawn(move || {
            let mut scratch = vec![0u8; READ_CHUNK];
            for range in &ranges {
                let end = range.offset + range.length;
                let mut offset = range.offset;
                while offset < end {
                    let len = (end - offset).min(READ_CHUNK as u64) as usize;
                    if range.read_exact_at(offset, &mut scratch[..len]).is_err() {
                        return;
                    }
                    offset += len as u64;
                }
            }
        });
}

/// Geometry of one matrix of a streaming unit.
struct PartSpec {
    name: String,
    rows: usize,
    cols: usize,
    /// Row pitch in elements (`>= cols`); the `pitch - cols` pad columns are
    /// zeros.
    pitch: usize,
    /// dtype the resident upload of this matrix carries.
    dtype: DType,
}

impl PartSpec {
    /// Device bytes of the matrix including its pad columns.
    fn region_bytes(&self) -> usize {
        self.rows * self.pitch * F16_BYTES
    }
}

struct PartPlan {
    spec: PartSpec,
    src: HostSrc,
    /// Byte offset inside the slot ([`STAGE_ALIGN`]-aligned).
    offset: usize,
}

/// The matrices one staging slot holds at a time, laid out back to back at
/// [`STAGE_ALIGN`]-aligned offsets.
struct UnitPlan {
    parts: Vec<PartPlan>,
    /// Slot bytes the unit occupies (end of its last matrix).
    bytes: usize,
}

impl UnitPlan {
    fn new(parts: Vec<(PartSpec, HostSrc)>) -> Self {
        let regions: Vec<usize> = parts.iter().map(|(spec, _)| spec.region_bytes()).collect();
        let (offsets, bytes) = stage_offsets(&regions);
        let parts = parts
            .into_iter()
            .zip(offsets)
            .map(|((spec, src), offset)| PartPlan { spec, src, offset })
            .collect();
        Self { parts, bytes }
    }

    fn open(hfq: &HfqFile, specs: Vec<PartSpec>) -> HipResult<Self> {
        let mut parts = Vec::with_capacity(specs.len());
        for spec in specs {
            let src = HostSrc::open(hfq, &spec.name, spec.rows * spec.cols)?;
            parts.push((spec, src));
        }
        Ok(Self::new(parts))
    }
}

/// Offsets of consecutive regions, each [`STAGE_ALIGN`]-aligned, and the end
/// of the last one.
fn stage_offsets(regions: &[usize]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(regions.len());
    let mut end = 0usize;
    for &region in regions {
        let offset = end.next_multiple_of(STAGE_ALIGN);
        offsets.push(offset);
        end = offset + region;
    }
    (offsets, end)
}

/// Streaming units in execution order: one per block (qkv, proj, fc1, fc2),
/// then the merger's fc1 and fc2.
fn unit_specs(config: &VisionConfig, fc2_ld: usize) -> Vec<Vec<PartSpec>> {
    let h = config.hidden_size;
    let mlp = config.mlp_dim;
    let merge_dim = h * config.spatial_merge_size * config.spatial_merge_size;
    let part = |name: String, rows: usize, cols: usize, pitch: usize, dtype: DType| PartSpec {
        name,
        rows,
        cols,
        pitch,
        dtype,
    };
    let mut units = Vec::with_capacity(config.num_layers + 2);
    for i in 0..config.num_layers {
        let p = format!("model.visual.blocks.{i}");
        units.push(vec![
            part(format!("{p}.attn.qkv.weight"), 3 * h, h, h, DType::Raw),
            part(format!("{p}.attn.proj.weight"), h, h, h, DType::Raw),
            part(format!("{p}.mlp.linear_fc1.weight"), mlp, h, h, DType::Raw),
            // A padded fc2 is typed F16 (the pitched LDS entries assert it),
            // exactly as the resident upload types it.
            part(
                format!("{p}.mlp.linear_fc2.weight"),
                h,
                mlp,
                fc2_ld,
                if fc2_ld != mlp {
                    DType::F16
                } else {
                    DType::Raw
                },
            ),
        ]);
    }
    units.push(vec![part(
        "model.visual.merger.linear_fc1.weight".to_string(),
        merge_dim,
        merge_dim,
        merge_dim,
        DType::Raw,
    )]);
    units.push(vec![part(
        "model.visual.merger.linear_fc2.weight".to_string(),
        config.out_hidden_size,
        merge_dim,
        merge_dim,
        DType::Raw,
    )]);
    units
}

/// Expand `rows` packed rows of `cols` F16 elements stored at the start of
/// `region` to row pitch `pitch` in place, zeroing every row's pad. `region`
/// is `rows * pitch` elements. Rows move last-first, so no move overwrites a
/// row that has not moved yet (`pitch >= cols`).
fn expand_pitched_rows_in_place(region: &mut [u8], rows: usize, cols: usize, pitch: usize) {
    debug_assert!(pitch >= cols && region.len() == rows * pitch * F16_BYTES);
    if pitch == cols {
        return;
    }
    let (row_bytes, pitch_bytes) = (cols * F16_BYTES, pitch * F16_BYTES);
    for r in (0..rows).rev() {
        let (from, to) = (r * row_bytes, r * pitch_bytes);
        if from != to {
            region.copy_within(from..from + row_bytes, to);
        }
        region[to + row_bytes..to + pitch_bytes].fill(0);
    }
}

/// Write `plan`'s matrices into `dst` (the slot image): each at its offset,
/// padded rows expanded, the alignment gaps zeroed — the image is a pure
/// function of the unit, whatever `dst` held before.
fn fill_unit(plan: &UnitPlan, dst: &mut [u8]) -> HipResult<()> {
    if dst.len() < plan.bytes {
        return Err(HipError::new(
            1,
            &format!(
                "vision tower stream: {} byte bounce buffer for a {} byte unit",
                dst.len(),
                plan.bytes
            ),
        ));
    }
    let mut end = 0usize;
    for part in &plan.parts {
        let spec = &part.spec;
        dst[end..part.offset].fill(0);
        end = part.offset + spec.region_bytes();
        let region = &mut dst[part.offset..end];
        part.src
            .read_into(&spec.name, &mut region[..spec.rows * spec.cols * F16_BYTES])?;
        expand_pitched_rows_in_place(region, spec.rows, spec.cols, spec.pitch);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct SlotMeta {
    /// Unit the slot holds (its copy queued or landed); `None` = empty.
    unit: Option<usize>,
    /// Acquired at least once since it was loaded.
    consumed: bool,
    /// `SlotTable::tick` at the last acquire.
    last_acquire: u64,
    /// `SlotTable::tick` when the load was queued.
    loaded_at: u64,
}

impl SlotMeta {
    const EMPTY: Self = Self {
        unit: None,
        consumed: false,
        last_acquire: 0,
        loaded_at: 0,
    };
}

/// The slot a new unit goes into: an empty one, else the consumed one
/// acquired longest ago. With every slot holding an unconsumed prefetch,
/// `allow_unconsumed` evicts the older-loaded one (a demand load), otherwise
/// there is no victim (a prefetch must not evict a prefetch).
fn pick_victim(slots: &[SlotMeta; 2], allow_unconsumed: bool) -> Option<usize> {
    if let Some(empty) = slots.iter().position(|s| s.unit.is_none()) {
        return Some(empty);
    }
    let pick = |consumed_only: bool, key: fn(&SlotMeta) -> u64| {
        slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.consumed || !consumed_only)
            .min_by_key(|(_, s)| key(s))
            .map(|(i, _)| i)
    };
    pick(true, |s| s.last_acquire).or_else(|| {
        allow_unconsumed
            .then(|| pick(false, |s| s.loaded_at))
            .flatten()
    })
}

/// What each staging slot holds and the order they were loaded and used in:
/// the pure bookkeeping behind [`TowerStream::acquire`] and
/// [`TowerStream::prefetch`].
#[derive(Clone, Copy)]
struct SlotTable {
    slots: [SlotMeta; 2],
    /// Monotonic counter ordering loads and acquires.
    tick: u64,
}

impl SlotTable {
    const EMPTY: Self = Self {
        slots: [SlotMeta::EMPTY; 2],
        tick: 0,
    };

    fn holding(&self, unit: usize) -> Option<usize> {
        self.slots.iter().position(|s| s.unit == Some(unit))
    }

    /// `(slot, load)` serving an acquire of `unit`: the slot already holding
    /// it, else the victim slot, which `load` says must be staged first.
    fn plan_acquire(&self, unit: usize) -> Option<(usize, bool)> {
        match self.holding(unit) {
            Some(slot) => Some((slot, false)),
            None => pick_victim(&self.slots, true).map(|slot| (slot, true)),
        }
    }

    /// The slot a prefetch of `unit` stages into; `None` when a slot holds it
    /// already or both hold unconsumed prefetches.
    fn plan_prefetch(&self, unit: usize) -> Option<usize> {
        if self.holding(unit).is_some() {
            return None;
        }
        pick_victim(&self.slots, false)
    }

    fn mark_empty(&mut self, slot: usize) {
        self.slots[slot].unit = None;
    }

    fn mark_loaded(&mut self, slot: usize, unit: usize) {
        self.tick += 1;
        self.slots[slot] = SlotMeta {
            unit: Some(unit),
            consumed: false,
            last_acquire: 0,
            loaded_at: self.tick,
        };
    }

    fn mark_acquired(&mut self, slot: usize) {
        self.tick += 1;
        let meta = &mut self.slots[slot];
        meta.consumed = true;
        meta.last_acquire = self.tick;
    }
}

struct Slot {
    /// `hipMalloc` staging buffer of `slot_bytes` (not pooled).
    buf: DeviceBuffer,
    /// `ready[slot]` / `free[slot]` have been recorded at least once.
    ready_recorded: bool,
    free_recorded: bool,
}

struct StreamState {
    slots: [Slot; 2],
    table: SlotTable,
    /// Pinned host staging, one per slot. Never freed implicitly: a copy may
    /// still be reading it, so only [`TowerStream::free_gpu`] releases it,
    /// after draining the copy stream.
    bounce: ManuallyDrop<[PinnedHostBuffer; 2]>,
    copy: Stream,
    /// Recorded on `copy` after a slot's H2D.
    ready: [Event; 2],
    /// Recorded on the launch stream after the kernels reading a slot.
    free: [Event; 2],
}

// SAFETY: `PinnedHostBuffer` (a raw `hipHostMalloc` pointer it exclusively
// owns) is the only member that is not already `Send`; `DeviceBuffer`,
// `Stream` and `Event` are `Send` handles. The state sits behind
// `TowerStream`'s mutex and is driven by the one thread that owns the GPU;
// `TowerStream::free_gpu` drains the copy stream before releasing the pinned
// buffers.
unsafe impl Send for StreamState {}

impl StreamState {
    /// Allocate both staging slots, both pinned bounce buffers, the copy
    /// stream and the four events; on failure everything already created is
    /// released.
    fn create(hip: &HipRuntime, slot_bytes: usize) -> HipResult<Self> {
        let mut bufs: Vec<DeviceBuffer> = Vec::with_capacity(2);
        let mut bounces: Vec<PinnedHostBuffer> = Vec::with_capacity(2);
        let mut events: Vec<Event> = Vec::with_capacity(4);
        let mut copy: Option<Stream> = None;
        let built = (|| -> HipResult<()> {
            for _ in 0..2 {
                bufs.push(hip.malloc(slot_bytes)?);
            }
            for _ in 0..2 {
                bounces.push(hip.host_buffer(slot_bytes)?);
            }
            copy = Some(hip.stream_create_non_blocking()?);
            for _ in 0..4 {
                events.push(hip.event_create_with_flags(HIP_EVENT_DISABLE_TIMING)?);
            }
            Ok(())
        })();
        if let Err(e) = built {
            for event in events {
                let _ = hip.event_destroy(event);
            }
            if let Some(stream) = copy {
                let _ = hip.stream_destroy(stream);
            }
            for buf in bufs {
                let _ = hip.free(buf);
            }
            // `bounces` drop here: nothing was ever copied out of them.
            return Err(e);
        }
        let (Ok([buf0, buf1]), Ok(bounces), Ok([ready0, ready1, free0, free1]), Some(copy)) = (
            <[DeviceBuffer; 2]>::try_from(bufs),
            <[PinnedHostBuffer; 2]>::try_from(bounces),
            <[Event; 4]>::try_from(events),
            copy,
        ) else {
            unreachable!("every resource was created");
        };
        let slot = |buf| Slot {
            buf,
            ready_recorded: false,
            free_recorded: false,
        };
        Ok(Self {
            slots: [slot(buf0), slot(buf1)],
            table: SlotTable::EMPTY,
            bounce: ManuallyDrop::new(bounces),
            copy,
            ready: [ready0, ready1],
            free: [free0, free1],
        })
    }

    /// Stage `plan` (unit `unit`) into `slot`: read it into the slot's pinned
    /// bounce buffer, then queue ONE H2D on the copy stream (after the
    /// kernels that last read the slot) and record `ready`. On error the
    /// slot is left empty.
    fn load_into(
        &mut self,
        hip: &HipRuntime,
        plan: &UnitPlan,
        unit: usize,
        slot: usize,
    ) -> HipResult<()> {
        self.table.mark_empty(slot);
        if self.slots[slot].ready_recorded {
            // The previous H2D out of this bounce buffer has completed.
            hip.event_synchronize(&self.ready[slot])?;
        }
        fill_unit(plan, &mut self.bounce[slot].as_bytes_mut()[..plan.bytes])?;
        let waited = if self.slots[slot].free_recorded {
            hip.stream_wait_event(&self.copy, &self.free[slot])
        } else {
            Ok(())
        };
        let src: &[u8] = &self.bounce[slot].as_bytes_mut()[..plan.bytes];
        let queued = waited
            .and_then(|()| {
                hip.memcpy_htod_async(
                    &self.slots[slot].buf.byte_view(0, plan.bytes),
                    src,
                    &self.copy,
                )
            })
            .and_then(|()| hip.event_record(&self.ready[slot], Some(&self.copy)));
        if let Err(e) = queued {
            // A queued copy may still be reading the bounce buffer.
            let _ = hip.stream_synchronize(&self.copy);
            return Err(e);
        }
        self.slots[slot].ready_recorded = true;
        self.table.mark_loaded(slot, unit);
        Ok(())
    }
}

/// A unit staged for the launch stream. Its kernels' enqueue must be followed
/// by [`TowerStream::release`] (success or not).
struct StagedUnit {
    slot: usize,
    /// Non-owning views of the unit's matrices, in [`unit_specs`] order, with
    /// the dtype/shape the resident upload would carry.
    mats: Vec<GpuTensor>,
}

/// Streamed large matrices of the tower (see the section comment above).
/// Interior mutability: the encode paths hold `&VisionWeights`.
pub struct TowerStream {
    units: Vec<UnitPlan>,
    /// Bytes of each staging slot and pinned bounce buffer: the largest unit
    /// rounded up to [`STAGE_ALIGN`].
    slot_bytes: usize,
    state: Mutex<Box<StreamState>>,
}

impl TowerStream {
    /// Open every matrix's host source and allocate the staging resources;
    /// nothing large is uploaded. The container bytes are pulled into the
    /// page cache in the background ([`warm_page_cache`]).
    fn new(hfq: &HfqFile, config: &VisionConfig, gpu: &mut Gpu, fc2_ld: usize) -> HipResult<Self> {
        let units = unit_specs(config, fc2_ld)
            .into_iter()
            .map(|specs| UnitPlan::open(hfq, specs))
            .collect::<HipResult<Vec<_>>>()?;
        let slot_bytes = units
            .iter()
            .map(|u| u.bytes)
            .max()
            .unwrap_or(0)
            .next_multiple_of(STAGE_ALIGN);
        gpu.bind_thread()?;
        let state = StreamState::create(&gpu.hip, slot_bytes)?;
        warm_page_cache(&units);
        Ok(Self {
            units,
            slot_bytes,
            state: Mutex::new(Box::new(state)),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Box<StreamState>> {
        // Every field is individually valid at each point a panic can occur
        // (slot tags advance only after the work they describe is queued),
        // so a poisoned lock is still usable.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Make `unit` available to the launch stream: use the slot already
    /// holding it, else demand-load it into the victim slot; then order the
    /// launch stream after the slot's `ready` event.
    fn acquire(&self, gpu: &Gpu, unit: usize) -> HipResult<StagedUnit> {
        let plan = self
            .units
            .get(unit)
            .ok_or_else(|| HipError::new(1, &format!("vision tower stream: no unit {unit}")))?;
        gpu.bind_thread()?;
        let mut st = self.lock();
        let (slot, load) = st
            .table
            .plan_acquire(unit)
            .ok_or_else(|| HipError::new(1, "vision tower stream: no staging slot"))?;
        if load {
            st.load_into(&gpu.hip, plan, unit, slot)?;
        }
        gpu.hip
            .stream_wait_event_on(gpu.active_stream.as_ref(), &st.ready[slot])?;
        st.table.mark_acquired(slot);
        let mats = plan
            .parts
            .iter()
            .map(|part| GpuTensor {
                buf: st.slots[slot]
                    .buf
                    .byte_view(part.offset, part.spec.region_bytes()),
                shape: vec![part.spec.rows * part.spec.pitch],
                dtype: part.spec.dtype,
            })
            .collect();
        Ok(StagedUnit { slot, mats })
    }

    /// Record `free[slot]` on the launch stream: the copy stream may
    /// overwrite the slot once every kernel enqueued since its `acquire` has
    /// run. MUST follow every successful `acquire`, after whatever kernels
    /// were enqueued (even when a launch failed).
    fn release(&self, gpu: &Gpu, slot: usize) {
        let mut st = self.lock();
        let recorded = gpu.bind_thread().and_then(|()| {
            gpu.hip
                .event_record(&st.free[slot], gpu.active_stream.as_ref())
        });
        match recorded {
            Ok(()) => st.slots[slot].free_recorded = true,
            Err(e) => {
                // The copy stream can no longer be ordered after this unit's
                // kernels: drain the device so the slot is idle before
                // anything overwrites it.
                eprintln!(
                    "  vision tower stream: releasing slot {slot} failed ({e}); draining the device"
                );
                let _ = gpu.hip.device_synchronize();
                st.slots[slot].free_recorded = false;
            }
        }
    }

    /// Queue the copy of `unit` ahead of its use. No-op past the last unit,
    /// when a slot already holds it, or when both slots hold unconsumed
    /// prefetches. A failed prefetch leaves its slot empty; the demand
    /// [`Self::acquire`] repeats the load and reports the error.
    fn prefetch(&self, gpu: &Gpu, unit: usize) {
        let Some(plan) = self.units.get(unit) else {
            return;
        };
        let mut st = self.lock();
        let Some(victim) = st.table.plan_prefetch(unit) else {
            return;
        };
        if gpu.bind_thread().is_ok() {
            let _ = st.load_into(&gpu.hip, plan, unit, victim);
        }
    }

    /// Run `f` with `unit`'s matrices staged; `release` always follows `f`.
    fn with_unit<R>(
        &self,
        gpu: &mut Gpu,
        unit: usize,
        f: impl FnOnce(&mut Gpu, &[GpuTensor]) -> HipResult<R>,
    ) -> HipResult<R> {
        let staged = self.acquire(gpu, unit)?;
        let result = f(gpu, &staged.mats);
        self.release(gpu, staged.slot);
        result
    }

    /// Drain the copy stream and the device, then release the staging
    /// buffers, events, stream and pinned memory. Best-effort, never panics.
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.bind_thread();
        let state = *self
            .state
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let StreamState {
            slots,
            bounce,
            copy,
            ready,
            free,
            ..
        } = state;
        let drained = gpu
            .hip
            .stream_synchronize(&copy)
            .and_then(|()| gpu.hip.device_synchronize());
        for event in ready.into_iter().chain(free) {
            let _ = gpu.hip.event_destroy(event);
        }
        let _ = gpu.hip.stream_destroy(copy);
        for slot in slots {
            let _ = gpu.hip.free(slot.buf);
        }
        // An undrained copy may still read the pinned buffers: leak them
        // rather than free memory under the DMA engine.
        if drained.is_ok() {
            drop(ManuallyDrop::into_inner(bounce));
        }
    }
}

// ─── CPU-side per-image precomputes (pos_embed interp + 2D rotary) ───────────

/// Bilinearly interpolate the learned `(K×K, hidden)` position-embedding table
/// down to the actual `(grid_h, grid_w)`, then reorder into the 2×2 spatial-
/// merge-grouped layout that `extract_patches` emits. Output: `[n*hidden]`.
///
/// HF reference: `Qwen3_5VisionModel.fast_pos_embed_interpolate`. For each
/// output position `(py, px)` the table is sampled at the four corners
/// `(floor(py'), floor(px'))`, `(floor(py'), ceil(px'))`, `(ceil(py'),
/// floor(px'))`, `(ceil(py'), ceil(px'))` with `py' = py * (K-1)/(grid_h-1)`
/// — i.e. `linspace(0, K-1, grid_h)` — and the four embeddings are blended by
/// the standard bilinear weights `((1-dh)(1-dw), (1-dh)dw, dh(1-dw), dh dw)`.
///
/// The output is permuted from `(grid_h, grid_w, hidden)` row-major into the
/// same 2x2-grouped order as `extract_patches`: for each merged-block (gy, gx)
/// in row-major, the four intra-block patches `(sy, sx) ∈ {0,1}²` are
/// consecutive — matching `view(mh, ms, mw, ms, h).permute(0, 2, 1, 3, 4)`.
pub fn fast_pos_embed_interpolate(
    pos_embed: &[f32],
    hidden: usize,
    grid_h: usize,
    grid_w: usize,
    num_grid_per_side: usize,
    merge_size: usize,
) -> Vec<f32> {
    assert!(grid_h % merge_size == 0 && grid_w % merge_size == 0);
    assert!(pos_embed.len() == num_grid_per_side * num_grid_per_side * hidden);

    // linspace(0, K-1, N) — torch semantics. With N==1 the only output is 0.0.
    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![start];
        }
        let step = (end - start) / (n as f32 - 1.0);
        (0..n).map(|i| start + step * i as f32).collect()
    }

    let kmax = (num_grid_per_side - 1) as f32;
    let h_idxs = linspace(0.0, kmax, grid_h);
    let w_idxs = linspace(0.0, kmax, grid_w);

    // Torch `.int()` truncates toward zero; with these non-negative values it
    // is equivalent to floor. Ceil is floor+1 clamped to K-1.
    let mut h_floor = vec![0usize; grid_h];
    let mut h_ceil = vec![0usize; grid_h];
    let mut dh = vec![0.0f32; grid_h];
    for (i, &v) in h_idxs.iter().enumerate() {
        let f = v as i32 as usize;
        h_floor[i] = f;
        h_ceil[i] = (f + 1).min(num_grid_per_side - 1);
        dh[i] = v - f as f32;
    }
    let mut w_floor = vec![0usize; grid_w];
    let mut w_ceil = vec![0usize; grid_w];
    let mut dw = vec![0.0f32; grid_w];
    for (j, &v) in w_idxs.iter().enumerate() {
        let f = v as i32 as usize;
        w_floor[j] = f;
        w_ceil[j] = (f + 1).min(num_grid_per_side - 1);
        dw[j] = v - f as f32;
    }

    let n = grid_h * grid_w;
    // (grid_h, grid_w, hidden) row-major buffer, accumulator at f32.
    let mut interp = vec![0.0f32; n * hidden];
    for py in 0..grid_h {
        let hf = h_floor[py];
        let hc = h_ceil[py];
        let dhi = dh[py];
        for px in 0..grid_w {
            let wf = w_floor[px];
            let wc = w_ceil[px];
            let dwi = dw[px];

            let w_ff = (1.0 - dhi) * (1.0 - dwi);
            let w_fc = (1.0 - dhi) * dwi;
            let w_cf = dhi * (1.0 - dwi);
            let w_cc = dhi * dwi;

            let out_off = (py * grid_w + px) * hidden;
            let r_ff = (hf * num_grid_per_side + wf) * hidden;
            let r_fc = (hf * num_grid_per_side + wc) * hidden;
            let r_cf = (hc * num_grid_per_side + wf) * hidden;
            let r_cc = (hc * num_grid_per_side + wc) * hidden;
            for d in 0..hidden {
                interp[out_off + d] = w_ff * pos_embed[r_ff + d]
                    + w_fc * pos_embed[r_fc + d]
                    + w_cf * pos_embed[r_cf + d]
                    + w_cc * pos_embed[r_cc + d];
            }
        }
    }

    // Permute to 2x2 spatial-merge-grouped order:
    //   (gy, gx, sy, sx) ↦ patch row-major (py = gy*ms+sy, px = gx*ms+sx).
    let mh = grid_h / merge_size;
    let mw = grid_w / merge_size;
    let mut out = vec![0.0f32; n * hidden];
    let mut out_idx = 0usize;
    for gy in 0..mh {
        for gx in 0..mw {
            for sy in 0..merge_size {
                for sx in 0..merge_size {
                    let py = gy * merge_size + sy;
                    let px = gx * merge_size + sx;
                    let src = (py * grid_w + px) * hidden;
                    let dst = out_idx * hidden;
                    out[dst..dst + hidden].copy_from_slice(&interp[src..src + hidden]);
                    out_idx += 1;
                }
            }
        }
    }
    out
}

/// Compute per-patch cos/sin tables for the vision tower's 2D rotary, in the
/// same 2x2-grouped patch order as `extract_patches`.
///
/// HF reference: `Qwen3_5VisionModel.rot_pos_emb`. The inner rotary dim is
/// `head_dim/2`; frequencies are
///     inv_freq[i] = 1 / theta^(2i / (head_dim/2))    for i in [0, head_dim/4)
/// and the per-patch trig argument is
///     concat( row_idx * inv_freq, col_idx * inv_freq )    → (head_dim/2,)
/// where `row_idx`/`col_idx` are the un-merged (gy*ms+sy, gx*ms+sx) positions.
///
/// We only store the `head_dim/2` half — HF then concatenates `(rope, rope)`
/// along the last dim, so the trig values at `d` and `d + head_dim/2` are
/// identical. The rotary kernel reuses one scalar for both halves.
///
/// `theta` is the rotary base (HF default 10000.0); read from
/// `VisionConfig::rope_theta` at the call site.
///
/// Returns `(cos, sin)` each of length `n * (head_dim/2)`.
pub fn compute_vision_rope_cos_sin(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    merge_size: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim % 4 == 0,
        "head_dim must be divisible by 4 (got {head_dim})"
    );
    let rot_dim = head_dim / 2; // total rotary feature width
    let inner = head_dim / 4; // = rot_dim / 2 = #frequencies

    // inv_freq[i] = 1 / theta^(2i / rot_dim)  for i in [0, inner)
    let inv_freq: Vec<f32> = (0..inner)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rot_dim as f32))
        .collect();

    let n = grid_h * grid_w;
    let mut cos_t = vec![0.0f32; n * rot_dim];
    let mut sin_t = vec![0.0f32; n * rot_dim];

    let mh = grid_h / merge_size;
    let mw = grid_w / merge_size;
    let mut out_idx = 0usize;
    for gy in 0..mh {
        for gx in 0..mw {
            for sy in 0..merge_size {
                for sx in 0..merge_size {
                    let row_idx = (gy * merge_size + sy) as f32;
                    let col_idx = (gx * merge_size + sx) as f32;
                    let row_base = out_idx * rot_dim;
                    let col_base = row_base + inner;
                    for i in 0..inner {
                        let angle_r = row_idx * inv_freq[i];
                        let angle_c = col_idx * inv_freq[i];
                        cos_t[row_base + i] = angle_r.cos();
                        sin_t[row_base + i] = angle_r.sin();
                        cos_t[col_base + i] = angle_c.cos();
                        sin_t[col_base + i] = angle_c.sin();
                    }
                    out_idx += 1;
                }
            }
        }
    }
    (cos_t, sin_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-CPU helper checks. The Python reference values come from
    // `Qwen3_5VisionModel.fast_pos_embed_interpolate` / `rot_pos_emb` —
    // see the doc-comments above for the exact formulas.

    #[test]
    fn pos_embed_interp_identity_when_grid_matches_table() {
        // grid_h == grid_w == num_grid_per_side: every linspace point lands
        // exactly on a table entry, so the bilinear weights collapse to
        // pick a single corner and the interp output is a permutation of
        // the input table — 2x2-grouped.
        let k = 4usize;
        let hidden = 3usize;
        let merge = 2usize;
        // Table: pos_embed[r*k+c, d] = (r*k+c)*10 + d
        let mut table = vec![0.0f32; k * k * hidden];
        for r in 0..k {
            for c in 0..k {
                for d in 0..hidden {
                    table[(r * k + c) * hidden + d] = (r * k + c) as f32 * 10.0 + d as f32;
                }
            }
        }
        let out = fast_pos_embed_interpolate(&table, hidden, k, k, k, merge);
        assert_eq!(out.len(), k * k * hidden);
        // Walk 2x2-grouped: (gy, gx, sy, sx) → table[(gy*2+sy, gx*2+sx)].
        let mut idx = 0usize;
        for gy in 0..(k / merge) {
            for gx in 0..(k / merge) {
                for sy in 0..merge {
                    for sx in 0..merge {
                        let r = gy * merge + sy;
                        let c = gx * merge + sx;
                        for d in 0..hidden {
                            let want = (r * k + c) as f32 * 10.0 + d as f32;
                            let got = out[idx * hidden + d];
                            assert!(
                                (got - want).abs() < 1e-4,
                                "idx={idx} d={d} got={got} want={want}"
                            );
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn pos_embed_interp_bilinear_midpoint() {
        // 2x2 table → request 3x3 grid. The center sample is at the exact
        // midpoint of all four corners with weight 1/4 each.
        let k = 2usize;
        let hidden = 1usize;
        let merge = 1usize;
        let table = vec![10.0f32, 20.0f32, 30.0f32, 40.0f32]; // [(0,0)=10, (0,1)=20, (1,0)=30, (1,1)=40]
        let out = fast_pos_embed_interpolate(&table, hidden, 3, 3, k, merge);
        // 3x3 in row-major (no merge): center at out[4*hidden+0].
        assert!((out[0] - 10.0).abs() < 1e-4); // (0,0) → (0,0)
        assert!((out[2] - 20.0).abs() < 1e-4); // (0,2) → (0,1)
        assert!((out[6] - 30.0).abs() < 1e-4); // (2,0) → (1,0)
        assert!((out[8] - 40.0).abs() < 1e-4); // (2,2) → (1,1)
        assert!((out[4] - 25.0).abs() < 1e-4); // center = mean of all 4
    }

    #[test]
    fn rope_table_concat_row_then_col() {
        // For grid=(2,2), merge=1, head_dim=8 → rot_dim=4, inner=2.
        // The token at (row=1, col=0) should have cos[:inner] = cos(1*inv_freq)
        // and cos[inner:] = cos(0*inv_freq) = 1.
        let head_dim = 8usize;
        let (cos_t, sin_t) = compute_vision_rope_cos_sin(2, 2, head_dim, 1, 10000.0);
        let inner = head_dim / 4;
        let rot = head_dim / 2;
        assert_eq!(cos_t.len(), 4 * rot);
        // Patch (row=1, col=0) is at index gy*2+gx with merge=1 → gy=1, gx=0 → idx 2.
        let base = 2 * rot;
        // Col half should be all 1 / 0 since col_idx=0.
        for i in 0..inner {
            assert!(
                (cos_t[base + inner + i] - 1.0).abs() < 1e-6,
                "cos col half should be 1"
            );
            assert!(
                sin_t[base + inner + i].abs() < 1e-6,
                "sin col half should be 0"
            );
        }
        // Row half (row_idx=1): cos(inv_freq[0]) = cos(1.0) ≈ 0.5403
        let inv0 = 1.0f32; // 10000^0 = 1
        assert!((cos_t[base] - inv0.cos()).abs() < 1e-5);
        assert!((sin_t[base] - inv0.sin()).abs() < 1e-5);
    }

    /// Non-square grid (4×6 ≠ 6×6 table): bilinear interpolation across
    /// rows + h-major outer permutation. Catches an h/w transpose bug in
    /// the 2x2-group permute that the square cases can't see.
    #[test]
    fn pos_embed_interp_identity_non_square() {
        let k = 6usize;
        let hidden = 2usize;
        let merge = 2usize;
        let mut table = vec![0.0f32; k * k * hidden];
        for r in 0..k {
            for c in 0..k {
                for d in 0..hidden {
                    table[(r * k + c) * hidden + d] = (r * 100 + c) as f32 + (d as f32) * 0.5;
                }
            }
        }
        // grid_h=4, grid_w=6 — both ≤ K, both even, distinct so a w/h swap
        // in the permutation would produce a visibly wrong byte at a known
        // out_idx.
        let grid_h = 4usize;
        let grid_w = 6usize;
        let out = fast_pos_embed_interpolate(&table, hidden, grid_h, grid_w, k, merge);
        assert_eq!(out.len(), grid_h * grid_w * hidden);

        // Spot-check: 2x2-grouped patch at (gy=1, gx=2, sy=0, sx=1) is at:
        //   out_idx = gy * (gw * 4) + gx * 4 + sy * 2 + sx
        //   gw = grid_w / 2 = 3 → out_idx = 1*12 + 2*4 + 0*2 + 1 = 21
        // Maps to (py = gy*2+sy, px = gx*2+sx) = (2, 5).
        // linspace(0, K-1, grid_h)[2] for K=6, N=4 → 2 * 5/3 ≈ 3.333
        // linspace(0, K-1, grid_w)[5] for K=6, N=6 → 5.0 exact
        // So we sample a bilinear blend between rows 3 and 4 at column 5.
        let py_lin = 2.0f32 * 5.0 / 3.0;
        let r_floor = py_lin as i32 as usize;
        let r_ceil = (r_floor + 1).min(k - 1);
        let dh = py_lin - r_floor as f32;
        let want =
            (1.0 - dh) * table[(r_floor * k + 5) * hidden] + dh * table[(r_ceil * k + 5) * hidden];
        let got = out[21 * hidden];
        assert!(
            (got - want).abs() < 1e-4,
            "non-square spot check failed: got={got} want={want}"
        );
    }

    /// Rectangular `compute_vision_rope_cos_sin`: a 2×4 grid sanity check
    /// that distinguishes the row-half vs col-half indexing — col_idx=2
    /// at head_dim=8 (rot_dim=4, inner=2) yields a deterministic table
    /// value the row-half cannot match.
    #[test]
    fn rope_table_rectangular_grid() {
        let head_dim = 8usize;
        let (cos_t, sin_t) = compute_vision_rope_cos_sin(2, 4, head_dim, 1, 10000.0);
        let inner = head_dim / 4; // 2
        let rot = head_dim / 2; // 4
        assert_eq!(cos_t.len(), 8 * rot);
        // Patch (row=1, col=2): gy=1, gx=2, sy=0, sx=0 → idx = 1*4 + 2 = 6
        let base = 6 * rot;
        // Row half uses row_idx=1: cos[base..base+inner] = cos(1 * inv_freq[0..inner])
        // inv_freq[0] = 1, inv_freq[1] = 10000^(-2/4) = 0.01
        let inv = [1.0f32, 0.01f32];
        for i in 0..inner {
            assert!((cos_t[base + i] - (1.0 * inv[i]).cos()).abs() < 1e-5);
            assert!((sin_t[base + i] - (1.0 * inv[i]).sin()).abs() < 1e-5);
        }
        // Col half uses col_idx=2:
        for i in 0..inner {
            assert!((cos_t[base + inner + i] - (2.0 * inv[i]).cos()).abs() < 1e-5);
            assert!((sin_t[base + inner + i] - (2.0 * inv[i]).sin()).abs() < 1e-5);
        }
    }

    /// `rope_theta` actually changes the table — defends against a future
    /// accidental hardcode regression in `compute_vision_rope_cos_sin`.
    #[test]
    fn rope_table_honors_theta_param() {
        let (c1, _) = compute_vision_rope_cos_sin(2, 2, 8, 1, 10000.0);
        let (c2, _) = compute_vision_rope_cos_sin(2, 2, 8, 1, 50000.0);
        // Some entry must differ — pick a non-row=0, non-col=0 patch so the
        // angle is non-zero on both halves.
        let any_diff = c1.iter().zip(&c2).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "theta change must shift at least one trig value");
    }

    // ── F32-oracle → F16 validation (pure helper, no GPU) ────────────────────

    #[test]
    fn checked_f32_to_f16_accepts_boundaries_and_zero() {
        // Finite boundaries and zero must succeed, and must produce the same
        // bits as the unchecked f32_to_f16 (sanity: we didn't change accepted inputs).
        let valid: &[f32] = &[0.0, -0.0, 65504.0, -65504.0, 1.0, -1.0, 6.1035e-5];
        for (idx, &v) in valid.iter().enumerate() {
            let bits =
                checked_f32_to_f16(v, "test_tensor", idx).expect("valid F32 should narrow to F16");
            assert_eq!(
                bits,
                f32_to_f16(v),
                "value {v} at {idx} mismatched unchecked conversion"
            );
        }
        // Bulk helper (production path) must also accept the same payload via bytes.
        let data: Vec<u8> = valid.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes =
            f32_bytes_to_f16_bytes_checked(&data, valid.len(), "test_tensor").expect("bulk valid");
        assert_eq!(bytes.len(), valid.len() * 2);
        for (i, &v) in valid.iter().enumerate() {
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            assert_eq!(bits, f32_to_f16(v));
        }
    }

    #[test]
    fn checked_f32_to_f16_rejects_nan() {
        let err = checked_f32_to_f16(f32::NAN, "vision.test_nan", 3).unwrap_err();
        assert_eq!(err.code, 1);
        assert!(
            err.message.contains("vision.test_nan"),
            "error must name tensor: {}",
            err.message
        );
        assert!(
            err.message.contains("3"),
            "error must name element index: {}",
            err.message
        );
        // Production bytes helper propagates the same error with correct index.
        let vals = [1.0, f32::NAN, 2.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let err2 =
            f32_bytes_to_f16_bytes_checked(&data, vals.len(), "vision.test_nan").unwrap_err();
        assert_eq!(err2.code, 1);
        assert!(err2.message.contains("vision.test_nan"));
        assert!(err2.message.contains('1'));
    }

    #[test]
    fn checked_f32_to_f16_rejects_infinities() {
        for &inf in &[f32::INFINITY, f32::NEG_INFINITY] {
            let err = checked_f32_to_f16(inf, "vision.test_inf", 0).unwrap_err();
            assert_eq!(err.code, 1);
            assert!(err.message.contains("vision.test_inf"));
            // Must not silently become Inf bits (0x7C00 / 0xFC00).
            assert!(err.message.contains("non-finite") || err.message.contains("Inf"));
        }
        let vals = [0.0, f32::INFINITY];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let err = f32_bytes_to_f16_bytes_checked(&data, vals.len(), "vision.test_inf").unwrap_err();
        assert_eq!(err.code, 1);
        assert!(err.message.contains("1"));
    }

    #[test]
    fn checked_f32_to_f16_rejects_finite_overflow() {
        // Finite magnitudes just beyond F16 max must be rejected, not saturated to Inf.
        let overflows: &[f32] = &[65505.0, 70000.0, 1e6, -65505.0, -1e5];
        for (idx, &v) in overflows.iter().enumerate() {
            let err = checked_f32_to_f16(v, "vision.test_overflow", idx).unwrap_err();
            assert_eq!(
                err.code, 1,
                "overflow {v} should error with invalid-value code"
            );
            assert!(
                err.message.contains("vision.test_overflow"),
                "tensor name missing: {}",
                err.message
            );
            assert!(err.message.contains(&idx.to_string()));
            // Message should mention the offending value or range.
            assert!(
                err.message.contains("65504") || err.message.contains(&v.to_string()),
                "overflow message should describe range/value: {}",
                err.message
            );
        }
        // Production bytes helper on overflow slice must fail at first overflow element.
        let vals: [f32; 3] = [0.0, 70000.0, 1.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let err =
            f32_bytes_to_f16_bytes_checked(&data, vals.len(), "vision.test_overflow").unwrap_err();
        assert_eq!(err.code, 1);
        assert!(err.message.contains("1"));
    }

    #[test]
    fn f32_bytes_helper_rejects_short_and_malformed_payload() {
        // Payload too short: fewer than n complete F32 elements.
        let vals = [1.0f32, 2.0, 3.0];
        let mut data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Request 3 elements but supply only bytes for 2.
        let short = &data[..8];
        let err = f32_bytes_to_f16_bytes_checked(short, 3, "vision.test_short").unwrap_err();
        assert_eq!(err.code, 1);
        assert!(err.message.contains("vision.test_short"));
        assert!(err.message.contains("too short") || err.message.contains("need"));
        // Non-multiple-of-4 trailing bytes.
        data.push(0xFF);
        let err2 =
            f32_bytes_to_f16_bytes_checked(&data, vals.len(), "vision.test_malformed").unwrap_err();
        assert_eq!(err2.code, 1);
        assert!(err2.message.contains("vision.test_malformed"));
        assert!(err2.message.contains("multiple of 4") || err2.message.contains("trailing"));
    }

    // ── Streamed-tower pure helpers (no GPU) ─────────────────────────────────

    /// The 27B Qwen3.8 tower geometry.
    fn config_27b() -> VisionConfig {
        VisionConfig {
            hidden_size: 1152,
            num_heads: 16,
            head_dim: 72,
            num_layers: 27,
            mlp_dim: 4304,
            patch_size: 16,
            temporal_patch_size: 2,
            out_hidden_size: 5120,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// Deterministic non-zero bytes for a packed `[rows, cols]` F16 matrix.
    fn packed_rows(rows: usize, cols: usize) -> Vec<u8> {
        (0..rows * cols * F16_BYTES)
            .map(|i| (i % 251) as u8 + 1)
            .collect()
    }

    fn part_spec(name: &str, rows: usize, cols: usize, pitch: usize) -> PartSpec {
        PartSpec {
            name: name.to_string(),
            rows,
            cols,
            pitch,
            dtype: DType::Raw,
        }
    }

    #[test]
    fn residency_follows_vision_mode() {
        assert_eq!(
            TowerResidency::from_vision_mode("resident"),
            TowerResidency::Resident
        );
        for mode in ["auto", "on", "off", "", "Resident"] {
            assert_eq!(
                TowerResidency::from_vision_mode(mode),
                TowerResidency::Streamed,
                "{mode:?}"
            );
        }
    }

    #[test]
    fn host_mapped_opt_in_preserves_explicit_vision_policy() {
        use TowerResidency::*;
        for (mode, enabled, expected) in [
            ("auto", false, Streamed),
            ("on", false, Streamed),
            ("auto", true, HostMapped),
            ("on", true, HostMapped),
            ("resident", false, Resident),
            ("resident", true, Resident),
            ("off", true, Streamed),
            ("", true, Streamed),
        ] {
            assert_eq!(TowerResidency::with_host_mapped(mode, enabled), expected);
        }
    }

    #[test]
    fn expand_pitched_rows_matches_the_padded_upload_layout() {
        for (rows, cols, pitch) in [(1, 3, 8), (5, 7, 8), (4, 6, 6), (9, 13, 16), (3, 1, 64)] {
            let packed = packed_rows(rows, cols);
            // `load_f16_gpu_padded`'s construction.
            let mut want = vec![0u8; rows * pitch * F16_BYTES];
            for r in 0..rows {
                want[r * pitch * 2..(r * pitch + cols) * 2]
                    .copy_from_slice(&packed[r * cols * 2..(r + 1) * cols * 2]);
            }
            // The slot region starts out dirty (a previous unit's bytes).
            let mut region = vec![0xAAu8; rows * pitch * F16_BYTES];
            region[..packed.len()].copy_from_slice(&packed);
            expand_pitched_rows_in_place(&mut region, rows, cols, pitch);
            assert_eq!(region, want, "rows={rows} cols={cols} pitch={pitch}");
        }
    }

    #[test]
    fn stage_offsets_align_every_region() {
        let (offsets, end) = stage_offsets(&[100, 5000, 4096]);
        assert_eq!(offsets, [0, 4096, 12288]);
        assert_eq!(end, 12288 + 4096);
        assert_eq!(stage_offsets(&[]), (vec![], 0));
    }

    #[test]
    fn unit_layout_27b() {
        let cfg = config_27b();
        // fc2_ld 4352 = 4304 rounded up to 64 (LDS route).
        let specs = unit_specs(&cfg, 4352);
        assert_eq!(specs.len(), 27 + 2);
        let regions = |u: &Vec<PartSpec>| u.iter().map(PartSpec::region_bytes).collect::<Vec<_>>();
        // Block: qkv 3456x1152, proj 1152x1152, fc1 4304x1152, fc2 1152x4352.
        assert_eq!(
            regions(&specs[0]),
            [7_962_624, 2_654_208, 9_916_416, 10_027_008]
        );
        let (offsets, block_bytes) = stage_offsets(&regions(&specs[0]));
        assert_eq!(offsets, [0, 7_962_624, 10_616_832, 20_533_248]);
        assert_eq!(block_bytes, 30_560_256);
        // Merger fc1 4608x4608 and fc2 5120x4608.
        let merger_fc1 = stage_offsets(&regions(&specs[27])).1;
        let merger_fc2 = stage_offsets(&regions(&specs[28])).1;
        assert_eq!((merger_fc1, merger_fc2), (42_467_328, 47_185_920));
        let slot_bytes = specs
            .iter()
            .map(|u| stage_offsets(&regions(u)).1)
            .max()
            .unwrap()
            .next_multiple_of(STAGE_ALIGN);
        assert_eq!(slot_bytes, 47_185_920); // 45 MiB
                                            // A padded fc2 is typed F16 like its resident upload; every other
                                            // matrix (and an unpadded fc2) is Raw.
        let dtypes = |u: &Vec<PartSpec>| u.iter().map(|p| p.dtype).collect::<Vec<_>>();
        assert_eq!(
            dtypes(&specs[3]),
            [DType::Raw, DType::Raw, DType::Raw, DType::F16]
        );
        assert_eq!(dtypes(&specs[28]), [DType::Raw]);
        assert_eq!(dtypes(&unit_specs(&cfg, 4304)[0])[3], DType::Raw);
    }

    #[test]
    fn fill_unit_lays_out_matrices_and_zeroes_gaps_and_pads() {
        let a = packed_rows(3, 5);
        let b = packed_rows(2, 6);
        let plan = UnitPlan::new(vec![
            (part_spec("a", 3, 5, 5), HostSrc::Owned(a.clone())),
            (part_spec("b", 2, 6, 8), HostSrc::Owned(b.clone())),
        ]);
        assert_eq!(plan.parts[0].offset, 0);
        assert_eq!(plan.parts[1].offset, 4096);
        assert_eq!(plan.bytes, 4096 + 2 * 8 * F16_BYTES);

        let mut dst = vec![0xAAu8; plan.bytes + 16];
        fill_unit(&plan, &mut dst).unwrap();
        assert_eq!(&dst[..a.len()], &a[..]);
        assert!(dst[a.len()..4096].iter().all(|&v| v == 0), "gap not zeroed");
        for r in 0..2 {
            let row = 4096 + r * 16;
            assert_eq!(&dst[row..row + 12], &b[r * 12..(r + 1) * 12]);
            assert!(dst[row + 12..row + 16].iter().all(|&v| v == 0), "pad {r}");
        }
        assert!(dst[plan.bytes..].iter().all(|&v| v == 0xAA), "overran unit");
        assert!(fill_unit(&plan, &mut dst[..plan.bytes - 1]).is_err());
    }

    fn slot_meta(
        unit: Option<usize>,
        consumed: bool,
        last_acquire: u64,
        loaded_at: u64,
    ) -> SlotMeta {
        SlotMeta {
            unit,
            consumed,
            last_acquire,
            loaded_at,
        }
    }

    #[test]
    fn victim_is_empty_then_older_consumed_then_older_loaded() {
        let held = slot_meta(Some(0), true, 5, 1);
        // An empty slot goes first, whichever it is and whoever asks.
        assert_eq!(pick_victim(&[held, SlotMeta::EMPTY], false), Some(1));
        assert_eq!(pick_victim(&[SlotMeta::EMPTY, held], true), Some(0));
        // Both consumed: the one acquired longest ago.
        let consumed = [
            slot_meta(Some(0), true, 7, 1),
            slot_meta(Some(1), true, 3, 2),
        ];
        assert_eq!(pick_victim(&consumed, false), Some(1));
        // A consumed slot is evicted before an (older-loaded) unconsumed prefetch.
        let mixed = [
            slot_meta(Some(2), false, 0, 1),
            slot_meta(Some(1), true, 9, 5),
        ];
        assert_eq!(pick_victim(&mixed, false), Some(1));
        assert_eq!(pick_victim(&mixed, true), Some(1));
        // Both hold unconsumed prefetches: a prefetch declines, a demand load
        // evicts the older-loaded one.
        let unconsumed = [
            slot_meta(Some(4), false, 0, 8),
            slot_meta(Some(5), false, 0, 6),
        ];
        assert_eq!(pick_victim(&unconsumed, false), None);
        assert_eq!(pick_victim(&unconsumed, true), Some(1));
    }

    /// `TowerStream::acquire`'s bookkeeping; `true` when the unit was already
    /// staged (no demand load).
    fn sim_acquire(t: &mut SlotTable, unit: usize) -> bool {
        let (slot, load) = t.plan_acquire(unit).unwrap();
        if load {
            t.mark_empty(slot);
            t.mark_loaded(slot, unit);
        }
        t.mark_acquired(slot);
        !load
    }

    /// `TowerStream::prefetch`'s bookkeeping.
    fn sim_prefetch(t: &mut SlotTable, unit: usize, units: usize) {
        if unit >= units {
            return;
        }
        if let Some(slot) = t.plan_prefetch(unit) {
            t.mark_empty(slot);
            t.mark_loaded(slot, unit);
        }
    }

    /// One tower encode as `VisionTowerJob` drives the stream: prologue
    /// prefetches 0 and 1, block `i` acquires `i` then prefetches `i + 2`
    /// (reaching the merger units), `finish` acquires both merger units.
    /// Returns the number of demand loads.
    fn sim_encode(t: &mut SlotTable, blocks: usize) -> usize {
        let units = blocks + 2;
        let mut demand = 0;
        sim_prefetch(t, 0, units);
        sim_prefetch(t, 1, units);
        for unit in 0..units {
            demand += usize::from(!sim_acquire(t, unit));
            sim_prefetch(t, unit + 2, units);
        }
        demand
    }

    #[test]
    fn double_buffering_never_demand_loads_in_steady_state() {
        let mut t = SlotTable::EMPTY;
        // A cold first encode and every later one on the same table stage
        // each unit one step ahead of its use.
        for _ in 0..3 {
            assert_eq!(sim_encode(&mut t, 27), 0);
        }
        // A one-block tower (the merger units are prefetched straight away).
        let mut one_block = SlotTable::EMPTY;
        assert_eq!(sim_encode(&mut one_block, 1), 0);
    }

    #[test]
    fn prefetch_never_evicts_an_unconsumed_prefetch() {
        let mut t = SlotTable::EMPTY;
        sim_prefetch(&mut t, 0, 29);
        sim_prefetch(&mut t, 1, 29);
        sim_prefetch(&mut t, 2, 29);
        assert_eq!(t.holding(0), Some(0));
        assert_eq!(t.holding(1), Some(1));
        assert_eq!(t.holding(2), None);
        // A demand load of a unit nobody prefetched evicts the older-loaded
        // prefetch, and the table then serves it without another load.
        assert!(!sim_acquire(&mut t, 9));
        assert_eq!(t.holding(0), None);
        assert_eq!(t.holding(9), Some(0));
        assert!(sim_acquire(&mut t, 1));
        // Re-prefetching a held unit is a no-op.
        assert_eq!(t.plan_prefetch(1), None);
    }

    #[test]
    fn activation_peak_27b() {
        let cfg = config_27b();
        // The binding phase is the block's fc1 linear. Per patch: x 4608 +
        // rope cos/sin 288 + tmp2 4608 + fc1 output 17216 + the larger of the
        // LDS route's F16 input copy (2304) and the non-WMMA transposed
        // output (17216) = 43936 bytes.
        assert_eq!(tower_activation_peak_bytes(&cfg, 4096), 43_936 * 4096);
        assert_eq!(tower_activation_peak_bytes(&cfg, 7812), 43_936 * 7812);
        assert_eq!(tower_activation_peak_bytes(&cfg, 0), 0);
    }

    #[test]
    fn max_patches_is_the_pixel_budget_over_one_patch() {
        let cfg = config_27b();
        assert_eq!(
            max_image_patches(&cfg),
            crate::image::vision_max_pixels() / (16 * 16)
        );
    }
}

// ─── GPU vision forward (no CPU roundtrips for compute) ──────────────────────

/// `HIPFIRE_VIT_GEMM=mb8` forces every tower linear onto the register-blocked
/// `gemm_f16_wmma_mb8` (the pre-LDS path); anything else (default `auto`)
/// lets [`vit_lds_route`] pick the LDS-tiled WMMA GEMM. Read once.
fn vit_gemm_mb8() -> bool {
    use std::sync::LazyLock;
    static MB8: LazyLock<bool> = LazyLock::new(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_VIT_GEMM").as_deref(),
            Ok("mb8")
        )
    });
    *MB8
}

/// K granularity of the LDS GEMM entries (`K % 64 == 0`).
const VIT_LDS_K_ALIGN: usize = 64;

/// Whether a tower linear with `out_dim` outputs takes the LDS-tiled WMMA
/// route: gfx1201 (exact) or a gfx11 wave32-WMMA arch, `out_dim >= 256`, and
/// `HIPFIRE_VIT_GEMM` is not `mb8`. Pure function of process-constant state,
/// so the loader (fc2 weight padding) and `linear_f16_ld` always agree.
fn vit_lds_route(gpu: &Gpu, out_dim: usize) -> bool {
    !vit_gemm_mb8() && out_dim >= 256 && (gpu.arch == "gfx1201" || gpu.arch_caps.has_wmma_w32())
}

/// gfx1201 tile policy, copied from the qwen35-embed encoder's
/// `projection_gfx1201_tile` (all three tiles are admitted by both the
/// packed and the pitched split-1 entries).
fn vit_gfx1201_tile(out_dim: usize) -> rdna_compute::gemm::LdsTileSplitK {
    use rdna_compute::gemm::LdsTileSplitK;
    match out_dim {
        0..512 => LdsTileSplitK::new(64, 128, 32, 64, true, 1),
        512..1024 => LdsTileSplitK::new(128, 128, 32, 64, false, 1),
        _ => LdsTileSplitK::new(128, 256, 32, 64, false, 1),
    }
}

/// Vision linear: Y[n, out_dim] = W_f16[out_dim, in_dim] @ X_f32[n, in_dim]^T + bias.
/// Weights are packed (row pitch == `in_dim`); see [`linear_f16_ld`] for the
/// zero-padded fc2 weight.
fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    linear_f16_ld(gpu, w, in_dim, x, bias, out_dim, in_dim, n)
}

/// [`linear_f16`] with an explicit weight row pitch `w_ld` (elements).
///
/// Routing (see [`vit_lds_route`]):
/// * **LDS route** (gfx1201 exact, or gfx11 `has_wmma_w32`; `out_dim >= 256`;
///   `HIPFIRE_VIT_GEMM != mb8`): X is cast F32 → F16 into a scratch
///   `[n, K_pad]` (`K_pad = round_up(in_dim, 64)`, pad columns written as
///   zeros), then the LDS-staged WMMA GEMM runs with `K = K_pad`. The weight
///   must be stored at pitch `K_pad` (`w_ld == K_pad`; zero pad columns), so
///   the extra terms are exact `0 * 0`. gfx1201 uses the split-1 tiles of
///   `gemm_f16_x_f16_wmma_lds_splitk[_ld]` (no fused bias, so `bias_add_f32`
///   follows); gfx11 uses `gemm_f16_x_f16_wmma_lds_auto_ld` with the bias fused.
/// * **mb8**: `gemm_f16_wmma_mb8` (F32 X) + `bias_add_f32`, on any other WMMA
///   arch and whenever the LDS route is off (also for a packed weight with
///   `in_dim % 64 != 0`).
/// * Non-WMMA arches keep the naive `gemm_f16` + transpose fallback.
///
/// Precision: unchanged. `gemm_f16_wmma_mb8` already rounds each X element
/// to F16 for its WMMA B operand, so the explicit F32 -> F16 cast here is the
/// same rounding; both accumulate in F32 over ascending 16-wide K steps (the
/// tower shapes measured bit-identical to mb8 on gfx1201, see
/// `bench_vit_gemm_shapes`). Weights unchanged (already F16). No bf16/fp8.
#[allow(clippy::too_many_arguments)]
fn linear_f16_ld(
    gpu: &mut Gpu,
    w: &GpuTensor,
    w_ld: usize,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let k_pad = in_dim.next_multiple_of(VIT_LDS_K_ALIGN);
    let lds = vit_lds_route(gpu, out_dim) && w_ld == k_pad;
    if !lds && w_ld != in_dim {
        return Err(HipError::new(
            1,
            &format!(
                "linear_f16: weight row pitch {w_ld} needs the LDS GEMM route \
                 (in_dim {in_dim}, K_pad {k_pad}, out_dim {out_dim}) but it is not active"
            ),
        ));
    }
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    if lds {
        let x16 = match gpu.alloc_tensor(&[n * k_pad], DType::F16) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(y);
                return Err(e);
            }
        };
        let r = (|| -> HipResult<()> {
            if k_pad == in_dim {
                gpu.cast_f32_to_f16(x, &x16)?;
            } else {
                gpu.cast_f32_to_f16_pitched(x, &x16, n, in_dim, k_pad)?;
            }
            if gpu.arch == "gfx1201" {
                let tile = vit_gfx1201_tile(out_dim);
                if k_pad == in_dim {
                    gpu.gemm_f16_x_f16_wmma_lds_splitk(w, &x16, &y, out_dim, k_pad, n, tile)?;
                } else {
                    gpu.gemm_f16_x_f16_wmma_lds_splitk_ld(
                        w, &x16, &y, out_dim, k_pad, n, tile, k_pad, k_pad,
                    )?;
                }
                gpu.bias_add_f32(&y, bias, n, out_dim)
            } else {
                // gfx11: bias fused into the GEMM epilogue.
                gpu.gemm_f16_x_f16_wmma_lds_auto_ld(
                    w,
                    &x16,
                    &y,
                    Some(bias),
                    out_dim,
                    k_pad,
                    n,
                    k_pad,
                    k_pad,
                )
            }
        })();
        let fr = gpu.free_tensor(x16);
        if let Err(e) = r.and(fr) {
            let _ = gpu.free_tensor(y);
            return Err(e);
        }
        return Ok(y);
    }
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        // GEMM: Y_t[out_dim, n] = W[out_dim, in_dim] @ X[n, in_dim]^T
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

/// Diagnostic-only stage dump. Writes `<name>.bin` (raw little-endian f32) plus
/// a `<name>.json` sidecar in the layout `benchmarks/vision/diff_dumps.py`
/// expects, so hipfire stages can be diffed against
/// `benchmarks/vision/dump_hf_reference.py` output stage-by-stage.
///
/// No-op unless `HIPFIRE_VL_DUMP_DIR` is set. This is the "Gap 3" per-block
/// tooling called for in `benchmarks/vision/comparison-2026-05-23.md`
/// ("would let us catch any silent regression in future kernel work touching
/// the vision tower"). Downloading each stage forces a sync, so this is
/// strictly a debugging path and must stay env-gated.
fn vl_dump_slice(dir: &std::path::Path, name: &str, data: &[f32], shape: &[usize]) {
    use std::io::Write;
    let _ = std::fs::create_dir_all(dir);
    if let Ok(mut f) = std::fs::File::create(dir.join(format!("{name}.bin"))) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = f.write_all(&bytes);
    }
    let shape_s = shape
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let _ = std::fs::write(
        dir.join(format!("{name}.json")),
        format!(r#"{{"shape":[{shape_s}],"dtype":"<f4"}}"#),
    );
}

fn vl_dump_tensor(
    gpu: &mut Gpu,
    dir: Option<&std::path::Path>,
    name: &str,
    t: &GpuTensor,
    shape: &[usize],
) -> HipResult<()> {
    let Some(dir) = dir else { return Ok(()) };
    let data = gpu.download_f32(t)?;
    vl_dump_slice(dir, name, &data, shape);
    Ok(())
}

/// Resumable vision-tower encode: prologue once, ONE transformer layer per
/// [`VisionTowerJob::step_layer`] call, merger epilogue in
/// [`VisionTowerJob::finish`].
///
/// The slot engine's GPU thread is the exclusive GPU owner, so a monolithic
/// encode wedges every other slot's decode for the whole tower pass — at a
/// 78×78 grid (6084 patches) that is ~9.7 s even on the Q-tiled attention
/// kernel. Splitting the pass into per-layer steps lets the engine interleave
/// other slots' decode batches between tower layers, bounding the
/// co-tenant stall to one layer instead of the whole encode.
///
/// Stream invariant: every kernel is enqueued on the same launch stream
/// (`gpu.active_stream`; the default stream when unset), and per-layer
/// scratch tensors (`tmp`, `qkv`, `attn_out`, `proj`, `tmp2`, `fc1`, `fc2`)
/// are freed within the same step. Pool reuse of a freed buffer stays
/// submission-ordered even with other slots' kernels interleaved between
/// steps: all activations share that one stream. Streamed tower weights
/// ([`TowerResidency::Streamed`]) add ONE more stream, a private copy stream
/// that carries only the weight H2D copies and is ordered against the launch
/// stream by per-slot events (see [`TowerStream`]): a block's kernels wait on
/// the `ready` event of its copy, and the next copy into that staging slot
/// waits on the `free` event recorded after those kernels. Activations never
/// leave the launch stream. If `rdna_compute` ever uses multiple streams for
/// the vision activations, this must be revisited (per-step syncs or
/// stream-attached frees). See review notes in `vision_rev_claude.md`.
pub struct VisionTowerJob {
    /// Hidden residual `[n, h]`, live across all layers.
    x: GpuTensor,
    rope_cos_gpu: GpuTensor,
    rope_sin_gpu: GpuTensor,
    n: usize,
    grid_h: usize,
    grid_w: usize,
    /// Next tower layer to run (`0..num_layers`).
    next_layer: usize,
    /// Self-attention kernel for this image (HIPFIRE_VIT_ATTN / arch / head_dim).
    attn: VitAttn,
    dump_dir: Option<std::path::PathBuf>,
    t0: std::time::Instant,
}

/// ViT self-attention kernel choice, made once per image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VitAttn {
    /// WMMA flash kernel, head_dim <= 80 (zero-padded to 80), gfx11/gfx12.
    Wmma,
    /// Scalar Q-tiled flash kernel, any head_dim <= 128.
    QTiled,
    /// Per-(head, query) fallback, any head_dim.
    Naive,
}

/// Resolve `HIPFIRE_VIT_ATTN` (`auto|wmma|qtiled|naive`, default auto).
/// auto: Wmma when the arch has wave32 WMMA and head_dim <= 80, else QTiled
/// when head_dim <= 128, else Naive. Explicit `wmma` errors if the arch lacks
/// WMMA or head_dim > 80; explicit `qtiled` with head_dim > 128 falls back to
/// naive inside the dispatcher.
fn select_vit_attn(gpu: &Gpu, head_dim: usize) -> HipResult<VitAttn> {
    let has_wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12();
    let var = hipfire_config::developer_var("HIPFIRE_VIT_ATTN");
    match var.as_deref() {
        Ok("naive") => Ok(VitAttn::Naive),
        Ok("qtiled") => Ok(VitAttn::QTiled),
        Ok("wmma") => {
            if has_wmma && head_dim <= 80 {
                Ok(VitAttn::Wmma)
            } else {
                Err(HipError::new(
                    0,
                    &format!(
                        "HIPFIRE_VIT_ATTN=wmma needs wave32 WMMA and head_dim <= 80 \
                         (arch={}, head_dim={head_dim})",
                        gpu.arch
                    ),
                ))
            }
        }
        _ => Ok(if has_wmma && head_dim <= 80 {
            VitAttn::Wmma
        } else if head_dim <= 128 {
            VitAttn::QTiled
        } else {
            VitAttn::Naive
        }),
    }
}

/// One block's large matrices as the kernels take them: borrowed from the
/// resident tensors, or non-owning views into a staging slot.
struct LayerMats<'a> {
    qkv_w: &'a GpuTensor,
    proj_w: &'a GpuTensor,
    fc1_w: &'a GpuTensor,
    fc2_w: &'a GpuTensor,
}

impl<'a> LayerMats<'a> {
    fn resident(m: &'a VisionLayerMats) -> Self {
        Self {
            qkv_w: &m.qkv_w,
            proj_w: &m.proj_w,
            fc1_w: &m.fc1_w,
            fc2_w: &m.fc2_w,
        }
    }

    fn staged(views: &'a [GpuTensor]) -> HipResult<Self> {
        match views {
            [qkv_w, proj_w, fc1_w, fc2_w] => Ok(Self {
                qkv_w,
                proj_w,
                fc1_w,
                fc2_w,
            }),
            _ => Err(HipError::new(
                1,
                "vision tower stream: block unit is not four matrices",
            )),
        }
    }
}

impl VisionTowerJob {
    /// Tower prologue: upload patches, patch-embed, interpolate + add the
    /// learned position embedding, upload the 2D rope tables.
    pub fn new(
        gpu: &mut Gpu,
        weights: &VisionWeights,
        config: &VisionConfig,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> HipResult<Self> {
        let h = config.hidden_size;
        let n = grid_h * grid_w;
        let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
        let t0 = std::time::Instant::now();

        // Diagnostic stage dumps (env-gated; see `vl_dump_slice`).
        let dump_dir: Option<std::path::PathBuf> =
            hipfire_config::developer_var("HIPFIRE_VL_DUMP_DIR")
                .ok()
                .map(Into::into);
        let dd = dump_dir.as_deref();
        if dd.is_some() {
            eprintln!(
                "  [VL-DUMP] stage dumps enabled ({} patches, {grid_h}x{grid_w})",
                n
            );
        }
        eprintln!(
            "  vision forward (GPU): {} patches, {}x{} grid",
            n, grid_h, grid_w
        );

        if let Some(d) = dd {
            vl_dump_slice(d, "pixel_values", patches, &[n, patch_dim]);
        }

        // Upload patches [n, patch_dim]
        let x_patches = gpu.upload_f32(patches, &[n * patch_dim])?;

        // Patch embedding: linear_f16 → [n, h]
        let x = linear_f16(
            gpu,
            &weights.patch_embed_w,
            &x_patches,
            &weights.patch_embed_b,
            h,
            patch_dim,
            n,
        )?;
        weights.prefetch_unit(gpu, 0);
        gpu.free_tensor(x_patches)?;
        vl_dump_tensor(gpu, dd, "patch_embed", &x, &[n, h])?;

        // Bilinear-interpolate the learned (K×K, h) pos_embed table down to the
        // actual (grid_h, grid_w) and reorder into 2x2-grouped patch sequence,
        // then add. HF's `fast_pos_embed_interpolate`.
        let num_grid_per_side = (config.num_position_embeddings as f64).sqrt().round() as usize;
        // Defensive re-check (never a panic): `load_vision_weights` already
        // refused a non-square table at model load, so reaching here with one
        // means the weights bypassed that loader. A typed error keeps the
        // engine thread alive instead of poisoning every slot.
        if num_grid_per_side == 0
            || num_grid_per_side * num_grid_per_side != config.num_position_embeddings
        {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "num_position_embeddings ({}) must be a perfect square",
                    config.num_position_embeddings
                ),
            ));
        }
        let pos_embed_interp = fast_pos_embed_interpolate(
            &weights.pos_embed,
            h,
            grid_h,
            grid_w,
            num_grid_per_side,
            config.spatial_merge_size,
        );
        let pos_embed_gpu = gpu.upload_f32(&pos_embed_interp, &[n * h])?;
        gpu.add_inplace_f32(&x, &pos_embed_gpu)?;
        gpu.free_tensor(pos_embed_gpu)?;
        vl_dump_tensor(gpu, dd, "post_pos_embed", &x, &[n, h])?;
        if let Some(d) = dd {
            vl_dump_slice(d, "pos_embed_interp", &pos_embed_interp, &[n, h]);
        }

        // Compute the 2D rotary cos/sin tables once per image and upload. The
        // kernel reads `head_dim/2` floats per token for each of cos/sin (HF's
        // `cat((rope, rope), dim=-1)` makes the two head_dim halves see the same
        // angle, so we store the half only).
        let rot_dim_half = config.head_dim / 2;
        let (rope_cos, rope_sin) = compute_vision_rope_cos_sin(
            grid_h,
            grid_w,
            config.head_dim,
            config.spatial_merge_size,
            config.rope_theta,
        );
        let rope_cos_gpu = gpu.upload_f32(&rope_cos, &[n * rot_dim_half])?;
        let rope_sin_gpu = gpu.upload_f32(&rope_sin, &[n * rot_dim_half])?;

        // Attention dispatch (see `select_vit_attn`): the WMMA flash kernel
        // (head_dim <= 80, padded to 80 for head_dim 72) is the production
        // path on wave32-WMMA parts — the scalar Q-tiled kernel and the
        // per-(head, query) naive kernel (1.02 s/layer at a 68x68 grid on
        // gfx1101; naive = 92% of tower time at N=4096 on gfx1201) remain as
        // fallbacks / A-B switches via HIPFIRE_VIT_ATTN=auto|wmma|qtiled|naive.
        let attn = select_vit_attn(gpu, config.head_dim)?;
        weights.prefetch_unit(gpu, 1);

        Ok(Self {
            x,
            rope_cos_gpu,
            rope_sin_gpu,
            n,
            grid_h,
            grid_w,
            next_layer: 0,
            attn,
            dump_dir,
            t0,
        })
    }

    /// Like [`Self::new`] but takes device-resident patches (the VCN JPEG
    /// decode path in `image`). Skips the host→device upload — `x_patches`
    /// is consumed by the patch embedding and freed immediately after.
    pub fn new_patches(
        gpu: &mut Gpu,
        weights: &VisionWeights,
        config: &VisionConfig,
        x_patches: GpuTensor,
        grid_h: usize,
        grid_w: usize,
    ) -> HipResult<Self> {
        let h = config.hidden_size;
        let n = grid_h * grid_w;
        let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
        let t0 = std::time::Instant::now();

        let dump_dir: Option<std::path::PathBuf> =
            hipfire_config::developer_var("HIPFIRE_VL_DUMP_DIR")
                .ok()
                .map(Into::into);
        let dd = dump_dir.as_deref();
        if dd.is_some() {
            eprintln!(
                "  [VL-DUMP] stage dumps enabled ({} patches, {grid_h}x{grid_w})",
                n
            );
        }
        eprintln!(
            "  vision forward (GPU): {} patches, {}x{} grid",
            n, grid_h, grid_w
        );

        if let Some(d) = dd {
            // Env-gated debug dump: a download failure must still release the
            // owned input before propagating.
            match gpu.download_f32(&x_patches) {
                Ok(host) => vl_dump_slice(d, "pixel_values", &host, &[n, patch_dim]),
                Err(e) => {
                    let _ = gpu.free_tensor(x_patches);
                    return Err(e);
                }
            }
        }

        // Patch embedding: linear_f16 → [n, h]. Last use of the owned input:
        // release it here — not at tower end — on both success and failure.
        let x = match linear_f16(
            gpu,
            &weights.patch_embed_w,
            &x_patches,
            &weights.patch_embed_b,
            h,
            patch_dim,
            n,
        ) {
            Ok(x) => {
                weights.prefetch_unit(gpu, 0);
                gpu.free_tensor(x_patches)?;
                x
            }
            Err(e) => {
                let _ = gpu.free_tensor(x_patches);
                return Err(e);
            }
        };
        vl_dump_tensor(gpu, dd, "patch_embed", &x, &[n, h])?;

        // Bilinear-interpolate the learned (K×K, h) pos_embed table down to the
        // actual (grid_h, grid_w) and reorder into 2x2-grouped patch sequence,
        // then add. HF's `fast_pos_embed_interpolate`.
        let num_grid_per_side = (config.num_position_embeddings as f64).sqrt().round() as usize;
        if num_grid_per_side == 0
            || num_grid_per_side * num_grid_per_side != config.num_position_embeddings
        {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "num_position_embeddings ({}) must be a perfect square",
                    config.num_position_embeddings
                ),
            ));
        }
        let pos_embed_interp = fast_pos_embed_interpolate(
            &weights.pos_embed,
            h,
            grid_h,
            grid_w,
            num_grid_per_side,
            config.spatial_merge_size,
        );
        let pos_embed_gpu = gpu.upload_f32(&pos_embed_interp, &[n * h])?;
        gpu.add_inplace_f32(&x, &pos_embed_gpu)?;
        gpu.free_tensor(pos_embed_gpu)?;
        vl_dump_tensor(gpu, dd, "post_pos_embed", &x, &[n, h])?;
        if let Some(d) = dd {
            vl_dump_slice(d, "pos_embed_interp", &pos_embed_interp, &[n, h]);
        }

        let rot_dim_half = config.head_dim / 2;
        let (rope_cos, rope_sin) = compute_vision_rope_cos_sin(
            grid_h,
            grid_w,
            config.head_dim,
            config.spatial_merge_size,
            config.rope_theta,
        );
        let rope_cos_gpu = gpu.upload_f32(&rope_cos, &[n * rot_dim_half])?;
        let rope_sin_gpu = gpu.upload_f32(&rope_sin, &[n * rot_dim_half])?;

        let attn = select_vit_attn(gpu, config.head_dim)?;
        weights.prefetch_unit(gpu, 1);

        Ok(Self {
            x,
            rope_cos_gpu,
            rope_sin_gpu,
            n,
            grid_h,
            grid_w,
            next_layer: 0,
            attn,
            dump_dir,
            t0,
        })
    }

    /// Run the NEXT single tower layer. Returns `true` when the last layer
    /// has just run — the job is then ready for [`Self::finish`]; calling
    /// `step_layer` again after that panics (contract violation).
    pub fn step_layer(
        &mut self,
        gpu: &mut Gpu,
        weights: &VisionWeights,
        config: &VisionConfig,
    ) -> HipResult<bool> {
        assert!(
            self.next_layer < config.num_layers,
            "step_layer past the last tower layer"
        );
        let h = config.hidden_size;
        let n = self.n;
        let li = self.next_layer;
        let lw = &weights.layers[li];

        match &weights.mats {
            TowerMats::Resident { layers, .. } => {
                self.layer_kernels(gpu, config, lw, LayerMats::resident(&layers[li]))?;
            }
            TowerMats::Streamed(stream) => {
                // `with_unit` records the slot's release after the kernels
                // were enqueued, also when one of them failed to launch.
                stream.with_unit(gpu, li, |gpu, staged| {
                    self.layer_kernels(gpu, config, lw, LayerMats::staged(staged)?)
                })?;
                // Block `li + 2` (or a merger matrix) takes this slot next,
                // copying while the next block computes.
                stream.prefetch(gpu, li + 2);
            }
        }

        let dd = self.dump_dir.as_deref();
        vl_dump_tensor(gpu, dd, &format!("block_{li:02}"), &self.x, &[n, h])?;
        self.next_layer += 1;
        let done = self.next_layer == config.num_layers;
        if done {
            vl_dump_tensor(gpu, dd, "pre_merger", &self.x, &[n, h])?;
        }
        Ok(done)
    }

    /// The kernel sequence of one block, `mats` being its large matrices.
    fn layer_kernels(
        &self,
        gpu: &mut Gpu,
        config: &VisionConfig,
        lw: &VisionLayerWeights,
        mats: LayerMats<'_>,
    ) -> HipResult<()> {
        let h = config.hidden_size;
        let n = self.n;

        // LayerNorm1 → tmp
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(
            &self.x,
            &lw.norm1_w,
            &lw.norm1_b,
            &tmp,
            n,
            h,
            config.norm_eps,
        )?;

        // QKV projection → [n, 3h]
        let qkv = linear_f16(gpu, mats.qkv_w, &tmp, &lw.qkv_b, 3 * h, h, n)?;
        gpu.free_tensor(tmp)?;

        // 2D rotary on Q and K
        gpu.apply_rope_2d_vision_f32(
            &qkv,
            &self.rope_cos_gpu,
            &self.rope_sin_gpu,
            n,
            h,
            config.num_heads,
            config.head_dim,
        )?;

        // Self-attention
        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
        match self.attn {
            VitAttn::Naive => {
                gpu.vit_attention_f32(&qkv, &attn_out, n, h, config.num_heads, config.head_dim)?
            }
            VitAttn::QTiled => gpu.vit_attention_qtiled_f32(
                &qkv,
                &attn_out,
                n,
                h,
                config.num_heads,
                config.head_dim,
            )?,
            VitAttn::Wmma => gpu.vit_attention_wmma_f32(
                &qkv,
                &attn_out,
                n,
                h,
                config.num_heads,
                config.head_dim,
            )?,
        }
        gpu.free_tensor(qkv)?;

        // Output projection + residual
        let proj = linear_f16(gpu, mats.proj_w, &attn_out, &lw.proj_b, h, h, n)?;
        gpu.free_tensor(attn_out)?;

        // Residual: x += proj
        gpu.add_inplace_f32(&self.x, &proj)?;
        gpu.free_tensor(proj)?;

        // LayerNorm2 → tmp
        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(
            &self.x,
            &lw.norm2_w,
            &lw.norm2_b,
            &tmp2,
            n,
            h,
            config.norm_eps,
        )?;

        // MLP: fc1 → GELU → fc2 + residual
        let fc1 = linear_f16(gpu, mats.fc1_w, &tmp2, &lw.fc1_b, config.mlp_dim, h, n)?;
        gpu.free_tensor(tmp2)?;
        gpu.gelu_tanh_f32(&fc1, &fc1, n * config.mlp_dim)?;
        let fc2 = linear_f16_ld(
            gpu,
            mats.fc2_w,
            lw.fc2_ld,
            &fc1,
            &lw.fc2_b,
            h,
            config.mlp_dim,
            n,
        )?;
        gpu.free_tensor(fc1)?;

        // Residual: x += fc2
        gpu.add_inplace_f32(&self.x, &fc2)?;
        gpu.free_tensor(fc2)?;
        Ok(())
    }

    /// Tower epilogue: final norm + spatial merge + merger MLP. Consumes the
    /// job and frees its GPU buffers on the happy path.
    pub fn finish(
        self,
        gpu: &mut Gpu,
        weights: &VisionWeights,
        config: &VisionConfig,
    ) -> HipResult<Vec<f32>> {
        let Self {
            x,
            rope_cos_gpu,
            rope_sin_gpu,
            n,
            grid_h,
            grid_w,
            dump_dir,
            t0,
            ..
        } = self;
        let h = config.hidden_size;

        // Single sync at end of all layers (avoids per-layer sync overhead).
        // Every error arm below releases the tensors this epilogue owns: the
        // tower's device buffers are neither pooled nor released by Drop, so a
        // failure partway through the merger must not strand them.
        if let Err(e) = gpu.hip.device_synchronize() {
            let _ = gpu.free_tensor(x);
            let _ = gpu.free_tensor(rope_cos_gpu);
            let _ = gpu.free_tensor(rope_sin_gpu);
            return Err(e);
        }
        gpu.free_tensor(rope_cos_gpu)?;
        gpu.free_tensor(rope_sin_gpu)?;
        eprintln!(
            "  vision forward complete ({:.2}s)",
            t0.elapsed().as_secs_f32()
        );

        // Spatial merge: [n, h] → [n_merged, merge_dim] (CPU rearrange, small data)
        let sms = config.spatial_merge_size;
        let merged_h = grid_h / sms;
        let merged_w = grid_w / sms;
        let n_merged = merged_h * merged_w;
        let merge_dim = h * sms * sms;

        // LayerNorm all patches
        let normed = gpu.alloc_tensor(&[n * h], DType::F32)?;
        if let Err(e) = gpu.layernorm_batched(
            &x,
            &weights.merger_norm_w,
            &weights.merger_norm_b,
            &normed,
            n,
            h,
            config.norm_eps,
        ) {
            let _ = gpu.free_tensor(normed);
            let _ = gpu.free_tensor(x);
            return Err(e);
        }
        gpu.free_tensor(x)?;

        // Download for 2x2 rearrange (only ~3.6MB, one-time cost)
        let normed_data = match gpu.download_f32(&normed) {
            Ok(data) => data,
            Err(e) => {
                let _ = gpu.free_tensor(normed);
                return Err(e);
            }
        };
        gpu.free_tensor(normed)?;

        // Patches in `normed_data` are stored in 2x2-block-grouped order (see
        // `extract_patches`), so the 4 patches that merge into one output token
        // are CONSECUTIVE in the buffer: src indices [out_idx*4 .. out_idx*4+4].
        let mut merged = vec![0.0f32; n_merged * merge_dim];
        for my in 0..merged_h {
            for mx in 0..merged_w {
                let out_idx = my * merged_w + mx;
                for sub in 0..(sms * sms) {
                    let src = out_idx * (sms * sms) + sub;
                    merged[out_idx * merge_dim + sub * h..out_idx * merge_dim + sub * h + h]
                        .copy_from_slice(&normed_data[src * h..src * h + h]);
                }
            }
        }

        // Merger MLP on GPU
        let merged_gpu = gpu.upload_f32(&merged, &[n_merged * merge_dim])?;
        let m1 = match weights.with_merger_w(gpu, MergerFc::Fc1, |gpu, w| {
            linear_f16(
                gpu,
                w,
                &merged_gpu,
                &weights.merger_fc1_b,
                merge_dim,
                merge_dim,
                n_merged,
            )
        }) {
            Ok(m1) => m1,
            Err(e) => {
                let _ = gpu.free_tensor(merged_gpu);
                return Err(e);
            }
        };
        gpu.free_tensor(merged_gpu)?;
        if let Err(e) = gpu.gelu_tanh_f32(&m1, &m1, n_merged * merge_dim) {
            let _ = gpu.free_tensor(m1);
            return Err(e);
        }

        let m2 = match weights.with_merger_w(gpu, MergerFc::Fc2, |gpu, w| {
            linear_f16(
                gpu,
                w,
                &m1,
                &weights.merger_fc2_b,
                config.out_hidden_size,
                merge_dim,
                n_merged,
            )
        }) {
            Ok(m2) => m2,
            Err(e) => {
                let _ = gpu.free_tensor(m1);
                return Err(e);
            }
        };
        gpu.free_tensor(m1)?;

        let result = match gpu.download_f32(&m2) {
            Ok(result) => result,
            Err(e) => {
                let _ = gpu.free_tensor(m2);
                return Err(e);
            }
        };
        gpu.free_tensor(m2)?;

        if let Some(d) = dump_dir.as_deref() {
            vl_dump_slice(
                d,
                "post_merger",
                &result,
                &[n_merged, config.out_hidden_size],
            );
            vl_dump_slice(d, "merged_pre_mlp", &merged, &[n_merged, merge_dim]);
        }

        eprintln!(
            "  vision done: {} tokens × {} dims ({:.2}s)",
            n_merged,
            config.out_hidden_size,
            t0.elapsed().as_secs_f32()
        );
        Ok(result)
    }

    /// Free a job that will never reach `finish` (slot aborted/evicted
    /// mid-encode). Consumes the job; best-effort — keeps going after
    /// individual free failures and returns the first error.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        let Self {
            x,
            rope_cos_gpu,
            rope_sin_gpu,
            ..
        } = self;
        let mut first_err: Option<HipError> = None;
        for t in [x, rope_cos_gpu, rope_sin_gpu] {
            if let Err(e) = gpu.free_tensor(t) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Monolithic encode: prologue, every tower layer back-to-back, merger
/// epilogue. Thin wrapper over [`VisionTowerJob`] — the sequential VL path
/// and the lab examples use this; the slot engine drives the job directly so
/// other slots' decode steps can interleave between layers.
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    config: &VisionConfig,
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> HipResult<Vec<f32>> {
    let mut job = VisionTowerJob::new(gpu, weights, config, patches, grid_h, grid_w)?;
    let mut done = false;
    while !done {
        done = job.step_layer(gpu, weights, config)?;
    }
    job.finish(gpu, weights, config)
}

/// `vision_forward` without the host→device patch upload: encode patches
/// already resident on the device (the VCN product path in `image`). Takes
/// `x_patches` BY VALUE and frees it immediately after patch embedding —
/// its last use — so the peak drops by the full image tensor instead of
/// holding it through the whole tower, and no later (or embedding-error)
/// path can strand it. Same contract for CPU-uploaded and VCN-produced
/// input: the caller owns nothing after the call.
pub fn vision_forward_patches(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    config: &VisionConfig,
    x_patches: GpuTensor,
    grid_h: usize,
    grid_w: usize,
) -> HipResult<Vec<f32>> {
    let mut job = VisionTowerJob::new_patches(gpu, weights, config, x_patches, grid_h, grid_w)?;
    let mut done = false;
    while !done {
        done = job.step_layer(gpu, weights, config)?;
    }
    job.finish(gpu, weights, config)
}

/// Largest patch count the product preprocessing can emit: the image pixel
/// budget (`HIPFIRE_VL_MAX_PIXELS`, default 2 MP) over the pixels of one
/// patch.
pub fn max_image_patches(config: &VisionConfig) -> usize {
    crate::image::vision_max_pixels() / (config.patch_size * config.patch_size).max(1)
}

/// Upper bound on the device bytes the tower's activations hold at once while
/// encoding one image of `n_patches` patches. Weights are not counted
/// (resident ones are separate, streamed ones are the two fixed staging
/// slots). It is the maximum, over every allocation phase of the prologue,
/// of one block ([`VisionTowerJob::step_layer`]) and of the merger epilogue
/// ([`VisionTowerJob::finish`]), of the device tensors live at that phase,
/// taken in allocation order. A `linear_f16_ld` holds its output plus the
/// larger of the LDS route's F16 copy of its input and the non-WMMA route's
/// transposed output, whichever route the arch takes; the WMMA attention adds
/// its three packed F16 q/k/v buffers (the Q-tiled and naive kernels allocate
/// nothing). The pool returns its idle buffers to HIP before an allocation
/// would otherwise fail, so the live-set maximum is what must stay free.
pub fn tower_activation_peak_bytes(config: &VisionConfig, n_patches: usize) -> usize {
    const F32: usize = 4;
    let n = n_patches;
    let h = config.hidden_size;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let merge_sq = config.spatial_merge_size * config.spatial_merge_size;
    let merge_dim = h * merge_sq;
    let n_merged = n.div_ceil(merge_sq.max(1));
    let k_pad = |k: usize| k.next_multiple_of(VIT_LDS_K_ALIGN);
    let linear = |rows: usize, in_dim: usize, out_dim: usize| {
        rows * out_dim * F32 + (rows * k_pad(in_dim) * F16_BYTES).max(rows * out_dim * F32)
    };
    // `[n, h]` F32: the residual `x`, and each of tmp / attn_out / proj /
    // tmp2 / fc2.
    let act = n * h * F32;
    let qkv = act * 3;
    let fc1 = n * config.mlp_dim * F32;
    // Live through every block: the residual and the rope cos/sin tables.
    let block = act + 2 * n * (config.head_dim / 2) * F32;
    let attn_pack = if config.head_dim <= 80 {
        3 * n * config.num_heads * 80 * F16_BYTES
    } else {
        0
    };
    [
        // Prologue: patch upload + patch-embed linear; pos-embed upload
        // (also the merger's layernorm: x + normed).
        n * patch_dim * F32 + linear(n, patch_dim, h),
        act + act,
        // Block: layernorm1 / layernorm2 output; qkv linear; attention; output
        // projection; fc1 linear; fc2 linear.
        block + act,
        block + act + linear(n, h, 3 * h),
        block + qkv + act + attn_pack,
        block + act + linear(n, h, h),
        block + act + linear(n, h, config.mlp_dim),
        block + fc1 + linear(n, config.mlp_dim, h),
        // Merger epilogue: fc1 linear (merged upload live); fc2 linear (m1
        // live).
        n_merged * merge_dim * F32 + linear(n_merged, merge_dim, merge_dim),
        n_merged * merge_dim * F32 + linear(n_merged, merge_dim, config.out_hidden_size),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
}
