// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GGUF-native weight source for the dense Qwen3.5 / 3.6 / 3.8 family
//! (`HIPFIRE_GGUF_NATIVE=1`, hipfire arch 5).
//!
//! The weights stay in their ggml block bytes (`DType::Ggml*`, never
//! re-quantised); this source only undoes the data edits the exporter made
//! (see [`crate::gguf_map`]) so the engine's unchanged GDN / attention / norm
//! kernels read what they expect:
//!
//! * RMSNorm weights are stored with `+1` folded in, which is exactly what the
//!   engine's norm kernels take: loaded as stored (the HFQ path adds the `+1`
//!   itself).
//! * `ssm_a = -exp(A_log)`: the engine's gate kernels compute
//!   `-expf(a_log)`, so `A_log = ln(-ssm_a)` (evaluated in f64).
//! * The depthwise conv kernel is `[conv_dim, K]` already (squeezed).
//! * Value heads stored tiled (`g = j * nk + kh`) instead of grouped
//!   (`h = kh * r + j`): row blocks of `in_proj_qkv` (V part), `in_proj_z`,
//!   `in_proj_a/b`, the V channels of the conv kernel and the per-head vectors
//!   `A_log` / `dt_bias` are permuted on the host (a row permutation is a
//!   memcpy of whole rows, never a re-quantisation). The K axis of `out_proj`
//!   cannot be permuted inside 256-element blocks, so its bytes are kept and
//!   the head permutation travels with the weight as a K-chunk map
//!   ([`rdna_compute::ggml_ops`]) that the decode producer and the prefill
//!   dequantisation apply to the activation / destination columns.
//!
//! The NextN/MTP block is never loaded (AR only).

use crate::gguf_map::{
    ExpectedTensor, GgufTransform, Qwen35GgufHparams, Qwen35GgufMap, TensorClass, TensorGroup,
    VHeadOrder,
};
use crate::qwen35::{LayerWeights, Qwen35Config};
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::gguf::{GgmlType, GgufFile, TensorInfo};
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_load::WeightSource;
use hipfire_runtime::offload::LayerResidency;
use hipfire_runtime::weight_backend::WeightBackend;
use rdna_compute::ggml_ops::{
    ggml_build_kmap, ggml_kmap_offset, ggml_tiled_to_grouped_heads, ggml_weight_buffer_bytes,
    GGML_KMAP_CHUNK,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

/// The dtype carrying a ggml type's bytes, or `None` when the type is not
/// served natively (tier-1 types of the admission plan).
pub fn ggml_type_dtype(ty: GgmlType) -> Option<DType> {
    Some(match ty {
        GgmlType::Q4K => DType::GgmlQ4K,
        GgmlType::Q5K => DType::GgmlQ5K,
        GgmlType::Q6K => DType::GgmlQ6K,
        GgmlType::Q8_0 => DType::GgmlQ8_0,
        GgmlType::IQ4XS => DType::GgmlIQ4XS,
        _ => return None,
    })
}

/// `HIPFIRE_GGUF_CODEC_ONLY=1`: prefill dequantises each layer to BF16-rounded
/// weights (the codec-only quality tier: host-equivalent dequant, BF16 weights,
/// F32-class activations carried as an exact hi + lo F16 pair, F32 accumulate,
/// no q8_1 activation quantisation) instead of plain F16. Prefill
/// and scoring only; the decode GEMV always runs the shipped A8 route.
pub const CODEC_ONLY_ENV: &str = "HIPFIRE_GGUF_CODEC_ONLY";

/// `HIPFIRE_GGUF_A8_EMU=1`: round every prefill GEMM input through q8_1
/// (per-32 int8 + f16 scale) before the multiply, so scoring through the
/// prefill route charges the `ggml.a8q81` activation quantisation the decode
/// route applies. Emulation of the shipped route's activation numerics in
/// prefill (the integer dot and f32 scale fold of the decode kernels are
/// replaced by the F16 GEMM on dequantised weights); measurement tier only.
pub const A8_EMU_ENV: &str = "HIPFIRE_GGUF_A8_EMU";

/// Prefill dequantisation target of the loaded model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillDequant {
    /// F16 weights, the engine's F16 WMMA GEMM (shipped route, A16 prefill).
    F16,
    /// BF16-rounded weights (F32 dequantisation plus its F16 copy), hi + lo
    /// activation split on the F16 WMMA GEMM (codec-only tier).
    Bf16RoundedF32,
}

impl PrefillDequant {
    pub fn from_env() -> Self {
        if hipfire_config::developer_var(CODEC_ONLY_ENV).is_ok_and(|v| v == "1") {
            PrefillDequant::Bf16RoundedF32
        } else {
            PrefillDequant::F16
        }
    }

    pub fn elem_bytes(self) -> usize {
        match self {
            PrefillDequant::F16 => 2,
            PrefillDequant::Bf16RoundedF32 => 6,
        }
    }
}

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

/// Grouped head `h` -> tiled (stored) head position `g` for a value-head axis
/// the exporter re-ordered: the stored head index to read for grouped head `h`.
#[inline]
fn stored_head(h: usize, nk: usize, r: usize) -> usize {
    (h % r) * nk + h / r
}

fn f32_from_bytes(ty: GgmlType, data: &[u8]) -> Result<Vec<f32>, HipError> {
    match ty {
        GgmlType::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GgmlType::F16 => Ok(data
            .chunks_exact(2)
            .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        GgmlType::BF16 => Ok(data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        other => Err(err(format!(
            "load refused: GGUF tensor type {} is not a float type this loader reads on the host",
            other.name()
        ))),
    }
}

/// Reorder `rows` (each `row_len` units) in blocks of `block_rows` rows: output
/// block `h` copies stored block `stored_head(h)`. `skip` leading units are
/// copied verbatim.
fn permute_row_blocks<T: Copy>(
    data: &[T],
    row_len: usize,
    skip_rows: usize,
    block_rows: usize,
    nk: usize,
    r: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..skip_rows * row_len]);
    let heads = nk * r;
    for h in 0..heads {
        let g = stored_head(h, nk, r);
        let start = (skip_rows + g * block_rows) * row_len;
        out.extend_from_slice(&data[start..start + block_rows * row_len]);
    }
    debug_assert_eq!(out.len(), data.len());
    out
}

/// Whole-model GGUF weight source.
pub struct GgufSource<'a> {
    gguf: &'a GgufFile,
    config: &'a Qwen35Config,
    map: Qwen35GgufMap,
    by_hipfire: HashMap<String, usize>,
    prefill: PrefillDequant,
}

impl<'a> GgufSource<'a> {
    pub fn new(gguf: &'a GgufFile, hp: &'a Qwen35GgufHparams) -> Self {
        let map = Qwen35GgufMap::new(hp);
        let by_hipfire = map
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.hipfire_name.clone(), i))
            .collect();
        Self {
            gguf,
            config: &hp.config,
            map,
            by_hipfire,
            prefill: PrefillDequant::from_env(),
        }
    }

    pub fn prefill_dequant(&self) -> PrefillDequant {
        self.prefill
    }

    fn lookup(&self, hipfire_name: &str) -> HipResult<(&ExpectedTensor, &TensorInfo, &'a [u8])> {
        let e = self
            .by_hipfire
            .get(hipfire_name)
            .map(|&i| &self.map.tensors[i])
            .ok_or_else(|| {
                err(format!(
                    "load refused: '{hipfire_name}' is not a tensor of the dense qwen35 GGUF export"
                ))
            })?;
        if e.group == TensorGroup::Mtp {
            return Err(err(format!(
                "load refused: '{}' is an MTP-block tensor; GGUF-native serving loads the trunk only",
                e.gguf_name
            )));
        }
        let info = self.gguf.find_tensor(&e.gguf_name).ok_or_else(|| {
            err(format!(
                "load refused: GGUF is missing tensor '{}' (hipfire '{hipfire_name}')",
                e.gguf_name
            ))
        })?;
        let data = self.gguf.try_tensor_data(info).ok_or_else(|| {
            err(format!(
                "load refused: GGUF tensor '{}' lies outside the file; the download may be incomplete",
                e.gguf_name
            ))
        })?;
        Ok((e, info, data))
    }

    fn layer_name(layer: usize, rel: &str) -> String {
        format!("model.language_model.layers.{layer}.{rel}")
    }

    /// `(nk, r)` when the exporter re-ordered the value heads.
    fn tiling(&self) -> Option<(usize, usize)> {
        let c = self.config;
        (c.linear_num_key_heads != c.linear_num_value_heads).then(|| {
            (
                c.linear_num_key_heads,
                c.linear_num_value_heads / c.linear_num_key_heads,
            )
        })
    }

    fn row_perm_params(
        &self,
        order: VHeadOrder,
    ) -> HipResult<Option<(usize, usize, usize, usize)>> {
        // (skip_rows, block_rows, nk, r)
        let c = self.config;
        let Some((nk, r)) = self.tiling() else {
            return Ok(None);
        };
        if c.linear_num_value_heads % c.linear_num_key_heads != 0 {
            return Err(err(
                "load refused: linear_num_value_heads is not a multiple of linear_num_key_heads",
            ));
        }
        Ok(match order {
            VHeadOrder::Native | VHeadOrder::TiledCols { .. } => None,
            VHeadOrder::TiledRows { head_dim } => Some((0, head_dim, nk, r)),
            VHeadOrder::TiledVector => Some((0, 1, nk, r)),
            VHeadOrder::TiledQkvValueRows { head_dim } => Some((
                2 * c.linear_num_key_heads * c.linear_key_head_dim,
                head_dim,
                nk,
                r,
            )),
            VHeadOrder::TiledConvValueChannels { head_dim } => Some((
                2 * c.linear_num_key_heads * c.linear_key_head_dim,
                head_dim,
                nk,
                r,
            )),
        })
    }

    /// Upload a matrix tensor. Ggml block types keep their bytes (rows
    /// permuted per the exporter's head order; `out_proj` gets its K-chunk
    /// map); float matrices are widened to F32.
    fn upload_matrix(
        &self,
        gpu: &mut Gpu,
        hipfire_name: &str,
        m: usize,
        k: usize,
    ) -> HipResult<WeightTensor> {
        let (e, info, data) = self.lookup(hipfire_name)?;
        let rows = info.shape.get(1).copied().unwrap_or(1);
        let cols = info.shape.first().copied().unwrap_or(0);
        if (rows, cols) != (m, k) {
            return Err(err(format!(
                "load refused: GGUF tensor '{}' is {rows}x{cols}, the model needs {m}x{k}",
                e.gguf_name
            )));
        }
        let perm = self.row_perm_params(e.transform.v_order)?;
        if let Some(dtype) = ggml_type_dtype(info.dtype) {
            let row_bytes = info.dtype.row_bytes(cols).ok_or_else(|| {
                err(format!(
                    "load refused: GGUF tensor '{}' row of {cols} {} elements is not whole blocks",
                    e.gguf_name,
                    info.dtype.name()
                ))
            })?;
            let kmap = match e.transform.v_order {
                VHeadOrder::TiledCols { head_dim } if self.tiling().is_some() => {
                    if head_dim != GGML_KMAP_CHUNK || k % GGML_KMAP_CHUNK != 0 {
                        return Err(err(format!(
                            "load refused: GGUF tensor '{}' has value head dim {head_dim}; the \
                             K-chunk map of the head permutation supports {GGML_KMAP_CHUNK}",
                            e.gguf_name
                        )));
                    }
                    let (nk, r) = self.tiling().expect("checked");
                    Some(ggml_build_kmap(&ggml_tiled_to_grouped_heads(nk, r)))
                }
                _ => None,
            };
            let total = ggml_weight_buffer_bytes(dtype, m, k, kmap.is_some())
                .ok_or_else(|| err("load refused: ggml weight extent overflows"))?;
            let body: std::borrow::Cow<'_, [u8]> = match perm {
                None => std::borrow::Cow::Borrowed(data),
                Some((skip, block_rows, nk, r)) => std::borrow::Cow::Owned(permute_row_blocks(
                    data, row_bytes, skip, block_rows, nk, r,
                )),
            };
            let upload: std::borrow::Cow<'_, [u8]> = match &kmap {
                None => body,
                Some(map) => {
                    let off = ggml_kmap_offset(dtype, m, k).expect("checked");
                    let mut v = vec![0u8; total];
                    v[..body.len()].copy_from_slice(&body);
                    v[off..off + map.len()].copy_from_slice(map);
                    std::borrow::Cow::Owned(v)
                }
            };
            let upload: &[u8] = &upload;
            debug_assert_eq!(upload.len(), total);
            let buf = gpu.upload_raw(upload, &[upload.len()])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: dtype,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            });
        }
        // Float matrices: widen to F32 (no ggml block bytes to keep).
        let mut vals = f32_from_bytes(info.dtype, data)?;
        if let Some((skip, block_rows, nk, r)) = perm {
            vals = permute_row_blocks(&vals, cols, skip, block_rows, nk, r);
        }
        if matches!(e.transform.v_order, VHeadOrder::TiledCols { .. }) && self.tiling().is_some() {
            return Err(err(format!(
                "load refused: GGUF tensor '{}' is {} and needs a column head permutation, which \
                 is only supported for ggml block types",
                e.gguf_name,
                info.dtype.name()
            )));
        }
        let buf = gpu.upload_f32(&vals, &[m * k])?;
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

    fn read_f32_tensor(&self, hipfire_name: &str, n: usize) -> HipResult<Vec<f32>> {
        let (e, info, data) = self.lookup(hipfire_name)?;
        let mut vals = f32_from_bytes(info.dtype, data)?;
        if vals.len() != n {
            return Err(err(format!(
                "load refused: GGUF tensor '{}' holds {} values, the model needs {n}",
                e.gguf_name,
                vals.len()
            )));
        }
        let t: GgufTransform = e.transform;
        if t.a_log_neg_exp {
            for v in &mut vals {
                if !(*v < 0.0) {
                    return Err(err(format!(
                        "load refused: GGUF tensor '{}' (ssm_a = -exp(A_log)) holds {v}; it must be negative",
                        e.gguf_name
                    )));
                }
                *v = (-(*v as f64)).ln() as f32;
            }
        }
        if let Some((skip, block_rows, nk, r)) = self.row_perm_params(t.v_order)? {
            let row_len = match t.v_order {
                VHeadOrder::TiledConvValueChannels { .. } => self.config.conv_kernel_dim,
                _ => 1,
            };
            vals = permute_row_blocks(&vals, row_len, skip, block_rows, nk, r);
        }
        Ok(vals)
    }

    /// Bytes the prefill dequantisation region must hold: the largest single
    /// weight (prefill dequantises one weight at a time) at the prefill element
    /// size. The LM head is never bulk-dequantised.
    pub fn materialize_bytes(&self) -> usize {
        let c = self.config;
        let qkv = c.linear_num_key_heads * c.linear_key_head_dim * 2
            + c.linear_num_value_heads * c.linear_value_head_dim;
        let d_inner = c.linear_num_value_heads * c.linear_value_head_dim;
        let q_out = c.n_heads * c.head_dim * 2;
        let kv = c.n_kv_heads * c.head_dim;
        let largest = [
            c.hidden_dim * c.dim,
            qkv * c.dim,
            d_inner * c.dim,
            q_out * c.dim,
            kv * c.dim,
            c.dim * c.n_heads * c.head_dim,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        (largest * self.prefill.elem_bytes() + 255) & !255
    }
}

/// Per-layer backend feeding `layer_driver::load_layer`.
struct GgufBackend<'s, 'a> {
    src: &'s GgufSource<'a>,
    gpu: &'s mut Gpu,
    layer: usize,
}

impl WeightBackend for GgufBackend<'_, '_> {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let name = GgufSource::layer_name(self.layer, &format!("{rel}.weight"));
        self.src.upload_matrix(self.gpu, &name, m, k)
    }

    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let name = GgufSource::layer_name(self.layer, rel);
        let n: usize = shape.iter().product();
        // Stored with +1 folded in: exactly the weight the norm kernels take.
        let vals = self.src.read_f32_tensor(&name, n)?;
        self.gpu.upload_f32(&vals, shape)
    }

    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = GgufSource::layer_name(self.layer, rel);
        let vals = self.src.read_f32_tensor(&name, n)?;
        self.gpu.upload_f32(&vals, &[n])
    }

    fn bias(&mut self, rel: &str, _n: usize) -> HipResult<GpuTensor> {
        Err(err(format!(
            "load refused: dense qwen35 GGUF export has no bias tensor '{rel}'"
        )))
    }

    fn free_tensor(&mut self, tensor: GpuTensor) {
        let _ = self.gpu.free_tensor(tensor);
    }
}

impl WeightSource for GgufSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices != 1 {
            return Err(err(
                "load refused: GGUF-native serving runs on one device (no pipeline or tensor split)",
            ));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd...");
        // Persistent scratch of the ggml route, before any KV admission sizes
        // against free memory.
        gpu.ggml_ensure_scratch()?;
        let a8_emulation = hipfire_config::developer_var(A8_EMU_ENV).is_ok_and(|v| v == "1");
        gpu.ggml_set_prefill_mode(rdna_compute::ggml_ops::GgmlPrefill {
            codec_only: self.prefill == PrefillDequant::Bf16RoundedF32,
            a8_emulation,
        });
        gpu.ggml_ensure_materialize(self.materialize_bytes())?;
        eprintln!(
            "  [gguf] prefill route: {}{}",
            match self.prefill {
                PrefillDequant::F16 => "dequant->F16 GEMM (A16 activations)",
                PrefillDequant::Bf16RoundedF32 =>
                    "codec-only tier: dequant->BF16-rounded weights, hi+lo F16 WMMA GEMM (no activation quantisation)",
            },
            if a8_emulation { ", q8_1 activation emulation ON" } else { "" }
        );
        let name = "model.language_model.embed_tokens.weight";
        let (e, info, data) = self.lookup(name)?;
        let Some(dtype) = ggml_type_dtype(info.dtype) else {
            return Err(err(format!(
                "load refused: GGUF tensor '{}' is {}; the native embedding gather serves the ggml \
                 block types (IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0)",
                e.gguf_name,
                info.dtype.name()
            )));
        };
        if info.shape != [self.config.dim, self.config.vocab_size] {
            return Err(err(format!(
                "load refused: GGUF tensor '{}' has shape {:?}, expected [{}, {}]",
                e.gguf_name, info.shape, self.config.dim, self.config.vocab_size
            )));
        }
        let buf = gpu.upload_raw(data, &[data.len()])?;
        Ok((buf, EmbeddingFormat::Ggml(dtype)))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        let vals = self.read_f32_tensor("model.language_model.norm.weight", self.config.dim)?;
        gpu.upload_f32(&vals, &[self.config.dim])
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let c = self.config;
        if self.by_hipfire.get("lm_head.weight").is_some_and(|&i| {
            self.gguf
                .find_tensor(&self.map.tensors[i].gguf_name)
                .is_some()
        }) {
            eprintln!("  loading output (separate lm_head)...");
            return Ok((
                self.upload_matrix(gpu, "lm_head.weight", c.vocab_size, c.dim)?,
                false,
            ));
        }
        if !can_alias {
            return Err(err(
                "load refused: tied GGUF embeddings need the single-device alias path",
            ));
        }
        eprintln!("  loading output (tied embeddings, aliased)...");
        Ok((
            hipfire_runtime::weight_backend::tied_lm_head_alias(
                embd,
                embd_fmt,
                c.vocab_size,
                c.dim,
            ),
            true,
        ))
    }

    fn read_layer(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        residency: LayerResidency,
    ) -> HipResult<LayerWeights> {
        if residency != LayerResidency::DEVICE {
            return Err(err(
                "load refused: GGUF-native serving keeps every layer on the device (no offload tier)",
            ));
        }
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?})...",
            self.config.n_layers, self.config.layer_types[layer_idx]
        );
        let mut backend = GgufBackend {
            src: self,
            gpu,
            layer: layer_idx,
        };
        crate::layer_driver::load_layer(&mut backend, self.config, layer_idx, |_, _, _| {
            Err(err(
                "load refused: GGUF-native serving covers the dense export only",
            ))
        })
    }

    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer) {
        layer.free_gpu(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_head_inverts_exporter_reorder() {
        let (nk, r) = (16usize, 3usize);
        // grouped h = kh*r + j sits at stored position g = j*nk + kh.
        for kh in 0..nk {
            for j in 0..r {
                assert_eq!(stored_head(kh * r + j, nk, r), j * nk + kh);
            }
        }
        let map = ggml_tiled_to_grouped_heads(nk, r);
        for h in 0..nk * r {
            assert_eq!(map[stored_head(h, nk, r)] as usize, h);
        }
    }

    #[test]
    fn row_block_permutation_moves_whole_blocks() {
        // 2 key heads, 2 values per key head, 2 rows per head, 3 units per row.
        let (nk, r, rows, row_len) = (2usize, 2usize, 2usize, 3usize);
        let skip = 1usize;
        let n_rows = skip + nk * r * rows;
        let data: Vec<u32> = (0..n_rows * row_len).map(|i| i as u32).collect();
        let out = permute_row_blocks(&data, row_len, skip, rows, nk, r);
        assert_eq!(&out[..skip * row_len], &data[..skip * row_len]);
        // HF order (kh0 j0, kh0 j1, kh1 j0, kh1 j1) reads stored blocks (0, 2, 1, 3).
        for (h, g) in [(0usize, 0usize), (1, 2), (2, 1), (3, 3)] {
            let o = (skip + h * rows) * row_len;
            let s = (skip + g * rows) * row_len;
            assert_eq!(
                &out[o..o + rows * row_len],
                &data[s..s + rows * row_len],
                "h={h}"
            );
        }
    }

    #[test]
    fn ggml_type_table_is_the_tier_one_set() {
        for (t, d) in [
            (GgmlType::Q4K, DType::GgmlQ4K),
            (GgmlType::Q5K, DType::GgmlQ5K),
            (GgmlType::Q6K, DType::GgmlQ6K),
            (GgmlType::Q8_0, DType::GgmlQ8_0),
            (GgmlType::IQ4XS, DType::GgmlIQ4XS),
        ] {
            assert_eq!(ggml_type_dtype(t), Some(d));
        }
        assert_eq!(ggml_type_dtype(GgmlType::Q4_0), None);
        assert_eq!(ggml_type_dtype(GgmlType::IQ3S), None);
        assert_eq!(ggml_type_dtype(GgmlType::F16), None);
    }
}
