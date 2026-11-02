// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM2-VL vision: SigLIP2-NaFlex tower + multi-modal projector.
//!
//! GPU kernels (all pre-existing, gfx1101-validated via qwen35-vl):
//! `gemm_f16[_wmma_mb8]`, `layernorm_batched`, `vit_attention_f32`,
//! `gelu_tanh_f32` (tower MLP), `add_inplace_f32`, `bias_add_f32`,
//! `transpose_f32`. The projector's exact-erf GELU runs on host between
//! two GPU linears (`libm::erff`; ACT2FN["gelu"] is torch.erf, NOT tanh).
//!
//! SigLIP2 has separate q/k/v projections; they are concatenated into a
//! single `[3h, h]` F16 weight + `[3h]` bias at LOAD time so the forward
//! can emit the packed `[n, 3h]` q|k|v buffer `vit_attention_f32` consumes
//! (one gemm per layer instead of three, and no strided pack pass). F16 and
//! F32 artifact q/k/v share that path (F32 is narrowed to F16 at concat).

use crate::config::VisionConfig;
use crate::image::{Prepared, SubImage};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use rdna_compute::{DType, Gpu, GpuTensor};

const QT_F16: u8 = 1;
const QT_F32: u8 = 2;

// ─── GPU-side weights ────────────────────────────────────────────────────────

pub struct VisionLayerWeights {
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    /// Fused `[3h, h]` of stacked q/k/v row-major F16.
    pub qkv_w: GpuTensor,
    pub qkv_b: GpuTensor,
    pub proj_w: GpuTensor,
    pub proj_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub fc1_w: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor,
    pub fc2_b: GpuTensor,
}

impl VisionLayerWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.norm1_w,
            self.norm1_b,
            self.qkv_w,
            self.qkv_b,
            self.proj_w,
            self.proj_b,
            self.norm2_w,
            self.norm2_b,
            self.fc1_w,
            self.fc1_b,
            self.fc2_w,
            self.fc2_b,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

pub struct VisionWeights {
    /// Patch embedding Linear rows `[1152, 768]` normalized to `(dy,dx,c)`
    /// input layout. Handles both serializations seen in the wild
    /// (Linear `[out, ps·ps·C]` or Conv `[out, C, ps, ps]`) per narrow-spec R1.
    pub patch_embed_w: GpuTensor,
    pub patch_embed_b: GpuTensor,
    /// Learned position table `[num_position_embeddings · hidden]` on CPU:
    /// every sub-image resizes a different slice out of it (small buffer).
    pub pos_embed: Vec<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub post_ln_w: GpuTensor,
    pub post_ln_b: GpuTensor,
    // ── projector ──
    /// `[2048, 4608]` F16 — input vector is the pixel-unshuffle block from
    /// [`pixel_unshuffle_tokens`] (columns-pair-first channel interleave).
    pub proj1_w: GpuTensor,
    pub proj1_b: GpuTensor,
    /// `[2048, 2048]` F16.
    pub proj2_w: GpuTensor,
    pub proj2_b: GpuTensor,
}

impl VisionWeights {
    /// Free every GPU buffer (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embed_w);
        let _ = gpu.free_tensor(self.patch_embed_b);
        for l in self.layers {
            l.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(self.post_ln_w);
        let _ = gpu.free_tensor(self.post_ln_b);
        let _ = gpu.free_tensor(self.proj1_w);
        let _ = gpu.free_tensor(self.proj1_b);
        let _ = gpu.free_tensor(self.proj2_w);
        let _ = gpu.free_tensor(self.proj2_b);
    }
}

// ─── Artifact validation (CPU; unit-tested) ─────────────────────────────────

fn lookup<'a>(hfq: &'a HfqFile, name: &str) -> Result<(&'a HfqTensorInfo, &'a [u8]), String> {
    hfq.tensor_data(name)
        .ok_or_else(|| format!("vision tensor not found: {name}"))
}

/// Validate dtype (F16/F32), rank, exact dims, and element/byte counts.
/// Returns the element count on success.
fn validate_dense(
    name: &str,
    quant_type: u8,
    shape: &[u32],
    nbytes: usize,
    expected: &[usize],
) -> Result<usize, String> {
    if shape.len() != expected.len() {
        return Err(format!(
            "{name}: rank {} (want {})",
            shape.len(),
            expected.len()
        ));
    }
    for (i, (&got, &want)) in shape.iter().zip(expected.iter()).enumerate() {
        if got as usize != want {
            return Err(format!("{name}: dim[{i}]={got} (want {want})"));
        }
    }
    let elems: usize = expected.iter().copied().product();
    let width = match quant_type {
        QT_F16 => 2usize,
        QT_F32 => 4usize,
        other => {
            return Err(format!(
                "{name}: unsupported vision quant_type={other} (expected F16=1 or F32=2)"
            ));
        }
    };
    let want_bytes = elems
        .checked_mul(width)
        .ok_or_else(|| format!("{name}: element/byte count overflow ({elems} × {width})"))?;
    if nbytes != want_bytes {
        return Err(format!(
            "{name}: {nbytes} bytes for {elems} elems (want {want_bytes})"
        ));
    }
    Ok(elems)
}

fn decode_f32(quant_type: u8, data: &[u8]) -> Vec<f32> {
    match quant_type {
        QT_F16 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        QT_F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => Vec::new(),
    }
}

/// Append one dense matrix as F16 bytes. F32 inputs are narrowed; F16 is copied.
fn append_f16_weight(
    dst: &mut Vec<u8>,
    name: &str,
    quant_type: u8,
    shape: &[u32],
    data: &[u8],
    expected: &[usize],
) -> Result<(), String> {
    let n = validate_dense(name, quant_type, shape, data.len(), expected)?;
    match quant_type {
        QT_F16 => dst.extend_from_slice(data),
        QT_F32 => {
            for c in data.chunks_exact(4).take(n) {
                dst.extend_from_slice(
                    &f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes(),
                );
            }
        }
        _ => unreachable!("validate_dense admits only F16/F32"),
    }
    Ok(())
}

fn fold_conv_patch_rows(raw: &[f32], out: usize, c_n: usize, kh: usize, kw: usize) -> Vec<f32> {
    let in_dim = c_n * kh * kw;
    let mut folded = vec![0.0f32; raw.len()];
    for o in 0..out {
        for dy in 0..kh {
            for dx in 0..kw {
                for c in 0..c_n {
                    folded[o * in_dim + (dy * kw + dx) * c_n + c] =
                        raw[o * in_dim + c * kh * kw + dy * kw + dx];
                }
            }
        }
    }
    folded
}

fn patch_weight_rows_from(
    name: &str,
    quant_type: u8,
    shape: &[u32],
    data: &[u8],
    cfg: &VisionConfig,
) -> Result<(Vec<f32>, usize, usize), String> {
    let hidden = cfg.hidden_size;
    let patch_in = cfg.patch_size * cfg.patch_size * cfg.num_channels;
    let expected: Vec<usize> = match shape.len() {
        2 => vec![hidden, patch_in],
        4 => vec![hidden, cfg.num_channels, cfg.patch_size, cfg.patch_size],
        other => {
            return Err(format!(
                "{name}: unexpected rank {other} (want Linear [out,in] or Conv [out,C,kh,kw]) — \
                 refusing to guess the layout"
            ));
        }
    };
    let n = validate_dense(name, quant_type, shape, data.len(), &expected)?;
    let raw = decode_f32(quant_type, data);
    if raw.len() != n {
        return Err(format!("{name}: decoded {} elems, want {n}", raw.len()));
    }
    match shape.len() {
        2 => Ok((raw, hidden, patch_in)),
        4 => Ok((
            fold_conv_patch_rows(
                &raw,
                hidden,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            hidden,
            patch_in,
        )),
        _ => unreachable!("rank already validated"),
    }
}

// ─── Tensor loading helpers (F16/F32-in-artifact → F32 or F16 GPU) ───────────

fn load_f32_cpu(hfq: &HfqFile, name: &str, expected: &[usize]) -> Result<Vec<f32>, String> {
    let (info, data) = lookup(hfq, name)?;
    let n = validate_dense(name, info.quant_type, &info.shape, data.len(), expected)?;
    let vals = decode_f32(info.quant_type, data);
    if vals.len() != n {
        return Err(format!("{name}: decoded {} elems, want {n}", vals.len()));
    }
    Ok(vals)
}

fn load_f32_gpu(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected: &[usize],
) -> Result<GpuTensor, String> {
    let vals = load_f32_cpu(hfq, name, expected)?;
    gpu.upload_f32(&vals, &[vals.len()])
        .map_err(|e| format!("lfm2-vl vision upload {name}: {e:?}"))
}

fn load_f16_gpu(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected: &[usize],
) -> Result<GpuTensor, String> {
    let (info, data) = lookup(hfq, name)?;
    let n = validate_dense(name, info.quant_type, &info.shape, data.len(), expected)?;
    match info.quant_type {
        QT_F16 => gpu
            .upload_raw(data, &[n])
            .map_err(|e| format!("lfm2-vl vision upload {name}: {e:?}")),
        QT_F32 => {
            let mut f16_bytes = Vec::with_capacity(n * 2);
            append_f16_weight(&mut f16_bytes, name, QT_F32, &info.shape, data, expected)?;
            gpu.upload_raw(&f16_bytes, &[n])
                .map_err(|e| format!("lfm2-vl vision upload {name}: {e:?}"))
        }
        _ => unreachable!("validate_dense admits only F16/F32"),
    }
}

/// Normalize the patch-embedding weight to `[out, ps·ps·C]` rows ordered for
/// input vectors laid out `(dy, dx, c)` (HF convert_image_to_patches order).
///
/// Artifact serializations handled:
/// - Linear `[1152, 768]`: already aligned with `(dy,dx,c)` — verbatim.
/// - Conv `[1152, C, kh, kw]`: kernel dims reorder to `(dy,dx,c)`.
fn patch_weight_rows(
    hfq: &HfqFile,
    cfg: &VisionConfig,
) -> Result<(Vec<f32>, usize, usize), String> {
    const NAME: &str = "model.vision_tower.vision_model.embeddings.patch_embedding.weight";
    let (info, data) = lookup(hfq, NAME)?;
    patch_weight_rows_from(NAME, info.quant_type, &info.shape, data, cfg)
}

fn concat_qkv(hfq: &HfqFile, i: usize, hidden: usize) -> Result<(Vec<u8>, Vec<f32>), String> {
    // Stack independent q/k/v weights into one [3h, h] F16 buffer so the
    // forward's single fused gemm produces the q|k|v-packed layout
    // `vit_attention_f32` expects. Biases concatenate likewise. F32 q/k/v
    // are narrowed to F16 here — same contract as `load_f16_gpu`.
    let pfx = format!("model.vision_tower.vision_model.encoder.layers.{i}.self_attn");
    let mut w = Vec::with_capacity(6 * hidden * hidden);
    let mut b = Vec::with_capacity(3 * hidden);
    for part in ["q_proj", "k_proj", "v_proj"] {
        let wname = format!("{pfx}.{part}.weight");
        let (info, data) = lookup(hfq, &wname)?;
        append_f16_weight(
            &mut w,
            &wname,
            info.quant_type,
            &info.shape,
            data,
            &[hidden, hidden],
        )?;
        let bname = format!("{pfx}.{part}.bias");
        b.extend(load_f32_cpu(hfq, &bname, &[hidden])?);
    }
    Ok((w, b))
}

#[derive(Default)]
struct LayerLoadGuard {
    norm1_w: Option<GpuTensor>,
    norm1_b: Option<GpuTensor>,
    qkv_w: Option<GpuTensor>,
    qkv_b: Option<GpuTensor>,
    proj_w: Option<GpuTensor>,
    proj_b: Option<GpuTensor>,
    norm2_w: Option<GpuTensor>,
    norm2_b: Option<GpuTensor>,
    fc1_w: Option<GpuTensor>,
    fc1_b: Option<GpuTensor>,
    fc2_w: Option<GpuTensor>,
    fc2_b: Option<GpuTensor>,
}

impl LayerLoadGuard {
    fn cleanup(&mut self, gpu: &mut Gpu) {
        for slot in [
            &mut self.norm1_w,
            &mut self.norm1_b,
            &mut self.qkv_w,
            &mut self.qkv_b,
            &mut self.proj_w,
            &mut self.proj_b,
            &mut self.norm2_w,
            &mut self.norm2_b,
            &mut self.fc1_w,
            &mut self.fc1_b,
            &mut self.fc2_w,
            &mut self.fc2_b,
        ] {
            if let Some(t) = slot.take() {
                let _ = gpu.free_tensor(t);
            }
        }
    }

    fn take(mut self) -> VisionLayerWeights {
        VisionLayerWeights {
            norm1_w: self.norm1_w.take().unwrap(),
            norm1_b: self.norm1_b.take().unwrap(),
            qkv_w: self.qkv_w.take().unwrap(),
            qkv_b: self.qkv_b.take().unwrap(),
            proj_w: self.proj_w.take().unwrap(),
            proj_b: self.proj_b.take().unwrap(),
            norm2_w: self.norm2_w.take().unwrap(),
            norm2_b: self.norm2_b.take().unwrap(),
            fc1_w: self.fc1_w.take().unwrap(),
            fc1_b: self.fc1_b.take().unwrap(),
            fc2_w: self.fc2_w.take().unwrap(),
            fc2_b: self.fc2_b.take().unwrap(),
        }
    }
}

#[derive(Default)]
struct VisionLoadGuard {
    patch_embed_w: Option<GpuTensor>,
    patch_embed_b: Option<GpuTensor>,
    layers: Vec<VisionLayerWeights>,
    post_ln_w: Option<GpuTensor>,
    post_ln_b: Option<GpuTensor>,
    proj1_w: Option<GpuTensor>,
    proj1_b: Option<GpuTensor>,
    proj2_w: Option<GpuTensor>,
    proj2_b: Option<GpuTensor>,
}

impl VisionLoadGuard {
    fn cleanup(&mut self, gpu: &mut Gpu) {
        for slot in [
            &mut self.patch_embed_w,
            &mut self.patch_embed_b,
            &mut self.post_ln_w,
            &mut self.post_ln_b,
            &mut self.proj1_w,
            &mut self.proj1_b,
            &mut self.proj2_w,
            &mut self.proj2_b,
        ] {
            if let Some(t) = slot.take() {
                let _ = gpu.free_tensor(t);
            }
        }
        for l in self.layers.drain(..) {
            l.free_gpu(gpu);
        }
    }

    fn finish(mut self, pos_embed: Vec<f32>) -> VisionWeights {
        VisionWeights {
            patch_embed_w: self.patch_embed_w.take().unwrap(),
            patch_embed_b: self.patch_embed_b.take().unwrap(),
            pos_embed,
            layers: std::mem::take(&mut self.layers),
            post_ln_w: self.post_ln_w.take().unwrap(),
            post_ln_b: self.post_ln_b.take().unwrap(),
            proj1_w: self.proj1_w.take().unwrap(),
            proj1_b: self.proj1_b.take().unwrap(),
            proj2_w: self.proj2_w.take().unwrap(),
            proj2_b: self.proj2_b.take().unwrap(),
        }
    }
}

fn load_one_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &VisionConfig,
    i: usize,
) -> Result<VisionLayerWeights, String> {
    let h = cfg.hidden_size;
    let p = format!("model.vision_tower.vision_model.encoder.layers.{i}");
    let mut g = LayerLoadGuard::default();
    let result = (|| {
        let (qkv_w, qkv_b) = concat_qkv(hfq, i, h)?;
        g.norm1_w = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.layer_norm1.weight"),
            &[h],
        )?);
        g.norm1_b = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.layer_norm1.bias"),
            &[h],
        )?);
        g.qkv_w = Some(
            gpu.upload_raw(&qkv_w, &[3 * h * h])
                .map_err(|e| format!("lfm2-vl vision upload {p} qkv_w: {e:?}"))?,
        );
        g.qkv_b = Some(
            gpu.upload_f32(&qkv_b, &[3 * h])
                .map_err(|e| format!("lfm2-vl vision upload {p} qkv_b: {e:?}"))?,
        );
        g.proj_w = Some(load_f16_gpu(
            hfq,
            gpu,
            &format!("{p}.self_attn.out_proj.weight"),
            &[h, h],
        )?);
        g.proj_b = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.self_attn.out_proj.bias"),
            &[h],
        )?);
        g.norm2_w = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.layer_norm2.weight"),
            &[h],
        )?);
        g.norm2_b = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.layer_norm2.bias"),
            &[h],
        )?);
        g.fc1_w = Some(load_f16_gpu(
            hfq,
            gpu,
            &format!("{p}.mlp.fc1.weight"),
            &[cfg.mlp_dim, h],
        )?);
        g.fc1_b = Some(load_f32_gpu(
            hfq,
            gpu,
            &format!("{p}.mlp.fc1.bias"),
            &[cfg.mlp_dim],
        )?);
        g.fc2_w = Some(load_f16_gpu(
            hfq,
            gpu,
            &format!("{p}.mlp.fc2.weight"),
            &[h, cfg.mlp_dim],
        )?);
        g.fc2_b = Some(load_f32_gpu(hfq, gpu, &format!("{p}.mlp.fc2.bias"), &[h])?);
        Ok(())
    })();
    match result {
        Ok(()) => Ok(g.take()),
        Err(e) => {
            g.cleanup(gpu);
            Err(e)
        }
    }
}

pub fn load_vision_weights(
    hfq: &HfqFile,
    cfg: &VisionConfig,
    gpu: &mut Gpu,
) -> Result<VisionWeights, String> {
    let h = cfg.hidden_size;
    if cfg.num_heads == 0 || h != cfg.num_heads * cfg.head_dim {
        return Err(format!(
            "lfm2-vl vision: hidden_size={h} is not num_heads={} × head_dim={}",
            cfg.num_heads, cfg.head_dim
        ));
    }
    if cfg.downsample_factor == 0 {
        return Err("lfm2-vl vision: downsample_factor must be ≥ 1".into());
    }
    let k_side = (cfg.num_position_embeddings as f64).sqrt() as usize;
    if k_side == 0 || k_side * k_side != cfg.num_position_embeddings {
        return Err(format!(
            "lfm2-vl vision: num_position_embeddings={} is not a square table side",
            cfg.num_position_embeddings
        ));
    }

    match gpu.arch.as_str() {
        "gfx1100" | "gfx1101" | "gfx1102" => {}
        other => eprintln!(
            "  ⚠ vision tower not yet validated on {other}; results may differ from \
             the RDNA3-wave32 reference",
        ),
    }

    eprintln!("  loading LFM2-VL vision tower (GPU)...");
    let mut guard = VisionLoadGuard::default();
    let loaded = (|| {
        let (pw_rows, out_rows, _in_dim) = patch_weight_rows(hfq, cfg)?;
        let pw_u16: Vec<u8> = pw_rows
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        // Upload shape stays `[out_rows]` — GEMM dims are passed explicitly.
        guard.patch_embed_w = Some(
            gpu.upload_raw(&pw_u16, &[out_rows])
                .map_err(|e| format!("lfm2-vl vision upload patch_embed.weight: {e:?}"))?,
        );
        guard.patch_embed_b = Some(load_f32_gpu(
            hfq,
            gpu,
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
            &[h],
        )?);
        let pos_embed = load_f32_cpu(
            hfq,
            "model.vision_tower.vision_model.embeddings.position_embedding.weight",
            &[cfg.num_position_embeddings, h],
        )?;

        for i in 0..cfg.num_layers {
            if i % 9 == 0 {
                eprintln!("  loading vision block {i}/{}...", cfg.num_layers);
            }
            match load_one_layer(hfq, gpu, cfg, i) {
                Ok(layer) => guard.layers.push(layer),
                Err(e) => return Err(e),
            }
        }

        eprintln!("  loading post_layernorm + projector...");
        let ds = cfg.downsample_factor;
        let proj1_in = h * ds * ds;
        guard.post_ln_w = Some(load_f32_gpu(
            hfq,
            gpu,
            "model.vision_tower.vision_model.post_layernorm.weight",
            &[h],
        )?);
        guard.post_ln_b = Some(load_f32_gpu(
            hfq,
            gpu,
            "model.vision_tower.vision_model.post_layernorm.bias",
            &[h],
        )?);
        guard.proj1_w = Some(load_f16_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_1.weight",
            &[cfg.projector_hidden_size, proj1_in],
        )?);
        guard.proj1_b = Some(load_f32_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_1.bias",
            &[cfg.projector_hidden_size],
        )?);
        guard.proj2_w = Some(load_f16_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_2.weight",
            &[cfg.out_hidden_size, cfg.projector_hidden_size],
        )?);
        guard.proj2_b = Some(load_f32_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_2.bias",
            &[cfg.out_hidden_size],
        )?);
        Ok(pos_embed)
    })();
    match loaded {
        Ok(pos_embed) => Ok(guard.finish(pos_embed)),
        Err(e) => {
            guard.cleanup(gpu);
            Err(e)
        }
    }
}

// ─── CPU per-image precomputes (exact-index, unit-tested) ───────────────────

/// Bilinear resize of the learned `(K,K,D)` position table to `(gh, gw)`,
/// torch semantics: `F.interpolate(mode="bilinear", align_corners=False,
/// antialias=True)`. Upscale axes are plain two-tap bilinear sampling
/// (antialias is a no-op there); downscale axes widen to a normalized
/// triangle filter over the source footprint, matching torch's antialiased
/// path. Narrow-spec §2.1/R3 documents this as the accepted approximation
/// point (resize-filter residual, below model sensitivity).
pub fn resize_pos_embed(
    table: &[f32],
    k_side: usize,
    dim: usize,
    gh: usize,
    gw: usize,
) -> Vec<f32> {
    struct Taps {
        idx: Vec<usize>,
        w: Vec<f64>,
    }
    fn axis_taps(dst_len: usize, src_len: usize) -> Vec<Taps> {
        let scale = src_len as f64 / dst_len as f64;
        let aa = scale > 1.0;
        (0..dst_len)
            .map(|i| {
                // align_corners=False sample center
                let center = (i as f64 + 0.5) * scale - 0.5;
                if !aa {
                    // torch bilinear: indices are clamped but the split
                    // fraction is taken against the UNCLAMPED floor, so
                    // off-table centers replicate the border value.
                    let l0 = center.floor();
                    let frac = center - l0;
                    let lo_i = (l0 as i64).clamp(0, src_len as i64 - 1);
                    let hi_i = ((l0 as i64) + 1).clamp(0, src_len as i64 - 1);
                    Taps {
                        idx: vec![lo_i as usize, hi_i as usize],
                        w: vec![1.0 - frac, frac],
                    }
                } else {
                    let support = scale;
                    let lo = ((center - support).floor().max(0.0)) as usize;
                    let hi = ((center + support).ceil()).min((src_len - 1) as f64) as usize;
                    let mut idx = Vec::new();
                    let mut w = Vec::new();
                    let inv = 1.0 / scale;
                    for s in lo..=hi {
                        let d = (s as f64 - center).abs();
                        let t = (1.0 - d * inv).max(0.0);
                        idx.push(s);
                        w.push(t);
                    }
                    let sum: f64 = w.iter().sum();
                    if sum > 0.0 {
                        for v in w.iter_mut() {
                            *v /= sum;
                        }
                    }
                    Taps { idx, w }
                }
            })
            .collect()
    }

    let th = axis_taps(gh, k_side);
    let tw = axis_taps(gw, k_side);

    let mut out = vec![0.0f32; gh * gw * dim];
    for oy in 0..gh {
        for ox in 0..gw {
            let mut acc = vec![0.0f64; dim];
            let mut total = 0.0f64;
            for (&sy, &wy) in th[oy].idx.iter().zip(&th[oy].w) {
                for (&sx, &wx) in tw[ox].idx.iter().zip(&tw[ox].w) {
                    let wt = wy * wx;
                    if wt == 0.0 {
                        continue;
                    }
                    let base = (sy * k_side + sx) * dim;
                    for d in 0..dim {
                        acc[d] += (table[base + d] as f64) * wt;
                    }
                    total += wt;
                }
            }
            let norm = if total > 0.0 { total } else { 1.0 };
            let dst = (oy * gw + ox) * dim;
            for d in 0..dim {
                out[dst + d] = (acc[d] / norm) as f32;
            }
        }
    }
    out
}

/// HF `Lfm2VlMultiModalProjector::pixel_unshuffle(f=2)` on features arranged
/// `(gh, gw, channels)` (patch grid after unpad). Token vector component
/// `t = di·(2·C) + dj·C + c` reads feature patch `(2br+di, 2bc+dj)[c]` —
/// i.e. columns pair FIRST (dj stride C), rows second (di stride 2C):
/// replicated verbatim from the pinned HF ops including their swapped dim
/// naming. `factor` is 2 for every published LFM2-VL checkpoint; other even
/// factors fall back to a generic strided formulation.
pub fn pixel_unshuffle_tokens(
    feat: &[f32],
    gh: usize,
    gw: usize,
    ch: usize,
    factor: usize,
) -> Vec<f32> {
    let blocks_y = gh / factor;
    let blocks_x = gw / factor;
    let out_ch = ch * factor * factor;
    let mut out = vec![0.0f32; blocks_y * blocks_x * out_ch];
    // Generic loop derived from HF's reshape/permute chain: token vector is
    // di-major, then dj, then channel (see narrow spec §1.4 derivation).
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            for di in 0..factor {
                for dj in 0..factor {
                    let src_row = by * factor + di;
                    let src_col = bx * factor + dj;
                    let src = (src_row * gw + src_col) * ch;
                    // column offset first: t = di*(ch*f) + dj*ch   (f=2 ⇒ verified
                    // against the reshape trace); generalize stride order:
                    let dst_off = (by * blocks_x + bx) * out_ch;
                    for c in 0..ch {
                        // di-major then dj then c for any factor ≥ 2
                        out[dst_off + (di * factor * ch) + (dj * ch) + c] = feat[src + c];
                    }
                }
            }
        }
    }
    out
}

/// Exact-GELU (`x·Φ(x)` with erf Φ) used by the projector between linear_1
/// and linear_2 — ACT2FN["gelu"], distinct from the tower's tanh approx.
pub fn gelu_exact_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x *= 0.5 * (1.0 + libm::erff(*x / std::f32::consts::SQRT_2));
    }
}

// ─── GPU forward ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct ForwardScratch {
    live: Vec<Option<GpuTensor>>,
}

impl ForwardScratch {
    fn hold(&mut self, t: GpuTensor) -> usize {
        self.live.push(Some(t));
        self.live.len() - 1
    }

    fn t(&self, i: usize) -> Result<&GpuTensor, String> {
        self.live
            .get(i)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| "lfm2-vl vision: scratch slot empty".to_string())
    }

    fn release(&mut self, gpu: &mut Gpu, i: usize) -> Result<(), String> {
        match self.live.get_mut(i).and_then(Option::take) {
            Some(t) => gpu.free_tensor(t).map_err(ehip("free scratch")),
            None => Ok(()),
        }
    }

    fn drop_all(&mut self, gpu: &mut Gpu) {
        for slot in &mut self.live {
            if let Some(t) = slot.take() {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

/// Vision linear Y[n, out] = W_f16[out, in] @ X[n, in]^T + bias.
/// Same routing as qwen35-vl: WMMA row-major writer on wave32 arches,
/// naive gemm + transpose elsewhere.
fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: Option<&GpuTensor>,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> Result<GpuTensor, String> {
    let y = gpu
        .alloc_tensor(&[n * out_dim], DType::F32)
        .map_err(ehip("linear alloc"))?;
    let inner = (|| -> Result<(), String> {
        if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
            gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)
                .map_err(ehip("gemm_wmma"))?;
        } else {
            let yt = gpu
                .alloc_tensor(&[out_dim * n], DType::F32)
                .map_err(ehip("linear yt"))?;
            let r: Result<(), String> = (|| -> Result<(), String> {
                gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)
                    .map_err(ehip("gemm"))?;
                gpu.transpose_f32(&yt, &y, out_dim, n)
                    .map_err(ehip("transpose"))?;
                Ok(())
            })();
            let free_yt = gpu.free_tensor(yt);
            r?;
            free_yt.map_err(ehip("free yt"))?;
        }
        if let Some(b) = bias {
            gpu.bias_add_f32(&y, b, n, out_dim).map_err(ehip("bias"))?;
        }
        Ok(())
    })();
    match inner {
        Ok(()) => Ok(y),
        Err(e) => {
            let _ = gpu.free_tensor(y);
            Err(e)
        }
    }
}

/// Encode one request image into projected text-space tokens, concatenated
/// across its sub-images (tiles row-major then thumbnail). Row count equals
/// [`Prepared::total_tokens`]; failure to match fails loud BEFORE splicing.
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &VisionConfig,
    prepared: &Prepared,
) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let heads = cfg.num_heads;
    let head_dim = cfg.head_dim;
    if heads == 0 || h != heads * head_dim {
        return Err(format!(
            "lfm2-vl vision: hidden_size={h} is not num_heads={heads} × head_dim={head_dim}"
        ));
    }
    let k_side = (cfg.num_position_embeddings as f64).sqrt() as usize;
    if k_side == 0 || k_side * k_side != cfg.num_position_embeddings {
        return Err(format!(
            "lfm2-vl vision: num_position_embeddings={} is not a square table side",
            cfg.num_position_embeddings
        ));
    }

    let all_tokens: usize = prepared.total_tokens(cfg);
    let mut out = Vec::with_capacity(all_tokens * cfg.out_hidden_size);

    let t0 = std::time::Instant::now();
    for sub in &prepared.sub_images {
        out.extend(tower_and_project_sub_image(
            gpu, weights, cfg, sub, h, heads, head_dim, k_side,
        )?);
    }
    eprintln!(
        "  vision done: {} sub-images, {} tokens × {} dims ({:.2}s)",
        prepared.sub_images.len(),
        all_tokens,
        cfg.out_hidden_size,
        t0.elapsed().as_secs_f32(),
    );
    Ok(out)
}

fn tower_and_project_sub_image(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &VisionConfig,
    sub: &SubImage,
    h: usize,
    heads: usize,
    head_dim: usize,
    k_side: usize,
) -> Result<Vec<f32>, String> {
    let mut scratch = ForwardScratch::default();
    let result = tower_and_project_sub_image_inner(
        gpu,
        weights,
        cfg,
        sub,
        h,
        heads,
        head_dim,
        k_side,
        &mut scratch,
    );
    scratch.drop_all(gpu);
    result
}

fn tower_and_project_sub_image_inner(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &VisionConfig,
    sub: &SubImage,
    h: usize,
    heads: usize,
    head_dim: usize,
    k_side: usize,
    scratch: &mut ForwardScratch,
) -> Result<Vec<f32>, String> {
    let patches = sub.patches(cfg);
    let n = sub.gh(cfg) * sub.gw(cfg);
    let patch_dim = cfg.patch_size * cfg.patch_size * cfg.num_channels;
    if n == 0 || patches.len() != n * patch_dim {
        return Err(format!(
            "lfm2-vl vision: sub-image patches {} for grid {n} × {patch_dim}",
            patches.len()
        ));
    }
    let eps = cfg.norm_eps;

    // patch embed → [n, h]
    let xp = scratch.hold(
        gpu.upload_f32(&patches, &[n * patch_dim])
            .map_err(ehip("upload patches"))?,
    );
    let x = scratch.hold(linear_f16(
        gpu,
        &weights.patch_embed_w,
        scratch.t(xp)?,
        Some(&weights.patch_embed_b),
        h,
        patch_dim,
        n,
    )?);
    scratch.release(gpu, xp)?;

    // position embed add
    let pos = resize_pos_embed(&weights.pos_embed, k_side, h, sub.gh(cfg), sub.gw(cfg));
    let pos_gpu = scratch.hold(
        gpu.upload_f32(&pos, &[pos.len()])
            .map_err(ehip("upload pos"))?,
    );
    gpu.add_inplace_f32(scratch.t(x)?, scratch.t(pos_gpu)?)
        .map_err(ehip("pos add"))?;
    scratch.release(gpu, pos_gpu)?;

    // encoder layers: pre-LN attn residual + pre-LN MLP residual
    for lw in &weights.layers {
        let tmp = scratch.hold(
            gpu.alloc_tensor(&[n * h], DType::F32)
                .map_err(ehip("alloc ln1"))?,
        );
        gpu.layernorm_batched(
            scratch.t(x)?,
            &lw.norm1_w,
            &lw.norm1_b,
            scratch.t(tmp)?,
            n,
            h,
            eps,
        )
        .map_err(ehip("ln1"))?;
        let qkv = scratch.hold(linear_f16(
            gpu,
            &lw.qkv_w,
            scratch.t(tmp)?,
            Some(&lw.qkv_b),
            3 * h,
            h,
            n,
        )?);
        scratch.release(gpu, tmp)?;

        let attn_out = scratch.hold(
            gpu.alloc_tensor(&[n * h], DType::F32)
                .map_err(ehip("alloc attn"))?,
        );
        gpu.vit_attention_f32(scratch.t(qkv)?, scratch.t(attn_out)?, n, h, heads, head_dim)
            .map_err(ehip("vit_attention"))?;
        scratch.release(gpu, qkv)?;

        let proj = scratch.hold(linear_f16(
            gpu,
            &lw.proj_w,
            scratch.t(attn_out)?,
            Some(&lw.proj_b),
            h,
            h,
            n,
        )?);
        scratch.release(gpu, attn_out)?;
        gpu.add_inplace_f32(scratch.t(x)?, scratch.t(proj)?)
            .map_err(ehip("resid1"))?;
        scratch.release(gpu, proj)?;

        let tmp2 = scratch.hold(
            gpu.alloc_tensor(&[n * h], DType::F32)
                .map_err(ehip("alloc ln2"))?,
        );
        gpu.layernorm_batched(
            scratch.t(x)?,
            &lw.norm2_w,
            &lw.norm2_b,
            scratch.t(tmp2)?,
            n,
            h,
            eps,
        )
        .map_err(ehip("ln2"))?;
        let fc1 = scratch.hold(linear_f16(
            gpu,
            &lw.fc1_w,
            scratch.t(tmp2)?,
            Some(&lw.fc1_b),
            cfg.mlp_dim,
            h,
            n,
        )?);
        scratch.release(gpu, tmp2)?;
        gpu.gelu_tanh_f32(scratch.t(fc1)?, scratch.t(fc1)?, n * cfg.mlp_dim)
            .map_err(ehip("gelu(tanh)"))?;
        let fc2 = scratch.hold(linear_f16(
            gpu,
            &lw.fc2_w,
            scratch.t(fc1)?,
            Some(&lw.fc2_b),
            h,
            cfg.mlp_dim,
            n,
        )?);
        scratch.release(gpu, fc1)?;
        gpu.add_inplace_f32(scratch.t(x)?, scratch.t(fc2)?)
            .map_err(ehip("resid2"))?;
        scratch.release(gpu, fc2)?;
    }

    // post_layernorm (final LN — no pooling head)
    let normed = scratch.hold(
        gpu.alloc_tensor(&[n * h], DType::F32)
            .map_err(ehip("alloc post-ln"))?,
    );
    gpu.layernorm_batched(
        scratch.t(x)?,
        &weights.post_ln_w,
        &weights.post_ln_b,
        scratch.t(normed)?,
        n,
        h,
        eps,
    )
    .map_err(ehip("post-ln"))?;
    scratch.release(gpu, x)?;

    // download for CPU rearranges (small buffers)
    let feats = gpu
        .download_f32(scratch.t(normed)?)
        .map_err(ehip("download tower"))?;
    scratch.release(gpu, normed)?;
    gpu.hip
        .device_synchronize()
        .map_err(ehip("post-tower sync"))?;

    // 2×2 pixel-unshuffle merge → [tok, h*ds*ds]
    let ds = cfg.downsample_factor;
    if ds == 0 || sub.gh(cfg) % ds != 0 || sub.gw(cfg) % ds != 0 {
        return Err(format!(
            "lfm2-vl vision: grid {}×{} not divisible by downsample_factor={ds}",
            sub.gh(cfg),
            sub.gw(cfg)
        ));
    }
    let merged = pixel_unshuffle_tokens(&feats, sub.gh(cfg), sub.gw(cfg), h, ds);
    let tok = merged.len() / (h * ds * ds).max(1);
    if tok == 0 || merged.len() != tok * h * ds * ds {
        return Err(format!(
            "lfm2-vl vision: unshuffle produced {} floats for hidden={h} ds={ds}",
            merged.len()
        ));
    }

    // projector linear_1 → erf-GELU (host, exact) → linear_2
    let m1_in = scratch.hold(
        gpu.upload_f32(&merged, &[merged.len()])
            .map_err(ehip("upload merged"))?,
    );
    let mid_gpu = scratch.hold(linear_f16(
        gpu,
        &weights.proj1_w,
        scratch.t(m1_in)?,
        Some(&weights.proj1_b),
        cfg.projector_hidden_size,
        h * ds * ds,
        tok,
    )?);
    scratch.release(gpu, m1_in)?;
    let mut mid = gpu
        .download_f32(scratch.t(mid_gpu)?)
        .map_err(ehip("download proj1"))?;
    scratch.release(gpu, mid_gpu)?;
    gelu_exact_inplace(&mut mid);
    let act = scratch.hold(
        gpu.upload_f32(&mid, &[mid.len()])
            .map_err(ehip("re-upload act"))?,
    );

    let y = scratch.hold(linear_f16(
        gpu,
        &weights.proj2_w,
        scratch.t(act)?,
        Some(&weights.proj2_b),
        cfg.out_hidden_size,
        cfg.projector_hidden_size,
        tok,
    )?);
    scratch.release(gpu, act)?;
    let result = gpu
        .download_f32(scratch.t(y)?)
        .map_err(ehip("download proj2"))?;
    scratch.release(gpu, y)?;
    if result.len() != tok * cfg.out_hidden_size {
        return Err(format!(
            "lfm2-vl vision: projector produced {} floats for {tok} tokens × {} dims",
            result.len(),
            cfg.out_hidden_size
        ));
    }
    Ok(result)
}

fn ehip(phase: &'static str) -> impl Fn(hip_bridge::HipError) -> String {
    move |e| format!("lfm2-vl vision {phase}: {e:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity case: resizing the table to its own side must return it
    /// unchanged (center exactly on each source sample, weight 1).
    #[test]
    fn pos_resize_identity_at_native_side() {
        let k = 4usize;
        let d = 3usize;
        let table: Vec<f32> = (0..k * k * d).map(|i| i as f32).collect();
        let out = resize_pos_embed(&table, k, d, k, k);
        assert_eq!(out.len(), table.len());
        for (a, b) in out.iter().zip(&table) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// Upscale must interpolate: a 2×2 constant-value table resized to 3×3
    /// keeps the value everywhere (constant field invariant under bilinear).
    #[test]
    fn pos_resize_upscale_constant_field() {
        let k = 2usize;
        let d = 2usize;
        let mut table = vec![0.0f32; k * k * d];
        for r in 0..k {
            for c in 0..k {
                for dd in 0..d {
                    table[(r * k + c) * d + dd] = ((r * k + c) * 10 + dd) as f32;
                }
            }
        }
        let out = resize_pos_embed(&table, k, d, 4, 4);
        assert_eq!(out.len(), 16 * d);
        // Corner (0,0) samples center of cell (0,0): exact table value.
        assert_eq!(out[0], table[0]);
        assert_eq!(out[1], table[1]);
    }

    /// Antialiased downscale conserves local mean better than naive corner
    /// sampling: over a checkerboard table the coarse output between cells
    /// must land near the average, not snap to one cell.
    #[test]
    fn pos_resize_downscale_smooths() {
        let k = 8usize;
        let d = 1usize;
        let mut table = vec![0.0f32; k * k];
        for r in 0..k {
            for c in 0..k {
                table[r * k + c] = if (r + c) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let out = resize_pos_embed(&table, k, d, 2, 2);
        for v in &out {
            assert!(
                *v > 0.15 && *v < 0.85,
                "downscaled checkerboard should approach ~0.5, got {v}"
            );
        }
    }

    /// Locks the LFM2-VL projector interleave derived from HF's swapped-dim
    /// reshape/permute chain: columns pair first (stride C), rows second
    /// (stride 2C). Any future "obvious" reordering fails loudly here.
    #[test]
    fn pixel_unshuffle_index_contract() {
        let gh = 2;
        let gw = 2;
        let ch = 2;
        // feat[patch_row][patch_col][ch] = 100·row + 10·col + ch
        let mut feat = vec![0.0f32; gh * gw * ch];
        for r in 0..gh {
            for c in 0..gw {
                for cc in 0..ch {
                    feat[(r * gw + c) * ch + cc] = (100 * r + 10 * c + cc) as f32;
                }
            }
        }
        let out = pixel_unshuffle_tokens(&feat, gh, gw, ch, 2);
        assert_eq!(out.len(), ch * 4);
        // single block, flattened as [di][dj][channel]:
        assert_eq!(out[0] as usize, 0); // di=0, dj=0, c=0
        assert_eq!(out[2] as usize, 10); // di=0, dj=1, c=0
        assert_eq!(out[4] as usize, 100); // di=1, dj=0, c=0
        assert_eq!(out[7] as usize, 111); // di=1, dj=1, c=1
    }

    #[test]
    fn gelu_exact_known_values() {
        let mut v = vec![0.0f32, 1.0, -1.0];
        gelu_exact_inplace(&mut v);
        assert!(v[0].abs() < 1e-6);
        assert!((v[1] - 0.841_344_7).abs() < 1e-5); // Φ(1)=0.841344…
        assert!((v[2] + 0.158_655_2).abs() < 1e-5); // 1−Φ(1), negative input
    }

    fn tiny_cfg() -> VisionConfig {
        VisionConfig {
            hidden_size: 2,
            num_heads: 1,
            head_dim: 2,
            num_layers: 1,
            mlp_dim: 4,
            patch_size: 2,
            num_channels: 1,
            num_position_embeddings: 4,
            projector_hidden_size: 4,
            out_hidden_size: 4,
            downsample_factor: 2,
            ..VisionConfig::default()
        }
    }

    #[test]
    fn validate_dense_accepts_f16_and_f32_exact_layout() {
        assert_eq!(validate_dense("w", QT_F16, &[2, 2], 8, &[2, 2]).unwrap(), 4);
        assert_eq!(
            validate_dense("w", QT_F32, &[2, 2], 16, &[2, 2]).unwrap(),
            4
        );
        assert_eq!(validate_dense("b", QT_F16, &[3], 6, &[3]).unwrap(), 3);
    }

    #[test]
    fn validate_dense_rejects_rank_dim_bytes_dtype() {
        let rank = validate_dense("t", QT_F16, &[2], 8, &[2, 2]).unwrap_err();
        assert!(rank.contains("t:"), "{rank}");
        assert!(rank.contains("rank"), "{rank}");

        let dim = validate_dense("t", QT_F16, &[3, 2], 12, &[2, 2]).unwrap_err();
        assert!(dim.contains("dim[0]=3"), "{dim}");

        let bytes = validate_dense("t", QT_F16, &[2, 2], 7, &[2, 2]).unwrap_err();
        assert!(bytes.contains("7 bytes"), "{bytes}");

        let qt = validate_dense("t", 6, &[2, 2], 8, &[2, 2]).unwrap_err();
        assert!(qt.contains("quant_type=6"), "{qt}");
    }

    #[test]
    fn qkv_f32_narrows_to_same_f16_bytes() {
        let hidden = 2;
        let f32s = [1.0f32, -2.0, 0.5, 3.0];
        let f32_bytes: Vec<u8> = f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = f32s
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        let mut from_f32 = Vec::new();
        append_f16_weight(
            &mut from_f32,
            "q",
            QT_F32,
            &[2, 2],
            &f32_bytes,
            &[hidden, hidden],
        )
        .unwrap();
        let mut from_f16 = Vec::new();
        append_f16_weight(
            &mut from_f16,
            "q",
            QT_F16,
            &[2, 2],
            &f16_bytes,
            &[hidden, hidden],
        )
        .unwrap();
        assert_eq!(from_f32, from_f16);
        assert_eq!(from_f32.len(), 8);
    }

    #[test]
    fn concat_qkv_weight_path_rejects_f16_assert_equivalent() {
        // Previously concat_qkv asserted quant_type==1. F32 must be a Result.
        let mut dst = Vec::new();
        let err = append_f16_weight(&mut dst, "q_proj.weight", 3, &[2, 2], &[0u8; 8], &[2, 2])
            .unwrap_err();
        assert!(err.contains("q_proj.weight"), "{err}");
        assert!(err.contains("quant_type=3"), "{err}");
        assert!(dst.is_empty());
    }

    #[test]
    fn patch_linear_and_conv_match_config_and_fold() {
        let cfg = tiny_cfg();
        let name = "patch";
        // Linear [2, 4] = hidden × (ps²·C)
        let lin: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let lin_bytes: Vec<u8> = lin.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (rows, out, inn) =
            patch_weight_rows_from(name, QT_F32, &[2, 4], &lin_bytes, &cfg).unwrap();
        assert_eq!((out, inn), (2, 4));
        assert_eq!(rows, lin);

        // Conv [2, 1, 2, 2] — same 8 elems, reorder (c, dy, dx) → (dy, dx, c).
        // For C=1 the fold is identity.
        let (folded, out2, inn2) =
            patch_weight_rows_from(name, QT_F32, &[2, 1, 2, 2], &lin_bytes, &cfg).unwrap();
        assert_eq!((out2, inn2), (2, 4));
        assert_eq!(folded, lin);

        let bad_rank =
            patch_weight_rows_from(name, QT_F32, &[2, 4, 1], &lin_bytes, &cfg).unwrap_err();
        assert!(bad_rank.contains("rank 3"), "{bad_rank}");

        let bad_in = patch_weight_rows_from(name, QT_F32, &[2, 3], &lin_bytes, &cfg).unwrap_err();
        assert!(bad_in.contains("dim[1]"), "{bad_in}");
    }

    #[test]
    fn conv_patch_fold_reorders_channel_major_kernel() {
        // out=1, C=2, kh=kw=1 → in_dim=2. Conv store is [C, kh, kw] per out row.
        let cfg = VisionConfig {
            hidden_size: 1,
            patch_size: 1,
            num_channels: 2,
            ..tiny_cfg()
        };
        // conv[0, c, 0, 0] = c+1 → raw [1, 2]
        let raw = [1.0f32, 2.0];
        let bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (folded, out, inn) =
            patch_weight_rows_from("pe", QT_F32, &[1, 2, 1, 1], &bytes, &cfg).unwrap();
        assert_eq!((out, inn), (1, 2));
        assert_eq!(folded, vec![1.0, 2.0]);

        // 2×2 kernel, C=2, out=1. raw layout c-major: for each c, dy, dx.
        let cfg2 = VisionConfig {
            hidden_size: 1,
            patch_size: 2,
            num_channels: 2,
            ..tiny_cfg()
        };
        // c=0: [10, 11, 12, 13] (dy,dx), c=1: [20, 21, 22, 23]
        let raw2: Vec<f32> = vec![10., 11., 12., 13., 20., 21., 22., 23.];
        let bytes2: Vec<u8> = raw2.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (folded2, _, inn2) =
            patch_weight_rows_from("pe", QT_F32, &[1, 2, 2, 2], &bytes2, &cfg2).unwrap();
        assert_eq!(inn2, 8);
        // folded (dy, dx, c): (0,0)=[10,20], (0,1)=[11,21], (1,0)=[12,22], (1,1)=[13,23]
        assert_eq!(folded2, vec![10., 20., 11., 21., 12., 22., 13., 23.]);
    }
}
