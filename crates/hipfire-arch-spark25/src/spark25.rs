// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Spark-X2.5 weights + decode state.
//!
//! HF tensor names are used verbatim (no rename). Fused `q_k_v_proj` stays a
//! single `WeightTensor`; the forward path GEMVs once then splits Q|K|V via
//! D→D views. Embedding is tied to lm_head (`model.embedding.weight`).

use crate::config::Spark25Config;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use hipfire_runtime::weight_backend::{load_embedding, tied_lm_head_alias};
use rdna_compute::{DType, Gpu, GpuTensor};

/// HF embedding table name (not `embed_tokens`).
pub const EMBED_TENSOR_NAME: &str = "model.embedding.weight";
/// Final RMSNorm gamma.
pub const FINAL_NORM_TENSOR_NAME: &str = "model.norm.weight";

pub struct Spark25LayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    /// Fused Q|K|V projection: rows = q_dim + 2*kv_dim, cols = dim.
    pub q_k_v_proj: WeightTensor,
    /// Headwise attention gate: [n_heads, dim] → n_heads scalars.
    pub g_proj: WeightTensor,
    /// out_proj: [dim, q_dim].
    pub out_proj: WeightTensor,
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub struct Spark25Weights {
    pub embed_tokens: GpuTensor,
    pub embed_format: EmbeddingFormat,
    /// Tied lm_head alias of `embed_tokens` (or a separate tensor if untied).
    pub lm_head: WeightTensor,
    /// True when `lm_head.buf` aliases `embed_tokens` and must not be freed twice.
    pub lm_head_aliases_embd: bool,
    pub layers: Vec<Spark25LayerWeights>,
    pub final_norm: GpuTensor,
}

pub struct Spark25State {
    pub max_seq: usize,
    pub n_tokens: usize,
    /// Dual KV caches: sliding (window 512) + full. Sized by `layer_types`.
    pub kv_sliding: KvCache,
    pub kv_full: KvCache,
    /// Per-layer slot into the matching type cache.
    pub kv_slot_for_layer: Vec<usize>,
    pub n_sliding: usize,
    pub n_full: usize,
    // Scratch buffers
    pub x: GpuTensor,
    pub residual: GpuTensor,
    pub tmp: GpuTensor,
    pub qkv_buf: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub attn_out: GpuTensor,
    pub attn_gate: GpuTensor,
    pub gate_ffn: GpuTensor,
    pub up_ffn: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub logits: GpuTensor,
    pub pos_buf: hip_bridge::DeviceBuffer,
    pub pos_host: Box<i32>,
}

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("spark25: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

fn widen_to_f32(qt: u8, data: &[u8]) -> Option<Vec<f32>> {
    match qt {
        1 => Some(
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        2 => Some(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        3 => Some(dequant_q8_0(data)),
        16 => Some(
            data.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        _ => None,
    }
}

/// Load a 1D F16/BF16/F32/Q8 vector → F32 GpuTensor (RMSNorm gammas).
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data = widen_to_f32(qt, &data)
        .ok_or_else(|| format!("spark25: expected F16/BF16/F32/Q8 for {name}, got qt={qt}"))?;
    let expected: usize = shape.iter().product();
    if expected != 0 && f32_data.len() != expected {
        return Err(format!(
            "spark25: {name}: got {} elements, expected {expected}",
            f32_data.len()
        ));
    }
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("spark25: upload {name}: {e:?}"))
}

/// quant_type → WeightTensor. MQ4V2 (qt=44) is first-class; never fold to V1.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    // Host-widen F16/BF16 to F32 for GEMV (no native F16 GEMV path on all arches).
    if qt == 1 || qt == 16 {
        let f32_data = widen_to_f32(qt, &data)
            .ok_or_else(|| format!("spark25: widen failed for {name} qt={qt}"))?;
        if f32_data.len() != m * k {
            return Err(format!(
                "spark25: {name}: F16/BF16 elements {} != m*k {}",
                f32_data.len(),
                m * k
            ));
        }
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu
            .upload_raw(bytes, &[m, k])
            .map_err(|e| format!("spark25: upload F32 {name}: {e:?}"))?;
        return Ok(WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        });
    }
    let dtype = match qt {
        2 => {
            let buf = gpu
                .upload_raw(&data, &[m, k])
                .map_err(|e| format!("spark25: upload F32 {name}: {e:?}"))?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            });
        }
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        11 => DType::HFQ3G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        19 => DType::MQ4G256, // MG4G256 layout-identical
        // qt=44 MQ4G256V2 — distinct container; NEVER map to V1.
        44 => DType::MQ4G256V2,
        45 => DType::MQ4CG256,
        47 => DType::MQ6G256V2,
        other => {
            return Err(format!(
                "spark25: unsupported quant_type {other} for {name}"
            ))
        }
    };
    let buf = gpu
        .upload_raw(&data, &[data.len()])
        .map_err(|e| format!("spark25: upload {name}: {e:?}"))?;
    let awq_scale = if dtype.supports_awq_sidecar() {
        hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k)
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale,
    })
}

fn layer_tensor(layer: usize, rel: &str) -> String {
    format!("model.layers.{layer}.{rel}")
}

fn build_kv_slot_map(cfg: &Spark25Config) -> (Vec<usize>, usize, usize) {
    let mut slots = vec![0usize; cfg.n_layers];
    let mut sliding = 0usize;
    let mut full = 0usize;
    for i in 0..cfg.n_layers {
        if cfg.is_sliding(i) {
            slots[i] = sliding;
            sliding += 1;
        } else {
            slots[i] = full;
            full += 1;
        }
    }
    (slots, sliding, full)
}

impl Spark25Weights {
    pub fn load(hfq: &HfqFile, cfg: &Spark25Config, gpu: &mut Gpu) -> Result<Self, String> {
        // MQ rotated formats need FWHT sign LUTs before any GEMV.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("spark25: ensure_mq_signs: {e:?}"))?;

        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let q_dim = cfg.q_dim();
        let qkv_rows = cfg.qkv_rows();
        let n_heads = cfg.n_heads;
        let vocab = cfg.vocab_size;

        eprintln!("  spark25: loading embedding ({EMBED_TENSOR_NAME})...");
        let (eqt, edata) = read_tensor(hfq, EMBED_TENSOR_NAME)?;
        let (embed_tokens, embed_format) = load_embedding(gpu, eqt, &edata, vocab, dim)
            .map_err(|e| format!("spark25: load embedding: {e:?}"))?;
        let (lm_head, lm_head_aliases_embd) = if cfg.tie_word_embeddings {
            eprintln!("  spark25: tying lm_head to embedding...");
            (
                tied_lm_head_alias(&embed_tokens, embed_format, vocab, dim),
                true,
            )
        } else if hfq.find_tensor_info("lm_head.weight").is_some() {
            eprintln!("  spark25: loading separate lm_head.weight...");
            let w = load_wt(hfq, gpu, "lm_head.weight", vocab, dim)?;
            (w, false)
        } else {
            return Err(
                "spark25: tie_word_embeddings=false but `lm_head.weight` is absent; refusing to \
                 alias the embedding as an independent head"
                    .into(),
            );
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if i % 8 == 0 || i + 1 == cfg.n_layers {
                eprintln!("  spark25: loading layer {i}/{}...", cfg.n_layers);
            }
            let input_layernorm =
                load_f32(hfq, gpu, &layer_tensor(i, "input_layernorm.weight"), &[dim])?;
            let post_attention_layernorm = load_f32(
                hfq,
                gpu,
                &layer_tensor(i, "post_attention_layernorm.weight"),
                &[dim],
            )?;
            let q_k_v_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "self_attn.q_k_v_proj.weight"),
                qkv_rows,
                dim,
            )?;
            let g_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "self_attn.g_proj.weight"),
                n_heads,
                dim,
            )?;
            let out_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "self_attn.out_proj.weight"),
                dim,
                q_dim,
            )?;
            let gate_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "mlp.gate_proj.weight"),
                hidden,
                dim,
            )?;
            let up_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "mlp.up_proj.weight"),
                hidden,
                dim,
            )?;
            let down_proj = load_wt(
                hfq,
                gpu,
                &layer_tensor(i, "mlp.down_proj.weight"),
                dim,
                hidden,
            )?;
            layers.push(Spark25LayerWeights {
                input_layernorm,
                post_attention_layernorm,
                q_k_v_proj,
                g_proj,
                out_proj,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        eprintln!("  spark25: loading final norm...");
        let final_norm = load_f32(hfq, gpu, FINAL_NORM_TENSOR_NAME, &[dim])?;

        Ok(Self {
            embed_tokens,
            embed_format,
            lm_head,
            lm_head_aliases_embd,
            layers,
            final_norm,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Spark25Weights {
            embed_tokens,
            embed_format: _,
            lm_head,
            lm_head_aliases_embd,
            layers,
            final_norm,
        } = self;
        let _ = gpu.free_tensor(embed_tokens);
        if !lm_head_aliases_embd {
            lm_head.free_all(gpu);
        }
        let _ = gpu.free_tensor(final_norm);
        for l in layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            l.q_k_v_proj.free_all(gpu);
            l.g_proj.free_all(gpu);
            l.out_proj.free_all(gpu);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

impl Spark25State {
    pub fn new(cfg: &Spark25Config, gpu: &mut Gpu, max_seq: usize) -> Result<Self, String> {
        // Truthful device/kernel ceiling for `attention_q8_0_kv_swa`.
        // The kernel indexes `scores[t]` by absolute `t in [t_lo, seq_len)` and sizes
        // dynamic LDS from the launch `seq_len_hint` as
        //   LDS = (seq + block + head_dim) * 4,
        //   block = next_power_of_two(max(seq, head_dim)).min(256)
        // (see `Gpu::attention_q8_0_kv_swa` in rdna-compute/src/attention.rs).
        // That is byte-identical to `attention_q8_0_kv_independent_lds_bytes`, so the
        // supported cap is derived from the existing device property
        // `attention_q8_0_kv_independent_shared_mem_limit` (hipDeviceAttribute
        // MaxSharedMemoryPerBlock, ordinal 74; 64 KiB RDNA fallback) — no new core
        // API, no magic constant. `cfg.max_position_embeddings` (1M on the official
        // checkpoint) is the model rope range, not an admission promise.
        let shm_limit = gpu.attention_q8_0_kv_independent_shared_mem_limit();
        let kernel_cap = gpu.attention_q8_0_kv_independent_max_lane_capacity(cfg.head_dim);
        if kernel_cap == 0 {
            return Err(format!(
                "spark25: attention_q8_0_kv_swa cannot launch on this device \
                 (shared-mem limit {shm_limit} bytes fits no seq_len at head_dim {})",
                cfg.head_dim
            ));
        }
        let requested = max_seq.max(1);
        if requested > kernel_cap {
            return Err(format!(
                "spark25: requested max_seq {requested} exceeds the supported \
                 attention_q8_0_kv_swa kernel/device cap {kernel_cap} (shared-mem limit \
                 {shm_limit} bytes, head_dim {}; LDS=(seq+block+head_dim)*4, \
                 block=next_pow2(max(seq,head_dim)).min(256); model rope range is {} but \
                 this path cannot admit unsupported dynamic LDS)",
                cfg.head_dim, cfg.max_position_embeddings
            ));
        }
        // The clamp only shrinks toward the model rope range, so the admitted value
        // stays within the kernel cap checked above; state.max_seq is truthful.
        let max_seq = requested.min(cfg.max_position_embeddings.max(1));
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_rows = cfg.qkv_rows();
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let vocab = cfg.vocab_size;

        let (kv_slot_for_layer, n_sliding, n_full) = build_kv_slot_map(cfg);
        if n_sliding + n_full != cfg.n_layers {
            return Err(format!(
                "spark25: kv slot map invariant broken: sliding={n_sliding} full={n_full} layers={}",
                cfg.n_layers
            ));
        }

        // Sliding: Q8, physical capacity = max_seq (SWA window applied at attn).
        // Full: Q8, full max_seq. Both use absolute positions like gemma4 eager.
        let kv_sliding = if n_sliding > 0 {
            KvCache::new_gpu_q8(gpu, n_sliding, n_kv, head_dim, max_seq)
                .map_err(|e| format!("spark25: sliding kv: {e:?}"))?
        } else {
            // Zero-layer placeholder so free_gpu is uniform.
            KvCache::new_gpu_q8(gpu, 1, n_kv, head_dim, 1)
                .map_err(|e| format!("spark25: empty sliding kv: {e:?}"))?
        };
        let kv_full = if n_full > 0 {
            KvCache::new_gpu_q8(gpu, n_full, n_kv, head_dim, max_seq)
                .map_err(|e| format!("spark25: full kv: {e:?}"))?
        } else {
            KvCache::new_gpu_q8(gpu, 1, n_kv, head_dim, 1)
                .map_err(|e| format!("spark25: empty full kv: {e:?}"))?
        };

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("spark25: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("spark25: alloc {label}: {e:?}"))
        };

        Ok(Self {
            max_seq,
            n_tokens: 0,
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            n_sliding,
            n_full,
            x: alloc(gpu, dim, "x")?,
            residual: alloc(gpu, dim, "residual")?,
            tmp: alloc(gpu, dim, "tmp")?,
            qkv_buf: alloc(gpu, qkv_rows, "qkv_buf")?,
            q: alloc(gpu, q_dim, "q")?,
            k: alloc(gpu, kv_dim, "k")?,
            v: alloc(gpu, kv_dim, "v")?,
            attn_out: alloc(gpu, q_dim, "attn_out")?,
            attn_gate: alloc(gpu, n_heads, "attn_gate")?,
            gate_ffn: alloc(gpu, hidden, "gate_ffn")?,
            up_ffn: alloc(gpu, hidden, "up_ffn")?,
            ffn_hidden: alloc(gpu, hidden, "ffn_hidden")?,
            logits: alloc(gpu, vocab, "logits")?,
            pos_buf,
            pos_host: Box::new(0i32),
        })
    }

    /// REQUIRED session lifecycle: zero both KV caches and rewind the token cursor.
    ///
    /// Always rewinds the host cursor (`n_tokens`, `pos_host`) and always
    /// attempts BOTH cache clears, even if the first fails, so a partially
    /// cleared state is never reported as clean: single-side failures return
    /// that side's error, dual failures return both combined.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        *self.pos_host = 0;
        let sliding = self
            .kv_sliding
            .clear_gpu(gpu)
            .map_err(|e| format!("spark25 reset: clear sliding kv: {e:?}"));
        let full = self
            .kv_full
            .clear_gpu(gpu)
            .map_err(|e| format!("spark25 reset: clear full kv: {e:?}"));
        match (sliding, full) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(e),
            (Err(a), Err(b)) => Err(format!("spark25 reset failed: [{a}] [{b}]")),
        }
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Spark25State {
            max_seq: _,
            n_tokens: _,
            kv_sliding,
            kv_full,
            kv_slot_for_layer: _,
            n_sliding: _,
            n_full: _,
            x,
            residual,
            tmp,
            qkv_buf,
            q,
            k,
            v,
            attn_out,
            attn_gate,
            gate_ffn,
            up_ffn,
            ffn_hidden,
            logits,
            pos_buf,
            pos_host: _,
        } = self;
        let _ = kv_sliding.free_gpu(gpu);
        let _ = kv_full.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            x, residual, tmp, qkv_buf, q, k, v, attn_out, attn_gate, gate_ffn, up_ffn, ffn_hidden,
            logits,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
