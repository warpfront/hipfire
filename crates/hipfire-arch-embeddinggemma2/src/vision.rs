// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Gemma-4 vision tower: patches → `[N, text_hidden]` rows ready to
//! splice into the trunk's embedding stream.
//!
//! Formula-level ground truth: `docs/plans/embeddinggemma2-tower-spec.md`
//! §VISION (transformers 5.19.0 `models/gemma4/modeling_gemma4.py`).
//!
//! Forward (f32 activations, f16 WMMA weights through the crate's `Enc`
//! dispatch — the qwen35-embed numerics contract):
//!
//! - **Patch embedder**: `h = 2*(patches - 0.5) · Wᵢₙᵀ` (`input_proj`, no
//!   bias) then `+ position_embedding_table[0][x] + [1][y]`. The table is a
//!   host f32 copy and the lookup is a host gather: the two rows a patch
//!   needs are 768 floats each and the alternative (a device gather kernel
//!   over a 63 MB table) buys nothing at this row count.
//! - **16 sandwich layers**: `rmsnorm → q/k/v → per-head q/k RMSNorm
//!   (learned, width `head_dim`) + scale-less v RMSNorm → axial 2-D RoPE →
//!   attention (scale 1.0, pad keys masked) → o_proj → post-attention norm on
//!   the branch → residual → pre-FFN norm → `down(gelu_tanh(gate(x)) * up(x))`
//!   → post-FFN norm on the branch → residual`. No final tower norm.
//! - **Pooler**: 3×3 mean over each bin, `sqrt(hidden)` scale (f32, host).
//!   Bin `b = x/k + (grid_w/k)*(y/k)`; only real bins exist in our patch list
//!   (the processor keeps real patches only), so no empty-bin strip is needed
//!   — that strip is what HF's `pooler_mask` does for its zero padding.
//! - **Embedder**: scale-less RMSNorm (`eps` = tower `rms_norm_eps`, no
//!   weight) then `embed_vision.embedding_projection` `[text_hidden, hidden]`
//!   → `[N, text_hidden]`.
//!
//! Numerics note: our RMSNorm kernel uses `rsqrtf` where HF's `Gemma4RMSNorm`
//! uses `pow(mean_sq + eps, -0.5)`. Both are fp32; the difference is ≤1 ulp,
//! three orders of magnitude below the f16 weight-operand noise floor this
//! arch already accepts. It is NOT byte-identical to the CPU reference — the
//! vision oracle slice measures the band.
//!
//! The tower has no `use_clipped_linears` / `standardize` tensors on this
//! checkpoint and both config flags are refused when true (see
//! [`crate::config::parse_vision_config`]).

use crate::config::VisionConfig;
use crate::forward::Enc;
use crate::image::ProcessedImage;
use crate::names;
use crate::weights::{decode_f16, decode_f32, upload_f16_words, upload_f32_slice};
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::{Gpu, GpuTensor};

pub struct VisionLayerWeights {
    pub input_ln: GpuTensor,
    pub post_attn_ln: GpuTensor,
    pub pre_ff_ln: GpuTensor,
    pub post_ff_ln: GpuTensor,
    /// `[q_dim, hidden]` f16, head-dim rows PERMUTED for the axial RoPE
    /// (see [`axial_perm`]).
    pub q_proj: GpuTensor,
    /// `[kv_dim, hidden]` f16, permuted like `q_proj`.
    pub k_proj: GpuTensor,
    /// `[kv_dim, hidden]` f16 — V is never rotated, so this keeps the
    /// checkpoint's own head-dim order (o_proj depends on that order).
    pub v_proj: GpuTensor,
    pub o_proj: GpuTensor,
    /// `[head_dim]` f32, permuted like `q_proj`.
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
}

/// The loaded vision tower (the `.vl` sidecar's `vision_tower.*` +
/// `embed_vision.*` groups).
pub struct VisionTower {
    pub config: VisionConfig,
    /// `[hidden, patch_pixels]` f16 — the patch-embedder GEMM's weight.
    pub patch_in_proj: GpuTensor,
    /// HOST `[2, position_embedding_size, hidden]` f32: exactness at gather
    /// (the sidecar stores it f32 for the same reason).
    pub position_table: Vec<f32>,
    pub layers: Vec<VisionLayerWeights>,
    /// `[text_hidden, hidden]` f16.
    pub embed_projection: GpuTensor,
    /// The trunk's hidden width — the embedder's output width.
    pub text_hidden: usize,
    /// Ones vectors for the two scale-less norms.
    v_ones: GpuTensor,
    e_ones: GpuTensor,
}

// ─── the axial RoPE head-dim permutation ───────────────────────────────────

/// Kernel index → Gemma head dim, for the axial H-H-W-W RoPE.
///
/// HF builds the table as `cat([h, h, w, w])` over two equal HALVES —
/// `Gemma4VisionRotaryEmbedding.recomposition_frequencies` — and
/// `apply_multidimensional_rope` (modeling_gemma4.py L848) splits the head
/// dim into those two halves and rotates each with `rotate_half`, i.e. with
/// pairs `(j, j + head_dim/4)` *inside* the half: gemma dims `0..head_dim/2`
/// are the x-angles and `head_dim/2..` the y-angles, so the gemma pairs are
/// `(j, j + q)` for `q = head_dim/4` in the x half and `(half + j, half + q +
/// j)` in the y half.
///
/// The rope kernel (`rope_2d_halfsplit_f32`) instead pairs `(d, d +
/// head_dim/2)` and reads independent cos/sin per side, so the gemma pairs
/// must land at kernel indices `(d, d + half)`. This map does that: the
/// kernel's head-dim quarters come out as `[x, y, x, y]` where the
/// checkpoint's own layout is `[x, x, y, y]`.
///
/// Costs one row permutation at upload and changes nothing else: Q and K
/// carry the SAME permutation, and a permutation is orthogonal, so every
/// `q·k` score is bit-identical; V (and therefore o_proj's input order) is
/// untouched.
fn axial_perm(head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    let q = head_dim / 4;
    let mut perm = vec![0usize; head_dim];
    for j in 0..q {
        // x half: gemma (j, j + q) → kernel (j, j + half).
        perm[j] = j;
        perm[j + half] = j + q;
        // y half: gemma (half + j, half + q + j) → kernel (q + j, q + j + half).
        perm[q + j] = half + j;
        perm[half + q + j] = half + q + j;
    }
    perm
}

/// Apply [`axial_perm`] to the rows of a `[n_heads * head_dim, width]`
/// matrix, one head block at a time.
fn permute_head_rows_f16(
    words: &[u16],
    n_heads: usize,
    head_dim: usize,
    width: usize,
    perm: &[usize],
) -> Vec<u16> {
    let mut out = words.to_vec();
    for h in 0..n_heads {
        for d in 0..head_dim {
            let dst = (h * head_dim + d) * width;
            let src = (h * head_dim + perm[d]) * width;
            out[dst..dst + width].copy_from_slice(&words[src..src + width]);
        }
    }
    out
}

/// Apply [`axial_perm`] to a 1-D vector (`q_norm`/`k_norm`, one vector
/// shared by every head).
fn permute_vec_f32(v: &[f32], perm: &[usize]) -> Vec<f32> {
    perm.iter().map(|&p| v[p]).collect()
}

/// `inv_freq[i] = theta^(-2i/(head_dim/2))`, `i in 0..head_dim/4` — the same
/// frequencies on both axes (HF `compute_axial_rope_parameters`).
fn axial_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let spatial = head_dim / 2;
    let quarter = head_dim / 4;
    (0..quarter)
        .map(|i| {
            let exponent = (2 * i) as f32 / spatial as f32;
            theta.powf(-exponent)
        })
        .collect()
}

/// The cos/sin tables the rope kernel consumes: `[rows, head_dim]`, in
/// PERMUTED head-dim order, zero for rows past `n` (padding rows — their
/// outputs are dropped).
fn rope_tables(
    cfg: &VisionConfig,
    positions: &[i32],
    n: usize,
    rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim;
    let q = hd / 4;
    let inv = axial_inv_freq(hd, cfg.rope_theta);
    let perm = axial_perm(hd);
    let mut cos = vec![0.0f32; rows * hd];
    let mut sin = vec![0.0f32; rows * hd];
    for r in 0..n.min(rows) {
        let x = positions[r * 2] as f32;
        let y = positions[r * 2 + 1] as f32;
        for d in 0..hd {
            let z = perm[d];
            // The gemma dim's quarter ∈ {x, x, y, y} (its two halves are the
            // x- and y-angle halves); the angle is that axis times the
            // frequency of the position within the quarter.
            let axis = if z / q < 2 { x } else { y };
            let angle = axis * inv[z % q];
            cos[r * hd + d] = angle.cos();
            sin[r * hd + d] = angle.sin();
        }
    }
    (cos, sin)
}

// ─── loading ───────────────────────────────────────────────────────────────

/// Stream the `.vl` sidecar's vision tower to the GPU.
pub fn load_vision_tower(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &VisionConfig,
    text_hidden: usize,
) -> Result<VisionTower, String> {
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let perm = axial_perm(hd);

    let read_f16_words = |name: &str| -> Result<Vec<u16>, String> {
        let t = names::read_tower(src, name)?;
        let words = decode_f16(&t, name)?;
        names::release_tower(src, name);
        Ok(words)
    };

    let patch_in_proj = {
        let name = "vision_tower.patch_embedder.input_proj.weight";
        let words = read_f16_words(name)?;
        upload_f16_words(gpu, &words, hidden, cfg.patch_pixels(), name)?
    };

    let position_table = {
        let name = "vision_tower.patch_embedder.position_embedding_table";
        let t = names::read_tower(src, name)?;
        let v = decode_f32(&t, name)?;
        names::release_tower(src, name);
        let want = 2 * cfg.position_axis_stride();
        if v.len() != want {
            return Err(format!(
                "embeddinggemma2: tower tensor `{name}` has {} elems, expected {want} \
                 (2 x {} x {hidden})",
                v.len(),
                cfg.position_embedding_size
            ));
        }
        v
    };

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("vision_tower.encoder.layers.{li}");
        let q_name = format!("{p}.self_attn.q_proj.linear.weight");
        let q_words = read_f16_words(&q_name)?;
        if q_words.len() != cfg.q_dim() * hidden {
            return Err(format!(
                "embeddinggemma2: tower tensor `{q_name}` has {} elems, expected {}x{hidden}",
                q_words.len(),
                cfg.q_dim()
            ));
        }
        let k_name = format!("{p}.self_attn.k_proj.linear.weight");
        let k_words = read_f16_words(&k_name)?;
        if k_words.len() != cfg.kv_dim() * hidden {
            return Err(format!(
                "embeddinggemma2: tower tensor `{k_name}` has {} elems, expected {}x{hidden}",
                k_words.len(),
                cfg.kv_dim()
            ));
        }
        let q_perm = permute_head_rows_f16(&q_words, cfg.n_heads, hd, hidden, &perm);
        let k_perm = permute_head_rows_f16(&k_words, cfg.n_kv_heads, hd, hidden, &perm);

        let vec_f32 = |gpu: &mut Gpu, name: &str, n: usize| -> Result<GpuTensor, String> {
            let t = names::read_tower(src, name)?;
            let v = decode_f32(&t, name)?;
            names::release_tower(src, name);
            if v.len() != n {
                return Err(format!(
                    "embeddinggemma2: tower tensor `{name}` has {} elems, expected {n}",
                    v.len()
                ));
            }
            upload_f32_slice(gpu, &v, name)
        };
        let norm_perm = |gpu: &mut Gpu, name: &str| -> Result<GpuTensor, String> {
            let t = names::read_tower(src, name)?;
            let v = decode_f32(&t, name)?;
            names::release_tower(src, name);
            if v.len() != hd {
                return Err(format!(
                    "embeddinggemma2: tower tensor `{name}` has {} elems, expected {hd}",
                    v.len()
                ));
            }
            let v = permute_vec_f32(&v, &perm);
            upload_f32_slice(gpu, &v, name)
        };

        layers.push(VisionLayerWeights {
            input_ln: vec_f32(gpu, &format!("{p}.input_layernorm.weight"), hidden)?,
            post_attn_ln: vec_f32(gpu, &format!("{p}.post_attention_layernorm.weight"), hidden)?,
            pre_ff_ln: vec_f32(
                gpu,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                hidden,
            )?,
            post_ff_ln: vec_f32(
                gpu,
                &format!("{p}.post_feedforward_layernorm.weight"),
                hidden,
            )?,
            q_proj: upload_f16_words(gpu, &q_perm, cfg.q_dim(), hidden, &q_name)?,
            k_proj: upload_f16_words(gpu, &k_perm, cfg.kv_dim(), hidden, &k_name)?,
            // V keeps the checkpoint order (it is never rotated).
            v_proj: {
                let name = format!("{p}.self_attn.v_proj.linear.weight");
                let words = read_f16_words(&name)?;
                upload_f16_words(gpu, &words, cfg.kv_dim(), hidden, &name)?
            },
            o_proj: {
                let name = format!("{p}.self_attn.o_proj.linear.weight");
                let words = read_f16_words(&name)?;
                upload_f16_words(gpu, &words, hidden, cfg.q_dim(), &name)?
            },
            q_norm: norm_perm(gpu, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: norm_perm(gpu, &format!("{p}.self_attn.k_norm.weight"))?,
            gate_proj: {
                let name = format!("{p}.mlp.gate_proj.linear.weight");
                let words = read_f16_words(&name)?;
                upload_f16_words(gpu, &words, cfg.intermediate_size, hidden, &name)?
            },
            up_proj: {
                let name = format!("{p}.mlp.up_proj.linear.weight");
                let words = read_f16_words(&name)?;
                upload_f16_words(gpu, &words, cfg.intermediate_size, hidden, &name)?
            },
            down_proj: {
                let name = format!("{p}.mlp.down_proj.linear.weight");
                let words = read_f16_words(&name)?;
                upload_f16_words(gpu, &words, hidden, cfg.intermediate_size, &name)?
            },
        });
    }

    let embed_projection = {
        let name = "embed_vision.embedding_projection.weight";
        let words = read_f16_words(name)?;
        if words.len() != text_hidden * hidden {
            return Err(format!(
                "embeddinggemma2: tower tensor `{name}` has {} elems, expected \
                 {text_hidden}x{hidden} (the text hidden width)",
                words.len()
            ));
        }
        upload_f16_words(gpu, &words, text_hidden, hidden, name)?
    };

    let v_ones = gpu
        .upload_f32(&vec![1.0f32; hd], &[hd])
        .map_err(|e| format!("embeddinggemma2: tower v_norm ones: {e:?}"))?;
    let e_ones = gpu
        .upload_f32(&vec![1.0f32; hidden], &[hidden])
        .map_err(|e| format!("embeddinggemma2: tower embedder ones: {e:?}"))?;

    Ok(VisionTower {
        config: *cfg,
        patch_in_proj,
        position_table,
        layers,
        embed_projection,
        text_hidden,
        v_ones,
        e_ones,
    })
}

impl VisionTower {
    /// A synthetic tower for GPU smoke tests and examples: deterministic
    /// small random weights, no checkpoint, the same tensor set and the same
    /// head-dim permutation the real loader produces. Real loads go through
    /// [`load_vision_tower`]; this exists so `examples/vision_smoke.rs` can
    /// drive patch-embed → layers → pooler → embedder end to end without the
    /// `.vl` sidecar.
    pub fn synthetic(
        gpu: &mut Gpu,
        cfg: &VisionConfig,
        text_hidden: usize,
        seed: u64,
    ) -> Result<Self, String> {
        use half::f16;

        struct Rng(u64);
        impl Rng {
            /// splitmix64 → [0, 1).
            fn next(&mut self) -> f32 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 40) as f32) / ((1u32 << 24) as f32)
            }
            fn mat(&mut self, rows: usize, cols: usize) -> Vec<u16> {
                (0..rows * cols)
                    .map(|_| f16::from_f32(self.next() * 0.04 - 0.02).to_bits())
                    .collect()
            }
            fn vec(&mut self, n: usize) -> Vec<f32> {
                (0..n).map(|_| 1.0 + self.next() * 0.02 - 0.01).collect()
            }
        }

        let mut rng = Rng(seed | 1);
        let hd = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let perm = axial_perm(hd);

        let f16_raw =
            |rng: &mut Rng, gpu: &mut Gpu, rows: usize, cols: usize| -> Result<GpuTensor, String> {
                let v = rng.mat(rows, cols);
                gpu.upload_f16_bits(&v, &[rows, cols])
                    .map_err(|e| format!("synthetic tower upload: {e:?}"))
            };
        let f32_vec = |rng: &mut Rng, gpu: &mut Gpu, n: usize| -> Result<GpuTensor, String> {
            let v = rng.vec(n);
            gpu.upload_f32(&v, &[n])
                .map_err(|e| format!("synthetic tower upload: {e:?}"))
        };

        let patch_in_proj = f16_raw(&mut rng, gpu, hidden, cfg.patch_pixels())?;
        let position_table: Vec<f32> = (0..2 * cfg.position_axis_stride())
            .map(|_| rng.next() * 0.04 - 0.02)
            .collect();

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            // Head-dim rows carry the same permutation the loader applies, so
            // the RoPE tables stay valid for the synthetic weights too.
            let q_raw = f16_raw(&mut rng, gpu, cfg.q_dim(), hidden)?;
            let qp = permute_head_rows_f16(
                &gpu.download_f16_bits(&q_raw)
                    .map_err(|e| format!("synthetic download: {e:?}"))?,
                cfg.n_heads,
                hd,
                hidden,
                &perm,
            );
            gpu.free_tensor(q_raw)
                .map_err(|e| format!("synthetic free: {e:?}"))?;
            let k_raw = f16_raw(&mut rng, gpu, cfg.kv_dim(), hidden)?;
            let kp = permute_head_rows_f16(
                &gpu.download_f16_bits(&k_raw)
                    .map_err(|e| format!("synthetic download: {e:?}"))?,
                cfg.n_kv_heads,
                hd,
                hidden,
                &perm,
            );
            gpu.free_tensor(k_raw)
                .map_err(|e| format!("synthetic free: {e:?}"))?;
            layers.push(VisionLayerWeights {
                input_ln: f32_vec(&mut rng, gpu, hidden)?,
                post_attn_ln: f32_vec(&mut rng, gpu, hidden)?,
                pre_ff_ln: f32_vec(&mut rng, gpu, hidden)?,
                post_ff_ln: f32_vec(&mut rng, gpu, hidden)?,
                q_proj: gpu
                    .upload_f16_bits(&qp, &[cfg.q_dim(), hidden])
                    .map_err(|e| format!("synthetic upload: {e:?}"))?,
                k_proj: gpu
                    .upload_f16_bits(&kp, &[cfg.kv_dim(), hidden])
                    .map_err(|e| format!("synthetic upload: {e:?}"))?,
                v_proj: f16_raw(&mut rng, gpu, cfg.kv_dim(), hidden)?,
                o_proj: f16_raw(&mut rng, gpu, hidden, cfg.q_dim())?,
                q_norm: {
                    let mut v = rng.vec(hd);
                    v = permute_vec_f32(&v, &perm);
                    gpu.upload_f32(&v, &[hd])
                        .map_err(|e| format!("synthetic upload: {e:?}"))?
                },
                k_norm: {
                    let mut v = rng.vec(hd);
                    v = permute_vec_f32(&v, &perm);
                    gpu.upload_f32(&v, &[hd])
                        .map_err(|e| format!("synthetic upload: {e:?}"))?
                },
                gate_proj: f16_raw(&mut rng, gpu, cfg.intermediate_size, hidden)?,
                up_proj: f16_raw(&mut rng, gpu, cfg.intermediate_size, hidden)?,
                down_proj: f16_raw(&mut rng, gpu, hidden, cfg.intermediate_size)?,
            });
        }

        let embed_projection = f16_raw(&mut rng, gpu, text_hidden, hidden)?;
        let v_ones = gpu
            .upload_f32(&vec![1.0f32; hd], &[hd])
            .map_err(|e| format!("synthetic tower ones: {e:?}"))?;
        let e_ones = gpu
            .upload_f32(&vec![1.0f32; hidden], &[hidden])
            .map_err(|e| format!("synthetic tower ones: {e:?}"))?;

        Ok(VisionTower {
            config: *cfg,
            patch_in_proj,
            position_table,
            layers,
            embed_projection,
            text_hidden,
            v_ones,
            e_ones,
        })
    }

    /// The `.vl` sidecar probe every caller uses: presence of the patch
    /// embedder's projection (`EMBED_GEMMA_VISION_PROBE_TENSOR`).
    pub fn sidecar_has_tower(src: &ModelSource) -> bool {
        names::tower_has(src, "vision_tower.patch_embedder.input_proj.weight")
    }

    /// Free every GPU allocation. The position table is host memory.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let VisionTower {
            position_table: _,
            patch_in_proj,
            layers,
            embed_projection,
            v_ones,
            e_ones,
            ..
        } = self;
        for l in layers {
            for t in [
                l.input_ln,
                l.post_attn_ln,
                l.pre_ff_ln,
                l.post_ff_ln,
                l.q_proj,
                l.k_proj,
                l.v_proj,
                l.o_proj,
                l.q_norm,
                l.k_norm,
                l.gate_proj,
                l.up_proj,
                l.down_proj,
            ] {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in [patch_in_proj, embed_projection, v_ones, e_ones] {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Rows the tower runs for `n` real patches.
    ///
    /// The attention matrix is materialized (`[rows, rows]` f32), so the
    /// padded-row variant is only worth it where the eager GEMM path exists;
    /// everywhere else (small images, non-WMMA arches) we run exactly `n`
    /// rows, where mask-free attention is already correct.
    fn tower_rows(&self, n: usize, gpu: &Gpu) -> usize {
        let wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12();
        if n >= 256 && wmma && self.config.head_dim % 64 == 0 {
            n.div_ceil(64) * 64
        } else {
            n
        }
    }

    /// Project one image's patches: `[n_tokens, text_hidden]` rows, ready to
    /// be written over the placeholder rows of the text sequence.
    pub fn encode_image(&self, gpu: &mut Gpu, img: &ProcessedImage) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let heads = cfg.n_heads;
        let kv_heads = cfg.n_kv_heads;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let eps = cfg.rms_norm_eps;
        let n = img.n_patches;
        if n == 0 {
            return Err("embeddinggemma2: image produced zero patches".into());
        }
        if img.patches.len() != n * cfg.patch_pixels() || img.positions.len() != n * 2 {
            return Err(
                "embeddinggemma2: image processor output is inconsistent with its patch \
                 count"
                    .into(),
            );
        }
        let rows = self.tower_rows(n, gpu);
        // ── prologue: patch embed on the host, one upload ──────────────
        // `2*(p - 0.5)` is folded into the copy; padding rows (past `n`) are
        // zero and their outputs are dropped.
        let mut host_x = vec![0.0f32; rows * cfg.patch_pixels()];
        host_x[..n * cfg.patch_pixels()].copy_from_slice(&img.patches);
        for v in host_x.iter_mut() {
            *v = 2.0 * (*v - 0.5);
        }
        let mut host_pos = vec![0.0f32; rows * hidden];
        for r in 0..n {
            let x = img.positions[r * 2] as usize;
            let y = img.positions[r * 2 + 1] as usize;
            let xb = x * hidden;
            let yb = cfg.position_axis_stride() + y * hidden;
            let rb = r * hidden;
            for d in 0..hidden {
                host_pos[rb + d] = self.position_table[xb + d] + self.position_table[yb + d];
            }
        }

        let mut e = Enc::new(gpu);
        let patch_x = e.upload(&host_x, rows * cfg.patch_pixels())?;
        let hidden_state = e.gemm(
            &self.patch_in_proj,
            &patch_x,
            hidden,
            cfg.patch_pixels(),
            rows,
        )?;
        e.free(patch_x)?;
        let pos = e.upload(&host_pos, rows * hidden)?;
        e.add_inplace(&hidden_state, &pos, "vision patch positions")?;
        e.free(pos)?;

        // RoPE tables for this image (per-patch, permuted head order).
        let (cos_h, sin_h) = rope_tables(cfg, &img.positions, n, rows);
        let cos = e.upload(&cos_h, rows * hd)?;
        let sin = e.upload(&sin_h, rows * hd)?;

        for (li, layer) in self.layers.iter().enumerate() {
            let l = |msg: &str| format!("vision layer {li}: {msg}");

            let normed = e.alloc(rows * hidden)?;
            e.rmsnorm(&hidden_state, &layer.input_ln, &normed, rows, hidden, eps)
                .map_err(|err| l(&err))?;

            let mut q = e.gemm(&layer.q_proj, &normed, qd, hidden, rows)?;
            let k = e.gemm(&layer.k_proj, &normed, kvd, hidden, rows)?;
            let v = e.gemm(&layer.v_proj, &normed, kvd, hidden, rows)?;
            e.free(normed)?;

            e.rmsnorm(&q, &layer.q_norm, &q, rows * heads, hd, eps)
                .map_err(|err| l(&err))?;
            e.rmsnorm(&k, &layer.k_norm, &k, rows * kv_heads, hd, eps)
                .map_err(|err| l(&err))?;
            e.rmsnorm(&v, &self.v_ones, &v, rows * kv_heads, hd, eps)
                .map_err(|err| l(&err))?;

            // Axial 2-D RoPE (H-H-W-W) on Q and K only, in the permuted
            // head-dim order the weights were uploaded in.
            e.rope_2d(&q, &k, &cos, &sin, rows, heads, kv_heads, hd)
                .map_err(|err| l(&err))?;

            // Eager path masks keys past `n` (the padded rows); the fallback
            // runs rows == n and needs no mask. Attention scale is 1.0 — the
            // QK norms replace 1/sqrt(d).
            let ctx = e.alloc(rows * qd)?;
            e.attention(&mut q, &k, &v, &ctx, rows, n, heads, kv_heads, hd, qd, kvd)
                .map_err(|err| l(&err))?;
            e.free(q)?;
            e.free(k)?;
            e.free(v)?;

            let o = e.gemm(&layer.o_proj, &ctx, hidden, qd, rows)?;
            e.free(ctx)?;
            e.rmsnorm(&o, &layer.post_attn_ln, &o, rows, hidden, eps)
                .map_err(|err| l(&err))?;
            e.add_inplace(&hidden_state, &o, &l("residual attn"))?;
            e.free(o)?;
            e.trace(
                || format!("vision layer {li} post-attn"),
                &hidden_state,
                rows * hidden,
            )?;

            let normed2 = e.alloc(rows * hidden)?;
            e.rmsnorm(&hidden_state, &layer.pre_ff_ln, &normed2, rows, hidden, eps)
                .map_err(|err| l(&err))?;
            let g = e.gemm(
                &layer.gate_proj,
                &normed2,
                cfg.intermediate_size,
                hidden,
                rows,
            )?;
            let u = e.gemm(
                &layer.up_proj,
                &normed2,
                cfg.intermediate_size,
                hidden,
                rows,
            )?;
            e.free(normed2)?;
            e.gelu_tanh(&g, rows * cfg.intermediate_size)
                .map_err(|err| l(&err))?;
            e.mul(&g, &u, &g).map_err(|err| l(&err))?;
            e.free(u)?;
            let d = e.gemm(&layer.down_proj, &g, hidden, cfg.intermediate_size, rows)?;
            e.free(g)?;
            e.rmsnorm(&d, &layer.post_ff_ln, &d, rows, hidden, eps)
                .map_err(|err| l(&err))?;
            e.add_inplace(&hidden_state, &d, &l("residual mlp"))?;
            e.free(d)?;
        }
        e.free(cos)?;
        e.free(sin)?;

        // ── pooler + embedder (host pooling, exact f32) ────────────────
        let raw = e.download(&hidden_state)?;
        e.free(hidden_state)?;
        let pooled = pool_bins(&raw, n, img, cfg)?;

        let pooled_t = e.upload(&pooled, img.n_tokens * hidden)?;
        e.rmsnorm(
            &pooled_t,
            &self.e_ones,
            &pooled_t,
            img.n_tokens,
            hidden,
            eps,
        )?;
        let emb = e.gemm(
            &self.embed_projection,
            &pooled_t,
            self.text_hidden,
            hidden,
            img.n_tokens,
        )?;
        e.free(pooled_t)?;
        let out = e.download(&emb)?;
        e.free(emb)?;
        Ok(out)
    }
}

/// Project every processed image through the tower, in order: one
/// `[n_tokens, text_hidden]` block per image.
pub fn encode_images(
    gpu: &mut Gpu,
    tower: &VisionTower,
    media: &[ProcessedImage],
) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(media.len());
    for m in media {
        out.push(tower.encode_image(gpu, m)?);
    }
    Ok(out)
}

/// Compute `ProjectedImage`-side pooling: 3×3 mean over each bin, times
/// `sqrt(hidden)`, over the first `n` rows of `raw` (`[rows, hidden]`).
/// Only real patches contribute — padding rows are simply never read.
fn pool_bins(
    raw: &[f32],
    n: usize,
    img: &ProcessedImage,
    cfg: &VisionConfig,
) -> Result<Vec<f32>, String> {
    let hidden = cfg.hidden_size;
    let k = cfg.pooling_kernel_size;
    let bins_x = img.grid_w / k;
    let bins_y = img.grid_h / k;
    if bins_x * bins_y != img.n_tokens {
        return Err(format!(
            "embeddinggemma2: bin grid {bins_y}x{bins_x} does not match the processor's \
             {} tokens",
            img.n_tokens
        ));
    }
    let mut pooled = vec![0.0f32; img.n_tokens * hidden];
    for np in 0..n {
        let px = np % img.grid_w;
        let py = np / img.grid_w;
        let bin = (px / k) + bins_x * (py / k);
        let src = np * hidden;
        let dst = bin * hidden;
        for d in 0..hidden {
            pooled[dst + d] += raw[src + d];
        }
    }
    let inv = 1.0f32 / (k * k) as f32;
    let root = (hidden as f32).sqrt();
    for v in pooled.iter_mut() {
        *v *= inv * root;
    }
    Ok(pooled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_perm_lands_the_gemma_pairs_on_the_kernel_pairs() {
        for hd in [64usize, 128] {
            let p = axial_perm(hd);
            assert_eq!(p.len(), hd);
            let mut seen = p.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..hd).collect::<Vec<_>>(), "not a permutation");

            let q = hd / 4;
            let half = hd / 2;
            // The kernel rotates the pair (d, d + half). It must carry exactly
            // HF's pair: `rotate_half` inside the gemma x half pairs
            // (j, j + q), inside the y half (half + j, half + q + j).
            for d in 0..half {
                let (a, b) = (p[d], p[d + half]);
                let want = if d < q {
                    (d, d + q) // x half
                } else {
                    (d - q + half, d - q + half + q) // y half
                };
                assert_eq!((a, b), want, "kernel pair {d} -> gemma {a},{b}");
            }
            // Kernel quarters are [x, y, x, y]; the gemma halves are the
            // [x, x] / [y, y] split the rope tables read (z / q < 2 → x).
            for d in 0..hd {
                assert_eq!((p[d] / q < 2), d / q % 2 == 0);
            }
        }
    }

    #[test]
    fn inv_freq_matches_the_checkpoint_formula() {
        // inv_freq[i] = 100^(-2i/32) for i in 0..16.
        let inv = axial_inv_freq(64, 100.0);
        assert_eq!(inv.len(), 16);
        assert!((inv[0] - 1.0).abs() < 1e-7);
        for (i, v) in inv.iter().enumerate() {
            let want = 100f32.powf(-((2 * i) as f32) / 32.0);
            assert!((v - want).abs() < 1e-9, "i={i}");
        }
    }

    /// HF's own rotation, transcribed from `modeling_gemma4.py`
    /// (`apply_multidimensional_rope` + `rotate_half` +
    /// `recomposition_frequencies`), on a gemma-order head vector.
    fn hf_rope(x: &[f32], xpos: f32, ypos: f32, inv: &[f32]) -> Vec<f32> {
        let hd = x.len();
        let half = hd / 2;
        let q = hd / 4;
        let mut out = vec![0.0f32; hd];
        for (part, pos) in [(0usize, xpos), (1, ypos)] {
            let cos: Vec<f32> = inv.iter().map(|f| (pos * f).cos()).collect();
            let sin: Vec<f32> = inv.iter().map(|f| (pos * f).sin()).collect();
            let base = part * half;
            // rotate_half on the half: x1 = half[:q], x2 = half[q:] where the
            // frequencies are laid out [h, h] / [w, w] (q values each).
            let xp = &x[base..base + half];
            for j in 0..q {
                out[base + j] = xp[j] * cos[j] - xp[q + j] * sin[j];
                out[base + q + j] = xp[q + j] * cos[j] + xp[j] * sin[j];
            }
        }
        out
    }

    /// The corrected permutation, the permuted tables and the halfsplit kernel
    /// must reproduce HF's rotation exactly — this is the rule the fix pins.
    #[test]
    fn rope_tables_plus_perm_reproduce_the_hf_rotation() {
        let hd = 64;
        let inv = axial_inv_freq(hd, 100.0);
        let perm = axial_perm(hd);
        let half = hd / 2;
        // A handful of positions, including the ones the parity cases use
        // (48x48 → x,y ≤ 47; 3x840 → x ≤ 839) plus a non-multiple of the
        // pooling grid.
        let positions: [(i32, i32); 7] = [
            (0, 0),
            (1, 0),
            (5, 3),
            (47, 47),
            (41, 56),
            (65, 35),
            (839, 2),
        ];
        let mut pos_flat = Vec::new();
        for (x, y) in positions {
            pos_flat.push(x);
            pos_flat.push(y);
        }
        let cfg = test_config(hd);
        let n = positions.len();
        let (cos, sin) = rope_tables(&cfg, &pos_flat, n, n);

        let x_in: Vec<f32> = (0..hd)
            .map(|i| ((i * 37 % 101) as f32) / 101.0 - 0.5)
            .collect();
        // gemma index → kernel index (the inverse of `perm`).
        let mut rev = vec![0usize; hd];
        for d in 0..hd {
            rev[perm[d]] = d;
        }
        for r in 0..n {
            let (xpos, ypos) = (positions[r].0 as f32, positions[r].1 as f32);
            let want = hf_rope(&x_in, xpos, ypos, &inv);
            // gemma order → kernel order.
            let xk: Vec<f32> = (0..hd).map(|d| x_in[perm[d]]).collect();
            let mut got = vec![0.0f32; hd];
            for d in 0..half {
                let e = d + half;
                let (cd, sd) = (cos[r * hd + d], sin[r * hd + d]);
                let (ce, se) = (cos[r * hd + e], sin[r * hd + e]);
                got[d] = xk[d] * cd - xk[e] * sd;
                got[e] = xk[e] * ce + xk[d] * se;
            }
            // kernel order → gemma order.
            for d in 0..hd {
                assert!(
                    (got[rev[d]] - want[d]).abs() < 1e-5,
                    "row {r} (x={xpos}, y={ypos}) dim {d}: {} vs {}",
                    got[rev[d]],
                    want[d]
                );
            }
            // The kernel's per-side tables must agree within a pair (it reads
            // both, and a mismatch would make the pair a shear, not a rotation).
            for d in 0..half {
                assert!((cos[r * hd + d] - cos[r * hd + d + half]).abs() < 1e-6);
                assert!((sin[r * hd + d] - sin[r * hd + d + half]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rope_tables_place_the_axes_on_the_right_dims() {
        let cfg = test_config(64);
        // Patch 0 at (x=1, y=0): the x axis must show up on the kernel's x
        // quarters (0..16 and 32..48) and the (zero) y axis on the others.
        let positions = [1i32, 0i32];
        let (cos, sin) = rope_tables(&cfg, &positions, 1, 2);
        assert_eq!(cos.len(), 2 * 64);
        let inv = axial_inv_freq(64, 100.0);
        // Kernel dim 0 holds gemma dim 0 → x * inv_freq[0] = 1.0.
        assert!((cos[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[0] - 1.0f32.sin()).abs() < 1e-6);
        // Kernel dim 32 holds gemma dim 16 (the x half's second quarter) →
        // x * inv_freq[0].
        assert!((cos[32] - 1.0f32.cos()).abs() < 1e-6);
        assert!((sin[32] - 1.0f32.sin()).abs() < 1e-6);
        // Kernel dim 8 holds gemma dim 8 → x * inv_freq[8].
        let want = (1.0 * inv[8]).cos();
        assert!((cos[8] - want).abs() < 1e-6);
        // Kernel dim 16 holds gemma dim 32 — the y axis, and y = 0 → cos 1.
        assert!((cos[16] - 1.0).abs() < 1e-6);
        assert!((sin[16]).abs() < 1e-6);
        // Rows past `n` are zero (padding rows).
        assert_eq!(&cos[64..], &[0.0f32; 64]);
    }

    /// The checkpoint's vision config, the one every kernel-adjacent test
    /// shares.
    fn test_config(head_dim: usize) -> VisionConfig {
        VisionConfig {
            hidden_size: 768,
            num_layers: 16,
            intermediate_size: 3072,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rope_theta: 100.0,
            max_soft_tokens: 280,
        }
    }

    #[test]
    fn pool_bins_means_nine_patches_and_scales_by_root_hidden() {
        let cfg = VisionConfig {
            hidden_size: 768,
            num_layers: 16,
            intermediate_size: 3072,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rope_theta: 100.0,
            max_soft_tokens: 280,
        };
        // A 2x1 bin grid (6x3 patches), hidden 3 for arithmetic that fits a
        // comment: every patch row is a constant, so each bin's mean is the
        // average of its nine rows.
        let grid_h = 3;
        let grid_w = 6;
        let n = grid_h * grid_w;
        let small = VisionConfig {
            hidden_size: 3,
            ..cfg
        };
        let img = ProcessedImage {
            patches: Vec::new(),
            positions: vec![0; n * 2],
            grid_h,
            grid_w,
            n_patches: n,
            n_tokens: 2,
        };
        let mut raw = vec![0.0f32; n * 3];
        for np in 0..n {
            for d in 0..3 {
                raw[np * 3 + d] = np as f32;
            }
        }
        // Padding rows past `n` must not contribute.
        raw.extend_from_slice(&[1000.0f32; 6]);
        let pooled = pool_bins(&raw, n, &img, &small).expect("pool");
        assert_eq!(pooled.len(), 2 * 3);
        let scale = 1.0 / 9.0 * 3f32.sqrt();
        // Bin 0 = columns 0..2 of rows 0..2 → patches {0,1,2, 6,7,8, 12,13,14}.
        let bin0: f32 = [0, 1, 2, 6, 7, 8, 12, 13, 14]
            .iter()
            .map(|v| *v as f32)
            .sum();
        // Bin 1 = columns 3..5 of rows 0..2 → {3,4,5, 9,10,11, 15,16,17}.
        let bin1: f32 = [3, 4, 5, 9, 10, 11, 15, 16, 17]
            .iter()
            .map(|v| *v as f32)
            .sum();
        for d in 0..3 {
            assert!((pooled[d] - bin0 / 9.0 * 3f32.sqrt()).abs() < 1e-4);
            assert!((pooled[3 + d] - bin1 / 9.0 * 3f32.sqrt()).abs() < 1e-4);
        }
        assert!((pooled[0] - bin0 * scale).abs() < 1e-4);
    }

    #[test]
    fn pool_bins_refuses_a_grid_that_does_not_divide_into_bins() {
        let cfg = VisionConfig {
            hidden_size: 2,
            num_layers: 1,
            intermediate_size: 8,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 8,
            rope_theta: 100.0,
            max_soft_tokens: 70,
        };
        let img = ProcessedImage {
            patches: Vec::new(),
            positions: vec![0; 8],
            grid_h: 2,
            grid_w: 4,
            n_patches: 8,
            n_tokens: 3, // wrong: 2x4 grids to (3*4)/9 = 0 bins
        };
        assert!(pool_bins(&vec![0.0; 16], 8, &img, &cfg).is_err());
    }
}
