// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Gemma-4 audio tower: log-mel frames → `[M, text_hidden]` rows ready
//! to splice into the trunk's embedding stream.
//!
//! Formula-level ground truth: `docs/plans/embeddinggemma2-tower-spec.md`
//! §AUDIO (transformers 5.19.0 `models/gemma4/modeling_gemma4.py`,
//! `Gemma4AudioModel` / `Gemma4AudioLayer` / `Gemma4AudioAttention`).
//!
//! Forward (f32 activations, f16 WMMA weights through the crate's `Enc`
//! dispatch — the same numerics contract as the text trunk and the vision
//! tower):
//!
//! - **Subsample conv stack**: two `Conv2d(k=3, s=2, p=1)` stages, each with
//!   a time mask, LayerNorm (affine, no bias, eps = `rms_norm_eps`) and ReLU,
//!   over the mel as `(1, 1, frames, mel_bins)`; then `[T', F'·C]` flattened
//!   `f*C + c`. This runs on the HOST in exact f32: it is ~1 % of the tower's
//!   MACs (≈ 21 M for an 11 s clip) and the two time masks are host-side
//!   truth, so a device kernel plus an im2col round trip would buy noise.
//!   The stage-0 input mask is a no-op by construction — the frontend already
//!   zeroes exactly the frames the mask marks invalid — while the halved mask
//!   at stage 1 is applied for real (those rows are nonzero).
//! - **`input_proj_linear`** (hidden→hidden, no activation, no clip: a plain
//!   `nn.Linear` in the reference).
//! - **12 conformer layers**: macaron FFN₁ → attention → light conv → macaron
//!   FFN₂ → output norm (NOT residual-added).
//! - **Every clippable linear** (`use_clipped_linears`, true on this
//!   checkpoint: 10 per layer) clamps its INPUT to the checkpoint's
//!   calibrated bounds and its output likewise; when the flag is false no
//!   bound tensor is read and no clamp runs. The reference's
//!   `gradient_clipping` clamps (`min(1e10, dtype max)` = 1e10 here) are KEPT
//!   as real launches: they are inert for any finite f32 activation (1e10 is
//!   past the point where the forward is meaningless), but they are part of
//!   the reference's semantics and cost one elementwise pass.
//! - **Attention** (chunk 12, left context 12, right 0, block context 24):
//!   `q = q_proj(h)·head_dim^-0.5·softplus(per_dim_scale)·ln2⁻¹`,
//!   `k = k_proj(h)·ln(1+e)·ln2⁻¹`, then a blocked Transformer-XL-style
//!   relative-position bias, the `/50 → tanh → ×50` softcap, the reference's
//!   5-D mask, and an f32 softmax over the 24-wide block context. The mask is
//!   INVERTED relative to its own definition and that inversion is reproduced
//!   faithfully: the reference forces `attn_implementation="eager"`, whose
//!   mask interface (`masking_utils.eager_mask`) converts the boolean
//!   `padding ∧ sliding-window` mask to an additive 0/−inf float mask, while
//!   `Gemma4AudioAttention.forward` fills with the Bool-mask idiom
//!   `masked_fill(attention_mask.logical_not(), −1e9)`. A slot therefore keeps
//!   its score iff it is inside the sequence AND OUTSIDE the causal window `0
//!   <= q − kv < context_left − 1`. See [`attention`] for the derivation and
//!   `examples/audio_lab.rs` for the instrument that pins it. The four
//!   projections are `Enc` GEMMs; the blocked attention itself runs on the
//!   HOST — 0.2 % of the layer's MACs (≈ 14 M) and pure index arithmetic, where
//!   a device kernel would be a 5-D layout with no oracle here. The tower runs
//!   the WHOLE subsampled sequence (`steps` rows) and returns its first `M`
//!   rows: the inversion above is what makes that necessary — a valid row keys
//!   into the padded tail, and a tail row's own attention output becomes the
//!   next layer's key/value row — so truncating to `M` before the forward
//!   drops real keys (≈15 % rel-L2 on the clips where `M < steps`, i.e. where
//!   the frame count leaves a padded quantum on the subsampled axis).
//! - **Tail**: `output_proj` (+bias, no activation) → the audio embedder
//!   (scale-less RMSNorm over `output_proj_dims` → `embed_audio`'s
//!   `[text_hidden, output_proj_dims]` projection) → `[M, text_hidden]`.
//!
//! Three elementwise kernels the shared dispatch does not carry are JIT'd
//! from this module's own source (`eg2_audio_ops`): the clip clamp, the GLU,
//! and the causal depthwise conv. Everything else is a shared `Enc` GEMM or
//! `rdna-compute` kernel.

use crate::audio::ProcessedAudio;
use crate::config::AudioConfig;
use crate::forward::Enc;
use crate::names;
use crate::weights::{decode_f16, decode_f32, upload_f16_words, upload_f32_slice};
use half::f16;
use hip_bridge::KernargBlob;
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::{Gpu, GpuTensor};
use std::ffi::c_void;

// ─── the three kernels the tower needs beyond the shared dispatch ──────────

const AUDIO_MODULE: &str = "eg2_audio_ops";
const CLAMP_FN: &str = "eg2_audio_clamp_f32";
const GLU_FN: &str = "eg2_audio_glu_f32";
const DWCONV_FN: &str = "eg2_audio_causal_dwconv_f32";

/// `y = clamp(x, lo, hi)`; in place when `x == y`.
const AUDIO_KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void eg2_audio_clamp_f32(
    const float* __restrict__ x, float* __restrict__ y, int n, float lo, float hi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

// GLU over the last dim: y[r, j] = x[r, j] / (1 + exp(-x[r, half + j])).
extern "C" __global__ void eg2_audio_glu_f32(
    const float* __restrict__ x, float* __restrict__ y, int rows, int half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * half) return;
    int r = i / half;
    int j = i - r * half;
    float a = x[r * 2 * half + j];
    float b = x[r * 2 * half + half + j];
    y[i] = a / (1.0f + expf(-b));
}

// Causal depthwise conv: y[t, c] = sum_j w[c, j] * x[t - (k-1) + j, c],
// with x out of range read as zero (the reference's left pad of k-1).
extern "C" __global__ void eg2_audio_causal_dwconv_f32(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
    int tokens, int channels, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * channels) return;
    int t = i / channels;
    int c = i - t * channels;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        int ti = t - (k - 1) + j;
        if (ti >= 0) acc += x[ti * channels + c] * w[c * k + j];
    }
    y[i] = acc;
}
"#;

fn launch_1d(
    gpu: &mut Gpu,
    func: &str,
    blob: &mut KernargBlob,
    work: usize,
    what: &str,
) -> Result<(), String> {
    gpu.ensure_kernel_public(AUDIO_MODULE, AUDIO_KERNEL_SRC, func)
        .map_err(|e| format!("embeddinggemma2: {what}: kernel: {e:?}"))?;
    let grid = ((work as u64).div_ceil(256)).max(1) as u32;
    blob.pad_to(16);
    gpu.launch_kernel_blob(func, [grid, 1, 1], [256, 1, 1], 0, blob.as_mut_slice())
        .map_err(|e| format!("embeddinggemma2: {what}: {e:?}"))
}

/// `y = clamp(x, c.in_min, c.in_max)` (in place when `x == y`).
fn clamp_launch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
    c: Clip,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let mut blob = KernargBlob::new();
    blob.push_ptr(x.buf.as_ptr() as *const c_void);
    blob.push_ptr(y.buf.as_ptr() as *const c_void);
    blob.push_i32(n as i32);
    blob.push_f32(c.in_min);
    blob.push_f32(c.in_max);
    launch_1d(gpu, CLAMP_FN, &mut blob, n, "audio clamp")
}

/// `y = glu(x)` over the last dim: `[rows, 2*half] → [rows, half]`.
fn glu_launch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    half: usize,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let mut blob = KernargBlob::new();
    blob.push_ptr(x.buf.as_ptr() as *const c_void);
    blob.push_ptr(y.buf.as_ptr() as *const c_void);
    blob.push_i32(rows as i32);
    blob.push_i32(half as i32);
    launch_1d(gpu, GLU_FN, &mut blob, rows * half, "audio glu")
}

/// The causal depthwise conv, left pad `k-1`.
fn dwconv_launch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &GpuTensor,
    y: &GpuTensor,
    tokens: usize,
    channels: usize,
    k: usize,
) -> Result<(), String> {
    if tokens == 0 {
        return Ok(());
    }
    let mut blob = KernargBlob::new();
    blob.push_ptr(x.buf.as_ptr() as *const c_void);
    blob.push_ptr(w.buf.as_ptr() as *const c_void);
    blob.push_ptr(y.buf.as_ptr() as *const c_void);
    blob.push_i32(tokens as i32);
    blob.push_i32(channels as i32);
    blob.push_i32(k as i32);
    launch_1d(
        gpu,
        DWCONV_FN,
        &mut blob,
        tokens * channels,
        "audio causal dwconv",
    )
}

// ─── weights ───────────────────────────────────────────────────────────────

/// One clippable linear's calibrated bounds (four 0-d scalars per linear).
///
/// `pub(crate)`: no public signature takes one — the loader builds them from
/// the checkpoint and the forward applies them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Clip {
    pub in_min: f32,
    pub in_max: f32,
    pub out_min: f32,
    pub out_max: f32,
}

impl Clip {
    /// The same bounds as an input clamp, for the GEMM's output.
    fn out(self) -> Clip {
        Clip {
            in_min: self.out_min,
            in_max: self.out_max,
            out_min: self.out_min,
            out_max: self.out_max,
        }
    }
}

/// A macaron feed-forward block: `clamp → rmsnorm → lin1 → silu → lin2 →
/// clamp → rmsnorm → ×residual_weight → +residual`.
struct MacaronFfn {
    lin1: GpuTensor,
    lin2: GpuTensor,
    pre_norm: GpuTensor,
    post_norm: GpuTensor,
    lin1_clip: Option<Clip>,
    lin2_clip: Option<Clip>,
}

impl MacaronFfn {
    fn into_tensors(self) -> [GpuTensor; 4] {
        [self.lin1, self.lin2, self.pre_norm, self.post_norm]
    }
}

struct AudioLayer {
    ff1: MacaronFfn,
    ff2: MacaronFfn,
    lconv_pre_norm: GpuTensor,
    lconv_lin_start: GpuTensor,
    lconv_lin_end: GpuTensor,
    /// f32 `[hidden * k]` — the depthwise causal conv's weights, host f32
    /// (1 k values: an f16 device copy would only add a convert).
    lconv_depthwise: GpuTensor,
    lconv_conv_norm: GpuTensor,
    lconv_start_clip: Option<Clip>,
    lconv_end_clip: Option<Clip>,
    q_proj: GpuTensor,
    k_proj: GpuTensor,
    v_proj: GpuTensor,
    attn_post: GpuTensor,
    relative_k_proj: GpuTensor,
    q_clip: Option<Clip>,
    k_clip: Option<Clip>,
    v_clip: Option<Clip>,
    post_clip: Option<Clip>,
    /// HOST `[head_dim]`: `head_dim^-0.5 · ln2⁻¹ · softplus(per_dim_scale)`.
    /// Folded at load so the attention's q scaling is free and the softplus
    /// runs where its threshold semantics are exact.
    q_scale: Vec<f32>,
    norm_pre_attn: GpuTensor,
    norm_post_attn: GpuTensor,
    norm_out: GpuTensor,
}

/// The subsample conv stack's host weights (`subsampling_conv_channels`).
struct SubSample {
    /// `[c0, 1, 3, 3]` flattened `[c0 * 9]`.
    w0: Vec<f32>,
    n0: Vec<f32>,
    /// `[c1, c0, 3, 3]` flattened `[c1 * c0 * 9]`.
    w1: Vec<f32>,
    n1: Vec<f32>,
}

impl SubSample {
    /// `[frames, bins]` mel (invalid trailing frames already zero) →
    /// `[steps, hidden]` tokens, where `steps` is the whole subsampled time
    /// axis: the mask's survivors are a PREFIX of it, so the answer's `M`
    /// rows are its first `M` — but the reference keeps the rest in play as
    /// keys, so they are returned too.
    fn forward(
        &self,
        cfg: &AudioConfig,
        bins: usize,
        clip: &ProcessedAudio,
    ) -> Result<Vec<f32>, String> {
        let frames = clip.plan.frames;
        let valid = clip.plan.valid_frames;
        let rows = clip.plan.n_tokens;
        let c0 = cfg.subsampling_conv_channels[0];
        let c1 = cfg.subsampling_conv_channels[1];
        if frames == 0 || rows == 0 {
            return Err("embeddinggemma2: audio subsample got no frames".into());
        }
        if self.w0.len() != c0 * 9 || self.w1.len() != c1 * c0 * 9 {
            return Err("embeddinggemma2: audio subsample conv weights are the wrong shape".into());
        }
        let f0 = (bins - 1) / 2 + 1;
        let t0 = (frames - 1) / 2 + 1;
        let f1 = (f0 - 1) / 2 + 1;
        let t1 = (t0 - 1) / 2 + 1;
        if f1 * c1 != cfg.hidden_size {
            return Err(format!(
                "embeddinggemma2: two stride-2 convs over {bins} mel bins produce {f1}x{c1} \
                 = {} flattened values, not the tower's hidden width {}",
                f1 * c1,
                cfg.hidden_size
            ));
        }

        // ── stage 0: conv → LayerNorm → ReLU, then the halved time mask ──
        // (The reference masks the input BEFORE the conv; the frontend has
        // already zeroed exactly those frames, so that multiply is a no-op
        // and its only effect — the mask it returns — is applied below.)
        let mut a = vec![0.0f32; t0 * f0 * c0];
        for ti in 0..t0 {
            for wi in 0..f0 {
                let pos = (ti * f0 + wi) * c0;
                for oc in 0..c0 {
                    let mut acc = 0.0f32;
                    for dy in 0..3usize {
                        let hi = ti as isize * 2 + dy as isize - 1;
                        if hi < 0 || hi as usize >= frames {
                            continue;
                        }
                        for dx in 0..3usize {
                            let wj = wi as isize * 2 + dx as isize - 1;
                            if wj < 0 || wj as usize >= bins {
                                continue;
                            }
                            acc += clip.mel[hi as usize * bins + wj as usize]
                                * self.w0[oc * 9 + dy * 3 + dx];
                        }
                    }
                    a[pos + oc] = acc;
                }
                layernorm_relu(&mut a[pos..pos + c0], &self.n0, cfg.rms_norm_eps);
            }
        }
        // `mask[:, ::2]`: zero the whole channel row where frame 2*ti is not
        // real. Those rows ARE nonzero (a conv tap can reach a valid frame).
        for ti in 0..t0 {
            if ti * 2 >= valid {
                let pos = ti * f0 * c0;
                a[pos..pos + f0 * c0].fill(0.0);
            }
        }

        // ── stage 1: conv → LayerNorm → ReLU ─────────────────────────────
        // (No mask after the last stage — the reference returns the mask but
        // nothing consumes it; the survivors are selected by index instead.)
        let mut b = vec![0.0f32; t1 * f1 * c1];
        for ti in 0..t1 {
            for wi in 0..f1 {
                let pos = (ti * f1 + wi) * c1;
                for oc in 0..c1 {
                    let mut acc = 0.0f32;
                    for dy in 0..3usize {
                        let hi = ti as isize * 2 + dy as isize - 1;
                        if hi < 0 || hi as usize >= t0 {
                            continue;
                        }
                        for dx in 0..3usize {
                            let wj = wi as isize * 2 + dx as isize - 1;
                            if wj < 0 || wj as usize >= f0 {
                                continue;
                            }
                            let src = (hi as usize * f0 + wj as usize) * c0;
                            // The weight is [oc][ic][dy][dx]: `ic = 0`'s base,
                            // then `+ ic * 9` walks the input channels.
                            let wbase = oc * c0 * 9 + dy * 3 + dx;
                            for ic in 0..c0 {
                                acc += a[src + ic] * self.w1[wbase + ic * 9];
                            }
                        }
                    }
                    b[pos + oc] = acc;
                }
                layernorm_relu(&mut b[pos..pos + c1], &self.n1, cfg.rms_norm_eps);
            }
        }

        // The `(t, f, c)` layout IS the flatten (`f*C + c`), so the tokens are
        // the leading values in time order. The WHOLE `t1` sequence is
        // returned, not just the `M` valid rows: the reference runs its tower
        // (and its attention mask) over the full subsampled length, and the
        // inverted mask lets a valid row key into the padded tail — see
        // [`attention`] and [`AudioTower::encode_audio`].
        Ok(b)
    }
}

/// `LayerNorm(affine, no bias)` followed by ReLU, on one position vector.
fn layernorm_relu(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        let y = (*v - mean) * inv * gamma[i];
        *v = if y > 0.0 { y } else { 0.0 };
    }
}

/// The loaded audio tower (the `.vl` sidecar's `audio_tower.*` +
/// `embed_audio.*` groups).
pub struct AudioTower {
    pub config: AudioConfig,
    /// The frontend's mel width — the subsample stack's input width.
    mel_bins: usize,
    conv: SubSample,
    /// `[hidden, subsample_input_dim]` f16.
    input_proj: GpuTensor,
    layers: Vec<AudioLayer>,
    /// `[output_proj_dims, hidden]` f16 + `[output_proj_dims]` f32 bias.
    output_proj: GpuTensor,
    output_bias: GpuTensor,
    /// `[text_hidden, output_proj_dims]` f16 (`embed_audio`).
    embed_projection: GpuTensor,
    /// Ones f32 `[output_proj_dims]` for the embedder's scale-less norm.
    e_ones: GpuTensor,
    /// HOST `[n_rel, hidden]`: `[sin | cos]` of `pos_id · inv_timescale`
    /// with `pos_id = context_size/2 … 0`.
    rel_pos: Vec<f32>,
    /// The trunk's hidden width — the embedder's output width.
    text_hidden: usize,
}

impl AudioTower {
    /// The `.vl` sidecar probe every caller uses: presence of the tower's
    /// output projection.
    pub fn sidecar_has_tower(src: &ModelSource) -> bool {
        names::tower_has(src, "audio_tower.output_proj.weight")
    }

    /// Free every GPU allocation (the host weights drop with the struct).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let AudioTower {
            conv: _,
            input_proj,
            layers,
            output_proj,
            output_bias,
            embed_projection,
            e_ones,
            ..
        } = self;
        for l in layers {
            let AudioLayer {
                ff1,
                ff2,
                lconv_pre_norm,
                lconv_lin_start,
                lconv_lin_end,
                lconv_depthwise,
                lconv_conv_norm,
                q_proj,
                k_proj,
                v_proj,
                attn_post,
                relative_k_proj,
                q_scale: _,
                norm_pre_attn,
                norm_post_attn,
                norm_out,
                ..
            } = l;
            for t in [
                lconv_pre_norm,
                lconv_lin_start,
                lconv_lin_end,
                lconv_depthwise,
                lconv_conv_norm,
                q_proj,
                k_proj,
                v_proj,
                attn_post,
                relative_k_proj,
                norm_pre_attn,
                norm_post_attn,
                norm_out,
            ] {
                let _ = gpu.free_tensor(t);
            }
            for t in ff1.into_tensors().into_iter().chain(ff2.into_tensors()) {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in [
            input_proj,
            output_proj,
            output_bias,
            embed_projection,
            e_ones,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Project one clip: `[M, text_hidden]` rows, ready to be written over
    /// the placeholder rows of the text sequence.
    pub fn encode_audio(&self, gpu: &mut Gpu, clip: &ProcessedAudio) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let rows = clip.plan.n_tokens;
        if rows == 0 {
            return Err(
                "embeddinggemma2: the audio clip produced zero soft tokens (too short for \
                 one mel frame?)"
                    .into(),
            );
        }
        if clip.mel.len() != clip.plan.frames * self.mel_bins {
            return Err(format!(
                "embeddinggemma2: audio frontend produced {} mel values for {} frames of {} \
                 bins",
                clip.mel.len(),
                clip.plan.frames,
                self.mel_bins
            ));
        }
        if clip.plan.n_tokens > clip.plan.frames {
            return Err(format!(
                "embeddinggemma2: audio token count {} exceeds its {} mel frames",
                clip.plan.n_tokens, clip.plan.frames
            ));
        }

        // ── subsample conv stack (host, exact f32) ─────────────────────
        let host_x = self.conv.forward(cfg, self.mel_bins, clip)?;
        // The reference runs its tower over the WHOLE subsampled sequence
        // (`steps` rows), not only the `M` valid ones: its attention mask is
        // inverted, so a valid row keys into the padded tail, and the tail rows'
        // own outputs feed the next layer's keys. The rows past `M` are computed
        // and then dropped — the returned block is truncated to `M` at the end
        // (see `attention` for the rule that needs them).
        let steps = host_x.len() / hidden;
        if steps == 0 || steps * hidden != host_x.len() || steps < rows {
            return Err(format!(
                "embeddinggemma2: audio subsample produced {} values for hidden {hidden} \
                 (expected at least {rows} whole rows)",
                host_x.len()
            ));
        }

        let mut e = Enc::new(gpu);
        let n = steps * hidden;
        {
            let x = e.upload(&host_x, n)?;
            let h = e.gemm(&self.input_proj, &x, hidden, hidden, steps)?;
            e.free(x)?;
            let mut h = h;
            // The relative-position table is constant across layers.
            let rel_pos = e.upload(&self.rel_pos, self.rel_pos.len())?;
            for (li, l) in self.layers.iter().enumerate() {
                let tag = |m: &str, err: String| format!("audio layer {li}: {m}: {err}");
                self.macaron(&mut e, &l.ff1, &mut h, steps)
                    .map_err(|x| tag("ffn1", x))?;
                self.attention_block(&mut e, l, &mut h, rows, steps, &rel_pos)
                    .map_err(|x| tag("attention", x))?;
                self.lconv_block(&mut e, l, &mut h, steps)
                    .map_err(|x| tag("lconv", x))?;
                self.macaron(&mut e, &l.ff2, &mut h, steps)
                    .map_err(|x| tag("ffn2", x))?;

                // ── output norm (no residual add) ──────────────────────
                let t = e.alloc(n)?;
                clamp_launch(e.gpu_mut(), &h, &t, n, gradient_clip(cfg))?;
                e.rmsnorm(&t, &l.norm_out, &h, steps, hidden, cfg.rms_norm_eps)
                    .map_err(|x| tag("norm_out", x))?;
                e.free(t)?;
            }
            e.free(rel_pos)?;

            // ── output projection + embedder ───────────────────────────
            // Per-row ops, so the left-over tail rows cost nothing to carry
            // and the valid prefix is what the caller gets.
            let dims = cfg.output_proj_dims;
            let out = e.gemm(&self.output_proj, &h, dims, hidden, steps)?;
            e.free(h)?;
            e.gpu_mut()
                .bias_add_f32(&out, &self.output_bias, steps, dims)
                .map_err(|err| format!("audio output_proj bias: {err:?}"))?;
            e.rmsnorm(&out, &self.e_ones, &out, steps, dims, cfg.rms_norm_eps)?;
            let emb = e.gemm(&self.embed_projection, &out, self.text_hidden, dims, steps)?;
            e.free(out)?;
            let mut result = e.download(&emb)?;
            e.free(emb)?;
            result.truncate(rows * self.text_hidden);
            Ok(result)
        }
    }

    /// One clippable linear: input clamp (when the checkpoint carries
    /// bounds) → GEMM → output clamp.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        e: &mut Enc<'_>,
        w: &GpuTensor,
        x: &GpuTensor,
        clip: Option<Clip>,
        rows: usize,
        out: usize,
        k: usize,
    ) -> Result<GpuTensor, String> {
        let y = match clip {
            Some(c) => {
                let xc = e.alloc(rows * k)?;
                clamp_launch(e.gpu_mut(), x, &xc, rows * k, c)?;
                let y = e.gemm(w, &xc, out, k, rows)?;
                e.free(xc)?;
                clamp_launch(e.gpu_mut(), &y, &y, rows * out, c.out())?;
                y
            }
            None => e.gemm(w, x, out, k, rows)?,
        };
        Ok(y)
    }

    /// A macaron FFN block, in place on `h`.
    fn macaron(
        &self,
        e: &mut Enc<'_>,
        f: &MacaronFfn,
        h: &mut GpuTensor,
        rows: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let n = rows * hidden;
        let t = e.alloc(n)?;
        clamp_launch(e.gpu_mut(), h, &t, n, gradient_clip(cfg))?;
        e.rmsnorm(&t, &f.pre_norm, &t, rows, hidden, cfg.rms_norm_eps)?;
        let a = self.linear(e, &f.lin1, &t, f.lin1_clip, rows, cfg.ffn_dim(), hidden)?;
        e.free(t)?;
        e.gpu_mut()
            .silu_f32(&a, &a)
            .map_err(|err| format!("silu: {err:?}"))?;
        let b = self.linear(e, &f.lin2, &a, f.lin2_clip, rows, hidden, cfg.ffn_dim())?;
        e.free(a)?;
        clamp_launch(e.gpu_mut(), &b, &b, n, gradient_clip(cfg))?;
        e.rmsnorm(&b, &f.post_norm, &b, rows, hidden, cfg.rms_norm_eps)?;
        e.gpu_mut()
            .scaled_add_inplace_cpu_scalar_f32(h, &b, cfg.residual_weight)
            .map_err(|err| format!("residual: {err:?}"))?;
        e.free(b)?;
        Ok(())
    }

    /// The attention block: three clipped projections, the host attention,
    /// the clipped output projection, and the residual.
    fn attention_block(
        &self,
        e: &mut Enc<'_>,
        l: &AudioLayer,
        h: &mut GpuTensor,
        rows: usize,
        steps: usize,
        rel_pos: &GpuTensor,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let n = steps * hidden;
        let t = e.alloc(n)?;
        clamp_launch(e.gpu_mut(), h, &t, n, gradient_clip(cfg))?;
        e.rmsnorm(&t, &l.norm_pre_attn, &t, steps, hidden, cfg.rms_norm_eps)?;

        let qk = [
            (&l.q_proj, l.q_clip),
            (&l.k_proj, l.k_clip),
            (&l.v_proj, l.v_clip),
        ];
        let mut proj = Vec::with_capacity(3);
        for (w, clip) in qk {
            proj.push(self.linear(e, w, &t, clip, steps, cfg.q_dim(), hidden)?);
        }
        e.free(t)?;
        let q = e.download(&proj[0])?;
        let k = e.download(&proj[1])?;
        let v = e.download(&proj[2])?;
        for p in proj {
            e.free(p)?;
        }
        // Relative keys: `relative_k_proj` (no clip: a plain nn.Linear).
        let n_rel = self.rel_pos.len() / hidden;
        let rk = e.gemm(&l.relative_k_proj, rel_pos, hidden, hidden, n_rel)?;
        let rel = e.download(&rk)?;
        e.free(rk)?;

        // The attention answers every `steps` query rows: the tail rows are
        // not part of the answer, but they are keys for the valid rows and
        // their own outputs become the next layer's key/value rows.
        let ctx_host = attention(cfg, &l.q_scale, &q, &k, &v, &rel, rows, steps)?;
        let ctx = e.upload(&ctx_host, n)?;
        let post = self.linear(e, &l.attn_post, &ctx, l.post_clip, steps, hidden, hidden)?;
        e.free(ctx)?;
        clamp_launch(e.gpu_mut(), &post, &post, n, gradient_clip(cfg))?;
        e.rmsnorm(
            &post,
            &l.norm_post_attn,
            &post,
            steps,
            hidden,
            cfg.rms_norm_eps,
        )?;
        e.add_inplace(h, &post, "audio attention residual")?;
        e.free(post)?;
        Ok(())
    }

    /// The light depthwise convolution block, in place on `h`.
    fn lconv_block(
        &self,
        e: &mut Enc<'_>,
        l: &AudioLayer,
        h: &mut GpuTensor,
        rows: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let n = rows * hidden;
        // NB: no gradient clamp before this norm — the reference's lconv
        // block goes straight from the residual to `pre_layer_norm`.
        let pl = e.alloc(n)?;
        e.rmsnorm(h, &l.lconv_pre_norm, &pl, rows, hidden, cfg.rms_norm_eps)?;
        let start = self.linear(
            e,
            &l.lconv_lin_start,
            &pl,
            l.lconv_start_clip,
            rows,
            cfg.lconv_dim(),
            hidden,
        )?;
        e.free(pl)?;
        let g = e.alloc(n)?;
        glu_launch(e.gpu_mut(), &start, &g, rows, hidden)?;
        e.free(start)?;
        let c = e.alloc(n)?;
        dwconv_launch(
            e.gpu_mut(),
            &g,
            &l.lconv_depthwise,
            &c,
            rows,
            hidden,
            cfg.conv_kernel_size,
        )?;
        e.free(g)?;
        clamp_launch(e.gpu_mut(), &c, &c, n, gradient_clip(cfg))?;
        e.rmsnorm(&c, &l.lconv_conv_norm, &c, rows, hidden, cfg.rms_norm_eps)?;
        e.gpu_mut()
            .silu_f32(&c, &c)
            .map_err(|err| format!("silu: {err:?}"))?;
        let end = self.linear(
            e,
            &l.lconv_lin_end,
            &c,
            l.lconv_end_clip,
            rows,
            hidden,
            hidden,
        )?;
        e.free(c)?;
        e.add_inplace(h, &end, "audio lconv residual")?;
        e.free(end)?;
        Ok(())
    }
}

/// A plain dot product over two slices of the same length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// The blocked local attention, on the host.
///
/// `q`/`k`/`v` are `[steps, q_dim]` — the WHOLE subsampled sequence, not just
/// its `rows` valid prefix; `rel` is the projected relative-position table
/// `[n_rel, q_dim]`; `q_scale` is the per head-dim query scale (see
/// [`q_scale_from`]). Returns `[steps, q_dim]`.
///
/// Per head: queries are blocked by `attention_chunk_size` (the tail block is
/// zero-padded), each block's keys/values come from a `context_size`-wide
/// window starting `left-1` before the block, and the relative-bias matrix
/// gets the Transformer-XL skew (`rel_shift`) before the softcap: for query
/// `i` inside a block and window slot `j`, the bias is `R[j - i]`, where
/// `R[k]` is the embedding of position `n_rel - 1 - k` — i.e. of distance
/// `n_rel - 1 - k` back from the query. Slot `j = i` therefore carries the
/// window's oldest distance (`n_rel - 1 = left - 1`) and `j = i + n_rel - 1`
/// the zero distance (`kv == q`).
///
/// Every slot the reference's 5-D mask fills holds
/// `attention_invalid_logits_value` AFTER the softcap, exactly where the fill
/// happens, so it leaves the softmax rather than merely scoring zero. That
/// mask is separately INVERTED (see the visibility loop below: the reference
/// forces `attn_implementation="eager"`, whose mask interface hands a
/// Bool-mask idiom an additive float mask), so the slots that survive are the
/// ones OUTSIDE the mask's own definition, which is
/// `valid_key ∧ window` — the mask's query axis carries no term of its own,
/// not even for the padded-tail rows:
///
/// ```text
/// keep(q, kv) = 0 <= kv < steps                        # inside the block context
///               AND NOT (kv < rows AND window)         # ... and not the inclusion
/// ```
///
/// Both facts matter. A valid query keys into the padded TAIL
/// (`rows <= kv < steps`), which no window can include, and an invalid (tail)
/// query — which a later layer still reads as a key — inverts to "the tail
/// rows alone" rather than to an unfiltered context. The only
/// relative-position row a valid query's surviving slot can carry is `R[0]` —
/// distance `left - 1` — because every other past distance is inside the
/// window, a valid future survivor sits at `j < i` where `rel_shift` gives 0,
/// and the tail survivors are at `dist <= -rows`.
fn attention(
    cfg: &AudioConfig,
    q_scale: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rel: &[f32],
    rows: usize,
    steps: usize,
) -> Result<Vec<f32>, String> {
    let hd = cfg.head_dim;
    let heads = cfg.n_heads;
    let hw = heads * hd;
    let chunk = cfg.attention_chunk_size;
    // The block-context pad. It is ALSO the window's left extent: the
    // reference derives both from the same config field
    // (`max_past_horizon = attention_context_left - 1` and
    // `sliding_window_mask_function((attention_context_left - 1, right))`), so
    // the 24-slot block context is exactly wide enough to hold every slot a
    // query's `in_seq && !window` decision has to look at.
    let past = cfg.attention_context_left - 1;
    let right = cfg.attention_context_right;
    let ctx_size = cfg.context_size();
    let n_rel = ctx_size / 2 + 1;
    let cap = cfg.attention_logit_cap;
    let invalid = cfg.attention_invalid_logits_value;
    let k_scale = k_scale_of();
    if q.len() != steps * hw || k.len() != steps * hw || v.len() != steps * hw {
        return Err(format!(
            "inputs are not {steps}x{hw} (q {}, k {}, v {})",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if rows > steps {
        return Err(format!(
            "the {rows} valid rows exceed the {steps}-row sequence"
        ));
    }
    if rel.len() != n_rel * hw {
        return Err(format!(
            "relative keys are {} values, expected {n_rel}x{hw}",
            rel.len()
        ));
    }
    if q_scale.len() != hd {
        return Err(format!(
            "query scale has {} values, expected head_dim {hd}",
            q_scale.len()
        ));
    }

    let blocks = steps.div_ceil(chunk);
    let padded = blocks * chunk;
    // Scaled q/k, zero-padded to whole blocks.
    let mut qs = vec![0.0f32; padded * hw];
    let mut ks = vec![0.0f32; padded * hw];
    let mut vs = vec![0.0f32; padded * hw];
    for r in 0..steps {
        let base = r * hw;
        for h in 0..heads {
            let hb = h * hd;
            for d in 0..hd {
                qs[base + hb + d] = q[base + hb + d] * q_scale[d];
                ks[base + hb + d] = k[base + hb + d] * k_scale;
                vs[base + hb + d] = v[base + hb + d];
            }
        }
    }
    let mut out = vec![0.0f32; steps * hw];
    let mut scores = vec![0.0f32; ctx_size];
    let mut probs = vec![0.0f32; ctx_size];
    // Which block-context slots the reference's 5-D mask leaves surviving for
    // the current query. It does not depend on the head, so it is built once
    // per query rather than once per head.
    let mut visible = vec![false; ctx_size];
    for b in 0..blocks {
        // This block's key/value window: rows [b*chunk - past, …+ctx).
        let base = (b * chunk) as isize - past as isize;
        for i in 0..chunk {
            let qi = b * chunk + i;
            // The tail block's padding queries are not part of the sequence:
            // the reference's blocked layout computes and discards them, and so
            // does this loop.
            if qi >= steps {
                continue;
            }
            // The reference's audio attention INVERTS its 5-D mask, and the
            // inversion is load-bearing. Its code masks with
            // `attn_weights.masked_fill(attention_mask.logical_not(), -1e9)` —
            // the Bool-mask idiom ("True = attended") — but the reference
            // generator runs `attn_implementation="eager"`, and HF's
            // `eager_mask` hands the audio attention an ADDITIVE float mask
            // (`torch.where(bool_mask, 0, -inf)`, `masking_utils.py` L541).
            // `logical_not` then fires on the positions the mask means to
            // INCLUDE, so a slot survives exactly when the mask does NOT
            // include it: `valid_key && window` is what gets filled, and
            // everything else inside the block-context mask keeps its score.
            //
            // The mask's QUERY axis never enters: a dumped 4-D mask row is
            // `valid_key && window` for the invalid (padded-tail) queries too,
            // so those rows invert to "the tail rows only" rather than to an
            // unfiltered context (checked at 1 409 and 8 449 samples — the
            // latter has `M = 13 < t1 = 14`, where a query-validity term is
            // 2e-3..5e-3 rel-L2 wrong).
            //
            // Checked bit-exact against the f32 reference on all six audio
            // cases with `examples/audio_lab.rs`: this rule reproduces them to
            // ≤1e-7, while the non-inverted reading (what the tower produces
            // under `sdpa`, and any "block window" reading) is off by
            // 3e-2 .. 1.4e-1.
            for (j, vis) in visible.iter_mut().enumerate() {
                let kv = base + j as isize;
                let dist = qi as isize - kv;
                // Inside the 4-D mask's real range: the block-context pads
                // (`kv < 0`, the tail block, `kv >= steps`) are outside it, and
                // `_convert_4d_mask_to_blocked_5d` pads with `value=False`,
                // which the additive mask turns into a fill.
                let in_mask = kv >= 0 && (kv as usize) < steps;
                let valid_kv = kv >= 0 && (kv as usize) < rows;
                // `sliding_window_mask_function((context_left - 1,
                // context_right))`: `in_left` is `0 <= dist < context_left-1`
                // and `in_right` is `-right < dist < 0` (empty here, since
                // `attention_context_right = 0`).
                let in_left = dist >= 0 && dist < past as isize;
                let in_right = dist < 0 && -dist < right as isize;
                *vis = in_mask && !(valid_kv && (in_left || in_right));
            }
            for h in 0..heads {
                let qbase = qi * hw + h * hd;
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut ac = 0.0f32;
                    if visible[j] {
                        let kbase = (base + j as isize) as usize * hw + h * hd;
                        ac = dot(&qs[qbase..qbase + hd], &ks[kbase..kbase + hd]);
                    }
                    // `rel_shift`: out[i][j] = x_pad[i*ctx + j], where
                    // `x_pad` holds the n_rel raw positions in rows of ctx+1
                    // columns.
                    let flat = i * ctx_size + j;
                    let (pr, pc) = (flat / (ctx_size + 1), flat % (ctx_size + 1));
                    let mut bd = 0.0f32;
                    if pr < chunk && pc < n_rel {
                        let rbase = pc * hw + h * hd;
                        bd = dot(&qs[qbase..qbase + hd], &rel[rbase..rbase + hd]);
                    }
                    *score = if visible[j] {
                        ((ac + bd) / cap).tanh() * cap
                    } else {
                        invalid
                    };
                }
                // fp32 softmax over the block context. A query can have NO
                // surviving slot at all (the first block's mid queries: every
                // in-mask slot is inside the window), and the reference answers
                // that with a uniform distribution over the FILLED slots —
                // every score is `attention_invalid_logits_value`, so
                // `exp(s − max)` is 1 and the sum is the context width. That is
                // not a degenerate case to paper over: it is what the reference
                // computes, and the sum is never zero.
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (p, s) in probs.iter_mut().zip(scores.iter()) {
                    let e = (*s - max).exp();
                    *p = e;
                    sum += e;
                }
                if sum > 0.0 {
                    for p in probs.iter_mut() {
                        *p /= sum;
                    }
                }
                let obase = qi * hw + h * hd;
                // Accumulate over EVERY slot, not only the surviving ones: the
                // uniform case above spreads probability over the whole block
                // context, and the reference's values outside the sequence are
                // the context padding's ZEROS. A filled slot's probability is 0
                // whenever anything survived, so this is exact either way.
                for (d, o) in out[obase..obase + hd].iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (j, p) in probs.iter().enumerate() {
                        let kv = base + j as isize;
                        if kv >= 0 && (kv as usize) < steps {
                            acc += *p * vs[kv as usize * hw + h * hd + d];
                        }
                    }
                    *o = acc;
                }
            }
        }
    }
    Ok(out)
}

/// `[q_dim]` scaling: `head_dim^-0.5 · ln2⁻¹ · softplus(per_dim_scale)`, with
/// torch's softplus threshold (linear past 20, matching `F.softplus`).
fn q_scale_from(cfg: &AudioConfig, per_dim: &[f32]) -> Vec<f32> {
    let hd = cfg.head_dim;
    let base = (hd as f32).powf(-0.5) / std::f32::consts::LN_2;
    per_dim[..hd]
        .iter()
        .map(|&x| {
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            base * sp
        })
        .collect()
}

/// `k_proj`'s scalar scale: `ln(1+e) · ln2⁻¹`, the reference's stored value.
fn k_scale_of() -> f32 {
    (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2
}

/// The `gradient_clipping` bound: `min(config, dtype max)` — 1e10 in f32,
/// i.e. inert for any finite activation.
fn gradient_clip(cfg: &AudioConfig) -> Clip {
    let g = cfg.gradient_clipping.min(f32::MAX);
    Clip {
        in_min: -g,
        in_max: g,
        out_min: -g,
        out_max: g,
    }
}

/// The relative-position table (`Gemma4AudioRelPositionalEncoding`):
/// `[sin(pos·inv) | cos(pos·inv)]` for `pos = context/2 … 0` and
/// `inv[i] = exp(i · −ln(10000)/(hidden/2 − 1))`.
fn relative_positions(cfg: &AudioConfig) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let half = hidden / 2;
    let n_rel = cfg.context_size() / 2 + 1;
    let mut inv = vec![0.0f64; half];
    let inc = 10_000f64.ln() / (half.max(2) - 1) as f64;
    for (i, v) in inv.iter_mut().enumerate() {
        *v = (-(i as f64) * inc).exp();
    }
    let mut out = vec![0.0f32; n_rel * hidden];
    for t in 0..n_rel {
        let pos = (n_rel - 1 - t) as f64; // context/2 down to 0
        for i in 0..half {
            let a = pos * inv[i];
            out[t * hidden + i] = a.sin() as f32;
            out[t * hidden + half + i] = a.cos() as f32;
        }
    }
    out
}

// ─── loading ───────────────────────────────────────────────────────────────

/// Stream the `.vl` sidecar's audio tower to the GPU.
///
/// `mel_bins` is the frontend's width: the subsample stack and the
/// `input_proj_linear` flatten are both built for it, so a checkpoint whose
/// conv widths disagree with the mel bank is a load error, not a silent
/// reshape.
pub fn load_audio_tower(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &AudioConfig,
    mel_bins: usize,
    text_hidden: usize,
) -> Result<AudioTower, String> {
    let hidden = cfg.hidden_size;
    let [c0, c1] = cfg.subsampling_conv_channels;
    let proj_in = cfg.subsample_input_dim(mel_bins);
    if proj_in != hidden {
        return Err(format!(
            "embeddinggemma2: audio subsample flattens to {proj_in} values for {mel_bins} \
             mel bins, but the tower's hidden width is {hidden}"
        ));
    }
    let read_f16_words = |name: &str| -> Result<Vec<u16>, String> {
        let t = names::read_tower(src, name)?;
        let words = decode_f16(&t, name)?;
        names::release_tower(src, name);
        Ok(words)
    };
    let mat = |gpu: &mut Gpu, name: &str, rows: usize, cols: usize| -> Result<GpuTensor, String> {
        let words = read_f16_words(name)?;
        if words.len() != rows * cols {
            return Err(format!(
                "embeddinggemma2: tower tensor `{name}` has {} elems, expected {rows}x{cols}",
                words.len()
            ));
        }
        upload_f16_words(gpu, &words, rows, cols, name)
    };
    let vecf = |gpu: &mut Gpu, name: &str, n: usize| -> Result<GpuTensor, String> {
        let v = read_f32_vec(src, name, n)?;
        upload_f32_slice(gpu, &v, name)
    };
    let host_f32 = |name: &str, n: usize| -> Result<Vec<f32>, String> {
        let words = read_f16_words(name)?;
        if words.len() != n {
            return Err(format!(
                "embeddinggemma2: tower tensor `{name}` has {} elems, expected {n}",
                words.len()
            ));
        }
        Ok(words.iter().map(|w| f16::from_bits(*w).to_f32()).collect())
    };
    let scalar = |name: &str| -> Result<f32, String> {
        let v = read_f32_vec(src, name, 1)?;
        Ok(v[0])
    };
    let clip = |prefix: &str, lin: &str| -> Result<Option<Clip>, String> {
        if !cfg.use_clipped_linears {
            return Ok(None);
        }
        Ok(Some(Clip {
            in_min: scalar(&format!("{prefix}.{lin}.input_min"))?,
            in_max: scalar(&format!("{prefix}.{lin}.input_max"))?,
            out_min: scalar(&format!("{prefix}.{lin}.output_min"))?,
            out_max: scalar(&format!("{prefix}.{lin}.output_max"))?,
        }))
    };

    let cp = "audio_tower.subsample_conv_projection";
    let conv = SubSample {
        w0: host_f32(&format!("{cp}.layer0.conv.weight"), c0 * 9)?,
        n0: read_f32_vec(src, &format!("{cp}.layer0.norm.weight"), c0)?,
        w1: host_f32(&format!("{cp}.layer1.conv.weight"), c1 * c0 * 9)?,
        n1: read_f32_vec(src, &format!("{cp}.layer1.norm.weight"), c1)?,
    };
    let input_proj = mat(
        gpu,
        &format!("{cp}.input_proj_linear.weight"),
        hidden,
        proj_in,
    )?;

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("audio_tower.layers.{li}");
        let ffn = |gpu: &mut Gpu, which: &str| -> Result<MacaronFfn, String> {
            let fp = format!("{p}.{which}");
            Ok(MacaronFfn {
                lin1: mat(
                    gpu,
                    &format!("{fp}.ffw_layer_1.linear.weight"),
                    cfg.ffn_dim(),
                    hidden,
                )?,
                lin2: mat(
                    gpu,
                    &format!("{fp}.ffw_layer_2.linear.weight"),
                    hidden,
                    cfg.ffn_dim(),
                )?,
                pre_norm: vecf(gpu, &format!("{fp}.pre_layer_norm.weight"), hidden)?,
                post_norm: vecf(gpu, &format!("{fp}.post_layer_norm.weight"), hidden)?,
                lin1_clip: clip(&fp, "ffw_layer_1")?,
                lin2_clip: clip(&fp, "ffw_layer_2")?,
            })
        };
        let ap = format!("{p}.self_attn");
        let per_dim = read_f32_vec(src, &format!("{ap}.per_dim_scale"), cfg.head_dim)?;
        let dw = {
            let name = format!("{p}.lconv1d.depthwise_conv1d.weight");
            let v = host_f32(&name, hidden * cfg.conv_kernel_size)?;
            upload_f32_slice(gpu, &v, &name)?
        };
        let lp = format!("{p}.lconv1d");
        layers.push(AudioLayer {
            ff1: ffn(gpu, "feed_forward1")?,
            ff2: ffn(gpu, "feed_forward2")?,
            lconv_pre_norm: vecf(gpu, &format!("{lp}.pre_layer_norm.weight"), hidden)?,
            lconv_lin_start: mat(
                gpu,
                &format!("{lp}.linear_start.linear.weight"),
                cfg.lconv_dim(),
                hidden,
            )?,
            lconv_lin_end: mat(
                gpu,
                &format!("{lp}.linear_end.linear.weight"),
                hidden,
                hidden,
            )?,
            lconv_depthwise: dw,
            lconv_conv_norm: vecf(gpu, &format!("{lp}.conv_norm.weight"), hidden)?,
            lconv_start_clip: clip(&lp, "linear_start")?,
            lconv_end_clip: clip(&lp, "linear_end")?,
            q_proj: mat(gpu, &format!("{ap}.q_proj.linear.weight"), hidden, hidden)?,
            k_proj: mat(gpu, &format!("{ap}.k_proj.linear.weight"), hidden, hidden)?,
            v_proj: mat(gpu, &format!("{ap}.v_proj.linear.weight"), hidden, hidden)?,
            attn_post: mat(gpu, &format!("{ap}.post.linear.weight"), hidden, hidden)?,
            relative_k_proj: mat(gpu, &format!("{ap}.relative_k_proj.weight"), hidden, hidden)?,
            q_clip: clip(&ap, "q_proj")?,
            k_clip: clip(&ap, "k_proj")?,
            v_clip: clip(&ap, "v_proj")?,
            post_clip: clip(&ap, "post")?,
            q_scale: q_scale_from(cfg, &per_dim),
            norm_pre_attn: vecf(gpu, &format!("{p}.norm_pre_attn.weight"), hidden)?,
            norm_post_attn: vecf(gpu, &format!("{p}.norm_post_attn.weight"), hidden)?,
            norm_out: vecf(gpu, &format!("{p}.norm_out.weight"), hidden)?,
        });
    }

    let dims = cfg.output_proj_dims;
    let output_proj = mat(gpu, "audio_tower.output_proj.weight", dims, hidden)?;
    let output_bias = vecf(gpu, "audio_tower.output_proj.bias", dims)?;
    let embed_projection = mat(
        gpu,
        "embed_audio.embedding_projection.weight",
        text_hidden,
        dims,
    )?;
    let e_ones = gpu
        .upload_f32(&vec![1.0f32; dims], &[dims])
        .map_err(|e| format!("embeddinggemma2: audio embedder ones: {e:?}"))?;

    Ok(AudioTower {
        config: *cfg,
        mel_bins,
        conv,
        input_proj,
        layers,
        output_proj,
        output_bias,
        embed_projection,
        e_ones,
        rel_pos: relative_positions(cfg),
        text_hidden,
    })
}

/// Read a tower tensor as f32 (the f16/BF16/f32 decode gates, then widen).
fn read_f32_vec(src: &ModelSource, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let t = names::read_tower(src, name)?;
    let v = decode_f32(&t, name)?;
    names::release_tower(src, name);
    if v.len() != n {
        return Err(format!(
            "embeddinggemma2: tower tensor `{name}` has {} elems, expected {n}",
            v.len()
        ));
    }
    Ok(v)
}

impl AudioTower {
    /// A synthetic tower for GPU smoke tests and examples: deterministic
    /// small random weights, no checkpoint, the same tensor set the real
    /// loader produces (including real clip bounds, so the clamp path runs).
    /// Real loads go through [`load_audio_tower`]; this exists so
    /// `examples/audio_smoke.rs` can drive subsample → layers → projector end
    /// to end without the `.vl` sidecar.
    pub fn synthetic(
        gpu: &mut Gpu,
        cfg: &AudioConfig,
        mel_bins: usize,
        text_hidden: usize,
        seed: u64,
    ) -> Result<Self, String> {
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
            fn vec(&mut self, n: usize) -> Vec<f32> {
                (0..n).map(|_| 1.0 + self.next() * 0.02 - 0.01).collect()
            }
        }
        let mut rng = Rng(seed | 1);
        let hidden = cfg.hidden_size;
        let [c0, c1] = cfg.subsampling_conv_channels;
        if cfg.subsample_input_dim(mel_bins) != hidden {
            return Err("synthetic audio tower: mel bins do not match the tower width".into());
        }

        let mat =
            |rng: &mut Rng, gpu: &mut Gpu, rows: usize, cols: usize| -> Result<GpuTensor, String> {
                let words: Vec<u16> = (0..rows * cols)
                    .map(|_| f16::from_f32(rng.next() * 0.06 - 0.03).to_bits())
                    .collect();
                gpu.upload_f16_bits(&words, &[rows, cols])
                    .map_err(|e| format!("synthetic audio upload: {e:?}"))
            };
        let vecf = |rng: &mut Rng, gpu: &mut Gpu, n: usize| -> Result<GpuTensor, String> {
            let v = rng.vec(n);
            gpu.upload_f32(&v, &[n])
                .map_err(|e| format!("synthetic audio upload: {e:?}"))
        };
        // Bounds tight enough to actually clamp, so the kernel is exercised
        // rather than merely loaded.
        let clip = Some(Clip {
            in_min: -8.0,
            in_max: 8.0,
            out_min: -8.0,
            out_max: 8.0,
        });

        let conv = SubSample {
            w0: (0..c0 * 9).map(|_| rng.next() * 0.1 - 0.05).collect(),
            n0: rng.vec(c0),
            w1: (0..c1 * c0 * 9)
                .map(|_| rng.next() * 0.05 - 0.025)
                .collect(),
            n1: rng.vec(c1),
        };
        let input_proj = mat(&mut rng, gpu, hidden, hidden)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(AudioLayer {
                ff1: MacaronFfn {
                    lin1: mat(&mut rng, gpu, cfg.ffn_dim(), hidden)?,
                    lin2: mat(&mut rng, gpu, hidden, cfg.ffn_dim())?,
                    pre_norm: vecf(&mut rng, gpu, hidden)?,
                    post_norm: vecf(&mut rng, gpu, hidden)?,
                    lin1_clip: clip,
                    lin2_clip: clip,
                },
                ff2: MacaronFfn {
                    lin1: mat(&mut rng, gpu, cfg.ffn_dim(), hidden)?,
                    lin2: mat(&mut rng, gpu, hidden, cfg.ffn_dim())?,
                    pre_norm: vecf(&mut rng, gpu, hidden)?,
                    post_norm: vecf(&mut rng, gpu, hidden)?,
                    lin1_clip: clip,
                    lin2_clip: clip,
                },
                lconv_pre_norm: vecf(&mut rng, gpu, hidden)?,
                lconv_lin_start: mat(&mut rng, gpu, cfg.lconv_dim(), hidden)?,
                lconv_lin_end: mat(&mut rng, gpu, hidden, hidden)?,
                lconv_depthwise: gpu
                    .upload_f32(
                        &(0..hidden * cfg.conv_kernel_size)
                            .map(|_| rng.next() * 0.4 - 0.2)
                            .collect::<Vec<f32>>(),
                        &[hidden * cfg.conv_kernel_size],
                    )
                    .map_err(|e| format!("synthetic audio upload: {e:?}"))?,
                lconv_conv_norm: vecf(&mut rng, gpu, hidden)?,
                lconv_start_clip: clip,
                lconv_end_clip: clip,
                q_proj: mat(&mut rng, gpu, hidden, hidden)?,
                k_proj: mat(&mut rng, gpu, hidden, hidden)?,
                v_proj: mat(&mut rng, gpu, hidden, hidden)?,
                attn_post: mat(&mut rng, gpu, hidden, hidden)?,
                relative_k_proj: mat(&mut rng, gpu, hidden, hidden)?,
                q_clip: clip,
                k_clip: clip,
                v_clip: clip,
                post_clip: clip,
                q_scale: q_scale_from(cfg, &rng.vec(cfg.head_dim)),
                norm_pre_attn: vecf(&mut rng, gpu, hidden)?,
                norm_post_attn: vecf(&mut rng, gpu, hidden)?,
                norm_out: vecf(&mut rng, gpu, hidden)?,
            });
        }

        let dims = cfg.output_proj_dims;
        let output_proj = mat(&mut rng, gpu, dims, hidden)?;
        let output_bias = vecf(&mut rng, gpu, dims)?;
        let embed_projection = mat(&mut rng, gpu, text_hidden, dims)?;
        let e_ones = gpu
            .upload_f32(&vec![1.0f32; dims], &[dims])
            .map_err(|e| format!("synthetic audio ones: {e:?}"))?;

        Ok(AudioTower {
            config: *cfg,
            mel_bins,
            conv,
            input_proj,
            layers,
            output_proj,
            output_bias,
            embed_projection,
            e_ones,
            rel_pos: relative_positions(cfg),
            text_hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{audio_token_count, ProcessedAudio};

    /// A toy tower geometry: 4 mel bins → two stride-2 convs → 1 frequency ×
    /// 4 channels = the 4-wide hidden; 1 head of 1 dim; chunk 3 with 2 back.
    fn toy() -> AudioConfig {
        AudioConfig {
            hidden_size: 4,
            num_layers: 1,
            n_heads: 1,
            head_dim: 1,
            rms_norm_eps: 1e-6,
            output_proj_dims: 4,
            conv_kernel_size: 3,
            attention_chunk_size: 3,
            attention_context_left: 3,
            attention_context_right: 0,
            attention_logit_cap: 1e6,
            attention_invalid_logits_value: -1e9,
            residual_weight: 0.5,
            subsampling_conv_channels: [4, 4],
            use_clipped_linears: true,
            gradient_clipping: 1e10,
        }
    }

    /// A subsample whose both stages carry ONE tap, so the expected values
    /// are hand-computable:
    ///  - stage 0 reads frame `2·ti + 1`, mel column `2·wi` into channel 0;
    ///  - stage 1 reads stage-0 row `2·t + 1`, frequency `2·wi` of channel 0.
    fn single_tap_subsample(cfg: &AudioConfig) -> SubSample {
        let [c0, c1] = cfg.subsampling_conv_channels;
        let mut w0 = vec![0.0f32; c0 * 9];
        w0[2 * 3 + 1] = 1.0; // (oc 0, dy 2, dx 1)
        let mut w1 = vec![0.0f32; c1 * c0 * 9];
        w1[2 * 3 + 1] = 1.0; // (oc 0, ic 0, dy 2, dx 1)
        SubSample {
            w0,
            n0: vec![1.0; c0],
            w1,
            n1: vec![1.0; c1],
        }
    }

    /// A mel whose only positive cells are the ones the stage-0 tap reads
    /// (odd frame, even column); everything else is negative, so an
    /// off-by-one tap reads a negative value and ReLUs to zero instead of
    /// quietly producing the same positive number.
    fn checker_mel(frames: usize, bins: usize) -> Vec<f32> {
        (0..frames * bins)
            .map(|k| {
                let (i, j) = (k / bins, k % bins);
                if i % 2 == 1 && j % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    fn clip(frames: usize, valid: usize, mel: Vec<f32>) -> ProcessedAudio {
        ProcessedAudio {
            mel,
            plan: crate::audio::AudioPlan {
                samples: 0,
                padded_samples: 0,
                frames,
                valid_frames: valid,
                n_tokens: audio_token_count(frames, valid),
            },
        }
    }

    /// One nonzero channel of a 4-wide vector normalizes to exactly √3 after
    /// `LayerNorm(affine, no bias) → ReLU`:
    ///   mean x/4, var = (9+1+1+1)x²/16/4 = 3x²/16,
    ///   y₀ = (3x/4)·4/(x√3) = √3, the other three ReLU to 0.
    const ROOT3: f32 = 1.732_050_8;

    #[test]
    fn subsample_halves_time_and_carries_the_halved_mask() {
        let cfg = toy();
        let sub = single_tap_subsample(&cfg);
        let (frames, bins) = (12usize, 4usize);
        let mel = checker_mel(frames, bins);

        // 12 frames → 6 stage-0 rows → 3 tokens. All frames real.
        let all = sub
            .forward(&cfg, bins, &clip(frames, 12, mel.clone()))
            .unwrap();
        assert_eq!(all.len(), 3 * 4);
        for t in 0..3 {
            assert!(
                (all[t * 4] - ROOT3).abs() < 1e-3,
                "token {t} channel 0: {}",
                all[t * 4]
            );
            for c in 1..4 {
                assert!(all[t * 4 + c].abs() < 1e-6, "token {t} channel {c}");
            }
        }

        // 10 real frames: `mask[:, ::2]` keeps rows 0..4 and zeroes row 5, so
        // the last token — whose only tap IS row 5 — comes out all-zero. The
        // fixture deliberately leaves the invalid frames themselves nonzero,
        // so this isolates the mask from the frontend's own zeroing.
        let masked = sub.forward(&cfg, bins, &clip(frames, 10, mel)).unwrap();
        assert_eq!(masked.len(), 3 * 4, "the same three tokens survive");
        assert!((masked[0] - ROOT3).abs() < 1e-3);
        assert!((masked[4] - ROOT3).abs() < 1e-3);
        assert!(
            masked[8..].iter().all(|v| v.abs() < 1e-6),
            "masked row must give an all-zero token: {:?}",
            &masked[8..]
        );
    }

    fn assert_rows(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "row {i}: {g} vs {w} ({got:?})");
        }
    }

    #[test]
    fn attention_keeps_the_complement_of_the_causal_window() {
        let cfg = toy();
        // q = k = 0 → every surviving score is 0, so the output is the MEAN of
        // the keys each query can reach. The reference's mask is inverted, so
        // a query reaches every in-sequence key of its block context EXCEPT
        // itself and the one key behind it (`context_left - 1 = 2` wide) —
        // future keys included.
        let q = vec![0.0f32; 5];
        let k = vec![0.0f32; 5];
        let v: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let rel = vec![0.0f32; 3]; // n_rel = context/2 + 1 = 3
        let out = attention(&cfg, &[1.0], &q, &k, &v, &rel, 5, 5).unwrap();
        // q0: block 0 (kv -2..2) → keep {1,2} → 1.5
        // q1: → keep {2}            → 2.0
        // q2: → keep {0}            → 0.0
        // q3,q4: block 1 (kv 1..5) → keep {1,4} / {1,2} → 2.5 / 1.5
        assert_rows(&out, &[1.5, 2.0, 0.0, 2.5, 1.5]);
    }

    /// A query can have NO surviving slot — the whole block context is inside
    /// the causal window — and the reference answers that with a uniform
    /// distribution over the context, not a zero row. `q = 11` of the real
    /// geometry (chunk 12, context 24) is exactly that case.
    #[test]
    fn attention_is_uniform_when_nothing_survives_the_window() {
        let cfg = AudioConfig {
            attention_chunk_size: 12,
            attention_context_left: 13,
            ..toy()
        };
        let rows = 12;
        let hw = cfg.n_heads * cfg.head_dim;
        assert_eq!(hw, 1);
        let q = vec![1.0f32; rows];
        let k = vec![0.0f32; rows];
        // v[i] = i, so the uniform mean over the block context is identifiable.
        let v: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        let rel = vec![0.0f32; (cfg.context_size() / 2 + 1) * hw];
        let out = attention(&cfg, &[1.0], &q, &k, &v, &rel, rows, rows).unwrap();
        // Query 11 is the last of block 0, whose context is kv -12..11: every
        // in-sequence slot (0..11) is inside the 12-wide window, so all 24
        // slots are filled and the row is the mean of the block context's
        // values — the 12 in-sequence ones plus 12 context ZEROS:
        // (0+…+11)/24 = 2.75.
        assert!((out[11] - 2.75).abs() < 1e-5, "q11 = {}", out[11]);
        // Query 0 sits at the sequence's left edge, so its accepted slots are
        // the block context's in-sequence ones minus kv 0: {1..11} → 6.0.
        assert!((out[0] - 6.0).abs() < 1e-5, "q0 = {}", out[0]);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// The reference's sequence is the WHOLE subsampled length, and its
    /// inverted mask makes the padded tail a real KEY for the valid rows — so
    /// every `steps` row is run even when only the first `rows` of them are the
    /// answer. `rows = 2, steps = 3` is the shape of a 1 409-sample clip; a
    /// tower that truncates to `rows` before the forward is ~15 % rel-L2 off.
    #[test]
    fn attention_keys_into_the_padded_tail() {
        let cfg = toy();
        let q = vec![0.0f32; 3];
        let k = vec![0.0f32; 3];
        // Weighted so the surviving SETS are readable in the answer.
        let v = vec![10.0f32, 0.0, 0.0];
        let rel = vec![0.0f32; 3];
        // q0 keeps the tail row and its future neighbour ({1, 2} → 0); q1
        // keeps the tail row alone (0); q2 is an INVALID query: the mask's
        // query axis carries no term, so its inclusion is still
        // `valid_key ∧ window` = {1} and it keeps {0, 2} (→ 5). A tower that
        // drops the tail, or that lets an invalid query keep everything,
        // answers 0 or 10/3 here.
        let out = attention(&cfg, &[1.0], &q, &k, &v, &rel, 2, 3).unwrap();
        assert_rows(&out, &[0.0, 0.0, 5.0]);
        // Same rows, no tail: q2 is now a VALID query, its window ({1, 2}) both
        // fills and hides the tail, so it keeps kv 0 alone (→ 10).
        let out = attention(&cfg, &[1.0], &q, &k, &v, &rel, 3, 3).unwrap();
        assert_rows(&out, &[0.0, 0.0, 10.0]);
    }

    #[test]
    fn attention_rel_shift_places_each_distance_on_the_right_slot() {
        let cfg = toy();
        // q = 1, k = 0 → only the relative bias scores. rel[0] is the embedding
        // of position n_rel-1 = 2, i.e. of the key TWO steps back — the only
        // past distance the inverted mask lets through (every shorter one is
        // inside the window, and future slots sit at `j < i`, where the skew
        // zeroes the bias).
        let q = vec![1.0f32; 6];
        let k = vec![0.0f32; 6];
        let v: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let rel = vec![100.0f32, 0.0, 0.0];
        let out = attention(&cfg, &[1.0], &q, &k, &v, &rel, 6, 6).unwrap();
        // The distance-2 key of query q is kv = q - 2, and the bias lands on
        // it in EVERY block (its window slot is `j = i`), so the softmax hands
        // that key everything: q2→v0=0, q3→v1=1, q4→v2=2, q5→v3=3. Queries 0
        // and 1 have no in-sequence distance-2 key, so their bias is invisible
        // and they keep the plain means 1.5 and 2.0. Without the skew — or
        // with it mirrored — queries 3..5 fall back to the plain means
        // 3.333 / 1.5 / 2.0, so these six rows pin exactly where the bias
        // lands.
        assert_rows(&out, &[1.5, 2.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn attention_refuses_shape_mismatches() {
        let cfg = toy();
        let rel = vec![0.0f32; 3];
        let good = vec![0.0f32; 4];
        // q/k/v must be rows × q_dim.
        assert!(attention(&cfg, &[1.0], &good, &good, &good[..3], &rel, 4, 4).is_err());
        // The relative table must be n_rel × q_dim.
        assert!(attention(&cfg, &[1.0], &good, &good, &good, &rel[..2], 4, 4).is_err());
        // The query scale must be head_dim wide.
        assert!(attention(&cfg, &[1.0, 2.0], &good, &good, &good, &rel, 4, 4).is_err());
        // The valid prefix cannot be longer than the sequence it sits in.
        assert!(attention(&cfg, &[1.0], &good, &good, &good, &rel, 5, 4).is_err());
    }

    #[test]
    fn q_scale_folds_the_reference_factors() {
        let cfg = toy();
        // per_dim_scale = 0 → softplus(0) = ln 2, so the ln2 factors cancel:
        // q_scale = head_dim^-0.5 exactly.
        let s = q_scale_from(&cfg, &[0.0]);
        assert!((s[0] - 1.0).abs() < 1e-6);
        // torch's softplus is linear past 20: softplus(25) = 25.
        let big = AudioConfig { head_dim: 4, ..cfg };
        let s = q_scale_from(&big, &[25.0, 0.0, -25.0, 1.0]);
        let base = 4f32.powf(-0.5) / std::f32::consts::LN_2;
        assert!((s[0] - base * 25.0).abs() < 1e-5);
        assert!((s[1] - base * std::f32::consts::LN_2).abs() < 1e-6);
        assert!(s[2].abs() < 1e-6, "softplus(-25) underflows to 0");
        assert!((s[3] - base * (1.0 + 1f32.exp()).ln()).abs() < 1e-6);
    }

    #[test]
    fn k_scale_is_the_stored_constant() {
        let want = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        assert!((k_scale_of() - want).abs() < 1e-7);
        assert!((k_scale_of() - 1.894_0).abs() < 1e-3, "≈ ln(1+e)/ln2");
    }

    #[test]
    fn relative_positions_are_sin_cos_of_the_decaying_timescales() {
        let cfg = toy(); // hidden 4 → 2 timescales, context 5 → n_rel 3
        let rel = relative_positions(&cfg);
        assert_eq!(rel.len(), 3 * 4);
        // pos_id runs context/2 … 0, so the LAST row is the zero position:
        // sin 0 = 0 on the left half, cos 0 = 1 on the right half.
        assert_eq!(&rel[8..12], &[0.0, 0.0, 1.0, 1.0]);
        // Row 0 holds pos_id 2 (n_rel-1): angle = 2 · inv[i],
        // inv[i] = exp(i · -ln(10000)/1).
        let inc = 10_000f64.ln();
        for i in 0..2usize {
            let angle = 2.0f64 * (-(i as f64) * inc).exp();
            assert!((rel[i] as f64 - angle.sin()).abs() < 1e-6, "sin {i}");
            assert!((rel[2 + i] as f64 - angle.cos()).abs() < 1e-6, "cos {i}");
        }
        // The last row's position is the 0 position, not the largest.
        assert!((rel[8] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn layernorm_relu_matches_the_reference_definition() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        layernorm_relu(&mut x, &[1.0; 4], 0.0);
        // mean 2.5, biased var 1.25, so (x - mean)/√1.25 then ReLU.
        let inv = 1.0 / 1.25f32.sqrt();
        assert_eq!(x[0], 0.0);
        assert_eq!(x[1], 0.0);
        assert!((x[2] - 0.5 * inv).abs() < 1e-6);
        assert!((x[3] - 1.5 * inv).abs() < 1e-6);
        // A NEGATIVE gamma flips the negative half positive, so it survives
        // the ReLU — proof the affine weight is applied before the activation.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        layernorm_relu(&mut x, &[-5.0, 1.0, 1.0, 1.0], 0.0);
        assert!((x[0] - 1.5 * inv * 5.0).abs() < 1e-6);
        assert!((x[2] - 0.5 * inv).abs() < 1e-6);
    }

    #[test]
    fn the_gradient_clamp_is_inert_in_f32() {
        let cfg = toy();
        let c = gradient_clip(&cfg);
        // min(gradient_clipping, f32::MAX) — 1e10 here, not f32::MAX.
        assert_eq!(c.in_min, -1e10);
        assert_eq!(c.in_max, 1e10);
        for v in [0.0f32, 1.0, -1234.5, 1e9, -1e9] {
            assert_eq!(v.clamp(c.in_min, c.in_max), v, "{v} must pass through");
        }
    }

    #[test]
    fn subsample_refuses_a_mel_width_that_does_not_flatten_to_hidden() {
        let cfg = toy();
        let sub = single_tap_subsample(&cfg);
        // 6 bins → f0 = 3 → f1 = 2 → 2 × 4 = 8 ≠ hidden 4.
        let err = sub
            .forward(&cfg, 6, &clip(4, 4, vec![0.0; 4 * 6]))
            .unwrap_err();
        assert!(err.contains("hidden width"), "{err}");
    }
}
