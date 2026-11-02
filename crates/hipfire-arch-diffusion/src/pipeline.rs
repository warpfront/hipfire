// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU image-generation pipeline orchestration for FLUX.1 and FLUX.2 (Klein).
//!
//! [`load_pipe`] opens a diffusers-format pipe directory into one bundle and
//! detects the family from the transformer's `config.json`: FLUX.1 carries
//! `transformer/`, `text_encoder/` (CLIP-L), `text_encoder_2/` (T5-XXL),
//! `vae/` and `scheduler/`; FLUX.2 Klein carries `transformer/`,
//! `text_encoder/` (a Qwen3 text encoder), `tokenizer/`, `vae/` and `scheduler/`
//! and no `text_encoder_2/` at all. The text stack is a [`TextCond`] enum,
//! so a bundle cannot be asked for the other family's conditioning.
//!
//! [`generate_img_prompt`] drives the full denoise loop — conditioning
//! (FLUX.1: T5 hidden → `txt`, CLIP pooled → `vec`; FLUX.2: the Qwen3
//! residual stream after the three [`KLEIN_TAPS`] layers → `txt`, no pooled
//! vector), packed-latent init, optional VAE-encoded reference tokens (the
//! Klein edit path), per-step MMDiT forward (`flux::forward`), flow-Euler
//! step, unpack + latent denormalize, VAE decode and PNG postprocess
//! (diffusers `(x+1)/2` denormalize + ×255 banker's round).
//!
//! The single-block MLP activation is GELU-tanh in BFL, diffusers and
//! ComfyUI alike; the `MlpAct` knob on the forward exists only so a future
//! deviation can be pinned (see the doc on [`flux::MlpAct`]).
//!
//! On-GPU execution (BF16 weights + existing GEMM/attention tables) is the
//! later kernel phase; this CPU reference is the numeric oracle both the
//! fixture gate and the GPU path validate against.

use std::path::Path;

use crate::clip::{self, ClipWeights};
use crate::clip_gpu::{self, GpuClipWeights};
use crate::config::{FluxDiffusionConfig, FluxFamily};
use crate::flux::{self, FinalAdaLNOrder, FluxWeights, MlpAct};
use crate::flux_gpu::{self, GpuFluxWeights};
use crate::klein_prompt::{self, KLEIN_MIN_LEN, KLEIN_PAD_ID};
use crate::qwen3::{self, Qwen3Plan, Qwen3Weights, KLEIN_TAPS};
use crate::qwen3_gpu::{self, GpuQwen3Weights};
use crate::refimg::RefImage;
use crate::scheduler::{self, ShiftRule};
use crate::t5::{self, T5Weights};
use crate::t5_gpu::{self, GpuT5Weights};
use crate::tokenizer::{encode_t5, Gpt2Bpe, UnigramVocab};
use crate::vae::{self, LatentNorm, VaeDecoderWeights, VaeEncoderWeights};
use crate::vae_gpu;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
use std::borrow::Cow;
use std::path::PathBuf;

/// Scheduler + VAE coefficients pulled from the pipe's configs.
#[derive(Debug, Clone)]
pub struct PipeMeta {
    pub num_train_timesteps: u32,
    /// How the sigma schedule is shifted: FLUX.1's fixed `shift` from
    /// `scheduler_config.json`, or FLUX.2 Klein's resolution-dependent
    /// empirical `mu` (see [`scheduler::sigma_pairs_ruled`]).
    pub shift_rule: ShiftRule,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f32,
    pub max_shift: f32,
    pub latent_norm: LatentNorm,
    pub max_seq: usize,
}

/// The mmapped checkpoint components a streaming bundle keeps open, with the
/// naming plans that say which tensor holds what.
///
/// Holding these costs address space, not resident memory: the point of
/// streaming is that a weight is read once, converted to f16, handed to the
/// device, and its pages released. Keeping them also lets the CPU reference
/// path re-materialise f32 tables on demand
/// ([`FluxPipeBundle::transformer_host`] / [`FluxPipeBundle::t5_host`])
/// instead of the bundle carrying ~65 GB of them forever.
pub struct PipeSources {
    pub transformer: Box<dyn ModelSource + Send>,
    pub transformer_plan: flux::FluxPlan,
    /// The text encoder component: `text_encoder_2/` (T5-XXL) for FLUX.1,
    /// `text_encoder/` (the Qwen3 text encoder) for FLUX.2 Klein.
    pub text: Box<dyn ModelSource + Send>,
    pub text_plan: TextPlan,
}

/// Naming plan for whichever text encoder the pipe's family uses. Pure
/// metadata — the variant must match [`TextCond`]'s.
pub enum TextPlan {
    T5(t5::T5Plan),
    Qwen3(Qwen3Plan),
}

/// The text-conditioning stack, one variant per FLUX family.
///
/// FLUX.1 conditions on T5-XXL (`txt`) plus CLIP-L (the pooled `vec`); FLUX.2
/// Klein conditions on a Qwen3 causal-LM text encoder alone, concatenating the
/// residual stream after the three [`KLEIN_TAPS`] layers. The two share no
/// tensor, no tokenizer and no framing, so they are an enum rather than a
/// bag of `Option`s — a Klein pipe cannot be asked for a CLIP pooled vector
/// and a FLUX.1 pipe cannot be asked for a chat template.
pub enum TextCond {
    T5Clip {
        /// Host T5 tables. In streaming mode these are the LIGHT set — the
        /// token embedding and the relative-attention bias, the only two the
        /// GPU encoder reads back from the host ([`T5Weights::is_light`]); the
        /// 24 layers of linears (~18.5 GB) are streamed to the device and
        /// never decoded here. [`FluxPipeBundle::t5_host`] materialises the
        /// full set for the CPU encoder.
        t5: T5Weights,
        clip: ClipWeights,
        /// T5 unigram tokenizer (`tokenizer_2/tokenizer.json`).
        t5_tokenizer: UnigramVocab,
        /// CLIP GPT-2-style BPE tokenizer (`tokenizer/*`).
        clip_tokenizer: Gpt2Bpe,
        /// CLIP framing ids (BOS / EOT / pad).
        clip_bos: u32,
        clip_eot: u32,
        clip_pad: u32,
        /// T5 pad id.
        t5_pad: u32,
        /// GPU-resident T5 encoder weights. `None` when the GPU text encoders
        /// are off (`HIPFIRE_T5_GPU=0`) or the checkpoint has no GPU path (an
        /// ungated T5 v1.0 FFN) — the conditioning then runs on the host.
        gpu_t5: Option<GpuT5Weights>,
        /// GPU-resident CLIP encoder weights; `None` under the same
        /// conditions.
        gpu_clip: Option<GpuClipWeights>,
        /// Set once the T5 GPU upload has been attempted and declined —
        /// either the checkpoint has no GPU path or the ~9.3 GB f16
        /// allocation failed. Stops `ensure_gpu` from re-attempting that
        /// allocation, and re-printing the reason, on every subsequent
        /// `img_generate`.
        gpu_t5_declined: bool,
        /// The same latch for CLIP (~0.25 GB). Separate from the T5 one on
        /// purpose: the two encoders route independently.
        gpu_clip_declined: bool,
    },
    Qwen3 {
        /// Host Qwen3 tables — the LIGHT set (embedding only) in streaming
        /// mode, see [`Qwen3Weights::is_light`]. The CPU encoder needs the
        /// full set and materialises it through
        /// [`FluxPipeBundle::qwen3_host`].
        host: Qwen3Weights,
        plan: Qwen3Plan,
        tokenizer: Tokenizer,
        /// Right-padding id: the tokenizer's `<|endoftext|>`, falling back to
        /// the published [`KLEIN_PAD_ID`].
        pad_id: u32,
        /// Where `tokenizer` came from, so a template that does not tokenize
        /// as Klein expects can name the file to look at.
        tokenizer_path: PathBuf,
        /// GPU-resident Qwen3 text encoder (uploaded once by
        /// [`FluxPipeBundle::ensure_gpu`]). `None` = the conditioning runs on
        /// the host `qwen3::encode_taps` reference instead — the same
        /// route/fallback split T5 has.
        gpu: Option<GpuQwen3Weights>,
        /// Set once the Qwen3 GPU upload has been attempted and declined —
        /// an unsupported checkpoint geometry, or the f16 allocation failing.
        /// Stops `ensure_gpu` re-attempting a multi-GB upload, and reprinting
        /// the reason, on every subsequent `img_generate`. The Klein twin of
        /// `gpu_t5_declined`.
        declined: bool,
    },
}

impl TextCond {
    /// The family this stack conditions — the invariant that pairs it with
    /// [`FluxPipeBundle::transformer_cfg`].
    pub fn family(&self) -> FluxFamily {
        match self {
            TextCond::T5Clip { .. } => FluxFamily::Flux1,
            TextCond::Qwen3 { .. } => FluxFamily::Flux2,
        }
    }
}

/// The whole tiny pipe on the host, ready to run.
pub struct FluxPipeBundle {
    /// Host f32 transformer tables — ~47 GB at FLUX.1-dev geometry.
    ///
    /// `None` in streaming mode, which is what [`load_pipe`] produces when it
    /// can keep the checkpoint open: the GPU upload reads the mmap directly
    /// and this is never built. The CPU reference path materialises it on
    /// demand through [`Self::transformer_host`].
    pub transformer: Option<FluxWeights>,
    pub transformer_cfg: FluxDiffusionConfig,
    /// The text-conditioning stack, one variant per family — see
    /// [`TextCond`].
    pub cond: TextCond,
    pub vae: VaeDecoderWeights,
    /// The VAE ENCODER, loaded only for a family that can take reference
    /// images (FLUX.2 Klein's edit path). `None` for FLUX.1, and for a
    /// config-only VAE load.
    pub vae_enc: Option<VaeEncoderWeights>,
    pub meta: PipeMeta,
    /// Final-head adaLN chunk order, which DIFFERS between the two checkpoint
    /// families and is not discoverable from the tensors themselves.
    ///
    /// BFL's `LastLayer` does `shift, scale = adaLN(vec).chunk(2)`; diffusers'
    /// `AdaLayerNormContinuous` does `scale, shift = chunk(...)`. Getting it
    /// backwards modulates the final projection with the wrong halves, which
    /// does not fail — it yields a velocity that under-denoises, so the image
    /// keeps its coarse structure but never loses its noise. The block-parity
    /// gate covers double/single blocks only, so it cannot catch this.
    pub final_order: FinalAdaLNOrder,
    /// GPU-resident transformer weights (uploaded once via [`ensure_gpu`]);
    /// `None` = host-only CPU execution.
    pub gpu_weights: Option<GpuFluxWeights>,
    /// GPU-resident VAE decoder weights (uploaded once via [`ensure_gpu`]).
    /// `None` = not uploaded; the decode then falls back to a per-generation
    /// upload. Keeping them resident removes ~200 MB of host->device traffic
    /// and ~150 allocations from every image.
    pub gpu_vae: Option<vae_gpu::GpuVaeDecoderWeights>,
    /// GPU-resident VAE ENCODER weights, for the FLUX.2 edit path's reference
    /// images (uploaded once by [`ensure_gpu`](FluxPipeBundle::ensure_gpu)
    /// whenever the bundle carries a host encoder). `None` = no encoder, or a
    /// FLUX.1 pipe; the reference encode then falls back to the host
    /// `vae::encode`, which at 1024² is minutes rather than milliseconds.
    pub gpu_vae_enc: Option<vae_gpu::GpuVaeEncoderWeights>,
    /// Set once the VAE-encoder GPU upload has been attempted and declined —
    /// the same latch shape as `gpu_t5_declined` / `gpu_clip_declined` /
    /// `TextCond::Qwen3::declined`, and for the same reason: without it every
    /// subsequent `ensure_gpu` retries an allocation that already failed, on
    /// a device that is already full, and re-prints the decline line.
    pub gpu_vae_enc_declined: bool,
    /// Per-prompt conditioning cache, keyed `(prompt, family, t5_seq)`. Lives
    /// for the bundle's lifetime, i.e. the daemon session.
    pub cond_cache: CondCache,
    /// Open checkpoint mmaps + naming plans. `Some` = streaming mode.
    /// `None` = every host table is already resident (synthetic fixtures, and
    /// any caller that built a bundle without a checkpoint directory).
    pub sources: Option<PipeSources>,
}

/// One cached conditioning result: everything a denoise loop needs from the
/// text encoders, for one `(prompt, t5_seq)`.
pub struct CondEntry {
    pub prompt: String,
    /// Which family encoded this. Part of the key: the same prompt at the
    /// same length is a DIFFERENT tensor under T5-XXL than under the Qwen3
    /// text encoder, and one bundle per process is not guaranteed.
    pub family: FluxFamily,
    pub t5_seq: usize,
    /// `last_hidden_state` **on the device**, f32 `[t5_seq, d_model]`. Stays
    /// resident so the MMDiT's `txt_in` reads it without a host round trip
    /// (see [`flux_gpu::gpu_forward_txt_dev`]).
    pub t5_hidden: GpuTensor,
    /// CLIP pooled vector, host-side: the MMDiT's `vector_in` embedder takes
    /// it as a `Vec<f32>` and it is only `pooled_projection_dim` wide.
    pub clip_pooled: Vec<f32>,
}

/// LRU conditioning cache, `(prompt, t5_seq)` → `(t5_hidden, clip_pooled)`.
///
/// Why it earns its keep: on the target machine the host T5 encode is
/// **54.9 s per prompt** (CLIP another 0.24 s) against a ComfyUI fixed cost
/// of 12.7 s for the entire conditioning + VAE stack. Even with the encoders
/// on the GPU, re-encoding an unchanged prompt is pure waste — an interactive
/// session sweeping seeds or step counts on one prompt pays it once.
///
/// Entries hold DEVICE memory (`t5_hidden`), so eviction frees; a dropped
/// `CondCache` does not (`GpuTensor` has no `Drop`). [`Self::clear`] and
/// [`FluxPipeBundle::free_gpu`] are the release paths.
///
/// The `t5_seq` half of the key is not decorative: the same prompt padded to
/// a different sequence length is a different `[t5_seq, d_model]` tensor and
/// a different `[heads, t5_seq, t5_seq]` relative bias.
pub struct CondCache {
    /// Most-recently-used first.
    entries: Vec<CondEntry>,
    capacity: usize,
    enabled: bool,
}

impl Default for CondCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CondCache {
    /// Capacity fixed at 8: a `[256, 4096]` f32 hidden state is 4 MB, so the
    /// whole cache is 32 MB — noise beside the 24 GB checkpoint, and deep
    /// enough for an interactive prompt-rotation session.
    pub const CAPACITY: usize = 8;

    /// Enabled unless `HIPFIRE_IMG_COND_CACHE=0`. When disabled, nothing is
    /// stored at all: conditioning takes the `CondSlot::Owned` path and is
    /// freed after the denoise loop, so the kill-switch reproduces the
    /// uncached behaviour rather than only skipping the lookup.
    ///
    /// Read once, at pipe load. Flipping the env var mid-session takes effect
    /// on the next model load, not the next request.
    pub fn new() -> Self {
        let enabled =
            hipfire_config::developer_var("HIPFIRE_IMG_COND_CACHE").map_or(true, |v| v != "0");
        Self {
            entries: Vec::new(),
            capacity: Self::CAPACITY,
            enabled,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index of the entry for `(prompt, family, t5_seq)`, promoting it to
    /// MRU. Always `None` when the cache is disabled.
    fn lookup(&mut self, prompt: &str, family: FluxFamily, t5_seq: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let pos = self
            .entries
            .iter()
            .position(|e| e.t5_seq == t5_seq && e.family == family && e.prompt == prompt)?;
        let e = self.entries.remove(pos);
        self.entries.insert(0, e);
        Some(0)
    }

    /// Insert as MRU and hand back the entry pushed past `capacity`, if any.
    ///
    /// Split from [`Self::insert`] so the LRU order and eviction can be unit
    /// tested without a GPU — the freeing half is the only part that needs
    /// one. At most one entry can be evicted, since entries go in one at a
    /// time.
    fn push(&mut self, entry: CondEntry) -> Option<CondEntry> {
        self.entries.insert(0, entry);
        if self.entries.len() > self.capacity {
            return self.entries.pop();
        }
        None
    }

    /// Insert as MRU, evicting (and freeing) the LRU tail past `capacity`.
    fn insert(&mut self, gpu: &mut Gpu, entry: CondEntry) -> Result<usize, String> {
        if let Some(old) = self.push(entry) {
            gpu.free_tensor(old.t5_hidden)
                .map_err(|e| format!("cond cache: free evicted t5_hidden: {e:?}"))?;
        }
        Ok(0)
    }

    /// Free every cached device tensor and empty the cache.
    pub fn clear(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        for e in self.entries.drain(..) {
            gpu.free_tensor(e.t5_hidden)
                .map_err(|err| format!("cond cache: free t5_hidden: {err:?}"))?;
        }
        Ok(())
    }
}

impl FluxPipeBundle {
    /// Optional activation override: `None` = BFL GELU-tanh (real weights).
    pub const fn mlp_act_default() -> MlpAct {
        MlpAct::GeluTanh
    }

    /// Which FLUX generation this pipe is.
    pub fn family(&self) -> FluxFamily {
        self.transformer_cfg.family
    }

    /// The host f32 transformer tables, materialised from the checkpoint if
    /// the bundle is in streaming mode.
    ///
    /// Borrowed when they are already resident, owned when they had to be
    /// decoded — and the owned case is ~47 GB at FLUX.1-dev geometry, so this
    /// is deliberately NOT cached in the bundle. Only the CPU reference path
    /// and the parity harnesses call it; the GPU path reads
    /// [`Self::gpu_weights`].
    pub fn transformer_host(&self) -> Result<Cow<'_, FluxWeights>, String> {
        if let Some(w) = &self.transformer {
            return Ok(Cow::Borrowed(w));
        }
        let s = self.sources.as_ref().ok_or(
            "flux pipe: no host transformer weights and no open checkpoint to \
             materialise them from",
        )?;
        Ok(Cow::Owned(s.transformer_plan.materialize(
            s.transformer.as_ref(),
            &self.transformer_cfg,
        )?))
    }

    /// The full host T5 tables for the CPU reference encoder, materialised
    /// from the checkpoint when the bundle only holds the light set.
    /// ~18.5 GB owned for T5-XXL, and NOT cached — the borrow-only form, for
    /// callers that hold `&self` and run once (the CPU reference path, the
    /// parity harnesses). A serving path that will encode again must use
    /// [`Self::latch_t5_host`] instead.
    pub fn t5_host(&self) -> Result<Cow<'_, T5Weights>, String> {
        let t5 = match &self.cond {
            TextCond::T5Clip { t5, .. } => t5,
            TextCond::Qwen3 { .. } => {
                return Err("flux pipe: this is a FLUX.2 pipe — it has no T5 encoder".into())
            }
        };
        if !t5.is_light() {
            return Ok(Cow::Borrowed(t5));
        }
        let plan = self.t5_plan()?;
        let s = self.sources.as_ref().expect("t5_plan implies sources");
        Ok(Cow::Owned(plan.materialize(s.text.as_ref())?))
    }

    /// The open T5 naming plan, or an error naming what is actually open.
    fn t5_plan(&self) -> Result<&t5::T5Plan, String> {
        match self.sources.as_ref().map(|s| &s.text_plan) {
            Some(TextPlan::T5(p)) => Ok(p),
            _ => Err(
                "flux pipe: T5 linears were streamed and no checkpoint is open to re-read them"
                    .into(),
            ),
        }
    }

    /// The FULL host Qwen3 tables for the CPU reference encoder,
    /// materialised from the checkpoint when the bundle only holds the light
    /// (embedding-only) set. The Klein twin of [`Self::t5_host`], with the
    /// same borrow-or-own contract: not cached, so a serving path that will
    /// encode again should use [`Self::latch_qwen3_host`].
    pub fn qwen3_host(&self) -> Result<Cow<'_, Qwen3Weights>, String> {
        let host = match &self.cond {
            TextCond::Qwen3 { host, .. } => host,
            TextCond::T5Clip { .. } => {
                return Err("klein: this is a FLUX.1 pipe — it has no Qwen3 encoder".into())
            }
        };
        if !host.is_light() {
            return Ok(Cow::Borrowed(host));
        }
        let (plan, src) = self.qwen3_source()?;
        Ok(Cow::Owned(plan.materialize(src)?))
    }

    /// The open Qwen3 naming plan and its checkpoint.
    fn qwen3_source(&self) -> Result<(&Qwen3Plan, &dyn ModelSource), String> {
        match self.sources.as_ref() {
            Some(PipeSources {
                text,
                text_plan: TextPlan::Qwen3(plan),
                ..
            }) => Ok((plan, text.as_ref())),
            _ => Err(
                "klein: the Qwen3 layers were streamed and no checkpoint is open to re-read them"
                    .into(),
            ),
        }
    }

    /// Materialise the full host Qwen3 tables **into the bundle and keep
    /// them** — the Klein twin of [`Self::latch_t5_host`], for the same
    /// reason: this runs only once the GPU encoder is out for the session, so
    /// every later prompt lands here too and re-decoding per prompt would be
    /// strictly worse than holding one copy.
    ///
    /// Idempotent: after the first call the host set is no longer light.
    pub fn latch_qwen3_host(&mut self) -> Result<&Qwen3Weights, String> {
        let light = matches!(&self.cond, TextCond::Qwen3 { host, .. } if host.is_light());
        if light {
            let (plan, src) = self.qwen3_source()?;
            let full = plan.materialize(src)?;
            eprintln!(
                "qwen3 host: the GPU encoder is unavailable, so the full host Qwen3 tables \
                 are now resident for the session — decoding them per prompt instead would \
                 be worse on a host already short of memory"
            );
            match &mut self.cond {
                TextCond::Qwen3 { host, .. } => *host = full,
                TextCond::T5Clip { .. } => unreachable!("checked by `light` above"),
            }
        }
        match &self.cond {
            TextCond::Qwen3 { host, .. } => Ok(host),
            TextCond::T5Clip { .. } => {
                Err("klein: this is a FLUX.1 pipe — it has no Qwen3 encoder".into())
            }
        }
    }

    /// Materialise the full host T5 tables **into the bundle and keep them**.
    ///
    /// This is the deliberate exception to the whole point of streaming, and
    /// it is chosen knowingly. It runs only when the GPU T5 upload was
    /// declined — an unsupported checkpoint, or the ~9.3 GB f16 allocation
    /// failing — after which `gpu_t5_declined` latches and every subsequent
    /// prompt routes to the host encoder. Re-decoding on each call would churn
    /// ~18.5 GB of allocation and 4.7 G bf16→f32 conversions **per uncached
    /// prompt**, on a host that has just told us it is short of memory. Paying
    /// the 18.5 GB once and holding it is strictly better than paying it
    /// repeatedly, even though holding it is what this task set out to avoid.
    ///
    /// Logs once, on the transition, because a bundle that silently grew by
    /// 18.5 GB is exactly the kind of thing a later memory investigation needs
    /// to see in the log.
    ///
    /// Idempotent: after the first call `self.t5` is no longer light, so this
    /// is a plain borrow.
    pub fn latch_t5_host(&mut self) -> Result<&T5Weights, String> {
        let light = matches!(&self.cond, TextCond::T5Clip { t5, .. } if t5.is_light());
        if light {
            let full = {
                let plan = self.t5_plan()?;
                let s = self.sources.as_ref().expect("t5_plan implies sources");
                plan.materialize(s.text.as_ref())?
            };
            eprintln!(
                "t5 host: the GPU encoder is unavailable, so the full host T5 tables \
                 (~18.5 GB for T5-XXL) are now resident for the session — decoding them \
                 per prompt instead would be worse on a host already short of memory"
            );
            match &mut self.cond {
                TextCond::T5Clip { t5, .. } => *t5 = full,
                TextCond::Qwen3 { .. } => unreachable!("checked by `light` above"),
            }
        }
        match &self.cond {
            TextCond::T5Clip { t5, .. } => Ok(t5),
            TextCond::Qwen3 { .. } => {
                Err("flux pipe: this is a FLUX.2 pipe — it has no T5 encoder".into())
            }
        }
    }

    /// Upload the transformer AND text-encoder weights to `gpu`, once.
    /// Idempotent: a subsequent `img_generate` reuses the resident buffers, so
    /// a multi-request serve session pays the upload only on the first GPU
    /// generation.
    ///
    /// **Streaming.** When the bundle still has its checkpoint open
    /// ([`PipeSources`], which is what [`load_pipe`] produces), the transformer
    /// and T5 weights are read tensor-by-tensor out of the mmap, converted to
    /// f16 on the host, and uploaded directly — no whole-model f32 host table
    /// is ever built, and none is left behind afterwards. The `from_host`
    /// route remains for bundles assembled without a checkpoint directory
    /// (synthetic fixtures).
    ///
    /// The text encoders are skipped when `HIPFIRE_T5_GPU=0`
    /// ([`text_encoders_on_gpu`]) — conditioning then runs through
    /// `t5::encode` / `clip::encode` on the host, which is the numeric oracle
    /// and the fallback for a checkpoint the GPU path refuses (ungated T5
    /// v1.0, non-`quick_gelu` CLIP). That host path needs the FULL T5 tables,
    /// which streaming does not keep, so it re-materialises them through
    /// [`Self::t5_host`] — the one case that still pays the ~18.5 GB.
    ///
    /// **VRAM cost of the text encoders**: T5-XXL is ~4.7 G parameters over 24
    /// layers, so its f16 residency is **~9.3 GB** (each layer is 4×4096² +
    /// 3×4096×10240 ≈ 193 M params); CLIP-L is **~0.25 GB**. That is on top of
    /// the transformer. On a card where that does not fit, the encoder upload
    /// is the thing that fails, not the transformer.
    ///
    /// **A failed text-encoder upload is NOT a failed `ensure_gpu`.** The host
    /// encoders serve every checkpoint, so an OOM (or an unsupported
    /// checkpoint) leaves the slot `None`, latches, logs the reason once, and
    /// lets conditioning run on the host. Propagating it would turn an
    /// `img_generate` that worked before this feature existed into a hard
    /// error — a strictly worse outcome than a slower one. A failed
    /// *transformer* upload does still propagate: nothing else can run it.
    pub fn ensure_gpu(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Before ANY upload: install the stream the whole diffusion path runs
        // on, once, permanently, for this `Gpu`. See
        // `flux_gpu::install_forward_stream` — it is deliberately not tied to
        // the activation dtype, and `HIPFIRE_FLUX_F16_ACT` does not affect it.
        flux_gpu::install_forward_stream(gpu)?;
        if self.gpu_weights.is_none() {
            // Streaming first: it never builds the ~47 GB f32 host table, and
            // on a unified-memory box that table is what pushes the whole
            // process (GPU allocations included) into swap.
            self.gpu_weights = Some(match (&self.sources, &self.transformer) {
                (Some(s), _) => GpuFluxWeights::from_stream(
                    gpu,
                    s.transformer.as_ref(),
                    &s.transformer_plan,
                    &self.transformer_cfg,
                )?,
                (None, Some(host)) => GpuFluxWeights::from_host(gpu, host, &self.transformer_cfg)?,
                (None, None) => {
                    return Err(
                        "flux pipe: ensure_gpu has neither host transformer weights nor an \
                         open checkpoint to stream from"
                            .into(),
                    )
                }
            });
        }
        // The VAE decoder weights are resident too. They used to be uploaded
        // and freed inside every `generate_txt2img_steps_gpu`, which paid
        // ~200 MB of host->device traffic and ~150 allocation/free pairs per
        // image on a device already holding the transformer. A config-only
        // VAE (`HIPFIRE_VAE_CONFIG_ONLY=1`) has no weights to upload and no
        // decode to run.
        if self.gpu_vae.is_none() && !self.vae.is_config_only() {
            self.gpu_vae = Some(vae_gpu::GpuVaeDecoderWeights::from_host(gpu, &self.vae)?);
        }
        // The VAE ENCODER, for the FLUX.2 edit path. Only a Klein pipe loads
        // one (`load_pipe`), and it is small beside the decoder — but a
        // reference image encoded on the host is a minutes-long scalar
        // convolution stack, so a resident device copy is the difference
        // between an edit request being interactive and being unusable.
        //
        // **A failed encoder upload is NOT a failed `ensure_gpu`**, for the
        // same reason a failed text-encoder upload is not: `vae::encode` on
        // the host serves every reference, so an OOM leaves the slot `None`,
        // latches, logs the reason once, and lets `build_ref_tokens` run on
        // the host. Propagating it would turn a plain Klein TXT2IMG request —
        // which never encodes a reference at all — into a hard load error
        // because a path it does not use could not be accelerated.
        if self.gpu_vae_enc.is_none() && !self.gpu_vae_enc_declined {
            if let Some(enc) = &self.vae_enc {
                let gb = vae_encoder_f32_gib(enc);
                match vae_gpu::GpuVaeEncoderWeights::from_host(gpu, enc) {
                    Ok(w) => self.gpu_vae_enc = Some(w),
                    Err(e) => {
                        eprintln!(
                            "vae encoder gpu: upload failed, staying on the host encoder \
                             (~{gb:.1} GB needed) — {e}"
                        );
                        self.gpu_vae_enc_declined = true;
                    }
                }
            }
        }
        let t5_plan = match self.sources.as_ref().map(|s| &s.text_plan) {
            Some(TextPlan::T5(p)) => Some(p),
            _ => None,
        };
        if let TextCond::T5Clip {
            t5,
            clip,
            gpu_t5,
            gpu_clip,
            gpu_t5_declined,
            gpu_clip_declined,
            ..
        } = &mut self.cond
        {
            if text_encoders_on_gpu() {
                if gpu_t5.is_none() && !*gpu_t5_declined {
                    // In streaming mode the checkpoint's own tensor names
                    // decide gated-vs-ungated, so the support check does not
                    // need (and must not need) a decoded host table.
                    let why = match t5_plan {
                        Some(p) => GpuT5Weights::unsupported_plan(&p.config, p.gated),
                        None => GpuT5Weights::unsupported(t5),
                    };
                    match why {
                        Some(why) => {
                            eprintln!("t5 gpu: staying on the host encoder — {why}");
                            *gpu_t5_declined = true;
                        }
                        None => {
                            let up = match (t5_plan, &self.sources) {
                                (Some(p), Some(s)) => {
                                    GpuT5Weights::from_stream(gpu, s.text.as_ref(), p)
                                }
                                _ => GpuT5Weights::from_host(gpu, t5),
                            };
                            match up {
                                Ok(w) => *gpu_t5 = Some(w),
                                Err(e) => {
                                    eprintln!(
                                        "t5 gpu: upload failed, staying on the host encoder \
                                         (~9.3 GB f16 needed) — {e}"
                                    );
                                    *gpu_t5_declined = true;
                                }
                            }
                        }
                    }
                }
                if gpu_clip.is_none() && !*gpu_clip_declined {
                    match GpuClipWeights::unsupported(clip) {
                        Some(why) => {
                            eprintln!("clip gpu: staying on the host encoder — {why}");
                            *gpu_clip_declined = true;
                        }
                        None => match GpuClipWeights::from_host(gpu, clip) {
                            Ok(w) => *gpu_clip = Some(w),
                            Err(e) => {
                                eprintln!(
                                    "clip gpu: upload failed, staying on the host encoder \
                                     (~0.25 GB f16 needed) — {e}"
                                );
                                *gpu_clip_declined = true;
                            }
                        },
                    }
                }
            }
        }
        // ── FLUX.2 Klein: the Qwen3 text encoder, same latch shape as T5 ───────
        // Read `self.sources` BEFORE borrowing `self.cond` mutably: the two
        // are disjoint fields, but `qwen3_source()` takes `&self` and would
        // borrow the whole bundle.
        let qwen3_src = match self.sources.as_ref() {
            Some(PipeSources {
                text,
                text_plan: TextPlan::Qwen3(p),
                ..
            }) => Some((p, text.as_ref())),
            _ => None,
        };
        if let TextCond::Qwen3 {
            host,
            gpu: gpu_qwen3,
            declined,
            ..
        } = &mut self.cond
        {
            if text_encoders_on_gpu() && gpu_qwen3.is_none() && !*declined {
                let gb = qwen3_f16_gib(&host.config);
                // Streaming first, for the same reason the transformer
                // streams: `from_host` needs the full ~16 GB f32 host set at
                // Klein 4B, which is precisely what the light load avoided.
                let up = match qwen3_src {
                    Some((p, src)) => GpuQwen3Weights::from_stream(gpu, src, p),
                    None => GpuQwen3Weights::from_host(gpu, host),
                };
                match up {
                    Ok(w) => *gpu_qwen3 = Some(w),
                    Err(e) => {
                        eprintln!(
                            "qwen3 gpu: upload failed, staying on the host encoder \
                             (~{gb:.1} GB f16 needed) — {e}"
                        );
                        *declined = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// Return every GPU buffer this bundle owns — transformer weights, the
    /// VAE decoder, text encoder weights, and the conditioning cache — to
    /// the pool. Idempotent.
    ///
    /// `GpuTensor`/`DeviceBuffer` have no `Drop`, so dropping a bundle without
    /// this leaks its whole device footprint for the life of the process.
    /// Returns the number of buffers freed.
    ///
    /// **Best-effort: every slot is emptied even when one release fails.** The
    /// conditioning cache is the small one (a few MB of embeddings) and the
    /// transformer is ~24 GB; short-circuiting on the cache's error used to
    /// leak everything behind it. Every `take`/`free_gpu` therefore runs, the
    /// FIRST error is kept, and it is returned only once all four slots are
    /// released.
    pub fn free_gpu(&mut self, gpu: &mut Gpu) -> Result<usize, String> {
        let mut freed = self.cond_cache.len();
        let mut first_err: Option<String> = None;
        if let Err(e) = self.cond_cache.clear(gpu) {
            first_err.get_or_insert(e);
        }
        if let TextCond::T5Clip {
            gpu_t5, gpu_clip, ..
        } = &mut self.cond
        {
            if let Some(t5) = gpu_t5.take() {
                freed += t5.free_gpu(gpu);
            }
            if let Some(clip) = gpu_clip.take() {
                freed += clip.free_gpu(gpu);
            }
        }
        if let TextCond::Qwen3 { gpu: gpu_qwen3, .. } = &mut self.cond {
            if let Some(q) = gpu_qwen3.take() {
                freed += q.free_gpu(gpu);
            }
        }
        if let Some(v) = self.gpu_vae_enc.take() {
            freed += v.free_gpu(gpu);
        }
        if let Some(v) = self.gpu_vae.take() {
            freed += v.free_gpu(gpu);
        }
        if let Some(w) = self.gpu_weights.take() {
            freed += w.free_gpu(gpu);
        }
        match first_err {
            Some(e) => Err(format!("{e} [released the remaining {freed} buffers]")),
            None => Ok(freed),
        }
    }
}

/// Device residency of a Qwen3 text encoder's f16 linears, in GiB — the number the
/// decline message quotes so an OOM says how much it wanted.
///
/// Counts the seven per-layer projections only: q/o are `heads*head_dim x
/// hidden`, k/v are `kv_heads*head_dim x hidden`, and gate/up/down are
/// `intermediate x hidden`. The norms are f32 vectors (kilobytes) and the
/// embedding table stays on the host, so both are noise at this scale.
fn qwen3_f16_gib(cfg: &crate::qwen3::Qwen3Config) -> f64 {
    let (d, inter) = (cfg.hidden as f64, cfg.intermediate as f64);
    let qd = (cfg.heads * cfg.head_dim) as f64;
    let kvd = (cfg.kv_heads * cfg.head_dim) as f64;
    let per_layer = 2.0 * qd * d + 2.0 * kvd * d + 3.0 * inter * d;
    2.0 * cfg.layers as f64 * per_layer / (1024.0 * 1024.0 * 1024.0)
}

/// Device residency of the VAE encoder, in GiB — the number the decline
/// message quotes so an OOM says how much it wanted.
///
/// Summed from the HOST tensors rather than re-derived from `VaeConfig`, so
/// it cannot drift from what `GpuVaeEncoderWeights::upload_into` actually
/// uploads: that path is `F16 = false` throughout (the encode's `Run` is
/// built with `conv_gemm = false`), so a device element is 4 bytes, exactly
/// like the host one.
fn vae_encoder_f32_gib(enc: &VaeEncoderWeights) -> f64 {
    fn resnet(r: &crate::vae::VaeResnet) -> usize {
        r.norm1_w.data.len()
            + r.norm1_b.data.len()
            + r.conv1_w.data.len()
            + r.conv1_b.data.len()
            + r.norm2_w.data.len()
            + r.norm2_b.data.len()
            + r.conv2_w.data.len()
            + r.conv2_b.data.len()
            + r.nin_shortcut_w.as_ref().map_or(0, |t| t.data.len())
            + r.nin_shortcut_b.as_ref().map_or(0, |t| t.data.len())
    }
    let mut n = enc.conv_in_w.data.len()
        + enc.conv_in_b.data.len()
        + enc.conv_norm_out_w.data.len()
        + enc.conv_norm_out_b.data.len()
        + enc.conv_out_w.data.len()
        + enc.conv_out_b.data.len();
    for b in &enc.down_blocks {
        n += b.resnets.iter().map(resnet).sum::<usize>()
            + b.downsample_w.as_ref().map_or(0, |t| t.data.len())
            + b.downsample_b.as_ref().map_or(0, |t| t.data.len());
    }
    n += enc.mid_resnet.iter().map(resnet).sum::<usize>();
    if let Some(a) = &enc.mid_attn {
        n += a.group_norm_w.data.len()
            + a.group_norm_b.data.len()
            + a.q_w.data.len()
            + a.q_b.data.len()
            + a.k_w.data.len()
            + a.k_b.data.len()
            + a.v_w.data.len()
            + a.v_b.data.len()
            + a.out_w.data.len()
            + a.out_b.data.len();
    }
    if let Some((w, b)) = &enc.quant_conv {
        n += w.data.len() + b.data.len();
    }
    4.0 * n as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// The attention route's per-request token budget: text + generated image +
/// reference tokens.
///
/// The routes carry no hard limit of their own; 32768 is what keeps the
/// LDS-free grid dimensions sane, and it is far above anything the edit path
/// reaches in practice (a 1024² generation is 1024 image tokens and four
/// 1024² references another 4096). Refusing by name beats a launch-geometry
/// failure deep inside a kernel.
pub const MAX_ROUTE_TOKENS: usize = 32_768;

/// Check one request's joint sequence against [`MAX_ROUTE_TOKENS`].
///
/// A standalone function so the rule is unit-testable without a device, and
/// so the error names all three counts: "too many tokens" is not actionable,
/// "4 references at 1024² is what did it" is.
fn check_route_budget(n_txt: usize, n_img: usize, n_ref: usize) -> Result<(), String> {
    let n_all = n_txt + n_img + n_ref;
    if n_all > MAX_ROUTE_TOKENS {
        return Err(format!(
            "edit: {n_all} tokens exceed the attention route budget of {MAX_ROUTE_TOKENS} \
             ({n_txt} text + {n_img} generated + {n_ref} reference)"
        ));
    }
    Ok(())
}

/// `HIPFIRE_IMG_PROFILE=1` prints per-stage wall time on the GPU txt2img path.
pub fn img_profile_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_IMG_PROFILE").is_ok_and(|v| v != "0")
}

/// Print one denoise step's per-kernel-family attribution to stderr, gated
/// (by the caller) on `HIPFIRE_PROFILE`. `table` is
/// [`flux_gpu::take_step_profile`]'s GPU families plus the `host.sampler` /
/// `host.other` entries the denoise loop adds; `wall_us` is the step's real
/// wall time, measured on the host around the whole loop iteration (the
/// `Gpuf`-internal GPU timers never see host-only work, so this is the only
/// place that can compute `gap`).
///
/// This is the "sum(kernel families) vs step wall time, gap" line: the gap
/// is the host-side work between kernels that no GPU timer attributes.
fn print_step_family_profile(
    step: usize,
    table: &std::collections::BTreeMap<&'static str, f64>,
    wall_us: f64,
) {
    let mut rows: Vec<(&&str, &f64)> = table.iter().collect();
    rows.sort_by(|a, b| b.1.total_cmp(a.1));
    eprintln!("[flux-profile] step {step}:");
    for (family, us) in &rows {
        eprintln!("  {family:<20} {:8.3} ms", **us / 1000.0);
    }
    let sum_us: f64 = table.values().sum();
    let gap_us = wall_us - sum_us;
    eprintln!(
        "  {:<20} sum={:.3}s wall={:.3}s gap={:.3}s ({:.1}%)",
        "[total]",
        sum_us / 1e6,
        wall_us / 1e6,
        gap_us / 1e6,
        100.0 * gap_us / wall_us
    );
}

/// Whether the T5/CLIP conditioning runs on the GPU. `HIPFIRE_T5_GPU=0` sends
/// it back to the host `t5::encode` / `clip::encode` reference — the escape
/// hatch when a numeric question needs the f32 oracle, not the normal route
/// (the host encode is 54.9 s per prompt at real geometry).
pub fn text_encoders_on_gpu() -> bool {
    hipfire_config::developer_var("HIPFIRE_T5_GPU").map_or(true, |v| v != "0")
}

/// Final-head adaLN chunk order for a checkpoint family.
///
/// BFL's `LastLayer` does `shift, scale = adaLN(vec).chunk(2)`; diffusers'
/// `AdaLayerNormContinuous` does `scale, shift = chunk(...)`. The halves are
/// therefore SWAPPED between the two families holding otherwise identical
/// weights.
///
/// This is a one-line mapping with an outsized blast radius: get it wrong and
/// the final projection is modulated by the wrong halves, the predicted
/// velocity is wrong, and the denoise loop never fully removes the noise — the
/// image keeps its composition but stays grainy. Nothing errors. The
/// block-parity gate cannot see it either, because it stops at the last single
/// block and never reaches the head.
pub const fn final_order_for_checkpoint(is_bfl: bool) -> FinalAdaLNOrder {
    if is_bfl {
        FinalAdaLNOrder::ShiftScale
    } else {
        FinalAdaLNOrder::ScaleShift
    }
}

/// Serialize a `[1, ch, h, w]` latent as a ComfyUI `.latent` file: safetensors
/// with an F32 `latent_tensor` plus an empty `latent_format_version_0` marker.
///
/// The marker is NOT optional. Without it ComfyUI's `LoadLatent` takes its
/// legacy SD1.5 branch and multiplies the tensor by `1/0.18215 ≈ 5.489`,
/// which decodes to a recognisable but violently over-saturated image — it
/// reads as a model bug and is not one. ComfyUI's own `SaveLatent` writes the
/// same zero-length tensor.
///
/// `data` must already be in VAE space (`÷scaling_factor + shift_factor`);
/// ComfyUI latents are post-`process_latent_out`, not raw model space.
pub fn comfy_latent_bytes(data: &[f32], ch: usize, h: usize, w: usize) -> Vec<u8> {
    assert_eq!(data.len(), ch * h * w, "latent length != ch*h*w");
    let n = data.len() * 4;
    let mut header = format!(
        r#"{{"latent_tensor":{{"dtype":"F32","shape":[1,{ch},{h},{w}],"data_offsets":[0,{n}]}},"latent_format_version_0":{{"dtype":"F32","shape":[0],"data_offsets":[{n},{n}]}}}}"#
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + n);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parse a ComfyUI `.latent` file, returning `(data, [b, ch, h, w])`.
///
/// The inverse of [`comfy_latent_bytes`]. Deliberately a minimal safetensors
/// reader rather than a dependency: this is used by the golden-latent gate,
/// which must be able to read what ComfyUI's `SaveLatent` writes.
pub fn read_comfy_latent(bytes: &[u8]) -> Result<(Vec<f32>, [usize; 4]), String> {
    if bytes.len() < 8 {
        return Err("comfy latent: file shorter than its header length".into());
    }
    let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr_end = 8 + hdr_len;
    if bytes.len() < hdr_end {
        return Err("comfy latent: truncated header".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..hdr_end])
        .map_err(|e| format!("comfy latent: header is not JSON: {e}"))?;
    let t = header
        .get("latent_tensor")
        .ok_or("comfy latent: no `latent_tensor` entry")?;
    let dtype = t.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
    let shape: Vec<usize> = t
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or("comfy latent: no shape")?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    if shape.len() != 4 {
        return Err(format!("comfy latent: expected a 4-D shape, got {shape:?}"));
    }
    let offs = t
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .ok_or("comfy latent: no data_offsets")?;
    let (a, b) = (
        offs[0].as_u64().unwrap_or(0) as usize,
        offs[1].as_u64().unwrap_or(0) as usize,
    );
    let raw = bytes
        .get(hdr_end + a..hdr_end + b)
        .ok_or("comfy latent: data_offsets past end of file")?;
    let data = crate::flux::decode_dtype(dtype, raw)?;
    let want: usize = shape.iter().product();
    if data.len() != want {
        return Err(format!(
            "comfy latent: {} elements for shape {shape:?} (want {want})",
            data.len()
        ));
    }
    Ok((data, [shape[0], shape[1], shape[2], shape[3]]))
}

/// `(rel_inf, rel_l2)` of `a` against reference `b`.
///
/// `rel_inf = max|a-b| / max|b|` and `rel_l2 = ||a-b|| / ||b||`, both
/// accumulated in f64 so a 1M-element latent does not lose the small
/// differences to the summation itself.
///
/// Report BOTH, because they answer different questions. `rel_l2` is the bulk
/// agreement and is what a tolerance should key on; `rel_inf` is dominated by
/// the single worst element, so it moves under an outlier that changes nothing
/// visually. A pair like `rel_inf 2.0e-1, rel_l2 2.7e-2` means "agrees almost
/// everywhere, disagrees at a few points" — reading either number alone would
/// have given the wrong verdict there.
pub fn latent_rel_error(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "latent length mismatch");
    let max_b = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut max_abs = 0f32;
    let mut sq_err = 0f64;
    let mut sq_ref = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sq_err += (d as f64) * (d as f64);
        sq_ref += (*y as f64) * (*y as f64);
    }
    (
        max_abs / max_b.max(1e-12),
        (sq_err.sqrt() / sq_ref.sqrt().max(1e-12)) as f32,
    )
}

/// Open a diffusers-format FLUX pipe directory into a host bundle.
///
/// The transformer's `config.json` decides the family, and the family decides
/// what else the directory must hold: FLUX.1 wants `text_encoder_2/` (T5-XXL),
/// `text_encoder/` (CLIP-L) and both tokenizers; FLUX.2 Klein wants
/// `text_encoder/` (the Qwen3 text encoder) and `tokenizer/tokenizer.json`, and no
/// `text_encoder_2/` exists at all.
pub fn load_pipe(pipe_dir: &Path) -> Result<FluxPipeBundle, String> {
    let open = |sub: &str| {
        SafetensorsSource::open(&pipe_dir.join(sub))
            .map_err(|e| format!("pipeline: cannot open {pipe_dir:?}/{sub}: {e:?}"))
    };
    let tx = open("transformer")?;

    let tx_cfg_json: serde_json::Value = serde_json::from_str(tx.metadata_json())
        .map_err(|e| format!("pipeline: transformer config.json invalid: {e}"))?;
    let tx_cfg_json = tx_cfg_json.get("config").cloned().unwrap_or(tx_cfg_json);
    let transformer_cfg = FluxDiffusionConfig::from_json(&tx_cfg_json)?;
    // Two naming conventions exist in the wild for the same weights: the
    // diffusers layout (`transformer_blocks.0.attn.to_q.weight`, sharded) and
    // the BFL single-file layout that ComfyUI and the official release ship
    // (`double_blocks.0.img_mod.lin.weight`). Detect rather than require a
    // conversion step, so a ComfyUI model tree loads directly — that is the
    // only FLUX checkpoint present on the bench machine.
    let transformer_plan = flux::FluxPlan::detect(&tx, &transformer_cfg);
    let is_bfl = transformer_plan.layout == flux::FluxLayout::Bfl;
    // Catch a key-layout mismatch before anything is decoded or uploaded. This
    // bites for DIFFUSERS checkpoints, whose plan is a hand-written mapping
    // that can disagree with the manifest's shapes. It is a tautology for BFL
    // ones, whose plan is BUILT from the manifest — there, a checkpoint that
    // is missing a tensor or has the wrong shape is still only caught when the
    // upload reaches that key (`FluxPlan::locate`, which names it), exactly as
    // `load_weights` did.
    transformer_plan.validate(&transformer_cfg)?;
    // BFL's `LastLayer` chunks (shift, scale); diffusers — which is the only
    // layout FLUX.2 Klein ships in — chunks (scale, shift).
    let final_order = final_order_for_checkpoint(is_bfl);
    let vae_src = open("vae")?;
    // `HIPFIRE_VAE_CONFIG_ONLY=1` loads the VAE config without its weights,
    // for callers that stop at the latent and decode elsewhere (see
    // `VaeDecoderWeights::config_only`). The default loads the full decoder,
    // which reads both the LDM/taming and diffusers FLUX VAE namings.
    let config_only =
        hipfire_config::developer_var("HIPFIRE_VAE_CONFIG_ONLY").is_ok_and(|v| v != "0");
    let vae = if config_only {
        VaeDecoderWeights::config_only(&vae_src)?
    } else {
        VaeDecoderWeights::load(&vae_src)?
    };

    // scheduler/scheduler_config.json
    let sched_raw = std::fs::read_to_string(pipe_dir.join("scheduler/scheduler_config.json"))
        .map_err(|e| format!("pipeline: scheduler config: {e}"))?;
    let sv: serde_json::Value = serde_json::from_str(&sched_raw)
        .map_err(|e| format!("pipeline: scheduler config.json invalid: {e}"))?;

    let flux2 = transformer_cfg.is_flux2();
    let meta = pipe_meta_from_scheduler(&sv, &vae, flux2)?;

    let (cond, text_src, text_plan, vae_enc) = if flux2 {
        // ── FLUX.2 Klein: one Qwen3 text encoder, no CLIP, no text_encoder_2 ──
        let text_src = open("text_encoder")?;
        let plan = Qwen3Plan::detect(&text_src)?;
        klein_check_text_encoder(&plan)?;
        // Streaming: the embedding table only. The CPU encoder materialises
        // the layers on demand (`qwen3_host`), the GPU one streams them.
        let host = plan.materialize_light(&text_src)?;
        let tokenizer_path = pipe_dir.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_tokenizer_json(&tokenizer_path)
            .map_err(|e| format!("klein: {}: {e:?}", tokenizer_path.display()))?
            .ok_or_else(|| format!("klein: {} missing", tokenizer_path.display()))?;
        let pad_id = tokenizer
            .special_token_id("<|endoftext|>")
            .unwrap_or(KLEIN_PAD_ID);
        // The reference (edit) path VAE-encodes its images on the host, so
        // the encoder half is loaded for Klein — and only for Klein, since a
        // FLUX.1 pipe has no path that can use it.
        let vae_enc = if config_only {
            None
        } else {
            Some(VaeEncoderWeights::load(&vae_src)?)
        };
        klein_check_packing(&transformer_cfg, &meta, &vae)?;
        (
            TextCond::Qwen3 {
                host,
                plan: plan.clone(),
                tokenizer,
                pad_id,
                tokenizer_path,
                gpu: None,
                declined: false,
            },
            text_src,
            TextPlan::Qwen3(plan),
            vae_enc,
        )
    } else {
        // ── FLUX.1: T5-XXL + CLIP-L ────────────────────────────────────
        let t5_src = open("text_encoder_2")?;
        let clip_src = open("text_encoder")?;
        // STREAMING MODE. Nothing model-sized is decoded here: the
        // transformer's f32 tables (~47 GB at FLUX.1-dev geometry) and T5's
        // linears (~18.5 GB) are read straight out of these mmaps by
        // `ensure_gpu`, one tensor at a time, and the CPU reference path
        // re-materialises them on demand. Loading them eagerly and keeping
        // them alongside the f16 device copies is what drove the host into
        // zram swap and the (unified-memory) GPU allocations with it.
        let t5_plan = t5::T5Plan::detect(&t5_src)?;
        let t5 = t5_plan.materialize_light(&t5_src)?;
        // CLIP-L (~0.5 GB f32) and the VAE decoder (~0.3 GB) stay fully
        // resident: both are needed on the host anyway — CLIP for its
        // token/position embedding gather, the VAE for `vae::decode` when no
        // GPU decoder is up — and together they are under 1 GB, well inside
        // the budget the streaming change exists to protect.
        let clip = ClipWeights::load(&clip_src)?;
        // Tokenizers (byte-exact vs the golden ids for ASCII prompts; the T5
        // Precompiled charsmap normalizer is identity here, non-ASCII
        // deferred).
        let t5_tok_raw = std::fs::read_to_string(pipe_dir.join("tokenizer_2/tokenizer.json"))
            .map_err(|e| format!("pipeline: t5 tokenizer.json: {e}"))?;
        let t5_tok_json: serde_json::Value = serde_json::from_str(&t5_tok_raw)
            .map_err(|e| format!("pipeline: t5 tokenizer.json invalid: {e}"))?;
        let t5_tokenizer = UnigramVocab::from_tokenizer_json(&t5_tok_json)?;
        let clip_tokenizer = Gpt2Bpe::load(&pipe_dir.join("tokenizer"))?;
        let clip_eot = clip_tokenizer.eot_id;
        let clip_bos = clip_tokenizer.bos_id;
        (
            TextCond::T5Clip {
                t5,
                clip,
                t5_tokenizer,
                clip_tokenizer,
                clip_bos,
                clip_eot,
                // CLIP-L pads up to 77 with the EOT id (`pad_with_end` in
                // ComfyUI's SDTokenizer); the file's vocab has no separate
                // pad token.
                clip_pad: clip_eot,
                t5_pad: 0,
                gpu_t5: None,
                gpu_clip: None,
                gpu_t5_declined: false,
                gpu_clip_declined: false,
            },
            t5_src,
            TextPlan::T5(t5_plan),
            None,
        )
    };

    Ok(FluxPipeBundle {
        transformer: None,
        transformer_cfg,
        cond,
        vae,
        vae_enc,
        meta,
        final_order,
        gpu_weights: None,
        gpu_vae: None,
        gpu_vae_enc: None,
        gpu_vae_enc_declined: false,
        cond_cache: CondCache::new(),
        sources: Some(PipeSources {
            transformer: Box::new(tx),
            transformer_plan,
            text: Box::new(text_src),
            text_plan,
        }),
    })
}

/// The sidecar packs that accompany a trunk pack, one set per family.
/// `hipfire-quantize --flux-pipe` writes them next to the trunk as
/// `<base>-t5.hfq` / `<base>-clip.hfq` / `<base>-vae.hfq` (FLUX.1) or
/// `<base>-qwen3.hfq` / `<base>-vae.hfq` (FLUX.2 Klein).
pub enum HfqSidecars {
    Flux1 {
        t5: hipfire_runtime::hfq::HfqFile,
        clip: hipfire_runtime::hfq::HfqFile,
        vae: hipfire_runtime::hfq::HfqFile,
    },
    Flux2 {
        qwen3: hipfire_runtime::hfq::HfqFile,
        vae: hipfire_runtime::hfq::HfqFile,
    },
}

/// Load a FLUX pipe from HFQ component packs (the output of
/// `hipfire-quantize --flux-pipe …`): the trunk (arch 40 FLUX.1 or arch 45
/// FLUX.2 Klein) plus its sidecars, each an HFQ file whose metadata envelope
/// carries its own `config` (and, on the trunk, the scheduler config plus the
/// embedded tokenizer blobs). This is the only form the daemon loads; a
/// diffusers pipe directory is the packer's input.
///
/// Everything downstream consumes the same loaders as [`load_pipe`] because
/// each reads through `&dyn ModelSource`; [`HfqModelSource`] bridges the HFQ
/// tensor index to that trait. The trunk's family must match the sidecar
/// set, so a mis-paired file fails here by name instead of reaching the wrong
/// text-encoder loader.
pub fn load_pipe_hfq(
    trunk: hipfire_runtime::hfq::HfqFile,
    sidecars: HfqSidecars,
) -> Result<FluxPipeBundle, String> {
    use hipfire_runtime::hfq::HfqModelSource;

    let tx = Box::new(HfqModelSource::from_hfq(trunk));
    let meta_value: serde_json::Value = serde_json::from_str(tx.metadata_json())
        .map_err(|e| format!("pipeline: transformer .hfq metadata invalid: {e}"))?;
    let cfg_json = meta_value
        .get("config")
        .cloned()
        .ok_or("pipeline: transformer .hfq metadata has no `config` object")?;
    let transformer_cfg = FluxDiffusionConfig::from_json(&cfg_json)?;
    let flux2 = transformer_cfg.is_flux2();
    let transformer_plan = flux::FluxPlan::detect(&*tx, &transformer_cfg);
    let is_bfl = transformer_plan.layout == flux::FluxLayout::Bfl;
    transformer_plan.validate(&transformer_cfg)?;
    let final_order = final_order_for_checkpoint(is_bfl);

    let (text_hfq, clip_hfq, vae_hfq) = match (flux2, sidecars) {
        (false, HfqSidecars::Flux1 { t5, clip, vae }) => (t5, Some(clip), vae),
        (true, HfqSidecars::Flux2 { qwen3, vae }) => (qwen3, None, vae),
        (false, HfqSidecars::Flux2 { .. }) => {
            return Err(
                "flux (HFQ): the trunk is FLUX.1 (arch 40) but the sidecars are a \
                        FLUX.2 Klein set (qwen3 + vae); it needs t5 + clip + vae"
                    .into(),
            )
        }
        (true, HfqSidecars::Flux1 { .. }) => {
            return Err(
                "flux (HFQ): the trunk is FLUX.2 Klein (arch 45) but the sidecars \
                        are a FLUX.1 set (t5 + clip + vae); it needs qwen3 + vae"
                    .into(),
            )
        }
    };

    let vae_box = Box::new(HfqModelSource::from_hfq(vae_hfq));
    let config_only =
        hipfire_config::developer_var("HIPFIRE_VAE_CONFIG_ONLY").is_ok_and(|v| v != "0");
    let vae = if config_only {
        VaeDecoderWeights::config_only(&*vae_box)?
    } else {
        VaeDecoderWeights::load(&*vae_box)?
    };

    let sv = meta_value
        .get("scheduler_config")
        .cloned()
        .ok_or("pipeline: transformer .hfq metadata has no `scheduler_config`")?;
    let meta = pipe_meta_from_scheduler(&sv, &vae, flux2)?;

    let tokenizers = meta_value
        .get("tokenizer")
        .cloned()
        .ok_or("pipeline: transformer .hfq metadata has no embedded `tokenizer` blobs")?;

    // The text encoder streams like the dir load: only the light set is
    // decoded here, the layers are read out of the pack by `ensure_gpu` (or
    // materialised on demand by the CPU path).
    let text_box = Box::new(HfqModelSource::from_hfq(text_hfq));
    let (cond, text_plan, vae_enc) = if flux2 {
        // ── FLUX.2 Klein: one Qwen3 text encoder, no CLIP ──
        let plan = Qwen3Plan::detect(&*text_box)?;
        klein_check_text_encoder(&plan)?;
        let host = plan.materialize_light(&*text_box)?;
        let qwen_tok = tokenizers
            .get("qwen")
            .and_then(|x| x.as_str())
            .ok_or("pipeline: transformer .hfq metadata tokenizer has no `qwen`")?;
        let tokenizer = Tokenizer::from_hf_json(qwen_tok)
            .map_err(|e| format!("klein: embedded qwen tokenizer.json: {e:?}"))?;
        let pad_id = tokenizer
            .special_token_id("<|endoftext|>")
            .unwrap_or(KLEIN_PAD_ID);
        // The tokenizer lives inside the trunk pack; name it that way in
        // diagnostics.
        let tokenizer_path = PathBuf::from(format!("{}#tokenizer.qwen", tx.path().display()));
        // The reference (edit) path VAE-encodes its images on the host, so
        // the encoder half is loaded for Klein — and only for Klein.
        let vae_enc = if config_only {
            None
        } else {
            Some(VaeEncoderWeights::load(&*vae_box)?)
        };
        klein_check_packing(&transformer_cfg, &meta, &vae)?;
        (
            TextCond::Qwen3 {
                host,
                plan: plan.clone(),
                tokenizer,
                pad_id,
                tokenizer_path,
                gpu: None,
                declined: false,
            },
            TextPlan::Qwen3(plan),
            vae_enc,
        )
    } else {
        // ── FLUX.1: T5-XXL + CLIP-L ──
        let t5_tok_json = tokenizers
            .get("t5")
            .cloned()
            .ok_or("pipeline: transformer .hfq metadata tokenizer has no `t5`")?;
        let t5_tokenizer = UnigramVocab::from_tokenizer_json(&t5_tok_json)?;
        let clip_vocab = tokenizers
            .get("clip_vocab")
            .ok_or("pipeline: transformer .hfq metadata tokenizer has no `clip_vocab`")?;
        let clip_merges = tokenizers
            .get("clip_merges")
            .and_then(|x| x.as_str())
            .ok_or("pipeline: transformer .hfq metadata tokenizer has no `clip_merges`")?;
        let clip_tokenizer = Gpt2Bpe::from_parts(clip_vocab, clip_merges)?;
        let clip_eot = clip_tokenizer.eot_id;
        let clip_bos = clip_tokenizer.bos_id;
        let t5_plan = t5::T5Plan::detect(&*text_box)?;
        let t5_light = t5_plan.materialize_light(&*text_box)?;
        let clip_box = Box::new(HfqModelSource::from_hfq(
            clip_hfq.expect("a FLUX.1 sidecar set carries the clip pack"),
        ));
        let clip_weights = ClipWeights::load(&*clip_box)?;
        (
            TextCond::T5Clip {
                t5: t5_light,
                clip: clip_weights,
                t5_tokenizer,
                clip_tokenizer,
                clip_bos,
                clip_eot,
                // CLIP-L pads up to 77 with the EOT id (`pad_with_end`); the
                // vocab has no separate pad token — same as the dir load.
                clip_pad: clip_eot,
                t5_pad: 0,
                gpu_t5: None,
                gpu_clip: None,
                gpu_t5_declined: false,
                gpu_clip_declined: false,
            },
            TextPlan::T5(t5_plan),
            None,
        )
    };

    Ok(FluxPipeBundle {
        transformer: None,
        transformer_cfg,
        cond,
        vae,
        vae_enc,
        meta,
        final_order,
        gpu_weights: None,
        gpu_vae: None,
        gpu_vae_enc: None,
        gpu_vae_enc_declined: false,
        cond_cache: CondCache::new(),
        sources: Some(PipeSources {
            transformer: tx,
            transformer_plan,
            text: text_box,
            text_plan,
        }),
    })
}

/// The Klein conditioning taps the residual stream after layer
/// `max(KLEIN_TAPS)`; a shorter text encoder would condition on zeros.
fn klein_check_text_encoder(plan: &Qwen3Plan) -> Result<(), String> {
    let need = KLEIN_TAPS.iter().copied().max().unwrap_or(0);
    if plan.config.layers < need {
        return Err(format!(
            "klein: text_encoder has {} layers but the Klein conditioning taps the \
             residual stream after layer {need} ({KLEIN_TAPS:?}) — a shorter text encoder \
             would condition on zeros",
            plan.config.layers
        ));
    }
    Ok(())
}

/// Cross-config: the transformer's packed token width must be exactly what
/// the VAE hands it, and the latent BatchNorm must have one statistic per
/// packed column. A mismatch here decodes to noise rather than failing, so it
/// is checked once, at load, with both numbers named.
fn klein_check_packing(
    transformer_cfg: &FluxDiffusionConfig,
    meta: &PipeMeta,
    vae: &VaeDecoderWeights,
) -> Result<(), String> {
    let packed = transformer_cfg.patch_in();
    if let LatentNorm::BatchNorm { mean, .. } = &meta.latent_norm {
        if mean.len() != packed {
            return Err(format!(
                "klein pipe: the VAE latent BatchNorm has {} channels but the \
                 transformer packs {packed} per token",
                mean.len()
            ));
        }
    }
    let vae_packed =
        vae.config.latent_channels * vae.config.latent_patch.0 * vae.config.latent_patch.1;
    if vae_packed != packed {
        return Err(format!(
            "klein pipe: the VAE packs {vae_packed} columns per token \
             (latent_channels {} x patch {}x{}) but the transformer's in_channels \
             give {packed}",
            vae.config.latent_channels, vae.config.latent_patch.0, vae.config.latent_patch.1
        ));
    }
    Ok(())
}

/// Scheduler + latent-norm coefficients from a parsed `scheduler_config`
/// object. [`load_pipe`] reads the pipe's file; HFQ packs embed the same JSON
/// in the trunk metadata.
fn pipe_meta_from_scheduler(
    sv: &serde_json::Value,
    vae: &VaeDecoderWeights,
    flux2: bool,
) -> Result<PipeMeta, String> {
    let f = |k: &str, def: f32| {
        sv.get(k)
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(def)
    };
    let u = |k: &str, def: u32| {
        sv.get(k)
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .unwrap_or(def)
    };
    Ok(PipeMeta {
        num_train_timesteps: u("num_train_timesteps", 1000),
        shift_rule: if flux2 {
            ShiftRule::Empirical
        } else {
            ShiftRule::Fixed(f("shift", 1.0))
        },
        base_image_seq_len: u("base_image_seq_len", 256) as usize,
        max_image_seq_len: u("max_image_seq_len", 4096) as usize,
        base_shift: f("base_shift", 0.5),
        max_shift: f("max_shift", 1.15),
        latent_norm: vae.latent_norm.clone(),
        max_seq: if flux2 { KLEIN_MIN_LEN } else { 256 },
    })
}

/// Tokenize a prompt into the framed conditioning inputs for the pipe's
/// family.
///
/// FLUX.1: the diffusers framing — `txt_ids/mask` (T5) padded to `txt_seq`,
/// `clip_ids/mask` padded to 77.
///
/// FLUX.2 Klein: the ComfyUI chat template through the Qwen3 tokenizer,
/// right-padded to at least `txt_seq` and NEVER truncated (a long prompt
/// keeps every token), with the CLIP halves empty — Klein has no pooled
/// conditioning at all.
pub fn condition_prompt(
    b: &FluxPipeBundle,
    prompt: &str,
    txt_seq: usize,
) -> Result<Txt2ImgConditioning, String> {
    match &b.cond {
        TextCond::T5Clip {
            t5_tokenizer,
            clip_tokenizer,
            clip_bos,
            clip_eot,
            clip_pad,
            t5_pad,
            ..
        } => {
            let (mut txt_ids, mut txt_mask) = encode_t5(t5_tokenizer, prompt, 1);
            // T5 frame: [ids..., </s>] padded with pad (golden MAX_SEQ=64)
            if txt_ids.len() < txt_seq {
                txt_ids.resize(txt_seq, *t5_pad);
                txt_mask.resize(txt_seq, 0);
            } else {
                txt_ids.truncate(txt_seq);
                txt_mask.truncate(txt_seq);
            }
            let mut clip_ids = vec![*clip_bos];
            clip_ids.extend(clip_tokenizer.encode(prompt));
            clip_ids.push(*clip_eot);
            if clip_ids.len() < 77 {
                clip_ids.resize(77, *clip_pad);
            } else {
                clip_ids.truncate(77);
            }
            let clip_mask = vec![1u8; 77];
            Ok(Txt2ImgConditioning {
                txt_ids,
                txt_mask,
                clip_ids,
                clip_mask,
            })
        }
        TextCond::Qwen3 {
            tokenizer,
            pad_id,
            tokenizer_path,
            ..
        } => {
            let p = klein_prompt::encode_klein_prompt(tokenizer, prompt, *pad_id, txt_seq);
            // The template opens with `<|im_start|>`, so the first id must be
            // that token. It would not be if the tokenizer prepended a BOS
            // (`add_bos_token`) or did not register the chatml specials as
            // atomic tokens — both silently shift every position and change
            // the conditioning rather than failing.
            let im_start = tokenizer.special_token_id("<|im_start|>").ok_or_else(|| {
                format!(
                    "klein: {} has no `<|im_start|>` token — it is not a Qwen3 chatml \
                     tokenizer",
                    tokenizer_path.display()
                )
            })?;
            if p.ids.first() != Some(&im_start) {
                return Err(format!(
                    "klein: the templated prompt tokenizes to {:?}..., not `<|im_start|>` \
                     (id {im_start}) — check {}",
                    &p.ids[..p.ids.len().min(4)],
                    tokenizer_path.display()
                ));
            }
            Ok(Txt2ImgConditioning {
                txt_ids: p.ids,
                txt_mask: p.mask,
                clip_ids: Vec::new(),
                clip_mask: Vec::new(),
            })
        }
    }
}

/// Pre-tokenized conditioning (same shape the parity golden records). The
/// `clip_*` halves are empty for FLUX.2 Klein, which has no CLIP encoder.
pub struct Txt2ImgConditioning {
    pub txt_ids: Vec<u32>,
    pub txt_mask: Vec<u8>,
    pub clip_ids: Vec<u32>,
    pub clip_mask: Vec<u8>,
}

/// Inputs for one CPU txt2img run (pre-tokenized; tokenizer wiring lives
/// with the daemon path).
pub struct Txt2ImgInput<'a> {
    /// Text-encoder ids: T5 for FLUX.1, the templated Qwen3 prompt for
    /// FLUX.2 Klein.
    pub txt_ids: &'a [u32],
    pub txt_mask: &'a [u8],
    /// CLIP framing; empty for FLUX.2 Klein.
    pub clip_ids: &'a [u32],
    pub clip_mask: &'a [u8],
    /// Encoded reference images (the FLUX.2 edit path), appended to the image
    /// stream after the generated tokens and held FIXED across every step.
    /// Empty for plain txt2img.
    pub references: &'a [RefTokens],
    /// Optional PACKED init latents (`n_img × patch_in`, the transformer
    /// input layout); `None` = zeros (the golden always supplies them; the
    /// CLI path seeds and packs its own noise).
    pub init_latents: Option<&'a [f32]>,
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    /// Single-block MLP activation — GELU-tanh (BFL, diffusers and
    /// ComfyUI agree; the knob exists to pin any future deviation).
    pub mlp_act: MlpAct,
    /// Prompt half of the conditioning-cache key, paired with
    /// `txt_ids.len()`. `None` (a pre-tokenized harness with no prompt string)
    /// BYPASSES the cache: the request encodes once, owns its conditioning for
    /// the generation, and frees it at the end — it is never looked up and
    /// never inserted, so two different pre-tokenized inputs cannot key alike.
    /// GPU path only; the CPU path has no cache.
    pub prompt_key: Option<&'a str>,
}

/// One reference image, VAE-encoded and packed into the transformer's image
/// stream: `packed` is `[n, patch_in]`, `ids` its `n` 4-axis RoPE ids.
///
/// Reference tokens are computed ONCE, before the denoise loop, and are
/// unchanged by it: they condition every step but are never denoised. Their
/// time axis (`10 * (i + 1)`, index-dependent) is what separates one
/// reference from another — and both from the generated grid at time 0 — in
/// RoPE space.
#[derive(Debug, Clone)]
pub struct RefTokens {
    pub packed: Vec<f32>,
    pub ids: Vec<[f32; 4]>,
    pub n: usize,
}

/// VAE-encode, pack and normalize the reference images for the edit path.
///
/// The result is in the SAME space the denoise loop's latents live in: the
/// packed columns are normalized with the pipe's [`LatentNorm`], so a
/// reference token and a generated token are directly concatenable.
///
/// One function for both backends: pass `Some(gpu)` and the encode runs on
/// [`vae_gpu::gpu_encode`] against the resident encoder weights, `None` (or a
/// bundle whose encoder was never uploaded) and it falls back to the host
/// [`vae::encode`] reference. Everything after the encode — the geometry
/// checks, the 2×2 pack, the [`LatentNorm`] normalize and the `10*(i+1)` RoPE
/// time axis — is shared, so the two backends cannot drift in how a
/// reference becomes a token.
///
/// Public because the golden gate (`gpu_klein_golden_latent --ref-latent`)
/// compares its output against a ComfyUI `VAEEncode` → `SaveLatent` of the
/// same reference. That check has to run the PRODUCT's encode path, not a
/// re-implementation of it in the example — a parallel copy would agree with
/// itself while the shipped path drifted.
pub fn build_ref_tokens(
    b: &FluxPipeBundle,
    gpu: Option<&mut Gpu>,
    refs: &[RefImage],
) -> Result<Vec<RefTokens>, String> {
    if refs.is_empty() {
        return Ok(Vec::new());
    }
    let mut gpu = gpu;
    let enc = b
        .vae_enc
        .as_ref()
        .ok_or("edit: this pipe has no VAE encoder (a config-only VAE cannot encode)")?;
    let latent = enc.config.latent_channels;
    // The encoder downsamples by 2^(blocks-1), the same factor the decoder
    // upsamples by, and the 2x2 latent patch packs 4 cells per token — so
    // both sides must be a multiple of 2 * up.
    let up = vae_upscale(b);
    let width = latent * 4;
    let mut out = Vec::with_capacity(refs.len());
    for (i, r) in refs.iter().enumerate() {
        if r.width % (2 * up) != 0 || r.height % (2 * up) != 0 {
            return Err(format!(
                "edit: reference {i} is {}x{}, which is not a multiple of {} (VAE \
                 compression {up} x the 2x2 latent patch)",
                r.width,
                r.height,
                2 * up
            ));
        }
        let want = enc.config.in_channels * r.width * r.height;
        if r.pixels.len() != want {
            return Err(format!(
                "edit: reference {i} carries {} floats, expected {want} \
                 ({} x {}x{})",
                r.pixels.len(),
                enc.config.in_channels,
                r.height,
                r.width
            ));
        }
        let z = match (&mut gpu, b.gpu_vae_enc.as_ref()) {
            (Some(g), Some(genc)) => vae_gpu::gpu_encode(g, genc, &r.pixels, r.height, r.width)?,
            _ => {
                // The host reference encode is a ~30-deep scalar convolution
                // stack: milliseconds on the device, MINUTES at 1024². Say so
                // once per reference, because from the wire the only symptom
                // is an `img_generate` that appears to hang before step 1 —
                // and `ensure_gpu` declining the encoder upload (an OOM) is
                // the normal way to end up here.
                eprintln!(
                    "reference encode: host VAE encoder (minutes at 1024²) — \
                     reference {i} at {}x{}",
                    r.width, r.height
                );
                vae::encode(enc, &r.pixels, r.height, r.width)
            }
        };
        let (lh, lw) = (r.height / up, r.width / up);
        let (packed, n) = scheduler::pack_latents(&z, latent, lh, lw);
        let packed = scheduler::normalize_packed(&packed, n, width, &b.meta.latent_norm);
        // Time axis 10*(i+1): reference order is semantic, and index 0 must
        // not collide with the generated grid's time 0.
        let ids = flux::rope_ids_for_grid((lh / 2, lw / 2), 10.0 * (i as f32 + 1.0));
        debug_assert_eq!(ids.len(), n);
        out.push(RefTokens { packed, ids, n });
    }
    Ok(out)
}

/// Per-step record for the parity harness.
#[derive(Debug, Clone)]
pub struct StepRecord {
    pub t_model: f32,
    pub latents_in: Vec<f32>,
    pub noise_pred: Vec<f32>,
    pub latents_out: Vec<f32>,
}

/// txt2img outputs: per-step latents, decoded image tensor and PNG bytes.
pub struct Txt2ImgOutput {
    pub steps: Vec<StepRecord>,
    /// Decoded VAE output, `[3][height][width]`.
    pub image: Vec<f32>,
    pub image_shape: (usize, usize),
    pub png: Vec<u8>,
}

/// Run the CPU txt2img pipeline (batch 1).
pub fn generate_txt2img(b: &FluxPipeBundle, input: &Txt2ImgInput) -> Result<Txt2ImgOutput, String> {
    generate_txt2img_steps(b, input, &mut |_, _| {})
}

/// Make sure the conditioning for this request is in `b.cond_cache`, and
/// return its index.
///
/// Cache key is `(prompt, t5_seq)` — `input.prompt_key` and
/// `input.txt_ids.len()`. A request with no `prompt_key` (a pre-tokenized
/// parity harness) BYPASSES the cache entirely: [`is_cacheable`] requires
/// `Some`, so such a request neither looks up nor inserts, and its
/// conditioning is owned by the generation and freed with it. That is what
/// keeps two DIFFERENT pre-tokenized inputs at the same `t5_seq` from
/// colliding on a shared `""` key. `generate_txt2img_prompt_gpu` — the
/// daemon/CLI entry, and the only path a user reaches — always passes a key,
/// so the cache is live on the product path.
///
/// Route selection:
/// - `HIPFIRE_T5_GPU` unset/≠0 **and** both encoders uploaded → GPU encoders.
/// - otherwise → the host `t5::encode`/`clip::encode` reference, whose
///   `[len, d_model]` result is uploaded so the denoise loop sees the same
///   device-resident tensor either way. That keeps exactly ONE downstream
///   code path, so the fallback cannot silently diverge in how `txt` is fed.
fn ensure_conditioning(
    b: &mut FluxPipeBundle,
    gpu: &mut Gpu,
    input: &Txt2ImgInput,
) -> Result<CondSlot, String> {
    // Cacheable only with a real prompt key. A `None` key must NOT fall back
    // to `""`: every pre-tokenized caller (the golden-latent, velocity-step1
    // and pipeline-parity harnesses) passes `None`, so `""` is a live key that
    // two DIFFERENT pre-tokenized inputs at the same `t5_seq` would both hit
    // in one process — the second silently getting the first's conditioning.
    // Uncacheable requests encode, run, and free.
    let cacheable = is_cacheable(b.cond_cache.is_enabled(), input.prompt_key);
    let t5_seq = input.txt_ids.len();
    let family = b.transformer_cfg.family;
    if cacheable {
        let prompt = input.prompt_key.expect("cacheable implies Some");
        if let Some(idx) = b.cond_cache.lookup(prompt, family, t5_seq) {
            return Ok(CondSlot::Cached(idx));
        }
    }

    // The two encoders route INDEPENDENTLY. A CLIP checkpoint the GPU path
    // refuses must not drag T5 back to the host with it — T5 is 54.9 s there
    // against CLIP's 0.24 s, so a shared verdict would trade the entire win
    // for the cheap half.
    let on_gpu = text_encoders_on_gpu();
    let txt_dim = b.transformer_cfg.txt_hidden_dim;

    // ── FLUX.2 Klein: one encoder, and no pooled vector at all ───────
    // The Qwen3 tap concat IS the txt stream, so the entry's `t5_hidden`
    // holds it and `clip_pooled` is empty — the FLUX.2 forward never reads
    // a pooled vector (`pooled_projection_dim` is 0). Sharing `CondEntry`
    // rather than adding a variant is what keeps `denoise_and_decode` one
    // code path for both families; the `family` half of the cache key is
    // what stops a Klein entry ever being handed to a FLUX.1 forward.
    if matches!(&b.cond, TextCond::Qwen3 { .. }) {
        let txt_dev = klein_conditioning(b, gpu, input, on_gpu, txt_dim)?;
        let entry = CondEntry {
            prompt: input.prompt_key.unwrap_or_default().to_string(),
            family,
            t5_seq,
            t5_hidden: txt_dev,
            clip_pooled: Vec::new(),
        };
        return if cacheable {
            Ok(CondSlot::Cached(b.cond_cache.insert(gpu, entry)?))
        } else {
            Ok(CondSlot::Owned(entry))
        };
    }

    let t5_uploaded = matches!(&b.cond, TextCond::T5Clip { gpu_t5, .. } if gpu_t5.is_some());

    let t5_hidden = if on_gpu && t5_uploaded {
        // Disjoint field borrows: `gpu_t5` mutably, `t5` (the light host set:
        // embedding table + relative bias) immutably.
        let TextCond::T5Clip { t5, gpu_t5, .. } = &mut b.cond else {
            unreachable!("t5_uploaded implies the T5Clip variant")
        };
        let gt5 = gpu_t5.as_mut().expect("checked is_some");
        t5_gpu::encode(gpu, gt5, t5, input.txt_ids, input.txt_mask)?
    } else {
        // The host encoder needs the FULL tables. Latch them into the bundle
        // rather than decoding per prompt: this branch means the GPU encoder
        // is out for the session, so every future prompt lands here too. See
        // `FluxPipeBundle::latch_t5_host` for why holding ~18.5 GB beats
        // re-decoding it.
        let (hidden, _t5_layers) = {
            let host = b.latch_t5_host()?;
            t5::encode(host, input.txt_ids, input.txt_mask)
        };
        // Uploaded so the denoise loop sees the same device-resident
        // tensor either way: exactly ONE downstream path, so the host
        // fallback cannot diverge in how `txt` reaches `txt_in`.
        let rows = input.txt_ids.len();
        if hidden.len() != rows * txt_dim {
            return Err(format!(
                "cond: host t5 hidden is {} floats, expected {rows}x{txt_dim} — the T5 \
                 d_model must equal the transformer's txt_hidden_dim",
                hidden.len()
            ));
        }
        gpu.upload_f32(&hidden, &[rows, txt_dim])
            .map_err(|e| format!("cond: upload host t5_hidden: {e:?}"))?
    };

    if t5_hidden.shape != vec![input.txt_ids.len(), txt_dim] {
        // Freed before bailing: `GpuTensor` has no `Drop`, and this is the
        // one place where an owned tensor exists outside the cache.
        let _ = gpu.free_tensor(t5_hidden);
        return Err(format!(
            "cond: t5 hidden shape mismatch — the T5 d_model must equal the \
             transformer's txt_hidden_dim ({txt_dim})"
        ));
    }

    let TextCond::T5Clip { clip, gpu_clip, .. } = &b.cond else {
        unreachable!("the Qwen3 early return above")
    };
    let clip_pooled = match (on_gpu, gpu_clip.as_ref()) {
        (true, Some(gclip)) => {
            match clip_gpu::encode(gpu, gclip, clip, input.clip_ids, input.clip_mask) {
                Ok((_last, pooled)) => pooled,
                Err(e) => {
                    // The T5 tensor is already on the device and owned by
                    // nothing yet; release it rather than leak the generation.
                    let _ = gpu.free_tensor(t5_hidden);
                    return Err(e);
                }
            }
        }
        _ => clip::encode(clip, input.clip_ids, input.clip_mask).1,
    };

    let entry = CondEntry {
        prompt: input.prompt_key.unwrap_or_default().to_string(),
        family,
        t5_seq,
        t5_hidden,
        clip_pooled,
    };
    if cacheable {
        Ok(CondSlot::Cached(b.cond_cache.insert(gpu, entry)?))
    } else {
        Ok(CondSlot::Owned(entry))
    }
}

/// FLUX.2 Klein's whole text conditioning: the Qwen3 residual stream after
/// the three [`KLEIN_TAPS`] layers, concatenated per token into
/// `[len, taps * hidden]` and left **on the device**.
///
/// Route selection mirrors T5's, for the same reasons:
/// - `HIPFIRE_T5_GPU` unset/≠0 **and** the text encoder uploaded → [`qwen3_gpu::encode_taps`],
///   whose result is already a device tensor. That IS the txt stream — the
///   FLUX.2 forward's `context_embedder` reads it in place, so this route
///   never downloads and never re-uploads the conditioning.
/// - otherwise → the host [`qwen3::encode_taps`] f32 oracle, whose result is
///   uploaded so the denoise loop sees the same device-resident tensor either
///   way. Exactly ONE downstream path, so the fallback cannot diverge in how
///   `txt` reaches `txt_in`.
///
/// The width check is the load-bearing one: `taps.len() * hidden` must equal
/// the transformer's `joint_attention_dim` (7680 = 3 × 2560 on Klein 4B). A
/// mismatch would otherwise be a silently reinterpreted GEMM.
fn klein_conditioning(
    b: &mut FluxPipeBundle,
    gpu: &mut Gpu,
    input: &Txt2ImgInput,
    on_gpu: bool,
    txt_dim: usize,
) -> Result<GpuTensor, String> {
    let rows = input.txt_ids.len();
    let uploaded = matches!(&b.cond, TextCond::Qwen3 { gpu, .. } if gpu.is_some());
    let dev = if on_gpu && uploaded {
        let TextCond::Qwen3 {
            host, gpu: gqwen, ..
        } = &b.cond
        else {
            unreachable!("uploaded implies the Qwen3 variant")
        };
        let gw = gqwen.as_ref().expect("checked is_some");
        qwen3_gpu::encode_taps(gpu, gw, host, input.txt_ids, input.txt_mask, &KLEIN_TAPS)?
    } else {
        // The host encoder needs the FULL tables. Latch them into the bundle
        // rather than decoding per prompt: reaching here means the GPU
        // encoder is out for the session, so every later prompt lands here
        // too — see `FluxPipeBundle::latch_qwen3_host`.
        let taps = {
            let host = b.latch_qwen3_host()?;
            qwen3::encode_taps(host, input.txt_ids, input.txt_mask, &KLEIN_TAPS)
        };
        if taps.len() != rows * txt_dim {
            return Err(format!(
                "cond: the host Qwen3 taps are {} floats, expected {rows}x{txt_dim} — \
                 {} taps x the text encoder's hidden must equal the transformer's \
                 joint_attention_dim",
                taps.len(),
                KLEIN_TAPS.len()
            ));
        }
        gpu.upload_f32(&taps, &[rows, txt_dim])
            .map_err(|e| format!("cond: upload host qwen3 taps: {e:?}"))?
    };
    if dev.shape != vec![rows, txt_dim] {
        // Freed before bailing: `GpuTensor` has no `Drop`, and this tensor is
        // owned by nothing yet.
        let _ = gpu.free_tensor(dev);
        return Err(format!(
            "cond: qwen3 tap shape mismatch — {} taps x the text encoder's hidden must equal \
             the transformer's joint_attention_dim ({txt_dim})",
            KLEIN_TAPS.len()
        ));
    }
    Ok(dev)
}

/// Whether a request's conditioning may enter the cache: the cache must be on
/// AND the request must carry a real prompt key.
///
/// A standalone function so the rule is unit-testable without a device — see
/// `absent_prompt_key_bypasses_the_cache`.
fn is_cacheable(cache_enabled: bool, prompt_key: Option<&str>) -> bool {
    cache_enabled && prompt_key.is_some()
}

/// Where this generation's conditioning lives.
///
/// `Cached` is an index into `b.cond_cache`, which owns the device tensor and
/// frees it on eviction. `Owned` is a one-shot the caller must free after the
/// denoise loop — the path taken when the cache is off or the request carries
/// no prompt key, so an uncacheable request cannot leak and cannot pollute the
/// cache with a key that is not really a key.
enum CondSlot {
    Cached(usize),
    Owned(CondEntry),
}

/// GPU-resident twin of [`generate_txt2img_steps`]. Identical conditioning,
/// latent geometry, timestep schedule and Euler step; the transformer forward
/// is lifted to [`flux_gpu::gpu_forward_txt_dev`], the T5/CLIP conditioning to
/// [`t5_gpu`]/[`clip_gpu`], and the final VAE decode to
/// [`vae_gpu::gpu_decode`] (parity vs ComfyUI rel_l2 0.0031 on the golden
/// latent). Only the PNG postprocess stays on the host.
/// Requires the bundle's weights uploaded via [`FluxPipeBundle::ensure_gpu`].
/// Numeric parity with the CPU loop is the `gpu_pipeline_parity` gate's job.
///
/// Takes `&mut FluxPipeBundle` because the conditioning cache lives in the
/// bundle: it owns device tensors and must be able to evict (and free) them.
///
/// `input.prompt_key`, when set, is the cache key's prompt half. A `None` key
/// (a pre-tokenized parity harness with no prompt string) BYPASSES the cache:
/// it encodes, runs, and frees. It must not key on `""` — see [`CondSlot`].
pub fn generate_txt2img_steps_gpu(
    b: &mut FluxPipeBundle,
    gpu: &mut Gpu,
    input: &Txt2ImgInput,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    // ── conditioning: cache lookup, else encode ──────────────────────
    // Done FIRST and as a separate `&mut b` phase, so the body can hold plain
    // `&b` borrows of the conditioning and the transformer weights at once.
    let t_cond = std::time::Instant::now();
    let slot = ensure_conditioning(b, gpu, input)?;
    if img_profile_enabled() {
        let how = match &slot {
            CondSlot::Cached(_) => "cache",
            CondSlot::Owned(_) => "encode",
        };
        eprintln!(
            "[img-profile] {:<14} {:8.3} s ({how})",
            "conditioning",
            t_cond.elapsed().as_secs_f64()
        );
    }
    let out = {
        let cond: &CondEntry = match &slot {
            CondSlot::Cached(i) => &b.cond_cache.entries[*i],
            CondSlot::Owned(e) => e,
        };
        denoise_and_decode(b, gpu, input, cond, on_step)
    };
    // Runs on the error path too, which is the point of the split: an owned
    // (uncacheable) conditioning is not reachable from anywhere else, and
    // `GpuTensor` has no `Drop`.
    if let CondSlot::Owned(e) = slot {
        gpu.free_tensor(e.t5_hidden)
            .map_err(|err| format!("cond: free one-shot t5_hidden: {err:?}"))?;
    }
    out
}

/// The denoise loop + VAE decode, given conditioning that someone else owns.
///
/// Split out of [`generate_txt2img_steps_gpu`] purely so the caller can free a
/// one-shot conditioning on every exit path, including the error ones.
fn denoise_and_decode(
    b: &FluxPipeBundle,
    gpu: &mut Gpu,
    input: &Txt2ImgInput,
    cond: &CondEntry,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    let gw = b.gpu_weights.as_ref().ok_or_else(|| {
        "generate_txt2img_steps_gpu: GPU weights not uploaded (FluxPipeBundle::ensure_gpu)"
            .to_string()
    })?;
    let cfg = &b.transformer_cfg;
    // `HIPFIRE_IMG_PROFILE=1` prints wall time per stage to stderr, so the
    // per-generation fixed cost (conditioning, VAE, PNG) is attributable
    // without a harness change. Off by default: the daemon's stderr is the
    // wire. The conditioning stage is timed by the caller.
    let profile = img_profile_enabled();
    let mut stage = std::time::Instant::now();
    let lap = |name: &str, stage: &mut std::time::Instant| {
        if profile {
            eprintln!(
                "[img-profile] {name:<14} {:8.3} s",
                stage.elapsed().as_secs_f64()
            );
            *stage = std::time::Instant::now();
        }
    };

    let clip_pooled = cond.clip_pooled.clone();
    // The T5 hidden state stays on the device for the whole denoise loop:
    // `gpu_forward_txt_dev` borrows it instead of re-uploading 4 MB per step.
    let txt_dev = &cond.t5_hidden;
    // Shape is validated in `ensure_conditioning`, which is the only writer.
    let n_txt = cond.t5_hidden.shape[0];
    debug_assert_eq!(cond.t5_hidden.shape[1], cfg.txt_hidden_dim);

    let latent_h = input.height;
    let latent_w = input.width;
    let ch = cfg.latent_channels;
    let patch_size = cfg.patch_size;
    let n_img = (latent_h / 2) * (latent_w / 2);
    let init: Vec<f32> = match input.init_latents {
        Some(l) => l.to_vec(),
        None => vec![0.0f32; n_img * cfg.patch_size * cfg.patch_size * ch],
    };

    // FLUX.1: `Fixed`, i.e. `sigma_pairs(steps, shift)` exactly as before.
    // FLUX.2 Klein: the exponential empirical-mu shift, whose `image_seq_len`
    // is the GENERATED token count — reference tokens condition the forward
    // but must not move the schedule.
    let pairs = scheduler::sigma_pairs_ruled(input.steps, b.meta.shift_rule, n_img);
    let grid = (latent_h / 2, latent_w / 2);

    // Reference (edit) tokens: VAE-encoded, packed and normalized ONCE by the
    // caller, appended to the image stream after the generated tokens with
    // their own RoPE ids, and held FIXED for every step. Empty for plain
    // txt2img, where `img_ids` stays `None` and the FLUX.1 path runs
    // byte-identically to before.
    let refs = input.references;
    let n_ref: usize = refs.iter().map(|r| r.n).sum();
    let img_ids: Option<Vec<[f32; 4]>> = if refs.is_empty() {
        None
    } else {
        let mut ids = flux::rope_ids_for_grid(grid, 0.0);
        for r in refs {
            ids.extend_from_slice(&r.ids);
        }
        Some(ids)
    };
    // The joint attention route sees text + generated + reference tokens as
    // one sequence. Refuse an over-budget request by name here, where the
    // three counts are still legible, rather than inside a kernel launch.
    check_route_budget(n_txt, n_img, n_ref)?;
    let num_train_timesteps = b.meta.num_train_timesteps as f32;
    let final_order = b.final_order;

    // `HIPFIRE_PROFILE=1` additionally prints a per-kernel-family table for
    // each denoise step (finer than the `[img-profile]` stage lines above,
    // which only see the whole loop as one "denoise_loop" span). Cheap to
    // check once per generation-load. The four `Instant::now()` calls per
    // step below (`step_t0`, `t_host_pre`, `t_sampler`, `t_host_post`) are
    // NOT gated on it — they run unconditionally, at ~100 ns each, whatever
    // this flag is. What IS gated is turning them into a duration and
    // printing the table: each `.elapsed()` conversion is wrapped in
    // `family_profile.then(...)`, and `print_step_family_profile` itself is
    // called only `if family_profile`. So an unset env var skips the
    // conversions and the print, not the timestamps.
    let family_profile = hipfire_config::developer_var_os("HIPFIRE_PROFILE").is_some();

    let mut latents = init;
    let mut records = Vec::with_capacity(input.steps);
    for (step, (sigma, sigma_next)) in pairs.iter().enumerate() {
        let step_t0 = std::time::Instant::now();
        let t_host_pre = std::time::Instant::now();
        let t_sched = scheduler::timestep_for_sigma(*sigma, num_train_timesteps);
        let t_model = t_sched / num_train_timesteps;
        let mut img = latents.clone();
        for r in refs {
            img.extend_from_slice(&r.packed);
        }
        let host_pre_us = family_profile.then(|| t_host_pre.elapsed().as_secs_f64() * 1e6);
        let noise_pred = flux_gpu::gpu_forward_txt_dev(
            gpu,
            cfg,
            gw,
            &flux::FluxForwardInput {
                timestep: t_model,
                pooled: clip_pooled.clone(),
                // FLUX.1-dev is guidance-DISTILLED: its `guidance_in` embedder
                // is part of the model and the reference always feeds it (the
                // ComfyUI graph's FluxGuidance node, default 3.5). Passing
                // None silently skips that embedder and degrades the image
                // rather than failing. FLUX.1-schnell has no guidance embedder
                // (`guidance_embed_dim` 0), so key off the config.
                // FLUX.2 Klein has no guidance embedder at all — its
                // `time_guidance_embed` carries only the timestep half — so
                // the family test comes first and the env override cannot
                // reach it.
                guidance: if !cfg.is_flux2() && cfg.guidance_embed_dim > 0 {
                    Some(
                        hipfire_config::developer_var("HIPFIRE_FLUX_GUIDANCE")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(3.5f32),
                    )
                } else {
                    None
                },
                // The text stream is `txt_dev` (device-resident); the host
                // field is unread on this path.
                txt: Vec::new(),
                img,
                grid,
                mlp_act: input.mlp_act,
                final_order,
                img_ids: img_ids.clone(),
            },
            txt_dev,
            n_txt,
        )?;
        // Per-kernel-family GPU attribution for the forward just run, if
        // `HIPFIRE_PROFILE` was set — `None` otherwise. Must be taken
        // immediately after the call it belongs to: `take_step_profile`
        // returns the LAST resolved table, and the next `gpu_forward_txt_dev`
        // (next step) overwrites it.
        let family_table = flux_gpu::take_step_profile();
        debug_assert_eq!(
            noise_pred.len(),
            (n_img + n_ref) * cfg.patch_size * cfg.patch_size * cfg.latent_channels
        );
        // Only the GENERATED tokens are denoised; the head also predicted a
        // velocity for every reference token and those are dropped.
        let noise_pred = match n_ref {
            0 => noise_pred,
            _ => noise_pred[..n_img * cfg.patch_in()].to_vec(),
        };
        let t_sampler = std::time::Instant::now();
        let next: Vec<f32> = scheduler::euler_step(&latents, &noise_pred, *sigma, *sigma_next);
        let sampler_us = family_profile.then(|| t_sampler.elapsed().as_secs_f64() * 1e6);
        let t_host_post = std::time::Instant::now();
        records.push(StepRecord {
            t_model,
            latents_in: latents.clone(),
            noise_pred,
            latents_out: next.clone(),
        });
        latents = next;
        on_step(step + 1, input.steps);
        let host_post_us = family_profile.then(|| t_host_post.elapsed().as_secs_f64() * 1e6);
        if let Some(mut table) = family_table {
            // `host.sampler` = the Euler step; `host.other` = everything else
            // this loop iteration does on the CPU (schedule math, the latent
            // clone, `StepRecord` bookkeeping, `on_step`). Neither one syncs
            // the GPU — `family_table`'s GPU entries were already resolved
            // (one sync) inside `gpu_forward_txt_dev`'s `Gpuf::finish()`.
            *table.entry("host.sampler").or_insert(0.0) += sampler_us.unwrap_or(0.0);
            *table.entry("host.other").or_insert(0.0) +=
                host_pre_us.unwrap_or(0.0) + host_post_us.unwrap_or(0.0);
            print_step_family_profile(step, &table, step_t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    lap("denoise_loop", &mut stage);

    // Stop at the latent when the VAE was loaded config-only. Callers that do
    // this decode elsewhere (see `VaeDecoderWeights::config_only`); the final
    // latent is `steps.last().latents_out`.
    if hipfire_config::developer_var("HIPFIRE_VAE_CONFIG_ONLY").is_ok_and(|v| v != "0") {
        return Ok(Txt2ImgOutput {
            steps: records,
            image: vec![],
            image_shape: (0, 0),
            png: vec![],
        });
    }

    // Unpack → scale → VAE decode → PNG (identical to CPU). The decode runs
    // on the GPU by default (`vae_gpu` — parity vs ComfyUI rel_l2 0.0031 on
    // the golden latent); the single-threaded CPU `vae::decode` of a 1024²
    // image takes minutes, so `HIPFIRE_VAE_GPU=0` is the escape hatch back to
    // the reference, not the normal route.
    let packed_in = patch_size * patch_size * ch;
    let ch_unpacked = packed_in / 4;
    let denormed = scheduler::denormalize_packed(&latents, n_img, packed_in, &b.meta.latent_norm);
    let scaled = scheduler::unpack_latents(&denormed, n_img, ch_unpacked, latent_h, latent_w);
    lap("latent_unpack", &mut stage);
    let image = if hipfire_config::developer_var("HIPFIRE_VAE_GPU").map_or(true, |v| v != "0") {
        let img = match b.gpu_vae.as_ref() {
            // The normal route: weights already resident from `ensure_gpu`.
            Some(gv) => vae_gpu::gpu_decode(gpu, gv, &scaled, latent_h, latent_w)?.0,
            // Callers that reached here without `ensure_gpu` having uploaded
            // the VAE still decode correctly, just at the old per-generation
            // upload cost.
            None => {
                let gv = vae_gpu::GpuVaeDecoderWeights::from_host(gpu, &b.vae)?;
                lap("vae_upload", &mut stage);
                let out = vae_gpu::gpu_decode(gpu, &gv, &scaled, latent_h, latent_w);
                gv.free_gpu(gpu);
                out?.0
            }
        };
        lap("vae_decode", &mut stage);
        img
    } else {
        vae::decode(&b.vae, &scaled, latent_h, latent_w)
    };
    let up = vae_upscale(b);
    let png = postprocess_png(&image, input.width * up, input.height * up);
    lap("png_encode", &mut stage);
    Ok(Txt2ImgOutput {
        steps: records,
        image,
        image_shape: (input.width * up, input.height * up),
        png,
    })
}

/// Spatial upscale the VAE decoder applies to its latent input:
/// `2^(block_out_channels.len()−1)` (each up block but the last upsamples 2×).
/// The tiny fixture has one block → ×1 (latent dims == pixel dims); a real
/// FLUX VAE has four → ×8.
pub fn vae_upscale(b: &FluxPipeBundle) -> usize {
    1usize << b.vae.config.block_out_channels.len().saturating_sub(1)
}

/// [`generate_txt2img`] with a per-step progress callback
/// `on_step(step_index, total_steps)` fired after each denoise step
/// completes (the daemon's `img_progress` events).
pub fn generate_txt2img_steps(
    b: &FluxPipeBundle,
    input: &Txt2ImgInput,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    let cfg = &b.transformer_cfg;

    // Conditioning. The CPU path is the f32 oracle, so it needs the FULL host
    // tables — in streaming mode they are decoded here, used for this call,
    // and dropped on return.
    //
    // FLUX.1: T5 hidden → txt, CLIP pooled → vec. The transformer txt stream
    // is the T5 hidden states, BFL/diffusers both project via txt_in;
    // diffusers pads the T5 sequence to max_seq and the golden captured that
    // padding, so txt rows = t5_hidden rows.
    //
    // FLUX.2 Klein: the Qwen3 residual stream after the three KLEIN_TAPS
    // layers, concatenated per token — `[len, 3 * hidden]`. There is no
    // pooled vector at all (`pooled_projection_dim` is 0 and the FLUX.2
    // forward never reads it).
    let tx_host = b.transformer_host()?;
    let (txt, clip_pooled) = match &b.cond {
        TextCond::T5Clip { clip, .. } => {
            let t5_host = b.t5_host()?;
            let (t5_hidden, _t5_layers) = t5::encode(&t5_host, input.txt_ids, input.txt_mask);
            let (_, pooled, _clip_layers) = clip::encode(clip, input.clip_ids, input.clip_mask);
            (t5_hidden, pooled)
        }
        TextCond::Qwen3 { .. } => {
            let host = b.qwen3_host()?;
            let taps = qwen3::encode_taps(&host, input.txt_ids, input.txt_mask, &KLEIN_TAPS);
            (taps, Vec::new())
        }
    };
    let n_txt = input.txt_ids.len();
    if txt.len() != n_txt * cfg.txt_hidden_dim {
        return Err(format!(
            "cond: the text encoder produced {} floats for {n_txt} tokens, expected \
             {n_txt}x{} — the encoder width must equal the transformer's \
             joint_attention_dim",
            txt.len(),
            cfg.txt_hidden_dim
        ));
    }

    // Latent geometry: VAE compression is 2^(blocks-1); FLUX additionally
    // packs 2×2 patch cells, so the denoise grid is H/2 × W/2 (diffusers
    // prepare_latents/_unpack_latents).
    let latent_h = input.height;
    let latent_w = input.width;
    let patch_in = cfg.patch_in();
    let n_img = (latent_h / 2) * (latent_w / 2);
    let init: Vec<f32> = match input.init_latents {
        Some(l) => l.to_vec(),
        None => vec![0.0f32; n_img * patch_in],
    };

    // Timesteps. FLUX.1: diffusers `linspace(1, 1/steps, steps)` + the fixed
    // shift transform. FLUX.2 Klein: the same linspace under the exponential
    // empirical-mu shift, whose `image_seq_len` is the GENERATED token count
    // — reference tokens condition the forward but do not move the schedule.
    let pairs = scheduler::sigma_pairs_ruled(input.steps, b.meta.shift_rule, n_img);
    let grid = (latent_h / 2, latent_w / 2);

    // Reference (edit) tokens: appended to the image stream after the
    // generated tokens, with their own RoPE ids, and held fixed for every
    // step. Empty for plain txt2img, where the ids stay implicit and the
    // FLUX.1 path runs byte-identically to before.
    let refs = input.references;
    let n_ref: usize = refs.iter().map(|r| r.n).sum();
    let img_ids: Option<Vec<[f32; 4]>> = if refs.is_empty() {
        None
    } else {
        let mut ids = flux::rope_ids_for_grid(grid, 0.0);
        for r in refs {
            ids.extend_from_slice(&r.ids);
        }
        Some(ids)
    };
    // Same rule, same place in the sequence as the GPU path
    // (`denoise_and_decode`): the joint attention route sees text +
    // generated + reference tokens as one sequence, and an over-budget
    // request is refused by NAME here, where the three counts are still
    // legible. The CPU body used to skip this entirely — the check is not
    // about the device, it is about the request.
    check_route_budget(n_txt, n_img, n_ref)?;

    // Denoise loop — the transformer sees packed tokens like the BFL
    // reference (patch-in 4 ch/token at patch_size 1).
    let mut latents = init;
    let mut records = Vec::with_capacity(input.steps);
    for (step, (sigma, sigma_next)) in pairs.iter().enumerate() {
        let t_sched = scheduler::timestep_for_sigma(*sigma, b.meta.num_train_timesteps as f32);
        let t_model = t_sched / b.meta.num_train_timesteps as f32;
        let mut img = latents.clone();
        for r in refs {
            img.extend_from_slice(&r.packed);
        }
        let noise_pred = flux::forward(
            cfg,
            &tx_host,
            &flux::FluxForwardInput {
                timestep: t_model,
                pooled: clip_pooled.clone(),
                // FLUX.1-dev is guidance-DISTILLED: its `guidance_in` embedder
                // is part of the model and the reference always feeds it (the
                // ComfyUI graph's FluxGuidance node, default 3.5). Passing
                // None silently skips that embedder and degrades the image
                // rather than failing. FLUX.1-schnell has no guidance embedder
                // (`guidance_embed_dim` 0), so key off the config.
                // FLUX.2 Klein has no guidance embedder at all — its
                // `time_guidance_embed` carries only the timestep half — so
                // the family test comes first and the env override cannot
                // reach it. Identical to `denoise_and_decode`'s gate: a
                // config-driven difference between the CPU oracle and the GPU
                // path is a difference in the thing being compared.
                guidance: if !cfg.is_flux2() && cfg.guidance_embed_dim > 0 {
                    Some(
                        hipfire_config::developer_var("HIPFIRE_FLUX_GUIDANCE")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(3.5f32),
                    )
                } else {
                    None
                },
                txt: txt.clone(),
                img,
                grid,
                mlp_act: input.mlp_act,
                final_order: b.final_order, // diffusers golden semantics
                img_ids: img_ids.clone(),
            },
        );
        debug_assert_eq!(noise_pred.len(), (n_img + n_ref) * patch_in);
        // Only the GENERATED tokens are denoised; the reference tokens the
        // head also predicted for are dropped.
        let noise_pred = match n_ref {
            0 => noise_pred,
            _ => noise_pred[..n_img * patch_in].to_vec(),
        };
        let next: Vec<f32> = scheduler::euler_step(&latents, &noise_pred, *sigma, *sigma_next);
        records.push(StepRecord {
            t_model,
            latents_in: latents.clone(),
            noise_pred,
            latents_out: next.clone(),
        });
        latents = next;
        on_step(step + 1, input.steps);
    }

    // Unpack → denormalize → VAE decode. The unpacked latent channel count is
    // the packed-in divided by the 2×2 patch cell (diffusers
    // `in_channels // 4`), which equals the VAE's latent_channels; the VAE
    // config is the authority for the decode input.
    let packed_in = patch_in;
    let ch_unpacked = packed_in / 4;
    let denormed = scheduler::denormalize_packed(&latents, n_img, packed_in, &b.meta.latent_norm);
    let scaled = scheduler::unpack_latents(&denormed, n_img, ch_unpacked, latent_h, latent_w);
    let image = vae::decode(&b.vae, &scaled, latent_h, latent_w);
    // The decoder upsamples by 2^(blocks−1); PNG dims are PIXEL dims.
    let up = vae_upscale(b);
    let png = postprocess_png(&image, input.width * up, input.height * up);
    Ok(Txt2ImgOutput {
        steps: records,
        image,
        image_shape: (input.width * up, input.height * up),
        png,
    })
}

/// Prompt-level txt2img: tokenize, seed noise, validate geometry, run the
/// denoise loop with a progress callback. This is the daemon/CLI entry —
/// the parity harness uses [`generate_txt2img`] directly because the golden
/// pins its own conditioning ids and init latents.
///
/// `width`/`height` are PIXEL dims; latent dims are `dim / vae_upscale`
/// (must divide evenly, and each latent dim must be even for the 2×2 patch
/// packing). `seed` drives [`scheduler::seeded_gaussian`] — same seed →
/// byte-identical PNG.
pub fn generate_txt2img_prompt(
    b: &FluxPipeBundle,
    prompt: &str,
    width: usize,
    height: usize,
    steps: usize,
    seed: u64,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    generate_img_prompt(
        b,
        prompt,
        Some(width),
        Some(height),
        steps,
        seed,
        &[],
        on_step,
    )
}

/// Validate one image request and resolve it to the LATENT grid
/// `(latent_w, latent_h)` the denoise loop runs on.
///
/// The single owner of the request-level rules, called by BOTH
/// [`generate_img_prompt`] and [`generate_img_prompt_gpu`]. It used to be
/// ~40 lines duplicated between them, which is exactly the shape that lets
/// the CPU and GPU entry points drift apart in what they refuse and in what
/// they say when they refuse it — a divergence no test notices, because each
/// path's tests assert against its own copy.
///
/// The checks and their ORDER are load-bearing and unchanged: family before
/// count before size, so a FLUX.1 pipe handed five references is told it is
/// the wrong family rather than that it passed too many.
fn resolve_request_geometry(
    b: &FluxPipeBundle,
    width: Option<usize>,
    height: Option<usize>,
    steps: usize,
    references: &[RefImage],
) -> Result<(usize, usize), String> {
    if !references.is_empty() && !b.transformer_cfg.is_flux2() {
        return Err("reference images need a FLUX.2 pipe".into());
    }
    if references.len() > 4 {
        return Err(format!(
            "at most 4 reference images (got {})",
            references.len()
        ));
    }
    let (width, height) = match (width, height, references.first()) {
        (Some(w), Some(h), _) => (w, h),
        // The reference has already been snapped to a multiple of 16
        // (`refimg::target_size`), so its size is a legal output size.
        (None, None, Some(r)) => (r.width, r.height),
        (None, None, None) => (1024, 1024),
        _ => return Err("width and height must be given together".into()),
    };
    if steps == 0 || steps > 128 {
        return Err(format!("steps must be in 1..=128, got {steps}"));
    }
    let up = vae_upscale(b);
    if width == 0 || height == 0 || width > 8192 || height > 8192 {
        return Err(format!(
            "width/height must be in 1..=8192, got {width}x{height}"
        ));
    }
    if width % up != 0 || height % up != 0 {
        return Err(format!(
            "width/height must be divisible by the VAE compression factor {up}, got {width}x{height}"
        ));
    }
    let latent_w = width / up;
    let latent_h = height / up;
    if latent_w % 2 != 0 || latent_h % 2 != 0 {
        return Err(format!(
            "latent dims (size/{up} = {latent_w}x{latent_h}) must be even for 2×2 patch packing"
        ));
    }
    Ok((latent_w, latent_h))
}

/// Prompt-level image generation with optional REFERENCE images (the FLUX.2
/// Klein edit path) — the general form [`generate_txt2img_prompt`] wraps.
///
/// `width`/`height` are PIXEL dims and may be omitted only together: with a
/// reference the output takes the first reference's (already snapped) size,
/// and with neither it is 1024x1024. References are FLUX.2-only, capped at 4
/// (ComfyUI's templates use at most that), and each is VAE-encoded, packed
/// and normalized ONCE here — the denoise loop then conditions on them
/// unchanged, at RoPE time id `10 * (i + 1)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_img_prompt(
    b: &FluxPipeBundle,
    prompt: &str,
    width: Option<usize>,
    height: Option<usize>,
    steps: usize,
    seed: u64,
    references: &[RefImage],
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    let (latent_w, latent_h) = resolve_request_geometry(b, width, height, steps, references)?;
    let cond = condition_prompt(b, prompt, b.meta.max_seq)?;
    let cfg = &b.transformer_cfg;
    let ch_packed = cfg.patch_in();
    let n_img = (latent_h / 2) * (latent_w / 2);
    let noise = scheduler::seeded_gaussian(n_img * ch_packed, seed);
    let refs = build_ref_tokens(b, None, references)?;
    let input = Txt2ImgInput {
        txt_ids: &cond.txt_ids,
        txt_mask: &cond.txt_mask,
        clip_ids: &cond.clip_ids,
        clip_mask: &cond.clip_mask,
        references: &refs,
        init_latents: Some(&noise),
        height: latent_h,
        width: latent_w,
        steps,
        mlp_act: FluxPipeBundle::mlp_act_default(),
        prompt_key: None,
    };
    generate_txt2img_steps(b, &input, on_step)
}

/// GPU-resident twin of [`generate_txt2img_prompt`] (the daemon/CLI entry).
/// The backend must match the weights: call [`FluxPipeBundle::ensure_gpu`]
/// first, then pass `&mut Gpu` for the transformer forward.
///
/// A thin wrapper over [`generate_img_prompt_gpu`] with no references, so
/// plain txt2img and the edit path cannot drift: there is one GPU generation
/// body, and `&[]` is what makes it the FLUX.1 one.
pub fn generate_txt2img_prompt_gpu(
    b: &mut FluxPipeBundle,
    gpu: &mut Gpu,
    prompt: &str,
    width: usize,
    height: usize,
    steps: usize,
    seed: u64,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    generate_img_prompt_gpu(
        b,
        gpu,
        prompt,
        Some(width),
        Some(height),
        steps,
        seed,
        &[],
        on_step,
    )
}

/// GPU-resident twin of [`generate_img_prompt`]: prompt-level image
/// generation with optional REFERENCE images (the FLUX.2 Klein edit path).
///
/// Same validation and size rule as the CPU twin — `width`/`height` are PIXEL
/// dims and may be omitted only together (with a reference the output takes
/// the first reference's already-snapped size, with neither it is
/// 1024×1024), at most 4 references, and references need a FLUX.2 pipe. Each
/// reference is VAE-encoded ONCE here (on the device when
/// [`FluxPipeBundle::ensure_gpu`] uploaded the encoder), packed and
/// normalized; the denoise loop then conditions on them unchanged at RoPE
/// time id `10 * (i + 1)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_img_prompt_gpu(
    b: &mut FluxPipeBundle,
    gpu: &mut Gpu,
    prompt: &str,
    width: Option<usize>,
    height: Option<usize>,
    steps: usize,
    seed: u64,
    references: &[RefImage],
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Txt2ImgOutput, String> {
    let (latent_w, latent_h) = resolve_request_geometry(b, width, height, steps, references)?;
    let cond = condition_prompt(b, prompt, b.meta.max_seq)?;
    let cfg = &b.transformer_cfg;
    let ch_packed = cfg.patch_in();
    let n_img = (latent_h / 2) * (latent_w / 2);
    let noise = scheduler::seeded_gaussian(n_img * ch_packed, seed);
    // Encoded before the loop and unchanged by it. `Some(gpu)` takes the
    // device encoder when `ensure_gpu` uploaded one; `build_ref_tokens` falls
    // back to the host `vae::encode` otherwise. `refs` is empty for txt2img,
    // and then this is a no-op that allocates nothing.
    let refs = build_ref_tokens(b, Some(gpu), references)?;
    let input = Txt2ImgInput {
        txt_ids: &cond.txt_ids,
        txt_mask: &cond.txt_mask,
        clip_ids: &cond.clip_ids,
        clip_mask: &cond.clip_mask,
        references: &refs,
        init_latents: Some(&noise),
        height: latent_h,
        width: latent_w,
        steps,
        mlp_act: FluxPipeBundle::mlp_act_default(),
        prompt_key: Some(prompt),
    };
    generate_txt2img_steps_gpu(b, gpu, &input, on_step)
}

/// Diffusers `VaeImageProcessor.postprocess` → PNG bytes: denormalize
/// `(x+1)/2` clamped [0,1], ×255 with banker's rounding (torch/numpy
/// `round`), RGB, PNG-encoded via the `image` crate.
pub fn postprocess_png(image: &[f32], width: usize, height: usize) -> Vec<u8> {
    // decode output is channel-major [3][h][w]; interleave per pixel RGB
    let mut px = Vec::with_capacity(width * height * 3);
    let ch = if image.len() == 3 * width * height {
        3
    } else {
        1
    };
    for y in 0..height {
        for x in 0..width {
            for c in 0..ch {
                let v = ((image[c * width * height + y * width + x] + 1.0) * 0.5).clamp(0.0, 1.0)
                    * 255.0;
                px.push(v.round_ties_even() as u8);
            }
        }
    }
    let rgb: image::RgbImage = image::RgbImage::from_raw(width as u32, height as u32, px)
        .expect("postprocess: image dimensions inconsistent");
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(rgb)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("postprocess: PNG encode failed");
    out
}
#[cfg(test)]
mod comfy_interop_tests {
    use super::*;

    /// The two checkpoint families swap the adaLN halves. Pinning the mapping
    /// matters because getting it backwards produces a plausible-looking but
    /// permanently grainy image rather than an error, and no block-scoped
    /// gate reaches the final head where it lands.
    #[test]
    fn final_order_follows_checkpoint_family() {
        // BFL `LastLayer`: `shift, scale = adaLN(vec).chunk(2, dim=1)`.
        assert!(matches!(
            final_order_for_checkpoint(true),
            FinalAdaLNOrder::ShiftScale
        ));
        // diffusers `AdaLayerNormContinuous`: `scale, shift = chunk(...)`.
        assert!(matches!(
            final_order_for_checkpoint(false),
            FinalAdaLNOrder::ScaleShift
        ));
    }

    /// A ComfyUI `.latent` without `latent_format_version_0` is silently
    /// rescaled by 1/0.18215 on load. The marker must always be emitted.
    #[test]
    fn comfy_latent_carries_the_version_marker() {
        let data = vec![0.25f32; 2 * 3 * 4];
        let bytes = comfy_latent_bytes(&data, 2, 3, 4);
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + hdr_len]).expect("header is utf8");
        assert!(
            header.contains("latent_format_version_0"),
            "missing the marker; ComfyUI would rescale by 1/0.18215: {header}"
        );
        assert!(
            header.contains(r#""shape":[1,2,3,4]"#),
            "wrong shape: {header}"
        );
        // safetensors requires an 8-byte-aligned header.
        assert_eq!(hdr_len % 8, 0, "header not padded to 8 bytes");
        assert_eq!(
            bytes.len(),
            8 + hdr_len + data.len() * 4,
            "payload truncated"
        );
    }

    /// `latent_rel_error` must report zero for identical input, and must
    /// separate a single outlier (which moves `rel_inf` only) from a spread
    /// difference (which moves both). A metric that conflated those would let
    /// the golden gate call "agrees except at one point" a bulk divergence.
    #[test]
    fn latent_rel_error_separates_outlier_from_spread() {
        let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let (i0, l0) = latent_rel_error(&b, &b);
        assert_eq!((i0, l0), (0.0, 0.0), "identical latents must compare equal");

        let mut one = b.clone();
        one[7] += 1.0;
        let (i1, l1) = latent_rel_error(&one, &b);

        let spread: Vec<f32> = b.iter().map(|v| v + 1.0 / 8.0).collect();
        let (i2, l2) = latent_rel_error(&spread, &b);

        // Both carry the SAME total squared error: 1.0 at one point, versus
        // 64 x (1/8)^2 = 1.0 spread over every point. That is the whole point
        // of the case. `rel_l2` cannot tell them apart — it is a bulk metric
        // and by construction it must not — while `rel_inf` separates them by
        // 8x. Neither number is the "right" one; the pair is the answer, which
        // is why both are returned and both are printed.
        assert!(
            (l1 - l2).abs() <= 1e-6,
            "equal total squared error must give equal rel_l2: {l1} vs {l2}"
        );
        assert!(
            i1 > i2 * 4.0,
            "one outlier must dominate rel_inf: {i1} vs {i2}"
        );
    }

    /// The `.latent` writer and reader must round-trip, so the golden gate
    /// reads back exactly what a run wrote.
    #[test]
    fn comfy_latent_round_trips() {
        let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes = comfy_latent_bytes(&data, 2, 3, 4);
        let (back, shape) = read_comfy_latent(&bytes).expect("parse");
        assert_eq!(shape, [1, 2, 3, 4]);
        assert_eq!(back, data, "latent did not survive a write/read round trip");
    }

    /// `pack_latents` / `unpack_latents` must round-trip, and must use the
    /// diffusers ordering `c*4 + dh*2 + dw` within a 2x2 patch. A
    /// self-consistent but wrong ordering would still round-trip, so the
    /// index assertion below is the part that pins the convention.
    #[test]
    fn latent_pack_unpack_round_trips_in_diffusers_order() {
        let (c, h, w) = (2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let (packed, tokens) = scheduler::pack_latents(&x, c, h, w);
        assert_eq!(tokens, (h / 2) * (w / 2));
        let back = scheduler::unpack_latents(&packed, tokens, c, h, w);
        assert_eq!(back, x, "pack/unpack is not an identity");

        // Patch (ph=0, pw=0), channel 1, offset (dh=1, dw=0) must sit at
        // token 0, lane c*4 + dh*2 + dw = 1*4 + 2 = 6.
        let expect = x[1 * h * w + 1 * w];
        assert_eq!(packed[6], expect, "2x2 patch lane order is not c*4+dh*2+dw");
    }
}

#[cfg(test)]
mod cond_cache_tests {
    use super::*;

    /// A metadata-only entry. `null_for_test` buffers must never reach a HIP
    /// call, and `push`/`lookup` never touch the device — only `insert`/`clear`
    /// do, which is exactly why `push` exists.
    fn entry(prompt: &str, t5_seq: usize) -> CondEntry {
        family_entry(prompt, FluxFamily::Flux1, t5_seq)
    }

    fn family_entry(prompt: &str, family: FluxFamily, t5_seq: usize) -> CondEntry {
        CondEntry {
            prompt: prompt.to_string(),
            family,
            t5_seq,
            t5_hidden: GpuTensor::null_for_test(),
            clip_pooled: vec![t5_seq as f32],
        }
    }

    /// A FLUX.1 lookup — the family every entry in these cases carries unless
    /// it says otherwise.
    fn lookup(c: &mut CondCache, prompt: &str, t5_seq: usize) -> Option<usize> {
        c.lookup(prompt, FluxFamily::Flux1, t5_seq)
    }

    fn cache(capacity: usize) -> CondCache {
        CondCache {
            entries: Vec::new(),
            capacity,
            enabled: true,
        }
    }

    #[test]
    fn t5_seq_is_part_of_the_key() {
        // The same prompt padded to a different length is a DIFFERENT
        // `[t5_seq, d_model]` tensor and a different relative-bias table.
        // Keying on the prompt alone would hand the denoise loop a tensor of
        // the wrong shape — which does not fail loudly, it feeds `txt_in` the
        // wrong number of rows.
        let mut c = cache(4);
        assert!(c.push(entry("a lighthouse", 256)).is_none());
        assert_eq!(lookup(&mut c, "a lighthouse", 256), Some(0));
        assert_eq!(
            lookup(&mut c, "a lighthouse", 512),
            None,
            "same prompt at a different t5_seq must miss"
        );
        assert_eq!(lookup(&mut c, "a different prompt", 256), None);
    }

    #[test]
    fn lookup_promotes_to_mru_and_eviction_takes_the_lru() {
        let mut c = cache(2);
        c.push(entry("first", 8));
        c.push(entry("second", 8));
        // Touch the older entry: it must become MRU, so the NEXT insert
        // evicts "second", not "first". A cache that promoted nothing would
        // evict the entry just proven to be in use.
        assert_eq!(lookup(&mut c, "first", 8), Some(0));
        let evicted = c.push(entry("third", 8)).expect("capacity 2 must evict");
        assert_eq!(evicted.prompt, "second");
        assert_eq!(lookup(&mut c, "first", 8), Some(0));
        assert_eq!(lookup(&mut c, "third", 8), Some(0));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn disabled_cache_never_reports_a_hit() {
        // `HIPFIRE_IMG_COND_CACHE=0` must reproduce uncached behaviour, not
        // just skip the eviction bookkeeping: an entry can still be resident
        // (the in-flight run needs somewhere to hold its device tensor) and
        // must still miss.
        let mut c = cache(4);
        c.enabled = false;
        c.push(entry("held", 256));
        assert_eq!(c.len(), 1, "entry is resident");
        assert_eq!(lookup(&mut c, "held", 256), None, "but must not be reused");
    }

    #[test]
    fn family_is_part_of_the_key() {
        // The same prompt at the same length is a completely different tensor
        // under T5-XXL than under the Klein Qwen3 text encoder — different width,
        // different values. Keying without the family would hand a FLUX.2
        // forward a FLUX.1 conditioning of the wrong width.
        let mut c = cache(4);
        c.push(family_entry("a lighthouse", FluxFamily::Flux1, 256));
        assert_eq!(lookup(&mut c, "a lighthouse", 256), Some(0));
        assert_eq!(
            c.lookup("a lighthouse", FluxFamily::Flux2, 256),
            None,
            "the same prompt under another family must miss"
        );
        c.push(family_entry("a lighthouse", FluxFamily::Flux2, 256));
        assert_eq!(c.lookup("a lighthouse", FluxFamily::Flux2, 256), Some(0));
        assert_eq!(c.len(), 2, "both families coexist");
    }

    #[test]
    fn capacity_is_the_documented_eight() {
        assert_eq!(CondCache::CAPACITY, 8);
    }

    /// `prompt_key: None` must BYPASS the cache, not key on `""`.
    ///
    /// Every pre-tokenized caller (`gpu_flux_golden_latent`,
    /// `flux_pipeline_parity`) passes `None`. If
    /// `None` collapsed to `""` that would be a live key, and two different
    /// pre-tokenized inputs at the same `t5_seq` in one process would collide:
    /// the second run would silently reuse the first run's conditioning and
    /// its parity numbers would be meaningless.
    ///
    /// Exercises the shipped predicate `ensure_conditioning` calls, not a
    /// restatement of it.
    #[test]
    fn absent_prompt_key_bypasses_the_cache() {
        assert!(!is_cacheable(true, None), "no prompt key must not cache");
        assert!(
            !is_cacheable(false, Some("a prompt")),
            "disabled must not cache"
        );
        assert!(!is_cacheable(false, None));
        assert!(is_cacheable(true, Some("a prompt")));

        // And the empty string is NOT special-cased into a bypass: a caller
        // that genuinely conditions on an empty prompt still gets a cache
        // entry, which is correct — an empty prompt is a real prompt.
        assert!(is_cacheable(true, Some("")));

        // A cache that DID receive two `""`-keyed entries at the same t5_seq
        // would hand the second caller the first's tensor. Pinned here so the
        // bypass is not "fixed" by making `""` a sentinel inside the cache.
        let mut c = cache(4);
        c.push(entry("", 256));
        assert_eq!(
            lookup(&mut c, "", 256),
            Some(0),
            "`\"\"` is an ordinary key inside the cache — the bypass must live \
             in ensure_conditioning, not here"
        );
    }
}

/// End-to-end cover for the FLUX.2 (Klein) pipe: a whole tiny pipe written to
/// disk — transformer, Qwen3 text encoder, tokenizer, VAE (encoder + decoder +
/// BatchNorm statistics) and scheduler — loaded through [`load_pipe`] and run
/// through the CPU txt2img and reference-edit entry points.
///
/// The geometry is the real Klein wiring in miniature, and every cross-config
/// relation the loader asserts holds: the VAE's `latent_channels 2` × its
/// `patch_size [2, 2]` is the transformer's `in_channels 8`, and the Qwen3
/// `hidden 16` × the three [`crate::qwen3::KLEIN_TAPS`] is its
/// `joint_attention_dim 48`. Getting any of those wrong is exactly the class
/// of bug a single-component synthetic test cannot see.
#[cfg(test)]
mod klein_pipe_tests {
    use super::*;
    use crate::config::FluxFamily;
    use crate::flux::test_fixtures::{temp_dir, write_safetensors, NamedTensor};
    use crate::refimg::RefImage;
    use crate::scheduler::ShiftRule;
    use serde_json::json;
    use std::path::Path;

    /// The tiny Qwen3 text encoder: 27 layers because Klein taps the residual
    /// stream after layer 27 (a shorter text encoder is refused at load), everything
    /// else as small as the shapes allow.
    fn text_encoder_config() -> serde_json::Value {
        json!({
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 27,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 24,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 40,
            "tie_word_embeddings": true,
        })
    }

    /// `in_channels 8` = the VAE's `latent_channels 2` × its `2 x 2` latent
    /// patch; `joint_attention_dim 48` = Qwen3 `hidden 16` × 3 taps.
    fn transformer_config() -> serde_json::Value {
        json!({
            "_class_name": "Flux2Transformer2DModel",
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "axes_dims_rope": [4, 4, 4, 4],
            "num_layers": 2,
            "num_single_layers": 1,
            "patch_size": 1,
            "in_channels": 8,
            "joint_attention_dim": 48,
            "mlp_ratio": 3.0,
            "rope_theta": 2000,
        })
    }

    fn vae_config() -> serde_json::Value {
        json!({
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 2,
            "block_out_channels": [8, 8, 16, 16],
            "layers_per_block": 1,
            "norm_num_groups": 4,
            "mid_block_add_attention": true,
            "use_quant_conv": true,
            "use_post_quant_conv": true,
            "batch_norm_eps": 0.0001,
            "patch_size": [2, 2],
        })
    }

    /// The published Klein `scheduler_config.json` (spec section 2.4). The
    /// `shift` / `base_shift` / `max_shift` entries are deliberately present
    /// and deliberately unused: Klein computes `mu` from its own empirical
    /// formula, and a pipe that honoured `shift 3.0` here would denoise on
    /// the wrong schedule.
    fn scheduler_config() -> serde_json::Value {
        json!({
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "use_dynamic_shifting": true,
            "time_shift_type": "exponential",
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "shift": 3.0,
        })
    }

    /// A 40-entry Qwen3-shaped vocabulary: the five Klein template specials,
    /// newline, the SentencePiece space marker, `a`..`z`, and fillers up to
    /// `vocab_size`. Every character the template can produce is in it, so
    /// nothing is silently dropped and no id can land outside the embedding
    /// table.
    fn tokenizer_json() -> String {
        let mut vocab = serde_json::Map::new();
        vocab.insert("\n".into(), json!(0));
        vocab.insert("\u{2581}".into(), json!(1));
        for (i, c) in ('a'..='z').enumerate() {
            vocab.insert(c.to_string(), json!(2 + i));
        }
        let specials = [
            ("<|im_start|>", 28),
            ("<|im_end|>", 29),
            ("<think>", 30),
            ("</think>", 31),
            ("<|endoftext|>", 32),
        ];
        for (content, id) in specials {
            vocab.insert(content.into(), json!(id));
        }
        for i in 33..40 {
            vocab.insert(format!("<|extra_{i}|>"), json!(i));
        }
        let added: Vec<serde_json::Value> = specials
            .iter()
            .map(|(content, id)| json!({ "id": id, "content": content, "special": true }))
            .collect();
        json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": [] },
            "added_tokens": added,
        })
        .to_string()
    }

    fn write_component(dir: &Path, config: &serde_json::Value, tensors: &[NamedTensor]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        write_safetensors(&dir.join("model.safetensors"), tensors);
    }

    /// Write a complete, loadable FLUX.2 Klein pipe directory.
    fn write_tiny_klein_pipe(dir: &Path) {
        let tx_cfg_json = transformer_config();
        let tx_cfg = FluxDiffusionConfig::from_json(&tx_cfg_json).unwrap();
        let plan = flux::FluxPlan::flux2_diffusers(&tx_cfg);
        write_component(
            &dir.join("transformer"),
            &tx_cfg_json,
            &crate::flux::test_fixtures::plan_tensors(&tx_cfg, &plan),
        );

        let te_cfg_json = text_encoder_config();
        let te_plan = crate::qwen3::Qwen3Plan {
            config: crate::qwen3::Qwen3Config::from_json(&te_cfg_json).unwrap(),
        };
        write_component(
            &dir.join("text_encoder"),
            &te_cfg_json,
            &crate::qwen3::test_fixtures::checkpoint_tensors(&te_plan),
        );

        let vae_cfg_json = vae_config();
        let vae_cfg = crate::vae::VaeConfig::from_json(&vae_cfg_json).unwrap();
        write_component(
            &dir.join("vae"),
            &vae_cfg_json,
            &crate::vae::test_fixtures::checkpoint_tensors(&vae_cfg),
        );

        std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
        std::fs::write(dir.join("tokenizer/tokenizer.json"), tokenizer_json()).unwrap();
        std::fs::create_dir_all(dir.join("scheduler")).unwrap();
        std::fs::write(
            dir.join("scheduler/scheduler_config.json"),
            scheduler_config().to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            json!({ "_class_name": "Flux2KleinPipeline" }).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn tiny_klein_pipe_runs_txt2img_and_edit_on_the_cpu() {
        let dir = temp_dir("tiny-klein-pipe");
        write_tiny_klein_pipe(&dir);
        let b = load_pipe(&dir).unwrap();
        assert_eq!(b.transformer_cfg.family, FluxFamily::Flux2);
        assert!(matches!(b.cond, TextCond::Qwen3 { .. }));
        assert_eq!(b.meta.shift_rule, ShiftRule::Empirical);
        assert!(matches!(b.meta.latent_norm, LatentNorm::BatchNorm { .. }));
        assert!(b.vae_enc.is_some());

        let out = generate_img_prompt(&b, "a cat", Some(64), Some(48), 2, 7, &[], &mut |_, _| {})
            .unwrap();
        assert_eq!(out.image_shape, (64, 48));
        assert_eq!(out.steps.len(), 2);
        assert!(out.png.len() > 8);
        assert!(out.image.iter().all(|v| v.is_finite()));

        // Edit: one 32x32 reference and no width/height — the output takes the
        // reference's size.
        let r = RefImage {
            width: 32,
            height: 32,
            pixels: vec![0.0; 3 * 32 * 32],
        };
        let out2 = generate_img_prompt(
            &b,
            "a cat",
            None,
            None,
            2,
            7,
            std::slice::from_ref(&r),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(out2.image_shape, (32, 32));
        assert_eq!(out2.steps.len(), 2);

        // The reference tokens must actually reach the forward: the same
        // prompt, seed and geometry WITHOUT a reference must predict a
        // different velocity.
        let plain = generate_img_prompt(&b, "a cat", Some(32), Some(32), 2, 7, &[], &mut |_, _| {})
            .unwrap();
        assert_ne!(
            plain.steps[0].noise_pred, out2.steps[0].noise_pred,
            "reference tokens did not change the predicted velocity"
        );
    }

    /// The Klein prompt framing: templated, right-padded to `KLEIN_MIN_LEN`,
    /// opening on `<|im_start|>`, with no CLIP halves. Cheap (tokenizer only),
    /// so it is separate from the end-to-end run above.
    #[test]
    fn klein_conditioning_is_templated_and_right_padded() {
        let dir = temp_dir("tiny-klein-cond");
        write_tiny_klein_pipe(&dir);
        let b = load_pipe(&dir).unwrap();
        assert_eq!(b.meta.max_seq, crate::klein_prompt::KLEIN_MIN_LEN);
        let cond = condition_prompt(&b, "a cat", b.meta.max_seq).unwrap();
        assert_eq!(cond.txt_ids.len(), b.meta.max_seq);
        assert_eq!(cond.txt_mask.len(), b.meta.max_seq);
        assert_eq!(cond.txt_ids[0], 28, "must open on `<|im_start|>`");
        let real = cond.txt_mask.iter().filter(|m| **m == 1).count();
        assert!(
            real > 8 && real < b.meta.max_seq,
            "template tokens {real} should be a small real prefix of the pad frame"
        );
        assert!(cond.txt_ids[real..].iter().all(|&i| i == 32), "pads");
        assert!(
            cond.clip_ids.is_empty() && cond.clip_mask.is_empty(),
            "Klein has no CLIP conditioning"
        );
    }

    /// The route budget is a GPU-path guard, but the rule itself is pure, so
    /// it is pinned here without a device. The boundary matters: exactly
    /// `MAX_ROUTE_TOKENS` must pass (a real 1024² edit with four references
    /// is nowhere near it, and an off-by-one that refused the legal maximum
    /// would only ever be found by the request it refused), and the message
    /// must name all three counts so the caller knows which one to shrink.
    #[test]
    fn the_route_budget_refuses_only_past_the_cap() {
        assert!(check_route_budget(512, 1024, 4096).is_ok(), "a real edit");
        assert!(
            check_route_budget(MAX_ROUTE_TOKENS, 0, 0).is_ok(),
            "exactly the cap is legal"
        );
        let err = check_route_budget(MAX_ROUTE_TOKENS, 1, 0).unwrap_err();
        assert!(err.contains("32769 tokens exceed"), "{err}");
        assert!(err.contains("32768 text"), "{err}");
        assert!(err.contains("1 generated"), "{err}");
        assert!(err.contains("0 reference"), "{err}");
    }

    /// The decline message quotes a size, and a size that is wrong by an
    /// order of magnitude is worse than none: it sends an OOM investigation
    /// at the wrong allocation. Klein 4B's Qwen3 text encoder is 36 layers of
    /// hidden 2560 / intermediate 9728 with 32 q heads and 8 kv heads at
    /// head_dim 128 — ~6.8 GiB of f16 linears.
    #[test]
    fn the_qwen3_size_estimate_matches_klein_4b() {
        let cfg = crate::qwen3::Qwen3Config {
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9728,
            rope_theta: 1e6,
            eps: 1e-6,
            vocab: 151936,
            tie_embeddings: true,
        };
        let gib = qwen3_f16_gib(&cfg);
        assert!((6.5..7.0).contains(&gib), "{gib} GiB");
    }

    /// The request-shape guards on the edit entry point.
    #[test]
    fn edit_requests_fail_closed() {
        let dir = temp_dir("tiny-klein-guards");
        write_tiny_klein_pipe(&dir);
        let b = load_pipe(&dir).unwrap();
        let noop = &mut |_: usize, _: usize| {};
        // `Txt2ImgOutput` is not `Debug` (it carries whole images), so these
        // unwrap the error half by hand.
        let refuse = |r: Result<Txt2ImgOutput, String>| match r {
            Ok(_) => panic!("generate_img_prompt accepted a request it must refuse"),
            Err(e) => e,
        };

        let err = refuse(generate_img_prompt(
            &b,
            "a cat",
            Some(64),
            None,
            2,
            7,
            &[],
            noop,
        ));
        assert!(
            err.contains("width and height must be given together"),
            "{err}"
        );

        let five: Vec<RefImage> = (0..5)
            .map(|_| RefImage {
                width: 32,
                height: 32,
                pixels: vec![0.0; 3 * 32 * 32],
            })
            .collect();
        let err = refuse(generate_img_prompt(
            &b, "a cat", None, None, 2, 7, &five, noop,
        ));
        assert!(err.contains("at most 4 reference images"), "{err}");

        // A reference whose sides are not a multiple of 2 x the VAE
        // compression cannot be packed into 2x2 latent patches. The output
        // size is given here, so this is the REFERENCE geometry check rather
        // than the output one (which enforces the same rule on its own dims).
        let odd = [RefImage {
            width: 40,
            height: 32,
            pixels: vec![0.0; 3 * 40 * 32],
        }];
        let err = refuse(generate_img_prompt(
            &b,
            "a cat",
            Some(64),
            Some(48),
            2,
            7,
            &odd,
            noop,
        ));
        assert!(err.contains("not a multiple of 16"), "{err}");
    }

    /// The cross-config assertion: the transformer's packed token width and
    /// the VAE's latent geometry describe the SAME tensor, and a mismatch
    /// decodes to noise rather than failing — so it is caught at load, with
    /// both numbers named.
    #[test]
    fn load_rejects_a_transformer_vae_width_mismatch() {
        let dir = temp_dir("tiny-klein-mismatch");
        write_tiny_klein_pipe(&dir);
        // 12 = 3 x the 2x2 patch, so the plan and the manifest still agree
        // with each other — only the VAE disagrees with both.
        let mut cfg = transformer_config();
        cfg["in_channels"] = json!(12);
        std::fs::write(dir.join("transformer/config.json"), cfg.to_string()).unwrap();
        let err = match load_pipe(&dir) {
            Ok(_) => panic!("load_pipe accepted a pipe it must refuse"),
            Err(e) => e,
        };
        assert!(
            err.contains('8') && err.contains("12"),
            "the error must name both widths: {err}"
        );
    }

    /// A text encoder shorter than the deepest Klein tap would condition on
    /// zeros — `encode_taps` returns a zero block for a tap that never fires.
    #[test]
    fn load_rejects_a_text_encoder_shorter_than_the_taps() {
        let dir = temp_dir("tiny-klein-shorttext");
        write_tiny_klein_pipe(&dir);
        let mut cfg = text_encoder_config();
        cfg["num_hidden_layers"] = json!(4);
        std::fs::write(dir.join("text_encoder/config.json"), cfg.to_string()).unwrap();
        let err = match load_pipe(&dir) {
            Ok(_) => panic!("load_pipe accepted a pipe it must refuse"),
            Err(e) => e,
        };
        assert!(
            err.contains("4 layers") && err.contains("27"),
            "the error must name the text encoder depth and the tap: {err}"
        );
    }
}
