// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Config for the EmbeddingGemma-2 text encoder (`embedding_gemma2_text`).
//!
//! Parsed from the composite `config.json`'s `text_config` plus the
//! sentence-transformers sidecars in the model dir (`prompts` for the two
//! kinds; no length keys ship with this release, so both default to the
//! model's 8 192-token context window). `per_layer_config` in the
//! serialized config is a map of zero-padded layer-index strings
//! (`"05"`, `"11"`, …) to `{head_dim, num_key_value_heads}` overrides —
//! on this release the four full-attention layers widen to head_dim 512
//! with a single KV head while every other layer keeps the base 256/2.

/// Which sentence-transformers prompt/truncation profile an encode uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Query,
    Document,
}

/// Layer attention kinds, parsed from `text_config.layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// Per-layer attention geometry: resolved once per layer at parse (base
/// values + that layer's `per_layer_config` override).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerAttn {
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

/// Soft-token counts HF's `Gemma4ImageProcessor` accepts
/// (`_SUPPORTED_SOFT_TOKENS`); anything else is refused, not clamped.
pub const SUPPORTED_SOFT_TOKENS: &[usize] = &[70, 140, 280, 560, 1120];

/// The Gemma-4 vision tower's geometry — `config.json`'s `vision_config`
/// (`model_type` `gemma4_vision`), the config the checkpoint's
/// `vision_tower.*` tensors are shaped for.
///
/// Held by [`EmbedConfig::vision`] and consumed by [`crate::image`] (the
/// resize/patchify geometry) and [`crate::vision`] (the tower itself).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    /// Rows in each axis of `patch_embedder.position_embedding_table`.
    pub position_embedding_size: usize,
    /// Axial 2-D RoPE base (`rope_parameters.rope_theta`, 100 here).
    pub rope_theta: f32,
    /// `max_soft_tokens` — the per-image budget the resize targets.
    /// One of [`SUPPORTED_SOFT_TOKENS`].
    pub max_soft_tokens: usize,
}

impl VisionConfig {
    /// Pixel side of one patch grid cell when the resize rounds down:
    /// `pooling_kernel_size * patch_size` (48 on this release).
    pub fn side_mult(&self) -> usize {
        self.pooling_kernel_size * self.patch_size
    }

    /// Patch budget per image: `max_soft_tokens * pooling_kernel_size²`
    /// (2 520 on this release; the processor pads up to it).
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Elements in one patch: `3 * patch_size²`, the patch-embedder GEMM's
    /// K (h→w→channel layout).
    pub fn patch_pixels(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Elements in one row of `position_embedding_table` per axis: the
    /// table is `[2, position_embedding_size, hidden_size]`.
    pub fn position_axis_stride(&self) -> usize {
        self.position_embedding_size * self.hidden_size
    }
}

/// The Gemma-4 audio tower's geometry — `config.json`'s `audio_config`
/// (`model_type` `gemma4_audio`), the config the checkpoint's
/// `audio_tower.*` tensors are shaped for.
///
/// Held by [`EmbedConfig::audio`] and consumed by [`crate::audio_tower`]
/// (the conformer tower). The FRONTEND geometry is a different config
/// ([`AudioFrontendConfig`], from `processor_config.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    /// `output_proj_dims` — the tower's output width, the audio embedder's
    /// input width (1536 here; the text width is 512 and not related).
    pub output_proj_dims: usize,
    /// Depthwise light-conv kernel (`conv_kernel_size`, 5, causal).
    pub conv_kernel_size: usize,
    /// Chunked attention: query block size.
    pub attention_chunk_size: usize,
    /// `attention_context_left` — inclusive window past length + 1.
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    /// `attention_logit_cap` — the softcap `w/cap → tanh → ×cap`.
    pub attention_logit_cap: f32,
    /// `attention_invalid_logits_value` — what a key outside the block
    /// context (the left/right edges of the sequence) is filled with before
    /// the softmax, i.e. excluded. The M kept rows' windows are all-valid
    /// EXCEPT at the first `left-1` positions and past the last real frame,
    /// where this is what removes the padded keys from the denominator.
    pub attention_invalid_logits_value: f32,
    /// Macaron FFN residual weight (0.5).
    pub residual_weight: f32,
    /// `subsampling_conv_channels` — the two stages' output channels.
    pub subsampling_conv_channels: [usize; 2],
    /// When true every clippable linear clamps its input and output to the
    /// checkpoint's calibrated bounds (this checkpoint: true, and the bounds
    /// are real finite values). When false the bounds are not read at all.
    pub use_clipped_linears: bool,
    /// `gradient_clipping` — the reference clamps the residual stream to
    /// `min(gradient_clipping, dtype_max)` at six points per layer. On this
    /// checkpoint that is 1e10, i.e. provably inert for f32 activations.
    pub gradient_clipping: f32,
}

impl AudioConfig {
    /// `n_heads * head_dim` — the attention projections' output width.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// `num_attention_heads * head_dim` is the model width here; the
    /// conformer's q/k/v are all full width.
    pub fn ffn_dim(&self) -> usize {
        self.hidden_size * 4
    }

    /// `linear_start`'s output width (GLU input, half of it survives).
    pub fn lconv_dim(&self) -> usize {
        self.hidden_size * 2
    }

    /// The blocked attention's sliding context: `chunk + past + future`
    /// (24 on this checkpoint: 12 + 12 + 0).
    pub fn context_size(&self) -> usize {
        self.attention_chunk_size + self.attention_context_left - 1 + self.attention_context_right
    }

    /// Frequencies out of the two stride-2 conv stages for a `mel_bins`-wide
    /// input: `mel_bins → (bins-1)/2+1 → …`. The checkpoint's 128 mel bins
    /// land on 32, which is the factor the flatten's `f*C + c` index and the
    /// projection width are built from.
    pub fn conv_freq_out(&self, mel_bins: usize) -> usize {
        let mut w = mel_bins;
        for _ in 0..self.subsampling_conv_channels.len() {
            w = if w < 3 { 0 } else { (w - 1) / 2 + 1 };
        }
        w
    }

    /// The subsample projection's input width: `freq_out * channels[1]`.
    pub fn subsample_input_dim(&self, mel_bins: usize) -> usize {
        self.conv_freq_out(mel_bins) * self.subsampling_conv_channels[1]
    }
}

/// The audio frontend's geometry: `processor_config.json`'s
/// `feature_extractor` block (HF `Gemma4AudioFeatureExtractor`).
///
/// The extractor class's own defaults are the fallback when no processor
/// config is readable (an `.hfq` carries the model config only), and they
/// reproduce this checkpoint exactly. Unsupported extractor knobs are
/// REFUSED rather than silently dropped — dithering, preemphasis, an input
/// scale factor, per-bin normalisation, left padding and FFT overdrive all
/// change the features this implementation produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioFrontendConfig {
    pub sampling_rate: usize,
    /// Analysis window in samples (20 ms at 16 kHz → 320).
    pub frame_length: usize,
    /// Frame step in samples (10 ms at 16 kHz → 160).
    pub hop_length: usize,
    /// RFFT length: the window is right-zero-padded to it (512).
    pub fft_length: usize,
    /// Mel filters (`feature_size`, 128).
    pub mel_bins: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    /// `log(mel + floor)`.
    pub mel_floor: f32,
}

/// Samples per padding quantum. The processor calls the extractor with
/// `pad_to_multiple_of=128` (the extractor's own default, there for TPU
/// shapes); it is what decides the frame count, and therefore the soft-token
/// count, so it is part of the frontend contract rather than a tuning knob.
pub const AUDIO_PAD_TO_MULTIPLE_OF: usize = 128;

impl Default for AudioFrontendConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            mel_bins: 128,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            mel_floor: 1e-3,
        }
    }
}

impl AudioFrontendConfig {
    /// Parse a `processor_config.json` (or its `feature_extractor` block).
    /// Keys absent from the file keep the class defaults; the ones that would
    /// change the features are refused.
    pub fn from_processor_json(pc: &serde_json::Value) -> Result<Self, String> {
        let fe = pc.get("feature_extractor").unwrap_or(pc);
        let mut cfg = Self::default();

        if let Some(fs) = u64_of(fe, "feature_size") {
            cfg.mel_bins = fs as usize;
        }
        if let Some(sr) = u64_of(fe, "sampling_rate") {
            cfg.sampling_rate = sr as usize;
        }
        if let Some(v) = u64_of(fe, "frame_length") {
            cfg.frame_length = v as usize;
        }
        if let Some(v) = u64_of(fe, "hop_length") {
            cfg.hop_length = v as usize;
        }
        if let Some(v) = u64_of(fe, "fft_length") {
            cfg.fft_length = v as usize;
        }
        if let Some(v) = f64_of(fe, "min_frequency") {
            cfg.min_frequency = v;
        }
        if let Some(v) = f64_of(fe, "max_frequency") {
            cfg.max_frequency = v;
        }
        if let Some(v) = f64_of(fe, "mel_floor") {
            cfg.mel_floor = v as f32;
        }

        for (key, why) in [
            ("dither", "we add no dither noise"),
            (
                "preemphasis",
                "the HTK preemphasis branch is not implemented",
            ),
            ("input_scale_factor", "we do not rescale the waveform"),
            ("max_length", "the clip is never truncated (see the caller)"),
        ] {
            let changed = match fe.get(key) {
                Some(x) if x.is_f64() => {
                    x.as_f64().unwrap_or(0.0)
                        != if key == "input_scale_factor" {
                            1.0
                        } else {
                            0.0
                        }
                }
                _ => false,
            };
            if changed {
                return Err(format!(
                    "processor_config feature_extractor.{key} is not the default: {why} — \
                     refusing rather than silently producing different mel features"
                ));
            }
        }
        if fe.get("fft_overdrive").and_then(|v| v.as_bool()) == Some(true) {
            return Err(
                "processor_config feature_extractor.fft_overdrive=true doubles the FFT \
                 length; refusing rather than silently producing different mel features"
                    .into(),
            );
        }
        for key in ["per_bin_mean", "per_bin_stddev"] {
            if fe.get(key).map(|v| !v.is_null()).unwrap_or(false) {
                return Err(format!(
                    "processor_config feature_extractor.{key} is set: per-bin mel \
                     normalisation is not implemented; refusing rather than silently \
                     dropping it"
                ));
            }
        }
        if let Some(v) = fe.get("padding_side").and_then(|v| v.as_str()) {
            if v != "right" {
                return Err(format!(
                    "processor_config feature_extractor.padding_side={v:?}: only right \
                     padding is implemented"
                ));
            }
        }

        cfg.validate()?;
        Ok(cfg)
    }

    /// Cross-field invariants the frontend's math depends on.
    pub fn validate(&self) -> Result<(), String> {
        if self.sampling_rate == 0 || self.frame_length == 0 || self.hop_length == 0 {
            return Err(
                "audio frontend: sampling_rate / frame_length / hop_length must be > 0".into(),
            );
        }
        if self.frame_length % 2 != 0 {
            return Err(format!(
                "audio frontend: frame_length {} must be even (pad_left = frame_length / 2)",
                self.frame_length
            ));
        }
        if self.fft_length < self.frame_length || !self.fft_length.is_power_of_two() {
            return Err(format!(
                "audio frontend: fft_length {} must be a power of two ≥ frame_length {} \
                 (the radix-2 transform + right zero padding)",
                self.fft_length, self.frame_length
            ));
        }
        if self.mel_bins == 0 || self.fft_length / 2 + 1 < 2 {
            return Err("audio frontend: mel_bins / fft_length too small".into());
        }
        let (lo, hi) = (self.min_frequency, self.max_frequency);
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi <= lo {
            return Err(format!(
                "audio frontend: min_frequency {lo} / max_frequency {hi} invalid \
                 (both must be finite Hz with 0 <= min < max)"
            ));
        }
        if self.max_frequency > (self.sampling_rate as f64) / 2.0 {
            return Err(format!(
                "audio frontend: max_frequency {} exceeds the Nyquist rate {}",
                self.max_frequency,
                self.sampling_rate as f64 / 2.0
            ));
        }
        if self.mel_floor.is_nan() || self.mel_floor <= 0.0 {
            return Err(format!(
                "audio frontend: mel_floor {} must be > 0",
                self.mel_floor
            ));
        }
        Ok(())
    }

    /// FFT bins the mel bank consumes: `fft_length / 2 + 1`.
    pub fn fft_bins(&self) -> usize {
        self.fft_length / 2 + 1
    }

    /// Samples fed to the transform chain: the clip padded up to the
    /// processor's `pad_to_multiple_of` quantum.
    pub fn padded_samples(&self, samples: usize) -> usize {
        samples.div_ceil(AUDIO_PAD_TO_MULTIPLE_OF) * AUDIO_PAD_TO_MULTIPLE_OF
    }

    /// Mel frames for a padded sample count: the semicausal pad adds
    /// `frame_length / 2` samples, the unfold window is `frame_length + 1`
    /// wide and steps by `hop_length`, and the last sample of each window is
    /// dropped (`frame_length` samples survive).
    pub fn frames(&self, padded_samples: usize) -> usize {
        let pad_left = self.frame_length / 2;
        let total = padded_samples + pad_left;
        let window = self.frame_length + 1;
        if total < window {
            0
        } else {
            (total - window) / self.hop_length + 1
        }
    }

    /// Frames whose whole analysis window is real audio: the mask is checked
    /// at each frame's LAST sample (`i*hop + frame_length`), i.e. a frame is
    /// valid only when everything it covers is.
    pub fn valid_frames(&self, samples: usize) -> usize {
        let pad_left = self.frame_length / 2;
        let frames = self.frames(self.padded_samples(samples));
        let mut valid = 0;
        while valid < frames && valid * self.hop_length + self.frame_length < pad_left + samples {
            valid += 1;
        }
        valid
    }
}

/// How the video frontend meets its `max_frames` budget
/// (`processor_config.json:video_processor.overflow_strategy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowStrategy {
    /// `np.linspace(0, total-1, max_frames, dtype=int)` — an even stride
    /// across the clip (HF's `"uniform"`).
    Uniform,
    /// Keep the first `max_frames` (HF's `"truncate"`).
    Truncate,
}

/// The Gemma-4 video frontend's geometry: `processor_config.json`'s
/// `video_processor` block (HF `EmbeddingGemma2VideoProcessor`).
///
/// The class defaults are the fallback when no processor config is readable
/// (an `.hfq` carries the model config only). `max_soft_tokens` is the one
/// knob whose class default (70) is NOT this checkpoint's value: its sidecar
/// says 140, and 140 is what reproduces the float32 reference, so that is the
/// default here too (same posture as [`AudioFrontendConfig`], whose defaults
/// are this checkpoint's extractor).
///
/// Knobs that would change the features are REFUSED rather than dropped:
/// `add_timestamps` (the `mm:ss ` text this leg does not synthesize),
/// `num_frames` (HF's own sampler raises on it), a `max_soft_tokens` outside
/// the supported set (HF raises), an unknown `overflow_strategy` (HF raises),
/// a non-identity normalization, a rescale factor that is not 1/255, a
/// `resample` that is not bicubic, and `do_resize`/`do_rescale`/
/// `do_convert_rgb` turned off. `fps` is read but unused — a decoded frame
/// list carries no capture rate, which is exactly the case where HF skips its
/// `fps` pass too (`sample_frames`' `metadata.fps is None` branch).
///
/// `default_to_square` / `size` are inert here: the video processor overrides
/// the whole resize with `aspect_ratio_preserving_resize` (patch_size ×
/// max_soft_tokens × pooling_kernel_size), so they cannot change a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoFrontendConfig {
    /// One of [`SUPPORTED_SOFT_TOKENS`] — the per-FRAME soft-token budget the
    /// frame resize targets (`max_patches = max_soft_tokens * k²`).
    pub max_soft_tokens: usize,
    /// Checked against the tower's own `vision_config` patch_size (the
    /// patch-embedder GEMM is built for it).
    pub patch_size: usize,
    /// Same, for the pooler's kernel.
    pub pooling_kernel_size: usize,
    /// `max_frames` — the sampler's frame budget (32 on this release).
    pub max_frames: usize,
    /// `do_sample_frames`: false keeps every supplied frame (HF then never
    /// calls its sampler at all).
    pub do_sample_frames: bool,
    /// How the budget is met when the clip is longer than `max_frames`.
    pub overflow: OverflowStrategy,
    /// `rescale_factor` — must be 1/255 (the frame pipeline's only scaling).
    pub rescale_factor: f32,
}

impl Default for VideoFrontendConfig {
    fn default() -> Self {
        Self {
            // The checkpoint's sidecar value. HF's class default is 70; 140 is
            // what this checkpoint's video_processor declares and what the
            // float32 reference runs.
            max_soft_tokens: 140,
            patch_size: 16,
            pooling_kernel_size: 3,
            max_frames: 32,
            do_sample_frames: true,
            overflow: OverflowStrategy::Uniform,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

impl VideoFrontendConfig {
    /// Parse a `processor_config.json` (or its `video_processor` block).
    /// Keys absent from the file keep the defaults; the ones that would
    /// change the frames are refused.
    pub fn from_processor_json(pc: &serde_json::Value) -> Result<Self, String> {
        let vp = pc.get("video_processor").unwrap_or(pc);
        let mut cfg = Self::default();

        if let Some(v) = u64_of(vp, "max_soft_tokens") {
            cfg.max_soft_tokens = v as usize;
        }
        if let Some(v) = u64_of(vp, "patch_size") {
            cfg.patch_size = v as usize;
        }
        if let Some(v) = u64_of(vp, "pooling_kernel_size") {
            cfg.pooling_kernel_size = v as usize;
        }
        if let Some(v) = u64_of(vp, "max_frames") {
            cfg.max_frames = v as usize;
        }
        if let Some(v) = vp.get("do_sample_frames").and_then(|v| v.as_bool()) {
            cfg.do_sample_frames = v;
        }
        if let Some(v) = f64_of(vp, "rescale_factor") {
            cfg.rescale_factor = v as f32;
        }
        if let Some(v) = vp.get("overflow_strategy").and_then(|v| v.as_str()) {
            cfg.overflow = match v {
                "uniform" => OverflowStrategy::Uniform,
                "truncate" => OverflowStrategy::Truncate,
                other => {
                    return Err(format!(
                        "processor_config video_processor.overflow_strategy {other:?} is not \
                         one of [\"truncate\", \"uniform\"] — refusing rather than sampling \
                         frames a different way than the checkpoint's processor does"
                    ))
                }
            };
        }

        if !SUPPORTED_SOFT_TOKENS.contains(&cfg.max_soft_tokens) {
            return Err(format!(
                "processor_config video_processor.max_soft_tokens {}: must be one of \
                 {SUPPORTED_SOFT_TOKENS:?} (HF's own check)",
                cfg.max_soft_tokens
            ));
        }
        if cfg.patch_size == 0 || cfg.pooling_kernel_size == 0 {
            return Err(
                "processor_config video_processor patch_size / pooling_kernel_size must be > 0"
                    .into(),
            );
        }
        if cfg.max_frames == 0 {
            return Err(
                "processor_config video_processor.max_frames 0 would sample no frames at all"
                    .into(),
            );
        }
        if vp.get("add_timestamps").and_then(|v| v.as_bool()) == Some(true) {
            return Err(
                "processor_config video_processor.add_timestamps=true: the `mm:ss ` timestamp \
                 text is not implemented (this checkpoint ships add_timestamps=false); \
                 refusing rather than emitting a prompt with different tokens"
                    .into(),
            );
        }
        for key in ["num_frames"] {
            if vp.get(key).map(|v| !v.is_null()).unwrap_or(false) {
                return Err(format!(
                    "processor_config video_processor.{key} is set: HF's own sampler raises \
                     \"Sampling with `num_frames` is not supported\" on it — use \
                     `max_frames` (+ `overflow_strategy`) instead"
                ));
            }
        }
        // The frame pipeline's scaling: `[0,1]` rescale and nothing else.
        if (cfg.rescale_factor - (1.0f32 / 255.0)).abs() > 0.0 {
            return Err(format!(
                "processor_config video_processor.rescale_factor {} is not 1/255 — the frame \
                 pipeline rescales by 1/255 only; refusing rather than producing different \
                 pixels",
                cfg.rescale_factor
            ));
        }
        match vp.get("do_resize").and_then(|v| v.as_bool()) {
            Some(false) => {
                return Err(
                    "processor_config video_processor.do_resize=false: every frame must go \
                     through the aspect-preserving resize (the tower's patch grid is built \
                     for it)"
                        .into(),
                )
            }
            _ => {}
        }
        match vp.get("do_rescale").and_then(|v| v.as_bool()) {
            Some(false) => {
                return Err(
                    "processor_config video_processor.do_rescale=false: the tower reads \
                     `[0,1]` pixels"
                        .into(),
                )
            }
            _ => {}
        }
        match vp.get("do_convert_rgb").and_then(|v| v.as_bool()) {
            Some(false) => {
                return Err(
                    "processor_config video_processor.do_convert_rgb=false: the patch \
                     layout is 3-channel RGB"
                        .into(),
                )
            }
            _ => {}
        }
        for key in ["do_center_crop", "do_pad"] {
            if vp.get(key).and_then(|v| v.as_bool()) == Some(true) {
                return Err(format!(
                    "processor_config video_processor.{key}=true is not implemented"
                ));
            }
        }
        if let Some(v) = vp.get("crop_size").filter(|v| !v.is_null()) {
            return Err(format!(
                "processor_config video_processor.crop_size {v} is set but cropping is not \
                 implemented"
            ));
        }
        if let Some(v) = u64_of(vp, "resample") {
            // PILImageResampling: 3 = BICUBIC, the only filter the transcribed
            // uint8 resize kernel implements.
            if v != 3 {
                return Err(format!(
                    "processor_config video_processor.resample {v} is not BICUBIC (3) — the \
                     resize kernel is a transcription of torchvision's bicubic uint8 path"
                ));
            }
        }
        // Normalization must be the identity either way: `do_normalize=false`
        // is this checkpoint's setting, and mean 0 / std 1 makes the flag moot.
        let zero_mean = vp
            .get("image_mean")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().all(|x| x.as_f64() == Some(0.0)))
            .unwrap_or(true);
        let unit_std = vp
            .get("image_std")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().all(|x| x.as_f64() == Some(1.0)))
            .unwrap_or(true);
        if !zero_mean || !unit_std {
            return Err(
                "processor_config video_processor image_mean/image_std are not 0/1: per-channel \
                 normalization is not implemented; refusing rather than feeding the tower \
                 different pixels"
                    .into(),
            );
        }

        Ok(cfg)
    }

    /// Patch budget per frame: `max_soft_tokens * pooling_kernel_size²`
    /// (1 260 with this checkpoint's 140).
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

/// The parsed config.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub layer_types: Vec<LayerType>,
    /// Resolved per layer, same length as `layer_types`.
    pub layer_attn: Vec<LayerAttn>,
    pub n_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// Inclusive radius: a sliding layer attends to |i-j| <= window.
    pub sliding_window: usize,
    pub sliding_rope_theta: f32,
    pub full_rope_theta: f32,
    /// PLE width (`hidden_size_per_layer_input`); 0 disables the PLE paths.
    pub ple_dim: usize,
    /// Final projection width (`embedding_dim`).
    pub embedding_dim: usize,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,

    // Multimodal placeholder ids (recognized so they can be refused with a
    // pointing error rather than embedded as text).
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,
    /// The begin/end-of-image markers a text image marker expands to
    /// (`[boi, image_token×N, eoi]`).
    pub boi_token_id: u32,
    pub eoi_token_id: u32,

    /// The begin/end-of-audio markers a text audio marker expands to
    /// (`[boa, audio_token×M, eoa]`).
    pub boa_token_id: u32,
    pub eoa_token_id: u32,

    /// The Gemma-4 vision tower's geometry when the checkpoint declares a
    /// `vision_config`. `None` = text-only config. The tower's WEIGHTS are
    /// not part of the trunk: they arrive in the `.vl` sidecar.
    pub vision: Option<VisionConfig>,

    /// The Gemma-4 audio tower's geometry when the checkpoint declares an
    /// `audio_config`. `None` = no audio tower. Like the vision tower, its
    /// WEIGHTS arrive in the `.vl` sidecar.
    pub audio: Option<AudioConfig>,

    /// The audio frontend's geometry. Always present (the extractor has
    /// defaults) — the tower is what may be absent.
    pub audio_frontend: AudioFrontendConfig,

    /// The video frontend's geometry. Always present (its processor has
    /// defaults) — the tower it drives is the VISION tower, which may be
    /// absent.
    pub video_frontend: VideoFrontendConfig,

    // ── sentence-transformers profile ──────────────────────────────────
    /// Task-instruction prefix prepended to the raw text per kind.
    pub query_prompt: String,
    pub document_prompt: String,
    /// Max token count (prompt + bos + eos included) per kind.
    pub query_length: usize,
    pub document_length: usize,
}

/// This release's defaults: the model card's prompt prefixes and the 8 192
/// shared context window. A checkpoint carrying its own values in
/// `config_sentence_transformers.json` / `sentence_bert_config.json` wins.
pub const DEFAULT_QUERY_PROMPT: &str = "task: search result | query: ";
pub const DEFAULT_DOCUMENT_PROMPT: &str = "title: none | text: ";
pub const DEFAULT_QUERY_LENGTH: usize = 8192;
pub const DEFAULT_DOCUMENT_LENGTH: usize = 8192;

fn u64_of(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn f64_of(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

/// Unwrap a checkpoint-metadata blob to the composite config object (both
/// source kinds wrap the checkpoint config under `"config"`).
pub fn config_from_metadata(meta: &serde_json::Value) -> &serde_json::Value {
    meta.get("config").unwrap_or(meta)
}

fn parse_layer_types(
    text: &serde_json::Value,
    num_layers: usize,
) -> Result<Vec<LayerType>, String> {
    let raw = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            "text_config.layer_types is required (the builder emits it; a \
             hand-written config must spell every layer out)"
                .to_string()
        })?;
    if raw.len() != num_layers {
        return Err(format!(
            "text_config.layer_types has {} entries for num_hidden_layers {num_layers}",
            raw.len()
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(i, v)| match v.as_str().unwrap_or("") {
            "sliding_attention" => Ok(LayerType::Sliding),
            "full_attention" => Ok(LayerType::Full),
            other => Err(format!(
                "layer_types[{i}] = {other:?}: embeddinggemma2 supports \
                 sliding_attention / full_attention"
            )),
        })
        .collect()
}

/// Parse a `per_layer_config` key as a layer index. HF serializes the keys
/// zero-padded (`"05"`); accept plain digits too.
fn layer_index_key(key: &str) -> Option<usize> {
    key.parse::<usize>().ok().or_else(|| {
        let t = key.trim_start_matches('0');
        if t.is_empty() {
            Some(0)
        } else {
            t.parse::<usize>().ok()
        }
    })
}

/// Parse the composite config's `vision_config` (the Gemma-4 vision tower's
/// geometry). `Ok(None)` = the config declares no vision tower at all.
///
/// Config-gated behaviour is REFUSED here rather than silently ignored: a
/// checkpoint that ever sets `use_clipped_linears` or `standardize` true
/// needs clamp reads / std-bias-subtract tensors this implementation does
/// not carry, and quietly dropping them would change the tower's output.
pub fn parse_vision_config(config: &serde_json::Value) -> Result<Option<VisionConfig>, String> {
    let Some(v) = config.get("vision_config").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if let Some(mt) = v.get("model_type").and_then(|m| m.as_str()) {
        if mt != "gemma4_vision" {
            return Err(format!(
                "vision_config.model_type {mt:?}: embeddinggemma2 implements the \
                 gemma4_vision tower"
            ));
        }
    }
    for (key, why) in [
        (
            "use_clipped_linears",
            "the clip bounds are checkpoint buffers this implementation does not \
             load; refusing rather than silently dropping the clamps",
        ),
        (
            "standardize",
            "the std_bias/std_scale buffers are checkpoint buffers this \
             implementation does not load; refusing rather than silently \
             dropping the standardization",
        ),
    ] {
        if v.get(key).and_then(|x| x.as_bool()) == Some(true) {
            return Err(format!("vision_config.{key}=true: {why}"));
        }
    }
    let rope = v.get("rope_parameters").cloned().unwrap_or_default();
    let rope_type = rope
        .get("rope_type")
        .and_then(|r| r.as_str())
        .unwrap_or("axial");
    if rope_type != "axial" {
        return Err(format!(
            "vision_config.rope_parameters.rope_type {rope_type:?}: this tower's \
             RoPE is the axial 2-D H-H-W-W layout (only \"axial\" is supported)"
        ));
    }
    let rope_theta = f64_of(&rope, "rope_theta").unwrap_or(100.0) as f32;

    let hidden_size = u64_of(v, "hidden_size").ok_or("vision_config.hidden_size")? as usize;
    let num_layers =
        u64_of(v, "num_hidden_layers").ok_or("vision_config.num_hidden_layers")? as usize;
    let intermediate_size =
        u64_of(v, "intermediate_size").ok_or("vision_config.intermediate_size")? as usize;
    let n_heads =
        u64_of(v, "num_attention_heads").ok_or("vision_config.num_attention_heads")? as usize;
    let n_kv_heads = u64_of(v, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
    let base_head_dim = u64_of(v, "head_dim").unwrap_or(0) as usize;
    let head_dim = if base_head_dim == 0 {
        if n_heads == 0 {
            return Err("vision_config.num_attention_heads must be > 0".into());
        }
        hidden_size / n_heads
    } else {
        base_head_dim
    };
    let rms_norm_eps = f64_of(v, "rms_norm_eps").unwrap_or(1e-6) as f32;
    let patch_size = u64_of(v, "patch_size").unwrap_or(16) as usize;
    let pooling_kernel_size = u64_of(v, "pooling_kernel_size").unwrap_or(3) as usize;
    let position_embedding_size = u64_of(v, "position_embedding_size").unwrap_or(10240) as usize;
    let max_soft_tokens = u64_of(v, "max_soft_tokens").unwrap_or(280) as usize;

    let act = v
        .get("hidden_activation")
        .or_else(|| v.get("hidden_act"))
        .and_then(|a| a.as_str())
        .unwrap_or("gelu_pytorch_tanh");
    if act != "gelu_pytorch_tanh" {
        return Err(format!(
            "vision_config.hidden_activation {act:?}: only gelu_pytorch_tanh is supported"
        ));
    }
    if v.get("attention_bias").and_then(|b| b.as_bool()) == Some(true) {
        return Err("vision_config.attention_bias=true is not supported (bias-free)".into());
    }
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(format!(
            "vision_config.max_soft_tokens {max_soft_tokens}: must be one of \
             {SUPPORTED_SOFT_TOKENS:?} (HF Gemma4ImageProcessor's set)"
        ));
    }
    if pooling_kernel_size == 0 || patch_size == 0 {
        return Err("vision_config.pooling_kernel_size / patch_size must be > 0".into());
    }
    if hidden_size % 16 != 0 || intermediate_size % 16 != 0 {
        return Err(format!(
            "vision_config hidden {hidden_size} / intermediate {intermediate_size} must be \
             multiples of 16 (GEMM contract)"
        ));
    }
    // head_dim must carry the WMMA K tile for the eager attention path and
    // the four equal quarters the axial RoPE splits into.
    if head_dim == 0 || head_dim % 64 != 0 {
        return Err(format!(
            "vision_config head_dim {head_dim} must be a nonzero multiple of 64 \
             (the attention GEMM's K tile + the axial RoPE's four quarters)"
        ));
    }
    if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(format!(
            "vision_config {n_heads} heads not divisible by {n_kv_heads} kv heads"
        ));
    }
    if (3 * patch_size * patch_size) % 16 != 0 {
        return Err(format!(
            "vision_config patch_size {patch_size}: 3*patch² patch elements must be a \
             multiple of 16 (the patch-embedder GEMM's K)"
        ));
    }

    Ok(Some(VisionConfig {
        hidden_size,
        num_layers,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_norm_eps,
        patch_size,
        pooling_kernel_size,
        position_embedding_size,
        rope_theta,
        max_soft_tokens,
    }))
}

/// Parse the composite config's `audio_config` (the Gemma-4 audio tower's
/// geometry). `Ok(None)` = the config declares no audio tower at all.
///
/// Config-gated behaviour is refused rather than silently ignored, the same
/// contract as [`parse_vision_config`]: an activation other than silu, a
/// chunked attention geometry the blocked mask cannot express, or a tower
/// whose subsample flatten does not line up with the mel bank are all load
/// errors. `use_clipped_linears` is NOT refused — the tower honours it (the
/// bounds are read and applied when true, not read at all when false).
pub fn parse_audio_config(config: &serde_json::Value) -> Result<Option<AudioConfig>, String> {
    let Some(v) = config.get("audio_config").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if let Some(mt) = v.get("model_type").and_then(|m| m.as_str()) {
        if mt != "gemma4_audio" {
            return Err(format!(
                "audio_config.model_type {mt:?}: embeddinggemma2 implements the \
                 gemma4_audio tower"
            ));
        }
    }
    let act = v
        .get("hidden_act")
        .or_else(|| v.get("hidden_activation"))
        .and_then(|a| a.as_str())
        .unwrap_or("silu");
    if act != "silu" {
        return Err(format!(
            "audio_config.hidden_act {act:?}: this tower's FFN and light-conv \
             activations are silu only"
        ));
    }
    if v.get("attention_bias").and_then(|b| b.as_bool()) == Some(true) {
        return Err("audio_config.attention_bias=true is not supported (bias-free)".into());
    }

    let hidden_size = u64_of(v, "hidden_size").ok_or("audio_config.hidden_size")? as usize;
    let num_layers =
        u64_of(v, "num_hidden_layers").ok_or("audio_config.num_hidden_layers")? as usize;
    let n_heads =
        u64_of(v, "num_attention_heads").ok_or("audio_config.num_attention_heads")? as usize;
    if hidden_size == 0 || n_heads == 0 || num_layers == 0 {
        return Err("audio_config: hidden_size / num_attention_heads / \
                    num_hidden_layers must be > 0"
            .into());
    }
    if hidden_size % n_heads != 0 {
        return Err(format!(
            "audio_config: hidden_size {hidden_size} is not divisible by \
             num_attention_heads {n_heads} (head_dim is derived as the quotient)"
        ));
    }
    let head_dim = hidden_size / n_heads;
    if head_dim % 2 != 0 {
        return Err(format!(
            "audio_config: head_dim {head_dim} must be even (the relative-position \
             encoding splits it into sin/cos timescales)"
        ));
    }
    let output_proj_dims = u64_of(v, "output_proj_dims").unwrap_or(hidden_size as u64) as usize;
    let conv_kernel_size = u64_of(v, "conv_kernel_size").unwrap_or(5) as usize;
    let attention_chunk_size = u64_of(v, "attention_chunk_size").unwrap_or(12) as usize;
    let attention_context_left = u64_of(v, "attention_context_left").unwrap_or(13) as usize;
    let attention_context_right = u64_of(v, "attention_context_right").unwrap_or(0) as usize;
    let attention_logit_cap = f64_of(v, "attention_logit_cap").unwrap_or(50.0) as f32;
    let attention_invalid_logits_value =
        f64_of(v, "attention_invalid_logits_value").unwrap_or(-1e9) as f32;
    let residual_weight = f64_of(v, "residual_weight").unwrap_or(0.5) as f32;
    let gradient_clipping = f64_of(v, "gradient_clipping").unwrap_or(1e10) as f32;
    let use_clipped_linears = v
        .get("use_clipped_linears")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let channels: Vec<usize> = v
        .get("subsampling_conv_channels")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|e| e.as_u64())
                .map(|e| e as usize)
                .collect()
        })
        .unwrap_or_else(|| vec![128, 32]);
    let channels: [usize; 2] = channels.as_slice().try_into().map_err(|_| {
        format!(
            "audio_config.subsampling_conv_channels must hold exactly 2 stage widths, \
             got {channels:?}"
        )
    })?;

    if conv_kernel_size == 0 || attention_chunk_size == 0 || attention_context_left == 0 {
        return Err("audio_config: conv_kernel_size / attention_chunk_size / \
                    attention_context_left must be > 0"
            .into());
    }
    if attention_logit_cap.is_nan() || attention_logit_cap <= 0.0 {
        return Err(format!(
            "audio_config.attention_logit_cap {attention_logit_cap} must be > 0 \
             (the softcap divides by it)"
        ));
    }
    if attention_invalid_logits_value.is_nan() || attention_invalid_logits_value >= 0.0 {
        return Err(format!(
            "audio_config.attention_invalid_logits_value {attention_invalid_logits_value} \
             must be negative: it is the fill that removes an out-of-window key from the \
             softmax, so a non-negative value would give it weight"
        ));
    }
    if gradient_clipping.is_nan() || gradient_clipping <= 0.0 {
        return Err(format!(
            "audio_config.gradient_clipping {gradient_clipping} must be > 0"
        ));
    }
    if channels[0] == 0 || channels[1] == 0 {
        return Err("audio_config.subsampling_conv_channels must be nonzero".into());
    }
    // The block context (`chunk + past + future`) is built so that every
    // query's window sits inside it by construction: the window is
    // `left - 1` back and `right` forward, and the context pads exactly that
    // far on each side. Nothing to check — recorded so a reader does not go
    // looking for a cross-field invariant that does not exist.
    for (what, w) in [
        ("hidden_size", hidden_size),
        ("output_proj_dims", output_proj_dims),
        ("ffn dim", hidden_size * 4),
        ("lconv dim", hidden_size * 2),
        ("subsampling_conv_channels[0]", channels[0]),
        ("subsampling_conv_channels[1]", channels[1]),
    ] {
        if w % 16 != 0 {
            return Err(format!(
                "audio_config {what} {w} must be a multiple of 16 (GEMM contract)"
            ));
        }
    }
    // The subsample projection's input is the flattened conv output:
    // `(channels[0] / 4) * channels[1]`. The `/4` is the frequency decimation
    // of two stride-2 stages on the checkpoint's 128-bin mel input; the
    // cross-check against the real mel bank happens once the frontend config
    // is known (see `parse_config`).
    let want_proj_in = (channels[0] / 4) * channels[1];
    if want_proj_in % 16 != 0 {
        return Err(format!(
            "audio_config: the subsample projection input {want_proj_in} must be a \
             multiple of 16 (GEMM contract)"
        ));
    }

    Ok(Some(AudioConfig {
        hidden_size,
        num_layers,
        n_heads,
        head_dim,
        rms_norm_eps: f64_of(v, "rms_norm_eps").unwrap_or(1e-6) as f32,
        output_proj_dims,
        conv_kernel_size,
        attention_chunk_size,
        attention_context_left,
        attention_context_right,
        attention_logit_cap,
        attention_invalid_logits_value,
        residual_weight,
        subsampling_conv_channels: channels,
        use_clipped_linears,
        gradient_clipping,
    }))
}

/// Parse the composite `config.json` (a `serde_json::Value` of its contents).
pub fn parse_config(config: &serde_json::Value) -> Result<EmbedConfig, String> {
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config (embeddinggemma2 is a composite model)")?;

    let hidden_size = u64_of(text, "hidden_size").ok_or("text_config.hidden_size")? as usize;
    let num_layers =
        u64_of(text, "num_hidden_layers").ok_or("text_config.num_hidden_layers")? as usize;
    let n_heads = u64_of(text, "num_attention_heads").ok_or("num_attention_heads")? as usize;
    let base_head_dim = u64_of(text, "head_dim").unwrap_or(0) as usize;
    let base_kv = u64_of(text, "num_key_value_heads").unwrap_or(0) as usize;
    if n_heads == 0 || base_kv == 0 {
        return Err(format!(
            "num_attention_heads={n_heads} / num_key_value_heads={base_kv}: both must be > 0"
        ));
    }
    let head_dim = if base_head_dim == 0 {
        hidden_size / n_heads
    } else {
        base_head_dim
    };
    let intermediate_size = u64_of(text, "intermediate_size").ok_or("intermediate_size")? as usize;
    let vocab_size = u64_of(text, "vocab_size").ok_or("vocab_size")? as usize;
    let rms_norm_eps = f64_of(text, "rms_norm_eps").unwrap_or(1e-6) as f32;
    let sliding_window = u64_of(text, "sliding_window").unwrap_or(512) as usize;

    let act = text
        .get("hidden_activation")
        .or_else(|| text.get("hidden_act"))
        .and_then(|v| v.as_str())
        .unwrap_or("gelu_pytorch_tanh");
    if act != "gelu_pytorch_tanh" {
        return Err(format!(
            "hidden_activation {act:?}: only gelu_pytorch_tanh is supported"
        ));
    }
    if text.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
        return Err("attention_bias=true is not supported (bias-free projections)".into());
    }
    // This arch has no `is_causal` key at all — attention is bidirectional by
    // construction. A future checkpoint that declares causal attention would
    // be silently wrong here, so refuse it.
    if text.get("is_causal").and_then(|v| v.as_bool()) == Some(true) {
        return Err("is_causal=true is not the bidirectional encoder this arch implements".into());
    }

    // rope_parameters nests per-layer-type theta.
    let rope = text.get("rope_parameters").cloned().unwrap_or_default();
    let rope_theta_of = |ty: &str, default: f64| -> f64 {
        rope.get(ty)
            .and_then(|r| f64_of(r, "rope_theta"))
            .unwrap_or(default)
    };
    let sliding_rope_theta = rope_theta_of("sliding_attention", 10_000.0) as f32;
    let full_rope_theta = rope_theta_of("full_attention", 1_000_000.0) as f32;

    let layer_types = parse_layer_types(text, num_layers)?;

    // Resolve per-layer geometry: base (head_dim, kv) overridden by
    // per_layer_config entries keyed by layer index.
    let mut layer_attn = vec![
        LayerAttn {
            head_dim,
            n_kv_heads: base_kv,
        };
        num_layers
    ];
    if let Some(plc) = text.get("per_layer_config").and_then(|v| v.as_object()) {
        for (key, ov) in plc {
            let Some(li) = layer_index_key(key) else {
                // Per-type entries ("sliding_attention") only shape building
                // upstream; the serialized per-layer values are authoritative.
                continue;
            };
            if li >= num_layers {
                return Err(format!(
                    "per_layer_config[{key}] = layer {li} >= num_hidden_layers {num_layers}"
                ));
            }
            let a = &mut layer_attn[li];
            if let Some(hd) = u64_of(ov, "head_dim") {
                a.head_dim = hd as usize;
            }
            if let Some(kv) = u64_of(ov, "num_key_value_heads") {
                a.n_kv_heads = kv as usize;
            }
        }
    }
    for (li, a) in layer_attn.iter().enumerate() {
        if a.head_dim == 0 || a.head_dim % 2 != 0 || a.head_dim % 64 != 0 {
            return Err(format!(
                "layer {li}: head_dim {} must be an even multiple of 64 \
                 (rope pairs + the WMMA GEMM's K tile)",
                a.head_dim
            ));
        }
        if a.n_kv_heads == 0 || n_heads % a.n_kv_heads != 0 {
            return Err(format!(
                "layer {li}: {n_heads} heads not divisible by {} kv heads",
                a.n_kv_heads
            ));
        }
    }
    if hidden_size % 16 != 0 || intermediate_size % 16 != 0 {
        return Err(format!(
            "hidden_size {hidden_size} / intermediate_size {intermediate_size} must be \
             multiples of 16 (GEMM contract)"
        ));
    }

    let ple_dim = u64_of(text, "hidden_size_per_layer_input").unwrap_or(0) as usize;
    if ple_dim != 0 && ple_dim % 16 != 0 {
        return Err(format!(
            "hidden_size_per_layer_input {ple_dim} must be a multiple of 16"
        ));
    }
    let embedding_dim = u64_of(text, "embedding_dim").ok_or("text_config.embedding_dim")? as usize;
    if embedding_dim % 16 != 0 {
        return Err(format!(
            "embedding_dim {embedding_dim} must be a multiple of 16"
        ));
    }

    let bos_token_id = u64_of(text, "bos_token_id").unwrap_or(2) as u32;
    let eos_token_id = u64_of(text, "eos_token_id").unwrap_or(1) as u32;
    let pad_token_id = u64_of(text, "pad_token_id").unwrap_or(0) as u32;
    let image_token_id = u64_of(config, "image_token_id").unwrap_or(258_880) as u32;
    let video_token_id = u64_of(config, "video_token_id").unwrap_or(258_884) as u32;
    let audio_token_id = u64_of(config, "audio_token_id").unwrap_or(258_881) as u32;
    let boi_token_id = u64_of(config, "boi_token_id").unwrap_or(255_999) as u32;
    let eoi_token_id = u64_of(config, "eoi_token_id").unwrap_or(258_882) as u32;
    let boa_token_id = u64_of(config, "boa_token_id").unwrap_or(256_000) as u32;
    // The release spells the end-of-audio marker `eoa_token_index`.
    let eoa_token_id = u64_of(config, "eoa_token_id")
        .or_else(|| u64_of(config, "eoa_token_index"))
        .unwrap_or(258_883) as u32;
    for (name, id) in [
        ("bos", bos_token_id),
        ("eos", eos_token_id),
        ("image", image_token_id),
        ("video", video_token_id),
        ("audio", audio_token_id),
        ("boi", boi_token_id),
        ("eoi", eoi_token_id),
        ("boa", boa_token_id),
        ("eoa", eoa_token_id),
    ] {
        if id as usize >= vocab_size {
            return Err(format!("{name}_token_id {id} >= vocab_size {vocab_size}"));
        }
    }

    // The tower geometry (`vision_config`) is config-side truth; its WEIGHTS
    // come from the `.vl` sidecar at load. A vision_config with
    // use_clipped_linears/standardize set is refused inside the parser.
    let vision = parse_vision_config(config)?;
    let audio = parse_audio_config(config)?;
    let audio_frontend = AudioFrontendConfig::default();
    let video_frontend = VideoFrontendConfig::default();
    // The subsample flatten (`f*C + c`) has to line up with the mel bank the
    // frontend actually produces — a checkpoint whose conv widths disagree
    // with its mel bins would otherwise fail much later, inside the GEMM.
    if let Some(a) = audio {
        let want = (a.subsampling_conv_channels[0] / 4) * a.subsampling_conv_channels[1];
        let got = a.subsample_input_dim(audio_frontend.mel_bins);
        if want != got {
            return Err(format!(
                "audio_config: the subsample projection expects {want} inputs \
                 ((channels[0]/4) * channels[1]) but two stride-2 convs over the frontend's \
                 {} mel bins produce {got}",
                audio_frontend.mel_bins
            ));
        }
    }

    Ok(EmbedConfig {
        hidden_size,
        num_layers,
        layer_types,
        layer_attn,
        n_heads,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        sliding_window,
        sliding_rope_theta,
        full_rope_theta,
        ple_dim,
        embedding_dim,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        image_token_id,
        video_token_id,
        audio_token_id,
        boi_token_id,
        eoi_token_id,
        boa_token_id,
        eoa_token_id,
        vision,
        audio,
        audio_frontend,
        video_frontend,
        query_prompt: DEFAULT_QUERY_PROMPT.to_string(),
        document_prompt: DEFAULT_DOCUMENT_PROMPT.to_string(),
        query_length: DEFAULT_QUERY_LENGTH,
        document_length: DEFAULT_DOCUMENT_LENGTH,
    })
}

impl EmbedConfig {
    /// The vision geometry the VIDEO leg resizes frames against: the tower's
    /// own `vision_config` with the video processor's per-frame patch budget
    /// (`max_soft_tokens`), i.e. 1 260 patches instead of the image leg's
    /// 2 520.
    ///
    /// `patch_size` / `pooling_kernel_size` must agree with `vision_config`:
    /// the tower's patch-embedder GEMM and pooler are built for those, so a
    /// sidecar that disagrees is a load-order problem, not a resize knob.
    pub fn frame_vision(&self) -> Result<VisionConfig, String> {
        let Some(mut v) = self.vision else {
            return Err(
                "embeddinggemma2: video input but this checkpoint's config declares no \
                 vision_config (the frames ride the Gemma-4 vision tower)"
                    .into(),
            );
        };
        let vf = &self.video_frontend;
        if vf.patch_size != v.patch_size || vf.pooling_kernel_size != v.pooling_kernel_size {
            return Err(format!(
                "processor_config video_processor patch_size/pooling_kernel_size \
                 ({}/{}) disagree with vision_config's ({}/{}): the tower's patch embedder \
                 and pooler are built for the latter, so frames cannot be resized against \
                 the former",
                vf.patch_size, vf.pooling_kernel_size, v.patch_size, v.pooling_kernel_size
            ));
        }
        v.max_soft_tokens = vf.max_soft_tokens;
        Ok(v)
    }

    /// Overlay `sentence_bert_config.json` / the folded HFQ sidecar (same key
    /// schema). Applied before [`Self::apply_st_config`], whose keys win.
    pub fn apply_sbert_config(&mut self, sb: &serde_json::Value) {
        self.apply_st_config(sb);
    }

    /// Overlay `config_sentence_transformers.json` (or the `sentence_transformers`
    /// object the quantizer folds into an `.hfq`): the two kinds' prompt
    /// prefixes and, when a release ships them, its own length limits.
    pub fn apply_st_config(&mut self, st: &serde_json::Value) {
        if let Some(prompts) = st.get("prompts").and_then(|v| v.as_object()) {
            if let Some(q) = prompts.get("query").and_then(|v| v.as_str()) {
                self.query_prompt = q.to_string();
            }
            if let Some(d) = prompts.get("document").and_then(|v| v.as_str()) {
                self.document_prompt = d.to_string();
            }
        }
        if let Some(q) = u64_of(st, "query_length") {
            self.query_length = q as usize;
        }
        if let Some(d) = u64_of(st, "document_length") {
            self.document_length = d as usize;
        }
    }

    pub fn kind_prompt(&self, kind: InputKind) -> &str {
        match kind {
            InputKind::Query => &self.query_prompt,
            InputKind::Document => &self.document_prompt,
        }
    }

    pub fn kind_length(&self, kind: InputKind) -> usize {
        match kind {
            InputKind::Query => self.query_length,
            InputKind::Document => self.document_length,
        }
    }

    /// Attention geometry of one layer.
    pub fn attn(&self, layer: usize) -> LayerAttn {
        self.layer_attn[layer]
    }

    /// Whether a sliding layer's window actually restricts anything at this
    /// length: |i-j| <= window is vacuous once n-1 <= window.
    pub fn window_bites(&self, layer: usize, n: usize) -> bool {
        self.layer_types[layer] == LayerType::Sliding && n > self.sliding_window + 1
    }

    pub fn rope_theta(&self, layer: usize) -> f32 {
        match self.layer_types[layer] {
            LayerType::Sliding => self.sliding_rope_theta,
            LayerType::Full => self.full_rope_theta,
        }
    }

    pub fn q_dim(&self, layer: usize) -> usize {
        self.n_heads * self.layer_attn[layer].head_dim
    }

    pub fn kv_dim(&self, layer: usize) -> usize {
        self.layer_attn[layer].n_kv_heads * self.layer_attn[layer].head_dim
    }

    /// The embedding scale applied to `embed_tokens` output: √hidden (HF's
    /// `TextScaledWordEmbedding`).
    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    /// PLE projection scale: hidden^-0.5 (HF's `per_layer_model_projection_scale`).
    pub fn ple_scale(&self) -> f32 {
        (self.hidden_size as f32).recip().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composite config.json of google/embeddinggemma-2, trimmed to the
    /// fields the parser reads (values verbatim).
    const EG2_CONFIG: &str = r#"{
      "architectures": ["EmbeddingGemma2Model"],
      "model_type": "embedding_gemma2",
      "text_config": {
        "attention_bias": false,
        "bos_token_id": 2,
        "embedding_dim": 768,
        "eos_token_id": 1,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 512,
        "hidden_size_per_layer_input": 512,
        "intermediate_size": 2048,
        "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "max_position_embeddings": 262144,
        "model_type": "embedding_gemma2_text",
        "num_attention_heads": 4,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "per_layer_config": {
          "05": {"head_dim": 512, "num_key_value_heads": 1},
          "11": {"head_dim": 512, "num_key_value_heads": 1},
          "17": {"head_dim": 512, "num_key_value_heads": 1},
          "23": {"head_dim": 512, "num_key_value_heads": 1}
        },
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "sliding_window": 512,
        "vocab_size": 262144
      },
      "audio_config": {
        "attention_chunk_size": 12,
        "attention_context_left": 13,
        "attention_context_right": 0,
        "attention_invalid_logits_value": -1000000000.0,
        "attention_logit_cap": 50.0,
        "conv_kernel_size": 5,
        "gradient_clipping": 10000000000.0,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "model_type": "gemma4_audio",
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "output_proj_dims": 1536,
        "residual_weight": 0.5,
        "rms_norm_eps": 1e-06,
        "subsampling_conv_channels": [128, 32],
        "use_clipped_linears": true
      },
      "audio_token_id": 258881,
      "boa_token_id": 256000,
      "eoa_token_index": 258883
    }"#;

    fn parse() -> EmbedConfig {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        parse_config(&v).unwrap()
    }

    #[test]
    fn parses_the_release_config() {
        let cfg = parse();
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.layer_types[5], LayerType::Full);
        assert_eq!(cfg.layer_types[23], LayerType::Full);
        assert_eq!(cfg.layer_types[0], LayerType::Sliding);
        // Per-layer overrides: full layers widen to hd 512 / 1 kv head, the
        // rest keep the 256/2 base.
        assert_eq!(
            cfg.attn(0),
            LayerAttn {
                head_dim: 256,
                n_kv_heads: 2
            }
        );
        assert_eq!(
            cfg.attn(5),
            LayerAttn {
                head_dim: 512,
                n_kv_heads: 1
            }
        );
        assert_eq!(
            cfg.attn(23),
            LayerAttn {
                head_dim: 512,
                n_kv_heads: 1
            }
        );
        assert_eq!(cfg.q_dim(0), 1024);
        assert_eq!(cfg.kv_dim(0), 512);
        assert_eq!(cfg.q_dim(5), 2048);
        assert_eq!(cfg.kv_dim(5), 512);
        assert_eq!(cfg.rope_theta(0), 10_000.0);
        assert_eq!(cfg.rope_theta(5), 1_000_000.0);
        assert_eq!(cfg.ple_dim, 512);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.bos_token_id, 2);
        assert_eq!(cfg.eos_token_id, 1);
        assert!((cfg.embed_scale() - (512.0f32).sqrt()).abs() < 1e-6);
        // Window is a radius: it only bites past window+1 rows.
        assert!(!cfg.window_bites(0, 513));
        assert!(cfg.window_bites(0, 514));
        assert!(!cfg.window_bites(5, 8192)); // full layer never masked
                                             // Sidecar defaults before overlays.
        assert_eq!(cfg.query_prompt, "task: search result | query: ");
        assert_eq!(cfg.document_prompt, "title: none | text: ");
        assert_eq!(cfg.query_length, 8192);
    }

    #[test]
    fn sidecar_overlay_applies() {
        let mut cfg = parse();
        let st: serde_json::Value = serde_json::from_str(
            r#"{"prompts":{"query":"task: clustering | query: ",
                "document":"title: t | text: "},
                "query_length":128,"document_length":512}"#,
        )
        .unwrap();
        cfg.apply_st_config(&st);
        assert_eq!(cfg.query_prompt, "task: clustering | query: ");
        assert_eq!(cfg.document_prompt, "title: t | text: ");
        assert_eq!(cfg.query_length, 128);
        assert_eq!(cfg.document_length, 512);
    }

    #[test]
    fn rejects_bad_layer_types_and_geometry() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let mut bad = v.clone();
        bad["text_config"]["layer_types"][0] = serde_json::json!("linear_attention");
        let err = parse_config(&bad).unwrap_err();
        assert!(err.contains("linear_attention"), "{err}");

        // head_dim must be an even multiple of 64 (the GEMM K tile + rope pairs).
        let mut bad2 = v.clone();
        bad2["text_config"]["per_layer_config"]["05"]["head_dim"] = serde_json::json!(200);
        let err = parse_config(&bad2).unwrap_err();
        assert!(err.contains("head_dim"), "{err}");

        // kv heads must divide the head count.
        let mut bad3 = v.clone();
        bad3["text_config"]["per_layer_config"]["05"]["num_key_value_heads"] = serde_json::json!(3);
        let err = parse_config(&bad3).unwrap_err();
        assert!(err.contains("divisible"), "{err}");

        // A per-layer override past the layer count is refused, not ignored.
        let mut bad4 = v.clone();
        bad4["text_config"]["per_layer_config"]["99"] = serde_json::json!({"head_dim": 512});
        let err = parse_config(&bad4).unwrap_err();
        assert!(err.contains("layer 99"), "{err}");

        // Non-tanh activation, attention bias, causal: all refused.
        let mut bad5 = v.clone();
        bad5["text_config"]["hidden_activation"] = serde_json::json!("silu");
        assert!(parse_config(&bad5)
            .unwrap_err()
            .contains("gelu_pytorch_tanh"));
        let mut bad6 = v.clone();
        bad6["text_config"]["attention_bias"] = serde_json::json!(true);
        assert!(parse_config(&bad6).unwrap_err().contains("attention_bias"));
        let mut bad7 = v.clone();
        bad7["text_config"]["is_causal"] = serde_json::json!(true);
        assert!(parse_config(&bad7).unwrap_err().contains("is_causal"));
    }

    #[test]
    fn parses_the_audio_tower_and_frontend() {
        let cfg = parse();
        let a = cfg.audio.expect("the release declares an audio tower");
        assert_eq!(a.hidden_size, 1024);
        assert_eq!(a.num_layers, 12);
        assert_eq!(a.n_heads, 8);
        assert_eq!(a.head_dim, 128);
        assert_eq!(a.q_dim(), 1024);
        assert_eq!(a.ffn_dim(), 4096);
        assert_eq!(a.lconv_dim(), 2048);
        assert_eq!(a.output_proj_dims, 1536);
        assert_eq!(a.conv_kernel_size, 5);
        assert_eq!(a.attention_chunk_size, 12);
        assert_eq!(a.attention_context_left, 13);
        assert_eq!(a.attention_context_right, 0);
        // chunk + past(13-1) + future(0) — the 24-wide block context.
        assert_eq!(a.context_size(), 24);
        assert_eq!(a.attention_logit_cap, 50.0);
        assert_eq!(a.attention_invalid_logits_value, -1e9);
        assert_eq!(a.residual_weight, 0.5);
        assert_eq!(a.subsampling_conv_channels, [128, 32]);
        assert!(a.use_clipped_linears);
        assert_eq!(a.gradient_clipping, 1e10);
        // Two stride-2 k3 p1 stages over 128 mel bins land on 32 frequencies,
        // which is the `/4` the projection width is built from.
        assert_eq!(a.conv_freq_out(128), 32);
        assert_eq!(a.subsample_input_dim(128), 1024);

        // The audio markers the text stream expands to.
        assert_eq!(cfg.audio_token_id, 258_881);
        assert_eq!(cfg.boa_token_id, 256_000);
        assert_eq!(cfg.eoa_token_id, 258_883); // from eoa_token_index

        // The frontend geometry is the extractor's default until a processor
        // config is overlaid.
        let fe = cfg.audio_frontend;
        assert_eq!(fe.sampling_rate, 16_000);
        assert_eq!(fe.frame_length, 320);
        assert_eq!(fe.hop_length, 160);
        assert_eq!(fe.fft_length, 512);
        assert_eq!(fe.fft_bins(), 257);
        assert_eq!(fe.mel_bins, 128);
        assert_eq!(fe.mel_floor, 1e-3);
        assert_eq!((fe.min_frequency, fe.max_frequency), (0.0, 8000.0));
    }

    #[test]
    fn processor_config_overrides_the_frontend_defaults() {
        // The checkpoint's own processor_config.json, trimmed.
        let pc: serde_json::Value = serde_json::from_str(
            r#"{"audio_ms_per_token":40,"audio_seq_length":280,
                "feature_extractor":{"dither":0.0,"feature_size":128,"fft_length":512,
                "fft_overdrive":false,"frame_length":320,"hop_length":160,
                "input_scale_factor":1.0,"max_frequency":8000.0,"mel_floor":0.001,
                "min_frequency":0.0,"padding_side":"right","padding_value":0.0,
                "per_bin_mean":null,"per_bin_stddev":null,"preemphasis":0.0,
                "preemphasis_htk_flavor":true,"return_attention_mask":true,
                "sampling_rate":16000}}"#,
        )
        .unwrap();
        let fe = AudioFrontendConfig::from_processor_json(&pc).unwrap();
        assert_eq!(fe, AudioFrontendConfig::default());

        // Every knob that would change the features is refused, not ignored.
        for (key, val) in [
            ("dither", serde_json::json!(0.0001)),
            ("preemphasis", serde_json::json!(0.97)),
            ("input_scale_factor", serde_json::json!(2.0)),
            ("fft_overdrive", serde_json::json!(true)),
            ("per_bin_mean", serde_json::json!([0.0])),
            ("padding_side", serde_json::json!("left")),
        ] {
            let mut bad = pc.clone();
            bad["feature_extractor"][key] = val;
            let err = AudioFrontendConfig::from_processor_json(&bad).unwrap_err();
            assert!(err.contains(key), "{key}: {err}");
        }

        // A geometry the math cannot express is refused too.
        let mut bad = pc.clone();
        bad["feature_extractor"]["fft_length"] = serde_json::json!(300);
        assert!(AudioFrontendConfig::from_processor_json(&bad)
            .unwrap_err()
            .contains("power of two"));
        let mut bad = pc.clone();
        bad["feature_extractor"]["max_frequency"] = serde_json::json!(9000.0);
        assert!(AudioFrontendConfig::from_processor_json(&bad)
            .unwrap_err()
            .contains("Nyquist"));
    }

    /// The checkpoint's `vision_config`, trimmed to what the frame path reads
    /// (the fixture above is a trunk-only config).
    fn tower() -> VisionConfig {
        VisionConfig {
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
        }
    }

    /// The checkpoint's `processor_config.json:video_processor` block sets the
    /// video frontend (140 soft tokens per FRAME, not the class default 70),
    /// and the frame geometry that follows from it is what the float32
    /// reference ran.
    #[test]
    fn processor_config_sets_the_video_frontend_and_the_frame_budget() {
        let pc: serde_json::Value = serde_json::from_str(
            r#"{"video_processor":{"add_timestamps":false,"do_convert_rgb":true,
                "do_normalize":false,"do_rescale":true,"do_resize":true,
                "do_sample_frames":true,"fps":1,"image_mean":[0.0,0.0,0.0],
                "image_std":[1.0,1.0,1.0],"max_frames":32,"max_soft_tokens":140,
                "overflow_strategy":"uniform","patch_size":16,"pooling_kernel_size":3,
                "resample":3,"rescale_factor":0.00392156862745098,"return_metadata":false,
                "video_processor_type":"EmbeddingGemma2VideoProcessor"}}"#,
        )
        .unwrap();
        let vf = VideoFrontendConfig::from_processor_json(&pc).unwrap();
        assert_eq!(vf.max_soft_tokens, 140);
        assert_eq!(vf.max_patches(), 1260);
        assert_eq!(vf.patch_size, 16);
        assert_eq!(vf.pooling_kernel_size, 3);
        assert_eq!(vf.max_frames, 32);
        assert!(vf.do_sample_frames);
        assert_eq!(vf.overflow, OverflowStrategy::Uniform);
        assert_eq!(vf.rescale_factor, 1.0 / 255.0);
        // No processor config at all (an `.hfq`): the defaults are this
        // checkpoint's, so a text-only-shaped metadata block still frames
        // correctly.
        assert_eq!(VideoFrontendConfig::default(), vf);

        // The frame geometry: the video budget, not the image config's 280.
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let mut cfg = parse_config(&v).unwrap();
        cfg.vision = Some(tower());
        assert_eq!(cfg.vision.unwrap().max_soft_tokens, 280);
        cfg.video_frontend = vf;
        let frame = cfg.frame_vision().unwrap();
        assert_eq!(frame.max_soft_tokens, 140);
        assert_eq!(frame.max_patches(), 1260);
        assert_eq!(frame.patch_size, cfg.vision.unwrap().patch_size);
    }

    /// Every knob that would change the frames is refused, not dropped.
    #[test]
    fn rejects_video_processor_knobs_that_change_the_frames() {
        let base: serde_json::Value = serde_json::from_str(
            r#"{"video_processor":{"max_soft_tokens":140,"patch_size":16,
                "pooling_kernel_size":3,"max_frames":32,"overflow_strategy":"uniform",
                "add_timestamps":false,"do_sample_frames":true,"rescale_factor":0.00392156862745098}}"#,
        )
        .unwrap();
        let ok = VideoFrontendConfig::from_processor_json(&base).unwrap();
        assert_eq!(ok.max_soft_tokens, 140);

        for (key, val) in [
            ("add_timestamps", serde_json::json!(true)),
            ("num_frames", serde_json::json!(32)),
            ("overflow_strategy", serde_json::json!("drop")),
            ("max_soft_tokens", serde_json::json!(100)),
            ("max_soft_tokens", serde_json::json!(0)),
            ("max_frames", serde_json::json!(0)),
            ("do_resize", serde_json::json!(false)),
            ("do_rescale", serde_json::json!(false)),
            ("do_convert_rgb", serde_json::json!(false)),
            ("resample", serde_json::json!(2)),
            ("rescale_factor", serde_json::json!(0.5)),
            ("image_mean", serde_json::json!([0.5, 0.5, 0.5])),
            ("image_std", serde_json::json!([2.0, 2.0, 2.0])),
            ("do_center_crop", serde_json::json!(true)),
            ("do_pad", serde_json::json!(true)),
            ("crop_size", serde_json::json!({"height": 48, "width": 48})),
        ] {
            let mut bad = base.clone();
            bad["video_processor"][key] = val;
            let err = VideoFrontendConfig::from_processor_json(&bad).unwrap_err();
            assert!(
                err.contains(key) || err.contains("max_soft_tokens") || err.contains("crop_size"),
                "{key}: {err}"
            );
        }

        // A supported alternative is accepted rather than refused: a
        // checkpoint that samples to 140 with `truncate`, or not at all.
        let mut alt = base.clone();
        alt["video_processor"]["overflow_strategy"] = serde_json::json!("truncate");
        alt["video_processor"]["max_soft_tokens"] = serde_json::json!(70);
        assert_eq!(
            VideoFrontendConfig::from_processor_json(&alt)
                .unwrap()
                .overflow,
            OverflowStrategy::Truncate
        );
        let mut off = base.clone();
        off["video_processor"]["do_sample_frames"] = serde_json::json!(false);
        assert!(
            !VideoFrontendConfig::from_processor_json(&off)
                .unwrap()
                .do_sample_frames
        );
    }

    /// A video processor whose patch geometry disagrees with the tower's own
    /// `vision_config` cannot be run: the patch embedder is built for the
    /// latter.
    #[test]
    fn a_frame_geometry_that_disagrees_with_the_tower_is_refused() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let mut cfg = parse_config(&v).unwrap();
        cfg.vision = Some(tower());
        cfg.video_frontend.pooling_kernel_size = 2;
        let err = cfg.frame_vision().unwrap_err();
        assert!(err.contains("disagree with vision_config"), "{err}");
        cfg.video_frontend = VideoFrontendConfig::default();
        cfg.vision = None;
        let err = cfg.frame_vision().unwrap_err();
        assert!(err.contains("no vision_config"), "{err}");
    }

    #[test]
    fn rejects_bad_audio_config() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let mut bad = v.clone();
        bad["audio_config"]["hidden_act"] = serde_json::json!("gelu");
        assert!(parse_config(&bad).unwrap_err().contains("silu"));
        let mut bad = v.clone();
        bad["audio_config"]["attention_logit_cap"] = serde_json::json!(0.0);
        assert!(parse_config(&bad).unwrap_err().contains("logit_cap"));
        // A non-negative "invalid" fill would give out-of-window keys weight.
        let mut bad = v.clone();
        bad["audio_config"]["attention_invalid_logits_value"] = serde_json::json!(0.0);
        assert!(parse_config(&bad)
            .unwrap_err()
            .contains("attention_invalid_logits_value"));
        let mut bad = v.clone();
        bad["audio_config"]["num_attention_heads"] = serde_json::json!(7);
        assert!(parse_config(&bad).unwrap_err().contains("divisible"));
        // A conv width that disagrees with the mel bank fails the cross-check
        // rather than much later inside the projection GEMM. (64 is a legal
        // GEMM width, so this reaches the cross-check: (64/4)*32 = 512 vs the
        // 1024 two stride-2 stages over 128 mel bins produce.)
        let mut bad = v.clone();
        bad["audio_config"]["subsampling_conv_channels"] = serde_json::json!([64, 32]);
        assert!(parse_config(&bad)
            .unwrap_err()
            .contains("subsample projection"));
        // A tower-less config is legal (text-only), just absent.
        let mut text_only = v.clone();
        text_only["audio_config"] = serde_json::Value::Null;
        let cfg = parse_config(&text_only).unwrap();
        assert!(cfg.audio.is_none());
        assert_eq!(cfg.audio_frontend, AudioFrontendConfig::default());
    }

    #[test]
    fn config_from_metadata_unwraps_both_shapes() {
        let v: serde_json::Value = serde_json::from_str(EG2_CONFIG).unwrap();
        let wrapped = serde_json::json!({"architecture": "embedding_gemma2", "config": v.clone()});
        assert_eq!(
            parse_config(config_from_metadata(&wrapped))
                .unwrap()
                .num_layers,
            24
        );
        assert_eq!(
            parse_config(config_from_metadata(&v)).unwrap().num_layers,
            24
        );
    }
}
