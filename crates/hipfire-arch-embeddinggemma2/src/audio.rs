// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Gemma-4 audio frontend: raw container bytes → the checkpoint's
//! log-mel features, plus the soft-token count `M` they expand to.
//!
//! Formula-level ground truth: `docs/plans/embeddinggemma2-tower-spec.md`
//! §AUDIO (transformers 5.19.0 `models/gemma4/feature_extraction_gemma4.py`,
//! `Gemma4AudioFeatureExtractor`). Ported exactly:
//!
//! 1. **WAV container → f32 PCM** ([`decode_wav`]), then right-zero-pad the
//!    clip to `pad_to_multiple_of = 128` samples (`padding="longest"` on a
//!    single clip). The pad is load-bearing: 0.5 s is 8 000 samples → 8 064,
//!    which is the difference between 50 mel frames and 49, and between
//!    M = 13 and M = 12.
//! 2. **Semicausal pad**: 160 zeros in front (`frame_length / 2`), so the
//!    first frame is centred on t = 0.
//! 3. **Frames**: unfold size 321 (frame + 1), step 160, then drop the last
//!    sample of each → 320 windowed samples. `nf = (L' + 160 − 321)/160 + 1`.
//! 4. **Window**: periodic Hann `0.5 − 0.5·cos(2πn/320)`.
//! 5. **Spectrum**: rfft n = 512 (the frame is right-zero-padded), then the
//!    **magnitude** (`|X|`, not `|X|²`) for the 257 bins.
//! 6. **Mel**: `magnitude @ filterbank`, 257 → 128, HTK spacing, `norm=None`,
//!    0–8000 Hz.
//! 7. **Log**: `log(mel + 1e-3)`.
//! 8. **Frame validity**: frame `i` is real iff its whole analysis window is
//!    (`mask[i*160 + 320]` on the padded signal); invalid frames are zeroed
//!    (`speech * mask[..., None]`), NOT dropped — they still occupy a row and
//!    still select a subsample phase.
//!
//! The soft-token count is the mask's two stride-2 subsample steps
//! ([`audio_token_count`]): the conv stack halves the time axis twice and
//! carries the mask through `mask[:, ::2]`, and the tokens HF writes into the
//! text stream are the mask's SURVIVORS (`audio_features[audio_mask]`), i.e.
//! `sum(mask₂)` — not `nf`, and not `ceil(nf/4)`. On this checkpoint the value
//! is uncapped: the processor's `audio_seq_length = 280` is not applied to the
//! waveform or to the count (15 s → 375, 30 s → 750), so the only upper bound
//! is the kind's sequence limit, enforced by the caller's room accounting.
//!
//! Numerics: the transform and the mel matmul run in f32 where the reference
//! is f64 (numpy). The relative error is ~1e-7 — three orders below the f16
//! weight-operand noise floor the tower itself accepts.
//!
//! The frequency layout is the reference's: `(height, width, channel)` after
//! the convs flattens to `f * C + c` with `f` the frequency and `c` the
//! channel, which is the order `input_proj_linear` [1024, 1024] expects.

use crate::config::AudioFrontendConfig;

/// One clip's decoded + planned shape, before the FFTs are paid.
///
/// Cheap to compute from the sample count alone, which is what lets the
/// caller refuse an over-long clip before transforming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioPlan {
    /// Real samples decoded from the container.
    pub samples: usize,
    /// Samples fed to the transform chain: `samples` padded up to the
    /// processor's 128-sample quantum.
    pub padded_samples: usize,
    /// Mel frames `nf` (including the invalid trailing one).
    pub frames: usize,
    /// Frames whose whole analysis window is real audio.
    pub valid_frames: usize,
    /// `M` — soft tokens the audio placeholder run expands to.
    pub n_tokens: usize,
}

/// `M`, HF's mask-subsample count: apply `t → (t−1)/2 + 1` to the frame
/// count twice, carrying the validity mask through `mask[:, ::2]` each time,
/// and count the survivors.
///
/// `mask` is a prefix (a frame is valid iff every sample it covers is, and
/// the invalid tail is exactly the padded quantum), so the halved mask is a
/// prefix too and its survivor count is `ceil(valid/2)` clipped to the new
/// length. Hand-checked against the float32 reference: 0.1 s → nf 10, 9
/// valid → M 3; 0.5 s → 50, 49 → 13; 1.0 s → 99, 99 → 25; 2.0 s → 199, 199 →
/// 50; 11.2 s → 1119, 1119 → 280; 15 s → 1499, 1499 → 375.
pub fn audio_token_count(frames: usize, valid_frames: usize) -> usize {
    /// One `Conv2d(k=3, s=2, p=1)` time step: `(t + 2 − 3)/2 + 1`.
    fn subsample_len(t: usize) -> usize {
        if t == 0 {
            0
        } else {
            (t - 1) / 2 + 1
        }
    }
    /// Survivors of `mask[::2][:len]` for a prefix-valid mask.
    fn halved(valid: usize, len: usize) -> usize {
        valid.div_ceil(2).min(len)
    }
    let t0 = subsample_len(frames);
    let v0 = halved(valid_frames, t0);
    halved(v0, subsample_len(t0))
}

/// One clip's processor output: what [`crate::audio_tower`] consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedAudio {
    /// `[frames, mel_bins]` row-major log-mel; the invalid trailing frames
    /// are exactly zero (the reference multiplies them by the frame mask).
    pub mel: Vec<f32>,
    /// The plan this was built from (frame counts, `M`).
    pub plan: AudioPlan,
}

impl ProcessedAudio {
    pub fn frames(&self) -> usize {
        self.plan.frames
    }

    /// `M` — the placeholder rows this clip fills.
    pub fn n_tokens(&self) -> usize {
        self.plan.n_tokens
    }
}

// ─── the container decoder ─────────────────────────────────────────────────

/// Decode a WAV container to mono f32 PCM at `want_rate`.
///
/// Accepted: `RIFF`/`WAVE`, `fmt ` formats 1 (integer PCM, 16-bit) and 3
/// (IEEE float, 32-bit), 1 or 2 channels (2 is averaged, which is what a
/// converter's `-ac 1` produces), at exactly the frontend's sample rate.
///
/// Everything else is refused BY NAME with the conversion to make, never
/// guessed at: other containers (MP3/OGG/FLAC/MP4), other widths, more
/// channels, and any other sample rate (we do not resample — a silent
/// resample would be a different model input, not a convenience).
pub fn decode_wav(bytes: &[u8], want_rate: usize) -> Result<Vec<f32>, String> {
    let conv = |what: &str| {
        format!(
            "embeddinggemma2: audio input is {what}; convert it to {want_rate} Hz mono \
             PCM16/f32 WAV (e.g. `ffmpeg -i in -ar {want_rate} -ac 1 -c:a pcm_s16le \
             out.wav`), or send a WAV directly"
        )
    };
    if bytes.len() < 12 {
        return Err(conv("shorter than a RIFF header"));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        // Name the container we can see, so the error says what to convert.
        let what = if bytes.starts_with(b"OggS") {
            "an OGG/Opus stream"
        } else if bytes.starts_with(b"fLaC") {
            "a FLAC stream"
        } else if bytes.starts_with(b"ID3")
            || (bytes.len() > 1 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
        {
            "an MP3 stream"
        } else if bytes.len() > 11 && &bytes[4..8] == b"ftyp" {
            "an MP4/M4A container"
        } else {
            "not a RIFF/WAVE container"
        };
        return Err(conv(what));
    }

    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let mut fmt: Option<(usize, usize, usize, usize)> = None; // (tag, channels, rate, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if body > bytes.len() {
            break;
        }
        let end = body.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " if size >= 16 => {
                let mut tag = u16_at(body) as usize;
                let channels = u16_at(body + 2) as usize;
                let rate = u32_at(body + 4) as usize;
                let bits = u16_at(body + 14) as usize;
                // WAVE_FORMAT_EXTENSIBLE: the real tag is the first two bytes
                // of the SubFormat GUID, 24 bytes into the chunk.
                if tag == 0xFFFE && size >= 26 {
                    tag = u16_at(body + 24) as usize;
                }
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => data = Some((body, end - body)),
            _ => {}
        }
        // Chunks are word-aligned.
        pos = body + size + (size & 1);
    }

    let (tag, channels, rate, bits) = fmt.ok_or_else(|| conv("a WAV with no `fmt ` chunk"))?;
    let (off, len) = data.ok_or_else(|| conv("a WAV with no `data` chunk"))?;
    if rate != want_rate {
        return Err(format!(
            "embeddinggemma2: audio is {rate} Hz but this checkpoint's frontend runs at \
             {want_rate} Hz — resample it (`ffmpeg -i in -ar {want_rate} -ac 1 out.wav`); \
             a silent resample would be a different model input"
        ));
    }
    if channels == 0 || channels > 2 {
        return Err(conv(&format!("{channels}-channel WAV")));
    }
    let width = match (tag, bits) {
        (1, 16) => 2,
        (3, 32) => 4,
        (1, b) => return Err(conv(&format!("{b}-bit integer PCM WAV"))),
        (3, b) => return Err(conv(&format!("{b}-bit float WAV"))),
        (t, _) => return Err(conv(&format!("WAV format tag {t}"))),
    };
    let frame_bytes = width * channels;
    let frames = len / frame_bytes;
    if frames == 0 {
        return Err("embeddinggemma2: audio input decodes to zero samples".into());
    }
    let mut pcm = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = off + f * frame_bytes;
        let mut acc = 0.0f32;
        for c in 0..channels {
            let o = base + c * width;
            acc += match tag {
                1 => f32::from(i16::from_le_bytes([bytes[o], bytes[o + 1]])) / 32768.0,
                _ => f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
            };
        }
        pcm.push(acc / channels as f32);
    }
    Ok(pcm)
}

// ─── the frontend ──────────────────────────────────────────────────────────

/// A clip's decoded samples plus its plan — everything the transform chain
/// needs, with the container work already done and no FFT paid yet.
#[derive(Clone, Debug)]
pub struct AudioInput {
    pub pcm: Vec<f32>,
    pub plan: AudioPlan,
}

/// Decode the container and compute the plan. The mel transform is
/// deliberately NOT run here: the caller can refuse an over-long clip on the
/// plan alone (a 10-minute WAV would otherwise pay ~2 s of FFTs to be
/// rejected).
pub fn decode_and_plan(bytes: &[u8], fe: &AudioFrontendConfig) -> Result<AudioInput, String> {
    fe.validate()?;
    let pcm = decode_wav(bytes, fe.sampling_rate)?;
    Ok(AudioInput {
        plan: plan(fe, pcm.len()),
        pcm,
    })
}

/// The frame/token accounting for a clip of `samples` real samples.
pub fn plan(fe: &AudioFrontendConfig, samples: usize) -> AudioPlan {
    let padded_samples = fe.padded_samples(samples);
    let frames = fe.frames(padded_samples);
    let valid_frames = fe.valid_frames(samples);
    AudioPlan {
        samples,
        padded_samples,
        frames,
        valid_frames,
        n_tokens: audio_token_count(frames, valid_frames),
    }
}

/// Run the transform chain: `[frames, mel_bins]` log-mel, invalid frames
/// zeroed.
pub fn mels(fe: &AudioFrontendConfig, input: &AudioInput) -> ProcessedAudio {
    let plan = input.plan;
    let mut mel = vec![0.0f32; plan.frames * fe.mel_bins];
    if plan.frames > 0 {
        let mut front = Frontend::new(fe);
        let pad_left = fe.frame_length / 2;
        for i in 0..plan.frames {
            let start = i * fe.hop_length;
            front.windowed_frame(&input.pcm, start, pad_left);
            let row = &mut mel[i * fe.mel_bins..(i + 1) * fe.mel_bins];
            front.mel_into(row);
            if i >= plan.valid_frames {
                // `speech * mask[..., None]`: an invalid frame is a real ZERO
                // row, not an absent one — it still drives the subsample.
                row.fill(0.0);
            }
        }
    }
    ProcessedAudio { mel, plan }
}

/// [`decode_and_plan`] + [`mels`]: the whole frontend for one clip.
pub fn process_audio(bytes: &[u8], fe: &AudioFrontendConfig) -> Result<ProcessedAudio, String> {
    let input = decode_and_plan(bytes, fe)?;
    Ok(mels(fe, &input))
}

/// The window/twiddle/mel-bank tables, built once per clip and reused for
/// every frame.
struct Frontend<'a> {
    cfg: &'a AudioFrontendConfig,
    hann: Vec<f32>,
    /// `[fft_bins, mel_bins]` row-major triangular filters.
    bank: Vec<f32>,
    bit_rev: Vec<usize>,
    /// `(cos, −sin)` of `−2πk/N`, interleaved: `[fft_length]`.
    twiddle: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
}

impl<'a> Frontend<'a> {
    fn new(cfg: &'a AudioFrontendConfig) -> Self {
        let n = cfg.fft_length;
        // Periodic Hann, matching `window_function(320, "hann", periodic=True)`
        // (`np.hanning(321)[:-1]`): 0.5 − 0.5·cos(2πn/320).
        let hann: Vec<f32> = (0..cfg.frame_length)
            .map(|i| {
                let x = 2.0 * std::f64::consts::PI * i as f64 / cfg.frame_length as f64;
                (0.5 - 0.5 * x.cos()) as f32
            })
            .collect();

        let bank = mel_filter_bank(cfg);

        let bits = n.trailing_zeros() as usize;
        let bit_rev = (0..n)
            .map(|i| {
                let mut r = 0usize;
                for b in 0..bits {
                    r |= ((i >> b) & 1) << (bits - 1 - b);
                }
                r
            })
            .collect();
        let mut twiddle = vec![0.0f32; n];
        for k in 0..n / 2 {
            let ang = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            twiddle[2 * k] = ang.cos() as f32;
            twiddle[2 * k + 1] = ang.sin() as f32;
        }

        Self {
            cfg,
            hann,
            bank,
            bit_rev,
            twiddle,
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }

    /// Fill the FFT input with frame `i`: the semicausally padded,
    /// windowed `frame_length` samples, right-zero-padded to `fft_length`.
    ///
    /// `pcm` is the DECODED clip, not the `pad_to_multiple_of`-padded one: a
    /// frame's window can run past its end (and past `pcm.len()`), because the
    /// pads on both sides are zeros — the semicausal one in front and the
    /// quantum pad behind (`padding_value = 0.0`). Reading them as zeros here
    /// is what the reference sees; bounding the read by the padded length
    /// instead would index the pad region out of `pcm` (a panic on any clip
    /// whose last frame overruns, e.g. 1 409 samples).
    fn windowed_frame(&mut self, pcm: &[f32], start: usize, pad_left: usize) {
        self.re.fill(0.0);
        self.im.fill(0.0);
        for i in 0..self.cfg.frame_length {
            // Unfold index `start + i` into the left-padded signal.
            let src = start as isize + i as isize - pad_left as isize;
            let v = if src >= 0 && (src as usize) < pcm.len() {
                pcm[src as usize]
            } else {
                0.0
            };
            self.re[i] = v * self.hann[i];
        }
    }

    /// In-place radix-2 FFT of `re`/`im`.
    fn fft(&mut self) {
        let n = self.cfg.fft_length;
        for i in 0..n {
            let j = self.bit_rev[i];
            if i < j {
                self.re.swap(i, j);
                self.im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let tw = 2 * k * step;
                    let (wr, wi) = (self.twiddle[tw], self.twiddle[tw + 1]);
                    let (i, j) = (start + k, start + k + half);
                    let (xr, xi) = (self.re[j], self.im[j]);
                    let (tr, ti) = (xr * wr - xi * wi, xr * wi + xi * wr);
                    self.re[j] = self.re[i] - tr;
                    self.im[j] = self.im[i] - ti;
                    self.re[i] += tr;
                    self.im[i] += ti;
                }
            }
            len *= 2;
        }
    }

    /// `log(magnitude @ bank + floor)` for the frame in `re`/`im`.
    fn mel_into(&mut self, out: &mut [f32]) {
        self.fft();
        let bins = self.cfg.fft_bins();
        let mel_bins = self.cfg.mel_bins;
        for (j, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..bins {
                let m = (self.re[k] * self.re[k] + self.im[k] * self.im[k]).sqrt();
                acc += m * self.bank[k * mel_bins + j];
            }
            *o = (acc + self.cfg.mel_floor).ln();
        }
    }
}

/// HF `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")`
/// as `[fft_bins, mel_bins]`: `mel_freqs = linspace(hertz_to_mel(0),
/// hertz_to_mel(8000), 130)`, back to Hz, then triangular filters over
/// `linspace(0, 8000, 257)`. Built in f64 (the reference builds it in numpy
/// f64) and stored f32.
fn mel_filter_bank(cfg: &AudioFrontendConfig) -> Vec<f32> {
    let bins = cfg.fft_bins();
    let mel_bins = cfg.mel_bins;
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(cfg.min_frequency);
    let mel_max = hz_to_mel(cfg.max_frequency);
    let filter_freqs: Vec<f64> = (0..mel_bins + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (mel_bins + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|k| (cfg.sampling_rate as f64 / 2.0) * k as f64 / (bins - 1) as f64)
        .collect();

    let mut bank = vec![0.0f32; bins * mel_bins];
    for k in 0..bins {
        for j in 0..mel_bins {
            let down = -(filter_freqs[j] - fft_freqs[k]) / (filter_freqs[j + 1] - filter_freqs[j]);
            let up =
                (filter_freqs[j + 2] - fft_freqs[k]) / (filter_freqs[j + 2] - filter_freqs[j + 1]);
            bank[k * mel_bins + j] = down.min(up).max(0.0) as f32;
        }
    }
    bank
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AUDIO_PAD_TO_MULTIPLE_OF;

    fn fe() -> AudioFrontendConfig {
        AudioFrontendConfig::default()
    }

    /// A canonical 16 kHz mono WAV around `pcm` (format 1, 16-bit).
    fn wav16(pcm: &[f32], rate: u32, channels: u16) -> Vec<u8> {
        let width = 2usize;
        let frame_bytes = width * channels as usize;
        let data_len = pcm.len() * width; // `pcm` is already interleaved here
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&((rate as usize * frame_bytes) as u32).to_le_bytes());
        out.extend_from_slice(&(frame_bytes as u16).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for v in pcm {
            let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&q.to_le_bytes());
        }
        out
    }

    fn wav_f32(pcm: &[f32], rate: u32) -> Vec<u8> {
        let data_len = pcm.len() * 4;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 4).to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for v in pcm {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// The six counts the float32 reference run produced, with the frame
    /// arithmetic spelled out so a reader can re-derive each one:
    ///
    /// ```text
    /// L' = ceil(L/128)·128          nf = (L' + 160 − 321)/160 + 1
    /// M  = ((nf−1)/2+1) then ((t−1)/2+1), carrying the valid mask
    /// ```
    #[test]
    fn plan_matches_the_reference_token_counts() {
        let fe = fe();
        for (ms, nf, valid, m) in [
            // 0.1 s: 1 600 → 1 664; nf 10; frame 9 ends at 1760 ≥ 160+1600.
            (100u64, 10usize, 9usize, 3usize),
            // 0.5 s: 8 000 → 8 064 (the pad is load-bearing: without it nf 49
            // and M would be 12).
            (500, 50, 49, 13),
            // 1.0 s: 16 000 (already a multiple of 128); all 99 frames real.
            (1000, 99, 99, 25),
            (2000, 199, 199, 50),
            // 11.2 s: 179 200, all real → t0 560 → t1 280.
            (11_200, 1119, 1119, 280),
            // 15 s: 240 000, all real → t0 750 → t1 375. NO 280 cap.
            (15_000, 1499, 1499, 375),
            // 30 s: 480 000 → t0 1 500 → t1 750 (still uncapped).
            (30_000, 2999, 2999, 750),
        ] {
            let samples = (ms as usize) * (fe.sampling_rate / 1000);
            let p = plan(&fe, samples);
            assert_eq!(p.samples, samples, "{ms} ms samples");
            assert_eq!(p.frames, nf, "{ms} ms frames");
            assert_eq!(p.valid_frames, valid, "{ms} ms valid frames");
            assert_eq!(p.n_tokens, m, "{ms} ms tokens");
            assert_eq!(p.padded_samples % AUDIO_PAD_TO_MULTIPLE_OF, 0);
            assert!(
                p.padded_samples >= samples
                    && p.padded_samples < samples + AUDIO_PAD_TO_MULTIPLE_OF
            );
        }
    }

    #[test]
    fn token_count_halves_the_mask_twice() {
        // A full mask: the count is the pure subsample chain.
        assert_eq!(audio_token_count(10, 10), 3);
        assert_eq!(audio_token_count(50, 50), 13);
        // The halving carries validity: the last valid frame can fall out.
        assert_eq!(audio_token_count(11, 10), 3);
        assert_eq!(audio_token_count(2, 1), 1);
        assert_eq!(audio_token_count(1, 1), 1);
        assert_eq!(audio_token_count(0, 0), 0);
        // A clip too short for a single frame has no tokens at all.
        assert_eq!(audio_token_count(0, 0), 0);
    }

    /// A frame's analysis window runs past the decoded clip into the
    /// `pad_to_multiple_of` region and (at the front) into the semicausal pad;
    /// BOTH read as zeros. 1 409 samples is the minimal repro: its last frame
    /// (index 8, window 1 120…1 439) overruns the clip, and the pad quantum is
    /// 1 536, so a bound of "the padded length" indexes 1 409…1 535 out of
    /// `pcm` — a panic, not a wrong number.
    #[test]
    fn frames_read_the_pad_region_as_zeros() {
        let fe = fe();
        let pcm: Vec<f32> = (0..1409).map(|i| (i as f32 * 0.017).sin() * 0.4).collect();
        let p = plan(&fe, pcm.len());
        assert_eq!((p.frames, p.valid_frames, p.n_tokens), (9, 8, 2));
        let short = mels(
            &fe,
            &AudioInput {
                pcm: pcm.clone(),
                plan: p,
            },
        );
        // Spelling the same clip's pad out (1 536 = 12 · 128 is a whole
        // quantum, so nothing is added) must give the same first 8 frames —
        // they are the frames the short clip can vouch for.
        let mut long = pcm;
        long.resize(1536, 0.0);
        let pl = plan(&fe, long.len());
        assert_eq!(pl.padded_samples, long.len());
        let padded = mels(
            &fe,
            &AudioInput {
                pcm: long,
                plan: pl,
            },
        );
        assert_eq!(short.mel.len(), padded.mel.len());
        assert_eq!(short.mel[..8 * fe.mel_bins], padded.mel[..8 * fe.mel_bins]);
        // The short clip's ninth frame is INVALID — its window leaves the clip
        // — so it is the reference's zero row; the padded clip's ninth is real.
        assert!(short.mel[8 * fe.mel_bins..].iter().all(|v| *v == 0.0));
        assert!(padded.mel[8 * fe.mel_bins..].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn wav_decode_accepts_pcm16_float32_mono_and_stereo() {
        let pcm: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let mono = decode_wav(&wav16(&pcm, 16_000, 1), 16_000).unwrap();
        assert_eq!(mono.len(), 1000);
        for (a, b) in mono.iter().zip(&pcm) {
            assert!((a - b).abs() < 1e-4, "pcm16 round trip: {a} vs {b}");
        }

        let f = decode_wav(&wav_f32(&pcm, 16_000), 16_000).unwrap();
        assert_eq!(f, pcm, "float32 is carried exactly");

        // Stereo interleaved L=+0.5, R=-0.5 → the channel mean is 0.
        let mut stereo = Vec::new();
        for _ in 0..100 {
            stereo.push(0.5f32);
            stereo.push(-0.5f32);
        }
        let s = decode_wav(&wav16(&stereo, 16_000, 2), 16_000).unwrap();
        assert_eq!(s.len(), 100);
        assert!(s.iter().all(|v| v.abs() < 1e-4), "stereo downmix: {s:?}");
    }

    #[test]
    fn wav_decode_refuses_by_name() {
        let pcm = vec![0.0f32; 64];
        // Wrong rate.
        let err = decode_wav(&wav16(&pcm, 44_100, 1), 16_000).unwrap_err();
        assert!(err.contains("44100 Hz"), "{err}");
        assert!(err.contains("resample"), "{err}");
        // Other containers, named.
        assert!(decode_wav(b"OggS\0\0\0\0\0\0\0\0", 16_000)
            .unwrap_err()
            .contains("OGG"));
        assert!(decode_wav(b"ID3\x04\0\0\0\0\0\0\0\0", 16_000)
            .unwrap_err()
            .contains("MP3"));
        assert!(decode_wav(b"\0\0\0\x20ftypM4A ", 16_000)
            .unwrap_err()
            .contains("MP4"));
        assert!(decode_wav(&[0u8; 4], 16_000).unwrap_err().contains("RIFF"));
        // 24-bit PCM is refused rather than mis-decoded.
        let mut w = wav16(&pcm, 16_000, 1);
        w[34] = 24; // bits per sample
        assert!(decode_wav(&w, 16_000).unwrap_err().contains("24-bit"));
        // A WAV whose `data` chunk is empty decodes to nothing.
        let w = wav16(&[], 16_000, 1);
        assert!(decode_wav(&w, 16_000).unwrap_err().contains("zero samples"));
    }

    #[test]
    fn fft_matches_a_naive_dft() {
        let fe = fe();
        let mut front = Frontend::new(&fe);
        // A deterministic, non-symmetric frame: the 512-point transform of
        // the windowed 320 samples must equal the O(n²) DFT.
        let x: Vec<f32> = (0..fe.frame_length)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        front.re[..fe.frame_length].copy_from_slice(&x);
        front.im.fill(0.0);
        front.fft();
        let n = fe.fft_length;
        for k in [0usize, 1, 7, 128, 255, 256] {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &v) in x.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                re += v as f64 * ang.cos();
                im += v as f64 * ang.sin();
            }
            assert!(
                (front.re[k] as f64 - re).abs() < 2e-2,
                "bin {k}: {} vs {re}",
                front.re[k]
            );
            assert!((front.im[k] as f64 - im).abs() < 2e-2, "bin {k} imag");
        }
    }

    #[test]
    fn hann_is_periodic_and_the_bank_is_triangular() {
        let fe = fe();
        let front = Frontend::new(&fe);
        // Periodic Hann: w[0] = 0, w[160] = 1, w[n] = w[320-n].
        assert!(front.hann[0].abs() < 1e-6);
        assert!((front.hann[160] - 1.0).abs() < 1e-6);
        for n in 1..160 {
            assert!((front.hann[n] - front.hann[320 - n]).abs() < 1e-6, "n={n}");
        }
        assert!(front.hann.iter().all(|v| (0.0..=1.0).contains(v)));

        // The bank is 257 x 128, every filter is nonzero somewhere and none
        // exceeds 1 (a triangular filter's peak).
        assert_eq!(front.bank.len(), fe.fft_bins() * fe.mel_bins);
        for j in 0..fe.mel_bins {
            let col: Vec<f32> = (0..fe.fft_bins())
                .map(|k| front.bank[k * fe.mel_bins + j])
                .collect();
            let peak = col.iter().cloned().fold(0.0f32, f32::max);
            assert!(peak <= 1.0 + 1e-6, "filter {j} peak {peak}");
        }
        // Bin 0 is 0 Hz — the first filter's LEFT EDGE — so that weight is
        // exactly 0, and bin 1 (31.25 Hz) is already past its right edge
        // (mel_to_hz(2·step) ≈ 27.9 Hz): with 257 bins over 128 HTK filters
        // the lowest filter catches no bin at all. That is exactly the
        // all-zero-filter case HF's extractor suppresses a warning for, so
        // pin it rather than pretend otherwise.
        let first: Vec<f32> = (0..fe.fft_bins())
            .map(|k| front.bank[k * fe.mel_bins])
            .collect();
        assert!(first.iter().all(|v| *v == 0.0), "{first:?}");
        let empty = (0..fe.mel_bins)
            .filter(|j| (0..fe.fft_bins()).all(|k| front.bank[k * fe.mel_bins + j] == 0.0))
            .count();
        assert!(empty <= 8, "{empty} all-zero mel filters");
        // A middle filter does catch bins, and its weights form a triangle:
        // contiguous, unimodal, peaking at 1 minus at most half a bin's worth
        // of misalignment (bin pitch 31.25 Hz against ~47 Hz half-widths).
        let mid = fe.mel_bins / 2;
        let col: Vec<f32> = (0..fe.fft_bins())
            .map(|k| front.bank[k * fe.mel_bins + mid])
            .collect();
        let nz: Vec<usize> = (0..col.len()).filter(|k| col[*k] > 0.0).collect();
        assert!(!nz.is_empty(), "mid filter catches no bins");
        assert!(
            nz.windows(2).all(|w| w[1] == w[0] + 1),
            "nonzero weights are not contiguous: {nz:?}"
        );
        let apex = nz
            .iter()
            .copied()
            .max_by(|a, b| col[*a].partial_cmp(&col[*b]).unwrap())
            .unwrap();
        assert!(
            col[..apex].windows(2).all(|w| w[1] >= w[0]),
            "filter does not rise to its apex"
        );
        assert!(
            col[apex..].windows(2).all(|w| w[1] <= w[0]),
            "filter does not fall from its apex"
        );
        assert!(
            (col[apex] - 1.0).abs() < 0.35,
            "apex {} is not near 1",
            col[apex]
        );
        // No filter crosses 8000 Hz: the last bin is the top edge.
        let last = (fe.fft_bins() - 1) * fe.mel_bins;
        assert!(front.bank[last + fe.mel_bins - 1].abs() < 1e-6);
    }

    #[test]
    fn a_pure_tone_lands_in_the_right_mel_filter() {
        let fe = fe();
        let tone = 1000.0f32; // Hz
        let n = 16_000 / 2; // 0.5 s
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * tone * i as f32 / 16_000.0).sin())
            .collect();
        let p = process_audio(&wav_f32(&pcm, 16_000), &fe).unwrap();
        assert_eq!(p.plan.frames, 50);
        assert_eq!(p.plan.n_tokens, 13);
        assert_eq!(p.mel.len(), 50 * 128);
        // A steady middle frame: its peak filter must contain 1 000 Hz.
        let row = &p.mel[25 * 128..26 * 128];
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // HTK filters span 0..8000 Hz; filter j's center is at
        // mel_to_hz(linspace(mel(0), mel(8000), 130)[j+1]). 1 kHz is well up
        // the scale, so a sane peak is nonzero and short of the top.
        assert!(peak > 4 && peak < 90, "peak filter {peak}");
        // The invalid trailing frame (49) is exactly zero.
        assert!(p.mel[49 * 128..50 * 128].iter().all(|v| *v == 0.0));
        // A real frame is finite and not all-zero.
        assert!(row.iter().all(|v| v.is_finite()));
        assert!(row.iter().any(|v| *v != 0.0));
        // log(mel + 1e-3) of a zero frame would be -6.9, so a silent frame is
        // distinguishable from an invalid one.
        let silent = process_audio(&wav_f32(&vec![0.0; n], 16_000), &fe).unwrap();
        assert!((silent.mel[25 * 128] + 6.907755).abs() < 1e-3);
    }

    #[test]
    fn the_mel_matches_the_hf_extractor_cell_for_cell() {
        // Reference values from the REAL extractor (transformers 5.17
        // `Gemma4AudioFeatureExtractor`, the checkpoint's own parameters):
        //
        //   x = [((i % 41) - 20) / 40 for i in 0..1600]   # 0.1 s @ 16 kHz
        //   fe = Gemma4AudioFeatureExtractor(feature_size=128, sampling_rate=16000,
        //        frame_length_ms=20, hop_length_ms=10, min_frequency=0,
        //        max_frequency=8000, preemphasis=0, dither=0,
        //        input_scale_factor=1, mel_floor=1e-3)
        //   mel = fe(x[None, :], return_tensors="np")["input_features"][0]
        //
        // The ramp is integer arithmetic in both languages, so the signal
        // round-trips a float32 WAV bit-exactly and any difference is ours.
        // The extractor runs in numpy f64; the tolerance is the f32 gap.
        let fe = fe();
        let pcm: Vec<f32> = (0..1600).map(|i| ((i % 41) as f32 - 20.0) / 40.0).collect();
        let p = process_audio(&wav_f32(&pcm, 16_000), &fe).unwrap();
        assert_eq!(
            (p.plan.frames, p.plan.valid_frames, p.plan.n_tokens),
            (10, 9, 3)
        );
        let row = |f: usize| &p.mel[f * 128..(f + 1) * 128];
        let close = |got: f32, want: f64, what: &str| {
            assert!(
                (got as f64 - want).abs() < 2e-5,
                "{what}: {got} vs {want} ({})",
                got as f64 - want
            );
        };
        // The first mel filter catches no FFT bin at all (its triangle spans
        // 0..27.9 Hz while the bins step 31.25 Hz), so its value is exactly
        // the floor: log(1e-3). The reference answers the same.
        close(row(0)[0], -6.907755375e+00, "mel[0][0]");
        close(row(1)[0], -6.907755375e+00, "mel[1][0]");
        for (i, want) in [
            9.940393567e-01,
            -1.910234243e-01,
            8.030541539e-01,
            2.952116132e-01,
            7.414117455e-01,
        ]
        .iter()
        .enumerate()
        {
            close(row(0)[1 + i], *want, "mel[0][1..]");
        }
        for (i, want) in [
            -9.057840705e-01,
            4.068132937e-01,
            1.085303426e+00,
            1.471474051e+00,
            1.274184108e+00,
            4.954383075e-01,
        ]
        .iter()
        .enumerate()
        {
            close(row(0)[64 + i], *want, "mel[0][64..]");
        }
        for (i, want) in [-6.907755375e+00, -3.990302563e+00, -5.060257435e+00]
            .iter()
            .enumerate()
        {
            close(row(1)[i], *want, "mel[1][0..]");
        }
        for (i, want) in [
            -2.445197582e+00,
            -2.723629951e+00,
            -2.188641071e+00,
            -1.241876960e+00,
            -9.719585180e-01,
            1.292459607e+00,
        ]
        .iter()
        .enumerate()
        {
            close(row(3)[30 + i], *want, "mel[3][30..]");
        }
        for (i, want) in [
            6.610319018e-01,
            8.066686243e-02,
            1.649600983e+00,
            -7.260666490e-01,
            1.313844800e+00,
            1.062931061e+00,
            3.078370094e-01,
            1.613431811e+00,
        ]
        .iter()
        .enumerate()
        {
            close(row(8)[120 + i], *want, "mel[8][120..]");
        }
        // The frame past the mask is exactly zero, not log(floor).
        assert!(row(9).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_clip_too_short_for_one_frame_has_no_tokens() {
        let fe = fe();
        // 4 samples: the padded 128 + 160 is shorter than the 321 window.
        let p = process_audio(&wav_f32(&[0.0; 4], 16_000), &fe).unwrap();
        assert_eq!(p.plan.frames, 0);
        assert_eq!(p.n_tokens(), 0);
        assert!(p.mel.is_empty());
    }
}
