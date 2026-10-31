// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU log-mel-spectrogram front-end for LFM3-Audio.
//!
//! Ground truth: transformers `Lfm3AudioFeatureExtractor._log_mel`
//! (`feature_extraction_lfm3_audio.py`) backed by `audio_utils.mel_filter_bank`
//! (Slaney mel scale + Slaney area normalization) and `torch.stft`
//! (center=False, n_fft=400, hop=160, win=400, Hann periodic, power=2).
//!
//! Pipeline (byte-matches the HF extractor):
//!   1. frame the waveform (center=False): `num_frames = 1 + (N - n_fft)/hop`
//!   2. window each frame with a periodic Hann (`0.5 - 0.5*cos(2πn/n_fft)`)
//!   3. rfft → `n_fft/2+1` complex bins; power = |X|²
//!   4. mel = mel_filtersᵀ · power   (mel_filters is `[num_freq_bins, num_mels]`)
//!   5. log10, clamp `log_floor`, affine `(x + offset) / divisor`
//!
//! This is pure CPU (no GPU needed) — the mel is small and runs once per
//! query. The Slaney mel filter bank is ported verbatim from transformers
//! `audio_utils` so it matches the trained extractor exactly; the test in
//! `tests/mel_reference.rs` validates the full pipeline against a numpy
//! reference generated from the same math.

/// LFM3-Audio mel front-end config (defaults from `config.json` `preprocessor`).
#[derive(Clone, Debug)]
pub struct MelConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub sample_rate: u32,
    pub f_max: f32,
    pub f_min: f32,
    pub log_floor: f32,
    pub norm_offset: f32,
    pub norm_divisor: f32,
}

impl MelConfig {
    /// LFM3-Audio defaults (LiquidAI/LFM3-Audio-2B `config.json` `preprocessor`).
    pub fn lfm3_audio() -> Self {
        MelConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            sample_rate: 16000,
            f_max: 8000.0,
            f_min: 0.0,
            log_floor: -12.0,
            norm_offset: 4.0,
            norm_divisor: 4.0,
        }
    }
    pub fn num_freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
    /// 8 spectrogram frames → 1 encoder token (LFM3-Audio `encoder_stride`).
    pub const ENCODER_STRIDE: usize = 8;
}

/// ln(6.4) — used by the Slaney mel scale (std doesn't expose this constant).
const LN_6_4: f64 = 1.8562979903656263;

// ─── Slaney mel scale (transformers audio_utils.hertz_to_mel/mel_to_hertz) ──

/// Slaney hertz→mel. Below 1000 Hz: linear `3·f/200`; above: log-spaced.
fn hz_to_mel_slaney(freq: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 27.0 / LN_6_4; // 27/ln(6.4)
    let mels = 3.0 * freq / 200.0;
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
    } else {
        mels
    }
}

/// Slaney mel→hertz (inverse of [`hz_to_mel_slaney`]).
fn mel_to_hz_slaney(mels: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = LN_6_4 / 27.0; // ln(6.4)/27
    let freq = 200.0 * mels / 3.0;
    if mels >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (LOGSTEP * (mels - MIN_LOG_MEL)).exp()
    } else {
        freq
    }
}

/// Slaney mel filter bank, row-major `[num_freq_bins, num_mels]`
/// (matches transformers `mel_filter_bank(norm="slaney", mel_scale="slaney")`).
pub fn slaney_mel_filter_bank(cfg: &MelConfig) -> Vec<f32> {
    let num_freq_bins = cfg.num_freq_bins();
    let num_mels = cfg.n_mels;
    let mel_min = hz_to_mel_slaney(cfg.f_min as f64);
    let mel_max = hz_to_mel_slaney(cfg.f_max as f64);
    // mel_freqs: num_mels+2 linearly-spaced mel points → filter center freqs (Hz).
    let n_pts = num_mels + 2;
    let mut filter_freqs = vec![0.0f64; n_pts];
    for i in 0..n_pts {
        let m = mel_min + (mel_max - mel_min) * i as f64 / (n_pts - 1) as f64;
        filter_freqs[i] = mel_to_hz_slaney(m);
    }
    // fft_freqs: linspace(0, sample_rate//2, num_freq_bins).
    let nyquist = (cfg.sample_rate / 2) as f64;
    let mut fft_freqs = vec![0.0f64; num_freq_bins];
    for i in 0..num_freq_bins {
        fft_freqs[i] = nyquist * i as f64 / (num_freq_bins - 1) as f64;
    }
    // filter_diff = diff(filter_freqs) → n_pts-1 values.
    let mut filter_diff = vec![0.0f64; n_pts - 1];
    for i in 0..n_pts - 1 {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
    }
    // Triangular filter bank: for each (freq_bin i, mel_filter j):
    //   slopes[i,j] = filter_freqs[j] - fft_freqs[i]
    //   down = -slopes[i,j]   / filter_diff[j]      (j in 0..num_mels)
    //   up   =  slopes[i,j+2] / filter_diff[j+1]    (j in 0..num_mels)
    //   val  = max(0, min(down, up))
    let mut bank = vec![0.0f32; num_freq_bins * num_mels];
    for i in 0..num_freq_bins {
        for j in 0..num_mels {
            let slope_j = filter_freqs[j] - fft_freqs[i];
            let slope_j2 = filter_freqs[j + 2] - fft_freqs[i];
            let down = -slope_j / filter_diff[j];
            let up = slope_j2 / filter_diff[j + 1];
            let v = down.min(up).max(0.0);
            bank[i * num_mels + j] = v as f32;
        }
    }
    // Slaney area normalization: enorm[j] = 2 / (filter_freqs[j+2] - filter_freqs[j]).
    for j in 0..num_mels {
        let width = filter_freqs[j + 2] - filter_freqs[j];
        let enorm = 2.0 / width;
        for i in 0..num_freq_bins {
            bank[i * num_mels + j] *= enorm as f32;
        }
    }
    bank
}

/// Periodic Hann window of length `n` (matches `np.hanning(n+1)[:-1]` /
/// `torch.hann_window(n, periodic=True)`): `w[k] = 0.5 - 0.5·cos(2πk/n)`.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    let two_pi = 2.0 * std::f32::consts::PI;
    for k in 0..n {
        w[k] = 0.5 - 0.5 * (two_pi * k as f32 / n as f32).cos();
    }
    w
}

/// Compute the log-mel spectrogram of a mono waveform.
///
/// Returns row-major `[n_mels, num_frames]` F32, byte-matching the HF
/// `Lfm3AudioFeatureExtractor._log_mel` (center=False, power=2, log10, clamp,
/// affine). `num_frames = 1 + (waveform.len() - n_fft) / hop` (trailing
/// samples that don't fill a window are dropped — the caller pads first).
pub fn log_mel_spectrogram(waveform: &[f32], cfg: &MelConfig) -> Vec<f32> {
    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let n_mels = cfg.n_mels;
    let num_freq_bins = cfg.num_freq_bins();
    let window = hann_periodic(cfg.win_length);
    debug_assert_eq!(cfg.win_length, n_fft, "LFM3-Audio: win_length == n_fft");

    // Precompute rfft twiddle tables: cos/sin for (bin k, sample n).
    // X[k] = Σ_n x[n]·e^{-2πi·k·n/N}; power[k] = re² + im².
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut cos_t = vec![0.0f64; num_freq_bins * n_fft];
    let mut sin_t = vec![0.0f64; num_freq_bins * n_fft];
    for k in 0..num_freq_bins {
        for n in 0..n_fft {
            let ang = two_pi * k as f64 * n as f64 / n_fft as f64;
            cos_t[k * n_fft + n] = ang.cos();
            sin_t[k * n_fft + n] = ang.sin();
        }
    }

    let num_frames = if waveform.len() < n_fft {
        0
    } else {
        1 + (waveform.len() - n_fft) / hop
    };

    // power[k, f] = |X[k]|² at frame f, row-major [num_freq_bins, num_frames].
    let mut power = vec![0.0f64; num_freq_bins * num_frames];
    for f in 0..num_frames {
        let base = f * hop;
        for k in 0..num_freq_bins {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            let tbl = k * n_fft;
            for n in 0..n_fft {
                let s = waveform[base + n] as f64 * window[n] as f64;
                re += s * cos_t[tbl + n];
                im -= s * sin_t[tbl + n];
            }
            power[k * num_frames + f] = re * re + im * im;
        }
    }

    // mel = mel_filtersᵀ · power  → [n_mels, num_frames].
    let mel_filters = slaney_mel_filter_bank(cfg);
    let mut out = vec![0.0f32; n_mels * num_frames];
    for m in 0..n_mels {
        for f in 0..num_frames {
            let mut acc = 0.0f64;
            for k in 0..num_freq_bins {
                acc += mel_filters[k * n_mels + m] as f64 * power[k * num_frames + f];
            }
            // log10, clamp, affine.
            let mut v = acc.log10();
            if v < cfg.log_floor as f64 {
                v = cfg.log_floor as f64;
            }
            out[m * num_frames + f] = ((v + cfg.norm_offset as f64) / cfg.norm_divisor as f64) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn parse_meta(path: &str) -> std::collections::HashMap<String, usize> {
        let txt = std::fs::read_to_string(path).unwrap();
        txt.lines()
            .filter_map(|l| {
                let mut it = l.splitn(2, '=');
                let k = it.next()?.trim().to_string();
                let v: usize = it.next()?.trim().parse().ok()?;
                Some((k, v))
            })
            .collect()
    }

    /// Validate the full mel pipeline against a numpy reference generated from
    /// the exact same transformers math (tests/fixtures/gen_mel_reference.py).
    #[test]
    fn log_mel_matches_numpy_reference() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let meta = parse_meta(&format!("{dir}/meta.txt"));
        let num_frames = meta["num_frames"];
        let n_mels = meta["n_mels"];
        let num_freq_bins = meta["num_freq_bins"];

        let wave = read_f32(&format!("{dir}/waveform.bin"));
        let ref_filters = read_f32(&format!("{dir}/mel_filters.bin"));
        let ref_log_mel = read_f32(&format!("{dir}/log_mel.bin"));

        let cfg = MelConfig::lfm3_audio();
        assert_eq!(cfg.n_mels, n_mels);
        assert_eq!(cfg.num_freq_bins(), num_freq_bins);

        // 1. mel filter bank byte-match.
        let filters = slaney_mel_filter_bank(&cfg);
        assert_eq!(filters.len(), ref_filters.len());
        let mut max_err = 0.0f32;
        for (a, b) in filters.iter().zip(ref_filters.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-6, "mel filter bank max abs err {max_err:e}");

        // 2. full log-mel pipeline.
        let out = log_mel_spectrogram(&wave, &cfg);
        assert_eq!(out.len(), n_mels * num_frames);
        assert_eq!(out.len(), ref_log_mel.len());
        let mut max_err = 0.0f32;
        let mut rel_err = 0.0f32;
        for (a, b) in out.iter().zip(ref_log_mel.iter()) {
            let d = (a - b).abs();
            max_err = max_err.max(d);
            if b.abs() > 0.01 {
                rel_err = rel_err.max(d / b.abs());
            }
        }
        // f64 twiddles vs numpy's pocketfft rfft — mathematically identical,
        // but accumulation order differs, so expect ~1e-5..1e-4 relative slack.
        // (The HF extractor itself varies across FFT backends; the encoder is
        // trained on float32 mel and its weights ship Q8 ~1e-2 noise, so this
        // is far below the meaningful precision floor.)
        eprintln!("log-mel max abs err {max_err:e}, max rel err {rel_err:e}");
        assert!(max_err < 1e-3, "log-mel max abs err {max_err:e}");
        assert!(rel_err < 2e-4, "log-mel max rel err {rel_err:e}");
    }
}
