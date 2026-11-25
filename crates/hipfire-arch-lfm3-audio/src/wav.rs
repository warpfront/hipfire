// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal PCM WAV decoder for LFM3-Audio input.
//!
//! Decodes a RIFF/WAVE file to a mono F32 waveform at its own sample rate.
//! Supports 16/24/32-bit signed-integer and 32-bit
//! float PCM (mono or multi-channel→mean) — the common ASR test-audio
//! formats. Multi-channel clips are mixed to mono by averaging channels
//! (matching the HF `channel_reduction: "mean"`). The HF `load_audio` backend
//! (librosa/torchcodec) resamples to 16 kHz; [`resample_linear`] does the
//! same for the ASR route. Headers are untrusted: the sample rate and clip
//! length are bounded and an input with no complete sample frame is refused.
//!
//! Limitations (NITS-1): `WAVE_FORMAT_EXTENSIBLE` and 8-bit PCM are rejected
//! (not in the LFM3-Audio test set); a trailing partial frame shorter than one
//! interleave is dropped by `chunks_exact` (sub-millisecond at 16 kHz — below
//! the encoder's 8-frame / 80 ms token granularity).

/// A decoded mono waveform + its sample rate.
pub struct Waveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Lowest / highest `fmt ` sample rate accepted. The rate scales the resampled
/// length (`len * target / rate`), so a 1 Hz header would turn a small upload
/// into a huge allocation; real capture rates sit well inside this range.
pub const MIN_SAMPLE_RATE: u32 = 1_000;
pub const MAX_SAMPLE_RATE: u32 = 384_000;

/// Longest clip accepted, in seconds of audio.
pub const MAX_DURATION_SECS: u64 = 600;

#[derive(Debug)]
pub enum WavError {
    NotWav,
    BadHeader(String),
    UnsupportedFormat(String),
    SampleRateMismatch { got: u32, expected: u32 },
    NoData,
    TooLong { secs: u64, max_secs: u64 },
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::NotWav => write!(f, "not a RIFF/WAVE file"),
            WavError::BadHeader(s) => write!(f, "bad WAV header: {s}"),
            WavError::UnsupportedFormat(s) => write!(f, "unsupported WAV format: {s}"),
            WavError::SampleRateMismatch { got, expected } => {
                write!(
                    f,
                    "sample rate {got} Hz != expected {expected} Hz (resample first)"
                )
            }
            WavError::NoData => write!(f, "WAV has no complete sample frame"),
            WavError::TooLong { secs, max_secs } => {
                write!(f, "WAV is {secs} s long; the limit is {max_secs} s")
            }
        }
    }
}

/// Decode a 16-bit PCM WAV to a mono F32 waveform at `expected_sr`.
pub fn decode_wav(bytes: &[u8], expected_sr: u32) -> Result<Waveform, WavError> {
    let w = decode_wav_any(bytes)?;
    if w.sample_rate != expected_sr {
        return Err(WavError::SampleRateMismatch {
            got: w.sample_rate,
            expected: expected_sr,
        });
    }
    Ok(w)
}

/// Decode a PCM WAV to a mono F32 waveform at its own sample rate (the caller
/// decides whether to resample — the daemon's ASR route resamples to the
/// model's rate so 24 kHz TTS/browser audio works without pre-conversion).
pub fn decode_wav_any(bytes: &[u8]) -> Result<Waveform, WavError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotWav);
    }
    let mut pos = 12usize;
    let mut fmt_audio_format: u16 = 0;
    let mut fmt_channels: u16 = 0;
    let mut fmt_sample_rate: u32 = 0;
    let mut fmt_bits_per_sample: u16 = 0;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(chunk_size).min(bytes.len());
        match chunk_id {
            b"fmt " => {
                if body_end < body_start + 16 {
                    return Err(WavError::BadHeader("fmt chunk too short".into()));
                }
                let b = &bytes[body_start..body_end];
                fmt_audio_format = u16::from_le_bytes([b[0], b[1]]);
                fmt_channels = u16::from_le_bytes([b[2], b[3]]);
                fmt_sample_rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                fmt_bits_per_sample = u16::from_le_bytes([b[14], b[15]]);
            }
            b"data" => {
                data = &bytes[body_start..body_end];
            }
            _ => {}
        }
        // chunks are word-aligned (pad to even)
        pos = body_start
            .saturating_add(chunk_size)
            .saturating_add(chunk_size & 1);
    }

    if fmt_sample_rate == 0 {
        return Err(WavError::BadHeader("no fmt chunk".into()));
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&fmt_sample_rate) {
        return Err(WavError::BadHeader(format!(
            "sample rate {fmt_sample_rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
        )));
    }
    if data.is_empty() {
        return Err(WavError::NoData);
    }
    let channels = fmt_channels.max(1) as usize;

    let samples: Vec<f32> = match (fmt_audio_format, fmt_bits_per_sample) {
        (1, 16) => data
            .chunks_exact(2 * channels)
            .map(|frame| {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let s = i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]) as f32;
                    acc += s / 32768.0;
                }
                acc / channels as f32
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * channels)
            .map(|frame| {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    acc += f32::from_le_bytes([
                        frame[4 * c],
                        frame[4 * c + 1],
                        frame[4 * c + 2],
                        frame[4 * c + 3],
                    ]);
                }
                acc / channels as f32
            })
            .collect(),
        (1, 32) => data
            .chunks_exact(4 * channels)
            .map(|frame| {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let s = i32::from_le_bytes([
                        frame[4 * c],
                        frame[4 * c + 1],
                        frame[4 * c + 2],
                        frame[4 * c + 3],
                    ]) as f32;
                    acc += s / 2147483648.0;
                }
                acc / channels as f32
            })
            .collect(),
        (1, 24) => data
            .chunks_exact(3 * channels)
            .map(|frame| {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let b = &frame[3 * c..3 * c + 3];
                    // 24-bit signed little-endian → i32
                    let s =
                        ((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16)) << 8 >> 8;
                    acc += s as f32 / 8388608.0;
                }
                acc / channels as f32
            })
            .collect(),
        _ => {
            return Err(WavError::UnsupportedFormat(format!(
                "format {} ({}-bit)",
                fmt_audio_format, fmt_bits_per_sample
            )))
        }
    };
    // A data chunk shorter than one interleaved frame decodes to nothing.
    if samples.is_empty() {
        return Err(WavError::NoData);
    }
    check_duration(samples.len(), fmt_sample_rate)?;

    Ok(Waveform {
        samples,
        sample_rate: fmt_sample_rate,
    })
}

fn check_duration(num_samples: usize, sample_rate: u32) -> Result<(), WavError> {
    let secs = num_samples as u64 / u64::from(sample_rate.max(1));
    if secs > MAX_DURATION_SECS {
        return Err(WavError::TooLong {
            secs,
            max_secs: MAX_DURATION_SECS,
        });
    }
    Ok(())
}

/// Linear resample to `target_sr` (the ASR route's resampler). Refuses an
/// empty waveform, a rate outside [`MIN_SAMPLE_RATE`]..=[`MAX_SAMPLE_RATE`]
/// and a clip longer than [`MAX_DURATION_SECS`] instead of indexing past the
/// end or sizing the output from an untrusted header.
pub fn resample_linear(w: Waveform, target_sr: u32) -> Result<Waveform, WavError> {
    if w.samples.is_empty() {
        return Err(WavError::NoData);
    }
    for rate in [w.sample_rate, target_sr] {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate) {
            return Err(WavError::BadHeader(format!(
                "sample rate {rate} Hz is outside {MIN_SAMPLE_RATE}..={MAX_SAMPLE_RATE} Hz"
            )));
        }
    }
    check_duration(w.samples.len(), w.sample_rate)?;
    if w.sample_rate == target_sr {
        return Ok(w);
    }
    // Both rates are bounded and the clip is at most MAX_DURATION_SECS long,
    // so `n` is at most MAX_DURATION_SECS * MAX_SAMPLE_RATE (+1); the checked
    // ops make that explicit rather than relying on it.
    let n = (w.samples.len() as u64)
        .checked_mul(u64::from(target_sr))
        .map(|v| v / u64::from(w.sample_rate))
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| WavError::BadHeader("resampled length overflows".into()))?
        .max(1);
    let last = w.samples.len() - 1;
    let mut out = vec![0.0f32; n];
    let ratio = w.sample_rate as f64 / target_sr as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 * ratio;
        let i0 = (src as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let f = (src - i0 as f64) as f32;
        *o = w.samples[i0] * (1.0 - f) + w.samples[i1] * f;
    }
    Ok(Waveform {
        samples: out,
        sample_rate: target_sr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a synthetic 16-bit mono PCM WAV at 16 kHz and verify the
    /// decoded samples match the input.
    #[test]
    fn round_trips_16bit_mono_wav() {
        let sr = 16000u32;
        let samples: Vec<f32> = (0..sr as usize * 2)
            .map(|i| (0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin()))
            .collect();
        let bytes = encode_pcm16_mono(&samples, sr);
        let wav = decode_wav(&bytes, sr).unwrap();
        assert_eq!(wav.sample_rate, sr);
        assert_eq!(wav.samples.len(), samples.len());
        let mut max_err = 0.0f32;
        for (a, b) in wav.samples.iter().zip(samples.iter()) {
            // 16-bit quantization: step = 1/32768 ≈ 3e-5.
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 5e-5, "round-trip max err {max_err:e}");
    }

    /// Encode a mono F32 vector as a 16-bit PCM WAV (test helper).
    fn encode_pcm16_mono(samples: &[f32], sr: u32) -> Vec<u8> {
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|&s| {
                let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                v.to_le_bytes()
            })
            .collect();
        pcm16_mono_wav(&data, sr)
    }

    /// A 16-bit mono PCM WAV around raw `data` bytes (padded to even, as RIFF
    /// chunks are).
    fn pcm16_mono_wav(data: &[u8], sr: u32) -> Vec<u8> {
        let data_size = data.len() as u32;
        let pad = data.len() & 1;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_size + pad as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&sr.to_le_bytes()); // sample rate (u32)
        out.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate (u32) = sr*channels*bits/8
        out.extend_from_slice(&2u16.to_le_bytes()); // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(data);
        out.extend(std::iter::repeat_n(0u8, pad));
        out
    }

    /// fivetide's #1066 review, blocking item 2: a 46-byte PCM16 44.1 kHz WAV
    /// whose data chunk is 1 byte decoded to zero samples, and the resampler
    /// indexed `samples[0]` ("index out of bounds: the len is 0 but the index
    /// is 0"), killing the daemon. Both the decoder and the resampler now
    /// refuse it.
    #[test]
    fn data_chunk_shorter_than_one_frame_is_refused() {
        let bytes = pcm16_mono_wav(&[0x7f], 44_100);
        assert_eq!(bytes.len(), 46);
        assert!(matches!(decode_wav_any(&bytes), Err(WavError::NoData)));
        let empty = Waveform {
            samples: Vec::new(),
            sample_rate: 44_100,
        };
        assert!(matches!(
            resample_linear(empty, 16_000),
            Err(WavError::NoData)
        ));
    }

    /// The resampled length is `len * target / rate`: a 1 Hz header must be
    /// refused, not turned into a 16000x allocation.
    #[test]
    fn absurd_sample_rate_is_refused() {
        let bytes = pcm16_mono_wav(&[0u8; 64], 1);
        assert!(matches!(
            decode_wav_any(&bytes),
            Err(WavError::BadHeader(_))
        ));
        let w = Waveform {
            samples: vec![0.0; 32],
            sample_rate: 1,
        };
        assert!(matches!(
            resample_linear(w, 16_000),
            Err(WavError::BadHeader(_))
        ));
    }

    #[test]
    fn clip_longer_than_the_limit_is_refused() {
        let sr = MIN_SAMPLE_RATE;
        let w = Waveform {
            samples: vec![0.0; (MAX_DURATION_SECS as usize + 1) * sr as usize],
            sample_rate: sr,
        };
        assert!(matches!(
            resample_linear(w, 16_000),
            Err(WavError::TooLong { .. })
        ));
    }

    #[test]
    fn resamples_44k1_to_16k() {
        let w = Waveform {
            samples: vec![0.5; 44_100],
            sample_rate: 44_100,
        };
        let r = resample_linear(w, 16_000).unwrap();
        assert_eq!(r.sample_rate, 16_000);
        assert_eq!(r.samples.len(), 16_000);
        assert!(r.samples.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }
}
