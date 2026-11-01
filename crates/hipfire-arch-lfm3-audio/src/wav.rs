// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal PCM WAV decoder for LFM3-Audio input.
//!
//! Decodes a RIFF/WAVE file to a mono F32 waveform at the model's expected
//! sample rate (16 kHz). Supports 16/24/32-bit signed-integer and 32-bit
//! float PCM (mono or multi-channel→mean) — the common ASR test-audio
//! formats. Multi-channel clips are mixed to mono by averaging channels
//! (matching the HF `channel_reduction: "mean"`). The HF `load_audio` backend
//! (librosa/torchcodec) resamples to 16 kHz; this decoder requires the file to
//! already be 16 kHz (resampling is a follow-up — the feature extractor checks
//! `sampling_rate` and refuses a mismatch).
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

#[derive(Debug)]
pub enum WavError {
    NotWav,
    BadHeader(String),
    UnsupportedFormat(String),
    SampleRateMismatch { got: u32, expected: u32 },
    NoData,
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::NotWav => write!(f, "not a RIFF/WAVE file"),
            WavError::BadHeader(s) => write!(f, "bad WAV header: {s}"),
            WavError::UnsupportedFormat(s) => write!(f, "unsupported WAV format: {s}"),
            WavError::SampleRateMismatch { got, expected } => {
                write!(f, "sample rate {got} Hz != expected {expected} Hz (resample first)")
            }
            WavError::NoData => write!(f, "WAV has no data chunk"),
        }
    }
}

/// Decode a 16-bit PCM WAV to a mono F32 waveform at `expected_sr`.
pub fn decode_wav(bytes: &[u8], expected_sr: u32) -> Result<Waveform, WavError> {
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
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_size).min(bytes.len());
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
        pos = body_start + chunk_size + (chunk_size & 1);
    }

    if fmt_sample_rate == 0 {
        return Err(WavError::BadHeader("no fmt chunk".into()));
    }
    if fmt_sample_rate != expected_sr {
        return Err(WavError::SampleRateMismatch { got: fmt_sample_rate, expected: expected_sr });
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
                    acc += f32::from_le_bytes([frame[4 * c], frame[4 * c + 1], frame[4 * c + 2], frame[4 * c + 3]]);
                }
                acc / channels as f32
            })
            .collect(),
        (1, 32) => data
            .chunks_exact(4 * channels)
            .map(|frame| {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let s = i32::from_le_bytes([frame[4 * c], frame[4 * c + 1], frame[4 * c + 2], frame[4 * c + 3]]) as f32;
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
                    let s = ((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16)) << 8 >> 8;
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

    Ok(Waveform { samples, sample_rate: fmt_sample_rate })
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
        let data_size = data.len() as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_size).to_le_bytes());
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
        out.extend_from_slice(&data);
        out
    }
}
