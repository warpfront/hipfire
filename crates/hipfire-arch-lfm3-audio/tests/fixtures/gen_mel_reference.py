#!/usr/bin/env python3
# Generate reference log-mel features byte-matching the HF
# `Lfm3AudioFeatureExtractor._log_mel` (transformers `audio_utils.mel_filter_bank`
# Slaney bank + `torch.stft` center=False n_fft=400 hop=160 win=400 power=2
# log10 clamp(-12) affine (x+4)/4). Pure numpy (mirrors the HF numpy path).
#
# Outputs (raw little-endian f32) into the crate's tests/fixtures/:
#   waveform.bin      [16000]            1s deterministic test signal
#   mel_filters.bin   [201*128] row-major (num_freq_bins, num_mel_filters)
#   log_mel.bin       [128*num_frames] row-major (num_mels, num_frames)
#   meta.txt          num_frames / dims so the Rust test can size reads
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# --- Slaney mel scale (transformers audio_utils.hertz_to_mel / mel_to_hertz) ---
def hz_to_mel(freq):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if np.isscalar(freq):
        if freq >= min_log_hertz:
            mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    else:
        log_region = freq >= min_log_hertz
        mels = np.asarray(mels, dtype=np.float64)
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels

def mel_to_hz(mels):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if np.isscalar(mels):
        if mels >= min_log_mel:
            freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    else:
        mels = np.asarray(mels, dtype=np.float64)
        freq = np.asarray(freq, dtype=np.float64)
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq

def triangular_filter_bank(fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes))

def mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency, max_frequency, sampling_rate):
    mel_min = hz_to_mel(min_frequency)
    mel_max = hz_to_mel(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    mel_filters = triangular_filter_bank(fft_freqs, filter_freqs)
    # Slaney area normalization
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters = mel_filters * enorm[None, :]
    return mel_filters.astype(np.float32)

# --- HF Lfm3AudioFeatureExtractor constants ---
N_FFT = 400
HOP = 160
WIN = 400
N_MELS = 128
SR = 16000
F_MAX = 8000.0
LOG_FLOOR = -12.0
NORM_OFFSET = 4.0
NORM_DIVISOR = 4.0

# --- deterministic 1s test waveform: sum of sines + seeded noise ---
rng = np.random.RandomState(0xA1B2C3D4)
t = np.arange(SR, dtype=np.float64) / SR
wave = (0.3*np.sin(2*np.pi*220*t)
        + 0.2*np.sin(2*np.pi*440*t)
        + 0.1*np.sin(2*np.pi*880*t)
        + 0.05*rng.standard_normal(SR))
wave = wave.astype(np.float32)

# --- mel filter bank [201, 128] ---
mel_filters = mel_filter_bank(N_FFT // 2 + 1, N_MELS, 0.0, F_MAX, SR)

# --- STFT (center=False) matching torch.stft ---
window = np.hanning(WIN + 1)[:-1].astype(np.float64)  # periodic Hann
num_frames = 1 + (len(wave) - N_FFT) // HOP
power = np.empty((N_FFT // 2 + 1, num_frames), dtype=np.float64)
for i in range(num_frames):
    frame = wave[i*HOP:i*HOP+N_FFT].astype(np.float64) * window
    X = np.fft.rfft(frame, n=N_FFT)
    power[:, i] = np.abs(X) ** 2

# --- log-mel ---
log_mel = (mel_filters.T.astype(np.float64) @ power)
log_mel = np.log10(log_mel)
log_mel = np.clip(log_mel, a_min=LOG_FLOOR, a_max=None)
log_mel = (log_mel + NORM_OFFSET) / NORM_DIVISOR
log_mel = log_mel.astype(np.float32)

wave.tofile(OUT / "waveform.bin")
mel_filters.tofile(OUT / "mel_filters.bin")
log_mel.tofile(OUT / "log_mel.bin")
with open(OUT / "meta.txt", "w") as f:
    f.write(f"num_samples={len(wave)}\n")
    f.write(f"num_frames={num_frames}\n")
    f.write(f"n_mels={N_MELS}\n")
    f.write(f"n_fft={N_FFT}\n")
    f.write(f"hop={HOP}\n")
    f.write(f"num_freq_bins={N_FFT//2+1}\n")
print(f"wrote waveform {wave.shape}, mel_filters {mel_filters.shape}, log_mel {log_mel.shape} ({num_frames} frames)")
