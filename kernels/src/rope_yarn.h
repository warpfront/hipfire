// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

// YaRN ("NTK-by-parts") RoPE scaling, as a by-value kernel argument. Mirrors
// `rdna_compute::rope::RopeYarn`, which lowers a checkpoint's `rope_parameters`
// (HF `_compute_yarn_parameters`) for one rotary width:
//   inv_freq'[pair] = inv_freq[pair] * (ramp * freq_scale + (1 - ramp)),
//   ramp            = clamp((pair - corr_low) / (corr_high - corr_low), 0, 1),
// and cos/sin are multiplied by `mscale` (HF `attention_scaling`).
// `freq_scale == 1 && mscale == 1` is plain RoPE: the angle and the sin/cos
// pass through untouched, so unscaled launches stay bit-identical.

struct RopeYarn {
    float freq_scale;  // 1 / factor
    float corr_low;
    float corr_high;
    float mscale;
};

// sin/cos of `angle` (= position * inv_freq[pair]) under `yarn`.
static __device__ __forceinline__ void rope_yarn_sincos(
    float angle, int pair, RopeYarn yarn, float* sine, float* cosine) {
    if (yarn.freq_scale != 1.0f) {
        float span = yarn.corr_high - yarn.corr_low;
        float ramp = fminf(fmaxf(((float)pair - yarn.corr_low)
            / (span == 0.0f ? 0.001f : span), 0.0f), 1.0f);
        angle *= ramp * yarn.freq_scale + (1.0f - ramp);
    }
    *sine = sinf(angle);
    *cosine = cosf(angle);
    if (yarn.mscale != 1.0f) {
        *sine *= yarn.mscale;
        *cosine *= yarn.mscale;
    }
}
