// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// Diagnostic only: emulate a bf16 gate/up h handoff without changing f32
// storage or any consumer. This extra pass is NOT a performance candidate.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
extern "C" __global__ void round_h_bf16_quality_gfx12(float* h, unsigned long long count) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) h[i] = float(hip_bfloat16::round_to_bfloat16(h[i]));
}
