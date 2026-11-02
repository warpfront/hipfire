// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#ifndef HF_DEVICE_WAIT_DEADLINE_H
#include "device_wait_deadline.h"
#endif

// Graph-resident three/four-rank producer barrier for gfx1201 TP. hipStreamWriteValue
// and hipStreamWaitValue instantiate inside HIP graphs on ROCm 7.14, but their
// replay observes the prior epoch after the first launch. Kernel nodes do replay
// correctly, so publish and consume the epoch through system-scope atomics.
//
// The waits are bounded by device_wait_deadline.h. When a peer rank stalls,
// the waiter records its site and epoch in `wait_block` and exits the wave.
// The rank's later kernels then read stale peer data, and the host fails the
// step when it reads the block after the step.

extern "C" __global__ void tp_graph_signal_store_gfx1201(
    unsigned int *__restrict__ signal,
    unsigned int epoch) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_exchange(
            signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

extern "C" __global__ void tp_graph_signal_wait2_gfx1201(
    const unsigned int *__restrict__ signal0,
    const unsigned int *__restrict__ signal1,
    unsigned int epoch,
    hf_device_wait_block *__restrict__ wait_block) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    hf_bounded_wait(
        [&] {
            return __hip_atomic_load(
                       signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >= epoch &&
                   __hip_atomic_load(
                       signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >= epoch;
        },
        wait_block, HF_WAIT_SITE_TP_SIGNAL_WAIT2_GFX1201, epoch);
}

extern "C" __global__ void tp_graph_signal_wait3_gfx1201(
    const unsigned int *__restrict__ signal0,
    const unsigned int *__restrict__ signal1,
    const unsigned int *__restrict__ signal2,
    unsigned int epoch,
    hf_device_wait_block *__restrict__ wait_block) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    hf_bounded_wait(
        [&] {
            return __hip_atomic_load(
                       signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >= epoch &&
                   __hip_atomic_load(
                       signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >= epoch &&
                   __hip_atomic_load(
                       signal2, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >= epoch;
        },
        wait_block, HF_WAIT_SITE_TP_SIGNAL_WAIT3_GFX1201, epoch);
}
