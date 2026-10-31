// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#ifndef HF_DEVICE_WAIT_DEADLINE_H
#include "device_wait_deadline.h"
#endif

// Cross-device EP boundary (#863): publish-before-wait on shared monotonic
// epochs, with no host epoch argument, so a retained body can replay it
// unchanged. Each rank owns one epoch word that only it increments. The
// fetch_add releases this rank's earlier stream work at system scope and
// yields the boundary's epoch e; the wait then acquires every rank's word at
// system scope until each has reached e. Every rank runs the same boundary
// sequence from the same zeroed words, so the k-th boundary is epoch k on all
// ranks. Comparison is wrap-safe (signed difference).
//
// The wait is bounded by device_wait_deadline.h. A stalled peer or a host
// poison (a nonzero code written into this device's wait block) stops the
// wave; the host fails the step when it reads the block.

extern "C" __global__ void ep_boundary_signal_wait(
    unsigned int *__restrict__ own,
    const unsigned int *__restrict__ e0,
    const unsigned int *__restrict__ e1,
    const unsigned int *__restrict__ e2,
    const unsigned int *__restrict__ e3,
    const unsigned int *__restrict__ e4,
    const unsigned int *__restrict__ e5,
    const unsigned int *__restrict__ e6,
    const unsigned int *__restrict__ e7,
    int ranks,
    hf_device_wait_block *__restrict__ wait_block) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned int epoch =
        __hip_atomic_fetch_add(own, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM) + 1u;
    const unsigned int *words[8] = {e0, e1, e2, e3, e4, e5, e6, e7};
    hf_bounded_wait(
        [&] {
            for (int p = 0; p < ranks; ++p) {
                const unsigned int seen = __hip_atomic_load(
                    words[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
                if ((int)(seen - epoch) < 0) return false;
            }
            return true;
        },
        wait_block, HF_WAIT_SITE_EP_BOUNDARY_WAIT, epoch);
}
