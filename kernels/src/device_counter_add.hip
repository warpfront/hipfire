// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Advance a device-resident i32 counter in place: *counter += delta.
// One thread, one block. Used inside captured graphs so a replay advances a
// position/length that other captured kernels read from device memory, with no
// per-replay host write.
extern "C" __global__ void device_counter_add_i32(int* counter, int delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *counter += delta;
    }
}
