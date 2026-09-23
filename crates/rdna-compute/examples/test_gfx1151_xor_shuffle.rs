// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bit-exact gfx1151 validation for the reviewed XOR cross-lane lowering table.
//!
//! This compares raw 32-bit lane payloads, not floating-point values, so a
//! passing result proves each lowering has the same lane mapping as
//! `__shfl_xor(..., width=32)`.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};

const CASES: usize = 257;
const LANES: usize = 32;
const STRIDES: [usize; 5] = [1, 2, 4, 8, 16];
const KERNEL_NAME: &str = "test_gfx1151_xor_shuffle";
const KERNEL_SRC: &str = concat!(
    r#"
#include <hip/hip_runtime.h>
"#,
    include_str!("../../../kernels/src/gfx1151_xor_shuffle.h"),
    r#"
extern "C" __global__ __launch_bounds__(32, 16)
void test_gfx1151_xor_shuffle(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    int cases
) {
    const int test_case = blockIdx.x;
    const int lane = threadIdx.x;
    if (test_case >= cases || lane >= 32) return;
    const int value = (int)input[test_case * 32 + lane];
    const int strides[5] = {1, 2, 4, 8, 16};
    #pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
        const int stride = strides[stage];
        const unsigned int reference =
            (unsigned int)__shfl_xor(value, stride, 32);
        const unsigned int candidate =
            (unsigned int)hipfire_xor_shuffle_b32_gfx1151(value, stride);
        const int offset = ((test_case * 5 + stage) * 32 + lane) * 2;
        output[offset] = reference;
        output[offset + 1] = candidate;
    }
}
"#
);

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151", "this proof is exact-architecture only");
    gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME)
        .expect("compile XOR-shuffle proof kernel");

    let mut state = 0x9e37_79b9_u32;
    let mut input = Vec::with_capacity(CASES * LANES);
    for test_case in 0..CASES {
        for lane in 0..LANES {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            input.push(
                state
                    ^ (test_case as u32).wrapping_mul(0x85eb_ca6b)
                    ^ (lane as u32).wrapping_mul(0xc2b2_ae35),
            );
        }
    }

    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&input[..]),
        )
    };
    let input_gpu = gpu
        .upload_raw(input_bytes, &[input_bytes.len()])
        .expect("upload lane payloads");
    let output_words = CASES * STRIDES.len() * LANES * 2;
    let output_gpu = gpu
        .zeros(&[output_words * 4], DType::Raw)
        .expect("allocate proof output");

    let mut kernargs = KernargBlob::new();
    kernargs.push_ptr(input_gpu.buf.as_ptr());
    kernargs.push_ptr(output_gpu.buf.as_ptr());
    kernargs.push_i32(CASES as i32);
    gpu.launch_kernel_blob(
        KERNEL_NAME,
        [CASES as u32, 1, 1],
        [LANES as u32, 1, 1],
        0,
        kernargs.as_mut_slice(),
    )
    .expect("launch XOR-shuffle proof");
    gpu.hip.device_synchronize().expect("proof synchronize");

    let mut output = vec![0u32; output_words];
    let output_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(&output[..]),
        )
    };
    gpu.hip
        .memcpy_dtoh(output_bytes, &output_gpu.buf)
        .expect("download proof output");

    let mut mismatches = 0usize;
    for test_case in 0..CASES {
        for (stage, stride) in STRIDES.into_iter().enumerate() {
            for lane in 0..LANES {
                let offset = ((test_case * STRIDES.len() + stage) * LANES + lane) * 2;
                let reference = output[offset];
                let candidate = output[offset + 1];
                if reference != candidate {
                    mismatches += 1;
                    if mismatches <= 8 {
                        eprintln!(
                            "mismatch case={test_case} lane={lane} stride={stride}: \
                             reference=0x{reference:08x} candidate=0x{candidate:08x}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "gfx1151 XOR-shuffle lowering mismatch");
    println!(
        "PASS gfx1151 XOR shuffle: {} raw-bit comparisons across strides {:?}",
        CASES * LANES * STRIDES.len(),
        STRIDES
    );
}
