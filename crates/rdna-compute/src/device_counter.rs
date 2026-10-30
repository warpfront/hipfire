// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic device-resident counter advance: `*counter += delta` on a 4-byte
//! device i32. Graph-capture safe (blob launch); lets a captured program advance
//! a position/length that its other kernels read from device memory.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};

pub const DEVICE_COUNTER_ADD_SRC: &str =
    include_str!("../../../kernels/src/device_counter_add.hip");
const MODULE: &str = "device_counter_add";
const FUNC: &str = "device_counter_add_i32";

impl Gpu {
    /// `*counter += delta` where `counter` holds a device i32 (first 4 bytes).
    pub fn device_counter_add_i32(&mut self, counter: &GpuTensor, delta: i32) -> HipResult<()> {
        if counter.byte_size() < 4 {
            return Err(HipError::new(
                0,
                "device_counter_add_i32: counter must hold a 4-byte device i32",
            ));
        }
        if !matches!(counter.dtype, DType::Raw | DType::F32) {
            return Err(HipError::new(
                0,
                &format!(
                    "device_counter_add_i32: counter dtype {:?} is not a raw 4-byte scalar",
                    counter.dtype
                ),
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(MODULE, DEVICE_COUNTER_ADD_SRC, FUNC)?;
        let cp = counter.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &delta as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(FUNC, [1, 1, 1], [1, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(cp);
            b.push_i32(delta);
            b
        })
    }
}
