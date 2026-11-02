// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launcher for the batched DFlash tape-capture copy
//! (`kernels/src/tape_copy_batched.hip`): up to [`TAPE_COPY_MAX`] D→D copies
//! per launch, descriptors passed by value in the kernarg segment. Self-
//! contained via `include_str!` (no shared-registry edits), launched through
//! `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dispatch::Gpu;
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::tape_copy_batched`].
pub const TAPE_COPY_BATCHED_SRC: &str = include_str!("../../../kernels/src/tape_copy_batched.hip");
/// Module key and device symbol.
pub const TAPE_COPY_BATCHED_SYMBOL: &str = "tape_copy_batched";
/// Descriptors per launch (`TAPE_COPY_MAX` in the kernel).
pub const TAPE_COPY_MAX: usize = 48;
/// Bytes per block (`TAPE_COPY_CHUNK` in the kernel).
const TAPE_COPY_CHUNK: u64 = 16384;

/// One D→D copy: `bytes` from device address `src` to device address `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeCopy {
    pub dst: u64,
    pub src: u64,
    pub bytes: u64,
}

impl TapeCopy {
    /// `bytes` from `src + src_offset` to `dst + dst_offset`.
    pub fn new(
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        bytes: usize,
    ) -> Self {
        Self {
            dst: dst.as_ptr() as u64 + dst_offset as u64,
            src: src.as_ptr() as u64 + src_offset as u64,
            bytes: bytes as u64,
        }
    }
}

/// `#[repr(C)]` mirror of `struct TapeCopyBatch` (the single by-value kernel
/// argument): 3 x 48 u64 then the u32 count.
#[repr(C)]
struct TapeCopyBatchArgs {
    src: [u64; TAPE_COPY_MAX],
    dst: [u64; TAPE_COPY_MAX],
    cnt: [u64; TAPE_COPY_MAX],
    n: u32,
}

impl Gpu {
    /// Perform `copies` (disjoint destinations) in `ceil(len / 48)` launches
    /// on the active stream. Equivalent, byte for byte, to one
    /// `memcpy_dtod_at_auto` per entry in order.
    pub fn tape_copy_batched(&mut self, copies: &[TapeCopy]) -> HipResult<()> {
        self.bind_thread()?;
        if copies.is_empty() {
            return Ok(());
        }
        self.ensure_kernel(
            TAPE_COPY_BATCHED_SYMBOL,
            TAPE_COPY_BATCHED_SRC,
            TAPE_COPY_BATCHED_SYMBOL,
        )?;
        for chunk in copies.chunks(TAPE_COPY_MAX) {
            let mut args = TapeCopyBatchArgs {
                src: [0; TAPE_COPY_MAX],
                dst: [0; TAPE_COPY_MAX],
                cnt: [0; TAPE_COPY_MAX],
                n: chunk.len() as u32,
            };
            let mut max_bytes = 0u64;
            for (i, c) in chunk.iter().enumerate() {
                args.src[i] = c.src;
                args.dst[i] = c.dst;
                args.cnt[i] = c.bytes;
                max_bytes = max_bytes.max(c.bytes);
            }
            if max_bytes == 0 {
                continue;
            }
            let grid_x = max_bytes.div_ceil(TAPE_COPY_CHUNK) as u32;
            let mut params: Vec<*mut c_void> = vec![&mut args as *mut _ as *mut c_void];
            let blob_args = &args;
            self.launch_maybe_blob(
                TAPE_COPY_BATCHED_SYMBOL,
                [grid_x, chunk.len() as u32, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = KernargBlob::new();
                    for v in blob_args
                        .src
                        .iter()
                        .chain(&blob_args.dst)
                        .chain(&blob_args.cnt)
                    {
                        b.push_u64(*v);
                    }
                    b.push_u32(blob_args.n);
                    b
                },
            )?;
        }
        Ok(())
    }
}
