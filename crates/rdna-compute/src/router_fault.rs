// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host side of the router non-finite fault word.
//!
//! The Qwen4 router producers (`moe_router_softmax_top10_f32[_fast]`,
//! `moe_topk_renorm_top10_batched`, `moe_router_logits_guard_top10`) fail a row
//! closed when any of its 512 logits is NaN or +/-Inf: every expert marker is
//! `-1`, every weight `0`, and the row is recorded here. The word is one
//! host-mapped `u64` per device, written by the device with plain system-scope
//! stores (no PCIe atomics, like [`crate::device_wait`]). It lives as long as
//! the `Gpu`, so captured graphs that baked in its address stay valid, and the
//! producer carries the sealed call's layer index as a kernarg, so a graph
//! replay that executes no host router launcher still reports `layer, row`.
//!
//! Encoding: `0` = clear, otherwise `((row + 1) << 24) | (layer + 1)`, where
//! `row` is the global row (`row_base` + the row inside the launch). A device
//! write keeps the smaller code (earliest row, then earliest layer) unless
//! stores race, in which case the last store wins; either way the reported
//! pair really faulted.
//!
//! The host checks at every point a result is read back: the `Gpu::download_*`
//! readers and [`crate::Gpu::take_device_wait_error`] fold the check in, so the
//! batch sampler, the AR sample copy and `Gpus::take_device_wait_errors` report
//! it too. Reading the word clears it ([`crate::Gpu::take_router_fault`]); a
//! caller that starts a request clears stale state with
//! [`crate::Gpu::clear_router_fault`].

use hip_bridge::{HipError, HipResult, HipRuntime};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bits of the code holding `layer + 1` (the row sits above them).
pub const ROUTER_FAULT_LAYER_BITS: u32 = 24;
const LAYER_MASK: u64 = (1u64 << ROUTER_FAULT_LAYER_BITS) - 1;

/// A router row that carried a non-finite logit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterFault {
    pub device_id: i32,
    /// Sealed call layer index passed to the producer.
    pub layer: u32,
    /// Global row: the producer's `row_base` plus the row inside the launch
    /// (the token index within the prefill batch / chunk offset; 0 for decode).
    pub row: u64,
}

impl RouterFault {
    /// The device encoding of `(layer, row)`. `layer` must be below
    /// `2^24 - 1`.
    pub fn encode(layer: u32, row: u64) -> u64 {
        ((row + 1) << ROUTER_FAULT_LAYER_BITS) | (u64::from(layer) + 1)
    }

    /// Decode a non-zero code; `None` for the clear word or a malformed one.
    pub fn decode(device_id: i32, code: u64) -> Option<Self> {
        let layer_plus_one = code & LAYER_MASK;
        let row_plus_one = code >> ROUTER_FAULT_LAYER_BITS;
        if layer_plus_one == 0 || row_plus_one == 0 {
            return None;
        }
        Some(Self {
            device_id,
            layer: (layer_plus_one - 1) as u32,
            row: row_plus_one - 1,
        })
    }

    /// The request-failing error.
    pub fn to_hip_error(&self) -> HipError {
        HipError::new(0, &self.to_string())
    }
}

impl fmt::Display for RouterFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MoE router non-finite logits on GPU {}: layer {} row {} had a NaN or Inf router \
             logit; the row was failed closed (expert -1, weight 0) and the request was aborted",
            self.device_id, self.layer, self.row
        )
    }
}

/// One device's host-mapped fault word, owned by `Gpu`.
#[derive(Debug)]
pub(crate) struct RouterFaultMapping {
    /// `hipHostMalloc` pointer (host view, and what `hipHostFree` takes).
    host: usize,
    /// `hipHostGetDevicePointer` alias handed to kernels.
    device: usize,
    /// Row offset the arch sets for the router launches that follow (the prefill
    /// chunk's first token position; 0 for decode). A plain host-side value read
    /// when a launch is encoded, never by the device.
    row_base: AtomicU64,
}

impl RouterFaultMapping {
    pub(crate) fn alloc(hip: &HipRuntime, device_id: i32) -> HipResult<Self> {
        let host = hip.host_malloc(
            std::mem::size_of::<u64>(),
            hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT,
        )?;
        let device = match hip.host_get_device_pointer(host, 0) {
            Ok(device) if !device.is_null() => device,
            other => {
                let _ = hip.host_free(host);
                return Err(match other {
                    Err(error) => error,
                    Ok(_) => HipError::new(
                        0,
                        &format!("device {device_id}: null device alias for the router fault word"),
                    ),
                });
            }
        };
        // SAFETY: fresh hipHostMalloc allocation of one aligned u64, not yet
        // visible to any kernel.
        unsafe { host.cast::<u64>().write(0) };
        Ok(Self {
            host: host as usize,
            device: device as usize,
            row_base: AtomicU64::new(0),
        })
    }

    pub(crate) fn device_ptr(&self) -> *mut c_void {
        self.device as *mut c_void
    }

    /// Global row of the next router launch's row 0 (see [`RouterFault::row`]).
    pub(crate) fn row_base(&self) -> usize {
        self.row_base.load(Ordering::Relaxed) as usize
    }

    pub(crate) fn set_row_base(&self, row: usize) {
        self.row_base.store(row as u64, Ordering::Relaxed);
    }

    pub(crate) fn peek(&self, device_id: i32) -> Option<RouterFault> {
        // SAFETY: `host` is the live allocation made in `alloc`; it is freed
        // only by `free`, which consumes `self`. The device writes it with
        // system-scope stores, so the access is volatile.
        let code = unsafe { std::ptr::read_volatile(self.host as *const u64) };
        RouterFault::decode(device_id, code)
    }

    /// Read and clear. A store that lands between the read and the clear is
    /// lost, which only drops a second record while a first is being reported.
    pub(crate) fn take(&self, device_id: i32) -> Option<RouterFault> {
        let fault = self.peek(device_id);
        if fault.is_some() {
            self.clear();
        }
        fault
    }

    pub(crate) fn clear(&self) {
        // SAFETY: as in `peek`.
        unsafe { std::ptr::write_volatile(self.host as *mut u64, 0) };
    }

    /// Release the word. The caller must have synchronized the device.
    pub(crate) fn free(self, hip: &HipRuntime) -> HipResult<()> {
        hip.host_free(self.host as *mut c_void)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_round_trips_and_orders_by_row_then_layer() {
        assert_eq!(RouterFault::decode(3, 0), None);
        let code = RouterFault::encode(39, 1_234_567);
        assert_eq!(
            RouterFault::decode(3, code),
            Some(RouterFault {
                device_id: 3,
                layer: 39,
                row: 1_234_567
            })
        );
        assert!(RouterFault::encode(40, 5) < RouterFault::encode(0, 6));
        assert!(RouterFault::encode(2, 5) < RouterFault::encode(3, 5));
        assert_eq!(RouterFault::encode(0, 0), (1 << 24) | 1);
    }

    #[test]
    fn message_names_layer_and_row() {
        let message = RouterFault {
            device_id: 1,
            layer: 7,
            row: 42,
        }
        .to_string();
        for needle in ["GPU 1", "layer 7", "row 42", "failed closed"] {
            assert!(
                message.contains(needle),
                "{needle:?} missing from {message:?}"
            );
        }
    }
}
