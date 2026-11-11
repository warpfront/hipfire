// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Cold-cache timing shared by `hip_loop` (lane nodes) and `pm_loop` (`iu4_gemm`, `qkvzaset`):
//! the flush buffers and eviction, the rotating weight-pool sizing, and the record fields.
//!
//! Production runs a verify GEMM against a cold L2 / MALL / TLB / instruction cache: the other ~25
//! kernels of a decode step evict it between two launches of the same GEMM, and each layer reads its
//! OWN weights. A hot loop (the same weights, back to back) lets a <= 64 MiB weight matrix stay
//! resident in the 64 MB last-level cache and under-reads DRAM-bound kernels (`pm_loop`, N=64
//! `full_add_b1s`: 100.6 -> 62.6 us for a change that read 122.3 -> 98.4 us in the model).
//!
//! * **Flush**: a 2 x [`FLUSH_BYTES`] device-to-device copy after every timed dispatch, enqueued on
//!   the default stream OUTSIDE the per-dispatch hipEvent interval ([`Flush::evict`]).
//! * **Pool**: distinct seeded weight sets whose total exceeds [`POOL_BYTES`]; dispatch `i` uses set
//!   `i % sets` ([`pool_sets`], odd so ABBA arms do not alias set parity).
//!
//! `pm_loop` composes them as a [`ColdMode`]: `hot` (neither), `pool` (weights cold only: the pool
//! exceeds L2 + MALL, activations, instructions and TLB entries of the small buffers stay warm),
//! `flush-l2` (the pool plus a copy of 2 x the device's L2 per buffer: evicts L2 only) and `flush`
//! (the pool plus the full 2 x 96 MiB copy: also MALL, TLB, instruction and scalar caches).
//!
//! Included with `#[path = "support/cold.rs"] mod cold;`.
#![allow(dead_code)]

use hip_bridge::{DeviceBuffer, HipRuntime};
use serde_json::{json, Value};

/// A timing pool's weight sets together exceed this many bytes (production runs against a cold
/// last-level cache: the other ~25 kernels between two launches of a decode step evict it).
pub const POOL_BYTES: usize = 192 << 20;
/// Bytes of each of the two device-to-device flush buffers.
pub const FLUSH_BYTES: usize = 96 << 20;
/// Most weight sets a pool holds.
pub const MAX_SETS: usize = 161;
/// `hipDeviceAttributeL2CacheSize` in this ROCm's `hip_runtime_api.h` (`enum hipDeviceAttribute_t`:
/// the 20th member of the CUDA-compatible block, after `hipDeviceAttributeKernelExecTimeout`).
pub const HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: i32 = 19;
/// gfx1201 (Navi 48, R9700 / RX 9070 XT) L2 per the AMD specification, used when the device
/// attribute is missing or implausible. (`rdna_compute::profiler`'s table lists 4 MB for RDNA4.)
pub const L2_DEFAULT_BYTES: usize = 8 << 20;

/// How a timing loop treats the caches between dispatches. Ordered by how much is evicted;
/// `--flush` and `--flush-l2` are exclusive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColdMode {
    /// The same weights back to back, nothing evicted.
    #[default]
    Hot,
    /// Rotating weight pool only.
    Pool,
    /// The pool plus a copy of 2 x L2 per buffer after every dispatch.
    FlushL2,
    /// The pool plus the full 2 x [`FLUSH_BYTES`] copy after every dispatch.
    Flush,
}

impl ColdMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Pool => "pool",
            Self::FlushL2 => "flush-l2",
            Self::Flush => "flush",
        }
    }
    /// The CLI flag that selects the mode (empty for hot).
    pub fn flag(self) -> &'static str {
        match self {
            Self::Hot => "",
            Self::Pool => "--pool",
            Self::FlushL2 => "--flush-l2",
            Self::Flush => "--flush",
        }
    }
    /// The timing pool is rotated.
    pub fn pool(self) -> bool {
        self != Self::Hot
    }
    /// Caches are evicted between dispatches.
    pub fn flushes(self) -> bool {
        matches!(self, Self::FlushL2 | Self::Flush)
    }
    /// Two flags together: the stronger mode, except `--flush` with `--flush-l2`.
    pub fn combine(self, other: Self) -> Result<Self, String> {
        match (self, other) {
            (Self::Flush, Self::FlushL2) | (Self::FlushL2, Self::Flush) => {
                Err("--flush and --flush-l2 are exclusive".into())
            }
            (a, b) => Ok(a.max(b)),
        }
    }
}

/// The device's L2 size: the HIP attribute when it is plausible (1 MiB ..= 1 GiB), else
/// [`L2_DEFAULT_BYTES`]. Returns `(bytes, source)`; the source is recorded, never hidden.
pub fn l2_bytes(hip: &HipRuntime, device: i32) -> (usize, &'static str) {
    match hip.get_device_attribute(HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device) {
        Ok(v) if (1 << 20..=1 << 30).contains(&(v as i64)) => (v as usize, "device attribute"),
        _ => (L2_DEFAULT_BYTES, "documented gfx1201 default (8 MiB)"),
    }
}

/// Bytes of each flush buffer of `mode` given the device L2 (`None` for modes that do not flush).
pub fn flush_bytes(mode: ColdMode, l2: usize) -> Option<usize> {
    match mode {
        ColdMode::Flush => Some(FLUSH_BYTES),
        ColdMode::FlushL2 => Some(2 * l2),
        _ => None,
    }
}

/// Record the mode on a workload's `shape` object. `flush` stays the full-flush bool; the pool and
/// the L2 flush add their own keys, so shapes of different modes never match one another (a
/// `pm-census noise` group keys on the shape) and a hot shape keeps its bytes.
pub fn annotate_shape(shape: &mut Value, mode: ColdMode, sets: usize, set_bytes: usize) {
    if !mode.pool() {
        return;
    }
    shape["flush"] = json!(mode == ColdMode::Flush);
    if mode == ColdMode::FlushL2 {
        shape["flush_l2"] = json!(true);
    }
    shape["timing_pool"] = pool_json(sets, set_bytes);
}

/// How many distinct weight sets a timing pool of `set_bytes` each needs: their total exceeds
/// [`POOL_BYTES`], the count is odd (so ABBA arms do not alias set parity), clamped to 3..=161.
pub fn pool_sets(set_bytes: usize) -> usize {
    let mut n = POOL_BYTES / set_bytes.max(1) + 1;
    if n % 2 == 0 {
        n += 1;
    }
    n.clamp(3, MAX_SETS)
}

/// The record's `timing_pool` object (`hip_loop` and `pm_loop` share the key names).
pub fn pool_json(sets: usize, bytes_per_set: usize) -> Value {
    json!({"sets": sets, "bytes_per_set": bytes_per_set,
        "total_bytes": sets * bytes_per_set, "dispatch_i_uses_set": "i % sets"})
}

/// `timing.flush_method` of the full flush.
pub fn flush_method() -> String {
    flush_method_of(FLUSH_BYTES)
}

/// `timing.flush_method` of a flush with `bytes` per buffer.
pub fn flush_method_of(bytes: usize) -> String {
    format!(
        "2 x {} device-to-device copy after every timed dispatch, outside the per-dispatch hipEvent interval",
        if bytes % (1 << 20) == 0 {
            format!("{} MiB", bytes >> 20)
        } else {
            format!("{bytes} B")
        }
    )
}

/// The two flush buffers: [`Flush::evict`] copies `a` into `b`, streaming `2 x FLUSH_BYTES` through
/// L2, the last-level cache and the TLBs, and running the copy kernel through the instruction cache.
pub struct Flush {
    a: DeviceBuffer,
    b: DeviceBuffer,
    bytes: usize,
}

impl Flush {
    /// The full flush: two [`FLUSH_BYTES`] buffers.
    pub fn alloc(hip: &HipRuntime) -> Result<Self, String> {
        Self::alloc_bytes(hip, FLUSH_BYTES)
    }

    /// Two buffers of `bytes` each.
    pub fn alloc_bytes(hip: &HipRuntime, bytes: usize) -> Result<Self, String> {
        let a = hip.malloc(bytes).map_err(|e| e.to_string())?;
        match hip.malloc(bytes) {
            Ok(b) => Ok(Self { a, b, bytes }),
            Err(e) => {
                let _ = hip.free(a);
                Err(e.to_string())
            }
        }
    }

    /// Bytes of each buffer (the copy moves this many bytes read and written).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn evict(&self, hip: &HipRuntime) -> Result<(), String> {
        hip.memcpy_dtod(&self.b, &self.a, self.bytes)
            .map_err(|e| e.to_string())
    }

    pub fn free(self, hip: &HipRuntime) {
        let _ = hip.free(self.a);
        let _ = hip.free(self.b);
    }
}

/// Evict through `flush` when present (the timing loops' `evict` step).
pub fn evict(hip: &HipRuntime, flush: Option<&Flush>) -> Result<(), String> {
    match flush {
        Some(f) => f.evict(hip),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_exceeds_the_last_level_cache_with_an_odd_count() {
        for set_bytes in [
            1usize << 20,
            20 << 20,
            45 << 20,
            47_349_760,
            94_699_520,
            300 << 20,
        ] {
            let n = pool_sets(set_bytes);
            assert_eq!(n % 2, 1, "{set_bytes}");
            assert!((3..=MAX_SETS).contains(&n));
            assert!(n * set_bytes > POOL_BYTES || n == MAX_SETS, "{set_bytes}");
        }
        assert_eq!(pool_sets(0), MAX_SETS);
        assert_eq!(pool_sets(usize::MAX), 3);
    }

    #[test]
    fn cold_modes_combine_and_order() {
        use ColdMode::*;
        assert_eq!(ColdMode::default(), Hot);
        assert!(!Hot.pool() && Pool.pool() && FlushL2.pool() && Flush.pool());
        assert!(!Hot.flushes() && !Pool.flushes() && FlushL2.flushes() && Flush.flushes());
        assert_eq!(Hot.combine(Pool).unwrap(), Pool);
        assert_eq!(Pool.combine(Flush).unwrap(), Flush);
        assert_eq!(FlushL2.combine(Pool).unwrap(), FlushL2);
        assert_eq!(Flush.combine(Flush).unwrap(), Flush);
        assert!(Flush.combine(FlushL2).unwrap_err().contains("exclusive"));
        assert!(FlushL2.combine(Flush).is_err());
        assert_eq!(
            [Hot, Pool, FlushL2, Flush].map(|m| (m.name(), m.flag())),
            [
                ("hot", ""),
                ("pool", "--pool"),
                ("flush-l2", "--flush-l2"),
                ("flush", "--flush")
            ]
        );
    }

    #[test]
    fn flush_sizes_follow_the_mode() {
        let l2 = 8 << 20;
        assert_eq!(flush_bytes(ColdMode::Hot, l2), None);
        assert_eq!(flush_bytes(ColdMode::Pool, l2), None);
        assert_eq!(flush_bytes(ColdMode::FlushL2, l2), Some(16 << 20));
        assert_eq!(flush_bytes(ColdMode::Flush, l2), Some(FLUSH_BYTES));
        // The L2 copy is larger than L2, the pool is larger than L2 + the 64 MB MALL.
        assert!(2 * l2 > l2 && POOL_BYTES > l2 + (64 << 20));
        assert!(flush_method_of(16 << 20).starts_with("2 x 16 MiB device-to-device copy"));
        assert!(flush_method_of(1000).starts_with("2 x 1000 B"));
    }

    #[test]
    fn shape_annotation_keeps_hot_bytes_and_separates_modes() {
        let base = json!({"M": 1});
        let shape = |m| {
            let mut v = base.clone();
            annotate_shape(&mut v, m, 5, 100);
            v
        };
        assert_eq!(shape(ColdMode::Hot), base);
        let (p, l, f) = (
            shape(ColdMode::Pool),
            shape(ColdMode::FlushL2),
            shape(ColdMode::Flush),
        );
        assert_eq!(
            (p["flush"].clone(), l["flush"].clone(), f["flush"].clone()),
            (json!(false), json!(false), json!(true))
        );
        assert!(p.get("flush_l2").is_none() && f.get("flush_l2").is_none());
        assert_eq!(l["flush_l2"], true);
        for v in [&p, &l, &f] {
            assert_eq!(v["timing_pool"]["sets"], 5);
        }
        assert!(p != l && l != f && p != f && p != base);
    }

    #[test]
    fn record_fields_name_the_pool_and_the_flush() {
        let p = pool_json(5, 47_349_760);
        assert_eq!(p["sets"], 5);
        assert_eq!(p["total_bytes"], 5u64 * 47_349_760);
        assert_eq!(p["dispatch_i_uses_set"], "i % sets");
        assert!(flush_method().starts_with("2 x 96 MiB"));
        assert_eq!(FLUSH_BYTES, 96 << 20);
        assert_eq!(POOL_BYTES, 192 << 20);
    }
}
