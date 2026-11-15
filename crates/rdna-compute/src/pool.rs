// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU memory pool — eliminates hipMalloc/hipFree overhead in the hot loop.
//! Pre-allocates buffers of common sizes and reuses them via a free list.

use hip_bridge::{DeviceBuffer, HipResult, HipRuntime, HIP_ERROR_OUT_OF_MEMORY};
use std::collections::HashMap;

const MIN_ALLOC: usize = 256;

/// The device allocator behind the pool: `HipRuntime` in production, a
/// host-only fake in the unit tests.
trait DeviceMemory {
    fn malloc(&self, size: usize) -> HipResult<DeviceBuffer>;
    fn free(&self, buf: DeviceBuffer) -> HipResult<()>;
}

impl DeviceMemory for HipRuntime {
    fn malloc(&self, size: usize) -> HipResult<DeviceBuffer> {
        HipRuntime::malloc(self, size)
    }

    fn free(&self, buf: DeviceBuffer) -> HipResult<()> {
        HipRuntime::free(self, buf)
    }
}

/// A pool of GPU buffers, bucketed by size.
/// Requesting a buffer returns one from the pool (if available) or allocates new.
/// Returning a buffer puts it back in the pool for reuse.
pub struct GpuPool {
    /// Free buffers bucketed by size (rounded up to power of 2)
    free_lists: HashMap<usize, Vec<DeviceBuffer>>,
    /// Total bytes currently allocated (for diagnostics)
    pub total_allocated: usize,
    pub total_reused: usize,
    pub total_new: usize,
    /// Times a drain returned at least one pooled buffer to HIP (the
    /// out-of-memory retries, the VMM pre-map drain and the unload drain).
    pub drains: u64,
    /// Test seam: allocations still to let through before one is forced to miss
    /// the free lists and fail as an injected `hipErrorOutOfMemory`, so the real
    /// retry in `alloc_in` (drain every free list, allocate again once) runs on
    /// demand. `None` outside a test that armed it; one match on the alloc path.
    oom_countdown: Option<u32>,
    /// Injected out-of-memory failures consumed so far.
    pub injected_oom: u64,
}

impl GpuPool {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            total_new: 0,
            drains: 0,
            oom_countdown: None,
            injected_oom: 0,
        }
    }

    /// Free-list bucket key. Buffers group by power-of-2 bucket so a
    /// decode-hot scratch of size X reliably finds a reusable slot from
    /// a previous step. The bucket is ONLY a reuse key — the actual HIP
    /// allocation uses the exact requested size (see `alloc`), so there
    /// is no VRAM padding waste.
    fn bucket_key(size: usize) -> usize {
        const MIN: usize = 256;
        if size <= MIN {
            MIN
        } else {
            size.next_power_of_two()
        }
    }

    /// Get a buffer of at least `size` bytes. Reuses the smallest pooled
    /// buffer in the same bucket that is large enough; otherwise allocates
    /// from HIP at the EXACT requested size.
    ///
    /// Exact HIP allocation matters for large buffers: previously,
    /// target's 15 GB of per-layer weights on 27B sprawled into
    /// ~100–500 MB power-of-2 buckets that each padded up to 2×,
    /// leaving no contiguous room for the ~3.5 GB draft to load on
    /// 24 GB cards. With exact sizing the padding is zero, all
    /// intended bytes are used, and the draft fits.
    ///
    /// Pooled buffers are invisible to HIP, so a `hipMalloc` can run out of
    /// memory while gigabytes sit in other buckets. On an out-of-memory
    /// error with buffers pooled, every free list is returned to HIP and
    /// the allocation is retried exactly once.
    pub fn alloc(&mut self, hip: &HipRuntime, size: usize) -> HipResult<DeviceBuffer> {
        self.alloc_in(hip, size)
    }

    fn alloc_in<M: DeviceMemory>(&mut self, mem: &M, size: usize) -> HipResult<DeviceBuffer> {
        // Test seam: the armed allocation is forced down the miss path below and
        // fails as an injected hipErrorOutOfMemory, whatever the free lists hold.
        let injected = match self.oom_countdown {
            Some(0) => {
                self.oom_countdown = None;
                self.injected_oom += 1;
                true
            }
            Some(n) => {
                self.oom_countdown = Some(n - 1);
                false
            }
            None => false,
        };
        let bucket = Self::bucket_key(size);
        if let Some(list) = self.free_lists.get_mut(&bucket).filter(|_| !injected) {
            // Best fit, newest first: a rebuild with the same request sizes
            // then reuses every pooled buffer whatever order they were freed
            // in. LIFO pop could hand a small request the large buffer and
            // leave the large request only an undersized one (G4.1 growth).
            let mut best: Option<usize> = None;
            for (i, buf) in list.iter().enumerate().rev() {
                if buf.size() >= size && best.is_none_or(|b| buf.size() < list[b].size()) {
                    best = Some(i);
                    if buf.size() == size {
                        break;
                    }
                }
            }
            if let Some(i) = best {
                self.total_reused += 1;
                return Ok(list.swap_remove(i));
            }
            // Every pooled buffer here is undersized: return them to HIP —
            // better to re-allocate at the right size than to carry
            // undersized buffers around.
            for buf in list.drain(..) {
                let _ = mem.free(buf);
            }
        }
        // No suitable buffer — allocate at exact requested size. Round up
        // to the nearest 256 B for alignment; HIP may round further but
        // this keeps our accounting honest and avoids tiny-alloc churn.
        let actual = if size < MIN_ALLOC { MIN_ALLOC } else { size };
        self.total_new += 1;
        self.total_allocated += actual;
        let first = if injected {
            Err(hip_bridge::HipError::new(
                HIP_ERROR_OUT_OF_MEMORY,
                "injected hipMalloc out of memory (test seam)",
            ))
        } else {
            mem.malloc(actual)
        };
        match first {
            Err(e) if e.code == HIP_ERROR_OUT_OF_MEMORY && self.pooled_bytes() > 0 => {
                let pooled = self.pooled_bytes();
                self.drain_in(mem);
                eprintln!(
                    "GpuPool: hipMalloc({actual} B) out of memory; returned {pooled} B of \
                     pooled buffers to HIP and retrying once"
                );
                mem.malloc(actual)
            }
            r => r,
        }
    }

    /// Arm (`Some(n)`) or disarm (`None`) the injected out-of-memory seam: the
    /// allocation after `n` further ones is forced to miss the free lists and
    /// fail as `hipErrorOutOfMemory` before `hipMalloc`. An empty pool fails it
    /// exactly as a real out-of-memory with nothing to return does; a non-empty
    /// pool takes the production drain-and-retry-once path.
    pub fn inject_oom(&mut self, after_allocs: Option<u32>) {
        self.oom_countdown = after_allocs;
    }

    /// Bytes parked in the free lists (allocated from HIP, owned by no tensor).
    pub(crate) fn pooled_bytes(&self) -> usize {
        self.free_lists
            .values()
            .flatten()
            .map(DeviceBuffer::size)
            .sum()
    }

    /// Return a buffer to the pool for reuse. The buffer's ACTUAL
    /// capacity is what gets reused — we key the free-list by the
    /// power-of-2 bucket so same-size-shaped requests hit the same
    /// slot.
    pub fn free(&mut self, buf: DeviceBuffer) {
        let bucket = Self::bucket_key(buf.size());
        self.free_lists.entry(bucket).or_default().push(buf);
    }

    /// Actually free all pooled buffers (call on cleanup).
    pub fn drain(&mut self, hip: &HipRuntime) {
        self.drain_in(hip);
    }

    fn drain_in<M: DeviceMemory>(&mut self, mem: &M) {
        let mut any = false;
        for (_, list) in self.free_lists.drain() {
            for buf in list {
                any = true;
                let _ = mem.free(buf);
            }
        }
        self.drains += u64::from(any);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hip_bridge::HipError;
    use std::cell::{Cell, RefCell};

    /// Host-only allocator with a fixed byte budget. Buffers are non-owning
    /// descriptors over fake addresses; nothing is ever dereferenced.
    struct FakeMemory {
        capacity: usize,
        live: RefCell<HashMap<usize, usize>>,
        next_addr: Cell<usize>,
        mallocs: Cell<usize>,
        frees: Cell<usize>,
        fail_code: Option<u32>,
    }

    impl FakeMemory {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                live: RefCell::new(HashMap::new()),
                next_addr: Cell::new(0x1000),
                mallocs: Cell::new(0),
                frees: Cell::new(0),
                fail_code: None,
            }
        }

        fn used(&self) -> usize {
            self.live.borrow().values().sum()
        }
    }

    impl DeviceMemory for FakeMemory {
        fn malloc(&self, size: usize) -> HipResult<DeviceBuffer> {
            self.mallocs.set(self.mallocs.get() + 1);
            if let Some(code) = self.fail_code {
                return Err(HipError::new(code, "fake hipMalloc"));
            }
            if self.used() + size > self.capacity {
                return Err(HipError::new(HIP_ERROR_OUT_OF_MEMORY, "fake hipMalloc"));
            }
            let addr = self.next_addr.get();
            self.next_addr.set(addr + size);
            self.live.borrow_mut().insert(addr, size);
            // SAFETY: test-only descriptor over a fake address; never dereferenced.
            Ok(unsafe { DeviceBuffer::from_raw(addr as *mut _, size) })
        }

        fn free(&self, buf: DeviceBuffer) -> HipResult<()> {
            self.frees.set(self.frees.get() + 1);
            self.live
                .borrow_mut()
                .remove(&(buf.as_ptr() as usize))
                .expect("free of a live fake buffer");
            Ok(())
        }
    }

    #[test]
    fn oom_drains_pooled_buffers_and_retries_once() {
        let mem = FakeMemory::new(4096);
        let mut pool = GpuPool::new();
        for _ in 0..2 {
            let a = pool.alloc_in(&mem, 1024).unwrap();
            let b = pool.alloc_in(&mem, 1024).unwrap();
            pool.free(a);
            pool.free(b);
        }
        assert_eq!(pool.pooled_bytes(), 2048);
        // A different bucket cannot reuse the parked 1 KiB buffers and does
        // not fit beside them: one OOM, a drain, one successful retry.
        let mallocs = mem.mallocs.get();
        let big = pool.alloc_in(&mem, 3072).unwrap();
        assert_eq!(big.size(), 3072);
        assert_eq!(mem.mallocs.get() - mallocs, 2, "exactly one retry");
        assert_eq!(pool.pooled_bytes(), 0, "free lists drained");
        assert_eq!(mem.used(), 3072);
        pool.free(big);
    }

    #[test]
    fn oom_with_empty_pool_fails_without_retry() {
        let mem = FakeMemory::new(1024);
        let mut pool = GpuPool::new();
        let err = pool.alloc_in(&mem, 2048).err().expect("must fail");
        assert_eq!(err.code, HIP_ERROR_OUT_OF_MEMORY);
        assert_eq!(mem.mallocs.get(), 1);
    }

    #[test]
    fn oom_after_drain_reports_the_retry_error() {
        let mem = FakeMemory::new(2048);
        let mut pool = GpuPool::new();
        let a = pool.alloc_in(&mem, 1024).unwrap();
        pool.free(a);
        let err = pool.alloc_in(&mem, 4096).err().expect("must fail");
        assert_eq!(err.code, HIP_ERROR_OUT_OF_MEMORY);
        assert_eq!(mem.mallocs.get(), 3, "first alloc, OOM, one retry");
        assert_eq!(mem.frees.get(), 1);
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn other_errors_do_not_drain() {
        let mut mem = FakeMemory::new(4096);
        let mut pool = GpuPool::new();
        let a = pool.alloc_in(&mem, 1024).unwrap();
        pool.free(a);
        mem.fail_code = Some(1);
        let err = pool.alloc_in(&mem, 2048).err().expect("must fail");
        assert_eq!(err.code, 1);
        assert_eq!(mem.frees.get(), 0);
        assert_eq!(pool.pooled_bytes(), 1024);
    }

    #[test]
    fn same_sizes_reuse_every_pooled_buffer_in_any_free_order() {
        // 2304 B and 4096 B share the 4096 bucket (qwen35 prefill scratch:
        // dn_z_fold_batch beside dn_q_raw_batch / dn_s_tape_f32).
        let sizes = [2304, 4096, 4096];
        for reverse_free in [false, true] {
            let mem = FakeMemory::new(1 << 20);
            let mut pool = GpuPool::new();
            let mut bufs: Vec<_> = sizes
                .iter()
                .map(|&s| pool.alloc_in(&mem, s).unwrap())
                .collect();
            if reverse_free {
                bufs.reverse();
            }
            bufs.into_iter().for_each(|b| pool.free(b));
            let rebuilt: Vec<_> = sizes
                .iter()
                .map(|&s| pool.alloc_in(&mem, s).unwrap())
                .collect();
            assert_eq!(
                mem.mallocs.get(),
                sizes.len(),
                "rebuild allocated (reverse_free={reverse_free})"
            );
            assert_eq!(mem.frees.get(), 0, "rebuild discarded a pooled buffer");
            for (buf, &size) in rebuilt.iter().zip(&sizes) {
                assert_eq!(buf.size(), size, "best fit takes the exact-size buffer");
            }
            rebuilt.into_iter().for_each(|b| pool.free(b));
        }
    }

    #[test]
    fn injected_oom_runs_the_real_drain_and_retry_and_leaves_live_buffers() {
        // Plenty of room: only the armed seam makes this allocation fail, so the
        // drain-and-retry below is the production path, not a capacity artefact.
        let mem = FakeMemory::new(1 << 20);
        let mut pool = GpuPool::new();
        let live = pool.alloc_in(&mem, 4096).unwrap();
        let live_addr = live.as_ptr() as usize;
        let parked: Vec<_> = (0..3).map(|_| pool.alloc_in(&mem, 1024).unwrap()).collect();
        parked.into_iter().for_each(|b| pool.free(b));
        assert_eq!(pool.pooled_bytes(), 3072);
        assert_eq!((pool.drains, pool.injected_oom), (0, 0));

        // The next allocation WOULD be served from the free list; the seam forces
        // it to miss, fail, drain every pooled buffer and retry once.
        pool.inject_oom(Some(0));
        let mallocs = mem.mallocs.get();
        let fresh = pool.alloc_in(&mem, 1024).unwrap();
        assert_eq!(fresh.size(), 1024);
        assert_eq!(
            mem.mallocs.get() - mallocs,
            1,
            "the injected failure never reaches hipMalloc; only the retry does"
        );
        assert_eq!(mem.frees.get(), 3, "every parked buffer returned to HIP");
        assert_eq!(pool.pooled_bytes(), 0);
        assert_eq!((pool.drains, pool.injected_oom), (1, 1));
        assert!(
            mem.live.borrow().contains_key(&live_addr),
            "a buffer a tensor still owns is not touched by the drain"
        );
        // One-shot: the next miss reaches hipMalloc without a drain.
        pool.free(fresh);
        let again = pool.alloc_in(&mem, 16384).unwrap();
        assert_eq!((pool.drains, pool.injected_oom), (1, 1));
        pool.free(again);
        pool.free(live);
    }

    #[test]
    fn injected_oom_fires_after_the_armed_number_of_allocations() {
        let mem = FakeMemory::new(1 << 20);
        let mut pool = GpuPool::new();
        let a = pool.alloc_in(&mem, 1024).unwrap();
        pool.free(a);
        pool.inject_oom(Some(2));
        let a = pool.alloc_in(&mem, 1024).unwrap();
        let b = pool.alloc_in(&mem, 2048).unwrap();
        assert_eq!((pool.drains, pool.injected_oom), (0, 0), "two let through");
        pool.free(a);
        pool.free(b);
        let c = pool.alloc_in(&mem, 2048).unwrap();
        assert_eq!((pool.drains, pool.injected_oom), (1, 1), "the third fires");
        assert_eq!(pool.pooled_bytes(), 0);
        pool.free(c);
    }

    #[test]
    fn injected_oom_with_an_empty_pool_fails_like_a_real_one() {
        let mem = FakeMemory::new(1 << 20);
        let mut pool = GpuPool::new();
        pool.inject_oom(Some(0));
        let err = pool.alloc_in(&mem, 2048).err().expect("must fail");
        assert_eq!(err.code, HIP_ERROR_OUT_OF_MEMORY);
        assert_eq!(mem.mallocs.get(), 0);
        assert_eq!((pool.drains, pool.injected_oom), (0, 1));
    }

    #[test]
    fn disarming_clears_an_unconsumed_injection() {
        let mem = FakeMemory::new(1 << 20);
        let mut pool = GpuPool::new();
        let a = pool.alloc_in(&mem, 1024).unwrap();
        pool.free(a);
        pool.inject_oom(Some(0));
        pool.inject_oom(None);
        let a = pool.alloc_in(&mem, 1024).unwrap();
        assert_eq!((pool.drains, pool.injected_oom), (0, 0));
        assert_eq!(mem.mallocs.get(), 1, "served from the free list");
        pool.free(a);
    }
}
