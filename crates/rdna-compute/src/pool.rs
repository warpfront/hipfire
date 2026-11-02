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
}

impl GpuPool {
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            total_new: 0,
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
        let bucket = Self::bucket_key(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
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
        match mem.malloc(actual) {
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
        for (_, list) in self.free_lists.drain() {
            for buf in list {
                let _ = mem.free(buf);
            }
        }
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
}
