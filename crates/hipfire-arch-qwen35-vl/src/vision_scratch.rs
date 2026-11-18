// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Own every temporary until its last submitted use or a checked rollback.
//! Kernels only receive borrowed views. Success keeps the existing single-
//! stream pool reuse; an early return fences before returning live owners.

use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ops::{Deref, DerefMut};

pub(super) trait Backend {
    type Tensor;
    type View;
    fn alloc(&mut self, shape: &[usize], dtype: DType) -> HipResult<Self::Tensor>;
    fn view(owner: &Self::Tensor) -> Self::View;
    fn matches(owner: &Self::Tensor, view: &Self::View) -> bool;
    fn upload(&mut self, view: &Self::View, data: &[f32]) -> HipResult<()>;
    fn release(&mut self, owner: Self::Tensor) -> HipResult<()>;
    fn synchronize(&mut self) -> HipResult<()>;
}

impl Backend for Gpu {
    type Tensor = GpuTensor;
    type View = GpuTensor;
    fn alloc(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.alloc_tensor(shape, dtype)
    }
    fn view(owner: &GpuTensor) -> GpuTensor {
        GpuTensor {
            buf: owner.buf.byte_view(0, owner.buf.size()),
            shape: owner.shape.clone(),
            dtype: owner.dtype,
        }
    }
    fn matches(owner: &GpuTensor, view: &GpuTensor) -> bool {
        owner.buf.as_ptr() == view.buf.as_ptr()
    }
    fn upload(&mut self, view: &GpuTensor, data: &[f32]) -> HipResult<()> {
        // Stage ownership before the synchronous copy. Gpu::upload_f32's
        // alloc_then_init returns its owner to the pool on copy failure,
        // which would bypass this scope's required failure fence.
        // SAFETY: the entire initialized f32 slice is viewed as bytes; its
        // byte length is exactly size_of_val(data) and remains live for copy.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        self.hip.memcpy_htod(&view.buf, bytes)
    }
    fn release(&mut self, owner: GpuTensor) -> HipResult<()> {
        self.free_tensor(owner)
    }
    fn synchronize(&mut self) -> HipResult<()> {
        self.hip.device_synchronize()
    }
}

pub(super) struct Scratch<'a, B: Backend = Gpu> {
    backend: &'a mut B,
    owners: Vec<B::Tensor>,
}

impl<'a, B: Backend> Scratch<'a, B> {
    pub(super) fn new(backend: &'a mut B) -> Self {
        Self {
            backend,
            owners: Vec::new(),
        }
    }

    pub(super) fn adopt(&mut self, owner: B::Tensor) -> B::View {
        let view = B::view(&owner);
        debug_assert!(!self.owners.iter().any(|t| B::matches(t, &view)));
        self.owners.push(owner);
        view
    }

    pub(super) fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> HipResult<B::View> {
        shape
            .iter()
            .try_fold(dtype.size(), |bytes, &n| bytes.checked_mul(n))
            .ok_or_else(|| HipError::new(1, "vision scratch allocation size overflow"))?;
        let owner = self.backend.alloc(shape, dtype)?;
        Ok(self.adopt(owner))
    }

    pub(super) fn upload_f32(&mut self, data: &[f32], shape: &[usize]) -> HipResult<B::View> {
        let len = shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim));
        if len != Some(data.len()) {
            return Err(HipError::new(1, "vision upload shape differs from payload"));
        }
        let view = self.alloc_tensor(shape, DType::F32)?;
        self.backend.upload(&view, data)?;
        Ok(view)
    }

    /// Transfer a successful prologue's real owner into the resumable job.
    /// The view is consumed so the caller cannot accidentally free it again.
    pub(super) fn take(&mut self, view: B::View) -> B::Tensor {
        let index = self
            .owners
            .iter()
            .position(|t| B::matches(t, &view))
            .expect("vision scratch view must name a live owned tensor");
        self.owners.swap_remove(index)
    }

    /// Only call after the last use was successfully submitted on the same
    /// stream. Failure paths use rollback instead of this immediate release.
    pub(super) fn free_tensor(&mut self, view: B::View) -> HipResult<()> {
        let owner = self.take(view);
        self.backend.release(owner)
    }

    pub(super) fn rollback(&mut self) -> HipResult<()> {
        if self.owners.is_empty() {
            return Ok(());
        }
        if let Err(error) = self.backend.synchronize() {
            let count = self.owners.len();
            // A failed fence is not cancellation. Quarantine these owners
            // until process exit: neither pool reuse nor hipFree is safe.
            for owner in self.owners.drain(..) {
                std::mem::forget(owner);
            }
            return Err(HipError::new(
                error.code,
                &format!(
                    "vision scratch fence failed: {error}; quarantined {count} GPU allocations"
                ),
            ));
        }
        let mut first = None;
        for owner in self.owners.drain(..).rev() {
            if let Err(error) = self.backend.release(owner) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

impl<B: Backend> Drop for Scratch<'_, B> {
    fn drop(&mut self) {
        if let Err(error) = self.rollback() {
            eprintln!("vision scratch rollback failed: {error}");
        }
    }
}

impl Deref for Scratch<'_> {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        self.backend
    }
}
impl DerefMut for Scratch<'_> {
    fn deref_mut(&mut self) -> &mut Gpu {
        self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[derive(Default)]
    struct Fake {
        operations: usize,
        fail_at: usize,
        live: BTreeSet<usize>,
        released: Vec<usize>,
        syncs: usize,
        fail_sync: bool,
        fail_release: Option<usize>,
    }
    impl Fake {
        fn operation(&mut self) -> HipResult<()> {
            self.operations += 1;
            if self.operations == self.fail_at {
                Err(HipError::new(1, "injected failure"))
            } else {
                Ok(())
            }
        }
    }
    impl Backend for Fake {
        type Tensor = usize;
        type View = usize;
        fn alloc(&mut self, _: &[usize], _: DType) -> HipResult<usize> {
            self.operation()?;
            self.live.insert(self.operations);
            Ok(self.operations)
        }
        fn view(owner: &usize) -> usize {
            *owner
        }
        fn matches(owner: &usize, view: &usize) -> bool {
            owner == view
        }
        fn upload(&mut self, _: &usize, _: &[f32]) -> HipResult<()> {
            self.operation()
        }
        fn release(&mut self, owner: usize) -> HipResult<()> {
            assert!(self.live.remove(&owner), "double release {owner}");
            self.released.push(owner);
            if self.fail_release == Some(owner) {
                Err(HipError::new(1, "free failed"))
            } else {
                Ok(())
            }
        }
        fn synchronize(&mut self) -> HipResult<()> {
            self.syncs += 1;
            if self.fail_sync {
                Err(HipError::new(1, "fence failed"))
            } else {
                Ok(())
            }
        }
    }
    fn pipeline(fake: &mut Fake) -> HipResult<()> {
        let mut scratch = Scratch::new(fake);
        let input = scratch.upload_f32(&[1.0; 4], &[4])?;
        let tmp = scratch.alloc_tensor(&[4], DType::F32)?;
        scratch.backend.operation()?; // a submitted kernel
        scratch.free_tensor(input)?;
        let output = scratch.alloc_tensor(&[4], DType::F32)?;
        scratch.backend.operation()?;
        scratch.free_tensor(tmp)?;
        scratch.free_tensor(output)
    }

    #[test]
    fn allocation_upload_and_kernel_failures_release_all_live_owners() {
        for fail_at in 1..=6 {
            let mut fake = Fake {
                fail_at,
                ..Fake::default()
            };
            assert!(pipeline(&mut fake).is_err(), "failure point {fail_at}");
            assert!(
                fake.live.is_empty(),
                "failure point {fail_at}: {:?}",
                fake.live
            );
            assert_eq!(fake.syncs, usize::from(fail_at > 1));
        }
    }
    #[test]
    fn normal_pool_reuse_and_prologue_commit_add_no_fences() {
        let mut fake = Fake::default();
        pipeline(&mut fake).unwrap();
        assert_eq!(fake.syncs, 0);
        assert!(fake.live.is_empty());
        let owner = {
            let mut scratch = Scratch::new(&mut fake);
            let view = scratch.alloc_tensor(&[4], DType::F32).unwrap();
            scratch.take(view)
        };
        assert_eq!(fake.syncs, 0);
        assert!(fake.live.contains(&owner));
        let mut finish = Scratch::new(&mut fake);
        finish.adopt(owner);
        finish.rollback().unwrap();
        assert!(finish.backend.live.is_empty());
        assert_eq!(finish.backend.syncs, 1);
    }
    #[test]
    fn failed_fence_never_returns_pending_allocations_to_pool() {
        let mut fake = Fake {
            fail_at: 4,
            fail_sync: true,
            ..Fake::default()
        };
        assert!(pipeline(&mut fake).is_err());
        assert_eq!(fake.syncs, 1);
        assert!(fake.released.is_empty());
        assert_eq!(fake.live.len(), 2);
    }
    #[test]
    fn rollback_continues_after_one_release_fails() {
        let mut fake = Fake {
            fail_release: Some(2),
            ..Fake::default()
        };
        let mut scratch = Scratch::new(&mut fake);
        scratch.alloc_tensor(&[4], DType::F32).unwrap();
        scratch.alloc_tensor(&[4], DType::F32).unwrap();
        assert!(scratch.rollback().is_err());
        assert!(scratch.backend.live.is_empty());
        assert_eq!(scratch.backend.released, [2, 1]);
    }
}
