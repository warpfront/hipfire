//! Test-only serialization of the one device shared by every GPU unit test in
//! this crate.
//!
//! `cargo test -p rdna-compute --lib` runs tests on parallel threads of one
//! process. GPU tests then interfere: one test's open stream capture rejects
//! another's legacy-stream copies ("operation would make the legacy stream
//! depend on a capturing blocking stream"), and allocator/VRAM bookkeeping
//! assertions (byte-exact `hipMemGetInfo` free across a window) break when a
//! sibling allocates meanwhile. Every GPU unit test therefore opens its device
//! through [`init`], which holds [`GPU_TEST_LOCK`] for the handle's lifetime.
//! Poisoning is ignored: a panic in one test must not fail the rest.

use crate::Gpu;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The one lock serializing every GPU unit test in the crate.
pub(crate) static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Take [`GPU_TEST_LOCK`] directly, for tests that build their device some
/// other way. Bind the guard for the test's duration.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    GPU_TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A device handle that holds [`GPU_TEST_LOCK`] until dropped. Fields drop in
/// declaration order, so the device is torn down before the lock is released.
pub(crate) struct TestGpu {
    gpu: Gpu,
    _guard: MutexGuard<'static, ()>,
}

impl Deref for TestGpu {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

impl DerefMut for TestGpu {
    fn deref_mut(&mut self) -> &mut Gpu {
        &mut self.gpu
    }
}

/// [`Gpu::init`] under [`GPU_TEST_LOCK`]. The lock is taken first, so device
/// initialization itself is also serialized.
pub(crate) fn init() -> crate::HipResult<TestGpu> {
    let guard = lock();
    Ok(TestGpu {
        gpu: Gpu::init()?,
        _guard: guard,
    })
}
