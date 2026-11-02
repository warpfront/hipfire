// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Error types for HIP runtime operations.

use std::ffi::CStr;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Raw HIP error code.
pub type HipErrorCode = u32;

/// HIP operation result.
pub type HipResult<T> = Result<T, HipError>;

/// `hipErrorOutOfMemory` — the device allocator could not satisfy the request.
pub const HIP_ERROR_OUT_OF_MEMORY: HipErrorCode = 2;

/// `hipErrorInvalidImage` — the device code object handed to `hipModuleLoad`
/// is not valid for this GPU (wrong ISA, or a stale cross-build/cross-toolchain
/// `.hsaco` left in a shared kernel cache). Recoverable by recompiling from source.
pub const HIP_ERROR_INVALID_IMAGE: HipErrorCode = 200;
pub const HIP_ERROR_PEER_ACCESS_UNSUPPORTED: HipErrorCode = 217;
pub const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorCode = 704;
pub const HIP_ERROR_PEER_ACCESS_NOT_ENABLED: HipErrorCode = 705;

/// `hipErrorIllegalAddress` — the raw status of a device-side memory fault.
pub const HIP_ERROR_ILLEGAL_ADDRESS: HipErrorCode = 700;

/// Process-wide "first illegal address" observer registry.
///
/// Every nonzero HIP status becomes a [`HipError`] through [`HipError::new`] or
/// [`HipError::from_code`] (`HipRuntime::check`, `hipDeviceEnablePeerAccess`,
/// the `hipHostFree` drop path, VMM `hipMemCreate`/`Map`/`SetAccess`/`Unmap`/
/// `Release`, module load/get-function/launch — all of them funnel through
/// `check`). Those two constructors call [`observe_raw_code`], as do the raw
/// statuses that never become a `HipError`: `HipRuntime::last_error()` (return
/// unchanged) and the ignored returns in the `PinnedHostBuffer` /
/// `PinnedTransferBuffer` `Drop` paths. Each costs one integer compare unless
/// the code is 700. The first 700 of the process claims `fired` and runs the
/// registered hooks once; later 700s return immediately. With no hook
/// registered the claim is still consumed but nothing runs, and no HIP/HSA call
/// is ever made by this module.
struct FirstFault {
    fired: AtomicBool,
    hooks: Mutex<Vec<fn()>>,
}

impl FirstFault {
    const fn new() -> Self {
        Self {
            fired: AtomicBool::new(false),
            hooks: Mutex::new(Vec::new()),
        }
    }

    fn register(&self, hook: fn()) {
        self.hooks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(hook);
    }

    #[inline]
    fn observe(&self, code: HipErrorCode) {
        if code == HIP_ERROR_ILLEGAL_ADDRESS {
            self.fire_once();
        }
    }

    #[cold]
    #[inline(never)]
    fn fire_once(&self) {
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        // Move the registered hooks out under the lock, run them without it:
        // hooks may register more hooks (never run: the claim is spent) or
        // fault again (the second 700 sees `fired` and returns).
        let hooks = std::mem::take(
            &mut *self
                .hooks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for hook in hooks {
            // A panicking observer must neither skip the remaining observers
            // nor change the error the caller is about to receive.
            let _ = std::panic::catch_unwind(hook);
        }
    }
}

static FIRST_ILLEGAL_ADDRESS: FirstFault = FirstFault::new();

/// Register `hook` to run once, on the first raw `hipErrorIllegalAddress`
/// (700) this process converts into a [`HipError`]. Registrations accumulate
/// (the diagnostic and daemon layers each register their own; nothing is
/// overwritten) and every hook runs, in registration order, on the thread that
/// observed the fault, without any registry lock held.
///
/// Register at startup/load, before the HIP calls of interest: a hook added
/// after the first 700 never runs. The hook must not itself rely on HIP
/// error propagation; a panic inside it is caught and discarded.
pub fn register_first_illegal_address_hook(hook: fn()) {
    FIRST_ILLEGAL_ADDRESS.register(hook);
}

/// Observe a raw HIP status at a conversion boundary that does not build a
/// [`HipError`] (`last_error()`, ignored statuses in `Drop` paths).
#[inline]
pub(crate) fn observe_raw_code(code: HipErrorCode) {
    FIRST_ILLEGAL_ADDRESS.observe(code);
}

/// Callsite context for a failed kernel launch: which kernel was being
/// launched and where the launch was issued from.
///
/// Boxed behind [`HipError::context`] so the common `HipResult<()>` path
/// carries no heap payload — only launch failures that opt in via
/// [`HipError::with_kernel`] allocate.
#[derive(Debug, Clone)]
pub struct LaunchContext {
    /// Kernel/function name passed to the launch helper.
    pub kernel: String,
    /// Source file that issued the launch (`Location::caller` of `with_kernel`).
    pub file: &'static str,
    /// Source line that issued the launch.
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct HipError {
    pub code: HipErrorCode,
    pub message: String,
    /// Launch attribution. `None` for non-launch errors (alloc, memcpy, sync).
    pub context: Option<Box<LaunchContext>>,
}

impl HipError {
    pub fn new(code: HipErrorCode, context: &str) -> Self {
        observe_raw_code(code);
        Self {
            code,
            message: format!("{context} (hipError={code})"),
            context: None,
        }
    }

    /// Attach launch attribution: the kernel being launched and the callsite
    /// that issued it. `#[track_caller]` makes `file:line` the *caller's*
    /// location, so each launch funnel calls this once and the recorded
    /// callsite is the dispatch line, not this helper.
    #[track_caller]
    pub fn with_kernel(mut self, kernel: &str) -> Self {
        let loc = std::panic::Location::caller();
        self.context = Some(Box::new(LaunchContext {
            kernel: kernel.to_string(),
            file: loc.file(),
            line: loc.line(),
        }));
        self
    }

    pub(crate) fn from_code(
        code: HipErrorCode,
        context: &str,
        get_string: Option<&unsafe extern "C" fn(u32) -> *const i8>,
    ) -> Self {
        observe_raw_code(code);
        let detail = get_string
            .and_then(|f| {
                // SAFETY: `f` is hipGetErrorString from a loaded HipRuntime; it
                // returns a static NUL-terminated message or null for unknown codes.
                let ptr = unsafe { f(code) };
                if ptr.is_null() {
                    None
                } else {
                    Some(
                        // SAFETY: non-null pointer from hipGetErrorString points at a
                        // static C string valid for the process lifetime.
                        unsafe { CStr::from_ptr(ptr) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
            })
            .unwrap_or_else(|| format!("error code {code}"));
        Self {
            code,
            message: format!("{context}: {detail}"),
            context: None,
        }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HipError({}): {}", self.code, self.message)?;
        if let Some(ctx) = self.context.as_deref() {
            write!(f, " [kernel={} at {}:{}]", ctx.kernel, ctx.file, ctx.line)?;
        }
        Ok(())
    }
}

impl std::error::Error for HipError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_without_context_is_unchanged() {
        let e = HipError::new(719, "hipModuleLaunchKernel");
        assert_eq!(
            e.to_string(),
            "HipError(719): hipModuleLaunchKernel (hipError=719)"
        );
    }

    #[test]
    fn display_with_kernel_names_kernel_and_callsite() {
        let e = HipError::new(719, "hipModuleLaunchKernel").with_kernel("gemv_hfq4g256");
        let s = e.to_string();
        assert!(s.contains("HipError(719)"), "keeps the status: {s}");
        assert!(s.contains("gemv_hfq4g256"), "names the kernel: {s}");
        assert!(s.contains("error.rs"), "names the callsite file: {s}");
    }

    use std::sync::atomic::AtomicUsize;

    #[test]
    fn first_700_runs_every_hook_once_and_second_700_is_silent() {
        static A: AtomicUsize = AtomicUsize::new(0);
        static B: AtomicUsize = AtomicUsize::new(0);
        let reg = FirstFault::new();
        reg.register(|| {
            A.fetch_add(1, Ordering::SeqCst);
        });
        reg.register(|| {
            B.fetch_add(1, Ordering::SeqCst);
        });
        reg.observe(0);
        reg.observe(719);
        reg.observe(1);
        assert_eq!((A.load(Ordering::SeqCst), B.load(Ordering::SeqCst)), (0, 0));
        reg.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        assert_eq!((A.load(Ordering::SeqCst), B.load(Ordering::SeqCst)), (1, 1));
        reg.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        reg.observe(719);
        assert_eq!((A.load(Ordering::SeqCst), B.load(Ordering::SeqCst)), (1, 1));
    }

    #[test]
    fn no_hooks_registered_is_a_silent_noop() {
        let reg = FirstFault::new();
        reg.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        reg.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        assert!(reg.fired.load(Ordering::SeqCst));
    }

    #[test]
    fn panicking_hook_does_not_block_later_hooks() {
        static RAN: AtomicUsize = AtomicUsize::new(0);
        let reg = FirstFault::new();
        reg.register(|| panic!("observer bug"));
        reg.register(|| {
            RAN.fetch_add(1, Ordering::SeqCst);
        });
        reg.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        assert_eq!(RAN.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn hook_may_register_and_refault_without_deadlock_or_second_run() {
        static RAN: AtomicUsize = AtomicUsize::new(0);
        static REG: FirstFault = FirstFault::new();
        REG.register(|| {
            RAN.fetch_add(1, Ordering::SeqCst);
            REG.register(|| {});
            REG.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        });
        REG.observe(HIP_ERROR_ILLEGAL_ADDRESS);
        assert_eq!(RAN.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn concurrent_700s_fire_exactly_once() {
        static RAN: AtomicUsize = AtomicUsize::new(0);
        static REG: FirstFault = FirstFault::new();
        REG.register(|| {
            RAN.fetch_add(1, Ordering::SeqCst);
        });
        let threads: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(|| REG.observe(HIP_ERROR_ILLEGAL_ADDRESS)))
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(RAN.load(Ordering::SeqCst), 1);
    }

    /// The only test touching the process-global registry: both conversion
    /// constructors detect 700, and the second 700 records nothing.
    #[test]
    fn constructors_detect_700_once_through_the_global_registry() {
        static RAN: AtomicUsize = AtomicUsize::new(0);
        register_first_illegal_address_hook(|| {
            RAN.fetch_add(1, Ordering::SeqCst);
        });
        let _ = HipError::new(1, "not a fault");
        let _ = HipError::from_code(719, "not a fault", None);
        assert_eq!(RAN.load(Ordering::SeqCst), 0);
        let e = HipError::from_code(HIP_ERROR_ILLEGAL_ADDRESS, "hipMemSetAccess", None);
        assert_eq!(e.code, 700);
        assert_eq!(RAN.load(Ordering::SeqCst), 1);
        let _ = HipError::new(HIP_ERROR_ILLEGAL_ADDRESS, "hipHostFree");
        let _ = HipError::from_code(HIP_ERROR_ILLEGAL_ADDRESS, "hipModuleLaunchKernel", None);
        assert_eq!(RAN.load(Ordering::SeqCst), 1);
    }
}
