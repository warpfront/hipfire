// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Error types for HIP runtime operations.

use std::ffi::CStr;
use std::fmt;

/// Raw HIP error code.
pub type HipErrorCode = u32;

/// HIP operation result.
pub type HipResult<T> = Result<T, HipError>;

/// `hipErrorInvalidImage` — the device code object handed to `hipModuleLoad`
/// is not valid for this GPU (wrong ISA, or a stale cross-build/cross-toolchain
/// `.hsaco` left in a shared kernel cache). Recoverable by recompiling from source.
pub const HIP_ERROR_INVALID_IMAGE: HipErrorCode = 200;
pub const HIP_ERROR_PEER_ACCESS_UNSUPPORTED: HipErrorCode = 217;
pub const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorCode = 704;
pub const HIP_ERROR_PEER_ACCESS_NOT_ENABLED: HipErrorCode = 705;

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
}
