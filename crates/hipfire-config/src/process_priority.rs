// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Windows process priority class (#1144). A Windows child process inherits
//! its parent's class only when that class is Idle or BelowNormal, so a daemon
//! started by a Task Scheduler task at the task default (priority 7,
//! BelowNormal) runs below every Normal-priority thread on the machine.
//! hipfire never changes the class; the daemon warns at start and
//! `hipfire diag` shows it.

use std::fmt;

/// A `GetPriorityClass` value (processthreadsapi.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorityClass {
    Idle,
    BelowNormal,
    Normal,
    AboveNormal,
    High,
    Realtime,
    /// A value outside the documented set.
    Other(u32),
}

impl PriorityClass {
    /// Map the raw `*_PRIORITY_CLASS` value.
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0x0000_0040 => Self::Idle,
            0x0000_4000 => Self::BelowNormal,
            0x0000_0020 => Self::Normal,
            0x0000_8000 => Self::AboveNormal,
            0x0000_0080 => Self::High,
            0x0000_0100 => Self::Realtime,
            other => Self::Other(other),
        }
    }

    /// Idle or BelowNormal: the classes a child inherits, and the ones that
    /// lose the CPU to any Normal-priority work.
    pub fn is_low(self) -> bool {
        matches!(self, Self::Idle | Self::BelowNormal)
    }
}

impl fmt::Display for PriorityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => f.write_str("Idle"),
            Self::BelowNormal => f.write_str("BelowNormal"),
            Self::Normal => f.write_str("Normal"),
            Self::AboveNormal => f.write_str("AboveNormal"),
            Self::High => f.write_str("High"),
            Self::Realtime => f.write_str("Realtime"),
            Self::Other(raw) => write!(f, "0x{raw:x}"),
        }
    }
}

/// The one-line warning for a low class, naming the fix; `None` for any
/// other class.
pub fn low_priority_warning(class: PriorityClass) -> Option<String> {
    class.is_low().then(|| {
        format!(
            "hipfire: WARNING: this process runs at {class} priority, inherited from the process that started it (a Task Scheduler task at its default priority 7 is BelowNormal); Windows schedules it behind all Normal-priority work, which slows generation. Fix: start hipfire from a Normal-priority process: set the task's priority to 4-6 (export the task, change <Priority>7</Priority>, re-import it), or launch with `start \"\" /normal hipfire ...`."
        )
    })
}

/// The calling process's class; `None` when `GetPriorityClass` fails.
#[cfg(windows)]
pub fn current() -> Option<PriorityClass> {
    #[link(name = "kernel32")]
    extern "system" {
        /// `HANDLE GetCurrentProcess(void)`: a pseudo handle, never closed.
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        /// `DWORD GetPriorityClass(HANDLE)`; zero on failure.
        fn GetPriorityClass(process: *mut std::ffi::c_void) -> u32;
    }
    // SAFETY: GetCurrentProcess returns the constant pseudo handle of this
    // process, valid for its lifetime; GetPriorityClass only reads it.
    let raw = unsafe { GetPriorityClass(GetCurrentProcess()) };
    (raw != 0).then(|| PriorityClass::from_raw(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_values_map_to_documented_classes() {
        assert_eq!(PriorityClass::from_raw(0x40), PriorityClass::Idle);
        assert_eq!(PriorityClass::from_raw(0x4000), PriorityClass::BelowNormal);
        assert_eq!(PriorityClass::from_raw(0x20), PriorityClass::Normal);
        assert_eq!(PriorityClass::from_raw(0x8000), PriorityClass::AboveNormal);
        assert_eq!(PriorityClass::from_raw(0x80), PriorityClass::High);
        assert_eq!(PriorityClass::from_raw(0x100), PriorityClass::Realtime);
        assert_eq!(PriorityClass::from_raw(0x7), PriorityClass::Other(7));
        assert_eq!(PriorityClass::BelowNormal.to_string(), "BelowNormal");
        assert_eq!(PriorityClass::Other(0x7).to_string(), "0x7");
    }

    #[test]
    fn only_idle_and_below_normal_warn() {
        for class in [PriorityClass::Idle, PriorityClass::BelowNormal] {
            let warning = low_priority_warning(class).expect("low class warns");
            assert!(warning.contains(&class.to_string()));
            assert!(warning.contains("priority 7"));
            assert!(warning.contains("Fix:"));
            assert!(!warning.contains('\n'));
        }
        for class in [
            PriorityClass::Normal,
            PriorityClass::AboveNormal,
            PriorityClass::High,
            PriorityClass::Realtime,
            PriorityClass::Other(7),
        ] {
            assert_eq!(low_priority_warning(class), None);
        }
    }
}
