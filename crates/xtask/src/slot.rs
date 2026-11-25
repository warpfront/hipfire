// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The host build slot as a value.
//!
//! On hosts with `/home/kaden/qcal/locks/build-slot.sh`, every cargo build runs
//! inside one granted slot. The slot script marks a granted command with
//! `BUILD_SLOT_HELD`; children inherit it. A [`Slot`] is proof that this
//! process runs inside a slot (or that the host has no slot script). Only
//! [`Slot::take`] makes one, and only the top-level subcommand calls it, once:
//! when no slot is held it re-runs the whole xtask invocation under the slot
//! script and exits with its status, so everything below runs holding the one
//! grant. Code that runs cargo takes `&Slot`; there is no way to ask for a
//! second grant from inside the first.
//!
//! 2026-11-24: `regen-generated.sh --all` called the slot script per cargo step
//! from inside the land lane's slot and waited forever for a second grant.
//! [`decide`] never asks for a slot while the marker is set.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment marker the slot script sets on a granted command.
pub const MARKER: &str = "BUILD_SLOT_HELD";
/// The host slot script. Absent on CI and contributor machines.
pub const SCRIPT: &str = "/home/kaden/qcal/locks/build-slot.sh";
/// Set on the re-run child, so a slot script that fails to set [`MARKER`]
/// is an error instead of an endless chain of re-runs.
const REENTERED: &str = "XTASK_SLOT_REENTERED";

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// A slot is already held by this process tree: run directly.
    Held,
    /// No slot script on this host: run directly.
    Absent,
    /// Re-run this invocation under the slot script.
    Acquire(PathBuf),
}

/// What to do, given the inherited marker and the slot script path.
pub fn decide(marker: Option<&OsString>, script: &Path) -> Decision {
    if marker.is_some_and(|m| !m.is_empty()) {
        Decision::Held
    } else if script.is_file() {
        Decision::Acquire(script.to_path_buf())
    } else {
        Decision::Absent
    }
}

/// Proof that this process runs inside the build slot (or that there is none).
#[derive(Debug)]
pub struct Slot {
    _private: (),
}

impl Slot {
    /// Take the slot for the whole invocation. `args` are this process's
    /// arguments after the program name. Returns only when the slot is held or
    /// absent; otherwise re-runs this invocation under the slot script and
    /// exits with its status.
    pub fn take(args: &[OsString]) -> Slot {
        let marker = std::env::var_os(MARKER);
        match decide(marker.as_ref(), Path::new(SCRIPT)) {
            Decision::Held | Decision::Absent => Slot { _private: () },
            Decision::Acquire(script) => {
                if std::env::var_os(REENTERED).is_some() {
                    eprintln!(
                        "xtask: {} granted a slot without setting {MARKER}",
                        script.display()
                    );
                    std::process::exit(2);
                }
                let exe = match std::env::current_exe() {
                    Ok(exe) => exe,
                    Err(e) => {
                        eprintln!("xtask: cannot locate own executable: {e}");
                        std::process::exit(2);
                    }
                };
                let status = Command::new(&script)
                    .arg(exe)
                    .args(args)
                    .env(REENTERED, "1")
                    .status();
                match status {
                    Ok(s) => std::process::exit(s.code().unwrap_or(1)),
                    Err(e) => {
                        eprintln!("xtask: cannot run {}: {e}", script.display());
                        std::process::exit(2);
                    }
                }
            }
        }
    }

    /// A `cargo` command run inside this slot. It is never wrapped in the slot
    /// script again.
    pub fn cargo(&self) -> Command {
        Command::new("cargo")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn existing_script() -> PathBuf {
        // Any file that exists stands in for the slot script.
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
    }

    /// Regression for the 2026-11-24 build-slot re-entry deadlock: a call made
    /// inside a granted slot must not ask for a second one, even where the slot
    /// script exists.
    #[test]
    fn nested_call_inside_held_slot_takes_no_second_slot() {
        let held = OsString::from("3:4242");
        assert_eq!(decide(Some(&held), &existing_script()), Decision::Held);
    }

    #[test]
    fn unheld_with_script_acquires_once() {
        let script = existing_script();
        assert_eq!(decide(None, &script), Decision::Acquire(script.clone()));
        assert_eq!(
            decide(Some(&OsString::new()), &script),
            Decision::Acquire(script)
        );
    }

    #[test]
    fn no_script_runs_directly() {
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("no-such-build-slot.sh");
        assert_eq!(decide(None, &missing), Decision::Absent);
    }

    #[test]
    fn cargo_under_a_slot_is_plain_cargo() {
        let slot = Slot { _private: () };
        let cmd = slot.cargo();
        assert_eq!(cmd.get_program(), "cargo");
        assert_eq!(cmd.get_args().count(), 0);
    }
}
