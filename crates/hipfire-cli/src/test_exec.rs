// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Test-only: install fake executables without ever holding a write descriptor.

use std::path::Path;
use std::process::Command;

/// Install `text` at `path` as an executable script, without this process ever
/// holding a write descriptor on it. A script written with `fs::write` and then
/// exec'd fails `ETXTBSY` whenever another test thread forks while the write
/// descriptor is open (the child keeps its copy until its own exec); this
/// crate's tests spawn fake daemons and CLIs from many threads. The write is
/// done by a short-lived `sh` that has exited when this returns, so no fork of
/// this process can ever inherit a writer of the script.
pub(crate) fn write_executable(path: &Path, text: &str) {
    let status = Command::new("sh")
        .args(["-c", "printf %s \"$1\" > \"$2\" && chmod 755 \"$2\"", "sh"])
        .arg(text)
        .arg(path)
        .status()
        .expect("spawn sh to install the script");
    assert!(status.success(), "installing {}: {status}", path.display());
}
