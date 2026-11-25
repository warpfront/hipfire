// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cargo xtask <cmd>`: repository tooling. It sequences and locks; the
//! generators and checkers it calls stay where they are. It never links a
//! workspace crate: anything that needs product code runs as a subprocess.
//!
//! Locks are values: the top-level subcommand takes the build slot once
//! ([`slot::Slot::take`]) and passes it down.

mod regen;
mod slot;

use std::ffi::OsString;
use std::path::PathBuf;

const USAGE: &str = "usage: cargo xtask <command>\n\ncommands:\n  regen [--all|--staged]   regenerate generated files; prints the files it changed";

#[derive(Debug, PartialEq, Eq)]
enum Cmd {
    Regen(regen::Mode),
}

fn parse(args: &[OsString]) -> Result<Cmd, String> {
    match args.split_first() {
        Some((cmd, rest)) if cmd == "regen" => regen::parse_mode(rest).map(Cmd::Regen),
        _ => Err(USAGE.to_string()),
    }
}

/// The checkout this xtask was built from (`crates/xtask/../..`).
fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(manifest)
}

fn main() {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let cmd = match parse(&args) {
        Ok(cmd) => cmd,
        Err(usage) => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    let root = repo_root();
    let code = match cmd {
        Cmd::Regen(mode) => {
            let plan = regen::Plan::new(mode, || regen::staged_paths(&root));
            let slot = plan.needs_cargo().then(|| slot::Slot::take(&args));
            regen::run(&root, &plan, slot.as_ref())
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<OsString> {
        a.iter().map(OsString::from).collect()
    }

    #[test]
    fn regen_dispatches_with_its_mode() {
        assert_eq!(parse(&args(&["regen"])), Ok(Cmd::Regen(regen::Mode::Fast)));
        assert_eq!(
            parse(&args(&["regen", "--all"])),
            Ok(Cmd::Regen(regen::Mode::All))
        );
        assert_eq!(
            parse(&args(&["regen", "--staged"])),
            Ok(Cmd::Regen(regen::Mode::Staged))
        );
    }

    #[test]
    fn unknown_or_missing_command_is_usage() {
        assert_eq!(parse(&args(&[])), Err(USAGE.to_string()));
        assert_eq!(parse(&args(&["regenerate"])), Err(USAGE.to_string()));
        assert_eq!(parse(&args(&["--all"])), Err(USAGE.to_string()));
        assert_eq!(
            parse(&args(&["regen", "--bogus"])),
            Err(regen::USAGE.to_string())
        );
    }

    #[test]
    fn root_is_the_checkout() {
        assert!(repo_root().join("crates/xtask/Cargo.toml").is_file());
        assert!(repo_root().join("Cargo.toml").is_file());
    }
}
