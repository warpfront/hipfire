// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cargo xtask regen [--all|--staged]`: regenerate every generated file that a
//! tool derives from the tree and a gate checks for drift. On a clean tree it
//! changes nothing.
//!
//! ```text
//! cargo xtask regen            fast set (Python, <30s)
//! cargo xtask regen --all      fast set + the cargo-built goldens
//! cargo xtask regen --staged   fast set, plus the cargo goldens whose inputs the
//!                              staged diff touches
//! ```
//! Every mode prints the files the run changed, one per line on stdout (the
//! pre-commit hook stages them). Progress goes to stderr.
//!
//! ```text
//! Fast set                                     tool                                      drift gate
//!   crates/*/map.md generated blocks           check-crate-maps.py <crate>...            check-crate-maps.py --check
//!   tests/golden/flags-defaults/
//!     developer-callsites.tsv                  flags-golden-extract.py --write           flags-golden-extract.py
//!   docs/CONFIG.md lifecycle table,
//!   docs/env-vars.md env inventory             check-lifecycle.py --write                check-lifecycle.py
//! Cargo set (needs a build; inside the one build slot, see slot.rs)
//!   tests/golden/flags-defaults/*.tsv (rest),
//!   tests/golden/flags-legacy-spelling/*.tsv   flags_defaults_golden / legacy_spelling   cargo test (ci.yml)
//!   crates/hipfire-tui/tests/snapshots/*.txt   UPDATE_SNAPSHOTS=1 cargo test golden_    cargo test (ci.yml)
//! ```
//!
//! NOT regenerated here, because each is a human decision a tool must not make:
//! ratchet ceilings (scripts/leanup-thresholds.txt), the selector-receipts
//! baseline (shrink-only), lifecycle labels (`// lifecycle:` marks and statuses),
//! fixture/quality pins (fixtures/pins.toml), the kernel-registry census
//! allowlist and flagship/route-plan source pins, the generated CDNA tables
//! (need AMD's XML; check-pm-generated-pins.py), the hand-written prose of
//! docs/env-vars.md (check-env-docs.py) and CHANGELOG entries.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::slot::Slot;

pub const USAGE: &str = "usage: cargo xtask regen [--all|--staged]";

const FAST_OUTPUTS: &[&str] = &[
    "crates/*/map.md",
    "tests/golden/flags-defaults/developer-callsites.tsv",
    "docs/CONFIG.md",
    "docs/env-vars.md",
];
const FLAGS_OUTPUTS: &[&str] = &[
    "tests/golden/flags-defaults",
    "tests/golden/flags-legacy-spelling",
];
const TUI_OUTPUTS: &[&str] = &["crates/hipfire-tui/tests/snapshots"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Fast,
    All,
    Staged,
}

/// Parse the arguments after `regen`.
pub fn parse_mode(args: &[OsString]) -> Result<Mode, String> {
    match args {
        [] => Ok(Mode::Fast),
        [a] if a == "--all" => Ok(Mode::All),
        [a] if a == "--staged" => Ok(Mode::Staged),
        _ => Err(USAGE.to_string()),
    }
}

/// Which cargo-built goldens a run rebuilds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub flags_cargo: bool,
    pub tui_cargo: bool,
}

impl Plan {
    /// `staged` is `git diff --cached --name-only --diff-filter=ACMR`; it is
    /// only read in `--staged` mode.
    pub fn new(mode: Mode, staged: impl FnOnce() -> String) -> Plan {
        match mode {
            Mode::Fast => Plan {
                flags_cargo: false,
                tui_cargo: false,
            },
            Mode::All => Plan {
                flags_cargo: true,
                tui_cargo: true,
            },
            Mode::Staged => {
                let staged = staged();
                Plan {
                    flags_cargo: staged.lines().any(touches_flags_inputs),
                    tui_cargo: staged.lines().any(touches_tui_inputs),
                }
            }
        }
    }

    pub fn needs_cargo(&self) -> bool {
        self.flags_cargo || self.tui_cargo
    }

    fn outputs(&self) -> Vec<&'static str> {
        let mut out = FAST_OUTPUTS.to_vec();
        if self.flags_cargo {
            out.extend_from_slice(FLAGS_OUTPUTS);
        }
        if self.tui_cargo {
            out.extend_from_slice(TUI_OUTPUTS);
        }
        out
    }
}

fn touches_flags_inputs(path: &str) -> bool {
    [
        "crates/hipfire-config/",
        "crates/hipfire-runtime/src/config",
        "crates/rdna-compute/src/feature_flags.rs",
        "crates/rdna-compute/src/arch_caps.rs",
    ]
    .iter()
    .any(|p| path.starts_with(p))
}

fn touches_tui_inputs(path: &str) -> bool {
    ["crates/hipfire-tui/", "crates/hipfire-config/"]
        .iter()
        .any(|p| path.starts_with(p))
}

/// The staged paths, as `--staged` reads them.
pub fn staged_paths(root: &Path) -> String {
    Command::new("git")
        .args(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
        .current_dir(root)
        .stderr(Stdio::inherit())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

struct Step {
    /// The command line as reported in the header and the failure list.
    shown: Vec<String>,
    cmd: Command,
}

fn python(root: &Path, args: &[String]) -> Step {
    let mut cmd = Command::new("python3");
    cmd.args(args).current_dir(root);
    let mut shown = vec!["python3".to_string()];
    shown.extend(args.iter().cloned());
    Step { shown, cmd }
}

fn cargo(slot: &Slot, root: &Path, env: Option<(&str, &str)>, args: &[&str]) -> Step {
    let mut cmd = slot.cargo();
    cmd.args(args).current_dir(root);
    let mut shown = Vec::new();
    if let Some((k, v)) = env {
        cmd.env(k, v);
        shown.push("env".to_string());
        shown.push(format!("{k}={v}"));
    }
    shown.push("cargo".to_string());
    shown.extend(args.iter().map(|a| a.to_string()));
    Step { shown, cmd }
}

/// Crates with a `map.md`, by name, in byte order.
fn mapped_crates(root: &Path) -> Vec<String> {
    let mut crates: Vec<String> = std::fs::read_dir(root.join("crates"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().join("map.md").is_file())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    crates.sort();
    crates
}

fn steps(root: &Path, plan: &Plan, slot: Option<&Slot>) -> Vec<Step> {
    let mut maps = vec!["scripts/check-crate-maps.py".to_string()];
    maps.extend(mapped_crates(root));
    let mut steps = vec![
        python(root, &maps),
        python(
            root,
            &["scripts/flags-golden-extract.py".into(), "--write".into()],
        ),
        python(
            root,
            &["scripts/check-lifecycle.py".into(), "--write".into()],
        ),
    ];
    if plan.needs_cargo() {
        let slot = slot.expect("a plan with cargo steps runs inside a build slot");
        if plan.flags_cargo {
            steps.push(cargo(
                slot,
                root,
                None,
                &[
                    "test",
                    "-p",
                    "hipfire-runtime",
                    "--test",
                    "flags_defaults_golden",
                    "--",
                    "--ignored",
                    "regenerate_flags_defaults_golden",
                ],
            ));
            steps.push(cargo(
                slot,
                root,
                None,
                &[
                    "test",
                    "-p",
                    "hipfire-runtime",
                    "--lib",
                    "--",
                    "--ignored",
                    "regenerate_legacy_spelling_golden",
                ],
            ));
        }
        if plan.tui_cargo {
            steps.push(cargo(
                slot,
                root,
                Some(("UPDATE_SNAPSHOTS", "1")),
                &["test", "-p", "hipfire-tui", "golden_"],
            ));
        }
    }
    steps
}

/// Run one step with stdout and stderr merged onto our stderr, dropping the
/// per-crate "already up to date" lines that are noise on every commit.
fn run_step(step: &mut Step) -> bool {
    eprintln!(
        "== {} ==",
        step.shown
            .iter()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    );
    let (reader, writer) = match std::io::pipe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("xtask regen: pipe: {e}");
            return false;
        }
    };
    let writer2 = match writer.try_clone() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("xtask regen: pipe: {e}");
            return false;
        }
    };
    step.cmd.stdin(Stdio::null()).stdout(writer).stderr(writer2);
    let mut child = match step.cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xtask regen: cannot run {}: {e}", step.shown[0]);
            return false;
        }
    };
    // The Command still owns the parent's copies of the write end; drop them so
    // the reader sees EOF when the child exits.
    step.cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let stderr = std::io::stderr();
    let mut err = stderr.lock();
    for line in BufReader::new(reader).split(b'\n').map_while(Result::ok) {
        if !line.ends_with(b": already up to date") {
            let _ = err.write_all(&line);
            let _ = err.write_all(b"\n");
        }
    }
    child.wait().map(|s| s.success()).unwrap_or(false)
}

/// Modified and untracked files under `outputs`, with their contents.
fn snapshot(root: &Path, outputs: &[&str]) -> HashMap<String, Vec<u8>> {
    let listed = Command::new("git")
        .args(["ls-files", "-mo", "--exclude-standard", "-z", "--"])
        .args(outputs)
        .current_dir(root)
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    listed
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .filter_map(|p| {
            let path = String::from_utf8_lossy(p).into_owned();
            // A deleted file is listed by -m but has no content: skip it, as
            // the shell script's sha1sum did.
            std::fs::read(root.join(&path)).ok().map(|c| (path, c))
        })
        .collect()
}

/// Files this run changed: modified/untracked now, and not with the same
/// content before. Sorted by path.
fn changed(before: &HashMap<String, Vec<u8>>, after: &HashMap<String, Vec<u8>>) -> Vec<String> {
    let mut out: Vec<String> = after
        .iter()
        .filter(|(path, content)| before.get(*path) != Some(*content))
        .map(|(path, _)| path.clone())
        .collect();
    out.sort();
    out
}

/// Run the plan; returns the process exit code.
pub fn run(root: &Path, plan: &Plan, slot: Option<&Slot>) -> i32 {
    let outputs = plan.outputs();
    let before = snapshot(root, &outputs);
    let mut failed = Vec::new();
    for mut step in steps(root, plan, slot) {
        if !run_step(&mut step) {
            failed.push(step.shown.join(" "));
        }
    }
    if !failed.is_empty() {
        eprintln!("xtask regen: {} generator(s) failed:", failed.len());
        for f in &failed {
            eprintln!("  {f}");
        }
        return 1;
    }
    let after = snapshot(root, &outputs);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for path in changed(&before, &after) {
        let _ = writeln!(out, "{path}");
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<OsString> {
        a.iter().map(OsString::from).collect()
    }

    #[test]
    fn modes_parse() {
        assert_eq!(parse_mode(&args(&[])), Ok(Mode::Fast));
        assert_eq!(parse_mode(&args(&["--all"])), Ok(Mode::All));
        assert_eq!(parse_mode(&args(&["--staged"])), Ok(Mode::Staged));
    }

    #[test]
    fn bad_arguments_are_refused() {
        for bad in [
            &["--al"][..],
            &["all"],
            &["--all", "--staged"],
            &["--staged", "x"],
            &[""],
        ] {
            assert_eq!(parse_mode(&args(bad)), Err(USAGE.to_string()), "{bad:?}");
        }
    }

    #[test]
    fn fast_and_all_never_read_the_index() {
        let p = Plan::new(Mode::Fast, || panic!("fast mode read the index"));
        assert!(!p.needs_cargo());
        let p = Plan::new(Mode::All, || panic!("--all read the index"));
        assert_eq!(
            p,
            Plan {
                flags_cargo: true,
                tui_cargo: true
            }
        );
    }

    #[test]
    fn staged_selects_the_cargo_goldens_its_inputs_touch() {
        let plan = |s: &str| Plan::new(Mode::Staged, || s.to_string());
        assert_eq!(
            plan("docs/README.md\ncrates/hipfire-cli/src/main.rs\n"),
            Plan {
                flags_cargo: false,
                tui_cargo: false
            }
        );
        assert_eq!(
            plan("crates/hipfire-config/src/lib.rs\n"),
            Plan {
                flags_cargo: true,
                tui_cargo: true
            }
        );
        assert_eq!(
            plan("crates/hipfire-runtime/src/config_flags.rs\n"),
            Plan {
                flags_cargo: true,
                tui_cargo: false
            }
        );
        assert_eq!(
            plan("crates/rdna-compute/src/feature_flags.rs\n"),
            Plan {
                flags_cargo: true,
                tui_cargo: false
            }
        );
        assert_eq!(
            plan("crates/rdna-compute/src/arch_caps.rs\n"),
            Plan {
                flags_cargo: true,
                tui_cargo: false
            }
        );
        assert_eq!(
            plan("crates/rdna-compute/src/gemm.rs\n"),
            Plan {
                flags_cargo: false,
                tui_cargo: false
            }
        );
        assert_eq!(
            plan("crates/hipfire-tui/src/app.rs\n"),
            Plan {
                flags_cargo: false,
                tui_cargo: true
            }
        );
        // Anchored at the start of the path, as the shell grep was.
        assert_eq!(
            plan("x/crates/hipfire-tui/src/app.rs\n"),
            Plan {
                flags_cargo: false,
                tui_cargo: false
            }
        );
    }

    #[test]
    fn outputs_follow_the_plan() {
        let fast = Plan {
            flags_cargo: false,
            tui_cargo: false,
        };
        assert_eq!(fast.outputs(), FAST_OUTPUTS);
        let all = Plan {
            flags_cargo: true,
            tui_cargo: true,
        };
        assert_eq!(
            all.outputs().len(),
            FAST_OUTPUTS.len() + FLAGS_OUTPUTS.len() + TUI_OUTPUTS.len()
        );
    }

    #[test]
    fn changed_lists_new_and_rewritten_files_only() {
        let m = |e: &[(&str, &str)]| {
            e.iter()
                .map(|(p, c)| (p.to_string(), c.as_bytes().to_vec()))
                .collect::<HashMap<_, _>>()
        };
        let before = m(&[("a", "1"), ("b", "2")]);
        let after = m(&[("a", "1"), ("b", "3"), ("c", "4")]);
        assert_eq!(
            changed(&before, &after),
            vec!["b".to_string(), "c".to_string()]
        );
        assert!(changed(&after, &after).is_empty());
    }
}
