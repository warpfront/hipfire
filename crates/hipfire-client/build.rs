// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stamps the workspace HEAD commit the same way `hipfire-cli/build.rs` does,
//! so `Engine` can compare its own build against the daemon's `--version`
//! answer. Same override env var, same `unknown` fallback outside a repo.

use std::{env, path::PathBuf, process::Command};

fn git_head(repo: &std::path::Path) -> Option<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.join("../..");
    let commit = env::var("HIPFIRE_BUILD_COMMIT_OVERRIDE")
        .ok()
        .or_else(|| git_head(&repo))
        .unwrap_or_else(|| "unknown".into());
    println!(
        "cargo:rustc-env=HIPFIRE_BUILD_COMMIT={}",
        commit.replace(['\r', '\n'], " ")
    );
    println!("cargo:rerun-if-env-changed=HIPFIRE_BUILD_COMMIT_OVERRIDE");
    if let Ok(head) = Command::new("git")
        .current_dir(&repo)
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
    {
        if head.status.success() {
            let path = String::from_utf8_lossy(&head.stdout).trim().to_owned();
            if !path.is_empty() {
                println!("cargo:rerun-if-changed={path}");
            }
        }
    }
}
