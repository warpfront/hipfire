// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Pinned Mesa checkout verification, patch application and build orchestration.
//!
//! The pinned original checkout (/home/kaden/qcal/peacemaker/mesa) is never touched. A separate
//! isolated worktree (detached at the pin) carries the explicit developer patch; both the patch
//! bytes and every patched file are bound by SHA-256 in `mesa/pin.json`.

use crate::util::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn pin_path() -> PathBuf {
    tool_dir().join("mesa/pin.json")
}

pub fn load_pin() -> Res<Value> {
    read_json(&pin_path())
}

pub fn patch_path(pin: &Value) -> Res<PathBuf> {
    Ok(tool_dir().join("mesa").join(jstr(pin, "patch.file")?))
}

fn git(src: &Path, args: &[&str]) -> Res<String> {
    let out = run(Command::new("git").arg("-C").arg(src).args(args))?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Verify that `src` is exactly the pinned commit plus exactly the pinned patch: HEAD equals the
/// pinned commit, the only dirty paths are the patched files, and each of those hashes to the
/// pinned value. Returns evidence for the manifest.
pub fn verify_tree(src: &Path, pin: &Value) -> Res<Value> {
    let commit = jstr(pin, "commit")?;
    let head = git(src, &["rev-parse", "HEAD"])?.trim().to_owned();
    if head != commit {
        return Err(format!("mesa HEAD {head} != pinned commit {commit}"));
    }
    let patch = patch_path(pin)?;
    let patch_sha = sha256_file(&patch)?;
    let want_patch = jstr(pin, "patch.sha256")?;
    if patch_sha != want_patch {
        return Err(format!(
            "patch {} sha256 {patch_sha} != pinned {want_patch}",
            patch.display()
        ));
    }
    let files = jget(pin, "patched_files_sha256")?
        .as_object()
        .ok_or("patched_files_sha256 must be an object")?;
    let status = git(src, &["status", "--porcelain", "--untracked-files=all"])?;
    let dirty: BTreeSet<String> = status
        .lines()
        .filter(|l| l.len() > 3)
        .map(|l| l[3..].to_owned())
        .collect();
    let expect: BTreeSet<String> = files.keys().cloned().collect();
    if dirty != expect {
        return Err(format!(
            "mesa tree dirty set {dirty:?} != pinned patched file set {expect:?} (run `aco-export build-mesa` on a clean isolated worktree)"
        ));
    }
    let mut evidence = serde_json::Map::new();
    for (rel, want) in files {
        let got = sha256_file(&src.join(rel))?;
        if Some(got.as_str()) != want.as_str() {
            return Err(format!("patched file {rel}: sha256 {got} != pinned {want}"));
        }
        evidence.insert(rel.clone(), json!(got));
    }
    Ok(json!({"commit": head, "patch_sha256": patch_sha, "patched_files_sha256": evidence}))
}

/// `aco-export build-mesa --mesa-src DIR --build-dir DIR [--jobs N] [--meson PATH] [--ninja PATH]`
pub fn cmd_build_mesa(a: &Args) -> Res<()> {
    let pin = load_pin()?;
    let src = a.path("mesa-src")?;
    let build = a.path("build-dir")?;
    let original = Path::new("/home/kaden/qcal/peacemaker/mesa");
    if std::fs::canonicalize(&src).map_err(err)? == std::fs::canonicalize(original).map_err(err)? {
        return Err(
            "refusing to patch the pinned original checkout; pass the isolated worktree".into(),
        );
    }
    let head = git(&src, &["rev-parse", "HEAD"])?.trim().to_owned();
    if head != jstr(&pin, "commit")? {
        return Err(format!(
            "mesa HEAD {head} != pinned commit {}",
            jstr(&pin, "commit")?
        ));
    }
    let status = git(&src, &["status", "--porcelain", "--untracked-files=all"])?;
    let patch = patch_path(&pin)?;
    if status.trim().is_empty() {
        let want = jstr(&pin, "patch.sha256")?;
        let got = sha256_file(&patch)?;
        if got != want {
            return Err(format!("patch sha256 {got} != pinned {want}"));
        }
        run(Command::new("git")
            .arg("-C")
            .arg(&src)
            .args(["apply", "--check"])
            .arg(&patch))?;
        run(Command::new("git")
            .arg("-C")
            .arg(&src)
            .arg("apply")
            .arg(&patch))?;
        eprintln!("applied {}", patch.display());
    }
    let evidence = verify_tree(&src, &pin)?;

    let meson = a.opt("meson").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        let p = PathBuf::from(home).join(".local/bin/meson");
        if p.is_file() {
            p
        } else {
            PathBuf::from("meson")
        }
    });
    let ninja = a
        .opt("ninja")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ninja"));
    if !build.join("build.ninja").is_file() {
        let mut c = Command::new(&meson);
        c.arg("setup").arg(&build).arg(&src);
        for o in jarr(&pin, "meson_options")? {
            c.arg(o.as_str().ok_or("meson option must be a string")?);
        }
        run(&mut c)?;
    }
    let jobs = a.opt("jobs").unwrap_or("8");
    let mut c = Command::new(&ninja);
    c.arg("-C").arg(&build).arg("-j").arg(jobs);
    for t in jarr(&pin, "build_targets")? {
        c.arg(t.as_str().ok_or("build target must be a string")?);
    }
    run(&mut c)?;

    let mut outs = serde_json::Map::new();
    for t in jarr(&pin, "build_targets")? {
        let t = t.as_str().unwrap();
        outs.insert(t.to_owned(), json!(sha256_file(&build.join(t))?));
    }
    let report = json!({
        "mesa_tree": evidence,
        "meson": tool_identity(&meson, "--version", 1)?,
        "ninja": tool_identity(&which(ninja.to_str().unwrap())?, "--version", 1)?,
        "build_dir": build.display().to_string(),
        "targets_sha256": outs,
    });
    println!("{}", String::from_utf8_lossy(&canon(&report)));
    Ok(())
}

/// `aco-export build-host --out PATH`: compile the plain-Vulkan host with the system C compiler.
pub fn cmd_build_host(a: &Args) -> Res<()> {
    let out = a.path("out")?;
    let src = tool_dir().join("host/vk_aco_host.c");
    let cc = a.opt("cc").unwrap_or("cc");
    let pk = run(Command::new("pkg-config").args(["--cflags", "--libs", "vulkan"]))?;
    let flags = String::from_utf8_lossy(&pk.stdout).into_owned();
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(err)?;
    }
    let mut c = Command::new(cc);
    c.args([
        "-O2",
        "-std=gnu11",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ])
    .arg(&src)
    .arg("-o")
    .arg(&out);
    for f in flags.split_whitespace() {
        c.arg(f);
    }
    run(&mut c)?;
    println!(
        "{}",
        String::from_utf8_lossy(&canon(&json!({
            "host": out.display().to_string(),
            "host_sha256": sha256_file(&out)?,
            "host_source_sha256": sha256_file(&src)?,
            "cc": tool_identity(&which(cc)?, "--version", 1)?,
        })))
    );
    Ok(())
}
