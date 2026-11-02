// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Shared helpers: hashing, canonical JSON, argument parsing, process execution.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub type Res<T> = Result<T, String>;

pub fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn sha256_file(path: &Path) -> Res<String> {
    let b = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(sha256_hex(&b))
}

/// Canonical JSON bytes: compact, object keys sorted (serde_json's default BTreeMap order), one
/// trailing newline. Every hash in the manifest is over bytes produced by this function or over
/// raw file bytes, never over an in-memory C struct.
pub fn canon(v: &Value) -> Vec<u8> {
    let mut b = serde_json::to_vec(v).expect("serialize");
    b.push(b'\n');
    b
}

pub fn canon_sha256(v: &Value) -> String {
    sha256_hex(&canon(v))
}

pub fn read_json(path: &Path) -> Res<Value> {
    let b = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&b).map_err(|e| format!("parse {}: {e}", path.display()))
}

pub fn write_file(path: &Path, bytes: &[u8]) -> Res<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn jget<'a>(v: &'a Value, path: &str) -> Res<&'a Value> {
    let mut cur = v;
    for part in path.split('.') {
        cur = cur
            .get(part)
            .ok_or_else(|| format!("required field `{path}` is absent"))?;
    }
    Ok(cur)
}

pub fn jstr<'a>(v: &'a Value, path: &str) -> Res<&'a str> {
    jget(v, path)?
        .as_str()
        .ok_or_else(|| format!("field `{path}` must be a string"))
}

pub fn ju64(v: &Value, path: &str) -> Res<u64> {
    jget(v, path)?
        .as_u64()
        .ok_or_else(|| format!("field `{path}` must be an unsigned integer"))
}

pub fn jbool(v: &Value, path: &str) -> Res<bool> {
    jget(v, path)?
        .as_bool()
        .ok_or_else(|| format!("field `{path}` must be a boolean"))
}

pub fn jarr<'a>(v: &'a Value, path: &str) -> Res<&'a Vec<Value>> {
    jget(v, path)?
        .as_array()
        .ok_or_else(|| format!("field `{path}` must be an array"))
}

/// Repository root: tools/aco-export/../..
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

pub fn tool_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

pub fn default_spec() -> PathBuf {
    repo_root().join("crates/peacemaker-lift/tests/fixtures/aco/producer.json")
}

/// `--key value` / `--flag` parser. Repeated keys are kept in order.
pub struct Args {
    pub map: BTreeMap<String, Vec<String>>,
}

impl Args {
    pub fn parse(argv: &[String], flags: &[&str]) -> Res<Self> {
        let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument `{a}`"))?;
            if flags.contains(&key) {
                map.entry(key.to_owned()).or_default().push(String::new());
                i += 1;
            } else {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for --{key}"))?;
                map.entry(key.to_owned()).or_default().push(v.clone());
                i += 2;
            }
        }
        Ok(Self { map })
    }

    pub fn opt(&self, k: &str) -> Option<&str> {
        self.map.get(k).and_then(|v| v.last()).map(String::as_str)
    }

    pub fn req(&self, k: &str) -> Res<&str> {
        self.opt(k).ok_or_else(|| format!("--{k} is required"))
    }

    pub fn flag(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }

    pub fn path(&self, k: &str) -> Res<PathBuf> {
        let p = PathBuf::from(self.req(k)?);
        Ok(if p.is_absolute() {
            p
        } else {
            std::env::current_dir().map_err(err)?.join(p)
        })
    }
}

pub fn run(cmd: &mut Command) -> Res<Output> {
    let out = cmd
        .output()
        .map_err(|e| format!("spawn {:?}: {e}", cmd.get_program()))?;
    if !out.status.success() {
        return Err(format!(
            "{:?} failed ({}):\n{}{}",
            cmd.get_program(),
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out)
}

pub fn which(name: &str) -> Res<PathBuf> {
    if name.contains('/') {
        return Ok(PathBuf::from(name));
    }
    let path = std::env::var_os("PATH").ok_or("PATH is unset")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(format!("`{name}` not found in PATH"))
}

/// Identity of an external tool: resolved path, SHA-256 of the executable, and the first
/// `max_lines` lines of `<tool> <version_arg>`.
pub fn tool_identity(path: &Path, version_arg: &str, max_lines: usize) -> Res<Value> {
    let real =
        fs::canonicalize(path).map_err(|e| format!("canonicalize {}: {e}", path.display()))?;
    let out = Command::new(&real)
        .arg(version_arg)
        .output()
        .map_err(|e| format!("{}: {e}", real.display()))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let version: Vec<&str> = text.lines().take(max_lines).collect();
    Ok(serde_json::json!({
        "path": real.display().to_string(),
        "sha256": sha256_file(&real)?,
        "version_arg": version_arg,
        "version": version.join("\n"),
    }))
}
