// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! First-load receipt corpus for the Raon railgun owner.
//!
//! `auto` finds a receipt corpus beside the installed pack or in the per-user
//! corpus directory; a source build on a fresh home has neither. This module
//! builds the per-user one from the objects the process already loaded by
//! running the shipped `railgun-jit-corpus from-cache` once (the runtime never
//! links the analyzer):
//!
//! ```text
//! railgun-jit-corpus from-cache --arch A --program P --cache <hot dir>
//!     [--pack <installed pack arch dir>] --out <per-user corpus dir>
//! ```
//!
//! Contract (agreed with the subcommand): `--out` is the final corpus root
//! (`index.json` at its top), published atomically, nothing partial at `--out`
//! on any non-zero exit; exit 0 = covers the whole program, 2 = refused (a
//! deterministic reason, nothing published), anything else = error; the last
//! non-empty stdout line is one JSON object `{"status":"ok"|"refused"|"error",
//! "reason":..}`.
//!
//! The call is blocking, once per process, made by the owner right after the
//! decode graph's warm-up step (that step JIT-compiles every decode module into
//! the hot cache); it is never on a per-token path afterwards. A refusal (binary
//! absent, an unwritable output parent, a refused or failed run, a timeout) is
//! persisted in the hot cache directory keyed by a fingerprint of everything the
//! outcome depends on, so a fresh process with the same cache and binary reads
//! the recorded reason instead of re-running it.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, UNIX_EPOCH};

use railgun_corpus::sha256_hex;

/// The shipped subcommand binary.
pub const BINARY: &str = "railgun-jit-corpus";
/// `HIPFIRE_RAON_RAILGUN_LOCAL_CORPUS`: `0` forbids spawning the subcommand
/// (default `1`).
pub const ENV_LOCAL_CORPUS: &str = "HIPFIRE_RAON_RAILGUN_LOCAL_CORPUS";
/// Wall-clock cap of one run; the child is killed past it.
pub const TIMEOUT: Duration = Duration::from_secs(180);
/// Refusal record, in the hot cache directory.
pub const MARKER: &str = "railgun-corpus.refused.json";
/// Captured stdout + stderr of the last run, in the hot cache directory.
pub const LOG: &str = "railgun-corpus-from-cache.log";
/// Bytes of the log tail quoted when the run printed no JSON reason.
const TAIL_BYTES: usize = 400;

/// Everything one run depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvisionSpec {
    pub arch: String,
    /// `raon-ar-decode-<arch>`.
    pub program: String,
    /// The compiler's toolchain id (part of the fingerprint: the corpus pin is
    /// its SHA-256).
    pub toolchain_id: String,
    /// The arch-keyed hot cache directory; also holds the marker and log.
    pub hot_dir: PathBuf,
    pub pack_dir: Option<PathBuf>,
    /// The per-user corpus directory to publish.
    pub out_dir: PathBuf,
}

/// A successful run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provisioned {
    pub out_dir: PathBuf,
    pub millis: u128,
    /// The subcommand's one JSON line.
    pub report: String,
}

/// Where [`locate_binary`] looks, in order.
#[derive(Clone, Debug, Default)]
pub struct BinaryLookup<'a> {
    /// Directory of the running executable.
    pub exe_dir: Option<&'a Path>,
    /// `<hipfire home>/bin`.
    pub home_bin: Option<&'a Path>,
    /// The `PATH` value.
    pub path: Option<&'a OsStr>,
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// No execute bit off unix: any regular file counts.
#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    fs::metadata(p).map(|m| m.is_file()).unwrap_or(false)
}

/// First executable `name`: beside the running executable, in the hipfire
/// home's `bin`, then on `PATH`.
pub fn locate_binary(l: &BinaryLookup<'_>, name: &str) -> Option<PathBuf> {
    let from_path: Vec<PathBuf> = l
        .path
        .map(|p| std::env::split_paths(p).collect())
        .unwrap_or_default();
    l.exe_dir
        .into_iter()
        .chain(l.home_bin)
        .map(Path::to_path_buf)
        .chain(from_path)
        .map(|d| d.join(name))
        .find(|c| is_executable(c))
}

fn stamp(p: &Path) -> String {
    match fs::metadata(p) {
        Ok(m) => format!(
            "{}:{}",
            m.len(),
            m.modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos())
        ),
        Err(_) => "absent".to_owned(),
    }
}

/// Digest of everything a run's outcome depends on: arch, program, toolchain,
/// pack path, the subcommand binary (path, size, mtime) and every file of the
/// hot cache directory (name, size, mtime; the marker and log excluded).
pub fn fingerprint(spec: &ProvisionSpec, binary: Option<&Path>) -> String {
    let mut text = format!(
        "{}\n{}\n{}\n{:?}\n",
        spec.arch, spec.program, spec.toolchain_id, spec.pack_dir
    );
    match binary {
        Some(b) => text.push_str(&format!("{}:{}\n", b.display(), stamp(b))),
        None => text.push_str("binary:absent\n"),
    }
    let mut files: Vec<(String, String)> = fs::read_dir(&spec.hot_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                stamp(&e.path()),
            )
        })
        .filter(|(n, _)| n != MARKER && n != LOG)
        .collect();
    files.sort();
    for (n, s) in files {
        text.push_str(&format!("{n}:{s}\n"));
    }
    sha256_hex(text.as_bytes())
}

fn marker_path(spec: &ProvisionSpec) -> PathBuf {
    spec.hot_dir.join(MARKER)
}

/// The persisted refusal for `fingerprint`, if one was recorded for exactly it.
fn recorded_refusal(spec: &ProvisionSpec, fingerprint: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(&fs::read(marker_path(spec)).ok()?).ok()?;
    (v.get("fingerprint")?.as_str()? == fingerprint)
        .then(|| v.get("reason")?.as_str().map(str::to_owned))
        .flatten()
}

fn record_refusal(spec: &ProvisionSpec, fingerprint: &str, reason: &str) {
    let body = serde_json::json!({"fingerprint": fingerprint, "reason": reason});
    // Best effort: a missing marker only costs a re-run next process.
    let _ = fs::write(marker_path(spec), body.to_string());
}

/// The last non-empty line of `log` that is a JSON object with a `status`.
fn last_report(log: &str) -> Option<serde_json::Value> {
    log.lines()
        .rev()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .find_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            v.get("status")?;
            Some(v)
        })
}

fn tail(log: &str) -> String {
    let t = log.trim();
    let start = t.len().saturating_sub(TAIL_BYTES);
    let start = (start..=t.len())
        .find(|i| t.is_char_boundary(*i))
        .unwrap_or(t.len());
    t[start..].replace('\n', " | ")
}

/// Run `binary from-cache` for `spec`, blocking, at most `timeout`.
///
/// `Err` is the exact reason (also recorded for this fingerprint unless it was
/// already recorded). A recorded refusal for the same fingerprint is returned
/// without spawning. Nothing is spawned when `binary` is `None`.
pub fn provision(
    spec: &ProvisionSpec,
    binary: Option<&Path>,
    timeout: Duration,
) -> Result<Provisioned, String> {
    let fp = fingerprint(spec, binary);
    if let Some(reason) = recorded_refusal(spec, &fp) {
        return Err(format!(
            "{reason} (recorded for this cache; delete {} to retry)",
            marker_path(spec).display()
        ));
    }
    match run(spec, binary, timeout) {
        Ok(done) => {
            let _ = fs::remove_file(marker_path(spec));
            Ok(done)
        }
        Err(f) => {
            if f.persist {
                record_refusal(spec, &fp, &f.reason);
            }
            Err(f.reason)
        }
    }
}

/// A failed run. `persist` is `false` for a refusal a retry can fix: the
/// subcommand named modules with no hot cache entry (the process had not
/// compiled them yet), so a later process (whose cache then differs, or whose
/// run loaded more) must try again instead of reading a stale refusal.
struct Failure {
    reason: String,
    persist: bool,
}

impl From<String> for Failure {
    fn from(reason: String) -> Self {
        Self {
            reason,
            persist: true,
        }
    }
}

/// The report lists modules the program needs that have no hot cache entry.
fn names_missing_entries(report: &Option<serde_json::Value>) -> bool {
    report
        .as_ref()
        .and_then(|v| v.get("missing")?.as_array())
        .is_some_and(|m| !m.is_empty())
}

fn run(
    spec: &ProvisionSpec,
    binary: Option<&Path>,
    timeout: Duration,
) -> Result<Provisioned, Failure> {
    let binary = binary.ok_or_else(|| {
        format!("{BINARY} not found beside the executable, in $HIPFIRE_HOME/bin or on PATH")
    })?;
    let parent = spec
        .out_dir
        .parent()
        .ok_or_else(|| format!("corpus directory {} has no parent", spec.out_dir.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    let log_path = spec.hot_dir.join(LOG);
    let log = fs::File::create(&log_path)
        .map_err(|e| format!("cannot write {}: {e}", log_path.display()))?;
    let log_err = log
        .try_clone()
        .map_err(|e| format!("cannot write {}: {e}", log_path.display()))?;
    let mut cmd = Command::new(binary);
    cmd.arg("from-cache")
        .arg("--arch")
        .arg(&spec.arch)
        .arg("--program")
        .arg(&spec.program)
        .arg("--cache")
        .arg(&spec.hot_dir);
    if let Some(pack) = &spec.pack_dir {
        cmd.arg("--pack").arg(pack);
    }
    cmd.arg("--out")
        .arg(&spec.out_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot run {}: {e}", binary.display()))?;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "{BINARY} from-cache did not finish within {} s and was killed",
                    timeout.as_secs()
                )
                .into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(format!("waiting for {BINARY}: {e}").into()),
        }
    };
    let text = fs::read_to_string(&log_path).unwrap_or_default();
    let report = last_report(&text);
    let reason = |fallback: &str| {
        report
            .as_ref()
            .and_then(|v| v.get("reason")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| {
                let t = tail(&text);
                if t.is_empty() {
                    fallback.to_owned()
                } else {
                    format!("{fallback}: {t}")
                }
            })
    };
    if !status.success() {
        let code = status
            .code()
            .map_or("killed by a signal".to_owned(), |c| format!("exit {c}"));
        return Err(Failure {
            reason: format!("{BINARY} from-cache {code}: {}", reason("no report")),
            persist: !names_missing_entries(&report),
        });
    }
    let ok = report
        .as_ref()
        .and_then(|v| v.get("status")?.as_str())
        .is_some_and(|s| s == "ok");
    if !ok {
        return Err(format!(
            "{BINARY} from-cache exited 0 without an ok report: {}",
            reason("no report")
        )
        .into());
    }
    if !spec.out_dir.join(railgun_corpus::CORPUS_INDEX).is_file() {
        return Err(format!(
            "{BINARY} from-cache reported ok but {} holds no {}",
            spec.out_dir.display(),
            railgun_corpus::CORPUS_INDEX
        )
        .into());
    }
    Ok(Provisioned {
        out_dir: spec.out_dir.clone(),
        millis: started.elapsed().as_millis(),
        report: report.map(|v| v.to_string()).unwrap_or_default(),
    })
}

// The tests drive fake `from-cache` shell scripts, so they are unix-only.
// `pub(crate)`: `railgun_decode`'s tests spawn fake binaries too and must hold
// the same spawn lock (`tests::spawn_lock`).
#[cfg(all(test, unix))]
pub(crate) mod tests;
