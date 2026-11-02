// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next GDN `q8_ef` (live f16 error-feedback residual, plain-Q8 ring and
//! snapshots, residual zeroed on rollback/restore) product-route and restore
//! tests.
//!
//! Policy (PLAN.md Track B(a), POLICY-DIFF.md): for `q8_ef` the cross-route
//! comparisons (MTP vs AR IDs, MTP vs AR live state, restored vs cold
//! continuation) are **diagnostics**, reported and never asserted. Same-history
//! determinism, reset semantics, route counters, finiteness and ownership are
//! **hard**. `greedy_mtp_identity_hw` / `session_cache_hw` keep their hard
//! equality for Q8 and F32 and refuse `q8_ef`; this file is where `q8_ef` runs.
//!
//! Every test is `#[ignore]`d (real HIP GPU, the canonical
//! `qwen3.8-flash-next-gptq3.mq4`) and refuses to run unless
//! `HIPFIRE_GDN_TEST_STATE_QUANT=q8_ef` is set explicitly (an unset variable
//! means `q8` for the other harnesses, so it never selects `q8_ef` here), and
//! unless it runs inside a fabric hold that holds the Flash-Next host lock with
//! `--test-threads=1` (the `session_cache_hw` guard, copied).
//!
//! ```text
//! HIPFIRE_GDN_TEST_STATE_QUANT=q8_ef HIPFIRE_MTP_IDENTITY_MODEL=$M \
//!   HIPFIRE_SESSION_CACHE_MODEL=$M fabric-run.sh ... --hostmem-lock -- \
//!   q8_ef_product_routes_hw --ignored --nocapture --test-threads=1
//! ```
//!
//! # `q8_ef_product_routes_on_flash_next`
//!
//! The greedy identity harness' arms (`ar`, `batched`, `incremental`,
//! `adaptive`, `full-head`) on its `p1` (1200 tokens) and `seasons` (256 tokens)
//! cases, each arm in **two fresh child processes** of this test binary
//! (`q8_ef_product_routes_arm`, same child pattern, env scrubbing, forwarded
//! `HIPFIRE_GDN_TEST_STATE_QUANT` and `HIPFIRE_REPLAY_BACKEND`, per-arm
//! `<arm>.rN.identity.json` compared with the identity the parent expects).
//!
//! Hard:
//! - both runs of an arm produce identical IDs and an identical final live
//!   state digest (`Qwen4State::debug_digest`, which includes every layer's
//!   `DigestRow::GdnResidual`) (same route, same inputs, same binary =>
//!   same result). This is hard for every arm except `adaptive` when its two
//!   runs took different route decisions. A *decision* is one window's
//!   (position before, drafts, verify route). The `adaptive` chooser is
//!   wall-clock driven, and the q8_ef residual reset history follows the route
//!   decisions, so two fresh adaptive runs that decided differently legitimately
//!   diverge; PLAN requires same-route (same decisions) determinism and says
//!   adaptive wall-clock decisions are recorded and distribution-tested, not
//!   assumed replay-identical. So for `adaptive`: if the two runs' decisions
//!   differ anywhere, the IDs/digest comparison is diagnostic only (the first
//!   differing window is recorded in the summary); if the decisions are
//!   identical, IDs and digest must be identical (hard). Every other arm stays
//!   hard regardless of history. The two runs' window histories (positions,
//!   drafts, accepted, committed, residual reset counters, verify route) are
//!   recorded and quoted in a failure message as diagnostic context only;
//!   they are never an assertion of their own;
//! - route counters from the drafter's own `QWEN4_MTP_TRACE` window events
//!   (`batched` / `full-head`: batched windows only, and at least one rejected
//!   window; `incremental`: interleaved windows only; `adaptive`: at least one
//!   batched window; every window with drafts has a trace event);
//! - residual reset counter (`Qwen4State::gdn_residual_resets`, the same
//!   number `gdn_residual_resets` of the trace lines): a *rolled-back* batched
//!   window (committed <= drafts: the target verified drafts + 1 rows and
//!   committing fewer rolls rows back from the ring, either a rejection
//!   (accepted < drafts) or a full accept whose commit stopped early, e.g. at a
//!   turn boundary) increases it (ring rollback or restore); a *full-commit*
//!   batched window (committed = drafts + 1) or a target-only window leaves it
//!   unchanged. Batched coverage counts real rejections (accepted < drafts). An
//!   *interleaved* window is not assumed to reset or not to reset: its counter
//!   rise is recorded, and with the lab state tap every window's reset-counter
//!   rise must equal the number of residual events the tap saw (kind
//!   `Reset` / `Restore` / `PartialRollback`);
//! - zero check at the reset: after every rolled-back window the child
//!   downloads every layer's residual; on a batched window (ring rollback, no
//!   target forward after it) all of it must be exactly zero. The check runs
//!   only where the GPU captures verify rows (`native_mtp_row_capture`);
//!   elsewhere the rollback is a restore plus replay forward and the check is
//!   reported `not applicable`;
//! - the residual is zero and the reset counter is 1 right after load, the
//!   residual is finite (no f16 NaN/Inf) in every scan, and the `ar` arm ends
//!   with a non-zero residual (the EF plane is live);
//! - non-finite logits: the `ar` arm downloads its logits after the prefill
//!   and after every decode step and requires them finite. The MTP arms'
//!   verify-row logits are consumed on the device by acceptance; they are
//!   observed only through the read-only lab window tap
//!   (`Qwen4MtpDrafter::lab_observe_windows` / `lab_take_window_observation`,
//!   `lab_window::LabForwardLogits::nonfinite`, per window every speculative
//!   forward's output rows). With it (`--features lab`) every window's
//!   non-finite count must be zero, the tap's residual-event list must not be
//!   empty on a window whose reset counter rose, and the tap's reset delta must
//!   equal the state counter's; the capture-ring half is recorded.
//!
//! **INCOMPLETE (fail closed):** without the lab tap (a non-lab build) or when
//! the tap returned nothing for a window, the MTP verify-row logits are
//! unobservable: the test writes its summary and then fails with the list of
//! unobservable mandatory checks. It never claims finiteness of those logits
//! from the IDs.
//!
//! Diagnostics in `summary.json` / `summary.txt` per case: AR-vs-arm first
//! differing ID and differing digest categories, per-arm tau (committed tokens
//! per window), accepted/drafts, rejected windows, reset counts, residual scans
//! and the decoded text of every arm.
//!
//! # `q8_ef_restore_routes_on_flash_next`
//!
//! Public session API (`session_plan`, `prefill_final(reused)`,
//! `session_commit`, the `SessionState` trait) and the bundle's transaction
//! snapshot/restore, in one process (`HIPFIRE_SESSION_CACHE_MODEL`):
//!
//! - cold A / cold A again / cold B, then restore A and B from the pool (each
//!   twice) and continue decoding 16 greedy tokens. Hard: every leg bumps the
//!   reset counter; two cold histories and two restored histories of the same
//!   prompt give identical IDs and identical live digests (prefill and final);
//!   logits finite after every step. Diagnostic: restored vs cold continuation
//!   first differing ID, differing digest categories, decoded text;
//! - `SessionState::restore_parts` of the live state (the production
//!   session/prefix/disk restore entry that zeroes the residual): hard zero
//!   immediately after it, reset counter +1, Q8 recurrent/conv/position rows
//!   unchanged, and two continuations from the restored state identical;
//! - `Qwen4Bundle::snapshot`/`restore` (the transaction restore): hard zero
//!   immediately after, counter +1, recurrent/conv/position rows equal to the
//!   snapshot-time digest, two restored continuations identical; diagnostic
//!   uninterrupted-vs-restored continuation;
//! - `Qwen4Bundle::reset`: hard zero immediately after, counter +1;
//! - the native-MTP route: cold/restored prefill seeds and digests, twice each
//!   (decode windows are not compared here: they need the wall-clock chooser).
//!
//! Disk-tier restore is not exercised by this test (the disk envelope carries
//! the same `q8_ef-reset` scope); the session scope string must contain
//! `q8_ef-reset`.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::mtp_spec::{native_mtp_row_capture, Qwen4MtpDrafter};
use hipfire_arch_qwen4::{GdnStateFormat, Qwen4KvBackend, Qwen4StateFormat};
use hipfire_dispatch::pipeline::DraftHeadPolicy;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[path = "common/hw_identity.rs"]
mod hw_identity;
use hw_identity::Counters;

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const SESSION_MODEL_ENV: &str = "HIPFIRE_SESSION_CACHE_MODEL";
const OUT_ENV: &str = "HIPFIRE_MTP_IDENTITY_OUT";
const ARM_ENV: &str = "HIPFIRE_MTP_IDENTITY_ARM";
/// `<case>:<run>` for a child, e.g. `p1:2`.
const CASE_ENV: &str = "HIPFIRE_MTP_IDENTITY_CASE";
const ARM_TEST: &str = "q8_ef_product_routes_arm";
/// Harness-control variables: they differ between a parent and each child
/// without changing what the arm computes, so identities exclude them.
const CONTROL_ENV: [&str; 4] = [MODEL_ENV, OUT_ENV, ARM_ENV, CASE_ENV];
const RESTORE_CONTROL_ENV: [&str; 3] = [MODEL_ENV, SESSION_MODEL_ENV, OUT_ENV];
/// The launch-backend variable, the only graph-like switch Qwen4 honors in
/// process (`Gpu::init` -> `ReplayController::from_config`).
const REPLAY_BACKEND_ENV: &str = "HIPFIRE_REPLAY_BACKEND";
const GRAPH_POLICY_NOTE: &str = "graph policy is the retained-replay backend: hip = eager launches, auto = AR decode records then replays a retained body; verify forwards (2+ rows) always launch eager; HIPFIRE_GRAPH is not read by Qwen4";
const GDN_MODE: &str = "q8_ef";

/// Why the MTP arms' verify-row logits cannot be checked for non-finite values.
const MTP_LOGITS_UNOBSERVABLE: &str = "the MTP verify-row logits are consumed on the device by acceptance and are exposed only by the lab window tap (hipfire_arch_qwen4::lab_window::LabForwardLogits, cfg(feature = \"lab\")): build this test with --features lab and a window tap that observed every window's forwards";

const PROMPT: &[u8] = include_bytes!("fixtures/qwen4_mtp_identity_p1.txt");
/// `hipx:~/pm-wave/fns3/in/p1.txt` (md5 `c91f8380a035dc842416b1be5efa8197`).
const PROMPT_SHA256: &str = "4ba1d4d971b99ace1fd294e1cdb087c432d87a155c203bddef87552ce02353a7";
const PROMPT_TOKENS: usize = 1047;
const MAX_SEQ: usize = 16384;
const RESTORE_MAX_SEQ: usize = 32768;
const MTP_K: usize = 3;
/// Greedy tokens decoded after a restore leg / a transaction continuation.
const DECODE: usize = 16;
const TX_DECODE: usize = 8;

/// The serve battery's factual prompt (`scripts/serve_harness.py`).
const SEASONS: &str = "What causes the seasons on Earth? Answer in exactly three sentences.";
/// Prompt IDs `hipfire serve` prefilled for SEASONS (Gate041z, daemon diag dump).
const SEASONS_IDS: [u32; 25] = [
    248045, 846, 198, 3710, 10814, 279, 15127, 383, 8964, 30, 21134, 303, 6681, 2250, 22157, 13,
    248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

// ------------------------------------------------------------------ mode

/// `HIPFIRE_GDN_TEST_STATE_QUANT` must be set explicitly to `q8_ef`: the other
/// harnesses read an unset variable as `q8`, and an implicit `q8_ef` must never
/// be possible here.
fn explicit_q8ef(value: Option<&str>) -> Result<(), String> {
    let env = hw_identity::GDN_TEST_STATE_QUANT_ENV;
    match value {
        None => Err(format!(
            "{env} is unset: this harness requires {env}={GDN_MODE} to be set explicitly"
        )),
        Some(value) => {
            let (_, format) = hw_identity::parse_gdn_test_state_quant(Some(value))?;
            if format == GdnStateFormat::Q8Ef {
                Ok(())
            } else {
                Err(format!(
                    "{env}={value} is not {GDN_MODE}: use greedy_mtp_identity_hw / session_cache_hw for q8 and fp32"
                ))
            }
        }
    }
}

fn require_explicit_q8ef() -> Result<(), String> {
    let env = hw_identity::GDN_TEST_STATE_QUANT_ENV;
    match std::env::var_os(env) {
        None => explicit_q8ef(None),
        Some(raw) => match raw.to_str() {
            Some(value) => explicit_q8ef(Some(value)),
            None => Err(format!("{env}={raw:?} is not UTF-8")),
        },
    }
}

fn model_from(env: &str) -> PathBuf {
    require_explicit_q8ef().unwrap_or_else(|e| panic!("{e}"));
    let path = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{env} must name qwen3.8-flash-next-gptq3.mq4"));
    assert!(path.is_file(), "{env}={} is not a file", path.display());
    path
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen4-q8ef-routes-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// The launch backend this process requests, canonical (`hip`, `auto`,
/// `shadow`); unset or empty is `hip`. An unknown value is refused.
fn replay_backend_request() -> Result<&'static str, String> {
    match std::env::var(REPLAY_BACKEND_ENV) {
        Err(std::env::VarError::NotPresent) => Ok("hip"),
        Err(std::env::VarError::NotUnicode(value)) => {
            Err(format!("{REPLAY_BACKEND_ENV}={value:?} is not UTF-8"))
        }
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "" | "hip" | "off" => Ok("hip"),
            "auto" | "redline" => Ok("auto"),
            "shadow" => Ok("shadow"),
            other => Err(format!(
                "{REPLAY_BACKEND_ENV}={other:?} is not a known backend (accepted: hip, off, auto, redline, shadow)"
            )),
        },
    }
}

/// The canonical name of the backend the runtime resolved.
fn replay_backend_name(request: rdna_compute::replay::ReplayBackendRequest) -> &'static str {
    use rdna_compute::replay::ReplayBackendRequest;
    match request {
        ReplayBackendRequest::Hip => "hip",
        ReplayBackendRequest::Shadow => "shadow",
        ReplayBackendRequest::Auto => "auto",
    }
}

// ------------------------------------------------------------ fabric hold

/// The Flash-Next host lock fabric takes with `--hostmem-lock` (shared for
/// correct, exclusive for timing holds).
const HOSTMEM_LOCK: &str = "hostmem-flashnext.lock";

/// `/proc/<pid>/stat` fields after the command: 1 = ppid, 19 = starttime.
fn proc_stat(pid: u32, index: usize) -> Option<u64> {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    raw.get(raw.rfind(')')? + 2..)?
        .split_whitespace()
        .nth(index)?
        .parse()
        .ok()
}

/// Granted flocks on files named [`HOSTMEM_LOCK`] that `pids` hold, read from
/// `/proc/locks` through each process's open files: `(pid, mode)`.
fn hostmem_lock_holders(pids: impl Iterator<Item = u32>) -> Vec<(u32, String)> {
    use std::os::unix::fs::MetadataExt;
    let locks = std::fs::read_to_string("/proc/locks").unwrap_or_default();
    let mut held = Vec::new();
    for pid in pids {
        let Ok(fds) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
            continue;
        };
        for fd in fds.flatten() {
            let Ok(target) = std::fs::read_link(fd.path()) else {
                continue;
            };
            if target.file_name().and_then(|n| n.to_str()) != Some(HOSTMEM_LOCK) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&target) else {
                continue;
            };
            let inode = format!(":{}", meta.ino());
            for line in locks.lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() >= 6
                    && !f.contains(&"->")
                    && f[1] == "FLOCK"
                    && f[4] == pid.to_string()
                    && f[5].ends_with(&inode)
                    && !held.iter().any(|(p, _): &(u32, String)| *p == pid)
                {
                    held.push((pid, f[3].to_string()));
                }
            }
        }
    }
    held
}

/// Why a Flash-Next load must not run here, or `None` inside its own hold:
/// the harness runs one test at a time (two loads in one process exhaust host
/// memory), `chain` (this process, then its ancestors) contains the live fabric
/// wrapper `FABRIC_JOB` names (`pid:starttime:cards`), and a process on the
/// chain up to that wrapper holds [`HOSTMEM_LOCK`].
fn hold_refusal(
    threads_one: bool,
    marker: &str,
    chain: &[u32],
    start_of: impl Fn(u32) -> Option<u64>,
    holds_lock: impl Fn(&[u32]) -> bool,
) -> Option<String> {
    if !threads_one {
        return Some("each test loads Flash-Next: run with --test-threads=1".into());
    }
    let mut parts = marker.splitn(3, ':');
    let wrapper = parts.next().and_then(|p| p.parse::<u32>().ok());
    let start = parts.next().and_then(|s| s.parse::<u64>().ok());
    let Some((wrapper, start)) = wrapper.zip(start) else {
        return Some("no fabric hold (FABRIC_JOB unset)".into());
    };
    let Some(depth) = chain.iter().position(|&pid| pid == wrapper) else {
        return Some(format!(
            "fabric wrapper {wrapper} is not an ancestor of this process"
        ));
    };
    if start_of(wrapper) != Some(start) {
        return Some(format!(
            "fabric wrapper {wrapper} is gone (stale FABRIC_JOB)"
        ));
    }
    if !holds_lock(&chain[..=depth]) {
        return Some(format!(
            "fabric job {marker} does not hold {HOSTMEM_LOCK} (no --hostmem-lock)"
        ));
    }
    None
}

/// Panic before a Flash-Next load outside its own fabric hold
/// ([`hold_refusal`]), naming the processes that hold the host lock.
fn require_fabric_hold() {
    let args: Vec<String> = std::env::args().collect();
    let threads_one = std::env::var("RUST_TEST_THREADS").is_ok_and(|v| v == "1")
        || args.iter().any(|a| a == "--test-threads=1")
        || args
            .windows(2)
            .any(|w| w[0] == "--test-threads" && w[1] == "1");
    let mut chain = vec![std::process::id()];
    while let Some(ppid) = proc_stat(*chain.last().unwrap(), 1).filter(|&p| p > 1) {
        chain.push(ppid as u32);
    }
    let marker = std::env::var("FABRIC_JOB").unwrap_or_default();
    let refusal = hold_refusal(
        threads_one,
        &marker,
        &chain,
        |pid| proc_stat(pid, 19),
        |pids| !hostmem_lock_holders(pids.iter().copied()).is_empty(),
    );
    if let Some(why) = refusal {
        let all = std::fs::read_dir("/proc")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok());
        panic!(
            "refusing to load Flash-Next: {why}. {HOSTMEM_LOCK} holders (pid, mode): {:?}. Run \
             inside `fabric-run.sh ... --hostmem-gb <true peak> --hostmem-lock -- \
             q8_ef_product_routes_hw --ignored --test-threads=1 ...` (see the module docs)",
            hostmem_lock_holders(all)
        );
    }
}

// ------------------------------------------------------- pure diagnostics

/// Index of the first element that differs, or the shorter length when one
/// sequence is a prefix of the other.
fn first_difference<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

/// `(nonzero, nonfinite)` f16 elements of little-endian residual `bytes`. An
/// element is non-zero when any bit differs from `+0.0` (a reset writes `+0.0`
/// exactly); it is non-finite when its exponent field is all ones.
fn scan_f16_bits(bytes: &[u8]) -> (u64, u64) {
    let (mut nonzero, mut nonfinite) = (0u64, 0u64);
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        nonzero += u64::from(bits != 0);
        nonfinite += u64::from(bits & 0x7c00 == 0x7c00);
    }
    (nonzero, nonfinite)
}

/// The residual of every GDN layer of one state at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidualScan {
    layers: usize,
    elements: u64,
    nonzero: u64,
    nonzero_layers: usize,
    nonfinite: u64,
}

impl ResidualScan {
    fn to_json(self) -> Value {
        json!({
            "layers": self.layers,
            "elements": self.elements,
            "nonzero_elements": self.nonzero,
            "nonzero_layers": self.nonzero_layers,
            "nonfinite_elements": self.nonfinite,
        })
    }
}

/// Download and scan every GDN layer's EF residual (full device sync first).
fn scan_residuals(
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    scratch: &mut Vec<u8>,
) -> Result<ResidualScan, String> {
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    let mut scan = ResidualScan::default();
    for (layer, state) in bundle.state.gdn.iter().enumerate() {
        let residual = state
            .ef_residual
            .as_ref()
            .ok_or_else(|| format!("GDN layer {layer} has no EF residual"))?;
        scratch.resize(residual.byte_size(), 0);
        gpu.hip
            .memcpy_dtoh(&mut scratch[..], &residual.buf)
            .map_err(|e| format!("download GDN layer {layer} residual: {e}"))?;
        let (nonzero, nonfinite) = scan_f16_bits(scratch);
        scan.layers += 1;
        scan.elements += (scratch.len() / 2) as u64;
        scan.nonzero += nonzero;
        scan.nonzero_layers += usize::from(nonzero > 0);
        scan.nonfinite += nonfinite;
    }
    Ok(scan)
}

type Digest32 = Vec<(String, String)>;

/// The category of a `debug_digest` key, for compact difference reports.
fn digest_category(key: &str) -> &'static str {
    if key.ends_with(".ef_residual") {
        "gdn.ef_residual"
    } else if key.starts_with("gdn[") && key.ends_with(".recurrent") {
        "gdn.recurrent"
    } else if key.starts_with("gdn[") && key.ends_with(".conv") {
        "gdn.conv"
    } else if key.starts_with("qsa[") {
        "qsa"
    } else if key.starts_with("row_capture") {
        "row_capture"
    } else if key == "position" || key == "gdn_live" {
        "position"
    } else if key.starts_with("ple") {
        "ple"
    } else {
        "other"
    }
}

/// Keys whose value differs between two digests, or exists in only one.
fn digest_diff(a: &Digest32, b: &Digest32) -> Vec<String> {
    let left: BTreeMap<&str, &str> = a.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let right: BTreeMap<&str, &str> = b.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let mut keys: Vec<String> = left
        .iter()
        .filter(|(key, value)| right.get(*key).map_or(true, |other| other != *value))
        .map(|(key, _)| (*key).to_string())
        .collect();
    keys.extend(
        right
            .keys()
            .filter(|key| !left.contains_key(*key))
            .map(|key| (*key).to_string()),
    );
    keys.sort();
    keys
}

fn category_counts(keys: &[String]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for key in keys {
        *counts.entry(digest_category(key)).or_insert(0) += 1;
    }
    counts
}

/// Keys of `categories` that differ between two digests (a hard comparison of
/// the Q8 state a restore must reproduce).
fn digest_diff_in(a: &Digest32, b: &Digest32, categories: &[&str]) -> Vec<String> {
    digest_diff(a, b)
        .into_iter()
        .filter(|key| categories.contains(&digest_category(key)))
        .collect()
}

fn digest_to_json(digest: &Digest32) -> Value {
    Value::Array(digest.iter().map(|(k, v)| json!([k, v])).collect())
}

fn digest_from_json(value: &Value) -> Result<Digest32, String> {
    value
        .as_array()
        .ok_or("digest is not an array")?
        .iter()
        .map(|row| {
            let key = row.get(0).and_then(Value::as_str);
            let digest = row.get(1).and_then(Value::as_str);
            key.zip(digest)
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .ok_or_else(|| format!("malformed digest row {row}"))
        })
        .collect()
}

/// The live digest of the target state; every GDN layer must contribute its
/// `ef_residual` row (`DigestRow::GdnResidual`).
fn live_digest(bundle: &Qwen4Bundle, gpu: &mut Gpu) -> Result<Digest32, String> {
    let rows = bundle.state.debug_digest(gpu).map_err(|e| e.to_string())?;
    let layers = bundle.state.gdn.len();
    let residual_rows = rows
        .iter()
        .filter(|(key, _)| key.ends_with(".ef_residual"))
        .count();
    if layers == 0 || residual_rows != layers {
        return Err(format!(
            "live digest has {residual_rows} ef_residual rows for {layers} GDN layers"
        ));
    }
    Ok(rows)
}

fn residual_rows_in(digest: &Digest32) -> usize {
    digest
        .iter()
        .filter(|(key, _)| key.ends_with(".ef_residual"))
        .count()
}

// ------------------------------------------------------- window records

/// One native MTP window as the child observed it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WindowRow {
    index: u64,
    position_before: u64,
    position_after: u64,
    k_budget: u64,
    drafts: u64,
    accepted: u64,
    committed: u64,
    mtp_windows_after: u64,
    ar_windows_after: u64,
    ngram_windows_after: u64,
    retired_after: bool,
    resets_before: u64,
    resets_after: u64,
    /// Residual scan after a rejected window (`None`: not scanned).
    residual_nonzero: Option<u64>,
    residual_nonfinite: Option<u64>,
    /// Verify-row logits observed by the lab window tap (`None`: no tap).
    logits_rows: Option<u64>,
    logits_nonfinite: Option<u64>,
    /// Capture-ring half the window's armed verify wrote (lab state tap).
    ring_half: Option<u64>,
    /// Residual transitions the lab state tap saw (`None`: no tap).
    residual_events: Option<Vec<String>>,
}

fn field_u64(value: &Value, name: &str) -> Result<u64, String> {
    value[name]
        .as_u64()
        .ok_or_else(|| format!("window row field {name} is missing or not an integer"))
}

fn field_opt_u64(value: &Value, name: &str) -> Result<Option<u64>, String> {
    match &value[name] {
        Value::Null => Ok(None),
        other => other
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("window row field {name} is not an integer or null")),
    }
}

impl WindowRow {
    fn rejected(&self) -> bool {
        self.accepted < self.drafts
    }

    /// The target verified drafts + 1 rows; committing fewer than that rolls
    /// rows back from the ring: a rejection (accepted < drafts) or a full
    /// accept whose commit stopped early, e.g. at a turn boundary.
    fn rolled_back(&self) -> bool {
        self.committed <= self.drafts
    }

    fn to_json(&self, counters: Value) -> Value {
        json!({
            "index": self.index,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "k_budget": self.k_budget,
            "drafts": self.drafts,
            "accepted": self.accepted,
            "committed": self.committed,
            "mtp_windows_after": self.mtp_windows_after,
            "ar_windows_after": self.ar_windows_after,
            "ngram_windows_after": self.ngram_windows_after,
            "retired_after": self.retired_after,
            "resets_before": self.resets_before,
            "resets_after": self.resets_after,
            "residual_nonzero_elements": self.residual_nonzero,
            "residual_nonfinite_elements": self.residual_nonfinite,
            "logits_rows": self.logits_rows,
            "logits_nonfinite": self.logits_nonfinite,
            "ring_half": self.ring_half,
            "residual_events": self.residual_events,
            "hip_counters": counters,
        })
    }

    fn from_json(value: &Value) -> Result<WindowRow, String> {
        Ok(WindowRow {
            index: field_u64(value, "index")?,
            position_before: field_u64(value, "position_before")?,
            position_after: field_u64(value, "position_after")?,
            k_budget: field_u64(value, "k_budget")?,
            drafts: field_u64(value, "drafts")?,
            accepted: field_u64(value, "accepted")?,
            committed: field_u64(value, "committed")?,
            mtp_windows_after: field_u64(value, "mtp_windows_after")?,
            ar_windows_after: field_u64(value, "ar_windows_after")?,
            ngram_windows_after: field_u64(value, "ngram_windows_after")?,
            retired_after: value["retired_after"]
                .as_bool()
                .ok_or("window row field retired_after is not a bool")?,
            resets_before: field_u64(value, "resets_before")?,
            resets_after: field_u64(value, "resets_after")?,
            residual_nonzero: field_opt_u64(value, "residual_nonzero_elements")?,
            residual_nonfinite: field_opt_u64(value, "residual_nonfinite_elements")?,
            logits_rows: field_opt_u64(value, "logits_rows")?,
            logits_nonfinite: field_opt_u64(value, "logits_nonfinite")?,
            ring_half: field_opt_u64(value, "ring_half")?,
            residual_events: match &value["residual_events"] {
                Value::Null => None,
                Value::Array(items) => Some(
                    items
                        .iter()
                        .map(|item| {
                            item.as_str()
                                .map(str::to_string)
                                .ok_or("window row residual_events entry is not a string")
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                _ => return Err("window row residual_events is not an array or null".into()),
            },
        })
    }
}

/// One `QWEN4_MTP_TRACE` window event of the drafter.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TraceWindow {
    position: u64,
    k: u64,
    accepted: u64,
    /// `batched` or `interleaved`.
    route: String,
    /// `gdn_residual_resets` (cumulative, emitted before the window's own
    /// rollback).
    resets: Option<u64>,
}

fn trace_number(line: &str, key: &str) -> Option<u64> {
    let tail = &line[line.find(key)? + key.len()..];
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn trace_string(line: &str, key: &str) -> Option<String> {
    let tail = &line[line.find(key)? + key.len()..];
    let tail = tail.strip_prefix('"')?;
    Some(tail[..tail.find('"')?].to_string())
}

fn parse_trace(stderr: &str) -> Result<Vec<TraceWindow>, String> {
    let mut windows = Vec::new();
    for line in stderr.lines() {
        if !(line.starts_with("QWEN4_MTP_TRACE ") && line.contains("\"event\":\"window\"")) {
            continue;
        }
        let missing = |what: &str| format!("trace window line without {what}: {line}");
        windows.push(TraceWindow {
            position: trace_number(line, "\"position\":").ok_or_else(|| missing("position"))?,
            k: trace_number(line, "\"k\":").ok_or_else(|| missing("k"))?,
            accepted: trace_number(line, "\"accepted\":").ok_or_else(|| missing("accepted"))?,
            route: trace_string(line, "\"verify_route\":")
                .ok_or_else(|| missing("verify_route"))?,
            resets: trace_number(line, "\"gdn_residual_resets\":"),
        });
    }
    Ok(windows)
}

/// What the lab window tap (`feature = "lab"`) observed of one window: the
/// verify-row logits' non-finite count, the capture-ring half and the
/// residual transitions. Everything is `None` when the tap is not available.
#[derive(Default)]
struct WindowTap {
    logits_rows: Option<u64>,
    logits_nonfinite: Option<u64>,
    ring_half: Option<u64>,
    residual_events: Option<Vec<String>>,
    /// The tap's own `gdn_residual_resets` before and after the window.
    resets_seen: Option<(u64, u64)>,
}

/// Arm the drafter's read-only window tap (no-op without `feature = "lab"`,
/// where [`take_window_tap`] reports nothing observed).
#[cfg(feature = "lab")]
fn arm_window_taps(drafter: &mut Qwen4MtpDrafter) {
    drafter.lab_observe_windows(hipfire_arch_qwen4::lab_window::LabWindowArm {
        armed: true,
        keep_logits: false,
    });
}

#[cfg(not(feature = "lab"))]
fn arm_window_taps(_drafter: &mut Qwen4MtpDrafter) {}

#[cfg(feature = "lab")]
fn take_window_tap(drafter: &mut Qwen4MtpDrafter) -> Result<WindowTap, String> {
    let Some(window) = drafter.lab_take_window_observation() else {
        return Ok(WindowTap::default());
    };
    if let Some(error) = &window.error {
        return Err(format!("the lab tap saw the window fail: {error}"));
    }
    let rows: usize = window.forwards.iter().map(|f| f.rows).sum();
    let nonfinite: u64 = window
        .forwards
        .iter()
        .flat_map(|f| f.nonfinite.iter())
        .sum();
    Ok(WindowTap {
        logits_rows: Some(rows as u64),
        logits_nonfinite: Some(nonfinite),
        ring_half: window.ring_half.map(u64::from),
        residual_events: Some(
            window
                .state
                .residual
                .iter()
                .map(|event| format!("{:?}", event.kind))
                .collect(),
        ),
        resets_seen: Some((
            window.gdn_residual_resets_before,
            window.gdn_residual_resets_after,
        )),
    })
}

#[cfg(not(feature = "lab"))]
fn take_window_tap(_drafter: &mut Qwen4MtpDrafter) -> Result<WindowTap, String> {
    Ok(WindowTap::default())
}

// ---------------------------------------------------------------- arms

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Case {
    P1,
    Seasons,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::P1 => "p1",
            Case::Seasons => "seasons",
        }
    }

    fn generated(self) -> usize {
        match self {
            Case::P1 => 1200,
            Case::Seasons => 256,
        }
    }

    fn parse(name: &str) -> Option<Case> {
        [Case::P1, Case::Seasons]
            .into_iter()
            .find(|case| case.name() == name)
    }
}

/// `<case>:<run>` as carried by `HIPFIRE_MTP_IDENTITY_CASE`.
fn parse_case_run(value: &str) -> Result<(Case, u32), String> {
    let (case, run) = value
        .split_once(':')
        .ok_or_else(|| format!("{CASE_ENV}={value:?} is not <case>:<run>"))?;
    let case = Case::parse(case).ok_or_else(|| format!("unknown case in {CASE_ENV}={value:?}"))?;
    let run: u32 = run
        .parse()
        .map_err(|_| format!("run in {CASE_ENV}={value:?} is not a number"))?;
    if run == 0 {
        return Err(format!("run in {CASE_ENV}={value:?} must be 1 or more"));
    }
    Ok((case, run))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Ar,
    Batched,
    Incremental,
    Adaptive,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    incremental: Option<&'static str>,
    draft_head: Option<&'static str>,
    route: Route,
}

const ARMS: [Arm; 5] = [
    Arm {
        name: "ar",
        incremental: None,
        draft_head: None,
        route: Route::Ar,
    },
    Arm {
        name: "batched",
        incremental: Some("0"),
        draft_head: None,
        route: Route::Batched,
    },
    Arm {
        name: "incremental",
        incremental: Some("1"),
        draft_head: None,
        route: Route::Incremental,
    },
    Arm {
        name: "adaptive",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
    },
    Arm {
        name: "full-head",
        incremental: Some("0"),
        draft_head: Some("full"),
        route: Route::Batched,
    },
];

fn stem(arm: Arm, run: u32) -> String {
    format!("{}.r{run}", arm.name)
}

/// What a parent changes in an arm's child environment: everything that can
/// force a route or alter the drafter is cleared, then the arm's own settings
/// are applied. The AR floor is left at its default (on) for every arm (a
/// forced route bypasses it). The GDN test mode and the launch backend are
/// resolved once by the parent and forwarded explicitly, so a child never runs
/// a different mode.
fn arm_overrides(arm: Arm) -> Result<Vec<(&'static str, Option<&'static str>)>, String> {
    let (state_request, _) = hw_identity::gdn_test_state_quant()?;
    if state_request != GDN_MODE {
        return Err(format!(
            "{} resolved to {state_request}, not {GDN_MODE}",
            hw_identity::GDN_TEST_STATE_QUANT_ENV
        ));
    }
    Ok(vec![
        ("HIPFIRE_MTP_AR_FLOOR", None),
        ("HIPFIRE_MTP_PHASE_TIMING", None),
        ("HIPFIRE_MTP_NGRAM", None),
        ("HIPFIRE_MTP_TRACE", (arm.route != Route::Ar).then_some("1")),
        ("HIPFIRE_MTP_INCREMENTAL", arm.incremental),
        ("HIPFIRE_MTP_DRAFT_HEAD", arm.draft_head),
        ("HIPFIRE_NGRAM_MOD_N_MATCH", None),
        ("HIPFIRE_NGRAM_MOD_N_MIN", None),
        ("HIPFIRE_NGRAM_MOD_N_MAX", None),
        (hw_identity::GDN_TEST_STATE_QUANT_ENV, Some(state_request)),
        (REPLAY_BACKEND_ENV, Some(replay_backend_request()?)),
    ])
}

/// The child must already run under its arm's environment.
fn require_arm_env(arm: Arm) -> Result<(), String> {
    for (name, want) in arm_overrides(arm)? {
        let have = std::env::var(name).ok();
        if have.as_deref() != want {
            return Err(format!(
                "{name} is {have:?} but arm {} requires {want:?} (the parent forwards it explicitly)",
                arm.name
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------- window rules

/// Counters of one arm run, from the child's window rows joined with the
/// drafter's trace events by window start position.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RouteStats {
    windows: usize,
    batched: usize,
    interleaved: usize,
    target_only: usize,
    drafts: u64,
    accepted: u64,
    committed: u64,
    rejected: usize,
    rejected_batched: usize,
    rejected_interleaved: usize,
    truncated_batched: usize,
    full_commit_batched: usize,
    zero_checked: usize,
    logits_observed: usize,
    logits_unobserved: usize,
    ring_halves: [usize; 2],
    batched_reset_events: u64,
    interleaved_reset_events: u64,
}

impl RouteStats {
    fn to_json(&self) -> Value {
        json!({
            "windows": self.windows,
            "batched_windows": self.batched,
            "interleaved_windows": self.interleaved,
            "target_only_windows": self.target_only,
            "drafts": self.drafts,
            "accepted": self.accepted,
            "committed": self.committed,
            "tau_committed_per_window": ratio(self.committed, self.windows as u64),
            "accepted_per_draft": ratio(self.accepted, self.drafts),
            "rejected_windows": self.rejected,
            "rejected_batched_windows": self.rejected_batched,
            "rejected_interleaved_windows": self.rejected_interleaved,
            "truncated_batched_windows": self.truncated_batched,
            "full_commit_batched_windows": self.full_commit_batched,
            "zero_checked_rolled_back_batched_windows": self.zero_checked,
            "windows_with_observed_verify_logits": self.logits_observed,
            "windows_without_observed_verify_logits": self.logits_unobserved,
            "ring_half_0_windows": self.ring_halves[0],
            "ring_half_1_windows": self.ring_halves[1],
            "reset_events_on_batched_windows": self.batched_reset_events,
            "reset_events_on_interleaved_windows": self.interleaved_reset_events,
        })
    }
}

/// The route and reset rules (module docs) over one run's windows. Returns the
/// counters and every violated rule. `zero_check` is whether a rolled-back
/// batched window ends in a ring rollback with no target forward after it.
fn check_windows(
    route: Route,
    rows: &[WindowRow],
    trace: &[TraceWindow],
    zero_check: bool,
) -> (RouteStats, Vec<String>) {
    let mut stats = RouteStats::default();
    let mut errors = Vec::new();
    for window in trace {
        if window.resets.is_none() {
            errors.push(format!(
                "trace window at position {} has no gdn_residual_resets field",
                window.position
            ));
        }
    }
    for row in rows {
        stats.windows += 1;
        stats.drafts += row.drafts;
        stats.accepted += row.accepted;
        stats.committed += row.committed;
        let at = format!("window {} (position {})", row.index, row.position_before);
        let delta = match row.resets_after.checked_sub(row.resets_before) {
            Some(delta) => delta,
            None => {
                errors.push(format!("{at}: reset counter went backwards"));
                0
            }
        };
        if row.rejected() {
            stats.rejected += 1;
        }
        if let Some(nonfinite) = row.residual_nonfinite {
            if nonfinite != 0 {
                errors.push(format!("{at}: {nonfinite} non-finite residual elements"));
            }
        }
        if let Some(events) = &row.residual_events {
            if events.len() as u64 != delta {
                errors.push(format!(
                    "{at}: the reset counter rose {delta}x but the state tap saw {} residual events",
                    events.len()
                ));
            }
        }
        if let Some(nonfinite) = row.logits_nonfinite {
            if nonfinite != 0 {
                errors.push(format!("{at}: {nonfinite} non-finite verify-row logits"));
            }
        }
        match row.logits_rows {
            Some(rows) if rows > 0 => stats.logits_observed += 1,
            _ => stats.logits_unobserved += 1,
        }
        if let Some(half) = row.ring_half {
            stats.ring_halves[(half & 1) as usize] += 1;
        }
        let traced = trace.iter().find(|t| t.position == row.position_before);
        if row.drafts == 0 {
            stats.target_only += 1;
            if delta != 0 {
                errors.push(format!(
                    "{at}: target-only window reset the residual {delta}x"
                ));
            }
            continue;
        }
        let Some(traced) = traced else {
            errors.push(format!("{at}: window with drafts has no trace event"));
            continue;
        };
        if traced.accepted != row.accepted {
            errors.push(format!(
                "{at}: trace accepted {} but the window accepted {}",
                traced.accepted, row.accepted
            ));
        }
        match traced.route.as_str() {
            "batched" => {
                stats.batched += 1;
                if row.rolled_back() {
                    if row.rejected() {
                        stats.rejected_batched += 1;
                    } else {
                        stats.truncated_batched += 1;
                    }
                    stats.batched_reset_events += delta;
                    if delta == 0 {
                        errors.push(format!(
                            "{at}: rolled-back batched window (accepted {}/{}, committed {}) did not reset the residual",
                            row.accepted, row.drafts, row.committed
                        ));
                    }
                    if zero_check {
                        match row.residual_nonzero {
                            Some(0) => stats.zero_checked += 1,
                            Some(n) => errors.push(format!(
                                "{at}: residual is not zero after the rolled-back batched window ({n} non-zero elements)"
                            )),
                            None => errors.push(format!(
                                "{at}: rolled-back batched window was not scanned for the zero check"
                            )),
                        }
                    }
                } else {
                    stats.full_commit_batched += 1;
                    if delta != 0 {
                        errors.push(format!(
                            "{at}: full-commit batched window (committed {} = drafts + 1) reset the residual {delta}x",
                            row.committed
                        ));
                    }
                }
            }
            "interleaved" => {
                stats.interleaved += 1;
                stats.interleaved_reset_events += delta;
                if row.rejected() {
                    stats.rejected_interleaved += 1;
                }
            }
            other => errors.push(format!("{at}: unknown verify route {other:?}")),
        }
    }
    let coverage = match route {
        Route::Ar => (rows.is_empty() && trace.is_empty())
            .then_some(())
            .ok_or("ar arm ran MTP windows"),
        Route::Batched => (stats.batched > 0 && stats.interleaved == 0 && stats.rejected_batched > 0)
            .then_some(())
            .ok_or("batched arm needs batched windows only and at least one rejected window (no reset coverage)"),
        Route::Incremental => (stats.interleaved > 0 && stats.batched == 0)
            .then_some(())
            .ok_or("incremental arm needs interleaved windows only"),
        Route::Adaptive => (stats.batched > 0)
            .then_some(())
            .ok_or("adaptive arm took no batched window"),
    };
    if let Err(why) = coverage {
        errors.push(format!(
            "route counters: {why} ({} batched, {} interleaved, {} rejected batched windows)",
            stats.batched, stats.interleaved, stats.rejected_batched
        ));
    }
    (stats, errors)
}

/// The deterministic part of a run's window history.
fn history(rows: &[WindowRow], trace: &[TraceWindow]) -> Vec<String> {
    rows.iter()
        .map(|row| {
            let route = trace
                .iter()
                .find(|t| t.position == row.position_before)
                .map_or("-", |t| t.route.as_str());
            format!(
                "{}:{}:{}:{}:{}:{}:{route}",
                row.position_before,
                row.drafts,
                row.accepted,
                row.committed,
                row.resets_before,
                row.resets_after
            )
        })
        .collect()
}

/// One window's route decision per entry: (position before, drafts, verify
/// route). Unlike [`history`] it has no outcome (accepted/committed/reset
/// counters), only what the chooser decided.
fn decisions(rows: &[WindowRow], trace: &[TraceWindow]) -> Vec<String> {
    rows.iter()
        .map(|row| {
            let route = trace
                .iter()
                .find(|t| t.position == row.position_before)
                .map_or("-", |t| t.route.as_str());
            format!("{}:{}:{route}", row.position_before, row.drafts)
        })
        .collect()
}

// ------------------------------------------------------------- loading

/// A loaded Flash-Next bundle.
struct Loaded {
    bundle: Qwen4Bundle,
    gpu: Gpu,
    tokenizer: Tokenizer,
    template: String,
    state_request: &'static str,
    state_format: Qwen4StateFormat,
    replay_backend: &'static str,
    backend: Qwen4KvBackend,
    vocab: usize,
}

/// Payload bytes of a manifest entry for expert placement sizing: an ordinary
/// tensor reads its stored size; a slab layer arena is one span of the
/// container, not an index tensor, and declares its byte size as its logical
/// shape.
fn slab_payload_bytes(
    hfq: &HfqFile,
    entry: &hipfire_runtime::weight_manifest::WeightEntry,
) -> Option<u64> {
    if entry
        .name
        .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
    {
        return match entry.logical_shape.as_slice() {
            [bytes] => Some(*bytes as u64),
            _ => None,
        };
    }
    hfq.tensor_data(&entry.name)
        .map(|(_, bytes)| bytes.len() as u64)
}

/// The loaded bundle's GDN state is `q8_ef`: Q8 recurrent storage, one F16
/// residual plane per layer, and a session scope that names the reset ABI.
fn resolved_state_evidence(
    bundle: &Qwen4Bundle,
    request: &str,
    state_format: Qwen4StateFormat,
) -> Result<String, String> {
    if state_format.gdn != GdnStateFormat::Q8Ef {
        return Err(format!("GDN state resolved to {:?}", state_format.gdn));
    }
    let config = &bundle.config;
    let (heads, key_dim, value_dim) = (
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    );
    let state_bytes = state_format.gdn.state_bytes(heads, key_dim, value_dim);
    let residual_bytes = state_format.gdn.residual_bytes(heads, key_dim, value_dim);
    if residual_bytes == 0 || bundle.state.gdn.is_empty() {
        return Err("q8_ef state has no residual or no GDN layers".into());
    }
    for (layer, state) in bundle.state.gdn.iter().enumerate() {
        if state.recurrent.dtype != state_format.gdn.dtype()
            || state.recurrent.byte_size() != state_bytes
        {
            return Err(format!("GDN layer {layer} recurrent is not the Q8 slot"));
        }
        match &state.ef_residual {
            Some(ef) if ef.dtype == DType::F16 && ef.byte_size() == residual_bytes => {}
            _ => {
                return Err(format!(
                    "GDN layer {layer} has no F16 residual of {residual_bytes} bytes"
                ))
            }
        }
    }
    let scope = SessionState::snapshot_scope(bundle, SessionRoute::Ar).ok_or("no session scope")?;
    if !scope.contains("q8_ef-reset") {
        return Err(format!("session scope {scope:?} lacks the q8_ef-reset ABI"));
    }
    Ok(format!(
        "{} state_quant={request} gdn_format={} gdn_layers={} recurrent_bytes_per_layer={state_bytes} residual_bytes_per_layer={residual_bytes} scope={scope}",
        hw_identity::GDN_TEST_STATE_QUANT_ENV,
        state_format.gdn.name(),
        bundle.state.gdn.len()
    ))
}

/// Load the model named by `model` under the `q8_ef` test mode with the
/// shared production expert residency of `greedy_mtp_identity_hw`, a forward
/// attached and (`cache_budget`) a session cache.
fn load(model: &Path, max_seq: usize, cache_budget: Option<u64>) -> Result<Loaded, String> {
    require_fabric_hold();
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    if let Some(devices) = hipfire_config::devices::startup_devices() {
        let archs = devices
            .iter()
            .map(|device| device.arch.as_str())
            .collect::<Vec<_>>();
        if residency::keeps_host_memory_out_of_reclaim(hipfire_arch_qwen4::ARCH_ID, &archs) {
            hip_bridge::keep_host_memory_out_of_reclaim(&format!(
                "Qwen4 model on {}",
                archs.join(",")
            ));
        }
    }
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let replay_backend = replay_backend_name(gpu.replay.request());
    let requested_backend = replay_backend_request()?;
    if replay_backend != requested_backend {
        return Err(format!(
            "{REPLAY_BACKEND_ENV} resolved to the {replay_backend} backend, not the {requested_backend} the parent forwarded"
        ));
    }
    let domain = cache_budget
        .map(|_| CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id));
    // Admission is GPU-arch aware: an expert-slab artifact admits only on a
    // supported arch. Initialising the GPU allocates no model memory.
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|e| e.to_string())?;
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|e| e.to_string())?;
    let (state_request, state_format) = hw_identity::resolve_gdn_test_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &gpu,
        &receipt.config,
    )?;
    if state_format.gdn != GdnStateFormat::Q8Ef {
        return Err(format!(
            "{} resolved to GDN state {:?}, not q8_ef",
            hw_identity::GDN_TEST_STATE_QUANT_ENV,
            state_format.gdn
        ));
    }
    let backend = Qwen4KvBackend::automatic(&gpu);
    let use_ranges = gpu.is_uma();
    let mut weights = receipt.manifest.weights.clone();
    let explicit = residency::expert_vram_layers_from_env()?;
    let placement = if use_ranges {
        explicit
    } else {
        let (non_expert, per_layer) =
            residency::resident_split(&weights, |entry| slab_payload_bytes(&hfq, entry))?;
        let experts =
            residency::routed_expert_bytes(&weights, |entry| slab_payload_bytes(&hfq, entry))?;
        let (free, _) = gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
        let free = free as u64;
        let chunk = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
            &gpu,
            max_seq,
            scratch_mode,
        );
        let context = hipfire_arch_qwen4::Qwen4ContextCommit::for_expert_reserve(
            backend,
            gpu.is_uma(),
            max_seq,
            chunk,
            gpu.vmm_recommended_granularity().unwrap_or(1),
        );
        // Keep every arm's placement identical, including the AR arm.
        let head = residency::language_head_dtype(&weights)
            .ok_or("qwen4: manifest has no language head")?;
        let row_capture = hipfire_arch_qwen4::mtp_spec::native_mtp_capture(
            &gpu,
            &receipt.config,
            state_format.gdn,
            MTP_K,
        );
        let mtp = hipfire_arch_qwen4::mtp_spec::native_mtp_device_bytes(
            &receipt.config,
            &context,
            chunk,
            MTP_K,
            head,
            row_capture,
            hipfire_arch_qwen4::mtp_spec::native_mtp_fill_scratch(&gpu),
        )
        .ok_or("qwen4: native MTP device bytes overflow")?;
        let gather = rdna_compute::tensor_ops::qsa_gather_workspace_needed(
            &gpu,
            state_format.qsa,
            state_format.qsa_f16_arena,
        )
        .then(|| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                receipt.config.num_key_value_heads,
                context.tokens,
            )
            .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
            .ok_or("qwen4: QSA gather scratch size overflows")
        })
        .transpose()?;
        let reserve = residency::auto_vram_reserve(
            &receipt.config,
            &context,
            chunk,
            scratch_mode,
            state_format.layout(),
            Some(mtp),
            gather,
        )?;
        match residency::resolve_expert_vram_layers(explicit, || {
            Ok(residency::fits_fully_resident(
                free, non_expert, experts, reserve,
            ))
        })? {
            Some(residency::ExpertVramLayers::Auto) => {
                let required = non_expert
                    .checked_add(reserve)
                    .ok_or("qwen4: non-expert weights plus reserve overflow")?;
                if free < required {
                    return Err(
                        "qwen4: not enough free VRAM for non-expert weights plus auto reserve"
                            .into(),
                    );
                }
                Some(residency::ExpertVramLayers::Layers(
                    residency::auto_vram_layers(
                        free,
                        non_expert,
                        per_layer,
                        receipt.config.num_hidden_layers,
                        reserve,
                    ),
                ))
            }
            placement => placement,
        }
    };
    let mut vram_layers = residency::moe_layer_count(&receipt.config);
    if let Some(policy) = placement {
        if use_ranges {
            return Err(format!(
                "qwen4: {} places experts in host RAM, which only a discrete GPU needs",
                residency::EXPERT_VRAM_LAYERS_ENV
            ));
        }
        let residency::ExpertVramLayers::Layers(layers) = policy else {
            return Err("qwen4: unresolved auto expert placement".into());
        };
        residency::place_routed_experts(&mut weights, layers);
        vram_layers = layers.min(receipt.config.num_hidden_layers);
    }
    let host_bytes =
        residency::host_mapped_bytes(&weights, |entry| slab_payload_bytes(&hfq, entry))?;
    residency::check_host_ram(
        host_bytes,
        rdna_compute::kv_slots::mem_available_bytes(),
        residency::ttm_pool_estimate(),
    )?;
    residency::check_gtt_cap(host_bytes, residency::gtt_budget())?;
    eprintln!(
        "qwen4 expert placement: {vram_layers} resident layers, {host_bytes} host-mapped bytes"
    );
    if use_ranges {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if use_ranges || entry.residency.is_external() {
                return source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name));
            }
            let (info, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))?;
            Ok(SourcePayload::Borrowed { info, bytes })
        },
    )
    .map_err(|e| e.to_string())?;
    let vocab = receipt.config.vocab_size;
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    if cache_budget.is_some() {
        bundle.set_turn_snapshots(false);
    }
    bundle
        .attach_forward(&mut gpu, max_seq, scratch_mode)
        .map_err(|e| e.to_string())?;
    let evidence = resolved_state_evidence(&bundle, state_request, state_format)?;
    println!(
        "{evidence} replay_backend={replay_backend} backend={}",
        backend.name()
    );
    if let (Some(budget), Some(domain)) = (cache_budget, domain) {
        bundle.attach_session_cache(SessionCache::new(domain, budget));
    }
    Ok(Loaded {
        bundle,
        gpu,
        tokenizer,
        template,
        state_request,
        state_format,
        replay_backend,
        backend,
        vocab,
    })
}

// ----------------------------------------------------------- AR decoding

/// The first maximal index of `logits`.
fn host_argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// Download `logits` into `host` and require every value finite.
fn check_logits_finite(
    gpu: &Gpu,
    logits: &GpuTensor,
    host: &mut Vec<f32>,
    what: &str,
) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    gpu.download_f32_into(logits, host)
        .map_err(|e| format!("{what}: download logits: {e}"))?;
    match host.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(format!(
            "{what}: non-finite logit {} at vocabulary index {index}",
            host[index]
        )),
        None => Ok(()),
    }
}

/// Decode `n` greedy tokens, checking the logits finite after every step.
/// `start`: `None` takes the device argmax of the logits the previous forward
/// left (valid only right after a prefill or an AR step); `Some(token)` feeds
/// that token first. Returns the tokens fed and the next pending token (the
/// host argmax of the last logits, which restores resume from).
fn ar_decode(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    host: &mut Vec<f32>,
    start: Option<u32>,
    n: usize,
    what: &str,
) -> Result<(Vec<u32>, u32), String> {
    let mut ids = Vec::with_capacity(n);
    let mut first = start;
    for step in 0..n {
        let id = bundle
            .forward_token_or_argmax(gpu, first.take(), logits)
            .map_err(|e| e.to_string())?;
        ids.push(id);
        check_logits_finite(gpu, logits, host, &format!("{what} step {step}"))?;
    }
    Ok((ids, host_argmax(host)))
}

// ----------------------------------------------------- product-route arm

struct RunData {
    ids: Vec<u32>,
    text: String,
    state: Value,
    digest: Digest32,
    windows: Vec<WindowRow>,
    request_stats: String,
    trace: Vec<TraceWindow>,
    header: String,
}

fn spawn_arm(
    arm: Arm,
    case: Case,
    run: u32,
    model: &Path,
    model_id: &Value,
    dir: &Path,
) -> Result<RunData, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let expected = hw_identity::config_identity(
        &hw_identity::env_snapshot(&arm_overrides(arm)?, &CONTROL_ENV),
        model_id,
    );
    let stem = stem(arm, run);
    // An earlier result of this arm must never stand in for this run.
    for stale in [
        "ids.json",
        "txt",
        "identity.json",
        "windows.json",
        "state.json",
    ] {
        let _ = fs::remove_file(dir.join(format!("{stem}.{stale}")));
    }
    let mut child = Command::new(&exe);
    child
        .args([
            "--exact",
            ARM_TEST,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm.name)
        .env(CASE_ENV, format!("{}:{run}", case.name()))
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .stdin(Stdio::null());
    for (name, value) in arm_overrides(arm)? {
        match value {
            Some(value) => {
                child.env(name, value);
            }
            None => {
                child.env_remove(name);
            }
        }
    }
    let output = child
        .output()
        .map_err(|e| format!("spawn {stem} arm: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::write(dir.join(format!("{stem}.out")), &output.stdout).map_err(|e| e.to_string())?;
    fs::write(dir.join(format!("{stem}.err")), &output.stderr).map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{stem} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        ));
    }
    let read = |suffix: &str| -> Result<Vec<u8>, String> {
        fs::read(dir.join(format!("{stem}.{suffix}"))).map_err(|e| format!("{stem}.{suffix}: {e}"))
    };
    let json = |suffix: &str| -> Result<Value, String> {
        serde_json::from_slice(&read(suffix)?).map_err(|e| format!("{stem}.{suffix}: {e}"))
    };
    let ids: Vec<u32> =
        serde_json::from_value(json("ids.json")?).map_err(|e| format!("{stem}.ids.json: {e}"))?;
    if ids.len() != case.generated() {
        return Err(format!(
            "{stem}: {} ids, expected {}",
            ids.len(),
            case.generated()
        ));
    }
    let recorded = json("identity.json")?;
    if recorded["config"] != expected {
        return Err(format!(
            "{stem} arm ran under a different identity than the parent expects (logs in {})",
            dir.display()
        ));
    }
    if recorded["gdn_state"] != json!({"request": GDN_MODE, "resolved": GDN_MODE}) {
        return Err(format!(
            "{stem} arm resolved a different GDN state format than {}={GDN_MODE} requests: {}",
            hw_identity::GDN_TEST_STATE_QUANT_ENV,
            recorded["gdn_state"]
        ));
    }
    let want_backend = replay_backend_request()?;
    if recorded["graph_policy"]["replay_backend"] != json!(want_backend) {
        return Err(format!(
            "{stem} arm ran a different launch backend than {REPLAY_BACKEND_ENV} requests"
        ));
    }
    let state = json("state.json")?;
    let digest = digest_from_json(&state["digest"])?;
    let (windows, request_stats) = if arm.route == Route::Ar {
        (Vec::new(), String::new())
    } else {
        let doc = json("windows.json")?;
        let rows = doc["windows"]
            .as_array()
            .ok_or_else(|| format!("{stem}.windows.json has no windows array"))?
            .iter()
            .map(WindowRow::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        (
            rows,
            doc["request_stats"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        )
    };
    let trace = parse_trace(&stderr)?;
    let text = String::from_utf8_lossy(&read("txt")?).into_owned();
    let header = format!(
        "gdn_state_quant={GDN_MODE} gdn_format={GDN_MODE}\nreplay_backend={want_backend} replay_state={} HIPFIRE_GRAPH={} ({GRAPH_POLICY_NOTE})\n",
        recorded["graph_policy"]["replay_state"], recorded["graph_policy"]["HIPFIRE_GRAPH"],
    );
    Ok(RunData {
        ids,
        text,
        state,
        digest,
        windows,
        request_stats,
        trace,
        header,
    })
}

/// Hard failures and INCOMPLETE (unobservable mandatory) items of one run's
/// `state.json`.
fn check_state(arm: Arm, label: &str, data: &RunData) -> (Vec<String>, Vec<String>) {
    let (mut failures, mut incomplete) = (Vec::new(), Vec::new());
    let state = &data.state;
    let layers = state["layers"].as_u64().unwrap_or(0) as usize;
    if layers == 0 || residual_rows_in(&data.digest) != layers {
        failures.push(format!(
            "{label}: final digest has {} ef_residual rows for {layers} GDN layers",
            residual_rows_in(&data.digest)
        ));
    }
    if state["resets"]["at_load"] != json!(1) {
        failures.push(format!(
            "{label}: reset counter at load is {}, expected 1 (the initial reset)",
            state["resets"]["at_load"]
        ));
    }
    if state["initial_residual"]["nonzero_elements"] != json!(0)
        || state["initial_residual"]["nonfinite_elements"] != json!(0)
    {
        failures.push(format!(
            "{label}: residual after load is not zero and finite: {}",
            state["initial_residual"]
        ));
    }
    if state["final_residual"]["nonfinite_elements"] != json!(0) {
        failures.push(format!(
            "{label}: final residual has non-finite elements: {}",
            state["final_residual"]
        ));
    }
    if arm.route == Route::Ar {
        if state["final_residual"]["nonzero_layers"]
            .as_u64()
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "{label}: the AR arm ended with an all-zero residual: the EF plane is not live"
            ));
        }
        if state["logits"]["observed"] != json!(true) || state["logits"]["nonfinite"] != json!(0) {
            failures.push(format!(
                "{label}: AR logits were not observed finite: {}",
                state["logits"]
            ));
        }
    } else {
        let capture = state["row_capture"].as_bool().unwrap_or(false);
        let (stats, errors) = check_windows(arm.route, &data.windows, &data.trace, capture);
        failures.extend(errors.into_iter().map(|e| format!("{label}: {e}")));
        if stats.logits_unobserved > 0 || stats.logits_observed == 0 {
            incomplete.push(format!(
                "{label}: non-finite check of the MTP verify-row logits is unobservable for {} of {} windows: {}",
                stats.logits_unobserved.max(usize::from(stats.logits_observed == 0)),
                stats.windows,
                state["logits"]["reason"].as_str().unwrap_or(MTP_LOGITS_UNOBSERVABLE)
            ));
        }
    }
    (failures, incomplete)
}

struct PairVerdict {
    ids_equal: bool,
    first_id_diff: Option<usize>,
    digest_equal: bool,
    digest_diff_keys: Vec<String>,
    /// Diagnostic context only: whether the two runs took the same window
    /// history (positions, drafts, accepted, committed, reset counters,
    /// verify route). It is never an assertion of its own.
    history_equal: bool,
    first_history_diff: Option<usize>,
    /// First window where the two runs' route decisions (position before,
    /// drafts, verify route) differ; `None`: identical decisions. Excuses an
    /// IDs/digest difference on the wall-clock `adaptive` route only.
    first_decision_diff: Option<usize>,
}

fn compare_pair(a: &RunData, b: &RunData) -> PairVerdict {
    let first_id_diff = first_difference(&a.ids, &b.ids);
    let digest_diff_keys = digest_diff(&a.digest, &b.digest);
    let (ha, hb) = (history(&a.windows, &a.trace), history(&b.windows, &b.trace));
    let first_history_diff = first_difference(&ha, &hb);
    let (da, db) = (
        decisions(&a.windows, &a.trace),
        decisions(&b.windows, &b.trace),
    );
    let first_decision_diff = first_difference(&da, &db);
    PairVerdict {
        ids_equal: first_id_diff.is_none(),
        first_id_diff,
        digest_equal: digest_diff_keys.is_empty(),
        digest_diff_keys,
        history_equal: first_history_diff.is_none(),
        first_history_diff,
        first_decision_diff,
    }
}

impl PairVerdict {
    /// The window-history context appended to a failure message.
    fn context(&self) -> String {
        if self.history_equal {
            "the two window histories are identical".to_string()
        } else {
            format!(
                "the two window histories differ first at window {:?} (route/accept decisions differ between the runs)",
                self.first_history_diff
            )
        }
    }

    /// The adaptive-only diagnostic verdict: the wall-clock chooser took
    /// different route decisions in the two runs, so two-run equality is not
    /// asserted.
    fn diagnostic_note(&self, route: Route) -> Option<String> {
        match (route, self.first_decision_diff) {
            (Route::Adaptive, Some(window)) => Some(format!(
                "adaptive: route decisions differ at window {window} (wall-clock chooser) -> two-run equality is diagnostic"
            )),
            _ => None,
        }
    }

    /// Hard failures: both fresh runs of every arm must give identical IDs and
    /// an identical final live digest, except an `adaptive` pair whose route
    /// decisions differ (the chooser is wall-clock driven; PLAN requires
    /// same-route determinism), which is diagnostic only.
    fn failures(&self, label: &str, route: Route) -> Vec<String> {
        let mut failures = Vec::new();
        if self.diagnostic_note(route).is_some() {
            return failures;
        }
        if !self.ids_equal {
            failures.push(format!(
                "{label}: the two fresh runs' IDs differ at generated index {:?}; {}",
                self.first_id_diff,
                self.context()
            ));
        }
        if !self.digest_equal {
            failures.push(format!(
                "{label}: the two fresh runs' final live digests differ in {} rows: {:?}; {}",
                self.digest_diff_keys.len(),
                category_counts(&self.digest_diff_keys),
                self.context()
            ));
        }
        failures
    }

    fn to_json(&self, route: Route) -> Value {
        json!({
            "ids_identical": self.ids_equal,
            "first_id_diff": self.first_id_diff,
            "final_digest_identical": self.digest_equal,
            "final_digest_differing_categories": category_counts(&self.digest_diff_keys),
            "window_history_identical": self.history_equal,
            "first_window_history_diff": self.first_history_diff,
            "decisions_identical": self.first_decision_diff.is_none(),
            "first_decision_diff": self.first_decision_diff,
            "diagnostic_only": self.diagnostic_note(route),
            "note": "hard: IDs and final live digest of the two fresh runs must be identical for every arm, except an adaptive pair whose route decisions (position before, drafts, verify route per window) differ (wall-clock chooser; PLAN requires same-route determinism): then the comparison is diagnostic; the window history is diagnostic context",
        })
    }
}

struct CaseReport {
    summary: Value,
    text: String,
    failures: Vec<String>,
    incomplete: Vec<String>,
}

fn run_product_case(case: Case, model: &Path, base: &Path) -> Result<CaseReport, String> {
    let dir = base.join(case.name());
    fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let model_id = hw_identity::model_identity(model, &dir)?;
    let (mut failures, mut incomplete) = (Vec::new(), Vec::new());
    let mut runs: Vec<(Arm, Vec<RunData>)> = Vec::new();
    for arm in ARMS {
        let mut pair = Vec::new();
        for run in 1..=2u32 {
            let data = spawn_arm(arm, case, run, model, &model_id, &dir)?;
            let label = format!("{}/{}.r{run}", case.name(), arm.name);
            let (f, i) = check_state(arm, &label, &data);
            failures.extend(f);
            incomplete.extend(i);
            eprintln!(
                "{label}: {} ids, {} windows, final resets {}",
                data.ids.len(),
                data.windows.len(),
                data.state["resets"]["final"]
            );
            pair.push(data);
        }
        runs.push((arm, pair));
    }
    let mut header = String::new();
    let mut text = String::new();
    let mut arms_json = serde_json::Map::new();
    let ar_run = &runs[0].1[0];
    for (arm, pair) in &runs {
        let label = format!("{}/{}", case.name(), arm.name);
        let verdict = compare_pair(&pair[0], &pair[1]);
        failures.extend(verdict.failures(&label, arm.route));
        let first = &pair[0];
        if header.is_empty() {
            header = first.header.clone();
        }
        let capture = first.state["row_capture"].as_bool().unwrap_or(false);
        let (stats, _) = check_windows(arm.route, &first.windows, &first.trace, capture);
        let vs_ar = first_difference(&ar_run.ids, &first.ids);
        let ar_digest_diff = digest_diff(&ar_run.digest, &first.digest);
        let ar_id = vs_ar.and_then(|i| ar_run.ids.get(i).copied());
        let arm_id = vs_ar.and_then(|i| first.ids.get(i).copied());
        text.push_str(&format!(
            "{}: ids {} vs ar: {}; windows {} (batched {}, interleaved {}, rejected {}), tau {:.3?}, accepted/drafts {}/{}, resets at load/after prefill/final {}/{}/{}; two fresh runs: ids {}, final digest {}{}\n",
            arm.name,
            first.ids.len(),
            match vs_ar {
                None => "identical (diagnostic)".to_string(),
                Some(i) => format!("first difference at index {i}: ar {ar_id:?} vs {arm_id:?} (diagnostic)"),
            },
            stats.windows,
            stats.batched,
            stats.interleaved,
            stats.rejected,
            ratio(stats.committed, stats.windows as u64),
            stats.accepted,
            stats.drafts,
            first.state["resets"]["at_load"],
            first.state["resets"]["after_prefill"],
            first.state["resets"]["final"],
            if verdict.ids_equal { "identical" } else { "DIFFER" },
            if verdict.digest_equal { "identical" } else { "DIFFER" },
            format!(
                "{}{}",
                if verdict.history_equal { "" } else { " (window histories differ: context only)" },
                verdict
                    .diagnostic_note(arm.route)
                    .map_or_else(String::new, |note| format!("; {note}")),
            ),
        ));
        arms_json.insert(
            arm.name.to_string(),
            json!({
                "two_fresh_runs": verdict.to_json(arm.route),
                "vs_ar_diagnostic": {
                    "first_id_diff": vs_ar,
                    "ar_id": ar_id,
                    "arm_id": arm_id,
                    "final_digest_differing_categories": category_counts(&ar_digest_diff),
                },
                "windows": stats.to_json(),
                "request_stats": first.request_stats,
                "resets": first.state["resets"],
                "final_residual": first.state["final_residual"],
                "logits": first.state["logits"],
                "row_capture": capture,
                "text_path": format!("{}.r1.txt", arm.name),
                "text": first.text,
            }),
        );
    }
    text.push_str("\ndecoded text (run 1 of each arm):\n");
    for (arm, pair) in &runs {
        text.push_str(&format!("--- {} ---\n{}\n", arm.name, pair[0].text));
    }
    let summary = json!({
        "case": case.name(),
        "gdn_state": {"request": GDN_MODE, "resolved": GDN_MODE},
        "header": header,
        "arms": arms_json,
        "hard_failures": failures,
        "incomplete": incomplete,
        "verdict": if !failures.is_empty() { "FAIL" } else if !incomplete.is_empty() { "INCOMPLETE" } else { "PASS" },
    });
    let verdict = summary["verdict"].as_str().unwrap_or("FAIL");
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        dir.join("summary.txt"),
        format!(
            "{header}verdict={verdict}\n{text}\nhard failures:\n{}\nincomplete (unobservable mandatory checks):\n{}\n",
            failures.join("\n"),
            incomplete.join("\n")
        ),
    )
    .map_err(|e| e.to_string())?;
    Ok(CaseReport {
        summary,
        text,
        failures,
        incomplete,
    })
}

#[test]
#[ignore = "requires a real HIP GPU, HIPFIRE_GDN_TEST_STATE_QUANT=q8_ef and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn q8_ef_product_routes_on_flash_next() {
    require_fabric_hold();
    let model = model_from(MODEL_ENV);
    let base = out_dir().join("q8ef-routes");
    let (mut failures, mut incomplete) = (Vec::new(), Vec::new());
    for case in [Case::P1, Case::Seasons] {
        let report = run_product_case(case, &model, &base)
            .unwrap_or_else(|e| panic!("{} case: {e}", case.name()));
        eprint!("{}", report.text);
        eprintln!("{} verdict {}", case.name(), report.summary["verdict"]);
        failures.extend(report.failures);
        incomplete.extend(report.incomplete);
    }
    assert!(
        failures.is_empty(),
        "q8_ef product routes (summaries in {}):\n{}",
        base.display(),
        failures.join("\n")
    );
    assert!(
        incomplete.is_empty(),
        "q8_ef product routes INCOMPLETE: mandatory checks are unobservable (summaries in {}):\n{}",
        base.display(),
        incomplete.join("\n")
    );
}

/// One arm in a fresh process; spawned by `spawn_arm`.
#[test]
#[ignore = "child process of q8_ef_product_routes_on_flash_next"]
fn q8_ef_product_routes_arm() {
    let Ok(name) = std::env::var(ARM_ENV) else {
        return;
    };
    let arm = ARMS
        .iter()
        .copied()
        .find(|arm| arm.name == name)
        .unwrap_or_else(|| panic!("unknown {ARM_ENV}={name}"));
    let value = std::env::var(CASE_ENV).unwrap_or_else(|_| panic!("{CASE_ENV} is not set"));
    let (case, run) = parse_case_run(&value).unwrap_or_else(|e| panic!("{e}"));
    let dir = out_dir();
    if let Err(error) = run_arm(arm, case, run, &model_from(MODEL_ENV), &dir) {
        panic!("{} arm: {error}", arm.name);
    }
}

fn run_arm(arm: Arm, case: Case, run: u32, model: &Path, dir: &Path) -> Result<(), String> {
    require_explicit_q8ef()?;
    require_arm_env(arm)?;
    let Loaded {
        mut bundle,
        mut gpu,
        tokenizer,
        template,
        state_request,
        state_format,
        replay_backend,
        backend,
        vocab,
    } = load(model, MAX_SEQ, None)?;
    let stem = stem(arm, run);
    let frame = |user: &'static str, system: Option<&'static str>| JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system,
        user,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let tokens = match case {
        Case::P1 => {
            let digest = Sha256::digest(PROMPT);
            let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            if digest != PROMPT_SHA256 {
                return Err(format!(
                    "prompt fixture sha256 {digest}, expected {PROMPT_SHA256}"
                ));
            }
            let prompt: &'static str = std::str::from_utf8(PROMPT).map_err(|e| e.to_string())?;
            let tokens = tokenizer.encode(&frame(prompt, Some("")).render()?);
            if tokens.len() != PROMPT_TOKENS {
                return Err(format!(
                    "rendered prompt is {} tokens, expected {PROMPT_TOKENS}",
                    tokens.len()
                ));
            }
            tokens
        }
        Case::Seasons => {
            // `hipfire serve` renders an OpenAI messages request through
            // `render_messages` (no system block when none is sent).
            let user = Message {
                role: Role::User,
                content: SEASONS.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            };
            let rendered = frame(SEASONS, None).render_messages(&[user], None, None)?;
            let tokens = tokenizer.encode(&rendered);
            if tokens != SEASONS_IDS {
                return Err(format!(
                    "rendered seasons prompt {tokens:?} is not the serve prompt {SEASONS_IDS:?}"
                ));
            }
            tokens
        }
    };
    let capture = native_mtp_row_capture(&gpu, &bundle.config);
    let head = hipfire_config::developer_var("HIPFIRE_MTP_DRAFT_HEAD")
        .unwrap_or_else(|_| "mq2r".to_string());
    let head_copy = DraftHeadPolicy::parse(&head).copy;
    if (arm.draft_head == Some("full")) == head_copy.is_some() {
        return Err(format!(
            "draft head policy {head:?} resolved to copy {head_copy:?}"
        ));
    }
    println!(
        "ARM {stem} arch={} prompt_tokens={} state_quant={state_request} gdn_format={} state={state_format:?} replay_backend={replay_backend} backend={} row_capture={capture} draft_head={head}",
        gpu.arch,
        tokens.len(),
        state_format.gdn.name(),
        backend.name()
    );
    let mut scratch: Vec<u8> = Vec::new();
    // Initial reset: counted once and the residual starts exactly zero.
    let resets_at_load = bundle.state.gdn_residual_resets();
    let initial = scan_residuals(&bundle, &mut gpu, &mut scratch)?;
    let generated = case.generated();
    let mut ids = Vec::with_capacity(generated + MTP_K);
    let mut windows: Vec<Value> = Vec::new();
    let resets_after_prefill;
    let mut request_stats = String::new();
    let logits_report;
    let final_digest;
    let final_scan;
    let final_resets;
    if arm.route == Route::Ar {
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        let mut host = Vec::new();
        bundle
            .forward_chunk_final(&mut gpu, &tokens, &logits, None)
            .map_err(|e| e.to_string())?;
        resets_after_prefill = bundle.state.gdn_residual_resets();
        check_logits_finite(&gpu, &logits, &mut host, "AR prefill")?;
        while ids.len() < generated {
            let step = ids.len();
            ids.push(
                bundle
                    .forward_token_or_argmax(&mut gpu, None, &logits)
                    .map_err(|e| e.to_string())?,
            );
            check_logits_finite(&gpu, &logits, &mut host, &format!("AR step {step}"))?;
        }
        logits_report =
            json!({"observed": true, "checked_forwards": 1 + generated, "nonfinite": 0});
        final_digest = live_digest(&bundle, &mut gpu)?;
        final_scan = scan_residuals(&bundle, &mut gpu, &mut scratch)?;
        final_resets = bundle.state.gdn_residual_resets();
        gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        if arm.route == Route::Batched && !capture {
            return Err("this GPU's GDN route does not capture verify rows".into());
        }
        bundle
            .attach_mtp(&mut gpu, MAX_SEQ)
            .map_err(|e| e.to_string())?;
        let mut drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None);
        arm_window_taps(&mut drafter);
        let mut seed = drafter
            .mtp_prefill(&mut gpu, &mut bundle, &tokens, &tokens, 0, false, &|| false)?
            .into_first_token()?;
        resets_after_prefill = bundle.state.gdn_residual_resets();
        ids.push(seed);
        let eos = bundle.config.eos_token_id;
        while ids.len() < generated {
            let position = bundle.state.position;
            let resets_before = bundle.state.gdn_residual_resets();
            let started = Counters::now();
            let window = drafter.mtp_step(
                &mut gpu,
                &mut bundle,
                position,
                seed,
                &ids,
                MTP_K,
                eos,
                None,
            )?;
            let counters = Counters::now().since(started);
            let resets_after = bundle.state.gdn_residual_resets();
            let after = drafter.request_stats();
            let tap = take_window_tap(&mut drafter)?;
            if let Some((before, after)) = tap.resets_seen {
                if after.saturating_sub(before) != resets_after.saturating_sub(resets_before) {
                    return Err(format!(
                        "window at position {position}: the lab tap saw residual resets {before}->{after}, the state counter {resets_before}->{resets_after}"
                    ));
                }
            }
            let mut row = WindowRow {
                index: windows.len() as u64,
                position_before: position as u64,
                position_after: bundle.state.position as u64,
                k_budget: MTP_K as u64,
                drafts: window.drafts_generated as u64,
                accepted: window.accepted as u64,
                committed: window.committed.len() as u64,
                mtp_windows_after: after.mtp_windows as u64,
                ar_windows_after: after.ar_windows as u64,
                ngram_windows_after: after.ngram_mod_windows as u64,
                retired_after: after.mtp_retired,
                resets_before,
                resets_after,
                residual_nonzero: None,
                residual_nonfinite: None,
                logits_rows: tap.logits_rows,
                logits_nonfinite: tap.logits_nonfinite,
                ring_half: tap.ring_half,
                residual_events: tap.residual_events,
            };
            if let Some(nonfinite) = row.logits_nonfinite {
                if nonfinite != 0 {
                    return Err(format!(
                        "window at position {position}: {nonfinite} non-finite verify-row logits"
                    ));
                }
            }
            if row.rolled_back() {
                let scan = scan_residuals(&bundle, &mut gpu, &mut scratch)?;
                row.residual_nonzero = Some(scan.nonzero);
                row.residual_nonfinite = Some(scan.nonfinite);
                if scan.nonfinite != 0 {
                    return Err(format!(
                        "window {}: {} non-finite residual elements after a rolled-back window",
                        row.index, scan.nonfinite
                    ));
                }
            }
            windows.push(row.to_json(counters.to_json()));
            seed = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
        ids.truncate(generated);
        let stats = drafter.request_stats();
        println!("STATS {stats:?}");
        request_stats = format!("{stats:?}");
        logits_report = json!({
            "lab_feature": cfg!(feature = "lab"),
            "reason": MTP_LOGITS_UNOBSERVABLE,
        });
        final_digest = live_digest(&bundle, &mut gpu)?;
        final_scan = scan_residuals(&bundle, &mut gpu, &mut scratch)?;
        final_resets = bundle.state.gdn_residual_resets();
        fs::write(
            dir.join(format!("{stem}.windows.json")),
            serde_json::to_vec_pretty(&json!({
                "windows": windows,
                "request_stats": request_stats,
            }))
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Box::new(drafter).mtp_free(&mut gpu);
    }
    let layers = bundle.state.gdn.len();
    let evidence = resolved_state_evidence(&bundle, state_request, state_format)?;
    fs::write(
        dir.join(format!("{stem}.ids.json")),
        serde_json::to_vec(&ids).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(dir.join(format!("{stem}.txt")), tokenizer.decode(&ids))
        .map_err(|e| e.to_string())?;
    fs::write(
        dir.join(format!("{stem}.state.json")),
        serde_json::to_vec_pretty(&json!({
            "arm": arm.name,
            "case": case.name(),
            "run": run,
            "gdn_state": {"request": state_request, "resolved": state_format.gdn.name()},
            "evidence": evidence,
            "layers": layers,
            "row_capture": capture,
            "resets": {
                "at_load": resets_at_load,
                "after_prefill": resets_after_prefill,
                "final": final_resets,
            },
            "initial_residual": initial.to_json(),
            "final_residual": final_scan.to_json(),
            "logits": logits_report,
            "request_stats": request_stats,
            "digest": digest_to_json(&final_digest),
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let identity = hw_identity::config_identity(
        &hw_identity::env_snapshot(&[], &CONTROL_ENV),
        &hw_identity::model_identity(model, dir)?,
    );
    fs::write(
        dir.join(format!("{stem}.identity.json")),
        serde_json::to_vec_pretty(&json!({
            "arm": arm.name,
            "case": case.name(),
            "run": run,
            "config": identity,
            "gdn_state": {
                "request": state_request,
                "resolved": state_format.gdn.name(),
            },
            "graph_policy": {
                "replay_backend": replay_backend,
                "replay_state": format!("{:?}", gpu.replay.state()),
                "HIPFIRE_GRAPH": std::env::var("HIPFIRE_GRAPH").ok(),
                "note": GRAPH_POLICY_NOTE,
            },
            "receipt": hw_identity::receipt(&gpu, model),
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    bundle.free_gpu(&mut gpu).map_err(|e| e.to_string())
}

// --------------------------------------------------------- restore routes

fn prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// A prefill (cold or restored) plus its greedy continuation.
struct Leg {
    label: String,
    reused: usize,
    ids: Vec<u32>,
    pending: u32,
    prefill_digest: Digest32,
    final_digest: Digest32,
    reset_events: u64,
}

struct Restore<'a> {
    bundle: Qwen4Bundle,
    gpu: Gpu,
    logits: GpuTensor,
    host: Vec<f32>,
    scratch: Vec<u8>,
    tokenizer: &'a Tokenizer,
    failures: Vec<String>,
    notes: Vec<Value>,
}

impl Restore<'_> {
    fn fail(&mut self, message: String) {
        eprintln!("FAIL {message}");
        self.failures.push(message);
    }

    /// Hard: the residual is exactly zero and finite right now.
    fn require_zero(&mut self, label: &str) -> Result<(), String> {
        let scan = scan_residuals(&self.bundle, &mut self.gpu, &mut self.scratch)?;
        if scan.nonzero != 0 || scan.nonfinite != 0 {
            self.fail(format!(
                "{label}: residual is not zero immediately after the reset: {}",
                scan.to_json()
            ));
        }
        Ok(())
    }

    fn ar_leg(&mut self, label: &str, tokens: &[u32], reused: usize) -> Result<Leg, String> {
        let before = self.bundle.state.gdn_residual_resets();
        self.bundle
            .prefill_final(&mut self.gpu, tokens, reused, &self.logits)
            .map_err(|e| format!("{label}: prefill: {e}"))?;
        let reset_events = self.bundle.state.gdn_residual_resets() - before;
        check_logits_finite(
            &self.gpu,
            &self.logits,
            &mut self.host,
            &format!("{label} prefill"),
        )?;
        let prefill_digest = live_digest(&self.bundle, &mut self.gpu)?;
        self.bundle.session_commit();
        let (ids, pending) = ar_decode(
            &mut self.bundle,
            &mut self.gpu,
            &self.logits,
            &mut self.host,
            None,
            DECODE,
            label,
        )?;
        let final_digest = live_digest(&self.bundle, &mut self.gpu)?;
        if reset_events == 0 {
            self.fail(format!(
                "{label}: the {} prefill did not reset the residual",
                if reused == 0 { "cold" } else { "restoring" }
            ));
        }
        Ok(Leg {
            label: label.to_string(),
            reused,
            ids,
            pending,
            prefill_digest,
            final_digest,
            reset_events,
        })
    }

    /// Hard: two legs of one history are identical.
    fn require_same(&mut self, a: &Leg, b: &Leg) {
        if let Some(index) = first_difference(&a.ids, &b.ids) {
            self.fail(format!(
                "{} vs {}: IDs differ at index {index}",
                a.label, b.label
            ));
        }
        for (what, x, y) in [
            ("prefill digest", &a.prefill_digest, &b.prefill_digest),
            ("final digest", &a.final_digest, &b.final_digest),
        ] {
            let keys = digest_diff(x, y);
            if !keys.is_empty() {
                self.fail(format!(
                    "{} vs {}: {what} differs in {} rows: {:?}",
                    a.label,
                    b.label,
                    keys.len(),
                    category_counts(&keys)
                ));
            }
        }
    }

    /// Diagnostic: restored vs cold continuation.
    fn diagnose(&mut self, name: &str, cold: &Leg, restored: &Leg) {
        let first = first_difference(&cold.ids, &restored.ids);
        let prefill = digest_diff(&cold.prefill_digest, &restored.prefill_digest);
        let last = digest_diff(&cold.final_digest, &restored.final_digest);
        self.notes.push(json!({
            "name": name,
            "diagnostic": "restored vs cold continuation (never asserted for q8_ef)",
            "reused": restored.reused,
            "first_id_diff": first,
            "cold_id": first.and_then(|i| cold.ids.get(i)),
            "restored_id": first.and_then(|i| restored.ids.get(i)),
            "prefill_digest_differing_categories": category_counts(&prefill),
            "final_digest_differing_categories": category_counts(&last),
            "cold_text": self.tokenizer.decode(&cold.ids),
            "restored_text": self.tokenizer.decode(&restored.ids),
            "reset_events": {"cold": cold.reset_events, "restored": restored.reset_events},
        }));
    }
}

/// The residual of `bundle` right after the production session restore entry
/// (`SessionState::restore_parts`, which zeroes it) applied to the live state's
/// own metadata, before any forward: `(scan, reset events)`.
fn session_restore_in_place(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    scratch: &mut Vec<u8>,
) -> Result<(ResidualScan, u64), String> {
    let position = bundle.state.position;
    let SnapshotParts { meta, layout: _ } =
        SessionState::snapshot_parts(&mut *bundle, SessionRoute::Ar, position)?;
    let before = bundle.state.gdn_residual_resets();
    SessionState::restore_parts(&mut *bundle, &mut *gpu, SessionRoute::Ar, &meta).map(|_| ())?;
    let scan = scan_residuals(bundle, gpu, scratch)?;
    SessionState::finish_restore(&mut *bundle, &mut *gpu, SessionRoute::Ar, &meta)?;
    Ok((scan, bundle.state.gdn_residual_resets() - before))
}

fn run_restore(model: &Path, dir: &Path) -> Result<(Value, String, Vec<String>), String> {
    let Loaded {
        bundle,
        mut gpu,
        tokenizer,
        state_request,
        state_format,
        replay_backend,
        vocab,
        ..
    } = load(model, RESTORE_MAX_SEQ, Some(u64::MAX >> 1))?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
    let mut r = Restore {
        bundle,
        gpu,
        logits,
        host: Vec::new(),
        scratch: Vec::new(),
        tokenizer: &tokenizer,
        failures: Vec::new(),
        notes: Vec::new(),
    };
    let chunk = r.bundle.spec_chunk_rows().ok_or("no chunk rows")?;
    let a = prompt(0xa, 3 * chunk + 300);
    let mut b = a[..chunk].to_vec();
    b.extend(prompt(0xb, chunk + 200));
    if a.len() + DECODE > RESTORE_MAX_SEQ {
        return Err(format!(
            "prompt A ({} tokens, chunk {chunk}) does not fit max_seq {RESTORE_MAX_SEQ}",
            a.len()
        ));
    }
    let mut zero_checks: Vec<String> = Vec::new();

    // Initial reset.
    let at_load = r.bundle.state.gdn_residual_resets();
    if at_load != 1 {
        r.fail(format!("reset counter at load is {at_load}, expected 1"));
    }
    r.require_zero("load")?;
    zero_checks.push("load: zero".into());

    // AR route: cold twice, then pool restores twice each.
    let cold_a1 = r.ar_leg("cold-A#1", &a, 0)?;
    let cold_a2 = r.ar_leg("cold-A#2", &a, 0)?;
    let cold_b = r.ar_leg("cold-B", &b, 0)?;
    r.require_same(&cold_a1, &cold_a2);
    let mut last = None;
    for (name, tokens, cold, links) in [("A", &a, &cold_a1, 3usize), ("B", &b, &cold_b, 2usize)] {
        let mut legs = Vec::new();
        for n in 1..=2 {
            let reused = r.bundle.session_plan(tokens, SessionRoute::Ar);
            if reused != links * chunk {
                return Err(format!(
                    "restore {name}#{n}: the session plan offers {reused} tokens, expected the {links}-chunk snapshot ({})",
                    links * chunk
                ));
            }
            legs.push(r.ar_leg(&format!("restored-{name}#{n}"), tokens, reused)?);
        }
        r.require_same(&legs[0], &legs[1]);
        r.diagnose(&format!("AR {name}"), cold, &legs[0]);
        last = legs.pop();
    }
    let last = last.ok_or("no restored leg")?;

    // Transaction restore and the production session restore entry, from the
    // live state the last restored leg left (non-zero residual required).
    let pre = scan_residuals(&r.bundle, &mut r.gpu, &mut r.scratch)?;
    if pre.nonzero_layers == 0 {
        r.fail("the residual is all zero after decoding: restore zeroing is not exercised".into());
    }
    let digest_p = live_digest(&r.bundle, &mut r.gpu)?;
    let q8_rows = ["position", "gdn.recurrent", "gdn.conv"];
    let pending = last.pending;
    let snap0 = r.bundle.snapshot(&mut r.gpu).map_err(|e| e.to_string())?;
    let (uninterrupted, _) = ar_decode(
        &mut r.bundle,
        &mut r.gpu,
        &r.logits,
        &mut r.host,
        Some(pending),
        TX_DECODE,
        "uninterrupted",
    )?;
    let digest_u = live_digest(&r.bundle, &mut r.gpu)?;
    let resets = r.bundle.state.gdn_residual_resets();
    r.bundle
        .restore(&mut r.gpu, snap0)
        .map_err(|e| e.to_string())?;
    r.require_zero("transaction restore")?;
    zero_checks.push("transaction restore: zero immediately".into());
    if r.bundle.state.gdn_residual_resets() != resets + 1 {
        r.fail("transaction restore did not bump the reset counter exactly once".into());
    }
    let restored = live_digest(&r.bundle, &mut r.gpu)?;
    let keys = digest_diff_in(&digest_p, &restored, &q8_rows);
    if !keys.is_empty() {
        r.fail(format!(
            "transaction restore changed the Q8 state rows: {keys:?}"
        ));
    }
    let snap1 = r.bundle.snapshot(&mut r.gpu).map_err(|e| e.to_string())?;
    let (tx1, _) = ar_decode(
        &mut r.bundle,
        &mut r.gpu,
        &r.logits,
        &mut r.host,
        Some(pending),
        TX_DECODE,
        "restored#1",
    )?;
    let digest_1 = live_digest(&r.bundle, &mut r.gpu)?;
    r.bundle
        .restore(&mut r.gpu, snap1)
        .map_err(|e| e.to_string())?;
    r.require_zero("transaction restore #2")?;
    zero_checks.push("transaction restore #2: zero immediately".into());
    let (tx2, _) = ar_decode(
        &mut r.bundle,
        &mut r.gpu,
        &r.logits,
        &mut r.host,
        Some(pending),
        TX_DECODE,
        "restored#2",
    )?;
    let digest_2 = live_digest(&r.bundle, &mut r.gpu)?;
    if let Some(index) = first_difference(&tx1, &tx2) {
        r.fail(format!(
            "two transaction-restored continuations differ at index {index}"
        ));
    }
    let keys = digest_diff(&digest_1, &digest_2);
    if !keys.is_empty() {
        r.fail(format!(
            "two transaction-restored continuations end in different live digests: {:?}",
            category_counts(&keys)
        ));
    }
    let tx_first = first_difference(&uninterrupted, &tx1);
    r.notes.push(json!({
        "name": "transaction restore",
        "diagnostic": "uninterrupted vs restored continuation (never asserted for q8_ef)",
        "first_id_diff": tx_first,
        "final_digest_differing_categories": category_counts(&digest_diff(&digest_u, &digest_1)),
        "uninterrupted_text": tokenizer.decode(&uninterrupted),
        "restored_text": tokenizer.decode(&tx1),
    }));

    // Production session restore entry on the live (non-zero residual) state.
    let pending_s = host_argmax(&r.host);
    let pre = scan_residuals(&r.bundle, &mut r.gpu, &mut r.scratch)?;
    if pre.nonzero_layers == 0 {
        r.fail("the residual is all zero before the session restore probe".into());
    }
    let digest_before = live_digest(&r.bundle, &mut r.gpu)?;
    let (scan, events) = {
        let Restore {
            bundle,
            gpu,
            scratch,
            ..
        } = &mut r;
        session_restore_in_place(bundle, gpu, scratch)?
    };
    zero_checks.push("session restore entry: zero immediately".into());
    if scan.nonzero != 0 || scan.nonfinite != 0 {
        r.fail(format!(
            "SessionState::restore_parts left a non-zero residual: {}",
            scan.to_json()
        ));
    }
    if events != 1 {
        r.fail(format!(
            "SessionState::restore_parts bumped the reset counter {events}x, expected 1"
        ));
    }
    let after = live_digest(&r.bundle, &mut r.gpu)?;
    let keys = digest_diff_in(&digest_before, &after, &q8_rows);
    if !keys.is_empty() {
        r.fail(format!(
            "the session restore entry changed the Q8 state rows: {keys:?}"
        ));
    }
    let snap2 = r.bundle.snapshot(&mut r.gpu).map_err(|e| e.to_string())?;
    let (s1, _) = ar_decode(
        &mut r.bundle,
        &mut r.gpu,
        &r.logits,
        &mut r.host,
        Some(pending_s),
        TX_DECODE,
        "session-restored#1",
    )?;
    let digest_s1 = live_digest(&r.bundle, &mut r.gpu)?;
    r.bundle
        .restore(&mut r.gpu, snap2)
        .map_err(|e| e.to_string())?;
    let (s2, _) = ar_decode(
        &mut r.bundle,
        &mut r.gpu,
        &r.logits,
        &mut r.host,
        Some(pending_s),
        TX_DECODE,
        "session-restored#2",
    )?;
    let digest_s2 = live_digest(&r.bundle, &mut r.gpu)?;
    if let Some(index) = first_difference(&s1, &s2) {
        r.fail(format!(
            "two session-restored continuations differ at index {index}"
        ));
    }
    let keys = digest_diff(&digest_s1, &digest_s2);
    if !keys.is_empty() {
        r.fail(format!(
            "two session-restored continuations end in different live digests: {:?}",
            category_counts(&keys)
        ));
    }

    // Reset.
    let before = r.bundle.state.gdn_residual_resets();
    r.bundle.reset(&mut r.gpu).map_err(|e| e.to_string())?;
    r.require_zero("reset")?;
    zero_checks.push("reset: zero immediately".into());
    if r.bundle.state.gdn_residual_resets() != before + 1 {
        r.fail("Qwen4Bundle::reset did not bump the reset counter exactly once".into());
    }

    // Native-MTP route: prefill seeds and digests (cold twice, restored twice).
    r.bundle
        .attach_mtp(&mut r.gpu, RESTORE_MAX_SEQ)
        .map_err(|e| e.to_string())?;
    let mut drafter = Qwen4MtpDrafter::new(MTP_K, RESTORE_MAX_SEQ, None);
    let mut mtp_notes = Vec::new();
    let mut mtp_leg = |r: &mut Restore<'_>,
                       label: &str,
                       tokens: &[u32],
                       reused: usize|
     -> Result<(u32, Digest32, u64), String> {
        let before = r.bundle.state.gdn_residual_resets();
        let seed = drafter
            .mtp_prefill(
                &mut r.gpu,
                &mut r.bundle,
                tokens,
                &tokens[reused..],
                reused,
                reused > 0,
                &|| false,
            )
            .and_then(|outcome| outcome.into_first_token())
            .map_err(|e| format!("{label}: {e}"))?;
        let events = r.bundle.state.gdn_residual_resets() - before;
        let digest = live_digest(&r.bundle, &mut r.gpu)?;
        r.bundle.session_commit();
        if events == 0 {
            r.fail(format!(
                "{label}: the MTP prefill did not reset the residual"
            ));
        }
        Ok((seed, digest, events))
    };
    let cold_a1 = mtp_leg(&mut r, "MTP cold-A#1", &a, 0)?;
    let cold_a2 = mtp_leg(&mut r, "MTP cold-A#2", &a, 0)?;
    let cold_b = mtp_leg(&mut r, "MTP cold-B", &b, 0)?;
    if cold_a1.0 != cold_a2.0 || !digest_diff(&cold_a1.1, &cold_a2.1).is_empty() {
        r.fail("MTP cold-A twice: seed or live digest differs".into());
    }
    for (name, tokens, cold, links) in [("A", &a, &cold_a1, 3usize), ("B", &b, &cold_b, 2usize)] {
        let mut legs = Vec::new();
        for n in 1..=2 {
            let reused = r.bundle.session_plan(tokens, SessionRoute::Mtp);
            if reused != links * chunk {
                return Err(format!(
                    "MTP restore {name}#{n}: the session plan offers {reused} tokens, expected {}",
                    links * chunk
                ));
            }
            legs.push(mtp_leg(
                &mut r,
                &format!("MTP restored-{name}#{n}"),
                tokens,
                reused,
            )?);
        }
        if legs[0].0 != legs[1].0 || !digest_diff(&legs[0].1, &legs[1].1).is_empty() {
            r.fail(format!(
                "MTP restored-{name} twice: seed or live digest differs"
            ));
        }
        mtp_notes.push(json!({
            "name": format!("MTP {name}"),
            "diagnostic": "restored vs cold prefill (never asserted for q8_ef)",
            "cold_seed": cold.0,
            "restored_seed": legs[0].0,
            "final_digest_differing_categories":
                category_counts(&digest_diff(&cold.1, &legs[0].1)),
            "reset_events": {"cold": cold.2, "restored": legs[0].2},
        }));
    }
    Box::new(drafter).mtp_free(&mut r.gpu);
    r.notes.extend(mtp_notes);

    let identity = hw_identity::config_identity(
        &hw_identity::env_snapshot(&[], &RESTORE_CONTROL_ENV),
        &hw_identity::model_identity(model, dir)?,
    );
    let summary = json!({
        "gdn_state": {"request": state_request, "resolved": state_format.gdn.name()},
        "replay_backend": replay_backend,
        "chunk_rows": chunk,
        "decode_tokens": DECODE,
        "zero_checks_passed_or_reported": zero_checks,
        "observations": r.notes,
        "hard_failures": r.failures,
        "config": identity,
        "not_exercised": ["disk-tier restore (same q8_ef-reset scope; covered by the envelope tests)"],
        "verdict": if r.failures.is_empty() { "PASS" } else { "FAIL" },
    });
    let failures = r.failures.clone();
    let text = format!(
        "gdn_state_quant={state_request} gdn_format={}\nreplay_backend={replay_backend}\nverdict={}\nzero checks: {}\nobservations:\n{}\nhard failures:\n{}\n",
        state_format.gdn.name(),
        summary["verdict"].as_str().unwrap_or("FAIL"),
        summary["zero_checks_passed_or_reported"],
        serde_json::to_string_pretty(&summary["observations"]).unwrap_or_default(),
        failures.join("\n"),
    );
    r.gpu.free_tensor(r.logits).map_err(|e| e.to_string())?;
    r.bundle.free_gpu(&mut r.gpu).map_err(|e| e.to_string())?;
    Ok((summary, text, failures))
}

#[test]
#[ignore = "requires a real HIP GPU, HIPFIRE_GDN_TEST_STATE_QUANT=q8_ef and HIPFIRE_SESSION_CACHE_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn q8_ef_restore_routes_on_flash_next() {
    require_fabric_hold();
    let model = model_from(SESSION_MODEL_ENV);
    let dir = out_dir().join("q8ef-restore");
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let (summary, text, failures) =
        run_restore(&model, &dir).unwrap_or_else(|e| panic!("q8_ef restore routes: {e}"));
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary).expect("summary json"),
    )
    .expect("write summary.json");
    fs::write(dir.join("summary.txt"), &text).expect("write summary.txt");
    eprint!("{text}");
    assert!(
        failures.is_empty(),
        "q8_ef restore routes (summary in {}):\n{}",
        dir.display(),
        failures.join("\n")
    );
}

// ------------------------------------------------------------ CPU tests

#[test]
fn q8ef_explicit_mode_refuses_unset_and_other_modes() {
    assert!(explicit_q8ef(None).unwrap_err().contains("unset"));
    assert!(explicit_q8ef(Some("q8")).unwrap_err().contains("not q8_ef"));
    assert!(explicit_q8ef(Some("fp32"))
        .unwrap_err()
        .contains("not q8_ef"));
    assert!(explicit_q8ef(Some("bogus")).is_err());
    assert_eq!(explicit_q8ef(Some("q8_ef")), Ok(()));
}

#[test]
fn q8ef_scan_f16_bits_counts_nonzero_and_nonfinite() {
    let bits: [u16; 6] = [0x0000, 0x8000, 0x3c00, 0x7c00, 0xfc00, 0x7e01];
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    // -0.0, 1.0, +inf, -inf and a NaN are non-zero; inf and NaN are non-finite.
    assert_eq!(scan_f16_bits(&bytes), (5, 3));
    assert_eq!(scan_f16_bits(&[0u8; 64]), (0, 0));
}

#[test]
fn q8ef_first_difference_reports_prefix_and_mismatch() {
    assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 3]), None);
    assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
    assert_eq!(first_difference(&[1, 2], &[1, 2, 3]), Some(2));
    assert_eq!(first_difference::<u32>(&[], &[]), None);
}

#[test]
fn q8ef_parse_case_run_requires_case_and_positive_run() {
    assert_eq!(parse_case_run("p1:2"), Ok((Case::P1, 2)));
    assert_eq!(parse_case_run("seasons:1"), Ok((Case::Seasons, 1)));
    assert!(parse_case_run("p1").is_err());
    assert!(parse_case_run("copy:1").is_err());
    assert!(parse_case_run("p1:0").is_err());
    assert!(parse_case_run("p1:x").is_err());
}

#[test]
fn q8ef_digest_helpers_classify_diff_and_roundtrip() {
    let row = |k: &str, v: &str| (k.to_string(), v.to_string());
    let a = vec![
        row("position", "10"),
        row("gdn[0].recurrent", "aa"),
        row("gdn[0].ef_residual", "bb"),
        row("qsa[0].full_keys", "cc"),
    ];
    let mut b = a.clone();
    b[2].1 = "zz".into();
    b.push(row("row_capture[0].ring", "dd"));
    let keys = digest_diff(&a, &b);
    assert_eq!(keys, vec!["gdn[0].ef_residual", "row_capture[0].ring"]);
    let counts = category_counts(&keys);
    assert_eq!(counts["gdn.ef_residual"], 1);
    assert_eq!(counts["row_capture"], 1);
    assert!(digest_diff_in(&a, &b, &["position", "gdn.recurrent", "gdn.conv"]).is_empty());
    assert_eq!(digest_diff_in(&a, &b, &["gdn.ef_residual"]).len(), 1);
    assert_eq!(residual_rows_in(&a), 1);
    assert_eq!(digest_from_json(&digest_to_json(&a)).unwrap(), a);
    assert!(digest_from_json(&json!([["only-key"]])).is_err());
    assert_eq!(digest_category("gdn[3].conv"), "gdn.conv");
    assert_eq!(digest_category("ple_conv"), "ple");
    assert_eq!(digest_category("hyper_feedback"), "other");
}

#[test]
fn q8ef_trace_lines_parse_route_position_and_resets() {
    let stderr = "noise\nQWEN4_MTP_TRACE {\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"batched\",\"accept_route\":\"greedy\",\"position\":1047,\"k\":3,\"accepted\":1,\"rows\":[{\"draft\":5,\"pick\":5,\"match\":true,\"margin\":0.5}],\"committed\":[5, 7],\"baseline\":true,\"gdn_residual_resets\":2}\nQWEN4_MTP_TRACE {\"event\":\"window\",\"source\":\"mtp\",\"verify_route\":\"interleaved\",\"accept_route\":\"greedy\",\"position\":1049,\"k\":3,\"accepted\":3,\"rows\":[],\"committed\":[1]}\n";
    let windows = parse_trace(stderr).unwrap();
    assert_eq!(
        windows,
        vec![
            TraceWindow {
                position: 1047,
                k: 3,
                accepted: 1,
                route: "batched".into(),
                resets: Some(2)
            },
            TraceWindow {
                position: 1049,
                k: 3,
                accepted: 3,
                route: "interleaved".into(),
                resets: None
            },
        ]
    );
    assert!(parse_trace("QWEN4_MTP_TRACE {\"event\":\"window\",\"k\":3}").is_err());
    assert!(parse_trace("nothing here").unwrap().is_empty());
}

#[cfg(test)]
fn window(index: u64, position: u64, drafts: u64, accepted: u64, resets: (u64, u64)) -> WindowRow {
    WindowRow {
        index,
        position_before: position,
        position_after: position + accepted + 1,
        k_budget: 3,
        drafts,
        accepted,
        committed: accepted + 1,
        mtp_windows_after: index + 1,
        ar_windows_after: 0,
        ngram_windows_after: 0,
        retired_after: false,
        resets_before: resets.0,
        resets_after: resets.1,
        residual_nonzero: None,
        residual_nonfinite: None,
        logits_rows: None,
        logits_nonfinite: None,
        ring_half: None,
        residual_events: None,
    }
}

#[cfg(test)]
fn traced(position: u64, accepted: u64, route: &str) -> TraceWindow {
    TraceWindow {
        position,
        k: 3,
        accepted,
        route: route.into(),
        resets: Some(1),
    }
}

#[test]
fn q8ef_window_rules_accept_a_well_formed_batched_history() {
    let mut rejected = window(0, 100, 3, 1, (1, 2));
    rejected.residual_nonzero = Some(0);
    rejected.residual_nonfinite = Some(0);
    let rows = vec![rejected, window(1, 102, 3, 3, (2, 2))];
    let trace = vec![traced(100, 1, "batched"), traced(102, 3, "batched")];
    let (stats, errors) = check_windows(Route::Batched, &rows, &trace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(stats.rejected_batched, 1);
    assert_eq!(stats.truncated_batched, 0);
    assert_eq!(stats.full_commit_batched, 1);
    assert_eq!(stats.zero_checked, 1);
    assert_eq!(stats.batched_reset_events, 1);
    assert_eq!(stats.committed, 6);
    assert_eq!(ratio(stats.committed, stats.windows as u64), Some(3.0));
}

#[test]
fn q8ef_window_rules_reject_missing_reset_nonzero_residual_and_route_errors() {
    let trace = vec![traced(100, 1, "batched")];
    // Rejected batched window with no reset event and an unscanned residual.
    let rows = vec![window(0, 100, 3, 1, (1, 1))];
    let (_, errors) = check_windows(Route::Batched, &rows, &trace, true);
    assert!(
        errors.iter().any(|e| e.contains("did not reset")),
        "{errors:?}"
    );
    assert!(
        errors.iter().any(|e| e.contains("not scanned")),
        "{errors:?}"
    );
    // Non-zero residual after the rejection.
    let mut row = window(0, 100, 3, 1, (1, 2));
    row.residual_nonzero = Some(7);
    let (_, errors) = check_windows(Route::Batched, &[row.clone()], &trace, true);
    assert!(errors.iter().any(|e| e.contains("not zero")), "{errors:?}");
    // The zero check is skipped (not applicable) without row capture.
    let (_, errors) = check_windows(Route::Batched, &[row], &trace, false);
    assert!(errors.is_empty(), "{errors:?}");
    // A full-commit batched window (committed = drafts + 1) never resets.
    let rows = vec![window(0, 100, 3, 3, (1, 2))];
    let (_, errors) = check_windows(Route::Adaptive, &rows, &[traced(100, 3, "batched")], true);
    assert!(
        errors.iter().any(|e| e.contains("full-commit")),
        "{errors:?}"
    );
    // Interleaved windows take no assumed reset (see the lab-tap test).
    let rows = vec![window(0, 100, 3, 1, (1, 2))];
    let (_, errors) = check_windows(
        Route::Incremental,
        &rows,
        &[traced(100, 1, "interleaved")],
        true,
    );
    assert!(errors.is_empty(), "{errors:?}");
    // A window with drafts and no trace event, and a trace/window mismatch.
    let (_, errors) = check_windows(Route::Adaptive, &[window(0, 100, 3, 3, (1, 1))], &[], true);
    assert!(
        errors.iter().any(|e| e.contains("no trace event")),
        "{errors:?}"
    );
    let (_, errors) = check_windows(
        Route::Adaptive,
        &[window(0, 100, 3, 3, (1, 1))],
        &[traced(100, 2, "batched")],
        true,
    );
    assert!(
        errors.iter().any(|e| e.contains("trace accepted")),
        "{errors:?}"
    );
    // Target-only windows never reset.
    let (stats, errors) =
        check_windows(Route::Adaptive, &[window(0, 100, 0, 0, (1, 2))], &[], true);
    assert_eq!(stats.target_only, 1);
    assert!(
        errors.iter().any(|e| e.contains("target-only")),
        "{errors:?}"
    );
}

#[cfg(test)]
fn truncated_window(resets: (u64, u64)) -> WindowRow {
    // drafts 3, accepted 3, committed 3: the commit stopped at a turn
    // boundary, so the bonus row was rolled back.
    let mut row = window(0, 100, 3, 3, resets);
    row.committed = 3;
    row
}

#[test]
fn q8ef_truncated_full_accept_batched_window_is_a_rolled_back_window() {
    let trace = vec![traced(100, 3, "batched")];
    assert!(truncated_window((1, 1)).rolled_back());
    assert!(!truncated_window((1, 1)).rejected());
    assert!(!window(0, 100, 3, 3, (1, 1)).rolled_back());
    assert!(window(0, 100, 3, 1, (1, 1)).rolled_back());
    // (a) A reset and a zero scan: no error; counted as truncated, not rejected.
    let mut row = truncated_window((1, 2));
    row.residual_nonzero = Some(0);
    row.residual_nonfinite = Some(0);
    let (stats, errors) = check_windows(Route::Adaptive, &[row], &trace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(stats.rejected, 0);
    assert_eq!(stats.rejected_batched, 0);
    assert_eq!(stats.truncated_batched, 1);
    assert_eq!(stats.full_commit_batched, 0);
    assert_eq!(stats.zero_checked, 1);
    assert_eq!(stats.batched_reset_events, 1);
    // (b) Without a reset it is a failure.
    let mut row = truncated_window((1, 1));
    row.residual_nonzero = Some(0);
    let (_, errors) = check_windows(Route::Adaptive, &[row], &trace, true);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("rolled-back batched window") && e.contains("did not reset")),
        "{errors:?}"
    );
    // A non-zero or unscanned residual after it is a failure too.
    let mut row = truncated_window((1, 2));
    row.residual_nonzero = Some(5);
    let (_, errors) = check_windows(Route::Adaptive, &[row], &trace, true);
    assert!(errors.iter().any(|e| e.contains("not zero")), "{errors:?}");
    let (_, errors) = check_windows(Route::Adaptive, &[truncated_window((1, 2))], &trace, true);
    assert!(
        errors.iter().any(|e| e.contains("not scanned")),
        "{errors:?}"
    );
    // (c) A full-commit window (committed 4) with a reset is a failure.
    let (_, errors) = check_windows(
        Route::Adaptive,
        &[window(0, 100, 3, 3, (1, 2))],
        &trace,
        true,
    );
    assert!(
        errors.iter().any(|e| e.contains("full-commit")),
        "{errors:?}"
    );
    // ... and passes with no reset.
    let (stats, errors) = check_windows(
        Route::Adaptive,
        &[window(0, 100, 3, 3, (1, 1))],
        &trace,
        true,
    );
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(stats.full_commit_batched, 1);
}

#[test]
fn q8ef_window_rules_require_route_coverage() {
    let rows = vec![window(0, 100, 3, 3, (1, 1))];
    let trace = vec![traced(100, 3, "batched")];
    // No rejected batched window: no reset coverage on a batched arm.
    let (_, errors) = check_windows(Route::Batched, &rows, &trace, true);
    assert!(
        errors.iter().any(|e| e.contains("no reset coverage")),
        "{errors:?}"
    );
    let (_, errors) = check_windows(Route::Incremental, &rows, &trace, true);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("interleaved windows only")),
        "{errors:?}"
    );
    let (_, errors) = check_windows(Route::Adaptive, &rows, &trace, true);
    assert!(errors.is_empty(), "{errors:?}");
    let (_, errors) = check_windows(Route::Ar, &rows, &trace, true);
    assert!(errors.iter().any(|e| e.contains("ar arm")), "{errors:?}");
    let mut no_resets = trace.clone();
    no_resets[0].resets = None;
    let (_, errors) = check_windows(Route::Adaptive, &rows, &no_resets, true);
    assert!(
        errors.iter().any(|e| e.contains("gdn_residual_resets")),
        "{errors:?}"
    );
}

#[test]
fn q8ef_window_rules_use_the_lab_tap_observations() {
    let trace = vec![traced(100, 1, "batched")];
    let mut row = window(0, 100, 3, 1, (1, 2));
    row.residual_nonzero = Some(0);
    row.logits_rows = Some(4);
    row.logits_nonfinite = Some(0);
    row.ring_half = Some(1);
    row.residual_events = Some(vec!["PartialRollback".into()]);
    let (stats, errors) = check_windows(Route::Batched, &[row.clone()], &trace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!((stats.logits_observed, stats.logits_unobserved), (1, 0));
    assert_eq!(stats.ring_halves, [0, 1]);
    // A non-finite verify logit is a hard failure.
    let mut bad = row.clone();
    bad.logits_nonfinite = Some(3);
    let (_, errors) = check_windows(Route::Batched, &[bad], &trace, true);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("non-finite verify-row logits")),
        "{errors:?}"
    );
    // The state tap's residual events must account for exactly the counter's
    // rise, in either direction.
    let mut silent = row.clone();
    silent.residual_events = Some(Vec::new());
    let (_, errors) = check_windows(Route::Batched, &[silent], &trace, true);
    assert!(
        errors.iter().any(|e| e.contains("state tap saw 0")),
        "{errors:?}"
    );
    let mut extra = row.clone();
    extra.residual_events = Some(vec!["PartialRollback".into(), "Restore".into()]);
    let (_, errors) = check_windows(Route::Batched, &[extra], &trace, true);
    assert!(
        errors.iter().any(|e| e.contains("state tap saw 2")),
        "{errors:?}"
    );
    // An interleaved window is judged by the hook data, never by an assumed
    // reset: a rejected one with no reset and no events passes, one whose
    // counter rose passes exactly when the tap saw those events, and without
    // a tap the rise is only recorded.
    let itrace = vec![traced(100, 1, "interleaved")];
    let mut quiet = window(0, 100, 3, 1, (1, 1));
    quiet.residual_events = Some(Vec::new());
    let (stats, errors) = check_windows(Route::Incremental, &[quiet], &itrace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(
        (stats.rejected_interleaved, stats.interleaved_reset_events),
        (1, 0)
    );
    let mut observed = window(0, 100, 3, 1, (1, 2));
    observed.residual_events = Some(vec!["Restore".into()]);
    let (stats, errors) = check_windows(Route::Incremental, &[observed.clone()], &itrace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(stats.interleaved_reset_events, 1);
    observed.residual_events = Some(Vec::new());
    let (_, errors) = check_windows(Route::Incremental, &[observed], &itrace, true);
    assert!(
        errors.iter().any(|e| e.contains("state tap saw 0")),
        "{errors:?}"
    );
    let untapped_rise = window(0, 100, 3, 1, (1, 2));
    let (stats, errors) = check_windows(Route::Incremental, &[untapped_rise], &itrace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(stats.interleaved_reset_events, 1);
    // Without the tap nothing is observed: the run is INCOMPLETE, not clean.
    let mut untapped = row;
    untapped.logits_rows = None;
    untapped.logits_nonfinite = None;
    let (stats, errors) = check_windows(Route::Batched, &[untapped], &trace, true);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!((stats.logits_observed, stats.logits_unobserved), (0, 1));
}

#[test]
fn q8ef_window_history_and_row_json_roundtrip() {
    let row = window(2, 200, 3, 2, (4, 5));
    let value = row.to_json(json!([1, 2]));
    assert_eq!(WindowRow::from_json(&value).unwrap(), row);
    let mut scanned = row.clone();
    scanned.residual_nonzero = Some(0);
    scanned.residual_nonfinite = Some(0);
    scanned.logits_rows = Some(4);
    scanned.logits_nonfinite = Some(0);
    scanned.ring_half = Some(1);
    scanned.residual_events = Some(vec!["Restore".to_string(), "Reset".to_string()]);
    assert_eq!(
        WindowRow::from_json(&scanned.to_json(json!([]))).unwrap(),
        scanned
    );
    assert!(WindowRow::from_json(&json!({"index": 1})).is_err());
    let h = history(&[row], &[traced(200, 2, "batched")]);
    assert_eq!(h, vec!["200:3:2:3:4:5:batched"]);
    assert_eq!(
        history(&[window(0, 9, 0, 0, (1, 1))], &[]),
        vec!["9:0:0:1:1:1:-"]
    );
}

#[test]
fn q8ef_pair_verdicts_assert_identity_with_history_context() {
    let make = |ids: Vec<u32>, digest_value: &str| RunData {
        ids,
        text: String::new(),
        state: Value::Null,
        digest: vec![("gdn[0].ef_residual".to_string(), digest_value.to_string())],
        windows: vec![window(0, 100, 3, 1, (1, 2))],
        request_stats: String::new(),
        trace: vec![traced(100, 1, "batched")],
        header: String::new(),
    };
    let (a, b) = (make(vec![1, 2], "x"), make(vec![1, 2], "x"));
    assert!(compare_pair(&a, &b)
        .failures("t", Route::Batched)
        .is_empty());
    assert!(compare_pair(&a, &b)
        .failures("t", Route::Adaptive)
        .is_empty());
    // A residual difference alone is a hard failure.
    let c = make(vec![1, 2], "y");
    let failures = compare_pair(&a, &c).failures("t", Route::Batched);
    assert!(
        failures.iter().any(|f| f.contains("final live digests")),
        "{failures:?}"
    );
    assert!(
        failures
            .iter()
            .any(|f| f.contains("histories are identical")),
        "{failures:?}"
    );
    // An ID difference is hard.
    let d = make(vec![1, 3], "x");
    let failures = compare_pair(&a, &d).failures("t", Route::Batched);
    assert!(
        failures.iter().any(|f| f.contains("IDs differ")),
        "{failures:?}"
    );
    // A different accept history with identical route decisions (position,
    // drafts, verify route) is only the failure's context, on every route
    // (adaptive included): the decisions are the same.
    let mut e = make(vec![1, 3], "z");
    e.windows[0].accepted = 2;
    e.trace[0].accepted = 2;
    let verdict = compare_pair(&a, &e);
    assert!(!verdict.history_equal);
    assert_eq!(verdict.first_decision_diff, None);
    for route in [Route::Batched, Route::Adaptive] {
        let failures = verdict.failures("t", route);
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(
            failures
                .iter()
                .all(|f| f.contains("window histories differ first at window")),
            "{failures:?}"
        );
    }
    let json = verdict.to_json(Route::Adaptive);
    assert_eq!(json["window_history_identical"], json!(false));
    assert_eq!(json["decisions_identical"], json!(true));
    assert_eq!(json["diagnostic_only"], Value::Null);
    // Identical IDs and digest pass even when the histories differ: the
    // history is not an assertion of its own.
    let mut f = make(vec![1, 2], "x");
    f.windows[0].accepted = 2;
    f.trace[0].accepted = 2;
    let verdict = compare_pair(&a, &f);
    assert!(!verdict.history_equal);
    assert!(verdict.failures("t", Route::Batched).is_empty());
}

#[cfg(test)]
fn pair_run(ids: Vec<u32>, digest_value: &str) -> RunData {
    RunData {
        ids,
        text: String::new(),
        state: Value::Null,
        digest: vec![("gdn[0].ef_residual".to_string(), digest_value.to_string())],
        windows: vec![window(0, 100, 3, 1, (1, 2)), window(1, 102, 3, 3, (2, 2))],
        request_stats: String::new(),
        trace: vec![traced(100, 1, "batched"), traced(102, 3, "batched")],
        header: String::new(),
    }
}

/// Same as [`pair_run`] but window 1 was decided differently (2 drafts).
#[cfg(test)]
fn pair_run_other_decision(ids: Vec<u32>, digest_value: &str) -> RunData {
    let mut run = pair_run(ids, digest_value);
    run.windows[1].drafts = 2;
    run
}

#[test]
fn q8ef_adaptive_pair_with_different_decisions_is_diagnostic() {
    // (d) Different route decisions and different IDs/digest: no failure.
    let a = pair_run(vec![1, 2], "x");
    let b = pair_run_other_decision(vec![1, 3], "y");
    let verdict = compare_pair(&a, &b);
    assert_eq!(verdict.first_decision_diff, Some(1));
    assert!(!verdict.ids_equal && !verdict.digest_equal && !verdict.history_equal);
    assert!(verdict.failures("t", Route::Adaptive).is_empty());
    let json = verdict.to_json(Route::Adaptive);
    assert_eq!(json["decisions_identical"], json!(false));
    assert_eq!(json["first_decision_diff"], json!(1));
    assert_eq!(json["window_history_identical"], json!(false));
    assert_eq!(json["first_window_history_diff"], json!(1));
    assert_eq!(json["ids_identical"], json!(false));
    assert_eq!(
        json["diagnostic_only"],
        json!("adaptive: route decisions differ at window 1 (wall-clock chooser) -> two-run equality is diagnostic")
    );
    assert_eq!(
        verdict.diagnostic_note(Route::Adaptive).as_deref(),
        Some("adaptive: route decisions differ at window 1 (wall-clock chooser) -> two-run equality is diagnostic")
    );
    // A differing verify route is a differing decision too.
    let mut c = pair_run(vec![1, 3], "y");
    c.trace[0].route = "interleaved".into();
    let verdict = compare_pair(&a, &c);
    assert_eq!(verdict.first_decision_diff, Some(0));
    assert!(verdict.failures("t", Route::Adaptive).is_empty());
}

#[test]
fn q8ef_adaptive_pair_with_identical_decisions_must_be_identical() {
    // (e) Identical decisions, different IDs and digest: hard failures.
    let a = pair_run(vec![1, 2], "x");
    let b = pair_run(vec![1, 3], "y");
    let verdict = compare_pair(&a, &b);
    assert_eq!(verdict.first_decision_diff, None);
    assert_eq!(verdict.diagnostic_note(Route::Adaptive), None);
    let failures = verdict.failures("t", Route::Adaptive);
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(
        failures.iter().any(|f| f.contains("IDs differ")),
        "{failures:?}"
    );
    assert!(
        failures.iter().any(|f| f.contains("final live digests")),
        "{failures:?}"
    );
    let json = verdict.to_json(Route::Adaptive);
    assert_eq!(json["decisions_identical"], json!(true));
    assert_eq!(json["first_decision_diff"], Value::Null);
}

#[test]
fn q8ef_non_adaptive_pairs_stay_hard_with_different_histories() {
    // (f) A batched pair with different histories/decisions and IDs: failure.
    let a = pair_run(vec![1, 2], "x");
    let b = pair_run_other_decision(vec![1, 3], "y");
    let verdict = compare_pair(&a, &b);
    assert!(!verdict.history_equal);
    assert_eq!(verdict.first_decision_diff, Some(1));
    for route in [Route::Ar, Route::Batched, Route::Incremental] {
        let failures = verdict.failures("t", route);
        assert_eq!(failures.len(), 2, "{route:?}: {failures:?}");
        assert_eq!(verdict.diagnostic_note(route), None);
    }
    assert_eq!(
        verdict.to_json(Route::Batched)["diagnostic_only"],
        Value::Null
    );
}

#[test]
fn q8ef_arm_table_matches_the_identity_harness_routes() {
    let names: Vec<&str> = ARMS.iter().map(|arm| arm.name).collect();
    assert_eq!(
        names,
        ["ar", "batched", "incremental", "adaptive", "full-head"]
    );
    assert_eq!(Case::P1.generated(), 1200);
    assert_eq!(Case::Seasons.generated(), 256);
    assert_eq!(stem(ARMS[1], 2), "batched.r2");
    assert_eq!(host_argmax(&[0.5, 2.0, 2.0, 1.0]), 1);
}

#[test]
fn q8ef_fabric_hold_guard_refuses_loads_outside_a_hold() {
    let chain = [30, 20, 10];
    let start = |pid| (pid == 20).then_some(77);
    let refusal = |threads, marker: &str, holds: bool| {
        hold_refusal(threads, marker, &chain, start, |pids: &[u32]| {
            holds && pids.contains(&20)
        })
    };
    assert!(refusal(false, "20:77:B", true)
        .unwrap()
        .contains("--test-threads=1"));
    assert!(refusal(true, "", true)
        .unwrap()
        .contains("FABRIC_JOB unset"));
    assert!(refusal(true, "40:77:B", true)
        .unwrap()
        .contains("not an ancestor"));
    assert!(refusal(true, "20:78:B", true).unwrap().contains("stale"));
    assert!(refusal(true, "20:77:B", false)
        .unwrap()
        .contains("--hostmem-lock"));
    assert_eq!(refusal(true, "20:77:B", true), None);
}

/// The holder scan finds a real shared flock on a file named like the host
/// lock, held by a child `flock(1)` process, and nothing for this process.
#[test]
fn q8ef_hostmem_lock_holders_reads_proc_locks() {
    let dir = std::env::temp_dir().join(format!("q8ef-routes-hw-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lock = dir.join(HOSTMEM_LOCK);
    std::fs::write(&lock, b"").unwrap();
    // `flock` holds the lock while its `cat` child waits on our stdin pipe,
    // which closing below ends.
    let Ok(mut child) = std::process::Command::new("flock")
        .arg("-s")
        .arg(&lock)
        .arg("cat")
        .stdin(std::process::Stdio::piped())
        .spawn()
    else {
        eprintln!("skip: flock(1) unavailable");
        return;
    };
    let pid = child.id();
    let mut held = Vec::new();
    for _ in 0..100 {
        held = hostmem_lock_holders(std::iter::once(pid));
        if !held.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(hostmem_lock_holders(std::iter::once(std::process::id())).is_empty());
    drop(child.stdin.take());
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(held, vec![(pid, "READ".to_string())]);
}
