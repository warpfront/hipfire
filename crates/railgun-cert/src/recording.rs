//! Recording-predicate inventory (railgun design §2.3, §5 G0, milestone MR).
//!
//! Authoring a program by recording only sees the kernels chosen while the
//! runtime believes it is being recorded or captured. [`DECISIONS`] lists
//! every runtime condition that consults that state, and [`SITES`] binds each
//! `(file, function)` that consults it to one decision. [`scan_sources`] is
//! the source scan that keeps the two in sync: a new `is_recording()`-style
//! check without an inventory row fails `cargo test -p railgun-cert`.
//!
//! Verdicts are per decision, relative to the eager branch:
//! - `Invariant`: kernel sequence, grids, blocks, LDS and argument values are
//!   the same on both branches (launch mechanism, lifecycle, refusals).
//! - `ByteExact`: a different kernel set or launch shape, with a cited
//!   argument or oracle that every output byte is identical.
//! - `NotEquivalent`: a different kernel set or shape with no byte-exactness
//!   evidence (or measured drift).
//!
//! [`Decision::reaches`] names the default programs in which the branch the
//! railgun recorder takes (`is_recording() == true`, `capture_mode == false`)
//! differs from the eager branch or from the pre-railgun default process's
//! branch. A `NotEquivalent` decision that reaches a program refuses that
//! program as a railgun default over the stated domain
//! ([`refused_default_programs`]).
use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

/// Runtime state a predicate consults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    /// `ReplayController::is_recording()` (Redline/railgun recorder warmup).
    IsRecording,
    /// `GraphState::capture_mode` (HIP stream capture in flight).
    CaptureMode,
    /// `ReplayController::is_enabled()`: process-level backend request.
    ReplayEnabled,
    /// `ReplayController::state()` compared against a lifecycle state.
    ReplayState,
    /// `verify.capturing` / `replay.capturing` graph slots.
    GraphSlotCapturing,
    /// A body parameter the composer sets when the body is captured
    /// (`capture_safe`, `graph_safe`).
    DeclaredCaptureBody,
    /// `ReplayController::is_g0_observing()` (the G0 eager arm).
    G0Observing,
}

/// What the recording-dependent branch changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Kernarg blob vs `kernelParams`, recorder bookkeeping, async vs sync H2D.
    LaunchMechanism,
    /// Capture begin/end/abort, invalidation and auto-finalize bookkeeping.
    Lifecycle,
    /// The recorded branch returns an error (or `false` from a `try_`) before
    /// any launch; the caller's other branch is inventoried separately.
    Refusal,
    /// Different kernels (or a different number of launches).
    KernelSelection,
    /// Same kernels, different grid/LDS/argument values.
    LaunchShape,
    /// Decided once per process from the backend request, not per forward.
    ProcessRoute,
    /// A debug assertion only.
    Assertion,
    /// Diagnostic/oracle code that is not on a served path.
    Oracle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Invariant,
    ByteExact,
    NotEquivalent,
}

/// Default programs (design §0 D10, §6 cutover list, §7 M0a).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Program {
    /// gfx1201 Qwen3.5 dense MQ4 H2 plain-AR decode (Redline default).
    H2Gfx1201,
    /// Qwen3.5 MoE MQ4R plain-AR decode tapes (Redline default).
    Mq4rGfx1201,
    Mq4rGfx1100,
    Mq4rGfx1151,
    /// DeepSeek4 MQ2R gfx1151 plain-AR decode tape (Redline default).
    Ds4Gfx1151,
    /// DFlash2 B=16 cycle on gfx1201 (draft + verify + commit; pre-railgun
    /// default: verify HipGraph after a direct warmup window).
    DflashGfx1201,
}

pub const PROGRAMS: &[Program] = &[
    Program::H2Gfx1201,
    Program::Mq4rGfx1201,
    Program::Mq4rGfx1100,
    Program::Mq4rGfx1151,
    Program::Ds4Gfx1151,
    Program::DflashGfx1201,
];

/// A default program whose recorded branch differs, and where.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Reach {
    pub program: Program,
    /// The part of the program's domain where the branches differ.
    pub domain: &'static str,
    /// Which comparison sees the difference: `g0` (recorded vs eager),
    /// `arm3` (railgun process vs pre-railgun default process), or both.
    pub seen_by: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Decision {
    pub id: &'static str,
    pub predicates: &'static [Predicate],
    /// The condition, with its location of record.
    pub condition: &'static str,
    pub effect: Effect,
    /// Kernels or shapes the condition switches (recorded branch first).
    pub switches: &'static str,
    /// Kernel symbols whose selection or shape the G0 harness may see differ
    /// because of this decision; the harness attributes differences with it.
    pub kernels: &'static [&'static str],
    pub verdict: Verdict,
    pub evidence: &'static str,
    pub reaches: &'static [Reach],
}

/// One `(file, function)` that consults recording state. `occurrences` is the
/// number of predicate tokens [`scan_sources`] counts in that function (file
/// relative to `crates/`; `<item>` = outside any function, e.g. a field).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Site {
    pub file: &'static str,
    pub function: &'static str,
    pub occurrences: u32,
    /// The decisions the function's predicate tokens belong to.
    pub decisions: &'static [&'static str],
}

/// A launch a default program's forward issues outside the recorder funnels
/// on purpose: a host input the program consumes, launched before the
/// program in every lowering (eager, recorded, HIP graph, PM4), so it is not
/// part of the recorded program and not a recording-dependent decision. The
/// G0 harness accepts an arm's non-funnel HIP launches (HIP launches minus
/// funnel launches) only when they equal the declared `per_forward` sum of
/// the program under test, in both arms; anything else is a bypass.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PreProgramLaunch {
    pub id: &'static str,
    /// Kernel symbol the launch dispatches.
    pub kernel: &'static str,
    /// Launch site and caller, with line citations.
    pub site: &'static str,
    /// Launches per forward of each listed program.
    pub per_forward: u32,
    pub programs: &'static [Program],
    pub evidence: &'static str,
}

// ── source scan ─────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum Tok {
    Ident(String),
    Punct(char),
}

/// Rust tokens with comments, string/char literals and numbers removed.
fn lex(src: &str) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let ident_start = |c: u8| c == b'_' || c.is_ascii_alphabetic();
    let ident_cont = |c: u8| c == b'_' || c.is_ascii_alphanumeric();
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while i < b.len() {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
        } else if (c == b'r' || (c == b'b' && b.get(i + 1) == Some(&b'r')))
            && {
                let s = if c == b'b' { i + 2 } else { i + 1 };
                let mut j = s;
                while b.get(j) == Some(&b'#') {
                    j += 1;
                }
                b.get(j) == Some(&b'"') && (j > s || c == b'r' || c == b'b')
            }
        {
            let s = if c == b'b' { i + 2 } else { i + 1 };
            let mut j = s;
            while b.get(j) == Some(&b'#') {
                j += 1;
            }
            let hashes = j - s;
            i = j + 1;
            while i < b.len() {
                if b[i] == b'"' && b[i + 1..].iter().take(hashes).filter(|&&h| h == b'#').count() == hashes {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
        } else if c == b'b' && b.get(i + 1) == Some(&b'"') {
            i += 1;
        } else if c == b'\'' {
            if b.get(i + 1) == Some(&b'\\') {
                i += 2;
                while i < b.len() && b[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            } else if b.get(i + 2) == Some(&b'\'') {
                i += 3;
            } else {
                // Lifetime or label: the identifier that follows is harmless.
                i += 1;
            }
        } else if c.is_ascii_digit() {
            while i < b.len() && (ident_cont(b[i]) || b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
                i += 1;
            }
        } else if ident_start(c) {
            let s = i;
            while i < b.len() && ident_cont(b[i]) {
                i += 1;
            }
            out.push(Tok::Ident(src[s..i].to_owned()));
        } else {
            out.push(Tok::Punct(c as char));
            i += 1;
        }
    }
    out
}

fn ident(t: Option<&Tok>) -> Option<&str> {
    match t {
        Some(Tok::Ident(s)) => Some(s),
        _ => None,
    }
}

fn punct(t: Option<&Tok>, p: char) -> bool {
    t == Some(&Tok::Punct(p))
}

/// The predicate a token at `i` consults, if any.
fn predicate_at(t: &[Tok], i: usize) -> Option<Predicate> {
    let name = ident(t.get(i))?;
    let prev = i.checked_sub(1).and_then(|p| t.get(p));
    let is_def = ident(prev) == Some("fn");
    let replay_method = || punct(prev, '.') && i >= 2 && ident(t.get(i - 2)) == Some("replay") && punct(t.get(i + 1), '(');
    match name {
        "is_recording" | "should_auto_finalize_capture" if !is_def => Some(Predicate::IsRecording),
        "replay_recording" | "redline_recording" => Some(Predicate::IsRecording),
        "capture_mode" | "graph_capture" => Some(Predicate::CaptureMode),
        "is_enabled" if replay_method() => Some(Predicate::ReplayEnabled),
        "state" if replay_method() => Some(Predicate::ReplayState),
        "capturing" => Some(Predicate::GraphSlotCapturing),
        "capture_safe" | "graph_safe" => Some(Predicate::DeclaredCaptureBody),
        "is_g0_observing" if !is_def => Some(Predicate::G0Observing),
        _ => None,
    }
}

/// Per-function predicate-token counts for one source file. Test-only code
/// (`#[cfg(test)]` items) is skipped; tokens outside any function are keyed
/// `<item>`.
pub fn scan_file(src: &str) -> BTreeMap<String, u32> {
    struct Frame {
        name: Option<String>,
        test: bool,
        depth: usize,
    }
    let t = lex(src);
    let mut counts = BTreeMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    let mut depth = 0usize;
    let mut nest = 0usize;
    let mut pending_fn: Option<(String, usize)> = None;
    let mut pending_item: Option<usize> = None;
    let mut pending_test = false;
    let mut i = 0;
    while i < t.len() {
        match &t[i] {
            Tok::Punct('#') if punct(t.get(i + 1), '[') && ident(t.get(i + 2)) == Some("cfg") && punct(t.get(i + 3), '(') && ident(t.get(i + 4)) == Some("test") && punct(t.get(i + 5), ')') => {
                pending_test = true;
                i += 6;
                continue;
            }
            Tok::Ident(k) if k == "fn" => {
                if let Some(name) = ident(t.get(i + 1)) {
                    pending_fn = Some((name.to_owned(), nest));
                }
            }
            Tok::Ident(k) if k == "mod" || k == "impl" => pending_item = Some(nest),
            Tok::Punct('(') | Tok::Punct('[') => nest += 1,
            Tok::Punct(')') | Tok::Punct(']') => nest = nest.saturating_sub(1),
            Tok::Punct(';') => {
                if pending_fn.as_ref().is_some_and(|(_, n)| *n == nest) {
                    pending_fn = None;
                }
                if pending_item == Some(nest) {
                    pending_item = None;
                }
                if nest == 0 || stack.is_empty() {
                    pending_test = false;
                }
            }
            Tok::Punct('{') => {
                depth += 1;
                let fn_here = pending_fn.take().filter(|(_, n)| *n == nest).map(|(name, _)| name);
                let item_here = pending_item.take().is_some();
                if fn_here.is_some() || item_here || pending_test {
                    stack.push(Frame { name: fn_here, test: std::mem::take(&mut pending_test), depth });
                }
            }
            Tok::Punct('}') => {
                if stack.last().is_some_and(|f| f.depth == depth) {
                    stack.pop();
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        let in_test = stack.iter().any(|f| f.test) || (pending_test && pending_fn.is_some());
        if !in_test && predicate_at(&t, i).is_some() {
            // Signature tokens belong to the function being declared.
            let function = pending_fn
                .as_ref()
                .map(|(name, _)| name.clone())
                .or_else(|| stack.iter().rev().find_map(|f| f.name.clone()))
                .unwrap_or_else(|| "<item>".to_owned());
            *counts.entry(function).or_insert(0) += 1;
        }
        i += 1;
    }
    counts
}

/// Scan every `crates/<crate>/src/**/*.rs` below `crates_dir` (this crate
/// excluded) and return `(file relative to crates_dir, function) → count`.
pub fn scan_sources(crates_dir: &Path) -> std::io::Result<BTreeMap<(String, String), u32>> {
    fn walk(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n != "target") {
                    walk(&path, files)?;
                }
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    let mut crates: Vec<_> = std::fs::read_dir(crates_dir)?.collect::<Result<Vec<_>, _>>()?;
    crates.sort_by_key(|e| e.file_name());
    for krate in crates {
        let src = krate.path().join("src");
        if krate.file_name() == env!("CARGO_PKG_NAME") || !src.is_dir() {
            continue;
        }
        walk(&src, &mut files)?;
    }
    let mut out = BTreeMap::new();
    for file in files {
        let rel = file.strip_prefix(crates_dir).unwrap_or(&file).to_string_lossy().replace('\\', "/");
        for (function, n) in scan_file(&std::fs::read_to_string(&file)?) {
            out.insert((rel.clone(), function), n);
        }
    }
    Ok(out)
}

// ── inventory queries ───────────────────────────────────────────────────────

pub fn decision(id: &str) -> Option<&'static Decision> {
    DECISIONS.iter().find(|d| d.id == id)
}

/// `(file, function) → occurrences` as the inventory declares them.
pub fn inventory_counts() -> BTreeMap<(String, String), u32> {
    SITES.iter().map(|s| ((s.file.to_owned(), s.function.to_owned()), s.occurrences)).collect()
}

/// Differences between a scan and the inventory, one line each; empty when
/// they agree.
pub fn check(scanned: &BTreeMap<(String, String), u32>) -> Vec<String> {
    let declared = inventory_counts();
    let mut out = Vec::new();
    for ((file, function), n) in scanned {
        match declared.get(&(file.clone(), function.clone())) {
            None => out.push(format!("new recording predicate: {file} :: {function} ({n} token(s)) has no SITES row")),
            Some(d) if d != n => out.push(format!("changed recording predicate: {file} :: {function} has {n} token(s), SITES says {d}")),
            _ => {}
        }
    }
    for (file, function) in declared.keys() {
        if !scanned.contains_key(&(file.clone(), function.clone())) {
            out.push(format!("stale SITES row: {file} :: {function} no longer consults recording state"));
        }
    }
    out
}

/// The certificate field `recording_dependent_decisions` for one program:
/// every decision that reaches it, with its call sites.
#[derive(Debug, Serialize)]
pub struct RecordingDependentDecision {
    pub id: &'static str,
    pub verdict: Verdict,
    pub domain: &'static str,
    pub seen_by: &'static str,
    pub sites: Vec<String>,
}

pub fn recording_dependent_decisions(program: Program) -> Vec<RecordingDependentDecision> {
    DECISIONS
        .iter()
        .flat_map(|d| d.reaches.iter().filter(move |r| r.program == program).map(move |r| (d, r)))
        .map(|(d, r)| RecordingDependentDecision {
            id: d.id,
            verdict: d.verdict,
            domain: r.domain,
            seen_by: r.seen_by,
            sites: SITES.iter().filter(|s| s.decisions.contains(&d.id)).map(|s| format!("{}::{}", s.file, s.function)).collect(),
        })
        .collect()
}

/// Programs refused as a railgun default: a `NotEquivalent` decision reaches
/// them. Each entry carries the refusing decisions and their domains.
pub fn refused_default_programs() -> BTreeMap<Program, Vec<(&'static str, &'static str)>> {
    let mut out: BTreeMap<Program, Vec<(&'static str, &'static str)>> = BTreeMap::new();
    for d in DECISIONS.iter().filter(|d| d.verdict == Verdict::NotEquivalent) {
        for r in d.reaches {
            out.entry(r.program).or_default().push((d.id, r.domain));
        }
    }
    out
}

/// Declared pre-program launches of `program` ([`PreProgramLaunch`]).
pub fn pre_program_launches(program: Program) -> Vec<&'static PreProgramLaunch> {
    PRE_PROGRAM_LAUNCHES.iter().filter(|l| l.programs.contains(&program)).collect()
}

pub fn inventory_json() -> serde_json::Value {
    let programs: serde_json::Map<String, serde_json::Value> = PROGRAMS
        .iter()
        .map(|&p| {
            let refused = refused_default_programs().remove(&p).unwrap_or_default();
            (
                serde_json::to_value(p).unwrap().as_str().unwrap().to_owned(),
                serde_json::json!({
                    "recording_dependent_decisions": recording_dependent_decisions(p),
                    "refused_as_default": !refused.is_empty(),
                    "refused_by": refused.iter().map(|(id, domain)| serde_json::json!({"decision": id, "domain": domain})).collect::<Vec<_>>(),
                    "pre_program_launches": pre_program_launches(p).iter().map(|l| l.id).collect::<Vec<_>>(),
                }),
            )
        })
        .collect();
    serde_json::json!({
        "schema": "railgun-recording-inventory",
        "version": 1,
        "decisions": DECISIONS,
        "sites": SITES,
        "pre_program_launches": PRE_PROGRAM_LAUNCHES,
        "programs": programs,
    })
}

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

pub fn inventory_markdown() -> String {
    let mut s = String::new();
    s.push_str("| # | decision | predicates | condition (location) | effect | switches | verdict | evidence | reaches (program: domain [seen by]) | sites |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
    for (n, d) in DECISIONS.iter().enumerate() {
        let preds: Vec<String> = d.predicates.iter().map(|p| serde_json::to_value(p).unwrap().as_str().unwrap().to_owned()).collect();
        let reaches: Vec<String> = d.reaches.iter().map(|r| format!("{}: {} [{}]", serde_json::to_value(r.program).unwrap().as_str().unwrap(), r.domain, r.seen_by)).collect();
        let sites: Vec<String> = SITES.iter().filter(|x| x.decisions.contains(&d.id)).map(|x| format!("`{}::{}`×{}", x.file, x.function, x.occurrences)).collect();
        s.push_str(&format!(
            "| {} | `{}` | {} | {} | {} | {} | **{}** | {} | {} | {} |\n",
            n + 1,
            d.id,
            preds.join(", "),
            md_cell(d.condition),
            serde_json::to_value(d.effect).unwrap().as_str().unwrap(),
            md_cell(d.switches),
            serde_json::to_value(d.verdict).unwrap().as_str().unwrap(),
            md_cell(d.evidence),
            if reaches.is_empty() { "none".to_owned() } else { md_cell(&reaches.join("; ")) },
            sites.join("<br>"),
        ));
    }
    s
}

/// `railgun-cert recording-inventory json|markdown|scan|check [CRATES_DIR]`.
pub fn cli(args: &[std::path::PathBuf]) -> Result<(), String> {
    let mode = args.first().and_then(|a| a.to_str()).ok_or("usage: railgun-cert recording-inventory json|markdown|scan|check [CRATES_DIR]")?;
    let crates_dir = args.get(1).cloned().unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."));
    match mode {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&inventory_json()).map_err(|e| e.to_string())?);
            Ok(())
        }
        "markdown" => {
            print!("{}", inventory_markdown());
            Ok(())
        }
        "scan" => {
            for ((file, function), n) in scan_sources(&crates_dir).map_err(|e| e.to_string())? {
                println!("{file}\t{function}\t{n}");
            }
            Ok(())
        }
        "check" => {
            let problems = check(&scan_sources(&crates_dir).map_err(|e| e.to_string())?);
            for p in &problems {
                eprintln!("{p}");
            }
            if problems.is_empty() { Ok(()) } else { Err(format!("{} inventory mismatch(es)", problems.len())) }
        }
        other => Err(format!("unknown recording-inventory mode {other}")),
    }
}

// ── the inventory ───────────────────────────────────────────────────────────
//
// Line numbers are at `13cd404c3` (the base of this inventory). Default
// programs and their pre-railgun mechanisms: H2 / MQ4R / DS4 are Redline
// tapes (recorded with `is_recording()`, `capture_mode == false`; the AR
// HipGraph is off whenever `replay.is_enabled()`, qwen35/forward.rs:1712-1716);
// the DFlash gfx1201 verify replays a HipGraph (`capture_mode == true`) after
// a direct warmup window (speculative.rs:263-293 admits MQ4G256V2 on gfx1201),
// its dense draft runs direct (draft FFN graph is MoE-only,
// speculative.rs:232-247). The railgun recorder is `Gpu::replay` (design §6),
// so railgun authoring sees `is_recording() == true, capture_mode == false`.

use Effect::*;
use Predicate::*;
use Program::*;
use Verdict::*;

const fn reach(program: Program, domain: &'static str, seen_by: &'static str) -> Reach {
    Reach { program, domain, seen_by }
}

pub const DECISIONS: &[Decision] = &[
    Decision {
        id: "launch-funnel-blob-path",
        predicates: &[IsRecording, CaptureMode],
        condition: "`record || capture_mode || force_blob_path` in Gpu::launch_maybe_blob_bound (dispatch.rs:2416-2522) and scratch::launch_maybe_blob (scratch.rs:356-431); `use_blob_path` (scratch.rs:457); the recorder's own `if !is_recording() return` (replay.rs:6107, 6186)",
        effect: LaunchMechanism,
        switches: "kernarg blob (`extra`) vs `kernelParams`; recorded/captured copy of the blob. Same function, grid, block, LDS",
        kernels: &[],
        verdict: Invariant,
        evidence: "both branches launch `self.functions[func_name]` with the caller's grid/block/shared_mem; the blob builder encodes the values `params` points at (dispatch.rs:2495-2497 contract). G0 compares the blob bytes of both arms",
        reaches: &[],
    },
    Decision {
        id: "g0-observation",
        predicates: &[IsRecording, G0Observing],
        condition: "`ReplayController::begin_g0_observation` refuses while recording; `begin_capture` refuses while observing; the launch funnels (Gpu::launch_maybe_blob_bound, Gpu::launch_blob_recorded, scratch::launch_maybe_blob) take the blob path while observing",
        effect: LaunchMechanism,
        switches: "observed copy of each funnel launch; `is_recording()` stays false",
        kernels: &[],
        verdict: Invariant,
        evidence: "observation only reads the blob; `replay::tests::g0_observation_keeps_the_eager_branch_and_excludes_recording`",
        reaches: &[],
    },
    Decision {
        id: "scratch-launch-plumbing",
        predicates: &[CaptureMode],
        condition: "`capture_mode` forwarded into scratch::launch_maybe_blob by rotate_x_mq*/rotate_quantize_x_mq8, ensure_q8_1_mmq_x(128), ensure_int4/int8_mmq_x, prepare_mq4v2_fp8_x*, convert_fp16_x_uncached (dispatch.rs, gemv.rs, scratch.rs)",
        effect: LaunchMechanism,
        switches: "launch mechanism only (see launch-funnel-blob-path); rotations and quantizers always run",
        kernels: &[],
        verdict: Invariant,
        evidence: "the flag reaches no condition other than the funnel's `record || capture_mode || force_blob_path` (e.g. scratch.rs:1859-1918 rotate_x_mq launches unconditionally)",
        reaches: &[],
    },
    Decision {
        id: "fp16-fp8-x-conversion-cache",
        predicates: &[IsRecording, CaptureMode],
        condition: "`scratch_must_convert = is_recording || capture_mode || cached_ptr != src_ptr` (scratch.rs:445-452) in ScratchState::ensure_fp16_x (scratch.rs:994) and ensure_fp8_x (scratch.rs:1157)",
        effect: KernelSelection,
        switches: "recorded/captured: the F32→F16 (`convert_f32_to_f16`) or F32→FP8 (`pack_f32_to_fp8_gfx12`) conversion is launched before every consumer; eager: skipped when the cached source pointer matches (one launch fewer per cache hit)",
        kernels: &["convert_f32_to_f16", "pack_f32_to_fp8_gfx12"],
        verdict: ByteExact,
        evidence: "the conversion is a pure elementwise function of the source bytes into the same scratch; the eager skip is taken only on the cache-coherence premise that the source is unchanged since the cached conversion (every producer invalidates via `invalidate_x_caches_for`, scratch.rs:1842-1849). Obligation: the premise itself (a stale-pointer hit is an eager-path bug, cf. muse-glimmer/forward.rs:3660-3668); G2 arm 3 confirms",
        reaches: &[reach(DflashGfx1201, "every verify window (MQ4V2 WMMA GEMMs take F16 x; `convert_f32_to_f16` is in the cycle's kernel list) and the direct draft GEMMs", "g0; arm3 in the pre-railgun direct warmup window")],
    },
    Decision {
        id: "graph-capture-lifecycle",
        predicates: &[CaptureMode, GraphSlotCapturing],
        condition: "HipGraph begin/end/abort/drop bookkeeping: graph.rs (begin/end_[verify_|replay_]graph_capture, abort, destroy), mtp_spec.rs:1063-1090, dflash.rs:2537-2564, speculative.rs:1707/3243 resets, qwen35 dense-TP abort/drop (forward.rs:4887-4908, 5193), spec_impl.rs:423, `init_with_device` defaults",
        effect: Lifecycle,
        switches: "sets/clears `capture_mode`, capture slots and blob storage; no kernel choice",
        kernels: &[],
        verdict: Invariant,
        evidence: "no launch is selected by these branches; the captured body's own predicates are inventoried separately",
        reaches: &[],
    },
    Decision {
        id: "replay-recorder-lifecycle",
        predicates: &[IsRecording, ReplayState],
        condition: "`should_auto_finalize_capture` (replay.rs:4586-4588) in qwen35 forward_scratch (forward.rs:1831), deepseek4 decode_step_with_graph (forward.rs:3642), lfm2moe decode_step_with_retained_replay (state checks, forward.rs:106-170); gemma4 lowered forward_scratch keeps its AR HipGraph off while `is_recording()` (lowered.rs)",
        effect: Lifecycle,
        switches: "finish the one-shot capture after the recorded forward",
        kernels: &[],
        verdict: Invariant,
        evidence: "runs after the forward body; selects no kernel",
        reaches: &[],
    },
    Decision {
        id: "capture-safe-invalidation",
        predicates: &[IsRecording, CaptureMode, ReplayEnabled, ReplayState],
        condition: "`scratch_growth_invalidates(graph_captured, capture_mode, recording)` (scratch.rs:573-579) via invalidate_for_scratch_growth (dispatch.rs:4648-4660); invalidate_for_kv_mode_switch poisons Redline when enabled (dispatch.rs:4610-4619)",
        effect: Lifecycle,
        switches: "defer graph/route invalidation while a capture or recording is in flight",
        kernels: &[],
        verdict: Invariant,
        evidence: "invalidation drops executable state only; the launch that follows is the same kernel",
        reaches: &[],
    },
    Decision {
        id: "memcpy-htod-auto",
        predicates: &[CaptureMode],
        condition: "Gpu::memcpy_htod_auto: async H2D on the active stream while capturing, else synchronous (dispatch.rs:2336-2347)",
        effect: LaunchMechanism,
        switches: "hipMemcpyHtoDAsync vs hipMemcpyHtoD; same bytes",
        kernels: &[],
        verdict: Invariant,
        evidence: "same source and destination bytes; ordering is stream order in both cases",
        reaches: &[],
    },
    Decision {
        id: "eager-only-refusals",
        predicates: &[IsRecording, CaptureMode, ReplayEnabled, GraphSlotCapturing],
        condition: "entry points that return an error (or `false`) when recorded/captured before any launch: prepare_mq4v2_fp8_x (dispatch.rs:2880), gfx12 FP8 WMMA GEMMs and producers (gemm.rs *_fp8*, gemv.rs *_fp8_gfx12_batched), F2 launch (gemm.rs:9782), Lloyd MMQ pad-to-128 (pad_f32_batch_to_128, gemm.rs:20157), wide gfx11 FA2 Q16 scratch growth (attention.rs:5895-5908), DFlash gfx1100 bulk state copy (dflash_state_copy.rs:130), VMM KV growth (saddle-core kv.rs:1908), DS4 heterogeneous G4 (forward.rs:3267-3273), MoE CPU top-k fallback (pipeline/mod.rs:1914)",
        effect: Refusal,
        switches: "error before any launch; the admission predicate that routes to the eager-only entry is inventoried at the caller",
        kernels: &[],
        verdict: Invariant,
        evidence: "the refused branch launches nothing; a recording that reaches it fails loudly instead of recording a different body",
        reaches: &[],
    },
    Decision {
        id: "gfx906-mmq-debug-assert",
        predicates: &[CaptureMode],
        condition: "`debug_assert!(should_use_mmq(batch) || capture_mode)` in the gfx906 MMQ launchers (gemm.rs:18939, 19121, 19275)",
        effect: Assertion,
        switches: "nothing in release builds",
        kernels: &[],
        verdict: Invariant,
        evidence: "debug assertion only",
        reaches: &[],
    },
    Decision {
        id: "mmq-screen-skipped-under-capture",
        predicates: &[IsRecording, CaptureMode],
        condition: "mmq_screen_weight returns `false` without screening or caching when `capture_mode || is_recording()` and the weight has no cached verdict (dispatch.rs:3603)",
        effect: KernelSelection,
        switches: "eager: the synthetic WMMA-vs-MMQ screen decides MMQ vs WMMA and caches it; recorded/captured with no cached verdict: WMMA",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different GEMM kernels, no byte oracle. Only the gfx906/HFQ4 MMQ route screens (gemm.rs:4385, hipfire-runtime arch.rs:239); no railgun default program takes it",
        reaches: &[],
    },
    Decision {
        id: "prefill-producer-fusions",
        predicates: &[IsRecording, CaptureMode],
        condition: "a8_prefill_active, iu4_producer_sidecar_active, iu4_silu_quant_fused_active, iu4_producer_quant_fused_active, iu4_gfx11_producer_quant_fused_active, fp8_stream_active: `!is_recording() && !capture_mode && batch >= 64 && k % 256 == 0` (dispatch.rs:3123-3208)",
        effect: KernelSelection,
        switches: "eager: A8/IU4/FP8 producer+quantize fusions and their MMQ consumers; recorded: the unfused producer + standalone quantizer + incumbent GEMM",
        kernels: &["gemm_mq4g256v2_mmq*", "quantize*", "*_quant_fused*"],
        verdict: NotEquivalent,
        evidence: "different quantization/producer kernels; no byte-exact oracle. Unreachable from every default program: all require batch >= 64, decode members have b = 1 and the DFlash verify b = 16",
        reaches: &[],
    },
    Decision {
        id: "gfx12-mqv2-prefill-projection-routes",
        predicates: &[IsRecording, CaptureMode],
        condition: "gfx1201 MQ*V2 projection dispatchers: IU4-direct MMQ and FP8-WMMA candidates (`!is_recording() && !capture_mode && batch >= 64`) and the BT8 weight-reuse tile (`mqv2_prefill_batch_tile`, gfx1201 Qkv N >= 96, gemm.rs:663-705) in gemm_{qkvza,qkv,gate_up}_hfq4g256_wmma_gfx12_mq4v2, gemm_hfq4g256_residual_wmma_gfx12_mq4v2, gemm_qkv_mq{2,5,6}g256v2_wmma_gfx12",
        effect: KernelSelection,
        switches: "eager: iu4/fp8/BT8 kernels; recorded: the historical base `*_mq4g256v2_wmma_gfx12` launch contract",
        kernels: &["gemm_qkv_mq4g256v2_wmma_gfx1201_bt*", "gemm_*_mq4g256v2_wmma_gfx12", "gemm_*_fp8*"],
        verdict: NotEquivalent,
        evidence: "different tilings/quantization, no byte-exact oracle. Unreachable from defaults: every eager-only branch needs N >= 64 (BT8 N >= 96); the DFlash verify is N = 16",
        reaches: &[],
    },
    Decision {
        id: "gfx11-mqv2-prefill-projection-routes",
        predicates: &[IsRecording, CaptureMode],
        condition: "gfx1100/gfx1151 MQ*V2 dispatchers: MMQ/IU4 at batch >= 64, BT/MW weight-reuse tiles (`mqv2_prefill_batch_tile`/`mqv2_mw_waves`, N >= 96), IU4 column-adjacent route and V2B/V2C tiles (gemm.rs:20266-20300, 20500-20540), try_gemm_*_mqv2_gfx11_reuse, MQ2V2 gfx11 routes",
        effect: KernelSelection,
        switches: "eager: MMQ/IU4/BT/MW/column kernels; recorded: base `*_mq4g256v2_wmma` or the row-major IU4 wrapper",
        kernels: &["gemm_*_mq*g256v2_*", "gemm_mq4g256v2_mmq*"],
        verdict: NotEquivalent,
        evidence: "different kernels; the IU4 row-major capture wrapper is called 'arithmetic-identical' only in a comment (gemm.rs:20520-20523), unproven. Unreachable from defaults: gfx11 MQ4R decodes HFQ4 with GEMVs (b = 1); every branch needs N >= 64",
        reaches: &[],
    },
    Decision {
        id: "gfx1100-mq4v2-verify-tier",
        predicates: &[IsRecording, CaptureMode],
        condition: "exact gfx1100 N <= 16 verify tiers: gate/up LDS-stage (`!is_recording() && !capture_mode`, gemm.rs:32712-32725, and the F16 twin mq_f16_producers.rs:487), residual LDS-stage/split-K (`!is_recording()` only, gemm.rs:34592-34610)",
        effect: KernelSelection,
        switches: "eager: *_gfx1100_ldsstage / ksplit kernels; recorded: base `gemm_*_mq4g256v2_wmma`. The residual tier ignores capture_mode, so a HipGraph keeps it while a recording does not",
        kernels: &["gemm_*_mq4g256v2_wmma*"],
        verdict: NotEquivalent,
        evidence: "different reduction order; no byte-exact oracle. Reaches no default program (gfx1100 MQ4V2 verify = DFlash on gfx1100, not a default)",
        reaches: &[],
    },
    Decision {
        id: "gfx906-cdna3-prefill-routes",
        predicates: &[CaptureMode],
        condition: "gfx906 dp4a, CDNA3 rocBLAS/MFMA and gfx942 E8 rocBLAS branches gated `!capture_mode` in gemm_{qkvza,qkv,gate_up}_hfq{4,6}g256, gemm_hfq4g256(_residual), gemm_hfq6g256_{residual,batched_lmhead}, rocblas_gemm_mfp4e8_soa_prefill_auto",
        effect: KernelSelection,
        switches: "eager: dp4a/rocBLAS/MFMA; captured: fp16 wave64 / WMMA fallbacks",
        kernels: &["gemm_*dp4a*", "gemm_*mfma*"],
        verdict: NotEquivalent,
        evidence: "different arithmetic. Arch-unreachable: no default program runs on gfx906/CDNA3",
        reaches: &[],
    },
    Decision {
        id: "gfx1201-fp8-f2-row-route",
        predicates: &[IsRecording, CaptureMode],
        condition: "fp8_f2_row_active: `G12_FP8_F2 && gfx1201 && !is_recording() && !capture_mode && n >= 512` (gemm.rs:9655-9670)",
        effect: KernelSelection,
        switches: "eager: fragment-repacked FP8 F2 row GEMM; recorded: FP16 MQ4V2 WMMA",
        kernels: &["mq4v2_fp8_fragment_repack_gfx1201", "gemm_mq4g256v2_wmma_fp8_gfx12*"],
        verdict: NotEquivalent,
        evidence: "different precision path. Unreachable: n >= 512 (G12_FP8_F2_MIN_N, gemm.rs:418)",
        reaches: &[],
    },
    Decision {
        id: "gfx1100-f16-projection-fast-route",
        predicates: &[IsRecording, CaptureMode],
        condition: "mq_f16_projection_fast_route: ChainVerify && gfx1100 && 1 <= n <= 16 && dim % 256 == 0 && !capture_mode && !is_recording() (hipfire-dispatch pipeline/batched.rs)",
        effect: KernelSelection,
        switches: "eager: fused_rmsnorm_rotate_mq[_awq]_f16_batched + *_mq4g256v2_wmma_f16; recorded: F32 producer + convert_f32_to_f16 + base *_mq4g256v2_wmma",
        kernels: &["fused_rmsnorm_rotate_mq*_f16_batched", "gemm_*_mq4g256v2_wmma_f16", "convert_f32_to_f16"],
        verdict: ByteExact,
        evidence: "operation-order-exact F16 producers (mq_f16_producers.rs:5-20); GPU oracle `hipfire-arch-qwen35/examples/test_mq_f16_projection_producers_gfx1100.rs` memcmps the F16 bytes and the projection outputs; design §2.3 (prefill.rs:6020-6022). Reaches no default program (gfx1100 DFlash); design §7 counts it as the gfx1100 DFlash graph-lowering loss",
        reaches: &[],
    },
    Decision {
        id: "widened-prefill-batching",
        predicates: &[IsRecording, CaptureMode],
        condition: "wide_candidate: ordinary whole-stack prefill with no PBS/cap/tree/tape/hidden ring/fusion and `!capture_mode && !is_recording()` (prefill.rs:2786-2794)",
        effect: LaunchShape,
        switches: "eager: widened chunk rows (up to 8192); recorded: legacy 512-row chunk cadence",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "a different chunk partition changes every chunked kernel's batch; no byte-exact oracle. Unreachable: the DFlash verify passes gdn_tape/hidden_rb (never a wide candidate) and prefill is not a railgun default member",
        reaches: &[],
    },
    Decision {
        id: "gfx1151-fa-pair-merge",
        predicates: &[IsRecording, CaptureMode],
        condition: "fa_pair_merge_admitted: two full 512-row chunks on gfx1151, `!capture_mode && !replay_recording` (prefill.rs:12062-12073, 2955-2975)",
        effect: KernelSelection,
        switches: "eager: one merged 1024-row FA2 pair; recorded: two 512-row attends",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different attention batching; no oracle. Unreachable: prefill-only",
        reaches: &[],
    },
    Decision {
        id: "q8-multirow-verify-attention",
        predicates: &[IsRecording, CaptureMode],
        condition: "q8_multirow_attn_admitted: gfx1100|gfx1201 && Q8 KV && head_dim ∈ {128, 256} && 4 <= n <= 32 && logical ctx > HIPFIRE_FA_PERTOKEN_MIN_CTX (default 4096) && !tree && !independent && !capture_mode && !replay_recording (prefill.rs:9218-9239, 13088-13099, 12445-12469); the launcher also refuses under recording (attention.rs:6971)",
        effect: KernelSelection,
        switches: "eager: kv_cache_write_q8_0_batched + attention_flash_q8_0_rows{4,8}_d{4,8} + attention_flash_asym_reduce_batched; recorded: the batched masked attend (attention_q8_0_kv_batched / attention_flash_q8_0_tile_batched)",
        kernels: &["attention_flash_q8_0_rows*", "attention_flash_asym_reduce_batched", "attention_q8_0_kv_batched*", "attention_flash_q8_0_tile_batched*", "attention_flash_q8_0_reduce*", "kv_cache_write_q8_0_batched"],
        verdict: NotEquivalent,
        evidence: "measured not bit-exact: worst relative error 4.222e-7 vs the batched attend (commit de2448f07 kernel oracle); excluded from recording by c90252322 because its tile grid is sized from the host logical context",
        reaches: &[reach(DflashGfx1201, "logical context > 4096 (every verify window past the default HIPFIRE_FA_PERTOKEN_MIN_CTX; Q8 KV, B = 16, head_dim 256)", "g0; arm3 in the pre-railgun direct warmup window")],
    },
    Decision {
        id: "captured-batch-attention-physical-cap",
        predicates: &[CaptureMode],
        condition: "`max_ctx_len = if capture_mode { kv_cache.physical_cap } else { logical_max_ctx }` (qwen35 prefill.rs:13112-13116; llama.rs forward_prefill_chunk:2698), feeding the batched Q8 attend crossover `max_ctx_len <= 4096` on gfx1201 (hipfire-dispatch families/attention.rs:2264-2273)",
        effect: KernelSelection,
        switches: "captured: attention sized by physical_cap (attention_flash_q8_0_tile_batched when physical_cap > 4096); recorded or eager: sized by the logical context (attention_q8_0_kv_batched at ctx <= 4096) and a logical `max_ctx_len` kernarg",
        kernels: &["attention_q8_0_kv_batched*", "attention_flash_q8_0_tile_batched*", "attention_flash_q8_0_reduce*"],
        verdict: NotEquivalent,
        evidence: "different attention kernels (different fp32 reduction order, cf. the 0.44-logit direct-vs-graph drift of non-flash vs flash, qwen35 forward.rs:1296-1311); only decoded text was compared at ~10K (families/attention.rs:2036-2038). Consults only capture_mode, so the railgun recorder takes the eager branch while the pre-railgun default replays a HipGraph; a recorded tape also freezes a logical-length kernel choice (G1 must split at 4096)",
        reaches: &[reach(DflashGfx1201, "every verify window replayed by the pre-railgun HipGraph with physical_cap > 4096 and logical context <= 4096 (kernel differs); above 4096 the max_ctx_len kernarg differs", "arm3")],
    },
    Decision {
        id: "gdn-chunk-scan",
        predicates: &[IsRecording, CaptureMode],
        condition: "gdn_chunk_scan admission: gfx12_gdn_chunk_scan && MQ4V2 dense && Q8 state && sequential, no tree/tape/hidden ring/fusion && !capture_mode && !is_recording() && n >= 64 (prefill.rs:13118-13160)",
        effect: KernelSelection,
        switches: "eager: chunked GatedDeltaNet scan; recorded: per-token GDN",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different recurrence evaluation order; no byte-exact oracle. Unreachable: n >= 64 and gdn_tape must be None (the DFlash verify carries a GDN tape)",
        reaches: &[],
    },
    Decision {
        id: "kv-tier-capture-forces-flash",
        predicates: &[CaptureMode],
        condition: "`use_flash = capture_mode || flash_mode == 2 || (flash_mode == 1 && pos + 1 >= 2048)` in KvTierPlan q8/fp8/bf16_attend_key (hipfire-dispatch kv_tier.rs:460-490) and the ep_batch twin (ep_batch.rs:3826, 4531); callers forward `capture_mode` into KvTierInputs (qwen35 forward.rs:4315, hipfire-dispatch pipeline/batched_attention.rs execute_fa_attend_step and pipeline/hybrid.rs attend, gemma4, llama, qwen2, hipfire-runtime llama.rs)",
        effect: KernelSelection,
        switches: "captured: flash tile attention; eager/recorded with flash_mode 0|1 at short context: non-flash attention_*_kv",
        kernels: &["attention_q8_0_kv", "attention_fp8_e4m3_kv", "attention_bf16_kv", "attention_flash_q8_0_tile", "attention_flash_fp8_e4m3_tile", "attention_flash_bf16_tile", "attention_flash_q8_0_reduce"],
        verdict: NotEquivalent,
        evidence: "non-flash vs flash differ in fp32 reduction order (~0.44 logit delta, qwen35 forward.rs:1296-1311). Unreachable under the default configuration: flash_mode = 2 on every gfx11/gfx12 arch (hipfire-runtime llama.rs:3972-3979), which dominates the capture term; HIPFIRE_ATTN_FLASH=0|1 makes it reachable (and must refuse the railgun default)",
        reaches: &[],
    },
    Decision {
        id: "attention-tile-grid-superset",
        predicates: &[IsRecording, CaptureMode],
        condition: "replay_stable_tile_count(actual, max, capture_mode, is_recording()) = max_tiles when captured or recorded (attention.rs:214-225) in attention_flash_fp8_e4m3_tile (7613), attention_flash_q8_0_windowed_impl (7429), attention_flash_bf16_windowed (2377), attention_flash_fwht{2,3,4}",
        effect: LaunchShape,
        switches: "grid.y = ceil(max_seq / tile) (GQA fp8 tile: min(that, 512)) when recorded; ceil(seq_len / tile) eager",
        kernels: &["attention_flash_fp8_e4m3_tile*", "attention_flash_q8_0_tile*", "attention_flash_bf16_tile", "attention_flash_fwht*_tile", "attention_flash_asym*_tile"],
        verdict: ByteExact,
        evidence: "by construction: the dynamic length comes from pos_buf; a tile workgroup with tile_start >= seq_len returns before any store (attention_flash_q8_0_tile.hip:103-113); the GQA fp8 tile computes each tile_id < n_tiles once whatever gridDim.y is (attention_flash_fp8_e4m3_tile_gqa.gfx1201.hip:281-294); the reduce bounds tiles by pos_buf (attention_flash_q8_0_reduce.hip:45-46). Grid only; no kernarg depends on launch_tiles. Redline already records the superset, so arm 3 sees no difference",
        reaches: &[
            reach(H2Gfx1201, "every position with ceil(seq_len/128) < min(ceil(max_seq/128), 512): grid.y of attention_flash_fp8_e4m3_tile_gqa_gfx1201", "g0"),
            reach(Mq4rGfx1201, "every position with seq_len < max_seq - tile: grid.y of attention_flash_q8_0_tile", "g0"),
            reach(Mq4rGfx1100, "every position with seq_len < max_seq - tile: grid.y of attention_flash_q8_0_tile", "g0"),
            reach(Mq4rGfx1151, "every position with seq_len < max_seq - tile: grid.y of attention_flash_q8_0_tile", "g0"),
        ],
    },
    Decision {
        id: "asym-attention-tile-grid-capture-only",
        predicates: &[IsRecording, CaptureMode],
        condition: "`launch_tiles = if capture_mode { max_tiles } else { actual_tiles }` without the recording term in attention_flash_asym{2,3,4} (attention.rs:10470, 10862, 11201) and gemma4_ext fwht3_hd512 (gemma4_ext.rs:174); asym3_hd512 consults both (gemma4_ext.rs:39)",
        effect: LaunchShape,
        switches: "captured: max-tile grid; eager and recorded: actual-tile grid (a recording freezes the grid at the recording length)",
        kernels: &["attention_flash_asym*_tile", "attention_flash_fwht3_tile*"],
        verdict: NotEquivalent,
        evidence: "tile early-exit not verified for the asym kernels; the missing is_recording term is a length hazard for any recorded asym route (G1). Unreachable: no default program uses asym KV",
        reaches: &[],
    },
    Decision {
        id: "gfx1100-asym3-q8-pair-write",
        predicates: &[IsRecording],
        condition: "gfx1100_asym3_q8_pair_enabled: gfx1100 && head_dim == 256 && !is_recording() (attention.rs:177-185)",
        effect: KernelSelection,
        switches: "eager: kv_cache_write_asym3_q8_pair_gfx1100 (16 dispatches/token fewer); recorded: separate K/V writes",
        kernels: &["kv_cache_write_asym3_q8_pair_gfx1100", "kv_cache_write_*"],
        verdict: NotEquivalent,
        evidence: "no byte-exact oracle cited. Unreachable: asym3 KV on gfx1100 is not a default program",
        reaches: &[],
    },
    Decision {
        id: "fa2-prefill-attention",
        predicates: &[IsRecording, CaptureMode],
        condition: "gfx1201/gfx11 GQA-fused FA2 prefill: `!is_recording() && !capture_mode && 64 <= batch <= 512` (attention.rs:3703-3745)",
        effect: KernelSelection,
        switches: "eager: attention_q8_0_fa2_gqa_*; recorded: incumbent batched attend",
        kernels: &["attention_q8_0_fa2_gqa*"],
        verdict: NotEquivalent,
        evidence: "different attention kernel. Unreachable: batch >= 64",
        reaches: &[],
    },
    Decision {
        id: "flash-attn-ck-prefill",
        predicates: &[IsRecording, CaptureMode],
        condition: "optional CK flash-attention prefill: refused when recording or capturing (flash_attn_ck.rs:675-681, 995-1001, select_packed_prefill_capabilities:221-226); needs HIPFIRE_FLASH_ATTN_CK_LIB and flash-prefill opt-in",
        effect: KernelSelection,
        switches: "eager: CK library launches (outside the recorder funnels); recorded: native attend",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different kernels, launched outside the funnels. Unreachable by default: no CK library is loaded and flash-prefill is declined for speculative verify on gfx12 (families/attention.rs:898-900)",
        reaches: &[],
    },
    Decision {
        id: "dflash-gfx1100-fusions",
        predicates: &[IsRecording, CaptureMode],
        condition: "gfx1100-only DFlash fusions refused under capture/recording: fused 5-way hidden commit/scatter (dflash_hidden_scatter.rs:73, 206), draft collapse split-K route (dflash_draft_fusion.rs:122)",
        effect: KernelSelection,
        switches: "eager: dflash_hidden_scatter5 / overwrite split-K draft GEMM; recorded: per-layer copy loop / base GEMM",
        kernels: &["dflash_hidden_scatter5*", "gemm_mq4g256v2_overwrite_ksplit*"],
        verdict: NotEquivalent,
        evidence: "the scatter is a pure row copy (likely exact) but the split-K draft GEMM changes reduction order; not separately proven. Arch-unreachable from the gfx1201 DFlash default",
        reaches: &[],
    },
    Decision {
        id: "ds4-gfx942-compressor-sentinel-gate",
        predicates: &[CaptureMode],
        condition: "DS4 compressor host gate: gfx942 && !capture_mode && (pos + 1) % ratio != 0 skips the compressor (forward.rs:2236-2250)",
        effect: KernelSelection,
        switches: "eager: compressor kernels skipped on non-boundary tokens; captured: launched and sentinel-early-returned",
        kernels: &["compressor_*"],
        verdict: ByteExact,
        evidence: "the host condition equals the device sentinel `commit_slot = -1` that every buf-variant commit kernel early-returns on (forward.rs:2238-2243): skipped launches are no-ops. Arch-unreachable (gfx942 only; the DS4 default is gfx1151)",
        reaches: &[],
    },
    Decision {
        id: "ds4-gfx942-ffn-overlap",
        predicates: &[CaptureMode],
        condition: "gfx942_ffn_overlap_on && !capture_mode (forward.rs:4390-4391)",
        effect: KernelSelection,
        switches: "eager: shared/routed FFN on two streams; captured: serial FFN",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "stream overlap not proven order-independent. Arch-unreachable (gfx942)",
        reaches: &[],
    },
    Decision {
        id: "replay-enabled-process-route",
        predicates: &[ReplayEnabled],
        condition: "process-level `replay.is_enabled()`: qwen35 AR HipGraph off (forward.rs:1712-1716), DS4 retained FFN split tape (forward.rs:4402-4417) and eligibility (3487-3500), lfm2moe eligibility (forward.rs:57)",
        effect: ProcessRoute,
        switches: "Redline process: split-FFN tape body, no AR HipGraph; HIP process: ffn_stub + routed FFN, AR HipGraph",
        kernels: &[],
        verdict: Invariant,
        evidence: "fixed by the backend request for the whole process, so the eager and recorded forward of one process agree. Arm-3 condition: a railgun process must present the same `is_enabled()` as the pre-railgun default (true for H2/MQ4R/DS4 Redline defaults)",
        reaches: &[],
    },
    Decision {
        id: "dspark-capture-safe-verify-body",
        predicates: &[DeclaredCaptureBody, IsRecording, CaptureMode],
        condition: "DS4 DSpark verify body parameter `capture_safe` (spec_impl.rs:155-230, forward.rs:9242-10952, 12511-12690) and dspark_requires_typed_device_copy = is_recording() || capture_mode || force_blob_path (forward.rs:9056-9058)",
        effect: KernelSelection,
        switches: "capture_safe: fixed-node body (host-sized compressed/top-k grids at max, typed device copies); ordinary: host-sized grids, direct top-k",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "an oracle exists (redline_dspark_shadow_pm4 `direct_capture_exact`, hipfire-generate redline.rs:2121) but no result is cited here. Unreachable: DSpark is not a default program",
        reaches: &[],
    },
    Decision {
        id: "dflash-draft-graph-safe-body",
        predicates: &[DeclaredCaptureBody],
        condition: "draft_ffn_layer(graph_safe) (hipfire-runtime dflash.rs:2566-2630): `memcpy_dtod_auto` + `add_f32_graph_safe` when captured, `hip.memcpy_dtod` + `add_f32` otherwise",
        effect: LaunchMechanism,
        switches: "same `add_f32` function, grid and arguments; funnel launch on the active stream vs a raw `hip.launch_kernel` on the null stream",
        kernels: &["add_f32"],
        verdict: Invariant,
        evidence: "rdna-compute norm.rs:308-374: identical kernel, grid, block and argument values. Recorder coverage gap, not a selection difference: with graph_safe = false (the dense DFlash default; the draft FFN graph is MoE-only, speculative.rs:232-247) the add bypasses the funnels, so a recording of the draft misses one launch per draft layer and G0 reports the bypass until the railgun draft is authored with graph_safe = true",
        reaches: &[],
    },
    Decision {
        id: "tp-graph-capture-barrier",
        predicates: &[CaptureMode],
        condition: "barrier_rank_streams_reuse: all 3|4 TP ranks capturing → capture_tp_graph_barrier (hipfire-runtime multi_gpu.rs:840-848)",
        effect: LaunchMechanism,
        switches: "captured cross-rank signal barrier vs event barrier",
        kernels: &[],
        verdict: Invariant,
        evidence: "synchronisation only; multi-GPU TP is not a default program",
        reaches: &[],
    },
    Decision {
        id: "redline-oracle-diagnostics",
        predicates: &[DeclaredCaptureBody, ReplayState],
        condition: "hipfire-generate redline.rs and hipfire-daemon oracle handlers: capture_safe arms of the DSpark oracle, replay state in JSON responses",
        effect: Oracle,
        switches: "which body an oracle arm runs; route state reporting",
        kernels: &[],
        verdict: Invariant,
        evidence: "diagnostic requests only, never on a served path",
        reaches: &[],
    },
    Decision {
        id: "verify-attention-wmma-gqa",
        predicates: &[IsRecording],
        condition: "launch_verify_gqa and attention_verify_wmma_with return `false` when `!flags.verify_attn || is_recording()` (attention.rs:7860, 8062)",
        effect: KernelSelection,
        switches: "eager or HipGraph: the verify GQA / WMMA attention tables; recorded: the caller's incumbent verify attend",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different attention kernels, no byte oracle. Unreachable by default: `HIPFIRE_VERIFY_ATTN` / `kernel.verify_attn` is opt-in (feature_flags.rs verify_attn default false)",
        reaches: &[],
    },
    Decision {
        id: "dflash-gdn-replay-multilayer",
        predicates: &[IsRecording],
        condition: "gdn_replay_ml_eligible: exact gfx1201 && !gdn_replay_ml_off && !is_recording() && fast GDN requant && head_dim 128 && 1 <= n_steps <= 16 (dflash_gdn_replay.rs:145-160), consulted by the DFlash commit's GDN replay (speculative.rs:1792)",
        effect: KernelSelection,
        switches: "eager or HipGraph: the two-launch multi-layer GDN replay; recorded: per-layer GDN replay launches",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different launch set with no cited byte oracle. Default-on on gfx1201 (opt-out HIPFIRE_GDN_REPLAY_ML_OFF); the condition ignores capture_mode, so the pre-railgun DFlash HipGraph keeps it while a railgun recording does not",
        reaches: &[reach(Program::DflashGfx1201, "DFlash commit GDN replay, 1 <= accepted steps <= 16", "arm3")],
    },
    Decision {
        id: "select-regrid-row-selectors",
        predicates: &[IsRecording],
        condition: "select_regrid_enabled: exact gfx1201 && !select_regrid_off && !is_recording() (select_regrid.rs:45-47), consulted by the batched top-K and argmax selectors (sampling.rs:129, 838)",
        effect: KernelSelection,
        switches: "eager or HipGraph: select_regrid 1024-thread row selectors (+ topk_values_fixup_f32 on tied rows); recorded: topk_values_batched_f32 / argmax_f32_batched",
        kernels: &["topk_values_batched_f32", "argmax_f32_batched"],
        verdict: ByteExact,
        evidence: "select_regrid.rs module header: outputs byte-identical to the shipping kernels for every input, ties and NaNs included; examples/test_select_regrid checks it",
        reaches: &[reach(Program::DflashGfx1201, "DFlash2 selector top-K and verify argmax", "arm3")],
    },
    Decision {
        id: "qwen4-f16-wmma-prefill",
        predicates: &[IsRecording, CaptureMode],
        condition: "QWEN4_F16_WMMA routes at rows >= QWEN4_F16_WMMA_MIN_TOKENS and `!is_recording() && !capture_mode`: qwen4_f16_wmma_applies, qwen4_bf16_streams (gemm.rs:27873-27902), gated_delta_chunk_route (tensor_ops.rs:559), qsa_dense_wmma_applies (tensor_ops.rs:3685), qsa_sparse_wmma_applies (tensor_ops.rs:3947, rows >= QSA_ATTENTION_HG12_MIN_ROWS), qsa_gathered_wmma_applies (tensor_ops.rs:3858, on unless HIPFIRE_QWEN4_QSA_WMMA_GATHER=0; F32 state rows >= QSA_ATTENTION_HG12_MIN_ROWS)",
        effect: KernelSelection,
        switches: "eager: F16 WMMA BF16 GEMM / chunked GDN / dense, sparse or gathered QSA WMMA; recorded: incumbent qwen4 kernels",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different kernels, no byte oracle. Prefill-only (row threshold), and no railgun default program is a qwen4 program",
        reaches: &[],
    },
    Decision {
        id: "qwen4-indexed-attention-lds-bound",
        predicates: &[IsRecording, CaptureMode],
        condition: "indexed_attention_attention_batch_impl: `shape_selected = if is_recording() || capture_mode { p.shape_selected } else { max_selected }` (tensor_ops.rs:3569)",
        effect: LaunchShape,
        switches: "recorded/captured: position-independent LDS bound; eager: LDS sized to the chunk's longest row",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different launch shape with no cited byte oracle; no railgun default program is a qwen4 program",
        reaches: &[],
    },
    Decision {
        id: "qwen4-qsa-select-from-scores",
        predicates: &[IsRecording, CaptureMode],
        condition: "indexed_attention_select_batch_impl: parallel F32 select with the pinned index geometry (4 heads, index_dim % 4 == 0 and <= 128, budget <= QSA_SELECT_FROM_SCORES_MAX_BUDGET) and `!is_recording() && !capture_mode` takes indexed_attention_select_rows8 (tensor_ops.rs:2963-2974)",
        effect: KernelSelection,
        switches: "eager: indexed_attention_select_scores_rows{8,16}_f32 + indexed_attention_select_from_scores; recorded: the incumbent batched select",
        kernels: &[],
        verdict: ByteExact,
        evidence: "selection bytes unchanged (indexed_attention_select_rows8 contract); rdna-compute tests batched_select_ranking_matches_the_serial_selection_sort and past_budget_select_matches_serial_on_production_geometry compare the live route with the serial selection. No railgun default program is a qwen4 program",
        reaches: &[],
    },
    Decision {
        id: "packed-mq4-prefill",
        predicates: &[IsRecording, CaptureMode],
        condition: "packed_mq4_admitted: packed_mq4_prefill && gfx1100 && !(capture_mode || is_recording()) && n % 256 == 0 && listed (m, k) (packed_mq4.rs:16-42)",
        effect: KernelSelection,
        switches: "eager: PR #616 packed MQ4 Q8-group32 GEMM; recorded: incumbent MQ4 GEMM",
        kernels: &[],
        verdict: NotEquivalent,
        evidence: "different numerics (Q8 activations). Unreachable by default: packed_mq4_prefill is default-off and n % 256 == 0 excludes decode",
        reaches: &[],
    },
    Decision {
        id: "qwen4-opt-in-eager-fusions",
        predicates: &[IsRecording, CaptureMode],
        condition: "opt-in Qwen4 prefill fusions taken only with `!is_recording() && !capture_mode`: hyper_read_pairs_write (HIPFIRE_QWEN4_HC_FUSE >= 1, layer_ops.rs), gemm_mq6g256v2_hcw_applies (HC_FUSE >= 2, gemm.rs), absorbs_clear (HIPFIRE_QWEN4_MOE_COMBINE_ZINIT, moe_program.rs; also not under a retained body), indexed_attention_select_batch_impl (HIPFIRE_QWEN4_QSA_SELECT_EXACT, tensor_ops.rs), execute_grouped_depthwise (HIPFIRE_QWEN4_PLE_FUSE, layer_ops.rs), gemm_bf16_xf16_f16_wmma (HIPFIRE_QWEN4_HC_DOWN_TILE, gemm.rs), qwen4_trunk_iu4_applies and gemm_qwen4_trunk_mq4_xf16 (HIPFIRE_QWEN4_TRUNK_IU4, gemm.rs), hc_row_fold_applies (HIPFIRE_QWEN4_HC_ROW_FOLD, hc_row_fold.rs), moe_router_softmax_top10_f32 (HIPFIRE_QWEN4_ROUTER_FAST, moe.rs)",
        effect: KernelSelection,
        switches: "eager: fused HC read/write gates, MQ6 attention-output GEMM with the HC write epilogue, zero-init grouped combine without the moe_output fill, exact tile-sort QSA selector, fused PLE block, 160x64 HC-down tile, masked trunk IU4 / exact-activation MQ4 trunk GEMMs, HC row fold, fast router top-10; recorded/captured: the incumbent unfused launches",
        kernels: &[],
        verdict: ByteExact,
        evidence: "scoped rdna-compute tests on gfx1151 and gfx1201 compare each fused arm with the unfused launches byte for byte (CHANGELOG 0.4.1, Flash-Next performance); HC_FUSE (level 3) and MOE_COMBINE_ZINIT default on for exact gfx1151 only, QSA_SELECT_EXACT is default off, and no railgun default program is a qwen4 program",
        reaches: &[],
    },
    Decision {
        id: "qwen4-g2-expert-stage",
        predicates: &[GraphSlotCapturing],
        condition: "G2 long-prefill expert staging in hipfire-arch-qwen4 forward_chunk_scoped (gpu_forward.rs): a chunk of >= HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS rows stages host-mapped expert layers into borrowed VRAM only when the active stream is not capturing (`stream_is_capturing`)",
        effect: LaunchShape,
        switches: "eager: host layers' QT44/QT53 expert bytes DMA-copied into the donor layers' VRAM, the MoE launches read them through the stage pointer tables, donors restored before the forward returns; stream capture: the same MoE kernels read the host-mapped experts",
        kernels: &[],
        verdict: ByteExact,
        evidence: "no arithmetic changes, only where the unchanged expert bytes are read from; Flash-Next 16K logits on gfx1201 are byte-identical with HIPFIRE_QWEN4_EXPERT_STAGE on and off (CHANGELOG 0.4.1, Flash-Next performance); default on for exact gfx1201 only, and no railgun default program is a qwen4 program",
        reaches: &[],
    },
];
pub const SITES: &[Site] = &[
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "attention_block_batched_mixed", occurrences: 16, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "attention_block_batched_swa_only", occurrences: 3, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "compressor_forward_impl", occurrences: 1, decisions: &["ds4-gfx942-compressor-sentinel-gate"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "decode_step_heterogeneous", occurrences: 4, decisions: &["eager-only-refusals"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "decode_step_with_graph", occurrences: 2, decisions: &["replay-enabled-process-route", "replay-recorder-lifecycle"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "ds4_moe_block_core", occurrences: 2, decisions: &["ds4-gfx942-ffn-overlap", "replay-enabled-process-route"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "dspark_requires_typed_device_copy", occurrences: 2, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-deepseek4/src/forward.rs", function: "forward_prefill_batch_chunk_impl", occurrences: 3, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-deepseek4/src/spec_impl.rs", function: "dspark_verify_forward", occurrences: 2, decisions: &["dspark-capture-safe-verify-body", "graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-deepseek4/src/spec_impl.rs", function: "redline_dspark_verify_direct", occurrences: 2, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-deepseek4/src/spec_impl.rs", function: "redline_dspark_verify_pm4", occurrences: 1, decisions: &["dspark-capture-safe-verify-body"] },
    Site { file: "hipfire-arch-gemma4/src/lowered.rs", function: "forward_prefill_batch_v2", occurrences: 6, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-gemma4/src/lowered.rs", function: "forward_scratch", occurrences: 3, decisions: &["graph-capture-lifecycle", "replay-recorder-lifecycle"] },
    Site { file: "hipfire-arch-gemma4/src/lowered.rs", function: "full_layer_decode_impl", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-gemma4/src/lowered.rs", function: "run_attend", occurrences: 4, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-gemma4/src/lowered.rs", function: "sliding_layer_decode_impl", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-lfm2moe/src/forward.rs", function: "decode_step", occurrences: 1, decisions: &["replay-enabled-process-route"] },
    Site { file: "hipfire-arch-lfm2moe/src/forward.rs", function: "decode_step_with_retained_replay", occurrences: 3, decisions: &["replay-recorder-lifecycle"] },
    Site { file: "hipfire-arch-llama/src/arch.rs", function: "forward_scratch_layers", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-qwen2/src/qwen2.rs", function: "attend_plan", occurrences: 1, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-qwen35/src/mtp_spec.rs", function: "abort_mtp_proposal_graph_capture", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/mtp_spec.rs", function: "begin_mtp_proposal_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/mtp_spec.rs", function: "end_mtp_proposal_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/ep_batch.rs", function: "forward_scratch_layers_multi", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/forward.rs", function: "dense_tp_abort_captures", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/forward.rs", function: "dense_tp_drop_graphs", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/forward.rs", function: "forward_scratch", occurrences: 2, decisions: &["replay-enabled-process-route", "replay-recorder-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/forward.rs", function: "forward_scratch_dense_tp", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/forward.rs", function: "kv_cache_attention_dispatch", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/prefill.rs", function: "fa_pair_merge_admitted", occurrences: 4, decisions: &["gfx1151-fa-pair-merge"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/prefill.rs", function: "forward_batch_chunk_impl", occurrences: 5, decisions: &["q8-multirow-verify-attention", "captured-batch-attention-physical-cap", "gdn-chunk-scan"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/prefill.rs", function: "forward_prefill_batch_with_pbs_opts_inner", occurrences: 4, decisions: &["widened-prefill-batching", "gfx1151-fa-pair-merge"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/prefill.rs", function: "forward_prefill_chunk_pair", occurrences: 2, decisions: &["q8-multirow-verify-attention"] },
    Site { file: "hipfire-arch-qwen35/src/qwen35/prefill.rs", function: "q8_multirow_attn_admitted", occurrences: 4, decisions: &["q8-multirow-verify-attention"] },
    Site { file: "hipfire-arch-qwen35/src/arch.rs", function: "load_weights", occurrences: 1, decisions: &["replay-enabled-process-route"] },
    Site { file: "hipfire-arch-qwen35/src/speculative.rs", function: "replay_gdn", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen35/src/speculative.rs", function: "rides_ordinary_prefill", occurrences: 2, decisions: &["widened-prefill-batching", "gdn-chunk-scan"] },
    Site { file: "hipfire-arch-qwen35/src/speculative.rs", function: "verify_dflash_block_inner", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-arch-qwen4/src/gpu_forward.rs", function: "forward_chunk_scoped", occurrences: 6, decisions: &["replay-recorder-lifecycle", "qwen4-g2-expert-stage"] },
    Site { file: "hipfire-daemon/src/main.rs", function: "main", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-dispatch-tests/src/llama.rs", function: "tier_inputs_base", occurrences: 1, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/families/kv_tier.rs", function: "<item>", occurrences: 1, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/families/kv_tier.rs", function: "bf16_attend_key", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/families/kv_tier.rs", function: "derive", occurrences: 5, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/families/kv_tier.rs", function: "fp8_attend_key", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/families/kv_tier.rs", function: "q8_attend_key", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/pipeline/batched.rs", function: "mq_f16_projection_fast_route", occurrences: 2, decisions: &["gfx1100-f16-projection-fast-route"] },
    Site { file: "hipfire-dispatch/src/pipeline/batched_attention.rs", function: "execute_fa_attend_step", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/pipeline/hybrid.rs", function: "attend", occurrences: 2, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-dispatch/src/pipeline/layer_ops.rs", function: "hyper_read_pairs_write", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "hipfire-dispatch/src/pipeline/layer_ops.rs", function: "execute_grouped_depthwise", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "hipfire-dispatch/src/pipeline/mod.rs", function: "dump_hidden_localize", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-dispatch/src/pipeline/moe_program.rs", function: "absorbs_clear", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "hipfire-dispatch/src/pipeline/sealed_moe.rs", function: "validate_for_gpu", occurrences: 1, decisions: &["eager-only-refusals"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_bench_decode_deepseek4", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_bench_decode_lfm2moe", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_greedy_trace", occurrences: 2, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_qwen4_replay_failure", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_run_dspark_direct_arm", occurrences: 2, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_shadow_dspark_verify_pm4", occurrences: 12, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-generate/src/redline.rs", function: "redline_shadow_qwen4", occurrences: 1, decisions: &["redline-oracle-diagnostics"] },
    Site { file: "hipfire-runtime/src/dflash.rs", function: "abort_draft_ffn_graph_capture", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-runtime/src/dflash.rs", function: "begin_draft_ffn_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-runtime/src/dflash.rs", function: "draft_ffn_layer", occurrences: 3, decisions: &["dflash-draft-graph-safe-body"] },
    Site { file: "hipfire-runtime/src/dflash.rs", function: "end_draft_ffn_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "hipfire-runtime/src/llama.rs", function: "forward_prefill_chunk", occurrences: 3, decisions: &["captured-batch-attention-physical-cap", "kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-runtime/src/llama.rs", function: "tier_inputs", occurrences: 1, decisions: &["kv-tier-capture-forces-flash"] },
    Site { file: "hipfire-runtime/src/multi_gpu.rs", function: "barrier_rank_streams_reuse", occurrences: 1, decisions: &["tp-graph-capture-barrier"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_asym2", occurrences: 1, decisions: &["asym-attention-tile-grid-capture-only"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_asym3_impl", occurrences: 1, decisions: &["asym-attention-tile-grid-capture-only"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_asym4", occurrences: 1, decisions: &["asym-attention-tile-grid-capture-only"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_bf16_windowed", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_fp8_e4m3_tile", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_fwht2", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_fwht3", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_fwht4", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_q8_0_rows_masked", occurrences: 1, decisions: &["q8-multirow-verify-attention"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_q8_0_windowed_impl", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_q8_0_fa2_gqa_gfx11", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_q8_0_flash_prefill_wmma", occurrences: 2, decisions: &["fa2-prefill-attention"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_verify_wmma_with", occurrences: 1, decisions: &["verify-attention-wmma-gqa"] },
    Site { file: "rdna-compute/src/attention.rs", function: "gfx1100_asym3_q8_pair_enabled", occurrences: 1, decisions: &["gfx1100-asym3-q8-pair-write"] },
    Site { file: "rdna-compute/src/attention.rs", function: "gfx12_q8_fa2_prefill_admitted", occurrences: 2, decisions: &["fa2-prefill-attention"] },
    Site { file: "rdna-compute/src/attention.rs", function: "launch_verify_gqa", occurrences: 1, decisions: &["verify-attention-wmma-gqa"] },
    Site { file: "rdna-compute/src/attention.rs", function: "replay_stable_tile_count", occurrences: 4, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/dflash_draft_fusion.rs", function: "draft_collapse_mq4v2_route", occurrences: 2, decisions: &["dflash-gfx1100-fusions"] },
    Site { file: "rdna-compute/src/dflash_hidden_scatter.rs", function: "dflash_hidden_commit5_applicable", occurrences: 2, decisions: &["dflash-gfx1100-fusions"] },
    Site { file: "rdna-compute/src/dflash_hidden_scatter.rs", function: "dflash_hidden_scatter5_try", occurrences: 2, decisions: &["dflash-gfx1100-fusions"] },
    Site { file: "rdna-compute/src/dflash_state_copy.rs", function: "dflash_state_bulk_copy_gfx1100_on_stream", occurrences: 1, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "a8_prefill_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "convert_fp16_x_uncached", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_fp16_x", occurrences: 3, decisions: &["fp16-fp8-x-conversion-cache"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_fp8_x", occurrences: 3, decisions: &["fp16-fp8-x-conversion-cache"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_int4_mmq_x", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_int8_mmq_x", occurrences: 1, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_q8_1_mmq_x", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "ensure_q8_1_mmq_x128", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "fp8_stream_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "init_with_device", occurrences: 3, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "invalidate_for_kv_mode_switch", occurrences: 2, decisions: &["capture-safe-invalidation"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "invalidate_for_scratch_growth", occurrences: 2, decisions: &["capture-safe-invalidation"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "iu4_gfx11_producer_quant_fused_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "iu4_producer_quant_fused_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "iu4_producer_sidecar_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "iu4_silu_quant_fused_active", occurrences: 2, decisions: &["prefill-producer-fusions"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "launch_blob_recorded", occurrences: 3, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "launch_maybe_blob_bound", occurrences: 4, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "memcpy_dtoh_auto", occurrences: 1, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "memcpy_htod_auto", occurrences: 1, decisions: &["memcpy-htod-auto"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "mmq_screen_weight", occurrences: 2, decisions: &["mmq-screen-skipped-under-capture"] },
    Site { file: "rdna-compute/src/dispatch.rs", function: "prepare_mq4v2_fp8_x", occurrences: 6, decisions: &["eager-only-refusals", "scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/flash_attn_ck.rs", function: "<item>", occurrences: 2, decisions: &["flash-attn-ck-prefill"] },
    Site { file: "rdna-compute/src/flash_attn_ck.rs", function: "select_packed_prefill_capabilities", occurrences: 2, decisions: &["flash-attn-ck-prefill"] },
    Site { file: "rdna-compute/src/flash_attn_ck.rs", function: "try_flash_attn_ck_q8_d256_prefill", occurrences: 6, decisions: &["flash-attn-ck-prefill"] },
    Site { file: "rdna-compute/src/flash_attn_ck.rs", function: "try_flash_attn_ck_transformed_prefill_impl", occurrences: 6, decisions: &["flash-attn-ck-prefill"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "fp8_f2_row_active", occurrences: 2, decisions: &["gfx1201-fp8-f2-row-route"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "fp8_f2_row_launch", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "pad_f32_batch_to_128", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq4g256", occurrences: 2, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq4g256_mmq_gfx906_prequant", occurrences: 2, decisions: &["gfx906-mmq-debug-assert"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2", occurrences: 4, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared_lloyd", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_hfq6g256", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_mq2g256v2_wmma_gfx11", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_gate_up_mq4g256v2_wmma", occurrences: 6, decisions: &["gfx11-mqv2-prefill-projection-routes", "gfx1100-mq4v2-verify-tier"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256", occurrences: 2, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_mmq_set_gfx906", occurrences: 2, decisions: &["gfx906-mmq-debug-assert"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_residual", occurrences: 3, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_residual_wmma_gfx12_mq4v2", occurrences: 4, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_prepared_lloyd", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq6g256_batched_lmhead", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_hfq6g256_residual", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_mq2g256v2_residual_wmma_gfx11", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_mq4g256v2_mmq_prequant_iu4", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_mq4g256v2_residual_wmma", occurrences: 5, decisions: &["gfx11-mqv2-prefill-projection-routes", "gfx1100-mq4v2-verify-tier"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256", occurrences: 2, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256_mmq_gfx906", occurrences: 2, decisions: &["gfx906-mmq-debug-assert"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2", occurrences: 6, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared_lloyd", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_hfq6g256", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_mq2g256v2_wmma_gfx11", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_mq2g256v2_wmma_gfx12", occurrences: 2, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_mq4g256v2_wmma", occurrences: 4, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_mq5g256v2_wmma_gfx12", occurrences: 2, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkv_mq6g256v2_wmma_gfx12", occurrences: 2, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq4g256", occurrences: 3, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2", occurrences: 4, decisions: &["gfx12-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared_lloyd", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_hfq6g256", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_mq2g256v2_wmma_gfx11", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qkvza_mq4g256v2_wmma", occurrences: 4, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "iu4_v2_tile", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "rocblas_gemm_mfp4e8_soa_prefill_auto", occurrences: 1, decisions: &["gfx906-cdna3-prefill-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "try_gemm_gate_up_mqv2_gfx11_reuse", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "try_gemm_mqv2_residual_gfx11_reuse", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "try_gemm_qkv_mqv2_gfx11_reuse", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "try_gemm_qkvza_mqv2_gfx11_reuse", occurrences: 2, decisions: &["gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemma4_ext.rs", function: "attention_flash_asym3_hd512", occurrences: 2, decisions: &["asym-attention-tile-grid-capture-only"] },
    Site { file: "rdna-compute/src/gemma4_ext.rs", function: "attention_flash_fwht3_hd512", occurrences: 1, decisions: &["asym-attention-tile-grid-capture-only"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "fused_rmsnorm_rotate_mq_fp8_gfx12_batched", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "fused_silu_rotate_mq_fp8_gfx12_batched_impl", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "gated_norm_rotate_mq_fp8_gfx12_batched", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_quantize_x_mq8", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_x_mq_128", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_x_mq_awq", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_x_mq_awq_batched", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_x_mq_dual_fp8", occurrences: 3, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/gemv.rs", function: "rotate_x_mq_fp8_gfx12_batched", occurrences: 2, decisions: &["eager-only-refusals"] },
    Site { file: "rdna-compute/src/graph.rs", function: "<item>", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "abort_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "begin_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "begin_graph_capture_relaxed", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "begin_replay_graph_capture", occurrences: 5, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "begin_verify_graph_capture", occurrences: 5, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "end_graph_capture", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "end_graph_capture_segment", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "end_replay_graph_capture", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "end_verify_graph_capture", occurrences: 2, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "replay_graph_destroy_all", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/graph.rs", function: "verify_graph_destroy_all", occurrences: 1, decisions: &["graph-capture-lifecycle"] },
    Site { file: "rdna-compute/src/mq_f16_producers.rs", function: "gemm_gate_up_mq4g256v2_wmma_f16", occurrences: 2, decisions: &["gfx1100-mq4v2-verify-tier"] },
    Site { file: "rdna-compute/src/packed_mq4.rs", function: "packed_mq4_admitted", occurrences: 2, decisions: &["packed-mq4-prefill"] },
    Site { file: "rdna-compute/src/replay.rs", function: "begin_g0_observation", occurrences: 1, decisions: &["g0-observation"] },
    Site { file: "rdna-compute/src/replay.rs", function: "retained_body_active", occurrences: 1, decisions: &["replay-recorder-lifecycle"] },
    Site { file: "rdna-compute/src/replay.rs", function: "record_hip_launch_typed_bound", occurrences: 1, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/replay.rs", function: "record_hip_launch_with_accesses", occurrences: 1, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/replay.rs", function: "should_auto_finalize_capture", occurrences: 1, decisions: &["replay-recorder-lifecycle"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "convert_fp16_x_uncached", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_fp16_x", occurrences: 4, decisions: &["fp16-fp8-x-conversion-cache"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_fp8_x", occurrences: 4, decisions: &["fp16-fp8-x-conversion-cache"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_int4_mmq_x", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_int8_mmq_x", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_q8_1_mmq_x", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "ensure_q8_1_mmq_x128", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "launch_maybe_blob", occurrences: 5, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "prepare_mq4v2_fp8_x", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "prepare_mq4v2_fp8_x_f32", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "prepare_mq4v2_fp8_x_impl", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_quantize_x_mq8", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq_128", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq_awq", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq_awq_batched", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq_batched", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "rotate_x_mq_dual_fp8", occurrences: 2, decisions: &["scratch-launch-plumbing"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "scratch_growth_invalidates", occurrences: 2, decisions: &["capture-safe-invalidation"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "scratch_must_convert", occurrences: 4, decisions: &["fp16-fp8-x-conversion-cache"] },
    Site { file: "rdna-compute/src/scratch.rs", function: "use_blob_path", occurrences: 4, decisions: &["launch-funnel-blob-path"] },
    Site { file: "rdna-compute/src/select_regrid.rs", function: "select_regrid_enabled", occurrences: 1, decisions: &["select-regrid-row-selectors"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "gated_delta_chunk_route", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "indexed_attention_attention_batch_impl", occurrences: 2, decisions: &["qwen4-indexed-attention-lds-bound"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "indexed_attention_select_batch_impl", occurrences: 4, decisions: &["qwen4-opt-in-eager-fusions", "qwen4-qsa-select-from-scores"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "qsa_dense_wmma_applies", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "qsa_gathered_wmma_applies", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "rdna-compute/src/tensor_ops.rs", function: "qsa_sparse_wmma_applies", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "rdna-compute/src/attention.rs", function: "attention_flash_f16_windowed", occurrences: 2, decisions: &["attention-tile-grid-superset"] },
    Site { file: "rdna-compute/src/dflash_gdn_replay.rs", function: "gdn_replay_ml_eligible", occurrences: 1, decisions: &["dflash-gdn-replay-multilayer"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_mq6g256v2_hcw_applies", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_bf16_xf16_f16_wmma", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_qwen4_trunk_mq4_xf16", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "qwen4_trunk_iu4_applies", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/hc_row_fold.rs", function: "hc_row_fold_applies", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/moe.rs", function: "moe_router_softmax_top10_f32", occurrences: 2, decisions: &["qwen4-opt-in-eager-fusions"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "gemm_mq6g256v2_xf16_applies", occurrences: 2, decisions: &["gfx12-mqv2-prefill-projection-routes", "gfx11-mqv2-prefill-projection-routes"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "qwen4_bf16_streams", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "rdna-compute/src/gemm.rs", function: "qwen4_f16_wmma_applies", occurrences: 2, decisions: &["qwen4-f16-wmma-prefill"] },
    Site { file: "saddle-core/src/kv.rs", function: "grow_vmm_tensors", occurrences: 1, decisions: &["eager-only-refusals"] },
];

// Line numbers at `7454b5f64` (railgun/m2).
pub const PRE_PROGRAM_LAUNCHES: &[PreProgramLaunch] = &[PreProgramLaunch {
    id: "qwen35-token-embedding-q8",
    kernel: "embedding_q8",
    site: "rdna-compute/src/embedding.rs:37-64 (Gpu::embedding_lookup_q8: raw hip.launch_kernel, null stream, token id as a by-value kernarg), called from hipfire-arch-qwen35/src/qwen35/forward.rs:1719-1734 (forward_scratch, before the AQL/PM4/HIP-graph/direct route branches at :1737-1830)",
    per_forward: 1,
    programs: &[H2Gfx1201],
    evidence: "railgun G01 G0 (gfx1201 card B, H2 group-alpha-refit, Q8 token embedding): 900 HIP launches vs 899 funnel launches per forward in both arms at 10 contexts; rocprofv3 kernel trace shows embedding_q8 dispatched immediately before the pos H2D copy and the 899-launch program in each decode forward; the recorded 899-launch program contains no embedding kernel",
}];

#[cfg(test)]
mod tests {
    use super::*;

    fn crates_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    /// The MR gate: every function in `crates/*/src` that consults recording
    /// or capture state has an inventory row with the same token count, and
    /// no row is stale. Adding an `is_recording()`-style check fails here
    /// until the decision it changes is classified in `DECISIONS`.
    #[test]
    fn every_recording_predicate_in_the_tree_has_an_inventory_row() {
        let scanned = scan_sources(&crates_dir()).unwrap();
        let problems = check(&scanned);
        assert!(problems.is_empty(), "recording inventory out of date:\n{}", problems.join("\n"));
    }

    #[test]
    fn inventory_rows_are_consistent() {
        let mut ids = std::collections::BTreeSet::new();
        for d in DECISIONS {
            assert!(ids.insert(d.id), "duplicate decision {}", d.id);
            assert!(SITES.iter().any(|s| s.decisions.contains(&d.id)), "decision {} has no site", d.id);
            assert!(!d.evidence.is_empty(), "decision {} has no evidence", d.id);
            match d.verdict {
                // An invariant decision by definition changes nothing a
                // program could observe, so it cannot reach one.
                Verdict::Invariant => assert!(d.reaches.is_empty(), "invariant {} reaches a program", d.id),
                Verdict::ByteExact | Verdict::NotEquivalent => {
                    assert!(matches!(d.effect, Effect::KernelSelection | Effect::LaunchShape), "{} verdict needs a selection/shape effect", d.id)
                }
            }
        }
        for s in SITES {
            assert!(!s.decisions.is_empty());
            for id in s.decisions {
                assert!(decision(id).is_some(), "{}::{} names unknown decision {id}", s.file, s.function);
            }
        }
        let mut launch_ids = std::collections::BTreeSet::new();
        for l in PRE_PROGRAM_LAUNCHES {
            assert!(launch_ids.insert(l.id), "duplicate pre-program launch {}", l.id);
            assert!(l.per_forward > 0 && !l.programs.is_empty(), "{} declares no launch", l.id);
            assert!(!l.site.is_empty() && !l.evidence.is_empty(), "{} has no site or evidence", l.id);
        }
    }

    #[test]
    fn scan_counts_code_tokens_only() {
        let src = r##"
            // gpu.replay.is_recording() in a comment
            /* capture_mode in a /* nested */ block */
            fn decide(gpu: &Gpu, capture_mode: bool) -> bool {
                let s = "capture_mode"; let r = r#"is_recording()"#; let c = '"';
                let _ = gpu.graphs.capture_mode;
                gpu.replay.is_enabled() && !gpu.replay.is_recording()
            }
            pub fn is_recording(&self) -> bool { self.state == 1 }
            struct GraphState { pub capture_mode: bool, capturing: Option<usize> }
            #[cfg(test)]
            mod tests { fn t() { let _ = capture_mode; } }
            #[cfg(test)]
            fn helper(capture_mode: bool) {}
            impl X { fn other(&self) { let _ = self.x.state(); let _ = gpu.replay.state(); } }
        "##;
        let counts = scan_file(src);
        assert_eq!(counts.get("decide"), Some(&4), "{counts:?}");
        assert_eq!(counts.get("<item>"), Some(&2), "{counts:?}");
        assert_eq!(counts.get("other"), Some(&1), "{counts:?}");
        assert_eq!(counts.get("is_recording"), None);
        assert_eq!(counts.get("t"), None);
        assert_eq!(counts.get("helper"), None);
    }
}
