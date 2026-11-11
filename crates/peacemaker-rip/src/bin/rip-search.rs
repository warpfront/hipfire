//! `rip-search`: verified kernel-variant search over `.rip` scripts (CPU stages S0-S4).
//!
//! ```text
//! rip-search plan FILE.rip [--arch A] [--variant V]... [--json]
//! rip-search run  FILE.rip --cases DIR --out RUNDIR [--arch gfx1201] [--variant V]...
//!                 [--jobs N] [--mem-floor 0.25] [--reference-elf-sha256 HEX] [--limit N]
//! ```
//!
//! A script declares its search space with `const SEARCH_AXES` (see
//! `peacemaker_rip::search`). A candidate is one point of `VARIANTS x product(axes)` for
//! one arch; its source is the script with each axis const's literal spliced in place.
//! Per candidate:
//!
//! - S0 splice (`search::with_axes`);
//! - S1 `peacemaker_rip::check` (script `Err` is `refused`, any other error `invalid`);
//! - S2 native assemble, lift, emulator admission (`not-liftable`, `not-executable`);
//! - S3 `peacemaker_rip::certify` (`uncertified`);
//! - S4 every case of `--cases` (a flat directory of `run --dump` JSONs of the REFERENCE
//!   point, sorted by file name) emulated and compared byte for byte (`not-bitwise`).
//!
//! Passing S1-S4 is `certified`. The reference point (the declared const values) runs
//! through the same stages first and must certify, and with `--reference-elf-sha256`
//! must produce that ELF, else the run aborts with exit 2.
//!
//! Outputs under RUNDIR: `ledger.jsonl` (schema `rip-search.candidate/1`, sorted by
//! index, no timestamps), `summary.json`, `run.json` (the only wall-clock file),
//! `objects/<id>.co` and `sources/<id>.rip` (certified unique objects),
//! `manifest.tsv`, and `work/` (scratch; nothing is written under the system temp dir).
//! `candidate id = first 16 hex of sha256(arch 0x00 variant 0x00 candidate_source)`.
//!
//! Exit codes: 0 success, 1 I/O failure, 2 usage or gate failure.

use peacemaker_ir::emu::Emulator;
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::Arch;
use peacemaker_rip::features::sha256_hex;
use peacemaker_rip::run::Case;
use peacemaker_rip::search::{self, Axis};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage:
  rip-search plan FILE.rip [--arch A] [--variant V]... [--json]
  rip-search run FILE.rip --cases DIR --out RUNDIR [--arch gfx1201] [--variant V]...
                 [--jobs N (default 16)] [--mem-floor F (default 0.25)]
                 [--reference-elf-sha256 HEX] [--limit N]";

struct Opts {
    file: String,
    arch: Arch,
    variants: Vec<String>,
    json: bool,
    cases: Option<PathBuf>,
    out: Option<PathBuf>,
    jobs: usize,
    mem_floor: f64,
    ref_sha: Option<String>,
    limit: Option<usize>,
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        file: String::new(),
        arch: Arch::Gfx1201,
        variants: Vec::new(),
        json: false,
        cases: None,
        out: None,
        jobs: 16,
        mem_floor: 0.25,
        ref_sha: None,
        limit: None,
    };
    let mut it = args.iter();
    let mut file: Option<String> = None;
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let value = |it: &mut std::slice::Iter<String>| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--json" => o.json = true,
            "--arch" => {
                let v = value(&mut it)?;
                o.arch = peacemaker_rip::parse_arch(&v)?;
            }
            "--variant" => {
                let v = value(&mut it)?;
                for part in v.split(',').filter(|p| !p.is_empty()) {
                    if !o.variants.iter().any(|x| x == part) {
                        o.variants.push(part.to_string());
                    }
                }
            }
            "--cases" => o.cases = Some(PathBuf::from(value(&mut it)?)),
            "--out" => o.out = Some(PathBuf::from(value(&mut it)?)),
            "--jobs" => {
                let v = value(&mut it)?;
                o.jobs = v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| format!("--jobs: `{v}` is not a positive integer"))?;
            }
            "--mem-floor" => {
                let v = value(&mut it)?;
                o.mem_floor = v
                    .parse::<f64>()
                    .ok()
                    .filter(|f| (0.0..1.0).contains(f))
                    .ok_or_else(|| format!("--mem-floor: `{v}` is not a fraction in [0, 1)"))?;
            }
            "--reference-elf-sha256" => {
                let v = value(&mut it)?.to_ascii_lowercase();
                if v.len() != 64 || !v.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("--reference-elf-sha256 needs 64 hex digits".into());
                }
                o.ref_sha = Some(v);
            }
            "--limit" => {
                let v = value(&mut it)?;
                o.limit = Some(
                    v.parse::<usize>()
                        .map_err(|_| format!("--limit: `{v}` is not an integer"))?,
                );
            }
            f if f.starts_with("--") => return Err(format!("unknown option `{f}`")),
            _ => {
                if file.replace(a.clone()).is_some() {
                    return Err(format!("unexpected argument `{a}`"));
                }
            }
        }
    }
    o.file = file.ok_or("no FILE.rip given")?;
    Ok(o)
}

/// The script, its axes and the variants to search.
struct Space {
    source: String,
    name: String,
    axes: Vec<Axis>,
    variants: Vec<String>,
}

fn load_space(o: &Opts) -> Result<Space, String> {
    let source = std::fs::read_to_string(&o.file).map_err(|e| format!("{}: {e}", o.file))?;
    let name = o.file.clone();
    let axes = search::search_axes(&source, &name)?;
    let variants = if o.variants.is_empty() {
        peacemaker_rip::variants(&source, &name)?
    } else {
        o.variants.clone()
    };
    if variants.is_empty() {
        return Err(format!(
            "{name}: the script declares no variants (`const VARIANTS`); pass --variant"
        ));
    }
    Ok(Space {
        source,
        name,
        axes,
        variants,
    })
}

fn axis_json(a: &Axis) -> Value {
    json!({"name": a.name, "class": a.class, "ty": a.ty, "line": a.line,
           "reference": a.reference, "values": a.values})
}

fn plan(o: &Opts) -> u8 {
    let space = match load_space(o) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let Some(n) = search::point_count(&space.axes) else {
        eprintln!("error: {}: the axis product overflows", space.name);
        return 2;
    };
    let candidates = n.saturating_mul(space.variants.len());
    if o.json {
        let doc = json!({
            "schema": "rip-search.plan/1",
            "file": space.name,
            "source_sha256": sha256_hex(space.source.as_bytes()),
            "arch": o.arch.name(),
            "variants": space.variants,
            "axes": space.axes.iter().map(axis_json).collect::<Vec<_>>(),
            "points_per_variant": n,
            "candidates": candidates,
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap());
        return 0;
    }
    println!("file: {}", space.name);
    println!("arch: {}", o.arch.name());
    println!(
        "variants ({}): {}",
        space.variants.len(),
        space.variants.join(", ")
    );
    println!("axes ({}):", space.axes.len());
    for a in &space.axes {
        println!(
            "  {:<12} {:<6} class={:<10} line {:<5} reference={} values=[{}]",
            a.name,
            a.ty,
            a.class,
            a.line,
            a.reference,
            a.values.join(", ")
        );
    }
    println!("points per variant: {n}");
    println!("candidates: {candidates}");
    0
}

// ---------------------------------------------------------------------------
// Case corpus
// ---------------------------------------------------------------------------

fn load_cases(dir: &Path) -> Result<(Vec<(String, Case)>, String), String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(format!("{}: no case .json files", dir.display()));
    }
    let mut cases = Vec::new();
    let mut corpus: Vec<u8> = Vec::new();
    for name in names {
        let path = dir.join(&name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        corpus.extend_from_slice(name.as_bytes());
        corpus.push(0);
        corpus.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        corpus.extend_from_slice(&bytes);
        let case = serde_json::from_slice::<Value>(&bytes)
            .map_err(|e| e.to_string())
            .and_then(Case::parse)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        cases.push((name, case));
    }
    Ok((cases, sha256_hex(&corpus)))
}

// ---------------------------------------------------------------------------
// Candidate evaluation
// ---------------------------------------------------------------------------

struct Ctx {
    source: String,
    name: String,
    arch: Arch,
    cases: Vec<(String, Case)>,
    work: PathBuf,
}

#[derive(Clone)]
struct Rec {
    id: String,
    index: usize,
    variant: String,
    axes: Vec<(String, String)>,
    source_sha256: String,
    s_text_sha256: Option<String>,
    elf_sha256: Option<String>,
    dedup_of: Option<String>,
    resources: Option<peacemaker_rip::Resources>,
    stage: &'static str,
    verdict: &'static str,
    reason: Option<String>,
    m7_sha256: Option<String>,
    emu_outputs_sha256: Option<String>,
    /// The linked object and candidate source, kept only for certified candidates.
    elf: Option<Vec<u8>>,
    source: String,
}

impl Rec {
    fn new(index: usize, variant: &str, point: &[(String, String)]) -> Self {
        Rec {
            id: String::new(),
            index,
            variant: variant.to_string(),
            axes: point.to_vec(),
            source_sha256: String::new(),
            s_text_sha256: None,
            elf_sha256: None,
            dedup_of: None,
            resources: None,
            stage: "S0",
            verdict: "invalid",
            reason: None,
            m7_sha256: None,
            emu_outputs_sha256: None,
            elf: None,
            source: String::new(),
        }
    }

    fn fail(mut self, stage: &'static str, verdict: &'static str, reason: String) -> Self {
        self.stage = stage;
        self.verdict = verdict;
        self.reason = Some(reason);
        self
    }

    fn json(&self, arch: Arch, cases_sha: &str) -> Value {
        let axes: BTreeMap<&str, &str> = self
            .axes
            .iter()
            .map(|(a, v)| (a.as_str(), v.as_str()))
            .collect();
        let certified = self.verdict == "certified";
        json!({
            "schema": "rip-search.candidate/1",
            "id": self.id,
            "index": self.index,
            "arch": arch.name(),
            "variant": self.variant,
            "axes": axes,
            "source_sha256": self.source_sha256,
            "s_text_sha256": self.s_text_sha256,
            "elf_sha256": self.elf_sha256,
            "dedup_of": self.dedup_of,
            "resources": self.resources.as_ref().map(|r| json!({
                "vgprs": r.vgprs, "sgprs": r.sgprs, "lds": r.lds_bytes,
                "instructions": r.instructions,
            })),
            "stage_reached": self.stage,
            "verdict": self.verdict,
            "reason": self.reason,
            "exactness": certified.then_some("bitwise-evidenced"),
            "m7_sha256": self.m7_sha256,
            "cases_sha256": cases_sha,
            "emu_outputs_sha256": self.emu_outputs_sha256,
        })
    }
}

fn candidate_id(arch: Arch, variant: &str, source: &str) -> String {
    let mut buf = Vec::with_capacity(source.len() + variant.len() + 16);
    buf.extend_from_slice(arch.name().as_bytes());
    buf.push(0);
    buf.extend_from_slice(variant.as_bytes());
    buf.push(0);
    buf.extend_from_slice(source.as_bytes());
    sha256_hex(&buf)[..16].to_string()
}

fn evaluate(ctx: &Ctx, index: usize, variant: &str, point: &[(String, String)]) -> Rec {
    let arch = ctx.arch;
    let mut rec = Rec::new(index, variant, point);
    // S0
    let src = match search::with_axes(&ctx.source, &ctx.name, point) {
        Ok(s) => s,
        Err(e) => {
            rec.id = candidate_id(arch, variant, &format!("S0 {point:?}"));
            return rec.fail("S0", "invalid", e);
        }
    };
    rec.id = candidate_id(arch, variant, &src);
    rec.source_sha256 = sha256_hex(src.as_bytes());
    // S1
    let checked = peacemaker_rip::check(&src, &ctx.name, arch, variant);
    let Some(emitted) = checked.emitted else {
        let errors: Vec<&peacemaker_rip::diag::Diagnostic> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == peacemaker_rip::diag::Severity::Error)
            .collect();
        let refused = errors.iter().any(|d| d.code == "R0502");
        let reason = errors
            .iter()
            .map(|d| match d.span {
                Some(s) => format!("{}:{}:{}: {} {}", d.file, s.line, s.col, d.code, d.message),
                None => format!("{}: {} {}", d.file, d.code, d.message),
            })
            .collect::<Vec<_>>()
            .join("\n");
        let reason = if reason.is_empty() {
            "no emitted kernel".to_string()
        } else {
            reason
        };
        return rec.fail("S1", if refused { "refused" } else { "invalid" }, reason);
    };
    rec.resources = checked.resources;
    rec.s_text_sha256 = Some(sha256_hex(emitted.s_text.as_bytes()));
    // S2
    let symbol = match peacemaker_rip::symbol_of(&emitted) {
        Ok(s) => s,
        Err(e) => return rec.fail("S2", "not-liftable", e),
    };
    let elf = match peacemaker_isa::native::assemble(&emitted.s_text, arch) {
        Ok(e) => e,
        Err(e) => return rec.fail("S2", "not-liftable", format!("assemble: {e}")),
    };
    rec.elf_sha256 = Some(sha256_hex(&elf));
    let lifted = match peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    ) {
        Ok(l) => l,
        Err(e) => return rec.fail("S2", "not-liftable", format!("lift: {e:?}")),
    };
    let emulator = match Emulator::new(&lifted.program, &symbol) {
        Ok(e) => e,
        Err(e) => return rec.fail("S2", "not-executable", e),
    };
    // S3
    let dir = ctx.work.join(&rec.id);
    let certs = match peacemaker_rip::certify(
        std::slice::from_ref(&emitted),
        &emitted.s_text,
        arch,
        &rec.id,
        &dir,
    ) {
        Ok(c) => {
            let _ = std::fs::remove_dir_all(&dir);
            c
        }
        Err(e) => return rec.fail("S3", "uncertified", e),
    };
    let Some(cert) = certs.first() else {
        return rec.fail(
            "S3",
            "uncertified",
            "certification returned no certificate".into(),
        );
    };
    rec.m7_sha256 = Some(sha256_hex(cert.m7.to_string().as_bytes()));
    // S4
    let mut outputs: Vec<u8> = Vec::new();
    let mut failure: Option<String> = None;
    for (file, case) in &ctx.cases {
        let actual = match case.emulate(&emulator) {
            Ok(a) => a,
            Err(e) => {
                failure = Some(format!("{file}: emulator: {e}"));
                break;
            }
        };
        outputs.extend_from_slice(file.as_bytes());
        outputs.push(0);
        for buffer in &actual {
            outputs.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
            outputs.extend_from_slice(buffer);
        }
        let report = case.compare(&actual);
        if report["ok"] != json!(true) {
            let first = report["buffers"].as_object().and_then(|m| {
                m.iter().find(|(_, r)| {
                    r["changed"] == json!(true)
                        || !r["first_mismatch"].is_null()
                        || r["guard_ok"] == json!(false)
                })
            });
            failure = Some(match first {
                Some((name, r)) => format!("{file}: buffer `{name}`: {r}"),
                None => format!("{file}: {report}"),
            });
            break;
        }
    }
    rec.emu_outputs_sha256 = Some(sha256_hex(&outputs));
    if let Some(reason) = failure {
        return rec.fail("S4", "not-bitwise", reason);
    }
    rec.stage = "S4";
    rec.verdict = "certified";
    rec.elf = Some(elf);
    rec.source = src;
    rec
}

// ---------------------------------------------------------------------------
// Host guards and run plumbing
// ---------------------------------------------------------------------------

fn mem_available_fraction() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |key: &str| -> Option<f64> {
        text.lines().find_map(|l| {
            l.strip_prefix(key)?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse()
                .ok()
        })
    };
    Some(field("MemAvailable:")? / field("MemTotal:")?)
}

fn wait_for_memory(floor: f64) {
    while mem_available_fraction().is_some_and(|f| f < floor) {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn io(path: &Path, e: std::io::Error) -> String {
    format!("{}: {e}", path.display())
}

fn write_file(path: &Path, bytes: impl AsRef<[u8]>) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| io(path, e))
}

fn axes_text(axes: &[(String, String)]) -> String {
    axes.iter()
        .map(|(a, v)| format!("{a}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The manifest's `axes` column: one compact JSON object (string values), as the bench reads it.
fn axes_json(axes: &[(String, String)]) -> String {
    let mut m = serde_json::Map::new();
    for (a, v) in axes {
        m.insert(a.clone(), Value::String(v.clone()));
    }
    Value::Object(m).to_string()
}

enum Failure {
    /// Usage or gate failure: exit 2.
    Gate(String),
    /// I/O failure: exit 1.
    Io(String),
}

impl From<String> for Failure {
    fn from(s: String) -> Self {
        Failure::Io(s)
    }
}

fn run(o: &Opts) -> Result<(), Failure> {
    let started = unix_millis();
    let space = load_space(o).map_err(Failure::Gate)?;
    for a in &space.axes {
        if a.class != "schedule" {
            return Err(Failure::Gate(format!(
                "{}:{}: axis `{}` has exactness class `{}`; phase 1 searches only `schedule` axes",
                space.name, a.line, a.name, a.class
            )));
        }
    }
    let cases_dir = o
        .cases
        .as_ref()
        .ok_or_else(|| Failure::Gate("run needs --cases DIR".into()))?;
    let out = o
        .out
        .as_ref()
        .ok_or_else(|| Failure::Gate("run needs --out RUNDIR".into()))?;
    if o.ref_sha.is_some() && space.variants.len() != 1 {
        return Err(Failure::Gate(
            "--reference-elf-sha256 needs exactly one variant".into(),
        ));
    }
    let (cases, cases_sha) = load_cases(cases_dir).map_err(Failure::Gate)?;
    let pts = search::points(&space.axes);
    let npoints = pts.len();
    let total = npoints
        .checked_mul(space.variants.len())
        .ok_or_else(|| Failure::Gate("the candidate count overflows".into()))?;
    let enumerated = o.limit.map_or(total, |l| l.min(total));

    for sub in ["objects", "sources", "work"] {
        let d = out.join(sub);
        std::fs::create_dir_all(&d).map_err(|e| io(&d, e))?;
    }
    let ctx = Ctx {
        source: space.source.clone(),
        name: space.name.clone(),
        arch: o.arch,
        cases,
        work: out.join("work"),
    };

    // Twin gate: the reference point of every variant must certify.
    let reference = search::reference_point(&space.axes);
    let ref_pos = pts.iter().position(|p| *p == reference).ok_or_else(|| {
        Failure::Gate("internal: the reference point is not in the enumeration".into())
    })?;
    let mut reference_recs: HashMap<usize, Rec> = HashMap::new();
    for (vi, variant) in space.variants.iter().enumerate() {
        let index = vi * npoints + ref_pos;
        let rec = evaluate(&ctx, index, variant, &reference);
        if rec.verdict != "certified" {
            return Err(Failure::Gate(format!(
                "reference point of variant `{variant}` ({}) is `{}` at {}: {}",
                axes_text(&reference),
                rec.verdict,
                rec.stage,
                rec.reason.as_deref().unwrap_or("")
            )));
        }
        if let (Some(want), Some(got)) = (&o.ref_sha, &rec.elf_sha256) {
            if want != got {
                return Err(Failure::Gate(format!(
                    "twin gate: reference ELF sha256 {got} differs from --reference-elf-sha256 {want}"
                )));
            }
        }
        reference_recs.insert(index, rec);
    }

    // The pool.
    let jobs = o.jobs.min(enumerated.max(1));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Rec>> = Mutex::new(Vec::with_capacity(enumerated));
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::new();
        for n in 0..jobs {
            let worker = || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= enumerated {
                    break;
                }
                let rec = match reference_recs.get(&i) {
                    Some(r) => r.clone(),
                    None => {
                        wait_for_memory(o.mem_floor);
                        let variant = &space.variants[i / npoints];
                        let point = &pts[i % npoints];
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            evaluate(&ctx, i, variant, point)
                        }))
                        .unwrap_or_else(|_| {
                            let mut r = Rec::new(i, variant, point);
                            r.id = candidate_id(ctx.arch, variant, &format!("panic {i}"));
                            r.fail("S0", "internal-error", "the evaluator panicked".into())
                        })
                    }
                };
                results.lock().unwrap().push(rec);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 25 == 0 || d == enumerated {
                    eprintln!("[rip-search] {d}/{enumerated}");
                }
            };
            let h = std::thread::Builder::new()
                .name(format!("rip-search-{n}"))
                .stack_size(256 << 20)
                .spawn_scoped(scope, worker)
                .map_err(|e| format!("cannot start worker {n}: {e}"))?;
            handles.push(h);
        }
        for h in handles {
            h.join()
                .map_err(|_| "a worker thread panicked".to_string())?;
        }
        Ok(())
    })?;
    let mut recs = results.into_inner().unwrap();
    recs.sort_by_key(|r| r.index);

    // Dedup by linked object: later candidates with an already-certified ELF.
    let mut seen: HashMap<String, String> = HashMap::new();
    for r in recs.iter_mut() {
        if r.verdict != "certified" {
            continue;
        }
        let sha = r.elf_sha256.clone().unwrap_or_default();
        match seen.get(&sha) {
            Some(first) => {
                r.dedup_of = Some(first.clone());
                r.elf = None;
                r.source = String::new();
            }
            None => {
                seen.insert(sha, r.id.clone());
            }
        }
    }

    // Outputs.
    let mut ledger = String::new();
    let mut manifest = String::from("id\taxes\telf_sha256\tvgprs\tsgprs\tinstructions\n");
    let mut by_verdict: BTreeMap<&str, usize> = BTreeMap::new();
    let mut certified = 0usize;
    let mut unique = 0usize;
    for r in &recs {
        ledger.push_str(&serde_json::to_string(&r.json(o.arch, &cases_sha)).unwrap());
        ledger.push('\n');
        *by_verdict.entry(r.verdict).or_default() += 1;
        if r.verdict == "certified" {
            certified += 1;
        }
        if let (Some(elf), true) = (&r.elf, r.dedup_of.is_none()) {
            unique += 1;
            write_file(&out.join("objects").join(format!("{}.co", r.id)), elf)?;
            write_file(
                &out.join("sources").join(format!("{}.rip", r.id)),
                &r.source,
            )?;
            let (v, s, n) = r
                .resources
                .as_ref()
                .map_or((0, 0, 0), |x| (x.vgprs, x.sgprs, x.instructions));
            manifest.push_str(&format!(
                "{}\t{}\t{}\t{v}\t{s}\t{n}\n",
                r.id,
                axes_json(&r.axes),
                r.elf_sha256.as_deref().unwrap_or("")
            ));
        }
    }
    write_file(&out.join("ledger.jsonl"), ledger)?;
    write_file(&out.join("manifest.tsv"), manifest)?;
    let mut ref_ids = BTreeMap::new();
    for (vi, variant) in space.variants.iter().enumerate() {
        let r = &reference_recs[&(vi * npoints + ref_pos)];
        ref_ids.insert(
            variant.clone(),
            json!({"id": r.id, "elf_sha256": r.elf_sha256}),
        );
    }
    let summary = json!({
        "schema": "rip-search.summary/1",
        "file": space.name,
        "source_sha256": sha256_hex(space.source.as_bytes()),
        "arch": o.arch.name(),
        "variants": space.variants,
        "space": total,
        "enumerated": enumerated,
        "limit": o.limit,
        "by_verdict": by_verdict,
        "certified": certified,
        "unique_objects": unique,
        "reference_id": (space.variants.len() == 1).then(|| ref_ids[&space.variants[0]]["id"].clone()),
        "references": ref_ids,
        "cases": ctx.cases.len(),
        "cases_sha256": cases_sha,
    });
    write_file(
        &out.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )?;
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(|b| sha256_hex(&b));
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let run_doc = json!({
        "schema": "rip-search.run/1",
        "argv": std::env::args().collect::<Vec<_>>(),
        "binary_sha256": exe,
        "start_unix_ms": started,
        "end_unix_ms": unix_millis(),
        "jobs": jobs,
        "mem_floor": o.mem_floor,
        "host": host,
    });
    write_file(
        &out.join("run.json"),
        serde_json::to_string_pretty(&run_doc).unwrap() + "\n",
    )?;
    let _ = std::fs::remove_dir(out.join("work"));
    eprintln!(
        "[rip-search] enumerated {enumerated}, certified {certified}, unique objects {unique}; {}",
        by_verdict
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        eprintln!("error: no command given\n\n{USAGE}");
        return ExitCode::from(2);
    };
    if matches!(cmd.as_str(), "help" | "--help" | "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if cmd != "plan" && cmd != "run" {
        eprintln!("error: unknown command `{cmd}`\n\n{USAGE}");
        return ExitCode::from(2);
    }
    let opts = match parse_opts(rest) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if cmd == "plan" {
        return ExitCode::from(plan(&opts));
    }
    match run(&opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Gate(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
        Err(Failure::Io(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
