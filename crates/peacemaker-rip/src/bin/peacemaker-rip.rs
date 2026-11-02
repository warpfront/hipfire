//! `peacemaker-rip`: check, run, format, explain and compile `.rip` kernel scripts without cargo.
//!
//! ```text
//! peacemaker-rip check FILE [--arch LIST|all] [--variant LIST] [--json]
//! peacemaker-rip features FILE [--arch LIST|all] [--variant LIST] [--routes FILE] [--json]
//! peacemaker-rip run FILE [--case FILE.json] [--arch LIST|all] [--variant LIST] [--json] [--dump OUT]
//! peacemaker-rip build [--certify] [--json] SOURCE.rip ARCH VARIANT OUTPUT.s
//! peacemaker-rip build [--certify] [--json] --module SOURCE.rip ARCH VARIANT[,VARIANT..] MODULE_NAME OUTPUT.s
//! peacemaker-rip fmt [--check] FILE...
//! peacemaker-rip explain CODE | --list
//! peacemaker-rip help
//! ```
//!
//! - `check` compiles every (arch, variant) of FILE and reports located, coded
//!   diagnostics (errors and warnings) plus one summary line per target.
//!   `--arch` is a comma list of `gfx1100`, `gfx1151`, `gfx1201`, or `all`
//!   (default `gfx1151,gfx1201`). `--variant` is a comma list (default: the
//!   variants declared by the script's `enum Kind`). `--json` prints one JSON
//!   document on stdout instead of the human text. Exit 0 when there are no
//!   errors (warnings allowed), 1 on errors, 2 on a usage error.
//! - `features` reports shared check resources, linked-object schedule features,
//!   emulator coverage, and SHA256 provenance without a GPU. `--routes` passes
//!   a cost-lint routes JSON document unchanged; failures are per-target errors.
//! - `run` compiles every (arch, variant) of FILE, executes the declared grid
//!   on the CPU emulator and checks every buffer against `--case` (default:
//!   FILE with its extension replaced by `.json`). No GPU or ROCm tools are
//!   involved. Target selection (`--arch`/`--variant`) works exactly like
//!   `check`. `--json` prints the flat schema-1 report for one target, or
//!   `{schema:1,ok,results:[reports]}` for several. `--dump OUT` needs exactly
//!   one target and writes the behavior-capture case (behavior, not
//!   correctness) even when expectations fail. Exit 0 iff every target passes,
//!   1 on failures, 2 on a usage error.
//! - `build` writes the assembly. `ARCH` is `gfx1151`, `gfx1201`, `gfx1100` or
//!   `both` (gfx1151 and gfx1201; `OUTPUT` then gets the arch before its
//!   extension: `out.gfx1151.s`). `--certify` assembles and links the emitted
//!   code object into `$RIP_CERT_DIR` (default: a temporary directory), then
//!   runs the existing passes on it: wait-ledger replay and M7 (byte-exact
//!   lift, no obligations). The bundler's host target identity is the
//!   toolchain default; to reproduce a committed `.hxaco` recipe configure the
//!   library's `Toolchain` instead. `--json` prints the diagnostics (or the
//!   outputs) as one JSON document on stdout. The legacy form
//!   `peacemaker-rip [--certify] [--module] SOURCE ARCH ...` is `build`.
//! - `fmt` formats FILEs in place with rustfmt (token stream verified
//!   unchanged); `--check` prints the files that would change and exits 1.
//! - `explain CODE` prints the long-form explanation of a diagnostic code;
//!   `explain --list` lists every code with its title.

use peacemaker_isa::Arch;
use peacemaker_rip::diag::{self, Diagnostic, Severity};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage:
  peacemaker-rip check FILE [--arch LIST|all] [--variant LIST] [--json]
  peacemaker-rip features FILE [--arch LIST|all] [--variant LIST] [--routes FILE] [--json]
  peacemaker-rip run FILE [--case FILE.json] [--arch LIST|all] [--variant LIST] [--json] [--dump OUT]
  peacemaker-rip build [--certify] [--json] SOURCE.rip ARCH VARIANT OUTPUT.s
  peacemaker-rip build [--certify] [--json] --module SOURCE.rip ARCH VARIANT[,VARIANT..] MODULE_NAME OUTPUT.s
  peacemaker-rip fmt [--check] FILE...
  peacemaker-rip explain CODE | --list
  peacemaker-rip help

check:   compile every (arch, variant) and report diagnostics; exit 0 ok (warnings allowed), 1 errors, 2 usage.
         --arch LIST|all   comma list of gfx1100,gfx1151,gfx1201 (default: the script's `const ARCHES`, else gfx1151,gfx1201; all = declared ARCHES, else all three)
         --variant LIST    comma list (default: the script's `const VARIANTS`, else its `enum Kind`)
         --json            one JSON document on stdout
features: report offline resource, schedule and emulator facts; target selection is the same as check.
         --routes FILE     cost-lint routes JSON keyed by exact kernel symbol
run:    execute every (arch, variant) on the CPU emulator and check buffers; exit 0 iff all pass, 1 failures, 2 usage.
         --case FILE       case JSON (default: FILE with its extension replaced by `.json`)
         --arch LIST|all   same target selection as check
         --variant LIST    comma list (default: the script's `const VARIANTS`, else its `enum Kind`)
         --json            one target prints the flat schema-1 report; several print {schema:1,ok,results:[reports]}
         --dump OUT        write the behavior-capture case; exactly one target required, written even when expectations fail
build:   write assembly; ARCH is gfx1151, gfx1201, gfx1100 or both. --certify assembles, links and certifies.
         The legacy form `peacemaker-rip [--certify] [--module] SOURCE ARCH ...` is `build`.
fmt:     format in place with rustfmt; --check lists files that would change and exits 1.
explain: print the long-form explanation of a diagnostic code; --list lists all codes.";

fn usage_error(msg: &str) -> u8 {
    eprintln!("error: {msg}\n\n{USAGE}");
    2
}

/// Parse `--arch`: a comma list of arches, `both` (gfx1151, gfx1201) or
/// `all`. Bare `all` expands to `declared` when the script declares `const
/// ARCHES`, else to every arch; an explicit list never consults the const.
fn arches(a: &str, declared: Option<&[Arch]>) -> Result<Vec<Arch>, String> {
    match a {
        "both" => return Ok(vec![Arch::Gfx1151, Arch::Gfx1201]),
        "all" => {
            return Ok(match declared {
                Some(ds) => ds.to_vec(),
                None => peacemaker_rip::ALL_ARCHES.to_vec(),
            })
        }
        _ => {}
    }
    let mut out = Vec::new();
    for part in a.split(',') {
        let arch = part
            .parse::<Arch>()
            .map_err(|_| format!("unsupported architecture {part}"))?;
        if !out.contains(&arch) {
            out.push(arch);
        }
    }
    Ok(out)
}

fn out_path(out: &str, arch: Arch, many: bool) -> PathBuf {
    let p = Path::new(out);
    if !many {
        return p.to_path_buf();
    }
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("s");
    p.with_file_name(format!("{stem}.{}.{ext}", arch.name()))
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

/// `--flag VALUE` or `--flag=VALUE`.
fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let prefix = format!("{flag}=");
    let Some(i) = args
        .iter()
        .position(|a| a == flag || a.starts_with(&prefix))
    else {
        return Ok(None);
    };
    let arg = args.remove(i);
    if let Some(v) = arg.strip_prefix(&prefix) {
        return Ok(Some(v.to_string()));
    }
    if i < args.len() {
        return Ok(Some(args.remove(i)));
    }
    Err(format!("`{flag}` needs a value"))
}

fn reject_unknown_options(args: &[String]) -> Result<(), String> {
    match args.iter().find(|a| a.starts_with("--")) {
        Some(a) => Err(format!("unknown option `{a}`")),
        None => Ok(()),
    }
}

fn tool_error(file: &str, message: String) -> Diagnostic {
    Diagnostic {
        code: "R0901",
        severity: Severity::Error,
        message,
        file: file.to_string(),
        span: None,
        help: None,
        suggestion: None,
        notes: Vec::new(),
    }
}

fn print_diagnostic(d: &Diagnostic, source: &str) {
    eprintln!("{}\n", d.render(source).trim_end());
}

fn print_json(v: &serde_json::Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(v).expect("a JSON value serializes")
    );
}

fn feature_failure(
    file: &str,
    source: Option<&str>,
    routes_supplied: bool,
    json: bool,
    stage: &str,
    message: &str,
) -> u8 {
    if json {
        print_json(&serde_json::json!({
            "schema": 1, "file": file,
            "source_sha256": source.map(peacemaker_rip::features::source_sha256), "ok": false,
            "routes_supplied": routes_supplied, "targets": {},
            "error": {"stage": stage, "message": message},
        }));
    } else {
        eprintln!("error {stage}: {message}");
    }
    1
}

fn print_features(report: &serde_json::Value) {
    let Some(targets) = report["targets"].as_object() else {
        return;
    };
    for (key, target) in targets {
        println!("{key}:");
        if let Some(r) = target["resources"].as_object() {
            println!(
                "  {} instructions, {} VGPRs, {} SGPRs, LDS {} B, wave {}",
                r["instructions"], r["vgprs"], r["sgprs"], r["lds_bytes"], r["wave_size"],
            );
        }
        println!(
            "  sha256 source={} assembly={}",
            report["source_sha256"].as_str().unwrap_or("unavailable"),
            target["assembly_sha256"].as_str().unwrap_or("unavailable"),
        );
        if let Some(error) = target["error"].as_object() {
            println!(
                "  error {}: {}",
                error["stage"].as_str().unwrap_or("unknown"),
                error["message"].as_str().unwrap_or("unknown"),
            );
            continue;
        }
        let routes = report["routes_supplied"].as_bool() == Some(true);
        println!("  cost: routes_supplied={routes}, drain_pressure=null (requires a pair)");
        if let Some(kernels) = target["cost"]["kernels"].as_array() {
            for kernel in kernels {
                if let Some(loops) = kernel["loops"].as_array() {
                    for (index, cost) in loops.iter().enumerate() {
                        println!(
                            "    loop {index}: wmma_per_trip={}, valu_per_wmma={}, wait_depth_before_wmma={}, independent_chains={}, padding_fraction={}, wmma_per_k128={}, prefetch_distance_epochs={}",
                            cost["wmma_per_trip"], cost["valu_per_wmma"],
                            cost["wait_depth_before_wmma"], cost["independent_chains"],
                            cost["padding_fraction"], cost["wmma_per_k128"],
                            cost["prefetch_distance_epochs"],
                        );
                    }
                }
            }
        }
        let coverage = &target["emulator"];
        let unmodeled = coverage["unmodeled"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|entry| {
                format!(
                    "{}={}",
                    entry["mnemonic"].as_str().unwrap_or("?"),
                    entry["count"]
                )
            })
            .collect::<Vec<_>>();
        println!(
            "  emulator: modeled={}/{}, unmodeled={}{}",
            coverage["modeled"],
            coverage["instructions"],
            if unmodeled.is_empty() {
                "none".into()
            } else {
                unmodeled.join(", ")
            },
            if unmodeled.is_empty() {
                ""
            } else {
                " (CPU execution refused)"
            },
        );
    }
}

// ---------------------------------------------------------------- check

fn inspect(mut args: Vec<String>, feature_mode: bool) -> u8 {
    let json = take_flag(&mut args, "--json");
    let arch_arg = match take_value(&mut args, "--arch") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    let variant_arg = match take_value(&mut args, "--variant") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    let routes_arg = if feature_mode {
        match take_value(&mut args, "--routes") {
            Ok(v) => v,
            Err(e) => return usage_error(&e),
        }
    } else {
        None
    };
    if let Err(e) = reject_unknown_options(&args) {
        return usage_error(&e);
    }
    if args.len() != 1 {
        return usage_error(if feature_mode {
            "`features` takes exactly one FILE"
        } else {
            "`check` takes exactly one FILE"
        });
    }
    let file = args.remove(0);
    let source = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            let d = tool_error(&file, format!("{file}: {e}"));
            if feature_mode {
                return feature_failure(
                    &file,
                    None,
                    routes_arg.is_some(),
                    json,
                    "source",
                    &d.message,
                );
            }
            if json {
                print_json(
                    &serde_json::json!({"file": file, "ok": false, "results": [], "diagnostics": [d.to_json()]}),
                );
            } else {
                print_diagnostic(&d, "");
            }
            return 1;
        }
    };
    // Discovery runs over the source twice (variants, arches) so a failure
    // of either is reported with its own location before any compile.
    let failed = |raw: String| {
        if feature_mode {
            return feature_failure(
                &file,
                Some(&source),
                routes_arg.is_some(),
                json,
                "discovery",
                &raw,
            );
        }
        let d = Diagnostic::from_raw(&raw, &file, &source, Severity::Error);
        if json {
            print_json(
                &serde_json::json!({"file": file, "ok": false, "results": [], "diagnostics": [d.to_json()]}),
            );
        } else {
            print_diagnostic(&d, &source);
        }
        1
    };
    let variants: Vec<String> = match &variant_arg {
        Some(list) => list
            .split(',')
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .collect(),
        None => match peacemaker_rip::variants(&source, &file) {
            Ok(v) => v,
            Err(raw) => return failed(raw),
        },
    };
    if variants.is_empty() {
        return usage_error(&format!(
            "{file} declares no `VARIANTS` (nor `enum Kind` variants); pass `--variant NAME[,NAME..]`"
        ));
    }
    // An explicit `--arch` list wins over the script; without one the
    // declared `ARCHES` (when present and well-formed) is the target list.
    let declared: Option<Vec<Arch>> = match peacemaker_rip::arches(&source, &file) {
        Ok(None) => None,
        Ok(Some(names)) => {
            let mut ds = Vec::new();
            for n in &names {
                match peacemaker_rip::parse_arch(n) {
                    Ok(a) => {
                        if !ds.contains(&a) {
                            ds.push(a);
                        }
                    }
                    Err(e) => return usage_error(&format!("{file}: const ARCHES: {e}")),
                }
            }
            Some(ds)
        }
        Err(raw) => return failed(raw),
    };
    let arches = match &arch_arg {
        Some(list) => match arches(list, declared.as_deref()) {
            Ok(a) => a,
            Err(e) => return usage_error(&format!("{file}: {e}")),
        },
        None => declared.unwrap_or_else(|| peacemaker_rip::DEFAULT_ARCHES.to_vec()),
    };
    if arches.is_empty() {
        return usage_error(&format!(
            "{file} declares an empty `ARCHES`; pass `--arch NAME[,NAME..]`"
        ));
    }
    if feature_mode {
        let routes = match routes_arg {
            Some(path) => match std::fs::read_to_string(&path) {
                Ok(text) => Some(text),
                Err(e) => {
                    return feature_failure(
                        &file,
                        Some(&source),
                        true,
                        json,
                        "routes",
                        &format!("{path}: {e}"),
                    )
                }
            },
            None => None,
        };
        let report = peacemaker_rip::features::collect(
            &source,
            &file,
            &arches,
            &variants,
            routes.as_deref(),
        );
        let ok = report["ok"].as_bool() == Some(true);
        if json {
            print_json(&report);
        } else {
            print_features(&report);
        }
        return u8::from(!ok);
    }

    let mut all_ok = true;
    let mut results = Vec::new();
    for arch in &arches {
        for variant in &variants {
            let checked = peacemaker_rip::check(&source, &file, *arch, variant);
            let errors = checked
                .diagnostics
                .iter()
                .filter(|d| d.severity == Severity::Error)
                .count();
            let warnings = checked.diagnostics.len() - errors;
            let ok = errors == 0;
            all_ok &= ok;
            if json {
                let resources = checked.resources.as_ref().map(|r| {
                    serde_json::json!({"vgprs": r.vgprs, "sgprs": r.sgprs, "lds_bytes": r.lds_bytes, "instructions": r.instructions})
                });
                results.push(serde_json::json!({
                    "arch": arch.name(),
                    "variant": variant,
                    "ok": ok,
                    "resources": resources,
                    "diagnostics": checked.diagnostics.iter().map(Diagnostic::to_json).collect::<Vec<_>>(),
                }));
                continue;
            }
            for d in &checked.diagnostics {
                print_diagnostic(d, &source);
            }
            if ok {
                let r = checked.resources.as_ref();
                let mut line = match r {
                    Some(r) => format!(
                        "ok {}/{variant}: {} instructions, {} VGPRs, {} SGPRs, LDS {} B",
                        arch.name(),
                        r.instructions,
                        r.vgprs,
                        r.sgprs,
                        r.lds_bytes
                    ),
                    None => format!("ok {}/{variant}", arch.name()),
                };
                if warnings > 0 {
                    line.push_str(&format!(
                        ", {warnings} warning{}",
                        if warnings == 1 { "" } else { "s" }
                    ));
                }
                eprintln!("{line}");
            } else {
                eprintln!(
                    "error {}/{variant}: {errors} error{}",
                    arch.name(),
                    if errors == 1 { "" } else { "s" }
                );
            }
        }
    }
    if json {
        print_json(&serde_json::json!({"file": file, "ok": all_ok, "results": results}));
    }
    if all_ok {
        0
    } else {
        1
    }
}

// ---------------------------------------------------------------- build

struct Built {
    arch: Arch,
    path: PathBuf,
    kernels: usize,
}

struct BuildArgs {
    certify: bool,
    json: bool,
    module: bool,
    args: Vec<String>,
}

/// Diagnostics for a failed build: what `check` reports for the variants up to the failing one
/// (errors and warnings, with the codes `check` assigns), else the bare error string classified.
fn failure_diagnostics(
    source: &str,
    file: &str,
    arch: Arch,
    variants: &[&str],
    raw: &str,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for variant in variants {
        let checked = peacemaker_rip::check(source, file, arch, variant);
        let failed = checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error);
        out.extend(checked.diagnostics);
        if failed {
            return out;
        }
    }
    vec![Diagnostic::from_raw(raw, file, source, Severity::Error)]
}

fn build_one(
    b: &BuildArgs,
    source: &str,
    arch: Arch,
    many: bool,
) -> Result<Built, Vec<Diagnostic>> {
    let file = &b.args[0];
    let want = if b.module { 5 } else { 4 };
    let (emitted, text, name) = if b.module {
        let variants: Vec<&str> = b.args[2].split(',').collect();
        let (emitted, text, _) =
            peacemaker_rip::module_named(source, file, arch, &variants, &b.args[3])
                .map_err(|e| failure_diagnostics(source, file, arch, &variants, &e))?;
        (emitted, text, b.args[3].clone())
    } else {
        let variants = [b.args[2].as_str()];
        let e = peacemaker_rip::compile_named(source, file, arch, variants[0])
            .map_err(|e| failure_diagnostics(source, file, arch, &variants, &e))?;
        let text = e.s_text.clone();
        let name = peacemaker_rip::symbol_of(&e)
            .map_err(|e| vec![Diagnostic::from_raw(&e, file, source, Severity::Error)])?;
        (vec![e], text, name)
    };
    let out = out_path(&b.args[want - 1], arch, many);
    std::fs::write(&out, &text)
        .map_err(|e| vec![tool_error(file, format!("{}: {e}", out.display()))])?;
    eprintln!(
        "{}: {} kernel(s) for {} -> {}",
        file,
        emitted.len(),
        arch.name(),
        out.display()
    );
    if b.certify {
        #[cfg(feature = "certify")]
        {
            let dir = std::env::var_os("RIP_CERT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::temp_dir().join(format!(
                        "rip-cert-{}-{}",
                        std::process::id(),
                        arch.name()
                    ))
                });
            let certs =
                peacemaker_rip::certify(&emitted, &text, arch, &name, &dir).map_err(|e| {
                    let mut d = Diagnostic::from_raw(&e, file, source, Severity::Error);
                    d.code = "R0701";
                    vec![d]
                })?;
            for c in certs {
                eprintln!(
                    "certified {} ({}): lift {}, obligations {}",
                    c.symbol,
                    arch.name(),
                    c.m7["lift"],
                    c.m7["obligations"]
                );
            }
        }
        #[cfg(not(feature = "certify"))]
        {
            let _ = &name;
            return Err(vec![tool_error(
                file,
                "built without the `certify` feature".into(),
            )]);
        }
    }
    Ok(Built {
        arch,
        path: out,
        kernels: emitted.len(),
    })
}

fn build(mut args: Vec<String>) -> u8 {
    let certify = take_flag(&mut args, "--certify");
    let module = take_flag(&mut args, "--module");
    let json = take_flag(&mut args, "--json");
    if let Err(e) = reject_unknown_options(&args) {
        return usage_error(&e);
    }
    let want = if module { 5 } else { 4 };
    if args.len() != want {
        return usage_error("bad arguments");
    }
    let b = BuildArgs {
        certify,
        json,
        module,
        args,
    };
    let file = b.args[0].clone();
    let source = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            return report_build_failure(
                &b,
                &file,
                "",
                vec![tool_error(&file, format!("{file}: {e}"))],
            )
        }
    };
    let arch_list = match arches(&b.args[1], None) {
        Ok(a) => a,
        Err(e) => return report_build_failure(&b, &file, &source, vec![tool_error(&file, e)]),
    };
    let many = arch_list.len() > 1;
    let mut built = Vec::new();
    for arch in arch_list {
        match build_one(&b, &source, arch, many) {
            Ok(o) => built.push(o),
            Err(ds) => return report_build_failure(&b, &file, &source, ds),
        }
    }
    if b.json {
        let outputs: Vec<_> = built.iter().map(|o| serde_json::json!({"arch": o.arch.name(), "path": o.path.display().to_string(), "kernels": o.kernels})).collect();
        print_json(
            &serde_json::json!({"file": file, "ok": true, "outputs": outputs, "diagnostics": []}),
        );
    }
    0
}

fn report_build_failure(
    b: &BuildArgs,
    file: &str,
    source: &str,
    diagnostics: Vec<Diagnostic>,
) -> u8 {
    if b.json {
        print_json(
            &serde_json::json!({"file": file, "ok": false, "outputs": [], "diagnostics": diagnostics.iter().map(Diagnostic::to_json).collect::<Vec<_>>()}),
        );
    } else {
        for d in &diagnostics {
            print_diagnostic(d, source);
        }
    }
    1
}
// ---------------------------------------------------------------- run

fn run_json_failure(file: &str, stage: &str, message: String) {
    print_json(&peacemaker_rip::run::failure(
        file, None, None, None, stage, &message,
    ));
}

fn default_case_path(file: &str) -> String {
    let mut path = PathBuf::from(file);
    path.set_extension("json");
    path.to_string_lossy().into_owned()
}

fn print_report(report: &serde_json::Value) {
    let file = report["file"].as_str().unwrap_or("?");
    let arch = report["arch"].as_str().unwrap_or("?");
    let variant = report["variant"].as_str().unwrap_or("?");
    let case = report["case"].as_str().unwrap_or("?");
    let prefix = format!("{file} {arch}/{variant} case {case}");
    if let Some(buffers) = report["buffers"].as_object() {
        for (name, entry) in buffers {
            match entry["kind"].as_str() {
                Some("input") => {
                    let changed = entry["changed"].as_bool().unwrap_or(false);
                    if changed {
                        match entry.get("first_mismatch") {
                            Some(offset) => {
                                println!("{prefix} input {name}: changed at byte {offset}")
                            }
                            None => println!("{prefix} input {name}: changed"),
                        }
                    } else {
                        println!("{prefix} input {name}: unchanged");
                    }
                }
                Some("output") => {
                    let mismatch = entry.get("first_mismatch");
                    let guard_ok = entry["guard_ok"].as_bool().unwrap_or(true);
                    let mut detail = match mismatch {
                        Some(index) => format!("first mismatch at element {index}"),
                        None => "matches".to_string(),
                    };
                    if !entry["max_abs_diff"].is_null() {
                        detail.push_str(&format!(" max_abs_diff={}", entry["max_abs_diff"]));
                    }
                    let guard = if guard_ok {
                        "guard ok"
                    } else {
                        "guard damaged"
                    };
                    println!("{prefix} output {name}: {detail}, {guard}");
                }
                _ => println!("{prefix} buffer {name}: {entry}"),
            }
        }
    }
    if let Some(error) = report.get("error").and_then(|e| e.as_object()) {
        println!(
            "{prefix}: failed at {}: {}",
            error["stage"].as_str().unwrap_or("unknown"),
            error["message"].as_str().unwrap_or("unknown")
        );
    } else if report["ok"].as_bool() == Some(true) {
        println!("{prefix}: pass");
    } else {
        println!("{prefix}: failed");
    }
}

fn dump_error(stage: &str, message: String, json: bool, file: &str) -> u8 {
    if json {
        print_json(&peacemaker_rip::run::failure(
            file, None, None, None, "dump", &message,
        ));
    } else {
        eprintln!("error {stage}: {message}");
    }
    1
}

fn run(mut args: Vec<String>) -> u8 {
    let json = take_flag(&mut args, "--json");
    let arch_arg = match take_value(&mut args, "--arch") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    let variant_arg = match take_value(&mut args, "--variant") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    let case_arg = match take_value(&mut args, "--case") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    let dump_arg = match take_value(&mut args, "--dump") {
        Ok(v) => v,
        Err(e) => return usage_error(&e),
    };
    if let Err(e) = reject_unknown_options(&args) {
        return usage_error(&e);
    }
    if args.len() != 1 {
        return usage_error("`run` takes exactly one FILE");
    }
    let file = args.remove(0);
    let source = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            let message = format!("{file}: {e}");
            if json {
                run_json_failure(&file, "source", message);
            } else {
                eprintln!("error source: {message}");
            }
            return 1;
        }
    };
    let failed = |stage: &str, raw: String| {
        if json {
            run_json_failure(&file, stage, raw);
        } else {
            let d = Diagnostic::from_raw(&raw, &file, &source, Severity::Error);
            print_diagnostic(&d, &source);
        }
        1
    };
    let variants: Vec<String> = match &variant_arg {
        Some(list) => list
            .split(',')
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .collect(),
        None => match peacemaker_rip::variants(&source, &file) {
            Ok(v) => v,
            Err(raw) => return failed("discovery", raw),
        },
    };
    if variants.is_empty() {
        return usage_error(&format!(
            "{file} declares no `VARIANTS` (nor `enum Kind` variants); pass `--variant NAME[,NAME..]`"
        ));
    }
    let declared: Option<Vec<Arch>> = match peacemaker_rip::arches(&source, &file) {
        Ok(None) => None,
        Ok(Some(names)) => {
            let mut ds = Vec::new();
            for n in &names {
                match peacemaker_rip::parse_arch(n) {
                    Ok(a) => {
                        if !ds.contains(&a) {
                            ds.push(a);
                        }
                    }
                    Err(e) => return usage_error(&format!("{file}: const ARCHES: {e}")),
                }
            }
            Some(ds)
        }
        Err(raw) => return failed("discovery", raw),
    };
    let arch_list = match &arch_arg {
        Some(list) => match arches(list, declared.as_deref()) {
            Ok(a) => a,
            Err(e) => return usage_error(&format!("{file}: {e}")),
        },
        None => declared.unwrap_or_else(|| peacemaker_rip::DEFAULT_ARCHES.to_vec()),
    };
    if arch_list.is_empty() {
        return usage_error(&format!(
            "{file} declares an empty `ARCHES`; pass `--arch NAME[,NAME..]`"
        ));
    }
    let targets: Vec<(Arch, &str)> = arch_list
        .iter()
        .flat_map(|a| variants.iter().map(|v| (*a, v.as_str())))
        .collect();
    let case_path = match case_arg {
        Some(path) => path,
        None => default_case_path(&file),
    };
    let case_json = match std::fs::read_to_string(&case_path) {
        Ok(s) => s,
        Err(e) => {
            let message = format!("{case_path}: {e}");
            if json {
                run_json_failure(&file, "case", message);
            } else {
                eprintln!("error case: {message}");
            }
            return 1;
        }
    };
    if dump_arg.is_some() && targets.len() != 1 {
        return usage_error("--dump needs exactly one target");
    }
    let mut outcomes: Vec<peacemaker_rip::run::Outcome> = Vec::with_capacity(targets.len());
    for (arch, variant) in &targets {
        outcomes.push(peacemaker_rip::run::execute(
            &source,
            &file,
            *arch,
            variant,
            &case_json,
            dump_arg.is_some(),
        ));
    }
    let all_ok = outcomes
        .iter()
        .all(|o| o.report["ok"].as_bool() == Some(true));
    let dump_outcome = dump_arg.as_ref().zip(outcomes.first());
    if let Some((dump_path, outcome)) = dump_outcome {
        match &outcome.dump {
            Some(dump) => {
                let text = serde_json::to_string_pretty(dump).expect("a JSON value serializes");
                if let Err(e) = std::fs::write(dump_path, format!("{text}\n")) {
                    return dump_error("dump", format!("{dump_path}: {e}"), json, &file);
                }
            }
            None => {
                if json {
                    print_json(&outcome.report);
                } else {
                    print_report(&outcome.report);
                }
                return 1;
            }
        }
    }
    if json {
        if outcomes.len() == 1 {
            print_json(&outcomes[0].report);
        } else {
            print_json(
                &serde_json::json!({"schema": 1, "ok": all_ok, "results": outcomes.iter().map(|o| &o.report).collect::<Vec<_>>()}),
            );
        }
    } else {
        for outcome in &outcomes {
            print_report(&outcome.report);
        }
    }
    u8::from(!all_ok)
}

// ---------------------------------------------------------------- fmt

fn fmt(mut args: Vec<String>) -> u8 {
    let check_only = take_flag(&mut args, "--check");
    if let Err(e) = reject_unknown_options(&args) {
        return usage_error(&e);
    }
    if args.is_empty() {
        return usage_error("`fmt` takes at least one FILE");
    }
    let mut status = 0;
    for file in &args {
        let source = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error[R0901]: {file}: {e}");
                status = 1;
                continue;
            }
        };
        let formatted = match peacemaker_rip::srcfmt::format_source(&source) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error[R0901]: {file}: {e}");
                status = 1;
                continue;
            }
        };
        if formatted == source {
            continue;
        }
        if check_only {
            println!("{file}");
            status = 1;
        } else if let Err(e) = std::fs::write(file, &formatted) {
            eprintln!("error[R0901]: {file}: {e}");
            status = 1;
        } else {
            eprintln!("formatted {file}");
        }
    }
    status
}

// ---------------------------------------------------------------- explain

fn explain(args: Vec<String>) -> u8 {
    if args.len() != 1 {
        return usage_error("`explain` takes one CODE or `--list`");
    }
    if args[0] == "--list" {
        for c in diag::codes() {
            println!("{}  {}", c.code, c.title);
        }
        return 0;
    }
    if args[0].starts_with("--") {
        return usage_error(&format!("unknown option `{}`", args[0]));
    }
    match diag::explain(&args[0].to_ascii_uppercase()) {
        Some(doc) => {
            println!("{}: {}\n\n{}", doc.code, doc.title, doc.body.trim_end());
            0
        }
        None => {
            eprintln!("error: unknown diagnostic code `{}`; run `peacemaker-rip explain --list` for the codes", args[0]);
            1
        }
    }
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return ExitCode::from(usage_error("no command given"));
    }
    let code = match args[0].as_str() {
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            0
        }
        "check" => inspect(args.split_off(1), false),
        "features" => inspect(args.split_off(1), true),
        "run" => run(args.split_off(1)),
        "build" => build(args.split_off(1)),
        "fmt" => fmt(args.split_off(1)),
        "explain" => explain(args.split_off(1)),
        _ => build(args),
    };
    ExitCode::from(code)
}
