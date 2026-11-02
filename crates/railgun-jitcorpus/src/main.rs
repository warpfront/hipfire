// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `railgun-jit-corpus` — the offline JIT receipt corpus (railgun design §2.5).
//!
//! Every command takes `--work DIR` (default `./railgun-jit-corpus-work`): the
//! runtime compiler's hot cache (`HIPFIRE_KERNEL_CACHE`) and scratch live
//! there, never in the user's shared kernel cache.
//!
//! build  [--out DIR] [--build-dir DIR] [--arch A]... [--jobs N]
//!     Compile every runtime-JIT module of the admitted default programs and
//!     of railgun's own kernels (`programs.json`) with the runtime's recipe,
//!     certify each object with railgun-cert, write `DIR/v2/<build host key>/`.
//!     Also certifies the arch's embedded pm_decode twin images (the objects
//!     the product default loads in place of some hipcc modules) and records
//!     them as `pm_native` under `native/<arch>/`.
//!     Sources compile in the fixed `--build-dir` (default `DIR/build`) so
//!     rebuilds repeat the bytes.
//! Both `build` and `check` take `--window-sequence FILE` (repeatable): the
//! node sequence the certified DFlash window builder dumped
//! (`HIPFIRE_DFLASH_RAILGUN_SEQUENCE_OUT`), added as the node set
//! `dflash-whole16-gfx1201` so the corpus covers every module of that plan.
//! Both also take `--program-sequence FILE` (repeatable) with an optional
//! `--program-arch A` (default gfx1201): the Raon plain-AR decode body's launch
//! sequence (`sequence.jsonl` of the probe's inventory phase: `symbol`, `module`
//! and the loaded object's `sha256` per kernel node), added as its own program
//! `raon-ar-decode-<arch>` (never relabelled as the DFlash window). `check` then
//! requires every node to resolve to a verified record for exactly that object.
//! `--only-program ID` (repeatable) restricts `build`/`check` to the named
//! programs (plus railgun's own kernels), e.g. `raon-ar-decode-gfx1201`.
//! check  --corpus DIR
//!     CI gate: every node of every admitted default program and of
//!     railgun's own kernels must resolve, under the runtime's current JIT
//!     key (twin nodes: by embedded image digest), to a verified record.
//!     Exit 1 and name every missing node otherwise.
//! verify-install  --corpus DIR --install-root DIR [--arch A]...
//!     CPU-only dry run of the runtime's kernel loader: with the CWD set to
//!     `DIR` (the runtime probes `kernels/compiled/<arch>` under its CWD), load
//!     every hipcc module of the selected programs through
//!     `KernelCompiler::compile_for_symbol` and require the object it resolves
//!     to to be a file of that pack with the corpus record's SHA-256 (embedded
//!     images: the `pm_native` record). Exit 1 names every module that would not
//!     run the corpus bytes. Set `HIPFIRE_KERNEL_CACHE` to a private dir.
//! repro  --corpus DIR [--arch A]... [--jobs N] [--json FILE]
//!     Recompile every corpus module: at the build paths, through runtime-style
//!     unique temporaries twice, with the include order permuted, and with
//!     `-fuse-cuid=none`.
//! verify-runtime-cache --capture FILE --arch A
//!     Check a Redline capture's content-keyed JIT cache entries against the
//!     registry source and the runtime's cache-key function.
use std::path::PathBuf;
use std::process::ExitCode;

use railgun_corpus::JitCorpus;
use railgun_jitcorpus as corpus;

type Result<T, E = String> = std::result::Result<T, E>;

struct Args {
    out: PathBuf,
    build_dir: Option<PathBuf>,
    corpus: Option<PathBuf>,
    work: PathBuf,
    json: Option<PathBuf>,
    capture: Option<PathBuf>,
    archs: Vec<String>,
    jobs: usize,
    window_sequences: Vec<PathBuf>,
    program_sequences: Vec<PathBuf>,
    program_arch: String,
    only_programs: Vec<String>,
    install_root: Option<PathBuf>,
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        out: PathBuf::from("railgun-jit-corpus"),
        build_dir: None,
        corpus: None,
        work: PathBuf::from("railgun-jit-corpus-work"),
        json: None,
        capture: None,
        archs: Vec::new(),
        jobs: 16,
        window_sequences: Vec::new(),
        program_sequences: Vec::new(),
        program_arch: "gfx1201".to_owned(),
        only_programs: Vec::new(),
        install_root: None,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => a.out = value()?.into(),
            "--build-dir" => a.build_dir = Some(value()?.into()),
            "--corpus" => a.corpus = Some(value()?.into()),
            "--work" => a.work = value()?.into(),
            "--json" => a.json = Some(value()?.into()),
            "--capture" => a.capture = Some(value()?.into()),
            "--arch" => a.archs.push(value()?),
            "--window-sequence" => a.window_sequences.push(value()?.into()),
            "--program-sequence" => a.program_sequences.push(value()?.into()),
            "--program-arch" => a.program_arch = value()?,
            "--only-program" => a.only_programs.push(value()?),
            "--install-root" => a.install_root = Some(value()?.into()),
            "--jobs" => a.jobs = value()?.parse().map_err(|e| format!("--jobs: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn run() -> Result<bool> {
    let mut argv = std::env::args().skip(1);
    let command = argv
        .next()
        .ok_or("usage: railgun-jit-corpus build|check|repro|verify-runtime-cache ...")?;
    let mut args = parse(argv)?;
    corpus::install_product_defaults()?;
    std::fs::create_dir_all(&args.work).map_err(|e| format!("{}: {e}", args.work.display()))?;
    args.work =
        std::fs::canonicalize(&args.work).map_err(|e| format!("{}: {e}", args.work.display()))?;
    // The runtime compiler seeds and writes a hot cache when constructed.
    std::env::set_var("HIPFIRE_KERNEL_CACHE", args.work.join("hot-cache"));
    let mut inventory = corpus::inventory()?;
    for (i, path) in args.window_sequences.iter().enumerate() {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let id = if i == 0 {
            "dflash-whole16-gfx1201".to_owned()
        } else {
            format!("dflash-whole16-gfx1201-{}", i + 1)
        };
        inventory.add_window_sequence(&id, "gfx1201", &text)?;
    }
    for (i, path) in args.program_sequences.iter().enumerate() {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let id = corpus::raon_ar_program_id(&args.program_arch, i);
        inventory
            .add_program_sequence(&id, &args.program_arch, &text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    if !args.only_programs.is_empty() {
        inventory.retain_programs(&args.only_programs)?;
    }
    if args.archs.is_empty() {
        args.archs = corpus::archs(&inventory);
    }
    match command.as_str() {
        "build" => {
            let build_dir = args.build_dir.unwrap_or_else(|| args.out.join("build"));
            let dir = corpus::build(
                &inventory,
                &corpus::BuildOptions {
                    out: args.out,
                    build_dir,
                    archs: args.archs,
                    jobs: args.jobs,
                },
            )?;
            println!("{}", dir.display());
            Ok(true)
        }
        "check" => {
            let dir = args.corpus.ok_or("check needs --corpus")?;
            let jit =
                JitCorpus::open(&dir).map_err(|e| format!("railgun JIT corpus rejected: {e}"))?;
            let index = jit.index();
            println!(
                "corpus {} (build host {} {}, toolchain {} `{}`, source {})",
                dir.display(),
                index.build_host.hostname,
                &index.build_host.key[..16],
                &index.toolchain.pin[..16],
                index.toolchain.id,
                index.source_commit
            );
            let programs: Vec<_> = corpus::current_program_keys(&inventory)?
                .into_iter()
                .filter(|p| args.archs.contains(&p.arch))
                .collect();
            let coverage = corpus::check(&jit, &programs);
            println!(
                "{:<28} {:<8} {:>5} {:>7} {:>6} {:>7} {:>5} {:>7}",
                "program", "arch", "nodes", "covered", "proven", "unknown", "cond", "missing"
            );
            for c in &coverage {
                println!(
                    "{:<28} {:<8} {:>5} {:>7} {:>6} {:>7} {:>5} {:>7}",
                    c.program,
                    c.arch,
                    c.nodes,
                    c.covered,
                    c.proven,
                    c.unknown,
                    c.conditional,
                    c.missing.len()
                );
            }
            // Not errors: the record exists, but the node is not Proven on its own
            // (conditional: valid only under the runtime's generation-bound witness;
            // unknown: the owner will decline the program).
            for c in &coverage {
                for n in &c.unproven {
                    println!(
                        "NOTE railgun JIT corpus: {} ({}) node {} is {}{}",
                        c.program,
                        c.arch,
                        n.symbol,
                        n.status,
                        if n.status == "conditional" {
                            " (needs a witness at run time)"
                        } else {
                            ""
                        }
                    );
                }
            }
            let mut ok = true;
            for c in coverage.iter().filter(|c| !c.missing.is_empty()) {
                ok = false;
                for m in &c.missing {
                    eprintln!(
                        "ERROR railgun JIT corpus: {} ({}) node {} has no fact record: {}",
                        c.program, c.arch, m.symbol, m.reason
                    );
                }
            }
            if !ok {
                eprintln!("ERROR railgun JIT corpus check FAILED: admitted default programs or railgun's own kernels have nodes without records");
            }
            Ok(ok)
        }
        "verify-install" => {
            let dir = args.corpus.ok_or("verify-install needs --corpus")?;
            let root = args
                .install_root
                .ok_or("verify-install needs --install-root")?;
            let jit =
                JitCorpus::open(&dir).map_err(|e| format!("railgun JIT corpus rejected: {e}"))?;
            // `verify_install` changes the working directory: resolve the report path first.
            let json_out = args
                .json
                .as_ref()
                .map(|p| std::path::absolute(p).map_err(|e| format!("{}: {e}", p.display())))
                .transpose()?;
            let verdicts = corpus::verify_install(&inventory, &jit, &root, &args.archs)?;
            let mut ok = true;
            for v in &verdicts {
                ok &= v.ok;
                println!(
                    "{:<8} {:<56} {:<9} {} {}",
                    v.arch,
                    v.module,
                    v.source,
                    v.loaded_sha256.as_deref().unwrap_or("-"),
                    if v.ok {
                        "ok".to_owned()
                    } else {
                        format!("FAIL: {}", v.detail)
                    }
                );
            }
            let modules = verdicts.iter().filter(|v| v.source != "toolchain");
            println!(
                "{} modules: {} load the corpus bytes (toolchain pin {})",
                modules.clone().count(),
                modules.filter(|v| v.ok).count(),
                if verdicts.iter().any(|v| v.source == "toolchain" && !v.ok) {
                    "MISMATCH"
                } else {
                    "ok"
                }
            );
            if let Some(path) = &json_out {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&verdicts).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(ok)
        }
        "repro" => {
            let dir = args.corpus.ok_or("repro needs --corpus")?;
            let jit = JitCorpus::open(&dir).map_err(|e| e.to_string())?;
            let results = corpus::repro(&jit, &args.work.join("repro"), &args.archs, args.jobs)?;
            if let Some(path) = &args.json {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&results).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            let count = |f: &dyn Fn(&corpus::Repro) -> bool| {
                results.iter().filter(|r| r.error.is_none() && f(r)).count()
            };
            let errors: Vec<_> = results.iter().filter(|r| r.error.is_some()).collect();
            println!("modules {} (errors {})", results.len(), errors.len());
            println!(
                "rebuild at corpus build paths: raw-equal {}",
                count(&|r| r.rebuild_raw_equal)
            );
            println!(
                "runtime-style unique temporaries x2: raw-equal {}, code-equal {}",
                count(&|r| r.jit_raw_equal),
                count(&|r| r.jit_code_equal)
            );
            println!(
                "include order permuted: raw-equal {}, code-equal {}",
                count(&|r| r.include_order_raw_equal),
                count(&|r| r.include_order_code_equal)
            );
            println!(
                "unique temporaries with -fuse-cuid=none: raw-equal {}",
                count(&|r| r.nocuid_raw_equal)
            );
            for r in errors {
                eprintln!(
                    "ERROR {} {}: {}",
                    r.arch,
                    r.module,
                    r.error.as_deref().unwrap_or_default()
                );
            }
            Ok(true)
        }
        "verify-runtime-cache" => {
            let capture = args.capture.ok_or("verify-runtime-cache needs --capture")?;
            let arch = args
                .archs
                .first()
                .ok_or("verify-runtime-cache needs --arch")?
                .clone();
            let verdicts = corpus::verify_runtime_cache(&capture, &arch)?;
            let mut ok = true;
            for v in &verdicts {
                ok &= v.source_equal && v.key_equal;
                println!(
                    "{:<48} {:<44} {} source={} key={}",
                    v.symbol,
                    v.module,
                    v.runtime_key,
                    if v.source_equal { "equal" } else { "DIFFERS" },
                    if v.key_equal { "equal" } else { "DIFFERS" }
                );
            }
            println!(
                "{} launches: {} with registry source and runtime cache key both equal",
                verdicts.len(),
                verdicts
                    .iter()
                    .filter(|v| v.source_equal && v.key_equal)
                    .count()
            );
            Ok(ok)
        }
        other => Err(format!("unknown command {other}")),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("railgun-jit-corpus: {e}");
            ExitCode::FAILURE
        }
    }
}
