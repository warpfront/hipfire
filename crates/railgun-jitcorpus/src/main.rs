//! `railgun-jit-corpus` — the offline JIT receipt corpus (railgun design §2.5).
//!
//! Every command takes `--work DIR` (default `./railgun-jit-corpus-work`): the
//! runtime compiler's hot cache (`HIPFIRE_KERNEL_CACHE`) and scratch live
//! there, never in the user's shared kernel cache.
//!
//! build  [--out DIR] [--build-dir DIR] [--arch A]... [--jobs N]
//!     Compile every runtime-JIT module of the admitted default programs and
//!     of railgun's own kernels (`programs.json`) with the runtime's recipe,
//!     certify each object with railgun-cert, write `DIR/v1/<build host key>/`.
//!     Sources compile in the fixed `--build-dir` (default `DIR/build`) so
//!     rebuilds repeat the bytes.
//! check  --corpus DIR
//!     CI gate: every node of every admitted default program and of
//!     railgun's own kernels must resolve, under the runtime's current JIT
//!     key, to a verified record. Exit 1 and name every missing node
//!     otherwise.
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
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        out: PathBuf::from("railgun-jit-corpus"), build_dir: None,
        corpus: None, work: PathBuf::from("railgun-jit-corpus-work"), json: None, capture: None, archs: Vec::new(), jobs: 16,
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
            "--jobs" => a.jobs = value()?.parse().map_err(|e| format!("--jobs: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn run() -> Result<bool> {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().ok_or("usage: railgun-jit-corpus build|check|repro|verify-runtime-cache ...")?;
    let mut args = parse(argv)?;
    corpus::install_product_defaults()?;
    std::fs::create_dir_all(&args.work).map_err(|e| format!("{}: {e}", args.work.display()))?;
    args.work = std::fs::canonicalize(&args.work).map_err(|e| format!("{}: {e}", args.work.display()))?;
    // The runtime compiler seeds and writes a hot cache when constructed.
    std::env::set_var("HIPFIRE_KERNEL_CACHE", args.work.join("hot-cache"));
    let inventory = corpus::inventory()?;
    if args.archs.is_empty() {
        args.archs = corpus::archs(&inventory);
    }
    match command.as_str() {
        "build" => {
            let build_dir = args.build_dir.unwrap_or_else(|| args.out.join("build"));
            let dir = corpus::build(&inventory, &corpus::BuildOptions { out: args.out, build_dir, archs: args.archs, jobs: args.jobs })?;
            println!("{}", dir.display());
            Ok(true)
        }
        "check" => {
            let dir = args.corpus.ok_or("check needs --corpus")?;
            let jit = JitCorpus::open(&dir).map_err(|e| format!("railgun JIT corpus rejected: {e}"))?;
            let index = jit.index();
            println!("corpus {} (build host {} {}, toolchain {} `{}`, source {})", dir.display(), index.build_host.hostname,
                &index.build_host.key[..16], &index.toolchain.pin[..16], index.toolchain.id, index.source_commit);
            let programs: Vec<_> = corpus::current_program_keys(&inventory)?.into_iter().filter(|p| args.archs.contains(&p.arch)).collect();
            let coverage = corpus::check(&jit, &programs);
            println!("{:<28} {:<8} {:>5} {:>7} {:>6} {:>7} {:>7}", "program", "arch", "nodes", "covered", "proven", "unknown", "missing");
            for c in &coverage {
                println!("{:<28} {:<8} {:>5} {:>7} {:>6} {:>7} {:>7}", c.program, c.arch, c.nodes, c.covered, c.proven, c.unknown, c.missing.len());
            }
            let mut ok = true;
            for c in coverage.iter().filter(|c| !c.missing.is_empty()) {
                ok = false;
                for m in &c.missing {
                    eprintln!("ERROR railgun JIT corpus: {} ({}) node {} has no fact record: {}", c.program, c.arch, m.symbol, m.reason);
                }
            }
            if !ok {
                eprintln!("ERROR railgun JIT corpus check FAILED: admitted default programs or railgun's own kernels have nodes without records");
            }
            Ok(ok)
        }
        "repro" => {
            let dir = args.corpus.ok_or("repro needs --corpus")?;
            let jit = JitCorpus::open(&dir).map_err(|e| e.to_string())?;
            let results = corpus::repro(&jit, &args.work.join("repro"), &args.archs, args.jobs)?;
            if let Some(path) = &args.json {
                std::fs::write(path, serde_json::to_vec_pretty(&results).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            }
            let count = |f: &dyn Fn(&corpus::Repro) -> bool| results.iter().filter(|r| r.error.is_none() && f(r)).count();
            let errors: Vec<_> = results.iter().filter(|r| r.error.is_some()).collect();
            println!("modules {} (errors {})", results.len(), errors.len());
            println!("rebuild at corpus build paths: raw-equal {}", count(&|r| r.rebuild_raw_equal));
            println!("runtime-style unique temporaries x2: raw-equal {}, code-equal {}", count(&|r| r.jit_raw_equal), count(&|r| r.jit_code_equal));
            println!("include order permuted: raw-equal {}, code-equal {}", count(&|r| r.include_order_raw_equal), count(&|r| r.include_order_code_equal));
            println!("unique temporaries with -fuse-cuid=none: raw-equal {}", count(&|r| r.nocuid_raw_equal));
            for r in errors {
                eprintln!("ERROR {} {}: {}", r.arch, r.module, r.error.as_deref().unwrap_or_default());
            }
            Ok(true)
        }
        "verify-runtime-cache" => {
            let capture = args.capture.ok_or("verify-runtime-cache needs --capture")?;
            let arch = args.archs.first().ok_or("verify-runtime-cache needs --arch")?.clone();
            let verdicts = corpus::verify_runtime_cache(&capture, &arch)?;
            let mut ok = true;
            for v in &verdicts {
                ok &= v.source_equal && v.key_equal;
                println!("{:<48} {:<44} {} source={} key={}", v.symbol, v.module, v.runtime_key,
                    if v.source_equal { "equal" } else { "DIFFERS" }, if v.key_equal { "equal" } else { "DIFFERS" });
            }
            println!("{} launches: {} with registry source and runtime cache key both equal",
                verdicts.len(), verdicts.iter().filter(|v| v.source_equal && v.key_equal).count());
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
