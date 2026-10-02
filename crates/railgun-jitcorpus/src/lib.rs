//! railgun-jitcorpus: the offline JIT receipt corpus (railgun design §2.5,
//! milestone MJ).
//!
//! The admitted default programs (`programs.json`, one per route and arch,
//! nodes taken from each program's recorded capture) name the runtime-JIT
//! module behind every launch; `programs.json` also names railgun's own
//! kernels (the copy kernels its lowering launches), required on each arch
//! railgun certifies. The generated source and compile recipe for a
//! module are the runtime's own JIT cache-key inputs: the source expression
//! its launcher passes to `ensure_kernel`, as inventoried by
//! `rdna_compute::kernel_registry::corpus_entries`, and
//! `KernelCompiler::recipe_for_source` under the product-default process
//! configuration. `build` compiles each module with the runtime's hipcc argv
//! on this host, certifies the object with `railgun-cert`, and writes a
//! versioned, per-build-host corpus (`railgun_corpus` layout) whose pack
//! directories the runtime can load verbatim. `check` is the CI gate: every
//! node of every node set must resolve, under the runtime's current JIT key,
//! to a record with a verified receipt.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use hipfire_config::{ConfigLayer, ProcessConfig, CONFIG_SCHEMA_VERSION};
use railgun_corpus::{
    sha256_hex, BuildHost, CorpusIndex, CorpusRecord, JitCorpus, KernelVerdict, MissingNode, NodeRecord, ProgramNode, ProgramRecord,
    RejectedObject, ToolchainPin, CORPUS_INDEX, CORPUS_SCHEMA, CORPUS_VERSION,
};
use rdna_compute::kernel_registry::{self, KernelEntry};
use rdna_compute::{FeatureFlags, KernelCompiler};
use serde::{Deserialize, Serialize};

pub const PROGRAMS_JSON: &str = include_str!("../programs.json");
pub const PROGRAMS_SCHEMA: &str = "railgun-admitted-programs";

type Result<T, E = String> = std::result::Result<T, E>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema: String,
    pub version: u32,
    pub admission: String,
    pub programs: Vec<Program>,
    pub railgun_kernels: RailgunKernels,
}

/// The kernels railgun's own lowering launches, on every arch in `archs`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RailgunKernels {
    pub source: String,
    pub archs: Vec<String>,
    /// `(symbol, module)`.
    pub nodes: Vec<(String, String)>,
}

/// Nodes the corpus must cover on one arch: an admitted program's, or
/// railgun's own kernels (id `railgun-kernels-<arch>`).
pub struct NodeSet<'a> {
    pub id: String,
    pub arch: &'a str,
    /// `(symbol, module)`.
    pub nodes: &'a [(String, String)],
}

impl Inventory {
    /// The admitted programs, then railgun's own kernels per arch.
    pub fn node_sets(&self) -> Vec<NodeSet<'_>> {
        let programs = self.programs.iter().map(|p| NodeSet { id: p.id.clone(), arch: &p.arch, nodes: &p.nodes });
        let railgun = self.railgun_kernels.archs.iter()
            .map(|arch| NodeSet { id: format!("railgun-kernels-{arch}"), arch, nodes: &self.railgun_kernels.nodes });
        programs.chain(railgun).collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub id: String,
    pub arch: String,
    pub route: String,
    pub golden: Golden,
    pub nodes_from: NodesFrom,
    /// `(symbol, module)` for every distinct kernel the program launches.
    pub nodes: Vec<(String, String)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Golden {
    pub sequence_hash: String,
    pub launches: u64,
    pub unique_kernels: u64,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodesFrom {
    pub capture: String,
    pub sequence_hash: String,
    pub launches: u64,
    /// The capture is the sealed golden itself, not an earlier recording.
    pub matches_golden: bool,
    pub note: String,
}

pub fn inventory() -> Result<Inventory> {
    let inventory: Inventory = serde_json::from_str(PROGRAMS_JSON).map_err(|e| format!("programs.json: {e}"))?;
    if inventory.schema != PROGRAMS_SCHEMA || inventory.version != 1 {
        return Err(format!("programs.json: schema {} v{}", inventory.schema, inventory.version));
    }
    Ok(inventory)
}

fn product_config() -> ProcessConfig {
    ProcessConfig { schema_version: CONFIG_SCHEMA_VERSION, values: ConfigLayer::default() }
}

/// Pin the process to the product-default configuration before anything
/// reads it, so no user TOML or `HIPFIRE_*` override reaches a recipe.
pub fn install_product_defaults() -> Result<()> {
    match hipfire_config::install_process_config(product_config()) {
        Ok(()) => Ok(()),
        Err(_) if hipfire_config::active_process_config() == Some(&product_config()) => Ok(()),
        Err(_) => Err("a non-default hipfire process configuration is already active".into()),
    }
}

/// `FeatureFlags::hipcc_extra_flags` under the product defaults: the
/// process-global extra flags every JIT compile on `arch` carries.
pub fn default_extra_flags(arch: &str) -> String {
    FeatureFlags::from_process_config(arch, &product_config()).hipcc_extra_flags
}

/// The modules one arch's programs need, as registry entries.
pub struct ArchPlan {
    pub arch: String,
    pub extra_flags: String,
    pub entries: Vec<KernelEntry>,
}

/// Resolve every node of every node set on `arch` to its registry entry.
pub fn plan(inventory: &Inventory, arch: &str) -> Result<ArchPlan> {
    let extra_flags = default_extra_flags(arch);
    let all = kernel_registry::corpus_entries(arch, &extra_flags).map_err(|e| format!("{arch}: {e:?}"))?;
    let mut seen = BTreeSet::new();
    for entry in &all {
        if !seen.insert(entry.module) {
            return Err(format!("{arch}: registry module {} appears twice", entry.module));
        }
    }
    let mut needed = BTreeSet::new();
    let mut errors = Vec::new();
    for set in inventory.node_sets().iter().filter(|s| s.arch == arch) {
        for (symbol, module) in set.nodes {
            match all.iter().find(|e| e.module == module) {
                None => errors.push(format!("{}: node {symbol}: module {module} is not in the {arch} registry", set.id)),
                Some(e) if !e.symbols.contains(&symbol.as_str()) => {
                    errors.push(format!("{}: node {symbol}: registry module {module} does not export it", set.id))
                }
                Some(_) => {
                    needed.insert(module.as_str());
                }
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let entries = all.into_iter().filter(|e| needed.contains(e.module)).collect();
    Ok(ArchPlan { arch: arch.to_owned(), extra_flags, entries })
}

/// The archs the inventory covers, in first-seen order.
pub fn archs(inventory: &Inventory) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in inventory.node_sets() {
        if !out.iter().any(|a| a == s.arch) {
            out.push(s.arch.to_owned());
        }
    }
    out
}

/// A program's nodes keyed by the runtime's current JIT inputs.
#[derive(Clone, Debug)]
pub struct ProgramKeys {
    pub id: String,
    pub arch: String,
    /// `(symbol, module, packaging_key)`.
    pub nodes: Vec<(String, String, String)>,
}

/// Key every node set's nodes with `KernelCompiler::packaging_hash_for` over
/// the registry's current source and recipe (no compiler needed).
pub fn current_program_keys(inventory: &Inventory) -> Result<Vec<ProgramKeys>> {
    let mut out = Vec::new();
    for arch in archs(inventory) {
        let plan = plan(inventory, &arch)?;
        let keys: BTreeMap<&str, String> = plan.entries.iter()
            .map(|e| (e.module, KernelCompiler::packaging_hash_for(&arch, e.module, e.source(), &plan.extra_flags)))
            .collect();
        for set in inventory.node_sets().iter().filter(|s| s.arch == arch) {
            let nodes = set.nodes.iter().map(|(s, m)| (s.clone(), m.clone(), keys[m.as_str()].clone())).collect();
            out.push(ProgramKeys { id: set.id.clone(), arch: arch.clone(), nodes });
        }
    }
    Ok(out)
}

/// Coverage of one program by a corpus.
#[derive(Clone, Debug, Serialize)]
pub struct Coverage {
    pub program: String,
    pub arch: String,
    pub nodes: usize,
    pub covered: usize,
    /// Covered nodes whose A1 verdict is `proven` / `unknown`.
    pub proven: usize,
    pub unknown: usize,
    pub missing: Vec<MissingNode>,
}

/// The CI check: every node must resolve under its current key to a record
/// whose pack object and receipt both verify.
pub fn check(corpus: &JitCorpus, programs: &[ProgramKeys]) -> Vec<Coverage> {
    let pin = corpus.toolchain_pin().to_owned();
    programs.iter().map(|program| {
        let mut missing = Vec::new();
        let mut resolved: Vec<(&str, String, &CorpusRecord)> = Vec::new();
        for (symbol, module, key) in &program.nodes {
            let Some(record) = corpus.record_for_module(&program.arch, module, key) else {
                let rejected = corpus.index().rejected.iter()
                    .find(|r| r.arch == program.arch && &r.module == module && &r.packaging_key == key);
                missing.push(match rejected {
                    Some(r) => MissingNode { symbol: symbol.clone(), artifact_sha256: r.artifact_sha256.clone(),
                        reason: format!("railgun-cert rejected {module}: {}", r.reason) },
                    None => MissingNode { symbol: symbol.clone(), artifact_sha256: String::new(),
                        reason: format!("no record for module {module} under the runtime's current key {key}") },
                });
                continue;
            };
            let object = corpus.root().join(&record.object);
            match std::fs::read(&object) {
                Ok(bytes) if sha256_hex(&bytes) == record.artifact_sha256 => resolved.push((symbol, record.artifact_sha256.clone(), record)),
                Ok(_) => missing.push(MissingNode { symbol: symbol.clone(), artifact_sha256: record.artifact_sha256.clone(),
                    reason: format!("{} does not hash to the record", object.display()) }),
                Err(e) => missing.push(MissingNode { symbol: symbol.clone(), artifact_sha256: record.artifact_sha256.clone(),
                    reason: format!("{}: {e}", object.display()) }),
            }
        }
        let nodes: Vec<ProgramNode> = resolved.iter().map(|(s, sha, _)| ProgramNode { symbol: s, artifact_sha256: sha }).collect();
        if let Err(e) = corpus.check_program(&program.id, &program.arch, &pin, &nodes) {
            missing.extend(e.missing);
        }
        let failed: BTreeSet<&str> = missing.iter().map(|m| m.symbol.as_str()).collect();
        let verdict = |s: &str, r: &CorpusRecord| r.kernels.iter().find(|k| k.symbol == s).map(|k| k.status.clone()).unwrap_or_default();
        let ok: Vec<String> = resolved.iter().filter(|(s, ..)| !failed.contains(s)).map(|(s, _, r)| verdict(s, r)).collect();
        Coverage {
            program: program.id.clone(), arch: program.arch.clone(), nodes: program.nodes.len(), covered: ok.len(),
            proven: ok.iter().filter(|v| *v == "proven").count(), unknown: ok.iter().filter(|v| *v == "unknown").count(), missing,
        }
    }).collect()
}

// ---------------------------------------------------------------- build

pub struct BuildOptions {
    /// Corpus base; the corpus lands in `<out>/v<version>/<host key>`.
    pub out: PathBuf,
    /// Fixed compile directory. hipcc names the `__hip_cuid_*` marker after
    /// the input path and argv, so a fixed directory repeats the bytes.
    pub build_dir: PathBuf,
    pub archs: Vec<String>,
    pub jobs: usize,
}

struct Compiled {
    object: Vec<u8>,
    /// Err: the certifier rejected the object (it ships without a record).
    receipt: Result<railgun_cert::Receipt>,
    headers: BTreeSet<String>,
}

/// `f` over `items` on `jobs` threads, results in item order.
fn run_parallel<T: Sync, R: Send>(items: &[T], jobs: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs.clamp(1, items.len().max(1))).map(|_| scope.spawn(|| {
            let mut local = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i) else { break };
                local.push((i, f(item)));
            }
            local
        })).collect();
        workers.into_iter().flat_map(|w| w.join().expect("worker panicked")).collect()
    });
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

fn run(program: &str, args: &[String], rocm_path: Option<&Path>) -> Result<std::process::Output> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(root) = rocm_path {
        cmd.env("ROCM_PATH", root);
    }
    cmd.output().map_err(|e| format!("{program}: {e}"))
}

/// Headers the device preprocess of `argv` reads (`-E -H`).
fn headers(argv: &[String], rocm_path: Option<&Path>) -> Result<BTreeSet<String>> {
    let mut args: Vec<String> = Vec::new();
    let mut it = argv[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--genco" => args.push("--cuda-device-only".into()),
            "-o" => {
                it.next();
                args.extend(["-o".into(), "/dev/null".into()]);
            }
            _ => args.push(a.clone()),
        }
    }
    args.extend(["-E".into(), "-H".into()]);
    let out = run(&argv[0], &args, rocm_path)?;
    if !out.status.success() {
        return Err(format!("preprocess failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stderr).lines()
        .filter_map(|l| l.trim_start_matches('.').strip_prefix(' ').filter(|_| l.starts_with('.')))
        .map(|p| std::fs::canonicalize(p).map(|c| c.display().to_string()).unwrap_or_else(|_| p.to_owned()))
        .collect())
}

fn compile_and_certify(compiler: &KernelCompiler, entry: &KernelEntry, dir: &Path) -> Result<Compiled> {
    let src = dir.join(format!("{}.hip", entry.module));
    let obj = dir.join(format!("{}.hsaco", entry.module));
    let _ = std::fs::remove_file(&obj);
    compiler.compile_to_paths(entry.module, entry.source(), &src, &obj).map_err(|e| format!("{}: {e}", entry.module))?;
    let object = std::fs::read(&obj).map_err(|e| format!("{}: {e}", obj.display()))?;
    let receipt = railgun_cert::certify(&object, None).map_err(|e| e.to_string());
    if let Ok(receipt) = &receipt {
        let missing: Vec<&str> = entry.symbols.iter().copied().filter(|s| !receipt.railgun.kernels.iter().any(|k| k.symbol == *s)).collect();
        if !missing.is_empty() {
            return Err(format!("{}: object does not export registry symbol(s) {}", entry.module, missing.join(", ")));
        }
    }
    let headers = headers(&compiler.jit_argv(entry.module, entry.source(), &src, &obj), compiler.rocm_env_root())?;
    Ok(Compiled { object, receipt, headers })
}

fn git_revision() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let git = |args: &[&str]| Command::new("git").arg("-C").arg(dir).args(args).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let head = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--", "../../kernels", "../rdna-compute", "../railgun-cert", "../railgun-corpus", "."])
        .is_some_and(|s| !s.is_empty());
    if dirty { format!("{head}-dirty") } else { head }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes)
}

/// Compile, certify and index every module the inventory's node sets use
/// on `opts.archs`. Returns the corpus directory.
pub fn build(inventory: &Inventory, opts: &BuildOptions) -> Result<PathBuf> {
    install_product_defaults()?;
    let build_dir = std::fs::canonicalize(&opts.build_dir).or_else(|_| {
        std::fs::create_dir_all(&opts.build_dir).and_then(|_| std::fs::canonicalize(&opts.build_dir))
    }).map_err(|e| format!("{}: {e}", opts.build_dir.display()))?;
    let versioned = opts.out.join(format!("v{CORPUS_VERSION}"));
    let staging = versioned.join(format!(".staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    let mut records = Vec::new();
    let mut rejected = Vec::new();
    let mut all_headers = BTreeSet::new();
    let mut toolchain: Option<(String, Option<PathBuf>, String)> = None;
    for arch in &opts.archs {
        let plan = plan(inventory, arch)?;
        let compiler = KernelCompiler::new(arch, plan.extra_flags.clone()).map_err(|e| format!("{arch}: {e}"))?;
        let id = compiler.toolchain_id().to_owned();
        if id.is_empty() {
            return Err(format!("{arch}: no hipcc; the corpus is built with the pinned toolchain"));
        }
        let hipcc = compiler.jit_argv("probe", "", Path::new("probe.hip"), Path::new("probe.hsaco"))[0].clone();
        match &toolchain {
            Some((t, _, _)) if *t != id => return Err(format!("{arch}: toolchain {id} differs from {t}")),
            Some(_) => {}
            None => toolchain = Some((id, compiler.rocm_env_root().map(Path::to_path_buf), hipcc)),
        }
        let dir = build_dir.join(arch);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        eprintln!("{arch}: {} modules, extra flags `{}`", plan.entries.len(), plan.extra_flags);
        let results = run_parallel(&plan.entries, opts.jobs, |entry| compile_and_certify(&compiler, entry, &dir));
        let failures: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        if !failures.is_empty() {
            return Err(failures.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
        }
        for (entry, compiled) in plan.entries.iter().zip(results.into_iter().map(Result::unwrap)) {
            let artifact = sha256_hex(&compiled.object);
            let symbols: Vec<String> = entry.symbols.iter().map(|s| (*s).to_owned()).collect();
            let object_src = dir.join(format!("{}.hsaco", entry.module));
            compiler.publish_package(entry.module, entry.source(), &symbols, &object_src, &staging.join("pack").join(arch))?;
            let source_rel = staging.join("sources").join(arch).join(format!("{}.hip", entry.module));
            std::fs::create_dir_all(source_rel.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::write(&source_rel, entry.source()).map_err(|e| e.to_string())?;
            all_headers.extend(compiled.headers);
            let receipt = match compiled.receipt {
                Ok(receipt) => receipt,
                Err(reason) => {
                    eprintln!("{arch}: {} rejected by railgun-cert: {reason}", entry.module);
                    rejected.push(RejectedObject {
                        arch: arch.clone(), artifact_sha256: artifact, module: entry.module.to_owned(), symbols,
                        source_sha256: sha256_hex(entry.source().as_bytes()),
                        packaging_key: compiler.packaging_hash(entry.module, entry.source()),
                        object: format!("pack/{arch}/{}.hsaco", entry.module), reason,
                    });
                    continue;
                }
            };
            let receipt_rel = format!("receipts/{arch}/{artifact}.receipt.json");
            let receipt_bytes = write_json(&staging.join(&receipt_rel), &receipt)?;
            records.push(CorpusRecord {
                arch: arch.clone(),
                artifact_sha256: artifact,
                module: entry.module.to_owned(),
                symbols,
                source_sha256: sha256_hex(entry.source().as_bytes()),
                extra_flags: plan.extra_flags.clone(),
                flags: entry.flags.clone(),
                scheduler_profile: entry.scheduler_profile.clone().unwrap_or_default(),
                packaging_key: compiler.packaging_hash(entry.module, entry.source()),
                jit_cache_key: compiler.jit_cache_key(entry.module, entry.source()),
                argv: compiler.jit_argv(entry.module, entry.source(), &dir.join(format!("{}.hip", entry.module)), &object_src),
                object: format!("pack/{arch}/{}.hsaco", entry.module),
                receipt: receipt_rel,
                receipt_sha256: sha256_hex(&receipt_bytes),
                kernels: receipt.railgun.kernels.iter()
                    .map(|k| KernelVerdict { symbol: k.symbol.clone(), status: k.status.clone(), differential: k.differential.status.clone() })
                    .collect(),
            });
        }
    }
    let (id, rocm_env_root, hipcc) = toolchain.ok_or("no arch selected")?;
    let version = run(&hipcc, &["--version".into()], rocm_env_root.as_deref())?;
    let clang_root = rocm_env_root.clone().or_else(hipfire_config::rocm::root).ok_or("no ROCm root")?;
    let clang = std::fs::canonicalize(clang_root.join("lib/llvm/bin/clang")).map_err(|e| format!("clang: {e}"))?;
    let clang_sha256 = sha256_hex(&std::fs::read(&clang).map_err(|e| format!("{}: {e}", clang.display()))?);
    let mut header_lines = String::new();
    for h in &all_headers {
        let bytes = std::fs::read(h).map_err(|e| format!("{h}: {e}"))?;
        header_lines.push_str(&format!("{h}\t{}\n", sha256_hex(&bytes)));
    }
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_owned()).unwrap_or_default();
    let toolchain = ToolchainPin { pin: sha256_hex(id.as_bytes()), id, version_sha256: sha256_hex(&version.stdout) };
    let mut build_host = BuildHost {
        key: String::new(), hostname, hipcc, rocm_env_root: rocm_env_root.map(|p| p.display().to_string()),
        clang_sha256, header_set_sha256: sha256_hex(header_lines.as_bytes()), build_dir: build_dir.display().to_string(),
    };
    build_host.key = sha256_hex(&serde_json::to_vec(&(&build_host, &toolchain.version_sha256)).map_err(|e| e.to_string())?);
    std::fs::write(staging.join("headers.tsv"), &header_lines).map_err(|e| e.to_string())?;

    let programs = inventory.node_sets().iter().filter(|s| opts.archs.iter().any(|a| a == s.arch)).map(|p| ProgramRecord {
        id: p.id.clone(), arch: p.arch.to_owned(),
        nodes: p.nodes.iter().map(|(symbol, module)| NodeRecord {
            symbol: symbol.clone(), module: module.clone(),
            artifact_sha256: records.iter().find(|r| r.arch == p.arch && &r.module == module).map(|r| &r.artifact_sha256)
                .or_else(|| rejected.iter().find(|r| r.arch == p.arch && &r.module == module).map(|r| &r.artifact_sha256))
                .expect("planned module built").clone(),
        }).collect(),
    }).collect();
    let index = CorpusIndex {
        schema: CORPUS_SCHEMA.into(), version: CORPUS_VERSION, source_commit: git_revision(),
        build_host, toolchain, records, rejected, programs,
    };
    write_json(&staging.join(CORPUS_INDEX), &index)?;
    let dest = versioned.join(&index.build_host.key[..16]);
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&staging, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    JitCorpus::open(&dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

// ---------------------------------------------------------------- repro

/// One module's reproducibility result. `raw` compares object bytes;
/// `code` compares every kernel's text, descriptor and metadata digests.
#[derive(Clone, Debug, Serialize)]
pub struct Repro {
    pub arch: String,
    pub module: String,
    /// Recompiled at the corpus build paths: identical to the corpus object.
    pub rebuild_raw_equal: bool,
    /// Two compiles through runtime-style unique temporary object paths.
    pub jit_raw_equal: bool,
    pub jit_code_equal: bool,
    /// The same compile with an aliased include root before vs after `-I<root>/include`.
    pub include_order_raw_equal: bool,
    pub include_order_code_equal: bool,
    /// Runtime-style unique paths again, with `-fuse-cuid=none`.
    pub nocuid_raw_equal: bool,
    pub error: Option<String>,
}

fn code_equal(a: &[u8], b: &[u8]) -> Result<bool> {
    let id = |bytes: &[u8]| railgun_cert::code_identity(bytes).map_err(|e| e.to_string()).map(|v| {
        v.into_iter().map(|k| (k.symbol, k.text_sha256, k.kd_sha256, k.metadata_sha256)).collect::<Vec<_>>()
    });
    Ok(id(a)? == id(b)?)
}

fn compile_argv(argv: &[String], rocm: Option<&Path>, out: &Path) -> Result<Vec<u8>> {
    let _ = std::fs::remove_file(out);
    let o = run(&argv[0], &argv[1..], rocm)?;
    if !o.status.success() {
        return Err(format!("hipcc failed: {}", String::from_utf8_lossy(&o.stderr)));
    }
    std::fs::read(out).map_err(|e| format!("{}: {e}", out.display()))
}

fn with_output(argv: &[String], out: &Path) -> Vec<String> {
    let mut v = argv.to_vec();
    let i = v.iter().position(|a| a == "-o").expect("argv has -o");
    v[i + 1] = out.display().to_string();
    v
}

fn repro_one(corpus: &JitCorpus, record: &CorpusRecord, compiler: &KernelCompiler, work: &Path, alias: &str) -> Result<Repro> {
    let source = std::fs::read_to_string(corpus.root().join(format!("sources/{}/{}.hip", record.arch, record.module)))
        .map_err(|e| e.to_string())?;
    if sha256_hex(source.as_bytes()) != record.source_sha256 {
        return Err("corpus source does not match its record".into());
    }
    let rocm = compiler.rocm_env_root();
    let module = record.module.as_str();
    let build = Path::new(&corpus.index().build_host.build_dir).join(&record.arch);
    // 1. The corpus build paths again.
    let (src, obj) = (build.join(format!("{module}.hip")), build.join(format!("{module}.hsaco")));
    compiler.compile_to_paths(module, &source, &src, &obj).map_err(|e| e.to_string())?;
    let rebuild = std::fs::read(&obj).map_err(|e| e.to_string())?;
    // 2. Runtime-style: `{stem}.hip` in the cache, unique temporary objects.
    let dir = work.join(&record.arch);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stem = format!("{module}.{}", record.jit_cache_key);
    let jit_src = dir.join(format!("{stem}.hip"));
    std::fs::write(&jit_src, &source).map_err(|e| e.to_string())?;
    let temp = |tag: &str| dir.join(format!(".{stem}.{}_{tag}.hsaco.tmp", std::process::id()));
    let base = compiler.jit_argv(module, &source, &jit_src, &temp("a"));
    let jit_a = compile_argv(&base, rocm, &temp("a"))?;
    let jit_b = compile_argv(&with_output(&base, &temp("b")), rocm, &temp("b"))?;
    // 3. Include order: an alias of the include root before, then after it.
    let root_i = base.iter().position(|a| a.starts_with("-I")).ok_or("argv has no -I")?;
    let inc_out = dir.join(format!("{module}.include.hsaco"));
    let mut fwd = with_output(&base, &inc_out);
    fwd.insert(root_i, format!("-I{alias}"));
    let mut rev = with_output(&base, &inc_out);
    rev.insert(root_i + 1, format!("-I{alias}"));
    let inc_fwd = compile_argv(&fwd, rocm, &inc_out)?;
    let inc_rev = compile_argv(&rev, rocm, &inc_out)?;
    // 4. Unique paths with the cuid marker disabled.
    let nocuid = |tag: &str| {
        let mut v = with_output(&base, &temp(tag));
        v.insert(1, "-fuse-cuid=none".into());
        compile_argv(&v, rocm, &temp(tag))
    };
    let (nc_a, nc_b) = (nocuid("na")?, nocuid("nb")?);
    for tag in ["a", "b", "na", "nb"] {
        let _ = std::fs::remove_file(temp(tag));
    }
    Ok(Repro {
        arch: record.arch.clone(), module: module.to_owned(),
        rebuild_raw_equal: sha256_hex(&rebuild) == record.artifact_sha256,
        jit_raw_equal: jit_a == jit_b, jit_code_equal: code_equal(&jit_a, &jit_b)?,
        include_order_raw_equal: inc_fwd == inc_rev, include_order_code_equal: code_equal(&inc_fwd, &inc_rev)?,
        nocuid_raw_equal: nc_a == nc_b, error: None,
    })
}

/// Recompile every corpus module in the variants [`Repro`] names.
pub fn repro(corpus: &JitCorpus, work: &Path, archs: &[String], jobs: usize) -> Result<Vec<Repro>> {
    install_product_defaults()?;
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let mut out = Vec::new();
    for arch in archs {
        let records: Vec<&CorpusRecord> = corpus.index().records.iter().filter(|r| &r.arch == arch).collect();
        let Some(first) = records.first() else { continue };
        let compiler = KernelCompiler::new(arch, first.extra_flags.clone()).map_err(|e| e.to_string())?;
        if sha256_hex(compiler.toolchain_id().as_bytes()) != corpus.toolchain_pin() {
            return Err(format!("{arch}: local toolchain is not the corpus pin"));
        }
        let root_include = first.argv.iter().find_map(|a| a.strip_prefix("-I")).ok_or("record argv has no -I")?;
        let alias = std::fs::canonicalize(root_include).map(|p| p.display().to_string()).unwrap_or_default();
        let alias = if alias.is_empty() || alias == root_include { format!("{root_include}/.") } else { alias };
        out.extend(run_parallel(&records, jobs, |r| repro_one(corpus, r, &compiler, work, &alias).unwrap_or_else(|e| Repro {
            arch: r.arch.clone(), module: r.module.clone(), rebuild_raw_equal: false, jit_raw_equal: false, jit_code_equal: false,
            include_order_raw_equal: false, include_order_code_equal: false, nocuid_raw_equal: false, error: Some(e),
        })));
    }
    Ok(out)
}

// ---------------------------------------------------------------- runtime cache

/// One launch of a runtime capture checked against the registry: the
/// runtime's own JIT cache entry `{module}.{key}.hip` is the generated source
/// it compiled, and `key` is its cache key over that source and the recipe.
#[derive(Clone, Debug, Serialize)]
pub struct CacheVerdict {
    pub symbol: String,
    pub module: String,
    pub runtime_key: String,
    /// The registry's source is byte-identical to the runtime's.
    pub source_equal: bool,
    /// `jit_cache_key` over the registry source and recipe equals the runtime's key.
    pub key_equal: bool,
}

/// Verify a Redline capture whose artifacts are content-keyed JIT cache
/// entries against the registry and the runtime's cache-key function.
pub fn verify_runtime_cache(capture: &Path, arch: &str) -> Result<Vec<CacheVerdict>> {
    install_product_defaults()?;
    let text = std::fs::read_to_string(capture).map_err(|e| format!("{}: {e}", capture.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    fn find(v: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
        match v {
            serde_json::Value::Object(m) if m.contains_key("sequence_hash") => m.get("sequence").and_then(|s| s.as_array()),
            serde_json::Value::Object(m) => m.values().find_map(find),
            serde_json::Value::Array(a) => a.iter().find_map(find),
            _ => None,
        }
    }
    let sequence = find(&value).ok_or("capture has no redline sequence")?;
    let mut launches = BTreeMap::new();
    for l in sequence {
        let (Some(k), Some(a)) = (l["kernel"].as_str(), l["artifact"].as_str()) else { continue };
        launches.insert(k.to_owned(), PathBuf::from(a));
    }
    let extra = default_extra_flags(arch);
    let compiler = KernelCompiler::new(arch, extra.clone()).map_err(|e| e.to_string())?;
    let registry = kernel_registry::corpus_entries(arch, &extra).map_err(|e| format!("{e:?}"))?;
    let mut out = Vec::new();
    for (symbol, artifact) in launches {
        let name = artifact.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let Some((module, key)) = name.strip_suffix(".hsaco").and_then(|s| s.rsplit_once('.')) else {
            return Err(format!("{symbol}: {name} is not a content-keyed cache entry"));
        };
        let runtime_source = std::fs::read(artifact.with_extension("hip")).map_err(|e| format!("{}: {e}", artifact.display()))?;
        let entry = registry.iter().find(|e| e.module == module);
        out.push(CacheVerdict {
            symbol, module: module.to_owned(), runtime_key: key.to_owned(),
            source_equal: entry.is_some_and(|e| e.source().as_bytes() == runtime_source.as_slice()),
            key_equal: entry.is_some_and(|e| compiler.jit_cache_key(module, e.source()) == key),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_admitted_program_node_resolves_to_a_registry_module_that_exports_it() {
        install_product_defaults().unwrap();
        let inventory = inventory().unwrap();
        for arch in archs(&inventory) {
            let plan = plan(&inventory, &arch).unwrap_or_else(|e| panic!("{e}"));
            assert!(!plan.entries.is_empty(), "{arch}");
        }
    }
}
