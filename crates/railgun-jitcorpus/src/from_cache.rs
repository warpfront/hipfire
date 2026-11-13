// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `from-cache`: a local receipt corpus over the user's OWN already-compiled
//! kernel objects. CPU only, no hipcc compile, no network, no GPU: it certifies
//! (railgun-cert, the same `certify(object, radiowave sidecar)` as `build`)
//! the objects the runtime would load for a checked program's nodes and writes
//! the `railgun_corpus` layout `build` writes, so
//! `JitCorpus::open(<out>)` and `check` accept it.
//!
//! # Where an object comes from (never one the runtime would not load)
//!
//! For every registry module a node set needs, under the runtime's current
//! source and recipe (product defaults):
//!
//! 1. the installed pack (`--pack`, `kernels/compiled/<arch>`), verbatim, when
//!    `rdna_compute::kernel_pack::verify_module` (the loader's own indexed
//!    check: packaging key = source + recipe, flags, ABI, object SHA-256,
//!    `.hash` sidecar, symbols, toolchain) accepts it;
//! 2. else the hot cache entry `<module>.<jit cache key>.hsaco` whose content
//!    key equals the key the runtime computes now (or the validated legacy
//!    `<module>.hsaco` + `.hash` pair whose hash is that key).
//!
//! Anything else (missing object, stale key, symbol not exported, certifier
//! rejection, embedded object rejected) refuses the whole run: no corpus is
//! published. Embedded native objects (`pm_decode` twins, the Raon
//! `gated_delta_vector_pm` image) are certified by digest exactly as `build`
//! does; a hipcc object of a twin module is optional (the product default
//! loads the twin) and recorded only when present and certifiable.
//!
//! # Toolchain pin
//!
//! The pin is SHA-256 of the toolchain id the runtime itself computes
//! (`KernelCompiler::local_toolchain_id`, the first line of `hipcc --version`;
//! `WindowCorpus::open` compares `SHA-256(compiler.toolchain_id())`). Without a
//! device compiler (or under `HIPFIRE_NO_DEVICE_COMPILER`) the runtime's id is
//! empty and the pin is SHA-256 of the empty string; the same function is used
//! here, so the corpus is opened exactly when the runtime would open it. The
//! probe runs `hipcc --version` only; nothing is compiled. `build_host` and
//! `toolchain.version_sha256` fields a compile would fill (clang, header set,
//! full version output) are marked `from-cache` / empty: nothing was
//! preprocessed or compiled. Records carry no `argv` (no compiler was invoked).
//!
//! # Publication
//!
//! The corpus is built in `<out>.tmp-<pid>` beside `out` and renamed into
//! place; a run that refuses or fails removes it and leaves `out` as it was.
//! An existing `out` holding byte-identical `index.json` and verifying clean
//! is left untouched (idempotent); otherwise it is moved aside, replaced by one
//! rename and deleted. Concurrent runs use distinct temp siblings; whichever
//! renames last wins and the loser's rename, which can only meet a non-empty
//! directory, is resolved by the identical-index check, so `out` is always a
//! complete corpus.
use super::*;

use std::time::Instant;

use rdna_compute::kernel_pack;

/// A checked program list embedded in the binary.
pub struct EmbeddedProgram {
    pub id: &'static str,
    pub arch: &'static str,
    /// `(symbol, module)` node lines, [`Inventory::add_program_nodes`] format.
    pub jsonl: &'static str,
}

/// The programs `from-cache` knows, by id.
pub const EMBEDDED_PROGRAMS: &[EmbeddedProgram] = &[
    // The shipped decode body: the installed mq4-xts (MQ8 attention, layout-B GEMVs).
    EmbeddedProgram {
        id: "raon-ar-decode-gfx1201",
        arch: "gfx1201",
        jsonl: include_str!("../programs/raon-ar-decode-gfx1201.jsonl"),
    },
    // The earlier a8attn-s artifact (A8 attention GEMVs).
    EmbeddedProgram {
        id: "raon-ar-decode-a8attn-gfx1201",
        arch: "gfx1201",
        jsonl: include_str!("../programs/raon-ar-decode-a8attn-gfx1201.jsonl"),
    },
    // The same two decode bodies on the 7900 XTX (exact gfx1100; the owner asks for
    // `raon-ar-decode-<arch>`).
    EmbeddedProgram {
        id: "raon-ar-decode-gfx1100",
        arch: "gfx1100",
        jsonl: include_str!("../programs/raon-ar-decode-gfx1100.jsonl"),
    },
    EmbeddedProgram {
        id: "raon-ar-decode-a8attn-gfx1100",
        arch: "gfx1100",
        jsonl: include_str!("../programs/raon-ar-decode-a8attn-gfx1100.jsonl"),
    },
];

pub struct FromCacheOptions {
    pub arch: String,
    /// An [`EMBEDDED_PROGRAMS`] id.
    pub program: String,
    /// The hot JIT cache: the cache root (holding `<arch>/`) or the arch dir.
    pub cache: PathBuf,
    /// The installed pack arch dir (`<root>/kernels/compiled/<arch>`).
    pub pack: Option<PathBuf>,
    /// The corpus directory (holds `index.json`).
    pub out: PathBuf,
    pub jobs: usize,
    /// The toolchain id the runtime compiles under; `None` probes it with
    /// `KernelCompiler::local_toolchain_id` (empty when no device compiler).
    pub toolchain_id: Option<String>,
}

/// A node `from-cache` could not certify.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MissingObject {
    pub symbol: String,
    pub module: String,
    pub why: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The first reason, one line.
    pub reason: String,
    pub missing: Vec<MissingObject>,
}

#[derive(Clone, Debug)]
pub struct FromCacheOk {
    pub program: String,
    pub arch: String,
    pub out: PathBuf,
    pub toolchain_pin: String,
    /// Nodes of the requested program.
    pub nodes: usize,
    /// Distinct certified objects in the corpus.
    pub objects: usize,
    pub proven: usize,
    pub conditional: usize,
    pub unknown: usize,
    pub millis: u128,
}

#[derive(Clone, Debug)]
pub enum FromCacheOutcome {
    Ok(FromCacheOk),
    Refused(Refusal),
}

fn refuse(reason: impl Into<String>) -> Result<FromCacheOutcome> {
    Ok(FromCacheOutcome::Refused(Refusal {
        reason: reason.into(),
        missing: Vec::new(),
    }))
}

/// Removes the temp sibling unless disarmed (refusal, error, panic).
struct TempDir(Option<PathBuf>);

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

fn nonempty_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0)
}

fn holds_hsaco(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut it| {
        it.any(|e| {
            e.is_ok_and(|e| {
                e.path().extension().is_some_and(|x| x == "hsaco") && nonempty_file(&e.path())
            })
        })
    })
}

/// The hot cache dir: `<cache>/<arch>` when that is a directory, else `<cache>`
/// itself; `None` when the chosen dir holds no `*.hsaco`.
fn resolve_cache(cache: &Path, arch: &str) -> Option<PathBuf> {
    let sub = cache.join(arch);
    let dir = if sub.is_dir() { sub } else { cache.to_owned() };
    holds_hsaco(&dir).then_some(dir)
}

/// An object found for a module, with the files that travel with it.
struct Located {
    object: Vec<u8>,
    radiowave: Option<String>,
    /// Pack-sourced: the `.hash` and `.index.json` bytes, copied verbatim.
    pack_meta: Option<(Vec<u8>, Vec<u8>)>,
}

struct Finder<'a> {
    arch: &'a str,
    extra_flags: &'a str,
    toolchain_id: &'a str,
    pack: Option<&'a Path>,
    cache: &'a Path,
}

fn read_sidecar(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn locate(l: &Finder, entry: &KernelEntry, symbols: &[&str]) -> Result<Located> {
    let module = entry.module;
    let mut pack_note = None;
    if let Some(dir) = l.pack {
        match kernel_pack::verify_module(
            dir,
            l.arch,
            l.extra_flags,
            module,
            entry.source(),
            symbols,
            Some(l.toolchain_id),
        ) {
            Ok(Some(found)) => {
                let object = std::fs::read(&found.object)
                    .map_err(|e| format!("{}: {e}", found.object.display()))?;
                let read = |suffix: &str| {
                    std::fs::read(dir.join(format!("{module}.{suffix}")))
                        .map_err(|e| format!("{module}.{suffix}: {e}"))
                };
                return Ok(Located {
                    object,
                    radiowave: read_sidecar(&dir.join(format!("{module}.radiowave.json"))),
                    pack_meta: Some((read("hash")?, read("index.json")?)),
                });
            }
            Ok(None) => {}
            Err(e) => pack_note = Some(format!("pack object rejected ({e})")),
        }
    }
    let key = KernelCompiler::jit_cache_key_for(
        l.arch,
        module,
        entry.source(),
        l.extra_flags,
        l.toolchain_id,
    );
    let hot = l.cache.join(format!("{module}.{key}.hsaco"));
    let found = if nonempty_file(&hot) {
        Some((hot, l.cache.join(format!("{module}.{key}.radiowave.json"))))
    } else {
        let legacy = l.cache.join(format!("{module}.hsaco"));
        let stored = std::fs::read_to_string(l.cache.join(format!("{module}.hash")));
        (nonempty_file(&legacy) && stored.is_ok_and(|s| s.trim() == key))
            .then(|| (legacy, l.cache.join(format!("{module}.radiowave.json"))))
    };
    match found {
        Some((object, sidecar)) => Ok(Located {
            object: std::fs::read(&object).map_err(|e| format!("{}: {e}", object.display()))?,
            radiowave: read_sidecar(&sidecar),
            pack_meta: None,
        }),
        None => {
            let none =
                format!("no hot cache entry {module}.{key}.hsaco under the runtime's current key");
            Err(match pack_note {
                Some(note) => format!("{note}; {none}"),
                None => {
                    if l.pack.is_some() {
                        format!("not in the pack; {none}")
                    } else {
                        none
                    }
                }
            })
        }
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

/// Certify the objects of the embedded program `opts.program` from the user's
/// own cache and pack and publish the corpus at `opts.out`. `Err` is a failure
/// (exit 1); a program the objects cannot certify is a
/// [`FromCacheOutcome::Refused`] (exit 2) and publishes nothing.
pub fn from_cache(opts: &FromCacheOptions) -> Result<FromCacheOutcome> {
    let Some(program) = EMBEDDED_PROGRAMS.iter().find(|p| p.id == opts.program) else {
        return refuse(format!(
            "unknown program id `{}` (known: {})",
            opts.program,
            EMBEDDED_PROGRAMS
                .iter()
                .map(|p| p.id)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    };
    from_cache_program(opts, program)
}

/// [`from_cache`] over an explicit checked program list (`opts.program` is
/// ignored in favour of `program`): the registry of embedded programs is
/// [`EMBEDDED_PROGRAMS`]; tests pass a miniature table.
pub fn from_cache_program(
    opts: &FromCacheOptions,
    program: &EmbeddedProgram,
) -> Result<FromCacheOutcome> {
    let started = Instant::now();
    install_product_defaults()?;
    let arch = opts.arch.as_str();
    if program.arch != arch {
        return refuse(format!(
            "arch `{arch}` is not the arch `{}` of program {}",
            program.arch, program.id
        ));
    }
    let mut inventory = inventory()?;
    inventory.programs.clear();
    inventory.railgun_kernels.archs.retain(|a| a == arch);
    inventory.add_program_nodes(program.id, arch, program.jsonl)?;
    let plan = match plan(&inventory, arch) {
        Ok(plan) => plan,
        Err(e) => return refuse(e.lines().next().unwrap_or_default().to_owned()),
    };
    let Some(cache) = resolve_cache(&opts.cache, arch) else {
        return refuse(format!("no cache entries in {}", opts.cache.display()));
    };
    let toolchain_id = match &opts.toolchain_id {
        Some(id) => id.clone(),
        None => KernelCompiler::local_toolchain_id().unwrap_or_default(),
    };

    // Stage next to `out`; every exit but the final rename removes it.
    let out_name = opts
        .out
        .file_name()
        .ok_or_else(|| format!("{}: --out has no directory name", opts.out.display()))?;
    let parent = match opts.out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_owned(),
        _ => PathBuf::from("."),
    };
    std::fs::create_dir_all(&parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let mut temp_name = out_name.to_owned();
    temp_name.push(format!(".tmp-{}", std::process::id()));
    let staging_path = parent.join(temp_name);
    let _ = std::fs::remove_dir_all(&staging_path);
    std::fs::create_dir_all(&staging_path)
        .map_err(|e| format!("{}: {e}", staging_path.display()))?;
    let staging_buf = staging_path.clone();
    let staging: &Path = &staging_buf;
    let mut temp = TempDir(Some(staging_path));

    // The symbols each module must export for the node sets.
    let sets = inventory.node_sets();
    let mut wanted: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for set in sets.iter().filter(|s| s.arch == arch) {
        for (symbol, module) in set.nodes {
            let symbols = wanted.entry(module.as_str()).or_default();
            if !symbols.contains(&symbol.as_str()) {
                symbols.push(symbol.as_str());
            }
        }
    }
    let lookup = Finder {
        arch,
        extra_flags: &plan.extra_flags,
        toolchain_id: &toolchain_id,
        pack: opts.pack.as_deref(),
        cache: &cache,
    };
    let located: Vec<Result<Located>> = plan
        .entries
        .iter()
        .map(|e| locate(&lookup, e, &wanted[e.module]))
        .collect();
    let certified = run_parallel(&located, opts.jobs, |l| match l {
        Ok(l) => {
            railgun_cert::certify(&l.object, l.radiowave.as_deref()).map_err(|e| e.to_string())
        }
        Err(e) => Err(e.clone()),
    });

    let mut records: Vec<CorpusRecord> = Vec::new();
    let mut rejected: Vec<RejectedObject> = Vec::new();
    // module -> why no object of it is certified.
    let mut failed: BTreeMap<&str, String> = BTreeMap::new();
    let mut exports: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for ((entry, located), receipt) in plan.entries.iter().zip(located).zip(certified) {
        let twin = plan.twins.iter().any(|t| t.module == entry.module);
        let found = match (located, receipt) {
            (Ok(found), Ok(receipt)) => Ok((found, receipt)),
            (Ok(_), Err(reason)) => Err(format!("railgun-cert rejected the object: {reason}")),
            (Err(why), _) => Err(why),
        };
        let (found, receipt) = match found {
            Ok(ok) => ok,
            // The product default loads the embedded twin: its hipcc object is optional.
            Err(why) if twin => {
                eprintln!("{arch}: {}: no hipcc object recorded ({why})", entry.module);
                continue;
            }
            Err(why) => {
                failed.insert(entry.module, why);
                continue;
            }
        };
        let missing: Vec<&str> = entry
            .symbols
            .iter()
            .copied()
            .filter(|s| !receipt.railgun.kernels.iter().any(|k| k.symbol == *s))
            .collect();
        if !missing.is_empty() && !twin {
            failed.insert(
                entry.module,
                format!("object does not export symbol(s) {}", missing.join(", ")),
            );
            continue;
        }
        if !missing.is_empty() {
            continue;
        }
        exports.insert(
            entry.module,
            receipt
                .railgun
                .kernels
                .iter()
                .map(|k| k.symbol.clone())
                .collect(),
        );
        let artifact = sha256_hex(&found.object);
        let symbols: Vec<String> = entry.symbols.iter().map(|s| (*s).to_owned()).collect();
        let pack_dir = staging.join("pack").join(arch);
        let write = |dir: &Path, name: String, bytes: &[u8]| -> Result<()> {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            std::fs::write(dir.join(&name), bytes).map_err(|e| format!("{name}: {e}"))
        };
        write(&pack_dir, format!("{}.hsaco", entry.module), &found.object)?;
        if let Some(json) = &found.radiowave {
            write(
                &pack_dir,
                format!("{}.radiowave.json", entry.module),
                json.as_bytes(),
            )?;
        }
        if let Some((hash, index)) = &found.pack_meta {
            write(&pack_dir, format!("{}.hash", entry.module), hash)?;
            write(&pack_dir, format!("{}.index.json", entry.module), index)?;
        }
        write(
            &staging.join("sources").join(arch),
            format!("{}.hip", entry.module),
            entry.source().as_bytes(),
        )?;
        let receipt_rel = format!("receipts/{arch}/{artifact}.receipt.json");
        let receipt_bytes = write_json(&staging.join(&receipt_rel), &receipt)?;
        records.push(CorpusRecord {
            arch: arch.to_owned(),
            artifact_sha256: artifact,
            module: entry.module.to_owned(),
            origin: ObjectOrigin::Hipcc,
            symbols,
            source_sha256: sha256_hex(entry.source().as_bytes()),
            extra_flags: plan.extra_flags.clone(),
            flags: entry.flags.clone(),
            scheduler_profile: entry.scheduler_profile.clone().unwrap_or_default(),
            packaging_key: KernelCompiler::packaging_hash_for(
                arch,
                entry.module,
                entry.source(),
                &plan.extra_flags,
            ),
            jit_cache_key: KernelCompiler::jit_cache_key_for(
                arch,
                entry.module,
                entry.source(),
                &plan.extra_flags,
                &toolchain_id,
            ),
            argv: Vec::new(),
            object: format!("pack/{arch}/{}.hsaco", entry.module),
            receipt: receipt_rel,
            receipt_sha256: sha256_hex(&receipt_bytes),
            kernels: receipt
                .railgun
                .kernels
                .iter()
                .map(|k| KernelVerdict {
                    symbol: k.symbol.clone(),
                    status: k.status.clone(),
                    differential: k.differential.status.clone(),
                })
                .collect(),
        });
    }
    // Embedded native objects, by digest.
    certify_natives(
        &opts.arch,
        &plan,
        opts.jobs,
        staging,
        &mut records,
        &mut rejected,
    )?;

    // Every node must resolve to a certified object, in node order.
    let mut missing: Vec<MissingObject> = Vec::new();
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for set in sets.iter().filter(|s| s.arch == arch) {
        for (symbol, module) in set.nodes {
            if !seen.insert((symbol, module)) {
                continue;
            }
            let why = if let Some(r) = rejected.iter().find(|r| &r.module == module) {
                Some(format!(
                    "railgun-cert rejected the embedded object: {}",
                    r.reason
                ))
            } else if records
                .iter()
                .any(|r| r.origin == ObjectOrigin::PmNative && &r.module == module)
            {
                None
            } else if let Some(why) = failed.get(module.as_str()) {
                Some(why.clone())
            } else if exports
                .get(module.as_str())
                .is_some_and(|e| !e.contains(symbol))
            {
                Some(format!("symbol {symbol} is not exported by the object"))
            } else if exports.contains_key(module.as_str()) {
                None
            } else {
                Some("no certified object".to_owned())
            };
            if let Some(why) = why {
                missing.push(MissingObject {
                    symbol: symbol.clone(),
                    module: module.clone(),
                    why,
                });
            }
        }
    }
    if let Some(first) = missing.first() {
        return Ok(FromCacheOutcome::Refused(Refusal {
            reason: format!(
                "node {} (module {}): {}",
                first.symbol, first.module, first.why
            ),
            missing,
        }));
    }

    let programs = program_records(&inventory, &[arch.to_owned()], &records, &rejected);
    let toolchain = ToolchainPin {
        pin: sha256_hex(toolchain_id.as_bytes()),
        id: toolchain_id.clone(),
        version_sha256: String::new(),
    };
    let mut build_host = BuildHost {
        key: String::new(),
        hostname: hostname(),
        hipcc: "from-cache".into(),
        rocm_env_root: None,
        clang_sha256: "from-cache".into(),
        header_set_sha256: "from-cache".into(),
        build_dir: "from-cache".into(),
    };
    build_host.key = sha256_hex(
        &serde_json::to_vec(&(&build_host, &toolchain.version_sha256))
            .map_err(|e| e.to_string())?,
    );
    let pin = toolchain.pin.clone();
    let index = CorpusIndex {
        schema: CORPUS_SCHEMA.into(),
        version: CORPUS_VERSION,
        source_commit: git_revision(),
        build_host,
        toolchain,
        records,
        rejected,
        programs,
    };
    let index_bytes = write_json(&staging.join(CORPUS_INDEX), &index)?;

    let counts = verify_corpus(staging, &inventory, arch, program.id)?;
    publish(
        staging,
        &opts.out,
        &index_bytes,
        &inventory,
        arch,
        program.id,
    )?;
    // `publish` consumed the staging dir (renamed, or discarded as identical).
    temp.0 = None;
    let distinct: BTreeSet<&String> = index.records.iter().map(|r| &r.artifact_sha256).collect();
    Ok(FromCacheOutcome::Ok(FromCacheOk {
        program: program.id.to_owned(),
        arch: arch.to_owned(),
        out: opts.out.clone(),
        toolchain_pin: pin,
        nodes: counts.nodes,
        objects: distinct.len(),
        proven: counts.proven,
        conditional: counts.conditional,
        unknown: counts.unknown,
        millis: started.elapsed().as_millis(),
    }))
}

/// `JitCorpus::open` + the CI `check` over the program's nodes (and railgun's
/// own kernels): every node must resolve to a verified record.
fn verify_corpus(dir: &Path, inventory: &Inventory, arch: &str, program: &str) -> Result<Coverage> {
    let corpus = JitCorpus::open(dir).map_err(|e| format!("corpus rejected: {e}"))?;
    let programs: Vec<ProgramKeys> = current_program_keys(inventory)?
        .into_iter()
        .filter(|p| p.arch == arch)
        .collect();
    let coverage = check(&corpus, &programs);
    if let Some(m) = coverage.iter().flat_map(|c| &c.missing).next() {
        return Err(format!(
            "from-cache corpus failed its own check: node {}: {}",
            m.symbol, m.reason
        ));
    }
    coverage
        .into_iter()
        .find(|c| c.program == program)
        .ok_or_else(|| format!("program {program} missing from the corpus check"))
}

/// Move the verified `staging` corpus to `out` (see the module docs).
fn publish(
    staging: &Path,
    out: &Path,
    index_bytes: &[u8],
    inventory: &Inventory,
    arch: &str,
    program: &str,
) -> Result<()> {
    let identical = |out: &Path| {
        std::fs::read(out.join(CORPUS_INDEX)).is_ok_and(|b| b == index_bytes)
            && verify_corpus(out, inventory, arch, program).is_ok()
    };
    if out.exists() {
        if identical(out) {
            return std::fs::remove_dir_all(staging)
                .map_err(|e| format!("{}: {e}", staging.display()));
        }
        let mut aside = out.as_os_str().to_owned();
        aside.push(format!(".old-{}", std::process::id()));
        let aside = PathBuf::from(aside);
        let _ = std::fs::remove_dir_all(&aside);
        std::fs::rename(out, &aside).map_err(|e| format!("{}: {e}", out.display()))?;
        if let Err(e) = std::fs::rename(staging, out) {
            let _ = std::fs::rename(&aside, out);
            return Err(format!("{}: {e}", out.display()));
        }
        let _ = std::fs::remove_dir_all(&aside);
        return Ok(());
    }
    match std::fs::rename(staging, out) {
        Ok(()) => Ok(()),
        // A concurrent run published first: fine when it is the same corpus.
        Err(_) if identical(out) => {
            std::fs::remove_dir_all(staging).map_err(|e| format!("{}: {e}", staging.display()))
        }
        Err(e) => Err(format!("{}: {e}", out.display())),
    }
}
