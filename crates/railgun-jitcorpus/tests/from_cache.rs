// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `from-cache`: a corpus over the user's own objects. The objects are the real
//! gfx1201 `railgun_copy` / `railgun_fill` hipcc objects with their Radiowave
//! manifests (`tests/fixtures/from-cache`); the pack index and hot-cache names
//! are generated here from the runtime's own key functions, so no fixture goes
//! stale with a registry change. A miniature program table (the railgun copy and
//! fill nodes, one embedded `pm_decode` twin node and the sealed embedded Raon
//! `gated_delta_vector_pm` node) stands in for the real program; the real
//! program list is checked against the registry separately. CPU only.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use railgun_corpus::{sha256_hex, ArtifactKey, JitCorpus, ObjectOrigin, CORPUS_INDEX};
use railgun_jitcorpus::{
    check, current_program_keys, default_extra_flags, from_cache, from_cache_program,
    install_product_defaults, inventory, EmbeddedProgram, FromCacheOk, FromCacheOptions,
    FromCacheOutcome, Refusal, EMBEDDED_PROGRAMS,
};
use rdna_compute::kernel_pack::KERNEL_CACHE_ABI;
use rdna_compute::kernel_registry::{self, KernelEntry};
use rdna_compute::KernelCompiler;

const ARCH: &str = "gfx1201";
const TOOLCHAIN: &str = "HIP version: from-cache-test";
const COPY: &[u8] = include_bytes!("fixtures/from-cache/railgun_copy.hsaco");
const COPY_RW: &str = include_str!("fixtures/from-cache/railgun_copy.radiowave.json");
const FILL: &[u8] = include_bytes!("fixtures/from-cache/railgun_fill.hsaco");
const FILL_RW: &str = include_str!("fixtures/from-cache/railgun_fill.radiowave.json");

/// railgun's copy and fill, a `pm_decode` twin node (its hipcc object absent from
/// the cache: the product default loads the embedded image) and the sealed Raon
/// embedded image.
const MINI: EmbeddedProgram = EmbeddedProgram {
    id: "mini-from-cache-gfx1201",
    arch: ARCH,
    jsonl: concat!(
        "# miniature program table for the from-cache tests\n",
        "{\"kind\":\"kernel\",\"symbol\":\"railgun_copy\",\"module\":\"railgun_copy\"}\n",
        "{\"kind\":\"kernel\",\"symbol\":\"railgun_copy_batch\",\"module\":\"railgun_copy\"}\n",
        "{\"kind\":\"kernel\",\"symbol\":\"railgun_fill_u32\",\"module\":\"railgun_fill\"}\n",
        "{\"kind\":\"kernel\",\"symbol\":\"fused_qkv_mq4g256v2\",\"module\":\"fused_qkv_hfq4g256_mq4v2\"}\n",
        "{\"kind\":\"fill\"}\n",
        "{\"kind\":\"kernel\",\"symbol\":\"causal_conv_depthwise_silu_rows_f32_pm_gfx1201\",\"module\":\"gated_delta_vector_pm_gfx1201\"}\n",
    ),
};

fn entry(module: &str) -> KernelEntry {
    install_product_defaults().unwrap();
    kernel_registry::corpus_entries(ARCH, &default_extra_flags(ARCH))
        .unwrap()
        .into_iter()
        .find(|e| e.module == module)
        .unwrap_or_else(|| panic!("registry module {module}"))
}

fn packaging_key(module: &str) -> String {
    let e = entry(module);
    KernelCompiler::packaging_hash_for(ARCH, module, e.source(), &default_extra_flags(ARCH))
}

fn jit_key(module: &str, toolchain: &str) -> String {
    let e = entry(module);
    KernelCompiler::jit_cache_key_for(
        ARCH,
        module,
        e.source(),
        &default_extra_flags(ARCH),
        toolchain,
    )
}

fn write(path: &Path, bytes: impl AsRef<[u8]>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// One installed pack object the way `publish_package` writes it.
fn pack_module(dir: &Path, module: &str, object: &[u8], radiowave: &str, key: &str) {
    let e = entry(module);
    let recipe =
        KernelCompiler::recipe_for_source(ARCH, module, e.source(), &default_extra_flags(ARCH));
    let index = serde_json::json!({
        "version": 1,
        "module": module,
        "symbols": e.symbols,
        "arch": ARCH,
        "source_sha256": sha256_hex(e.source().as_bytes()),
        "flags": recipe.flags,
        "scheduler_profile": recipe.scheduler_profile.unwrap_or_default(),
        "cache_abi": KERNEL_CACHE_ABI,
        "packaging_key": key,
        "object_sha256": sha256_hex(object),
        "toolchain_id": TOOLCHAIN,
        "toolchain_sha256": sha256_hex(TOOLCHAIN.as_bytes()),
    });
    write(&dir.join(format!("{module}.hsaco")), object);
    write(&dir.join(format!("{module}.hash")), key);
    write(
        &dir.join(format!("{module}.index.json")),
        serde_json::to_vec_pretty(&index).unwrap(),
    );
    write(&dir.join(format!("{module}.radiowave.json")), radiowave);
}

/// One hot-cache object under the runtime's content key.
fn hot_module(dir: &Path, module: &str, object: &[u8], radiowave: &str) {
    let key = jit_key(module, TOOLCHAIN);
    write(&dir.join(format!("{module}.{key}.hsaco")), object);
    write(
        &dir.join(format!("{module}.{key}.radiowave.json")),
        radiowave,
    );
}

struct Env {
    root: PathBuf,
    pack: PathBuf,
    /// The cache root: the hot objects live in `<root>/gfx1201`.
    cache: PathBuf,
    out: PathBuf,
}

impl Env {
    fn new(name: &str) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("from-cache-{name}"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self {
            pack: root.join("pack").join(ARCH),
            cache: root.join("hot"),
            out: root.join("deep").join("out"),
            root,
        }
    }

    fn hot(&self) -> PathBuf {
        self.cache.join(ARCH)
    }

    /// copy from the pack, fill from the hot cache: both sources in one corpus.
    fn standard(&self) {
        pack_module(
            &self.pack,
            "railgun_copy",
            COPY,
            COPY_RW,
            &packaging_key("railgun_copy"),
        );
        hot_module(&self.hot(), "railgun_fill", FILL, FILL_RW);
    }

    fn options(&self) -> FromCacheOptions {
        FromCacheOptions {
            arch: ARCH.into(),
            program: MINI.id.into(),
            cache: self.cache.clone(),
            pack: Some(self.pack.clone()),
            out: self.out.clone(),
            jobs: 4,
            toolchain_id: Some(TOOLCHAIN.into()),
        }
    }

    fn run(&self) -> FromCacheOutcome {
        from_cache_program(&self.options(), &MINI).unwrap_or_else(|e| panic!("from-cache: {e}"))
    }

    fn ok(&self) -> FromCacheOk {
        match self.run() {
            FromCacheOutcome::Ok(ok) => ok,
            FromCacheOutcome::Refused(r) => panic!("refused: {r:?}"),
        }
    }

    fn refused(&self) -> Refusal {
        match self.run() {
            FromCacheOutcome::Refused(r) => r,
            FromCacheOutcome::Ok(ok) => panic!("certified: {ok:?}"),
        }
    }

    /// No staging, aside or temp sibling is left next to `out`.
    fn assert_no_leftovers(&self) {
        let parent = self.out.parent().unwrap();
        let name = self.out.file_name().unwrap().to_string_lossy().into_owned();
        let leftovers: Vec<String> = std::fs::read_dir(parent)
            .map(|d| {
                d.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                    .filter(|n| n.starts_with(&name) && n != &name)
                    .collect()
            })
            .unwrap_or_default();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    }
}

/// Every file under `dir` with its bytes.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path.strip_prefix(root).unwrap().display().to_string();
                out.insert(rel, std::fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn mini_inventory() -> railgun_jitcorpus::Inventory {
    install_product_defaults().unwrap();
    let mut inv = inventory().unwrap();
    inv.programs.clear();
    inv.railgun_kernels.archs.retain(|a| a == ARCH);
    inv.add_program_nodes(MINI.id, ARCH, MINI.jsonl).unwrap();
    inv
}

#[test]
fn certifies_pack_and_hot_objects_like_build_and_opens_and_checks() {
    let env = Env::new("parity");
    env.standard();
    let ok = env.ok();
    assert_eq!(ok.program, MINI.id);
    assert_eq!(ok.arch, ARCH);
    assert_eq!(ok.out, env.out);
    assert_eq!(ok.toolchain_pin, sha256_hex(TOOLCHAIN.as_bytes()));
    // The mini program's 5 distinct (symbol, module) nodes in 4 objects.
    assert_eq!(ok.nodes, 5);
    // copy, fill, the twin's embedded image and the sealed embedded image.
    assert_eq!(ok.objects, 4);
    assert!(ok.proven + ok.conditional + ok.unknown >= ok.nodes - 1);

    let corpus = JitCorpus::open(&env.out).expect("JitCorpus::open");
    let index = corpus.index();
    assert_eq!(index.toolchain.pin, sha256_hex(TOOLCHAIN.as_bytes()));
    assert_eq!(index.toolchain.id, TOOLCHAIN);
    assert!(index.rejected.is_empty());

    // `build`'s record for these objects, derived independently of from-cache.
    for (module, object, sidecar) in [
        ("railgun_copy", COPY, COPY_RW),
        ("railgun_fill", FILL, FILL_RW),
    ] {
        let e = entry(module);
        let record = corpus
            .record_for_module(ARCH, module, &packaging_key(module))
            .unwrap_or_else(|| panic!("record for {module}"));
        let receipt = railgun_cert::certify(object, Some(sidecar)).unwrap();
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        assert_eq!(record.arch, ARCH);
        assert_eq!(record.artifact_sha256, sha256_hex(object));
        assert_eq!(record.origin, ObjectOrigin::Hipcc);
        assert_eq!(record.symbols, e.symbols);
        assert_eq!(record.source_sha256, sha256_hex(e.source().as_bytes()));
        assert_eq!(record.extra_flags, default_extra_flags(ARCH));
        assert_eq!(record.flags, e.flags);
        assert_eq!(
            record.scheduler_profile,
            e.scheduler_profile.clone().unwrap_or_default()
        );
        assert_eq!(record.packaging_key, packaging_key(module));
        assert_eq!(record.jit_cache_key, jit_key(module, TOOLCHAIN));
        assert!(record.argv.is_empty(), "no compiler was invoked");
        assert_eq!(record.object, format!("pack/{ARCH}/{module}.hsaco"));
        assert_eq!(
            record.receipt,
            format!("receipts/{ARCH}/{}.receipt.json", sha256_hex(object))
        );
        assert_eq!(record.receipt_sha256, sha256_hex(&receipt_bytes));
        assert_eq!(
            std::fs::read(env.out.join(&record.receipt)).unwrap(),
            receipt_bytes
        );
        assert_eq!(std::fs::read(env.out.join(&record.object)).unwrap(), object);
        let verdicts: Vec<_> = receipt
            .railgun
            .kernels
            .iter()
            .map(|k| {
                (
                    k.symbol.clone(),
                    k.status.clone(),
                    k.differential.status.clone(),
                )
            })
            .collect();
        let recorded: Vec<_> = record
            .kernels
            .iter()
            .map(|k| (k.symbol.clone(), k.status.clone(), k.differential.clone()))
            .collect();
        assert_eq!(recorded, verdicts, "{module} verdicts");
        assert_eq!(
            std::fs::read_to_string(env.out.join(format!("pack/{ARCH}/{module}.radiowave.json")))
                .unwrap(),
            sidecar
        );
        assert_eq!(
            std::fs::read_to_string(env.out.join(format!("sources/{ARCH}/{module}.hip"))).unwrap(),
            e.source()
        );
    }
    // The pack-sourced object travels verbatim with its index and hash.
    for suffix in ["hash", "index.json"] {
        assert_eq!(
            std::fs::read(env.out.join(format!("pack/{ARCH}/railgun_copy.{suffix}"))).unwrap(),
            std::fs::read(env.pack.join(format!("railgun_copy.{suffix}"))).unwrap(),
            "{suffix}"
        );
    }
    assert!(!env
        .out
        .join(format!("pack/{ARCH}/railgun_fill.index.json"))
        .exists());

    // Programs: railgun's own kernels, then the mini program, every node resolved.
    let ids: Vec<&str> = index.programs.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["railgun-kernels-gfx1201", MINI.id]);
    let mini = index.programs.iter().find(|p| p.id == MINI.id).unwrap();
    assert_eq!(mini.nodes.len(), 5);
    assert_eq!(mini.nodes[0].artifact_sha256, sha256_hex(COPY));
    assert_eq!(mini.nodes[2].artifact_sha256, sha256_hex(FILL));

    // The CI check passes over the whole inventory the corpus was built for.
    let programs = current_program_keys(&mini_inventory()).unwrap();
    for coverage in check(&corpus, &programs) {
        assert!(coverage.missing.is_empty(), "{coverage:?}");
    }
}

#[test]
fn pm_native_objects_are_certified_by_image_digest_with_no_hipcc_object_for_the_twin() {
    let env = Env::new("native");
    env.standard();
    env.ok();
    let corpus = JitCorpus::open(&env.out).unwrap();
    let inv = mini_inventory();
    let plan = railgun_jitcorpus::plan(&inv, ARCH).unwrap();
    assert_eq!(plan.twins.len(), 1);
    assert_eq!(plan.embedded.len(), 1);
    let twin = plan.twins[0];
    let twin_sha: String = twin
        .image_sha256
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let embedded = &plan.embedded[0];
    let pin = corpus.toolchain_pin().to_owned();
    for (module, sha) in [
        (twin.module, twin_sha.as_str()),
        (embedded.module, embedded.image_sha256),
    ] {
        let record = corpus
            .record(&ArtifactKey {
                arch: ARCH,
                toolchain_pin: &pin,
                artifact_sha256: sha,
            })
            .unwrap_or_else(|| panic!("pm_native record for {module}"));
        assert_eq!(record.origin, ObjectOrigin::PmNative);
        assert_eq!(record.module, module);
        assert_eq!(
            sha256_hex(&std::fs::read(env.out.join(&record.object)).unwrap()),
            sha
        );
    }
    // No hipcc record of the twin module: the cache held none and none is needed.
    assert!(corpus
        .index()
        .records
        .iter()
        .all(|r| r.module != twin.module || r.origin == ObjectOrigin::PmNative));
}

#[test]
fn cache_root_and_arch_dir_forms_give_identical_corpora() {
    let root_form = Env::new("form-root");
    root_form.standard();
    root_form.ok();

    let dir_form = Env::new("form-dir");
    dir_form.standard();
    let mut opts = dir_form.options();
    opts.cache = dir_form.hot();
    match from_cache_program(&opts, &MINI).unwrap() {
        FromCacheOutcome::Ok(_) => {}
        FromCacheOutcome::Refused(r) => panic!("arch-dir form refused: {r:?}"),
    }
    assert_eq!(
        std::fs::read(root_form.out.join(CORPUS_INDEX)).unwrap(),
        std::fs::read(dir_form.out.join(CORPUS_INDEX)).unwrap()
    );
    assert_eq!(snapshot(&root_form.out), snapshot(&dir_form.out));
}

#[test]
fn the_toolchain_pin_is_the_runtimes_even_without_a_device_compiler() {
    // The probe (`hipcc --version` first line, empty without a compiler) is what
    // `WindowCorpus::open` hashes; hot keys are computed under the same id.
    install_product_defaults().unwrap(); // before anything reads the process configuration
    let local = KernelCompiler::local_toolchain_id().unwrap_or_default();
    let env = Env::new("pin");
    pack_module(
        &env.pack,
        "railgun_copy",
        COPY,
        COPY_RW,
        &packaging_key("railgun_copy"),
    );
    hot_module_under(&env.hot(), "railgun_fill", FILL, FILL_RW, &local);
    let mut opts = env.options();
    opts.toolchain_id = None;
    // A pack built by another compiler is rejected by the loader when this host
    // has one; with none it loads verbatim. Either way the corpus pins `local`.
    let outcome = from_cache_program(&opts, &MINI).unwrap();
    match outcome {
        FromCacheOutcome::Ok(ok) => assert_eq!(ok.toolchain_pin, sha256_hex(local.as_bytes())),
        FromCacheOutcome::Refused(r) => {
            assert!(
                !local.is_empty() && r.reason.contains("toolchain mismatch"),
                "{r:?}"
            );
        }
    }
}

fn hot_module_under(dir: &Path, module: &str, object: &[u8], radiowave: &str, toolchain: &str) {
    let key = jit_key(module, toolchain);
    write(&dir.join(format!("{module}.{key}.hsaco")), object);
    write(
        &dir.join(format!("{module}.{key}.radiowave.json")),
        radiowave,
    );
}

#[test]
fn a_legacy_hot_pair_is_accepted_only_when_its_hash_is_the_current_key() {
    let env = Env::new("legacy");
    pack_module(
        &env.pack,
        "railgun_copy",
        COPY,
        COPY_RW,
        &packaging_key("railgun_copy"),
    );
    let hot = env.hot();
    write(&hot.join("railgun_fill.hsaco"), FILL);
    write(&hot.join("railgun_fill.radiowave.json"), FILL_RW);
    write(&hot.join("railgun_fill.hash"), "0000000000000000");
    let key = jit_key("railgun_fill", TOOLCHAIN);
    let first = env.refused();
    assert_eq!(
        first.reason,
        why(
            "railgun_fill",
            "railgun_fill_u32",
            &format!("not in the pack; no hot cache entry railgun_fill.{key}.hsaco under the runtime's current key")
        )
    );
    write(&hot.join("railgun_fill.hash"), format!("{key}\n"));
    env.ok();
}

fn why(module: &str, symbol: &str, text: &str) -> String {
    format!("node {symbol} (module {module}): {text}")
}

#[test]
fn refuses_a_missing_object_naming_its_node_and_key() {
    let env = Env::new("missing");
    pack_module(
        &env.pack,
        "railgun_copy",
        COPY,
        COPY_RW,
        &packaging_key("railgun_copy"),
    );
    // The cache holds entries, just not the fill module's.
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    let r = env.refused();
    let key = jit_key("railgun_fill", TOOLCHAIN);
    assert_eq!(
        r.reason,
        why(
            "railgun_fill",
            "railgun_fill_u32",
            &format!("not in the pack; no hot cache entry railgun_fill.{key}.hsaco under the runtime's current key")
        )
    );
    // The fill node is in both node sets and named once.
    let nodes: Vec<(&str, &str)> = r
        .missing
        .iter()
        .map(|m| (m.symbol.as_str(), m.module.as_str()))
        .collect();
    assert_eq!(nodes, [("railgun_fill_u32", "railgun_fill")]);
    assert!(!env.out.exists(), "nothing published");
    env.assert_no_leftovers();
}

#[test]
fn refuses_a_stale_packaging_key_and_never_uses_the_pack_object() {
    let env = Env::new("stale");
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    // An index whose packaging key is not the runtime's current key for the source and recipe.
    pack_module(&env.pack, "railgun_fill", FILL, FILL_RW, "0123456789abcdef");
    let r = env.refused();
    let key = jit_key("railgun_fill", TOOLCHAIN);
    let index_path = env.pack.join("railgun_fill.index.json");
    assert_eq!(
        r.reason,
        why(
            "railgun_fill",
            "railgun_fill_u32",
            &format!(
                "pack object rejected ({}: index mismatch: packaging key differs); no hot cache entry railgun_fill.{key}.hsaco under the runtime's current key",
                index_path.display()
            )
        )
    );
    assert!(!env.out.exists());
    env.assert_no_leftovers();
}

#[test]
fn refuses_a_pack_object_whose_bytes_do_not_match_its_index() {
    let env = Env::new("tampered");
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    pack_module(
        &env.pack,
        "railgun_fill",
        FILL,
        FILL_RW,
        &packaging_key("railgun_fill"),
    );
    write(&env.pack.join("railgun_fill.hsaco"), COPY);
    let r = env.refused();
    assert!(r.reason.contains("object SHA-256 mismatch"), "{}", r.reason);
}

#[test]
fn refuses_an_object_the_certifier_rejects_with_its_reason() {
    let env = Env::new("garbage");
    let garbage = b"not a code object";
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    hot_module(&env.hot(), "railgun_fill", garbage, "{}");
    let reason = railgun_cert::certify(garbage, Some("{}"))
        .unwrap_err()
        .to_string();
    let r = env.refused();
    assert_eq!(
        r.reason,
        why(
            "railgun_fill",
            "railgun_fill_u32",
            &format!("railgun-cert rejected the object: {reason}")
        )
    );
    assert!(!env.out.exists());
}

#[test]
fn refuses_an_object_that_does_not_export_a_node_symbol() {
    let env = Env::new("symbol");
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    // The copy object is a valid certified object, but it has no `railgun_fill_u32`.
    hot_module(&env.hot(), "railgun_fill", COPY, COPY_RW);
    let r = env.refused();
    assert_eq!(
        r.reason,
        why(
            "railgun_fill",
            "railgun_fill_u32",
            "object does not export symbol(s) railgun_fill_u32"
        )
    );
    assert!(!env.out.exists());
}

#[test]
fn refuses_unknown_program_bad_arch_and_an_empty_cache_with_exact_reasons() {
    let env = Env::new("global");
    env.standard();

    let mut opts = env.options();
    opts.program = "no-such-program".into();
    let FromCacheOutcome::Refused(r) = from_cache(&opts).unwrap() else {
        panic!("unknown program certified")
    };
    let known: Vec<&str> = EMBEDDED_PROGRAMS.iter().map(|p| p.id).collect();
    assert_eq!(
        r.reason,
        format!(
            "unknown program id `no-such-program` (known: {})",
            known.join(", ")
        )
    );
    assert!(known.contains(&"raon-ar-decode-gfx1201"));

    let mut opts = env.options();
    opts.program = "raon-ar-decode-gfx1201".into();
    opts.arch = "gfx1100".into();
    let FromCacheOutcome::Refused(r) = from_cache(&opts).unwrap() else {
        panic!("wrong arch certified")
    };
    assert_eq!(
        r.reason,
        "arch `gfx1100` is not the arch `gfx1201` of program raon-ar-decode-gfx1201"
    );

    let empty = env.root.join("empty-cache");
    std::fs::create_dir_all(empty.join(ARCH)).unwrap();
    let mut opts = env.options();
    opts.cache = empty.clone();
    let FromCacheOutcome::Refused(r) = from_cache_program(&opts, &MINI).unwrap() else {
        panic!("empty cache certified")
    };
    assert_eq!(r.reason, format!("no cache entries in {}", empty.display()));
    opts.cache = env.root.join("does-not-exist");
    let FromCacheOutcome::Refused(r) = from_cache_program(&opts, &MINI).unwrap() else {
        panic!("missing cache certified")
    };
    assert_eq!(
        r.reason,
        format!("no cache entries in {}", opts.cache.display())
    );
    assert!(!env.out.exists());
    env.assert_no_leftovers();
}

#[test]
fn a_second_run_is_identical_and_a_refused_run_leaves_an_existing_out_untouched() {
    let env = Env::new("atomic");
    env.standard();
    // A second hot entry keeps the cache non-empty once the fill object is removed.
    hot_module(&env.hot(), "railgun_copy", COPY, COPY_RW);
    env.ok();
    let first = snapshot(&env.out);
    env.assert_no_leftovers();

    // Same inputs: same corpus bytes, nothing left behind.
    env.ok();
    assert_eq!(snapshot(&env.out), first);
    env.assert_no_leftovers();

    // A refused run (the fill object is gone) never touches the published corpus.
    let key = jit_key("railgun_fill", TOOLCHAIN);
    std::fs::remove_file(env.hot().join(format!("railgun_fill.{key}.hsaco"))).unwrap();
    let refused = env.refused();
    assert!(
        refused.reason.contains("railgun_fill"),
        "{}",
        refused.reason
    );
    assert_eq!(snapshot(&env.out), first);
    env.assert_no_leftovers();

    // A damaged published corpus is replaced whole by a good run.
    hot_module(&env.hot(), "railgun_fill", FILL, FILL_RW);
    std::fs::write(env.out.join(CORPUS_INDEX), b"{}").unwrap();
    std::fs::remove_file(env.out.join(format!("pack/{ARCH}/railgun_copy.hsaco"))).unwrap();
    env.ok();
    assert_eq!(snapshot(&env.out), first);
    env.assert_no_leftovers();
    JitCorpus::open(&env.out).unwrap();
}

#[test]
fn an_existing_temp_sibling_of_a_dead_run_is_replaced_not_trusted() {
    let env = Env::new("stale-temp");
    env.standard();
    let temp = env
        .out
        .with_file_name(format!("out.tmp-{}", std::process::id()));
    write(&temp.join("junk"), "left by a crashed run");
    env.ok();
    assert!(!temp.exists());
    assert!(!env.out.join("junk").exists());
}

fn cli(args: &[&str]) -> (i32, Vec<String>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_railgun-jit-corpus"))
        .arg("from-cache")
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    (
        output.status.code().unwrap(),
        stdout.lines().map(str::to_owned).collect(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn the_cli_prints_one_json_status_line_and_exits_2_on_refusal_and_1_on_error() {
    let env = Env::new("cli");
    let cache = env.cache.to_string_lossy().into_owned();
    let out = env.out.to_string_lossy().into_owned();
    std::fs::create_dir_all(env.hot()).unwrap();

    let (code, lines, _) = cli(&[
        "--arch",
        ARCH,
        "--program",
        "nope",
        "--cache",
        &cache,
        "--out",
        &out,
    ]);
    assert_eq!(code, 2);
    assert_eq!(lines.len(), 1, "{lines:?}");
    let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(v["status"], "refused");
    assert_eq!(
        v["reason"],
        "unknown program id `nope` (known: raon-ar-decode-gfx1201)"
    );
    assert_eq!(v["missing"], serde_json::json!([]));
    assert!(v["out"].is_null());

    let (code, lines, _) = cli(&[
        "--arch",
        ARCH,
        "--program",
        "raon-ar-decode-gfx1201",
        "--cache",
        &cache,
        "--out",
        &out,
    ]);
    assert_eq!(code, 2);
    let v: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(v["status"], "refused");
    assert_eq!(v["reason"], format!("no cache entries in {cache}"));
    assert!(!env.out.exists());

    let (code, lines, _) = cli(&["--arch", ARCH]);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(v["status"], "error");
    assert_eq!(v["reason"], "from-cache needs --program");
}

#[test]
fn a_node_with_one_object_missing_is_named_with_symbol_module_and_why_in_the_cli_line() {
    let env = Env::new("cli-missing");
    install_product_defaults().unwrap(); // before anything reads the process configuration
    let local = KernelCompiler::local_toolchain_id().unwrap_or_default();
    hot_module_under(&env.hot(), "railgun_copy", COPY, COPY_RW, &local);
    let (code, lines, _) = cli(&[
        "--arch",
        ARCH,
        "--program",
        "raon-ar-decode-gfx1201",
        "--cache",
        &env.cache.to_string_lossy(),
        "--out",
        &env.out.to_string_lossy(),
    ]);
    assert_eq!(code, 2);
    assert_eq!(lines.len(), 1);
    let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    let missing = v["missing"].as_array().unwrap();
    // Every module of the real program but the copy kernels is uncovered.
    assert!(missing.len() > 20, "{}", missing.len());
    let first = &missing[0];
    assert_eq!(first["symbol"], "railgun_fill_u32");
    assert_eq!(first["module"], "railgun_fill");
    assert!(first["why"]
        .as_str()
        .unwrap()
        .starts_with("no hot cache entry railgun_fill."));
    assert_eq!(
        v["reason"],
        format!(
            "node railgun_fill_u32 (module railgun_fill): {}",
            first["why"].as_str().unwrap()
        )
    );
}

/// The embedded real program list is the recorded probe sequence's node set and
/// every one of its modules resolves in the registry (so `from-cache` can key it).
#[test]
fn the_embedded_raon_program_is_the_recorded_sequences_node_set() {
    install_product_defaults().unwrap();
    let embedded = EMBEDDED_PROGRAMS
        .iter()
        .find(|p| p.id == "raon-ar-decode-gfx1201")
        .unwrap();
    assert_eq!(embedded.arch, ARCH);
    let mut from_list = inventory().unwrap();
    from_list
        .add_program_nodes(embedded.id, ARCH, embedded.jsonl)
        .unwrap();
    let mut from_fixture = inventory().unwrap();
    from_fixture
        .add_program_sequence(
            embedded.id,
            ARCH,
            include_str!("fixtures/raon-ar-decode-nodes.jsonl"),
        )
        .unwrap();
    assert_eq!(
        from_list.sequence_sets[0].nodes,
        from_fixture.sequence_sets[0].nodes
    );
    assert_eq!(from_list.sequence_sets[0].nodes.len(), 32);
    assert!(from_list.sequence_sets[0].loaded_sha256.is_empty());
    // A plain list needs no digest: a node line without one parses.
    let mut inv = inventory().unwrap();
    inv.add_program_nodes(
        "x",
        ARCH,
        "{\"kind\":\"kernel\",\"symbol\":\"railgun_copy\",\"module\":\"railgun_copy\"}\n",
    )
    .unwrap();
    assert!(inv
        .node_sets()
        .iter()
        .any(|s| s.id == "x" && s.loaded_sha256.is_none()));
    let plan = railgun_jitcorpus::plan(&from_list, ARCH).unwrap();
    for (_, module) in &from_list.sequence_sets[0].nodes {
        assert!(
            plan.entries.iter().any(|e| &e.module == module)
                || plan.embedded.iter().any(|n| &n.module == module),
            "{module}"
        );
    }
}
