//! CPU-only launcher ABI gate for every kernel changed by the MoE router
//! non-finite guard (commit 38ce67a44).
//!
//! For each live production launcher of a changed HIP kernel or PeaceMaker
//! image, the launch site's real `kernelParams` array and raw `KernargBlob`
//! push sequence are extracted from the Rust source and compared with the ABI
//! the kernel itself declares: the `__global__` signature (read by an
//! independent preprocessor/parser, with the Rust-side `#define` prefixes the
//! JIT really sees) or the HSA metadata embedded in the PM image. Argument
//! count, per-argument offset/size/class and the raw end offset must all agree.
//! Nothing here uses GPU, hipcc or the runtime.
//!
//! Coverage is closed: every function that mentions a changed kernel must be a
//! checked launcher, a source-checked delegation to ABI-checked sites, an explicitly
//! classified non-launcher (verified to have no launch site), an inert
//! `#[cfg(any())]` copy, or test code; every changed symbol must be reached by a
//! checked launcher or be listed as unlaunched in
//! `fixtures/all-launcher-abi/inventory.json`. Unknown syntax fails closed.

mod check;
mod decls;
mod flow;
mod hip;
mod layout;
mod lex;
mod scan;

use check::*;
use decls::*;
use hip::Decl;
use scan::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

// ---- inventory --------------------------------------------------------------

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn load_inventory() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/all-launcher-abi/inventory.json");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn strings(v: &Value, key: &str) -> Vec<String> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("inventory key `{key}` must be an array"))
        .iter()
        .map(|s| {
            s.as_str()
                .unwrap_or_else(|| panic!("inventory `{key}` entries must be strings"))
                .to_string()
        })
        .collect()
}

fn string_map(v: &Value, key: &str) -> BTreeMap<String, String> {
    v[key]
        .as_object()
        .unwrap_or_else(|| panic!("inventory key `{key}` must be an object"))
        .iter()
        .map(|(k, r)| {
            let reason = r
                .as_str()
                .unwrap_or_else(|| panic!("inventory `{key}.{k}` reason must be a string"));
            assert!(
                !reason.trim().is_empty(),
                "inventory `{key}.{k}` needs a reason"
            );
            (k.clone(), reason.to_string())
        })
        .collect()
}

/// Validate complete ownership before any emitter-specific metadata filtering.
fn image_symbol_owners(
    inv: &Value,
    images: &[String],
    syms: &Symbols,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, String> {
    let emitters: BTreeSet<String> = strings(inv, "pm_emitters").into_iter().collect();
    let owned = inv["emitter_images"]
        .as_object()
        .ok_or("emitter_images must be an object")?;
    if owned.keys().cloned().collect::<BTreeSet<_>>() != emitters {
        return Err("emitter_images must list exactly the changed PM emitters".into());
    }
    let mut owners: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (emitter, list) in owned {
        for image in list
            .as_array()
            .ok_or("emitter image list must be an array")?
        {
            let image = image.as_str().ok_or("emitter image must be a string")?;
            if !owners
                .entry(image.into())
                .or_default()
                .insert(emitter.clone())
            {
                return Err(format!("{image}: duplicate emitter image entry"));
            }
        }
    }
    let image_set: BTreeSet<String> = images.iter().cloned().collect();
    if owners.keys().cloned().collect::<BTreeSet<_>>() != image_set {
        return Err("every changed/related image must be owned".into());
    }
    let mut partitions = BTreeMap::new();
    if let Some(value) = inv.get("image_symbol_owners") {
        for (image, value) in value
            .as_object()
            .ok_or("image_symbol_owners must be an object")?
        {
            let listed = owners
                .get(image)
                .ok_or_else(|| format!("{image}: partition for unknown image"))?;
            let entries = value
                .as_object()
                .ok_or("image symbol partition must be an object")?;
            let origin = format!("image:{image}");
            let names = syms
                .origin_symbols
                .get(&origin)
                .ok_or_else(|| format!("{image}: missing metadata exports"))?;
            let exports: BTreeSet<String> = names.iter().cloned().collect();
            if exports.is_empty()
                || exports.len() != names.len()
                || entries.keys().cloned().collect::<BTreeSet<_>>() != exports
            {
                return Err(format!(
                    "{image}: partition must exactly match distinct metadata exports"
                ));
            }
            let mut partition = BTreeMap::new();
            let mut assigned = BTreeSet::new();
            for (name, owner) in entries {
                let owner = owner.as_str().ok_or("symbol owner must be a string")?;
                if !emitters.contains(owner) || !listed.contains(owner) {
                    return Err(format!("{image}:{name}: unknown or unlisted owner {owner}"));
                }
                if inv["emitter_abi"][owner].as_str() != Some("const:KERNARG_BYTES") {
                    return Err(format!(
                        "{image}:{name}: partition owner requires const:KERNARG_BYTES"
                    ));
                }
                if !syms.decls(name).iter().any(|(o, _)| o == &origin) {
                    return Err(format!("{image}:{name}: missing metadata layout"));
                }
                assigned.insert(owner.to_string());
                partition.insert(name.clone(), owner.to_string());
            }
            if &assigned != listed {
                return Err(format!("{image}: every listed owner must own an export"));
            }
            partitions.insert(image.clone(), partition);
        }
    }
    for (image, listed) in owners {
        if listed.len() > 1 && !partitions.contains_key(&image) {
            return Err(format!(
                "{image}: owned by two emitters without an explicit partition"
            ));
        }
    }
    Ok(partitions)
}

#[test]
fn image_symbol_partition_controls() {
    let inv = serde_json::json!({
        "pm_emitters": ["gate_up.rs", "down.rs"],
        "emitter_images": {"gate_up.rs": ["mixed.hxaco"], "down.rs": ["mixed.hxaco"]},
        "emitter_abi": {"gate_up.rs": "const:KERNARG_BYTES", "down.rs": "const:KERNARG_BYTES"},
        "image_symbol_owners": {"mixed.hxaco": {"gu": "gate_up.rs", "down": "down.rs"}}
    });
    let images = vec!["mixed.hxaco".to_string()];
    let mut syms = Symbols::default();
    let origin = "image:mixed.hxaco".to_string();
    syms.origin_symbols
        .insert(origin.clone(), vec!["gu".into(), "down".into()]);
    for (name, end) in [("gu", 56), ("down", 52)] {
        syms.pm.insert(
            name.into(),
            vec![(origin.clone(), Decl { args: vec![], end })],
        );
    }
    let valid = image_symbol_owners(&inv, &images, &syms).unwrap();
    assert_eq!(valid["mixed.hxaco"]["gu"], "gate_up.rs");
    assert_eq!(valid["mixed.hxaco"]["down"], "down.rs");
    let reject = |bad: &Value| assert!(image_symbol_owners(bad, &images, &syms).is_err());
    let mut bad = inv.clone();
    bad["image_symbol_owners"]["mixed.hxaco"]
        .as_object_mut()
        .unwrap()
        .remove("gu");
    reject(&bad);
    let mut bad = inv.clone();
    bad["image_symbol_owners"]["mixed.hxaco"]["unknown"] = "gate_up.rs".into();
    reject(&bad);
    let mut bad = inv.clone();
    bad["image_symbol_owners"]["mixed.hxaco"]["gu"] = "unknown.rs".into();
    reject(&bad);
    let mut bad = inv.clone();
    bad["image_symbol_owners"]["mixed.hxaco"]["down"] = "gate_up.rs".into();
    reject(&bad);
    let mut bad = inv.clone();
    let partition = bad["image_symbol_owners"]["mixed.hxaco"].clone();
    bad["image_symbol_owners"]["unknown.hxaco"] = partition;
    reject(&bad);
    let mut bad = inv.clone();
    bad.as_object_mut().unwrap().remove("image_symbol_owners");
    reject(&bad);
    let mut bad = inv.clone();
    bad["emitter_abi"]["down.rs"] = "image-metadata".into();
    reject(&bad);
    syms.pm.remove("gu");
    assert!(image_symbol_owners(&inv, &images, &syms).is_err());
}

// ---- source discovery ---------------------------------------------------------

const LAUNCH_PRIMITIVES: [&str; 5] = [
    "launch_maybe_blob",
    "launch_params_blob",
    "launch_kernel_blob",
    "launch_blob_recorded",
    ".launch_kernel(",
];

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rs_files(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

fn is_probe_path(rel: &str) -> bool {
    rel.contains("/examples/") || rel.contains("/tests/") || rel.contains("/benches/")
}

/// Rust that can launch a kernel: all of `rdna-compute/src`, any other crate's
/// `src` that calls a launch primitive, and every `examples/`, `tests/` and
/// `benches/` file that both calls a launch primitive and names a MoE-family
/// word (`hints`, plus the AWQ-indexed family). This test's own sources are
/// excluded: they quote primitives and kernel names as data.
fn scan_paths(root: &Path, hints: &[String], forced: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root.join("crates")) else {
        return out;
    };
    for c in rd.flatten() {
        let mut files = Vec::new();
        for sub in ["src", "examples", "tests", "benches"] {
            rs_files(&c.path().join(sub), &mut files);
        }
        for f in files {
            let rel = f
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if rel.contains("/all_launcher_abi/") || rel.ends_with("/tests/all_launcher_abi.rs") {
                continue;
            }
            if rel.starts_with("crates/rdna-compute/src/") || rel == KERNELS_RS {
                out.push(rel);
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&f) {
                if !LAUNCH_PRIMITIVES.iter().any(|p| text.contains(p)) {
                    continue;
                }
                let family =
                    hints.iter().any(|h| text.contains(h.as_str())) || text.contains("awq_indexed");
                if !is_probe_path(&rel) || family {
                    out.push(rel);
                }
            }
        }
    }
    // Files the inventory's audits name are scanned whatever they contain: an audit scope in
    // a file the discovery filter skipped would otherwise silently drop out of the gate.
    for f in forced {
        assert!(
            root.join(f).is_file(),
            "the inventory names `{f}` as an audited file but it does not exist"
        );
        if !out.contains(f) {
            out.push(f.clone());
        }
    }
    out.sort();
    out
}

struct Fixture {
    root: PathBuf,
    inv: Value,
    syms: Symbols,
    analysis: Analysis,
    scanned: usize,
}

/// The full source analysis is immutable and slow: the two tests that need it share one
/// (test-only cache, no product code involved).
static FIXTURE: LazyLock<Fixture> = LazyLock::new(build_fixture);

fn build_fixture() -> Fixture {
    let root = root();
    let inv = load_inventory();
    // Files whose functions a launcher's blob or kernel name is built by (`use super::common::*`)
    // but that would not pass the discovery filter themselves.
    let mut forced: Vec<String> = if inv.get("scan_files").is_some() {
        strings(&inv, "scan_files")
    } else {
        Vec::new()
    };
    for (k, v) in inv["old_audited"].as_object().into_iter().flatten() {
        if let Some((file, _)) = k.rsplit_once("::") {
            forced.push(file.to_string());
        }
        for sc in v["scopes"].as_array().into_iter().flatten() {
            if let Some(f) = sc["file"].as_str() {
                forced.push(f.to_string());
            }
        }
    }
    let paths = scan_paths(&root, &strings(&inv["family"], "substrings"), &forced);
    let scanned = paths.len();
    let world = World::load(&root, &paths).unwrap_or_else(|e| panic!("scan: {e}"));
    let mut syms = Symbols::default();
    build_hip(
        &world,
        &root,
        &strings(&inv, "hip_sources"),
        &string_map(&inv, "old_consts"),
        &string_map(&inv, "source_tables"),
        &mut syms,
    )
    .unwrap_or_else(|e| panic!("hip: {e}"));
    build_external(
        &root,
        "crates/rdna-compute/tests/fixtures/all-launcher-abi/external_t8_abi.json",
        &mut syms,
    )
    .unwrap_or_else(|e| panic!("external: {e}"));
    let mut images = strings(&inv, "pm_images");
    images.extend(strings(&inv, "pm_related_images"));
    build_pm(&root, &images, &mut syms).unwrap_or_else(|e| panic!("pm: {e}"));
    assert!(
        syms.unevaluable.is_empty(),
        "changed HIP constants whose text cannot be evaluated: {:?}",
        syms.unevaluable
    );
    let analysis = analyze(&world, &syms, &policy(&inv));
    Fixture {
        root,
        inv,
        syms,
        analysis,
        scanned,
    }
}

fn policy(inv: &Value) -> Policy {
    let mut forwarders = BTreeMap::new();
    let mut checked_delegations = BTreeMap::new();
    for (k, v) in inv["forwarders"]
        .as_object()
        .expect("inventory key `forwarders` must be an object")
    {
        let reason = v["reason"]
            .as_str()
            .unwrap_or_else(|| panic!("forwarders.{k}.reason"));
        assert!(!reason.trim().is_empty(), "forwarders.{k} needs a reason");
        if v.get("checked_chain").is_some() {
            assert!(
                v.get("to").is_none(),
                "forwarders.{k}: choose to or checked_chain"
            );
            let returns = match v["flow"].as_str() {
                Some("calls") => false,
                Some("returns") => true,
                _ => panic!("forwarders.{k}.flow must be calls or returns"),
            };
            checked_delegations.insert(
                k.clone(),
                CheckedDelegation {
                    chain: strings(v, "checked_chain"),
                    returns,
                },
            );
            continue;
        }
        let to = v["to"]
            .as_str()
            .unwrap_or_else(|| panic!("forwarders.{k}.to"));
        forwarders.insert(k.clone(), (to.to_string(), reason.to_string()));
    }
    let mut probe_candidates = BTreeMap::new();
    for (k, v) in inv["probe_candidates"]
        .as_object()
        .expect("inventory key `probe_candidates` must be an object")
    {
        let reason = v["reason"]
            .as_str()
            .unwrap_or_else(|| panic!("probe_candidates.{k}.reason"));
        assert!(
            !reason.trim().is_empty(),
            "probe_candidates.{k} needs a reason"
        );
        probe_candidates.insert(k.clone(), strings(v, "images"));
    }
    let mut old_audited = BTreeMap::new();
    for (k, v) in inv["old_audited"]
        .as_object()
        .expect("inventory key `old_audited` must be an object")
    {
        let reason = v["reason"]
            .as_str()
            .unwrap_or_else(|| panic!("old_audited.{k}.reason"));
        assert!(!reason.trim().is_empty(), "old_audited.{k} needs a reason");
        let baseline = v["baseline"]
            .as_str()
            .unwrap_or_else(|| panic!("old_audited.{k}.baseline"));
        let fingerprints = v["fingerprints"]
            .as_object()
            .unwrap_or_else(|| panic!("old_audited.{k}.fingerprints"))
            .iter()
            .map(|(f, h)| {
                (
                    f.clone(),
                    h.as_str()
                        .unwrap_or_else(|| panic!("old_audited.{k}.fingerprints.{f}"))
                        .to_string(),
                )
            })
            .collect();
        old_audited.insert(
            k.clone(),
            OldAudit {
                reason: reason.to_string(),
                baseline: baseline.to_string(),
                callers: strings(v, "callers"),
                fingerprints,
                guard_idents: if v.get("guard_idents").is_some() {
                    strings(v, "guard_idents")
                } else {
                    Vec::new()
                },
                scopes: v
                    .get("scopes")
                    .and_then(|x| x.as_array())
                    .map(|a| {
                        a.iter()
                            .map(|sc| {
                                let name = sc["name"]
                                    .as_str()
                                    .unwrap_or_else(|| panic!("old_audited.{k}.scopes[].name"))
                                    .to_string();
                                let sreason = sc["reason"]
                                    .as_str()
                                    .unwrap_or_else(|| panic!("old_audited.{k}.scopes.{name}.reason"));
                                assert!(!sreason.trim().is_empty(), "old_audited.{k}.scopes.{name} needs a reason");
                                OldScope {
                                    name: name.clone(),
                                    reason: sreason.to_string(),
                                    file: sc["file"]
                                        .as_str()
                                        .unwrap_or_else(|| panic!("old_audited.{k}.scopes.{name}.file"))
                                        .to_string(),
                                    callers: if sc.get("callers").is_some() { strings(sc, "callers") } else { Vec::new() },
                                    fingerprints: sc["fingerprints"]
                                        .as_object()
                                        .unwrap_or_else(|| panic!("old_audited.{k}.scopes.{name}.fingerprints"))
                                        .iter()
                                        .map(|(f, h)| {
                                            (
                                                f.clone(),
                                                h.as_str()
                                                    .unwrap_or_else(|| panic!("old_audited.{k}.scopes.{name}.fingerprints.{f}"))
                                                    .to_string(),
                                            )
                                        })
                                        .collect(),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            },
        );
    }
    Policy {
        non_launching: string_map(inv, "non_launching"),
        inert: string_map(inv, "inert"),
        forwarders,
        checked_delegations,
        old_variants: string_map(inv, "old_variants"),
        probe_candidates,
        old_helpers: string_map(inv, "old_helpers"),
        old_audited,
    }
}

fn report(f: &Fixture) {
    let launchers: BTreeSet<(String, String, u32)> = f
        .analysis
        .reports
        .iter()
        .filter(|r| !r.sites.is_empty() && !r.inert)
        .map(|r| (r.file.clone(), r.func.clone(), r.line))
        .collect();
    let inert = f.analysis.reports.iter().filter(|r| r.inert).count();
    let tests = f
        .analysis
        .reports
        .iter()
        .filter(|r| r.is_test && r.sites.is_empty())
        .count();
    let probes = launchers
        .iter()
        .filter(|(file, _, _)| is_probe_path(file))
        .count();
    println!(
        "all_launcher_abi: scanned {} files; {} HIP + {} PM symbols; {} live launcher fns ({} in examples/tests/benches) / {} checked sites / {} (symbol, declaration) pairs; {} inert cfg(any()) fns; {} test-only mentions; {} covered symbols; {} OLD-variant exemptions used",
        f.scanned,
        f.syms.hip.len(),
        f.syms.pm.len(),
        launchers.len(),
        probes,
        f.analysis.checked_sites,
        f.analysis.checked_pairs,
        inert,
        tests,
        f.analysis.covered.len(),
        f.analysis.old_used.len()
    );
}

// ---- tests ---------------------------------------------------------------------

fn hip_files(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let p = e.path();
        if p.is_dir() {
            hip_files(&p, &format!("{prefix}{name}/"), out);
        } else if name.ends_with(".hip") {
            out.push(format!("{prefix}{name}"));
        }
    }
}

#[test]
fn inventory_matches_the_changed_kernel_catalogue() {
    let root = root();
    let inv = load_inventory();
    let hip = strings(&inv, "hip_sources");
    let uniq: BTreeSet<&String> = hip.iter().collect();
    assert_eq!(uniq.len(), hip.len(), "duplicate hip_sources entries");
    for f in &hip {
        assert!(
            f.starts_with("kernels/src/") && f.ends_with(".hip"),
            "{f}: not a kernels/src HIP source"
        );
        assert!(root.join(f).is_file(), "{f}: missing");
    }
    let emitters = strings(&inv, "pm_emitters");
    for e in &emitters {
        assert!(root.join(e).is_file(), "{e}: missing PM emitter");
    }
    let mut images = strings(&inv, "pm_images");
    let related = strings(&inv, "pm_related_images");
    for i in images.iter().chain(related.iter()) {
        assert!(root.join(i).is_file(), "{i}: missing PM image");
    }
    images.extend(related);
    // Shared physical images require a complete, metadata-validated partition.
    let mut syms = Symbols::default();
    build_pm(&root, &images, &mut syms).unwrap_or_else(|e| panic!("pm: {e}"));
    image_symbol_owners(&inv, &images, &syms).unwrap_or_else(|e| panic!("{e}"));
    // Closed MoE family: every moe-family HIP source (recursive under kernels/src) is
    // either changed or reviewed-unchanged.
    let subs = strings(&inv["family"], "substrings");
    let reviewed = string_map(&inv["family"], "reviewed_unchanged");
    let changed_rel: BTreeSet<String> = hip
        .iter()
        .map(|f| f.trim_start_matches("kernels/src/").to_string())
        .collect();
    let mut all_hip: Vec<String> = Vec::new();
    hip_files(&root.join("kernels/src"), "", &mut all_hip);
    let mut unlisted = Vec::new();
    for rel in &all_hip {
        let name = rel.rsplit('/').next().unwrap();
        if subs.iter().any(|s| name.contains(s.as_str()))
            && !changed_rel.contains(rel)
            && !reviewed.contains_key(rel)
        {
            unlisted.push(rel.clone());
        }
    }
    unlisted.sort();
    assert!(
        unlisted.is_empty(),
        "MoE-family HIP sources neither in the changed catalogue nor reviewed-unchanged (list them in inventory.json after review): {unlisted:?}"
    );
    for name in reviewed.keys() {
        assert!(
            !changed_rel.contains(name),
            "{name}: reviewed-unchanged but also in the changed catalogue"
        );
        assert!(
            root.join("kernels/src").join(name).is_file(),
            "{name}: reviewed-unchanged source no longer exists"
        );
    }
    // Old-ABI fixtures are never production callers.
    for (path, _) in string_map(&inv, "old_fixtures") {
        assert!(
            path.contains("/examples/") || path.contains("/tests/"),
            "{path}: old fixtures live in examples/ or tests/"
        );
        let text =
            std::fs::read_to_string(root.join(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            text.contains("OLD_"),
            "{path}: expected inline OLD_* kernel sources"
        );
    }
}

#[test]
fn live_launchers_match_declared_kernel_abi() {
    let f = &*FIXTURE;
    report(f);
    let (abi, _) = violations(&f.analysis, &policy(&f.inv));
    assert!(
        f.analysis.checked_sites > 0,
        "no launch site was checked: the scan is broken"
    );
    assert!(
        abi.is_empty(),
        "launcher ABI mismatches ({}):\n{}",
        abi.len(),
        abi.join("\n")
    );
}

#[test]
fn changed_kernel_coverage_is_closed() {
    let f = &*FIXTURE;
    report(f);
    let (_, mut problems) = violations(&f.analysis, &policy(&f.inv));
    let unl = string_map(&f.inv, "unlaunched");
    let all = f.syms.all_names();
    for s in &all {
        if !f.analysis.covered.contains(s) && !unl.contains_key(s) {
            problems.push(format!("changed symbol `{s}` is reached by no checked live launcher and is not listed as unlaunched"));
        }
    }
    for s in unl.keys() {
        if !all.contains(s) {
            problems.push(format!(
                "unlaunched entry `{s}` is not a changed kernel symbol"
            ));
        }
        if f.analysis.covered.contains(s) {
            problems.push(format!(
                "unlaunched entry `{s}` is now reached by a live launcher: remove it"
            ));
        }
    }
    // Every changed source and image declares at least one kernel.
    for src in strings(&f.inv, "hip_sources") {
        if f.syms
            .origin_symbols
            .get(&format!("file:{src}"))
            .map_or(true, |v| v.is_empty())
        {
            problems.push(format!("{src}: declares no __global__ kernel"));
        }
    }
    for img in strings(&f.inv, "pm_images")
        .into_iter()
        .chain(strings(&f.inv, "pm_related_images"))
    {
        if f.syms
            .origin_symbols
            .get(&format!("image:{img}"))
            .map_or(true, |v| v.is_empty())
        {
            problems.push(format!("{img}: exports no kernel"));
        }
    }
    // Images must really be the embedded ones.
    let kernels_text = std::fs::read_to_string(f.root.join(KERNELS_RS)).unwrap();
    for img in strings(&f.inv, "pm_images")
        .into_iter()
        .chain(strings(&f.inv, "pm_related_images"))
    {
        let base = img.rsplit('/').next().unwrap();
        if !kernels_text.contains(&format!("include_bytes!(\"../../../{img}\")"))
            && !kernels_text.contains(base)
        {
            problems.push(format!("{img}: not embedded by rdna-compute kernels.rs"));
        }
    }
    assert!(
        problems.is_empty(),
        "coverage problems ({}):\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// Consistency of the emitters' kernarg CONSTANTS (`KERNARG_BYTES`, and for the fp8
/// HIP-twin emitter `KERNARG_OFFSETS`) with the embedded PM image metadata and the HIP
/// twin's declaration. This does NOT verify the emitters' per-argument declarations:
/// the PM metadata embedded in the images (read independently by `build_pm`) is the
/// declared-ABI authority the launchers are checked against, not the emitter source.
#[test]
fn pm_emitter_kernarg_constants_agree_with_image_extents_and_the_hip_twin() {
    let root = root();
    let inv = load_inventory();
    let mut syms = Symbols::default();
    let mut images = strings(&inv, "pm_images");
    images.extend(strings(&inv, "pm_related_images"));
    build_pm(&root, &images, &mut syms).unwrap_or_else(|e| panic!("pm: {e}"));
    let partitions = image_symbol_owners(&inv, &images, &syms)
        .unwrap_or_else(|e| panic!("image ownership: {e}"));
    let owned = inv["emitter_images"].as_object().expect("emitter_images");
    let abi = string_map(&inv, "emitter_abi");
    let mut problems = Vec::new();
    for (emitter, list) in owned {
        let text = std::fs::read_to_string(root.join(emitter))
            .unwrap_or_else(|e| panic!("{emitter}: {e}"));
        let ts = lex::lex(&text).unwrap_or_else(|e| panic!("{emitter}: {e}"));
        let items = lex::walk_items(&ts);
        let consts: BTreeMap<String, Vec<lex::Tt>> = items
            .consts
            .iter()
            .map(|(n, c)| (n.clone(), c.init()))
            .collect();
        let imgs: Vec<String> = list
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_str().unwrap().to_string())
            .collect();
        let mode = abi.get(emitter).map(|s| s.as_str()).unwrap_or("");
        // Image metadata self-consistency per image and the emitter's declared
        // `KERNARG_BYTES` when it has one.
        let declared_bytes: Option<u32> = match consts.get("KERNARG_BYTES").map(|v| v.as_slice()) {
            Some([lex::Tt::N(n, _)]) => n.parse::<u32>().ok(),
            _ => None,
        };
        for img in &imgs {
            let origin = format!("image:{img}");
            let names = syms
                .origin_symbols
                .get(&origin)
                .cloned()
                .unwrap_or_default();
            if names.is_empty() {
                problems.push(format!("{emitter}: image {img} exports no kernel"));
            }
            let mut ends: BTreeSet<u32> = BTreeSet::new();
            for n in &names {
                if partitions
                    .get(img)
                    .is_some_and(|owners| owners.get(n) != Some(emitter))
                {
                    continue;
                }
                for (o, d) in syms.decls(n) {
                    if o == origin {
                        ends.insert(d.end);
                        if let Some(bytes) = declared_bytes {
                            if mode.starts_with("const:KERNARG_BYTES") && d.end != bytes {
                                problems.push(format!(
                                    "{emitter}: KERNARG_BYTES={bytes} but {img}:{n} declares explicit arguments ending at {}",
                                    d.end
                                ));
                            }
                        }
                    }
                }
            }
            if ends.len() > 1 && !mode.contains("mixed") {
                problems.push(format!("{emitter}: {img} exports kernels with different explicit-argument ends {ends:?}"));
            }
        }
        if mode.starts_with("const:KERNARG_BYTES") && declared_bytes.is_none() {
            problems.push(format!(
                "{emitter}: inventory says KERNARG_BYTES is a literal constant but none was found"
            ));
        }
        if mode.starts_with("hip_twin:") {
            // The emitter's own offsets/bytes versus the HIP twin's declaration.
            let twin = mode.trim_start_matches("hip_twin:");
            let (twin_file, twin_symbol) = twin.split_once('#').expect("hip_twin:file#symbol");
            let hip_text = std::fs::read_to_string(root.join(twin_file))
                .unwrap_or_else(|e| panic!("{twin_file}: {e}"));
            let decl = hip::decls_of(&hip_text).unwrap_or_else(|e| panic!("{twin_file}: {e}"));
            let d = decl
                .get(twin_symbol)
                .unwrap_or_else(|| panic!("{twin_file}: no kernel {twin_symbol}"));
            let offsets: Vec<u32> = match consts.get("KERNARG_OFFSETS").map(|v| v.as_slice()) {
                Some([g]) if g.is_g('[') => g
                    .group()
                    .unwrap()
                    .iter()
                    .filter_map(|t| {
                        if let lex::Tt::N(n, _) = t {
                            n.parse::<u32>().ok()
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            // The PM emitter lists 32-bit words: a 64-bit integer argument appears as lo, hi.
            let mut want: Vec<u32> = Vec::new();
            for a in &d.args {
                want.push(a.offset);
                if a.size == 8 && a.kind == 'i' {
                    want.push(a.offset + 4);
                }
            }
            if offsets != want {
                problems.push(format!("{emitter}: KERNARG_OFFSETS {offsets:?} differ from {twin_symbol} declaration {want:?}"));
            }
            if declared_bytes != Some(d.end) {
                problems.push(format!(
                    "{emitter}: KERNARG_BYTES {declared_bytes:?} differs from {twin_symbol} end {}",
                    d.end
                ));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "emitter/image ABI problems:\n{}",
        problems.join("\n")
    );
}

// ---- negative controls: the gate must fail on the known bug shapes ----------------

fn run_synth_at(
    path: &str,
    src: &str,
    symbols: &[(&str, Decl)],
    pol: &Policy,
) -> (Vec<String>, Vec<String>) {
    let world = World::from_sources(Path::new("."), vec![(path.to_string(), src.to_string())])
        .expect("synthetic source lexes");
    let mut syms = Symbols::default();
    for (name, decl) in symbols {
        syms.hip.insert(
            (*name).to_string(),
            vec![("synthetic".to_string(), decl.clone())],
        );
    }
    let a = analyze(&world, &syms, pol);
    violations(&a, pol)
}

fn run_synth(src: &str, symbols: &[(&str, Decl)], pol: &Policy) -> (Vec<String>, Vec<String>) {
    run_synth_at("synthetic.rs", src, symbols, pol)
}

fn no_policy() -> Policy {
    Policy::default()
}

/// A K1-shaped launcher: seven pointers then ten 32-bit words, the last being
/// `reserved` at byte 92 (96-byte kernarg). `drop_params_reserved` and
/// `drop_blob_reserved` reproduce the beta slab-MoE bug and its blob twin.
fn k1_source(drop_params_reserved: bool, drop_blob_reserved: bool, swap_params: bool) -> String {
    let ptrs = ["wp", "ids", "offs", "slot_ids", "xp", "y0p", "y1p"];
    let words = [
        "mv", "kv", "uv", "cv", "dv", "srows", "xs", "ys", "nexp", "reserved",
    ];
    let mut lets = String::new();
    for p in ptrs {
        lets.push_str(&format!("        let {p} = t.buf.as_ptr();\n"));
    }
    for w in words {
        lets.push_str(&format!("        let {w} = 0u32;\n"));
    }
    let mut pnames: Vec<&str> = ptrs.iter().chain(words.iter()).copied().collect();
    if drop_params_reserved {
        pnames.pop();
    }
    if swap_params {
        let n = pnames.len();
        pnames.swap(n - 3, n - 2);
    }
    let params: String = pnames
        .iter()
        .map(|p| format!("            &{p} as *const _ as *mut c_void,\n"))
        .collect();
    let mut pushes = String::new();
    for p in ptrs {
        pushes.push_str(&format!("                b.push_ptr({p});\n"));
    }
    let mut bw: Vec<&str> = words.to_vec();
    if drop_blob_reserved {
        bw.pop();
    }
    pushes.push_str(&format!("                for word in [{}] {{\n                    b.push_u32(word);\n                }}\n", bw.join(", ")));
    format!(
        "impl Gpu {{\n    fn k1(&mut self) -> HipResult<()> {{\n{lets}        let mut params = [\n{params}        ];\n        self.launch_maybe_blob(\"k1_symbol\", [1, 1, 1], [32, 1, 1], 0, &mut params, || {{\n            let mut b = hip_bridge::KernargBlob::new();\n{pushes}                b\n        }})\n    }}\n}}\n"
    )
}

fn k1_decl() -> Decl {
    let mut a = vec![(8u32, 'p'); 7];
    a.extend(vec![(4u32, 'i'); 10]);
    Decl::from_scalars(&a)
}

#[test]
fn negative_k1_shape_baseline_passes() {
    let (abi, cov) = run_synth(
        &k1_source(false, false, false),
        &[("k1_symbol", k1_decl())],
        &no_policy(),
    );
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    assert_eq!(
        k1_decl().args.last().unwrap().offset,
        92,
        "reserved word sits at byte 92"
    );
    assert_eq!(k1_decl().end, 96);
}

#[test]
fn negative_params_omit_reserved_argument_is_rejected() {
    // The beta bug: 16 kernelParams entries for a 17-argument kernel.
    let (abi, _) = run_synth(
        &k1_source(true, false, false),
        &[("k1_symbol", k1_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("kernelParams array has 16 entries, declared 17")),
        "{abi:?}"
    );
}

#[test]
fn negative_blob_omit_reserved_argument_is_rejected() {
    let (abi, _) = run_synth(
        &k1_source(false, true, false),
        &[("k1_symbol", k1_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("raw blob has 16 arguments, declared 17")),
        "{abi:?}"
    );
}

#[test]
fn negative_params_order_mistake_is_rejected() {
    let (abi, _) = run_synth(
        &k1_source(false, false, true),
        &[("k1_symbol", k1_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter().any(|p| p.contains("kernelParams points at")),
        "{abi:?}"
    );
}

fn small_source(params: &str, pushes: &str, lets: &str) -> String {
    format!(
        "impl Gpu {{\n    fn small(&mut self) -> HipResult<()> {{\n{lets}        let mut params = [\n{params}        ];\n        self.launch_maybe_blob(\"small_symbol\", [1, 1, 1], [32, 1, 1], 0, &mut params, || {{\n            let mut b = hip_bridge::KernargBlob::new();\n{pushes}            b\n        }})\n    }}\n}}\n"
    )
}

const SMALL_LETS: &str = "        let wp = t.buf.as_ptr();\n        let nv = n as i32;\n        let sc = 1.0f32;\n        let big = k as u64;\n";
const SMALL_PARAMS: &str = "            &wp as *const _ as *mut c_void,\n            &nv as *const _ as *mut c_void,\n            &sc as *const _ as *mut c_void,\n            &big as *const _ as *mut c_void,\n";

fn small_decl() -> Decl {
    // ptr@0, int@8, float@12, u64@16 (natural alignment), end 24.
    Decl::from_scalars(&[(8, 'p'), (4, 'i'), (4, 'f'), (8, 'i')])
}

#[test]
fn negative_blob_layout_and_alignment_mistakes_are_rejected() {
    let good = "            b.push_ptr(wp);\n            b.push_i32(nv);\n            b.push_f32(sc);\n            b.push_u64(big);\n";
    let (abi, cov) = run_synth(
        &small_source(SMALL_PARAMS, good, SMALL_LETS),
        &[("small_symbol", small_decl())],
        &no_policy(),
    );
    assert!(
        abi.is_empty() && cov.is_empty(),
        "baseline: {abi:?} {cov:?}"
    );
    // Wrong width: the u64 pushed as a 32-bit word shrinks the blob and shifts nothing else but the end.
    let narrow = good.replace("b.push_u64(big)", "b.push_u32(big)");
    let (abi, _) = run_synth(
        &small_source(SMALL_PARAMS, &narrow, SMALL_LETS),
        &[("small_symbol", small_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("arg 3") && p.contains("8 bytes")),
        "{abi:?}"
    );
    // Order mistake: i32 and f32 swapped (same width, different class).
    let swapped = "            b.push_ptr(wp);\n            b.push_f32(sc);\n            b.push_i32(nv);\n            b.push_u64(big);\n";
    let (abi, _) = run_synth(
        &small_source(SMALL_PARAMS, swapped, SMALL_LETS),
        &[("small_symbol", small_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("float/integer class mismatch")),
        "{abi:?}"
    );
    // Pointer pushed after the words: the declared pointer offset is violated.
    let late_ptr = "            b.push_i32(nv);\n            b.push_f32(sc);\n            b.push_ptr(wp);\n            b.push_u64(big);\n";
    let (abi, _) = run_synth(
        &small_source(SMALL_PARAMS, late_ptr, SMALL_LETS),
        &[("small_symbol", small_decl())],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("arg 0") || p.contains("arg 2")),
        "{abi:?}"
    );
    // A declaration with a non-natural offset (PM metadata may place arguments freely):
    // the blob's natural padding then differs.
    let mut odd = small_decl();
    odd.args[3].offset = 20;
    odd.end = 28;
    let (abi, _) = run_synth(
        &small_source(SMALL_PARAMS, good, SMALL_LETS),
        &[("small_symbol", odd)],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("offset 16") && p.contains("offset 20")),
        "{abi:?}"
    );
    // Trailing bytes: declared kernarg end beyond the blob (reserved tail not pushed).
    let mut tail = small_decl();
    tail.end = 32;
    let (abi, _) = run_synth(
        &small_source(SMALL_PARAMS, good, SMALL_LETS),
        &[("small_symbol", tail)],
        &no_policy(),
    );
    assert!(
        abi.iter().any(|p| p.contains("raw blob ends at byte 24")),
        "{abi:?}"
    );
}

#[test]
fn negative_launch_params_blob_macro_dropping_an_argument_is_rejected() {
    // The gfx1030 probe shape: 11 kernelParams entries, 10 captured blob values.
    let names = [
        "ep",
        "tp",
        "sp",
        "xp",
        "yp",
        "m_val",
        "k_val",
        "xrd_val",
        "mt_val",
        "xsr_val",
        "n_experts_val",
    ];
    let mut lets = String::new();
    for n in &names[..5] {
        lets.push_str(&format!("        let {n} = t.buf.as_ptr();\n"));
    }
    for n in &names[5..] {
        lets.push_str(&format!("        let {n} = 1 as i32;\n"));
    }
    let params: String = names
        .iter()
        .map(|p| format!("            &{p} as *const _ as *mut c_void,\n"))
        .collect();
    let blob_vals = |v: &[&str]| v.join(", ");
    let src = |vals: &str| {
        format!(
            "impl Gpu {{\n    fn probe(&mut self) -> HipResult<()> {{\n{lets}        let mut params: Vec<*mut c_void> = vec![\n{params}        ];\n        let result = launch_params_blob!(self, \"macro_symbol\", [1, 1, 1], [32, 1, 1], 0, params; {vals});\n        result\n    }}\n}}\n"
        )
    };
    let mut a = vec![(8u32, 'p'); 5];
    a.extend(vec![(4u32, 'i'); 6]);
    let decl = Decl::from_scalars(&a);
    let (abi, cov) = run_synth(
        &src(&blob_vals(&names)),
        &[("macro_symbol", decl.clone())],
        &no_policy(),
    );
    assert!(
        abi.is_empty() && cov.is_empty(),
        "macro baseline: {abi:?} {cov:?}"
    );
    let (abi, _) = run_synth(
        &src(&blob_vals(&names[..10])),
        &[("macro_symbol", decl)],
        &no_policy(),
    );
    assert!(
        abi.iter()
            .any(|p| p.contains("raw blob has 10 arguments, declared 11")),
        "{abi:?}"
    );
}

#[test]
fn negative_conditional_variants_are_checked_per_branch() {
    // moe_router_softmax_top10_f32 shape: one launcher, two symbols, one extra
    // argument in the fast branch of both the params array and the blob.
    let src = |blob_fast: &str| {
        format!(
            "impl Gpu {{\n    fn router(&mut self, fast: bool) -> HipResult<()> {{\n        let func: &str = if fast {{ \"router_fast\" }} else {{ \"router_plain\" }};\n        let lp = l.buf.as_ptr();\n        let nv = 1i32;\n        let tc = 2i32;\n        let mut params: Vec<*mut c_void> = vec![\n            &lp as *const _ as *mut c_void,\n            &nv as *const _ as *mut c_void,\n        ];\n        if fast {{\n            params.push(&tc as *const _ as *mut c_void);\n        }}\n        self.launch_maybe_blob(func, [1, 1, 1], [32, 1, 1], 0, &mut params, || {{\n            let mut b = hip_bridge::KernargBlob::new();\n            b.push_ptr(lp);\n            b.push_i32(nv);\n{blob_fast}            b\n        }})\n    }}\n}}\n"
        )
    };
    let plain = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let fast = Decl::from_scalars(&[(8, 'p'), (4, 'i'), (4, 'i')]);
    let syms = [("router_plain", plain), ("router_fast", fast)];
    let (abi, cov) = run_synth(
        &src("            if fast {\n                b.push_i32(tc);\n            }\n"),
        &syms,
        &no_policy(),
    );
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The fast branch forgets the blob word while still declaring the params entry.
    let (abi, _) = run_synth(&src(""), &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a branch-specific omission must be reported"
    );
}

#[test]
fn negative_unlisted_callers_and_unsupported_syntax_fail_closed() {
    let decl = Decl::from_scalars(&[(8, 'p')]);
    let syms = [("tracked_symbol", decl)];
    // A function that mentions a changed kernel but has no launch site must be classified.
    let mention = "impl Gpu {\n    fn loader(&mut self) {\n        self.ensure_kernel(\"tracked_symbol\", SRC, \"tracked_symbol\");\n    }\n}\n";
    let (_, cov) = run_synth(mention, &syms, &no_policy());
    assert!(
        cov.iter().any(|c| c.contains("no checked launch site")),
        "{cov:?}"
    );
    let mut pol = Policy::default();
    pol.non_launching
        .insert("synthetic.rs::loader".into(), "loader only".into());
    let (abi, cov) = run_synth(mention, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A stale classification is rejected.
    pol.non_launching
        .insert("synthetic.rs::gone".into(), "stale".into());
    let (_, cov) = run_synth(mention, &syms, &pol);
    assert!(
        cov.iter().any(|c| c.contains("stale non_launching")),
        "{cov:?}"
    );
    // A launcher built with an unsupported params construction fails closed.
    let unsupported = "impl Gpu {\n    fn odd(&mut self) {\n        let mut params = Vec::with_capacity(1);\n        params.extend(other());\n        self.launch_maybe_blob(\"tracked_symbol\", g, b, 0, &mut params, || { let mut x = hip_bridge::KernargBlob::new(); x.push_ptr(p); x });\n    }\n}\n";
    let (_, cov) = run_synth(unsupported, &syms, &no_policy());
    assert!(cov.iter().any(|c| c.contains("UNSUPPORTED")), "{cov:?}");
}

#[test]
fn negative_inert_cfg_any_exclusion_is_explicit_and_not_a_blanket_skip() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("inert_symbol", decl)];
    let body = "        let wp = t.buf.as_ptr();\n        let nv = n as i32;\n        let mut params = [\n            &wp as *const _ as *mut c_void,\n        ];\n        self.launch_maybe_blob(\"inert_symbol\", g, b, 0, &mut params, || {\n            let mut b = hip_bridge::KernargBlob::new();\n            b.push_ptr(wp);\n            b\n        })\n";
    let inert_src = format!(
        "impl Gpu {{\n    #[cfg(any())]\n    fn old_copy(&mut self) {{\n{body}    }}\n}}\n"
    );
    // Not listed: reported.
    let (_, cov) = run_synth(&inert_src, &syms, &no_policy());
    assert!(
        cov.iter().any(|c| c.contains("not listed as inert")),
        "{cov:?}"
    );
    // Listed: excluded, no ABI complaint.
    let mut pol = Policy::default();
    pol.inert.insert(
        "synthetic.rs::old_copy".into(),
        "cfg(any()) never compiled".into(),
    );
    let (abi, cov) = run_synth(&inert_src, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The identical live function is a real failure.
    let live_src = format!("impl Gpu {{\n    fn live_copy(&mut self) {{\n{body}    }}\n}}\n");
    let (abi, _) = run_synth(&live_src, &syms, &pol);
    assert!(!abi.is_empty(), "the live twin of an inert bug must fail");
}

#[test]
fn hip_preprocessor_reads_alias_macros_parameter_macros_and_target_conditionals() {
    let src = "#include <hip/hip_runtime.h>\n#ifndef KNAME\n#define KNAME real_kernel\n#endif\n#define ARGS const float* __restrict__ x, int n\n#if ABI_EXTRA\n#define EXTRA , int extra\n#else\n#define EXTRA\n#endif\nextern \"C\" __global__ __launch_bounds__(64) void KNAME(ARGS EXTRA, unsigned long long stride, _Float16 h, int n_experts) {}\n";
    let d = hip::decls_of(src).expect("parse");
    let k = d
        .get("real_kernel")
        .expect("alias resolved to the defined kernel name");
    let layout: Vec<(u32, u32)> = k.args.iter().map(|a| (a.offset, a.size)).collect();
    assert_eq!(
        layout,
        vec![(0, 8), (8, 4), (16, 8), (24, 2), (28, 4)],
        "natural alignment incl. 2-byte half"
    );
    assert_eq!(k.end, 32);
    // An argument layout that differs between targets is refused, not guessed.
    let arch_dep =
        "extern \"C\" __global__ void k(\n#if defined(__gfx1151__)\nint a,\n#endif\nint b) {}\n";
    assert!(
        hip::decls_of(arch_dep).is_err(),
        "target-dependent signature must fail closed"
    );
}

#[test]
fn negative_forwarded_kernel_names_extend_the_launcher_and_fail_closed_when_unlisted() {
    // A launcher whose kernel name arrives through a `&str` parameter: the
    // caller's symbol must be checked against the launcher's real layout.
    let src = "impl Gpu {\n    fn launch_impl(&mut self, entry: Option<&'static str>) -> HipResult<()> {\n        let kname = match entry {\n            None => \"base_symbol\",\n            Some(e) => e,\n        };\n        let wp = t.buf.as_ptr();\n        let nv = n as i32;\n        let mut params = [\n            &wp as *const _ as *mut c_void,\n            &nv as *const _ as *mut c_void,\n        ];\n        self.launch_maybe_blob(kname, g, b, 0, &mut params, || {\n            let mut x = hip_bridge::KernargBlob::new();\n            x.push_ptr(wp);\n            x.push_i32(nv);\n            x\n        })\n    }\n    fn entry_a(&mut self) -> HipResult<()> {\n        self.launch_impl(Some(\"extra_symbol\"))\n    }\n}\n";
    let two = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let three = Decl::from_scalars(&[(8, 'p'), (4, 'i'), (4, 'i')]);
    // Unlisted forwarder: fail closed.
    let (_, cov) = run_synth(
        src,
        &[("base_symbol", two.clone()), ("extra_symbol", two.clone())],
        &no_policy(),
    );
    assert!(
        cov.iter().any(|c| c.contains("without a forwarders entry")),
        "{cov:?}"
    );
    // Listed forwarder with a matching declaration: clean.
    let mut pol = Policy::default();
    pol.forwarders.insert(
        "synthetic.rs::entry_a".into(),
        ("launch_impl".into(), "passes the entry name".into()),
    );
    let (abi, cov) = run_synth(
        src,
        &[("base_symbol", two.clone()), ("extra_symbol", two.clone())],
        &pol,
    );
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The forwarded symbol declares an extra argument the launcher does not pass.
    let (abi, _) = run_synth(src, &[("base_symbol", two), ("extra_symbol", three)], &pol);
    assert!(abi.iter().any(|p| p.contains("extra_symbol")), "{abi:?}");

    // Explicit call/returned-descriptor chains never add candidates or exempt
    // a terminal ABI. Broken edges, missing symbols and bad blobs fail closed.
    for returns in [false, true] {
        let source = if returns {
            "fn descriptor() -> &'static str { \"chain_symbol\" }\nfn launch() { let _ = descriptor();"
        } else {
            "fn descriptor() { let _ = \"chain_symbol\"; launch(); }\nfn launch() {"
        };
        let source = format!("{source}\nlet wp = t.buf.as_ptr();\nlet nv = n as i32;\nlet mut params = [&wp as *const _ as *mut c_void, &nv as *const _ as *mut c_void];\nself.launch_maybe_blob(\"chain_symbol\", g, b, 0, &mut params, || {{ let mut x = hip_bridge::KernargBlob::new(); x.push_ptr(wp); x.push_i32(nv); x }});\n}}");
        let mut policy = Policy::default();
        policy.checked_delegations.insert(
            "synthetic.rs::descriptor".into(),
            CheckedDelegation {
                chain: vec![
                    "synthetic.rs::descriptor".into(),
                    "synthetic.rs::launch".into(),
                ],
                returns,
            },
        );
        let symbols = [("chain_symbol", Decl::from_scalars(&[(8, 'p'), (4, 'i')]))];
        let (abi, coverage) = run_synth(&source, &symbols, &policy);
        assert!(
            abi.is_empty() && coverage.is_empty(),
            "{abi:?} {coverage:?}"
        );
        let broken = source.replace(
            if returns {
                "descriptor();"
            } else {
                "launch();"
            },
            "unrelated();",
        );
        let (_, coverage) = run_synth(&broken, &symbols, &policy);
        assert!(
            coverage.iter().any(|e| e.contains("no longer calls")),
            "{coverage:?}"
        );
        let bad_blob = source.replace("x.push_i32(nv);", "");
        let (abi, coverage) = run_synth(&bad_blob, &symbols, &policy);
        assert!(
            !abi.is_empty() && coverage.iter().any(|e| e.contains("terminal launch")),
            "{abi:?} {coverage:?}"
        );
        let missing = source.replacen("\"chain_symbol\"", "\"other_symbol\"", 1);
        let mut symbols = symbols.to_vec();
        symbols.push(("other_symbol", Decl::from_scalars(&[(8, 'p'), (4, 'i')])));
        let (_, coverage) = run_synth(&missing, &symbols, &policy);
        assert!(
            coverage.iter().any(|e| e.contains("every origin symbol")),
            "{coverage:?}"
        );
    }

    // The slab shape returns two nested descriptors before the real 17-arg
    // launch. Field initializers must retain their constructing function's
    // local bindings, including tuple-bound branch results.
    let descriptors = r#"
fn descriptor(which: bool) -> Option<Descriptor> {
    let (module, symbols) = if which {
        ("module_a", ["k1_symbol", "second_symbol"])
    } else {
        ("module_b", ["second_symbol", "k1_symbol"])
    };
    Some(Descriptor { module, symbols })
}
fn pick(which: bool, down: usize) -> Option<Pick> {
    let tile = descriptor(which)?;
    Some(Pick { symbol: tile.symbols[down] })
}
"#;
    let mut policy = Policy::default();
    policy.checked_delegations.insert(
        "synthetic.rs::descriptor".into(),
        CheckedDelegation {
            chain: vec![
                "synthetic.rs::descriptor".into(),
                "synthetic.rs::pick".into(),
                "synthetic.rs::k1".into(),
            ],
            returns: true,
        },
    );
    let symbols = [("k1_symbol", k1_decl()), ("second_symbol", k1_decl())];
    let nested = |drop_params, drop_blob, swap| {
        let launcher = k1_source(drop_params, drop_blob, swap)
            .replace("\"k1_symbol\"", "func")
            .replace(
                "        let wp",
                "        let pick = pick(which, down).expect(\"selected\");\n        let func = pick.symbol;\n        let wp",
            );
        format!("{descriptors}\n{launcher}")
    };
    let good = nested(false, false, false);
    let (abi, coverage) = run_synth(&good, &symbols, &policy);
    assert!(
        abi.is_empty() && coverage.is_empty(),
        "{abi:?} {coverage:?}"
    );
    for bad in [
        nested(true, false, false),
        nested(false, true, false),
        nested(false, false, true),
    ] {
        let (abi, coverage) = run_synth(&bad, &symbols, &policy);
        assert!(
            !abi.is_empty() && coverage.iter().any(|e| e.contains("terminal launch")),
            "{abi:?} {coverage:?}"
        );
    }
    for bad in [
        good.replace("descriptor(which)?", "unrelated(which)?"),
        good.replace("let func = pick.symbol;", "let func = \"untracked\";"),
        good.replace("symbol: tile.symbols[down]", "symbol: \"untracked\""),
        good.replace(
            "    Some(Pick { symbol: tile.symbols[down] })",
            "    let discarded = Pick { symbol: tile.symbols[down] };\n    Some(Pick { symbol: \"untracked\" })",
        ),
        format!("{good}\nfn descriptor(which: bool) -> Option<Descriptor> {{ None }}"),
    ] {
        let (_, coverage) = run_synth(&bad, &symbols, &policy);
        assert!(!coverage.is_empty(), "{coverage:?}");
    }
}

#[test]
fn registry_source_factories_declare_the_indexed_abi() {
    // The JIT/registry corpus is the source the runtime really compiles
    // (including factory-built text). For every changed symbol it produces, the
    // declaration parsed from that text must equal one the launcher check used.
    let root = root();
    let inv = load_inventory();
    let world = World::load(&root, &[KERNELS_RS.to_string()]).unwrap_or_else(|e| panic!("{e}"));
    let mut syms = Symbols::default();
    // This world holds kernels.rs only: no example-side OLD_* consts or tables to classify.
    build_hip(
        &world,
        &root,
        &strings(&inv, "hip_sources"),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &mut syms,
    )
    .unwrap_or_else(|e| panic!("hip: {e}"));
    let names = syms.all_names();
    let mut checked = 0usize;
    let mut problems = Vec::new();
    for arch in rdna_compute::kernel_registry::SUPPORTED_ARCHES {
        let entries = rdna_compute::kernel_registry::corpus_entries(arch, "")
            .unwrap_or_else(|e| panic!("corpus_entries({arch}): {e:?}"));
        for entry in entries {
            let hits: Vec<&&str> = entry
                .symbols
                .iter()
                .filter(|s| names.contains(**s))
                .collect();
            if hits.is_empty() {
                continue;
            }
            let parsed = match hip::decls_of(entry.source()) {
                Ok(p) => p,
                Err(e) => {
                    problems.push(format!(
                        "{arch}/{}: registry source does not parse: {e}",
                        entry.module
                    ));
                    continue;
                }
            };
            for s in hits {
                let Some(d) = parsed.get(*s) else {
                    problems.push(format!(
                        "{arch}/{}: registry source declares no kernel `{s}`",
                        entry.module
                    ));
                    continue;
                };
                checked += 1;
                if !syms
                    .decls(s)
                    .iter()
                    .any(|(_, known)| hip::same_layout(known, d))
                {
                    problems.push(format!(
                        "{arch}/{}: `{s}` registry declaration differs from every indexed declaration",
                        entry.module
                    ));
                }
            }
        }
    }
    assert!(
        checked > 0,
        "the registry corpus contains no changed kernel: the cross-check is vacuous"
    );
    println!("registry cross-check: {checked} (arch, module, symbol) declarations matched");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

// ---- negative controls: probes (examples/tests/benches) and sliced params ---------

const PROBE_PATH: &str = "crates/x/examples/probe.rs";

fn probe_blob_source(pushes: &str) -> String {
    format!(
        "fn probe(hip: &Hip, m: &Module, extra: Option<u32>) {{\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    let mut b = hip_bridge::KernargBlob::new();\n{pushes}    hip.launch_kernel_blob(f, [1, 1, 1], [32, 1, 1], 0, None, &mut b).unwrap();\n}}\n"
    )
}

fn probe_decl() -> Decl {
    Decl::from_scalars(&[(8, 'p'), (4, 'i'), (4, 'i')])
}

#[test]
fn negative_probe_blob_omission_is_rejected() {
    let syms = [("probe_sym", probe_decl())];
    let good = probe_blob_source("    b.push_ptr(p);\n    b.push_u32(n);\n    b.push_u32(r);\n");
    let (abi, cov) = run_synth_at(PROBE_PATH, &good, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let omitted = probe_blob_source("    b.push_ptr(p);\n    b.push_u32(n);\n");
    let (abi, _) = run_synth_at(PROBE_PATH, &omitted, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a probe blob that drops the last argument must fail"
    );
    let reordered =
        probe_blob_source("    b.push_u32(n);\n    b.push_ptr(p);\n    b.push_u32(r);\n");
    let (abi, _) = run_synth_at(PROBE_PATH, &reordered, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a probe blob with a wrong layout must fail"
    );
}

#[test]
fn negative_probe_old_abi_variants_are_exempt_only_when_listed_per_call() {
    let syms = [("probe_sym", probe_decl())];
    let src = probe_blob_source(
        "    b.push_ptr(p);\n    b.push_u32(n);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n",
    );
    let (abi, _) = run_synth_at(PROBE_PATH, &src, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "the short (OLD) variant must be checked unless exempted"
    );
    let mut pol = Policy::default();
    pol.old_variants.insert(
        format!("{PROBE_PATH}::probe#let Some (r) = extra=false"),
        "builds the OLD ABI on purpose".into(),
    );
    let (abi, cov) = run_synth_at(PROBE_PATH, &src, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The checked (true) variant still fails when it drops an argument.
    let bad = probe_blob_source(
        "    b.push_ptr(p);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n",
    );
    let (abi, _) = run_synth_at(PROBE_PATH, &bad, &syms, &pol);
    assert!(
        !abi.is_empty(),
        "the exemption must not hide the new-ABI variant"
    );
    // A stale exemption is rejected.
    pol.old_variants.insert(
        format!("{PROBE_PATH}::probe#extra.is_some()=false"),
        "stale".into(),
    );
    let (_, cov) = run_synth_at(PROBE_PATH, &src, &syms, &pol);
    assert!(
        cov.iter().any(|c| c.contains("stale old_variants")),
        "{cov:?}"
    );
}

fn probe_bytes_source(writes: &str) -> String {
    format!(
        "fn probe(hip: &Hip, m: &Module) {{\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    let mut a = [0u8; 16];\n{writes}    hip.launch_kernel_blob(f, [1, 1, 1], [32, 1, 1], 0, None, &mut a).unwrap();\n}}\n"
    )
}

#[test]
fn negative_probe_byte_array_kernarg_omission_and_reserved_handling() {
    let mut decl = probe_decl();
    let syms = |d: &Decl| [("probe_sym", d.clone())];
    let w0 = "    a[0..8].copy_from_slice(&p.to_le_bytes());\n";
    let w1 = "    a[8..12].copy_from_slice(&n.to_le_bytes());\n";
    let w2 = "    a[12..16].copy_from_slice(&r.to_le_bytes());\n";
    let full = probe_bytes_source(&format!("{w0}{w1}{w2}"));
    let (abi, cov) = run_synth_at(PROBE_PATH, &full, &syms(&decl), &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let omitted = probe_bytes_source(&format!("{w0}{w1}"));
    let (abi, _) = run_synth_at(PROBE_PATH, &omitted, &syms(&decl), &no_policy());
    assert!(
        !abi.is_empty(),
        "an unwritten declared kernarg word must fail"
    );
    // An argument the kernel declares as `reserved` may stay unwritten.
    decl.args[2].text = "unsigned int reserved".into();
    let (abi, cov) = run_synth_at(PROBE_PATH, &omitted, &syms(&decl), &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A write that straddles declared arguments is wrong.
    let straddle = probe_bytes_source(&format!(
        "{w0}{w1}    a[10..14].copy_from_slice(&r.to_le_bytes());\n"
    ));
    let (abi, _) = run_synth_at(PROBE_PATH, &straddle, &syms(&decl), &no_policy());
    assert!(
        !abi.is_empty(),
        "a write that matches no declared argument must fail"
    );
    // A too-short byte array cannot hold the declared kernarg.
    let short = full.replace("[0u8; 16]", "[0u8; 12]");
    let (abi, _) = run_synth_at(PROBE_PATH, &short, &syms(&decl), &no_policy());
    assert!(
        !abi.is_empty(),
        "a kernarg array shorter than the declared bytes must fail"
    );
}

fn sliced_source(else_params: &str) -> String {
    format!(
        "impl Gpu {{\n    fn ep(&mut self, wide: bool) -> HipResult<()> {{\n        let a = t.buf.as_ptr();\n        let b = t.buf.as_ptr();\n        let c = t.buf.as_ptr();\n        let name = if wide {{ \"ep2_sym\" }} else {{ \"epn_sym\" }};\n        let mut params = [std::ptr::null_mut::<c_void>(); 3];\n        let used = if wide {{\n            params[..2].copy_from_slice(&[&a as *const _ as *mut c_void, &b as *const _ as *mut c_void]);\n            2\n        }} else {{\n{else_params}        }};\n        self.launch_maybe_blob(name, g, bl, 0, &mut params[..used], || {{\n            let mut x = hip_bridge::KernargBlob::new();\n            x.push_ptr(a);\n            x.push_ptr(b);\n            if wide {{\n            }} else {{\n                x.push_ptr(c);\n            }}\n            x\n        }})\n    }}\n}}\n"
    )
}

#[test]
fn negative_sliced_params_branches_are_checked_per_branch() {
    let two = Decl::from_scalars(&[(8, 'p'), (8, 'p')]);
    let three = Decl::from_scalars(&[(8, 'p'), (8, 'p'), (8, 'p')]);
    let syms = [("ep2_sym", two), ("epn_sym", three)];
    let good = sliced_source(
        "            params[0] = &a as *const _ as *mut c_void;\n            params[1] = &b as *const _ as *mut c_void;\n            params[2] = &c as *const _ as *mut c_void;\n            3\n",
    );
    let (abi, cov) = run_synth(&good, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The wide branch forgets the third params entry while the blob carries it.
    let bad = sliced_source(
        "            params[0] = &a as *const _ as *mut c_void;\n            params[1] = &b as *const _ as *mut c_void;\n            2\n",
    );
    let (abi, _) = run_synth(&bad, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a sliced params branch that omits an argument must fail"
    );
}

// ---- negative controls: hidden helper callers, kernel/layout correlation, zero-init ---

#[test]
fn negative_probe_helper_forwarding_caller_cannot_hide() {
    let syms = [("probe_sym", probe_decl())];
    let helper = |pushes: &str| {
        format!(
            "fn run(hip: &Hip, m: &Module) {{\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    helper(hip, f);\n}}\nfn helper(hip: &Hip, f: Function) {{\n    let mut b = hip_bridge::KernargBlob::new();\n{pushes}    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut b).unwrap();\n}}\n"
        )
    };
    // The forwarded blob is checked against the kernel the CALL names: complete passes ..
    let good = helper("    b.push_ptr(p);\n    b.push_u32(n);\n    b.push_u32(r);\n");
    let (abi, cov) = run_synth_at(PROBE_PATH, &good, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // .. and a short forwarded blob cannot hide behind the caller that mentions the kernel.
    let bad = helper("    b.push_ptr(p);\n");
    let (abi, _) = run_synth_at(PROBE_PATH, &bad, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a short blob forwarded through a helper must fail"
    );
    // A helper nobody calls, which no kernel-mentioning function can reach, has nothing to hide.
    let loader = "fn load(hip: &Hip, m: &Module) {\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    drop(f);\n}\nfn helper(hip: &Hip, f: Function) {\n    let mut b = hip_bridge::KernargBlob::new();\n    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut b).unwrap();\n}\n";
    let (abi, cov) = run_synth_at(PROBE_PATH, loader, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The handle escapes to a function nobody defines here: it could launch the kernel
    // unseen, so the dispatch is unresolved and the gate fails closed.
    let escape = "fn run(hip: &Hip, m: &Module) {\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    unknown_helper(f);\n}\nfn helper(hip: &Hip, f: Function) {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut b).unwrap();\n}\n";
    let (_, cov) = run_synth_at(PROBE_PATH, escape, &syms, &no_policy());
    assert!(
        cov.iter()
            .any(|c| c.contains("escapes to `unknown_helper`")),
        "{cov:?}"
    );
}

#[test]
fn negative_swapped_kernel_to_layout_branches_are_rejected() {
    let two = Decl::from_scalars(&[(8, 'p'), (8, 'p')]);
    let three = Decl::from_scalars(&[(8, 'p'), (8, 'p'), (8, 'p')]);
    let syms = [("ep2_sym", two), ("epn_sym", three)];
    let good = sliced_source(
        "            params[0] = &a as *const _ as *mut c_void;\n            params[1] = &b as *const _ as *mut c_void;\n            params[2] = &c as *const _ as *mut c_void;\n            3\n",
    );
    let (abi, cov) = run_synth(&good, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Both branches are internally consistent, but each launches the other's kernel.
    let swapped = good.replace(
        "if wide { \"ep2_sym\" } else { \"epn_sym\" }",
        "if wide { \"epn_sym\" } else { \"ep2_sym\" }",
    );
    assert_ne!(swapped, good);
    let (abi, _) = run_synth(&swapped, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a kernel/layout branch swap must fail even though each layout matches some candidate"
    );
    // The selector is a helper whose formal differs in name from the caller's condition:
    // correlation follows the ACTUAL argument, so the correct call still passes ..
    let pick = "fn pick(w: bool) -> &'static str {\n    if w { \"ep2_sym\" } else { \"epn_sym\" }\n}\nimpl Gpu {";
    let make = |call: &str| {
        good.replace(
            "let name = if wide { \"ep2_sym\" } else { \"epn_sym\" };",
            &format!("let name = {call};"),
        )
        .replace("impl Gpu {", pick)
    };
    let direct = make("pick(wide)");
    assert_ne!(direct, good);
    // `pick` only returns a kernel name (it launches nothing): classified explicitly.
    let mut pick_pol = Policy::default();
    pick_pol.non_launching.insert(
        "synthetic.rs::pick".into(),
        "name selector helper, launches nothing".into(),
    );
    let (abi, cov) = run_synth(&direct, &syms, &pick_pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // .. and an actual argument that inverts or recomputes the condition no longer
    // matches the layout branches (it must satisfy every candidate, which it cannot).
    for bad in ["pick(!wide)", "pick(if wide { false } else { true })"] {
        let src = make(bad);
        let (abi, _) = run_synth(&src, &syms, &no_policy());
        assert!(!abi.is_empty(), "selector fed `{bad}` must fail");
    }
}

/// A probe whose launch takes its kernel handle from a source the tracer cannot name
/// (`ctx.func()`), so only the symbols the probe mentions identify the candidates.
fn probe_fallback_source(pushes: &str) -> String {
    format!(
        "fn probe(hip: &Hip, ctx: &Ctx) {{\n    let _names = (\"a_sym\", \"b_sym\");\n    let mut b = hip_bridge::KernargBlob::new();\n{pushes}    hip.launch_kernel_blob(ctx.func(), [1, 1, 1], [32, 1, 1], 0, None, &mut b).unwrap();\n}}\n"
    )
}

#[test]
fn negative_nameless_probe_candidates_must_share_one_abi() {
    let pushes = "    b.push_ptr(p);\n    b.push_u32(n);\n    b.push_u32(r);\n";
    let src = probe_fallback_source(pushes);
    let other = Decl::from_scalars(&[(8, 'p'), (8, 'p')]);
    let (_, cov) = run_synth_at(
        PROBE_PATH,
        &src,
        &[("a_sym", probe_decl()), ("b_sym", other)],
        &no_policy(),
    );
    assert!(cov.iter().any(|c| c.contains("different ABIs")), "{cov:?}");
    // Identical candidate ABIs: the layout is checked against all of them.
    let same = [("a_sym", probe_decl()), ("b_sym", probe_decl())];
    let (abi, cov) = run_synth_at(PROBE_PATH, &src, &same, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let bad = probe_fallback_source("    b.push_ptr(p);\n    b.push_u32(n);\n");
    let (abi, _) = run_synth_at(PROBE_PATH, &bad, &same, &no_policy());
    assert!(
        !abi.is_empty(),
        "a layout that is wrong for the shared ABI must fail"
    );
}

#[test]
fn negative_reserved_padding_needs_a_proven_zero_initialised_array() {
    let mut decl = probe_decl();
    decl.args[2].text = "unsigned int reserved".into();
    let syms = [("probe_sym", decl)];
    let w = "    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[8..12].copy_from_slice(&n.to_le_bytes());\n";
    let ok = probe_bytes_source(w);
    let (abi, cov) = run_synth_at(PROBE_PATH, &ok, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Not zero: the unwritten reserved word would carry arbitrary bytes.
    let dirty = ok.replace("[0u8; 16]", "[0xffu8; 16]");
    let (_, cov) = run_synth_at(PROBE_PATH, &dirty, &syms, &no_policy());
    assert!(cov.iter().any(|c| c.contains("UNSUPPORTED")), "{cov:?}");
    // Zero at declaration but modified by something the proof cannot follow.
    let filled = probe_bytes_source(&format!("{w}    a.fill(7);\n"));
    let (_, cov) = run_synth_at(PROBE_PATH, &filled, &syms, &no_policy());
    assert!(cov.iter().any(|c| c.contains("UNSUPPORTED")), "{cov:?}");
}

#[test]
fn negative_old_helper_must_really_build_the_suffixed_name() {
    let helper = |body: &str| {
        format!(
            "const OLD_SUFFIX: &str = \"__beta\";\nimpl Cx {{\n    fn launch_old(&self, sym: &str, blob: KernargBlob) {{\n        let _unused = OLD_SUFFIX;\n{body}    }}\n}}\n"
        )
    };
    let launch = |name: &str| {
        format!("        self.gpu.launch_kernel_blob({name}, [1, 1, 1], [1, 1, 1], 0, &mut blob.into_vec()).unwrap();\n")
    };
    let mut pol = Policy::default();
    pol.old_helpers.insert(
        format!("{PROBE_PATH}::launch_old"),
        "OLD fixture kernels only".into(),
    );
    let syms = [("probe_sym", probe_decl())];
    // Genuine: the launched name is `<sym>{OLD_SUFFIX}`, directly or through one local.
    let direct = helper(&launch("&format!(\"{sym}{OLD_SUFFIX}\")"));
    let (abi, cov) = run_synth_at(PROBE_PATH, &direct, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let via_local = helper(&format!(
        "        let name = format!(\"{{sym}}{{OLD_SUFFIX}}\");\n{}",
        launch("&name")
    ));
    let (abi, cov) = run_synth_at(PROBE_PATH, &via_local, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Forged: the launch uses the CURRENT kernel name; OLD_SUFFIX is still mentioned.
    let forged = helper(&launch("&format!(\"{sym}\")"));
    let (_, cov) = run_synth_at(PROBE_PATH, &forged, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("OLD_SUFFIX")), "{cov:?}");
    // Forged through a local: the suffixed name is built but not the one launched.
    let decoy = helper(&format!(
        "        let name = format!(\"{{sym}}\");\n        let _decoy = format!(\"{{sym}}{{OLD_SUFFIX}}\");\n{}",
        launch("&name")
    ));
    let (_, cov) = run_synth_at(PROBE_PATH, &decoy, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("OLD_SUFFIX")), "{cov:?}");
}

// ---- negative controls: params entries, launch-exact blobs, raw arrays, zip order -----

#[test]
fn negative_params_entries_must_be_understood_and_unmutated() {
    let good = "            b.push_ptr(wp);\n            b.push_i32(nv);\n            b.push_f32(sc);\n            b.push_u64(big);\n";
    let syms = [("small_symbol", small_decl())];
    // Parentheses are normalised, so a correct parenthesised entry passes ..
    let ok_params = SMALL_PARAMS.replacen(
        "&wp as *const _ as *mut c_void",
        "(&wp as *const _ as *mut c_void)",
        1,
    );
    let (abi, cov) = run_synth(
        &small_source(&ok_params, good, SMALL_LETS),
        &syms,
        &no_policy(),
    );
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // .. and a parenthesised entry that points at the wrong local is still caught.
    let bad_params = SMALL_PARAMS.replacen(
        "&wp as *const _ as *mut c_void",
        "(&nv as *const _ as *mut c_void)",
        1,
    );
    let (abi, _) = run_synth(
        &small_source(&bad_params, good, SMALL_LETS),
        &syms,
        &no_policy(),
    );
    assert!(!abi.is_empty(), "a wrong parenthesised entry must fail");
    // An entry shape the reader does not understand is refused, never skipped.
    let unknown =
        SMALL_PARAMS.replacen("&wp as *const _ as *mut c_void", "std::ptr::null_mut()", 1);
    let (_, cov) = run_synth(
        &small_source(&unknown, good, SMALL_LETS),
        &syms,
        &no_policy(),
    );
    assert!(
        cov.iter()
            .any(|c| c.contains("unrecognised kernelParams entry")),
        "{cov:?}"
    );
    // Any mutation other than `push` between declaration and launch is refused.
    let mutated = small_source(SMALL_PARAMS, good, SMALL_LETS).replace(
        "        self.launch_maybe_blob",
        "        params.reverse();\n        self.launch_maybe_blob",
    );
    assert!(mutated.contains("params.reverse()"));
    let (_, cov) = run_synth(&mutated, &syms, &no_policy());
    assert!(cov.iter().any(|c| c.contains("reverse")), "{cov:?}");
}

fn zip_source(zip_iter: &str) -> String {
    format!(
        "impl Gpu {{\n    fn ep(&mut self, wide: bool) -> HipResult<()> {{\n        let ids = t.buf.as_ptr();\n        let tab = t.buf.as_ptr();\n        let srcs: [*mut c_void; 2] = std::array::from_fn(|r| sources[r] as *mut c_void);\n        let name = if wide {{ \"ep4_sym\" }} else {{ \"ep2_sym\" }};\n        let mut params = [std::ptr::null_mut::<c_void>(); 4];\n        let used = if wide {{\n            params[0] = &ids as *const _ as *mut c_void;\n            params[1] = &tab as *const _ as *mut c_void;\n            for (slot, src) in params[2..4].iter_mut().zip({zip_iter}) {{\n                *slot = src as *const _ as *mut c_void;\n            }}\n            4\n        }} else {{\n            params[0] = &ids as *const _ as *mut c_void;\n            params[1] = &tab as *const _ as *mut c_void;\n            2\n        }};\n        self.launch_maybe_blob(name, g, bl, 0, &mut params[..used], || {{\n            let mut x = hip_bridge::KernargBlob::new();\n            x.push_ptr(ids);\n            x.push_ptr(tab);\n            if wide {{\n                for src in srcs {{\n                    x.push_ptr(src);\n                }}\n            }}\n            x\n        }})\n    }}\n}}\n"
    )
}

#[test]
fn negative_params_loop_source_order_must_match_the_blob() {
    let four = Decl::from_scalars(&[(8, 'p'); 4]);
    let two = Decl::from_scalars(&[(8, 'p'); 2]);
    let syms = [("ep4_sym", four), ("ep2_sym", two)];
    let (abi, cov) = run_synth(&zip_source("srcs.iter()"), &syms, &no_policy());
    assert!(
        abi.is_empty() && cov.is_empty(),
        "forward zip order: {abi:?} {cov:?}"
    );
    // Reversed (or otherwise adapted) sources no longer line up with the forward blob loop.
    for adapter in [
        "srcs.iter().rev()",
        "srcs.iter().skip(1)",
        "srcs.iter().step_by(1)",
    ] {
        let (abi, cov) = run_synth(&zip_source(adapter), &syms, &no_policy());
        assert!(
            !abi.is_empty() || cov.iter().any(|c| c.contains("UNSUPPORTED")),
            "`{adapter}` must be rejected: {abi:?} {cov:?}"
        );
    }
}

#[test]
fn negative_probe_blob_is_the_prefix_before_the_launch_with_actual_helper_arguments() {
    let syms = [("probe_sym", probe_decl())];
    let launch =
        "    hip.launch_kernel_blob(f, [1, 1, 1], [32, 1, 1], 0, None, &mut b).unwrap();\n";
    let head = "fn probe(hip: &Hip, m: &Module) {\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n";
    let ok = format!("{head}    b.push_u32(r);\n{launch}}}\n");
    let (abi, cov) = run_synth_at(PROBE_PATH, &ok, &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The last push only happens AFTER the launch: the launched blob is short.
    let late = format!("{head}{launch}    b.push_u32(r);\n}}\n");
    let (abi, _) = run_synth_at(PROBE_PATH, &late, &syms, &no_policy());
    assert!(!abi.is_empty(), "a push after the launch must not count");
    // A helper's Option parameter is bound to the call's actual argument.
    let helper = "fn blob_with(extra: Option<u32>) -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n    b\n}\n";
    let call = |arg: &str| {
        format!(
            "{helper}fn probe(hip: &Hip, m: &Module) {{\n    let f = hip.get_function(m, \"probe_sym\").unwrap();\n    let blob = blob_with({arg});\n    let mut bytes = blob.into_vec();\n    hip.launch_kernel_blob(f, [1, 1, 1], [32, 1, 1], 0, None, &mut bytes).unwrap();\n}}\n"
        )
    };
    let (abi, cov) = run_synth_at(PROBE_PATH, &call("Some(7)"), &syms, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run_synth_at(PROBE_PATH, &call("None"), &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "the short (None) blob is what this launch carries and must fail"
    );
}

#[test]
fn negative_raw_array_writes_are_checked_per_branch_and_cannot_be_clobbered() {
    let mut decl = probe_decl();
    decl.args[2].text = "unsigned int reserved".into();
    let syms = [("probe_sym", probe_decl())];
    let w0 = "    a[0..8].copy_from_slice(&p.to_le_bytes());\n";
    let w1 = "    a[8..12].copy_from_slice(&n.to_le_bytes());\n";
    let w2 = "    a[12..16].copy_from_slice(&r.to_le_bytes());\n";
    let (abi, cov) = run_synth_at(
        PROBE_PATH,
        &probe_bytes_source(&format!("{w0}{w1}{w2}")),
        &syms,
        &no_policy(),
    );
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The last required write only happens on one path: the other path is short.
    let cond = probe_bytes_source(&format!("{w0}{w1}    if flag {{\n    {w2}    }}\n"));
    let (abi, _) = run_synth_at(PROBE_PATH, &cond, &syms, &no_policy());
    assert!(
        !abi.is_empty(),
        "a conditional required write must fail on the path that skips it"
    );
    // Passing the array to anything else before the launch voids the zero-init proof.
    let clobbered = probe_bytes_source(&format!("{w0}{w1}    clobber(&mut a);\n"));
    let (_, cov) = run_synth_at(PROBE_PATH, &clobbered, &[("probe_sym", decl)], &no_policy());
    assert!(cov.iter().any(|c| c.contains("UNSUPPORTED")), "{cov:?}");
}

#[test]
fn negative_old_helper_named_override_and_shadowing_are_rejected() {
    let helper = |sig: &str, body: &str| {
        format!(
            "const OLD_SUFFIX: &str = \"__beta\";\nimpl Cx {{\n    fn launch_old({sig}) {{\n{body}        self.gpu.launch_kernel_blob(&format!(\"{{sym}}{{OLD_SUFFIX}}{{TAIL}}\"), [1, 1, 1], [1, 1, 1], 0, &mut blob.into_vec()).unwrap();\n    }}\n}}\n"
        )
    };
    let mut pol = Policy::default();
    pol.old_helpers.insert(
        format!("{PROBE_PATH}::launch_old"),
        "OLD fixture kernels only".into(),
    );
    let syms = [("probe_sym", probe_decl())];
    let sig = "&self, sym: &str, blob: KernargBlob";
    // The plain helper (template ends with the suffix placeholder) is accepted.
    let plain = helper(sig, "")
        .replace("{{TAIL}}", "")
        .replace("{TAIL}", "");
    let (abi, cov) = run_synth_at(PROBE_PATH, &plain, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Named override: `OLD_SUFFIX = ""` makes the "OLD" name the current kernel.
    let over = plain.replace(
        "format!(\"{sym}{OLD_SUFFIX}\")",
        "format!(\"{sym}{OLD_SUFFIX}\", OLD_SUFFIX = \"\")",
    );
    assert_ne!(over, plain);
    let (_, cov) = run_synth_at(PROBE_PATH, &over, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("OLD_SUFFIX")), "{cov:?}");
    // A local that shadows the const.
    let shadow = helper(sig, "        let OLD_SUFFIX = \"\";\n").replace("{TAIL}", "");
    let (_, cov) = run_synth_at(PROBE_PATH, &shadow, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("OLD_SUFFIX")), "{cov:?}");
    // A parameter that shadows it.
    let param =
        helper("&self, sym: &str, blob: KernargBlob, OLD_SUFFIX: &str", "").replace("{TAIL}", "");
    let (_, cov) = run_synth_at(PROBE_PATH, &param, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("OLD_SUFFIX")), "{cov:?}");
}

#[test]
fn hip_declared_abi_readers_fail_closed_on_arch_macro_and_declarator_traps() {
    // Every registry-supported architecture is evaluated.
    for a in rdna_compute::kernel_registry::SUPPORTED_ARCHES {
        assert!(
            hip::ARCHES.iter().any(|(n, _)| n == a),
            "{a} is registry-supported but never evaluated"
        );
    }
    // A gfx906-only extra argument makes the layout target-dependent: refused.
    let arch906 = "extern \"C\" __global__ void k(int a\n#if defined(__gfx906__)\n, int extra\n#endif\n) {}\n";
    assert!(
        hip::decls_of(arch906).is_err(),
        "gfx906-dependent signature must fail closed"
    );
    // Object macros expand textually in `#if`: `N*3` is 1+2*3 = 7, not (1+2)*3 = 9.
    let prec = "#define N 1+2\n#if N*3 == 7\nextern \"C\" __global__ void k(int a, int extra) {}\n#else\nextern \"C\" __global__ void k(int a) {}\n#endif\n";
    let d = hip::decls_of(prec).expect("parse");
    assert_eq!(
        d["k"].args.len(),
        2,
        "precedence must follow textual expansion"
    );
    // An array parameter adjusts to an 8-byte pointer, not an int.
    let arr = hip::decls_of("extern \"C\" __global__ void k(int a[2]) {}\n").expect("parse");
    assert_eq!((arr["k"].args[0].size, arr["k"].args[0].kind), (8, 'p'));
    // Declarators this reader does not parse are refused instead of guessed.
    assert!(hip::decls_of("extern \"C\" __global__ void k(int (*f)(int)) {}\n").is_err());
    assert!(hip::decls_of("extern \"C\" __global__ void k(int a = 3) {}\n").is_err());
}

// ---- negative controls: the blob operand's origin is proven from the live source ------

const GOOD_FN: &str = "fn good_blob() -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    b.push_u32(r);\n    b\n}\n";
const SHORT_FN: &str = "fn short_blob() -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    b\n}\n";
const GET_F: &str = "    let f = hip.get_function(m, \"probe_sym\").unwrap();\n";

fn launch_with(operand: &str) -> String {
    format!("    hip.launch_kernel_blob(f, [1, 1, 1], [32, 1, 1], 0, None, {operand}).unwrap();\n")
}

fn probe_with(body: &str) -> String {
    format!("{GOOD_FN}{SHORT_FN}fn probe(hip: &Hip, m: &Module) {{\n{GET_F}{body}}}\n")
}

#[test]
fn negative_probe_blob_origin_is_proven_from_the_live_operand_not_a_nearby_builder() {
    let syms = [("probe_sym", probe_decl())];
    let run = |src: &str| run_synth_at(PROBE_PATH, src, &syms, &no_policy());
    // The correct builder exists but the LIVE operand is something the tracer cannot
    // prove: refused, the unused builder is not a substitute.
    let untraceable = probe_with(&format!(
        "    let mut unused = good_blob();\n    let mut bad: Vec<u8> = vec![0u8; 16];\n{}",
        launch_with("&mut bad")
    ));
    let (_, cov) = run(&untraceable);
    assert!(
        cov.iter().any(|c| c.contains("cannot be traced")),
        "{cov:?}"
    );
    // Two builders in one function: only the one actually launched counts.
    let builders = |launched: &str| {
        probe_with(&format!(
            "    let mut good = hip_bridge::KernargBlob::new();\n    good.push_ptr(p);\n    good.push_u32(n);\n    good.push_u32(r);\n    let mut bad = hip_bridge::KernargBlob::new();\n    bad.push_ptr(p);\n    bad.push_u32(n);\n{}",
            launch_with(launched)
        ))
    };
    let (abi, cov) = run(&builders("&mut good"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run(&builders("&mut bad"));
    assert!(
        !abi.is_empty(),
        "launching the short builder must fail although the good one exists"
    );
    // A collection: every element that can reach the launch is checked.
    let coll = |extra: &str| {
        probe_with(&format!(
            "    let mut blobs: Vec<Vec<u8>> = Vec::new();\n    blobs.push(good_blob().into_vec());\n{extra}    for b in blobs.iter_mut() {{\n    {}    }}\n",
            launch_with("b")
        ))
    };
    let (abi, cov) = run(&coll(""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run(&coll("    blobs.push(short_blob().into_vec());\n"));
    assert!(
        !abi.is_empty(),
        "one short element in the collection must fail"
    );
    let (_, cov) = run(&coll("    blobs.extend(others);\n"));
    assert!(cov.iter().any(|c| c.contains("extend")), "{cov:?}");
    // A helper's Option parameter is bound to its actual argument; an unknown actual
    // keeps both branches and the short one fails.
    let helper = "fn blob_with(extra: Option<u32>) -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n    b\n}\n";
    let with_actual = |setup: &str, arg: &str| {
        format!(
            "{helper}fn probe(hip: &Hip, m: &Module) {{\n{GET_F}{setup}    let blob = blob_with({arg});\n    let mut bytes = blob.into_vec();\n{}}}\n",
            launch_with("&mut bytes")
        )
    };
    let (abi, cov) = run(&with_actual("", "Some(1)"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run(&with_actual("    let kx = pick_extra();\n", "kx"));
    assert!(
        !abi.is_empty(),
        "an actual the tracer cannot evaluate must not select the good branch"
    );
    // A parameter operand: every caller of the helper supplies a blob that is checked.
    let param = |callers: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}fn launch_it(hip: &Hip, m: &Module, bytes: &mut Vec<u8>) {{\n{GET_F}{}}}\n{callers}",
            launch_with("bytes")
        )
    };
    let a = "fn a(hip: &Hip, m: &Module) {\n    let mut x = good_blob().into_vec();\n    launch_it(hip, m, &mut x);\n}\n";
    let b = "fn b(hip: &Hip, m: &Module) {\n    let mut y = short_blob().into_vec();\n    launch_it(hip, m, &mut y);\n}\n";
    let (abi, cov) = run(&param(a));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run(&param(&format!("{a}{b}")));
    assert!(
        !abi.is_empty(),
        "a second caller passing a short blob must fail"
    );
    let (_, cov) = run(&param(""));
    assert!(cov.iter().any(|c| c.contains("never passed")), "{cov:?}");
    // A struct field: every initialiser of the field is checked.
    let field = |inits: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}fn go(hip: &Hip, m: &Module, c: &mut Case) {{\n{GET_F}{}}}\n{inits}",
            launch_with("&mut c.blob")
        )
    };
    let one = "fn mk() -> Case {\n    let b = good_blob().into_vec();\n    Case { blob: b }\n}\n";
    let two = "fn mk2() -> Case {\n    let s = short_blob().into_vec();\n    Case { blob: s }\n}\n";
    let (abi, cov) = run(&field(one));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run(&field(&format!("{one}{two}")));
    assert!(
        !abi.is_empty(),
        "a second initialiser with a short blob must fail"
    );
    let (_, cov) = run(&field(""));
    assert!(
        cov.iter().any(|c| c.contains("never initialised")),
        "{cov:?}"
    );
}

#[test]
fn negative_shared_memo_never_mixes_callers_that_pass_the_same_name_with_different_origins() {
    let syms = [("probe_sym", probe_decl())];
    let helper = "fn blob_with(extra: Option<u32>) -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n    b\n}\n";
    // Two probes pass the identical identifier `extra`, bound to different literals.
    let probe = |name: &str, bound: &str| {
        format!(
            "fn {name}(hip: &Hip, m: &Module) {{\n{GET_F}    let extra = {bound};\n    let mut bytes = blob_with(extra).into_vec();\n{}}}\n",
            launch_with("&mut bytes")
        )
    };
    let run = |a: &str, b: &str| {
        run_synth_at(
            PROBE_PATH,
            &format!("{helper}{}{}", probe("pa", a), probe("pb", b)),
            &syms,
            &no_policy(),
        )
    };
    let (abi, cov) = run("Some(1)", "Some(2)");
    assert!(
        abi.is_empty() && cov.is_empty(),
        "both callers pass a value: {abi:?} {cov:?}"
    );
    // Whichever caller is analysed first (and so warms the shared memo), the one that
    // passes `None` launches the short blob and must be reported.
    let (abi, _) = run("Some(1)", "None");
    assert!(
        abi.iter().any(|p| p.contains(" pb:")),
        "second caller (None) must fail: {abi:?}"
    );
    let (abi, _) = run("None", "Some(1)");
    assert!(
        abi.iter().any(|p| p.contains(" pa:")),
        "first caller (None) must fail: {abi:?}"
    );
    assert!(
        !abi.iter().any(|p| p.contains(" pb:")),
        "the good caller must not inherit the bad one: {abi:?}"
    );
}

#[test]
fn negative_stale_literals_and_mutable_aliases_cannot_select_the_good_branch() {
    let syms = [("probe_sym", probe_decl())];
    let helper = "fn blob_with(extra: Option<u32>) -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    if let Some(r) = extra {\n        b.push_u32(r);\n    }\n    b\n}\n";
    let probe = |between: &str, decl: &str| {
        format!(
            "{helper}fn pa(hip: &Hip, m: &Module) {{\n{GET_F}    let {decl} extra = Some(1);\n{between}    let mut bytes = blob_with(extra).into_vec();\n{}}}\n",
            launch_with("&mut bytes")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // The immutable `Some` baseline passes ..
    let (abi, cov) = run(&probe("", ""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // .. an untouched `mut` binding too ..
    let (abi, cov) = run(&probe("", "mut"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // .. but any change between the binding and the call keeps the short `None` branch.
    for between in [
        "    extra = None;\n",
        "    let alias = &mut extra;\n    *alias = None;\n",
        "    reset(&mut extra);\n",
        "    extra.take();\n",
    ] {
        let (abi, _) = run(&probe(between, "mut"));
        assert!(
            !abi.is_empty(),
            "`{between}` must not leave the stale `Some` selected"
        );
    }
}

#[test]
fn negative_mutable_alias_of_a_helper_returned_blob_is_refused() {
    let syms = [("probe_sym", probe_decl())];
    let src = |between: &str| {
        format!(
            "{GOOD_FN}fn pa(hip: &Hip, m: &Module) {{\n{GET_F}    let mut b = good_blob().into_vec();\n{between}{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src(""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    for between in [
        "    let a = &mut b;\n    a.push(0);\n",
        "    b = other();\n",
        "    scribble(&mut b);\n",
        "    b[0] = 9;\n",
    ] {
        let (abi, cov) = run(&src(between));
        assert!(
            !abi.is_empty() || !cov.is_empty(),
            "`{between}` must be refused"
        );
    }
}

/// The external `_t8` family pins the declared 96B/17-arg layout: the fixture
/// loader requires the exact eight-symbol family and the byte proof covers
/// every declared word.
fn t8_external_syms() -> Vec<(String, Decl)> {
    let root = root();
    let mut syms = Symbols::default();
    build_external(
        &root,
        "crates/rdna-compute/tests/fixtures/all-launcher-abi/external_t8_abi.json",
        &mut syms,
    )
    .expect("external fixture loads");
    let mut out: Vec<(String, Decl)> = Vec::new();
    for (name, list) in &syms.pm {
        for (origin, decl) in list {
            if origin.starts_with("external:") {
                out.push((name.clone(), decl.clone()));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn t8_bytes_source(writes: &str) -> String {
    format!(
        "fn probe(hip: &Hip, m: &Module) {{\n    let f = hip.module_get_function(m, \"mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8\").unwrap();\n    let mut ka = [0u8; 96];\n{writes}    hip.launch_kernel_blob(f, [8, 80, 8], [256, 1, 1], 0, None, &mut ka).unwrap();\n}}\n"
    )
}

fn t8_full_writes() -> String {
    let mut w = String::new();
    for i in 0..7 {
        w.push_str(&format!(
            "    ka[{}..{}].copy_from_slice(&p{i}.to_le_bytes());\n",
            8 * i,
            8 * i + 8
        ));
    }
    for i in 0..10 {
        w.push_str(&format!(
            "    ka[{}..{}].copy_from_slice(&v{i}.to_le_bytes());\n",
            56 + 4 * i,
            60 + 4 * i
        ));
    }
    w
}

#[test]
fn negative_external_t8_family_layout_is_pinned() {
    let syms = t8_external_syms();
    assert_eq!(syms.len(), 8, "eight _t8 symbols declared: {syms:?}");
    // All eight share one identical ABI: the hand-handle launch cannot tie a
    // layout variant to one kernel, so any difference must fail.
    for w in syms.windows(2) {
        assert!(
            hip::same_layout(&w[0].1, &w[1].1),
            "{} vs {}",
            w[0].0,
            w[1].0
        );
    }
    for (name, d) in &syms {
        assert_eq!(d.args.len(), 17, "{name}");
        assert_eq!(d.end, 96, "{name}");
        assert_eq!(d.args[0].offset, 0, "{name}");
        assert_eq!(d.args[7].offset, 56, "{name}");
        assert_eq!(d.args[16].offset, 92, "{name}");
    }
    // The full 17-write zero-array proof passes against every declared symbol.
    let full = t8_bytes_source(&t8_full_writes());
    let refs: Vec<(&str, Decl)> = syms.iter().map(|(n, d)| (n.as_str(), d.clone())).collect();
    let (abi, cov) = run_synth_at(PROBE_PATH, &full, &refs, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A missing export fails closed at load time: the fixture controller
    // requires the exact eight-symbol family, so the probe target
    // `mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8` cannot silently drop out of
    // scope. (Filtering it from the synth decl list proves nothing: the
    // scanner only tracks names in its symbol table, so that launch site is
    // then intentionally ignored and both violation lists stay empty.)
    {
        let text = std::fs::read_to_string(
            root().join("crates/rdna-compute/tests/fixtures/all-launcher-abi/external_t8_abi.json"),
        )
        .expect("fixture");
        let mut v: serde_json::Value = serde_json::from_str(&text).expect("parse");
        let map = v["symbols"].as_object_mut().expect("symbols object");
        assert_eq!(map.len(), 8);
        assert!(map
            .remove("mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8")
            .is_some());
        let dir = std::env::temp_dir().join(format!("t8neg-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        std::fs::write(
            dir.join("external_t8_abi.json"),
            serde_json::to_string(&v).unwrap(),
        )
        .expect("write");
        let mut syms = Symbols::default();
        let r = build_external(&dir, "external_t8_abi.json", &mut syms);
        std::fs::remove_dir_all(&dir).ok();
        assert!(r.is_err(), "missing export must fail closed: {r:?}");
    }
    // The zero-initialised `reserved`@92 word may stay unwritten (existing
    // `compare_bytes` reserved handling): the zero-init proof keeps it zero.
    let no_res =
        t8_full_writes().replace("    ka[92..96].copy_from_slice(&v9.to_le_bytes());\n", "");
    let (abi, cov) = run_synth_at(PROBE_PATH, &t8_bytes_source(&no_res), &refs, &no_policy());
    assert!(
        abi.is_empty() && cov.is_empty(),
        "omitted zero-init reserved must pass: {abi:?} {cov:?}"
    );
    // Dropping a live word (n_experts@88) must fail instead.
    let short =
        t8_full_writes().replace("    ka[88..92].copy_from_slice(&v8.to_le_bytes());\n", "");
    let (abi, _) = run_synth_at(PROBE_PATH, &t8_bytes_source(&short), &refs, &no_policy());
    assert!(
        !abi.is_empty(),
        "unwritten n_experts@88 word must fail: {abi:?}"
    );
    // A straddling write matches no declared argument.
    let straddle = format!(
        "{}    ka[86..90].copy_from_slice(&v8.to_le_bytes());\n",
        short
    );
    let (abi, _) = run_synth_at(PROBE_PATH, &t8_bytes_source(&straddle), &refs, &no_policy());
    assert!(!abi.is_empty(), "straddling write must fail: {abi:?}");
    let small = full.replace("[0u8; 96]", "[0u8; 92]");
    let (abi, _) = run_synth_at(PROBE_PATH, &small, &refs, &no_policy());
    assert!(!abi.is_empty(), "92-byte array must fail: {abi:?}");
}

#[test]
fn negative_external_t8_loader_fails_closed() {
    let root = root();
    let good = std::fs::read_to_string(
        root.join("crates/rdna-compute/tests/fixtures/all-launcher-abi/external_t8_abi.json"),
    )
    .expect("fixture");
    let load = |text: &str| {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "t8neg-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let p = dir.join("external_t8_abi.json");
        std::fs::write(&p, text).expect("write");
        let mut syms = Symbols::default();
        let r = build_external(&dir, "external_t8_abi.json", &mut syms);
        std::fs::remove_dir_all(&dir).ok();
        r
    };
    // A missing export is rejected: the family is exactly eight symbols.
    let mut v: serde_json::Value = serde_json::from_str(&good).expect("parse");
    let syms = v["symbols"].as_object().unwrap().clone();
    assert_eq!(syms.len(), 8);
    let dropped = syms.keys().next().unwrap().clone();
    v.as_object_mut().unwrap().remove("symbols").unwrap();
    let mut kept = serde_json::Map::new();
    for (k, s) in syms {
        if k != dropped {
            kept.insert(k, s);
        }
    }
    v.as_object_mut()
        .unwrap()
        .insert("symbols".into(), kept.into());
    assert!(
        load(&serde_json::to_string(&v).unwrap()).is_err(),
        "missing export must fail closed"
    );
    // Wrong offset/size/name/kind/value_kind/segment, unknown fields,
    // wrapping numerics, bad provenance, missing contract, wrong module, and
    // non-family symbols are all hard errors, never silent defaults.
    let mutate = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut w: serde_json::Value = serde_json::from_str(&good).expect("parse");
        f(&mut w);
        load(&serde_json::to_string(&w).unwrap())
    };
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][8]["offset"] = 84.into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][0]["size"] = 4.into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][16]["name"] = "pad".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][0]["kind"] = "v".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][0]["value_kind"] =
            "by_value".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][8]["value_kind"] =
            "global_buffer".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["segment"] = 104.into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][1]["bogus"] = 1.into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w.as_object_mut().unwrap().insert("bogus".into(), 1.into());
    })
    .is_err());
    // A `u64` offset of 2^32 + 0 would wrap to 0 under `as u32`: refused.
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["args"][0]["offset"] =
            4294967296u64.into();
    })
    .is_err());
    // Provenance: truncated/uppercase digests, missing dirs, missing contract.
    assert!(mutate(&|w| {
        w["provenance"]["gfx1201_module_co_sha256"] =
            "c7b528dcee126e0188d43f371dbad4a0f81b0cd962d71c0d392ec8c9865aa5".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["provenance"]["cn_module_co_sha256"] =
            "4CFAAEC12F0585EE1811259429E3BE981DC7F9934A7EB933A591C67C81407FA2".into();
    })
    .is_err());
    assert!(mutate(&|w| {
        w["provenance"]
            .as_object_mut()
            .unwrap()
            .remove("cn_artifact_dir");
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]
            .as_object_mut()
            .unwrap()
            .remove("contract");
    })
    .is_err());
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["contract"] =
            "shapes/other.json".into();
    })
    .is_err());
    // Wrong module association (Cn symbol claimed in the gfx1201 module).
    assert!(mutate(&|w| {
        w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"]["module"] =
            "mq_i4_moe_gemv_pm_gfx1201".into();
    })
    .is_err());
    // A non-family `_t8` stem is refused even with the suffix.
    assert!(mutate(&|w| {
        let s = w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"].clone();
        w["symbols"]
            .as_object_mut()
            .unwrap()
            .insert("mq_i4_moe_gemv_bogus_r256_gate_up_t8".into(), s);
    })
    .is_err());
    // A non-_t8 symbol smuggled into the external family is refused.
    assert!(mutate(&|w| {
        let s = w["symbols"]["mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8"].clone();
        w["symbols"]
            .as_object_mut()
            .unwrap()
            .insert("mq_i4_moe_gemv_plain".into(), s);
    })
    .is_err());
}

#[test]
fn self_referential_format_names_resolve_without_looping() {
    // `let name = format!("{name}_nt4")` reads the PREVIOUS `name`; resolving it must not
    // re-enter its own initialiser.
    let src = "impl Gpu {\n    fn go(&mut self, nt: usize) -> HipResult<()> {\n        let name = \"base_sym\";\n        let name = format!(\"{name}_nt4\");\n        let wp = t.buf.as_ptr();\n        let mut params = [\n            &wp as *const _ as *mut c_void,\n        ];\n        self.launch_maybe_blob(&name, g, b, 0, &mut params, || {\n            let mut x = hip_bridge::KernargBlob::new();\n            x.push_ptr(wp);\n            x\n        })\n    }\n}\n";
    let one = Decl::from_scalars(&[(8, 'p')]);
    let (abi, cov) = run_synth(src, &[("base_sym_nt4", one)], &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
}

#[test]
fn self_shadow_name_keeps_the_suffixed_kernel_as_the_only_target() {
    let src = |n: usize| {
        let params: String = (0..n)
            .map(|i| format!("            &p{i} as *const _ as *mut c_void,\n"))
            .collect();
        let lets: String = (0..n)
            .map(|i| format!("        let p{i} = t.buf.as_ptr();\n"))
            .collect();
        let pushes: String = (0..n)
            .map(|i| format!("            x.push_ptr(p{i});\n"))
            .collect();
        format!("impl Gpu {{\n    fn go(&mut self) -> HipResult<()> {{\n        let name = \"small6\";\n        let name = format!(\"{{name}}_n8\");\n{lets}        let mut params = [\n{params}        ];\n        self.launch_maybe_blob(&name, g, b, 0, &mut params, || {{\n            let mut x = hip_bridge::KernargBlob::new();\n{pushes}            x\n        }})\n    }}\n}}\n")
    };
    let syms = [
        ("small6", Decl::from_scalars(&[(8, 'p'); 6])),
        ("small6_n8", Decl::from_scalars(&[(8, 'p'); 12])),
    ];
    let (abi, cov) = run_synth(&src(12), &syms, &no_policy());
    assert!(
        abi.is_empty() && cov.is_empty(),
        "correct 12-argument launch: {abi:?} {cov:?}"
    );
    // A 6-argument launch would only match the BASE kernel; the launched one is small6_n8.
    let (abi, cov) = run_synth(&src(6), &syms, &no_policy());
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the base ABI must not stand in for the suffixed kernel"
    );
}

#[test]
fn negative_shared_memo_separates_callers_with_the_same_name_bound_to_different_values_or_collections(
) {
    let syms = [("probe_sym", probe_decl())];
    // Same identifier `wide`, bound to true / false, selecting a branch of one helper.
    let helper = "fn blob_w(wide: bool) -> KernargBlob {\n    let mut b = hip_bridge::KernargBlob::new();\n    b.push_ptr(p);\n    b.push_u32(n);\n    if wide {\n        b.push_u32(r);\n    }\n    b\n}\n";
    let probe = |name: &str, bound: &str| {
        format!(
            "fn {name}(hip: &Hip, m: &Module) {{\n{GET_F}    let wide = {bound};\n    let mut bytes = blob_w(wide).into_vec();\n{}}}\n",
            launch_with("&mut bytes")
        )
    };
    let run = |a: &str, b: &str| {
        run_synth_at(
            PROBE_PATH,
            &format!("{helper}{}{}", probe("pa", a), probe("pb", b)),
            &syms,
            &no_policy(),
        )
    };
    let (abi, cov) = run("true", "true");
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, _) = run("true", "false");
    assert!(
        abi.iter().any(|p| p.contains(" pb:")) && !abi.iter().any(|p| p.contains(" pa:")),
        "{abi:?}"
    );
    let (abi, _) = run("false", "true");
    assert!(
        abi.iter().any(|p| p.contains(" pa:")) && !abi.iter().any(|p| p.contains(" pb:")),
        "{abi:?}"
    );
    // Same identifier `blob`, a good collection in one caller and a short one in the other.
    let src = |first: &str, second: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}fn launch_it(hip: &Hip, m: &Module, bytes: &mut Vec<u8>) {{\n{GET_F}{}}}\nfn ca(hip: &Hip, m: &Module) {{\n    let mut blob = {first}().into_vec();\n    launch_it(hip, m, &mut blob);\n}}\nfn cb(hip: &Hip, m: &Module) {{\n    let mut blob = {second}().into_vec();\n    launch_it(hip, m, &mut blob);\n}}\n",
            launch_with("bytes")
        )
    };
    let run2 = |a: &str, b: &str| run_synth_at(PROBE_PATH, &src(a, b), &syms, &no_policy());
    let (abi, cov) = run2("good_blob", "good_blob");
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The launcher is shared, so the short blob of either caller is reported, in both orders.
    let (abi, _) = run2("good_blob", "short_blob");
    assert!(!abi.is_empty(), "short second caller must fail");
    let (abi, _) = run2("short_blob", "good_blob");
    assert!(!abi.is_empty(), "short first caller must fail");
}

// ---- negative controls: per-call pairing of kernel, flag and blob -----------------------

fn gl_l_decls() -> [(&'static str, Decl); 2] {
    [
        ("k_gl", probe_decl()),
        ("k_l", Decl::from_scalars(&[(8, 'p'), (4, 'i')])),
    ]
}

#[test]
fn negative_tuple_loop_rows_keep_each_kernel_with_its_own_flag_and_blob() {
    let src = |rows: &str| {
        format!(
            r#"const GL_FN: &str = "k_gl";
const L_FN: &str = "k_l";
fn gl_blob() -> KernargBlob {{
    let mut b = hip_bridge::KernargBlob::new();
    b.push_ptr(p);
    b.push_u32(n);
    b.push_u32(r);
    b
}}
fn l_blob() -> KernargBlob {{
    let mut b = hip_bridge::KernargBlob::new();
    b.push_ptr(p);
    b.push_u32(n);
    b
}}
fn mk() -> Case {{
    Case {{ blob_gl: gl_blob().into_vec(), blob_l: l_blob().into_vec() }}
}}
fn run_helper(gpu: &Gpu, func: &str, c: &Case, gl: bool) {{
    let mut blob = if gl {{ c.blob_gl.clone() }} else {{ c.blob_l.clone() }};
    gpu.launch_kernel_blob(func, g, bl, 0, &mut blob).unwrap();
}}
fn driver(gpu: &Gpu, c: &Case) {{
    for (gl, name) in [{rows}] {{
        run_helper(gpu, name, c, gl);
    }}
}}
"#
        )
    };
    let run = |rows: &str| run_synth_at(PROBE_PATH, &src(rows), &gl_l_decls(), &no_policy());
    let (abi, cov) = run("(true, GL_FN), (false, L_FN)");
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The flag and the kernel swapped: the GL kernel receives the Lloyd blob (and vice versa).
    let (abi, cov) = run("(false, GL_FN), (true, L_FN)");
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "swapped tuple rows must fail"
    );
    // A row naming both kernels cannot be tied to one blob layout.
    let (abi, cov) = run("(true, GL_FN), (true, L_FN)");
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a wrong flag for the second row must fail"
    );
}

#[test]
fn negative_helper_blob_comes_from_the_object_the_call_passes_not_a_same_named_field() {
    let src = |first: &str, second: &str| {
        format!(
            r#"const GL_FN: &str = "k_gl";
const L_FN: &str = "k_l";
struct Up {{ args: Vec<u8> }}
fn upload(gl: bool) -> Up {{
    let mut b = hip_bridge::KernargBlob::new();
    b.push_ptr(p);
    b.push_u32(n);
    if gl {{
        b.push_u32(r);
    }}
    Up {{ args: b.into_vec() }}
}}
fn check(gpu: &Gpu, func: &str, up: &mut Up) {{
    gpu.launch_kernel_blob(func, g, bl, 0, &mut up.args).unwrap();
}}
fn go(gpu: &Gpu) {{
    let mut up_gl = upload(true);
    let mut up_l = upload(false);
    check(gpu, GL_FN, &mut {first});
    check(gpu, L_FN, &mut {second});
}}
"#
        )
    };
    let run = |a: &str, b: &str| run_synth_at(PROBE_PATH, &src(a, b), &gl_l_decls(), &no_policy());
    let (abi, cov) = run("up_gl", "up_l");
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The correct objects both exist, but the calls pass the other one.
    let (abi, cov) = run("up_l", "up_gl");
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "passing the other object must fail"
    );
    let (abi, cov) = run("up_l", "up_l");
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the GL kernel launched with the Lloyd object must fail"
    );
}

#[test]
fn negative_pointer_array_elements_keep_the_origin_of_each_caller() {
    let src = |first: &str, second: &str| {
        format!(
            r#"fn blob2(ptrs: [u64; 2], n: i32) -> KernargBlob {{
    let mut b = hip_bridge::KernargBlob::new();
    for p in ptrs {{
        b.push_u64(p);
    }}
    b.push_i32(n);
    b
}}
fn user(hip: &Hip, m: &Module) {{
    let f = hip.get_function(m, "probe_sym").unwrap();
    let a = {first};
    let c = {second};
    let mut bytes = blob2([a, c], 1).into_vec();
    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut bytes).unwrap();
}}
"#
        )
    };
    let decl = Decl::from_scalars(&[(8, 'p'), (8, 'p'), (4, 'i')]);
    let run = |a: &str, b: &str| {
        run_synth_at(
            PROBE_PATH,
            &src(a, b),
            &[("probe_sym", decl.clone())],
            &no_policy(),
        )
    };
    let (abi, cov) = run("x.buf.as_ptr() as u64", "y.buf.as_ptr() as u64");
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A plain integer in any slot is not a pointer, wherever it sits.
    let (abi, _) = run("7u64", "y.buf.as_ptr() as u64");
    assert!(!abi.is_empty(), "an integer in the first slot must fail");
    let (abi, _) = run("x.buf.as_ptr() as u64", "7u64");
    assert!(!abi.is_empty(), "an integer in the last slot must fail");
}

#[test]
fn negative_prior_launch_helpers_are_reads_but_unknown_aliases_are_not() {
    let src = |between: &str| {
        format!(
            r#"fn touch(gpu: &Gpu, f: &str, blob: &mut Vec<u8>) {{
    gpu.launch_kernel_blob(f, g, bl, 0, blob).unwrap();
}}
fn pad(blob: &mut Vec<u8>) {{
    blob.push(0);
}}
fn user(hip: &Hip, m: &Module) {{
    let f = hip.get_function(m, "probe_sym").unwrap();
    let mut b = good_blob().into_vec();
{between}    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut b).unwrap();
}}
{GOOD_FN}"#
        )
    };
    let syms = [("probe_sym", probe_decl())];
    let run = |s: &str| run_synth_at(PROBE_PATH, &s, &syms, &no_policy());
    let (abi, cov) = run(&src(""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // An earlier call of a helper that only launches the blob is a read.
    let (abi, cov) = run(&src("    touch(gpu, \"probe_sym\", &mut b);\n"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A helper that changes it, or one the tracer cannot see into, is not.
    let (abi, cov) = run(&src("    pad(&mut b);\n"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a mutating helper must be refused"
    );
    let (abi, cov) = run(&src("    mystery(&mut b);\n"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "an unknown callee must be refused"
    );
}

// ---- negative controls: audited OLD-baseline launch chains ---------------------------------

const AUDIT_HELPER: &str = "fn launch_raw(gpu: &Gpu, f: &str, blob: &mut Vec<u8>) {\n    gpu.launch_kernel_blob(f, g, bl, 0, blob).unwrap();\n}\n";

fn audit_source(old_bytes: &str, extra_caller: &str) -> String {
    format!(
        "{AUDIT_HELPER}fn old_blob() -> Vec<u8> {{\n    {old_bytes}\n}}\nfn old_case(gpu: &Gpu) {{\n    let mut b = old_blob();\n    launch_raw(gpu, \"probe_sym\", &mut b);\n}}\n{extra_caller}"
    )
}

/// Pin the chain of `src` the way the inventory does.
fn audit_policy(src: &str) -> Policy {
    let world = World::from_sources(
        Path::new("."),
        vec![(PROBE_PATH.to_string(), src.to_string())],
    )
    .expect("lexes");
    let file = world.file(PROBE_PATH).expect("file");
    let fingerprints: BTreeMap<String, String> = ["launch_raw", "old_blob", "old_case"]
        .iter()
        .map(|n| {
            (
                n.to_string(),
                fingerprint(file.fns.iter().find(|f| f.name == *n).expect("fn")),
            )
        })
        .collect();
    let mut pol = Policy::default();
    pol.old_audited.insert(
        format!("{PROBE_PATH}::launch_raw"),
        OldAudit {
            reason: "OLD fixture launches only".into(),
            baseline: "f1a457b".into(),
            callers: vec!["old_case".into()],
            fingerprints,
            guard_idents: Vec::new(),
            scopes: Vec::new(),
        },
    );
    pol
}

#[test]
fn negative_audited_old_chain_exempts_only_the_pinned_chain() {
    // The OLD helper launches a short blob for a changed kernel: a real violation without the audit.
    let syms = [("probe_sym", probe_decl())];
    let base = audit_source("std::fs::read(old_dir.join(\"old.hxaco\")).unwrap()", "");
    let (abi, cov) = run_synth_at(PROBE_PATH, &base, &syms, &no_policy());
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "without the audit the helper is checked and fails"
    );
    let pol = audit_policy(&base);
    let (abi, cov) = run_synth_at(PROBE_PATH, &base, &syms, &pol);
    assert!(
        abi.is_empty() && cov.is_empty(),
        "the audited chain is exempt: {abi:?} {cov:?}"
    );
    // Whitespace and comments are not a change.
    let reformatted = base.replace("std::fs::read(", "std::fs::read( /* same */ ");
    let (abi, cov) = run_synth_at(PROBE_PATH, &reformatted, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The loader now reads the CURRENT bytes: the pin no longer matches, so nothing is exempt.
    let current = audit_source("std::fs::read(current_image()).unwrap()", "");
    let (_, cov) = run_synth_at(PROBE_PATH, &current, &syms, &pol);
    assert!(
        cov.iter()
            .any(|c| c.contains("changed") && c.contains("old_blob")),
        "{cov:?}"
    );
    // A new caller of the helper that was never audited.
    let newcaller = audit_source("std::fs::read(old_dir.join(\"old.hxaco\")).unwrap()", "fn fresh(gpu: &Gpu) {\n    let mut b = cur();\n    launch_raw(gpu, \"probe_sym\", &mut b);\n}\n");
    let (_, cov) = run_synth_at(PROBE_PATH, &newcaller, &syms, &pol);
    assert!(cov.iter().any(|c| c.contains("callers")), "{cov:?}");
    // The audited caller itself changes (a NEW-path launch smuggled in).
    let edited = base.replace("launch_raw(gpu, \"probe_sym\", &mut b);", "launch_raw(gpu, \"probe_sym\", &mut b);\n    gpu.launch_kernel_blob(\"probe_sym\", g, bl, 0, &mut b).unwrap();");
    let (_, cov) = run_synth_at(PROBE_PATH, &edited, &syms, &pol);
    assert!(
        cov.iter()
            .any(|c| c.contains("changed") && c.contains("old_case")),
        "{cov:?}"
    );
    // The helper reached WITHOUT a direct call inside an audited caller: a function-pointer
    // alias, a macro argument, or a reference from another function.
    for stray in [
        "fn sneak(gpu: &Gpu) {\n    let f = launch_raw;\n    let mut b = cur();\n    f(gpu, \"probe_sym\", &mut b);\n}\n",
        "macro_rules! go {\n    ($g:expr) => { launch_raw($g, \"probe_sym\", &mut cur()) };\n}\n",
        "fn macro_user(gpu: &Gpu) {\n    run_it!(launch_raw, gpu);\n}\n",
    ] {
        let src = audit_source("std::fs::read(old_dir.join(\"old.hxaco\")).unwrap()", stray);
        let (_, cov) = run_synth_at(PROBE_PATH, &src, &syms, &pol);
        assert!(cov.iter().any(|c| c.contains("beyond its definition")), "{stray}: {cov:?}");
    }
}

#[test]
fn negative_closure_parameter_shadows_the_outer_blob_collection() {
    let syms = [("probe_sym", probe_decl())];
    let src = |decls: &str, call: &str, inner: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut blobs: Vec<Vec<u8>> = vec![good_blob().into_vec()];\n{decls}    let time_one = |hip: &Hip, vi: usize, blobs: &mut Vec<Vec<u8>>| {{\n{inner}        hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut blobs[vi]).unwrap();\n    }};\n    {call}\n}}\n"
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // The blob the closure launches is the one the call passes: the (good) outer collection.
    let (abi, cov) = run(&src("", "time_one(hip, 0, &mut blobs);", ""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A harmless use of the SHADOWING parameter inside the closure does not taint the outer one.
    let (abi, cov) = run(&src(
        "",
        "time_one(hip, 0, &mut blobs);",
        "        let _n = blobs.len();\n",
    ));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Another call passes a short blob: the closure launches it, so it must fail even though
    // the good outer collection exists.
    let other = "    let mut other: Vec<Vec<u8>> = vec![short_blob().into_vec()];\n";
    let (abi, cov) = run(&src(other, "time_one(hip, 0, &mut other);", ""));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the passed (short) collection is what is launched"
    );
    // A closure WITHOUT such a parameter that mutates the captured outer collection changes
    // what is launched: refused.
    let captured = format!(
        "{GOOD_FN}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut blobs: Vec<Vec<u8>> = vec![good_blob().into_vec()];\n    let mut wipe = || {{\n        blobs.clear();\n    }};\n    hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, &mut blobs[0]).unwrap();\n}}\n"
    );
    let (abi, cov) = run(&captured);
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a captured mutation of the launched collection must fail"
    );
}

// ---- tuple-let projection, const-table fields, closure readers, field moves ---------------

#[test]
fn tuple_let_names_come_from_their_own_component_and_const_table_fields_are_followed() {
    let src = |l_extra: &str| {
        format!(
            r#"struct Entry {{ down: &'static str }}
const fn entry(down: &'static str) -> Entry {{ Entry {{ down }} }}
const TABLE: [Entry; 2] = [entry("k_gl"), entry("k_l")];
fn launch(gpu: &Gpu, f: &str, args: &mut KernargBlob) {{
    gpu.launch_kernel_blob(f, g, bl, 0, args.as_mut_slice()).unwrap();
}}
fn go(gpu: &mut Gpu, e: Entry, down: bool) {{
    let (f, n) = if down {{ (e.down, 1) }} else {{ (e.down, 2) }};
    let mut a = KernargBlob::new();
    a.push_ptr(p);
    a.push_u32(r);
{l_extra}    launch(gpu, f, &mut a);
}}
fn driver() {{
    let mut gpu = Gpu::init().unwrap();
    go(&mut gpu, TABLE[0], true);
}}
"#
        )
    };
    let two = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let ok = [("k_gl", two.clone()), ("k_l", two.clone())];
    let (abi, cov) = run_synth_at(PROBE_PATH, &src(""), &ok, &no_policy());
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The table's second kernel declares one more argument than the blob carries.
    let bad = [
        ("k_gl", two.clone()),
        ("k_l", Decl::from_scalars(&[(8, 'p'), (4, 'i'), (4, 'i')])),
    ];
    let (abi, cov) = run_synth_at(PROBE_PATH, &src(""), &bad, &no_policy());
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a table kernel the blob does not fit must fail"
    );
}

#[test]
fn negative_closure_callee_is_a_read_only_if_its_body_only_launches_the_blob() {
    let syms = [("probe_sym", probe_decl())];
    let src = |body: &str| {
        format!(
            "{GOOD_FN}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut b = good_blob().into_vec();\n    let bench = |hip: &Hip, bytes: &mut [u8], n: usize| {{\n{body}    }};\n    bench(hip, &mut b, 3);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let ok = "        for _ in 0..n {\n            hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, bytes).unwrap();\n        }\n";
    let (abi, cov) = run(&src(ok));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let bad = "        bytes.push(0);\n        hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, bytes).unwrap();\n";
    let (abi, cov) = run(&src(bad));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a closure that changes the blob is not a read"
    );
}

#[test]
fn negative_blob_moved_into_a_struct_field_is_traced_to_each_constructor() {
    let syms = [("probe_sym", probe_decl())];
    let src = |extra: &str, build: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}struct Case {{ gl_args: Vec<u8> }}\nfn build() -> Case {{\n    let gl_args = good_blob().into_vec();\n    Case {{ gl_args }}\n}}\n{extra}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let a = {build}();\n    let mut g = a.gl_args.clone();\n{}}}\n",
            launch_with("&mut g")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src("", "build"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let other = "fn build2() -> Case {\n    let gl_args = short_blob().into_vec();\n    Case { gl_args }\n}\n";
    let (abi, cov) = run(&src(other, "build2"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the short constructor's blob is what this call launches"
    );
}

#[test]
fn negative_struct_field_blob_is_checked_from_the_constructor_to_the_launch() {
    let syms = [("probe_sym", probe_decl())];
    let src = |between: &str| {
        format!(
            "{GOOD_FN}struct Case {{ gl_args: Vec<u8>, other: u32 }}\nimpl Case {{\n    fn peek(&self) -> u32 {{ self.other }}\n    fn wipe(&mut self) {{ self.gl_args.clear(); }}\n}}\nfn build() -> Case {{\n    let gl_args = good_blob().into_vec();\n    Case {{ gl_args, other: 1 }}\n}}\nfn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut c = build();\n{between}{}}}\n",
            launch_with("&mut c.gl_args")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src(""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // Other fields may change, reads and clones of the blob field are fine, `&self` methods too.
    let (abi, cov) = run(&src("    c.other = 7;\n    let _n = c.gl_args.len();\n    let _s = c.gl_args.clone();\n    let _p = c.peek();\n"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    for between in [
        "    c.gl_args.clear();\n",
        "    c.gl_args.as_mut_slice().fill(0);\n",
        "    let alias = c.gl_args.as_mut_slice();\n    alias.fill(0);\n",
        "    for x in c.gl_args.iter_mut() {\n            *x = 0;\n        }\n",
        "    let alias = &mut c.gl_args;\n    alias.clear();\n",
        "    c.gl_args = Vec::new();\n",
        "    c.gl_args[0] = 9;\n",
        "    let moved = c.gl_args;\n",
        "    c.wipe();\n",
        "    scribble(&mut c);\n",
        "    c = other_case();\n",
    ] {
        let (abi, cov) = run(&src(between));
        assert!(
            !abi.is_empty() || !cov.is_empty(),
            "`{between}` must be refused"
        );
    }
}

#[test]
fn negative_shadowed_closure_is_judged_at_the_call_not_by_the_last_binding() {
    let syms = [("probe_sym", probe_decl())];
    let src = |first: &str, second: &str| {
        format!(
            "{GOOD_FN}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut b = good_blob().into_vec();\n    let touch = |bytes: &mut [u8]| {first};\n    touch(&mut b);\n    let touch = |bytes: &mut [u8]| {second};\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // A closure that only launches the blob, shadowed later by an empty one: fine.
    let launches = "{ hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, bytes).unwrap(); }";
    let (abi, cov) = run(&src(launches, "{}"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A mutating closure at the call, followed by a harmless shadow: refused.
    let (abi, cov) = run(&src("bytes.fill(0)", "{}"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the closure visible at the call mutates the blob"
    );
}

#[test]
fn negative_mutable_views_of_a_local_blob_must_be_launch_arguments() {
    let syms = [("probe_sym", probe_decl())];
    let src = |between: &str| {
        format!(
            "{GOOD_FN}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut b = good_blob().into_vec();\n{between}{}}}\n",
            launch_with("b.as_mut_slice()")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src("    let _n = b.len();\n    let _c = b.clone();\n"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    for between in [
        "    b.as_mut_slice().fill(0);\n",
        "    let alias = b.as_mut_slice();\n    alias.fill(0);\n",
        "    for x in b.iter_mut() {\n        *x = 0;\n    }\n",
        "    let mut it = b.iter_mut();\n    *it.next().unwrap() = 0;\n",
    ] {
        let (abi, cov) = run(&src(between));
        assert!(
            !abi.is_empty() || !cov.is_empty(),
            "`{between}` must be refused"
        );
    }
}

#[test]
fn negative_old_handle_sources_must_be_pinned_functions() {
    // Every function that mentions an OLD-handle source ident must be pinned by the audit,
    // wherever it is: an unpinned one could feed the exempt helper a different image.
    let syms = [("probe_sym", probe_decl())];
    let base = audit_source("std::fs::read(old_dir.join(\"old.hxaco\")).unwrap()", "");
    let mut pol = audit_policy(&base);
    for au in pol.old_audited.values_mut() {
        au.guard_idents = vec!["old_image_handle".to_string()];
    }
    let (abi, cov) = run_synth_at(PROBE_PATH, &base, &syms, &pol);
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let stray = audit_source(
        "std::fs::read(old_dir.join(\"old.hxaco\")).unwrap()",
        "fn other(gpu: &Gpu) {\n    let h = old_image_handle(gpu);\n}\n",
    );
    let (_, cov) = run_synth_at(PROBE_PATH, &stray, &syms, &pol);
    assert!(
        cov.iter()
            .any(|c| c.contains("old_image_handle") && c.contains("not a pinned function")),
        "{cov:?}"
    );
}

#[test]
fn negative_whole_object_passed_on_must_leave_the_blob_field_alone() {
    let syms = [("probe_sym", probe_decl())];
    let src = |between: &str| {
        // `fwd_obj` only exists in the variants that call it (an unused helper forwarding an
        // unsupplied kernel name is a different, correct refusal).
        let fwd = if between.contains("fwd_obj") {
            "fn fwd_obj(gpu: &Gpu, f: &str, up: &mut Up) {\n    peek_obj(gpu, f, up);\n}\n"
        } else {
            ""
        };
        format!(
            "{GOOD_FN}struct Up {{ args: Vec<u8>, y: u32 }}\nfn build() -> Up {{\n    Up {{ args: good_blob().into_vec(), y: 1 }}\n}}\nfn peek_obj(gpu: &Gpu, f: &str, up: &mut Up) {{\n    let _n = up.y;\n    gpu.launch_kernel_blob(f, g, bl, 0, &mut up.args).unwrap();\n}}\nfn wipe_obj(gpu: &Gpu, f: &str, up: &mut Up) {{\n    up.args.clear();\n}}\n{fwd}fn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut c = build();\n{between}{}}}\n",
            launch_with("&mut c.args")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    for ok in [
        "    peek_obj(hip, \"probe_sym\", &mut c);\n",
        "    fwd_obj(hip, \"probe_sym\", &mut c);\n",
    ] {
        let (abi, cov) = run(&src(ok));
        assert!(abi.is_empty() && cov.is_empty(), "`{ok}`: {abi:?} {cov:?}");
    }
    for bad in [
        "    wipe_obj(hip, \"probe_sym\", &mut c);\n",
        "    mystery_obj(hip, &mut c);\n",
    ] {
        let (abi, cov) = run(&src(bad));
        assert!(
            !abi.is_empty() || !cov.is_empty(),
            "`{bad}` must be refused"
        );
    }
}

#[test]
fn negative_padded_kernarg_array_needs_exact_writes() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("probe_sym", decl)];
    let src = |min: &str, second: &str| {
        format!(
            "fn kernarg(p: u64, n: u32, len: usize) -> Vec<u8> {{\n    let mut a = vec![0u8; len.max({min})];\n    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[{second}].copy_from_slice(&n.to_le_bytes());\n    a\n}}\nfn user(hip: &Hip, m: &Module) {{\n{GET_F}    let mut b = kernarg(x.buf.as_ptr() as u64, 3, 12);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src("12", "8..12"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, cov) = run(&src("12", "8..10"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a short write must fail"
    );
}

#[test]
fn negative_nested_for_pattern_projects_through_zip_and_enumerate() {
    let syms = [("probe_sym", probe_decl())];
    let src = |second: &str| {
        format!(
            "{GOOD_FN}{SHORT_FN}fn one(hip: &Hip, m: &Module, launches: &[u32], blobs: &mut [Vec<u8>]) {{\n{GET_F}    for (i, (l, blob)) in launches.iter().zip(blobs.iter_mut()).enumerate() {{\n        hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, blob).unwrap();\n    }}\n}}\nfn driver(hip: &Hip, m: &Module) {{\n    let launches = [1u32, 2];\n    let mut blobs = vec![good_blob().into_vec(), {second}.into_vec()];\n    one(hip, m, &launches, &mut blobs);\n}}\n"
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let (abi, cov) = run(&src("good_blob()"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    let (abi, cov) = run(&src("short_blob()"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "the second element is what is launched second"
    );
}

#[test]
fn negative_run_time_padded_kernarg_length_must_be_one_proven_size() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("probe_sym", decl)];
    let src = |pre: &str, len_arg: &str| {
        format!(
            "fn kernarg(p: u64, n: u32, len: usize) -> Vec<u8> {{\n    let mut a = vec![0u8; len.max(12)];\n    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[8..12].copy_from_slice(&n.to_le_bytes());\n    a\n}}\nfn user(hip: &Hip, m: &Module, free: usize) {{\n{GET_F}{pre}    let mut b = kernarg(x.buf.as_ptr() as u64, 3, {len_arg});\n{}}}\nfn driver(hip: &Hip, m: &Module) {{\n    user(hip, m, 16);\n    user(hip, m, 24);\n}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // The padded length is a literal that equals the declared size.
    let (abi, cov) = run(&src("", "12"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A local 128 pads the blob to 128 bytes against a 12-byte declaration: refused.
    let (abi, cov) = run(&src("    let len = 128;\n", "len"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "128 against a 12-byte declaration must fail"
    );
    // An arbitrary runtime length (a parameter another caller sets to 16) is not one proven size.
    let (abi, cov) = run(&src("", "free"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "an unproven length must fail"
    );
}

#[test]
fn negative_struct_field_length_is_traced_from_the_receiver_not_the_type_name() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("probe_sym", decl)];
    let src = |pad_arg: &str, extra: &str| {
        format!(
            "struct Pad {{ len: usize }}\nfn unrelated() {{\n    let _p = Pad {{ len: 12 }};\n}}\n{extra}fn kernarg(p: u64, n: u32, pad: &Pad) -> Vec<u8> {{\n    let mut a = vec![0u8; pad.len.max(12)];\n    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[8..12].copy_from_slice(&n.to_le_bytes());\n    a\n}}\nfn user(hip: &Hip, m: &Module) {{\n{GET_F}    {pad_arg}\n    let mut b = kernarg(x.buf.as_ptr() as u64, 3, &pad);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // The receiver is constructed right here with the declared size: proven.
    let (abi, cov) = run(&src("let pad = Pad { len: 12 };", ""));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The receiver comes from a constructor the tracer cannot see; an unrelated literal of the
    // same type with the right value must not stand in for it.
    let (abi, cov) = run(&src("let pad = opaque_external_constructor();", ""));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "an unknown constructor must fail"
    );
    // Same through a parameter nobody passes.
    let orphan = "fn orphan(hip: &Hip, m: &Module, pad: &Pad) {\n    let mut b = kernarg(x.buf.as_ptr() as u64, 3, pad);\n}\n";
    let (abi, cov) = run(&src("let pad = Pad { len: 12 };", orphan));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a caller that passes an unproven receiver must fail"
    );
}

#[test]
fn negative_clone_of_a_struct_is_identity_only_when_derived() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("probe_sym", decl)];
    let src = |attr: &str, custom: &str, make: &str| {
        format!(
            "{attr}struct Pad {{ len: usize }}\n{custom}fn kernarg(p: u64, n: u32, pad: &Pad) -> Vec<u8> {{\n    let mut a = vec![0u8; pad.len.max(12)];\n    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[8..12].copy_from_slice(&n.to_le_bytes());\n    a\n}}\nfn user(hip: &Hip, m: &Module) {{\n{GET_F}    let pad = {make};\n    let mut b = kernarg(x.buf.as_ptr() as u64, 3, &pad);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // A derived Clone is the same value.
    let (abi, cov) = run(&src("#[derive(Clone)]\n", "", "Pad { len: 12 }.clone()"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // A hand-written Clone can return anything: not the literal that was cloned.
    let custom =
        "impl Clone for Pad {\n    fn clone(&self) -> Self {\n        Pad { len: 128 }\n    }\n}\n";
    let (abi, cov) = run(&src("", custom, "Pad { len: 12 }.clone()"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a custom Clone must not be assumed identity"
    );
    // No visible derive at all: not provable either.
    let (abi, cov) = run(&src("", "", "Pad { len: 12 }.clone()"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "an unproven Clone must fail"
    );
}

#[test]
fn negative_expression_bodied_closures_keep_every_branch() {
    let syms = [("probe_sym", probe_decl())];
    let src = |body: &str| {
        format!(
            "{GOOD_FN}fn user(hip: &Hip, m: &Module, flag: bool) {{\n{GET_F}    let mut b = good_blob().into_vec();\n    let touch = |bytes: &mut Vec<u8>| {body};\n    touch(&mut b);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // An `if` expression that mutates in its FIRST branch is not a read, whatever the last
    // brace group says.
    let (abi, cov) = run(&src("if flag { bytes.clear() } else { }"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a mutating first branch must be refused"
    );
    // Both branches harmless: a read.
    let (abi, cov) = run(&src("if flag { drop(1) } else { }"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
}

#[test]
fn bare_match_closure_values_are_the_successful_arms() {
    let decl = Decl::from_scalars(&[(8, 'p'), (4, 'i')]);
    let syms = [("probe_sym", decl)];
    let src = |ok_arm: &str| {
        format!(
            "fn kernarg(p: u64, n: u32, len: usize) -> Vec<u8> {{\n    let mut a = vec![0u8; len.max(12)];\n    a[0..8].copy_from_slice(&p.to_le_bytes());\n    a[8..12].copy_from_slice(&n.to_le_bytes());\n    a\n}}\nfn user(hip: &Hip, m: &Module) -> Result<()> {{\n{GET_F}    let size = |s: u32| match probe(s) {{ Some({ok_arm}) => Ok({ok_arm}), other => Err(other) }};\n    let len = size(1)?;\n    let mut b = kernarg(x.buf.as_ptr() as u64, 3, len);\n{}}}\n",
            launch_with("&mut b")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    // The only successful value is 12, the declared size; `Err(..)` is no value.
    let (abi, cov) = run(&src("12"));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    // The only successful value is 40: not the declared size.
    let (abi, cov) = run(&src("40"));
    assert!(
        !abi.is_empty() || !cov.is_empty(),
        "a successful 40 against a 12-byte declaration must fail"
    );
}

#[test]
fn negative_collection_helper_with_nested_for_leaf_is_a_read_only_if_the_leaf_only_launches() {
    let syms = [("probe_sym", probe_decl())];
    let src = |leaf_body: &str| {
        format!(
            "{GOOD_FN}fn one(hip: &Hip, m: &Module, launches: &[u32], blobs: &mut [Vec<u8>]) {{\n{GET_F}    for (i, (l, blob)) in launches.iter().zip(blobs.iter_mut()).enumerate() {{\n{leaf_body}    }}\n}}\nfn user(hip: &Hip, m: &Module) {{\n{GET_F}    let launches = [1u32, 2];\n    let mut blobs = vec![good_blob().into_vec(), good_blob().into_vec()];\n    one(hip, m, &launches, &mut blobs);\n{}}}\n",
            launch_with("&mut blobs[0]")
        )
    };
    let run = |s: &str| run_synth_at(PROBE_PATH, s, &syms, &no_policy());
    let launch =
        "        hip.launch_kernel_blob(f, [1, 1, 1], [1, 1, 1], 0, None, blob).unwrap();\n";
    let (abi, cov) = run(&src(launch));
    assert!(abi.is_empty() && cov.is_empty(), "{abi:?} {cov:?}");
    for hidden in [
        format!("        blob.clear();\n{launch}"),
        format!("        let alias = blob;\n        alias.clear();\n{launch}"),
        format!("        {launch}        blob.push(0);\n"),
    ] {
        let (abi, cov) = run(&src(&hidden));
        assert!(
            !abi.is_empty() || !cov.is_empty(),
            "`{hidden}` must be refused"
        );
    }
}

// ---- canonical immutable-array kernelParams map/append -----------------------

fn canonical_params_map_source() -> String {
    r#"impl Gpu {
    fn mapped(&mut self) -> HipResult<()> {
        let pointers = [x.buf.as_ptr(), y.buf.as_ptr()];
        let dims = [3i32, 7i32];
        let mut params: Vec<*mut c_void> = pointers.iter().map(|p| p as *const _ as *mut c_void).collect();
        params.extend(dims.iter().map(|n| n as *const _ as *mut c_void));
        self.launch_maybe_blob("mapped_symbol", g, bl, 0, &mut params, || {
            let mut b = KernargBlob::new();
            for p in pointers { b.push_ptr(p); }
            for n in dims { b.push_i32(n); }
            b
        })
    }
}"#.to_string()
}

#[test]
fn canonical_params_map_and_extend_preserve_origins_and_fail_closed() {
    let source = canonical_params_map_source();
    let symbols = [(
        "mapped_symbol",
        Decl::from_scalars(&[(8, 'p'), (8, 'p'), (4, 'i'), (4, 'i')]),
    )];
    let (abi, coverage) = run_synth(&source, &symbols, &no_policy());
    assert!(
        abi.is_empty() && coverage.is_empty(),
        "canonical map/append: {abi:?} {coverage:?}"
    );
    // Existing explicit literal-index blob arguments name the same array cells.
    let indexed_blob = source
        .replace(
            "for p in pointers { b.push_ptr(p); }",
            "b.push_ptr(pointers[0]); b.push_ptr(pointers[1]);",
        )
        .replace(
            "for n in dims { b.push_i32(n); }",
            "b.push_i32(dims[0]); b.push_i32(dims[1]);",
        );
    let (abi, coverage) = run_synth(&indexed_blob, &symbols, &no_policy());
    assert!(
        abi.is_empty() && coverage.is_empty(),
        "indexed map/append: {abi:?} {coverage:?}"
    );
    // Adapters can reorder/drop elements; unknown closures/sources can change
    // origins; a mutable source could change the values referenced by params.
    for bad in [
        source.replace("pointers.iter().map", "pointers.iter().rev().map"),
        source.replace("pointers.iter().map", "pointers.iter().skip(1).map"),
        source.replace("dims.iter().map", "dims.iter().rev().map"),
        source.replace("dims.iter().map", "dims.iter().filter(|_| true).map"),
        source.replace("|p| p as *const _", "|p| other as *const _"),
        source.replace(
            "|p| p as *const _ as *mut c_void",
            "|p| { p as *const _ as *mut c_void }",
        ),
        source.replace("let pointers =", "let mut pointers ="),
        source.replace("let dims =", "let mut dims ="),
        source.replace(
            "        self.launch_maybe_blob",
            "        let dims = [9i32, 11i32];\n        self.launch_maybe_blob",
        ),
        source.replace(
            "let pointers = [x.buf.as_ptr(), y.buf.as_ptr()];",
            "let pointers = pointer_helper();",
        ),
        source.replace("params.extend(dims", "params.extend(pointers"),
        source.replace(
            "        self.launch_maybe_blob",
            "        params.reverse();\n        self.launch_maybe_blob",
        ),
        source.replace(
            "        params.extend(dims.iter().map(|n| n as *const _ as *mut c_void));\n",
            "",
        ),
    ] {
        let (abi, coverage) = run_synth(&bad, &symbols, &no_policy());
        assert!(
            !abi.is_empty() || !coverage.is_empty(),
            "unproved mapping must fail: {bad}"
        );
    }
    // A closure's void-pointer cast is NOT the ABI of the element it points at.
    // The immutable array's actual elements must still have the declared type.
    for bad in [
        source.replace("[3i32, 7i32]", "[3f32, 7f32]"),
        source.replace("[x.buf.as_ptr(), y.buf.as_ptr()]", "[x.buf.as_ptr(), 7u64]"),
    ] {
        let (abi, coverage) = run_synth(&bad, &symbols, &no_policy());
        assert!(
            !abi.is_empty() || !coverage.is_empty(),
            "mapped wrong element type must fail: {bad}"
        );
    }
}

#[test]
fn canonical_params_map_checks_actual_few_row_launchers_against_hip() {
    // Only the two actual Rust files needed by these launches, not the whole
    // closed-coverage fixture. The shared gate below still scans every file.
    let root = root();
    let paths = vec![
        "crates/rdna-compute/src/moe_few.rs".to_string(),
        "crates/rdna-compute/src/gemm.rs".to_string(),
    ];
    let world = World::load(&root, &paths).expect("actual few-row sources parse");
    let inverse_source = std::fs::read_to_string(
        root.join("kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4.gfx1151.hip"),
    )
    .unwrap();
    let down_source = std::fs::read_to_string(
        root.join("kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip"),
    )
    .unwrap();
    let inverse = hip::decls_of(&inverse_source).expect("actual inverse HIP declarations");
    let down = hip::decls_of(&format!(
        "#define QT53_DOWN_ENTRY gemv_mq4g128v2_moe_down_top10_k640\n#define QT53_DOWN_K640 1\n{down_source}"
    )).expect("actual special down HIP declaration");
    let mut symbols = Symbols::default();
    for (name, declaration) in [
        (
            "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse_k2560",
            inverse["gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse_k2560"].clone(),
        ),
        (
            "gemv_mq4g128v2_moe_down_top10_k640",
            down["gemv_mq4g128v2_moe_down_top10_k640"].clone(),
        ),
    ] {
        eprintln!(
            "actual few-row ABI: {name} args={} end={} declaration={declaration:?}",
            declaration.args.len(),
            declaration.end
        );
        symbols.hip.insert(
            name.to_string(),
            vec![("actual HIP".to_string(), declaration)],
        );
    }
    let analysis = analyze(&world, &symbols, &no_policy());
    let (abi, coverage) = violations(&analysis, &no_policy());
    assert!(
        abi.is_empty() && coverage.is_empty(),
        "actual few-row ABI: {abi:?} {coverage:?}"
    );
    assert!(
        analysis.checked_sites >= 2,
        "both real launch sites must be checked"
    );
    for name in symbols.all_names() {
        assert!(
            analysis.covered.contains(&name),
            "actual live launch not checked: {name}"
        );
    }
    eprintln!(
        "actual few-row checked sites={} covered={:?}",
        analysis.checked_sites, analysis.covered
    );
}
