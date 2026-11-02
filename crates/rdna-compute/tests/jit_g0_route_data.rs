// SPDX-License-Identifier: Apache-2.0
// G0 data gate only: no compiler process, HIP initialization, or route runtime.
use rdna_compute::{kernel_registry, Gpu, KernelCompiler};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jit-g0/");
const MANIFESTS: &[&str] = &[
    "27b-xts-gfx1201.routes.json",
    "27b-xts-gfx1100.routes.json",
    "27b-xts-gfx1151.routes.json",
    "flash-next-gfx1201.routes.json",
    "flash-next-gfx1100.routes.json",
    "flash-next-gfx1151.routes.json",
];
fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(ROOT).join(name)).unwrap()).unwrap()
}
fn strings(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect()
}
fn image_identity(entry: &Value) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        entry["kind"],
        entry["sha256"],
        entry["flags"],
        entry["scheduler_profile"],
        entry["cache_abi"]
    )
}
fn validate_public_entries(entries: &[Value]) -> Result<(), String> {
    let mut selected = BTreeMap::new();
    for entry in entries {
        let identity = image_identity(entry);
        for symbol in strings(&entry["symbols"]) {
            if let Some(previous) = selected.insert(symbol, identity.clone()) {
                if previous != identity {
                    return Err(format!("conflicting image for {symbol}"));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn six_header_specific_recipes_match_baseline_factories() {
    for name in MANIFESTS {
        let manifest = fixture(name);
        let arch = manifest["arch"].as_str().unwrap();
        assert_eq!(manifest["baseline"], "83cb363e8a");
        assert_eq!(
            manifest["mixed_fixture_role"],
            "superset only; not default route membership"
        );
        let corpus = kernel_registry::corpus_entries(arch, "-DIU4_A4_CANDIDATES=2").unwrap();
        let sets = std::iter::once(&manifest["default_entries"]).chain(
            manifest["alternate_arms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arm| {
                    if let Some(entries) = arm.get("entries") {
                        entries
                    } else {
                        let reference = arm["entries_ref"].as_str().unwrap_or_else(|| {
                            panic!(
                                "{name}: arm {} has neither entries nor entries_ref",
                                arm["name"]
                            )
                        });
                        manifest.get(reference).unwrap_or_else(|| {
                            panic!(
                                "{name}: arm {} references missing field {reference}",
                                arm["name"]
                            )
                        })
                    }
                }),
        );
        for entries in sets {
            let entries = entries.as_array().unwrap();
            validate_public_entries(entries).unwrap();
            for entry in entries {
                assert_eq!(entry["arch"], arch);
                assert!(!strings(&entry["symbols"]).is_empty());
                if entry["kind"] == "hip" {
                    let module = entry["module"].as_str().unwrap();
                    let oracle = corpus
                        .iter()
                        .find(|e| e.module == module)
                        .unwrap_or_else(|| panic!("{name}: missing source factory for {module}"));
                    // `entry["sha256"]` is the historical capture's source-text hash:
                    // incidental wording drift must not fail this gate. The live
                    // binding is the recipe below, derived from the current
                    // registry source via `recipe_for_source`.
                    let recipe = KernelCompiler::recipe_for_source(
                        arch,
                        module,
                        oracle.source(),
                        "-DIU4_A4_CANDIDATES=2",
                    );
                    assert_eq!(serde_json::to_value(&recipe.flags).unwrap(), entry["flags"]);
                    assert_eq!(
                        recipe.scheduler_profile.as_deref(),
                        entry["scheduler_profile"].as_str()
                    );
                    assert_eq!(
                        recipe.cache_abi as u64,
                        entry["cache_abi"].as_u64().unwrap()
                    );
                    let required = strings(&entry["symbols"]);
                    let missing: BTreeSet<_> = required
                        .iter()
                        .copied()
                        .filter(|s| !oracle.symbols.contains(s))
                        .collect();
                    // `registry_missing_symbols` is the immutable G0 measurement of the
                    // legacy registry (the Q8 gap of HOLD1-R2-REVIEW section 3). It stays
                    // as data; the live registry must now close every measured symbol.
                    assert!(missing.is_empty(),
                        "{name}:{module}: required symbols missing from the live registry: {missing:?}");
                    let historical: BTreeSet<_> = strings(&entry["registry_missing_symbols"])
                        .into_iter()
                        .collect();
                    assert!(
                        historical.iter().all(|s| required.contains(s)),
                        "{name}:{module}: historical gap names a symbol the route never required"
                    );
                    assert_eq!(
                        entry["export_receipt"]["sha256"].as_str().unwrap().len(),
                        64
                    );
                    for symbol in required {
                        assert!(
                            strings(&entry["exports"]).contains(&symbol),
                            "{name}:{module}:{symbol}"
                        );
                    }
                } else {
                    assert_eq!(entry["kind"], "embedded");
                    let image = Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../..")
                        .join(entry["repo_image"].as_str().unwrap());
                    let bytes = std::fs::read(image).unwrap();
                    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), entry["sha256"]);
                    assert!(strings(&entry["flags"]).is_empty());
                    for symbol in strings(&entry["symbols"]) {
                        assert!(
                            strings(&entry["image_exports"]).contains(&symbol),
                            "missing native export {symbol}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn all_current_cold_sources_have_selected_symbols() {
    let manifest = fixture("27b-xts-gfx1201.routes.json");
    let cold = fixture("cold-24.json");
    assert_eq!(cold.as_array().unwrap().len(), 24);
    for source in cold.as_array().unwrap() {
        let entry = manifest["default_entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["module"] == source["module"] && e["sha256"] == source["sha256"])
            .unwrap();
        assert!(!strings(&entry["symbols"]).is_empty());
    }
}

#[test]
fn headers_are_quant_and_shape_specific_not_mixed_trace_defaults() {
    for name in MANIFESTS {
        let m = fixture(name);
        let family = m["family"].as_str().unwrap();
        let h = fixture(if family == "27b-xts" {
            "header-27b.json"
        } else {
            "header-flash-next.json"
        });
        assert_eq!(m["header"]["header_sha256"], h["header_sha256"]);
        assert_eq!(m["header"]["quant_census"], h["quant_census"]);
        assert!(h["tensors"].as_array().unwrap().iter().all(|t| !t["name"]
            .as_str()
            .unwrap()
            .is_empty()
            && t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .all(|d| d.as_u64().unwrap() > 0)));
        for e in m["default_entries"].as_array().unwrap() {
            let module = e["module"].as_str().unwrap();
            assert!(!module.starts_with("dflash_") || module == "dflash_state_bulk_copy_gfx1100");
            assert!(!module.contains("ddtree") && !module.contains("dspark"));
            if family == "27b-xts" {
                assert!(!module.contains("moe"));
            }
        }
        if family == "flash-next" && m["arch"] == "gfx1100" {
            assert_eq!(m["evidence_status"], "source-predicted");
            assert_eq!(m["effective_policy"]["spec"], "ar");
            assert!(m["effective_policy"]["host_mapped_experts"]
                .as_bool()
                .unwrap());
        }
    }
}

#[test]
fn same_source_aliases_union_symbols_without_erasing_module_names() {
    let manifest = fixture("27b-xts-gfx1201.routes.json");
    let entries = manifest["default_entries"].as_array().unwrap();
    let mut groups: BTreeMap<String, (BTreeSet<&str>, BTreeSet<&str>)> = BTreeMap::new();
    for e in entries.iter().filter(|e| e["kind"] == "hip") {
        let (modules, symbols) = groups.entry(image_identity(e)).or_default();
        modules.insert(e["module"].as_str().unwrap());
        symbols.extend(strings(&e["symbols"]));
    }
    assert_eq!(groups.len(), 43);
    assert!(groups
        .values()
        .any(|(m, s)| m.contains("qwen35_fa_prep_batched_gfx1201")
            && m.contains("qwen35_fa_prep_fp8q_nogate_batched_gfx1201")
            && s.contains("qwen35_fa_prep_batched_gfx1201")
            && s.contains("qwen35_fa_prep_fp8q_nogate_batched_gfx1201")));
}

#[test]
fn conflicting_public_entry_images_refuse_even_if_modules_differ() {
    let m = fixture("27b-xts-gfx1201.routes.json");
    let e = m["default_entries"].as_array().unwrap()[0].clone();
    let mut conflict = e.clone();
    conflict["module"] = Value::from("alternate");
    conflict["sha256"] = Value::from("different-image");
    assert!(validate_public_entries(&[e.clone(), conflict]).is_err());
    let mut alias = e.clone();
    alias["module"] = Value::from("same-recipe-alias");
    assert!(validate_public_entries(&[e, alias]).is_ok());
}

#[test]
fn architecture_defaults_and_current_shape_ports_are_not_blanket_aliases() {
    assert_eq!(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS, 512);
    assert_eq!(rdna_compute::gemm::QWEN4_MQ6_X4_PM_MIN_M_GFX1151, 2560);
    for (arch, level) in [("gfx1151", 3), ("gfx1201", 1), ("gfx1100", 0)] {
        let m = fixture(&format!("flash-next-{arch}.routes.json"));
        assert_eq!(m["effective_policy"]["hc_fuse"], level);
    }
    assert_eq!(rdna_compute::gemm::QWEN4_MQ6_X4_PM_MIN_N_GFX1151, 2048);
    assert_eq!(
        Gpu::qwen4_lds_tile_gfx1201(320, 10240).entry(),
        "qwen4_wmma_lds_64_128_32_64_k64_p_s4"
    );
    assert_eq!(
        Gpu::qwen4_lds_tile_gfx1201(640, 2560).entry(),
        "qwen4_wmma_lds_128_128_32_64_k64"
    );
    assert_eq!(
        Gpu::qwen4_lds_tile_gfx1201(2560, 640).entry(),
        "qwen4_wmma_lds_128_256_32_64_k64"
    );
}

#[test]
fn g0_pass_is_not_an_a3_ready_plan_or_an_inadmissible_fixture_guarantee() {
    for name in MANIFESTS {
        let m = fixture(name);
        assert_eq!(m["g0_gate"], "PASS");
        assert_eq!(m["is_closed_ready_plan"], false);
        assert!(!strings(&m["a3_coverage_not_proven"]).is_empty());
        assert!(m["alternate_arms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|arm| arm["name"] == "serial-compiler-kill-arm"
                && arm["entries_ref"] == "default_entries"));
        assert!(m["explicit_spec_inventory_role"]
            .as_str()
            .unwrap()
            .contains("superset"));
        if *name == "flash-next-gfx1100.routes.json" {
            assert!(m["g0_fixture_exclusion"]
                .as_str()
                .unwrap()
                .contains("not admissible"));
        } else {
            assert!(m["g0_fixture_exclusion"].is_null());
        }
    }
    // The G0 fixtures once recorded four live `tensor_ops` exports the registry
    // did not declare (a pack index without them rejected the packaged object);
    // the registry now declares them, so no fixture entry may record a gap.
    let m = fixture("flash-next-gfx1201.routes.json");
    let tensor_ops = m["default_entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["module"] == "tensor_ops")
        .unwrap();
    assert!(strings(&tensor_ops["registry_missing_symbols"]).is_empty());
}

/// Every Q8 symbol a measured flash-next route (state, prefill, decode and
/// KV paths) requests of `tensor_ops` is in the packaged index of its arch,
/// not only in the JIT corpus: the sealed gfx1100 package omitted nine.
#[test]
fn measured_q8_route_symbols_are_in_the_packaged_index() {
    let mut covered = 0usize;
    for arch in ["gfx1100", "gfx1151", "gfx1201"] {
        let m = fixture(&format!("flash-next-{arch}.routes.json"));
        let packaged = kernel_registry::entries(arch, "-DIU4_A4_CANDIDATES=2").unwrap();
        let tensor_ops = packaged.iter().find(|e| e.module == "tensor_ops").unwrap();
        let arms = std::iter::once(&m["default_entries"]).chain(
            m["alternate_arms"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|arm| arm.get("entries")),
        );
        let mut q8 = BTreeSet::new();
        for entries in arms {
            for entry in entries
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["module"] == "tensor_ops")
            {
                q8.extend(
                    strings(&entry["symbols"])
                        .into_iter()
                        .filter(|s| s.contains("q8")),
                );
            }
        }
        // Each arch's fixture records its own Q8 requests. gfx1151's trace is a pre-failure
        // observation (its native prefill never ran), so it recorded only the two GDN state
        // codecs; gfx1100/gfx1201 recorded the Q8 KV paths too. Nothing is asserted that an
        // arch's trace did not record: its remaining Q8 capability is the source-export
        // closure in the `kernel_registry` tests.
        assert!(
            !q8.is_empty(),
            "{arch}: fixture records no Q8 tensor_ops route symbols"
        );
        if arch != "gfx1151" {
            for kv in [
                "indexed_attention_decode_prologue_q8",
                "indexed_attention_cache_append_q8_batched",
                "indexed_attention_attention_q8_batched",
            ] {
                assert!(q8.contains(kv), "{arch}: fixture lost recorded {kv}");
            }
        }
        covered += 1;
        for symbol in q8 {
            assert!(
                tensor_ops.symbols.contains(&symbol),
                "{arch}: packaged index omits {symbol}"
            );
        }
    }
    assert_eq!(covered, 3);
}
