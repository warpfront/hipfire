// SPDX-License-Identifier: Apache-2.0
//! Flagship kernel-pack regression gate (CPU only, no compiler, no GPU).
//!
//! `fixtures/flagship-pack-gfx1201.tsv` is the measured module set of the six
//! flagship gfx1201 routes (Qwen3.8 27B AR / default DFlash / MTP, Qwen3.8
//! flash-next AR / MTP, Qwen3.6 35B-A3B mq4rs AR), copied from the
//! `HIPFIRE_MODULE_LOAD_AUDIT` records (`audit_hip` / `audit_embedded` in each
//! case.json) of the published v0.4.4 binaries on a clean HOME / kernel cache.
//! A module the runtime loaded and the pack lacked fell back to hipcc, and a
//! pack-only host failed. Every measured hip row must therefore be in
//! `kernel_registry::entries` (the set `hipfire-kernel-pack` compiles) with the
//! measured symbol, hipcc flags, scheduler profile and source SHA-256.
//!
//! `fixtures/flagship-pack-shared.tsv` is the portable subset (shipped for
//! gfx1100 and gfx1151 as well) that must be registered on every release arch.
//! gfx1201-only modules are asserted for gfx1201 only.

use rdna_compute::kernel_registry::{entries, walked_route_plans, SUPPORTED_ARCHES};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const FLAGSHIP: &str = include_str!("fixtures/flagship-pack-gfx1201.tsv");
const FLAGSHIP_1151: &str = include_str!("fixtures/flagship-pack-gfx1151.tsv");
const FLAGSHIP_1100: &str = include_str!("fixtures/flagship-pack-gfx1100.tsv");
const SHARED: &str = include_str!("fixtures/flagship-pack-shared.tsv");
/// Extra hipcc flags the registry CLI prints for every measured release arch.
const EXTRA_FLAGS: &str = "-DIU4_A4_CANDIDATES=2";
const RELEASE_ARCHES: [&str; 3] = ["gfx1201", "gfx1100", "gfx1151"];
const ROUTES: [&str; 6] = [
    "27b-ar",
    "27b-dflash-default",
    "27b-mtp",
    "fn-ar",
    "fn-mtp",
    "a3b-ar",
];
/// Modules the v0.4.4 pack lacked (hipcc-compiled at first request).
const FORMERLY_MISSING: [&str; 8] = [
    "attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm",
    "attention_flash_reduce_dsplit_gfx1201_vmm",
    "attention_fp8_e4m3_kv_batched_segs",
    "conv1d_silu_split_qknorm_b256_rows",
    "gated_delta_net_q8_compact3_b2_rows",
    "gated_norm_mq_rotate_awq_k6144_gfx1201_rows",
    "kv_cache_write_fp8_e4m3_batched_vmm",
    "qwen36_27b_fa_prep_gfx1201_rows",
];

#[derive(Debug)]
struct Row {
    route: String,
    arch: String,
    kind: String,
    module: String,
    symbol: String,
    flags: String,
    scheduler: String,
    sha256: String,
}

fn rows() -> Vec<Row> {
    rows_of(FLAGSHIP)
}

fn rows_of(text: &str) -> Vec<Row> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            assert_eq!(c.len(), 8, "bad fixture row: {l}");
            assert_eq!(c[7].len(), 64, "bad sha256: {l}");
            Row {
                route: c[0].into(),
                arch: c[1].into(),
                kind: c[2].into(),
                module: c[3].into(),
                symbol: c[4].into(),
                flags: c[5].into(),
                scheduler: c[6].into(),
                sha256: c[7].into(),
            }
        })
        .collect()
}

/// `#route` header counts: route -> (hip, embedded).
fn header_counts() -> BTreeMap<String, (usize, usize)> {
    header_counts_of(FLAGSHIP)
}

fn header_counts_of(text: &str) -> BTreeMap<String, (usize, usize)> {
    let field = |c: &[&str], key: &str| -> usize {
        c.iter()
            .find_map(|f| f.strip_prefix(key))
            .unwrap_or_else(|| panic!("route header missing {key}"))
            .parse()
            .unwrap()
    };
    text.lines()
        .filter_map(|l| l.strip_prefix("#route "))
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c[0].to_owned(), (field(&c, "hip="), field(&c, "embedded=")))
        })
        .collect()
}

fn sha(source: &str) -> String {
    Sha256::digest(source.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn fixture_holds_complete_measured_sets_for_all_six_routes() {
    let rows = rows();
    let counts = header_counts();
    assert_eq!(counts.keys().map(String::as_str).collect::<Vec<_>>(), {
        let mut r = ROUTES.to_vec();
        r.sort_unstable();
        r
    });
    for route in ROUTES {
        let (hip, emb) = counts[route];
        let mine: Vec<&Row> = rows.iter().filter(|r| r.route == route).collect();
        assert_eq!(
            mine.iter().filter(|r| r.kind == "hip").count(),
            hip,
            "{route}: hip row count differs from the measured audit"
        );
        assert_eq!(
            mine.iter().filter(|r| r.kind == "embedded").count(),
            emb,
            "{route}: embedded row count differs from the measured audit"
        );
        let measured = match route {
            "27b-ar" => (64, 9),
            "27b-dflash-default" => (60, 7),
            "27b-mtp" => (93, 10),
            "fn-ar" => (68, 4),
            "fn-mtp" => (69, 4),
            "a3b-ar" => (53, 10),
            _ => unreachable!(),
        };
        assert_eq!((hip, emb), measured, "{route}: measured set changed");
        let unique: BTreeSet<(&str, &str)> = mine
            .iter()
            .map(|r| (r.kind.as_str(), r.module.as_str()))
            .collect();
        assert_eq!(unique.len(), mine.len(), "{route}: duplicate module rows");
        assert!(mine.iter().all(|r| r.arch == "gfx1201"));
    }
}

fn assert_measured_hip_modules_registered(arch: &str, fixture: &str) {
    let registry = entries(arch, EXTRA_FLAGS).unwrap();
    let mut failures = Vec::new();
    for r in rows_of(fixture).iter().filter(|r| r.kind == "hip") {
        let Some(e) = registry.iter().find(|e| e.module == r.module) else {
            failures.push(format!(
                "{arch}: {} [{}]: not registered",
                r.module, r.route
            ));
            continue;
        };
        if !e.symbols.contains(&r.symbol.as_str()) {
            failures.push(format!(
                "{} [{}]: symbol {} not in {:?}",
                r.module, r.route, r.symbol, e.symbols
            ));
        }
        if e.flags.join(" ") != r.flags {
            failures.push(format!(
                "{} [{}]: flags differ: registry `{}` measured `{}`",
                r.module,
                r.route,
                e.flags.join(" "),
                r.flags
            ));
        }
        if e.scheduler_profile.as_deref().unwrap_or("") != r.scheduler {
            failures.push(format!(
                "{} [{}]: scheduler profile {:?} != measured {:?}",
                r.module, r.route, e.scheduler_profile, r.scheduler
            ));
        }
        if sha(e.source()) != r.sha256 {
            failures.push(format!(
                "{} [{}]: source sha256 {} != measured {}",
                r.module,
                r.route,
                sha(e.source()),
                r.sha256
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pack coverage regressions:\n{}",
        failures.join("\n")
    );
}

/// hipcc modules the A3B mq4rs AR decode loads that were added after the v0.4.4
/// measurement. A release binary on a host without hipcc binds only packed
/// kernels, so each must be registered for gfx1201 (the set
/// `hipfire-kernel-pack` compiles) AND listed in the `a3b-ar` route of the
/// fixture with its exact source SHA-256 (the registry test above then pins
/// symbol, flags and source). A module missing from either fails here.
const A3B_AR_POST_MEASUREMENT_HIP_MODULES: [&str; 16] = [
    "fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201",
    // Lane-batched FullAttention of the exact CbVmm route: the batched VMM fp8
    // KV write, the group-8 flash tile twin, the fused reduce + gate + rotate
    // epilogue twin and the 16Q/2K FA prep rows twin.
    "kv_cache_write_fp8_e4m3_batched_vmm",
    "attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm",
    "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201_vmm",
    "qwen35_fa_prep_gfx1201_rows",
    // Lane-batched lm_head of the exact CbVmm route (`head_lanes.rs`).
    "gemv_mq4g256v2_head_lanes_gfx1201",
    // Lane-batched multi-weight projection GEMV of the exact VMM batch route
    // (DeltaNet QKVZA, FullAttention QKV, MoE router/selector/shared quartet).
    "gemv_mq4g256v2_lane_proj_k2048_gfx1201",
    // Expert-deduplicating routed gate/up and down + combine of the exact
    // CbVmm route (`moe_decode_lanes.rs`).
    "moe_dedup_mq4rs_gfx1201",
    // Lane-batched twins the exact CbVmm route launches for the A3B DeltaNet
    // mixer, the MoE block (also the fallback of the dedup kernels and the
    // one-lane path) and the `wo` residual.
    "conv1d_silu_split_qknorm_b256_rows",
    "gated_delta_net_q8_compact2_b2_rows",
    "gated_norm_mq_rotate_gfx1201_rows",
    "gemv_mq4g128v2_moe_ninepath_d4_rows",
    "gemv_mq4g256v2_moe_gate_up_k8_indexed_rows",
    "gemv_mq4g256v2_residual_lanes_gfx1201",
    "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows",
    "moe_router_softmax_topk_k8_wave64_rows",
];

#[test]
fn a3b_ar_post_measurement_decode_modules_ship_in_the_gfx1201_pack() {
    let registry = entries("gfx1201", EXTRA_FLAGS).unwrap();
    let rows = rows();
    for module in A3B_AR_POST_MEASUREMENT_HIP_MODULES {
        assert!(
            registry.iter().any(|e| e.module == module),
            "{module}: not registered for gfx1201 (a pack-only host cannot bind it)"
        );
        assert!(
            rows.iter()
                .any(|r| r.route == "a3b-ar" && r.kind == "hip" && r.module == module),
            "{module}: missing from the a3b-ar route of flagship-pack-gfx1201.tsv"
        );
    }
}

/// The gfx1201 Z/beta/alpha fold split the dense 27B verify batch
/// (`gemm_zba_fold_g12_iu4_prepared`) launches after the v0.4.4 measurement:
/// it must be registered for gfx1201 and listed in both verify routes.
#[test]
fn zba_fold_split_module_ships_in_the_gfx1201_pack() {
    let module = "split_mq4v2_z_betaalpha";
    let registry = entries("gfx1201", EXTRA_FLAGS).unwrap();
    assert!(
        registry
            .iter()
            .any(|e| e.module == module && e.symbols.contains(&"split_mq4v2_z_betaalpha_g12")),
        "{module}: g12 split not registered for gfx1201 (a pack-only host cannot bind it)"
    );
    let rows = rows();
    for route in ["27b-dflash-default", "27b-mtp"] {
        assert!(
            rows.iter()
                .any(|r| r.route == route && r.kind == "hip" && r.module == module),
            "{module}: missing from the {route} route of flagship-pack-gfx1201.tsv"
        );
    }
}

#[test]
fn every_measured_flagship_hip_module_is_registered_exactly_on_gfx1201() {
    assert_measured_hip_modules_registered("gfx1201", FLAGSHIP);
}

#[test]
fn every_measured_flagship_hip_module_is_registered_exactly_on_gfx1151() {
    assert_measured_hip_modules_registered("gfx1151", FLAGSHIP_1151);
}

#[test]
fn every_measured_flagship_hip_module_is_registered_exactly_on_gfx1100() {
    assert_measured_hip_modules_registered("gfx1100", FLAGSHIP_1100);
}

#[test]
fn halo_and_rdna3_fixtures_pin_measured_route_counts() {
    let cases: [(&str, &str, &[(&str, usize)]); 2] = [
        (
            "gfx1151",
            FLAGSHIP_1151,
            &[
                ("27b-ar", 41),
                ("27b-dflash-default", 59),
                ("27b-mtp", 58),
                ("a3b-ar-iu4off", 55),
            ],
        ),
        (
            "gfx1100",
            FLAGSHIP_1100,
            &[
                ("27b-ar", 34),
                ("27b-dflash-default", 55),
                ("27b-mtp", 65),
                ("fn-ar", 30),
                ("fn-mtp", 36),
                ("a3b-ar-iu4off", 53),
            ],
        ),
    ];
    for (arch, fixture, expected) in cases {
        let rows = rows_of(fixture);
        let counts = header_counts_of(fixture);
        assert_eq!(counts.len(), expected.len(), "{arch}: route set");
        for (route, hip) in expected {
            assert_eq!(counts[*route], (*hip, 0), "{arch} {route}: header counts");
            let mine: Vec<&Row> = rows.iter().filter(|r| r.route == *route).collect();
            assert_eq!(mine.len(), *hip, "{arch} {route}: row count");
            assert!(mine.iter().all(|r| r.arch == arch && r.kind == "hip"));
            let unique: BTreeSet<&str> = mine.iter().map(|r| r.module.as_str()).collect();
            assert_eq!(
                unique.len(),
                mine.len(),
                "{arch} {route}: duplicate modules"
            );
        }
    }
}

#[test]
fn halo_select_regrid_and_topk_fixup_are_registered_on_gfx1151() {
    let registry = entries("gfx1151", EXTRA_FLAGS).unwrap();
    for m in ["select_regrid", "topk_values_fixup_f32"] {
        assert!(
            rows_of(FLAGSHIP_1151)
                .iter()
                .any(|r| r.module == m && r.kind == "hip"),
            "{m} missing from the measured gfx1151 fixture"
        );
        assert!(
            registry.iter().any(|e| e.module == m),
            "{m} not registered for gfx1151"
        );
    }
}

#[test]
fn formerly_missing_modules_are_measured_and_registered() {
    let rows = rows();
    let measured: BTreeSet<&str> = rows
        .iter()
        .filter(|r| r.kind == "hip")
        .map(|r| r.module.as_str())
        .collect();
    let registry = entries("gfx1201", EXTRA_FLAGS).unwrap();
    for m in FORMERLY_MISSING {
        assert!(
            measured.contains(m),
            "{m} missing from the measured fixture"
        );
        let e = registry.iter().find(|e| e.module == m);
        let e = e.unwrap_or_else(|| panic!("{m} not registered for gfx1201"));
        let expected: &[&str] = match m {
            "attention_fp8_e4m3_kv_batched_segs" => &[
                "attention_fp8_e4m3_kv_batched_segs",
                "attention_q8_0_kv_batched",
                "attention_q8_0_kv_independent_masked_windowed",
            ],
            "gated_delta_net_q8_compact3_b2_rows" => &[
                "gated_delta_net_q8_compact3_b2_rows",
                "gated_delta_net_q8_fast_independent_masked",
            ],
            _ => std::slice::from_ref(&m),
        };
        let mut actual: Vec<&str> = e.symbols.iter().copied().collect();
        actual.sort_unstable();
        let mut want: Vec<&str> = expected.to_vec();
        want.sort_unstable();
        assert_eq!(actual, want, "{m}: exported symbol set");
    }
}

#[test]
fn embedded_native_images_are_recorded_but_not_registry_sources() {
    let rows = rows();
    let registry = entries("gfx1201", EXTRA_FLAGS).unwrap();
    let embedded: Vec<&Row> = rows.iter().filter(|r| r.kind == "embedded").collect();
    assert!(!embedded.is_empty());
    for r in embedded {
        assert!(r.flags.is_empty() && r.scheduler.is_empty(), "{}", r.module);
        // Native images carry an image hash, not a hipcc source: a hip registry
        // entry of the same name would have to match no hip row's identity.
        if let Some(e) = registry.iter().find(|e| e.module == r.module) {
            assert_ne!(sha(e.source()), r.sha256, "{}", r.module);
        }
    }
}

#[test]
fn portable_flagship_subset_is_registered_on_every_release_arch() {
    let measured: BTreeSet<(String, String)> = rows()
        .into_iter()
        .filter(|r| r.kind == "hip")
        .map(|r| (r.module, r.symbol))
        .collect();
    let mut shared = Vec::new();
    for l in SHARED
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let c: Vec<&str> = l.split('\t').collect();
        assert_eq!(c.len(), 3, "bad shared row: {l}");
        assert!(
            measured.contains(&(c[0].to_owned(), c[1].to_owned())),
            "{}: shared row is not a measured gfx1201 module",
            c[0]
        );
        assert!(
            !c[0].contains("gfx12"),
            "{}: arch-specific module in shared set",
            c[0]
        );
        shared.push((c[0], c[1]));
    }
    assert!(shared.len() >= 80);
    let mut failures = Vec::new();
    for arch in RELEASE_ARCHES {
        assert!(SUPPORTED_ARCHES.contains(&arch));
        let registry = entries(arch, EXTRA_FLAGS).unwrap();
        for (module, symbol) in &shared {
            match registry.iter().find(|e| e.module == *module) {
                None => failures.push(format!("{arch}: {module} not registered")),
                Some(e) => {
                    if !e.symbols.contains(symbol) {
                        failures.push(format!("{arch}: {module} lacks symbol {symbol}"));
                    }
                    if !e
                        .flags
                        .iter()
                        .any(|f| f == &format!("--offload-arch={arch}"))
                    {
                        failures.push(format!("{arch}: {module} flags target another arch"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "portable coverage regressions:\n{}",
        failures.join("\n")
    );
}

/// The exact BF16 split QSA selector is the exact-gfx1100 Flash-Next default
/// (`HIPFIRE_QWEN4_QSA_SPLIT_GFX1100`, on unless `0`), so the gfx1100 AR and
/// native-MTP route module sets must carry its two modules. The v0.4.4
/// `flagship-pack-gfx1100.tsv` predates the default and is never regenerated;
/// the required set is therefore its measured fn-ar / fn-mtp modules plus the
/// walked gfx1100 plan's hip modules (which include the split modules) plus
/// these two, each of which must be a registry entry (what `hipfire-kernel-pack`
/// compiles) with the symbols the split launchers bind, the gfx1100 recipe, and
/// the exact kernel source the gfx1201 measured rows pin. A module missing from
/// the pack, from either walked route, or with a different source fails here.
#[test]
fn gfx1100_flash_next_pack_carries_the_bf16_split_qsa_modules() {
    const SPLIT: [(&str, &[&str]); 2] = [
        (
            "qsa_bf16_scores",
            &[
                "indexed_attention_select_scores_rows16_bf16",
                "indexed_attention_select_scores_rows8_bf16",
            ],
        ),
        (
            "qsa_filter_select",
            &[
                "qsa_filter_compact",
                "qsa_filter_finish",
                "qsa_filter_threshold",
            ],
        ),
    ];
    let pack = entries("gfx1100", EXTRA_FLAGS).unwrap();
    let plan = walked_route_plans()
        .unwrap()
        .iter()
        .find(|plan| plan.model_arch == 16 && plan.arch == "gfx1100")
        .expect("walked gfx1100 Flash-Next plan");
    let measured_1201 = rows();
    let measured_1100 = rows_of(FLAGSHIP_1100);
    let mut failures = Vec::new();
    for (route, native_mtp, fixture_route) in
        [("ar", false, "fn-ar"), ("native_mtp", true, "fn-mtp")]
    {
        let mut required: BTreeSet<&str> = measured_1100
            .iter()
            .filter(|r| r.route == fixture_route && r.kind == "hip")
            .map(|r| r.module.as_str())
            .collect();
        let walked: BTreeSet<&str> = plan
            .rows
            .iter()
            .filter(|row| row.native_mtp == native_mtp && !row.embedded)
            .map(|row| row.module)
            .collect();
        required.extend(&walked);
        required.extend(SPLIT.iter().map(|(module, _)| *module));
        for module in &required {
            if !pack.iter().any(|e| e.module == *module) {
                failures.push(format!("gfx1100 {route}: {module} not in the pack"));
            }
        }
        for (module, symbols) in SPLIT {
            if !walked.contains(module) {
                failures.push(format!("gfx1100 {route}: {module} not in the walked plan"));
            }
            let Some(e) = pack.iter().find(|e| e.module == module) else {
                continue;
            };
            for symbol in symbols {
                if !e.symbols.contains(symbol) {
                    failures.push(format!("gfx1100 {route}: {module} lacks {symbol}"));
                }
            }
            if !e.flags.iter().any(|f| f == "--offload-arch=gfx1100") {
                failures.push(format!("gfx1100 {route}: {module} targets another arch"));
            }
            let pinned = measured_1201
                .iter()
                .find(|r| r.route == fixture_route && r.kind == "hip" && r.module == module)
                .unwrap_or_else(|| panic!("{module}: no measured gfx1201 {fixture_route} row"));
            if sha(e.source()) != pinned.sha256 {
                failures.push(format!(
                    "gfx1100 {route}: {module} source sha256 {} != pinned {}",
                    sha(e.source()),
                    pinned.sha256
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gfx1100 Flash-Next split QSA pack regressions:\n{}",
        failures.join("\n")
    );
}
