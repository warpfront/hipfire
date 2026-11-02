// SPDX-License-Identifier: Apache-2.0
// Flash-Next walked route plan: CPU-only planner tests. No compiler process or HIP.
//
// `route_plans/qwen3.8-flash-next.gfx1201.tsv` (the canonical monolithic header)
// and `route_plans/qwen3.8-flash-next-base.mq4-xts.gfx1201.tsv` (the split header:
// the canonical header minus the 128 PLE ngram_embedding shard tensors) are
// generated from module-load
// audits of the real dispatch (`scripts/route_plan_walk.py`);
// `route_plans/qwen3.8-flash-next.gfx1100.tsv` is the canonical header walked on the XTX
// (`route_plan_walk.py --profile xtx-fn-bounded`, open: never sealed). These tests check
// that every walked kernel is planned (walk ⊆ plan), that every planned HIP
// module is the release pack's object with the walked source (plan ⊆ pack),
// and that every load the walks did not cover refuses (lazy loading) instead
// of planning partially. Older Flash-Next traces (jit-g0, kernel-trace) ran
// other policies (Q8 QSA, host-mapped experts, an earlier dispatch) and are
// not a closure oracle for this route.
use rdna_compute::kernel_registry::{
    entries, route_entries, route_layout_sha256, walked_route_plans, KernelSpecRoute,
    KernelTensorMeta, PlannedKernel, RegistryError, RouteKernelInput, RouteKernelPlan,
    WalkedRoutePlan,
};
use rdna_compute::{DType, FeatureFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(FIXTURES).join(name)).unwrap()).unwrap()
}

/// `hipfire-runtime` `loader_api::kernel_dtype_for_quant_type` for the tags a
/// Flash-Next header carries.
fn dtype(tag: u64) -> DType {
    match tag {
        3 => DType::Q8_0,
        16 => DType::BF16,
        44 => DType::MQ4G256V2,
        47 => DType::MQ6G256V2,
        53 => DType::MQ4G128V2,
        54 => DType::Raw,
        other => panic!("header fixture quant tag {other} has no test mapping"),
    }
}

/// The canonical Flash-Next GPTQ3 header (`jit-g0/header-flash-next.json`,
/// header SHA-256 8703bb16…, the artifact the gfx1201 walks loaded).
fn flash_next_header() -> Vec<(String, DType, Vec<usize>)> {
    fixture("jit-g0/header-flash-next.json")["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_owned(),
                dtype(t["quant_tag"].as_u64().unwrap()),
                t["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect()
}

fn metas(header: &[(String, DType, Vec<usize>)]) -> Vec<KernelTensorMeta<'_>> {
    header
        .iter()
        .map(|(name, dtype, shape)| KernelTensorMeta {
            name,
            dtype: *dtype,
            shape,
        })
        .collect()
}

fn builtin_flags(arch: &str) -> FeatureFlags {
    let resolved = hipfire_config::resolve(std::iter::empty()).unwrap();
    let config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
    FeatureFlags::from_process_config(arch, &config)
}

/// Every walked Flash-Next plan for an arch (one per header layout).
fn walked_plans(arch: &str) -> Vec<&'static WalkedRoutePlan> {
    walked_route_plans()
        .unwrap()
        .iter()
        .filter(|plan| plan.model_arch == 16 && plan.arch == arch)
        .collect()
}

/// The walked plan whose own header layout is `layout`.
fn walked(arch: &str, layout: &str) -> &'static WalkedRoutePlan {
    let mut plans = walked_plans(arch)
        .into_iter()
        .filter(|plan| plan.layout == layout);
    let plan = plans
        .next()
        .unwrap_or_else(|| panic!("no walked Flash-Next plan for {arch} layout {layout}"));
    assert!(
        plans.next().is_none(),
        "two walked Flash-Next plans for {arch} layout {layout}"
    );
    plan
}

/// Header layout of the split MQ4 XTS artifact (the walked split plan).
const SPLIT_LAYOUT: &str = "65811e8a696c6b68d25a0c5bf9f971cef40caf60f47a83498a41297a15597c5c";
const NGRAM_SHARDS: usize = 128;

fn ngram_shard_name(shard: usize) -> String {
    format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight")
}

/// The split header: the canonical header minus exactly the ngram shard family.
fn split_header(canonical: &[(String, DType, Vec<usize>)]) -> Vec<(String, DType, Vec<usize>)> {
    let removed: Vec<String> = (0..NGRAM_SHARDS).map(ngram_shard_name).collect();
    for name in &removed {
        assert!(
            canonical.iter().any(|(n, _, _)| n == name),
            "canonical header lacks {name}"
        );
    }
    let split: Vec<_> = canonical
        .iter()
        .filter(|(name, _, _)| !removed.contains(name))
        .cloned()
        .collect();
    assert_eq!(canonical.len() - split.len(), NGRAM_SHARDS);
    assert!(split
        .iter()
        .all(|(name, _, _)| !name.contains("ngram_embedding.shard_")));
    split
}

/// One header layout a walked Flash-Next plan covers, on one exact arch.
struct Layout {
    name: &'static str,
    arch: &'static str,
    header: Vec<(String, DType, Vec<usize>)>,
    sha: String,
}

fn layout(
    name: &'static str,
    arch: &'static str,
    header: Vec<(String, DType, Vec<usize>)>,
) -> Layout {
    let sha = route_layout_sha256(&metas(&header));
    Layout {
        name,
        arch,
        header,
        sha,
    }
}

/// The gfx1201 walked layouts: the canonical monolithic header and the split header.
fn layouts() -> Vec<Layout> {
    let canonical = flash_next_header();
    let split = split_header(&canonical);
    [("canonical", canonical), ("split", split)]
        .into_iter()
        .map(|(name, header)| layout(name, "gfx1201", header))
        .collect()
}

/// Every (arch, layout) with a walked plan: the gfx1201 layouts, and the canonical
/// header on gfx1100 (the XTX route walk, `route_plans/qwen3.8-flash-next.gfx1100.tsv`).
fn cases() -> Vec<Layout> {
    let mut out = layouts();
    out.push(layout("canonical", "gfx1100", flash_next_header()));
    out
}

/// The walked axes, each at its first walked value (what the loader resolves
/// for the default route).
fn walked_axes(plan: &WalkedRoutePlan) -> Vec<(&'static str, String)> {
    plan.axes
        .iter()
        .map(|(name, values)| (*name, values[0].to_owned()))
        .collect()
}

struct Load<'a> {
    arch: &'a str,
    tensors: &'a [KernelTensorMeta<'a>],
    max_seq: usize,
    spec: KernelSpecRoute,
    flags: &'a FeatureFlags,
    axes: &'a [(&'a str, String)],
}

fn plan(load: &Load<'_>) -> Result<RouteKernelPlan, RegistryError> {
    let axis = |name: &str| {
        load.axes
            .iter()
            .find(|(n, _)| *n == name)
            .map_or("", |(_, v)| v.as_str())
    };
    route_entries(&RouteKernelInput {
        arch: load.arch,
        model_arch: 16,
        tensors: load.tensors,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: 0,
        max_seq: load.max_seq,
        prefill_chunk_rows: 0,
        kv_k: axis("qsa"),
        kv_v: axis("qsa"),
        state_quant: axis("gdn"),
        spec: load.spec,
        host_mapped_experts: false,
        retained_decode: true,
        hipcc_extra_flags: &load.flags.hipcc_extra_flags,
        flags: load.flags,
        route_axes: load.axes,
    })
}

/// `symbol -> (module, kind)` of a plan; one owner per symbol.
fn bindings(plan: &RouteKernelPlan) -> BTreeMap<&'static str, (&'static str, bool)> {
    let mut out = BTreeMap::new();
    for entry in &plan.entries {
        for &symbol in entry.symbols() {
            let embedded = matches!(entry, PlannedKernel::Embedded { .. });
            assert!(
                out.insert(symbol, (entry.module(), embedded)).is_none(),
                "symbol {symbol} planned twice"
            );
        }
    }
    out
}

/// AR and native-MTP plans of one layout (on its own arch) at `max_seq` 16,384.
fn route_plans(layout: &Layout) -> (RouteKernelPlan, RouteKernelPlan) {
    let arch = layout.arch;
    let tensors = metas(&layout.header);
    let walked = walked(arch, &layout.sha);
    let flags = builtin_flags(arch);
    let axes = walked_axes(walked);
    let load = |spec| Load {
        arch,
        tensors: &tensors,
        max_seq: 16_384,
        spec,
        flags: &flags,
        axes: &axes,
    };
    let ar = plan(&load(KernelSpecRoute::Ar))
        .unwrap_or_else(|e| panic!("{arch} {} AR route plan: {e:?}", layout.name));
    let mtp = plan(&load(KernelSpecRoute::NativeMtp))
        .unwrap_or_else(|e| panic!("{arch} {} native-MTP route plan: {e:?}", layout.name));
    (ar, mtp)
}

/// Each header layout resolves to exactly one walked plan: the canonical header
/// and the split header (canonical minus the 128 ngram shards) have distinct
/// layouts, plus the independently walked split mq2 header.
#[test]
fn walked_layouts_are_the_canonical_and_split_flash_next_headers() {
    let layouts = layouts();
    assert_eq!(layouts[0].header.len(), 1325);
    assert_eq!(layouts[1].header.len(), 1197);
    assert_eq!(layouts[1].sha, SPLIT_LAYOUT);
    assert_ne!(layouts[0].sha, layouts[1].sha);
    let plans = walked_plans("gfx1201");
    assert_eq!(
        plans.len(),
        layouts.len() + 1,
        "canonical layouts plus split mq2"
    );
    assert!(plans.iter().any(|plan| {
        plan.layout == "92d2c376d98535639df92432b6255c8de577ac43f57a5e9f32fbc71c970d471a"
    }));
    for layout in &layouts {
        assert_eq!(
            walked("gfx1201", &layout.sha).layout,
            layout.sha,
            "{}",
            layout.name
        );
    }
    assert!(plans.iter().all(|plan| plan.family == plans[0].family));
}

/// walk ⊆ plan: every (module, symbol) the walked dispatch bound is planned
/// under its route with the current registry recipe (module identity, walked
/// symbol set, current flags/profile/source), and nothing else is, for each
/// header layout's own plan. The checked-in row digests are the historical
/// walk-time capture provenance and are never compared here; regeneration is
/// the existing `scripts/route_plan_walk.py` + `scripts/route_plan_build.py`
/// generator, not a hash alias or a partial plan.
#[test]
fn every_walked_kernel_is_planned() {
    for layout in &cases() {
        let (ar, mtp) = route_plans(layout);
        let walked = walked(layout.arch, &layout.sha);
        let inventory =
            entries(layout.arch, &builtin_flags(layout.arch).hipcc_extra_flags).unwrap();
        for (plan, native_mtp) in [(&ar, false), (&mtp, true)] {
            let rows: Vec<_> = walked
                .rows
                .iter()
                .filter(|row| row.native_mtp == native_mtp)
                .collect();
            assert_eq!(
                plan.entries.len(),
                rows.len(),
                "{} {} native_mtp={native_mtp}",
                layout.arch,
                layout.name
            );
            for row in rows {
                let entry = plan
                    .entries
                    .iter()
                    .find(|entry| entry.module() == row.module)
                    .unwrap_or_else(|| {
                        panic!("{}: walked {} is not planned", layout.name, row.module)
                    });
                assert_eq!(
                    entry.symbols(),
                    &row.symbols[..],
                    "{} {}",
                    layout.name,
                    row.module
                );
                if let PlannedKernel::Hip(e) = entry {
                    let live = inventory
                        .iter()
                        .find(|live| live.module == row.module)
                        .unwrap_or_else(|| {
                            panic!("{}: registry has no walked {}", layout.name, row.module)
                        });
                    assert_eq!(
                        e.source(),
                        live.source(),
                        "{} {} is not the current registry source",
                        layout.name,
                        row.module
                    );
                    assert_eq!(
                        e.flags, live.flags,
                        "{} {} flags are not the current registry recipe",
                        layout.name, row.module
                    );
                    assert_eq!(
                        e.scheduler_profile, live.scheduler_profile,
                        "{} {} profile is not the current registry recipe",
                        layout.name, row.module
                    );
                } else {
                    let sha = match entry {
                        PlannedKernel::Embedded { image, .. } => {
                            format!("{:x}", Sha256::digest(image))
                        }
                        PlannedKernel::Hip(_) => unreachable!(),
                    };
                    assert_eq!(sha, row.sha256, "{} {}", layout.name, row.module);
                }
                assert_eq!(
                    row.embedded,
                    matches!(entry, PlannedKernel::Embedded { .. }),
                    "{} {}",
                    layout.name,
                    row.module
                );
            }
        }
        // The MTP route falls back to AR, so its plan carries every AR binding.
        let (ar, mtp) = (bindings(&ar), bindings(&mtp));
        for (symbol, owner) in &ar {
            assert_eq!(
                mtp.get(symbol),
                Some(owner),
                "{}: AR binding {symbol} missing from the MTP plan",
                layout.name
            );
        }
    }
}

/// plan ⊆ pack: every planned HIP module is an installer-inventory entry (what
/// `scripts/build-kernel-pack.sh` ships) with the planned source, recipe and
/// every planned symbol in its index, so an installed pack satisfies the whole
/// plan with zero compiles. Every walked arch's and layout's plans.
#[test]
fn every_planned_hip_module_is_in_the_release_pack() {
    for layout in &cases() {
        let flags = builtin_flags(layout.arch);
        let pack = entries(layout.arch, &flags.hipcc_extra_flags).unwrap();
        let (ar, mtp) = route_plans(layout);
        let mut hip = 0;
        for entry in ar.entries.iter().chain(&mtp.entries) {
            let PlannedKernel::Hip(planned) = entry else {
                continue;
            };
            let packed = pack
                .iter()
                .find(|e| e.module == planned.module)
                .unwrap_or_else(|| {
                    panic!(
                        "{} {}: planned {} is not in the {} pack",
                        layout.arch, layout.name, planned.module, layout.arch
                    )
                });
            assert_eq!(
                packed.source(),
                planned.source(),
                "{} {} {}",
                layout.arch,
                layout.name,
                planned.module
            );
            assert_eq!(
                packed.flags, planned.flags,
                "{} {} {}",
                layout.arch, layout.name, planned.module
            );
            assert_eq!(
                packed.scheduler_profile, planned.scheduler_profile,
                "{} {} {}",
                layout.arch, layout.name, planned.module
            );
            for symbol in planned.symbols {
                assert!(
                    packed.symbols.contains(symbol),
                    "{} {}: pack index of {} lacks {symbol}",
                    layout.arch,
                    layout.name,
                    planned.module
                );
            }
            hip += 1;
        }
        assert!(hip > 0, "{} {}", layout.arch, layout.name);
    }
}

/// Loads with policies the walks did not cover get no plan (the caller keeps
/// lazy loading); a kernel the walks missed on a planned load warns and loads
/// (`scratch::route_load_tests`). Checked against each layout's own plan.
#[test]
fn unwalked_flash_next_policies_keep_lazy_loading() {
    for layout in &cases() {
        let flags = builtin_flags(layout.arch);
        let tensors = metas(&layout.header);
        let walked = walked(layout.arch, &layout.sha);
        let axes = walked_axes(walked);
        let base = Load {
            arch: layout.arch,
            tensors: &tensors,
            max_seq: walked.max_seq,
            spec: KernelSpecRoute::NativeMtp,
            flags: &flags,
            axes: &axes,
        };
        let refused =
            |load: &Load<'_>| matches!(plan(load), Err(RegistryError::UnsupportedRoute(_)));
        let name = format!("{} {}", layout.arch, layout.name);
        assert!(
            plan(&base).is_ok(),
            "{name}: the walked ceiling itself is admitted"
        );
        assert!(
            refused(&Load {
                max_seq: walked.max_seq + 1,
                ..base
            }),
            "{name}"
        );
        for spec in [
            KernelSpecRoute::DFlash,
            KernelSpecRoute::DDTree,
            KernelSpecRoute::DSpark,
        ] {
            assert!(refused(&Load { spec, ..base }), "{name} {spec:?}");
        }
        // A policy axis outside the walks (the other QSA format than the walked one),
        // a missing axis, an unknown axis.
        let mut other_qsa = axes.clone();
        let qsa = &mut other_qsa
            .iter_mut()
            .find(|(axis, _)| *axis == "qsa")
            .unwrap()
            .1;
        *qsa = if qsa == "q8" { "fp8" } else { "q8" }.into();
        assert!(
            refused(&Load {
                axes: &other_qsa,
                ..base
            }),
            "{name}"
        );
        assert!(
            refused(&Load {
                axes: &axes[1..],
                ..base
            }),
            "{name}"
        );
        let mut extra = axes.clone();
        extra.push(("unwalked_axis", "on".into()));
        assert!(
            refused(&Load {
                axes: &extra,
                ..base
            }),
            "{name}"
        );
        // Any non-default feature flag, including kernel.pm_decode.
        let mut changed = flags.clone();
        changed.hipcc_extra_flags.push_str(" -DFN_ROUTE_TEST=1");
        assert!(
            refused(&Load {
                flags: &changed,
                ..base
            }),
            "{name}"
        );
        let mut pm = flags.clone();
        pm.pm_decode = !pm.pm_decode;
        assert!(refused(&Load { flags: &pm, ..base }), "{name}");
        // Archs without a walked plan have no closed route, and another arch's loads never
        // borrow this arch's plan (the axes passed are this arch's walked ones).
        for arch in ["gfx1100", "gfx1151", "gfx906", "gfx1201"] {
            if arch == layout.arch {
                continue;
            }
            let arch_flags = builtin_flags(arch);
            assert!(
                refused(&Load {
                    arch,
                    flags: &arch_flags,
                    ..base
                }),
                "{name} {arch}"
            );
        }
    }
}

/// A header that is neither walked layout (a re-quantized artifact, a partial
/// ngram shard family, an extra tensor) has no walked route, whichever walked
/// layout it was derived from, on each walked arch.
#[test]
fn unknown_header_layout_has_no_walked_route() {
    let canonical = flash_next_header();
    let shard_0 = canonical
        .iter()
        .find(|(name, _, _)| *name == ngram_shard_name(0))
        .unwrap()
        .clone();
    let last_shard = ngram_shard_name(NGRAM_SHARDS - 1);
    for layout in cases() {
        let flags = builtin_flags(layout.arch);
        let walked_layouts: Vec<String> = walked_plans(layout.arch)
            .iter()
            .map(|plan| plan.layout.to_owned())
            .collect();
        let axes = walked_axes(walked(layout.arch, &layout.sha));
        let mut unknown = Vec::new();
        let mut dropped = layout.header.clone();
        dropped.pop();
        unknown.push(("last tensor dropped", dropped));
        let mut extra = layout.header.clone();
        extra.push(("extra.weight".to_owned(), DType::Q8_0, vec![1]));
        unknown.push(("extra tensor", extra));
        let mut partial = layout.header.clone();
        if layout.name == "split" {
            partial.push(shard_0.clone());
        } else {
            partial.retain(|(name, _, _)| *name != last_shard);
        }
        unknown.push(("partial ngram shard family", partial));
        for (what, header) in unknown {
            let tensors = metas(&header);
            let sha = route_layout_sha256(&tensors);
            assert!(
                !walked_layouts.contains(&sha),
                "{} {} {what} must not hash to a walked layout",
                layout.arch,
                layout.name
            );
            for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
                let load = Load {
                    arch: layout.arch,
                    tensors: &tensors,
                    max_seq: 16_384,
                    spec,
                    flags: &flags,
                    axes: &axes,
                };
                match plan(&load) {
                    Err(RegistryError::UnsupportedRoute(why)) => {
                        assert!(
                            why.contains("has no walked route"),
                            "{} {} {what} {spec:?}: {why}",
                            layout.arch,
                            layout.name
                        )
                    }
                    other => panic!(
                        "{} {} {what} {spec:?}: expected UnsupportedRoute, got {:?}",
                        layout.arch,
                        layout.name,
                        other.map(|p| p.entries.len())
                    ),
                }
            }
        }
    }
}

/// The gfx1100 plan: the canonical header walked once on the XTX (q8 QSA and GDN
/// state, legacy KV backend, `max_seq` 262,144), left open (a bounded walk never
/// seals a route), and no other header has a gfx1100 route.
#[test]
fn gfx1100_plan_is_the_canonical_header_walked_open_at_q8_legacy() {
    let canonical = layout("canonical", "gfx1100", flash_next_header());
    assert_eq!(walked_plans("gfx1100").len(), 1);
    let plan = walked("gfx1100", &canonical.sha);
    assert_eq!(plan.max_seq, 262_144);
    assert!(plan.sealed.is_empty(), "a bounded walk never seals");
    assert!(plan.heads.is_none());
    assert_eq!(
        plan.axes,
        vec![
            ("gdn", vec!["q8"]),
            ("kv_backend", vec!["legacy"]),
            ("qsa", vec!["q8"]),
            ("qsa_f16_arena", vec!["off"]),
            ("qsa_pooled_bf16", vec!["off"]),
        ]
    );
    // The split header was walked on gfx1201 only.
    let split = layouts().remove(1);
    let flags = builtin_flags("gfx1100");
    let tensors = metas(&split.header);
    let axes = walked_axes(plan);
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        match self::plan(&Load {
            arch: "gfx1100",
            tensors: &tensors,
            max_seq: 16_384,
            spec,
            flags: &flags,
            axes: &axes,
        }) {
            Err(RegistryError::UnsupportedRoute(why)) => {
                assert!(why.contains("has no walked route"), "{spec:?}: {why}")
            }
            other => panic!(
                "split on gfx1100 {spec:?}: {:?}",
                other.map(|p| p.entries.len())
            ),
        }
    }
}

/// The gfx1100 route includes the kernels the XTX default route binds since the
/// landed Peacemaker/lite work: the embedded MQ6 residual PM twin (`_w4`, `_w8`;
/// the committed image) and the lite hg4 batched Q8 attention, on both spec routes.
#[test]
fn gfx1100_plan_carries_the_landed_kernels_on_both_routes() {
    let canonical = layout("canonical", "gfx1100", flash_next_header());
    let (ar, mtp) = route_plans(&canonical);
    for (route, plan) in [("ar", &ar), ("native_mtp", &mtp)] {
        let bound = bindings(plan);
        for symbol in [
            "gemm_mq6g256v2_residual_pm_gfx1100_w4",
            "gemm_mq6g256v2_residual_pm_gfx1100_w8",
        ] {
            assert_eq!(
                bound.get(symbol),
                Some(&("gemm_mq6g256v2_residual_pm_gfx1100", true)),
                "{route} {symbol}"
            );
        }
        assert_eq!(
            bound.get("indexed_attention_attention_q8_batched_hg4_lite"),
            Some(&("tensor_ops", false)),
            "{route}"
        );
        let image = plan
            .entries
            .iter()
            .find_map(|entry| match entry {
                PlannedKernel::Embedded {
                    module: "gemm_mq6g256v2_residual_pm_gfx1100",
                    image,
                    ..
                } => Some(*image),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{route}: no embedded MQ6 residual PM entry"));
        assert_eq!(
            format!("{:x}", Sha256::digest(image)),
            rdna_compute::kernels::GEMM_MQ6G256V2_RESIDUAL_PM_GFX1100_SHA256.trim(),
            "{route}"
        );
    }
}

fn sha_hex(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

/// The exact BF16 split QSA selector is the gfx1100 default
/// (`HIPFIRE_QWEN4_QSA_SPLIT_GFX1100` on unless `0`), so its two modules are in
/// the gfx1100 plan of BOTH spec routes with exactly the symbols the split
/// launchers bind (`tensor_ops::qsa_bf16_split_select`: rows8/rows16 scores and
/// the filter trio); a post-ready first split load no longer binds lazily.
#[test]
fn gfx1100_plan_carries_the_bf16_split_qsa_modules_on_both_routes() {
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
    let canonical = layout("canonical", "gfx1100", flash_next_header());
    let walked = walked("gfx1100", &canonical.sha);
    let (ar, mtp) = route_plans(&canonical);
    for (route, plan, native_mtp) in [("ar", &ar, false), ("native_mtp", &mtp, true)] {
        let bound = bindings(plan);
        for (module, symbols) in SPLIT {
            let row = walked
                .rows
                .iter()
                .find(|row| row.native_mtp == native_mtp && row.module == module)
                .unwrap_or_else(|| panic!("{route}: no walked row for {module}"));
            assert!(!row.embedded, "{route} {module}");
            assert_eq!(row.symbols, symbols, "{route} {module} walked symbols");
            for &symbol in symbols {
                assert_eq!(
                    bound.get(symbol),
                    Some(&(module, false)),
                    "{route} {symbol}"
                );
            }
        }
    }
}

/// The sha256 column of every gfx1100 HIP row is the EFFECTIVE source the loader
/// hands to the module load (`scratch::module_load_or_recompile` hashes its
/// `source` argument for `HIPFIRE_MODULE_LOAD_AUDIT`). The route preload passes
/// `PlannedKernel::Hip(entry).source()` (`scratch::prepare_route_modules`), the
/// lazy launchers pass the `kernels::*_SRC` constant they share with the
/// registry entry; no arch define or include is prepended on either path. Both
/// are checked against the row for every route, so a plan row can neither name
/// a stale digest nor a source the loader would not load.
#[test]
fn gfx1100_hip_rows_hash_the_effective_source_the_loader_binds() {
    let canonical = layout("canonical", "gfx1100", flash_next_header());
    let walked = walked("gfx1100", &canonical.sha);
    let (ar, mtp) = route_plans(&canonical);
    let inventory = entries("gfx1100", &builtin_flags("gfx1100").hipcc_extra_flags).unwrap();
    let mut checked = 0;
    for (route, plan, native_mtp) in [("ar", &ar, false), ("native_mtp", &mtp, true)] {
        for row in walked
            .rows
            .iter()
            .filter(|row| row.native_mtp == native_mtp && !row.embedded)
        {
            let Some(PlannedKernel::Hip(planned)) = plan
                .entries
                .iter()
                .find(|entry| entry.module() == row.module)
            else {
                panic!("{route}: {} is not a planned HIP entry", row.module);
            };
            assert_eq!(
                sha_hex(planned.source()),
                row.sha256,
                "{route} {}: row sha256 is not the planned effective source",
                row.module
            );
            let live = inventory
                .iter()
                .find(|e| e.module == row.module)
                .unwrap_or_else(|| panic!("{route}: pack lacks {}", row.module));
            assert_eq!(
                sha_hex(live.source()),
                row.sha256,
                "{route} {}: row sha256 is not the registry source",
                row.module
            );
            checked += 1;
        }
    }
    assert!(checked > 80, "only {checked} gfx1100 hip rows checked");
    // Lazy launcher inputs of the split route: the constants themselves.
    for (module, source) in [
        (
            "qsa_bf16_scores",
            rdna_compute::kernels::QSA_BF16_SCORES_SRC,
        ),
        (
            "qsa_filter_select",
            rdna_compute::kernels::QSA_FILTER_SELECT_SRC,
        ),
    ] {
        let live = inventory.iter().find(|e| e.module == module).unwrap();
        assert_eq!(
            live.source(),
            source,
            "{module}: registry != launcher source"
        );
        for plan in [&ar, &mtp] {
            let row_sha = walked
                .rows
                .iter()
                .find(|row| row.module == module)
                .map(|row| row.sha256)
                .unwrap();
            assert_eq!(sha_hex(source), row_sha, "{module}");
            assert!(plan.entries.iter().any(|e| e.module() == module));
        }
    }
}
