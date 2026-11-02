// SPDX-License-Identifier: Apache-2.0
// A3 route-closure gate: CPU-only planner tests. No compiler process or HIP.
use rdna_compute::kernel_registry::{
    route_entries, validate_route_plan, KernelEntry, KernelSpecRoute, KernelTensorMeta,
    PlannedKernel, RegistryError, RouteKernelInput, RouteKernelPlan,
};
use rdna_compute::{DType, FeatureFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const G0: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jit-g0/");
const A3: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jit-a3/");

fn fixture(dir: &str, name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(dir).join(name)).unwrap()).unwrap()
}

fn dtype(tag: u64) -> DType {
    match tag {
        1 => DType::F16,
        2 => DType::F32,
        3 => DType::Q8_0,
        13 => DType::MQ4G256,
        16 => DType::BF16,
        44 => DType::MQ4G256V2,
        47 => DType::MQ6G256V2,
        53 => DType::MQ4G128V2,
        54 => DType::Raw,
        other => panic!("header fixture quant tag {other} has no test mapping"),
    }
}

/// Owned header metadata so `KernelTensorMeta` can borrow names and shapes.
struct Header {
    arch_id: u32,
    tensors: Vec<(String, DType, Vec<usize>)>,
}

fn header(dir: &str, name: &str) -> Header {
    let value = fixture(dir, name);
    Header {
        arch_id: value["arch_id"].as_u64().unwrap() as u32,
        tensors: value["tensors"]
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
            .collect(),
    }
}

fn metas<'a>(headers: &[&'a Header]) -> Vec<KernelTensorMeta<'a>> {
    headers
        .iter()
        .flat_map(|h| h.tensors.iter())
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

struct Load<'a> {
    arch: &'a str,
    model_arch: u32,
    tensors: &'a [KernelTensorMeta<'a>],
    heads: (usize, usize, usize),
    kv: (&'a str, &'a str),
    spec: KernelSpecRoute,
    flags: &'a FeatureFlags,
}

fn plan(load: &Load<'_>) -> Result<RouteKernelPlan, RegistryError> {
    route_entries(&RouteKernelInput {
        arch: load.arch,
        model_arch: load.model_arch,
        tensors: load.tensors,
        n_heads: load.heads.0,
        n_kv_heads: load.heads.1,
        head_dim: load.heads.2,
        max_seq: 262_144,
        prefill_chunk_rows: 8_192,
        kv_k: load.kv.0,
        kv_v: load.kv.1,
        state_quant: "q8",
        spec: load.spec,
        host_mapped_experts: false,
        retained_decode: true,
        hipcc_extra_flags: &load.flags.hipcc_extra_flags,
        flags: load.flags,
        route_axes: &[],
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `module -> (kind, image/source sha256, flags, profile, symbols)`.
fn manifest(
    plan: &RouteKernelPlan,
) -> BTreeMap<String, (String, String, Value, Value, BTreeSet<String>)> {
    let mut out = BTreeMap::new();
    for entry in &plan.entries {
        let symbols = entry.symbols().iter().map(|s| s.to_string()).collect();
        let row = match entry {
            PlannedKernel::Hip(e) => (
                "hip".to_owned(),
                sha256(e.source().as_bytes()),
                serde_json::json!(e.flags),
                serde_json::json!(e.scheduler_profile),
                symbols,
            ),
            PlannedKernel::Embedded { image, .. } => (
                "embedded".to_owned(),
                sha256(image),
                Value::Null,
                Value::Null,
                symbols,
            ),
        };
        assert!(
            out.insert(entry.module().to_owned(), row).is_none(),
            "duplicate module {}",
            entry.module()
        );
    }
    out
}

/// The 27B plan for the canonical header minus the tensors `keep_tensor`
/// rejects (every `*.awq_scale.weight` sidecar for a no-AWQ load) plus the MTP head.
fn plan_27b(
    spec: KernelSpecRoute,
    pm_decode: bool,
    keep_tensor: impl Fn(&str) -> bool,
) -> RouteKernelPlan {
    let mut trunk = header(G0, "header-27b.json");
    trunk.tensors.retain(|(name, _, _)| keep_tensor(name));
    let head = header(A3, "header-27b-mtp.json");
    let tensors = metas(&[&trunk, &head]);
    let mut flags = builtin_flags("gfx1201");
    flags.pm_decode = pm_decode;
    plan(&Load {
        arch: "gfx1201",
        model_arch: trunk.arch_id,
        tensors: &tensors,
        heads: (24, 4, 256),
        kv: ("fp8", "fp8"),
        spec,
        flags: &flags,
    })
    .unwrap()
}

/// The frozen A3 manifests and historical observed identities describe the hipcc closure:
/// the `kernel.pm_decode=false` opt-out. Their recorded `sha256` values are historical
/// capture hashes, never compared against current sources. The default (twins on) closure is
/// derived from it exactly in [`pm_decode_twins_replace_their_hip_route_entries`].
fn route_27b(spec: KernelSpecRoute) -> (Value, RouteKernelPlan) {
    (
        fixture(A3, "27b-xts-gfx1201.a3.json"),
        plan_27b(spec, false, |_| true),
    )
}

fn route_key(spec: KernelSpecRoute) -> &'static str {
    match spec {
        KernelSpecRoute::Ar => "ar",
        KernelSpecRoute::NativeMtp => "native_mtp",
        _ => unreachable!(),
    }
}

#[test]
fn qwen36_27b_gfx1201_plans_match_frozen_recipe_manifests() {
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        let (fixture, plan) = route_27b(spec);
        assert_eq!(
            fixture["hipcc_extra_flags"],
            builtin_flags("gfx1201").hipcc_extra_flags.as_str()
        );
        let route = &fixture["routes"][route_key(spec)];
        let got = manifest(&plan);
        // HIP source-text hashes are historical capture provenance, not current
        // recipe identities. Embedded binary image hashes remain strict.
        let mut want: BTreeMap<String, (String, Value, Value, BTreeSet<String>)> = BTreeMap::new();
        for e in route["entries"].as_array().unwrap() {
            if e["kind"].as_str() == Some("embedded") {
                assert_eq!(
                    got[e["module"].as_str().unwrap()].1.as_str(),
                    e["sha256"].as_str().unwrap(),
                    "{spec:?}: embedded image differs from the captured image"
                );
            }
            let symbols = e["symbols"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect();
            want.insert(
                e["module"].as_str().unwrap().to_owned(),
                (
                    e["kind"].as_str().unwrap().to_owned(),
                    e["flags"].clone(),
                    e["scheduler_profile"].clone(),
                    symbols,
                ),
            );
        }
        let got_recipe: BTreeMap<String, (String, Value, Value, BTreeSet<String>)> = got
            .iter()
            .map(|(module, (kind, _, flags, profile, symbols))| {
                (
                    module.clone(),
                    (
                        kind.clone(),
                        flags.clone(),
                        profile.clone(),
                        symbols.clone(),
                    ),
                )
            })
            .collect();
        assert_eq!(
            got_recipe, want,
            "{spec:?} plan drifted from its frozen manifest"
        );
        let symbols: usize = got.values().map(|v| v.4.len()).sum();
        assert_eq!(symbols, route["symbol_count"].as_u64().unwrap() as usize);
    }
}

#[test]
fn every_historical_route_symbol_is_planned_with_current_identity() {
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        let (fixture, plan) = route_27b(spec);
        let got = manifest(&plan);
        let observed = fixture["observed"][route_key(spec)].as_array().unwrap();
        assert!(!observed.is_empty());
        for o in observed {
            let module = o["module"].as_str().unwrap();
            let symbol = o["symbol"].as_str().unwrap();
            let (kind, digest, flags, profile, symbols) = got
                .get(module)
                .unwrap_or_else(|| panic!("{spec:?}: observed module {module} not planned"));
            assert!(
                symbols.contains(symbol),
                "{spec:?}: observed {module}:{symbol} not planned"
            );
            assert_eq!(kind, o["kind"].as_str().unwrap(), "{module}");
            // Only HIP source text is incidental; embedded image identity is not.
            if kind == "hip" {
                assert_eq!(flags, &o["flags"], "{module}: observed recipe flags differ");
                assert_eq!(profile, &o["scheduler_profile"], "{module}");
            } else {
                assert_eq!(
                    digest,
                    o["sha256"].as_str().unwrap(),
                    "{module}: embedded image differs"
                );
            }
        }
    }
}

#[test]
fn ar_plan_carries_no_native_mtp_head_or_unrelated_model_recipe() {
    let (_, ar) = route_27b(KernelSpecRoute::Ar);
    let (fixture, mtp) = route_27b(KernelSpecRoute::NativeMtp);
    let ar = manifest(&ar);
    let mtp = manifest(&mtp);
    assert!(
        ar.keys().all(|m| mtp.contains_key(m)),
        "MTP route must keep the AR fallback closure"
    );
    for m in fixture["routes"]["native_mtp"]["mtp_only_modules"]
        .as_array()
        .unwrap()
    {
        assert!(
            !ar.contains_key(m.as_str().unwrap()),
            "AR plan carries MTP-only {m}"
        );
    }
    for module in mtp.keys() {
        for foreign in [
            "qwen4",
            "deepseek4",
            "gemma4",
            "glimmer",
            "minimax",
            "qsa_",
            "ddtree",
            "dspark",
            "_gfx1151",
            "eagle",
        ] {
            assert!(!module.contains(foreign), "unrelated recipe {module}");
        }
    }
}

const MTP_HEAD_KERNELS: [&str; 5] = [
    "fused_rmsnorm_mq_rotate",
    "fused_silu_mul_mq_rotate",
    "fused_gate_up_hfq4g256",
    "fused_qkv_hfq4g256",
    "gemv_hfq4g256_residual",
];

/// The MQ4-v1 sidecar uses the typed layer program's plain fused producers
/// and residual projections even when every trunk projection has AWQ scales.
#[test]
fn native_mtp_closes_sidecar_kernels_independently_of_trunk_awq() {
    for pm_decode in [false, true] {
        for keep_awq in [false, true] {
            let keep = |name: &str| keep_awq || !name.ends_with(".awq_scale.weight");
            let ar = manifest(&plan_27b(KernelSpecRoute::Ar, pm_decode, keep));
            let mtp = manifest(&plan_27b(KernelSpecRoute::NativeMtp, pm_decode, keep));
            for module in MTP_HEAD_KERNELS {
                let entry = mtp.get(module).unwrap_or_else(|| {
                    panic!("pm_decode={pm_decode} trunk_awq={keep_awq}: missing {module}")
                });
                assert!(entry.4.contains(module), "missing export {module}");
                if keep_awq {
                    assert!(
                        !ar.contains_key(module),
                        "AR must not gain head-only {module}"
                    );
                }
            }
        }
    }
}

/// A no-AWQ 27B load (the canonical header minus every `*.awq_scale.weight`
/// sidecar) launches the plain rotate producers: RMSNorm, SwiGLU and gated
/// norm rotation, the standalone rotate, the gfx1201 non-slab IU4 prefill
/// producers (`awq == None` never selects the slab twins) and the non-slab
/// `_b1` GEMMs that read them. The AWQ-only closure planned none of them.
#[test]
fn no_awq_27b_plan_closes_the_plain_rotate_producers() {
    const PLAIN: [(&str, &str); 12] = [
        ("fused_rmsnorm_mq_rotate", "fused_rmsnorm_mq_rotate"),
        (
            "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
            "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
        ),
        ("fused_silu_mul_mq_rotate", "fused_silu_mul_mq_rotate"),
        (
            "fused_silu_mul_mq_rotate_i4_gfx12",
            "fused_silu_mul_mq_rotate_i4_gfx12",
        ),
        (
            "gated_norm_mq_rotate_k6144_gfx1201",
            "gated_norm_mq_rotate_k6144_gfx1201",
        ),
        (
            "gated_norm_mq_rotate_i4_gfx12_v2",
            "gated_norm_mq_rotate_i4_gfx12_v2",
        ),
        (
            "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
            "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
        ),
        ("mq_rotate_x", "mq_rotate_x"),
        ("mq_rotate_x_i4_gfx12", "mq_rotate_x_i4_gfx12"),
        (
            "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1",
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1",
        ),
        (
            "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1",
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1",
        ),
        (
            "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1",
            "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1",
        ),
    ];
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        for pm_decode in [false, true] {
            let plan = plan_27b(spec, pm_decode, |name| !name.ends_with(".awq_scale.weight"));
            let got = manifest(&plan);
            for (module, symbol) in PLAIN {
                let (_, _, _, _, symbols) = got.get(module).unwrap_or_else(|| {
                    panic!("{spec:?} pm_decode={pm_decode}: no-AWQ plan lacks module {module}")
                });
                assert!(
                    symbols.contains(symbol),
                    "{spec:?} pm_decode={pm_decode}: no-AWQ plan lacks {module}:{symbol}"
                );
            }
        }
    }
}

/// Plain-only modules of the 27B route by producer family; `mq_rotate_x` is
/// already in the AWQ closure and not listed.
const PLAIN_FAMILIES: [(&str, &[&str]); 4] = [
    (
        "rmsnorm",
        &[
            "fused_rmsnorm_mq_rotate",
            "fused_rmsnorm_mq_rotate_i4_gfx12_v2",
        ],
    ),
    (
        "silu",
        &[
            "fused_silu_mul_mq_rotate",
            "fused_silu_mul_mq_rotate_i4_gfx12",
        ],
    ),
    (
        "gated_norm",
        &[
            "gated_norm_mq_rotate_k6144_gfx1201",
            "gated_norm_mq_rotate_i4_gfx12_v2",
            "gated_norm_mq_rotate_i4_gfx12_v2_xbf16",
        ],
    ),
    ("rotate", &["mq_rotate_x_i4_gfx12"]),
];
const PLAIN_B1: &str = "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1";
const PLAIN_B1_SILU_SYMBOL: &str = "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1";

type Manifest = BTreeMap<String, (String, String, Value, Value, BTreeSet<String>)>;

/// The plan manifest with the sidecars of the given weight stems
/// (e.g. `.mlp.down_proj`) removed from the canonical AWQ header.
fn manifest_without_sidecars(spec: KernelSpecRoute, pm_decode: bool, stems: &[&str]) -> Manifest {
    let dropped: Vec<String> = stems
        .iter()
        .map(|s| format!("{s}.awq_scale.weight"))
        .collect();
    manifest(&plan_27b(spec, pm_decode, |name| {
        !dropped.iter().any(|d| name.ends_with(d.as_str()))
    }))
}

/// The AWQ trunk closure is untouched: plain-only producers belong only to
/// the MTP head. Dropping sidecars of weights that feed no plain trunk
/// producer changes nothing.
#[test]
fn awq_27b_plan_is_unchanged_by_the_plain_closure() {
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        for pm_decode in [false, true] {
            let awq = manifest_without_sidecars(spec, pm_decode, &[]);
            for (family, modules) in PLAIN_FAMILIES {
                for module in modules {
                    assert!(
                        awq.contains_key(*module)
                            == (matches!(spec, KernelSpecRoute::NativeMtp)
                                && MTP_HEAD_KERNELS.contains(module)),
                        "{spec:?} pm_decode={pm_decode}: AWQ plan carries {family} module {module}"
                    );
                }
            }
            assert!(
                !awq.contains_key(PLAIN_B1),
                "{spec:?} pm_decode={pm_decode}: AWQ plan carries the non-slab _b1 bundle"
            );
            let unrelated = manifest_without_sidecars(
                spec,
                pm_decode,
                &[
                    ".self_attn.k_proj",
                    ".self_attn.v_proj",
                    ".linear_attn.in_proj_a",
                    ".linear_attn.in_proj_b",
                    ".mlp.up_proj",
                ],
            );
            assert_eq!(
                awq, unrelated,
                "{spec:?} pm_decode={pm_decode}: sidecars no producer reads changed the plan"
            );
        }
    }
}

/// Each trunk producer family is planned exactly when a weight it feeds lacks
/// its sidecar, in addition to the independent MTP head closure. The non-slab
/// `_b1` bundle and its F1-lite symbol still depend only on the trunk.
#[test]
fn partial_sidecars_plan_only_their_plain_families() {
    let cases: [(&[&str], &[&str]); 5] = [
        (&[".mlp.down_proj"], &["silu"]),
        (&[".linear_attn.out_proj"], &["gated_norm"]),
        (&[".self_attn.o_proj"], &["rotate"]),
        (&[".self_attn.q_proj"], &["rmsnorm"]),
        (&[".linear_attn.in_proj_qkv"], &["rmsnorm"]),
    ];
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        for pm_decode in [false, true] {
            for (stems, want) in cases {
                let got = manifest_without_sidecars(spec, pm_decode, stems);
                for (family, modules) in PLAIN_FAMILIES {
                    for module in modules {
                        assert_eq!(
                            got.contains_key(*module),
                            want.contains(&family)
                                || (matches!(spec, KernelSpecRoute::NativeMtp)
                                    && MTP_HEAD_KERNELS.contains(module)),
                            "{spec:?} pm_decode={pm_decode} without {stems:?}: module {module} ({family})"
                        );
                    }
                }
                let (_, _, _, _, symbols) = got.get(PLAIN_B1).unwrap_or_else(|| {
                    panic!("{spec:?} pm_decode={pm_decode} without {stems:?}: no _b1 bundle")
                });
                assert!(symbols.contains("gemm_mq4g256v2_residual_mmq_iu4_full_add_b1"));
                assert_eq!(
                    symbols.contains(PLAIN_B1_SILU_SYMBOL),
                    want.contains(&"rmsnorm"),
                    "{spec:?} pm_decode={pm_decode} without {stems:?}: F1-lite _b1 symbol"
                );
            }
        }
    }
}

/// Families aggregate across layers: layer 0 lacks only its gate sidecar
/// (plain RMSNorm producer, AWQ down) while layer 1 lacks only its down
/// sidecar. Layer 0 still needs the F1-lite `_b1` gate/up SwiGLU symbol even
/// though the model also has a plain SwiGLU family.
#[test]
fn cross_layer_partial_sidecars_keep_the_f1lite_b1_symbol() {
    const GATE0: &str = "model.language_model.layers.0.mlp.gate_proj.awq_scale.weight";
    const DOWN1: &str = "model.language_model.layers.1.mlp.down_proj.awq_scale.weight";
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        for pm_decode in [false, true] {
            let got = manifest(&plan_27b(spec, pm_decode, |name| {
                name != GATE0 && name != DOWN1
            }));
            for (family, modules) in PLAIN_FAMILIES {
                for module in modules {
                    assert_eq!(
                        got.contains_key(*module),
                        matches!(family, "rmsnorm" | "silu"),
                        "{spec:?} pm_decode={pm_decode}: module {module} ({family})"
                    );
                }
            }
            let (_, _, _, _, symbols) = &got[PLAIN_B1];
            assert!(
                symbols.contains(PLAIN_B1_SILU_SYMBOL),
                "{spec:?} pm_decode={pm_decode}: layer 0 needs the F1-lite _b1 symbol"
            );
        }
    }
}

#[test]
fn explicit_drafter_and_unclosed_routes_refuse() {
    let trunk = header(G0, "header-27b.json");
    let head = header(A3, "header-27b-mtp.json");
    let with_head = metas(&[&trunk, &head]);
    let trunk_only = metas(&[&trunk]);
    let flags = builtin_flags("gfx1201");
    let base = Load {
        arch: "gfx1201",
        model_arch: trunk.arch_id,
        tensors: &with_head,
        heads: (24, 4, 256),
        kv: ("fp8", "fp8"),
        spec: KernelSpecRoute::Ar,
        flags: &flags,
    };
    let refused = |load: &Load<'_>| matches!(plan(load), Err(RegistryError::UnsupportedRoute(_)));
    for spec in [
        KernelSpecRoute::DFlash,
        KernelSpecRoute::DDTree,
        KernelSpecRoute::DSpark,
    ] {
        assert!(refused(&Load { spec, ..base }), "{spec:?} must refuse");
    }
    assert!(refused(&Load {
        spec: KernelSpecRoute::NativeMtp,
        tensors: &trunk_only,
        ..base
    }));
    assert!(
        plan(&Load {
            tensors: &trunk_only,
            ..base
        })
        .is_ok(),
        "AR needs no sidecar head"
    );
    assert!(refused(&Load {
        kv: ("q8", "q8"),
        ..base
    }));
    assert!(refused(&Load {
        kv: ("fp8", "q8"),
        ..base
    }));
    for arch in ["gfx1151", "gfx1100"] {
        let arch_flags = builtin_flags(arch);
        assert!(
            refused(&Load {
                arch,
                flags: &arch_flags,
                ..base
            }),
            "{arch} 27B is not closed"
        );
    }
    let fn_header = header(G0, "header-flash-next.json");
    let fn_tensors = metas(&[&fn_header]);
    assert!(refused(&Load {
        model_arch: fn_header.arch_id,
        tensors: &fn_tensors,
        ..base
    }));
    // Same model id and heads, different tensors: not this route.
    let mut renamed = with_head.clone();
    renamed.retain(|t| !t.name.ends_with("mlp.gate_proj.weight"));
    assert!(refused(&Load {
        tensors: &renamed,
        ..base
    }));
    // Any non-default feature flag is outside the walked closure.
    let mut changed = flags.clone();
    changed.hipcc_extra_flags.push_str(" -DA3_TEST=1");
    assert!(refused(&Load {
        flags: &changed,
        ..base
    }));
}

#[test]
fn conflicting_public_entry_plans_refuse() {
    let (_, good) = route_27b(KernelSpecRoute::Ar);
    validate_route_plan(&good).unwrap();
    let hip = good
        .entries
        .iter()
        .find_map(|e| match e {
            PlannedKernel::Hip(e) if e.module == "rmsnorm_f32_rowsplit" => Some(e),
            _ => None,
        })
        .unwrap();
    let clone = |module: &'static str, source: &str| {
        PlannedKernel::Hip(KernelEntry {
            arch: hip.arch,
            module,
            symbols: hip.symbols,
            source: source.to_owned().into(),
            flags: hip.flags.clone(),
            scheduler_profile: hip.scheduler_profile.clone(),
        })
    };
    // Two modules exporting one public entry name.
    let two_images = RouteKernelPlan {
        entries: vec![
            clone("rmsnorm_f32_rowsplit", hip.source()),
            clone("rmsnorm_alt", hip.source()),
        ],
    };
    assert!(matches!(
        validate_route_plan(&two_images),
        Err(RegistryError::ConflictingPlan(_))
    ));
    // One module name, two recipes.
    let two_recipes = RouteKernelPlan {
        entries: vec![
            clone("rmsnorm_f32_rowsplit", hip.source()),
            clone("rmsnorm_f32_rowsplit", "changed"),
        ],
    };
    assert!(matches!(
        validate_route_plan(&two_recipes),
        Err(RegistryError::ConflictingPlan(_))
    ));
    // An embedded image and a HIP module both claiming an entry.
    let mixed = RouteKernelPlan {
        entries: vec![
            clone("rmsnorm_f32_rowsplit", hip.source()),
            PlannedKernel::Embedded {
                module: "pm_twin",
                image: b"x",
                radiowave_json: None,
                symbols: hip.symbols,
            },
        ],
    };
    assert!(matches!(
        validate_route_plan(&mixed),
        Err(RegistryError::ConflictingPlan(_))
    ));
    // Identical recipes under one module name union, they do not conflict.
    let same = RouteKernelPlan {
        entries: vec![
            clone("rmsnorm_f32_rowsplit", hip.source()),
            clone("rmsnorm_f32_rowsplit", hip.source()),
        ],
    };
    validate_route_plan(&same).unwrap();
}

/// The accepted PM multi-column objects (`pm_xbatch.rs`): `(module, image
/// sha256, symbol)`.
const PM_XBATCH_OBJECTS: [(&str, &str, &str); 3] = [
    (
        "gemv_hfq4g256_xbatch_mq4v2_pm",
        "97c2872246c123a03bbb4751fed75b2f7235654482c96b007eb69c2ffbb41b59",
        "gemv_mq4g256v2_xbatch_pm",
    ),
    (
        "gemv_hfq4g256_residual_xbatch_mq4v2_pm",
        "7ad33f79e552f550a46a96c0863760bb3ac31ac8d7223e256a59ade20eb4ade1",
        "gemv_mq4g256v2_residual_xbatch",
    ),
    (
        "gemv_hfq4g256_multirow_xbatch_mq4v2_pm",
        "fadb665f00c35bea787fda7c6956d9af61f5f6d1e60f759c2c7f2c23a1ddd9bf",
        "gemv_mq4g256v2_multirow_r2_xbatch_pm",
    ),
];

/// Default `kernel.pm_decode` on exact gfx1201: every accepted twin replaces its HIP
/// entry (same module and symbols, embedded image), the replaced HIP recipe
/// is the exact source the oracle accepted against, the VMM exact route's
/// hipcc x-batch GEMV fallback gives way to the accepted PM multi-column
/// objects (`pm_xbatch`), nothing else changes, and the flag never reaches
/// another architecture.
#[test]
fn pm_decode_twins_replace_their_hip_route_entries() {
    use rdna_compute::pm_decode_twins::GFX1201_TWINS;
    let trunk = header(G0, "header-27b.json");
    let head = header(A3, "header-27b-mtp.json");
    let tensors = metas(&[&trunk, &head]);
    for spec in [KernelSpecRoute::Ar, KernelSpecRoute::NativeMtp] {
        let on_flags = builtin_flags("gfx1201");
        assert!(
            on_flags.pm_decode,
            "pm_decode must default on for exact gfx1201"
        );
        let mut off_flags = on_flags.clone();
        off_flags.pm_decode = false;
        let load = |flags| Load {
            arch: "gfx1201",
            model_arch: trunk.arch_id,
            tensors: &tensors,
            heads: (24, 4, 256),
            kv: ("fp8", "fp8"),
            spec,
            flags,
        };
        let off_plan = plan(&load(&off_flags)).unwrap();
        let on_plan = plan(&load(&on_flags)).unwrap();
        validate_route_plan(&on_plan).unwrap();
        let mut off = manifest(&off_plan);
        let mut on = manifest(&on_plan);
        for twin in GFX1201_TWINS {
            let hip = off
                .remove(twin.module)
                .unwrap_or_else(|| panic!("{} not in HIP route", twin.module));
            assert_eq!(hip.0, "hip");
            let source_hex: String = twin
                .source_sha256
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                hip.1, source_hex,
                "{}: route HIP source is not the oracle's incumbent",
                twin.module
            );
            let pm = on.remove(twin.module).unwrap();
            assert_eq!(pm.0, "embedded", "{}", twin.module);
            assert_eq!(pm.1, sha256(twin.image), "{}", twin.module);
            assert_eq!(pm.4, hip.4, "{}: symbols changed", twin.module);
        }
        let xbatch = off
            .remove("gemv_mq4g256v2_xbatch")
            .expect("hipcc x-batch fallback not in the pm_decode=false route");
        assert_eq!(xbatch.0, "hip");
        assert!(
            !on.contains_key("gemv_mq4g256v2_xbatch"),
            "pm_decode route keeps the hipcc x-batch fallback"
        );
        for (module, digest, symbol) in PM_XBATCH_OBJECTS {
            let pm = on
                .remove(module)
                .unwrap_or_else(|| panic!("{module} not in the pm_decode route"));
            assert_eq!(pm.0, "embedded", "{module}");
            assert_eq!(
                pm.1, digest,
                "{module}: embedded image is not the accepted object"
            );
            assert_eq!(pm.4, BTreeSet::from([symbol.to_owned()]), "{module}");
        }
        assert_eq!(off, on, "{spec:?}: pm_decode changed a non-twin entry");
    }
    let mut other = builtin_flags("gfx1151");
    other.pm_decode = true;
    let trunk_only = metas(&[&trunk]);
    let refused = plan(&Load {
        arch: "gfx1151",
        model_arch: trunk.arch_id,
        tensors: &trunk_only,
        heads: (24, 4, 256),
        kv: ("fp8", "fp8"),
        spec: KernelSpecRoute::Ar,
        flags: &other,
    });
    assert!(refused.is_err(), "gfx1151 has no closed 27B route");
}
