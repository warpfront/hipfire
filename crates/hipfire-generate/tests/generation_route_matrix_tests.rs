// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generation route matrix tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

/// Baseline inputs that select nothing special (unknown arch, no EP/PP/spec).
fn base() -> GenerationRouteInputs {
    GenerationRouteInputs {
        dense_tp: false,
        arch_id: 255,
        ep: false,
        pp: 1,
        has_speculator: false,
        speculator_is_mtp: false,
        speculator_is_dflash2: false,
        deepseek4_spec_requested: false,
        ngram_can_sample: false,
        temp: 0.0,
        user_explicit_sampling: false,
        min_p: None,
        nonneutral_penalties: false,
        force_ar_chat: false,
        temp_spec_env_off: false,
        fast_sample_on: true,
        supports_temp_swor: false,
        supports_chain_nucleus_verify: false,
        kv_adaptive: false,
    }
}

#[test]
fn dspark_request_is_independent_of_mtp_mode() {
    // DSpark short-circuits before any MTP verdict, whatever the declared floor.
    for floor in [false, true] {
        assert!(deepseek4_spec_requested_from_policy(
            Some("dspark"),
            "off",
            "off",
            false,
            floor,
        ));
    }
    assert!(!deepseek4_spec_requested_from_policy(
        None, "off", "auto", true, true,
    ));
    assert!(deepseek4_spec_requested_from_policy(
        None, "auto", "auto", true, true,
    ));
}

#[test]
fn deepseek4_native_mtp_auto_without_a_measured_floor_stays_ar_and_explicit_on_still_arms() {
    // DeepSeek V4's native MTP declares no measured AR floor.
    let floor = hipfire_arch_deepseek4::MTP_MEASURED_AR_FLOOR;
    assert!(!floor);
    // `auto` with the weights present (a loaded head or a DSpark sidecar's
    // presence flag) must not rearm without the capability.
    assert!(!deepseek4_spec_requested_from_policy(
        None, "auto", "auto", true, floor,
    ));
    // No weights: `auto` is AR either way.
    assert!(!deepseek4_spec_requested_from_policy(
        None, "auto", "auto", false, true,
    ));
    // A declared floor admits `auto` (weights present) exactly as before.
    assert!(deepseek4_spec_requested_from_policy(
        None, "auto", "auto", true, true,
    ));
    // Explicit `on` (model or process) still requests MTP without the floor.
    assert!(deepseek4_spec_requested_from_policy(
        None, "auto", "on", false, floor,
    ));
    assert!(deepseek4_spec_requested_from_policy(
        None, "on", "auto", false, floor,
    ));
    // Explicit `off` never does, floor or not.
    for floor in [false, true] {
        assert!(!deepseek4_spec_requested_from_policy(
            None, "off", "on", true, floor,
        ));
        assert!(!deepseek4_spec_requested_from_policy(
            None, "auto", "off", true, floor,
        ));
    }
}

/// One canonical input row that selects each ALL variant (coverage guard).
/// New enum variants must add a row here or `route_capability_table_covers_all_variants` fails.
fn capability_rows() -> Vec<(GenerationRoute, GenerationRouteInputs)> {
    vec![
        (
            GenerationRoute::QwenAr,
            GenerationRouteInputs {
                arch_id: 5,
                ..base()
            },
        ),
        (
            GenerationRoute::Qwen4Ar,
            GenerationRouteInputs {
                arch_id: 16,
                ..base()
            },
        ),
        (
            GenerationRoute::Qwen4Spec,
            GenerationRouteInputs {
                arch_id: 16,
                has_speculator: true,
                speculator_is_mtp: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::QwenDflash,
            GenerationRouteInputs {
                arch_id: 5,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::Qwen2Ar,
            GenerationRouteInputs {
                arch_id: 7,
                ..base()
            },
        ),
        (
            GenerationRoute::Qwen2Spec,
            GenerationRouteInputs {
                arch_id: 7,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::Deepseek4Ar,
            GenerationRouteInputs {
                arch_id: 9,
                ..base()
            },
        ),
        (
            GenerationRoute::Deepseek4Ep,
            GenerationRouteInputs {
                arch_id: 9,
                ep: true,
                // EP beats DS4 arch short-circuit even with spec flags set.
                has_speculator: true,
                deepseek4_spec_requested: true,
                ..base()
            },
        ),
        (
            GenerationRoute::Deepseek4Spec,
            GenerationRouteInputs {
                arch_id: 9,
                has_speculator: true,
                deepseek4_spec_requested: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::CohereAr,
            GenerationRouteInputs {
                arch_id: 12,
                ..base()
            },
        ),
        (
            GenerationRoute::CohereSpec,
            GenerationRouteInputs {
                arch_id: 12,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            // Maple has no spec variant. `has_speculator: true` is set
            // deliberately: arch 15 must route to MapleAr even when the
            // carrier built a drafter, because there is no maple verify
            // path. A row with has_speculator false would not test that.
            GenerationRoute::MapleAr,
            GenerationRouteInputs {
                arch_id: 15,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            // Gemma 4 has one route. `has_speculator: true` is deliberate: an
            // attached EAGLE drafter must not move a turn off `gemma4_ar`
            // (the developer-gated spec branch is chosen inside the producer).
            GenerationRoute::Gemma4Ar,
            GenerationRouteInputs {
                arch_id: 13,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::MiniMaxAr,
            GenerationRouteInputs {
                arch_id: 10,
                ..base()
            },
        ),
        (
            GenerationRoute::MiniMaxEp,
            GenerationRouteInputs {
                arch_id: 10,
                ep: true,
                has_speculator: true,
                ..base()
            },
        ),
        (
            GenerationRoute::MiniMaxSpec,
            GenerationRouteInputs {
                arch_id: 10,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::LfmAr,
            GenerationRouteInputs {
                arch_id: 11,
                ..base()
            },
        ),
        (
            GenerationRoute::LfmSpec,
            GenerationRouteInputs {
                arch_id: 11,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::LlamaAr,
            GenerationRouteInputs {
                arch_id: 0,
                ..base()
            },
        ),
        (
            GenerationRoute::LlamaSpec,
            GenerationRouteInputs {
                arch_id: 0,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::PipelineParallel,
            GenerationRouteInputs {
                arch_id: 5,
                pp: 2,
                // PP still beats spec when no arch short-circuit.
                has_speculator: true,
                ..base()
            },
        ),
        (
            GenerationRoute::DotsOcr,
            GenerationRouteInputs {
                arch_id: 8,
                ..base()
            },
        ),
        (
            GenerationRoute::GlimmerAr,
            GenerationRouteInputs {
                arch_id: 14,
                ..base()
            },
        ),
        (
            GenerationRoute::GlimmerSpec,
            GenerationRouteInputs {
                arch_id: 14,
                has_speculator: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::RaonAr,
            GenerationRouteInputs {
                arch_id: 19,
                ..base()
            },
        ),
        (
            // Raon with the shared DFlash2 drafter loaded: a greedy request
            // takes the spec route (an unrelated speculator would not).
            GenerationRoute::RaonSpec,
            GenerationRouteInputs {
                arch_id: 19,
                has_speculator: true,
                speculator_is_dflash2: true,
                temp: 0.0,
                ..base()
            },
        ),
        (
            GenerationRoute::Unknown,
            GenerationRouteInputs {
                arch_id: 99,
                ..base()
            },
        ),
    ]
}

/// Exact proven-safe producer set (contract).
const SAFE_ROUTES: &[GenerationRoute] = &[
    GenerationRoute::QwenAr,
    GenerationRoute::Qwen4Ar,
    // Native Qwen4 MTP: same `<tool_call>` router as Qwen4 AR over committed
    // tokens, tool grammar forced off, so greedy tools match AR byte-for-byte.
    GenerationRoute::Qwen4Spec,
    GenerationRoute::QwenDflash,
    GenerationRoute::Deepseek4Ar,
    GenerationRoute::Deepseek4Ep,
    GenerationRoute::Deepseek4Spec,
    GenerationRoute::GlimmerAr,
    GenerationRoute::GlimmerSpec,
    // Arch 15. Tool-safe on the LEGACY wire contract — its carrier keeps
    // `semantic_contract_version: None` (no router-backed producer, and the v2
    // fold would misfile Maple's `<think>` span as content). `generate_maple`
    // emits a `{"type":"tool_calls"}` event plus a `finish_reason=tool_calls`
    // terminal, parsing calls with the same Qwen `<tool_call>` parser as
    // `qwen_ar` because Maple's vendor template emits the identical shape.
    GenerationRoute::MapleAr,
    // Arch 13, legacy contract like Maple. `generate_gemma4` renders tools
    // through the embedded template, withholds the model's native
    // `<|tool_call>call:NAME{ARGS}<tool_call|>` blocks from content and stages
    // the parsed calls on a `finish_reason=tool_calls` commit_ready terminal.
    GenerationRoute::Gemma4Ar,
    // Arch 19 (Raon-1-mini). Shares Qwen4 AR's `generate_ar_with_forward`
    // producer: XML `<tool_call><function=..>` through the strict
    // `ToolOutputRouter`, semantic contract v2 via the carrier caps.
    GenerationRoute::RaonAr,
    // Arch 19 with the shared DFlash2 drafter: the same router over committed
    // tokens (`Qwen35Emit` via the Raon carrier, grammar off).
    GenerationRoute::RaonSpec,
];

/// Pure gate model mirroring generate()'s tools preflight:
/// deny before RNG/gen_start when tools nonempty && !supports_tools.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GateOutcome {
    allowed: bool,
    error_count: usize,
    class: Option<&'static str>,
    retryable: Option<bool>,
    mutated_generation_side: bool,
    route: GenerationRoute,
}

fn pure_tools_gate(route: GenerationRoute, tools_nonempty: bool) -> GateOutcome {
    if tools_nonempty && !route.supports_tools() {
        GateOutcome {
            allowed: false,
            error_count: 1,
            class: Some("unsupported"),
            retryable: Some(false),
            mutated_generation_side: false,
            route,
        }
    } else {
        GateOutcome {
            allowed: true,
            error_count: 0,
            class: None,
            retryable: None,
            mutated_generation_side: false,
            route,
        }
    }
}

#[test]
fn route_capability_table_covers_all_variants() {
    let rows = capability_rows();
    assert_eq!(
        rows.len(),
        GenerationRoute::ALL.len(),
        "capability table must list every GenerationRoute::ALL variant"
    );
    for &variant in GenerationRoute::ALL {
        let hit = rows.iter().any(|(r, _)| *r == variant);
        assert!(
            hit,
            "missing capability row for {:?}; add an explicit selector input",
            variant
        );
    }
    // Each row's selector must actually produce the labeled route.
    for (expected, inputs) in &rows {
        let got = select_generation_route(inputs);
        assert_eq!(
            got, *expected,
            "capability row for {:?} selected {:?}",
            expected, got
        );
    }
}

/// `generate` falls through to its generic AR body on `QwenAr` and, when
/// `generate_dflash` declines a request over the drafter's ctx capacity, on
/// `QwenDflash`; both need that body's `Qwen35Bundle` branch. No other route
/// may take it, only the Qwen3.5 carrier (arch 5|6) selects the two, and
/// arch 17 (the embedding encoder, GatedDeltaNet state but no Qwen DFlash
/// caps) must not.
#[test]
fn qwen35_ar_body_is_taken_by_qwen_ar_and_a_declined_qwen_dflash() {
    for &route in GenerationRoute::ALL {
        let expected = matches!(route, GenerationRoute::QwenAr | GenerationRoute::QwenDflash);
        assert_eq!(route.runs_qwen35_ar_body(), expected, "{route:?}");
    }
    for (route, inputs) in capability_rows() {
        if route.runs_qwen35_ar_body() {
            assert!(
                matches!(inputs.arch_id, 5 | 6),
                "{route:?} selected for arch {}",
                inputs.arch_id
            );
        }
    }
    for has_speculator in [false, true] {
        let embed = GenerationRouteInputs {
            arch_id: 17,
            has_speculator,
            ..base()
        };
        assert!(!select_generation_route(&embed).runs_qwen35_ar_body());
    }
}

#[test]
fn route_matrix_tools_absent_and_present() {
    for (route, inputs) in capability_rows() {
        let selected = select_generation_route(&inputs);
        assert_eq!(selected, route);

        let safe = SAFE_ROUTES.contains(&route);
        assert_eq!(
            route.supports_tools(),
            safe,
            "{:?} supports_tools mismatch vs SAFE_ROUTES",
            route
        );

        // Tools absent: always allowed, zero errors, no mutation.
        let absent = pure_tools_gate(route, false);
        assert!(absent.allowed, "{:?} tools-absent must allow", route);
        assert_eq!(absent.error_count, 0);
        assert!(absent.class.is_none());
        assert!(!absent.mutated_generation_side);

        // Tools present: safe allows; unsafe emits exactly one nonretryable unsupported.
        let present = pure_tools_gate(route, true);
        if safe {
            assert!(present.allowed, "{:?} safe+tools must allow", route);
            assert_eq!(present.error_count, 0);
            assert!(!present.mutated_generation_side);
        } else {
            assert!(!present.allowed, "{:?} unsafe+tools must deny", route);
            assert_eq!(present.error_count, 1, "{:?} exactly one error", route);
            assert_eq!(present.class, Some("unsupported"));
            assert_eq!(present.retryable, Some(false));
            assert!(
                !present.mutated_generation_side,
                "{:?} deny must not mutate generation side",
                route
            );
        }
    }
}

#[test]
fn qwen4_native_mtp_route_requires_explicit_greedy_request() {
    let mtp = GenerationRouteInputs {
        arch_id: 16,
        has_speculator: true,
        speculator_is_mtp: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&mtp), GenerationRoute::Qwen4Spec);

    // Greedy argmax ignores top_p/top_k/min_p, and serve forwards top_p/top_k
    // whenever the client or the registry sets one, so their presence must
    // keep a greedy request on native MTP.
    for greedy in [
        GenerationRouteInputs {
            user_explicit_sampling: true,
            ..mtp
        },
        GenerationRouteInputs {
            min_p: Some(0.1),
            ..mtp
        },
    ] {
        assert_eq!(
            select_generation_route(&greedy),
            GenerationRoute::Qwen4Spec,
            "greedy Qwen4 request with argmax-neutral sampler fields must use native MTP: {greedy:?}"
        );
    }

    for refused in [
        GenerationRouteInputs { temp: 0.7, ..mtp },
        GenerationRouteInputs {
            temp: 0.7,
            user_explicit_sampling: true,
            ..mtp
        },
        GenerationRouteInputs {
            nonneutral_penalties: true,
            ..mtp
        },
        GenerationRouteInputs {
            force_ar_chat: true,
            ..mtp
        },
        GenerationRouteInputs {
            temp_spec_env_off: true,
            ..mtp
        },
        GenerationRouteInputs {
            kv_adaptive: true,
            ..mtp
        },
        GenerationRouteInputs {
            speculator_is_mtp: false,
            ..mtp
        },
        GenerationRouteInputs {
            has_speculator: false,
            ..mtp
        },
    ] {
        assert_eq!(
            select_generation_route(&refused),
            GenerationRoute::Qwen4Ar,
            "Qwen4 native MTP must refuse non-greedy or non-explicit inputs: {refused:?}"
        );
    }
}

#[test]
fn qwen4_sampled_mtp_route_honors_penalties() {
    let sampled = GenerationRouteInputs {
        arch_id: 16,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: true,
        temp: 0.7,
        user_explicit_sampling: true,
        ..base()
    };
    assert_eq!(
        select_generation_route(&sampled),
        GenerationRoute::Qwen4Spec
    );
    // Sampled MTP honors repeat/presence/frequency penalties: the verifier's
    // target applies the same host policy as the AR sampler.
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            nonneutral_penalties: true,
            ..sampled
        }),
        GenerationRoute::Qwen4Spec,
        "sampled Qwen4 MTP with penalties must stay on native MTP"
    );
    // The verifier's target applies min_p as the AR sampler does, so it does
    // not demote.
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            min_p: Some(0.05),
            ..sampled
        }),
        GenerationRoute::Qwen4Spec
    );
    // Every other refusal keeps the request on AR, with or without penalties.
    for penalties in [false, true] {
        for refused in [
            GenerationRouteInputs {
                supports_temp_swor: false,
                nonneutral_penalties: penalties,
                ..sampled
            },
            GenerationRouteInputs {
                force_ar_chat: true,
                nonneutral_penalties: penalties,
                ..sampled
            },
            GenerationRouteInputs {
                temp_spec_env_off: true,
                nonneutral_penalties: penalties,
                ..sampled
            },
            GenerationRouteInputs {
                kv_adaptive: true,
                nonneutral_penalties: penalties,
                ..sampled
            },
        ] {
            assert_eq!(
                select_generation_route(&refused),
                GenerationRoute::Qwen4Ar,
                "sampled Qwen4 request must stay on AR: {refused:?}"
            );
        }
    }
}

/// Qwen4 pipeline parallelism: sampled, unpenalized requests take native MTP
/// (Leviathan verify, distribution-exact) exactly as on one card. The kill
/// switch (`speculation.mtp_sampled = 0` → `supports_temp_swor` false, and
/// the naive sampled mode under PP, which the route input also reports as
/// not sampled-capable) and any penalty keep the PP AR producer.
#[test]
fn qwen4_pp_sampled_mtp_route_admits_unpenalized_leviathan_only() {
    for pp in [2usize, 4, 8] {
        let sampled = GenerationRouteInputs {
            arch_id: 16,
            pp,
            has_speculator: true,
            speculator_is_mtp: true,
            supports_temp_swor: true,
            temp: 0.7,
            ..base()
        };
        for admitted in [
            sampled,
            GenerationRouteInputs {
                user_explicit_sampling: true,
                min_p: Some(0.05),
                ..sampled
            },
            GenerationRouteInputs {
                temp: 0.0,
                ..sampled
            },
        ] {
            assert_eq!(
                select_generation_route(&admitted),
                GenerationRoute::Qwen4Spec,
                "pp {pp}: {admitted:?}"
            );
        }
        for refused in [
            // HIPFIRE_MTP_SAMPLED=0 / PP naive mode.
            GenerationRouteInputs {
                supports_temp_swor: false,
                ..sampled
            },
            GenerationRouteInputs {
                nonneutral_penalties: true,
                ..sampled
            },
            GenerationRouteInputs {
                nonneutral_penalties: true,
                temp: 0.0,
                ..sampled
            },
            GenerationRouteInputs {
                force_ar_chat: true,
                ..sampled
            },
            GenerationRouteInputs {
                temp_spec_env_off: true,
                ..sampled
            },
            GenerationRouteInputs {
                kv_adaptive: true,
                ..sampled
            },
        ] {
            assert_eq!(
                select_generation_route(&refused),
                GenerationRoute::Qwen4Ar,
                "pp {pp}: {refused:?}"
            );
        }
    }
}

#[test]
fn qwen35_mtp_route_honors_penalties_but_dflash_stays_ar() {
    // --- Arch 5 native MTP: penalties never demote a servable request. ---
    let mtp = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: true,
        nonneutral_penalties: true,
        ..base()
    };
    let sampled = GenerationRouteInputs {
        temp: 0.7,
        user_explicit_sampling: true,
        ..mtp
    };
    assert_eq!(
        select_generation_route(&sampled),
        GenerationRoute::QwenDflash,
        "sampled Qwen3.x MTP + penalties"
    );
    let greedy = GenerationRouteInputs { temp: 0.0, ..mtp };
    assert_eq!(
        select_generation_route(&greedy),
        GenerationRoute::QwenDflash,
        "greedy Qwen3.x MTP + penalties"
    );
    // Dense-TP EP carries the MTP drafter and honors penalties too.
    for temp in [0.7_f32, 0.0] {
        let dense = GenerationRouteInputs {
            ep: true,
            dense_tp: true,
            temp,
            ..mtp
        };
        assert_eq!(
            select_generation_route(&dense),
            GenerationRoute::QwenDflash,
            "dense-TP EP MTP + penalties at temp {temp}"
        );
    }
    for temp in [0.7_f32, 0.0] {
        let forced = GenerationRouteInputs {
            force_ar_chat: true,
            temp,
            ..mtp
        };
        assert_eq!(
            select_generation_route(&forced),
            GenerationRoute::QwenAr,
            "force_ar_chat + penalties must stay AR at temp {temp}"
        );
    }

    // --- Arch 5 DFlash (not MTP): penalties keep the request on AR. ---
    let dflash = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: false,
        ..base()
    };
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            temp: 0.0,
            ..dflash
        }),
        GenerationRoute::QwenDflash,
        "greedy DFlash without penalties"
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            temp: 0.0,
            nonneutral_penalties: true,
            ..dflash
        }),
        GenerationRoute::QwenAr,
        "greedy DFlash + penalties"
    );
    // DDTree SWOR (temp>0, supports_temp_swor, no chain nucleus, no
    // user-explicit sampling).
    let ddtree = GenerationRouteInputs {
        supports_temp_swor: true,
        supports_chain_nucleus_verify: false,
        user_explicit_sampling: false,
        temp: 0.7,
        ..dflash
    };
    assert_eq!(
        select_generation_route(&ddtree),
        GenerationRoute::QwenDflash,
        "DDTree SWOR without penalties"
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            nonneutral_penalties: true,
            ..ddtree
        }),
        GenerationRoute::QwenAr,
        "DDTree SWOR + penalties"
    );
}

#[test]
fn llama_dflash_greedy_route_refuses_penalties() {
    // Arch 1 is a llama-DFlash carrier (caps().is_llama_dflash()).
    let greedy = GenerationRouteInputs {
        arch_id: 1,
        has_speculator: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&greedy), GenerationRoute::LlamaSpec);
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            nonneutral_penalties: true,
            ..greedy
        }),
        GenerationRoute::LlamaAr
    );
}

#[test]
fn qwen_cache_planner_is_ineligible_cold() {
    let plan = hipfire_generate::qwen::plan_from_rendered(
        &[10, 11],
        vec![10, 11, 12],
        false,
        &[],
        false,
        "mtp",
    );
    assert!(!plan.cache_hit);
    assert_eq!(plan.start_pos, 0);
    assert_eq!(plan.cached_tokens, 0);
    assert_eq!(plan.resume_from, None);
    assert_eq!(plan.new_tokens, vec![10, 11, 12]);
}

/// Native Qwen4 MTP never uses the shared conversation planner, including on a
/// transition from an AR turn: its reusable prefixes are the bundle's session
/// cache (`ArchModel::session_plan` with `SessionRoute::Mtp`), planned
/// separately.
#[test]
fn qwen4_mtp_cache_planner_forces_cold_after_ar_transition() {
    // Native Qwen4 MTP bypasses the shared planner.
    assert!(hipfire_generate::qwen::spec_cache_disabled_for(true, false));
    // Qwen3.5/3.6/3.8 MTP and DFlash keep the prompt cache: the rule is keyed
    // on the loaded family, not on the speculator name "mtp".
    assert!(!hipfire_generate::qwen::spec_cache_disabled_for(
        false, false
    ));
    // HIPFIRE_QWEN_PROMPT_CACHE=0 still disables it for every family.
    assert!(hipfire_generate::qwen::spec_cache_disabled_for(false, true));
}

/// Asserts every `PromptCachePlan` field against the expected start and resume.
/// `start == 0` means a cold miss.
fn assert_plan(
    plan: &hipfire_generate::qwen::PromptCachePlan,
    rendered: &[u32],
    start: usize,
    resume: Option<usize>,
    what: &str,
) {
    assert_eq!(plan.rendered, rendered, "{what}: rendered");
    assert_eq!(plan.start_pos, start, "{what}: start_pos");
    assert_eq!(plan.cached_tokens, start, "{what}: cached_tokens");
    assert_eq!(plan.cache_hit, start > 0, "{what}: cache_hit");
    assert_eq!(plan.resume_from, resume, "{what}: resume_from");
    assert_eq!(
        plan.new_tokens,
        rendered[start..].to_vec(),
        "{what}: new_tokens"
    );
}

/// Shared planner (Qwen3.5-family AR/DFlash): lineage = the committed live
/// tokens, one exact end-of-prompt checkpoint at an arbitrary
/// (non-chunk-aligned) position `p`.
#[test]
fn eop_checkpoint_plans_use_the_shared_planner() {
    use hipfire_generate::qwen::plan_from_rendered as plan;
    // Distinct divergent suffix tokens never collide with the 0..n lineage.
    let fork = |keep: usize, extra: usize| -> Vec<u32> {
        (0..keep as u32)
            .chain((0..extra as u32).map(|i| 1_000_000 + i))
            .collect()
    };
    for (p, l) in [(37usize, 50usize), (1903, 2100)] {
        let lineage: Vec<u32> = (0..l as u32).collect();
        let ckpts = [p];

        // (a) live extension: the lineage is a strict prefix of the request.
        let rendered: Vec<u32> = (0..(l + 20) as u32).collect();
        let got = plan(&lineage, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &rendered, l, None, "live extension");

        // (b) divergence with p <= lcp < l: rewind to the checkpoint.
        let rendered = fork(p + 8, 3);
        let got = plan(&lineage, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(
            &got,
            &rendered,
            p,
            Some(p),
            "divergence past the checkpoint",
        );

        // (c) divergence exactly at the checkpoint.
        let rendered = fork(p, 2);
        let got = plan(&lineage, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &rendered, p, Some(p), "divergence at the checkpoint");

        // (d) lcp below the checkpoint: cold.
        let rendered = fork(p - 17, 4);
        let got = plan(&lineage, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &rendered, 0, None, "lcp below the checkpoint");

        // (e) exact match: resume from the checkpoint when it leaves a tail.
        let got = plan(&lineage, lineage.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &lineage, p, Some(p), "exact match, p < n");
        // Exact match whose length equals the checkpoint leaves nothing to
        // replay: cold.
        let at_p: Vec<u32> = (0..p as u32).collect();
        let got = plan(&at_p, at_p.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &at_p, 0, None, "exact match, p == n");

        // (f) shorter request that still covers the checkpoint.
        let rendered: Vec<u32> = (0..(p + 7) as u32).collect();
        let got = plan(&lineage, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &rendered, p, Some(p), "shorter request, lcp >= p");
        // A shorter request ending exactly at the checkpoint has no tail: cold.
        let got = plan(&lineage, at_p.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &at_p, 0, None, "shorter request ending at p");

        // (g) checkpoint-only lineage extended: strict-extension branch, no resume.
        let rendered: Vec<u32> = (0..(p + 23) as u32).collect();
        let got = plan(&at_p, rendered.clone(), true, &ckpts, true, "t");
        assert_plan(&got, &rendered, p, None, "checkpoint-only lineage extended");

        // (h) cache ineligible: cold even on a divergence that would resume.
        let rendered = fork(p + 8, 3);
        let got = plan(&lineage, rendered.clone(), false, &ckpts, true, "t");
        assert_plan(&got, &rendered, 0, None, "cache ineligible");

        // (i) resume disabled: divergence is cold, a pure extension still hits.
        let got = plan(&lineage, rendered.clone(), true, &ckpts, false, "t");
        assert_plan(&got, &rendered, 0, None, "resume disabled, divergence");
        let ext: Vec<u32> = (0..(l + 20) as u32).collect();
        let got = plan(&lineage, ext.clone(), true, &ckpts, false, "t");
        assert_plan(&got, &ext, l, None, "resume disabled, extension");
    }
}

// Qwen4 does not plan through the shared conversation planner: its AR and
// native-MTP prefixes come from the bundle's session cache
// (`ArchModel::session_plan` with `SessionRoute::{Ar,Mtp}`), which returns a
// bare `usize` start and never continues a live session. The tests below pin
// the shared `plan_from_rendered` planner that the Qwen3.5-family AR/DFlash
// routes still use.

/// Inputs reach the shared planner unchanged: the local Live/EOP candidate is
/// `plan.start_pos`. An empty lineage (no live record, no EOP record) is a
/// plain miss.
#[test]
fn empty_lineage_is_a_local_miss() {
    use hipfire_generate::qwen::plan_from_rendered as plan;
    let rendered: Vec<u32> = (0..64).collect();
    for ckpts in [&[][..], &[16usize][..], &[16usize, 40][..]] {
        for resume in [false, true] {
            let got = plan(&[], rendered.clone(), true, ckpts, resume, "q35");
            assert_plan(&got, &rendered, 0, None, "empty lineage");
        }
    }
    // Live record plus an EOP checkpoint: the candidate stays the lineage's
    // own deepest hit.
    let live: Vec<u32> = (0..40).collect();
    let got = plan(&live, rendered.clone(), true, &[16], true, "q35");
    assert_plan(&got, &rendered, 40, None, "live candidate");
    // EOP-only lineage (checkpoint-length record) yields the EOP candidate.
    let eop: Vec<u32> = (0..16).collect();
    let got = plan(&eop, rendered.clone(), true, &[16], true, "q35");
    assert_plan(&got, &rendered, 16, None, "EOP candidate");
}

/// A candidate equal to the prompt length leaves no suffix to prefill, so it is
/// never a hit: every plan keeps a nonempty `new_tokens` tail and
/// `start_pos < rendered.len()`.
#[test]
fn local_candidate_never_covers_the_whole_prompt() {
    use hipfire_generate::qwen::plan_from_rendered as plan;
    let seq = |n: usize| -> Vec<u32> { (0..n as u32).collect() };
    for n in 1..=24usize {
        let rendered = seq(n);
        let mut lineages = vec![Vec::new(), seq(n), seq(n + 3)];
        for keep in 0..=n {
            // Shared prefix of `keep` tokens, then diverging ids.
            lineages.push(
                seq(keep)
                    .into_iter()
                    .chain([900_000, 900_001])
                    .collect::<Vec<u32>>(),
            );
            lineages.push(seq(keep));
        }
        for lineage in &lineages {
            for ckpts in [
                vec![],
                vec![n],
                vec![n + 1],
                vec![(n - 1).max(1)],
                vec![1, (n / 2).max(1), n],
            ] {
                for (eligible, resume) in [(true, true), (true, false), (false, true)] {
                    let ctx = format!(
                        "n={n} lineage_len={} ckpts={ckpts:?} eligible={eligible} resume={resume}",
                        lineage.len()
                    );
                    let got = plan(lineage, rendered.clone(), eligible, &ckpts, resume, "q35");
                    assert!(got.start_pos < n, "{ctx}: start_pos {} >= n", got.start_pos);
                    assert!(!got.new_tokens.is_empty(), "{ctx}: empty suffix");
                    assert_eq!(got.new_tokens, rendered[got.start_pos..].to_vec(), "{ctx}");
                    assert_eq!(got.cached_tokens, got.start_pos, "{ctx}");
                    assert_eq!(got.cache_hit, got.start_pos > 0, "{ctx}");
                    assert_eq!(got.rendered, rendered, "{ctx}");
                }
            }
        }
    }
}

/// A disabled cache (`HIPFIRE_QWEN_PROMPT_CACHE=0` or eviction active) reaches
/// the shared planner as `cache_eligible = false`: cold even where a live
/// extension or checkpoint resume would hit.
#[test]
fn cache_disabled_inputs_plan_cold_through_the_shared_planner() {
    use hipfire_generate::qwen::plan_from_rendered as plan;
    let lineage: Vec<u32> = (0..50).collect();
    let extension: Vec<u32> = (0..70).collect();
    let divergent: Vec<u32> = (0..45).chain(1_000_000..1_000_003).collect();
    for rendered in [&extension, &divergent] {
        let got = plan(&lineage, rendered.clone(), false, &[37], true, "q35");
        assert_plan(&got, rendered, 0, None, "cache disabled");
    }
    // The same inputs with the cache enabled hit, so the cold plans above are
    // due to eligibility alone.
    let got = plan(&lineage, extension.clone(), true, &[37], true, "q35");
    assert_plan(&got, &extension, 50, None, "cache enabled extension");
    let got = plan(&lineage, divergent.clone(), true, &[37], true, "q35");
    assert_plan(&got, &divergent, 37, Some(37), "cache enabled divergence");
}

/// Route isolation: only the native MTP speculator and the DFlash2 drafter
/// select the Qwen4 spec route; an n-gram (non-MTP, non-DFlash2) speculator
/// on arch 16 stays on Qwen4 AR even when it can sample. Both routes plan
/// their prefix through the bundle's session cache (`SessionRoute::Ar` /
/// `SessionRoute::Mtp`), never through the shared conversation planner.
#[test]
fn non_mtp_speculators_on_qwen4_stay_on_ar() {
    for ngram_can_sample in [false, true] {
        let ngram = GenerationRouteInputs {
            arch_id: 16,
            has_speculator: true,
            speculator_is_mtp: false,
            ngram_can_sample,
            ..base()
        };
        assert_eq!(
            select_generation_route(&ngram),
            GenerationRoute::Qwen4Ar,
            "{ngram:?}"
        );
    }
}

/// DFlash2 on Qwen4: greedy and sampled requests take the spec route on one
/// card; any token penalty (DFlash2 applies none), pipeline parallelism,
/// adaptive KV and the force-AR switches keep the AR producer.
#[test]
fn qwen4_dflash2_takes_the_spec_route_single_card_without_penalties() {
    let dflash2 = GenerationRouteInputs {
        arch_id: 16,
        has_speculator: true,
        speculator_is_dflash2: true,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: true,
        ..base()
    };
    for temp in [0.0f32, 1.0] {
        let request = GenerationRouteInputs { temp, ..dflash2 };
        assert_eq!(
            select_generation_route(&request),
            GenerationRoute::Qwen4Spec,
            "{request:?}"
        );
        for refused in [
            GenerationRouteInputs {
                nonneutral_penalties: true,
                ..request
            },
            GenerationRouteInputs { pp: 2, ..request },
            GenerationRouteInputs {
                kv_adaptive: true,
                ..request
            },
            GenerationRouteInputs {
                force_ar_chat: true,
                ..request
            },
            GenerationRouteInputs {
                temp_spec_env_off: true,
                ..request
            },
            GenerationRouteInputs {
                has_speculator: false,
                ..request
            },
        ] {
            assert_eq!(
                select_generation_route(&refused),
                GenerationRoute::Qwen4Ar,
                "{refused:?}"
            );
        }
    }
    // A sampled request needs a drafter that verifies at temperature.
    let greedy_only = GenerationRouteInputs {
        temp: 1.0,
        supports_temp_swor: false,
        ..dflash2
    };
    assert_eq!(
        select_generation_route(&greedy_only),
        GenerationRoute::Qwen4Ar
    );
}

/// Native MTP stays the native Qwen4 route whether or not an n-gram-style
/// sampling capability flag is present, and AR routes remain independent of it.
#[test]
fn qwen4_native_mtp_with_ngram_flag_remains_the_native_route() {
    let mtp = GenerationRouteInputs {
        arch_id: 16,
        has_speculator: true,
        speculator_is_mtp: true,
        ..base()
    };
    for ngram_can_sample in [false, true] {
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                ngram_can_sample,
                ..mtp
            }),
            GenerationRoute::Qwen4Spec,
            "ngram_can_sample={ngram_can_sample}"
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                has_speculator: false,
                speculator_is_mtp: false,
                ngram_can_sample,
                ..mtp
            }),
            GenerationRoute::Qwen4Ar,
            "ngram_can_sample={ngram_can_sample} without speculator"
        );
    }
}

#[test]
fn exact_safe_set_is_qwen_ar_qwen4_ar_mtp_dflash_ds4_ar_ep_spec_glimmer_ar_spec_maple_ar_and_gemma4_ar(
) {
    let mut from_all: Vec<GenerationRoute> = GenerationRoute::ALL
        .iter()
        .copied()
        .filter(|r| r.supports_tools())
        .collect();
    from_all.sort_by_key(|r| r.name());
    let mut expected = SAFE_ROUTES.to_vec();
    expected.sort_by_key(|r| r.name());
    assert_eq!(from_all, expected);
    assert_eq!(from_all.len(), 13);
    // Negative: every other ALL member is denied for tools.
    for &r in GenerationRoute::ALL {
        if !SAFE_ROUTES.contains(&r) {
            assert!(!r.supports_tools(), "{:?} must not be tool-safe", r);
        }
    }
}

/// Gemma 4 (arch 13) used to fall through every arch arm to `unknown`, so a
/// tools request was refused as "tools are not supported on producer route
/// unknown". It now has its own named, tool-capable route whatever the
/// sampling or drafter state; only EP (no Gemma EP producer) stays unknown.
#[test]
fn gemma4_is_classified_and_routes_tools() {
    for (temp, has_speculator, nonneutral_penalties) in [
        (0.0, false, false),
        (0.0, true, false),
        (0.7, true, false),
        (0.7, false, true),
    ] {
        let route = select_generation_route(&GenerationRouteInputs {
            arch_id: 13,
            temp,
            has_speculator,
            nonneutral_penalties,
            ..base()
        });
        assert_eq!(route, GenerationRoute::Gemma4Ar);
        assert_eq!(route.name(), "gemma4_ar");
        assert!(pure_tools_gate(route, true).allowed);
        assert!(!route.supports_stop());
        let adapter = generation_route_adapter(route).expect("gemma4 adapter");
        assert_eq!(adapter.route, GenerationRoute::Gemma4Ar);
    }
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 13,
            ep: true,
            ..base()
        }),
        GenerationRoute::Unknown
    );
}

#[test]
fn stop_sequences_are_honoured_exactly_on_the_qwen_semantic_producers() {
    // Qwen4 AR and native MTP joined the Qwen3.5-family AR/DFlash routes: an
    // eval request with `stop` on Flash-Next used to be refused with 400.
    let mut with_stop: Vec<&str> = GenerationRoute::ALL
        .iter()
        .filter(|r| r.supports_stop())
        .map(|r| r.name())
        .collect();
    with_stop.sort_unstable();
    assert_eq!(
        with_stop,
        [
            "qwen4_ar",
            "qwen4_spec",
            "qwen_ar",
            "qwen_dflash",
            "raon_ar",
            "raon_spec"
        ]
    );
    // Both Qwen4 routes a Flash-Next load can select carry tools AND stop, so
    // attaching MTP never changes which request shapes are accepted.
    for route in [GenerationRoute::Qwen4Ar, GenerationRoute::Qwen4Spec] {
        assert!(route.supports_tools() && route.supports_stop(), "{route:?}");
    }
}

#[test]
fn dense_tp_qwen_with_mtp_takes_the_spec_route_only_when_it_can_serve() {
    // Dense TP (arch 5 under an EP-shaped topology) with an MTP drafter serves
    // greedy and sampled-verify requests on the spec route; anything it cannot
    // verify stays on the EP AR arm.
    let tp_mtp = GenerationRouteInputs {
        arch_id: 5,
        ep: true,
        dense_tp: true,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(
        select_generation_route(&tp_mtp),
        GenerationRoute::QwenDflash
    );
    let sampled = GenerationRouteInputs {
        temp: 0.7,
        ..tp_mtp
    };
    assert_eq!(
        select_generation_route(&sampled),
        GenerationRoute::QwenDflash
    );
    // A drafter that cannot verify sampled output keeps temp>0 on the AR arm.
    let greedy_only = GenerationRouteInputs {
        temp: 0.7,
        supports_temp_swor: false,
        ..tp_mtp
    };
    assert_eq!(
        select_generation_route(&greedy_only),
        GenerationRoute::QwenAr
    );
    // Forced AR chat, adaptive KV and non-MTP drafters never reach it.
    let forced_ar = GenerationRouteInputs {
        force_ar_chat: true,
        ..tp_mtp
    };
    assert_eq!(select_generation_route(&forced_ar), GenerationRoute::QwenAr);
    let adaptive = GenerationRouteInputs {
        kv_adaptive: true,
        ..tp_mtp
    };
    assert_eq!(select_generation_route(&adaptive), GenerationRoute::QwenAr);
    let ngram = GenerationRouteInputs {
        speculator_is_mtp: false,
        ..tp_mtp
    };
    assert_eq!(select_generation_route(&ngram), GenerationRoute::QwenAr);
    // The MoE EP topology keeps its EP route even with a drafter attached.
    let moe = GenerationRouteInputs {
        arch_id: 6,
        dense_tp: false,
        ..tp_mtp
    };
    assert_eq!(select_generation_route(&moe), GenerationRoute::QwenAr);
}

#[test]
fn precedence_ep_before_arch_short_circuit() {
    // EP on DS4 with spec requested → Deepseek4Ep, not Spec/Ar.
    let i = GenerationRouteInputs {
        arch_id: 9,
        ep: true,
        has_speculator: true,
        deepseek4_spec_requested: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ep);
    // EP on MiniMax with n-gram spec → MiniMaxEp, not Spec.
    let i = GenerationRouteInputs {
        arch_id: 10,
        ep: true,
        has_speculator: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxEp);
    // EP on an unregistered arch → Unknown (still EP-first).
    let i = GenerationRouteInputs {
        arch_id: 99,
        ep: true,
        has_speculator: true,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::Unknown);
}

#[test]
fn qwen_ep_dense_tp_selects_qwen_ar_semantic_contract() {
    for arch_id in [5, 6] {
        let with_ep = GenerationRouteInputs {
            arch_id,
            ep: true,
            ..base()
        };
        let route = select_generation_route(&with_ep);
        assert_eq!(route, GenerationRoute::QwenAr);

        let id = format!("qwen-ep-{arch_id}");
        let mut sink = Vec::new();
        generation_route_adapter(route)
            .expect("every selected route has an adapter")
            .emit_start(&mut sink, &id);
        let start: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(start["type"], "gen_start");
        assert_eq!(start["contract_version"], 2);
    }

    // The same Qwen route remains the ordinary AR route without EP.
    let without_ep = GenerationRouteInputs {
        arch_id: 5,
        ep: false,
        ..base()
    };
    assert_eq!(
        select_generation_route(&without_ep),
        GenerationRoute::QwenAr
    );
}

#[test]
fn precedence_arch_short_circuit_before_pp() {
    // Qwen2 + pp>1 still short-circuits to Qwen2, never PipelineParallel.
    let i = GenerationRouteInputs {
        arch_id: 7,
        pp: 4,
        has_speculator: false,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::Qwen2Ar);
    let i = GenerationRouteInputs {
        arch_id: 9,
        pp: 2,
        has_speculator: false,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ar);
    let i = GenerationRouteInputs {
        arch_id: 11,
        pp: 2,
        has_speculator: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::LfmSpec);
    let i = GenerationRouteInputs {
        arch_id: 12,
        pp: 2,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::CohereAr);
    let i = GenerationRouteInputs {
        arch_id: 10,
        pp: 2,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxAr);
    let i = GenerationRouteInputs {
        arch_id: 8,
        pp: 2,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::DotsOcr);
}

#[test]
fn precedence_pp_before_qwen_mtp() {
    let i = GenerationRouteInputs {
        arch_id: 5,
        pp: 2,
        temp: 0.0,
        has_speculator: true,
        ..base()
    };
    assert_eq!(
        select_generation_route(&i),
        GenerationRoute::PipelineParallel
    );
}

#[test]
fn mtp_speculator_routes_through_qwen_dflash() {
    // Greedy MTP uses the generic QwenDflash wrapper.
    let i = GenerationRouteInputs {
        arch_id: 6,
        has_speculator: true,
        speculator_is_mtp: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
    // Sampled MTP with user-explicit sampling and min_p stays on spec
    // when supports_temp_verify — unlike DFlash-specific restrictions.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: true,
        temp: 0.7,
        user_explicit_sampling: true,
        min_p: Some(0.05),
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
    // DDTree SWOR (supports_temp_swor, no chain nucleus) + user-explicit
    // non-temperature controls still falls to AR.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: false,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: false,
        temp: 0.7,
        user_explicit_sampling: true,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
    // MTP without supports_temp_verify at temp>0 falls to AR.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: false,
        temp: 0.7,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
}

#[test]
fn dflash2_selector_chain_nucleus_routes() {
    // Registry sampling profile: temp>0 + explicit top_p/top_k + min_p=0
    // with DFlash2 selector-chain nucleus → QwenDflash (not misclassified
    // as DDTree SWOR).
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: false,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: true,
        ngram_can_sample: true,
        fast_sample_on: true,
        temp: 1.0,
        user_explicit_sampling: true,
        min_p: Some(0.0),
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);

    // Nonzero min_p still falls to AR (DFlash ignores min_p).
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: true,
        ngram_can_sample: true,
        fast_sample_on: true,
        temp: 1.0,
        user_explicit_sampling: true,
        min_p: Some(0.05),
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);

    // Non-neutral penalties remain on AR because selector-chain verify
    // does not implement repeat/presence/frequency penalties.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: true,
        ngram_can_sample: true,
        fast_sample_on: true,
        temp: 1.0,
        user_explicit_sampling: true,
        nonneutral_penalties: true,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);

    // DDTree SWOR + explicit controls remains QwenAr (no chain nucleus).
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: false,
        ngram_can_sample: true,
        temp: 0.7,
        user_explicit_sampling: true,
        min_p: None,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);

    // Existing sampled MTP still selects QwenDflash with explicit controls.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        speculator_is_mtp: true,
        supports_temp_swor: true,
        supports_chain_nucleus_verify: false,
        temp: 0.7,
        user_explicit_sampling: true,
        min_p: Some(0.05),
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);

    // Legacy sampled chain (supports_temp_swor=false) unchanged: still
    // engages with nucleus via ngram_can_sample + fast_sample.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        supports_temp_swor: false,
        supports_chain_nucleus_verify: false,
        ngram_can_sample: true,
        fast_sample_on: true,
        temp: 0.7,
        user_explicit_sampling: true,
        min_p: None,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
}

#[test]
fn precedence_dflash_vs_ar() {
    // Qwen greedy + speculator → DFlash.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
    // force_ar_chat → AR.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        temp: 0.0,
        force_ar_chat: true,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
    // No speculator → AR.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: false,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
    // kv_adaptive blocks Qwen DFlash → AR.
    let i = GenerationRouteInputs {
        arch_id: 5,
        has_speculator: true,
        temp: 0.0,
        kv_adaptive: true,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
    // Llama greedy + spec → LlamaSpec; without → LlamaAr.
    let i = GenerationRouteInputs {
        arch_id: 1,
        has_speculator: true,
        temp: 0.0,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::LlamaSpec);
    let i = GenerationRouteInputs {
        arch_id: 1,
        has_speculator: false,
        ..base()
    };
    assert_eq!(select_generation_route(&i), GenerationRoute::LlamaAr);
}

#[test]
fn precedence_arch_spec_vs_ar_matrix() {
    // Qwen2
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 7,
            has_speculator: true,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::Qwen2Spec
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 7,
            has_speculator: true,
            temp: 0.7,
            ngram_can_sample: false,
            ..base()
        }),
        GenerationRoute::Qwen2Ar
    );
    // DeepSeek4
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 9,
            has_speculator: true,
            deepseek4_spec_requested: true,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::Deepseek4Spec
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 9,
            has_speculator: true,
            deepseek4_spec_requested: false,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::Deepseek4Ar
    );
    // Cohere
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 12,
            has_speculator: true,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::CohereSpec
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 12,
            has_speculator: false,
            ..base()
        }),
        GenerationRoute::CohereAr
    );
    // MiniMax
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 10,
            has_speculator: true,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::MiniMaxSpec
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 10,
            has_speculator: false,
            ..base()
        }),
        GenerationRoute::MiniMaxAr
    );
    // LFM
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 11,
            has_speculator: true,
            temp: 0.0,
            ..base()
        }),
        GenerationRoute::LfmSpec
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 11,
            has_speculator: true,
            temp: 0.8,
            ngram_can_sample: false,
            ..base()
        }),
        GenerationRoute::LfmAr
    );
    // dots + unknown
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 8,
            has_speculator: true,
            pp: 2,
            ..base()
        }),
        GenerationRoute::DotsOcr
    );
    assert_eq!(
        select_generation_route(&GenerationRouteInputs {
            arch_id: 42,
            ..base()
        }),
        GenerationRoute::Unknown
    );
}

#[test]
fn pure_gate_unsafe_tools_one_nonretryable_no_mutation() {
    for &route in GenerationRoute::ALL {
        if route.supports_tools() {
            continue;
        }
        let o = pure_tools_gate(route, true);
        assert_eq!(o.error_count, 1);
        assert_eq!(o.class, Some("unsupported"));
        assert_eq!(o.retryable, Some(false));
        assert!(!o.allowed);
        assert!(!o.mutated_generation_side);
        // Correlated: outcome carries the denied route identity.
        assert_eq!(o.route, route);
    }
}

#[test]
fn pure_gate_tools_absent_always_allowed() {
    for &route in GenerationRoute::ALL {
        let o = pure_tools_gate(route, false);
        assert!(o.allowed, "{:?} tools-absent", route);
        assert_eq!(o.error_count, 0);
        assert!(!o.mutated_generation_side);
    }
}

/// `Write` probe that observes whether a start adapter flushes the real
/// writer after the `gen_start` bytes were delivered to it.
struct FlushObservingWriter {
    bytes: Vec<u8>,
    /// Bytes present at the most recent `flush()`; trails `bytes.len()` when
    /// writes after the last flush were never flushed.
    flushed_through: usize,
    flush_count: usize,
}

impl std::io::Write for FlushObservingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flushed_through = self.bytes.len();
        self.flush_count += 1;
        Ok(())
    }
}

#[test]
fn all_route_starts_flush_gen_start_to_real_writer() {
    // Behavioral pin for the `gen_start` flush guarantee: every production
    // start adapter must deliver `gen_start` bytes to the real writer *and*
    // flush it, so piped daemon stdout shows `gen_start` throughout prefill
    // (canonical behavior in `hipfire_engine::emit::emit_gen_start`). A
    // `Vec`-only cardinality assertion cannot observe this; a `Write` that
    // records `flush()` can. New `GenerationRoute::ALL` variants are covered
    // automatically by iterating `ALL`.
    for &route in GenerationRoute::ALL {
        let mut writer = FlushObservingWriter {
            bytes: Vec::new(),
            flushed_through: 0,
            flush_count: 0,
        };
        // Unique id per route: the production adapter holds a per-request
        // start latch, so a shared id would suppress every start after the
        // first and the test would observe nothing.
        let id = format!("route-start-flush-{route:?}");
        generation_route_adapter(route)
            .unwrap_or_else(|| panic!("missing production adapter for {}", route.name()))
            .emit_start_with(&mut writer, &id, false);
        assert!(
            !writer.bytes.is_empty(),
            "{route:?} wrote no gen_start bytes"
        );
        let event: serde_json::Value = serde_json::from_str(
            std::str::from_utf8(&writer.bytes)
                .unwrap_or_else(|_| panic!("{route:?} gen_start is not UTF-8"))
                .trim(),
        )
        .unwrap_or_else(|_| panic!("{route:?} gen_start is not one JSON envelope"));
        assert_eq!(event["type"], "gen_start", "{route:?} start envelope");
        assert!(
            writer.flush_count >= 1,
            "{route:?} never flushed the real writer"
        );
        assert_eq!(
            writer.flushed_through,
            writer.bytes.len(),
            "{route:?} flushed before all gen_start bytes were written"
        );
    }
}

#[test]
fn route_cancel_releases_start_latch_and_claims_once() {
    let id = "route-cancel-lifecycle";
    let attempt = 7001;
    activate_terminal_control(id, attempt);
    set_active_attempt_id(attempt);
    let mut sink = Vec::new();
    emit_generation_start(GenerationRoute::GlimmerAr, &mut sink, id, false);
    emit_generation_cancel(GenerationRoute::GlimmerAr, &mut sink, id, 3);
    let first: Vec<_> = std::str::from_utf8(&sink)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert_eq!(first.len(), 3);
    assert_eq!(first[0]["type"], "gen_start");
    assert_eq!(first[1]["type"], "aborted");
    assert_eq!(first[2]["finish_reason"], "aborted");

    // The terminal claim remains one-shot, but the route-start latch is
    // released by the cancel wrapper, so a same-key fallback start is visible.
    emit_generation_start(GenerationRoute::GlimmerAr, &mut sink, id, false);
    emit_generation_cancel(GenerationRoute::GlimmerAr, &mut sink, id, 4);
    let all: Vec<_> = std::str::from_utf8(&sink)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert_eq!(
        all.iter()
            .filter(|event| event["type"] == "gen_start")
            .count(),
        2
    );
    assert_eq!(
        all.iter()
            .filter(|event| event["type"] == "aborted")
            .count(),
        1
    );
    clear_terminal_control();
    set_active_attempt_id(0);
}

/// VL and pipeline-parallel `done` carry `finish_reason`: a budget that runs
/// out is `length`, a terminator (even on the last budget token) is `stop`.
#[test]
fn vl_and_pp_done_report_length_on_budget_exhaustion() {
    assert_eq!(length_or_stop(16, 16, false), "length");
    assert_eq!(length_or_stop(16, 16, true), "stop");
    assert_eq!(length_or_stop(5, 16, true), "stop");
    // Early exit without a terminator (loop guard / forced EOS) is a stop.
    assert_eq!(length_or_stop(5, 16, false), "stop");
}
