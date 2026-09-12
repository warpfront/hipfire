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
        arch_id: 255,
        ep: false,
        pp: 1,
        has_speculator: false,
        speculator_is_mtp: false,
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
    assert!(deepseek4_spec_requested_from_policy(
        Some("dspark"),
        "off",
        "off",
        false,
    ));
    assert!(!deepseek4_spec_requested_from_policy(
        None, "off", "auto", true,
    ));
    assert!(deepseek4_spec_requested_from_policy(
        None, "auto", "auto", true,
    ));
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
fn exact_safe_set_is_qwen_ar_dflash_ds4_ar_ep_spec_glimmer_ar_spec_and_maple_ar() {
    let mut from_all: Vec<GenerationRoute> = GenerationRoute::ALL
        .iter()
        .copied()
        .filter(|r| r.supports_tools())
        .collect();
    from_all.sort_by_key(|r| r.name());
    let mut expected = SAFE_ROUTES.to_vec();
    expected.sort_by_key(|r| r.name());
    assert_eq!(from_all, expected);
    assert_eq!(from_all.len(), 8);
    // Negative: every other ALL member is denied for tools.
    for &r in GenerationRoute::ALL {
        if !SAFE_ROUTES.contains(&r) {
            assert!(!r.supports_tools(), "{:?} must not be tool-safe", r);
        }
    }
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
