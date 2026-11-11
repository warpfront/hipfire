//! Cache-specific sampled admission through the production native-MTP router.
//! Uses the production Qwen4 builder and route selector, without a GPU/model.

use hipfire_arch_qwen4::{mtp_spec::build_qwen4_mtp_speculator, GdnStateFormat};
use hipfire_generate::ar::{select_generation_route, GenerationRoute, GenerationRouteInputs};
use hipfire_runtime::spec::{SpecRequestConfig, Speculator};

fn inputs(speculator: &dyn Speculator, temp: f32) -> GenerationRouteInputs {
    GenerationRouteInputs {
        arch_id: hipfire_arch_qwen4::ARCH_ID,
        ep: false,
        dense_tp: false,
        pp: 1,
        has_speculator: true,
        speculator_is_mtp: speculator.name() == "mtp",
        speculator_is_dflash2: speculator.name() == "dflash2",
        speculator_supports_penalties: speculator.supports_penalties(),
        deepseek4_spec_requested: false,
        ngram_can_sample: false,
        temp,
        user_explicit_sampling: temp > 0.0,
        min_p: None,
        nonneutral_penalties: false,
        force_ar_chat: false,
        temp_spec_env_off: false,
        fast_sample_on: true,
        supports_temp_swor: speculator.supports_temp_verify(),
        supports_chain_nucleus_verify: speculator.supports_chain_nucleus_verify(),
        kv_adaptive: false,
    }
}

#[test]
fn actual_native_builder_routes_cached_greedy_sampled_greedy_without_global_disable() {
    // The positive control must actually exercise sampled native capability.
    // No HIPFIRE_MTP_SAMPLED=0 workaround: a globally disabled test cannot
    // distinguish the cache-specific gate from the original reviewer bug.
    assert!(
        hipfire_config::mtp_sampled_enabled(),
        "run this matrix with the ordinary default/on sampled policy"
    );
    for gdn in [
        GdnStateFormat::Q8,
        GdnStateFormat::F32,
        GdnStateFormat::Q8Ef,
    ] {
        for actual_cache_enabled in [false, true] {
            let mut speculator = build_qwen4_mtp_speculator(
                7,
                None,
                512,
                None,
                None,
                None,
                false,
                gdn,
                actual_cache_enabled,
                None,
            );
            assert_eq!(speculator.name(), "mtp");
            let sampled_allowed = !actual_cache_enabled && gdn != GdnStateFormat::Q8Ef;
            assert_eq!(speculator.supports_temp_verify(), sampled_allowed);
            assert_eq!(speculator.requires_greedy(), !sampled_allowed);
            // Reconfigure the same real speculator as a long-lived server
            // would: sampled refusal must not permanently demote later greedy.
            for temp in [0.0, 1.0e-7, 0.3, 0.0] {
                speculator.configure_request(SpecRequestConfig {
                    temp,
                    ..SpecRequestConfig::default()
                });
                let expected = if temp <= 1.0e-6 || sampled_allowed {
                    GenerationRoute::Qwen4Spec
                } else {
                    GenerationRoute::Qwen4Ar
                };
                assert_eq!(
                    select_generation_route(&inputs(speculator.as_ref(), temp)),
                    expected,
                    "gdn={gdn:?}, actual_cache={actual_cache_enabled}, temp={temp}"
                );
            }
        }
    }
}

#[test]
fn cached_native_builder_does_not_override_existing_forced_ar_or_penalty_routes() {
    let speculator = build_qwen4_mtp_speculator(
        7,
        None,
        512,
        None,
        None,
        None,
        false,
        GdnStateFormat::Q8,
        true,
        None,
    );
    let greedy = inputs(speculator.as_ref(), 0.0);
    assert_eq!(select_generation_route(&greedy), GenerationRoute::Qwen4Spec);
    for refused in [
        GenerationRouteInputs {
            force_ar_chat: true,
            ..greedy
        },
        GenerationRouteInputs {
            temp_spec_env_off: true,
            ..greedy
        },
        GenerationRouteInputs {
            kv_adaptive: true,
            ..greedy
        },
        GenerationRouteInputs {
            nonneutral_penalties: true,
            ..greedy
        },
    ] {
        assert_eq!(select_generation_route(&refused), GenerationRoute::Qwen4Ar);
    }
}
