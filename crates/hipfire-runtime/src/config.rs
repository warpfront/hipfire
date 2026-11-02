// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for hipfire-runtime.
//!
//! All `HIPFIRE_*` env vars are read exactly once via the global
//! `RuntimeConfig::get()` accessor. Runtime hot paths access config
//! fields instead of hitting `std::env::var` on every call.

use std::sync::OnceLock;

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — MQ4R route selection by file extension; 0.5.0 selects by HFQ metadata (no runtime warning: only way to reach the route today)
/// Automatic Redline runtime default for single-GPU MQ4R models.
pub fn mq4r_redline_default(gpu_arch: &str, model_path: &str, pp: usize, tp: usize) -> bool {
    matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201")
        && pp == 1
        && tp == 1
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"))
}

/// Automatic retained-Redline admission for a loaded model.
///
/// MQ4R keeps its existing cross-model policy with two carve-outs. Muse
/// Glimmer (arch 14) IS an MQ4R Redline SKU, but it is not lowered to
/// Redline PM4 yet, so automatic admission is withheld until that lowering
/// lands — the extension-only policy would otherwise route Glimmer onto a
/// dispatch path that cannot yet execute it. Delete this carve-out together
/// with the lowering. Gemma 4 (arch 13, every variant) is never admitted
/// automatically: its retained tape is not validated (the E4B PM4 shadow is
/// not exact), so a Gemma 4 `.mq4r` stays on the HIP graph unless Redline is
/// selected explicitly. DeepSeek4 MQ2R is narrower still: only the certified
/// gfx1151 single-GPU AR route is admitted, and an installed drafter keeps the
/// model on its speculative execution path. Qwen3.5 dense (arch 5) on exact
/// gfx1201 is admitted for any weight format on the single-GPU plain-AR route
/// (no drafter): its retained PM4 decode tape is byte-identical to the HIP
/// AR graph. Qwen4 (Flash-Next) on exact gfx1201 is admitted the same way:
/// single GPU, no drafter (a native MTP head is a drafter, so only an
/// MTP-off load replays), byte-identical greedy text to the HIP graph.
/// The symmetric-expert A3B file (`qwen3_5_moe`, `.mq4rs`) on exact gfx1201 is
/// admitted the same way (single GPU, no drafter, PM4 transport), with greedy
/// ids identical to the HIP route. `replay.backend = "hip"` opts out. A Qwen3.5-family process configured
/// for CPU-executed partial offload (`memory.offload_exec = "cpu"` with a
/// spill on either tier — an explicit layer or routed-expert count, or
/// explicit `auto`) gets no retained default: its
/// spilled layers' GEMVs run on the host between GPU launches, which a
/// retained tape does not record. Expert parallelism (`ep` > 1, Flash-Next EP)
/// runs one program per rank with a cross-device row gather between kernels,
/// which a single-device retained tape cannot record, so it is never admitted.
pub fn retained_redline_default(
    gpu_arch: &str,
    model_arch: &str,
    model_path: &str,
    pp: usize,
    tp: usize,
    ep: usize,
    has_drafter: bool,
) -> bool {
    if ep != 1 {
        return false;
    }
    if model_arch.eq_ignore_ascii_case("muse_glimmer") || model_arch.eq_ignore_ascii_case("gemma4")
    {
        return false;
    }
    if is_qwen35_family(model_arch) && cpu_offload_configured() {
        return false;
    }
    if mq4r_redline_default(gpu_arch, model_path, pp, tp) {
        return true;
    }
    if gpu_arch.eq_ignore_ascii_case("gfx1201")
        && (model_arch.eq_ignore_ascii_case("qwen3_5") || model_arch.eq_ignore_ascii_case("qwen4"))
        && pp == 1
        && tp == 1
        && !has_drafter
    {
        return true;
    }
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — A3B symmetric-MQ4R (`.mq4rs`) route selection by file extension; 0.5.0 selects by HFQ metadata (symmetric IU4 headers verified at load) (no runtime warning: only way to reach the route today)
    if gpu_arch.eq_ignore_ascii_case("gfx1201")
        && model_arch.eq_ignore_ascii_case("qwen3_5_moe")
        && pp == 1
        && tp == 1
        && !has_drafter
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4rs"))
    {
        return true;
    }
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — DS4 MQ2R selection by file extension; 0.5.0 selects by HFQ metadata (no runtime warning: only way to reach the route today)
    gpu_arch.eq_ignore_ascii_case("gfx1151")
        && model_arch.eq_ignore_ascii_case("deepseek4")
        && pp == 1
        && tp == 1
        && !has_drafter
        && std::path::Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mq2r"))
}

fn is_qwen35_family(model_arch: &str) -> bool {
    model_arch.eq_ignore_ascii_case("qwen3_5") || model_arch.eq_ignore_ascii_case("qwen3_5_moe")
}

fn cpu_offload_configured() -> bool {
    // Match both placement tiers: only explicit auto/manual budgets configure
    // a spill, and unset keeps every model fully resident.
    let (layer, expert) = hipfire_config::memory::effective_offload_budgets();
    cpu_offload_configured_from(hipfire_config::memory::offload_exec(), layer, expert)
}

/// Pure form of [`cpu_offload_configured`] over already-resolved inputs, so the
/// composition is testable without the process snapshot (which `process_value`
/// caches once per process and a unit test therefore cannot control).
fn cpu_offload_configured_from(
    exec: hipfire_config::memory::OffloadExec,
    layer: hipfire_config::memory::OffloadBudget,
    expert: hipfire_config::memory::OffloadBudget,
) -> bool {
    use hipfire_config::memory::{OffloadBudget, OffloadExec};
    // `passback` runs the CPU arm of a spilled step too, so it is a CPU
    // offload for this predicate exactly like `cpu`.
    if !matches!(exec, OffloadExec::Cpu | OffloadExec::Passback) {
        return false;
    }
    !matches!((layer, expert), (OffloadBudget::Full, OffloadBudget::Full))
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub normalize_prompt: bool,
    pub prompt_token_heat: bool,
    pub prompt_heat_json: bool,
    pub prompt_heat_limit: usize,
    pub attention_flash_mode: String,
    pub dflash_ngram_block: Option<bool>,
    pub dflash_fast_sample: bool,
    pub draft_f16: bool,
    pub draft_gemm_dump: bool,
    pub draft_subphase: bool,
    pub prefill_batched: bool,
    pub flash_partials_batch: Option<usize>,
    pub lm_head_f16: String,
    /// Tensor-parallel RCCL all-reduce toggle. `None` (unset) → RCCL is used
    /// (default). `Some(false)` (HIPFIRE_TP_USE_RCCL=0) → opt out of the RCCL
    /// path. `Some(true)` → force on. Read by `multi_gpu::Gpus::ensure_rccl`.
    pub tp_use_rccl: Option<bool>,
    pub ngram_loop_threshold: usize,
    pub ngram_window: usize,
    pub ngram_draft: bool,
    pub ngram_k: usize,
    pub ngram_min_count: u32,
    pub kv_mode: String,
    pub kv_adaptive: String,
    pub chat_template_file: Option<String>,
    pub prompt_cache_capacity: usize,
    pub prompt_cache_unbounded: bool,
    pub experimental_budget_alert: bool,
    pub max_total_think_tokens: usize,
    pub devices: Option<String>,
    pub emulate_gpus: Option<usize>,
    pub allow_mixed_arch: bool,
    pub uniform_vram_tolerance_gb: Option<f32>,
    pub mtp_mode: String,
    /// Explicit numeric `HIPFIRE_MTP_K` / `speculation.mtp_k`. `None` = unset
    /// or `auto`: the consumer applies its default (`hipfire_config::
    /// mtp_k_default` over the GPU's row-exact verify rows for native Qwen4
    /// MTP, else `MTP_K_DEFAULT`).
    pub mtp_k: Option<usize>,
    /// `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR`: YaRN RoPE factor
    /// override applied by [`crate::rope::rope_scaling`]. `None` = the
    /// checkpoint's own `rope_parameters`.
    pub yarn_factor: Option<f64>,
    /// Opt-in gfx1151 XDNA NPU spillover (`kernel.npu_spillover`,
    /// `HIPFIRE_NPU_SPILLOVER`). Default false; snapshotted once here, never
    /// re-read from the environment on hot paths.
    pub npu_spillover: bool,
}

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub fn get() -> &'static RuntimeConfig {
    CONFIG.get_or_init(|| {
        RuntimeConfig::from_process_config(hipfire_config::active_or_local_process_config())
    })
}

pub fn init() {
    get();
}

pub fn init_with(config: RuntimeConfig) -> std::result::Result<(), RuntimeConfig> {
    CONFIG.set(config)
}

impl RuntimeConfig {
    pub fn from_process_config(config: &hipfire_config::ProcessConfig) -> Self {
        Self::from_lookup(|name| config.legacy_value(name))
    }

    fn from_lookup(mut value: impl FnMut(&str) -> Option<String>) -> Self {
        let normalize_prompt = match value("HIPFIRE_NORMALIZE_PROMPT").as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            _ => true,
        };

        let prompt_heat_json = value("HIPFIRE_PROMPT_HEAT_JSON").as_deref() == Some("1");
        let prompt_heat_limit: usize = value("HIPFIRE_PROMPT_HEAT_LIMIT")
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        Self {
            normalize_prompt,
            prompt_token_heat: value("HIPFIRE_PROMPT_TOKEN_HEAT").as_deref() == Some("1"),
            prompt_heat_json,
            prompt_heat_limit,
            attention_flash_mode: value("HIPFIRE_ATTN_FLASH").unwrap_or_else(|| "auto".into()),
            dflash_ngram_block: match value("HIPFIRE_DFLASH_NGRAM_BLOCK").as_deref() {
                Some("1") | Some("true") | Some("on") => Some(true),
                Some("0") | Some("false") | Some("off") => Some(false),
                _ => None,
            },
            dflash_fast_sample: value("HIPFIRE_DFLASH_FAST_SAMPLE").as_deref() != Some("0"),
            draft_f16: value("HIPFIRE_DRAFT_F16").as_deref() != Some("0"),
            draft_gemm_dump: value("HIPFIRE_DRAFT_GEMM_DUMP").as_deref() == Some("1"),
            draft_subphase: value("HIPFIRE_DRAFT_SUBPHASE").as_deref() == Some("1"),
            prefill_batched: value("HIPFIRE_PREFILL_BATCHED").as_deref() != Some("0"),
            flash_partials_batch: value("HIPFIRE_FLASH_PARTIALS_BATCH")
                .and_then(|s| s.parse::<usize>().ok()),
            lm_head_f16: value("HIPFIRE_LM_HEAD_F16").unwrap_or_else(|| "auto".into()),
            tp_use_rccl: value("HIPFIRE_TP_USE_RCCL")
                .as_deref()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false")),
            // Default OFF (0 = disabled). The content-blind 4-gram guard can
            // false-positive on repeated markdown/list scaffolding inside
            // sampled reasoning and force-EOS before a visible answer. The
            // think-budget force-close handles runaway reasoning instead;
            // operators can opt this guard back in with an explicit threshold.
            ngram_loop_threshold: value("HIPFIRE_NGRAM_LOOP_THRESHOLD")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            ngram_window: value("HIPFIRE_NGRAM_WINDOW")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            // `auto` is not an explicit arm here (the daemon-direct fallback);
            // on/1/true is, via the shared strict tri-state parser.
            ngram_draft: matches!(
                value("HIPFIRE_NGRAM_DRAFT")
                    .as_deref()
                    .map(hipfire_config::parse_tri_state_alias),
                Some(Ok(Some(true)))
            ),
            ngram_k: value("HIPFIRE_NGRAM_DRAFT_K")
                .and_then(|v| v.parse().ok())
                .unwrap_or(12),
            ngram_min_count: value("HIPFIRE_NGRAM_MIN_COUNT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            kv_mode: value("HIPFIRE_KV_MODE").unwrap_or_else(|| "auto".into()),
            kv_adaptive: value("HIPFIRE_KV_ADAPTIVE").unwrap_or_else(|| "off".into()),
            chat_template_file: value("HIPFIRE_CHAT_TEMPLATE_FILE")
                .filter(|value| !value.is_empty()),
            prompt_cache_capacity: value("HIPFIRE_PROMPT_CACHE_CAP")
                .and_then(|value| value.parse().ok())
                .unwrap_or(32),
            prompt_cache_unbounded: value("HIPFIRE_PROMPT_CACHE_UNBOUNDED").as_deref() == Some("1"),
            experimental_budget_alert: value("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT").as_deref()
                == Some("1"),
            max_total_think_tokens: value("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            // Each `hardware.devices` entry resolves to one physical card,
            // installed as a ROCr selector with HIP logical `0..N-1` before
            // GPU initialization. The engine addresses that set as 0..N-1.
            devices: value("HIPFIRE_DEVICES")
                .filter(|value| !value.is_empty())
                .map(|value| {
                    (0..value.split(',').count())
                        .map(|device| device.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                }),
            emulate_gpus: value("HIPFIRE_EMULATE_GPUS")
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value >= 2),
            allow_mixed_arch: value("HIPFIRE_ALLOW_MIXED_ARCH").as_deref() == Some("1"),
            uniform_vram_tolerance_gb: value("HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB")
                .and_then(|s| s.parse().ok()),
            mtp_mode: value("HIPFIRE_MTP_MODE").unwrap_or_else(|| "auto".into()),
            mtp_k: value("HIPFIRE_MTP_K").and_then(|value| value.parse().ok()),
            yarn_factor: value("HIPFIRE_YARN_FACTOR").and_then(|value| value.parse().ok()),
            npu_spillover: value("HIPFIRE_NPU_SPILLOVER").as_deref() == Some("1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mq4r_redline_default, retained_redline_default as redline_with_ep, RuntimeConfig};
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};

    /// The pre-EP call shape: every case below is a single-rank (`ep == 1`) load.
    fn retained_redline_default(
        gpu_arch: &str,
        model_arch: &str,
        model_path: &str,
        pp: usize,
        tp: usize,
        has_drafter: bool,
    ) -> bool {
        redline_with_ep(gpu_arch, model_arch, model_path, pp, tp, 1, has_drafter)
    }

    #[test]
    fn expert_parallel_never_admits_retained_redline() {
        let fn_xts = "/models/qwen3.8-flash-next-gptq3.mq4";
        assert!(redline_with_ep("gfx1201", "qwen4", fn_xts, 1, 1, 1, false));
        for ep in [2, 4] {
            assert!(!redline_with_ep(
                "gfx1201", "qwen4", fn_xts, 1, 1, ep, false
            ));
            assert!(!redline_with_ep(
                "gfx1201", "qwen3_5", fn_xts, 1, 1, ep, false
            ));
            assert!(!redline_with_ep(
                "gfx1201",
                "qwen4",
                "/models/fn.mq4r",
                1,
                1,
                ep,
                false
            ));
        }
    }

    #[test]
    fn npu_spillover_is_off_unless_process_snapshot_enables_it() {
        let process = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert!(!cfg.npu_spillover);
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.npu_spillover", "true").unwrap();
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: "config.toml".into(),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert!(cfg.npu_spillover);
    }
    #[test]
    fn ngram_loop_guard_is_off_by_default() {
        let process = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert_eq!(cfg.ngram_loop_threshold, 0);
    }

    #[test]
    fn normalize_prompt_accepts_no_as_false() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("prompt.normalize", "no").unwrap();
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: "config.toml".into(),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap();
        let cfg = RuntimeConfig::from_process_config(&process);
        assert!(!cfg.normalize_prompt);
    }

    #[test]
    fn process_config_populates_runtime_without_ambient_env() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("prompt.normalize", "false").unwrap();
        layer
            .set_cli("generation.loop_guard_threshold", "12")
            .unwrap();
        layer.set_cli("hardware.devices", "2,3").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let config = RuntimeConfig::from_process_config(&process);

        assert!(!config.normalize_prompt);
        assert_eq!(config.ngram_loop_threshold, 12);
        assert_eq!(config.devices.as_deref(), Some("0,1"));
        assert!(config.prefill_batched, "sparse arch defaults remain intact");
    }

    #[test]
    fn emulate_gpus_parser_requires_at_least_two() {
        fn parse(raw: Option<&str>) -> Option<usize> {
            RuntimeConfig::from_lookup(|name| {
                if name == "HIPFIRE_EMULATE_GPUS" {
                    raw.map(str::to_owned)
                } else {
                    None
                }
            })
            .emulate_gpus
        }

        for (raw, expected) in [
            (None, None),
            (Some("malformed"), None),
            (Some("0"), None),
            (Some("1"), None),
            (Some("2"), Some(2)),
            (Some("4"), Some(4)),
        ] {
            assert_eq!(
                parse(raw),
                expected,
                "unexpected emulation parse for {raw:?}"
            );
        }
    }

    #[test]
    fn redline_default_is_limited_to_exact_runtime_arches_and_single_gpu_mq4r() {
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(mq4r_redline_default(
                gpu_arch,
                "/models/qwen3.6-35b-a3b.mq4r",
                1,
                1,
            ));
        }

        assert!(mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-27b.mq4r",
            1,
            1,
        ));

        assert!(!mq4r_redline_default(
            "gfx1200",
            "/models/QWEN3.6-35B-A3B.MQ4R",
            1,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4",
            1,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            2,
            1,
        ));
        assert!(!mq4r_redline_default(
            "gfx1201",
            "/models/qwen3.6-35b-a3b.mq4r",
            1,
            2,
        ));
    }

    #[test]
    fn a3b_mq4rs_takes_retained_redline_on_gfx1201_only() {
        let f = "/models/qwen3.6-35b-a3b-d53.mq4rs";
        assert!(retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            f,
            1,
            1,
            false
        ));
        assert!(retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            "/models/A3B.MQ4RS",
            1,
            1,
            false
        ));
        // drafter (MTP-loaded), other arches, multi-GPU, other families, other extensions
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            f,
            1,
            1,
            true
        ));
        for arch in ["gfx1200", "gfx1100", "gfx1151", "gfx942"] {
            assert!(!retained_redline_default(
                arch,
                "qwen3_5_moe",
                f,
                1,
                1,
                false
            ));
        }
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            f,
            2,
            1,
            false
        ));
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            f,
            1,
            2,
            false
        ));
        assert!(!retained_redline_default(
            "gfx1201", "qwen3", f, 1, 1, false
        ));
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen3_5_moe",
            "/models/a3b.mq4",
            1,
            1,
            false
        ));
        assert!(!redline_with_ep(
            "gfx1201",
            "qwen3_5_moe",
            f,
            1,
            1,
            2,
            false
        ));
    }

    #[test]
    fn muse_glimmer_mq4r_defers_redline_until_pm4_lowering() {
        // Glimmer's trunk is a genuine MQ4R Redline SKU, but it is not lowered
        // to Redline PM4 yet, so the extension-only policy must not admit it.
        // Retire this test with the carve-out when the lowering lands.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(
                mq4r_redline_default(gpu_arch, "/models/muse-glimmer-30b.mq4r", 1, 1),
                "extension-only policy still matches the Glimmer trunk"
            );
            assert!(
                !retained_redline_default(
                    gpu_arch,
                    "muse_glimmer",
                    "/models/muse-glimmer-30b.mq4r",
                    1,
                    1,
                    false,
                ),
                "glimmer must stay on HIP by default on {gpu_arch}"
            );
            assert!(
                !retained_redline_default(
                    gpu_arch,
                    "muse_glimmer",
                    "/models/muse-glimmer-30b.mq4r",
                    1,
                    1,
                    true,
                ),
                "a loaded drafter does not re-admit glimmer on {gpu_arch}"
            );
            // The carve-out is glimmer-scoped, not a global MQ4R retreat.
            assert!(
                retained_redline_default(
                    gpu_arch,
                    "qwen3_6_moe",
                    "/models/qwen3.6-35b-a3b.mq4r",
                    1,
                    1,
                    false,
                ),
                "qwen MQ4R keeps auto-Redline on {gpu_arch}"
            );
        }
    }

    #[test]
    fn gemma4_never_takes_the_retained_redline_default() {
        // Gemma 4's retained tape is unvalidated, so no Gemma 4 load replays
        // by default: not an MQ4R name, not on the gfx1201 dense route, with
        // or without the EAGLE drafter, on any arch.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx942"] {
            for path in [
                "/models/gemma4-12b-it.mq4r",
                "/models/GEMMA4-E4B-IT.MQ4R",
                "/models/gemma-4-26b-a4b-it.mq4r",
                "/models/gemma4-12b-it.mq4",
                "/models/gemma4-e4b-it.q8.hfq",
            ] {
                for has_drafter in [false, true] {
                    assert!(
                        !retained_redline_default(gpu_arch, "gemma4", path, 1, 1, has_drafter),
                        "gemma4 {path} must stay on HIP by default on {gpu_arch}"
                    );
                }
            }
        }
        // The carve-out is Gemma-scoped: the same MQ4R name on another
        // family keeps its default.
        assert!(retained_redline_default(
            "gfx1201",
            "qwen3_6_moe",
            "/models/gemma4-12b-it.mq4r",
            1,
            1,
            false,
        ));
    }

    #[test]
    fn deepseek4_mq2r_redline_default_requires_gfx1151_ar() {
        assert!(retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            true,
        ));
        assert!(!retained_redline_default(
            "gfx1100",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "qwen3_5_moe",
            "/models/deepseek-v4-flash-0731.mq2r",
            1,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2r",
            2,
            1,
            false,
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2",
            1,
            1,
            false,
        ));
    }

    #[test]
    fn qwen35_dense_redline_default_requires_gfx1201_single_gpu_ar() {
        let h2 = "/models/h2.group-alpha-refit.hfq";
        assert!(retained_redline_default(
            "gfx1201", "qwen3_5", h2, 1, 1, false
        ));
        // A drafter keeps the model on its speculative path.
        assert!(!retained_redline_default(
            "gfx1201", "qwen3_5", h2, 1, 1, true
        ));
        // Pipeline / tensor parallel stay on HIP.
        assert!(!retained_redline_default(
            "gfx1201", "qwen3_5", h2, 2, 1, false
        ));
        assert!(!retained_redline_default(
            "gfx1201", "qwen3_5", h2, 1, 2, false
        ));
        // Exact gfx1201 only; other dense archs and the MoE sibling stay on HIP.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1200"] {
            assert!(!retained_redline_default(
                gpu_arch, "qwen3_5", h2, 1, 1, false
            ));
        }
        for model_arch in ["qwen3_5_moe", "qwen3", "qwen2", "gemma4"] {
            assert!(!retained_redline_default(
                "gfx1201", model_arch, h2, 1, 1, false
            ));
        }
    }

    #[test]
    fn qwen4_redline_default_requires_gfx1201_single_gpu_without_mtp() {
        let fn_xts = "/models/qwen3.8-flash-next-gptq3.mq4";
        assert!(retained_redline_default(
            "gfx1201", "qwen4", fn_xts, 1, 1, false
        ));
        // The native MTP head is a drafter: an MTP load keeps its spec path.
        assert!(!retained_redline_default(
            "gfx1201", "qwen4", fn_xts, 1, 1, true
        ));
        assert!(!retained_redline_default(
            "gfx1201", "qwen4", fn_xts, 2, 1, false
        ));
        assert!(!retained_redline_default(
            "gfx1201", "qwen4", fn_xts, 1, 2, false
        ));
        // A plain `.mq4` Flash-Next stays on HIP off gfx1201.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1200"] {
            assert!(!retained_redline_default(
                gpu_arch, "qwen4", fn_xts, 1, 1, false
            ));
        }
    }

    #[test]
    fn cpu_offload_requires_explicit_spill_on_either_tier() {
        // Pin the exec-mode/budget composition without reading the process
        // snapshot, which is cached once per process.
        use super::cpu_offload_configured_from;
        use hipfire_config::memory::{OffloadBudget, OffloadExec};

        // Not cpu: never a cpu-offload model, whatever the budgets say.
        for budget in [
            OffloadBudget::Full,
            OffloadBudget::Auto,
            OffloadBudget::Layers(4),
        ] {
            assert!(!cpu_offload_configured_from(
                OffloadExec::Pcie,
                budget,
                budget
            ));
        }
        // CPU with both tiers unset is not configured on any model.
        assert!(!cpu_offload_configured_from(
            OffloadExec::Cpu,
            OffloadBudget::Full,
            OffloadBudget::Full
        ));
        assert!(!cpu_offload_configured_from(
            OffloadExec::Passback,
            OffloadBudget::Full,
            OffloadBudget::Full
        ));
        // cpu with either tier placed at all is configured — the expert tier and
        // the `auto` spelling included, which the older layer-tier-only check
        // missed.
        for (layer, expert) in [
            (OffloadBudget::Layers(4), OffloadBudget::Full),
            (OffloadBudget::Full, OffloadBudget::Layers(4)),
            (OffloadBudget::Auto, OffloadBudget::Auto),
            (OffloadBudget::Auto, OffloadBudget::Full),
            (OffloadBudget::Full, OffloadBudget::Auto),
        ] {
            assert!(cpu_offload_configured_from(OffloadExec::Cpu, layer, expert));
            // `passback` runs the CPU arm of a spilled step, so it counts too.
            assert!(cpu_offload_configured_from(
                OffloadExec::Passback,
                layer,
                expert
            ));
        }
    }
}
