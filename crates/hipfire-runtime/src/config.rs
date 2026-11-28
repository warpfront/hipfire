// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable env-var resolution for hipfire-runtime.
//!
//! All `HIPFIRE_*` env vars are read exactly once via the global
//! `RuntimeConfig::get()` accessor. Runtime hot paths access config
//! fields instead of hitting `std::env::var` on every call.

use crate::route_sku::{DenseWeights, RouteSku};
use std::sync::OnceLock;

/// Exact GPU arches whose retained PM4 decode of the symmetric-expert A3B SKU
/// ([`RouteSku::Mq4rs`]) has served greedy-identity and decode-timing receipts
/// (REDLINE.md §11.6). gfx1151 is not one of them.
const A3B_MQ4RS_RETAINED_ARCHES: [&str; 2] = ["gfx1100", "gfx1201"];

/// Automatic Redline runtime default for single-GPU MQ4R models. The SKU is
/// the file's identity ([`RouteSku::Mq4r`], see [`crate::route_sku`]); the
/// file name is never read.
pub fn mq4r_redline_default(gpu_arch: &str, route: Option<RouteSku>, pp: usize, tp: usize) -> bool {
    matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201")
        && pp == 1
        && tp == 1
        && route == Some(RouteSku::Mq4r)
}

/// The speculative drafter a load attached, as the retained-Redline admission
/// sees it. Admission is typed by drafter kind, never by model name: a kind
/// joins a cell only once its retained AR continuation is proved there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachedDraft {
    /// No drafter: plain-AR decode only.
    None,
    /// A DFlash drafter on a Qwen3.5 trunk. Its AR forwards stage the draft's
    /// extraction-layer hidden rows into fixed load-lifetime rows inside the
    /// retained tape and commit them outside it, so the drafter's context
    /// survives a retained AR step.
    Dflash,
    /// A native MTP head (Qwen3.5/3.6 NextN, Flash-Next, DeepSeek V4).
    NativeMtp,
    /// Every other drafter (DSpark, EAGLE, DFlash2, n-gram, …).
    Other,
}

impl AttachedDraft {
    /// The kind of the load's speculator, by its `Speculator::name`.
    ///
    /// `batched_lanes` is whether the load stages the continuous-batching
    /// speculative lanes (`continuous_batch_size > 1` with
    /// `serve_vmm_batch.spec`). A DFlash load that serves through those lanes
    /// is `Other`: its gfx1201 save-free DeltaNet verify refuses under an
    /// enabled retained controller and swaps the owners a retained tape would
    /// bake, so that load keeps the HIP route.
    pub fn from_speculator(name: Option<&str>, batched_lanes: bool) -> Self {
        match name {
            None => Self::None,
            Some("dflash") if !batched_lanes => Self::Dflash,
            Some("mtp") => Self::NativeMtp,
            Some(_) => Self::Other,
        }
    }

    /// Stable label for load logs (`drafter=…`).
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "off",
            Self::Dflash => "dflash",
            Self::NativeMtp => "mtp",
            Self::Other => "other",
        }
    }
}

/// Automatic retained-Redline admission for a loaded model.
///
/// MQ4R ([`RouteSku::Mq4r`]) keeps its existing cross-model policy with three carve-outs. Muse
/// Glimmer (arch 14) IS an MQ4R Redline SKU, but it is not lowered to
/// Redline PM4 yet, so automatic admission is withheld until that lowering
/// lands — the SKU-only policy would otherwise route Glimmer onto a
/// dispatch path that cannot yet execute it. Delete this carve-out together
/// with the lowering. Gemma 4 (arch 13, every variant) is never admitted
/// automatically: its retained tape is not validated (the E4B PM4 shadow is
/// not exact), so a Gemma 4 MQ4R file stays on the HIP graph unless Redline is
/// selected explicitly. DeepSeek4 MQ2R is narrower still: only the certified
/// gfx1151 single-GPU AR route is admitted, and an installed drafter keeps the
/// model on its speculative execution path. Qwen3.5 dense (arch 5) on exact
/// gfx1201 is admitted for any weight format on the single-GPU plain-AR route
/// (no drafter), and with a DFlash drafter ([`AttachedDraft::Dflash`]) only for
/// symmetric MQ4G256V2 ([`dflash_pm4_weights_qualified`]): its
/// retained PM4 decode tape is byte-identical to the HIP AR graph. Qwen4
/// (Flash-Next) on exact gfx1201 is admitted the same way but with no drafter
/// only (a native MTP head is a drafter, so only an MTP-off load replays),
/// byte-identical greedy text to the HIP graph.
/// The symmetric-expert A3B SKU (`qwen3_5_moe`, [`RouteSku::Mq4rs`]) on exact gfx1100 or
/// gfx1201 (`A3B_MQ4RS_RETAINED_ARCHES`) is admitted the same way (single
/// GPU, no drafter, PM4 transport), with greedy ids identical to the HIP route
/// on both (gfx1100: x1.22-1.32 decode tok/s, REDLINE.md §11.6); gfx1151 has no
/// receipts and stays on HIP. A Qwen3.5 dense trunk on exact gfx1100 is
/// admitted for single-GPU AR decode (no drafter, or a DFlash drafter) when its
/// decode weight
/// format ([`DenseWeights`], read from the tensor index and HFQ metadata, never
/// the file name) is in [`gfx1100_dense_weights_qualified`]: the MQ3G256V2
/// tape needs the GFX10/GFX11 scratch rings (`gemv_mq3g256v2_residual` has a
/// 16-byte private segment), and served `generate` greedy ids are identical to
/// the HIP graph at x1.06-1.17 the decode tok/s. `replay.backend = "hip"` opts out. A Qwen3.5-family process configured
/// for CPU-executed partial offload (`memory.offload_exec = "cpu"` with a
/// spill on either tier — an explicit layer or routed-expert count, or
/// explicit `auto`) gets no retained default: its
/// spilled layers' GEMVs run on the host between GPU launches, which a
/// retained tape does not record. Expert parallelism (`ep` > 1, Flash-Next EP)
/// runs one program per rank with a cross-device row gather between kernels,
/// which a single-device retained tape cannot record, so it is never admitted.
///
/// Qwen4 (Flash-Next) on gfx1100/gfx1151 is never admitted automatically,
/// the MQ4R SKU included (HIPF-254): recording closes the eager-only QSA split-KV
/// decode attention, so the replay differs from the HIP route and measured
/// slower on gfx1151.
pub fn retained_redline_default(
    gpu_arch: &str,
    model_arch: &str,
    route: Option<RouteSku>,
    weights: Option<DenseWeights>,
    pp: usize,
    tp: usize,
    ep: usize,
    draft: AttachedDraft,
) -> bool {
    // Only the two Qwen3.5-dense cells admit a DFlash drafter; every other
    // rule treats any attached drafter as excluding the load.
    let qwen35_dense_draft_ok = matches!(draft, AttachedDraft::None | AttachedDraft::Dflash);
    let has_drafter = draft != AttachedDraft::None;
    if ep != 1 {
        return false;
    }
    // Inkling-2-Mini is admitted only through [`inkling2_redline_default`] (its
    // typed certificate and recipe, a fact of the load): never by a generic
    // route-SKU, weight-format or file-name allowance.
    if model_arch.eq_ignore_ascii_case("inkling2") {
        return false;
    }
    if model_arch.eq_ignore_ascii_case("muse_glimmer") || model_arch.eq_ignore_ascii_case("gemma4")
    {
        return false;
    }
    // HIPF-254: recording a Qwen4 body closes the eager-only QSA split-KV decode
    // attention (gfx1100/gfx1151), so the replay differs from default HIP and
    // measured ~5-8% slower on gfx1151. Lift only with a perf A/B that wins.
    if model_arch.eq_ignore_ascii_case("qwen4") && matches!(gpu_arch, "gfx1100" | "gfx1151") {
        return false;
    }
    if is_qwen35_family(model_arch) && cpu_offload_configured() {
        return false;
    }
    // gfx1100 Qwen3.5 dense AR: a qualified decode weight format; the
    // receipts per format are the 27B served identity + tok/s in REDLINE.md §3.
    // A DFlash drafter admits the same formats: the DFlash + MQ3G256V2
    // (`mq3-xt`, `mq3-xts`) receipt is in REDLINE.md §3.
    if gpu_arch == "gfx1100"
        && model_arch.eq_ignore_ascii_case("qwen3_5")
        && pp == 1
        && tp == 1
        && qwen35_dense_draft_ok
        && weights.is_some_and(gfx1100_dense_weights_qualified)
    {
        return true;
    }
    if mq4r_redline_default(gpu_arch, route, pp, tp) {
        return true;
    }
    if gpu_arch.eq_ignore_ascii_case("gfx1201") && pp == 1 && tp == 1 {
        // Plain AR: any weight format. A DFlash drafter narrows this to the
        // symmetric MQ4G256V2 trunk, the only format with a measured DFlash
        // PM4 receipt on gfx1201.
        if model_arch.eq_ignore_ascii_case("qwen3_5")
            && match draft {
                AttachedDraft::None => true,
                AttachedDraft::Dflash => weights.is_some_and(dflash_pm4_weights_qualified),
                _ => false,
            }
        {
            return true;
        }
        if model_arch.eq_ignore_ascii_case("qwen4") && !has_drafter {
            return true;
        }
    }
    if A3B_MQ4RS_RETAINED_ARCHES
        .iter()
        .any(|arch| gpu_arch.eq_ignore_ascii_case(arch))
        && model_arch.eq_ignore_ascii_case("qwen3_5_moe")
        && pp == 1
        && tp == 1
        && !has_drafter
        && route == Some(RouteSku::Mq4rs)
    {
        return true;
    }
    gpu_arch.eq_ignore_ascii_case("gfx1151")
        && model_arch.eq_ignore_ascii_case("deepseek4")
        && pp == 1
        && tp == 1
        && !has_drafter
        && route == Some(RouteSku::Mq2r)
}

/// Automatic retained-Redline (PM4) default of Inkling-2-Mini (arch 24, the
/// frozen one-row decode body). `qualified` is the load's own typed fact
/// (`hipfire_arch_inkling2::retained_default_qualified`: gfx1201, the Mini fixed
/// schedule, the typed-XTS certificate, A8 experts with the fused producer
/// recipe, F16 KV, FA2 attention, and the single product switch
/// `PM4_DEFAULT_ON`); this adds only the topology: one device (PP = TP = EP = 1)
/// and no drafter. Every other model, tier or fixture stays on `retained_redline_default`.
pub fn inkling2_redline_default(
    gpu_arch: &str,
    pp: usize,
    tp: usize,
    ep: usize,
    draft: AttachedDraft,
    qualified: bool,
) -> bool {
    qualified
        && gpu_arch.eq_ignore_ascii_case("gfx1201")
        && pp == 1
        && tp == 1
        && ep == 1
        && draft == AttachedDraft::None
}

/// On-context (tokens) at or above which dense native MTP arms under `auto`
/// whatever it costs relative to the no-head context: a head that still leaves
/// this much room does not take a usable window away (Kaden's ruling,
/// 2026-11-22). Not keyed by model, format, card name or price.
pub const DENSE_MTP_AUTO_CONTEXT_FLOOR: usize = 131_072;

/// Facts the dense native-MTP `auto` decision reads. Every field is a fact of
/// the load, none a model name, weight-format label, acceptance rate or timing.
#[derive(Debug, Clone, Copy)]
pub struct DenseMtpAutoFacts<'a> {
    /// Architecture id of the loaded model (dense Qwen3.5-class = 5).
    pub arch_id: u32,
    /// Exact device arch string (`gfx1100` / `gfx1201`); never a marketing tier.
    pub gpu_arch: &'a str,
    /// A singleton, no-slots load.
    pub singleton: bool,
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
    /// Every weight and the KV cache live on the device (no host spill).
    pub fully_resident: bool,
    /// The text route the head would run under is an executable one (no
    /// adaptive KV, no vision-only route).
    pub text_route_supported: bool,
    /// A head matching the trunk resolved (bundled trailer or sidecar).
    pub head_resolved: bool,
    /// Another speculator (DFlash, DSpark) is attached or declared and loaded.
    pub other_speculator_attached: bool,
    /// Context the load admits without the head (from one capacity snapshot).
    pub cap_off: usize,
    /// Context the load admits with the head (same snapshot).
    pub cap_on: usize,
    /// `Some(n)` when the context was authored (`memory.max_seq` / `--max-seq`).
    pub authored_max_seq: Option<usize>,
    /// The with-head plan places every owner the no-head plan does.
    pub same_placement: bool,
    /// The with-head plan fits at all (no refusal, no spill).
    pub on_fits: bool,
}

/// The exact GPUs dense native MTP arms on under `auto` ([`dense_mtp_auto`]
/// step 1). The loader's route plan reads the same list to plan the native-MTP
/// route before the capacity verdict exists.
pub const DENSE_MTP_AUTO_GPUS: &[&str] = &["gfx1100", "gfx1201"];

/// Why [`dense_mtp_auto`] decided as it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseMtpAutoReason {
    /// Not dense arch 5.
    NotDenseQwen35,
    /// Not exact gfx1100 / gfx1201.
    UnsupportedGpu,
    /// Not a singleton single-GPU (PP/TP/EP) load.
    Topology,
    /// Weights or KV spill to the host.
    NotFullyResident,
    /// The text route cannot run the head.
    UnsupportedRoute,
    /// No head resolved.
    NoHead,
    /// Another speculator is attached.
    OtherSpeculator,
    /// Zero capacity (either plan) or arithmetic overflow.
    InvalidCapacity,
    /// The with-head plan does not fit.
    OnDoesNotFit,
    /// The with-head plan places owners differently.
    PlacementChanged,
    /// An authored context the head does not admit unchanged.
    AuthoredNotAdmitted,
    /// Omitted context: below both the retention line and the absolute floor.
    BelowRetentionAndFloor,
    /// Armed: the authored context is admitted unchanged with the head.
    AdmittedAuthored,
    /// Armed: the with-head context keeps at least 95 % of the no-head one.
    AdmittedRetention,
    /// Armed: the with-head context is at least [`DENSE_MTP_AUTO_CONTEXT_FLOOR`].
    AdmittedFloor,
}

impl DenseMtpAutoReason {
    /// Stable token for the load log line and `hipfire fit`.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NotDenseQwen35 => "not-dense-qwen35",
            Self::UnsupportedGpu => "unsupported-gpu",
            Self::Topology => "not-single-gpu",
            Self::NotFullyResident => "not-fully-resident",
            Self::UnsupportedRoute => "unsupported-route",
            Self::NoHead => "no-head",
            Self::OtherSpeculator => "other-speculator",
            Self::InvalidCapacity => "invalid-capacity",
            Self::OnDoesNotFit => "head-does-not-fit",
            Self::PlacementChanged => "placement-changed",
            Self::AuthoredNotAdmitted => "authored-context-not-admitted",
            Self::BelowRetentionAndFloor => "below-retention-and-floor",
            Self::AdmittedAuthored => "authored-context-admitted",
            Self::AdmittedRetention => "retention>=0.95",
            Self::AdmittedFloor => "context>=131072",
        }
    }
}

/// The one immutable `auto` verdict of a dense native-MTP load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMtpAutoDecision {
    pub attach: bool,
    pub reason: DenseMtpAutoReason,
    pub cap_off: usize,
    pub cap_on: usize,
}

impl DenseMtpAutoDecision {
    /// Tokens the with-head context sits above (positive) or below (negative)
    /// the 0.95 retention line `ceil(19 * cap_off / 20)`.
    pub fn margin_tokens(&self) -> i128 {
        let line = (19u128 * self.cap_off as u128).div_ceil(20);
        self.cap_on as i128 - line as i128
    }
}

/// Whether dense native MTP arms under `speculation.mtp = auto`.
///
/// Order: (1) arch-level qualification: dense arch 5, exact gfx1100/gfx1201,
/// singleton single-GPU, fully resident, a supported text route, a resolved
/// head and no other speculator; (2) invalid capacity or a with-head plan that
/// does not fit declines; (3) an authored context arms only when both plans
/// admit exactly that context and place the same owners (never clamped to fit);
/// (4) an omitted context arms when the with-head context keeps at least 95 %
/// of the no-head one (checked in `u128`, no float) or is at least
/// [`DENSE_MTP_AUTO_CONTEXT_FLOOR`] tokens, with identical placement. Pure: no
/// timing, no model, no weight format, no acceptance rate.
pub fn dense_mtp_auto(f: &DenseMtpAutoFacts<'_>) -> DenseMtpAutoDecision {
    use DenseMtpAutoReason as R;
    let decide = |attach: bool, reason: R| DenseMtpAutoDecision {
        attach,
        reason,
        cap_off: f.cap_off,
        cap_on: f.cap_on,
    };
    if f.arch_id != 5 {
        return decide(false, R::NotDenseQwen35);
    }
    if !DENSE_MTP_AUTO_GPUS.contains(&f.gpu_arch) {
        return decide(false, R::UnsupportedGpu);
    }
    if !f.singleton || f.pp != 1 || f.tp != 1 || f.ep != 1 {
        return decide(false, R::Topology);
    }
    if !f.fully_resident {
        return decide(false, R::NotFullyResident);
    }
    if !f.text_route_supported {
        return decide(false, R::UnsupportedRoute);
    }
    if !f.head_resolved {
        return decide(false, R::NoHead);
    }
    if f.other_speculator_attached {
        return decide(false, R::OtherSpeculator);
    }
    if f.cap_off == 0 {
        return decide(false, R::InvalidCapacity);
    }
    if !f.on_fits || f.cap_on == 0 {
        return decide(false, R::OnDoesNotFit);
    }
    if !f.same_placement {
        return decide(false, R::PlacementChanged);
    }
    if let Some(authored) = f.authored_max_seq {
        return if f.cap_off == authored && f.cap_on == authored {
            decide(true, R::AdmittedAuthored)
        } else {
            decide(false, R::AuthoredNotAdmitted)
        };
    }
    if 20u128 * f.cap_on as u128 >= 19u128 * f.cap_off as u128 {
        decide(true, R::AdmittedRetention)
    } else if f.cap_on >= DENSE_MTP_AUTO_CONTEXT_FLOOR {
        decide(true, R::AdmittedFloor)
    } else {
        decide(false, R::BelowRetentionAndFloor)
    }
}

/// The one load-log line of a dense `auto` verdict, `basis` naming where the
/// capacities came from (`measured` after load; `planned` from `hipfire fit`).
pub fn dense_mtp_auto_log_line(
    d: &DenseMtpAutoDecision,
    authored: Option<usize>,
    basis: &str,
) -> String {
    format!(
        "mtp auto: {} reason={} cap_off={} cap_on={} authored={} margin_tokens={} basis={}",
        if d.attach { "attach" } else { "decline" },
        d.reason.label(),
        d.cap_off,
        d.cap_on,
        authored.map_or("omitted".to_string(), |n| n.to_string()),
        d.margin_tokens(),
        basis
    )
}

fn is_qwen35_family(model_arch: &str) -> bool {
    model_arch.eq_ignore_ascii_case("qwen3_5") || model_arch.eq_ignore_ascii_case("qwen3_5_moe")
}

/// Decode weight formats whose gfx1100 Qwen3.5-dense retained PM4 tape has its
/// own served greedy-identity and decode-timing receipts (REDLINE.md §3):
/// MQ3G256V2 (qt49), symmetric and asymmetric, and symmetric MQ4G256V2 (qt44,
/// the default 27B `mq4-xts`). Asymmetric MQ4G256V2 has no receipts yet. A
/// format joins only with both receipts.
pub fn gfx1100_dense_weights_qualified(weights: DenseWeights) -> bool {
    matches!(
        weights,
        DenseWeights::Mq3V2 { .. } | DenseWeights::Mq4V2 { symmetric: true }
    )
}

/// Weight formats with a DFlash-paired retained PM4 receipt on gfx1201:
/// symmetric MQ4G256V2 only. DFlash + MQ3G256V2 or asymmetric MQ4G256V2 has
/// none there (gfx1100 admits DFlash for every
/// [`gfx1100_dense_weights_qualified`] format).
pub fn dflash_pm4_weights_qualified(weights: DenseWeights) -> bool {
    matches!(weights, DenseWeights::Mq4V2 { symmetric: true })
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
    /// `memory.embed_host` / `HIPFIRE_EMBED_HOST` (default on): a Qwen3.5-family
    /// load on a discrete GPU places its untied passthrough token embedding in
    /// pinned host RAM. Read by the loader and the capacity projection.
    pub embed_host: bool,
    /// `generation.engine_loop` / `HIPFIRE_ENGINE_LOOP` = `engine`: allowlisted
    /// routes run on the shared engine loop (G12 PoC). Default `legacy`.
    pub engine_loop: bool,
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
            embed_host: value("HIPFIRE_EMBED_HOST").as_deref() != Some("0"),
            engine_loop: value("HIPFIRE_ENGINE_LOOP").as_deref() == Some("engine"),
        }
    }
}

#[cfg(test)]
mod legacy_spelling_golden;

#[cfg(test)]
mod tests {
    use super::{
        gfx1100_dense_weights_qualified, mq4r_redline_default as mq4r_with_route,
        retained_redline_default as redline_with_route, AttachedDraft as D, RuntimeConfig,
    };
    use crate::route_sku::{DenseWeights, RouteSku};
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};

    /// Every attached-draft kind, for truth tables.
    const KINDS: [D; 4] = [D::None, D::Dflash, D::NativeMtp, D::Other];

    /// The SKU a fixture name stands for. Production resolves the SKU from the
    /// HFQ metadata or the registry card and never reads a name; these tests
    /// spell the already-resolved SKU through the fixture file name so each
    /// case stays one line.
    fn fixture_route(model_path: &str) -> Option<RouteSku> {
        let lower = model_path.to_ascii_lowercase();
        [
            (".mq4rs", RouteSku::Mq4rs),
            (".mq4r", RouteSku::Mq4r),
            (".mq2r", RouteSku::Mq2r),
        ]
        .into_iter()
        .find_map(|(suffix, sku)| lower.ends_with(suffix).then_some(sku))
    }

    fn mq4r_redline_default(gpu_arch: &str, model_path: &str, pp: usize, tp: usize) -> bool {
        mq4r_with_route(gpu_arch, fixture_route(model_path), pp, tp)
    }

    fn redline_with_ep(
        gpu_arch: &str,
        model_arch: &str,
        model_path: &str,
        pp: usize,
        tp: usize,
        ep: usize,
        draft: D,
    ) -> bool {
        redline_with_route(
            gpu_arch,
            model_arch,
            fixture_route(model_path),
            None,
            pp,
            tp,
            ep,
            draft,
        )
    }

    /// The pre-EP call shape: every case below is a single-rank (`ep == 1`) load.
    fn retained_redline_default(
        gpu_arch: &str,
        model_arch: &str,
        model_path: &str,
        pp: usize,
        tp: usize,
        draft: D,
    ) -> bool {
        redline_with_ep(gpu_arch, model_arch, model_path, pp, tp, 1, draft)
    }

    #[test]
    fn route_comes_from_the_sku_never_from_the_file_name() {
        // A path named like every removed extension but with no resolved SKU
        // gets no automatic route; a SKU on an unrelated name does.
        for name in ["/m/x.mq4r", "/m/x.mq4rs", "/m/x.mq2r", "/m/X.MQ4R"] {
            for (arch, model_arch) in [
                ("gfx1100", "qwen3_6_moe"),
                ("gfx1151", "deepseek4"),
                ("gfx1201", "qwen3_5_moe"),
            ] {
                assert!(
                    !redline_with_route(arch, model_arch, None, None, 1, 1, 1, D::None),
                    "{name} with no resolved SKU on {arch}"
                );
            }
            assert!(!mq4r_with_route("gfx1201", None, 1, 1), "{name}");
        }
        assert!(redline_with_route(
            "gfx1100",
            "qwen3_6_moe",
            Some(RouteSku::Mq4r),
            None,
            1,
            1,
            1,
            D::None
        ));
        assert!(redline_with_route(
            "gfx1201",
            "qwen3_5_moe",
            Some(RouteSku::Mq4rs),
            None,
            1,
            1,
            1,
            D::None
        ));
        assert!(redline_with_route(
            "gfx1151",
            "deepseek4",
            Some(RouteSku::Mq2r),
            None,
            1,
            1,
            1,
            D::None
        ));
        // A SKU only opens its own predicate.
        assert!(!mq4r_with_route("gfx1201", Some(RouteSku::Mq4rs), 1, 1));
        assert!(!mq4r_with_route("gfx1201", Some(RouteSku::Mq2r), 1, 1));
        assert!(!redline_with_route(
            "gfx1151",
            "qwen3_5_moe",
            Some(RouteSku::Mq4rs),
            None,
            1,
            1,
            1,
            D::None
        ));
    }

    #[test]
    fn expert_parallel_never_admits_retained_redline() {
        let fn_xts = "/models/qwen3.8-flash-next-gptq3.mq4";
        assert!(redline_with_ep(
            "gfx1201",
            "qwen4",
            fn_xts,
            1,
            1,
            1,
            D::None
        ));
        for ep in [2, 4] {
            for draft in KINDS {
                assert!(!redline_with_ep(
                    "gfx1201", "qwen4", fn_xts, 1, 1, ep, draft
                ));
                assert!(!redline_with_ep(
                    "gfx1201", "qwen3_5", fn_xts, 1, 1, ep, draft
                ));
                assert!(!redline_with_ep(
                    "gfx1201",
                    "qwen4",
                    "/models/fn.mq4r",
                    1,
                    1,
                    ep,
                    draft
                ));
            }
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
    fn a3b_mq4rs_takes_retained_redline_on_gfx1100_and_gfx1201_only() {
        let f = "/models/qwen3.6-35b-a3b-d53.mq4rs";
        for arch in ["gfx1100", "gfx1201"] {
            assert!(retained_redline_default(
                arch,
                "qwen3_5_moe",
                f,
                1,
                1,
                D::None
            ));
            assert!(retained_redline_default(
                arch,
                "qwen3_5_moe",
                "/models/A3B.MQ4RS",
                1,
                1,
                D::None
            ));
            // any drafter (DFlash included), multi-GPU, EP, other families,
            // other SKUs
            for draft in [D::Dflash, D::NativeMtp, D::Other] {
                assert!(!retained_redline_default(
                    arch,
                    "qwen3_5_moe",
                    f,
                    1,
                    1,
                    draft
                ));
            }
            assert!(!retained_redline_default(
                arch,
                "qwen3_5_moe",
                f,
                2,
                1,
                D::None
            ));
            assert!(!retained_redline_default(
                arch,
                "qwen3_5_moe",
                f,
                1,
                2,
                D::None
            ));
            assert!(!redline_with_ep(arch, "qwen3_5_moe", f, 1, 1, 2, D::None));
            assert!(!retained_redline_default(arch, "qwen3", f, 1, 1, D::None));
            assert!(!retained_redline_default(
                arch,
                "qwen3_5_moe",
                "/models/a3b.mq4",
                1,
                1,
                D::None
            ));
        }
        for arch in ["gfx1200", "gfx1101", "gfx1151", "gfx942"] {
            assert!(
                !retained_redline_default(arch, "qwen3_5_moe", f, 1, 1, D::None),
                "{arch} must stay on HIP"
            );
        }
    }

    #[test]
    fn a3b_mq4rs_stays_on_hip_on_gfx1151() {
        // Halo has no A3B MQ4RS retained receipts: the HIP graph keeps it.
        assert!(!redline_with_route(
            "gfx1151",
            "qwen3_5_moe",
            Some(RouteSku::Mq4rs),
            None,
            1,
            1,
            1,
            D::None
        ));
        assert!(!super::A3B_MQ4RS_RETAINED_ARCHES.contains(&"gfx1151"));
    }

    #[test]
    fn gfx1100_qwen35_dense_admits_qualified_weight_formats_and_nothing_wider() {
        let gfx1100 = |model_arch: &str, weights, pp, tp, ep, draft| {
            redline_with_route("gfx1100", model_arch, None, weights, pp, tp, ep, draft)
        };
        let qualified = [
            DenseWeights::Mq3V2 { symmetric: false },
            DenseWeights::Mq3V2 { symmetric: true },
            DenseWeights::Mq4V2 { symmetric: true },
        ];
        for w in qualified {
            assert!(gfx1100_dense_weights_qualified(w));
            // No drafter: the qualified plain-AR tape. A DFlash drafter is
            // admitted for the same formats (DFlash + MQ3 receipt on gfx1100).
            assert!(gfx1100("qwen3_5", Some(w), 1, 1, 1, D::None), "{w:?}");
            assert!(gfx1100("qwen3_5", Some(w), 1, 1, 1, D::Dflash), "{w:?}");
            // Any other drafter, multi-GPU and expert parallelism keep the HIP
            // graph.
            assert!(!gfx1100("qwen3_5", Some(w), 1, 1, 1, D::NativeMtp), "{w:?}");
            assert!(!gfx1100("qwen3_5", Some(w), 1, 1, 1, D::Other), "{w:?}");
            assert!(!gfx1100("qwen3_5", Some(w), 2, 1, 1, D::None), "{w:?}");
            assert!(!gfx1100("qwen3_5", Some(w), 1, 2, 1, D::None), "{w:?}");
            assert!(!gfx1100("qwen3_5", Some(w), 1, 1, 2, D::None), "{w:?}");
            // Other GPU arches (gfx1201 admits every Qwen3.5 dense format on
            // its own rule) and other families stay on HIP.
            for arch in ["gfx1030", "gfx1101", "gfx1151", "gfx1200", "gfx942"] {
                assert!(
                    !redline_with_route(arch, "qwen3_5", None, Some(w), 1, 1, 1, D::None),
                    "{arch} must stay on HIP for {w:?}"
                );
            }
            for model_arch in ["qwen3_5_moe", "qwen3", "qwen4", "gemma4", "deepseek4"] {
                assert!(
                    !gfx1100(model_arch, Some(w), 1, 1, 1, D::None),
                    "{model_arch} {w:?} must stay on HIP on gfx1100"
                );
            }
        }
        // Unqualified (no receipts) or unknown formats keep the HIP graph.
        let w = DenseWeights::Mq4V2 { symmetric: false };
        assert!(!gfx1100_dense_weights_qualified(w));
        assert!(!gfx1100("qwen3_5", Some(w), 1, 1, 1, D::None), "{w:?}");
        assert!(!gfx1100("qwen3_5", Some(w), 1, 1, 1, D::Dflash), "{w:?}");
        assert!(!gfx1100("qwen3_5", None, 1, 1, 1, D::None));
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
            for draft in KINDS {
                assert!(
                    !retained_redline_default(
                        gpu_arch,
                        "muse_glimmer",
                        "/models/muse-glimmer-30b.mq4r",
                        1,
                        1,
                        draft,
                    ),
                    "glimmer must stay on HIP by default on {gpu_arch} (drafter {draft:?})"
                );
            }
            // The carve-out is glimmer-scoped, not a global MQ4R retreat.
            assert!(
                retained_redline_default(
                    gpu_arch,
                    "qwen3_6_moe",
                    "/models/qwen3.6-35b-a3b.mq4r",
                    1,
                    1,
                    D::None,
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
                for draft in KINDS {
                    assert!(
                        !retained_redline_default(gpu_arch, "gemma4", path, 1, 1, draft),
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
            D::None,
        ));
    }

    #[test]
    fn deepseek4_mq2r_redline_default_requires_gfx1151_ar() {
        let mq2r = "/models/deepseek-v4-flash-0731.mq2r";
        assert!(retained_redline_default(
            "gfx1151",
            "deepseek4",
            mq2r,
            1,
            1,
            D::None
        ));
        for draft in [D::Dflash, D::NativeMtp, D::Other] {
            assert!(!retained_redline_default(
                "gfx1151",
                "deepseek4",
                mq2r,
                1,
                1,
                draft
            ));
        }
        assert!(!retained_redline_default(
            "gfx1100",
            "deepseek4",
            mq2r,
            1,
            1,
            D::None
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "qwen3_5_moe",
            mq2r,
            1,
            1,
            D::None
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            mq2r,
            2,
            1,
            D::None
        ));
        assert!(!retained_redline_default(
            "gfx1151",
            "deepseek4",
            "/models/deepseek-v4-flash-0731.mq2",
            1,
            1,
            D::None,
        ));
    }

    #[test]
    fn qwen35_dense_redline_default_requires_gfx1201_single_gpu_ar() {
        let h2 = "/models/h2.group-alpha-refit.hfq";
        // No drafter, any weight format; a DFlash drafter only with symmetric
        // MQ4G256V2 weights (see `attached_draft_admission_truth_table`).
        assert!(retained_redline_default(
            "gfx1201",
            "qwen3_5",
            h2,
            1,
            1,
            D::None
        ));
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen3_5",
            h2,
            1,
            1,
            D::Dflash
        ));
        // Any other drafter keeps the model on its speculative path.
        for draft in [D::NativeMtp, D::Other] {
            assert!(!retained_redline_default(
                "gfx1201", "qwen3_5", h2, 1, 1, draft
            ));
        }
        // Pipeline / tensor parallel stay on HIP.
        for draft in KINDS {
            assert!(!retained_redline_default(
                "gfx1201", "qwen3_5", h2, 2, 1, draft
            ));
            assert!(!retained_redline_default(
                "gfx1201", "qwen3_5", h2, 1, 2, draft
            ));
        }
        // Exact gfx1201 only; other dense archs and the MoE sibling stay on HIP.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1200"] {
            assert!(!retained_redline_default(
                gpu_arch,
                "qwen3_5",
                h2,
                1,
                1,
                D::None
            ));
        }
        for model_arch in ["qwen3_5_moe", "qwen3", "qwen2", "gemma4"] {
            for draft in KINDS {
                assert!(!retained_redline_default(
                    "gfx1201", model_arch, h2, 1, 1, draft
                ));
            }
        }
    }

    #[test]
    fn qwen4_redline_default_requires_gfx1201_single_gpu_without_mtp() {
        let fn_xts = "/models/qwen3.8-flash-next-gptq3.mq4";
        assert!(retained_redline_default(
            "gfx1201",
            "qwen4",
            fn_xts,
            1,
            1,
            D::None
        ));
        // The native MTP head is a drafter: an MTP load keeps its spec path.
        // The DFlash admission is Qwen3.5-dense only.
        for draft in [D::Dflash, D::NativeMtp, D::Other] {
            assert!(!retained_redline_default(
                "gfx1201", "qwen4", fn_xts, 1, 1, draft
            ));
        }
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen4",
            fn_xts,
            2,
            1,
            D::None
        ));
        assert!(!retained_redline_default(
            "gfx1201",
            "qwen4",
            fn_xts,
            1,
            2,
            D::None
        ));
        // A plain `.mq4` Flash-Next stays on HIP off gfx1201.
        for gpu_arch in ["gfx1100", "gfx1151", "gfx1200"] {
            assert!(!retained_redline_default(
                gpu_arch,
                "qwen4",
                fn_xts,
                1,
                1,
                D::None
            ));
        }
        // HIPF-254: a Flash-Next `.mq4r` stays on HIP on gfx1100/gfx1151; the
        // extension route still admits it on gfx1201 and admits other archs.
        let fn_mq4r = "/models/qwen3.8-flash-next.mq4r";
        for gpu_arch in ["gfx1100", "gfx1151"] {
            assert!(!retained_redline_default(
                gpu_arch,
                "qwen4",
                fn_mq4r,
                1,
                1,
                D::None
            ));
            assert!(retained_redline_default(
                gpu_arch,
                "qwen3_5",
                "/models/m.mq4r",
                1,
                1,
                D::None
            ));
        }
        assert!(retained_redline_default(
            "gfx1201",
            "qwen4",
            fn_mq4r,
            1,
            1,
            D::None
        ));
    }

    /// Every [`D`] kind against every admitted cell (single GPU): DFlash joins
    /// exactly the two Qwen3.5-dense cells (gfx1100: every qualified format,
    /// symmetric and asymmetric MQ3 included; gfx1201: symmetric MQ4 only, so
    /// gfx1201 + Mq3V2 / asymmetric MQ4 / unknown weights admits plain AR
    /// only), the MQ4R SKU rule ignores the drafter as before, and every other
    /// cell admits no drafter at all.
    #[test]
    fn attached_draft_admission_truth_table() {
        let mq4 = Some(DenseWeights::Mq4V2 { symmetric: true });
        let mq3 = Some(DenseWeights::Mq3V2 { symmetric: true });
        let mq3_asym = Some(DenseWeights::Mq3V2 { symmetric: false });
        let unqualified = Some(DenseWeights::Mq4V2 { symmetric: false });
        let cells: [(&str, &str, Option<RouteSku>, Option<DenseWeights>, &[D]); 16] = [
            ("gfx1201", "qwen3_5", None, mq4, &[D::None, D::Dflash]),
            ("gfx1201", "qwen3_5", None, None, &[D::None]),
            ("gfx1201", "qwen3_5", None, mq3, &[D::None]),
            ("gfx1201", "qwen3_5", None, mq3_asym, &[D::None]),
            ("gfx1201", "qwen3_5", None, unqualified, &[D::None]),
            ("gfx1100", "qwen3_5", None, mq4, &[D::None, D::Dflash]),
            ("gfx1100", "qwen3_5", None, mq3, &[D::None, D::Dflash]),
            ("gfx1100", "qwen3_5", None, mq3_asym, &[D::None, D::Dflash]),
            ("gfx1100", "qwen3_5", None, unqualified, &[]),
            ("gfx1151", "qwen3_5", None, mq4, &[]),
            ("gfx1201", "qwen4", None, None, &[D::None]),
            ("gfx1201", "qwen3_6_moe", Some(RouteSku::Mq4r), None, &KINDS),
            (
                "gfx1201",
                "qwen3_5_moe",
                Some(RouteSku::Mq4rs),
                None,
                &[D::None],
            ),
            (
                "gfx1100",
                "qwen3_5_moe",
                Some(RouteSku::Mq4rs),
                None,
                &[D::None],
            ),
            (
                "gfx1151",
                "deepseek4",
                Some(RouteSku::Mq2r),
                None,
                &[D::None],
            ),
            ("gfx1201", "gemma4", Some(RouteSku::Mq4r), None, &[]),
        ];
        for (gpu_arch, model_arch, route, weights, admitted) in cells {
            for draft in KINDS {
                assert_eq!(
                    redline_with_route(gpu_arch, model_arch, route, weights, 1, 1, 1, draft),
                    admitted.contains(&draft),
                    "{gpu_arch} {model_arch} {route:?} {weights:?} {draft:?}"
                );
            }
        }
    }

    #[test]
    fn attached_draft_kind_comes_from_the_speculator() {
        assert_eq!(D::from_speculator(None, false), D::None);
        assert_eq!(D::from_speculator(None, true), D::None);
        assert_eq!(D::from_speculator(Some("dflash"), false), D::Dflash);
        // A DFlash load serving through the continuous-batching DFlash lanes
        // keeps the HIP route (their gfx1201 save-free verify).
        assert_eq!(D::from_speculator(Some("dflash"), true), D::Other);
        assert_eq!(D::from_speculator(Some("mtp"), false), D::NativeMtp);
        for name in ["dspark", "eagle", "dflash2", "ngram", "spec"] {
            assert_eq!(D::from_speculator(Some(name), false), D::Other, "{name}");
        }
        assert_eq!(D::Dflash.label(), "dflash");
        assert_eq!(D::None.label(), "off");
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

#[cfg(test)]
mod dense_mtp_auto_tests {
    use super::*;

    fn base() -> DenseMtpAutoFacts<'static> {
        DenseMtpAutoFacts {
            arch_id: 5,
            gpu_arch: "gfx1201",
            singleton: true,
            pp: 1,
            tp: 1,
            ep: 1,
            fully_resident: true,
            text_route_supported: true,
            head_resolved: true,
            other_speculator_attached: false,
            cap_off: 262_144,
            cap_on: 262_144,
            authored_max_seq: None,
            same_placement: true,
            on_fits: true,
        }
    }

    fn with(edit: impl FnOnce(&mut DenseMtpAutoFacts<'static>)) -> DenseMtpAutoFacts<'static> {
        let mut facts = base();
        edit(&mut facts);
        facts
    }

    fn caps(off: usize, on: usize) -> DenseMtpAutoFacts<'static> {
        with(|f| (f.cap_off, f.cap_on) = (off, on))
    }

    fn verdict(f: DenseMtpAutoFacts<'_>) -> (bool, DenseMtpAutoReason) {
        let d = dense_mtp_auto(&f);
        (d.attach, d.reason)
    }

    #[track_caller]
    fn check(f: DenseMtpAutoFacts<'_>, attach: bool, reason: DenseMtpAutoReason) {
        assert_eq!(verdict(f), (attach, reason));
    }

    #[test]
    fn arch_level_qualification_declines_with_a_typed_reason() {
        use DenseMtpAutoReason as R;
        type Patch = fn(&mut DenseMtpAutoFacts<'static>);
        let cases: [(Patch, R); 14] = [
            (|f| f.arch_id = 4, R::NotDenseQwen35),
            (|f| f.arch_id = 6, R::NotDenseQwen35),
            (|f| f.arch_id = 16, R::NotDenseQwen35),
            (|f| f.gpu_arch = "gfx1101", R::UnsupportedGpu),
            (|f| f.gpu_arch = "gfx1151", R::UnsupportedGpu),
            (|f| f.gpu_arch = "gfx942", R::UnsupportedGpu),
            (|f| f.singleton = false, R::Topology),
            (|f| f.pp = 2, R::Topology),
            (|f| f.tp = 2, R::Topology),
            (|f| f.ep = 2, R::Topology),
            (|f| f.fully_resident = false, R::NotFullyResident),
            (|f| f.text_route_supported = false, R::UnsupportedRoute),
            (|f| f.head_resolved = false, R::NoHead),
            (|f| f.other_speculator_attached = true, R::OtherSpeculator),
        ];
        for (patch, want) in cases {
            check(with(patch), false, want);
        }
        for arch in ["gfx1100", "gfx1201"] {
            assert!(verdict(with(|f| f.gpu_arch = arch)).0);
        }
    }

    #[test]
    fn invalid_capacity_and_non_fitting_head_decline() {
        use DenseMtpAutoReason as R;
        check(with(|f| f.cap_off = 0), false, R::InvalidCapacity);
        check(with(|f| f.on_fits = false), false, R::OnDoesNotFit);
        check(with(|f| f.cap_on = 0), false, R::OnDoesNotFit);
        let moved = with(|f| f.same_placement = false);
        check(moved, false, R::PlacementChanged);
        // usize::MAX capacities cannot overflow the cross-multiplication.
        assert!(dense_mtp_auto(&caps(usize::MAX, usize::MAX)).attach);
    }

    #[test]
    fn retention_line_is_exact_at_the_boundary() {
        use DenseMtpAutoReason as R;
        // 95 % of 120,000 is exactly 114,000 (below the 131,072 floor).
        check(caps(120_000, 113_999), false, R::BelowRetentionAndFloor);
        check(caps(120_000, 114_000), true, R::AdmittedRetention);
        check(caps(120_000, 114_001), true, R::AdmittedRetention);
        // Above the floor a shortfall against the line still arms.
        check(caps(200_000, 189_999), true, R::AdmittedFloor);
        check(caps(200_000, 190_000), true, R::AdmittedRetention);
        // Non-multiple of 20: ceil(19 * 262,143 / 20) = 249,036.
        check(caps(262_143, 249_035), true, R::AdmittedFloor);
        check(caps(262_143, 249_036), true, R::AdmittedRetention);
        // ceil(19 * 100,001 / 20) = 95,001.
        check(caps(100_001, 95_000), false, R::BelowRetentionAndFloor);
        check(caps(100_001, 95_001), true, R::AdmittedRetention);
        assert_eq!(dense_mtp_auto(&caps(120_000, 114_000)).margin_tokens(), 0);
        assert_eq!(dense_mtp_auto(&caps(120_000, 113_999)).margin_tokens(), -1);
    }

    #[test]
    fn absolute_floor_arms_a_context_the_retention_rule_declines() {
        use DenseMtpAutoReason as R;
        assert_eq!(DENSE_MTP_AUTO_CONTEXT_FLOOR, 131_072);
        // gfx1100 mq4-xts at omitted max_seq measured 246,158 -> 188,897 (0.767).
        let gfx1100 = |on| DenseMtpAutoFacts {
            gpu_arch: "gfx1100",
            ..caps(246_158, on)
        };
        check(gfx1100(188_897), true, R::AdmittedFloor);
        check(gfx1100(131_072), true, R::AdmittedFloor);
        check(gfx1100(131_071), false, R::BelowRetentionAndFloor);
        // The same ratio at a tenth of the context stays AR.
        check(caps(24_615, 18_889), false, R::BelowRetentionAndFloor);
    }

    #[test]
    fn authored_context_arms_only_when_admitted_unchanged_by_both_plans() {
        use DenseMtpAutoReason as R;
        let authored = |n, off, on| DenseMtpAutoFacts {
            authored_max_seq: Some(n),
            ..caps(off, on)
        };
        let (n, big) = (32_768, 262_144);
        check(authored(n, n, n), true, R::AdmittedAuthored);
        // The head would shrink the authored context: never clamped to fit.
        check(authored(n, n, 20_000), false, R::AuthoredNotAdmitted);
        // The no-head load itself does not admit it.
        check(authored(n, 20_000, 20_000), false, R::AuthoredNotAdmitted);
        // The floor and the retention line do not rescue a shrunk authored context.
        check(authored(big, big, 200_000), false, R::AuthoredNotAdmitted);
        let refused = DenseMtpAutoFacts {
            on_fits: false,
            ..authored(n, n, 0)
        };
        check(refused, false, R::OnDoesNotFit);
    }

    #[test]
    fn log_line_carries_reason_capacities_margin_and_basis() {
        let d = dense_mtp_auto(&caps(200_000, 190_500));
        assert_eq!(
            dense_mtp_auto_log_line(&d, None, "measured"),
            "mtp auto: attach reason=retention>=0.95 cap_off=200000 cap_on=190500 authored=omitted margin_tokens=500 basis=measured"
        );
        let d = dense_mtp_auto(&caps(100_000, 50_000));
        assert!(dense_mtp_auto_log_line(&d, Some(8192), "planned").starts_with(
            "mtp auto: decline reason=below-retention-and-floor cap_off=100000 cap_on=50000 authored=8192 margin_tokens=-45000 basis=planned"
        ));
    }
}
