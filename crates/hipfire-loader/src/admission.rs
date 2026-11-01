// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Source-aware admission (device-mesh G2).
//!
//! Classifies a retained source once and decides one effective topology BEFORE
//! any destructive side effect (prior-model teardown, VMM init, remap, GPU
//! allocation, carrier entry, collective creation). The load route consumes the
//! [`SourceAdmission`]'s already-open [`ModelSource`] — never re-opening or
//! re-classifying the path. A refusal here leaves whatever model is currently
//! loaded untouched: no teardown, no allocation, no carrier entry, no cache
//! mutation.

use crate::Carrier;
pub use hipfire_arch_qwen4::admission::{
    qwen4_pp_auto_fit, qwen4_pp_edge_max_bytes, qwen4_pp_rank_lines, Qwen4PpAutoFit,
    Qwen4PpBudgetInputs, Qwen4PpRankFit, QWEN4_PP_FIRST_BAND_MIN_LAYERS, QWEN4_PP_RESERVE_BYTES,
};
use hipfire_arch_qwen4::config::{Qwen4Config, ARCH_ID as QWEN4_ARCH_ID};
use hipfire_arch_qwen4::{InputModality, Qwen4Capabilities};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode::{KvMode, VMode};
pub use hipfire_runtime::loader_api::BatchLoadRequest;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::mesh_placement::{
    pp_rank_order, resolve_edge_transports, AdmittedMesh, AdmittedRank, BandPlacement,
    EdgeTransportPolicy,
};
fn qwen4_vision_tensor_name(name: &str) -> bool {
    [
        "model.visual.",
        "model.vision_tower.",
        "model.vision_projection.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "visual.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

/// Omission requests automatic VMM where the actual owner can allocate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBackendRequest {
    Automatic,
    Explicit(KvBackend),
}

impl KvBackendRequest {
    pub fn from_override(raw: Option<&str>) -> Result<Self, String> {
        raw.filter(|s| !s.is_empty())
            .map(|s| s.parse().map(Self::Explicit).map_err(|e| format!("{e}")))
            .unwrap_or(Ok(Self::Automatic))
    }
}

/// The sampled per-load switch only changes implicit Qwen defaults.
pub fn qwen_default_q8_enabled() -> bool {
    !matches!(
        hipfire_config::developer_var("HIPFIRE_QWEN_KV_DEFAULT_Q8")
            .ok()
            .as_deref(),
        Some("0")
    )
}

/// Effective Qwen3.5 V axis: a typed `kv_v` wins, else developer
/// `HIPFIRE_KV_V` (empty / `q8` = neutral). Resolved once in admission so
/// VMM sizing and the carrier load see the same pair. The env fallback only
/// applies to the Qwen3.5 carrier (arch 5/6) on single-rank tensor topology;
/// other owners never honored it.
fn resolve_kv_v(typed: Option<&str>, arch_id: u32, tp: usize) -> Option<String> {
    let env = hipfire_config::developer_var("HIPFIRE_KV_V").ok();
    resolve_kv_v_from(typed, env.as_deref(), arch_id, tp).map(str::to_string)
}

fn resolve_kv_v_from<'a>(
    typed: Option<&'a str>,
    env: Option<&'a str>,
    arch_id: u32,
    tp: usize,
) -> Option<&'a str> {
    if let Some(v) = typed.filter(|s| !s.is_empty()) {
        return Some(v);
    }
    if !matches!(arch_id, 5 | 6) || tp > 1 {
        return None;
    }
    env.filter(|v| !matches!(*v, "" | "q8"))
}

/// Single authoritative operator warning for an admitted legacy trunk.
pub fn legacy_warning(backend: KvBackend, reason: Option<&str>) -> Option<String> {
    if backend != KvBackend::Legacy {
        return None;
    }
    Some(match reason {
        Some(reason) => format!(
            "WARNING HIPFIRE_KV_BACKEND=legacy: automatic VMM selection unavailable ({reason}); using legacy KV storage. Inspect before benchmarking or filing a PR."
        ),
        None => "WARNING HIPFIRE_KV_BACKEND=legacy: explicitly selected legacy KV storage (--kv-backend legacy or memory.kv_backend); inspect before benchmarking or filing a PR.".to_string(),
    })
}

/// Admitted gfx11 Qwen3.5-family VMM KV combinations.
///
/// Q8 retains the existing gfx1100/gfx1151 and multi-rank admission. Modern
/// FWHT K tiers paired with Q8/Lloyd V use the same static row layouts as their
/// legacy-storage owners, but are admitted only on exact gfx1100 TP1: extending
/// this predicate must not accidentally promote TP2 or gfx1151 without separate
/// lifecycle proof. Deprecated Givens `legacy-asym*` remains fail-closed.
fn qwen35_gfx11_vmm_supported(
    gpu_arch: &str,
    arch_id: u32,
    tp: usize,
    mode_raw: &str,
    k_raw: Option<&str>,
    v_raw: Option<&str>,
) -> bool {
    if !matches!(gpu_arch, "gfx1100" | "gfx1151") || !matches!(arch_id, 5 | 6) {
        return false;
    }
    let requested_k = k_raw
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(mode_raw)
        .trim();
    if matches!(requested_k, "" | "auto" | "q8") {
        return v_raw.is_none_or(|value| matches!(value.trim(), "" | "q8"));
    }
    if gpu_arch != "gfx1100" || tp != 1 {
        return false;
    }
    let Ok(k) = hipfire_runtime::kv_mode::parse_qwen_k_name(requested_k) else {
        return false;
    };
    let requested_v = v_raw
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("q8");
    let Ok(v) = hipfire_runtime::kv_mode::parse_qwen_v_name(requested_v) else {
        return false;
    };
    matches!(k, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
        && matches!(v, VMode::Q8 | VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4)
}

/// An FWHT K override of the Q8/default mode has no certified gfx1151 VMM path.
fn qwen35_gfx1151_fwht_k_override(
    gpu_arch: &str,
    arch_id: u32,
    mode_raw: &str,
    k_raw: Option<&str>,
) -> bool {
    gpu_arch == "gfx1151"
        && matches!(arch_id, 5 | 6)
        && matches!(mode_raw.trim(), "" | "auto" | "q8")
        && k_raw.is_some_and(|raw| {
            matches!(
                hipfire_runtime::kv_mode::parse_qwen_k_name(raw),
                Ok(KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
            )
        })
}

#[derive(Clone, Copy)]
pub struct KvBackendHints<'a> {
    pub kv_mode: Option<&'a str>,
    pub kv_k: Option<&'a str>,
    pub kv_v: Option<&'a str>,
    pub qwen_default_q8: bool,
    pub kv_adaptive: Option<&'a str>,
    pub cask: Option<&'a hipfire_runtime::loader_api::CaskConfig>,
    pub deepseek4_heterogeneous: bool,
    pub vmm_runtime_available: bool,
    /// Free device bytes before this load, queried from HIP at admission.
    pub free_vram_bytes: Option<usize>,
}

impl KvBackendHints<'_> {
    pub const fn without_device() -> Self {
        Self {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: true,
            kv_adaptive: None,
            cask: None,
            deepseek4_heterogeneous: false,
            vmm_runtime_available: false,
            free_vram_bytes: None,
        }
    }
}

/// The one effective topology admitted for a load. `tp>1` (expert-parallel) and
/// `pp>1` (pipeline-parallel) are mutually exclusive; both default to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveTopology {
    Single,
    Pipeline(usize),
    Expert(usize),
}

/// Optional load-time controls that must be rejected at Qwen4 admission
/// before any previous model teardown or GPU allocation.
#[derive(Clone, Copy, Default)]
pub struct SourceAdmissionOptions {
    pub spec: SpecLoadCfg,
    /// True only when `memory.kv_adaptive` is an active policy. The CLI
    /// serializes its ordinary `off` schema default as a nonempty string.
    pub kv_adaptive: bool,
    pub eagle_drafter: bool,
    pub cask: bool,
    pub state_quant: bool,
    pub non_single_compute: bool,
    pub expert_count_override: bool,
    pub pflash: bool,
}

/// Return whether a per-load adaptive-KV value requests the active controller.
/// The CLI schema default is `Some("off")`, which must remain ordinary AR.
/// Empty values are also inactive because daemon normalization drops them.
pub fn qwen4_kv_adaptive_requested(value: Option<&str>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "off")
}

/// Whether a Qwen4 load attaches its native MTP head. The executable HFQM
/// manifest always carries the validated head, and Qwen4 declares a measured
/// AR floor ([`hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR`]), so `auto`
/// (`None`) attaches it and `Some(false)` keeps AR. The retained Redline default (a `.mq4r` name)
/// owns the load instead: its tape is the single-row AR forward, which no MTP
/// window runs, so `auto` stays AR there and an explicit `Some(true)` refuses.
/// Under expert parallelism (`ep` > 1) and pipeline parallelism (`pp` > 1) the
/// head is attached only on an explicit request: `Some(true)` admits it (EP:
/// the drafter is replicated on every rank; PP: the output band owns the one
/// head), while `auto` (`None`) and `Some(false)` stay AR, so the EP/PP
/// default and flag-off bytes are unchanged.
pub(crate) fn qwen4_native_mtp(
    spec: SpecLoadCfg,
    gpu_arch: &str,
    path: &str,
    pp: usize,
    tp: usize,
    ep: usize,
) -> Result<bool, String> {
    use hipfire_config::load_summary::{note, Facet};
    if pp > 1 || ep > 1 {
        let on = spec.mtp == Some(true);
        if !on {
            note(
                Facet::SpecOff,
                if spec.mtp == Some(false) {
                    "MTP off: speculation.mtp=off"
                } else {
                    "MTP off: pp/ep topology (explicit --spec mtp arms it)"
                },
            );
        }
        return Ok(on);
    }
    let retained = hipfire_runtime::config::retained_redline_default(
        gpu_arch, "qwen4", path, pp, tp, ep, true,
    );
    let floor = hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR;
    let verdict = qwen4_native_mtp_for_floor(spec.mtp, retained, floor);
    if verdict == Ok(false) {
        note(
            Facet::SpecOff,
            format!(
                "MTP off: {}",
                qwen4_mtp_off_reason(spec.mtp, retained, floor)
            ),
        );
    }
    verdict
}

/// The reason [`qwen4_native_mtp_for_floor`] returned `Ok(false)`, for the
/// load summary. Only the label of the arm that already decided; the policy
/// itself stays in [`qwen4_native_mtp_for_floor`].
fn qwen4_mtp_off_reason(
    spec_mtp: Option<bool>,
    retained: bool,
    measured_ar_floor: bool,
) -> &'static str {
    if spec_mtp == Some(false) {
        "speculation.mtp=off"
    } else if retained {
        ".mq4r retained replay path"
    } else if !measured_ar_floor {
        "no measured AR floor (speculation.mtp=auto)"
    } else {
        "declined by policy"
    }
}

/// [`qwen4_native_mtp`]'s policy with the retained-Redline fact and the
/// producer's declared measured AR floor passed in, so the table is testable
/// without a GPU. `auto` attaches only when the producer declared the floor
/// ([`hipfire_config::mtp_mode_admits`]) and the load is not the retained
/// default; explicit `on` ignores the floor and refuses the retained default.
pub(crate) fn qwen4_native_mtp_for_floor(
    spec_mtp: Option<bool>,
    retained: bool,
    measured_ar_floor: bool,
) -> Result<bool, String> {
    use hipfire_config::MtpMode;
    match MtpMode::from_tri_state(spec_mtp) {
        MtpMode::Off => Ok(false),
        MtpMode::On if retained => Err(
            "qwen4: native MTP (speculation.mtp = on) cannot be used with the MQ4R pre-recorded \
             launch path (retained Redline) this model would run on. Set speculation.mtp = off, \
             or load the non-MQ4R HFQM build of this model"
                .into(),
        ),
        MtpMode::On => Ok(true),
        auto @ MtpMode::Auto => {
            Ok(!retained && hipfire_config::mtp_mode_admits(auto, measured_ar_floor))
        }
    }
}

/// Native MTP once the profile is known: the head exists only where the
/// profile declares one (`config.mtp`).  `auto` turns off for a head-free
/// profile; an explicit `speculation.mtp = "on"` (`--spec mtp`) is refused
/// rather than guessed, before any allocation.  A profile with the head keeps
/// the policy decision [`qwen4_native_mtp`] made.
pub(crate) fn qwen4_native_mtp_for_head(
    native_mtp: bool,
    spec_mtp: Option<bool>,
    has_mtp_head: bool,
    profile: &str,
) -> Result<bool, String> {
    match (has_mtp_head, spec_mtp) {
        (true, _) => Ok(native_mtp),
        (false, Some(true)) => Err(format!(
            "qwen4: native MTP (--spec mtp) was requested, but profile {profile} has no MTP head; \
             load a profile with an MTP head or leave speculation.mtp unset"
        )),
        (false, _) => Ok(false),
    }
}

/// Sidecar PLE rows (`memory.ple_format`), the hot-row store
/// (`memory.ple_hot_store_bytes`), the prewarm and the rank table are
/// Flash-Next-only: a developer profile reads its embedded rows directly.  An
/// explicit request for any of them on a developer profile is refused before
/// allocation; `auto` is skipped, with the returned note for the load log.
/// Flash-Next never reaches either arm.
pub(crate) fn qwen4_developer_ple_policy(
    flash_next: bool,
    id: &str,
    format: hipfire_config::memory::PleFormatRequest,
    hot_store: hipfire_config::memory::PleHotStoreBudget,
) -> Result<Option<String>, String> {
    use hipfire_config::memory::{PleFormatRequest, PleHotStoreBudget};
    if flash_next {
        return Ok(None);
    }
    if format != PleFormatRequest::Auto {
        return Err(format!(
            "qwen4: memory.ple_format = {format:?} requests PLE sidecar rows, which are Flash-Next-only; \
             developer profile {id} reads its embedded PLE rows (unset memory.ple_format)"
        ));
    }
    match hot_store {
        PleHotStoreBudget::Bytes(0) => Ok(None),
        PleHotStoreBudget::Bytes(bytes) => Err(format!(
            "qwen4: memory.ple_hot_store_bytes = {bytes} requests the PLE hot-row store, which is Flash-Next-only; \
             developer profile {id} reads its embedded PLE rows (set it to 0)"
        )),
        PleHotStoreBudget::Auto => Ok(Some(format!(
            "ple hot store and sidecar prewarm: off (developer profile {id} reads its embedded PLE rows)"
        ))),
    }
}

/// Whether an admitted Qwen4 native MTP head stays attached once
/// `host_mapped_experts` routed-expert tensors were placed in pinned host RAM
/// (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`). Device-resident loads (Strix Halo:
/// MTP text = AR on 8/8, 1.41x), exact gfx1201 (R9700, `auto` placement:
/// greedy MTP text = AR on 8/8 prompts, faster on every one) and exact gfx1100
/// (7900 XTX, trunk D2 CPU arm plus the MTP head D2 executor: +17.8% / +18.9% /
/// +19.8% over AR at 32K / 131K / 250K, target ids identical, per-request AR
/// floor still guards) keep the `auto` default on; other arches with host-mapped experts are unmeasured, so
/// `auto` keeps AR there and only an explicit `speculation.mtp = "on"`
/// (`--spec mtp`) attaches the head. `speculation.mtp = "off"` always keeps
/// AR (the admitted head is never attached then).
///
/// Thin twin over the shared [`hipfire_config::mtp_host_spill_policy`]: the
/// qwen4 admitted list is `&["gfx1100", "gfx1201"]` (the arch clause is data, not a
/// second function). Kept so the two policy call sites in `carriers.rs` keep
/// their typed-`SpecLoadCfg` spelling.
pub(crate) fn qwen4_mtp_with_host_mapped_experts(
    spec: SpecLoadCfg,
    gpu_arch: &str,
    host_mapped_experts: usize,
) -> bool {
    let mode = hipfire_config::MtpMode::from_tri_state(spec.mtp);
    hipfire_config::mtp_host_spill_policy(
        mode,
        host_mapped_experts,
        gpu_arch,
        &["gfx1100", "gfx1201"],
    )
    .attach
}

/// Fail-closed list for partial GPU offload, raised where the arch is dispatched.
///
/// An arch that never reaches the placement resolver must still be refused rather
/// than handed a resident load: "arch crates are typed declarations and the loader
/// owns placement" is only enforceable if an arch that cannot honour a placement
/// never quietly receives one. Returns `Err` naming the offending key and the
/// reason; `Ok(())` when no spill was requested or the source can honour it.
///
/// The trigger stays here — an *explicitly configured* spill (either budget key
/// off `Full`) is refused on an unsupported path (`spill_requested`, so the
/// MoE auto-fit default never trips this gate) — but every refusal dimension
/// is `hipfire_config::memory::spill_support`'s, mapped below to the existing
/// sentences. Wording is verbatim; only the derivation is shared.
pub(crate) fn partial_offload_refusal(
    arch_id: u32,
    source: &ModelSource,
    tp: usize,
    pp: usize,
    ep_experts: Option<usize>,
) -> Result<(), String> {
    use hipfire_config::memory::{spill_requested, spill_support, SpillSupport};
    if !spill_requested() {
        return Ok(());
    }
    let keys = "memory.gpu_layer_budget / memory.moe_expert_budget";
    // Facts the predicate owns: the loader's topology counts, the EP expert
    // count, and the model flags from the source — the tower tensor probe and
    // the parsed Qwen3.5 config (paged/REAP). Tower *tensor* presence decides
    // text-vs-VL, never config metadata (`config.is_vl_text` is the
    // nested-name text wrapper, which a plain text checkpoint carries too), so
    // ask the same probe the admission record uses.
    let ModelSource::Hfq(hfq) = source else {
        return match spill_support(arch_id, tp, pp, ep_experts, false, false, false) {
            SpillSupport::Supported => Ok(()),
            SpillSupport::Refused(refusal) => Err(spill_refusal_message(refusal, keys)),
        };
    };
    let is_vl = probe_vision(source, arch_id)?;
    let (paged_experts, reap_keep) = if matches!(arch_id, 5 | 6) {
        let config = hipfire_arch_qwen35::qwen35::config_from_hfq(hfq)
            .map_err(|e| format!("qwen35 config: {e}"))?;
        (config.paged_experts, config.reap_keep.is_some())
    } else if arch_id == 11 {
        // LFM2.5 parses its own REAP keep-map inside `Lfm2MoeConfig::from_hfq`
        // (`apply_reap_plan`). It has no paged-expert route.
        let config = hipfire_arch_lfm2moe::Lfm2MoeConfig::from_hfq(hfq)
            .map_err(|e| format!("lfm2moe config: {e}"))?;
        (false, config.reap_keep.is_some())
    } else {
        (false, false)
    };
    match spill_support(arch_id, tp, pp, ep_experts, paged_experts, reap_keep, is_vl) {
        SpillSupport::Supported => Ok(()),
        SpillSupport::Refused(refusal) => Err(spill_refusal_message(refusal, keys)),
    }
}

/// The loader gate's sentences for each `SpillRefusal` dimension — verbatim
/// from the hand-written checks this replaced (`{keys}` is the two budget
/// keys; non-arch arms interpolate their own count).
fn spill_refusal_message(refusal: hipfire_config::memory::SpillRefusal, keys: &str) -> String {
    use hipfire_config::memory::SpillRefusal::*;
    match refusal {
        // Only the Qwen3.5 (dense 5 / MoE 6) and LFM2.5 (11) storage seams
        // place a layer or a routed expert in host RAM. Qwen4 has its own
        // expert-residency route and the other arches have none, so either
        // knob is a request this loader cannot honour.
        Arch(arch_id) => format!(
            "load refused: partial GPU offload ({keys}) is implemented for Qwen3.5 dense and MoE \
             and LFM2.5 only; this source is arch id {arch_id}. Unset both keys for this model"
        ),
        Tp(tp) => format!(
            "load refused: partial GPU offload ({keys}) supports the single-GPU topology only; \
             tp={tp} loads every layer on every rank and never places one in host RAM. Unset both \
             keys for this model"
        ),
        Pp(pp) => format!(
            "load refused: partial GPU offload ({keys}) supports the single-GPU topology only; \
             pp={pp} would put the first pipeline stage's spilled weights in host RAM, which is \
             not supported yet. Unset both keys for this model"
        ),
        ExpertParallel(_) => format!(
            "load refused: partial GPU offload ({keys}) cannot be combined with expert parallelism \
             — an expert is pinned to one rank and its weights stay on that rank's device. Unset \
             both keys for this model"
        ),
        VisionTower => format!(
            "load refused: partial GPU offload ({keys}) cannot place the Qwen3.5-VL vision tower, \
             which is loaded by a separate path. Unset both keys for this model"
        ),
        PagedExperts => format!(
            "load refused: partial GPU offload ({keys}) cannot be combined with paged experts — \
             the weight pager manages those buffers itself and would conflict with the \
             placement. Unset both keys for this model"
        ),
        ReapKeepMap => format!(
            "load refused: partial GPU offload ({keys}) cannot be combined with a REAP keep-map — \
             a pruned-expert model has no host-placement path. Unset both keys for this model"
        ),
    }
}

/// The source-only portion of Qwen4 admission. The validated config and
/// inventory are reused by the executable Single carrier without reopening
/// or reclassifying the HFQM path.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4SourceAdmission {
    pub config: Qwen4Config,
    pub effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    pub modality: InputModality,
    pub native_mtp: bool,
    pub manifest_source_tensors: usize,
}

/// Refuse unsupported Qwen4 request dimensions before touching source records.
/// In particular, image/video requests cannot make the boundary inspect vision
/// payloads and then silently fall back to text.
pub fn qwen4_request_admission(
    effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    modality: InputModality,
    native_mtp: bool,
) -> Result<(), String> {
    let capabilities = if native_mtp {
        Qwen4Capabilities::text_mtp()
    } else {
        Qwen4Capabilities::text_ar()
    };
    if !capabilities.supports_modality(modality) {
        return Err(format!(
            "qwen4: {modality:?} input is not supported by this model route; Qwen4 loads accept \
             text only"
        ));
    }
    if native_mtp && !capabilities.native_mtp {
        return Err(
            "qwen4: native MTP is not supported by this model file; set speculation.mtp = off"
                .into(),
        );
    }
    if !effective_mesh.is_qwen4_admitted() {
        return Err(format!(
            "qwen4: this parallelism combination is not supported (topology.pp={}, tp={}, ep={}). \
             Use pp=1 and tp=1 with topology.ep from 1 to 8, or tp=1 and ep=1 with topology.pp \
             from 2 to 8",
            effective_mesh.pp, effective_mesh.tp, effective_mesh.ep
        ));
    }
    Ok(())
}

/// Exact architectures Qwen4 expert parallelism is admitted on: the EP gather
/// and the grouped MoE route are certified on gfx1100 (7900 XTX) and gfx1201
/// (R9700). Every rank must be the SAME one of them (homogeneous).
pub const QWEN4_EP_ARCHS: [&str; 2] = ["gfx1100", "gfx1201"];

/// The refusal for a safetensors directory handed to a Flash-Next (qwen4) load.
pub(crate) const QWEN4_SAFETENSORS_REFUSAL: &str = "qwen4: safetensors is a conversion input, \
     not an executable HFQ artifact; quantize it first: hipfire quantize <dir>";

/// Under the developer-only logical emulation (`HIPFIRE_EMULATE_GPUS`) the
/// ranks also run aliased on gfx1151 (Strix Halo has one GPU): that proves
/// byte parity and lifecycle on one device, never a physical EP admission.
const QWEN4_EP_EMULATED_ARCHS: [&str; 1] = ["gfx1151"];

fn qwen4_ep_arch_admitted(arch: &str, emulated: bool) -> bool {
    QWEN4_EP_ARCHS.contains(&arch) || (emulated && QWEN4_EP_EMULATED_ARCHS.contains(&arch))
}

/// Qwen4 expert parallelism needs `ep` ranks of one exact, homogeneous
/// gfx1100 or gfx1201 arch and one visible device per rank (or, under the
/// developer-only logical emulation `HIPFIRE_EMULATE_GPUS`, at least one
/// physical device that the ranks alias onto).
/// Pre-allocation check from what admission already knows: the daemon's own
/// device architecture and the visible device count (`None` skips the count,
/// as in [`visible_device_count`]). `ep <= 1` never refuses. Per-device
/// architectures and peer access are checked with real devices by
/// [`qwen4_ep_preflight`] at load.
pub(crate) fn qwen4_ep_geometry_refusal(
    ep: usize,
    pp: usize,
    tp: usize,
    gpu_arch: &str,
    n_devices: Option<usize>,
    emulated: bool,
) -> Option<String> {
    if ep <= 1 {
        return None;
    }
    if pp != 1 || tp != 1 {
        return Some(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) cannot combine with pipeline/tensor parallelism (pp={pp}, tp={tp})"
        ));
    }
    if !qwen4_ep_arch_admitted(gpu_arch, emulated) {
        return Some(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) requires gfx1100 or gfx1201 on every rank \
             (gfx1151 only under HIPFIRE_EMULATE_GPUS); device 0 is {gpu_arch}"
        ));
    }
    let needed = if emulated { 1 } else { ep };
    match n_devices {
        Some(n) if n < needed => Some(if emulated {
            format!(
                "qwen4: logical expert parallelism emulation (topology.ep = {ep}) needs at least 1 visible device, {n} visible"
            )
        } else {
            format!(
                "qwen4: expert parallelism (topology.ep = {ep}) needs {ep} visible devices, {n} visible \
                 (hardware.devices / HIPFIRE_DEVICES)"
            )
        }),
        _ => None,
    }
}

/// Rank -> physical HIP device table for `ep` ranks over `physical_count`
/// visible devices. Physical mode: rank `r` runs on device `r` and `ep`
/// devices must be visible. Emulated (`HIPFIRE_EMULATE_GPUS`, developer-only):
/// rank `r` runs on physical device `r % physical_count` (the
/// `multi_gpu` alias rule) and one physical device is enough.
pub fn qwen4_ep_rank_devices(
    ep: usize,
    physical_count: usize,
    emulated: bool,
) -> Result<Vec<i32>, String> {
    let ids: Vec<i32> = (0..ep as i32).collect();
    if emulated {
        if physical_count == 0 {
            return Err(format!(
                "qwen4: logical expert parallelism emulation (topology.ep = {ep}) needs at least 1 visible device, 0 visible"
            ));
        }
        Ok(hipfire_runtime::multi_gpu::alias_ids(
            &ids,
            physical_count as i32,
        ))
    } else if physical_count < ep {
        Err(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) needs {ep} visible devices, {physical_count} visible"
        ))
    } else {
        Ok(ids)
    }
}

/// Pure core of [`qwen4_ep_preflight`]: `devices[r]` is rank `r`'s physical
/// device (see [`qwen4_ep_rank_devices`]), `archs[d]` is physical device `d`'s
/// architecture; `peer(a, b)` is whether device `a` can access device `b`.
/// Every rank's device must be the same exact gfx1100 or gfx1201 arch (or,
/// with `emulated`, gfx1151) as rank 0's, and every directed pair of DISTINCT
/// physical devices must be accessible (ranks aliased onto one device need no
/// peer access).
pub fn qwen4_ep_device_refusal(
    ep: usize,
    devices: &[i32],
    archs: &[String],
    emulated: bool,
    mut peer: impl FnMut(usize, usize) -> Result<bool, String>,
) -> Result<(), String> {
    if devices.len() < ep {
        return Err(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) needs {ep} rank devices, {} mapped",
            devices.len()
        ));
    }
    let mut arch_of = Vec::with_capacity(ep);
    for &device in devices.iter().take(ep) {
        match usize::try_from(device).ok().and_then(|d| archs.get(d)) {
            Some(arch) => arch_of.push(arch),
            None => {
                return Err(format!(
                    "qwen4: expert parallelism (topology.ep = {ep}) needs {ep} visible devices, {} visible",
                    archs.len()
                ));
            }
        }
    }
    let first = arch_of[0];
    for (rank, arch) in arch_of.iter().enumerate() {
        if !qwen4_ep_arch_admitted(arch, emulated) {
            return Err(format!(
                "qwen4: expert parallelism (topology.ep = {ep}) requires gfx1100 or gfx1201 on every rank \
                 (gfx1151 only under HIPFIRE_EMULATE_GPUS); rank {rank} is {arch}"
            ));
        }
        if *arch != first {
            return Err(format!(
                "qwen4: expert parallelism (topology.ep = {ep}) requires the same architecture on every rank; rank 0 is {first}, rank {rank} is {arch}"
            ));
        }
    }
    let mut probed: Vec<(i32, i32)> = Vec::new();
    for from in 0..ep {
        for to in 0..ep {
            let (a, b) = (devices[from], devices[to]);
            if a == b || probed.contains(&(a, b)) {
                continue;
            }
            probed.push((a, b));
            if !peer(a as usize, b as usize)? {
                return Err(format!(
                    "qwen4: expert parallelism (topology.ep = {ep}) needs peer access between every pair of ranks; \
                     device {a} cannot access device {b}"
                ));
            }
        }
    }
    Ok(())
}

/// Load-time EP device check with real devices: the rank devices
/// ([`qwen4_ep_rank_devices`]: devices `0..ep` of this process's HIP view, in
/// the `hardware.devices` order, or the `r % device_count` aliases under
/// `HIPFIRE_EMULATE_GPUS`) must be visible, share one exact gfx1100 or gfx1201
/// arch, and every directed pair of distinct devices must report
/// `hipDeviceCanAccessPeer`. Returns the rank → HIP device id table.
pub fn qwen4_ep_preflight(hip: &hip_bridge::HipRuntime, ep: usize) -> Result<Vec<i32>, String> {
    let count = hip
        .device_count()
        .map_err(|error| format!("qwen4: EP device count: {error}"))?
        .max(0) as usize;
    let emulated = hipfire_runtime::config::get().emulate_gpus.is_some();
    let devices = qwen4_ep_rank_devices(ep, count, emulated)?;
    let archs = (0..count.min(ep))
        .map(|device| {
            hip.get_arch(device as i32)
                .map(|arch| arch.split(':').next().unwrap_or("").to_string())
                .map_err(|error| format!("qwen4: EP device {device} arch: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    qwen4_ep_device_refusal(ep, &devices, &archs, emulated, |from, to| {
        hip.can_access_peer(from as i32, to as i32)
            .map_err(|error| format!("qwen4: EP peer query {from}->{to}: {error}"))
    })?;
    Ok(devices)
}

/// The expert-parallel degree a loaded model of `arch_id` runs with: `1`
/// everywhere except Qwen4 with `topology.ep` > 1 (and no `HIPFIRE_QWEN4_EP=0`
/// kill switch). An invalid configured value was refused at admission, so it
/// reads as `1` here.
pub fn effective_ep(arch_id: u32) -> usize {
    if arch_id == QWEN4_ARCH_ID {
        hipfire_config::topology::qwen4_ep().unwrap_or(1)
    } else {
        1
    }
}

/// Device memory a Qwen4 load keeps free beyond the growth it can name
/// (context pages, gather workspace, native MTP request buffers): the first
/// long request's other lazy allocations (measured ~1.9 GiB on Strix Halo
/// with 8192-row chunks), kernel-cache warmup and VMM page rounding.
pub const QWEN4_MEMORY_HEADROOM_BYTES: u64 = 2 << 30;

/// Smallest context an automatic Qwen4 load accepts.
const QWEN4_MIN_AUTO_CONTEXT: usize = 4096;

/// What a Qwen4 load admits of the device memory left free after its own
/// allocations ([`qwen4_memory_admission`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4MemoryAdmission {
    /// Free device memory once the load's own allocations are made.
    pub free: u64,
    /// Target and MTP-head QSA context arenas still unmapped through
    /// `max_seq` (VMM storage maps them as forwards reach them), including
    /// the opt-in F16 arena of the QSA layers.
    pub context_growth: u64,
    /// Gathered QSA prefill workspace still uncommitted through `max_seq`.
    pub gather_growth: u64,
    /// Buffers the first native MTP request allocates and keeps.
    pub mtp_request: u64,
    /// The PLE page cache budget the PLE prewarm / in-load fill was asked
    /// to keep resident at serve time (0 when no PLE rows are cached).
    pub ple_budget: u64,
    /// The PLE page cache budget admitted: `ple_budget` shrunk to what the
    /// free memory leaves after the context, the gather workspace, the MTP
    /// request buffers and the headroom. On unified memory the page cache
    /// counts as free memory but is reclaimable, so it yields to the
    /// context; this is 0 when only the context itself fits.
    pub resident_ple: u64,
    pub headroom: u64,
    /// Largest context whose growth fits with no PLE page cache resident
    /// (at most the assembled capacity).
    pub card_cap: usize,
    /// The admitted context.
    pub max_seq: usize,
}

impl Qwen4MemoryAdmission {
    /// Everything a request through `max_seq` still commits after load.
    pub fn growth(&self) -> u64 {
        self.context_growth
            .saturating_add(self.gather_growth)
            .saturating_add(self.mtp_request)
    }

    /// What must be free after load: [`Self::growth`] plus the admitted
    /// resident PLE and the headroom.
    pub fn required(&self) -> u64 {
        self.growth()
            .saturating_add(self.resident_ple)
            .saturating_add(self.headroom)
    }

    /// Whether the context took part of the requested PLE page cache budget
    /// (`resident_ple` < `ple_budget`).
    pub fn ple_shrunk(&self) -> bool {
        self.resident_ple < self.ple_budget
    }
}

/// Admit what a Qwen4 load commits after load against the `free` device
/// memory its own allocations left, keeping `headroom` free: VMM context
/// pages and the gathered workspace through a context of `t` tokens
/// (`growth(t)` = `(context, gather)`, monotone, assembled for `capacity`
/// tokens) and the native MTP request buffers. `card_cap` is the largest
/// context that fits with no PLE page cache resident. The `resident_ple`
/// bytes of PLE page cache the prewarm / in-load fill ask to keep resident
/// (unified memory counts them as free) are reclaimable, so the context
/// takes precedence: the admitted PLE budget is shrunk to what the free
/// memory leaves after the context. An automatic load (`requested` = `None`)
/// admits `min(capacity, card_cap)`; an explicit `max_seq` must fit whole.
/// The session cache is not charged: its snapshots yield to context growth.
pub fn qwen4_memory_admission(
    capacity: usize,
    requested: Option<usize>,
    free: u64,
    mtp_request: u64,
    resident_ple: u64,
    headroom: u64,
    growth: impl Fn(usize) -> (u64, u64),
) -> Result<Qwen4MemoryAdmission, String> {
    let fixed = mtp_request.saturating_add(headroom);
    let fits = |tokens: usize| {
        let (context, gather) = growth(tokens);
        context.saturating_add(gather).saturating_add(fixed) <= free
    };
    let card_cap = if !fits(0) {
        0
    } else {
        // Largest fitting context in [0, capacity].
        let (mut lo, mut hi) = (0usize, capacity);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    };
    let max_seq = requested.unwrap_or(capacity.min(card_cap));
    let (context_growth, gather_growth) = growth(max_seq);
    let left_for_ple = free
        .saturating_sub(context_growth)
        .saturating_sub(gather_growth)
        .saturating_sub(fixed);
    let admission = Qwen4MemoryAdmission {
        free,
        context_growth,
        gather_growth,
        mtp_request,
        ple_budget: resident_ple,
        resident_ple: resident_ple.min(left_for_ple),
        headroom,
        card_cap,
        max_seq,
    };
    let detail = || {
        format!(
            "a {max_seq}-token context needs {} bytes free after load even with no resident PLE page cache ({context_growth} bytes QSA context incl. any F16 arena + {gather_growth} bytes QSA gather workspace + {mtp_request} bytes native MTP request buffers = {} bytes growth, + {headroom} bytes headroom), and {free} bytes is free after load",
            admission.growth().saturating_add(headroom),
            admission.growth(),
        )
    };
    match requested {
        None if max_seq < QWEN4_MIN_AUTO_CONTEXT.min(capacity) => Err(format!(
            "qwen4: device memory cannot hold an automatic context: {}; with the current free host memory the context fits up to {card_cap} tokens (this limit moves with free host memory); free host memory",
            detail()
        )),
        Some(requested) if requested > card_cap => Err(format!(
            "qwen4: max_seq {requested} does not fit device memory: {}; with the current free host memory the context fits up to {card_cap} tokens (this limit moves with free host memory); lower memory.max_seq to at most {card_cap} or omit it",
            detail()
        )),
        _ => Ok(admission),
    }
}

/// Whether `hfq` carries the Qwen4 PLE n-gram table in either form: the 128
/// shard tensors (monolithic) or the split-base marker.  Anything else is no
/// PLE-bearing base at all and keeps failing in the artifact admission with
/// its own inventory/config diagnostic instead of a PLE-rows one.
fn qwen4_ple_rows_layout_present(hfq: &hipfire_runtime::hfq::HfqFile) -> bool {
    hfq.ple_rows_marker().is_some()
        || hfq
            .tensors()
            .iter()
            .any(|tensor| hipfire_arch_qwen4::weights::is_ple_shard_name(&tensor.name))
}

/// Resolve and attach the Qwen4 PLE rows source (`memory.ple_format`) on an
/// opened HFQ source.  Sidecars are looked up beside `path` exactly as given,
/// then beside its canonical target.  A missing sidecar is an error: the load
/// path never derives (`hipfire ple derive` is the explicit step).  Idempotent;
/// a no-op for non-HFQ sources and for a source that is not a PLE-bearing
/// Qwen4 base.  Returns the choice for the load log (`None` when nothing was
/// resolved).
pub fn resolve_qwen4_ple_rows(
    source: &mut ModelSource,
    path: &str,
    gpu_arch: &str,
) -> Result<Option<hipfire_runtime::ple_sidecar::PleRowsChoice>, String> {
    let ModelSource::Hfq(hfq) = source else {
        return Ok(None);
    };
    resolve_qwen4_ple_rows_hfq(hfq, path, gpu_arch)
}

/// [`resolve_qwen4_ple_rows`] on an opened HFQ file.  A base without the
/// Flash-Next PLE layout (a developer profile's embedded rows, or a profile
/// without PLE) has no sidecar to resolve and yields `None`.
pub fn resolve_qwen4_ple_rows_hfq(
    hfq: &mut hipfire_runtime::hfq::HfqFile,
    path: &str,
    gpu_arch: &str,
) -> Result<Option<hipfire_runtime::ple_sidecar::PleRowsChoice>, String> {
    resolve_qwen4_ple_rows_requested(hfq, path, hipfire_config::memory::ple_format(), gpu_arch)
}

/// [`resolve_qwen4_ple_rows_hfq`] for an explicit `memory.ple_format`
/// request.  The load path's resolve policy lives here: it never derives a
/// missing q4 sidecar, whatever the request (a q4 request alone is not
/// consent to write a ~29 GB file; `hipfire ple derive` is the explicit step).
fn resolve_qwen4_ple_rows_requested(
    hfq: &mut hipfire_runtime::hfq::HfqFile,
    path: &str,
    request: hipfire_config::memory::PleFormatRequest,
    gpu_arch: &str,
) -> Result<Option<hipfire_runtime::ple_sidecar::PleRowsChoice>, String> {
    if !qwen4_ple_rows_layout_present(hfq) {
        return Ok(None);
    }
    if qwen4_ple_rows_fp8(hfq) {
        // Stock-FP8 PLE rows are the artifact's own qt=56 shards (their scale
        // is per shard): there is no q4/q8 sidecar to resolve or derive.
        return match request {
            hipfire_config::memory::PleFormatRequest::Auto => Ok(None),
            format => Err(format!(
                "qwen4: memory.ple_format = {format:?} requests PLE sidecar rows, but this \
                 stock-FP8 artifact reads its embedded FP8E4M3PT PLE rows; unset memory.ple_format"
            )),
        };
    }
    hipfire_runtime::ple_sidecar::resolve_and_attach_ple_rows(
        hfq,
        std::path::Path::new(path),
        request,
        gpu_arch,
        hipfire_runtime::ple_sidecar::ResolvePolicy {
            derive_missing_q4: false,
            progress: None,
        },
    )
    .map(Some)
    .map_err(|error| format!("qwen4: PLE rows: {error}"))
}

/// Whether the base's embedded PLE shards are qt=56 FP8E4M3PT rows (the
/// stock-FP8 recipe).
fn qwen4_ple_rows_fp8(hfq: &hipfire_runtime::hfq::HfqFile) -> bool {
    hfq.tensors().iter().any(|tensor| {
        tensor.quant_type == 56 && hipfire_arch_qwen4::weights::is_ple_shard_name(&tensor.name)
    })
}

/// Validate a reserved arch-16 HFQM source without allocating weights.
///
/// The architecture crate owns the complete source/index contract.  Loader
/// admission only adds request/topology policy and retains the receipt's
/// value-only config/count for the carrier handoff.
pub fn admit_qwen4_source(
    source: &ModelSource,
    effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    modality: InputModality,
    native_mtp: bool,
    spec_mtp: Option<bool>,
    gpu_arch: &str,
) -> Result<Qwen4SourceAdmission, String> {
    qwen4_request_admission(effective_mesh, modality, native_mtp)?;
    let ModelSource::Hfq(hfq) = source else {
        return Err(QWEN4_SAFETENSORS_REFUSAL.into());
    };
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(gpu_arch))
        .map_err(|error| error.to_string())?;
    // An inventory-valid developer profile the executor cannot run (a
    // synthetic MoE, a RoPE base the QSA kernels lack, a build without
    // `qwen4-dev-profiles`) is refused here, not after the weights upload.
    hipfire_arch_qwen4::bundle::admit_executable_profile(&receipt.config)
        .map_err(|error| format!("qwen4: {error}"))?;
    // The profile is known now: the head-free and developer-PLE refusals
    // happen here, before any weight is allocated or any model torn down.
    let native_mtp = qwen4_native_mtp_for_head(
        native_mtp,
        spec_mtp,
        receipt.config.mtp.is_some(),
        receipt.config.profile.id(),
    )?;
    // The recipe's device admission: a stock-FP8 artifact needs FP8 expert
    // kernels (gfx942 only), one device and every routed expert resident.
    let host_mapped_experts = hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()
        .map_err(|error| format!("qwen4: {error}"))?
        .is_some();
    // A pipeline band keeps every routed expert resident (packed aggregates
    // or, for an expert-slab artifact, its own layers' slab arenas); no band
    // layer is host-mapped.
    if effective_mesh.pp > 1 && host_mapped_experts {
        return Err(qwen4_pp_unsupported(
            effective_mesh.pp,
            "host-mapped routed experts (HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS)",
        ));
    }
    hipfire_arch_qwen4::admission::admit_recipe_on_device(
        receipt.recipe(),
        gpu_arch,
        effective_mesh,
        native_mtp,
        host_mapped_experts && receipt.recipe().fp8_experts(),
    )
    .map_err(|error| error.to_string())?;
    qwen4_developer_ple_policy(
        receipt.config.profile.is_flash_next(),
        receipt.config.profile.id(),
        hipfire_config::memory::ple_format(),
        hipfire_config::memory::ple_hot_store_bytes(),
    )?;
    Ok(Qwen4SourceAdmission {
        config: receipt.config,
        effective_mesh,
        modality,
        native_mtp,
        manifest_source_tensors: receipt.source_tensor_count,
    })
}

/// A source admitted before any destructive side effect.
pub struct SourceAdmission {
    /// The already-open source. The single/pp route consumes it (no second
    /// open); the EP route re-opens `path` per rank and drops this handle.
    pub source: ModelSource,
    pub arch_id: u32,
    pub is_dir: bool,
    /// Tower-tensor presence decides text-vs-VL; config metadata alone never
    /// does (remediation contract `179a20d7f`).
    pub has_vision: bool,
    pub topology: EffectiveTopology,
    pub kv_backend: KvBackend,
    /// Single admitted sequence limit consumed by both daemon and loader.
    pub max_seq: usize,
    /// Reason for automatic legacy fallback; absent for explicit selection.
    pub kv_backend_reason: Option<String>,
    /// Effective context and the two default limits (when Qwen owns growing VMM KV).
    pub sequence: Option<SequenceResolution>,
    /// Why an automatic sequence bound retained legacy behavior.
    pub sequence_reason: Option<&'static str>,
    /// True when the automatic legacy fallback measured from free VRAM
    /// ([`qwen35_legacy_fit`]) instead of taking the fixed 4096. Backend-gated,
    /// not OS-gated; the daemon reports bound `legacy-fit` for these loads.
    pub legacy_measured: bool,
    pub kv_backend_request: KvBackendRequest,
    pub qwen_default_q8: bool,
    /// Effective V axis ([`resolve_kv_v`]); the loader hands exactly this to
    /// the carrier.
    pub kv_v: Option<String>,
    /// The resolved carrier (single/pp path). `None` for expert-parallel, which
    /// dispatches on `arch_id` directly rather than through the registry.
    pub carrier: Option<&'static dyn Carrier>,
    /// Validated vision-tower sidecar (`params.vision` / `HIPFIRE_VISION_SIDECAR`).
    /// `Some` only when the sidecar opened, carries the trunk's VL family arch
    /// id (Qwen3.5-VL 5|6, EmbeddingGemma-2 18), and holds that family's tower
    /// probe tensor; the single/pp route threads it into `LoadCtx`, which the
    /// arch crate opens at load. `None` = trunk-only (or explicit opt-out via
    /// empty string).
    pub vision_path: Option<std::path::PathBuf>,
    /// The resolved `vision.mode` ladder value (`auto`/`resident`/`on`/`off`). The
    /// carrier needs it to gate `<stem>.vl` sibling discovery: `vision_path`
    /// alone cannot distinguish "off" (never probe a sibling) from "auto with
    /// no explicit sidecar" (probe allowed). `off` suppresses discovery.
    pub vision_mode: String,
    /// The one admitted mesh: topology, physical identity of every logical
    /// rank (HIP ordinal + PCI id + arch), per-edge transports and
    /// numerical-policy identity. Normalized here exactly once; no later
    /// stage rereads the raw pp/tp/ep request.
    pub mesh: AdmittedMesh,
    /// Pipeline band placement (`Some` only for the pp>1 Qwen3.5 load). Built
    /// from the single `BandPlacement` authority at admission.
    pub pp_bands: Option<BandPlacement>,
    /// Explicit per-stage layer counts, read from `HIPFIRE_PP_LAYERS` once at
    /// admission and normalized through `BandPlacement::from_counts`. `None`
    /// = the historical uniform split (`Gpus::init_uniform`, which keeps its
    /// free-VRAM delta gate); `Some` = `Gpus::init_layers`. Carriers consume
    /// this value and never reread the environment.
    pub pp_layer_counts: Option<Vec<usize>>,
    /// The continuous-batch request of this load (default off). Admission
    /// never sets it; the caller that will stage lanes after the load sets it
    /// on the admitted handle before the load, and the carrier forwards it
    /// through `LoadCtx::batch_request`.
    pub batch_request: BatchLoadRequest,
}
/// The physical VMM reservation equals `max_seq` without eviction; with
/// CASK eviction it is `min(max_seq, eviction_window)`. Only the former must
/// grow to the full card-admitted bound. Explicit overrides may exceed either
/// limit by operator request.
#[derive(Debug, Clone)]
pub struct SequenceResolution {
    pub max_seq: usize,
    pub bound: &'static str,
    pub model_ctx: usize,
    pub card_cap: usize,
    pub kv_mode: &'static str,
    /// The pre-teardown capacity projection this resolution was admitted
    /// against ([`qwen35_card_projection`]); `None` where no projection runs
    /// (Qwen4 on unified memory measures after load).
    pub projection: Option<Qwen35CardProjection>,
}

/// The pre-teardown context-capacity projection of a single-card Qwen3.5
/// VMM load ([`qwen35_card_projection`]), itemised so `hipfire fit` reports
/// the same numbers admission refuses on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35CardProjection {
    /// Free device bytes before the load (`KvBackendHints::free_vram_bytes`).
    pub free: usize,
    /// Projected resident weights: the HFQ file plus 12.5% and 512 MiB.
    pub weights: usize,
    /// The minimum viable prefill scratch reservation.
    pub scratch: usize,
    /// VMM page rounding across the K/V owners.
    pub page_rounding: usize,
    /// Selected K+V bytes per token over the full-attention layers.
    pub bytes_per_token: usize,
    /// Largest context the remaining free memory holds.
    pub card_cap: usize,
    /// The checkpoint's trained window (`max_position_embeddings`).
    pub model_ctx: usize,
    pub kv_mode: &'static str,
}

/// The context a Qwen4 admission works from: an omitted `max_seq` on a
/// discrete GPU is the default context (it keeps no automatic sequence);
/// everything else, including an explicit value on any GPU, is unchanged.
fn qwen4_requested_context(unified_memory: bool, max_seq: usize) -> usize {
    if max_seq == 0 && !unified_memory {
        hipfire_arch_qwen4::QWEN4_DEFAULT_CONTEXT
    } else {
        max_seq
    }
}

/// The Qwen4 sequence on a unified-memory APU: an omitted `max_seq` is
/// automatic (the model's context, capped by the measured card capacity at
/// load), an explicit one is honored and re-checked against the free memory
/// after load ([`qwen4_memory_admission`]). A discrete GPU resolves no
/// sequence, so its requested context is never memory-admitted here.
fn qwen4_context_sequence(
    unified_memory: bool,
    max_seq: usize,
    model_ctx: usize,
    kv_mode: &'static str,
) -> Option<SequenceResolution> {
    unified_memory.then(|| SequenceResolution {
        max_seq: if max_seq == 0 { model_ctx } else { max_seq },
        bound: if max_seq == 0 { "pending" } else { "user" },
        model_ctx,
        card_cap: 0,
        kv_mode,
        projection: None,
    })
}

/// Pure text-vs-VL decision. The vision tower tensor decides; configuration
/// metadata alone never does. Every Qwen3.5-family HF config embeds
/// `vision_config` even for text-only quantized artifacts, so a config marker
/// without the tower is the text backbone, not a refusal.
///
/// LFM2 is the one exception that *refuses*: a tower tensor with no parseable
/// `vision_config` metadata is malformed (carriers.rs:1553-1558) and fails
/// closed rather than silently loading as text.
///
/// Contract provenance: remediation commit `179a20d7f` ("classify Qwen3.5/LFM2
/// sources by vision tower tensor, not config markers").
pub fn classify_vision(
    arch_id: u32,
    has_vision_tensor: bool,
    has_vision_config: bool,
) -> Result<bool, String> {
    match arch_id {
        // Qwen3.5 dense (5) / MoE (6): the tower tensor alone decides.
        5 | 6 => Ok(has_vision_tensor),
        // LFM2 (11): tower + config both required; tower-without-config refuses.
        11 => {
            if has_vision_tensor && !has_vision_config {
                return Err(
                    "lfm2moe: artifact carries vision tensors but no vision_config \
                     metadata — requantize with --include-vision"
                        .into(),
                );
            }
            Ok(has_vision_tensor)
        }
        // Qwen4's conditional-generation name does not imply a vision route.
        // A text artifact is capable when no tower records are present; a
        // tower-bearing artifact is refused rather than probed as an image
        // request or silently downgraded to text.
        QWEN4_ARCH_ID => {
            if has_vision_tensor {
                Err(
                    "qwen4: this model file contains vision tensors, but Qwen4 loads text only; \
                     use the text-only build of the model"
                        .into(),
                )
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

/// Read the vision-tower probes out of an already-open source and fold them
/// through [`classify_vision`]. Read-only: probes the HFQ tensor index and (for
/// LFM2) parses `vision_config` metadata; touches no GPU state.
fn probe_vision(src: &ModelSource, arch_id: u32) -> Result<bool, String> {
    let ModelSource::Hfq(hfq) = src else {
        return classify_vision(arch_id, false, false);
    };
    let (has_tensor, has_config) = match arch_id {
        5 | 6 => (
            hfq.tensor_data("model.visual.patch_embed.proj.weight")
                .is_some(),
            // Qwen3.5 does not use the config in the decision; config parse is
            // soft (carriers.rs:527-544). A dummy `false` is never read.
            false,
        ),
        11 => (
            hfq.tensor_data("model.vision_tower.vision_model.embeddings.patch_embedding.weight")
                .is_some(),
            hipfire_arch_lfm2_vl::vision_config_from_hfq(hfq).is_some(),
        ),
        QWEN4_ARCH_ID => (
            hfq.tensors()
                .iter()
                .any(|tensor| qwen4_vision_tensor_name(&tensor.name)),
            false,
        ),
        _ => (false, false),
    };
    classify_vision(arch_id, has_tensor, has_config)
}

/// Tower probe tensor shared by the Qwen3.5-VL trunk and its `.vl` sidecar.
const VISION_PROBE_TENSOR: &str = "model.visual.patch_embed.proj.weight";

/// EmbeddingGemma-2 (arch 18) tower probe: the vision patch embedder. The
/// checkpoint carries the tower under flat `vision_tower.*` names — there is
/// no `model.visual.` alias — so the Qwen probe never resolves there, and the
/// two families must not share one probe.
const EMBED_GEMMA_VISION_PROBE_TENSOR: &str = "vision_tower.patch_embedder.input_proj.weight";

/// The `.vl` sidecar contract of one VL family:
/// `(sidecar arch ids the quantizer stamps on the pack, their spelling in
/// refusals, the probe tensor proving the tower shipped, the pack remedy)`.
/// The trunk-side gate and the sidecar-side gate share it, so a pack stamped
/// by another family — whose tower names never resolve against this trunk —
/// is refused instead of being opened as the wrong tower.
fn vision_sidecar_family(
    arch_id: u32,
) -> Option<(&'static [u32], &'static str, &'static str, &'static str)> {
    match arch_id {
        // Qwen3.5-VL dense (5) / MoE (6): the `--vision-only` pack of
        // `model.visual.*`; either trunk accepts either family stamp.
        5 | 6 => Some((
            &[5, 6],
            "5|6",
            VISION_PROBE_TENSOR,
            "pack the tower with `hipfire-quantize <hf-dir> --include-vision \
             --include-prefix model.visual.`",
        )),
        // EmbeddingGemma-2 (18): the trunk is tower-less by construction and
        // ships its own arch-18 `.vl` pack of the `vision_tower.*` group.
        18 => Some((
            &[18],
            "18",
            EMBED_GEMMA_VISION_PROBE_TENSOR,
            "requantize with `hipfire-quantize <hf-dir>` (the EmbeddingGemma-2 \
             pack writes the `.vl` tower sidecar beside the trunk)",
        )),
        _ => None,
    }
}

/// Validate the optional vision-tower sidecar and return its resolved path.
/// Fail-closed: an unopenable, wrong-family, or tower-less sidecar refuses
/// with a message naming the remedy. The sidecar opens read-only as a
/// SEPARATE `HfqFile` (never `attach_overlay` — REAP rejects additive tensor
/// names, hfq.rs:391-427). Empty string counts as unset (explicit opt-out,
/// same semantics as `HIPFIRE_DFLASH_DRAFT`).
fn resolve_vision_sidecar(
    vision: Option<&str>,
    arch_id: u32,
    is_dir: bool,
) -> Result<Option<std::path::PathBuf>, String> {
    let path = match vision.filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => return Ok(None),
    };
    if is_dir {
        return Err(format!(
            "vision sidecar '{path}' requires an HFQ trunk: safetensors directory \
             sources cannot carry a sidecar tower"
        ));
    }
    let Some((sidecar_archs, expected, probe, remedy)) = vision_sidecar_family(arch_id) else {
        return Err(format!(
            "vision sidecar '{path}' requested for arch_id={arch_id}: vision sidecars \
             only serve Qwen3.5-VL (arch_id 5|6) and EmbeddingGemma-2 (arch_id 18) trunks"
        ));
    };
    let sidecar = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("vision sidecar '{path}': open failed: {e}"))?;
    if !sidecar_archs.contains(&sidecar.arch_id) {
        return Err(format!(
            "vision sidecar '{path}' has arch_id={} (expected {expected}): {remedy}",
            sidecar.arch_id
        ));
    }
    if sidecar.tensor_data(probe).is_none() {
        return Err(format!(
            "vision sidecar '{path}' carries no vision tower tensor '{probe}': {remedy}"
        ));
    }
    Ok(Some(std::path::PathBuf::from(path)))
}

/// Maple head-overlay arch id (`hipfire-quantize --head-only` carriers).
const MAPLE_ARCH_ID: u32 = 15;

/// Validate a `--head` overlay against the already-open base and attach it so
/// the retained source IS the effective base+head: loading consumes it with
/// no second open. Every refusal fires here, before prior-model teardown.
/// Empty string counts as unset (explicit opt-out, same as vision/draft).
/// Refusals, never silent fallbacks: serving the base head when an overlay
/// was requested hands back a model the operator did not ask for.
fn admit_head_overlay(
    head: Option<&str>,
    base: &mut ModelSource,
    arch_id: u32,
    topology: EffectiveTopology,
) -> Result<(), String> {
    let path = match head.filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => return Ok(()),
    };
    if base.is_dir() {
        return Err(format!(
            "--head '{path}' requires an HFQ trunk: safetensors directory \
             sources cannot carry a head overlay"
        ));
    }
    if topology != EffectiveTopology::Single {
        return Err(format!(
            "--head '{path}' requires a single-device load (tp=1, pp=1): \
             expert/pipeline-parallel loads use the head baked into the model file"
        ));
    }
    if arch_id != MAPLE_ARCH_ID {
        return Err(format!(
            "--head '{path}' requested for arch_id={arch_id}: head overlays only \
             serve Maple (arch_id 15) — refusing rather than silently serving \
             the base head"
        ));
    }
    let ModelSource::Hfq(hfq) = &mut *base else {
        return Err(format!("--head '{path}' requires an HFQ trunk"));
    };
    // The overlay slot holds at most one file: a REAP splice already
    // installed there would be silently discarded by the head attach.
    // Conservatively refuse the combination instead of stacking overlays.
    if hfq.has_overlay() {
        return Err(format!(
            "--head '{path}' cannot combine with an active REAP overlay \
             (single overlay slot): disable one of them"
        ));
    }
    let ov = hipfire_runtime::hfq::HfqFile::open_at_offset(std::path::Path::new(path), 0)
        .map_err(|e| format!("head overlay '{path}': open failed: {e}"))?;
    hfq.attach_opened_head(ov, std::path::Path::new(path))?;
    Ok(())
}

/// Resolve the single carrier that claims a source, refusing no-carrier and
/// ambiguous-carrier sources exactly as the load entries do.
fn resolve_carrier(src: &ModelSource) -> Result<&'static dyn Carrier, String> {
    let mut matches = crate::REGISTRY.iter().copied().filter(|c| c.probe(src));
    let carrier = matches
        .next()
        .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
    if let Some(other) = matches.next() {
        return Err(format!(
            "ambiguous carrier dispatch for {}: '{}' and '{}' both claim it",
            src.describe(),
            carrier.name(),
            other.name()
        ));
    }
    Ok(carrier)
}

/// Pure DFlash lm-head refusal: `Some(message)` when a draft is attached but the
/// target's lm_head/embed quant type is not admitted for DFlash's batched
/// lm-head verify on `gpu_arch`. MQ V2 (qt 44/47/48/49/50) and MQ3 (qt 17)
/// lm_heads reach `gemm_mq*g256v2_batched_lmhead` / the MQ3 batched lm-head in
/// the DFlash verify; DFlash admits them only on the gfx11/gfx12 WMMA set, where
/// that path is validated. gfx1030 (RDNA2, no WMMA) is outside it (#771): the
/// generic non-WMMA x-batch scalar fallback exists in rdna-compute but has not
/// been validated for DFlash verify, so the load refuses here instead of
/// failing at the first `spec_step` after the prior model was torn down.
fn dflash_lm_head_refusal(lm_qt: Option<u8>, gpu_arch: &str) -> Option<String> {
    if crate::dflash_lm_head_quant_supported(lm_qt, gpu_arch) {
        return None;
    }
    let qt_desc = match lm_qt {
        Some(44) => "quant_type=44, MQ4G256V2".to_string(),
        Some(47) => "quant_type=47, MQ6G256V2".to_string(),
        Some(48) => "quant_type=48, MQ5G256V2".to_string(),
        Some(49) => "quant_type=49, MQ3G256V2".to_string(),
        Some(50) => "quant_type=50, MQ2G256V2".to_string(),
        Some(17) => "quant_type=17, MQ3G256".to_string(),
        Some(qt) => format!("quant_type={qt}"),
        None => "no lm_head/embed_tokens tensor found".to_string(),
    };
    Some(format!(
        "DFlash draft requested but batched lm_head admission for this model's output layer \
         ({qt_desc}) is unsupported on {gpu_arch}: DFlash's batched lm_head verify is validated \
         only on gfx11/gfx12 WMMA GPUs for this weight format. Run autoregressive decoding \
         instead: turn DFlash off (speculation.dflash = off, e.g. `hipfire config <model> set \
         dflash_mode off` or `--spec off`; unset developer.dflash_draft / \
         HIPFIRE_DFLASH_DRAFT), or load a build of the model whose output layer is \
         Q8/HFQ4/MQ4G256 (qt 3/6/13). Autoregressive decode of this model is unaffected"
    ))
}

/// Read-only DFlash lm-head quant refusal: a draft is attached but the target's
/// lm_head/embed quant type is not admitted for the batched GEMM verify paths.
/// Mirrors the gemma4-entry pre-allocation check (lib.rs) so the refusal fires
/// at admission instead of after prior-model teardown.
fn df_lash_lm_head_admission(
    hfq: &hipfire_runtime::hfq::HfqFile,
    draft_path: Option<&str>,
    gpu_arch: &str,
) -> Result<(), String> {
    if draft_path.is_none() {
        return Ok(());
    }
    let lm_qt = hfq
        .tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
        .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
        .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
        .map(|(info, _)| info.quant_type);
    match dflash_lm_head_refusal(lm_qt, gpu_arch) {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// FLUX/Klein image-gen arch refusal: the trunk GEMM (`gemm_wmma_lds256`)
/// and `attention_flux_vtk/v2_wmma` use the gfx11
/// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` intrinsic, which hipcc
/// rejects on gfx12 ("needs target feature wmma-256b-insts,wavefrontsize32").
/// Admits exactly the `has_wmma_w32` set (`arch_caps.rs:143-145`: `is_rdna3`,
/// NOT the `has_wmma_w32_gfx12` gfx12 variant) — pure on the `gpu_arch`
/// string because admission is read-only and never inits a GPU. `None` for
/// non-diffusion archs and for gfx11; `Some(reason)` otherwise, so the load
/// refuses before any allocation with the prior model still loaded.
fn flux_arch_refusal(arch_id: u32, gpu_arch: &str) -> Option<String> {
    if !matches!(arch_id, 40 | 45) {
        return None;
    }
    let gfx11_wmma_w32 = matches!(
        gpu_arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152"
    );
    (!gfx11_wmma_w32).then(|| {
        format!(
            "image generation (arch 40/45) needs an RDNA3/3.5 GPU (gfx11 wave32 WMMA); \
             detected {gpu_arch}. See docs/IMAGEGEN.md §1."
        )
    })
}

/// #666 G11: Muse Glimmer (arch 14) is served on one device only until the
/// G13 dense axes give it a pipeline/tensor placement. Any parallel degree
/// (`pp`, `tp`, `topology.ep`) is refused here, before teardown or
/// allocation, instead of reaching a generic EP or carrier refusal or being
/// ignored (`topology.ep`).
fn muse_glimmer_topology_refusal(arch_id: u32, pp: usize, tp: usize, ep: usize) -> Option<String> {
    (arch_id == 14 && (pp > 1 || tp > 1 || ep > 1)).then(|| {
        format!(
            "muse_glimmer: single-device only; pp={pp}, tp={tp}, topology.ep={ep} requested \
             (no pipeline, tensor or expert parallel placement exists for arch 14)"
        )
    })
}

/// Pipeline/tensor-parallel geometry admission (#666 G2). `Gpus::init_uniform`
/// refuses `n_layers < n_devices` deep inside GPU init (`multi_gpu.rs`) —
/// after the daemon has already torn down the resident model for the pp path
/// — and `init_tp`/`init_ep` refuse a zero degree or an unsatisfiable device
/// count the same way. These gates mirror those deep invariants here, before
/// any teardown, so an impossible `pp`/`tp` leaves the prior model serving.
/// The `init_*` checks stay as last-resort invariants; this is an additional
/// earlier gate, not a replacement.
///
/// Reachability: `admit_source` holds the already-open source but no per-arch
/// config parser (the `config_from_hfq` readers live in the feature-gated
/// arch crates, which this unconditional module cannot import). `n_layers` is
/// therefore probed out of the HF `{config}` envelope (`num_hidden_layers`,
/// the HF-convention key every pp-capable source carries, descending into
/// `text_config` exactly as qwen35 `from_config_value` does) rather than
/// parsed per-arch. A source with no layer key fails this gate open (`None`)
/// — the visible-device gate below still guards it, and every non-qwen35
/// `pp > 1` is already refused by `admit_topology` before this runs.
fn n_layers_from_metadata_json(metadata_json: &str) -> Option<usize> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let config = meta.get("config").unwrap_or(&meta);
    // Composite checkpoints nest the trunk config under `text_config`
    // (mirrors qwen35 `from_config_value`, which descends there first).
    // Probe the nested node, then the outer node, then the flat blob.
    let text_config = config.get("text_config");
    for node in [text_config.unwrap_or(config), config, &meta] {
        for key in ["num_hidden_layers", "n_layers", "num_layers"] {
            if let Some(n) = node.get(key).and_then(|v| v.as_u64()) {
                if n > 0 {
                    return Some(n as usize);
                }
            }
        }
    }
    None
}

/// Layer count declared by an already-open source, via the metadata envelope
/// probe above. Read-only: parses the in-memory JSON the open already
/// returned; touches no GPU state.
fn source_n_layers(source: &ModelSource) -> Option<usize> {
    match source {
        ModelSource::Hfq(hfq) => n_layers_from_metadata_json(&hfq.metadata_json),
        ModelSource::Dir(dir) => n_layers_from_metadata_json(
            hipfire_runtime::model_source::ModelSource::metadata_json(dir),
        ),
    }
}
/// Qwen's trained window comes from the checkpoint's
/// `config.text_config.max_position_embeddings` (or flat `config`), not from
/// a registry tag or the serving default. The same metadata envelope feeds
/// `qwen35::config_from_metadata_json`.
fn trained_context(source: &ModelSource) -> Option<usize> {
    let raw = match source {
        ModelSource::Hfq(hfq) => &hfq.metadata_json,
        ModelSource::Dir(dir) => hipfire_runtime::model_source::ModelSource::metadata_json(dir),
    };
    let meta: serde_json::Value = serde_json::from_str(raw).ok()?;
    let config = meta.get("config").unwrap_or(&meta);
    config
        .get("text_config")
        .unwrap_or(config)
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0)
}

/// Smallest projected context an automatic single-card Qwen VMM load accepts
/// ([`resolve_sequence`]); below it the load is refused before teardown.
pub const QWEN35_MIN_AUTO_CONTEXT: usize = 512;

/// VMM page rounding reserved across the K/V owners in the projection.
const QWEN35_VMM_PAGE_ROUNDING_BYTES: usize = 128 << 20;

/// Largest context `t` in `[0, cap]` whose mapped-KV growth plus the
/// per-request prefill `transient` fits in `free` bytes. Measured after the
/// whole Qwen load (weights, KV owners' initial prefix, DeltaNet, scratch,
/// drafts) so every load-time owner is already charged in `free`. `growth(t)`
/// is the physical bytes still to map to hold `t` tokens; it is monotone in
/// `t`, which makes the answer a binary search. Returns 0 when not even the
/// empty context fits; a `growth` error propagates unchanged.
pub fn qwen35_measured_fit(
    free: usize,
    transient: usize,
    cap: usize,
    mut growth: impl FnMut(usize) -> Result<usize, String>,
) -> Result<usize, String> {
    let mut fits =
        |t: usize| -> Result<bool, String> { Ok(growth(t)?.saturating_add(transient) <= free) };
    if !fits(0)? {
        return Ok(0);
    }
    // Invariant: `lo` fits; `hi` is the largest candidate.
    let (mut lo, mut hi) = (0usize, cap);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid)? {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(lo)
}

#[cfg(test)]
mod qwen35_measured_fit_tests {
    use super::qwen35_measured_fit;

    /// Linear growth with 100 B/token, mapped in whole 1000 B chunks above a
    /// 500 B already-mapped prefix (monotone step function).
    fn stepped(t: usize) -> Result<usize, String> {
        let need = t * 100;
        Ok(if need <= 500 {
            0
        } else {
            (need - 500).div_ceil(1000) * 1000
        })
    }

    #[test]
    fn qwen35_measured_fit_exact_boundary() {
        // free - transient = 3000 -> growth <= 3000 -> need <= 3500 -> t = 35.
        assert_eq!(qwen35_measured_fit(3200, 200, 10_000, stepped), Ok(35));
        // One byte short of the boundary drops a whole chunk (t = 25).
        assert_eq!(qwen35_measured_fit(3199, 200, 10_000, stepped), Ok(25));
        // Linear growth: exact byte boundary, no rounding.
        assert_eq!(
            qwen35_measured_fit(1000, 0, 10_000, |t| Ok(t * 10)),
            Ok(100)
        );
        assert_eq!(qwen35_measured_fit(999, 0, 10_000, |t| Ok(t * 10)), Ok(99));
    }

    #[test]
    fn qwen35_measured_fit_is_bounded_by_cap() {
        assert_eq!(
            qwen35_measured_fit(usize::MAX, 0, 777, |t| Ok(t * 10)),
            Ok(777)
        );
        assert_eq!(qwen35_measured_fit(1 << 40, 0, 0, |t| Ok(t)), Ok(0));
    }

    #[test]
    fn qwen35_measured_fit_nothing_fits_is_zero() {
        // Transient alone exceeds free.
        assert_eq!(qwen35_measured_fit(100, 101, 4096, |_| Ok(0)), Ok(0));
        // Growth of even the empty context exceeds free.
        assert_eq!(qwen35_measured_fit(100, 0, 4096, |_| Ok(101)), Ok(0));
        assert_eq!(qwen35_measured_fit(0, 0, 4096, |t| Ok(t + 1)), Ok(0));
    }

    #[test]
    fn qwen35_measured_fit_propagates_growth_error() {
        let err = qwen35_measured_fit(1 << 30, 0, 4096, |_| Err("boom".to_string()));
        assert_eq!(err, Err("boom".to_string()));
        // An error raised only deep in the search still surfaces.
        let err = qwen35_measured_fit(1 << 30, 0, 4096, |t| {
            if t > 1000 {
                Err("late".to_string())
            } else {
                Ok(0)
            }
        });
        assert_eq!(err, Err("late".to_string()));
    }
}

/// Only Qwen's single-card cache has the matching VMM KV-row accounting.
/// The HFQ file is a lower bound on loaded weights; include 12.5% for
/// expanded raw tensors/runtime state and 512 MiB for allocator/transients.
/// Reserve only the minimum 512-row PBS plus the widened guard's existing
/// 1 GiB headroom and 128 MiB for VMM page rounding across K/V owners.
/// Per-request prefill selects wider scratch from then-current mapped-KV free VRAM.
///
/// The one projection admission refuses on ([`resolve_sequence`]); `hipfire
/// fit` reads the same value. It is a pre-teardown fit check, never the
/// effective bound: the Qwen carrier remeasures free VRAM after the weights
/// upload, immediately before KV reservation.
pub fn qwen35_card_projection(
    source: &ModelSource,
    gpu_arch: &str,
    hints: KvBackendHints<'_>,
) -> Result<Qwen35CardProjection, String> {
    use hipfire_runtime::kv_mode;
    let ModelSource::Hfq(hfq) = source else {
        return Err("automatic Qwen VMM context sizing requires an HFQ weight manifest".into());
    };
    let config = hipfire_arch_qwen35::qwen35::config_from_metadata_json(&hfq.metadata_json)?;
    let model_ctx = trained_context(source)
        .ok_or("Qwen checkpoint has no config.text_config.max_position_embeddings")?;
    let raw_mode = hints
        .kv_mode
        .unwrap_or(hipfire_runtime::config::get().kv_mode.as_str());
    let adaptive = hints.kv_adaptive.is_some_and(|v| !matches!(v, "" | "off"));
    let auto_native = kv_mode::qwen35_native_auto_eligible(
        gpu_arch,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        1,
        adaptive,
        hints.cask.is_some_and(|c| c.sidecar.is_some()),
    );
    let policy =
        kv_mode::qwen35_policy_for_native(&kv_mode::QWEN35_HFQ_POLICY, raw_mode, auto_native);
    let pair = kv_mode::resolve_kv_pair(
        raw_mode,
        hints.kv_k,
        hints.kv_v,
        &policy,
        gpu_arch,
        hints.qwen_default_q8,
    )
    .map_err(|e| format!("Qwen KV sequence: {e}"))?;
    let bytes_per_token =
        hipfire_arch_qwen35::qwen35::prefill::vmm_kv_token_bytes(&config, pair, adaptive)
            .ok_or("Qwen KV token stride overflow or no full-attention layers")?;
    let free = hints
        .free_vram_bytes
        .ok_or("cannot query free VRAM for automatic VMM context")?;
    let file_bytes = usize::try_from(
        std::fs::metadata(hfq.path())
            .map_err(|e| format!("Qwen weight size: {e}"))?
            .len(),
    )
    .map_err(|_| "Qwen weight file size exceeds usize")?;
    let weights = file_bytes
        .saturating_add(file_bytes / 8)
        .saturating_add(512 << 20);
    let scratch =
        hipfire_arch_qwen35::qwen35::prefill::minimum_prefill_reservation_bytes(&config, gpu_arch)
            .ok_or("Qwen minimum prefill scratch sizing overflow")?;
    let card_cap = free
        .saturating_sub(weights)
        .saturating_sub(scratch)
        .saturating_sub(QWEN35_VMM_PAGE_ROUNDING_BYTES)
        / bytes_per_token;
    Ok(Qwen35CardProjection {
        free,
        weights,
        scratch,
        page_rounding: QWEN35_VMM_PAGE_ROUNDING_BYTES,
        bytes_per_token,
        card_cap,
        model_ctx,
        kv_mode: kv_mode::qwen_k_display_name(pair.k()),
    })
}

/// Legacy KV contiguity margin: the legacy backend allocates the full admitted
/// context contiguously at load while VMM grows on demand, so free-total
/// overstates what one allocation can take. Start 512 MiB; tune from the
/// live test.
const QWEN35_LEGACY_CONTIGUITY_MARGIN_BYTES: usize = 512 << 20;

/// Measured automatic context for one legacy-backend Qwen3.5 dense load.
struct Qwen35LegacyFit {
    max_seq: usize,
    card_cap: usize,
    model_ctx: usize,
}

/// Measured automatic context for the legacy KV backend, or `None` where the
/// shape is outside the admitted set (fixed 4096 survives there).
///
/// Backend-gated, not OS-gated: any load that resolves `KvBackend::Legacy`
/// measures — explicit `--kv-backend legacy` on any OS, or automatic with a
/// VMM refusal (including Windows, where VMM mapping/graph semantics are
/// uncertified and automatic therefore lands here). Start: Qwen3.5 dense;
/// MoE keeps its existing behavior, and Qwen4 keeps its own legacy refusal
/// (both gated out by the caller).
fn qwen35_legacy_fit(
    source: &ModelSource,
    gpu_arch: &str,
    hints: KvBackendHints<'_>,
) -> Result<Option<Qwen35LegacyFit>, String> {
    let ModelSource::Hfq(hfq) = source else {
        return Ok(None);
    };
    if hints.free_vram_bytes.is_none() {
        return Ok(None);
    }
    let config = hipfire_arch_qwen35::qwen35::config_from_metadata_json(&hfq.metadata_json)
        .map_err(|e| format!("Qwen legacy context config: {e}"))?;
    if config.num_experts != 0 {
        return Ok(None);
    }
    let projection = qwen35_card_projection(source, gpu_arch, hints)?;
    // Same parts as the VMM projection, plus the contiguity margin. No
    // vision term, matching the VMM projection (the Qwen35 reserve is zero
    // on this path); the margin covers tower scratch.
    let room = projection
        .free
        .saturating_sub(projection.weights)
        .saturating_sub(projection.scratch)
        .saturating_sub(projection.page_rounding)
        .saturating_sub(QWEN35_LEGACY_CONTIGUITY_MARGIN_BYTES);
    let card_cap = room / projection.bytes_per_token;
    if card_cap < QWEN35_MIN_AUTO_CONTEXT {
        return Err("Qwen legacy load refused: the model weights, the minimum prefill scratch and the KV cache do not fit in free VRAM. Free GPU memory, set a smaller memory.max_seq, or use a smaller quantization of this model".into());
    }
    Ok(Some(Qwen35LegacyFit {
        max_seq: projection.model_ctx.min(card_cap),
        card_cap,
        model_ctx: projection.model_ctx,
    }))
}

fn resolve_sequence(
    source: &ModelSource,
    requested: usize,
    gpu_arch: &str,
    hints: KvBackendHints<'_>,
) -> Result<SequenceResolution, String> {
    let projection = qwen35_card_projection(source, gpu_arch, hints)?;
    // Projection is a pre-teardown fit refusal, never the effective bound.
    // Weight upload changes free VRAM; the Qwen carrier remeasures that
    // *after* weights and immediately before KV reservation.
    if projection.card_cap < QWEN35_MIN_AUTO_CONTEXT && requested == 0 {
        return Err("Qwen VMM load refused: the model weights, the minimum prefill scratch and the KV cache do not fit in free VRAM. Free GPU memory, set a smaller memory.max_seq, or use a smaller quantization of this model".into());
    }
    let (max_seq, bound) = if requested != 0 {
        (requested, "user")
    } else {
        (projection.model_ctx, "pending")
    };
    Ok(SequenceResolution {
        max_seq,
        bound,
        model_ctx: projection.model_ctx,
        card_cap: 0,
        kv_mode: projection.kv_mode,
        projection: Some(projection),
    })
}

/// Count a comma-separated device list (`hardware.devices`, visibility envs).
fn count_device_list(value: &str) -> usize {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .count()
}

/// Visible HIP device count for the pp/tp geometry gates. Read-only
/// precedence: `hardware.devices` (the same source
/// `Gpus::resolve_device_ids` consults) first, then the visibility envs the
/// daemon inherits, then a `hipGetDeviceCount` query — which initializes no
/// device, creates no context, and allocates nothing. `None` = unknown (e.g.
/// CPU-only unit tests) and the device-count gate is skipped; the `n_layers`
/// gate still guards `pp`.
fn visible_device_count() -> Option<usize> {
    if let Some(devices) = hipfire_runtime::config::get().devices.as_deref() {
        return Some(count_device_list(devices));
    }
    for var in [
        hipfire_config::devices::HIP_VISIBLE_DEVICES,
        hipfire_config::devices::ROCR_VISIBLE_DEVICES,
    ] {
        if let Ok(value) = std::env::var(var) {
            if !value.trim().is_empty() {
                return Some(count_device_list(&value));
            }
        }
    }
    hip_bridge::HipRuntime::load()
        .and_then(|hip| hip.device_count())
        .map(|n| n.max(0) as usize)
        .ok()
}

/// Pipeline-parallel geometry refusal. `pp == 0` can never satisfy
/// `init_uniform` (`n_devices must be >= 1`); `pp` above the source's layer
/// count can never satisfy its `n_layers < n_devices` invariant (each device
/// must own at least one layer); `pp` above the visible device count can
/// never bind its ranks. `pp == 1` always passes: a single device owns every
/// layer by construction.
fn pp_geometry_refusal(
    pp: usize,
    n_layers: Option<usize>,
    n_devices: Option<usize>,
) -> Option<String> {
    if pp == 0 {
        return Some(
            "load refused: topology.pp=0 is not a valid pipeline-parallel degree; use 1 for a \
             single GPU or the number of GPUs to split across"
                .to_string(),
        );
    }
    if let Some(layers) = n_layers {
        if pp > layers {
            return Some(format!(
                "load refused: pp={pp} exceeds n_layers={layers} for this source — each GPU \
                 needs at least one model layer, so lower topology.pp to at most {layers}"
            ));
        }
    }
    if let Some(devices) = n_devices {
        if pp > devices {
            return Some(format!(
                "load refused: pp={pp} exceeds visible HIP device count={devices} — pipeline \
                 parallelism needs one GPU per stage, so lower topology.pp or make more GPUs \
                 visible (hardware.devices)"
            ));
        }
    }
    None
}

/// Expert/tensor-parallel geometry refusal. Unlike `pp`, `tp` shards
/// within-layer work (`init_tp`/`init_ep` run every layer on every rank), so
/// the layer count does not bound it — only the zero degree
/// (`init_ep: ep_size must be >= 1`, `init_tp: tp_size must be >= 1`) and the
/// visible device count (one device per rank) do. `tp == 1` always passes.
fn tp_geometry_refusal(tp: usize, n_devices: Option<usize>) -> Option<String> {
    if tp == 0 {
        return Some(
            "load refused: tp=0 is not a valid parallel degree; use 1 for a single GPU or the \
             number of GPUs to split across"
                .to_string(),
        );
    }
    if let Some(devices) = n_devices {
        if tp > devices {
            return Some(format!(
                "load refused: tp={tp} exceeds visible HIP device count={devices} — parallel \
                 loads need one GPU per rank, so lower the degree or make more GPUs visible \
                 (hardware.devices)"
            ));
        }
    }
    None
}

/// Numerical-policy identity of every route that has not declared its own.
pub const LEGACY_NUMERICAL_POLICY: &str = "legacy";

/// Normalize raw parallel degrees into the one rectangular [`DeviceMesh`].
/// Size-one axes are dropped (`pp=tp=ep=1` is the empty single-device mesh).
/// Zero degrees and cardinality overflow are refused here instead of being
/// clamped to one by `DeviceMesh::rect`.
pub fn normalize_mesh_shape(pp: usize, tp: usize, ep: usize) -> Result<DeviceMesh, String> {
    if pp == 0 || tp == 0 || ep == 0 {
        return Err(format!(
            "mesh degree is zero (pp={pp}, tp={tp}, ep={ep}); every parallel degree must be >= 1"
        ));
    }
    let axes: Vec<(DimKind, usize)> = [(DimKind::Pp, pp), (DimKind::Tp, tp), (DimKind::Ep, ep)]
        .into_iter()
        .filter(|&(_, size)| size > 1)
        .collect();
    DeviceMesh::rect(&axes).map_err(|e| format!("mesh shape pp={pp}, tp={tp}, ep={ep}: {e}"))
}

/// Physical identity of the visible devices in HIP logical order, from the
/// KFD topology records the visibility resolver already produced. The HIP
/// ordinal is the position (HIP always sees `0..N-1` inside the ROCr-filtered
/// set); the PCI id is what survives remapping.
pub fn visible_ranks(devices: &[hipfire_config::devices::GpuDevice]) -> Vec<AdmittedRank> {
    devices
        .iter()
        .enumerate()
        .map(|(logical, device)| AdmittedRank {
            logical,
            device_ordinal: logical as i32,
            pci_bus_id: Some(device.bdf.to_string()),
            arch: device.arch.clone(),
        })
        .collect()
}

/// The visible devices admission binds ranks to. Uses the KFD identity when
/// it covers `needed` ranks (read-only sysfs; no HIP init, no context, no
/// allocation). Otherwise (no readable topology, e.g. CPU-only unit tests,
/// WSL2/ROCDXG) ranks carry no PCI identity and the daemon's device arch:
/// per-device architectures are verified against real devices at load.
fn visible_device_ranks(gpu_arch: &str, needed: usize) -> Vec<AdmittedRank> {
    if let Some(devices) = hipfire_config::devices::startup_devices() {
        if devices.len() >= needed {
            return visible_ranks(&devices);
        }
    }
    let count = visible_device_count().unwrap_or(needed).max(1);
    (0..count)
        .map(|logical| AdmittedRank {
            logical,
            device_ordinal: logical as i32,
            pci_bus_id: None,
            arch: gpu_arch.to_string(),
        })
        .collect()
}

/// How logical ranks bind to the visible devices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RankBinding {
    /// Logical rank `r` runs on visible device `r`: every route that existed
    /// before the admitted mesh (Qwen3.5 PP/EP/TP, Qwen4 EP, Single).
    Visible,
    /// `HIPFIRE_EMULATE_GPUS`: rank `r` aliases onto visible device
    /// `r % visible` (the `multi_gpu::alias_ids` rule); ranks may share a card.
    Emulated,
    /// The admitted generic PP route: the first visible device (the primary,
    /// output-owning card) is the LAST logical stage, so logical stages stay
    /// in increasing layer order. Pure pipeline meshes only.
    OutputCardLast,
}

/// Per-edge transport admission for routes that provision edges.
pub struct EdgeAdmission<'a> {
    pub policy: EdgeTransportPolicy,
    /// Whether a peer read from logical rank `src` to logical rank `dst` was
    /// byte-qualified. Pure: admission never probes peer access.
    pub qualified_peer: &'a dyn Fn(usize, usize) -> bool,
}

/// Directed PP edges `(src, dst)` as logical rank (flat mesh device) ids: every
/// device sends to the device with the same TP/EP coordinates one PP stage on.
fn pp_edges(mesh: &DeviceMesh) -> Result<Vec<(usize, usize)>, String> {
    let Some(axis) = mesh.axes().iter().position(|a| a.kind == DimKind::Pp) else {
        return Ok(Vec::new());
    };
    let stages = mesh.axes()[axis].size;
    let mut edges = Vec::new();
    for dev in 0..mesh.n_devices() {
        let mut coord = mesh.coord_of(dev).map_err(|e| e.to_string())?;
        if coord[axis] + 1 < stages {
            coord[axis] += 1;
            edges.push((dev, mesh.device_of(&coord).map_err(|e| e.to_string())?));
        }
    }
    Ok(edges)
}

/// Refuse an enumeration with a negative or repeated ordinal or a repeated PCI id.
fn validate_distinct_visible(visible: &[AdmittedRank]) -> Result<(), String> {
    let mut ordinals = std::collections::HashSet::new();
    let mut pci_ids = std::collections::HashSet::new();
    for rank in visible {
        if rank.device_ordinal < 0 || !ordinals.insert(rank.device_ordinal) {
            return Err("visible devices contain an invalid or duplicate ordinal".to_string());
        }
        if let Some(pci) = &rank.pci_bus_id {
            if !pci_ids.insert(pci.as_str()) {
                return Err("visible devices contain a duplicate PCI identity".to_string());
            }
        }
    }
    Ok(())
}

/// Bind a normalized mesh shape to physical devices: the single admission
/// value. Pure: no HIP, no peer probing, no allocation of device resources.
/// `edges: None` keeps the route's existing transports (every pre-admitted-mesh
/// route); `Some` records one transport decision per directed PP edge.
pub fn admit_mesh(
    shape: DeviceMesh,
    family: &str,
    numerical_policy: &str,
    visible: &[AdmittedRank],
    binding: RankBinding,
    edges: Option<EdgeAdmission<'_>>,
) -> Result<AdmittedMesh, String> {
    let needed = shape.n_devices();
    validate_distinct_visible(visible)?;
    let too_few = || {
        format!(
            "mesh needs {needed} visible devices, {} visible",
            visible.len()
        )
    };
    let ranks: Vec<AdmittedRank> = match binding {
        RankBinding::Visible => {
            if visible.len() < needed {
                return Err(too_few());
            }
            visible[..needed]
                .iter()
                .enumerate()
                .map(|(logical, rank)| AdmittedRank {
                    logical,
                    ..rank.clone()
                })
                .collect()
        }
        RankBinding::Emulated => {
            if visible.is_empty() {
                return Err(too_few());
            }
            (0..needed)
                .map(|logical| AdmittedRank {
                    logical,
                    ..visible[logical % visible.len()].clone()
                })
                .collect()
        }
        RankBinding::OutputCardLast => {
            if shape.size_of(DimKind::Tp) != 1 || shape.size_of(DimKind::Ep) != 1 {
                return Err(
                    "output-card-last rank order is defined for pure pipeline meshes only"
                        .to_string(),
                );
            }
            if visible.len() < needed {
                return Err(too_few());
            }
            pp_rank_order(&visible[..needed])?
        }
    };
    let edges = match edges {
        None => Vec::new(),
        Some(edge) => {
            resolve_edge_transports(edge.policy, edge.qualified_peer, &pp_edges(&shape)?)?
        }
    };
    Ok(AdmittedMesh {
        mesh: shape,
        ranks,
        edges,
        family: family.to_string(),
        numerical_policy: numerical_policy.to_string(),
    })
}

/// Generic per-stage layer counts (`HIPFIRE_PP_LAYERS`), read once at
/// admission. Empty is unset.
fn pp_layers_request() -> Option<String> {
    hipfire_config::developer_var("HIPFIRE_PP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Normalized pipeline bands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PpBands {
    pub placement: BandPlacement,
    /// Counts the operator declared; `None` for the uniform default.
    pub explicit_counts: Option<Vec<usize>>,
}

/// Normalize the pipeline split once. `requested` is the raw
/// `HIPFIRE_PP_LAYERS` text (`None` = unset/empty). Counts are in logical
/// stage order and must sum to the trunk layer count. Every legacy load-time
/// verdict keeps its exact text; they now fire before any teardown.
pub fn normalize_pp_bands(
    mesh: &DeviceMesh,
    n_layers: usize,
    requested: Option<&str>,
) -> Result<PpBands, String> {
    let pp = mesh.size_of(DimKind::Pp);
    let Some(spec) = requested else {
        if n_layers < pp {
            return Err(format!(
                "load refused: pp={pp} exceeds n_layers ({n_layers}) — each GPU needs at least one model layer, so lower topology.pp to at most {n_layers}"
            ));
        }
        return Ok(PpBands {
            placement: BandPlacement::uniform(mesh, n_layers)?,
            explicit_counts: None,
        });
    };
    let counts: Result<Vec<usize>, _> =
        spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
    let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
    if counts.len() != pp {
        return Err(format!(
            "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
            counts.len(),
            pp
        ));
    }
    let sum = counts.iter().fold(0usize, |acc, &c| acc.saturating_add(c));
    if sum != n_layers {
        return Err(format!(
            "HIPFIRE_PP_LAYERS sum={sum} != n_layers={n_layers}"
        ));
    }
    if counts.contains(&0) {
        return Err(
            "load refused: every pipeline stage needs at least one layer, but a stage was given zero".to_string(),
        );
    }
    Ok(PpBands {
        placement: BandPlacement::from_counts(mesh, n_layers, &counts)?,
        explicit_counts: Some(counts),
    })
}

/// Developer variable selecting how pipeline edges move bytes:
/// `auto` (default), `peer` or `host`.
pub const MESH_TRANSPORT_ENV: &str = "HIPFIRE_MESH_TRANSPORT";

/// Numerical-policy identity of the admitted Qwen4 PP route: every band edge
/// executes the cut-safe Single schedule (no fusion state across a cut).
pub const QWEN4_PP_NUMERICAL_POLICY: &str = "qwen4-pp-cut-safe-v1";

/// The one refusal sentence of every feature the Qwen4 PP route does not run.
pub(crate) fn qwen4_pp_unsupported(pp: usize, what: &str) -> String {
    format!(
        "qwen4: pipeline parallelism (topology.pp = {pp}) does not support {what}; drop the option or set topology.pp = 1"
    )
}

/// Parse [`MESH_TRANSPORT_ENV`] (`None`/empty = `auto`).
pub fn parse_mesh_transport(raw: Option<&str>) -> Result<EdgeTransportPolicy, String> {
    match raw.map(str::trim) {
        None | Some("") | Some("auto") => Ok(EdgeTransportPolicy::Auto),
        Some("peer") => Ok(EdgeTransportPolicy::Peer),
        Some("host") => Ok(EdgeTransportPolicy::Host),
        Some(other) => Err(format!(
            "{MESH_TRANSPORT_ENV}={other:?} is not one of auto, peer, host"
        )),
    }
}

/// The MTP n-gram modifier does not run on a pipeline mesh. With native MTP
/// on, an explicit request for it (`speculation.mtp_ngram = on`, i.e.
/// `hipfire_config::mtp_ngram_enabled()`) refuses before any allocation;
/// `auto` and `off` are accepted and the load builds the speculator without
/// an n-gram pool (`auto` would arm it on gfx1201, so the PP default is
/// pinned off). Inert without native MTP. `pp <= 1` never refuses.
pub(crate) fn qwen4_pp_mtp_ngram_refusal(
    pp: usize,
    native_mtp: bool,
    explicit_on: bool,
) -> Result<(), String> {
    if pp > 1 && native_mtp && explicit_on {
        return Err(qwen4_pp_unsupported(
            pp,
            "the MTP n-gram modifier (speculation.mtp_ngram = on; leave it auto or off)",
        ));
    }
    Ok(())
}

/// Features the Qwen4 PP route does not run, refused from the load options
/// before any source, device or cache work. `pp <= 1` never refuses.
pub(crate) fn qwen4_pp_option_refusal(
    pp: usize,
    draft_path: Option<&str>,
    options: &SourceAdmissionOptions,
) -> Result<(), String> {
    if pp <= 1 {
        return Ok(());
    }
    let spec = options.spec;
    let unsupported = [
        (draft_path.is_some() || spec.dflash == Some(true), "DFlash"),
        (spec.dspark == Some(true), "DSpark"),
        (spec.ngram_draft == Some(true), "n-gram drafting"),
        (crate::carriers::qwen4_ddtree_requested(spec), "DDTree"),
        (options.kv_adaptive, "adaptive KV"),
        (options.eagle_drafter, "an EAGLE drafter"),
        (options.cask, "CASK"),
        (options.pflash, "PFlash"),
        (
            options.expert_count_override,
            "an experts-per-token override",
        ),
        (options.non_single_compute, "a non-Single compute placement"),
    ];
    match unsupported.iter().find(|(requested, _)| *requested) {
        Some((_, what)) => Err(qwen4_pp_unsupported(pp, what)),
        None => Ok(()),
    }
}

/// Explicit environment/config requests the Qwen4 PP route does not run:
/// host-mapped band layers, the expert hotset, expert staging, turn snapshots
/// (`HIPFIRE_QWEN4_TURN_SNAPSHOTS=1`), the session-cache disk tier and the NPU
/// trunk (`npu` is the already resolved mode for the rank architecture).
/// Unset values are accepted: the PP route never enables any of them. The
/// in-memory session cache is not refused: every band attaches its own
/// (`memory.session_cache_bytes`, opportunistic and uncharged exactly as in
/// Single; only the disk tier and turn snapshots stay refused). `pp <= 1`
/// never refuses.
pub(crate) fn qwen4_pp_env_refusal(
    pp: usize,
    env: &dyn Fn(&str) -> Option<String>,
    npu: hipfire_arch_qwen4::npu_trunk::NpuTrunkMode,
    session_cache_disk_bytes: u64,
) -> Result<(), String> {
    use hipfire_arch_qwen4::npu_trunk::NpuTrunkMode;
    if pp <= 1 {
        return Ok(());
    }
    let hotset = hipfire_arch_qwen4::expert_hotset::HOTSET_ENV;
    let vram_layers = hipfire_arch_qwen4::expert_residency::EXPERT_VRAM_LAYERS_ENV;
    let stage = hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV;
    let turns = "HIPFIRE_QWEN4_TURN_SNAPSHOTS";
    let unsupported = [
        (
            env(vram_layers).is_some(),
            format!("host-mapped routed experts ({vram_layers})"),
        ),
        (
            env(hotset).is_some_and(|value| value != "0"),
            format!("the expert hotset ({hotset}; set 0)"),
        ),
        (
            env(stage).as_deref() == Some("1"),
            format!("expert staging ({stage})"),
        ),
        (
            env(turns).as_deref() == Some("1"),
            format!("session snapshots ({turns})"),
        ),
        (
            session_cache_disk_bytes > 0,
            "the session cache disk tier (memory.session_cache_disk_bytes; set 0)".to_string(),
        ),
        (
            matches!(npu, NpuTrunkMode::On | NpuTrunkMode::Emulate),
            format!("the NPU trunk (fusions.npu_trunk = {})", npu.as_str()),
        ),
    ];
    match unsupported.iter().find(|(requested, _)| *requested) {
        Some((_, what)) => Err(qwen4_pp_unsupported(pp, what)),
        None => Ok(()),
    }
}

/// The admitted ranks of a Qwen4 PP mesh: known PCI identity on every rank,
/// one homogeneous gfx1100 or gfx1201, and the output owner (last logical
/// rank = the daemon's device 0) agreeing with the daemon's architecture.
pub(crate) fn qwen4_pp_rank_refusal(
    pp: usize,
    ranks: &[AdmittedRank],
    gpu_arch: &str,
) -> Result<(), String> {
    let prefix = format!("qwen4: pipeline parallelism (topology.pp = {pp})");
    let Some(first) = ranks.first() else {
        return Err(format!("{prefix} needs at least one rank"));
    };
    for rank in ranks {
        if rank.pci_bus_id.is_none() {
            return Err(format!(
                "{prefix} needs a known PCI identity for every rank; rank {} (device {}) has none (KFD topology unreadable)",
                rank.logical, rank.device_ordinal
            ));
        }
        if !qwen4_ep_arch_admitted(&rank.arch, false) {
            return Err(format!(
                "{prefix} requires gfx1100 or gfx1201 on every rank; rank {} is {}",
                rank.logical, rank.arch
            ));
        }
        if rank.arch != first.arch {
            return Err(format!(
                "{prefix} requires the same architecture on every rank; rank 0 is {}, rank {} is {}",
                first.arch, rank.logical, rank.arch
            ));
        }
    }
    if first.arch != gpu_arch {
        return Err(format!(
            "{prefix} requires every rank to match the primary device; ranks are {}, device 0 is {gpu_arch}",
            first.arch
        ));
    }
    Ok(())
}

/// The admitted Qwen4 PP mesh and bands. Pure: no HIP, no peer probing, no
/// allocation. `visible` is the HIP-order device identity; ranks bind
/// `OutputCardLast` (the primary device owns the output band). Transport per
/// directed edge comes from `policy` and `qualified_peer` (logical rank ids,
/// pre-side-effect byte-qualification evidence: there is none in production,
/// so `auto` resolves to host-mapped and a forced `peer` refuses). Bands come
/// from `requested_layers` (`HIPFIRE_PP_LAYERS`) or the uniform split, with
/// the PLE layer pinned to the first band (`with_first_band_min(2)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn admit_qwen4_pp_mesh(
    pp: usize,
    tp: usize,
    ep: usize,
    n_layers: usize,
    visible: &[AdmittedRank],
    gpu_arch: &str,
    emulated: bool,
    policy: EdgeTransportPolicy,
    qualified_peer: &dyn Fn(usize, usize) -> bool,
    requested_layers: Option<&str>,
) -> Result<(AdmittedMesh, PpBands), String> {
    let prefix = format!("qwen4: pipeline parallelism (topology.pp = {pp})");
    let stages = hipfire_arch_qwen4::EffectiveMesh::PP_MIN_DEGREE
        ..=hipfire_arch_qwen4::EffectiveMesh::PP_MAX_DEGREE;
    if !stages.contains(&pp) || tp != 1 || ep != 1 {
        return Err(format!(
            "{prefix} is admitted for 2..=8 stages with tp = ep = 1 (tp={tp}, ep={ep})"
        ));
    }
    if emulated {
        return Err(format!(
            "{prefix} owns physical devices; HIPFIRE_EMULATE_GPUS aliasing is not admitted"
        ));
    }
    if visible.len() < pp {
        return Err(format!(
            "{prefix} needs {pp} visible devices, {} visible (hardware.devices / HIPFIRE_DEVICES)",
            visible.len()
        ));
    }
    let ordered = pp_rank_order(&visible[..pp]).map_err(|e| format!("{prefix}: {e}"))?;
    qwen4_pp_rank_refusal(pp, &ordered, gpu_arch)?;
    let shape = normalize_mesh_shape(pp, 1, 1).map_err(|e| format!("{prefix}: {e}"))?;
    let mesh = admit_mesh(
        shape,
        "qwen4",
        QWEN4_PP_NUMERICAL_POLICY,
        visible,
        RankBinding::OutputCardLast,
        Some(EdgeAdmission {
            policy,
            qualified_peer,
        }),
    )
    .map_err(|e| {
        if policy == EdgeTransportPolicy::Peer {
            format!(
                "{prefix}: peer-to-peer transfers between the pipeline GPUs were requested ({MESH_TRANSPORT_ENV}=peer) but could not be verified: {e}. Use {MESH_TRANSPORT_ENV}=auto or host"
            )
        } else {
            format!("{prefix}: {e}")
        }
    })?;
    let bands = normalize_pp_bands(&mesh.mesh, n_layers, requested_layers)
        .map_err(|e| format!("{prefix}: {e}"))?;
    let placement = bands
        .placement
        .with_first_band_min(QWEN4_PP_FIRST_BAND_MIN_LAYERS)
        .map_err(|e| format!("{prefix}: {e} (the PLE layer lives in the first band)"))?;
    Ok((
        mesh,
        PpBands {
            placement,
            explicit_counts: bands.explicit_counts,
        },
    ))
}

/// Read-only source admission: open the source, classify `arch_id` + vision,
/// decide the effective topology, and refuse every unsupported/contradictory
/// combination — without touching GPU state, VMM, or any prior model.
///
/// Refusals mirror the current-master daemon/loader refusals so no
/// currently-served route changes; they simply fire before destructive work.
/// Callers that do not expose the optional load-time controls get the
/// conservative [`SourceAdmissionOptions`] defaults.
pub fn admit_source(
    path: &str,
    tp: usize,
    pp: usize,
    request: KvBackendRequest,
    draft_path: Option<&str>,
    gpu_arch: &str,
    vision: Option<&str>,
    // Resolved `vision.mode` (`off`/`auto`/`on`) — gates `.vl` sibling
    // discovery in the carrier. `off` = never probe a sibling.
    vision_mode: &str,
    head: Option<&str>,
    max_seq: usize,
    hints: KvBackendHints<'_>,
) -> Result<SourceAdmission, String> {
    admit_source_with_options(
        path,
        tp,
        pp,
        request,
        draft_path,
        gpu_arch,
        vision,
        vision_mode,
        head,
        max_seq,
        hints,
        SourceAdmissionOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn admit_source_with_options(
    path: &str,
    tp: usize,
    pp: usize,
    request: KvBackendRequest,
    draft_path: Option<&str>,
    gpu_arch: &str,
    vision: Option<&str>,
    // Resolved `vision.mode` (`off`/`auto`/`on`), as in [`admit_source`].
    vision_mode: &str,
    head: Option<&str>,
    max_seq: usize,
    hints: KvBackendHints<'_>,
    options: SourceAdmissionOptions,
) -> Result<SourceAdmission, String> {
    // #666 G2: zero parallel degrees are never servable (see the geometry
    // gates above) and previously collapsed silently into `Single`, hiding
    // a meaningless request. Refuse before the source is even opened.
    if pp == 0 {
        return Err(pp_geometry_refusal(pp, None, None).expect("pp=0 always refuses"));
    }
    if tp == 0 {
        return Err(tp_geometry_refusal(tp, None).expect("tp=0 always refuses"));
    }
    let mut source = ModelSource::from_path(path)?;
    let arch_id = source
        .arch_id()
        .ok_or_else(|| format!("unrecognized source: {}", source.describe()))?;
    let kv_v = resolve_kv_v(hints.kv_v, arch_id, tp);
    let hints = KvBackendHints {
        kv_v: kv_v.as_deref(),
        ..hints
    };
    let is_dir = source.is_dir();

    // Refuse before backend selection or sequence budgeting, even without VMM
    // runtime/platform support; only an explicit legacy request opts out.
    if request != KvBackendRequest::Explicit(KvBackend::Legacy)
        && qwen35_gfx1151_fwht_k_override(
            gpu_arch,
            arch_id,
            hints
                .kv_mode
                .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.as_str()),
            hints.kv_k,
        )
    {
        return Err(
            "Qwen KV admission: no certified gfx1151 VMM FWHT path for the kv_k override; use --kv-backend legacy"
                .into(),
        );
    }

    // FLUX/Klein need gfx11 wave32 WMMA (the trunk GEMM and vtk/v2 use the
    // gfx11 WMMA intrinsic hipcc rejects on gfx12). Refuse here — before the
    // topology branch and any allocation — so the prior model stays loaded.
    if let Some(refusal) = flux_arch_refusal(arch_id, gpu_arch) {
        return Err(refusal);
    }
    if arch_id == 14 {
        let ep = hipfire_config::topology::ep()?;
        if let Some(refusal) = muse_glimmer_topology_refusal(arch_id, pp, tp, ep) {
            return Err(refusal);
        }
    }
    // Weight-host v1 (HIPFIRE_WEIGHT_HOST=1) imports only a dense Qwen3.5 HFQ
    // (arch 5) on one device; anything else would be a private load, which
    // the opt-in forbids. Refused before teardown, so the prior model stays.
    if hipfire_arch_qwen35::carrier::weight_host_socket()?.is_some()
        && (arch_id != 5 || is_dir || tp != 1 || pp != 1)
    {
        return Err(format!(
            "HIPFIRE_WEIGHT_HOST=1 only supports a single-GPU dense Qwen3.5 HFQ model (arch 5, tp=1, pp=1); this load is arch {arch_id}{}, tp={tp}, pp={pp}. Unset HIPFIRE_WEIGHT_HOST or load a supported model",
            if is_dir { " (directory source)" } else { "" }
        ));
    }
    // Qwen4 sequence on a unified-memory APU: an omitted max_seq is automatic
    // (the model's context, capped by the measured card capacity at load);
    // explicit values are honored within the limits below. A discrete GPU
    // keeps the default context for an omitted max_seq and no sequence.
    let unified_memory = hipfire_config::is_unified_memory_arch(gpu_arch);
    let max_seq = if arch_id == QWEN4_ARCH_ID {
        qwen4_requested_context(unified_memory, max_seq)
    } else {
        max_seq
    };
    let mut qwen35_ep_experts = None;
    let mut qwen4_sequence = None;
    let mut qwen4_pp: Option<(AdmittedMesh, PpBands)> = None;
    let mesh_shape: DeviceMesh;
    let (topology, carrier) = if arch_id == QWEN4_ARCH_ID {
        // Arch 16 is an executable local-path carrier, but only after its
        // complete source-only boundary succeeds. Keep this before vision/head
        // handling so every refusal remains pre-allocation.
        if max_seq > hipfire_arch_qwen4::QWEN4_MAX_CONTEXT {
            return Err(format!(
                "qwen4: max_seq {max_seq} exceeds the admitted context limit {}",
                hipfire_arch_qwen4::QWEN4_MAX_CONTEXT
            ));
        }
        let qwen4_kv =
            hipfire_runtime::kv_mode::resolve_qwen4(hints.kv_mode.unwrap_or(""), gpu_arch)?;
        // Expert-parallel degree: `topology.ep`, forced to 1 by the
        // `HIPFIRE_QWEN4_EP=0` kill switch. Resolved once, before any
        // teardown, and re-resolved identically by the carrier at load.
        let ep = hipfire_config::topology::qwen4_ep()?;
        let native_mtp = qwen4_native_mtp(options.spec, gpu_arch, path, pp, tp, ep)?;
        let spec_mtp = options.spec.mtp;
        let carrier =
            crate::carrier_for(arch_id).ok_or_else(|| "no carrier for qwen4".to_string())?;
        crate::carriers::qwen4_ep_option_refusal(ep, draft_path, &options)?;
        qwen4_pp_option_refusal(pp, draft_path, &options)?;
        qwen4_pp_mtp_ngram_refusal(pp, native_mtp, hipfire_config::mtp_ngram_enabled())?;
        carrier.admit_options(draft_path, options)?;
        if let Some(refusal) = qwen4_ep_geometry_refusal(
            ep,
            pp,
            tp,
            gpu_arch,
            visible_device_count(),
            hipfire_runtime::config::get().emulate_gpus.is_some(),
        ) {
            return Err(refusal);
        }
        if pp > 1 {
            let npu = hipfire_arch_qwen4::npu_trunk::NpuTrunkMode::from_env()
                .map_err(|error| format!("qwen4: {error}"))?
                .resolve(gpu_arch);
            qwen4_pp_env_refusal(
                pp,
                &|name| hipfire_config::developer_var(name).ok(),
                npu,
                hipfire_config::memory::session_cache_disk_bytes(),
            )?;
        }
        // The PLE rows source is chosen here, before the resident model is
        // torn down: a refusal (missing sidecar, binding mismatch) leaves the
        // loaded model in place; nothing is derived on this path.  The
        // attached sidecar travels with `source` to the carrier.
        resolve_qwen4_ple_rows(&mut source, path, gpu_arch)?;
        // The one mesh normalization: Qwen4's EffectiveMesh is derived from it.
        // A malformed request (zero degree / overflow) keeps its raw degrees so
        // Qwen4's own refusal text is unchanged.
        let shape = normalize_mesh_shape(pp, tp, ep);
        let effective_mesh = match &shape {
            Ok(mesh) => hipfire_arch_qwen4::EffectiveMesh::from_device_mesh(mesh),
            Err(_) => hipfire_arch_qwen4::EffectiveMesh::new(pp, tp, ep),
        };
        let admitted = admit_qwen4_source(
            &source,
            effective_mesh,
            InputModality::Text,
            native_mtp,
            spec_mtp,
            gpu_arch,
        )?;
        // Only an admitted (hence well-formed) mesh reaches here.
        mesh_shape = shape?;
        if pp > 1 {
            let policy = parse_mesh_transport(
                hipfire_config::developer_var(MESH_TRANSPORT_ENV)
                    .ok()
                    .as_deref(),
            )
            .map_err(|error| {
                format!("qwen4: pipeline parallelism (topology.pp = {pp}): {error}")
            })?;
            // No pre-side-effect byte-qualification evidence exists: `auto`
            // resolves to host-mapped edges, a forced `peer` refuses.
            qwen4_pp = Some(admit_qwen4_pp_mesh(
                pp,
                tp,
                ep,
                admitted.config.num_hidden_layers,
                &visible_device_ranks(gpu_arch, pp),
                gpu_arch,
                hipfire_runtime::config::get().emulate_gpus.is_some(),
                policy,
                &|_, _| false,
                pp_layers_request().as_deref(),
            )?);
        }
        // Native positions, or the YaRN-extended `factor x original` context.
        let positions = admitted.config.context_limit();
        if max_seq > positions {
            return Err(format!(
                "qwen4: max_seq {max_seq} exceeds the model's trained context of {positions} tokens \
                 (max_position_embeddings, or YaRN factor x original; see attention.yarn_factor). \
                 Lower memory.max_seq to at most {positions}"
            ));
        }
        let model_ctx = positions.min(hipfire_arch_qwen4::QWEN4_MAX_CONTEXT);
        qwen4_sequence = qwen4_context_sequence(
            unified_memory,
            max_seq,
            model_ctx,
            hipfire_runtime::kv_mode::qwen_k_display_name(qwen4_kv.mode),
        );
        let topology = if pp > 1 {
            EffectiveTopology::Pipeline(pp)
        } else {
            EffectiveTopology::Single
        };
        (topology, Some(resolve_carrier(&source)?))
    } else if tp > 1 {
        // Expert-parallel admission (HFQ-only). Mirrors
        // `load_model_ep_with_kv_mode`'s arch_id dispatch + per-arch VMM
        // refusal: DeepSeek V4 (9) serves vmm by design; Qwen3.5-MoE (5|6)
        // and MiniMax (10) refuse it (replicated-KV EP has no VMM path).
        if is_dir {
            return Err(
                "expert/tensor parallelism (tp>1) is not supported for safetensors directory sources; load a single HFQ file or set tp=1"
                    .into(),
            );
        }
        if !matches!(arch_id, 5 | 6 | 9 | 10) {
            return Err(crate::ep_unsupported_arch_message(arch_id));
        }
        // Qwen3.5 density probe (HFQ-only past the `is_dir` refusal above, so
        // the destructure is infallible): dense (`num_experts == 0`) runs the
        // dense-TP path with one VMM arena per rank device, so the MoE/EP VMM
        // refusal below does not apply to it. Parsed once here; the MoE
        // topology gate below reuses it instead of re-opening the source.
        let qwen35_num_experts: Option<usize> = if matches!(arch_id, 5 | 6) {
            let ModelSource::Hfq(hfq) = &source else {
                return Err("expert parallelism for Qwen3.5 needs an HFQ model file, not a safetensors directory; load a single HFQ file or set tp=1".to_string());
            };
            Some(
                hipfire_arch_qwen35::qwen35::config_from_hfq(hfq)
                    .map_err(|e| format!("qwen35 config: {e}"))?
                    .num_experts,
            )
        } else {
            None
        };
        qwen35_ep_experts = qwen35_num_experts;
        // #666 G2: refuse an unsatisfiable rank count before any teardown or
        // GPU init (`init_ep` needs one device per rank). Runs after the
        // arch/VMM refusals above so their messages are unchanged.
        if let Some(refusal) = tp_geometry_refusal(tp, visible_device_count()) {
            return Err(refusal);
        }
        // Qwen3.5 MoE EP topology gate: admit tp=2|4 where the sealed
        // per-rank loader admits (experts divide evenly), using the loader's
        // established predicate and exact error text. Runs after the
        // arch/VMM/geometry refusals above so their messages are unchanged.
        // Dense Qwen3.5 (num_experts == 0) is unaffected. Must stay paired
        // with the loader backstop text.
        if let Some(num_experts) = qwen35_num_experts {
            if let Some(refusal) = crate::qwen35_ep_moe_topology_refusal(arch_id, num_experts, tp) {
                return Err(refusal);
            }
        }
        let moe = matches!(arch_id, 9 | 10) || qwen35_num_experts.is_some_and(|n| n > 0);
        mesh_shape = if moe {
            normalize_mesh_shape(1, 1, tp)?
        } else {
            normalize_mesh_shape(1, tp, 1)?
        };
        (EffectiveTopology::Expert(tp), None)
    } else {
        // Single / pipeline-parallel via the carrier registry.
        let carrier = resolve_carrier(&source)?;
        // Selection follows topology validation; unsupported automatic requests
        // fall back rather than making a non-Qwen source unloadable.
        // #666 G2: refuse an impossible pipeline degree before any teardown
        // (`init_uniform` would fail `n_layers < n_devices` deep in GPU init
        // with the resident model already gone). Runs after the carrier's
        // own topology refusal so currently-refused archs keep their message;
        // only sources that would otherwise proceed to GPU init reach this.
        if let Some(refusal) =
            pp_geometry_refusal(pp, source_n_layers(&source), visible_device_count())
        {
            return Err(refusal);
        }
        let topology = if pp > 1 {
            EffectiveTopology::Pipeline(pp)
        } else {
            EffectiveTopology::Single
        };
        mesh_shape = normalize_mesh_shape(pp, 1, 1)?;
        (topology, Some(carrier))
    };
    // Partial GPU offload: fail closed where the arch is dispatched, before any
    // teardown, so a source that cannot honour a configured spill never receives
    // a resident load.
    partial_offload_refusal(arch_id, &source, tp, pp, qwen35_ep_experts)?;
    let heterogeneous_reason = hints.deepseek4_heterogeneous && arch_id == 9;
    // Qwen4 owns its own VMM QSA layout and certification (gfx1151/gfx1201), so
    // its capability replaces the Qwen/KvCache predicates below; its
    // topology is Single by the refusals above.
    let unsupported = if arch_id == QWEN4_ARCH_ID {
        hipfire_arch_qwen4::qwen4_vmm_refusal(gpu_arch, hints.vmm_runtime_available)
    } else if heterogeneous_reason {
        Some(
            "this mixed-GPU DeepSeek setup keeps its compressor cache in a layout the VMM backend cannot manage"
                .to_string(),
        )
    } else if !hints.vmm_runtime_available {
        Some(
            "this GPU runtime does not provide the virtual-memory (VMM) support the backend needs"
                .to_string(),
        )
    } else if let Some(reason) = hipfire_config::devices::vmm_kv_platform_refusal() {
        Some(reason)
    } else if gpu_arch != "gfx1201"
        && !qwen35_gfx11_vmm_supported(
            gpu_arch,
            arch_id,
            tp,
            hints
                .kv_mode
                .unwrap_or(hipfire_runtime::config::get().kv_mode.as_str()),
            hints.kv_k,
            hints.kv_v,
        )
    {
        Some(format!(
            "{gpu_arch} has no tested VMM KV path for the selected --kv-mode"
        ))
    } else if pp > 1 {
        Some("pipeline parallelism (topology.pp > 1) does not support VMM KV".to_string())
    } else if tp > 1 && (arch_id == 10 || qwen35_ep_experts.is_some_and(|n| n > 0)) {
        Some("multi-GPU MoE/expert-parallel loads keep a full KV copy on every GPU, which VMM KV does not support".to_string())
    } else if tp > 1 && arch_id == 9 {
        Some("DeepSeek expert parallelism with VMM KV needs VMM support on every GPU, and at least one GPU lacks it".to_string())
    } else if carrier.is_some_and(|c| !matches!(c.name(), "qwen35" | "deepseek4" | "muse_glimmer"))
    {
        Some(format!(
            "carrier {} has no VMM KV owner",
            carrier.unwrap().name()
        ))
    } else if arch_id == 14 && hints.cask.is_some_and(|c| c.sidecar.is_some()) {
        Some("Muse Glimmer does not support a CASK sidecar together with VMM KV".to_string())
    } else {
        None
    };
    let (kv_backend, kv_backend_reason) = match (request, unsupported) {
        (KvBackendRequest::Explicit(KvBackend::Legacy), _) => (KvBackend::Legacy, None),
        (KvBackendRequest::Explicit(KvBackend::Vmm), Some(reason)) => {
            return Err(format!("KV backend 'vmm' is not available for this load: {reason}. Use --kv-backend legacy (memory.kv_backend = legacy), or a model and GPU that support VMM KV"));
        }
        (KvBackendRequest::Explicit(KvBackend::Vmm), None)
        | (KvBackendRequest::Automatic, None) => (KvBackend::Vmm, None),
        (KvBackendRequest::Automatic, Some(reason)) => (KvBackend::Legacy, Some(reason)),
    };
    if kv_backend == KvBackend::Legacy
        && matches!(arch_id, 5 | 6)
        && hints
            .cask
            .is_some_and(|c| c.sidecar.is_some() && c.handoff_tokens > 0)
        && hints.kv_adaptive.is_some_and(|s| !matches!(s, "" | "off"))
    {
        return Err("kv_adaptive -> CASK handoff needs the VMM KV backend, which is not available for this load; disable the CASK sidecar/handoff or use a GPU that supports VMM KV".into());
    }
    if let Some(carrier) = carrier {
        carrier.admit_topology(arch_id, is_dir, pp, kv_backend)?;
    }
    // Qwen MoE on a discrete card keeps the prior 32768 CLI/config default:
    // automatic context there is not validated against expert placement. On
    // unified memory its prefill scratch is charged like dense
    // (`PrefillBatchScratch::projected_allocation_bytes`), so it sizes itself.
    let pinned_moe = if !unified_memory
        && matches!(arch_id, 5 | 6)
        && topology == EffectiveTopology::Single
        && kv_backend == KvBackend::Vmm
        && (hints.free_vram_bytes.is_some() || max_seq == 0)
    {
        if let ModelSource::Hfq(hfq) = &source {
            hipfire_arch_qwen35::qwen35::config_from_hfq(hfq)
                .map_err(|e| format!("qwen35 config: {e}"))?
                .num_experts
                != 0
        } else {
            false
        }
    } else {
        false
    };
    // A safetensors directory (ParoQuant) has no HFQ manifest to project from,
    // and its carrier allocates KV at the admitted `max_seq` without a
    // post-weight remeasure, so it keeps a fixed context (HIPF-34).
    let dir_fixed_context = is_dir && matches!(arch_id, 5 | 6) && kv_backend == KvBackend::Vmm;
    let sequence = if matches!(arch_id, 5 | 6)
        && !is_dir
        && topology == EffectiveTopology::Single
        && kv_backend == KvBackend::Vmm
        && !pinned_moe
        && (hints.free_vram_bytes.is_some() || max_seq == 0)
    {
        Some(resolve_sequence(&source, max_seq, gpu_arch, hints)?)
    } else if let Some(sequence) = qwen4_sequence {
        // Automatic Qwen4 context grows VMM pages on demand; legacy storage
        // would allocate all of it at load.
        if sequence.bound == "pending" && kv_backend != KvBackend::Vmm {
            return Err(format!(
                "qwen4: automatic max_seq needs the VMM KV backend, which is not in use ({}); set memory.max_seq to a fixed value to use legacy KV storage",
                kv_backend_reason.as_deref().unwrap_or("legacy requested")
            ));
        }
        Some(sequence)
    } else {
        None
    };
    let (effective_seq, legacy_measured): (usize, bool) = match &sequence {
        Some(resolved) => (resolved.max_seq, false),
        None if max_seq != 0 => (max_seq, false),
        None if pinned_moe => (32768, false),
        None if matches!(arch_id, 5 | 6)
            && !is_dir
            && matches!(topology, EffectiveTopology::Single)
            && kv_backend == KvBackend::Legacy =>
        {
            // Measured legacy fit (backend-gated, not OS-gated): explicit
            // legacy on any OS, or automatic with a VMM refusal.
            match qwen35_legacy_fit(&source, gpu_arch, hints)? {
                Some(fit) => (fit.max_seq, true),
                None => (4096, false),
            }
        }
        None => (4096, false),
    };
    let sequence_reason = match (max_seq, pinned_moe, dir_fixed_context) {
        (0, true, _) => {
            Some("max_seq auto-capacity unsupported for MoE on a discrete GPU; using prior default")
        }
        (0, _, true) => Some(
            "max_seq auto-capacity unsupported for safetensors directory sources; using default",
        ),
        _ => None,
    };
    let mut has_vision = probe_vision(&source, arch_id)?;
    // Shared tower sidecar (registry `vision` slot / `params.vision` /
    // `HIPFIRE_VISION_SIDECAR`), validated fail-closed; a tower-bearing
    // sidecar promotes a tower-less trunk to VL.
    let vision_path = resolve_vision_sidecar(vision, arch_id, is_dir)?;
    if vision_path.is_some() {
        has_vision = true;
    }
    if let ModelSource::Hfq(hfq) = &source {
        df_lash_lm_head_admission(hfq, draft_path, gpu_arch)?;
        if draft_path.is_some() {
            let mq3_supported = crate::is_dflash_lm_head_wmma_arch(gpu_arch) && arch_id == 5;
            let unsupported = hfq
                .first_tensor_with_quant_type(18)
                .map(|name| ("MQ2 (qt=18)", name))
                .or_else(|| {
                    (!mq3_supported)
                        .then(|| hfq.first_tensor_with_quant_type(17))
                        .flatten()
                        .map(|name| ("MQ3 (qt=17)", name))
                });
            if let Some((quant, tensor)) = unsupported {
                return Err(format!(
                    "DFlash draft requested but model contains {quant} weight `{tensor}` \
                     without a supported batched verifier on {gpu_arch} (arch_id={arch_id}); \
                     reload without a draft or use an MQ4/HFQ4/Q8 target"
                ));
            }
        }
    }
    // Head overlay (`params.head`): validated AND attached to the retained
    // base here, so the admitted source is already effective and loading
    // consumes it with no second open. Any refusal leaves the prior model
    // loaded — the overlay is in-memory only; no GPU state is touched.
    admit_head_overlay(head, &mut source, arch_id, topology)?;
    // Split K/V is a Qwen-family contract, resolved before the daemon can
    // unload the resident model. Non-Qwen owners retain their existing modes.
    let k_axis = hints.kv_k.filter(|s| !s.is_empty());
    let v_axis = hints.kv_v.filter(|s| !s.is_empty());
    if !matches!(arch_id, 1 | 5 | 6) && (k_axis.is_some() || v_axis.is_some()) {
        return Err(format!(
            "--kv-k/--kv-v are Qwen-only (arch_id={arch_id}); use --kv-mode for this model"
        ));
    }
    let mut qwen_n_layers: Option<usize> = None;
    if matches!(arch_id, 1 | 5 | 6) {
        use hipfire_runtime::kv_mode::{self, KvMode};
        let policy = if arch_id == 1 {
            &kv_mode::DIR_SAFETENSORS_POLICY
        } else if tp > 1 {
            &kv_mode::QWEN35_TP_POLICY
        } else if pp > 1 {
            &kv_mode::QWEN35_PP_POLICY
        } else if is_dir {
            &kv_mode::QWEN35_PARO_POLICY
        } else {
            &kv_mode::QWEN35_HFQ_POLICY
        };
        let mode = hints
            .kv_mode
            .unwrap_or(hipfire_runtime::config::get().kv_mode.as_str())
            .to_ascii_lowercase();
        let qwen_cfg = if matches!(arch_id, 5 | 6) {
            Some(
                match &source {
                    ModelSource::Hfq(h) => hipfire_arch_qwen35::qwen35::config_from_hfq(h),
                    ModelSource::Dir(d) => hipfire_arch_qwen35::qwen35::config_from_safetensors(d),
                }
                .map_err(|e| format!("Qwen KV admission config: {e}"))?,
            )
        } else {
            None
        };
        qwen_n_layers = qwen_cfg.as_ref().map(|cfg| cfg.n_layers);
        let adaptive = hints.kv_adaptive.is_some_and(|v| !matches!(v, "" | "off"));
        let auto_native = qwen_cfg.as_ref().is_some_and(|cfg| {
            kv_mode::qwen35_native_auto_eligible(
                gpu_arch,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                pp,
                adaptive,
                hints.cask.is_some_and(|c| c.sidecar.is_some()),
            )
        });
        let policy = kv_mode::qwen35_policy_for_native(policy, &mode, auto_native);
        if adaptive && (k_axis.is_some() || v_axis.is_some()) {
            return Err("adaptive Qwen KV cannot combine with fixed --kv-k/--kv-v; disable kv_adaptive or omit both axes".into());
        }
        let pair = kv_mode::resolve_kv_pair(
            &mode,
            k_axis,
            v_axis,
            &policy,
            gpu_arch,
            hints.qwen_default_q8,
        )
        .map_err(|e| format!("Qwen KV admission: {e}"))?;
        if let Some(cfg) = qwen_cfg.as_ref() {
            use hipfire_dispatch::families::attention::{
                gfx1100_split_verify_min_ctx, Gfx1100SplitVerifyModel,
            };
            let flags = rdna_compute::FeatureFlags::from_active_config(gpu_arch);
            let min_ctx = hipfire_config::developer_var("HIPFIRE_FA_PERTOKEN_MIN_CTX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4_096);
            let split_envelope = gfx1100_split_verify_min_ctx(&Gfx1100SplitVerifyModel {
                arch: gpu_arch,
                verify_attn: flags.verify_attn,
                split_verify: flags.gfx1100_fa2_split_verify,
                num_experts: cfg.num_experts,
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
                resolved_min_ctx: (min_ctx > 0).then_some(min_ctx),
            })
            .is_some();
            // All source/topology owners pass this boundary, including PP,
            // dense TP and directory sources that bypass the HFQ carrier.
            hipfire_arch_qwen35::carrier::validate_gfx1100_split_verify_kv(
                split_envelope,
                !adaptive
                    && matches!(
                        pair,
                        kv_mode::KvPair::Native(KvMode::Q8)
                            | kv_mode::KvPair::Split(KvMode::Q8, kv_mode::VMode::Q8)
                    ),
                kv_backend,
            )?;
        }
        let k = pair.k();
        if (pp > 1 || tp > 1) && pair.v().is_some_and(|v| v != kv_mode::VMode::Q8) {
            return Err(format!(
                "Qwen {} has no multi-GPU Lloyd-V constructor; use --kv-v q8",
                policy.site
            ));
        }
        if tp > 1 && matches!(k, KvMode::Fp8 | KvMode::Bf16) {
            return Err(
                "Qwen native fp8/bf16 is single-GPU only; use --kv-mode q8 for TP/EP".into(),
            );
        }
        if arch_id == 1 && matches!(k, KvMode::Asym3 | KvMode::Asym4) {
            use hipfire_runtime::arch::Architecture;
            let head_dim = match &source {
                ModelSource::Hfq(h) => {
                    <hipfire_arch_llama::Llama as Architecture>::config_from_hfq(h)
                        .map_err(|e| format!("Qwen KV admission config: {e}"))?
                        .head_dim
                }
                ModelSource::Dir(d) => {
                    hipfire_runtime::hfq::config_from_safetensors_llama(d)
                        .map_err(|e| format!("Qwen KV admission config: {e}"))?
                        .head_dim
                }
            };
            if head_dim != 256 {
                return Err(format!(
                    "Qwen legacy-asym K requires head_dim=256 (got {head_dim}); use --kv-mode q8"
                ));
            }
        }
        if let Some(cfg) = qwen_cfg.as_ref() {
            if matches!(
                k,
                KvMode::Fp8 | KvMode::Bf16 | KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4
            ) {
                if matches!(k, KvMode::Fp8 | KvMode::Bf16) {
                    if let Some(why) = kv_mode::qwen35_native_refusal(
                        k,
                        gpu_arch,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        pp,
                        adaptive,
                        hints.cask.is_some_and(|c| c.sidecar.is_some()),
                    ) {
                        return Err(why);
                    }
                }
                if matches!(k, KvMode::Asym2 | KvMode::Asym3 | KvMode::Asym4) && cfg.head_dim != 256
                {
                    return Err(format!(
                        "Qwen legacy-asym K requires head_dim=256 (got {}); use --kv-mode q8",
                        cfg.head_dim
                    ));
                }
            }
        }
        if !adaptive
            && hints.cask.is_some_and(|c| c.sidecar.is_some())
            && (matches!(k, KvMode::Fwht2 | KvMode::Fwht3 | KvMode::Fwht4)
                || pair.v().is_some_and(|v| v != kv_mode::VMode::Q8))
        {
            return Err("static CASK requires q8 or legacy-asym K with q8 V; use --kv-k legacy-asym3 --kv-v q8 or disable CASK".into());
        }
    }

    // One admitted mesh. Visible-device identity (HIP ordinal + PCI id + arch)
    // is bound here once; placement and carriers consume it and never reread
    // the raw pp/tp/ep request. Existing routes keep their rank order and
    // transports (`RankBinding::Visible`/`Emulated`, no edge decisions); the
    // Qwen4 PP route arrives with its mesh and bands already admitted.
    let (mesh, pp_bands, pp_layer_counts) = match qwen4_pp {
        Some((mesh, bands)) => (mesh, Some(bands.placement), bands.explicit_counts),
        None => {
            let shape = mesh_shape;
            let binding = if hipfire_runtime::config::get().emulate_gpus.is_some() {
                RankBinding::Emulated
            } else {
                RankBinding::Visible
            };
            let visible = visible_device_ranks(gpu_arch, shape.n_devices());
            let family = carrier.map_or_else(|| format!("arch{arch_id}"), |c| c.name().to_string());
            let mesh = admit_mesh(
                shape,
                &family,
                LEGACY_NUMERICAL_POLICY,
                &visible,
                binding,
                None,
            )?;
            // Qwen3.5 pipeline bands: `HIPFIRE_PP_LAYERS` is read here, once,
            // after every other admission verdict, and normalized through
            // `BandPlacement`.
            let (pp_bands, pp_layer_counts) = if matches!(topology, EffectiveTopology::Pipeline(_))
            {
                let n_layers = qwen_n_layers.ok_or_else(|| {
                    "qwen35 pipeline admission: layer count unavailable".to_string()
                })?;
                let bands =
                    normalize_pp_bands(&mesh.mesh, n_layers, pp_layers_request().as_deref())?;
                (Some(bands.placement), bands.explicit_counts)
            } else {
                (None, None)
            };
            (mesh, pp_bands, pp_layer_counts)
        }
    };
    Ok(SourceAdmission {
        source,
        arch_id,
        is_dir,
        has_vision,
        topology,
        kv_backend,
        kv_backend_reason,
        max_seq: effective_seq,
        sequence,
        sequence_reason,
        legacy_measured,
        kv_backend_request: request,
        qwen_default_q8: hints.qwen_default_q8,
        kv_v,
        carrier,
        vision_path,
        vision_mode: vision_mode.to_string(),
        mesh,
        pp_bands,
        pp_layer_counts,
        batch_request: BatchLoadRequest::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #771: gfx1030 + qwen3.8 MQ4V2 (qt=44) + DFlash reached
    /// `gemm_mq4g256v2_batched_lmhead` at the first `spec_step`. The gate
    /// refuses at admission with an actionable message; every other arch /
    /// quant / non-DFlash combination is untouched.
    mod dflash_lm_head_refusal_771 {
        use super::*;

        #[test]
        fn gfx1030_mq4v2_refuses_with_actionable_message() {
            let msg = dflash_lm_head_refusal(Some(44), "gfx1030")
                .expect("qt=44 + DFlash on gfx1030 must refuse at admission");
            for needle in [
                "gfx1030",
                "quant_type=44",
                "MQ4G256V2",
                "batched lm_head admission",
                "unsupported",
                "speculation.dflash = off",
                "autoregressive",
            ] {
                assert!(
                    msg.to_lowercase().contains(&needle.to_lowercase()),
                    "refusal must mention {needle:?}: {msg}"
                );
            }
            // A generic non-WMMA x-batch fallback exists; the refusal is an
            // admission/validation boundary, not a missing-kernel claim.
            for banned in ["no batched", "missing", "no v2 source", "no kernel"] {
                assert!(
                    !msg.to_lowercase().contains(banned),
                    "refusal must not claim a missing kernel ({banned:?}): {msg}"
                );
            }
        }

        #[test]
        fn other_non_wmma_arches_refuse_v2_and_mq3_lm_heads() {
            for arch in ["gfx1030", "gfx1010", "gfx942", "gfx906"] {
                for qt in [17u8, 44, 47, 48, 49, 50] {
                    let msg = dflash_lm_head_refusal(Some(qt), arch)
                        .unwrap_or_else(|| panic!("qt={qt} on {arch} must refuse"));
                    assert!(msg.contains(arch), "{msg}");
                    assert!(msg.contains(&format!("quant_type={qt}")), "{msg}");
                }
            }
        }

        #[test]
        fn wmma_arches_keep_mq4v2_dflash() {
            for arch in [
                "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
            ] {
                for qt in [44u8, 47, 48, 49, 50, 17] {
                    assert_eq!(
                        dflash_lm_head_refusal(Some(qt), arch),
                        None,
                        "{arch} qt={qt}"
                    );
                }
            }
        }

        #[test]
        fn gfx1030_keeps_dflash_for_supported_lm_head_quants() {
            // Q8_0 (3), HFQ4G256 (6), MQ4G256 (13) lm_heads never reach a V2
            // batched lm_head kernel; they stay admitted on gfx1030.
            for qt in [3u8, 6, 13] {
                assert_eq!(dflash_lm_head_refusal(Some(qt), "gfx1030"), None, "qt={qt}");
            }
        }

        #[test]
        fn refusal_message_for_unknown_and_missing_lm_head() {
            let missing = dflash_lm_head_refusal(None, "gfx1030").expect("missing refuses");
            assert!(
                missing.contains("no lm_head/embed_tokens tensor"),
                "{missing}"
            );
            let other = dflash_lm_head_refusal(Some(45), "gfx1100").expect("qt45 refuses");
            assert!(
                other.contains("quant_type=45") && other.contains("gfx1100"),
                "{other}"
            );
        }
    }

    mod mesh_normalization {
        use super::*;
        use hipfire_runtime::mesh_placement::EdgeTransport;

        fn rank(ordinal: i32, pci: Option<&str>) -> AdmittedRank {
            AdmittedRank {
                logical: ordinal as usize,
                device_ordinal: ordinal,
                pci_bus_id: pci.map(str::to_string),
                arch: "gfx1100".to_string(),
            }
        }

        fn six() -> Vec<AdmittedRank> {
            (0..6)
                .map(|i| rank(i, Some(&format!("0000:{:02x}:00.0", i + 3))))
                .collect()
        }

        fn admit(
            pp: usize,
            tp: usize,
            ep: usize,
            visible: &[AdmittedRank],
            binding: RankBinding,
        ) -> Result<AdmittedMesh, String> {
            admit_mesh(
                normalize_mesh_shape(pp, tp, ep)?,
                "test",
                LEGACY_NUMERICAL_POLICY,
                visible,
                binding,
                None,
            )
        }

        #[test]
        fn single_is_the_identity_mesh() {
            let mesh = admit(1, 1, 1, &six(), RankBinding::Visible).unwrap();
            assert!(mesh.mesh.axes().is_empty());
            assert_eq!(mesh.mesh.n_devices(), 1);
            assert_eq!(mesh.ranks, vec![rank(0, Some("0000:03:00.0"))]);
            assert!(mesh.edges.is_empty());
            assert_eq!(mesh.numerical_policy, LEGACY_NUMERICAL_POLICY);
        }

        #[test]
        fn zero_degree_and_overflow_are_refused_not_clamped() {
            for (pp, tp, ep) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
                let err = normalize_mesh_shape(pp, tp, ep).unwrap_err();
                assert!(err.contains("mesh degree is zero"), "{err}");
            }
            assert!(normalize_mesh_shape(usize::MAX, 2, 1).is_err());
        }

        #[test]
        fn rectangular_shapes_drop_size_one_axes() {
            let mesh = normalize_mesh_shape(2, 3, 1).unwrap();
            assert_eq!(mesh.n_devices(), 6);
            assert_eq!(mesh.size_of(DimKind::Pp), 2);
            assert_eq!(mesh.size_of(DimKind::Tp), 3);
            assert_eq!(mesh.axes().len(), 2);
            let ep = normalize_mesh_shape(1, 1, 4).unwrap();
            assert_eq!(ep.size_of(DimKind::Ep), 4);
            assert_eq!(ep.axes().len(), 1);
        }

        #[test]
        fn visible_binding_keeps_rank_order_and_identity() {
            let mesh = admit(2, 1, 1, &six(), RankBinding::Visible).unwrap();
            let ordinals: Vec<_> = mesh.ranks.iter().map(|r| r.device_ordinal).collect();
            assert_eq!(ordinals, [0, 1]);
            assert_eq!(mesh.ranks[1].pci_bus_id.as_deref(), Some("0000:04:00.0"));
            assert_eq!(mesh.ranks[1].logical, 1);
        }

        #[test]
        fn pp6_output_card_last_rotates_complete_identity_records() {
            let mesh = admit(6, 1, 1, &six(), RankBinding::OutputCardLast).unwrap();
            let ordinals: Vec<_> = mesh.ranks.iter().map(|r| r.device_ordinal).collect();
            assert_eq!(ordinals, [1, 2, 3, 4, 5, 0]);
            let logical: Vec<_> = mesh.ranks.iter().map(|r| r.logical).collect();
            assert_eq!(logical, [0, 1, 2, 3, 4, 5]);
            assert_eq!(mesh.ranks[5].pci_bus_id.as_deref(), Some("0000:03:00.0"));
            assert_eq!(mesh.ranks[0].pci_bus_id.as_deref(), Some("0000:04:00.0"));
            // Only the first `needed` visible devices participate.
            let pp2 = admit(2, 1, 1, &six(), RankBinding::OutputCardLast).unwrap();
            let ordinals: Vec<_> = pp2.ranks.iter().map(|r| r.device_ordinal).collect();
            assert_eq!(ordinals, [1, 0]);
        }

        #[test]
        fn output_card_last_is_pure_pipeline_only() {
            let err = admit(2, 2, 1, &six(), RankBinding::OutputCardLast).unwrap_err();
            assert!(err.contains("pure pipeline"), "{err}");
        }

        #[test]
        fn too_few_duplicate_or_negative_devices_refuse() {
            let two = six()[..2].to_vec();
            assert!(admit(3, 1, 1, &two, RankBinding::Visible)
                .unwrap_err()
                .contains("needs 3 visible devices, 2 visible"));
            let dup_ordinal = vec![rank(0, None), rank(0, None)];
            assert!(admit(2, 1, 1, &dup_ordinal, RankBinding::Visible)
                .unwrap_err()
                .contains("duplicate ordinal"));
            let dup_pci = vec![rank(0, Some("0000:03:00.0")), rank(1, Some("0000:03:00.0"))];
            assert!(admit(2, 1, 1, &dup_pci, RankBinding::Visible)
                .unwrap_err()
                .contains("duplicate PCI"));
            let negative = vec![rank(-1, None), rank(1, None)];
            assert!(admit(2, 1, 1, &negative, RankBinding::Visible).is_err());
            assert!(admit(1, 1, 1, &[], RankBinding::Visible).is_err());
        }

        #[test]
        fn emulation_aliases_ranks_onto_the_visible_devices() {
            let two = six()[..2].to_vec();
            let mesh = admit(1, 1, 4, &two, RankBinding::Emulated).unwrap();
            let ordinals: Vec<_> = mesh.ranks.iter().map(|r| r.device_ordinal).collect();
            assert_eq!(ordinals, [0, 1, 0, 1]);
            let logical: Vec<_> = mesh.ranks.iter().map(|r| r.logical).collect();
            assert_eq!(logical, [0, 1, 2, 3]);
            assert!(admit(1, 1, 2, &[], RankBinding::Emulated).is_err());
        }

        #[test]
        fn edges_follow_pp_adjacency_and_the_policy() {
            let shape = || normalize_mesh_shape(3, 1, 1).unwrap();
            let all_peer = |_: usize, _: usize| true;
            let only_first = |src: usize, dst: usize| (src, dst) == (0, 1);
            let go = |policy, qualified: &dyn Fn(usize, usize) -> bool| {
                admit_mesh(
                    shape(),
                    "test",
                    "p",
                    &six(),
                    RankBinding::Visible,
                    Some(EdgeAdmission {
                        policy,
                        qualified_peer: qualified,
                    }),
                )
            };
            let host = go(EdgeTransportPolicy::Host, &all_peer).unwrap();
            assert_eq!(
                host.edges
                    .iter()
                    .map(|e| (e.src, e.dst))
                    .collect::<Vec<_>>(),
                [(0, 1), (1, 2)]
            );
            assert!(host
                .edges
                .iter()
                .all(|e| e.transport == EdgeTransport::HostMapped));
            let auto = go(EdgeTransportPolicy::Auto, &only_first).unwrap();
            assert_eq!(auto.edges[0].transport, EdgeTransport::Peer);
            assert_eq!(auto.edges[1].transport, EdgeTransport::HostMapped);
            assert!(go(EdgeTransportPolicy::Peer, &only_first).is_err());
            assert!(go(EdgeTransportPolicy::Peer, &all_peer).is_ok());
        }

        #[test]
        fn composed_pp_tp_edges_keep_tp_coordinates() {
            // [pp, tp] row-major: device = pp * 2 + tp.
            let mesh = normalize_mesh_shape(2, 2, 1).unwrap();
            assert_eq!(pp_edges(&mesh).unwrap(), [(0, 2), (1, 3)]);
            assert!(pp_edges(&normalize_mesh_shape(1, 1, 4).unwrap())
                .unwrap()
                .is_empty());
        }

        #[test]
        fn visible_ranks_carry_physical_pci_identity() {
            use hipfire_config::devices::{GpuDevice, PciBdf};
            let devices = [GpuDevice {
                index: 4,
                rocr_index: 2,
                node: 7,
                arch: "gfx1201".to_string(),
                unique_id: None,
                bdf: PciBdf {
                    domain: 0,
                    bus: 0x44,
                    device: 0,
                    function: 0,
                },
            }];
            let ranks = visible_ranks(&devices);
            assert_eq!(ranks.len(), 1);
            assert_eq!(ranks[0].device_ordinal, 0);
            assert_eq!(ranks[0].pci_bus_id.as_deref(), Some("0000:44:00.0"));
            assert_eq!(ranks[0].arch, "gfx1201");
        }

        fn pp_mesh(pp: usize) -> DeviceMesh {
            normalize_mesh_shape(pp, 1, 1).unwrap()
        }

        fn counts(bands: &PpBands) -> Vec<usize> {
            (0..bands.placement.stages())
                .map(|s| bands.placement.band(s).len())
                .collect()
        }

        #[test]
        fn unset_pp_layers_is_the_historical_uniform_split() {
            let bands = normalize_pp_bands(&pp_mesh(3), 10, None).unwrap();
            assert_eq!(bands.explicit_counts, None);
            assert_eq!(counts(&bands), [4, 3, 3]);
            assert_eq!(
                normalize_pp_bands(&pp_mesh(3), 2, None).unwrap_err(),
                "load refused: pp=3 exceeds n_layers (2) — each GPU needs at least one model layer, so lower topology.pp to at most 2"
            );
        }

        #[test]
        fn explicit_pp_layers_normalize_through_the_band_placement() {
            let mesh = pp_mesh(6);
            let bands = normalize_pp_bands(&mesh, 48, Some("9,8,8,8,8,7")).unwrap();
            assert_eq!(bands.explicit_counts, Some(vec![9, 8, 8, 8, 8, 7]));
            assert_eq!(bands.placement.band(0), 0..9);
            assert_eq!(bands.placement.band(5), 41..48);
            assert_eq!(bands.placement.stage_of_layer(9), 1);
            bands.placement.validate_for(&mesh, 48).unwrap();
            // Stale placement: another mesh (new epoch) is refused.
            assert!(bands.placement.validate_for(&pp_mesh(6), 48).is_err());
            let spaced = normalize_pp_bands(&pp_mesh(2), 32, Some("16, 16")).unwrap();
            assert_eq!(counts(&spaced), [16, 16]);
            // The first-band PLE minimum is enforced on the placement value.
            assert!(bands.placement.clone().with_first_band_min(9).is_ok());
            assert!(bands.placement.with_first_band_min(10).is_err());
        }

        #[test]
        fn pp_layers_refusals_keep_their_exact_text() {
            let mesh = pp_mesh(2);
            let err = |spec| normalize_pp_bands(&mesh, 32, Some(spec)).unwrap_err();
            assert!(err("a,b").starts_with("HIPFIRE_PP_LAYERS parse: "));
            assert_eq!(err("32"), "HIPFIRE_PP_LAYERS has 1 entries, expected pp=2");
            assert_eq!(err("16,15"), "HIPFIRE_PP_LAYERS sum=31 != n_layers=32");
            assert_eq!(
                err("0,32"),
                "load refused: every pipeline stage needs at least one layer, but a stage was given zero"
            );
            assert_eq!(
                err("18446744073709551615,2"),
                "HIPFIRE_PP_LAYERS sum=18446744073709551615 != n_layers=32"
            );
        }
    }

    /// The load path never derives a missing q4 sidecar: a `q4` request with
    /// the q4 rows absent from the selected path and its canonical target is
    /// an error naming both locations and both remedies, and nothing is
    /// written next to either.
    #[cfg(unix)]
    #[test]
    fn qwen4_ple_load_path_refuses_instead_of_deriving_a_missing_q4() {
        use hipfire_config::memory::PleFormatRequest;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};

        let root = std::env::temp_dir().join(format!(
            "hipfire-loader-ple-no-derive-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let store = root.join("store");
        let models = root.join("models");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let target = store.join("flash-next.mq4-xts");
        let marker = format!(
            "{{\"{}\":{{\"version\":1}}}}",
            hipfire_runtime::ple_sidecar::PLE_BASE_MARKER_KEY
        );
        let tensor = HfqMemTensor {
            name: "model.language_model.layers.1.ple.ple_embedding.layer_multipliers".into(),
            quant_type: 54,
            shape: vec![3],
            group_size: 0,
            data: (0u8..24).collect(),
        };
        write_hfqm_package_mem(&target, 16, &marker, &[tensor]).unwrap();
        let link = models.join("flash-next.mq4-xts");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let mut hfq = HfqFile::open(&link).unwrap();
        let link_str = link.to_str().unwrap();
        for request in [PleFormatRequest::Q4, PleFormatRequest::Q8] {
            let error = resolve_qwen4_ple_rows_requested(&mut hfq, link_str, request, "gfx1151")
                .unwrap_err();
            let selected = format!("{}.ple-", link.display());
            let canonical = format!("{}.ple-", std::fs::canonicalize(&link).unwrap().display());
            assert!(
                error.contains(&selected) && error.contains(&canonical),
                "{error}"
            );
            assert!(error.contains("hipfire pull"), "{error}");
        }
        let error =
            resolve_qwen4_ple_rows_requested(&mut hfq, link_str, PleFormatRequest::Q4, "gfx1151")
                .unwrap_err();
        assert!(error.contains("hipfire ple derive --format q4"), "{error}");
        assert!(hfq.ple_rows().is_none());
        for dir in [&store, &models] {
            let written: Vec<_> = std::fs::read_dir(dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            assert_eq!(written.len(), 1, "{dir:?}: {written:?}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn gfx1100_qwen35_vmm_admits_fwht_split_matrix_only_on_tp1() {
        for mode in [
            "fwht2", "fwht3", "fwht4", "asym2", "asym3", "asym4", "turbo", "turbo2", "turbo3",
            "turbo4",
        ] {
            assert!(qwen35_gfx11_vmm_supported(
                "gfx1100", 5, 1, mode, None, None
            ));
            assert!(qwen35_gfx11_vmm_supported(
                "gfx1100", 6, 1, mode, None, None
            ));
            assert!(!qwen35_gfx11_vmm_supported(
                "gfx1100", 5, 2, mode, None, None
            ));
        }
        for k in ["fwht2", "fwht3", "fwht4"] {
            for v in ["lloyd2", "lloyd3", "lloyd4"] {
                assert!(qwen35_gfx11_vmm_supported(
                    "gfx1100",
                    5,
                    1,
                    k,
                    None,
                    Some(v)
                ));
            }
        }
        for k in ["legacy-asym2", "legacy-asym3", "legacy-asym4"] {
            for v in ["q8", "lloyd3"] {
                assert!(!qwen35_gfx11_vmm_supported(
                    "gfx1100",
                    5,
                    1,
                    k,
                    None,
                    Some(v)
                ));
            }
        }
        assert!(qwen35_gfx11_vmm_supported(
            "gfx1100",
            5,
            1,
            "q8",
            Some("fwht3"),
            Some("q8")
        ));
    }

    #[test]
    fn gfx11_qwen35_vmm_preserves_q8_matrix_and_other_refusals() {
        for arch in ["gfx1100", "gfx1151"] {
            for arch_id in [5, 6] {
                assert!(qwen35_gfx11_vmm_supported(
                    arch, arch_id, 2, "q8", None, None
                ));
            }
        }
        assert!(!qwen35_gfx11_vmm_supported(
            "gfx1100",
            5,
            1,
            "q8",
            None,
            Some("lloyd3")
        ));
        assert!(!qwen35_gfx11_vmm_supported(
            "gfx1151", 5, 1, "fwht3", None, None
        ));
        assert!(!qwen35_gfx11_vmm_supported(
            "gfx1100", 9, 1, "fwht3", None, None
        ));
        assert!(qwen35_gfx11_vmm_supported(
            "gfx1100", 5, 1, "fwht4", None, None
        ));
    }

    #[test]
    fn gfx1151_fwht_k_override_predicate_is_scoped() {
        let refuses = qwen35_gfx1151_fwht_k_override;
        for mode in ["", "auto", "q8", " q8 "] {
            for k in [
                "fwht2", "fwht3", "fwht4", "asym2", "asym3", "asym4", "turbo2", "turbo3", "turbo4",
                "turbo", " fwht3 ",
            ] {
                for arch_id in [5, 6] {
                    assert!(refuses("gfx1151", arch_id, mode, Some(k)));
                    for arch in ["gfx1100", "gfx1201", "gfx1030"] {
                        assert!(!refuses(arch, arch_id, mode, Some(k)));
                    }
                }
                assert!(!refuses("gfx1151", 9, mode, Some(k)));
            }
            for k in [
                None,
                Some(""),
                Some(" "),
                Some("q8"),
                Some("legacy-asym2"),
                Some("legacy-asym3"),
                Some("legacy-asym4"),
                Some("fwht5"),
            ] {
                for arch in ["gfx1100", "gfx1151", "gfx1201"] {
                    assert!(!refuses(arch, 5, mode, k));
                }
            }
        }
        for mode in ["fwht2", "fwht3", "fwht4", "legacy-asym3", "fp8", "bf16"] {
            assert!(!refuses("gfx1151", 5, mode, Some("fwht3")));
        }
    }
    fn admit_source(
        path: &str,
        tp: usize,
        pp: usize,
        raw: Option<&str>,
        draft: Option<&str>,
        arch: &str,
        vision: Option<&str>,
        vision_mode: &str,
        head: Option<&str>,
        max_seq: usize,
    ) -> Result<SourceAdmission, String> {
        super::admit_source(
            path,
            tp,
            pp,
            KvBackendRequest::from_override(raw)?,
            draft,
            arch,
            vision,
            vision_mode,
            head,
            max_seq,
            KvBackendHints::without_device(),
        )
    }

    /// Every Qwen3.5-family HF config embeds `vision_config` even for text-only
    /// quantized artifacts (the 27B/A3B production files all do). The tower
    /// tensor decides; config markers alone classify as the text backbone,
    /// never refuse. This is the contract from remediation `179a20d7f`.
    #[test]
    fn qwen35_config_marker_without_tower_is_text() {
        assert_eq!(classify_vision(5, false, true).unwrap(), false); // dense
        assert_eq!(classify_vision(6, false, true).unwrap(), false); // MoE
    }

    #[test]
    fn qwen35_tower_tensor_decides_vl() {
        assert_eq!(classify_vision(5, true, false).unwrap(), true);
        assert_eq!(classify_vision(6, true, false).unwrap(), true);
    }

    #[test]
    fn lfm2_tower_without_config_refuses() {
        assert!(classify_vision(11, true, false).is_err());
    }

    #[test]
    fn lfm2_config_without_tower_is_text() {
        assert_eq!(classify_vision(11, false, true).unwrap(), false);
    }

    #[test]
    fn non_vision_archs_are_never_vl() {
        for arch in [0u32, 1, 7, 9, 10, 22] {
            assert_eq!(classify_vision(arch, true, true).unwrap(), false);
        }
    }

    #[test]
    fn qwen4_conditional_generation_without_vision_is_text_capable() {
        assert_eq!(classify_vision(QWEN4_ARCH_ID, false, true).unwrap(), false);
        assert!(classify_vision(QWEN4_ARCH_ID, true, false).is_err());
    }

    #[test]
    fn qwen4_native_mtp_is_default_on_and_yields_to_retained_redline() {
        let with = |mtp| SpecLoadCfg {
            mtp,
            ..SpecLoadCfg::default()
        };
        let (hfq, mq4r) = ("m/qwen3.8-flash-next.mq4", "m/qwen3.8-flash-next.mq4r");
        assert_eq!(
            qwen4_native_mtp(with(None), "gfx1151", hfq, 1, 1, 1),
            Ok(true)
        );
        assert_eq!(
            qwen4_native_mtp(with(Some(false)), "gfx1151", hfq, 1, 1, 1),
            Ok(false)
        );
        assert_eq!(
            qwen4_native_mtp(with(Some(true)), "gfx1151", hfq, 1, 1, 1),
            Ok(true)
        );
        // A `.mq4r` load on a retained-default GPU is the retained AR route.
        assert_eq!(
            qwen4_native_mtp(with(None), "gfx1151", mq4r, 1, 1, 1),
            Ok(false)
        );
        assert!(qwen4_native_mtp(with(Some(true)), "gfx1151", mq4r, 1, 1, 1).is_err());
        // Off the retained-default GPUs the name does not claim the load.
        assert_eq!(
            qwen4_native_mtp(with(None), "gfx1030", mq4r, 1, 1, 1),
            Ok(true)
        );
        // Expert parallelism attaches the head only on an explicit `on`:
        // `auto`/off stay AR.
        for ep in [2, 4] {
            assert_eq!(
                qwen4_native_mtp(with(None), "gfx1201", hfq, 1, 1, ep),
                Ok(false)
            );
            assert_eq!(
                qwen4_native_mtp(with(Some(false)), "gfx1201", hfq, 1, 1, ep),
                Ok(false)
            );
            assert_eq!(
                qwen4_native_mtp(with(Some(true)), "gfx1201", hfq, 1, 1, ep),
                Ok(true)
            );
        }
    }

    #[test]
    fn qwen4_request_admission_matrix_covers_ep_pp_tp() {
        use hipfire_arch_qwen4::EffectiveMesh;
        let admit = |pp, tp, ep, mtp| {
            qwen4_request_admission(EffectiveMesh::new(pp, tp, ep), InputModality::Text, mtp)
        };
        for ep in 1..=8 {
            assert_eq!(admit(1, 1, ep, false), Ok(()), "ep {ep}");
        }
        assert_eq!(admit(1, 1, 1, true), Ok(()));
        for ep in [0, 9, 16] {
            assert!(
                admit(1, 1, ep, false)
                    .unwrap_err()
                    .contains("topology.ep from 1 to 8"),
                "ep {ep}"
            );
        }
        for pp in 2..=8 {
            assert_eq!(admit(pp, 1, 1, false), Ok(()), "pp {pp}");
            // Native MTP is admitted on the mesh (the output band owns the head).
            assert_eq!(admit(pp, 1, 1, true), Ok(()), "pp {pp}");
        }
        for (pp, tp, ep) in [(1, 2, 1), (2, 1, 2), (1, 2, 4), (2, 2, 1), (9, 1, 1)] {
            assert!(admit(pp, tp, ep, false).is_err(), "pp {pp} tp {tp} ep {ep}");
        }
        for ep in 2..=8 {
            assert_eq!(admit(1, 1, ep, true), Ok(()), "ep {ep}");
        }
    }

    #[test]
    fn qwen4_ep_geometry_refusal_names_the_missing_requirement() {
        assert_eq!(
            qwen4_ep_geometry_refusal(1, 1, 1, "gfx1030", Some(1), false),
            None
        );
        assert_eq!(
            qwen4_ep_geometry_refusal(4, 1, 1, "gfx1201", Some(4), false),
            None
        );
        assert_eq!(
            qwen4_ep_geometry_refusal(6, 1, 1, "gfx1100", Some(6), false),
            None
        );
        assert_eq!(
            qwen4_ep_geometry_refusal(8, 1, 1, "gfx1100", Some(8), false),
            None
        );
        assert_eq!(
            qwen4_ep_geometry_refusal(2, 1, 1, "gfx1201", None, false),
            None
        );
        let arch = qwen4_ep_geometry_refusal(4, 1, 1, "gfx1151", Some(4), false).unwrap();
        assert!(
            arch.contains("gfx1100 or gfx1201") && arch.contains("expert parallelism"),
            "{arch}"
        );
        let count = qwen4_ep_geometry_refusal(4, 1, 1, "gfx1201", Some(2), false).unwrap();
        assert!(
            count.contains("4 visible devices") && count.contains("2 visible"),
            "{count}"
        );
        assert!(qwen4_ep_geometry_refusal(2, 2, 1, "gfx1201", Some(2), false).is_some());
        assert!(qwen4_ep_geometry_refusal(2, 1, 2, "gfx1201", Some(2), false).is_some());
        // Emulation needs one physical device, not one per rank.
        assert_eq!(
            qwen4_ep_geometry_refusal(6, 1, 1, "gfx1100", Some(1), true),
            None
        );
        assert!(qwen4_ep_geometry_refusal(6, 1, 1, "gfx1100", Some(0), true).is_some());
        assert!(qwen4_ep_geometry_refusal(6, 1, 1, "gfx1030", Some(1), true).is_some());
        // gfx1151 runs logical ranks under emulation only.
        assert_eq!(
            qwen4_ep_geometry_refusal(4, 1, 1, "gfx1151", Some(1), true),
            None
        );
    }

    #[test]
    fn qwen4_ep_rank_devices_maps_physical_and_aliased_ranks() {
        // Physical mode: rank r is device r, and every rank needs its own device.
        assert_eq!(qwen4_ep_rank_devices(4, 4, false), Ok(vec![0, 1, 2, 3]));
        assert_eq!(qwen4_ep_rank_devices(2, 8, false), Ok(vec![0, 1]));
        let short = qwen4_ep_rank_devices(6, 4, false).unwrap_err();
        assert!(
            short.contains("6 visible devices") && short.contains("4 visible"),
            "{short}"
        );
        // Emulation: rank r -> physical device r % count.
        assert_eq!(qwen4_ep_rank_devices(2, 1, true), Ok(vec![0, 0]));
        assert_eq!(qwen4_ep_rank_devices(3, 1, true), Ok(vec![0, 0, 0]));
        assert_eq!(qwen4_ep_rank_devices(6, 1, true), Ok(vec![0; 6]));
        assert_eq!(
            qwen4_ep_rank_devices(6, 4, true),
            Ok(vec![0, 1, 2, 3, 0, 1])
        );
        assert_eq!(
            qwen4_ep_rank_devices(8, 4, true),
            Ok(vec![0, 1, 2, 3, 0, 1, 2, 3])
        );
        assert!(qwen4_ep_rank_devices(6, 0, true).is_err());
    }

    #[test]
    fn qwen4_ep_device_refusal_checks_arch_count_and_every_directed_pair() {
        let homogeneous = |arch: &str, n: usize| vec![arch.to_string(); n];
        let ranks = |n: i32| (0..n).collect::<Vec<i32>>();
        let all = |_: usize, _: usize| Ok(true);
        // Homogeneous gfx1201 and gfx1100 sets are admitted at every degree.
        for arch in ["gfx1201", "gfx1100"] {
            for ep in 2..=8usize {
                assert_eq!(
                    qwen4_ep_device_refusal(
                        ep,
                        &ranks(ep as i32),
                        &homogeneous(arch, ep),
                        false,
                        all
                    ),
                    Ok(()),
                    "{arch} ep {ep}"
                );
            }
        }
        assert_eq!(
            qwen4_ep_device_refusal(2, &ranks(2), &homogeneous("gfx1201", 4), false, all),
            Ok(())
        );
        // Too few devices.
        assert!(
            qwen4_ep_device_refusal(4, &ranks(4), &homogeneous("gfx1201", 3), false, all)
                .unwrap_err()
                .contains("3 visible")
        );
        // A different admitted arch on rank 2 refuses (mixed) and names the ranks.
        let mut mixed = homogeneous("gfx1201", 4);
        mixed[2] = "gfx1100".to_string();
        let error = qwen4_ep_device_refusal(4, &ranks(4), &mixed, false, all).unwrap_err();
        assert!(
            error.contains("same architecture")
                && error.contains("rank 2")
                && error.contains("gfx1100"),
            "{error}"
        );
        // An unadmitted arch refuses even when homogeneous.
        for arch in ["gfx1151", "gfx1030", "gfx942", ""] {
            let error = qwen4_ep_device_refusal(2, &ranks(2), &homogeneous(arch, 2), false, all)
                .unwrap_err();
            assert!(
                error.contains("gfx1100 or gfx1201") && error.contains("rank 0"),
                "{arch}: {error}"
            );
        }
        // An arch beyond the rank count is not part of the mesh.
        assert_eq!(
            qwen4_ep_device_refusal(2, &ranks(2), &mixed, false, all),
            Ok(())
        );
        // Every directed pair is probed; one missing direction refuses.
        let mut probed = Vec::new();
        qwen4_ep_device_refusal(4, &ranks(4), &homogeneous("gfx1201", 4), false, |a, b| {
            probed.push((a, b));
            Ok(true)
        })
        .unwrap();
        assert_eq!(probed.len(), 12);
        let error =
            qwen4_ep_device_refusal(4, &ranks(4), &homogeneous("gfx1201", 4), false, |a, b| {
                Ok(!(a == 3 && b == 1))
            })
            .unwrap_err();
        assert!(error.contains("device 3 cannot access device 1"), "{error}");
        // A failed query is an error, not "no peer".
        assert!(qwen4_ep_device_refusal(
            2,
            &ranks(2),
            &homogeneous("gfx1201", 2),
            false,
            |_, _| Err("hip".to_string())
        )
        .unwrap_err()
        .contains("hip"));
        // Aliased ranks (emulation): same-device pairs are never probed and a
        // single physical device needs no peer access at all.
        let one = homogeneous("gfx1100", 1);
        let mut probes = 0;
        assert_eq!(
            qwen4_ep_device_refusal(6, &[0; 6], &one, true, |_, _| {
                probes += 1;
                Ok(false)
            }),
            Ok(())
        );
        assert_eq!(probes, 0);
        // Four physical devices under six ranks: only distinct physical pairs
        // are probed, once each.
        let mut pairs = Vec::new();
        qwen4_ep_device_refusal(
            6,
            &[0, 1, 2, 3, 0, 1],
            &homogeneous("gfx1100", 4),
            true,
            |a, b| {
                pairs.push((a, b));
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(pairs.len(), 12);
        // Aliased devices are still arch-checked; gfx1151 is admitted only aliased.
        assert!(
            qwen4_ep_device_refusal(6, &[0; 6], &homogeneous("gfx1030", 1), true, all).is_err()
        );
        assert_eq!(
            qwen4_ep_device_refusal(6, &[0; 6], &homogeneous("gfx1151", 1), true, all),
            Ok(())
        );
    }

    #[test]
    fn effective_ep_is_one_for_other_archs() {
        assert_eq!(effective_ep(5), 1);
        assert_eq!(effective_ep(9), 1);
    }

    #[test]
    fn qwen4_host_mapped_experts_make_native_mtp_opt_in() {
        let with = |mtp| SpecLoadCfg {
            mtp,
            ..SpecLoadCfg::default()
        };
        // Device-resident experts: the admitted head stays on under `auto`.
        assert!(qwen4_mtp_with_host_mapped_experts(with(None), "gfx1100", 0));
        // Host-mapped experts: `auto` keeps AR off gfx1100/gfx1201, an explicit `on`
        // keeps MTP everywhere, and gfx1100/gfx1201 keep the head under `auto`.
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1100",
            21
        ));
        assert!(!qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1151",
            21
        ));
        assert!(!qwen4_mtp_with_host_mapped_experts(
            with(Some(false)),
            "gfx1100",
            21
        ));
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(Some(true)),
            "gfx1100",
            21
        ));
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1201",
            21
        ));
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(Some(true)),
            "gfx1201",
            21
        ));
    }

    #[test]
    fn qwen4_head_free_profile_turns_auto_mtp_off_and_refuses_an_explicit_request() {
        // A profile with the head keeps the policy decision.
        assert_eq!(qwen4_native_mtp_for_head(true, None, true, "p"), Ok(true));
        assert_eq!(
            qwen4_native_mtp_for_head(false, Some(false), true, "p"),
            Ok(false)
        );
        assert_eq!(
            qwen4_native_mtp_for_head(true, Some(true), true, "p"),
            Ok(true)
        );
        // No head: auto and `off` resolve to AR, `--spec mtp` is refused.
        assert_eq!(qwen4_native_mtp_for_head(true, None, false, "p"), Ok(false));
        assert_eq!(
            qwen4_native_mtp_for_head(false, Some(false), false, "p"),
            Ok(false)
        );
        let error = qwen4_native_mtp_for_head(true, Some(true), false, "tiny-dense").unwrap_err();
        assert!(
            error.contains("--spec mtp")
                && error.contains("tiny-dense")
                && error.contains("no MTP head"),
            "{error}"
        );
    }

    #[test]
    fn qwen4_developer_profiles_refuse_explicit_sidecar_and_hot_store_requests() {
        use hipfire_config::memory::{PleFormatRequest, PleHotStoreBudget};
        // Flash-Next is never touched by the developer policy.
        for format in [
            PleFormatRequest::Auto,
            PleFormatRequest::Q4,
            PleFormatRequest::Q8,
        ] {
            for hot in [
                PleHotStoreBudget::Auto,
                PleHotStoreBudget::Bytes(0),
                PleHotStoreBudget::Bytes(1 << 30),
            ] {
                assert_eq!(
                    qwen4_developer_ple_policy(true, "fn", format, hot),
                    Ok(None)
                );
            }
        }
        // A developer profile: defaults pass, `auto` hot store is skipped with
        // a note, explicit sidecar/hot requests are refused.
        let dev = |format, hot| qwen4_developer_ple_policy(false, "tiny-dense", format, hot);
        assert_eq!(
            dev(PleFormatRequest::Auto, PleHotStoreBudget::Bytes(0)),
            Ok(None)
        );
        let note = dev(PleFormatRequest::Auto, PleHotStoreBudget::Auto)
            .unwrap()
            .expect("note");
        assert!(
            note.contains("tiny-dense") && note.contains("off"),
            "{note}"
        );
        for format in [PleFormatRequest::Q4, PleFormatRequest::Q8] {
            let error = dev(format, PleHotStoreBudget::Bytes(0)).unwrap_err();
            assert!(
                error.contains("memory.ple_format") && error.contains("tiny-dense"),
                "{error}"
            );
        }
        let error = dev(PleFormatRequest::Auto, PleHotStoreBudget::Bytes(4096)).unwrap_err();
        assert!(
            error.contains("memory.ple_hot_store_bytes") && error.contains("tiny-dense"),
            "{error}"
        );
    }

    mod qwen4_ddtree_admission {
        use super::super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        fn write_probe(name: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-qwen4-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            write_hfqm_package_mem(
                &path,
                QWEN4_ARCH_ID,
                "{}",
                &[HfqMemTensor {
                    name: "model.embed_tokens.weight".into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                }],
            )
            .unwrap();
            path
        }

        fn cleanup(path: &std::path::Path) {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }

        fn admit(
            path: &std::path::Path,
            spec: SpecLoadCfg,
            kv_adaptive_override: Option<&str>,
        ) -> Result<SourceAdmission, String> {
            admit_source_with_options(
                path.to_str().unwrap(),
                1,
                1,
                KvBackendRequest::Automatic,
                None,
                "gfx1151",
                None,
                "auto",
                None,
                2048,
                KvBackendHints::without_device(),
                SourceAdmissionOptions {
                    spec,
                    kv_adaptive: qwen4_kv_adaptive_requested(kv_adaptive_override),
                    ..Default::default()
                },
            )
        }

        #[test]
        fn resolved_defaults_pass_source_gate_but_active_overrides_refuse() {
            let path = write_probe("ddtree");
            let defaults = SpecLoadCfg {
                dflash: Some(false),
                dspark: Some(false),
                ngram_draft: Some(false),
                ddtree_budget: Some(0),
                ddtree_topk: Some(4),
                mtp: Some(false),
                ..SpecLoadCfg::default()
            };
            let err = admit(&path, defaults, Some("off"))
                .map(|_| ())
                .expect_err("minimal probe must fail later on invalid Qwen4 config");
            assert!(
                !err.contains("qwen4: unsupported option(s)"),
                "resolved schema defaults must pass the DDTree source gate: {err}"
            );
            assert!(
                err.contains("config admission failed"),
                "defaults must reach Qwen4 config validation: {err}"
            );

            let active = SpecLoadCfg {
                ddtree_budget: Some(1),
                ddtree_topk: Some(4),
                ..defaults
            };
            let err = admit(&path, active, Some("off"))
                .map(|_| ())
                .expect_err("active DDTree must refuse");
            assert!(
                err.contains("qwen4: unsupported option(s): DDTree;"),
                "active DDTree must be refused at source admission: {err}"
            );

            let nondefault_topk = SpecLoadCfg {
                ddtree_budget: Some(0),
                ddtree_topk: Some(5),
                ..defaults
            };
            let err = admit(&path, nondefault_topk, Some("off"))
                .map(|_| ())
                .expect_err("non-default top-K must refuse");
            assert!(
                err.contains("qwen4: unsupported option(s): DDTree;"),
                "non-default DDTree top-K must be refused at source admission: {err}"
            );

            let err = admit(&path, defaults, Some("balanced"))
                .map(|_| ())
                .expect_err("active adaptive KV must refuse");
            assert!(
                err.contains("qwen4: unsupported option(s): adaptive KV;"),
                "active adaptive KV must be refused at source admission: {err}"
            );
            for (option, options) in [
                (
                    "PFlash",
                    SourceAdmissionOptions {
                        spec: defaults,
                        pflash: true,
                        ..Default::default()
                    },
                ),
                (
                    "EAGLE",
                    SourceAdmissionOptions {
                        spec: defaults,
                        eagle_drafter: true,
                        ..Default::default()
                    },
                ),
                (
                    "expert-count override",
                    SourceAdmissionOptions {
                        spec: defaults,
                        expert_count_override: true,
                        ..Default::default()
                    },
                ),
            ] {
                let err = admit_source_with_options(
                    path.to_str().unwrap(),
                    1,
                    1,
                    KvBackendRequest::Automatic,
                    None,
                    "gfx1151",
                    None,
                    "auto",
                    None,
                    2048,
                    KvBackendHints::without_device(),
                    options,
                )
                .map(|_| ())
                .expect_err(option);
                assert!(
                    err.contains("unsupported"),
                    "{option} must refuse before config: {err}"
                );
            }
            cleanup(&path);
        }
    }
    #[test]
    fn qwen4_request_refuses_image_video_and_non_single_before_source_access() {
        let single = hipfire_arch_qwen4::EffectiveMesh::single();
        assert!(qwen4_request_admission(single, InputModality::Text, true).is_ok());
        assert!(qwen4_request_admission(single, InputModality::Text, false).is_ok());
        assert!(qwen4_request_admission(single, InputModality::Image, true).is_err());
        assert!(qwen4_request_admission(single, InputModality::Video, true).is_err());
        assert!(qwen4_request_admission(
            hipfire_arch_qwen4::EffectiveMesh::new(1, 2, 1),
            InputModality::Text,
            true
        )
        .is_err());
    }

    /// The EP VMM refusal is per-arch, mirroring `load_model_ep_with_kv_mode`:
    /// Qwen3.5 (5|6) and MiniMax (10) refuse `vmm`; DeepSeek V4 (9) serves it.
    /// A blanket gate here would refuse the DS4 EP + vmm load master serves.
    #[test]
    fn backend_request_preserves_explicit_opt_out() {
        assert_eq!(
            KvBackendRequest::from_override(None).unwrap(),
            KvBackendRequest::Automatic
        );
        assert_eq!(
            KvBackendRequest::from_override(Some("legacy")).unwrap(),
            KvBackendRequest::Explicit(KvBackend::Legacy)
        );
        assert_eq!(
            KvBackendRequest::from_override(Some("vmm")).unwrap(),
            KvBackendRequest::Explicit(KvBackend::Vmm)
        );
        let migration = KvBackendRequest::from_override(Some("contiguous")).unwrap_err();
        assert!(migration.contains("legacy"), "{migration}");
        assert!(KvBackendRequest::from_override(Some("invalid")).is_err());
    }

    #[test]
    fn legacy_warning_discloses_provenance() {
        assert!(legacy_warning(KvBackend::Vmm, None).is_none());
        let automatic =
            legacy_warning(KvBackend::Legacy, Some("device gfx942 unsupported")).unwrap();
        assert!(automatic.contains("HIPFIRE_KV_BACKEND=legacy"));
        assert!(
            automatic.contains("automatic VMM selection unavailable (device gfx942 unsupported)")
        );
        let explicit = legacy_warning(KvBackend::Legacy, None).unwrap();
        assert!(explicit.contains("HIPFIRE_KV_BACKEND=legacy"));
        assert!(explicit.contains("explicitly selected"));
        assert!(!explicit.contains("unavailable"));
    }

    /// FLUX/Klein admit exactly the gfx11 wave32 WMMA set: gfx1201 (and any
    /// non-gfx11 arch) refuses with the RDNA3/3.5 remedy before any
    /// allocation; gfx1100/gfx1151 admit; non-diffusion archs are untouched.
    #[test]
    fn flux_arch_refusal_is_gfx11_only() {
        for arch_id in [40u32, 45] {
            let err = flux_arch_refusal(arch_id, "gfx1201")
                .expect("gfx1201 FLUX/Klein must refuse at admission");
            assert!(err.contains("RDNA3/3.5"), "reason names remedy: {err}");
            assert!(err.contains("gfx11 wave32 WMMA"), "reason: {err}");
            assert!(err.contains("gfx1201"), "reason names detected arch: {err}");
            assert!(err.contains("IMAGEGEN.md"), "reason: {err}");
            assert!(flux_arch_refusal(arch_id, "gfx1200").is_some());
            assert_eq!(flux_arch_refusal(arch_id, "gfx1100"), None);
            assert_eq!(flux_arch_refusal(arch_id, "gfx1151"), None);
        }
        // Non-diffusion archs never hit this gate, on any arch string.
        assert_eq!(flux_arch_refusal(5, "gfx1201"), None);
        assert_eq!(flux_arch_refusal(9, "gfx1201"), None);
    }

    #[test]
    fn muse_glimmer_refuses_every_parallel_axis() {
        for (pp, tp, ep) in [(2, 1, 1), (1, 2, 1), (1, 4, 1), (1, 1, 2), (1, 1, 4)] {
            let err = muse_glimmer_topology_refusal(14, pp, tp, ep)
                .unwrap_or_else(|| panic!("pp={pp} tp={tp} ep={ep} must refuse"));
            assert!(err.contains("single-device only"), "{err}");
        }
        assert_eq!(muse_glimmer_topology_refusal(14, 1, 1, 1), None);
        // Other architectures keep their own topology gates.
        assert_eq!(muse_glimmer_topology_refusal(5, 2, 2, 2), None);
    }

    /// #666 G2 geometry gates: the metadata probe reads the HF `{config}`
    /// envelope every pp-capable source carries, and fails open (`None`) on
    /// sources with no layer key rather than refusing a servable load.
    #[test]
    fn metadata_probe_reads_layer_keys() {
        let enveloped = r#"{"config":{"num_hidden_layers":64}}"#;
        assert_eq!(n_layers_from_metadata_json(enveloped), Some(64));
        // Composite checkpoints nest the trunk under `text_config` (the
        // production qwen3.8 envelope); the nested node wins.
        assert_eq!(
            n_layers_from_metadata_json(
                r#"{"config":{"text_config":{"num_hidden_layers":64},"vision_config":{}}}"#
            ),
            Some(64)
        );
        assert_eq!(
            n_layers_from_metadata_json(r#"{"config":{"n_layers":32}}"#),
            Some(32)
        );
        // Flat (unenveloped) metadata and zero layers fail open, not refused.
        assert_eq!(
            n_layers_from_metadata_json(r#"{"num_hidden_layers":12}"#),
            Some(12)
        );
        assert_eq!(n_layers_from_metadata_json("{}"), None);
        assert_eq!(
            n_layers_from_metadata_json(r#"{"config":{"num_hidden_layers":0}}"#),
            None
        );
        assert_eq!(n_layers_from_metadata_json("not json"), None);
    }

    /// `pp` geometry mirrors `Gpus::init_uniform`'s deep invariants
    /// (`n_devices >= 1`, `n_layers >= n_devices`): pp=99 on a 64-layer
    /// source refuses naming `init_uniform`; pp=1 (the default) and any
    /// servable degree pass; unknown geometry fails open.
    #[test]
    fn pp_geometry_mirrors_init_uniform() {
        let err_zero = pp_geometry_refusal(0, Some(64), Some(4)).expect("pp=0 refuses");
        assert!(err_zero.contains("pp=0"), "reason: {err_zero}");
        assert!(
            err_zero.contains("topology.pp"),
            "names the setting: {err_zero}"
        );
        let err = pp_geometry_refusal(99, Some(64), Some(4)).expect("pp=99 refuses");
        assert!(err.contains("pp=99"), "reason: {err}");
        assert!(err.contains("n_layers=64"), "reason: {err}");
        assert!(err.contains("topology.pp"), "names the setting: {err}");
        let err = pp_geometry_refusal(8, Some(64), Some(4)).expect("pp=8 > 4 devices");
        assert!(err.contains("device count=4"), "reason: {err}");
        // Servable degrees pass untouched.
        assert_eq!(pp_geometry_refusal(1, Some(64), Some(4)), None);
        assert_eq!(pp_geometry_refusal(4, Some(64), Some(4)), None);
        assert_eq!(pp_geometry_refusal(64, Some(64), Some(64)), None);
        // Unknown geometry fails open — the deep check stays last-resort.
        assert_eq!(pp_geometry_refusal(99, None, None), None);
        assert_eq!(pp_geometry_refusal(1, None, None), None);
    }

    /// `tp` geometry mirrors `init_tp`/`init_ep`: no layer bound (every rank
    /// runs every layer), only the zero degree and the device count.
    #[test]
    fn tp_geometry_checks_degree_and_devices_only() {
        let err_zero = tp_geometry_refusal(0, Some(4)).expect("tp=0 refuses");
        assert!(err_zero.contains("tp=0"), "reason: {err_zero}");
        assert!(
            err_zero.contains("single GPU"),
            "names the remedy: {err_zero}"
        );
        let err = tp_geometry_refusal(99, Some(4)).expect("tp=99 refuses");
        assert!(err.contains("tp=99"), "reason: {err}");
        assert!(err.contains("device count=4"), "reason: {err}");
        assert_eq!(tp_geometry_refusal(1, Some(4)), None);
        assert_eq!(tp_geometry_refusal(4, Some(4)), None);
        assert_eq!(tp_geometry_refusal(99, None), None);
    }

    /// Device-list counting follows the comma-separated convention shared by
    /// `hardware.devices` and the visibility envs.
    #[test]
    fn device_list_counting() {
        assert_eq!(count_device_list("0,1,2,3"), 4);
        assert_eq!(count_device_list("0"), 1);
        assert_eq!(count_device_list(" 0, 1 ,,"), 2);
        assert_eq!(count_device_list(""), 0);
    }

    /// Vision-sidecar fixtures: minimal HFQ files via the in-memory writer.
    /// The trunk is a tower-less text pack; the sidecar carries just the
    /// trunk family's probe tensor. Admission is read-only — no GPU needed.
    mod vision_sidecar {
        use super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        /// The probe tensor admission looks for on an `arch_id` sidecar.
        fn probe_for(arch_id: u32) -> &'static str {
            if arch_id == 18 {
                super::super::EMBED_GEMMA_VISION_PROBE_TENSOR
            } else {
                super::super::VISION_PROBE_TENSOR
            }
        }

        fn write_hfq(name: &str, arch_id: u32, with_tower: bool) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-vision-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            let mut tensors = vec![HfqMemTensor {
                name: "model.embed_tokens.weight".into(),
                quant_type: 1,
                shape: vec![4, 4],
                group_size: 0,
                data: vec![0u8; 32],
            }];
            if with_tower {
                tensors.push(HfqMemTensor {
                    name: probe_for(arch_id).into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                });
            }
            let metadata = if arch_id == 5 {
                r#"{"config":{"text_config":{"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":4,"vocab_size":1000,"max_position_embeddings":131072}}}"#
            } else {
                "{}"
            };
            write_hfqm_package_mem(&path, arch_id, metadata, &tensors).unwrap();
            path
        }

        fn cleanup(path: &std::path::Path) {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }

        /// Tower-less trunk + tower-bearing sidecar admits as VL on qwen35.
        #[test]
        fn sidecar_promotes_tower_less_trunk_to_vl() {
            let trunk = write_hfq("a-trunk", 5, false);
            let sidecar = write_hfq("a-sidecar", 5, true);
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .expect("tower sidecar must admit");
            assert!(admitted.has_vision, "sidecar tower promotes trunk to VL");
            assert_eq!(admitted.vision_path, Some(sidecar.clone()));
            assert!(
                admitted.carrier.is_some_and(|c| c.name() == "qwen35"),
                "trunk still routes to qwen35"
            );
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// A sidecar without the tower tensor refuses with the pack remedy.
        #[test]
        fn sidecar_without_tower_refuses() {
            let trunk = write_hfq("b-trunk", 5, false);
            let sidecar = write_hfq("b-sidecar", 5, false);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("tower-less sidecar must refuse");
            assert!(err.contains("no vision tower tensor"), "remedy: {err}");
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// A sidecar stamped with the wrong arch refuses.
        #[test]
        fn sidecar_with_wrong_arch_refuses() {
            let trunk = write_hfq("c-trunk", 5, false);
            let sidecar = write_hfq("c-sidecar", 9, true);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("wrong-arch sidecar must refuse");
            assert!(err.contains("arch_id=9"), "names the sidecar arch: {err}");
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// EmbeddingGemma-2: tower-less arch-18 trunk + its arch-18 `.vl` pack
        /// admits, and the resolved path reaches the carrier through
        /// `LoadCtx` exactly as the Qwen3.5-VL sidecar does.
        #[test]
        fn embed_gemma_sidecar_promotes_tower_less_trunk() {
            let trunk = write_hfq("eg2-trunk", 18, false);
            let sidecar = write_hfq("eg2-sidecar", 18, true);
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .expect("EmbeddingGemma-2 tower sidecar must admit");
            assert!(
                admitted.has_vision,
                "sidecar tower promotes the tower-less arch-18 trunk"
            );
            assert_eq!(admitted.vision_path, Some(sidecar.clone()));
            assert!(
                admitted
                    .carrier
                    .is_some_and(|c| c.name() == "embeddinggemma2"),
                "trunk still routes to embeddinggemma2"
            );
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// An arch-18 pack without the EmbeddingGemma-2 tower probe refuses
        /// with the requantize remedy — never silently text-only.
        #[test]
        fn embed_gemma_sidecar_without_tower_refuses() {
            let trunk = write_hfq("eg2-b-trunk", 18, false);
            let sidecar = write_hfq("eg2-b-sidecar", 18, false);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("tower-less arch-18 sidecar must refuse");
            assert!(err.contains("no vision tower tensor"), "remedy: {err}");
            assert!(
                err.contains(super::super::EMBED_GEMMA_VISION_PROBE_TENSOR),
                "names the EmbeddingGemma-2 probe: {err}"
            );
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// The two families do not interchange: an arch-18 trunk refuses a
        /// Qwen3.5-VL sidecar (and vice versa) — the wrong tower's names would
        /// never resolve against the trunk.
        #[test]
        fn sidecar_families_do_not_interchange() {
            let eg2_trunk = write_hfq("eg2-x-trunk", 18, false);
            let qwen_sidecar = write_hfq("eg2-x-qwen-sidecar", 5, true);
            let err = admit_source(
                eg2_trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(qwen_sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("Qwen sidecar on an arch-18 trunk must refuse");
            assert!(err.contains("arch_id=5"), "names the sidecar arch: {err}");
            assert!(err.contains("expected 18"), "names the family: {err}");
            cleanup(&eg2_trunk);
            cleanup(&qwen_sidecar);

            let qwen_trunk = write_hfq("eg2-y-trunk", 5, false);
            let eg2_sidecar = write_hfq("eg2-y-sidecar", 18, true);
            let err = admit_source(
                qwen_trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(eg2_sidecar.to_str().unwrap()),
                "auto",
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("EmbeddingGemma-2 sidecar on a Qwen trunk must refuse");
            assert!(err.contains("arch_id=18"), "names the sidecar arch: {err}");
            assert!(err.contains("expected 5|6"), "names the family: {err}");
            cleanup(&qwen_trunk);
            cleanup(&eg2_sidecar);
        }

        /// A sidecar requested for any arch outside the two VL families is
        /// refused before the file is even opened.
        #[test]
        fn sidecar_for_a_non_vl_arch_refuses() {
            let missing = std::env::temp_dir().join("hipfire-vision-admit-never-exists.hfq");
            let _ = std::fs::remove_file(&missing);
            let err =
                super::super::resolve_vision_sidecar(Some(missing.to_str().unwrap()), 13, false)
                    .map(|_| ())
                    .expect_err("text-only arch must refuse a sidecar request");
            assert!(
                err.contains("arch_id=13") && err.contains("EmbeddingGemma-2"),
                "names the arch and both VL families: {err}"
            );
        }
        #[test]
        fn automatic_backend_resolves_by_capability_not_source_spelling() {
            let trunk = write_hfq("kv-request", 5, false);
            let path = trunk.to_str().unwrap();
            let hints = KvBackendHints {
                kv_mode: Some("q8"),
                kv_k: None,
                kv_v: None,
                qwen_default_q8: true,
                kv_adaptive: None,
                cask: None,
                deepseek4_heterogeneous: false,
                vmm_runtime_available: true,
                free_vram_bytes: None,
            };
            let admit = |request, arch| {
                super::super::admit_source(
                    path, 1, 1, request, None, arch, None, "auto", None, 4096, hints,
                )
            };
            let auto = admit(KvBackendRequest::Automatic, "gfx1201").unwrap();
            assert_eq!(auto.kv_backend, KvBackend::Vmm);
            assert!(auto.kv_backend_reason.is_none());
            let explicit = admit(KvBackendRequest::Explicit(KvBackend::Legacy), "gfx1201").unwrap();
            assert_eq!(explicit.kv_backend, KvBackend::Legacy);
            assert!(explicit.kv_backend_reason.is_none());
            let unsupported = admit(KvBackendRequest::Automatic, "gfx940").unwrap();
            assert_eq!(unsupported.kv_backend, KvBackend::Legacy);
            assert!(unsupported.kv_backend_reason.unwrap().contains("gfx940"));
            let error = admit(KvBackendRequest::Explicit(KvBackend::Vmm), "gfx940")
                .map(|_| ())
                .unwrap_err();
            assert!(error.contains("gfx940") && error.contains("legacy"));
            cleanup(&trunk);
        }
        /// P2: the V axis admission sizes VMM KV with is the one the carrier
        /// loads (`SourceAdmission::kv_v` → `LoadCtx::kv_v_override`), for
        /// every `HIPFIRE_KV_V` setting.
        #[test]
        fn kv_v_resolves_once_for_admission_and_carrier() {
            use super::super::resolve_kv_v_from as r;
            for env in [None, Some(""), Some("q8")] {
                assert_eq!(r(None, env, 5, 1), None, "neutral env {env:?}");
                assert_eq!(r(Some("lloyd3"), env, 5, 1), Some("lloyd3"));
            }
            assert_eq!(r(None, Some("lloyd3"), 5, 1), Some("lloyd3"));
            assert_eq!(r(None, Some("lloyd3"), 6, 1), Some("lloyd3"));
            assert_eq!(
                r(Some("q8"), Some("lloyd3"), 5, 1),
                Some("q8"),
                "typed wins"
            );
            assert_eq!(
                r(Some(""), Some("lloyd3"), 5, 1),
                Some("lloyd3"),
                "empty typed = unset"
            );
            assert_eq!(
                r(None, Some("lloyd3"), 1, 1),
                None,
                "non-qwen35 ignores env"
            );
            assert_eq!(r(None, Some("lloyd3"), 5, 2), None, "EP ignores env");

            let trunk = write_hfq("kv-v-once", 5, false);
            let admit = |kv_mode, kv_v| {
                super::super::admit_source(
                    trunk.to_str().unwrap(),
                    1,
                    1,
                    KvBackendRequest::Automatic,
                    None,
                    "gfx1100",
                    None,
                    "auto",
                    None,
                    4096,
                    KvBackendHints {
                        kv_mode: Some(kv_mode),
                        kv_v,
                        ..KvBackendHints::without_device()
                    },
                )
                .map(|a| a.kv_v)
            };
            assert_eq!(
                admit("fwht3", Some("lloyd3")).unwrap().as_deref(),
                Some("lloyd3")
            );
            assert_eq!(admit("q8", Some("q8")).unwrap().as_deref(), Some("q8"));
            // The pair is validated at admission, before the resident model is
            // touched — not first discovered by the carrier.
            let err = admit("q8", Some("lloyd3")).unwrap_err();
            assert!(err.contains("Lloyd V"), "{err}");
            cleanup(&trunk);
        }
        /// gfx1151 FWHT K overrides refuse without silently falling back.
        /// Explicit legacy and mode-only FWHT fallback remain available.
        #[test]
        fn gfx1151_fwht_k_override_refuses_instead_of_legacy_fallback() {
            let trunk = write_hfq("kv-axis-refusal", 5, false);
            let path = trunk.to_str().unwrap();
            let admit = |request, arch, kv_mode, kv_k, vmm_runtime_available| {
                super::super::admit_source(
                    path,
                    1,
                    1,
                    request,
                    None,
                    arch,
                    None,
                    "auto",
                    None,
                    4096,
                    KvBackendHints {
                        kv_mode,
                        kv_k,
                        kv_v: None,
                        qwen_default_q8: true,
                        kv_adaptive: None,
                        cask: None,
                        deepseek4_heterogeneous: false,
                        vmm_runtime_available,
                        free_vram_bytes: None,
                    },
                )
            };
            let auto = KvBackendRequest::Automatic;
            for k in ["fwht2", "fwht3", "fwht4"] {
                for mode in [None, Some("auto"), Some("q8")] {
                    for runtime_available in [true, false] {
                        for request in [auto, KvBackendRequest::Explicit(KvBackend::Vmm)] {
                            let err = admit(request, "gfx1151", mode, Some(k), runtime_available)
                                .map(|_| ())
                                .unwrap_err();
                            assert!(
                                err.contains("no certified gfx1151 VMM FWHT path")
                                    && err.contains("--kv-backend legacy"),
                                "{err}"
                            );
                        }
                        let legacy = admit(
                            KvBackendRequest::Explicit(KvBackend::Legacy),
                            "gfx1151",
                            mode,
                            Some(k),
                            runtime_available,
                        )
                        .unwrap();
                        assert_eq!(legacy.kv_backend, KvBackend::Legacy);
                        assert!(legacy.kv_backend_reason.is_none());
                    }
                }
                let gfx1100 = admit(auto, "gfx1100", Some("q8"), Some(k), true).unwrap();
                assert_eq!(gfx1100.kv_backend, KvBackend::Vmm);
                assert!(gfx1100.kv_backend_reason.is_none());
            }
            let mode_spelling = admit(auto, "gfx1151", Some("fwht3"), None, true).unwrap();
            assert_eq!(mode_spelling.kv_backend, KvBackend::Legacy);
            assert!(mode_spelling
                .kv_backend_reason
                .unwrap()
                .contains("no tested VMM KV path"));
            let default = admit(auto, "gfx1151", Some("q8"), None, true).unwrap();
            assert_eq!(default.kv_backend, KvBackend::Vmm);
            let unavailable = admit(auto, "gfx1151", Some("q8"), None, false).unwrap();
            assert_eq!(unavailable.kv_backend, KvBackend::Legacy);
            assert!(unavailable
                .kv_backend_reason
                .unwrap()
                .contains("does not provide the virtual-memory (VMM) support"));
            let gfx1201 = admit(auto, "gfx1201", Some("q8"), Some("fwht3"), true).unwrap();
            assert_eq!(gfx1201.kv_backend, KvBackend::Vmm);
            assert!(gfx1201.kv_backend_reason.is_none());
            cleanup(&trunk);
        }
        #[test]
        fn moe_pins_legacy_context_on_discrete_and_sizes_itself_on_uma() {
            let path = write_hfq("moe-capacity", 6, false);
            let metadata = r#"{"config":{"text_config":{"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":4,"vocab_size":1000,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":1024,"max_position_embeddings":131072}}}"#;
            let embedding = HfqMemTensor {
                name: "model.embed_tokens.weight".into(),
                quant_type: 1,
                shape: vec![4, 4],
                group_size: 0,
                data: vec![0u8; 32],
            };
            write_hfqm_package_mem(&path, 6, metadata, &[embedding]).unwrap();
            let hints = KvBackendHints {
                kv_mode: Some("q8"),
                kv_k: None,
                kv_v: None,
                qwen_default_q8: true,
                kv_adaptive: None,
                cask: None,
                deepseek4_heterogeneous: false,
                vmm_runtime_available: true,
                free_vram_bytes: Some(24 << 30),
            };
            let omitted = super::super::admit_source(
                path.to_str().unwrap(),
                1,
                1,
                KvBackendRequest::Automatic,
                None,
                "gfx1100",
                None,
                "auto",
                None,
                0,
                hints,
            )
            .expect("MoE omission should not require dense PBS accounting");
            assert_eq!(omitted.max_seq, 32768);
            assert_eq!(omitted.kv_backend, KvBackend::Vmm);
            assert!(omitted.sequence_reason.is_some());
            let explicit = super::super::admit_source(
                path.to_str().unwrap(),
                1,
                1,
                KvBackendRequest::Automatic,
                None,
                "gfx1100",
                None,
                "auto",
                None,
                50000,
                hints,
            )
            .expect("explicit MoE max_seq must remain authoritative");
            assert_eq!(explicit.max_seq, 50000);
            assert!(explicit.sequence_reason.is_none());
            // Unified memory sizes MoE like dense: the trained window, capped
            // by the carrier's post-weight card measurement.
            let uma = super::super::admit_source(
                path.to_str().unwrap(),
                1,
                1,
                KvBackendRequest::Automatic,
                None,
                "gfx1151",
                None,
                "auto",
                None,
                0,
                hints,
            )
            .expect("UMA MoE omission sizes automatically");
            let sequence = uma.sequence.expect("UMA MoE resolves a sequence");
            assert_eq!((sequence.max_seq, sequence.bound), (131072, "pending"));
            assert!(uma.sequence_reason.is_none());
            cleanup(&path);
        }
        /// HIPF-34: a safetensors directory (ParoQuant) has no HFQ manifest to
        /// project from and its carrier never remeasures post-weight VRAM, so
        /// it keeps a fixed context under VMM KV instead of being refused.
        #[test]
        fn safetensors_dir_skips_automatic_vmm_sizing() {
            let dir =
                std::env::temp_dir().join(format!("hipfire-paro-dir-admit-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("config.json"),
                r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"text_config":{"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":4,"vocab_size":1000,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":1024,"max_position_embeddings":131072}}"#,
            )
            .unwrap();
            let header =
                br#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]}}"#;
            let mut shard = (header.len() as u64).to_le_bytes().to_vec();
            shard.extend_from_slice(header);
            shard.extend_from_slice(&[0u8; 32]);
            std::fs::write(dir.join("model.safetensors"), shard).unwrap();
            let hints = KvBackendHints {
                kv_mode: Some("q8"),
                vmm_runtime_available: true,
                free_vram_bytes: Some(96 << 30),
                ..KvBackendHints::without_device()
            };
            let admit = |max_seq| {
                super::super::admit_source(
                    dir.to_str().unwrap(),
                    1,
                    1,
                    KvBackendRequest::Automatic,
                    None,
                    "gfx1151",
                    None,
                    "off",
                    None,
                    max_seq,
                    hints,
                )
            };
            let explicit = admit(4096).expect("Dir MoE with explicit max_seq must load");
            assert_eq!(explicit.kv_backend, KvBackend::Vmm);
            assert_eq!(explicit.max_seq, 4096);
            assert!(explicit.sequence.is_none());
            assert!(explicit.sequence_reason.is_none());
            let omitted = admit(0).expect("Dir MoE without max_seq must load");
            assert_eq!(omitted.max_seq, 4096);
            assert!(omitted.sequence.is_none());
            assert!(omitted.sequence_reason.is_some());
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Measured legacy fit is backend-gated, not OS-gated: the automatic
        /// legacy fallback measures from free VRAM (bound `legacy-fit`)
        /// instead of taking the fixed 4096, on any load that resolves the
        /// legacy backend — explicit `--kv-backend legacy` on any OS, or
        /// automatic with a VMM refusal (which is every Windows load).
        mod legacy_fit {
            use super::*;

            fn hints(free: Option<usize>) -> KvBackendHints<'static> {
                KvBackendHints {
                    kv_mode: Some("q8"),
                    kv_k: None,
                    kv_v: None,
                    qwen_default_q8: true,
                    kv_adaptive: None,
                    cask: None,
                    deepseek4_heterogeneous: false,
                    vmm_runtime_available: true,
                    free_vram_bytes: free,
                }
            }

            fn admit_legacy(
                path: &std::path::Path,
                max_seq: usize,
                free: Option<usize>,
            ) -> Result<SourceAdmission, String> {
                crate::admission::admit_source(
                    path.to_str().unwrap(),
                    1,
                    1,
                    KvBackendRequest::Explicit(KvBackend::Legacy),
                    None,
                    "gfx1100",
                    None,
                    "auto",
                    None,
                    max_seq,
                    hints(free),
                )
            }

            #[test]
            fn automatic_measures_instead_of_fixed_4096() {
                let trunk = write_hfq("legacy-fit-dense", 5, false);
                let admitted =
                    admit_legacy(&trunk, 0, Some(24 << 30)).expect("legacy automatic must admit");
                assert_eq!(admitted.kv_backend, KvBackend::Legacy);
                assert!(admitted.sequence.is_none());
                assert!(admitted.legacy_measured, "measured, not fixed");
                // 24 GiB free dwarfs the tiny trunk: the trained window wins.
                assert_eq!(admitted.max_seq, 131072);
                cleanup(&trunk);
            }

            #[test]
            fn automatic_caps_below_model_ctx_when_tight() {
                let trunk = write_hfq("legacy-fit-tight", 5, false);
                // Derive a free budget that leaves room for exactly 10_000
                // tokens past every projected owner, so the pin is exact and
                // immune to scratch/stride changes.
                let target = 10_000usize;
                let free = {
                    let hfq = hipfire_runtime::hfq::HfqFile::open(&trunk).unwrap();
                    let source = hipfire_runtime::loader_api::ModelSource::Hfq(hfq);
                    let projection = crate::admission::qwen35_card_projection(
                        &source,
                        "gfx1100",
                        hints(Some(1 << 60)),
                    )
                    .expect("projection");
                    projection.weights
                        + projection.scratch
                        + projection.page_rounding
                        + super::super::super::QWEN35_LEGACY_CONTIGUITY_MARGIN_BYTES
                        + target * projection.bytes_per_token
                };
                let admitted =
                    admit_legacy(&trunk, 0, Some(free)).expect("derived budget must fit");
                assert!(admitted.legacy_measured);
                assert_eq!(admitted.max_seq, target);
                cleanup(&trunk);
            }

            #[test]
            fn automatic_refuses_when_nothing_fits() {
                let trunk = write_hfq("legacy-fit-oom", 5, false);
                let err = admit_legacy(&trunk, 0, Some(256 << 20))
                    .map(|_| ())
                    .expect_err("weights alone exceed 256 MiB");
                assert!(err.contains("legacy"), "{err}");
                assert!(err.contains("memory.max_seq"), "{err}");
                cleanup(&trunk);
            }

            #[test]
            fn fixed_4096_pinned_without_free_vram() {
                let trunk = write_hfq("legacy-fit-nofree", 5, false);
                let admitted =
                    admit_legacy(&trunk, 0, None).expect("no measurement must still load");
                assert_eq!(admitted.max_seq, 4096);
                assert!(!admitted.legacy_measured);
                assert!(admitted.sequence.is_none());
                cleanup(&trunk);
            }

            #[test]
            fn explicit_max_seq_unchanged() {
                let trunk = write_hfq("legacy-fit-explicit", 5, false);
                let admitted =
                    admit_legacy(&trunk, 8192, Some(24 << 30)).expect("explicit must admit");
                assert_eq!(admitted.max_seq, 8192);
                assert!(!admitted.legacy_measured);
                cleanup(&trunk);
            }

            #[test]
            fn moe_keeps_fixed_4096() {
                let path = write_hfq("legacy-fit-moe", 6, false);
                let metadata = r#"{"config":{"text_config":{"hidden_size":2048,"num_hidden_layers":4,"num_attention_heads":16,"num_key_value_heads":4,"vocab_size":1000,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":1024,"max_position_embeddings":131072}}}"#;
                let embedding = HfqMemTensor {
                    name: "model.embed_tokens.weight".into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                };
                write_hfqm_package_mem(&path, 6, metadata, &[embedding]).unwrap();
                let admitted =
                    admit_legacy(&path, 0, Some(24 << 30)).expect("MoE legacy automatic must load");
                assert_eq!(admitted.max_seq, 4096);
                assert!(!admitted.legacy_measured);
                cleanup(&path);
            }
        }
    }
    /// The loader gate and the arch resolvers reach the same verdict for the
    /// same facts: the gate's refusal (when a spill was requested) is exactly
    /// the predicate's `Refused` arm, and silence is exactly `Supported`. The
    /// arch side is asserted in the arch crate (`spill_resolver_agrees_with_*`
    /// there calls the same predicate through its own facts — this pins the
    /// gate half of the agreement here, where the snapshot trigger lives).
    mod spill_agreement {
        use super::*;
        use hipfire_config::memory::{spill_support, SpillRefusal, SpillSupport};

        /// The arch resolver's verdict for these facts: one call, not a
        /// condition list — the shape both twins must copy.
        fn arch_verdict(
            arch_id: u32,
            tp: usize,
            pp: usize,
            ep_experts: Option<usize>,
            paged_experts: bool,
            reap_keep: bool,
            is_vl: bool,
        ) -> SpillSupport {
            spill_support(arch_id, tp, pp, ep_experts, paged_experts, reap_keep, is_vl)
        }

        fn verdicts_agree(
            arch_id: u32,
            tp: usize,
            pp: usize,
            ep_experts: Option<usize>,
            paged_experts: bool,
            reap_keep: bool,
            is_vl: bool,
        ) {
            let gate = spill_support(arch_id, tp, pp, ep_experts, paged_experts, reap_keep, is_vl);
            assert_eq!(
                arch_verdict(arch_id, tp, pp, ep_experts, paged_experts, reap_keep, is_vl),
                gate,
                "loader gate and arch resolver disagree"
            );
        }

        #[test]
        fn loader_gate_and_arch_resolver_agree_on_every_dimension() {
            // Supported shape (dense included): both stay silent.
            for arch_id in [5u32, 6] {
                verdicts_agree(arch_id, 1, 1, None, false, false, false);
            }
            // Every refusal dimension: both refuse, same arm.
            verdicts_agree(9, 1, 1, None, false, false, false);
            verdicts_agree(6, 2, 1, None, false, false, false);
            verdicts_agree(6, 1, 2, None, false, false, false);
            verdicts_agree(6, 1, 1, Some(4), false, false, false);
            verdicts_agree(6, 1, 1, None, false, false, true);
            verdicts_agree(6, 1, 1, None, true, false, false);
            verdicts_agree(6, 1, 1, None, false, true, false);
        }

        #[test]
        fn refusal_messages_name_the_key_and_the_reason() {
            // Spot-check the arm the auto-fit unification added: the VL
            // message is verbatim (tower, lowered loader, unset-both-keys).
            let msg = spill_refusal_message(SpillRefusal::VisionTower, "the-keys");
            assert!(msg.contains("the-keys"), "names the key: {msg}");
            assert!(msg.contains("vision tower"), "names the reason: {msg}");
            assert!(msg.contains("Unset both keys"), "names the remedy: {msg}");
            // Every other arm renders non-empty too (verbatim sentences above).
            for refusal in [
                SpillRefusal::Arch(9),
                SpillRefusal::Tp(2),
                SpillRefusal::Pp(2),
                SpillRefusal::ExpertParallel(4),
                SpillRefusal::PagedExperts,
                SpillRefusal::ReapKeepMap,
            ] {
                assert!(
                    !spill_refusal_message(refusal, "the-keys").is_empty(),
                    "{refusal:?} must render"
                );
            }
        }

        #[test]
        fn dense_unset_model_stays_resident_and_silent() {
            // The zero-diff default: a dense model with both keys unset is
            // compatible (`Supported`), and `Full`/`Full` budgets resolve to
            // the all-device placement with no capacity check — the refusal
            // below can never fire on this path.
            use hipfire_config::memory::{effective_offload_budgets, OffloadBudget};
            assert_eq!(
                spill_support(5, 1, 1, None, false, false, false),
                SpillSupport::Supported
            );
            assert_eq!(
                effective_offload_budgets(),
                (OffloadBudget::Full, OffloadBudget::Full)
            );
        }
    }
    mod head_overlay {
        use super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};

        fn write_tensors(
            name: &str,
            arch_id: u32,
            specs: &[(&str, u8, Vec<u32>, Vec<u8>)],
        ) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-head-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            let tensors: Vec<HfqMemTensor> = specs
                .iter()
                .map(|(n, qt, shape, data)| HfqMemTensor {
                    name: (*n).into(),
                    quant_type: *qt,
                    shape: shape.clone(),
                    group_size: 0,
                    data: data.clone(),
                })
                .collect();
            write_hfqm_package_mem(&path, arch_id, "{}", &tensors).unwrap();
            path
        }

        fn maple_trunk(name: &str) -> std::path::PathBuf {
            write_tensors(
                name,
                15,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 3, vec![2, 4], vec![1u8; 32]),
                ],
            )
        }

        fn maple_head(name: &str, byte: u8) -> std::path::PathBuf {
            write_tensors(
                name,
                15,
                &[("lm_head.weight", 13, vec![2, 4], vec![byte; 32])],
            )
        }

        fn cleanup(paths: &[std::path::PathBuf]) {
            for path in paths {
                let _ = std::fs::remove_file(path);
                let _ = std::fs::remove_dir(path.parent().unwrap());
            }
        }

        fn head_bytes(admitted: &SourceAdmission) -> Vec<u8> {
            let ModelSource::Hfq(hfq) = &admitted.source else {
                panic!("expected HFQ source");
            };
            hfq.tensor_data("lm_head.weight")
                .map(|(_, d)| d.to_vec())
                .expect(" admitted source must serve lm_head.weight")
        }

        /// A valid maple head admits and shadows the base head in the
        /// retained source — loading consumes this, never reopens the file.
        #[test]
        fn head_on_maple_admits_and_shadows_base() {
            let trunk = maple_trunk("d-trunk");
            let head = maple_head("d-head", 7);
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(head.to_str().unwrap()),
                4096,
            )
            .expect("valid maple head must admit");
            assert!(
                admitted.carrier.is_some_and(|c| c.name() == "maple"),
                "trunk still routes to maple"
            );
            assert_eq!(
                head_bytes(&admitted),
                vec![7u8; 32],
                "retained source serves the overlay head, not the base"
            );
            cleanup(&[trunk, head]);
        }

        /// Empty head string opts out exactly like vision/draft.
        #[test]
        fn empty_head_string_is_unset() {
            let trunk = maple_trunk("e-trunk");
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(""),
                4096,
            )
            .expect("empty head must admit as unset");
            assert_eq!(
                head_bytes(&admitted),
                vec![1u8; 32],
                "unset head serves the baked base head"
            );
            cleanup(&[trunk]);
        }

        /// A head on a non-Maple trunk refuses instead of being ignored.
        #[test]
        fn head_on_non_maple_refuses() {
            let trunk = write_tensors(
                "f-trunk",
                5,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 3, vec![2, 4], vec![1u8; 32]),
                ],
            );
            let head = maple_head("f-head", 7);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("non-maple head must refuse");
            assert!(err.contains("only serve Maple"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A head with expert-parallel topology refuses in preflight.
        #[test]
        fn head_on_ep_topology_refuses() {
            let trunk = maple_trunk("g-trunk");
            let mut base = ModelSource::from_path(trunk.to_str().unwrap()).expect("open trunk");
            let err = super::super::admit_head_overlay(
                Some("g-head"),
                &mut base,
                15,
                EffectiveTopology::Expert(2),
            )
            .expect_err("EP head must refuse");
            assert!(err.contains("single-device"), "refusal: {err}");
            cleanup(&[trunk]);
        }

        /// A head cannot stack on an installed REAP overlay (single slot).
        #[test]
        fn head_with_reap_overlay_refuses() {
            let trunk = maple_trunk("h-trunk");
            let plan = std::env::temp_dir()
                .join(format!("hipfire-head-admit-{}-h-plan", std::process::id()));
            std::fs::create_dir_all(&plan).unwrap();
            // Install a REAP overlay through the injected plan (deterministic:
            // no process-config snapshot involved).
            let plan_file = plan.join("overlay.hfq");
            let staged = write_tensors(
                "h-ov",
                15,
                &[("lm_head.weight", 8, vec![2, 4], vec![9u8; 32])],
            );
            std::fs::rename(&staged, &plan_file).unwrap();
            let head = maple_head("h-head", 7);
            let base = HfqFile::open_with_reap_plan(&trunk, Some(&plan)).expect("open trunk");
            assert!(base.has_overlay(), "REAP overlay must install");
            let mut source = ModelSource::Hfq(base);
            let err = super::super::admit_head_overlay(
                Some(head.to_str().unwrap()),
                &mut source,
                15,
                EffectiveTopology::Single,
            )
            .expect_err("REAP+head must refuse");
            assert!(err.contains("REAP"), "refusal: {err}");
            cleanup(&[trunk, head, plan_file, staged]);
            let _ = std::fs::remove_dir(&plan);
        }

        /// A truncated head (valid header/index, short payload) refuses.
        #[test]
        fn truncated_head_refuses() {
            let trunk = maple_trunk("i-trunk");
            let head = maple_head("i-head", 7);
            let len = std::fs::metadata(&head).unwrap().len();
            std::fs::OpenOptions::new()
                .write(true)
                .open(&head)
                .unwrap()
                .set_len(len - 10)
                .unwrap();
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("truncated head must refuse");
            assert!(err.contains("truncated"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A head stamped for another arch refuses.
        #[test]
        fn head_with_wrong_arch_refuses() {
            let trunk = maple_trunk("j-trunk");
            let head = write_tensors(
                "j-head",
                9,
                &[("lm_head.weight", 13, vec![2, 4], vec![7u8; 32])],
            );
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("wrong-arch head must refuse");
            assert!(err.contains("arch_id"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A full model passed as --head refuses (single-tensor guard).
        #[test]
        fn full_model_as_head_refuses() {
            let trunk = maple_trunk("k-trunk");
            let head = write_tensors(
                "k-head",
                15,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 13, vec![2, 4], vec![7u8; 32]),
                ],
            );
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                "auto",
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("full model as head must refuse");
            assert!(err.contains("expected only"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }
    }

    mod qwen4_memory {
        use super::super::{qwen4_memory_admission, QWEN4_MEMORY_HEADROOM_BYTES};

        const MIB: u64 = 1 << 20;
        const CAPACITY: usize = 262_144;
        /// Strix Halo receipt: 98,304 MiB carve-out, 78,547 MiB used after a
        /// default Flash-Next load (native MTP, max_seq 262144, VMM).
        const HALO_FREE: u64 = (98_304 - 78_547) * MIB;
        const MTP_REQUEST: u64 = 740 * MIB;

        /// Linear growth: 15,392 MiB of target + MTP QSA arenas and 512 MiB
        /// of gather workspace over the 262,144-token capacity.
        fn growth(tokens: usize) -> (u64, u64) {
            let tokens = tokens.min(CAPACITY) as u64;
            (
                15_392 * MIB * tokens / CAPACITY as u64,
                512 * MIB * tokens / CAPACITY as u64,
            )
        }

        fn fixed() -> u64 {
            MTP_REQUEST + QWEN4_MEMORY_HEADROOM_BYTES
        }

        #[test]
        fn halo_default_admits_the_whole_model_context() {
            let admission = qwen4_memory_admission(
                CAPACITY,
                None,
                HALO_FREE,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap();
            assert_eq!(admission.max_seq, CAPACITY);
            assert_eq!(admission.card_cap, CAPACITY);
            assert_eq!(admission.context_growth, 15_392 * MIB);
            assert_eq!(admission.gather_growth, 512 * MIB);
            assert!(admission.growth() + admission.headroom <= HALO_FREE);
        }

        #[test]
        fn automatic_context_caps_at_the_largest_fit() {
            let free = 12 << 30;
            let admission = qwen4_memory_admission(
                CAPACITY,
                None,
                free,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap();
            let cap = admission.card_cap;
            assert!(cap > 0 && cap < CAPACITY);
            assert_eq!(admission.max_seq, cap);
            let total = |tokens| {
                let (context, gather) = growth(tokens);
                context + gather + fixed()
            };
            assert!(total(cap) <= free);
            assert!(total(cap + 1) > free);
        }

        #[test]
        fn explicit_context_is_honored_when_it_fits() {
            let admission = qwen4_memory_admission(
                CAPACITY,
                Some(CAPACITY),
                HALO_FREE,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap();
            assert_eq!(admission.max_seq, CAPACITY);
            let short = qwen4_memory_admission(
                8192,
                Some(8192),
                HALO_FREE,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap();
            assert_eq!(short.max_seq, 8192);
            assert_eq!(short.card_cap, 8192);
        }

        #[test]
        fn explicit_context_that_cannot_grow_refuses_at_load() {
            let free = 12 << 30;
            let cap = qwen4_memory_admission(
                CAPACITY,
                None,
                free,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap()
            .card_cap;
            let error = qwen4_memory_admission(
                CAPACITY,
                Some(CAPACITY),
                free,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap_err();
            assert!(
                error.contains("max_seq 262144 does not fit device memory"),
                "{error}"
            );
            assert!(error.contains(&format!("at most {cap}")), "{error}");
            assert!(
                error.contains(&format!(
                    "with the current free host memory the context fits up to {cap} tokens"
                )),
                "{error}"
            );
            assert!(
                error.contains("this limit moves with free host memory)"),
                "{error}"
            );
            assert!(!error.contains("PLE prewarm budget"), "{error}");
            assert!(
                error.contains(&format!(
                    "{} bytes QSA context incl. any F16 arena + {} bytes QSA gather workspace + {} bytes native MTP request buffers",
                    15_392 * MIB,
                    512 * MIB,
                    740 * MIB
                )),
                "{error}"
            );
        }

        #[test]
        fn automatic_context_refuses_when_nothing_fits() {
            let error = qwen4_memory_admission(
                CAPACITY,
                None,
                fixed() - 1,
                MTP_REQUEST,
                0,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth,
            )
            .unwrap_err();
            assert!(
                error.contains("cannot hold an automatic context"),
                "{error}"
            );
            assert!(
                error.contains("this limit moves with free host memory)"),
                "{error}"
            );
            assert!(!error.contains("PLE prewarm budget"), "{error}");
        }

        /// Strix Halo serve receipt (Flash-Next, native MTP on, max_seq
        /// 262144): 62,336 bytes of QSA context and 2,048 bytes of gather
        /// workspace per token.
        mod halo_256k {
            use super::{qwen4_memory_admission, QWEN4_MEMORY_HEADROOM_BYTES};

            const CAPACITY: usize = 262_144;
            const FREE: u64 = 34_674_597_888;
            const MTP_REQUEST: u64 = 1_237_893_184;
            /// Requested resident PLE: 14 GiB.
            const PLE: u64 = 15_032_385_536;

            fn growth(tokens: usize) -> (u64, u64) {
                (tokens as u64 * 62_336, tokens as u64 * 2_048)
            }

            fn admit(
                requested: Option<usize>,
                free: u64,
                ple: u64,
            ) -> Result<super::super::super::Qwen4MemoryAdmission, String> {
                qwen4_memory_admission(
                    CAPACITY,
                    requested,
                    free,
                    MTP_REQUEST,
                    ple,
                    QWEN4_MEMORY_HEADROOM_BYTES,
                    growth,
                )
            }

            #[test]
            fn halo_256k_serve_fits_by_shrinking_ple() {
                assert_eq!(growth(CAPACITY), (16_341_008_384, 536_870_912));
                for requested in [Some(CAPACITY), None] {
                    let admission = admit(requested, FREE, PLE).unwrap();
                    assert_eq!(admission.max_seq, 262_144);
                    assert_eq!(admission.card_cap, 262_144);
                    assert_eq!(admission.ple_budget, 15_032_385_536);
                    assert_eq!(admission.resident_ple, 14_411_341_760);
                    assert!(admission.ple_shrunk());
                    // The shrunk PLE takes the card to exactly its free memory.
                    assert_eq!(admission.required(), FREE);
                }
            }

            #[test]
            fn halo_256k_refuses_when_the_context_itself_does_not_fit() {
                let free = 19_000_000_000;
                let error = admit(Some(CAPACITY), free, PLE).unwrap_err();
                assert!(error.contains("max_seq 262144 does not fit"), "{error}");
                // The cap is the largest context that fits with PLE 0.
                let cap = 242_523;
                let total = |tokens: usize| {
                    let (context, gather) = growth(tokens);
                    context + gather + MTP_REQUEST + QWEN4_MEMORY_HEADROOM_BYTES
                };
                assert!(total(cap) <= free);
                assert!(total(cap + 1) > free);
                assert!(error.contains(&format!("at most {cap}")), "{error}");
                assert!(
                    error.contains("needs 20263256128 bytes free after load"),
                    "{error}"
                );
                assert!(!error.contains("PLE prewarm budget"), "{error}");
                // The cap does not depend on the requested PLE budget.
                for ple in [0, PLE, u64::MAX] {
                    assert_eq!(admit(None, free, ple).unwrap().card_cap, cap);
                }
                let auto = admit(None, free, PLE).unwrap();
                assert_eq!(auto.max_seq, cap);
                // The 22,336 bytes the cap leaves free are all the PLE gets.
                assert_eq!(auto.resident_ple, 22_336);
                assert!(auto.ple_shrunk());
            }

            #[test]
            fn halo_256k_keeps_the_full_ple_when_everything_fits() {
                let free = 40u64 << 30;
                let admission = admit(Some(CAPACITY), free, PLE).unwrap();
                assert_eq!(admission.max_seq, 262_144);
                assert_eq!(admission.resident_ple, 15_032_385_536);
                assert_eq!(admission.ple_budget, 15_032_385_536);
                assert!(!admission.ple_shrunk());
                assert_eq!(admission.required(), 20_263_256_128 + PLE);
            }
        }
    }

    /// Strix Halo (gfx1151, 121 GiB shared host/GTT) admission of the
    /// Flash-Next geometry with the opt-in QSA F16 arena, against the
    /// measured free memory and the resident PLE budget. Growth comes from
    /// the real `Qwen4Config` (48 layers, every 4th full attention = 12 QSA
    /// layers, 2 KV heads x head_dim 256, 1 index head x 128, compress 4),
    /// row-exact like `Qwen4State::context_growth_bytes`; the gathered
    /// workspace is the real `qsa_gathered_wmma_scratch_bytes` when the
    /// arena is off (with it on the route does not need the shared workspace).
    mod qwen4_halo_arena {
        use super::super::{
            qwen4_context_sequence, qwen4_memory_admission, qwen4_requested_context,
            QWEN4_MEMORY_HEADROOM_BYTES,
        };
        use hipfire_arch_qwen4::config::{
            Qwen4Config, ARCHITECTURE_NAME, MODEL_TYPE, TEXT_MODEL_TYPE,
        };
        use rdna_compute::tensor_ops::{qsa_gathered_wmma_scratch_bytes, QsaStateLayout};

        const MIB: u64 = 1 << 20;
        const CAPACITY: usize = 262_144;
        /// Free device memory after the load (min of HIP free and
        /// MemAvailable + TTM pool).
        const FREE: u64 = 33_605 * MIB;
        /// Resident PLE budget: `ple_prewarm_budget_for(Auto, 23,038 MiB
        /// MemAvailable)` = 23,038 - 8,192 MiB (below the 15 GiB cap).
        const PLE: u64 = 14_846 * MIB;
        /// Free device memory with the whole PLE budget already taken out:
        /// what a context must fit in when the PLE page cache yields to it
        /// (18,759 MiB, the card the pre-shrink admission refused on).
        const TIGHT_FREE: u64 = FREE - PLE;

        fn config() -> Qwen4Config {
            let layers: Vec<_> = (0..48)
                .map(|idx| {
                    if idx % 4 == 3 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                })
                .collect();
            // Parsed from text: a `json!` literal this size overflows the
            // macro's recursion limit.
            let layers = serde_json::to_string(&layers).unwrap();
            let text = r#"{
                "architectures": ["@ARCH@"],
                "model_type": "@MODEL@",
                "text_config": {
                    "model_type": "@TEXT@",
                    "dtype": "bfloat16",
                    "hidden_size": 2560,
                    "vocab_size": 248320,
                    "num_hidden_layers": 48,
                    "max_position_embeddings": 262144,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "partial_rotary_factor": 0.25,
                    "rope_parameters": {"rope_theta": 10000000.0},
                    "attention_bias": false,
                    "layer_types": @LAYERS@,
                    "full_attention_interval": 4,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "mamba_ssm_dtype": "float32",
                    "indexer_n_heads": 4,
                    "indexer_kv_heads": 1,
                    "indexer_head_dim": 128,
                    "indexer_budget": 2048,
                    "indexer_compress_ratio": 4,
                    "num_experts": 512,
                    "num_experts_per_tok": 10,
                    "moe_intermediate_size": 640,
                    "shared_expert_intermediate_size": 640,
                    "norm_topk_prob": true,
                    "hc_count": 4,
                    "hc_lowrank": 320,
                    "ple_layer_ids": [2],
                    "ple_conv_kernel_size": 4,
                    "ple_embed_dim": 2560,
                    "split_ngram_parts": 128,
                    "heads_per_ngram": 8,
                    "ngram_size": 3,
                    "ngram_vocab_size_base": 20000000,
                    "make_ngram_vocab_size_divisible_by": 128,
                    "eos_token_id": 248044,
                    "tie_word_embeddings": false,
                    "mtp_num_hidden_layers": 1,
                    "mtp_use_dedicated_embeddings": false,
                    "output_gate_type": "sigmoid",
                    "mtp": {
                        "hybrid": true,
                        "layer_types": ["full_attention"],
                        "num_hidden_layers": 1,
                        "rope_theta": 10000000.0
                    }
                }
            }"#
            .replace("@ARCH@", ARCHITECTURE_NAME)
            .replace("@MODEL@", MODEL_TYPE)
            .replace("@TEXT@", TEXT_MODEL_TYPE)
            .replace("@LAYERS@", &layers);
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            Qwen4Config::from_value(&value).expect("Flash-Next geometry")
        }

        /// `(context, gather)` growth of the target's QSA layers through
        /// `tokens` (nothing mapped yet, no native MTP head: the receipt's
        /// 20,352 MiB): per layer the rows `tokens` need, from the real
        /// `qsa_context_arena_bytes` (pooled keys by `ceil(t / 4)`, the F16
        /// arena rounded to four tokens), as `Qwen4State::context_growth_bytes`
        /// counts them.
        fn growth(arena: bool) -> impl Fn(usize) -> (u64, u64) {
            growth_with(arena, false)
        }

        /// [`growth`] with the pooled BF16 mirror (`pooled_bf16`) on or off:
        /// the mirror adds `ceil(t / 4)` blocks of 128 BF16 channels per layer.
        fn growth_with(arena: bool, mirror: bool) -> impl Fn(usize) -> (u64, u64) {
            let config = config();
            assert_eq!(config.n_full_layers(), 12);
            let layout = QsaStateLayout {
                f16_arena: arena,
                pooled_bf16: mirror,
                ..QsaStateLayout::F32
            };
            let layers = config.n_full_layers() as u64;
            let kv_heads = config.num_key_value_heads;
            move |tokens| {
                let tokens = tokens.min(CAPACITY);
                let gather = if arena {
                    0
                } else {
                    qsa_gathered_wmma_scratch_bytes(kv_heads, tokens).unwrap() as u64
                };
                let context =
                    layers * config.qsa_context_arena_bytes(tokens, layout).unwrap() as u64;
                (context, gather)
            }
        }

        fn admit(
            requested: Option<usize>,
            arena: bool,
            resident_ple: u64,
        ) -> Result<super::super::Qwen4MemoryAdmission, String> {
            admit_at(FREE, requested, arena, resident_ple)
        }

        fn admit_at(
            free: u64,
            requested: Option<usize>,
            arena: bool,
            resident_ple: u64,
        ) -> Result<super::super::Qwen4MemoryAdmission, String> {
            qwen4_memory_admission(
                CAPACITY,
                requested,
                free,
                0,
                resident_ple,
                QWEN4_MEMORY_HEADROOM_BYTES,
                growth(arena),
            )
        }

        #[test]
        fn config_costs_are_the_expected_halo_figures() {
            let (with_arena, _) = growth(true)(CAPACITY);
            let (plain, gather) = growth(false)(CAPACITY);
            assert_eq!(plain, 14_208 * MIB);
            assert_eq!(gather, 512 * MIB);
            assert_eq!(with_arena, 20_352 * MIB);
            assert_eq!(with_arena - plain, 6_144 * MIB);
            // Row-exact: 12 x (4608 t + 512 ceil(t/4) + 8192 ceil(t/4)).
            let cost = |t: u64| 12 * (4608 * t + 512 * t.div_ceil(4) + 8192 * t.div_ceil(4));
            for t in [215_245u64, 215_246, 215_247, 215_248, CAPACITY as u64] {
                assert_eq!(growth(true)(t as usize).0, cost(t), "{t}");
            }
            assert_eq!(cost(215_245), 17_522_743_296);
            assert_eq!(cost(215_246), 17_522_798_592);
        }

        #[test]
        fn halo_arena_context_fits_by_shrinking_ple() {
            // Without any PLE: 20,352 + 2,048 <= 33,605 MiB.
            let admission = admit(Some(CAPACITY), true, 0).unwrap();
            assert_eq!(admission.max_seq, CAPACITY);
            assert_eq!(admission.card_cap, CAPACITY);
            assert_eq!(admission.required(), (20_352 + 2_048) * MIB);
            assert!(!admission.ple_shrunk());
            // The requested 14,846 MiB PLE budget on top would take it over
            // the card (20,352 + 2,048 + 14,846 = 37,246 MiB > 33,605 MiB by
            // 3,641 MiB); the context takes precedence and the PLE shrinks to
            // the 11,205 MiB the card has left.
            assert_eq!((20_352 + 2_048 + 14_846) * MIB - FREE, 3_641 * MIB);
            let admission = admit(Some(CAPACITY), true, PLE).unwrap();
            assert_eq!(admission.max_seq, CAPACITY);
            assert_eq!(admission.card_cap, CAPACITY);
            assert_eq!(admission.ple_budget, PLE);
            assert_eq!(admission.resident_ple, 11_205 * MIB);
            assert!(admission.ple_shrunk());
            assert_eq!(admission.required(), FREE);
            // Auto resolves to the same whole context and PLE.
            assert_eq!(admit(None, true, PLE).unwrap(), admission);
        }

        #[test]
        fn halo_arena_context_over_the_card_is_refused_with_exact_cap() {
            let error = admit_at(TIGHT_FREE, Some(CAPACITY), true, PLE).unwrap_err();
            assert_eq!(TIGHT_FREE - QWEN4_MEMORY_HEADROOM_BYTES, 17_522_753_536);
            assert!(
                error.contains("max_seq 262144 does not fit device memory"),
                "{error}"
            );
            assert!(error.contains("at most 215245"), "{error}");
            for part in [
                format!("needs {} bytes free after load", (20_352 + 2_048) * MIB),
                format!("{} bytes QSA context incl. any F16 arena", 20_352 * MIB),
                "0 bytes QSA gather workspace".to_string(),
                "0 bytes native MTP request buffers".to_string(),
                format!("= {} bytes growth", 20_352 * MIB),
                format!("{} bytes headroom", QWEN4_MEMORY_HEADROOM_BYTES),
                format!("{TIGHT_FREE} bytes is free after load"),
            ] {
                assert!(error.contains(&part), "{part}: {error}");
            }
            assert!(
                error.contains(
                    "with the current free host memory the context fits up to 215245 tokens"
                ),
                "{error}"
            );
            assert!(
                error.contains("this limit moves with free host memory)"),
                "{error}"
            );
            assert!(!error.contains("PLE prewarm budget"), "{error}");
        }

        #[test]
        fn halo_arena_card_cap_is_the_largest_fit() {
            let admission = admit_at(TIGHT_FREE, None, true, PLE).unwrap();
            assert_eq!(admission.card_cap, 215_245);
            assert_eq!(admission.max_seq, 215_245);
            // The cap leaves 10,240 bytes: all the PLE the context allows.
            assert_eq!(admission.ple_budget, PLE);
            assert_eq!(admission.resident_ple, 10_240);
            assert!(admission.ple_shrunk());
            assert_eq!(admission.required(), TIGHT_FREE);
            let next = growth(true)(215_246).0 + QWEN4_MEMORY_HEADROOM_BYTES;
            assert_eq!(next - TIGHT_FREE, 45_056);
            // The cap does not depend on the requested PLE budget.
            for ple in [0, PLE, u64::MAX] {
                assert_eq!(
                    admit_at(TIGHT_FREE, None, true, ple).unwrap().card_cap,
                    215_245
                );
            }
            // An explicit context at the cap is honored, one past it refuses.
            assert_eq!(
                admit_at(TIGHT_FREE, Some(215_245), true, PLE)
                    .unwrap()
                    .max_seq,
                215_245
            );
            assert!(admit_at(TIGHT_FREE, Some(215_246), true, PLE).is_err());
        }

        /// The default-on pooled BF16 mirror is charged row-exact
        /// (`256 * ceil(t / 4)` bytes per QSA layer) in the growth the
        /// admission sizes the context against.
        #[test]
        fn halo_pooled_bf16_mirror_is_counted_in_the_card_cap() {
            let off = growth_with(true, false);
            let on = growth_with(true, true);
            for t in [1usize, 4, 5, 215_245, 215_246, CAPACITY] {
                let t_u = t as u64;
                assert_eq!(on(t).0 - off(t).0, 12 * 256 * t_u.div_ceil(4), "{t}");
            }
            assert_eq!(on(CAPACITY).0 - off(CAPACITY).0, 192 * MIB);
            let admission = qwen4_memory_admission(
                CAPACITY,
                None,
                TIGHT_FREE,
                0,
                PLE,
                QWEN4_MEMORY_HEADROOM_BYTES,
                &on,
            )
            .unwrap();
            let cap = admission.card_cap;
            assert!(cap < 215_245, "the mirror lowers the cap: {cap}");
            let total = |t: usize| on(t).0 + on(t).1 + QWEN4_MEMORY_HEADROOM_BYTES;
            assert!(total(cap) <= TIGHT_FREE);
            assert!(total(cap + 1) > TIGHT_FREE);
            let error = qwen4_memory_admission(
                CAPACITY,
                Some(CAPACITY),
                TIGHT_FREE,
                0,
                PLE,
                QWEN4_MEMORY_HEADROOM_BYTES,
                &on,
            )
            .unwrap_err();
            assert!(error.contains(&format!("at most {cap}")), "{error}");
            // On the measured card the whole context fits with the mirror
            // counted: the PLE shrinks to 33,605 - 20,544 - 2,048 MiB.
            let whole = qwen4_memory_admission(
                CAPACITY,
                Some(CAPACITY),
                FREE,
                0,
                PLE,
                QWEN4_MEMORY_HEADROOM_BYTES,
                &on,
            )
            .unwrap();
            assert_eq!(whole.context_growth, 20_544 * MIB);
            assert_eq!(whole.resident_ple, 11_013 * MIB);
            assert!(whole.ple_shrunk());
        }

        #[test]
        fn halo_without_the_arena_accepts_the_whole_context() {
            for requested in [None, Some(CAPACITY)] {
                let admission = admit(requested, false, PLE).unwrap();
                assert_eq!(admission.max_seq, CAPACITY);
                assert_eq!(admission.card_cap, CAPACITY);
                assert_eq!(admission.context_growth, 14_208 * MIB);
                assert_eq!(admission.gather_growth, 512 * MIB);
                // 14,208 + 512 + 14,846 + 2,048 = 31,614 MiB <= 33,605 MiB:
                // the whole PLE budget is kept.
                assert_eq!(admission.resident_ple, PLE);
                assert!(!admission.ple_shrunk());
                assert_eq!(admission.required(), 31_614 * MIB);
            }
        }

        #[test]
        fn discrete_arch_resolves_no_sequence_and_keeps_the_explicit_context() {
            const KV: &str = "bf16";
            let model_ctx = 262_144;
            for arch in ["gfx1100", "gfx1201"] {
                let unified = hipfire_config::is_unified_memory_arch(arch);
                assert!(!unified, "{arch}");
                // An omitted max_seq is the default context, an explicit
                // 262144 is kept; neither resolves a memory-admitted sequence.
                assert_eq!(
                    qwen4_requested_context(unified, 0),
                    hipfire_arch_qwen4::QWEN4_DEFAULT_CONTEXT
                );
                assert_eq!(qwen4_requested_context(unified, CAPACITY), CAPACITY);
                assert!(qwen4_context_sequence(unified, 0, model_ctx, KV).is_none());
                assert!(qwen4_context_sequence(unified, CAPACITY, model_ctx, KV).is_none());
            }
            // Halo resolves a sequence: omitted is pending at the model
            // context, explicit is the user's bound (checked after load).
            let halo = hipfire_config::is_unified_memory_arch("gfx1151");
            assert!(halo);
            assert_eq!(qwen4_requested_context(halo, 0), 0);
            let pending = qwen4_context_sequence(halo, 0, model_ctx, KV).unwrap();
            assert_eq!((pending.max_seq, pending.bound), (model_ctx, "pending"));
            let user = qwen4_context_sequence(halo, CAPACITY, model_ctx, KV).unwrap();
            assert_eq!((user.max_seq, user.bound), (CAPACITY, "user"));
            assert_eq!(user.card_cap, 0);
        }

        #[test]
        fn halo_free_is_the_gtt_clamp_of_the_measured_host_memory() {
            use rdna_compute::uma_memory::{device_free_from, DeviceMemory};
            // 121 GiB GTT-backed APU (1 GiB carve-out): hipMemGetInfo still
            // shows 40,000 MiB free, MemAvailable + TTM pool only 33,605 MiB.
            let halo = DeviceMemory {
                device_free: 40_000 * MIB,
                device_total: 121 << 30,
                uma: true,
                carveout_total: Some(1 << 30),
                host_available: Some(FREE),
            };
            assert_eq!(device_free_from(halo), Some(FREE));
            // The host side unreadable: refuse to size blind.
            let blind = DeviceMemory {
                host_available: None,
                ..halo
            };
            assert_eq!(device_free_from(blind), None);
            // A carve-out APU and a discrete card trust hipMemGetInfo.
            let carve = DeviceMemory {
                carveout_total: Some(121 << 30),
                ..halo
            };
            assert_eq!(device_free_from(carve), Some(40_000 * MIB));
            let discrete = DeviceMemory { uma: false, ..halo };
            assert_eq!(device_free_from(discrete), Some(40_000 * MIB));
        }
    }
}

#[cfg(test)]
mod qwen4_pp_admission_tests {
    use super::*;
    use hipfire_arch_qwen4::npu_trunk::NpuTrunkMode;
    use hipfire_runtime::mesh_placement::EdgeTransport;

    fn rank(ordinal: i32, pci: Option<&str>, arch: &str) -> AdmittedRank {
        AdmittedRank {
            logical: ordinal as usize,
            device_ordinal: ordinal,
            pci_bus_id: pci.map(str::to_string),
            arch: arch.to_string(),
        }
    }

    fn visible(n: i32, arch: &str) -> Vec<AdmittedRank> {
        (0..n)
            .map(|i| rank(i, Some(&format!("0000:{:02x}:00.0", i + 3)), arch))
            .collect()
    }

    fn mesh(
        pp: usize,
        visible: &[AdmittedRank],
        policy: EdgeTransportPolicy,
        peer: bool,
        layers: Option<&str>,
    ) -> Result<(AdmittedMesh, PpBands), String> {
        admit_qwen4_pp_mesh(
            pp,
            1,
            1,
            48,
            visible,
            "gfx1100",
            false,
            policy,
            &move |_, _| peer,
            layers,
        )
    }

    #[test]
    fn transport_variable_truth_table() {
        assert_eq!(parse_mesh_transport(None), Ok(EdgeTransportPolicy::Auto));
        assert_eq!(
            parse_mesh_transport(Some("")),
            Ok(EdgeTransportPolicy::Auto)
        );
        assert_eq!(
            parse_mesh_transport(Some("auto")),
            Ok(EdgeTransportPolicy::Auto)
        );
        assert_eq!(
            parse_mesh_transport(Some(" peer ")),
            Ok(EdgeTransportPolicy::Peer)
        );
        assert_eq!(
            parse_mesh_transport(Some("host")),
            Ok(EdgeTransportPolicy::Host)
        );
        for bad in ["Peer", "p2p", "1", "host,peer"] {
            let error = parse_mesh_transport(Some(bad)).unwrap_err();
            assert!(error.contains(MESH_TRANSPORT_ENV), "{bad}: {error}");
        }
    }

    #[test]
    fn accepted_cell_binds_output_card_last_and_pins_the_ple_band() {
        for (pp, arch) in [(2usize, "gfx1100"), (6, "gfx1100"), (8, "gfx1100")] {
            let vis = visible(pp as i32, arch);
            let (mesh, bands) = mesh(pp, &vis, EdgeTransportPolicy::Auto, false, None).unwrap();
            assert_eq!(mesh.family, "qwen4");
            assert_eq!(mesh.numerical_policy, QWEN4_PP_NUMERICAL_POLICY);
            // Logical stage s < S-1 runs on visible device s+1; the last stage
            // (the output band) on visible device 0, PCI identity attached.
            for stage in 0..pp {
                let expected = (stage as i32 + 1) % pp as i32;
                assert_eq!(mesh.ranks[stage].logical, stage);
                assert_eq!(mesh.ranks[stage].device_ordinal, expected);
                assert_eq!(
                    mesh.ranks[stage].pci_bus_id.as_deref(),
                    Some(format!("0000:{:02x}:00.0", expected + 3).as_str())
                );
            }
            assert_eq!(mesh.ranks[pp - 1].device_ordinal, 0);
            assert_eq!(mesh.edges.len(), pp - 1);
            assert!(bands.explicit_counts.is_none());
            let placement = &bands.placement;
            assert_eq!(placement.stages(), pp);
            assert!(placement.band(0).len() >= 2);
            let counts: Vec<usize> = (0..pp).map(|s| placement.band(s).len()).collect();
            assert_eq!(counts.iter().sum::<usize>(), 48);
        }
        // gfx1201 is the other admitted architecture.
        let vis = visible(4, "gfx1201");
        assert!(admit_qwen4_pp_mesh(
            4,
            1,
            1,
            48,
            &vis,
            "gfx1201",
            false,
            EdgeTransportPolicy::Auto,
            &|_, _| false,
            None
        )
        .is_ok());
        // PP6 uniform split is 8 layers per stage (48 / 6).
        let vis = visible(6, "gfx1100");
        let (_, bands) = mesh(6, &vis, EdgeTransportPolicy::Auto, false, None).unwrap();
        assert_eq!(bands.placement.band(0), 0..8);
        // Only visible devices beyond the stage count are ignored.
        let vis = visible(8, "gfx1100");
        let (mesh8, _) = mesh(4, &vis, EdgeTransportPolicy::Auto, false, None).unwrap();
        assert_eq!(mesh8.ranks.len(), 4);
        assert_eq!(mesh8.ranks[3].device_ordinal, 0);
    }

    #[test]
    fn generic_counts_come_from_the_pp_layers_request() {
        let vis = visible(6, "gfx1100");
        let (_, bands) = mesh(
            6,
            &vis,
            EdgeTransportPolicy::Auto,
            false,
            Some("9,8,8,8,8,7"),
        )
        .unwrap();
        assert_eq!(bands.explicit_counts, Some(vec![9, 8, 8, 8, 8, 7]));
        assert_eq!(bands.placement.band(0), 0..9);
        assert_eq!(bands.placement.band(5), 41..48);
        let vis2 = visible(2, "gfx1100");
        let (_, bands) = mesh(2, &vis2, EdgeTransportPolicy::Auto, false, Some("2,46")).unwrap();
        assert_eq!(bands.placement.band(0), 0..2);
        // A first band under 2 layers cuts the PLE layer off the input band.
        for spec in ["1,47", "0,48"] {
            let error = mesh(2, &vis2, EdgeTransportPolicy::Auto, false, Some(spec)).unwrap_err();
            assert!(error.contains("topology.pp = 2"), "{spec}: {error}");
        }
        let error = mesh(2, &vis2, EdgeTransportPolicy::Auto, false, Some("1,47")).unwrap_err();
        assert!(error.contains("at least 2"), "{error}");
        // Count mismatches and wrong sums keep the generic verdicts.
        for spec in ["24", "24,24,0", "20,20", "a,b"] {
            let error = mesh(2, &vis2, EdgeTransportPolicy::Auto, false, Some(spec)).unwrap_err();
            assert!(
                error.contains("topology.pp = 2") && error.contains("HIPFIRE_PP_LAYERS"),
                "{spec}: {error}"
            );
        }
    }

    #[test]
    fn edge_policy_decides_each_directed_edge_with_no_fallback() {
        let vis = visible(4, "gfx1100");
        let transports = |policy, peer| {
            mesh(4, &vis, policy, peer, None)
                .map(|(mesh, _)| mesh.edges.iter().map(|e| e.transport).collect::<Vec<_>>())
        };
        // No byte-qualification evidence: auto is host-mapped.
        assert_eq!(
            transports(EdgeTransportPolicy::Auto, false).unwrap(),
            vec![EdgeTransport::HostMapped; 3]
        );
        assert_eq!(
            transports(EdgeTransportPolicy::Host, true).unwrap(),
            vec![EdgeTransport::HostMapped; 3]
        );
        // With evidence auto and peer pick peer.
        assert_eq!(
            transports(EdgeTransportPolicy::Auto, true).unwrap(),
            vec![EdgeTransport::Peer; 3]
        );
        assert_eq!(
            transports(EdgeTransportPolicy::Peer, true).unwrap(),
            vec![EdgeTransport::Peer; 3]
        );
        // A forced peer without evidence refuses, naming topology.pp and the variable.
        let error = transports(EdgeTransportPolicy::Peer, false).unwrap_err();
        assert!(
            error.contains("topology.pp = 4")
                && error.contains(MESH_TRANSPORT_ENV)
                && error.contains("peer-to-peer transfers between the pipeline GPUs"),
            "{error}"
        );
        // Edge ids are logical stage pairs.
        let (mesh, _) = mesh(4, &vis, EdgeTransportPolicy::Host, false, None).unwrap();
        let pairs: Vec<_> = mesh.edges.iter().map(|e| (e.src, e.dst)).collect();
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn every_mesh_refusal_names_topology_pp_before_side_effects() {
        let vis = visible(6, "gfx1100");
        let admit = |pp, tp, ep, v: &[AdmittedRank], arch: &str, emulated| {
            admit_qwen4_pp_mesh(
                pp,
                tp,
                ep,
                48,
                v,
                arch,
                emulated,
                EdgeTransportPolicy::Auto,
                &|_, _| false,
                None,
            )
            .map(|_| ())
        };
        assert!(admit(2, 1, 1, &vis, "gfx1100", false).is_ok());
        let mut refusals = vec![
            admit(1, 1, 1, &vis, "gfx1100", false),
            admit(9, 1, 1, &visible(9, "gfx1100"), "gfx1100", false),
            admit(2, 2, 1, &vis, "gfx1100", false),
            admit(2, 1, 2, &vis, "gfx1100", false),
            admit(2, 1, 1, &vis, "gfx1100", true),
            admit(7, 1, 1, &vis, "gfx1100", false),
        ];
        // Unknown PCI identity on any rank.
        let mut unknown = vis.clone();
        unknown[3].pci_bus_id = None;
        refusals.push(admit(6, 1, 1, &unknown, "gfx1100", false));
        // Mixed architectures; unadmitted architecture; daemon mismatch.
        let mut mixed = vis.clone();
        mixed[2].arch = "gfx1201".to_string();
        refusals.push(admit(6, 1, 1, &mixed, "gfx1100", false));
        refusals.push(admit(6, 1, 1, &visible(6, "gfx1151"), "gfx1151", false));
        refusals.push(admit(6, 1, 1, &visible(6, "gfx1030"), "gfx1030", false));
        refusals.push(admit(6, 1, 1, &vis, "gfx1201", false));
        // Duplicate PCI identity / ordinal.
        let mut dup = vis.clone();
        dup[4].pci_bus_id = dup[1].pci_bus_id.clone();
        refusals.push(admit(6, 1, 1, &dup, "gfx1100", false));
        let mut dup_ordinal = vis.clone();
        dup_ordinal[4].device_ordinal = 1;
        refusals.push(admit(6, 1, 1, &dup_ordinal, "gfx1100", false));
        for refusal in refusals {
            let error = refusal.unwrap_err();
            assert!(error.contains("topology.pp"), "{error}");
        }
        let error = admit(2, 1, 1, &vis[..1], "gfx1100", false).unwrap_err();
        assert!(error.contains("needs 2 visible devices"), "{error}");
        let mut unknown = vis.clone();
        unknown[1].pci_bus_id = None;
        let error = admit(2, 1, 1, &unknown, "gfx1100", false).unwrap_err();
        assert!(error.contains("PCI identity"), "{error}");
    }

    fn options_with(f: impl FnOnce(&mut SourceAdmissionOptions)) -> SourceAdmissionOptions {
        let mut options = SourceAdmissionOptions::default();
        f(&mut options);
        options
    }

    #[test]
    fn unsupported_load_options_are_refused_under_pp_only() {
        let cases: Vec<(&str, Option<&str>, SourceAdmissionOptions)> = vec![
            (
                "DFlash",
                Some("draft.hfq"),
                SourceAdmissionOptions::default(),
            ),
            ("DFlash", None, options_with(|o| o.spec.dflash = Some(true))),
            ("DSpark", None, options_with(|o| o.spec.dspark = Some(true))),
            (
                "n-gram drafting",
                None,
                options_with(|o| o.spec.ngram_draft = Some(true)),
            ),
            (
                "DDTree",
                None,
                options_with(|o| o.spec.ddtree_budget = Some(1)),
            ),
            ("adaptive KV", None, options_with(|o| o.kv_adaptive = true)),
            ("EAGLE", None, options_with(|o| o.eagle_drafter = true)),
            ("CASK", None, options_with(|o| o.cask = true)),
            ("PFlash", None, options_with(|o| o.pflash = true)),
            (
                "experts-per-token",
                None,
                options_with(|o| o.expert_count_override = true),
            ),
            (
                "compute placement",
                None,
                options_with(|o| o.non_single_compute = true),
            ),
        ];
        assert_eq!(
            qwen4_pp_option_refusal(2, None, &SourceAdmissionOptions::default()),
            Ok(())
        );
        for (name, draft, options) in cases {
            let error = qwen4_pp_option_refusal(2, draft, &options).unwrap_err();
            assert!(
                error.contains("topology.pp = 2") && error.contains(name),
                "{name}: {error}"
            );
            // Single and EP never see the PP refusal.
            assert_eq!(
                qwen4_pp_option_refusal(1, draft, &options),
                Ok(()),
                "{name}"
            );
        }
        // Explicit `off` values are accepted.
        let off = options_with(|o| {
            o.spec.mtp = Some(false);
            o.spec.dflash = Some(false);
            o.spec.dspark = Some(false);
            o.spec.ngram_draft = Some(false);
        });
        assert_eq!(qwen4_pp_option_refusal(8, None, &off), Ok(()));
    }

    #[test]
    fn qwen4_spill_policy_after_the_floor_gate_is_unchanged_for_flash_next() {
        let with = |mtp| SpecLoadCfg {
            mtp,
            ..SpecLoadCfg::default()
        };
        // Fully resident: the head stays attached unless explicitly off.
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(
                qwen4_mtp_with_host_mapped_experts(with(None), arch, 0),
                "{arch}"
            );
            assert!(
                qwen4_mtp_with_host_mapped_experts(with(Some(true)), arch, 0),
                "{arch}"
            );
            assert!(
                !qwen4_mtp_with_host_mapped_experts(with(Some(false)), arch, 0),
                "{arch}"
            );
        }
        // Host-mapped experts: `auto` keeps the head only on the admitted gfx1100/gfx1201;
        // explicit `on` attaches everywhere; `off` never.
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1201",
            24
        ));
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1100",
            24
        ));
        assert!(!qwen4_mtp_with_host_mapped_experts(
            with(None),
            "gfx1151",
            24
        ));
        assert!(qwen4_mtp_with_host_mapped_experts(
            with(Some(true)),
            "gfx1100",
            24
        ));
        assert!(!qwen4_mtp_with_host_mapped_experts(
            with(Some(false)),
            "gfx1201",
            24
        ));
    }

    #[test]
    fn qwen4_mtp_off_reason_labels_exactly_the_declined_arms() {
        for mtp in [None, Some(true), Some(false)] {
            for retained in [false, true] {
                for floor in [false, true] {
                    if qwen4_native_mtp_for_floor(mtp, retained, floor) == Ok(false) {
                        let reason = qwen4_mtp_off_reason(mtp, retained, floor);
                        assert_ne!(
                            reason, "declined by policy",
                            "unlabelled decline for ({mtp:?}, {retained}, {floor})"
                        );
                    }
                }
            }
        }
        assert_eq!(
            qwen4_mtp_off_reason(None, true, true),
            ".mq4r retained replay path"
        );
        assert_eq!(
            qwen4_mtp_off_reason(None, false, false),
            "no measured AR floor (speculation.mtp=auto)"
        );
        assert_eq!(
            qwen4_mtp_off_reason(Some(false), false, true),
            "speculation.mtp=off"
        );
    }

    #[test]
    fn qwen4_native_mtp_auto_follows_the_declared_floor_and_explicit_on_ignores_it() {
        // (mtp, retained, floor) -> verdict.
        // The Qwen4 producer's own declaration admits auto on the non-retained load.
        assert!(hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR);
        assert_eq!(
            qwen4_native_mtp_for_floor(None, false, hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR),
            Ok(true)
        );
        // A producer without the floor keeps `auto` AR, retained or not.
        for retained in [false, true] {
            assert_eq!(qwen4_native_mtp_for_floor(None, retained, false), Ok(false));
        }
        // The retained default keeps `auto` AR even with the floor.
        assert_eq!(qwen4_native_mtp_for_floor(None, true, true), Ok(false));
        // Explicit `on` ignores the floor declaration; retained still refuses it.
        for floor in [false, true] {
            assert_eq!(
                qwen4_native_mtp_for_floor(Some(true), false, floor),
                Ok(true)
            );
            assert!(qwen4_native_mtp_for_floor(Some(true), true, floor).is_err());
            assert_eq!(
                qwen4_native_mtp_for_floor(Some(false), false, floor),
                Ok(false)
            );
            assert_eq!(
                qwen4_native_mtp_for_floor(Some(false), true, floor),
                Ok(false)
            );
        }
    }

    #[test]
    fn only_the_qwen4_carrier_declares_a_measured_ar_floor() {
        // Capability is declared by the producer, defaulting false: Qwen3.5/3.6
        // and DeepSeek V4 (native MTP without a floor) must auto -> AR, and every
        // other carrier keeps the trait default.
        for carrier in crate::registry_carriers() {
            let expected = carrier.claims_arch_id(hipfire_arch_qwen4::ARCH_ID, false);
            assert_eq!(
                carrier.mtp_measured_ar_floor(),
                expected,
                "{} floor declaration",
                carrier.name()
            );
        }
        assert!(!hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR);
        assert!(!hipfire_arch_deepseek4::MTP_MEASURED_AR_FLOOR);
        for arch_id in [5u32, 6, 9] {
            let carrier = crate::carrier_for(arch_id).expect("carrier");
            assert!(!carrier.mtp_measured_ar_floor(), "arch {arch_id}");
        }
        assert!(crate::carrier_for(hipfire_arch_qwen4::ARCH_ID)
            .expect("qwen4 carrier")
            .mtp_measured_ar_floor());
    }

    #[test]
    fn native_mtp_under_pp_is_explicit_only_and_ar_otherwise() {
        let with = |mtp| SpecLoadCfg {
            mtp,
            ..SpecLoadCfg::default()
        };
        for pp in [2usize, 6, 8] {
            for arch in ["gfx1100", "gfx1201"] {
                // `auto` and `off` keep the explicit AR default.
                assert_eq!(
                    qwen4_native_mtp(with(None), arch, "m/fn.mq4", pp, 1, 1),
                    Ok(false)
                );
                assert_eq!(
                    qwen4_native_mtp(with(Some(false)), arch, "m/fn.mq4", pp, 1, 1),
                    Ok(false)
                );
                // An explicit request attaches the output band's head, also
                // for a `.mq4r` name (no retained route exists under PP).
                for path in ["m/fn.mq4", "m/fn.mq4r"] {
                    assert_eq!(
                        qwen4_native_mtp(with(Some(true)), arch, path, pp, 1, 1),
                        Ok(true),
                        "{arch} pp {pp} {path}"
                    );
                }
            }
        }
        // Single is unchanged.
        assert_eq!(
            qwen4_native_mtp(with(None), "gfx1151", "m/qwen3.8-flash-next.mq4", 1, 1, 1),
            Ok(true)
        );
        // The request boundary admits MTP on a PP mesh and still refuses the
        // meshes it never admitted.
        for pp in 2..=8usize {
            let mesh = hipfire_arch_qwen4::EffectiveMesh::new(pp, 1, 1);
            assert!(qwen4_request_admission(mesh, InputModality::Text, false).is_ok());
            assert!(qwen4_request_admission(mesh, InputModality::Text, true).is_ok());
        }
        for (pp, tp, ep) in [(2, 2, 1), (2, 1, 2), (9, 1, 1)] {
            let mesh = hipfire_arch_qwen4::EffectiveMesh::new(pp, tp, ep);
            assert!(qwen4_request_admission(mesh, InputModality::Text, false).is_err());
            assert!(qwen4_request_admission(mesh, InputModality::Text, true).is_err());
        }
    }

    #[test]
    fn pp_mtp_ngram_refusal_truth_table() {
        for pp in [1usize, 2, 8] {
            for native_mtp in [false, true] {
                for explicit_on in [false, true] {
                    let result = qwen4_pp_mtp_ngram_refusal(pp, native_mtp, explicit_on);
                    if pp > 1 && native_mtp && explicit_on {
                        // Only an explicit `on` next to native MTP on a mesh refuses.
                        let error = result.unwrap_err();
                        assert!(
                            error.contains(&format!("topology.pp = {pp}"))
                                && error.contains("n-gram"),
                            "pp {pp}: {error}"
                        );
                    } else {
                        // Single, AR (native MTP off) and `auto`/`off` are untouched.
                        assert_eq!(
                            result,
                            Ok(()),
                            "pp {pp} native {native_mtp} explicit {explicit_on}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_other_speculator_stays_refused_next_to_pp_native_mtp() {
        // Native MTP alone is admitted under PP; each other speculation
        // mechanism still refuses, also when native MTP is requested too.
        let mtp = options_with(|o| o.spec.mtp = Some(true));
        assert_eq!(qwen4_pp_option_refusal(2, None, &mtp), Ok(()));
        let refused: Vec<(&str, Option<&str>, SourceAdmissionOptions)> = vec![
            ("DFlash", Some("draft.hfq"), mtp),
            (
                "DFlash",
                None,
                options_with(|o| {
                    o.spec.mtp = Some(true);
                    o.spec.dflash = Some(true);
                }),
            ),
            (
                "DSpark",
                None,
                options_with(|o| {
                    o.spec.mtp = Some(true);
                    o.spec.dspark = Some(true);
                }),
            ),
            (
                "n-gram drafting",
                None,
                options_with(|o| {
                    o.spec.mtp = Some(true);
                    o.spec.ngram_draft = Some(true);
                }),
            ),
            (
                "DDTree",
                None,
                options_with(|o| {
                    o.spec.mtp = Some(true);
                    o.spec.ddtree_budget = Some(1);
                }),
            ),
        ];
        for (name, draft, options) in refused {
            let error = qwen4_pp_option_refusal(4, draft, &options).unwrap_err();
            assert!(
                error.contains("topology.pp = 4") && error.contains(name),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn environment_requests_the_pp_route_cannot_run_are_refused() {
        let none = |_: &str| -> Option<String> { None };
        let ok = |env: &dyn Fn(&str) -> Option<String>, npu, disk| {
            qwen4_pp_env_refusal(2, env, npu, disk)
        };
        assert_eq!(ok(&none, NpuTrunkMode::Off, 0), Ok(()));
        assert_eq!(ok(&none, NpuTrunkMode::Auto, 0), Ok(()));
        let set = |name: &'static str, value: &'static str| {
            move |query: &str| (query == name).then(|| value.to_string())
        };
        let refused: Vec<(&str, Result<(), String>)> = vec![
            (
                "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS",
                ok(
                    &set("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "auto"),
                    NpuTrunkMode::Off,
                    0,
                ),
            ),
            (
                "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS",
                ok(
                    &set("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "0"),
                    NpuTrunkMode::Off,
                    0,
                ),
            ),
            (
                "HIPFIRE_QWEN4_EXPERT_HOTSET",
                ok(
                    &set("HIPFIRE_QWEN4_EXPERT_HOTSET", "/tmp/x"),
                    NpuTrunkMode::Off,
                    0,
                ),
            ),
            (
                hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV,
                ok(
                    &set(hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV, "1"),
                    NpuTrunkMode::Off,
                    0,
                ),
            ),
            (
                "HIPFIRE_QWEN4_TURN_SNAPSHOTS",
                ok(
                    &set("HIPFIRE_QWEN4_TURN_SNAPSHOTS", "1"),
                    NpuTrunkMode::Off,
                    0,
                ),
            ),
            ("session_cache_disk_bytes", ok(&none, NpuTrunkMode::Off, 1)),
            ("fusions.npu_trunk = on", ok(&none, NpuTrunkMode::On, 0)),
            (
                "fusions.npu_trunk = emulate",
                ok(&none, NpuTrunkMode::Emulate, 0),
            ),
        ];
        for (name, result) in refused {
            let error = result.unwrap_err();
            assert!(
                error.contains("topology.pp = 2") && error.contains(name),
                "{name}: {error}"
            );
        }
        // Explicit off values are accepted (hotset 0, stage 0, snapshots 0).
        for (name, value) in [
            ("HIPFIRE_QWEN4_EXPERT_HOTSET", "0"),
            (hipfire_arch_qwen4::gpu_forward::QWEN4_EXPERT_STAGE_ENV, "0"),
            ("HIPFIRE_QWEN4_TURN_SNAPSHOTS", "0"),
        ] {
            assert_eq!(
                ok(&set(name, value), NpuTrunkMode::Off, 0),
                Ok(()),
                "{name}"
            );
        }
        // Single is never refused.
        assert_eq!(
            qwen4_pp_env_refusal(
                1,
                &set("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "auto"),
                NpuTrunkMode::On,
                9
            ),
            Ok(())
        );
    }

    mod source_order {
        use super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        fn probe(name: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-qwen4-pp-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            write_hfqm_package_mem(
                &path,
                QWEN4_ARCH_ID,
                "{}",
                &[HfqMemTensor {
                    name: "model.embed_tokens.weight".into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                }],
            )
            .unwrap();
            path
        }

        fn admit(
            path: &std::path::Path,
            tp: usize,
            pp: usize,
            options: SourceAdmissionOptions,
        ) -> Result<(), String> {
            admit_source_with_options(
                path.to_str().unwrap(),
                tp,
                pp,
                KvBackendRequest::Automatic,
                None,
                "gfx1100",
                None,
                "auto",
                None,
                2048,
                KvBackendHints::without_device(),
                options,
            )
            .map(|_| ())
        }

        #[test]
        fn pp_refusals_fire_before_the_source_is_inspected() {
            let path = probe("order");
            let defaults = SpecLoadCfg {
                dflash: Some(false),
                dspark: Some(false),
                ngram_draft: Some(false),
                ddtree_budget: Some(0),
                ddtree_topk: Some(4),
                mtp: Some(false),
                ..SpecLoadCfg::default()
            };
            // Admitted options reach the (invalid) source's own validation.
            let error = admit(
                &path,
                1,
                2,
                SourceAdmissionOptions {
                    spec: defaults,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(error.contains("config admission failed"), "{error}");
            // An explicit native MTP request is admitted under PP too: it
            // reaches the source's own validation instead of a PP refusal.
            let error = admit(
                &path,
                1,
                2,
                SourceAdmissionOptions {
                    spec: SpecLoadCfg {
                        mtp: Some(true),
                        ..defaults
                    },
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(
                error.contains("config admission failed") && !error.contains("topology.pp"),
                "{error}"
            );
            // Every refusal precedes it and names topology.pp.
            for options in [
                SourceAdmissionOptions {
                    spec: SpecLoadCfg {
                        dflash: Some(true),
                        mtp: Some(true),
                        ..defaults
                    },
                    ..Default::default()
                },
                SourceAdmissionOptions {
                    spec: SpecLoadCfg {
                        dspark: Some(true),
                        mtp: Some(true),
                        ..defaults
                    },
                    ..Default::default()
                },
                SourceAdmissionOptions {
                    spec: defaults,
                    cask: true,
                    ..Default::default()
                },
                SourceAdmissionOptions {
                    spec: defaults,
                    pflash: true,
                    ..Default::default()
                },
                SourceAdmissionOptions {
                    spec: SpecLoadCfg {
                        ngram_draft: Some(true),
                        ..defaults
                    },
                    ..Default::default()
                },
            ] {
                let error = admit(&path, 1, 2, options).unwrap_err();
                assert!(
                    error.contains("topology.pp = 2") && !error.contains("config admission failed"),
                    "{error}"
                );
            }
            // tp with pp is refused at the request boundary, before the source.
            let error = admit(
                &path,
                2,
                2,
                SourceAdmissionOptions {
                    spec: defaults,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(
                error.contains("topology.pp from 2 to 8")
                    && !error.contains("config admission failed"),
                "{error}"
            );
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }
    }
}
