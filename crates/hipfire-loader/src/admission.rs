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
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::ModelSource;

/// The one effective topology admitted for a load. `tp>1` (expert-parallel) and
/// `pp>1` (pipeline-parallel) are mutually exclusive; both default to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveTopology {
    Single,
    Pipeline(usize),
    Expert(usize),
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
    /// The resolved carrier (single/pp path). `None` for expert-parallel, which
    /// dispatches on `arch_id` directly rather than through the registry.
    pub carrier: Option<&'static dyn Carrier>,
    /// Validated vision-tower sidecar (`params.vision` / `HIPFIRE_VISION_SIDECAR`).
    /// `Some` only when the sidecar opened, carries arch_id 5|6, and holds the
    /// tower probe tensor; the single/pp route threads it into `LoadCtx`.
    /// `None` = trunk-only (or explicit opt-out via empty string).
    pub vision_path: Option<std::path::PathBuf>,
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
        _ => (false, false),
    };
    classify_vision(arch_id, has_tensor, has_config)
}

/// Tower probe tensor shared by the trunk and the vision sidecar.
const VISION_PROBE_TENSOR: &str = "model.visual.patch_embed.proj.weight";

/// Validate the optional vision-tower sidecar and return its resolved path.
/// Fail-closed: an unopenable, non-5|6, or tower-less sidecar refuses with a
/// message naming the remedy. The sidecar opens read-only as a SEPARATE
/// `HfqFile` (never `attach_overlay` — REAP rejects additive tensor names,
/// hfq.rs:391-427). Empty string counts as unset (explicit opt-out, same
/// semantics as `HIPFIRE_DFLASH_DRAFT`).
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
    if !matches!(arch_id, 5 | 6) {
        return Err(format!(
            "vision sidecar '{path}' requested for arch_id={arch_id}: vision sidecars \
             only serve Qwen3.5-VL trunks (arch_id 5|6)"
        ));
    }
    let sidecar = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("vision sidecar '{path}': open failed: {e}"))?;
    if !matches!(sidecar.arch_id, 5 | 6) {
        return Err(format!(
            "vision sidecar '{path}' has arch_id={} (expected 5|6): pack the tower \
             with `hipfire-quantize <hf-dir> --include-vision --include-prefix model.visual.`",
            sidecar.arch_id
        ));
    }
    if sidecar.tensor_data(VISION_PROBE_TENSOR).is_none() {
        return Err(format!(
            "vision sidecar '{path}' carries no vision tower tensor \
             '{VISION_PROBE_TENSOR}': pack the tower with `hipfire-quantize <hf-dir> \
             --include-vision --include-prefix model.visual.`"
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
    if !crate::dflash_lm_head_quant_supported(lm_qt, gpu_arch) {
        let qt_desc = match lm_qt {
            Some(qt) => format!("quant_type={qt}"),
            None => "no lm_head/embed_tokens tensor found".to_string(),
        };
        return Err(format!(
            "DFlash draft requested but target lm_head {qt_desc} is not supported \
             on gfx11+gfx12 WMMA ({gpu_arch})."
        ));
    }
    Ok(())
}

/// Expert-parallel VMM refusal, mirroring `load_model_ep_with_kv_mode`'s
/// per-arch dispatch: VMM is single-device, so the EP arches whose loaders
/// have no VMM path (Qwen3.5 5|6, MiniMax 10) refuse it. DeepSeek V4 (9) is
/// the one EP arch that serves vmm by design and stays vmm-capable here.
fn ep_vmm_refusal(arch_id: u32, kv_backend: KvBackend) -> Option<String> {
    (kv_backend == KvBackend::Vmm && matches!(arch_id, 5 | 6 | 10))
        .then(|| format!("KV backend '{}' requires tp=1", kv_backend.as_str()))
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
            "image generation (arch 40/45) requires RDNA3/3.5 (gfx11 wave32 WMMA); \
             detected {gpu_arch}. See docs/IMAGEGEN.md §1."
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
        hipfire_config::HIP_VISIBLE_DEVICES,
        hipfire_config::ROCR_VISIBLE_DEVICES,
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
            "load refused: pp=0 is not a valid pipeline-parallel degree \
             (Gpus::init_uniform requires n_devices >= 1; multi_gpu.rs)"
                .to_string(),
        );
    }
    if let Some(layers) = n_layers {
        if pp > layers {
            return Some(format!(
                "load refused: pp={pp} exceeds n_layers={layers} for this source \
                 (Gpus::init_uniform refuses n_layers < n_devices — each device must \
                 own at least one layer; multi_gpu.rs)"
            ));
        }
    }
    if let Some(devices) = n_devices {
        if pp > devices {
            return Some(format!(
                "load refused: pp={pp} exceeds visible HIP device count={devices} \
                 (Gpus::init_uniform cannot bind that many ranks; multi_gpu.rs)"
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
            "load refused: tp=0 is not a valid parallel degree \
             (Gpus::init_ep/init_tp require ep_size/tp_size >= 1; multi_gpu.rs)"
                .to_string(),
        );
    }
    if let Some(devices) = n_devices {
        if tp > devices {
            return Some(format!(
                "load refused: tp={tp} exceeds visible HIP device count={devices} \
                 (Gpus::init_ep/init_tp need one device per rank; multi_gpu.rs)"
            ));
        }
    }
    None
}

/// Read-only source admission: open the source, classify `arch_id` + vision,
/// decide the effective topology, and refuse every unsupported/contradictory
/// combination — without touching GPU state, VMM, or any prior model.
///
/// Refusals mirror the current-master daemon/loader refusals so no
/// currently-served route changes; they simply fire before destructive work.
pub fn admit_source(
    path: &str,
    tp: usize,
    pp: usize,
    kv_backend_override: Option<&str>,
    draft_path: Option<&str>,
    gpu_arch: &str,
    vision: Option<&str>,
    head: Option<&str>,
    max_seq: usize,
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
    let is_dir = source.is_dir();
    let kv_backend: KvBackend = kv_backend_override
        .unwrap_or("contiguous")
        .parse()
        .map_err(|err| format!("{err}"))?;

    // FLUX/Klein need gfx11 wave32 WMMA (the trunk GEMM and vtk/v2 use the
    // gfx11 WMMA intrinsic hipcc rejects on gfx12). Refuse here — before the
    // topology branch and any allocation — so the prior model stays loaded.
    if let Some(refusal) = flux_arch_refusal(arch_id, gpu_arch) {
        return Err(refusal);
    }

    let (topology, carrier) = if tp > 1 {
        // Expert-parallel admission (HFQ-only). Mirrors
        // `load_model_ep_with_kv_mode`'s arch_id dispatch + per-arch VMM
        // refusal: DeepSeek V4 (9) serves vmm by design; Qwen3.5 (5|6) and
        // MiniMax (10) refuse it (single-device backend, no EP VMM path).
        if is_dir {
            return Err(
                "EP not supported for safetensors directory sources (load as a single HFQ file)"
                    .into(),
            );
        }
        if !matches!(arch_id, 5 | 6 | 9 | 10) {
            return Err(format!(
                "EP not supported for arch_id={arch_id} (expected 5|6 for Qwen3.5, 9 for DeepSeek V4 or 10 for MiniMax)"
            ));
        }
        if let Some(refusal) = ep_vmm_refusal(arch_id, kv_backend) {
            return Err(refusal);
        }
        // #666 G2: refuse an unsatisfiable rank count before any teardown or
        // GPU init (`init_ep` needs one device per rank). Runs after the
        // arch/VMM refusals above so their messages are unchanged.
        if let Some(refusal) = tp_geometry_refusal(tp, visible_device_count()) {
            return Err(refusal);
        }
        (EffectiveTopology::Expert(tp), None)
    } else {
        // Single / pipeline-parallel via the carrier registry.
        let carrier = resolve_carrier(&source)?;
        if kv_backend == KvBackend::Vmm
            && !matches!(carrier.name(), "qwen35" | "deepseek4" | "muse_glimmer")
        {
            return Err(format!(
                "KV backend 'vmm' currently supports qwen3.5, deepseek4, and Muse Glimmer only (selected carrier: {})",
                carrier.name()
            ));
        }
        if kv_backend == KvBackend::Vmm && pp > 1 {
            return Err(
                "KV backend 'vmm' is single-device and does not support pipeline parallelism (pp>1); \
                 use a different kv_cache backend or load with pp=1"
                    .to_string(),
            );
        }
        carrier.admit_topology(arch_id, is_dir, pp, kv_backend)?;
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
        (topology, Some(carrier))
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
        // Gemma 4 lowered min-context: refuse before teardown/alloc so a
        // small max_seq leaves the prior model serving. Eager stays exempt.
        if matches!(arch_id, 13 | 22) {
            let use_lowered = hipfire_arch_gemma4::gemma4_source_uses_lowered(hfq, false);
            hipfire_arch_gemma4::gemma4_context_admission(max_seq, use_lowered)?;
        }
    }
    // Head overlay (`params.head`): validated AND attached to the retained
    // base here, so the admitted source is already effective and loading
    // consumes it with no second open. Any refusal leaves the prior model
    // loaded — the overlay is in-memory only; no GPU state is touched.
    admit_head_overlay(head, &mut source, arch_id, topology)?;

    Ok(SourceAdmission {
        source,
        arch_id,
        is_dir,
        has_vision,
        topology,
        kv_backend,
        carrier,
        vision_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The EP VMM refusal is per-arch, mirroring `load_model_ep_with_kv_mode`:
    /// Qwen3.5 (5|6) and MiniMax (10) refuse `vmm`; DeepSeek V4 (9) serves it.
    /// A blanket gate here would refuse the DS4 EP + vmm load master serves.
    #[test]
    fn ep_vmm_refusal_is_per_arch() {
        assert!(ep_vmm_refusal(5, KvBackend::Vmm).is_some());
        assert!(ep_vmm_refusal(6, KvBackend::Vmm).is_some());
        assert!(ep_vmm_refusal(10, KvBackend::Vmm).is_some());
        // DeepSeek V4 is vmm-capable.
        assert!(ep_vmm_refusal(9, KvBackend::Vmm).is_none());
        // Non-vmm backends are never refused.
        assert!(ep_vmm_refusal(5, KvBackend::Contiguous).is_none());
        assert!(ep_vmm_refusal(9, KvBackend::Contiguous).is_none());
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
            err_zero.contains("init_uniform"),
            "names deep site: {err_zero}"
        );
        let err = pp_geometry_refusal(99, Some(64), Some(4)).expect("pp=99 refuses");
        assert!(err.contains("pp=99"), "reason: {err}");
        assert!(err.contains("n_layers=64"), "reason: {err}");
        assert!(err.contains("init_uniform"), "names deep site: {err}");
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
        assert!(err_zero.contains("init_ep"), "names deep site: {err_zero}");
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
    /// The trunk is a tower-less arch-5 text pack; the sidecar carries just
    /// the probe tensor. Admission is read-only — no GPU needed.
    mod vision_sidecar {
        use super::super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

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
                    name: super::super::VISION_PROBE_TENSOR.into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                });
            }
            write_hfqm_package_mem(&path, arch_id, "{}", &tensors).unwrap();
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
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("wrong-arch sidecar must refuse");
            assert!(err.contains("arch_id=9"), "names the sidecar arch: {err}");
            cleanup(&trunk);
            cleanup(&sidecar);
        }
    }
    mod head_overlay {
        use super::super::*;
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
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("full model as head must refuse");
            assert!(err.contains("expected only"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }
    }
}
