// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Arch-agnostic loader contract. Concrete `ModelState`/`LoadedModel`
//! and the registry live top-of-DAG in `hipfire-loader`; this module
//! holds only what the arch crates need to implement a carrier.

use crate::hfq::HfqFile;
use crate::kv_backend::KvBackend;
use crate::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

/// A model on disk, before we know its arch. Carries either a parsed
/// HFQ header or a directory (safetensors/ParoQuant — probed later).
pub enum ModelSource {
    Hfq(HfqFile),
    Dir(SafetensorsSource),
}

impl ModelSource {
    /// Open a model from either an HFQ file or a safetensors directory
    /// based on whether `path` is a file or directory.
    pub fn from_path(path: &str) -> Result<Self, String> {
        if Path::new(path).is_dir() {
            Ok(ModelSource::Dir(
                SafetensorsSource::open(Path::new(path)).map_err(|e| format!("{e:?}"))?,
            ))
        } else {
            Ok(ModelSource::Hfq(
                HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?,
            ))
        }
    }

    /// The HFQ or safetensors arch_id.
    pub fn arch_id(&self) -> Option<u32> {
        match self {
            ModelSource::Hfq(h) => Some(h.arch_id),
            ModelSource::Dir(s) => Some(s.arch_id()),
        }
    }

    /// Whether this source is a safetensors directory (vs an HFQ file).
    /// Carriers route on this because the HFQ and `derive_arch_id`
    /// namespaces are distinct (e.g. Qwen2 is HFQ id 7 but dir id 1).
    pub fn is_dir(&self) -> bool {
        matches!(self, ModelSource::Dir(_))
    }

    /// Human-readable description for logging.
    pub fn describe(&self) -> String {
        match self {
            ModelSource::Hfq(h) => format!("HFQ arch_id={}", h.arch_id),
            ModelSource::Dir(s) => format!("safetensors-dir arch_id={}", s.arch_id()),
        }
    }
}

/// Owned header tensor metadata for kernel route planning
/// (`rdna_compute::kernel_registry::KernelTensorMeta` borrows from it).
/// Header-only: no tensor payload is read.
pub struct HeaderTensorMeta {
    pub name: String,
    pub dtype: rdna_compute::DType,
    pub shape: Vec<usize>,
}

/// Storage dtype for an HFQ `quant_type` tag. Tags without a planner-relevant
/// mapping become `Raw`, which no closed route fingerprint accepts.
fn kernel_dtype_for_quant_type(quant_type: u8) -> rdna_compute::DType {
    use rdna_compute::DType;
    match quant_type {
        1 => DType::F16,
        2 => DType::F32,
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        16 => DType::BF16,
        44 => DType::MQ4G256V2,
        47 => DType::MQ6G256V2,
        48 => DType::MQ5G256V2,
        49 => DType::MQ3G256V2,
        50 => DType::MQ2G256V2,
        53 => DType::MQ4G128V2,
        55 => DType::FP8E4M3B128,
        56 => DType::FP8E4M3PT,
        _ => DType::Raw,
    }
}

/// Header tensor names, storage dtypes and shapes of `hfq`.
pub fn header_tensor_meta(hfq: &HfqFile) -> Vec<HeaderTensorMeta> {
    hfq.tensor_infos()
        .iter()
        .map(|info| HeaderTensorMeta {
            name: info.name.clone(),
            dtype: kernel_dtype_for_quant_type(info.quant_type),
            shape: info.shape.iter().map(|&d| d as usize).collect(),
        })
        .collect()
}

/// Qwen's load-time sequence decision: capacity is filled from actual free
/// VRAM after weights upload and before its VMM cache is constructed.
#[derive(Clone, Copy)]
pub struct SequenceHint {
    pub model_ctx: usize,
    pub automatic: bool,
    pub card_cap: usize,
    /// Device bytes the automatic max_seq leaves free beyond the minimum
    /// prefill reservation: the largest admitted image's activation scratch
    /// and, when placement precedes tower upload (Flash Next), tower weights.
    pub reserve_bytes: usize,
}

/// Per-load continuous-batch request, carried from admission to the carrier
/// so a load can size memory for the batch it will stage afterwards. The
/// default is off (`vmm_batch == false`): every load that does not ask keeps
/// its plan unchanged. Only Qwen4 expert-parallel loads read it; every other
/// carrier ignores it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchLoadRequest {
    /// The VMM continuous-batching route is requested (`serve_vmm_batch`).
    pub vmm_batch: bool,
    /// Requested lane count (`continuous_batch_size`); below 2 is no batch.
    pub continuous_batch_size: usize,
    /// Requested step row budget (`serve.max_batch_tokens`); the staged
    /// budget is `max(row_budget, continuous_batch_size)`.
    pub row_budget: usize,
}

impl BatchLoadRequest {
    /// Lanes to reserve for, when the request asks for a batch (VMM route and
    /// more than one lane); `None` when it is off.
    pub fn lanes(&self) -> Option<usize> {
        (self.vmm_batch && self.continuous_batch_size > 1).then_some(self.continuous_batch_size)
    }
}

/// Everything a carrier's `load` needs beyond the source itself.
pub struct LoadCtx<'a> {
    pub path: &'a str,
    pub max_seq: usize,
    pub sequence: Option<SequenceHint>,
    /// DeepSeek V4-only physical compute placement. The default is `Single`;
    /// other carriers must ignore it.
    pub deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    /// DeepSeek V4-only routed-expert fanout override. `None` preserves the
    /// checkpoint value; other carriers must ignore it.
    pub deepseek4_experts_per_token: Option<usize>,
    pub draft_path: Option<&'a str>,
    /// Shared Qwen3.5-VL vision-tower sidecar (`qwen3.8-27b-vision.hfq`,
    /// `params.vision` / `HIPFIRE_VISION_SIDECAR`). Read by `Qwen35Carrier`
    /// only, and only when the trunk itself carries no tower tensor: the
    /// sidecar opens as a separate `HfqFile` (never an overlay) and its tower
    /// loads against the trunk's `vision_config_from_hfq`. `None` = trunk-only
    /// (or text-only when the trunk has no tower either).
    pub vision_path: Option<PathBuf>,
    /// Qwen MTP head sidecar (`.mtp`) resolved by the CLI (`params.mtp`):
    /// models dir, then beside the path as typed, then beside the canonical
    /// trunk. Read by `Qwen35Carrier` only, after the bundled trailer probe.
    /// `None` = look beside the canonical trunk (`<trunk>.with_extension("mtp")`).
    pub mtp_path: Option<PathBuf>,
    /// Resolved `vision.mode` (`off`/`auto`/`resident`/`on`). Gates `<stem>.vl`
    /// sibling discovery in the carrier — `vision_path` alone cannot
    /// distinguish "off" (never probe) from "auto with no explicit sidecar"
    /// (probe). `off` suppresses discovery; `auto`/`resident`/`on` allow it.
    /// `resident` uploads the whole tower to VRAM; every other mode streams it.
    pub vision_mode: String,
    pub kv_mode_override: Option<&'a str>,
    /// Authored Qwen-only K and V overrides. None preserves the selected whole-cache mode.
    pub kv_k_override: Option<&'a str>,
    pub kv_v_override: Option<&'a str>,
    /// Sampled once per load from HIPFIRE_QWEN_KV_DEFAULT_Q8; never per token.
    pub qwen_default_q8: bool,
    // NOTE: head overlays (`--head`) deliberately have NO LoadCtx field. They
    // validate and attach in `admit_source` before teardown, so the admitted
    // source the carrier consumes is already effective — threading a second
    // path here would reopen the file and reintroduce the pre-teardown
    // validation gap. The offline coherence example uses
    // `load_maple_from_hfq_with_head` directly.
    pub kv_backend: KvBackend,
    pub kv_adaptive_override: Option<&'a str>,
    pub state_quant_override: Option<&'a str>,
    pub cask: &'a CaskConfig,
    pub pp: usize,
    /// Admission-only explicit pipeline layer counts (one entry per rank,
    /// summing to the model layer count). Populated solely from
    /// `SourceAdmission::pp_layer_counts`; carriers must never reread the
    /// environment. `None` = carrier-default balanced placement.
    pub pp_layer_counts: Option<&'a [usize]>,
    /// Admitted PP mesh (physical rank identity and edge transports) for a
    /// carrier with a mesh-executor PP route (Qwen4 `topology.pp`). Populated
    /// solely from `SourceAdmission::mesh` when `pp > 1`; `None` otherwise.
    pub admitted_mesh: Option<&'a crate::mesh_placement::AdmittedMesh>,
    /// Admitted band placement matching `admitted_mesh` (from
    /// `SourceAdmission::pp_bands`). `None` outside a mesh-executor PP load.
    pub band_placement: Option<&'a crate::mesh_placement::BandPlacement>,
    pub spec: SpecLoadCfg,
    pub gpu: &'a mut Gpu,
    /// Gemma4 EAGLE drafter path (arch 22 `gemma4_unified_assistant`), separate
    /// from `draft_path` (Qwen DFlash) so a DFlash .hfq can never be routed
    /// into the EAGLE loader by accident. `None` = no drafter (AR-only).
    /// Only `Gemma4Carrier` reads this.
    pub gemma4_drafter_path: Option<&'a str>,
    /// Gemma4 EAGLE draft_len (verify block = draft_len + 1). Validated at
    /// load time via `gemma4_eagle_spec_len` (1..=5, default 3). Meaningful
    /// only when `gemma4_drafter_path` is `Some`.
    pub gemma4_draft_len: usize,
    /// Opt-in gfx1151 XDNA NPU spillover sidecar, verified at load time
    /// (`XdnaSidecarDescriptor::load_verified`) and admitted only when the
    /// `kernel.npu_spillover` process flag is on AND the host arch is exactly
    /// `gfx1151` (see `admit_for_arch`). `None` = GPU-only; carriers must not
    /// open `/dev/accel`, allocate NPU mirrors, or change numerics when this
    /// is `None`. Populated by the loader from the daemon `xdna` load param;
    /// projected CLI-side by `load_params` from the registry `xdna` slot.
    pub xdna: Option<hipfire_registry::XdnaSidecarDescriptor>,
    /// Continuous-batch request of this load (default off). Read by
    /// `Qwen4Carrier` for an expert-parallel load only: it reserves the AR
    /// lane store's device bytes before splitting the experts. Populated
    /// solely from `SourceAdmission::batch_request`.
    pub batch_request: BatchLoadRequest,
}

/// Per-load model-free n-gram speculator settings, resolved by the CLI through
/// the config ladder (env > flag > per-model > global) and forwarded in the
/// `load` message params. `None` fields mean "the CLI said nothing" — the loader
/// then falls back to the legacy env vars (`HIPFIRE_NGRAM_DRAFT*`) so a daemon
/// driven directly (no hipfire CLI) keeps working. Env always *wins* over these
/// when set, matching the top of the ladder.
///
/// The master `speculation` selector lives entirely CLI-side: it is lowered into
/// the per-mechanism signals (`dflash_mode`/`draft`, `mtp_mode`, and this), so
/// `build_speculator`'s first-match cascade (dflash > mtp > n-gram) naturally
/// yields the chosen mechanism without the loader needing a selector of its own.
#[derive(Clone, Copy, Default)]
pub struct SpecLoadCfg {
    /// Enable the model-free n-gram drafter for this load. `None` = unspecified.
    pub ngram_draft: Option<bool>,
    /// n-gram draft window K (`HIPFIRE_NGRAM_DRAFT_K`). `None` = loader default.
    pub ngram_k: Option<usize>,
    /// n-gram min match count (`HIPFIRE_NGRAM_MIN_COUNT`). `None` = loader default.
    pub ngram_min_count: Option<u32>,
    /// DDTree verify budget — max tree nodes (`HIPFIRE_DDTREE_BUDGET`). `None` =
    /// loader default (0 = chain-mode DFlash, no ddtree). Mirrors `ngram_k`: a
    /// CLI-forwarded draft tuning knob, env-wins-else-param in the loader.
    pub ddtree_budget: Option<usize>,
    /// DDTree per-position top-K width (`HIPFIRE_DDTREE_TOPK`). `None` = default.
    pub ddtree_topk: Option<usize>,
    /// DSpark draft module (deepseek4 `-dspark` sidecar) enable, lowered from the
    /// `speculation` selector: `Some(true)` = `dspark` mode (load + force),
    /// `Some(false)` = another mechanism selected (skip load + build),
    /// `None` = `auto` (load if the sidecar exists, prefer over in-trunk MTP).
    /// Replaces the old `HIPFIRE_DEEPSEEK4_DSPARK` / `HIPFIRE_DEEPSEEK4_LOAD_DSPARK`
    /// env gates — both fold into this one mode.
    pub dspark: Option<bool>,
    /// DSpark confidence-truncation threshold (`--dspark-conf-threshold`),
    /// forwarded ONLY when the user set it. `None` = use the per-arch carrier
    /// default (qwen3 0.1, deepseek4 0.3) — the CLI no longer imposes a global
    /// default that would shadow those. Env `HIPFIRE_{QWEN3,DEEPSEEK4}_DSPARK_CONF_THRESHOLD`
    /// still wins over this in the builder.
    pub dspark_conf_threshold: Option<f32>,
    /// Qwen MTP (NextN) enable, lowered from the `speculation` selector:
    /// `Some(true)` = `mtp` mode (load + force, whatever the producer
    /// declares), `Some(false)` = another mechanism selected (skip load +
    /// build), `None` = `auto`: the head loads only for a producer that
    /// declares a measured AR floor (`hipfire_config::mtp_mode_admits`;
    /// native Qwen4 only — Qwen3.5/3.6 and DeepSeek V4 keep plain AR) and
    /// only if a bundled trailer or `.mtp` sidecar exists.
    pub mtp: Option<bool>,
    /// MTP draft window K. `Some` = explicit (load param, `HIPFIRE_MTP_K`, or
    /// `speculation.mtp_k`); `None` = `auto`: each producer applies its own
    /// default (`hipfire_config::mtp_k_default` over the GPU's row-exact verify
    /// rows for native Qwen4 MTP, else `hipfire_config::MTP_K_DEFAULT`).
    pub mtp_k: Option<usize>,
    /// Qwen DFlash draft enable, lowered from `dflash_mode`: `Some(true)` =
    /// `on` (fail the load when the draft is missing or unloadable),
    /// `Some(false)` = `off` (skip), `None` = `auto` (load when present,
    /// log-and-AR fallback otherwise).
    pub dflash: Option<bool>,
    /// DFlash adaptive verify-block (`dflash_adaptive_b`, default false).
    /// `None` = loader default (off: fixed block). Env `HIPFIRE_DFLASH_ADAPTIVE_B=0`
    /// forces the fixed full block at build, mirroring
    /// `HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0`. Mutually exclusive with the
    /// retained-PM4 verify route (which needs the fixed B=16 shape).
    pub dflash_adaptive_b: Option<bool>,
    /// DFlash2 proposal width of this load (`speculation.dflash_block`),
    /// already resolved per serving configuration by
    /// [`crate::dflash::DflashBlockPolicy::resolve`]. `None` (no daemon
    /// resolution, e.g. a direct loader caller) =
    /// [`crate::dflash::DFLASH_BLOCK_UNBATCHED`]. Legacy DFlash drafts keep
    /// their declared width.
    pub dflash_block: Option<usize>,
    /// `speculation.dflash_block` was `auto` (the default): a producer with
    /// its own measured default width may use it instead of `dflash_block`
    /// (Qwen4 DFlash2). `false` = an explicit width, or no daemon resolution.
    pub dflash_block_auto: bool,
}

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
/// CASK/TriAttention params forwarded by the CLI at load time.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    pub cask_m_folding: bool,
    /// One-way adaptive-KV -> plain TriAttention handoff position. Zero keeps
    /// the legacy mutual exclusion between adaptive KV and eviction.
    pub handoff_tokens: usize,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}

impl CaskConfig {
    /// Resolve the bounded physical KV window used by CASK/TriAttention.
    ///
    /// Keep this next to the shared load contract so the architecture carrier
    /// allocates KV with the exact same cap later used to size the eviction
    /// context. A mismatch is especially dangerous for VMM: the virtual reserve
    /// and mapped-prefix guards must agree with the eviction trigger before the
    /// first device allocation.
    pub fn physical_cap(&self, max_seq: usize) -> Result<usize, String> {
        let override_cap = hipfire_config::developer_var("HIPFIRE_KV_PHYSICAL_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        self.physical_cap_with_override(max_seq, override_cap)
    }

    /// Pure resolution seam used by CPU tests. An explicit override is clamped
    /// to the safe CASK window, preserving the historical environment-variable
    /// behavior without mutating process-global state in tests.
    pub fn physical_cap_with_override(
        &self,
        max_seq: usize,
        override_cap: Option<usize>,
    ) -> Result<usize, String> {
        if self.sidecar.is_none() {
            return Ok(max_seq);
        }
        // A staged adaptive cache reserves its K/V arenas at the adaptive
        // floor for the full advertised context. It cannot use the ordinary
        // eviction-bounded physical cap before the handoff has completed.
        if self.handoff_tokens != 0 {
            return Ok(max_seq);
        }
        let trigger = self
            .budget
            .checked_add(self.beta)
            .ok_or_else(|| "CASK budget + beta overflowed".to_string())?;
        let floor = trigger
            .checked_add(4)
            .ok_or_else(|| "CASK physical-cap safety margin overflowed".to_string())?;
        if floor > max_seq {
            return Err(format!(
                "CASK requires max_seq >= budget + beta + 4 (got max_seq={max_seq}, budget={}, beta={})",
                self.budget, self.beta
            ));
        }
        let derived = trigger.saturating_add(256).min(max_seq);
        Ok(override_cap.unwrap_or(derived).clamp(floor, max_seq))
    }
}

#[cfg(test)]
mod cask_config_tests {
    use super::CaskConfig;

    fn enabled() -> CaskConfig {
        CaskConfig {
            sidecar: Some("centers.bin".into()),
            budget: 4096,
            beta: 128,
            ..CaskConfig::default()
        }
    }

    #[test]
    fn disabled_cask_keeps_full_physical_cap() {
        assert_eq!(
            CaskConfig::default()
                .physical_cap_with_override(32_768, Some(1024))
                .unwrap(),
            32_768
        );
    }

    #[test]
    fn enabled_cask_derives_and_clamps_physical_cap() {
        let cask = enabled();
        assert_eq!(cask.physical_cap_with_override(32_768, None).unwrap(), 4480);
        assert_eq!(
            cask.physical_cap_with_override(32_768, Some(1)).unwrap(),
            4228
        );
        assert_eq!(
            cask.physical_cap_with_override(5000, Some(99_999)).unwrap(),
            5000
        );
    }

    #[test]
    fn enabled_cask_rejects_an_impossible_window() {
        let err = enabled()
            .physical_cap_with_override(4200, None)
            .unwrap_err();
        assert!(err.contains("budget + beta + 4"), "{err}");
    }

    #[test]
    fn staged_handoff_keeps_full_adaptive_physical_cap() {
        let mut cask = enabled();
        cask.handoff_tokens = 8192;
        assert_eq!(
            cask.physical_cap_with_override(32_768, Some(1024)).unwrap(),
            32_768,
            "adaptive floor reservation must not be clamped to the eviction window"
        );
    }
}

#[cfg(test)]
mod ep_lane_reservation_request_tests {
    use super::BatchLoadRequest;

    #[test]
    fn ep_lane_reservation_request_is_off_by_default_and_for_one_lane() {
        assert_eq!(BatchLoadRequest::default().lanes(), None);
        let one = BatchLoadRequest {
            vmm_batch: true,
            continuous_batch_size: 1,
            row_budget: 4096,
        };
        assert_eq!(one.lanes(), None);
        let no_vmm = BatchLoadRequest {
            vmm_batch: false,
            continuous_batch_size: 8,
            row_budget: 4096,
        };
        assert_eq!(no_vmm.lanes(), None);
        let zero = BatchLoadRequest {
            vmm_batch: true,
            continuous_batch_size: 0,
            row_budget: 0,
        };
        assert_eq!(zero.lanes(), None);
        let on = BatchLoadRequest {
            vmm_batch: true,
            continuous_batch_size: 4,
            row_budget: 0,
        };
        assert_eq!(on.lanes(), Some(4));
    }
}
