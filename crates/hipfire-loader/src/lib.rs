// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Top-of-DAG model loader. Owns `LoadedModel`, the carrier registry,
//! and `load_model` — the single arch-dispatch point for the daemon.

pub mod admission;
pub mod batch_staging;
pub mod bench_topology;
mod carriers;
pub use carriers::*;
pub mod decide;
/// GGUF as a native model source (CPU side): parse, name map, per-tensor
/// admission, byte extents, bpw census. Default-off (`HIPFIRE_GGUF_NATIVE=1`).
pub mod gguf_native;
mod qwen4_slab;

/// Speculative-decode build/glue (RAII slot guard now; `DflashSpeculator` +
/// `build_speculator` at Stages 1-2). Lives here at the top of the DAG where
/// both `LoadedModel` and the arch crates are in scope.
pub mod spec_build;

#[cfg(target_os = "linux")]
pub use hipfire_arch_qwen4::npu_trunk::set_diagnostic_gpu_route;

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_gemma4 as gemma4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_muse_glimmer as glimmer;
use hipfire_arch_qwen35::qwen35::{self};
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_arch_raon as raon;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::device_mesh::DimKind;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode;
use hipfire_runtime::llama;
use hipfire_runtime::llama::{KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{SpecEmit, SpecEmitCtx, SpecTargetGuard, Speculator};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use rdna_compute::Gpu;
use std::any::Any;
use std::path::Path;

/// Plain-language refusal for a model family that cannot be split across GPUs
/// in pipeline mode. `family` is the carrier name (kept as the stable prefix),
/// `pp` is the requested degree. Names the setting that fixes it.
pub(crate) fn pp_unsupported_message(family: &str, is_dir: bool, pp: usize) -> String {
    if is_dir {
        format!(
            "{family}: a safetensors model directory cannot be split across GPUs in pipeline \
             mode (topology.pp = {pp}). Set topology.pp = 1 to load it on one GPU, or convert it \
             to a packed .hfq file first"
        )
    } else {
        format!(
            "{family}: this model family cannot be split across GPUs in pipeline mode \
             (topology.pp = {pp}). Set topology.pp = 1 to load it on one GPU"
        )
    }
}

pub trait Carrier: Send + Sync {
    fn name(&self) -> &'static str;
    /// Whether this carrier claims a given `arch_id`. `is_dir` distinguishes
    /// the two namespaces: HFQ-header ids (`HfqFile::arch_id`) vs the
    /// `derive_arch_id` ids emitted for safetensors directories. Kept as a
    /// pure `(u32, bool) -> bool` fn so the registry's disjointness can be
    /// unit-tested without constructing a real `ModelSource`.
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool;
    /// Default probe delegates to [`Carrier::claims_arch_id`]; carriers only
    /// implement the pure id predicate.
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(id) if self.claims_arch_id(id, src.is_dir()))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;
    /// Read-only admission refusal for a topology/shape combination this
    /// carrier cannot serve. Runs before any allocation or collective so a
    /// refusal leaves no side effect (device-mesh G2 admission). Default:
    /// pipeline-parallel (pp>1) is unsupported; HFQ and Dir get distinct
    /// messages matching each carrier's load-time refusal. Qwen3.5 overrides —
    /// it is the only carrier with a pp>1 path (`load_qwen35_pp`).
    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if pp > 1 {
            return Err(pp_unsupported_message(self.name(), is_dir, pp));
        }
        Ok(())
    }

    /// Read-only, per-carrier request-option refusal before GPU allocation.
    /// The default leaves other architectures' option behavior unchanged.
    fn admit_options(
        &self,
        _draft_path: Option<&str>,
        _options: admission::SourceAdmissionOptions,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Declared capabilities for this arch. Default is the conservative
    /// “no capability” set — carriers override to declare what they support.
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps::default()
    }

    /// Per-arch sampling defaults (`temp`, `top_p`, `repeat_penalty`).
    /// Default is the `else` arm of the daemon ladder (`0.3/0.8/1.0`).
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::default()
    }

    /// Grammar `Config` for constrained tool-call decoding.
    ///
    /// The default resolves the two `HIPFIRE_QWEN35_NGRAM_*` operator
    /// tunables and applies to **every** arch, because that is what the
    /// daemon's generic AR path did when it called
    /// `hipfire_arch_qwen35::grammar_config::resolve_qwen35_grammar_config()`
    /// unconditionally. The variable names are historically qwen-scoped; the
    /// behaviour never was. Preserved verbatim rather than "fixed", since
    /// narrowing it to Qwen would silently change decoding for anyone who
    /// sets those variables against another model.
    ///
    /// Resolution lives in `hipfire-arch-qwen35` (its `spec_emit` is the other
    /// caller) and is *called* here rather than duplicated: the loader already
    /// depends on every arch crate, so this is a downward call, and it leaves
    /// exactly one definition of the env contract in the tree.
    fn grammar_config(&self) -> saddle_core::grammar::json::Config {
        hipfire_arch_qwen35::grammar_config::resolve_qwen35_grammar_config()
    }

    /// Borrow this model's spec-decode target out of `state`, arch-erased as a
    /// [`SpecTargetGuard`]. This is the daemon's single dispatch for the
    /// spec-decode path — it then only ever sees `&mut dyn SpecTarget`, never an
    /// arch type. Default (AR-only carriers): `Err` WITHOUT touching `state` —
    /// only an override may `state.take()`.
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        Err(format!(
            "{}: spec-decode target unsupported; load without speculation: --spec off \
             (speculation.mode = \"off\")",
            self.name()
        ))
    }

    /// Construct this model's per-token spec-decode emitter from the
    /// model-independent [`SpecEmitCtx`]. The arch's emitter extracts its own
    /// grammar schema from `ctx.tools` (raw JSON) internally. Default: `Err`
    /// (arch has no spec emitter).
    fn make_spec_emitter<'a>(
        &self,
        _ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Err(format!(
            "{}: spec emitter unsupported; load without speculation: --spec off \
             (speculation.mode = \"off\")",
            self.name()
        ))
    }

    // ── GenDispatch (wave2) — arch-erased generation/bench hooks ──────────
    // These are ADDITIVE ONLY; CarrierPolicy is concurrently adding caps() and
    // sampling_defaults() in the same trait. Do not restructure.

    /// Bench-prefill dispatch for `bench_prefill` (daemon.rs:15793-16008).
    /// The ~215-line `if m.arch_id == 5 { ... } else if ...` selecting a
    /// per-arch prefill path. Each carrier implements its arch's body
    /// verbatim; the daemon resolves `carrier_for(arch_id)` and calls this
    /// once. `None` = not handled (caller falls through), `Some(run_ok)`
    /// = handled. `prefill_err` is set only by the Glimmer path, mirroring
    /// the daemon's mutable capture.
    fn bench_prefill(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        None
    }

    /// Bench-decode prime for the generic Qwen/Glimmer path
    /// (daemon.rs:16238 `prime_error` if arch_id==14 else Qwen). Return
    /// `Some(prime_error)` if handled, `None` if not this carrier's arch.
    /// The outer `Option` is dispatch; the inner `Option<String>` is the
    /// prime error (None = prime succeeded).
    fn bench_decode_prime(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _synthetic: &[u32],
    ) -> Option<Option<String>> {
        None
    }

    /// Bench-decode run for the generic Qwen/Glimmer path
    /// (daemon.rs:16325 `run_ok` if arch_id==14 else Qwen). Return
    /// `Some((run_ok, decode_err))` if handled, `None` otherwise.
    /// `decode_err` captures the first failing iteration's diagnostic for
    /// Glimmer.
    fn bench_decode_run(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _context: usize,
        _iterations: usize,
        _decode_err: &mut Option<String>,
    ) -> Option<bool> {
        None
    }

    /// Spawn this arch's multi-slot engine, arch-erased as a
    /// [`hipfire_runtime::serve::SlotEngineHandle`]. This is the daemon's
    /// single slot-mode dispatch — it resolves `carrier_for(arch_id)` and
    /// calls this rather than naming `hipfire_arch_qwen35::serve_engine`.
    /// `loader` depends on the arch crates, so the concrete engine can be
    /// boxed here without a loader->arch->loader cycle.
    ///
    /// Default: `Err` (arch has no multi-slot engine). Only the qwen35
    /// carrier overrides today (arch_id 5|6).
    fn spawn_slot_engine(
        &self,
        _cfg: hipfire_runtime::serve::SlotEngineConfig,
    ) -> Result<Box<dyn hipfire_runtime::serve::SlotEngineHandle>, String> {
        Err(format!("{}: no multi-slot engine", self.name()))
    }

    /// Slot-mode admission preflight for one of this carrier's checkpoints
    /// that the daemon's own (qwen35-HFQ-shaped) preflight cannot open — arch
    /// 17's encoder reads its geometry and tokenizer from HF/dir metadata, and
    /// its checkpoint may be a safetensors directory rather than a container.
    ///
    /// `None` = not this carrier's model, so the daemon falls through to its
    /// own preflight; `Some(Err)` = ours and refused. Called before any GPU
    /// work, on the same path that already refuses a bad load without
    /// disturbing the resident model.
    ///
    /// Default: `None` — a carrier whose checkpoints are all qwen35-shaped
    /// HFQ leaves the daemon's preflight to do its job.
    fn preflight_slot_model(
        &self,
        _path: &std::path::Path,
    ) -> Option<Result<hipfire_runtime::serve::SlotModelPreflight, String>> {
        None
    }

    /// Producer capability: this arch's native MTP drafter ships a measured
    /// autoregressive floor, so `speculation.mtp = auto` may attach it
    /// (`hipfire_config::mtp_mode_admits`). Default `false`: an `auto` MTP is
    /// declined, the head is not loaded, and the model keeps plain AR; only an
    /// explicit `on` arms it. A carrier overrides with its producer crate's
    /// `MTP_MEASURED_AR_FLOOR` — the declaration lives with the drafter, never
    /// in a central arch table.
    fn mtp_measured_ar_floor(&self) -> bool {
        false
    }

    // ── decide (Jev-compatible logit readout) ──────────────────────────────
    /// Whether this arch implements the three decide hooks below.
    fn decide_supported(&self) -> bool {
        false
    }
    /// Prefill `tokens` at `start_pos` and set `m.seq_pos = start_pos +
    /// tokens.len()`. With `want_logits` the last position's full-vocab
    /// logits are downloaded and returned; without it (the decide runner's
    /// shared-prefix prefill, whose logits are never read) the device→host
    /// copy is skipped and an empty `Vec` is returned.
    fn decide_prefill_logits(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _tokens: &[u32],
        _start_pos: usize,
        _want_logits: bool,
    ) -> Option<Result<Vec<f32>, String>> {
        None
    }
    /// Snapshot `seq_pos` + recurrent state.
    fn decide_save(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        None
    }
    /// Restore a snapshot taken by `decide_save` on the same model.
    fn decide_restore(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        None
    }
    /// KV compaction offset (`KvCache::compact_offset`) of the model's
    /// positional KV cache: 0 means uncompacted, so a positional rewind of
    /// `seq_pos` is valid. `None` = hook not provided.
    fn decide_kv_compact_offset(&self, _m: &LoadedModel) -> Option<usize> {
        None
    }
}

/// The single registry lookup the daemon's spec path routes through: resolve the
/// carrier that claims `arch_id`, so the daemon never arch-matches for the
/// spec-decode guard / emitter. `is_dir` is `false` here because every
/// spec-capable arch is disjoint on the bare HFQ `arch_id` (qwen35 5|6, llama
/// 0|1, qwen2 7, deepseek4 9) and all carriers ignore the dir flag; if a future
/// arch needs HFQ-vs-dir disambiguation in the spec path, thread a retained
/// `is_dir` from load time rather than re-deriving it here.
pub fn carrier_for(arch_id: u32) -> Option<&'static dyn Carrier> {
    REGISTRY
        .iter()
        .copied()
        .find(|c| c.claims_arch_id(arch_id, false))
}

/// All registered carriers (read-only view for capability tests).
pub fn registry_carriers() -> impl Iterator<Item = &'static dyn Carrier> {
    REGISTRY.iter().copied()
}

// ─── Typed routing (replaces stringly `c.name() == "..."` predicates) ──────
// Each route is an exact `arch_id` match — no carrier `name()` or broader
// `claims_arch_id` set leaks through. Gemma 22 is deliberately excluded from
// `GenerationEarlyRoute::Gemma4` (arch 22 is a sidecar drafter, never a
// primary generate target). See blocker 1.

/// Continuous-batch staging route. `None` = not batch-capable (fallback to sequential).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousBatchRoute {
    Qwen35,
    Lfm2Moe,
    /// Flash-Next (qwen4, arch 16) exact AR lanes. Staged only through the
    /// VMM branch (`serve.vmm_batch`); there is no fixed-lane qwen4 body.
    Qwen4,
}
/// Exact arch_id -> continuous-batch route. qwen35 5|6 and qwen4 16 admit:
/// LFM2 (11) has no servable batch path (see the lfm2moe carrier caps), so
/// the route refuses it and no batch state is ever allocated. No carrier
/// probing — pure id match. `ContinuousBatchRoute::Lfm2Moe` stays for the
/// staging body, which is now unreachable.
pub fn continuous_batch_route(arch_id: u32) -> Option<ContinuousBatchRoute> {
    match arch_id {
        5 | 6 => Some(ContinuousBatchRoute::Qwen35),
        16 => Some(ContinuousBatchRoute::Qwen4),
        _ => None,
    }
}

/// Bench-decode redline route. Exhaustive — every arch maps to exactly one
/// variant, so a `match` without wildcard is a compile error when a new
/// bench arch is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchDecodeRoute {
    Deepseek4,
    Lfm2Moe,
    Qwen35,
    Gemma4,
    MuseGlimmer,
    Qwen4,
    Raon,
    Unsupported,
}
pub fn bench_decode_route(arch_id: u32) -> BenchDecodeRoute {
    match arch_id {
        9 => BenchDecodeRoute::Deepseek4,
        11 => BenchDecodeRoute::Lfm2Moe,
        5 | 6 => BenchDecodeRoute::Qwen35,
        13 => BenchDecodeRoute::Gemma4,
        14 => BenchDecodeRoute::MuseGlimmer,
        16 => BenchDecodeRoute::Qwen4,
        19 => BenchDecodeRoute::Raon,
        _ => BenchDecodeRoute::Unsupported,
    }
}

/// The daemon's stable per-arch label: the retained-Redline config key and
/// the status `model_arch` field. Unlisted ids read as dense `qwen3`.
pub fn arch_label(arch_id: u32) -> &'static str {
    match arch_id {
        5 => "qwen3_5",
        6 => "qwen3_5_moe",
        7 => "qwen2",
        8 => "dots-ocr",
        9 => "deepseek4",
        10 => "minimax_m2",
        11 => "lfm2moe",
        12 => "north_mini_code",
        13 => "gemma4",
        14 => "muse_glimmer",
        16 => "qwen4",
        17 => "qwen3_5_embed",
        18 => "embedding_gemma2",
        19 => "raon1_mini",
        40 => "flux_mmdit",
        45 => "flux2_mmdit",
        _ => "qwen3",
    }
}

/// Vision route. `None` = no vision encoder (text-only). The daemon still
/// gates on `has_image`/`has_vl`; this route only selects the per-arch
/// vision implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionRoute {
    DotsOcr,
    QwenVl,
    Lfm2Vl,
    None,
}
pub fn vision_route(arch_id: u32) -> VisionRoute {
    // Declared-capability gate: text-only arches declare `supports_images == false`
    // and must return `None` even if the arch_id table would say otherwise.
    // The table itself cannot be removed: it discriminates *which* vision
    // implementation to run (QwenVl vs DotsOcr vs Lfm2Vl have distinct generate
    // bodies in `hipfire_generate::vision`), not just whether vision is present.
    // Consulting caps here makes the gate declarative without changing behaviour.
    // The converse is not implied: an images-capable arch with no generate-side
    // VL body still returns `None` — the embed-only encoders (arch 17, 18) take
    // their media through the bundle, and 18's `None` is pinned by test.
    let caps = carrier_for(arch_id).map(|c| c.caps()).unwrap_or_default();
    if !caps.supports_images {
        return VisionRoute::None;
    }
    match arch_id {
        8 => VisionRoute::DotsOcr,
        5 | 6 => VisionRoute::QwenVl,
        11 => VisionRoute::Lfm2Vl,
        _ => VisionRoute::None,
    }
}

/// Image-generation route: diffusion checkpoint trunks are
/// loadable components that must NEVER ride the text `generate` path, and
/// text models must never ride `img_generate`. The daemon consults this in
/// both directions as the fail-closed gate (the same posture as the `toy`
/// 0xFF rule and the VL `has_image` gate above).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImgRoute {
    /// FLUX.1 MMDiT pipe (arch 40) — `img_generate` dispatches the CPU
    /// pipeline through the `FluxPipeModel` bundle.
    Flux,
    /// FLUX.2 Klein MMDiT pipe (arch 45) — `img_generate` dispatches the CPU
    /// pipeline through the same `FluxPipeModel` bundle, plus reference-image
    /// edit via `images[]`.
    Flux2,
    None,
}
impl ImgRoute {
    /// True for any diffusion image-gen route (`Flux`/`Flux2`); false for
    /// `None`. The daemon's text-generate and bench_prefill gates use this so
    /// a new diffusion family is refused by construction rather than by an
    /// enumerated arch-id list that can drift.
    pub fn is_diffusion(&self) -> bool {
        !matches!(self, ImgRoute::None)
    }
}
pub fn img_route(arch_id: u32) -> ImgRoute {
    match arch_id {
        40 => ImgRoute::Flux,
        45 => ImgRoute::Flux2,
        _ => ImgRoute::None,
    }
}

/// Slot-mode admission preflight for `path`, asked of every carrier that
/// provides one. The first carrier that recognizes the checkpoint answers;
/// `None` from all of them means the daemon's own preflight owns the path
/// (and its refusals).
pub fn slot_preflight(
    path: &std::path::Path,
) -> Option<Result<hipfire_runtime::serve::SlotModelPreflight, String>> {
    REGISTRY
        .iter()
        .copied()
        .find_map(|c| c.preflight_slot_model(path))
}
/// Which non-chat wire verb an arch serves.
///
/// The daemon gates `generate` / `img_generate` / `embed` on these predicates
/// rather than on an `arch_id` literal of its own, so arch knowledge stays in
/// the loader (the same rule `img_route` follows). Today only the qwen3.5
/// multi-vector embedding encoder (arch 17) is non-generating: it answers
/// `embed` and nothing else — it has no LM head and no KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVerb {
    /// The default: `generate` (and, for a diffusion family, `img_generate`).
    Generate,
    /// The multi-vector embedding encoder: `embed` only.
    Embed,
}

/// Classify an arch's wire verb. `Generate` is the default so a new chat arch
/// needs no change here; a new non-generating arch is added explicitly.
pub fn wire_verb(arch_id: u32) -> WireVerb {
    match arch_id {
        17 | 18 => WireVerb::Embed,
        _ => WireVerb::Generate,
    }
}

impl WireVerb {
    /// True when the arch answers `generate` (and so refuses `embed`).
    pub fn is_generate(&self) -> bool {
        matches!(self, WireVerb::Generate)
    }
}

/// EP prompt construction route. DeepSeek4 (arch 9) uses the DSML prompt
/// builder; all other EP arches (10 MiniMax, etc.) use Jinja.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpPromptRoute {
    Dsml,
    Jinja,
}
pub fn ep_prompt_route(arch_id: u32) -> EpPromptRoute {
    match arch_id {
        9 => EpPromptRoute::Dsml,
        _ => EpPromptRoute::Jinja,
    }
}

/// EP EOS token selection route. MiniMax (10) carries its own EOS; all others
/// (including DeepSeek4) use `deepseek4_eos_tok`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpEosRoute {
    Deepseek4,
    Minimax,
    Qwen35,
}
pub fn ep_eos_route(arch_id: u32) -> EpEosRoute {
    match arch_id {
        10 => EpEosRoute::Minimax,
        5 | 6 => EpEosRoute::Qwen35,
        _ => EpEosRoute::Deepseek4,
    }
}

/// Early generation short-circuit route (Gemma4 eager, Glimmer eager).
/// `None` = no early short-circuit (fall through to the generic
/// `select_generation_route` / DFlash/spec dispatch). Critically, arch 22
/// (Gemma4 EAGLE drafter sidecar) maps to `None` — only arch 13 is a
/// primary Gemma4 generate target. See blocker 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationEarlyRoute {
    Gemma4,
    MuseGlimmer,
}
pub fn generation_early_route(arch_id: u32) -> Option<GenerationEarlyRoute> {
    match arch_id {
        13 => Some(GenerationEarlyRoute::Gemma4),
        14 => Some(GenerationEarlyRoute::MuseGlimmer),
        _ => None,
    }
}

// ─── Registry ─────────────────────────────────────────────────────────

const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier,
    &Qwen4Carrier,
    &Qwen35Carrier,
    &LlamaCarrier,
    &DotsOcrCarrier,
    &Deepseek4Carrier,
    &MinimaxCarrier,
    &Lfm2MoeCarrier,
    &Cohere2MoeCarrier,
    &MapleCarrier,
    &Gemma4Carrier,
    &MuseGlimmerCarrier,
    &FluxDiffusionCarrier,
    &Qwen35EmbedCarrier,
    &EmbeddingGemma2Carrier,
    &RaonCarrier,
];

// ─── Constants ────────────────────────────────────────────────────────

/// Built-in Qwen3.5/3.6 chat template (froggeric/Qwen at HF).
/// Fallback when arch 5/6 has no configured/per-model override and the HFQ
/// lacks an embedded `tokenizer_config.chat_template` (older Qwen3.5/3.6 files).
/// Qwen3.8 ships an official template with `reasoning_effort` and uses that.
const FROGGERIC_QWEN35_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/qwen35-froggeric-v20.jinja");

/// Built-in LFM2.5 chat template.
const LFM2_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/lfm2-liquidai.jinja");

/// Built-in Gemma 4 IT chat template (arch_id=13).
const GEMMA4_TEMPLATE: &str = include_str!("../../hipfire-runtime/templates/gemma-4-it.jinja");

// ─── Eviction policy wrapper ──────────────────────────────────────────

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
pub enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    pub fn maybe_evict(
        &self,
        gpu: &mut rdna_compute::Gpu,
        kv: &mut llama::KvCache,
        physical: usize,
    ) -> hip_bridge::HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    pub fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    pub fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

// `DdtreeState`, `DflashState`, `load_dflash_state`, and the `DflashSpeculator`
// impl now live in `hipfire_arch_qwen35::dflash_spec` — all qwen35 + runtime
// types, so the loader only constructs and routes them, never owns the DFlash
// mechanics.

// ─── AsstTurnCache ────────────────────────────────────────────────────

/// Per-turn token cache for V4F prefix-cache stability.
///
/// Entries are keyed by the content/tool-call fingerprint PLUS the producer
/// reasoning identity (`entry.reasoning.text`, missing and empty being the same
/// identity). Two turns that normalize to the same content fingerprint (e.g.
/// distinct think-only turns with empty content) therefore coexist instead of
/// overwriting each other. `by_content` indexes the identities present for each
/// content fingerprint so the reasoning-less fallback is an O(1) uniqueness
/// check rather than a scan.
pub struct AsstTurnCache {
    cap: Option<usize>,
    map: std::collections::HashMap<AsstTurnKey, hipfire_runtime::prompt_frame::CachedAssistantTurn>,
    order: std::collections::VecDeque<AsstTurnKey>,
    by_content: std::collections::HashMap<u64, Vec<AsstTurnKey>>,
}

/// Cache slot identity: content/tools fingerprint + fingerprint of NONEMPTY
/// producer reasoning (`None` for missing/empty).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AsstTurnKey {
    fp: u64,
    reasoning: Option<u64>,
}

impl AsstTurnKey {
    fn new(fp: u64, reasoning: Option<&str>) -> Self {
        use std::hash::{Hash, Hasher};
        let reasoning = reasoning.filter(|r| !r.is_empty()).map(|r| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        });
        Self { fp, reasoning }
    }
}

impl AsstTurnCache {
    pub fn new_from_env() -> Self {
        let unbounded = hipfire_config::developer_var("HIPFIRE_PROMPT_CACHE_UNBOUNDED")
            .ok()
            .as_deref()
            == Some("1");
        let cap = if unbounded {
            None
        } else {
            Some(
                hipfire_config::developer_var("HIPFIRE_PROMPT_CACHE_CAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32),
            )
        };
        Self::with_cap(cap)
    }

    fn with_cap(cap: Option<usize>) -> Self {
        Self {
            cap,
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            by_content: std::collections::HashMap::new(),
        }
    }

    fn touch_mru(&mut self, key: AsstTurnKey) {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }

    /// True when any reasoning identity is cached for this content fingerprint.
    pub fn contains_key(&self, fp: &u64) -> bool {
        self.by_content.contains_key(fp)
    }

    /// Reasoning-less lookup: hits only when exactly one identity is cached
    /// for `fp`. See [`Self::get_for_reasoning`].
    pub fn get(&mut self, fp: &u64) -> Option<&hipfire_runtime::prompt_frame::CachedAssistantTurn> {
        self.get_for_reasoning(fp, None)
    }

    /// Nonempty `reasoning` selects exactly that producer-reasoning identity.
    /// `None`/empty hits only when exactly ONE identity is cached for `fp`
    /// (ambiguity → miss). A hit refreshes MRU order.
    pub fn get_for_reasoning(
        &mut self,
        fp: &u64,
        reasoning: Option<&str>,
    ) -> Option<&hipfire_runtime::prompt_frame::CachedAssistantTurn> {
        let wanted = reasoning.filter(|r| !r.is_empty());
        let key = match wanted {
            Some(r) => {
                let key = AsstTurnKey::new(*fp, Some(r));
                // Guard the 64-bit reasoning hash with the stored text.
                let exact = self
                    .map
                    .get(&key)
                    .and_then(|t| t.reasoning.as_ref())
                    .is_some_and(|b| b.text == r);
                if !exact {
                    return None;
                }
                key
            }
            None => match self.by_content.get(fp).map(Vec::as_slice) {
                Some([only]) => *only,
                _ => return None,
            },
        };
        self.touch_mru(key);
        self.map.get(&key)
    }

    pub fn insert(&mut self, fp: u64, turn: hipfire_runtime::prompt_frame::CachedAssistantTurn) {
        let key = AsstTurnKey::new(fp, turn.reasoning.as_ref().map(|b| b.text.as_str()));
        if self.map.contains_key(&key) {
            self.map.insert(key, turn);
            self.touch_mru(key);
            return;
        }
        if let Some(c) = self.cap {
            while self.order.len() >= c {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                    if let Some(keys) = self.by_content.get_mut(&old.fp) {
                        keys.retain(|k| *k != old);
                        if keys.is_empty() {
                            self.by_content.remove(&old.fp);
                        }
                    }
                } else {
                    break;
                }
            }
        }
        self.map.insert(key, turn);
        self.order.push_back(key);
        self.by_content.entry(fp).or_default().push(key);
    }

    /// Drop all cached assistant-turn token sequences (authoritative cold reset).
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.by_content.clear();
    }

    /// Read-only emptiness probe for test/snapshot surfaces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// `ModelState` was the closed 12-variant enum that stored `LoadedModel.state`.
// It has been replaced by `Option<Box<dyn ArchModel>>` (see `LoadedModel.state`
// below). All per-architecture teardown now lives in each bundle's
// `ArchModel::free_gpu`; there is no central match any more.
/// `ArchModel` for the one bundle the loader defines itself.
///
/// Every other architecture implements the trait in its own crate. Glimmer's
/// bundle type lives here rather than in `hipfire-arch-muse-glimmer`, so the
/// impl has to live here too — the orphan rule leaves no choice. Moving the
/// type into its crate is Phase 2's job; until then this is the honest place.
impl hipfire_runtime::arch_model::ArchModel for MuseGlimmerBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "muse_glimmer"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        // Glimmer keeps its KV inside GlimmerState rather than owning a
        // runtime `KvCache` directly, so there is nothing to hand back.
        None
    }
    fn reset_session_state(&mut self, _gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        // Mirrors daemon reset arms (main.rs:3339 / 3390): `bundle.reset_session_state()`.
        MuseGlimmerBundle::reset_session_state(self);
        Ok(())
    }
    fn free_gpu(self: Box<Self>, gpu: &mut rdna_compute::Gpu) {
        // This WAS the `ModelState::MuseGlimmer` arm of `unload_model`, moved
        // verbatim, drafter first. Per PR #566: freeing only one of
        // state/weights leaks ~1.3 GB over five load cycles, so both sides
        // are mandatory.
        let b = *self;
        if let Some(drafter) = b.drafter {
            drafter.scratch.free_gpu(gpu);
            drafter.weights.free_gpu(gpu);
        }
        b.state.free_gpu(gpu);
        b.weights.free_gpu(gpu);
    }
}

/// `ArchModel` for the bundle the loader defines itself.
///
/// `Deepseek4HeterogeneousBundle` is declared here rather than in its arch
/// crate, so — orphan rule — the impl must be here too.
impl hipfire_runtime::arch_model::ArchModel for Deepseek4HeterogeneousBundle {
    fn dim(&self) -> usize {
        self.model.config.hidden_size
    }
    fn n_layers(&self) -> usize {
        self.model.config.num_hidden_layers
    }
    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "deepseek4"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }
    fn free_gpu(self: Box<Self>, _gpu: &mut rdna_compute::Gpu) {
        // Self-owned two-device transaction: `Drop` frees each resource on its
        // exact owner and drains both pools. Calling anything else here would
        // double-free, which is why this body is a drop and not a sequence.
        drop(self);
    }
}

/// Self-owned gfx1100+dense / gfx1151+routed DeepSeek V4 state. The model
/// owns both HIP devices and tears them down in its `Drop` implementation;
/// the daemon's ordinary single-device `Gpu` must never free these buffers.
pub struct Deepseek4HeterogeneousBundle {
    pub model: hipfire_arch_deepseek4::DeepseekV4HeterogeneousModel,
    pub eos_tok: u32,
}

/// LFM2.5-MoE (arch_id=11) GPU bundle. Re-exported from the arch crate, which
/// owns it so `impl SpecTarget for Lfm2MoeBundle` (the n-gram verify seam, incl.
/// the conv-state snapshot/rollback) can live next to the forward it drives
/// (orphan rule). Field-identical to the prior loader-local struct. `eos_tok` is
/// resolved at load time and rides along so the generate path doesn't re-tokenize.
pub use lfm2moe::Lfm2MoeBundle;
pub use raon::RaonBundle;

/// MiniMax-M2 (arch_id=10) GPU bundle. Re-exported from the arch crate, which
/// owns it so `impl SpecTarget for MiniMaxBundle` (the n-gram verify seam) can
/// live next to the forward it drives (orphan rule). Field-identical to the
/// prior loader-local struct (`config`/`weights`/`state`/`eos_tok`).
pub use minimax::MiniMaxBundle;

/// Cohere2-MoE / North-Mini-Code (arch_id=12) GPU bundle. Re-exported from the
/// arch crate, which owns it so `impl SpecTarget for Cohere2MoeBundle` (the
/// n-gram verify seam) lives next to the forward it drives (orphan rule).
/// Field-identical to the prior loader-local struct.
pub use cohere2moe::Cohere2MoeBundle;

/// Gemma 4 (arch_id=13) GPU bundle, owned by its arch crate (which also
/// implements `ArchModel` for it).
pub use gemma4::Gemma4Bundle;

/// Muse Glimmer 30B dense text (arch_id=14) GPU bundle — eager dense path.
pub struct MuseGlimmerBundle {
    pub config: glimmer::config::GlimmerConfig,
    pub weights: glimmer::glimmer::GlimmerWeights,
    pub state: glimmer::glimmer::GlimmerState,
    pub eos_tok: u32,
    /// Optional DFlash drafter (arch 23, 5-layer diffusion) loaded when
    /// `HIPFIRE_DFLASH_DRAFT` (or `params.draft`) is set. OFF by default
    /// (`dflash_mode=off` forces `ctx.draft_path=None`). When `None`, the
    /// daemon runs AR-only; when `Some`, the target can be driven via the
    /// generic DFlash `Speculator` (see `carriers.rs`).
    pub drafter: Option<GlimmerDrafterBundle>,
    /// Host-side target hidden history for DFlash: per-position concat of
    /// residual hidden at target_layer_ids [1,13,25,37,49] (5*6656=33280 f32/pos)
    /// in that order. Grows by 1 row per committed token; used as
    /// `target_hidden` input to the drafter's encoder.fc. Empty until first
    /// prefill completes. Order matches the concat the drafter's fc expects —
    /// mixing layers silently degrades tau.
    ///
    /// Authoritative only when device hidden capture is disabled. When the
    /// device log is enabled, that log is the source of truth and this vector
    /// is retained solely as an exact host fallback (session rewind must not
    /// mutate it in device mode).
    pub target_hidden_host: Vec<f32>,
}

impl MuseGlimmerBundle {
    /// f32 elements per captured position (`num_extract * hidden`), or `None`
    /// when no DFlash drafter is loaded.
    #[inline]
    pub fn capture_row_elems(&self) -> Option<usize> {
        self.drafter
            .as_ref()
            .map(|d| d.config.num_extract() * d.config.hidden)
    }

    /// Full session reset: target KV cursor + device log metadata, host capture
    /// vector, and drafter absolute history watermarks.
    pub fn reset_session_state(&mut self) {
        self.state.reset();
        self.target_hidden_host.clear();
        if let Some(d) = self.drafter.as_mut() {
            d.scratch.reset_history();
        }
    }

    /// Whether [`Self::rewind_session_to`] can safely land on `target`.
    ///
    /// Rejects `target > state.n_tokens`. Device capture requires an Idle log
    /// that still retains `target` in its ring. Host-backed DFlash requires the
    /// host vector to be an exact whole-row packing of the current
    /// `state.n_tokens` (no gap/partial row) with at least `target` rows. With
    /// no drafter, rewind is target-KV-only and always allowed within range.
    pub fn can_rewind_session_to(&self, target: usize) -> bool {
        if target > self.state.n_tokens {
            return false;
        }
        if self.state.device_hidden_capture_enabled() {
            return self
                .state
                .target_hidden_log()
                .map(|log| log.can_rewind_to(target))
                .unwrap_or(false);
        }
        match self.capture_row_elems() {
            None => true,
            Some(row_elems) => {
                if row_elems == 0 {
                    return false;
                }
                let host_len = self.target_hidden_host.len();
                if host_len % row_elems != 0 {
                    return false;
                }
                let host_rows = host_len / row_elems;
                host_rows >= target && host_rows == self.state.n_tokens
            }
        }
    }

    /// Rewind target KV (+ device log when enabled), host capture (host-backed
    /// DFlash only), and drafter history to `target`.
    ///
    /// All fallible checks run before any mutation. Device mode never touches
    /// `target_hidden_host`. On success: `state.n_tokens == target`, device log
    /// end == target when enabled, host rows == target when host-backed+dflash,
    /// and drafter `kv_abs_end <= target`.
    pub fn rewind_session_to(&mut self, target: usize) -> Result<(), String> {
        if !self.can_rewind_session_to(target) {
            return Err(format!(
                "muse glimmer: cannot rewind session to {target} (n_tokens={}, device_capture={})",
                self.state.n_tokens,
                self.state.device_hidden_capture_enabled()
            ));
        }

        // Compute host truncate length before mutating anything.
        let host_new_len = if self.state.device_hidden_capture_enabled() {
            None
        } else if let Some(row_elems) = self.capture_row_elems() {
            let new_len = target.checked_mul(row_elems).ok_or_else(|| {
                format!(
                    "muse glimmer: host capture truncate overflow target={target} row_elems={row_elems}"
                )
            })?;
            Some(new_len)
        } else {
            None
        };

        glimmer::forward::rollback_to(&mut self.state, target)?;

        if let Some(new_len) = host_new_len {
            self.target_hidden_host.truncate(new_len);
        }
        if let Some(d) = self.drafter.as_mut() {
            d.scratch.rewind_history(target);
        }
        Ok(())
    }

    /// Current target KV cursor (`state.n_tokens`).
    #[inline]
    pub fn n_tokens(&self) -> usize {
        self.state.n_tokens
    }

    /// Whether the device hidden-capture log is installed on the target state.
    #[inline]
    pub fn device_hidden_capture_enabled(&self) -> bool {
        self.state.device_hidden_capture_enabled()
    }

    /// Committed absolute end of the device hidden log, when installed.
    #[inline]
    pub fn device_hidden_log_end(&self) -> Option<usize> {
        self.state
            .target_hidden_log()
            .map(|log| log.committed_abs_end())
    }

    /// Drafter scratch ctx capacity (`None` without a drafter).
    #[inline]
    pub fn drafter_ctx_capacity(&self) -> Option<usize> {
        self.drafter.as_ref().map(|d| d.scratch.ctx_capacity())
    }

    /// Host-backed capture row count when `capture_row_elems` is known and the
    /// host vector is an exact whole-row packing; otherwise `None`.
    #[inline]
    pub fn host_capture_rows(&self) -> Option<usize> {
        let row_elems = self.capture_row_elems()?;
        if row_elems == 0 || self.target_hidden_host.len() % row_elems != 0 {
            return None;
        }
        Some(self.target_hidden_host.len() / row_elems)
    }
}

/// Muse Glimmer DFlash drafter (arch 23) — `muse_glimmer_assistant` 5-layer
/// diffusion draft head (encoder.fc + output_norm_enc, block 16, mask 201818,
/// target_layer_ids [1,13,25,37,49]). No embed/lm_head (uses target's).
/// Stored alongside the arch-14 target when `HIPFIRE_DFLASH_DRAFT` is set.
pub struct GlimmerDrafterBundle {
    pub config: glimmer::drafter::GlimmerDrafterConfig,
    pub weights: glimmer::drafter::GlimmerDrafterWeights,
    /// Scratch sized to `max_seq` (target's ctx capacity). Allocated once at
    /// load time, freed on unload.
    pub scratch: glimmer::drafter::GlimmerDrafterScratch,
}

/// v1 draft length for gemma4 EAGLE (`params.spec`). dl=3 is the validated
/// config.
pub const GEMMA4_EAGLE_DRAFT_LEN: usize = 3;

/// Gate for the `params.spec` knob: draft_len 1..=5 accepted; absent means
/// the validated default (`GEMMA4_EAGLE_DRAFT_LEN` = 3). Mirrors the PR's
/// `gemma4_eagle_spec_len` — refuse-don't-degrade for unvalidated lengths.
pub fn gemma4_eagle_spec_len(spec: Option<u64>) -> Result<usize, String> {
    match spec {
        None => Ok(GEMMA4_EAGLE_DRAFT_LEN),
        Some(n) if (1..=5).contains(&n) => Ok(n as usize),
        Some(n) => Err(format!(
            "gemma4 EAGLE supports spec=1..=5 (draft_len; default 3, dl <= 5              parity-validated on gfx1201); got spec={n} — reload with a              supported spec or drop params.drafter."
        )),
    }
}

// ─── LoadedModel ──────────────────────────────────────────────────────

pub struct LoadedModel {
    pub arch_id: u32,
    pub pp: usize,
    pub pp_gpus: Option<Gpus>,
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    pub ep: Option<EpState>,
    // Shared arch state — every arch lives here as `Box<dyn ArchModel>`.
    // Previously `ModelState::DotsOcr` / `ModelState::Qwen35` etc; now a
    // trait object so adding an architecture does not edit a closed enum.
    // All consumers go through the typed accessors below (`dots_ocr()`,
    // `qwen35()`, etc.) which downcast via `Any`.
    pub state: Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
    // DeepSeek V4 (arch_id=9) single-GPU prefill scratch now lives inside
    // `Deepseek4Bundle.pbs` (moved from here so `LoadedModel` can become arch-free).
    // DeepSeek V4 (arch_id=9) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Ds4), NOT in `state`, so there is no Deepseek4Bundle for EP
    pub deepseek4_eos_tok: u32,
    // MiniMax-M2 (arch_id=10) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Minimax), NOT in `state`, so `minimax()` is None for EP
    // models — the eos must be carried here (mirrors `deepseek4_eos_tok`).
    pub minimax_eos_tok: u32,
    // Qwen3.5/3.6 (arch_id=5|6) EP serve eos. The EP path stores model state in
    // `ep` (EpArch::Qwen35), NOT in `state`, so the eos must be carried here
    // (mirrors `deepseek4_eos_tok` / `minimax_eos_tok`).
    pub qwen35_eos_tok: u32,
    // LFM2.5-8B-A1B (arch_id=11) and MiniMax-M2 (arch_id=10) live in
    // `state` as `Box<dyn ArchModel>` so unload teardown is
    // via `ArchModel::free_gpu` (no central match any more).
    // MTP config
    pub mtp_mode: String,
    pub mtp_k: usize,
    pub mtp_weights_present: bool,
    // Qwen3.5/3.6 native MTP (NextN) head (arch_id=21). Loaded once at model
    // load when a bundled `.mq4-mtp` trailer OR a separate `.mtp` sidecar is
    // present alongside the trunk. Persistent for the life of the model;
    // `generate_qwen35_mtp` allocates a fresh per-request `MtpSpecState`
    // against it (so the recurrent MTP-KV never bleeds across requests). None
    // for every other arch and for qwen35 trunks without an MTP head.
    // Shared
    pub tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer>,
    pub seq_pos: usize,
    pub max_seq: usize,
    pub physical_cap: usize,
    pub sequence: Option<crate::admission::SequenceResolution>,
    pub eviction: Option<Eviction>,
    pub kv_adaptive: Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub dflash_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub asst_turn_cache: AsstTurnCache,
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
    pub model_path: String,
    /// The model's speculative-decode drafter+verifier, when a draft model is
    /// loaded (`Box<dyn Speculator>` so the daemon's decode loop is agnostic to
    /// DFlash chain / DDTree tree / future MTP). Replaces the old
    /// `dflash: Option<DflashState>` field — the `DflashState` now lives inside
    /// the `DflashSpeculator` impl behind this trait object.
    pub speculator: Option<Box<dyn Speculator>>,
    pub chat_template: Option<String>,
    /// Where `chat_template` came from after override/embedded/fallback
    /// selection (`None` iff `chat_template` is `None`): human-readable,
    /// names user-override file paths exactly. Published only in the daemon's
    /// effort-probe diagnostic line.
    pub chat_template_source: Option<String>,
    // Author-recommended sampling defaults, baked into the .hfq's
    // `generation_config` metadata and read at load time on the HFQ source
    // path (raw-safetensors PP path leaves them `None`). The generate handler
    // falls back to these when the request omits the matching knob, before the
    // arch-ladder defaults. `rec_min_p` / `rec_presence_penalty` are NOT carried
    // in generation_config (they reach the daemon only via the request), so they
    // stay `None` on the load path.
    pub rec_temperature: Option<f32>,
    pub rec_top_p: Option<f32>,
    pub rec_top_k: Option<f32>,
    pub rec_min_p: Option<f32>,
    pub rec_presence_penalty: Option<f32>,
}

impl LoadedModel {
    /// Shared-field skeleton: arch state None, pp = 1, all non-core arch slots
    /// None, collections empty, mtp defaults, asst cache from env. Callers set
    /// only the fields they own via struct-update (`..LoadedModel::skeleton(..)`).
    pub fn skeleton(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        chat_template_source: Option<String>,
    ) -> Self {
        LoadedModel {
            arch_id,
            pp: 1,
            ep: None,
            pp_gpus: None,
            pp_dn_la_to_device: None,
            state: None,
            deepseek4_eos_tok: 0,
            minimax_eos_tok: 0,
            qwen35_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: hipfire_config::MTP_K_DEFAULT,
            mtp_weights_present: false,
            tokenizer: Some(tokenizer),
            seq_pos: 0,
            max_seq,
            physical_cap,
            sequence: None,
            eviction: None,
            kv_adaptive: None,
            conversation_tokens: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
            prefill_checkpoints: Vec::new(),
            dflash_checkpoints: Vec::new(),
            decoded_vocab: None,
            model_path,
            speculator: None,
            chat_template,
            chat_template_source,
            rec_temperature: None,
            rec_top_p: None,
            rec_top_k: None,
            rec_min_p: None,
            rec_presence_penalty: None,
        }
    }

    /// Whether this model's producer declared a measured AR floor
    /// ([`Carrier::mtp_measured_ar_floor`]); `false` for an arch with no
    /// registered carrier. Every MTP consumer of an `auto` mode reads this
    /// (not head presence) so a loaded head never rearms `auto` by itself.
    pub fn mtp_measured_ar_floor(&self) -> bool {
        carrier_for(self.arch_id).is_some_and(|carrier| carrier.mtp_measured_ar_floor())
    }

    /// `mtp_mode` after the producer-capability gate
    /// ([`hipfire_config::mtp_effective_mode_name`]): `auto` on a producer
    /// without a measured AR floor is `"off"`; `on`/`off` are unchanged.
    pub fn effective_mtp_mode(&self) -> &'static str {
        hipfire_config::mtp_effective_mode_name(&self.mtp_mode, self.mtp_measured_ar_floor())
    }

    /// LFM2.5-MoE bundle if this model is arch_id=11, else None.
    /// Typed accessors for the remaining bundles.
    ///
    /// Six of these already existed; these six close the set so EVERY bundle is
    /// reachable the same way. That uniformity is the point: once every
    /// consumer goes through an accessor instead of destructuring
    /// `ModelState` itself, swapping the storage to `Box<dyn ArchModel>`
    /// changes these bodies and nothing else. Without them, that swap would
    /// have to rewrite ~143 call sites in `hipfire-generate` at the same time
    /// as changing the type — one unreviewable change instead of two safe ones.
    pub fn qwen35(&self) -> Option<&hipfire_arch_qwen35::Qwen35Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>())
    }

    pub fn qwen35_mut(&mut self) -> Option<&mut hipfire_arch_qwen35::Qwen35Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>())
    }

    /// Qwen4 bundle if this model is arch_id=16, else None.
    pub fn qwen4(&self) -> Option<&hipfire_arch_qwen4::bundle::Qwen4Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_qwen4::bundle::Qwen4Bundle>())
    }

    pub fn qwen4_mut(&mut self) -> Option<&mut hipfire_arch_qwen4::bundle::Qwen4Bundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen4::bundle::Qwen4Bundle>()
        })
    }

    /// Whole-request PLE serving guard of a Qwen4 model: while it is held
    /// the PLE row store counts as busy, so background PLE prewarm readers
    /// stay paused. `None` for any other architecture, or a Qwen4 profile
    /// without PLE. Hold it for the full lifespan of a request (admission to
    /// terminal event) and drop it on every exit.
    pub fn ple_serving_guard(&self) -> Option<hipfire_runtime::external_rows::ServingGuard> {
        self.qwen4().and_then(|bundle| bundle.ple_serving_guard())
    }

    /// The Qwen4 PP root bundle and its generic `Gpus` as disjoint borrows
    /// (`state` and `pp_gpus` are separate fields), for generation closures
    /// that need both at once. `None` unless this is a Qwen4 bundle whose
    /// `pp_gpus` is present.
    pub fn qwen4_mesh_mut(
        &mut self,
    ) -> Option<(&mut hipfire_arch_qwen4::bundle::Qwen4Bundle, &mut Gpus)> {
        let bundle = self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen4::bundle::Qwen4Bundle>()
        })?;
        let gpus = self.pp_gpus.as_mut()?;
        Some((bundle, gpus))
    }

    pub fn llama(&self) -> Option<&hipfire_arch_llama::LlamaBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_llama::LlamaBundle>())
    }

    pub fn llama_mut(&mut self) -> Option<&mut hipfire_arch_llama::LlamaBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>())
    }

    pub fn gemma4(&self) -> Option<&Gemma4Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Gemma4Bundle>())
    }

    pub fn gemma4_mut(&mut self) -> Option<&mut Gemma4Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Gemma4Bundle>())
    }

    pub fn muse_glimmer(&self) -> Option<&MuseGlimmerBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<MuseGlimmerBundle>())
    }

    pub fn muse_glimmer_mut(&mut self) -> Option<&mut MuseGlimmerBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<MuseGlimmerBundle>())
    }

    /// Qwen3.5 multi-vector embedding bundle if this model is arch_id=17,
    /// else `None`. The daemon `embed` verb's typed handle.
    pub fn qwen35_embed_mut(&mut self) -> Option<&mut hipfire_arch_qwen35_embed::EmbedBundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35_embed::EmbedBundle>()
        })
    }

    /// EmbeddingGemma-2 bundle if this model is arch_id=18, else `None`.
    /// The daemon `embed` verb's typed handle.
    pub fn embeddinggemma2_mut(
        &mut self,
    ) -> Option<&mut hipfire_arch_embeddinggemma2::EmbedGemmaBundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_embeddinggemma2::EmbedGemmaBundle>()
        })
    }

    /// Read-only twin of [`Self::embeddinggemma2_mut`] for the residency
    /// probes (`has_vision_encoder` / `has_audio_encoder` / `has_video_route`)
    /// and any other inspection that needs no mutation.
    pub fn embeddinggemma2(&self) -> Option<&hipfire_arch_embeddinggemma2::EmbedGemmaBundle> {
        self.state.as_deref().and_then(|s| {
            (s as &dyn Any).downcast_ref::<hipfire_arch_embeddinggemma2::EmbedGemmaBundle>()
        })
    }

    pub fn deepseek4_heterogeneous_mut(&mut self) -> Option<&mut Deepseek4HeterogeneousBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Deepseek4HeterogeneousBundle>())
    }

    pub fn lfm2moe(&self) -> Option<&Lfm2MoeBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Lfm2MoeBundle>())
    }

    pub fn lfm2moe_mut(&mut self) -> Option<&mut Lfm2MoeBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Lfm2MoeBundle>())
    }

    /// Raon-1-mini bundle if this model is arch_id=19, else None.
    pub fn raon(&self) -> Option<&RaonBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<RaonBundle>())
    }

    pub fn raon_mut(&mut self) -> Option<&mut RaonBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<RaonBundle>())
    }

    /// MiniMax-M2 bundle if this model is arch_id=10, else None.
    pub fn minimax(&self) -> Option<&MiniMaxBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<MiniMaxBundle>())
    }

    pub fn minimax_mut(&mut self) -> Option<&mut MiniMaxBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<MiniMaxBundle>())
    }

    /// Qwen2 bundle if this model is arch_id=7 (plain qwen2 via `Qwen2Carrier`),
    /// else None.
    pub fn qwen2_mut(&mut self) -> Option<&mut hipfire_arch_qwen2::Qwen2Bundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_qwen2::Qwen2Bundle>())
    }

    /// Cohere2-MoE bundle if this model is arch_id=12, else None.
    pub fn cohere2moe(&self) -> Option<&Cohere2MoeBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<Cohere2MoeBundle>())
    }

    pub fn cohere2moe_mut(&mut self) -> Option<&mut Cohere2MoeBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<Cohere2MoeBundle>())
    }

    /// Maple-Preview bundle if this model is arch_id=15, else None.
    pub fn maple(&self) -> Option<&hipfire_arch_maple::MapleBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_maple::MapleBundle>())
    }

    pub fn maple_mut(&mut self) -> Option<&mut hipfire_arch_maple::MapleBundle> {
        self.state
            .as_deref_mut()
            .and_then(|s| (s as &mut dyn Any).downcast_mut::<hipfire_arch_maple::MapleBundle>())
    }

    /// DeepSeek V4 bundle if this model is a single-GPU arch_id=9, else None.
    /// (EP/pp ds4 keeps its state in `ep` (EpArch::Ds4), so this is None there.)
    pub fn deepseek4(&self) -> Option<&hipfire_arch_deepseek4::Deepseek4Bundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_deepseek4::Deepseek4Bundle>())
    }

    pub fn deepseek4_mut(&mut self) -> Option<&mut hipfire_arch_deepseek4::Deepseek4Bundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        })
    }

    /// Qwen35-VL vision config if this model is arch_id=5|6 and was loaded
    /// with a vision tower (`model.visual.patch_embed.proj.weight` present),
    /// else None. Text-only Qwen35 returns None.
    pub fn vision_config(&self) -> Option<&qwen35_vl::VisionConfig> {
        self.qwen35().and_then(|b| b.vision_config.as_ref())
    }

    pub fn ack_dims(&self) -> (usize, usize, usize) {
        if let Some(st) = self.state.as_ref() {
            let arch = st.as_ref() as &dyn hipfire_runtime::arch_model::ArchModel;
            return (arch.dim(), arch.n_layers(), arch.vocab_size());
        }
        // EP and dense-TP loads keep the model in `ep`, leaving `state` empty.
        self.ep
            .as_ref()
            .map(|ep| ep.inner.model_dims())
            .unwrap_or((0, 0, 0))
    }

    pub fn vision_weights(&self) -> Option<&qwen35_vl::VisionWeights> {
        self.qwen35().and_then(|b| b.vision_weights.as_ref())
    }

    pub fn vision_config_mut(&mut self) -> Option<&mut qwen35_vl::VisionConfig> {
        self.qwen35_mut().and_then(|b| b.vision_config.as_mut())
    }

    pub fn vision_weights_mut(&mut self) -> Option<&mut qwen35_vl::VisionWeights> {
        self.qwen35_mut().and_then(|b| b.vision_weights.as_mut())
    }

    /// Declared vision capability across ALL carriers that can carry a
    /// tower: qwen35-vl (arch 5/6 bundle field), dots-ocr (arch 8),
    /// lfm2-vl (arch-11 bundle field), and embeddinggemma2 (arch 18, whose
    /// `.vl` pack carries the Gemma4 tower). The daemon's image gate and the
    /// `loaded` ack's `vl` fact consult this instead of the qwen-typed
    /// accessors above.
    pub fn has_vision_encoder(&self) -> bool {
        if self.dots_ocr().is_some() {
            return true;
        }
        if let Some(b) = self.lfm2moe() {
            if b.vision_config.is_some() {
                return true;
            }
        }
        if self.embeddinggemma2().is_some_and(|b| b.vision.is_some()) {
            return true;
        }
        self.qwen35().is_some_and(|b| b.vision_config.is_some())
    }

    /// Declared audio capability: an audio tower is resident in this load.
    /// Today only embeddinggemma2 (arch 18) carries one (in the same `.vl`
    /// pack as the vision tower); every other arch answers `false`, matching
    /// the embed body's per-modality refusals.
    pub fn has_audio_encoder(&self) -> bool {
        self.embeddinggemma2().is_some_and(|b| b.audio.is_some())
    }

    /// Declared video capability: the arch serves the embed route's video
    /// leg. Video has no tower of its own — arch 18's frames ride the vision
    /// tower, one pass per frame — so this is the vision tower's residency on
    /// the one arch with a video leg, and `false` everywhere else (arch 17's
    /// encoder has no video leg; chat VL arches take no video input).
    pub fn has_video_route(&self) -> bool {
        self.embeddinggemma2().is_some_and(|b| b.vision.is_some())
    }

    /// LFM2-VL (arch 11) projected-vision config + weights, when loaded
    /// from an artifact quantized with `--include-vision`.
    pub fn lfm2_vision(
        &self,
    ) -> Option<(
        &hipfire_arch_lfm2_vl::VisionConfig,
        &hipfire_arch_lfm2_vl::VisionWeights,
    )> {
        let b = self.lfm2moe()?;
        Some((b.vision_config.as_ref()?, b.vision_weights.as_ref()?))
    }

    /// DotsOcr bundle if this model is arch_id=8, else None.
    pub fn dots_ocr(&self) -> Option<&hipfire_arch_dots_ocr::DotsOcrBundle> {
        self.state
            .as_deref()
            .and_then(|s| (s as &dyn Any).downcast_ref::<hipfire_arch_dots_ocr::DotsOcrBundle>())
    }

    pub fn dots_ocr_mut(&mut self) -> Option<&mut hipfire_arch_dots_ocr::DotsOcrBundle> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        })
    }

    /// FLUX diffusion pipe bundle (arch 40) — the `img_generate` body's
    /// handle. `None` for every text/vision model (paired with
    /// [`img_route`] as the fail-closed gate).
    pub fn flux_pipe(&self) -> Option<&hipfire_arch_diffusion::arch_model::FluxPipeModel> {
        self.state.as_deref().and_then(|s| {
            (s as &dyn Any).downcast_ref::<hipfire_arch_diffusion::arch_model::FluxPipeModel>()
        })
    }

    pub fn flux_pipe_mut(
        &mut self,
    ) -> Option<&mut hipfire_arch_diffusion::arch_model::FluxPipeModel> {
        self.state.as_deref_mut().and_then(|s| {
            (s as &mut dyn Any).downcast_mut::<hipfire_arch_diffusion::arch_model::FluxPipeModel>()
        })
    }

    /// Arch-agnostic view of the loaded model, when any is loaded.
    pub fn as_arch_model(&self) -> Option<&dyn hipfire_runtime::arch_model::ArchModel> {
        self.state
            .as_deref()
            .map(|b| b as &dyn hipfire_runtime::arch_model::ArchModel)
    }
    pub fn as_arch_model_mut(&mut self) -> Option<&mut dyn hipfire_runtime::arch_model::ArchModel> {
        self.state
            .as_deref_mut()
            .map(|b| b as &mut dyn hipfire_runtime::arch_model::ArchModel)
    }
    /// pp>1 skeleton — sets the load-bearing multi-GPU fields together so
    /// they cannot be set piecemeal (`pp_gpus`/`pp_dn_la_to_device` are
    /// `.expect()`ed in unload). The per-device `Qwen35ScratchSet` that was
    /// previously the fourth field here now lives inside `Qwen35Bundle`
    /// (`b.pp_scratch_set`) and is freed via `free_gpu_multi` in the same
    /// pp>1 unload arm — co-locating it with the bundle's own `scratch`
    /// guarantees the set cannot be dropped without the bundle (and vice
    /// versa), eliminating the silent-VRAM-leak window a standalone
    /// `LoadedModel.pp_scratch_set: Option<Qwen35ScratchSet>` created.
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        chat_template_source: Option<String>,
        pp: usize,
        pp_gpus: Gpus,
        pp_dn_la_to_device: Vec<u8>,
    ) -> Self {
        LoadedModel {
            pp,
            pp_gpus: Some(pp_gpus),
            pp_dn_la_to_device: Some(pp_dn_la_to_device),
            ..LoadedModel::skeleton(
                arch_id,
                tokenizer,
                max_seq,
                physical_cap,
                model_path,
                chat_template,
                chat_template_source,
            )
        }
    }
}

/// Expert-parallel serving state.
pub struct EpState {
    pub gpus: Gpus,
    pub inner: EpArch,
}

pub enum EpArch {
    Ds4 {
        config: hipfire_arch_deepseek4::DeepseekV4Config,
        weights: Vec<hipfire_arch_deepseek4::DeepseekV4Weights>,
        state: Vec<hipfire_arch_deepseek4::DeepseekV4State>,
        partials: Vec<rdna_compute::GpuTensor>,
        /// Exact gfx1201 MQ2R TP3/TP4 batched-prefill scratch, one per rank.
        /// Empty for every other EP route.
        prefill: Vec<hipfire_arch_deepseek4::forward::PrefillBatchScratch>,
    },
    Minimax {
        config: minimax::MiniMaxConfig,
        weights: Vec<minimax::MiniMaxWeights>,
        state: Vec<minimax::MiniMaxState>,
        partials: Vec<rdna_compute::GpuTensor>,
    },
    Qwen35 {
        config: hipfire_arch_qwen35::qwen35::Qwen35Config,
        weights: Vec<hipfire_arch_qwen35::qwen35::Qwen35Weights>,
        /// Sequential EP decode state, one entry per rank: replicated KV +
        /// DeltaNet + scratch plus the rank's routed-output partial scratch.
        /// Present for every admitted MoE EP load (tp=2|4); the optional
        /// continuous-batch `batch` below stays an independent opt-in that
        /// remains tp=4-only via `validate_ep_batch_compatibility`.
        kv_caches: Vec<llama::KvCache>,
        dn_states: Vec<hipfire_arch_qwen35::qwen35::DeltaNetState>,
        scratches: Vec<hipfire_arch_qwen35::qwen35::Qwen35Scratch>,
        partials: Vec<rdna_compute::GpuTensor>,
        batch: Option<hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchEpState>,
        /// Load-owned sequential prefill scratch, one entry per rank on its
        /// owning device: `prefill_pbs[r]` is a `PrefillBatchScratch` sized
        /// `[chunk x dim]` with `chunk = min(max_seq, 512)` and
        /// `cap_gdn_tape = false`, and `prefill_partials[r]` is its zeroed
        /// `[chunk*dim]` F32 routed partial. Always nonempty after load;
        /// drained (empty) once the optional batch owner is staged, because
        /// batch mode serves prefill from its own `seed_pbs`/`seed_partials`.
        /// Serve branch: `batch.is_some()` borrows the seeds,
        /// `batch.is_none()` borrows these; serve never constructs either.
        prefill_pbs: Vec<hipfire_arch_qwen35::qwen35::PrefillBatchScratch>,
        prefill_partials: Vec<rdna_compute::GpuTensor>,
    },
    /// Dense Qwen tensor parallelism. Kept separate from `Qwen35`, whose
    /// ownership and scheduling contract is four-rank routed-expert EP.
    Qwen35DenseTp {
        shard: hipfire_runtime::tp_shard::ShardConfig,
        configs: Vec<hipfire_arch_qwen35::qwen35::Qwen35Config>,
        weights: Vec<hipfire_arch_qwen35::qwen35::Qwen35Weights>,
        kv_caches: Vec<llama::KvCache>,
        dn_states: Vec<hipfire_arch_qwen35::qwen35::DeltaNetState>,
        scratches: Vec<hipfire_arch_qwen35::qwen35::Qwen35Scratch>,
    },
    /// Dense Qwen tensor parallelism on the generic mesh executor
    /// (`HIPFIRE_DENSE_TP_ROUTE=mesh`): one `Qwen35TpRank` per device (weights,
    /// KV, DeltaNet state and scratch), the split plan of exactly those shards
    /// and the program validated against them once at load. The ranks are
    /// driven epoch by epoch through `Gpus::submit_mesh_epoch`.
    Qwen35MeshTp {
        global: hipfire_arch_qwen35::qwen35::Qwen35Config,
        ranks: Vec<hipfire_arch_qwen35::qwen35::mesh_tp::Qwen35TpRank>,
        plan: hipfire_runtime::mesh_split::SplitPlan,
        program: hipfire_arch_qwen35::qwen35::mesh_tp::Qwen35TpProgram,
        /// The last mesh op id issued (strictly increasing per epoch).
        next_op: u64,
        /// Tickets of the epochs enqueued and not yet drained, oldest first.
        /// Drained together (`Gpus::drain_mesh_epochs`) at the serve loop's
        /// rank-0 logits read, at per-request reset and abort, at unload and
        /// by the bench sync; never left across a request boundary.
        pending: Vec<hipfire_runtime::multi_gpu::executor::MeshTicket>,
    },
}

impl EpArch {
    /// Whether this EP load's serve loop applies a request's repeat, presence
    /// and frequency penalties to its logits. The Qwen3.5 loops (dense TP on
    /// either route, MoE EP) do, over rank 0's logits with the single-GPU host
    /// arithmetic; DeepSeek4 and MiniMax sample without a penalty stage, so
    /// the daemon refuses penalties for them.
    pub fn applies_request_penalties(&self) -> bool {
        matches!(
            self,
            EpArch::Qwen35 { .. } | EpArch::Qwen35DenseTp { .. } | EpArch::Qwen35MeshTp { .. }
        )
    }

    // A dense-TP rank config keeps global dim/n_layers/vocab: `local_dense_tp_config` narrows only head counts and `hidden_dim`.
    pub fn model_dims(&self) -> (usize, usize, usize) {
        match self {
            EpArch::Ds4 { config, .. } => (
                config.hidden_size,
                config.num_hidden_layers,
                config.vocab_size,
            ),
            EpArch::Minimax { config, .. } => (
                config.hidden_size,
                config.num_hidden_layers,
                config.vocab_size,
            ),
            EpArch::Qwen35 { config, .. } => (config.dim, config.n_layers, config.vocab_size),
            EpArch::Qwen35DenseTp { configs, .. } => configs
                .first()
                .map(|c| (c.dim, c.n_layers, c.vocab_size))
                .unwrap_or((0, 0, 0)),
            EpArch::Qwen35MeshTp { global, .. } => (global.dim, global.n_layers, global.vocab_size),
        }
    }
}

// ─── Helper functions ─────────────────────────────────────────────────

/// Provenance text for the embedded HFQ tokenizer_config template.
const SRC_EMBEDDED_HFQ: &str = "embedded HFQ tokenizer_config.chat_template";
/// Provenance text for the safetensors-directory template files.
pub(crate) const SRC_DIR_JINJA: &str = "model-dir chat_template.jinja";
pub(crate) const SRC_DIR_TOKENIZER_CONFIG: &str = "model-dir tokenizer_config.json chat_template";

/// Layer 1 (resolved config) + Layer 2 (per-model ~/.hipfire/templates).
/// Returns `(template, source)`; the source names the exact override file.
fn resolve_chat_template_overrides(model_path: &str) -> Option<(String, String)> {
    if let Some(config_path) = hipfire_runtime::config::get().chat_template_file.as_deref() {
        if !config_path.is_empty() {
            match std::fs::read_to_string(config_path) {
                Ok(s) => {
                    eprintln!("[chat_template] using configured template {config_path}");
                    return Some((
                        s,
                        format!("user override: configured chat_template_file {config_path}"),
                    ));
                }
                Err(e) => eprintln!(
                    "[chat_template] configured template {config_path} failed to read ({e}); falling through"
                ),
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let basename = std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !basename.is_empty() {
            let per_model = std::path::Path::new(&home)
                .join(".hipfire")
                .join("templates")
                .join(format!("{basename}.j2"));
            if per_model.is_file() {
                match std::fs::read_to_string(&per_model) {
                    Ok(s) => {
                        eprintln!(
                            "[chat_template] using per-model override {}",
                            per_model.display()
                        );
                        let source =
                            format!("user override: per-model template {}", per_model.display());
                        return Some((s, source));
                    }
                    Err(e) => eprintln!(
                        "[chat_template] per-model file {} failed to read ({e}); falling through",
                        per_model.display()
                    ),
                }
            }
        }
    }
    None
}

/// Effective template for an HFQ load plus its provenance description.
/// `(None, None)` when the artifact has no template and no built-in applies.
fn resolve_chat_template(hfq: &HfqFile, model_path: &str) -> (Option<String>, Option<String>) {
    if let Some((s, source)) = resolve_chat_template_overrides(model_path) {
        return (Some(s), Some(source));
    }
    match hfq.arch_id {
        // Prefer HFQ-embedded tokenizer_config.chat_template (Qwen3.8 official
        // low/medium/xhigh reasoning_effort). Fall back to froggeric for older
        // Qwen3.5/3.6 files that lack one.
        5 | 6 => {
            let (t, source) = qwen35_template_from_embedded(hfq.chat_template(), model_path);
            return (Some(t), Some(source));
        }
        11 => {
            if let Some(t) = hfq.chat_template() {
                return (Some(t), Some(SRC_EMBEDDED_HFQ.to_string()));
            }
            return (
                Some(LFM2_TEMPLATE.to_string()),
                Some("built-in fallback: LFM2 liquidai template".to_string()),
            );
        }
        12 => {
            if let Some(t) = hfq.chat_template_named("tool_use") {
                return (
                    Some(
                        t.replace("<|START_RESPONSE|>", "<|START_TEXT|>")
                            .replace("<|END_RESPONSE|>", "<|END_TEXT|>")
                            .replace("{{message.tool_plan}}", "{{ message.tool_plan or '' }}")
                            .replace("{{ tc['function']['name'] }}", "{{ tc.name }}")
                            .replace(
                                "{{ tc['function']['arguments']|tojson }}",
                                "{{ tc.arguments|tojson }}",
                            ),
                    ),
                    Some(
                        "embedded HFQ tokenizer_config.chat_template[tool_use] (rewritten for cohere2moe)"
                            .to_string(),
                    ),
                );
            }
        }
        _ => {}
    }
    let t = hfq.chat_template();
    let source = t.as_ref().map(|_| SRC_EMBEDDED_HFQ.to_string());
    (t, source)
}

/// Arch 5/6 template after configured/per-model overrides: embedded HFQ
/// `tokenizer_config.chat_template` when present, else froggeric fallback.
/// Ornith 1.5 basenames get a load-time Qwen3.8-shaped `reasoning_effort`
/// adaptation of the embedded (non-native) parent template; other models and
/// the froggeric missing-template path are left alone. Returns the template
/// and its provenance description.
fn qwen35_template_from_embedded(embedded: Option<String>, model_path: &str) -> (String, String) {
    match embedded {
        Some(t) if is_ornith15_artifact(model_path) => {
            let (t, adapted) = adapt_ornith15_embedded_chat_template(t);
            let source = if adapted {
                format!("{SRC_EMBEDDED_HFQ} (ornith-1.5 reasoning_effort adapter applied)")
            } else {
                SRC_EMBEDDED_HFQ.to_string()
            };
            (t, source)
        }
        Some(t) => (t, SRC_EMBEDDED_HFQ.to_string()),
        None => (
            FROGGERIC_QWEN35_TEMPLATE.to_string(),
            "built-in fallback: froggeric qwen35 template (artifact has no embedded chat_template)"
                .to_string(),
        ),
    }
}

/// Case-insensitive basename match for Ornith 1.5 MQ4/MQ4R artifacts.
/// Scope is exact: filename prefix `ornith-1.5-` or legacy `ornith1.5-`,
/// AND final extension case-insensitively exactly `.mq4` or `.mq4r`.
/// Other extensions (`.mq6`, `.hfq`, `.json`, sidecars) and bare names
/// are out of scope even when the Ornith 1.5 prefix matches.
fn is_ornith15_artifact(model_path: &str) -> bool {
    let basename = std::path::Path::new(model_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let lower = basename.to_ascii_lowercase();
    let prefix_ok = lower.starts_with("ornith-1.5-") || lower.starts_with("ornith1.5-");
    if !prefix_ok {
        return false;
    }
    // Path::extension is the final suffix after the last `.`.
    let ext = std::path::Path::new(basename)
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    ext == "mq4" || ext == "mq4r"
}

/// Qwen3.8 reasoning_effort system-prompt block (official chat_template.jinja).
const QWEN38_REASONING_BLOCK: &str = "\
{%- set reasoning_instructions = '' %}\n\
{%- if enable_thinking is undefined or enable_thinking is true %}\n\
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n\
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n\
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n\
    {%- endif %}\n\
    {%- if resolved_reasoning_effort == 'xhigh' %}\n\
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n\
    {%- elif resolved_reasoning_effort == 'low' %}\n\
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n\
    {%- endif %}\n\
{%- endif %}\n";

/// Unique marker: tools gate opening the system tools block.
const ORNITH_TOOLS_GATE: &str = "{%- if tools and tools is iterable and tools is not mapping %}";

/// Unique compound marker: tools system open immediately followed by the
/// `# Tools` header (no reasoning_instructions yet).
const ORNITH_TOOLS_SYSTEM_OPEN: &str = concat!(
    "{%- if tools and tools is iterable and tools is not mapping %}\n",
    "    {{- '<|im_start|>system\\n' }}\n",
    "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}"
);

/// Replacement: same tools open, with non-empty reasoning_instructions
/// prepended into the system turn (Qwen3.8 structure).
const ORNITH_TOOLS_SYSTEM_OPEN_ADAPTED: &str = concat!(
    "{%- if tools and tools is iterable and tools is not mapping %}\n",
    "    {{- '<|im_start|>system\\n' }}\n",
    "    {%- if reasoning_instructions %}\n",
    "        {{- reasoning_instructions + '\\n\\n' }}\n",
    "    {%- endif %}\n",
    "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}"
);

/// Unique marker: no-tools branch that only emits a system turn when a
/// system message exists (no effort injection, no no-system path).
const ORNITH_NO_TOOLS_SYSTEM_BRANCH: &str = concat!(
    "{%- else %}\n",
    "    {%- if messages[0].role == 'system' %}\n",
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
    "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n",
    "    {%- endif %}\n",
    "{%- endif %}"
);

/// Replacement: Qwen3.8 system/no-system logic with reasoning_instructions.
/// Double space before `+ content` matches the official Qwen3.8 template.
const ORNITH_NO_TOOLS_SYSTEM_BRANCH_ADAPTED: &str = concat!(
    "{%- else %}\n",
    "    {%- if messages[0].role == 'system' %}\n",
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
    "        {%- if content %}\n",
    "            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}\n",
    "        {%- elif reasoning_instructions %}\n",
    "            {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n",
    "        {%- endif %}\n",
    "    {%- elif reasoning_instructions %}\n",
    "        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n",
    "    {%- endif %}\n",
    "{%- endif %}"
);

/// Load-time adapter for Ornith 1.5 embedded Qwen3.5/3.6 templates: inject the
/// official Qwen3.8 `reasoning_effort` low/medium/xhigh system-prompt contract
/// via exact-marker rewrite. Already-native templates and marker drift are
/// returned unchanged so `probe_effort_capability` stays honest.
fn adapt_ornith15_embedded_chat_template(template: String) -> (String, bool) {
    if template.contains("reasoning_effort") {
        return (template, false);
    }
    match try_adapt_ornith15_qwen35_to_effort_native(&template) {
        Ok(adapted) => (adapted, true),
        Err(reason) => {
            eprintln!(
                "[chat_template] ornith-1.5 reasoning_effort adapter skipped ({reason}); keeping embedded template"
            );
            (template, false)
        }
    }
}

/// Atomic exact-marker rewrite. Preflights every required unique marker, then
/// applies all three substitutions. Any missing/duplicate marker → Err so the
/// caller returns the original string.
fn try_adapt_ornith15_qwen35_to_effort_native(template: &str) -> Result<String, String> {
    let gate_n = template.matches(ORNITH_TOOLS_GATE).count();
    if gate_n != 1 {
        return Err(format!("expected 1 tools gate marker, found {gate_n}"));
    }
    let tools_open_n = template.matches(ORNITH_TOOLS_SYSTEM_OPEN).count();
    if tools_open_n != 1 {
        return Err(format!(
            "expected 1 tools system-open marker, found {tools_open_n}"
        ));
    }
    let no_tools_n = template.matches(ORNITH_NO_TOOLS_SYSTEM_BRANCH).count();
    if no_tools_n != 1 {
        return Err(format!(
            "expected 1 no-tools system branch marker, found {no_tools_n}"
        ));
    }

    // Insert the reasoning block immediately before the tools gate.
    let mut out = template.replacen(
        ORNITH_TOOLS_GATE,
        &(QWEN38_REASONING_BLOCK.to_string() + ORNITH_TOOLS_GATE),
        1,
    );
    // Prepend non-empty instruction into the tools system turn.
    out = out.replacen(
        ORNITH_TOOLS_SYSTEM_OPEN,
        ORNITH_TOOLS_SYSTEM_OPEN_ADAPTED,
        1,
    );
    // Replace the no-tools system/no-system branch.
    out = out.replacen(
        ORNITH_NO_TOOLS_SYSTEM_BRANCH,
        ORNITH_NO_TOOLS_SYSTEM_BRANCH_ADAPTED,
        1,
    );

    // Honesty post-conditions: effort vocabulary + both instruction literals +
    // tools/system/no-system insertion branches present; original no-tools
    // emission gone.
    let has_effort_default = out.contains("reasoning_effort|default('xhigh')");
    let has_rungs = out.contains("('xhigh', 'medium', 'low')");
    let has_low = out.contains(
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
    );
    let has_xhigh = out.contains(
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    );
    // Jinja source stores `\n` as backslash+n inside quotes.
    let has_tools_prepend = out.contains("reasoning_instructions + '\\n\\n'");
    let has_no_system = out.contains("{%- elif reasoning_instructions %}");
    let original_no_tools_gone = !out.contains(ORNITH_NO_TOOLS_SYSTEM_BRANCH);
    if !(has_effort_default
        && has_rungs
        && has_low
        && has_xhigh
        && has_tools_prepend
        && has_no_system
        && original_no_tools_gone)
    {
        return Err("post-condition failed after marker rewrite".into());
    }
    Ok(out)
}

/// Rewrite the Onyx/Harmony chat template for Muse Glimmer (arch 14) so
/// its tool accessors match the flat `prompt_frame::ToolCall` (`tc.name`,
/// `tc.arguments`) and so a spliced cached tool body (`tc.rendered_body`)
/// replaces the regenerated ATEM text. Validates substitution counts and
/// fails loudly on upstream drift (a silent miss would revive the present
/// bug where the whole render raises and falls back to a bare unframed
/// prompt).
pub fn rewrite_muse_glimmer_onyx_template(template: &str) -> Result<String, String> {
    // Validate and count upstream accessor forms before mutating.
    let name_count = template.matches("tc.function.name").count();
    if name_count != 3 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 3 occurrences of `tc.function.name`, found {name_count}; upstream template drift?"
        ));
    }
    let args_count = template.matches("tc.function.arguments").count();
    if args_count != 1 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 1 occurrence of `tc.function.arguments`, found {args_count}; upstream template drift?"
        ));
    }
    let header = "{%- macro render_atem(tc) -%}";
    let header_count = template.matches(header).count();
    if header_count != 1 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 1 `render_atem` header, found {header_count}; upstream template drift?"
        ));
    }
    let tail = "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endmacro -%}";
    let tail_count = template.matches(tail).count();
    if tail_count != 2 {
        return Err(format!(
            "muse_glimmer Onyx template rewrite: expected 2 ATEM tails, found {tail_count}; upstream template drift?"
        ));
    }

    // Flat accessors.
    let mut out = template
        .replace("tc.function.name", "tc.name")
        .replace("tc.function.arguments", "tc.arguments");

    // Verbatim splice branch: if a cached body was spliced as `rendered_body`,
    // emit it directly, otherwise render the ATEM XML.
    out = out.replacen(
        header,
        &format!(
            "{}{}",
            header,
            "{%- if tc.rendered_body is defined and tc.rendered_body -%}{{- tc.rendered_body -}}{%- else -%}"
        ),
        1,
    );
    out = out.replacen(
        tail,
        "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endif -%}{%- endmacro -%}",
        1,
    );

    // Post-condition: no leftover nested accessor, branch present.
    if out.contains("tc.function.") {
        return Err(
            "muse_glimmer Onyx template rewrite: leftover `tc.function.` after rewrite".into(),
        );
    }
    if !out.contains("tc.rendered_body") {
        return Err(
            "muse_glimmer Onyx template rewrite: missing `tc.rendered_body` branch after rewrite"
                .into(),
        );
    }
    Ok(out)
}

pub(crate) fn parse_state_quant(
    mode: Option<&str>,
) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported state_quant '{other}' for the DeltaNet recurrent state; choose q8 (the default), fp32 or q4"
        )),
    }
}

// ─── Load functions ───────────────────────────────────────────────────

// ─── Core arch carrier load ─────────────────────────────────────────────

/// What the whole Qwen load left on the card, measured by
/// [`qwen35_post_load_fit`].
struct Qwen35PostLoadFit {
    /// Free VRAM after every load-time owner exists.
    free: usize,
    /// Per-request prefill transient: the minimum prefill reservation (incl.
    /// the widened-PBS headroom) plus VMM page slack.
    prefill: usize,
    /// DeltaNet checkpoint rings a request can fill ([`Qwen35LazyCharges`]).
    checkpoints: usize,
    /// Native-MTP per-request owners ([`Qwen35LazyCharges`]).
    mtp: usize,
    /// `prefill + checkpoints + mtp`: kept free of KV mapping.
    transient: usize,
    /// Largest context whose remaining KV mapping plus `transient` fits `free`.
    fit: usize,
    /// Bytes still to map for `cap` tokens (reported as a per-token rate).
    growth_at_cap: usize,
}

/// Owners a request creates lazily after the load, which no load-time
/// allocation shows up for but which a filled context must still find room
/// for (they came out of the KV room as `hipMalloc` failures at fill).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Qwen35LazyCharges {
    /// DeltaNet checkpoint rings: one charged `snapshot_bytes` per ring.
    checkpoints: usize,
    /// Native MTP `MtpSpecState` (first request) + prompt-fill transient.
    mtp: usize,
}

/// Bytes of the checkpoint rings one load can populate at once.
///
/// Each snapshot mirrors `dn_state` ([`DeltaNetSnapshot::bytes_for`]); only the
/// first of a ring is charged. `take_dn_checkpoint` adds further ones (up to
/// the cap, `HIPFIRE_CACHE_CKPT_MAX`) only from memory the admitted context
/// does not need (`Gpu::kv_fit_floor`). Two rings can be live together: the AR
/// prefill ring (`LoadedModel.prefill_checkpoints`, gated by
/// `HIPFIRE_CACHE_CKPT_RESUME` != 0, cap `ckpt_max()`) and, when a DFlash or
/// native-MTP speculator is attached, that speculator's own ring (gated by
/// `HIPFIRE_DFLASH_CKPT_RESUME`, i.e. `mtp_cache_policy().checkpoint_resume`,
/// cap = the same `HIPFIRE_CACHE_CKPT_MAX` knob). `ckpt_max()`,
/// `mtp_cache_policy().checkpoint_cap` and the DFlash builder all read that
/// one knob with default 8 and floor 1, so a single `cap` serves every ring.
fn qwen35_checkpoint_ring_bytes(snapshot_bytes: usize, speculator_ring: bool) -> usize {
    let policy = hipfire_config::mtp_cache_policy();
    let ar_ring = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_RESUME")
        .ok()
        .as_deref()
        != Some("0");
    let rings = usize::from(ar_ring) + usize::from(speculator_ring && policy.checkpoint_resume);
    snapshot_bytes.saturating_mul(rings)
}

/// [`Qwen35LazyCharges`] for this load. `mtp` is the native-MTP head (and the
/// requested chain depth) only when the MTP speculator is the one attached.
fn qwen35_lazy_charges(
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
    dn_state: &hipfire_arch_qwen35::qwen35::DeltaNetState,
    gpu: &Gpu,
    speculator_ring: bool,
    mtp: Option<(&hipfire_arch_qwen35::mtp_head::Qwen35MtpHead, usize)>,
) -> Result<Qwen35LazyCharges, String> {
    let snapshot_bytes = usize::try_from(DeltaNetSnapshot::bytes_for(dn_state))
        .map_err(|_| "DeltaNet snapshot sizing overflow".to_string())?;
    let checkpoints = qwen35_checkpoint_ring_bytes(snapshot_bytes, speculator_ring);
    let mtp = match mtp {
        None => 0,
        Some((head, requested_k)) => {
            use hipfire_arch_qwen35::mtp_spec::{
                mtp_head_fills_batched, mtp_prompt_fill_requested_bytes, MtpSpecState,
            };
            let max_n = hipfire_arch_qwen35::mtp_speculator::mtp_drafter_max_n(requested_k);
            let verify_capacity = hipfire_arch_qwen35::mtp_speculator::mtp_verify_capacity(max_n);
            let state = MtpSpecState::requested_bytes(
                &gpu.arch,
                config,
                snapshot_bytes,
                &head.config,
                head.weights.compressed_vocab_size,
                max_n,
                verify_capacity,
            )?;
            let fill = mtp_prompt_fill_requested_bytes(
                config,
                &head.config,
                mtp_head_fills_batched(head),
                hipfire_arch_qwen35::qwen35::prefill_max_batch(gpu),
            );
            state.saturating_add(fill)
        }
    };
    Ok(Qwen35LazyCharges { checkpoints, mtp })
}

/// Measured fit of the single-card Qwen VMM cache after the full load
/// (weights, KV owners' initial prefix, DeltaNet, scratch, drafts). The
/// carrier's pre-KV `card_cap` is provisional: it cannot charge what is
/// allocated afterwards. `free` honours the UMA clamp and the emulation cap.
/// `lazy` charges the per-request owners created after the load, so
/// `free - transient` is real KV room.
fn qwen35_post_load_fit(
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
    kv: &llama::KvCache,
    gpu: &Gpu,
    cap: usize,
    lazy: Qwen35LazyCharges,
) -> Result<Qwen35PostLoadFit, String> {
    let (free, _) = gpu
        .device_mem_info()
        .map_err(|e| format!("Qwen post-load VRAM query: {e}"))?;
    let minimum =
        hipfire_arch_qwen35::qwen35::prefill::minimum_prefill_reservation_bytes(config, &gpu.arch)
            .ok_or("Qwen minimum PBS sizing overflow")?;
    // Same page-rounding slack the carrier's `card_cap` keeps.
    let prefill = minimum.saturating_add(128 << 20);
    let transient = prefill
        .saturating_add(lazy.checkpoints)
        .saturating_add(lazy.mtp);
    let growth = |t: usize| {
        kv.planned_mapped_growth_bytes(gpu, t)
            .map_err(|e| format!("Qwen KV growth plan for {t} tokens: {e}"))
    };
    let fit = admission::qwen35_measured_fit(free, transient, cap, growth)?;
    let growth_at_cap = growth(cap)?;
    Ok(Qwen35PostLoadFit {
        free,
        prefill,
        checkpoints: lazy.checkpoints,
        mtp: lazy.mtp,
        transient,
        fit,
        growth_at_cap,
    })
}

/// Hard-error free for unfinished qwen35 finish path: bundle + optional VL.
///
/// `free_qwen35_bundle` returns every buffer to the GPU pool; only a drain
/// hands the VRAM back to the driver, and `unload_model` is normally the
/// one that drains. A load that fails here never reaches `unload_model`,
/// so without the drain the whole target (~15 GB on a 27B) stayed pooled —
/// the hw-gate Fable seat measured ~5 GB retained after a refused
/// `dflash_mode=on` load on top of the next resident model, compounding on
/// every lazy serve retry. Mirror `unload_model`: invalidate graphs, drain.
fn rollback_unfinished_qwen35(
    err: String,
    bundle: Qwen35Bundle,
    vision_weights: Option<qwen35_vl::VisionWeights>,
    gpu: &mut Gpu,
) -> String {
    let mut notes = Vec::new();
    if let Err(c) = hipfire_arch_qwen35::free_qwen35_bundle(bundle, gpu) {
        notes.push(c);
    }
    if let Some(vw) = vision_weights {
        vw.free_gpu(gpu);
    }
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    if notes.is_empty() {
        err
    } else {
        format!("{err}; cleanup also failed: {}", notes.join("; "))
    }
}

/// CASK / plain eviction setup. Only hard-error stage in finish_qwen35_load
/// before the bundle is published into LoadedModel.
fn build_qwen35_eviction(
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
    physical_cap: usize,
    activation_gate: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ctx: &mut LoadCtx,
) -> Result<Option<Eviction>, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    let Some(ref sidecar_path) = ctx.cask.sidecar else {
        return Ok(None);
    };
    let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
        use std::io::ErrorKind;
        let p = Path::new(sidecar_path);
        let why = match e.kind() {
            ErrorKind::NotFound if p.symlink_metadata().is_ok() => {
                format!("dangling symlink (target absent): {sidecar_path}")
            }
            ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
            ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
            ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
            _ => format!("read error ({e}): {sidecar_path}"),
        };
        format!(
            "cask sidecar load failed — {why} (CASK is deprecated and will be removed in 0.5.0; not supported; regen: hipfire sidecar-gen, or clear memory.cask.sidecar)"
        )
    })?;
    let fa_layer_ids: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if *t == LayerType::FullAttention {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if fa_layer_ids.is_empty() {
        eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
        return Ok(None);
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    let mut base = EvictionCtx::new(
        ctx.gpu,
        &centers,
        fa_layer_ids,
        ctx.cask.budget,
        ctx.cask.beta,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
        physical_cap,
    )
    .map_err(|e| format!("build EvictionCtx: {e}"))?;
    if let Some(gate) = activation_gate {
        base.set_activation_gate(gate);
    }
    if ctx.cask.cask_m_folding {
        eprintln!(
            "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
            ctx.cask.core_frac, ctx.cask.fold_m, ctx.cask.budget, ctx.cask.beta, physical_cap
        );
        Ok(Some(Eviction::Cask(CaskCtx::new(
            base,
            ctx.cask.core_frac,
            ctx.cask.fold_m,
        ))))
    } else {
        eprintln!(
            "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
            ctx.cask.budget, ctx.cask.beta, physical_cap
        );
        Ok(Some(Eviction::Plain(base)))
    }
}

/// Bundled `.mq4-mtp` trailer first, then the sidecar (`sidecar`, else the
/// trunk's sibling `.mtp`). A head whose hidden size or vocab differs from the
/// trunk is refused before upload. Returns the head, or the errors explaining
/// why none loaded (empty when no head exists); the caller decides whether a
/// missing head is fatal (`mtp=on`) or an AR fallback. `device` names the
/// device in the log line when the head does not live on the daemon's own GPU.
fn resolve_qwen35_mtp_head(
    trunk_path: &Path,
    sidecar: Option<std::path::PathBuf>,
    trunk_shape: (usize, usize),
    gpu: &mut rdna_compute::Gpu,
    physical_cap: usize,
    device: Option<&str>,
) -> (
    Option<hipfire_arch_qwen35::mtp_head::Qwen35MtpHead>,
    Vec<String>,
) {
    let (resolved, errors) = resolve_qwen35_mtp_head_sourced(
        trunk_path,
        sidecar,
        trunk_shape,
        gpu,
        physical_cap,
        device,
    );
    (resolved.map(|(head, _)| head), errors)
}

/// [`resolve_qwen35_mtp_head`] plus a short description of where the head
/// came from (`bundled <file>` / `sidecar <file>`), for the auto-attach notice.
fn resolve_qwen35_mtp_head_sourced(
    trunk_path: &Path,
    sidecar: Option<std::path::PathBuf>,
    (trunk_dim, trunk_vocab): (usize, usize),
    gpu: &mut rdna_compute::Gpu,
    physical_cap: usize,
    device: Option<&str>,
) -> (
    Option<(hipfire_arch_qwen35::mtp_head::Qwen35MtpHead, String)>,
    Vec<String>,
) {
    use hipfire_arch_qwen35::mtp_head;
    let tag = device.map(|d| format!(", {d}")).unwrap_or_default();
    let sidecar = sidecar.unwrap_or_else(|| trunk_path.with_extension("mtp"));
    let mut load = |path: &Path, offset: u64| {
        mtp_head::check_mtp_head_for_trunk(path, offset, trunk_dim, trunk_vocab)?;
        mtp_head::load_mtp_head_at_offset(path, gpu, physical_cap, offset)
            .map_err(|e| format!("{}: load failed: {e}", path.display()))
    };
    let file_name = |path: &Path| {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string())
    };
    let mut head_opt: Option<(mtp_head::Qwen35MtpHead, String)> = None;
    let mut errors: Vec<String> = Vec::new();
    match mtp_head::detect_bundled_mtp_offset(trunk_path) {
        Ok(Some(offset)) => match load(trunk_path, offset) {
            Ok(h) => {
                eprintln!(
                    "  MTP head loaded (bundled .mq4-mtp{tag}): n_embd={} vocab={}",
                    h.config.n_embd, h.config.vocab_size
                );
                head_opt = Some((h, format!("bundled {}", file_name(trunk_path))));
            }
            Err(e) => errors.push(format!("bundled trailer: {e}")),
        },
        Ok(None) => {}
        Err(e) => errors.push(format!("bundled trailer unreadable: {e}")),
    }
    if head_opt.is_none() && sidecar.exists() {
        match load(&sidecar, 0) {
            Ok(h) => {
                let after = if errors.is_empty() {
                    ""
                } else {
                    " after bundled error"
                };
                eprintln!(
                    "  MTP head loaded (sidecar {}{after}{tag}): n_embd={} vocab={}",
                    sidecar.display(),
                    h.config.n_embd,
                    h.config.vocab_size
                );
                head_opt = Some((h, format!("sidecar {}", file_name(&sidecar))));
                errors.clear();
            }
            Err(e) => errors.push(format!("sidecar {e}")),
        }
    }
    (head_opt, errors)
}

/// Build a `LoadedModel` from a carrier `Bundle`, shared fields, and
/// eviction/DFlash state. This is the common body for qwen35 dispatch
/// where eviction and DFlash need per-arch type info.
///
/// Hard errors before publish free the bundle and any preloaded VL weights.
fn finish_qwen35_load(
    bundle: Qwen35Bundle,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    physical_cap: usize,
    arch_id: u32,
    chat_template: Option<String>,
    chat_template_source: Option<String>,
    ctx: &mut LoadCtx,
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
) -> Result<LoadedModel, String> {
    // ── Eviction (only hard-error stage before publish) ────────────
    // Built before long-lived borrows so rollback can move `bundle`.
    let activation_gate = bundle
        .kv_adaptive
        .as_ref()
        .and_then(|adaptive| adaptive.eviction_gate());
    let eviction = match build_qwen35_eviction(&bundle.config, physical_cap, activation_gate, ctx) {
        Ok(e) => e,
        Err(e) => {
            return Err(rollback_unfinished_qwen35(
                e,
                bundle,
                vision_weights,
                ctx.gpu,
            ));
        }
    };

    // Extract references for DFlash/spec setup (borrow, don't move)
    let config = &bundle.config;
    let dn_state = &bundle.dn_state;

    // Adaptive KV cannot combine with generic (DSpark/DFlash/n-gram/MTP via
    // build_speculator) drafters. Suppress before any GPU drafter alloc.
    let adaptive_blocks_generic_spec = bundle.kv_adaptive.is_some();
    if adaptive_blocks_generic_spec {
        eprintln!(
            "  kv_adaptive engaged — suppressing generic speculator path (DSpark/DFlash/n-gram/bundled MTP-spec)"
        );
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::SpecOff,
            "kv_adaptive suppresses speculation",
        );
    }

    // ── DSpark sidecar (wins over DFlash/MTP/n-gram) ───────────────
    // The drafter is a dense-qwen3 body (llama crate); it drives the qwen35
    // ModelSlot target via the SpecTarget DSpark capture hooks. Discovered as
    // `<stem>-dspark.<ext>` next to the trunk, independent of ctx.draft_path.
    let dspark_speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> =
        if adaptive_blocks_generic_spec {
            None
        } else if ctx.spec.dspark != Some(false) {
            let base = std::path::Path::new(ctx.path);
            let sidecar_path = match (base.parent(), base.file_stem(), base.extension()) {
                (Some(parent), Some(stem), Some(ext)) => Some(parent.join(format!(
                    "{}-dspark.{}",
                    stem.to_string_lossy(),
                    ext.to_string_lossy()
                ))),
                _ => None,
            };
            match sidecar_path.filter(|p| p.exists()) {
                Some(p) => {
                    eprintln!("  qwen35: opening DSpark sidecar HFQ {p:?}");
                    match hipfire_runtime::hfq::HfqFile::open(&p) {
                        Ok(mut sidecar) => {
                            sidecar.drop_mmap();
                            match hipfire_arch_llama::dspark_body::load_qwen3_dspark(
                                &sidecar, ctx.gpu,
                            ) {
                                Ok(Some((dspark_weights, assets))) => {
                                    let block = dspark_weights.cfg.block_size;
                                    // Reduced-vocab drafters (ORNITH) ship a compressed
                                    // lm_head; run_heads reads vocab from lm_head.shape[0].
                                    let vocab = if dspark_weights.cfg.draft_vocab_size > 0 {
                                        dspark_weights.cfg.draft_vocab_size
                                    } else {
                                        assets.config.vocab_size
                                    };
                                    let stage_norm = assets.weights.output_norm.shallow_clone();
                                    // upload_raw sets dtype=Raw; the data is F16.
                                    let mut lm_head = assets.weights.output.buf.shallow_clone();
                                    lm_head.dtype = rdna_compute::DType::F16;
                                    lm_head.shape = vec![vocab];
                                    let conf_threshold = hipfire_config::developer_var(
                                        "HIPFIRE_QWEN35_DSPARK_CONF_THRESHOLD",
                                    )
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .or(ctx.spec.dspark_conf_threshold)
                                    .unwrap_or(0.1f32);
                                    eprintln!(
                                        "  qwen35 DSpark enabled (block={}, target_layers={:?}, draft_vocab={}, conf={:.2})",
                                        block,
                                        dspark_weights.cfg.target_layer_ids,
                                        vocab,
                                        conf_threshold
                                    );
                                    match hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
                                        assets,
                                        &dspark_weights.cfg,
                                        ctx.gpu,
                                    ) {
                                        Ok(body) => {
                                            hipfire_config::load_summary::note(
                                                hipfire_config::load_summary::Facet::SpecOn,
                                                format!("dspark (block {block})"),
                                            );
                                            Some(hipfire_runtime::dspark_core::build_dspark_speculator(
                                            body,
                                            dspark_weights,
                                            stage_norm,
                                            lm_head,
                                            block,
                                            physical_cap,
                                            conf_threshold,
                                            true, // sampled verify (temp>0) supported
                                            hipfire_runtime::dspark_block_controller::WindowCost::Measured { max_ratio: 0.5 },
                                        ))
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "  qwen35: DSpark body build failed: {e} — AR/other"
                                            );
                                            hipfire_config::load_summary::note(
                                                hipfire_config::load_summary::Facet::SpecOff,
                                                format!("DSpark body build failed: {e}"),
                                            );
                                            None
                                        }
                                    }
                                }
                                Ok(None) => {
                                    eprintln!(
                                        "  qwen35: DSpark sidecar {p:?} has no dspark_* metadata — skipping"
                                    );
                                    hipfire_config::load_summary::note(
                                        hipfire_config::load_summary::Facet::SpecOff,
                                        "DSpark sidecar failed: no dspark_* metadata",
                                    );
                                    None
                                }
                                Err(e) => {
                                    eprintln!("  qwen35: WARNING DSpark sidecar load failed: {e}");
                                    hipfire_config::load_summary::note(
                                        hipfire_config::load_summary::Facet::SpecOff,
                                        format!("DSpark sidecar failed: {e}"),
                                    );
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("  qwen35: WARNING cannot open DSpark sidecar {p:?}: {e}");
                            hipfire_config::load_summary::note(
                                hipfire_config::load_summary::Facet::SpecOff,
                                format!("DSpark sidecar failed: cannot open: {e}"),
                            );
                            None
                        }
                    }
                }
                None => None,
            }
        } else {
            None
        };

    // ── DFlash (skipped when a DSpark sidecar won) ─────────────────
    let dflash = if adaptive_blocks_generic_spec || dspark_speculator.is_some() {
        None
    } else if let Some(dp) = ctx.draft_path {
        match hipfire_arch_qwen35::dflash_spec::load_dflash_state(
            dp,
            physical_cap,
            config,
            dn_state,
            ctx.gpu,
            ctx.spec.ddtree_budget,
            ctx.spec.ddtree_topk,
            eviction.is_some(),
            &bundle.weights,
            // Exact plain Q8 KV (not asym/fwht variants). Matches design:
            // flags on the already-built cache — never re-resolve the string.
            {
                let kv = &bundle.kv_cache;
                kv.quant_q8
                    && !kv.quant_fwht
                    && !kv.quant_asym2
                    && !kv.quant_asym3
                    && !kv.quant_asym4
                    && matches!(kv.v_mode, llama::VMode::Q8)
            },
            bundle.kv_cache.uses_vmm_backend(),
            bundle.scratch.flash_partials.numel(),
            // finish_qwen35_load is the single-GPU carrier path.
            true,
            // Fail-closed: adaptive KV starts FWHT4 and tier-switches at runtime.
            // Upstream also suppresses DFlash when adaptive is Some; this keeps
            // admission honest if a future path reaches load_dflash_state.
            bundle.kv_adaptive.is_some(),
            ctx.spec
                .dflash_block
                .unwrap_or(hipfire_runtime::dflash::DFLASH_BLOCK_UNBATCHED),
        ) {
            Ok(s) => {
                eprintln!(
                    "  DFlash draft loaded: {} (layers={}, hidden={}, block={}, runtime_block={})",
                    dp,
                    s.draft_config.n_layers,
                    s.draft_config.hidden,
                    s.draft_config.block_size,
                    s.block_size
                );
                hipfire_config::load_summary::note(
                    hipfire_config::load_summary::Facet::SpecOn,
                    format!(
                        "dflash (draft {}, block {})",
                        Path::new(dp)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| dp.to_string()),
                        s.block_size
                    ),
                );
                Some(s)
            }
            Err(e) => {
                if ctx.spec.dflash == Some(true) {
                    return Err(rollback_unfinished_qwen35(
                        format!(
                            "DFlash draft required (speculation.dflash = on) but failed to load ({dp}): {e}"
                        ),
                        bundle,
                        vision_weights,
                        ctx.gpu,
                    ));
                }
                eprintln!(
                    "  DFlash draft load failed ({}): {} — falling back to AR only",
                    dp, e
                );
                hipfire_config::load_summary::note(
                    hipfire_config::load_summary::Facet::SpecOff,
                    format!("DFlash draft load failed: {e}"),
                );
                None
            }
        }
    } else {
        None
    };
    // MTP over a host spill: the `0xac000` SIGABRT recorded in `6273965e7`
    // (ornith-1.5:35b-a3b + moe_expert_budget=auto) no longer reproduces —
    // expert-spill (graph on/off) and layer-spill arms load the head and decode.
    // Upstream candidates in between (unbisected): `3b3d2ab45` MQ6 shared
    // gate/up alias, graded host-expert offload. So the hard refusal is gone:
    // explicit `speculation.mtp = "on"` (`--spec mtp`) attaches the head over a
    // spill. But `auto` still degrades to AR on the PCIe one, where the verify
    // re-reads host-mapped experts over the link and each extra pass can cost
    // more than the accepted tokens return. That is the ornith-1.5-35b-a3b
    // finding — `docs/perf-checkpoints/2026-10-04-mtp-over-trunk-expert-spill.md`
    // (gfx1201, greedy 9-prompt genre battery: MTP 34.9 vs AR 45.1 tok/s avg;
    // single merge_sort: 10.7 vs 44.6) and `…-moe-offload-ar-vs-mtp.md`
    // (MTP 23.4 pcie / 23.5 cpu vs AR 64.2 / 50.5) — both on the pre-CPU-stack
    // build, and both show the loss on *both* seams. Keyed on the *resolved*
    // placement (host-mapped count from the loaded weights), not on MoE-ness, so
    // a MoE that resolves fully resident keeps MTP under `auto`.
    //
    // The PCIe decline is therefore fixture-bound to those ornith records, and
    // the newest same-box record points the other way even there: on
    // `qwen3.6-35b-a3b.mq4p` (2026-10-11, 16 spilled expert layers) MTP led AR
    // on 8/8 cells, +13.9-22.4% pcie and +33.4-43.8% cpu. Extending or dropping
    // the decline needs a re-measure on the affected fixture, not this comment.
    //
    // CPU-deferred experts (memory.offload_exec=cpu) were excluded outright — the
    // splice recomputes the routed FFN on the host instead of re-reading it over
    // PCIe, and the CPU arms of that same 2026-10-11 record are its fastest MTP
    // cells. That exception is withdrawn: it armed the head with no opt-in on the
    // default state of an offloaded MoE, so a drafter that is unverified *on the
    // loaded placement* — or wrong there, as measured on `qwen3.6-35b-a3b.mq6`
    // (garbled verify windows at tau 0.04-0.13 while its single-row AR is
    // coherent) — spec-decoded silently. MTP is a gain only where it works, so
    // the CPU seam is opt-in like the PCIe one; those 2026-10-11 cells still
    // exist under `--spec mtp` / `speculation.mtp = "on"`. Resolved from the
    // loaded placement, not config.
    let cpu_deferred_moe = bundle.weights.has_cpu_deferred_experts();
    let (host_mapped_expert_blobs, host_mapped_expert_bytes) =
        hipfire_arch_qwen35::qwen35::host_mapped_expert_accounting(&bundle.weights.layers);
    if host_mapped_expert_blobs > 0 {
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Placement,
            format!(
                "experts: {host_mapped_expert_blobs} expert blobs ({:.1} GiB) host-mapped ({})",
                host_mapped_expert_bytes as f64 / (1u64 << 30) as f64,
                if cpu_deferred_moe {
                    "offload_exec=cpu"
                } else {
                    "offload_exec=pcie"
                },
            ),
        );
    }
    // Same shared policy as the slots arming site: `auto` (typed `None`)
    // declines over a host-mapped spill, `on` (`Some(true)`) arms, `off`
    // (`Some(false)`) stays AR via the gate below. qwen35's admitted list is
    // empty. Worded once in `hipfire_config::mtp_host_spill_decline_line`.
    let mtp_spill_mode = hipfire_config::MtpMode::from_tri_state(ctx.spec.mtp);
    // Producer capability gate first: qwen35's drafter declares no measured AR
    // floor (`hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR`), so `auto` keeps
    // plain AR and the head is never loaded (the retained AR spine keeps its
    // memory); only an explicit `on` arms it. A floor decline supersedes the
    // spill decline below (its line would misdescribe the cause).
    let mtp_floor_declined = matches!(arch_id, 5 | 6)
        && mtp_spill_mode == hipfire_config::MtpMode::Auto
        && !hipfire_config::mtp_mode_admits(
            mtp_spill_mode,
            hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR,
        );
    let mtp_spill_degrade = matches!(arch_id, 5 | 6)
        && !mtp_floor_declined
        && !hipfire_config::mtp_host_spill_policy(
            mtp_spill_mode,
            host_mapped_expert_blobs,
            ctx.gpu.arch.as_str(),
            &[],
        )
        .attach
        && !matches!(mtp_spill_mode, hipfire_config::MtpMode::Off);
    if mtp_spill_degrade {
        eprintln!(
            "{}",
            hipfire_config::mtp_host_spill_decline_line(
                "qwen35",
                host_mapped_expert_blobs,
                host_mapped_expert_bytes,
                if cpu_deferred_moe {
                    "offload_exec=cpu"
                } else {
                    "offload_exec=pcie"
                },
            )
        );
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::SpecOff,
            format!(
                "MTP off: {host_mapped_expert_blobs} expert blobs host-mapped ({:.2} GiB, {})",
                host_mapped_expert_bytes as f64 / (1u64 << 30) as f64,
                if cpu_deferred_moe {
                    "offload_exec=cpu"
                } else {
                    "offload_exec=pcie"
                },
            ),
        );
    }
    if mtp_floor_declined
        && !adaptive_blocks_generic_spec
        && eviction.is_none()
        && dflash.is_none()
        && dspark_speculator.is_none()
    {
        eprintln!("{}", hipfire_config::mtp_no_floor_decline_line("qwen35"));
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::SpecOff,
            "MTP off: no measured AR floor (speculation.mtp=auto)",
        );
    }
    // ── qwen35 MTP head (single resolver: bundled .mq4-mtp trailer then .mtp sidecar) ──
    // Precedence: DSpark > DFlash > MTP > n-gram. Gate: only arch 5/6, no adaptive/eviction,
    // and a typed mtp the producer-capability gate admits (`on`; `auto` only with a
    // declared measured AR floor, which qwen35 lacks, so `auto` never loads a head).
    // Bundled first, then the sidecar the CLI resolved
    // (`ctx.mtp_path`), else `<trunk>.mtp`; physical_cap is the KV window. A head
    // whose hidden size or vocab differs from the trunk is refused before upload.
    // On missing/refused/failed: error when forced on (mtp=on), auto logs/falls back.
    let mtp: Option<(hipfire_arch_qwen35::mtp_head::Qwen35MtpHead, String)> =
        if adaptive_blocks_generic_spec
            || eviction.is_some()
            || !matches!(arch_id, 5 | 6)
            || ctx.spec.mtp == Some(false)
            || mtp_spill_degrade
            || mtp_floor_declined
            || dflash.is_some()
            || dspark_speculator.is_some()
        {
            None
        } else {
            // `ctx.path` is canonical, so a sidecar beside a symlinked trunk is not
            // its sibling; the CLI resolves that case into `ctx.mtp_path`. Direct
            // daemon clients keep the canonical-sibling lookup.
            let (resolved, errors) = resolve_qwen35_mtp_head_sourced(
                Path::new(ctx.path),
                ctx.mtp_path.clone(),
                (config.dim, config.vocab_size),
                &mut *ctx.gpu,
                physical_cap,
                None,
            );
            if resolved.is_none() {
                if ctx.spec.mtp == Some(true) {
                    let reason = if errors.is_empty() {
                        "no bundled trailer or .mtp sidecar found".to_string()
                    } else {
                        errors.join("; ")
                    };
                    return Err(rollback_unfinished_qwen35(
                        format!("MTP head required (mtp=on) but not loaded: {reason}"),
                        bundle,
                        vision_weights,
                        ctx.gpu,
                    ));
                }
                if errors.is_empty() {
                    hipfire_config::load_summary::note(
                        hipfire_config::load_summary::Facet::SpecOff,
                        "MTP off: model has no MTP head (no bundled trailer or .mtp sidecar)",
                    );
                } else {
                    eprintln!(
                        "  MTP head not loaded: {} — falling back to AR/n-gram",
                        errors.join("; ")
                    );
                    hipfire_config::load_summary::note(
                        hipfire_config::load_summary::Facet::SpecOff,
                        format!("MTP head not loaded: {}", errors.join("; ")),
                    );
                }
            }
            resolved
        };
    if matches!(arch_id, 5 | 6) && ctx.spec.mtp == Some(false) {
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::SpecOff,
            "MTP off: speculation.mtp=off",
        );
    }
    if mtp.is_some() {
        if let Some(reason) = hipfire_arch_qwen35::qwen35::weights::native_mtp_refusal(
            &bundle.weights.layers,
            &ctx.gpu.arch,
            ctx.gpu.flags.fp16_disabled,
        ) {
            if let Some((head, _)) = mtp {
                head.free_gpu(ctx.gpu);
            }
            return Err(rollback_unfinished_qwen35(
                reason.to_string(),
                bundle,
                vision_weights,
                ctx.gpu,
            ));
        }
    }
    let mtp_present = mtp.is_some();
    let mtp = mtp.map(|(mut head, _source)| {
        // Exact gfx1100: draft over a fixed token subset sliced from the trunk
        // lm_head (`HIPFIRE_GFX1100_DRAFT_VOCAB=0` keeps the full-vocab draft).
        hipfire_arch_qwen35::draft_vocab::attach_trunk_draft_vocab(
            &mut *ctx.gpu,
            &mut head,
            &bundle.weights.output,
            &tokenizer,
        );
        head
    });
    // Auto-attached drafters (mode `auto`, not forced `on`) get one line that
    // names the artifact and the switch, so an A/B never starts with a hunt.
    let auto_draft = match (ctx.spec.dflash, &dflash, ctx.draft_path) {
        (None, Some(_), Some(dp)) => Some(
            Path::new(dp)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| dp.to_string()),
        ),
        _ => None,
    };
    if mtp_present && ctx.spec.mtp == Some(true) {
        eprintln!("  qwen35 MTP: armed explicitly (speculation.mtp=on / --spec mtp)");
    }
    // Drafts that actually loaded (named in the post-load fit line).
    let drafts_attached = {
        let mut names = Vec::new();
        if dflash.is_some() {
            names.push("dflash");
        }
        if mtp_present {
            names.push("mtp");
        }
        if dspark_speculator.is_some() {
            names.push("dspark");
        }
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join("+")
        }
    };
    // Per-request owners the speculator/AR paths create lazily, charged to the
    // post-load fit below. `build_speculator` order: DSpark > DFlash > MTP >
    // n-gram; the MTP head is only attached when none of the earlier arms is
    // and eviction is off (the same guard `build_speculator` re-asserts).
    let lazy_charges = {
        let spec_slot_free = !adaptive_blocks_generic_spec && dspark_speculator.is_none();
        let mtp_attaches = spec_slot_free && dflash.is_none() && eviction.is_none();
        let speculator_ring = spec_slot_free
            && eviction.is_none()
            && (dflash.is_some() || (mtp_attaches && mtp.is_some()));
        let mtp_sizing = if mtp_attaches && matches!(arch_id, 5 | 6) {
            mtp.as_ref()
                .map(|head| (head, crate::spec_build::mtp_requested_k(&ctx.spec)))
        } else {
            None
        };
        qwen35_lazy_charges(config, dn_state, &*ctx.gpu, speculator_ring, mtp_sizing)
    };
    // Pick the arch-generic speculator: a loaded DFlash draft → DflashSpeculator,
    // else an MTP head → MtpSpeculator<Qwen35MtpDrafter>, else (opt-in) the
    // model-free n-gram drafter. `eviction` is borrowed (not moved) here so it is
    // still available for the struct literal below; `config`/`dn_state` are
    // borrowed only for the n-gram arm's scratch construction. `None` ⇒ AR-only.
    // DSpark wins over DFlash/MTP/n-gram when its sidecar loaded.
    let speculator = if adaptive_blocks_generic_spec {
        None
    } else {
        dspark_speculator.or_else(|| {
            crate::spec_build::build_speculator(
                arch_id,
                dflash,
                mtp,
                eviction.is_none(),
                physical_cap,
                ctx.spec,
            )
        })
    };
    if let Some(name) = speculator.as_ref().map(|s| s.name()) {
        if let (Some(draft), "dflash") = (&auto_draft, name) {
            eprintln!(
                "DFlash: on (auto, draft {draft}) — disable with --spec off or speculation.dflash off"
            );
        }
    }

    // ── Measured post-load fit ─────────────────────────────────────
    // Every load-time owner (KV prefix, DeltaNet, scratch, DFlash/MTP/DSpark,
    // eviction) exists now, so free VRAM is the true room left for mapping KV
    // and for the per-request prefill transient. The carrier's pre-KV
    // `card_cap` could not charge any of it. Owners already built stay as
    // built (they are only larger than needed).
    let mut physical_cap = physical_cap;
    let mut fit_refusal: Option<String> = None;
    // A later (or non-Qwen) load must never inherit a previous load's floor.
    ctx.gpu.kv_fit_floor = None;
    if let Some(mut seq) = ctx.sequence {
        if bundle.kv_cache.uses_vmm_backend() {
            let lazy = lazy_charges.clone();
            match lazy.and_then(|lazy| {
                qwen35_post_load_fit(config, &bundle.kv_cache, &*ctx.gpu, physical_cap, lazy)
            }) {
                Err(e) => fit_refusal = Some(e),
                Ok(m) => {
                    let requested = ctx.max_seq;
                    let per_token = m.growth_at_cap / physical_cap.max(1);
                    let verdict = if seq.automatic {
                        if m.fit >= physical_cap {
                            format!("{requested} unchanged")
                        } else if m.fit < admission::QWEN35_MIN_AUTO_CONTEXT {
                            fit_refusal = Some(format!(
                                "Qwen VMM load refused: after the weights, drafts ({drafts_attached}) and scratch loaded, {} tokens fit (free {} B, per-request reserve {} B = prefill {} + DeltaNet checkpoints {} + MTP {}), below the minimum automatic context {}; free GPU memory, disable the draft (speculation.dflash off), or use a smaller quantization of this model",
                                m.fit,
                                m.free,
                                m.transient,
                                m.prefill,
                                m.checkpoints,
                                m.mtp,
                                admission::QWEN35_MIN_AUTO_CONTEXT
                            ));
                            format!("{requested} -> refused")
                        } else {
                            physical_cap = m.fit;
                            ctx.max_seq = ctx.max_seq.min(m.fit);
                            seq.card_cap = m.fit;
                            ctx.sequence = Some(seq);
                            format!("{requested} -> {}", m.fit)
                        }
                    } else if requested.min(physical_cap) > m.fit {
                        let message = format!(
                            "max_seq {requested} does not fit this card: after the weights, drafts and scratch loaded, {} tokens fit (free {} B, per-request reserve {} B = prefill {} + DeltaNet checkpoints {} + MTP {}); lower memory.max_seq to at most {} or omit it for automatic sizing",
                            m.fit, m.free, m.transient, m.prefill, m.checkpoints, m.mtp, m.fit
                        );
                        if hipfire_config::oom_guard_mode() == hipfire_config::OomGuardMode::Off {
                            eprintln!("warning: {message} (memory.oom_guard off: loading anyway)");
                            format!("{requested} kept (exceeds fit {}; oom_guard off)", m.fit)
                        } else {
                            fit_refusal = Some(message);
                            format!("{requested} -> refused")
                        }
                    } else {
                        format!("{requested} unchanged")
                    };
                    eprintln!(
                        "max_seq fit: measured after full load: free={} transient={} (prefill {} + checkpoints {} + mtp {}) kv_mapped_growth_per_token≈{per_token} drafts={drafts_attached} ; context {verdict}",
                        m.free, m.transient, m.prefill, m.checkpoints, m.mtp
                    );
                    hipfire_config::load_summary::note(
                        hipfire_config::load_summary::Facet::ContextFit,
                        match verdict.strip_suffix(" unchanged") {
                            Some(_) => "measured fit unchanged".to_string(),
                            None => format!("measured fit {verdict}"),
                        },
                    );
                    // Further DeltaNet checkpoints yield to the admitted context.
                    ctx.gpu.kv_fit_floor = Some(rdna_compute::KvFitFloor {
                        admitted_tokens: ctx.max_seq.min(physical_cap),
                        reserve_bytes: m.transient,
                    });
                }
            }
        }
    }

    // Move adaptive controller out of the bundle before parking the rest in
    // `Box<dyn ArchModel>`. LoadedModel.kv_adaptive is the runtime home for downshift hooks.
    let mut bundle = bundle;
    bundle.vision_config = vision_config;
    bundle.vision_weights = vision_weights;
    let kv_adaptive = bundle.kv_adaptive.take();
    let state: Option<Box<dyn hipfire_runtime::arch_model::ArchModel>> = Some(Box::new(bundle));
    let mut model = LoadedModel {
        state,
        eviction,
        speculator,
        max_seq: ctx.max_seq,
        kv_adaptive,
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            ctx.max_seq,
            physical_cap,
            ctx.path.to_string(),
            chat_template,
            chat_template_source,
        )
    };
    model.mtp_weights_present = mtp_present;
    if let Some(message) = fit_refusal {
        // The speculator, eviction and bundle are all owned by `model` now:
        // the regular unload path frees every one of them.
        return Err(match unload_model(model, ctx.gpu) {
            Ok(()) => message,
            Err(e) => format!("{message}; cleanup also failed: {e}"),
        });
    }
    Ok(model)
}

// ─── Main public API ──────────────────────────────────────────────────

/// gfx11 + gfx12 targets with WMMA-backed DFlash batched lm_head GEMM paths.
fn is_dflash_lm_head_wmma_arch(gpu_arch: &str) -> bool {
    matches!(
        gpu_arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
    )
}

/// Pre-allocation predicate: whether a target lm_head quant_type is admitted
/// for DFlash on `gpu_arch`. Always admits qt 3/6/13; admits legacy qt17 and
/// V2 qt44/47/48/49/50 on the gfx11+gfx12 WMMA set only.
fn dflash_lm_head_quant_supported(lm_qt: Option<u8>, gpu_arch: &str) -> bool {
    match lm_qt {
        Some(3 | 6 | 13) => true,
        Some(17 | 44 | 47 | 48 | 49 | 50) => is_dflash_lm_head_wmma_arch(gpu_arch),
        _ => false,
    }
}

/// Load a model from an HFQ file (or safetensors directory). This is the
/// single arch-dispatch point via the carrier registry.
#[allow(clippy::too_many_arguments)]
pub fn load_model(
    path: &str,
    max_seq: usize,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    load_model_with_kv_backend(
        path,
        max_seq,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        draft_path,
        kv_mode_override,
        None,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
    )
}

/// Load a model with an optional per-load KV storage backend override.
///
/// Head overlays (`--head`) are admission-only: they validate and attach in
/// `admit_source` before teardown, so this pre-admission entry point cannot
/// carry one — use the `admit_source` → `load_admitted_*` route instead.
#[allow(clippy::too_many_arguments)]
pub fn load_model_with_kv_backend(
    path: &str,
    max_seq: usize,
    deepseek4_experts_per_token: Option<usize>,
    deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_backend_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    load_model_with_gemma4_drafter(
        path,
        max_seq,
        deepseek4_experts_per_token,
        deepseek4_compute_placement,
        draft_path,
        None,
        None,
        GEMMA4_EAGLE_DRAFT_LEN,
        kv_mode_override,
        None,
        None,
        kv_backend_override,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
    )
}

/// Load a model with Gemma4 EAGLE drafter support. Mirrors
/// `load_model_with_kv_backend` but threads `gemma4_drafter_path` /
/// `gemma4_draft_len` (params.drafter / params.spec) separately from
/// `draft_path` (Qwen DFlash) so a DFlash .hfq can never be routed into the
/// EAGLE loader by accident. When `gemma4_drafter_path` is `None` the Gemma4
/// eager path is AR-only.
#[allow(clippy::too_many_arguments)]
pub fn load_model_with_gemma4_drafter(
    path: &str,
    max_seq: usize,
    deepseek4_experts_per_token: Option<usize>,
    deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    draft_path: Option<&str>,
    head_path: Option<&str>,
    gemma4_drafter_path: Option<&str>,
    gemma4_draft_len: usize,
    kv_mode_override: Option<&str>,
    kv_k_override: Option<&str>,
    kv_v_override: Option<&str>,
    kv_backend_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    // Validate draft_len early (refuse-don't-degrade, same rule as daemon).
    let _ = gemma4_eagle_spec_len(Some(gemma4_draft_len as u64))
        .map_err(|e| format!("gemma4 drafter: {e}"))?;
    let admission = crate::admission::admit_source_with_options(
        path,
        1, // this entry serves tp<=1
        pp,
        crate::admission::KvBackendRequest::from_override(kv_backend_override)?,
        draft_path,
        gpu.arch.as_str(),
        None,
        // No per-request vision_mode on this entry point — preserve the
        // sibling probe (auto).
        "auto",
        head_path,
        max_seq,
        crate::admission::KvBackendHints {
            kv_mode: kv_mode_override,
            kv_k: kv_k_override,
            kv_v: kv_v_override,
            qwen_default_q8: crate::admission::qwen_default_q8_enabled(),
            kv_adaptive: kv_adaptive_override,
            cask: Some(cask),
            deepseek4_heterogeneous: !matches!(
                deepseek4_compute_placement,
                hipfire_config::Deepseek4ComputePlacement::Single
            ),
            vmm_runtime_available: gpu.vmm_recommended_granularity().is_ok(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        },
        crate::admission::SourceAdmissionOptions {
            spec,
            kv_adaptive: crate::admission::qwen4_kv_adaptive_requested(kv_adaptive_override),
            eagle_drafter: gemma4_drafter_path.is_some(),
            cask: cask.sidecar.is_some(),
            state_quant: state_quant_override.is_some(),
            non_single_compute: !matches!(
                deepseek4_compute_placement,
                hipfire_config::Deepseek4ComputePlacement::Single
            ),
            expert_count_override: deepseek4_experts_per_token.is_some(),
            pflash: false,
        },
    )?;
    let warning = crate::admission::legacy_warning(
        admission.kv_backend,
        admission.kv_backend_reason.as_deref(),
    );
    let loaded = load_admitted_with_gemma4_drafter(
        admission,
        path,
        max_seq,
        deepseek4_experts_per_token,
        deepseek4_compute_placement,
        draft_path,
        gemma4_drafter_path,
        gemma4_draft_len,
        None,
        kv_mode_override,
        kv_k_override,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        spec,
        gpu,
    );
    if loaded.is_ok() {
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
    }
    loaded
}

// ─── Planned-route kernel preload ────────────────────────────────────

/// Host RAM a compile worker never takes: the shared 1-GiB safety reserve.
const ROUTE_COMPILE_SAFETY_RESERVE: u64 = 1 << 30;
/// Extra host RAM kept free while workers overlap the weight load, for the
/// loader's own read/staging/upload buffers.
const ROUTE_LOAD_STAGING_RESERVE: u64 = 2 << 30;

/// Effective load policy the kernel route planner needs, resolved from the
/// admitted request before any GPU allocation.
struct RouteRequest<'a> {
    path: &'a str,
    source: &'a ModelSource,
    pp: usize,
    max_seq: usize,
    draft_path: Option<&'a str>,
    mtp_path: Option<&'a Path>,
    vision_path: Option<&'a Path>,
    vision_mode: &'a str,
    kv_mode_override: Option<&'a str>,
    kv_k_override: Option<&'a str>,
    kv_v_override: Option<&'a str>,
    qwen_default_q8: bool,
    kv_adaptive_override: Option<&'a str>,
    state_quant_override: Option<&'a str>,
    cask: &'a CaskConfig,
    spec: SpecLoadCfg,
    kv_backend: KvBackend,
}

/// A begun route, checked against the loaded model before it finishes.
/// `spec` is the speculative route its closed plan was resolved for; `None`
/// for a predicted-family fallback (no closed plan): that route is always
/// open and nothing of the model can leave it.
struct PlannedRoute {
    spec: Option<rdna_compute::kernel_registry::KernelSpecRoute>,
    predicted: Option<PredictedRoute>,
}

/// The predicted-family set a route compiles beyond (or instead of) a closed
/// plan, summarised for the load log.
struct PredictedRoute {
    family: &'static str,
    basis: &'static str,
    modules: usize,
}

/// Resolve this load's kernel route plan from header metadata and the
/// carrier's own policy resolvers. `Err` is the refusal reason: the route is
/// outside every closed plan and keeps lazy kernel loading.
fn resolve_route_plan(
    req: &RouteRequest<'_>,
    gpu: &Gpu,
) -> Result<
    (
        rdna_compute::kernel_registry::RouteKernelPlan,
        rdna_compute::kernel_registry::KernelSpecRoute,
        rdna_compute::kernel_registry::RouteSeal,
    ),
    String,
> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    use rdna_compute::kernel_registry::{KernelSpecRoute, KernelTensorMeta, RouteKernelInput};
    let ModelSource::Hfq(hfq) = req.source else {
        return Err("safetensors directory source".into());
    };
    // Developer kill switch, and how route walks (scripts/route_plan_walk.py)
    // observe a route's whole lazy closure without the plan they regenerate.
    // Qwen3.5 resolves its axes (and logs them for the walk) before refusing.
    let route_plan_off = !hipfire_config::developer_bool("HIPFIRE_ROUTE_PLAN", true);
    const ROUTE_PLAN_OFF: &str = "HIPFIRE_ROUTE_PLAN=0";
    match hfq.arch_id {
        // Qwen3.5 dense (5) and MoE (6: A3B, Qwen3.6, Ornith) share the qwen35
        // carrier: the hand-walked 27B closure and the walked plans share this
        // policy.
        5 | 6 => {}
        _ if route_plan_off => return Err(ROUTE_PLAN_OFF.into()),
        hipfire_arch_qwen4::ARCH_ID => return resolve_qwen4_route_plan(req, gpu, hfq),
        other => return Err(format!("model arch {other}")),
    }
    if req.pp != 1 {
        return Err(format!("pp={}", req.pp));
    }
    if req.cask.sidecar.is_some() {
        return Err("CASK/TriAttention eviction sidecar".into());
    }
    let adaptive = req
        .kv_adaptive_override
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_adaptive.clone());
    if !matches!(adaptive.as_str(), "" | "off") {
        return Err(format!("kv_adaptive={adaptive}"));
    }
    if req.draft_path.is_some() {
        return Err("DFlash draft".into());
    }
    if req.spec.ngram_draft == Some(true) {
        return Err("n-gram drafter".into());
    }
    // `finish_qwen35_load` gives a `<stem>-dspark.<ext>` sibling precedence.
    let base = Path::new(req.path);
    if req.spec.dspark != Some(false) {
        if let (Some(parent), Some(stem), Some(ext)) =
            (base.parent(), base.file_stem(), base.extension())
        {
            let sidecar = parent.join(format!(
                "{}-dspark.{}",
                stem.to_string_lossy(),
                ext.to_string_lossy()
            ));
            if sidecar.exists() {
                return Err("DSpark sidecar".into());
            }
        }
    }
    // The tower's kernels are outside the closed route plan whether the tower
    // is resident or streamed, so every non-`off` mode that resolves a tower
    // (`auto`, `resident`, `on`) falls back to the ordinary loader.
    if req.vision_path.is_some()
        || hfq
            .find_tensor_info("model.visual.patch_embed.proj.weight")
            .is_some()
        || (req.vision_mode != "off"
            && hipfire_runtime::sidecar::resolve_vl_sidecar(req.path).is_some())
    {
        return Err("vision tower".into());
    }
    let config =
        <hipfire_arch_qwen35::Qwen35 as hipfire_runtime::arch::Architecture>::config_from_hfq(hfq)
            .map_err(|e| e.to_string())?;
    // Same K/V resolution as the qwen35 HFQ carrier (adaptive/CASK refused above).
    let mode_raw = req
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
    let auto_native = kv_mode::qwen35_native_auto_eligible(
        gpu.arch.as_str(),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        req.pp,
        false,
        false,
    );
    let policy =
        kv_mode::qwen35_policy_for_native(&kv_mode::QWEN35_HFQ_POLICY, &mode_raw, auto_native);
    let pair = kv_mode::resolve_kv_pair(
        &mode_raw,
        req.kv_k_override.filter(|s| !s.is_empty()),
        req.kv_v_override.filter(|s| !s.is_empty()),
        &policy,
        gpu.arch.as_str(),
        req.qwen_default_q8,
    )
    .map_err(|e| e.to_string())?;
    let (kv_k, kv_v) = match pair {
        kv_mode::KvPair::Native(k) => {
            let name = kv_mode::qwen_k_display_name(k);
            (name, name)
        }
        kv_mode::KvPair::Split(k, v) => (
            kv_mode::qwen_k_display_name(k),
            kv_mode::qwen_v_display_name(v),
        ),
    };
    let state_quant = match parse_state_quant(req.state_quant_override)? {
        StateQuant::Q8 => "q8",
        StateQuant::FP32 => "fp32",
        StateQuant::Q4 => "q4",
    };
    // Native MTP head, in the loader's order: bundled trailer, then sidecar.
    // A head that later fails its trunk check falls back to AR, a subset of
    // the MTP plan. The plan applies the loader's producer-capability gate:
    // `auto` without a declared measured AR floor (qwen35's case) loads no
    // head, so the plan is the AR route.
    let head = if !hipfire_config::mtp_mode_admits(
        hipfire_config::MtpMode::from_tri_state(req.spec.mtp),
        hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR,
    ) {
        None
    } else {
        match hipfire_arch_qwen35::mtp_head::detect_bundled_mtp_offset(base) {
            Ok(Some(offset)) => Some(HfqFile::open_at_offset(base, offset)),
            Ok(None) => {
                let sidecar = req
                    .mtp_path
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| base.with_extension("mtp"));
                sidecar.exists().then(|| HfqFile::open(&sidecar))
            }
            Err(e) => return Err(format!("bundled MTP trailer unreadable: {e}")),
        }
        .transpose()
        .map_err(|e| format!("MTP head header: {e}"))?
    };
    let spec = if head.is_some() {
        KernelSpecRoute::NativeMtp
    } else {
        KernelSpecRoute::Ar
    };
    let mut owned = hipfire_runtime::loader_api::header_tensor_meta(hfq);
    if let Some(head) = &head {
        owned.extend(hipfire_runtime::loader_api::header_tensor_meta(head));
    }
    let tensors: Vec<KernelTensorMeta<'_>> = owned
        .iter()
        .map(|t| KernelTensorMeta {
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
        })
        .collect();
    // Routed-expert placement: a closed plan is the fully resident one. The
    // pager, a REAP keep-map and any configured spill (host-placed experts or
    // layers, which run other kernels) have no closed route.
    if config.paged_experts {
        return Err("paged experts".into());
    }
    if config.reap_keep.is_some() {
        return Err("REAP keep-map".into());
    }
    {
        use hipfire_config::memory::{effective_offload_budgets, OffloadBudget};
        if effective_offload_budgets() != (OffloadBudget::Full, OffloadBudget::Full) {
            return Err("memory.gpu_layer_budget / memory.moe_expert_budget spill".into());
        }
    }
    // The walked-plan axes: every resolved policy a walk must have covered
    // exactly (`#! axis` lines). Names match `scripts/route_plan_build.py`.
    let prefill_chunk_rows = qwen35::prefill::prefill_chunk_rows_requested(gpu);
    let axes = [
        ("kv_k", kv_k.to_owned()),
        ("kv_v", kv_v.to_owned()),
        ("state_quant", state_quant.to_owned()),
        ("kv_backend", req.kv_backend.as_str().to_owned()),
        ("prefill_chunk_rows", prefill_chunk_rows.to_string()),
        // MoE shape the tensor layout does not pin (top-k), and the resident
        // placement the refusals above guarantee; dense models walk 0 / 0.
        ("num_experts", config.num_experts.to_string()),
        ("experts_per_tok", config.num_experts_per_tok.to_string()),
        ("expert_placement", "resident".to_owned()),
    ];
    // Developer-only: the resolved route facts `route_plan_build.py` reads
    // from a walk's stderr (the Qwen3.5 counterpart of `qwen4 state:`).
    if hipfire_config::developer_var("HIPFIRE_MODULE_LOAD_AUDIT").as_deref() == Ok("1") {
        let resolved: Vec<String> = axes
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        eprintln!(
            "  route plan axes: arch={} model_arch={} spec={spec:?} heads={},{},{} max_seq={} {}",
            gpu.arch,
            hfq.arch_id,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            req.max_seq,
            resolved.join(" ")
        );
    }
    if route_plan_off {
        return Err(ROUTE_PLAN_OFF.into());
    }
    let (plan, seal) = rdna_compute::kernel_registry::route_entries_sealed(&RouteKernelInput {
        arch: gpu.arch.as_str(),
        model_arch: hfq.arch_id,
        tensors: &tensors,
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: req.max_seq,
        prefill_chunk_rows,
        kv_k,
        kv_v,
        state_quant,
        spec,
        host_mapped_experts: false,
        // Closed plans cover both direct and retained-replay decode.
        retained_decode: true,
        hipcc_extra_flags: &gpu.flags.hipcc_extra_flags,
        flags: &gpu.flags,
        route_axes: &axes,
    })
    .map_err(|e| format!("{e:?}"))?;
    Ok((plan, spec, seal))
}

/// Flash-Next (Qwen4) route plan: the walked plan (`kernel_registry`
/// `route_plans/`) for this header layout and exact arch, resolved with the
/// carrier's own policy resolvers. Every option the walks did not cover
/// refuses here, so the load keeps lazy kernel loading.
fn resolve_qwen4_route_plan(
    req: &RouteRequest<'_>,
    gpu: &Gpu,
    hfq: &HfqFile,
) -> Result<
    (
        rdna_compute::kernel_registry::RouteKernelPlan,
        rdna_compute::kernel_registry::KernelSpecRoute,
        rdna_compute::kernel_registry::RouteSeal,
    ),
    String,
> {
    use rdna_compute::kernel_registry::{KernelSpecRoute, KernelTensorMeta, RouteKernelInput};
    use rdna_compute::tensor_ops::QsaKvFormat;
    if req.pp != 1 {
        return Err(format!("pp={}", req.pp));
    }
    let ep = hipfire_config::topology::qwen4_ep()?;
    if ep > 1 {
        return Err(format!("expert parallelism (topology.ep = {ep})"));
    }
    if req.draft_path.is_some() || req.cask.sidecar.is_some() {
        return Err("a drafter or CASK sidecar".into());
    }
    if crate::admission::qwen4_kv_adaptive_requested(req.kv_adaptive_override) {
        return Err("adaptive KV".into());
    }
    if hipfire_runtime::config::retained_redline_default(
        &gpu.arch, "qwen4", req.path, req.pp, 1, ep, true,
    ) {
        return Err("retained Redline route".into());
    }
    let default_k = hipfire_config::mtp_k_default(
        hipfire_arch_qwen4::ARCH_ID,
        rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&gpu.arch),
    );
    if let Some(k) = req
        .spec
        .mtp_k
        .or(hipfire_runtime::config::get().mtp_k)
        .filter(|&k| k != default_k)
    {
        return Err(format!("MTP K={k} (walked K={default_k})"));
    }
    let native_mtp =
        crate::admission::qwen4_native_mtp(req.spec, &gpu.arch, req.path, req.pp, 1, ep)?;
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
        .map_err(|error| error.to_string())?;
    if hipfire_arch_qwen4::resolve_trunk_overlay(
        hfq,
        Path::new(req.path),
        &gpu.arch,
        &receipt.config,
    )?
    .is_some()
    {
        return Err("trunk overlay".into());
    }
    let kv_mode = req
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
    let state = hipfire_arch_qwen4::resolve_state_format(
        &kv_mode,
        req.state_quant_override.unwrap_or(""),
        gpu,
        &receipt.config,
    )?;
    // Names as the load log prints them (`qwen4 state:` line), which the
    // route walk records.
    let qsa = match state.qsa {
        QsaKvFormat::Fp8 => "fp8",
        QsaKvFormat::Fp8Fnuz => "fp8-fnuz",
        QsaKvFormat::Q8 => "q8",
        QsaKvFormat::F32 => "f32",
    };
    let on_off = |on: bool| if on { "on" } else { "off" }.to_owned();
    let axes = [
        ("qsa", qsa.to_owned()),
        (
            "kv_backend",
            hipfire_arch_qwen4::Qwen4KvBackend::from(req.kv_backend)
                .name()
                .to_owned(),
        ),
        ("gdn", state.gdn.name().to_owned()),
        ("qsa_f16_arena", on_off(state.qsa_f16_arena)),
        ("qsa_pooled_bf16", on_off(state.qsa_pooled_bf16)),
    ];
    let spec = if native_mtp {
        KernelSpecRoute::NativeMtp
    } else {
        KernelSpecRoute::Ar
    };
    let owned = hipfire_runtime::loader_api::header_tensor_meta(hfq);
    let tensors: Vec<KernelTensorMeta<'_>> = owned
        .iter()
        .map(|t| KernelTensorMeta {
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
        })
        .collect();
    let (plan, seal) = rdna_compute::kernel_registry::route_entries_sealed(&RouteKernelInput {
        arch: gpu.arch.as_str(),
        model_arch: hfq.arch_id,
        tensors: &tensors,
        n_heads: receipt.config.num_attention_heads,
        n_kv_heads: receipt.config.num_key_value_heads,
        head_dim: receipt.config.head_dim,
        max_seq: req.max_seq,
        prefill_chunk_rows: 0,
        kv_k: qsa,
        kv_v: qsa,
        state_quant: state.gdn.name(),
        spec,
        host_mapped_experts: false,
        retained_decode: true,
        hipcc_extra_flags: &gpu.flags.hipcc_extra_flags,
        flags: &gpu.flags,
        route_axes: &axes,
    })
    .map_err(|e| format!("{e:?}"))?;
    Ok((plan, spec, seal))
}

/// User-facing note for the lazy kernel path: a cold kernel cache compiles
/// each kernel at its first launch, which is the slow first run users see.
const LAZY_KERNELS_NOTE: &str = "compiling kernels on first use (one-time, cached afterwards)…";

/// Start a route's kernel load: the bounded batch compile overlaps the
/// weight load when the host budget admits a worker alongside it; otherwise
/// the whole plan is compiled and loaded before any weight allocation. A
/// load no budget admits, or any planned or predicted module that fails to
/// compile, fails here, before destructive work and never at the first
/// request.
///
/// Everything predictable is compiled before the model is loaded: a refused
/// route (no closed plan) and an open walked plan both add the model
/// family's predicted module set (the arch's whole installer inventory
/// minus the plan; the family trace only chooses which symbols bind; see
/// `kernel_registry::predicted_kernels`). That coverage is never a closure:
/// the route finishes open, never sealed. `HIPFIRE_ROUTE_PLAN=0` (a route
/// walk) returns `None` and loads lazily with no preload, so the walk
/// records the real dispatch. `HIPFIRE_ROUTE_PLAN_SEAL=1` (a sealed replay)
/// loads only the exact plan, with no predicted extras, and seals it; a
/// refused route then stays lazy so the replay fails loudly.
fn begin_route_kernels(
    req: &RouteRequest<'_>,
    gpu: &mut Gpu,
) -> Result<Option<PlannedRoute>, String> {
    use rdna_compute::compile_jobs::{
        compile_job_budget, host_available_memory, parse_job_override,
    };
    use rdna_compute::kernel_registry::{PredictKernelInput, RouteKernelPlan, RouteSeal};
    gpu.abort_route_kernel_load()
        .map_err(|e| format!("kernel route reset: {e}"))?;
    // Resolve first: a walk (`HIPFIRE_ROUTE_PLAN=0`) still needs the axes
    // line the resolver logs, then loads lazily with nothing preloaded.
    let resolved = resolve_route_plan(req, gpu);
    if !hipfire_config::developer_bool("HIPFIRE_ROUTE_PLAN", true) {
        eprintln!(
            "  kernels: {LAZY_KERNELS_NOTE} [HIPFIRE_ROUTE_PLAN=0: route walk, nothing preloaded]"
        );
        return Ok(None);
    }
    let sealed_replay = hipfire_config::developer_bool("HIPFIRE_ROUTE_PLAN_SEAL", false);
    let (plan, spec, mut open, refusal) = match resolved {
        Ok((plan, spec, _)) if sealed_replay => (plan, Some(spec), None, None),
        Ok((plan, spec, RouteSeal::Sealed)) => (plan, Some(spec), None, None),
        Ok((plan, spec, RouteSeal::Open(family))) => (
            plan,
            Some(spec),
            Some(format!("{family} {} {spec:?}", gpu.arch)),
            None,
        ),
        Err(reason) if sealed_replay => {
            eprintln!(
                "  kernels: {LAZY_KERNELS_NOTE} [HIPFIRE_ROUTE_PLAN_SEAL=1: no closed route plan: {reason}]"
            );
            return Ok(None);
        }
        Err(reason) => (
            RouteKernelPlan {
                entries: Vec::new(),
            },
            None,
            None,
            Some(reason),
        ),
    };
    // A sealed plan is an exact closure and takes no speculative extras.
    let predicted = if open.is_none() && refusal.is_none() {
        None
    } else {
        let model_arch = match req.source {
            ModelSource::Hfq(hfq) => Some(hfq.arch_id),
            _ => None,
        };
        match rdna_compute::kernel_registry::predicted_kernels(
            &PredictKernelInput {
                arch: gpu.arch.as_str(),
                model_arch,
                hipcc_extra_flags: &gpu.flags.hipcc_extra_flags,
                pm_decode: gpu.flags.pm_decode,
                // Only an exact recognised plan lets the family trace choose binds.
                exact_plan: refusal.is_none(),
            },
            &plan,
        ) {
            Ok(predicted) => Some(predicted),
            Err(e) => {
                eprintln!(
                    "  kernels: no predicted kernel set for {} ({e:?})",
                    gpu.arch
                );
                None
            }
        }
    };
    if let Some(reason) = &refusal {
        // No registry inventory for this arch: nothing is predictable.
        let Some(predicted) = &predicted else {
            eprintln!("  kernels: {LAZY_KERNELS_NOTE} [no closed route plan: {reason}]");
            return Ok(None);
        };
        open = Some(format!(
            "{} {} predicted fallback",
            predicted.family, gpu.arch
        ));
    }
    let summary = predicted.as_ref().map(|p| PredictedRoute {
        family: p.family,
        basis: p.basis.label(),
        modules: p.modules.len(),
    });
    let predicted_note = predicted.as_ref().map_or_else(String::new, |p| {
        format!(
            " + {} predicted {} modules [{}] ({} symbols in {} modules bound, {} compile-only, {} omitted)",
            p.modules.len(),
            p.family,
            p.basis.label(),
            p.bound_symbols(),
            p.bound_modules(),
            p.modules.len() - p.bound_modules(),
            p.omitted.len(),
        )
    });
    let route_note = match (&spec, &refusal) {
        (Some(spec), _) => format!("{spec:?} route plan ({} modules)", plan.entries.len()),
        (None, Some(reason)) => format!("no closed route plan ({reason}); predicted fallback"),
        (None, None) => unreachable!("a refusal or a plan"),
    };
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .map_err(|e| format!("cannot determine compile CPU affinity: {e}"))?;
    let override_jobs = parse_job_override(
        "HIPFIRE_JIT_JOBS",
        hipfire_config::developer_var("HIPFIRE_JIT_JOBS")
            .ok()
            .as_deref(),
    )?;
    let available = host_available_memory();
    let overlap = compile_job_budget(
        cores,
        available,
        ROUTE_COMPILE_SAFETY_RESERVE + ROUTE_LOAD_STAGING_RESERVE,
        override_jobs,
    );
    let started = std::time::Instant::now();
    match overlap {
        Ok(jobs) => {
            gpu.begin_route_kernel_load_predicted(plan, predicted, jobs, open)
                .map_err(|e| format!("kernel route compile: {e}"))?;
            eprintln!(
                "  kernels: {route_note}{predicted_note}, compiling with up to {jobs} jobs during weight load"
            );
        }
        Err(overlap) => {
            let jobs = compile_job_budget(cores, available, ROUTE_COMPILE_SAFETY_RESERVE, override_jobs)
                .map_err(|e| {
                    format!("load refused: not enough free host memory to start even one GPU-kernel compile worker ({overlap}; before weights: {e}). Free some system memory and retry")
                })?;
            gpu.begin_route_kernel_load_predicted(plan, predicted, jobs, open)
                .and_then(|()| gpu.ensure_route_modules_preloaded())
                .map_err(|e| {
                    let _ = gpu.abort_route_kernel_load();
                    format!("kernel route compile: {e}")
                })?;
            eprintln!(
                "  kernels: {route_note}{predicted_note} loaded before weights with {jobs} jobs in {:.2}s ({overlap})",
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(Some(PlannedRoute {
        spec,
        predicted: summary,
    }))
}

/// Log how a route finished: where its kernels were compiled (before ready
/// versus lazily during load) and the open-route contract after ready.
fn log_open_route(
    route: &str,
    predicted: Option<&PredictedRoute>,
    stats: rdna_compute::scratch::RouteKernelStats,
) {
    match predicted {
        Some(predicted) => eprintln!(
            "  kernels: {route} ready, open (never sealed): {}/{} predicted {} modules [{}] compiled before ready \
             ({} symbols in {} modules bound, {} compile-only); {} unplanned kernel requests during load \
             (counts requests, not compiles: includes binds of already compiled or embedded modules); \
             off-plan loads after ready stay lazy, warn and are counted",
            stats.predicted_compiled,
            predicted.modules,
            predicted.family,
            predicted.basis,
            stats.predicted_bound_symbols,
            stats.predicted_bound_modules,
            stats.predicted_compiled.saturating_sub(stats.predicted_bound_modules),
            stats.lazy_during_load,
        ),
        None => eprintln!(
            "  kernels: {route} preloaded (open: off-plan loads stay lazy and warn); \
             {} unplanned kernel requests during load (counts requests, not compiles)",
            stats.lazy_during_load
        ),
    }
}

/// Finish a begun route once the model loaded: every planned and predicted
/// module is compiled and on the GPU before ready. A sealed route then
/// refuses an unplanned kernel; an open one (walked plans and every
/// predicted fallback) loads it lazily, warns and counts it. A model that
/// resolved outside a closed plan still completes its predictable
/// compilation and finishes open, never sealed, keeping the counters.
fn finish_route_kernels(
    planned: PlannedRoute,
    model: &LoadedModel,
    gpu: &mut Gpu,
) -> Result<(), String> {
    use rdna_compute::kernel_registry::KernelSpecRoute;
    let route = match planned.spec {
        Some(spec) => format!("{spec:?} route"),
        None => "predicted fallback".to_owned(),
    };
    let mismatch = match planned.spec {
        None => None,
        Some(spec) => {
            if model.kv_adaptive.is_some() {
                Some("adaptive KV engaged")
            } else if model.eviction.is_some() {
                Some("eviction engaged")
            } else if model.speculator.is_some()
                && !(spec == KernelSpecRoute::NativeMtp && model.mtp_weights_present)
            {
                Some("a speculator outside the plan")
            } else {
                None
            }
        }
    };
    if let Some(why) = mismatch {
        eprintln!("  kernels: {route} left the planned route: {why}; completing its predictable kernels, route stays open");
        gpu.finish_route_kernel_load_open(format!("{route} (left the plan: {why})"))
            .map_err(|e| format!("kernel route preload: {e}"))?;
        log_open_route(&route, planned.predicted.as_ref(), gpu.route_kernel_stats());
        return Ok(());
    }
    gpu.finish_route_kernel_load()
        .map_err(|e| format!("kernel route preload: {e}"))?;
    if gpu.route_kernels_sealed() {
        eprintln!("  kernels: {route} preloaded and sealed");
        return Ok(());
    }
    log_open_route(&route, planned.predicted.as_ref(), gpu.route_kernel_stats());
    Ok(())
}

/// The load-time KHO scratch warning line: only on a unified-memory GPU, and
/// only at [`rdna_compute::uma_memory::KHO_WARN_BYTES`] or more.
fn kho_scratch_load_warning(uma: bool, scratch: Option<u64>) -> Option<String> {
    if !uma {
        return None;
    }
    rdna_compute::uma_memory::kho_scratch_warning(scratch?).map(|text| format!("  warning: {text}"))
}

/// Print [`kho_scratch_load_warning`] at most once per process (every load
/// path of the daemon and `hipfire serve` goes through
/// [`load_admitted_with_gemma4_drafter`]).
fn warn_kho_scratch_once(uma: bool) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    if !uma {
        return;
    }
    WARNED.call_once(|| {
        if let Some(line) =
            kho_scratch_load_warning(uma, rdna_compute::uma_memory::kho_scratch_bytes())
        {
            eprintln!("{line}");
        }
    });
}

/// Consume an already-admitted source: the retained [`SourceAdmission`] handle
/// plus its resolved carrier. Destructive work — VMM readiness, carrier load
/// with its allocations and collectives — begins here, only after admission has
/// succeeded. `mtp_path` is the CLI-resolved Qwen MTP sidecar (`params.mtp`);
/// `None` keeps the loader's `<trunk>.mtp` lookup.
#[allow(clippy::too_many_arguments)]
pub fn load_admitted_with_gemma4_drafter(
    admission: crate::admission::SourceAdmission,
    path: &str,
    _max_seq: usize,
    deepseek4_experts_per_token: Option<usize>,
    deepseek4_compute_placement: hipfire_config::Deepseek4ComputePlacement,
    draft_path: Option<&str>,
    gemma4_drafter_path: Option<&str>,
    gemma4_draft_len: usize,
    mtp_path: Option<&Path>,
    kv_mode_override: Option<&str>,
    kv_k_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    spec: SpecLoadCfg,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let crate::admission::SourceAdmission {
        source,
        kv_backend,
        qwen_default_q8,
        kv_v,
        max_seq,
        sequence,
        carrier,
        vision_path,
        vision_mode,
        pp_layer_counts,
        mesh,
        pp_bands,
        batch_request,
        ..
    } = admission;
    let carrier =
        carrier.ok_or_else(|| "single/pp admission must resolve a carrier".to_string())?;
    // The admitted-context floor is per load: a later (non-Qwen) load never inherits it.
    gpu.kv_fit_floor = None;
    ensure_vmm_ready_for_load(gpu)?;
    warn_kho_scratch_once(gpu.is_uma());
    let rec_sampling = match &source {
        ModelSource::Hfq(hfq) => hfq.recommended_sampling(),
        _ => None,
    };
    // Planner-closed routes start their bounded kernel compile now, overlapped
    // with the non-dispatching weight load, and preload every planned module
    // before the first dispatch. Refused routes keep lazy loading.
    let route_begin_started = std::time::Instant::now();
    let planned_route = begin_route_kernels(
        &RouteRequest {
            path,
            source: &source,
            pp,
            max_seq,
            draft_path,
            mtp_path,
            vision_path: vision_path.as_deref(),
            vision_mode: &vision_mode,
            kv_mode_override,
            kv_k_override,
            kv_v_override: kv_v.as_deref(),
            qwen_default_q8,
            kv_adaptive_override,
            state_quant_override,
            cask,
            spec,
            kv_backend,
        },
        gpu,
    )?;
    rdna_compute::load_phases::add(
        "kernels.route_begin",
        rdna_compute::load_phases::ns_since(route_begin_started),
        0,
    );
    rdna_compute::load_phases::note(
        "kernels",
        if planned_route.is_some() {
            "route_plan"
        } else {
            "lazy"
        },
    );
    let mut ctx = LoadCtx {
        path,
        max_seq,
        sequence: sequence
            .as_ref()
            .map(|s| hipfire_runtime::loader_api::SequenceHint {
                model_ctx: s.model_ctx,
                automatic: s.bound != "user",
                card_cap: 0,
                reserve_bytes: 0,
            }),
        deepseek4_compute_placement,
        deepseek4_experts_per_token,
        draft_path,
        vision_path,
        mtp_path: mtp_path.map(Path::to_path_buf),
        vision_mode,
        kv_mode_override,
        kv_k_override,
        kv_v_override: kv_v.as_deref(),
        qwen_default_q8,
        kv_backend,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        pp_layer_counts: pp_layer_counts.as_deref(),
        admitted_mesh: pp_bands.as_ref().map(|_| &mesh),
        band_placement: pp_bands.as_ref(),
        spec,
        gpu,
        gemma4_drafter_path,
        gemma4_draft_len,
        // Slice 4 (integration) populates this from the daemon `xdna` param
        // after flag + exact-gfx1151 admission; until then loads stay GPU-only.
        xdna: None,
        batch_request,
    };
    let mut result = match carrier.load(source, &mut ctx) {
        Ok(result) => result,
        Err(error) => {
            // The carrier already unwound its resources; join any compile
            // workers so none outlives the refused load.
            return Err(match ctx.gpu.abort_route_kernel_load() {
                Ok(()) => error,
                Err(e) => format!("{error}; kernel compile cleanup also failed: {e}"),
            });
        }
    };
    if let Some(planned) = planned_route {
        let route_wait_started = std::time::Instant::now();
        let finished = finish_route_kernels(planned, &result, ctx.gpu);
        rdna_compute::load_phases::add(
            "kernels.route_wait",
            rdna_compute::load_phases::ns_since(route_wait_started),
            0,
        );
        if let Err(error) = finished {
            return Err(match unload_model(result, ctx.gpu) {
                Ok(()) => error,
                Err(e) => format!("{error}; cleanup also failed: {e}"),
            });
        }
    }
    result.sequence = sequence.map(|mut s| {
        if let Some(measured) = ctx.sequence {
            s.card_cap = measured.card_cap;
            s.max_seq = result.max_seq;
            s.bound = if !measured.automatic {
                "user"
            } else if measured.model_ctx <= measured.card_cap {
                "model"
            } else {
                "card"
            };
        }
        s
    });
    if result.pp > 1 && result.pp_gpus.is_none() {
        return Err("pp>1 LoadedModel missing pp_gpus — carrier bug".into());
    }
    if let Some(rec) = rec_sampling {
        result.rec_temperature = rec.temperature;
        result.rec_top_p = rec.top_p;
        result.rec_top_k = rec.top_k.map(|k| k as f32);
    }
    Ok(result)
}

fn load_cohere2moe(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let config = <cohere2moe::Cohere2Moe as Architecture>::config_from_hfq(&hfq)?;
    let weights = <cohere2moe::Cohere2Moe as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = cohere2moe::Cohere2MoeState::new_with_max_seq(gpu, &config, max_seq)
        .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("<|END_OF_TURN_TOKEN|>")
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(255001)
    };
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        state: Some(Box::new(Cohere2MoeBundle {
            config,
            weights,
            state,
            eos_tok,
        })),
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}

// ─── MMQ screening ────────────────────────────────────────────────────

// ─── EP load functions ────────────────────────────────────────────────

/// EP partial-load fault injector. Reads `HIPFIRE_EP_FAIL_RANK` so the GPU
/// cleanup test can force a deterministic mid-load failure after a given rank and
/// assert the staging guard reclaimed every loaded rank's VRAM. Gated behind the
/// `ep-fault-inject` feature: production/default builds compile the `None` stub
/// below, so a stray `HIPFIRE_EP_FAIL_RANK` in the environment can NEVER fail a
/// real EP load.
#[cfg(feature = "ep-fault-inject")]
fn ep_fail_rank() -> Option<usize> {
    match hipfire_config::developer_var("HIPFIRE_EP_FAIL_RANK").ok() {
        Some(s) if !s.is_empty() => s.parse::<usize>().ok(),
        _ => None,
    }
}

#[cfg(not(feature = "ep-fault-inject"))]
fn ep_fail_rank() -> Option<usize> {
    None
}

/// Staging guard for the ds4 EP load (transactional partial-load cleanup). Owns
/// the `Gpus` orchestrator plus the per-rank weights / state / partials as they
/// are built up. If the load fails mid-way (a `?` early return, or the
/// `HIPFIRE_EP_FAIL_RANK` fault), `Drop` explicitly frees every rank's VRAM
/// (weights → state → partial) and drains each device's pool, so a failed EP load
/// leaks NO VRAM. On success the caller calls `into_parts()` to disarm the guard
/// and move ownership into the `LoadedModel`.
struct Ds4EpStaging {
    /// `Option` so `into_parts` can move the `Gpus` out on success without a
    /// placeholder. `None` after a successful disarm.
    gpus: Option<Gpus>,
    weights: Vec<deepseek4::DeepseekV4Weights>,
    state: Vec<deepseek4::DeepseekV4State>,
    partials: Vec<rdna_compute::GpuTensor>,
    prefill: Vec<deepseek4::forward::PrefillBatchScratch>,
}

impl Ds4EpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
            prefill: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<deepseek4::DeepseekV4Weights>,
        Vec<deepseek4::DeepseekV4State>,
        Vec<rdna_compute::GpuTensor>,
        Vec<deepseek4::forward::PrefillBatchScratch>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        let prefill = std::mem::take(&mut self.prefill);
        (gpus, weights, state, partials, prefill)
    }
}

impl Drop for Ds4EpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP ds4 load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for (r, s) in self.state.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                s.free_gpu(dev);
            }
        }
        for (r, p) in self.partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for (r, pbs) in self.prefill.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                pbs.free_gpu(dev);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Staging guard for the MiniMax EP load — mirror of `Ds4EpStaging` with the
/// MiniMax weight/state types.
struct MinimaxEpStaging {
    gpus: Option<Gpus>,
    weights: Vec<minimax::MiniMaxWeights>,
    state: Vec<minimax::MiniMaxState>,
    partials: Vec<rdna_compute::GpuTensor>,
}

impl MinimaxEpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            state: Vec::new(),
            partials: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<minimax::MiniMaxWeights>,
        Vec<minimax::MiniMaxState>,
        Vec<rdna_compute::GpuTensor>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let state = std::mem::take(&mut self.state);
        let partials = std::mem::take(&mut self.partials);
        (gpus, weights, state, partials)
    }
}

impl Drop for MinimaxEpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP minimax load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for (r, s) in self.state.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                s.free_gpu(dev);
            }
        }
        for (r, p) in self.partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
    }
}
/// Staging guard for the Qwen3.5 EP load — mirror of `MinimaxEpStaging` with
/// the Qwen weight types plus the sequential per-rank decode state (KV,
/// DeltaNet, scratch, partials). Owns the `Gpus` orchestrator plus per-rank
/// weights / state as they are built up. If the load fails mid-way (`?`
/// early return, or the `HIPFIRE_EP_FAIL_RANK` fault), `Drop` explicitly
/// frees every rank's VRAM on its owning device and drains each device's
/// pool, so a failed EP load leaks NO VRAM and publishes NO partial object.
/// On success the caller calls `into_parts()` to disarm the guard and move
/// ownership into the `LoadedModel`.
struct Qwen35EpStaging {
    gpus: Option<Gpus>,
    weights: Vec<qwen35::Qwen35Weights>,
    kv_caches: Vec<llama::KvCache>,
    dn_states: Vec<qwen35::DeltaNetState>,
    scratches: Vec<qwen35::Qwen35Scratch>,
    partials: Vec<rdna_compute::GpuTensor>,
    prefill_pbs: Vec<qwen35::PrefillBatchScratch>,
    prefill_partials: Vec<rdna_compute::GpuTensor>,
}

impl Qwen35EpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            kv_caches: Vec::new(),
            dn_states: Vec::new(),
            scratches: Vec::new(),
            partials: Vec::new(),
            prefill_pbs: Vec::new(),
            prefill_partials: Vec::new(),
        }
    }
    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("staging gpus taken")
    }
    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<qwen35::Qwen35Weights>,
        Vec<llama::KvCache>,
        Vec<qwen35::DeltaNetState>,
        Vec<qwen35::Qwen35Scratch>,
        Vec<rdna_compute::GpuTensor>,
        Vec<qwen35::PrefillBatchScratch>,
        Vec<rdna_compute::GpuTensor>,
    ) {
        let gpus = self.gpus.take().expect("into_parts called twice");
        let weights = std::mem::take(&mut self.weights);
        let kv_caches = std::mem::take(&mut self.kv_caches);
        let dn_states = std::mem::take(&mut self.dn_states);
        let scratches = std::mem::take(&mut self.scratches);
        let partials = std::mem::take(&mut self.partials);
        let prefill_pbs = std::mem::take(&mut self.prefill_pbs);
        let prefill_partials = std::mem::take(&mut self.prefill_partials);
        (
            gpus,
            weights,
            kv_caches,
            dn_states,
            scratches,
            partials,
            prefill_pbs,
            prefill_partials,
        )
    }
}

impl Drop for Qwen35EpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        eprintln!(
            "[loader] EP qwen35 load failed — freeing {} partially-loaded rank(s) (no VRAM leak)",
            self.weights.len()
        );
        for (r, s) in self.scratches.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = s.free_gpu(dev);
            }
        }
        for (r, s) in self.dn_states.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                s.free_gpu(dev);
            }
        }
        for (r, kv) in self.kv_caches.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = kv.free_gpu(dev);
            }
        }
        for (r, p) in self.partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for (r, pbs) in self.prefill_pbs.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = pbs.free_gpu(dev);
            }
        }
        for (r, p) in self.prefill_partials.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                let _ = dev.free_tensor(p);
            }
        }
        for (r, w) in self.weights.drain(..).enumerate() {
            if let Some(dev) = gpus.devices.get_mut(r) {
                let _ = dev.bind_thread();
                w.free_gpu(dev);
            }
        }
        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Transactional owner for dense Qwen TP construction. Every vector is kept
/// rank-aligned; an early return frees only the objects that were published
/// into the guard, on their owning device.
struct Qwen35DenseTpStaging {
    gpus: Option<Gpus>,
    weights: Vec<qwen35::Qwen35Weights>,
    kv_caches: Vec<llama::KvCache>,
    dn_states: Vec<qwen35::DeltaNetState>,
    scratches: Vec<qwen35::Qwen35Scratch>,
}

impl Qwen35DenseTpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            weights: Vec::new(),
            kv_caches: Vec::new(),
            dn_states: Vec::new(),
            scratches: Vec::new(),
        }
    }

    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("dense TP staging gpus taken")
    }

    #[allow(clippy::type_complexity)]
    fn into_parts(
        mut self,
    ) -> (
        Gpus,
        Vec<qwen35::Qwen35Weights>,
        Vec<llama::KvCache>,
        Vec<qwen35::DeltaNetState>,
        Vec<qwen35::Qwen35Scratch>,
    ) {
        (
            self.gpus.take().expect("dense TP into_parts called twice"),
            std::mem::take(&mut self.weights),
            std::mem::take(&mut self.kv_caches),
            std::mem::take(&mut self.dn_states),
            std::mem::take(&mut self.scratches),
        )
    }
}

impl Drop for Qwen35DenseTpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        for (rank, scratch) in self.scratches.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                let _ = scratch.free_gpu(gpu);
            }
        }
        for (rank, state) in self.dn_states.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                state.free_gpu(gpu);
            }
        }
        for (rank, kv) in self.kv_caches.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                let _ = kv.free_gpu(gpu);
            }
        }
        for (rank, weights) in self.weights.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                let _ = gpu.bind_thread();
                weights.free_gpu(gpu);
            }
        }
        // Best-effort: reclaim peer-rooted reduce scratch reserved pre-enable_peer_all
        // so a failed dense-TP load cannot strand VRAM after peer mapping.
        let _ = gpus.free_peer_reduce_scratch();
        for gpu in &mut gpus.devices {
            let _ = gpu.bind_thread();
            gpu.invalidate_weight_caches();
            gpu.invalidate_graph_state();
            gpu.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Supported-topology validation for Qwen3.5-MoE under expert-parallel load.
/// Pure so the contract is unit-testable. Sequential EP serves tp=2|4 where
/// the sealed per-rank loader admits: the rank count must divide `num_experts`
/// (else `ShardConfig` cannot build the Stride map and `forward_ep` has no
/// owner for every expert). Dense Qwen3.5 (`num_experts == 0`) is unaffected
/// and keeps its dense-TP path. Refuse here, before `Gpus::init_ep` and the
/// per-rank weight upload, instead of after a full multi-rank load — the same
/// predicate gates both admission (before teardown) and load (backstop), so
/// the text cannot drift between the two.
pub fn qwen35_ep_moe_topology_refusal(
    arch_id: u32,
    num_experts: usize,
    tp: usize,
) -> Option<String> {
    if num_experts == 0 {
        return None;
    }
    if tp != 2 && tp != 4 {
        return Some(format!(
            "Qwen3.5-MoE (arch_id={arch_id}) EP supports tp=2|4, got tp={tp}"
        ));
    }
    if num_experts % tp != 0 {
        return Some(format!(
            "Qwen3.5-MoE (arch_id={arch_id}) expert parallelism: tp={tp} does not divide num_experts={num_experts}; choose a tp that divides the expert count, or set tp=1"
        ));
    }
    None
}

/// Message constructor shared by [`ep_admission`] and the per-entry match
/// backstops so the refusal text cannot drift between the two.
pub(crate) fn ep_unsupported_arch_message(arch_id: u32) -> String {
    format!(
        "expert parallelism (tp>1) is not supported for arch_id={arch_id}; it works for Qwen3.5 (5|6), DeepSeek V4 (9) and MiniMax (10). Set tp=1 for this model"
    )
}

/// EP load admission by arch_id. Only archs with an `EpArch` variant may enter
/// expert-parallel load (9/DeepSeek4, 10/MiniMax, 5|6/Qwen3.5); LFM2 (11),
/// Cohere2 (12) and anything else must fail here — right after the host-side
/// HFQ probe, before any device init — otherwise they would reach
/// `generate_ep` with no correct server. Pure so the contract is
/// unit-testable; both `load_model_ep_*` entries call it before dispatching.
pub fn ep_admission(arch_id: u32) -> Result<(), String> {
    match arch_id {
        5 | 6 | 9 | 10 => Ok(()),
        id => Err(ep_unsupported_arch_message(id)),
    }
}

/// Expert-parallel (EP) model load — shards the routed experts across `tp` ranks
/// (`Gpus::init_ep` + per-arch sharded weight load), wrapped in a staging guard so
/// a mid-load failure frees every already-loaded rank's VRAM (no leak, prior model
/// at the call site left intact). ds4 (arch_id 9) and MiniMax (arch_id 10) only.
///
/// KNOWN RESIDUAL — constructor-mid-failure leak (scoped follow-up, NOT fixed):
/// the staging guard frees every rank that has been COMPLETED and `push`ed, so a
/// failure BETWEEN ranks leaks no VRAM. But a failure INSIDE a single rank's
/// constructor — after it uploaded some tensors but before it returns `Ok` —
/// leaks those partial allocations (`GpuTensor` has no `Drop`). The fault injector
/// (`HIPFIRE_EP_FAIL_RANK`) fires AFTER a rank's constructor returns `Ok`, so it
/// tests the completed-rank cleanup path (which IS fixed), not this inner window.
/// The proper fix is an unwind-safe allocation-tracking loader refactor. Deferred.
pub fn load_model_ep(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    load_model_ep_with_compressor_cache(
        path,
        max_seq,
        tp,
        hipfire_config::Deepseek4CompressorCache::F32,
    )
}

/// DeepSeek V4 stores its long-lived compressor cache as F32 or F16 only; it
/// has no block-quantised path. Rather than reject the quantised selectors,
/// map every sub-F16 request up to F16 — the nearest storage the model
/// actually implements, and the one whose intent ("smaller than F32") the
/// caller expressed. Redirecting rather than failing keeps `q8`, the historic
/// DS4 default, and the `asym`/`fwht`/`turbo` families usable, and it is a
/// widening in precision terms: F16 carries more of the value than any of
/// them would have.
///
/// This is deliberately not silent. A caller that asked for `q8` and received
/// F16 storage is running a different configuration from the F32 golden, so
/// the redirect is reported once at resolve time.
fn resolve_deepseek4_compressor_cache_kv_mode(
    kv_mode: Option<&str>,
) -> Result<hipfire_config::Deepseek4CompressorCache, String> {
    let raw = kv_mode.unwrap_or("f32").to_ascii_lowercase();
    match raw.as_str() {
        "" | "auto" | "f32" => Ok(hipfire_config::Deepseek4CompressorCache::F32),
        "f16" => Ok(hipfire_config::Deepseek4CompressorCache::F16),
        "q8" | "asym2" | "asym3" | "asym4" | "fwht2" | "fwht3" | "fwht4" | "turbo" | "turbo3"
        | "turbo4" => {
            eprintln!(
                "[deepseek4] kv_cache={raw} has no DeepSeek V4 implementation; \
                 using the F16 compressor cache instead (nearest supported storage \
                 below F32). Pass --kv f32 for the golden configuration."
            );
            Ok(hipfire_config::Deepseek4CompressorCache::F16)
        }
        other => Err(format!(
            "DeepSeek V4 kv_cache={other} is not recognised; use f32 (golden/default) or f16"
        )),
    }
}

/// EP load using the standard user-facing KV selector. DeepSeek V4 maps f32/f16
/// to its long-lived compressor-cache storage; MiniMax retains its historical
/// EP behavior and does not inherit DeepSeek-specific policy.
pub fn load_model_ep_with_kv_mode(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
) -> Result<LoadedModel, String> {
    // Classify once and admit before any side effect. EP is HFQ-only and
    // dispatches on arch_id, so the admission retains the arch_id decision and
    // the per-rank file re-open happens inside the EP load (unchanged).
    let hip = hip_bridge::HipRuntime::load().ok();
    let gpu_arch = hip
        .as_ref()
        .and_then(|h| h.get_arch(0).ok())
        .unwrap_or_default();
    let vmm_runtime_available = hip.as_ref().is_some_and(|h| {
        (0..tp).all(|rank| {
            h.get_arch(rank as i32).ok().as_deref() == Some(gpu_arch.as_str())
                && h.mem_get_allocation_granularity(
                    &hip_bridge::HipMemAllocationProp::device_pinned(rank as i32),
                    hip_bridge::HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
                )
                .is_ok()
        })
    });
    let admission = crate::admission::admit_source(
        path,
        tp,
        1,
        crate::admission::KvBackendRequest::from_override(kv_backend)?,
        None,
        &gpu_arch,
        None,
        "auto",
        None,
        max_seq,
        crate::admission::KvBackendHints {
            kv_mode,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: crate::admission::qwen_default_q8_enabled(),
            kv_adaptive: None,
            cask: None,
            deepseek4_heterogeneous: false,
            vmm_runtime_available,
            free_vram_bytes: None, // EP has no single-card KV capacity envelope.
        },
    )?;
    let warning = crate::admission::legacy_warning(
        admission.kv_backend,
        admission.kv_backend_reason.as_deref(),
    );
    let loaded = load_model_ep_admitted(
        admission,
        path,
        max_seq,
        tp,
        kv_mode,
        None,
        None,
        state_quant,
        None,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
    );
    if loaded.is_ok() {
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
    }
    loaded
}

/// Dispatch an already-admitted expert-parallel source on its `arch_id` —
/// admission's classification is not repeated. Unlike the single/pp route,
/// the per-arch EP loaders still re-open `path` per rank, so the retained
/// `SourceAdmission.source` is dropped here rather than consumed. Destructive
/// (per-rank allocation + collectives) begins inside the per-arch loaders.
pub fn load_model_ep_admitted(
    admission: crate::admission::SourceAdmission,
    path: &str,
    _max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
    state_quant: Option<&str>,
    mtp_path: Option<&Path>,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> Result<LoadedModel, String> {
    let kv_backend = admission.kv_backend;
    let qwen_default_q8 = admission.qwen_default_q8;
    let max_seq = admission.max_seq;
    match admission.arch_id {
        9 => load_model_ep_ds4(
            path,
            max_seq,
            tp,
            resolve_deepseek4_compressor_cache_kv_mode(kv_mode)?,
            kv_backend,
        ),
        10 => load_model_ep_minimax(path, max_seq, tp),
        5 | 6 => load_model_ep_qwen35(
            path,
            max_seq,
            tp,
            kv_mode,
            kv_k,
            kv_v,
            qwen_default_q8,
            Some(kv_backend.as_str()),
            state_quant,
            mtp_path,
            spec,
        ),
        // Backstop: `admit_source` above already refused every other arch_id.
        // Route through the shared constructor (not `unreachable!`) so the
        // refusal survives a future edit that drops the early classification,
        // and so this message never drifts from `ep_admission`'s.
        id => Err(ep_unsupported_arch_message(id)),
    }
}

/// Expert-parallel load with an explicit DeepSeek V4 compressor-cache storage
/// policy. The compatibility wrapper above deliberately retains the historical
/// F32 route.
pub fn load_model_ep_with_compressor_cache(
    path: &str,
    max_seq: usize,
    tp: usize,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    // Admission: refuse archs with no `EpArch` before any per-arch device init.
    ep_admission(hfq.arch_id)?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(path, max_seq, tp, compressor_cache, KvBackend::Vmm),
        10 if compressor_cache == hipfire_config::Deepseek4CompressorCache::F32 => {
            load_model_ep_minimax(path, max_seq, tp)
        }
        10 => Err("the DeepSeek V4 compressor-cache setting (memory.kv_cache = f16) does not apply to MiniMax models; remove it for this model".to_string()),
        5 | 6 if compressor_cache == hipfire_config::Deepseek4CompressorCache::F32 => {
            load_model_ep_with_kv_mode(path, max_seq, tp, None, None, None)
        }
        5 | 6 => {
            Err("the DeepSeek V4 compressor-cache setting (memory.kv_cache = f16) does not apply to Qwen3.5 models; remove it for this model".to_string())
        }
        // Backstop: `ep_admission` above already refused these; route through the
        // shared constructor (not `unreachable!`) so the refusal survives a
        // future edit that drops the early call.
        id => Err(ep_unsupported_arch_message(id)),
    }
}

fn load_model_ep_ds4(
    path: &str,
    max_seq: usize,
    tp: usize,
    compressor_cache: hipfire_config::Deepseek4CompressorCache,
    kv_backend: KvBackend,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let mut config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    // The EP/TP serve path has no speculative verifier. Do not auto-discover
    // and replicate an adjacent DSpark sidecar onto every rank when it cannot
    // be selected; ordinary AR must retain that memory for per-rank KV state.
    config.load_dspark = false;
    let arch_id = hfq.arch_id;
    let n_exp = config.n_routed_experts;

    // Host-side metadata work (chat template + author-recommended sampling) BEFORE
    // any GPU allocation / EP hipGraph capture. `recommended_sampling()` reparses
    // the .hfq metadata_json (serde_json::from_str); doing that post-allocation but
    // pre-capture churns the host heap and — on gfx12 / ROCm 7.2, which snapshots
    // buffer addresses at graph-instantiate — slows the captured EP-decode graph
    // replay. Same regression as load_model (gfx12 A3B 99→50), mirrored here for the
    // ds4 EP path; see project_gfx12_hipgraph_late_host_alloc_clobber. The EP graph
    // itself (deepseek4 forward.rs begin_graph_capture) is untouched — it still
    // captures + engages; this only settles the heap before it instantiates.
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();

    let gpus =
        Gpus::init_ep(tp, config.num_hidden_layers).map_err(|e| format!("init_ep: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_ep gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    // Bind the recorded mesh to the loaded topology: `init_ep` is the only
    // thing distinguishing this `Gpus` from a TP one (identical devices,
    // bands, and pre-flight), so fail loudly here — where both the mesh and
    // the rank count are known — if a future constructor ever records the
    // wrong axis instead of loading silently mislabeled.
    let ep = gpus.mesh.size_of(DimKind::Ep);
    if ep != n {
        return Err(format!(
            "init_ep mesh records Ep={ep} for {n} devices, expected tp={tp} (mesh axis out of sync with constructed ranks)"
        ));
    }
    eprintln!("[loader] EP load: tp={tp} arch=ds4 experts={n_exp} (rank r owns e%{tp}==r)");
    let shard = ShardConfig::new_uneven_experts(
        tp,
        /*tp_kv_replicate=*/ true,
        n_exp,
        ExpertAssign::Stride,
    )
    .map_err(|e| format!("ShardConfig: {e:?}"))?;
    // Transactional partial-load: build per-rank weights/state/partials INTO the
    // staging guard. Every `?` below early-returns while `staging` is alive, so
    // its `Drop` frees the ranks already loaded.
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = Ds4EpStaging::new(gpus);
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = deepseek4::DeepseekV4::load_weights_sharded(&mut h, &config, dev, &shard, r)
            .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        // Deterministic partial-load fault for testing the cleanup path. Fires
        // AFTER ranks 0..=r loaded; the guard's Drop frees them all.
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic ds4 EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut st =
            deepseek4::DeepseekV4State::new(&config).map_err(|e| format!("state {r}: {e:?}"))?;
        st.compressor_cache_backend = kv_backend;
        staging.state.push(st);
        let p = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size], rdna_compute::DType::F32)
            .map_err(|e| format!("partial {r}: {e:?}"))?;
        staging.partials.push(p);
    }
    // Exact-gated gfx1201 MQ2R TP3/TP4 graph + batched-prefill substrate.
    // Allocate every long-lived pointer before enabling peer access so peer
    // mappings and later graph capture observe the final allocation layout.
    let gfx1201_mq2r_tp = matches!(tp, 3 | 4)
        && config.mq2r
        && !config.mq2rxt
        && staging
            .gpus_mut()
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201());
    if gfx1201_mq2r_tp {
        // Keep compressor caches replicated and device-local on the normal
        // path. Peer-reading the block-cyclic experiment was exact but reduced
        // 21K NIAH decode from 41.15 to 11.10 tok/s. Capacity growth therefore
        // remains entirely device-local; F16 storage is what makes 1M feasible.
        // B=1024 is the certified short-context throughput schedule. At a
        // declared long-context ceiling, B=512 gives back about 1.34 GiB/rank
        // while retaining most grouped-MoE occupancy. Preserve that established
        // F32 schedule. An explicitly selected F16 cache may trade prefill
        // throughput for additional physical depth with B=128 at an extreme
        // declared ceiling. Replicated TP3 still tops out below 1M; the smaller
        // batch maximizes its measured F16 capacity without regressing F32.
        const SHORT_CONTEXT_BATCH: usize = 1024;
        const LONG_CONTEXT_BATCH: usize = 512;
        const EXTREME_CONTEXT_BATCH: usize = 128;
        const LONG_CONTEXT_THRESHOLD: usize = 32 * 1024;
        const EXTREME_CONTEXT_THRESHOLD: usize = 256 * 1024;
        let compressor_cache_dtype = match compressor_cache {
            hipfire_config::Deepseek4CompressorCache::F32 => rdna_compute::DType::F32,
            hipfire_config::Deepseek4CompressorCache::F16 => rdna_compute::DType::F16,
        };
        for state in &mut staging.state {
            state.compressor_cache_dtype = compressor_cache_dtype;
        }
        let ep_prefill_max_batch = if compressor_cache
            == hipfire_config::Deepseek4CompressorCache::F16
            && max_seq > EXTREME_CONTEXT_THRESHOLD
        {
            EXTREME_CONTEXT_BATCH
        } else if max_seq > LONG_CONTEXT_THRESHOLD {
            LONG_CONTEXT_BATCH
        } else {
            SHORT_CONTEXT_BATCH
        };
        let projected = deepseek4::forward::PrefillBatchScratch::projected_allocation_bytes(
            &config,
            ep_prefill_max_batch,
        )?;
        // TP across several devices has no per-rank budget: refuse under a
        // HIPFIRE_EMU profile (a no-op when emulation is off).
        staging.gpus_mut().devices[0]
            .emu_refuse_multi_device("gfx1201 TP batched prefill")
            .map_err(|e| format!("{e:?}"))?;
        for rank in 0..n {
            let dev = &mut staging.gpus_mut().devices[rank];
            dev.bind_thread()
                .map_err(|e| format!("bind gfx1201 TP prefill rank {rank}: {e:?}"))?;
            // Capped per-device query (UMA-safe clamp, then the emu cap).
            let (free, _total) = dev
                .device_mem_info_for(dev.device_id)
                .map_err(|e| format!("query gfx1201 TP prefill VRAM rank {rank}: {e:?}"))?;
            if free < projected {
                return Err(format!(
                    "gfx1201 TP prefill rank {rank}: need {:.2} GiB scratch, only {:.2} GiB free",
                    projected as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64,
                ));
            }
            let pbs =
                deepseek4::forward::PrefillBatchScratch::new(dev, &config, ep_prefill_max_batch)
                    .map_err(|e| format!("allocate gfx1201 TP prefill rank {rank}: {e}"))?;
            staging.prefill.push(pbs);
        }
        eprintln!(
            "[loader] gfx1201 TP{tp} batched prefill: B={} scratch={:.2} GiB/rank compressor_cache={compressor_cache}",
            ep_prefill_max_batch,
            projected as f64 / (1u64 << 30) as f64,
        );
        staging
            .gpus_mut()
            .prepare_tp_graph_signals(config.num_hidden_layers * 2)
            .map_err(|e| format!("prepare gfx1201 TP graph signals: {e:?}"))?;
    } else if compressor_cache == hipfire_config::Deepseek4CompressorCache::F16 {
        // Single-device MQ2R also carries the F16 compressor cache on any
        // architecture whose kernels can compile it. Admission is the
        // capability predicate rather than a chip list: the two F16 sources
        // select their WMMA fragment layout in-source, so the set that can run
        // them is "wave32 WMMA on RDNA3 or RDNA4", and that fact belongs in
        // arch_caps next to the kernels it describes. Storage stays confined
        // to main_kv_cache and indexer_kv_cache; commit arithmetic remains F32
        // before the single F32-to-F16 store.
        let f16_single = tp <= 1
            && config.mq2r
            && !config.mq2rxt
            && staging
                .gpus_mut()
                .devices
                .iter()
                .all(|device| device.arch_caps.supports_ds4_f16_compressor_cache());
        if !f16_single {
            return Err(format!(
                "DeepSeek V4 compressor_cache=f16 requires MQ2R on TP3/TP4 gfx1201, or single-device MQ2R on an architecture with wave32 WMMA (RDNA3/RDNA4); got tp={tp}, mq2r={}, mq2rxt={}, devices={}",
                config.mq2r,
                config.mq2rxt,
                staging
                    .gpus_mut()
                    .devices
                    .iter()
                    .map(|device| device.arch.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials, prefill) = staging.into_parts();

    let eos_tok: u32 = {
        let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
        if ids.len() == 1 {
            ids[0]
        } else {
            1
        }
    };
    // chat_template + rec extracted pre-allocation above (gfx12 hipGraph hazard).
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Ds4 {
                config,
                weights,
                state,
                partials,
                prefill,
            },
        }),
        deepseek4_eos_tok: eos_tok,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}

fn load_model_ep_minimax(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let n_exp = config.num_local_experts;

    // Host-side metadata work (chat template + author-recommended sampling) BEFORE
    // any GPU allocation / EP hipGraph capture. `recommended_sampling()` reparses
    // the .hfq metadata_json (serde_json::from_str); doing that post-allocation but
    // pre-capture churns the host heap and — on gfx12 / ROCm 7.2, which snapshots
    // buffer addresses at graph-instantiate — slows the captured EP-decode graph
    // replay. Same regression as load_model (gfx12 A3B 99→50), mirrored here for the
    // minimax EP path; see project_gfx12_hipgraph_late_host_alloc_clobber. The EP
    // graph itself (minimax forward.rs begin_graph_capture) is untouched — it still
    // captures + engages; this only settles the heap before it instantiates.
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();

    let gpus =
        Gpus::init_ep(tp, config.num_hidden_layers).map_err(|e| format!("init_ep: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_ep gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    // Bind the recorded mesh to the loaded topology: `init_ep` is the only
    // thing distinguishing this `Gpus` from a TP one (identical devices,
    // bands, and pre-flight), so fail loudly here — where both the mesh and
    // the rank count are known — if a future constructor ever records the
    // wrong axis instead of loading silently mislabeled.
    let ep = gpus.mesh.size_of(DimKind::Ep);
    if ep != n {
        return Err(format!(
            "init_ep mesh records Ep={ep} for {n} devices, expected tp={tp} (mesh axis out of sync with constructed ranks)"
        ));
    }
    eprintln!("[loader] EP load: tp={tp} arch=minimax experts={n_exp} (rank r owns e%{tp}==r)");
    let shard = ShardConfig::new(
        tp,
        /*tp_kv_replicate=*/ true,
        n_exp,
        ExpertAssign::Stride,
    )
    .map_err(|e| format!("ShardConfig: {e:?}"))?;
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = MinimaxEpStaging::new(gpus);
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = minimax::MiniMaxWeights::load(&mut h, &config, dev, Some((&shard, r)))
            .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic minimax EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let st = {
            let dev = &mut staging.gpus_mut().devices[r];
            minimax::MiniMaxState::new_with_max_seq(dev, &config, max_seq)
                .map_err(|e| format!("state {r}: {e:?}"))?
        };
        staging.state.push(st);
        let p = staging.gpus_mut().devices[r]
            .zeros(&[config.hidden_size], rdna_compute::DType::F32)
            .map_err(|e| format!("partial {r}: {e:?}"))?;
        staging.partials.push(p);
    }
    let peer = staging
        .gpus_mut()
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e:?}"))?;
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    eprintln!("[loader] EP load complete: {n} ranks, peer_access={peer}");
    let (gpus, weights, state, partials) = staging.into_parts();

    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("[e~[")
            .or_else(|| try_one("<|im_end|>"))
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(1)
    };
    // chat_template + rec extracted pre-allocation above (gfx12 hipGraph hazard).
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Minimax {
                config,
                weights,
                state,
                partials,
            },
        }),
        minimax_eos_tok: eos_tok,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}
#[allow(clippy::too_many_arguments)]
fn load_model_ep_qwen35(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
    qwen_default_q8: bool,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
    mtp_path: Option<&Path>,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    let hfq_probe = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    if hfq_probe.arch_id != 5 && hfq_probe.arch_id != 6 {
        return Err(format!(
            "expert parallelism (tp>1) via this path needs a Qwen3.5 model (arch 5 or 6), got arch {}",
            hfq_probe.arch_id
        ));
    }
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_probe.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq_probe).map_err(|e| format!("qwen35 config: {e}"))?;
    if config.num_experts == 0 {
        drop(hfq_probe);
        let dense_tp_route = hipfire_config::developer_var("HIPFIRE_DENSE_TP_ROUTE").ok();
        let route = parse_dense_tp_route(dense_tp_route.as_deref())?;
        let load = match resolve_dense_tp_route(route, &config, tp, spec.mtp == Some(true)) {
            DenseTpRoute::Mesh => load_model_tp_qwen35_mesh,
            _ => load_model_tp_qwen35_dense,
        };
        return load(
            path,
            max_seq,
            tp,
            kv_mode,
            kv_k,
            kv_v,
            qwen_default_q8,
            kv_backend,
            state_quant,
            mtp_path,
            spec,
        );
    }
    if let Some(reason) = qwen35_ep_moe_topology_refusal(hfq_probe.arch_id, config.num_experts, tp)
    {
        return Err(reason);
    }
    let state_quant_resolved = parse_state_quant(state_quant)?;
    let config_mode = hipfire_runtime::config::get();
    let kv_raw = kv_mode.unwrap_or(config_mode.kv_mode.as_str());
    let pair = kv_mode::resolve_kv_pair(
        kv_raw,
        kv_k,
        kv_v,
        &kv_mode::QWEN35_TP_POLICY,
        "multi-gpu",
        qwen_default_q8,
    )
    .map_err(|e| format!("Qwen EP KV: {e}"))?;
    let kv_mode_resolved = pair.k();
    let Some(v_mode) = pair.v() else {
        return Err("Qwen native KV has no multi-GPU constructor".into());
    };
    if v_mode != llama::VMode::Q8 {
        return Err("Qwen EP has no multi-GPU Lloyd-V constructor; use --kv-v q8".into());
    }
    eprintln!(
        "  KV cache: requested mode={kv_raw}, effective K={} V={}",
        kv_mode::qwen_k_display_name(kv_mode_resolved),
        kv_mode::qwen_v_display_name(v_mode)
    );
    // MoE EP has no VMM KV owner; admission selected its legacy cache.
    let _ = kv_backend;
    if config.paged_experts {
        return Err("EP qwen35: paged_experts must be false".to_string());
    }
    if config.reap_keep.is_some() {
        return Err(
            "EP qwen35: this REAP-pruned model cannot run with expert parallelism; set tp=1"
                .to_string(),
        );
    }
    let arch_id = hfq_probe.arch_id;
    let n_exp = config.num_experts;
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq_probe, path);
    let rec = hfq_probe.recommended_sampling();
    let gpus = Gpus::init_ep(tp, config.n_layers).map_err(|e| format!("init_ep: {e:?}"))?;
    let n = gpus.devices.len();
    if n != tp {
        return Err(format!(
            "init_ep gave {n} devices, expected tp={tp} (check ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)"
        ));
    }
    // Bind the recorded mesh to the loaded topology: `init_ep` is the only
    // thing distinguishing this `Gpus` from a TP one (identical devices,
    // bands, and pre-flight), so fail loudly here — where both the mesh and
    // the rank count are known — if a future constructor ever records the
    // wrong axis instead of loading silently mislabeled.
    let ep = gpus.mesh.size_of(DimKind::Ep);
    if ep != n {
        return Err(format!(
            "init_ep mesh records Ep={ep} for {n} devices, expected tp={tp} (mesh axis out of sync with constructed ranks)"
        ));
    }
    for (idx, dev) in gpus.devices.iter().enumerate() {
        if !dev.arch_caps.is_gfx1201() {
            return Err(format!(
                "EP qwen35 requires all {tp}×gfx1201, rank {idx} is {}",
                dev.arch.as_str()
            ));
        }
    }
    eprintln!(
        "[loader] EP load: tp={tp} arch=qwen35 experts={n_exp} (rank r owns e%{tp}==r via Stride, replicated KV)"
    );
    let shard = ShardConfig::new(tp, true, n_exp, ExpertAssign::Stride)
        .map_err(|e| format!("ShardConfig: {e:?}"))?;
    let fail_rank = ep_fail_rank();
    let _ = fail_rank;
    let mut staging = Qwen35EpStaging::new(gpus);
    // The sealed loader needs the same mesh + physical topology the runtime
    // holds: clone both up front so each rank load agrees with `init_ep`.
    let ep_mesh = staging.gpus_mut().mesh.clone();
    let ep_physical_devices: Vec<i32> = staging
        .gpus_mut()
        .devices
        .iter()
        .map(|dev| dev.device_id)
        .collect();
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("bind {r}: {e:?}"))?;
        let mut h = HfqFile::open(Path::new(path)).map_err(|e| format!("reopen rank {r}: {e}"))?;
        let dev = &mut staging.gpus_mut().devices[r];
        let w = qwen35::load_weights_ep_rank(
            &mut h,
            dev,
            &config,
            &ep_mesh,
            &ep_physical_devices,
            shard.clone(),
            r,
        )
        .map_err(|e| format!("shard load rank {r}: {e:?}"))?;
        staging.weights.push(w);
        if fail_rank == Some(r) {
            return Err(format!(
                "HIPFIRE_EP_FAIL_RANK={r}: synthetic qwen35 EP load failure after rank {r} (testing partial-load cleanup)"
            ));
        }
    }
    // Sequential decode state, one replica per rank: EP shards routed experts
    // only, so KV/DeltaNet/scratch use the full (unsharded) config exactly as
    // the sealed loader's replicated-KV contract requires. Staged into the
    // guard, so a mid-loop failure frees every published rank via `Drop`.
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == qwen35::LayerType::FullAttention)
        .collect();
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("EP qwen35 state bind rank {r}: {e:?}"))?;
        let dims = KvDims {
            layers: KvLayers::Mask(is_kv_layer.clone()),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            max_seq,
            physical_cap: Some(max_seq),
        };
        let kv = <llama::KvCache as KvCacheExt>::from_mode_with_backend(
            kv_mode_resolved,
            KvBackend::Legacy,
            KvTarget::Single(&mut staging.gpus_mut().devices[r]),
            &dims,
        )
        .map_err(|e| format!("EP qwen35 KV rank {r}: {e:?}"))?;
        staging.kv_caches.push(kv);
        let dn = qwen35::DeltaNetState::new_with_quant(
            &mut staging.gpus_mut().devices[r],
            &config,
            state_quant_resolved,
        )
        .map_err(|e| format!("EP qwen35 DeltaNet rank {r}: {e:?}"))?;
        staging.dn_states.push(dn);
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(
            &mut staging.gpus_mut().devices[r],
            &config,
            128,
            max_seq,
        )
        .map_err(|e| format!("EP qwen35 scratch rank {r}: {e:?}"))?;
        staging.scratches.push(scratch);
        let partial = staging.gpus_mut().devices[r]
            .zeros(&[config.dim], rdna_compute::DType::F32)
            .map_err(|e| format!("EP qwen35 partial rank {r}: {e:?}"))?;
        staging.partials.push(partial);
    }
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("ensure_rank_streams: {e:?}"))?;
    // Sequential prefill scratch, one full chunk per rank on its owning
    // device: a `PrefillBatchScratch` with the DeltaNet S-tape omitted (plain
    // prefill never reads `dn_s_tape`; `cap_gdn_tape = false`) plus a zeroed
    // `[chunk*dim]` F32 routed partial. `chunk = min(max_seq, prefill_max_batch_ep())`
    // matches the shared EP batch `prefill_chunk` policy (`HIPFIRE_PREFILL_MAX_BATCH`); serve windows are `<= chunk`.
    // Staged in the guard so a mid-loop failure frees every published rank
    // via `Drop`, and allocated BEFORE any `enable_peer_all` (ROCm cannot
    // retroactively map late peer-visible allocs).
    let prefill_chunk = max_seq.min(qwen35::prefill_max_batch_ep());
    if prefill_chunk == 0 {
        return Err("EP qwen35: max_seq must be nonzero for prefill scratch".to_string());
    }
    let prefill_partial_len = prefill_chunk
        .checked_mul(config.dim)
        .ok_or_else(|| "EP qwen35: prefill partial length overflow".to_string())?;
    for r in 0..n {
        staging.gpus_mut().devices[r]
            .bind_thread()
            .map_err(|e| format!("EP qwen35 prefill bind rank {r}: {e:?}"))?;
        let pbs = qwen35::PrefillBatchScratch::new_opt(
            &mut staging.gpus_mut().devices[r],
            &config,
            prefill_chunk,
            false,
        )
        .map_err(|e| format!("EP qwen35 prefill PBS rank {r}: {e:?}"))?;
        staging.prefill_pbs.push(pbs);
        let pre_partial = staging.gpus_mut().devices[r]
            .zeros(&[prefill_partial_len], rdna_compute::DType::F32)
            .map_err(|e| format!("EP qwen35 prefill partial rank {r}: {e:?}"))?;
        staging.prefill_partials.push(pre_partial);
    }
    eprintln!(
        "[loader] EP load complete: {n} ranks, peer access deferred until post-batch allocation"
    );
    let (gpus, weights, kv_caches, dn_states, scratches, partials, prefill_pbs, prefill_partials) =
        staging.into_parts();
    let eos_tok: u32 = {
        let ids = tokenizer.encode("<|im_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            config.eos_token
        }
    };
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Qwen35 {
                config,
                weights,
                kv_caches,
                dn_states,
                scratches,
                partials,
                batch: None,
                prefill_pbs,
                prefill_partials,
            },
        }),
        qwen35_eos_tok: eos_tok,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn load_model_tp_qwen35_dense(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
    qwen_default_q8: bool,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
    mtp_path: Option<&Path>,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("qwen35 config: {e}"))?;
    let shard = ShardConfig::new(tp, false, 0, ExpertAssign::Stride)
        .map_err(|e| format!("dense TP ShardConfig: {e}"))?;
    let layouts = qwen35::dense_tp_rank_layouts(&config, &shard)
        .map_err(|e| format!("dense TP layout: {e}"))?;
    let state_quant_resolved = parse_state_quant(state_quant)?;
    let kv_backend_raw = kv_backend.unwrap_or("legacy");
    let kv_backend_resolved: KvBackend = kv_backend_raw.parse().map_err(|err| format!("{err}"))?;
    let config_mode = hipfire_runtime::config::get();
    let kv_raw = kv_mode.unwrap_or(config_mode.kv_mode.as_str());
    let pair = kv_mode::resolve_kv_pair(
        kv_raw,
        kv_k,
        kv_v,
        &kv_mode::QWEN35_TP_POLICY,
        "multi-gpu",
        qwen_default_q8,
    )
    .map_err(|e| format!("Qwen dense TP KV: {e}"))?;
    let kv_mode_resolved = pair.k();
    let Some(v_mode) = pair.v() else {
        return Err("Qwen native KV has no multi-GPU constructor".into());
    };
    if v_mode != llama::VMode::Q8 {
        return Err("Qwen dense TP has no multi-GPU Lloyd-V constructor; use --kv-v q8".into());
    }
    eprintln!(
        "  KV cache: requested mode={kv_raw}, effective K={} V={}",
        kv_mode::qwen_k_display_name(kv_mode_resolved),
        kv_mode::qwen_v_display_name(v_mode)
    );
    // Preflight weights before GPU allocation (validates qt geometry/blob/sidecar).
    qwen35::preflight_weights_dense_tp(&hfq, &config, &shard)?;
    let configs = layouts
        .iter()
        .map(|layout| qwen35::local_dense_tp_config(&config, layout))
        .collect::<Vec<_>>();
    let arch_id = hfq.arch_id;
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();
    let eos_tok = {
        let ids = tokenizer.encode("<|im_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            config.eos_token
        }
    };
    drop(hfq);

    let gpus = Gpus::init_tp(tp, config.n_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    if gpus.devices.len() != tp {
        return Err(format!(
            "init_tp gave {} devices, expected tp={tp}",
            gpus.devices.len()
        ));
    }
    let mut staging = Qwen35DenseTpStaging::new(gpus);
    // Per-rank VMM preflight (mirrors the single-GPU `ensure_vmm_ready_for_load`
    // gate at the top of `load_model_with_kv_backend`): refuse the load while
    // any rank device reports pending teardown. Fresh `init_tp` devices are
    // idle, so this is a no-op success today; it keeps load/unload symmetric
    // if devices are ever reused across loads.
    for rank in 0..tp {
        ensure_vmm_ready_for_load(&mut staging.gpus_mut().devices[rank])
            .map_err(|err| format!("dense TP VMM preflight rank {rank}: {err}"))?;
    }
    for rank in 0..tp {
        staging.gpus_mut().devices[rank]
            .bind_thread()
            .map_err(|e| format!("dense TP bind rank {rank}: {e:?}"))?;
        let mut rank_hfq = HfqFile::open(Path::new(path))
            .map_err(|e| format!("dense TP reopen rank {rank}: {e}"))?;
        let weights = qwen35::load_weights_dense_tp_rank(
            &mut rank_hfq,
            &config,
            &mut staging.gpus_mut().devices[rank],
            &layouts[rank],
        )
        .map_err(|e| format!("dense TP weight load rank {rank}: {e:?}"))?;
        staging.weights.push(weights);

        let local = &configs[rank];
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();
        let dims = KvDims {
            layers: KvLayers::Mask(is_kv_layer),
            n_kv_heads: local.n_kv_heads,
            head_dim: local.head_dim,
            max_seq,
            physical_cap: Some(max_seq),
        };
        let kv = <llama::KvCache as KvCacheExt>::from_mode_with_backend(
            kv_mode_resolved,
            kv_backend_resolved,
            KvTarget::Single(&mut staging.gpus_mut().devices[rank]),
            &dims,
        )
        .map_err(|e| format!("dense TP KV rank {rank}: {e:?}"))?;
        staging.kv_caches.push(kv);
        let dn = qwen35::DeltaNetState::new_with_quant(
            &mut staging.gpus_mut().devices[rank],
            local,
            state_quant_resolved,
        )
        .map_err(|e| format!("dense TP DeltaNet rank {rank}: {e:?}"))?;
        staging.dn_states.push(dn);
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(
            &mut staging.gpus_mut().devices[rank],
            local,
            128,
            max_seq,
        )
        .map_err(|e| format!("dense TP scratch rank {rank}: {e:?}"))?;
        staging.scratches.push(scratch);
    }
    // Probe complete peer topology without mutation. Complete → enable + RCCL;
    // mixed → host-staged allreduce (no peer enable, no device reduce scratch).
    let peer = staging
        .gpus_mut()
        .can_access_peer_all()
        .map_err(|e| format!("dense TP can_access_peer_all: {e:?}"))?;
    if peer {
        // TP2 whole-token decode graph (exact gfx1201 pair by default, gfx1100
        // pair with `HIPFIRE_TP_FULL_GRAPH=1`): the runtime admits it and
        // prepares its signal tape before peer access is enabled.
        let barriers = config.n_layers * 2;
        let whole_token_graph = staging
            .gpus_mut()
            .prepare_tp2_whole_token_graph(barriers, config.dim * std::mem::size_of::<f32>())
            .map_err(|e| format!("prepare TP2 whole-token graph: {e:?}"))?;
        if whole_token_graph {
            let arch = staging.gpus_mut().devices[0].arch.clone();
            eprintln!(
                "[loader] dense qwen {arch} TP2 whole-token graph signal tape: {barriers} barriers"
            );
        }
        let enabled = staging
            .gpus_mut()
            .enable_peer_all()
            .map_err(|e| format!("dense TP enable_peer_all: {e:?}"))?;
        if !enabled {
            return Err(
                "dense TP enable_peer_all returned false after can_access_peer_all".to_string(),
            );
        }
    } else {
        // Mixed P2P topology: select host-staged allreduce; do not enable peers
        // or reserve device reduction scratch.
        eprintln!(
            "[loader] dense qwen TP mixed topology: host-staged allreduce (no peer enable/scratch)"
        );
    }
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("dense TP ensure_rank_streams: {e:?}"))?;
    // The MTP head is replicated on rank 0 (the trunk stays sharded) only for
    // an explicit `speculation.mtp = on` (`spec.mtp == Some(true)`). `auto`
    // (`None`) and `off` keep the AR mesh loop that dense TP ran before MTP
    // was wired on the mesh, so a sidecar beside the trunk never enables it
    // by itself. A missing head is a load error, as `mtp=on` is on one GPU.
    // Resolved while `staging` still owns the ranks: the error returns through
    // its `Drop`, which frees every rank's weights, KV, DeltaNet state and
    // scratch instead of stranding the sharded trunk in a live daemon.
    let mtp_head = if spec.mtp == Some(true) {
        let (head, errors) = resolve_qwen35_mtp_head(
            Path::new(path),
            // The CLI-resolved sidecar (a symlinked trunk's canonical path
            // has no sibling head), else the trunk's sibling `.mtp`.
            mtp_path.map(Path::to_path_buf),
            (config.dim, config.vocab_size),
            &mut staging.gpus_mut().devices[0],
            max_seq,
            Some("dense TP rank 0"),
        );
        if head.is_none() {
            let reason = if errors.is_empty() {
                "no bundled trailer or .mtp sidecar found".to_string()
            } else {
                errors.join("; ")
            };
            return Err(format!(
                "MTP head required (mtp=on) but not loaded: {reason}"
            ));
        }
        head
    } else {
        None
    };
    let (gpus, weights, kv_caches, dn_states, scratches) = staging.into_parts();
    eprintln!("[loader] dense qwen TP load complete: {tp} ranks, peer_access={peer}");
    eprintln!(
        "[loader] dense qwen TP route=hand (hand dense-TP forward): {tp} ranks, peer_access={peer}"
    );
    eprintln!(
        "[loader] dense qwen TP exchange={} (HIPFIRE_TP_EXCHANGE)",
        hipfire_runtime::multi_gpu::tp_exchange::tp_exchange_label(peer)
    );
    let mtp_present = mtp_head.is_some();
    // Block-verify drafters cannot run on the mesh (`Qwen35DenseTpTarget`
    // refuses `verify_block`), so an `ngram=on` request must not build one
    // here: it would sit inert behind the AR route and lie in the load log.
    let mesh_spec = hipfire_runtime::loader_api::SpecLoadCfg {
        ngram_draft: Some(false),
        ..spec
    };
    if spec.ngram_draft == Some(true) {
        eprintln!("  dense TP: n-gram drafter refused (block verify is not wired on the mesh)");
    }
    let speculator =
        crate::spec_build::build_speculator(arch_id, None, mtp_head, true, max_seq, mesh_spec);

    Ok(LoadedModel {
        speculator,
        mtp_weights_present: mtp_present,
        ep: Some(EpState {
            gpus,
            inner: EpArch::Qwen35DenseTp {
                shard,
                configs,
                weights,
                kv_caches,
                dn_states,
                scratches,
            },
        }),
        qwen35_eos_tok: eos_tok,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}

/// The dense Qwen TP route `HIPFIRE_DENSE_TP_ROUTE` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseTpRoute {
    /// The default: the generic mesh executor wherever it admits the load,
    /// else the hand route ([`resolve_dense_tp_route`]).
    Auto,
    /// The hand dense-TP forward (`load_model_tp_qwen35_dense`): the rollback,
    /// and the only route for uneven TP3/TP5 splits and dense-TP MTP.
    Hand,
    /// The generic mesh executor (`load_model_tp_qwen35_mesh`), forced: a
    /// load it does not admit fails instead of falling back.
    Mesh,
}

/// `HIPFIRE_DENSE_TP_ROUTE`: `auto` (also unset or empty), `mesh` or `hand`;
/// any other value is a load error.
fn parse_dense_tp_route(value: Option<&str>) -> Result<DenseTpRoute, String> {
    match value {
        None | Some("") | Some("auto") => Ok(DenseTpRoute::Auto),
        Some("hand") => Ok(DenseTpRoute::Hand),
        Some("mesh") => Ok(DenseTpRoute::Mesh),
        Some(other) => Err(format!(
            "HIPFIRE_DENSE_TP_ROUTE={other:?} is not a dense TP route; accepted values: \
             `auto` (default; also unset or empty), `mesh` (generic mesh executor), `hand` (hand route)"
        )),
    }
}

/// `Auto` takes the mesh route unless the load needs what only the hand route
/// has: an uneven head/FFN split (TP3/TP5 on the 27B) or dense-TP MTP
/// (`speculation.mtp = on`). Logs the fallback reason. `Hand`/`Mesh` pass through.
fn resolve_dense_tp_route(
    route: DenseTpRoute,
    config: &qwen35::Qwen35Config,
    tp: usize,
    mtp: bool,
) -> DenseTpRoute {
    if route != DenseTpRoute::Auto {
        return route;
    }
    let reason = if mtp {
        Some("MTP requested (the mesh route has no MTP verify)".to_string())
    } else {
        qwen35::mesh_tp::tp_rank_layouts(config, tp)
            .err()
            .map(|e| format!("uneven split: {e}"))
    };
    match reason {
        Some(reason) => {
            eprintln!("[loader] dense qwen TP route=auto -> hand: {reason}");
            DenseTpRoute::Hand
        }
        None => DenseTpRoute::Mesh,
    }
}

#[cfg(test)]
mod dense_tp_route_tests {
    use super::{parse_dense_tp_route, DenseTpRoute};

    #[test]
    fn route_knob_selects_mesh_and_refuses_unknown_values() {
        assert_eq!(parse_dense_tp_route(None), Ok(DenseTpRoute::Auto));
        assert_eq!(parse_dense_tp_route(Some("")), Ok(DenseTpRoute::Auto));
        assert_eq!(parse_dense_tp_route(Some("auto")), Ok(DenseTpRoute::Auto));
        assert_eq!(parse_dense_tp_route(Some("hand")), Ok(DenseTpRoute::Hand));
        assert_eq!(parse_dense_tp_route(Some("mesh")), Ok(DenseTpRoute::Mesh));
        for bad in ["Mesh", "1", "generic", " mesh"] {
            let error = parse_dense_tp_route(Some(bad)).unwrap_err();
            assert!(
                error.contains("`mesh`") && error.contains("`hand`") && error.contains("`auto`"),
                "{error}"
            );
        }
    }
}

/// Transactional owner for the mesh-route dense Qwen TP construction: the
/// `Gpus` (with its provisioned mesh edges) and the ranks loaded so far. An
/// early return drains the mesh edges first and then frees every rank on its
/// own device, so a failed load strands no VRAM. A failed drain quarantines:
/// the ranks and the `Gpus` are leaked rather than freed under possibly
/// in-flight work.
struct Qwen35MeshTpStaging {
    gpus: Option<Gpus>,
    ranks: Vec<qwen35::mesh_tp::Qwen35TpRank>,
}

impl Qwen35MeshTpStaging {
    fn new(gpus: Gpus) -> Self {
        Self {
            gpus: Some(gpus),
            ranks: Vec::new(),
        }
    }

    fn gpus_mut(&mut self) -> &mut Gpus {
        self.gpus.as_mut().expect("mesh TP staging gpus taken")
    }

    fn into_parts(mut self) -> (Gpus, Vec<qwen35::mesh_tp::Qwen35TpRank>) {
        (
            self.gpus.take().expect("mesh TP into_parts called twice"),
            std::mem::take(&mut self.ranks),
        )
    }
}

impl Drop for Qwen35MeshTpStaging {
    fn drop(&mut self) {
        let Some(mut gpus) = self.gpus.take() else {
            return;
        };
        if let Err(error) = gpus.drain_mesh_edges() {
            eprintln!(
                "[loader] dense qwen TP mesh load failed and the mesh drain failed ({error}); \
                 resources quarantined until process restart"
            );
            std::mem::forget(std::mem::take(&mut self.ranks));
            std::mem::forget(gpus);
            return;
        }
        for (rank, state) in self.ranks.drain(..).enumerate() {
            if let Some(gpu) = gpus.devices.get_mut(rank) {
                state.free_gpu(gpu);
            }
        }
        for gpu in &mut gpus.devices {
            let _ = gpu.bind_thread();
            gpu.invalidate_weight_caches();
            gpu.invalidate_graph_state();
            gpu.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
    }
}

/// Dense Qwen tensor parallelism on the generic mesh executor
/// (`HIPFIRE_DENSE_TP_ROUTE=mesh`): the same signature and preamble as
/// [`load_model_tp_qwen35_dense`], then `Qwen35TpRank`s provisioned with the
/// executor's byte-qualified reduction transport. AR decode and chunked
/// prefill only: MTP, uneven layouts and non-legacy KV backends are refused
/// before any GPU allocation.
#[allow(clippy::too_many_arguments)]
fn load_model_tp_qwen35_mesh(
    path: &str,
    max_seq: usize,
    tp: usize,
    kv_mode: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
    qwen_default_q8: bool,
    kv_backend: Option<&str>,
    state_quant: Option<&str>,
    mtp_path: Option<&Path>,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
) -> Result<LoadedModel, String> {
    use hipfire_arch_qwen35::qwen35::mesh_tp::{
        qwen35_tp_plan_for_ranks, tp_rank_layouts, Qwen35TpProgram, Qwen35TpRank,
    };
    use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank, BandPlacement};
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};

    let _ = mtp_path;
    let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("qwen35 config: {e}"))?;
    let shard = ShardConfig::new(tp, false, 0, ExpertAssign::Stride)
        .map_err(|e| format!("dense TP ShardConfig: {e}"))?;
    let state_quant_resolved = parse_state_quant(state_quant)?;
    let kv_backend_raw = kv_backend.unwrap_or("legacy");
    let kv_backend_resolved: KvBackend = kv_backend_raw.parse().map_err(|err| format!("{err}"))?;
    let config_mode = hipfire_runtime::config::get();
    let kv_raw = kv_mode.unwrap_or(config_mode.kv_mode.as_str());
    let pair = kv_mode::resolve_kv_pair(
        kv_raw,
        kv_k,
        kv_v,
        &kv_mode::QWEN35_TP_POLICY,
        "multi-gpu",
        qwen_default_q8,
    )
    .map_err(|e| format!("Qwen dense TP KV: {e}"))?;
    let kv_mode_resolved = pair.k();
    let Some(v_mode) = pair.v() else {
        return Err("Qwen native KV has no multi-GPU constructor".into());
    };
    if v_mode != llama::VMode::Q8 {
        return Err("Qwen dense TP has no multi-GPU Lloyd-V constructor; use --kv-v q8".into());
    }
    eprintln!(
        "  KV cache: requested mode={kv_raw}, effective K={} V={}",
        kv_mode::qwen_k_display_name(kv_mode_resolved),
        kv_mode::qwen_v_display_name(v_mode)
    );
    // Preflight weights before GPU allocation (validates qt geometry/blob/sidecar).
    qwen35::preflight_weights_dense_tp(&hfq, &config, &shard)?;

    // Refusals of the mesh route, all before any GPU allocation.
    if spec.mtp == Some(true) {
        return Err(
            "MTP is not wired on the mesh TP route; set HIPFIRE_DENSE_TP_ROUTE=auto \
                    (or hand) for dense-TP MTP, or use speculation off"
                .into(),
        );
    }
    if spec.ngram_draft == Some(true) {
        eprintln!("  dense TP: n-gram drafter refused (block verify is not wired on the mesh)");
    }
    tp_rank_layouts(&config, tp).map_err(|e| format!("dense TP mesh layout: {e}"))?;

    let arch_id = hfq.arch_id;
    let (chat_template, chat_template_source) = resolve_chat_template(&hfq, path);
    let rec = hfq.recommended_sampling();
    let eos_tok = {
        let ids = tokenizer.encode("<|im_end|>");
        if ids.len() == 1 {
            ids[0]
        } else {
            config.eos_token
        }
    };

    let gpus = Gpus::init_tp(tp, config.n_layers).map_err(|e| format!("init_tp: {e:?}"))?;
    if gpus.devices.len() != tp {
        return Err(format!(
            "init_tp gave {} devices, expected tp={tp}",
            gpus.devices.len()
        ));
    }
    let mut staging = Qwen35MeshTpStaging::new(gpus);
    for rank in 0..tp {
        ensure_vmm_ready_for_load(&mut staging.gpus_mut().devices[rank])
            .map_err(|err| format!("dense TP VMM preflight rank {rank}: {err}"))?;
    }
    // Mesh work never runs on the null stream: every device owns a launch
    // stream before the edges are provisioned.
    hipfire_runtime::ep::ensure_rank_streams(staging.gpus_mut())
        .map_err(|e| format!("dense TP ensure_rank_streams: {e:?}"))?;
    // The hand serve route's prefill chunk; also the rank batch scratch rows.
    let chunk = qwen35::prefill_max_batch_tp(&staging.gpus_mut().devices[0], tp).max(1);
    for rank in 0..tp {
        let loaded = Qwen35TpRank::load(
            &mut hfq,
            &config,
            &mut staging.gpus_mut().devices[rank],
            rank,
            tp,
            kv_mode_resolved,
            kv_backend_resolved,
            state_quant_resolved,
            max_seq,
            chunk,
        )
        .map_err(|e| format!("dense TP mesh load rank {rank}: {e}"))?;
        staging.ranks.push(loaded);
    }
    drop(hfq);

    let mesh = staging.gpus_mut().mesh.clone();
    let mut admitted_ranks = Vec::with_capacity(tp);
    for (logical, gpu) in staging.gpus_mut().devices.iter().enumerate() {
        gpu.bind_thread()
            .map_err(|e| format!("dense TP bind rank {logical}: {e:?}"))?;
        let pci_bus_id = gpu
            .hip
            .device_pci_bus_id(gpu.device_id)
            .map_err(|e| format!("dense TP PCI bus id rank {logical}: {e:?}"))?;
        admitted_ranks.push(AdmittedRank {
            logical,
            device_ordinal: gpu.device_id,
            pci_bus_id: Some(pci_bus_id),
            arch: gpu.arch.clone(),
        });
    }
    let admitted = AdmittedMesh {
        mesh: mesh.clone(),
        ranks: admitted_ranks,
        edges: Vec::new(),
        family: "qwen35-tp".into(),
        numerical_policy: "f32".into(),
    };
    let placement =
        BandPlacement::uniform(&mesh, 1).map_err(|e| format!("dense TP mesh placement: {e}"))?;
    let max_bytes = (chunk * config.dim * std::mem::size_of::<f32>()).next_multiple_of(16);
    // TP2 whole-token decode graph (`HIPFIRE_TP_FULL_GRAPH`, on by default for
    // an exact gfx1100 or gfx1201 pair), as the hand loader: the runtime
    // admits it and prepares its signal tape (one barrier per collective, two
    // per layer) and payload exchange before peer access is enabled, which
    // the provisioning's `qualify_peer` does.
    let barriers = config.n_layers * 2;
    let peer = staging
        .gpus_mut()
        .can_access_peer_all()
        .map_err(|e| format!("dense TP mesh can_access_peer_all: {e:?}"))?;
    if peer {
        let whole_token_graph = staging
            .gpus_mut()
            .prepare_tp2_whole_token_graph(barriers, config.dim * std::mem::size_of::<f32>())
            .map_err(|e| format!("prepare TP2 whole-token graph: {e:?}"))?;
        if whole_token_graph {
            let arch = staging.gpus_mut().devices[0].arch.clone();
            eprintln!(
                "[loader] dense qwen {arch} mesh TP2 whole-token graph signal tape: {barriers} barriers"
            );
        } else if staging
            .gpus_mut()
            .prepare_tp_push_whole_token_graph(config.dim * std::mem::size_of::<f32>())
        {
            let arch = staging.gpus_mut().devices[0].arch.clone();
            eprintln!(
                "[loader] dense qwen {arch} mesh TP{tp} whole-token graph peer-push mailboxes: {barriers} collectives"
            );
        }
    }
    staging
        .gpus_mut()
        .provision_mesh_edges(&admitted, &placement, max_bytes)
        .map_err(|e| format!("dense TP mesh provisioning: {e}"))?;
    let transport = staging
        .gpus_mut()
        .mesh_tp_transport()
        .ok_or("dense TP mesh provisioning left no TP transport")?;
    // Every rank's buffers exist now. Idempotent after the provisioning's
    // qualification; it sets the flag the TP2 direct-peer reduction (and so
    // the whole-token graph) requires. Emulated ranks share a device and
    // report `false`, which keeps the copy-fold reduction.
    if transport == "peer" {
        staging
            .gpus_mut()
            .enable_peer_all()
            .map_err(|e| format!("dense TP mesh enable_peer_all: {e:?}"))?;
    }

    let plan = qwen35_tp_plan_for_ranks(&config, &staging.ranks, &mesh)
        .map_err(|e| format!("dense TP mesh plan: {e}"))?;
    if plan.collectives.len() != barriers {
        return Err(format!(
            "dense TP mesh plan has {} collectives per epoch, not the {barriers} \
             (two per layer) the whole-token graph signal tape is sized for",
            plan.collectives.len()
        ));
    }
    let tp_epoch = staging
        .gpus_mut()
        .mesh_tp_epoch()
        .ok_or("dense TP mesh provisioning left no TP group")?;
    let program = Qwen35TpProgram::new(&config, &staging.ranks, &plan, tp_epoch)
        .map_err(|e| format!("dense TP mesh program: {e}"))?;
    eprintln!(
        "[loader] dense qwen TP route=mesh (generic mesh executor): {tp} ranks, \
         transport={transport}, prefill chunk={chunk}, {} collectives/epoch",
        plan.collectives.len()
    );
    eprintln!(
        "[loader] dense qwen TP exchange={} (HIPFIRE_TP_EXCHANGE)",
        hipfire_runtime::multi_gpu::tp_exchange::tp_exchange_label(transport == "peer")
    );

    let (gpus, ranks) = staging.into_parts();
    // No drafter runs on the mesh route: `mtp=on` was refused above and block
    // verify is not wired, so the speculator is built with neither.
    let mesh_spec = hipfire_runtime::loader_api::SpecLoadCfg {
        ngram_draft: Some(false),
        ..spec
    };
    let speculator =
        crate::spec_build::build_speculator(arch_id, None, None, true, max_seq, mesh_spec);

    Ok(LoadedModel {
        speculator,
        ep: Some(EpState {
            gpus,
            inner: EpArch::Qwen35MeshTp {
                global: config,
                ranks,
                plan,
                program,
                next_op: 0,
                pending: Vec::new(),
            },
        }),
        qwen35_eos_tok: eos_tok,
        rec_temperature: rec.and_then(|r| r.temperature),
        rec_top_p: rec.and_then(|r| r.top_p),
        rec_top_k: rec.and_then(|r| r.top_k.map(|k| k as f32)),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
            chat_template_source,
        )
    })
}

/// Retry pending VMM arenas and refuse progress while any remain registered.
pub fn ensure_vmm_ready_for_load(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    gpu.ensure_vmm_cleaned().map_err(|err| {
        format!(
            "load refused: the previous model's GPU virtual memory is still being released ({err}); retry the unload, or restart the process"
        )
    })
}

// ─── Unload ───────────────────────────────────────────────────────────

/// True when `m` is a Qwen4 mesh-executor PP load: the bundle holds the band
/// bundles and plans while `pp_gpus` is the sole hardware owner (`Gpus` is
/// `!Send`, so it cannot live inside the `Send` bundle); `m.pp` keeps the
/// admitted PP degree and `pp_dn_la_to_device` stays `None`.
fn qwen4_pp_mesh_owner(m: &LoadedModel) -> bool {
    m.state.as_ref().is_some_and(|state| {
        (state.as_ref() as &dyn Any)
            .downcast_ref::<hipfire_arch_qwen4::bundle::Qwen4Bundle>()
            .is_some_and(|bundle| bundle.pp_mesh_active())
    })
}

/// What unload reports when the Qwen4 PP teardown failed: the bundle's mesh
/// teardown drains every epoch and edge before freeing anything, and a failed
/// drain keeps all resources (devices, bands, edges) alive and unreclaimed.
fn qwen4_pp_quarantine_report(error: &str) -> String {
    format!(
        "qwen4 topology.pp unload failed ({error}); mesh resources quarantined until process restart"
    )
}

/// Unload a Qwen4 mesh-executor PP model. `Qwen4Bundle::pp_free_gpu` drains
/// epochs and edges first and then frees every band on its own device; a
/// failed teardown quarantines: the bundle leaks the root and bands, and the
/// `Gpus` and the rest of the model are leaked here, never freed under
/// possibly in-flight work. Unload reports the failure.
fn unload_qwen4_pp(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    if let Some(spec) = m.speculator.take() {
        spec.free(gpu);
    }
    if let Some(ev) = m.eviction.take() {
        ev.free_gpu(gpu);
    }
    for (_, snap) in std::mem::take(&mut m.prefill_checkpoints) {
        snap.free_gpu(gpu);
    }
    for (_, snap) in std::mem::take(&mut m.dflash_checkpoints) {
        snap.free_gpu(gpu);
    }
    let Some(mut gpus) = m.pp_gpus.take() else {
        std::mem::forget(m);
        return Err(qwen4_pp_quarantine_report("the PP model carries no Gpus"));
    };
    let bundle =
        match m.state.take().map(|state| {
            (state as Box<dyn Any>).downcast::<hipfire_arch_qwen4::bundle::Qwen4Bundle>()
        }) {
            Some(Ok(bundle)) => bundle,
            other => {
                std::mem::forget(other);
                std::mem::forget(gpus);
                std::mem::forget(m);
                return Err(qwen4_pp_quarantine_report(
                    "model state is not a Qwen4 bundle",
                ));
            }
        };
    if let Err(error) = hipfire_arch_qwen4::bundle::Qwen4Bundle::pp_free_gpu(*bundle, &mut gpus) {
        std::mem::forget(gpus);
        std::mem::forget(m);
        return Err(qwen4_pp_quarantine_report(&error.to_string()));
    }
    for device in gpus.devices.iter_mut() {
        let _ = device.bind_thread();
        device.invalidate_weight_caches();
        device.invalidate_graph_state();
        device.drain_pool();
    }
    drop(gpus);
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    gpu.ensure_vmm_cleaned().map_err(|e| e.to_string())
}

pub fn unload_model(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    gpu.kv_fit_floor = None;
    // The next load (or none) starts lazy; a sealed route never outlives its model.
    if let Err(e) = gpu.abort_route_kernel_load() {
        eprintln!("  kernels: route reset on unload: {e}");
    }
    // The CPU-weights gate is model-scoped: every successful load sets it from
    // the resolved placement, so clear it here as the model leaves the device.
    gpu.set_host_cpu_weights(false);
    // EP unload-free. An EP model owns its own `Gpus` (the daemon's single `gpu`
    // is unused for tp>1). Without this branch a SUCCESSFUL EP unload leaked every
    // per-rank weight / state / partial. Free per-rank weights → state → partials
    // on each owning device, invalidate caches + graph state, drain each pool, then
    // drop the `Gpus` (tears down comms + devices). The daemon's `gpu` is untouched.
    // (The `partials` free here is what reclaims the ds4/minimax per-rank dummy
    // all-reduce buffer that would otherwise leak per load/unload cycle.)
    if let Some(ep) = m.ep.take() {
        let EpState { mut gpus, inner } = ep;
        let mut ep_first_err: Option<String> = None;
        match inner {
            EpArch::Ds4 {
                weights,
                state,
                partials,
                prefill,
                ..
            } => {
                for (r, w) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        w.free_gpu(dev);
                    }
                }
                for (r, s) in state.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        s.free_gpu(dev);
                    }
                }
                for (r, p) in partials.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        let _ = dev.free_tensor(p);
                    }
                }
                for (r, pbs) in prefill.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        pbs.free_gpu(dev);
                    }
                }
            }
            EpArch::Minimax {
                weights,
                state,
                partials,
                ..
            } => {
                for (r, w) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        w.free_gpu(dev);
                    }
                }
                for (r, s) in state.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        s.free_gpu(dev);
                    }
                }
                for (r, p) in partials.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        let _ = dev.free_tensor(p);
                    }
                }
            }
            EpArch::Qwen35 {
                weights,
                kv_caches,
                dn_states,
                scratches,
                partials,
                batch,
                prefill_pbs,
                prefill_partials,
                ..
            } => {
                // Quiesce every rank before any free: in-flight prefill/decode
                // kernels must retire before their scratch is reclaimed
                // (mirrors the bind+sync prologue of `free_gpu`).
                for dev in gpus.devices.iter_mut() {
                    let _ = dev.bind_thread();
                    let _ = dev.hip.device_synchronize();
                }
                // Tear down captured HIP graphs before reclaiming any tensors they
                // may still reference. Shared post-match invalidate stays for
                // other EP arches and remains idempotent here.
                for dev in gpus.devices.iter_mut() {
                    let _ = dev.bind_thread();
                    dev.invalidate_graph_state();
                }

                if let Some(b) = batch {
                    if let Err(e) = b.free_gpu(&mut gpus) {
                        if ep_first_err.is_none() {
                            ep_first_err = Some(e.to_string());
                        }
                    }
                }
                for (r, pbs) in prefill_pbs.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        if let Err(e) = pbs.free_gpu(dev) {
                            if ep_first_err.is_none() {
                                ep_first_err =
                                    Some(format!("unload qwen35 EP prefill PBS rank {r}: {e:?}"));
                            }
                        }
                    }
                }
                for (r, p) in prefill_partials.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        let _ = dev.free_tensor(p);
                    }
                }
                for (rank, scratch) in scratches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        if let Err(e) = scratch.free_gpu(dev) {
                            if ep_first_err.is_none() {
                                ep_first_err =
                                    Some(format!("unload qwen35 EP scratch rank {rank}: {e:?}"));
                            }
                        }
                    }
                }
                for (rank, state) in dn_states.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        state.free_gpu(dev);
                    }
                }
                for (rank, kv) in kv_caches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        let _ = kv.free_gpu(dev);
                    }
                }
                for (r, w) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        w.free_gpu(dev);
                    } else if ep_first_err.is_none() {
                        ep_first_err = Some(format!("unload qwen35: missing device for rank {r}"));
                    }
                }
                for (r, p) in partials.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(r) {
                        let _ = dev.bind_thread();
                        let _ = dev.free_tensor(p);
                    }
                }
            }
            EpArch::Qwen35DenseTp {
                weights,
                kv_caches,
                dn_states,
                scratches,
                ..
            } => {
                for (rank, scratch) in scratches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        if let Err(e) = scratch.free_gpu(dev) {
                            if ep_first_err.is_none() {
                                ep_first_err = Some(format!(
                                    "unload dense qwen TP scratch rank {rank}: {e:?}"
                                ));
                            }
                        }
                    }
                }
                for (rank, state) in dn_states.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        state.free_gpu(dev);
                    }
                }
                for (rank, kv) in kv_caches.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        if let Err(e) = kv.free_gpu(dev) {
                            if ep_first_err.is_none() {
                                ep_first_err =
                                    Some(format!("unload dense qwen TP KV rank {rank}: {e:?}"));
                            }
                        }
                        // Per-rank VMM teardown gate (mirrors the single-GPU
                        // unload tail): retry arenas retained by a failed free
                        // and refuse a clean handoff while any remain. No-op
                        // success for contiguous (no arenas registered).
                        if let Err(e) = dev.ensure_vmm_cleaned() {
                            if ep_first_err.is_none() {
                                ep_first_err =
                                    Some(format!("unload dense qwen TP VMM rank {rank}: {e:?}"));
                            }
                        }
                    }
                }
                for (rank, weights) in weights.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        let _ = dev.bind_thread();
                        weights.free_gpu(dev);
                    }
                }
            }
            EpArch::Qwen35MeshTp {
                ranks, mut pending, ..
            } => {
                // Retire the undrained epochs (one ticketed drain), then stop
                // and drain every epoch and mesh edge before freeing
                // anything the epochs touch; a failed drain quarantines, so
                // the ranks and the devices leak instead of being freed under
                // possibly in-flight work.
                let pending_drain = if pending.is_empty() {
                    Ok(())
                } else {
                    gpus.drain_mesh_epochs(std::mem::take(&mut pending))
                };
                if let Err(e) = pending_drain.and_then(|()| gpus.drain_mesh_edges()) {
                    std::mem::forget(ranks);
                    std::mem::forget(gpus);
                    return Err(format!(
                        "unload dense qwen TP mesh: {e}; mesh resources quarantined until process restart"
                    ));
                }
                for (rank, state) in ranks.into_iter().enumerate() {
                    if let Some(dev) = gpus.devices.get_mut(rank) {
                        state.free_gpu(dev);
                        // Per-rank VMM teardown gate, as the hand route's.
                        if let Err(e) = dev.ensure_vmm_cleaned() {
                            if ep_first_err.is_none() {
                                ep_first_err =
                                    Some(format!("unload dense qwen TP VMM rank {rank}: {e:?}"));
                            }
                        }
                    }
                }
            }
        }
        // Ahead of the pool teardown: the EP arm returns before the single-GPU
        // `spec.free(gpu)` below, and a mesh drafter's per-rank buffers are not
        // reachable from the daemon's `gpu`.
        if let Some(spec) = m.speculator.take() {
            spec.free_multi(&mut gpus);
        }
        // Reclaim unleased peer-rooted collective scratch before device pool
        // teardown. Idempotent across EP variants; fold first error into
        // ep_first_err so unload reports cleanup failure.
        if let Err(e) = gpus.free_peer_reduce_scratch() {
            if ep_first_err.is_none() {
                ep_first_err = Some(format!("unload dense TP peer reduce scratch: {e:?}"));
            }
        }

        for dev in gpus.devices.iter_mut() {
            let _ = dev.bind_thread();
            dev.invalidate_weight_caches();
            dev.invalidate_graph_state();
            dev.drain_pool();
        }
        let _ = gpus.free_tp_graph_signals();
        if let Some(b) = m.qwen35_mut() {
            if let Some(batch_state) = b.qwen35_decode_batch.take() {
                // Single-GPU Qwen batch is not expected on EP, but free it on the
                // provided single gpu before multi-device teardown to avoid leaking
                // if staging ever leaves residual state.
                let _ = batch_state.free_gpu(gpu);
            }
        }
        let _ = gpu;
        if let Some(err) = ep_first_err {
            return Err(err);
        }
        return Ok(());
        // `gpus` drops here, tearing down comms + devices.
    }
    // Qwen4 mesh-executor PP: the root bundle owns the `Gpus` (`pp_gpus` is
    // `None`), so it must leave before the legacy Qwen3.5 `m.pp > 1` arm.
    if qwen4_pp_mesh_owner(&m) {
        return unload_qwen4_pp(m, gpu);
    }
    if m.pp > 1 {
        if let Some(b) = m.qwen35_mut() {
            if let Some(batch_state) = b.qwen35_decode_batch.take() {
                // Single-GPU batch state is not expected for pp>1, but free it on
                // the provided single gpu before multi-device teardown to avoid
                // leaking if a test ever stages it.
                let _ = batch_state.free_gpu(gpu);
            }
        }
        let mut gpus = m.pp_gpus.take().expect("pp>1 must carry pp_gpus");
        if let Some(state) = m.state.take() {
            // multi-GPU resources. Otherwise just drop the box (original match did
            // the same — non-qwen35 pp>1 is unreachable and had an empty arm).
            if let Ok(mut b) =
                (state as Box<dyn Any>).downcast::<hipfire_arch_qwen35::Qwen35Bundle>()
            {
                if let Some(scratch_set) = b.pp_scratch_set.take() {
                    scratch_set.free_gpu_multi(&mut gpus);
                }
                b.kv_cache.free_gpu_multi(&mut gpus);
                let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                b.dn_state.free_gpu_multi(&mut gpus, &la_to_device);
                b.weights.free_gpu_multi(&mut gpus);
                b.scratch.free_gpu(&mut gpus.devices[0]);
            }
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return Ok(());
    }
    // Quiesce retained-PM4 (if any) before freeing any captured owner. Unknown
    // quiescence → keep model quarantined; daemon restart is containment.
    if let Some(spec) = &mut m.speculator {
        if let Err(reason) = spec.quiesce(gpu) {
            eprintln!("dflash verify PM4: unload refused — unknown quiescence: {reason}");
            std::mem::forget(m);
            return Err(format!(
                "dflash verify PM4: unload refused after unknown quiescence ({reason}); \
                 model remains quarantined until process restart"
            ));
        }
    }
    if let Some(spec) = m.speculator.take() {
        // Frees the drafter's GPU buffers (draft weights + scratch) AND its
        // checkpoint ring — a drafter that forgets is a compile error, not a
        // silent VRAM leak. The vestigial `m.dflash_checkpoints` (now always
        // empty) is still drained below for defense-in-depth.
        spec.free(gpu);
    }
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    let mut first_err: Option<String> = None;
    let mut note = |r: Result<(), String>| {
        if let Err(err) = r {
            if first_err.is_none() {
                first_err = Some(err);
            }
        }
    };
    for (_, snap) in m.prefill_checkpoints {
        snap.free_gpu(gpu);
    }
    for (_, snap) in m.dflash_checkpoints {
        snap.free_gpu(gpu);
    }
    // Weight-host lease (imported weights): taken out before the bundle free
    // so the infallible `ArchModel::free_gpu` cannot drop it; released at
    // the end only when graphs, pool and VMM teardown are proven clean.
    let mut weight_lease = None;
    if let Some(state) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) {
        if let Some(batch_state) = state.qwen35_decode_batch.take() {
            let _ = batch_state.free_gpu(gpu);
        }
        weight_lease = state.weight_lease.take();
    }
    // Free arch-specific GPU state from the carrier bundle via `ArchModel::free_gpu`.
    if let Some(state) = m.state {
        state.free_gpu(gpu);
    }
    // deepseek4 single-GPU scratch now lives inside `Deepseek4Bundle.pbs` and is
    // freed via `ArchModel::free_gpu` above — no separate `m.deepseek4_pbs` block.
    // Qwen35 vision tower is now inside the bundle and freed via
    // `ArchModel::free_gpu` above. Nothing to do here — the old
    // `m.vision_weights` field is gone.
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
    // After ordinary frees/pool drain, retry any VMM arenas retained by a
    // failed free_tensor. Success is reported only when none remain.
    note(gpu.ensure_vmm_cleaned().map_err(|e| e.to_string()));
    match (first_err, weight_lease) {
        (Some(err), Some(lease)) => {
            drop(lease);
            Err(format!(
                "{err}; weight-host lease closed without release (teardown not proven clean)"
            ))
        }
        (Some(err), None) => Err(err),
        (None, Some(lease)) => lease.release(),
        (None, None) => Ok(()),
    }
}

#[cfg(test)]
mod ep_admission_tests {
    use super::{admission, ep_admission, qwen35_ep_moe_topology_refusal};
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::path::{Path, PathBuf};

    #[test]
    fn qwen35_moe_ep_topology_gate_admits_2_and_4_where_sealed_loader_can() {
        // Supported: tp=2|4 dividing the expert count (arch 6 and mis-stamped 5).
        assert_eq!(qwen35_ep_moe_topology_refusal(6, 128, 2), None);
        assert_eq!(qwen35_ep_moe_topology_refusal(6, 128, 4), None);
        assert_eq!(qwen35_ep_moe_topology_refusal(5, 128, 4), None);
        // Adjacent supported: dense Qwen3.5 (either arch id) keeps its EP path
        // at any degree — the dense branch never reaches this gate.
        assert_eq!(qwen35_ep_moe_topology_refusal(5, 0, 4), None);
        assert_eq!(qwen35_ep_moe_topology_refusal(6, 0, 3), None);
        // Unsupported degree refuses naming the supported set.
        let err = qwen35_ep_moe_topology_refusal(6, 128, 3).expect("tp=3 MoE EP must refuse");
        assert!(err.contains("Qwen3.5-MoE"), "reason names the model: {err}");
        assert!(err.contains("tp=2|4"), "reason names the set: {err}");
        assert!(err.contains('6'), "reason names the arch: {err}");
        // Indivisible expert count refuses naming both sides.
        let err = qwen35_ep_moe_topology_refusal(6, 6, 4).expect("indivisible experts must refuse");
        assert!(err.contains("num_experts=6"), "reason: {err}");
        assert!(err.contains("tp=4"), "reason: {err}");
    }

    #[test]
    fn ep_without_eparch_refuses_but_served_archs_admit() {
        // LFM2 (11), Cohere2 (12) and anything else with no `EpArch` variant:
        for arch in [11u32, 12, 13, 0, 99] {
            let err = match ep_admission(arch) {
                Ok(()) => panic!("arch {arch} + EP must refuse"),
                Err(e) => e,
            };
            assert!(
                err.contains("expert parallelism (tp>1) is not supported"),
                "reason: {err}"
            );
            assert!(
                err.contains(&arch.to_string()),
                "reason names the arch: {err}"
            );
        }
        // Adjacent supported: DS4, MiniMax and Qwen3.5 keep their EP entries.
        for arch in [5u32, 6, 9, 10] {
            assert!(ep_admission(arch).is_ok(), "arch {arch} + EP must admit");
        }
    }

    /// Tiny consumer-facing load owner used to exercise the admission boundary:
    /// a successful candidate would tear down and replace the active model, while
    /// a refusal must leave its identity and observable response untouched.
    #[derive(Debug, PartialEq, Eq)]
    struct ActiveModel {
        identity: &'static str,
        response: Vec<u8>,
    }

    impl ActiveModel {
        fn request(&self) -> Vec<u8> {
            self.response.clone()
        }
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct LoadEffects {
        teardowns: usize,
        allocations: usize,
        publications: usize,
    }

    /// `tag` must be unique per caller: tests run in one process, so a
    /// pid-only filename is shared by parallel tests that write, read, and
    /// unlink the same path — a flake, not a fixture.
    fn qwen35_moe_fixture(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hipfire-loader-qwen35-moe-admission-{tag}-{}.hfq",
            std::process::id()
        ));
        let metadata = serde_json::json!({
            "config": {
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "vocab_size": 8,
                "num_experts": 4,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 2,
                "shared_expert_intermediate_size": 2
            }
        })
        .to_string();
        write_hfqm_package_mem(
            &path,
            6,
            &metadata,
            &[HfqMemTensor {
                name: "model.embed_tokens.weight".into(),
                quant_type: 1,
                shape: vec![8, 4],
                group_size: 0,
                data: vec![0; 32],
            }],
        )
        .expect("write qwen35 MoE admission fixture");
        path
    }

    /// Mirror the daemon's source-admission boundary without touching a GPU:
    /// destructive effects are reachable only after `admit_source` succeeds.
    /// `tp` is threaded so the unsupported-degree refusal is exercised through
    /// the same entry the daemon uses.
    fn attempt_candidate_swap(
        candidate: &Path,
        tp: usize,
        request: admission::KvBackendRequest,
        arch: &str,
        active: &mut ActiveModel,
        effects: &mut LoadEffects,
    ) -> Result<(), String> {
        let admitted = admission::admit_source(
            candidate.to_str().expect("fixture path is UTF-8"),
            tp,
            1,
            request,
            None,
            arch,
            None,
            "auto",
            None,
            4096,
            admission::KvBackendHints::without_device(),
        )?;

        // These represent the externally observable transaction stages in the
        // daemon: prior-model teardown, candidate allocation, publication.
        effects.teardowns += 1;
        active.response.clear();
        effects.allocations += 1;
        effects.publications += 1;
        drop(admitted);
        Ok(())
    }

    /// An unsupported EP degree for a routed Qwen3.5 candidate is refused
    /// before teardown while an earlier model is active. No
    /// teardown/allocation/publication occurs, and the active model remains
    /// identifiable and usable. The exact refusal text is pinned by the pure
    /// `qwen35_ep_moe_topology_refusal` test above; here only the boundary
    /// matters (on GPU boxes the device-count gate may fire first, which is
    /// equally before-teardown).
    #[test]
    fn qwen35_moe_ep_unsupported_degree_preserves_active_model() {
        let candidate = qwen35_moe_fixture("ep-degree");
        let mut active = ActiveModel {
            identity: "qwen3.6:27b-a3b-active",
            response: b"active-model-response".to_vec(),
        };
        let before_identity = active.identity;
        let before_response = active.request();
        let mut effects = LoadEffects::default();

        attempt_candidate_swap(
            &candidate,
            3,
            admission::KvBackendRequest::Automatic,
            "gfx1201",
            &mut active,
            &mut effects,
        )
        .expect_err("Qwen3.5 MoE EP tp=3 candidate must refuse before teardown");
        assert_eq!(effects, LoadEffects::default());
        assert_eq!(active.identity, before_identity);
        assert_eq!(active.request(), before_response);

        let _ = std::fs::remove_file(candidate);
    }
    #[test]
    fn explicit_unsupported_vmm_preserves_active_model() {
        let candidate = qwen35_moe_fixture("vmm-explicit");
        let mut active = ActiveModel {
            identity: "resident",
            response: b"still-serving".to_vec(),
        };
        let before = active.request();
        let mut effects = LoadEffects::default();
        let refusal = attempt_candidate_swap(
            &candidate,
            1,
            admission::KvBackendRequest::Explicit(hipfire_runtime::kv_backend::KvBackend::Vmm),
            "gfx1100",
            &mut active,
            &mut effects,
        )
        .unwrap_err();
        assert!(refusal.contains("vmm") && refusal.contains("not available"));
        assert_eq!(effects, LoadEffects::default());
        assert_eq!(active.request(), before);
        let _ = std::fs::remove_file(candidate);
    }
}

#[cfg(test)]
mod lfm2_batch_admission_tests {
    #[test]
    fn lfm2_continuous_batch_never_admits_but_qwen_still_does() {
        use super::{carrier_for, continuous_batch_route};
        // Arch 11: the route refuses, so staging takes the fallback arm and
        // no Lfm2DecodeBatchState is ever allocated; the caps gate in
        // `is_batch_request_eligible` (and the engine scheduler) agrees.
        assert_eq!(continuous_batch_route(11), None);
        let caps = carrier_for(11).expect("lfm2moe carrier").caps();
        assert!(
            !caps.supports_continuous_batch,
            "lfm2moe caps must stay false while no batch path is servable"
        );
        // Adjacent supported: qwen35 5|6 still admit continuous batching.
        assert!(continuous_batch_route(5).is_some());
        assert!(continuous_batch_route(6).is_some());
        assert!(
            carrier_for(5)
                .expect("qwen35 carrier")
                .caps()
                .supports_continuous_batch
        );
        // qwen4 (16) admits the exact fn-lanes route, staged only under
        // `serve.vmm_batch`.
        assert_eq!(
            continuous_batch_route(16),
            Some(super::ContinuousBatchRoute::Qwen4)
        );
        assert!(
            carrier_for(16)
                .expect("qwen4 carrier")
                .caps()
                .supports_continuous_batch
        );
    }
}

#[cfg(test)]
mod kho_scratch_warning_tests {
    use super::kho_scratch_load_warning;

    #[test]
    fn kho_load_warning_is_uma_only_and_needs_one_gib() {
        let hipx = Some(13_536u64 << 20);
        assert_eq!(
            kho_scratch_load_warning(true, hipx).as_deref(),
            Some("  warning: KHO scratch reserves 13.2 GiB unusable by the GPU; boot with kho=off to reclaim it")
        );
        assert_eq!(kho_scratch_load_warning(false, hipx), None);
        assert_eq!(kho_scratch_load_warning(true, None), None);
        assert_eq!(kho_scratch_load_warning(true, Some((1 << 30) - 1)), None);
        assert!(kho_scratch_load_warning(true, Some(1 << 30)).is_some());
    }
}

#[cfg(test)]
mod asst_turn_cache_tests {
    use super::AsstTurnCache;
    use hipfire_runtime::prompt_frame::{CachedAssistantBody, CachedAssistantTurn};

    fn turn(reasoning: Option<&str>, marker: u32) -> CachedAssistantTurn {
        CachedAssistantTurn {
            reasoning: reasoning.map(|text| CachedAssistantBody {
                token_ids: Vec::new(),
                text: text.to_string(),
            }),
            tools: Vec::new(),
            content: Some(CachedAssistantBody {
                token_ids: vec![marker],
                text: String::new(),
            }),
        }
    }

    fn marker(t: Option<&CachedAssistantTurn>) -> Option<u32> {
        t.and_then(|t| t.content.as_ref()).map(|c| c.token_ids[0])
    }

    #[test]
    fn distinct_reasoning_turns_with_same_content_fp_coexist() {
        let mut c = AsstTurnCache::with_cap(Some(8));
        c.insert(7, turn(Some("alpha"), 1));
        c.insert(7, turn(Some("beta"), 2));
        assert_eq!(marker(c.get_for_reasoning(&7, Some("alpha"))), Some(1));
        assert_eq!(marker(c.get_for_reasoning(&7, Some("beta"))), Some(2));
        assert_eq!(marker(c.get_for_reasoning(&7, Some("gamma"))), None);
    }

    #[test]
    fn empty_content_turns_with_different_reasoning_coexist_and_missing_equals_empty() {
        let mut c = AsstTurnCache::with_cap(Some(8));
        c.insert(0, turn(Some("think one"), 1));
        c.insert(0, turn(Some("think two"), 2));
        // Absent and empty reasoning are the same identity: replace, not coexist.
        c.insert(9, turn(None, 3));
        c.insert(9, turn(Some(""), 4));
        assert_eq!(marker(c.get_for_reasoning(&0, Some("think one"))), Some(1));
        assert_eq!(marker(c.get_for_reasoning(&0, Some("think two"))), Some(2));
        assert_eq!(marker(c.get_for_reasoning(&9, None)), Some(4));
        assert_eq!(marker(c.get_for_reasoning(&9, Some(""))), Some(4));
    }

    #[test]
    fn absent_or_empty_reasoning_misses_when_content_fp_is_ambiguous() {
        let mut c = AsstTurnCache::with_cap(Some(8));
        c.insert(7, turn(Some("alpha"), 1));
        c.insert(7, turn(Some("beta"), 2));
        assert_eq!(marker(c.get_for_reasoning(&7, None)), None);
        assert_eq!(marker(c.get_for_reasoning(&7, Some(""))), None);
        assert_eq!(marker(c.get(&7)), None);
    }

    #[test]
    fn unique_identity_serves_absent_or_empty_reasoning_fallback() {
        let mut c = AsstTurnCache::with_cap(Some(8));
        c.insert(7, turn(Some("alpha"), 1));
        assert_eq!(marker(c.get_for_reasoning(&7, None)), Some(1));
        assert_eq!(marker(c.get_for_reasoning(&7, Some(""))), Some(1));
        assert_eq!(marker(c.get(&7)), Some(1));
        // Nonempty reasoning is exact: no fallback to the unique entry.
        assert_eq!(marker(c.get_for_reasoning(&7, Some("other"))), None);
    }

    #[test]
    fn eviction_removes_ambiguity_and_clear_drops_all_indices() {
        let mut c = AsstTurnCache::with_cap(Some(2));
        c.insert(7, turn(Some("alpha"), 1));
        c.insert(7, turn(Some("beta"), 2));
        assert_eq!(marker(c.get(&7)), None);
        // Third distinct entry evicts the LRU ("alpha"), leaving "beta" unique.
        c.insert(8, turn(None, 3));
        assert_eq!(marker(c.get_for_reasoning(&7, Some("alpha"))), None);
        assert_eq!(marker(c.get(&7)), Some(2));
        assert!(c.contains_key(&7));
        // Evicting the last identity of a content fp removes it entirely.
        c.insert(10, turn(None, 4));
        c.insert(11, turn(None, 5));
        assert!(!c.contains_key(&7));
        assert_eq!(marker(c.get(&7)), None);
        c.clear();
        assert!(c.is_empty());
        assert!(!c.contains_key(&11));
        assert_eq!(marker(c.get(&11)), None);
        // Cache remains usable and unambiguous after clear.
        c.insert(7, turn(Some("gamma"), 6));
        assert_eq!(marker(c.get(&7)), Some(6));
    }

    #[test]
    fn replacement_keeps_slot_and_refreshes_mru_under_cap() {
        let mut c = AsstTurnCache::with_cap(Some(2));
        c.insert(1, turn(Some("a"), 1));
        c.insert(2, turn(Some("b"), 2));
        // Replace oldest: no growth, becomes MRU, new body served.
        c.insert(1, turn(Some("a"), 10));
        assert_eq!(marker(c.get_for_reasoning(&1, Some("a"))), Some(10));
        // Next insert evicts fp 2 (now LRU), not the refreshed fp 1.
        c.insert(3, turn(Some("c"), 3));
        assert_eq!(marker(c.get_for_reasoning(&2, Some("b"))), None);
        assert_eq!(marker(c.get_for_reasoning(&1, Some("a"))), Some(10));
        assert_eq!(marker(c.get_for_reasoning(&3, Some("c"))), Some(3));
        // A lookup hit also refreshes MRU: touch 1, then insert evicts 3.
        let _ = c.get_for_reasoning(&1, Some("a"));
        c.insert(4, turn(Some("d"), 4));
        assert_eq!(marker(c.get_for_reasoning(&3, Some("c"))), None);
        assert_eq!(marker(c.get_for_reasoning(&1, Some("a"))), Some(10));
    }
}

#[cfg(test)]
#[cfg(test)]
mod wire_verb_tests {
    use super::{wire_verb, WireVerb};

    #[test]
    fn embed_reserved_for_arch_17_and_generate_is_the_default() {
        assert_eq!(wire_verb(17), WireVerb::Embed);
        assert!(!wire_verb(17).is_generate());
        // Every other exercised arch stays a generating model, so a new chat
        // arch needs no change here and only a non-generating one is added
        // explicitly.
        for a in [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40, 45] {
            assert_eq!(wire_verb(a), WireVerb::Generate, "arch {a}");
            assert!(wire_verb(a).is_generate(), "arch {a}");
        }
    }
}

#[cfg(test)]
mod registry_tests {
    use super::{resolve_deepseek4_compressor_cache_kv_mode, REGISTRY};

    /// `arch_label` is the one wire name for the load ACK, `diag` and the
    /// multi-slot preflight: a loaded dots-ocr model must never report the
    /// `qwen3` fallback.
    #[test]
    fn arch_label_names_vision_and_slot_arches() {
        use super::arch_label;
        assert_eq!(arch_label(8), "dots-ocr");
        assert_eq!(arch_label(5), "qwen3_5");
        assert_eq!(arch_label(6), "qwen3_5_moe");
        assert_eq!(arch_label(0), "qwen3");
    }

    #[test]
    fn deepseek4_kv_mode_is_truthful_and_fail_closed() {
        use hipfire_config::Deepseek4CompressorCache::{F16, F32};

        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(None).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("auto")).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("f32")).unwrap(),
            F32
        );
        assert_eq!(
            resolve_deepseek4_compressor_cache_kv_mode(Some("f16")).unwrap(),
            F16
        );
        // Sub-F16 selectors redirect to F16 rather than failing: DS4 stores
        // its compressor cache as F32 or F16 only, and F16 is the nearest
        // implemented storage below F32 — a widening relative to what these
        // ask for. Only unrecognised strings still fail closed.
        for mode in [
            "q8", "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4", "turbo", "turbo3", "turbo4",
        ] {
            assert_eq!(
                resolve_deepseek4_compressor_cache_kv_mode(Some(mode)).unwrap(),
                F16,
                "{mode} should redirect to F16"
            );
        }
        let error = resolve_deepseek4_compressor_cache_kv_mode(Some("nonsense")).unwrap_err();
        assert!(error.contains("not recognised"), "{error}");
    }

    /// Every known arch_id must be claimed by AT MOST one carrier, for both
    /// source namespaces (HFQ header ids and `derive_arch_id` dir ids). This
    /// guards the otherwise-silent first-match overlap in `load_model`: add a
    /// carrier whose `claims_arch_id` collides with an existing one and this
    /// fails in CI instead of mis-routing weights at runtime.
    #[test]
    fn carriers_are_disjoint() {
        // Sweep well past the assigned range, plus the reserved sentinels
        // (20 = DFlash draft, 0xFF = toy/template — neither should dispatch).
        let ids = (0u32..=64).chain([20, 0xFF]);
        for id in ids {
            for is_dir in [false, true] {
                let claimers: Vec<&str> = REGISTRY
                    .iter()
                    .filter(|c| c.claims_arch_id(id, is_dir))
                    .map(|c| c.name())
                    .collect();
                assert!(
                    claimers.len() <= 1,
                    "arch_id={id} is_dir={is_dir} claimed by multiple carriers: {claimers:?}"
                );
            }
        }
    }

    /// Pin the intended routing so a future probe edit can't silently move an
    /// existing model to the wrong carrier. `is_dir` matters in general, but
    /// Qwen2 routes to the qwen2 carrier in BOTH forms (HFQ id 7 and dir, which
    /// derives to id 7) so its Q/K/V attention biases load — the llama-family
    /// dir loader (id 1) drops them.
    #[test]
    fn known_ids_route_as_expected() {
        let cases: &[(u32, bool, &str)] = &[
            (7, false, "qwen2"),
            (7, true, "qwen2"),
            (5, false, "qwen35"),
            (6, false, "qwen35"),
            (5, true, "qwen35"),
            (6, true, "qwen35"),
            (0, false, "llama"),
            (1, false, "llama"),
            (0, true, "llama"),
            (1, true, "llama"),
            (8, false, "dots_ocr"),
            (9, false, "deepseek4"),
            (10, false, "minimax"),
            (11, false, "lfm2moe"),
            (12, false, "cohere2moe"),
            (19, false, "raon1_mini"),
            (40, false, "flux"),
            (40, true, "flux"),
            (45, false, "flux"),
            (45, true, "flux"),
        ];
        for &(id, is_dir, want) in cases {
            let got: Vec<&str> = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, is_dir))
                .map(|c| c.name())
                .collect();
            assert_eq!(
                got,
                vec![want],
                "arch_id={id} is_dir={is_dir} should route to exactly [{want}]"
            );
        }
    }

    /// The unassigned HFQ ids 2..=4 must reach NO carrier — this is the
    /// regression guard for fix B (the old `arch_id < 5` open range silently
    /// loaded them as llama).
    #[test]
    fn unassigned_low_ids_match_nothing() {
        for id in [2u32, 3, 4] {
            let n = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, false))
                .count();
            assert_eq!(n, 0, "arch_id={id} (unassigned) should match no carrier");
        }
    }

    /// The Onyx/Harmony template exactly as Muse Glimmer's .hfq carries it, checked in so
    /// the rewrite's substitution-count contract is pinned against the real artifact rather
    /// than a hand-written approximation. If upstream republishes the template with a
    /// different accessor shape, THIS TEST fails before a user ever sees a bare unframed
    /// prompt at runtime.
    const ONYX_TEMPLATE: &str =
        include_str!("../../hipfire-runtime/templates/muse-glimmer-onyx.jinja");

    #[test]
    fn muse_glimmer_onyx_template_rewrite() {
        use super::rewrite_muse_glimmer_onyx_template;

        // The real carried template must rewrite cleanly.
        let out = rewrite_muse_glimmer_onyx_template(ONYX_TEMPLATE)
            .expect("the checked-in Onyx template must rewrite");

        // Flat accessors: the runtime's `prompt_frame::ToolCall` is flat, so every
        // `tc.function.*` dereference must be gone or the template's own
        // `raise_exception` fires and the whole render falls back to a bare prompt.
        assert_eq!(out.matches("tc.function.").count(), 0);
        assert_eq!(
            out.matches("tc.name").count(),
            ONYX_TEMPLATE.matches("tc.function.name").count()
        );
        assert_eq!(
            out.matches("tc.arguments").count(),
            ONYX_TEMPLATE.matches("tc.function.arguments").count()
        );

        // The verbatim-splice branch must wrap `render_atem`'s body exactly once, so a
        // cached tool body replaces the regenerated ATEM XML.
        // `{%- if tc.rendered_body is defined and tc.rendered_body -%}{{- tc.rendered_body -}}`
        assert_eq!(out.matches("tc.rendered_body").count(), 3);
        assert!(out.contains("{%- if tc.rendered_body is defined and tc.rendered_body -%}"));
        assert!(out.contains("{%- endif -%}{%- endmacro -%}"));

        // Only `render_atem` is wrapped. `render_tool_defs` ends with a BYTE-IDENTICAL ATEM
        // tail (it embeds a worked example of the call syntax), so a `replace`-all here
        // would inject a stray `{%- endif -%}` into the tool-definition preamble and break
        // every tools-bearing prompt. Pin both halves: exactly one tail stays bare, exactly
        // one is wrapped.
        let bare_tail = "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endmacro -%}";
        let wrapped_tail =
            "{{- '</atem:invoke>\\n</atem:function_calls>' -}}{%- endif -%}{%- endmacro -%}";
        assert_eq!(
            out.matches(bare_tail).count(),
            1,
            "render_tool_defs tail must stay bare"
        );
        assert_eq!(
            out.matches(wrapped_tail).count(),
            1,
            "render_atem tail must be wrapped"
        );

        // Upstream drift must fail LOUDLY rather than silently leaving nested accessors.
        let missing = "{%- macro render_atem(tc) -%}hello{%- endmacro -%}";
        let err = rewrite_muse_glimmer_onyx_template(missing).unwrap_err();
        assert!(
            err.contains("expected 3 occurrences"),
            "unexpected error for drifted template: {err}"
        );
    }

    #[test]
    fn vision_caps_declare_images() {
        // Text-only arches must not declare image support; the two VL
        // arches (Qwen3.5-VL 5/6 and dots.ocr 8) must.
        let qwen35 = REGISTRY
            .iter()
            .find(|c| c.name() == "qwen35")
            .expect("qwen35 carrier missing");
        let dots = REGISTRY
            .iter()
            .find(|c| c.name() == "dots_ocr")
            .expect("dots_ocr carrier missing");
        assert!(
            qwen35.caps().supports_images,
            "qwen35 (arch 5/6) must declare supports_images"
        );
        assert!(
            dots.caps().supports_images,
            "dots_ocr (arch 8) must declare supports_images"
        );
        // VisionRoute must be declarative via caps, not just arch_id.
        assert_eq!(super::vision_route(5), super::VisionRoute::QwenVl);
        assert_eq!(super::vision_route(6), super::VisionRoute::QwenVl);
        assert_eq!(super::vision_route(8), super::VisionRoute::DotsOcr);
        assert_eq!(super::vision_route(11), super::VisionRoute::Lfm2Vl);
        assert_eq!(super::vision_route(0), super::VisionRoute::None);
        assert_eq!(super::vision_route(7), super::VisionRoute::None);
        assert_eq!(super::vision_route(9), super::VisionRoute::None);
        // Text-only carriers must stay false.
        assert!(
            REGISTRY
                .iter()
                .find(|c| c.name() == "lfm2moe")
                .unwrap()
                .caps()
                .supports_images,
            "lfm2moe (arch 11, lfm2_vl artifacts) must declare supports_images —              tower-less checkpoints still refuse images via has_vision_encoder()"
        );
        // EmbeddingGemma-2 (arch 18) declares the capability: the checkpoint's
        // vision tower ships in the arch-18 `.vl` pack, and the per-instance
        // gate is the loaded bundle's tower. It has no generate-side VL route
        // — embed inputs dispatch through the bundle — so `vision_route(18)`
        // must stay `None` even with the capability declared.
        let eg2 = REGISTRY
            .iter()
            .find(|c| c.name() == "embeddinggemma2")
            .expect("embeddinggemma2 carrier missing");
        assert!(
            eg2.caps().supports_images,
            "embeddinggemma2 (arch 18) must declare supports_images — tower-less trunks (no .vl pack) still refuse images via the bundle check"
        );
        assert_eq!(
            super::vision_route(18),
            super::VisionRoute::None,
            "embeddinggemma2 is embed-only: no generate-side VL route"
        );
        for name in [
            "qwen2",
            "llama",
            "deepseek4",
            "minimax",
            "cohere2moe",
            "gemma4",
            "muse_glimmer",
        ] {
            let c = REGISTRY.iter().find(|c| c.name() == name).unwrap();
            assert!(
                !c.caps().supports_images,
                "{name} must not declare supports_images"
            );
        }
    }

    /// Every carrier whose reasoning contract replays prior turns' reasoning in
    /// history (Qwen Jinja `<think>` replay, Muse Glimmer) must have serve
    /// forward `reasoning_content` for the arch label its load handshake
    /// reports. A miss renders an empty think block: the model loses its own
    /// earlier reasoning, and the next prompt stops extending the previous one,
    /// so the Qwen4 prefix cache never hits. DeepSeek4 and Gemma4 templates
    /// drop history reasoning, so they are not required.
    #[test]
    fn thinking_contract_arches_forward_history_reasoning() {
        use super::{arch_label, carrier_for};
        use saddle_core::caps::{arch_replays_history_reasoning, ReasoningContract};
        let mut covered = Vec::new();
        for id in 0u32..=255 {
            let Some(carrier) = carrier_for(id) else {
                continue;
            };
            let contract = carrier.caps().reasoning_contract;
            if !matches!(
                contract,
                ReasoningContract::QwenJinja | ReasoningContract::MuseGlimmer
            ) {
                continue;
            }
            // Maple (15) has no `arch_label` entry: it reports the fallback
            // `qwen3`, shared with Qwen3 dense, so serve cannot single it out.
            if id == 15 {
                continue;
            }
            let label = arch_label(id);
            assert!(
                arch_replays_history_reasoning(label),
                "arch {id} ({}) declares {contract:?}, but serve drops history \
                 reasoning_content for its label {label:?}",
                carrier.name()
            );
            covered.push(label);
        }
        for label in [
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4",
            "muse_glimmer",
            "raon1_mini",
        ] {
            assert!(covered.contains(&label), "{label} not covered: {covered:?}");
        }
    }

    /// Pin every carrier's declared `ArchCaps` AND every arch_id route-table
    /// output. The route tables in `lib.rs` (continuous_batch / bench_decode /
    /// vision / ep_prompt / ep_eos / generation_early) are ROUTING
    /// discriminators — each variant selects a distinct function body — so
    /// they cannot be expressed as a boolean capability and must not drift
    /// silently when a carrier's caps row is edited. If you intentionally
    /// re-route an architecture, update this pin in the same commit.
    #[test]
    fn caps_and_route_tables_are_pinned() {
        use super::{
            bench_decode_route, continuous_batch_route, ep_eos_route, ep_prompt_route,
            generation_early_route, img_route, vision_route, BenchDecodeRoute,
            ContinuousBatchRoute, EpEosRoute, EpPromptRoute, GenerationEarlyRoute, ImgRoute,
            VisionRoute,
        };
        use saddle_core::caps::{ArchCaps, DflashKind, ReasoningContract};

        let caps_of = |name: &str| -> ArchCaps {
            REGISTRY
                .iter()
                .find(|c| c.name() == name)
                .unwrap_or_else(|| panic!("{name} carrier missing"))
                .caps()
        };

        // ── Per-carrier declared capabilities (11 architectures) ──
        let text_only = ArchCaps::default();
        assert_eq!(caps_of("qwen2"), text_only);
        assert_eq!(
            caps_of("qwen35"),
            ArchCaps {
                reasoning_contract: ReasoningContract::QwenJinja,
                supports_continuous_batch: true,
                supports_ep_batch: true,
                dflash: Some(DflashKind::Qwen),
                supports_mtp: true,
                spec_excludes_adaptive: true,
                semantic_contract_version: Some(2),
                has_deltanet: true,
                supports_images: true,
            }
        );
        assert_eq!(
            caps_of("llama"),
            ArchCaps {
                dflash: Some(DflashKind::Llama),
                ..text_only
            }
        );
        assert_eq!(
            caps_of("dots_ocr"),
            ArchCaps {
                supports_images: true,
                ..text_only
            }
        );
        assert_eq!(
            caps_of("deepseek4"),
            ArchCaps {
                reasoning_contract: ReasoningContract::DeepSeek4,
                ..text_only
            }
        );
        assert_eq!(caps_of("minimax"), text_only);
        // lfm2moe declares supports_continuous_batch: false — the batch state
        // was allocated and never driven (eligibility always false), so the
        // capability is truthful only when false. Single-stream LFM unaffected.
        assert_eq!(
            caps_of("lfm2moe"),
            ArchCaps {
                supports_images: true,
                ..text_only
            }
        );
        assert_eq!(caps_of("cohere2moe"), text_only);
        assert_eq!(
            caps_of("raon1_mini"),
            ArchCaps {
                reasoning_contract: ReasoningContract::QwenJinja,
                semantic_contract_version: Some(2),
                ..text_only
            }
        );
        assert_eq!(
            caps_of("gemma4"),
            ArchCaps {
                reasoning_contract: ReasoningContract::GemmaBoolean,
                ..text_only
            }
        );
        assert_eq!(
            caps_of("muse_glimmer"),
            ArchCaps {
                reasoning_contract: ReasoningContract::MuseGlimmer,
                semantic_contract_version: Some(2),
                ..text_only
            }
        );

        // ── continuous_batch_route: 5|6 -> Qwen35, 16 -> Qwen4 (11/LFM2 refuses) ──
        // The Some/None half duplicates caps().supports_continuous_batch; the
        // variant picks between distinct staging bodies in batch_staging.
        for id in 0u32..=22 {
            let want = match id {
                5 | 6 => Some(ContinuousBatchRoute::Qwen35),
                16 => Some(ContinuousBatchRoute::Qwen4),
                _ => None,
            };
            assert_eq!(
                continuous_batch_route(id),
                want,
                "continuous_batch_route({id})"
            );
            // The capability half must stay consistent with the declared cap.
            let declared = super::carrier_for(id)
                .map(|c| c.caps().supports_continuous_batch)
                .unwrap_or(false);
            assert_eq!(
                want.is_some(),
                declared,
                "continuous_batch_route({id}).is_some() disagrees with carrier caps"
            );
        }

        // ── bench_decode_route: 9, 11, 5|6, 13, 14, 16, 19; everything else Unsupported ──
        for id in 0u32..=22 {
            let want = match id {
                9 => BenchDecodeRoute::Deepseek4,
                11 => BenchDecodeRoute::Lfm2Moe,
                5 | 6 => BenchDecodeRoute::Qwen35,
                13 => BenchDecodeRoute::Gemma4,
                14 => BenchDecodeRoute::MuseGlimmer,
                16 => BenchDecodeRoute::Qwen4,
                19 => BenchDecodeRoute::Raon,
                _ => BenchDecodeRoute::Unsupported,
            };
            assert_eq!(bench_decode_route(id), want, "bench_decode_route({id})");
        }

        // ── vision_route: 8 -> DotsOcr, 5|6 -> QwenVl, 11 -> Lfm2Vl; gated by supports_images ──
        for id in 0u32..=14 {
            let want = match id {
                8 => VisionRoute::DotsOcr,
                5 | 6 => VisionRoute::QwenVl,
                11 => VisionRoute::Lfm2Vl,
                _ => VisionRoute::None,
            };
            assert_eq!(vision_route(id), want, "vision_route({id})");
        }

        // ── img_route: exactly 40 -> Flux, 45 -> Flux2; every text/vision
        // arch (and the toy sentinel) stays None, pinning the fail-closed
        // gate in both directions (text generate refuses 40/45,
        // img_generate refuses everything else). ──
        for id in (0u32..=14).chain([20, 40, 45, 22, 0xFF]) {
            let want = match id {
                40 => ImgRoute::Flux,
                45 => ImgRoute::Flux2,
                _ => ImgRoute::None,
            };
            assert_eq!(img_route(id), want, "img_route({id})");
        }

        // ── ep_prompt_route: 9 -> Dsml, everything else Jinja ──
        for id in 0u32..=14 {
            let want = if id == 9 {
                EpPromptRoute::Dsml
            } else {
                EpPromptRoute::Jinja
            };
            assert_eq!(ep_prompt_route(id), want, "ep_prompt_route({id})");
        }

        // ── ep_eos_route: Qwen, MiniMax and DeepSeek carry distinct EOS ──
        for id in 0u32..=14 {
            let want = match id {
                5 | 6 => EpEosRoute::Qwen35,
                10 => EpEosRoute::Minimax,
                _ => EpEosRoute::Deepseek4,
            };
            assert_eq!(ep_eos_route(id), want, "ep_eos_route({id})");
        }

        // ── generation_early_route: 13 -> Gemma4, 14 -> MuseGlimmer; arch 22
        // (Gemma4 EAGLE drafter sidecar) must stay None — it is never a
        // primary generate target.
        for id in (0u32..=14).chain([22]) {
            let want = match id {
                13 => Some(GenerationEarlyRoute::Gemma4),
                14 => Some(GenerationEarlyRoute::MuseGlimmer),
                _ => None,
            };
            assert_eq!(
                generation_early_route(id),
                want,
                "generation_early_route({id})"
            );
        }
    }

    #[test]
    fn dflash_lm_head_always_admits_qt3_6_13() {
        use super::dflash_lm_head_quant_supported;
        for qt in [3u8, 6, 13] {
            for arch in [
                "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx942",
            ] {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "qt={qt} must always be admitted on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_qt17_on_wmma_set_only() {
        use super::dflash_lm_head_quant_supported;
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(
                dflash_lm_head_quant_supported(Some(17u8), arch),
                "qt=17 must be admitted on WMMA arch {arch}"
            );
        }
        for arch in ["gfx1030", "gfx942", "gfx1010"] {
            assert!(
                !dflash_lm_head_quant_supported(Some(17), arch),
                "qt=17 must be rejected on non-WMMA arch {arch}"
            );
        }
    }

    #[test]
    fn dflash_lm_head_v2_admitted_on_gfx11_gfx12_wmma() {
        use super::dflash_lm_head_quant_supported;
        let v2 = [44u8, 47, 48, 49, 50];
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
            for qt in v2 {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be admitted on {arch}"
                );
            }
        }
        // Full WMMA set beyond the focused four.
        for arch in ["gfx1101", "gfx1102", "gfx1150"] {
            for qt in v2 {
                assert!(
                    dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be admitted on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_v2_rejected_on_gfx1030_and_gfx942() {
        use super::dflash_lm_head_quant_supported;
        for arch in ["gfx1030", "gfx942"] {
            for qt in [44u8, 47, 48, 49, 50] {
                assert!(
                    !dflash_lm_head_quant_supported(Some(qt), arch),
                    "V2 qt={qt} must be rejected on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_rejects_unknown_qt_and_missing() {
        use super::dflash_lm_head_quant_supported;
        // qt45 MQ4CG256, legacy MQ2 qt18, F16, missing tensor — all rejected.
        for qt in [None, Some(18u8), Some(45), Some(16), Some(1)] {
            for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
                assert!(
                    !dflash_lm_head_quant_supported(qt, arch),
                    "qt={qt:?} must be rejected on {arch}"
                );
            }
        }
    }

    #[test]
    fn dflash_lm_head_wmma_arch_set() {
        use super::is_dflash_lm_head_wmma_arch;
        for arch in [
            "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            assert!(is_dflash_lm_head_wmma_arch(arch), "{arch}");
        }
        for arch in ["gfx1030", "gfx942", "gfx1010", "gfx908"] {
            assert!(!is_dflash_lm_head_wmma_arch(arch), "{arch}");
        }
    }

    #[test]
    fn qwen_embedded_template_wins_over_froggeric() {
        // Qwen3.8-style embedded template carries official reasoning_effort.
        let embedded = "{% if reasoning_effort %}{{ reasoning_effort }}{% endif %}".to_string();
        let (got, source) =
            super::qwen35_template_from_embedded(Some(embedded.clone()), "qwen3.8-27b.mq4");
        assert_eq!(got, embedded);
        assert!(got.contains("reasoning_effort"));
        assert_ne!(got.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
        assert_eq!(source, super::SRC_EMBEDDED_HFQ);
    }

    #[test]
    fn qwen_missing_template_uses_froggeric_fallback() {
        // Legacy Qwen3.5/3.6 HFQs without tokenizer_config.chat_template.
        // Ornith path without embedded also keeps plain froggeric (no adapt).
        let (got, source) = super::qwen35_template_from_embedded(None, "ornith-1.5-35b-a3b.mq4");
        assert_eq!(got.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
        assert!(source.starts_with("built-in fallback: froggeric"));
        let (got2, _) = super::qwen35_template_from_embedded(None, "qwen3.6-35b.mq4");
        assert_eq!(got2.as_str(), super::FROGGERIC_QWEN35_TEMPLATE);
    }

    /// Official Qwen3.8 low / xhigh instruction literals (exact contract).
    const ORNITH_LOW_INSTR: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    const ORNITH_XHIGH_INSTR: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

    /// Minimal Qwen3.5/3.6-shaped skeleton with the exact Ornith/official
    /// tools + no-tools system markers the load-time adapter rewrites, plus
    /// enough surrounding Jinja to actually render tools / system / user-only
    /// branches through `JinjaChatFrame::render_messages`.
    fn ornith_parent_template_skeleton() -> String {
        // This is an independent fixture copied from the embedded Ornith parent
        // template shape. Do not construct it from the production marker
        // constants: the fixture must catch whitespace drift in those markers.
        concat!(
            "{%- macro render_content(content, do_vision_count, is_system_content=false) %}{{- content }}{%- endmacro %}\n",
            "{%- if tools and tools is iterable and tools is not mapping %}\n",
            "    {{- '<|im_start|>system\\n' }}\n",
            "    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}\n",
            "    {%- for tool in tools %}\n",
            "        {{- \"\\n\" }}\n",
            "        {{- tool | tojson }}\n",
            "    {%- endfor %}\n",
            "    {{- \"\\n</tools>\" }}\n",
            "    {%- if messages[0].role == 'system' %}\n",
            "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
            "        {%- if content %}\n",
            "            {{- '\\n\\n' + content }}\n",
            "        {%- endif %}\n",
            "    {%- endif %}\n",
            "    {{- '<|im_end|>\\n' }}\n",
            "{%- else %}\n",
            "    {%- if messages[0].role == 'system' %}\n",
            "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n",
            "        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endif %}\n",
            "{%- for message in messages %}\n",
            "    {%- if message.role != 'system' %}\n",
            "        {%- set content = render_content(message.content, false)|trim %}\n",
            "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n",
            "    {%- endif %}\n",
            "{%- endfor %}",
        )
        .to_string()
    }

    /// Hermetic GPT-2-BPE tokenizer sufficient for Jinja string rendering
    /// (bos override is empty; encode path is unused by these tests).
    fn test_tokenizer() -> hipfire_runtime::tokenizer::Tokenizer {
        // GPT-2 mode trigger (`Ġ`) + full byte fallback so any short
        // ASCII content round-trips if a future test encodes.
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""system": 2"#.to_string());
        entries.push(r#""user": 3"#.to_string());
        entries.push(r#""assistant": 4"#.to_string());
        entries.push(r#""\n": 5"#.to_string());
        entries.push(r#""Ġ": 6"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{escaped}": {}"#, 100 + b));
        }
        let vocab = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}}
                ]
            }}"#
        );
        hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n = 0u32;
        for b2 in 0u32..=255u32 {
            if !bs.contains(&b2) {
                bs.push(b2);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (i, &bv) in bs.iter().enumerate() {
            if bv == b as u32 {
                return char::from_u32(cs[i]).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    fn test_msg(
        role: hipfire_runtime::prompt_frame::Role,
        content: &str,
    ) -> hipfire_runtime::prompt_frame::Message {
        hipfire_runtime::prompt_frame::Message {
            role,
            content: content.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }
    }

    fn adapted_ornith_template() -> String {
        super::qwen35_template_from_embedded(
            Some(ornith_parent_template_skeleton()),
            "ornith-1.5-35b-a3b.mq4",
        )
        .0
    }

    #[test]
    fn ornith_adapter_provenance_reports_applied_vs_plain_embedded() {
        let (adapted, source) = super::qwen35_template_from_embedded(
            Some(ornith_parent_template_skeleton()),
            "ornith-1.5-35b-a3b.mq4",
        );
        assert!(adapted.contains("reasoning_effort"));
        assert!(source.starts_with(super::SRC_EMBEDDED_HFQ));
        assert!(source.contains("adapter applied"));
        // Marker drift: the adapter is skipped, the embedded text is kept.
        let drifted = "{{ messages }}".to_string();
        let (kept, source) =
            super::qwen35_template_from_embedded(Some(drifted.clone()), "ornith-1.5-35b-a3b.mq4");
        assert_eq!(kept, drifted);
        assert_eq!(source, super::SRC_EMBEDDED_HFQ);
    }

    fn render_ornith(
        template: &str,
        messages: &[hipfire_runtime::prompt_frame::Message],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> String {
        let tok = test_tokenizer();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer: &tok,
            template,
            system: None,
            user: "",
            enable_thinking,
            bos_token: Some(""),
            reasoning_strength: None,
            reasoning_effort,
        };
        frame
            .render_messages(messages, tools, None)
            .unwrap_or_else(|e| panic!("render failed (effort={reasoning_effort:?}): {e}"))
    }

    /// Extract the first `<|im_start|>system ... <|im_end|>` body (no markers).
    fn first_system_body(rendered: &str) -> Option<&str> {
        const START: &str = "<|im_start|>system\n";
        const END: &str = "<|im_end|>";
        let i = rendered.find(START)?;
        let rest = &rendered[i + START.len()..];
        let j = rest.find(END)?;
        Some(&rest[..j])
    }

    #[test]
    fn ornith15_canonical_and_legacy_basenames_adapt() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "/models/ornith-1.5-35b-a3b.mq4",
            "/models/ornith-1.5-35b-a3b.mq4r",
            "/models/ORNITH-1.5-35B-A3B.MQ4",
            "/models/ORNITH-1.5-35B-A3B.MQ4R",
            "/models/ornith1.5-35b-a3b.mq4",
            "ornith1.5-35b-a3b.mq4r",
        ] {
            let (got, source) = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert!(
                source.contains("adapter applied"),
                "path {path} must report the adapter in its source: {source}"
            );
            assert!(
                got.contains("reasoning_effort|default('xhigh')"),
                "path {path} must adapt: missing default"
            );
            assert!(
                got.contains("('xhigh', 'medium', 'low')"),
                "path {path} must accept exact rungs"
            );
            assert!(
                got.contains(ORNITH_LOW_INSTR),
                "path {path} missing low instruction"
            );
            assert!(
                got.contains(ORNITH_XHIGH_INSTR),
                "path {path} missing xhigh instruction"
            );
            // medium injects nothing: no medium instruction string.
            assert!(
                !got.contains("Reasoning effort is set to medium"),
                "path {path} must not steer medium"
            );
            // tools prepend + system/no-system branches.
            assert!(
                got.contains("reasoning_instructions + '\\n\\n'"),
                "path {path} missing tools/system prepend"
            );
            assert!(
                got.contains("{%- elif reasoning_instructions %}"),
                "path {path} missing no-system branch"
            );
            assert_ne!(got, skeleton, "path {path} must change the template");
        }
    }

    #[test]
    fn unrelated_qwen_basename_is_untouched() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "qwen3.6-35b-a3b.mq4",
            "qwen3.5-397b.mq4",
            "ornith-1.0-35b.mq4",
            "ornith-2.0-35b-a3b.mq4",
            "not-ornith-1.5-35b.mq4",
        ] {
            let (got, source) = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert_eq!(got, skeleton, "path {path} must not adapt");
            assert_eq!(source, super::SRC_EMBEDDED_HFQ, "path {path}");
            assert!(!got.contains("reasoning_effort"), "path {path}");
        }
    }

    #[test]
    fn ornith15_wrong_extension_is_untouched() {
        let skeleton = ornith_parent_template_skeleton();
        for path in [
            "ornith-1.5-35b-a3b.mq6",
            "ornith-1.5-35b-a3b.hfq",
            "ornith-1.5-35b-a3b",
            "ornith-1.5-35b-a3b.json",
            "ornith-1.5-35b-a3b.mq4.sidecar",
            "ornith1.5-35b-a3b.MQ6",
            "/models/ORNITH-1.5-35B-A3B.hfq",
            "ornith-1.5-35b-a3b.mq4r.bak",
        ] {
            let (got, source) = super::qwen35_template_from_embedded(Some(skeleton.clone()), path);
            assert_eq!(got, skeleton, "path {path} must not adapt (extension gate)");
            assert_eq!(source, super::SRC_EMBEDDED_HFQ, "path {path}");
            assert!(
                !got.contains("reasoning_effort"),
                "path {path} must not gain effort"
            );
        }
    }

    #[test]
    fn already_native_ornith_template_stays_unchanged() {
        let native = format!(
            "{}{}",
            super::QWEN38_REASONING_BLOCK,
            ornith_parent_template_skeleton()
        );
        assert!(native.contains("reasoning_effort"));
        let (got, source) =
            super::qwen35_template_from_embedded(Some(native.clone()), "ornith-1.5-35b-a3b.mq4");
        assert_eq!(got, native, "already-native must not be rewritten");
        assert_eq!(source, super::SRC_EMBEDDED_HFQ);
    }

    #[test]
    fn marker_drift_returns_original_unadapted() {
        // Missing the no-tools branch marker → adapter must no-op.
        let drifted = format!(
            "{}\n{{%- set ns = namespace(x=1) %}}",
            super::ORNITH_TOOLS_SYSTEM_OPEN
        );
        let (got, source) =
            super::qwen35_template_from_embedded(Some(drifted.clone()), "ornith-1.5-35b-a3b.mq4");
        assert_eq!(source, super::SRC_EMBEDDED_HFQ);
        assert_eq!(
            got, drifted,
            "drift must keep original so probe stays honest"
        );
        assert!(!got.contains("reasoning_effort|default('xhigh')"));

        // Duplicate tools gate without the compound tools-open → no-op.
        let dup = format!(
            "{gate}\n{gate}\n{no_tools}",
            gate = super::ORNITH_TOOLS_GATE,
            no_tools = super::ORNITH_NO_TOOLS_SYSTEM_BRANCH
        );
        let (got2, source2) =
            super::qwen35_template_from_embedded(Some(dup.clone()), "ornith1.5-35b-a3b.mq4r");
        assert_eq!(got2, dup);
        assert_eq!(source2, super::SRC_EMBEDDED_HFQ);
    }

    #[test]
    fn ornith15_adapted_jinja_renders_effort_contract() {
        use hipfire_runtime::prompt_frame::Role;

        let template = adapted_ornith_template();
        assert!(
            template.contains("reasoning_effort|default('xhigh')"),
            "adapter must inject effort block"
        );

        let tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo",
                "parameters": {"type": "object", "properties": {}}
            }
        });
        let tools = [tool];

        let user_only = [test_msg(Role::User, "hi")];
        let with_system = [test_msg(Role::System, "SYS"), test_msg(Role::User, "hi")];

        // Four effort states × three structural branches.
        let efforts: [(Option<&str>, Option<&str>); 4] = [
            (None, Some(ORNITH_XHIGH_INSTR)),          // undefined → xhigh
            (Some("low"), Some(ORNITH_LOW_INSTR)),     // explicit low
            (Some("medium"), None),                    // medium: no instruction
            (Some("xhigh"), Some(ORNITH_XHIGH_INSTR)), // explicit xhigh
        ];

        for (effort, want_instr) in efforts {
            // 1) tools present
            let r = render_ornith(&template, &with_system, Some(&tools), true, effort);
            let sys = first_system_body(&r).expect("tools path emits system");
            assert!(
                sys.contains("# Tools"),
                "tools branch missing header (effort={effort:?}): {sys:?}"
            );
            match want_instr {
                Some(instr) => {
                    assert!(
                        sys.starts_with(instr),
                        "tools: instruction must lead system turn (effort={effort:?}): {sys:?}"
                    );
                    assert!(
                        sys.contains(&format!("{instr}\n\n# Tools")),
                        "tools: instruction must sit before # Tools (effort={effort:?})"
                    );
                    // system content still appended after tools block
                    assert!(sys.contains("SYS"), "tools: system content lost");
                }
                None => {
                    assert!(
                        !sys.contains("Reasoning effort is set to"),
                        "medium must not inject instruction (tools): {sys:?}"
                    );
                    assert!(
                        sys.starts_with("# Tools"),
                        "medium tools system starts at # Tools: {sys:?}"
                    );
                }
            }

            // 2) no tools + existing system message
            let r = render_ornith(&template, &with_system, None, true, effort);
            let sys = first_system_body(&r).expect("system path emits system");
            match want_instr {
                Some(instr) => {
                    assert_eq!(
                        sys,
                        format!("{instr}\n\nSYS"),
                        "no-tools+system placement (effort={effort:?})"
                    );
                }
                None => {
                    assert_eq!(sys, "SYS", "medium no-tools+system is bare content");
                }
            }
            assert!(
                r.contains("<|im_start|>user\nhi<|im_end|>"),
                "user turn must remain (effort={effort:?})"
            );

            // 3) no tools + user-only (no system message)
            let r = render_ornith(&template, &user_only, None, true, effort);
            match want_instr {
                Some(instr) => {
                    let sys = first_system_body(&r).expect("user-only effort injects system");
                    assert_eq!(
                        sys, instr,
                        "no-tools user-only system body (effort={effort:?})"
                    );
                }
                None => {
                    assert!(
                        first_system_body(&r).is_none(),
                        "medium user-only must not invent a system turn: {r:?}"
                    );
                }
            }
            assert!(
                r.contains("<|im_start|>user\nhi<|im_end|>"),
                "user turn must remain user-only (effort={effort:?})"
            );
        }

        // Undefined effort matches explicit xhigh on every branch.
        for tools_arg in [Some(tools.as_slice()), None] {
            for msgs in [with_system.as_slice(), user_only.as_slice()] {
                // Skip tools+user_only if tools need messages[0] system check —
                // still valid: tools path tolerates non-system first message.
                let undef = render_ornith(&template, msgs, tools_arg, true, None);
                let xhigh = render_ornith(&template, msgs, tools_arg, true, Some("xhigh"));
                assert_eq!(
                    undef,
                    xhigh,
                    "undefined must match xhigh (tools={})",
                    tools_arg.is_some()
                );
            }
        }

        // Thinking disabled: effort instructions absent on every branch.
        for effort in [None, Some("low"), Some("medium"), Some("xhigh")] {
            for (msgs, tools_arg) in [
                (with_system.as_slice(), Some(tools.as_slice())),
                (with_system.as_slice(), None),
                (user_only.as_slice(), None),
            ] {
                let r = render_ornith(&template, msgs, tools_arg, false, effort);
                assert!(
                    !r.contains("Reasoning effort is set to"),
                    "thinking-off must drop effort instr (effort={effort:?}): {r:?}"
                );
            }
        }
    }

    #[test]
    fn ornith15_adapted_template_probe_effort_capability() {
        let template = adapted_ornith_template();
        let tok = test_tokenizer();
        let cap = hipfire_runtime::prompt_frame::probe_effort_capability(&tok, &template);
        assert!(cap.native, "adapted Ornith template must be effort-native");
        assert_eq!(
            cap.supported,
            vec!["low", "medium", "xhigh"],
            "supported rungs must be exactly the Qwen3.8 contract"
        );
    }
}

#[cfg(all(test, feature = "arch-qwen35"))]
mod mtp_sidecar_probe_tests {
    use std::path::Path;

    #[test]
    fn loader_probe_includes_stem_sidecar_for_mq4v2_hfq() {
        // finish_qwen35_load uses find_mtp_sidecar; pin the product spelling
        // `qwen3.5-4b.mq4v2.hfq` → `qwen3.5-4b.mtp` so a last-extension-only
        // probe cannot silently return.
        let c = hipfire_arch_qwen35::mtp_head::mtp_sidecar_candidates(Path::new(
            "/models/qwen3.5-4b.mq4v2.hfq",
        ));
        assert!(c.iter().any(|p| p.ends_with("qwen3.5-4b.mtp")), "{c:?}");
    }
}
