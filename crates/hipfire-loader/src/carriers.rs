// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use crate::spec_build::Qwen35SlotGuard;
use crate::Carrier;
use crate::{
    finish_qwen35_load, resolve_chat_template, resolve_chat_template_overrides, LoadedModel,
    SRC_DIR_JINJA, SRC_DIR_TOKENIZER_CONFIG,
};
use hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::HfqModelSource;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::model_source::ModelSource as _;
use hipfire_runtime::spec::{InPlaceGuard, SpecEmit, SpecEmitCtx, SpecTargetGuard};
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use std::any::Any;

// The ChatML/Hermes per-token emitter (`Qwen35Emit`) is shared by every
// ChatML-family spec arm — qwen35 DFlash AND the llama/qwen2 n-gram paths all
// drive it (they already share qwen35's tool-call grammar). It physically lives
// in the qwen35 crate; the llama/qwen2 carriers wiring it here is composition-
// root glue, not an arch→arch dependency (those arch crates never name it). A
// future cleanup could hoist the emitter + grammar into the runtime.
use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;

// ─── Source-only metadata (tokenizer / chat_template / arch_id) ───────
//
// The single seam for the source-varying-but-arch-invariant axis. Adding a
// future source kind (e.g. GGUF) is one new `match` arm here plus the
// irreducible per-arch `(config, weights)` block in each carrier. Lives in
// `hipfire-loader` (not `loader_api`) because it calls `resolve_chat_template`,
// which reads the loader's built-in arch templates.
//
// NOTE: `arch_id` extraction is purely source-varying (`hfq.arch_id` vs
// `source.arch_id()`), so it belongs here — but the *values* live in two
// distinct namespaces (HFQ header ids vs `derive_arch_id` dir ids). A GGUF
// plug-in author must pick the correct namespace, not assume a single one.
struct SourceMeta {
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<String>,
    /// Provenance description of `chat_template` (see `LoadedModel::chat_template_source`).
    chat_template_source: Option<String>,
    arch_id: u32,
}

/// Discover a `.vl` sidecar file for the given trunk model path.
///
/// Same resolution the daemon uses (they are one implementation now, in
/// [`hipfire_runtime::sidecar`]): `HIPFIRE_VL_FILE` env, then the
/// `<stem>.vl` sibling candidates (`<stem>` also strips `.hfq` and a quant
/// suffix, so `model.mq4v2.hfq` finds `model.vl`).
fn discover_vl_path(model_path: &str) -> Option<std::path::PathBuf> {
    hipfire_runtime::sidecar::resolve_vl_sidecar(model_path)
}

/// `MemAvailable - CMA` the automatic prewarm of the PLE hot-row pack leaves at
/// ready (`memory.ple_hot_store_bytes = auto`: budget = `MemAvailable - CMA` - this,
/// capped at the pack). The pack's pages are clean page cache, counted in
/// `MemAvailable` and dropped (never swapped) under pressure; the reserve
/// covers what a request allocates without first reclaiming prewarmed pages.
const PLE_HOT_PREWARM_RESERVE_BYTES: u64 = 4 << 30;

/// Open the PLE hot-row store before the weight sweep: the `<model>.plehot`
/// rank sidecar and the `<rows source>.plehotrows` hot-row pack cut from it
/// (`hipfire-quantize ple-hot-pack`).  The rows source is the model file
/// (monolithic artifact) or the PLE rows sidecar (`.ple-q4` / `.ple-q8`): the
/// pack holds those rows verbatim, is bound to that file's length, and its
/// worker verifies it against that file during the load; the row store reads
/// the pack's rows from the pack once verified. The caller attaches the result
/// to the bundle after assembly and sets the prewarm budget once the model is
/// ready; dropping it stops and joins the worker, so every load error path
/// quiesces it.
///
/// `None` = off: the kill switch (`0`, the default), a non-unified-memory
/// GPU, an HFQ overlay, PLE rows that are neither Q8F16 nor Q4LM, or a missing
/// or unusable sidecar or pack. `auto` is silent when it does not apply; an
/// explicit budget says why not.
fn start_ple_hot_store(
    uma: bool,
    hfq: &hipfire_runtime::hfq::HfqFile,
    metadata: &hipfire_arch_qwen4::ple::PleHashMetadata,
    descriptors: &[hipfire_runtime::model_source::SourceRangeDescriptor],
    rows_path: &std::path::Path,
    rows_file_bytes: u64,
    model_path: &str,
) -> Option<hipfire_arch_qwen4::bundle::PleHotWorker> {
    use hipfire_config::memory::PleHotStoreBudget;
    let started = std::time::Instant::now();
    let explicit = match hipfire_config::memory::ple_hot_store_bytes() {
        PleHotStoreBudget::Bytes(0) => return None,
        PleHotStoreBudget::Bytes(bytes) => Some(bytes),
        PleHotStoreBudget::Auto => None,
    };
    let skip = |why: &str| {
        if explicit.is_some() {
            eprintln!("  qwen4 ple hot store: off ({why})");
        }
    };
    if !uma {
        skip("needs a unified-memory GPU");
        return None;
    }
    if hfq.has_overlay() {
        skip("an HFQ overlay is attached; shard offsets would not match the model file");
        return None;
    }
    let plan = match hipfire_arch_qwen4::bundle::ple_hot_plan(
        metadata,
        descriptors,
        rows_path,
        rows_file_bytes,
    ) {
        Ok(Some(plan)) => plan,
        Ok(None) => {
            skip("PLE rows are neither Q8F16 nor Q4LM");
            return None;
        }
        Err(error) => {
            eprintln!("  qwen4 ple hot store: off (PLE plan: {error})");
            return None;
        }
    };
    let sidecar = hipfire_config::memory::ple_hot_rank_file()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{model_path}.plehot")));
    let pack = hipfire_config::memory::ple_hot_pack_file()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(format!("{}.plehotrows", plan.source_path.display()))
        });
    if !sidecar.is_file() {
        skip(&format!("no sidecar at {}", sidecar.display()));
        return None;
    }
    if !pack.is_file() {
        skip(&format!("no hot-row pack at {}", pack.display()));
        return None;
    }
    let rank = match hipfire_runtime::ple_hot_store::PleHotRank::open(
        &sidecar,
        plan.padded_rows,
        &plan.fingerprint_hex,
    ) {
        Ok(rank) => rank,
        Err(error) => {
            eprintln!(
                "  qwen4 ple hot store: off (sidecar {}: {error})",
                sidecar.display()
            );
            return None;
        }
    };
    let store = match hipfire_runtime::ple_hot_store::HotRowStore::open(
        rank,
        &pack,
        plan.rows_per_shard,
        plan.row_encoding,
        plan.encoded_row_bytes,
        &plan.shard_offsets,
        plan.source_file_bytes,
    ) {
        Ok(store) => store,
        Err(error) => {
            eprintln!("  qwen4 ple hot store: off ({error})");
            return None;
        }
    };
    eprintln!(
        "  qwen4 ple hot store: sidecar {}, pack {}, setup {} ms",
        sidecar.display(),
        pack.display(),
        started.elapsed().as_millis()
    );
    #[cfg(unix)]
    {
        let handle = store.spawn_worker(&plan.source_path, plan.shard_offsets);
        Some(hipfire_arch_qwen4::bundle::PleHotWorker::new(store, handle))
    }
    #[cfg(not(unix))]
    {
        // Unreachable: the non-Unix `HotRowStore::open` above always returns
        // Err, and that match arm returns None before this point.
        unreachable!("PLE hot-row packs are unsupported on Windows; run on Linux")
    }
}

/// Bytes of the PLE hot-row pack to prewarm now that the model is ready: an
/// explicit `memory.ple_hot_store_bytes`, or (`auto`) `MemAvailable - CMA` minus
/// [`PLE_HOT_PREWARM_RESERVE_BYTES`]; the store caps it at the pack's size.
fn ple_hot_prewarm_budget() -> (u64, Option<u64>) {
    let available = hipfire_runtime::ple_hot_store::mem_available_bytes();
    let budget = match hipfire_config::memory::ple_hot_store_bytes() {
        hipfire_config::memory::PleHotStoreBudget::Bytes(bytes) => bytes,
        hipfire_config::memory::PleHotStoreBudget::Auto => available.map_or(0, |available| {
            available.saturating_sub(PLE_HOT_PREWARM_RESERVE_BYTES)
        }),
    };
    (budget, available)
}

/// Why [`ple_prewarm_budget`] came out zero, for the load log.
fn ple_prewarm_off_reason(available: Option<u64>, explicit_zero: bool) -> String {
    if explicit_zero {
        "memory.ple_prewarm_bytes = 0".to_string()
    } else {
        format!(
            "MemAvailable - CMA {} MiB leaves no headroom over the {} MiB reserve",
            available.map_or(0, |bytes| bytes / (1 << 20)),
            PLE_PREWARM_RESERVE_BYTES / (1 << 20)
        )
    }
}

/// Default (`auto`) q4 PLE table prewarm: the hottest 15 GiB of rows (about
/// as many rows as 28 GB of Q8), the knee of the warm-rows value curve.
const PLE_PREWARM_AUTO_BYTES: u64 = 15 << 30;

/// Host RAM an `auto` PLE table prewarm leaves free. Larger than the hot-row
/// store's reserve: on Halo (MemAvailable ~24 GiB at ready) prewarming the
/// whole 26.8 GiB q4 table pushed reclaim into the request, and Ciru 8K fell
/// to 1,734 tok/s from 1,828 at 15 GiB.
const PLE_PREWARM_RESERVE_BYTES: u64 = 8 << 30;

/// Bytes of the q4 PLE table to prewarm now that the model is ready: an
/// explicit `memory.ple_prewarm_bytes`, or (`auto`) [`PLE_PREWARM_AUTO_BYTES`]
/// capped at `MemAvailable - CMA` minus [`PLE_PREWARM_RESERVE_BYTES`]; the
/// prewarm caps it at the table's data.  The flag is whether the budget was
/// set to zero.
fn ple_prewarm_budget() -> (u64, Option<u64>, bool) {
    let available = hipfire_runtime::ple_hot_store::mem_available_bytes();
    let (budget, explicit_zero) =
        ple_prewarm_budget_for(hipfire_config::memory::ple_prewarm_bytes(), available);
    (budget, available, explicit_zero)
}

/// [`ple_prewarm_budget`] for a given request and `MemAvailable - CMA`.
fn ple_prewarm_budget_for(
    request: hipfire_config::memory::PleHotStoreBudget,
    available: Option<u64>,
) -> (u64, bool) {
    match request {
        hipfire_config::memory::PleHotStoreBudget::Bytes(bytes) => (bytes, bytes == 0),
        hipfire_config::memory::PleHotStoreBudget::Auto => (
            available.map_or(0, |available| {
                available
                    .saturating_sub(PLE_PREWARM_RESERVE_BYTES)
                    .min(PLE_PREWARM_AUTO_BYTES)
            }),
            false,
        ),
    }
}

/// `auto` in-load fill of a rank-ordered q4 PLE table: the hottest 14 GiB.
/// That is B* of the offline row-hit curve of the shipped rank (2026-10-11,
/// `plehot/hcurve_q4.py`): the smallest prefix whose modelled held-out prose
/// and agentic 8K prefill (TTFT = 4,177 + (1 − h) × 1,135 ms) is within 1% of
/// the 15 GiB prewarm's (prose h 0.923 vs 0.937, agentic 0.809 vs 0.836).
#[cfg_attr(not(unix), allow(dead_code))]
const PLE_LOAD_FILL_AUTO_BYTES: u64 = 14 << 30;

/// Read rate cap of the in-load fill: a small share of the NVMe's sequential
/// bandwidth, so the weight upload keeps the drive.
#[cfg_attr(not(unix), allow(dead_code))]
const PLE_LOAD_FILL_BYTES_PER_SEC: u64 = 200_000_000;

/// Bytes of a rank-ordered q4 PLE table to read while the model loads: the
/// `requested` `memory.ple_load_fill_bytes` (`auto` = [`PLE_LOAD_FILL_AUTO_BYTES`])
/// capped at an explicit `memory.ple_prewarm_bytes`, at the table, and at the
/// host RAM available now (`MemAvailable - CMA` plus TTM's page pool, which the
/// weights' GTT allocations take first) minus the weights still to load into
/// host RAM minus [`PLE_PREWARM_RESERVE_BYTES`] (0 when unknown).
#[cfg_attr(not(unix), allow(dead_code))]
fn ple_load_fill_bytes(
    requested: hipfire_config::memory::PleHotStoreBudget,
    prewarm: hipfire_config::memory::PleHotStoreBudget,
    host_available: Option<u64>,
    host_weight_bytes: u64,
    table_bytes: u64,
) -> u64 {
    use hipfire_config::memory::PleHotStoreBudget;
    let requested = match requested {
        PleHotStoreBudget::Bytes(bytes) => bytes,
        PleHotStoreBudget::Auto => PLE_LOAD_FILL_AUTO_BYTES,
    };
    let prewarm = match prewarm {
        PleHotStoreBudget::Bytes(bytes) => bytes,
        PleHotStoreBudget::Auto => u64::MAX,
    };
    let host = host_available.map_or(0, |available| {
        available
            .saturating_sub(host_weight_bytes)
            .saturating_sub(PLE_PREWARM_RESERVE_BYTES)
    });
    requested.min(prewarm).min(host).min(table_bytes)
}

/// Host RAM the PLE page cache keeps resident at serve time, from one budget
/// snapshot: the prewarm `budget` or the in-load fill's `load_fill` target,
/// whichever is larger, capped at the `data_bytes` of the table/pack/sidecar
/// that holds the rows. On unified memory these pages count in
/// `MemAvailable` (and so in the device free memory) while reclaimable, so a
/// device admission must reserve them to keep the rows resident.
fn ple_resident_bytes(budget: u64, load_fill: u64, data_bytes: u64) -> u64 {
    budget.max(load_fill).min(data_bytes)
}

/// Start the worker of a rank-ordered q4 PLE table before the weight sweep,
/// with an in-load fill of its hottest prefix when
/// [`ple_load_fill_bytes`] allows one. The worker prewarms on from there
/// once the model is ready; dropping it stops and joins it. Also returns the
/// fill's byte target (the table prefix the load keeps resident).
#[cfg(unix)]
fn start_ple_table_worker(
    store: &std::sync::Arc<hipfire_runtime::ple_hot_store::HotRowStore>,
    host_weight_bytes: u64,
) -> (hipfire_arch_qwen4::bundle::PleHotWorker, u64) {
    const MIB: u64 = 1 << 20;
    let host_available = rdna_compute::uma_memory::host_available_bytes();
    let bytes = ple_load_fill_bytes(
        hipfire_config::memory::ple_load_fill_bytes(),
        hipfire_config::memory::ple_prewarm_bytes(),
        host_available,
        host_weight_bytes,
        store.data_bytes(),
    );
    eprintln!(
        "  qwen4 ple load fill: {} MiB of a {} MiB q4 table while loading, at most {} MB/s \
         (host available {} MiB incl. TTM pool, weights to host RAM {} MiB)",
        bytes / MIB,
        store.data_bytes() / MIB,
        PLE_LOAD_FILL_BYTES_PER_SEC / 1_000_000,
        host_available.map_or(0, |bytes| bytes / MIB),
        host_weight_bytes / MIB,
    );
    let fill = (bytes > 0).then_some(hipfire_runtime::ple_hot_store::LoadFill {
        bytes,
        bytes_per_sec: PLE_LOAD_FILL_BYTES_PER_SEC,
    });
    (
        hipfire_arch_qwen4::bundle::PleHotWorker::new(
            store.clone(),
            store.spawn_table_worker(fill),
        ),
        bytes,
    )
}

fn resolve_source_meta(src: &ModelSource, path: &str) -> Result<SourceMeta, String> {
    match src {
        ModelSource::Hfq(hfq) => {
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let (chat_template, chat_template_source) = resolve_chat_template(hfq, path);
            Ok(SourceMeta {
                tokenizer,
                chat_template,
                chat_template_source,
                arch_id: hfq.arch_id,
            })
        }
        ModelSource::Dir(source) => {
            let arch_id = source.arch_id();
            let (chat_template, chat_template_source) =
                resolve_dir_chat_template(source, path, arch_id);
            Ok(SourceMeta {
                tokenizer: tokenizer_from_dir(source)?,
                chat_template,
                chat_template_source,
                arch_id,
            })
        }
    }
}

/// Safetensors-directory template selection with provenance: user overrides,
/// then the directory's own `chat_template.jinja` / `tokenizer_config.json`,
/// then the arch's built-in default. `path` is the model directory.
fn resolve_dir_chat_template(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
    path: &str,
    arch_id: u32,
) -> (Option<String>, Option<String>) {
    if let Some((template, description)) = resolve_chat_template_overrides(path) {
        return (Some(template), Some(description));
    }
    if let Some(template) = source.chat_template() {
        return (Some(template), Some(dir_template_description(path)));
    }
    match arch_default_template(arch_id) {
        Some(template) => (
            Some(template),
            Some(format!(
                "built-in fallback: arch {arch_id} default template (model dir has no chat template)"
            )),
        ),
        None => (None, None),
    }
}

/// Mirrors `SafetensorsSource::chat_template`'s file order: a non-blank
/// `chat_template.jinja` wins, else `tokenizer_config.json`.
fn dir_template_description(dir: &str) -> String {
    let jinja = std::path::Path::new(dir).join("chat_template.jinja");
    if std::fs::read_to_string(&jinja).is_ok_and(|s| !s.trim().is_empty()) {
        format!("{SRC_DIR_JINJA} {}", jinja.display())
    } else {
        let config = std::path::Path::new(dir).join("tokenizer_config.json");
        format!("{SRC_DIR_TOKENIZER_CONFIG} {}", config.display())
    }
}

/// Folds the "no tokenizer.json / failed to parse" block duplicated verbatim
/// in every Dir arm today.
fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("could not parse the tokenizer at {} (the file may be corrupt; re-download the model): {e}", tok_path.display()))?
            .ok_or_else(|| format!("the tokenizer file {} has no usable vocabulary; re-download the model", tok_path.display()))
    } else {
        Err("no tokenizer.json found in the model directory; add the model's tokenizer.json or re-download the model".into())
    }
}

/// Returns the first candidate string that tokenizes to exactly one token, or 1.
fn resolve_eos_tok(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, candidates: &[&str]) -> u32 {
    for s in candidates {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    1
}

/// Dir-source diagnostic: arch_id + quant_method. One-line call at the top of
/// every Dir-capable carrier's load(). Qwen35 prints a richer variant inline.
fn dir_diag(src: &ModelSource) {
    if let ModelSource::Dir(s) = src {
        let qm = s
            .quant_config()
            .map(|q| q.method.as_str())
            .unwrap_or("none");
        eprintln!("  safetensors arch_id={}, quant_method={qm}", s.arch_id());
    }
}

// ─── Qwen2Carrier ────────────────────────────────────────────────────

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str {
        "qwen2"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen2::Qwen2Bundle>()
        }) {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("qwen2: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // HFQ id 7 and qwen2 safetensors dirs (derive_arch_id → 7). Both route
        // here so the qwen2 Q/K/V `attention_bias=true` biases load (the
        // llama-family Dir loader drops them).
        arch_id == 7
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen2_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for &tok in synthetic {
            if hipfire_arch_qwen2::qwen2::forward_step(gpu, weights, config, state, tok).is_err() {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "qwen2",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). Qwen2
        // (arch_id=7, e.g. VibeThinker) impls `SpecTarget`, so it can be driven by
        // the arch-generic spec loop with no draft model. `None` ⇒ AR-only.
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── Qwen4Carrier ────────────────────────────────────────────────────

/// Keep host memory out of reclaim before the HIP runtime loads, in a process
/// about to load the HFQ model at `model`, when that is a Qwen4 model on
/// discrete GPUs (`expert_residency::keeps_host_memory_out_of_reclaim`). The
/// daemon calls it once device visibility is installed and before its first
/// HIP call: the runtime reads the switches once, when it loads. A file that
/// is not an HFQ container, or cards the KFD topology cannot name, keep
/// ROCm's defaults.
pub fn prepare_host_memory_for(model: &std::path::Path) {
    let Ok(arch_id) = hipfire_runtime::hfq::HfqFile::probe_arch_id(model) else {
        return;
    };
    let Some(devices) = hipfire_config::devices::startup_devices() else {
        return;
    };
    let archs = devices
        .iter()
        .map(|device| device.arch.as_str())
        .collect::<Vec<_>>();
    if hipfire_arch_qwen4::expert_residency::keeps_host_memory_out_of_reclaim(arch_id, &archs) {
        hip_bridge::keep_host_memory_out_of_reclaim(&format!("Qwen4 model on {}", archs.join(",")));
    }
}

/// Executable local-path Qwen4 carrier.  Distribution/product admission stays
/// outside this registry; this route only makes an already admitted HFQM
/// artifact loadable.
fn qwen4_use_range_payload(
    is_uma: bool,
    residency: hipfire_runtime::weight_manifest::WeightResidency,
) -> bool {
    is_uma || residency.is_external()
}

const QWEN4_DDTREE_DEFAULT_BUDGET: usize = 0;
const QWEN4_DDTREE_DEFAULT_TOPK: usize = 4;

/// Return whether a Qwen4 load carries an active or non-default DDTree
/// request. The CLI resolves schema defaults before serializing load params,
/// so an ordinary AR load arrives as `Some(0)`/`Some(4)` rather than `None`.
/// A non-default top-K remains unsupported even when the budget is zero.
pub(crate) const fn qwen4_ddtree_requested(spec: SpecLoadCfg) -> bool {
    match (spec.ddtree_budget, spec.ddtree_topk) {
        (Some(budget), _) if budget != QWEN4_DDTREE_DEFAULT_BUDGET => true,
        (_, Some(topk)) if topk != QWEN4_DDTREE_DEFAULT_TOPK => true,
        _ => false,
    }
}

pub struct Qwen4Carrier;

impl Carrier for Qwen4Carrier {
    fn name(&self) -> &'static str {
        "qwen4"
    }

    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen4::bundle::Qwen4Bundle>()
        }) {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("qwen4: spec target state mismatch".into()),
        }
    }

    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Same close the Qwen4 AR producer splices on a spent think budget.
        Ok(Qwen35Emit::from_ctx_template_think_close(ctx))
    }

    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool {
        arch_id == hipfire_arch_qwen4::ARCH_ID && !is_dir
    }

    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if is_dir {
            return Err(
                "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
            );
        }
        if pp == 0 || pp > QWEN4_PP_MAX {
            return Err(format!(
                "qwen4: topology.pp = {pp} is outside the admitted 1..={QWEN4_PP_MAX}"
            ));
        }
        Ok(())
    }
    fn admit_options(
        &self,
        draft_path: Option<&str>,
        options: crate::admission::SourceAdmissionOptions,
    ) -> Result<(), String> {
        let spec = options.spec;
        let unsupported = [
            (draft_path.is_some(), "draft model (draft_path)"),
            (options.eagle_drafter, "EAGLE drafter"),
            (options.kv_adaptive, "adaptive KV"),
            (
                spec.dflash.is_some_and(|enabled| enabled),
                "DFlash (speculation.dflash)",
            ),
            (
                spec.dspark.is_some_and(|enabled| enabled),
                "DSpark (speculation.dspark)",
            ),
            (
                spec.ngram_draft.is_some_and(|enabled| enabled),
                "n-gram draft (speculation.ngram_draft)",
            ),
            (qwen4_ddtree_requested(spec), "DDTree"),
            (options.cask, "CASK (memory.cask)"),
            (options.non_single_compute, "non-single compute placement"),
            (options.expert_count_override, "expert-count override"),
            (
                options.pflash,
                "PFlash (prefill_compression; set it to off)",
            ),
        ]
        .into_iter()
        .filter_map(|(set, name)| set.then_some(name))
        .collect::<Vec<_>>();
        if !unsupported.is_empty() {
            return Err(format!(
                "qwen4: unsupported option(s): {}; turn them off to load Flash-Next",
                unsupported.join(", ")
            ));
        }
        Ok(())
    }

    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            // The only qwen4 batch route is the exact fn-lanes route, staged
            // under `serve.vmm_batch` (`batch_staging::stage_qwen4_lanes`);
            // there is no fixed-lane qwen4 batch state.
            supports_continuous_batch: true,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: true,
            spec_excludes_adaptive: true,
            semantic_contract_version: Some(2),
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::QwenJinja,
        }
    }

    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }

    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        self.admit_topology(
            hipfire_arch_qwen4::ARCH_ID,
            src.is_dir(),
            ctx.pp,
            ctx.kv_backend,
        )?;
        // Pipeline-parallel (topology.pp 2..=8): the mesh-executor route. The
        // Single and expert-parallel paths below are untouched.
        if ctx.pp > 1 {
            return load_qwen4_pp_model(src, ctx);
        }
        // Admission resolved the QSA context storage (automatic: VMM where
        // `qwen4_vmm_refusal` is clear); the device must still agree.
        let backend = hipfire_arch_qwen4::Qwen4KvBackend::from(ctx.kv_backend);
        if backend == hipfire_arch_qwen4::Qwen4KvBackend::Vmm {
            if let Some(reason) = hipfire_arch_qwen4::qwen4_vmm_refusal(
                &ctx.gpu.arch,
                ctx.gpu.vmm_recommended_granularity().is_ok(),
            ) {
                return Err(format!(
                    "qwen4: KV backend 'vmm' is not available: {reason}; use --kv-backend legacy (memory.kv_backend = legacy)"
                ));
            }
        }
        let vmm_granularity = match backend {
            hipfire_arch_qwen4::Qwen4KvBackend::Vmm => ctx
                .gpu
                .vmm_recommended_granularity()
                .map_err(|error| format!("qwen4: VMM granularity: {error}"))?,
            hipfire_arch_qwen4::Qwen4KvBackend::Legacy => 1,
        };
        if ctx.max_seq == 0 || ctx.max_seq > hipfire_arch_qwen4::QWEN4_MAX_CONTEXT {
            return Err(format!(
                "qwen4: max_seq {} is outside the admitted context range 1..={}",
                ctx.max_seq,
                hipfire_arch_qwen4::QWEN4_MAX_CONTEXT
            ));
        }
        let ep = hipfire_config::topology::qwen4_ep()?;
        let native_mtp =
            crate::admission::qwen4_native_mtp(ctx.spec, &ctx.gpu.arch, ctx.path, ctx.pp, 1, ep)?;
        let max_k = qwen4_native_mtp_k(ctx.spec, &ctx.gpu.arch);
        let options = crate::admission::SourceAdmissionOptions {
            spec: ctx.spec,
            kv_adaptive: crate::admission::qwen4_kv_adaptive_requested(ctx.kv_adaptive_override),
            eagle_drafter: ctx.gemma4_drafter_path.is_some(),
            cask: ctx.cask.sidecar.is_some(),
            state_quant: ctx.state_quant_override.is_some(),
            non_single_compute: !matches!(
                &ctx.deepseek4_compute_placement,
                hipfire_config::Deepseek4ComputePlacement::Single
            ),
            expert_count_override: ctx.deepseek4_experts_per_token.is_some(),
            pflash: false,
        };
        qwen4_ep_option_refusal(ep, ctx.draft_path, &options)?;
        self.admit_options(ctx.draft_path, options)?;
        let meta = resolve_source_meta(&src, ctx.path)?;
        let shared = Qwen4RankShared {
            path: ctx.path.to_string(),
            max_seq: ctx.max_seq,
            sequence: ctx.sequence,
            backend,
            vmm_granularity,
            kv_mode: kv_mode_from_ctx(ctx),
            state_quant: ctx.state_quant_override.map(str::to_string),
            spec: ctx.spec,
            native_mtp,
            max_k,
        };
        // Expert parallelism: devices and peer access are verified with real
        // devices, and the load's seal planned, before rank 0 allocates anything.
        let mut ep_plan = plan_qwen4_ep(ep, ctx.pp, ctx.gpu, &shared, &src)?;
        // Every rank applies the same expert VRAM split: the minimum over
        // ranks of what each rank's free VRAM keeps, measured before rank 0
        // allocates.
        if let Some(plan) = ep_plan.as_mut() {
            let devices = plan.seal.physical_devices();
            let free = qwen4_ep_free_vram(ctx.gpu, devices)?;
            plan.expert_vram_layers =
                qwen4_ep_vram_layers(ctx.gpu, &shared, plan.seal.ranks(), &free, &src)?;
            eprintln!(
                "qwen4 EP plan: {} ranks on devices {:?} (mesh epoch {}, {} sealed expert groups), free VRAM per rank {:?} MiB, expert VRAM split {} MoE layers on every rank",
                plan.seal.ranks(),
                devices,
                plan.seal.mesh().epoch().as_u64(),
                plan.seal.plans().len(),
                free.iter().map(|bytes| bytes >> 20).collect::<Vec<_>>(),
                plan.expert_vram_layers
            );
        }
        let mut loaded = load_qwen4_rank(
            ctx.gpu,
            ep_plan.as_ref().map(|plan| plan.rank(0)),
            &shared,
            ep_plan.as_ref().map(|plan| plan.expert_vram_layers),
            Some(Qwen4SessionSeed {
                tokenizer: &meta.tokenizer,
                chat_template: meta.chat_template.as_deref(),
            }),
            src,
        )?;
        // Seam: the EP mesh owner builds peer ranks 1..N-1 here, after rank 0
        // is loaded (see `attach_qwen4_ep_peers`).
        if let Some(plan) = &ep_plan {
            if let Err(error) = attach_qwen4_ep_peers(&mut loaded.bundle, ctx.gpu, plan) {
                let _ = loaded.bundle.free_gpu(ctx.gpu);
                return Err(error);
            }
        }
        if let (Some(sequence), Some(card_cap)) = (ctx.sequence.as_mut(), loaded.card_cap) {
            sequence.card_cap = card_cap;
        }
        let (native_mtp, max_k, max_seq) = (loaded.native_mtp, loaded.max_k, loaded.max_seq);
        let cpu_splice = loaded.bundle.has_cpu_deferred_experts();
        let max_k = if cpu_splice {
            max_k.min(hipfire_arch_qwen4::program::QWEN4_CPU_SPLICE_MAX_ROWS - 1)
        } else {
            max_k
        };
        // Host-placed experts on the CPU splice verify at most the 4-row CPU
        // window; otherwise the per-GPU proven row-exact verify rows apply.
        let verify_rows = if cpu_splice {
            hipfire_arch_qwen4::program::QWEN4_CPU_SPLICE_MAX_ROWS
        } else {
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&ctx.gpu.arch)
        };
        let speculator = native_mtp.then(|| {
            qwen4_native_mtp_speculator(
                max_k,
                max_seq,
                &ctx.gpu.arch,
                meta.tokenizer.special_token_id("<|im_end|>"),
                meta.tokenizer.special_token_id("</think>"),
                true,
                verify_rows,
            )
        });

        let mut model = LoadedModel {
            state: Some(Box::new(loaded.bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                max_seq,
                max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        };
        apply_qwen4_native_mtp(&mut model, native_mtp, max_k, ctx.spec);
        Ok(model)
    }

    /// Bench-prefill for Qwen4: the daemon has reset the request state; run
    /// the synthetic prompt through the tiled final-row prefill the AR path
    /// uses (the same call `bench_decode_prime` makes).
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.qwen4_mut()?;
        let vocab = bundle.config.vocab_size;
        let logits = match gpu.zeros(&[vocab], rdna_compute::DType::F32) {
            Ok(logits) => logits,
            Err(error) => {
                *prefill_err = Some(format!("qwen4 bench prefill logits: {error:?}"));
                return Some(false);
            }
        };
        let result = bundle.forward_chunk_final(gpu, synthetic, &logits, None);
        let _ = gpu.free_tensor(logits);
        if let Err(error) = result {
            *prefill_err = Some(format!("qwen4 bench prefill: {error}"));
            return Some(false);
        }
        Some(true)
    }

    /// Bench-decode prime for Qwen4: reset the request state, then run the
    /// synthetic context through the final-row prefill the AR path uses.
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let bundle = m.qwen4_mut()?;
        let vocab = bundle.config.vocab_size;
        if let Err(error) = bundle.reset(gpu) {
            return Some(Some(format!("qwen4 bench prime reset: {error}")));
        }
        let logits = match gpu.zeros(&[vocab], rdna_compute::DType::F32) {
            Ok(logits) => logits,
            Err(error) => return Some(Some(format!("qwen4 bench prime logits: {error:?}"))),
        };
        let result = bundle
            .forward_chunk_final(gpu, synthetic, &logits, None)
            .err()
            .map(|error| error.to_string());
        let _ = gpu.free_tensor(logits);
        Some(result)
    }

    /// Bench-decode run for Qwen4: one single-token forward per iteration, the
    /// same call the AR producer makes.
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        decode_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.qwen4_mut()?;
        let vocab = bundle.config.vocab_size;
        let logits = gpu.zeros(&[vocab], rdna_compute::DType::F32).ok()?;
        let mut ok = true;
        for i in 0..iterations {
            let token = 101 + (i as u32 % 1000);
            if let Err(error) = bundle.forward_token(gpu, token, &logits, None) {
                *decode_err = Some(format!("iter {i} pos {}: {error}", context + i));
                ok = false;
                break;
            }
        }
        let _ = gpu.free_tensor(logits);
        Some(ok)
    }
}

/// Expert parallelism refuses what the replicated single-device program does
/// not run: DFlash, DSpark, n-gram drafting, DDTree, drafters and the other
/// single-device-only load options. Native MTP (`--spec mtp`) is admitted: the
/// drafter is replicated on every rank. Names EP and the escape hatches.
/// `ep <= 1` never refuses (the single-device caps are unchanged).
pub(crate) fn qwen4_ep_option_refusal(
    ep: usize,
    draft_path: Option<&str>,
    options: &crate::admission::SourceAdmissionOptions,
) -> Result<(), String> {
    if ep <= 1 {
        return Ok(());
    }
    let spec = options.spec;
    let unsupported = [
        (draft_path.is_some() || spec.dflash == Some(true), "DFlash"),
        (spec.dspark == Some(true), "DSpark"),
        (spec.ngram_draft == Some(true), "n-gram drafting"),
        (qwen4_ddtree_requested(spec), "DDTree"),
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
        Some((_, what)) => Err(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) does not support {what}; \
             drop the option or set topology.ep = 1 (HIPFIRE_QWEN4_EP=0)"
        )),
        None => Ok(()),
    }
}

/// The native MTP draft window of a Qwen4 load: `speculation.mtp_k`, else the
/// process config, else the architecture default, clamped to `1..=10`. The one
/// resolution of the Single and the pipeline-parallel load.
fn qwen4_native_mtp_k(spec: SpecLoadCfg, gpu_arch: &str) -> usize {
    spec.mtp_k
        .or(hipfire_runtime::config::get().mtp_k)
        .unwrap_or_else(|| {
            hipfire_config::mtp_k_default_for_arch(hipfire_arch_qwen4::ARCH_ID, gpu_arch)
        })
        .clamp(1, 10)
}

/// The generic speculator around the native Qwen4 MTP head, with the load log
/// lines. `end_of_turn` is the tokenizer's `<|im_end|>` id and `think_close`
/// its `</think>` id. `ngram_modifier` false (a pipeline-parallel load) builds
/// it without an n-gram pool whatever `speculation.mtp_ngram` resolves to,
/// so no request can arm a takeover.
/// `verify_rows` is the verify window the n-gram copy lane's take cap reads:
/// the per-GPU row-exact rows, or the 4-row CPU splice window when host-placed
/// routed experts run on the CPU.
fn qwen4_native_mtp_speculator(
    max_k: usize,
    max_seq: usize,
    gpu_arch: &str,
    end_of_turn: Option<u32>,
    think_close: Option<u32>,
    ngram_modifier: bool,
    verify_rows: usize,
) -> Box<dyn hipfire_runtime::spec::Speculator> {
    eprintln!("  qwen4 native MTP speculator enabled (K={max_k})");
    let arch = hipfire_arch_qwen4::ARCH_ID;
    let mut copy_lane = false;
    let ngram = if !ngram_modifier {
        eprintln!("  qwen4 native MTP n-gram: off (topology.pp)");
        None
    } else {
        copy_lane = hipfire_config::mtp_ngram_copy_enabled_for_arch(arch, gpu_arch);
        match hipfire_arch_qwen4::mtp_spec::qwen4_ngram_mod_config(
            hipfire_config::ngram_mod_triple_for_arch(arch, gpu_arch),
        ) {
            Ok(config) => {
                let ngram_on = hipfire_config::mtp_ngram_enabled_for_arch(arch, gpu_arch);
                eprintln!(
                    "  qwen4 native MTP n-gram: {} (n_match={} n_min={} n_max={})",
                    if ngram_on { "on" } else { "off" },
                    config.n_match,
                    config.n_min,
                    config.n_max
                );
                if ngram_on && copy_lane {
                    eprintln!(
                        "  qwen4 native MTP n-gram copy lane: on (take_cap={})",
                        hipfire_arch_qwen4::mtp_spec::qwen4_ngram_take_cap(&config, verify_rows,)
                    );
                }
                Some(config)
            }
            Err(error) => {
                eprintln!("  qwen4 native MTP n-gram: unavailable ({error})");
                None
            }
        }
    };
    hipfire_arch_qwen4::mtp_spec::build_qwen4_mtp_speculator(
        max_k,
        max_seq,
        end_of_turn,
        think_close,
        ngram,
        copy_lane,
    )
}

/// The load's native MTP bookkeeping on its `LoadedModel`.
fn apply_qwen4_native_mtp(
    model: &mut LoadedModel,
    native_mtp: bool,
    max_k: usize,
    spec: SpecLoadCfg,
) {
    model.mtp_weights_present = native_mtp;
    if native_mtp {
        model.mtp_k = max_k;
        model.mtp_mode = if spec.mtp == Some(true) {
            "on".to_string()
        } else {
            "auto".to_string()
        };
    }
}

/// Whether this rank attaches the native MTP head: every Single / expert
/// parallel rank when it is on, a pipeline band only when it is the output
/// band (the head and its experts live there alone).
fn qwen4_rank_native_mtp(
    native_mtp: bool,
    band: Option<&hipfire_arch_qwen4::weights::Qwen4BandScope>,
) -> bool {
    native_mtp && band.map_or(true, |scope| scope.output)
}

/// The session cache budget one rank attaches: a pipeline band with native
/// MTP attaches none (the head's session snapshot is not run on a mesh, so
/// MTP sessions stay refused); every other rank keeps `configured`.
fn qwen4_pp_session_budget(pp_native_mtp: bool, configured: u64) -> u64 {
    if pp_native_mtp {
        0
    } else {
        configured
    }
}

// ─── Qwen4 rank loader (single device and one rank of expert parallelism) ───

/// What every rank of one Qwen4 load shares: the admitted request, owned and
/// `Send` so a mesh owner can hand a clone to each peer worker thread. The
/// fields are the load context the single-device carrier resolved before it
/// touched the device.
#[derive(Clone)]
pub struct Qwen4RankShared {
    pub path: String,
    pub max_seq: usize,
    pub sequence: Option<hipfire_runtime::loader_api::SequenceHint>,
    pub backend: hipfire_arch_qwen4::Qwen4KvBackend,
    pub vmm_granularity: usize,
    /// Resolved whole-cache KV mode name (`memory.kv_cache`).
    pub kv_mode: String,
    pub state_quant: Option<String>,
    pub spec: SpecLoadCfg,
    /// Native MTP after admission (EP: only on an explicit `--spec mtp`; every
    /// rank attaches its own head).
    pub native_mtp: bool,
    pub max_k: usize,
}

/// The source model's tokenizer and chat template, from which a session
/// cache's cache domain is derived. Single and expert-parallel loads hand one
/// only to the rank that serves requests' session cache (peers pass `None`
/// and attach none; EP peers align to rank 0's cache afterwards). A
/// `topology.pp` load hands every band its own copy of the same seed, so each
/// band attaches its own cache (no disk tier) and all bands share the model,
/// tokenizer, template and PLE identity.
#[derive(Clone, Copy)]
pub struct Qwen4SessionSeed<'a> {
    pub tokenizer: &'a hipfire_runtime::tokenizer::Tokenizer,
    pub chat_template: Option<&'a str>,
}

/// The disk tier a session cache attaches: `Some((dir, bytes))` for a Single
/// or EP rank load, never for a `topology.pp` band (`band`; PP admission also
/// refuses `memory.session_cache_disk_bytes`).
fn qwen4_session_disk_tier(
    band: bool,
    dir: std::path::PathBuf,
    bytes: u64,
) -> Option<(std::path::PathBuf, u64)> {
    (!band).then_some((dir, bytes))
}

/// One loaded rank.
pub struct Qwen4RankLoad {
    pub bundle: hipfire_arch_qwen4::bundle::Qwen4Bundle,
    /// Native MTP after the profile check and host-mapped policy (an EP rank
    /// attaches its own head when it is on).
    pub native_mtp: bool,
    pub max_k: usize,
    /// The admitted context (unified memory resolves it after load).
    pub max_seq: usize,
    /// Card-capacity cap of the memory admission, when a sequence was resolved.
    pub card_cap: Option<usize>,
}

/// The resolved expert-parallel mesh request: what the mesh owner needs after
/// rank 0 is loaded. `seal` is the load's one EP mesh epoch, every rank's
/// physical device and the sealed expert plans every rank's placement and
/// load identity derive from (rank 0 is the daemon's own device); `shared` is
/// the load context to give every peer's [`load_qwen4_rank`].
#[derive(Clone)]
pub struct Qwen4EpPlan {
    pub seal: std::sync::Arc<hipfire_arch_qwen4::ep_seal::Qwen4EpSeal>,
    pub shared: Qwen4RankShared,
    /// The expert VRAM split (MoE layers resident, of trunk + MTP) every rank
    /// applies: the minimum over ranks ([`qwen4_ep_vram_layers`]). Pass it as
    /// `Some(plan.expert_vram_layers)` to every [`load_qwen4_rank`].
    pub expert_vram_layers: usize,
}

impl Qwen4EpPlan {
    /// Rank `rank`'s binding to the seal, for [`load_qwen4_rank`].
    pub fn rank(&self, rank: usize) -> hipfire_arch_qwen4::ep_seal::Qwen4EpRankSeal {
        hipfire_arch_qwen4::ep_seal::Qwen4EpRankSeal {
            seal: self.seal.clone(),
            rank,
        }
    }
}

/// `rank->device` list for the plan-time emulation log line, e.g.
/// `0->0, 1->1, 2->0`.
fn qwen4_ep_emulation_map(devices: &[i32]) -> String {
    devices
        .iter()
        .enumerate()
        .map(|(rank, device)| format!("{rank}->{device}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Resolve the expert-parallel plan for `ep`: `None` for `ep <= 1`; otherwise
/// the geometry and (with real devices) architecture and peer-access checks of
/// [`crate::admission::qwen4_ep_preflight`], then the load's seal planned from
/// `src`'s tensor index, before rank 0 allocates anything.
fn plan_qwen4_ep(
    ep: usize,
    pp: usize,
    gpu: &rdna_compute::Gpu,
    shared: &Qwen4RankShared,
    src: &ModelSource,
) -> Result<Option<Qwen4EpPlan>, String> {
    if ep <= 1 {
        return Ok(None);
    }
    let emulated = hipfire_runtime::config::get().emulate_gpus.is_some();
    if let Some(refusal) =
        crate::admission::qwen4_ep_geometry_refusal(ep, pp, 1, &gpu.arch, None, emulated)
    {
        return Err(refusal);
    }
    if gpu.device_id != 0 {
        return Err(format!(
            "qwen4: expert parallelism (topology.ep = {ep}) ranks are devices 0..{ep} of the \
             process's device list; the daemon's device is {}, not 0",
            gpu.device_id
        ));
    }
    let devices = crate::admission::qwen4_ep_preflight(&gpu.hip, ep)?;
    if emulated {
        eprintln!(
            "qwen4 EP logical emulation (HIPFIRE_EMULATE_GPUS): rank -> physical device {}",
            qwen4_ep_emulation_map(&devices)
        );
    }
    let ModelSource::Hfq(hfq) = src else {
        return Err(
            "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
        );
    };
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
        .map_err(|error| error.to_string())?;
    let identity = hfq
        .source_identity()
        .map_err(|error| format!("qwen4: EP source identity: {error}"))?;
    let seal = hipfire_arch_qwen4::ep_seal::Qwen4EpSeal::seal(
        &receipt.manifest,
        receipt.config.num_hidden_layers,
        ep,
        &devices,
        &identity,
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    Ok(Some(Qwen4EpPlan {
        seal: std::sync::Arc::new(seal),
        shared: shared.clone(),
        expert_vram_layers: 0,
    }))
}

/// Open a source for a peer rank: the HFQM file at `shared.path` with its PLE
/// rows source resolved, as admission does for rank 0. Peers call this on
/// their own thread and pass the result to [`load_qwen4_rank`].
pub fn open_qwen4_rank_source(
    shared: &Qwen4RankShared,
    gpu_arch: &str,
) -> Result<ModelSource, String> {
    let mut source = ModelSource::from_path(&shared.path)?;
    crate::admission::resolve_qwen4_ple_rows(&mut source, &shared.path, gpu_arch)?;
    Ok(source)
}

/// **Seam for the EP mesh owner.** Called by [`Qwen4Carrier`]'s load after rank
/// 0 is fully loaded (and before the speculator / `LoadedModel` are built), only
/// when `topology.ep` > 1. The owner starts one persistent worker thread per
/// peer rank `1..ranks`, constructs that rank's `Gpu` on the seal's physical
/// device for it, calls [`open_qwen4_rank_source`] and [`load_qwen4_rank`]
/// there with `Some(plan.rank(rank))`, `&plan.shared` and no session seed,
/// builds the row-gather mesh over all ranks' bundles and attaches it to
/// `bundle` (rank 0). On `Err` the carrier frees rank 0's GPU state; the owner
/// must free any peer it already loaded.
///
/// Peers load on their own worker threads, one rank at a time in ascending
/// order (the host-RAM check of pinned experts assumes it), each at the agreed
/// expert split `plan.expert_vram_layers`; the mesh then builds the row-gather
/// transport, enables P2P and installs every rank's EP tables
/// (`Qwen4Bundle::attach_ep_mesh`).
pub fn attach_qwen4_ep_peers(
    bundle: &mut hipfire_arch_qwen4::bundle::Qwen4Bundle,
    gpu: &mut rdna_compute::Gpu,
    plan: &Qwen4EpPlan,
) -> Result<(), String> {
    let plan = plan.clone();
    let loader: std::sync::Arc<hipfire_arch_qwen4::ep::PeerLoadFn> =
        std::sync::Arc::new(move |rank| {
            let device = *plan
                .seal
                .physical_devices()
                .get(rank)
                .ok_or_else(|| format!("rank {rank} is outside the seal"))?;
            let mut peer_gpu = rdna_compute::Gpu::init_with_device(device)
                .map_err(|error| format!("init HIP device {device}: {error:?}"))?;
            let loaded = open_qwen4_rank_source(&plan.shared, &peer_gpu.arch).and_then(|source| {
                load_qwen4_rank(
                    &mut peer_gpu,
                    Some(plan.rank(rank)),
                    &plan.shared,
                    Some(plan.expert_vram_layers),
                    None,
                    source,
                )
            });
            match loaded {
                Ok(load) => Ok((peer_gpu, load.bundle)),
                Err(error) => Err(format!("load on device {device}: {error}")),
            }
        });
    bundle
        .attach_ep_mesh(gpu, loader)
        .map_err(|error| format!("qwen4: expert parallelism mesh setup failed: {error}"))
}

/// What the expert placement is sized from, for one rank of a `ranks`-way
/// load: the replicated non-expert bytes, one MoE layer's routed-expert bytes
/// on this rank, the `auto` reserve ([`residency::AutoReserve`]: the larger
/// of the historical measured formula and a bottom-up census of the first
/// full-context request, itemised; the census alone where it carries the
/// measured HIP runtime overhead, [`residency::runtime_overhead_bytes`]) and
/// its native MTP and QSA gather parts,
/// returned too. The one place this is computed, for the load and for
/// [`qwen4_ep_vram_layers`].
#[allow(clippy::too_many_arguments)]
fn qwen4_placement_inputs(
    gpu: &rdna_compute::Gpu,
    config: &hipfire_arch_qwen4::Qwen4Config,
    weights: &[hipfire_runtime::weight_manifest::WeightEntry],
    bytes_of: &dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    backend: hipfire_arch_qwen4::Qwen4KvBackend,
    use_ranges: bool,
    max_seq: usize,
    vmm_granularity: usize,
    spec: SpecLoadCfg,
    max_k: usize,
    native_mtp: bool,
    routed_experts: bool,
    state_format: hipfire_arch_qwen4::Qwen4StateFormat,
    scratch_mode: hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode,
    ranks: usize,
) -> Result<
    (
        u64,
        u64,
        hipfire_arch_qwen4::expert_residency::AutoReserve,
        Option<u64>,
        Option<u64>,
    ),
    String,
> {
    use hipfire_arch_qwen4::expert_residency as residency;
    let qsa_format = state_format.qsa;
    let (non_expert, layer_experts) = residency::resident_split_ranks(weights, bytes_of, ranks)
        .map_err(|error| format!("qwen4: {error}"))?;
    let chunk_rows =
        hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(&gpu.arch, max_seq);
    let context = hipfire_arch_qwen4::Qwen4ContextCommit::for_expert_reserve(
        backend,
        use_ranges,
        max_seq,
        chunk_rows,
        vmm_granularity,
    );
    // The context the census charges: the ACTUAL backend with the whole
    // admitted context mapped (a VMM arena maps ahead to `max_seq`, rounded
    // to the driver granularity), not the legacy-forced layout the historical
    // formula charges on a discrete card.
    let census_context =
        hipfire_arch_qwen4::Qwen4ContextCommit::new(backend, max_seq, max_seq, vmm_granularity);
    // Every placement host-maps at least the MTP layer's routed experts, so
    // the head is still attached after it only where the host-mapped policy
    // keeps it (exact gfx1201, or an explicit `--spec mtp`); only then does
    // it need VRAM.
    let mtp_kept = native_mtp
        && crate::admission::qwen4_mtp_with_host_mapped_experts(
            spec,
            &gpu.arch,
            usize::from(routed_experts),
        );
    let mtp_for =
        |context: &hipfire_arch_qwen4::Qwen4ContextCommit| -> Result<Option<u64>, String> {
            if !mtp_kept {
                return Ok(None);
            }
            let head = residency::language_head_dtype(weights)
                .ok_or("qwen4: manifest has no language head")?;
            let drafts = hipfire_arch_qwen4::mtp_spec::native_mtp_proposal_drafts(max_k, &gpu.arch);
            let row_capture = hipfire_arch_qwen4::mtp_spec::native_mtp_capture(
                gpu,
                config,
                state_format.gdn,
                drafts,
            );
            Ok(Some(
                hipfire_arch_qwen4::mtp_spec::native_mtp_device_bytes(
                    config,
                    context,
                    chunk_rows,
                    drafts,
                    head,
                    row_capture,
                    hipfire_arch_qwen4::mtp_spec::native_mtp_fill_scratch(gpu),
                )
                .ok_or("qwen4: native MTP device bytes overflow")?,
            ))
        };
    // The gathered QSA prefill attention's workspace, committed as the
    // context is (none when the route is off: the reserve is then unchanged).
    // Bytes already committed by an earlier load in this process are out of
    // `free` and only growth past them is charged.
    let gather_for =
        |context: &hipfire_arch_qwen4::Qwen4ContextCommit| -> Result<Option<u64>, String> {
            rdna_compute::tensor_ops::qsa_gather_workspace_needed(
                gpu,
                qsa_format,
                state_format.qsa_f16_arena,
            )
            .then(|| {
                rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                    config.num_key_value_heads,
                    context.tokens,
                )
                .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
                .ok_or("qwen4: QSA gather scratch size overflows".to_string())
            })
            .transpose()
        };
    let mtp_bytes = mtp_for(&context)?;
    let gather_bytes = gather_for(&context)?;
    // The measured HIP runtime overhead (exact gfx1201, discrete,
    // single-device): the census then carries it and the historical floor,
    // whose stale F32 context used to cover it, is retired.
    let runtime = residency::runtime_overhead_bytes(&gpu.arch, use_ranges, ranks);
    let historical = if runtime.is_some() {
        None
    } else {
        Some(
            residency::auto_vram_reserve(
                config,
                &context,
                chunk_rows,
                scratch_mode,
                state_format.layout(),
                mtp_bytes,
                gather_bytes,
            )
            .map_err(|error| format!("qwen4: {error}"))?,
        )
    };
    // The bottom-up census at the full admitted context.
    let state = residency::state_census(
        config,
        state_format,
        &census_context,
        chunk_rows,
        scratch_mode,
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    // Lazy F16 weight shadows of the BF16 owners the long-prefill F16 WMMA
    // arms read (exact gfx1201): the cold bound from the manifest, not the
    // warm cache.
    // The F16 WMMA arms run only at `route_rows >= 512`
    // (`gemm.rs` `qwen4_f16_wmma_applies`); no forward has more rows than the
    // chunk and gfx1201 has no canonical-row routing scope, so a smaller chunk
    // never creates a shadow.
    let f16_shadows = if gpu.qwen4_f16_wmma_gfx1201()
        && chunk_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
    {
        residency::f16_shadow_bytes(weights, config.num_hidden_layers)
            .map_err(|error| format!("qwen4: {error}"))?
    } else {
        0
    };
    // The kernels' pool workspaces at the largest requested chunk, exactly
    // (every short-tail remainder included). The dense GDN scan's scratch is
    // charged wherever `fn_gdn_dense::applies` can hold at load: exact gfx1201,
    // Q8 GDN state and the pinned 16-key/48-value-head, 128-dim, 10240-wide
    // geometry (its opt-out flag only lowers the charge, so it is ignored).
    let dense_gdn = gpu.arch == "gfx1201"
        && state_format.gdn == hipfire_arch_qwen4::GdnStateFormat::Q8
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
            == 10240;
    // The routed-expert I4 MoE terms exist only for a profile whose routed
    // experts run the grouped I4 route; a dense or synthetic profile
    // (no experts, or dimensions that route cannot take) charges none of
    // them and is never refused for them.
    let routed_i4 = routed_experts
        && config.num_experts > 0
        && config.num_experts_per_tok > 0
        && config.moe_intermediate_size > 0
        && config.hidden_size % 256 == 0
        && config.moe_intermediate_size % 128 == 0;
    let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
        hidden_size: config.hidden_size,
        moe_intermediate_size: config.moe_intermediate_size,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        dense_gdn,
        routed_i4,
        include_short_tails: true,
    };
    // Overflow (or a dimension an owning allocator rejects) refuses the load
    // with a clear error; there is no historical-only fallback.
    let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk_rows, &dims)
        .ok_or("qwen4: prefill workspace bytes cannot be computed (arithmetic overflow, or a hidden/MoE dimension the kernel allocators reject)")?;
    let census = residency::ReserveCensus {
        qsa_context: state.qsa_context,
        state: state.state,
        forward: state.forward,
        f16_shadows,
        workspace,
        // The split QSA selector's score/filter scratch (0 unless its route
        // can run).
        qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_scratch_bytes(
            gpu,
            state_format.layout().index_dtype(),
        ) as u64,
        mtp: mtp_for(&census_context)?.unwrap_or(0),
        gather: gather_for(&census_context)?.unwrap_or(0),
        runtime: runtime.unwrap_or(0),
    };
    let reserve = residency::AutoReserve::new(historical, census)
        .map_err(|error| format!("qwen4: {error}"))?;
    Ok((non_expert, layer_experts, reserve, mtp_bytes, gather_bytes))
}

/// Free VRAM of every device in `devices` (HIP logical ids), in order, as
/// [`rdna_compute::Gpu::device_mem_info_for`] reports it before any rank of
/// the load allocates. Run on the thread that owns `gpu` (the daemon thread):
/// other devices are queried by `device_mem_info_for`, which switches the
/// thread's current device with raw HIP setters and restores the previous
/// ACTUAL current device on every return path (`Gpu::bind_thread` cannot do
/// that: its per-thread cache still names `gpu`'s device, so it would skip the
/// call and leave every later rank-0 allocation on the last queried device).
///
/// Under `HIPFIRE_EMULATE_GPUS` several ranks alias one physical device
/// (`devices` repeats an id): that device's free VRAM is divided evenly among
/// the ranks on it ([`qwen4_ep_alias_free`]). Physical mode lists each device
/// once, so the division is by 1.
///
/// Under a `HIPFIRE_EMU` profile expert parallelism is refused (single-device
/// capacity budget); a one-device list is the budget owner's capped query.
pub fn qwen4_ep_free_vram(gpu: &rdna_compute::Gpu, devices: &[i32]) -> Result<Vec<u64>, String> {
    if devices.len() > 1 {
        gpu.emu_refuse_multi_device("qwen4 expert parallelism")
            .map_err(|error| format!("qwen4: {error}"))?;
    }
    let mut free = Vec::with_capacity(devices.len());
    for &device in devices {
        let (bytes, _) = gpu
            .device_mem_info_for(device)
            .map_err(|error| format!("qwen4: device {device} memory query: {error}"))?;
        free.push(bytes as u64);
    }
    Ok(qwen4_ep_alias_free(devices, &free))
}

/// Per-rank free VRAM from each rank's physical device free: divided by the
/// number of ranks aliased onto that device (`devices[r]`), so ranks sharing
/// a card split it instead of each claiming all of it.
pub fn qwen4_ep_alias_free(devices: &[i32], free: &[u64]) -> Vec<u64> {
    devices
        .iter()
        .zip(free)
        .map(|(device, &bytes)| {
            let aliased = devices
                .iter()
                .filter(|other| *other == device)
                .count()
                .max(1);
            bytes / aliased as u64
        })
        .collect()
}

/// The expert VRAM split every rank of an expert-parallel load applies, in
/// MoE layers resident (of trunk + MTP; the total means fully resident): the
/// minimum over ranks of what each rank's `free_per_rank[r]` keeps
/// ([`hipfire_arch_qwen4::expert_residency::ep_vram_layers`]), sized from the
/// source's own byte counts and the same reserve a load derives. Pass the
/// result as `expert_vram_layers` to **every** [`load_qwen4_rank`] (rank 0
/// included) so each layer is resident on all ranks or on none. `gpu` is any
/// rank's device (rank 0), `src` an opened source with its PLE rows resolved
/// ([`open_qwen4_rank_source`]); nothing is allocated.
pub fn qwen4_ep_vram_layers(
    gpu: &rdna_compute::Gpu,
    shared: &Qwen4RankShared,
    ranks: usize,
    free_per_rank: &[u64],
    src: &ModelSource,
) -> Result<usize, String> {
    let ModelSource::Hfq(hfq) = src else {
        return Err(
            "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
        );
    };
    // Logical emulation (`HIPFIRE_EMULATE_GPUS`) aliases every rank onto one
    // device, so a UMA APU (gfx1151) runs it with the per-alias VRAM split.
    if gpu.is_uma() && hipfire_runtime::config::get().emulate_gpus.is_none() {
        return Err("qwen4: expert parallelism needs discrete GPUs".into());
    }
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
        .map_err(|error| error.to_string())?;
    let recipe = receipt.recipe().clone();
    let config = receipt.config;
    let manifest = receipt
        .manifest
        .with_expert_parallel(hipfire_arch_qwen4::weights::Qwen4EpPlacement { rank: 0, ranks })
        .map_err(|error| format!("qwen4: {error}"))?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &shared.kv_mode,
        shared.state_quant.as_deref().unwrap_or(""),
        gpu,
        &config,
    )?;
    // The plan every rank's `load_qwen4_rank` takes: an expert-parallel load
    // never takes the compact plan (the owner-routed EP prefill refuses it),
    // so the reserve charges the `Full` plan.
    let scratch_mode = hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full;
    let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
        crate::qwen4_slab::qwen4_entry_bytes(&recipe, entry, |entry| {
            hfq.tensor_data(&entry.name)
                .map(|(_, bytes)| bytes.len() as u64)
        })
    };
    let (non_expert, layer_experts, reserve, ..) = qwen4_placement_inputs(
        gpu,
        &config,
        &manifest.weights,
        &bytes_of,
        shared.backend,
        false,
        shared.max_seq,
        shared.vmm_granularity,
        shared.spec,
        shared.max_k,
        shared.native_mtp,
        !config.ffn.is_dense(),
        state_format,
        scratch_mode,
        ranks,
    )?;
    // Same charges as each rank's load applies to an expert-parallel split
    // (no slab stage: `new_slab` has no EP route), so the agreed split never
    // exceeds what a rank's own check keeps.
    let reserve = hipfire_arch_qwen4::expert_residency::PlacementCharges::new(layer_experts, false)
        .and_then(|charges| charges.reserve(reserve.bytes()))
        .map_err(|error| format!("qwen4: {error}"))?;
    Ok(hipfire_arch_qwen4::expert_residency::ep_vram_layers(
        &config,
        free_per_rank,
        non_expert,
        layer_experts,
        reserve,
    ))
}

/// The per-rank view the rank body reads (field names match
/// [`LoadCtx`], from which the body was lifted).
struct RankCtx<'a> {
    gpu: &'a mut rdna_compute::Gpu,
    path: &'a str,
    max_seq: usize,
    sequence: Option<hipfire_runtime::loader_api::SequenceHint>,
    spec: SpecLoadCfg,
    state_quant_override: Option<&'a str>,
    kv_mode: &'a str,
}

/// Load one Qwen4 bundle onto `gpu`: manifest, expert placement and residency,
/// weight fulfillment, `Qwen4Bundle::assemble_with_ple_rows`, the PLE stores,
/// the forward program and the native MTP head.
///
/// `ep == None` is the single-device load, byte for byte what the carrier did
/// inline before. With `ep == Some(rank seal)` (`ranks` in 2..=8) this is one
/// rank of an expert-parallel load: the source must be the generation the
/// seal was planned from; routed experts (trunk and MTP head) are declared
/// `ExpertSharded { 512, Stride }` and uploaded compactly per the seal's
/// placement for the rank (this rank's experts `e % ranks == rank` ascending,
/// `local_count(512)` per stack, in VRAM or pinned host RAM per the residency
/// policy) under the seal's load identity (mesh epoch, rank, physical
/// device); everything else is replicated; the bundle is bound to the seal;
/// the rank attaches its own native MTP head when it is on; the rank prints
/// its fit line and an `qwen4 EP rank r/N` load line. `session` is rank 0's
/// session
/// cache seed. `expert_vram_layers` (`None` = the env/auto resolution of a
/// single-device load) pins the expert VRAM split in MoE layers resident of
/// trunk + MTP (the total = fully resident, the MTP layer last): an EP mesh
/// passes the same value, [`qwen4_ep_vram_layers`], to every rank so each
/// layer is resident on all ranks or none; a rank that cannot hold it
/// refuses before allocating.
///
/// Safe to call from any thread: it touches only `gpu` (bound to the calling
/// thread by its own methods), the source, and process-global read-only config.
pub fn load_qwen4_rank(
    gpu: &mut rdna_compute::Gpu,
    ep_seal: Option<hipfire_arch_qwen4::ep_seal::Qwen4EpRankSeal>,
    shared: &Qwen4RankShared,
    expert_vram_layers: Option<usize>,
    session: Option<Qwen4SessionSeed<'_>>,
    src: ModelSource,
) -> Result<Qwen4RankLoad, String> {
    load_qwen4_rank_scoped(gpu, ep_seal, shared, expert_vram_layers, session, src, None)
}

/// [`load_qwen4_rank`] for one pipeline band (`band == Some`) or the whole
/// model (`None`, the unchanged Single / expert-parallel load). A band loads
/// only its [`Qwen4BandScope`]: the admitted artifact is projected through
/// `Qwen4HfqmArtifact::for_band`, the manifest fulfills only the band's
/// resources on `gpu`, and the bundle is assembled with
/// `assemble_band_with_ple_rows`. A band attaches its own session cache from
/// `session`, with no disk tier (none at all under native MTP). Every routed
/// expert of the band is resident (no host-mapped layers, hot set or
/// automatic context); the output band alone attaches the native MTP head
/// when it is on (`shared.native_mtp`, [`qwen4_rank_native_mtp`]); and the
/// forward scratch plan is the one chosen from the full manifest so every
/// band agrees.
fn load_qwen4_rank_scoped(
    gpu: &mut rdna_compute::Gpu,
    ep_seal: Option<hipfire_arch_qwen4::ep_seal::Qwen4EpRankSeal>,
    shared: &Qwen4RankShared,
    expert_vram_layers: Option<usize>,
    session: Option<Qwen4SessionSeed<'_>>,
    src: ModelSource,
    band: Option<&hipfire_arch_qwen4::weights::Qwen4BandScope>,
) -> Result<Qwen4RankLoad, String> {
    let backend = shared.backend;
    let vmm_granularity = shared.vmm_granularity;
    let max_k = shared.max_k;
    let mut native_mtp = shared.native_mtp;
    let hotset_policy = hipfire_arch_qwen4::expert_hotset::HotsetPolicy::from_env()?;
    let placement_policy = hipfire_arch_qwen4::expert_hotset::PlacementPolicy::from_env()?;
    let hotset_budget = hipfire_arch_qwen4::expert_hotset::HotsetBudget::from_env()?;
    let ep = match &ep_seal {
        Some(bound) => {
            if let ModelSource::Hfq(hfq) = &src {
                let identity = hfq
                    .source_identity()
                    .map_err(|error| format!("qwen4: EP source identity: {error}"))?;
                bound.seal.check_source(&identity)?;
            }
            Some(
                bound
                    .seal
                    .placement(bound.rank)
                    .map_err(|error| format!("qwen4: {error}"))?,
            )
        }
        None => None,
    };
    let ep_ranks = ep.map_or(1, |placement| placement.ranks);
    // A band carries a session cache (`session`) like a Single load (except
    // under native MTP) and the output band the native MTP head; the other
    // Single-only features stay refused.
    if band.is_some()
        && (ep.is_some()
            || expert_vram_layers.is_some()
            || shared.sequence.is_some()
            || gpu.is_uma())
    {
        return Err(
            "qwen4: a topology.pp band load admits no expert parallelism, \
             automatic context or unified memory"
                .into(),
        );
    }
    // Only the input band owns PLE rows; the other bands neither report nor
    // prewarm them.
    let ple_owner = band.map_or(true, |scope| scope.input);
    let mut ctx = RankCtx {
        gpu,
        path: shared.path.as_str(),
        max_seq: shared.max_seq,
        sequence: shared.sequence,
        spec: shared.spec,
        state_quant_override: shared.state_quant.as_deref(),
        kv_mode: shared.kv_mode.as_str(),
    };
    let ModelSource::Hfq(mut hfq) = src else {
        return Err(
            "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
        );
    };
    // The PLE rows source (`memory.ple_format`).  Admission already
    // resolved and attached a sidecar on this source; resolving again is
    // idempotent and yields the choice for the load log (and covers a
    // source that reaches the carrier without admission).  A developer
    // profile's embedded rows (or a profile without PLE) have no sidecar
    // to resolve: `None`.
    let choice = crate::admission::resolve_qwen4_ple_rows_hfq(&mut hfq, ctx.path, &ctx.gpu.arch)?;
    // GPU-arch admission: an artifact the device cannot run is refused here,
    // before any allocation or mapping drop.
    let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(&hfq, Some(&ctx.gpu.arch))
        .map_err(|error| error.to_string())?;
    // Refused before any allocation (source admission applies the same
    // check): an inventory-valid developer profile the executor cannot
    // run would otherwise fail after the weights were fulfilled.
    hipfire_arch_qwen4::bundle::admit_executable_profile(&receipt.config)
        .map_err(|error| format!("qwen4: {error}"))?;
    // A band's forward scratch plan is chosen once, from the full manifest,
    // so every band attaches the same plan (a band-scoped manifest would
    // miss the PLE entries the selector inspects).
    let full_scratch_mode = match band {
        Some(_) => Some(
            hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
                ctx.gpu,
                &hfq,
                &receipt.config,
                &receipt.manifest,
            )
            .map_err(|error| error.to_string())?,
        ),
        None => None,
    };
    let receipt = match band {
        Some(scope) => receipt
            .for_band(scope, 0)
            .map_err(|error| format!("qwen4: topology.pp band projection failed: {error}"))?,
        None => receipt,
    };
    // The trunk overlay sidecar (`<model>.trunk-overlay`): bound to this
    // base while it is open, applied by `attach_forward`.  Only a bad
    // `HIPFIRE_QWEN4_TRUNK_OVERLAY` value fails the load.
    let trunk_overlay = hipfire_arch_qwen4::resolve_trunk_overlay(
        &hfq,
        std::path::Path::new(ctx.path),
        &ctx.gpu.arch,
        &receipt.config,
    )?;
    if band.is_some() && trunk_overlay.is_some() {
        return Err("qwen4: a trunk overlay sidecar cannot be used with pipeline parallelism (topology.pp > 1); set topology.pp = 1 or remove the overlay".into());
    }
    // The profile is known: the head-free MTP and developer-PLE refusals
    // happen here, before any allocation (source admission applies the
    // same checks).
    let requested_native_mtp = native_mtp;
    let head_native_mtp = crate::admission::qwen4_native_mtp_for_head(
        native_mtp,
        ctx.spec.mtp,
        receipt.config.mtp.is_some(),
        receipt.config.profile.id(),
    )?;
    // Only the output band of a PP load attaches the head (and only its
    // manifest declares the MTP weights it binds).
    native_mtp = qwen4_rank_native_mtp(head_native_mtp, band);
    if requested_native_mtp && !head_native_mtp {
        eprintln!(
            "  qwen4 native MTP: off (profile {} has no MTP head)",
            receipt.config.profile.id()
        );
    }
    if let Some(note) = crate::admission::qwen4_developer_ple_policy(
        receipt.config.profile.is_flash_next(),
        receipt.config.profile.id(),
        hipfire_config::memory::ple_format(),
        hipfire_config::memory::ple_hot_store_bytes(),
    )? {
        eprintln!("  qwen4 {note}");
    }
    // A rank-order table is its own hot-row pack and is served through
    // its store; every other source reads through 128 shard descriptors.
    let rank_table = receipt.ple_rows.rank_table().cloned();
    let ple_descriptors = match rank_table {
        Some(_) => None,
        None => Some(
            receipt
                .ple_row_descriptors(&hfq)
                .map_err(|error| error.to_string())?,
        ),
    };
    let (ple_rows_path, ple_rows_bytes) = match &receipt.ple_rows {
        hipfire_arch_qwen4::Qwen4PleRows::Absent => (std::path::PathBuf::new(), 0),
        _ => receipt
            .ple_rows_source(&hfq)
            .map_err(|error| error.to_string())?,
    };
    let (rows_encoding, rows_total) = match &receipt.ple_rows {
        hipfire_arch_qwen4::Qwen4PleRows::Absent => (String::new(), 0),
        hipfire_arch_qwen4::Qwen4PleRows::Sidecar { encoding, .. } => (
            encoding.dtype_name().to_string(),
            hfq.ple_rows().map_or(0, |rows| rows.header().data_bytes),
        ),
        hipfire_arch_qwen4::Qwen4PleRows::Embedded => {
            let descriptors = ple_descriptors.as_deref().unwrap_or_default();
            (
                descriptors
                    .first()
                    .map_or_else(String::new, |descriptor| descriptor.dtype.clone()),
                descriptors.iter().map(|descriptor| descriptor.length).sum(),
            )
        }
    };
    if ple_owner {
        match &choice {
            Some(choice) => {
                eprintln!(
                    "  qwen4 ple rows: {} from {} ({rows_encoding}, {} order, {:.1} GB)",
                    choice.format.name(),
                    match &choice.origin {
                        hipfire_runtime::ple_sidecar::PleRowsOrigin::Embedded => {
                            "embedded model file".to_string()
                        }
                        hipfire_runtime::ple_sidecar::PleRowsOrigin::Sidecar(path) => {
                            path.display().to_string()
                        }
                    },
                    if rank_table.is_some() { "rank" } else { "row" },
                    rows_total as f64 / 1e9
                );
                if let Some(note) = &choice.note {
                    eprintln!("  qwen4 ple rows: {note}");
                }
            }
            None => match &receipt.ple_rows {
                hipfire_arch_qwen4::Qwen4PleRows::Absent => eprintln!(
                    "  qwen4 ple rows: none (profile {} declares no PLE)",
                    receipt.config.profile.id()
                ),
                _ => eprintln!(
                    "  qwen4 ple rows: embedded developer rows ({rows_encoding}, {:.3} MB)",
                    rows_total as f64 / 1e6
                ),
            },
        }
    }
    // The row-order q4 sidecar's data section, for the post-ready
    // page-cache prewarm (when no hot store attaches).
    let q4_prewarm = hfq
        .ple_rows()
        .filter(|rows| {
            ple_owner
                && rows.encoding() == hipfire_runtime::external_rows::RowEncoding::Q4Lm
                && rows.order() == hipfire_runtime::ple_hot_pack::PackOrder::Row
        })
        .map(|rows| {
            (
                rows.path().to_path_buf(),
                rows.header().data_offset,
                rows.header().data_bytes,
            )
        });
    // The PLE rows source is part of what the forward pass computes with:
    // a different encoding, order or file is a different cache domain.
    let ple_identity = hfq.ple_rows().map_or_else(
        || "embedded".to_string(),
        |rows| {
            format!(
                "{}:{:?}:{}",
                rows.encoding().dtype_name(),
                rows.order(),
                rows.header()
                    .table
                    .as_ref()
                    .map_or("", |table| table.data_sha256.as_str())
            )
        },
    );
    let recipe = receipt.recipe().clone();
    let config = receipt.config;
    let mut manifest = receipt.manifest;
    // Expert parallelism: the routed expert stacks (trunk and MTP head)
    // are declared expert-sharded; rank `r` uploads experts `e % N == r`
    // compactly. The single-device manifest is not touched.
    if let Some(placement) = ep {
        manifest = manifest
            .with_expert_parallel(placement)
            .map_err(|error| format!("qwen4: {error}"))?;
    }
    let metadata = receipt.ple;
    // A dense FFN has no routed experts: its gate/up/down are ordinary
    // resident weights and the host-expert placement does not apply.
    let routed_experts = !config.ffn.is_dense();
    let mut placements = receipt.placements;
    let ple_rows = receipt.ple_rows;
    // Only the rank that serves requests' session cache (rank 0 / single)
    // carries a seed; its cache domain keys the snapshots.
    let session_domain = session.as_ref().map(|seed| {
        let mut domain = hipfire_runtime::serve_contract::CacheDomain::for_model(
            &hfq,
            seed.tokenizer,
            seed.chat_template,
            "qwen4",
            ctx.gpu.device_id,
        );
        domain
            .sidecar_digests
            .push(hipfire_runtime::serve_contract::sha256_len_prefixed(&[
                b"qwen4-ple-rows",
                ple_identity.as_bytes(),
            ]));
        domain
    });
    let configured_session_budget = hipfire_config::memory::session_cache_bytes();
    let session_budget =
        qwen4_pp_session_budget(band.is_some() && head_native_mtp, configured_session_budget);
    if session_budget != configured_session_budget && band.is_some_and(|scope| scope.output) {
        eprintln!(
            "  qwen4 session cache: off under topology.pp with native MTP (MTP session snapshots \
             are not run on a pipeline mesh; AR sessions are unaffected)"
        );
    }
    // An automatic context (omitted max_seq; unified memory only) is
    // sized after load from what is left (card_cap).
    let automatic = ctx.sequence.is_some_and(|sequence| sequence.automatic);
    // The forward scratch plan, decided from the actual routed-expert
    // payload while the mapping is still readable and before any
    // placement input: `auto` charges exactly the plan `attach_forward`
    // allocates below.
    // An expert-parallel load never takes the compact plan: the owner-routed
    // EP prefill refuses it, so the rank keeps `Full` and the split helper
    // charges the same.
    let scratch_mode = match full_scratch_mode {
        Some(mode) => mode,
        None if ep.is_some() => hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full,
        None => hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
            ctx.gpu, &hfq, &config, &manifest,
        )
        .map_err(|error| error.to_string())?,
    };
    if scratch_mode != hipfire_arch_qwen4::gpu_forward::Qwen4ForwardScratchMode::Full {
        let reason = if matches!(
            manifest.identity.recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ) {
            "every routed trunk layer is an expert-slab arena; HIPFIRE_QWEN4_COMPACT_SCRATCH=0 keeps the full plan"
        } else {
            "symmetric routed experts proven from the payload headers"
        };
        eprintln!("  qwen4 forward scratch: {scratch_mode:?} plan ({reason})");
    }
    let use_ranges = ctx.gpu.is_uma();
    if use_ranges {
        hfq.drop_mmap();
    }
    // The state formats (memory.kv_cache for QSA, state_quant for GDN),
    // resolved before any allocation.
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        ctx.kv_mode,
        ctx.state_quant_override.unwrap_or(""),
        ctx.gpu,
        &config,
    )?;
    let qsa_format = state_format.qsa;
    eprintln!(
        "  qwen4 state: QSA {} K/V ({} context storage), GDN {} recurrent",
        match qsa_format {
            hipfire_arch_qwen4::QsaKvFormat::Fp8 => "fp8",
            hipfire_arch_qwen4::QsaKvFormat::Fp8Fnuz => "fp8 (gfx942 FNUZ codes, F32 index keys)",
            hipfire_arch_qwen4::QsaKvFormat::Q8 => "q8 (Q8_0 blocks of 32)",
            hipfire_arch_qwen4::QsaKvFormat::F32 => "bf16 (exact F32 state)",
        },
        backend.name(),
        state_format.gdn.name()
    );
    if state_format.qsa_f16_arena {
        eprintln!("  qwen4 QSA F16 arena: on (BF-1, incremental exact prefill; decode F32)");
    }
    if state_format.qsa_pooled_bf16 {
        eprintln!("  qwen4 QSA pooled BF16 mirror: on (exact PM score input; selection unchanged)");
    }
    use hipfire_arch_qwen4::expert_residency as residency;
    const MIB: u64 = 1 << 20;
    // Unified memory dropped the mapping above: size from the index.
    let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
        // A slab layer arena is sized from the recipe's plan, never looked up
        // in the tensor index.
        crate::qwen4_slab::qwen4_entry_bytes(&recipe, entry, |entry| {
            if use_ranges {
                hfq.find_tensor_info(&entry.name)
                    .map(|info| info.data_size as u64)
            } else {
                hfq.tensor_data(&entry.name)
                    .map(|(_, bytes)| bytes.len() as u64)
            }
        })
    };
    // What `auto` sizes its placement from: free VRAM, the non-expert and
    // per-layer expert bytes, and the reserve (with its native MTP and
    // QSA gather parts). Unified memory charges context arenas as
    // committed (VMM: the first chunk's pages; legacy: all of `max_seq`),
    // never their virtual extent; a discrete card charges legacy storage
    // of `max_seq` whatever the backend, so VMM growth stays inside the
    // reserve and placement matches legacy.
    let auto_inputs = || -> Result<(u64, u64, u64, residency::AutoReserve, Option<u64>, Option<u64>), String> {
        let (free, _) = ctx
            .gpu
            .device_mem_info()
            .map_err(|error| format!("qwen4: device memory query: {error}"))?;
        let (non_expert, layer_experts, reserve, mtp_bytes, gather_bytes) = qwen4_placement_inputs(
            &*ctx.gpu,
            &config,
            &manifest.weights,
            &bytes_of,
            backend,
            use_ranges,
            ctx.max_seq,
            vmm_granularity,
            ctx.spec,
            max_k,
            native_mtp,
            routed_experts,
            state_format,
            scratch_mode,
            ep_ranks,
        )?;
        Ok((
            free as u64,
            non_expert,
            layer_experts,
            reserve,
            mtp_bytes,
            gather_bytes,
        ))
    };
    // What a whole-layer placement keeps free beyond the reserve: the safety
    // margin and, on a slab load that allocates them (`staged`), the two
    // dedicated long-prefill stage buffers sized from the largest layer.
    // The hot set path charges its own margin and tier-plan stage.
    let slab_stage = residency::dedicated_slab_stage(
        residency::expert_stage_enabled(&ctx.gpu.arch),
        matches!(
            manifest.identity.recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. }
        ),
        ep.is_some(),
    );
    let placement_charges = |layer_experts: u64, staged: bool| {
        residency::PlacementCharges::new(layer_experts, staged && slab_stage)
            .map_err(|error| format!("qwen4: {error}"))
    };
    let explicit =
        residency::expert_vram_layers_from_env().map_err(|error| format!("qwen4: {error}"))?;
    use hipfire_arch_qwen4::expert_hotset as hotsets;
    let model_path = std::path::Path::new(ctx.path);
    let hotset_env = hotsets::HotsetEnv {
        gpu_arch: &ctx.gpu.arch,
        unified_memory: use_ranges,
        expert_parallel: ep.is_some() || expert_vram_layers.is_some(),
        explicit_vram_layers: explicit.is_some(),
        source: match &manifest.identity.recipe {
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
                hotsets::HotsetSource::Aggregate
            }
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
                hotsets::HotsetSource::SlabArena
            }
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
                hotsets::HotsetSource::StockFp8
            }
            _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
        },
    };
    let whole_first = hotsets::decide_whole_first(placement_policy, &hotset_env, band.is_some());
    if let hotsets::WholeFirstDecision::Ignored(reason) = whole_first {
        eprintln!(
            "{}policy=whole-first not applied ({reason}); placement unchanged",
            hotsets::LOG_PREFIX
        );
    }
    // `whole-first` keeps the `hotset=0` whole-layer placement: the sidecar
    // is not consulted, so no tier plan, owner or route is ever built.
    let hotset_path = if whole_first == hotsets::WholeFirstDecision::Active {
        if hotset_budget.is_fixed() {
            return Err(format!(
                "qwen4: {} requires an applicable ranked hotset, but {}=whole-first keeps whole-layer placement",
                hotsets::HOT_BUDGET_MIB_ENV,
                hotsets::PLACEMENT_POLICY_ENV,
            ));
        }
        None
    } else {
        match hotsets::preflight(&hotset_policy, model_path, &hotset_env) {
            Ok(path) => Some(path),
            Err(skip) => {
                if hotset_budget.is_fixed() {
                    return Err(format!(
                        "qwen4: {} requires an applicable ranked hotset: {skip}",
                        hotsets::HOT_BUDGET_MIB_ENV,
                    ));
                }
                let path = hotset_policy.sidecar_path(model_path);
                if let Some(line) = skip.log_line(&path) {
                    eprintln!("{line}");
                }
                None
            }
        }
    };
    // All optional-sidecar decisions finish against the original manifest:
    // binding, source/manifest eligibility, host limits, plan and a fresh
    // free-VRAM check. A fixed budget fails closed; otherwise a refusal
    // leaves whole-layer placement untouched.
    let mut hotset_host_bytes = 0;
    let (hotset, hotset_plan) = if let Some(path) =
        hotset_path.as_deref().filter(|_| band.is_none())
    {
        let prepare = || -> Result<_, String> {
            let base = hotsets::HotsetBase::from_hfq(&hfq)?;
            let sidecar = hotsets::HotsetSidecar::load_bound(
                path,
                config.num_hidden_layers,
                config.num_experts,
                config.num_experts_per_tok,
                &base,
            )
            .map_err(|skip| skip.to_string())?;
            manifest
                .check_expert_tiers()
                .map_err(|error| error.to_string())?;
            let slabs =
                hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(&hfq, &config, &manifest)?;
            let (free, non_expert, _, reserve, mtp_bytes, gather_bytes) = auto_inputs()?;
            let reserve_note = reserve.describe();
            let reserve = reserve.bytes();
            let total =
                hotsets::require_budget(residency::expert_hotset_budget(free, non_expert, reserve))
                    .map_err(|skip| skip.to_string())?;
            // Retain the automatic fixed point for stage buffers. Under a
            // pin, reserve those buffers alongside the unchanged hot budget.
            let stage_enabled = residency::expert_stage_enabled(&ctx.gpu.arch);
            let mut stage = 0u64;
            let (plan, budget) = loop {
                let available = hotsets::require_budget(total.checked_sub(stage).ok_or_else(|| {
                    format!(
                        "the {} MiB expert budget cannot hold the {} MiB of long-prefill stage buffers; free GPU memory or lower the expert placement",
                        total / MIB,
                        stage / MIB
                    )
                }))
                .map_err(|skip| skip.to_string())?;
                let budget = hotset_budget.select(available)?;
                let plan = hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs(
                    &slabs,
                    sidecar.rank(),
                    budget,
                )
                .map_err(|error| error.to_string())?;
                hotsets::require_experts(plan.device_experts()).map_err(|skip| skip.to_string())?;
                let need = if stage_enabled {
                    residency::expert_tier_stage_bytes(&plan)
                } else {
                    0
                };
                if need <= stage {
                    break (plan, budget);
                }
                stage = need;
            };
            plan.check_executable(&ctx.gpu.arch)
                .map_err(|error| error.to_string())?;
            let host_bytes = plan
                .owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped)
                .checked_add(residency::hotset_host_mapped_bytes(
                    &manifest.weights,
                    bytes_of,
                    config.num_hidden_layers,
                )?)
                .ok_or("host byte accounting overflow")?;
            residency::check_host_ram(
                host_bytes,
                rdna_compute::kv_slots::mem_available_bytes(),
                residency::ttm_pool_estimate(),
            )?;
            residency::check_gtt_cap(host_bytes, residency::gtt_budget())?;
            // Re-query before either placement branch changes WeightEntries
            // or tier conversion removes aggregate tensors. A shortfall
            // fails a fixed pin; auto can still take whole-layer placement.
            let (latest_free, _) = ctx
                .gpu
                .device_mem_info()
                .map_err(|error| format!("memory re-query before materialize: {error}"))?;
            let latest_budget = hotsets::require_budget(
                residency::expert_hotset_budget(latest_free as u64, non_expert, reserve).and_then(
                    |bytes| {
                        bytes
                            .checked_sub(stage)
                            .ok_or_else(|| "no budget after long-prefill stage buffers".to_string())
                    },
                ),
            )
            .map_err(|skip| skip.to_string())?;
            // Check the entire requested ceiling, not just the rounded-down
            // payload: a pinned budget must never silently lose headroom.
            hotset_budget.select(latest_budget)?;
            if plan.owner_bytes(hipfire_arch_qwen4::expert_tier_plan::Placement::Device)
                > latest_budget
            {
                return Err(format!(
                    "free VRAM dropped to {} MiB before the experts were placed; the planned GPU-resident experts need more than the remaining {} MiB. Free GPU memory or place fewer experts on the GPU",
                    latest_free as u64 / MIB, latest_budget / MIB,
                ));
            }
            eprintln!(
                "  qwen4 expert hot set budget: {} MiB free - {} MiB non-expert weights - {} MiB reserve ({reserve_note}; {} MiB native MTP, {} MiB QSA gather) - {} bytes tier tables - {} MiB safety margin - {} MiB long-prefill stage buffers = {} MiB",
                free / MIB, non_expert / MIB, reserve / MIB,
                mtp_bytes.unwrap_or(0) / MIB, gather_bytes.unwrap_or(0) / MIB,
                residency::EXPERT_TIER_TABLE_BYTES,
                residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES / MIB, stage / MIB, (total - stage) / MIB,
            );
            if hotset_budget.is_fixed() {
                eprintln!(
                    "  qwen4 expert hot set fixed budget: {}={} MiB ({} bytes); ranked prefix independent of MTP mode",
                    hotsets::HOT_BUDGET_MIB_ENV, budget / MIB, budget,
                );
            }
            Ok((sidecar, std::sync::Arc::new(plan), budget, host_bytes))
        };
        match prepare() {
            Ok((sidecar, plan, budget, host_bytes)) => {
                hotset_host_bytes = host_bytes;
                (Some(sidecar), Some((plan, budget)))
            }
            Err(reason) => {
                if hotset_budget.is_fixed() {
                    return Err(format!(
                        "qwen4: {} could not be applied: {reason}",
                        hotsets::HOT_BUDGET_MIB_ENV,
                    ));
                }
                eprintln!(
                    "  qwen4 expert hot set: {} present, not applied ({reason}); whole-layer placement",
                    path.display(),
                );
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    if !routed_experts {
        residency::refuse_host_expert_request(explicit, config.profile.id())
            .map_err(|error| format!("qwen4: {error}"))?;
    }
    // Unset on a discrete card: fully resident only where every routed
    // expert, the non-expert weights and the `auto` reserve fit in free
    // VRAM; otherwise `auto` (an unconditional resident load would fail
    // `hipMalloc` part-way). Unified memory keeps the resident load, but
    // refuses it up front (with `memory.oom_guard`) when it does not fit
    // the free device memory, GTT clamped to host RAM: an overshoot there
    // reaches the global OOM killer.
    let mut measured = None;
    let mut unset_note = String::new();
    let placement = if band.is_some() {
        // A pipeline band keeps every routed expert resident: host-mapped
        // band layers need their own residency cell, and capacity was
        // admitted against the band census before any allocation.
        if explicit.is_some() {
            return Err(format!(
                "qwen4: {} cannot be used with pipeline parallelism (topology.pp > 1) because pipeline stages keep every expert on the GPU; unset it or set topology.pp = 1",
                residency::EXPERT_VRAM_LAYERS_ENV
            ));
        }
        None
    } else if hotset_plan.is_some() {
        None
    } else if let Some(layers) = expert_vram_layers {
        // The split an expert-parallel load agreed on (minimum over
        // ranks): applied as is, never re-derived per rank.
        Some(residency::ExpertVramLayers::Layers(layers))
    } else if use_ranges {
        if explicit.is_none() {
            let (free, non_expert, _, reserve, ..) = auto_inputs()?;
            let reserve_note = reserve.describe();
            let reserve = reserve.bytes();
            let experts =
                residency::routed_expert_bytes_ranks(&manifest.weights, bytes_of, ep_ranks)
                    .map_err(|error| format!("qwen4: {error}"))?;
            let fits = residency::fits_fully_resident(free, non_expert, experts, reserve);
            let detail = format!(
                    "{} MiB routed experts + {} MiB non-expert weights + {} MiB reserve ({reserve_note}) {} {} MiB free device memory",
                    experts / MIB,
                    non_expert / MIB,
                    reserve / MIB,
                    if fits { "fit in" } else { "exceed" },
                    free / MIB
                );
            if !fits {
                if state_format.qsa_f16_arena
                    || state_format.qsa_pooled_bf16
                    || hipfire_config::oom_guard_effective(Some(&ctx.gpu.arch))
                {
                    return Err(format!(
                            "qwen4: unified memory cannot hold the resident load ({detail}); free host memory, or set memory.oom_guard = off to load anyway"
                        ));
                }
                eprintln!("  qwen4: {detail} (memory.oom_guard off: loading anyway)");
            }
            unset_note = format!(" ({detail})");
        }
        explicit
    } else {
        residency::resolve_expert_vram_layers(explicit, || {
            let inputs = auto_inputs()?;
            let experts =
                residency::routed_expert_bytes_ranks(&manifest.weights, bytes_of, ep_ranks)
                    .map_err(|error| format!("qwen4: {error}"))?;
            let (free, non_expert, layer_experts) = (inputs.0, inputs.1, inputs.2);
            let base_reserve = inputs.3.bytes();
            // Fully resident has no host layer, so no stage buffers; the
            // safety margin is charged exactly as `auto` charges it.
            let charges = placement_charges(layer_experts, false)?;
            let reserve = charges
                .reserve(base_reserve)
                .map_err(|error| format!("qwen4: {error}"))?;
            let fits = residency::fits_fully_resident(free, non_expert, experts, reserve);
            unset_note = format!(
                    " ({} MiB routed experts + {} MiB non-expert weights + {} MiB reserve (base {}; plus {}) {} {} MiB free VRAM)",
                    experts / MIB,
                    non_expert / MIB,
                    reserve / MIB,
                    inputs.3.describe(),
                    charges.describe(),
                    if fits { "fit in" } else { "exceed" },
                    free / MIB
                );
            measured = Some(inputs);
            Ok(fits)
        })?
    };
    let moe_layers = residency::moe_layer_count(&config);
    // A split covering every MoE layer is the fully resident load (the MTP
    // layer's experts too).
    let placement = if expert_vram_layers.is_some_and(|layers| layers >= moe_layers) {
        None
    } else {
        placement
    };
    if config.expert_keep.is_some() && placement.is_some() {
        return Err("qwen4.expert_keep artifacts require fully resident single-device expert payloads; host-mapped/streamed expert placement is not supported (unset HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS and ensure the kept payload fits)".into());
    }
    // The agreed split must fit this rank: refuse before allocating.
    if let Some(layers) = expert_vram_layers {
        let (free, non_expert, layer_experts, reserve, ..) = auto_inputs()?;
        // Expert-parallel loads have no slab stage; the safety margin is
        // charged like `auto`'s.
        let reserve = placement_charges(layer_experts, false)?
            .reserve(reserve.bytes())
            .map_err(|error| format!("qwen4: {error}"))?;
        let own = residency::ep_rank_layers(&config, free, non_expert, layer_experts, reserve);
        if own < layers.min(moe_layers) {
            return Err(format!(
                    "qwen4: the agreed expert VRAM split of {layers} of {moe_layers} MoE layers does not fit this device: \
                     {} MiB free keeps only {own} (trunk {} MiB, {} MiB per expert layer, {} MiB reserve). Free GPU memory or lower the number of expert layers kept on the GPU",
                    free / MIB,
                    non_expert / MIB,
                    layer_experts / MIB,
                    reserve / MIB
                ));
        }
    }
    if !routed_experts && placement.is_some() {
        return Err(format!(
                "qwen4: profile {} has a dense FFN (no routed experts to move to host RAM) and its resident weights do not fit device memory{unset_note}; free VRAM or lower memory.max_seq",
                config.profile.id()
            ));
    }
    if hotset_plan.is_none() {
        eprintln!(
            "  qwen4 expert placement: {}{unset_note}",
            match (explicit, placement) {
                _ if expert_vram_layers.is_some() => format!(
                    "agreed split {} of {moe_layers} MoE layers resident",
                    expert_vram_layers.unwrap_or(0)
                ),
                (Some(residency::ExpertVramLayers::Layers(layers)), _) =>
                    format!("{}={layers}", residency::EXPERT_VRAM_LAYERS_ENV),
                (Some(residency::ExpertVramLayers::Auto), _) =>
                    format!("{}=auto", residency::EXPERT_VRAM_LAYERS_ENV),
                (None, None) => format!(
                    "{} unset, fully resident",
                    residency::EXPERT_VRAM_LAYERS_ENV
                ),
                (None, Some(_)) => format!("{} unset, auto", residency::EXPERT_VRAM_LAYERS_ENV),
            }
        );
    }
    // Host memory is kept out of reclaim only by a process started for a
    // Qwen4 model (`prepare_host_memory_for`) or with the placement set;
    // one started for another model (serve switching models) can no
    // longer, its runtime having read the switches.
    if !use_ranges && std::env::var_os("HSA_USERPTR_FOR_PAGED_MEM").is_none() {
        eprintln!(
                "  qwen4: host memory the GPU reads stays in reclaim (this process started for another model), so host-memory pressure can stall the load; start it for this model or with {} set",
                residency::EXPERT_VRAM_LAYERS_ENV
            );
    }
    // Measured before placement mutates the manifest (the fit line below),
    // with the same charged reserve the EP split check applied.
    let fit_inputs = ep
        .map(|_| -> Result<_, String> {
            let (free, non_expert, layer_experts, reserve, mtp, gather) = auto_inputs()?;
            let reserve = placement_charges(layer_experts, false)?
                .reserve(reserve.bytes())
                .map_err(|error| format!("qwen4: {error}"))?;
            Ok((free, non_expert, layer_experts, reserve, mtp, gather))
        })
        .transpose()?;
    let mut whole_first_plan = None;
    let borrowed_stage_donors = residency::expert_stage_enabled(&ctx.gpu.arch)
        && matches!(
            manifest.identity.recipe,
            hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. }
        )
        && ep.is_none();
    let mut applied_layers = moe_layers;
    if let Some(policy) = placement {
        if use_ranges {
            return Err(format!(
                "qwen4: {} puts experts in host RAM, which only a discrete GPU supports; unset it on this unified-memory device",
                residency::EXPERT_VRAM_LAYERS_ENV
            ));
        }
        let vram_layers = match policy {
            residency::ExpertVramLayers::Layers(layers) => layers,
            residency::ExpertVramLayers::Auto => {
                let (free, non_expert, layer_experts, auto, mtp_bytes, gather_bytes) =
                    match measured.take() {
                        Some(inputs) => inputs,
                        None => auto_inputs()?,
                    };
                let gather_note = gather_bytes
                    .map(|bytes| format!(", {} MiB QSA gather scratch", bytes / MIB))
                    .unwrap_or_default();
                // The reserve is the larger of the historical formula and the
                // census (every term itemised below); the safety margin and
                // (slab only) the dedicated long-prefill stage buffers are
                // charged on top.
                let charges = placement_charges(layer_experts, true)?;
                let charged = charges
                    .reserve(auto.bytes())
                    .map_err(|error| format!("qwen4: {error}"))?;
                eprintln!(
                        "  qwen4 auto expert placement: {} MiB free, {} MiB non-expert weights, {} MiB reserved (base {}; {} MiB native MTP{gather_note}; plus {}), {} MiB per expert layer",
                        free / MIB,
                        non_expert / MIB,
                        charged / MIB,
                        auto.describe(),
                        mtp_bytes.unwrap_or(0) / MIB,
                        charges.describe(),
                        layer_experts / MIB
                    );
                // With every routed expert in host RAM the load would
                // still leave less than the reserve, and fail later at
                // the first request (e.g. beside another process's
                // model): refuse before allocating instead.
                if whole_first == hotsets::WholeFirstDecision::Active {
                    let plan = residency::WholeFirstPlan::auto(
                        free,
                        non_expert,
                        layer_experts,
                        config.num_hidden_layers,
                        moe_layers,
                        auto.bytes(),
                        &charges,
                        borrowed_stage_donors,
                    )
                    .map_err(|error| format!("qwen4: {error}"))?;
                    let layers = plan.whole_layers;
                    whole_first_plan = Some(plan);
                    layers
                } else {
                    residency::auto_placement_layers(
                        free,
                        non_expert,
                        layer_experts,
                        config.num_hidden_layers,
                        auto.bytes(),
                        &charges,
                    )
                    .map_err(|error| format!("qwen4: {error}"))?
                }
            }
        };
        let moved = residency::place_routed_experts(&mut manifest.weights, vram_layers);
        applied_layers = vram_layers.min(config.num_hidden_layers);
        // This rank's pinned bytes; the host check covers every rank that
        // has not pinned yet (this one and the later ranks), each at its own
        // share (an uneven split gives the low ranks one more expert).
        let rank_host_bytes =
            residency::host_mapped_bytes_ranks(&manifest.weights, bytes_of, ep_ranks)
                .map_err(|error| format!("qwen4: {error}"))?;
        let own_rank = ep.map_or(0, |placement| placement.rank);
        let host_bytes = rank_host_bytes[own_rank];
        let host_check_bytes = rank_host_bytes[own_rank..]
            .iter()
            .fold(0u64, |total, bytes| total.saturating_add(*bytes));
        let mem_available = rdna_compute::kv_slots::mem_available_bytes();
        let ttm_pool = residency::ttm_pool_estimate();
        residency::check_host_ram(host_check_bytes, mem_available, ttm_pool)
            .map_err(|error| format!("qwen4: {error}"))?;
        residency::check_gtt_cap(host_check_bytes, residency::gtt_budget())
            .map_err(|error| format!("qwen4: {error}"))?;
        // SKU emulation: the same pinned-host (GTT) bytes against the capacity
        // ledger's shared-host/GTT room, in addition to the real checks above.
        // No-op when HIPFIRE_EMU is off.
        ctx.gpu
            .emu_check_host_allocation(host_check_bytes, true)
            .map_err(|error| format!("qwen4: {error}"))?;
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        eprintln!(
                "  qwen4 routed experts: layers 0..{} in VRAM, {moved} expert tensors ({:.1} GiB) in pinned host RAM \
                 (MemAvailable - CMA {:.1} GiB + {:.1} GiB estimated in TTM's page pool)",
                vram_layers.min(config.num_hidden_layers),
                gib(host_bytes),
                gib(mem_available.unwrap_or(0)),
                gib(ttm_pool)
            );
        if let Some(plan) = &whole_first_plan {
            let resident = residency::resident_routed_expert_bytes(&manifest.weights, bytes_of)
                .map_err(|error| format!("qwen4: {error}"))?;
            eprintln!("{}{}", hotsets::LOG_PREFIX, plan.describe(resident));
        }
        if native_mtp
            && !crate::admission::qwen4_mtp_with_host_mapped_experts(ctx.spec, &ctx.gpu.arch, moved)
        {
            native_mtp = false;
            eprintln!(
                "{}",
                hipfire_config::mtp_host_spill_decline_line(
                    "qwen4 native",
                    moved,
                    host_bytes,
                    &format!("on {}", ctx.gpu.arch),
                )
            );
        }
    }
    if whole_first == hotsets::WholeFirstDecision::Active && placement.is_none() {
        if let Some(inputs) = &measured {
            let plan = residency::WholeFirstPlan::fully_resident(moe_layers, inputs.2);
            let resident = residency::resident_routed_expert_bytes(&manifest.weights, bytes_of)
                .map_err(|error| format!("qwen4: {error}"))?;
            eprintln!("{}{}", hotsets::LOG_PREFIX, plan.describe(resident));
        }
    }
    if let Some((plan, _)) = &hotset_plan {
        // Trunk payloads are tier-owned; the unchanged MTP entries still use
        // their ordinary host-mapped WeightEntries.
        let mtp_tensors =
            residency::place_routed_experts(&mut manifest.weights, config.num_hidden_layers);
        manifest = manifest
            .with_expert_tiers()
            .map_err(|error| error.to_string())?;
        let expected: std::collections::BTreeSet<_> = manifest.resident_keys().collect();
        placements
            .retain(|placement| expected.contains(&(placement.name.as_str(), placement.layer)));
        // Host/GTT checks already passed while the original manifest was
        // intact, so none of these optional-sidecar decisions can fail here.
        let host_bytes = hotset_host_bytes;
        let host_experts = 48 * 512 - plan.device_experts();
        let host_projections = host_experts * 2 + mtp_tensors;
        if native_mtp
            && !crate::admission::qwen4_mtp_with_host_mapped_experts(
                ctx.spec,
                &ctx.gpu.arch,
                host_projections,
            )
        {
            native_mtp = false;
            eprintln!(
                "{}",
                hipfire_config::mtp_host_spill_decline_line(
                    "qwen4 native",
                    host_projections,
                    host_bytes,
                    &format!("on {}", ctx.gpu.arch),
                )
            );
        }
    }
    // Expert parallelism: this rank's fit line, at the split it applied.
    if let (Some(placement), Some((free, non_expert, layer_experts, reserve, ..))) =
        (ep, fit_inputs)
    {
        eprintln!(
            "{}",
            residency::ep_fit_rank_line(
                &config,
                placement.rank,
                placement.ranks,
                free,
                non_expert,
                layer_experts,
                reserve,
                applied_layers,
            )
        );
    }
    // Context-sized QSA arenas (trunk layers in `qsa_format`, plus the MTP
    // head's, which stays F32): what legacy storage allocates at assembly
    // (named in the refusal when it does not fit), and VMM storage only
    // reserves as address space.
    let context_state_mib = config
        .qsa_context_arena_bytes(ctx.max_seq, state_format.layout())
        .and_then(|trunk| trunk.checked_mul(config.n_full_layers()))
        .zip(if native_mtp {
            config.qsa_context_arena_bytes(ctx.max_seq, hipfire_arch_qwen4::QsaKvFormat::F32)
        } else {
            Some(0)
        })
        .and_then(|(trunk, mtp)| trunk.checked_add(mtp))
        .map_or(usize::MAX, |bytes| bytes >> 20);
    // An EP rank fulfills under its seal's identity (the load's one mesh
    // epoch, its logical rank, its physical device); one device is its own
    // single-device mesh.
    let (mesh, expected) = match &ep_seal {
        Some(bound) => (
            bound.seal.mesh().clone(),
            bound
                .seal
                .origin(bound.rank)
                .ok_or_else(|| format!("qwen4: rank {} is outside the EP seal", bound.rank))?,
        ),
        None => {
            let mesh = DeviceMesh::single().map_err(|error| format!("qwen4: mesh: {error}"))?;
            let expected = WeightOrigin::for_single(&mesh, ctx.gpu);
            (mesh, expected)
        }
    };
    // A rank-order PLE table's worker starts now: it reads the table's
    // hottest prefix at a capped rate while the weights load, and
    // prewarms on from there once the model is ready. The PLE hot-row
    // store also opens before the weight sweep (while the source index is
    // in hand); its worker verifies the pack meanwhile and prewarms it
    // once the model is ready. Dropping either on an error path stops and
    // joins its worker.
    #[cfg(unix)]
    let ple_table_worker = rank_table.as_ref().map(|store| {
        // What the sweep puts in host RAM: on unified memory every
        // uploaded weight (external PLE rows stay on disk), on a discrete
        // card the host-mapped experts.
        let host_weight_bytes = if use_ranges {
            manifest
                .weights
                .iter()
                .filter(|entry| !entry.residency.is_external())
                .filter_map(|entry| bytes_of(entry))
                .sum()
        } else {
            residency::host_mapped_bytes_ranks(&manifest.weights, bytes_of, ep_ranks)
                .map(|per_rank| {
                    per_rank[ep.map_or(0, |placement| placement.rank)]
                        + hotset_plan.as_ref().map_or(0, |(plan, _)| {
                            plan.owner_bytes(
                                hipfire_arch_qwen4::expert_tier_plan::Placement::HostMapped,
                            )
                        })
                })
                .unwrap_or(0)
        };
        start_ple_table_worker(store, host_weight_bytes)
    });
    #[cfg(unix)]
    let (ple_table_worker, ple_load_fill) = match ple_table_worker {
        Some((worker, fill)) => (Some(worker), fill),
        None => (None, 0),
    };
    #[cfg(not(unix))]
    let (ple_table_worker, ple_load_fill) = {
        if rank_table.is_some() {
            return Err(
                "qwen4: rank-ordered PLE table workers are unsupported on Windows; run on Linux"
                    .into(),
            );
        }
        (None::<hipfire_arch_qwen4::bundle::PleHotWorker>, 0u64)
    };
    let ple_hot = match (&metadata, ple_descriptors.as_deref()) {
        (Some(metadata), Some(descriptors)) if config.profile.is_flash_next() => {
            start_ple_hot_store(
                use_ranges,
                &hfq,
                metadata,
                descriptors,
                &ple_rows_path,
                ple_rows_bytes,
                ctx.path,
            )
        }
        _ => None,
    };
    // Every allocation of the weight sweep is immutable model data: railgun
    // check mode verifies it unchanged instead of snapshotting it (§2.4).
    let weights_role =
        hip_bridge::registry::role_scope(hip_bridge::registry::AllocationRole::Weights);
    let expert_tiers = if let Some((plan, budget)) = hotset_plan {
        use hipfire_arch_qwen4::expert_tier_plan::Placement;
        let device_experts = plan.device_experts();
        let lease = hipfire_arch_qwen4::expert_tier_owners::materialize(
            ctx.gpu,
            plan,
            &|offset, len| {
                let offset = usize::try_from(offset).map_err(|_| "hotset source offset overflow")?;
                let len = usize::try_from(len).map_err(|_| "hotset source length overflow")?;
                hfq.data_range(offset, len)
                    .ok_or_else(|| format!("HIPFIRE_QWEN4_EXPERT_HOTSET: source span {offset}+{len} is outside the artifact mmap"))
            },
        ).map_err(|error| format!(
            "HIPFIRE_QWEN4_EXPERT_HOTSET: materialize failed after tier placement was committed (owners may be partly allocated; cannot safely fall back): {error}"
        ))?;
        let digest = lease.logical_digest();
        use std::fmt::Write as _;
        let mut digest_hex = String::with_capacity(16);
        for byte in &digest[..8] {
            write!(&mut digest_hex, "{byte:02x}").expect("writing to a String");
        }
        let sidecar = hotset.as_ref().expect("a hotset plan has a sidecar");
        lease.bind_hotset_sidecar_sha256(sidecar.sha256())
            .map_err(|error| format!("hotset numerical identity: {error}"))?;
        eprintln!(
            "  qwen4 expert hot set: {} sha256 {}, budget {} MiB, {} experts in VRAM ({} MiB), {} host-mapped ({} MiB), generation {}, digest {}, bound to base {} B, index sha256 {}",
            hotset_path.as_ref().expect("a hotset plan has a path").display(),
            sidecar.sha256(),
            budget / MIB,
            device_experts, lease.owner_bytes(Placement::Device) / MIB,
            48 * 512 - device_experts, lease.owner_bytes(Placement::HostMapped) / MIB,
            lease.generation(), digest_hex,
            sidecar.base_size(), &sidecar.base_index_sha256()[..12],
        );
        eprintln!(
            "  qwen4 expert hot set identity: aggregate-MQ4 structure validated (no full artifact digest held by this loader; artifact not rehashed); model_sha256 {}, counts_sha256 {}, format_map {}, format_map_sha256 {}",
            sidecar.model_sha256(), sidecar.counts_sha256(), sidecar.format_map(), sidecar.format_map_sha256(),
        );
        Some(lease)
    } else {
        None
    };
    let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &ctx.gpu.arch);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        ctx.gpu,
        expected,
        |entry| {
            // A slab layer arena is one span of the container, not a tensor:
            // the recipe resolves it before the by-name read.
            crate::qwen4_slab::qwen4_fulfillment_payload(
                &source,
                &recipe,
                entry,
                qwen4_use_range_payload(use_ranges, entry.residency),
            )
        },
    )
    .map_err(|error| format!("qwen4: manifest fulfillment failed: {error}"))?;
    drop(weights_role);
    let scope = band
        .cloned()
        .unwrap_or_else(|| hipfire_arch_qwen4::weights::Qwen4BandScope::single(&config));
    let mut bundle = hipfire_arch_qwen4::bundle::Qwen4Bundle::assemble_band_with_ple_rows(
            config,
            &recipe,
            transaction,
            &placements,
            ctx.gpu,
            ctx.max_seq,
            metadata,
            ple_rows,
            state_format,
            backend,
            expert_tiers,
            &scope,
        )
        .map_err(|error| match backend {
            hipfire_arch_qwen4::Qwen4KvBackend::Legacy => format!(
                "qwen4: bundle assembly failed: {error} (max_seq {} needs {context_state_mib} MiB of QSA context state; lower memory.max_seq)",
                ctx.max_seq
            ),
            hipfire_arch_qwen4::Qwen4KvBackend::Vmm => format!(
                "qwen4: bundle assembly failed: {error} (VMM QSA context reserves {context_state_mib} MiB of address space for max_seq {})",
                ctx.max_seq
            ),
        })?;
    bundle.attach_npu_trunk_identity(npu_identity);
    // The assembled weights must report the seal's placement and load
    // identity for this rank (from the fulfilled projection census and the
    // store's origin); single-device weights report no placement.
    let bound = match ep_seal {
        Some(bound) => bundle.bind_ep_seal(bound.seal, bound.rank),
        None => match bundle.weights.ep_placement() {
            None => Ok(()),
            Some(found) => Err(format!(
                "qwen4: assembled weights report expert placement {found:?} on a single-device load"
            )),
        },
    };
    if let Err(error) = bound {
        let _ = bundle.free_gpu(ctx.gpu);
        return Err(error);
    }
    // The row store reads the pack's rows from the pack once the worker
    // has verified it.
    if let Some(hot) = ple_hot {
        if let Err(error) = bundle.attach_ple_hot_store(hot) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: PLE hot store setup failed: {detail}"));
        }
    }
    if let Some(overlay) = trunk_overlay {
        if let Err(error) = bundle.attach_trunk_overlay(overlay) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: trunk overlay setup failed: {detail}"));
        }
    }
    if let Err(error) = bundle.attach_forward(ctx.gpu, ctx.max_seq, scratch_mode) {
        let detail = error.to_string();
        let _ = bundle.free_gpu(ctx.gpu);
        return Err(format!("qwen4: forward setup failed: {detail}"));
    }
    if let Some(scope) = band {
        // `attach_forward` searches smaller chunk rungs when the largest does
        // not fit; a band never shrinks the canonical chunk on its own.
        let canonical = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(
            &ctx.gpu.arch,
            ctx.max_seq,
        );
        if bundle.spec_chunk_rows() != Some(canonical) {
            let got = bundle.spec_chunk_rows();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!(
                "qwen4: topology.pp band {:?} allocated chunk rows {got:?} but the canonical \
                 chunk is {canonical}; the rank's free VRAM cannot hold it and a band never \
                 shrinks the canonical chunk",
                scope.band
            ));
        }
    }
    // The CPU splice (D1) and the hot-set tier split (D2) arm inside
    // `attach_forward` (`memory.offload_exec=cpu`, any arch the hot set applies
    // on), so the loaded placement is only known here; the native-MTP
    // admission above is unchanged by it.
    let split_layers = bundle.split_layers().len();
    if split_layers > 0 {
        eprintln!(
            "  qwen4 routed experts: CPU tier split armed on {split_layers} hot-set layers (offload_exec=cpu): decode and <=4-row verify run hot-set-cold experts on the CPU; wider prefill stays on PCIe"
        );
    }
    if bundle.has_cpu_deferred_experts() && split_layers == 0 {
        eprintln!(
            "  qwen4 routed experts: CPU splice armed (offload_exec=cpu): decode and <=4-row verify run host-placed experts on the CPU; wider prefill stays on PCIe"
        );
    }
    if native_mtp {
        if let Err(error) = bundle.attach_mtp(ctx.gpu, ctx.max_seq) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: MTP setup failed: {detail}"));
        }
    }
    // One snapshot of the PLE page-cache budgets, taken before admission:
    // the prewarm after ready sets these budgets (capped at what admission
    // leaves them), and the admission sizes the pages they keep resident.
    // On unified memory those clean page-cache pages count in
    // `MemAvailable` (and so in the device free memory) while being
    // reclaimable, so the admission shrinks them to what the context leaves
    // free: the context takes precedence over the PLE page cache.
    // The in-load fill's own target (`ple_load_fill`, fixed when it
    // started) covers the part already resident; no residency census runs
    // against the still-reading fill.
    let mut ple_prewarm = ple_prewarm_budget();
    let mut ple_hot_prewarm = ple_hot_prewarm_budget();
    let ple_resident = if let Some(worker) = &ple_table_worker {
        ple_resident_bytes(ple_prewarm.0, ple_load_fill, worker.store().data_bytes())
    } else if let Some(store) = bundle.ple_hot_store() {
        ple_resident_bytes(ple_hot_prewarm.0, 0, store.data_bytes())
    } else if let Some((_, _, data_bytes)) = &q4_prewarm {
        ple_resident_bytes(ple_prewarm.0, 0, *data_bytes)
    } else {
        0
    };
    // Unified memory (the only place admission resolves a sequence):
    // admit what requests commit after load (VMM context and gather
    // pages, native MTP request buffers) against the device memory the
    // load left free (`device_mem_info`: GTT clamped to host RAM): an
    // automatic context is the model's, capped by what fits (card_cap);
    // an explicit one must fit whole. The session cache is not charged:
    // its snapshots yield to context growth. A discrete GPU keeps its
    // requested context unchecked, as before.
    let admission = (ctx.sequence.is_some() && band.is_none()).then(
        || -> Result<crate::admission::Qwen4MemoryAdmission, String> {
            let mtp_request = if native_mtp {
                let config = &bundle.config;
                let drafts =
                    hipfire_arch_qwen4::mtp_spec::native_mtp_proposal_drafts(max_k, &ctx.gpu.arch);
                hipfire_arch_qwen4::mtp_spec::native_mtp_request_bytes(
                    config,
                    ctx.max_seq,
                    bundle.spec_chunk_rows().unwrap_or(1).max(1),
                    drafts,
                    hipfire_arch_qwen4::mtp_spec::native_mtp_capture(
                        ctx.gpu,
                        config,
                        state_format.gdn,
                        drafts,
                    ),
                    hipfire_arch_qwen4::mtp_spec::native_mtp_fill_scratch(ctx.gpu),
                )
                .ok_or("qwen4: native MTP request bytes overflow")?
            } else {
                0
            };
            let (free, _) = ctx
                .gpu
                .device_mem_info()
                .map_err(|error| format!("qwen4: device memory query: {error}"))?;
            let gpu = &*ctx.gpu;
            crate::admission::qwen4_memory_admission(
                ctx.max_seq,
                (!automatic).then_some(ctx.max_seq),
                free as u64,
                mtp_request,
                ple_resident,
                crate::admission::QWEN4_MEMORY_HEADROOM_BYTES,
                |tokens| bundle.context_growth_bytes(gpu, tokens),
            )
        },
    );
    let max_seq = match admission {
        None => ctx.max_seq,
        Some(Ok(admission)) => {
            eprintln!(
                    "  qwen4 memory admission: {} MiB free after load; a {}-token context commits {} MiB more ({} MiB QSA context incl. any F16 arena + {} MiB gather workspace + {} MiB MTP request) + {} MiB resident PLE + {} MiB headroom; card_cap {} tokens (with the current free host memory the context fits up to {} tokens; this limit moves with free host memory)",
                    admission.free / MIB,
                    admission.max_seq,
                    admission.growth() / MIB,
                    admission.context_growth / MIB,
                    admission.gather_growth / MIB,
                    admission.mtp_request / MIB,
                    admission.resident_ple / MIB,
                    admission.headroom / MIB,
                    admission.card_cap,
                    admission.card_cap
                );
            if admission.ple_shrunk() {
                eprintln!(
                    "  qwen4 ple budget: {} MiB of {} MiB kept resident so the {}-token context fits (the context takes precedence over the reclaimable PLE page cache)",
                    admission.resident_ple / MIB,
                    admission.ple_budget / MIB,
                    admission.max_seq
                );
            }
            // The in-load fill already read its target, so pages beyond the
            // admitted budget stay as reclaimable page cache: this only caps
            // what the prewarm reads on from here.
            ple_prewarm.0 = ple_prewarm.0.min(admission.resident_ple);
            ple_hot_prewarm.0 = ple_hot_prewarm.0.min(admission.resident_ple);
            if let Some(sequence) = ctx.sequence.as_mut() {
                sequence.card_cap = admission.card_cap;
            }
            admission.max_seq
        }
        Some(Err(error))
            if automatic || hipfire_config::oom_guard_effective(Some(&ctx.gpu.arch)) =>
        {
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(error);
        }
        Some(Err(error)) => {
            eprintln!("  {error} (memory.oom_guard off: loading anyway)");
            ctx.max_seq
        }
    };
    if let (true, Some(session_domain), Some(seed)) =
        (session_budget > 0, session_domain, session.as_ref())
    {
        let chunk = bundle.spec_chunk_rows().unwrap_or(0);
        let one = hipfire_arch_qwen4::bundle::session_snapshot_bytes(
            &bundle.config,
            state_format,
            native_mtp,
            chunk,
        )
        .unwrap_or(0);
        let cache =
            hipfire_runtime::session_cache::SessionCache::new(session_domain, session_budget);
        // A PP band never attaches the disk tier (admission refuses it).
        let cache = match qwen4_session_disk_tier(
            band.is_some(),
            hipfire_config::memory::session_cache_dir(),
            hipfire_config::memory::session_cache_disk_bytes(),
        ) {
            Some((dir, bytes)) => cache.with_disk(&dir, bytes),
            None => cache,
        };
        bundle.attach_session_cache(cache);
        let band_tag = band.map_or_else(String::new, |scope| {
            format!(" (PP band layers {:?})", scope.band)
        });
        eprintln!(
            "  qwen4 session cache{band_tag}: {} MiB budget, {} MiB per {chunk}-token snapshot; snapshots yield to context growth",
            session_budget >> 20,
            one >> 20
        );
        bundle.set_turn_end_token(seed.tokenizer.special_token_id("<|im_end|>"));
        if bundle.turn_snapshots_active() {
            eprintln!(
                "  qwen4 session cache{band_tag}: message-end snapshots on (HIPFIRE_QWEN4_TURN_SNAPSHOTS)"
            );
        }
    }
    // The model is loaded and admitted: a rank-order PLE table's worker
    // (it serves from the start; the worker only reads ahead) ends its
    // in-load fill and prewarms on from there, or the PLE hot-row store's
    // worker prewarms the hottest rows into the page cache, while no
    // request is served (every PLE fetch pauses it).
    if let Some(worker) = ple_table_worker {
        let store = worker.store().clone();
        let (budget, available, explicit_zero) = ple_prewarm;
        store.set_prewarm_budget(budget);
        // The fill's own `load fill stopped at ready` line reports what it
        // left resident; the prewarm continues from there.
        if budget == 0 {
            eprintln!(
                "  qwen4 ple prewarm: off ({})",
                ple_prewarm_off_reason(available, explicit_zero),
            );
        } else {
            eprintln!(
                "  qwen4 ple prewarm: budget {} MiB of a {} MiB q4 sidecar (MemAvailable - CMA {} MiB)",
                budget.min(store.data_bytes()) / MIB,
                store.data_bytes() / MIB,
                available.map_or(0, |bytes| bytes / MIB),
            );
        }
        if let Err(error) = bundle.attach_ple_table_worker(worker) {
            let detail = error.to_string();
            let _ = bundle.free_gpu(ctx.gpu);
            return Err(format!("qwen4: PLE table worker setup failed: {detail}"));
        }
    } else if let Some(store) = bundle.ple_hot_store() {
        let (budget, available) = ple_hot_prewarm;
        store.set_prewarm_budget(budget);
        eprintln!(
            "  qwen4 ple hot store: prewarm budget {} MiB of a {} MiB pack (MemAvailable - CMA {} MiB)",
            budget.min(store.data_bytes()) / MIB,
            store.data_bytes() / MIB,
            available.map_or(0, |bytes| bytes / MIB),
        );
    } else if let Some((sidecar, data_offset, data_bytes)) = &q4_prewarm {
        // No hot store: the rows come from the q4 sidecar, whose pages a
        // request would otherwise fault in cold.  An idle-only sequential
        // read warms the page cache (clean pages, reclaimable).
        let (budget, available, explicit_zero) = ple_prewarm;
        if budget == 0 {
            eprintln!(
                "  qwen4 ple prewarm: off ({})",
                ple_prewarm_off_reason(available, explicit_zero)
            );
        } else {
            match bundle
                .ple_rows()
                .ok_or_else(|| std::io::Error::other("the bundle has no PLE row store"))
                .and_then(|rows| {
                    hipfire_runtime::ple_prewarm::PleRowsPrewarm::spawn(
                        sidecar,
                        *data_offset,
                        *data_bytes,
                        rows.activity(),
                    )
                }) {
                Ok(prewarm) => {
                    prewarm.set_budget(budget);
                    eprintln!(
                            "  qwen4 ple prewarm: budget {} MiB of a {} MiB q4 sidecar (MemAvailable - CMA {} MiB)",
                            budget.min(*data_bytes) / MIB,
                            data_bytes / MIB,
                            available.map_or(0, |bytes| bytes / MIB),
                        );
                    if let Err(error) = bundle.attach_ple_prewarm(prewarm) {
                        eprintln!("  qwen4 ple prewarm: off ({error})");
                    }
                }
                Err(error) => eprintln!("  qwen4 ple prewarm: off ({error})"),
            }
        }
    }
    match bundle.qsa_context_committed_bytes(ctx.gpu) {
            Ok((target, mtp)) => eprintln!(
                "  qwen4 QSA context: {} storage, {context_state_mib} MiB virtual for {} tokens, {} MiB target + {} MiB MTP committed at load",
                backend.name(),
                ctx.max_seq,
                target >> 20,
                mtp >> 20
            ),
            Err(error) => {
                let detail = error.to_string();
                let _ = bundle.free_gpu(ctx.gpu);
                return Err(format!("qwen4: QSA context census failed: {detail}"));
            }
        }
    if let Some(scope) = band {
        eprintln!(
            "qwen4 PP band layers {:?} (input {}, output {}): loaded on device {} (max_seq {max_seq})",
            scope.band, scope.input, scope.output, ctx.gpu.device_id,
        );
    }
    if let Some(placement) = ep {
        eprintln!(
            "qwen4 EP rank {}/{}: loaded on device {} ({} routed experts of {} per stack, native MTP {}, max_seq {max_seq})",
            placement.rank,
            placement.ranks,
            ctx.gpu.device_id,
            placement.local_count(bundle.config.num_experts),
            bundle.config.num_experts,
            if native_mtp { "on" } else { "off" },
        );
    }
    let card_cap = ctx.sequence.map(|sequence| sequence.card_cap);
    Ok(Qwen4RankLoad {
        bundle,
        native_mtp,
        max_k,
        max_seq,
        card_cap,
    })
}

// ─── Qwen4 pipeline-parallel (topology.pp) load ──────────────────────

/// Largest pipeline degree the Qwen4 mesh executor admits.
const QWEN4_PP_MAX: usize = 8;

/// Structural re-check of the admitted PP mesh at load, before any device is
/// touched (admission already refused anything else; this keeps the carrier
/// from loading against a mesh it was not admitted for): `pp` in 2..=8 with
/// `tp = ep = 1`, one rank per band in logical order, a known PCI identity on
/// every rank, homogeneous gfx1100 or gfx1201, and the daemon's own device
/// (the owner of generation buffers) equal to the output rank.
pub(crate) fn qwen4_pp_mesh_refusal(
    admitted: &hipfire_runtime::mesh_placement::AdmittedMesh,
    placement: &hipfire_runtime::mesh_placement::BandPlacement,
    pp: usize,
    daemon_device: i32,
    daemon_arch: &str,
) -> Result<(), String> {
    use hipfire_runtime::device_mesh::DimKind;
    if !(2..=QWEN4_PP_MAX).contains(&pp) {
        return Err(format!(
            "qwen4: topology.pp = {pp} is outside the admitted 2..={QWEN4_PP_MAX}"
        ));
    }
    let mesh = &admitted.mesh;
    if mesh.size_of(DimKind::Pp) != pp
        || mesh.size_of(DimKind::Tp) != 1
        || mesh.size_of(DimKind::Ep) != 1
        || mesh.n_devices() != pp
        || admitted.ranks.len() != pp
        || placement.stages() != pp
    {
        return Err(format!(
            "qwen4: topology.pp admits pipeline parallelism with tp = ep = 1 only; the admitted \
             mesh, ranks and band placement disagree about pp = {pp}"
        ));
    }
    placement
        .validate_for(mesh, placement.n_trunk_layers)
        .map_err(|error| format!("qwen4: topology.pp band placement: {error}"))?;
    for (index, rank) in admitted.ranks.iter().enumerate() {
        if rank.logical != index {
            return Err(format!(
                "qwen4: topology.pp rank {index} carries logical id {}",
                rank.logical
            ));
        }
        if rank.pci_bus_id.is_none() {
            return Err(format!(
                "qwen4: topology.pp rank {index} (device {}) has no known PCI identity",
                rank.device_ordinal
            ));
        }
        if rank.arch != daemon_arch || !matches!(rank.arch.as_str(), "gfx1100" | "gfx1201") {
            return Err(format!(
                "qwen4: topology.pp admits homogeneous gfx1100 or gfx1201 ranks only; rank \
                 {index} is {} while the output device is {daemon_arch}",
                rank.arch
            ));
        }
    }
    let output = admitted
        .ranks
        .last()
        .ok_or("qwen4: topology.pp mesh has no ranks")?;
    if output.device_ordinal != daemon_device {
        return Err(format!(
            "qwen4: topology.pp output rank is device {} but the daemon's generation device is \
             {daemon_device}; generation buffers must live on the output card",
            output.device_ordinal
        ));
    }
    Ok(())
}

/// Bring up one `Gpu` per admitted rank without enabling peer access:
/// `Gpus::init_layers` constructs the devices (arch and bind preflight, no
/// VRAM-delta gate) in the process's device order; they are then permuted to
/// the admitted logical rank order (output card last) and the retained mesh
/// is replaced by the admitted one, so `devices[r]` is exactly logical rank
/// `r`. A device the admitted ranks do not name refuses before any model
/// allocation.
fn init_qwen4_pp_gpus(
    admitted: &hipfire_runtime::mesh_placement::AdmittedMesh,
    placement: &hipfire_runtime::mesh_placement::BandPlacement,
) -> Result<hipfire_runtime::multi_gpu::Gpus, String> {
    // Read-only structural validation before any `Gpu` is constructed.
    let ordinals: std::collections::BTreeSet<i32> = admitted
        .ranks
        .iter()
        .map(|rank| rank.device_ordinal)
        .collect();
    if admitted.ranks.len() != placement.stages()
        || ordinals.len() != admitted.ranks.len()
        || ordinals.iter().any(|ordinal| *ordinal < 0)
    {
        return Err(
            "qwen4: topology.pp admitted ranks must be one distinct, non-negative device per band"
                .into(),
        );
    }
    let counts: Vec<usize> = (0..placement.stages())
        .map(|stage| placement.band(stage).len())
        .collect();
    let mut gpus = hipfire_runtime::multi_gpu::Gpus::init_layers(&counts)
        .map_err(|error| format!("qwen4: topology.pp device initialization failed: {error}"))?;
    let mut pool: Vec<Option<rdna_compute::Gpu>> = std::mem::take(&mut gpus.devices)
        .into_iter()
        .map(Some)
        .collect();
    let mut ordered = Vec::with_capacity(pool.len());
    for rank in &admitted.ranks {
        let slot = pool.iter_mut().find(|slot| {
            slot.as_ref()
                .is_some_and(|device| device.device_id == rank.device_ordinal)
        });
        match slot.and_then(Option::take) {
            Some(device) => ordered.push(device),
            None => {
                return Err(format!(
                    "qwen4: topology.pp rank {} names device {}, which the process did not open \
                     for {} pipeline ranks",
                    rank.logical,
                    rank.device_ordinal,
                    admitted.ranks.len()
                ));
            }
        }
    }
    gpus.devices = ordered;
    gpus.mesh = admitted.mesh.clone();
    // The mesh executor enqueues every band's work and every edge publish /
    // receive on the device's own launch stream (never the legacy null
    // stream); install it before any band allocation so loads and forwards
    // share one ordered stream per device.
    for (rank, gpu) in gpus.devices.iter_mut().enumerate() {
        gpu.ensure_capture_stream().map_err(|error| {
            format!("qwen4: topology.pp rank {rank} launch stream creation failed: {error}")
        })?;
    }
    Ok(gpus)
}

/// Unwind a failed PP load: drain the edge leases first (nothing was
/// submitted, so this only retires the transport), then free every loaded
/// band on its own device. A failed drain leaks the bands and the devices
/// (quarantine): nothing is freed under possibly in-flight work.
fn abort_qwen4_pp_load(
    mut gpus: hipfire_runtime::multi_gpu::Gpus,
    bands: Vec<hipfire_arch_qwen4::bundle::Qwen4Bundle>,
    error: String,
) -> String {
    if let Err(drain) = gpus.drain_mesh_edges() {
        std::mem::forget(bands);
        std::mem::forget(gpus);
        return format!(
            "{error}; mesh edge drain failed ({drain}): resources quarantined until process restart"
        );
    }
    let mut cleanup = Vec::new();
    for (rank, band) in bands.into_iter().enumerate() {
        if let Err(free) = band.free_gpu(&mut gpus.devices[rank]) {
            cleanup.push(format!("rank {rank}: {free}"));
        }
    }
    if cleanup.is_empty() {
        error
    } else {
        format!("{error}; cleanup also failed: {}", cleanup.join("; "))
    }
}

/// Load every band of one Qwen4 PP model and return the root (output-band)
/// bundle, which owns the generic `Gpus`, the other band bundles and the
/// placement ([`Qwen4Bundle::attach_pp_mesh`]); the daemon's `gpu` is the
/// output card and keeps only generation-owned buffers.
///
/// Order: structural mesh re-check, device bring-up (no peer enable), the
/// per-rank capacity census against each rank's free VRAM (refuses naming
/// `topology.pp` before any allocation), edge provisioning sized to the
/// largest canonical chunk's HC streams, then one band-scoped load per rank
/// (stage 0 on the admitted source, the rest on a fresh source each).
fn load_qwen4_pp_bundle(
    daemon: &mut rdna_compute::Gpu,
    admitted: &hipfire_runtime::mesh_placement::AdmittedMesh,
    placement: &hipfire_runtime::mesh_placement::BandPlacement,
    shared: &Qwen4RankShared,
    session: Option<Qwen4SessionSeed<'_>>,
    src: ModelSource,
) -> Result<
    (
        hipfire_arch_qwen4::bundle::Qwen4Bundle,
        hipfire_runtime::multi_gpu::Gpus,
        bool,
    ),
    String,
> {
    use hipfire_arch_qwen4 as q4;
    let stages = placement.stages();
    qwen4_pp_mesh_refusal(admitted, placement, stages, daemon.device_id, &daemon.arch)?;
    if daemon.is_uma() {
        return Err("qwen4: topology.pp needs discrete GPUs".into());
    }
    if shared.sequence.is_some() {
        return Err("qwen4: topology.pp admits no automatic context".into());
    }
    // Per-rank free VRAM from the daemon's runtime (no `Gpus`, no allocation):
    // every artifact and capacity refusal below happens before any device
    // beyond the daemon's own is brought up.
    daemon
        .emu_refuse_multi_device("qwen4 topology.pp")
        .map_err(|error| format!("qwen4: {error}"))?;
    let mut free_per_rank = Vec::with_capacity(stages);
    for rank in &admitted.ranks {
        let (free, _) = daemon
            .device_mem_info_for(rank.device_ordinal)
            .map_err(|error| {
                format!(
                    "qwen4: topology.pp rank {} (device {}) memory query: {error}",
                    rank.logical, rank.device_ordinal
                )
            })?;
        free_per_rank.push(free as u64);
    }
    let (scopes, edge_bytes, chunk_rows) = {
        let ModelSource::Hfq(hfq) = &src else {
            return Err(
                "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
            );
        };
        let receipt = q4::admit_hfqm_artifact_on(hfq, Some(&daemon.arch))
            .map_err(|error| error.to_string())?;
        q4::bundle::admit_executable_profile(&receipt.config)
            .map_err(|error| format!("qwen4: {error}"))?;
        let config = receipt.config.clone();
        let scopes = (0..stages)
            .map(|stage| q4::weights::Qwen4BandScope::from_placement(&config, placement, stage))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("qwen4: topology.pp: {error}"))?;
        let state_format = q4::resolve_state_format(
            &shared.kv_mode,
            shared.state_quant.as_deref().unwrap_or(""),
            daemon,
            &config,
        )?;
        let scratch_mode =
            q4::artifact::source_forward_scratch_mode(daemon, hfq, &config, &receipt.manifest)
                .map_err(|error| error.to_string())?;
        let chunk_rows =
            q4::gpu_forward::qwen4_prefill_chunk_requested(&daemon.arch, shared.max_seq);
        let context = q4::Qwen4ContextCommit::new(
            shared.backend,
            shared.max_seq,
            shared.max_seq,
            shared.vmm_granularity,
        );
        // The gathered QSA prefill attention's workspace is committed with
        // the context; every rank charges it (the headroom is the rest).
        let gather = rdna_compute::tensor_ops::qsa_gather_workspace_needed(
            daemon,
            state_format.qsa,
            state_format.qsa_f16_arena,
        )
        .then(|| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                context.tokens,
            )
            // Every rank is a fresh model `Gpu` that owns no daemon scratch:
            // charge the full gathered workspace per rank.
            .map(|bytes| bytes as u64)
            .ok_or("qwen4: QSA gather scratch size overflows".to_string())
        })
        .transpose()?
        .unwrap_or(0);
        // The per-band session cache (`memory.session_cache_bytes` on every
        // band, no disk tier) is deliberately not charged or reserved here,
        // exactly as in the Single census (`qwen4_memory_admission`): it is
        // opportunistic. Each cache lives on its own band's `Gpu`, so the
        // bands' VRAM is independent and one band's snapshots never take
        // another's room; the existing headroom (`reserve_bytes`) and the
        // growth guards (snapshots yield to context growth) bound the
        // allocation, so admission never depends on the cache budget.
        let reserve_bytes = crate::admission::QWEN4_MEMORY_HEADROOM_BYTES
            .checked_add(gather)
            .ok_or("qwen4: topology.pp reserve overflows")?;
        let recipe = receipt.recipe().clone();
        let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
            crate::qwen4_slab::qwen4_entry_bytes(&recipe, entry, |entry| {
                hfq.tensor_data(&entry.name)
                    .map(|(_, bytes)| bytes.len() as u64)
            })
        };
        // The native MTP reserves (the output band's head and request
        // buffers, every band's own row capture) exactly when the head
        // attaches.
        let native_mtp = crate::admission::qwen4_native_mtp_for_head(
            shared.native_mtp,
            shared.spec.mtp,
            config.mtp.is_some(),
            config.profile.id(),
        )?;
        let native_mtp_inputs = native_mtp.then(|| q4::expert_residency::NativeMtpCensusInputs {
            max_k: shared.max_k,
            row_capture: q4::mtp_spec::native_mtp_capture(
                daemon,
                &config,
                state_format.gdn,
                shared.max_k,
            ),
            batched_fill: q4::mtp_spec::native_mtp_fill_scratch(daemon),
        });
        let fits =
            crate::admission::qwen4_pp_band_budget(&crate::admission::Qwen4PpBudgetInputs {
                config: &config,
                placement,
                weights: &receipt.manifest.weights,
                bytes_of: &bytes_of,
                format: state_format,
                context: &context,
                chunk_rows,
                mode: scratch_mode,
                free_per_rank: &free_per_rank,
                reserve_bytes,
                native_mtp: native_mtp_inputs,
            })?;
        for fit in &fits {
            eprintln!(
                "  qwen4 PP rank {}: layers {:?} need {} MiB of {} MiB free{}",
                fit.stage,
                fit.band,
                fit.required >> 20,
                fit.free >> 20,
                if native_mtp {
                    format!(" (incl. {} MiB native MTP)", fit.native_mtp >> 20)
                } else {
                    String::new()
                }
            );
        }
        let edge_bytes = crate::admission::qwen4_pp_edge_max_bytes(&config, chunk_rows)?;
        (scopes, edge_bytes, chunk_rows)
    };
    let mut gpus = init_qwen4_pp_gpus(admitted, placement)?;
    gpus.provision_mesh_edges(admitted, placement, edge_bytes)
        .map_err(|error| format!("qwen4: topology.pp edge provisioning failed: {error}"))?;
    let mut bands: Vec<q4::bundle::Qwen4Bundle> = Vec::with_capacity(stages);
    let mut output_native_mtp = false;
    let mut admitted_source = Some(src);
    for stage in 0..stages {
        let source = match admitted_source.take() {
            Some(source) => Ok(source),
            None => open_qwen4_rank_source(shared, &gpus.devices[stage].arch),
        };
        let loaded = source.and_then(|source| {
            load_qwen4_rank_scoped(
                &mut gpus.devices[stage],
                None,
                shared,
                None,
                session,
                source,
                Some(&scopes[stage]),
            )
        });
        match loaded {
            Ok(load) => {
                output_native_mtp = load.native_mtp;
                bands.push(load.bundle);
            }
            Err(error) => {
                return Err(abort_qwen4_pp_load(
                    gpus,
                    bands,
                    format!("qwen4: topology.pp band {stage} of {stages} failed to load: {error}"),
                ));
            }
        }
    }
    eprintln!(
        "qwen4 PP plan: {stages} bands {:?} on devices {:?}, edge payload up to {} MiB ({chunk_rows} canonical rows)",
        (0..stages).map(|stage| placement.band(stage)).collect::<Vec<_>>(),
        admitted.ranks.iter().map(|rank| rank.device_ordinal).collect::<Vec<_>>(),
        edge_bytes >> 20,
    );
    let mut root = bands.pop().expect("a PP load has at least two bands");
    match root.attach_pp_mesh(&mut gpus, bands, placement.clone()) {
        Ok(()) => Ok((root, gpus, output_native_mtp)),
        Err(error) => {
            // The holder is installed before validation, so the root's own
            // teardown drains the epochs and edges and frees every band. A
            // failed drain quarantines: the root and bands were leaked by the
            // bundle, and the devices are leaked here.
            let detail = error.to_string();
            Err(match root.pp_free_gpu(&mut gpus) {
                Ok(()) => format!("qwen4: topology.pp mesh attach failed: {detail}"),
                Err(free) => {
                    std::mem::forget(gpus);
                    format!(
                        "qwen4: topology.pp mesh attach failed: {detail}; teardown also failed \
                         ({free}): resources quarantined until process restart"
                    )
                }
            })
        }
    }
}

/// The `Qwen4Carrier` load of a `topology.pp` 2..=8 model: the same
/// pre-allocation option and device refusals as the Single load, then the
/// mesh-executor bundle. `LoadedModel.pp` is the admitted degree and
/// `pp_gpus` holds the generic `Gpus` (the sole hardware owner, `!Send`, so
/// it cannot live in the `Send` bundle); `pp_dn_la_to_device` stays `None`.
fn load_qwen4_pp_model(src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
    let pp = ctx.pp;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::from(ctx.kv_backend);
    if backend == hipfire_arch_qwen4::Qwen4KvBackend::Vmm {
        if let Some(reason) = hipfire_arch_qwen4::qwen4_vmm_refusal(
            &ctx.gpu.arch,
            ctx.gpu.vmm_recommended_granularity().is_ok(),
        ) {
            return Err(format!(
                "qwen4: KV backend 'vmm' is not available: {reason}; use --kv-backend legacy (memory.kv_backend = legacy)"
            ));
        }
    }
    let vmm_granularity = match backend {
        hipfire_arch_qwen4::Qwen4KvBackend::Vmm => ctx
            .gpu
            .vmm_recommended_granularity()
            .map_err(|error| format!("qwen4: VMM granularity: {error}"))?,
        hipfire_arch_qwen4::Qwen4KvBackend::Legacy => 1,
    };
    if ctx.max_seq == 0 || ctx.max_seq > hipfire_arch_qwen4::QWEN4_MAX_CONTEXT {
        return Err(format!(
            "qwen4: max_seq {} is outside the admitted context range 1..={}",
            ctx.max_seq,
            hipfire_arch_qwen4::QWEN4_MAX_CONTEXT
        ));
    }
    let ep = hipfire_config::topology::qwen4_ep()?;
    if ep > 1 {
        return Err(format!(
            "qwen4: topology.pp = {pp} cannot combine with expert parallelism (topology.ep = {ep})"
        ));
    }
    let options = crate::admission::SourceAdmissionOptions {
        spec: ctx.spec,
        kv_adaptive: crate::admission::qwen4_kv_adaptive_requested(ctx.kv_adaptive_override),
        eagle_drafter: ctx.gemma4_drafter_path.is_some(),
        cask: ctx.cask.sidecar.is_some(),
        state_quant: ctx.state_quant_override.is_some(),
        non_single_compute: !matches!(
            &ctx.deepseek4_compute_placement,
            hipfire_config::Deepseek4ComputePlacement::Single
        ),
        expert_count_override: ctx.deepseek4_experts_per_token.is_some(),
        pflash: false,
    };
    crate::admission::qwen4_pp_option_refusal(pp, ctx.draft_path, &options)?;
    Qwen4Carrier.admit_options(ctx.draft_path, options)?;
    let (Some(admitted), Some(placement)) = (ctx.admitted_mesh, ctx.band_placement) else {
        return Err(format!(
            "qwen4: topology.pp = {pp} needs the admitted mesh and band placement from source \
             admission"
        ));
    };
    if ctx.sequence.is_some() {
        return Err(
            "qwen4: topology.pp has no automatic context sizing; set memory.max_seq".into(),
        );
    }
    let native_mtp =
        crate::admission::qwen4_native_mtp(ctx.spec, &ctx.gpu.arch, ctx.path, pp, 1, ep)?;
    crate::admission::qwen4_pp_mtp_ngram_refusal(
        pp,
        native_mtp,
        hipfire_config::mtp_ngram_enabled(),
    )?;
    let max_k = qwen4_native_mtp_k(ctx.spec, &ctx.gpu.arch);
    let meta = resolve_source_meta(&src, ctx.path)?;
    let shared = Qwen4RankShared {
        path: ctx.path.to_string(),
        max_seq: ctx.max_seq,
        sequence: None,
        backend,
        vmm_granularity,
        kv_mode: kv_mode_from_ctx(ctx),
        state_quant: ctx.state_quant_override.map(str::to_string),
        spec: ctx.spec,
        native_mtp,
        max_k,
    };
    // Every band attaches its own session cache from the source model's
    // tokenizer and template (the same seed the Single load takes).
    let session = Some(Qwen4SessionSeed {
        tokenizer: &meta.tokenizer,
        chat_template: meta.chat_template.as_deref(),
    });
    let (bundle, gpus, native_mtp) =
        load_qwen4_pp_bundle(ctx.gpu, admitted, placement, &shared, session, src)?;
    let max_seq = ctx.max_seq;
    let speculator = native_mtp.then(|| {
        qwen4_native_mtp_speculator(
            max_k,
            max_seq,
            &ctx.gpu.arch,
            meta.tokenizer.special_token_id("<|im_end|>"),
            meta.tokenizer.special_token_id("</think>"),
            false,
            rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&ctx.gpu.arch),
        )
    });
    let mut model = LoadedModel {
        state: Some(Box::new(bundle)),
        speculator,
        pp,
        pp_gpus: Some(gpus),
        ..LoadedModel::skeleton(
            meta.arch_id,
            meta.tokenizer,
            max_seq,
            max_seq,
            ctx.path.to_string(),
            meta.chat_template,
            meta.chat_template_source,
        )
    };
    apply_qwen4_native_mtp(&mut model, native_mtp, max_k, ctx.spec);
    Ok(model)
}

// ─── Qwen35Carrier ───────────────────────────────────────────────────

fn kv_mode_from_ctx(ctx: &LoadCtx) -> String {
    ctx.kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone())
}

/// Resolve Qwen K/V via the shared name table. Consumes `LoadCtx` mode + axis
/// overrides + `qwen_default_q8`; no local Qwen alias maps.
fn resolve_qwen_kv_pair(
    ctx: &LoadCtx,
    policy: &hipfire_runtime::kv_mode::KvModePolicy,
) -> Result<hipfire_runtime::kv_mode::KvPair, String> {
    use hipfire_runtime::kv_mode::{self, KvPair};
    let mode_raw = kv_mode_from_ctx(ctx);
    let pair = kv_mode::resolve_kv_pair(
        &mode_raw,
        ctx.kv_k_override,
        ctx.kv_v_override,
        policy,
        ctx.gpu.arch.as_str(),
        ctx.qwen_default_q8,
    )
    .map_err(|e| e.to_string())?;
    match pair {
        KvPair::Native(k) => eprintln!(
            "  KV cache: requested mode={mode_raw}, effective KV={} (site {})",
            kv_mode::qwen_k_display_name(k),
            policy.site
        ),
        KvPair::Split(k, v) => {
            let k_name = kv_mode::qwen_k_display_name(k);
            let v_name = kv_mode::qwen_v_display_name(v);
            eprintln!(
                "  KV cache: requested mode={mode_raw}, effective K={k_name} V={v_name} (site {})",
                policy.site
            );
            eprintln!("  K={k_name} V={v_name}");
        }
    }
    Ok(pair)
}

fn arch_default_template(arch_id: u32) -> Option<String> {
    match arch_id {
        5 | 6 => Some(super::FROGGERIC_QWEN35_TEMPLATE.to_string()),
        11 => Some(super::LFM2_TEMPLATE.to_string()),
        13 => Some(super::GEMMA4_TEMPLATE.to_string()),
        _ => None,
    }
}

/// Qwen3.5 pipeline-parallel (pp>1) load. Extracted from the carrier body so
/// the pp>1 multi-GPU tail (`skeleton_pp`) lives in one place; qwen35 is the
/// only carrier with a pp>1 path. KV policy (`QWEN35_PP_POLICY`), DeltaNet
/// quant, and scratch sizing are byte-identical to the previous inline block.
fn load_qwen35_pp(
    mut hfq_file: hipfire_runtime::hfq::HfqFile,
    meta: SourceMeta,
    ctx: &mut LoadCtx,
) -> Result<LoadedModel, String> {
    let pp = ctx.pp;
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq_file)
        .map_err(|e| format!("failed to read Qwen3.5 config: {e}"))?;
    // Band counts were normalized once at admission (`BandPlacement`, sum and
    // length validated against this config's layer count); explicit counts
    // skip the uniform free-VRAM delta gate exactly as the old
    // `HIPFIRE_PP_LAYERS` escape hatch did. The environment is not read here.
    let mut gpus = match ctx.pp_layer_counts {
        Some(counts) => {
            debug_assert_eq!(counts.len(), pp);
            debug_assert_eq!(counts.iter().sum::<usize>(), config.n_layers);
            hipfire_runtime::multi_gpu::Gpus::init_layers(counts).map_err(|e| format!("{e}"))?
        }
        None => hipfire_runtime::multi_gpu::Gpus::init_uniform(pp, config.n_layers)
            .map_err(|e| format!("{e}"))?,
    };
    // Discrete GPUs: keep model pages in the page cache across reads —
    // fadvise(DONTNEED)-per-tensor forces a full disk re-read on every load.
    // UMA keeps eviction (default) to avoid OOM vs hipMalloc staging.
    hfq_file.set_evict_page_cache(
        hipfire_config::developer_var("HIPFIRE_PAGE_EVICTION")
            .ok()
            .map(|v| v != "0")
            .unwrap_or_else(|| gpus.devices.iter().any(|g| g.is_uma())),
    );
    let _hfq_cache_warmer = hfq_file.start_cache_warmup();
    let layout = hipfire_arch_qwen35::qwen35::Layout::from_gpus(&gpus, config.n_layers);
    let mut hfq_source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq_file, &config);
    let weights =
        hipfire_arch_qwen35::qwen35::load_weights(&mut hfq_source, &mut gpus.devices, &layout)
            .map_err(|e| format!("{e}"))?;
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
        .collect();
    let pair = resolve_qwen_kv_pair(ctx, &hipfire_runtime::kv_mode::QWEN35_PP_POLICY)?;
    let mode = pair.k();
    if pair
        .v()
        .is_some_and(|v| v != hipfire_runtime::llama::VMode::Q8)
    {
        return Err(format!(
            "qwen35: --kv-v {} is single-GPU only; with pipeline parallelism (topology.pp > 1) use --kv-v q8 (site {})",
            hipfire_runtime::kv_mode::qwen_v_display_name(pair.v().expect("PP split pair")),
            hipfire_runtime::kv_mode::QWEN35_PP_POLICY.site
        ));
    }
    let dims = hipfire_runtime::llama::KvDims {
        layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(ctx.max_seq),
    };
    let kv = <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode(
        mode,
        hipfire_runtime::llama::KvTarget::Multi(&mut gpus),
        &dims,
    )
    .map_err(|e| format!("{e}"))?;
    let dn_quant =
        crate::parse_state_quant(ctx.state_quant_override).map_err(|e| format!("{e}"))?;
    let (dn, la_to_device) = hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant_multi(
        &mut gpus, &config, dn_quant,
    )
    .map_err(|e| format!("{e}"))?;
    let scratch_set = hipfire_arch_qwen35::qwen35::Qwen35ScratchSet::new_with_kv_max_multi(
        &mut gpus,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let gpu0 = &mut gpus.devices[0];
    let single_scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
        gpu0,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let bundle = hipfire_arch_qwen35::Qwen35Bundle {
        config,
        weights,
        scratch: single_scratch,
        kv_cache: kv,
        dn_state: dn,
        // Adaptive is single-GPU only; PP path never engages the controller.
        kv_adaptive: None,
        pp_scratch_set: Some(scratch_set),
        vision_config: None,
        vision_weights: None,
        qwen35_decode_batch: None,
        vmm_store: None,
        weight_lease: None,
    };
    Ok(LoadedModel {
        state: Some(Box::new(bundle)),
        ..LoadedModel::skeleton_pp(
            meta.arch_id,
            meta.tokenizer,
            ctx.max_seq,
            ctx.max_seq,
            ctx.path.to_string(),
            meta.chat_template,
            meta.chat_template_source,
            pp,
            gpus,
            la_to_device,
        )
    })
}

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str {
        "qwen35"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        // qwen35 moves its bundle out of `state` into the RAII Qwen35SlotGuard
        // (lazy HfqFile reopen, bundle restored on Drop — the #462 guard).
        Ok(Box::new(Qwen35SlotGuard::take(state, model_path)?))
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 5 = dense (+VL), 6 = MoE — same ids in both namespaces.
        matches!(arch_id, 5 | 6)
    }
    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        // Qwen3.5 is the only carrier with a pp>1 path (load_qwen35_pp), HFQ
        // only. Dir + pp>1 is refused; VMM + pp>1 is refused globally upstream.
        if pp > 1 && is_dir {
            return Err(crate::pp_unsupported_message("qwen35", true, pp));
        }
        Ok(())
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: true,
            supports_ep_batch: true,
            dflash: Some(saddle_core::caps::DflashKind::Qwen),
            supports_mtp: true,
            spec_excludes_adaptive: true,
            semantic_contract_version: Some(2),
            has_deltanet: true,
            // Architectural capability: Qwen3.5-VL tower is optional per-model
            // (probed via `model.visual.patch_embed.proj.weight`); this flag
            // declares that the arch CAN accept images when that tower is
            // present. Per-instance gating still checks `LoadedModel::vision_config`.
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::QwenJinja,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen35_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv_cache;
        let dn = &mut b.dn_state;
        match hipfire_arch_qwen35::qwen35::forward_prefill_batch(
            gpu, weights, config, synthetic, 0, kv, dn, scratch, None, None, None, None,
        ) {
            Ok(()) => Some(true),
            Err(e) => {
                *prefill_err = Some(format!("{e:?}"));
                Some(false)
            }
        }
    }
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let b = m.qwen35_mut().unwrap();
        Some(
            hipfire_arch_qwen35::qwen35::forward_prefill_batch(
                gpu,
                &b.weights,
                &b.config,
                synthetic,
                0,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
                None,
                None,
                None,
                None,
            )
            .err()
            .map(|e| format!("{e:?}")),
        )
    }
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        _decode_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.qwen35_mut().unwrap();
        let mut ok = true;
        for i in 0..iterations {
            let token = 101 + (i as u32 % 1000);
            if hipfire_arch_qwen35::qwen35::forward_scratch(
                gpu,
                &b.weights,
                &b.config,
                token,
                context + i,
                &mut b.kv_cache,
                &mut b.dn_state,
                &b.scratch,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.kv_backend == KvBackend::Vmm && ctx.pp > 1 {
            return Err(
                "qwen35: KV backend 'vmm' currently requires pp=1; use 'legacy' for pipeline parallelism"
                    .into(),
            );
        }
        // Dir + pp>1: early return before any diagnostics/meta resolution,
        // preserving the original error string and preventing tokenizer work.
        if ctx.pp > 1 {
            if let ModelSource::Dir(..) = &src {
                return Err(crate::pp_unsupported_message("qwen35", true, ctx.pp));
            }
        }
        // Per-source diagnostics stay at the call site, before resolve_source_meta.
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        match src {
            ModelSource::Hfq(mut hfq_file) => {
                // ── pp>1 path (pipeline-parallel) — extracted helper ──
                if ctx.pp > 1 {
                    return load_qwen35_pp(hfq_file, meta, ctx);
                }
                // Discrete GPUs: keep model pages in the page cache across
                // reads — fadvise(DONTNEED)-per-tensor forces a full disk
                // re-read on every load. UMA keeps eviction (default) to
                // avoid OOM vs hipMalloc staging.
                hfq_file.set_evict_page_cache(
                    hipfire_config::developer_var("HIPFIRE_PAGE_EVICTION")
                        .ok()
                        .map(|v| v != "0")
                        .unwrap_or_else(|| ctx.gpu.is_uma()),
                );
                let _hfq_cache_warmer = hfq_file.start_cache_warmup();

                // ── pp=1 path (single-GPU) ────────────────────

                // VL detection — tower loads from the trunk in-place, or from
                // the shared sidecar when the trunk is tower-less. The sidecar
                // opens as a SEPARATE HfqFile (never attach_overlay: REAP
                // rejects additive tensor names); the tower always sizes from
                // the trunk's vision_config_from_hfq, falling back to the
                // sidecar's metadata (text-only trunks predate the embedded
                // blob). Admission already refused tower-less / wrong-arch
                // sidecars, so a sidecar failure here fails the load closed —
                // an explicitly requested tower must never silently serve text.
                let (vision_config, vision_weights) = {
                    use hipfire_arch_qwen35_vl::Qwen35Vl;
                    use hipfire_runtime::arch::Architecture;

                    // Vision tower sidecar resolution: an explicit
                    // `--vision-path` (or HIPFIRE_VL_FILE) must never be
                    // shadowed by a co-located `<stem>.vl` sibling — the
                    // operator's tower wins, and sibling discovery is only
                    // the fallback.
                    let vl_path = ctx
                        .vision_path
                        .as_ref()
                        .map(std::path::PathBuf::from)
                        // `vision_mode=off` suppresses sibling discovery too —
                        // a co-located `<stem>.vl` must not load the tower
                        // under the documented text-only default (the daemon
                        // gates the explicit sidecar; this gates the probe).
                        .or_else(|| {
                            if ctx.vision_mode == "off" {
                                None
                            } else {
                                discover_vl_path(ctx.path)
                            }
                        });
                    let has_inline_vision = hfq_file
                        .tensor_data("model.visual.patch_embed.proj.weight")
                        .is_some();

                    if let Some(vl) = &vl_path {
                        eprintln!(
                            "  loading vision weights from .vl sidecar: {}",
                            vl.display()
                        );
                        let mut vl_hfq = hipfire_runtime::hfq::HfqFile::open(vl)
                            .map_err(|e| format!("open .vl file {}: {e}", vl.display()))?;
                        let vc = Qwen35Vl::config_from_hfq(&vl_hfq)
                            .map_err(|e| format!(".vl vision_config: {e}"))?;
                        // Identity: the tower's projector writes the trunk's
                        // text hidden width. A sidecar paired with a different
                        // trunk (a sibling `foo.vl` next to `bar.mq4`) would
                        // load "successfully" and then misalign every image
                        // embedding — refuse the mismatch at load time.
                        // Probe the trunk's final norm under its real HFQ
                        // names (`output_norm.weight` is GGUF-only and never
                        // resolves here, which would skip the check entirely).
                        let trunk_dim = [
                            "model.language_model.norm.weight",
                            "model.norm.weight",
                            "norm.weight",
                        ]
                        .iter()
                        .find_map(|n| hfq_file.find_tensor_info(n))
                        .and_then(|t| t.shape.first().copied())
                        .map(|d| d as usize);
                        if let Some(trunk_dim) = trunk_dim {
                            if vc.out_hidden_size != trunk_dim {
                                return Err(format!(
                                    "vision sidecar {} does not belong to this model {}: its output width {} \
                                     does not match the model's hidden size {}. Use the sidecar made for this model",
                                    vl.display(),
                                    ctx.path,
                                    vc.out_hidden_size,
                                    trunk_dim
                                ));
                            }
                        }

                        // Fail the load, loudly: swallowing the error here
                        // used to yield (Some(config), None) — a model whose
                        // daemon-side image gate stays open (has_vision keys
                        // off the config) but whose vision weights are gone.
                        // The sequential path then panics on the unwrap and
                        // the slots path rejects image requests with a
                        // misleading "no vision encoder" message. A corrupt
                        // or truncated .vl must be a load-time error the
                        // operator can see.
                        let vw = Qwen35Vl::load_weights(&mut vl_hfq, &vc, ctx.gpu)
                            .map_err(|e| format!("VL weight load from .vl: {e:?}"))?;
                        eprintln!(
                            "  VL model: vision encoder (hidden={}, layers={})",
                            vc.hidden_size, vc.num_layers
                        );
                        (Some(vc), Some(vw))
                    } else if has_inline_vision {
                        let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                        match vc {
                            Some(vc) => {
                                let vw = Qwen35Vl::load_weights(&mut hfq_file, &vc, ctx.gpu)
                                    .map_err(|e| eprintln!("  VL weight load failed: {e}"))
                                    .ok();
                                eprintln!(
                                    "  VL model: vision encoder (hidden={}, layers={})",
                                    vc.hidden_size, vc.num_layers
                                );
                                (Some(vc), vw)
                            }
                            _ => (None, None),
                        }
                    } else if let Some(p) = ctx.vision_path.as_ref() {
                        let mut sidecar =
                            hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(p)).map_err(
                                |e| format!("vision sidecar '{}': open failed: {e}", p.display()),
                            )?;
                        let vc = Qwen35Vl::config_from_hfq(&hfq_file)
                            .ok()
                            .or_else(|| Qwen35Vl::config_from_hfq(&sidecar).ok())
                            .ok_or_else(|| {
                                "qwen35-vl: vision tower requested but no vision_config in \
                                 trunk or sidecar metadata — requantize the trunk or pack \
                                 the sidecar with --include-vision"
                                    .to_string()
                            })?;
                        let vw =
                            Qwen35Vl::load_weights(&mut sidecar, &vc, ctx.gpu).map_err(|e| {
                                format!("vision sidecar '{}': tower load failed: {e}", p.display())
                            })?;
                        eprintln!(
                            "  VL model (sidecar {}): vision encoder (hidden={}, layers={})",
                            p.display(),
                            vc.hidden_size,
                            vc.num_layers
                        );
                        (Some(vc), Some(vw))
                    } else {
                        (None, None)
                    }
                };

                // Trunk bundle after optional VL upload. On bundle failure, reclaim
                // any vision weights already on-device (HFQ is single-pass: VL must
                // load from the same file before the carrier consumes it).
                let bundle = match hipfire_arch_qwen35::load_qwen35_bundle(
                    ModelSource::Hfq(hfq_file),
                    ctx,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        if let Some(vw) = vision_weights {
                            vw.free_gpu(ctx.gpu);
                        }
                        return Err(e);
                    }
                };
                let physical_cap = bundle.kv_cache.physical_cap;
                finish_qwen35_load(
                    bundle,
                    meta.tokenizer,
                    physical_cap,
                    meta.arch_id,
                    meta.chat_template,
                    meta.chat_template_source,
                    ctx,
                    vision_config,
                    vision_weights,
                )
            }
            ModelSource::Dir(source) => {
                let config = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source)
                    .map_err(|e| format!("failed to parse Qwen3.5 config from config.json: {e}"))?;
                if ctx.draft_path.is_some() {
                    eprintln!("  warning: DFlash (speculative decoding) is not supported for safetensors Dir sources; draft_path ignored");
                }
                if ctx.cask.sidecar.is_some() {
                    eprintln!("  warning: CASK eviction is not supported for safetensors Dir sources; eviction sidecar ignored");
                }
                // CPU-only before any GPU ownership (parity with HFQ carrier).
                let dn_quant = crate::parse_state_quant(ctx.state_quant_override)
                    .map_err(|e| format!("{e}"))?;
                eprintln!(
                    "  DeltaNet state quant: {}",
                    if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::FP32 {
                        "FP32"
                    } else if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::Q4 {
                        "Q4"
                    } else {
                        "Q8"
                    }
                );
                if config.dim < 2048 && dn_quant != hipfire_arch_qwen35::qwen35::StateQuant::FP32 {
                    eprintln!(
                        "  warning: model dim={} (<2048); FP32 DeltaNet state is recommended for small models (current: {})",
                        config.dim,
                        if dn_quant == hipfire_arch_qwen35::qwen35::StateQuant::Q4 {
                            "Q4"
                        } else {
                            "Q8"
                        }
                    );
                }
                let is_kv_layer: Vec<bool> = config
                    .layer_types
                    .iter()
                    .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
                    .collect();
                let native_eligible = hipfire_runtime::kv_mode::qwen35_native_eligible(
                    ctx.gpu.arch.as_str(),
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    ctx.pp,
                    false,
                    ctx.cask.sidecar.is_some(),
                );
                let policy = hipfire_runtime::kv_mode::qwen35_policy_for_native(
                    &hipfire_runtime::kv_mode::QWEN35_PARO_POLICY,
                    &kv_mode_from_ctx(ctx),
                    native_eligible,
                );
                let pair = resolve_qwen_kv_pair(ctx, &policy)?;
                let mode = pair.k();
                // Same native-tier admission as the HFQ carrier site: exact
                // gfx1201/dense/legacy-or-VMM. The Dir path still honors
                // resolved VMode below; adaptive/CASK stay HFQ-only.
                hipfire_arch_qwen35::carrier::validate_native_kv_admission(
                    mode,
                    &config,
                    ctx,
                    false,
                    ctx.kv_v_override.filter(|s| !s.is_empty()).is_some(),
                )?;
                let dims = hipfire_runtime::llama::KvDims {
                    layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer.clone()),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: ctx.max_seq,
                    physical_cap: Some(ctx.max_seq),
                };

                let mut paro_source =
                    hipfire_arch_qwen35::qwen35::ParoSource::new(&source, &config)
                        .map_err(|e| format!("ParoSource::new: {e:?}"))?;
                let paro_layout = hipfire_arch_qwen35::qwen35::Layout::single(config.n_layers);
                let weights = hipfire_arch_qwen35::qwen35::load_weights(
                    &mut paro_source,
                    std::slice::from_mut(ctx.gpu),
                    &paro_layout,
                )
                .map_err(|e| format!("load_weights: {e:?}"))?;
                hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);

                // Staged GPU free on every post-weight error (VMM arenas via free_gpu).
                // Propagate resolved VMode: VMM takes it in-constructor; legacy
                // Q8 uses from_mode; legacy Lloyd-V post-alloc reallocs.
                let kv_cache = match (ctx.kv_backend, pair) {
                    (KvBackend::Vmm, hipfire_runtime::kv_mode::KvPair::Native(_)) => {
                        match <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode_with_backend(
                            mode, KvBackend::Vmm, hipfire_runtime::llama::KvTarget::Single(ctx.gpu), &dims,
                        ) {
                            Ok(k) => k,
                            Err(e) => {
                                weights.free_gpu(ctx.gpu);
                                return Err(format!("KvCache: {e}"));
                            }
                        }
                    }
                    (KvBackend::Vmm, hipfire_runtime::kv_mode::KvPair::Split(_, vm)) => {
                        match hipfire_runtime::llama::KvCache::new_gpu_vmm_capped_filtered(
                            ctx.gpu,
                            &is_kv_layer,
                            config.n_kv_heads,
                            config.head_dim,
                            ctx.max_seq,
                            ctx.max_seq,
                            mode,
                            vm,
                        ) {
                            Ok(k) => k,
                            Err(e) => {
                                weights.free_gpu(ctx.gpu);
                                return Err(format!("KvCache: {e}"));
                            }
                        }
                    }
                    (KvBackend::Legacy, hipfire_runtime::kv_mode::KvPair::Native(_))
                    | (KvBackend::Legacy, hipfire_runtime::kv_mode::KvPair::Split(_, hipfire_runtime::llama::VMode::Q8)) => {
                        match <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode_with_backend(
                            mode,
                            KvBackend::Legacy,
                            hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                            &dims,
                        ) {
                            Ok(k) => k,
                            Err(e) => {
                                weights.free_gpu(ctx.gpu);
                                return Err(format!("KvCache: {e}"));
                            }
                        }
                    }
                    (KvBackend::Legacy, hipfire_runtime::kv_mode::KvPair::Split(_, vm)) => {
                        let mut kv = match <hipfire_runtime::llama::KvCache as hipfire_runtime::llama::KvCacheExt>::from_mode_with_backend(
                            mode,
                            KvBackend::Legacy,
                            hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                            &dims,
                        ) {
                            Ok(k) => k,
                            Err(e) => {
                                weights.free_gpu(ctx.gpu);
                                return Err(format!("KvCache: {e}"));
                            }
                        };
                        if let Err(e) = kv.set_v_mode_realloc(ctx.gpu, vm) {
                            let mut note = format!("set_v_mode_realloc: {e}");
                            if let Err(fe) = kv.free_gpu(ctx.gpu) {
                                note = format!("{note}; cleanup also failed: {fe}");
                            }
                            weights.free_gpu(ctx.gpu);
                            return Err(note);
                        }
                        kv
                    }
                };

                let dn_state = match hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant(
                    ctx.gpu, &config, dn_quant,
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        let mut note = format!("DeltaNetState::new_with_quant: {e:?}");
                        if let Err(fe) = kv_cache.free_gpu(ctx.gpu) {
                            note = format!("{note}; cleanup also failed: {fe}");
                        }
                        weights.free_gpu(ctx.gpu);
                        return Err(note);
                    }
                };
                let scratch = match hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
                    ctx.gpu,
                    &config,
                    2048,
                    ctx.max_seq,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let mut note = format!("Qwen35Scratch::new_with_kv_max: {e:?}");
                        if let Err(fe) = kv_cache.free_gpu(ctx.gpu) {
                            note = format!("{note}; cleanup also failed: {fe}");
                        }
                        dn_state.free_gpu(ctx.gpu);
                        weights.free_gpu(ctx.gpu);
                        return Err(note);
                    }
                };
                let bundle = hipfire_arch_qwen35::Qwen35Bundle {
                    config,
                    weights,
                    scratch,
                    kv_cache,
                    dn_state,
                    // Dir/safetensors path does not engage adaptive (HFQ carrier only).
                    kv_adaptive: None,
                    pp_scratch_set: None,
                    vision_config: None,
                    vision_weights: None,
                    qwen35_decode_batch: None,
                    vmm_store: None,
                    weight_lease: None,
                };
                Ok(LoadedModel {
                    state: Some(Box::new(bundle)),
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        meta.chat_template,
                        meta.chat_template_source,
                    )
                })
            }
        }
    }

    fn spawn_slot_engine(
        &self,
        cfg: hipfire_runtime::serve::SlotEngineConfig,
    ) -> Result<Box<dyn hipfire_runtime::serve::SlotEngineHandle>, String> {
        use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
        let engine = SlotEngine::spawn(EngineConfig {
            model_path: cfg.model_path,
            n_slots: cfg.n_slots,
            cap_tokens: cfg.cap_tokens,
            prefill_chunk: cfg.prefill_chunk,
            host_budget_bytes: cfg.host_budget_bytes,
            swap_dir: cfg.swap_dir,
            is_vl: cfg.is_vl,
            vl_path: cfg.vl_path,
            mtp_k: cfg.mtp_k,
            mtp_mode: cfg.mtp_mode,
            kv_mode_raw: cfg.kv_mode_raw,
            kv_backend: cfg.kv_backend,
            prefix_cache: cfg.prefix_cache,
            prefix_cache_max_bytes: cfg.prefix_cache_max_bytes,
            max_batch_tokens: cfg.max_batch_tokens,
            prefill_min_tokens: cfg.prefill_min_tokens,
            wait_max_count: cfg.wait_max_count,
            wait_max_bytes: cfg.wait_max_bytes,
            queue_timeout_ms: cfg.queue_timeout_ms,
            structured_jump_forward: cfg.structured_jump_forward,
            dflash_draft: cfg.dflash_draft,
            dflash_required: cfg.dflash_required,
        })
        .map_err(|e| format!("SlotEngine spawn: {e}"))?;
        Ok(Box::new(engine))
    }
    fn decide_supported(&self) -> bool {
        true
    }
    fn decide_prefill_logits(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Option<Result<Vec<f32>, String>> {
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let r = hipfire_arch_qwen35::qwen35::forward_prefill_batch(
            gpu,
            &b.weights,
            &b.config,
            tokens,
            start_pos,
            &mut b.kv_cache,
            &mut b.dn_state,
            &b.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("decide prefill: {e:?}"))
        .and_then(|()| {
            if !want_logits {
                return Ok(Vec::new());
            }
            gpu.download_f32(&b.scratch.logits)
                .map_err(|e| format!("decide logits: {e:?}"))
        });
        if r.is_ok() {
            m.seq_pos = start_pos + tokens.len();
        }
        Some(r)
    }
    fn decide_save(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        let seq_pos = m.seq_pos;
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let mut snap = match DeltaNetSnapshot::new_for(gpu, &b.dn_state) {
            Ok(s) => s,
            Err(e) => return Some(Err(format!("decide snapshot alloc: {e:?}"))),
        };
        if let Err(e) = snap.save_from(&b.dn_state, gpu) {
            snap.free_gpu(gpu);
            return Some(Err(format!("decide snapshot save: {e:?}")));
        }
        Some(Ok(crate::decide::DecideSnapshot {
            seq_pos,
            recurrent: Some(snap),
        }))
    }
    fn decide_restore(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let Some(rec) = snap.recurrent.as_ref() else {
            return Some(Err("decide: qwen35 snapshot has no recurrent state".into()));
        };
        if let Err(e) = rec.restore_to(&mut b.dn_state, gpu) {
            return Some(Err(format!("decide restore: {e:?}")));
        }
        m.seq_pos = snap.seq_pos;
        Some(Ok(()))
    }
    fn decide_kv_compact_offset(&self, m: &crate::LoadedModel) -> Option<usize> {
        m.qwen35().map(|b| b.kv_cache.compact_offset)
    }
}

// ─── LlamaCarrier ────────────────────────────────────────────────────

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str {
        "llama"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        }) {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("llama: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 0 = LLaMA/Mistral, 1 = plain Qwen3/Qwen2 (both namespaces).
        // Explicit allowlist (was an open `< 5` range that would silently
        // swallow any future HFQ id in 2..=4 into the llama path).
        matches!(arch_id, 0 | 1)
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: Some(saddle_core::caps::DflashKind::Llama),
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.llama_mut().unwrap();
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_runtime::llama::forward_scratch(
                gpu, weights, config, tok, i, kv, scratch, 0.0, 1.0, 42, 0, 1.0,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "llama",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        let mut bundle = hipfire_arch_llama::load_llama_bundle(src, ctx)?;

        // ── DSpark sidecar discovery ──────────────────────────────────────────
        // When a `<stem>-dspark.<ext>` sidecar exists alongside the main model
        // and speculation is not explicitly disabled (`ctx.spec.dspark != Some(false)`),
        // load the Qwen3-8B drafter body + DSpark globals into the bundle.
        //
        // The speculator BUILD arm (Task 10) reads bundle.dspark_weights +
        // bundle.dspark_assets to wire the DsparkDrafter into the serve path.
        // This block only does the load — no speculator is built here.
        if ctx.spec.dspark != Some(false) {
            let base_path = std::path::Path::new(ctx.path);
            let dspark_path: Option<std::path::PathBuf> = match (
                base_path.parent(),
                base_path.file_stem(),
                base_path.extension(),
            ) {
                (Some(parent), Some(stem), Some(ext)) => Some(parent.join(format!(
                    "{}-dspark.{}",
                    stem.to_string_lossy(),
                    ext.to_string_lossy()
                ))),
                _ => None,
            };
            if let Some(p) = dspark_path.filter(|p| p.exists()) {
                eprintln!("llama: opening DSpark sidecar HFQ {p:?}");
                match hipfire_runtime::hfq::HfqFile::open(&p) {
                    Ok(mut sidecar) => {
                        sidecar.drop_mmap();
                        match hipfire_arch_llama::dspark_body::load_qwen3_dspark(&sidecar, ctx.gpu)
                        {
                            Ok(Some((dspark_weights, dspark_assets))) => {
                                eprintln!(
                                    "  llama: DSpark sidecar loaded (block_size={}, target_layers={:?})",
                                    dspark_weights.cfg.block_size,
                                    dspark_weights.cfg.target_layer_ids,
                                );
                                bundle.dspark_weights = Some(dspark_weights);
                                bundle.dspark_assets = Some(dspark_assets);
                            }
                            Ok(None) => {
                                eprintln!(
                                    "  llama: DSpark sidecar {p:?} has no dspark_* metadata — skipping"
                                );
                            }
                            Err(e) => {
                                eprintln!("  llama: WARNING DSpark sidecar load failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  llama: WARNING cannot open DSpark sidecar {p:?}: {e}");
                    }
                }
            } else if ctx.spec.dspark == Some(true) {
                // Forced `--spec dspark` but the sidecar file is absent → we would
                // silently run AR. Warn (auto/`None` stays quiet — a missing sidecar
                // is the expected no-op there).
                eprintln!(
                    "  llama: WARNING `--spec dspark` requested but no `-dspark` sidecar found \
                     (expected `<stem>-dspark.<ext>` next to the model) — falling back to AR/other drafter"
                );
            }
        }

        // ── single shared tail ──
        // Precedence (arch_id=0/1): DSpark > DFlash > n-gram.
        //
        // DSpark sidecar speculator: present when the `-dspark` sidecar was loaded
        // (bundle.dspark_weights.is_some()) AND speculation is not explicitly disabled.
        // Consumes the assets from the bundle (moves them into the speculator body).
        //
        // If no DSpark sidecar is available, fall through to:
        // - DFlash generic speculator (arch_id=20 draft).
        // - Opt-in model-free n-gram (HIPFIRE_NGRAM_DRAFT=1).
        let speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> = if bundle
            .dspark_weights
            .is_some()
            && ctx.spec.dspark != Some(false)
        {
            let dspark_weights = bundle.dspark_weights.take().unwrap();
            let assets = bundle.dspark_assets.take().unwrap();
            let block = dspark_weights.cfg.block_size;
            let vocab = assets.config.vocab_size;

            // stage_norm = drafter's final `norm.weight` (output_norm in the sidecar).
            // Shallow-clone so the LlamaWeights (assets) owns the primary GpuTensor;
            // the speculator holds an alias that is freed before the weights on unload.
            let stage_norm = assets.weights.output_norm.shallow_clone();

            // lm_head fix: assets.weights.output.buf.dtype == Raw (upload_raw always
            // sets Raw), but the actual data layout is F16.  run_heads dispatches on
            // GpuTensor.dtype, so we shallow_clone and fix the dtype + shape here.
            // (The parity harness does the same at qwen3_dspark_parity.rs:215-217.)
            let mut lm_head = assets.weights.output.buf.shallow_clone();
            lm_head.dtype = rdna_compute::DType::F16;
            lm_head.shape = vec![vocab];

            // conf_threshold ladder: env > CLI arg > 0.1
            // Default 0.1 (sweep-tuned): 0.5 over-truncates (1.46/7 proposed);
            // 0.1 proposes ~6.94/7, +16.6% prose tok/s / +7.1% code tok/s.
            let conf_threshold =
                hipfire_config::developer_var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .or(ctx.spec.dspark_conf_threshold)
                    .unwrap_or(0.1f32);

            eprintln!(
                "  llama DSpark speculator enabled (sidecar, block={}, conf_threshold={:.2})",
                block, conf_threshold
            );
            let body = hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
                assets,
                &dspark_weights.cfg,
                ctx.gpu,
            )
            .map_err(|e| format!("llama DSpark body build failed: {e}"))?;
            Some(hipfire_runtime::dspark_core::build_dspark_speculator(
                body,
                dspark_weights,
                stage_norm,
                lm_head,
                block,
                ctx.max_seq,
                conf_threshold,
                // temp>0 sampled verify ENABLED: with lazy prefix sampling (only ~τ
                // lm_heads/window) qwen3 DSpark at temp>0 beats AR by ~+24% (29.6 vs
                // 23.8 tok/s on gfx1151 code) and stays distribution-identical to AR
                // (fused sample_top_p_pf, honors temp+top_p+top_k). The daemon routes
                // temp>0 llama through the chain path (requires_greedy()==false).
                true,
                hipfire_runtime::dspark_block_controller::WindowCost::Measured { max_ratio: 0.5 },
            ))
        } else if let Some(dp) = ctx.draft_path {
            // Peek at the draft's arch_id without consuming the path; the builder
            // opens it again internally.
            match hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(dp)) {
                Ok(draft_hfq) if draft_hfq.arch_id == 20 => {
                    // Parse DflashConfig to validate the cross-attention concat invariant
                    // (review finding L4): the drafter's hidden must equal the target dim.
                    let draft_cfg = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq)
                        .ok_or_else(|| {
                            format!(
                                "DFlash draft '{}' is arch_id=20 but its 'dflash' metadata block is \
                                 missing or malformed; re-download the draft",
                                dp
                            )
                        })?;
                    if bundle.config.dim != draft_cfg.hidden {
                        return Err(format!(
                            "DFlash draft '{}' has hidden size {} but the target model's is {}; a draft \
                                 must be made for this exact target model. Use the matching \
                                 draft or turn DFlash off (speculation.dflash = off)",
                            dp, draft_cfg.hidden, bundle.config.dim
                        ));
                    }
                    // Drop the peek handle before the builder reopens it.
                    drop(draft_hfq);
                    let spec = hipfire_runtime::dflash_generic::build_generic_dflash_speculator(
                        ctx.gpu,
                        dp,
                        &mut bundle,
                        ctx.max_seq,
                    )
                    .map_err(|e| format!("DFlash generic speculator build failed: {e}"))?;
                    eprintln!(
                        "  DFlash generic speculator loaded for arch {} target: {}",
                        meta.arch_id, dp
                    );
                    Some(spec)
                }
                // Not a DFlash draft or unreadable — log why and fall through to n-gram.
                Err(e) => {
                    eprintln!(
                        "  [hipfire] draft '{}' unreadable ({e}); DFlash speculator not built, falling back to n-gram",
                        dp
                    );
                    crate::spec_build::build_speculator(
                        meta.arch_id,
                        None,
                        None,
                        true,
                        ctx.max_seq,
                        ctx.spec,
                    )
                }
                Ok(draft_hfq) => {
                    eprintln!(
                        "  [hipfire] draft '{}' is arch_id={} (not 20 / DFlash); DFlash speculator not built, falling back to n-gram",
                        dp, draft_hfq.arch_id
                    );
                    crate::spec_build::build_speculator(
                        meta.arch_id,
                        None,
                        None,
                        true,
                        ctx.max_seq,
                        ctx.spec,
                    )
                }
            }
        } else {
            // No draft configured: opt-in model-free n-gram (HIPFIRE_NGRAM_DRAFT=1) or None.
            crate::spec_build::build_speculator(
                meta.arch_id,
                None,
                None,
                true,
                ctx.max_seq,
                ctx.spec,
            )
        };
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
    fn decide_supported(&self) -> bool {
        true
    }
    fn decide_prefill_logits(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Option<Result<Vec<f32>, String>> {
        let Some(b) = m.llama_mut() else {
            return Some(Err("decide: llama bundle missing".into()));
        };
        let r = hipfire_runtime::llama::forward_prefill_batch(
            gpu, &b.weights, &b.config, tokens, start_pos, &mut b.kv, &b.scratch, None,
        )
        .map_err(|e| format!("decide prefill: {e:?}"))
        .and_then(|()| {
            if !want_logits {
                return Ok(Vec::new());
            }
            gpu.download_f32(&b.scratch.logits)
                .map_err(|e| format!("decide logits: {e:?}"))
        });
        if r.is_ok() {
            m.seq_pos = start_pos + tokens.len();
        }
        Some(r)
    }
    fn decide_save(
        &self,
        m: &mut crate::LoadedModel,
        _gpu: &mut rdna_compute::Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        // Pure-attention: KV is positional, the cursor is the whole state.
        Some(Ok(crate::decide::DecideSnapshot {
            seq_pos: m.seq_pos,
            recurrent: None,
        }))
    }
    fn decide_restore(
        &self,
        m: &mut crate::LoadedModel,
        _gpu: &mut rdna_compute::Gpu,
        snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        m.seq_pos = snap.seq_pos;
        Some(Ok(()))
    }
    fn decide_kv_compact_offset(&self, m: &crate::LoadedModel) -> Option<usize> {
        m.llama().map(|b| b.kv.compact_offset)
    }
}

// ─── Non-core carriers ───────────────────────────────────────────────

// ─── DotsOcrCarrier ──────────────────────────────────────────────────

pub struct DotsOcrCarrier;
impl Carrier for DotsOcrCarrier {
    fn name(&self) -> &'static str {
        "dots_ocr"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 8
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.dots_ocr_mut().unwrap();
        let state = &mut bundle.state;
        let config = &bundle.config;
        let weights = &bundle.weights;
        let mut ok = true;
        for &tok in synthetic {
            if hipfire_arch_qwen2::qwen2::forward_step(gpu, &weights.text, &config.text, state, tok)
                .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "dots_ocr",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        let bundle = hipfire_arch_dots_ocr::load_dots_ocr_bundle(src, ctx)?;
        // Opt-in model-free n-gram speculator (HIPFIRE_NGRAM_DRAFT=1). dots.ocr's
        // text decoder IS Qwen2, so the n-gram arm drives it via the
        // `DotsOcrBundle: SpecTarget` impl — a strong fit because layout-JSON
        // output is densely self-repeating. The daemon's `generate_vl_dots_ocr`
        // routes to the spec decode loop when this is `Some` (vision prefill is
        // unchanged; only the decode phase becomes speculative).
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── Deepseek4Carrier ────────────────────────────────────────────────

pub struct Deepseek4Carrier;
impl Carrier for Deepseek4Carrier {
    fn name(&self) -> &'static str {
        "deepseek4"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        if state
            .as_ref()
            .is_some_and(|s| (s.as_ref() as &dyn Any).is::<crate::Deepseek4HeterogeneousBundle>())
        {
            Err("deepseek4: this mixed-GPU DeepSeek setup supports plain autoregressive decoding only; speculative decoding is not available".into())
        } else if let Some(b) = state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
        }) {
            Ok(Box::new(InPlaceGuard { bundle: b }))
        } else {
            Err("deepseek4: spec target state mismatch".into())
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(
            ctx,
        ))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 9
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::DeepSeek4,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.0, 1.0, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m
            .state
            .as_mut()
            .and_then(|s| {
                (s.as_mut() as &mut dyn Any)
                    .downcast_mut::<hipfire_arch_deepseek4::Deepseek4Bundle>()
            })
            .unwrap();
        let pbs = b
            .pbs
            .as_mut()
            .expect("deepseek4_pbs missing on arch_id=9 bench_prefill");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let ok = hipfire_arch_deepseek4::forward::forward_prefill_batch_chunked(
            config, weights, state, gpu, synthetic, 0, pbs,
        )
        .is_ok();
        if ok {
            state.n_tokens = n as u64;
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "deepseek4",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let compressor_cache =
            crate::resolve_deepseek4_compressor_cache_kv_mode(ctx.kv_mode_override)?;

        if !matches!(
            ctx.deepseek4_compute_placement,
            hipfire_config::Deepseek4ComputePlacement::Single
        ) {
            if ctx.kv_backend != KvBackend::Legacy {
                return Err(
                    "deepseek4: this mixed-GPU compressor placement needs the legacy KV backend; set memory.kv_backend = legacy"
                        .into(),
                );
            }
            let model = hipfire_arch_deepseek4::load_deepseek4_heterogeneous_model(
                &src,
                ctx,
                compressor_cache,
            )?;
            let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<｜end▁of▁sentence｜>"]);
            let advertised_context = model.config.max_position_embeddings;
            return Ok(LoadedModel {
                state: Some(Box::new(crate::Deepseek4HeterogeneousBundle {
                    model,
                    eos_tok,
                })),
                ..LoadedModel::skeleton(
                    meta.arch_id,
                    meta.tokenizer,
                    advertised_context,
                    advertised_context,
                    ctx.path.to_string(),
                    meta.chat_template,
                    meta.chat_template_source,
                )
            });
        }

        use hipfire_arch_deepseek4 as deepseek4;
        let deepseek4::Deepseek4LoadParts {
            config,
            weights,
            state,
            pbs,
        } = deepseek4::load_deepseek4_bundle(src, ctx, compressor_cache)?;
        let eos_tok = resolve_eos_tok(&meta.tokenizer, &["<｜end▁of▁sentence｜>"]);
        // deepseek4 MTP spec-decode capability: present iff the MTP addon weights loaded
        // (HIPFIRE_DEEPSEEK4_MTP_ADDON / .mtp-addon.hfq / HIPFIRE_DEEPSEEK4_LOAD_MTP). The
        // per-request spec gate (mtp_mode / HIPFIRE_DEEPSEEK4_SPEC_DECODE / temp<=eps) stays in
        // the generate path (T4 routing) — here we only build the capability. Undriven until T4:
        // the daemon's arch_id==9 branch still uses the bespoke generate_deepseek4 loop.
        // DSpark draft module (the `-dspark` sidecar) wins over the in-trunk MTP
        // layer when present. Built when the sidecar loaded AND the `speculation`
        // selector did not pick another mechanism (`ctx.spec.dspark != Some(false)`;
        // `None` = auto keeps the default-on behaviour). The threshold is the
        // CLI-forwarded `--dspark-conf-threshold` (env still wins in the builder).
        // `--spec dspark` (forced) but the sidecar was absent → we silently ran
        // AR before. Warn on the forced case only (auto/`None` legitimately falls
        // back without a sidecar and must stay quiet).
        if ctx.spec.dspark == Some(true) && weights.dspark.is_none() {
            eprintln!(
                "  deepseek4: WARNING `--spec dspark` requested but no `-dspark` sidecar was \
                 loaded (expected `<stem>-dspark.<ext>` next to the model) — falling back to MTP/AR"
            );
        }
        let dspark_enabled = weights.dspark.is_some() && ctx.spec.dspark != Some(false);
        let speculator: Option<Box<dyn hipfire_runtime::spec::Speculator>> = if dspark_enabled {
            let block = weights.dspark.as_ref().unwrap().cfg.block_size;
            let ctx_capacity = config.max_position_embeddings;
            eprintln!("  deepseek4 DSpark speculator enabled (sidecar, block={block})");
            Some(
                hipfire_arch_deepseek4::dspark_speculator::build_deepseek4_dspark_speculator(
                    &config,
                    &weights,
                    block,
                    ctx_capacity,
                    ctx.spec.dspark_conf_threshold,
                    // temp>0 sampled verify ENABLED in serving. The earlier "loses to
                    // AR → gate off" reasoning was a fixed-block measurement artifact;
                    // comprehensive temp=1.0 tests with the τ-adaptive block-depth
                    // controller show ds4 DSpark temp>0 BEATS AR, and the opt-in CACTUS
                    // acceptance-boost (request `cactus_delta`) adds more on top.
                    // Distribution-preserving at cactus_delta=0 (the default).
                    true,
                )
                .map_err(|e| format!("deepseek4 DSpark speculator build failed: {e}"))?,
            )
        } else if weights.mtp_layer.is_some() {
            // spec_k resolution MUST mirror daemon.rs:9349 (HIPFIRE_DEEPSEEK4_SPEC_K →
            // HIPFIRE_MTP_K → default 2) so T4's spec.k() matches the bespoke loop's window.
            let max_n: usize = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_SPEC_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .or(Some(
                    hipfire_runtime::config::get()
                        .mtp_k
                        .unwrap_or(hipfire_config::MTP_K_DEFAULT),
                ))
                .unwrap_or(2);
            let ctx_capacity = config.max_position_embeddings;
            eprintln!("  deepseek4 MTP speculator enabled (in-weights, K={max_n})");
            Some(
                hipfire_arch_deepseek4::mtp_speculator::build_deepseek4_mtp_speculator(
                    max_n,
                    ctx_capacity,
                ),
            )
        } else {
            None
        };
        let advertised_context = config.max_position_embeddings;
        eprintln!(
            "  deepseek4 KV cache: {} growth to advertised context {advertised_context}",
            if ctx.kv_backend == KvBackend::Vmm {
                "VMM"
            } else {
                "legacy"
            }
        );
        Ok(LoadedModel {
            state: Some(Box::new(deepseek4::Deepseek4Bundle {
                config,
                weights,
                state,
                eos_tok,
                pbs: Some(pbs),
            })),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                advertised_context,
                advertised_context,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── MinimaxCarrier ──────────────────────────────────────────────────

pub struct MinimaxCarrier;
impl Carrier for MinimaxCarrier {
    fn name(&self) -> &'static str {
        "minimax"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::MiniMaxBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("minimax: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Shared ChatML emitter (same one qwen2 reuses): MiniMax-M2 is ChatML
        // (`<|im_end|>`), so the generic think/tool-call/EOS scanning applies.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 10
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 1.0, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.minimax_mut().expect("arch_id=10 requires minimax bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_arch_minimax::forward::decode_step(
                config, weights, state, gpu, tok, i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "minimax",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        // Per-source diagnostic stays at the call site, before resolve_source_meta.
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_minimax::load_minimax_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── Lfm2MoeCarrier ──────────────────────────────────────────────────

pub struct Lfm2MoeCarrier;
impl Carrier for Lfm2MoeCarrier {
    fn name(&self) -> &'static str {
        "lfm2moe"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::Lfm2MoeBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("lfm2moe: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Shared ChatML emitter (same one qwen2/minimax reuse): LFM2.5 is ChatML
        // (`<|im_end|>`), no bespoke marker state machine.
        Ok(Qwen35Emit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 11
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            // Continuous batching is NOT servable: the generate-side eligibility
            // (`is_batch_request_eligible`) returns false unconditionally for
            // LFM, so staging a batch state only spends VRAM on state that is
            // never driven. Declare false so the route never admits it and the
            // state is never allocated. Single-stream LFM is unaffected.
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            // lfm2_vl artifacts carry the SigLIP2 tower + projector. The
            // declared-capability model works exactly like qwen35-vl here:
            // artifacts WITHOUT vision tensors still load fine and refuse
            // images at the daemon gate via `has_vision_encoder() == false`.
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.1, 0.80, 1.05)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m.lfm2moe_mut().expect("arch_id=11 requires lfm2moe bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if hipfire_arch_lfm2moe::forward::decode_step(
                config, weights, state, gpu, tok, i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "lfm2moe",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;

        // ── LFM2-VL detection + tower/projector upload ────────────────────
        // Mirrors the qwen35 HFQ arm: HFQ is single-pass, so the vision
        // stack MUST be read from the same file handle BEFORE
        // `load_lfm2moe_bundle` consumes it for the text trunk.
        let mut vision_cfg_out: Option<hipfire_arch_lfm2_vl::VisionConfig> = None;
        let mut vision_w_out: Option<hipfire_arch_lfm2_vl::VisionWeights> = None;
        if let ModelSource::Hfq(hfq_file) = &src {
            if hfq_file
                .tensor_data("model.vision_tower.vision_model.embeddings.patch_embedding.weight")
                .is_some()
            {
                match hipfire_arch_lfm2_vl::vision_config_from_hfq(hfq_file) {
                    Some(vc) => {
                        eprintln!(
                            "  LFM2-VL model: SigLIP2-NaFlex tower (hidden={}, layers={}) + projector",
                            vc.hidden_size, vc.num_layers
                        );
                        match hipfire_arch_lfm2_vl::load_vision_weights(hfq_file, &vc, ctx.gpu) {
                            Ok(vw) => {
                                vision_cfg_out = Some(vc);
                                vision_w_out = Some(vw);
                            }
                            Err(e) => {
                                return Err(format!(
                                    "lfm2moe: LFM2-VL vision weight load failed: {e:?}"
                                ));
                            }
                        }
                    }
                    None => {
                        return Err(
                            "lfm2moe: artifact carries vision tensors but no vision_config \
                             metadata — requantize with --include-vision"
                                .into(),
                        );
                    }
                }
            }
        }

        let bundle = match hipfire_arch_lfm2moe::load_lfm2moe_bundle(src, ctx) {
            Ok(mut b) => {
                b.vision_config = vision_cfg_out;
                b.vision_weights = vision_w_out;
                b
            }
            Err(e) => {
                // Trunk failed after the tower is already on-device — reclaim
                // it or the unload drain will miss these buffers entirely.
                if let Some(vw) = vision_w_out {
                    vw.free_gpu(ctx.gpu);
                }
                return Err(e);
            }
        };
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── Cohere2MoeCarrier ───────────────────────────────────────────────
// cohere2moe (arch_id 12, HFQ-only) landed upstream via the generic
// `HfqCarrier` fn-pointer registry entry. Our dedicated-carrier refactor
// removed that generic struct, so this wraps the still-standalone
// `crate::load_cohere2moe` with the same HFQ-extraction glue the old
// `HfqCarrier::load` used — keeping cohere2moe's load path byte-identical
// to upstream while fitting the dedicated-carrier registry.
pub struct Cohere2MoeCarrier;
impl Carrier for Cohere2MoeCarrier {
    fn name(&self) -> &'static str {
        "cohere2moe"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::Cohere2MoeBundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("cohere2moe: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        // Arch-specific emitter: North's agentic-marker state machine (markers
        // never surfaced, reasoning channel, ACTION→tool_calls) + the empty-turn
        // and think-budget generation guards via `take_forced`.
        Ok(hipfire_arch_cohere2moe::spec_emit::Cohere2MoeEmit::from_ctx(ctx))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 12 = Cohere2-MoE in both the HFQ and safetensors-Dir namespaces.
        arch_id == 12
    }
    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if pp > 1 {
            return Err(crate::pp_unsupported_message("cohere2moe", is_dir, pp));
        }
        Ok(())
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 0.95, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        _prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = m
            .cohere2moe_mut()
            .expect("arch_id=12 requires cohere2moe bundle");
        let config = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let mut ok = true;
        if hipfire_arch_cohere2moe::forward::forward_batch_supported(weights) && synthetic.len() > 1
        {
            let mut i = 0;
            while i < synthetic.len() {
                let end = (i + 256).min(synthetic.len());
                let start_pos = state.n_tokens;
                if hipfire_arch_cohere2moe::forward::forward_batch(
                    config,
                    weights,
                    state,
                    gpu,
                    &synthetic[i..end],
                    start_pos,
                )
                .is_err()
                {
                    ok = false;
                    break;
                }
                i = end;
            }
        } else {
            for (i, &tok) in synthetic.iter().enumerate() {
                if hipfire_arch_cohere2moe::forward::decode_step(
                    config, weights, state, gpu, tok, i as u32,
                )
                .is_err()
                {
                    ok = false;
                    break;
                }
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "cohere2moe",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_cohere2moe::load_cohere2moe_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── MapleCarrier ────────────────────────────────────────────────────
// maple (arch_id 15, HFQ-only). Maple-Preview's weights are natively ternary
// and are carried losslessly by qt=51 MQ2G256LloydU; there is no
// safetensors-Dir path, so `claims_arch_id` answers for the HFQ namespace only
// in practice and the Dir case is refused inside `load_maple_bundle` with a
// message naming the convert command.
pub struct MapleCarrier;
impl Carrier for MapleCarrier {
    fn name(&self) -> &'static str {
        "maple"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 15
    }
    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if pp > 1 {
            return Err(crate::pp_unsupported_message("maple", is_dir, pp));
        }
        Ok(())
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            reasoning_contract: saddle_core::caps::ReasoningContract::QwenJinja,
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 0.95, 1.0)
    }
    /// Batched prefill over `MAPLE_PREFILL_CHUNK`-sized chunks. `forward_batch`
    /// ERRORS above `MAPLE_PREFILL_MAX_B` rather than splitting silently, so the
    /// chunking here is mandatory, not an optimisation.
    ///
    /// The failure string is propagated through `prefill_err`, which the daemon
    /// interpolates into `bench_prefill forward failed: {e}` for every carrier —
    /// not just Glimmer. Dropping it would reduce an unsupported-tier refusal
    /// (which names the qt51 and router-mirror requirements) to a bare
    /// "forward failed".
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let b = (m.state.as_mut()?.as_mut() as &mut dyn Any)
            .downcast_mut::<hipfire_arch_maple::MapleBundle>()?;
        for (start, len) in hipfire_arch_maple::batch::prefill_chunks(
            synthetic.len(),
            hipfire_arch_maple::batch::MAPLE_PREFILL_CHUNK,
        ) {
            if let Err(e) = hipfire_arch_maple::forward::forward_batch(
                &b.config,
                &b.weights,
                &mut b.state,
                gpu,
                &synthetic[start..start + len],
                start,
            ) {
                *prefill_err = Some(e);
                return Some(false);
            }
        }
        Some(true)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "maple",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_maple::load_maple_bundle(src, ctx)?;
        let speculator = crate::spec_build::build_speculator(
            meta.arch_id,
            None,
            None,
            true,
            ctx.max_seq,
            ctx.spec,
        );
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── Gemma4Carrier ───────────────────────────────────────────────────

pub struct Gemma4Carrier;
impl Carrier for Gemma4Carrier {
    fn name(&self) -> &'static str {
        "gemma4"
    }
    fn spec_target_guard<'m>(
        &self,
        state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        match state
            .as_mut()
            .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<crate::Gemma4Bundle>())
        {
            Some(bundle) => Ok(Box::new(InPlaceGuard { bundle })),
            _ => Err("gemma4: spec target state mismatch".into()),
        }
    }
    fn make_spec_emitter<'a>(
        &self,
        ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Ok(Box::new(hipfire_arch_gemma4::emit::GemmaSpecEmit::new(ctx)))
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 13 = gemma4_text (primary), 22 = gemma4_unified_assistant (EAGLE drafter sidecar).
        // The drafter file (22) is loaded via params.drafter path, not as a primary serve model,
        // but claiming it here lets the quantizer's 22-stamped draft file be probed without
        // "no carrier" error and keeps the two namespaces aligned. Primary serve of 22 alone
        // would still need a target model, so it naturally fails later in generate routing.
        matches!(arch_id, 13 | 22)
    }
    fn admit_topology(
        &self,
        _arch_id: u32,
        is_dir: bool,
        pp: usize,
        _kv_backend: KvBackend,
    ) -> Result<(), String> {
        if pp > 1 {
            return Err(crate::pp_unsupported_message("gemma4", is_dir, pp));
        }
        Ok(())
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::GemmaBoolean,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(1.0, 0.95, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        if let Some(bundle) = m.gemma4_mut() {
            for (i, &tok) in synthetic.iter().enumerate() {
                if let Err(error) = hipfire_arch_gemma4::forward::decode_step(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    tok,
                    i as u32,
                ) {
                    *prefill_err = Some(format!("gemma4 bench prefill failed: {error:?}"));
                    return Some(false);
                }
            }
            return Some(true);
        }

        *prefill_err = Some("gemma4 bench prefill missing state".into());
        Some(false)
    }
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let mut error = None;
        match self.bench_prefill(m, gpu, synthetic, synthetic.len(), &mut error) {
            Some(true) => Some(None),
            Some(false) => {
                Some(Some(error.unwrap_or_else(|| {
                    "gemma4 bench decode prime failed".into()
                })))
            }
            None => None,
        }
    }
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        decode_err: &mut Option<String>,
    ) -> Option<bool> {
        if let Some(bundle) = m.gemma4_mut() {
            for i in 0..iterations {
                let token = 101 + (i as u32 % 1000);
                if let Err(error) = hipfire_arch_gemma4::forward::decode_step(
                    &bundle.config,
                    &bundle.weights,
                    &mut bundle.state,
                    gpu,
                    token,
                    (context + i) as u32,
                ) {
                    *decode_err = Some(format!("gemma4 bench decode failed: {error:?}"));
                    return Some(false);
                }
            }
            return Some(true);
        }

        *decode_err = Some("gemma4 bench decode missing state".into());
        Some(false)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "gemma4",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        if ctx.draft_path.is_some() {
            return Err("params.draft (a Qwen3.5 DFlash draft) is not supported for Gemma 4 (arch_id=13). For Gemma 4 speculative decoding, pass the Gemma 4 EAGLE drafter (arch 22) as params.drafter instead.".to_string());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let session_domain = match &src {
            ModelSource::Hfq(hfq) => Some(hipfire_runtime::serve_contract::CacheDomain::for_model(
                hfq,
                &meta.tokenizer,
                meta.chat_template.as_deref(),
                "gemma4",
                ctx.gpu.device_id,
            )),
            ModelSource::Dir(_) => None,
        };
        let mut bundle = hipfire_arch_gemma4::load_gemma4_bundle(src, ctx)?;
        let session_budget = hipfire_config::memory::session_cache_bytes();
        let snapshots = hipfire_runtime::session_cache::SessionState::snapshot_scope(
            &bundle,
            hipfire_runtime::session_cache::SessionRoute::Ar,
        )
        .is_some();
        if let Some(domain) = session_domain.filter(|_| session_budget > 0 && snapshots) {
            bundle.attach_session_cache(hipfire_runtime::session_cache::SessionCache::new(
                domain,
                session_budget,
            ));
            eprintln!(
                "  gemma4 session cache: {} MiB budget, {} MiB per {}-token snapshot",
                session_budget >> 20,
                bundle.session_snapshot_bytes() >> 20,
                hipfire_arch_gemma4::bundle::SESSION_STRIDE,
            );
        }
        bundle.eos_tok = resolve_eos_tok(
            &meta.tokenizer,
            &["<end_of_turn>", "<turn|>", "<eos>", "<|im_end|>"],
        );
        // Optional EAGLE drafter (arch-22) as the request speculator. A bad
        // sidecar logs and leaves the target AR-only.
        let mut speculator = None;
        if let Some(dp) = ctx.gemma4_drafter_path {
            let draft_len = crate::gemma4_eagle_spec_len(Some(ctx.gemma4_draft_len as u64))
                .map_err(|e| format!("gemma4 drafter spec_len: {e}"))?;
            match hipfire_arch_gemma4::Gemma4Drafter::load(dp, draft_len, &bundle, ctx.gpu) {
                Ok(drafter) => {
                    eprintln!(
                        "  gemma4 EAGLE drafter loaded: {} (layers={}, hidden={}, draft_len={draft_len})",
                        dp,
                        drafter.config().n_layers,
                        drafter.config().hidden,
                    );
                    speculator = Some(Box::new(hipfire_runtime::spec::MtpSpeculator::new(drafter))
                        as Box<dyn hipfire_runtime::spec::Speculator>);
                }
                Err(e) => eprintln!(
                    "  gemma4 EAGLE drafter load failed ({}): {} — falling back to AR only",
                    dp, e
                ),
            }
        }
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
}

// ─── MuseGlimmerCarrier ────────────────────────────────────────────────

pub struct MuseGlimmerCarrier;
impl Carrier for MuseGlimmerCarrier {
    fn name(&self) -> &'static str {
        "muse_glimmer"
    }
    fn spec_target_guard<'m>(
        &self,
        _state: &'m mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
        _model_path: &str,
    ) -> Result<Box<dyn SpecTargetGuard + 'm>, String> {
        Err("muse_glimmer: spec decode not yet wired (AR-only)".into())
    }
    fn make_spec_emitter<'a>(
        &self,
        _ctx: SpecEmitCtx<'a>,
    ) -> Result<Box<dyn SpecEmit + 'a>, String> {
        Err("muse_glimmer: spec emitter not yet wired".into())
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 14
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: Some(2),
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::MuseGlimmer,
        }
    }
    fn sampling_defaults(&self) -> saddle_core::sampling::SamplingDefaults {
        saddle_core::sampling::SamplingDefaults::new(0.3, 0.8, 1.0)
    }
    fn bench_prefill(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
        _n: usize,
        prefill_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.muse_glimmer_mut().unwrap();
        Some(if bundle.device_hidden_capture_enabled() {
            match hipfire_arch_muse_glimmer::forward::prefill_with_device_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
            ) {
                Ok(_) => true,
                Err(e) => {
                    *prefill_err = Some(e);
                    false
                }
            }
        } else {
            let mut hidden_out: Vec<f32> = Vec::new();
            match hipfire_arch_muse_glimmer::forward::prefill_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
                &[],
                &mut hidden_out,
            ) {
                Ok(_) => true,
                Err(e) => {
                    *prefill_err = Some(e);
                    false
                }
            }
        })
    }
    fn bench_decode_prime(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        synthetic: &[u32],
    ) -> Option<Option<String>> {
        let bundle = m.muse_glimmer_mut().unwrap();
        bundle.reset_session_state();
        Some(if bundle.device_hidden_capture_enabled() {
            hipfire_arch_muse_glimmer::forward::prefill_with_device_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
            )
            .err()
        } else {
            let mut hidden_out: Vec<f32> = Vec::new();
            hipfire_arch_muse_glimmer::forward::prefill_with_capture(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                gpu,
                synthetic,
                0,
                &[],
                &mut hidden_out,
            )
            .err()
        })
    }
    fn bench_decode_run(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        context: usize,
        iterations: usize,
        decode_err: &mut Option<String>,
    ) -> Option<bool> {
        let bundle = m.muse_glimmer_mut().unwrap();
        let config = &bundle.config;
        let weights = &bundle.weights;
        let state = &mut bundle.state;
        let mut ok = true;
        for i in 0..iterations {
            let token = 101 + (i as u32 % 1000);
            match hipfire_arch_muse_glimmer::forward::decode_step(
                config,
                weights,
                state,
                gpu,
                token,
                (context + i) as u32,
            ) {
                Ok(_) => {}
                Err(e) => {
                    *decode_err = Some(format!("iter {i} pos {}: {e}", context + i));
                    ok = false;
                    break;
                }
            }
        }
        Some(ok)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.kv_backend == KvBackend::Vmm && ctx.cask.sidecar.is_some() {
            return Err(
                "muse_glimmer: KV backend 'vmm' does not support CASK/TriAttention eviction; disable the sidecar or use 'legacy'"
                    .into(),
            );
        }
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "muse_glimmer",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        if ctx.gemma4_drafter_path.is_some() {
            return Err(
                "params.drafter (a Gemma 4 EAGLE drafter) is not supported for Muse Glimmer (arch_id=14). Remove params.drafter to decode without a draft."
                    .to_string(),
            );
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        match src {
            ModelSource::Hfq(hfq) => {
                // HFQ load path via the crate API (contract: from_hfq / load /
                // new_with_max_seq_backend with ctx.kv_backend).
                let config = hipfire_arch_muse_glimmer::config::GlimmerConfig::from_hfq(&hfq)?;
                let weights = hipfire_arch_muse_glimmer::glimmer::GlimmerWeights::load(
                    &hfq, &config, ctx.gpu,
                )?;
                let mut state =
                    match hipfire_arch_muse_glimmer::glimmer::GlimmerState::new_with_max_seq_backend(
                        ctx.gpu,
                        &config,
                        ctx.max_seq,
                        ctx.kv_backend,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            weights.free_gpu(ctx.gpu);
                            return Err(format!(
                                "muse_glimmer: GlimmerState::new_with_max_seq_backend failed: {e}"
                            ));
                        }
                    };
                // eos resolution by name with 200001 as the documented fallback.
                // 200001 is Glimmer's eos_token_id (config.eos_token, bos 200000).
                // Try tokenizer encode of common EOS surface forms first; if none
                // tokenize to a single id, fall back to 200001 (never 1 — the
                // generic fallback in resolve_eos_tok would be silently wrong
                // for Glimmer and would cause runaway generation).
                let eos_tok = {
                    let tok = resolve_eos_tok(
                        &meta.tokenizer,
                        &[
                            "<end_of_turn>",
                            "<|im_end|>",
                            "</s>",
                            "<|endoftext|>",
                            "<eos>",
                        ],
                    );
                    if tok == 1 {
                        200001
                    } else {
                        tok
                    }
                };
                // Optional DFlash drafter (arch 23). OFF by default — daemon
                // only populates ctx.draft_path when HIPFIRE_DFLASH_DRAFT is set
                // (or params.draft) and dflash_mode != "off". When present, load
                // the 5-layer diffusion draft head (encoder.fc / layers.* / norm)
                // and stash it on the bundle for the speculator to consume.
                // On any failure, log and fall back to AR-only (never fail the
                // target load because the draft is auxiliary).
                let drafter: Option<crate::GlimmerDrafterBundle> = if let Some(dp) =
                    ctx.draft_path.clone()
                {
                    match (|| -> Result<crate::GlimmerDrafterBundle, String> {
                        let dhfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&dp))
                            .map_err(|e| format!("open glimmer drafter '{dp}': {e}"))?;
                        if dhfq.arch_id
                            != hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_ARCH_ID
                        {
                            return Err(format!(
                                "glimmer drafter '{}' has arch_id {} but a Muse Glimmer drafter must be arch_id {} (muse_glimmer_assistant); DFlash drafts (arch 20) and Gemma 4 drafts (arch 22) do not work here",
                                dp, dhfq.arch_id, hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_ARCH_ID
                            ));
                        }
                        let dcfg =
                            hipfire_arch_muse_glimmer::drafter::GlimmerDrafterConfig::from_hfq(
                                &dhfq,
                            )?;
                        if dcfg.hidden != config.dim {
                            return Err(format!(
                                "glimmer drafter hidden size {} does not match the target model's {}; use the drafter made for this model",
                                dcfg.hidden, config.dim
                            ));
                        }
                        if dcfg.block_size != 16 {
                            eprintln!("glimmer drafter: WARNING block_size {} != 16 (expected diffusion recipe)", dcfg.block_size);
                        }
                        let dweights =
                            hipfire_arch_muse_glimmer::drafter::GlimmerDrafterWeights::load(
                                &dhfq, &dcfg, ctx.gpu,
                            )?;
                        // Freeze HIPFIRE_GLIMMER_CTX_CAP once at load (daemon/load default
                        // 256). Same value sizes drafter scratch and device hidden log.
                        let ctx_cap = {
                            let requested = hipfire_config::developer_var(
                                "HIPFIRE_GLIMMER_CTX_CAP",
                            )
                            .ok()
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .filter(|v| *v > 0)
                            .unwrap_or(
                                hipfire_arch_muse_glimmer::drafter::GLIMMER_DRAFTER_CTX_CAP_DEFAULT,
                            );
                            requested.clamp(1, ctx.max_seq)
                        };
                        let dscratch =
                            match hipfire_arch_muse_glimmer::drafter::GlimmerDrafterScratch::new(
                                ctx.gpu,
                                &dcfg,
                                ctx.max_seq,
                                ctx_cap,
                            ) {
                                Ok(s) => s,
                                Err(e) => {
                                    dweights.free_gpu(ctx.gpu);
                                    return Err(format!("glimmer drafter scratch: {e}"));
                                }
                            };
                        eprintln!(
                            "  glimmer DFlash drafter loaded: {} (layers={}, hidden={}, block={}, mask={}, ctx_cap={})",
                            dp, dcfg.n_layers, dcfg.hidden, dcfg.block_size, dcfg.mask_token_id, ctx_cap
                        );
                        Ok(crate::GlimmerDrafterBundle {
                            config: dcfg,
                            weights: dweights,
                            scratch: dscratch,
                        })
                    })() {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!("  glimmer DFlash drafter load failed ({}): {} — falling back to AR only", ctx.draft_path.as_deref().unwrap_or(""), e);
                            None
                        }
                    }
                } else {
                    None
                };
                // Device hidden capture is an opt-in experiment. The controlled
                // gfx1100 gate removed all capture D2Hs but regressed decode
                // throughput, so the unchanged host path remains the default.
                // Selection is frozen here — no runtime path flipping after load.
                if let Some(d) = &drafter {
                    let device_enabled =
                        hipfire_config::developer_var("HIPFIRE_GLIMMER_DEVICE_CAPTURE")
                            .ok()
                            .as_deref()
                            == Some("1");
                    if !device_enabled {
                        eprintln!(
                            "  glimmer hidden capture: backend=host (set HIPFIRE_GLIMMER_DEVICE_CAPTURE=1 to test device capture)"
                        );
                    } else {
                        // Same frozen cap used for scratch construction (no env re-read).
                        let ctx_cap = d.scratch.ctx_capacity();
                        let layers = &d.config.target_layer_ids;
                        match state.enable_device_hidden_capture(
                            ctx.gpu,
                            layers,
                            ctx_cap,
                            config.n_layers,
                            config.dim,
                        ) {
                            Ok(()) => {
                                eprintln!(
                                    "  glimmer hidden capture: backend=device capacity_rows={} target_layer_ids={:?} shape=[{}, {}, {}]",
                                    ctx_cap,
                                    layers,
                                    ctx_cap,
                                    layers.len(),
                                    config.dim,
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "  glimmer hidden capture: device enable failed ({e}) — continuing with host fallback"
                                );
                            }
                        }
                    }
                }
                let speculator = crate::spec_build::build_speculator(
                    meta.arch_id,
                    None,
                    None,
                    true,
                    ctx.max_seq,
                    ctx.spec,
                );
                let chat_template = match meta.chat_template {
                    Some(t) => match crate::rewrite_muse_glimmer_onyx_template(&t) {
                        Ok(rewritten) => Some(rewritten),
                        Err(e) => {
                            state.free_gpu(ctx.gpu);
                            weights.free_gpu(ctx.gpu);
                            if let Some(d) = drafter {
                                d.scratch.free_gpu(ctx.gpu);
                                d.weights.free_gpu(ctx.gpu);
                            }
                            return Err(e);
                        }
                    },
                    None => None,
                };
                let chat_template_source = meta
                    .chat_template_source
                    .map(|s| format!("{s} (muse-glimmer onyx rewrite)"));
                Ok(LoadedModel {
                    state: Some(Box::new(crate::MuseGlimmerBundle {
                        config,
                        weights,
                        state,
                        eos_tok,
                        drafter,
                        target_hidden_host: Vec::new(),
                    })),
                    speculator,
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        chat_template,
                        chat_template_source,
                    )
                })
            }
            ModelSource::Dir(source) => {
                let _ = source;
                return Err(
                    "muse_glimmer: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 14) or add config_from_source to hipfire-arch-muse-glimmer".into()
                );
            }
        }
    }
}

// ─── FluxDiffusionCarrier (arch 40 FLUX.1 / arch 45 FLUX.2 Klein) ──────
//
// Image-generation COMPONENT carrier: loads the HFQ component packs that
// `hipfire-quantize --flux-pipe` writes (the trunk plus its sidecars, found
// next to it by name) into a `FluxPipeModel`. A diffusers pipe directory is
// the packer's input, not a model — it is refused by name. The daemon must
// never text-generate on the result; `img_route` in lib.rs is the
// fail-closed gate in both directions.

pub struct FluxDiffusionCarrier;
impl Carrier for FluxDiffusionCarrier {
    fn name(&self) -> &'static str {
        "flux"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 40 || arch_id == 45
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        use hipfire_arch_diffusion::pipeline::HfqSidecars;

        let hfq = match &src {
            ModelSource::Hfq(hfq) => hfq,
            ModelSource::Dir(source) => {
                return Err(format!(
                    "flux (arch 40/45): {} is a diffusers pipe directory, which is not \
                     loadable; pack it with `hipfire-quantize --flux-pipe <pipe_dir> \
                     --output <base>.hfq` and load `<base>-transformer.hfq`",
                    source.path().display()
                ));
            }
        };
        // The trunk pack (arch 40 or 45) plus the sidecar packs the packer
        // wrote next to it: `<base>-t5.hfq` / `<base>-clip.hfq` / `<base>-vae.hfq`
        // for FLUX.1, `<base>-qwen3.hfq` / `<base>-vae.hfq` for FLUX.2 Klein,
        // when the trunk is `<base>-transformer.hfq`. The shared names the
        // registry sidecar slots land under (`t5-xxl.hfq` / `clip-l.hfq` /
        // `qwen3.hfq` / `vae.hfq`) are accepted too, so one T5 and one VAE
        // serve both schnell and dev.
        let arch_id = hfq.arch_id;
        let trunk_path = hfq.path().to_path_buf();
        let reopen = |p: &std::path::Path| -> Result<hipfire_runtime::hfq::HfqFile, String> {
            hipfire_runtime::hfq::HfqFile::open(p)
                .map_err(|e| format!("flux (HFQ): reopen {}: {e}", p.display()))
        };
        let file = trunk_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let stem = trunk_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(file);
        let base = stem.strip_suffix("-transformer").unwrap_or(stem);
        let find = |candidates: &[String]| -> Option<std::path::PathBuf> {
            candidates
                .iter()
                .map(|n| trunk_path.with_file_name(n))
                .find(|p| p.is_file())
        };
        let vae_names = [format!("{base}-vae.hfq"), "vae.hfq".into()];
        let Some(vae_path) = find(&vae_names) else {
            return Err(format!(
                "flux (HFQ): {file} needs the vae sidecar pack next to it \
                 (tried {base}-vae.hfq and the shared vae.hfq)"
            ));
        };
        let sidecars = match arch_id {
            40 => {
                let t5_names = [
                    format!("{base}-t5.hfq"),
                    "t5-xxl.hfq".into(),
                    "t5.hfq".into(),
                ];
                let clip_names = [
                    format!("{base}-clip.hfq"),
                    "clip-l.hfq".into(),
                    "clip.hfq".into(),
                ];
                let (Some(t5_path), Some(clip_path)) = (find(&t5_names), find(&clip_names)) else {
                    return Err(format!(
                        "flux (HFQ): {file} needs the t5 and clip sidecar packs next to it \
                         (tried {base}-t5.hfq, {base}-clip.hfq and the shared t5-xxl.hfq / \
                         clip-l.hfq)"
                    ));
                };
                HfqSidecars::Flux1 {
                    t5: reopen(&t5_path)?,
                    clip: reopen(&clip_path)?,
                    vae: reopen(&vae_path)?,
                }
            }
            45 => {
                let qwen3_names = [format!("{base}-qwen3.hfq"), "qwen3.hfq".into()];
                let Some(qwen3_path) = find(&qwen3_names) else {
                    return Err(format!(
                        "flux (HFQ): {file} needs the qwen3 sidecar pack next to it \
                         (tried {base}-qwen3.hfq and the shared qwen3.hfq)"
                    ));
                };
                HfqSidecars::Flux2 {
                    qwen3: reopen(&qwen3_path)?,
                    vae: reopen(&vae_path)?,
                }
            }
            other => {
                return Err(format!(
                "flux (HFQ): arch {other} is not a FLUX trunk pack (40 FLUX.1 / 45 FLUX.2 Klein)"
            ))
            }
        };
        // Reopen the trunk fresh: the loader's copy may have been prepared
        // (mmap dropped) for a UMA device, and the streaming load needs the
        // mapping alive for its whole run.
        let bundle =
            hipfire_arch_diffusion::pipeline::load_pipe_hfq(reopen(&trunk_path)?, sidecars)?;
        // Skeleton tokenizer for the `LoadedModel` contract. The img path
        // never encodes through it (the bundle's own tokenizers do the
        // conditioning; text `generate` refuses arch 40/45), so any parseable
        // vocab satisfies the field: the embedded CLIP `vocab.json` (FLUX.1)
        // or the embedded Qwen3 `tokenizer.json` (Klein), else a minimal
        // one-token vocab. No panic: fail closed as an error.
        let skeleton_tokenizer = {
            let meta: serde_json::Value =
                serde_json::from_str(hfq.metadata_json()).unwrap_or(serde_json::Value::Null);
            let tok = meta.get("tokenizer");
            tok.and_then(|t| t.get("clip_vocab"))
                .and_then(|vocab| {
                    let blob = serde_json::json!({ "model": { "vocab": vocab } });
                    hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&blob.to_string()).ok()
                })
                .or_else(|| {
                    tok.and_then(|t| t.get("qwen"))
                        .and_then(|q| q.as_str())
                        .and_then(|q| hipfire_runtime::tokenizer::Tokenizer::from_hf_json(q).ok())
                })
                .or_else(|| {
                    hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
                        r#"{"model":{"vocab":{"<pad>":0}}}"#,
                    )
                    .ok()
                })
                .ok_or("flux (HFQ): cannot build a skeleton tokenizer from the trunk metadata")?
        };
        Ok(LoadedModel {
            arch_id,
            state: Some(Box::new(
                hipfire_arch_diffusion::arch_model::FluxPipeModel { bundle },
            )),
            ..LoadedModel::skeleton(
                arch_id,
                skeleton_tokenizer,
                4096,
                4096,
                ctx.path.to_string(),
                None,
                None,
            )
        })
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        // Component only: no chat capabilities of any kind.
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            supports_images: false,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
}

// ─── qwen3.5 multi-vector embedding encoder (arch 17) ──────────────────────
//
// The non-chat wire verb precedent is img_generate: this carrier loads the
// model, but embedding requests flow through the daemon's `embed` verb →
// hipfire-generate's embed body, never through the generation routes.
pub struct Qwen35EmbedCarrier;
impl Carrier for Qwen35EmbedCarrier {
    fn name(&self) -> &'static str {
        "qwen35-embed"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 17
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: true,
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(crate::pp_unsupported_message(
                "qwen35-embed",
                matches!(src, ModelSource::Dir(_)),
                ctx.pp,
            ));
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_qwen35_embed::load_qwen35_embed_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator: None,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }
    /// The daemon's slot preflight reads the qwen35 HFQ layout; this arch's
    /// checkpoint is often a safetensors directory, and its geometry lives in
    /// `config.json` either way. Both facts are the arch crate's, so it reads
    /// them (see `qwen35_embed::carrier::preflight_slot`).
    fn preflight_slot_model(
        &self,
        path: &std::path::Path,
    ) -> Option<Result<hipfire_runtime::serve::SlotModelPreflight, String>> {
        match hipfire_arch_qwen35_embed::carrier::preflight_slot(path) {
            Ok(Some(p)) => Some(Ok(p)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
    fn spawn_slot_engine(
        &self,
        cfg: hipfire_runtime::serve::SlotEngineConfig,
    ) -> Result<Box<dyn hipfire_runtime::serve::SlotEngineHandle>, String> {
        use hipfire_arch_qwen35_embed::slot_engine::{EmbedEngineConfig, EmbedSlotEngine};
        let engine = EmbedSlotEngine::spawn(EmbedEngineConfig::from_slot_config(&cfg))
            .map_err(|e| format!("EmbedSlotEngine spawn: {e}"))?;
        Ok(Box::new(engine))
    }
}

// ─── EmbeddingGemma 2 single-vector embedding encoder (arch 18) ─────────────
//
// Same non-chat wire-verb shape as arch 17: loads the model, embedding
// requests flow through the daemon's `embed` verb → hipfire-generate's embed
// body. The checkpoint's text trunk is tower-less: the vision tower arrives
// as the shared `.vl` sidecar (admission resolves the path, the arch crate
// opens it at load), and a load without one refuses image inputs with a
// pointing error rather than silently embedding them.
pub struct EmbeddingGemma2Carrier;
impl Carrier for EmbeddingGemma2Carrier {
    fn name(&self) -> &'static str {
        "embeddinggemma2"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 18
    }
    fn caps(&self) -> saddle_core::caps::ArchCaps {
        saddle_core::caps::ArchCaps {
            supports_continuous_batch: false,
            supports_ep_batch: false,
            dflash: None,
            supports_mtp: false,
            spec_excludes_adaptive: false,
            semantic_contract_version: None,
            has_deltanet: false,
            // Architectural capability: the EmbeddingGemma-2 checkpoint ships
            // the Gemma4 vision tower in its own arch-18 `.vl` pack (probed
            // via `vision_tower.patch_embedder.input_proj.weight`, admission.rs
            // `EMBED_GEMMA_VISION_PROBE_TENSOR`); this flag declares that the
            // arch CAN accept images when that tower is present. Per-instance
            // gating still checks `EmbedGemmaBundle.vision.is_some()` — a
            // tower-less trunk (`vision_mode=off` / no sidecar) refuses images
            // with the pointing error.
            supports_images: true,
            reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
        }
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "embeddinggemma2: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "embeddinggemma2: safetensors + pp>1 unsupported",
            }
            .into());
        }
        dir_diag(&src);
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_embeddinggemma2::load_embed_gemma_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(Box::new(bundle)),
            speculator: None,
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
                meta.chat_template_source,
            )
        })
    }

    fn preflight_slot_model(
        &self,
        path: &std::path::Path,
    ) -> Option<Result<hipfire_runtime::serve::SlotModelPreflight, String>> {
        match hipfire_arch_embeddinggemma2::carrier::preflight_slot(path) {
            Ok(Some(p)) => Some(Ok(p)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// One weight copy, one GPU thread, no KV: the encoder's slot engine
    /// batches the requests that arrive together into a single forward.
    fn spawn_slot_engine(
        &self,
        cfg: hipfire_runtime::serve::SlotEngineConfig,
    ) -> Result<Box<dyn hipfire_runtime::serve::SlotEngineHandle>, String> {
        use hipfire_arch_embeddinggemma2::slot_engine::{EmbedEngineConfig, EmbedSlotEngine};
        let engine = EmbedSlotEngine::spawn(EmbedEngineConfig::from_slot_config(&cfg))
            .map_err(|e| format!("EmbedSlotEngine spawn: {e}"))?;
        Ok(Box::new(engine))
    }
}

#[cfg(test)]
mod qwen4_source_policy_tests {
    use super::qwen4_use_range_payload;
    use hipfire_runtime::weight_manifest::WeightResidency;

    #[test]
    fn discrete_resident_payload_stays_borrowed() {
        assert!(!qwen4_use_range_payload(false, WeightResidency::Resident));
    }

    #[test]
    fn uma_resident_payload_uses_ranges() {
        assert!(qwen4_use_range_payload(true, WeightResidency::Resident));
    }

    #[test]
    fn external_payload_uses_ranges_on_discrete() {
        assert!(qwen4_use_range_payload(
            false,
            WeightResidency::external_rows(128, 1)
        ));
    }
}

#[cfg(test)]
mod qwen4_admission_tests {
    use super::Qwen4Carrier;
    use crate::{admission::SourceAdmissionOptions, Carrier};
    use hipfire_runtime::loader_api::SpecLoadCfg;

    fn resolved_ar_spec() -> SpecLoadCfg {
        SpecLoadCfg {
            dflash: Some(false),
            dspark: Some(false),
            ngram_draft: Some(false),
            ddtree_budget: Some(0),
            ddtree_topk: Some(4),
            mtp: Some(false),
            ..SpecLoadCfg::default()
        }
    }

    #[test]
    fn schema_defaults_pass_carrier_gate_but_unsupported_options_refuse() {
        let defaults = SourceAdmissionOptions {
            spec: resolved_ar_spec(),
            ..Default::default()
        };
        assert!(Qwen4Carrier.admit_options(None, defaults).is_ok());
        for (option, options) in [
            (
                "adaptive KV",
                SourceAdmissionOptions {
                    kv_adaptive: true,
                    ..defaults
                },
            ),
            (
                "EAGLE",
                SourceAdmissionOptions {
                    eagle_drafter: true,
                    ..defaults
                },
            ),
            (
                "PFlash",
                SourceAdmissionOptions {
                    pflash: true,
                    ..defaults
                },
            ),
            (
                "expert-count override",
                SourceAdmissionOptions {
                    expert_count_override: true,
                    ..defaults
                },
            ),
            (
                "DeepSeek4 placement",
                SourceAdmissionOptions {
                    non_single_compute: true,
                    ..defaults
                },
            ),
        ] {
            let error = Qwen4Carrier.admit_options(None, options).expect_err(option);
            assert!(error.contains("unsupported"), "{option}: {error}");
        }
        assert!(Qwen4Carrier
            .admit_options(Some("draft.hfq"), defaults)
            .is_err());
    }

    #[test]
    fn pflash_alone_refusal_names_only_pflash() {
        let options = SourceAdmissionOptions {
            spec: resolved_ar_spec(),
            pflash: true,
            ..Default::default()
        };
        let error = Qwen4Carrier.admit_options(None, options).unwrap_err();
        assert_eq!(
            error,
            "qwen4: unsupported option(s): PFlash (prefill_compression; set it to off); turn them off to load Flash-Next"
        );
        for other in [
            "draft model",
            "EAGLE",
            "adaptive KV",
            "DFlash",
            "DSpark",
            "n-gram",
            "DDTree",
            "CASK",
            "non-single",
            "expert-count",
        ] {
            assert!(!error.contains(other), "{other}: {error}");
        }
    }

    #[test]
    fn expert_parallel_refuses_speculation_and_single_device_options_naming_ep() {
        use super::qwen4_ep_option_refusal;
        let defaults = SourceAdmissionOptions {
            spec: resolved_ar_spec(),
            ..Default::default()
        };
        // ep == 1 never refuses here (the single-device gate is `admit_options`).
        assert!(qwen4_ep_option_refusal(1, None, &defaults).is_ok());
        assert!(qwen4_ep_option_refusal(
            1,
            Some("draft.hfq"),
            &SourceAdmissionOptions {
                pflash: true,
                ..defaults
            }
        )
        .is_ok());
        let with_spec = |edit: &dyn Fn(&mut SpecLoadCfg)| {
            let mut spec = resolved_ar_spec();
            edit(&mut spec);
            SourceAdmissionOptions { spec, ..defaults }
        };
        let cases: Vec<(&str, Option<&str>, SourceAdmissionOptions)> = vec![
            ("DFlash", None, with_spec(&|s| s.dflash = Some(true))),
            ("DFlash", Some("draft.hfq"), defaults),
            ("DSpark", None, with_spec(&|s| s.dspark = Some(true))),
            ("n-gram", None, with_spec(&|s| s.ngram_draft = Some(true))),
            ("DDTree", None, with_spec(&|s| s.ddtree_budget = Some(8))),
            (
                "adaptive KV",
                None,
                SourceAdmissionOptions {
                    kv_adaptive: true,
                    ..defaults
                },
            ),
            (
                "EAGLE",
                None,
                SourceAdmissionOptions {
                    eagle_drafter: true,
                    ..defaults
                },
            ),
            (
                "CASK",
                None,
                SourceAdmissionOptions {
                    cask: true,
                    ..defaults
                },
            ),
            (
                "PFlash",
                None,
                SourceAdmissionOptions {
                    pflash: true,
                    ..defaults
                },
            ),
            (
                "experts-per-token",
                None,
                SourceAdmissionOptions {
                    expert_count_override: true,
                    ..defaults
                },
            ),
            (
                "compute placement",
                None,
                SourceAdmissionOptions {
                    non_single_compute: true,
                    ..defaults
                },
            ),
        ];
        for ep in [2usize, 3, 4, 6, 8] {
            assert!(
                qwen4_ep_option_refusal(ep, None, &defaults).is_ok(),
                "ep {ep} plain AR"
            );
            // Native MTP is admitted: the drafter is replicated on every rank.
            assert!(
                qwen4_ep_option_refusal(ep, None, &with_spec(&|s| s.mtp = Some(true))).is_ok(),
                "ep {ep} native MTP"
            );
            for (what, draft, options) in &cases {
                let error = qwen4_ep_option_refusal(ep, *draft, options).expect_err(what);
                assert!(
                    error.contains("expert parallelism")
                        && error.contains(&format!("topology.ep = {ep}")),
                    "{what}: {error}"
                );
                assert!(
                    error.to_lowercase().contains(&what.to_lowercase()),
                    "{what}: {error}"
                );
            }
        }
    }

    #[test]
    fn emulated_ranks_split_their_physical_device_free_vram() {
        use super::{qwen4_ep_alias_free, qwen4_ep_emulation_map};
        use crate::admission::qwen4_ep_rank_devices;
        let gib = 1u64 << 30;
        // Physical mode: every device listed once, free untouched.
        assert_eq!(
            qwen4_ep_alias_free(&[0, 1, 2], &[24 * gib, 20 * gib, 16 * gib]),
            vec![24 * gib, 20 * gib, 16 * gib]
        );
        // One physical device under 3 and 6 ranks.
        assert_eq!(
            qwen4_ep_alias_free(&[0, 0, 0], &[24 * gib; 3]),
            vec![8 * gib; 3]
        );
        assert_eq!(
            qwen4_ep_alias_free(&[0; 6], &[24 * gib; 6]),
            vec![4 * gib; 6]
        );
        // Four physical devices under 6 ranks: devices 0 and 1 carry two ranks.
        let devices = qwen4_ep_rank_devices(6, 4, true).unwrap();
        let free = [24 * gib, 20 * gib, 16 * gib, 12 * gib, 24 * gib, 20 * gib];
        assert_eq!(
            qwen4_ep_alias_free(&devices, &free),
            vec![12 * gib, 10 * gib, 16 * gib, 12 * gib, 12 * gib, 10 * gib]
        );
        // Four physical devices under 8 ranks: every device carries two.
        let devices = qwen4_ep_rank_devices(8, 4, true).unwrap();
        assert_eq!(
            qwen4_ep_alias_free(&devices, &[24 * gib; 8]),
            vec![12 * gib; 8]
        );
        assert_eq!(qwen4_ep_emulation_map(&[0, 1, 0]), "0->0, 1->1, 2->0");
    }
}

#[cfg(test)]
mod qwen4_ple_load_fill_tests {
    use super::{
        ple_load_fill_bytes, ple_prewarm_budget_for, ple_resident_bytes, PLE_LOAD_FILL_AUTO_BYTES,
        PLE_PREWARM_RESERVE_BYTES,
    };
    use hipfire_config::memory::PleHotStoreBudget::{Auto, Bytes};

    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_reads_b_star_when_the_host_has_room() {
        // Halo at load start: ~112 GiB available, ~66 GiB of weights to load.
        assert_eq!(
            ple_load_fill_bytes(Auto, Auto, Some(112 * GIB), 66 * GIB, 27 * GIB),
            PLE_LOAD_FILL_AUTO_BYTES
        );
    }

    #[test]
    fn caps_apply_in_every_mode() {
        // The memory left after the weights and the reserve.
        assert_eq!(
            ple_load_fill_bytes(Auto, Auto, Some(80 * GIB), 66 * GIB, 27 * GIB),
            80 * GIB - 66 * GIB - PLE_PREWARM_RESERVE_BYTES
        );
        assert_eq!(
            ple_load_fill_bytes(Bytes(20 * GIB), Auto, Some(60 * GIB), 66 * GIB, 27 * GIB),
            0
        );
        assert_eq!(ple_load_fill_bytes(Auto, Auto, None, 0, 27 * GIB), 0);
        // An explicit prewarm budget, and the table.
        assert_eq!(
            ple_load_fill_bytes(Auto, Bytes(2 * GIB), Some(112 * GIB), 66 * GIB, 27 * GIB),
            2 * GIB
        );
        assert_eq!(
            ple_load_fill_bytes(Auto, Bytes(0), Some(112 * GIB), 66 * GIB, 27 * GIB),
            0
        );
        assert_eq!(
            ple_load_fill_bytes(Bytes(u64::MAX), Auto, Some(u64::MAX), 0, 27 * GIB),
            27 * GIB
        );
        assert_eq!(
            ple_load_fill_bytes(Bytes(0), Auto, Some(112 * GIB), 66 * GIB, 27 * GIB),
            0
        );
    }

    #[test]
    fn halo_prewarm_budget_and_resident_charge() {
        const MIB: u64 = 1 << 20;
        // Halo: 23,038 MiB MemAvailable at admission leaves 23,038 - 8,192 =
        // 14,846 MiB, under the 15 GiB auto cap. The 14 GiB in-load fill is
        // inside it, so the resident charge is the budget.
        let (budget, zero) = ple_prewarm_budget_for(Auto, Some(23_038 * MIB));
        assert_eq!((budget, zero), (14_846 * MIB, false));
        assert_eq!(
            ple_resident_bytes(budget, PLE_LOAD_FILL_AUTO_BYTES, 27 * GIB),
            14_846 * MIB
        );
        // The auto cap, an explicit value, zero and unknown MemAvailable.
        assert_eq!(
            ple_prewarm_budget_for(Auto, Some(112 * GIB)),
            (15 * GIB, false)
        );
        assert_eq!(
            ple_prewarm_budget_for(Bytes(2 * GIB), None),
            (2 * GIB, false)
        );
        assert_eq!(ple_prewarm_budget_for(Bytes(0), Some(112 * GIB)), (0, true));
        assert_eq!(ple_prewarm_budget_for(Auto, None), (0, false));
    }

    #[test]
    fn resident_charge_covers_the_load_fill_and_is_capped_at_the_table() {
        // The fill's target counts when it exceeds a later, smaller budget.
        assert_eq!(ple_resident_bytes(GIB, 14 * GIB, 27 * GIB), 14 * GIB);
        assert_eq!(ple_resident_bytes(15 * GIB, 14 * GIB, 27 * GIB), 15 * GIB);
        // Never more than the table/pack/sidecar holds, and 0 when off.
        assert_eq!(ple_resident_bytes(40 * GIB, 14 * GIB, 27 * GIB), 27 * GIB);
        assert_eq!(ple_resident_bytes(0, 0, 27 * GIB), 0);
    }
}

#[cfg(test)]
mod qwen4_pp_load_tests {
    use super::qwen4_pp_mesh_refusal;
    use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
    use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank, BandPlacement};

    /// The admitted cell for `pp` ranks in output-card-last order: logical
    /// `s < pp - 1` is visible device `s + 1`, the output rank is device 0.
    fn cell(pp: usize, arch: &str) -> (AdmittedMesh, BandPlacement) {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, pp)]).unwrap();
        let placement = BandPlacement::uniform(&mesh, 48).unwrap();
        let ranks = (0..pp)
            .map(|logical| AdmittedRank {
                logical,
                device_ordinal: ((logical + 1) % pp) as i32,
                pci_bus_id: Some(format!("0000:0{logical}:00.0")),
                arch: arch.to_string(),
            })
            .collect();
        let admitted = AdmittedMesh {
            mesh,
            ranks,
            edges: Vec::new(),
            family: "qwen4".to_string(),
            numerical_policy: "test".to_string(),
        };
        (admitted, placement)
    }

    #[test]
    fn accepted_cells_are_homogeneous_gfx1100_or_gfx1201_with_the_output_card_as_daemon() {
        for arch in ["gfx1100", "gfx1201"] {
            for pp in [2, 3, 6, 8] {
                let (admitted, placement) = cell(pp, arch);
                assert_eq!(
                    qwen4_pp_mesh_refusal(&admitted, &placement, pp, 0, arch),
                    Ok(())
                );
            }
        }
    }

    #[test]
    fn every_refusal_names_topology_pp_before_any_device_is_touched() {
        let refuses = |admitted: &AdmittedMesh, placement: &BandPlacement, pp, device, arch| {
            let error = qwen4_pp_mesh_refusal(admitted, placement, pp, device, arch)
                .expect_err("this cell must refuse");
            assert!(error.contains("topology.pp"), "{error}");
        };
        let (admitted, placement) = cell(2, "gfx1100");
        // The daemon's generation device must be the output rank.
        refuses(&admitted, &placement, 2, 1, "gfx1100");
        // Mixed or unsupported architectures.
        refuses(&admitted, &placement, 2, 0, "gfx1201");
        let (other, other_placement) = cell(2, "gfx90a");
        refuses(&other, &other_placement, 2, 0, "gfx90a");
        let (mut mixed, mixed_placement) = cell(3, "gfx1100");
        mixed.ranks[1].arch = "gfx1201".to_string();
        refuses(&mixed, &mixed_placement, 3, 0, "gfx1100");
        // Unknown PCI identity.
        let (mut anonymous, anonymous_placement) = cell(2, "gfx1100");
        anonymous.ranks[0].pci_bus_id = None;
        refuses(&anonymous, &anonymous_placement, 2, 0, "gfx1100");
        // Degrees outside 2..=8 and a placement of another degree.
        refuses(&admitted, &placement, 1, 0, "gfx1100");
        let (nine, nine_placement) = cell(9, "gfx1100");
        refuses(&nine, &nine_placement, 9, 0, "gfx1100");
        let (_, three) = cell(3, "gfx1100");
        refuses(&admitted, &three, 2, 0, "gfx1100");
        // Logical ids must follow the rank order.
        let (mut shuffled, shuffled_placement) = cell(2, "gfx1100");
        shuffled.ranks[0].logical = 1;
        refuses(&shuffled, &shuffled_placement, 2, 0, "gfx1100");
    }
}

#[cfg(test)]
mod qwen4_pp_session_tests {
    use super::qwen4_session_disk_tier;
    use std::path::PathBuf;

    #[test]
    fn a_pp_band_never_attaches_the_disk_tier() {
        assert_eq!(
            qwen4_session_disk_tier(true, PathBuf::from("/cache"), 1 << 30),
            None
        );
        assert_eq!(
            qwen4_session_disk_tier(true, PathBuf::from("/cache"), 0),
            None
        );
    }

    #[test]
    fn a_single_or_ep_rank_keeps_its_disk_tier() {
        assert_eq!(
            qwen4_session_disk_tier(false, PathBuf::from("/cache"), 1 << 30),
            Some((PathBuf::from("/cache"), 1 << 30))
        );
        assert_eq!(
            qwen4_session_disk_tier(false, PathBuf::from("/cache"), 0),
            Some((PathBuf::from("/cache"), 0))
        );
    }
}

#[cfg(test)]
mod qwen4_pp_native_mtp_tests {
    use super::{qwen4_pp_session_budget, qwen4_rank_native_mtp};
    use hipfire_arch_qwen4::weights::Qwen4BandScope;

    fn band(band: std::ops::Range<usize>, input: bool, output: bool) -> Qwen4BandScope {
        Qwen4BandScope {
            band,
            input,
            output,
        }
    }

    #[test]
    fn only_the_output_band_attaches_the_head() {
        let (first, middle, last) = (
            band(0..16, true, false),
            band(16..32, false, false),
            band(32..48, false, true),
        );
        // Single and expert-parallel ranks (no band) follow the flag.
        assert!(qwen4_rank_native_mtp(true, None));
        assert!(!qwen4_rank_native_mtp(false, None));
        for scope in [&first, &middle] {
            assert!(!qwen4_rank_native_mtp(true, Some(scope)));
            assert!(!qwen4_rank_native_mtp(false, Some(scope)));
        }
        assert!(qwen4_rank_native_mtp(true, Some(&last)));
        // An AR load attaches no head anywhere.
        assert!(!qwen4_rank_native_mtp(false, Some(&last)));
        // A whole-model band (the Single scope) is its own output band.
        assert!(qwen4_rank_native_mtp(true, Some(&band(0..48, true, true))));
    }

    #[test]
    fn native_mtp_pp_attaches_no_session_cache_and_ar_keeps_its_budget() {
        for configured in [0u64, 1 << 30, 8 << 30] {
            assert_eq!(qwen4_pp_session_budget(false, configured), configured);
            assert_eq!(qwen4_pp_session_budget(true, configured), 0);
        }
    }
}

#[cfg(test)]
mod dir_template_provenance_tests {
    use super::dir_template_description;

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-dir-template-provenance-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn non_blank_jinja_file_is_named_with_its_exact_path() {
        let dir = scratch_dir("jinja");
        std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
        let described = dir_template_description(dir.to_str().unwrap());
        assert!(described.starts_with(crate::SRC_DIR_JINJA));
        assert!(described.ends_with(dir.join("chat_template.jinja").to_str().unwrap()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn blank_or_absent_jinja_falls_to_tokenizer_config_path() {
        let dir = scratch_dir("blank");
        std::fs::write(dir.join("chat_template.jinja"), "  \n").unwrap();
        let described = dir_template_description(dir.to_str().unwrap());
        assert!(described.starts_with(crate::SRC_DIR_TOKENIZER_CONFIG));
        assert!(described.ends_with(dir.join("tokenizer_config.json").to_str().unwrap()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
